"""Run TP2 compatibility probes or a frozen independent serving launch."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
MEASURED_ARMS = ("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def gpu_inventory():
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError("TP2 requires exactly two authorized CUDA devices")
    devices = [
        {
            "cuda_index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "properties": str(torch.cuda.get_device_properties(index)),
        }
        for index in range(2)
    ]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    uuids = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    if len(uuids) != 2 or len(set(uuids)) != 2:
        raise RuntimeError({"unexpected_visible_gpu_uuids": uuids})
    return {"devices": devices, "uuids": uuids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "measure"), required=True)
    parser.add_argument(
        "--arm",
        choices=(*MEASURED_ARMS, "triton-compat", "flashinfer-compat"),
        required=True,
    )
    parser.add_argument("--gpu-blocks", type=int)
    parser.add_argument(
        "--results-root", type=Path, default=Path("/results/tp2-followup-20261008")
    )
    args = parser.parse_args()
    if (args.arm in MEASURED_ARMS) != (args.phase == "measure"):
        parser.error("Use compatibility arms for preflight and ABBA arms for measure")
    root = args.results_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "http").mkdir(exist_ok=True)
    evidence_path = root / f"serve-{args.arm}.json"
    log_path = root / f"serve-{args.arm}.log"
    if evidence_path.exists() or log_path.exists():
        raise FileExistsError("Refusing to overwrite a TP2 launch")
    manifest = json.loads(Path("/artifacts/performance-source.json").read_text())
    if manifest["source_head"] != SOURCE_HEAD:
        raise RuntimeError("Source manifest does not match the reviewed PR head")
    source_files = {
        name: hashlib.sha256((Path("/source") / name).read_bytes()).hexdigest()
        for name in manifest["files"]
    }
    if not all(
        source_files[name] == value["sha256"]
        for name, value in manifest["files"].items()
    ):
        raise RuntimeError("Source manifest hashes do not match /source")
    inventory = gpu_inventory()
    protocol = None
    if args.phase == "measure":
        protocol = json.loads((root / "protocol.json").read_text())
        blocks = protocol["gpu_blocks"]
        if (
            protocol["source_head"] != SOURCE_HEAD
            or protocol["tensor_parallel_size"] != 2
            or protocol["gpu_uuids"] != inventory["uuids"]
            or (args.gpu_blocks is not None and args.gpu_blocks != blocks)
        ):
            raise RuntimeError("Measured launch differs from frozen TP2 protocol")
    else:
        blocks = args.gpu_blocks
        if blocks is None or blocks < 16:
            parser.error("Preflight requires --gpu-blocks >=16; probe 64 initially")
    backend = args.arm.split("-")[0]
    cache_base = Path("/cache") / root.name
    cache_root = cache_base / args.arm
    seeds = cache_base / f"{backend}-compat"
    if cache_root.exists():
        raise FileExistsError(cache_root)
    cache_root.mkdir(parents=True)
    if args.phase == "measure":
        for cache in ("vllm", "triton", "cuda", "flashinfer"):
            if (seeds / cache).exists():
                shutil.copytree(seeds / cache, cache_root / cache)
    env = os.environ.copy()
    env.update(
        PYTHONPATH="/source",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_CACHE_ROOT=str(cache_root / "vllm"),
        TRITON_CACHE_DIR=str(cache_root / "triton"),
        CUDA_CACHE_PATH=str(cache_root / "cuda"),
        FLASHINFER_WORKSPACE_BASE=str(cache_root / "flashinfer"),
        NCCL_DEBUG="INFO",
    )
    command = [
        "uv",
        "run",
        "--offline",
        "--no-project",
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/model",
        "--served-model-name",
        "qwen-fp8",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--language-model-only",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "1024",
        "--gpu-memory-utilization",
        "0.9",
        "--no-enable-prefix-caching",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--seed",
        "42",
        "--attention-config",
        json.dumps({"backend": "FLASH_ATTN", "flash_attn_version": 2}),
        "--kernel-config",
        json.dumps({"gdn_decode_backend": backend, "linear_backend": "marlin"}),
        "--num-gpu-blocks-override",
        str(blocks),
    ]
    record = {
        "arm": args.arm,
        "backend": backend,
        "phase": args.phase,
        "command": command,
        "source_head": SOURCE_HEAD,
        "source_files": source_files,
        "tensor_parallel_size": 2,
        "per_rank_gdn_shape": {"H": 8, "HV": 24, "K": 128, "V": 128},
        "gpu_blocks": blocks,
        "gpu_inventory": inventory,
        "gpu_uuids": inventory["uuids"],
        "state_dtype": "float32",
        "model_config_sha256": hashlib.sha256(
            Path("/model/config.json").read_bytes()
        ).hexdigest(),
        "runtime": json.loads((root / "runtime-probe.json").read_text()),
        "runtime_provenance": (
            "Fresh dual-device probe before TP2 preflight, "
            "not inside each model process"
        ),
        "cache_policy": (
            "Separate TP2 caches; measured caches seeded only from corresponding "
            "TP2 compatibility launch; startup excluded"
        ),
        "cache_seed": str(seeds) if args.phase == "measure" else None,
        "cache_root": str(cache_root),
        "protocol_sha256": hashlib.sha256(
            (root / "protocol.json").read_bytes()
        ).hexdigest()
        if protocol is not None
        else None,
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "starting",
    }
    write_json(evidence_path, record)
    with log_path.open("x", encoding="utf-8") as handle:
        server = subprocess.Popen(
            command,
            cwd="/source",
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["server_pid"] = server.pid
        write_json(evidence_path, record)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited during startup: {server.returncode}"
                    )
                try:
                    with opener.open(
                        "http://127.0.0.1:8000/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(3)
            else:
                raise TimeoutError("Startup exceeded 1200 seconds")
            capacity_logs = re.findall(
                r"GPU KV cache size: ([\d,]+) tokens",
                log_path.read_text(errors="replace"),
            )
            capacities = {int(value.replace(",", "")) for value in capacity_logs}
            if len(capacities) != 1 or min(capacities) < 8 * (512 + 128):
                raise RuntimeError(
                    {"insufficient_or_inconsistent_capacity": capacity_logs}
                )
            capacity = capacities.pop()
            if protocol and capacity != protocol["expected_kv_capacity_tokens"]:
                raise RuntimeError({"kv_capacity_differs_from_protocol": capacity})
            record["kv_capacity_tokens"] = capacity
            record["logged_kv_capacities"] = capacity_logs
            record["status"] = "ready"
            record["ready_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json(evidence_path, record)
            client = [
                "uv",
                "run",
                "--offline",
                "--no-project",
                sys.executable,
                "/artifacts/http_performance_client.py",
            ]
            fixture = root / "http-fixture.json"
            if not fixture.exists():
                subprocess.run(
                    client + ["fixture", "--fixture", str(fixture)],
                    check=True,
                    env=env,
                    timeout=60,
                )
            if args.phase == "measure":
                client_command = client + [
                    "bench",
                    "--fixture",
                    str(fixture),
                    "--arm",
                    args.arm,
                    "--output-dir",
                    str(root / "http"),
                    "--server-evidence",
                    str(evidence_path),
                    "--requests",
                    "16",
                    "--warmup-requests",
                    "8",
                ]
                result = root / "http" / f"{args.arm}.json"
            else:
                client_command = [
                    "uv",
                    "run",
                    "--offline",
                    "--no-project",
                    sys.executable,
                    "/artifacts/http_tp2_compatibility.py",
                    "--backend",
                    backend,
                    "--fixture",
                    str(fixture),
                    "--output-dir",
                    str(root / "http"),
                    "--server-evidence",
                    str(evidence_path),
                ]
                result = root / "http" / f"compat-{backend}.json"
            record["client_command"] = client_command
            write_json(evidence_path, record)
            subprocess.run(client_command, check=True, env=env, timeout=2400)
            if args.phase == "preflight":
                quality_path = root / f"quality-{backend}.json"
                quality_command = [
                    "uv",
                    "run",
                    "--offline",
                    "--no-project",
                    sys.executable,
                    "/reference-artifacts/quality_http.py",
                    "--arm",
                    backend,
                    "--output",
                    str(quality_path),
                    "--dataset",
                    "/oldcache/gsm8k-test.jsonl",
                    "--samples",
                    "100",
                    "--concurrency",
                    "8",
                    "--max-tokens",
                    "1750",
                    "--timeout",
                    "300",
                    "--server-evidence",
                    str(evidence_path),
                ]
                record["quality_command"] = quality_command
                record["quality_helper_sha256"] = hashlib.sha256(
                    Path("/reference-artifacts/quality_http.py").read_bytes()
                ).hexdigest()
                write_json(evidence_path, record)
                subprocess.run(quality_command, check=True, env=env, timeout=3600)
                record["quality_sha256"] = hashlib.sha256(
                    quality_path.read_bytes()
                ).hexdigest()
            record["status"] = "completed"
            record["client_result_sha256"] = hashlib.sha256(
                result.read_bytes()
            ).hexdigest()
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            record["server_exit"] = server.returncode
            record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            record["server_log_sha256"] = hashlib.sha256(
                log_path.read_bytes()
            ).hexdigest()
            write_json(evidence_path, record)


if __name__ == "__main__":
    main()
