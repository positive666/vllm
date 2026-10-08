"""Run a separate expanded-budget TP2 quality diagnostic without changing primaries."""

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

from tp2_supervisor import SOURCE_HEAD, gpu_inventory, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    parser.add_argument(
        "--results-root", type=Path, default=Path("/results/tp2-followup-20261008")
    )
    args = parser.parse_args()
    root = args.results_root
    folder = root / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"serve-{args.backend}.log"
    evidence_path = folder / f"serve-{args.backend}.json"
    output = folder / f"{args.backend}.json"
    if any(path.exists() for path in (log_path, evidence_path, output)):
        raise FileExistsError("Refusing to overwrite a diagnostic launch")
    source_manifest = Path("/artifacts/performance-source.json")
    manifest = json.loads(source_manifest.read_text())
    if manifest["source_head"] != SOURCE_HEAD:
        raise RuntimeError("Diagnostic source head differs")
    source_files = {
        name: hashlib.sha256((Path("/source") / name).read_bytes()).hexdigest()
        for name in manifest["files"]
    }
    if source_files != {
        name: value["sha256"] for name, value in manifest["files"].items()
    }:
        raise RuntimeError("Diagnostic source file hashes differ")
    inventory = gpu_inventory()
    cache = Path(f"/cache/tp2-diagnostic-{args.backend}")
    if cache.exists():
        raise FileExistsError(cache)
    cache.mkdir()
    seed = Path("/cache") / root.name / f"{args.backend}-compat"
    for kind in ("vllm", "triton", "cuda", "flashinfer"):
        if (seed / kind).exists():
            shutil.copytree(seed / kind, cache / kind)
    env = os.environ.copy()
    env.update(
        PYTHONPATH="/source",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
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
        "4096",
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
        json.dumps({"gdn_decode_backend": args.backend, "linear_backend": "marlin"}),
        "--num-gpu-blocks-override",
        "128",
    ]
    helper = Path("/artifacts/tp2_quality_diagnostic.py")
    record = {
        "backend": args.backend,
        "phase": "expanded-budget-diagnostic",
        "command": command,
        "source_head": SOURCE_HEAD,
        "source_files": source_files,
        "source_manifest_sha256": hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest(),
        "tensor_parallel_size": 2,
        "per_rank_gdn_shape": {"H": 8, "HV": 24, "K": 128, "V": 128},
        "gpu_blocks": 128,
        "max_model_len": 4096,
        "state_dtype": "float32",
        "gpu_inventory": inventory,
        "gpu_uuids": inventory["uuids"],
        "model_config_sha256": hashlib.sha256(
            Path("/model/config.json").read_bytes()
        ).hexdigest(),
        "runtime": json.loads((root / "runtime-probe.json").read_text()),
        "cache_root": str(cache),
        "cache_seed": str(seed),
        "cache_policy": (
            "Independent diagnostic cache copied from matching TP2 backend cache; "
            "framework specialization keys handle the new context length; "
            "no primary cache writes"
        ),
        "diagnostic_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Matched C1/C8 diagnostic changes model length, token budget and cache "
            "capacity together; no single-variable causal conclusion"
        ),
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
                    raise RuntimeError(f"Diagnostic server exited: {server.returncode}")
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
                raise TimeoutError("Diagnostic startup exceeded 1200 seconds")
            logged = re.findall(
                r"GPU KV cache size: ([\d,]+) tokens",
                log_path.read_text(errors="replace"),
            )
            capacities = {int(value.replace(",", "")) for value in logged}
            if len(capacities) != 1 or min(capacities) < 8 * 4096:
                raise RuntimeError({"diagnostic_capacity_insufficient": logged})
            record["kv_capacity_tokens"] = capacities.pop()
            record["logged_kv_capacities"] = logged
            record["status"] = "ready"
            record["ready_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            client = [
                "uv",
                "run",
                "--offline",
                "--no-project",
                sys.executable,
                str(helper),
                "--arm",
                args.backend,
                "--output",
                str(output),
                "--dataset",
                "/oldcache/gsm8k-test.jsonl",
                "--primary-quality",
                str(root / f"quality-{args.backend}.json"),
                "--max-tokens",
                "3500",
                "--server-evidence",
                str(evidence_path),
            ]
            record["client_command"] = client
            write_json(evidence_path, record)
            subprocess.run(client, check=True, env=env, timeout=3600)
            record["status"] = "completed"
            record["diagnostic_sha256"] = hashlib.sha256(
                output.read_bytes()
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
