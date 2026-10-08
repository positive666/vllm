"""Measure a frozen serving workload in independently identified processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2"))
    args = parser.parse_args()
    backend = args.arm.split("-")[0]
    root = Path("/results/performance-followup-20261008")
    root.mkdir(parents=True, exist_ok=True)
    (root / "http").mkdir(exist_ok=True)
    evidence_path = root / f"serve-{args.arm}.json"
    log_path = root / f"serve-{args.arm}.log"
    if evidence_path.exists() or log_path.exists():
        raise FileExistsError("Refusing to overwrite a serving launch")
    manifest = json.loads(Path("/artifacts/performance-source.json").read_text())
    source_files = {name: hashlib.sha256((Path("/source") / name).read_bytes()).hexdigest()
                    for name in manifest["files"]}
    assert all(source_files[name] == value["sha256"]
               for name, value in manifest["files"].items())
    env = os.environ.copy()
    cache_root = Path(f"/cache/performance-{args.arm}")
    seeds = Path(f"/cache/quality-{backend}-c8-r1")
    assert not cache_root.exists(), cache_root
    cache_root.mkdir()
    for cache in ("vllm", "triton", "cuda"):
        if (seeds / cache).exists():
            shutil.copytree(seeds / cache, cache_root / cache)
    env.update(PYTHONPATH="/source", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_CACHE_ROOT=str(cache_root / "vllm"),
               TRITON_CACHE_DIR=str(cache_root / "triton"),
               CUDA_CACHE_PATH=str(cache_root / "cuda"))
    command = [
        "uv", "run", "--offline", "--no-project", sys.executable, "-m",
        "vllm.entrypoints.openai.api_server", "--model", "/model",
        "--served-model-name", "qwen-fp8", "--host", "127.0.0.1", "--port", "8000",
        "--language-model-only", "--dtype", "bfloat16", "--tensor-parallel-size", "1",
        "--max-model-len", "2048", "--max-num-seqs", "32",
        "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.9",
        "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", "float32",
        "--seed", "42", "--attention-config",
        json.dumps({"backend": "FLASH_ATTN", "flash_attn_version": 2}),
        "--kernel-config", json.dumps({"gdn_decode_backend": backend, "linear_backend": "marlin"}),
        "--num-gpu-blocks-override", "64",
    ]
    record = {
        "arm": args.arm, "backend": backend, "command": command,
        "source_head": manifest["source_head"], "source_files": source_files,
        "state_dtype": "float32",
        "model_config_sha256": hashlib.sha256(Path("/model/config.json").read_bytes()).hexdigest(),
        "runtime": json.loads((root / "runtime-probe.json").read_text()),
        "runtime_provenance": "Fresh probe before this performance pipeline, not inside each model process",
        "cache_policy": "Separate per-launch caches seeded from the corresponding same-head quality cache; startup excluded",
        "cache_seed": str(seeds), "cache_root": str(cache_root),
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "starting",
    }
    write_json(evidence_path, record)
    with log_path.open("x") as handle:
        server = subprocess.Popen(command, cwd="/source", env=env,
                                  stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record["server_pid"] = server.pid
        write_json(evidence_path, record)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {server.returncode}")
                try:
                    with opener.open("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(3)
            else:
                raise TimeoutError("Startup exceeded 900 seconds")
            logged_capacity = re.findall(r"GPU KV cache size: ([\d,]+) tokens",
                                         log_path.read_text(errors="replace"))
            assert logged_capacity and set(logged_capacity) == {"21,845"}, logged_capacity
            record["kv_capacity_tokens"] = 21845
            record["status"] = "ready"
            record["ready_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json(evidence_path, record)
            client = ["uv", "run", "--offline", "--no-project", sys.executable,
                      "/artifacts/http_performance_client.py"]
            fixture = root / "http-fixture.json"
            if not fixture.exists():
                subprocess.run(client + ["fixture", "--fixture", str(fixture)],
                               check=True, env=env, timeout=60)
            measured = client + [
                "bench", "--fixture", str(fixture), "--arm", args.arm,
                "--output-dir", str(root / "http"), "--server-evidence", str(evidence_path),
                "--requests", "16", "--warmup-requests", "8",
            ]
            record["client_command"] = measured
            write_json(evidence_path, record)
            subprocess.run(measured, check=True, env=env, timeout=2400)
            record["status"] = "completed"
            record["client_result_sha256"] = hashlib.sha256(
                (root / "http" / f"{args.arm}.json").read_bytes()).hexdigest()
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            record["server_exit"] = server.returncode
            record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            record["server_log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
            write_json(evidence_path, record)


if __name__ == "__main__":
    main()
