"""Bounded TP2 actual-decode capture on an isolated eager diagnostic server."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.util
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
from types import SimpleNamespace

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    args = parser.parse_args()
    root = Path("/results") / args.run
    root.mkdir(exist_ok=False)
    shutil.copytree("/shadow-helpers", root / "helpers", ignore=shutil.ignore_patterns("__pycache__"))
    manifest_path = Path("/artifacts/performance-source.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_head"] == HEAD
    actual = {name: sha(Path("/source") / name) for name in manifest["files"]}
    assert actual == {k: v["sha256"] for k, v in manifest["files"].items()}
    diagnostic = load("frozen_diagnostic", Path("/artifacts/tp2_quality_diagnostic.py"))
    frozen = diagnostic.load_harness(Path("/reference-artifacts/quality_http.py"))
    client_args = SimpleNamespace(
        arm=args.backend,
        dataset=Path("/oldcache/gsm8k-test.jsonl"),
        primary_quality=Path("/oldresults/tp2-followup-20261008")
        / f"quality-{args.backend}.json",
        harness=Path("/reference-artifacts/quality_http.py"),
        model="qwen-fp8", base_url="http://127.0.0.1:8000", max_tokens=3500,
        timeout=900,
    )
    _, selected, fixture = diagnostic.plan(client_args, frozen)
    cache = Path("/cache") / args.run
    cache.mkdir(exist_ok=False)
    seed = Path("/seed-cache") / f"tp2-diagnostic-{args.backend}"
    for kind in ("vllm", "triton", "cuda", "flashinfer"):
        if (seed / kind).exists():
            shutil.copytree(seed / kind, cache / kind)
    env = os.environ.copy()
    env.update(
        PYTHONPATH="/shadow-helpers:/source",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
        NCCL_DEBUG="INFO", GDN_CAPTURE_ACTIVE="1",
        GDN_CAPTURE_CONFIG=str(root / "capture-config.json"),
    )
    # Python sampling is a correctness-only intervention; it is not a perf run.
    command = [
        "uv", "run", "--offline", "--no-project", sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/model", "--served-model-name", "qwen-fp8",
        "--host", "127.0.0.1", "--port", "8000", "--language-model-only",
        "--dtype", "bfloat16", "--tensor-parallel-size", "2",
        "--max-model-len", "4096", "--max-num-seqs", "32",
        "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.9",
        "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", "float32",
        "--seed", "42", "--enforce-eager",
        "--attention-config", json.dumps({"backend": "FLASH_ATTN", "flash_attn_version": 2}),
        "--kernel-config", json.dumps({"gdn_decode_backend": args.backend, "linear_backend": "marlin"}),
        "--num-gpu-blocks-override", "128",
    ]
    record = {
        "source_head": HEAD, "source_files": actual,
        "source_manifest_sha256": sha(manifest_path), "command": command,
        "backend": args.backend, "fixture": fixture,
        "helper_sha256": {p.name: sha(p) for p in Path("/shadow-helpers").glob("*.py")},
        "model_config_sha256": sha(Path("/model/config.json")),
        "scope": "Actual ordinary decode capture, eager TP2; changes scheduling. Not a performance run, not a reproduction of the prior graph trajectories.",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "starting",
    }
    write(root / "server.json", record)
    config = {
        "output_dir": str(root / "captures"), "start_marker": str(root / "START"),
        "early_calls": [1, 2], "sample_calls": [19, 253, 454, 995, 1750],
        "sample_layers": [0, 1, 2, 16, 32, 48, 62],
        "initial_batch_size": 8,
        "expected_module_sha256": actual[
            "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
        ],
    }
    write(root / "capture-config.json", config)
    with (root / "serve.log").open("x") as log:
        server = subprocess.Popen(command, env=env, cwd="/source", stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = server.pid
        write(root / "server.json", record)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}")
                try:
                    with opener.open("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(3)
            else:
                raise TimeoutError("Startup >1200 seconds")
            capacities = set(int(x.replace(",", "")) for x in re.findall(
                r"GPU KV cache size: ([\d,]+) tokens", (root / "serve.log").read_text()))
            assert len(capacities) == 1 and min(capacities) >= 32768, capacities
            record.update(status="ready", kv_capacity_tokens=capacities.pop())
            write(root / "server.json", record)
            (root / "START").touch(exist_ok=False)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(lambda item: frozen.one(client_args, item), selected))
            for row in rows:
                row.update(diagnostic.flags(row, frozen))
            write(root / "responses.json", {
                "fixture": fixture, "examples": rows,
                "summary": diagnostic.summarize(rows),
                "scope": "Eight targeted diagnostic requests, no accuracy or timing claim",
            })
            captures = list((root / "captures").glob("**/*.pt"))
            assert captures, "No actual decode captures"
            record.update(status="completed", captures=len(captures),
                          responses_sha256=sha(root / "responses.json"))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            record.update(server_exit=server.returncode,
                          finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          log_sha256=sha(root / "serve.log"))
            write(root / "server.json", record)


if __name__ == "__main__":
    main()
