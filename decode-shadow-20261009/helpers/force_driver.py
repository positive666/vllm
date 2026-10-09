"""Paired fixed-token-history probe using genuine incremental model decode.

Reference tokens are supplied deliberately. Output is never an accuracy or
serving performance result. All prompts are queued together; finished shorter
reference sequences are not extended to maintain an artificial batch of eight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path("/results") / args.run
    root.mkdir(exist_ok=False)
    manifest_path = Path("/artifacts/performance-source.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_head"] == "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
    assert all(sha(Path("/source") / name) == entry["sha256"] for name, entry in manifest["files"].items())
    reference_path = Path("/oldresults/tp2-followup-20261008/diagnostics/triton.json")
    reference = json.loads(reference_path.read_text())
    assert reference["arm"] == "triton"
    rows = next(r["examples"] for r in reference["rounds"] if r["concurrency"] == 8)
    assert [r["index"] for r in rows] == [198, 206, 209, 228, 255, 285, 292, 318]
    cache = Path("/cache") / args.run
    cache.mkdir(exist_ok=False)
    seed = Path("/seed-cache") / f"tp2-diagnostic-{args.backend}"
    for kind in ("vllm", "triton", "cuda", "flashinfer"):
        if (seed / kind).exists():
            shutil.copytree(seed / kind, cache / kind)
    os.environ.update(
        PYTHONPATH="/shadow-helpers/force_site:/shadow-helpers:/source",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VLLM_CACHE_ROOT=str(cache / "vllm"), TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_CACHE_PATH=str(cache / "cuda"), FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
        VLLM_WORKER_MULTIPROC_METHOD="spawn", NCCL_DEBUG="INFO",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    examples = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if isinstance(prompt_ids, Mapping):
            prompt_ids = prompt_ids["input_ids"]
        assert isinstance(prompt_ids, list) and all(type(t) is int for t in prompt_ids)
        assert len(prompt_ids) == row["usage"]["prompt_tokens"]
        assert row["token_ids_sha256"] == hashlib.sha256(
            json.dumps(row["token_ids"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(), "Reference tokens hash"
        examples.append({
            "index": row["index"], "prompt_token_ids": prompt_ids,
            "reference_token_ids": row["token_ids"],
            "max_tokens": min(len(row["token_ids"]), 512),
            "reference_token_count": len(row["token_ids"]),
        })
    fixture_path = root / "fixture.json"
    write_new(fixture_path, {"examples": examples, "reference_sha256": sha(reference_path)})
    config_path = root / "force-config.json"
    write_new(config_path, {
        "output_dir": str(root / "trace"), "fixture_path": str(fixture_path),
        "target_indices": [209, 255], "limit": 512,
        "expected_runner_sha256": sha(Path("/source/vllm/v1/worker/gpu/model_runner.py")),
    })
    if args.plan_only:
        print(json.dumps({"fixture_sha256": sha(fixture_path),
                          "examples": [{k: e[k] for k in ("index", "max_tokens", "reference_token_count")} for e in examples]}, indent=2))
        return
    os.environ.update(GDN_FORCE_ACTIVE="1", GDN_FORCE_CONFIG=str(config_path),
                      GDN_FORCE_RUNNER="v2")
    import force_decode_v2
    force_decode_v2.install()
    from vllm import LLM, SamplingParams
    options = {
        "model": "/model", "language_model_only": True, "dtype": "bfloat16",
        "tensor_parallel_size": 2, "max_model_len": 4096, "max_num_seqs": 32,
        "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": False, "mamba_ssm_cache_dtype": "float32", "seed": 42,
        "enforce_eager": True, "async_scheduling": False,
        "enable_trace_replay": True,
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "kernel_config": {"gdn_decode_backend": args.backend, "linear_backend": "marlin"},
        "num_gpu_blocks_override": 128,
    }
    record = {
        "backend": args.backend, "options": options, "fixture_sha256": sha(fixture_path),
        "config_sha256": sha(config_path), "reference_sha256": sha(reference_path),
        "engine_core_multiprocessing": False,
        "helper_sha256": {str(p.relative_to("/shadow-helpers")): sha(p)
                          for p in Path("/shadow-helpers").rglob("*.py") if "__pycache__" not in p.parts},
        "scope": "Teacher-forced incremental decode of frozen reference histories. Eager, synchronous TP2, no quality score or timing claim. Shorter references finish at their own lengths.",
        "status": "starting",
    }
    write_new(root / "launch.json", record)
    shutil.copytree("/shadow-helpers", root / "helpers", ignore=shutil.ignore_patterns("__pycache__"))
    llm = LLM(**options)
    prompts = [{"prompt_token_ids": e["prompt_token_ids"]} for e in examples]
    sampling = [SamplingParams(temperature=0, seed=42, ignore_eos=True,
                               max_tokens=e["max_tokens"], stop=[], stop_token_ids=[],
                               logprobs=None, prompt_logprobs=None,
                               trace_decode_token_ids=e["reference_token_ids"][:e["max_tokens"]])
                for e in examples]
    try:
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)
    assert len(outputs) == len(examples) == 8, "Eight actual request outputs"
    results = []
    for expected, output in zip(examples, outputs):
        assert list(output.prompt_token_ids) == expected["prompt_token_ids"]
        assert len(output.outputs) == 1
        observed = list(output.outputs[0].token_ids)
        assert observed == expected["reference_token_ids"][:expected["max_tokens"]], "Forced tokens mismatch"
        results.append({"index": expected["index"], "generated_tokens": len(observed),
                        "exact_forced_tokens": True, "token_ids": observed,
                        "token_ids_sha256": hashlib.sha256(json.dumps(observed, separators=(",", ":")).encode()).hexdigest(),
                        "finish_reason": output.outputs[0].finish_reason})
    write_new(root / "completed.json", {"status": "completed", "examples": results,
                                        "scope": record["scope"]})


if __name__ == "__main__":
    main()
