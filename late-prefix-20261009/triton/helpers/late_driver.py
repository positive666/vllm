"""Conditional long shared-history decode; artifact-only, never a quality score."""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from late_plan import HEAD, digest, read, require, sha, validate_inputs

def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

def helper_hashes(folder):
    return {str(path.relative_to(folder)).replace("\\", "/"): sha(path)
            for path in sorted(folder.rglob("*.py")) if "__pycache__" not in path.parts}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}", args.run), "Simple unique run name")
    lock, examples = validate_inputs(args.reference_root)
    helpers = Path(__file__).resolve().parent
    plan = {"backend": args.backend, "source_head": HEAD, "fixture": {"examples": examples},
            "reference_completed_sha256": lock["reference_files"]["completed.json"],
            "reference_lock_sha256": sha(args.reference_root / "reference-lock.json"),
            "helper_sha256": helper_hashes(helpers), "snapshot_steps": lock["selected_steps"],
            "target": lock["target"], "scope": lock["scope"],
            "prefill_scope": "First-decode GDN prestates and full prefill logits observed; full-attention KV equality is not observed."}
    if args.plan_only:
        print(json.dumps({"status": "CPU-plan-only", "requests": 8,
                          "observations": sum(row["max_tokens"] for row in examples),
                          "reference_completed_sha256": plan["reference_completed_sha256"],
                          "snapshot_steps": plan["snapshot_steps"], "helper_sha256": plan["helper_sha256"],
                          "gpu_execution": False}, indent=2))
        return
    source_manifest = read("/artifacts/performance-source.json")
    require(source_manifest["source_head"] == HEAD, "Reviewed production source")
    for name, entry in source_manifest["files"].items():
        require(sha(Path("/source") / name) == entry["sha256"], "Source manifest: " + name)
    for name, expected in lock["source_files"].items():
        require(sha(Path("/source") / name) == expected, "Locked runtime source: " + name)
    require(not any(key.startswith("GDN_") for key in os.environ), "No inherited observer environment")
    root = Path("/results") / args.run
    root.mkdir(exist_ok=False)
    free_bytes = shutil.disk_usage(root.parent).free
    require(free_bytes >= 15 * 1024**3, "At least15GiB owned result filesystem free for one arm")
    plan["result_filesystem_free_bytes_before"] = free_bytes
    plan["raw_recurrent_state_bytes_estimate"] = 12683575296
    cache = Path("/cache") / args.run
    cache.mkdir(exist_ok=False)
    seed = Path("/seed-cache") / f"tp2-diagnostic-{args.backend}"
    for name in ("vllm", "triton", "cuda", "flashinfer"):
        if (seed / name).exists():
            shutil.copytree(seed / name, cache / name)
    os.environ.update(PYTHONPATH="/late-helpers/late_site:/late-helpers:/source",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_CACHE_ROOT=str(cache / "vllm"), TRITON_CACHE_DIR=str(cache / "triton"),
                      CUDA_CACHE_PATH=str(cache / "cuda"), FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
                      VLLM_WORKER_MULTIPROC_METHOD="spawn", NCCL_DEBUG="INFO",
                      VLLM_USE_V2_MODEL_RUNNER="1", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    import importlib.metadata
    prior_launch = read(args.reference_root / "reference/launch.json")
    for name, expected in prior_launch["model_files"].items():
        require(sha(Path("/model") / name) == expected["sha256"], "Fresh model metadata: " + name)
    for package, version in prior_launch["runtime_versions"].items():
        require(importlib.metadata.version(package) == version, "Fresh runtime package: " + package)
    binding = read(args.reference_root / "reference/runtime-binding.json")
    for name, expected in binding["modules"].items():
        path = Path(expected["path"])
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                "Fresh runtime/native file: " + name)
    plan["fresh_model_files"] = prior_launch["model_files"]
    plan["fresh_runtime_versions"] = prior_launch["runtime_versions"]
    plan["fresh_runtime_files"] = binding["modules"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    require(tokenizer.eos_token_id == 248046, "Expected actual tokenizer EOS candidate")
    write_new(root / "fixture.json", plan["fixture"])
    cfg = {"output_dir": str(root / "trace"), "fixture_path": str(root / "fixture.json"),
           "target_indices": [209, 255], "limit": 3500, "snapshot_target": 255,
           "snapshot_steps": plan["snapshot_steps"], "eos_candidate_ids": [tokenizer.eos_token_id],
           "eos_observation_scope": "TokenizerEOS248046candidate only; additional model EOS/stop IDs are not raw-logit-covered",
           "expected_runner_sha256": lock["source_files"]["vllm/v1/worker/gpu/model_runner.py"],
           "expected_input_batch_sha256": lock["source_files"]["vllm/v1/worker/gpu/input_batch.py"]}
    write_new(root / "force-config.json", cfg)
    capture_cfg = {"output_dir": str(root / "snapshots"), "start_marker": str(root / "START"),
                   "early_calls": [], "sample_calls": [], "sample_layers": [], "initial_batch_size": 8,
                   "source_head": HEAD,
                   "expected_module_sha256": lock["source_files"]["vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"]}
    write_new(root / "capture-config.json", capture_cfg)
    os.environ.update(GDN_LATE_ACTIVE="1", GDN_FORCE_ACTIVE="1", GDN_FORCE_RUNNER="v2",
                      GDN_FORCE_CONFIG=str(root / "force-config.json"), GDN_CAPTURE_ACTIVE="1",
                      GDN_CAPTURE_CONFIG=str(root / "capture-config.json"))
    import late_bootstrap
    late_bootstrap.install()
    from vllm import LLM, SamplingParams
    options = {"model": "/model", "language_model_only": True, "dtype": "bfloat16",
               "tensor_parallel_size": 2, "max_model_len": 4096, "max_num_seqs": 32,
               "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.9,
               "enable_prefix_caching": False, "mamba_ssm_cache_dtype": "float32", "seed": 42,
               "enforce_eager": True, "async_scheduling": False, "enable_trace_replay": True,
               "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
               "kernel_config": {"gdn_decode_backend": args.backend, "linear_backend": "marlin"},
               "num_gpu_blocks_override": 128}
    plan.update(status="starting", options=options, source_files=lock["source_files"],
                source_manifest_sha256=sha(Path("/artifacts/performance-source.json")),
                fixture_sha256=sha(root / "fixture.json"), force_config_sha256=sha(root / "force-config.json"),
                capture_config_sha256=sha(root / "capture-config.json"))
    write_new(root / "launch.json", plan)
    shutil.copytree(helpers, root / "helpers", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(args.reference_root / "reference", root / "reference")
    shutil.copy2(args.reference_root / "reference-lock.json", root / "reference-lock.json")
    llm = None
    succeeded = False
    try:
        llm = LLM(**options)
        require(type(llm.llm_engine.engine_core).__name__ == "InprocClient", "Actual in-process core")
        from late_runtime import worker_runtime
        write_new(root / "runtime-before.json", {"engine_core_class": type(llm.llm_engine.engine_core).__name__,
                                               "worker_ranks": llm.collective_rpc(worker_runtime, timeout=60)})
        (root / "START").write_text("All8 frozen prompts ready before generation\n")
        params = [SamplingParams(temperature=0, seed=42, ignore_eos=True,
                                 max_tokens=row["max_tokens"], stop=[], stop_token_ids=[],
                                 logprobs=None, prompt_logprobs=None,
                                 trace_decode_token_ids=row["reference_token_ids"])
                  for row in examples]
        outputs = llm.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row in examples],
                               sampling_params=params, use_tqdm=False)
        require(len(outputs) == 8, "Exactly eight returned forced histories")
        results = []
        for expected, output in zip(examples, outputs):
            require(list(output.prompt_token_ids) == expected["prompt_token_ids"], "Actual unchanged prompt")
            require(len(output.outputs) == 1 and output.finished, "One completed forced history")
            observed = list(output.outputs[0].token_ids)
            require(observed == expected["reference_token_ids"], "Actual native forced history exact")
            results.append({"index": expected["index"], "token_ids": observed,
                            "token_ids_sha256": digest(observed), "exact_forced_tokens": True,
                            "finish_reason": output.outputs[0].finish_reason})
        write_new(root / "runtime-after.json", {"engine_core_class": type(llm.llm_engine.engine_core).__name__,
                                              "worker_ranks": llm.collective_rpc(worker_runtime, timeout=60)})
        write_new(root / "completed.json", {"status": "completed", "examples": results,
                                            "launch_sha256": sha(root / "launch.json"), "scope": lock["scope"]})
        succeeded = True
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30)
            write_new(root / "shutdown.json", {"status": "shutdown-returned", "generation_completed": succeeded})

if __name__ == "__main__":
    main()
