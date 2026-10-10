"""Unforced long generation: eager/graph x Triton/FlashInfer, eight fixed items.

Run each arm in a fresh process/cache. This uses LLM.generate with an in-process
engine core, so all eight requests are queued before its first engine step.
No runner patches, trace replay, logprobs or supplied output tokens are used.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

from long_free_common import (
    HEAD, INDICES, SCOPE, SOURCE_PATHS, TARGETS, digest, flags, frozen_scorer,
    jsonable, model_options, read_json, require, sampling_options, sha, summarize,
    write_new,
)


def worker_runtime(worker):
    """Read actual per-rank runtime state through public collective_rpc."""
    import torch
    import torch.distributed as dist
    from vllm.compilation.counter import compilation_counter
    from vllm.distributed.parallel_state import get_tp_group

    runner = worker.model_runner
    config = worker.vllm_config
    manager = runner.cudagraph_manager
    tp = get_tp_group()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    result = {
        "rank": worker.rank,
        "runner_class": type(runner).__module__ + "." + type(runner).__name__,
        "tp_rank": tp.rank_in_group, "tp_world_size": tp.world_size,
        "tp_device_backend": dist.get_backend(tp.device_group),
        "cuda_device": device, "gpu_name": props.name,
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "nccl_version": list(torch.cuda.nccl.version()),
        "config": {
            name: jsonable(getattr(config, name))
            for name in (
                "model_config", "parallel_config", "cache_config",
                "scheduler_config", "compilation_config", "kernel_config",
                "attention_config", "observability_config", "speculative_config",
            )
        },
        "compilation_counter": jsonable(compilation_counter),
        "graphs_captured": bool(manager and manager._graphs_captured),
        "captured_token_counts": manager.captured_token_counts() if manager else [],
        "graph_descriptors": [jsonable(item) for item in manager.graphs]
        if manager else [],
    }
    return result


def pending_graph_stats(llm):
    """Read native logging stats left since the last logging interval."""
    rows = []
    manager = llm.llm_engine.logger_manager
    if manager is not None:
        for adapter in manager.stat_loggers:
            for logger in getattr(adapter, "per_engine_stat_loggers", {}).values():
                collector = getattr(logger, "cudagraph_logging", None)
                if collector is not None:
                    rows.extend(jsonable(item) for item in collector.stats)
    return rows


def prepare(root, args):
    manifest_path = Path("/artifacts/performance-source.json")
    manifest = read_json(manifest_path)
    require(manifest["source_head"] == HEAD, "Reviewed source head")
    for name, entry in manifest["files"].items():
        require(sha(Path("/source") / name) == entry["sha256"], "Source hash: " + name)
    reference_path = Path("/oldresults/tp2-followup-20261008/diagnostics/triton.json")
    reference = read_json(reference_path)
    require(reference["arm"] == "triton", "Frozen reference backend")
    rows = next(item["examples"] for item in reference["rounds"]
                if item["concurrency"] == 8)
    require([item["index"] for item in rows] == INDICES, "Eight original C8 items")
    cache = Path("/cache") / args.run
    cache.mkdir(exist_ok=False)
    seed = Path("/seed-cache") / f"tp2-diagnostic-{args.backend}"
    seeded = []
    for kind in ("vllm", "triton", "cuda", "flashinfer"):
        if (seed / kind).exists():
            shutil.copytree(seed / kind, cache / kind)
            seeded.append(kind)
    os.environ.update(
        PYTHONPATH="/long-helpers:/source", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"), CUDA_CACHE_PATH=str(cache / "cuda"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
        VLLM_WORKER_MULTIPROC_METHOD="spawn", NCCL_DEBUG="INFO",
        VLLM_USE_V2_MODEL_RUNNER="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    require(not any(key.startswith("GDN_FORCE") for key in os.environ),
            "No inherited forcing environment")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    scorer = frozen_scorer()
    examples = []
    for row in rows:
        require(row["gold"] == scorer.answer_number(row["gold_text"]), "Frozen gold")
        require(row["prompt"] == row["question"] +
                "\nSolve step by step. End with #### followed by the final numeric answer.",
                "Frozen original prompt")
        require(digest(row["token_ids"]) == row["token_ids_sha256"], "Reference tokens")
        messages = [{"role": "user", "content": row["prompt"]}]
        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        require(isinstance(ids, list) and all(type(token) is int for token in ids),
                "Flat actual prompt token IDs")
        require(len(ids) == row["usage"]["prompt_tokens"], "Original prompt token count")
        require(len(ids) + 3500 <= 4096, "Requested model context")
        examples.append({
            key: row[key] for key in ("index", "question", "gold", "gold_text", "prompt")
        } | {
            "target": row["index"] in TARGETS, "chat_template_text": template,
            "prompt_token_ids": ids, "prompt_token_ids_sha256": digest(ids),
            "reference_completion_sha256": row["token_ids_sha256"],
        })
    fixture = {"examples": examples, "reference_sha256": sha(reference_path),
               "selection_rule": "Original C8 batch containing 209 and 255", "scope": SCOPE}
    write_new(root / "fixture.json", fixture)
    shutil.copy2(reference_path, root / "reference.json")
    shutil.copy2(manifest_path, root / "source-manifest.json")
    helper_dir = Path(__file__).resolve().parent
    helpers = {name: sha(helper_dir / name) for name in (
        "long_free_driver.py", "long_free_common.py", "frozen_quality_http.py",
        "http_bench_client.py",
    )}
    (root / "helpers").mkdir(exist_ok=False)
    for name in helpers:
        shutil.copy2(helper_dir / name, root / "helpers" / name)
    source = {name: sha(Path("/source") / name)
              for name in sorted(set(SOURCE_PATHS) | set(manifest["files"]))}
    model_files = {
        path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in Path("/model").iterdir()
        if path.is_file() and path.name in (
            "config.json", "generation_config.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.jinja", "tokenizer.json",
            "model.safetensors.index.json",
        )
    }
    versions = {}
    for package in ("torch", "flashinfer-python", "nvidia-cutlass-dsl", "triton",
                    "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "distribution metadata unavailable"
    launch = {
        "status": "prepared", "backend": args.backend, "mode": args.mode,
        "source_head": HEAD, "source_files": source,
        "source_manifest_sha256": sha(manifest_path), "helper_sha256": helpers,
        "fixture_sha256": sha(root / "fixture.json"),
        "reference_sha256": sha(reference_path), "model_files": model_files,
        "model_hash_scope": "Configuration/tokenizer/index only, not all weight shards",
        "runtime_versions": versions, "python": sys.version,
        "model_options": model_options(args.backend, args.mode),
        "sampling_options": sampling_options(),
        "tokenization_options": {"enable_thinking": False, "add_generation_prompt": True},
        "cache_seed": str(seed), "cache_seed_kinds": seeded,
        "environment": {key: os.environ[key] for key in (
            "PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_CACHE_ROOT",
            "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE",
            "VLLM_WORKER_MULTIPROC_METHOD", "NCCL_DEBUG", "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
        )},
        "command": [sys.executable, *sys.argv],
        "engine_core_multiprocessing": False, "all_requests_enqueued_before_step": True,
        "scope": SCOPE,
        "graph_proof": "Per-rank actual capture state plus native CUDA graph runtime stats/log tables; configuration alone is insufficient.",
    }
    write_new(root / "launch.json", launch)
    return examples, launch


def run(root, args):
    examples, launch = prepare(root, args)
    if args.plan_only:
        write_new(root / "plan-completed.json", {
            "status": "plan-only", "fixture_sha256": launch["fixture_sha256"],
            "source_head": HEAD, "requests": len(examples), "scope": SCOPE,
        })
        print({"status": "plan-only", "requests": 8, "fixture_sha256": launch["fixture_sha256"]},
              flush=True)
        return
    from vllm import LLM, SamplingParams
    llm = None
    completed = False
    try:
        llm = LLM(**launch["model_options"])
        require(type(llm.llm_engine.engine_core).__name__ == "InprocClient",
                "Actual in-process engine core")
        write_new(root / "runtime-before.json", {
            "engine_core_class": type(llm.llm_engine.engine_core).__name__,
            "worker_ranks": llm.collective_rpc(worker_runtime, timeout=60),
        })
        prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row in examples]
        params = SamplingParams(**launch["sampling_options"])
        require(params.trace_decode_token_ids is None, "No native trace token forcing")
        write_new(root / "sampling-runtime.json", {
            name: jsonable(getattr(params, name)) for name in params.__struct_fields__
        })
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
        require(len(outputs) == len(examples) == 8, "Eight actual outputs")
        results = []
        for expected, output in zip(examples, outputs):
            require(list(output.prompt_token_ids) == expected["prompt_token_ids"],
                    "Actual prompt IDs")
            require(len(output.outputs) == 1 and output.finished, "One finished completion")
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            row = {
                key: expected[key] for key in ("index", "question", "gold", "gold_text", "prompt", "target")
            } | {
                "request_id": output.request_id, "prompt_token_ids": list(output.prompt_token_ids),
                "prompt_token_ids_sha256": digest(list(output.prompt_token_ids)),
                "token_ids": tokens, "token_ids_sha256": digest(tokens),
                "text": completion.text, "finish_reason": completion.finish_reason,
                "stop_reason": jsonable(completion.stop_reason),
                "usage": {"prompt_tokens": len(output.prompt_token_ids),
                          "completion_tokens": len(tokens),
                          "total_tokens": len(output.prompt_token_ids) + len(tokens)},
            }
            row.update(flags(row["text"], row["gold"], row["finish_reason"]))
            results.append(row)
        write_new(root / "runtime-after.json", {
            "engine_core_class": type(llm.llm_engine.engine_core).__name__,
            "worker_ranks": llm.collective_rpc(worker_runtime, timeout=60),
            "native_pending_cudagraph_stats": pending_graph_stats(llm),
            "native_stats_scope": "Remainder since last logging interval; earlier native interval tables remain in the bound process log.",
        })
        write_new(root / "completed.json", {
            "status": "completed", "backend": args.backend, "mode": args.mode,
            "launch_sha256": sha(root / "launch.json"), "examples": results,
            "summary": summarize(results), "scope": SCOPE,
        })
        completed = True
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30)
            write_new(root / "shutdown.json", {
                "status": "shutdown-returned", "generation_completed": completed,
                "engine_core_class": type(llm.llm_engine.engine_core).__name__,
            })
    print({"status": "completed", "backend": args.backend, "mode": args.mode,
           **summarize(results)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}", args.run),
            "Unique simple run name")
    root = Path("/results") / args.run
    root.mkdir(exist_ok=False)
    try:
        run(root, args)
    except BaseException as error:
        write_new(root / "failure.json", {
            "status": "failed", "exception": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "backend": args.backend, "mode": args.mode,
        })
        raise


if __name__ == "__main__":
    main()
