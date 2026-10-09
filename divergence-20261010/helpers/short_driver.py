"""Three-arm bounded prefix diagnostic; twenty observed outputs then abort.

All eight original prompts enter the unchanged synchronous eager TP2 engine
before stepping. Original3500/min0/EOS sampling is retained; native trace fixes
only outputs0..18. The output at19 is native. This is not model evaluation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

REFERENCE_SHA = "d293b322876e8236f7799a688221ddd0d10e79f8046c6601db40fa6316421e78"
HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
INDICES = [198, 206, 209, 228, 255, 285, 292, 318]
ARMS = {"A": ("triton", "triton"), "B": ("flashinfer", "triton"),
        "C": ("flashinfer", "flashinfer")}
SCOPE = ("Twenty-output prefix intervention with original3500/min0/EOS settings. "
         "Outputs0..18 use fixed FI history; output19 remains native, followed by abort. "
         "Request-bound diagnostic bypass of native trace termination normalization retains generation/tokenizer-derived EOS. "
         "Synchronized observations change execution; no accuracy/performance or broad causal claim.")


def require(value, label):
    if not value:
        raise ValueError(label)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def enqueue_requests(engine, examples, make_params):
    """Keep public output/abort IDs separate from randomized internal IDs."""
    external_ids, by_external, internal_by_external = [], {}, {}
    for row in examples:
        external_id = str(row["index"])
        if external_id in by_external:
            raise ValueError("Duplicate external request identity")
        params = make_params(row)
        internal_id = engine.add_request(
            external_id, {"prompt_token_ids": row["prompt_token_ids"]}, params
        )
        external_ids.append(external_id)
        by_external[external_id] = row
        internal_by_external[external_id] = internal_id
    return external_ids, by_external, internal_by_external


def observe_twenty(engine, request_ids, consume, on_abort=None):
    """Abort on success or failure; never execute an output21 engine step."""
    count = 0
    try:
        for count in range(1, 21):
            consume(count, engine.step())
    finally:
        engine.abort_request(list(request_ids))
        if on_abort is not None:
            on_abort(count)


def hashes(folder, sha):
    return {str(path.relative_to(folder)).replace("\\", "/"): sha(path)
            for path in sorted(folder.rglob("*.py")) if "__pycache__" not in path.parts}


def validate_trace(root, results):
    from late_plan import read, require
    records = {}
    for path in sorted((root / "trace").glob("events-pid-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") != "sample_batch":
                continue
            require(event["sampling_sharded"] is False, "Actual unsharded sampler")
            for row in event["records"]:
                key = (event["rank"], row["index"], row["step"])
                require(key not in records, "Exactly one actual native choice per rank/request/step")
                records[key] = row
                if row["step"] in (18, 19):
                    from late_plan import sha
                    path = root / "trace" / row["raw_logits_file"]
                    require(path.stat().st_size == row["raw_logits_bytes"] and
                            sha(path) == row["raw_logits_full_sha256"], "Raw full-vocabulary file binding")
    expected = {(rank, index, step) for rank in (0, 1) for index in INDICES for step in range(20)}
    require(set(records) == expected, "Both actual TP ranks observe all8 twenty choices")
    for result in results:
        for rank in (0, 1):
            row = records[(rank, result["index"], 19)]
            require(row["step19_native_passthrough"] and row["trace_forced"] is False,
                    "Output20 was not replaced by trace")
            require(row["native_returned_token_id"] == row["original_sampler_token_id"] ==
                    result["token_ids"][19], "Returned output20 equals actual native sampler choice")
    return {"status": "native-boundary-verified", "actual_native_positions_per_rank": 160,
            "actual_native_positions_total": len(records), "prefix_len": 19,
            "unforced_output_index": 19, "raw_full_files": 32,
            "choices_at19": {str(index): records[(0, index, 19)]["original_sampler_token_id"]
                             for index in INDICES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=Path("/artifacts/performance-source.json"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    helpers = Path(__file__).resolve().parent
    sys.path.insert(0, "/late-helpers")
    from late_plan import digest, read, sha, validate_inputs
    from late_runtime import jsonable, worker_runtime
    lock, examples = validate_inputs(args.reference)
    require(lock["reference_files"]["completed.json"] == REFERENCE_SHA, "Immutable FI eager1 reference")
    require([row["index"] for row in examples] == INDICES, "Original eight prompt order")
    for row in examples:
        require(len(row["reference_token_ids"]) >= 20, "Complete first19 fixed prefix")
        row["max_tokens"] = 3500
    prior = read(args.reference / "reference/launch.json")
    original_sampling = prior["sampling_options"]
    require(original_sampling == {"temperature": 0, "seed": 42, "max_tokens": 3500,
                                   "min_tokens": 0, "ignore_eos": False, "stop": [],
                                   "stop_token_ids": [], "logprobs": None, "prompt_logprobs": None},
            "Exact original free sampling parameters")
    require(sum(len(row["prompt_token_ids"]) for row in examples) <=
            prior["model_options"]["max_num_batched_tokens"],
            "All eight prompt prefills fit the unchanged token budget")
    backend, leaf = ARMS[args.arm]
    options = prior["model_options"].copy()
    require(options["enforce_eager"] and options["async_scheduling"] is False,
            "Original eager synchronous engine profile")
    options["kernel_config"] = options["kernel_config"].copy()
    options["kernel_config"]["gdn_decode_backend"] = backend
    options["enable_trace_replay"] = True
    plan = {"status": "CPU-plan-only", "arm": args.arm, "configured_backend": backend,
            "actual_leaf_backend": leaf, "source_head": HEAD, "options": options,
            "sampling_options": original_sampling, "requests": 8, "prefix_len": 19,
            "observe_last_step": 19, "planned_sampler_positions_per_rank": 160,
            "reference_completed_sha256": REFERENCE_SHA,
            "reference_lock_sha256": sha(args.reference / "reference-lock.json"),
            "helper_sha256": hashes(helpers, sha), "scope": SCOPE, "gpu_execution": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir
    write_new(root / "fixture.json", {"examples": examples})
    plan["fixture_sha256"] = sha(root / "fixture.json")
    if args.plan_only:
        write_new(root / "plan.json", plan)
        print(json.dumps({"status": "CPU-plan-only", "arm": args.arm, "requests": 8,
                          "fixture_sha256": plan["fixture_sha256"], "gpu_execution": False}))
        return
    llm = None
    succeeded = False
    try:
        manifest = read(args.source_manifest)
        require(manifest["source_head"] == HEAD, "Reviewed actual source head")
        source_files = dict(lock["source_files"])
        source_files.update({name: entry["sha256"] for name, entry in manifest["files"].items()})
        for name, expected in source_files.items():
            require(sha(Path("/source") / name) == expected, "Fresh source binding: " + name)
        extra = ["vllm/v1/worker/gpu/sample/sampler.py", "vllm/v1/worker/gpu/sample/trace_replay.py",
                 "vllm/v1/worker/gpu/states.py", "vllm/v1/engine/llm_engine.py",
                 "vllm/v1/attention/backends/flash_attn.py", "vllm/v1/attention/backends/triton_attn.py"]
        extra.append("vllm/v1/engine/input_processor.py")
        extra.append("vllm/sampling_params.py")
        source_files.update({name: sha(Path("/source") / name) for name in extra})
        from short_normalize import SOURCE_SHA, SAMPLING_SOURCE_SHA
        require(source_files["vllm/v1/engine/input_processor.py"] == SOURCE_SHA,
                "Actual pinned InputProcessor for diagnostic normalization hook")
        require(source_files["vllm/sampling_params.py"] == SAMPLING_SOURCE_SHA,
                "Actual pinned SamplingParams generation/EOS source")
        for name, expected in prior["model_files"].items():
            require(sha(Path("/model") / name) == expected["sha256"], "Actual unchanged model metadata: " + name)
        for package, version in prior["runtime_versions"].items():
            require(importlib.metadata.version(package) == version, "Actual runtime package: " + package)
        binding = read(args.reference / "reference/runtime-binding.json")
        for name, expected in binding["modules"].items():
            path = Path(expected["path"])
            require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                    "Actual runtime/native file: " + name)
        require(not any(name.startswith(("GDN_", "SHORT_")) for name in os.environ),
                "No inherited observer environment")
        args.workspace.mkdir(parents=True, exist_ok=False)
        seed = Path("/seed-cache") / f"tp2-diagnostic-{backend}"
        for kind in ("vllm", "triton", "cuda", "flashinfer"):
            if (seed / kind).exists():
                shutil.copytree(seed / kind, args.workspace / kind)
        os.environ.update(PYTHONPATH="/short-helpers/short_site:/short-helpers:/late-helpers:/source",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          VLLM_CACHE_ROOT=str(args.workspace / "vllm"),
                          TRITON_CACHE_DIR=str(args.workspace / "triton"),
                          CUDA_CACHE_PATH=str(args.workspace / "cuda"),
                          FLASHINFER_WORKSPACE_BASE=str(args.workspace / "flashinfer"),
                          VLLM_WORKER_MULTIPROC_METHOD="spawn", NCCL_DEBUG="INFO",
                          VLLM_USE_V2_MODEL_RUNNER="1", VLLM_ENABLE_V1_MULTIPROCESSING="0")
        sys.path.insert(0, "/source")
        cfg = {"arm": args.arm, "configured_backend": backend, "actual_leaf_backend": leaf,
               "prefix_len": 19, "observe_last_step": 19, "fixture_path": str(root / "fixture.json"),
               "output_dir": str(root / "trace"), "capture_output_dir": str(root / "joint"),
               "start_marker": str(root / "START"), "snapshot_target": 255, "snapshot_steps": [1, 18, 19],
               "source_head": HEAD, "source_files": source_files,
               "expected_runner_sha256": source_files["vllm/v1/worker/gpu/model_runner.py"],
               "expected_input_batch_sha256": source_files["vllm/v1/worker/gpu/input_batch.py"]}
        write_new(root / "short-config.json", cfg)
        plan.update(status="starting", source_files=source_files,
                    source_manifest_sha256=sha(args.source_manifest),
                    config_sha256=sha(root / "short-config.json"), gpu_execution=True,
                    model_files=prior["model_files"], runtime_versions=prior["runtime_versions"],
                    runtime_files=binding["modules"],
                    reused_helper_sha256={name: sha(Path("/late-helpers") / name)
                                           for name in ("late_plan.py", "late_runtime.py")})
        plan["normalization_intervention"] = "Exact eight-request diagnostic bypass after generation/tokenizer EOS updates; no production source edit"
        write_new(root / "launch.json", plan)
        shutil.copytree(helpers, root / "helpers", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(args.reference / "reference", root / "reference")
        shutil.copy2(args.reference / "reference-lock.json", root / "reference-lock.json")
        os.environ.update(SHORT_FORCE_ACTIVE="1", SHORT_FORCE_CONFIG=str(root / "short-config.json"),
                          SHORT_TRACE_DIR=str(root / "trace"))
        import divergence_bootstrap
        divergence_bootstrap.install()
        import short_normalize
        short_normalize.install()
        from vllm import LLM, SamplingParams
        llm = LLM(**options)
        engine = llm.llm_engine
        require(type(engine.engine_core).__name__ == "InprocClient", "Actual synchronous in-process core")
        write_new(root / "runtime-before.json", {"engine_core_class": type(engine.engine_core).__name__,
                                               "worker_ranks": llm.collective_rpc(worker_runtime, timeout=60)})
        params_records = []
        def make_params(row):
            params = SamplingParams(**original_sampling,
                                    trace_decode_token_ids=row["reference_token_ids"][:19])
            params_records.append({"index": row["index"], "params": {
                name: jsonable(getattr(params, name)) for name in params.__struct_fields__}})
            return params
        actual_ids, rows_by_id, internal_by_external = enqueue_requests(
            engine, examples, make_params
        )
        write_new(root / "sampling-runtime.json", {"examples": params_records})
        write_new(root / "enqueued.json", {"all8_before_first_step": True, "external_request_ids": actual_ids,
                                          "internal_request_ids_by_external_id": internal_by_external,
                                          "indices_by_external_request_id": {key: value["index"] for key, value in rows_by_id.items()}})
        (root / "START").write_text("Eight original prompts queued before first engine step\n", encoding="utf-8")
        latest = {}
        with (root / "progress.jsonl").open("x", encoding="utf-8", newline="\n") as progress:
            def consume(count, outputs):
                observed = []
                for output in outputs:
                    require(output.request_id in rows_by_id and len(output.outputs) == 1,
                            "One output for an actual enqueued request")
                    completion = output.outputs[0]
                    expected = rows_by_id[output.request_id]
                    value = {"index": expected["index"], "request_id": output.request_id,
                             "prompt_token_ids": list(output.prompt_token_ids),
                             "token_ids": list(completion.token_ids), "text": completion.text,
                             "finished": output.finished, "finish_reason": completion.finish_reason,
                             "stop_reason": jsonable(completion.stop_reason), "engine_step": count}
                    observed.append(value)
                    latest[output.request_id] = value
                progress.write(json.dumps({"engine_step": count, "outputs": observed}, ensure_ascii=False) + "\n")
                progress.flush()
                require(len(observed) == 8 and len({row["request_id"] for row in observed}) == 8,
                        "Same eight effective outputs at each native execution")
                for value in observed:
                    fixture = rows_by_id[value["request_id"]]
                    require(value["prompt_token_ids"] == fixture["prompt_token_ids"], "Actual unchanged prompt IDs")
                    require(len(value["token_ids"]) == count, "Actual cumulative native output count")
                    require(value["token_ids"][:min(count, 19)] == fixture["reference_token_ids"][:min(count, 19)],
                            "Actual committed/returned fixed prefix")
                    require(not value["finished"] or count == 20, "Original EOS ended before observation boundary")
            def on_abort(count):
                write_new(root / "abort.json", {"status": "public-abort-returned", "request_ids": actual_ids,
                                               "request_id_kind": "external",
                                               "after_engine_steps": count,
                                               "unfinished_after": engine.get_num_unfinished_requests(),
                                               "further_model_steps": 0})
            observe_twenty(engine, actual_ids, consume, on_abort)
        require(engine.get_num_unfinished_requests() == 0, "Actual public abort removed all8 requests")
        results = [latest[req_id] | {"token_ids_sha256": digest(latest[req_id]["token_ids"])} for req_id in actual_ids]
        write_new(root / "short-outputs.json", {"examples": results, "scope": SCOPE,
                                              "intentional_abort": True, "accuracy_evaluation": False})
        boundary = validate_trace(root, results)
        write_new(root / "native-boundary.json", boundary)
        write_new(root / "runtime-after.json", {"engine_core_class": type(engine.engine_core).__name__,
                                              "worker_ranks": llm.collective_rpc(worker_runtime, timeout=60)})
        write_new(root / "completed.json", {"status": "completed", "arm": args.arm,
                                            "launch_sha256": sha(root / "launch.json"),
                                            "native_boundary": boundary, "examples": results,
                                            "intentional_abort": True, "scope": SCOPE})
        succeeded = True
    except BaseException as error:
        write_new(root / "failure.json", {"status": "failed", "exception": type(error).__name__,
                                          "message": str(error), "traceback": traceback.format_exc(), "arm": args.arm})
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30)
            write_new(root / "shutdown.json", {"status": "shutdown-returned", "diagnostic_completed": succeeded,
                                               "engine_core_class": type(llm.llm_engine.engine_core).__name__})


if __name__ == "__main__":
    main()
