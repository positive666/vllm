"""Audit long free-generation integrity; preserve all diagnostic outcomes.

This is a standard-library consumer. It never starts a model or interprets
forced tokens as quality evidence. Graph activation requires native runtime
dispatch observations from real requests, independently of capture settings.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import traceback
from collections import Counter
from pathlib import Path

from long_free_common import (
    HEAD, INDICES, SCORER_HASHES, SCOPE, TARGETS, digest, flags, frozen_scorer,
    model_options, read_json, require, sampling_options, sha, summarize, write_new,
)

PRODUCERS = {
    "long_free_driver.py": "dd61dc9c2924472a8065daff38e7e2a5af2774f770bc18fb2014ecc66e52fb8f",
    "long_free_common.py": "9fe3403eb45386cbf116fff22fb199930b7a7332c9e84abe4821d6f75fe59af3",
    **SCORER_HASHES,
}
DATASET_SHA = "ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
GRAPH_TABLE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*"
    r"((?:CUDAGraphMode\.)?(?:NONE|FULL|PIECEWISE))\s*\|\s*(\d+)\s*\|"
)


def host_evidence(root, workers):
    host_path = root.parent / (root.name + "-host.csv")
    process_path = root.parent / (root.name + "-processes.log")
    with host_path.open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle, skipinitialspace=True))
    require(bool(samples), "Actual host telemetry")
    gpu_uuids = {row["gpu_uuid"].removeprefix("GPU-") for row in workers}
    groups = {}
    for sample in samples:
        require(sample["uuid"].removeprefix("GPU-") in gpu_uuids,
                "Telemetry for the two actual GPUs")
        groups.setdefault(sample["utc"], []).append(sample)
    require(list(groups) == sorted(groups), "Monotonic host sample timestamps")
    for group in groups.values():
        require(len(group) == 2 and {row["uuid"].removeprefix("GPU-") for row in group} ==
                gpu_uuids, "Both GPUs in every host sample")
    first = groups[next(iter(groups))]
    last_time = next(reversed(groups))
    last = groups[last_time]
    for label, group in (("preflight", first), ("released", last)):
        require(all(int(row["memory.used"].split()[0]) <= 16 and
                    int(row["utilization.gpu"].split()[0]) == 0 for row in group),
                "Both GPUs idle at " + label)
    process_log = process_path.read_text(encoding="utf-8")
    stamps = list(re.finditer(r"(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s*$", process_log))
    require(bool(stamps) and stamps[-1].group().strip() == last_time,
            "Final process sample matches GPU sample")
    final_block = process_log[stamps[-1].end():]
    require(not re.search(r"(?m)^\s*GPU-[^,\n]+,\s*\d+,", final_block),
            "No GPU compute processes in final sample")
    return {
        "host_csv_sha256": sha(host_path), "process_log_sha256": sha(process_path),
        "host_samples": len(groups), "first_utc": first[0]["utc"], "last_utc": last_time,
        "first_gpu_rows": first, "last_gpu_rows": last,
        "final_gpu_compute_processes": [],
        "scope": "Agent experiment GPUs only; this does not inspect or claim other users' GPUs were idle.",
    }


def validate_example(row, expected):
    for key in ("index", "question", "gold", "gold_text", "prompt", "target"):
        require(row[key] == expected[key], "Frozen result field: " + key)
    prompt = row["prompt_token_ids"]
    tokens = row["token_ids"]
    require(prompt == expected["prompt_token_ids"], "Actual prompt token IDs")
    require(digest(prompt) == row["prompt_token_ids_sha256"] ==
            expected["prompt_token_ids_sha256"], "Prompt token SHA256")
    require(isinstance(tokens, list) and all(type(token) is int and token >= 0
                                            for token in tokens), "Actual output token IDs")
    require(0 < len(tokens) <= 3500, "Bounded nonempty generation")
    require(digest(tokens) == row["token_ids_sha256"], "Output token SHA256")
    require(len(prompt) + len(tokens) <= 4096, "Actual context bound")
    require(row["usage"] == {"prompt_tokens": len(prompt),
                              "completion_tokens": len(tokens),
                              "total_tokens": len(prompt) + len(tokens)}, "Actual usage counts")
    require(row["finish_reason"] in ("stop", "length"), "Expected native finish reason")
    if row["finish_reason"] == "length":
        require(len(tokens) == 3500, "Length termination at configured budget")
    require(isinstance(row["text"], str), "Actual native completion text")
    recomputed = flags(row["text"], row["gold"], row["finish_reason"])
    for name, value in recomputed.items():
        require(row[name] == value, "Recomputed score: " + name)
    return recomputed


def runtime_stats(root, after):
    log_path = root.parent / (root.name + ".log")
    log = ANSI.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    logged = [{"num_unpadded_tokens": int(unpadded),
               "num_padded_tokens": int(padded), "num_paddings": int(paddings),
               "runtime_mode": mode.split(".")[-1], "count": int(count)}
              for unpadded, padded, paddings, mode, count in GRAPH_TABLE.findall(log)]
    pending = [{**item, "runtime_mode": item["runtime_mode"].split(".")[-1], "count": 1}
               for item in after["native_pending_cudagraph_stats"]]
    counts = Counter()
    for item in logged + pending:
        require(item["num_paddings"] == item["num_padded_tokens"] -
                item["num_unpadded_tokens"], "Native graph padding accounting")
        require(item["num_paddings"] >= 0 and item["count"] > 0,
                "Positive native dispatch count")
        counts[item["runtime_mode"]] += item["count"]
    small_decode = [item for item in logged + pending
                    if 1 <= item["num_unpadded_tokens"] <= 8 and
                    item["runtime_mode"] in ("FULL", "PIECEWISE")]
    return {
        "log_sha256": sha(log_path), "log_bytes": log_path.stat().st_size,
        "native_logged_interval_rows": logged,
        "native_pending_rows": pending,
        "observed_dispatch_counts": dict(counts),
        "observed_decode_graph_dispatch_count": sum(item["count"] for item in small_decode),
        "scope": "Native dispatch modes during actual LLM.generate requests; counters are observational, not performance measurements.",
    }


def verify_workers(runtime, backend, mode):
    require(runtime["engine_core_class"] == "InprocClient", "Actual synchronous core")
    workers = runtime["worker_ranks"]
    require(len(workers) == 2 and {row["rank"] for row in workers} == {0, 1},
            "Both actual ranks")
    require({row["tp_rank"] for row in workers} == {0, 1}, "Distinct actual TP ranks")
    require(len({row["gpu_uuid"] for row in workers}) == 2 and
            all(row["gpu_uuid"] != "unavailable" for row in workers),
            "Two identified physical GPUs")
    for row in workers:
        require(row["tp_world_size"] == 2 and row["tp_rank"] in (0, 1) and
                row["tp_device_backend"] == "nccl", "Real TP2 NCCL")
        require(row["runner_class"] == "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
                "Actual native model runner V2")
        config = row["config"]
        model = config["model_config"]
        cache = config["cache_config"]
        scheduler = config["scheduler_config"]
        require(model["enable_trace_replay"] is False, "No runtime trace replay")
        require(model["seed"] == 42, "Resolved model seed")
        require(model["enforce_eager"] == (mode == "eager"), "Resolved eager flag")
        require(model["max_model_len"] == 4096 and str(model["dtype"]).endswith("bfloat16"),
                "Resolved model context/dtype")
        require(cache["enable_prefix_caching"] is False and
                cache["mamba_ssm_cache_dtype"] == "float32", "Resolved cache settings")
        require(cache["num_gpu_blocks_override"] == 128, "Actual block override")
        require(scheduler["async_scheduling"] is False and
                scheduler["max_num_seqs"] == 32 and
                scheduler["max_num_batched_tokens"] == 1024, "Resolved scheduler settings")
        require(config["parallel_config"]["tensor_parallel_size"] == 2,
                "Resolved TP size")
        require(config["kernel_config"]["gdn_decode_backend"] == backend and
                config["kernel_config"]["linear_backend"] == "marlin", "Resolved kernels")
        require(config["attention_config"]["backend"] == "FLASH_ATTN" and
                config["attention_config"]["flash_attn_version"] == 2, "Resolved FA2")
        require(config["observability_config"]["cudagraph_metrics"] is True,
                "Native CUDA graph observations enabled")
        require(config["speculative_config"] is None, "No speculative decoding")
        cg_mode = config["compilation_config"]["cudagraph_mode"]
        if mode == "eager":
            require(cg_mode == "NONE" and not row["graphs_captured"] and
                    not row["captured_token_counts"] and not row["graph_descriptors"],
                    "Eager has no actual captured graphs")
        else:
            compilation = config["compilation_config"]
            require(cg_mode == "FULL_AND_PIECEWISE" and
                    compilation["max_cudagraph_capture_size"] == 64 and
                    compilation["cudagraph_capture_sizes"] ==
                    [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64],
                    "Exact resolved default CUDA graph profile")
            require(row["graphs_captured"] and
                    bool(row["captured_token_counts"]) and bool(row["graph_descriptors"]) and
                    row["compilation_counter"]["num_cudagraph_captured"] > 0,
                    "Both graph workers actually captured graphs")
    return workers


def audit_run(root, trusted_reference, trusted_manifest, source_root=None, model_root=None):
    require(not (root / "failure.json").exists(), "No producer failure")
    require(not (root / "plan-completed.json").exists(), "Actual generation, not CPU plan")
    exit_path = root.parent / (root.name + ".exit")
    require(exit_path.read_text().strip() == "0", "Actual process exit 0")
    launch = read_json(root / "launch.json")
    fixture = read_json(root / "fixture.json")
    reference = read_json(root / "reference.json")
    manifest = read_json(root / "source-manifest.json")
    completed = read_json(root / "completed.json")
    before = read_json(root / "runtime-before.json")
    after = read_json(root / "runtime-after.json")
    sampling = read_json(root / "sampling-runtime.json")
    shutdown = read_json(root / "shutdown.json")
    require(launch["source_head"] == manifest["source_head"] == HEAD,
            "Unchanged reviewed head")
    require(sha(root / "source-manifest.json") == launch["source_manifest_sha256"] ==
            sha(trusted_manifest), "Trusted source manifest")
    require(sha(root / "reference.json") == launch["reference_sha256"] ==
            fixture["reference_sha256"] == sha(trusted_reference), "Trusted frozen reference")
    require(reference["full_fixture"]["dataset_sha256"] == DATASET_SHA,
            "Original frozen GSM8K dataset identity")
    require(launch["helper_sha256"] == PRODUCERS, "Reviewed frozen producer helpers")
    for name, expected in PRODUCERS.items():
        require(sha(root / "helpers" / name) == expected, "Snapshot helper bytes: " + name)
    require(sha(root / "fixture.json") == launch["fixture_sha256"], "Fixture bytes")
    require(sha(root / "launch.json") == completed["launch_sha256"], "Launch binding")
    require(launch["model_options"] == model_options(launch["backend"], launch["mode"]),
            "Exact requested model settings")
    require(launch["sampling_options"] == sampling_options(), "Exact requested sampling")
    for name, expected in sampling_options().items():
        # vLLM normalizes an empty stop list to None internally.
        actual = sampling[name]
        require(actual == expected or (name == "stop" and expected == [] and actual is None),
                "Actual runtime sampling: " + name)
    require(sampling["trace_decode_token_ids"] is None, "No supplied decode history")
    require(launch["engine_core_multiprocessing"] is False and
            launch["all_requests_enqueued_before_step"] is True, "Synchronous all-eight batch")
    for name, expected in manifest["files"].items():
        require(launch["source_files"][name] == expected["sha256"], "Manifest source: " + name)
    if source_root is not None:
        for name, expected in launch["source_files"].items():
            require(sha(source_root / name) == expected, "Source tree bytes: " + name)
    if model_root is not None:
        for name, expected in launch["model_files"].items():
            require(sha(model_root / name) == expected["sha256"], "Model metadata: " + name)
    require(shutdown == {"status": "shutdown-returned", "generation_completed": True,
                         "engine_core_class": "InprocClient"}, "Actual successful shutdown")
    require(completed["status"] == "completed" and
            completed["backend"] == launch["backend"] and completed["mode"] == launch["mode"],
            "Completed arm metadata")
    expected_rows = fixture["examples"]
    actual_rows = completed["examples"]
    ref_rows = next(item["examples"] for item in reference["rounds"]
                    if item["concurrency"] == 8)
    require([row["index"] for row in expected_rows] == INDICES ==
            [row["index"] for row in actual_rows] == [row["index"] for row in ref_rows],
            "Exactly eight selected outputs")
    for expected, actual, ref in zip(expected_rows, actual_rows, ref_rows):
        for key in ("index", "question", "gold", "gold_text", "prompt"):
            require(expected[key] == ref[key], "Original reference field: " + key)
        require(expected["target"] == (expected["index"] in TARGETS), "Frozen target mask")
        require(expected["gold"] == frozen_scorer().answer_number(expected["gold_text"]),
                "Independent frozen gold parsing")
        require(len(expected["prompt_token_ids"]) == ref["usage"]["prompt_tokens"],
                "Original prompt length")
        validate_example(actual, expected)
    require(completed["summary"] == summarize(actual_rows), "Recomputed outcome counts")
    before_workers = verify_workers(before, launch["backend"], launch["mode"])
    workers = verify_workers(after, launch["backend"], launch["mode"])
    for first, last in zip(sorted(before_workers, key=lambda row: row["rank"]),
                           sorted(workers, key=lambda row: row["rank"])):
        for key in ("rank", "tp_rank", "gpu_uuid", "gpu_name", "compute_capability",
                    "nccl_version", "runner_class"):
            require(first[key] == last[key], "Same startup/completed worker: " + key)
    graph = runtime_stats(root, after)
    if launch["mode"] == "eager":
        require(not any(key in graph["observed_dispatch_counts"]
                        for key in ("FULL", "PIECEWISE")), "No eager graph replay")
    else:
        require(graph["observed_decode_graph_dispatch_count"] > 0,
                "Actual graph replay during small decode batches")
    host = host_evidence(root, workers)
    input_hashes = {path.name: sha(path) for path in root.glob("*.json")}
    input_hashes[exit_path.name] = sha(exit_path)
    input_hashes[root.name + ".log"] = graph["log_sha256"]
    input_hashes[root.name + "-host.csv"] = host["host_csv_sha256"]
    input_hashes[root.name + "-processes.log"] = host["process_log_sha256"]
    return {
        "run": root.name, "backend": launch["backend"], "mode": launch["mode"],
        "summary": summarize(actual_rows), "graph_evidence": graph,
        "host_evidence": host,
        "gpu_uuids": [row["gpu_uuid"] for row in sorted(workers, key=lambda row: row["rank"])],
        "source_files": launch["source_files"], "fixture_sha256": launch["fixture_sha256"],
        "model_files": launch["model_files"], "runtime_versions": launch["runtime_versions"],
        "sampling_runtime": sampling, "input_sha256": input_hashes,
        "examples": actual_rows,
    }


def compare(left, right):
    rows = []
    for a, b in zip(left["examples"], right["examples"]):
        require(a["index"] == b["index"] and a["prompt_token_ids"] == b["prompt_token_ids"],
                "Paired actual prompt identities")
        minimum = min(len(a["token_ids"]), len(b["token_ids"]))
        first = next((index for index in range(minimum)
                      if a["token_ids"][index] != b["token_ids"][index]), None)
        if first is None and len(a["token_ids"]) != len(b["token_ids"]):
            first = minimum
        rows.append({
            "index": a["index"], "target": a["target"],
            "left": {key: a[key] for key in ("predicted", "raw_correct", "strict_correct",
                                              "truncated", "has_answer_marker", "finish_reason")},
            "right": {key: b[key] for key in ("predicted", "raw_correct", "strict_correct",
                                               "truncated", "has_answer_marker", "finish_reason")},
            "left_tokens": len(a["token_ids"]), "right_tokens": len(b["token_ids"]),
            "same_answer": a["predicted"] == b["predicted"],
            "same_generated_token_ids": a["token_ids"] == b["token_ids"],
            "first_token_divergence_0_based": first,
            "left_first_token": a["token_ids"][first] if first is not None and first < len(a["token_ids"]) else None,
            "right_first_token": b["token_ids"][first] if first is not None and first < len(b["token_ids"]) else None,
        })
    kind = ("same-cell-repeat" if (left["backend"], left["mode"]) ==
            (right["backend"], right["mode"]) else "backend-pair" if
            left["mode"] == right["mode"] else "mode-pair" if
            left["backend"] == right["backend"] else "both-factors-changed")
    return {"left_run": left["run"], "right_run": right["run"], "kind": kind,
            "same_answers": sum(row["same_answer"] for row in rows),
            "same_token_histories": sum(row["same_generated_token_ids"] for row in rows),
            "target_answer_or_truncation_difference": any(
                row["target"] and (not row["same_answer"] or
                                   row["left"]["truncated"] != row["right"]["truncated"])
                for row in rows), "examples": rows}


def audit(args):
    runs = [audit_run(path, args.reference, args.source_manifest,
                      args.source_root, args.model_root) for path in args.runs]
    require(len({item["run"] for item in runs}) == len(runs), "Unique run names")
    cells = {(item["backend"], item["mode"]) for item in runs}
    require(cells == {(backend, mode) for backend in ("triton", "flashinfer")
                     for mode in ("eager", "graph")}, "Complete four-cell initial matrix")
    for left, right in zip(runs, runs[1:]):
        for key in ("source_files", "fixture_sha256", "model_files", "runtime_versions", "gpu_uuids"):
            require(left[key] == right[key], "Same paired " + key)
        require(left["sampling_runtime"] == right["sampling_runtime"], "Same actual sampler config")
    comparisons = [compare(left, right) for left, right in itertools.combinations(runs, 2)]
    return {
        "integrity_pass": True, "scope": SCOPE, "source_head": HEAD,
        "auditor_sha256": sha(Path(__file__)),
        "reference_sha256": sha(args.reference),
        "source_manifest_sha256": sha(args.source_manifest),
        "runs": runs, "comparisons": comparisons,
        "cell_run_counts": {
            backend + "/" + mode: sum(row["backend"] == backend and row["mode"] == mode
                                       for row in runs)
            for backend, mode in sorted(cells)
        },
        "limitations": [
            "Eight selected diagnostic questions; no model-wide quality equivalence claim.",
            "Free generation can amplify low-margin differences; differences alone do not establish a kernel defect.",
            "Offline in-process synchronous execution differs from the preserved original HTTP scheduling protocol.",
            "Native completion text is scored independently; this standard-library audit does not re-decode model token IDs.",
            "Graph runtime counts prove observed replay, not performance or a root cause for any output difference.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite audit evidence")
    try:
        result = audit(args)
    except BaseException as error:
        write_new(args.output, {"integrity_pass": False, "exception": type(error).__name__,
                                "message": str(error), "traceback": traceback.format_exc()})
        raise
    write_new(args.output, result)
    print({"integrity_pass": True, "runs": len(result["runs"]),
           "requests": sum(row["summary"]["requests"] for row in result["runs"])})


if __name__ == "__main__":
    main()
