"""Independently audit raw TP2 correctness and matched quality evidence.

No GPU, HTTP or publication operations occur. Input records stay unchanged.
Validate recorded kernel metrics and recompute model-quality outcomes without
reading another audit's calculated results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
DATASET_SHA = "ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13"
GLOBAL = {"H": 16, "HV": 48, "K": 128, "V": 128}
LOCAL = {"H": 8, "HV": 24, "K": 128, "V": 128}
CASES = [
    (1, "int32", False),
    (8, "int32", False),
    (8, "int32", True),
    (1, "int64", False),
    (8, "int64", False),
    (8, "int64", True),
]
VERSIONS = {
    "torch": "2.13.0+cu129",
    "triton": "3.7.1",
    "flashinfer-python": "0.7.0.post1",
    "nvidia-cutlass-dsl": "4.8.0",
}
PROTOCOL = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 1750,
    "enable_thinking": False,
    "concurrency": 8,
    "return_token_ids": True,
    "prefix_caching": "must be disabled on server",
}
SUFFIX = "\nSolve step by step. End with #### followed by the final numeric answer."
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def uuid(value):
    return value.removeprefix("GPU-").lower()


def log_integrity(path, record_path, recorded_sha):
    # Share the exact closeout-message allowlist; quality outcomes stay independent.
    validator_path = Path(__file__).resolve().parent / "audit_tp2.py"
    spec = importlib.util.spec_from_file_location("tp2_log_validator", validator_path)
    require(spec is not None and spec.loader is not None, "Log validator import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = module.Audit()
    report = module.verify_log(path, record_path, recorded_sha, checks, path.name)
    require(not checks.errors, "Log closeout validation: " + "; ".join(checks.errors))
    return {
        **report,
        "validation_notes": checks.notes,
        "validator_sha256": sha(validator_path),
    }


def answer_number(text):
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    numbers = NUMBER.findall(text)
    if not numbers:
        return None
    try:
        value = Decimal(numbers[-1].replace(",", ""))
        return format(value.normalize(), "f") if value else "0"
    except InvalidOperation:
        return None


def metric(record, label):
    require(record.get("finite", True) is True, label + "/finite")
    require(record["pointwise_over_threshold"] == 0, label + "/pointwise bound")
    require(0 <= record["relative_l2"] < 0.01, label + "/relative L2 bound")
    require(record["max_abs"] >= 0, label + "/absolute error")
    for key, value in record.items():
        require(
            type(value) in (bool, int, float) and math.isfinite(value),
            label + "/numeric " + key,
        )
        if key.endswith("_step"):
            require(type(value) is int and 1 <= value <= 128, label + "/step " + key)


def correctness(results, source_files, manifest_sha, script, harness):
    folder = results / "correctness"
    combined_path = folder / "distributed-correctness.json"
    combined = read(combined_path)
    ranks = [read(folder / f"rank-{rank}.json") for rank in range(2)]
    require(combined["ranks"] == ranks, "External and aggregate rank records match")
    require(
        combined["status"] == "passed"
        and combined["world_size"] == 2
        and combined["distributed_backend"] == "nccl",
        "NCCL aggregate status",
    )
    require(
        combined["source_head"] == HEAD
        and combined["global_shape"] == GLOBAL
        and combined["per_rank_shape"] == LOCAL,
        "Aggregate source/shapes",
    )
    require(combined["passed_rank_cases"] == 12, "Aggregate 12 rank cases")
    devices, seeds, summaries = [], [], []
    for rank, row in enumerate(ranks):
        label = f"rank{rank}"
        require(
            row["status"] == "passed"
            and row["passed_cases"] == 6
            and len(row["cases"]) == 6,
            label + "/six cases",
        )
        require(
            (row["rank"], row["local_rank"], row["world_size"]) == (rank, rank, 2),
            label + "/rank identity",
        )
        require(
            row["source_head"] == HEAD and row["source_files"] == source_files,
            label + "/five source hashes",
        )
        require(
            len(source_files) == 5 and row["source_manifest_sha256"] == manifest_sha,
            label + "/source manifest",
        )
        require(
            row["script_sha256"] == sha(script)
            and row["harness_sha256"] == sha(harness),
            label + "/helper hashes",
        )
        require(row["runtime_versions"] == VERSIONS, label + "/runtime versions")
        require(
            row["global_shape"] == GLOBAL and row["local_shape"] == LOCAL,
            label + "/head shapes",
        )
        require(
            row["logical_head_shard"]
            == {
                "key_head_range": [rank * 8, (rank + 1) * 8],
                "value_head_range": [rank * 24, (rank + 1) * 24],
            },
            label + "/logical shard",
        )
        require(
            row["steps_per_case"] == 128
            and row["state_dtype"] == "float32"
            and row["input_dtype"] == "bfloat16",
            label + "/dtype/steps",
        )
        require(
            row["tolerances"]["relative_l2"] == 0.01
            and row["tolerances"]["float32_state"] == {"atol": 0.01, "rtol": 0.01},
            label + "/unchanged tolerance",
        )
        require(row["nccl_collective_sum"] == 3, label + "/actual NCCL allreduce")
        device = row["device"]
        require(
            device["logical_index"] == rank
            and device["name"] == "NVIDIA L20"
            and device["capability"] == [8, 9],
            label + "/real device",
        )
        devices.append(uuid(device["uuid"]))
        summary = {"rank": rank, "device": device, "cases": []}
        for i, (item, expected) in enumerate(zip(row["cases"], CASES, strict=True)):
            batch, index_dtype, padded = expected
            at = label + "/" + item["case"]
            require(
                item["status"] == "passed"
                and (item["batch"], item["index_dtype"], item["padding"]) == expected,
                at + "/case matrix",
            )
            require(item["seed"] == 20261008 + rank * 1000 + i, at + "/rank seed")
            seeds.append(item["seed"])
            require(
                item["local_A_log_shape"] == item["local_dt_bias_shape"] == [24],
                at + "/bias shape",
            )
            require(
                item["raw_output_shape"] == [batch, 1, 24, 128]
                and item["state_shape"] == [batch + 5, 24, 128, 128],
                at + "/IO shape",
            )
            require(
                item["state_strides"] == [393472, 16384, 128, 1]
                and item["qkv_strides"] == [8192, 1],
                at + "/packed strides",
            )
            require(
                item["gate_pointer_offset_bytes"] == 48
                and item["a_pointer_alignment_16_bytes"] is True,
                at + "/alignment",
            )
            indices = list(range(batch, 0, -1))
            if padded:
                indices[-2:] = [0, -1]
            require(item["initial_index_map"] == indices, at + "/padding indices")
            check = item["correctness"]
            require(check["steps"] == 128, at + "/trajectory length")
            for key in (
                "graph_replay",
                "metadata_remap",
                "inactive_slots_bitwise_unchanged",
                "cache_page_padding_bitwise_unchanged",
            ):
                require(check[key] is True, at + "/" + key)
            require(
                set(check["one_step"]) == {"triton", "flashinfer"},
                at + "/both wrappers",
            )
            for backend in ("triton", "flashinfer"):
                for name in ("raw_output", "output", "state"):
                    metric(
                        check["one_step"][backend][name],
                        at + "/one_step/" + backend + "/" + name,
                    )
            for name in ("raw_output", "output", "state"):
                metric(check["trajectory"][name], at + "/trajectory/" + name)
            require(
                item["rank_input_sample"]
                != ranks[1 - rank]["cases"][i]["rank_input_sample"],
                at + "/independent actual input",
            )
            summary["cases"].append(
                {
                    "case": item["case"],
                    "seed": item["seed"],
                    "status": "passed",
                    "one_step": check["one_step"],
                    "trajectory": check["trajectory"],
                    "inactive_state_and_padding_exact": True,
                    "graph_steps": 128,
                }
            )
        summaries.append(summary)
    require(len(set(devices)) == 2 and all(devices), "Distinct actual CUDA UUIDs")
    require(len(set(seeds)) == 12, "Distinct deterministic rank/case seeds")
    return {
        "status": "passed",
        "passed_rank_cases": 12,
        "world_size": 2,
        "source_head": HEAD,
        "source_files": source_files,
        "helper_sha256": sha(script),
        "harness_sha256": sha(harness),
        "ranks": summaries,
        "device_uuids": devices,
        "distributed_backend": "nccl",
        "nccl_sum": 3,
        "global_shape": GLOBAL,
        "per_rank_shape": LOCAL,
        "raw_sha256": {
            path.name: sha(path)
            for path in (combined_path, folder / "rank-0.json", folder / "rank-1.json")
        },
        "scope": "Recorded local-wrapper correctness; full model validated separately",
    }


def quality(report, dataset, arm):
    require(report["arm"] == arm and report["model"] == "qwen-fp8", arm + "/identity")
    require(report["protocol"] == PROTOCOL, arm + "/frozen quality protocol")
    indices = sorted(random.Random(42).sample(range(1319), 100))
    fixture = report["fixture"]
    require(
        fixture["indices"] == indices
        and fixture["samples"] == 100
        and fixture["selection_seed"] == 42
        and fixture["dataset_rows"] == 1319
        and fixture["dataset_sha256"] == DATASET_SHA
        and fixture["kind"] == "gsm8k_test_zero_shot",
        arm + "/dataset selection",
    )
    rows = report["examples"]
    require(
        len(rows) == 100 and [row["index"] for row in rows] == indices,
        arm + "/100 unique questions",
    )
    frozen_rows, computed = [], []
    for row in rows:
        at = arm + f"/question{row['index']}"
        item = dataset[row["index"]]
        require(
            row["question"] == item["question"] and row["gold_text"] == item["answer"],
            at + "/actual dataset question/gold",
        )
        require(
            row["gold"] == answer_number(item["answer"]),
            at + "/gold parsed independently",
        )
        require(row["prompt"] == row["question"] + SUFFIX, at + "/unchanged prompt")
        frozen_rows.append(
            {key: row[key] for key in ("index", "question", "gold", "gold_text")}
        )
        tokens, usage = row["token_ids"], row["usage"]
        require(
            0 < len(tokens) <= 1750
            and all(type(token) is int and token >= 0 for token in tokens),
            at + "/token IDs",
        )
        require(row["token_ids_sha256"] == digest(tokens), at + "/token hash")
        require(
            usage["completion_tokens"] == len(tokens)
            and usage["prompt_tokens"] > 0
            and usage["total_tokens"] == usage["prompt_tokens"] + len(tokens),
            at + "/usage",
        )
        require(isinstance(row["text"], str) and row["text"], at + "/nonempty output")
        prediction = answer_number(row["text"])
        raw_correct = prediction == row["gold"]
        truncated = row["finish_reason"] == "length"
        marker = "####" in row["text"]
        completed = row["finish_reason"] == "stop" and not truncated
        require(
            row["predicted"] == prediction
            and row["correct"] is raw_correct
            and row["truncated"] is truncated
            and row["has_answer_marker"] is marker,
            at + "/recomputed parser/flags",
        )
        computed.append(
            {
                "index": row["index"],
                "gold": row["gold"],
                "predicted": prediction,
                "raw_correct": raw_correct,
                "completed_correct": raw_correct and completed,
                "strict_correct": raw_correct and completed and marker,
                "finish_reason": row["finish_reason"],
                "truncated": truncated,
                "unparsed": prediction is None,
                "missing_marker": not marker,
                "token_count": len(tokens),
                "token_ids_sha256": row["token_ids_sha256"],
            }
        )
    require(
        digest(frozen_rows) == report["question_fixture_sha256"], arm + "/fixture hash"
    )
    summary = {
        "samples": 100,
        "correct": sum(row["raw_correct"] for row in computed),
        "truncated": sum(row["truncated"] for row in computed),
        "unparsed": sum(row["unparsed"] for row in computed),
        "missing_answer_marker": sum(row["missing_marker"] for row in computed),
        "http_failures": 0,
    }
    require(report["summary"] == summary, arm + "/recomputed summary")
    summary.update(
        completed_correct=sum(row["completed_correct"] for row in computed),
        strict_correct=sum(row["strict_correct"] for row in computed),
        unexpected_finish=sum(
            row["finish_reason"] not in ("stop", "length") for row in computed
        ),
    )
    return summary, computed


def launch(results, report, source_files, model_sha, device_uuids, args):
    arm = report["arm"]
    path = results / f"serve-{arm}-compat.json"
    record = read(path)
    log_path = path.with_suffix(".log")
    require(
        record["status"] == "completed" and "error" not in record,
        arm + "/completed server/quality run",
    )
    require(
        record["phase"] == "preflight" and record["backend"] == arm,
        arm + "/preflight backend",
    )
    require(
        record["tensor_parallel_size"] == 2 and record["per_rank_gdn_shape"] == LOCAL,
        arm + "/real TP2 config",
    )
    require(
        record["source_head"] == HEAD and record["source_files"] == source_files,
        arm + "/head/five source hashes",
    )
    require(record["model_config_sha256"] == model_sha, arm + "/actual model config")
    require(
        record["quality_sha256"] == sha(results / f"quality-{arm}.json"),
        arm + "/quality file hash",
    )
    require(record["supervisor_sha256"] == sha(args.supervisor), arm + "/supervisor")
    log_proof = log_integrity(log_path, path, record["server_log_sha256"])
    require(
        report["harness_sha256"]
        == record["quality_helper_sha256"]
        == sha(args.quality_harness),
        arm + "/frozen quality helper",
    )
    require(
        {uuid(value) for value in record["gpu_uuids"]} == set(device_uuids)
        and len(record["gpu_uuids"]) == 2,
        arm + "/same physical GPUs",
    )
    embedded = report["server_evidence"]
    require(embedded["status"] == "ready", arm + "/ready embedded snapshot")
    for key in (
        "arm",
        "backend",
        "phase",
        "command",
        "source_head",
        "source_files",
        "tensor_parallel_size",
        "per_rank_gdn_shape",
        "gpu_uuids",
        "gpu_inventory",
        "model_config_sha256",
        "runtime",
        "state_dtype",
        "quality_command",
        "quality_helper_sha256",
        "kv_capacity_tokens",
        "supervisor_sha256",
    ):
        require(embedded[key] == record[key], arm + "/embedded " + key)
    command = record["command"]
    require(
        command[command.index("--tensor-parallel-size") + 1] == "2"
        and "--no-enable-prefix-caching" in command,
        arm + "/CLI TP/prefix cache",
    )
    require(
        command[command.index("--mamba-ssm-cache-dtype") + 1] == "float32"
        and command[command.index("--dtype") + 1] == "bfloat16",
        arm + "/CLI dtypes",
    )
    normalized = command.copy()
    pos = normalized.index("--kernel-config") + 1
    kernel = json.loads(normalized[pos])
    require(
        kernel == {"gdn_decode_backend": arm, "linear_backend": "marlin"},
        arm + "/kernel config",
    )
    normalized[pos] = json.dumps(
        {**kernel, "gdn_decode_backend": "<backend>"}, sort_keys=True
    )
    runtime = record["runtime"]
    require(
        runtime["declared_source_head"] == HEAD
        and runtime["record_integrity_failures"] == [],
        arm + "/runtime provenance",
    )
    runtime_files = runtime["source_files"]
    require(
        {name for name in source_files if name.startswith("vllm/")}
        <= set(runtime_files),
        arm + "/runtime production source coverage",
    )
    for name, observed in runtime_files.items():
        require(
            name in source_files and observed["sha256"] == source_files[name],
            arm + "/runtime " + name,
        )
    require(runtime["torch"]["gpu_count"] == 2, arm + "/dual-device probe")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    capacities = {
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    }
    require(capacities == {record["kv_capacity_tokens"]}, arm + "/actual KV capacity")
    count = len(re.findall(r'POST /v1/chat/completions HTTP/1\.1" 200 OK', log))
    require(count == 100, arm + f"/HTTP 200 quality completions {count}/100")
    return (
        record,
        normalized,
        {
            "raw_quality_sha256": record["quality_sha256"],
            "supervisor_record_sha256": sha(path),
            "server_log_sha256": sha(log_path),
            "log_integrity": log_proof,
            "logged_quality_http_200": count,
            "runtime_source_files_verified": sorted(runtime_files),
        },
    )


def comparison(a, b, ca, cb):
    flips = {}
    for field in ("raw_correct", "completed_correct", "strict_correct"):
        flips[field] = {
            "triton_to_flashinfer_losses": [
                x["index"]
                for x, y in zip(ca, cb, strict=True)
                if x[field] and not y[field]
            ],
            "triton_to_flashinfer_gains": [
                x["index"]
                for x, y in zip(ca, cb, strict=True)
                if not x[field] and y[field]
            ],
        }
    issues, changes = [], []
    for x, y, sx, sy in zip(a["examples"], b["examples"], ca, cb, strict=True):
        require(
            x["index"] == y["index"]
            and x["question"] == y["question"]
            and x["gold"] == y["gold"],
            "Paired question/gold identity",
        )
        if x["token_ids"] != y["token_ids"]:
            first = next(
                (
                    i
                    for i, (u, v) in enumerate(zip(x["token_ids"], y["token_ids"]))
                    if u != v
                ),
                min(len(x["token_ids"]), len(y["token_ids"])),
            )
            changes.append(
                {
                    "index": x["index"],
                    "first_token_difference_zero_based": first,
                    "triton": sx,
                    "flashinfer": sy,
                }
            )
        if sx["predicted"] != sy["predicted"] or any(
            not row["strict_correct"]
            or row["unparsed"]
            or row["truncated"]
            or row["missing_marker"]
            for row in (sx, sy)
        ):
            issues.append(
                {
                    "index": x["index"],
                    "question": x["question"],
                    "gold": x["gold"],
                    "triton": {**sx, "text": x["text"]},
                    "flashinfer": {**sy, "text": y["text"]},
                }
            )
    return {
        "paired_questions": 100,
        "exact_token_sequences": sum(
            x["token_ids"] == y["token_ids"]
            for x, y in zip(a["examples"], b["examples"], strict=True)
        ),
        "matching_parsed_answers": sum(
            x["predicted"] is not None and x["predicted"] == y["predicted"]
            for x, y in zip(ca, cb, strict=True)
        ),
        "flips": flips,
        "changed_token_outputs": changes,
        "all_changed_answer_or_exceptional_outcomes": issues,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "raw")
    parser.add_argument(
        "--source-manifest", type=Path, default=root / "performance-source.json"
    )
    parser.add_argument(
        "--correctness-helper", type=Path, default=root / "tp2_gdn_correctness.py"
    )
    parser.add_argument(
        "--correctness-harness",
        type=Path,
        default=root.parent / "performance-followup-20261008/benchmark_current_gdn.py",
    )
    parser.add_argument(
        "--quality-harness",
        type=Path,
        default=root / "quality_http.py",
    )
    parser.add_argument("--supervisor", type=Path, default=root / "tp2_supervisor.py")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "gsm8k-test.jsonl",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=root / "model-config.json",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "audit-quality-correctness.json"
    )
    args = parser.parse_args()
    manifest = read(args.source_manifest)
    require(manifest["source_head"] == HEAD, "Frozen source head")
    source_files = {name: row["sha256"] for name, row in manifest["files"].items()}
    kernels = correctness(
        args.results,
        source_files,
        sha(args.source_manifest),
        args.correctness_helper,
        args.correctness_harness,
    )
    require(sha(args.dataset) == DATASET_SHA, "Full cached dataset SHA256")
    dataset = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(dataset) == 1319, "Full dataset has 1319 rows")
    model = read(args.model_config)["text_config"]
    require(
        (
            model["linear_num_key_heads"],
            model["linear_num_value_heads"],
            model["linear_key_head_dim"],
            model["linear_value_head_dim"],
        )
        == (16, 48, 128, 128),
        "Model matches global TP2 dimensions",
    )
    reports, computed, launches, commands, quality_results = {}, {}, {}, {}, {}
    for arm in ("triton", "flashinfer"):
        reports[arm] = read(args.results / f"quality-{arm}.json")
        summary, computed[arm] = quality(reports[arm], dataset, arm)
        record, commands[arm], hashes = launch(
            args.results,
            reports[arm],
            source_files,
            sha(args.model_config),
            kernels["device_uuids"],
            args,
        )
        launches[arm] = record
        quality_results[arm] = {
            "summary": summary,
            "integrity": hashes,
            "tensor_parallel_size": record["tensor_parallel_size"],
            "per_rank_gdn_shape": record["per_rank_gdn_shape"],
            "source_head": record["source_head"],
            "source_files": record["source_files"],
            "model_config_sha256": record["model_config_sha256"],
            "quality_harness_sha256": reports[arm]["harness_sha256"],
            "runtime_snapshot_digest": digest(record["runtime"]),
            "runtime_provenance": record["runtime_provenance"],
            "gpu_uuids": record["gpu_uuids"],
            "kv_capacity_tokens": record["kv_capacity_tokens"],
        }
    require(commands["triton"] == commands["flashinfer"], "Matched TP2 commands")
    for key in (
        "runtime",
        "source_files",
        "model_config_sha256",
        "gpu_uuids",
        "kv_capacity_tokens",
        "gpu_blocks",
    ):
        require(
            launches["triton"][key] == launches["flashinfer"][key],
            "Matched launches " + key,
        )
    for key in ("fixture", "question_fixture_sha256", "harness_sha256", "protocol"):
        require(
            reports["triton"][key] == reports["flashinfer"][key],
            "Matched quality " + key,
        )
    require(
        (args.results / "preflight.exit").read_text(encoding="utf-8").strip() == "0",
        "Preflight exit",
    )
    result = {
        "status": "passed",
        "source_head": HEAD,
        "source_files": source_files,
        "source_manifest_sha256": sha(args.source_manifest),
        "audit_script_sha256": sha(Path(__file__)),
        "log_closeout_notes": [
            note
            for row in quality_results.values()
            for note in row["integrity"]["log_integrity"]["validation_notes"]
        ],
        "correctness": kernels,
        "model_quality": {
            "backends": quality_results,
            "protocol": PROTOCOL,
            "dataset_sha256": DATASET_SHA,
            "question_fixture_sha256": reports["triton"]["question_fixture_sha256"],
            "comparison": comparison(
                reports["triton"],
                reports["flashinfer"],
                computed["triton"],
                computed["flashinfer"],
            ),
        },
        "limitations": [
            "One quality run per backend on 100 fixed questions",
            "Equal scores do not establish statistical accuracy equivalence",
            "Reused native binary; not a full CUDA13 source build",
            "No performance conclusion from correctness/quality records",
            (
                "The original Triton log hash binds a newline prefix; "
                "a separately verified closeout manifest binds its late suffix"
            ),
        ],
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed_rank_cases": 12,
                "quality": {
                    arm: row["summary"] for arm, row in quality_results.items()
                },
                "flips": result["model_quality"]["comparison"]["flips"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
