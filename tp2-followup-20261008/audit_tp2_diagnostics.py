"""Audit expanded TP2 diagnostic records without replacing the original screen.

Read raw records only. Recompute parser and token checks independently, and
retain wrong answers, truncations and all target flips as observed findings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
DATASET_SHA = "ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13"
INDICES = [198, 206, 209, 228, 255, 285, 292, 318]
TARGETS = [209, 255]
LOCAL = {"H": 8, "HV": 24, "K": 128, "V": 128}
GLOBAL = {"H": 16, "HV": 48, "K": 128, "V": 128}
SUFFIX = "\nSolve step by step. End with #### followed by the final numeric answer."
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
CHANGES = ["output budget", "maximum model context", "KV capacity"]
PRIMARY_PROTOCOL = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 1750,
    "enable_thinking": False,
    "concurrency": 8,
    "return_token_ids": True,
    "prefix_caching": "must be disabled on server",
}
PROTOCOL = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 3500,
    "enable_thinking": False,
    "return_token_ids": True,
    "prefix_caching": "must be disabled on server",
    "round_order": [
        {"concurrency": 1, "repeat": 1},
        {"concurrency": 8, "repeat": 1},
    ],
    "primary_max_tokens": 1750,
    "model_max_len": 4096,
    "gpu_blocks": 128,
    "minimum_actual_kv_tokens": 32768,
    "requests_per_backend": 16,
    "changed_settings": CHANGES,
}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def outcome(row, dataset, budget, label, diagnostic=False):
    index = row["index"]
    item = dataset[index]
    gold = answer_number(item["answer"])
    require(
        row["question"] == item["question"]
        and row["gold_text"] == item["answer"]
        and row["gold"] == gold
        and row["prompt"] == item["question"] + SUFFIX,
        label + "/dataset and prompt",
    )
    tokens, usage = row["token_ids"], row["usage"]
    require(
        0 < len(tokens) <= budget
        and all(type(token) is int and token >= 0 for token in tokens),
        label + "/token IDs",
    )
    require(row["token_ids_sha256"] == digest(tokens), label + "/token hash")
    require(
        type(usage["completion_tokens"]) is int
        and usage["completion_tokens"] == len(tokens)
        and type(usage["prompt_tokens"]) is int
        and usage["prompt_tokens"] > 0
        and usage["total_tokens"] == usage["prompt_tokens"] + len(tokens),
        label + "/usage",
    )
    require(isinstance(row["text"], str) and row["text"], label + "/output text")
    predicted = answer_number(row["text"])
    correct = predicted == gold
    truncated = row["finish_reason"] == "length"
    marker = "####" in row["text"]
    completed = row["finish_reason"] == "stop"
    fields = {
        "predicted": predicted,
        "correct": correct,
        "truncated": truncated,
        "has_answer_marker": marker,
    }
    if diagnostic:
        fields.update(
            raw_correct=correct,
            completed_correct=correct and completed,
            strict_correct=correct and completed and marker,
            unparsed=predicted is None,
        )
    for key, value in fields.items():
        require(
            row[key] == value and type(row[key]) is type(value),
            label + "/independent " + key,
        )
    return {
        "index": index,
        "gold": gold,
        "predicted": predicted,
        "raw_correct": correct,
        "completed_correct": correct and completed,
        "strict_correct": correct and completed and marker,
        "finish_reason": row["finish_reason"],
        "truncated": truncated,
        "has_answer_marker": marker,
        "unparsed": predicted is None,
        "token_count": len(tokens),
        "token_ids_sha256": digest(tokens),
    }


def summarize(rows):
    return {
        "requests": len(rows),
        "raw_correct": sum(row["raw_correct"] for row in rows),
        "strict_correct": sum(row["strict_correct"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "unparsed": sum(row["unparsed"] for row in rows),
        "missing_marker": sum(not row["has_answer_marker"] for row in rows),
    }


def log_integrity(path, record_path, recorded_sha):
    validator = Path(__file__).resolve().parent / "audit_tp2.py"
    spec = importlib.util.spec_from_file_location("diagnostic_log_validator", validator)
    require(spec is not None and spec.loader is not None, "Log validator import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = module.Audit()
    report = module.verify_log(path, record_path, recorded_sha, checks, path.name)
    require(not checks.errors, "Log verification: " + "; ".join(checks.errors))
    return {
        **report,
        "validation_notes": checks.notes,
        "validator_sha256": sha(validator),
    }


def normalized_command(command):
    normalized = command.copy()
    pos = normalized.index("--kernel-config") + 1
    config = json.loads(normalized[pos])
    normalized[pos] = json.dumps(
        {**config, "gdn_decode_backend": "<backend>"}, sort_keys=True
    )
    return normalized


def launch(args, arm, report, primary, source_files):
    folder = args.results / "diagnostics"
    path = folder / f"serve-{arm}.json"
    record = read(path)
    raw_path = folder / f"{arm}.json"
    previous = primary["server_evidence"]
    require(
        record["status"] == "completed"
        and "error" not in record
        and record["phase"] == "expanded-budget-diagnostic"
        and record["backend"] == arm,
        arm + "/complete diagnostic server",
    )
    require(
        record["source_head"] == HEAD
        and record["source_files"] == source_files
        and record["source_manifest_sha256"] == sha(args.source_manifest),
        arm + "/frozen source",
    )
    require(
        record["diagnostic_sha256"] == sha(raw_path)
        and record["diagnostic_helper_sha256"] == sha(args.client)
        and record["supervisor_sha256"] == sha(args.supervisor)
        and report["diagnostic_client_sha256"] == sha(args.client),
        arm + "/client and supervisor bytes",
    )
    require(
        record["tensor_parallel_size"] == 2
        and record["per_rank_gdn_shape"] == LOCAL
        and record["state_dtype"] == "float32",
        arm + "/TP2 local shapes and state",
    )
    require(
        record["gpu_blocks"] == 128
        and record["max_model_len"] == 4096
        and record["kv_capacity_tokens"] >= 32768,
        arm + "/expanded model and KV capacity",
    )
    require(
        record["model_config_sha256"] == sha(args.model_config)
        and len(record["gpu_uuids"]) == len(set(record["gpu_uuids"])) == 2,
        arm + "/actual model and two devices",
    )
    for key in ("source_head", "source_files", "model_config_sha256", "gpu_uuids"):
        require(record[key] == previous[key], arm + "/same primary " + key)
    require(record["runtime"] == previous["runtime"], arm + "/same runtime probe")
    runtime = record["runtime"]
    require(
        runtime["declared_source_head"] == HEAD
        and runtime["record_integrity_failures"] == []
        and runtime["torch"]["gpu_count"] == 2,
        arm + "/runtime provenance",
    )
    require(
        {name for name in source_files if name.startswith("vllm/")}
        <= set(runtime["source_files"]),
        arm + "/runtime production coverage",
    )
    for name, observed in runtime["source_files"].items():
        require(
            name in source_files and observed["sha256"] == source_files[name],
            arm + "/runtime " + name,
        )
    embedded = report["server_evidence"]
    require(embedded["status"] == "ready", arm + "/ready embedded snapshot")
    for key, value in embedded.items():
        if key != "status":
            require(record[key] == value, arm + "/embedded " + key)
    command = record["command"]
    for option, expected in (
        ("--tensor-parallel-size", "2"),
        ("--max-model-len", "4096"),
        ("--num-gpu-blocks-override", "128"),
        ("--dtype", "bfloat16"),
        ("--mamba-ssm-cache-dtype", "float32"),
        ("--seed", "42"),
    ):
        require(command[command.index(option) + 1] == expected, arm + "/" + option)
    require("--no-enable-prefix-caching" in command, arm + "/prefix caching off")
    require(
        json.loads(command[command.index("--kernel-config") + 1])
        == {"gdn_decode_backend": arm, "linear_backend": "marlin"},
        arm + "/backend choices",
    )
    original_command = previous["command"].copy()
    for option, value in (
        ("--max-model-len", "4096"),
        ("--num-gpu-blocks-override", "128"),
    ):
        original_command[original_command.index(option) + 1] = value
    require(command == original_command, arm + "/only stated server CLI changes")
    client_command = record["client_command"]
    for option, expected in (("--arm", arm), ("--max-tokens", "3500")):
        require(
            client_command[client_command.index(option) + 1] == expected,
            arm + "/client " + option,
        )
    log_path = path.with_suffix(".log")
    proof = log_integrity(log_path, path, record["server_log_sha256"])
    log = log_path.read_text(encoding="utf-8")
    capacities = {
        int(value.replace(",", ""))
        for value in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    }
    require(capacities == {record["kv_capacity_tokens"]}, arm + "/logged actual KV")
    statuses = re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', log)
    require(statuses == ["200"] * 16, arm + "/exact 16 HTTP 200 requests")
    return record, {
        "raw_client_sha256": sha(raw_path),
        "server_record_sha256": sha(path),
        "server_log_sha256": sha(log_path),
        "primary_quality_sha256": sha(args.results / f"quality-{arm}.json"),
        "client_helper_sha256": sha(args.client),
        "supervisor_sha256": sha(args.supervisor),
        "log_integrity": proof,
        "logged_diagnostic_http_200": len(statuses),
    }


def backend(args, arm, dataset, source_files):
    report = read(args.results / "diagnostics" / f"{arm}.json")
    primary_path = args.results / f"quality-{arm}.json"
    primary = read(primary_path)
    require(
        report["arm"] == primary["arm"] == arm
        and report["model"] == primary["model"] == "qwen-fp8",
        arm + "/model identity",
    )
    require(
        report["protocol"] == PROTOCOL
        and primary["protocol"] == PRIMARY_PROTOCOL
        and report["harness_sha256"] == primary["harness_sha256"] == sha(args.harness),
        arm + "/frozen client and protocols",
    )
    indices = sorted(random.Random(42).sample(range(1319), 100))
    selected = indices[16:24]
    require(selected == INDICES, "Original C8 selected batch")
    fixture = primary["fixture"]
    require(
        fixture["indices"] == indices
        and fixture["samples"] == 100
        and fixture["selection_seed"] == 42
        and fixture["dataset_rows"] == 1319
        and fixture["dataset_sha256"] == DATASET_SHA
        and fixture["kind"] == "gsm8k_test_zero_shot",
        arm + "/full original fixture",
    )
    frozen_rows = [
        {
            "index": index,
            "question": dataset[index]["question"],
            "gold": answer_number(dataset[index]["answer"]),
            "gold_text": dataset[index]["answer"],
        }
        for index in indices
    ]
    require(
        primary["question_fixture_sha256"]
        == report["full_question_fixture_sha256"]
        == digest(frozen_rows)
        and report["selected_question_fixture_sha256"] == digest(frozen_rows[16:24])
        and report["full_fixture"] == fixture
        and report["selected_indices"] == INDICES
        and report["target_indices"] == TARGETS
        and report["control_indices"] == [i for i in INDICES if i not in TARGETS]
        and report["primary_quality_sha256"] == sha(primary_path),
        arm + "/frozen selection and primary bytes",
    )
    primary_rows = primary["examples"]
    require(
        len(primary_rows) == 100 and [row["index"] for row in primary_rows] == indices,
        arm + "/complete primary records",
    )
    original = [
        outcome(row, dataset, 1750, arm + f"/original{row['index']}")
        for row in primary_rows
    ]
    primary_outcomes = report["primary_outcomes"]
    require(len(primary_outcomes) == 8, arm + "/eight original outcomes")
    for row, stored, computed in zip(
        primary_rows[16:24], primary_outcomes, original[16:24], strict=True
    ):
        expected = (
            {
                key: row[key]
                for key in (
                    "index",
                    "gold",
                    "finish_reason",
                    "text",
                    "token_ids_sha256",
                    "usage",
                )
            }
            | {
                key: computed[key]
                for key in (
                    "predicted",
                    "raw_correct",
                    "completed_correct",
                    "strict_correct",
                    "truncated",
                    "has_answer_marker",
                    "unparsed",
                )
            }
            | {"correct": computed["raw_correct"], "target": row["index"] in TARGETS}
        )
        require(stored == expected, arm + "/independent original outcome")
    rounds = report["rounds"]
    require(
        [(row["concurrency"], row["repeat"]) for row in rounds] == [(1, 1), (8, 1)],
        arm + "/C1 then C8 once",
    )
    outputs = []
    for group in rounds:
        concurrency = group["concurrency"]
        rows = group["examples"]
        require(
            len(rows) == 8 and [row["index"] for row in rows] == INDICES,
            arm + f"/C{concurrency} full question matrix",
        )
        calculated = [
            outcome(row, dataset, 3500, arm + f"/C{concurrency}/{row['index']}", True)
            for row in rows
        ]
        summary = summarize(calculated)
        require(group["summary"] == summary, arm + "/independent round summary")
        outputs.append(
            {
                "concurrency": concurrency,
                "repeat": 1,
                "summary": summary,
                "target_outcomes": [
                    row for row in calculated if row["index"] in TARGETS
                ],
                "all_outcomes": calculated,
                "exceptional_outcomes": [
                    {**value, "text": row["text"]}
                    for row, value in zip(rows, calculated, strict=True)
                    if not value["strict_correct"]
                    or value["unparsed"]
                    or value["truncated"]
                    or not value["has_answer_marker"]
                ],
            }
        )
    record, integrity = launch(args, arm, report, primary, source_files)
    return (
        report,
        record,
        {
            "integrity": integrity,
            "protocol": PROTOCOL,
            "selected_indices": INDICES,
            "primary_quality_sha256": sha(primary_path),
            "primary_target_outcomes": [
                row for row in original if row["index"] in TARGETS
            ],
            "rounds": outputs,
            "tensor_parallel_size": 2,
            "global_gdn_shape": GLOBAL,
            "per_rank_gdn_shape": LOCAL,
            "source_head": HEAD,
            "source_files": source_files,
            "model_config_sha256": record["model_config_sha256"],
            "gpu_uuids": record["gpu_uuids"],
            "runtime_snapshot_digest": digest(record["runtime"]),
            "kv_capacity_tokens": record["kv_capacity_tokens"],
        },
    )


def pair(a, b, ca, cb):
    raw_a, raw_b = a["examples"], b["examples"]
    require(
        [row["index"] for row in raw_a] == [row["index"] for row in raw_b] == INDICES,
        "Paired matrix",
    )
    return {
        "paired_questions": 8,
        "matching_parsed_answers": sum(
            x["predicted"] is not None and x["predicted"] == y["predicted"]
            for x, y in zip(ca, cb, strict=True)
        ),
        "exact_token_sequences": sum(
            x["token_ids"] == y["token_ids"] for x, y in zip(raw_a, raw_b, strict=True)
        ),
        "flips": {
            field: {
                "losses": [
                    x["index"]
                    for x, y in zip(ca, cb, strict=True)
                    if x[field] and not y[field]
                ],
                "gains": [
                    x["index"]
                    for x, y in zip(ca, cb, strict=True)
                    if not x[field] and y[field]
                ],
            }
            for field in ("raw_correct", "strict_correct")
        },
        "target_pairs": [
            {"index": x["index"], "gold": x["gold"], "triton": x, "flashinfer": y}
            for x, y in zip(ca, cb, strict=True)
            if x["index"] in TARGETS
        ],
        "changed_token_indices": [
            x["index"]
            for x, y in zip(raw_a, raw_b, strict=True)
            if x["token_ids"] != y["token_ids"]
        ],
        "first_token_differences": [
            {
                "index": x["index"],
                "first_difference_zero_based": next(
                    (
                        i
                        for i, (u, v) in enumerate(zip(x["token_ids"], y["token_ids"]))
                        if u != v
                    ),
                    min(len(x["token_ids"]), len(y["token_ids"])),
                ),
                "triton_tokens": len(x["token_ids"]),
                "flashinfer_tokens": len(y["token_ids"]),
            }
            for x, y in zip(raw_a, raw_b, strict=True)
            if x["token_ids"] != y["token_ids"]
        ],
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "raw")
    parser.add_argument("--dataset", type=Path, default=root / "gsm8k-test.jsonl")
    parser.add_argument("--harness", type=Path, default=root / "quality_http.py")
    parser.add_argument(
        "--client", type=Path, default=root / "tp2_quality_diagnostic.py"
    )
    parser.add_argument(
        "--supervisor", type=Path, default=root / "tp2_diagnostic_supervisor.py"
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=root / "performance-source.json"
    )
    parser.add_argument("--model-config", type=Path, default=root / "model-config.json")
    parser.add_argument("--output", type=Path, default=root / "audit-diagnostics.json")
    args = parser.parse_args()
    require(sha(args.dataset) == DATASET_SHA, "Full dataset SHA")
    dataset = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(dataset) == 1319, "1319 original dataset rows")
    model = read(args.model_config)["text_config"]
    require(
        (
            model["linear_num_key_heads"],
            model["linear_num_value_heads"],
            model["linear_key_head_dim"],
            model["linear_value_head_dim"],
        )
        == (16, 48, 128, 128),
        "Actual model global dimensions",
    )
    manifest = read(args.source_manifest)
    require(manifest["source_head"] == HEAD, "Frozen manifest head")
    source_files = {name: row["sha256"] for name, row in manifest["files"].items()}
    require(len(source_files) == 5, "All five frozen source files")
    reports, records, outputs = {}, {}, {}
    for arm in ("triton", "flashinfer"):
        reports[arm], records[arm], outputs[arm] = backend(
            args, arm, dataset, source_files
        )
    require(
        normalized_command(records["triton"]["command"])
        == normalized_command(records["flashinfer"]["command"]),
        "Matched diagnostic server commands",
    )
    for key in (
        "source_head",
        "source_files",
        "model_config_sha256",
        "runtime",
        "gpu_uuids",
        "kv_capacity_tokens",
        "gpu_blocks",
        "max_model_len",
    ):
        require(
            records["triton"][key] == records["flashinfer"][key],
            "Matched diagnostic " + key,
        )
    require(
        (args.results / "diagnostics/run.exit").read_text(encoding="utf-8").strip()
        == "0",
        "Diagnostic wrapper exit",
    )
    paired = {
        f"C{concurrency}": pair(
            reports["triton"]["rounds"][i],
            reports["flashinfer"]["rounds"][i],
            outputs["triton"]["rounds"][i]["all_outcomes"],
            outputs["flashinfer"]["rounds"][i]["all_outcomes"],
        )
        for i, concurrency in enumerate((1, 8))
    }
    result = {
        "status": "passed",
        "meaning_of_pass": (
            "Raw record, parser, protocol and provenance integrity checks passed; "
            "this does not mean all answers are correct or accuracy is equivalent"
        ),
        "scope": (
            "Targeted expanded-budget TP2 diagnostic; original 100 screen retained"
        ),
        "source_head": HEAD,
        "source_files": source_files,
        "source_manifest_sha256": sha(args.source_manifest),
        "audit_script_sha256": sha(Path(__file__)),
        "dataset_sha256": DATASET_SHA,
        "changed_settings": CHANGES,
        "backends": outputs,
        "paired": paired,
        "notes": [
            note
            for row in outputs.values()
            for note in row["integrity"]["log_integrity"]["validation_notes"]
        ],
        "limitations": [
            "Eight selected questions and one run per concurrency cannot estimate "
            "overall accuracy or establish backend equivalence",
            "Output budget, model context and KV capacity change together",
            "Wrong, unparsed, missing-marker and truncated responses remain in results",
            "The original 100-question scores and truncations remain unchanged",
            "Target-answer instability and any continued truncation are unresolved "
            "observations; successful HTTP requests do not establish "
            "quality correctness",
            "Shared log-closeout verifier checks bytes only; quality scores are "
            "independently recomputed here",
        ],
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "meaning_of_pass": result["meaning_of_pass"],
                "summaries": {
                    arm: [row["summary"] for row in output["rounds"]]
                    for arm, output in outputs.items()
                },
                "paired": paired,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
