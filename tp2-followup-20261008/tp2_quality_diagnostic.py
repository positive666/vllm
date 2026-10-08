"""Targeted TP2 diagnosis with an expanded budget; never a replacement eval.

Preserve the original 100-question screen. Repeat its eight-question C8 batch
containing 209 and 255 at C1 and C8 once each, with the same prompts and parser.
The context, output budget and KV capacity all increase; this is not an
isolated single-variable intervention or a new overall accuracy benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import sys
import urllib.parse
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
DATASET_SHA = "ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13"
INDICES = [198, 206, 209, 228, 255, 285, 292, 318]
TARGETS = [209, 255]


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_harness(path):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("frozen_quality_http", path)
    require(spec is not None and spec.loader is not None, "Frozen harness import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flags(row, frozen):
    predicted = frozen.answer_number(row["text"])
    correct = predicted == row["gold"]
    marker = "####" in row["text"]
    truncated = row["finish_reason"] == "length"
    completed = row["finish_reason"] == "stop" and not truncated
    return {
        "predicted": predicted,
        "correct": correct,
        "raw_correct": correct,
        "completed_correct": correct and completed,
        "strict_correct": correct and completed and marker,
        "truncated": truncated,
        "has_answer_marker": marker,
        "unparsed": predicted is None,
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


def plan(args, frozen):
    args.samples = 100
    rows, fixture = frozen.fixture_rows(args)
    require(fixture["dataset_sha256"] == DATASET_SHA, "Frozen dataset hash")
    offset = next(i for i, row in enumerate(rows) if row["index"] == 209)
    start = (offset // 8) * 8
    selected = rows[start : start + 8]
    require([row["index"] for row in selected] == INDICES, "Original C8 batch")
    primary = frozen.read_json(args.primary_quality)
    require(primary["arm"] == args.arm, "Primary backend")
    require(primary["harness_sha256"] == sha(args.harness), "Frozen client hash")
    require(
        primary["question_fixture_sha256"] == frozen.digest(rows),
        "Primary frozen 100-question fixture",
    )
    expected_protocol = {
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1750,
        "enable_thinking": False,
        "concurrency": 8,
        "return_token_ids": True,
        "prefix_caching": "must be disabled on server",
    }
    require(primary["protocol"] == expected_protocol, "Original screen protocol")
    original = primary["examples"][start : start + 8]
    require(
        [
            {key: row[key] for key in ("index", "question", "gold", "gold_text")}
            for row in original
        ]
        == selected,
        "Primary selected question/gold",
    )
    outcomes = [
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
        | flags(row, frozen)
        | {"target": row["index"] in TARGETS}
        for row in original
    ]
    return (
        primary,
        selected,
        {
            "scope": (
                "Targeted expanded-budget batching diagnosis; no overall accuracy claim"
            ),
            "selection_rule": (
                "Original C8 batch containing 209 and 255; six remaining controls"
            ),
            "selected_indices": INDICES,
            "target_indices": TARGETS,
            "control_indices": [index for index in INDICES if index not in TARGETS],
            "full_fixture": fixture,
            "full_question_fixture_sha256": frozen.digest(rows),
            "selected_question_fixture_sha256": frozen.digest(selected),
            "primary_quality_sha256": sha(args.primary_quality),
            "primary_outcomes": outcomes,
            "harness_sha256": sha(args.harness),
            "diagnostic_client_sha256": sha(Path(__file__)),
            "protocol": {
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
                "changed_settings": [
                    "output budget",
                    "maximum model context",
                    "KV capacity",
                ],
            },
        },
    )


def validate_server(server, primary, arm):
    previous = primary["server_evidence"]
    require(server["status"] == "ready", "Diagnostic server readiness")
    require(
        server["tensor_parallel_size"] == 2 and server["source_head"] == HEAD,
        "Same TP2 reviewed head",
    )
    for key in ("source_files", "model_config_sha256", "gpu_uuids"):
        require(server[key] == previous[key], "Same primary " + key)
    require(len(server["source_files"]) == 5, "All five source hashes")
    require(
        server["gpu_blocks"] == 128 and server["kv_capacity_tokens"] >= 32768,
        "Expanded KV capacity",
    )
    command = server["command"]
    for option, expected in (
        ("--tensor-parallel-size", "2"),
        ("--max-model-len", "4096"),
        ("--num-gpu-blocks-override", "128"),
        ("--dtype", "bfloat16"),
        ("--mamba-ssm-cache-dtype", "float32"),
        ("--seed", "42"),
    ):
        require(command[command.index(option) + 1] == expected, "CLI " + option)
    require("--no-enable-prefix-caching" in command, "Prefix cache disabled")
    config = json.loads(command[command.index("--kernel-config") + 1])
    require(
        config == {"gdn_decode_backend": arm, "linear_backend": "marlin"},
        "Same kernel/quantization backends",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--primary-quality", type=Path, required=True)
    parser.add_argument("--server-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--harness", type=Path, default=Path("/reference-artifacts/quality_http.py")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-fp8")
    parser.add_argument("--max-tokens", type=int, choices=(3500,), default=3500)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    frozen = load_harness(args.harness)
    primary, selected, metadata = plan(args, frozen)
    if args.plan_only:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in metadata.items()
                    if key not in ("primary_outcomes", "full_fixture")
                },
                indent=2,
            )
        )
        return
    require(args.server_evidence and args.output, "Server evidence and output paths")
    url = urllib.parse.urlparse(args.base_url)
    require(
        url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1"),
        "Loopback HTTP only",
    )
    require(
        not (url.username or url.password or url.query or url.fragment),
        "No URL credentials/query/fragment",
    )
    progress = args.output.with_suffix(".progress.json")
    require(
        not args.output.exists() and not progress.exists(),
        "Refusing to overwrite diagnostic evidence",
    )
    server = frozen.read_json(args.server_evidence)
    validate_server(server, primary, args.arm)
    report = {
        "arm": args.arm,
        "model": args.model,
        **metadata,
        "server_evidence": server,
        "rounds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for concurrency in (1, 8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            examples = list(pool.map(lambda item: frozen.one(args, item), selected))
        for row in examples:
            row.update(flags(row, frozen))
        round_record = {
            "concurrency": concurrency,
            "repeat": 1,
            "examples": examples,
            "summary": summarize(examples),
        }
        report["rounds"].append(round_record)
        progress.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"arm": args.arm, "concurrency": concurrency, **round_record["summary"]}
            ),
            flush=True,
        )
    require(
        sum(len(row["examples"]) for row in report["rounds"]) == 16,
        "Sixteen diagnostic requests",
    )
    frozen.exclusive_json(args.output, report)
    print(
        json.dumps({"arm": args.arm, "requests": 16, "scope": metadata["scope"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
