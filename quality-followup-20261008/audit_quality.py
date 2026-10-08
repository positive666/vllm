"""Independently audit frozen-source model quality evidence without editing it.

The input contract is the existing quality_http.py report and its completed
supervisor record/log. Recompute answer and token evidence, verify the matched
fixture/protocol/source, and preserve every exceptional outcome. This is a
bounded quality screen, never statistical accuracy equivalence or a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
EXPECTED_PROTOCOL = {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 1750,
    "enable_thinking": False,
    "concurrency": 8,
    "return_token_ids": True,
    "prefix_caching": "must be disabled on server",
}
PROMPT_SUFFIX = (
    "\nSolve step by step. End with #### followed by the final numeric answer."
)
SOURCE_HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
DATASET_SHA = "ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def answer_number(text):
    # Reimplemented from the frozen client's observable parsing contract.
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    matches = NUMBER.findall(text)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", ""))
        return format(value.normalize(), "f") if value else "0"
    except InvalidOperation:
        return None


def fixture_rows(report):
    return [
        {key: row[key] for key in ("index", "question", "gold", "gold_text")}
        for row in report["examples"]
    ]


def validate_examples(report, arm):
    examples = report["examples"]
    fixture = report["fixture"]
    require(len(examples) == fixture["samples"] == 100, f"{arm}: sample count")
    indices = [row["index"] for row in examples]
    expected_indices = sorted(random.Random(42).sample(range(1319), 100))
    require(indices == fixture["indices"] == expected_indices, f"{arm}: indices")
    require(fixture["dataset_rows"] == 1319, f"{arm}: dataset size")
    require(fixture["selection_seed"] == 42, f"{arm}: selection seed")
    require(fixture["kind"] == "gsm8k_test_zero_shot", f"{arm}: dataset kind")
    require(fixture["dataset_sha256"] == DATASET_SHA, f"{arm}: dataset hash")
    require(
        digest(fixture_rows(report)) == report["question_fixture_sha256"],
        f"{arm}: recomputed question fixture hash",
    )
    computed = []
    for row in examples:
        label = f"{arm}: question {row['index']}"
        require(answer_number(row["gold_text"]) == row["gold"], label + " gold")
        require(row["prompt"] == row["question"] + PROMPT_SUFFIX, label + " prompt")
        tokens = row["token_ids"]
        require(
            tokens and all(type(token) is int and token >= 0 for token in tokens),
            label + " output tokens",
        )
        require(digest(tokens) == row["token_ids_sha256"], label + " token hash")
        usage = row["usage"]
        require(usage["completion_tokens"] == len(tokens), label + " token usage")
        require(0 < len(tokens) <= 1750, label + " generation budget")
        require(
            usage["total_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"],
            label + " total usage",
        )
        require(usage["prompt_tokens"] > 0, label + " prompt usage")
        text = row["text"]
        require(isinstance(text, str) and bool(text), label + " output text")
        prediction = answer_number(text)
        raw = prediction == row["gold"]
        marker = "####" in text
        truncated = row["finish_reason"] == "length"
        stop = row["finish_reason"] == "stop"
        require(row["predicted"] == prediction, label + " parsed answer")
        require(row["correct"] is raw, label + " raw correctness")
        require(row["has_answer_marker"] is marker, label + " final marker")
        require(row["truncated"] is truncated, label + " truncation")
        require(row["seconds"] >= 0, label + " request elapsed time")
        computed.append(
            {
                "index": row["index"],
                "gold": row["gold"],
                "predicted": prediction,
                "raw_correct": raw,
                "completed_correct": raw and stop and not truncated,
                "strict_correct": raw and stop and not truncated and marker,
                "finish_reason": row["finish_reason"],
                "truncated": truncated,
                "unparsed": prediction is None,
                "missing_marker": not marker,
                "token_count": len(tokens),
                "token_ids_sha256": row["token_ids_sha256"],
            }
        )
    summary = {
        "samples": len(computed),
        "correct": sum(row["raw_correct"] for row in computed),
        "truncated": sum(row["truncated"] for row in computed),
        "unparsed": sum(row["unparsed"] for row in computed),
        "missing_answer_marker": sum(row["missing_marker"] for row in computed),
        "http_failures": 0,
    }
    require(summary == report["summary"], f"{arm}: recomputed summary")
    summary.update(
        completed_correct=sum(row["completed_correct"] for row in computed),
        strict_correct=sum(row["strict_correct"] for row in computed),
        unexpected_finish=sum(
            row["finish_reason"] not in ("stop", "length") for row in computed
        ),
        total_output_tokens=sum(row["token_count"] for row in computed),
    )
    return summary, computed


def normalize_command(command):
    normalized = command.copy()
    pos = normalized.index("--kernel-config") + 1
    config = json.loads(normalized[pos])
    config["gdn_decode_backend"] = "<backend>"
    normalized[pos] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return normalized


def validate_launch(report, evidence, log_path, quality_path, manifest, supervisor):
    arm = report["arm"]
    require(evidence["backend"] == arm, f"{arm}: supervisor backend")
    require(evidence["status"] == "completed", f"{arm}: completed supervisor")
    require("error" not in evidence, f"{arm}: supervisor error")
    require(evidence["repeat"] == 1, f"{arm}: repeat")
    require(evidence["concurrency"] == 8, f"{arm}: supervisor concurrency")
    require(evidence["source_head"] == manifest["source_head"], f"{arm}: source head")
    expected_files = {name: item["sha256"] for name, item in manifest["files"].items()}
    require(evidence["source_files"] == expected_files, f"{arm}: all source hashes")
    require(evidence["dataset_sha256"] == DATASET_SHA, f"{arm}: supervisor dataset")
    require(evidence["quality_sha256"] == sha(quality_path), f"{arm}: quality hash")
    require(evidence["server_log_sha256"] == sha(log_path), f"{arm}: server log hash")
    require(evidence["supervisor_sha256"] == sha(supervisor), f"{arm}: supervisor hash")
    embedded = report["server_evidence"]
    require(embedded["status"] == "ready", f"{arm}: quality ready snapshot")
    for key in (
        "backend",
        "repeat",
        "concurrency",
        "source_head",
        "source_files",
        "command",
        "runtime",
        "runtime_note",
        "model_config_sha256",
        "dataset_sha256",
        "supervisor_sha256",
        "server_pid",
        "quality_command",
        "started_utc",
        "ready_utc",
    ):
        require(embedded[key] == evidence[key], f"{arm}: embedded {key}")
    command = evidence["command"]
    require("--no-enable-prefix-caching" in command, f"{arm}: prefix caching")
    required_options = {
        "--model": "/model",
        "--served-model-name": "qwen-fp8",
        "--dtype": "bfloat16",
        "--tensor-parallel-size": "1",
        "--mamba-ssm-cache-dtype": "float32",
        "--seed": "42",
        "--num-gpu-blocks-override": "64",
    }
    for option, expected in required_options.items():
        require(command[command.index(option) + 1] == expected, f"{arm}: {option}")
    kernel = json.loads(command[command.index("--kernel-config") + 1])
    require(
        kernel == {"gdn_decode_backend": arm, "linear_backend": "marlin"},
        f"{arm}: kernel config",
    )
    runtime = evidence["runtime"]
    require(
        runtime["declared_source_head"] == manifest["source_head"],
        f"{arm}: declared runtime source",
    )
    runtime_source = runtime["source_files"]
    require(
        set(runtime_source)
        == {
            "vllm/config/kernel.py",
            "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
            "vllm/v1/attention/backends/gdn_attn.py",
        },
        f"{arm}: reused runtime production file scope",
    )
    for name, item in runtime_source.items():
        require(
            item["sha256"] == expected_files[name],
            f"{arm}: runtime {name}",
        )
    require(runtime["record_integrity_failures"] == [], f"{arm}: package integrity")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    capacities = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    require(capacities and set(capacities) == {"21,845"}, f"{arm}: KV capacity")
    successes = len(re.findall(r'POST /v1/chat/completions HTTP/1\.1" 200 OK', log))
    require(successes == 100, f"{arm}: logged HTTP successes {successes}/100")
    return {
        "raw_quality_sha256": sha(quality_path),
        "supervisor_record_sha256": digest(evidence),
        "server_log_sha256": sha(log_path),
        "supervisor_sha256": sha(supervisor),
        "source_hash_count": len(expected_files),
        "reused_runtime_source_hash_count": len(runtime_source),
        "logged_http_200": successes,
        "kv_capacity_tokens": 21845,
        "server_exit": evidence["server_exit"],
        "runtime_note": evidence["runtime_note"],
    }


def compare(left, right, left_stats, right_stats):
    changes, issues = [], []
    for a, b, sa, sb in zip(
        left["examples"], right["examples"], left_stats, right_stats, strict=True
    ):
        same_tokens = a["token_ids"] == b["token_ids"]
        prediction_changed = sa["predicted"] != sb["predicted"]
        raw_flip = sa["raw_correct"] != sb["raw_correct"]
        strict_flip = sa["strict_correct"] != sb["strict_correct"]
        if not same_tokens or prediction_changed or raw_flip or strict_flip:
            tokens_a, tokens_b = a["token_ids"], b["token_ids"]
            first = next(
                (i for i, (x, y) in enumerate(zip(tokens_a, tokens_b)) if x != y),
                min(len(tokens_a), len(tokens_b)),
            )
            changes.append(
                {
                    "index": a["index"],
                    "gold": a["gold"],
                    "prediction_changed": prediction_changed,
                    "raw_correctness_flip": raw_flip,
                    "strict_correctness_flip": strict_flip,
                    "first_token_difference_zero_based": first,
                    "token_context": {
                        "triton": tokens_a[max(0, first - 3) : first + 4],
                        "flashinfer": tokens_b[max(0, first - 3) : first + 4],
                    },
                    "triton": sa,
                    "flashinfer": sb,
                }
            )
        exceptional = any(
            row["truncated"]
            or row["unparsed"]
            or row["missing_marker"]
            or not row["strict_correct"]
            or row["finish_reason"] not in ("stop", "length")
            for row in (sa, sb)
        )
        if exceptional or raw_flip or strict_flip or prediction_changed:
            issues.append(
                {
                    "index": a["index"],
                    "question": a["question"],
                    "gold": a["gold"],
                    "triton": {**sa, "text": a["text"]},
                    "flashinfer": {**sb, "text": b["text"]},
                }
            )
    flip_summary = {}
    for field in ("raw_correct", "completed_correct", "strict_correct"):
        losses = [
            a["index"]
            for a, b in zip(left_stats, right_stats, strict=True)
            if a[field] and not b[field]
        ]
        gains = [
            a["index"]
            for a, b in zip(left_stats, right_stats, strict=True)
            if not a[field] and b[field]
        ]
        flip_summary[field] = {
            "triton_to_flashinfer_losses": losses,
            "triton_to_flashinfer_gains": gains,
        }
    return {
        "paired_questions": 100,
        "exact_token_sequences": sum(
            a["token_ids"] == b["token_ids"]
            for a, b in zip(left["examples"], right["examples"], strict=True)
        ),
        "same_parsed_answers": sum(
            a["predicted"] == b["predicted"]
            for a, b in zip(left_stats, right_stats, strict=True)
        ),
        "flips": flip_summary,
        "changed_outputs": changes,
        "all_exceptional_or_changed_answer_outcomes": issues,
    }


def markdown(audit):
    lines = [
        "# Matched new-head model quality screen",
        "",
        f"Source: `{audit['source_head']}`. All integrity checks passed.",
        "",
        "Same frozen 100-question GSM8K subset, zero-shot, temperature 0, seed 42,",
        "thinking disabled, concurrency 8 and 1,750 output-token budget. One",
        "successful launch and one quality run per backend on L20/SM89, TP1,",
        "BF16 inputs, FP32 state, Marlin, FA2 and 21,845 actual KV tokens.",
        "",
        "| Backend | Raw parsed correct | Completed correct | Strict correct | "
        "Truncated | Unparsed | Missing marker |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("triton", "flashinfer"):
        s = audit["backends"][arm]["summary"]
        lines.append(
            f"| {arm} | {s['correct']}/100 | {s['completed_correct']}/100 | "
            f"{s['strict_correct']}/100 | {s['truncated']} | {s['unparsed']} | "
            f"{s['missing_answer_marker']} |"
        )
    paired = audit["comparison"]
    lines += [
        "",
        "Strict correct requires a correct parsed answer, `finish_reason=stop`,",
        "no truncation and a `####` final-answer marker. Raw parsing may count",
        "a coincidentally correct trailing number in an unfinished response.",
        "",
        f"Exact token sequences match {paired['exact_token_sequences']}/100; "
        f"parsed final answers match {paired['same_parsed_answers']}/100.",
        "",
    ]
    for field, flips in paired["flips"].items():
        lines.append(
            f"- {field}: Triton-to-FI losses "
            f"{flips['triton_to_flashinfer_losses']}; gains "
            f"{flips['triton_to_flashinfer_gains']}."
        )
    issues = paired["all_exceptional_or_changed_answer_outcomes"]
    lines += [
        "",
        "## Every changed answer or exceptional outcome",
        "",
        "| Question | Gold | Triton parsed / finish / strict | "
        "FI parsed / finish / strict |",
        "| ---: | --- | --- | --- |",
    ]
    for issue in issues:
        a, b = issue["triton"], issue["flashinfer"]
        lines.append(
            f"| {issue['index']} | {issue['gold']} | {a['predicted']} / "
            f"{a['finish_reason']} / {a['strict_correct']} | {b['predicted']} / "
            f"{b['finish_reason']} / {b['strict_correct']} |"
        )
    if not issues:
        lines.append("| None | — | — | — |")
    lines += [
        "",
        "The audit JSON retains full text for every row above, all flips,",
        "all changed token sequences, and their first token divergence.",
        "",
        "## Scope and evidence",
        "",
        "This bounded single-run subset screens for visible quality regression.",
        "It does not establish statistical accuracy equivalence, performance gain,",
        "or reproducibility across launches. Equal counts do not erase output changes.",
        "Different answers alone do not attribute causality to this PR; repeat and",
        "C1 diagnostics are needed when outcomes warrant investigation.",
        "",
        "Source/head, the five source-file hashes, fixture, gold parser, dataset hash,",
        "client/supervisor hashes, protocol, token hashes, usage, launch settings,",
        "raw logs and 100 HTTP 200 completions per backend were checked.",
        "The model-config hash and runtime snapshot are matched between arms.",
        "The runtime probe was reused from the earlier review run rather than",
        "rerun inside these model processes. Native binary reuse and untested",
        "SM80/H20/TP/full CUDA13 build limitations remain.",
        "",
        audit["fixture_validation_note"],
        "",
        "Raw files are unchanged. `audit.json` and this report are derived evidence.",
        "",
    ]
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root)
    parser.add_argument("--triton", default="triton-c8-r1-quality.json")
    parser.add_argument("--flashinfer", default="flashinfer-c8-r1-quality.json")
    parser.add_argument(
        "--source-manifest", type=Path, default=root.parent / "source-manifest.json"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            root.parent.parent
            / "gdn-flashinfer-takeover-20261006"
            / "final"
            / "quality-triton.json"
        ),
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=(
            root.parent.parent / "gdn-flashinfer-takeover-20261006" / "quality_http.py"
        ),
    )
    parser.add_argument(
        "--supervisor", type=Path, default=root / "quality_supervisor.py"
    )
    parser.add_argument("--dataset", type=Path, default=root / "gsm8k-test.jsonl")
    args = parser.parse_args()
    results = args.results_dir.resolve()
    manifest = read_json(args.source_manifest)
    require(manifest["source_head"] == SOURCE_HEAD, "frozen source head")
    baseline = read_json(args.baseline)
    validate_examples(baseline, "frozen old fixture")
    reports, evidence, computed, backends = {}, {}, {}, {}
    for arm in ("triton", "flashinfer"):
        quality_path = results / getattr(args, arm)
        launch = quality_path.name.removesuffix("-quality.json")
        evidence_path = results / f"supervisor-{launch}.json"
        log_path = results / f"supervisor-{launch}.log"
        report, launch_record = read_json(quality_path), read_json(evidence_path)
        require(report["arm"] == arm, f"{arm}: quality arm")
        require(report["model"] == "qwen-fp8", f"{arm}: served model")
        require(report["protocol"] == EXPECTED_PROTOCOL, f"{arm}: frozen protocol")
        require(report["harness_sha256"] == sha(args.harness), f"{arm}: harness hash")
        require(
            report["question_fixture_sha256"] == baseline["question_fixture_sha256"],
            f"{arm}: previously frozen question fixture",
        )
        require(
            fixture_rows(report) == fixture_rows(baseline),
            f"{arm}: frozen gold/questions",
        )
        summary, stats = validate_examples(report, arm)
        integrity = validate_launch(
            report, launch_record, log_path, quality_path, manifest, args.supervisor
        )
        exit_path = results / f"{launch}.exit"
        require(
            exit_path.read_text(encoding="utf-8").strip() == "0", f"{arm}: client exit"
        )
        reports[arm], evidence[arm], computed[arm] = report, launch_record, stats
        backends[arm] = {
            "summary": summary,
            "integrity": integrity,
            "raw_quality_file": quality_path.name,
            "supervisor_file": evidence_path.name,
            "server_log_file": log_path.name,
        }
    a, b = evidence["triton"], evidence["flashinfer"]
    for key in ("runtime", "model_config_sha256", "source_files", "dataset_sha256"):
        require(a[key] == b[key], f"matched {key}")
    require(
        normalize_command(a["command"]) == normalize_command(b["command"]),
        "matched server command",
    )
    require(
        reports["triton"]["fixture"] == reports["flashinfer"]["fixture"],
        "matched dataset fixture",
    )
    note = (
        "The dataset hash and every frozen question/gold pair were checked against "
        "the prior frozen fixture; the full dataset file was not read locally."
    )
    if args.dataset:
        require(sha(args.dataset) == DATASET_SHA, "local full dataset hash")
        dataset = [
            json.loads(line)
            for line in args.dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(len(dataset) == 1319, "local dataset row count")
        for row in fixture_rows(reports["triton"]):
            item = dataset[row["index"]]
            require(
                item["question"] == row["question"]
                and item["answer"] == row["gold_text"],
                f"dataset row {row['index']}",
            )
        note = (
            "The full local dataset hash and all selected question/gold rows "
            "were checked."
        )
    run_exit = results / "run.exit"
    require(run_exit.read_text(encoding="utf-8").strip() == "0", "quality run exit")
    audit = {
        "source_head": manifest["source_head"],
        "status": "integrity_passed",
        "source_manifest_sha256": sha(args.source_manifest),
        "audit_script_sha256": sha(Path(__file__)),
        "harness_sha256": sha(args.harness),
        "supervisor_sha256": sha(args.supervisor),
        "dataset_sha256": DATASET_SHA,
        "question_fixture_sha256": reports["triton"]["question_fixture_sha256"],
        "protocol": EXPECTED_PROTOCOL,
        "protocol_sha256": digest(EXPECTED_PROTOCOL),
        "frozen_fixture_reference_sha256": sha(args.baseline),
        "model_config_sha256": a["model_config_sha256"],
        "matched_server_command_sha256": digest(normalize_command(a["command"])),
        "fixture_validation_note": note,
        "backends": backends,
        "comparison": compare(
            reports["triton"],
            reports["flashinfer"],
            computed["triton"],
            computed["flashinfer"],
        ),
        "scope": (
            "One C8 run per backend, fixed 100-question quality screen; "
            "no accuracy equivalence or performance claim"
        ),
        "limitations": [
            "Single L20/SM89, TP1; SM80 and H20 untested",
            "Reused runtime probe and native binary; full CUDA13 build unvalidated",
            "No same-backend repeat or C1 diagnostic in this audit",
            "A bounded subset does not establish statistical accuracy equivalence",
        ],
    }
    (results / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (results / "quality-results.md").write_text(markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "source_head": audit["source_head"],
                "scores": {arm: value["summary"] for arm, value in backends.items()},
                "flips": audit["comparison"]["flips"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
