"""Audit saved TP2 logprob observations; never contact the server or GPU."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import audit_tp2_diagnostics as contracts

API_FILES = (
    "vllm/entrypoints/openai/chat_completion/protocol.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
    "vllm/v1/sample/sampler.py",
    "vllm/config/model.py",
)
NATIVE = "3efc749cc7d32a0177376cdabc19d18f406ca2f5a002f933a48997882536d6d8"
TOKEN = re.compile(r"token_id:(\d+)")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def ident(value):
    match = TOKEN.fullmatch(value)
    require(match is not None, "Unambiguous token ID representation")
    return int(match[1])


def numeric_logprob(value):
    require(
        type(value) in (int, float) and math.isfinite(value) and -9999 < value <= 1e-5,
        "Finite raw logprob without missing-value sentinel",
    )
    return math.exp(value)


def observation(row, index):
    content = row["raw_response"]["choices"][0]["logprobs"]["content"]
    value = content[index]
    require(ident(value["token"]) == row["token_ids"][index], "Logprob token alignment")
    candidates = {}
    for item in value["top_logprobs"]:
        token = ident(item["token"])
        require(token not in candidates, "Unique candidate token ID")
        candidates[token] = {
            "token_id": token,
            "logprob": item["logprob"],
            "probability": numeric_logprob(item["logprob"]),
        }
    require(len(candidates) == 5, "Five raw candidates")
    top = sorted(candidates.values(), key=lambda item: -item["logprob"])
    return {
        "chosen_token_id": row["token_ids"][index],
        "chosen_logprob": value["logprob"],
        "chosen_probability": numeric_logprob(value["logprob"]),
        "top_candidates": top,
        "top2_logprob_margin": top[0]["logprob"] - top[1]["logprob"],
        "top2_probability_margin": top[0]["probability"] - top[1]["probability"],
    }


def position(a, b, index):
    x, y = observation(a, index), observation(b, index)
    tx = {item["token_id"]: item for item in x["top_candidates"]}
    ty = {item["token_id"]: item for item in y["top_candidates"]}
    return {
        "position_zero_based": index,
        "triton": x,
        "flashinfer": y,
        "shared_candidate_differences": [
            {
                "token_id": token,
                "flashinfer_minus_triton_logprob": ty[token]["logprob"]
                - tx[token]["logprob"],
                "flashinfer_minus_triton_probability": ty[token]["probability"]
                - tx[token]["probability"],
            }
            for token in sorted(set(tx) & set(ty))
        ],
        "missing_in_triton_top5": sorted(set(ty) - set(tx)),
        "missing_in_flashinfer_top5": sorted(set(tx) - set(ty)),
    }


def compare_records(a, b):
    output = []
    for x, y in zip(a["examples"], b["examples"], strict=True):
        require(
            x["index"] == y["index"]
            and x["prompt"] == y["prompt"]
            and x["prompt_token_ids"] == y["prompt_token_ids"],
            "Matched question and actual incoming prompt tokens",
        )
        common = min(len(x["token_ids"]), len(y["token_ids"]))
        first = next(
            (i for i in range(common) if x["token_ids"][i] != y["token_ids"][i]), None
        )
        limit = first + 1 if first is not None else common
        positions = [position(x, y, i) for i in range(limit)]
        differences = [
            delta
            for item in positions
            for delta in item["shared_candidate_differences"]
        ]
        output.append(
            {
                "index": x["index"],
                "gold": x["gold"],
                "triton_predicted": x["predicted"],
                "flashinfer_predicted": y["predicted"],
                "triton_finish_reason": x["finish_reason"],
                "flashinfer_finish_reason": y["finish_reason"],
                "matching_prefix_tokens": first if first is not None else common,
                "first_divergence_zero_based": first,
                "sequence_lengths": [len(x["token_ids"]), len(y["token_ids"])],
                "sequence_length_differs_without_token_divergence": first is None
                and len(x["token_ids"]) != len(y["token_ids"]),
                "common_incoming_prefix_positions_compared": limit,
                "minimum_top2_logprob_margin": {
                    arm: min(item[arm]["top2_logprob_margin"] for item in positions)
                    for arm in ("triton", "flashinfer")
                },
                "max_abs_shared_candidate_logprob_difference": max(
                    (
                        abs(item["flashinfer_minus_triton_logprob"])
                        for item in differences
                    ),
                    default=None,
                ),
                "max_abs_shared_candidate_probability_difference": max(
                    (
                        abs(item["flashinfer_minus_triton_probability"])
                        for item in differences
                    ),
                    default=None,
                ),
                "last_common_prefix_window_and_divergence": positions[-6:],
                "divergence": positions[-1] if first is not None else None,
            }
        )
    return output


def equal(expected, actual, label):
    if type(expected) is float:
        require(
            type(actual) in (float, int)
            and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12),
            label + "/float",
        )
    elif type(expected) is dict:
        require(type(actual) is dict and set(expected) == set(actual), label + "/keys")
        for key in expected:
            equal(expected[key], actual[key], label + "/" + key)
    elif type(expected) is list:
        require(
            type(actual) is list and len(expected) == len(actual), label + "/length"
        )
        for i, (x, y) in enumerate(zip(expected, actual, strict=True)):
            equal(x, y, label + "/" + str(i))
    else:
        require(type(expected) is type(actual) and expected == actual, label + "/value")


def backend(args, arm, dataset, source, api):
    folder = args.results / "logprob-diagnostics"
    raw_path, server_path = folder / f"{arm}.json", folder / f"serve-{arm}.json"
    report, server = contracts.read(raw_path), contracts.read(server_path)
    primary_path = args.results / f"quality-{arm}.json"
    primary = contracts.read(primary_path)
    diagnostic = contracts.read(args.results / "diagnostics" / f"serve-{arm}.json")
    require(
        report["status"] == "completed"
        and report["arm"] == arm
        and report["model"] == "qwen-fp8"
        and server["status"] == "completed"
        and server["backend"] == arm
        and server["phase"] == "top-logprobs-diagnostic"
        and "error" not in server,
        arm + "/complete raw and server",
    )
    require(
        server["logprob_sha256"] == contracts.sha(raw_path)
        and server["logprob_helper_sha256"]
        == report["diagnostic_client_sha256"]
        == contracts.sha(args.client)
        and server["supervisor_sha256"] == contracts.sha(args.supervisor)
        and report["fixture_helper_sha256"] == contracts.sha(args.fixture_helper)
        and report["harness_sha256"] == contracts.sha(args.harness),
        arm + "/frozen raw and helper bytes",
    )
    require(
        report["api_source_files"] == server["api_source_files"] == api,
        arm + "/frozen current API source",
    )
    for key in (
        "source_head",
        "source_files",
        "tensor_parallel_size",
        "per_rank_gdn_shape",
        "state_dtype",
        "gpu_uuids",
        "runtime",
        "model_config_sha256",
        "gpu_blocks",
        "kv_capacity_tokens",
        "max_model_len",
        "command",
    ):
        require(server[key] == diagnostic[key], arm + "/same expanded server " + key)
    require(
        server["source_head"] == contracts.HEAD
        and server["source_files"] == source
        and server["source_manifest_sha256"] == contracts.sha(args.source_manifest)
        and server["model_config_sha256"] == contracts.sha(args.model_config)
        and server["tensor_parallel_size"] == 2
        and server["per_rank_gdn_shape"] == contracts.LOCAL
        and server["gpu_blocks"] == 128
        and server["max_model_len"] == 4096
        and server["kv_capacity_tokens"] >= 32768
        and server["native_library"]["sha256"] == NATIVE
        and server["runtime"]["modules"]["vllm._C_stable_libtorch"]["sha256"] == NATIVE,
        arm + "/reviewed configuration and native runtime",
    )
    embedded = report["server_evidence"]
    require(embedded["status"] == "ready", arm + "/ready snapshot")
    for key, value in embedded.items():
        if key != "status":
            require(server[key] == value, arm + "/ready record " + key)
    protocol = {
        **contracts.PROTOCOL,
        "round_order": [{"concurrency": 8, "repeat": 1}],
        "requests_per_backend": 8,
        "logprobs": True,
        "top_logprobs": 5,
        "return_tokens_as_token_ids": True,
        "comparison_scope": "Shared generated-token prefix, through first divergence",
    }
    require(report["protocol"] == protocol, arm + "/matched protocol")
    require(
        report["primary_quality_sha256"] == contracts.sha(primary_path)
        and report["full_question_fixture_sha256"] == primary["question_fixture_sha256"]
        and report["full_fixture"] == primary["fixture"]
        and report["selected_indices"] == contracts.INDICES
        and report["target_indices"] == contracts.TARGETS,
        arm + "/same original fixture",
    )
    rows = report["examples"]
    require(
        len(rows) == 8 and [row["index"] for row in rows] == contracts.INDICES,
        arm + "/complete C8 matrix",
    )
    computed = []
    for row in rows:
        value = contracts.outcome(row, dataset, 3500, arm + "/output", True)
        request = {
            "model": "qwen-fp8",
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 3500,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "logprobs": True,
            "top_logprobs": 5,
            "return_tokens_as_token_ids": True,
        }
        require(row["request"] == request, arm + "/exact HTTP request")
        response = row["raw_response"]
        require(len(response["choices"]) == 1, arm + "/one raw choice")
        choice = response["choices"][0]
        require(
            choice["token_ids"] == row["token_ids"]
            and choice["message"]["content"] == row["text"]
            and choice["finish_reason"] == row["finish_reason"]
            and response["usage"] == row["usage"]
            and response["prompt_token_ids"] == row["prompt_token_ids"]
            and len(row["prompt_token_ids"]) == row["usage"]["prompt_tokens"],
            arm + "/raw response and extracted fields",
        )
        require(
            len(choice["logprobs"]["content"]) == len(row["token_ids"]),
            arm + "/logprob length",
        )
        for i in range(len(row["token_ids"])):
            observation(row, i)
        computed.append(value)
    summary = contracts.summarize(computed)
    require(report["summary"] == summary, arm + "/independent score summary")
    log_path = server_path.with_suffix(".log")
    proof = contracts.log_integrity(log_path, server_path, server["server_log_sha256"])
    log = log_path.read_text(encoding="utf-8")
    require(
        re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', log) == ["200"] * 8,
        arm + "/exact eight successful HTTP requests",
    )
    require(
        {
            int(x.replace(",", ""))
            for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
        }
        == {server["kv_capacity_tokens"]},
        arm + "/actual KV capacity",
    )
    return (
        report,
        server,
        {
            "summary": summary,
            "outcomes": computed,
            "target_outcomes": [
                row for row in computed if row["index"] in contracts.TARGETS
            ],
            "integrity": {
                "raw_client_sha256": contracts.sha(raw_path),
                "server_record_sha256": contracts.sha(server_path),
                "server_log_sha256": contracts.sha(log_path),
                "primary_quality_sha256": contracts.sha(primary_path),
                "helper_sha256": contracts.sha(args.client),
                "supervisor_sha256": contracts.sha(args.supervisor),
                "log_integrity": proof,
            },
        },
    )


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "raw")
    parser.add_argument("--dataset", type=Path, default=root / "gsm8k-test.jsonl")
    parser.add_argument("--harness", type=Path, default=root / "quality_http.py")
    parser.add_argument("--client", type=Path, default=root / "tp2_top_logprobs.py")
    parser.add_argument(
        "--supervisor", type=Path, default=root / "tp2_logprobs_supervisor.py"
    )
    parser.add_argument(
        "--fixture-helper", type=Path, default=root / "tp2_quality_diagnostic.py"
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=root / "performance-source.json"
    )
    parser.add_argument("--model-config", type=Path, default=root / "model-config.json")
    parser.add_argument(
        "--api-source",
        type=Path,
        default=Path("D:/code/vllm/.codex_tmp/gdn60403-main-recheck-20261008"),
    )
    parser.add_argument("--output", type=Path, default=root / "audit-logprobs.json")
    args = parser.parse_args()
    require(contracts.sha(args.dataset) == contracts.DATASET_SHA, "Dataset bytes")
    dataset = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(dataset) == 1319, "Full dataset")
    manifest = contracts.read(args.source_manifest)
    require(manifest["source_head"] == contracts.HEAD, "Reviewed source manifest")
    source = {name: row["sha256"] for name, row in manifest["files"].items()}
    api = {name: contracts.sha(args.api_source / name) for name in API_FILES}
    reports, servers, summaries = {}, {}, {}
    for arm in ("triton", "flashinfer"):
        reports[arm], servers[arm], summaries[arm] = backend(
            args, arm, dataset, source, api
        )
    require(
        contracts.normalized_command(servers["triton"]["command"])
        == contracts.normalized_command(servers["flashinfer"]["command"]),
        "Matched backend commands",
    )
    for key in ("gpu_uuids", "runtime", "kv_capacity_tokens"):
        require(servers["triton"][key] == servers["flashinfer"][key], "Matched " + key)
    comparison_path = args.results / "logprob-diagnostics/comparison.json"
    comparison = contracts.read(comparison_path)
    paired = compare_records(reports["triton"], reports["flashinfer"])
    require(
        comparison["status"] == "completed_observation"
        and comparison["helper_sha256"] == contracts.sha(args.client)
        and comparison["raw_sha256"]
        == {arm: summaries[arm]["integrity"]["raw_client_sha256"] for arm in reports},
        "Saved comparison hashes",
    )
    equal(
        paired, comparison["paired"], "Independently computed shared-prefix comparison"
    )
    require(
        (args.results / "logprob-diagnostics/run.exit")
        .read_text(encoding="utf-8")
        .strip()
        == "0",
        "Runner exit",
    )
    result = {
        "status": "passed",
        "meaning_of_pass": (
            "Raw observation integrity passed; no quality or cause claim"
        ),
        "source_head": contracts.HEAD,
        "source_files": source,
        "api_source_files": api,
        "audit_script_sha256": contracts.sha(Path(__file__)),
        "parser_contract_sha256": contracts.sha(Path(contracts.__file__)),
        "comparison_sha256": contracts.sha(comparison_path),
        "backends": summaries,
        "paired": paired,
        "limitations": [
            "Top-five probabilities cover part of the vocabulary",
            "Logprob collection can change scheduling and trajectories",
            "Comparison stops after the first divergent generated token",
            "Near ties do not establish adapter correctness or accuracy equivalence",
            "One C8 observation per backend; no forced decode or full-state shadow",
            "Float verification allows 1e-12 for cross-platform exp rounding only",
        ],
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "pairs": len(paired)}))


if __name__ == "__main__":
    main()
