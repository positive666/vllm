"""Observe C8 decode logprobs and compare only a genuinely shared token prefix.

Run against an already authorized matched server, once per backend. Compare
reads saved responses only. This does not force tokens, replace quality scores
or isolate the cause of a numerical difference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
import urllib.parse
from pathlib import Path

import tp2_quality_diagnostic as frozen_plan

API_FILES = (
    "vllm/entrypoints/openai/chat_completion/protocol.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
    "vllm/v1/sample/sampler.py",
    "vllm/config/model.py",
)
TOP = 5
SUFFIX = "\nSolve step by step. End with #### followed by the final numeric answer."
TOKEN = re.compile(r"token_id:(\d+)")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def token_id(value):
    match = TOKEN.fullmatch(value)
    require(match is not None, "Exact token-ID representation")
    return int(match[1])


def probability(value):
    require(type(value) in (float, int) and math.isfinite(value), "Finite logprob")
    require(value <= 1e-5 and value > -9999, "Observed raw logprob, no sentinel")
    return math.exp(value)


def step(row, index):
    content = row["raw_response"]["choices"][0]["logprobs"]["content"]
    value = content[index]
    require(
        token_id(value["token"]) == row["token_ids"][index], "Chosen token alignment"
    )
    chosen_logprob = value["logprob"]
    candidates = {}
    for candidate in value["top_logprobs"]:
        ident = token_id(candidate["token"])
        require(ident not in candidates, "Unique top-logprob candidates")
        candidates[ident] = {
            "token_id": ident,
            "logprob": candidate["logprob"],
            "probability": probability(candidate["logprob"]),
        }
    require(len(candidates) == TOP, "Exactly five top-logprobs")
    ranked = sorted(candidates.values(), key=lambda item: -item["logprob"])
    return {
        "chosen_token_id": row["token_ids"][index],
        "chosen_logprob": chosen_logprob,
        "chosen_probability": probability(chosen_logprob),
        "top_candidates": ranked,
        "top2_logprob_margin": ranked[0]["logprob"] - ranked[1]["logprob"],
        "top2_probability_margin": ranked[0]["probability"] - ranked[1]["probability"],
    }


def one(args, item, quality):
    prompt = item["question"] + SUFFIX
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 3500,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True,
        "top_logprobs": TOP,
        "return_tokens_as_token_ids": True,
    }
    started = time.perf_counter()
    response = quality.post_json(
        args.base_url, "/v1/chat/completions", request, args.timeout
    )
    require(len(response.get("choices", [])) == 1, "One completion")
    choice = response["choices"][0]
    tokens = choice.get("token_ids")
    text = choice["message"].get("content") or ""
    content = (choice.get("logprobs") or {}).get("content")
    require(
        text
        and tokens
        and all(type(token) is int and token >= 0 for token in tokens)
        and len(tokens) <= 3500,
        "Nonempty text and output token IDs",
    )
    require(
        content and len(content) == len(tokens), "Logprobs cover every output token"
    )
    require(
        response["usage"]["completion_tokens"] == len(tokens),
        "Completion usage equals token IDs",
    )
    row = {
        **item,
        "prompt": prompt,
        "text": text,
        "token_ids": tokens,
        "token_ids_sha256": quality.digest(tokens),
        "prompt_token_ids": response.get("prompt_token_ids"),
        "usage": response["usage"],
        "finish_reason": choice["finish_reason"],
        "request": request,
        "raw_response": response,
        "seconds": time.perf_counter() - started,
    }
    row.update(frozen_plan.flags(row, quality))
    require(row["prompt_token_ids"], "Raw prompt token IDs")
    for index in range(len(tokens)):
        step(row, index)
    return row


def run(args):
    quality = frozen_plan.load_harness(args.harness)
    primary, selected, metadata = frozen_plan.plan(args, quality)
    url = urllib.parse.urlparse(args.base_url)
    require(
        url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1"),
        "Loopback HTTP only",
    )
    require(
        not (url.username or url.password or url.query or url.fragment),
        "No credentials, query or fragment",
    )
    require(not args.output.exists(), "Refusing to overwrite raw logprob evidence")
    server = quality.read_json(args.server_evidence)
    frozen_plan.validate_server(server, primary, args.arm)
    require(
        server["runtime"] == primary["server_evidence"]["runtime"],
        "Same frozen TP2 runtime probe",
    )
    command = server["command"]
    require(
        "--logprobs-mode" not in command
        or command[command.index("--logprobs-mode") + 1] == "raw_logprobs",
        "Raw-logprobs mode, matching default",
    )
    api_hashes = {name: frozen_plan.sha(args.api_source / name) for name in API_FILES}
    require(
        server["logprob_helper_sha256"] == frozen_plan.sha(Path(__file__))
        and server["api_source_files"] == api_hashes,
        "Supervisor binds this helper and API-source bytes",
    )
    if args.plan_only:
        print(json.dumps({"indices": frozen_plan.INDICES, "api_hashes": api_hashes}))
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        examples = list(pool.map(lambda item: one(args, item, quality), selected))
    report = {
        **metadata,
        "arm": args.arm,
        "model": args.model,
        "status": "completed",
        "scope": "C8 raw-logprobs observation; no teacher forcing or accuracy claim",
        "diagnostic_client_sha256": frozen_plan.sha(Path(__file__)),
        "fixture_helper_sha256": frozen_plan.sha(Path(frozen_plan.__file__)),
        "server_evidence": server,
        "api_source_files": api_hashes,
        "protocol": {
            **metadata["protocol"],
            "round_order": [{"concurrency": 8, "repeat": 1}],
            "requests_per_backend": 8,
            "logprobs": True,
            "top_logprobs": TOP,
            "return_tokens_as_token_ids": True,
            "comparison_scope": (
                "Shared generated-token prefix, through first divergence"
            ),
        },
        "summary": frozen_plan.summarize(examples),
        "examples": examples,
        "limitations": [
            "Logprobs collection can change scheduling and generation trajectories",
            "Top five probabilities are incomplete distributions, not full logits",
            "After first divergent token, prefixes differ and causal comparison stops",
            "A small top-two margin indicates sensitivity; it does not prove "
            "correctness",
            "No forced decode, no prompt-only teacher-forcing substitute",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quality.exclusive_json(args.output, report)
    print(json.dumps({"arm": args.arm, **report["summary"]}), flush=True)


def paired_step(x, y, index):
    a, b = step(x, index), step(y, index)
    ma = {row["token_id"]: row for row in a["top_candidates"]}
    mb = {row["token_id"]: row for row in b["top_candidates"]}
    shared = sorted(set(ma) & set(mb))
    return {
        "position_zero_based": index,
        "triton": a,
        "flashinfer": b,
        "shared_candidate_differences": [
            {
                "token_id": ident,
                "flashinfer_minus_triton_logprob": mb[ident]["logprob"]
                - ma[ident]["logprob"],
                "flashinfer_minus_triton_probability": mb[ident]["probability"]
                - ma[ident]["probability"],
            }
            for ident in shared
        ],
        "missing_in_triton_top5": sorted(set(mb) - set(ma)),
        "missing_in_flashinfer_top5": sorted(set(ma) - set(mb)),
    }


def compare(args):
    a, b = [frozen_plan.sha(path) for path in (args.triton, args.flashinfer)]
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.triton, args.flashinfer)
    ]
    for report, arm in zip(reports, ("triton", "flashinfer"), strict=True):
        require(
            report["arm"] == arm
            and report["status"] == "completed"
            and report["summary"]["requests"] == 8
            and [row["index"] for row in report["examples"]] == frozen_plan.INDICES,
            "Complete matched C8 raw response matrix",
        )
        require(
            report["diagnostic_client_sha256"] == frozen_plan.sha(Path(__file__)),
            "Frozen logprob helper bytes",
        )
    for key in (
        "model",
        "selected_indices",
        "full_question_fixture_sha256",
        "selected_question_fixture_sha256",
        "harness_sha256",
        "api_source_files",
        "protocol",
    ):
        require(reports[0][key] == reports[1][key], "Matched logprob " + key)
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
    ):
        require(
            reports[0]["server_evidence"][key] == reports[1]["server_evidence"][key],
            "Matched server " + key,
        )
    quality = frozen_plan.load_harness(args.harness)
    paired = []
    for x, y in zip(reports[0]["examples"], reports[1]["examples"], strict=True):
        require(
            x["index"] == y["index"]
            and x["prompt"] == y["prompt"]
            and x["prompt_token_ids"] == y["prompt_token_ids"],
            "Same incoming prompt tokens",
        )
        for row in (x, y):
            require(
                row["token_ids_sha256"] == quality.digest(row["token_ids"]),
                "Raw token hash",
            )
            for index in range(len(row["token_ids"])):
                step(row, index)
        minimum = min(len(x["token_ids"]), len(y["token_ids"]))
        first = next(
            (
                index
                for index in range(minimum)
                if x["token_ids"][index] != y["token_ids"][index]
            ),
            None,
        )
        limit = first + 1 if first is not None else minimum
        shared_positions = [paired_step(x, y, i) for i in range(limit)]
        difference_values = [
            delta
            for position in shared_positions
            for delta in position["shared_candidate_differences"]
        ]
        paired.append(
            {
                "index": x["index"],
                "gold": x["gold"],
                "triton_predicted": x["predicted"],
                "flashinfer_predicted": y["predicted"],
                "triton_finish_reason": x["finish_reason"],
                "flashinfer_finish_reason": y["finish_reason"],
                "matching_prefix_tokens": first if first is not None else minimum,
                "first_divergence_zero_based": first,
                "sequence_lengths": [len(x["token_ids"]), len(y["token_ids"])],
                "sequence_length_differs_without_token_divergence": first is None
                and len(x["token_ids"]) != len(y["token_ids"]),
                "common_incoming_prefix_positions_compared": limit,
                "minimum_top2_logprob_margin": {
                    arm: min(
                        position[arm]["top2_logprob_margin"]
                        for position in shared_positions
                    )
                    for arm in ("triton", "flashinfer")
                },
                "max_abs_shared_candidate_logprob_difference": max(
                    (
                        abs(row["flashinfer_minus_triton_logprob"])
                        for row in difference_values
                    ),
                    default=None,
                ),
                "max_abs_shared_candidate_probability_difference": max(
                    (
                        abs(row["flashinfer_minus_triton_probability"])
                        for row in difference_values
                    ),
                    default=None,
                ),
                "last_common_prefix_window_and_divergence": shared_positions[-6:],
                "divergence": shared_positions[-1] if first is not None else None,
            }
        )
    output = {
        "status": "completed_observation",
        "raw_sha256": {"triton": a, "flashinfer": b},
        "helper_sha256": frozen_plan.sha(Path(__file__)),
        "paired": paired,
        "limitations": reports[0]["limitations"],
    }
    require(not args.output.exists(), "Refusing to overwrite logprob comparison")
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": output["status"], "pairs": len(paired)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--arm", choices=("triton", "flashinfer"), required=True)
    run_parser.add_argument("--dataset", type=Path, required=True)
    run_parser.add_argument("--primary-quality", type=Path, required=True)
    run_parser.add_argument("--server-evidence", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_parser.add_argument("--model", default="qwen-fp8")
    run_parser.add_argument("--timeout", type=float, default=300)
    run_parser.add_argument("--plan-only", action="store_true")
    run_parser.add_argument("--api-source", type=Path, default=Path("/source"))
    run_parser.add_argument(
        "--harness", type=Path, default=Path("/reference-artifacts/quality_http.py")
    )
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--triton", type=Path, required=True)
    compare_parser.add_argument("--flashinfer", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--harness", type=Path, default=Path(__file__).parent / "quality_http.py"
    )
    args = parser.parse_args()
    run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    main()
