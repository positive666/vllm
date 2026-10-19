"""Bounded deterministic HTTP quality screen with raw token evidence.

Run once per backend, on the same server configuration and same fixture. The
default fixed math prompts are a smoke screen, not a model accuracy benchmark.
An optional cached GSM8K test JSONL enables the same deterministic 100-example
zero-shot subset on each arm. Preserve truncation and unparsed-answer counts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.parse
from decimal import Decimal, InvalidOperation
from pathlib import Path

from http_bench_client import digest, exclusive_json, post_json, read_json

FIXED = [
    ("What is 17 plus 25?", "42"),
    ("What is 13 multiplied by 9?", "117"),
    (
        "A box has 12 pens. Seven are added, then four are taken out. How many remain?",
        "15",
    ),
    ("A train travels 240 km in 3 hours. What is its average speed in km/h?", "80"),
    (
        "Three notebooks cost 7 dollars each and five pens cost 3 dollars each. "
        "What is the total cost in dollars?",
        "36",
    ),
    (
        "A tank holds 80 liters. It is 35 percent full. "
        "How many liters does it contain?",
        "28",
    ),
    ("Solve 4x + 9 = 37 for x.", "7"),
    (
        "Two sides of a rectangle measure 12 and 5 cm. What is its area in square cm?",
        "60",
    ),
    ("A fair six-sided die is rolled once. How many possible outcomes are even?", "3"),
    ("What is the sum of all integers from 1 through 20?", "210"),
    (
        "A book costs 50 dollars before a 20 percent discount. "
        "What is the sale price in dollars?",
        "40",
    ),
    (
        "An item costs 36 dollars for 9 kilograms. "
        "What is the price per kilogram in dollars?",
        "4",
    ),
    ("小明有18元，买了3本每本4元的练习本。还剩多少元？", "6"),
    ("一辆车每小时行驶60公里，行驶2.5小时。共行驶多少公里？", "150"),
    ("学校有8个班，每班35人，一共有多少名学生？", "280"),
    ("一个长方形的长是9厘米，宽是4厘米。周长是多少厘米？", "26"),
]
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def answer_number(text):
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    matches = NUMBER.findall(text)
    if not matches:
        return None
    try:
        number = Decimal(matches[-1].replace(",", ""))
        return format(number.normalize(), "f") if number else "0"
    except InvalidOperation:
        return None


def fixture_rows(args):
    if args.dataset is None:
        return (
            [
                {"index": index, "question": question, "gold": gold}
                for index, (question, gold) in enumerate(FIXED)
            ],
            {
                "kind": "fixed_arithmetic_smoke",
                "samples": len(FIXED),
                "limitations": (
                    "hand-authored smoke questions, not a model evaluation dataset"
                ),
            },
        )
    dataset = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = sorted(random.Random(42).sample(range(len(dataset)), args.samples))
    rows = []
    for index in indices:
        item = dataset[index]
        gold = answer_number(item["answer"])
        if gold is None:
            raise ValueError(f"Cannot parse gold answer at dataset row {index}")
        rows.append(
            {
                "index": index,
                "question": item["question"],
                "gold": gold,
                "gold_text": item["answer"],
            }
        )
    return rows, {
        "kind": "gsm8k_test_zero_shot",
        "samples": args.samples,
        "selection_seed": 42,
        "indices": indices,
        "dataset_rows": len(dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "limitations": "bounded deterministic subset; no statistical equivalence claim",
    }


def one(args, item):
    prompt = (
        item["question"]
        + "\nSolve step by step. End with #### followed by the final numeric answer."
    )
    before = time.perf_counter()
    response = post_json(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": args.max_tokens,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        args.timeout,
    )
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise RuntimeError("Expected one completion")
    choice = choices[0]
    text = choice["message"].get("content") or ""
    tokens = choice.get("token_ids")
    if (
        not text
        or not tokens
        or not all(type(token) is int and token >= 0 for token in tokens)
    ):
        raise RuntimeError("Empty response or missing output token IDs")
    usage = response.get("usage")
    if usage is None or usage["completion_tokens"] != len(tokens):
        raise RuntimeError("Completion usage and token count differ")
    predicted = answer_number(text)
    return {
        **item,
        "prompt": prompt,
        "text": text,
        "token_ids": tokens,
        "token_ids_sha256": digest(tokens),
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length",
        "has_answer_marker": "####" in text,
        "predicted": predicted,
        "correct": predicted == item["gold"],
        "seconds": time.perf_counter() - before,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-fp8")
    parser.add_argument("--arm", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, choices=(1, 8), default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--server-evidence", type=Path)
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        parser.error("Only authorized loopback HTTP is allowed")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("URL must not contain credentials, query or fragment")
    if args.output.exists() or args.output.with_suffix(".progress.json").exists():
        raise FileExistsError("Refusing to overwrite quality evidence")
    rows, fixture = fixture_rows(args)
    report = {
        "arm": args.arm,
        "model": args.model,
        "fixture": fixture,
        "question_fixture_sha256": digest(rows),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "server_evidence": read_json(args.server_evidence)
        if args.server_evidence
        else None,
        "protocol": {
            "temperature": 0,
            "seed": 42,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
            "concurrency": args.concurrency,
            "return_token_ids": True,
            "prefix_caching": "must be disabled on server",
        },
        "examples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(rows), args.concurrency):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            completed = list(
                pool.map(
                    lambda item: one(args, item),
                    rows[offset : offset + args.concurrency],
                )
            )
        report["examples"].extend(completed)
        args.output.with_suffix(".progress.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "quality_progress",
                    "arm": args.arm,
                    "completed": len(report["examples"]),
                    "correct": sum(row["correct"] for row in report["examples"]),
                }
            ),
            flush=True,
        )
    examples = report["examples"]
    report["summary"] = {
        "samples": len(examples),
        "correct": sum(row["correct"] for row in examples),
        "truncated": sum(row["truncated"] for row in examples),
        "unparsed": sum(row["predicted"] is None for row in examples),
        "missing_answer_marker": sum(not row["has_answer_marker"] for row in examples),
        "http_failures": 0,
    }
    exclusive_json(args.output, report)
    print(json.dumps({"arm": args.arm, **report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
