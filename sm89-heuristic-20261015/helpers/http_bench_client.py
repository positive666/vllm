"""Fixed-token, closed-loop HTTP serving experiment; no third-party imports."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PROTOCOL = {
    "version": 1,
    "input_tokens": 512,
    "output_tokens": 64,
    "concurrency": [1, 8],
    "repeats": 3,
    "temperature": 0,
    "seed": 42,
    "ignore_eos": True,
    "min_tokens": 64,
    "stream": True,
    "return_token_ids": True,
    "stream_options": {"include_usage": True},
    "arrival": "closed-loop workers, next request immediately after completion",
    "connection": "one new urllib HTTP connection per request; loopback only",
    "ttft": "send-to-first nonempty completion text SSE chunk",
    "tpot": "(last-first nonempty token_ids chunk)/(usage.completion_tokens-1)",
    "limit": "SSE chunks may contain multiple or zero decoded tokens; no pure ITL",
}
HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def exclusive_json(path, value):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    at = (len(ordered) - 1) * quantile
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def summarize(records, elapsed):
    good = [row for row in records if row["success"]]
    total_tokens = sum(row["usage"]["completion_tokens"] for row in good)
    result = {
        "requests": len(records),
        "successful": len(good),
        "failed": len(records) - len(good),
        "wall_seconds": elapsed,
        "output_tokens": total_tokens,
        "output_tokens_per_second": total_tokens / elapsed,
        "requests_per_second": len(good) / elapsed,
    }
    for metric in ("ttft_ms", "tpot_ms", "request_seconds"):
        values = [row[metric] for row in good]
        result[metric] = {
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "mean": statistics.mean(values) if values else None,
        }
    return result


def post_json(base, route, payload, timeout):
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with HTTP_OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def make_fixture(args):
    # Synthetic shared prompt; this experiment measures serving, not accuracy.
    text = (
        "Continue a factual description of a city library. Describe the shelves, "
        "reading tables, opening hours, books, and visitors in plain English. "
        "The city library has tall windows and quiet rooms. "
    ) * 64
    response = post_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": text, "add_special_tokens": False},
        args.timeout,
    )
    tokens = response["tokens"][: PROTOCOL["input_tokens"]]
    if len(tokens) != PROTOCOL["input_tokens"]:
        raise RuntimeError("Tokenizer did not produce at least 512 tokens")
    if not all(type(token) is int and token >= 0 for token in tokens):
        raise RuntimeError("Invalid input token IDs")
    fixture = {
        "version": 1,
        "model": args.model,
        "purpose": "fixed synthetic serving load, not model accuracy evaluation",
        "input_tokens": tokens,
        "input_token_count": len(tokens),
        "input_tokens_sha256": digest(tokens),
        "tokenizer_add_special_tokens": False,
        "tokenizer_generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    exclusive_json(args.fixture, fixture)
    print(json.dumps({"fixture": args.fixture, "sha256": digest(tokens)}), flush=True)


def events(response, deadline):
    # SSE permits repeated data fields; blank lines terminate events.
    data = []
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Total request deadline exceeded")
        response.fp.raw._sock.settimeout(remaining)
        raw = response.readline()
        if not raw:
            break
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def one_request(args, fixture, index, worker, round_start):
    payload = {
        "model": args.model,
        "prompt": fixture["input_tokens"],
        "max_tokens": PROTOCOL["output_tokens"],
        "min_tokens": PROTOCOL["min_tokens"],
        "ignore_eos": True,
        "temperature": 0,
        "seed": PROTOCOL["seed"],
        "stream": True,
        "return_token_ids": True,
        "stream_options": PROTOCOL["stream_options"],
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    row = {
        "index": index,
        "worker": worker,
        "started_after_round_seconds": start - round_start,
        "success": False,
        "usage": None,
        "chunks": [],
        "token_chunks": [],
        "output_token_ids": [],
        "prompt_token_ids": None,
        "sse_events": 0,
        "done": False,
        "finish_reason": None,
    }
    texts = []
    try:
        with HTTP_OPENER.open(request, timeout=args.timeout) as response:
            row["http_status"] = response.status
            for data in events(response, start + args.timeout):
                received = time.perf_counter()
                if data.strip() == "[DONE]":
                    row["done"] = True
                    break
                event = json.loads(data)
                row["sse_events"] += 1
                if "error" in event:
                    raise RuntimeError("SSE error: " + str(event["error"])[:1000])
                if event.get("usage") is not None:
                    row["usage"] = event["usage"]
                choices = event.get("choices", [])
                if len(choices) > 1:
                    raise RuntimeError("Expected one completion choice")
                if choices:
                    choice = choices[0]
                    prompt_tokens = choice.get("prompt_token_ids")
                    if prompt_tokens is not None:
                        if prompt_tokens != fixture["input_tokens"]:
                            raise RuntimeError(
                                "Returned prompt token IDs differ from fixture"
                            )
                        row["prompt_token_ids"] = prompt_tokens
                    token_ids = choice.get("token_ids")
                    if token_ids:
                        if not all(
                            type(token) is int and token >= 0 for token in token_ids
                        ):
                            raise RuntimeError("Invalid output token IDs")
                        row["output_token_ids"].extend(token_ids)
                        row["token_chunks"].append(
                            {
                                "received_after_send_seconds": received - start,
                                "token_count": len(token_ids),
                            }
                        )
                    chunk = choice.get("text", "")
                    if choice.get("finish_reason") is not None:
                        row["finish_reason"] = choice["finish_reason"]
                    if chunk:
                        texts.append(chunk)
                        row["chunks"].append(
                            {
                                "received_after_send_seconds": received - start,
                                "text_chars": len(chunk),
                            }
                        )
        usage = row["usage"]
        if (
            not row["done"]
            or not row["chunks"]
            or not row["token_chunks"]
            or usage is None
        ):
            raise RuntimeError("Missing DONE, nonempty output, or terminal usage")
        if usage["prompt_tokens"] != PROTOCOL["input_tokens"]:
            raise RuntimeError("Server input token count is not 512")
        if usage["completion_tokens"] != PROTOCOL["output_tokens"]:
            raise RuntimeError("Server output token count is not 64")
        if len(row["output_token_ids"]) != PROTOCOL["output_tokens"]:
            raise RuntimeError("Streamed output token IDs do not total 64")
        if row["prompt_token_ids"] != fixture["input_tokens"]:
            raise RuntimeError("Stream did not return exact prompt token IDs")
        if row["finish_reason"] != "length":
            raise RuntimeError("Completion did not finish at fixed output length")
        first = row["chunks"][0]["received_after_send_seconds"]
        last = row["chunks"][-1]["received_after_send_seconds"]
        row["ttft_ms"] = first * 1000
        row["text_tpot_ms"] = (last - first) / (usage["completion_tokens"] - 1) * 1000
        first_token = row["token_chunks"][0]["received_after_send_seconds"]
        last_token = row["token_chunks"][-1]["received_after_send_seconds"]
        row["token_ttft_ms"] = first_token * 1000
        row["tpot_ms"] = (
            (last_token - first_token) / (usage["completion_tokens"] - 1) * 1000
        )
        row["success"] = True
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["error"] = str(error)[:2000]
        if isinstance(error, urllib.error.HTTPError):
            row["http_error_body"] = error.read(2000).decode("utf-8", "replace")
    end = time.perf_counter()
    row["request_seconds"] = end - start
    row["ended_after_round_seconds"] = end - round_start
    row["text"] = "".join(texts)
    row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    row["output_token_ids_sha256"] = digest(row["output_token_ids"])
    return row


def run_round(args, fixture, concurrency, repeat, kind):
    barrier = threading.Barrier(concurrency + 1)
    counter_lock = threading.Lock()
    counter = 0
    round_start = time.perf_counter()

    def worker(worker_index):
        nonlocal counter
        barrier.wait()
        records = []
        while True:
            with counter_lock:
                index = counter
                counter += 1
            if index >= args.active_requests:
                break
            records.append(one_request(args, fixture, index, worker_index, round_start))
        return records

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(worker, worker_index) for worker_index in range(concurrency)
        ]
        round_start = time.perf_counter()
        barrier.wait()
        rows = [row for future in futures for row in future.result()]
        elapsed = time.perf_counter() - round_start
    rows.sort(key=lambda row: row["index"])
    result = {
        "arm": args.arm,
        "kind": kind,
        "repeat": repeat,
        "concurrency": concurrency,
        "requests_target": args.active_requests,
        "input_tokens_sha256": fixture["input_tokens_sha256"],
        "protocol_sha256": digest(PROTOCOL),
        "records": rows,
        "summary": summarize(rows, elapsed),
    }
    exclusive_json(
        pathlib.Path(args.output_dir)
        / f"{args.arm}-{kind}-c{concurrency}-r{repeat}.json",
        result,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("arm", "kind", "repeat", "concurrency", "summary")
            }
        ),
        flush=True,
    )
    return result


def bench(args):
    fixture = read_json(args.fixture)
    tokens = fixture["input_tokens"]
    if (
        len(tokens) != PROTOCOL["input_tokens"]
        or digest(tokens) != fixture["input_tokens_sha256"]
    ):
        raise RuntimeError("Fixture token count/hash mismatch")
    if fixture["model"] != args.model:
        raise RuntimeError("Fixture model mismatch")
    if args.requests < 32:
        raise RuntimeError("At least 32 requests per measured round required")
    if args.warmup_requests < 8:
        raise RuntimeError("At least 8 requests per warmup concurrency required")
    paths = [pathlib.Path(args.output_dir) / f"{args.arm}.json"]
    for kind, repeats in (("warmup", [0]), ("measured", [1, 2, 3])):
        paths.extend(
            pathlib.Path(args.output_dir)
            / f"{args.arm}-{kind}-c{concurrency}-r{repeat}.json"
            for repeat in repeats
            for concurrency in PROTOCOL["concurrency"]
        )
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite an existing result")
    result = {
        "version": 1,
        "arm": args.arm,
        "model": args.model,
        "base_url": args.base_url,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "protocol_sha256": digest(PROTOCOL),
        "input_tokens_sha256": fixture["input_tokens_sha256"],
        "fixture_sha256": hashlib.sha256(
            pathlib.Path(args.fixture).read_bytes()
        ).hexdigest(),
        "client_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
        "request_timeout_seconds": args.timeout,
        "requests_per_round": args.requests,
        "server_evidence": read_json(args.server_evidence)
        if args.server_evidence
        else None,
        "warmup": [],
        "rounds": [],
    }
    args.active_requests = args.warmup_requests
    for concurrency in PROTOCOL["concurrency"]:
        warmup = run_round(args, fixture, concurrency, 0, "warmup")
        result["warmup"].append(warmup)
        if warmup["summary"]["failed"]:
            exclusive_json(pathlib.Path(args.output_dir) / f"{args.arm}.json", result)
            raise RuntimeError("Warmup failed; aborting measured rounds")
    args.active_requests = args.requests
    for repeat in range(1, 4):
        order = (
            PROTOCOL["concurrency"]
            if repeat % 2
            else list(reversed(PROTOCOL["concurrency"]))
        )
        for concurrency in order:
            result["rounds"].append(
                run_round(args, fixture, concurrency, repeat, "measured")
            )
    result["all_successful"] = all(
        row["summary"]["failed"] == 0 for row in result["rounds"]
    )
    exclusive_json(pathlib.Path(args.output_dir) / f"{args.arm}.json", result)
    if not result["all_successful"]:
        raise RuntimeError("Measured requests failed; inspect raw results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixture", "bench"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-fp8")
    parser.add_argument("--fixture", required=True)
    parser.add_argument(
        "--arm", choices=("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2")
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--server-evidence")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    parsed_url = urllib.parse.urlparse(args.base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        parser.error("Benchmark base URL must be loopback HTTP")
    if (
        parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
    ):
        parser.error("Benchmark base URL must not contain credentials/query/fragment")
    if not 0 < args.timeout <= 120:
        parser.error("Per-request timeout must be in (0, 120] seconds")
    if args.command == "fixture":
        make_fixture(args)
    else:
        if not args.arm or not args.output_dir:
            parser.error("bench requires --arm and --output-dir")
        bench(args)


if __name__ == "__main__":
    main()
