"""Independently audit fixed-workload ABBA serving performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
ARMS = ("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2")
INPUT_TOKENS = 512
OUTPUT_TOKENS = 128
REQUESTS = 16
WARMUP_REQUESTS = 8
CONCURRENCIES = (1, 8)
NATIVE_SHA = "3efc749cc7d32a0177376cdabc19d18f406ca2f5a002f933a48997882536d6d8"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def distribution(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "values": values,
    }


def summarize(rows, elapsed):
    result = {
        "requests": len(rows),
        "successful": len(rows),
        "failed": 0,
        "wall_seconds": elapsed,
        "output_tokens": len(rows) * OUTPUT_TOKENS,
        "output_tokens_per_second": len(rows) * OUTPUT_TOKENS / elapsed,
        "requests_per_second": len(rows) / elapsed,
    }
    for metric in ("ttft_ms", "tpot_ms", "request_seconds"):
        values = [row[metric] for row in rows]
        result[metric] = {
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "mean": statistics.mean(values),
        }
    return result


def compare_nested(actual, expected, check, label):
    for key, value in expected.items():
        at = f"{label}/{key}"
        if isinstance(value, dict):
            check(isinstance(actual.get(key), dict), at + ": missing mapping")
            compare_nested(actual.get(key, {}), value, check, at)
        elif isinstance(value, (float, int)):
            observed = actual.get(key)
            check(
                isinstance(observed, (float, int))
                and math.isfinite(observed)
                and math.isclose(observed, value, rel_tol=1e-9, abs_tol=1e-8),
                at + f": observed={observed!r}, recomputed={value!r}",
            )
        else:
            check(actual.get(key) == value, at + ": mismatch")


def normal_command(command):
    result = list(command)
    position = result.index("--kernel-config") + 1
    kernel = json.loads(result[position])
    kernel["gdn_decode_backend"] = "<backend>"
    result[position] = json.dumps(kernel, sort_keys=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors, notes = [], []

    def check(condition, label):
        if not condition:
            errors.append(label)

    report = {
        "source_head": HEAD,
        "protocol": {
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "concurrency": list(CONCURRENCIES),
            "requests_per_measured_round": REQUESTS,
            "warmup_requests_per_concurrency": WARMUP_REQUESTS,
            "rounds_per_concurrency_per_launch": 3,
            "launch_order": list(ARMS),
            "independent_launches_per_backend": 2,
        },
        "statistical_scope": (
            "Descriptive launch/round comparisons only. Requests within a launch "
            "are not independent hardware/process replications. No significance "
            "test or confidence interval based on pooled requests is asserted."
        ),
        "native_runtime_scope": (
            "The reused native library is tied to its recorded SHA256, not a "
            "fresh full CUDA13 build. Embedded inherited runtime probes retain "
            "their original timestamp/provenance."
        ),
        "launches": {},
        "comparisons": {},
        "errors": errors,
        "notes": notes,
    }
    manifest = read(args.source_manifest)
    check(manifest["source_head"] == HEAD, "source-manifest head mismatch")
    expected_hashes = {
        name: detail["sha256"] for name, detail in manifest["files"].items()
    }
    check(len(expected_hashes) == 5, "expected five source-file hashes")
    fixtures = [(path, read(path)) for path in args.results.rglob("*fixture*.json")]
    commands, launch_times, fixture_hashes, protocol_hashes = [], [], set(), set()
    client_hashes, runtime_hashes, model_hashes = set(), set(), set()
    measured_total = warmup_total = 0
    for arm in ARMS:
        paths = list(args.results.rglob(arm + ".json"))
        check(len(paths) == 1, f"{arm}: expected exactly one client total")
        if len(paths) != 1:
            continue
        path, launch = paths[0], read(paths[0])
        result = {"client_path": str(path), "client_sha256": sha(path)}
        report["launches"][arm] = result
        check(launch.get("arm") == arm, f"{arm}: client arm mismatch")
        client_hashes.add(launch.get("client_sha256"))
        protocol = launch.get("protocol", {})
        for key, expected in {
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "min_tokens": OUTPUT_TOKENS,
            "concurrency": list(CONCURRENCIES),
            "repeats": 3,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
        }.items():
            check(protocol.get(key) == expected, f"{arm}: protocol/{key} mismatch")
        protocol_sha = digest(protocol)
        check(launch.get("protocol_sha256") == protocol_sha, f"{arm}: protocol hash")
        protocol_hashes.add(protocol_sha)
        fixture_matches = [
            (p, f) for p, f in fixtures if sha(p) == launch.get("fixture_sha256")
        ]
        check(bool(fixture_matches), f"{arm}: exact fixture bytes not found")
        fixture_tokens = (
            fixture_matches[0][1]["input_tokens"] if fixture_matches else []
        )
        check(len(fixture_tokens) == INPUT_TOKENS, f"{arm}: fixture input length")
        check(
            all(type(t) is int and t >= 0 for t in fixture_tokens),
            f"{arm}: fixture invalid token IDs",
        )
        prompt_hash = digest(fixture_tokens)
        check(launch.get("input_tokens_sha256") == prompt_hash, f"{arm}: prompt hash")
        fixture_hashes.add(launch.get("fixture_sha256"))

        evidence = launch.get("server_evidence") or {}
        check(evidence.get("arm") == arm, f"{arm}: embedded server arm mismatch")
        backend = arm.split("-")[0]
        check(evidence.get("backend") == backend, f"{arm}: embedded backend mismatch")
        check(evidence.get("source_head") == HEAD, f"{arm}: server head mismatch")
        hashes = evidence.get("source_files", {})
        for name, expected in expected_hashes.items():
            check(hashes.get(name) == expected, f"{arm}: source hash {name}")
        runtime = evidence.get("runtime", {})
        runtime_hashes.add(digest(runtime))
        model_hashes.add(evidence.get("model_config_sha256"))
        check(runtime.get("declared_source_head") == HEAD, f"{arm}: runtime head")
        native = runtime.get("modules", {}).get("vllm._C_stable_libtorch", {})
        check(native.get("sha256") == NATIVE_SHA, f"{arm}: native library hash")
        result["runtime_probe_timestamp"] = runtime.get("created_utc")
        result["runtime_provenance"] = evidence.get("runtime_provenance")
        command = evidence.get("command", [])
        try:
            kernel = json.loads(command[command.index("--kernel-config") + 1])
            check(kernel.get("gdn_decode_backend") == backend, f"{arm}: kernel backend")
            check(kernel.get("linear_backend") == "marlin", f"{arm}: linear backend")
            check(
                command[command.index("--mamba-ssm-cache-dtype") + 1] == "float32",
                f"{arm}: FP32 SSM state required",
            )
            check(
                "--no-enable-prefix-caching" in command, f"{arm}: prefix cache enabled"
            )
            check(
                command[command.index("--num-gpu-blocks-override") + 1] == "64",
                f"{arm}: cache blocks",
            )
            for option, expected in {
                "--model": "/model",
                "--host": "127.0.0.1",
                "--port": "8000",
                "--dtype": "bfloat16",
                "--tensor-parallel-size": "1",
                "--max-model-len": "2048",
                "--max-num-seqs": "32",
                "--max-num-batched-tokens": "1024",
                "--gpu-memory-utilization": "0.9",
                "--seed": "42",
            }.items():
                check(
                    command[command.index(option) + 1] == expected,
                    f"{arm}: command {option}",
                )
            attention = json.loads(command[command.index("--attention-config") + 1])
            check(
                attention == {"backend": "FLASH_ATTN", "flash_attn_version": 2},
                f"{arm}: attention backend",
            )
            commands.append(normal_command(command))
        except (ValueError, IndexError, TypeError, json.JSONDecodeError) as error:
            check(False, f"{arm}: malformed command: {error}")
        completed = [
            (p, read(p))
            for p in args.results.rglob("serve-*.json")
            if read(p).get("arm") == arm and read(p).get("status") == "completed"
        ]
        check(len(completed) == 1, f"{arm}: expected one completed server launch")
        if completed:
            server_path, server = completed[0]
            result["server_path"] = str(server_path)
            result["server_sha256"] = sha(server_path)
            result["server_status"] = server.get("status")
            result["kv_capacity_tokens"] = server.get("kv_capacity_tokens")
            check(
                server.get("kv_capacity_tokens") == 21845,
                f"{arm}: KV capacity mismatch",
            )
            check(
                server.get("source_files") == hashes,
                f"{arm}: final/embedded source mismatch",
            )
            check(
                server.get("command") == command,
                f"{arm}: final/embedded command mismatch",
            )
            check(
                server.get("client_result_sha256") == sha(path),
                f"{arm}: server/client result hash mismatch",
            )
            server_log = server_path.with_suffix(".log")
            check(server_log.exists(), f"{arm}: raw server log missing")
            if server_log.exists():
                check(
                    server.get("server_log_sha256") == sha(server_log),
                    f"{arm}: server log hash mismatch",
                )
                capacities = re.findall(
                    r"GPU KV cache size: ([\d,]+) tokens",
                    server_log.read_text(errors="replace"),
                )
                check(
                    capacities and set(capacities) == {"21,845"},
                    f"{arm}: raw log KV capacity mismatch",
                )
            launch_times.append((arm, server.get("started_utc", "")))
        result["rounds"] = []
        for kind, expected_count, repeat_ids in (
            ("warmup", WARMUP_REQUESTS, (0,)),
            ("rounds", REQUESTS, (1, 2, 3)),
        ):
            rounds = launch.get(kind, [])
            expected_keys = {(c, r) for c in CONCURRENCIES for r in repeat_ids}
            keys = [(r.get("concurrency"), r.get("repeat")) for r in rounds]
            check(
                set(keys) == expected_keys and len(keys) == len(expected_keys),
                f"{arm}/{kind}: round matrix",
            )
            for round_record in rounds:
                concurrency, repeat = (
                    round_record.get("concurrency"),
                    round_record.get("repeat"),
                )
                label = f"{arm}/{kind}/c{concurrency}/r{repeat}"
                check(round_record.get("arm") == arm, label + ": arm mismatch")
                expected_kind = "measured" if kind == "rounds" else "warmup"
                check(round_record.get("kind") == expected_kind, label + ": kind")
                rows = round_record.get("records", [])
                check(len(rows) == expected_count, label + ": request count")
                check(
                    {r.get("index") for r in rows} == set(range(expected_count)),
                    label + ": request indexes",
                )
                check(
                    round_record.get("protocol_sha256") == protocol_sha,
                    label + ": protocol hash",
                )
                check(
                    round_record.get("input_tokens_sha256") == prompt_hash,
                    label + ": prompt hash",
                )
                check(
                    round_record.get("requests_target") == expected_count,
                    label + ": request target",
                )
                for row in rows:
                    at = label + f"/request{row.get('index')}"
                    check(row.get("success") is True, at + ": request failed")
                    check(
                        row.get("http_status") == 200 and row.get("done") is True,
                        at + ": HTTP/DONE",
                    )
                    check(row.get("finish_reason") == "length", at + ": finish reason")
                    usage, ids = row.get("usage") or {}, row.get("output_token_ids", [])
                    check(
                        usage.get("prompt_tokens") == INPUT_TOKENS, at + ": input usage"
                    )
                    check(
                        usage.get("completion_tokens") == OUTPUT_TOKENS,
                        at + ": output usage",
                    )
                    check(
                        len(ids) == OUTPUT_TOKENS
                        and all(type(t) is int and t >= 0 for t in ids),
                        at + ": output token IDs",
                    )
                    check(
                        row.get("output_token_ids_sha256") == digest(ids),
                        at + ": output hash",
                    )
                    check(
                        row.get("prompt_token_ids") == fixture_tokens,
                        at + ": input token IDs",
                    )
                    check(
                        row.get("text_sha256")
                        == hashlib.sha256(row.get("text", "").encode()).hexdigest(),
                        at + ": text hash",
                    )
                    text_chunks, token_chunks = (
                        row.get("chunks", []),
                        row.get("token_chunks", []),
                    )
                    check(
                        bool(text_chunks) and bool(token_chunks),
                        at + ": missing chunks",
                    )
                    check(
                        row.get("sse_events", 0)
                        >= max(len(text_chunks), len(token_chunks)),
                        at + ": SSE count",
                    )
                    if text_chunks and token_chunks:
                        times = [
                            chunk["received_after_send_seconds"]
                            for chunk in token_chunks
                        ]
                        check(
                            times == sorted(times) and times[0] >= 0,
                            at + ": token chunk times",
                        )
                        check(
                            sum(chunk["token_count"] for chunk in token_chunks)
                            == OUTPUT_TOKENS,
                            at + ": chunk token total",
                        )
                        compare_nested(
                            row,
                            {
                                "ttft_ms": text_chunks[0]["received_after_send_seconds"]
                                * 1000,
                                "token_ttft_ms": times[0] * 1000,
                                "tpot_ms": (times[-1] - times[0])
                                / (OUTPUT_TOKENS - 1)
                                * 1000,
                                "text_tpot_ms": (
                                    text_chunks[-1]["received_after_send_seconds"]
                                    - text_chunks[0]["received_after_send_seconds"]
                                )
                                / (OUTPUT_TOKENS - 1)
                                * 1000,
                                "request_seconds": row["ended_after_round_seconds"]
                                - row["started_after_round_seconds"],
                            },
                            check,
                            at,
                        )
                elapsed = round_record.get("summary", {}).get("wall_seconds", 0)
                check(elapsed > 0, label + ": elapsed missing/nonpositive")
                if rows and elapsed > 0 and all(r.get("success") for r in rows):
                    check(
                        max(r["ended_after_round_seconds"] for r in rows)
                        <= elapsed + 1e-7,
                        label + ": elapsed excludes a request",
                    )
                    recomputed = summarize(rows, elapsed)
                    compare_nested(
                        round_record["summary"], recomputed, check, label + "/summary"
                    )
                    result["rounds"].append(
                        {
                            "kind": kind,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "recomputed": recomputed,
                        }
                    )
                if kind == "rounds":
                    measured_total += len(rows)
                else:
                    warmup_total += len(rows)
        check(launch.get("all_successful") is True, f"{arm}: all_successful flag")
        result["metrics"] = {}
        for concurrency in CONCURRENCIES:
            matching = [
                r["recomputed"]
                for r in result["rounds"]
                if r["kind"] == "rounds" and r["concurrency"] == concurrency
            ]
            if len(matching) == 3:
                result["metrics"][str(concurrency)] = {
                    "output_tokens_per_second": distribution(
                        [r["output_tokens_per_second"] for r in matching]
                    ),
                    "ttft_p50_ms": distribution(
                        [r["ttft_ms"]["p50"] for r in matching]
                    ),
                    "tpot_p50_ms": distribution(
                        [r["tpot_ms"]["p50"] for r in matching]
                    ),
                }
    check(
        len(commands) == 4 and all(c == commands[0] for c in commands),
        "backend-normalized commands differ",
    )
    check(len(protocol_hashes) == 1, "protocols differ across launches")
    check(len(fixture_hashes) == 1, "fixture bytes differ across launches")
    check(len(runtime_hashes) == 1, "embedded runtime probe differs across launches")
    check(
        len(model_hashes) == 1 and None not in model_hashes,
        "model configuration differs across launches or hash missing",
    )
    check(
        len(client_hashes) == 1 and None not in client_hashes,
        "client script bytes differ across launches or hash missing",
    )
    client_path = Path(__file__).with_name("http_performance_client.py")
    if client_path.exists():
        report["expected_client_sha256"] = sha(client_path)
        check(client_hashes == {sha(client_path)}, "client script hash mismatch")
    check(
        [arm for arm, _ in sorted(launch_times, key=lambda item: item[1])]
        == list(ARMS),
        "launch timestamps do not follow ABBA",
    )
    check(
        measured_total == 384 and warmup_total == 64,
        "total measured/warmup request counts differ",
    )
    report["counts"] = {
        "measured_requests": measured_total,
        "warmup_requests": warmup_total,
    }
    for concurrency in CONCURRENCIES:
        comparisons = {}
        for metric in ("output_tokens_per_second", "ttft_p50_ms", "tpot_p50_ms"):
            try:
                values = {
                    arm: report["launches"][arm]["metrics"][str(concurrency)][metric][
                        "median"
                    ]
                    for arm in ARMS
                }
            except KeyError:
                continue
            throughput = metric == "output_tokens_per_second"
            gains = [
                (values[b] / values[a] - 1) * 100
                if throughput
                else (1 - values[b] / values[a]) * 100
                for a, b in (
                    ("triton-a1", "flashinfer-b1"),
                    ("triton-a2", "flashinfer-b2"),
                )
            ]
            triton = distribution([values["triton-a1"], values["triton-a2"]])
            fi = distribution([values["flashinfer-b1"], values["flashinfer-b2"]])
            overall_gain = (
                (fi["median"] / triton["median"] - 1) * 100
                if throughput
                else (1 - fi["median"] / triton["median"]) * 100
            )
            spread = max(triton["range"] / triton["median"], fi["range"] / fi["median"])
            round_spread = max(
                report["launches"][arm]["metrics"][str(concurrency)][metric]["range"]
                / values[arm]
                for arm in ARMS
            )
            noise = max(spread, round_spread) * 100
            consistent = all(g > 0 for g in gains) or all(g < 0 for g in gains)
            comparisons[metric] = {
                "triton_launch_medians": triton,
                "flashinfer_launch_medians": fi,
                "flashinfer_improvement_percent_of_launch_medians": overall_gain,
                "paired_improvement_percent_a1_b1_a2_b2": gains,
                "paired_direction_consistent": consistent,
                "observed_max_relative_launch_or_round_range_percent": noise,
                "descriptive_conclusion": (
                    "inconclusive_at_observed_variability"
                    if not consistent or abs(overall_gain) <= noise
                    else "consistent_descriptive_improvement"
                    if overall_gain > 0
                    else "consistent_descriptive_regression"
                ),
                "criterion_limit": (
                    "Observed-range comparison is a descriptive heuristic, "
                    "not a formal statistical confidence bound."
                ),
            }
        report["comparisons"][str(concurrency)] = comparisons
    report["validation_passed"] = not errors
    report["audit_script_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "validation_passed": not errors,
                "counts": report["counts"],
                "errors": errors,
                "comparisons": report["comparisons"],
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
