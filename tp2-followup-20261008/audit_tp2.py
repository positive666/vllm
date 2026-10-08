"""Read-only audit of TP2 fixed-token serving and sampled GPU ownership.

Use --results for the extracted remote result root, --source-manifest for the
frozen source-manifest.json, and --output for a new audit JSON. No GPU runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
ARMS = ("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2")
NATIVE = "3efc749cc7d32a0177376cdabc19d18f406ca2f5a002f933a48997882536d6d8"
SHAPE = {"H": 8, "HV": 24, "K": 128, "V": 128}


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
    p = (len(ordered) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)


def distribution(values):
    values = [v for v in values if v is not None]
    return {
        "samples": len(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "values": values,
    }


def option(command, key):
    return command[command.index(key) + 1]


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def nvidia_number(value):
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", value or "")
    return float(match[1]) if match else None


TAIL_INFO = (
    r"Check P2P Type isAllDirectP2p [01] directMode \d+ isAllCudaP2p [01]",
    r"\[Proxy Service(?: UDS)?\] Device \d+ CPU core \d+",
    r"proxy listening socket at [\d.]+<\d+>",
    r"ncclOsDlopen\(libnccl-(?:profiler|tuner)\.so\) failed: libnccl-(?:profiler|tuner)\.so: cannot open shared object file: No such file or directory",
    r"(?:PROFILER|TUNER)/Plugin: Could not find: libnccl-(?:profiler|tuner)\.so",
    r"threadThresholds [\d/| ]+",
    r"\d+ coll channels, \d+ collnet channels, \d+ nvls channels, \d+ p2p channels, \d+ p2p channels per peer",
    r"CC Off, workFifoBytes \d+",
    r"ncclCommInitRank comm 0x[0-9a-f]+ rank \d+ nranks 2 cudaDev \d+ nvmlDev \d+ busId [0-9a-f]+ commId 0x[0-9a-f]+ - Init COMPLETE",
    r"Init timings - ncclCommInitRank: rank \d+ nranks 2 total [\d.]+ \(kernels [\d.]+, alloc [\d.]+, bootstrap [\d.]+, allgathers [\d.]+, topo [\d.]+, graphs [\d.]+, connections [\d.]+, rest [\d.]+\)",
    r"Symmetric VA size=\d+GB",
    r"Channel \d+/\d+ : \d+\[\d+\] -> \d+\[\d+\] via P2P/CUMEM",
    r"Connected all rings, use ring PXN \d+ GDR \d+",
    r"misc/socket\.cc:\d+ \((?:socketProgress|socketWait|ncclSocketSend|ncclSocketTryRecv)\) -> 3",
)


def verify_log_runtime_messages(log, audit, label):
    """Classify only the exact caught optional DeepEP startup traceback."""
    lines = log.splitlines()
    prefix = re.compile(
        r"^\(APIServer pid=\d+\) WARNING \d\d-\d\d \d\d:\d\d:\d\d "
        r"\[import_utils\.py:418\] "
    )
    normalized = [prefix.sub("", line) for line in lines]
    ready = [i for i, line in enumerate(lines) if "Application startup complete" in line]
    requests = [i for i, line in enumerate(lines) if re.search(r'"POST |HTTP/\d', line)]
    startup_end = min(ready + requests) if ready or requests else -1
    accepted, unknown = [], []
    exact_sha = "464576bcbe1c0900fa6ede45895b3d988012375f5117375c872c865f5f35a6e4"
    for i, line in enumerate(normalized):
        if "Traceback (most recent call last)" not in line:
            continue
        end = i + 23
        block = "\n".join(normalized[i - 1:end + 1]) if i and end < len(lines) else ""
        block_sha = hashlib.sha256(block.encode("utf-8")).hexdigest()
        if (
            line == "Traceback (most recent call last):"
            and i > 0
            and normalized[i - 1] == "Module deep_ep was found but failed to import"
            and end < startup_end
            and block_sha == exact_sha
        ):
            accepted.append({
                "module_warning_line": i,
                "traceback_first_line": i + 1,
                "traceback_last_line": end + 1,
                "normalized_block_sha256": block_sha,
                "raw_block_sha256": hashlib.sha256(
                    "\n".join(lines[i - 1:end + 1]).encode("utf-8")
                ).hexdigest(),
                "terminal_error": "AssertionError: Cannot find package: nccl",
                "classification": "caught optional deep_ep startup import failure",
            })
        else:
            unknown.append({"line": i + 1, "text": lines[i]})
    runtime_errors = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(lines)
        if re.search(
            r"(?i)CUDA error|illegal memory access|NCCL.*(?:unhandled|error)|"
            r"NCCL\s+WARN|EngineDeadError|OutOfMemoryError|(?:^|\s)ERROR(?:\s|:)",
            line,
        )
    ]
    audit.check(not unknown, label + ": unclassified traceback " + repr(unknown))
    audit.check(not runtime_errors, label + ": engine/GPU/runtime error " + repr(runtime_errors))
    if accepted:
        audit.notes.append(
            f"{label}: {len(accepted)} exact DeepEP import traceback blocks before "
            "application readiness are retained and classified as caught optional "
            "dependency failures. import_utils._has_module catches Exception and "
            "returns False; this is not a claim that the whole log is warning-free "
            "or that NCCL transport is unavailable. Unknown tracebacks and actual "
            "CUDA/NCCL/worker/API errors still fail the audit."
        )
    return {
        "optional_startup_import_failures": accepted,
        "unclassified_tracebacks": unknown,
        "runtime_error_messages": runtime_errors,
        "source_contract": {
            "source_head": HEAD,
            "path": "vllm/utils/import_utils.py",
            "handler": "_has_module catches Exception, emits warning, returns False",
            "handler_first_line": 403,
        },
    }


def verify_log(path, record_path, recorded_sha, audit, label, *, manifest_override=None):
    """Accept only exact full bytes or a cryptographically bound closeout suffix."""
    audit.check(path.exists(), label + ": raw server log missing")
    if not path.exists():
        return {"mode": "missing"}
    data = path.read_bytes()
    actual_sha = hashlib.sha256(data).hexdigest()
    report = {"recorded_sha256": recorded_sha, "full_actual_sha256": actual_sha,
              "full_actual_bytes": len(data)}
    if actual_sha == recorded_sha:
        report["mode"] = "exact_full_log"
        return report
    closeout = path.parent / "log-closeout-manifest.json"
    audit.check(closeout.exists() or manifest_override is not None,
                label + ": changed log without closeout manifest")
    if not closeout.exists() and manifest_override is None:
        report["mode"] = "unverified_log_change"
        return report
    manifest = read(closeout) if manifest_override is None else manifest_override
    entry = manifest.get("entries", {}).get(path.name)
    audit.check(isinstance(entry, dict), label + ": no exact closeout entry")
    if not isinstance(entry, dict):
        report["mode"] = "unverified_log_change"
        return report
    before = len(audit.errors)
    # Independently scan newline boundaries rather than trusting the sidecar offset.
    digest_state, offset, matches = hashlib.sha256(), 0, []
    for line in data.splitlines(keepends=True):
        digest_state.update(line)
        offset += len(line)
        if line.endswith(b"\n") and digest_state.hexdigest() == recorded_sha:
            matches.append(offset)
    audit.check(len(matches) == 1, label + ": recorded SHA is not a unique newline prefix")
    if len(matches) != 1:
        report["mode"] = "unverified_log_change"
        return report
    prefix_bytes = matches[0]
    prefix, tail = data[:prefix_bytes], data[prefix_bytes:]
    tail_sha = hashlib.sha256(tail).hexdigest()
    try:
        tail_text = tail.decode("utf-8")
    except UnicodeDecodeError:
        audit.check(False, label + ": appended tail is not UTF-8")
        report["mode"] = "unverified_log_change"
        return report
    for key, expected in {
        "record_file": record_path.name,
        "original_record_sha256": sha(record_path),
        "recorded_prefix_bytes": prefix_bytes,
        "recorded_prefix_sha256": recorded_sha,
        "full_actual_bytes": len(data),
        "full_actual_sha256": actual_sha,
        "appended_tail_bytes": len(tail),
        "appended_tail_sha256": tail_sha,
        "tail_text": tail_text,
    }.items():
        audit.check(entry.get(key) == expected, label + ": closeout field/" + key)
    audit.check(bool(tail) and b"Application shutdown complete" in prefix,
                label + ": no nonempty post-shutdown closeout tail")
    audit.check(
        not re.search(r"(?i)CUDA error|illegal memory access|NCCL\s+(?:WARN|ERROR)|Traceback|EngineDeadError|OutOfMemoryError|POST |HTTP/|GDN", tail_text),
        label + ": runtime error/request/GDN content appended",
    )
    warning_line = (
        "/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: "
        "resource_tracker: There appear to be 1 leaked shared_memory objects "
        "to clean up at shutdown"
    )
    continuation = "warnings.warn('resource_tracker: There appear to be %d '"
    for index, line in enumerate(tail_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in (warning_line, continuation):
            continue
        # Two known fragments from concurrent stdout writers are accepted exactly.
        if index == 0 and stripped == "LER/Plugin: Could not find: libnccl-profiler.so":
            continue
        if stripped == "ared object file: No such file or directory":
            audit.check(
                b"ncclOsDlopen(libnccl-profiler.so) failed: "
                b"libnccl-profiler.so: cannot open sh" in prefix,
                label + ": orphan profiler-dlopen fragment lacks prefix context",
            )
            continue
        if re.fullmatch(
            r"\[?1\] NCCL INFO proxy listening socket at 172\.17\.0\.11<\d+>", stripped
        ):
            continue
        if stripped == (
            "NCCL INFO ncclOsDlopen(libnccl-tuner.so) failed: "
            "libnccl-tuner.so: cannot open shared object file: No such file or directory"
        ):
            continue
        match = re.fullmatch(r"(?:[0-9a-f]+:\d+:\d+ \[\d+\]|\]) NCCL INFO (.+)", stripped)
        audit.check(
            match is not None and any(re.fullmatch(pattern, match[1]) for pattern in TAIL_INFO),
            label + ": unrecognized appended message: " + stripped,
        )
    report.update({
        "mode": "verified_newline_prefix_with_exact_closeout_manifest"
        if len(audit.errors) == before else "unverified_log_change",
        "recorded_prefix_bytes": prefix_bytes,
        "appended_tail_bytes": len(tail), "appended_tail_sha256": tail_sha,
        "closeout_manifest_sha256": sha(closeout) if manifest_override is None else None,
        "candidate_closeout_manifest_sha256": digest(manifest_override)
        if manifest_override is not None else None,
        "tail_classification": "late buffered NCCL informational output and resource_tracker shutdown cleanup",
    })
    if len(audit.errors) == before:
        audit.notes.append(
            f"{label}: recorded log SHA verifies the first {prefix_bytes} bytes; "
            f"{len(tail)} appended bytes separately match the closeout manifest. "
            "Tail contains allowed late buffered NCCL informational output and "
            "resource_tracker shutdown cleanup; it is not described as only teardown. "
            "Original record/raw bytes were not changed."
        )
    return report


class Audit:
    def __init__(self):
        self.errors, self.notes = [], []

    def check(self, condition, label):
        if not condition:
            self.errors.append(label)

    def number(self, observed, expected, label, abs_tol=1e-8):
        self.check(
            type(observed) in (float, int)
            and math.isfinite(observed)
            and math.isclose(observed, expected, rel_tol=1e-9, abs_tol=abs_tol),
            f"{label}: got {observed!r}, expected {expected!r}",
        )

    def nested(self, observed, expected, label):
        if isinstance(expected, dict):
            self.check(isinstance(observed, dict), label + ": missing object")
            for key, value in expected.items():
                self.nested((observed or {}).get(key), value, label + "/" + key)
        elif type(expected) in (int, float):
            self.number(observed, expected, label)
        else:
            self.check(observed == expected, label + ": mismatch")

    def round(self, result, expected_count, tokens, label):
        rows = result.get("records", [])
        self.check(len(rows) == expected_count, label + ": request count")
        self.check(
            {r.get("index") for r in rows} == set(range(expected_count)),
            label + ": request indices",
        )
        metrics = {"ttft_ms": [], "tpot_ms": [], "request_seconds": []}
        for row in rows:
            at = label + f"/request{row.get('index')}"
            self.check(row.get("success") is True and not row.get("error"), at + ": success")
            self.check(row.get("http_status") == 200 and row.get("done") is True, at + ": HTTP/DONE")
            self.check(row.get("finish_reason") == "length", at + ": finish reason")
            self.check(
                row.get("usage") == {
                    "prompt_tokens": 512, "completion_tokens": 128, "total_tokens": 640,
                }, at + ": usage",
            )
            ids = row.get("output_token_ids", [])
            self.check(
                len(ids) == 128 and all(type(t) is int and t >= 0 for t in ids),
                at + ": output IDs",
            )
            self.check(row.get("output_token_ids_sha256") == digest(ids), at + ": IDs hash")
            self.check(row.get("prompt_token_ids") == tokens, at + ": prompt IDs")
            self.check(
                row.get("text_sha256")
                == hashlib.sha256(row.get("text", "").encode()).hexdigest(),
                at + ": text hash",
            )
            text, chunks = row.get("chunks", []), row.get("token_chunks", [])
            self.check(bool(text) and bool(chunks), at + ": chunk data missing")
            if not text or not chunks:
                continue
            token_times = [c["received_after_send_seconds"] for c in chunks]
            text_times = [c["received_after_send_seconds"] for c in text]
            self.check(
                token_times == sorted(token_times) and text_times == sorted(text_times)
                and token_times[0] >= 0 and text_times[0] >= 0,
                at + ": monotonic chunk times",
            )
            self.check(sum(c["token_count"] for c in chunks) == 128, at + ": chunk token total")
            self.check(row.get("sse_events", 0) >= max(len(text), len(chunks)), at + ": SSE count")
            recalculated = {
                "ttft_ms": 1000 * text_times[0],
                "tpot_ms": 1000 * (token_times[-1] - token_times[0]) / 127,
                "request_seconds": row["ended_after_round_seconds"]
                - row["started_after_round_seconds"],
            }
            self.nested(row, recalculated, at)
            self.number(row.get("token_ttft_ms"), 1000 * token_times[0], at + "/token_ttft")
            self.number(
                row.get("text_tpot_ms"), 1000 * (text_times[-1] - text_times[0]) / 127,
                at + "/text_tpot",
            )
            self.check(recalculated["request_seconds"] >= token_times[-1], at + ": duration")
            self.check(
                type(row.get("worker")) is int
                and 0 <= row["worker"] < result["concurrency"], at + ": worker",
            )
            for name in metrics:
                metrics[name].append(recalculated[name])
        elapsed = result.get("summary", {}).get("wall_seconds", 0)
        self.check(elapsed > 0, label + ": missing elapsed")
        if not rows or elapsed <= 0 or any(len(v) != len(rows) for v in metrics.values()):
            return None
        last = max(r["ended_after_round_seconds"] for r in rows)
        self.check(last <= elapsed + 1e-7, label + ": elapsed excludes requests")
        summary = {
            "requests": len(rows), "successful": len(rows), "failed": 0,
            "wall_seconds": elapsed, "output_tokens": 128 * len(rows),
            "output_tokens_per_second": 128 * len(rows) / elapsed,
            "requests_per_second": len(rows) / elapsed,
            **{name: {"p50": percentile(v, 0.5), "p95": percentile(v, 0.95),
                       "mean": statistics.mean(v)} for name, v in metrics.items()},
        }
        self.nested(result.get("summary"), summary, label + "/summary")
        return summary


def host_samples(path, expected_gpus, audit):
    """Parse host monitor UUID/PID snapshots; unrelated GPUs are permitted."""
    audit.check(path.exists(), "host telemetry missing")
    if not path.exists():
        return []
    samples = []
    blocks = path.read_text(encoding="utf-8", errors="replace").split("ROUND_END")
    if blocks[-1].strip():
        audit.notes.append("Incomplete final host monitor block retained but excluded.")
    for block in blocks[:-1]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        try:
            stamp = utc(lines[0])
        except ValueError:
            audit.notes.append("Host monitor non-snapshot output retained: " + lines[0][:200])
            continue
        section, point = "gpu", {"time_utc": stamp.isoformat(), "gpus": {}, "apps": [], "owned_pids": []}
        for line in lines[1:]:
            if line in ("COMPUTE_PROCESSES", "OWNED_CONTAINER_PIDS"):
                section = "apps" if line == "COMPUTE_PROCESSES" else "owned"
                continue
            if section == "owned":
                match = re.match(r"^\s*(\d+)\s+", line)
                if match:
                    point["owned_pids"].append(int(match[1]))
                continue
            fields = next(csv.reader([line], skipinitialspace=True))
            if section == "gpu" and len(fields) == 8 and fields[1].startswith("GPU-"):
                index, uuid, util, memory, temp, sm, mem_clock, power = fields
                if uuid in expected_gpus:
                    point["gpus"][uuid] = {
                        "host_index": int(index), "utilization_percent": nvidia_number(util),
                        "memory_used_mib": nvidia_number(memory), "temperature_c": nvidia_number(temp),
                        "sm_clock_mhz": nvidia_number(sm), "memory_clock_mhz": nvidia_number(mem_clock),
                        "power_w": nvidia_number(power),
                    }
            elif section == "apps" and len(fields) >= 2 and fields[0].startswith("GPU-"):
                if fields[0] in expected_gpus:
                    point["apps"].append({"uuid": fields[0], "pid": int(fields[1])})
        audit.check(set(point["gpus"]) == set(expected_gpus), "host snapshot missing selected GPU")
        owned = set(point["owned_pids"])
        for app in point["apps"]:
            audit.check(
                app["pid"] in owned,
                f"foreign/unattributed selected-GPU PID {app['pid']} at {point['time_utc']}",
            )
        samples.append(point)
    audit.check(bool(samples), "host monitor has no parsed snapshots")
    audit.check(
        [s["time_utc"] for s in samples] == sorted(s["time_utc"] for s in samples),
        "host monitor timestamps not monotonic",
    )
    return samples


def describe_host_windows(root, protocol, launches, audit):
    uuids = protocol["gpu_uuids"]
    points = host_samples(root / "host-telemetry.log", uuids, audit)
    report = {
        "raw_sha256": sha(root / "host-telemetry.log") if points else None,
        "samples": len(points), "selected_gpu_uuids": uuids, "launch_windows": {},
        "sampled_foreign_pid_count": sum(
            app["pid"] not in set(p["owned_pids"]) for p in points for app in p["apps"]
        ),
        "limitation": "5-second snapshots verify sampled PID ownership only; activity and throttling between samples remain unknown.",
    }
    fields = ("sm_clock_mhz", "memory_clock_mhz", "temperature_c",
              "power_w", "utilization_percent", "memory_used_mib")
    for arm, server in launches.items():
        start = utc(server["ready_utc"])
        end = utc(server["finished_utc"])
        selected = [p for p in points if start <= utc(p["time_utc"]) <= end]
        audit.check(bool(selected), arm + ": no host samples in serving window")
        report["launch_windows"][arm] = {
            "ready_utc": start.isoformat(), "finished_utc": end.isoformat(),
            "includes_warmup_and_cleanup": True,
            "gpus": {g: {f: distribution([p["gpus"][g][f] for p in selected if g in p["gpus"]])
                         for f in fields} for g in uuids},
            "observed_compute_pids": sorted({a["pid"] for p in selected for a in p["apps"]}),
        }
    topology = {}
    names = (
        ("host-topology.txt",),
        ("container-gpu-topology.txt", "gpu-topology.txt"),
        ("container-gpu-topology-p2p-read.txt", "gpu-topology-p2p-read.txt"),
        ("container-gpu-topology-p2p-write.txt", "gpu-topology-p2p-write.txt"),
    )
    for choices in names:
        path = next((root / name for name in choices if (root / name).exists()), root / choices[0])
        name = path.name
        audit.check(path.exists() and bool(path.read_text().strip()), "topology missing: " + name)
        if path.exists():
            topology[name] = {"sha256": sha(path), "text": path.read_text()}
    report["topology"] = topology
    # Preflight CSV schema is retained raw. Header-based CSV is independently checked.
    baseline_path = root / "host-preflight-gpu.csv"
    process_path = root / "host-preflight-processes.csv"
    audit.check(baseline_path.exists() and process_path.exists(), "host preflight raw files missing")
    if baseline_path.exists():
        report["baseline_gpu_csv_sha256"] = sha(baseline_path)
        baseline_lines = baseline_path.read_text().splitlines()
        report["baseline_gpu_csv"] = baseline_lines
        if baseline_lines and "uuid" in baseline_lines[0].lower():
            gpu_rows = list(csv.DictReader(baseline_lines, skipinitialspace=True))
            for g in uuids:
                matches = [r for r in gpu_rows if r.get("uuid", "").strip() == g]
                audit.check(len(matches) == 1, g + ": baseline UUID")
                if matches:
                    cleaned = {re.sub(r"\s*\[[^]]*\]", "", k).strip(): v for k, v in matches[0].items()}
                    memory = nvidia_number(cleaned.get("memory.used"))
                    util = nvidia_number(cleaned.get("utilization.gpu"))
                    audit.check(memory is not None and memory <= 32, g + ": baseline memory")
                    audit.check(util == 0, g + ": baseline utilization")
        else:
            audit.notes.append("Baseline GPU CSV lacks headers; verify documented query order before claiming independent idle validation.")
    if process_path.exists():
        report["baseline_processes_sha256"] = sha(process_path)
        occupied = [line for line in process_path.read_text().splitlines()
                    if any(g in line for g in uuids)]
        audit.check(not occupied, "selected GPUs have preflight compute processes")
    times_path = root / "round-file-times.json"
    audit.check(times_path.exists(), "original remote round times missing")
    if times_path.exists():
        times = read(times_path)
        report["round_times_sha256"] = sha(times_path)
        windows = []
        for timing in times["rows"]:
            if timing["arm"] not in ARMS or timing["kind"] not in ("warmup", "measured"):
                continue
            path = root / timing["path"]
            audit.check(path.exists() and sha(path) == timing["sha256"], "round time hash: " + str(path))
            if not path.exists():
                continue
            result = read(path)
            for field in ("arm", "kind", "repeat", "concurrency"):
                audit.check(timing[field] == result[field], "round time field: " + field)
            start, end = utc(timing["approx_start_utc"]), utc(timing["approx_end_utc"])
            audit.number((end - start).total_seconds(), result["summary"]["wall_seconds"],
                         "round window duration", abs_tol=1e-5)
            audit.number(end.timestamp(), timing["mtime_ns"] / 1e9, "round time mtime", abs_tol=1e-5)
            selected = [p for p in points if start <= utc(p["time_utc"]) <= end]
            windows.append({
                **{key: timing[key] for key in ("arm", "kind", "repeat", "concurrency")},
                "approx_start_utc": start.isoformat(), "approx_end_utc": end.isoformat(),
                "gpus": {g: {f: distribution([p["gpus"][g][f] for p in selected if g in p["gpus"]])
                             for f in fields} for g in uuids},
            })
        observed = [(w["arm"], w["kind"], w["repeat"], w["concurrency"]) for w in windows]
        expected = {(a, k, r, c) for a in ARMS for c in (1, 8)
                    for k, repeats in (("warmup", (0,)), ("measured", (1, 2, 3))) for r in repeats}
        audit.check(len(observed) == 32 and set(observed) == expected, "complete measured/warmup telemetry matrix")
        report["round_windows"] = windows
        report["measured_clock_temperature"] = {}
        for arm in ARMS:
            report["measured_clock_temperature"][arm] = {}
            for c in (1, 8):
                chosen = [w for w in windows if w["arm"] == arm
                          and w["kind"] == "measured" and w["concurrency"] == c]
                report["measured_clock_temperature"][arm][str(c)] = {
                    g: {f: distribution([v for w in chosen for v in w["gpus"][g][f]["values"]])
                        for f in fields} for g in uuids
                }
                for g in uuids:
                    audit.check(report["measured_clock_temperature"][arm][str(c)][g]["sm_clock_mhz"]["samples"] > 0,
                                f"{arm}/C{c}/{g}: no measured host clock samples")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit, root = Audit(), args.results
    protocol, manifest = read(root / "protocol.json"), read(args.source_manifest)
    audit.check(protocol.get("source_head") == HEAD == manifest.get("source_head"), "source head binding")
    audit.check(protocol.get("tensor_parallel_size") == 2, "frozen protocol TP2")
    audit.check(protocol.get("per_rank_gdn_shape") == SHAPE, "frozen per-rank shape")
    gpu_uuids = protocol.get("gpu_uuids", [])
    audit.check(len(gpu_uuids) == 2 and len(set(gpu_uuids)) == 2, "two unique UUIDs required")
    blocks, capacity = protocol["gpu_blocks"], protocol["expected_kv_capacity_tokens"]
    audit.check(type(blocks) is int and blocks > 0 and capacity >= 8 * 640, "frozen capacity")
    for key, value in {
        "order": list(ARMS), "input_tokens": 512, "output_tokens": 128,
        "requests_per_round": 16, "warmup_per_concurrency": 8, "rounds": 3,
        "concurrency": [1, 8], "measured_total": 384, "warmup_total": 64,
    }.items():
        audit.check(protocol["serving"].get(key) == value, "frozen serving/" + key)
    hashes = {name: item["sha256"] for name, item in manifest["files"].items()}
    report = {"source_head": HEAD, "tensor_parallel_size": 2,
              "protocol_sha256": sha(root / "protocol.json"), "launches": {},
              "comparisons": {}, "compatibility": {}, "errors": audit.errors, "notes": audit.notes}
    fixture = root / "http-fixture.json"
    tokens = read(fixture)["input_tokens"]
    audit.check(len(tokens) == 512 and all(type(t) is int and t >= 0 for t in tokens), "fixture tokens")
    audit.check(protocol.get("fixture_sha256") == sha(fixture), "frozen fixture hash")
    expected_protocol = protocol["http_protocol"]
    for key, value in {
        "input_tokens": 512, "output_tokens": 128, "min_tokens": 128,
        "concurrency": [1, 8], "repeats": 3, "temperature": 0, "seed": 42,
        "ignore_eos": True, "stream": True, "return_token_ids": True,
    }.items():
        audit.check(expected_protocol.get(key) == value, "frozen HTTP workload: " + key)
    audit.check(protocol["http_protocol_sha256"] == digest(expected_protocol), "frozen HTTP protocol hash")
    commands, times, servers = [], [], {}
    shared = {k: set() for k in ("client", "model", "runtime", "supervisor")}
    measured = warmup = 0
    for arm in (*ARMS, "triton-compat", "flashinfer-compat"):
        measured_arm = arm in ARMS
        backend = arm.split("-")[0]
        client_path = root / "http" / ((arm if measured_arm else "compat-" + backend) + ".json")
        server_path = root / ("serve-" + arm + ".json")
        client, server = read(client_path), read(server_path)
        audit.check(server.get("arm") == arm and server.get("backend") == backend, arm + ": identity")
        audit.check(server.get("status") == "completed", arm + ": server not completed")
        audit.check(server.get("source_head") == HEAD and server.get("source_files") == hashes, arm + ": source")
        audit.check(server.get("tensor_parallel_size") == 2 and server.get("per_rank_gdn_shape") == SHAPE, arm + ": TP2 shape")
        audit.check(server.get("gpu_uuids") == gpu_uuids, arm + ": UUID selection")
        audit.check(server.get("gpu_blocks") == blocks and server.get("kv_capacity_tokens") == capacity, arm + ": KV capacity")
        audit.check(server.get("state_dtype") == "float32", arm + ": state dtype")
        audit.check(server.get("client_result_sha256") == sha(client_path), arm + ": client result hash")
        embedded = client.get("server_evidence", {})
        for key in ("arm", "backend", "command", "source_head", "source_files", "tensor_parallel_size",
                    "per_rank_gdn_shape", "gpu_uuids", "gpu_blocks", "state_dtype", "model_config_sha256", "runtime"):
            audit.check(embedded.get(key) == server.get(key), arm + ": embedded evidence/" + key)
        command = server["command"]
        try:
            for key, value in {
                "--tensor-parallel-size": "2", "--mamba-ssm-cache-dtype": "float32",
                "--num-gpu-blocks-override": str(blocks), "--dtype": "bfloat16",
                "--max-model-len": "2048", "--max-num-seqs": "32",
                "--max-num-batched-tokens": "1024", "--seed": "42",
            }.items():
                audit.check(option(command, key) == value, arm + ": command " + key)
            audit.check("--no-enable-prefix-caching" in command, arm + ": prefix caching")
            audit.check(
                json.loads(option(command, "--kernel-config"))
                == {"gdn_decode_backend": backend, "linear_backend": "marlin"}, arm + ": kernel config",
            )
            audit.check(
                json.loads(option(command, "--attention-config"))
                == {"backend": "FLASH_ATTN", "flash_attn_version": 2}, arm + ": attention",
            )
            normalized = list(command)
            normalized[normalized.index("--kernel-config") + 1] = "<backend>"
            commands.append(normalized)
        except (ValueError, IndexError, TypeError) as exc:
            audit.check(False, arm + ": malformed command " + str(exc))
        runtime = server["runtime"]
        audit.check(runtime.get("declared_source_head") == HEAD, arm + ": runtime head")
        audit.check(runtime.get("torch", {}).get("gpu_count") == 2, arm + ": runtime visible device count")
        audit.check(
            runtime.get("modules", {}).get("vllm._C_stable_libtorch", {}).get("sha256") == NATIVE,
            arm + ": native binary identity",
        )
        for key, value in (("model", server.get("model_config_sha256")),
                           ("runtime", digest(runtime)), ("supervisor", server.get("supervisor_sha256"))):
            shared[key].add(value)
        log_path = server_path.with_suffix(".log")
        log_integrity = verify_log(log_path, server_path, server.get("server_log_sha256"), audit, arm)
        runtime_messages = {"mode": "missing_log"}
        if log_path.exists():
            log = log_path.read_text(encoding="utf-8", errors="replace")
            capacities = {int(n.replace(",", "")) for n in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)}
            audit.check(capacities == {capacity}, arm + ": actual log capacity")
            runtime_messages = verify_log_runtime_messages(log, audit, arm)
        audit.check(client.get("protocol") == expected_protocol, arm + ": shared HTTP protocol")
        audit.check(client.get("protocol_sha256") == digest(expected_protocol), arm + ": HTTP protocol hash")
        audit.check(client.get("fixture_sha256") == sha(fixture), arm + ": fixture bytes")
        audit.check(client.get("all_successful") is True, arm + ": client success flag")
        result = {"client_sha256": sha(client_path), "server_sha256": sha(server_path),
                  "log_integrity": log_integrity, "runtime_messages": runtime_messages,
                  "rounds": [], "metrics": {}}
        if measured_arm:
            shared["client"].add(client.get("client_sha256"))
            audit.check(client.get("client_sha256") == protocol["http_client_sha256"], arm + ": frozen HTTP client")
            audit.check(server.get("protocol_sha256") == sha(root / "protocol.json"), arm + ": frozen protocol bytes")
            report["launches"][arm], servers[arm] = result, server
            times.append((arm, utc(server["started_utc"])))
            specs = (("warmup", "warmup", 8, (0,)), ("rounds", "measured", 16, (1, 2, 3)))
        else:
            reference = protocol["compatibility_evidence"][backend]
            audit.check(reference["server_sha256"] == sha(server_path), arm + ": frozen compat server")
            audit.check(reference["client_sha256"] == sha(client_path), arm + ": frozen compat client")
            report["compatibility"][backend] = result
            specs = (("rounds", "compatibility", 8, (0,)),)
        for field, kind, count, repeats in specs:
            rounds = client.get(field, [])
            actual = [(r.get("concurrency"), r.get("repeat")) for r in rounds]
            expected = {(c, r) for c in (1, 8) for r in repeats}
            audit.check(set(actual) == expected and len(actual) == len(expected), arm + "/" + kind + ": round matrix")
            for round_result in rounds:
                c, repeat = round_result["concurrency"], round_result["repeat"]
                label = f"{arm}/{kind}/C{c}/r{repeat}"
                audit.check(round_result.get("arm") == arm and round_result.get("kind") == kind, label + ": round identity")
                audit.check(round_result.get("input_tokens_sha256") == digest(tokens), label + ": prompt hash")
                audit.check(round_result.get("protocol_sha256") == digest(expected_protocol), label + ": protocol hash")
                standalone = root / "http" / f"{arm}-{kind}-c{c}-r{repeat}.json"
                audit.check(standalone.exists() and read(standalone) == round_result, label + ": standalone file")
                summary = audit.round(round_result, count, tokens, label)
                if summary:
                    result["rounds"].append({"kind": kind, "concurrency": c, "repeat": repeat, "summary": summary})
                if kind == "measured":
                    measured += len(round_result["records"])
                elif kind == "warmup":
                    warmup += len(round_result["records"])
        for c in (1, 8):
            summaries = [r["summary"] for r in result["rounds"] if r["kind"] == "measured" and r["concurrency"] == c]
            if len(summaries) == 3:
                result["metrics"][str(c)] = {
                    "output_tokens_per_second": distribution([r["output_tokens_per_second"] for r in summaries]),
                    "ttft_p50_ms": distribution([r["ttft_ms"]["p50"] for r in summaries]),
                    "tpot_p50_ms": distribution([r["tpot_ms"]["p50"] for r in summaries]),
                }
    audit.check(len(commands) == 6 and all(c == commands[0] for c in commands), "commands differ beyond backend")
    audit.check([arm for arm, stamp in sorted(times, key=lambda x: x[1])] == list(ARMS), "actual launch order")
    audit.check(len({stamp for arm, stamp in times}) == 4, "distinct launch timestamps")
    for key, values in shared.items():
        audit.check(len(values) == 1 and None not in values, "shared " + key + " mismatch")
    audit.check((measured, warmup) == (384, 64), "measured/warmup totals")
    report["counts"] = {"measured": measured, "warmup": warmup, "compatibility": 32}
    for c in (1, 8):
        report["comparisons"][str(c)] = {}
        for metric in ("output_tokens_per_second", "ttft_p50_ms", "tpot_p50_ms"):
            groups = {a: report["launches"][a]["metrics"].get(str(c), {}).get(metric) for a in ARMS}
            if not all(groups.values()):
                audit.check(False, f"C{c}/{metric}: incomplete measurements")
                continue
            values = {a: d["median"] for a, d in groups.items()}
            triton = distribution([values["triton-a1"], values["triton-a2"]])
            fi = distribution([values["flashinfer-b1"], values["flashinfer-b2"]])
            def change(a, b):
                return 100 * (b / a - 1) if metric == "output_tokens_per_second" else 100 * (1 - b / a)
            report["comparisons"][str(c)][metric] = {
                "triton_launch_medians": triton, "flashinfer_launch_medians": fi,
                "flashinfer_improvement_percent": change(triton["median"], fi["median"]),
                "paired_improvement_percent": [
                    change(values[a], values[b]) for a, b in ((ARMS[0], ARMS[1]), (ARMS[3], ARMS[2]))
                ],
                "aggregation": "median of 3 per-round metrics per launch, then median of 2 launch medians per backend",
            }
    report["host_placement_and_telemetry"] = describe_host_windows(root, protocol, servers, audit)
    report["validation_passed"] = not audit.errors
    report["audit_script_sha256"] = sha(Path(__file__))
    report["limits"] = [
        "Two independent server launches per backend; no significance test or pooled-request confidence interval.",
        "Single fixed synthetic prompt at C1/C8; performance/compatibility is not accuracy evaluation.",
        "TP2 on L20 does not establish Hopper, BF16 state, full CUDA13 build, or all-workload support.",
        "Host sampled windows include warmup/cleanup; use round windows for measured-only clock comparisons.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("validation_passed", "errors", "notes", "counts", "comparisons")}, indent=2))
    raise SystemExit(0 if report["validation_passed"] else 1)


if __name__ == "__main__":
    main()
