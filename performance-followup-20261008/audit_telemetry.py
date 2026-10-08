"""Describe GPU telemetry within approximate remote serving round windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARMS = ("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2")
GPU_UUID = "GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675"
METRICS = {
    "sm_clock_mhz": ("clocks.current.sm", "clocks.sm"),
    "memory_clock_mhz": ("clocks.current.memory", "clocks.mem"),
    "temperature_c": ("temperature.gpu",),
    "power_w": ("power.draw",),
    "utilization_percent": ("utilization.gpu",),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normal_key(value):
    return re.sub(r"\s*\[[^]]*\]", "", value).strip().lower()


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def zone(offset):
    match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", offset)
    if not match:
        raise ValueError(f"Unsupported UTC offset: {offset}")
    sign = 1 if match[1] == "+" else -1
    return timezone(sign * timedelta(hours=int(match[2]), minutes=int(match[3])))


def csv_timestamp(value, local_zone):
    stripped = value.strip()
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return (
                datetime.strptime(stripped, pattern)
                .replace(tzinfo=local_zone)
                .astimezone(timezone.utc)
            )
        except ValueError:
            pass
    parsed = datetime.fromisoformat(stripped)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_zone)
    return parsed.astimezone(timezone.utc)


def numeric(value):
    if value is None:
        return None
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)", value)
    return float(match[1]) if match else None


def distribution(values):
    if not values:
        return {"samples": 0, "min": None, "median": None, "max": None}
    return {
        "samples": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def describe(samples):
    return {
        "samples": len(samples),
        **{
            name: distribution(
                [point[name] for point in samples if point[name] is not None]
            )
            for name in METRICS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--csv-utc-offset",
        help="Override only with evidence; default is the recorded container offset.",
    )
    args = parser.parse_args()
    csv_path = args.results / "gpu-telemetry.csv"
    times_path = args.results / "round-file-times.json"
    times = read(times_path)
    offset = args.csv_utc_offset or times["container_local_utc_offset"]
    local_zone = zone(offset)
    errors = []

    def check(condition, label):
        if not condition:
            errors.append(label)

    points = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            clean = {normal_key(key): value for key, value in row.items() if key}
            if not clean.get("timestamp"):
                continue
            check(
                clean.get("uuid", "").strip() == GPU_UUID,
                f"CSV row {index}: unauthorized/unexpected GPU UUID",
            )
            timestamp = csv_timestamp(clean["timestamp"], local_zone)
            points.append(
                {
                    "index": index,
                    "time": timestamp,
                    "time_utc": timestamp.isoformat(),
                    **{
                        name: numeric(
                            next((clean[key] for key in keys if key in clean), None)
                        )
                        for name, keys in METRICS.items()
                    },
                }
            )
    check(bool(points), "No telemetry points")
    check(
        [p["time"] for p in points] == sorted(p["time"] for p in points),
        "Telemetry timestamps are not monotonic",
    )
    intervals = [
        (right["time"] - left["time"]).total_seconds()
        for left, right in zip(points, points[1:])
    ]
    windows = []
    for row in times["rows"]:
        path = args.results / row["path"]
        check(path.exists(), f"Remote round file missing: {row['path']}")
        if not path.exists():
            continue
        result = read(path)
        check(sha(path) == row["sha256"], f"Round file hash: {row['path']}")
        for field in ("arm", "kind", "repeat", "concurrency"):
            check(row[field] == result[field], f"Round metadata/{field}: {path.name}")
        check(
            abs(row["wall_seconds"] - result["summary"]["wall_seconds"]) < 1e-9,
            f"Round elapsed: {path.name}",
        )
        start, end = utc(row["approx_start_utc"]), utc(row["approx_end_utc"])
        check(
            abs(end.timestamp() - row["mtime_ns"] / 1e9) < 1e-5,
            f"Round mtime timestamp: {path.name}",
        )
        check(
            abs((end - start).total_seconds() - row["wall_seconds"]) < 1e-5,
            f"Round start/end elapsed: {path.name}",
        )
        selected = [p for p in points if start <= p["time"] <= end]
        windows.append(
            {
                **{key: row[key] for key in ("arm", "kind", "repeat", "concurrency")},
                "path": row["path"],
                "approx_start_utc": start.isoformat(),
                "approx_end_utc": end.isoformat(),
                "summary": describe(selected),
                "selected_indexes": [p["index"] for p in selected],
            }
        )
    expected_matrix = {
        (arm, kind, concurrency, repeat)
        for arm in ARMS
        for kind, repeats in (("warmup", (0,)), ("measured", (1, 2, 3)))
        for concurrency in (1, 8)
        for repeat in repeats
    }
    actual_matrix = {
        (w["arm"], w["kind"], w["concurrency"], w["repeat"]) for w in windows
    }
    check(
        actual_matrix == expected_matrix and len(windows) == len(expected_matrix),
        "Incomplete/duplicated round window matrix",
    )
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for kind in ("warmup", "measured"):
            groups[arm][kind] = {}
            for concurrency in (1, 8):
                selected_windows = [
                    w
                    for w in windows
                    if w["arm"] == arm
                    and w["kind"] == kind
                    and w["concurrency"] == concurrency
                ]
                indexes = {i for w in selected_windows for i in w["selected_indexes"]}
                samples = [p for p in points if p["index"] in indexes]
                groups[arm][kind][str(concurrency)] = describe(samples)
                if kind == "measured":
                    check(bool(samples), f"{arm}/C{concurrency}: no measured samples")
    comparison = {}
    for concurrency in (1, 8):
        clock_medians = {
            arm: groups[arm]["measured"][str(concurrency)]["sm_clock_mhz"]["median"]
            for arm in ARMS
        }
        if not all(value is not None for value in clock_medians.values()):
            continue
        triton = statistics.median(
            [clock_medians["triton-a1"], clock_medians["triton-a2"]]
        )
        fi = statistics.median(
            [clock_medians["flashinfer-b1"], clock_medians["flashinfer-b2"]]
        )
        comparison[str(concurrency)] = {
            "measured_launch_sm_clock_medians_mhz": clock_medians,
            "flashinfer_vs_triton_clock_difference_percent": (fi / triton - 1) * 100,
            "paired_clock_difference_percent_a1_b1_a2_b2": [
                (clock_medians[b] / clock_medians[a] - 1) * 100
                for a, b in (
                    ("triton-a1", "flashinfer-b1"),
                    ("triton-a2", "flashinfer-b2"),
                )
            ],
            "inference_limit": (
                "Sampled clock differences are descriptive. Matching sampled "
                "medians cannot prove frequency or throttling was identical "
                "between samples, nor identify temperature as a cause."
            ),
        }
    report = {
        "validation_passed": not errors,
        "errors": errors,
        "gpu_uuid": GPU_UUID,
        "inputs": {
            "gpu_telemetry_sha256": sha(csv_path),
            "round_file_times_sha256": sha(times_path),
            "csv_utc_offset_used": offset,
            "csv_offset_override": args.csv_utc_offset,
            "container_local_datetime": times["container_local_datetime"],
            "container_timezone_names": times["container_timezone_names"],
        },
        "sampling": {
            "declared_period_seconds": 5,
            "observed_spacing_seconds": distribution(intervals),
            "total_samples": len(points),
        },
        "limitations": [
            times["limitation"],
            "Bare nvidia-smi timestamps use the recorded container UTC offset "
            "unless an evidence-backed explicit override is supplied.",
            "Telemetry samples every 5 seconds; these are approximate window "
            "descriptions and cannot establish absence of finer throttling.",
            "No clock, fan, power, or thermal control was changed by this audit. "
            "No causal attribution of performance to temperature is made.",
        ],
        "round_windows": windows,
        "launch_windows": groups,
        "clock_comparisons": comparison,
        "audit_script_sha256": sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "validation_passed",
                    "errors",
                    "sampling",
                    "clock_comparisons",
                )
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
