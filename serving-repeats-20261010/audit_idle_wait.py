"""Recompute the bounded between-launch GPU-idle gate from saved observations."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def audit(root):
    results = root / "results"
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "producer-manifest.json").read_text())
    expected_gpus = set(manifest["placement"]["gpu_uuids"])
    cid = (results / "container-id.txt").read_text().strip()
    raw = (results / "idle-wait.jsonl").read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    phases = ["initial"]
    for arm in protocol["order"]:
        phases.extend([f"before-{arm}", f"after-{arm}"])
    groups = []
    for record in records:
        if not groups or groups[-1][0] != record["phase"]:
            groups.append((record["phase"], []))
        groups[-1][1].append(record)
    assert [name for name, _ in groups] == phases, "Missing/reordered phase"
    summaries = []
    previous_time = None
    for phase, entries in groups:
        assert 2 <= len(entries) <= 40, (phase, "unbounded/incomplete wait")
        stable = 0
        for index, record in enumerate(entries, 1):
            assert record["container_id"] == cid, "Container identity changed"
            assert record["attempt"] == index, "Noncontiguous attempts"
            timestamp = datetime.fromisoformat(record["timestamp"])
            assert previous_time is None or timestamp >= previous_time
            previous_time = timestamp
            memory = list(csv.reader(record["memory_csv"].splitlines()))
            assert len(memory) == len(expected_gpus)
            assert {row[0].strip() for row in memory} == expected_gpus
            assert all(len(row) == 2 and row[1].strip().isdigit() for row in memory)
            compute = list(csv.reader(record["compute_csv"].splitlines()))
            owners = set(record["owned_pids"].split())
            assert owners and all(pid.isdigit() for pid in owners)
            for row in compute:
                assert len(row) == 3 and row[0].strip() in expected_gpus
                assert row[1].strip().isdigit()
            unmatched = any(row[1].strip() not in owners for row in compute)
            busy = bool(compute) or any(int(row[1]) > 16 for row in memory)
            assert record["busy"] is busy, "Recorded busy flag disagrees"
            assert record["unmatched"] is unmatched, "Recorded ownership disagrees"
            stable = 0 if busy else stable + 1
            assert stable < 2 or index == len(entries), "Ignored earlier success"
        assert stable == 2, (phase, "Missing two consecutive idle samples")
        summaries.append({
            "phase": phase,
            "observations": len(entries),
            "busy_observations": sum(row["busy"] for row in entries),
            "recorded_elapsed_seconds": (
                datetime.fromisoformat(entries[-1]["timestamp"])
                - datetime.fromisoformat(entries[0]["timestamp"])
            ).total_seconds(),
        })
    return {
        "status": "REVIEW_REQUIRED" if any(r["unmatched"] for r in records) else "PASS",
        "unmatched_observations": [r for r in records if r["unmatched"]],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "container_id": cid,
        "phases": summaries,
        "observations": len(records),
        "busy_observations": sum(row["busy"] for row in records),
        "limitations": ["NVML and docker-top snapshots are sequential, not atomic."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite an audit result"
    try:
        result = audit(args.root)
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "PASS" else 2 if result["status"] == "REVIEW_REQUIRED" else 1)


if __name__ == "__main__":
    main()
