"""Check each observed GPU PID against its sample's owned-container PID list.

GPU query and docker top are sequential, not atomic. Unmatched observations
remain visible. A PID seen in a neighboring owned-container sample is marked
as a possible sampling race; it is never promoted to same-sample confirmation.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import traceback
from pathlib import Path

STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
COMPUTE = re.compile(r"(GPU-[0-9a-fA-F-]{36}),\s*(\d+),\s*([^,]+)")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_samples(text):
    samples = []
    current = None
    after_header = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if STAMP.fullmatch(line):
            if current is not None:
                require(after_header, "Every sample has an actual docker top PID header")
            if samples:
                require(line > samples[-1]["utc"], "Strictly increasing sample timestamps")
            datetime.datetime.fromisoformat(line)
            current = {"utc": line, "gpu_compute": [], "owned_container_pids": []}
            samples.append(current)
            after_header = False
            continue
        require(current is not None, "Timestamp precedes process observations")
        if line == "PID":
            require(not after_header, "Exactly one docker top header per sample")
            after_header = True
            continue
        if after_header:
            require(line.isdigit() and int(line) > 0, "Only PID values from docker top")
            current["owned_container_pids"].append(int(line))
        else:
            match = COMPUTE.fullmatch(line)
            require(match is not None, "Only expected GPU PID rows before docker top")
            uuid, pid, memory = match.groups()
            require(int(pid) > 0, "Positive compute PID")
            current["gpu_compute"].append({"gpu_uuid": uuid, "pid": int(pid),
                                           "used_gpu_memory": memory.strip()})
    require(bool(samples) and after_header, "Complete nonempty process telemetry")
    for sample in samples:
        pids = sample["owned_container_pids"]
        require(bool(pids) and len(pids) == len(set(pids)), "Nonempty unique container PIDs")
        observations = [(row["gpu_uuid"], row["pid"]) for row in sample["gpu_compute"]]
        require(len(observations) == len(set(observations)), "Unique GPU/PID observations")
    return samples


def classify_samples(samples):
    owners = {}
    for ordinal, sample in enumerate(samples):
        for pid in sample["owned_container_pids"]:
            owners.setdefault(pid, []).append(ordinal)
    observations = 0
    confirmed = 0
    mismatches = []
    for ordinal, sample in enumerate(samples):
        current_pids = set(sample["owned_container_pids"])
        for row in sample["gpu_compute"]:
            observations += 1
            if row["pid"] in current_pids:
                confirmed += 1
                continue
            known = owners.get(row["pid"], [])
            nearby = [item for item in known if abs(item - ordinal) == 1]
            if nearby:
                category = "possible_adjacent_sample_spawn_or_exit_race"
                candidates = nearby
            elif known:
                category = "seen_in_nonadjacent_owned_sample_not_same_sample_confirmed"
                candidates = known
            else:
                category = "never_observed_in_owned_container_requires_investigation"
                candidates = []
            mismatches.append({
                "utc": sample["utc"], **row, "category": category,
                "candidate_owned_sample_utc": [samples[item]["utc"] for item in candidates],
            })
    unknown = [item for item in mismatches if item["category"] ==
               "never_observed_in_owned_container_requires_investigation"]
    return {
        "strict_same_sample_ownership_pass": not mismatches,
        "sample_count": len(samples), "compute_pid_observations": observations,
        "same_sample_confirmed_observations": confirmed,
        "unmatched_same_sample_observations": len(mismatches),
        "possible_adjacent_sample_race_observations": sum(
            item["category"] == "possible_adjacent_sample_spawn_or_exit_race"
            for item in mismatches),
        "unknown_ownership_observations": len(unknown),
        "unknown_ownership_pids": sorted({item["pid"] for item in unknown}),
        "mismatches": mismatches,
        "scope": (
            "Sampled PIDs only. Sequential GPU and container queries are not atomic; "
            "candidate race labels are observations, not proof of ownership at that instant. "
            "Unobserved short-lived activity and PID reuse between samples remain possible."
        ),
    }


def audit_run(run):
    process_path = run.parent / (run.name + "-processes.log")
    host_path = run.parent / (run.name + "-host.csv")
    exit_path = run.parent / (run.name + ".exit")
    require(exit_path.read_text().strip() == "0", "Completed actual run with exit 0")
    samples = parse_samples(process_path.read_text(encoding="utf-8"))
    runtime_path = run / "runtime-after.json"
    runtime = read_json(runtime_path)
    workers = runtime["worker_ranks"]
    uuids = {"GPU-" + row["gpu_uuid"].removeprefix("GPU-") for row in workers}
    require(len(uuids) == 2, "Two actual experiment GPUs")
    with host_path.open(encoding="utf-8", newline="") as handle:
        host = list(csv.DictReader(handle, skipinitialspace=True))
    groups = {}
    for row in host:
        groups.setdefault(row["utc"], []).append(row)
    require(list(groups) == [item["utc"] for item in samples],
            "Every process sample matches its GPU telemetry timestamp")
    for sample in samples:
        gpu_rows = groups[sample["utc"]]
        require(len(gpu_rows) == 2 and {row["uuid"] for row in gpu_rows} == uuids,
                "Two actual GPUs at every sample")
        require(all(row["gpu_uuid"] in uuids for row in sample["gpu_compute"]),
                "Observed compute rows belong to the experiment GPU scope")
    result = classify_samples(samples)
    return {"run": run.name, "input_sha256": {
        process_path.name: sha(process_path), host_path.name: sha(host_path),
        exit_path.name: sha(exit_path), "runtime-after.json": sha(runtime_path),
    }, "gpu_uuids": sorted(uuids), **result}


def audit(args):
    require(args.container_name == "gdn60403-late-v2-20261009", "Agent-owned namespace")
    monitor_text = args.monitor.read_text(encoding="utf-8")
    require("docker top " + args.container_name + " -eo pid" in monitor_text,
            "Bound monitor obtains only PIDs from this owned container")
    results = [audit_run(run) for run in args.runs]
    require(len({item["run"] for item in results}) == len(results), "Unique run names")
    unknown = sorted({pid for result in results for pid in result["unknown_ownership_pids"]})
    return {
        "integrity_pass": True, "auditor_sha256": sha(Path(__file__)),
        "monitor_sha256": sha(args.monitor), "owned_container_name": args.container_name,
        "strict_same_sample_ownership_pass": all(
            item["strict_same_sample_ownership_pass"] for item in results),
        "observed_same_sample_confirmations": sum(
            item["same_sample_confirmed_observations"] for item in results),
        "unmatched_same_sample_observations": sum(
            item["unmatched_same_sample_observations"] for item in results),
        "unknown_ownership_observations": sum(
            item["unknown_ownership_observations"] for item in results),
        "unknown_ownership_pids": unknown, "runs": results,
        "interpretation": (
            "Integrity pass means the telemetry was parsed and bound, not that every "
            "ownership check passed. Review strict_same_sample_ownership_pass and "
            "mismatches explicitly. This records no process command lines."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--container-name", default="gdn60403-late-v2-20261009")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite audit evidence")
    try:
        result = audit(args)
    except BaseException as error:
        write_new(args.output, {"integrity_pass": False, "exception": type(error).__name__,
                                "message": str(error), "traceback": traceback.format_exc()})
        raise
    write_new(args.output, result)
    print({name: result[name] for name in (
        "integrity_pass", "strict_same_sample_ownership_pass",
        "observed_same_sample_confirmations", "unmatched_same_sample_observations",
        "unknown_ownership_observations",
    )})


if __name__ == "__main__":
    main()
