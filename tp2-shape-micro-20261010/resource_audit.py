"""Audit recorded short-matrix supervision; never infer mathematical correctness."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
GPUS = {4: "GPU-4af84168-6d29-6128-fee3-1b146cce9b47",
        6: "GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e"}
STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
CID = re.compile(r"^[0-9a-f]{64}$")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stamp(value):
    require(bool(STAMP.fullmatch(value)), "Malformed UTC timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def number(value):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:MiB|%)?\s*", value)
    require(match is not None, "Malformed numeric telemetry")
    return float(match.group(1))


def process_frames(text):
    """Retain every compute observation, including an incomplete final frame."""
    frames = []
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if STAMP.fullmatch(line):
            stamp(line)
            current = {"timestamp": line, "compute": [], "owned_pids": [],
                       "owned_section_seen": False}
            frames.append(current)
            continue
        require(current is not None, "Process data before a timestamp")
        if line == "OWNED":
            require(not current["owned_section_seen"], "Duplicate OWNED section")
            current["owned_section_seen"] = True
        elif current["owned_section_seen"]:
            require(line.isdecimal() and int(line) > 0, "Malformed owned PID")
            current["owned_pids"].append(int(line))
        else:
            row = next(csv.reader([line], skipinitialspace=True))
            require(len(row) == 3, "Malformed compute observation")
            require(row[1].strip().isdecimal() and int(row[1]) > 0,
                    "Malformed compute PID")
            current["compute"].append({"gpu_uuid": row[0].strip(),
                                       "pid": int(row[1]),
                                       "used_memory_mib": number(row[2])})
    require(frames, "No process snapshots")
    times = [frame["timestamp"] for frame in frames]
    require(len(times) == len(set(times)), "Duplicate process timestamp")
    require(times == sorted(times), "Process timestamps are out of order")
    for frame in frames:
        require(len(frame["owned_pids"]) == len(set(frame["owned_pids"])),
                "Duplicate PID in one owned snapshot")
    return frames


def identities(text, expected_cid, expected_owner):
    records = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|")
        require(len(parts) == 4, "Malformed identity row; expected four pipe fields")
        when, actual_cid, owner, running = parts
        stamp(when)
        require(when not in records, "Duplicate identity timestamp")
        require(actual_cid == expected_cid and owner == expected_owner,
                "Monitor identity does not match expected CID and owner")
        require(running == "true", "Monitored container was not running")
        records[when] = {"container_id": actual_cid, "owner": owner,
                         "running": True}
    require(records, "No actual container-identity observations")
    return records


def gpu_frames(text):
    result = {}
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row:
            continue
        require(len(row) == 8, "Malformed timestamped GPU telemetry row")
        stamp(row[0])
        require(row[1].strip().isdecimal(), "Malformed physical GPU index")
        index = int(row[1])
        require(GPUS.get(index) == row[2].strip(), "Unexpected GPU index/UUID")
        frame = result.setdefault(row[0], {})
        require(index not in frame, "Duplicate GPU in a telemetry snapshot")
        frame[index] = {"gpu_uuid": row[2].strip(),
                        "memory_used_mib": number(row[3]),
                        "utilization_percent": number(row[4])}
    require(result, "No GPU telemetry snapshots")
    return result


def inspection(payload, expected_cid, expected_owner):
    require(isinstance(payload, list) and len(payload) == 1,
            "Full single-container docker inspect list is required")
    item = payload[0]
    require(item.get("Id") == expected_cid, "Full inspect CID mismatch")
    labels = item.get("Config", {}).get("Labels", {})
    require(labels.get("codex.owner") == expected_owner, "Full inspect owner mismatch")
    state = item.get("State")
    require(isinstance(state, dict), "Full inspect lacks actual State")
    return {"container_id": item["Id"], "owner": labels["codex.owner"],
            "state": state}


def audit(results, manifest_path, expected_owner):
    hashes = {}
    errors = []
    review = []
    gates = {}
    details = {}

    def read(name):
        data = (results / name).read_bytes()
        hashes[name] = {"sha256": sha(data), "bytes": len(data)}
        return data.decode("utf-8")

    def check(name, function):
        try:
            value = function()
            gates[name] = True
            return value
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            gates[name] = False
            errors.append({"gate": name, "error": str(error)})
            return None

    def exit_value(name, accepted):
        value = read(name).strip()
        require(re.fullmatch(r"-?\d+", value), "Malformed actual exit record")
        code = int(value)
        details[name] = code
        require(code in accepted, name + " has a nonaccepted actual exit")
        return code

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("source_head") == HEAD, "Unexpected producer source binding")
    placement = manifest["placement"]
    require(placement["gpu_indices"] == list(GPUS), "Expected physical GPU4/6 placement")
    require(placement["gpu_uuids"] == list(GPUS.values()), "Expected physical GPU UUIDs")
    for name, expected in manifest["host_scripts"].items():
        actual = (manifest_path.parent / name).read_bytes()
        require(sha(actual) == expected["sha256"] and len(actual) == expected["bytes"],
                "Frozen host-script bytes differ: " + name)
    expected_cid = read("container-id.txt").strip()
    require(bool(CID.fullmatch(expected_cid)), "Fresh full container ID is required")
    require(expected_owner and "|" not in expected_owner, "Explicit owner is required")
    check("matrix_exit", lambda: exit_value("matrix.exit", {0}))
    for arm in ("gpu4", "gpu6"):
        check("arm_" + arm + "_exit", lambda arm=arm: exit_value("micro-" + arm + ".exit", {0}))
        if (results / ("micro-" + arm + ".log")).is_file():
            read("micro-" + arm + ".log")
    check("cleanup_stop_exit", lambda: exit_value("cleanup-stop.exit", {0}))
    markers = [name for name in ("repeated-unmatched-owner.txt", "monitor-stop.log")
               if (results / name).exists()]
    details["monitor_stop_markers"] = markers
    for name in markers:
        read(name)
    check("monitor_no_failure_marker", lambda: require(not markers, "Monitor failure marker exists"))
    check("monitor_final_exit", lambda: exit_value("monitor-final.exit", {143}))
    details["monitor_143_scope"] = (
        "Expected supervisor termination only with no monitor-stop marker; "
        "does not erase partial telemetry or unmatched PID observations."
    )
    containers = {}
    for phase in ("initial", "before", "after"):
        containers[phase] = check(
            "container_" + phase + "_identity",
            lambda phase=phase: inspection(json.loads(read("container-inspect-" + phase + ".json")),
                                           expected_cid, expected_owner),
        )
    details["containers"] = containers
    if containers["initial"]:
        check("container_initial_running", lambda: require(
            containers["initial"]["state"].get("Running") is True,
            "Initial container was not running"))
    if containers["after"]:
        def stopped():
            state = containers["after"]["state"]
            require(state.get("Running") is False and state.get("Status") == "exited"
                    and state.get("Pid") == 0 and state.get("OOMKilled") is False
                    and state.get("Dead") is False and not state.get("Error"),
                    "Container is not demonstrably stopped without OOMKilled")
            require(bool(state.get("FinishedAt")), "Actual FinishedAt is missing")
        check("container_after_stopped_not_oomkilled", stopped)
    for name in ("cleanup-before.json", "cleanup-after.json", "cleanup-stop.log",
                 "matrix-completed.txt"):
        if (results / name).is_file():
            read(name)
    frame_list = check("process_log_format", lambda: process_frames(read("host-processes.log")))
    identity = check("identity_log_binding", lambda: identities(
        read("host-identities.log"), expected_cid, expected_owner))
    telemetry = check("gpu_log_format", lambda: gpu_frames(read("host-gpus.csv")))
    observations = []
    unmatched = []
    unknown_uuid = []
    incomplete = []
    all_owned = {pid for frame in frame_list or [] for pid in frame["owned_pids"]}
    for frame in frame_list or []:
        when = frame["timestamp"]
        identity_ok = identity is not None and when in identity
        gpu_ok = telemetry is not None and set(telemetry.get(when, {})) == set(GPUS)
        if not (identity_ok and gpu_ok and frame["owned_section_seen"]):
            incomplete.append({"timestamp": when, "identity": identity_ok,
                               "both_gpu_rows": gpu_ok,
                               "owned_section": frame["owned_section_seen"]})
        for raw in frame["compute"]:
            row = dict(raw, timestamp=when)
            uuid_ok = row["gpu_uuid"] in GPUS.values()
            pid_ok = row["pid"] in frame["owned_pids"]
            row["classification"] = "same_container_pid_snapshot" if identity_ok and pid_ok else "unmatched"
            if not uuid_ok:
                unknown_uuid.append(dict(row))
            if row["classification"] == "unmatched":
                row["reason"] = (
                    "Seen in another owned snapshot; possible exit/visibility race, not proven here"
                    if row["pid"] in all_owned else "Not seen in any captured owned PID snapshot"
                )
                unmatched.append(dict(row))
            observations.append(row)
    details["compute_observations"] = observations
    details["unmatched_observations"] = unmatched
    details["unknown_gpu_uuid_observations"] = unknown_uuid
    details["incomplete_process_frames"] = incomplete
    process_times = {frame["timestamp"] for frame in frame_list or []}
    details["identity_only_timestamps"] = sorted(set(identity or {}) - process_times)
    details["gpu_only_timestamps"] = sorted(set(telemetry or {}) - process_times)
    if unmatched:
        review.append("Every unmatched observation remains unresolved; possible races are not erased")
    if unknown_uuid:
        errors.append({"gate": "compute_gpu_placement", "error": "Unknown GPU UUID observed"})
    if incomplete or details["identity_only_timestamps"] or details["gpu_only_timestamps"]:
        review.append("Partial or unjoined telemetry remains visible")
    if not observations:
        errors.append({"gate": "compute_coverage", "error": "No recorded GPU compute observations"})
    def final_gpus():
        rows = list(csv.reader(io.StringIO(read("final-gpus.csv")), skipinitialspace=True))
        require(rows and len(rows[0]) == 4 and rows[0][0].strip() == "index", "Final GPU CSV header missing")
        output = {}
        for row in rows[1:]:
            if not row:
                continue
            require(len(row) == 4 and row[0].strip().isdecimal(), "Malformed final GPU row")
            index = int(row[0])
            require(index not in output and GPUS.get(index) == row[1].strip(), "Final GPU identity differs")
            used = number(row[2])
            require(used <= 16, "Final selected GPU memory exceeds idle guard")
            output[index] = {"gpu_uuid": row[1].strip(), "memory_used_mib": used,
                             "utilization_percent": number(row[3])}
        require(set(output) == set(GPUS), "Both final physical GPUs must be recorded")
        return output
    details["final_gpus"] = check("final_gpu_identity_memory", final_gpus)
    def final_processes():
        rows = list(csv.reader(io.StringIO(read("final-processes.csv")), skipinitialspace=True))
        require(rows and len(rows[0]) == 3 and rows[0][0].strip() == "gpu_uuid", "Final process CSV header missing")
        body = [row for row in rows[1:] if row and any(part.strip() for part in row)]
        final_rows = []
        for row in body:
            require(len(row) == 3 and row[1].strip().isdecimal(), "Malformed final compute row")
            item = {"gpu_uuid": row[0].strip(), "pid": int(row[1]),
                    "used_memory_mib": number(row[2]), "timestamp": None,
                    "observation_file": "final-processes.csv",
                    "classification": "unmatched_after_container_stop"}
            item["reason"] = ("Seen in a prior owned snapshot; possible exit/visibility race, not erased"
                              if item["pid"] in all_owned else "Not seen in any owned snapshot")
            final_rows.append(item)
            observations.append(dict(item))
            unmatched.append(dict(item))
            if item["gpu_uuid"] not in GPUS.values():
                unknown_uuid.append(dict(item))
        details["final_compute_observations"] = final_rows
        require(not body, "Final selected-GPU compute process list is not empty")
        return {"compute_pid_count": 0}
    details["final_processes"] = check("final_no_compute_pids", final_processes)
    strict = bool(observations and frame_list and identity and telemetry) and not (
        unmatched or unknown_uuid or incomplete or markers or details["identity_only_timestamps"]
        or details["gpu_only_timestamps"]
    )
    return {
        "schema": "gdn-independent-shape-micro-supervisor-v1",
        "status": "FAIL" if errors else "REVIEW_REQUIRED" if review else "PASS",
        "scope": "Recorded process/placement/cleanup only; no mathematical, accuracy, or performance verdict",
        "strict_ownership": strict,
        "mathematical_gate_evaluated": False,
        "expected_container_id": expected_cid, "expected_owner": expected_owner,
        "expected_physical_gpus": GPUS,
        "producer_manifest_sha256": sha(manifest_bytes),
        "auditor_sha256": sha(Path(__file__).read_bytes()),
        "input_files": hashes, "gates": gates, "failures": errors,
        "review_required_reasons": review,
        "counts": {"process_snapshots": len(frame_list or []),
                   "compute_observations": len(observations),
                   "matched_same_snapshot": len(observations) - len(unmatched),
                   "unmatched": len(unmatched), "unknown_uuid": len(unknown_uuid),
                   "partial_frames": len(incomplete)},
        "details": details,
        "limitations": [
            "Sequential NVML and docker-top sampling can race; an unmatched row stays unmatched",
            "Container OOMKilled=false describes the container, not every model-process failure",
            "This report checks raw supervision records; redacted copies require explicit raw/public maps",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-owner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing audit result")
    try:
        report = audit(args.results, args.manifest, args.expected_owner)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        report = {"schema": "gdn-independent-shape-micro-supervisor-v1", "status": "FAIL",
                  "strict_ownership": False, "mathematical_gate_evaluated": False,
                  "failure": str(error), "auditor_sha256": sha(Path(__file__).read_bytes())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "strict_ownership": report["strict_ownership"],
                      "counts": report.get("counts"), "mathematical_gate_evaluated": False}))
    return 0 if report["status"] == "PASS" else 2 if report["status"] == "REVIEW_REQUIRED" else 1


if __name__ == "__main__":
    sys.exit(main())
