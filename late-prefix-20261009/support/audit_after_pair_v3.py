"""Run frozen CPU consumers after both native capture arms have completed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

AUDITORS = {
    "audit_late_sampler.py": "674044c55c49328117283e538f90abd3584c67d53a1ef5fbe2d5181833eda43d",
    "audit_late_host.py": "62b336f3566cef56cdd2d46c09d26016f4261727bcd7388d852f2cb904638362",
    "audit_late_snapshots.py": "bab8c17cc979527f0dd906355bee89843814e91cc268eda8cce934c32118fe5e",
}
HOST_V3_SHA = '294361399debabd6922bc1616722c88d7b0c62e1eff7a6b07a04155257d97f37'
INDICES = (198, 206, 209, 228, 255, 285, 292, 318)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def checked_helpers(root):
    for name, expected in AUDITORS.items():
        require(sha(root / name) == expected, "Frozen consumer SHA: " + name)
    return AUDITORS.copy()


def commands(args):
    prefix = [args.uv, "run", "--offline", "--no-project", sys.executable]
    shared = ["--reference-root", str(args.reference_root), "--source-root", str(args.source_root),
              "--source-manifest", str(args.source_manifest)]
    return [
        ("native", prefix + [str(args.helper_root / "audit_late_sampler.py"),
                             "--triton-dir", str(args.triton_dir),
                             "--flashinfer-dir", str(args.flashinfer_dir), *shared,
                             "--output", str(args.output_root / "native.json")], 300),
        ("host", prefix + [str(args.host_auditor),
                           "--runs", str(args.triton_dir), str(args.flashinfer_dir),
                           "--monitor", str(args.monitor), "--container-name",
                           args.container_name, "--output",
                           str(args.output_root / "host.json")], 120),
        ("snapshots", prefix + [str(args.helper_root / "audit_late_snapshots.py"),
                                "--triton", str(args.triton_dir),
                                "--flashinfer", str(args.flashinfer_dir), *shared,
                                "--output", str(args.output_root / "snapshots.json")], 1800),
    ]


def completed_arm(root, backend):
    exit_path = root.parent / (root.name + ".exit")
    require(exit_path.read_text().strip() == "0", "Actual process exit0: " + root.name)
    require(not (root / "failure.json").exists(), "No producer failure")
    completed, fixture, launch = read(root / "completed.json"), read(root / "fixture.json"), read(root / "launch.json")
    require(completed["status"] == "completed" and launch["backend"] == backend and
            completed["launch_sha256"] == sha(root / "launch.json"), "Completed native launch")
    require(read(root / "shutdown.json") ==
            {"status": "shutdown-returned", "generation_completed": True}, "Actual successful shutdown")
    require([item["index"] for item in completed["examples"]] == list(INDICES) ==
            [item["index"] for item in fixture["examples"]], "All eight native histories")
    for item, expected in zip(completed["examples"], fixture["examples"]):
        require(item["token_ids"] == expected["reference_token_ids"] and
                item["exact_forced_tokens"] is True, "Full actual native forced output")
    require(sum(len(item["token_ids"]) for item in completed["examples"]) == 7378,
            "7378 actual forced positions")
    require(len(list((root / "snapshots").rglob("*.pt"))) == 672,
            "672 binary snapshots before complete consumer audit")
    host_path = root.parent / (root.name + "-host.csv")
    with host_path.open(encoding="utf-8", newline="") as handle:
        host_rows = list(csv.DictReader(handle, skipinitialspace=True))
    require(host_rows, "Actual host telemetry")
    last = [row for row in host_rows if row["utc"] == host_rows[-1]["utc"]]
    require(len(last) == 2 and all(float(row["memory.used"].split()[0]) < 16 and
            float(row["utilization.gpu"].split()[0]) == 0 for row in last),
            "Final sampled GPUs released")
    process_lines = (root.parent / (root.name + "-processes.log")).read_text().splitlines()
    last_stamp = host_rows[-1]["utc"]
    require(last_stamp in process_lines, "Final process sample matches GPU telemetry")
    last_process_rows = process_lines[process_lines.index(last_stamp) + 1:]
    require(not any(line.startswith("GPU-") for line in last_process_rows),
            "Final sampled experiment GPUs have no observed compute PID")
    return {"run": root.name, "exit_sha256": sha(exit_path),
            "completed_sha256": sha(root / "completed.json"),
            "shutdown_sha256": sha(root / "shutdown.json"), "positions": 7378,
            "binary_snapshot_count": 672, "last_gpu_sample_utc": host_rows[-1]["utc"]}


def run_stage(args, name, command, timeout_seconds, env):
    record = {"stage": name, "argv": command, "timeout_seconds": timeout_seconds,
              "gpu_execution": False}
    write_new(args.output_root / (name + "-command.json"), record)
    stdout = args.output_root / (name + ".stdout")
    stderr = args.output_root / (name + ".stderr")
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                   start_new_session=(os.name == "posix"))
        try:
            record["exit_code"] = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
            record.update(exit_code=124, timed_out=True)
    record.update(stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
    write_new(args.output_root / (name + "-status.json"), record)
    return record


def extract_summary(native, host, snapshots):
    pair = native["paircomparison"]
    diffs = pair["native_sampler_differences"]
    first = {}
    for item in diffs:
        first.setdefault(str(item["index"]), item["step"])
    counts = {backend: len(items) for backend, items in snapshots["snapshots_by_arm"].items()}
    require(counts == {"triton": 672, "flashinfer": 672}, "All 672 snapshots per arm audited")
    ownership = host["strict_same_sample_ownership_pass"]
    status = snapshots["status"]
    if status == "OBSERVED_INITIAL_PASS" and not ownership:
        status = "REVIEW_REQUIRED"
    return {
        "status": status, "snapshot_localization_status": snapshots["status"],
        "gpu_execution": False, "source_head": native["source_head"],
        "positions_per_arm": {key: value["global_unique_effective_records"]
                              for key, value in native["arms"].items()},
        "true_decode_positions_per_arm": {key: value["true_decode_observations"]
                                          for key, value in native["arms"].items()},
        "snapshots_per_arm": counts,
        "native_starting_condition_gate_pass": pair["native_starting_condition_gate_pass"],
        "full_prefill_logits_equal": pair["full_prefill_logits_equal"],
        "common_prefill_sampler": pair["common_prefill_sampler"],
        "matching_execution_contexts": pair["matching_execution_contexts"],
        "schedules_equal": pair["schedules_equal"],
        "native_sampler_difference_count": len(diffs),
        "first_backend_native_sampler_difference_step": first,
        "first_actual_sampler_reference_difference": pair["first_actual_sampler_reference_difference"],
        "common_observed_initial_gdn_state": snapshots["observed_states"]["common_observed_initial_state"],
        "strict_same_sample_ownership_pass": ownership,
        "unmatched_same_sample_observations": host["unmatched_same_sample_observations"],
        "unknown_ownership_observations": host["unknown_ownership_observations"],
        "interpretation": "CPU evidence audit only. NOGO or ownership mismatches remain explicit. Forced histories do not establish accuracy, natural stopping, performance or a decode root cause. Full-attention KV is unobserved; GPU replays remain a separate gate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("helper-root", "triton-dir", "flashinfer-dir", "reference-root",
                 "source-root", "source-manifest", "monitor", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--host-auditor", type=Path, required=True)
    parser.add_argument("--container-name", default="gdn60403-late-v3-20261009")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    require(args.container_name == "gdn60403-late-v3-20261009", "Owned v3 namespace")
    require(sha(args.host_auditor) == HOST_V3_SHA, "Namespace-only v3 host SHA")
    helpers = checked_helpers(args.helper_root)
    planned = commands(args)
    if args.plan_only:
        print(json.dumps({"status": "CPU-orchestration-plan", "gpu_execution": False,
                          "frozen_consumers": helpers,
                          "host_v3_sha256": HOST_V3_SHA,
                          "commands": [{"stage": name, "argv": command, "timeout_seconds": timeout}
                                       for name, command, timeout in planned]}, indent=2))
        return 0
    args.output_root.mkdir(exist_ok=False)
    summary = {"status": "AUDIT_ERROR", "gpu_execution": False, "stages": []}
    try:
        preflight = {"triton": completed_arm(args.triton_dir, "triton"),
                     "flashinfer": completed_arm(args.flashinfer_dir, "flashinfer")}
        write_new(args.output_root / "preflight.json", preflight)
        env = {key: value for key, value in os.environ.items() if not key.startswith("GDN_")}
        env["PYTHONPATH"] = os.pathsep.join((str(args.helper_root), str(args.source_root)))
        for name, command, timeout in planned:
            stage = run_stage(args, name, command, timeout, env)
            summary["stages"].append(stage)
            require(stage["exit_code"] == 0 or (name == "snapshots" and stage["exit_code"] == 2),
                    "Consumer process failed: " + name)
            report = read(args.output_root / (name + ".json"))
            require(report.get("integrity_pass") is True if name != "snapshots" else
                    report["status"] in ("OBSERVED_INITIAL_PASS", "NOGO"),
                    "Consumer evidence invalid: " + name)
        stages = summary["stages"]
        summary = extract_summary(read(args.output_root / "native.json"),
                                  read(args.output_root / "host.json"),
                                  read(args.output_root / "snapshots.json"))
        summary["stages"] = stages
    except Exception as error:
        summary.update(status="AUDIT_ERROR", exception=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
    finally:
        try:
            checked_helpers(args.helper_root)
            require(sha(args.host_auditor) == HOST_V3_SHA, "Unchanged v3 host SHA")
            summary["frozen_consumers_unchanged"] = True
        except Exception as error:
            summary.update(status="AUDIT_ERROR", frozen_consumers_unchanged=False,
                           helper_error=str(error))
        summary.update(orchestrator_sha256=sha(Path(__file__)), frozen_consumers=helpers,
                       host_v3_sha256=HOST_V3_SHA, owned_container_name=args.container_name)
        write_new(args.output_root / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "gpu_execution",
                                                  "frozen_consumers_unchanged")}, indent=2))
    return 0 if summary["status"] == "OBSERVED_INITIAL_PASS" else 2 if summary["status"] in (
        "NOGO", "REVIEW_REQUIRED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
