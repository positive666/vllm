"""Run bounded GDN checks sequentially and retain every failure and skip."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def junit_results(path):
    cases = []
    for node in ET.parse(path).getroot().iter("testcase"):
        status = next(
            (
                name
                for name in ("failure", "error", "skipped")
                if node.find(name) is not None
            ),
            "passed",
        )
        cases.append({"name": node.get("name"), "status": status})
    return {
        "counts": {
            name: sum(case["status"] == name for case in cases)
            for name in ("passed", "failure", "error", "skipped")
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/source"))
    parser.add_argument("--results", type=Path, default=Path("/results"))
    parser.add_argument(
        "--benchmark", type=Path, default=Path("/artifacts/benchmark_gdn_takeover.py")
    )
    parser.add_argument("--source-head")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    summary = {
        "started_utc": now(),
        "python": sys.executable,
        "source": str(args.source.resolve()),
        "tasks": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    probe = [
        sys.executable,
        str(Path(__file__).with_name("runtime_probe.py")),
        "--source",
        str(args.source),
        "--output",
        str(args.results / "runtime-probe.json"),
    ]
    if args.source_head:
        probe += ["--source-head", args.source_head]
    suites = [
        (
            "adapter-and-split",
            40,
            [
                "tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py",
                "tests/kernels/mamba/test_gdn_forward_core_split.py",
                "-k",
                "gdn_decode or flashinfer_decode or gdn_layer or forward_core_split",
            ],
        ),
        (
            "gdn-metadata",
            65,
            ["tests/v1/attention/test_gdn_metadata_builder.py"],
        ),
        ("gdn-config", 1, ["tests/test_config.py", "-k", "gdn_decode"]),
    ]
    tasks = [("runtime-probe", probe, None, None)]
    for name, expected, selections in suites:
        xml = args.results / f"{name}.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-ra",
            "--color=no",
            "-o",
            f"cache_dir={args.results / 'pytest-cache'}",
            "--junitxml",
            str(xml),
            *selections,
        ]
        tasks.append((name, command, expected, xml))
    tasks.append(
        ("benchmark-help", [sys.executable, str(args.benchmark), "--help"], None, None)
    )
    for name, command, expected, xml in tasks:
        item = {
            "name": name,
            "command": command,
            "started_utc": now(),
            "expected_case_count": expected,
        }
        log = args.results / f"{name}.log"
        print(json.dumps({"event": "start", "task": name}), flush=True)
        started = time.monotonic()
        started_wall_ns = time.time_ns()
        with log.open("w", encoding="utf-8") as output:
            try:
                result = subprocess.run(
                    command,
                    cwd=args.source,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                item["exit_code"] = result.returncode
            except subprocess.TimeoutExpired:
                item.update(exit_code=124, timed_out=True)
            except OSError as exc:
                item.update(exit_code=127, launch_error=f"{type(exc).__name__}: {exc}")
        item.update(
            finished_utc=now(),
            seconds=time.monotonic() - started,
            log=str(log),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        )
        if (
            xml is not None
            and xml.is_file()
            and xml.stat().st_mtime_ns >= started_wall_ns
        ):
            try:
                item["pytest"] = junit_results(xml)
                item["case_count_matches"] = len(item["pytest"]["cases"]) == expected
                if name == "native-mtp-diagnostic":
                    pure = [
                        case
                        for case in item["pytest"]["cases"]
                        if "pure-mtp" in case["name"]
                    ]
                    item["pure_mtp_cases"] = pure
                    item["pure_mtp_selection_assertions_passed"] = len(
                        pure
                    ) == 2 and all(case["status"] == "passed" for case in pure)
            except (ET.ParseError, OSError) as exc:
                item["junit_error"] = f"{type(exc).__name__}: {exc}"
        summary["tasks"].append(item)
        (args.results / "recheck-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (args.results / f"{name}.exit").write_text(
            str(item["exit_code"]) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "finished",
                    "task": name,
                    "exit_code": item["exit_code"],
                    "pytest_counts": item.get("pytest", {}).get("counts"),
                }
            ),
            flush=True,
        )
    summary["finished_utc"] = now()
    summary["all_tasks_exit_zero"] = all(
        task["exit_code"] == 0 for task in summary["tasks"]
    )
    pytest_tasks = [
        task for task in summary["tasks"] if task["expected_case_count"] is not None
    ]
    summary["all_expected_pytest_cases_passed"] = all(
        task.get("case_count_matches") is True
        and task.get("pytest", {}).get("counts", {}).get("passed")
        == task["expected_case_count"]
        for task in pytest_tasks
    )
    summary["scope"] = (
        "Targeted checks and native MTP diagnostic; "
        "benchmark help is not a measurement."
    )
    (args.results / "recheck-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return int(
        not summary["all_tasks_exit_zero"]
        or not summary["all_expected_pytest_cases_passed"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
