"""Replay one captured backend in a fresh process and isolated owned caches.

Each selected capture resets its own recorded prestate. This does not replay
a propagated token trajectory or establish model-quality equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from audit_late_snapshots import LAYERS, STEPS, index_arm, sha
from late_plan import HEAD, read, require, validate_inputs

FROZEN_REPLAY_SHA = "f8bc6de59f821b9846acd2b63ba221cb1e247ce8e97dec2a5f1e920e57db9e4a"
EXPECTED_DTYPE = {"triton": "torch.bfloat16", "flashinfer": "torch.float32"}
REFERENCE_COMPLETED_SHA = (
    "d293b322876e8236f7799a688221ddd0d10e79f8046c6601db40fa6316421e78"
)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def summarize_cases(cases, selected_count, discovered_count):
    """Reject tolerance-only fidelity, missing cases and partial coverage."""
    exact = sum(
        case["production_reproduction"]["attempted"]
        and case["production_reproduction"]["exact_reproduction"]
        for case in cases
    )
    tolerance = all(case["checks_within_frozen_tolerance"] for case in cases)
    passed = (
        selected_count > 0
        and len(cases) == selected_count
        and exact == selected_count
        and tolerance
    )
    accepted = passed and selected_count == discovered_count
    return {
        "exact_fidelity_count": exact,
        "all_cases_within_frozen_tolerance": tolerance,
        "selected_cases_pass": passed,
        "accepted": accepted,
        "status": (
            "completed"
            if accepted
            else "partial_completed"
            if passed
            else "completed_with_failed_checks"
        ),
    }


def prepare_arm(run_dir, backend, manifest_path, audit_path=None):
    """Validate exact complete capture/event/history joins without CUDA imports."""
    lock, fixtures = validate_inputs(run_dir)
    require(
        lock["reference_files"]["completed.json"] == REFERENCE_COMPLETED_SHA,
        "Reference is the independently verified failing FI eager-1 output",
    )
    manifest = read(manifest_path)
    launch = read(run_dir / "launch.json")
    require(
        manifest["source_head"] == HEAD == launch["source_head"],
        "Reviewed production source revision",
    )
    require(
        sha(manifest_path) == launch["source_manifest_sha256"],
        "Replay source manifest is the actual capture launch manifest",
    )
    for name, entry in manifest["files"].items():
        if name in launch["source_files"]:
            expected = entry["sha256"] if isinstance(entry, dict) else entry
            require(
                expected == launch["source_files"][name],
                "Launch/manifest source digest agrees: " + name,
            )
    indexed = index_arm(run_dir, backend, {row["index"]: row for row in fixtures})
    expected_keys = {
        (rank, layer, step) for rank in (0, 1) for layer in LAYERS for step in STEPS
    }
    require(
        set(indexed) == expected_keys and len(indexed) == 672,
        "Complete 672 rank/layer/actual-step captures",
    )
    inputs = []
    for key, entry in sorted(indexed.items()):
        meta = entry["record"]["metadata"]
        require(
            meta["original_shapes_strides"]["dt_bias"]["dtype"]
            == EXPECTED_DTYPE[backend],
            "Actual captured backend bias dtype",
        )
        require(
            meta["original_backend"] == backend and meta["rank"] == key[0],
            "One captured backend and actual TP rank",
        )
        inputs.append(
            {
                "path": str(entry["path"].resolve()),
                "sha256": entry["record"]["sha256"],
                "bytes": entry["record"]["bytes"],
                "sidecar_sha256": sha(entry["path"].with_suffix(".json")),
                "rank": key[0],
                "layer_idx": key[1],
                "step": key[2],
                "execution_id": meta["runtime_join"]["late_execution_id"],
                "captured_backend": backend,
                "captured_dt_bias_dtype": EXPECTED_DTYPE[backend],
            }
        )
    audit_binding = None
    if audit_path is not None:
        audit = read(audit_path)
        require(audit["source_head"] == HEAD, "Paired capture audit source")
        listed = audit["snapshots_by_arm"][backend]
        require(len(listed) == len(inputs), "Paired audit complete arm count")
        by_key = {
            (item["rank"], item["layer_idx"], item["step"]): item for item in listed
        }
        require(len(by_key) == len(inputs), "Unique paired audit capture keys")
        for item in inputs:
            other = by_key[(item["rank"], item["layer_idx"], item["step"])]
            require(
                Path(other["path"]).resolve() == Path(item["path"]),
                "Paired audit exact capture path",
            )
            for name in (
                "sha256",
                "execution_id",
                "captured_backend",
                "captured_dt_bias_dtype",
            ):
                require(
                    other[name] == item[name], "Paired audit capture binding: " + name
                )
        audit_binding = {
            "path": str(audit_path.resolve()),
            "sha256": sha(audit_path),
            "observed_initial_status": audit["status"],
            "scope": (
                "The local replay result does not override a paired "
                "initial-condition NOGO."
            ),
        }
    return inputs, audit_binding, lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=tuple(EXPECTED_DTYPE), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--capture-audit", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-captures",
        type=int,
        help="Explicit partial replay; never accepted as complete arm fidelity",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Capture joins only; no CUDA or numerical replay claim",
    )
    args = parser.parse_args()
    exit_path = args.output.with_suffix(".exit")
    receipt_path = args.output.with_suffix(".receipt.json")
    require(
        not any(path.exists() for path in (args.output, exit_path, receipt_path)),
        "Never overwrite replay evidence",
    )
    report = {
        "schema": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": HEAD,
        "backend": args.backend,
        "captured_dt_bias_dtype": EXPECTED_DTYPE[args.backend],
        "wrapper_sha256": sha(Path(__file__)),
        "acceptance_scope": (
            "Complete one-arm local fidelity/math checks only; paired initial "
            "conditions and global execution readiness are separate gates."
        ),
        "gpu_execution": False,
        "cases": [],
        "scope": {
            "independent_rank_local_updates": True,
            "reset_to_each_capture_prestate": True,
            "multi_gpu_execution_or_sequence_replay": False,
            "performance_or_model_accuracy_claim": False,
            "full_attention_kv_observed": False,
            "original_aliasing_or_absolute_addresses_restored": False,
            "state_scope": (
                "Captured compact active pages, null and sampled unused page "
                "only; synthetic storage-gap guards."
            ),
        },
    }
    exit_code = 1
    try:
        run_dir = args.run_dir.resolve()
        workspace = args.workspace.resolve()
        require(
            workspace != run_dir
            and run_dir not in workspace.parents
            and workspace not in run_dir.parents,
            "Separate owned replay workspace",
        )
        require(not workspace.exists(), "Fresh exclusive workspace required")
        require(
            args.max_captures is None or args.max_captures > 0,
            "Positive explicit subset limit",
        )
        inputs, audit_binding, lock = prepare_arm(
            run_dir, args.backend, args.source_manifest, args.capture_audit
        )
        helper = Path(__file__).with_name("frozen_replay_capture.py")
        require(sha(helper) == FROZEN_REPLAY_SHA, "Frozen replay math helper SHA")
        report.update(
            source_manifest_sha256=sha(args.source_manifest),
            reference_completed_sha256=lock["reference_files"]["completed.json"],
            capture_audit_binding=audit_binding,
            discovered_capture_count=len(inputs),
            frozen_replay_sha256=FROZEN_REPLAY_SHA,
            capture_inputs=inputs,
            consumer_helper_sha256=sha(
                Path(__file__).with_name("audit_late_snapshots.py")
            ),
            reference_helper_sha256=sha(Path(__file__).with_name("late_plan.py")),
        )
        selected = (
            inputs[: args.max_captures] if args.max_captures is not None else inputs
        )
        report.update(
            selected_capture_count=len(selected),
            all_discovered_captures_selected=len(selected) == len(inputs),
        )
        if args.preflight_only:
            report.update(status="preflight_only_no_numerical_claim", accepted=False)
            exit_code = 0
        else:
            require(
                "torch" not in sys.modules and "flashinfer" not in sys.modules,
                "Fresh interpreter before CUDA/backend imports",
            )
            require(
                not any(name.startswith("GDN_") for name in os.environ),
                "No inherited forcing/capture observer environment",
            )
            launch = read(run_dir / "launch.json")
            native_binding = {}
            for name, expected in launch["fresh_runtime_files"].items():
                path = Path(expected["path"])
                require(
                    path.is_file()
                    and path.stat().st_size == expected["bytes"]
                    and sha(path) == expected["sha256"],
                    "Fresh captured native/runtime file: " + name,
                )
                native_binding[name] = expected
            fresh_versions = {
                name: importlib.metadata.version(name)
                for name in launch["fresh_runtime_versions"]
            }
            require(
                fresh_versions == launch["fresh_runtime_versions"],
                "Fresh captured runtime package versions",
            )
            report.update(
                native_runtime_binding=native_binding,
                captured_runtime_versions=fresh_versions,
                environment={
                    name: os.environ.get(name)
                    for name in (
                        "FLA_USE_FAST_OPS",
                        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
                    )
                },
            )
            workspace.mkdir(parents=True, exist_ok=False)
            os.environ.update(
                FLASHINFER_WORKSPACE_BASE=str(workspace / "flashinfer"),
                TRITON_CACHE_DIR=str(workspace / "triton"),
                CUDA_CACHE_PATH=str(workspace / "cuda"),
                VLLM_CACHE_ROOT=str(workspace / "vllm"),
            )
            write_new(
                workspace / "owner.json",
                {
                    "pid": os.getpid(),
                    "backend": args.backend,
                    "bias_dtype": EXPECTED_DTYPE[args.backend],
                    "wrapper_sha256": report["wrapper_sha256"],
                    "scope": (
                        "Fresh one-arm process/cache; no other caches removed "
                        "or reused."
                    ),
                },
            )
            import frozen_replay_capture as frozen

            torch = frozen.torch
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            require(torch.cuda.is_available(), "CUDA replay environment required")
            device = torch.device(args.device)
            torch.cuda.set_device(device)
            versions = {
                name: importlib.metadata.version(name)
                for name in frozen.EXPECTED_VERSIONS
            }
            require(versions == frozen.EXPECTED_VERSIONS, "Pinned replay runtime")
            from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
                GDNDecode,
            )

            report.update(
                source_binding=frozen.validate_source(GDNDecode, args.source_manifest),
                runtime_versions=versions,
                python_version=platform.python_version(),
                tolerances=frozen.TOLERANCES,
                workspace=str(workspace),
                bias_mode="actual-only",
                bias_dtype_isolation=(
                    "One captured dtype per fresh interpreter and workspace; "
                    "no cache clearing."
                ),
            )
            properties = torch.cuda.get_device_properties(device)
            report["device"] = {
                "name": properties.name,
                "uuid": str(properties.uuid),
                "capability": [properties.major, properties.minor],
            }
            report["gpu_execution"] = True
            for item in selected:
                path = Path(item["path"])
                report["active_capture_input"] = item
                require(
                    sha(path) == item["sha256"]
                    and sha(path.with_suffix(".json")) == item["sidecar_sha256"],
                    "Capture unchanged immediately before replay",
                )
                case = frozen.replay_capture(GDNDecode, path, device, "actual-only")
                require(
                    case["metadata"]["original_backend"] == args.backend
                    and case["bias_ablation"]["actual_dtype"]
                    == EXPECTED_DTYPE[args.backend],
                    "Actual loaded backend and bias dtype",
                )
                report["cases"].append(case)
                print(
                    json.dumps(
                        {
                            "rank": item["rank"],
                            "layer_idx": item["layer_idx"],
                            "step": item["step"],
                            "exact_own_fidelity": case["production_reproduction"][
                                "exact_reproduction"
                            ],
                            "frozen_tolerance": case["checks_within_frozen_tolerance"],
                        }
                    ),
                    flush=True,
                )
            report.update(summarize_cases(report["cases"], len(selected), len(inputs)))
            exit_code = 0 if report["selected_cases_pass"] else 2
    except Exception as exc:
        report.update(
            status="failed",
            accepted=False,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    marker = (str(exit_code) + "\n").encode()
    report.update(
        exit_code=exit_code, exit_marker_sha256=hashlib.sha256(marker).hexdigest()
    )
    write_new(args.output, report)
    with exit_path.open("xb") as handle:
        handle.write(marker)
    write_new(
        receipt_path,
        {
            "report_sha256": sha(args.output),
            "report_bytes": args.output.stat().st_size,
            "exit_sha256": sha(exit_path),
            "exit_code": exit_code,
            "wrapper_sha256": report["wrapper_sha256"],
            "frozen_replay_sha256": FROZEN_REPLAY_SHA,
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "accepted": report.get("accepted", False),
                "exit_code": exit_code,
                "report_sha256": sha(args.output),
            }
        ),
        flush=True,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
