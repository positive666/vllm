"""Prepare bounded post-run evidence after real capture/replay records exist.

This standard-library exporter never executes a GPU kernel or publishes files.
Raw snapshot binaries stay on the experiment host except two named samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from public_log_redaction import sanitize_payload

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
WRAPPER_SHA = "2e6d71a5ece657186bf43cfda020272e39fc80363a6db593b35d5c3f428d4744"
MATH_SHA = "f8bc6de59f821b9846acd2b63ba221cb1e247ce8e97dec2a5f1e920e57db9e4a"
LAYERS = tuple(i for i in range(64) if i % 4 != 3)
STEPS = (1, 18, 19, 1750, 3497, 3498, 3499)
KEYS = {(rank, layer, step) for rank in (0, 1) for layer in LAYERS for step in STEPS}
DTYPES = {"triton": "torch.bfloat16", "flashinfer": "torch.float32"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def encoded(value):
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def key(metadata):
    return metadata["rank"], metadata["layer_idx"], metadata["call_index"]


def summarize_replay(report, backend, captures):
    require(
        report["source_head"] == HEAD and report["backend"] == backend,
        "Replay arm/source",
    )
    require(
        report["wrapper_sha256"] == WRAPPER_SHA
        and report["frozen_replay_sha256"] == MATH_SHA,
        "Frozen replay helper provenance",
    )
    require(
        report["gpu_execution"] is True and report["bias_mode"] == "actual-only",
        "Actual numerical replay, not preflight or bias ablation",
    )
    require(
        report["discovered_capture_count"] == report["selected_capture_count"] == 672
        and report["all_discovered_captures_selected"] is True,
        "Full arm selected; partial replay cannot be exported as complete",
    )
    require(len(report["cases"]) == 672, "All 672 numerical cases completed")
    require(
        report["status"] in ("completed", "completed_with_failed_checks"),
        "Replay launch failure remains a blocker; do not infer missing cases",
    )
    seen = set()
    exceptions = []
    exact = 0
    within = 0
    padding = Counter()
    maxima = {}
    for case in report["cases"]:
        metadata = case["metadata"]
        case_key = key(metadata)
        require(
            case_key in captures and case_key not in seen, "Unique complete replay key"
        )
        seen.add(case_key)
        require(
            case["capture_sha256"] == captures[case_key]["sha256"]
            and metadata["original_backend"] == backend
            and case["bias_ablation"]["actual_dtype"] == DTYPES[backend],
            "Replay references its own actual capture/backend/dtype",
        )
        identity = {
            "captured_backend": backend,
            "rank": case_key[0],
            "layer_idx": case_key[1],
            "step": case_key[2],
            "capture_sha256": case["capture_sha256"],
        }
        fidelity = case["production_reproduction"]
        require(
            fidelity["attempted"] is True and fidelity["backend"] == backend,
            "Own production fidelity was actually attempted",
        )
        exact += bool(fidelity["exact_reproduction"])
        within += bool(case["checks_within_frozen_tolerance"])
        if not fidelity["exact_reproduction"]:
            exceptions.append(
                {
                    **identity,
                    "check": "exact_own_production_fidelity",
                    "observed": fidelity,
                }
            )
        require(
            set(case["arms"]) == {"captured_actual_bias"}, "Only actual captured bias"
        )
        arm = case["arms"]["captured_actual_bias"]
        for replayed_backend, checks in arm["backends"].items():
            require(replayed_backend in DTYPES, "Supported replayed backend")
            padding[replayed_backend] += checks["padding_row_count"]
            for name in ("invariants_ok", "reference_tolerance_ok"):
                if not checks[name]:
                    exceptions.append(
                        {
                            **identity,
                            "replayed_backend": replayed_backend,
                            "check": name,
                            "observed": checks,
                        }
                    )
            for name, metrics in checks.items():
                if not name.startswith("vs_fp64_"):
                    continue
                for metric in ("relative_l2", "absmax"):
                    value = metrics[metric]
                    if value is not None:
                        label = replayed_backend + "/" + name + "/" + metric
                        maxima[label] = max(maxima.get(label, 0), value)
        for name in ("triton_vs_flashinfer_output", "triton_vs_flashinfer_state"):
            if not arm[name]["within_tolerance"]:
                exceptions.append({**identity, "check": name, "observed": arm[name]})
    require(seen == KEYS, "All exact rank/layer/step cases represented")
    require(exact == report["exact_fidelity_count"], "Recomputed exact fidelity count")
    return {
        "captured_backend": backend,
        "captured_dt_bias_dtype": DTYPES[backend],
        "cases": len(seen),
        "exact_own_fidelity": exact,
        "within_frozen_tolerance": within,
        "accepted_local_arm": report["accepted"],
        "tolerances": report["tolerances"],
        "observed_maxima": maxima,
        "observed_padding_rows_by_replayed_backend": dict(padding),
    }, exceptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-run", type=Path, required=True)
    parser.add_argument("--flashinfer-run", type=Path, required=True)
    parser.add_argument("--triton-replay", type=Path, required=True)
    parser.add_argument("--flashinfer-replay", type=Path, required=True)
    parser.add_argument("--capture-audit", type=Path, required=True)
    parser.add_argument("--native-audit", type=Path, required=True)
    parser.add_argument("--pair-summary", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--support-file",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="Include bounded logs/exits/telemetry/cleanup/preparation receipts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.output_dir.exists() and not args.archive.exists(),
        "Exclusive new outputs",
    )
    audit = read(args.capture_audit)
    native_audit = read(args.native_audit)
    pair_summary = read(args.pair_summary)
    require(native_audit["integrity_pass"] is True, "Native raw audit integrity")
    require(
        pair_summary["source_head"] == HEAD
        and pair_summary["snapshots_per_arm"]
        == {"triton": 672, "flashinfer": 672}
        and pair_summary["snapshot_localization_status"] == audit["status"],
        "Actual pair summary binds completed capture scope",
    )
    require(
        audit["source_head"] == HEAD and set(audit["snapshots_by_arm"]) == set(DTYPES),
        "Two-arm completed capture audit with reviewed source",
    )
    require(
        audit["status"] in ("OBSERVED_INITIAL_PASS", "NOGO"), "Completed capture audit"
    )
    source_manifest_sha = sha(args.source_manifest)
    pending = {}
    snapshots = {}
    summaries = {}
    exceptions = []
    if not pair_summary["strict_same_sample_ownership_pass"]:
        exceptions.append(
            {
                "scope": "host_process_ownership_observations",
                "check": "strict_same_sample_ownership",
                "observed": {
                    name: pair_summary[name]
                    for name in (
                        "strict_same_sample_ownership_pass",
                        "unmatched_same_sample_observations",
                        "unknown_ownership_observations",
                    )
                },
            }
        )
    if audit["status"] == "NOGO":
        exceptions.append(
            {
                "scope": "paired_observed_initial_conditions_or_schedule",
                "check": "paired_capture_initial_gate",
                "observed": {
                    name: audit[name]
                    for name in (
                        "recorded_schedules_equal",
                        "full_prefill_logits",
                        "observed_states",
                    )
                },
            }
        )

    def add(path, name):
        relative = PurePosixPath(name)
        require(
            name not in pending
            and not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in name
            and ":" not in name,
            "Unique safe public path",
        )
        require(path.is_file() and not path.is_symlink(), "Regular frozen input file")
        pending[name] = path

    for backend in DTYPES:
        run_dir = getattr(args, backend + "_run").resolve()
        replay_path = getattr(args, backend + "_replay")
        replay = read(replay_path)
        for suffix in (".log", ".exit", "-host.csv", "-processes.log"):
            path = run_dir.with_name(run_dir.name + suffix)
            add(path, "logs/" + backend + "/" + path.name)
        require(
            run_dir.with_name(run_dir.name + ".exit").read_text().strip() == "0",
            "Actual capture producer exited zero",
        )
        require(
            replay["source_manifest_sha256"] == source_manifest_sha,
            "Replay/source manifest binding",
        )
        require(
            replay["capture_audit_binding"]["sha256"] == sha(args.capture_audit),
            "Replay bound to the same paired capture audit",
        )
        entries = audit["snapshots_by_arm"][backend]
        indexed = {
            (item["rank"], item["layer_idx"], item["step"]): item for item in entries
        }
        require(len(entries) == 672 and set(indexed) == KEYS, "Full 672 arm captures")
        snapshot_dir = run_dir / "snapshots"
        actual = {path.resolve() for path in snapshot_dir.rglob("*.pt")}
        require(
            actual == {Path(item["path"]).resolve() for item in entries},
            "No omitted or unbound raw snapshot",
        )
        for item in entries:
            path = Path(item["path"]).resolve()
            require(
                snapshot_dir in path.parents and sha(path) == item["sha256"],
                "Frozen owned snapshot SHA",
            )
            sidecar = path.with_suffix(".json")
            event = read(sidecar)
            require(
                event["sha256"] == item["sha256"]
                and event["bytes"] == path.stat().st_size,
                "Raw snapshot sidecar exact SHA/size",
            )
            public_name = backend + "/" + path.relative_to(run_dir).as_posix()
            snapshots[public_name] = {
                **item,
                "bytes": path.stat().st_size,
                "sidecar_sha256": sha(sidecar),
            }
        summaries[backend], failures = summarize_replay(replay, backend, indexed)
        exceptions.extend(failures)
        for path in sorted(run_dir.rglob("*")):
            if (
                path.is_file()
                and path.suffix != ".pt"
                and "__pycache__" not in path.parts
            ):
                add(path, backend + "/" + path.relative_to(run_dir).as_posix())
        selected = indexed[(0, 0, 3499)]
        representative = Path(selected["path"])
        add(
            representative,
            backend + "/" + representative.relative_to(run_dir).as_posix(),
        )
        for path in (
            replay_path,
            replay_path.with_suffix(".exit"),
            replay_path.with_suffix(".receipt.json"),
        ):
            add(path, "reports/" + backend + "/" + path.name)
        require(
            replay_path.with_suffix(".exit").read_text().strip()
            == str(replay["exit_code"]),
            "Actual replay exit marker agrees with report",
        )
        receipt = read(replay_path.with_suffix(".receipt.json"))
        require(
            receipt["report_sha256"] == sha(replay_path)
            and receipt["exit_sha256"] == sha(replay_path.with_suffix(".exit")),
            "Replay report and exit receipt integrity",
        )

    for path in (
        args.capture_audit,
        args.native_audit,
        args.pair_summary,
        args.source_manifest,
    ):
        add(path, "reports/" + path.name)
    for path in args.support_file:
        add(path, "support/" + path.name)
    add(Path(__file__), "packaging/export_late_evidence.py")
    require(
        not set(pending).intersection(
            (
                "README.md",
                "summary.json",
                "snapshot-manifest.json",
                "exceptions.json",
                "public-redactions.json",
                "package-manifest.json",
            )
        ),
        "Reserved generated filenames",
    )
    require(
        len([name for name in pending if name.endswith(".pt")]) == 2,
        "Only two explicit representative binaries",
    )
    patterns = {
        "github_token": rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
        "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "authorization": (
            rb"(?i)authorization\s*[:=]\s*[\"']?(?:bearer|token|basic)\s+"
            rb"[A-Za-z0-9+/=_-]{16,}"
        ),
        "literal_credential": (
            rb"(?i)\b(?:password|passwd|pwd|api_key|access_token|secret_key)\b"
            rb"\s*[:=]\s*[\"'][^\"'\r\n]{3,}[\"']"
        ),
        "credential_url": rb"https?://[^\s/:@]+:[^\s/@]+@",
    }
    findings = [
        {"path": name, "category": category}
        for name, path in pending.items()
        if path.suffix != ".pt"
        for category, pattern in patterns.items()
        if re.search(pattern, path.read_bytes())
    ]
    require(
        not findings,
        "Credential-pattern findings (values suppressed): " + json.dumps(findings),
    )
    summary = {
        "scope": (
            "Forced common-history local numerical diagnostics only; no "
            "accuracy/performance or complete initial-state causal proof."
        ),
        "source_head": HEAD,
        "capture_initial_status": audit["status"],
        "raw_snapshots": len(snapshots),
        "raw_snapshot_bytes_retained_remotely": sum(
            item["bytes"] for item in snapshots.values()
        ),
        "representative_selection": (
            "Each arm: rank0/layer0/actual step3499, explicitly not exhaustive."
        ),
        "replay_by_captured_backend": summaries,
        "exception_count": len(exceptions),
        "full_attention_kv_observed": False,
        "prior_unforced_truncation_resolved": False,
        "native_pair_status": pair_summary["status"],
        "native_sampler_difference_count": pair_summary[
            "native_sampler_difference_count"
        ],
        "strict_same_sample_ownership_pass": pair_summary[
            "strict_same_sample_ownership_pass"
        ],
        "unmatched_same_sample_observations": pair_summary[
            "unmatched_same_sample_observations"
        ],
        "unknown_ownership_observations": pair_summary[
            "unknown_ownership_observations"
        ],
    }
    readme = (
        b"# Late-prefix GDN numerical diagnostics\n\n"
        b"Generated from completed real records. Forced histories are not model "
        b"accuracy or serving performance results. "
        b"The source is unchanged d8; prior unforced truncation remains unresolved.\n\n"
        b"`summary.json` reports all 672 updates per captured arm, exact "
        b"own-production fidelity and frozen-reference math. "
        b"`exceptions.json` lists each observed failed check. Capture initial "
        b"condition NOGO remains NOGO. "
        b"Full-attention KV, complete hidden states, original aliasing/absolute "
        b"addresses and full state pools are unobserved.\n\n"
        b"`snapshot-manifest.json` preserves SHA/size for all 1,344 raw binaries "
        b"retained remotely. Only rank0/layer0/step3499 from each arm is included "
        b"as an explicit representative. All sidecars, trace/producer/runtime/"
        b"helper records and supplied audit/host/cleanup receipts are retained. "
        b"No padding-row coverage is inferred from synthetic guards. "
        b"AI assistance was used.\n\n"
        b"## Raw audit and public-copy scope\n\n"
        b"`public-redactions.json` maps each changed observation record's raw "
        b"SHA/size to its public redacted SHA/size. Private network addresses "
        b"are removed from public logs/runtime observations. Auditors ran on "
        b"the original remote raw records; recorded consumer SHA bindings "
        b"continue to identify those raw files. Public redacted copies cannot "
        b"rerun every raw consumer provenance gate byte for byte, and consumer "
        b"acceptance rules are unchanged. Source, tensor and mathematical "
        b"evidence remain exact. Generic source/reference localhost defaults "
        b"and public RFC network definitions are not host observations.\n\n"
        b"The original pair status REVIEW_REQUIRED and any same-sample "
        b"ownership exception remain explicit despite passing numerical "
        b"replays. See supplied `retained-attempts.md` and attempt records for "
        b"the pre-model environment-gate failure; it did not execute model "
        b"kernels and the stopped container was not OOM-killed. The successful "
        b"attempt added the missing read-only runtime-addons mount with the "
        b"same frozen helpers, source and reference.\n"
    )
    generated = {
        "README.md": readme,
        "summary.json": encoded(summary),
        "snapshot-manifest.json": encoded({"source_head": HEAD, "files": snapshots}),
        "exceptions.json": encoded(exceptions),
        ".gitattributes": b"* -text whitespace=cr-at-eol,-blank-at-eof\n",
    }
    public, redactions = sanitize_payload(
        {name: path.read_bytes() for name, path in pending.items()}
    )
    public["packaging/public_log_redaction.py"] = Path(__file__).with_name(
        "public_log_redaction.py"
    ).read_bytes()
    generated["public-redactions.json"] = encoded(
        {
            "scope": (
                "Only public observation copies are redacted; original raw "
                "audit/source bindings remain unchanged. No address values "
                "are published in this map."
            ),
            "redacted_files": redactions,
        }
    )
    public.update(generated)
    manifest = {
        "source_head": HEAD,
        "public_copy_scope": "Explicit raw/public SHA mapping",
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in sorted(public.items())
        },
    }
    public["package-manifest.json"] = encoded(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in public.items():
        destination = args.output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    for name, expected in manifest["files"].items():
        path = args.output_dir / name
        require(
            path.stat().st_size == expected["bytes"]
            and sha(path) == expected["sha256"],
            "Public payload matches explicit manifest",
        )
    with zipfile.ZipFile(
        args.archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output_dir).as_posix())
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        require(
            len(names) == len(set(names)) == len(manifest["files"]) + 1,
            "Complete unique archive members",
        )
        require(
            set(names) == set(manifest["files"]) | {"package-manifest.json"},
            "Exact archive payload set",
        )
        for name, expected in manifest["files"].items():
            data = archive.read(name)
            require(
                len(data) == expected["bytes"]
                and hashlib.sha256(data).hexdigest() == expected["sha256"],
                "Archive member SHA/size",
            )
    print(
        json.dumps(
            {
                "status": "local_export_only_no_publication",
                "archive_sha256": sha(args.archive),
                "archive_bytes": args.archive.stat().st_size,
                "manifest_sha256": sha(args.output_dir / "package-manifest.json"),
                "redacted_file_count": len(redactions),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
