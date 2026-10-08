"""Build derived late-log closeout sidecars without changing raw records/logs.

Existing sidecar entries must verify unchanged. New entries are accepted only
after audit_tp2.verify_log validates an exact newline prefix and a bounded
informational/cleanup suffix. No GPU, service or publication operation occurs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from audit_tp2 import Audit, read, sha, verify_log


def entry_for(record_path, log_path):
    record, data = read(record_path), log_path.read_bytes()
    expected = record["server_log_sha256"]
    if re.fullmatch(r"[0-9a-f]{64}", expected or "") is None:
        raise ValueError("Invalid recorded log SHA: " + str(record_path))
    actual = hashlib.sha256(data).hexdigest()
    if actual == expected:
        return None
    state, offset, matches = hashlib.sha256(), 0, []
    for line in data.splitlines(keepends=True):
        state.update(line)
        offset += len(line)
        if line.endswith(b"\n") and state.hexdigest() == expected:
            matches.append(offset)
    if len(matches) != 1:
        raise ValueError("Recorded SHA is not a unique newline prefix: " + str(log_path))
    count = matches[0]
    prefix, tail = data[:count], data[count:]
    prefix_text, tail_text = prefix.decode("utf-8"), tail.decode("utf-8")
    return {
        "record_file": record_path.name,
        "original_record_sha256": sha(record_path),
        "recorded_prefix_bytes": count,
        "recorded_prefix_sha256": expected,
        "full_actual_bytes": len(data),
        "full_actual_sha256": actual,
        "appended_tail_bytes": len(tail),
        "appended_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "tail_classification": "late_buffered_nccl_initialization_and_shutdown_output",
        "prefix_quality_http200": len(re.findall(
            r'POST /v1/chat/completions HTTP/1\.1" 200 OK', prefix_text
        )),
        "prefix_compatibility_http200": len(re.findall(
            r'POST /v1/completions HTTP/1\.1" 200 OK', prefix_text
        )),
        "prefix_application_shutdown_complete": "Application shutdown complete" in prefix_text,
        "tail_quality_http200": len(re.findall(
            r'POST /v1/chat/completions HTTP/1\.1" 200 OK', tail_text
        )),
        "tail_text": tail_text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", type=Path, help="Optional new summary JSON; exclusive creation")
    args = parser.parse_args()
    root = args.results.resolve(strict=True)
    audit, plans, original_hashes = Audit(), {}, {}
    report = {"results_root": str(root), "dry_run": args.dry_run,
              "logs": {}, "sidecars": {}, "errors": audit.errors, "notes": audit.notes}
    records = sorted(root.rglob("serve-*.json"))
    if not records:
        raise ValueError("No server records under " + str(root))
    for record_path in records:
        # This builder handles only completed supervisor records with an original log hash.
        record = read(record_path)
        if "server_log_sha256" not in record:
            raise ValueError("Server record has no final log hash: " + str(record_path))
        log_path = record_path.with_suffix(".log")
        if not log_path.is_file():
            raise ValueError("Server log absent: " + str(log_path))
        directory = record_path.parent.resolve(strict=True)
        if not directory.is_relative_to(root):
            raise ValueError("Server record outside the named results root")
        if directory not in plans:
            sidecar = directory / "log-closeout-manifest.json"
            existing = read(sidecar) if sidecar.exists() else {"format_version": 1, "entries": {}}
            if not isinstance(existing.get("entries"), dict):
                raise ValueError("Malformed existing sidecar: " + str(sidecar))
            plans[directory] = {
                "path": sidecar, "existing": existing,
                "existing_sha256": sha(sidecar) if sidecar.exists() else None,
                "candidate": copy.deepcopy(existing), "added": [],
            }
        plan = plans[directory]
        original_hashes[record_path], original_hashes[log_path] = sha(record_path), sha(log_path)
        entry = entry_for(record_path, log_path)
        if entry is not None and log_path.name not in plan["candidate"]["entries"]:
            plan["candidate"]["entries"][log_path.name] = entry
            plan["added"].append(log_path.name)
        validation = verify_log(
            log_path, record_path, record["server_log_sha256"], audit,
            str(log_path.relative_to(root)), manifest_override=plan["candidate"],
        )
        report["logs"][str(log_path.relative_to(root))] = validation
    # Preserve and independently validate every pre-existing entry, including any
    # records not found by the initial serve-* scan.
    for directory, plan in plans.items():
        for name, entry in plan["existing"]["entries"].items():
            if Path(name).name != name or Path(entry["record_file"]).name != entry["record_file"]:
                raise ValueError("Non-local filename in existing closeout entry")
            log_path, record_path = directory / name, directory / entry["record_file"]
            if not log_path.is_file() or not record_path.is_file():
                raise ValueError("Existing closeout entry lacks original files")
            original_hashes.setdefault(record_path, sha(record_path))
            original_hashes.setdefault(log_path, sha(log_path))
            record = read(record_path)
            audit.check(
                entry == plan["candidate"]["entries"][name],
                "Pre-existing sidecar entry changed: " + name,
            )
            verify_log(log_path, record_path, record["server_log_sha256"], audit, name,
                       manifest_override=plan["existing"])
        if plan["added"]:
            plan["candidate"].setdefault("derived_updates", []).append({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "previous_manifest_sha256": plan["existing_sha256"],
                "added_log_names": sorted(plan["added"]),
                "builder_sha256": sha(Path(__file__)),
                "validator_sha256": sha(Path(__file__).with_name("audit_tp2.py")),
                "raw_bytes_changed": False,
            })
    for path, expected in original_hashes.items():
        audit.check(sha(path) == expected, "Raw file changed while being audited: " + str(path))
    if audit.errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    # Commit derived sidecars only after all roots, original entries and tails pass.
    for directory, plan in plans.items():
        sidecar = plan["path"]
        audit.check(
            (sha(sidecar) if sidecar.exists() else None) == plan["existing_sha256"],
            "Existing sidecar changed during validation: " + str(sidecar),
        )
    if audit.errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    for directory, plan in plans.items():
        sidecar = plan["path"]
        written = False
        if plan["added"] and not args.dry_run:
            encoded = (json.dumps(plan["candidate"], ensure_ascii=False, indent=2) + "\n").encode()
            with tempfile.NamedTemporaryFile(prefix=".closeout-derived-", dir=directory,
                                             delete=False) as handle:
                handle.write(encoded)
                temporary = Path(handle.name)
            try:
                temporary.replace(sidecar)
            finally:
                temporary.unlink(missing_ok=True)
            written = True
        report["sidecars"][str(sidecar.relative_to(root))] = {
            "existing_entries_preserved": sorted(plan["existing"]["entries"]),
            "new_entries": sorted(plan["added"]), "written": written,
            "previous_sha256": plan["existing_sha256"],
            "final_sha256": sha(sidecar) if sidecar.exists() else None,
        }
    # Recheck raw files after all derived writes; no alteration is permitted.
    for path, expected in original_hashes.items():
        audit.check(sha(path) == expected, "Raw file changed after sidecar write: " + str(path))
    report["validation_passed"] = not audit.errors
    report["server_logs_audited"] = len(records)
    report["builder_sha256"] = sha(Path(__file__))
    report["validator_sha256"] = sha(Path(__file__).with_name("audit_tp2.py"))
    if args.summary:
        with args.summary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["validation_passed"] else 1)


if __name__ == "__main__":
    main()
