"""Prepare immutable local evidence; never commit or publish."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    require(src != dst and src not in dst.parents and dst not in src.parents, "Overlapping source/destination")
    require(not dst.exists() and not args.archive.exists(), "Output already exists")
    quality = json.loads((src / "quality-final8-v1.json").read_text())
    ownership = json.loads((src / "ownership-final8-v1.json").read_text())
    require(digest((src / "quality-final8-v1.json").read_bytes()) == "6ad38410daa30971f5bbf64d40462c11f39f99a1b69c4afa8c3c5280b15cf4f1", "Quality audit changed")
    require(digest((src / "ownership-final8-v1.json").read_bytes()) == "588fb039d3f0eedfdf733d2bb4daf159aecd8f811fbe65fd17233bd7ae48c0a5", "Ownership audit changed")
    require(quality["integrity_pass"] and ownership["integrity_pass"], "Audit failed")
    require(digest((src / "helpers/audit_long_free.py").read_bytes()) == quality["auditor_sha256"], "Quality auditor snapshot changed")
    require(digest((src / "helpers/audit_host_ownership.py").read_bytes()) == ownership["auditor_sha256"], "Ownership auditor snapshot changed")
    require(digest((src / "helpers/monitor_host.sh").read_bytes()) == ownership["monitor_sha256"], "Monitor snapshot changed")
    runs = quality["runs"]
    require(len(runs) == 8 and len({r["run"] for r in runs}) == 8, "Incomplete matrix")
    payload = {}

    def add(source, name=None):
        name = name or source.relative_to(src).as_posix()
        require(safe_name(name) and name not in payload, "Unsafe/duplicate member")
        require(source.is_file() and not source.is_symlink(), "Missing/linked input")
        payload[name] = source.read_bytes()

    archive_inputs = {}
    for run in runs:
        name = run["run"]
        require(safe_name(name), "Unsafe run name")
        for filename, expected in run["input_sha256"].items():
            path = src / name / filename if (src / name / filename).is_file() else src / filename
            require(digest(path.read_bytes()) == expected, "Run input digest mismatch: " + name + "/" + filename)
        require((src / (name + ".exit")).read_text().strip() == "0", "Launch failed")
        completed = json.loads((src / name / "completed.json").read_text())
        require(len(completed["examples"]) == 8, "Missing outputs")
        launch = json.loads((src / name / "launch.json").read_text())
        for filename, expected in launch["helper_sha256"].items():
            require(digest((src / name / "helpers" / filename).read_bytes()) == expected, "Frozen producer helper changed")
        for path in sorted((src / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path)
        for suffix in (".exit", ".log", "-host.csv", "-processes.log"):
            add(src / (name + suffix))
        transport = src / (name + ".tar.gz")
        archive_inputs[transport.name] = {"sha256": digest(transport.read_bytes()), "bytes": transport.stat().st_size}

    required = ["protocol-plan.json", "design-note.md", "harness-review.md", "quality-final8-v1.json", "ownership-final8-v1.json", "repeat-comparison-local8-v1.json", "after8-current-idle.txt", "container-cleanup.txt"]
    for filename in required:
        add(src / filename)
    for name in ("initial-matrix", "reverse-matrix", "quality-final8-v1", "ownership-final8-v1"):
        require((src / (name + ".exit")).read_text().strip() == "0", "Matrix/auditor failed: " + name)
        add(src / (name + ".exit"))
        add(src / (name + ".log"))
    for filename in ("long-audit-local-initial-v1.json", "long-audit-remote-initial-v1.json", "triton-eager-1-audit-v1.json", "triton-eager-1-audit-v2.json", "ownership-initial4-v1.json", "quality-local5-v1.json", "ownership-local5-v1.json", "quality-local8-v1.json", "ownership-local8-v1.json", "flashinfer-eager-repeat-local-v1.json"):
        add(src / filename, "historical-audits/" + filename)
    for path in sorted((src / "helpers").iterdir()):
        if path.suffix in (".py", ".sh"):
            add(path)
    add(src / "evidence-readme-draft.md", "README.md")
    payload[".gitattributes"] = b"* -text whitespace=cr-at-eol,-blank-at-eof\n"

    patterns = {
        "github_token": rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
        "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "authorization": rb"(?i)authorization\s*[:=]\s*[\"']?(?:bearer|token|basic)\s+[A-Za-z0-9+/=_-]{16,}",
        "literal_credential": rb"(?i)\b(?:password|passwd|pwd|api_key|access_token|secret_key)\b\s*[:=]\s*[\"']([^\"'\r\n]{3,})[\"']",
        "credential_flag": rb"(?i)--(?:password|passwd|api-key|access-token)(?:=|\s+)[^\s$'\"{}]{3,}",
        "credential_url": rb"https?://[^\s/:@]+:[^\s/@]+@",
    }
    findings = []
    for name, data in payload.items():
        for category, pattern in patterns.items():
            if re.search(pattern, data):
                findings.append({"path": name, "category": category})
    require(not findings, "Credential scan findings (content suppressed): " + json.dumps(findings))
    manifest = {
        "schema_version": 1,
        "source_head": quality["source_head"],
        "scope": quality["scope"],
        "matrix_runs": [r["run"] for r in runs],
        "transport_archives": archive_inputs,
        "excluded": ["transport archives (digests retained)", "bytecode", "remote access helpers", "draft PR text", "historical network checkpoint"],
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in sorted(payload.items())},
    }
    payload["package-manifest.json"] = json_bytes(manifest)
    dst.mkdir(parents=True)
    for name, data in sorted(payload.items()):
        path = dst / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        require(len(names) == len(payload) == len(set(names)), "Archive member count mismatch")
        require(all(safe_name(name) for name in names), "Unsafe archive entry")
        for name in names:
            require(archive.read(name) == payload[name], "Archive bytes mismatch")
    receipt = {
        "status": "local_preparation_pass_no_publication",
        "archive": str(args.archive.resolve()),
        "archive_sha256": digest(args.archive.read_bytes()),
        "archive_bytes": args.archive.stat().st_size,
        "manifest_sha256": digest(payload["package-manifest.json"]),
        "payload_file_count": len(manifest["files"]),
        "archive_member_count": len(payload),
        "all_member_hashes_and_sizes_verified": True,
        "credential_scan_findings": findings,
        "credential_scan_scope": "High-confidence text-pattern scan of every payload file; not a proof that no secret exists.",
        "quality_integrity_pass": quality["integrity_pass"],
        "strict_same_sample_ownership_pass": ownership["strict_same_sample_ownership_pass"],
        "notes": ["No production code changed", "No commit/push/PR comment", "Separate owned-container stop receipt included; no files/container removal claim"],
    }
    args.receipt.write_bytes(json_bytes(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
