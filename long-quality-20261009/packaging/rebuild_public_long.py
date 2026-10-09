"""Rebuild the unpublished final8 candidate with explicit IP redaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from public_log_redaction import sanitize_payload

OLD_SHA = "fcacd39022cec5a006045b31ed134d316379f71b1abe9eac81cbddede50b2583"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.original_archive.read_bytes()) != OLD_SHA:
        raise ValueError("Original unpublished candidate SHA")
    if args.destination.exists() or args.archive.exists() or args.receipt.exists():
        raise ValueError("Exclusive new outputs required")
    with zipfile.ZipFile(args.original_archive) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    old_manifest = json.loads(original.pop("package-manifest.json"))
    if set(original) != set(old_manifest["files"]):
        raise ValueError("Original manifest payload set")
    for name, raw in original.items():
        expected = old_manifest["files"][name]
        if digest(raw) != expected["sha256"] or len(raw) != expected["bytes"]:
            raise ValueError("Original frozen bytes: " + name)
    public, redactions = sanitize_payload(original)
    public["public-redactions.json"] = encoded(
        {
            "original_candidate_archive_sha256": OLD_SHA,
            "redacted_files": redactions,
            "scope": (
                "Public observation copies only. Raw logs are retained "
                "locally/remotely; original audit log digests remain raw "
                "digests. Frozen source files remain exact; literal "
                "localhost defaults in source/reference are not observed "
                "host-network addresses."
            ),
        }
    )
    public["README.md"] += (
        b"\n## Public observation copies\n\n"
        b"Private network addresses in public log/runtime observation copies "
        b"are explicitly redacted. `public-redactions.json` maps original "
        b"raw SHA/size to each public redacted SHA/size. Original auditor "
        b"log digests still identify the retained raw records; redacted "
        b"copies do not claim original-byte identity. Source/helper/math "
        b"records are preserved unchanged. The unredacted unpublished "
        b"candidate is retained locally and is not for publication. The "
        b"original integrity audits ran against remote raw records. Public "
        b"redacted copies cannot rerun every raw consumer provenance gate "
        b"byte for byte; the consumer acceptance rules are unchanged.\n"
    )
    public["packaging/public_log_redaction.py"] = (
        Path(__file__).with_name("public_log_redaction.py").read_bytes()
    )
    public["packaging/rebuild_public_long.py"] = Path(__file__).read_bytes()
    patterns = {
        "token": rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
        "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "credential_url": rb"https?://[^\s/:@]+:[^\s/@]+@",
    }
    findings = [
        {"path": name, "category": category}
        for name, data in public.items()
        for category, pattern in patterns.items()
        if re.search(pattern, data)
    ]
    if findings:
        raise ValueError(
            "Credential findings (values suppressed): " + json.dumps(findings)
        )
    manifest = {
        **{key: value for key, value in old_manifest.items() if key != "files"},
        "public_copy_scope": (
            "Explicit private-address redaction with raw/public SHA mapping"
        ),
        "original_candidate_sha256": OLD_SHA,
        "files": {
            name: {"sha256": digest(data), "bytes": len(data)}
            for name, data in sorted(public.items())
        },
    }
    public["package-manifest.json"] = encoded(manifest)
    args.destination.mkdir(parents=True, exist_ok=False)
    for name, data in public.items():
        path = args.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with zipfile.ZipFile(
        args.archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, data in sorted(public.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(args.archive) as archive:
        if set(archive.namelist()) != set(public) or len(archive.namelist()) != len(
            public
        ):
            raise ValueError("Archive payload set")
        for name, data in public.items():
            if archive.read(name) != data:
                raise ValueError("Archive frozen bytes: " + name)
    receipt = {
        "status": "local_public_candidate_no_publication",
        "original_candidate_sha256": OLD_SHA,
        "archive_sha256": digest(args.archive.read_bytes()),
        "archive_bytes": args.archive.stat().st_size,
        "manifest_sha256": digest(public["package-manifest.json"]),
        "members": len(public),
        "redacted_file_count": len(redactions),
        "redaction_counts": sum(
            sum(item["redaction_counts"].values()) for item in redactions
        ),
        "all_member_bytes_verified": True,
        "credential_findings": findings,
        "source_and_math_payload_preserved": True,
        "original_raw_candidate_retained_not_publishable": True,
    }
    args.receipt.write_bytes(encoded(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
