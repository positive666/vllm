# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check public bundle hashes and exclude host-specific or credential material."""

import hashlib
import ipaddress
import json
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "versions.json").read_text(encoding="utf-8"))
    checked = []
    for item in [*manifest["patches"], manifest["public_runtime_lock"]]:
        path = ROOT / item["path"]
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        data = path.read_bytes()
        assert len(data) == item["bytes"], item["path"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
        checked.append(item["path"])
    results = json.loads((ROOT / "validation/results.json").read_text(encoding="utf-8"))
    aggregates = [
        results["Linux_installed_regressions"]["public_aggregate"],
        results["independent_registry_verification"],
    ]
    fp8 = results.get("FP8_two_node_TP16_acceptance")
    if isinstance(fp8, dict):
        aggregates.append(fp8["public_aggregate"])
    for item in aggregates:
        path = ROOT / "validation" / item["path"]
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        checked.append("validation/" + item["path"])
    lock = (ROOT / manifest["public_runtime_lock"]["path"]).read_text()
    assert len(re.findall(r"(?m)^[a-zA-Z0-9_.-]+==", lock)) == 34
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        assert not path.is_symlink()
        text = path.read_text(encoding="utf-8")
        for value in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            try:
                address = ipaddress.ip_address(value)
            except ValueError:
                continue
            assert address.is_loopback or not address.is_private, path.name
        assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}", text), path.name
        assert not re.search(r"-{5}BEGIN (?:[A-Z]+ )?PRIVATE KEY-{5}", text), path.name
        assert not re.search(r"[A-Za-z]:[/\\](?:Users|code)[/\\]", text), path.name
        assert ("glm53" + "-fleet-") not in text, path.name
    print(
        json.dumps(
            {
                "status": "PASS_PUBLIC_BUNDLE_STATIC_ONLY",
                "checked_hashes": checked,
                "public_locked_packages": 34,
                "files_checked": len(files),
                "serving_acceptance": manifest["validation"]["current_cache_candidate"],
            }
        )
    )


if __name__ == "__main__":
    main()
