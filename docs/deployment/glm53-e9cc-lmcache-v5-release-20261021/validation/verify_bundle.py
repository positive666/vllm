# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the static public docs package without registry or service access."""

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "clean_fp8": (
        "sha256:12e65dc97ab78d320689aad552f8e232ab74227f5ac1055e333c76b1aba76d45",
        45,
        0,
    ),
    "flash_producer": (
        "sha256:536d9bb5dd69adcb9306d90c4abbc592579bbd4efdbf263559eb049b2728eb5f",
        46,
        1,
    ),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load(relative):
    return json.loads((ROOT / relative).read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-published", action="store_true")
    args = parser.parse_args()
    manifest = load("BUNDLE_MANIFEST.json")
    allowed = {row["path"] for row in manifest["files"]}
    actual = {
        path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()
    }
    need(actual == allowed | {"BUNDLE_MANIFEST.json"}, "Exact public file set required")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        need(
            path.resolve().is_relative_to(ROOT)
            and not path.is_symlink()
            and path.stat().st_size == row["bytes"]
            and digest(path.read_bytes()) == row["sha256"],
            "Public file bytes differ: " + row["path"],
        )
    versions = load("versions.json")
    need(
        versions["sources"]["vllm_commit"] == "e9cc6244a0095ba6525230b49a8ecdc2cf7b0f01"
        and versions["sources"]["lmcache_base_commit"]
        == "2a34fa829b3c902c5b6393740239f6ff65bd0ff8"
        and versions["production"] is False,
        "Fixed new-main source pins and experimental status required",
    )
    registry = load("validation/registry-verification.json")
    for key, (config, count, private_files) in EXPECTED.items():
        profile = versions["profiles"][key]
        need(
            profile["config_digest"] == config
            and profile["rootfs_diffids_count"] == count
            and len(profile["rootfs_diffids"]) == count
            and profile["vllm_private_runtime_files"] == private_files
            and digest(
                json.dumps(profile["rootfs_diffids"], separators=(",", ":")).encode()
            )
            == profile["rootfs_diffids_canonical_json_sha256"],
            "Independent exact image and ordered layers required: " + key,
        )
        proof = registry["profiles"][key]
        need(
            proof["config_digest"] == config
            and proof["proposed_tag"] == profile["proposed_tag"],
            "Registry proof profile association differs",
        )
        if args.require_published or proof["registry_manifest_digest"] is not None:
            need(
                proof["anonymous_verified"] is True
                and profile["anonymous_registry_verified"] is True
                and re.fullmatch(
                    r"sha256:[a-f0-9]{64}", proof["registry_manifest_digest"]
                )
                and proof["registry_manifest_digest"]
                == profile["registry_manifest_digest"],
                "Actual published anonymous digest proof required",
            )
        else:
            need(
                proof["registry_manifest_digest"] is None
                and profile["registry_manifest_digest"] is None
                and proof["anonymous_verified"] is False
                and profile["anonymous_registry_verified"] is False,
                "Prepared package must preserve explicit publication pending",
            )
    need(
        versions["profiles"]["flash_producer"]["rootfs_diffids"][:-1]
        == versions["profiles"]["clean_fp8"]["rootfs_diffids"],
        "Flash must retain the exact clean FP8 parent layers",
    )
    results = load("validation/results.json")
    need(
        results["FP8_TP16"]["C32"]["all_gold_SSE_usage_correct"] is True
        and results["Flash_TP8"]["requests_correct"] == 8
        and results["Flash_TP8"]["model_C32_or_loading_cancel_or_TTL_long_run_tested"]
        is False
        and results["production_SLO_accepted"] is False,
        "Model-specific functional evidence and limits required",
    )
    for path in (ROOT / leaf for leaf in sorted(actual)):
        text = path.read_text(encoding="utf-8")
        for candidate in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            need(address.is_loopback or address.is_global, "Private address excluded")
        need(
            not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}", text)
            and not re.search(r"-{5}BEGIN (?:[A-Z]+ )?PRIVATE KEY-{5}", text)
            and not re.search(r"[A-Za-z]:[/\\](?:Users|code)[/\\]", text)
            and ("glm53" + "-fleet-") not in text
            and ("/data/glm53" + "-upstream-") not in text,
            "Private host path or credential material excluded: " + path.name,
        )
    print(
        json.dumps(
            {
                "status": "PASS_STATIC_PUBLIC_DOCS_NOT_SERVING_OR_REGISTRY_TEST",
                "files": len(actual),
                "profiles": list(EXPECTED),
                "registry_verified_by_this_command": False,
                "publication_required": args.require_published,
            }
        )
    )


if __name__ == "__main__":
    main()
