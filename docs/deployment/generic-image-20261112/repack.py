"""CPU-only image inventory, guarded repack planning and filesystem comparison."""

import argparse
import email
import hashlib
import json
import posixpath
import re
import subprocess
import tarfile
import uuid
from pathlib import Path, PurePosixPath

IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
CACHE_ROOTS = (
    "root/.cache/uv",
    "root/.cache/pip",
    "tmp/uv",
    "var/cache/apt/archives",
    "root/.cache/ccache",
)
SUPPORTED_CONFIG = {
    "User",
    "Env",
    "Entrypoint",
    "Cmd",
    "WorkingDir",
    "Labels",
    "ExposedPorts",
    "Volumes",
    "StopSignal",
    "Healthcheck",
    "Shell",
    "OnBuild",
    "ArgsEscaped",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def canonical_config(config):
    return {k: v for k, v in config.items() if v not in (None, False, "", [], {})}


def safe_path(value):
    value = value.removeprefix("/").removeprefix("./").rstrip("/")
    if not value or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError("Unsafe image path")
    if not re.fullmatch(r"[A-Za-z0-9_./+@=-]+", value):
        raise ValueError("Unsupported path characters")
    return value


def under(path, prefix):
    return path == prefix or path.startswith(prefix + "/")


def package_metadata(data):
    headers = email.message_from_bytes(data)
    return {"name": headers.get("Name"), "version": headers.get("Version")}


def inventory(stream):
    entries = {}
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            path = member.name.removeprefix("./").rstrip("/")
            if not path:
                continue
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise ValueError("Unsafe export member")
            if path in entries:
                raise ValueError("Duplicate export member")
            entry = {"mode": member.mode, "uid": member.uid, "gid": member.gid}
            if not member.isdir():
                entry["mtime"] = member.mtime
            entry["xattrs"] = {
                k: v for k, v in member.pax_headers.items() if "xattr" in k.lower()
            }
            if member.isfile():
                entry["type"] = "file"
                digest = hashlib.sha256()
                metadata = bytearray() if path.endswith(".dist-info/METADATA") else None
                source = archive.extractfile(member)
                while block := source.read(4 * 1024**2):
                    digest.update(block)
                    if metadata is not None:
                        metadata.extend(block)
                entry.update(size=member.size, sha256=digest.hexdigest())
                if metadata is not None:
                    entry["package"] = package_metadata(bytes(metadata))
            elif member.isdir():
                entry["type"] = "dir"
            elif member.issym():
                entry.update(type="symlink", target=member.linkname)
            elif member.islnk():
                entry.update(
                    type="hardlink",
                    target=member.linkname.removeprefix("./").removeprefix("/"),
                )
            elif member.ischr() or member.isblk() or member.isfifo():
                entry.update(
                    type="char"
                    if member.ischr()
                    else "block"
                    if member.isblk()
                    else "fifo",
                    major=member.devmajor,
                    minor=member.devminor,
                )
            else:
                raise ValueError("Unsupported export member type")
            entries[path] = entry
    for path, entry in entries.items():
        if entry["type"] == "hardlink":
            target = entry["target"]
            visited = {path}
            while entries[target]["type"] == "hardlink":
                if target in visited:
                    raise ValueError("Hardlink cycle")
                visited.add(target)
                target = entries[target]["target"]
            original = entries[target]
            if original["type"] != "file":
                raise ValueError("Hardlink target is not a file")
            entry.update(
                type="file",
                size=original["size"],
                sha256=original["sha256"],
                hardlink_target=entry.pop("target"),
            )
    return entries


def snapshot(args):
    if args.inspect_json:
        inspected = load(args.inspect_json)
        inspected = inspected[0] if isinstance(inspected, list) else inspected
    else:
        inspected = json.loads(
            subprocess.check_output(["docker", "image", "inspect", args.image_id])
        )[0]
    if not IMAGE_ID.fullmatch(args.image_id) or inspected["Id"] != args.image_id:
        raise ValueError("Candidate image ID mismatch")
    if inspected["Config"].get("Volumes"):
        raise ValueError(
            "docker export omits volume data; use a volume-preserving workflow"
        )
    if args.export_tar:
        with open(args.export_tar, "rb") as stream:
            entries = inventory(stream)
    else:
        container = None
        try:
            container = subprocess.check_output(
                [
                    "docker",
                    "create",
                    "--runtime=runc",
                    "--network=none",
                    "--entrypoint=/bin/true",
                    "--name=release-audit-" + uuid.uuid4().hex,
                    args.image_id,
                ],
                text=True,
            ).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", container):
                raise ValueError("Unexpected container ID")
            process = subprocess.Popen(
                ["docker", "export", container],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                entries = inventory(process.stdout)
            except BaseException:
                process.terminate()
                process.communicate()
                raise
            _, errors = process.communicate()
            if process.returncode:
                raise ValueError("docker export failed; no unfiltered stderr emitted")
        finally:
            if container and re.fullmatch(r"[0-9a-f]{64}", container):
                subprocess.run(
                    ["docker", "rm", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
    save(
        args.output,
        {
            "image_id": args.image_id,
            "architecture": inspected["Architecture"],
            "os": inspected["Os"],
            "config": inspected["Config"],
            "entries": entries,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "entries": len(entries),
                "container_started": False,
            }
        )
    )


def removal_paths(snapshot, policy):
    entries = snapshot["entries"]
    active = safe_path(policy["active_site_packages"])
    system = safe_path(policy["system_site_packages"])
    if under(active, system) or under(system, active):
        raise ValueError("Active venv and legacy system roots must be disjoint")
    removed = []
    for rule in policy.get("remove_shadowed", []):
        path = safe_path(rule["path"])
        if not re.fullmatch(
            r"0\.7\.2(?:\.post\d+)?(?:\+cu\d+)?", rule["legacy_version"]
        ):
            raise ValueError("Only the reviewed legacy 0.7.2 assets are eligible")
        if "/" in rule["legacy_name"] or rule["legacy_name"] not in (
            "flashinfer-cubin",
            "flashinfer-jit-cache",
        ):
            raise ValueError(
                "Only identified legacy FlashInfer payloads may be shadow-pruned"
            )
        if path != safe_path(policy["system_site_packages"]) + "/" + rule[
            "legacy_name"
        ].replace("-", "_"):
            raise ValueError("Legacy removal must name the exact system payload")
        old = entries[safe_path(rule["legacy_metadata"])]["package"]
        replacement_path = safe_path(rule["replacement_metadata"])
        if not under(replacement_path, active):
            raise ValueError("Replacement metadata is not in the active venv")
        replacement = entries[replacement_path]["package"]
        normalized = lambda name: re.sub(r"[-_.]+", "-", name).lower()
        if (
            normalized(old["name"]) != rule["legacy_name"]
            or old["version"] != rule["legacy_version"]
        ):
            raise ValueError("Legacy package guard failed")
        if (
            normalized(replacement["name"]) != rule["legacy_name"]
            or replacement["version"] != rule["replacement_version"]
            or replacement["version"] == old["version"]
        ):
            raise ValueError("Active replacement guard failed")
        if not any(under(name, path) for name in entries):
            raise ValueError("Legacy payload is missing")
        replacement_payload = active + "/" + rule["legacy_name"].replace("-", "_")
        if not any(
            under(name, replacement_payload) and entry["type"] == "file"
            for name, entry in entries.items()
        ):
            raise ValueError("Active replacement has metadata but no payload")
        if rule["legacy_name"] == "flashinfer-jit-cache":
            for provider in policy.get("required_provider_ids", ["sm90a"]):
                if not re.fullmatch(r"sm[0-9]+[a-z]?", provider):
                    raise ValueError("Invalid protected provider ID")
                prefix = replacement_payload + "/providers/" + provider
                if not any(
                    under(name, prefix) and entry["type"] == "file"
                    for name, entry in entries.items()
                ):
                    raise ValueError(
                        "A required active architecture provider is absent"
                    )
        removed.append(path)
    for path in policy.get("remove_caches", []):
        path = safe_path(path)
        if path not in CACHE_ROOTS:
            raise ValueError("Cache path is not in the narrow cache allowlist")
        removed.append(path)
    if len(set(removed)) != len(removed):
        raise ValueError("Duplicate exclusions")
    for name, entry in entries.items():
        if entry["type"] != "symlink" or any(under(name, x) for x in removed):
            continue
        target = entry["target"]
        target = posixpath.normpath(
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.join(posixpath.dirname(name), target)
        )
        if any(under(target, prefix) for prefix in removed):
            raise ValueError("A retained symlink depends on an excluded path")
    return removed


def quote(value):
    if any(char in str(value) for char in ("\r", "\n", "\0")):
        raise ValueError("Multiline Dockerfile value requires explicit review")
    return json.dumps(str(value), ensure_ascii=True).replace("$", "\\$")


def runtime_lines(config, *, os_name=None):
    unknown = {
        key
        for key, value in config.items()
        if key not in SUPPORTED_CONFIG and value not in (None, False, "", [], {})
    }
    if (
        unknown
        or config.get("Volumes")
        or config.get("OnBuild")
        or (config.get("ArgsEscaped") and os_name != "linux")
    ):
        raise ValueError("Unrepresentable runtime configuration; do not drop fields")
    lines = []
    names = set()
    for item in config.get("Env") or []:
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in names:
            raise ValueError("Unsupported or duplicate environment key")
        names.add(key)
        lines.append("ENV " + key + "=" + quote(value))
    for key, value in sorted((config.get("Labels") or {}).items()):
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Image labels must be strings")
        lines.append("LABEL " + quote(key) + "=" + quote(value))
    for port in sorted(config.get("ExposedPorts") or {}):
        if not re.fullmatch(r"\d{1,5}/(?:tcp|udp|sctp)", port):
            raise ValueError("Unsupported exposed port")
        lines.append("EXPOSE " + port)
    if config.get("Shell"):
        lines.append("SHELL " + json.dumps(config["Shell"]))
    if config.get("User"):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*", config["User"]):
            raise ValueError("Unsupported user/group syntax")
        lines.append("USER " + config["User"])
    if config.get("WorkingDir"):
        if not re.fullmatch(r"/[A-Za-z0-9_./+@=-]*", config["WorkingDir"]):
            raise ValueError(
                "Working directory requires explicit Dockerfile path review"
            )
        lines.append("WORKDIR " + config["WorkingDir"])
    if config.get("StopSignal"):
        if not re.fullmatch(r"[A-Za-z0-9]+", config["StopSignal"]):
            raise ValueError("Unsupported stop signal")
        lines.append("STOPSIGNAL " + config["StopSignal"])
    health = config.get("Healthcheck")
    if health:
        test = health["Test"]
        if test == ["NONE"]:
            lines.append("HEALTHCHECK NONE")
        else:
            flags = []
            names = {
                "Interval": "interval",
                "Timeout": "timeout",
                "StartPeriod": "start-period",
                "StartInterval": "start-interval",
                "Retries": "retries",
            }
            if set(health) - {"Test", *names}:
                raise ValueError("Unsupported healthcheck metadata")
            for key, flag in names.items():
                if health.get(key):
                    flags.append(
                        f"--{flag}={int(health[key])}"
                        + ("" if key == "Retries" else "ns")
                    )
            if test[0] == "CMD":
                command = json.dumps(test[1:])
            elif (
                test[0] == "CMD-SHELL"
                and len(test) == 2
                and "\n" not in test[1]
                and "\r" not in test[1]
            ):
                command = test[1]
            else:
                raise ValueError("Unsupported healthcheck command")
            lines.append("HEALTHCHECK " + " ".join(flags) + " CMD " + command)
    for key, directive in (("Entrypoint", "ENTRYPOINT"), ("Cmd", "CMD")):
        if config.get(key) is not None:
            if not isinstance(config[key], list):
                raise ValueError("Exec metadata must be an array")
            lines.append(directive + " " + json.dumps(config[key]))
    return lines


def plan(args):
    source, policy = load(args.snapshot), load(args.policy)
    if not IMAGE_ID.fullmatch(args.image_id) or source["image_id"] != args.image_id:
        raise ValueError("Source inventory is not bound to the exact candidate")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]+", args.source_ref):
        raise ValueError("Unsafe local source reference")
    removed = removal_paths(source, policy)
    parts = [safe_path(path) for path in policy.get("layer_partitions", [])]
    for i, path in enumerate(parts):
        if path not in source["entries"]:
            raise ValueError("Layer partition does not exist")
        if source["entries"][path]["type"] != "dir":
            raise ValueError(
                "Partitions must be real directories, never CUDA or SDK symlink aliases"
            )
        if any(under(path, x) or under(x, path) for x in parts[:i] + removed):
            raise ValueError("Layer partitions/exclusions overlap")
    config = dict(source["config"])
    labels = load(args.labels_json)
    for key, value in (config.get("Labels") or {}).items():
        if key.startswith("com.nvidia.") and labels.get(key) != value:
            raise ValueError("Preserve NVIDIA runtime/license labels exactly")
    for key in ("org.opencontainers.image.revision", "ai.positive666.lmcache.revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", labels.get(key, "")):
            raise ValueError("Final labels must provide exact current source revisions")
    active = safe_path(policy["active_site_packages"])
    versions = [
        entry["package"]["version"]
        for path, entry in source["entries"].items()
        if under(path, active)
        and entry.get("package", {}).get("name", "").lower() == "vllm"
    ]
    if (
        len(versions) != 1
        or labels.get("org.opencontainers.image.version") != versions[0]
    ):
        raise ValueError(
            "Final version label must match the unique installed active vLLM metadata"
        )
    commit = re.search(r"[+.]g([0-9a-f]{7,40})(?:[.]|$)", versions[0])
    if commit and not labels["org.opencontainers.image.revision"].startswith(commit[1]):
        raise ValueError(
            "vLLM revision label disagrees with installed version provenance"
        )
    lm_versions = [
        entry["package"]["version"]
        for path, entry in source["entries"].items()
        if under(path, active)
        and entry.get("package", {}).get("name", "").lower() == "lmcache"
    ]
    if len(lm_versions) != 1:
        raise ValueError("Exactly one active LMCache metadata record is required")
    commit = re.search(r"[+.]g([0-9a-f]{7,40})(?:[.]|$)", lm_versions[0])
    if commit and not labels["ai.positive666.lmcache.revision"].startswith(commit[1]):
        raise ValueError(
            "LMCache revision label disagrees with installed version provenance"
        )
    labels.setdefault("ai.positive666.lmcache.version", lm_versions[0])
    if labels["ai.positive666.lmcache.version"] != lm_versions[0]:
        raise ValueError("LMCache version label is stale")
    config["Labels"] = labels
    lines = [
        "# syntax=docker/dockerfile:1.19",
        "FROM " + args.source_ref + " AS frozen",
        "FROM scratch",
    ]
    lines.append(
        "COPY --from=frozen "
        + " ".join("--exclude=" + p for p in removed + parts)
        + " / /"
    )
    lines.extend(f"COPY --from=frozen /{part}/ /{part}/" for part in parts)
    lines.extend(runtime_lines(config, os_name=source["os"]))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "Dockerfile").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "source_image_id": args.image_id,
        "source_ref": args.source_ref,
        "architecture": source["architecture"],
        "os": source["os"],
        "removed_paths": removed,
        "layer_partitions": parts,
        "expected_runtime_config": config,
        "snapshot_sha256": hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
    }
    save(out / "REPACK_PLAN.json", result)
    build_lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        'cd -- "$(dirname -- "$0")"',
        f"actual=$(docker image inspect --format '{{{{.Id}}}}' '{args.source_ref}')",
        f"test \"$actual\" = '{args.image_id}'",
        "docker buildx build --pull=false --network=none --provenance=false "
        "--output type=oci,dest=image.oci.tar,compression=gzip --file Dockerfile .",
        f"actual=$(docker image inspect --format '{{{{.Id}}}}' '{args.source_ref}')",
        f"test \"$actual\" = '{args.image_id}'",
    ]
    (out / "build-repack.sh").write_text(
        "\n".join(build_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "plan": str(out / "REPACK_PLAN.json"),
                "removed_paths": removed,
                "partition_count": len(parts),
                "build_executed": False,
            }
        )
    )


def compare(args):
    before, after, expected = load(args.before), load(args.after), load(args.plan)
    if before["image_id"] != expected["source_image_id"]:
        raise ValueError("Wrong baseline image")
    removed = expected["removed_paths"]
    wanted = {
        p: e
        for p, e in before["entries"].items()
        if not any(under(p, x) for x in removed)
    }

    def normalized(entry):
        return {
            k: v for k, v in entry.items() if k not in {"hardlink_target", "package"}
        }

    missing = sorted(set(wanted) - set(after["entries"]))
    extra = sorted(set(after["entries"]) - set(wanted))
    changed = sorted(
        p
        for p in set(wanted) & set(after["entries"])
        if normalized(wanted[p]) != normalized(after["entries"][p])
    )
    expected_config, actual_config = (
        canonical_config(expected["expected_runtime_config"]),
        canonical_config(after["config"]),
    )
    config_changes = sorted(
        k
        for k in set(expected_config) | set(actual_config)
        if expected_config.get(k) != actual_config.get(k)
    )
    result = {
        "status": "PASS"
        if not (missing or extra or changed or config_changes)
        else "FAIL",
        "source_image_id": before["image_id"],
        "candidate_image_id": after["image_id"],
        "retained_entries": len(wanted),
        "removed_entries": len(before["entries"]) - len(wanted),
        "missing": missing,
        "unexpected": extra,
        "changed": changed,
        "runtime_config_changed_keys": config_changes,
        "hardlinks": "Resolved bytes compared; physical inode sharing not asserted",
    }
    save(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("missing", "unexpected", "changed")
            },
            indent=2,
        )
    )
    if result["status"] != "PASS":
        raise ValueError(
            "Repack file/runtime equivalence failed; inspect path-only report"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("snapshot")
    p.add_argument("--image-id", required=True)
    p.add_argument("--inspect-json", type=Path)
    p.add_argument("--export-tar", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=snapshot)
    p = subs.add_parser("plan")
    for name in ("snapshot", "policy"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image-id", required=True)
    p.add_argument("--source-ref", required=True)
    p.add_argument("--labels-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=plan)
    p = subs.add_parser("compare")
    for name in ("before", "after", "plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
