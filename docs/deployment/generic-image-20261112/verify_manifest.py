"""Verify local OCI or registry bytes, including every complete layer readback."""

import argparse
import base64
import gzip
import hashlib
import json
import re
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from repack import canonical_config

DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
GZIP_TYPES = {
    "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
}
TAR_TYPES = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.docker.image.rootfs.diff.tar",
}


def digest_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def check_descriptor(item):
    if (
        not DIGEST.fullmatch(item["digest"])
        or not isinstance(item["size"], int)
        or item["size"] < 0
    ):
        raise ValueError("Invalid OCI descriptor")


class HashReader:
    def __init__(self, source):
        self.source, self.digest, self.size = source, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data


def read_blob(source, descriptor, diff_id=None, inspect_tar=False):
    check_descriptor(descriptor)
    hashed = HashReader(source)
    if diff_id is None:
        if descriptor["size"] > 64 * 1024**2:
            raise ValueError("JSON config exceeds the explicit size bound")
        result = hashed.read(descriptor["size"] + 1)
        while hashed.read(1024**2):
            raise ValueError("JSON blob exceeds descriptor size")
        raw = result
    else:
        if not DIGEST.fullmatch(diff_id):
            raise ValueError("Invalid diffID")
        media = descriptor["mediaType"]
        if media in GZIP_TYPES:
            decoded = gzip.GzipFile(fileobj=hashed, mode="rb")
        elif media in TAR_TYPES:
            decoded = hashed
        else:
            raise ValueError("Unsupported layer encoding; export gzip or plain tar")
        uncompressed = HashReader(decoded)
        payload_paths = []
        if inspect_tar:
            with tarfile.open(
                fileobj=uncompressed, mode="r|", bufsize=4 * 1024**2
            ) as archive:
                for member in archive:
                    path = member.name.removeprefix("./").rstrip("/")
                    if path.startswith("/") or ".." in path.split("/"):
                        raise ValueError("Unsafe OCI layer path")
                    if path.rsplit("/", 1)[-1].startswith(".wh."):
                        raise ValueError(
                            "Flattened repack unexpectedly contains a whiteout"
                        )
                    if not member.isdir():
                        payload_paths.append(path)
        while uncompressed.read(4 * 1024**2):
            pass
        if decoded is not hashed:
            decoded.close()
        while hashed.read(1024**2):
            pass
        if "sha256:" + uncompressed.digest.hexdigest() != diff_id:
            raise ValueError("Layer uncompressed diffID mismatch")
        raw = None
    if (
        hashed.size != descriptor["size"]
        or "sha256:" + hashed.digest.hexdigest() != descriptor["digest"]
    ):
        raise ValueError("Complete blob size or compressed digest mismatch")
    check = {
        "digest": descriptor["digest"],
        "size": hashed.size,
        "diff_id": diff_id,
        "full_readback": True,
    }
    if diff_id is not None and inspect_tar:
        check["_payload_paths"] = payload_paths
    return raw, check


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.username or target.password:
            raise ValueError("Unsafe registry redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urlsplit(req.full_url).netloc != target.netloc:
            for name in list(redirected.headers):
                if name.lower() in ("authorization", "proxy-authorization"):
                    del redirected.headers[name]
        return redirected


class Registry:
    def __init__(self, reference, username=None, credential=None):
        registry, sep, rest = reference.partition("/")
        if not sep or not re.fullmatch(r"[A-Za-z0-9.-]+(?::\d+)?", registry):
            raise ValueError("Reference needs an explicit HTTPS registry")
        if "@" in rest:
            repository, ref = rest.rsplit("@", 1)
            if not DIGEST.fullmatch(ref):
                raise ValueError("Invalid digest reference")
        else:
            repository, sep, ref = rest.rpartition(":")
            if not sep or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", ref):
                raise ValueError("Explicit immutable tag or digest required")
        if not re.fullmatch(
            r"[a-z0-9][a-z0-9._/-]*", repository
        ) or ".." in repository.split("/"):
            raise ValueError("Invalid registry repository")
        self.registry, self.repository, self.reference = registry, repository, ref
        self.username, self.credential, self.token = username, credential, None
        self.opener, self.lock = build_opener(SafeRedirect()), threading.Lock()

    def authorize(self, challenge):
        if not challenge.startswith("Bearer "):
            raise ValueError("Unsupported registry authentication")
        fields = dict(re.findall(r'(\w+)="([^"\r\n]*)"', challenge))
        realm = urlsplit(fields["realm"])
        if (
            realm.scheme != "https"
            or realm.netloc != self.registry
            or realm.username
            or realm.password
            or realm.query
            or realm.fragment
        ):
            raise ValueError("Unexpected token issuer")
        query = urlencode(
            {
                "service": fields.get("service", self.registry),
                "scope": "repository:" + self.repository + ":pull",
            }
        )
        headers = {}
        if self.credential:
            if not self.username:
                raise ValueError("Username required for credential stdin")
            headers["Authorization"] = (
                "Basic "
                + base64.b64encode(
                    (self.username + ":" + self.credential).encode()
                ).decode()
            )
        with self.opener.open(
            Request(fields["realm"] + "?" + query, headers=headers), timeout=60
        ) as response:
            data = json.loads(response.read(2 * 1024**2))
        self.token = data.get("token") or data["access_token"]

    def open(self, suffix, accept=None):
        url = "https://" + self.registry + "/v2/" + self.repository + "/" + suffix
        for attempt in range(2):
            headers = {"Accept": accept} if accept else {}
            if self.token:
                headers["Authorization"] = "Bearer " + self.token
            try:
                return self.opener.open(Request(url, headers=headers), timeout=90)
            except HTTPError as error:
                if error.code != 401 or attempt:
                    raise ValueError(
                        "Registry HTTP status " + str(error.code)
                    ) from None
                challenge = error.headers.get("WWW-Authenticate", "")
                with self.lock:
                    self.authorize(challenge)
        raise ValueError("Registry authorization failed")


def verify_graph(
    manifest_raw,
    manifest_digest,
    get_blob,
    expected_plan=None,
    expected_config=None,
    workers=2,
    max_layer_bytes=10_000_000_000,
):
    if (
        not DIGEST.fullmatch(manifest_digest)
        or digest_bytes(manifest_raw) != manifest_digest
    ):
        raise ValueError("Manifest digest mismatch")
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("mediaType") not in MANIFEST_TYPES
        or manifest.get("schemaVersion") != 2
    ):
        raise ValueError("Expected a single image manifest, not an index or artifact")
    with get_blob(manifest["config"]) as source:
        config_raw, config_check = read_blob(source, manifest["config"])
    config = json.loads(config_raw)
    if expected_config and manifest["config"]["digest"] != expected_config:
        raise ValueError("Config digest is not the accepted repack image ID")
    if expected_plan:
        plan = json.loads(Path(expected_plan).read_text(encoding="utf-8-sig"))
        if (
            config.get("architecture") != plan["architecture"]
            or config.get("os") != plan["os"]
        ):
            raise ValueError("Platform mismatch")
        expected, actual = plan["expected_runtime_config"], config.get("config", {})
        if canonical_config(expected) != canonical_config(actual):
            raise ValueError("Runtime configuration mismatch")
    diff_ids = config["rootfs"]["diff_ids"]
    if config["rootfs"]["type"] != "layers" or len(diff_ids) != len(manifest["layers"]):
        raise ValueError("Layer/config diffID cardinality mismatch")
    history = config.get("history")
    if history is not None and sum(
        not x.get("empty_layer", False) for x in history
    ) != len(diff_ids):
        raise ValueError("Nonempty history count disagrees with layer count")

    def verify_one(pair):
        descriptor, diff = pair
        if descriptor["size"] > max_layer_bytes:
            raise ValueError("Layer exceeds the configured compressed-byte limit")
        with get_blob(descriptor) as source:
            return read_blob(source, descriptor, diff, inspect_tar=True)[1]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        layers = list(pool.map(verify_one, zip(manifest["layers"], diff_ids)))
    seen = set()
    for layer in layers:
        paths = layer.pop("_payload_paths")
        if len(paths) != len(set(paths)) or seen.intersection(paths):
            raise ValueError("Payload path is duplicated across flattened OCI layers")
        seen.update(paths)
        layer["payload_entries"] = len(paths)
    return {
        "status": "PASS_MANIFEST_CONFIG_ALL_BLOBS_FULL_READBACK",
        "manifest_digest": manifest_digest,
        "config_digest": manifest["config"]["digest"],
        "config": config_check,
        "layers": layers,
        "layer_count": len(layers),
        "unique_payload_entries": len(seen),
        "duplicate_payload_entries": 0,
        "compressed_bytes": sum(x["size"] for x in layers),
        "max_layer_bytes": max((x["size"] for x in layers), default=0),
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--layout", type=Path)
    group.add_argument("--registry-reference")
    parser.add_argument("--manifest-digest")
    parser.add_argument("--expected-config-digest")
    parser.add_argument("--expected-plan", type=Path)
    parser.add_argument("--username")
    parser.add_argument("--credential-stdin", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--max-layer-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    credential = sys.stdin.readline().rstrip("\r\n") if args.credential_stdin else None
    if args.layout:
        if credential or args.username:
            raise ValueError("Authentication is invalid for local verification")
        index = json.loads((args.layout / "index.json").read_bytes())
        if args.manifest_digest:
            descriptor = next(
                x for x in index["manifests"] if x["digest"] == args.manifest_digest
            )
        elif len(index["manifests"]) == 1:
            descriptor = index["manifests"][0]
        else:
            raise ValueError("Multiple OCI manifests; select an exact digest")
        check_descriptor(descriptor)
        manifest_digest = descriptor["digest"]

        def get_blob(item):
            check_descriptor(item)
            return (
                args.layout / "blobs/sha256" / item["digest"].split(":", 1)[1]
            ).open("rb")

        with get_blob(descriptor) as source:
            manifest_raw, _ = read_blob(source, descriptor)
    else:
        if not args.manifest_digest:
            raise ValueError(
                "Registry verification requires the accepted local manifest digest"
            )
        if not args.expected_config_digest:
            raise ValueError("Registry readback needs the accepted image config digest")
        registry = Registry(args.registry_reference, args.username, credential)
        manifest_digest = args.manifest_digest
        with registry.open(
            "manifests/" + quote(registry.reference, safe=":"), ",".join(MANIFEST_TYPES)
        ) as response:
            manifest_raw = response.read(64 * 1024**2 + 1)
            returned = response.headers.get("Docker-Content-Digest")
            if len(manifest_raw) > 64 * 1024**2 or (
                returned and returned != manifest_digest
            ):
                raise ValueError("Registry manifest size or digest header mismatch")

        def get_blob(item):
            check_descriptor(item)
            return registry.open("blobs/" + item["digest"])

    result = verify_graph(
        manifest_raw,
        manifest_digest,
        get_blob,
        args.expected_plan,
        args.expected_config_digest,
        args.workers,
        args.max_layer_bytes,
    )
    result["source"] = "registry GET" if args.registry_reference else "local OCI"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("config", "layers")}, indent=2
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never serialize HTTP exception URLs, signed redirects, headers or bodies.
        print(
            "Verification failed ("
            + type(error).__name__
            + "); no credentials or HTTP details emitted",
            file=sys.stderr,
        )
        sys.exit(1)
