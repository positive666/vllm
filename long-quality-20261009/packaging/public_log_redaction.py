"""Redact observed private network addresses without altering frozen code."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections import Counter

V4 = re.compile(rb"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
V6 = re.compile(
    rb"(?<![A-Za-z0-9_:])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f:.]*"
    rb"(?:%[A-Za-z0-9_.-]+)?(?![A-Za-z0-9_:])"
)
V4_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "169.254.0.0/16",
    )
)
V6_NETWORKS = tuple(ipaddress.ip_network(value) for value in ("fc00::/7", "fe80::/10"))
IMMUTABLE_SUFFIXES = (".py", ".sh", ".ps1", ".pt")


def private_address(raw, include_loopback=True):
    try:
        address = ipaddress.ip_address(raw.decode("ascii").split("%", 1)[0])
    except (UnicodeDecodeError, ValueError):
        return False
    if include_loopback and address.is_loopback:
        return True
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    networks = V4_NETWORKS if address.version == 4 else V6_NETWORKS
    return any(address in network for network in networks)


def address_counts(data, include_loopback=True):
    return {
        "ipv4": sum(
            private_address(match[0], include_loopback) for match in V4.finditer(data)
        ),
        "ipv6": sum(
            private_address(match[0], include_loopback) for match in V6.finditer(data)
        ),
    }


def redact(data, include_loopback=True):
    counts = Counter()

    def replace(match, version):
        if not private_address(match[0], include_loopback):
            return match[0]
        counts[version] += 1
        return ("<REDACTED_PRIVATE_" + version.upper() + ">").encode()

    data = V6.sub(lambda match: replace(match, "ipv6"), data)
    data = V4.sub(lambda match: replace(match, "ipv4"), data)
    return data, dict(counts)


def sanitize_payload(payload):
    """Keep code/tensors exact; map each redacted observation to original SHA."""
    sanitized = {}
    mapping = []
    for name, raw in sorted(payload.items()):
        if name.endswith(IMMUTABLE_SUFFIXES):
            if not name.endswith(".pt") and any(
                address_counts(raw, include_loopback=False).values()
            ):
                raise ValueError("Private address in immutable source: " + name)
            sanitized[name] = raw
            continue
        include_loopback = not name.endswith("/reference.json")
        public, counts = redact(raw, include_loopback=include_loopback)
        sanitized[name] = public
        if public != raw:
            mapping.append(
                {
                    "path": name,
                    "original_sha256": hashlib.sha256(raw).hexdigest(),
                    "original_bytes": len(raw),
                    "public_redacted_sha256": hashlib.sha256(public).hexdigest(),
                    "public_redacted_bytes": len(public),
                    "redaction_counts": counts,
                    "scope": (
                        "Private/loopback IPv4 and private/link-local/loopback "
                        "IPv6 observations only; no original address values "
                        "published. Generic reference localhost defaults "
                        "are preserved."
                    ),
                }
            )
        if any(address_counts(public, include_loopback=include_loopback).values()):
            raise ValueError("Private observed address remains: " + name)
    return sanitized, mapping
