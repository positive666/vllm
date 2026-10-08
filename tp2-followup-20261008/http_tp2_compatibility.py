"""Record fixed-token TP2 compatibility before freezing serving measurements."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from http_performance_client import (
    PROTOCOL,
    digest,
    exclusive_json,
    read_json,
    run_round,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "flashinfer"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-evidence", type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir / f"compat-{args.backend}.json"
    if target.exists():
        raise FileExistsError(target)
    fixture = read_json(args.fixture)
    if (
        len(fixture["input_tokens"]) != 512
        or digest(fixture["input_tokens"]) != fixture["input_tokens_sha256"]
    ):
        raise RuntimeError("Compatibility fixture hash/count mismatch")
    client = SimpleNamespace(
        base_url="http://127.0.0.1:8000",
        model="qwen-fp8",
        timeout=120,
        arm=f"{args.backend}-compat",
        output_dir=str(args.output_dir),
        active_requests=8,
    )
    report = {
        "backend": args.backend,
        "tensor_parallel_size": 2,
        "server_evidence": read_json(args.server_evidence),
        "protocol": PROTOCOL,
        "protocol_sha256": digest(PROTOCOL),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "http_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("http_performance_client.py").read_bytes()
        ).hexdigest(),
        "scope": (
            "Fixed token/stream compatibility only; not a model accuracy evaluation "
            "or an assertion of token-sequence equivalence"
        ),
        "rounds": [],
    }
    for concurrency in (1, 8):
        report["rounds"].append(
            run_round(client, fixture, concurrency, 0, "compatibility")
        )
        if report["rounds"][-1]["summary"]["failed"]:
            report["all_successful"] = False
            exclusive_json(target, report)
            raise RuntimeError("TP2 compatibility request failed")
    report["all_successful"] = True
    exclusive_json(target, report)


if __name__ == "__main__":
    main()
