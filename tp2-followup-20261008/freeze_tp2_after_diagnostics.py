"""Freeze TP2 measurements while preserving the primary budget-limited outcomes."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from http_performance_client import PROTOCOL, digest, exclusive_json, read_json

SOURCE_HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
DIAGNOSTIC_INDICES = [198, 206, 209, 228, 255, 285, 292, 318]


def diagnostic(root, backend, primary_server, primary_quality):
    server_path = root / "diagnostics" / f"serve-{backend}.json"
    client_path = root / "diagnostics" / f"{backend}.json"
    server = read_json(server_path)
    client = read_json(client_path)
    if (
        server["status"] != "completed"
        or server["backend"] != backend
        or server["source_head"] != SOURCE_HEAD
        or server["tensor_parallel_size"] != 2
        or server["gpu_blocks"] != 128
        or server["max_model_len"] != 4096
        or server["kv_capacity_tokens"] < 32768
        or server["diagnostic_sha256"]
        != hashlib.sha256(client_path.read_bytes()).hexdigest()
        or server["diagnostic_helper_sha256"] != client["diagnostic_client_sha256"]
        or client["arm"] != backend
        or client["selected_indices"] != DIAGNOSTIC_INDICES
        or client["primary_quality_sha256"] != primary_server["quality_sha256"]
        or client["full_question_fixture_sha256"]
        != primary_quality["question_fixture_sha256"]
        or client["harness_sha256"] != primary_server["quality_helper_sha256"]
        or client["protocol"]["max_tokens"] != 3500
        or client["protocol"]["temperature"] != 0
        or client["protocol"]["seed"] != 42
        or client["protocol"]["enable_thinking"] is not False
    ):
        raise RuntimeError(f"Diagnostic provenance/protocol differs: {backend}")
    for field in ("source_files", "gpu_uuids", "model_config_sha256", "runtime"):
        if server[field] != primary_server[field]:
            raise RuntimeError(f"Diagnostic changed {field}: {backend}")
    for field in (
        "source_head",
        "source_files",
        "gpu_uuids",
        "model_config_sha256",
        "tensor_parallel_size",
        "gpu_blocks",
        "max_model_len",
        "kv_capacity_tokens",
    ):
        if client["server_evidence"][field] != server[field]:
            raise RuntimeError(
                f"Diagnostic embedded evidence differs: {backend}/{field}"
            )
    command = server["command"]
    for flag, expected in (
        ("--tensor-parallel-size", "2"),
        ("--max-model-len", "4096"),
        ("--num-gpu-blocks-override", "128"),
        ("--dtype", "bfloat16"),
        ("--mamba-ssm-cache-dtype", "float32"),
        ("--seed", "42"),
    ):
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"Diagnostic CLI differs: {backend}/{flag}")
    normalized = list(command)
    normalized[normalized.index("--kernel-config") + 1] = "<backend>"
    if [row["concurrency"] for row in client["rounds"]] != [1, 8]:
        raise RuntimeError(f"Diagnostic C1/C8 matrix differs: {backend}")
    for row in client["rounds"]:
        if (
            row["repeat"] != 1
            or len(row["examples"]) != 8
            or row["summary"]["requests"] != 8
            or [example["index"] for example in row["examples"]] != DIAGNOSTIC_INDICES
            or any(example.get("error") for example in row["examples"])
            or any(
                not example["token_ids"]
                or example["usage"]["completion_tokens"] != len(example["token_ids"])
                for example in row["examples"]
            )
        ):
            raise RuntimeError(f"Incomplete diagnostic requests: {backend}")
    reference = {
        "server_file": str(server_path.relative_to(root)),
        "server_sha256": hashlib.sha256(server_path.read_bytes()).hexdigest(),
        "client_file": str(client_path.relative_to(root)),
        "client_sha256": hashlib.sha256(client_path.read_bytes()).hexdigest(),
        "round_summaries": [
            {"concurrency": row["concurrency"], "summary": row["summary"]}
            for row in client["rounds"]
        ],
    }
    return reference, server, client, normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("/results/tp2-followup-20261008")
    )
    args = parser.parse_args()
    root = args.results_root
    servers = {}
    clients = {}
    quality = {}
    references = {}
    for backend in ("triton", "flashinfer"):
        server_path = root / f"serve-{backend}-compat.json"
        client_path = root / "http" / f"compat-{backend}.json"
        quality_path = root / f"quality-{backend}.json"
        server = read_json(server_path)
        client = read_json(client_path)
        screen = read_json(quality_path)
        if (
            server["status"] != "completed"
            or server["source_head"] != SOURCE_HEAD
            or server["tensor_parallel_size"] != 2
            or not client["all_successful"]
            or client["protocol_sha256"] != digest(PROTOCOL)
            or server["client_result_sha256"]
            != hashlib.sha256(client_path.read_bytes()).hexdigest()
            or [row["concurrency"] for row in client["rounds"]] != [1, 8]
            or any(row["summary"]["successful"] != 8 for row in client["rounds"])
        ):
            raise RuntimeError(f"Invalid compatibility gate: {backend}")
        if (
            server["quality_sha256"]
            != hashlib.sha256(quality_path.read_bytes()).hexdigest()
            or screen["arm"] != backend
            or screen["summary"]["samples"] != 100
            or len(screen["examples"]) != 100
            or screen["summary"]["http_failures"] != 0
            or screen["protocol"]["max_tokens"] != 1750
            or screen["protocol"]["concurrency"] != 8
            or screen["harness_sha256"] != server["quality_helper_sha256"]
            or screen["server_evidence"]["source_head"] != SOURCE_HEAD
            or screen["server_evidence"]["tensor_parallel_size"] != 2
            or screen["server_evidence"]["source_files"] != server["source_files"]
            or screen["summary"]["correct"]
            != sum(row["predicted"] == row["gold"] for row in screen["examples"])
        ):
            raise RuntimeError(f"Incomplete quality gate: {backend}")
        servers[backend] = server
        clients[backend] = client
        quality[backend] = screen
        references[backend] = {
            "server_file": server_path.name,
            "server_sha256": hashlib.sha256(server_path.read_bytes()).hexdigest(),
            "client_file": str(client_path.relative_to(root)),
            "client_sha256": hashlib.sha256(client_path.read_bytes()).hexdigest(),
            "quality_file": quality_path.name,
            "quality_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        }
    first, second = servers["triton"], servers["flashinfer"]
    for field in (
        "gpu_blocks",
        "kv_capacity_tokens",
        "gpu_uuids",
        "source_files",
        "model_config_sha256",
        "state_dtype",
    ):
        if first[field] != second[field]:
            raise RuntimeError(f"Compatibility arms differ: {field}")
    if clients["triton"]["fixture_sha256"] != clients["flashinfer"]["fixture_sha256"]:
        raise RuntimeError("Compatibility fixtures differ")
    for field in ("fixture", "question_fixture_sha256", "harness_sha256", "protocol"):
        if quality["triton"][field] != quality["flashinfer"][field]:
            raise RuntimeError(f"Quality screens differ in {field}")
    diagnostics = {}
    diagnostic_records = {}
    diagnostic_clients = {}
    diagnostic_commands = {}
    for backend in ("triton", "flashinfer"):
        (
            diagnostics[backend],
            diagnostic_records[backend],
            diagnostic_clients[backend],
            diagnostic_commands[backend],
        ) = diagnostic(root, backend, servers[backend], quality[backend])
    if diagnostic_commands["triton"] != diagnostic_commands["flashinfer"]:
        raise RuntimeError("Diagnostic commands differ beyond backend")
    if (
        diagnostic_records["triton"]["kv_capacity_tokens"]
        != diagnostic_records["flashinfer"]["kv_capacity_tokens"]
    ):
        raise RuntimeError("Diagnostic arms have different actual KV capacity")
    for field in ("protocol", "selected_question_fixture_sha256", "selected_indices"):
        if (
            diagnostic_clients["triton"][field]
            != diagnostic_clients["flashinfer"][field]
        ):
            raise RuntimeError(f"Diagnostic fixture/protocol differs: {field}")
    comparisons = []
    for index, concurrency in enumerate((1, 8)):
        a = clients["triton"]["rounds"][index]["records"]
        b = clients["flashinfer"]["rounds"][index]["records"]
        if len(a) != 8 or len(b) != 8:
            raise RuntimeError("Compatibility count differs")
        comparisons.append(
            {
                "concurrency": concurrency,
                "paired_requests": 8,
                "identical_token_sequences": sum(
                    x["output_token_ids"] == y["output_token_ids"] for x, y in zip(a, b)
                ),
                "identical_token_positions": sum(
                    sum(
                        xi == yi
                        for xi, yi in zip(x["output_token_ids"], y["output_token_ids"])
                    )
                    for x, y in zip(a, b)
                ),
                "paired_token_positions": 8 * 128,
                "scope": (
                    "Descriptive compatibility comparison; not accuracy equivalence"
                ),
            }
        )
    protocol = {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_head": SOURCE_HEAD,
        "tensor_parallel_size": 2,
        "gpu_blocks": first["gpu_blocks"],
        "expected_kv_capacity_tokens": first["kv_capacity_tokens"],
        "gpu_uuids": first["gpu_uuids"],
        "per_rank_gdn_shape": {"H": 8, "HV": 24, "K": 128, "V": 128},
        "compatibility_evidence": references,
        "compatibility_output_comparison": comparisons,
        "quality_screen": {
            "summaries": {
                backend: screen["summary"] for backend, screen in quality.items()
            },
            "question_fixture_sha256": quality["triton"]["question_fixture_sha256"],
            "dataset_sha256": quality["triton"]["fixture"]["dataset_sha256"],
            "protocol": quality["triton"]["protocol"],
            "scope": (
                "Matched 100-item GSM8K screen, manually evaluated before freezing; "
                "not an accuracy equivalence proof"
            ),
        },
        "diagnostic_evidence": diagnostics,
        "diagnostic_policy": (
            "Primary 100-item scores/truncations remain unchanged. Matched extra "
            "C1/C8 diagnostics change output budget, context and KV capacity "
            "together; wrong/truncated answers are recorded, not erased. "
            "These records do not assert single-variable causality or equivalence."
        ),
        "freeze_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "http_protocol": PROTOCOL,
        "http_protocol_sha256": digest(PROTOCOL),
        "fixture_sha256": clients["triton"]["fixture_sha256"],
        "http_client_sha256": hashlib.sha256(
            Path("/artifacts/http_performance_client.py").read_bytes()
        ).hexdigest(),
        "serving": {
            "order": ["triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2"],
            "input_tokens": 512,
            "output_tokens": 128,
            "requests_per_round": 16,
            "warmup_per_concurrency": 8,
            "rounds": 3,
            "concurrency": [1, 8],
            "measured_total": 384,
            "warmup_total": 64,
            "independent_launches_per_backend": 2,
            "comparison": (
                "Descriptive per-launch medians/ranges and paired effects; "
                "no confidence bound or significance claim"
            ),
        },
        "source_code_changed": False,
    }
    exclusive_json(root / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "gpu_blocks": protocol["gpu_blocks"],
                "kv_capacity_tokens": protocol["expected_kv_capacity_tokens"],
                "gpu_uuids": protocol["gpu_uuids"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
