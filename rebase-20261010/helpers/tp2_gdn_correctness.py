"""Check frozen production GDN wrappers on two NCCL ranks at TP2 head shapes.

Run with the authorized container's uv-managed interpreter, for example:
uv run --offline --no-project /cache/gdn-runtime/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 \
    /artifacts/tp2_gdn_correctness.py \
    --harness /artifacts/benchmark_current_gdn.py \
    --source-manifest /artifacts/quality-followup-source.json \
    --output-dir /results/tp2-followup-20261008/correctness

Each rank compares real Triton and FI production wrappers against a one-step
reference and over 128 changing-input CUDA graph replays. This validates
distributed execution of rank-local TP2 shapes; checkpoint sharding and the
full model's TP communication belong to the separate real serving validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

SOURCE_HEAD = "abf17c7c071b1b6bdd81ca9756894998e5e8b32d"
GLOBAL_H, GLOBAL_HV, K, V, TP_SIZE = 16, 48, 128, 128, 2
CASES = (
    (1, "int32", False),
    (8, "int32", False),
    (8, "int32", True),
    (1, "int64", False),
    (8, "int64", False),
    (8, "int64", True),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_harness(path):
    spec = importlib.util.spec_from_file_location("frozen_gdn_benchmark", path)
    require(spec is not None and spec.loader is not None, "Load frozen harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SOURCE_HEAD == SOURCE_HEAD, "Harness frozen source head")
    return module


def validate_source(module, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["source_head"] == SOURCE_HEAD, "Manifest frozen source head")
    production_file = Path(inspect.getfile(module.GDNDecode)).resolve()
    binding = module.source_binding(production_file)
    root = Path(binding["root"])
    observed = {relative: sha(root / relative) for relative in manifest["files"]}
    expected = {
        relative: item["sha256"] for relative, item in manifest["files"].items()
    }
    require(observed == expected, "All frozen source-file hashes")
    return {
        "source_head": SOURCE_HEAD,
        "source_root": str(root),
        "source_files": observed,
        "source_manifest_sha256": sha(manifest_path),
    }


def run_rank(args, rank, local_rank, module, record, rank_path):
    local_h, local_hv = GLOBAL_H // TP_SIZE, GLOBAL_HV // TP_SIZE
    require((local_h, local_hv) == (8, 24), "TP2 local head shape")
    record.update(validate_source(module, args.source_manifest))
    versions = {name: module.package_version(name) for name in module.RUNTIME_VERSIONS}
    require(versions == module.RUNTIME_VERSIONS, "Pinned runtime versions")
    properties = torch.cuda.get_device_properties(local_rank)
    record.update(
        {
            "runtime_versions": versions,
            "device": {
                "logical_index": local_rank,
                "name": properties.name,
                "uuid": str(properties.uuid),
                "capability": [properties.major, properties.minor],
            },
            "global_shape": {"H": GLOBAL_H, "HV": GLOBAL_HV, "K": K, "V": V},
            "local_shape": {"H": local_h, "HV": local_hv, "K": K, "V": V},
            "logical_head_shard": {
                "key_head_range": [rank * local_h, (rank + 1) * local_h],
                "value_head_range": [rank * local_hv, (rank + 1) * local_hv],
            },
            "tolerances": module.TOLERANCES,
            "wrapper_boundary": "GDNDecode raw GPU output/state plus gated RMSNorm",
            "state_dtype": "float32",
            "input_dtype": "bfloat16",
            "steps_per_case": 128,
            "parameter_contract": "Grad-enabled Parameters; FI adapter detaches biases",
            "harness_sha256": sha(args.harness),
        }
    )
    write_json(rank_path, record)
    # Exercise an actual NCCL collective before concurrent rank-local kernels.
    marker = torch.tensor([rank + 1], device=f"cuda:{local_rank}", dtype=torch.int32)
    dist.all_reduce(marker, op=dist.ReduceOp.SUM)
    require(marker.item() == 3, "Two-rank NCCL collective result")
    record["nccl_collective_sum"] = 3
    with torch.inference_mode():
        for number, (batch, index_name, padding) in enumerate(CASES):
            seed = args.seed + rank * 1000 + number
            label = f"b{batch}-{index_name}-padding{int(padding)}"
            item = {
                "case": label,
                "seed": seed,
                "batch": batch,
                "index_dtype": index_name,
                "padding": padding,
                "status": "running",
            }
            record["cases"].append(item)
            write_json(rank_path, record)
            case = module.Case(
                batch,
                torch.float32,
                getattr(torch, index_name),
                seed,
                local_h,
                local_hv,
                padding,
            )
            require(
                case.alog.numel() == case.bias.numel() == local_hv,
                "Local per-head bias lengths",
            )
            require(case.qkv.stride(0) > case.qkv.shape[-1], "Packed QKVZ stride")
            item.update(
                {
                    "local_A_log_shape": list(case.alog.shape),
                    "local_dt_bias_shape": list(case.bias.shape),
                    "raw_output_shape": list(case.raw["triton"].shape),
                    "state_shape": list(case.initial.shape),
                    "state_strides": list(case.initial.stride()),
                    "qkv_strides": list(case.qkv.stride()),
                    "gate_pointer_offset_bytes": (
                        case.a.data_ptr() - case.ba.data_ptr()
                    ),
                    "a_pointer_alignment_16_bytes": case.a.data_ptr() % 16 == 0,
                    "initial_index_map": case.indices.cpu().tolist(),
                    "rank_input_sample": case.packed[:1, :8].float().cpu().tolist(),
                }
            )
            item["correctness"] = case.correctness(128)
            torch.cuda.synchronize()
            item["status"] = "passed"
            write_json(rank_path, record)
            print(
                json.dumps({"rank": rank, "case": label, "status": "passed"}),
                flush=True,
            )
            del case
            gc.collect()
    record["status"] = "passed"
    record["passed_cases"] = len(record["cases"])
    write_json(rank_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    require(world_size == TP_SIZE, "Exactly two torchrun ranks are required")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() >= TP_SIZE,
        "Two bound CUDA GPUs are required",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank-{rank}.json"
    require(not rank_path.exists(), "Refusing to overwrite rank evidence")
    torch.cuda.set_device(local_rank)
    record = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "status": "starting",
        "cases": [],
        "script_sha256": sha(Path(__file__)),
        "scope": "Two NCCL ranks execute TP2-shaped local GDN correctness",
    }
    write_json(rank_path, record)
    try:
        module = load_harness(args.harness)
        dist.init_process_group(
            "nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
            timeout=timedelta(minutes=10),
        )
        require(dist.get_backend() == "nccl", "NCCL process group")
        run_rank(args, rank, local_rank, module, record, rank_path)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=12)
        write_json(rank_path, record)
    try:
        if dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, record)
            passed = all(row["status"] == "passed" for row in gathered)
            if rank == 0:
                output = args.output_dir / "distributed-correctness.json"
                require(not output.exists(), "Refusing to overwrite global evidence")
                result = {
                    "status": "passed" if passed else "failed",
                    "source_head": SOURCE_HEAD,
                    "world_size": world_size,
                    "distributed_backend": "nccl",
                    "global_shape": {"H": GLOBAL_H, "HV": GLOBAL_HV, "K": K, "V": V},
                    "per_rank_shape": {"H": 8, "HV": 24, "K": K, "V": V},
                    "passed_rank_cases": sum(
                        row.get("passed_cases", 0) for row in gathered
                    ),
                    "ranks": gathered,
                    "scope": "Distributed local-wrapper correctness; no timing claim",
                    "limitations": [
                        "Does not exercise checkpoint weight loading or full-model TP",
                        "FP32 state and the frozen pinned runtime only",
                        "TP2 local HV24 does not enter the HV4/HV12 clone branch",
                    ],
                }
                write_json(output, result)
                print(
                    json.dumps(
                        {
                            "status": result["status"],
                            "passed_rank_cases": result["passed_rank_cases"],
                        }
                    ),
                    flush=True,
                )
            require(passed, "At least one rank failed; inspect preserved rank JSON")
        else:
            raise RuntimeError("Distributed setup failed; inspect rank JSON")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
