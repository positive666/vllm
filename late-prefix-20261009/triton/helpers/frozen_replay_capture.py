"""Replay captured real GDN decode inputs on both production backends.

This is a diagnostic artifact, not a benchmark or model-quality equivalence
test. Each capture is replayed independently, with identical inputs and state;
rank-local captures do not reproduce multi-GPU scheduling or communication.

Run only with an authorized uv-managed CUDA environment, for example:
uv run --offline --no-project /cache/gdn-runtime/bin/python replay_capture.py \
    --captures-dir /results/decode-shadow/captures \
    --source-manifest /artifacts/quality-followup-source.json \
    --output /results/decode-shadow/replay.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

SOURCE_HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
EXPECTED_VERSIONS = {
    "torch": "2.13.0+cu129",
    "triton": "3.7.1",
    "flashinfer-python": "0.7.0.post1",
    "nvidia-cutlass-dsl": "4.8.0",
}
TOLERANCES = {"atol": 0.01, "rtol": 0.01, "relative_l2_strictly_below": 0.01}
SOURCE_MATH_FILES = {
    "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py": (
        "fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755"
    ),
    "vllm/third_party/flash_linear_attention/ops/op.py": (
        "456da84fd12411cf53c96a90dd4f78f49afb454c3d51d2c524ea8e7106fe8ae5"
    ),
}
EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0
GUARD_VALUE = 0.9375


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def summary(value):
    value = value.detach().cpu().double().flatten()
    finite = torch.isfinite(value)
    filtered = value[finite]
    return {
        "numel": value.numel(),
        "finite_count": int(finite.sum()),
        "nan_count": int(torch.isnan(value).sum()),
        "positive_inf_count": int(torch.isposinf(value).sum()),
        "negative_inf_count": int(torch.isneginf(value).sum()),
        "all_finite": bool(finite.all()),
        "min": float(filtered.min()) if filtered.numel() else None,
        "max": float(filtered.max()) if filtered.numel() else None,
        "absmax": float(filtered.abs().max()) if filtered.numel() else None,
        "rms": float(filtered.square().mean().sqrt()) if filtered.numel() else None,
    }


def compare(observed, reference):
    require(observed.shape == reference.shape, "Comparison shapes must match")
    original_equal = torch.equal(observed.cpu(), reference.cpu())
    lhs = observed.detach().cpu().double()
    rhs = reference.detach().cpu().double()
    finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all())
    if not lhs.numel():
        return {
            "numel": 0,
            "exact_equal": True,
            "all_finite": True,
            "absmax": 0.0,
            "rms_error": 0.0,
            "reference_rms": 0.0,
            "relative_l2": None,
            "allclose": True,
            "relative_l2_below_threshold": True,
            "within_tolerance": True,
        }
    if not finite:
        return {
            "numel": lhs.numel(),
            "exact_equal": original_equal,
            "all_finite": False,
            "absmax": None,
            "rms_error": None,
            "reference_rms": None,
            "relative_l2": None,
            "allclose": False,
            "relative_l2_below_threshold": False,
            "within_tolerance": False,
        }
    delta = lhs - rhs
    rms = float(delta.square().mean().sqrt())
    reference_rms = float(rhs.square().mean().sqrt())
    # A global norm ratio, never per-element division near zero. Exact-zero
    # references use an explicit absolute rule rather than an invented floor.
    relative = rms / reference_rms if reference_rms else None
    relative_ok = (
        relative < TOLERANCES["relative_l2_strictly_below"]
        if relative is not None
        else rms == 0.0
    )
    close = bool(
        torch.allclose(lhs, rhs, atol=TOLERANCES["atol"], rtol=TOLERANCES["rtol"])
    )
    return {
        "numel": lhs.numel(),
        "exact_equal": original_equal,
        "all_finite": True,
        "absmax": float(delta.abs().max()),
        "rms_error": rms,
        "reference_rms": reference_rms,
        "relative_l2": relative,
        "allclose": close,
        "relative_l2_below_threshold": relative_ok,
        "within_tolerance": close and relative_ok,
    }


def layout_record(metadata, name):
    records = metadata["original_shapes_strides"]
    # Capture records use the production forward argument names.
    argument_name = {
        "production_output": "out",
        "indices_original": "ssm_state_indices",
    }.get(name, name)
    require(argument_name in records, f"Missing original layout for {name}")
    record = records[argument_name]
    require(isinstance(record, dict), f"Invalid layout metadata for {name}")
    return record


def preserved_tensor(value, layout, device, *, dtype=None):
    """Restore per-tensor strides/alignment; absolute pointers and aliases differ."""
    dtype = dtype or value.dtype
    shape = tuple(value.shape)
    strides = tuple(int(item) for item in layout["stride"])
    require(len(shape) == len(strides), "Original stride dimensionality")
    require(all(item > 0 for item in strides), "Only positive strides supported")
    size = 1
    for stride, length in sorted(zip(strides, shape)):
        if length > 1:
            require(stride >= size, "Capture layout must not overlap")
            size = stride * (length - 1) + size
    span = 1 + sum((length - 1) * stride for length, stride in zip(shape, strides))
    require(0 < span <= 128 * 1024 * 1024, "Bounded diagnostic allocation")
    itemsize = torch.empty((), dtype=dtype).element_size()
    padding = 256 // itemsize + 1
    storage = torch.full((span + padding,), GUARD_VALUE, dtype=dtype, device=device)
    target = int(layout["data_ptr_mod_256"])
    require(target % itemsize == 0, "Pointer alignment compatible with dtype")
    offset_bytes = (target - storage.data_ptr() % 256) % 256
    offset = offset_bytes // itemsize
    tensor = storage.as_strided(shape, strides, offset)
    tensor.copy_(value.to(device=device, dtype=dtype))
    logical = torch.zeros(storage.shape, dtype=torch.bool, device=device)
    logical.as_strided(shape, strides, offset).fill_(True)
    require(tuple(tensor.stride()) == strides, "Restored original strides")
    require(tensor.data_ptr() % 256 == target, "Restored original pointer modulo")
    return tensor, storage, logical


def preserved_bias(value, layout, device):
    # Bias ablations change dtype; preserve alignment rather than byte offsets.
    replacement = dict(layout)
    replacement["stride"] = [1]
    return preserved_tensor(value, replacement, device)[0]


def fp64_reference(tensors, metadata, bias):
    """Exact input values in FP64, with the mathematical kernel conventions.

    fused_recurrent.py packed decode: q/k norm epsilon=1e-6, scale=K^-0.5,
    softplus threshold=20; state is [pool, HV, V, K]. FP64 deliberately does
    not reproduce Triton/CuTe FP32 reduction orders or approximate intrinsics.
    """
    h = int(metadata["num_heads_qk"])
    hv = int(metadata["num_heads_v"])
    kdim = int(metadata["head_qk_dim"])
    vdim = int(metadata["head_v_dim"])
    indices = tensors["indices_compact"].long()
    active = indices > 0
    slots = indices[active]
    count = slots.numel()
    state = tensors["initial_state"].double().clone()
    output = torch.zeros((indices.numel(), 1, hv, vdim), dtype=torch.float64)
    q, k, v = (
        tensors["mixed_qkv"][active]
        .double()
        .split((h * kdim, h * kdim, hv * vdim), dim=-1)
    )
    q = q.reshape(count, h, kdim).repeat_interleave(hv // h, dim=1)
    k = k.reshape(count, h, kdim).repeat_interleave(hv // h, dim=1)
    if metadata["use_qk_l2norm_in_kernel"]:
        q = q / (q.square().sum(-1, keepdim=True) + EPS).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + EPS).sqrt()
    q = q * kdim**-0.5
    x = tensors["a"][active].double() + bias.double()
    softplus = F.softplus(x, beta=1.0, threshold=SOFTPLUS_THRESHOLD)
    g = -tensors["A_log"].double().exp() * softplus
    beta = tensors["b"][active].double().sigmoid()
    active_state = state[slots] * g.exp()[..., None, None]
    delta = v.reshape(count, hv, vdim) - (active_state * k[..., None, :]).sum(-1)
    active_state += (delta * beta[..., None])[..., None] * k[..., None, :]
    state[slots] = active_state
    output[active, 0] = (active_state * q[..., None, :]).sum(-1)
    return (
        output,
        state,
        {
            "a_plus_bias": summary(x),
            "softplus": summary(softplus),
            "log_decay_g": summary(g),
            "decay": summary(g.exp()),
            "beta": summary(beta),
            "q_normalized_scaled": summary(q),
            "k_normalized": summary(k),
            "reference_output": summary(output[active]),
            "reference_active_state": summary(state[slots]),
        },
    )


def load_capture(path):
    raw = path.read_bytes()
    sidecar_path = path.with_suffix(".json")
    require(sidecar_path.is_file(), "Capture hash sidecar required")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    require(sidecar["sha256"] == sha_bytes(raw), "Frozen capture sidecar hash")
    require(sidecar["bytes"] == len(raw), "Frozen capture sidecar size")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    require(set(payload) == {"metadata", "tensors"}, "Frozen capture top-level schema")
    meta, tensors = payload["metadata"], payload["tensors"]
    require(sidecar["metadata"] == meta, "Capture metadata matches hash sidecar")
    expected = {
        "mixed_qkv",
        "a",
        "b",
        "A_log",
        "dt_bias",
        "initial_state",
        "indices_original",
        "indices_compact",
        "production_output",
        "production_post_state",
    }
    require(expected <= set(tensors), "Required capture tensors")
    require(meta["source_head"] == SOURCE_HEAD, "Captured production source head")
    require(meta["original_backend"] in ("triton", "flashinfer"), "Captured backend")
    require(meta["use_qk_l2norm_in_kernel"] is True, "Production norm convention")
    h, hv = int(meta["num_heads_qk"]), int(meta["num_heads_v"])
    kdim, vdim = int(meta["head_qk_dim"]), int(meta["head_v_dim"])
    require(h > 0 and hv % h == 0 and (kdim, vdim) == (128, 128), "Supported shape")
    original, compact = tensors["indices_original"], tensors["indices_compact"]
    require(original.ndim == compact.ndim == 1, "Single-token decode indices")
    require(original.dtype in (torch.int32, torch.int64), "Index dtype")
    require(original.dtype == compact.dtype, "Capture index dtype preserved")
    require(original.shape == compact.shape, "Index row count")
    active = original > 0
    require(torch.equal(active, compact > 0), "Active rows preserved by remap")
    originals = original[active].tolist()
    require(len(originals) == len(set(originals)), "No duplicate active original slots")
    require(len(compact[active].unique()) == int(active.sum()), "Unique compact slots")
    slots = compact[active].long()
    require(slots.numel() > 0, "At least one active row")
    require(int(slots.max()) < tensors["initial_state"].shape[0], "Compact slot bounds")
    require(
        set(slots.tolist()) == set(range(1, slots.numel() + 1)), "Dense compact map"
    )
    mapping = meta["compact_to_original"]
    require(len(mapping) == tensors["initial_state"].shape[0], "Compact map length")
    require(mapping[0] == 0, "Reserved null slot mapping")
    require(
        all(
            mapping[int(new)] == int(old)
            for old, new in zip(original[active], compact[active])
        ),
        "Original-to-compact map values",
    )
    padding = original <= 0
    expected_padding = -1 if meta["original_backend"] == "flashinfer" else 0
    require(
        bool((original[padding] == expected_padding).all()), "Backend padding sentinel"
    )
    require(
        bool((compact[padding] == expected_padding).all()), "Compact padding sentinel"
    )
    require(torch.equal(tensors["padding_rows"], padding), "Captured padding row mask")
    require(meta["padding_rows"] == padding.tolist(), "Padding row metadata")
    batch = original.numel()
    require(
        tensors["mixed_qkv"].shape == (batch, 2 * h * kdim + hv * vdim), "QKV shape"
    )
    require(tensors["a"].shape == tensors["b"].shape == (batch, hv), "Gate shapes")
    require(
        tensors["A_log"].shape == tensors["dt_bias"].shape == (hv,),
        "Head parameter shape",
    )
    require(tensors["initial_state"].shape[1:] == (hv, vdim, kdim), "VK state layout")
    require(
        tensors["initial_state"].shape[0] >= slots.numel() + 2,
        "Slot0 and unused slot captured",
    )
    require(tensors["initial_state"].dtype == torch.float32, "FP32 production state")
    require(
        tensors["initial_state"].shape == tensors["production_post_state"].shape,
        "Captured poststate shape",
    )
    require(
        tensors["production_output"].shape == (batch, 1, hv, vdim),
        "Production output shape",
    )
    require(
        all(tensors[name].dtype == torch.bfloat16 for name in ("mixed_qkv", "a", "b")),
        "Exact BF16 inputs",
    )
    require(
        all(tensors[name].device.type == "cpu" for name in expected),
        "Capture stored on CPU",
    )
    require(
        all(summary(tensors[name])["all_finite"] for name in expected),
        "Finite captured tensors",
    )
    return meta, tensors, sha_bytes(raw)


def replay_backend(cls, metadata, tensors, bias, backend, device):
    inputs = {}
    layout_reports = {}
    for name in ("mixed_qkv", "a", "b", "A_log", "initial_state"):
        inputs[name], storage, logical = preserved_tensor(
            tensors[name], layout_record(metadata, name), device
        )
        if name == "initial_state":
            state_storage, state_logical = storage, logical
        if name == "A_log" and layout_record(metadata, name)["requires_grad"]:
            inputs[name].requires_grad_(True)
        layout_reports[name] = {
            "shape": list(inputs[name].shape),
            "stride": list(inputs[name].stride()),
            "data_ptr_mod_256": inputs[name].data_ptr() % 256,
            "source": layout_record(metadata, name),
        }
    inputs["dt_bias"] = preserved_bias(bias, layout_record(metadata, "dt_bias"), device)
    if layout_record(metadata, "dt_bias")["requires_grad"]:
        inputs["dt_bias"].requires_grad_(True)
    indices = tensors["indices_compact"].clone()
    indices[indices <= 0] = -1 if backend == "flashinfer" else 0
    index_layout = dict(layout_record(metadata, "indices_original"))
    inputs["ssm_state_indices"] = preserved_tensor(indices, index_layout, device)[0]
    # The production Triton wrapper requires a contiguous output tensor.
    out_layout = layout_record(metadata, "production_output")
    output, _, _ = preserved_tensor(
        torch.zeros_like(tensors["production_output"]), out_layout, device
    )
    require(output.is_contiguous(), "Original production output contiguous")
    op = cls.__new__(cls)
    object.__setattr__(op, "num_k_heads", int(metadata["num_heads_qk"]))
    object.__setattr__(op, "num_v_heads", int(metadata["num_heads_v"]))
    object.__setattr__(op, "head_k_dim", int(metadata["head_qk_dim"]))
    object.__setattr__(op, "head_v_dim", int(metadata["head_v_dim"]))
    if backend == "flashinfer":
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        object.__setattr__(
            op, "_flashinfer_decode", gated_delta_rule_decode_pretranspose
        )
    forward = cls.forward_native if backend == "triton" else cls.forward_cuda
    with torch.inference_mode():
        forward(op, out=output, **inputs)
    torch.cuda.synchronize(device)
    state = inputs["initial_state"].cpu()
    result_output = output.cpu()
    active = tensors["indices_compact"] > 0
    active_slots = tensors["indices_compact"][active].long()
    inactive = torch.ones(state.shape[0], dtype=torch.bool)
    inactive[active_slots] = False
    inactive_exact = torch.equal(state[inactive], tensors["initial_state"][inactive])
    guard_exact = bool((state_storage[~state_logical] == GUARD_VALUE).all())
    padding_count = int((~active).sum())
    padding_zero = torch.equal(
        result_output[~active], torch.zeros_like(result_output[~active])
    )
    return (
        result_output,
        state,
        {
            "backend": backend,
            "bias_dtype": str(bias.dtype),
            "input_layouts": layout_reports,
            "sampled_inactive_slots": torch.nonzero(inactive).flatten().tolist(),
            "sampled_inactive_state_exact": inactive_exact,
            "slot0_exact": torch.equal(state[0], tensors["initial_state"][0]),
            "synthetic_storage_guard_exact": guard_exact,
            "output_range": summary(result_output[active]),
            "active_state_range": summary(state[active_slots]),
            "padding_output_range": summary(result_output[~active]),
            "padding_row_count": padding_count,
            "shadow_padding_zero_exact": padding_zero,
            "padding_output_coverage": (
                "tested from zero-initialized shadow output"
                if padding_count
                else "not covered; no padding rows in capture"
            ),
            "invariants_ok": inactive_exact and guard_exact and padding_zero,
        },
    )


def replay_capture(cls, path, device, bias_mode):
    metadata, tensors, capture_sha = load_capture(path)
    active = tensors["indices_compact"] > 0
    slots = tensors["indices_compact"][active].long()
    bias = tensors["dt_bias"]
    rounded = bias.to(torch.bfloat16).to(torch.float32)
    record = {
        "capture_path": str(path),
        "capture_sha256": capture_sha,
        "capture_sidecar_sha256": sha_file(path.with_suffix(".json")),
        "metadata": metadata,
        "input_ranges": {name: summary(value) for name, value in tensors.items()},
        "bias_ablation": {
            "actual_dtype": str(bias.dtype),
            "actual_vs_bf16_rounded_fp32": compare(bias, rounded),
            "capture_has_fp32_bias_values": bias.dtype == torch.float32,
            "limitation": (
                "FP32 capture preserves actual adapter values, not proof of "
                "checkpoint precision. BF16 captures cannot recover unobserved "
                "pre-rounding FP32 values."
            ),
        },
        "arms": {},
    }
    if bias_mode == "bf16-dtype-only":
        variants = {"bf16_bias_dtype": bias.to(torch.bfloat16)}
    else:
        variants = {"captured_actual_bias": bias}
        if bias_mode == "main":
            require(
                bias.dtype == torch.float32,
                "Main two-arm mode requires actual FP32 bias; use actual-only "
                "for BF16-origin captures to avoid FI dtype cache collisions",
            )
            variants["bf16_rounded_bias_fp32"] = rounded
    outputs, states = {}, {}
    for variant, variant_bias in variants.items():
        ref_output, ref_state, intermediates = fp64_reference(
            tensors, metadata, variant_bias
        )
        rounded_ref_output = ref_output.to(tensors["production_output"].dtype)
        rounded_ref_state = ref_state.to(tensors["initial_state"].dtype)
        arm = {"reference_intermediates": intermediates, "backends": {}}
        for backend in ("triton", "flashinfer"):
            output, state, backend_record = replay_backend(
                cls, metadata, tensors, variant_bias, backend, device
            )
            backend_record["vs_fp64_math_output"] = compare(
                output[active], ref_output[active]
            )
            backend_record["vs_fp64_math_state"] = compare(
                state[slots], ref_state[slots]
            )
            backend_record["vs_fp64_rounded_output"] = compare(
                output[active], rounded_ref_output[active]
            )
            backend_record["vs_fp64_rounded_state"] = compare(
                state[slots], rounded_ref_state[slots]
            )
            backend_record["reference_tolerance_ok"] = (
                backend_record["vs_fp64_rounded_output"]["within_tolerance"]
                and backend_record["vs_fp64_rounded_state"]["within_tolerance"]
            )
            arm["backends"][backend] = backend_record
            outputs[variant, backend], states[variant, backend] = output, state
        arm["triton_vs_flashinfer_output"] = compare(
            outputs[variant, "flashinfer"][active], outputs[variant, "triton"][active]
        )
        arm["triton_vs_flashinfer_state"] = compare(
            states[variant, "flashinfer"][slots], states[variant, "triton"][slots]
        )
        record["arms"][variant] = arm
    original_backend = metadata["original_backend"]
    if bias_mode == "bf16-dtype-only":
        output = compare(
            outputs["bf16_bias_dtype", original_backend][active],
            tensors["production_output"][active],
        )
        state = compare(
            states["bf16_bias_dtype", original_backend],
            tensors["production_post_state"],
        )
        record["bf16_dtype_vs_captured_production"] = {
            "captured_backend": original_backend,
            "captured_bias_dtype": str(bias.dtype),
            "ablation_bias_dtype": "torch.bfloat16",
            "same_bias_values": torch.equal(bias.float(), rounded),
            "active_output": output,
            "all_captured_page_poststate": state,
            "scope": "Changed bias dtype; this comparison is not original "
            "production fidelity validation.",
        }
        record["production_reproduction"] = {
            "attempted": False,
            "reason": "BF16 dtype ablation omits actual-dtype FI calls to "
            "avoid FlashInfer in-memory and disk cache collisions.",
            "exact_reproduction": None,
            "within_tolerance": None,
        }
        record["checks_within_frozen_tolerance"] = all(
            item["invariants_ok"] and item["reference_tolerance_ok"]
            for arm in record["arms"].values()
            for item in arm["backends"].values()
        ) and all(
            arm["triton_vs_flashinfer_output"]["within_tolerance"]
            and arm["triton_vs_flashinfer_state"]["within_tolerance"]
            for arm in record["arms"].values()
        )
        return record
    original_output = compare(
        outputs["captured_actual_bias", original_backend][active],
        tensors["production_output"][active],
    )
    original_state = compare(
        states["captured_actual_bias", original_backend],
        tensors["production_post_state"],
    )
    record["production_reproduction"] = {
        "attempted": True,
        "backend": original_backend,
        "active_output": original_output,
        "all_captured_page_poststate": original_state,
        "exact_reproduction": original_output["exact_equal"]
        and original_state["exact_equal"],
        "within_tolerance": original_output["within_tolerance"]
        and original_state["within_tolerance"],
    }
    record["per_backend_bias_effect"] = (
        {
            backend: {
                "active_output": compare(
                    outputs["bf16_rounded_bias_fp32", backend][active],
                    outputs["captured_actual_bias", backend][active],
                ),
                "active_state": compare(
                    states["bf16_rounded_bias_fp32", backend][slots],
                    states["captured_actual_bias", backend][slots],
                ),
            }
            for backend in ("triton", "flashinfer")
        }
        if bias_mode == "main"
        else {
            "collected": False,
            "reason": "actual-only mode avoids a duplicate-value FP32 replay",
        }
    )
    record["checks_within_frozen_tolerance"] = (
        record["production_reproduction"]["within_tolerance"]
        and all(
            item["invariants_ok"] and item["reference_tolerance_ok"]
            for arm in record["arms"].values()
            for item in arm["backends"].values()
        )
        and all(
            arm["triton_vs_flashinfer_output"]["within_tolerance"]
            and arm["triton_vs_flashinfer_state"]["within_tolerance"]
            for arm in record["arms"].values()
        )
    )
    return record


def validate_source(cls, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["source_head"] == SOURCE_HEAD, "Frozen source manifest head")
    production_path = Path(inspect.getfile(cls)).resolve()
    root = production_path.parents[5]
    hashes = {}
    for relative, entry in manifest["files"].items():
        expected = entry["sha256"] if isinstance(entry, dict) else entry
        observed = sha_file(root / relative)
        require(observed == expected, f"Frozen source hash: {relative}")
        hashes[relative] = observed
    for relative, expected in SOURCE_MATH_FILES.items():
        observed = sha_file(root / relative)
        require(observed == expected, f"Frozen reference math source: {relative}")
        hashes[relative] = observed
    return {
        "source_head": SOURCE_HEAD,
        "source_root": str(root),
        "source_manifest_sha256": sha_file(manifest_path),
        "source_files": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-captures", type=int)
    parser.add_argument(
        "--bias-mode",
        choices=("main", "actual-only", "bf16-dtype-only"),
        default="main",
    )
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite replay evidence")
    if args.bias_mode == "bf16-dtype-only":
        require(
            os.environ.get("FLASHINFER_WORKSPACE_BASE"),
            "BF16 ablation requires a fresh process and an isolated "
            "FLASHINFER_WORKSPACE_BASE; do not reuse FP32 compiled artifacts",
        )
    paths = sorted(args.captures_dir.rglob("*.pt"))
    require(paths, "At least one capture required")
    discovered_count = len(paths)
    if args.max_captures is not None:
        require(args.max_captures > 0, "Positive capture selection limit")
        paths = paths[: args.max_captures]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), "CUDA replay environment required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    require(versions == EXPECTED_VERSIONS, "Pinned runtime versions")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import GDNDecode

    properties = torch.cuda.get_device_properties(device)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "helper_sha256": sha_file(Path(__file__)),
        "bias_mode": args.bias_mode,
        "flashinfer_workspace_base": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        "bias_dtype_isolation": (
            "One bias dtype per interpreter and per FlashInfer workspace; "
            "BF16 dtype ablation requires a separately launched process with "
            "an isolated FLASHINFER_WORKSPACE_BASE. No dependency cache is cleared."
        ),
        "status": "running",
        "capture_count": len(paths),
        "discovered_capture_count": discovered_count,
        "all_discovered_captures_selected": len(paths) == discovered_count,
        "torch_cpu_threads": 4,
        "cases": [],
        "tolerances": TOLERANCES,
        "runtime_versions": versions,
        "python_version": platform.python_version(),
        "device": {
            "name": properties.name,
            "uuid": str(properties.uuid),
            "capability": [properties.major, properties.minor],
        },
        "source_binding": validate_source(GDNDecode, args.source_manifest),
        "environment": {
            name: os.environ.get(name)
            for name in ("FLA_USE_FAST_OPS", "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE")
        },
        "reference": {
            "dtype": "float64",
            "qk_norm_epsilon": EPS,
            "softplus_threshold": SOFTPLUS_THRESHOLD,
            "note": (
                "Mathematical reference, not a reproduction of FP32 reduction "
                "order or approximate intrinsics."
            ),
        },
        "scope": {
            "real_decode_inputs": True,
            "independent_rank_local_replay": True,
            "multi_gpu_timing_or_sequence_replay": False,
            "original_per_tensor_strides_and_pointer_modulo_256_restored": True,
            "original_absolute_addresses_and_input_aliasing_restored": False,
            "state_scope": (
                "Only captured compact pages, slot0 and sampled unused page; "
                "other original pool pages are unobserved."
            ),
            "storage_gap_guard": (
                "Synthetic guard values in reconstructed storage; original "
                "page-padding values are not captured."
            ),
            "quality_conclusion": (
                "Within-tolerance replay differences do not establish "
                "harmlessness or model-quality equivalence."
            ),
        },
    }
    exit_code = 0
    try:
        for path in paths:
            case = replay_capture(GDNDecode, path, device, args.bias_mode)
            report["cases"].append(case)
            print(
                json.dumps(
                    {
                        "capture": str(path),
                        "checks_within_frozen_tolerance": case[
                            "checks_within_frozen_tolerance"
                        ],
                    }
                ),
                flush=True,
            )
        report["all_cases_within_frozen_tolerance"] = all(
            case["checks_within_frozen_tolerance"] for case in report["cases"]
        )
        fidelity_cases = [
            case
            for case in report["cases"]
            if case["production_reproduction"]["attempted"]
        ]
        report["production_fidelity_attempted_count"] = len(fidelity_cases)
        report["exact_fidelity_count"] = sum(
            case["production_reproduction"]["exact_reproduction"]
            for case in fidelity_cases
        )
        report["within_tol_only_fidelity_count"] = sum(
            case["production_reproduction"]["within_tolerance"]
            and not case["production_reproduction"]["exact_reproduction"]
            for case in fidelity_cases
        )
        report["all_cases_exact_production_fidelity"] = (
            report["exact_fidelity_count"] == len(fidelity_cases)
            if fidelity_cases
            else None
        )
        passed = report["all_cases_within_frozen_tolerance"] and (
            report["all_cases_exact_production_fidelity"]
            if fidelity_cases
            else args.bias_mode == "bf16-dtype-only"
        )
        report["status"] = "completed" if passed else "completed_with_failed_checks"
        exit_code = 0 if passed else 2
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mode also prevents a racing process from replacing evidence.
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
