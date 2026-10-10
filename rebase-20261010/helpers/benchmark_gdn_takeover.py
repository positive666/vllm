"""Compare the actual GDNDecode wrappers with identical packed inputs.

Boundary: post-convolution gate/QK normalization/recurrent decode followed by
the existing gated output RMSNorm. Projection, convolution and step-shared
metadata construction are excluded. Metadata index remapping is measured
separately, once per model step, rather than charged once per GDN layer.

Timing defaults to CUPTI with CUDA graphs; its automatic event fallback is
detected and recorded and is never presented as CUPTI kernel timing.
Cold runs rotate pointer-distinct active state exceeding 5x device L2. Allocation,
compilation, correctness checks, random generation and state reset are untimed.
Only public production wrapper configurations are compared; no private FI
kernel knobs, runtime patches or old CUDA branch are substituted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import statistics
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import GDNDecode
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd

H, HV, K, V = 16, 48, 128, 128
SCALE, EPS = K**-0.5, 1e-6
BACKENDS = ("triton", "flashinfer")
TOLERANCES = {
    "relative_l2": 0.01,
    "float32_state": {"atol": 0.01, "rtol": 0.01},
    "bfloat16_state": {"atol": 0.02, "rtol": 0.01},
    "scope": (
        "same dtype gate for raw output, state and normalized output; "
        "never relaxed for long trajectories"
    ),
}


def timed_samples(function, timer, repeat_ms, graph_iters):
    from flashinfer import testing

    common = dict(cold_l2_cache=False, dry_run_time_ms=25, repeat_time_ms=repeat_ms)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if timer == "cupti":
            samples = testing.bench_gpu_time_with_cupti(
                function, use_cuda_graph=True, **common
            )
        else:
            samples = testing.bench_gpu_time_with_cudagraph(
                function, num_iters_within_graph=graph_iters, **common
            )
    messages = [str(warning.message) for warning in caught]
    fallback = any("falling back" in message.lower() for message in messages)
    method = "cuda_graph_events" if timer == "graph" or fallback else "cupti_cuda_graph"
    return samples, {"method": method, "warnings": messages}


def package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def clone_layout(tensor):
    # Include trailing storage after the last page too, so every page has a
    # sentinel region whose accidental writes can be checked exactly.
    storage = torch.full(
        (tensor.shape[0] * tensor.stride(0),),
        0.9375,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result = storage.as_strided(tensor.shape, tensor.stride())
    return result.copy_(tensor)


def errors(actual, expected, atol, rtol):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-20)).item(),
        "pointwise_over_threshold": int(
            (delta.abs() > atol + rtol * expected.abs()).sum()
        ),
    }


def require_close(actual, expected, atol, rtol, label):
    record = errors(actual, expected, atol, rtol)
    if not (
        record["finite"]
        and record["relative_l2"] < TOLERANCES["relative_l2"]
        and record["pointwise_over_threshold"] == 0
    ):
        a, e = actual.float().flatten(), expected.float().flatten()
        mask = (a - e).abs() > atol + rtol * e.abs()
        indices = torch.where(mask)[0][:5]
        examples = [
            {
                "flat_index": int(index),
                "actual": float(a[index]),
                "expected": float(e[index]),
                "allowed_abs_error": float(atol + rtol * e[index].abs()),
            }
            for index in indices
        ]
        raise AssertionError(
            {
                "label": label,
                "metrics": record,
                "atol": atol,
                "rtol": rtol,
                "examples": examples,
            }
        )
    return record


class Case:
    def __init__(self, batch, dtype, index_dtype, seed, padding=False):
        torch.manual_seed(seed)
        self.batch, self.dtype, self.padding = batch, dtype, padding
        self.atol, self.rtol = (0.02 if dtype == torch.bfloat16 else 0.01), 0.01
        rand = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        self.packed = rand(batch, 2 * H * K + 2 * HV * V)
        self.qkv = self.packed[:, : 2 * H * K + HV * V]
        self.gate = self.packed[:, 2 * H * K + HV * V :].view(batch, HV, V)
        self.ba = rand(batch, 2 * HV)
        self.b, self.a = self.ba.chunk(2, dim=-1)
        self.alog = torch.randn(HV, device="cuda") * 0.1 - 2
        # The actual checkpoint stores dt_bias as BF16. Production Triton
        # keeps those values in BF16; the FI-only parameter stores the exact
        # same values in FP32 to satisfy the kernel ABI, once before forward.
        self.bias = (torch.randn(HV, device="cuda") * 0.1).bfloat16()
        self.fi_bias = self.bias.float()
        self.weight = (1 + 0.1 * torch.randn(V, device="cuda")).bfloat16()
        # Mamba cache pages can have unused trailing storage in every page.
        pages = torch.randn(batch + 5, HV * V * K + 256, device="cuda", dtype=dtype)
        self.initial = pages[:, : HV * V * K].view(batch + 5, HV, V, K)
        self.initial.mul_(0.01)
        self.states = {backend: clone_layout(self.initial) for backend in BACKENDS}
        self.raw = {
            backend: torch.zeros(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16)
            for backend in BACKENDS
        }
        self.outputs = {
            backend: torch.empty(batch, HV, V, device="cuda", dtype=torch.bfloat16)
            for backend in BACKENDS
        }
        self.indices = torch.arange(batch, 0, -1, device="cuda", dtype=index_dtype)
        if padding:
            if batch < 3:
                raise ValueError(
                    "Padding needs at least one active and two padded rows"
                )
            self.indices[-2:] = torch.tensor([0, -1], device="cuda", dtype=index_dtype)
        self.fi_indices = torch.empty_like(self.indices)
        self.padding_index = torch.full((), -1, device="cuda", dtype=index_dtype)
        self.remap_indices()
        self.ops = {}
        for backend in BACKENDS:
            config = VllmConfig(kernel_config=KernelConfig(gdn_decode_backend=backend))
            # Only dtype is consumed by the kernel wrapper's eligibility gate.
            # Loading the model/tokenizer belongs to the separate serving run.
            config.model_config = SimpleNamespace(dtype=torch.bfloat16)
            config.compilation_config.custom_ops = ["all"]
            with set_current_vllm_config(config):
                self.ops[backend] = GDNDecode(H, HV, K, V, dtype)
            if self.ops[backend].backend != backend:
                raise AssertionError((backend, self.ops[backend].backend))

    def close(self, actual, expected, label):
        return require_close(actual, expected, self.atol, self.rtol, label)

    def close_output(self, actual, expected, label):
        # Serving ignores padded rows. FI BF16 defines their output as
        # unspecified, whereas its FP32 kernel skips the output write.
        active = self.indices > 0
        return self.close(actual[active], expected[active], label)

    def remap_indices(self):
        torch.where(
            self.indices <= 0,
            self.padding_index,
            self.indices,
            out=self.fi_indices,
        )

    def run(self, backend, state=None):
        state = self.states[backend] if state is None else state
        indices = self.fi_indices if backend == "flashinfer" else self.indices
        bias = self.fi_bias if backend == "flashinfer" else self.bias
        self.ops[backend](
            self.qkv,
            self.a,
            self.b,
            self.alog,
            bias,
            state,
            indices,
            self.raw[backend],
        )
        layer_norm_fwd(
            self.raw[backend].view(-1, V),
            self.weight,
            None,
            EPS,
            z=self.gate.reshape(-1, V),
            out=self.outputs[backend].view(-1, V),
            norm_before_gate=True,
            is_rms_norm=True,
            activation="silu",
        )

    def reset(self):
        for backend in BACKENDS:
            self.states[backend].copy_(self.initial)
            self.raw[backend].zero_()

    def reference(self):
        state_ref = self.initial.clone()
        out = torch.zeros_like(self.outputs["triton"])
        raw_out = torch.zeros_like(self.raw["triton"])
        active = self.indices > 0
        slots = self.indices[active].long()
        n = slots.numel()
        q, k, v = self.qkv[active].double().split([H * K, H * K, HV * V], -1)
        q, k = [x.view(n, H, K).repeat_interleave(HV // H, 1) for x in (q, k)]
        q = q / (q.square().sum(-1, keepdim=True) + EPS).sqrt() * SCALE
        k = k / (k.square().sum(-1, keepdim=True) + EPS).sqrt()
        decay = (
            -self.alog.double().exp()
            * F.softplus(self.a[active].double() + self.bias.double())
        ).exp()
        state = self.initial[slots].double() * decay[..., None, None]
        delta = v.view(n, HV, V) - (state * k[..., None, :]).sum(-1)
        delta *= self.b[active].double().sigmoid()[..., None]
        state += delta[..., None] * k[..., None, :]
        raw = (state * q[..., None, :]).sum(-1).bfloat16().double()
        raw_out[active, 0] = raw.to(torch.bfloat16)
        normalized = raw * (raw.square().mean(-1, keepdim=True) + EPS).rsqrt()
        out[active] = (
            normalized * self.weight.double() * F.silu(self.gate[active].double())
        ).bfloat16()
        state_ref[slots] = state.to(self.dtype)
        return out, state_ref, raw_out

    def check_inactive(self):
        active_slots = self.indices[self.indices > 0].long()
        untouched = torch.ones(self.initial.shape[0], device="cuda", dtype=torch.bool)
        untouched[active_slots] = False
        for backend in BACKENDS:
            require_close(
                self.states[backend][untouched],
                self.initial[untouched],
                0,
                0,
                f"{backend}/inactive_state_including_null_slot0_exact",
            )
            page_stride = self.states[backend].stride(0)
            pages = self.states[backend].as_strided(
                (self.initial.shape[0], page_stride), (page_stride, 1)
            )
            page_padding = pages[:, HV * V * K :]
            require_close(
                page_padding,
                torch.full_like(page_padding, 0.9375),
                0,
                0,
                f"{backend}/cache_page_padding_exact",
            )

    def correctness(self, steps):
        self.reset()
        expected_output, expected_state, expected_raw = self.reference()
        record = {"one_step": {}, "trajectory": {}, "steps": steps}
        self.correctness_record = record
        for backend in BACKENDS:
            self.run(backend)
            self.check_inactive()
            record["one_step"][backend] = {
                "raw_output": self.close_output(
                    self.raw[backend], expected_raw, f"one_step/{backend}/raw_output"
                ),
                "output": self.close_output(
                    self.outputs[backend],
                    expected_output,
                    f"one_step/{backend}/normalized_output",
                ),
                "state": self.close(
                    self.states[backend], expected_state, f"one_step/{backend}/state"
                ),
            }
        self.check_inactive()
        # Capture with the actual cache strides and packed row strides. Remapping
        # modifies preallocated metadata; graphs must read its new contents.
        for backend in BACKENDS:
            for _ in range(3):
                self.run(backend)
        torch.cuda.synchronize()
        graphs = {}
        for backend in BACKENDS:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.run(backend)
            graphs[backend] = graph
        self.reset()
        for step in range(steps):
            self.packed.normal_()
            self.ba.normal_()
            if step:
                self.indices.copy_(self.indices.roll(1))
                self.remap_indices()
            # Production Qwen forward initializes core output before invoking
            # the wrapper. Padded outputs are excluded from parity; unused
            # cache slots, including slot 0, must still remain exact.
            for backend in BACKENDS:
                self.raw[backend].zero_()
            for backend in BACKENDS:
                graphs[backend].replay()
            self.check_inactive()
            raw_error = self.close_output(
                self.raw["flashinfer"],
                self.raw["triton"],
                f"trajectory/step{step + 1}/raw_output",
            )
            oe = self.close_output(
                self.outputs["flashinfer"],
                self.outputs["triton"],
                f"trajectory/step{step + 1}/normalized_output",
            )
            se = self.close(
                self.states["flashinfer"],
                self.states["triton"],
                f"trajectory/step{step + 1}/state",
            )
            self.check_inactive()
            for field, values in (
                ("raw_output", raw_error),
                ("output", oe),
                ("state", se),
            ):
                maxima = record["trajectory"].setdefault(field, {})
                for key, value in values.items():
                    if key != "finite" and value > maxima.get(key, -1):
                        maxima[key] = value
                        maxima[key + "_step"] = step + 1
        record["graph_replay"] = True
        record["metadata_remap"] = True
        record["inactive_slots_bitwise_unchanged"] = True
        record["cache_page_padding_bitwise_unchanged"] = True
        record["output_contract"] = (
            "compare positive-index rows only; padded outputs are ignored by serving; "
            "inactive state including null slot 0 and all page padding stay exact"
        )
        return record

    def timing(self, backend, cache, repeat_ms, l2_multiple, timer):
        active_bytes = self.batch * HV * V * K * self.initial.element_size()
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        rotations = (
            max(2, math.ceil(l2_multiple * l2 / active_bytes) + 1)
            if cache == "cold"
            else 1
        )
        buffers = [clone_layout(self.initial) for _ in range(rotations)]
        for _ in range(8):
            self.run(backend, buffers[0])
        torch.cuda.synchronize()

        def group():
            for state in buffers:
                self.run(backend, state)

        samples, timer_record = timed_samples(
            group, timer, repeat_ms, 1 if cache == "cold" else 16
        )
        samples_us = [value * 1000 / rotations for value in samples]
        median = statistics.median(samples_us)
        # The state is read and written once; report its theoretical traffic.
        # Gate/QKV/norm traffic is omitted and kernels can reuse cached state.
        return {
            "median_us": median,
            "p10_us": float(torch.tensor(samples_us).quantile(0.1)),
            "p90_us": float(torch.tensor(samples_us).quantile(0.9)),
            "samples": len(samples_us),
            "rotations": rotations,
            "active_state_bytes_per_rotation": active_bytes,
            "rotated_active_state_bytes": rotations * active_bytes,
            "state_read_write_GB_per_s": 2 * active_bytes / median / 1000,
            "timer": timer_record,
        }

    def metadata_timing(self, repeat_ms, timer):
        samples, timer_record = timed_samples(self.remap_indices, timer, repeat_ms, 16)
        return {
            "median_us": statistics.median(samples) * 1000,
            "scope": (
                "one preallocated pad-index mapping per model step, "
                "shared by all layers"
            ),
            "timer": timer_record,
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-ms", type=int, default=100)
    parser.add_argument("--l2-multiple", type=int, default=5)
    parser.add_argument("--timer", choices=("cupti", "graph"), default="cupti")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one authorized GPU")
    source = Path(inspect.getfile(GDNDecode))
    report = {
        "source_head": args.source_head,
        "source_file": str(source),
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": str(torch.cuda.get_device_properties(0)),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "packages": {
            p: package_version(p)
            for p in (
                "torch",
                "triton",
                "vllm",
                "flashinfer-python",
                "nvidia-cutlass-dsl",
            )
        },
        "shape": {"H": H, "HV": HV, "K": K, "V": V, "TP": 1},
        "boundary": "production GDNDecode + identical gated RMSNorm, post-convolution",
        "metadata_boundary": (
            "precomputed pad mapping outside per-layer boundary, measured separately"
        ),
        "timing": (
            "CUPTI with CUDA graphs by default; event fallback recorded per sample; "
            "state rotation for cold cache"
        ),
        "tuning": (
            "public production wrapper defaults; no private-kernel tuning "
            "or universal performance claim"
        ),
        "thresholds": TOLERANCES,
        "dt_bias_policy": (
            "same BF16 checkpoint-representable values; Triton BF16 parameter "
            "and FI FP32 representation prepared once outside timed forward"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Complete correctness for every configuration before publishing timing.
    cases = []
    try:
        for dtype_name in args.dtypes.split(","):
            for batch in map(int, args.batches.split(",")):
                for index_name in args.index_dtypes.split(","):
                    for padding in (False, True) if batch >= 3 else (False,):
                        row = {
                            "batch": batch,
                            "state_dtype": dtype_name,
                            "index_dtype": index_name,
                            "padding": padding,
                        }
                        report["cases"].append(row)
                        print(json.dumps({"event": "case_start", **row}), flush=True)
                        case = None
                        try:
                            case = Case(
                                batch,
                                getattr(torch, dtype_name),
                                getattr(torch, index_name),
                                args.seed,
                                padding,
                            )
                            row["layouts"] = {
                                name: {
                                    "shape": list(value.shape),
                                    "stride": list(value.stride()),
                                }
                                for name, value in (
                                    ("qkv", case.qkv),
                                    ("a", case.a),
                                    ("state", case.initial),
                                )
                            }
                            row["correctness"] = case.correctness(args.steps)
                            row["passed"] = True
                            # Time active, int32 production cases once. Padding and
                            # int64 cases retain correctness only.
                            if index_name == "int32" and not padding:
                                cases.append((case, row))
                        except Exception as exc:
                            row["passed"] = False
                            row["error"] = type(exc).__name__ + ": " + str(exc)
                            if case is not None:
                                row["correctness_progress"] = getattr(
                                    case, "correctness_record", {}
                                )
                            if (
                                isinstance(exc, AssertionError)
                                and exc.args
                                and isinstance(exc.args[0], dict)
                            ):
                                row["failure"] = exc.args[0]
                            # Numerical/configuration failures remain auditable;
                            # CUDA faults invalidate the process and must abort.
                            if any(
                                message in str(exc).lower()
                                for message in (
                                    "illegal memory access",
                                    "device-side assert",
                                    "unspecified launch failure",
                                )
                            ):
                                save()
                                raise
                        save()
                        if case is not None and not any(
                            saved is case for saved, _ in cases
                        ):
                            del case
        report["all_correctness_passed"] = all(
            row.get("passed", False) for row in report["cases"]
        )
        save()
        if not report["all_correctness_passed"]:
            raise AssertionError(
                "Correctness failed; all case diagnostics preserved, no timing was run"
            )
        if not args.correctness_only:
            for case, row in cases:
                row["timings"] = {
                    cache: {backend: [] for backend in BACKENDS}
                    for cache in ("warm", "cold")
                }
                case.reset()
                row["metadata_mapping_timing"] = case.metadata_timing(
                    args.repeat_ms, args.timer
                )
                for cache in ("warm", "cold"):
                    for round_index in range(args.rounds):
                        order = BACKENDS if round_index % 2 == 0 else BACKENDS[::-1]
                        for backend in order:
                            case.reset()
                            row["timings"][cache][backend].append(
                                case.timing(
                                    backend,
                                    cache,
                                    args.repeat_ms,
                                    args.l2_multiple,
                                    args.timer,
                                )
                            )
                        save()
                print(
                    json.dumps(
                        {
                            "event": "timing_done",
                            "batch": case.batch,
                            "state_dtype": str(case.dtype),
                            "timings": row["timings"],
                        }
                    ),
                    flush=True,
                )
        report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        save()
        raise
    finally:
        cases.clear()
        gc.collect()
    print("GDN_TAKEOVER_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
