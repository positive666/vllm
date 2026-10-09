"""Bounded real-decode capture; no shadow kernel and no production source edits.

Set GDN_CAPTURE_ACTIVE=1, GDN_CAPTURE_CONFIG=/absolute/config.json and prepend
this directory to PYTHONPATH. sitecustomize installs a deferred module hook in
every spawned process. This diagnostic requires --enforce-eager. Snapshot copies
and CPU serialization can change scheduling and the generated token trajectory;
these captures are not an uninstrumented performance or accuracy experiment.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import traceback

_TARGET = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_LOCK = threading.Lock()
_CONFIG = None
_OUTPUT = None
_START = None
_STARTED = False
_PATCHED = False


def _json_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _event(value):
    path = _OUTPUT / f"events-pid-{os.getpid()}.jsonl"
    with _LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()


def _fail_closed():
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    os._exit(97)


def _rank(context):
    import torch

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", context["tp_rank"]))


def _tensor_spec(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr_mod_16": tensor.data_ptr() % 16,
        "data_ptr_mod_128": tensor.data_ptr() % 128,
        "data_ptr_mod_256": tensor.data_ptr() % 256,
        "requires_grad": tensor.requires_grad,
    }


def _choose_capture(op, batch_size):
    global _STARTED
    if not _STARTED:
        if not _START.is_file():
            return None
        _STARTED = True
        _event({"event": "start_seen", "pid": os.getpid(),
                "start_marker_sha256": hashlib.sha256(_START.read_bytes()).hexdigest()})
    context = getattr(op, "_gdn_capture_context", None)
    if context is None:
        raise RuntimeError("Real GDN decode has no registered layer context")
    if not context["started"]:
        if batch_size != _CONFIG["initial_batch_size"]:
            context["before_first_b8_calls"] += 1
            return None
        context["started"] = True
    context["call_index"] += 1
    call_index = context["call_index"]
    if (call_index in _CONFIG["early_calls"] or
            (context["layer_idx"] in _CONFIG["sample_layers"] and
             call_index in _CONFIG["sample_calls"])):
        return context
    return None


def _capture_call(original, op, mixed_qkv, a, b, A_log, dt_bias,
                  initial_state, ssm_state_indices, out):
    import torch

    context = _choose_capture(op, mixed_qkv.shape[0])
    arguments = dict(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log,
                     dt_bias=dt_bias, initial_state=initial_state,
                     ssm_state_indices=ssm_state_indices, out=out)
    if context is None:
        return original(op, **arguments)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Capture requires eager execution outside CUDA graphs")
    if mixed_qkv.shape[0] != ssm_state_indices.numel():
        raise RuntimeError("Capture supports only ordinary packed one-token decode")
    original_indices = ssm_state_indices.detach().cpu().clone()
    raw_indices = original_indices.reshape(-1).tolist()
    joined = context["runtime_join"]["rows"]
    if raw_indices != [row["gdn_state_page"] for row in joined]:
        raise RuntimeError("Actual GDN argument indices differ from joined layer metadata")
    if op.backend == "flashinfer":
        if any(index < -1 or index == 0 for index in raw_indices):
            raise RuntimeError("Expected FI null=-1 and active vLLM blocks >0")
        padding = [index == -1 for index in raw_indices]
    elif op.backend == "triton":
        if any(index < 0 for index in raw_indices):
            raise RuntimeError("Expected Triton null=0 and active blocks >0")
        padding = [index == 0 for index in raw_indices]
    else:
        raise RuntimeError(f"Unexpected production backend: {op.backend}")
    active = [index for index, pad in zip(raw_indices, padding) if not pad]
    if len(active) != len(set(active)):
        raise RuntimeError("Duplicate active state indices make replay ambiguous")
    if any(index >= initial_state.shape[0] for index in active):
        raise RuntimeError("Active state index is out of bounds")
    active = sorted(active)
    excluded = {0, *active}
    unused_original = next((index for index in range(1, initial_state.shape[0])
                            if index not in excluded), None)
    compact_to_original = [0, *active]
    if unused_original is not None:
        compact_to_original.append(unused_original)
    gather = torch.tensor(compact_to_original, dtype=torch.long,
                          device=initial_state.device)
    compact_indices = original_indices.clone()
    for compact, original_index in enumerate(active, start=1):
        compact_indices[original_indices == original_index] = compact

    def compact_state():
        captured = initial_state.detach().index_select(0, gather)
        if unused_original is None:
            captured = torch.cat((captured, torch.zeros_like(captured[:1])), dim=0)
        return captured

    # Device copies precede the actual in-place update on the current stream.
    # Only selected pages are copied; the full recurrent state pool is untouched.
    gpu_snapshots = {
        "mixed_qkv": mixed_qkv.detach().clone(),
        "a": a.detach().clone(), "b": b.detach().clone(),
        "A_log": A_log.detach().clone(), "dt_bias": dt_bias.detach().clone(),
        "initial_state": compact_state(),
    }
    result = original(op, **arguments)
    gpu_snapshots["production_output"] = out.detach().clone()
    gpu_snapshots["production_post_state"] = compact_state()
    tensors = {name: tensor.cpu() for name, tensor in gpu_snapshots.items()}
    tensors["indices_original"] = original_indices
    tensors["indices_compact"] = compact_indices
    tensors["padding_rows"] = torch.tensor(padding, dtype=torch.bool)
    rank = _rank(context)
    metadata = {
        "schema_version": 2, "runtime_join": context["runtime_join"],
        "rank": rank, "tp_rank": context["tp_rank"],
        "source_head": _CONFIG.get("source_head",
            "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"),
        "source_head_scope": "experiment revision; actual module hash in patched-pid record",
        "pid": os.getpid(), "prefix": context["prefix"],
        "layer_idx": context["layer_idx"], "call_index": context["call_index"],
        "before_first_b8_calls": context["before_first_b8_calls"],
        "batch_size": mixed_qkv.shape[0], "original_backend": op.backend,
        "num_heads_qk": op.num_k_heads, "num_heads_v": op.num_v_heads,
        "head_qk_dim": op.head_k_dim, "head_v_dim": op.head_v_dim,
        "scale": op.head_k_dim ** -0.5, "use_qk_l2norm_in_kernel": True,
        "padding_rows": padding, "compact_to_original": compact_to_original +
            ([-1] if unused_original is None else []),
        "active_original_indices": active,
        "unused_compact_index": len(active) + 1,
        "unused_original_index": unused_original,
        "synthetic_unused_slot": unused_original is None,
        "original_shapes_strides": {name: _tensor_spec(tensor)
                                     for name, tensor in arguments.items()},
        "state_scope": "null, active pages and one sampled unused page only",
        "capture_scope": "ordinary packed decode in eager mode; no mixed-path coverage",
        "trajectory_scope": "copy and serialization synchronization may change scheduling",
        "values_scope": "CPU tensor exact values; compact copies do not preserve original strides",
    }
    path = (_OUTPUT / f"rank-{rank}" / f"layer-{context['layer_idx']:03d}" /
            f"call-{context['call_index']:06d}-pid-{os.getpid()}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save({"metadata": metadata, "tensors": tensors}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"event": "capture", "file": str(path.relative_to(_OUTPUT)),
              "sha256": digest, "bytes": path.stat().st_size,
              "metadata": metadata}
    _json_new(path.with_suffix(".json"), record)
    _event(record)
    return result


def _patch(module):
    global _PATCHED
    if _PATCHED:
        raise RuntimeError("GDN capture patch unexpectedly installed twice")
    source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    expected = _CONFIG.get("expected_module_sha256")
    if expected is not None and source_hash != expected:
        raise RuntimeError("GDN source hash differs from capture configuration")
    attention = module.QwenGatedDeltaNetAttention
    decode = module.GDNDecode
    init_original = attention.__init__
    signature = inspect.signature(init_original)

    @functools.wraps(init_original)
    def init_capture(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        prefix = bound.arguments["prefix"]
        init_original(self, *args, **kwargs)
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
        if match is None:
            raise RuntimeError(f"Cannot determine actual GDN layer from {prefix!r}")
        context = {"prefix": prefix, "layer_idx": int(match.group(1)),
                   "tp_rank": int(self.tp_rank), "started": False,
                   "call_index": 0, "before_first_b8_calls": 0}
        if hasattr(self.gdn_decode, "_gdn_capture_context"):
            raise RuntimeError("GDN capture context was already registered")
        self.gdn_decode._gdn_capture_context = context
        _event({"event": "layer_registered", "pid": os.getpid(),
                "rank": _rank(context), "context": dict(context)})

    attention.__init__ = init_capture
    for method in ("forward_native", "forward_cuda"):
        original = getattr(decode, method)

        def make_wrapper(function):
            @functools.wraps(function)
            def wrapper(self, mixed_qkv, a, b, A_log, dt_bias,
                        initial_state, ssm_state_indices, out):
                try:
                    return _capture_call(function, self, mixed_qkv, a, b, A_log,
                                         dt_bias, initial_state, ssm_state_indices, out)
                except BaseException:
                    _event({"event": "capture_failure", "pid": os.getpid(),
                            "traceback": traceback.format_exc()})
                    raise
            return wrapper

        setattr(decode, method, make_wrapper(original))
    _PATCHED = True
    _json_new(_OUTPUT / f"patched-pid-{os.getpid()}.json",
              {"event": "patched", "pid": os.getpid(), "module": _TARGET,
               "module_file": module.__file__, "module_sha256": source_hash,
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


class _CaptureLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        try:
            self.original.exec_module(module)
            _patch(module)
        except BaseException:
            _fail_closed()


class _CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot locate capture target {fullname}")
        spec.loader = _CaptureLoader(spec.loader)
        return spec


def install():
    """Install once in each API or spawned rank process; failures are fatal."""
    global _CONFIG, _OUTPUT, _START
    if os.environ.get("GDN_CAPTURE_ACTIVE") != "1":
        return
    if _CONFIG is not None:
        raise RuntimeError("Capture install called twice in one process")
    config_path = Path(os.environ["GDN_CAPTURE_CONFIG"])
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    for key in ("output_dir", "start_marker", "early_calls", "sample_calls",
                "sample_layers", "initial_batch_size"):
        if key not in config:
            raise ValueError(f"Missing capture configuration: {key}")
    for key in ("early_calls", "sample_calls", "sample_layers"):
        if not isinstance(config[key], list) or any(
            not isinstance(value, int) or value < (0 if key == "sample_layers" else 1)
            for value in config[key]
        ):
            raise ValueError(f"Invalid capture configuration: {key}")
    if not isinstance(config["initial_batch_size"], int) or config["initial_batch_size"] < 1:
        raise ValueError("Invalid initial_batch_size")
    _CONFIG = config
    _OUTPUT = Path(config["output_dir"]).resolve()
    _START = Path(config["start_marker"]).resolve()
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    _json_new(_OUTPUT / f"installed-pid-{os.getpid()}.json",
              {"event": "installed", "pid": os.getpid(),
               "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
               "config": config, "target": _TARGET})
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _CaptureFinder())


if __name__ == "__main__":
    raise SystemExit("Use sitecustomize and start the regular vLLM API server")
