"""Observe actual short-prefix GDN, convolution and initialized attention KV."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import re
import weakref

from divergence_contract import logical_slots, prefix_contract, validate_kv_layout
from divergence_leaf import arm_contract, patch_class

_BATCH = None
_LAST_EXECUTION_ID = 0
_EXECUTING = False
_PATCHED = set()
_SEEN = set()
_PENDING = {}
_PROFILE_CALLS = 0
_SOURCE_SHAS = {
    "vllm.v1.worker.gpu.model_runner":
        "48ae64b57dae7e26599df3e2a967a5f3ac0ad91861f7b879da989da0f5c67cf3",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn":
        "d4056cb0105ca7c13fb3fa41be8f8e2ba3a2ea4f1ac81115abe3f3f4e6b3b45d",
    "vllm.model_executor.layers.attention.attention":
        "bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2",
}


def set_owner(op, owner):
    """Keep layer identity without registering a parent as an nn.Module child."""
    op._divergence_owner_ref = weakref.ref(owner)


def last_execution_id():
    return _LAST_EXECUTION_ID


def prepared_context():
    return _BATCH


def _common():
    import short_force
    return short_force


def _event(row):
    common = _common()
    common._event({
        "pid": os.getpid(),
        "rank": common._rank(),
        "arm": common._CONFIG["arm"],
        **row,
    })


def _layer(prefix):
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        raise RuntimeError("Actual layer prefix lacks a layer number")
    return int(match[1])


def _tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.view(-1).view(__import__("torch").uint8).numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "original_shape": list(tensor.shape),
        "original_stride": list(tensor.stride()),
        "original_requires_grad": bool(tensor.requires_grad),
        "num_bytes": len(raw),
    }


def _unique(kind, layer, index, step):
    key = (kind, layer, index, step)
    if key in _SEEN:
        raise RuntimeError("Duplicate actual initial state or capture")
    _SEEN.add(key)


def patch(module):
    name = module.__name__
    if name not in _SOURCE_SHAS:
        return
    if name in _PATCHED:
        raise RuntimeError("Observation module patched twice")
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != _SOURCE_SHAS[name]:
        raise RuntimeError("Actual observation source differs from pinned d8")
    if name.endswith("model_runner"):
        patch_runner(module)
    elif name.endswith("qwen_gdn_linear_attn"):
        patch_gdn(module)
    else:
        patch_attention(module)
    _PATCHED.add(name)
    _event({"event": "short_capture_source", "module": name, "sha256": actual})


def patch_runner(module):
    cls = module.GPUModelRunner
    original_prepare = cls.prepare_inputs
    original_execute = cls.execute_model

    @functools.wraps(original_execute)
    def execute(self, *args, **kwargs):
        global _BATCH, _EXECUTING
        if _EXECUTING:
            raise RuntimeError("Nested short diagnostic execution")
        _EXECUTING, _BATCH = True, None
        try:
            return original_execute(self, *args, **kwargs)
        finally:
            _BATCH, _EXECUTING = None, False

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        global _BATCH, _LAST_EXECUTION_ID
        if not _EXECUTING or _BATCH is not None:
            raise RuntimeError("Input preparation lacks one real execute boundary")
        batch = original_prepare(self, *args, **kwargs)
        _LAST_EXECUTION_ID += 1
        rows, selected, target_step = [], False, None
        if not batch.has_prefill:
            import torch
            common = _common()
            if (self.scheduler_config.async_scheduling
                    or not self.model_config.enforce_eager
                    or batch.num_draft_tokens
                    or batch.num_tokens != batch.num_reqs
                    or list(batch.num_scheduled_tokens) != [1] * batch.num_reqs):
                raise RuntimeError("Short observation requires pure eager sync decode")
            slots = [int(item) for item in batch.idx_mapping_np]
            slot_tensor = torch.tensor(
                slots, dtype=torch.long, device=batch.input_ids.device
            )
            totals = self.req_states.total_len.gpu[slot_tensor].cpu().tolist()
            input_ids = batch.input_ids[:batch.num_reqs].cpu().tolist()
            positions = batch.positions[:batch.num_reqs].cpu().tolist()
            seq_lens = batch.seq_lens[:batch.num_reqs].cpu().tolist()
            starts = [int(item) for item in batch.query_start_loc_np]
            if len(set(slots)) != batch.num_reqs:
                raise RuntimeError("Duplicate actual persistent request slot")
            for row, (req_id, slot, total) in enumerate(
                zip(batch.req_ids, slots, totals)
            ):
                identity = common._REQUESTS.get(req_id)
                if identity is None:
                    raise RuntimeError("Prepared input has no native identity")
                fixture = common._BY_PROMPT[identity["prompt_key"]]
                prompt = fixture["prompt_token_ids"]
                step = int(total) - len(prompt)
                if not 1 <= step <= 19:
                    raise RuntimeError("Decode executed beyond the abort boundary")
                if (self.req_states.req_id_to_index.get(req_id) != slot
                        or starts[row] != row or starts[row + 1] != row + 1
                        or int(seq_lens[row]) != total):
                    raise RuntimeError("Actual input row/slot/sequence differs")
                info = {
                    "row": row, "req_id": req_id, "index": fixture["index"],
                    "request_state_slot": slot, "step": step,
                    "input_token": input_ids[row],
                    "input_position": positions[row], "total_length": total,
                    "prompt_length": len(prompt),
                    "seq_len": int(seq_lens[row]),
                }
                rows.append(info)
                if fixture["index"] == common._CONFIG["snapshot_target"]:
                    if target_step is not None:
                        raise RuntimeError("Duplicate target in actual batch")
                    target_step = step
            selected = target_step in common._CONFIG["snapshot_steps"]
            if selected:
                for info in rows:
                    identity = common._REQUESTS[info["req_id"]]
                    fixture = common._BY_PROMPT[identity["prompt_key"]]
                    history = self.req_states.all_token_ids.gpu[
                        info["request_state_slot"], :info["total_length"]
                    ].cpu().tolist()
                    info.update(prefix_contract(
                        fixture["prompt_token_ids"], fixture["reference_token_ids"],
                        history, info["step"], info["input_token"],
                        info["input_position"],
                    ))
        _BATCH = {
            "runner": self, "batch": batch,
            "execution_id": _LAST_EXECUTION_ID,
            "rows": rows, "target_step": target_step, "selected": selected,
        }
        _event({
            "event": "short_prepared_batch",
            "execution_id": _LAST_EXECUTION_ID,
            "req_ids": list(batch.req_ids),
            "num_scheduled_tokens": [
                int(item) for item in batch.num_scheduled_tokens
            ],
            "actual_target_step": target_step,
            "capture_selected": selected, "rows": rows,
        })
        return batch

    cls.execute_model, cls.prepare_inputs = execute, prepare


def _gdn_join(op, actual_page_tensor):
    import torch
    from vllm.forward_context import get_forward_context
    if _BATCH is None or _BATCH["batch"].has_prefill:
        raise RuntimeError("Decode leaf has no actual ordinary input batch")
    batch = _BATCH["batch"]
    owner_ref = getattr(op, "_divergence_owner_ref", None)
    owner = owner_ref() if owner_ref is not None else None
    if owner is None or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Leaf lacks eager owning-layer identity")
    meta = get_forward_context().attn_metadata[owner.prefix]
    if (meta.num_prefills or meta.num_spec_decodes
            or meta.num_actual_tokens != batch.num_reqs
            or meta.num_decodes != batch.num_reqs
            or meta.num_decode_tokens != batch.num_reqs):
        raise RuntimeError("Per-layer GDN metadata differs from actual decode")
    if (meta.non_spec_token_indx is not None
            and meta.non_spec_token_indx.cpu().tolist()
            != list(range(batch.num_reqs))):
        raise RuntimeError("Unreviewed GDN row permutation")
    expected_tensor = (
        meta.non_spec_flashinfer_state_indices_tensor
        if op.backend == "flashinfer" else meta.non_spec_state_indices_tensor
    )
    expected = expected_tensor.reshape(-1).cpu().tolist()
    actual = actual_page_tensor.reshape(-1).cpu().tolist()
    if actual != expected or len(actual) != batch.num_reqs:
        raise RuntimeError("Actual leaf state pages differ from layer metadata")
    if len(set(actual)) != len(actual) or any(int(page) <= 0 for page in actual):
        raise RuntimeError("Unsupported aliased/null/padded GDN page")
    contract = arm_contract(_common()._CONFIG["arm"])
    return {
        **contract, "pure_decode": True,
        "execution_id": _BATCH["execution_id"],
        "actual_target_step": _BATCH["target_step"],
        "selected": _BATCH["selected"],
        "prefix": owner.prefix, "layer": _layer(owner.prefix),
        "actual_pages": [int(item) for item in actual],
        "rows": [
            {**row, "gdn_state_page": int(actual[row["row"]])}
            for row in _BATCH["rows"]
        ],
    }


def _leaf_observer(stage, op, arguments, contract, join):
    global _PROFILE_CALLS
    import torch
    if stage == "before":
        if _BATCH is None:
            _PROFILE_CALLS += 1
            _event({
                "event": "short_excluded_profiling_leaf",
                "profile_call_number": _PROFILE_CALLS,
                "configured_backend": contract["configured_backend"],
                "actual_profile_leaf_backend": contract["configured_backend"],
                "scope": "No actual prepared request; excluded from all inference gates",
            })
            return {"excluded_profiling": True}
        join = _gdn_join(op, arguments["ssm_state_indices"])
        _event({
            "event": "short_actual_leaf_dispatch",
            "execution_id": join["execution_id"],
            "layer": join["layer"],
            "step": join["actual_target_step"],
            "configured_backend": contract["configured_backend"],
            "actual_leaf_backend": contract["actual_leaf_backend"],
        })
        if not join["selected"]:
            return join
        target = [
            row for row in join["rows"]
            if row["index"] == _common()._CONFIG["snapshot_target"]
        ]
        if len(target) != 1:
            raise RuntimeError("Selected capture lacks exactly one target")
        target = target[0]
        _unique("leaf", join["layer"], target["index"], target["step"])
        # Keep the native batch inputs and only the target recurrent page.
        # This is a propagated-state observation, not B8 exact local replay.
        payload = {
            name: arguments[name].detach().cpu().clone()
            for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias")
        }
        payload["target_state_before"] = arguments["initial_state"][
            target["gdn_state_page"]
        ].detach().cpu().clone()
        original = {
            name: _tensor_record(arguments[name])
            for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias")
        }
        _PENDING[id(op)] = (payload, original, target)
        return join
    if join.get("excluded_profiling") or not join["selected"]:
        return
    payload, original, target = _PENDING.pop(id(op))
    payload["out"] = arguments["out"].detach().cpu().clone()
    payload["target_state_after"] = arguments["initial_state"][
        target["gdn_state_page"]
    ].detach().cpu().clone()
    rank = _common()._rank()
    directory = Path(os.environ["SHORT_TRACE_DIR"]) / "captures"
    directory.mkdir(parents=True, exist_ok=True)
    name = (f"rank-{rank}-layer-{join['layer']}-step-{target['step']}"
            f"-eid-{join['execution_id']}")
    path = directory / f"{name}.pt"
    if path.exists():
        raise RuntimeError("Refusing to overwrite a capture")
    torch.save(payload, path)
    sidecar = {
        "schema": "gdn-short-capture-v1",
        "arm": contract["arm"], "rank": rank,
        "configured_backend": contract["configured_backend"],
        "actual_leaf_backend": contract["actual_leaf_backend"],
        "runtime_join": join, "original_tensors": original,
        "binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "binary_size": path.stat().st_size,
        "scope": (
            "Actual batch inputs/output and target propagated recurrent page. "
            "Not a full-state-pool replay or model accuracy measurement."
        ),
    }
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf8"
    )
    _event({
        "event": "short_leaf_capture", "execution_id": join["execution_id"],
        "layer": join["layer"], "index": target["index"],
        "step": target["step"], "file": path.name,
        "binary_sha256": sidecar["binary_sha256"],
    })


def patch_gdn(module):
    cls = module.QwenGatedDeltaNetAttention
    original_init = cls.__init__
    original_decode = cls._forward_core_decode_non_spec
    patch_class(module.GDNDecode, _common()._CONFIG["arm"], _leaf_observer)

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        set_owner(self.gdn_decode, self)

    @functools.wraps(original_decode)
    def decode(self, *args, **kwargs):
        if (_BATCH is not None and _BATCH["selected"]
                and _BATCH["target_step"] == 1):
            from vllm.forward_context import get_forward_context
            meta = get_forward_context().attn_metadata[self.prefix]
            pages = meta.non_spec_state_indices_tensor.cpu().tolist()
            if (len(pages) != len(_BATCH["rows"])
                    or len(set(pages)) != len(pages)
                    or any(int(page) <= 0 for page in pages)):
                raise RuntimeError("Initial GDN state pages are not active unique rows")
            conv = self.kv_cache[0]
            if not module.is_conv_state_dim_first():
                conv = conv.transpose(-1, -2)
            for info, page in zip(_BATCH["rows"], pages):
                for kind, pool in (
                    ("conv", conv), ("recurrent", self.kv_cache[1])
                ):
                    _unique(kind, _layer(self.prefix), info["index"], 1)
                    _event({
                        "event": "short_initial_state", "kind": kind,
                        "execution_id": _BATCH["execution_id"],
                        "layer": _layer(self.prefix), "prefix": self.prefix,
                        "index": info["index"], "step": 1,
                        "request_state_slot": info["request_state_slot"],
                        "state_page": int(page), "row": info["row"],
                        "req_id": info["req_id"],
                        "value": _tensor_record(pool[int(page)]),
                    })
        return original_decode(self, *args, **kwargs)

    cls.__init__, cls._forward_core_decode_non_spec = initialize, decode


def patch_attention(module):
    cls = module.Attention
    original = cls.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        if (_BATCH is not None and _BATCH["selected"]
                and _BATCH["target_step"] == 1):
            import torch
            from vllm.forward_context import get_forward_context
            fc = get_forward_context()
            prefix = self.layer_name
            meta = fc.attn_metadata[prefix]
            cache = self.kv_cache
            backend = (
                type(self.impl).__module__ + "." + type(self.impl).__qualname__
            )
            layout = validate_kv_layout(
                backend, list(cache.shape), self.head_size, str(cache.dtype)
            )
            if (self.kv_sharing_target_layer_name is not None
                    or getattr(meta, "use_cascade", False)
                    or getattr(self.impl, "sliding_window", None)
                    not in (None, (-1, -1))
                    or getattr(meta, "num_actual_tokens", None)
                    != len(_BATCH["rows"])):
                raise RuntimeError("Unreviewed shared/cascade/window/packed KV path")
            table = meta.block_table.cpu().tolist()
            seq_lens = meta.seq_lens.cpu().tolist()
            slot_map = fc.slot_mapping[prefix][:len(_BATCH["rows"])].cpu().tolist()
            if meta.query_start_loc.cpu().tolist() != list(
                range(len(_BATCH["rows"]) + 1)
            ):
                raise RuntimeError("KV row order differs from one-token input rows")
            for info in _BATCH["rows"]:
                row, valid = info["row"], info["prompt_length"]
                if info["step"] != 1 or int(seq_lens[row]) != valid + 1:
                    raise RuntimeError("Initial KV does not precede first decode")
                slots = logical_slots(
                    [int(item) for item in table[row]], valid,
                    layout["block_size"], layout["num_blocks"],
                )
                block_size = layout["block_size"]
                current_page = int(table[row][valid // block_size])
                if int(slot_map[row]) != (
                    current_page * block_size + valid % block_size
                ):
                    raise RuntimeError("Actual current KV slot differs from block table")
                pages = torch.tensor(
                    [item[0] for item in slots], device=cache.device
                )
                offsets = torch.tensor(
                    [item[1] for item in slots], device=cache.device
                )
                # Logical position order, heads, K then V; no unwritten tail.
                logical = cache[pages, :, offsets, :]
                _unique("kv", _layer(prefix), info["index"], 1)
                _event({
                    "event": "short_initial_state", "kind": "kv",
                    "execution_id": _BATCH["execution_id"], "layer": _layer(prefix),
                    "prefix": prefix, "index": info["index"], "step": 1,
                    "req_id": info["req_id"], "row": row,
                    "request_state_slot": info["request_state_slot"],
                    "valid_tokens": valid, "current_slot": int(slot_map[row]),
                    "logical_pages": list(dict.fromkeys(item[0] for item in slots)),
                    "backend": backend, "cache_layout": layout,
                    "cache_shape": list(cache.shape),
                    "cache_stride": list(cache.stride()),
                    "value": _tensor_record(logical),
                    "scope": "Initialized prompt K/V only; tail/current token excluded",
                })
        return original(self, *args, **kwargs)

    cls.forward = forward
