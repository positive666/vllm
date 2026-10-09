"""Join actual V2 input rows to actual GDN state pages before each update.

CPU snapshot copies require eager synchronous execution. No production state is
substituted, reset or shadow-updated. Request-state slots are never used as GDN
pool page numbers. Snapshots come from independently propagated backend states.
"""
from __future__ import annotations
import functools
import hashlib
import os
from pathlib import Path

_BATCH = None
_LAST_EXECUTION_ID = 0
_EXECUTING = False
_PATCHED = False

def last_execution_id():
    return _LAST_EXECUTION_ID

def patch_runner(module):
    global _PATCHED
    if _PATCHED:
        raise RuntimeError("Late input observation patched twice")
    import force_decode as common
    runner_cls = module.GPUModelRunner
    prepare_original = runner_cls.prepare_inputs
    execute_original = runner_cls.execute_model
    input_file = Path(module.__file__).parent / "input_batch.py"
    if hashlib.sha256(input_file.read_bytes()).hexdigest() != common._CONFIG["expected_input_batch_sha256"]:
        raise RuntimeError("Actual InputBatch source differs from frozen source")
    @functools.wraps(prepare_original)
    def prepare(self, *args, **kwargs):
        global _BATCH, _LAST_EXECUTION_ID
        if not _EXECUTING or _BATCH is not None:
            raise RuntimeError("Real input preparation lacks a unique execute boundary")
        batch = prepare_original(self, *args, **kwargs)
        _LAST_EXECUTION_ID += 1
        # One actual GPU step read per execute avoids per-layer joins on the
        # decode steps that do not request snapshots.
        target_step = None
        if not batch.has_prefill:
            target_rows = []
            for row, req_id in enumerate(batch.req_ids):
                identity = common._REQUESTS.get(req_id)
                if identity is None:
                    raise RuntimeError("Prepared input lacks the native request identity")
                fixture = common._BY_PROMPT[identity["prompt_key"]]
                if fixture["index"] == common._CONFIG["snapshot_target"]:
                    target_rows.append((row, req_id, fixture))
            if len(target_rows) > 1:
                raise RuntimeError("Prepared input contains duplicate snapshot targets")
            if target_rows:
                row, req_id, fixture = target_rows[0]
                slot = int(batch.idx_mapping_np[row])
                if self.req_states.req_id_to_index.get(req_id) != slot:
                    raise RuntimeError("Snapshot prefilter request slot differs")
                total = int(self.req_states.total_len.gpu[slot].cpu().item())
                target_step = total - len(fixture["prompt_token_ids"])
                if not 1 <= target_step < fixture["max_tokens"]:
                    raise RuntimeError("Snapshot prefilter is outside actual decode history")
        selected = target_step in common._CONFIG["snapshot_steps"]
        _BATCH = (self, batch, _LAST_EXECUTION_ID, target_step, selected)
        common._event({"event": "late_prepared_batch", "pid": os.getpid(),
                       "rank": common._rank(), "late_execution_id": _LAST_EXECUTION_ID,
                       "req_ids": list(batch.req_ids), "num_reqs": batch.num_reqs,
                       "num_scheduled_tokens": [int(item) for item in batch.num_scheduled_tokens],
                       "actual_target_step": target_step, "capture_selected": selected})
        return batch
    @functools.wraps(execute_original)
    def execute(self, *args, **kwargs):
        global _BATCH, _EXECUTING
        if _EXECUTING:
            raise RuntimeError("Nested model execution is outside the late diagnostic")
        _EXECUTING = True
        _BATCH = None
        try:
            return execute_original(self, *args, **kwargs)
        finally:
            _BATCH = None
            _EXECUTING = False
    runner_cls.prepare_inputs = prepare
    runner_cls.execute_model = execute
    _PATCHED = True

def choose(op, batch_size):
    if _BATCH is None:
        return None
    runner, batch, execution_id, prefilter_step, selected = _BATCH
    if not selected:
        return None
    import torch
    import force_decode as common
    from vllm.forward_context import get_forward_context
    if runner.scheduler_config.async_scheduling or not runner.model_config.enforce_eager:
        raise RuntimeError("Late captures require actual eager synchronous execution")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Late capture cannot run inside CUDA graph capture")
    context = getattr(op, "_gdn_capture_context", None)
    if context is None:
        raise RuntimeError("Missing exact layer identity")
    # Initial multi-token prefill does not invoke ordinary GDN decode; no mixed
    # or resumed prefill is accepted as an observed ordinary decode snapshot.
    if batch.has_prefill:
        return None
    if batch.num_draft_tokens or batch.num_reqs != batch_size or batch.num_tokens != batch_size:
        raise RuntimeError("Ordinary one-token decode row count differs from real input rows")
    if list(batch.num_scheduled_tokens) != [1] * batch_size:
        raise RuntimeError("Actual input row did not schedule one token")
    fc = get_forward_context()
    meta = fc.attn_metadata[context["prefix"]]
    if (meta.num_prefills != 0 or meta.num_spec_decodes != 0 or
            meta.num_actual_tokens != batch_size or meta.num_decodes != batch_size or
            meta.num_decode_tokens != batch_size):
        raise RuntimeError("Layer metadata is not the identical pure-decode input batch")
    mapping = meta.non_spec_token_indx
    if mapping is not None and mapping.cpu().tolist() != list(range(batch_size)):
        raise RuntimeError("Layer token permutation needs an explicit supported mapping")
    page_tensor = (meta.non_spec_flashinfer_state_indices_tensor if op.backend == "flashinfer"
                   else meta.non_spec_state_indices_tensor)
    if page_tensor is None or page_tensor.numel() != batch_size:
        raise RuntimeError("Missing exact per-row GDN page metadata")
    pages = page_tensor.reshape(-1).cpu().tolist()
    slots = [int(value) for value in batch.idx_mapping_np]
    if len(slots) != batch_size or len(set(slots)) != batch_size:
        raise RuntimeError("Invalid persistent request-state mapping")
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=batch.input_ids.device)
    totals = runner.req_states.total_len.gpu[slot_tensor].cpu().tolist()
    seq_lengths = batch.seq_lens.cpu().tolist()
    starts = [int(value) for value in batch.query_start_loc_np]
    input_ids = batch.input_ids[:batch_size].cpu().tolist()
    positions = batch.positions[:batch_size].cpu().tolist()
    rows = []
    for row, (req_id, slot, page) in enumerate(zip(batch.req_ids, slots, pages)):
        identity = common._REQUESTS.get(req_id)
        if identity is None or runner.req_states.req_id_to_index.get(req_id) != slot:
            raise RuntimeError("Real request lacks the verified native forced identity")
        fixture = common._BY_PROMPT[identity["prompt_key"]]
        prompt = fixture["prompt_token_ids"]
        total = int(totals[row])
        step = total - len(prompt)
        if not 1 <= step < fixture["max_tokens"]:
            raise RuntimeError("Actual decode position is outside the committed reference history")
        prefix = fixture["reference_token_ids"][:step]
        actual = runner.req_states.all_token_ids.gpu[slot, :total].cpu().tolist()
        if actual != prompt + prefix:
            raise RuntimeError("Actual input history differs from the frozen common prefix")
        if starts[row + 1] - starts[row] != 1 or starts[row] != row:
            raise RuntimeError("Packed decode row does not match input request row")
        if int(seq_lengths[row]) != total or input_ids[row] != prefix[-1] or positions[row] != total - 1:
            raise RuntimeError("Actual model token/position differs from verified prefix")
        if int(page) <= 0:
            raise RuntimeError("Expected an active GDN page, not a null or padded row")
        rows.append({"row": row, "req_id": req_id, "index": fixture["index"],
                     "request_state_slot": slot, "gdn_state_page": int(page), "step": step,
                     "input_token": input_ids[row], "input_position": positions[row],
                     "prompt_token_ids_sha256": common._digest_tokens(prompt),
                     "input_prefix_sha256": common._digest_tokens(prefix)})
    target_rows = [row for row in rows if row["index"] == common._CONFIG["snapshot_target"]]
    if len(target_rows) != 1:
        return None
    step = target_rows[0]["step"]
    if step != prefilter_step or step not in common._CONFIG["snapshot_steps"]:
        raise RuntimeError("Full snapshot history differs from actual GPU prefilter step")
    context["call_index"] = step
    context["runtime_join"] = {"late_execution_id": execution_id, "target_index": 255,
                               "actual_target_step": step, "rows": rows,
                               "scope": "Actual prepare_inputs row order + per-layer forward metadata page vector; GPU committed histories/input positions verified before this local update."}
    return context
