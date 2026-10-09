"""Observe native V2 trace replay without replacing its forcing implementation.

The V2 sampler applies TraceReplayState before returning to gather/post_update.
This helper records the actual sampler result immediately before native replay,
then verifies native replay's result and the committed GPU token history. The
observation copies synchronize and perturb scheduling; no accuracy/performance
claim follows. Configuration/fixtures and deferred import infrastructure are
shared with force_decode.py; the installed target is the actual V2 runner.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
import os
from types import SimpleNamespace
import traceback

import force_decode as common

_TARGET = "vllm.v1.worker.gpu.model_runner"


def _guard_runner(runner):
    if runner.scheduler_config.async_scheduling or runner.speculative_config is not None or runner.speculator is not None:
        raise RuntimeError("V2 trace observation requires synchronous non-speculative scheduling")
    if not runner.model_config.enforce_eager or not runner.model_config.enable_trace_replay:
        raise RuntimeError("V2 observation requires eager execution and native trace replay")
    if not runner.is_last_pp_rank or runner.pcp_manager is not None:
        raise RuntimeError("V2 observation supports only the last PP stage without PCP")


def _register(runner, new_request):
    req_id = new_request.req_id
    if req_id in common._REQUESTS:
        raise RuntimeError("Streaming/preempted re-registration is outside this bounded run")
    prompt = new_request.prompt_token_ids
    if prompt is None or list(new_request.prefill_token_ids) != list(prompt):
        raise RuntimeError("V2 initial prefill must contain only the unchanged actual prompt")
    state = SimpleNamespace(prompt_token_ids=list(prompt),
                            mm_features=new_request.mm_features,
                            prompt_embeds=getattr(new_request, "prompt_embeds", None),
                            sampling_params=new_request.sampling_params)
    _, fixture, identity = common._resolve_request(
        SimpleNamespace(requests={req_id: state}), req_id
    )
    planned = fixture["reference_token_ids"][:fixture["max_tokens"]]
    if list(new_request.sampling_params.trace_decode_token_ids or ()) != planned:
        raise RuntimeError("Actual native trace differs from the frozen fixture")
    slot = runner.req_states.req_id_to_index[req_id]
    if int(runner.req_states.prompt_len.np[slot]) != len(prompt):
        raise RuntimeError("V2 registered prompt length differs from actual frozen IDs")
    identity["native_seen_steps"] = set()
    identity["native_first_difference_on_rank"] = None
    identity["slot"] = slot
    if runner.sampler is not None:
        trace = runner.sampler.trace_replay_state
        if trace is None or int(trace.trace_len.np[slot]) != fixture["max_tokens"]:
            raise RuntimeError("Native V2 trace state was not installed for the actual request")
        actual_trace = trace.trace_token_ids.gpu[slot, :fixture["max_tokens"]].cpu().tolist()
        if actual_trace != planned:
            raise RuntimeError("Staged native trace values differ from the frozen reference")
    common._event({"event": "native_request_registered", "pid": os.getpid(),
                   "rank": common._rank(), "req_id": req_id, "index": fixture["index"],
                   "request_state_slot": slot, "native_trace_verified": True,
                   "native_eos_token_id": new_request.sampling_params._eos_token_id,
                   "native_all_stop_token_ids": sorted(new_request.sampling_params._all_stop_token_ids),
                   "native_stop_token_ids": list(new_request.sampling_params.stop_token_ids or []),
                   "native_ignore_eos": new_request.sampling_params.ignore_eos,
                   "phase_contract": "GPU total_len minus original prompt_len counts committed output tokens"})


class _TraceObserver:
    def __init__(self, delegate):
        self.delegate = delegate
        self.natural_sampled = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def apply_trace(self, sampled, idx_mapping):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Native trace replay ran more than once for this sampler call")
        self.natural_sampled = sampled.detach().clone()
        return self.delegate.apply_trace(sampled, idx_mapping)


class _SamplerObserver:
    def __init__(self, runner, delegate, global_batch):
        self.runner = runner
        self.delegate = delegate
        self.global_batch = global_batch

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, logits, local_batch):
        import torch

        runner = self.runner
        _guard_runner(runner)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V2 observation requires execution outside CUDA graphs")
        batch_size = local_batch.num_reqs
        req_ids = list(local_batch.req_ids)
        if (local_batch.num_draft_tokens or logits.ndim != 2 or
                logits.shape[0] != batch_size or len(req_ids) != batch_size):
            raise RuntimeError("V2 observation expects one logit row per real local request")
        if any(int(local_batch.cu_num_logits_np[row + 1] - local_batch.cu_num_logits_np[row]) != 1
               for row in range(batch_size)):
            raise RuntimeError("V2 expanded logit rows are outside the single-token plan")
        slots = [int(slot) for slot in local_batch.idx_mapping_np]
        if len(slots) != batch_size or len(set(slots)) != batch_size or any(slot < 0 for slot in slots):
            raise RuntimeError("Invalid V2 local row-to-request-state mapping")
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=logits.device)
        total_lengths = runner.req_states.total_len.gpu[slot_tensor].cpu().tolist()
        computed_counts = runner.req_states.num_computed_tokens.gpu[slot_tensor].cpu().tolist()
        seq_lengths = local_batch.seq_lens.cpu().tolist()
        global_req_ids = list(self.global_batch.req_ids)
        global_row_by_id = {req_id: row for row, req_id in enumerate(global_req_ids)}
        local_to_global = [global_row_by_id[req_id] for req_id in req_ids]
        records = []
        effective_rows = []
        for row, (req_id, slot) in enumerate(zip(req_ids, slots)):
            identity = common._REQUESTS.get(req_id)
            if identity is None or runner.req_states.req_id_to_index.get(req_id) != slot:
                raise RuntimeError("Unregistered request or inconsistent real V2 row mapping")
            fixture = common._BY_PROMPT[identity["prompt_key"]]
            prompt_length = int(runner.req_states.prompt_len.np[slot])
            prefill_length = int(runner.req_states.prefill_len.np[slot])
            if prefill_length != prompt_length or prompt_length != len(fixture["prompt_token_ids"]):
                raise RuntimeError("V2 prefill resumption or prompt mutation is outside this run")
            total = int(total_lengths[row])
            step = total - prompt_length
            if step < 0 or step > fixture["max_tokens"]:
                raise RuntimeError("V2 actual committed output count is outside the planned history")
            actual_history = runner.req_states.all_token_ids.gpu[slot, :total].cpu().tolist()
            reference_prefix = fixture["reference_token_ids"][:step]
            if actual_history != fixture["prompt_token_ids"] + reference_prefix:
                raise RuntimeError("Actual committed GPU prompt/output history differs from the forced prefix")
            if int(seq_lengths[row]) < prefill_length:
                if step != 0:
                    raise RuntimeError("A post-prefill request unexpectedly returned to partial prefill")
                continue
            if step >= fixture["max_tokens"] or step in identity["native_seen_steps"]:
                raise RuntimeError("Repeated or out-of-plan V2 effective sampling step")
            computed = int(computed_counts[row])
            scheduled = int(local_batch.num_scheduled_tokens[row])
            sequence_after = int(seq_lengths[row])
            if sequence_after != prompt_length + step or computed + scheduled != sequence_after:
                raise RuntimeError("V2 sample does not follow actual complete prompt/current-token execution")
            if step >= 1 and (computed != prompt_length + step - 1 or scheduled != 1):
                raise RuntimeError("V2 post-prefill step did not execute exactly one real decode token")
            logit_position = int(local_batch.logits_indices[row].item())
            actual_input_token = int(local_batch.input_ids[logit_position].item())
            actual_input_position = int(local_batch.positions[logit_position].item())
            expected_input = fixture["prompt_token_ids"][-1] if step == 0 else reference_prefix[-1]
            if actual_input_token != expected_input or actual_input_position != sequence_after - 1:
                raise RuntimeError("Real V2 model input token/position differs from the planned prefix")
            forced_id = fixture["reference_token_ids"][step]
            if forced_id < 0 or forced_id >= logits.shape[-1]:
                raise RuntimeError("Native forced token is outside the actual logit vocabulary")
            effective_rows.append(row)
            records.append({"req_id": req_id, "index": fixture["index"], "row": row,
                            "global_row": local_to_global[row], "recorded_sampling_rank": common._rank(),
                            "request_state_slot": slot, "step": step,
                            "phase": "prefill_first_token" if step == 0 else "real_decode",
                            "phase_contract": "GPU committed total_len-prompt_len; seq_len from actual model input",
                            "prompt_length": prompt_length, "prefill_length": prefill_length,
                            "total_length_before_sampling": total, "computed_before": computed,
                            "sequence_after_execute": sequence_after, "num_scheduled_tokens": scheduled,
                            "actual_input_token": actual_input_token, "actual_input_position": actual_input_position,
                            "output_length_before": step, "max_tokens": fixture["max_tokens"],
                            "forced_token_id": forced_id, "input_prefix_sha256": common._digest_tokens(reference_prefix),
                            "target": fixture["index"] in common._CONFIG["target_indices"]})

        raw = logits.detach().float().clone()
        top_values, top_ids = raw.topk(min(5, raw.shape[-1]), dim=-1)
        observed = {"top5_ids": top_ids.cpu().tolist(), "top5_raw_logits": top_values.cpu().tolist(),
                    "natural_raw_top1": raw.argmax(dim=-1).cpu().tolist(),
                    "raw_logsumexp": torch.logsumexp(raw, dim=-1).cpu().tolist()}
        forced_values = {record["row"]: float(raw[record["row"], record["forced_token_id"]].item())
                         for record in records}
        native_trace = self.delegate.trace_replay_state
        if native_trace is None:
            raise RuntimeError("Native TraceReplayState is absent from the actual sampler")
        trace_observer = _TraceObserver(native_trace)
        self.delegate.trace_replay_state = trace_observer
        try:
            output = self.delegate(logits, local_batch)
        finally:
            self.delegate.trace_replay_state = native_trace
        if trace_observer.calls != 1 or trace_observer.natural_sampled is None:
            raise RuntimeError("The native sampler did not execute actual trace replay")
        if output.sampled_token_ids.shape != (batch_size, 1) or output.logprobs_tensors is not None:
            raise RuntimeError("Expected actual V2 single-token output without API logprobs")
        sampled_counts = output.num_sampled.cpu().tolist()
        rejected_counts = output.num_rejected.cpu().tolist()
        if sampled_counts != [int(row in effective_rows) for row in range(batch_size)] or any(rejected_counts):
            raise RuntimeError("Native num_sampled/prefill mask differs from the observed effective rows")
        natural_sampled = trace_observer.natural_sampled.cpu().reshape(-1).tolist()
        actual_forced = output.sampled_token_ids.cpu().reshape(-1).tolist()
        for record in records:
            row = record["row"]
            if actual_forced[row] != record["forced_token_id"]:
                raise RuntimeError("Native V2 forcing did not return the planned real token")
            for name, values in observed.items():
                record[name] = values[row]
            record["forced_raw_logit"] = forced_values[row]
            record["original_sampler_token_id"] = natural_sampled[row]
            record["original_sampler_raw_logit"] = float(raw[row, natural_sampled[row]].item())
            record["raw_top_max"] = observed["top5_raw_logits"][row][0]
            record["original_sampler_raw_gap_from_top"] = (
                record["raw_top_max"] - record["original_sampler_raw_logit"]
            )
            record["native_forced_token_id"] = actual_forced[row]
            record["raw_logits_full_sha256"] = (hashlib.sha256(raw[row].contiguous().cpu().numpy().tobytes()).hexdigest() if record["step"] == 0 else None)
            record["eos_observation_scope"] = common._CONFIG["eos_observation_scope"]
            record["eos_candidate_ids"] = common._CONFIG["eos_candidate_ids"]
            record["eos_candidate_raw_logits"] = [float(raw[row, token].item()) for token in common._CONFIG["eos_candidate_ids"]]
            record["natural_sampler_differs_from_raw_top1"] = natural_sampled[row] != record["natural_raw_top1"]
            differs = natural_sampled[row] != record["forced_token_id"]
            identity = common._REQUESTS[record["req_id"]]
            identity["native_seen_steps"].add(record["step"])
            if differs and identity["native_first_difference_on_rank"] is None:
                identity["native_first_difference_on_rank"] = record["step"]
            record["natural_differs_from_reference"] = differs
            record["first_natural_reference_difference_on_rank"] = identity["native_first_difference_on_rank"]
        common._BATCH_INDEX += 1
        common._event({"event": "sample_batch", "pid": os.getpid(), "rank": common._rank(),
                       "batch_index": common._BATCH_INDEX, "late_execution_id": __import__("late_capture").last_execution_id(),
                       "actual_batch_size": batch_size,
                       "actual_req_ids": req_ids, "effective_rows": effective_rows,
                       "discarded_rows": [row for row in range(batch_size) if row not in effective_rows],
                       "model_batch_size_global": self.global_batch.num_reqs,
                       "global_req_ids": global_req_ids, "global_row_indices": local_to_global,
                       "sampler_batch_size_local": batch_size,
                       "sampling_sharded": runner.batch_sharder is not None,
                       "raw_logits_dtype": str(logits.dtype), "records": records,
                       "forcing_implementation": "native V2 TraceReplayState.apply_trace",
                       "scope": "synchronized shared-prefix observation; rank-local sampling shards; no quality/performance claim"})
        return output


def _patch(module):
    if common._PATCHED:
        raise RuntimeError("V2 observer patch installed twice")
    module_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if module_hash != common._CONFIG["expected_runner_sha256"]:
        raise RuntimeError("Actual V2 runner source differs from the configured file hash")
    runner_class = module.GPUModelRunner
    add_original = runner_class.add_requests
    sample_original = runner_class.sample

    @functools.wraps(add_original)
    def add_observed(self, scheduler_output):
        _guard_runner(self)
        result = add_original(self, scheduler_output)
        for request in scheduler_output.scheduled_new_reqs:
            _register(self, request)
        return result

    @functools.wraps(sample_original)
    def sample_observed(self, hidden_states, input_batch, grammar_output):
        _guard_runner(self)
        if grammar_output is not None:
            raise RuntimeError("Grammar-constrained logits are outside the raw diagnostic")
        original_sampler = self.sampler
        if original_sampler is None or isinstance(original_sampler, _SamplerObserver):
            raise RuntimeError("Actual native V2 sampler is absent or recursively wrapped")
        self.sampler = _SamplerObserver(self, original_sampler, input_batch)
        try:
            return sample_original(self, hidden_states, input_batch, grammar_output)
        except BaseException:
            common._event({"event": "force_failure", "pid": os.getpid(),
                           "traceback": traceback.format_exc()})
            raise
        finally:
            self.sampler = original_sampler

    runner_class.add_requests = add_observed
    runner_class.sample = sample_observed
    common._PATCHED = True
    related = [Path(module.__file__), Path(module.__file__).parent / "sample/sampler.py",
               Path(module.__file__).parent / "sample/trace_replay.py",
               Path(module.__file__).parent / "input_batch.py", Path(module.__file__).parent / "states.py"]
    common._json_new(common._OUTPUT / f"patched-pid-{os.getpid()}.json",
                     {"event": "patched", "pid": os.getpid(), "module": _TARGET,
                      "runner_relative_path": "vllm/v1/worker/gpu/model_runner.py",
                      "module_file": module.__file__, "module_sha256": module_hash,
                      "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "common_helper_sha256": hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
                      "related_source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in related},
                      "forcing_implementation": "native V2 TraceReplayState.apply_trace"})


def install():
    """Install actual V2 hooks; source/fixture checks remain fail closed."""
    if os.environ.get("GDN_FORCE_ACTIVE") != "1":
        return
    if common._CONFIG is not None:
        if common._TARGET != _TARGET:
            raise RuntimeError("A different model runner observer was already installed")
        return
    common._TARGET = _TARGET
    common._patch = _patch
    common.install()
