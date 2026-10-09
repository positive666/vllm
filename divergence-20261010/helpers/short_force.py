"""Observe native sampling around a nineteen-token native trace prefix.

The native TraceReplayState performs every substitution. The twentieth choice
is observed unchanged; the driver aborts afterward. Copies synchronize GPU
execution, so this bounded intervention is not an accuracy or timing result.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import threading
import traceback

PREFIX_LEN = 19
LAST_STEP = 19
_TARGET = "vllm.v1.worker.gpu.model_runner"
_CONFIG = None
_OUTPUT = None
_FIXTURES = None
_BY_PROMPT = None
_REQUESTS = {}
_PATCHED = False
_BATCH_INDEX = 0
_LOCK = threading.Lock()


def _digest_tokens(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _json_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _event(value):
    with _LOCK:
        with (_OUTPUT / f"events-pid-{os.getpid()}.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


def _rank():
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else int(os.getenv("RANK", "0"))


def validate_native_result(step, reference, natural, returned):
    """Check the native trace boundary without changing returned token IDs."""
    if type(step) is not int or not 0 <= step <= LAST_STEP:
        raise ValueError("Sampling outside the twenty-token observation window")
    expected = reference[step] if step < PREFIX_LEN else natural
    if returned != expected:
        raise ValueError("Native trace substitution/passthrough contract violated")
    return step < PREFIX_LEN


def validate_history(step, prompt, reference, history):
    """Require the actual committed GPU history to precede this native choice."""
    if type(step) is not int or not 0 <= step <= LAST_STEP:
        raise ValueError("Committed history outside the bounded window")
    if history != prompt + reference[:step]:
        raise ValueError("Actual GPU prompt/output history differs from fixed prefix")


def _guard_runner(runner):
    if (runner.scheduler_config.async_scheduling or runner.speculative_config is not None
            or runner.speculator is not None):
        raise RuntimeError("Bounded observation requires synchronous non-speculative execution")
    if not runner.model_config.enforce_eager or not runner.model_config.enable_trace_replay:
        raise RuntimeError("Actual eager native trace replay is required")
    if not runner.is_last_pp_rank or runner.pcp_manager is not None or runner.batch_sharder is not None:
        raise RuntimeError("This protocol requires unsharded TP2 sampling without PP/PCP")


def _register(runner, new_request):
    req_id = new_request.req_id
    prompt = list(new_request.prompt_token_ids or ())
    fixture = _BY_PROMPT.get(tuple(prompt))
    if fixture is None or req_id in _REQUESTS or any(
        value["index"] == fixture["index"] for value in _REQUESTS.values()
    ):
        raise RuntimeError("Unknown, duplicated or re-registered actual request")
    if list(new_request.prefill_token_ids) != prompt or new_request.mm_features:
        raise RuntimeError("Only unchanged text prompt prefill is supported")
    params = new_request.sampling_params
    if (params.n != 1 or params.temperature != 0 or params.seed != 42
            or params.max_tokens != 3500 or params.min_tokens != 0 or params.ignore_eos
            or params.logprobs is not None or params.prompt_logprobs is not None
            or params.stop or params.logit_bias
            or params.allowed_token_ids or params.bad_words or params.presence_penalty
            or params.frequency_penalty or params.repetition_penalty != 1
            or params.repetition_detection is not None):
        raise RuntimeError("Actual sampling differs from unchanged free-generation parameters")
    if (list(params.stop_token_ids) != [248044] or params._eos_token_id != 248046
            or set(params._all_stop_token_ids) != {248046, 248044}):
        raise RuntimeError("Actual stopping differs from model/tokenizer-derived original EOS IDs")
    prefix = fixture["reference_token_ids"][:PREFIX_LEN]
    if list(params.trace_decode_token_ids or ()) != prefix:
        raise RuntimeError("Actual native trace must contain exactly the fixed nineteen IDs")
    terminal_ids = set(params._all_stop_token_ids)
    if terminal_ids.intersection(prefix):
        raise RuntimeError("An original EOS/stop ID occurs inside the fixed prefix")
    slot = runner.req_states.req_id_to_index[req_id]
    if int(runner.req_states.prompt_len.np[slot]) != len(prompt):
        raise RuntimeError("Registered prompt length differs from frozen actual IDs")
    trace = runner.sampler.trace_replay_state
    if trace is None or int(trace.trace_len.np[slot]) != PREFIX_LEN:
        raise RuntimeError("Actual GPU native trace state has the wrong length")
    if trace.trace_token_ids.gpu[slot, :PREFIX_LEN].cpu().tolist() != prefix:
        raise RuntimeError("Staged native trace IDs differ from fixed prefix")
    _REQUESTS[req_id] = {"index": fixture["index"], "prompt_key": tuple(prompt),
                         "slot": slot, "native_seen_steps": set(),
                         "native_first_difference_on_rank": None}
    _event({"event": "native_request_registered", "pid": os.getpid(), "rank": _rank(),
            "req_id": req_id, "index": fixture["index"], "request_state_slot": slot,
            "prompt_sha256": _digest_tokens(prompt), "native_trace_verified": True,
            "native_trace_len": PREFIX_LEN, "native_ignore_eos": params.ignore_eos,
            "native_max_tokens": params.max_tokens, "native_min_tokens": params.min_tokens,
            "native_eos_token_id": params._eos_token_id,
            "native_stop_token_ids": list(params.stop_token_ids),
            "native_all_stop_token_ids": sorted(terminal_ids)})


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
            raise RuntimeError("Native trace ran more than once per sampler call")
        self.natural_sampled = sampled.detach().clone()
        return self.delegate.apply_trace(sampled, idx_mapping)


class _SamplerObserver:
    def __init__(self, runner, delegate, global_batch):
        self.runner, self.delegate, self.global_batch = runner, delegate, global_batch

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, logits, local_batch):
        global _BATCH_INDEX
        import torch
        import divergence_capture

        runner = self.runner
        _guard_runner(runner)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Observer must execute outside CUDA graphs")
        count = local_batch.num_reqs
        req_ids = list(local_batch.req_ids)
        if (local_batch.num_draft_tokens or logits.ndim != 2
                or logits.shape[0] != count or len(req_ids) != count):
            raise RuntimeError("Expected one raw logit row per actual request")
        if any(int(local_batch.cu_num_logits_np[row + 1]
                   - local_batch.cu_num_logits_np[row]) != 1 for row in range(count)):
            raise RuntimeError("Expanded logit rows are outside this protocol")
        slots = [int(value) for value in local_batch.idx_mapping_np]
        if len(slots) != count or len(set(slots)) != count or min(slots, default=0) < 0:
            raise RuntimeError("Invalid actual local row/request-state mapping")
        tensor_slots = torch.tensor(slots, dtype=torch.long, device=logits.device)
        totals = runner.req_states.total_len.gpu[tensor_slots].cpu().tolist()
        computed = runner.req_states.num_computed_tokens.gpu[tensor_slots].cpu().tolist()
        sequences = local_batch.seq_lens.cpu().tolist()
        global_ids = list(self.global_batch.req_ids)
        global_rows = {req_id: row for row, req_id in enumerate(global_ids)}
        records = []
        for row, (req_id, slot) in enumerate(zip(req_ids, slots)):
            identity = _REQUESTS.get(req_id)
            if identity is None or runner.req_states.req_id_to_index.get(req_id) != slot:
                raise RuntimeError("Unregistered actual sampler request")
            fixture = _BY_PROMPT[identity["prompt_key"]]
            prompt_len = int(runner.req_states.prompt_len.np[slot])
            prefill_len = int(runner.req_states.prefill_len.np[slot])
            if prefill_len != prompt_len or prompt_len != len(fixture["prompt_token_ids"]):
                raise RuntimeError("Unexpected prefill resumption/prompt mutation")
            total = int(totals[row])
            step = total - prompt_len
            history = runner.req_states.all_token_ids.gpu[slot, :total].cpu().tolist()
            validate_history(step, fixture["prompt_token_ids"], fixture["reference_token_ids"], history)
            sequence = int(sequences[row])
            if sequence < prefill_len:
                if step != 0:
                    raise RuntimeError("Post-prefill request returned to partial prefill")
                continue
            if identity["native_seen_steps"] != set(range(step)):
                raise RuntimeError("Repeated or skipped effective sampler step")
            scheduled = int(local_batch.num_scheduled_tokens[row])
            if sequence != prompt_len + step or int(computed[row]) + scheduled != sequence:
                raise RuntimeError("Sample did not follow complete current input execution")
            if step >= 1 and (int(computed[row]) != prompt_len + step - 1 or scheduled != 1):
                raise RuntimeError("Expected exactly one real decode input")
            position = int(local_batch.logits_indices[row].item())
            input_id = int(local_batch.input_ids[position].item())
            input_pos = int(local_batch.positions[position].item())
            expected_id = fixture["prompt_token_ids"][-1] if step == 0 else fixture["reference_token_ids"][step - 1]
            if input_id != expected_id or input_pos != sequence - 1:
                raise RuntimeError("Actual input token/position differs from fixed history")
            records.append({"req_id": req_id, "index": fixture["index"], "row": row,
                            "global_row": global_rows[req_id], "request_state_slot": slot,
                            "step": step, "phase": "prefill_first_token" if step == 0 else "real_decode",
                            "prompt_length": prompt_len, "prefill_length": prefill_len,
                            "total_length_before_sampling": total, "computed_before": int(computed[row]),
                            "sequence_after_execute": sequence, "num_scheduled_tokens": scheduled,
                            "actual_input_token": input_id, "actual_input_position": input_pos,
                            "input_prefix_sha256": _digest_tokens(fixture["reference_token_ids"][:step]),
                            "reference_token_id": fixture["reference_token_ids"][step]})
        raw = logits.detach().float().clone()
        top_values, top_ids = raw.topk(5, dim=-1)
        values, ids = top_values.cpu().tolist(), top_ids.cpu().tolist()
        trace = self.delegate.trace_replay_state
        if trace is None:
            raise RuntimeError("Actual native TraceReplayState is absent")
        observer = _TraceObserver(trace)
        self.delegate.trace_replay_state = observer
        try:
            output = self.delegate(logits, local_batch)
        finally:
            self.delegate.trace_replay_state = trace
        if observer.calls != 1 or observer.natural_sampled is None:
            raise RuntimeError("Original native sampler did not invoke trace exactly once")
        if output.sampled_token_ids.shape != (count, 1) or output.logprobs_tensors is not None:
            raise RuntimeError("Expected single-token native output without API logprobs")
        effective = [record["row"] for record in records]
        if (output.num_sampled.cpu().tolist() != [int(row in effective) for row in range(count)]
                or any(output.num_rejected.cpu().tolist())):
            raise RuntimeError("Native sampling mask differs from effective actual requests")
        natural = observer.natural_sampled.cpu().reshape(-1).tolist()
        returned = output.sampled_token_ids.cpu().reshape(-1).tolist()
        eid = divergence_capture.last_execution_id()
        for record in records:
            row, step = record["row"], record["step"]
            fixture = _BY_PROMPT[_REQUESTS[record["req_id"]]["prompt_key"]]
            forced = validate_native_result(step, fixture["reference_token_ids"], natural[row], returned[row])
            payload = raw[row].contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()
            record.update(original_sampler_token_id=natural[row], native_returned_token_id=returned[row],
                          trace_forced=forced, step19_native_passthrough=(step == 19 and returned[row] == natural[row]),
                          top5_ids=ids[row], top5_raw_logits=values[row], raw_logits_full_sha256=hashlib.sha256(payload).hexdigest(),
                          raw_logits_shape=[raw.shape[-1]], raw_logits_dtype="float32-le",
                          original_sampler_raw_logit=float(raw[row, natural[row]].item()),
                          selected_candidate_raw_logits={str(token): float(raw[row, token].item()) for token in (71072, 43659)})
            if step in (18, 19):
                filename = f"raw-rank-{_rank()}-index-{record['index']}-step-{step:02d}-eid-{eid:06d}.bin"
                with (_OUTPUT / filename).open("xb") as handle:
                    handle.write(payload)
                record["raw_logits_file"] = filename
                record["raw_logits_bytes"] = len(payload)
            identity = _REQUESTS[record["req_id"]]
            identity["native_seen_steps"].add(step)
            differs = natural[row] != record["reference_token_id"]
            if differs and identity["native_first_difference_on_rank"] is None:
                identity["native_first_difference_on_rank"] = step
            record["first_natural_reference_difference_on_rank"] = identity["native_first_difference_on_rank"]
        _BATCH_INDEX += 1
        _event({"event": "sample_batch", "pid": os.getpid(), "rank": _rank(),
                "batch_index": _BATCH_INDEX, "execution_id": eid, "late_execution_id": eid,
                "actual_batch_size": count, "actual_req_ids": req_ids, "effective_rows": effective,
                "global_req_ids": global_ids, "global_row_indices": [global_rows[req_id] for req_id in req_ids],
                "model_batch_size_global": self.global_batch.num_reqs, "sampling_sharded": False,
                "raw_logits_dtype": str(logits.dtype), "records": records,
                "forcing_implementation": "native TraceReplayState: nineteen IDs only",
                "scope": "Bounded prefix intervention; synchronized observation; no quality/performance claim"})
        return output


def patch(module):
    """Patch the actual runner once; the bootstrap owns deferred import order."""
    global _PATCHED
    if module.__name__ != _TARGET:
        return
    if _PATCHED:
        raise RuntimeError("Short sampler observer was installed twice")
    actual_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual_hash != _CONFIG["expected_runner_sha256"]:
        raise RuntimeError("Actual runner source differs from locked source")
    cls = module.GPUModelRunner
    add_original, sample_original = cls.add_requests, cls.sample

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
            raise RuntimeError("Grammar constraints are outside this raw-logit diagnostic")
        delegate = self.sampler
        if delegate is None or isinstance(delegate, _SamplerObserver):
            raise RuntimeError("Absent or recursively wrapped original sampler")
        self.sampler = _SamplerObserver(self, delegate, input_batch)
        try:
            return sample_original(self, hidden_states, input_batch, grammar_output)
        except BaseException:
            _event({"event": "short_failure", "pid": os.getpid(), "traceback": traceback.format_exc()})
            raise
        finally:
            self.sampler = delegate

    cls.add_requests, cls.sample = add_observed, sample_observed
    _PATCHED = True
    related = [Path(module.__file__), Path(module.__file__).parent / "sample/sampler.py",
               Path(module.__file__).parent / "sample/trace_replay.py",
               Path(module.__file__).parent / "input_batch.py", Path(module.__file__).parent / "states.py"]
    _json_new(_OUTPUT / f"patched-pid-{os.getpid()}.json", {
        "module": _TARGET, "module_sha256": actual_hash, "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "related_source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in related},
        "native_prefix_len": PREFIX_LEN, "native_unforced_observation_step": LAST_STEP})


def install():
    """Load immutable inputs; the bootstrap installs deferred module hooks."""
    global _CONFIG, _OUTPUT, _FIXTURES, _BY_PROMPT
    if os.getenv("SHORT_FORCE_ACTIVE") != "1" or _CONFIG is not None:
        return
    raw = Path(os.environ["SHORT_FORCE_CONFIG"]).read_bytes()
    config = json.loads(raw)
    if config["prefix_len"] != PREFIX_LEN or config["observe_last_step"] != LAST_STEP:
        raise ValueError("Exact nineteen-prefix/twentieth-native boundary required")
    fixture_raw = Path(config["fixture_path"]).read_bytes()
    fixtures = json.loads(fixture_raw)["examples"]
    if [row["index"] for row in fixtures] != [198, 206, 209, 228, 255, 285, 292, 318]:
        raise ValueError("Same original eight requests required")
    for row in fixtures:
        for field in ("prompt_token_ids", "reference_token_ids"):
            if not row[field] or any(type(token) is not int or token < 0 for token in row[field]):
                raise ValueError("Invalid actual prompt/reference token IDs")
        if len(row["reference_token_ids"]) <= LAST_STEP or row["max_tokens"] != 3500:
            raise ValueError("Keep original3500 budget and complete nineteen-token prefix")
    by_prompt = {tuple(row["prompt_token_ids"]): row for row in fixtures}
    if len(by_prompt) != 8:
        raise ValueError("Unique frozen token prompts required")
    _CONFIG, _FIXTURES, _BY_PROMPT = config, fixtures, by_prompt
    _OUTPUT = Path(os.environ["SHORT_TRACE_DIR"]).resolve()
    if _OUTPUT != Path(config["output_dir"]).resolve():
        raise ValueError("Trace output/config binding differs")
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    _json_new(_OUTPUT / f"installed-pid-{os.getpid()}.json", {
        "pid": os.getpid(), "config_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_raw).hexdigest(), "config": config,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with (_OUTPUT / f"events-pid-{os.getpid()}.jsonl").open("x", encoding="utf-8"):
        pass
