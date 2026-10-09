"""Force planned output tokens after real model execution, before bookkeeping.

This experiment keeps the original prompt as prefill, then advances actual
KV/SSM state with each forced token. GPU/CPU observations perturb scheduling;
the resulting logits diagnose a shared prefix, not uninstrumented accuracy or
performance. Enable GDN_FORCE_ACTIVE=1 and set GDN_FORCE_CONFIG. Installation
is deferred until GPUModelRunner is imported, including in spawned workers.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import traceback

_TARGET = "vllm.v1.worker.gpu_model_runner"
_LOCK = threading.Lock()
_CONFIG = None
_OUTPUT = None
_FIXTURES = None
_BY_PROMPT = None
_REQUESTS = {}
_PATCHED = False
_BATCH_INDEX = 0


def _digest_tokens(tokens):
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _fail_closed():
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    os._exit(97)


def _rank():
    import torch

    return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(
        os.environ.get("RANK", "0")
    )


def _resolve_request(runner, req_id):
    state = runner.requests[req_id]
    prompt = state.prompt_token_ids
    if prompt is None or state.mm_features or state.prompt_embeds is not None:
        raise RuntimeError("Teacher force supports only text token prompts")
    prompt_key = tuple(prompt)
    fixture = _BY_PROMPT.get(prompt_key)
    if fixture is None:
        raise RuntimeError(f"Unknown actual request prompt: {req_id}")
    identity = _REQUESTS.get(req_id)
    if identity is None:
        if any(item["index"] == fixture["index"] for item in _REQUESTS.values()):
            raise RuntimeError("Fixture was associated with more than one real request")
        params = state.sampling_params
        if params is None or not params.ignore_eos or params.n != 1:
            raise RuntimeError("Teacher force requires ignore_eos=True and n=1")
        if params.temperature != 0 or params.logprobs is not None or params.prompt_logprobs is not None:
            raise RuntimeError("Teacher force requires greedy sampling without API logprobs")
        if params.max_tokens != fixture["max_tokens"]:
            raise RuntimeError("Actual max_tokens differs from the frozen fixture plan")
        if params.stop or params.min_tokens or params.logit_bias or params.allowed_token_ids:
            raise RuntimeError("Unexpected stop strings or sampling logits constraints")
        if params.bad_words or params.presence_penalty or params.frequency_penalty or params.repetition_penalty != 1:
            raise RuntimeError("Unexpected sampling penalties or bad words")
        if params.repetition_detection is not None:
            raise RuntimeError("Repetition stopping is incompatible with the frozen plan")
        # Stored EOS/stop IDs can be legal even when ignore_eos is true. Reject
        # only a planned termination before the last forced token. The driver
        # must audit observed output token IDs and lengths afterward.
        terminal_ids = set(params.stop_token_ids or ())
        if params.eos_token_id is not None:
            terminal_ids.add(params.eos_token_id)
        early_tokens = fixture["reference_token_ids"][:fixture["max_tokens"] - 1]
        if terminal_ids.intersection(early_tokens):
            raise RuntimeError("An effective EOS/stop token could terminate the plan early")
        identity = {"index": fixture["index"], "prompt_key": prompt_key,
                    "next_step": 0, "first_natural_reference_difference": None}
        _REQUESTS[req_id] = identity
        _event({"event": "request_registered", "pid": os.getpid(), "rank": _rank(),
                "req_id": req_id, "index": fixture["index"],
                "prompt_tokens": len(prompt), "prompt_sha256": _digest_tokens(prompt),
                "reference_sha256": _digest_tokens(fixture["reference_token_ids"]),
                "max_tokens": fixture["max_tokens"], "ignore_eos": params.ignore_eos,
                "effective_eos_token_id": params.eos_token_id,
                "effective_stop_token_ids": list(params.stop_token_ids or ())})
    elif identity["prompt_key"] != prompt_key:
        raise RuntimeError("An existing request changed its actual token prompt")
    return state, fixture, identity


def _sample_forced(original, runner, logits, spec_decode_metadata):
    global _BATCH_INDEX
    import torch

    if runner.use_async_scheduling or runner.speculative_config is not None or spec_decode_metadata is not None:
        raise RuntimeError("Teacher force requires synchronous non-speculative scheduling")
    if not runner.model_config.enforce_eager or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Teacher force requires real eager model execution")
    req_ids = list(runner.input_batch.req_ids)
    batch_size = runner.input_batch.num_reqs
    if len(req_ids) != batch_size or logits is None or logits.ndim != 2 or logits.shape[0] != batch_size:
        raise RuntimeError("Logits rows must exactly match the current real request batch")
    records = []
    forced_ids = []
    valid_rows = []
    for row, req_id in enumerate(req_ids):
        if runner.input_batch.req_id_to_index.get(req_id) != row:
            raise RuntimeError("Request-to-row mapping is inconsistent")
        state, fixture, identity = _resolve_request(runner, req_id)
        if bool(runner.discard_request_mask.np[row]):
            continue
        step = len(state.output_token_ids)
        if step != identity["next_step"] or step >= fixture["max_tokens"]:
            raise RuntimeError("Repeated, skipped, or out-of-plan effective output step")
        reference_prefix = fixture["reference_token_ids"][:step]
        if state.output_token_ids != reference_prefix:
            raise RuntimeError("Cached request output history differs from the forced prefix")
        prompt_length = state.num_prompt_tokens
        input_history = runner.input_batch.token_ids_cpu[row, prompt_length:prompt_length + step].tolist()
        if input_history != reference_prefix:
            raise RuntimeError("Actual synchronous next-input history differs from the forced prefix")
        computed = int(runner.input_batch.num_computed_tokens_cpu[row])
        sequence_after = int(runner.optimistic_seq_lens_cpu[row])
        expected_sequence_after = prompt_length + step
        if sequence_after != expected_sequence_after:
            raise RuntimeError("Effective sample does not follow complete prompt/current token execution")
        if step >= 1 and computed != prompt_length + step - 1:
            raise RuntimeError("A post-prefill step did not execute exactly one real decode token")
        forced_id = fixture["reference_token_ids"][step]
        if forced_id < 0 or forced_id >= logits.shape[-1]:
            raise RuntimeError("Forced token is outside the actual logit vocabulary")
        records.append({"req_id": req_id, "index": fixture["index"], "row": row,
                        "step": step, "phase": "prefill_first_token" if step == 0 else "real_decode",
                        "prompt_length": prompt_length, "computed_before": computed,
                        "sequence_after_execute": sequence_after, "output_length_before": step,
                        "max_tokens": fixture["max_tokens"], "forced_token_id": forced_id,
                        "input_prefix_sha256": _digest_tokens(reference_prefix),
                        "target": fixture["index"] in _CONFIG["target_indices"]})
        valid_rows.append(row)
        forced_ids.append(forced_id)

    # Observe raw logits before the original sampler may mutate a float32 tensor.
    if valid_rows:
        row_tensor = torch.tensor(valid_rows, dtype=torch.long, device=logits.device)
        raw = logits.index_select(0, row_tensor).float()
        top_values, top_ids = raw.topk(min(5, raw.shape[-1]), dim=-1)
        natural_top1 = raw.argmax(dim=-1)
        logsumexp = torch.logsumexp(raw, dim=-1)
        force_tensor = torch.tensor(forced_ids, dtype=torch.long, device=logits.device)
        forced_values = raw.gather(1, force_tensor.unsqueeze(1)).squeeze(1)
        observed = {
            "top5_ids": top_ids.cpu().tolist(), "top5_raw_logits": top_values.cpu().tolist(),
            "natural_raw_top1": natural_top1.cpu().tolist(),
            "raw_logsumexp": logsumexp.cpu().tolist(),
            "forced_raw_logit": forced_values.cpu().tolist(),
        }
    output = original(runner, logits, spec_decode_metadata)
    if output.sampled_token_ids.shape != (batch_size, 1) or output.logprobs_tensors is not None:
        raise RuntimeError("Expected single-token sampling without returned logprobs")
    if valid_rows:
        sampled_original = output.sampled_token_ids[row_tensor, 0].cpu().tolist()
        for i, record in enumerate(records):
            for name, values in observed.items():
                record[name] = values[i]
            record["original_sampler_token_id"] = sampled_original[i]
            if sampled_original[i] != record["natural_raw_top1"]:
                raise RuntimeError("Actual greedy sampler differs from observed raw argmax")
            identity = _REQUESTS[record["req_id"]]
            differs = sampled_original[i] != record["forced_token_id"]
            if differs and identity["first_natural_reference_difference"] is None:
                identity["first_natural_reference_difference"] = record["step"]
            record["natural_differs_from_reference"] = differs
            record["first_natural_reference_difference"] = identity["first_natural_reference_difference"]
            identity["next_step"] += 1
        output.sampled_token_ids[row_tensor, 0] = force_tensor.to(output.sampled_token_ids.dtype)
    _BATCH_INDEX += 1
    _event({"event": "sample_batch", "pid": os.getpid(), "rank": _rank(),
            "batch_index": _BATCH_INDEX, "actual_batch_size": batch_size,
            "actual_req_ids": req_ids, "effective_rows": valid_rows,
            "discarded_rows": [row for row in range(batch_size) if row not in valid_rows],
            "raw_logits_dtype": str(logits.dtype), "records": records,
            "scope": "shared forced prefix; synchronization perturbs scheduling; no performance or quality claim"})
    return output


def _patch(module):
    global _PATCHED
    if _PATCHED:
        raise RuntimeError("Teacher-force patch installed twice")
    module_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if module_hash != _CONFIG["expected_runner_sha256"]:
        raise RuntimeError("GPUModelRunner source differs from the configured file hash")
    original = module.GPUModelRunner._sample

    @functools.wraps(original)
    def forced(self, logits, spec_decode_metadata):
        try:
            return _sample_forced(original, self, logits, spec_decode_metadata)
        except BaseException:
            _event({"event": "force_failure", "pid": os.getpid(),
                    "traceback": traceback.format_exc()})
            raise

    module.GPUModelRunner._sample = forced
    _PATCHED = True
    _json_new(_OUTPUT / f"patched-pid-{os.getpid()}.json",
              {"event": "patched", "pid": os.getpid(), "module": _TARGET,
               "module_file": module.__file__, "module_sha256": module_hash,
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


class _ForceLoader(importlib.abc.Loader):
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


class _ForceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot locate teacher-force target {fullname}")
        spec.loader = _ForceLoader(spec.loader)
        return spec


def install():
    """Enable the import hook exactly once; repeated driver install is harmless."""
    global _CONFIG, _OUTPUT, _FIXTURES, _BY_PROMPT
    if os.environ.get("GDN_FORCE_ACTIVE") != "1":
        return
    if os.environ.get("GDN_CAPTURE_ACTIVE") == "1":
        raise RuntimeError("Capture and teacher-force experiments must be separate")
    if _CONFIG is not None:
        return
    config_bytes = Path(os.environ["GDN_FORCE_CONFIG"]).read_bytes()
    config = json.loads(config_bytes)
    for key in ("output_dir", "fixture_path", "target_indices", "limit", "expected_runner_sha256"):
        if key not in config:
            raise ValueError(f"Missing teacher-force configuration: {key}")
    if not isinstance(config["limit"], int) or config["limit"] < 1:
        raise ValueError("Invalid teacher-force limit")
    fixture_bytes = Path(config["fixture_path"]).read_bytes()
    examples = json.loads(fixture_bytes)["examples"]
    by_prompt = {}
    indices = set()
    for example in examples:
        for key in ("index", "prompt_token_ids", "reference_token_ids", "max_tokens"):
            if key not in example:
                raise ValueError(f"Missing fixture field: {key}")
        for key in ("prompt_token_ids", "reference_token_ids"):
            if not isinstance(example[key], list) or not example[key] or any(
                not isinstance(token, int) or token < 0 for token in example[key]
            ):
                raise ValueError(f"Invalid fixture token list: {key}")
        planned = min(len(example["reference_token_ids"]), config["limit"])
        if example["max_tokens"] != planned:
            raise ValueError("Fixture max_tokens must equal the bounded reference length")
        key = tuple(example["prompt_token_ids"])
        if key in by_prompt or example["index"] in indices:
            raise ValueError("Duplicate actual prompt or fixture index")
        by_prompt[key] = example
        indices.add(example["index"])
    if len(examples) != 8 or not set(config["target_indices"]).issubset(indices):
        raise ValueError("Teacher force requires eight unique planned fixture requests")
    _CONFIG, _FIXTURES, _BY_PROMPT = config, examples, by_prompt
    _OUTPUT = Path(config["output_dir"]).resolve()
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    # Exclusive creation binds this PID to a new run before append-only records.
    _json_new(_OUTPUT / f"installed-pid-{os.getpid()}.json",
              {"event": "installed", "pid": os.getpid(), "target": _TARGET,
               "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
               "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
               "config": config, "planned_indices": sorted(indices)})
    with (_OUTPUT / f"events-pid-{os.getpid()}.jsonl").open("x", encoding="utf-8"):
        pass
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _ForceFinder())


if __name__ == "__main__":
    raise SystemExit("Install before importing LLM; run the separate bounded driver")
