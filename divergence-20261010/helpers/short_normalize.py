"""Request-scoped diagnostic bypass of native trace termination normalization.

Only the eight bound token prompts retain their already resolved free sampling
parameters. Native TraceReplayState still supplies outputs0..18. This deliberate
unsupported diagnostic mode is not a production fix or quality evaluation.
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

TARGET = "vllm.v1.engine.input_processor"
SOURCE_SHA = "97f66659814177d3f7b72b959f09201cbdd1794de21ec1542fe8970f18a1a2b6"
SAMPLING_SOURCE_SHA = "005eeca282874cde09864133f2100de6ae985439b93e50891194e8447a265999"
PRIMARY_EOS = 248046
GENERATION_EOS = [248046, 248044]
EXTRA_STOPS = [248044]
_INSTALLED = False
_PATCHED = False


def require(value, message):
    if not value:
        raise ValueError(message)


def snapshot(params):
    """Serialize actual resolved fields without importing vLLM or CUDA."""
    names = params.__struct_fields__ if hasattr(params, "__struct_fields__") else tuple(vars(params))
    def convert(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if value is None or type(value) in (int, float, str, bool):
            return value
        return str(value)
    return {name: convert(getattr(params, name)) for name in names}


def resolved_contract(params, fixture, prompt_len):
    require(prompt_len == len(fixture["prompt_token_ids"]), "Actual normalized prompt length differs")
    require(list(params.trace_decode_token_ids or ()) == fixture["reference_token_ids"][:19],
            "Native trace differs from the fixed nineteen-token prefix")
    require(params.max_tokens == 3500 and params.min_tokens == 0
            and params.ignore_eos is False and params.n == 1
            and params.temperature == 0 and params.seed == 42,
            "Original free sampling budget/greedy/EOS contract differs")
    require(not params.stop and list(params.stop_token_ids) == EXTRA_STOPS
            and params._eos_token_id == PRIMARY_EOS
            and set(params._all_stop_token_ids) == set(GENERATION_EOS),
            "Resolved model-derived EOS/stop IDs differ")
    require(not set(GENERATION_EOS).intersection(params.trace_decode_token_ids),
            "Original effective EOS occurs inside the forced prefix")


def wrap_methods(process_original, normalize_original, fixtures, emit):
    """Wrap the real methods with an exact request-local binding."""
    by_external = {str(row["index"]): row for row in fixtures}
    local = threading.local()
    seen = set()

    @functools.wraps(process_original)
    def process(self, request_id, prompt, params, *args, **kwargs):
        if not getattr(params, "trace_decode_token_ids", None):
            return process_original(self, request_id, prompt, params, *args, **kwargs)
        fixture = by_external.get(request_id)
        require(fixture is not None and request_id not in seen, "Unknown or duplicated diagnostic request")
        require(isinstance(prompt, dict) and set(prompt) == {"prompt_token_ids"}
                and list(prompt["prompt_token_ids"]) == fixture["prompt_token_ids"],
                "Actual diagnostic prompt is not the immutable token prompt")
        require(getattr(local, "context", None) is None, "Nested diagnostic input processing")
        require(params.max_tokens == 3500 and params.min_tokens == 0
                and params.ignore_eos is False and not params.stop and not params.stop_token_ids,
                "Original user stopping parameters differ")
        require(list(params.trace_decode_token_ids) == fixture["reference_token_ids"][:19],
                "Unbound native trace prefix")
        context = {"request_id": request_id, "fixture": fixture,
                   "original_params": params, "normalizations": 0}
        local.context = context
        try:
            result = process_original(self, request_id, prompt, params, *args, **kwargs)
            require(context["normalizations"] == 1, "Exact request-local normalization hook was not reached once")
            require(list(result.prompt_token_ids) == fixture["prompt_token_ids"], "Processed token prompt changed")
            resolved_contract(result.sampling_params, fixture, len(fixture["prompt_token_ids"]))
            seen.add(request_id)
            return result
        finally:
            local.context = None

    @functools.wraps(normalize_original)
    def normalize(self, params, prompt_len):
        if not getattr(params, "trace_decode_token_ids", None):
            return normalize_original(self, params, prompt_len)
        context = getattr(local, "context", None)
        require(context is not None, "Trace normalization lacks an exact bound request")
        require(params is not context["original_params"], "Original request-local cloning was bypassed")
        require(context["normalizations"] == 0, "Repeated diagnostic normalization")
        require(self.model_config.enable_trace_replay is True
                and self.model_config.max_model_len == 4096,
                "Actual native trace/context configuration differs")
        require(self.renderer.get_eos_token_id() == PRIMARY_EOS
                and self.generation_config_fields.get("eos_token_id") == GENERATION_EOS,
                "Actual tokenizer/generation EOS metadata differs")
        resolved_contract(params, context["fixture"], prompt_len)
        require(prompt_len + params.max_tokens <= self.model_config.max_model_len
                and prompt_len + 19 <= self.model_config.max_model_len,
                "Original output budget/prefix exceeds actual context")
        before = snapshot(params)
        context["normalizations"] += 1
        emit({"event": "short_trace_normalization_bypassed", "pid": os.getpid(),
              "external_req_id": context["request_id"], "index": context["fixture"]["index"],
              "prompt_length": prompt_len, "prompt_token_ids": context["fixture"]["prompt_token_ids"],
              "resolved_sampling_params": before, "unchanged_after": snapshot(params) == before,
              "input_processor_sha256": SOURCE_SHA,
              "intervention": "Skip only native trace termination normalization after generation/tokenizer updates; native trace nineteen IDs remains active",
              "scope": "Unsupported bounded diagnostic mode; not production behavior or accuracy evidence"})
        return None

    return process, normalize


def patch(module):
    global _PATCHED
    require(not _PATCHED and module.__name__ == TARGET, "Duplicate or wrong normalization module")
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA,
            "Actual InputProcessor source differs from pinned d8")
    import short_force
    require(short_force._CONFIG["source_files"]["vllm/v1/engine/input_processor.py"] == SOURCE_SHA,
            "InputProcessor source is not bound by the launch config")
    sampling = sys.modules.get("vllm.sampling_params")
    require(sampling is not None and hashlib.sha256(Path(sampling.__file__).read_bytes()).hexdigest() == SAMPLING_SOURCE_SHA
            and short_force._CONFIG["source_files"]["vllm/sampling_params.py"] == SAMPLING_SOURCE_SHA,
            "Actual SamplingParams generation/EOS source differs from pinned d8")
    cls = module.InputProcessor
    cls.process_inputs, cls._normalize_trace_replay_params = wrap_methods(
        cls.process_inputs, cls._normalize_trace_replay_params,
        short_force._FIXTURES, short_force._event)
    _PATCHED = True
    short_force._event({"event": "short_normalization_source", "pid": os.getpid(),
                        "module": TARGET, "source_sha256": SOURCE_SHA,
                        "sampling_source_sha256": SAMPLING_SOURCE_SHA,
                        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate
    def create_module(self, spec):
        method = getattr(self.delegate, "create_module", None)
        return method(spec) if method else None
    def exec_module(self, module):
        self.delegate.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        require(spec is not None and spec.loader is not None, "Missing actual InputProcessor module")
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    """Install before importing vLLM; ordinary untraced inputs keep native behavior."""
    global _INSTALLED
    if _INSTALLED:
        return
    require(os.getenv("SHORT_FORCE_ACTIVE") == "1", "Diagnostic activation missing")
    require(TARGET not in sys.modules, "Install before actual InputProcessor import")
    sys.meta_path.insert(0, _Finder())
    _INSTALLED = True
