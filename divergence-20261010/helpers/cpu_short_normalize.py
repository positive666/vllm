"""CPU contracts for the pinned real normalization/generation-config methods.

Purpose: keep original resolved stopping in a request-bound prefix diagnostic.
I/O: actual-method AST + cloned parameters become an unchanged resolved request.
Failure: native max19/EOS mutation leaks through, or unrelated trace bypasses it.
Cheapest level: stdlib AST extraction, source SHA checks and request-method wrapper;
no vLLM import, model construction, CUDA or fabricated accuracy result.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import short_normalize as helper

SAMPLING_SHA = helper.SAMPLING_SOURCE_SHA


def extract(path, expected, class_name, names):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, "Actual pinned source bytes"
    tree = ast.parse(raw.decode())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == set(names)
    node = ast.ClassDef(name="Extracted", bases=[], keywords=[], decorator_list=[],
                        body=[ast.Assign(targets=[ast.Name(id="__slots__", ctx=ast.Store())],
                                         value=ast.Tuple(elts=[], ctx=ast.Load()))] + methods)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Extracted"]


def rejection(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Expected diagnostic binding rejection")


def run_contracts(processor_source, sampling_source):
    ActualProcessor = extract(processor_source, helper.SOURCE_SHA, "InputProcessor",
                              {"_normalize_trace_replay_params"})
    ActualSampling = extract(sampling_source, SAMPLING_SHA, "SamplingParams",
                             {"update_from_generation_config", "update_from_tokenizer"})
    fields = ("n", "temperature", "seed", "max_tokens", "min_tokens", "ignore_eos", "stop",
              "stop_token_ids", "_eos_token_id", "_all_stop_token_ids", "trace_decode_token_ids", "bad_words")
    class Params(ActualSampling):
        __slots__ = fields
        __struct_fields__ = fields
        def __init__(self):
            self.n, self.temperature, self.seed = 1, 0, 42
            self.max_tokens, self.min_tokens, self.ignore_eos = 3500, 0, False
            self.stop, self.stop_token_ids, self.bad_words = [], [], []
            self._eos_token_id, self._all_stop_token_ids = None, set()
            self.trace_decode_token_ids = list(range(100, 119))
        def clone(self):
            return copy.deepcopy(self)
    fixture = {"index": 255, "prompt_token_ids": [1, 2, 3], "reference_token_ids": list(range(100, 130))}
    class Processor(ActualProcessor):
        def __init__(self):
            self.model_config = SimpleNamespace(max_model_len=4096, enable_trace_replay=True)
            self.renderer = SimpleNamespace(get_eos_token_id=lambda: 248046)
            self.generation_config_fields = {"eos_token_id": [248046, 248044]}
        def process_inputs(self, request_id, prompt, params):
            resolved = params.clone()
            resolved.update_from_generation_config(self.generation_config_fields, self.renderer.get_eos_token_id())
            resolved.update_from_tokenizer(None)
            self._normalize_trace_replay_params(resolved, len(prompt["prompt_token_ids"]))
            return SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], sampling_params=resolved)
    original = Params()
    resolved = original.clone()
    resolved.update_from_generation_config({"eos_token_id": [248046, 248044]}, 248046)
    assert resolved.stop_token_ids == [248044] and resolved._all_stop_token_ids == {248044, 248046}
    helper.resolved_contract(resolved, fixture, 3)
    ActualProcessor._normalize_trace_replay_params(Processor(), resolved, 3)
    assert resolved.max_tokens == 19 and resolved.ignore_eos and resolved._eos_token_id is None
    assert resolved.stop_token_ids == [] and resolved._all_stop_token_ids == set()

    emitted = []
    process, normalize = helper.wrap_methods(Processor.process_inputs, ActualProcessor._normalize_trace_replay_params,
                                             [fixture], emitted.append)
    Processor.process_inputs, Processor._normalize_trace_replay_params = process, normalize
    result = Processor().process_inputs("255", {"prompt_token_ids": [1, 2, 3]}, Params())
    helper.resolved_contract(result.sampling_params, fixture, 3)
    assert result.sampling_params.max_tokens == 3500 and len(result.sampling_params.trace_decode_token_ids) == 19
    assert emitted[0]["unchanged_after"] and emitted[0]["resolved_sampling_params"]["_eos_token_id"] == 248046
    assert original._eos_token_id is None and not original.stop_token_ids
    assert not hasattr(result.sampling_params, "__dict__"), "Exercise struct snapshot without vars()"

    rejection(Processor().process_inputs, "999", {"prompt_token_ids": [1, 2, 3]}, Params())
    rejected = Params()
    rejected.trace_decode_token_ids[-1] = 999
    fresh, fresh_normalize = helper.wrap_methods(process.__wrapped__, ActualProcessor._normalize_trace_replay_params,
                                                 [fixture], emitted.append)
    class Fresh(Processor):
        process_inputs, _normalize_trace_replay_params = fresh, fresh_normalize
    rejection(Fresh().process_inputs, "255", {"prompt_token_ids": [1, 2, 4]}, Params())
    rejection(Fresh().process_inputs, "255", {"prompt_token_ids": [1, 2, 3]}, rejected)
    rejected = Params()
    rejected.stop_token_ids = [999]
    rejection(Fresh().process_inputs, "255", {"prompt_token_ids": [1, 2, 3]}, rejected)
    rejected = Params()
    rejected.update_from_generation_config({"eos_token_id": [248046, 248044]}, 248046)
    rejection(Fresh()._normalize_trace_replay_params, rejected, 3)
    rejected._all_stop_token_ids.add(999)
    rejection(helper.resolved_contract, rejected, fixture, 3)
    assert len(emitted) == 1, "Rejected inputs never receive a bypass event"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-processor-source", type=Path, required=True)
    parser.add_argument("--sampling-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_contracts(args.input_processor_source, args.sampling_source)
    folder = Path(__file__).resolve().parent
    receipt = {"status": "CPU-contracts-passed", "gpu_execution": False,
               "source_sha256": {"input_processor": helper.SOURCE_SHA, "sampling_params": SAMPLING_SHA},
               "contracts": ["Actual native method demonstrates max19/EOS mutation",
                             "Actual generation-config EOS update + scoped bypass preserve cloned resolved fields",
                             "Unknown prompt/prefix/user-stop/unbound context/extra effective stop fail closed"],
               "helper_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                 for name in ("short_normalize.py", "cpu_short_normalize.py")}}
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
