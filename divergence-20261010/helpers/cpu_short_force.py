"""Two stdlib contracts: trace cutoff and immediate abort at output20.

Purpose: bound a prefix intervention, preserving the native twentieth choice.
I/O: GPU-history/native-choice checks and engine-step/abort orchestration.
Failure: trace accidentally overrides output20 or an output21 executes.
Cheapest check: pure CPU boundary cases and a fake public engine, no vLLM/GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from short_force import patch, validate_history, validate_native_result
from short_driver import enqueue_requests, observe_twenty


def rejects(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Expected bounded-contract rejection")


def prefix_contract():
    reference = list(range(100, 130))
    prompt = [1, 2, 3]
    assert validate_native_result(18, reference, 999, reference[18]) is True
    assert validate_native_result(19, reference, 71072, 71072) is False
    rejects(validate_native_result, 19, reference, 71072, reference[19])
    rejects(validate_native_result, 20, reference, 71072, 71072)
    validate_history(19, prompt, reference, prompt + reference[:19])
    rejects(validate_history, 19, prompt, reference, prompt + reference[:18])
    rejects(validate_history, 19, prompt, reference, prompt + reference[:18] + [999])


def abort_contract():
    class Engine:
        def __init__(self):
            self.steps = 0
            self.aborts = []
        def step(self):
            self.steps += 1
            assert self.steps <= 20
            return self.steps
        def abort_request(self, ids):
            self.aborts.append((self.steps, ids))
    ids = [str(value) for value in range(8)]
    engine = Engine()
    consumed = []
    observe_twenty(engine, ids, lambda count, output: consumed.append((count, output)))
    assert consumed == [(value, value) for value in range(1, 21)]
    assert engine.aborts == [(20, ids)]
    engine = Engine()
    def failed(count, output):
        if count == 19:
            raise ValueError("Preserved diagnostic failure")
    rejects(observe_twenty, engine, ids, failed)
    assert engine.steps == 19 and engine.aborts == [(19, ids)]


def identity_contract():
    class Engine:
        def __init__(self):
            self.external = []
        def add_request(self, external_id, prompt, params):
            self.external.append(external_id)
            return external_id + "-internal"
    engine = Engine()
    examples = [{"index": index, "prompt_token_ids": [index]}
                for index in (198, 255)]
    ids, by_external, internal = enqueue_requests(engine, examples, lambda row: object())
    assert ids == engine.external == ["198", "255"]
    assert by_external["255"] is examples[1]
    assert internal == {"198": "198-internal", "255": "255-internal"}
    assert set(by_external).isdisjoint(internal.values())


def import_scope_contract():
    from types import SimpleNamespace
    for name in ("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
                 "vllm.model_executor.layers.attention.attention"):
        module = SimpleNamespace(__name__=name)
        assert patch(module) is None
        assert vars(module) == {"__name__": name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix_contract()
    abort_contract()
    identity_contract()
    import_scope_contract()
    folder = Path(__file__).resolve().parent
    receipt = {"status": "CPU-contracts-passed", "contracts": 4, "gpu_execution": False,
               "checks": ["native prefix18 forced /19 passthrough and GPU history tamper",
                          "exact20 public steps then abort; failure abort before21",
                          "public output/abort IDs differ from randomized internal IDs",
                          "runner sampler patch ignores model-only module hooks"],
               "helper_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                 for name in ("short_force.py", "short_driver.py", "cpu_short_force.py")}}
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
