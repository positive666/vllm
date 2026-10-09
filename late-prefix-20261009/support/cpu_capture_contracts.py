"""CPU contracts for actual-step selection and fail-closed capture evidence.

The selection observer must read the actual GPU position once per execution;
48 nonselected layer calls must not repeat a history/page join. The result
consumer rejects hash, sampler-execution, prefix and coverage tampering.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import late_capture
from audit_late_snapshots import (assert_complete, expected_keys, validate_join,
                                 verify_capture_file, observed_pair)
from late_plan import digest


class ActualStepSelection(unittest.TestCase):
    def test_unselected_layers_share_one_actual_gpu_step_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input_batch.py").write_bytes(b"frozen input source")
            counter = []
            events = []
            class Value:
                def cpu(self):
                    return self
                def item(self):
                    return 5
            class GPU:
                def __getitem__(self, key):
                    counter.append(key)
                    return Value()
            batch = types.SimpleNamespace(has_prefill=False, req_ids=["r"],
                                          num_reqs=1, num_scheduled_tokens=[1],
                                          idx_mapping_np=[0],
                                          input_ids=types.SimpleNamespace(device="cuda:0"))
            class Runner:
                req_states = types.SimpleNamespace(total_len=types.SimpleNamespace(gpu=GPU()),
                                                   req_id_to_index={"r": 0})
                def prepare_inputs(self):
                    return batch
                def execute_model(self):
                    self.prepare_inputs()
                    for _ in range(48):
                        self.assert_unselected()
                    return "native execution returned"
                def assert_unselected(self):
                    if late_capture.choose(object(), 1) is not None:
                        raise AssertionError("Unselected layer captured")
            fake_common = types.SimpleNamespace(
                _CONFIG={"expected_input_batch_sha256": hashlib.sha256(
                    (root / "input_batch.py").read_bytes()).hexdigest(),
                    "snapshot_target": 255, "snapshot_steps": [1, 18, 19]},
                _REQUESTS={"r": {"prompt_key": "p"}},
                _BY_PROMPT={"p": {"index": 255, "prompt_token_ids": [1, 2, 3],
                                   "max_tokens": 3500}},
                _event=events.append, _rank=lambda: 0)
            fake_torch = types.SimpleNamespace(long="long", tensor=lambda *a, **k: [0])
            fake_module = types.SimpleNamespace(__file__=str(root / "model_runner.py"),
                                                GPUModelRunner=Runner)
            with patch.dict(sys.modules, {"force_decode": fake_common, "torch": fake_torch}):
                late_capture._PATCHED = False
                late_capture._LAST_EXECUTION_ID = 0
                late_capture._EXECUTING = False
                late_capture._BATCH = None
                late_capture.patch_runner(fake_module)
                self.assertEqual(Runner().execute_model(), "native execution returned")
            self.assertEqual(len(counter), 1)
            self.assertEqual(events[0]["actual_target_step"], 2)
            self.assertIs(events[0]["capture_selected"], False)
            self.assertIsNone(late_capture._BATCH)
            self.assertFalse(late_capture._EXECUTING)
            late_capture._PATCHED = False


class CaptureEvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.fixtures = {255: {"prompt_token_ids": [13], "reference_token_ids": [21, 22, 23],
                              "max_tokens": 3}}
        self.row = {"row": 0, "req_id": "r", "index": 255, "request_state_slot": 7,
                    "gdn_state_page": 11, "step": 1, "input_token": 21, "input_position": 1,
                    "prompt_token_ids_sha256": digest([13]), "input_prefix_sha256": digest([21])}
        self.meta = {"rank": 0, "call_index": 1, "batch_size": 1,
                     "runtime_join": {"late_execution_id": 9, "target_index": 255,
                                      "actual_target_step": 1, "rows": [self.row]}}
        self.prepared = {(0, 9): {"capture_selected": True, "actual_target_step": 1,
                                 "req_ids": ["r"], "num_reqs": 1, "num_scheduled_tokens": [1]}}
        record = {"index": 255, "step": 1, "req_id": "r", "phase": "real_decode",
                  "global_row": 0, "input_prefix_sha256": digest([21]),
                  "actual_input_token": 21, "actual_input_position": 1,
                  "recorded_sampling_rank": 0, "request_state_slot": 7}
        self.sampled = {(0, 9): {"global_req_ids": ["r"], "model_batch_size_global": 1,
                                "records": [record]}}

    def test_exact_snapshot_execution_history_join_is_accepted(self):
        validate_join(self.meta, self.prepared, self.sampled, self.fixtures)

    def test_empty_local_shard_uses_only_actual_owner_and_retains_missing_context(self):
        meta = copy.deepcopy(self.meta)
        meta["rank"] = 1
        prepared = {(1, 9): self.prepared[(0, 9)]}
        self.assertFalse(validate_join(meta, prepared, self.sampled, self.fixtures))
        self.assertNotIn((1, 9), self.sampled)

    def test_changed_execution_prefix_or_actual_page_is_rejected(self):
        for field, value in (("input_prefix_sha256", "bad"), ("input_token", 22),
                             ("gdn_state_page", 0)):
            with self.subTest(field=field):
                meta = copy.deepcopy(self.meta)
                meta["runtime_join"]["rows"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_join(meta, self.prepared, self.sampled, self.fixtures)
        meta = copy.deepcopy(self.meta)
        meta["runtime_join"]["late_execution_id"] = 10
        with self.assertRaises(KeyError):
            validate_join(meta, self.prepared, self.sampled, self.fixtures)

    def test_missing_or_extra_rank_layer_step_is_rejected(self):
        keys = expected_keys()
        self.assertEqual(len(keys), 672)
        assert_complete(keys)
        for changed in (keys - {(0, 0, 1)}, keys | {(2, 0, 1)}):
            with self.assertRaises(ValueError):
                assert_complete(changed)

    def test_changed_binary_is_rejected_even_when_sidecar_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "capture.pt"
            path.write_bytes(b"immutable snapshot bytes")
            event = {"file": "capture.pt", "bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metadata": {}}
            path.with_suffix(".json").write_text(json.dumps(event), encoding="utf-8")
            self.assertEqual(verify_capture_file(root, event), path)
            path.write_bytes(b"X" + path.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, "Exact captured binary SHA"):
                verify_capture_file(root, event)


class InitialObservedStateContracts(unittest.TestCase):
    def observed(self, changed_step):
        import torch
        indices = (198, 206, 209, 228, 255, 285, 292, 318)
        indexed = {backend: {key: (backend, key) for key in expected_keys()}
                   for backend in ("triton", "flashinfer")}
        def fake_load(entry):
            backend, key = entry
            requests = indices if key[2] in (1, 18, 19) else (255,)
            state = torch.zeros((len(requests) + 2, 1, 2, 2), dtype=torch.float32)
            if backend == "flashinfer" and key == (0, 0, changed_step):
                state[1, 0, 0, 0] = 1e-6
            joined = [{"index": index, "step": key[2], "input_prefix_sha256": "frozen",
                       "input_token": 17, "input_position": key[2], "gdn_state_page": row + 11}
                      for row, index in enumerate(requests)]
            metadata = {"runtime_join": {"rows": joined},
                        "compact_to_original": [0, *range(11, 11 + len(requests)), 99]}
            return {"metadata": metadata,
                    "tensors": {"initial_state": state, "production_post_state": state.clone(),
                                "mixed_qkv": torch.zeros((len(requests), 4)),
                                "a": torch.zeros((len(requests), 1)),
                                "b": torch.zeros((len(requests), 1)),
                                "A_log": torch.zeros(1), "dt_bias": torch.zeros(1),
                                "production_output": torch.zeros((len(requests), 1, 1, 2))}}
        with patch("audit_late_snapshots.load_snapshot", side_effect=fake_load):
            return observed_pair(indexed)

    def test_tiny_first_decode_difference_rejects_common_observed_start(self):
        result = self.observed(1)
        self.assertEqual(len(result["first_decode_active_states"]), 768)
        self.assertFalse(result["common_observed_initial_state"])
        self.assertEqual(sum(not row["prestate"]["exact_equal"]
                             for row in result["first_decode_active_states"]), 1)

    def test_later_state_difference_is_retained_without_inventing_initial_mismatch(self):
        result = self.observed(1750)
        self.assertTrue(result["common_observed_initial_state"])
        self.assertTrue(any(not row["prestate"]["exact_equal"]
                            for row in result["later_observed_states"]))


if __name__ == "__main__":
    unittest.main()
