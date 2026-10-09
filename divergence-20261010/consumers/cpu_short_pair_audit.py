"""Consumer contracts: strict initial gate and honest typed numerical metrics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import short_pair_audit as audit


class ConsumerContracts(unittest.TestCase):
    def test_effective_eos_struct_cannot_hide_native_trace_termination(self):
        original = {
            "trace_decode_token_ids": list(range(19)),
            "max_tokens": 3500, "min_tokens": 0, "ignore_eos": False,
            "_eos_token_id": None, "_all_stop_token_ids": [],
            "stop_token_ids": [], "watermarking": None, "temperature": 0,
            "output_kind": "CUMULATIVE",
        }
        resolved = {
            **original, "_eos_token_id": 248046,
            "_all_stop_token_ids": [248044, 248046],
            "stop_token_ids": [248044], "watermarking": False,
            "output_kind": "RequestOutputKind.CUMULATIVE",
        }
        audit.validate_resolved_sampling(original, resolved, list(range(19)))
        for changed in (
            {**resolved, "max_tokens": 19},
            {**resolved, "ignore_eos": True},
            {**resolved, "_all_stop_token_ids": []},
            {**resolved, "temperature": 1},
            {**resolved, "output_kind": "RequestOutputKind.DELTA"},
        ):
            with self.assertRaises(ValueError):
                audit.validate_resolved_sampling(original, changed, list(range(19)))

    def test_reference_and_fixture_are_bound_to_immutable_source(self):
        original = Path(__file__).parent.parent / "late-prefix-v2-20261009"
        lock = audit.read(original / "reference-lock.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            shutil.copytree(original / "reference", root)
            reference = audit.validate_reference(root, lock)
            fixtures = {
                row["index"]: {
                    "prompt_token_ids": row["prompt_token_ids"],
                    "reference_token_ids": row["token_ids"],
                    "max_tokens": 3500,
                }
                for row in reference["examples"]
            }
            audit.validate_fixture(fixtures, reference)
            fixtures[255]["reference_token_ids"] = [999] + fixtures[255]["reference_token_ids"][1:]
            with self.assertRaises(ValueError):
                audit.validate_fixture(fixtures, reference)
            path = root / "launch.json"
            value = json.loads(path.read_text(encoding="utf8"))
            value["model_options"]["seed"] = 123
            path.write_text(json.dumps(value), encoding="utf8")
            with self.assertRaises(ValueError):
                audit.validate_reference(root, lock)

    def test_observed_mismatch_is_retained_without_tolerance(self):
        template = {
            "initial": {
                ("kv", 0, 3, 255): {
                    "value": {"sha256": "same", "shape": [4, 2, 512],
                              "dtype": "torch.bfloat16"}
                }
            },
            "records": {
                (rank, index, 0): {"raw_logits_full_sha256": "same"}
                for rank in (0, 1) for index in audit.INDICES
            },
            "schedule": {"actual": [1, 18, 19]},
        }
        arms = {arm: copy.deepcopy(template) for arm in audit.ARMS}
        self.assertTrue(audit.observed_initial_comparison(arms)["gate"])
        arms["C"]["initial"][("kv", 0, 3, 255)]["value"]["sha256"] = "different"
        result = audit.observed_initial_comparison(arms)
        self.assertFalse(result["gate"])
        self.assertEqual(len(result["initial_differences"]), 2)
        self.assertEqual(result["initial_differences"][0]["second"]["sha256"],
                         "different")
        arms = {arm: copy.deepcopy(template) for arm in audit.ARMS}
        arms["B"]["records"][(1, 255, 0)]["raw_logits_full_sha256"] = "different"
        result = audit.observed_initial_comparison(arms)
        self.assertFalse(result["gate"])
        self.assertEqual(len(result["prefill_differences"]), 2)
        arms = {arm: copy.deepcopy(template) for arm in audit.ARMS}
        arms["C"]["schedule"] = {"actual": [1, 19, 18]}
        self.assertFalse(audit.observed_initial_comparison(arms)["gate"])

    def test_dtype_value_equality_is_separate_from_numerical_drift(self):
        import torch
        first = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        second = first.float()
        result = audit.tensor_metrics(first, second, torch)
        self.assertTrue(result["value_equal_fp32"])
        self.assertFalse(result["same_dtype"])
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["relative_l2_to_second"], 0.0)
        result = audit.tensor_metrics(torch.ones(2), torch.zeros(2), torch)
        self.assertTrue(result["zero_denominator"])
        self.assertIsNone(result["relative_l2_to_second"])
        with self.assertRaises(ValueError):
            audit.tensor_metrics(torch.tensor([float("nan")]), torch.ones(1), torch)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
