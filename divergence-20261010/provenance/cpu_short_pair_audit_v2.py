"""Actual metadata contract for the bounded consumer source-schema correction."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import short_pair_audit_v2 as audit


ROOT = Path(__file__).parent
METADATA = ROOT / "results-v6/metadata/results"
GPU_UUIDS = [
    "GPU-4af84168-6d29-6128-fee3-1b146cce9b47",
    "GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e",
]
ATTENTION = "vllm.model_executor.layers.attention.attention"


class ActualSourceContracts(unittest.TestCase):
    def test_actual_three_arm_source_schema_binds_import_pid_to_native_rank(self):
        for arm in audit.ARMS:
            root = METADATA / ("short-" + arm)
            launch, cfg = audit.read(root / "launch.json"), audit.read(root / "short-config.json")
            self.assertNotIn("vllm/model_executor/layers/attention/attention.py",
                             launch["source_files"])
            result = audit.validate_source_bindings(
                launch, cfg, audit.events(root), root / "helpers/divergence_capture.py"
            )
            self.assertEqual(len(result), 6)
            self.assertEqual({row["registered_tp_rank"] for row in result}, {0, 1})
            row = next(row for row in result
                       if row["registered_tp_rank"] == 1 and row["module"] == ATTENTION)
            self.assertEqual(row["rank_at_import"], 0)
            self.assertEqual(row["sha256"], audit.SOURCE_SHAS[
                "vllm/model_executor/layers/attention/attention.py"])

    def test_actual_source_evidence_missing_tampered_duplicate_or_wrong_pid_fails(self):
        root = METADATA / "short-A"
        launch, cfg = audit.read(root / "launch.json"), audit.read(root / "short-config.json")
        stream = audit.events(root)
        helper = root / "helpers/divergence_capture.py"
        worker_pid = next(row["pid"] for row in stream
                          if row.get("event") == "native_request_registered" and row["rank"] == 1)
        target = next(i for i, row in enumerate(stream)
                      if row.get("event") == "short_capture_source"
                      and row["module"] == ATTENTION and row["pid"] == worker_pid)
        mutations = []
        missing = copy.deepcopy(stream)
        del missing[target]
        mutations.append(missing)
        for field, value in (("sha256", "0" * 64), ("pid", 999999), ("arm", "C")):
            changed = copy.deepcopy(stream)
            changed[target][field] = value
            mutations.append(changed)
        duplicate = copy.deepcopy(stream)
        duplicate.append(copy.deepcopy(duplicate[target]))
        mutations.append(duplicate)
        for changed in mutations:
            with self.assertRaises(ValueError):
                audit.validate_source_bindings(launch, cfg, changed, helper)
        changed_launch = copy.deepcopy(launch)
        changed_launch["source_files"]["vllm/v1/worker/gpu/model_runner.py"] = "0" * 64
        with self.assertRaises(ValueError):
            audit.validate_source_bindings(changed_launch, cfg, stream, helper)
        with tempfile.TemporaryDirectory() as directory:
            changed_helper = Path(directory) / "divergence_capture.py"
            changed_helper.write_bytes(helper.read_bytes() + b"\n# contract mutation\n")
            with self.assertRaises(ValueError):
                audit.validate_source_bindings(launch, cfg, stream, changed_helper)

    def test_actual_metadata_index_advances_to_absent_raw_binary_without_mocking(self):
        # This local archive intentionally omits PT/bin data. Reaching this
        # precise gate proves the earlier real provenance/sampling checks pass;
        # it does not count as a complete numerical result or arm acceptance.
        for arm in audit.ARMS:
            with self.assertRaisesRegex(ValueError,
                                        "Raw full-vocabulary binary receipt mismatch"):
                audit.index_arm(METADATA / ("short-" + arm), arm, GPU_UUIDS)


if __name__ == "__main__":
    unittest.main()
