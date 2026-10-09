"""CPU contracts for frozen natural-history evidence before GPU use.

The plan maps retained unforced output IDs into a shared-history diagnostic.
Its contract is exact prompt/history identity and a bounded target truncation.
A tampered retained output must fail before native forcing can mask that error.
"""
from __future__ import annotations
import shutil
import tempfile
import unittest
from pathlib import Path
from late_plan import validate_inputs

ROOT = Path(__file__).resolve().parent.parent

class FrozenHistoryContracts(unittest.TestCase):
    def test_exact_retained_history_preserves_long_truncation(self):
        lock, rows = validate_inputs(ROOT)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["max_tokens"] for row in rows), 7378)
        self.assertEqual(next(row for row in rows if row["index"] == 255)["max_tokens"], 3500)
        self.assertIn(3499, lock["selected_steps"])

    def test_modified_retained_output_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory(prefix="gdn-late-contract-") as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "reference", root / "reference")
            shutil.copy2(ROOT / "reference-lock.json", root / "reference-lock.json")
            path = root / "reference/completed.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Frozen reference SHA: completed.json"):
                validate_inputs(root)

if __name__ == "__main__":
    unittest.main()
