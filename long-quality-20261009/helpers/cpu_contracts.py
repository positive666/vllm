"""Cheap behavior checks for scoring and immutable token evidence."""
import copy
import unittest

from audit_long_free import validate_example
from long_free_common import digest, flags


class EvidenceContracts(unittest.TestCase):
    def test_length_answer_is_retained_without_becoming_strict_success(self):
        score = flags("Work ends with #### 42", "42", "length")
        self.assertTrue(score["raw_correct"])
        self.assertTrue(score["truncated"])
        self.assertFalse(score["strict_correct"])
        self.assertFalse(flags("The answer is 42", "42", "stop")["strict_correct"])
        self.assertTrue(flags("#### 42", "42", "stop")["strict_correct"])

    def test_changed_token_or_score_evidence_is_rejected(self):
        expected = {"index": 209, "question": "q", "gold": "42", "gold_text": "#### 42",
                    "prompt": "p", "target": True, "prompt_token_ids": [1, 2],
                    "prompt_token_ids_sha256": digest([1, 2])}
        row = {**expected, "token_ids": [3, 4], "token_ids_sha256": digest([3, 4]),
               "text": "#### 42", "finish_reason": "stop",
               "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
               **flags("#### 42", "42", "stop")}
        validate_example(row, expected)
        damaged = copy.deepcopy(row)
        damaged["token_ids"][0] = 9
        with self.assertRaisesRegex(ValueError, "Output token SHA256"):
            validate_example(damaged, expected)
        damaged = copy.deepcopy(row)
        damaged["strict_correct"] = False
        with self.assertRaisesRegex(ValueError, "Recomputed score"):
            validate_example(damaged, expected)


if __name__ == "__main__":
    unittest.main()
