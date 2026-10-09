"""Reject injected unknown GPU PIDs and retain possible exit races."""
import copy
import unittest

from audit_host_ownership import classify_samples, parse_samples


class OwnershipContract(unittest.TestCase):
    def test_unknown_pid_is_not_promoted_to_owned_or_hidden_as_a_race(self):
        text = (
            "2026-10-09T09:00:00Z\n"
            "GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e, 10, 100 MiB\nPID\n1\n10\n"
            "2026-10-09T09:00:05Z\nPID\n1\n"
        )
        samples = parse_samples(text)
        self.assertTrue(classify_samples(samples)["strict_same_sample_ownership_pass"])
        damaged = copy.deepcopy(samples)
        damaged[0]["gpu_compute"][0]["pid"] = 99999
        result = classify_samples(damaged)
        self.assertFalse(result["strict_same_sample_ownership_pass"])
        self.assertEqual(result["unknown_ownership_pids"], [99999])
        self.assertEqual(result["possible_adjacent_sample_race_observations"], 0)
        exiting = copy.deepcopy(samples)
        exiting[1]["gpu_compute"] = copy.deepcopy(exiting[0]["gpu_compute"])
        result = classify_samples(exiting)
        self.assertFalse(result["strict_same_sample_ownership_pass"])
        self.assertEqual(result["unknown_ownership_pids"], [])
        self.assertEqual(result["possible_adjacent_sample_race_observations"], 1)


if __name__ == "__main__":
    unittest.main()
