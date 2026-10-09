"""Two CPU consumer contracts; synthetic records never become experiment evidence.

Purpose: validate native-history provenance without running vLLM. I/O: frozen
JSON traces produce physical owner records and explicit pair gates. Failure:
fabricating TP-rank observations, accepting wrong GPU input, or hiding changed
prefill/schedules. Cheapest level: stdlib temporary trace fixtures and consumer
functions, with no model, production edits or GPU execution.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import audit_late_sampler as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_trace(folder):
    root, source = folder / "run", folder / "source"
    trace = root / "trace"
    trace.mkdir(parents=True)
    related = {}
    for name in ("model_runner.py", "sample/sampler.py", "sample/trace_replay.py",
                 "input_batch.py", "states.py"):
        path = source / "vllm/v1/worker/gpu" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source contract: " + name, encoding="utf-8")
        related["/source/vllm/v1/worker/gpu/" + name] = audit.sha(path)
    fixtures = {index: {"index": index, "prompt_token_ids": [11 + row],
                        "reference_token_ids": [21 + row * 2, 22 + row * 2],
                        "max_tokens": 2}
                for row, index in enumerate((198, 255))}
    config = {"expected_runner_sha256": related["/source/vllm/v1/worker/gpu/model_runner.py"]}
    launch = {"force_config_sha256": "a" * 64, "fixture_sha256": "b" * 64,
              "helper_sha256": {"force_decode_v2.py": "c" * 64,
                                "force_decode.py": "d" * 64}}
    ids = ["req198", "req255"]
    for rank in (0, 1):
        pid = 100 + rank
        write(trace / f"installed-pid-{pid}.json",
              {"pid": pid, "event": "installed", "config_sha256": launch["force_config_sha256"],
               "fixture_sha256": launch["fixture_sha256"], "config": config,
               "planned_indices": sorted(fixtures), "target": "vllm.v1.worker.gpu.model_runner"})
        write(trace / f"patched-pid-{pid}.json",
              {"pid": pid, "event": "patched", "module_sha256": config["expected_runner_sha256"],
               "helper_sha256": "c" * 64, "common_helper_sha256": "d" * 64,
               "runner_relative_path": "vllm/v1/worker/gpu/model_runner.py",
               "module_file": "/source/vllm/v1/worker/gpu/model_runner.py",
               "module": "vllm.v1.worker.gpu.model_runner", "forcing_implementation": audit.NATIVE,
               "related_source_sha256": related})
        events = []
        base = {"pid": pid, "rank": rank}
        for row, (index, fixture) in enumerate(fixtures.items()):
            events.extend([
                {**base, "event": "request_registered", "req_id": ids[row], "index": index,
                 "prompt_tokens": 1, "prompt_sha256": audit.token_sha(fixture["prompt_token_ids"]),
                 "reference_sha256": audit.token_sha(fixture["reference_token_ids"]),
                 "max_tokens": 2, "ignore_eos": True},
                {**base, "event": "native_request_registered", "req_id": ids[row], "index": index,
                 "native_trace_verified": True, "request_state_slot": rank * 10 + row}])
        index = (198, 255)[rank]
        fixture = fixtures[index]
        for step in (0, 1):
            forced = fixture["reference_token_ids"][step]
            record = {
                "row": 0, "req_id": ids[rank], "index": index, "step": step,
                "global_row": rank, "recorded_sampling_rank": rank,
                "request_state_slot": rank * 10 + rank, "max_tokens": 2,
                "output_length_before": step, "forced_token_id": forced,
                "native_forced_token_id": forced, "prompt_length": 1, "prefill_length": 1,
                "input_prefix_sha256": audit.token_sha(fixture["reference_token_ids"][:step]),
                "total_length_before_sampling": 1 + step, "sequence_after_execute": 1 + step,
                "phase": "prefill_first_token" if step == 0 else "real_decode",
                "computed_before": step, "num_scheduled_tokens": 1,
                "actual_input_token": fixture["prompt_token_ids"][-1] if step == 0 else
                    fixture["reference_token_ids"][step - 1],
                "actual_input_position": step, "target": index == 255,
                "raw_logits_full_sha256": "e" * 64 if step == 0 else None,
                "eos_observation_scope": audit.EOS_SCOPE, "eos_candidate_ids": [248046],
                "eos_candidate_raw_logits": [0.0], "top5_ids": [forced, 501, 502, 503, 504],
                "top5_raw_logits": [5.0, 4.0, 3.0, 2.0, 1.0], "raw_logsumexp": 6.0,
                "forced_raw_logit": 5.0, "natural_raw_top1": forced,
                "original_sampler_token_id": forced, "raw_top_max": 5.0,
                "original_sampler_raw_logit": 5.0, "original_sampler_raw_gap_from_top": 0.0,
                "natural_sampler_differs_from_raw_top1": False,
                "natural_differs_from_reference": False,
                "first_natural_reference_difference_on_rank": None}
            events.extend([
                {**base, "event": "late_prepared_batch", "late_execution_id": step + 1,
                 "req_ids": ids, "num_reqs": 2, "num_scheduled_tokens": [1, 1],
                 "actual_target_step": None if step == 0 else step,
                 "capture_selected": step == 1},
                {**base, "event": "sample_batch", "late_execution_id": step + 1,
                 "batch_index": step + 1, "actual_req_ids": [ids[rank]], "actual_batch_size": 1,
                 "global_req_ids": ids, "global_row_indices": [rank], "model_batch_size_global": 2,
                 "sampler_batch_size_local": 1, "sampling_sharded": True,
                 "forcing_implementation": audit.NATIVE, "effective_rows": [0],
                 "discarded_rows": [], "records": [record]}])
        (trace / f"events-pid-{pid}.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return root, launch, config, fixtures, source


class SamplerConsumerContracts(unittest.TestCase):
    def test_physical_shards_join_exact_gpu_inputs_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root, launch, config, fixtures, source = make_trace(Path(temp))
            result = audit.audit_trace(root, launch, config, fixtures, source)
            records, _, physical, prepared, sampled, _, missing = result
            self.assertEqual(len(records), 4)
            self.assertEqual(len(physical), 4)
            self.assertEqual(records[198, 1]["ranks"], [0])
            self.assertEqual(records[255, 1]["ranks"], [1])
            self.assertNotIn((0, 2, 255, 1), physical)
            self.assertEqual(set(prepared), set(sampled))
            self.assertEqual(missing, [])
            path = root / "trace/events-pid-101.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[-1]["records"][0]["actual_input_token"] += 1
            path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Actual input/target identity"):
                audit.audit_trace(root, launch, config, fixtures, source)

    def test_changed_full_prefill_or_schedule_remains_explicit_nogo(self):
        with tempfile.TemporaryDirectory() as temp:
            root, launch, config, fixtures, source = make_trace(Path(temp))
            records, schedule, *_ = audit.audit_trace(root, launch, config, fixtures, source)
            paired_records = {"triton": records, "flashinfer": copy.deepcopy(records)}
            paired_schedules = {"triton": schedule, "flashinfer": copy.deepcopy(schedule)}
            baseline = audit.compare_pair(paired_records, paired_schedules, fixtures)
            self.assertTrue(baseline["native_starting_condition_gate_pass"])
            paired_records["flashinfer"][255, 0]["record"]["raw_logits_full_sha256"] = "f" * 64
            changed = audit.compare_pair(paired_records, paired_schedules, fixtures)
            self.assertFalse(changed["full_prefill_logits_equal"])
            self.assertFalse(changed["native_starting_condition_gate_pass"])
            self.assertEqual([item["index"] for item in changed["prefill"]
                              if not item["exact_equal"]], [255])
            paired_schedules["flashinfer"][1][-1]["global_row_indices"] = [0]
            changed = audit.compare_pair(paired_records, paired_schedules, fixtures)
            self.assertFalse(changed["schedules_equal"])
            self.assertFalse(changed["native_starting_condition_gate_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
