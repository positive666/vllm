"""Audit paired real-decode teacher-force traces using only the standard library.

This audits a bounded shared-prefix observation, not accuracy or performance.
No logits outside the recorded top five and forced token are reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import traceback
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_sha(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def validate_token_list(value):
    require(
        isinstance(value, list)
        and value
        and all(type(token) is int and token >= 0 for token in value),
        "Token list contract",
    )


def file_manifest(directory):
    return {
        str(path.relative_to(directory)): {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def frozen_source_path(recorded_path, source_root):
    """Map the frozen container path to its independently checked source root."""
    recorded = PurePosixPath(recorded_path)
    require(
        recorded.is_absolute() and ".." not in recorded.parts,
        "Safe recorded absolute source path",
    )
    relative = recorded.relative_to(PurePosixPath("/source"))
    require(relative.parts and relative.parts[0] == "vllm", "Frozen vLLM source path")
    return source_root.joinpath(*relative.parts)


def natural_observation(record):
    ids, values = record["top5_ids"], record["top5_raw_logits"]
    require(
        len(ids) == len(values) == 5 and len(set(ids)) == 5, "Five unique candidates"
    )
    require(
        all(type(token) is int and token >= 0 for token in ids), "Candidate token IDs"
    )
    require(
        all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ),
        "Finite raw logits",
    )
    require(values == sorted(values, reverse=True), "Descending top-five logits")
    lse = record["raw_logsumexp"]
    forced = record["forced_raw_logit"]
    require(
        math.isfinite(lse) and math.isfinite(forced),
        "Finite logsumexp and forced logit",
    )
    require(lse >= values[0] and lse >= forced, "Logsumexp bounds observed logits")
    natural = record["natural_raw_top1"]
    require(type(natural) is int and natural >= 0, "Natural argmax token")
    sampler = record["original_sampler_token_id"]
    require(type(sampler) is int and sampler >= 0, "Actual native sampler token")
    candidates = dict(zip(ids, values))
    if natural in candidates:
        require(candidates[natural] == values[0], "Observed natural token is a maximum")
    sampler_logit = record["original_sampler_raw_logit"]
    require(math.isfinite(sampler_logit), "Finite actual sampler raw logit")
    require(record["raw_top_max"] == values[0], "Recorded raw maximum")
    require(
        sampler_logit == values[0] and record["original_sampler_raw_gap_from_top"] == 0,
        "Actual greedy sampler selected an exact raw maximum",
    )
    require(
        record["natural_sampler_differs_from_raw_top1"] == (sampler != natural),
        "Actual sampler versus torch argmax difference flag",
    )
    if sampler in candidates:
        require(
            candidates[sampler] == sampler_logit,
            "Native sampler logit consistent with top five",
        )
    forced_id = record["forced_token_id"]
    if forced_id in candidates:
        require(
            candidates[forced_id] == forced,
            "Forced token raw logit consistent with top five",
        )
    return {
        "raw_argmax_token": natural,
        "actual_sampler_token": sampler,
        "actual_sampler_raw_logit": sampler_logit,
        "sampler_argmax_id_difference_is_exact_raw_tie": sampler != natural,
        "raw_logsumexp": lse,
        "top2_raw_logit_gap": values[0] - values[1],
        "forced_token_id": forced_id,
        "forced_raw_logit": forced,
        "forced_probability": math.exp(forced - lse),
        "top5": [
            {
                "token_id": token,
                "raw_logit": value,
                "probability": math.exp(value - lse),
            }
            for token, value in zip(ids, values)
        ],
        "natural_top1_in_recorded_top5": natural in candidates,
    }


def audit_arm(root, backend, args, reference_rows, reference_sha):
    launch = read_json(root / "launch.json")
    fixture = read_json(root / "fixture.json")
    config = read_json(root / "force-config.json")
    completed = read_json(root / "completed.json")
    require(launch["backend"] == backend, "Launch backend identity")
    require(
        launch["options"]["kernel_config"]["gdn_decode_backend"] == backend,
        "Effective backend option",
    )
    require(
        launch["fixture_sha256"] == sha(root / "fixture.json"), "Launch fixture hash"
    )
    require(
        launch["config_sha256"] == sha(root / "force-config.json"),
        "Launch force configuration hash",
    )
    require(
        launch["reference_sha256"] == fixture["reference_sha256"] == reference_sha,
        "Frozen reference file hash",
    )
    require(
        Path(config["fixture_path"]).name == "fixture.json"
        and Path(config["output_dir"]).name == "trace",
        "Declared fixture/trace path roles",
    )
    require(
        config["limit"] == 512 and set(config["target_indices"]) == {209, 255},
        "Frozen bounded target plan",
    )
    options = launch["options"]
    require(
        options["tensor_parallel_size"] == 2
        and options["enforce_eager"] is True
        and options["async_scheduling"] is False
        and options["enable_trace_replay"] is True,
        "Real eager synchronous TP2 contract",
    )
    require(
        options["enable_prefix_caching"] is False
        and options["mamba_ssm_cache_dtype"] == "float32",
        "Serving state configuration",
    )
    fixtures = {}
    prompts = set()
    for example in fixture["examples"]:
        index = example["index"]
        require(
            index not in fixtures and index in reference_rows,
            "Unique frozen fixture index",
        )
        validate_token_list(example["prompt_token_ids"])
        validate_token_list(example["reference_token_ids"])
        require(
            tuple(example["prompt_token_ids"]) not in prompts,
            "Unique actual prompt token sequence",
        )
        prompts.add(tuple(example["prompt_token_ids"]))
        row = reference_rows[index]
        require(
            example["reference_token_ids"] == row["token_ids"],
            "Complete reference token list exact",
        )
        require(
            token_sha(example["reference_token_ids"]) == row["token_ids_sha256"],
            "Reference token content hash",
        )
        require(
            len(example["prompt_token_ids"]) == row["usage"]["prompt_tokens"],
            "Original prompt token length",
        )
        require(
            example["max_tokens"] == min(len(example["reference_token_ids"]), 512),
            "Bounded forced length",
        )
        require(
            example["reference_token_count"] == len(example["reference_token_ids"]),
            "Untruncated reference count",
        )
        fixtures[index] = example
    require(
        len(fixtures) == 8 and set(fixtures) == set(reference_rows),
        "Exactly eight planned fixtures",
    )
    require(completed["status"] == "completed", "Driver completion status")
    completion_rows = {}
    for item in completed["examples"]:
        index = item["index"]
        require(
            index in fixtures and index not in completion_rows,
            "Unique completed fixture row",
        )
        expected = fixtures[index]["reference_token_ids"][
            : fixtures[index]["max_tokens"]
        ]
        require(
            item["generated_tokens"] == len(expected)
            and item["exact_forced_tokens"] is True,
            "Completed exact forced output length",
        )
        require(item["finish_reason"] == "length", "Fixed forced budget completion")
        require(item["token_ids"] == expected, "Actual returned token IDs exact")
        require(
            item["token_ids_sha256"] == token_sha(expected),
            "Actual returned token ID hash",
        )
        completion_rows[index] = item
    require(set(completion_rows) == set(fixtures), "All eight requests completed")
    helpers = launch["helper_sha256"]
    require(
        {
            "force_decode.py",
            "force_decode_v2.py",
            "force_driver.py",
            "force_site/sitecustomize.py",
        }
        <= set(helpers),
        "Actual force helper identities",
    )
    for name, expected in helpers.items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Safe recorded helper relative path",
        )
        require(
            sha(root / "helpers" / name) == expected,
            f"Frozen snapshot helper hash: {name}",
        )
    trace = root / "trace"
    installed = {
        int(path.stem.rsplit("-", 1)[1]): read_json(path)
        for path in trace.glob("installed-pid-*.json")
    }
    patched = {
        int(path.stem.rsplit("-", 1)[1]): read_json(path)
        for path in trace.glob("patched-pid-*.json")
    }
    require(installed and patched, "Actual model runner hook installed and patched")
    for pid, entry in installed.items():
        require(
            entry["pid"] == pid and entry["event"] == "installed",
            "Installed PID identity",
        )
        require(
            entry["config_sha256"] == launch["config_sha256"]
            and entry["fixture_sha256"] == launch["fixture_sha256"]
            and entry["config"] == config,
            "Runtime installed frozen inputs",
        )
        require(entry["planned_indices"] == sorted(fixtures), "Installed complete plan")
    for pid, entry in patched.items():
        require(
            pid in installed and entry["pid"] == pid and entry["event"] == "patched",
            "Patched installed PID",
        )
        require(
            entry["module_sha256"] == config["expected_runner_sha256"],
            "Actual runner module hash",
        )
        require(
            entry["helper_sha256"] == helpers["force_decode_v2.py"]
            and entry["common_helper_sha256"] == helpers["force_decode.py"],
            "Actual force hook helper hash",
        )
        source_path = frozen_source_path(entry["module_file"], args.source_root)
        require(
            entry["runner_relative_path"] == "vllm/v1/worker/gpu/model_runner.py"
            and PurePosixPath(entry["module_file"])
            == PurePosixPath("/source") / entry["runner_relative_path"],
            "Actual V2 runner relative source identity",
        )
        require(
            sha(source_path) == entry["module_sha256"],
            "On-disk actual runner source hash",
        )
        require(
            entry["module"]
            == installed[pid]["target"]
            == "vllm.v1.worker.gpu.model_runner",
            "Actual patched target module identity",
        )
        require(
            entry["forcing_implementation"] == "native V2 TraceReplayState.apply_trace",
            "Actual native V2 forcing implementation",
        )
        require(
            set(entry["related_source_sha256"])
            == {
                "/source/vllm/v1/worker/gpu/model_runner.py",
                "/source/vllm/v1/worker/gpu/sample/sampler.py",
                "/source/vllm/v1/worker/gpu/sample/trace_replay.py",
                "/source/vllm/v1/worker/gpu/input_batch.py",
                "/source/vllm/v1/worker/gpu/states.py",
            },
            "Actual V2 runner/sampler/trace/input/state source binding",
        )
        for name, expected in entry["related_source_sha256"].items():
            related = frozen_source_path(name, args.source_root)
            require(
                sha(related) == expected,
                "On-disk actual V2 related source hash",
            )
    streams = {}
    rank_records = {}
    rank_schedules = {}
    trace_paths = sorted(trace.glob("events-pid-*.jsonl"))
    require(trace_paths, "Runtime event traces exist")
    for path in trace_paths:
        match = re.fullmatch(r"events-pid-(\d+)\.jsonl", path.name)
        pid = int(match.group(1))
        require(pid in installed, "Event PID has installed configuration")
        registrations, native_slots, records, batches = {}, {}, {}, []
        first_difference = {}
        stream_rank = None
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            event = json.loads(line, parse_constant=reject_constant)
            require(event["pid"] == pid, f"Trace PID identity at {path}:{line_number}")
            require(event["event"] != "force_failure", "No runtime forcing failure")
            require(
                event["event"]
                in ("request_registered", "native_request_registered", "sample_batch"),
                "Known runtime trace event",
            )
            require(
                pid in patched, "Effective event produced by a patched model runner"
            )
            rank = event["rank"]
            require(type(rank) is int and 0 <= rank < 2, "Actual TP rank")
            stream_rank = rank if stream_rank is None else stream_rank
            require(stream_rank == rank, "One event stream retains actual rank")
            if event["event"] == "request_registered":
                req_id, index = event["req_id"], event["index"]
                require(
                    req_id not in registrations
                    and index in fixtures
                    and index not in registrations.values(),
                    "Unique actual request-to-fixture registration",
                )
                fixture_row = fixtures[index]
                require(
                    event["prompt_tokens"] == len(fixture_row["prompt_token_ids"])
                    and event["prompt_sha256"]
                    == token_sha(fixture_row["prompt_token_ids"]),
                    "Actual token prompt identity",
                )
                require(
                    event["reference_sha256"]
                    == token_sha(fixture_row["reference_token_ids"])
                    and event["max_tokens"] == fixture_row["max_tokens"]
                    and event["ignore_eos"] is True,
                    "Actual reference and sampling plan",
                )
                registrations[req_id] = index
                first_difference[index] = None
                continue
            if event["event"] == "native_request_registered":
                req_id, index = event["req_id"], event["index"]
                require(
                    req_id in registrations
                    and registrations[req_id] == index
                    and req_id not in native_slots
                    and event["native_trace_verified"] is True,
                    "Native request/trace registration",
                )
                slot = event["request_state_slot"]
                require(
                    type(slot) is int
                    and slot >= 0
                    and slot not in native_slots.values(),
                    "Unique native request state slot",
                )
                native_slots[req_id] = slot
                continue
            require(
                event["batch_index"] == len(batches) + 1,
                "Continuous actual sample batch counter",
            )
            req_ids = event["actual_req_ids"]
            size = event["actual_batch_size"]
            require(
                size == len(req_ids) and len(set(req_ids)) == size,
                "Actual unique request rows",
            )
            require(
                all(req_id in registrations for req_id in req_ids),
                "Actual request row mapping registered",
            )
            global_ids, global_rows = (
                event["global_req_ids"],
                event["global_row_indices"],
            )
            require(
                event["model_batch_size_global"] == len(global_ids)
                and len(global_ids) == len(set(global_ids))
                and all(req_id in native_slots for req_id in global_ids),
                "Actual global model batch mapping",
            )
            require(
                event["sampler_batch_size_local"] == size
                and len(global_rows) == size
                and len(set(global_rows)) == size
                and all(
                    0 <= global_row < len(global_ids)
                    and global_ids[global_row] == req_ids[row]
                    for row, global_row in enumerate(global_rows)
                ),
                "Actual local sampler-to-global-model row mapping",
            )
            require(
                event["forcing_implementation"]
                == "native V2 TraceReplayState.apply_trace",
                "Native forcing executed for actual sample batch",
            )
            effective = event["effective_rows"]
            discarded = event["discarded_rows"]
            require(
                sorted(effective + discarded) == list(range(size))
                and len(set(effective + discarded)) == size,
                "Effective/discarded row partition",
            )
            require(
                [record["row"] for record in event["records"]] == effective,
                "Effective records retain actual row mapping",
            )
            batch_indices = [registrations[req_id] for req_id in req_ids]
            batch_records = []
            for record in event["records"]:
                row, index, step = record["row"], record["index"], record["step"]
                require(
                    0 <= row < size
                    and req_ids[row] == record["req_id"]
                    and registrations[record["req_id"]] == index,
                    "Actual request/index/row identity",
                )
                fixture_row = fixtures[index]
                planned = fixture_row["max_tokens"]
                key = (index, step)
                require(
                    key not in records and 0 <= step < planned,
                    "Unique in-plan effective step",
                )
                require(
                    record["global_row"] == global_rows[row]
                    and record["recorded_sampling_rank"] == rank
                    and record["request_state_slot"] == native_slots[record["req_id"]],
                    "Actual global row/rank/native slot for effective record",
                )
                require(
                    record["output_length_before"] == step
                    and record["max_tokens"] == planned,
                    "Actual output progress",
                )
                require(
                    record["forced_token_id"]
                    == fixture_row["reference_token_ids"][step],
                    "Exact planned forced token",
                )
                require(
                    record["input_prefix_sha256"]
                    == token_sha(fixture_row["reference_token_ids"][:step]),
                    "Actual forced history hash",
                )
                prompt_length = len(fixture_row["prompt_token_ids"])
                require(
                    record["prompt_length"] == prompt_length
                    and record["prefill_length"] == prompt_length
                    and record["total_length_before_sampling"] == prompt_length + step
                    and record["sequence_after_execute"] == prompt_length + step,
                    "Executed actual sequence history",
                )
                require(
                    record["phase"]
                    == ("prefill_first_token" if step == 0 else "real_decode"),
                    "Actual prefill/decode phase",
                )
                if step >= 1:
                    require(
                        record["computed_before"] == prompt_length + step - 1
                        and record["num_scheduled_tokens"] == 1,
                        "Exactly one real decode token executed",
                    )
                require(
                    record["computed_before"] + record["num_scheduled_tokens"]
                    == record["sequence_after_execute"],
                    "Actual executed scheduled token count",
                )
                require(
                    record["actual_input_token"]
                    == (
                        fixture_row["prompt_token_ids"][-1]
                        if step == 0
                        else fixture_row["reference_token_ids"][step - 1]
                    )
                    and record["actual_input_position"] == prompt_length + step - 1,
                    "Actual executed model input token and position",
                )
                require(
                    record["native_forced_token_id"] == record["forced_token_id"],
                    "Native replay actually returned the forced token",
                )
                require(
                    record["target"] == (index in {209, 255}), "Target/control identity"
                )
                observation = natural_observation(record)
                differs = (
                    record["original_sampler_token_id"] != record["forced_token_id"]
                )
                require(
                    record["natural_differs_from_reference"] == differs,
                    "Natural/reference difference flag",
                )
                if differs and first_difference[index] is None:
                    first_difference[index] = step
                require(
                    record["first_natural_reference_difference_on_rank"]
                    == first_difference[index],
                    "First natural/reference difference progress",
                )
                records[key] = {
                    "record": record,
                    "observation": observation,
                    "batch_context": {
                        "size": size,
                        "fixture_indices": batch_indices,
                        "row": row,
                        "batch_index": event["batch_index"],
                        "effective_rows": effective,
                        "discarded_rows": discarded,
                        "sampling_rank": rank,
                        "global_size": len(global_ids),
                        "global_fixture_indices": [
                            registrations[req_id] for req_id in global_ids
                        ],
                        "global_row": global_rows[row],
                        "sampling_sharded": event["sampling_sharded"],
                        "computed_before": record["computed_before"],
                        "num_scheduled_tokens": record["num_scheduled_tokens"],
                        "actual_input_token": record["actual_input_token"],
                        "actual_input_position": record["actual_input_position"],
                    },
                }
                batch_records.append([index, step])
            batches.append(
                {
                    "fixture_indices": batch_indices,
                    "effective_records": batch_records,
                    "effective_rows": effective,
                    "discarded_rows": discarded,
                    "global_fixture_indices": [
                        registrations[req_id] for req_id in global_ids
                    ],
                    "global_row_indices": global_rows,
                    "sampling_sharded": event["sampling_sharded"],
                }
            )
        if records:
            require(
                set(registrations.values()) == set(fixtures)
                and set(native_slots) == set(registrations),
                "Complete actual request/native trace registration",
            )
            require(
                stream_rank not in rank_records,
                "One effective sampler process per actual rank",
            )
            rank_records[stream_rank] = records
            rank_schedules[stream_rank] = batches
        streams[pid] = {
            "rank": stream_rank,
            "registered_requests": len(registrations),
            "effective_records": len(records),
            "sample_batches": len(batches),
        }
    ranks = sorted(rank_records)
    require(
        ranks == [0] or ranks == [0, 1], "Explicit rank0-only sampler or both TP ranks"
    )
    expected_keys = {
        (index, step)
        for index, item in fixtures.items()
        for step in range(item["max_tokens"])
    }
    owners = {}
    canonical = {}
    duplicate_count = 0
    for rank in ranks:
        for key, observed in rank_records[rank].items():
            owners.setdefault(key, []).append(rank)
            if key not in canonical:
                canonical[key] = observed
                continue
            duplicate_count += 1
            previous = canonical[key]
            require(
                previous["batch_context"]["sampling_sharded"] is False
                and observed["batch_context"]["sampling_sharded"] is False,
                "Sharded sampling observations must not overlap across ranks",
            )
            excluded = {
                "req_id",
                "row",
                "recorded_sampling_rank",
                "request_state_slot",
                "first_natural_reference_difference_on_rank",
            }
            require(
                {
                    name: value
                    for name, value in previous["record"].items()
                    if name not in excluded
                }
                == {
                    name: value
                    for name, value in observed["record"].items()
                    if name not in excluded
                },
                "Duplicate rank history and logits agree",
            )
            require(
                all(
                    previous["batch_context"][name] == observed["batch_context"][name]
                    for name in ("global_size", "global_fixture_indices", "global_row")
                ),
                "Duplicate rank observation belongs to the same global model context",
            )
    require(
        set(canonical) == expected_keys,
        "Global shard union covers every continuous index/step exactly",
    )
    global_first = {
        index: next(
            (
                step
                for step in range(item["max_tokens"])
                if canonical[index, step]["record"]["original_sampler_token_id"]
                != item["reference_token_ids"][step]
            ),
            None,
        )
        for index, item in fixtures.items()
    }
    result = {
        "backend": backend,
        "input_files": file_manifest(root),
        "launch": launch,
        "config": config,
        "sampler_ranks": ranks,
        "sampler_rank_policy": "only rank0 observed"
        if ranks == [0]
        else "both ranks aggregated; every replicated observation verified",
        "duplicate_rank_record_count": duplicate_count,
        "global_unique_effective_records": len(canonical),
        "true_decode_observations": sum(step >= 1 for _, step in canonical),
        "sampler_argmax_exact_tie_id_difference_count": sum(
            item["observation"]["sampler_argmax_id_difference_is_exact_raw_tie"]
            for item in canonical.values()
        ),
        "per_fixture_rank_ownership_counts": {
            index: {
                rank: sum(
                    key[0] == index and rank in value for key, value in owners.items()
                )
                for rank in ranks
            }
            for index in fixtures
        },
        "global_first_actual_sampler_reference_difference": global_first,
        "streams": streams,
        "effective_records_per_rank": {
            rank: len(items) for rank, items in rank_records.items()
        },
        "completed_lengths": {
            index: item["generated_tokens"] for index, item in completion_rows.items()
        },
        "all_actual_returned_token_ids_verified": True,
        "integrity_pass": True,
    }
    return result, canonical, rank_schedules, fixtures


def compare_pair(triton, flashinfer, schedules, fixtures):
    require(set(triton) == set(flashinfer), "Identical paired index/step coverage")
    differences = []
    sampler_differences = []
    fixture_results = []
    for index, fixture in sorted(fixtures.items()):
        flips = []
        sampler_flips = []
        candidate_changes = []
        different_context = []
        for step in range(fixture["max_tokens"]):
            left, right = triton[index, step], flashinfer[index, step]
            lhs, rhs = left["record"], right["record"]
            for field in (
                "index",
                "step",
                "phase",
                "prompt_length",
                "max_tokens",
                "input_prefix_sha256",
                "forced_token_id",
                "target",
            ):
                require(
                    lhs[field] == rhs[field], f"Paired shared-prefix field: {field}"
                )
            same_context = left["batch_context"] == right["batch_context"]
            if not same_context:
                different_context.append(step)
            if (
                lhs["top5_ids"] != rhs["top5_ids"]
                or lhs["top5_raw_logits"] != rhs["top5_raw_logits"]
                or lhs["forced_raw_logit"] != rhs["forced_raw_logit"]
                or lhs["raw_logsumexp"] != rhs["raw_logsumexp"]
            ):
                candidate_changes.append(step)
            if lhs["natural_raw_top1"] != rhs["natural_raw_top1"]:
                flips.append(step)
            sampler_differs = (
                lhs["original_sampler_token_id"] != rhs["original_sampler_token_id"]
            )
            if sampler_differs:
                sampler_flips.append(step)
            if step in flips or sampler_differs:
                paired_candidates = []
                candidates = []
                for item in (lhs, rhs):
                    values = dict(zip(item["top5_ids"], item["top5_raw_logits"]))
                    values[item["forced_token_id"]] = item["forced_raw_logit"]
                    values[item["original_sampler_token_id"]] = item[
                        "original_sampler_raw_logit"
                    ]
                    values[item["natural_raw_top1"]] = item["raw_top_max"]
                    candidates.append(values)
                for token in sorted(set(candidates[0]) | set(candidates[1])):
                    observed = {}
                    for name, values, record in zip(
                        ("triton", "flashinfer"), candidates, (lhs, rhs)
                    ):
                        value = values.get(token)
                        observed[name] = {
                            "raw_logit": value,
                            "probability": math.exp(value - record["raw_logsumexp"])
                            if value is not None
                            else None,
                        }
                    paired_candidates.append({"token_id": token, **observed})
                detail = {
                    "index": index,
                    "step": step,
                    "target": index in {209, 255},
                    "phase": lhs["phase"],
                    "prefix_sha256": lhs["input_prefix_sha256"],
                    "same_actual_batch_context": same_context,
                    "triton": left["observation"],
                    "flashinfer": right["observation"],
                    "triton_batch_context": left["batch_context"],
                    "flashinfer_batch_context": right["batch_context"],
                    "paired_observed_candidates": paired_candidates,
                    "raw_argmax_differs": step in flips,
                    "actual_sampler_differs": sampler_differs,
                }
                if step in flips:
                    differences.append(detail)
                if sampler_differs:
                    sampler_differences.append(detail)
        fixture_results.append(
            {
                "index": index,
                "target": index in {209, 255},
                "planned_tokens": fixture["max_tokens"],
                "true_decode_observations": max(0, fixture["max_tokens"] - 1),
                "prefill_natural_top1_same": 0 not in flips,
                "prefill_actual_sampler_same": 0 not in sampler_flips,
                "prefill_recorded_values_same": 0 not in candidate_changes,
                "first_natural_top1_difference": min(flips) if flips else None,
                "first_real_decode_natural_top1_difference": next(
                    (step for step in flips if step >= 1), None
                ),
                "natural_top1_difference_steps": flips,
                "actual_sampler_difference_steps": sampler_flips,
                "first_actual_sampler_difference": min(sampler_flips)
                if sampler_flips
                else None,
                "first_real_decode_actual_sampler_difference": next(
                    (step for step in sampler_flips if step >= 1), None
                ),
                "recorded_candidate_difference_count": len(candidate_changes),
                "first_recorded_candidate_difference": min(candidate_changes)
                if candidate_changes
                else None,
                "different_actual_batch_context_steps": different_context,
            }
        )
    return {
        "all_forced_prefixes_and_tokens_identical": True,
        "actual_batch_schedules_identical": schedules[0] == schedules[1],
        "all_prefill_natural_top1_identical": all(
            item["prefill_natural_top1_same"] for item in fixture_results
        ),
        "all_prefill_recorded_values_identical": all(
            item["prefill_recorded_values_same"] for item in fixture_results
        ),
        "fixtures": fixture_results,
        "natural_top1_flips": differences,
        "raw_argmax_flips": differences,
        "actual_sampler_flips": sampler_differences,
        "all_prefill_actual_sampler_identical": all(
            item["prefill_actual_sampler_same"] for item in fixture_results
        ),
        "raw_argmax_flip_count": len(differences),
        "real_decode_raw_argmax_flip_count": sum(
            item["step"] >= 1 for item in differences
        ),
        "actual_sampler_flip_count": len(sampler_differences),
        "real_decode_actual_sampler_flip_count": sum(
            item["step"] >= 1 for item in sampler_differences
        ),
        "prefill_recorded_values_scope": "top five, forced logit, and logsumexp",
        "recorded_candidate_difference_scope": "candidate IDs/order and values",
        "batch_schedule_scope": "request batches; GPU padding shapes unobserved",
        "first_flip_scope": "first observed step for this bounded forced prefix only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-dir", "--triton", type=Path, required=True)
    parser.add_argument("--flashinfer-dir", "--flashinfer", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--helpers-dir",
        type=Path,
        help="Optional live helpers path; producer hashes use frozen run snapshots",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite audit evidence")
    require(
        all(
            not args.output.resolve().is_relative_to(root.resolve())
            for root in (args.triton_dir, args.flashinfer_dir)
        ),
        "Audit output outside immutable run inputs",
    )
    report = {
        "helper_sha256": sha(Path(__file__)),
        "integrity_pass": False,
        "input_dirs": {
            "triton": str(args.triton_dir),
            "flashinfer": str(args.flashinfer_dir),
        },
        "input_files": {
            "triton": file_manifest(args.triton_dir),
            "flashinfer": file_manifest(args.flashinfer_dir),
        },
        "reference_sha256": sha(args.reference),
        "source_manifest_sha256": sha(args.source_manifest),
        "scope": {
            "max_forced_tokens": 512,
            "true_incremental_decode_verified": False,
            "no_accuracy_or_performance_claim": True,
            "full_vocabulary_logits_compared": False,
            "repeatability_beyond_one_pair_verified": False,
            "low_margin_proves_harmlessness": False,
            "old_long_generation_root_cause_established": False,
            "first_global_causal_divergence_established": False,
            "first_step_is_prefill_then_steps_1_plus_are_decode": True,
        },
    }
    exit_code = 0
    try:
        manifest = read_json(args.source_manifest)
        require(
            manifest["source_head"] == "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6",
            "Frozen production source revision",
        )
        report["source_head"] = manifest["source_head"]
        source_hashes = {}
        for name, entry in manifest["files"].items():
            observed = sha(args.source_root / name)
            require(observed == entry["sha256"], f"Frozen source hash: {name}")
            source_hashes[name] = observed
        report["source_files"] = source_hashes
        reference = read_json(args.reference)
        require(reference["arm"] == "triton", "Triton diagnostic reference origin")
        rows = next(
            item["examples"] for item in reference["rounds"] if item["concurrency"] == 8
        )
        require(
            [item["index"] for item in rows]
            == [198, 206, 209, 228, 255, 285, 292, 318],
            "Frozen diagnostic reference indices",
        )
        reference_rows = {item["index"]: item for item in rows}
        left, triton, triton_schedule, left_fixtures = audit_arm(
            args.triton_dir, "triton", args, reference_rows, report["reference_sha256"]
        )
        right, flashinfer, flashinfer_schedule, right_fixtures = audit_arm(
            args.flashinfer_dir,
            "flashinfer",
            args,
            reference_rows,
            report["reference_sha256"],
        )
        require(left_fixtures == right_fixtures, "Identical paired fixture content")
        executed_helpers = {
            "force_decode.py",
            "force_decode_v2.py",
            "force_driver.py",
            "force_site/sitecustomize.py",
        }
        require(
            all(
                left["launch"]["helper_sha256"][name]
                == right["launch"]["helper_sha256"][name]
                for name in executed_helpers
            ),
            "Identical paired executed producer helper hashes",
        )
        if args.helpers_dir is not None:
            report["live_executed_helper_sha256"] = {
                name: sha(args.helpers_dir / name) for name in executed_helpers
            }
            report["live_executed_helpers_match_frozen_snapshot"] = all(
                report["live_executed_helper_sha256"][name]
                == left["launch"]["helper_sha256"][name]
                for name in executed_helpers
            )
        report["unexecuted_helper_manifest_differences"] = sorted(
            name
            for name in set(left["launch"]["helper_sha256"])
            | set(right["launch"]["helper_sha256"])
            if left["launch"]["helper_sha256"].get(name)
            != right["launch"]["helper_sha256"].get(name)
        )
        left_options = json.loads(json.dumps(left["launch"]["options"]))
        right_options = json.loads(json.dumps(right["launch"]["options"]))
        left_options["kernel_config"].pop("gdn_decode_backend")
        right_options["kernel_config"].pop("gdn_decode_backend")
        require(
            left_options == right_options,
            "All paired model options equal except GDN decode backend",
        )
        normalized = lambda config: {
            name: value
            for name, value in config.items()
            if name not in {"output_dir", "fixture_path"}
        }
        require(
            normalized(left["config"]) == normalized(right["config"]),
            "Paired forcing configuration equal except run paths",
        )
        report["arms"] = {"triton": left, "flashinfer": right}
        report["comparison"] = compare_pair(
            triton, flashinfer, (triton_schedule, flashinfer_schedule), left_fixtures
        )
        require(
            report["input_files"]["triton"] == file_manifest(args.triton_dir)
            and report["input_files"]["flashinfer"]
            == file_manifest(args.flashinfer_dir),
            "All input files remained unchanged throughout audit",
        )
        report["scope"]["true_incremental_decode_verified"] = True
        report["integrity_pass"] = True
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {"integrity_pass": report["integrity_pass"], "output": str(args.output)}
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
