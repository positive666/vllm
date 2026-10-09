"""Audit the two conditional shared-history arms without importing vLLM.

This checks actual native replay and model-execution identity. Forced output is
neither accuracy, natural stopping, performance nor a causal decode diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
REFERENCE_SHA = "d293b322876e8236f7799a688221ddd0d10e79f8046c6601db40fa6316421e78"
INDICES = (198, 206, 209, 228, 255, 285, 292, 318)
STEPS = (1, 18, 19, 1750, 3497, 3498, 3499)
NATIVE = "native V2 TraceReplayState.apply_trace"
EOS_SCOPE = ("TokenizerEOS248046candidate only; additional model EOS/stop IDs "
             "are not raw-logit-covered")


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def token_sha(tokens):
    return hashlib.sha256(json.dumps(tokens, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON: " + value)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant)


def relative(value):
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and ".." not in path.parts
            and "\\" not in value and ":" not in value, "Safe relative path")
    return path


def source_path(value, source_root):
    path = PurePosixPath(value)
    require(path.is_absolute() and ".." not in path.parts, "Safe source path")
    name = path.relative_to(PurePosixPath("/source"))
    require(name.parts[0] == "vllm", "Actual vLLM source path")
    return Path(source_root).joinpath(*name.parts)


def checked_tokens(value):
    require(isinstance(value, list) and value and
            all(type(item) is int and item >= 0 for item in value),
            "Actual nonempty token IDs")


def reference_inputs(root):
    root = Path(root)
    lock = read(root / "reference-lock.json")
    require(lock["source_head"] == HEAD and tuple(lock["indices"]) == INDICES
            and lock["target"] == 255 and tuple(lock["selected_steps"]) == STEPS
            and lock["limit"] == 3500, "Exact reviewed conditional input plan")
    require(lock["reference_files"]["completed.json"] == REFERENCE_SHA,
            "Original unforced FI eager1 reference")
    for name, expected in lock["reference_files"].items():
        require(sha(root.joinpath("reference", *relative(name).parts)) == expected,
                "Independent locked reference bytes: " + name)
    ref = root / "reference"
    require((ref / "run.exit").read_text().strip() == "0", "Reference exit0")
    completed, launch = read(ref / "completed.json"), read(ref / "launch.json")
    fixture = read(ref / "fixture.json")
    require(completed["status"] == "completed" and completed["backend"] == "flashinfer"
            and completed["mode"] == "eager", "Original unforced eager arm")
    require(launch["source_head"] == HEAD and completed["launch_sha256"] ==
            sha(ref / "launch.json") and launch["fixture_sha256"] ==
            sha(ref / "fixture.json"), "Reference launch and prompt binding")
    require([item["index"] for item in completed["examples"]] == list(INDICES)
            == [item["index"] for item in fixture["examples"]], "Eight selected inputs")
    rows = {}
    for item, prompt in zip(completed["examples"], fixture["examples"]):
        ids, prompt_ids = item["token_ids"], item["prompt_token_ids"]
        checked_tokens(ids)
        checked_tokens(prompt_ids)
        require(prompt_ids == prompt["prompt_token_ids"] and token_sha(prompt_ids) ==
                item["prompt_token_ids_sha256"] and token_sha(ids) ==
                item["token_ids_sha256"], "Actual frozen token identities")
        require(len(ids) <= 3500 and len(ids) + len(prompt_ids) <= 4096,
                "Full reference within context")
        rows[item["index"]] = {
            "index": item["index"], "prompt_token_ids": prompt_ids,
            "reference_token_ids": ids, "max_tokens": len(ids),
            "reference_token_count": len(ids),
            "reference_token_ids_sha256": token_sha(ids),
        }
    target = next(item for item in completed["examples"] if item["index"] == 255)
    require(len(target["token_ids"]) == 3500 and target["finish_reason"] == "length"
            and target["has_answer_marker"] is False, "Retained unforced termination gap")
    require(sum(item["max_tokens"] for item in rows.values()) == 7378,
            "Full eight-history 7378-position contract")
    return lock, rows, launch, read(ref / "runtime-binding.json")


def expected_options(backend):
    return {"model": "/model", "language_model_only": True, "dtype": "bfloat16",
            "tensor_parallel_size": 2, "max_model_len": 4096, "max_num_seqs": 32,
            "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": False, "mamba_ssm_cache_dtype": "float32",
            "seed": 42, "enforce_eager": True, "async_scheduling": False,
            "enable_trace_replay": True,
            "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
            "kernel_config": {"gdn_decode_backend": backend, "linear_backend": "marlin"},
            "num_gpu_blocks_override": 128}


def runtime_workers(runtime, backend):
    require(runtime["engine_core_class"] == "InprocClient", "Actual in-process core")
    workers = sorted(runtime["worker_ranks"], key=lambda item: item["rank"])
    require([item["rank"] for item in workers] == [0, 1], "Two actual workers")
    for worker in workers:
        require(worker["tp_rank"] == worker["rank"] and worker["tp_world_size"] == 2
                and worker["tp_device_backend"] == "nccl", "Actual TP2 NCCL")
        require(worker["runner_class"] == "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                and worker["gpu_name"] == "NVIDIA L20" and
                worker["compute_capability"] == [8, 9] and
                worker["gpu_uuid"] != "unavailable" and
                len(worker["nccl_version"]) == 3, "Actual L20 V2 worker")
        config = worker["config"]
        model, cache = config["model_config"], config["cache_config"]
        sched = config["scheduler_config"]
        require(model["seed"] == 42 and model["enforce_eager"] is True and
                model["enable_trace_replay"] is True and model["max_model_len"] == 4096
                and str(model["dtype"]).endswith("bfloat16") and
                model["quantization"] == "fp8" and
                model["multimodal_config"]["language_model_only"] is True,
                "Resolved model/seed/native replay")
        require(cache["enable_prefix_caching"] is False and
                cache["mamba_ssm_cache_dtype"] == "float32" and
                cache["num_gpu_blocks_override"] == 128 and
                cache["gpu_memory_utilization"] == 0.9, "Resolved cache settings")
        require(sched["async_scheduling"] is False and sched["max_num_seqs"] == 32
                and sched["max_num_batched_tokens"] == 1024 and
                config["parallel_config"]["tensor_parallel_size"] == 2 and
                config["parallel_config"]["enable_batch_sharded_sampling"] is False and
                config["parallel_config"]["pipeline_parallel_size"] == 1,
                "Resolved synchronous scheduler")
        require(config["kernel_config"]["gdn_decode_backend"] == backend and
                config["kernel_config"]["linear_backend"] == "marlin" and
                config["attention_config"]["backend"] == "FLASH_ATTN" and
                config["attention_config"]["flash_attn_version"] == 2 and
                config["speculative_config"] is None, "Resolved operators")
        require(config["compilation_config"]["cudagraph_mode"] == "NONE" and
                worker["graphs_captured"] is False and not worker["captured_token_counts"]
                and not worker["graph_descriptors"] and
                worker["compilation_counter"]["num_cudagraph_captured"] == 0,
                "Observed eager has no captured graphs")
    require(len({item["gpu_uuid"] for item in workers}) == 2, "Distinct physical GPUs")
    return workers


def provenance(root, backend, reference_root, source_root, source_manifest, inputs):
    lock, fixtures, reference_launch, binding = inputs
    root, reference_root = Path(root), Path(reference_root)
    require(not (root / "failure.json").exists(), "No producer failure")
    require((root.parent / (root.name + ".exit")).read_text().strip() == "0",
            "Actual arm process exit0")
    launch, completed = read(root / "launch.json"), read(root / "completed.json")
    config, capture = read(root / "force-config.json"), read(root / "capture-config.json")
    require(launch["backend"] == backend and launch["source_head"] == HEAD and
            launch["status"] == "starting" and launch["options"] == expected_options(backend),
            "Exact requested arm")
    manifest = read(source_manifest)
    require(manifest["source_head"] == HEAD and launch["source_manifest_sha256"] ==
            sha(source_manifest), "Independent reviewed source manifest")
    require(launch["source_files"] == lock["source_files"], "Independent locked runtime source")
    for name, expected in lock["source_files"].items():
        require(sha(Path(source_root).joinpath(*relative(name).parts)) == expected,
                "Actual runtime source: " + name)
    for name, expected in manifest["files"].items():
        require(sha(Path(source_root).joinpath(*relative(name).parts)) == expected["sha256"],
                "Actual manifest source: " + name)
    require(launch["reference_lock_sha256"] == sha(reference_root / "reference-lock.json")
            == sha(root / "reference-lock.json") and
            launch["reference_completed_sha256"] == REFERENCE_SHA,
            "Independent input lock binding")
    for name, expected in lock["reference_files"].items():
        require(sha(root.joinpath("reference", *relative(name).parts)) == expected,
                "Copied reference bytes: " + name)
    require(read(root / "fixture.json") == {"examples": list(fixtures.values())}
            == launch["fixture"] and sha(root / "fixture.json") == launch["fixture_sha256"],
            "Exact full-history fixture")
    require(sha(root / "force-config.json") == launch["force_config_sha256"] and
            sha(root / "capture-config.json") == launch["capture_config_sha256"],
            "Actual observer configurations")
    require(config["limit"] == 3500 and config["target_indices"] == [209, 255] and
            config["snapshot_target"] == 255 and tuple(config["snapshot_steps"]) == STEPS
            and config["eos_candidate_ids"] == [248046] and
            config["eos_observation_scope"] == EOS_SCOPE, "Exact conditional observer plan")
    recorded_root = PurePosixPath("/results") / root.name
    require(config["fixture_path"] == str(recorded_root / "fixture.json") and
            config["output_dir"] == str(recorded_root / "trace") and
            capture["output_dir"] == str(recorded_root / "snapshots") and
            capture["start_marker"] == str(recorded_root / "START"),
            "Owned actual observer input/output paths")
    require(config["expected_runner_sha256"] == lock["source_files"][
                "vllm/v1/worker/gpu/model_runner.py"] and
            config["expected_input_batch_sha256"] == lock["source_files"][
                "vllm/v1/worker/gpu/input_batch.py"], "Expected actual V2 source")
    require(capture["source_head"] == HEAD and capture["initial_batch_size"] == 8 and
            capture["early_calls"] == capture["sample_calls"] == capture["sample_layers"] == []
            and capture["expected_module_sha256"] == lock["source_files"][
                "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"],
            "Exact layer capture plan")
    require(launch["target"] == 255 and tuple(launch["snapshot_steps"]) == STEPS
            and launch["scope"] == lock["scope"] == completed["scope"], "Conditional scope")
    for key, expected in (("fresh_model_files", reference_launch["model_files"]),
                          ("fresh_runtime_versions", reference_launch["runtime_versions"]),
                          ("fresh_runtime_files", binding["modules"])):
        require(launch[key] == expected, "Fresh frozen model/native binding: " + key)
    helpers = launch["helper_sha256"]
    required = {"late_driver.py", "late_bootstrap.py", "late_plan.py", "late_capture.py",
                "late_runtime.py", "launch_capture.py", "force_decode.py",
                "force_decode_v2.py", "late_site/sitecustomize.py"}
    require(required <= set(helpers), "Actual producer helper identities")
    for name, expected in helpers.items():
        require(sha(root.joinpath("helpers", *relative(name).parts)) == expected,
                "Copied helper bytes: " + name)
    require(completed["status"] == "completed" and completed["launch_sha256"] ==
            sha(root / "launch.json"), "Completed actual launch")
    require([item["index"] for item in completed["examples"]] == list(INDICES),
            "All eight histories returned")
    for item in completed["examples"]:
        expected = fixtures[item["index"]]["reference_token_ids"]
        require(item["token_ids"] == expected and item["token_ids_sha256"] ==
                token_sha(expected) and item["exact_forced_tokens"] is True,
                "Actual returned native forced token IDs")
        require(item["finish_reason"] == "length", "Conditional fixed-budget stopping")
    require(read(root / "shutdown.json") ==
            {"status": "shutdown-returned", "generation_completed": True},
            "Actual successful core shutdown")
    before = runtime_workers(read(root / "runtime-before.json"), backend)
    after = runtime_workers(read(root / "runtime-after.json"), backend)
    for first, last in zip(before, after):
        for key in ("rank", "tp_rank", "gpu_uuid", "gpu_name", "compute_capability",
                    "nccl_version", "runner_class", "config"):
            require(first[key] == last[key], "Stable actual worker: " + key)
    return launch, config, after


def natural_observation(record):
    """Reuse the verified decode-shadow raw/native-greedy observation contract."""
    ids, values = record["top5_ids"], record["top5_raw_logits"]
    require(len(ids) == len(values) == 5 and len(set(ids)) == 5 and
            all(type(token) is int and token >= 0 for token in ids), "Five unique candidates")
    require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            and values == sorted(values, reverse=True), "Finite ordered raw logits")
    lse, forced = record["raw_logsumexp"], record["forced_raw_logit"]
    require(math.isfinite(lse) and math.isfinite(forced) and
            lse >= max(values[0], forced), "Finite bounded raw logits")
    natural, sampler = record["natural_raw_top1"], record["original_sampler_token_id"]
    require(type(natural) is int and natural >= 0 and type(sampler) is int and sampler >= 0,
            "Actual natural/native IDs")
    candidates = dict(zip(ids, values))
    require(natural not in candidates or candidates[natural] == values[0], "Natural maximum")
    require(record["raw_top_max"] == values[0] and
            record["original_sampler_raw_logit"] == values[0] and
            record["original_sampler_raw_gap_from_top"] == 0,
            "Actual greedy native sampler selected an exact raw maximum")
    require(record["natural_sampler_differs_from_raw_top1"] == (sampler != natural),
            "Natural/native raw-tie identity")
    require(sampler not in candidates or candidates[sampler] == values[0], "Native top5 value")
    require(record["forced_token_id"] not in candidates or
            candidates[record["forced_token_id"]] == forced, "Forced candidate value")
    return {"raw_argmax_token": natural, "actual_sampler_token": sampler,
            "sampler_argmax_id_difference_is_exact_raw_tie": sampler != natural,
            "top2_raw_logit_gap": values[0] - values[1],
            "forced_probability": math.exp(forced - lse)}


def installed_hooks(root, launch, config, fixtures, source_root):
    trace = root / "trace"
    installed = {int(path.stem.rsplit("-", 1)[1]): read(path)
                 for path in trace.glob("installed-pid-*.json")}
    patched = {int(path.stem.rsplit("-", 1)[1]): read(path)
               for path in trace.glob("patched-pid-*.json")}
    require(installed and patched, "Actual installed and patched model runner")
    for pid, event in installed.items():
        require(event["pid"] == pid and event["event"] == "installed" and
                event["config_sha256"] == launch["force_config_sha256"] and
                event["fixture_sha256"] == launch["fixture_sha256"] and
                event["config"] == config and event["planned_indices"] == sorted(fixtures),
                "Actual installed frozen inputs")
    related = {"/source/vllm/v1/worker/gpu/" + name for name in (
        "model_runner.py", "sample/sampler.py", "sample/trace_replay.py",
        "input_batch.py", "states.py")}
    for pid, event in patched.items():
        require(pid in installed and event["pid"] == pid and event["event"] == "patched"
                and event["module_sha256"] == config["expected_runner_sha256"] and
                event["helper_sha256"] == launch["helper_sha256"]["force_decode_v2.py"] and
                event["common_helper_sha256"] == launch["helper_sha256"]["force_decode.py"],
                "Actual patched observer/source bytes")
        require(event["runner_relative_path"] == "vllm/v1/worker/gpu/model_runner.py" and
                event["module_file"] == "/source/" + event["runner_relative_path"] and
                event["module"] == installed[pid]["target"] ==
                "vllm.v1.worker.gpu.model_runner" and event["forcing_implementation"] == NATIVE
                and set(event["related_source_sha256"]) == related,
                "Actual V2 native forcing implementation")
        for name, expected in event["related_source_sha256"].items():
            require(sha(source_path(name, source_root)) == expected,
                    "Independent actual V2 related source: " + name)
    return installed, patched


def audit_trace(root, launch, config, fixtures, source_root):
    """Retain actual sampler ownership; do not manufacture per-rank observations."""
    root = Path(root)
    installed, patched = installed_hooks(root, launch, config, fixtures, source_root)
    rank_records, schedules, prepared, sampled, execution_records, streams = {}, {}, {}, {}, {}, {}
    paths = sorted((root / "trace").glob("events-pid-*.jsonl"))
    require(paths, "Actual event traces exist")
    for path in paths:
        match = re.fullmatch(r"events-pid-(\d+)\.jsonl", path.name)
        require(match is not None, "Known trace filename")
        pid = int(match.group(1))
        require(pid in installed, "Trace PID installed")
        registrations, slots, records, batches, first = {}, {}, {}, [], {}
        stream_rank = None
        prepared_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line, parse_constant=reject_constant)
            kind, rank = event["event"], event["rank"]
            require(event["pid"] == pid and pid in patched and type(rank) is int and
                    rank in (0, 1), "Actual patched trace PID/rank")
            require(kind in {"request_registered", "native_request_registered",
                             "late_prepared_batch", "sample_batch"}, "Known successful event")
            stream_rank = rank if stream_rank is None else stream_rank
            require(stream_rank == rank, "Stable actual trace rank")
            if kind == "request_registered":
                req_id, index = event["req_id"], event["index"]
                require(req_id not in registrations and index in fixtures and
                        index not in registrations.values(), "Unique request fixture mapping")
                fixture = fixtures[index]
                require(event["prompt_tokens"] == len(fixture["prompt_token_ids"]) and
                        event["prompt_sha256"] == token_sha(fixture["prompt_token_ids"]) and
                        event["reference_sha256"] == token_sha(fixture["reference_token_ids"]) and
                        event["max_tokens"] == fixture["max_tokens"] and
                        event["ignore_eos"] is True, "Actual registered forced prompt/history")
                registrations[req_id], first[index] = index, None
                continue
            if kind == "native_request_registered":
                req_id = event["req_id"]
                require(registrations.get(req_id) == event["index"] and req_id not in slots
                        and event["native_trace_verified"] is True, "Native trace registration")
                slot = event["request_state_slot"]
                require(type(slot) is int and slot >= 0 and slot not in slots.values(),
                        "Actual unique request-state slot")
                slots[req_id] = slot
                continue
            execution_id = event["late_execution_id"]
            require(type(execution_id) is int and execution_id >= 1, "Actual execution ID")
            execution_key = (rank, execution_id)
            if kind == "late_prepared_batch":
                require(execution_key not in prepared and execution_id == prepared_count + 1,
                        "Unique continuous model execution boundary")
                req_ids = event["req_ids"]
                require(event["num_reqs"] == len(req_ids) == len(set(req_ids)) and
                        all(req_id in slots for req_id in req_ids) and
                        len(event["num_scheduled_tokens"]) == len(req_ids) and
                        all(type(count) is int and count >= 1 for count in
                            event["num_scheduled_tokens"]), "Actual prepared global input rows")
                target_step = event["actual_target_step"]
                require(target_step is None or (type(target_step) is int and
                        1 <= target_step < 3500 and
                        255 in [registrations[req_id] for req_id in req_ids]),
                        "Actual target decode progress")
                require(event["capture_selected"] == (target_step in STEPS),
                        "Actual selected-step prefilter")
                prepared[execution_key] = event
                if execution_id == 1:
                    require(set(registrations[req_id] for req_id in req_ids) == set(fixtures)
                            and all(count == len(fixtures[registrations[req_id]]["prompt_token_ids"])
                                    for req_id, count in zip(req_ids, event["num_scheduled_tokens"])),
                            "All eight full prompts execute in the first model batch")
                prepared_count += 1
                continue
            require(execution_key in prepared and execution_key not in sampled,
                    "Actual sample joined to unique same-rank prepared input")
            before = prepared[execution_key]
            require(event["batch_index"] == len(batches) + 1, "Continuous sampler batches")
            req_ids, size = event["actual_req_ids"], event["actual_batch_size"]
            global_ids, global_rows = event["global_req_ids"], event["global_row_indices"]
            require(size == len(req_ids) == len(set(req_ids)) and
                    all(req_id in registrations for req_id in req_ids), "Actual local sampler rows")
            require(global_ids == before["req_ids"] and
                    event["model_batch_size_global"] == len(global_ids) and
                    event["sampler_batch_size_local"] == size and len(global_rows) == size and
                    len(set(global_rows)) == size and all(type(row) is int and
                    0 <= row < len(global_ids) and global_ids[row] == req_ids[local]
                    for local, row in enumerate(global_rows)), "Exact local-to-global execution join")
            require(type(event["sampling_sharded"]) is bool and
                    event["forcing_implementation"] == NATIVE, "Actual native sampler mode")
            effective, discarded = event["effective_rows"], event["discarded_rows"]
            require(sorted(effective + discarded) == list(range(size)) and
                    len(set(effective + discarded)) == size and
                    [item["row"] for item in event["records"]] == effective,
                    "Actual effective/discarded sampler row partition")
            batch_indices = [registrations[req_id] for req_id in req_ids]
            global_indices = [registrations[req_id] for req_id in global_ids]
            batch_records = []
            for record in event["records"]:
                row, index, step = record["row"], record["index"], record["step"]
                key = (index, step)
                require(type(step) is int and 0 <= row < size and
                        req_ids[row] == record["req_id"] and
                        registrations[record["req_id"]] == index and key not in records,
                        "Unique actual sampler row/index/step")
                fixture = fixtures[index]
                prompt, reference = fixture["prompt_token_ids"], fixture["reference_token_ids"]
                length = len(prompt)
                require(0 <= step < fixture["max_tokens"] and record["max_tokens"] ==
                        fixture["max_tokens"] and record["output_length_before"] == step and
                        record["forced_token_id"] == record["native_forced_token_id"] ==
                        reference[step], "Actual native replay token/progress")
                require(record["global_row"] == global_rows[row] and
                        record["recorded_sampling_rank"] == rank and
                        record["request_state_slot"] == slots[record["req_id"]],
                        "Actual sampling rank/global row/request-state slot")
                require(record["input_prefix_sha256"] == token_sha(reference[:step]) and
                        record["prompt_length"] == record["prefill_length"] == length and
                        record["total_length_before_sampling"] ==
                        record["sequence_after_execute"] == length + step,
                        "Actual GPU committed history before native sampling")
                require(record["phase"] == ("prefill_first_token" if step == 0 else "real_decode")
                        and record["computed_before"] + record["num_scheduled_tokens"] ==
                        length + step and record["num_scheduled_tokens"] ==
                        before["num_scheduled_tokens"][global_rows[row]],
                        "Actual prepared/executed sequence length")
                require(step == 0 or (record["computed_before"] == length + step - 1 and
                        record["num_scheduled_tokens"] == 1), "One actual decode token")
                require(step != 0 or (record["computed_before"] == 0 and
                        record["num_scheduled_tokens"] == length and execution_id == 1),
                        "Full first-batch prefill before any decode")
                require(record["actual_input_token"] == (prompt[-1] if step == 0 else
                        reference[step - 1]) and record["actual_input_position"] == length + step - 1
                        and record["target"] == (index in (209, 255)), "Actual input/target identity")
                if index == 255:
                    require(before["actual_target_step"] == (None if step == 0 else step),
                            "Target GPU history agrees with selected-step prefilter")
                prefill_sha = record["raw_logits_full_sha256"]
                require((step == 0 and isinstance(prefill_sha, str) and
                        re.fullmatch(r"[0-9a-f]{64}", prefill_sha)) or
                        (step > 0 and prefill_sha is None), "Full prefill raw-logit SHA scope")
                require(record["eos_observation_scope"] == EOS_SCOPE and
                        record["eos_candidate_ids"] == [248046] and
                        len(record["eos_candidate_raw_logits"]) == 1 and
                        math.isfinite(record["eos_candidate_raw_logits"][0]),
                        "Finite tokenizer-EOS-only observation")
                observation = natural_observation(record)
                differs = record["original_sampler_token_id"] != reference[step]
                require(record["natural_differs_from_reference"] == differs,
                        "Native/reference difference flag")
                if differs and first[index] is None:
                    first[index] = step
                require(record["first_natural_reference_difference_on_rank"] == first[index],
                        "Actual per-rank first difference progress")
                context = {"size": size, "fixture_indices": batch_indices, "row": row,
                           "batch_index": event["batch_index"], "effective_rows": effective,
                           "discarded_rows": discarded, "sampling_rank": rank,
                           "global_size": len(global_ids), "global_fixture_indices": global_indices,
                           "global_req_ids": global_ids, "global_row": global_rows[row],
                           "sampling_sharded": event["sampling_sharded"],
                           "late_execution_id": execution_id,
                           "computed_before": record["computed_before"],
                           "num_scheduled_tokens": record["num_scheduled_tokens"],
                           "actual_input_token": record["actual_input_token"],
                           "actual_input_position": record["actual_input_position"]}
                entry = {"record": record, "observation": observation,
                         "batch_context": context, "context": context, "ranks": [rank]}
                records[key] = entry
                execution_records[rank, execution_id, index, step] = entry
                batch_records.append([index, step])
            sampled[execution_key] = event
            batches.append({"late_execution_id": execution_id, "fixture_indices": batch_indices,
                            "effective_records": batch_records, "effective_rows": effective,
                            "discarded_rows": discarded, "global_fixture_indices": global_indices,
                            "global_row_indices": global_rows,
                            "num_scheduled_tokens": before["num_scheduled_tokens"],
                            "sampling_sharded": event["sampling_sharded"]})
        if stream_rank is not None:
            require(set(registrations.values()) == set(fixtures) and set(slots) ==
                    set(registrations), "All eight actual native requests registered")
            require(stream_rank not in schedules, "One worker trace process per actual rank")
            schedules[stream_rank], rank_records[stream_rank] = batches, records
        streams[pid] = {"rank": stream_rank, "records": len(records),
                        "sample_batches": len(batches), "prepared_batches": prepared_count}
    require(set(schedules) == {0, 1}, "Both TP ranks have actual execution traces")
    missing_local_contexts = sorted(set(prepared) - set(sampled))
    require(set(sampled) <= set(prepared), "Samples belong to actual prepared executions")
    canonical = {}
    excluded = {"req_id", "row", "recorded_sampling_rank", "request_state_slot",
                "first_natural_reference_difference_on_rank"}
    for rank, records in sorted(rank_records.items()):
        for key, entry in records.items():
            if key not in canonical:
                canonical[key] = entry.copy()
                canonical[key]["ranks"] = [rank]
                continue
            previous = canonical[key]
            require(previous["batch_context"]["sampling_sharded"] is False and
                    entry["batch_context"]["sampling_sharded"] is False,
                    "Sharded physical sampler records never overlap")
            require({name: value for name, value in previous["record"].items()
                     if name not in excluded} ==
                    {name: value for name, value in entry["record"].items()
                     if name not in excluded}, "Replicated actual history/native logits agree")
            require(all(previous["batch_context"][name] == entry["batch_context"][name]
                        for name in ("late_execution_id", "global_size", "global_req_ids",
                                     "global_fixture_indices", "global_row")),
                    "Replicated records belong to same global model execution")
            previous["ranks"].append(rank)
    require(set(canonical) == {(index, step) for index, fixture in fixtures.items()
                             for step in range(fixture["max_tokens"])},
            "Global physical shard union covers every planned position")
    for execution_id in sorted({key[1] for key in prepared}):
        left, right = prepared[0, execution_id], prepared[1, execution_id]
        require(all(left[name] == right[name] for name in (
                    "req_ids", "num_reqs", "num_scheduled_tokens",
                    "actual_target_step", "capture_selected")),
                "Both ranks prepared the identical global input execution")
        actual_samples = [sampled[rank, execution_id] for rank in (0, 1)
                          if (rank, execution_id) in sampled]
        require(actual_samples, "At least one physical sampler observed per model execution")
        require(all(event["global_req_ids"] == left["req_ids"] for event in actual_samples),
                "Cross-rank actual sampler/global request identity")
    return (canonical, schedules, execution_records, prepared, sampled, streams,
            missing_local_contexts)


def compare_pair(records, schedules, fixtures):
    left, right = records["triton"], records["flashinfer"]
    require(set(left) == set(right), "Identical paired native observation coverage")
    fields = ("index", "step", "forced_token_id", "input_prefix_sha256", "prompt_length",
              "sequence_after_execute", "computed_before", "num_scheduled_tokens",
              "actual_input_token", "actual_input_position", "phase")
    context_fields = ("late_execution_id", "global_size", "global_fixture_indices", "global_row",
                      "sampling_sharded", "sampling_rank", "fixture_indices", "row",
                      "effective_rows", "discarded_rows")
    context_differences, native_differences = [], []
    prefill = []
    for key in sorted(left):
        a, b = left[key], right[key]
        changed = [name for name in fields if a["record"][name] != b["record"][name]]
        changed += ["context." + name for name in context_fields if
                    a["batch_context"][name] != b["batch_context"][name]]
        if changed:
            context_differences.append({"index": key[0], "step": key[1], "fields": changed})
        if a["record"]["original_sampler_token_id"] != b["record"]["original_sampler_token_id"]:
            native_differences.append({"index": key[0], "step": key[1],
                                      "triton": a["record"]["original_sampler_token_id"],
                                      "flashinfer": b["record"]["original_sampler_token_id"]})
        if key[1] == 0:
            prefill.append({"index": key[0],
                            "triton_sha256": a["record"]["raw_logits_full_sha256"],
                            "flashinfer_sha256": b["record"]["raw_logits_full_sha256"],
                            "exact_equal": a["record"]["raw_logits_full_sha256"] ==
                            b["record"]["raw_logits_full_sha256"],
                            "native_sampler_equal": a["record"]["original_sampler_token_id"] ==
                            b["record"]["original_sampler_token_id"]})
    schedule_equal = schedules["triton"] == schedules["flashinfer"]
    prefill_equal = all(item["exact_equal"] for item in prefill)
    same_prefill_sampler = all(item["native_sampler_equal"] for item in prefill)
    return {"schedules_equal": schedule_equal,
            "matching_execution_contexts": not context_differences,
            "full_prefill_logits_equal": prefill_equal,
            "common_prefill_sampler": same_prefill_sampler,
            "native_starting_condition_gate_pass": schedule_equal and not context_differences
                and prefill_equal and same_prefill_sampler,
            "prefill": prefill, "context_differences": context_differences,
            "native_sampler_differences": native_differences,
            "first_actual_sampler_reference_difference": {
                backend: {index: next((step for step in range(fixture["max_tokens"])
                          if table[index, step]["record"]["original_sampler_token_id"] !=
                          fixture["reference_token_ids"][step]), None)
                          for index, fixture in fixtures.items()}
                for backend, table in records.items()},
            "limitation": "GDN initial-prestate and full-attention KV equality are not proven by native sampler records."}


def audit_sampler_pair(triton_dir, flashinfer_dir, reference_root, source_root, source_manifest):
    inputs = reference_inputs(reference_root)
    fixtures = inputs[1]
    arms, records, schedules, executions, prepared, sampled, workers = {}, {}, {}, {}, {}, {}, {}
    launches = {}
    for backend, directory in (("triton", triton_dir), ("flashinfer", flashinfer_dir)):
        root = Path(directory)
        launch, config, workers[backend] = provenance(
            root, backend, reference_root, source_root, source_manifest, inputs)
        launches[backend] = launch
        (records[backend], schedules[backend], executions[backend], prepared[backend],
         sampled[backend], streams, missing_local) = audit_trace(
             root, launch, config, fixtures, source_root)
        paths = [path for path in root.glob("*.json")]
        paths += [path for path in (root / "trace").rglob("*") if path.is_file()]
        arms[backend] = {"run": root.name, "backend": backend, "integrity_pass": True,
                         "global_unique_effective_records": len(records[backend]),
                         "true_decode_observations": sum(step > 0 for _, step in records[backend]),
                         "physical_records_per_rank": {rank: sum(key[0] == rank for key in
                             executions[backend]) for rank in (0, 1)},
                         "streams": streams, "launch_sha256": sha(root / "launch.json"),
                         "missing_same_rank_sample_contexts": missing_local,
                         "same_rank_execution_join_complete": not missing_local,
                         "input_files": {str(path.relative_to(root)).replace("\\", "/"):
                             {"sha256": sha(path), "bytes": path.stat().st_size}
                             for path in sorted(paths)},
                         "runtime_workers": [{key: worker[key] for key in (
                             "rank", "tp_rank", "tp_world_size", "gpu_uuid", "gpu_name",
                             "compute_capability", "nccl_version", "tp_device_backend")}
                             for worker in workers[backend]]}
    for name in ("helper_sha256", "source_files", "reference_lock_sha256", "fixture_sha256",
                 "fresh_model_files", "fresh_runtime_versions", "fresh_runtime_files"):
        require(launches["triton"][name] == launches["flashinfer"][name],
                "Same paired provenance: " + name)
    require(arms["triton"]["runtime_workers"] == arms["flashinfer"]["runtime_workers"],
            "Same paired physical hardware and NCCL runtime")
    pair = compare_pair(records, schedules, fixtures)
    pair["same_rank_execution_join_complete"] = all(
        arm["same_rank_execution_join_complete"] for arm in arms.values())
    pair["missing_local_sampler_contexts"] = {
        backend: arm["missing_same_rank_sample_contexts"] for backend, arm in arms.items()}
    pair["native_starting_condition_gate_pass"] &= pair["same_rank_execution_join_complete"]
    return {"integrity_pass": True, "arms": arms, "fixtures": fixtures,
            "records_by_arm": records, "schedules_by_arm": schedules,
            "execution_records_by_arm": executions, "prepared_by_arm": prepared,
            "sample_contexts_by_arm": sampled,
            "paircomparison": pair,
            "auditor_sha256": sha(Path(__file__)), "source_head": HEAD,
            "scope": inputs[0]["scope"],
            "eos_observation_scope": EOS_SCOPE,
            "limitations": ["Forced token histories cannot measure natural stopping or accuracy.",
                            "Only tokenizer EOS248046 raw logits are observed.",
                            "Explicit sync/eager localization differs from the original HTTP default-async protocol."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("triton-dir", "flashinfer-dir", "reference-root", "source-root", "source-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit_sampler_pair(args.triton_dir, args.flashinfer_dir, args.reference_root,
                                args.source_root, args.source_manifest)
    excluded = {"records_by_arm", "execution_records_by_arm", "prepared_by_arm", "sample_contexts_by_arm"}
    report = {name: value for name, value in result.items() if name not in excluded}
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"integrity_pass": True, "starting_condition_gate_pass":
                      result["paircomparison"]["native_starting_condition_gate_pass"],
                      "observations_per_arm": 7378, "output": str(args.output)}))


if __name__ == "__main__":
    main()
