"""CPU-only audit of the three short-prefix diagnostic arms.

This consumer retains NOGO when observed initial conditions differ. It reports
propagated numerical differences, without calling them an integration bug,
accuracy regression, or a speedup. It never imports vLLM or launches a GPU.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import traceback

INDICES = (198, 206, 209, 228, 255, 285, 292, 318)
GDN_LAYERS = tuple(layer for layer in range(64) if layer % 4 != 3)
ATTENTION_LAYERS = tuple(range(3, 64, 4))
STEPS = (1, 18, 19)
ARMS = {"A": ("triton", "triton"), "B": ("flashinfer", "triton"),
        "C": ("flashinfer", "flashinfer")}
REFERENCE_SHA = "d293b322876e8236f7799a688221ddd0d10e79f8046c6601db40fa6316421e78"
REFERENCE_LOCK_SHA = "c56bc5f24109901190769b0047b809da4a7d57448884442afd6a1a507924c4bb"
HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
INPUT_PROCESSOR_SHA = "97f66659814177d3f7b72b959f09201cbdd1794de21ec1542fe8970f18a1a2b6"
SAMPLING_PARAMS_SHA = "005eeca282874cde09864133f2100de6ae985439b93e50891194e8447a265999"
PRODUCER_SHAS = {
    "cpu_divergence_contracts.py": "e397c9bbe2eee389b83f7db844db84f1a0f90a25a94a87d4ac7a065173d14dcc",
    "cpu_short_force.py": "e57cfa6812393c820cf3f10d7f1f725c2ee74e7ac7b8a026af3f6aef4d23b2d8",
    "cpu_short_normalize.py": "78a73709b661285079cd7e1a1ad57dbed453e34d24456930357c6d35dd707940",
    "divergence_bootstrap.py": "cadc35587c2d53bf0fa0b9003846037f7f9a1f6c3df416203c0dbe4e7e83f82f",
    "divergence_capture.py": "0e506d302e05df30ffd1aff3399dc626c2844bed0d5034558a75147fbfee8f78",
    "divergence_contract.py": "7b72a6c47f3b0e4cf2abad95db2a3432567213f4ce7e709d1961595c88d7064d",
    "divergence_leaf.py": "406635cbb25377ec643b3e83734f0ae07e691a1e775e4144ecec646872e5ddb2",
    "short_driver.py": "8a2a8012b7dcdb4e939d5c30b7ef5bf07e63c832d863c0df9e95db6025692376",
    "short_force.py": "b040055e49fa3e6069be76aba7b0348c29e4f85b6fe398a16dc05d6936ea04c8",
    "short_normalize.py": "0cfc982ff53a3f29b7b008c2d0788b680a7c6f922132497490c2d821462690bf",
    "short_site/sitecustomize.py": "b41924f6c0304a2e356be54dd74a4c556ebc0712350d82d4592da0c389d32c5f",
}
SOURCE_SHAS = {
    "vllm/v1/engine/input_processor.py": INPUT_PROCESSOR_SHA,
    "vllm/sampling_params.py": SAMPLING_PARAMS_SHA,
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py":
        "d4056cb0105ca7c13fb3fa41be8f8e2ba3a2ea4f1ac81115abe3f3f4e6b3b45d",
    "vllm/v1/worker/gpu/model_runner.py":
        "48ae64b57dae7e26599df3e2a967a5f3ac0ad91861f7b879da989da0f5c67cf3",
    "vllm/v1/worker/gpu/input_batch.py":
        "96163c486368f2cf0e539fe68eaa06f4df3ac814da8f7a92af2b04242e01bc6d",
    "vllm/model_executor/layers/attention/attention.py":
        "bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2",
    "vllm/v1/attention/backends/flash_attn.py":
        "131be4f05c0d35e5b615eb66f32280cffaa4e5794fbe34682580fb5bfe50ce3e",
    "vllm/v1/attention/backends/triton_attn.py":
        "da4fbd1a94d84f93696463c0661638e976fc1fca4bff1fa1d5bb23bc585d2809",
    "vllm/v1/worker/gpu/sample/trace_replay.py":
        "d2a7d3b3b55379efabe4740a4b7f14030126dc11be36f6ea974164ce42316a50",
}
SCOPE = (
    "CPU inspection of bounded nineteen-token native-prefix intervention. "
    "A request-bound unsupported InputProcessor hook bypasses native trace "
    "termination normalization after ordinary tokenizer/generation updates. "
    "Initial equality is only the observed prompt KV, conv/recurrent and raw "
    "prefill logits. Captured later tensors propagated separately. No whole "
    "hidden-state equality, full-pool replay, accuracy, performance or proven "
    "integration-bug claim."
)


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(tokens):
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode()
    ).hexdigest()


def new_json(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def events(root):
    result = []
    for path in sorted((root / "trace").glob("events-pid-*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf8").splitlines(), 1
        ):
            value = json.loads(line)
            value["_source"] = {"file": path.name, "line": line_number}
            result.append(value)
    require(result, "Missing actual event streams")
    require(not any(row.get("event") == "short_failure" for row in result),
            "Worker recorded a diagnostic failure")
    return result


def get_eid(row):
    return int(row["execution_id"])


def validate_reference(root, lock):
    """Bind every retained reference byte before trusting its model/runtime."""
    for name, expected in lock["reference_files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Unsafe frozen reference member")
        require(sha(root / relative) == expected, "Frozen reference mismatch: " + name)
    completed = read(root / "completed.json")
    require(completed["launch_sha256"] == sha(root / "launch.json"),
            "Immutable reference launch receipt differs")
    return completed


def validate_fixture(fixtures, reference):
    actual = {row["index"]: row for row in reference["examples"]}
    require(set(actual) == set(INDICES), "Immutable reference indices differ")
    for index in INDICES:
        row, original = fixtures[index], actual[index]
        require(row["prompt_token_ids"] == original["prompt_token_ids"]
                and row["reference_token_ids"] == original["token_ids"]
                and row["max_tokens"] == 3500,
                "Diagnostic fixture differs from the immutable actual reference")
        require(digest(original["prompt_token_ids"]) == original["prompt_token_ids_sha256"]
                and digest(original["token_ids"]) == original["token_ids_sha256"],
                "Immutable reference token receipts differ")


def validate_resolved_sampling(original, actual, reference_prefix):
    expected = {
        **original,
        "_eos_token_id": 248046,
        "_all_stop_token_ids": [248044, 248046],
        "stop_token_ids": [248044],
        "watermarking": False,
        # Pinned RequestOutputKind is Enum. The two frozen serializers use
        # enum.name and str(enum), respectively; only this spelling differs.
        "output_kind": "RequestOutputKind.CUMULATIVE",
    }
    require(original["output_kind"] == "CUMULATIVE"
            and original["trace_decode_token_ids"] == list(reference_prefix)
            and expected["max_tokens"] == 3500
            and expected["min_tokens"] == 0
            and expected["ignore_eos"] is False,
            "Original nineteen-prefix free sampling contract differs")
    require(actual == expected,
            "Resolved full sampling struct differs beyond native EOS/watermark updates")


def validate_source_bindings(launch, cfg, stream, capture_helper):
    """Bind attention through its actual guarded import event, not a fake key."""
    attention_path = "vllm/model_executor/layers/attention/attention.py"
    for name, expected in SOURCE_SHAS.items():
        if name == attention_path:
            continue
        require(launch["source_files"][name] == expected
                and cfg["source_files"][name] == expected,
                "Pinned actual source mismatch: " + name)
    # This frozen helper reads the real imported module bytes, rejects an SHA
    # mismatch, patches it and only then emits short_capture_source.
    require(sha(capture_helper) == PRODUCER_SHAS["divergence_capture.py"],
            "Actual capture guard helper differs")
    tree = ast.parse(Path(capture_helper).read_text(encoding="utf8"))
    source_map = next(
        ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_SOURCE_SHAS"
                for target in node.targets)
    )
    expected_modules = {
        "vllm.v1.worker.gpu.model_runner":
            SOURCE_SHAS["vllm/v1/worker/gpu/model_runner.py"],
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn":
            SOURCE_SHAS["vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"],
        "vllm.model_executor.layers.attention.attention":
            SOURCE_SHAS[attention_path],
    }
    require(source_map == expected_modules, "Frozen capture internal source guards differ")
    worker_ranks = {}
    for event in stream:
        if event.get("event") != "native_request_registered":
            continue
        pid, rank = int(event["pid"]), int(event["rank"])
        require(rank in (0, 1) and worker_ranks.get(pid, rank) == rank,
                "Native worker PID/rank identity changed")
        worker_ranks[pid] = rank
    require(len(worker_ranks) == 2 and set(worker_ranks.values()) == {0, 1},
            "Source evidence lacks both actual native worker identities")
    observed = {}
    for event in stream:
        if event.get("event") != "short_capture_source":
            continue
        module, pid = event["module"], int(event["pid"])
        require(module in expected_modules
                and event["sha256"] == expected_modules[module]
                and event["arm"] == cfg["arm"],
                "Actual guarded module import SHA differs")
        if pid not in worker_ranks:
            continue
        key = (pid, module)
        require(key not in observed, "Guarded module source was emitted twice")
        observed[key] = {
            "pid": pid, "registered_tp_rank": worker_ranks[pid],
            "rank_at_import": event["rank"], "module": module,
            "sha256": event["sha256"], "source": event["_source"],
        }
    require(set(observed) == {
        (pid, module) for pid in worker_ranks for module in expected_modules
    }, "Actual guarded source imports are incomplete for either TP worker")
    return [observed[key] for key in sorted(observed)]


def runtime_contract(root, configured, expected_gpu_uuids):
    snapshots = {}
    for moment in ("before", "after"):
        runtime = read(root / ("runtime-" + moment + ".json"))
        require(runtime["engine_core_class"] == "InprocClient",
                "Resolved engine core differs")
        workers = runtime["worker_ranks"]
        require(len(workers) == 2 and {row["rank"] for row in workers} == {0, 1},
                "Actual TP workers incomplete")
        normalized = {}
        for row in workers:
            cfg = row["config"]
            model, parallel = cfg["model_config"], cfg["parallel_config"]
            cache, scheduler = cfg["cache_config"], cfg["scheduler_config"]
            require(row["runner_class"] == "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                    and row["tp_world_size"] == 2 and row["tp_rank"] == row["rank"]
                    and row["tp_device_backend"] == "nccl"
                    and row["gpu_name"] == "NVIDIA L20"
                    and row["compute_capability"] == [8, 9], "Resolved GPU/runner differs")
            require(row["gpu_uuid"].removeprefix("GPU-")
                    == expected_gpu_uuids[row["rank"]].removeprefix("GPU-"),
                    "Actual rank GPU UUID differs from the explicitly frozen placement")
            require(model["dtype"] == "torch.bfloat16" and model["seed"] == 42
                    and model["enforce_eager"] is True
                    and model["enable_trace_replay"] is True
                    and model["max_model_len"] == 4096
                    and parallel["tensor_parallel_size"] == 2
                    and parallel["pipeline_parallel_size"] == 1
                    and parallel["enable_batch_sharded_sampling"] is False
                    and scheduler["async_scheduling"] is False
                    and cache["enable_prefix_caching"] is False
                    and cache["mamba_ssm_cache_dtype"] == "float32"
                    and cache["num_gpu_blocks_override"] == 128
                    and cfg["compilation_config"]["cudagraph_mode"] == "NONE"
                    and cfg["kernel_config"]["gdn_decode_backend"] == configured,
                    "Actual resolved model/cache/scheduler/leaf constructor differs")
            normalized[row["rank"]] = {
                name: row[name] for name in (
                    "runner_class", "tp_world_size", "tp_rank", "tp_device_backend",
                    "gpu_uuid", "gpu_name", "compute_capability", "nccl_version",
                )
            }
        require(len({row["gpu_uuid"] for row in normalized.values()}) == 2,
                "TP ranks do not bind two distinct GPUs")
        snapshots[moment] = normalized
    require(snapshots["before"] == snapshots["after"], "GPU/TP runtime changed")
    return snapshots["before"]


def index_arm(root, arm, expected_gpu_uuids):
    cfg, launch = read(root / "short-config.json"), read(root / "launch.json")
    completed, aborted = read(root / "completed.json"), read(root / "abort.json")
    enqueued, shutdown = read(root / "enqueued.json"), read(root / "shutdown.json")
    require(completed["status"] == "completed" and completed["arm"] == arm,
            "Incomplete diagnostic arm")
    require(completed["intentional_abort"] is True
            and shutdown["status"] == "shutdown-returned"
            and shutdown["diagnostic_completed"] is True, "Missing shutdown")
    require(aborted["status"] == "public-abort-returned"
            and aborted["request_id_kind"] == "external"
            and aborted["after_engine_steps"] == 20
            and aborted["further_model_steps"] == 0
            and aborted["unfinished_after"] == 0, "Abort boundary differs")
    configured, leaf = ARMS[arm]
    require(cfg["arm"] == launch["arm"] == arm
            and cfg["configured_backend"] == configured
            and cfg["actual_leaf_backend"] == leaf, "Wrong diagnostic leaf")
    require(cfg["prefix_len"] == 19 and cfg["observe_last_step"] == 19
            and cfg["snapshot_steps"] == list(STEPS)
            and cfg["snapshot_target"] == 255, "Changed bounded protocol")
    require(launch["source_head"] == HEAD
            and launch["reference_completed_sha256"] == REFERENCE_SHA,
            "Wrong source or reference")
    require(launch["config_sha256"] == sha(root / "short-config.json")
            and completed["launch_sha256"] == sha(root / "launch.json"),
            "Launch/config receipt mismatch")
    require(launch["helper_sha256"] == PRODUCER_SHAS,
            "Producer helper bytes differ from the independently frozen v5")
    resolved_runtime = runtime_contract(root, configured, expected_gpu_uuids)
    for name, expected in launch["helper_sha256"].items():
        require(sha(root / "helpers" / name) == expected, "Copied helper mismatch")
    for name, expected in (
        ("fixture.json", launch["fixture_sha256"]),
        ("reference-lock.json", launch["reference_lock_sha256"]),
    ):
        require(sha(root / name) == expected, "Frozen input mismatch: " + name)
    fixtures = {row["index"]: row for row in read(root / "fixture.json")["examples"]}
    require(tuple(fixtures) == INDICES, "Changed eight original prompt indices")
    lock = read(root / "reference-lock.json")
    require(sha(root / "reference-lock.json") == REFERENCE_LOCK_SHA
            and lock["reference_files"]["completed.json"] == REFERENCE_SHA
            and sha(root / "reference/completed.json") == REFERENCE_SHA,
            "Immutable actual FI eager reference mismatch")
    reference = validate_reference(root / "reference", lock)
    validate_fixture(fixtures, reference)
    prior = read(root / "reference/launch.json")
    require(launch["sampling_options"] == prior["sampling_options"]
            and launch["model_files"] == prior["model_files"]
            and launch["runtime_versions"] == prior["runtime_versions"]
            and launch["runtime_files"]
            == read(root / "reference/runtime-binding.json")["modules"],
            "Frozen model/runtime/sampling bindings differ")
    expected_options = {
        **prior["model_options"],
        "kernel_config": {
            **prior["model_options"]["kernel_config"],
            "gdn_decode_backend": configured,
        },
        "enable_trace_replay": True,
    }
    require(launch["options"] == expected_options,
            "Original engine profile changed beyond backend and native trace")
    external = enqueued["external_request_ids"]
    require(enqueued["all8_before_first_step"] is True and len(external) == 8
            and set(external) == {str(index) for index in INDICES},
            "Original public requests were not all queued")
    by_external = enqueued["indices_by_external_request_id"]
    internal = enqueued["internal_request_ids_by_external_id"]
    require(set(by_external) == set(internal) == set(external)
            and len(set(internal.values())) == 8
            and set(by_external.values()) == set(INDICES), "Invalid ID mapping")
    internal_index = {internal[key]: by_external[key] for key in external}
    outputs = {row["index"]: row for row in completed["examples"]}
    require(set(outputs) == set(INDICES), "Missing actual public outputs")
    for index, value in outputs.items():
        require(value["request_id"] in by_external
                and by_external[value["request_id"]] == index
                and value["prompt_token_ids"] == fixtures[index]["prompt_token_ids"]
                and len(value["token_ids"]) == 20
                and value["token_ids"][:19]
                == fixtures[index]["reference_token_ids"][:19]
                and value["token_ids_sha256"] == digest(value["token_ids"]),
                "Actual output identity/prefix mismatch")

    stream = events(root)
    source_evidence = validate_source_bindings(
        launch, cfg, stream, root / "helpers/divergence_capture.py"
    )
    raw_sampling = {
        row["index"]: row["params"]
        for row in read(root / "sampling-runtime.json")["examples"]
    }
    require(set(raw_sampling) == set(INDICES), "Raw request sampling structs incomplete")
    normalization_events = [
        row for row in stream if row.get("event") == "short_trace_normalization_bypassed"
    ]
    require(len(normalization_events) == 8, "Exactly eight request-scoped bypasses required")
    normalized = {}
    normalization_sources = [
        row for row in stream if row.get("event") == "short_normalization_source"
    ]
    require(len(normalization_sources) == 1
            and normalization_sources[0]["source_sha256"] == INPUT_PROCESSOR_SHA
            and normalization_sources[0]["sampling_source_sha256"] == SAMPLING_PARAMS_SHA
            and normalization_sources[0]["helper_sha256"] == PRODUCER_SHAS["short_normalize.py"],
            "Actual normalization module/helper binding differs")
    for event in normalization_events:
        index = event["index"]
        require(index in INDICES and index not in normalized
                and by_external[event["external_req_id"]] == index
                and event["input_processor_sha256"] == INPUT_PROCESSOR_SHA
                and event["unchanged_after"] is True
                and event["prompt_token_ids"] == fixtures[index]["prompt_token_ids"]
                and event["prompt_length"] == len(fixtures[index]["prompt_token_ids"]),
                "Normalization bypass is not exactly bound to an original request")
        validate_resolved_sampling(
            raw_sampling[index], event["resolved_sampling_params"],
            fixtures[index]["reference_token_ids"][:19],
        )
        normalized[index] = event["resolved_sampling_params"]
    registrations, prepared, sampled, initial, capture_events = {}, {}, {}, {}, {}
    dispatch = Counter()
    profiles = Counter()
    native_records = {}
    for event in stream:
        kind = event.get("event")
        if kind == "short_excluded_profiling_leaf":
            require(event["actual_profile_leaf_backend"] == configured,
                    "Excluded profiling changed the constructor backend")
            profiles[event["rank"]] += 1
            continue
        if kind not in {
            "native_request_registered", "short_prepared_batch", "sample_batch",
            "short_initial_state", "short_leaf_capture", "short_actual_leaf_dispatch",
        }:
            continue
        rank = int(event["rank"])
        require(rank in (0, 1), "Unexpected TP rank")
        if kind == "native_request_registered":
            key = (rank, event["index"])
            require(key not in registrations
                    and internal_index[event["req_id"]] == event["index"]
                    and event["native_trace_len"] == 19
                    and event["native_ignore_eos"] is False
                    and event["native_max_tokens"] == 3500
                    and event["native_min_tokens"] == 0
                    and event["native_eos_token_id"] == 248046
                    and event["native_stop_token_ids"] == [248044]
                    and event["native_all_stop_token_ids"] == [248044, 248046],
                    "Actual registration/terminal contract differs")
            registrations[key] = event
            continue
        eid = get_eid(event)
        if kind == "short_prepared_batch":
            key = (rank, eid)
            require(key not in prepared, "Repeated actual prepared execution")
            require(len(event["req_ids"]) == 8
                    and set(event["req_ids"]) == set(internal_index),
                    "Actual batch does not contain all eight identities")
            prepared[key] = event
        elif kind == "sample_batch":
            key = (rank, eid)
            require(key not in sampled and event["sampling_sharded"] is False
                    and event["actual_batch_size"] == 8
                    and event["actual_req_ids"] == event["global_req_ids"]
                    and len(event["records"]) == 8, "Changed native sampler batch")
            sampled[key] = event
            for row in event["records"]:
                index, step = int(row["index"]), int(row["step"])
                rkey = (rank, index, step)
                require(rkey not in native_records and index in INDICES
                        and 0 <= step <= 19
                        and internal_index[row["req_id"]] == index,
                        "Duplicate/unknown native position")
                require(row["input_prefix_sha256"] == digest(
                    fixtures[index]["reference_token_ids"][:step]),
                    "Committed history hash differs")
                returned = row["native_returned_token_id"]
                if step < 19:
                    require(row["trace_forced"] is True
                            and returned == fixtures[index]["reference_token_ids"][step],
                            "Native fixed-prefix mismatch")
                else:
                    require(row["trace_forced"] is False
                            and row["step19_native_passthrough"] is True
                            and returned == row["original_sampler_token_id"]
                            == outputs[index]["token_ids"][19],
                            "Output20 differs from the actual natural sample")
                if step in (18, 19):
                    raw_path = root / "trace" / row["raw_logits_file"]
                    require(raw_path.is_file()
                            and raw_path.stat().st_size == row["raw_logits_bytes"]
                            == 4 * row["raw_logits_shape"][0]
                            and sha(raw_path) == row["raw_logits_full_sha256"],
                            "Raw full-vocabulary binary receipt mismatch")
                native_records[rkey] = {**row, "execution_id": eid}
        elif kind == "short_initial_state":
            key = (event["kind"], rank, int(event["layer"]), int(event["index"]))
            require(key not in initial and event["step"] == 1,
                    "Duplicate/mispositioned initial observation")
            initial[key] = event
        elif kind == "short_leaf_capture":
            key = (rank, int(event["layer"]), int(event["step"]))
            require(key not in capture_events and event["index"] == 255,
                    "Repeated or wrong-target capture")
            capture_events[key] = event
        else:
            require(event["configured_backend"] == configured
                    and event["actual_leaf_backend"] == leaf,
                    "Actual dispatch differs from the planned kernel")
            dispatch[(rank, int(event["layer"]), int(event["step"]))] += 1

    expected_positions = {
        (rank, index, step) for rank in (0, 1) for index in INDICES
        for step in range(20)
    }
    require(set(native_records) == expected_positions, "Native positions incomplete")
    for index in INDICES:
        for step in range(20):
            first, second = (native_records[(rank, index, step)] for rank in (0, 1))
            for name in (
                "original_sampler_token_id", "native_returned_token_id",
                "raw_logits_full_sha256", "input_prefix_sha256",
            ):
                require(first[name] == second[name],
                        "Unsharded physical rank duplicate differs: " + name)
    require(set(registrations) == {(r, i) for r in (0, 1) for i in INDICES},
            "Registrations incomplete")
    require(set(prepared) == set(sampled)
            and len(prepared) == 40, "Actual execution/sample joins incomplete")
    expected_dispatch = {
        (rank, layer, step) for rank in (0, 1) for layer in GDN_LAYERS
        for step in range(1, 20)
    }
    require(set(dispatch) == expected_dispatch and all(n == 1 for n in dispatch.values()),
            "Actual per-layer dispatch coverage is not 19*48 perrank")
    expected_initial = {
        (kind, rank, layer, index)
        for kind, layers in (("conv", GDN_LAYERS), ("recurrent", GDN_LAYERS),
                             ("kv", ATTENTION_LAYERS))
        for rank in (0, 1) for layer in layers for index in INDICES
    }
    require(set(initial) == expected_initial, "1792 initial state records incomplete")
    for key, event in initial.items():
        expected_dtype = "torch.float32" if key[0] == "recurrent" else "torch.bfloat16"
        require(event["value"]["dtype"] == expected_dtype
                and event["value"]["num_bytes"] > 0,
                "Actual initialized state dtype/size differs")
    expected_captures = {
        (rank, layer, step) for rank in (0, 1) for layer in GDN_LAYERS for step in STEPS
    }
    require(set(capture_events) == expected_captures, "288 capture events incomplete")
    captures = {}
    for path in sorted((root / "trace/captures").glob("*.json")):
        sidecar = read(path)
        join = sidecar["runtime_join"]
        key = (sidecar["rank"], join["layer"], join["actual_target_step"])
        require(key in expected_captures and key not in captures,
                "Unknown or repeated sidecar")
        require(sidecar["arm"] == arm
                and sidecar["configured_backend"] == configured
                and sidecar["actual_leaf_backend"] == leaf,
                "Sidecar kernel/dtype specialization identity differs")
        binary = path.with_suffix(".pt")
        require(binary.is_file() and sha(binary) == sidecar["binary_sha256"]
                == capture_events[key]["binary_sha256"]
                and binary.stat().st_size == sidecar["binary_size"],
                "Snapshot binary/event binding differs")
        rank, _, step = key
        eid = int(join["execution_id"])
        event = prepared[(rank, eid)]
        require(eid == capture_events[key]["execution_id"]
                and join["actual_target_step"] == event["actual_target_step"]
                and event["capture_selected"] is True,
                "Capture does not join its actual prepared execution")
        require(len(join["rows"]) == 8
                and join["actual_pages"] == [
                    row["gdn_state_page"] for row in join["rows"]
                ], "Actual metadata page/row identity differs")
        for row in join["rows"]:
            index = row["index"]
            native = native_records[(rank, index, step)]
            require(row["req_id"] == native["req_id"]
                    and row["request_state_slot"] == native["request_state_slot"]
                    and row["input_prefix_sha256"] == native["input_prefix_sha256"]
                    and row["input_token"] == native["actual_input_token"]
                    and row["input_position"] == native["actual_input_position"]
                    and row["step"] == step and native["execution_id"] == eid,
                    "Snapshot history/position/page context does not join sampler")
        captures[key] = {
            "sidecar": sidecar, "binary": binary, "sidecar_file": path,
        }
    require(set(captures) == expected_captures, "288 actual binaries incomplete")
    require({entry["binary"].name for entry in captures.values()} == {
        path.name for path in (root / "trace/captures").glob("*.pt")
    }, "Unknown extra snapshot binaries")
    for key, event in initial.items():
        _, rank, _, index = key
        native = native_records[(rank, index, 1)]
        require(event["execution_id"] == native["execution_id"]
                and event["req_id"] == native["req_id"]
                and event["request_state_slot"] == native["request_state_slot"],
                "Initial state does not join actual first decode")
        if event["kind"] == "kv":
            require(event["valid_tokens"] == len(fixtures[index]["prompt_token_ids"])
                    and event["value"]["shape"][0] == event["valid_tokens"],
                    "Initial KV includes uninitialized/current positions")
    normalized_schedule = {}
    for (rank, eid), event in prepared.items():
        normalized_schedule[(rank, eid)] = {
            "indices": [internal_index[req_id] for req_id in event["req_ids"]],
            "num_scheduled_tokens": event["num_scheduled_tokens"],
            "actual_target_step": event["actual_target_step"],
        }
    for rank, eid in normalized_schedule:
        if rank == 0:
            require(normalized_schedule[(0, eid)] == normalized_schedule[(1, eid)],
                    "Actual TP execution schedules differ")
    return {
        "root": root, "launch": launch, "cfg": cfg, "fixtures": fixtures,
        "initial": initial, "records": native_records, "captures": captures,
        "schedule": normalized_schedule, "registrations": registrations,
        "summary": {
            "arm": arm, "initial_records": len(initial),
            "prefill_hashes": sum(key[2] == 0 for key in native_records),
            "native_positions": len(native_records), "captures": len(captures),
            "actual_inference_leaf_calls_per_rank": {
                str(rank): sum(n for (r, _, _), n in dispatch.items() if r == rank)
                for rank in (0, 1)
            },
            "excluded_profiling_leaf_calls_per_rank": dict(profiles),
            "choices_at19": {
                str(index): native_records[(0, index, 19)]["original_sampler_token_id"]
                for index in INDICES
            },
            "launch_sha256": sha(root / "launch.json"),
            "completed_sha256": sha(root / "completed.json"),
            "resolved_runtime": resolved_runtime,
            "request_bound_normalization_bypasses": len(normalized),
            "effective_model_eos_ids": [248044, 248046],
            "actual_guarded_module_source_evidence": source_evidence,
        },
    }


def tensor_bytes(tensor, torch):
    value = tensor.detach().cpu().contiguous()
    require(value.device.type == "cpu", "Consumer tensor is not CPU")
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_metrics(a, b, torch):
    require(a.shape == b.shape and torch.isfinite(a).all().item()
            and torch.isfinite(b).all().item(), "Nonfinite or shape-mismatched tensor")
    x, y = a.double(), b.double()
    delta = x - y
    denominator = float(torch.linalg.vector_norm(y).item())
    difference = float(torch.linalg.vector_norm(delta).item())
    relative = difference / denominator if denominator else (0.0 if not difference else None)
    return {
        "value_equal_fp32": torch.equal(a.float(), b.float()),
        "same_dtype": a.dtype == b.dtype,
        "max_abs": float(delta.abs().max().item()),
        "relative_l2_to_second": relative,
        "zero_denominator": denominator == 0,
    }


def observed_initial_comparison(arms):
    """Return exact observed mismatches without substituting tolerant equality."""
    initial_differences, prefill_differences, schedule_differences = [], [], []
    for pair in (("A", "B"), ("B", "C"), ("A", "C")):
        first, second = (arms[key] for key in pair)
        label = "-".join(pair)
        for key in first["initial"]:
            a, b = first["initial"][key]["value"], second["initial"][key]["value"]
            if (a["sha256"], a["shape"], a["dtype"]) != (
                b["sha256"], b["shape"], b["dtype"]
            ):
                initial_differences.append({"pair": label, "key": key,
                                            "first": a, "second": b})
        for rank in (0, 1):
            for index in INDICES:
                key = (rank, index, 0)
                if first["records"][key]["raw_logits_full_sha256"] != (
                    second["records"][key]["raw_logits_full_sha256"]
                ):
                    prefill_differences.append({"pair": label, "rank": rank,
                                                "index": index})
        if first["schedule"] != second["schedule"]:
            schedule_differences.append(label)
    gate = not (initial_differences or prefill_differences or schedule_differences)
    return {
        "gate": gate, "initial_differences": initial_differences,
        "prefill_differences": prefill_differences,
        "schedule_differences": schedule_differences,
    }


def compare(arms, output):
    import torch
    require(not torch.cuda.is_initialized() and not torch.cuda.is_available(),
            "Run this consumer in a CPU-only container without GPU access")
    torch.set_num_threads(1)
    initial = observed_initial_comparison(arms)
    gate = initial["gate"]
    details, maxima, relative_maxima, parameter_differences = [], {}, {}, []
    names = ("mixed_qkv", "a", "b", "A_log", "dt_bias", "target_state_before",
             "target_state_after", "out")
    for key in sorted(arms["A"]["captures"]):
        values = {}
        for arm, indexed in arms.items():
            entry = indexed["captures"][key]
            value = torch.load(entry["binary"], map_location="cpu", weights_only=True)
            require(set(value) == set(names), "Unexpected snapshot payload")
            for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias"):
                metadata = entry["sidecar"]["original_tensors"][name]
                require(hashlib.sha256(tensor_bytes(value[name], torch)).hexdigest()
                        == metadata["sha256"] and list(value[name].shape) == metadata["shape"]
                        and str(value[name].dtype) == metadata["dtype"],
                        "PT tensor differs from its bound native sidecar")
            require(str(value["dt_bias"].dtype) == (
                "torch.bfloat16" if arm == "A" else "torch.float32"
            ), "Actual bias promotion differs")
            require(str(value["A_log"].dtype) == "torch.float32"
                    and str(value["target_state_before"].dtype) == "torch.float32"
                    and str(value["target_state_after"].dtype) == "torch.float32",
                    "Actual parameter/state dtype differs")
            values[arm] = value
        case = {"rank": key[0], "layer": key[1], "step": key[2], "pairs": {}}
        for pair in (("A", "B"), ("B", "C"), ("A", "C")):
            label = "-".join(pair)
            case["pairs"][label] = {}
            for name in names:
                metric = tensor_metrics(values[pair[0]][name], values[pair[1]][name], torch)
                case["pairs"][label][name] = metric
                if name in ("A_log", "dt_bias") and not metric["value_equal_fp32"]:
                    parameter_differences.append({"pair": label, "key": key, "name": name})
                maxima_key = label + ":" + name
                existing = maxima.get(maxima_key)
                if existing is None or metric["max_abs"] > existing["max_abs"]:
                    maxima[maxima_key] = {"key": key, **metric}
                relative = metric["relative_l2_to_second"]
                existing_relative = relative_maxima.get(maxima_key)
                if (relative is not None and (
                    existing_relative is None
                    or relative > existing_relative["relative_l2_to_second"]
                )):
                    relative_maxima[maxima_key] = {"key": key, **metric}
        details.append(case)
    raw_cases = []
    for rank in (0, 1):
        for index in INDICES:
            for step in (18, 19):
                key = (rank, index, step)
                vectors, raw = {}, {"rank": rank, "index": index, "step": step,
                                    "arms": {}, "pairs": {}}
                for arm, indexed in arms.items():
                    row = indexed["records"][key]
                    path = indexed["root"] / "trace" / row["raw_logits_file"]
                    vector = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32)
                    require(list(vector.shape) == row["raw_logits_shape"]
                            and torch.isfinite(vector).all().item(), "Invalid raw logits")
                    vectors[arm] = vector
                    for token, expected in row["selected_candidate_raw_logits"].items():
                        require(float(vector[int(token)].item()) == expected,
                                "Candidate raw-logit receipt differs")
                    require(float(vector[row["original_sampler_token_id"]].item())
                            == row["original_sampler_raw_logit"], "Native raw-logit receipt differs")
                    raw["arms"][arm] = {
                        "native_choice": row["original_sampler_token_id"],
                        "returned_choice": row["native_returned_token_id"],
                        "raw_argmax": int(vector.argmax().item()),
                        "71072_minus_43659": float((vector[71072] - vector[43659]).item()),
                        "raw_full_sha256": row["raw_logits_full_sha256"],
                    }
                for pair in (("A", "B"), ("B", "C"), ("A", "C")):
                    raw["pairs"]["-".join(pair)] = tensor_metrics(
                        vectors[pair[0]], vectors[pair[1]], torch
                    )
                raw_cases.append(raw)
    gate = gate and not parameter_differences
    reproduction = {
        "A_expected71072": arms["A"]["records"][(0, 255, 19)]["original_sampler_token_id"] == 71072,
        "C_expected43659": arms["C"]["records"][(0, 255, 19)]["original_sampler_token_id"] == 43659,
    }
    summary = {
        "status": "OBSERVED_INITIAL_PASS" if gate else "NOGO",
        "first_flip_reproduction_pass": all(reproduction.values()),
        "bounded_interpretation_ready": gate and all(reproduction.values()),
        "scope": SCOPE, "gpu_execution": False,
        "cpu_torch_threads": 1,
        "arms": {key: value["summary"] for key, value in arms.items()},
        "initial_differences": initial["initial_differences"],
        "full_prefill_logit_differences": initial["prefill_differences"],
        "normalized_schedule_differences": initial["schedule_differences"],
        "parameter_value_differences_fp32": parameter_differences,
        "captured_keys_per_arm": len(details), "raw_logit_cases": len(raw_cases),
        "max_abs_cases": maxima,
        "max_relative_l2_cases": relative_maxima,
        "q255_step19_reproduction": reproduction,
        "scope_limits": [
            "Other requests also have forced FI histories0..18.",
            "Profiling calls are excluded; real inference calls are separately counted.",
            "KV observation covers initialized prompt positions, not later KV or all hidden states.",
            "Snapshots contain target propagated recurrent page, not full B8 pool replay.",
            "Matched observed starts permit a bounded numerical-contribution comparison only.",
            "Alog and bias value equality after FP32 conversion does not erase dtype specialization.",
            "Host GPU ownership/release remains an independent supervisor audit.",
        ],
    }
    new_json(output / "capture-comparisons.json", {"cases": details, "scope": SCOPE})
    new_json(output / "raw-logit-comparisons.json", {"cases": raw_cases, "scope": SCOPE})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-A", default="short-A")
    parser.add_argument("--arm-B", default="short-B")
    parser.add_argument("--arm-C", default="short-C")
    parser.add_argument("--expected-gpu-uuids", nargs=2, required=True,
                        help="Frozen UUIDs in physical TP rank0,rank1 order")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        arms = {
            arm: index_arm(
                args.matrix_root / getattr(args, "arm_" + arm), arm,
                args.expected_gpu_uuids,
            )
            for arm in ARMS
        }
        summary = compare(arms, args.output_dir)
        summary["consumer_sha256"] = sha(__file__)
        new_json(args.output_dir / "summary.json", summary)
        new_json(args.output_dir / "receipt.json", {
            "consumer_sha256": sha(__file__),
            "summary_sha256": sha(args.output_dir / "summary.json"),
            "capture_comparisons_sha256": sha(args.output_dir / "capture-comparisons.json"),
            "raw_logit_comparisons_sha256": sha(args.output_dir / "raw-logit-comparisons.json"),
            "status": summary["status"], "gpu_execution": False,
        })
        print(json.dumps({"status": summary["status"], "gpu_execution": False}))
        sys.exit(0 if summary["status"] == "OBSERVED_INITIAL_PASS" else 2)
    except BaseException as error:
        if not isinstance(error, SystemExit):
            new_json(args.output_dir / "failure.json", {
                "status": "INCOMPLETE_OR_INVALID", "exception": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(),
                "consumer_sha256": sha(__file__), "gpu_execution": False,
            })
        raise


if __name__ == "__main__":
    main()
