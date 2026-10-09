"""Validate late captures and their paired observational starting conditions.

CPU tensor loading is bounded to one snapshot pair. This is localization,
not an accuracy or throughput evaluation; full-attention KV is unobserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from late_plan import HEAD, digest, read, require, validate_inputs

LAYERS = tuple(i for i in range(64) if i % 4 != 3)
STEPS = (1, 18, 19, 1750, 3497, 3498, 3499)
BACKENDS = ("triton", "flashinfer")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def safe_relative(value):
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            "Safe relative captured path")
    return path


def verify_capture_file(directory, event):
    relative = safe_relative(event["file"])
    path = directory.joinpath(*relative.parts)
    require(path.is_file() and path.stat().st_size == event["bytes"] and
            sha(path) == event["sha256"], "Exact captured binary SHA and size")
    require(read(path.with_suffix(".json")) == event, "Sidecar equals actual capture event")
    return path


def expected_keys():
    return {(rank, layer, step) for rank in (0, 1)
            for layer in LAYERS for step in STEPS}


def assert_complete(keys):
    require(set(keys) == expected_keys(), "Exact 48 layers x two ranks x seven steps")


def sampler_contexts(root):
    prepared = {}
    sampled = {}
    for path in sorted((root / "trace").glob("events-pid-*.jsonl")):
        for event in rows(path):
            kind = event["event"]
            if kind not in ("late_prepared_batch", "sample_batch"):
                continue
            key = (event["rank"], event["late_execution_id"])
            table = prepared if kind == "late_prepared_batch" else sampled
            require(key not in table, "Unique rank/execution trace event")
            table[key] = event
    return prepared, sampled


def validate_join(meta, prepared, sampled, fixtures):
    rank = meta["rank"]
    join = meta["runtime_join"]
    execution_id = join["late_execution_id"]
    step = join["actual_target_step"]
    require(join["target_index"] == 255 and step in STEPS and
            meta["call_index"] == step, "Actual target step, not global call count")
    event = prepared[(rank, execution_id)]
    joined = join["rows"]
    require(event["capture_selected"] is True and event["actual_target_step"] == step,
            "Captured selected GPU target prefilter")
    require(event["req_ids"] == [row["req_id"] for row in joined] and
            event["num_reqs"] == len(joined) == meta["batch_size"] and
            event["num_scheduled_tokens"] == [1] * len(joined),
            "Exact actual prepared input row order")
    require([row["row"] for row in joined] == list(range(len(joined))),
            "Complete packed row order")
    own_sample = sampled.get((rank, execution_id))
    if own_sample is not None:
        require(own_sample["global_req_ids"] == event["req_ids"] and
                own_sample["model_batch_size_global"] == len(joined),
                "Same-rank native sampler actual global input batch")
    matches = [value for (other_rank, other_id), value in sampled.items()
               if other_id == execution_id and value["global_req_ids"] == event["req_ids"]]
    records = [record for value in matches for record in value["records"]]
    for row in joined:
        fixture = fixtures[row["index"]]
        row_step = row["step"]
        prefix = fixture["reference_token_ids"][:row_step]
        require(1 <= row_step < fixture["max_tokens"] and
                row["prompt_token_ids_sha256"] == digest(fixture["prompt_token_ids"]) and
                row["input_prefix_sha256"] == digest(prefix) and
                row["input_token"] == prefix[-1] and
                row["input_position"] == len(fixture["prompt_token_ids"]) + row_step - 1,
                "Exact frozen GPU history and actual input position")
        found = [record for record in records if record["index"] == row["index"]
                 and record["step"] == row_step and record["req_id"] == row["req_id"]]
        require(found, "Captured row has an exact native sampler execution record")
        for record in found:
            require(record["phase"] == "real_decode" and
                    record["global_row"] == row["row"] and
                    record["input_prefix_sha256"] == row["input_prefix_sha256"] and
                    record["actual_input_token"] == row["input_token"] and
                    record["actual_input_position"] == row["input_position"],
                    "Snapshot to actual native sampler row and history join")
            if record["recorded_sampling_rank"] == rank:
                require(record["request_state_slot"] == row["request_state_slot"],
                        "Same-rank persistent slot observation")
    targets = [row for row in joined if row["index"] == 255]
    require(len(targets) == 1 and targets[0]["step"] == step,
            "Unique actual target row at selected position")
    require(len({row["gdn_state_page"] for row in joined}) == len(joined) and
            all(row["gdn_state_page"] > 0 for row in joined),
            "Unique active actual GDN pages")
    return own_sample is not None


def index_arm(root, backend, fixtures):
    launch = read(root / "launch.json")
    config = read(root / "capture-config.json")
    require(launch["backend"] == backend and launch["source_head"] == HEAD,
            "Captured arm and reviewed source")
    require(sha(root / "capture-config.json") == launch["capture_config_sha256"],
            "Capture configuration bound to actual launch")
    require(config["initial_batch_size"] == 8 and config["early_calls"] == [] and
            config["sample_calls"] == [] and config["sample_layers"] == [],
            "Only actual selected-step captures")
    prepared, sampled = sampler_contexts(root)
    directory = root / "snapshots"
    events = [event for path in sorted(directory.glob("events-pid-*.jsonl"))
              for event in rows(path)]
    require(all(event["event"] in ("start_seen", "layer_registered", "capture")
                for event in events), "No capture failure or unknown event")
    indexed = {}
    workers = {}
    for event in events:
        if event["event"] != "capture":
            continue
        meta = event["metadata"]
        require(meta["schema_version"] == 2 and meta["rank"] == meta["tp_rank"] and
                meta["source_head"] == HEAD and meta["original_backend"] == backend,
                "Actual capture metadata arm, rank, revision")
        key = (meta["rank"], meta["layer_idx"], meta["call_index"])
        require(key not in indexed, "No duplicate layer/rank/step capture")
        own_sampler_observed = validate_join(meta, prepared, sampled, fixtures)
        path = verify_capture_file(directory, event)
        require(meta["pid"] > 0, "Actual capture worker PID")
        workers.setdefault(meta["rank"], set()).add(meta["pid"])
        registered = [item for item in events if item["event"] == "layer_registered"
                      and item["pid"] == meta["pid"] and item["rank"] == meta["rank"]
                      and item["context"]["layer_idx"] == meta["layer_idx"]]
        require(len(registered) == 1 and registered[0]["context"]["prefix"] == meta["prefix"],
                "Unique registered exact layer prefix")
        indexed[key] = {"path": path, "record": event,
                        "same_rank_sampler_context_observed": own_sampler_observed}
    assert_complete(indexed)
    require(set(workers) == {0, 1} and all(len(value) == 1 for value in workers.values())
            and len(set.union(*workers.values())) == 2, "Two independent TP worker PIDs")
    for rank, pids in workers.items():
        pid = next(iter(pids))
        installed = read(directory / ("installed-pid-" + str(pid) + ".json"))
        patched = read(directory / ("patched-pid-" + str(pid) + ".json"))
        require(installed["config_sha256"] == launch["capture_config_sha256"] and
                installed["config"] == config, "Actual worker capture installation")
        require(patched["module_sha256"] == config["expected_module_sha256"] and
                patched["helper_sha256"] == launch["helper_sha256"]["launch_capture.py"],
                "Actual GDN source and collector helper binding")
        require(sum(item["event"] == "start_seen" and item["pid"] == pid
                    for item in events) == 0,
                "Late collector selection does not use the replaced legacy start observer")
    require({str(item["path"].relative_to(directory)).replace(chr(92), "/")
             for item in indexed.values()} ==
            {str(path.relative_to(directory)).replace(chr(92), "/")
             for path in directory.rglob("*.pt")}, "No unbound binary capture")
    return indexed


def load_snapshot(entry):
    import torch
    blob = torch.load(entry["path"], map_location="cpu", weights_only=True)
    require(sha(entry["path"]) == entry["record"]["sha256"],
            "Captured binary remains unchanged through CPU loading")
    require(set(blob) == {"metadata", "tensors"} and
            blob["metadata"] == entry["record"]["metadata"],
            "Binary metadata equals SHA-bound sidecar/event")
    meta, tensors = blob["metadata"], blob["tensors"]
    require(set(tensors) == {"mixed_qkv", "a", "b", "A_log", "dt_bias", "initial_state",
                            "production_output", "production_post_state", "indices_original",
                            "indices_compact", "padding_rows"}, "Complete capture tensors")
    for tensor in tensors.values():
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu",
                "CPU tensor load only")
        require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
                "Finite observed tensor values")
    joined = meta["runtime_join"]["rows"]
    indices = tensors["indices_original"].reshape(-1).tolist()
    require(indices == [row["gdn_state_page"] for row in joined],
            "Actual kernel page vector equals exact metadata join")
    mapping = meta["compact_to_original"]
    active = sorted(indices)
    require(len(set(active)) == len(active) and meta["active_original_indices"] == active
            and mapping[:len(active) + 1] == [0, *active], "Exact compact active state mapping")
    require(tensors["indices_compact"].reshape(-1).tolist() ==
            [mapping.index(page) for page in indices], "Actual compact kernel indices")
    require(meta["padding_rows"] == [False] * len(joined) and
            tensors["padding_rows"].tolist() == [False] * len(joined),
            "Observed ordinary decode has no padded rows")
    initial, post = tensors["initial_state"], tensors["production_post_state"]
    require(initial.shape == post.shape and initial.dtype == post.dtype == torch.float32
            and list(initial.shape) == [len(mapping), 24, 128, 128],
            "TP2 FP32 compact recurrent state contract")
    for index in (0, meta["unused_compact_index"]):
        require(torch.equal(initial[index], post[index]), "Null and sampled unused page unchanged")
    for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias"):
        spec = meta["original_shapes_strides"][name]
        require(list(tensors[name].shape) == spec["shape"] and
                str(tensors[name].dtype) == spec["dtype"], "Original observed input shape/dtype")
    require(list(tensors["production_output"].shape) ==
            meta["original_shapes_strides"]["out"]["shape"], "Observed production output shape")
    expected_bias = "torch.float32" if meta["original_backend"] == "flashinfer" else "torch.bfloat16"
    require(str(tensors["dt_bias"].dtype) == expected_bias, "Actual per-backend bias specialization")
    return blob


def compare_tensor(left, right):
    import torch
    require(left.shape == right.shape and left.dtype == right.dtype,
            "Paired observed state shape/dtype")
    delta = left.double() - right.double()
    norm = float(torch.linalg.vector_norm(left.double()))
    error = float(torch.linalg.vector_norm(delta))
    return {"exact_equal": bool(torch.equal(left, right)),
            "max_abs_error": float(delta.abs().max()),
            "relative_l2_error": error / max(norm, 1e-30)}


def observed_pair(indexed):
    initial = []
    propagated = []
    mismatched_rows = []
    layer_inputs = []
    for key in sorted(expected_keys()):
        left = load_snapshot(indexed["triton"][key])
        right = load_snapshot(indexed["flashinfer"][key])
        lhs = left["metadata"]["runtime_join"]["rows"]
        rhs = right["metadata"]["runtime_join"]["rows"]
        fields = ("index", "step", "input_prefix_sha256", "input_token", "input_position")
        left_order = [tuple(row[field] for field in fields) for row in lhs]
        right_order = [tuple(row[field] for field in fields) for row in rhs]
        if left_order != right_order:
            mismatched_rows.append({"rank": key[0], "layer_idx": key[1], "step": key[2],
                                    "triton": left_order, "flashinfer": right_order})
        if left_order == right_order:
            inputs = {name: compare_tensor(left["tensors"][name], right["tensors"][name])
                      for name in ("mixed_qkv", "a", "b", "A_log")}
            inputs["dt_bias_actual_values_float32"] = compare_tensor(
                left["tensors"]["dt_bias"].float(), right["tensors"]["dt_bias"].float())
            layer_inputs.append({"rank": key[0], "layer_idx": key[1], "step": key[2],
                                 "inputs": inputs,
                                 "production_output": compare_tensor(
                                     left["tensors"]["production_output"],
                                     right["tensors"]["production_output"])})
        right_by_index = {row["index"]: row for row in rhs}
        for a in lhs:
            b = right_by_index.get(a["index"])
            if b is None or any(a[field] != b[field] for field in fields):
                continue
            ia = left["metadata"]["compact_to_original"].index(a["gdn_state_page"])
            ib = right["metadata"]["compact_to_original"].index(b["gdn_state_page"])
            value = {"rank": key[0], "layer_idx": key[1], "step": key[2], "index": a["index"],
                     "prestate": compare_tensor(left["tensors"]["initial_state"][ia],
                                                right["tensors"]["initial_state"][ib]),
                     "poststate": compare_tensor(left["tensors"]["production_post_state"][ia],
                                                 right["tensors"]["production_post_state"][ib])}
            (initial if key[2] == 1 else propagated).append(value)
    initial_keys = {(item["rank"], item["layer_idx"], item["index"]) for item in initial}
    required_initial = {(rank, layer, index) for rank in (0, 1) for layer in LAYERS
                        for index in (198, 206, 209, 228, 255, 285, 292, 318)}
    return {"first_decode_active_states": initial, "later_observed_states": propagated,
            "mismatched_capture_rows": mismatched_rows,
            "observed_layer_inputs_and_outputs": layer_inputs,
            "missing_paired_initial_rows": sorted(required_initial - initial_keys),
            "common_observed_initial_state": initial_keys == required_initial and
                all(item["prestate"]["exact_equal"] for item in initial),
            "captured_row_schedules_equal": not mismatched_rows}


def prefill_gate(native):
    values = []
    for index in sorted(native["fixtures"]):
        left = native["records_by_arm"]["triton"][(index, 0)]["record"]
        right = native["records_by_arm"]["flashinfer"][(index, 0)]["record"]
        lhs, rhs = left["raw_logits_full_sha256"], right["raw_logits_full_sha256"]
        require(isinstance(lhs, str) and isinstance(rhs, str) and len(lhs) == len(rhs) == 64,
                "Actual full vocabulary prefill logits SHA")
        values.append({"index": index, "triton_sha256": lhs, "flashinfer_sha256": rhs,
                       "exact_equal": lhs == rhs})
    return values


def audit_capture_pair(triton_dir, flashinfer_dir, reference_root, source_root, source_manifest):
    from audit_late_sampler import audit_sampler_pair
    lock, fixtures_list = validate_inputs(reference_root)
    fixtures = {item["index"]: item for item in fixtures_list}
    native = audit_sampler_pair(triton_dir, flashinfer_dir, reference_root, source_root, source_manifest)
    indexed = {backend: index_arm(Path(root), backend, fixtures)
               for backend, root in (("triton", triton_dir), ("flashinfer", flashinfer_dir))}
    observed = observed_pair(indexed)
    prefill = prefill_gate(native)
    schedules_equal = native["schedules_by_arm"]["triton"] == native["schedules_by_arm"]["flashinfer"]
    native_gate = native["paircomparison"]["native_starting_condition_gate_pass"]
    require(type(native_gate) is bool, "Native paired starting-condition boolean")
    common = observed["common_observed_initial_state"] and all(item["exact_equal"] for item in prefill)
    report = {"schema": 1, "status": "OBSERVED_INITIAL_PASS" if common and native_gate and schedules_equal and observed["captured_row_schedules_equal"] else "NOGO",
              "source_head": HEAD, "reference_completed_sha256": lock["reference_files"]["completed.json"],
              "native_sampler": native["arms"], "native_paircomparison": native["paircomparison"],
              "recorded_schedules_equal": schedules_equal, "native_starting_condition_gate_pass": native_gate,
              "full_prefill_logits": prefill,
              "observed_states": observed, "snapshots_by_arm": {},
              "scope": "Forced common-history localization only. Independently propagated GDN states are observed; full-attention KV is unobserved. Exact observed initial conditions are a prerequisite, not a complete causal proof.",
              "execution_readiness": "Per-backend exact production replay and frozen common-prestate math tolerance still require separate GPU replay reports."}
    for backend, values in indexed.items():
        report["snapshots_by_arm"][backend] = [
            {"path": str(entry["path"]), "sha256": entry["record"]["sha256"],
             "rank": key[0], "layer_idx": key[1], "step": key[2],
             "execution_id": entry["record"]["metadata"]["runtime_join"]["late_execution_id"],
             "captured_backend": backend,
             "same_rank_sampler_context_observed": entry["same_rank_sampler_context_observed"],
             "captured_dt_bias_dtype": entry["record"]["metadata"]["original_shapes_strides"]["dt_bias"]["dtype"]}
            for key, entry in sorted(values.items())]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--flashinfer", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Never overwrite an audit result")
    try:
        report = audit_capture_pair(args.triton, args.flashinfer, args.reference_root,
                                    args.source_root, args.source_manifest)
    except Exception as error:
        report = {"status": "INVALID_EVIDENCE", "error": str(error), "gpu_execution": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    raise SystemExit(0 if report["status"] == "OBSERVED_INITIAL_PASS" else 2)


if __name__ == "__main__":
    main()
