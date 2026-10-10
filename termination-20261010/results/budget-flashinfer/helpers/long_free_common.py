"""Immutable inputs and scoring shared by the bounded long-generation probe."""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import sys
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
INDICES = [198, 206, 209, 228, 255, 285, 292, 318]
TARGETS = [209, 255]
SCORER_HASHES = {
    "frozen_quality_http.py": "885dab543e73369a8fab05603274fbf654add1eeeae9f0c70e4324c3e138b74e",
    "http_bench_client.py": "858ca5fbeba25376476a9a43a2b1d100101ce3f99cc3d1785a273c7fdb4de464",
}
SCOPE = (
    "Eight selected long-generation diagnostic questions, including targets 209 "
    "and 255 and six controls. Unforced offline synchronous TP2; no population "
    "accuracy, statistical equivalence or serving performance claim. This phase "
    "raises only max_model_len=16384 and max_tokens=8192 to inspect native "
    "termination and verify overlap with the unchanged prior 3500-token run."
)
SOURCE_PATHS = [
    "vllm/entrypoints/llm.py",
    "vllm/engine/arg_utils.py",
    "vllm/config/observability.py",
    "vllm/config/compilation.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu/cudagraph_utils.py",
    "vllm/v1/worker/gpu/sample/sampler.py",
    "vllm/v1/worker/gpu/sample/trace_replay.py",
    "vllm/v1/metrics/loggers.py",
    "vllm/compilation/counter.py",
    "vllm/compilation/cuda_graph.py",
]


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(value):
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    return str(value)


def frozen_scorer():
    folder = Path(__file__).resolve().parent
    for name, expected in SCORER_HASHES.items():
        require(sha(folder / name) == expected, "Frozen scorer SHA256: " + name)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import frozen_quality_http
    return frozen_quality_http


def flags(text, gold, finish_reason):
    predicted = frozen_scorer().answer_number(text)
    correct = predicted == gold
    completed = finish_reason == "stop"
    return {
        "predicted": predicted,
        "raw_correct": correct,
        "completed_correct": correct and completed,
        "strict_correct": correct and completed and "####" in text,
        "truncated": finish_reason == "length",
        "has_answer_marker": "####" in text,
        "unparsed": predicted is None,
    }


def summarize(rows):
    return {
        "requests": len(rows),
        "raw_correct": sum(row["raw_correct"] for row in rows),
        "strict_correct": sum(row["strict_correct"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "unparsed": sum(row["unparsed"] for row in rows),
        "missing_marker": sum(not row["has_answer_marker"] for row in rows),
    }


def model_options(backend, mode):
    return {
        "model": "/model", "language_model_only": True, "dtype": "bfloat16",
        "tensor_parallel_size": 2, "max_model_len": 16384, "max_num_seqs": 32,
        "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": False, "mamba_ssm_cache_dtype": "float32",
        "seed": 42, "enforce_eager": mode == "eager", "async_scheduling": False,
        "enable_trace_replay": False, "disable_log_stats": False,
        "cudagraph_metrics": True,
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "kernel_config": {"gdn_decode_backend": backend, "linear_backend": "marlin"},
        "num_gpu_blocks_override": 128,
    }


def sampling_options():
    return {
        "temperature": 0, "seed": 42, "max_tokens": 8192, "min_tokens": 0,
        "ignore_eos": False, "stop": [], "stop_token_ids": [],
        "logprobs": None, "prompt_logprobs": None,
    }
