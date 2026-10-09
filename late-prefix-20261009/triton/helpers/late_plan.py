"""CPU-only immutable input validation for conditional late decode localization."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def require(value, label):
    if not value:
        raise ValueError(label)

def validate_inputs(root):
    root = Path(root)
    lock = read(root / "reference-lock.json")
    require(lock["source_head"] == HEAD, "Reviewed source head")
    for name, expected in lock["reference_files"].items():
        require(sha(root / "reference" / name) == expected, "Frozen reference SHA: " + name)
    require((root / "reference/run.exit").read_text().strip() == "0", "Unforced reference exited 0")
    completed = read(root / "reference/completed.json")
    launch = read(root / "reference/launch.json")
    fixture = read(root / "reference/fixture.json")
    before = read(root / "reference/runtime-before.json")
    require(completed["status"] == "completed" and completed["backend"] == "flashinfer" and
            completed["mode"] == "eager", "Frozen failing eager FI reference")
    require(launch["source_head"] == HEAD and completed["launch_sha256"] == sha(root / "reference/launch.json"),
            "Reference launch binding")
    require(launch["fixture_sha256"] == sha(root / "reference/fixture.json"), "Reference prompt fixture binding")
    require(before["engine_core_class"] == "InprocClient", "Reference all-at-once in-process execution")
    rows = completed["examples"]
    require([row["index"] for row in rows] == lock["indices"], "Same original eight requests")
    examples = []
    for row, original in zip(rows, fixture["examples"]):
        require(row["prompt_token_ids"] == original["prompt_token_ids"] and
                digest(row["prompt_token_ids"]) == row["prompt_token_ids_sha256"], "Exact actual input prompt IDs")
        require(digest(row["token_ids"]) == row["token_ids_sha256"], "Exact actual unforced completion IDs")
        require(0 < len(row["token_ids"]) <= lock["limit"], "Bounded reference output")
        require(len(row["prompt_token_ids"]) + len(row["token_ids"]) <= 4096, "Full history within context")
        examples.append({"index": row["index"], "prompt_token_ids": row["prompt_token_ids"],
                         "reference_token_ids": row["token_ids"], "max_tokens": len(row["token_ids"]),
                         "reference_token_count": len(row["token_ids"]),
                         "reference_token_ids_sha256": row["token_ids_sha256"]})
    target = next(row for row in rows if row["index"] == lock["target"])
    require(len(target["token_ids"]) == 3500 and target["finish_reason"] == "length" and
            not target["has_answer_marker"], "Retained target truncation and missing marker")
    require(lock["selected_steps"] == [1, 18, 19, 1750, 3497, 3498, 3499], "Exact selected actual decode positions")
    return lock, examples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    lock, rows = validate_inputs(args.root)
    print(json.dumps({"status": "CPU-plan-validated", "requests": len(rows),
                      "reference_completed_sha256": lock["reference_files"]["completed.json"],
                      "observations": sum(row["max_tokens"] for row in rows),
                      "selected_steps": lock["selected_steps"],
                      "gpu_execution": False}, indent=2))
