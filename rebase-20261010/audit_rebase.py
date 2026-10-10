"""Audit rebase validation records without running or changing model inference."""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HEAD = "abf17c7c071b1b6bdd81ca9756894998e5e8b32d"
sys.path.insert(0, str(ROOT / "helpers"))
from long_free_common import flags, summarize


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, label):
    if not value:
        raise ValueError(label)


def main():
    manifest = read(ROOT / "source-manifest.json")
    require(manifest["source_head"] == HEAD, "Source head")
    require((RESULTS / "v2/matrix.exit").read_text().strip() == "0", "Matrix exit")
    require((RESULTS / "v2/cleanup-stop.exit").read_text().strip() == "0", "Cleanup")
    require(not read(RESULTS / "v2/cleanup-after.json")["Running"], "Container stopped")
    require(not (RESULTS / "v2/repeated-unmatched-owner.txt").exists(), "No repeated ownership failure")
    cases = []
    for name, expected in (("adapter-and-split", 40), ("gdn-metadata", 65), ("gdn-config", 1)):
        rows = list(ET.parse(RESULTS / f"v2/focused/{name}.xml").getroot().iter("testcase"))
        require(len(rows) == expected, name + " count")
        require(all(not any(row.find(tag) is not None for tag in ("failure", "error", "skipped")) for row in rows), name + " statuses")
        cases += rows
    focused = read(RESULTS / "v2/focused/recheck-summary.json")
    require(focused["all_tasks_exit_zero"] and focused["all_expected_pytest_cases_passed"], "Focused runner result")
    for task in focused["tasks"]:
        require(sha(RESULTS / Path(task["log"]).relative_to("/results")) == task["log_sha256"], "Focused log hash")
    distributed = read(RESULTS / "v2/tp2-correctness/distributed-correctness.json")
    require(distributed["status"] == "passed" and distributed["passed_rank_cases"] == 12, "TP2 cases")
    require(distributed["source_head"] == HEAD, "TP2 head")
    arms = []
    for mode in ("graph", "eager"):
        for backend in ("triton", "flashinfer"):
            name = f"{backend}-{mode}"
            root = RESULTS / name
            launch, result = read(root / "launch.json"), read(root / "completed.json")
            require((RESULTS / f"v2/{name}.exit").read_text().strip() == "0", name + " exit")
            require(launch["source_head"] == HEAD and result["launch_sha256"] == sha(root / "launch.json"), name + " binding")
            require(result["status"] == "completed" and len(result["examples"]) == 8, name + " completions")
            require(read(root / "shutdown.json")["generation_completed"], name + " shutdown")
            for path, expected in launch["helper_sha256"].items():
                require(sha(root / "helpers" / path) == expected, name + " helper hash")
            for path, expected in manifest["files"].items():
                require(launch["source_files"][path] == expected["sha256"], name + " source hash")
            runtime = read(root / "runtime-after.json")
            require(len(runtime["worker_ranks"]) == 2, name + " rank count")
            require(all(r["tp_world_size"] == 2 and r["graphs_captured"] == (mode == "graph") for r in runtime["worker_ranks"]), name + " runtime mode")
            require(all(r["config"]["kernel_config"]["gdn_decode_backend"] == backend for r in runtime["worker_ranks"]), name + " runtime backend")
            for row in result["examples"]:
                digest = hashlib.sha256(json.dumps(row["token_ids"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                require(digest == row["token_ids_sha256"], name + " token hash")
                require(row["usage"]["completion_tokens"] == len(row["token_ids"]), name + " token count")
                for key, value in flags(row["text"], row["gold"], row["finish_reason"]).items():
                    require(row[key] == value, name + " recomputed score")
            require(result["summary"] == summarize(result["examples"]), name + " recomputed summary")
            targets = [{k: row[k] for k in ("index", "predicted", "gold", "finish_reason", "strict_correct", "usage")} for row in result["examples"] if row["target"]]
            arms.append({"backend": backend, "mode": mode, "summary": result["summary"], "targets": targets})
    output = {"status": "passed", "source_head": HEAD, "focused_passed": len(cases), "tp2_rank_cases": 12, "model_completions": 32, "arms": arms, "scope": "Record integrity and bounded compatibility; not accuracy equivalence or new performance measurement", "ownership_scope": "No two-consecutive-failure monitor stop; inspect individual unmatched observations separately"}
    (ROOT / "result-audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
