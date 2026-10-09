"""Derive compact claims from the saved, unmodified replay records."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data = json.loads(args.input.read_text())
cases = data["cases"]
summary = {
    "report_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    "report_status": data["status"], "cases": len(cases),
    "exact_fidelity_count": sum(c["production_reproduction"]["exact_reproduction"] for c in cases),
    "cases_frozen_tolerance_pass": sum(c["checks_within_frozen_tolerance"] for c in cases),
    "rank_counts": {str(r): sum(c["metadata"]["rank"] == r for c in cases) for r in (0, 1)},
    "tolerances": data["tolerances"],
    "scope": data["scope"], "metrics": {},
}
for variant in sorted({v for c in cases for v in c["arms"]}):
    records = []
    for c in cases:
        if variant not in c["arms"]:
            continue
        a = c["arms"][variant]
        for backend, br in a["backends"].items():
            for metric in ("vs_fp64_math_output", "vs_fp64_math_state", "vs_fp64_rounded_output", "vs_fp64_rounded_state"):
                records.append((f"{backend}/{metric}", br[metric], c["metadata"]))
        for metric in ("triton_vs_flashinfer_output", "triton_vs_flashinfer_state"):
            records.append((metric, a[metric], c["metadata"]))
    metrics = {}
    for name in sorted({item[0] for item in records}):
        selected = [(m, meta) for key, m, meta in records if key == name]
        finite = [(m, meta) for m, meta in selected if m.get("relative_l2") is not None]
        worst = max(finite, key=lambda item: item[0]["relative_l2"]) if finite else None
        metrics[name] = {
            "case_count": len(selected),
            "exact_equal_count": sum(m["exact_equal"] for m, _ in selected),
            "within_tolerance_count": sum(m["within_tolerance"] for m, _ in selected),
            "max_abs_error": max((m["absmax"] for m, _ in selected if m["absmax"] is not None), default=None),
            "max_relative_l2": worst[0]["relative_l2"] if worst else None,
            "worst_relative_case": {k: worst[1][k] for k in ("rank", "layer_idx", "call_index", "batch_size")} if worst else None,
        }
    summary["metrics"][variant] = metrics
if "error" in data:
    summary["error"] = data["error"]
with args.output.open("x") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
