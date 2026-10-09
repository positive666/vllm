"""Verify every frozen snapshot and summarize bounded capture coverage."""
import collections
import hashlib
import json
from pathlib import Path

root = Path("/results/fi-c8-v2")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


rows = []
for path in sorted((root / "captures").glob("**/*.pt")):
    side = path.with_suffix(".json")
    record = json.loads(side.read_text())
    assert record["sha256"] == sha(path)
    assert record["bytes"] == path.stat().st_size
    meta = record["metadata"]
    rows.append({
        "file": str(path.relative_to(root)), "sha256": record["sha256"],
        "bytes": path.stat().st_size, "sidecar_sha256": sha(side),
        "rank": meta["rank"], "layer": meta["layer_idx"],
        "call": meta["call_index"], "actual_batch": meta["batch_size"],
        "actual_metadata": meta,
    })
responses = json.loads((root / "responses.json").read_text())
server = json.loads((root / "server.json").read_text())
assert server["status"] == "completed" and len(responses["examples"]) == 8
counts = collections.Counter(row["rank"] for row in rows)
assert set(counts) == {0, 1}
for rank in (0, 1):
    for call in (1, 2):
        selected = [row for row in rows if row["rank"] == rank and row["call"] == call]
        assert len(selected) == 48 and all(row["actual_batch"] == 8 for row in selected)
summary = {
    "snapshot_count": len(rows), "snapshot_bytes": sum(row["bytes"] for row in rows),
    "rank_counts": dict(counts),
    "batch_counts": dict(collections.Counter(row["actual_batch"] for row in rows)),
    "call_counts": dict(collections.Counter(row["call"] for row in rows)),
    "early_coverage": "48 actual GDN layers x 2 ranks x first 2 B8 ordinary calls",
    "response_summary": responses["summary"],
    "responses": [{k: row[k] for k in ("index", "gold", "predicted", "strict_correct", "finish_reason", "usage")} for row in responses["examples"]],
    "scope": "Bounded eager diagnostic sampling, later calls limited to selected layers. Not a global first-divergence trace, not a model-quality evaluation or performance run.",
    "source_files": server["source_files"],
    "server_sha256": sha(root / "server.json"),
    "responses_sha256": sha(root / "responses.json"),
    "serve_log_closed_sha256": sha(root / "serve.log"),
    "files": rows,
}
with (root / "snapshot-manifest.json").open("x") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))
# A small smoke subset uses copies of frozen files, never alters the originals.
smoke = root / "replay-smoke"
smoke.mkdir(exist_ok=False)
for rank in (0, 1):
    row = next(row for row in rows if row["rank"] == rank and row["layer"] == 0 and row["call"] == 1)
    path = root / row["file"]
    (smoke / f"rank-{rank}.pt").hardlink_to(path)
    (smoke / f"rank-{rank}.json").hardlink_to(path.with_suffix(".json"))
