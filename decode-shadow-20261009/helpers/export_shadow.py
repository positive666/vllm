"""Export compact diagnostic evidence, retaining all frozen hashes and failures.

The full 7.3 GB capture corpus remains on the experiment host. Export two
representative actual snapshots and all replay/capture reports; never suggest
that the representatives alone validate the other hash-listed snapshots.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

root = Path("/results")
capture = root / "fi-c8-v2"
out = root / "shadow-export"
manifest = json.loads((capture / "snapshot-manifest.json").read_text())
for version in ("v2", "v3"):
    audit = json.loads((root / f"force-audit-{version}.json").read_text())
    assert audit["integrity_pass"] is True
for run in ("force-triton-v2", "force-flashinfer-v2", "force-triton-v3", "force-flashinfer-v3"):
    assert (root / f"{run}.exit").read_text().strip() == "0"
    assert (root / run / "completed.json").is_file()
    assert (root / f"{run}.log").is_file()
    assert (root / f"{run}-host.csv").is_file()
    assert (root / f"{run}-processes.log").is_file()
assert (capture / "replay-full-v2.exit").read_text().strip() == "0"
out.mkdir(exist_ok=False)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy(source, relative):
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


for name in ("server.json", "responses.json", "capture-config.json", "snapshot-manifest.json", "serve.log",
             "replay-smoke.json", "replay-smoke.log", "replay-smoke.exit", "replay_capture.smoke-v1.py",
             "replay-smoke-v2.json", "replay-smoke-v2.log", "replay-smoke-v2.exit", "replay-smoke-v2-summary.json",
             "replay-full-v2.json", "replay-full-v2.log", "replay-full-v2.exit", "replay-full-v2-summary.json",
             "replay-bf16-isolated-v2.json", "replay-bf16-isolated-v2.log", "replay-bf16-isolated-v2.exit"):
    copy(capture / name, Path("capture-and-replay") / name)
assert sha(capture / "serve.log") == manifest["serve_log_closed_sha256"]
for record in sorted((capture / "captures").glob("*.json*")):
    copy(record, Path("capture-and-replay/trace") / record.name)
for source in sorted((capture / "helpers").glob("**/*")):
    if source.is_file():
        copy(source, Path("capture-and-replay/frozen-capture-helpers") / source.relative_to(capture / "helpers"))
representatives = [
    next(row for row in manifest["files"] if row["rank"] == 0 and row["layer"] == 0 and row["call"] == 1),
    next(row for row in manifest["files"] if row["rank"] == 1 and row["layer"] == 2 and row["call"] == 995),
]
for row in representatives:
    path = capture / row["file"]
    assert sha(path) == row["sha256"]
    copy(path, Path("representative-snapshots") / row["file"])
    copy(path.with_suffix(".json"), Path("representative-snapshots") / Path(row["file"]).with_suffix(".json"))
for source in sorted(Path("/shadow-helpers").glob("**/*")):
    if source.is_file() and "__pycache__" not in source.parts:
        copy(source, Path("helpers") / source.relative_to("/shadow-helpers"))
for name in ("model-bias-probe.json", "capture-inspection.log", "fi-c8-v1-supervisor.log", "fi-c8-v1.exit",
             "fi-c8-v2-supervisor.log", "fi-c8-v2.exit", "fi-c8-v2-host.csv", "fi-c8-v2-processes.log",
             "force-plan-v1.log", "force-plan-v2.log", "force-triton-v1.log", "force-triton-v1.exit",
             "force-triton-v1.stopped.utc"):
    copy(root / name, Path("run-records") / name)
for run in ("fi-c8-v1", "force-triton-v1", "force-triton-v2", "force-flashinfer-v2", "force-triton-v3", "force-flashinfer-v3"):
    for source in sorted((root / run).glob("**/*")):
        if source.is_file() and "__pycache__" not in source.parts:
            copy(source, Path("run-records") / run / source.relative_to(root / run))
for pattern in ("force-*-v[23].log", "force-*-v[23].exit", "force-*-v[23]-host.csv", "force-*-v[23]-processes.log", "force-audit*.json"):
    for source in sorted(root.glob(pattern)):
        copy(source, Path("run-records") / source.name)
files = [{"file": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)}
         for p in sorted(out.glob("**/*")) if p.is_file()]
with (out / "export-manifest.json").open("x") as f:
    json.dump({"files": files, "representative_snapshots": representatives,
               "full_capture_corpus_bytes": manifest["snapshot_bytes"],
               "full_capture_corpus_exported": False,
               "scope": "Full reports and source/trace artifacts; two representative raw snapshots. Full capture corpus retained on the experiment host, bound by snapshot-manifest.json."}, f, indent=2)
archive = root / "decode-shadow-evidence.zip"
with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(out.glob("**/*")):
        if path.is_file():
            z.write(path, str(path.relative_to(out)))
receipt = {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive),
           "export_file_count": len(files) + 1, "manifest_sha256": sha(out / "export-manifest.json")}
with (root / "export-receipt.json").open("x") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
