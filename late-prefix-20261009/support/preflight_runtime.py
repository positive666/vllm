"""Check the exact captured runtime without importing model/CUDA modules."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


reference = Path("/late-reference/reference")
launch = json.loads((reference / "launch.json").read_text())
binding = json.loads((reference / "runtime-binding.json").read_text())
versions = {name: importlib.metadata.version(name)
            for name in launch["runtime_versions"]}
assert versions == launch["runtime_versions"], "Captured package versions"
for name, expected in binding["modules"].items():
    path = Path(expected["path"])
    assert path.stat().st_size == expected["bytes"]
    assert sha(path) == expected["sha256"], name
assert "torch" not in sys.modules and "flashinfer" not in sys.modules
report = {
    "runtime_packages_exact": True,
    "versions": versions,
    "runtime_modules_exact": True,
    "modules": binding["modules"],
    "gpu_execution": False,
    "frozen_helpers_changed": False,
    "restored_mount": "/oldcache/runtime-addons (read-only)",
    "source_head": binding["source_head"],
}
with Path("/results/runtime-preflight.json").open("x") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")
print(json.dumps({"runtime_packages_exact": True,
                  "runtime_modules_exact": True,
                  "packages": len(versions),
                  "modules": len(binding["modules"]),
                  "gpu_execution": False}))
