"""Archive the finished, isolated performance evidence without modifying it."""

import hashlib
import json
from pathlib import Path
import zipfile


root = Path("/results/performance-followup-20261008")
assert (root / "pipeline.exit").read_text().strip() == "0"
assert (root / "round-file-times.json").exists()
output = root.with_suffix(".zip")
files = [path for path in sorted(root.rglob("*")) if path.is_file()]
with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())
print(json.dumps({
    "archive": str(output), "files": len(files), "bytes": output.stat().st_size,
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}))
