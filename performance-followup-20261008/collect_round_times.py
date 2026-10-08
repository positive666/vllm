"""Collect remote filesystem times before transporting the immutable results."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


root = Path("/results/performance-followup-20261008")
rows = []
for path in sorted((root / "http").glob("*-r*.json")):
    contents = path.read_bytes()
    result = json.loads(contents)
    finish = path.stat().st_mtime_ns / 1e9
    wall = result["summary"]["wall_seconds"]
    rows.append({
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
        "arm": result["arm"], "kind": result["kind"],
        "repeat": result["repeat"], "concurrency": result["concurrency"],
        "wall_seconds": wall,
        "approx_start_utc": datetime.fromtimestamp(finish - wall, timezone.utc).isoformat(),
        "approx_end_utc": datetime.fromtimestamp(finish, timezone.utc).isoformat(),
    })
output = root / "round-file-times.json"
assert not output.exists(), output
output.write_text(json.dumps({
    "collected_utc": datetime.now(timezone.utc).isoformat(),
    "container_local_datetime": datetime.now().astimezone().isoformat(),
    "container_timezone_names": time.tzname,
    "container_local_utc_offset": time.strftime("%z"),
    "scope": "Remote original result file mtimes, collected before download",
    "limitation": "Approximate round end uses JSON file mtime; start subtracts measured wall_seconds. JSON serialization, file write and scheduling add untimed uncertainty. Not original per-round UTC instrumentation.",
    "rows": rows,
}, indent=2) + "\n")
