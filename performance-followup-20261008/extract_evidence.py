"""Verify and unpack the exact remote archive, retaining a content manifest."""

import hashlib
import json
from pathlib import Path
import zipfile


root = Path(__file__).parent
archive_path = root / "performance-evidence.zip"
archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
assert archive_sha == "b8ede3808ebd6278ffe55c9f8dfa4e92e62aadd6c53b90730e1230f7a19ca7b3"
raw = root / "raw"
members = []
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    for item in archive.infolist():
        path = (raw / item.filename).resolve()
        path.relative_to(raw.resolve())
        content = archive.read(item)
        if path.exists():
            assert path.read_bytes() == content, path
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        members.append({
            "path": item.filename, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
(root / "archive-manifest.json").write_text(json.dumps({
    "archive_sha256": archive_sha, "bytes": archive_path.stat().st_size,
    "member_count": len(members), "members": members,
}, indent=2) + "\n")
print(json.dumps({"archive_verified": True, "members": len(members)}))
