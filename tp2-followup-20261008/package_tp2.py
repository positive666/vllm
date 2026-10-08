"""Freeze or verify the original TP2 results without rewriting raw bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("archive", "extract"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.mode == "archive":
        for name in ("pipeline.exit", "preflight.exit", "distributed-correctness.exit"):
            assert (args.root / name).read_text().strip() == "0", name
        assert (args.root / "round-file-times.json").is_file()
        assert not args.archive.exists()
        files = [path for path in sorted(args.root.rglob("*")) if path.is_file()]
        with zipfile.ZipFile(args.archive, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for path in files:
                z.write(path, path.relative_to(args.root).as_posix())
        print(json.dumps({"archive": str(args.archive), "files": len(files),
                          "bytes": args.archive.stat().st_size,
                          "sha256": sha(args.archive.read_bytes())}))
    else:
        digest = sha(args.archive.read_bytes())
        assert args.sha256 and digest == args.sha256
        members = []
        with zipfile.ZipFile(args.archive) as z:
            assert z.testzip() is None
            for item in z.infolist():
                path = (args.root / item.filename).resolve()
                path.relative_to(args.root.resolve())
                assert not item.is_dir()
                data = z.read(item)
                if path.exists():
                    assert path.read_bytes() == data, path
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                members.append({"path": item.filename, "bytes": len(data),
                                "sha256": sha(data)})
        result = {"sha256": digest, "bytes": args.archive.stat().st_size,
                  "files": len(members), "members": members}
        (args.root.parent / "archive-manifest.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"archive_verified": True, "files": len(members)}))


if __name__ == "__main__":
    main()
