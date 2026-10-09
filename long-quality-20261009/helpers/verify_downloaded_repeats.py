"""Verify the owned repeat archives before local extraction."""
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

root = Path(__file__).resolve().parents[1]
archives = {
    "triton-eager-2": "3e70faaa083bf906a91331484e3bd53720cf83934ac3b06fa4d5202d915bea15",
    "flashinfer-graph-2": "86b0016b41fbac44a266f2854687958ffaee8f7e3e66979168be56394805e46d",
    "triton-graph-2": "d12c826bd4fe674809721b4c4cad41b9d39385e53cbc955ea5c9e7091a504e3b",
}
for name, expected in archives.items():
    path = root / (name + ".tar.gz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    assert not (root / name).exists(), "Refusing to overwrite an existing run"
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        assert len({member.name for member in members}) == len(members)
        for member in members:
            part = PurePosixPath(member.name)
            assert not part.is_absolute() and ".." not in part.parts
            assert member.isfile() or member.isdir()
            assert part.parts[0] == name or member.name in {
                name + ".log", name + ".exit", name + "-host.csv", name + "-processes.log"
            }
        bundle.extractall(root, filter="data")
    print({"run": name, "archive_sha256": expected, "members": len(members)})
