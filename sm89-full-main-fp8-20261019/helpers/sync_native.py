"""Copy identical generated package files, verifying the two tracked diffs."""

import hashlib
import json
from pathlib import Path
import shutil
import tarfile

base = Path('/work/source/baseline')
candidate = Path('/work/source/candidate')
tracked = {}
with tarfile.open('/work/main-source.tar.gz') as archive:
    for entry in archive:
        rel = entry.name.partition('/')[2]
        if entry.isfile():
            tracked[rel] = hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
allowed = {
    'vllm/third_party/flash_linear_attention/ops/fused_recurrent.py',
    'tests/kernels/fla/test_fused_recurrent_packed_decode.py',
}
changes = set()
for rel, sha in tracked.items():
    assert hashlib.sha256((base / rel).read_bytes()).hexdigest() == sha, rel
    if hashlib.sha256((candidate / rel).read_bytes()).hexdigest() != sha:
        changes.add(rel)
assert changes == allowed, changes
generated = {}
for file in (candidate / 'vllm').rglob('*'):
    if not file.is_file() or '__pycache__' in file.parts or file.suffix == '.pyc':
        continue
    rel = file.relative_to(candidate).as_posix()
    if rel not in tracked:
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        generated[rel] = hashlib.sha256(file.read_bytes()).hexdigest()
assert any(name.endswith('.so') for name in generated)
Path('/work/results/source-runtime-manifest.json').write_text(json.dumps({
    'base_commit': '3ddc6702a8264da4cf2428aa5ca2e267d84a8841',
    'native_commit': 'dd9e07a25febb05f1a44f2ef9e6c56803ee1a9b6',
    'tracked_files_verified': len(tracked), 'changed_paths': sorted(changes),
    'identical_generated_files': generated,
}, indent=2))
