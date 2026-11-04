"""Verify the integrated wheel and unchanged parent runtime before sealing it."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

ROOT = Path('/opt/image-lmcache-lifecycle-20261104-r1')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
lock = json.loads((ROOT / 'BASE_SOURCE_LOCK.json').read_bytes())
inputs = json.loads((ROOT / 'INTEGRATION_INPUTS.json').read_bytes())
site = Path(md.distribution('lmcache').locate_file(''))
dependencies = dict(lock['dependencies'])
dependencies['lmcache'] = inputs['lmcache_version']
for name, version in dependencies.items():
    assert md.version(name) == version, (name, md.version(name), version)
for name, digest in lock['native'].items():
    assert sha(site / name) == digest, ('Parent native changed', name)
for name, digest in lock['jit_headers'].items():
    assert sha(site / name) == digest, ('Parent JIT header changed', name)
checked = 0
generated = {}
for name, digest in json.loads((ROOT / 'SOURCE_FILES.json').read_bytes()).items():
    target = site / name
    if name.endswith('.py') or target.is_file():
        if '/_proto_gen/' in name and name.endswith(('_pb2.py', '_pb2.pyi', '_pb2_grpc.py')):
            assert target.is_file(), ('Generated binding missing', name)
            generated[name] = dict(source_sha256=digest, installed_sha256=sha(target),
                                   generator='grpcio-tools==1.78.0; official setup.py build_py')
            continue
        assert target.is_file() and sha(target) == digest, ('Source mismatch', name)
        checked += 1
patches = json.loads((ROOT / 'PATCHES.json').read_bytes())
for row in patches:
    assert sha(site / row['path']) == row['candidate_sha256'], row['path']
native = {path.relative_to(site).as_posix(): sha(path)
          for path in (site / 'lmcache').rglob('*.so')}
assert native, 'No rebuilt LMCache native libraries'
assert sha(site / 'lmcache/_version.py') == inputs['release_metadata']['release-version.py']
wheels = list(Path('/build/wheels').glob('lmcache-*.whl'))
assert len(wheels) == 1
lock['wheel_sha256'].update({path.name: sha(path) for path in wheels})
lock.update(lmcache_source_commit=inputs['upstream_base'],
            lmcache_source_tree=inputs['source_tree'],
            lmcache_version=inputs['lmcache_version'],
            lmcache_pr_heads={key: inputs[key] for key in ('pr6067', 'pr6068', 'pr6171')},
            lmcache_native=native, base_image=inputs['base_image'],
            lmcache_generated_bindings=generated,
            lmcache_release_metadata=inputs['release_metadata'],
            dependencies=dependencies, patches=patches,
            update_kind='rebuilt-lmcache-lifecycle-with-fixed-parent-runtime',
            model_validation='see-external-acceptance-receipt')
(ROOT / 'SOURCE_LOCK.json').write_text(json.dumps(lock, indent=2))
receipt = dict(status='PASS_INSTALL', lmcache_source_files_checked=checked,
               parent_native_files_unchanged=len(lock['native']),
               lmcache_native_files=len(native), patch_files=len(patches),
               dependency_versions_checked=len(dependencies),
               default_workarounds_added=False, model_validation='pending')
(ROOT / 'INSTALL_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
