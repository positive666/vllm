"""Audit dependencies visible to the inherited-site-packages runtime."""
import base64
import csv
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
from pathlib import Path
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name

ROOT=Path('/opt/image-lmcache-lifecycle-20261104-r1')
lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
site=Path(md.distribution('vllm').locate_file(''))
errors=[]
accepted_overrides=[]
files=0
for project,name in [('vllm','vllm'),('lmcache','lmcache'),('humming','humming-kernels')]:
    for path,digest,size in csv.reader(io.StringIO(md.distribution(name).read_text('RECORD'))):
        if not path.startswith(project+'/') or not digest:
            continue
        target=site/path
        algorithm,value=digest.split('=',1)
        assert algorithm=='sha256'
        actual=base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest()).decode().rstrip('=') if target.is_file() else None
        if actual!=value:
            errors.append('RECORD mismatch '+path)
        files+=1
for row in lock['patches']:
    if hashlib.sha256((site/row['path']).read_bytes()).hexdigest()!=row['candidate_sha256']:
        errors.append('Patch mismatch '+row['path'])
numa=site/'vllm/utils/numa_wrapper.sh'
if not numa.is_file() or not os.access(numa,os.X_OK):
    errors.append('NUMA wrapper missing or not executable')
for name in ('NCCL_NVLS_ENABLE','NVSHMEM_DISABLE_NVLS','VLLM_ALLREDUCE_USE_SYMM_MEM'):
    if name in os.environ:
        errors.append('Unexpected image override '+name)
pending=[('vllm',''),('lmcache',''),('humming-kernels','cu13')]
visited=set()
while pending:
    name,extra=pending.pop()
    key=(canonicalize_name(name),extra)
    if key in visited:
        continue
    visited.add(key)
    for entry in md.distribution(name).requires or []:
        requirement=Requirement(entry)
        if requirement.marker and not requirement.marker.evaluate(default_environment()|{'extra':extra}):
            continue
        try:
            version=md.version(requirement.name)
        except md.PackageNotFoundError:
            errors.append(f'Missing {name}: {entry}')
            continue
        if not requirement.specifier.contains(version,prereleases=True):
            if canonicalize_name(requirement.name) == 'nvidia-nccl-cu13' and version == '2.31.2':
                accepted_overrides.append(dict(parent=name,requirement=entry,installed=version,
                    reason='vLLM Dockerfile NCCL_VERSION=2.31.2; DeepEP rebuilt with same version'))
            else:
                errors.append(f'Conflict {name}: {entry}, found {version}')
        pending.append((requirement.name,''))
        pending.extend((requirement.name,value) for value in requirement.extras)
import torch
import runpy
nccl_probe=runpy.run_path(str(ROOT/'nccl_probe.py'))
import deep_ep
torch.ops.load_library(site/'vllm/_C_stable_libtorch.abi3.so')
import lmcache.cuda_ops
import lmcache.lmcache_native
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.prefetch_controller import PrefetchController
from lmcache.v1.multiprocess.modules.lookup import LookupModule
import vllm.envs as envs
from vllm.engine.arg_utils import EngineArgs
assert envs.VLLM_ALLREDUCE_USE_SYMM_MEM
assert EngineArgs.__dataclass_fields__['numa_bind'].default is None
result=dict(status='PASS' if not errors else 'FAIL',errors=errors,
    dependency_nodes=len(visited),record_files=files,patch_files=len(lock['patches']),
    runtime_interpreter=str(Path(os.sys.executable)),
    accepted_official_overrides=accepted_overrides,nccl_runtime=nccl_probe['versions'],
    torch_compiled_nccl=torch.cuda.nccl.version(),deep_ep_native=str(deep_ep._C.__file__),
    vllm_version=md.version('vllm'),lmcache_version=md.version('lmcache'),
    humming_version=md.version('humming-kernels'),native_imports=True,default_workarounds_added=False,
    note='Runtime includes system site packages; uv pip check sees only the venv-owned subset.')
Path('/work/RUNTIME_AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
raise SystemExit(bool(errors))
