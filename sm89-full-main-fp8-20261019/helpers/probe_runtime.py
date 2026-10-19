"""Check the real source/runtime combination before model measurements."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import torch
import vllm
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import fused_recurrent

assert torch.cuda.device_count() == 2
assert torch.version.cuda.startswith('13.')
assert Path(vllm.__file__).resolve().is_relative_to('/work/source')
assert current_platform.is_device_capability(89)
devices = []
for index in range(2):
    with torch.cuda.device(index):
        value = torch.ones((32, 32), device=f'cuda:{index}', dtype=torch.bfloat16)
        assert torch.equal(value @ value, torch.full_like(value, 32))
        torch.cuda.synchronize()
        devices.append({'index': index, 'name': torch.cuda.get_device_name(index),
                        'capability': torch.cuda.get_device_capability(index)})
loaded = Path('/proc/self/maps').read_text().splitlines()
result = {
    'python': sys.executable, 'vllm_file': vllm.__file__,
    'decode_module': fused_recurrent.__file__,
    'decode_sha256': hashlib.sha256(Path(fused_recurrent.__file__).read_bytes()).hexdigest(),
    'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
    'devices': devices,
    'cuda_libraries': sorted({line.split()[-1] for line in loaded if 'libcuda' in line}),
    'packages': {d.metadata['Name']: d.version for d in metadata.distributions()},
}
print(json.dumps(result, indent=2))
