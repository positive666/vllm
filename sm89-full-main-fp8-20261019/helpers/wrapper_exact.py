"""Reuse strided/paged correctness checks with full-main actual wrappers."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
import sweep

sweep.TILES = (32, 4)
modules = {}
for key, source, expected in [
    (32, 'baseline', 'b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e'),
    (4, 'candidate', 'bd51c3421cfb7ecb7d8d8b5eafd6536d9023f4269f6e8b6d8705ae457fd69a36'),
]:
    path = Path('/work/source') / source / 'vllm/third_party/flash_linear_attention/ops/fused_recurrent.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    name = f'vllm.third_party.flash_linear_attention.ops._exact_{source}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    modules[key] = module
functions = {k: m.fused_recurrent_gated_delta_rule_packed_decode for k, m in modules.items()}
pairs = [(2, 2), (2, 4), (2, 6), (2, 8), (4, 4), (4, 8), (4, 12),
         (4, 16), (8, 8), (8, 16), (8, 24), (8, 32), (16, 16),
         (16, 32), (16, 48), (16, 64)]
cases = sorted({(b, h, hv, 'float32') for h, hv in pairs
                for b in (1, max(1, 64 // hv), 64 // hv + 1)} |
               {(b, h, hv, 'bfloat16') for h, hv in pairs for b in (1, 64 // hv + 1)})
results = []
with torch.inference_mode():
    for i, (b, h, hv, dtype) in enumerate(cases):
        case = sweep.Case(b, h, hv, getattr(torch, dtype), functions, 4194301 + i)
        results.append({'B': b, 'H': h, 'Hv': hv, 'state_dtype': dtype,
                        **case.check(fp64=b == 1)})
        del case
    for b, h, hv in [(1, 8, 24), (2, 8, 24), (3, 8, 24), (7, 4, 8), (8, 4, 8), (9, 4, 8)]:
        case = sweep.Case(b, h, hv, torch.float32, functions, 104729, padding=b >= 3)
        results.append({'B': b, 'H': h, 'Hv': hv, 'state_dtype': 'float32',
                        **case.check(fp64=True, steps=128)})
        del case
result = {'torch': torch.__version__, 'cuda': torch.version.cuda,
          'single_step_cases': len(cases), 'graph_cases': 6,
          'graph_steps_each': 128, 'all_bit_exact': True, 'results': results}
Path('/work/results/wrapper-exact.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'results'}))
