"""Run the existing GPU suite against the exact candidate module.

Only the packed-decode module is replaced. Supporting Python/native dependencies
come from the recorded container image; this is not a full-main serving test.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import vllm.third_party.flash_linear_attention.ops as ops
import vllm.third_party.flash_linear_attention.ops.op as support

path = Path('/source/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py')
expected = 'bd51c3421cfb7ecb7d8d8b5eafd6536d9023f4269f6e8b6d8705ae457fd69a36'
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
support_base = Path('/source/vllm/third_party/flash_linear_attention/ops/op.py')
assert Path(support.__file__).read_text() == support_base.read_text(), 'op.py differs'
name = 'vllm.third_party.flash_linear_attention.ops._candidate_actual'
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
ops.fused_recurrent_gated_delta_rule = module.fused_recurrent_gated_delta_rule
ops.fused_recurrent_gated_delta_rule_packed_decode = (
    module.fused_recurrent_gated_delta_rule_packed_decode
)
print(json.dumps({'candidate_source': str(path), 'sha256': expected,
                  'support_op_matches_upstream': True}), flush=True)
raise SystemExit(pytest.main([
    '/source/tests/kernels/fla/test_fused_recurrent_packed_decode.py',
    '--confcutdir=/source/tests/kernels/fla', '-c', '/dev/null',
    '-v', '-ra', '-p', 'no:cacheprovider',
    '--junitxml=/results/gpu-pytest.xml',
]))
