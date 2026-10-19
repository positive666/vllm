"""Record actual packed GDN launches in diagnostic workers only."""

import json
import os
from pathlib import Path


class WorkerExtension:
    def init_worker_extension(self):
        from vllm.third_party.flash_linear_attention.ops import fused_recurrent

        target = fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
        rank = self.rank
        destination = Path(os.environ['GDN_DIAGNOSTIC_DIR'])
        destination.mkdir(parents=True, exist_ok=True)
        seen = set()
        worker = self
        backends_written = False

        class ObserveLaunch:
            def __getattr__(self, name):
                return getattr(target, name)

            def __getitem__(self, grid):
                launch = target[grid]

                def record(*args, **kwargs):
                    nonlocal backends_written
                    if not backends_written:
                        backends = {}
                        for name, module in worker.model_runner.model.named_modules():
                            method = getattr(module, 'quant_method', None)
                            kernel = getattr(method, 'fp8_linear', None)
                            if kernel is not None:
                                backends[name] = type(kernel).__name__
                        (destination / f'rank-{rank}-fp8-backends.json').write_text(
                            json.dumps(backends, indent=2))
                        backends_written = True
                    qkv, state = kwargs['mixed_qkv'], kwargs['h0']
                    row = {
                        'rank': rank, 'pid': os.getpid(),
                        'device': str(qkv.device), 'B': int(qkv.shape[0]),
                        'qkv_dtype': str(qkv.dtype), 'state_dtype': str(state.dtype),
                        'grid': list(grid),
                        **{name: kwargs[name] for name in (
                            'H', 'HV', 'K', 'V', 'BK', 'BV',
                            'USE_QK_L2NORM_IN_KERNEL', 'num_warps', 'num_stages')},
                    }
                    key = json.dumps(row, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        with (destination / f'rank-{rank}.jsonl').open('a') as out:
                            out.write(key + '\n')
                    return launch(*args, **kwargs)

                return record

        fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel = ObserveLaunch()
