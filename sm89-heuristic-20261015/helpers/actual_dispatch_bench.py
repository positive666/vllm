"""Actual upstream/candidate wrappers; correctness before cold CUPTI timing."""

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
import sweep


def profile_cold(fn, repeats):
    """Use Torch's bundled CUDA 12.9 CUPTI; exclude L2 flushing by kernel name."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size,
                        dtype=torch.int8, device='cuda')
    for _ in range(5):
        flush.zero_()
        graph.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            flush.zero_()
            graph.replay()
        torch.cuda.synchronize()
    events = [event for event in prof.events()
              if event.device_type == torch.autograd.DeviceType.CUDA
              and 'fused_recurrent_gated_delta_rule_packed_decode_kernel' in event.name]
    assert len(events) == repeats, [(e.name, e.device_type) for e in prof.events()]
    return [e.device_time_total / 1000 for e in events]


def load(arm, root, expected):
    path = Path(root) / 'vllm/third_party/flash_linear_attention/ops/fused_recurrent.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    name = f'vllm.third_party.flash_linear_attention.ops._actual_{arm}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LaunchProbe:
    def __init__(self, kernel):
        self.kernel, self.tiles = kernel, []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.tiles.append(kwargs['BV'])
            return self.kernel[grid](*args, **kwargs)
        return run


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timer', choices=['flashinfer', 'torch-cupti'], default='torch-cupti')
    parser.add_argument('--shared-timing-storage', action='store_true')
    args = parser.parse_args()
    modules = {
        32: load('base', '/baseline', 'b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e'),
        4: load('candidate', '/source', 'bd51c3421cfb7ecb7d8d8b5eafd6536d9023f4269f6e8b6d8705ae457fd69a36'),
    }
    functions = {k: m.fused_recurrent_gated_delta_rule_packed_decode for k, m in modules.items()}
    sweep.TILES = (32, 4)  # Existing Case helper now compares two actual wrappers.
    probes = {}
    for arm, module in modules.items():
        probes[arm] = LaunchProbe(module.fused_recurrent_gated_delta_rule_packed_decode_kernel)
        module.fused_recurrent_gated_delta_rule_packed_decode_kernel = probes[arm]
    pairs = [(2, 2), (2, 4), (2, 6), (2, 8), (4, 4), (4, 8), (4, 12),
             (4, 16), (8, 8), (8, 16), (8, 24), (8, 32), (16, 16),
             (16, 32), (16, 48), (16, 64)]
    cases = sorted({(b, h, hv, 'float32') for h, hv in pairs
                    for b in (1, max(1, 64 // hv), 64 // hv + 1)} |
                   {(b, h, hv, 'bfloat16') for h, hv in pairs
                    for b in (1, 64 // hv + 1)})
    if args.smoke:
        cases = [(1, 8, 24, 'float32')]
    rounds, repeats = (1, 10) if args.smoke else (7, 60)
    rng = random.Random(4194301)
    warm_a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
    warm_b = torch.empty_like(warm_a)
    warm_until = time.monotonic() + 3
    while time.monotonic() < warm_until:
        torch.mm(warm_a, warm_a, out=warm_b)
        torch.cuda.synchronize()
    del warm_a, warm_b
    start = time.monotonic()
    with open(args.output, 'w') as out:
        def emit(record):
            out.write(json.dumps(record) + '\n')
            out.flush()
        emit({'kind': 'metadata', 'scope': 'Actual upstream/candidate wrapper; kernel-only, CUDA 12.9; not full-main serving',
              'baseline': '/baseline', 'candidate': '/source', 'cases': len(cases),
              'rounds': rounds, 'repeats': repeats, 'device': str(torch.cuda.get_device_properties(0)),
              'shared_timing_storage': args.shared_timing_storage,
              'timer': args.timer + '; CUDA Graph; flush 2xL2 before each replay, excluded from kernel timing',
              'timing_helper_sha256': hashlib.sha256(inspect.getsource(
                  profile_cold if args.timer == 'torch-cupti' else bench_gpu_time_with_cupti).encode()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        if not args.smoke:
            for b, h, hv, dtype in [(3, 8, 24, torch.float32), (7, 4, 8, torch.float32),
                                    (8, 4, 8, torch.float32), (9, 4, 8, torch.float32),
                                    (3, 8, 24, torch.bfloat16)]:
                case = sweep.Case(b, h, hv, dtype, functions, 104729, padding=True)
                emit({'kind': 'multistep', 'B': b, 'H': h, 'HV': hv, 'dtype': str(dtype),
                      **case.check(fp64=True, steps=128)})
                del case
        for i, (b, h, hv, dtype_name) in enumerate(cases):
            dtype = getattr(torch, dtype_name)
            case = sweep.Case(b, h, hv, dtype, functions, 4194301 + i)
            key = {'B': b, 'H': h, 'HV': hv, 'dtype': dtype_name}
            correctness = case.check(fp64=b == 1)
            tiles = {str(arm): probes[arm].tiles[-1] for arm in modules}
            assert tiles['32'] == 32
            expected = 4 if dtype == torch.float32 and b * hv <= 64 else 32
            assert tiles['4'] == expected, (key, tiles)
            emit({'kind': 'correctness', **key, 'selected_tiles': tiles, **correctness})
            if args.shared_timing_storage:
                case.storage[4] = case.storage[32]
                case.states[4] = case.states[32]
                case.outputs[4] = case.outputs[32]
            for round_index in range(rounds):
                order = [32, 4]
                rng.shuffle(order)
                for arm in order:
                    case.reset(arm)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        if args.timer == 'torch-cupti':
                            samples = profile_cold(lambda: case.run(arm), repeats)
                        else:
                            samples = bench_gpu_time_with_cupti(
                                lambda: case.run(arm), dry_run_iters=5, repeat_iters=repeats,
                                use_cuda_graph=True, cold_l2_cache=True)
                    messages = [str(w.message) for w in caught]
                    assert not any('falling back' in m.lower() for m in messages), messages
                    assert len(samples) == repeats and min(samples) > 0
                    med = statistics.median(samples) * 1000
                    nbytes = 2 * b * hv * 128 * 128 * case.states[arm].element_size()
                    emit({'kind': 'timing', **key, 'arm': 'base' if arm == 32 else 'candidate',
                          'round': round_index, 'BV': tiles[str(arm)], 'median_us': med,
                          'samples_us': [v * 1000 for v in samples], 'warnings': messages,
                          'logical_state_rw_bytes': nbytes, 'logical_state_GBps': nbytes / med / 1000})
            print(json.dumps({'done': i + 1, 'total': len(cases), **key}), flush=True)
            del case
            gc.collect()
        emit({'kind': 'complete', 'cases': len(cases), 'elapsed_s': time.monotonic() - start})


if __name__ == '__main__':
    main()
