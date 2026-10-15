"""Cold-L2, kernel-only tile sweep; no model or distributed speedup claims.

Contract: packed BF16 QKV/gates plus paged BF16/FP32 state produce BF16
decode output and update only active state slots. Tile changes must preserve
output/state exactly; FP64 checks independently guard reference mistakes.
The cheapest check is the real kernel call, with padded/strided storage and
changing-input CUDA graphs checked before any performance measurements.
"""

import argparse
import datetime as dt
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
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

ROOT = Path(__file__).resolve().parent
SOURCE_HASH = 'b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e'
TILES = (32, 4, 8, 16)
K = V = 128


def load_functions():
    path = ROOT / 'upstream_fused_recurrent.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASH
    name = 'vllm.third_party.flash_linear_attention.ops._cold_sweep_upstream'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    original = module.fused_recurrent_gated_delta_rule_packed_decode
    source = inspect.getsource(original)
    marker = 'BV = min(triton.next_power_of_2(V), 32)'
    assert source.count(marker) == 1
    functions = {32: original}
    for tile in (4, 8, 16):
        namespace = dict(module.__dict__)
        exec(compile(source.replace(marker, f'BV = {tile}'),
                     f'<experimental-bv{tile}>', 'exec'), namespace)
        functions[tile] = namespace[original.__name__]
    return functions


class Case:
    def __init__(self, batch, h, hv, dtype, functions, seed, padding=False):
        torch.manual_seed(seed)
        self.batch, self.h, self.hv = batch, h, hv
        self.functions = functions
        self.packed = torch.randn(batch, 2 * h * K + 2 * hv * V,
                                  device='cuda', dtype=torch.bfloat16)
        self.qkv = self.packed[:, :2 * h * K + hv * V]
        self.ba = torch.randn(batch, 2 * hv, device='cuda', dtype=torch.bfloat16)
        self.b, self.a = self.ba.chunk(2, dim=-1)
        self.alog = torch.randn(hv, device='cuda') * 0.1 - 2
        self.bias = (torch.randn(hv, device='cuda') * 0.1).bfloat16()
        self.indices = torch.arange(batch, 0, -1, device='cuda', dtype=torch.int32)
        if padding:
            self.indices[-2:] = torch.tensor([0, -1], device='cuda')
        self.width = hv * V * K
        self.initial = torch.full((batch + 3, self.width + 256), 0.9375,
                                  device='cuda', dtype=dtype)
        self.initial[:, :self.width].normal_(std=0.01)
        self.storage = {tile: self.initial.clone() for tile in TILES}
        self.states = {tile: pages[:, :self.width].view(batch + 3, hv, V, K)
                       for tile, pages in self.storage.items()}
        self.outputs = {tile: torch.zeros(batch, 1, hv, V,
                                         device='cuda', dtype=torch.bfloat16)
                        for tile in TILES}

    def run(self, tile):
        return self.functions[tile](
            mixed_qkv=self.qkv, a=self.a, b=self.b, A_log=self.alog,
            dt_bias=self.bias, scale=K ** -0.5, initial_state=self.states[tile],
            out=self.outputs[tile], ssm_state_indices=self.indices,
            use_qk_l2norm_in_kernel=True)

    def reset(self, tile=None):
        for key in TILES if tile is None else (tile,):
            self.storage[key].copy_(self.initial)
            self.outputs[key].fill_(0.9375)

    def exact(self):
        for tile in TILES[1:]:
            torch.testing.assert_close(self.outputs[tile], self.outputs[32],
                                       rtol=0, atol=0)
            torch.testing.assert_close(self.storage[tile], self.storage[32],
                                       rtol=0, atol=0)
        active = self.indices[self.indices > 0].long()
        inactive = torch.ones(self.batch + 3, device='cuda', dtype=torch.bool)
        inactive[active] = False
        torch.testing.assert_close(self.storage[32][inactive],
                                   self.initial[inactive], rtol=0, atol=0)
        torch.testing.assert_close(self.storage[32][:, self.width:],
                                   self.initial[:, self.width:], rtol=0, atol=0)
        padded = self.indices <= 0
        assert not self.outputs[32][padded].count_nonzero().item()

    def reference(self):
        active = self.indices > 0
        slots = self.indices[active].long()
        n = len(slots)
        q, k, v = self.qkv[active].double().split(
            [self.h * K, self.h * K, self.hv * V], -1)
        q, k = [x.view(n, self.h, K).repeat_interleave(self.hv // self.h, 1)
                for x in (q, k)]
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * K ** -0.5
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        state = self.initial[:, :self.width].view(self.batch + 3, self.hv, V, K)
        state = state[slots].double()
        decay = (-self.alog.double().exp() *
                 F.softplus(self.a[active].double() + self.bias.double())).exp()
        state *= decay[..., None, None]
        delta = v.view(n, self.hv, V) - (state * k[..., None, :]).sum(-1)
        delta *= self.b[active].double().sigmoid()[..., None]
        state += delta[..., None] * k[..., None, :]
        out = (state * q[..., None, :]).sum(-1).to(torch.bfloat16)
        state = state.to(self.states[32].dtype)
        metrics = {}
        for label, actual, expected in (
            ('output', self.outputs[32][active, 0], out),
            ('state', self.states[32][slots], state),
        ):
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
            delta = actual.float() - expected.float()
            relative = (delta.norm() / expected.float().norm().clamp_min(1e-20)).item()
            assert relative < 0.01, (label, relative)
            metrics[label] = {'max_abs': delta.abs().max().item(), 'relative_l2': relative}
        return metrics

    def check(self, fp64=False, steps=0):
        self.reset()
        for tile in TILES:
            self.run(tile)
        self.exact()
        reference = self.reference() if fp64 else None
        if steps:
            graphs = {}
            for tile in TILES:
                for _ in range(3):
                    self.run(tile)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.run(tile)
                graphs[tile] = graph
            self.reset()
            for _ in range(steps):
                self.packed.normal_()
                self.ba.normal_()
                self.indices.copy_(self.indices.roll(1))
                for tile in TILES:
                    graphs[tile].replay()
                self.exact()
        return {'tile_output_and_full_storage_bit_exact': True,
                'inactive_and_page_padding_unchanged': True,
                'fp64': reference, 'changing_input_graph_steps': steps}


def emit(handle, record):
    handle.write(json.dumps(record) + '\n')
    handle.flush()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', default='1,2,4,8,16,32,64')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeat-iters', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--shapes', default='')
    parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 9), 'L20/SM89 only'
    functions = load_functions()
    manifest = json.loads((ROOT / 'model-shapes.json').read_text())
    pairs = [(row['H'], row['HV']) for row in manifest['shapes']]
    if args.shapes:
        pairs = [tuple(map(int, value.split(':'))) for value in args.shapes.split(',')]
    batches = list(map(int, args.batches.split(',')))
    cases = [(b, h, hv, dtype) for h, hv in pairs for b in batches
             for dtype in (torch.bfloat16, torch.float32)]
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    start = time.monotonic()
    with args.output.open('w') as out:
        emit(out, {'kind': 'metadata', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'seed': args.seed, 'source_sha256': SOURCE_HASH,
                   'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'manifest_sha256': hashlib.sha256((ROOT / 'model-shapes.json').read_bytes()).hexdigest(),
                   'args': vars(args) | {'output': str(args.output)},
                   'timer': 'CUPTI + CUDA graph; cold_l2_cache=True (2xL2 zero before EACH replay); reject fallback',
                   'operation': 'packed recurrent GDN decode only; no RMSNorm',
                   'state_policy': 'Reset outside measurement; repeated graph replays update state in place.',
                   'scope': 'Single GPU local-per-rank shapes, not distributed/model latency.'})
        if args.precheck:
            for h, hv in pairs:
                for dtype in (torch.bfloat16, torch.float32):
                    case = Case(3, h, hv, dtype, functions, args.seed, padding=True)
                    check = case.check(fp64=True, steps=128)
                    emit(out, {'kind': 'padding_correctness', 'H': h, 'HV': hv,
                               'B': 3, 'dtype': str(dtype), **check})
                    del case
                    gc.collect()
            print('padding/graph prechecks passed', flush=True)
        for index, (batch, h, hv, dtype) in enumerate(cases):
            case = Case(batch, h, hv, dtype, functions, args.seed + index)
            key = {'B': batch, 'H': h, 'HV': hv, 'dtype': str(dtype), 'K': K, 'V': V}
            emit(out, {'kind': 'correctness', **key, **case.check(fp64=batch == 1)})
            for round_index in range(args.rounds):
                order = list(TILES)
                rng.shuffle(order)
                for tile in order:
                    case.reset(tile)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        samples = bench_gpu_time_with_cupti(
                            lambda: case.run(tile), dry_run_iters=5,
                            repeat_iters=args.repeat_iters, use_cuda_graph=True,
                            cold_l2_cache=True)
                    messages = [str(row.message) for row in caught]
                    assert not any('falling back' in text.lower() for text in messages), messages
                    assert len(samples) == args.repeat_iters and min(samples) > 0
                    state_bytes = 2 * batch * hv * K * V * torch.tensor([], dtype=dtype).element_size()
                    median_us = statistics.median(samples) * 1000
                    emit(out, {'kind': 'timing', **key, 'BV': tile, 'round': round_index,
                               'samples_us': [x * 1000 for x in samples],
                               'median_us': median_us, 'warnings': messages,
                               'state_read_write_bytes': state_bytes,
                               'effective_state_GBps': state_bytes / median_us / 1000})
            print(json.dumps({'done': index + 1, 'total': len(cases), **key,
                              'elapsed_s': round(time.monotonic() - start, 1)}), flush=True)
            del case
            gc.collect()
        emit(out, {'kind': 'complete', 'cases': len(cases),
                   'elapsed_s': time.monotonic() - start})


if __name__ == '__main__':
    main()
