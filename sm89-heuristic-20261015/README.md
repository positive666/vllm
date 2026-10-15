# L20 / SM89 packed GDN decode heuristic

Measured source: vllm-project/vllm `fc8e08a2a607968226376c53a30cfa6337578f79`
plus [candidate.patch](candidate.patch). Only the dispatch and three existing test
parameters change. The kernel arithmetic is unchanged.

## Results

- GPU reference suite: 20 passed, no skips. Actual-wrapper comparison: 78 cases
  bit-exact for outputs/state; five padded cases passed 128 changing-input steps.
- Cold-cache selected region: 30 shapes, geometric-mean latency reduction 18.48%
  (15.61–27.71% by shape). The 48 BV32 controls changed +0.11% in aggregate.
- Timing: seven paired rounds, 60 samples per arm/shape, same state/output addresses,
  randomized arm order, CUDA Graph and 2x L2 flush before each replay. Torch's
  bundled CUDA 12.9 CUPTI/Kineto records only the target kernel, excluding the flush.
  Raw samples and per-round results are in [kernel](kernel).
- An earlier separate-buffer run is retained as actual-cold/actual-analysis;
  its noisier controls motivated the same-address run used for the final claims.

Supplemental BF16 27B / TP2, A/B/B/A fresh processes and independent JIT caches:

| Concurrency | Baseline tok/s | Candidate tok/s | Change | TPOT p50 base → candidate (ms) |
|---:|---:|---:|---:|---:|
| 1 | 26.289 | 26.372 | +0.31% | 36.988 → 36.867 |
| 2 | 46.719 | 46.844 | +0.27% | 40.758 → 40.639 |
| 8 | 153.026 | 152.744 | -0.18% | 45.267 → 45.779 |

Each start ran three rounds of 16 requests at each concurrency after warmup;
512 input / 128 forced output tokens. All 576 measured and 96 warmup requests
succeeded. Four arithmetic smoke runs each scored 16/16 without truncation;
13/16 response texts matched across all four runs. This is not a full accuracy eval.
[Summary](tp2/summary.json), [raw requests](tp2/http), and server logs are retained.
The four arm JSON files include the individual rounds and warmups.

## Reproduction

Runtime: L20, Torch 2.13.0+cu129, Triton 3.7.1, FlashInfer 0.6.16.post3,
Python 3.12.3, vLLM 0.27.1+cu129 dependencies. This uses the exact new decode
module with the image runtime, not a full latest-main build. Model architecture:
Qwen3_5ForConditionalGeneration; local BF16 27B checkpoint, not the earlier FP8
checkpoint. The config hash is recorded in each server JSON.

Use an isolated disposable container with baseline source at /baseline, patched
source at /source, helpers at /helpers, writable /results and /cache, and an
uv-managed environment at /cache/.venv. The kernel-only self-contained bundle is
[sm89-kernel-repro.tar.gz](sm89-kernel-repro.tar.gz).

```sh
CUDA_VISIBLE_DEVICES=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --offline --no-project /cache/.venv/bin/python /helpers/run_gpu_pytest.py
CUDA_VISIBLE_DEVICES=0 uv run --offline --no-project /cache/.venv/bin/python /helpers/actual_dispatch_bench.py --shared-timing-storage --output /results/actual-shared.jsonl
```

For TP2, expose two idle GPUs, mount the BF16 checkpoint at /model, and use the
recorded /opt/venv site-packages layout. The helper temporarily replaces the
decode module in the disposable container and restores it on exit.

```sh
uv run --offline --no-project /cache/.venv/bin/python /helpers/run_tp2.py
```

Offline analysis, with an equivalent uv-managed Python environment:

```sh
gzip -dk kernel/actual-shared.jsonl.gz tp2/http/*.json.gz
uv run --offline --no-project .venv/bin/python helpers/analyze_actual.py --input ../kernel/actual-shared.jsonl --output ../kernel/recomputed.json
uv run --offline --no-project .venv/bin/python helpers/analyze_tp2.py --root tp2
```

## Limits and audit

Performance is measured only on L20. Other SM89 cards and SM90 are not validated.
The current B200/MI355 FLA CI does not exercise the SM89 branch. The serving
differences are small; two starts per arm do not establish statistical significance.
C8 TPOT p50 worsened 1.13%. TPOT is a streaming-client estimate, not pure kernel time.

All four runs had 157,696 KV tokens. The retained BV32 binaries have identical code,
constants and runtime metadata; only .debug_line differs. The initial whole-cubin
equality assertion was too strict; the final section-level audit is [audit.json](tp2/audit.json).
There were no matched errors/preemptions in 36 timing windows and no unknown GPU
processes in 140 samples. Monitoring was every 10 seconds and began after A1 started,
so early A1 and brief interference are not fully covered. Setup-time Transformers
documentation warnings are retained in server logs.

GPU UUIDs/container hostname/private host addresses are removed from public copies.
[manifest.json](manifest.json) records original and public-content SHA256 values;
metrics, samples, outputs and source hashes are preserved. AI assistance was used
for implementation, experiments and analysis.
