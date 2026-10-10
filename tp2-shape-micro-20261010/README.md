# PR #60403: L20 measurements at model TP2 per-rank dimensions

Unchanged production head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6.
Two independent single-GPU runs, each H8/HV24, K=V=128: these are the
Qwen3.8-27B-FP8 TP2 per-rank dimensions, not distributed TP timing.
The raw shape.TP=1 describes the benchmark; model_TP=2 describes the model.

The measured GPU operation is production GDN decode plus identical gated
RMSNorm, after convolution. Public defaults were retained; no private tuning.
State is FP32, QKV/gates BF16, indices int32. Only recurrent state rotates
for the rotated condition (about 482/492 MiB for B1/B8); QKV/gates/output
are reused. No physical cache-hit/HBM counters were measured.

Each GPU used one fresh process and five timing rounds per condition.
Requested CUPTI was unavailable: all timings actually use the documented
CUDA-graph-event fallback. Values are medians of five round medians.
The rotated B1 condition has only 6 Triton / 8 FI event samples per round;
this is a bounded descriptive measurement, not a confidence interval.

| Physical GPU | Batch | State | Triton us | FI us | FI time change |
|---|---|---|---:|---:|---:|
| 4 | 1 | warm | 5.645 | 4.051 | -28.23% |
| 4 | 1 | rotated | 9.022 | 7.168 | -20.55% |
| 4 | 8 | warm | 14.330 | 11.322 | -20.99% |
| 4 | 8 | rotated | 35.974 | 38.010 | +5.66% |
| 6 | 1 | warm | 5.654 | 4.064 | -28.13% |
| 6 | 1 | rotated | 9.017 | 7.158 | -20.62% |
| 6 | 8 | warm | 14.365 | 11.267 | -21.56% |
| 6 | 8 | rotated | 35.976 | 38.008 | +5.65% |

Negative means lower GPU operation time. Warm B1/B8 and rotated B1 improve
in both runs; rotated B8 is 5.65–5.66% slower. This does not establish serving
latency/throughput improvement. Collectives, TP critical path, projections,
convolution and Python dispatch/detach CPU overhead are outside this timer.
The observations support keeping FI opt-in, not automatic batch/cache routing.

Both GPU processes exited 0. All six correctness configurations passed
(B1/B8 unpadded plus B8 padded on each card): one step, 128-step trajectory,
graph replay, inactive-state and padding sentinels. Existing pointwise
atol/rtol=1e-2 and relative-L2<1% thresholds were not changed. Maximum
recorded relative L2 across 36 metrics is 2.784023e-5.
This checks this operation, not full-model output equivalence.

The independent resource audit passed all 8/8 compute observations with
no unmatched/unknown/partial records. Cleanup exited 0; both GPUs were released.
Earlier C8 graph negatives and the late-prefix ownership REVIEW_REQUIRED
remain separate; this run does not erase them.

run_cells.sh records exact GPU commands. original-benchmark.py and the
producer manifest retain provenance: only shape acceptance and scope/GPU
metadata changed. The original auditor/receipt are preserved. v2 only permits
a fresh output path for reproducible CPU checks:

    uv run --offline --no-project <venv-python> -X utf8 audit_results_v2.py . rechecked-micro.json
    uv run --offline --no-project <venv-python> -X utf8 resource_audit.py --results results --manifest producer-manifest.json --expected-owner gdn60403-tp2shape-20261010 --output rechecked-resources.json

Public copies redact private IPv4 text only; raw/public hashes and byte ranges
are mapped. Original audit hashes describe raw inputs; public rechecks retain
their own input hashes. Full raw files remain local. AI assistance was used.
