# Exact-head L20 serving performance

Both independent raw-data audits pass with no errors. Source is
`d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`. All 384 measured and 64 warmup
requests succeed with exactly 512 input/128 output tokens, completed SSE streams
and correct usage/hash records. One fixed prompt, one text-only FP8 checkpoint,
TP1, BF16 execution, FP32 state. Startup and warmup excluded.

Four independent server processes ran Triton/FI/FI/Triton (ABBA). Each measured
16 requests per round at C1/C8 for three rounds. Values in this table are the
median of the three per-round metrics within each launch.

| Launch | C | Output tokens/s | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|---:|
| Triton A1 | 1 | 24.363427 | 295.113240 | 39.043238 |
| FI B1 | 1 | 24.414720 | 295.812314 | 38.944496 |
| FI B2 | 1 | 24.415827 | 295.864826 | 38.944438 |
| Triton A2 | 1 | 24.356097 | 295.772133 | 39.047701 |
| Triton A1 | 8 | 133.706484 | 1556.508284 | 48.210096 |
| FI B1 | 8 | 134.122580 | 1538.148986 | 48.234462 |
| FI B2 | 8 | 134.038386 | 1541.641582 | 48.239098 |
| Triton A2 | 8 | 133.763774 | 1555.455044 | 48.178610 |

Positive values in this table mean FI improvement: higher throughput, lower
TTFT/TPOT. The overall comparison uses the median of each backend's two launch
medians. Pair 1 is B1/A1; pair 2 is B2/A2. Maximum range means the larger observed
relative within-launch round range or between-launch range, not a confidence bound.

| C | Metric | Overall improvement | Pair 1 | Pair 2 | Maximum observed range | Descriptive result |
|---|---|---:|---:|---:|---:|---|
| 1 | Output tokens/s | +0.2279% | +0.2105% | +0.2452% | 0.0301% | small consistent observed gain |
| 8 | Output tokens/s | +0.2582% | +0.3112% | +0.2053% | 0.2116% | small consistent observed gain |
| 1 | TTFT p50 | -0.1340% | -0.2369% | -0.0313% | 0.5286% | inconclusive |
| 8 | TTFT p50 | +1.0338% | +1.1795% | +0.8881% | 0.8341% | descriptive improvement |
| 1 | TPOT p50 | +0.2587% | +0.2529% | +0.2645% | 0.0114% | small consistent observed gain |
| 8 | TPOT p50 | -0.0880% | -0.0505% | -0.1255% | 0.1195% | inconclusive |

**Conclusion:** this bounded run measures small throughput increases, around
0.2%–0.3%, rather than a material/general serving acceleration. C8 TPOT and C1
TTFT do not establish improvement. There are only two independent launches per
backend. The observed-range rule is descriptive, not a significance test; this
does not prove statistical equivalence, robustness across workloads, or an
optimal implementation. Request count is not independent experiment count.
`audit.json` retains full precision, every round and both paired effects.

The microbenchmark separately establishes a faster warm-state GDN GPU boundary
for H16/HV48, with an unfavorable rotated-state B8 median and no measured
synthetic HV4 benefit. See `micro-results.md`; those percentages must not be
presented as full-model throughput gains.

Telemetry audit matches all 32 approximate warmup/measured windows. Five-second
sampling gives 50–51 C1 measured samples and nine C8 samples per launch. All four
launches have measured median SM clock 2520 MHz and memory clock 9000 MHz at each
concurrency; sampled median paired frequency differences are zero. Measured
temperatures span 68–86 C and SM clocks vary below their median (C1 minima
2430–2445 MHz; C8 minima 2325–2340 MHz). This does not exclude transient clock or
thermal confounding. Windows derive from original remote JSON mtimes and wall
durations, with serialization/write uncertainty. No clock/fan/power setting was
changed. `telemetry-audit.json` preserves all ranges and limitations.

The reused native binary and pinned dependency wrapper are retained in the
runtime record; optional `vllm._C` is absent, while the required stable-libtorch
library/imports pass. Native MTP GPU execution is not tested. SM80 hardware, H20, CUDA13
full builds, TP and broader checkpoints/workloads remain unvalidated.

`performance-evidence.zip`: 61 immutable raw files, 3,205,210 bytes,
SHA256 `b8ede3808ebd6278ffe55c9f8dfa4e92e62aadd6c53b90730e1230f7a19ca7b3`.
`archive-manifest.json` records every member hash. The experiment container is
stopped; its GPU is released (1 MiB, 0%, no compute PIDs).
