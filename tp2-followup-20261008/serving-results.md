# Two-L20 TP2 serving performance

Source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`, local GDN shape H8/HV24/K128/V128 per rank.
Both backends use 64 GPU blocks and verified
21,845 KV tokens, BF16 execution, FP32 SSM
state, Marlin and FlashAttention2, with prefix caching disabled.

ABBA launches Triton/FI/FI/Triton share one fixed 512-input/128-output-token
fixture. C1/C8 each have three rounds of 16 requests plus eight warmups per
concurrency. All 384 measured and
64 warmup requests complete the fixed token protocol
without client errors; startup/warmup are excluded from performance measurements.
Closed-loop workers and loopback HTTP are included in this serving boundary.

| Launch | C | Output tokens/s | TTFT p50 ms | TPOT p50 ms |
|---|---|---|---|---|
| triton-a1 | 1 | 44.541503 | 172.706327 | 21.263700 |
| flashinfer-b1 | 1 | 44.714064 | 172.605970 | 21.176186 |
| flashinfer-b2 | 1 | 44.704385 | 172.856772 | 21.178497 |
| triton-a2 | 1 | 44.540295 | 172.563845 | 21.263694 |
| triton-a1 | 8 | 228.527523 | 847.469010 | 27.345048 |
| flashinfer-b1 | 8 | 229.002145 | 844.462439 | 27.224695 |
| flashinfer-b2 | 8 | 229.900691 | 810.465654 | 28.334594 |
| triton-a2 | 8 | 228.457473 | 847.221177 | 27.344613 |

Each launch value is the median of its three round metrics. Overall changes
compare medians of the two launch medians per backend; pair 1 is B1/A1 and
pair 2 B2/A2. Positive means higher FI throughput or lower FI TTFT/TPOT.
Negative results are retained; no observed-range significance rule is applied.

| C | Metric | FI improvement | Pair 1 | Pair 2 |
|---|---|---|---|---|
| 1 | output_tokens_per_second | +0.3779% | +0.3874% | +0.3684% |
| 1 | ttft_p50_ms | -0.0558% | +0.0581% | -0.1697% |
| 1 | tpot_p50_ms | +0.4061% | +0.4116% | +0.4007% |
| 8 | output_tokens_per_second | +0.4197% | +0.2077% | +0.6317% |
| 8 | ttft_p50_ms | +2.3463% | +0.3548% | +4.3384% |
| 8 | tpot_p50_ms | -1.5901% | +0.4401% | -3.6204% |

FI C8 TPOT is **1.59% slower**; paired signs differ.

These are descriptive results from two independent launches per backend,
not statistically established gains, equivalence or a general serving speedup.
Requests within a launch are not independent process/hardware replications.

Five-second host snapshots attribute selected-GPU compute PIDs to the owned
container; sampled foreign/unattributed PID count is
0. Approximate measured windows use remote
original JSON mtimes minus measured wall duration, with write/serialization
uncertainty. Sampled per-launch/concurrency clock medians and temperature ranges:

| GPU UUID | Median SM clock range MHz | Temperature range C |
|---|---|---|
| GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e | 2520–2520 | 69–84 |
| GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675 | 2520–2520 | 67–82 |

Matching sampled clocks cannot exclude transient throttling between samples.
Topology/P2P outputs, full clock ranges and PID evidence remain in the archive.
No clock, fan or power settings were changed for this run.

No TP2 decode microbenchmark was run. Earlier TP1 GDNDecode + RMSNorm GPU
latencies use a different measurement boundary; they are not full-model latency
changes and cannot be transferred to TP2. See the earlier
[TP1 evidence](../performance-followup-20261008) for its recorded CUDA graph
events fallback, warm-state gains, unfavorable rotated-state B8 median and
synthetic HV4 negative results.

Native libraries are reused with recorded hashes. SM80 hardware, H20, BF16 state,
a fresh full CUDA13 build and broader workloads remain unvalidated.

An optional DeepEP import probe is unavailable: the caught
`AssertionError: Cannot find package: nccl` is retained in startup logs. The dense
TP2 serving path does not exercise DeepEP; the separate two-rank NCCL correctness
check passes. DP+EP remains unvalidated, and these results do not establish a
fully validated runtime environment. Independent startup/log classification notes:

- triton-a1: recorded log SHA verifies the first 127709 bytes; 4610 appended bytes separately match the closeout manifest. Tail contains allowed late buffered NCCL informational output and resource_tracker shutdown cleanup; it is not described as only teardown. Original record/raw bytes were not changed.
- triton-a1: 4 exact DeepEP import traceback blocks before application readiness are retained and classified as caught optional dependency failures. import_utils._has_module catches Exception and returns False; this is not a claim that the whole log is warning-free or that NCCL transport is unavailable. Unknown tracebacks and actual CUDA/NCCL/worker/API errors still fail the audit.
- flashinfer-b1: recorded log SHA verifies the first 145125 bytes; 4612 appended bytes separately match the closeout manifest. Tail contains allowed late buffered NCCL informational output and resource_tracker shutdown cleanup; it is not described as only teardown. Original record/raw bytes were not changed.
- flashinfer-b1: 4 exact DeepEP import traceback blocks before application readiness are retained and classified as caught optional dependency failures. import_utils._has_module catches Exception and returns False; this is not a claim that the whole log is warning-free or that NCCL transport is unavailable. Unknown tracebacks and actual CUDA/NCCL/worker/API errors still fail the audit.
- flashinfer-b2: 4 exact DeepEP import traceback blocks before application readiness are retained and classified as caught optional dependency failures. import_utils._has_module catches Exception and returns False; this is not a claim that the whole log is warning-free or that NCCL transport is unavailable. Unknown tracebacks and actual CUDA/NCCL/worker/API errors still fail the audit.
- triton-a2: 4 exact DeepEP import traceback blocks before application readiness are retained and classified as caught optional dependency failures. import_utils._has_module catches Exception and returns False; this is not a claim that the whole log is warning-free or that NCCL transport is unavailable. Unknown tracebacks and actual CUDA/NCCL/worker/API errors still fail the audit.
- triton-compat: recorded log SHA verifies the first 131171 bytes; 4352 appended bytes separately match the closeout manifest. Tail contains allowed late buffered NCCL informational output and resource_tracker shutdown cleanup; it is not described as only teardown. Original record/raw bytes were not changed.
- triton-compat: 4 exact DeepEP import traceback blocks before application readiness are retained and classified as caught optional dependency failures. import_utils._has_module catches Exception and returns False; this is not a claim that the whole log is warning-free or that NCCL transport is unavailable. Unknown tracebacks and actual CUDA/NCCL/worker/API errors still fail the audit.
- flashinfer-compat: 4 exact DeepEP import traceback blocks before application readiness are retained and classified as caught optional dependency failures. import_utils._has_module catches Exception and returns False; this is not a claim that the whole log is warning-free or that NCCL transport is unavailable. Unknown tracebacks and actual CUDA/NCCL/worker/API errors still fail the audit.

`tp2-evidence.zip`: 129 immutable raw files, 5,372,696
bytes; SHA256 `775cf833e113209b0d649bc1873c9b06d1d9d8b6ed0a54f18a400019e0120b44`. All independent audits pass; their pass
status establishes evidence integrity, not favorable performance or quality.
