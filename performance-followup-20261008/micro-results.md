# Exact-head GDN microbenchmark

Source: `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`, NVIDIA L20 (SM89),
Torch 2.13.0+cu129, FlashInfer 0.7.0.post1, CuTeDSL 4.8.0, Triton 3.7.1.
FP32 state, BF16 QKV/gates, K=V=128, int32 indices. Production
`GDNDecode` GPU work plus identical gated RMSNorm; projections and convolution
are outside this boundary. Python dispatch and detach CPU overhead are not timed.
FI gate-copy GPU work is included for HV4. No private kernel tuning.

Values below are medians of three independently timed round medians. Negative
latency change means lower FI latency; it is not the percentage throughput gain.

| H/HV | Batch | State cache condition | Triton us | FI us | FI latency change |
|---|---:|---|---:|---:|---:|
| 16/48 (model TP1) | 1 | warm | 6.448 | 4.909 | -23.87% |
| 16/48 (model TP1) | 8 | warm | 20.173 | 17.261 | -14.44% |
| 16/48 (model TP1) | 1 | rotated | 12.122 | 10.406 | -14.16% |
| 16/48 (model TP1) | 8 | rotated | 74.537 | 75.239 | +0.94% |
| 2/4 (synthetic alignment shape) | 1 | warm | 5.008 | 5.168 | +3.19% |
| 2/4 (synthetic alignment shape) | 8 | warm | 7.466 | 7.728 | +3.51% |
| 2/4 (synthetic alignment shape) | 1 | rotated | 5.430 | 5.541 | +2.04% |
| 2/4 (synthetic alignment shape) | 8 | rotated | 11.673 | 11.733 | +0.51% |

The model shape benefits in the warm-state condition and B1 rotated-state case,
not in all tested conditions. The small-head synthetic shape has no measured
benefit. No ablation isolates clone cost as the cause of its regression.

All six configurations, including padded B8, pass strict pointwise atol/rtol
1e-2 and relative L2 <1%, one-step and 128-step replay, with inactive state and
page-padding sentinels unchanged. Correctness is completed before any timing.

Every timing uses **CUDA graph events**, recorded fallback because CUPTI is
unavailable. Requested CUPTI does not mean CUPTI actually ran. State-only
rotation uses >5x the 96 MiB L2: 483 MiB (model B1), 504 MiB (model B8).
QKV/gates/output are reused; no cache-hit or HBM counters were collected. This
does not establish that all inputs are cold or that every state read reaches HBM.
Reported theoretical state traffic rate is not measured physical HBM bandwidth.
Timing repeatedly updates state with fixed inputs, rather than changing tokens.

Cold synthetic B1 has only one event sample per round; the three rounds do not
establish statistical significance. Raw per-round medians/ranges and timer
warnings are retained in `raw/micro-current.json`.

These GPU timings do not establish end-to-end serving gains. The serving protocol
is frozen separately in `protocol.json` and uses four independent ABBA launches.
