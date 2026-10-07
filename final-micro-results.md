# Final L20 production GDN decode comparison

The final FP32-state integration passes all 10 supported correctness cases and
128-step trajectories. FlashInfer improves B1 and warm B8 in this measured
post-convolution boundary; cold B8 is 0.91% slower and B32 differences are small.
Keep `auto` on Triton. The separate final serving and quality reports show
near-baseline serving and the bounded quality screen. BF16 recurrent state
and H20/SM90 are not covered by this passing run.

## Final fixture and provenance

- One NVIDIA L20, SM89, 96 MiB L2, TP=1; H=16, HV=48, K=V=128.
- These dimensions match the local folder labeled `Qwen3.8-27B-FP8`; its config
  is `Qwen3_5ForConditionalGeneration` / `qwen3_5`, used in text-only mode.
  No official Qwen3.8 model identity is inferred from that folder label.
- BF16 packed QKVZ/BA with production row strides, FP32 recurrent state with
  padded page strides. B1/B8/B32, int32/int64 indices; B8/B32 test 0/-1 padding.
- All 48 checkpoint `dt_bias` tensors are BF16. Both arms read the same BF16
  representable values: Triton uses BF16, FI uses an exact FP32 promotion
  prepared once outside timing. A_log is FP32. The FP64 reference uses the same
  quantized bias values. Earlier synthetic-FP32-bias results are preliminary.
- Torch 2.13.0+cu129, Triton 3.7.1, official FI 0.7.0.post1, CUTLASS DSL 4.7.1.
  Python source is main `3e182185aa5d143b0e69c44607f58a0cc55f3971` plus this
  patch; native runtime is reused `0.30.1rc1.dev143+g29468dde8.cu129`.
  This is a mixed precompiled runtime validation, not a native build of main.

Raw result: `final-micro-production.json`, SHA256
`c23433198975bd0e111f78a823265c26af8b60e7f4ab159dca09348958c49409`.
Final GDN source:
`674ea2aef2cf945e23b37bb1bc3a9e6a7c861df20561eebbc4131ee6b5faa963`.
Harness:
`b4d6fd15f6617790a00000f93982419b5a05a1e4179a395b2f5325916a3ad3ed`.
Run UTC: 2026-10-06 14:54:04–14:54:23. `final-micro-summary.json` recomputes
the table and ranges from this immutable raw JSON.

## Correctness

All cases check a first-step FP64 mathematical reference, then a CUDA graph
trajectory using fresh inputs and changing state-slot mappings. Raw output,
normalized output and state use the unchanged atol=rtol=1e-2 pointwise gate,
plus relative L2 <1%. Padded output rows are ignored by the serving contract.
All inactive state, including null slot 0, and every cache-page padding element
remain bitwise unchanged throughout; these invariants are not approximate.

Maximum trajectory relative L2: raw output **0.013475%**, normalized output
**0.015861%**, state **0.000011865%**. Maximum absolute differences are
0.000244141, 0.03125 and 3.57628e-7, respectively. There are **zero pointwise
tolerance violations**. The normalized absolute maximum is evaluated against
the combined absolute-plus-relative gate, not an absolute-only threshold.

## Timed production boundary

The same actual production `GDNDecode` performs gates, Q/K normalization and
recurrent update, followed by identical gated output RMSNorm. Convolution and
projection are outside this boundary. Allocation, compilation, random input
generation, reset and logging are untimed. Three rounds alternate the arm order.
Preallocated pad-index mapping costs B1/B8/B32 **2.179/2.128/2.134 us for one
builder/cache-group remap**. The result is shared by that group's layers in a
model step; multiple GDN cache groups can require multiple remaps. It is separate
from the per-layer table, not charged once per layer or assumed to be one global
2.1-us cost for all 48 layers. The raw JSON's abbreviated once-per-model-step
scope label denotes this single measured remap. End-to-end serving includes the
actual metadata cost of the model's cache groups.

Microseconds per boundary call, median of three round medians. A positive
reduction denotes lower FI latency. Estimated state GB/s counts one read and
one write, omitting QKV, gates and normalization traffic.

| B | Cache | Triton us | FI us | FI latency reduction | State GB/s T / FI |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | warm | 6.438 | 4.858 | 24.55% | 977 / 1295 |
| 8 | warm | 19.952 | 17.254 | 13.52% | 2523 / 2917 |
| 32 | warm | 236.298 | 231.341 | 2.10% | 852 / 870 |
| 1 | cold | 12.128 | 10.411 | 14.16% | 519 / 604 |
| 8 | cold | 74.517 | 75.197 | -0.91% | 675 / 669 |
| 32 | cold | 311.189 | 307.151 | 1.30% | 647 / 655 |

| B | Cache | Triton round min–max us | FI round min–max us |
| ---: | --- | ---: | ---: |
| 1 | warm | 6.438–6.448 | 4.858–4.870 |
| 8 | warm | 19.946–20.093 | 17.254–17.376 |
| 32 | warm | 235.712–236.430 | 231.222–232.102 |
| 1 | cold | 12.122–12.129 | 10.409–10.411 |
| 8 | cold | 74.463–74.563 | 75.186–75.212 |
| 32 | cold | 310.958–311.405 | 307.073–307.261 |

## Timing limits and interpretation

CUPTI is unavailable. Every round records the fallback to **CUDA graph events**;
these are not CUPTI measurements. Cold runs rotate 161/21/6 pointer-distinct
state pools at B1/B8/B32, totaling 483/504/576 MiB active state, over 5x L2.
The fallback helper captures ten supplied groups per graph, returns duration
divided by ten, and the harness divides by the state-rotation count. B1 cold's
five samples per round each average 1,610 boundary calls; their p10/p90 are
grouped averages and cannot describe individual-call or request tails.
[Official timing helper](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/flashinfer/testing/utils.py#L1253)

Warm sample counts per round: B1 Triton 1,527–1,528 / FI 1,943–2,004; B8
493–499 / 569–574; B32 42 / 43. Cold samples are 5/6/5 per arm per round.
One host and three rounds do not establish statistical significance. Report
the measured cold B8 regression rather than rounding it into a win. The B32
changes do not establish a meaningful serving gain.

Warm state traffic can be served by cache, so its estimated GB/s is not HBM
bandwidth. B32 active state alone is 96 MiB, equal to L2 before other buffers;
the nominal warm run cannot be assumed to retain its entire working set.

This compares current public production configurations. The FP32 T=1 FI public
entry and in-tree Triton wrapper do not expose same-algorithm tile/warp/stage
tuning knobs. No FI files or private kernels were changed. Do not infer an
exhaustively tuned optimum or change `auto` based on these measurements.

The separate official BF16-state padding probe changed 785,198 elements in
reserved slot 0, while unused slots and page padding stayed exact. It supports
rejecting BF16-state opt-in; it is not part of the passing FP32 matrix.

The FI-only FP32 bias representation stabilizes this integration's compiled
signature. It is not a blanket public-API requirement: official docs allow
BF16 or FP32 `dt_bias`, and the compiler initially consumes its actual dtype.
See `fi-bias-cache-review.md` for the missing dtype in the compilation cache key.

## Reproduction

With `PYTHONPATH=/source` and the normal owned runtime. The identified reused
native extension is a source-tree `.so` symlink; the final run does not use the
preliminary bootstrap or import shims:

```sh
uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/benchmark_gdn_takeover.py \
  --source-head 3e182185aa5d143b0e69c44607f58a0cc55f3971 \
  --batches 1,8,32 --dtypes float32 --index-dtypes int32,int64 \
  --steps 128 --rounds 3 --output /results/final/micro-production.json
```

No serving or statistical quality-equivalence claim is supported by this
microbenchmark. Final quality uses the predeclared matched GSM8K 100-question
subset with a 1,750-token budget; the preliminary 1,024-token run is excluded.
