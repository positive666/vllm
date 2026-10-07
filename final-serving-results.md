# Final L20 Qwen FP8 serving and quality results

On this L20, opt-in FI serving performance is close to the current Triton
backend. The descriptive median paired throughput changes are **+0.279% at
C1** and **+0.113% at C8**. C8's second pair regresses by **0.208%**. TTFT
results vary by statistic and pair. These small differences and two independent
launches per backend do not establish a significant serving improvement.
`auto` remains Triton.

All **768 measured and 64 warmup requests succeed**, with no HTTP failures.
Each has exactly 512 input and 64 output tokens. Completed supervisor records,
embedded ready snapshots and raw log hashes are audited for all four launches.
Each uses the same three final production hashes, runtime and native binary,
64 fixed KV blocks and **21,845 actual KV tokens**, observed in every startup
log. No preliminary A1 or preliminary microbenchmark is included.

## Configuration and measurement

Local checkpoint folder labeled `Qwen3.8-27B-FP8`, whose config identifies
`Qwen3_5ForConditionalGeneration` / `qwen3_5`, used in text-only mode.
No official Qwen3.8 identity is inferred from the folder name. TP=1, BF16 input,
FP32 recurrent state,
Marlin FP8 linear backend, FlashAttention v2, seed 42, no prefix caching,
max model length 2,048, max 32 sequences and max 1,024 batched tokens. The
normalized commands match except `gdn_decode_backend=triton|flashinfer`.
The four independently identified processes launch in A1/B1/B2/A2 order.
Fresh VLLM/Triton/CUDA cache paths are configured by the supervisor; their
configuration is not independently observed inside the result JSON.

Each process has three measured rounds of 32 requests at C1 and C8, plus eight
warmup requests per concurrency. The values below are medians of three round
metrics, not medians pooled across all requests. TTFT measures the first
nonempty text chunk. TPOT divides the time between first and last token-ID
chunks by 63; it is not individual-token interarrival latency. C8 is closed
loop, so new prefills can occur while other requests decode.

## Per-launch medians

All latency values are milliseconds; throughput is output tokens/second.

| Launch | C | tok/s | TTFT mean | TTFT p50 | TPOT mean | TPOT p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton A1 | 1 | 23.218 | 295.251 | 295.329 | 39.057 | 39.057 |
| FI B1 | 1 | 23.285 | 295.205 | 295.230 | 38.930 | 38.928 |
| FI B2 | 1 | 23.275 | 296.493 | 296.411 | 38.930 | 38.930 |
| Triton A2 | 1 | 23.213 | 295.635 | 295.628 | 39.061 | 39.061 |
| Triton A1 | 8 | 104.158 | 1320.156 | 1274.485 | 56.614 | 55.341 |
| FI B1 | 8 | 104.610 | 1305.470 | 1277.869 | 56.513 | 55.228 |
| FI B2 | 8 | 104.382 | 1311.947 | 1276.587 | 56.576 | 55.231 |
| Triton A2 | 8 | 104.599 | 1308.122 | 1259.018 | 56.492 | 55.277 |

Range of the two launch medians for each backend:

| C | Backend | tok/s range | TTFT mean range | TTFT p50 range | TPOT mean range | TPOT p50 range |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Triton | 23.213–23.218 | 295.251–295.635 | 295.329–295.628 | 39.0570–39.0615 | 39.0568–39.0614 |
| 1 | FI | 23.275–23.285 | 295.205–296.493 | 295.230–296.411 | 38.9300–38.9304 | 38.9284–38.9297 |
| 8 | Triton | 104.158–104.599 | 1308.122–1320.156 | 1259.018–1274.485 | 56.4921–56.6139 | 55.2768–55.3412 |
| 8 | FI | 104.382–104.610 | 1305.470–1311.947 | 1276.587–1277.869 | 56.5135–56.5755 | 55.2281–55.2313 |

## Descriptive paired effects

Pairs are B1/A1 and B2/A2. Positive throughput change means faster; positive
latency reduction means lower latency. The median below is over two pair
effects. There is no confidence interval or significance claim. The 768
requests are not 768 independent GPU experiments.

| C | Throughput B1/A1 | Throughput B2/A2 | Median throughput change | Median TTFT mean reduction | Median TTFT p50 reduction | Median TPOT mean reduction | Median TPOT p50 reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.289% | +0.270% | +0.279% | -0.137% | -0.116% | +0.330% | +0.333% |
| 8 | +0.435% | -0.208% | +0.113% | +0.410% | -0.830% | +0.0148% | +0.143% |

The C8 second pair's throughput and mean TPOT regress; C8 TTFT p50 is slower
in both pairs. Preserve these mixed outcomes. Microbenchmark gains at B1 and
warm B8 do not imply a comparable full-model serving gain.

Synthetic-load token sequences match on 192/192 paired C1 requests and
133/192 paired C8 requests (63/96 in pair 1, 70/96 in pair 2). This observable
is separate from model accuracy and does not isolate causes of token changes.

## Matched model-quality screen

Same seeded 100-question GSM8K test subset, zero-shot, temperature 0, seed 42,
thinking disabled, C8 and the 1,750-token budget fixed before either final
backend run. A preliminary 1,024-token Triton run truncated four outputs;
it prompted this increase and is excluded from the final comparison. The raw dataset,
question/gold fixture, protocol and client hashes match. This is one bounded
quality run per backend, not statistical accuracy equivalence.

| Backend | Raw parsed correct | Completed and correct | Final-marker, untruncated correct | Truncated | Unparsed | Missing marker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton A1 | 96/100 | 95/100 | 95/100 | 1 | 0 | 1 |
| FI B1 | 95/100 | 95/100 | 95/100 | 0 | 0 | 0 |

Exact token sequences match on 63/100 questions; 37 differ. One parsed answer
and raw correctness outcome changes: **question 209**, gold 145. Triton hits
1,750 tokens without a final marker; its final-number fallback happens to
extract 145 and is credited by the raw parser. FI returns the completed but
incorrect answer 34800 with a final marker. First divergence is token index
480, zero-based. Raw flips are one loss and zero gains; completed-answer and
final-marker/untruncated correctness flips are zero. The equal conservative
95/100 counts do not establish quality equivalence. Common wrong IDs are
93, 255, 1161 and 1309. Full outputs and the sole truncation are preserved.

The preliminary 1,024-token baseline is excluded. No question, prompt or budget
was changed after inspecting these final A/B outcomes. See
`final-quality-results.md` for hashes and changed-answer details.

## Correctness, runtime and limits

- Final production-wrapper matrix: 10 FP32-state cases, 128 graph steps each,
  unchanged pointwise atol=rtol=1e-2 and relative L2 <1%; zero violations.
  Inactive/null0 state slots and page padding remain bitwise unchanged.
- Focused production-wrapper, real layer/loader and mixed-path tests: 31 passed.
  Full metadata suite: 63 passed; config selector test: 1 passed. These are
  separate commands, with their raw final logs retained.
- BF16-state FI opt-in is rejected. Its separate official-library padding
  diagnostic writes reserved slot 0 and does not meet vLLM's state invariant.
- Existing MTP diagnostics before the final bias fix and on unchanged baseline
  reproduce the same two dispatch failures with seven passing and seven skipped
  MTP cases. The reused CUDA12.9 binary lacks the fused operator under current
  CUDA13 build gating. These diagnostics are distinctly labeled; no new passing
  MTP kernel coverage or full current-main native build is claimed.
- Torch 2.13.0+cu129, Triton 3.7.1, official FI 0.7.0.post1 (RECORD verified),
  CUTLASS DSL 4.7.1. Main pins a newer DSL; the pinned environment and H20/SM90
  remain unmeasured. FI files were not patched and no private kernel was added.
- Final micro measurements use CUDA graph-event fallback because CUPTI is
  unavailable. Their grouped cold-cache samples and per-step metadata cost are
  disclosed separately in `final-micro-results.md`.
  Mapping is measured per metadata-builder/cache-group remap, reused by that
  group's layers; multiple groups can require multiple remaps. Serving timings
  already include the actual model's metadata work.

All final launches use GDN source SHA256
`674ea2aef2cf945e23b37bb1bc3a9e6a7c861df20561eebbc4131ee6b5faa963`
on main `3e182185aa5d143b0e69c44607f58a0cc55f3971` plus the eight-file patch.
Native extension SHA256:
`3efc749cc7d32a0177376cdabc19d18f406ca2f5a002f933a48997882536d6d8`.
The actual BF16 checkpoint biases are represented exactly in FI-only FP32
parameters initialized once; there is no per-forward bias cast.

## Raw evidence

`final/serving-summary.json`, SHA256
`27f19177fc80d3b3524570520456d5a6087f5c4180959823ce7e6b801dad4c2c`,
was generated once with frozen parser SHA256
`321f40d6f1c3a4a39dd5b5d5e0c4e03c9c39ea387ed5ae88ca872b9dcd796f00`.
It retains all round values, launch ranges, pair effects, quality outcomes,
four source/runtime snapshots and audited log hashes. Canonical raw results
are `final/http/{arm}.json`, `final/serve-final-{arm}.json`,
`final/logs/serve-final-{arm}.log` and `final/quality-{backend}.json`.

Dataset SHA256:
`ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13`.
Question fixture SHA256:
`281fc18d4cc24deb70f039966292226a93945424078523f5e4706c4d714b0f7e`.
The explicit evidence package manifest records every selected file's hash.
AI assistance was used. The human submitter's line-by-line review and test
execution remain required before submission.
