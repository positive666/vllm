# TP2 model quality and distributed correctness

Source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`, two L20 GPUs, BF16 inputs and FP32 state.
Expanded-budget C8 diagnostic: FI **6/8** strict correct versus Triton **7/8**, with **1 FI truncation(s)**. FI question 209: parsed 34800, gold 145, stop, 1355 tokens; FI question 255: parsed 20, gold 192, length, 3500 tokens. The retained FI C8 long-generation/output differences are an unresolved limitation in this bounded diagnostic, despite the larger budget. The default `auto` path remains Triton; FI remains opt-in. Integrity audit success does not establish accuracy equivalence or suitability as a general replacement.

One 100-question GSM8K run per backend, the same deterministic test subset,
zero-shot, temperature 0, seed 42, thinking disabled, concurrency 8 and a
1,750-token output budget. Both source/fixture/protocol/raw-integrity audits pass.

| Backend | Raw correct | Completed correct | Strict correct | Truncated | Unparsed | No #### |
|---|---|---|---|---|---|---|
| triton | 95/100 | 95/100 | 95/100 | 0 | 0 | 0 |
| flashinfer | 96/100 | 96/100 | 96/100 | 1 | 0 | 1 |

Strict correct requires a correct parsed answer, `finish_reason=stop`, no
truncation and a `####` answer marker. Equal counts do not erase changed answers.
Parsed answers match 98/100; exact token
sequences match 58/100.

- raw_correct: Triton-to-FI losses []; gains [209].
- completed_correct: Triton-to-FI losses []; gains [209].
- strict_correct: Triton-to-FI losses []; gains [209].

| Question | Gold | Triton parsed/finish/strict | FI parsed/finish/strict |
|---|---|---|---|
| 93 | 36 | 36.36 / stop / False | 36.36 / stop / False |
| 209 | 145 | 34800 / stop / False | 145 / stop / True |
| 255 | 192 | 176 / stop / False | 20 / length / False |
| 1161 | 170 | 140 / stop / False | 140 / stop / False |
| 1309 | 2280 | 2180 / stop / False | 2180 / stop / False |

`audit-quality-correctness.json` retains all answer flips, exceptional output
text and the first token divergence for changed sequences.

Log integrity retains a separately audited late suffix; original launch records and logs were not rewritten.

- serve-triton-compat.log: recorded log SHA verifies the first 131171 bytes; 4352 appended bytes separately match the closeout manifest. Tail contains allowed late buffered NCCL informational output and resource_tracker shutdown cleanup; it is not described as only teardown. Original record/raw bytes were not changed.

The prior same-head [TP1 screen](../quality-followup-20261008) reported 96/100 and 96/100 strict correct. TP1 and TP2 have different sharding and communication; a cross-TP score difference cannot be attributed to FI from these single runs.

This bounded single run does not prove accuracy equivalence or accuracy
improvement. Do not infer causality from one changed output or a score difference.

Two NCCL ranks pass all 12 rank/cases
at local H8/HV24/K128/V128. Each rank checks B1/B8, int32/int64 indices and B8
padding against the one-step reference and 128 changing-input graph replays,
using pointwise atol/rtol 1e-2 and relative L2 <1%, with exact inactive state
and page-padding checks. NCCL initialization/reduction is checked separately.
This validates local wrapper execution on both ranks; checkpoint sharding and
full-model TP communication are exercised by the separate serving launches.
Correctness records are not TP2 microbenchmark timings.

## Separate expanded-budget diagnostic

Expanded-budget C8 diagnostic: FI **6/8** strict correct versus Triton **7/8**, with **1 FI truncation(s)**. FI question 209: parsed 34800, gold 145, stop, 1355 tokens; FI question 255: parsed 20, gold 192, length, 3500 tokens. The retained FI C8 long-generation/output differences are an unresolved limitation in this bounded diagnostic, despite the larger budget. The default `auto` path remains Triton; FI remains opt-in. Integrity audit success does not establish accuracy equivalence or suitability as a general replacement.

The original 100-question records above remain unchanged, including truncations
and changed answers. A matched targeted diagnostic replays the original C8 batch
`[198, 206, 209, 228, 255, 285, 292, 318]`: questions 209/255 plus six controls,
once at C1 then once at C8 per backend, 16 requests per backend.
Its output budget is 3,500, context length 4,096 and GPU blocks 128, with
matched actual capacity >=32,768 tokens. Source/model/runtime/GPUs are unchanged.

| Backend | C | Requests | Raw correct | Strict correct | Controls strict | Truncated | Unparsed |
|---|---|---|---|---|---|---|---|
| triton | 1 | 8 | 7 | 7 | 6/6 | 0 | 0 |
| triton | 8 | 8 | 7 | 7 | 6/6 | 0 | 0 |
| flashinfer | 1 | 8 | 7 | 7 | 6/6 | 0 | 0 |
| flashinfer | 8 | 8 | 6 | 6 | 6/6 | 1 | 0 |

| Backend | C | Question | Gold | Parsed answer | Finish | Strict | Tokens |
|---|---|---|---|---|---|---|---|
| triton | 1 | 209 | 145 | 145 | stop | True | 2350 |
| triton | 1 | 255 | 192 | 176 | stop | False | 1482 |
| triton | 8 | 209 | 145 | 145 | stop | True | 1804 |
| triton | 8 | 255 | 192 | 176 | stop | False | 3315 |
| flashinfer | 1 | 209 | 145 | 145 | stop | True | 1597 |
| flashinfer | 1 | 255 | 192 | 176 | stop | False | 1835 |
| flashinfer | 8 | 209 | 145 | 34800 | stop | False | 1355 |
| flashinfer | 8 | 255 | 192 | 20 | length | False | 3500 |

`audit-diagnostics.json` independently retains the paired outcomes and flips.
This diagnostic changes output budget, model context and cache capacity together;
it cannot isolate a single cause, change the original 100-question score or prove
overall accuracy. Wrong and truncated outcomes are retained rather than discarded.


[Additional top-five logprob observation](additional-logprob-results.md) is separately scoped; it does not replace or resolve these retained quality outcomes.

