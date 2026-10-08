# Matched new-head model quality screen

Source: `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`. All integrity checks passed.

Same frozen 100-question GSM8K subset, zero-shot, temperature 0, seed 42,
thinking disabled, concurrency 8 and 1,750 output-token budget. One
successful launch and one quality run per backend on L20/SM89, TP1,
BF16 inputs, FP32 state, Marlin, FA2 and 21,845 actual KV tokens.

| Backend | Raw parsed correct | Completed correct | Strict correct | Truncated | Unparsed | Missing marker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| triton | 96/100 | 96/100 | 96/100 | 0 | 0 | 0 |
| flashinfer | 96/100 | 96/100 | 96/100 | 0 | 0 | 0 |

Strict correct requires a correct parsed answer, `finish_reason=stop`,
no truncation and a `####` final-answer marker. Raw parsing may count
a coincidentally correct trailing number in an unfinished response.

Exact token sequences match 62/100; parsed final answers match 100/100.

- raw_correct: Triton-to-FI losses []; gains [].
- completed_correct: Triton-to-FI losses []; gains [].
- strict_correct: Triton-to-FI losses []; gains [].

## Every changed answer or exceptional outcome

| Question | Gold | Triton parsed / finish / strict | FI parsed / finish / strict |
| ---: | --- | --- | --- |
| 93 | 36 | 36.36 / stop / False | 36.36 / stop / False |
| 255 | 192 | 96 / stop / False | 96 / stop / False |
| 1161 | 170 | 140 / stop / False | 140 / stop / False |
| 1309 | 2280 | 2180 / stop / False | 2180 / stop / False |

The audit JSON retains full text for every row above, all flips,
all changed token sequences, and their first token divergence.

## Scope and evidence

This bounded single-run subset screens for visible quality regression.
It does not establish statistical accuracy equivalence, performance gain,
or reproducibility across launches. Equal counts do not erase output changes.
Different answers alone do not attribute causality to this PR; repeat and
C1 diagnostics are needed when outcomes warrant investigation.

Source/head, the five source-file hashes, fixture, gold parser, dataset hash,
client/supervisor hashes, protocol, token hashes, usage, launch settings,
raw logs and 100 HTTP 200 completions per backend were checked.
The model-config hash and runtime snapshot are matched between arms.
The runtime probe was reused from the earlier review run rather than
rerun inside these model processes. Native binary reuse and untested
SM80/H20/TP/full CUDA13 build limitations remain.

The full local dataset hash and all selected question/gold rows were checked.

Raw files are unchanged. `audit.json` and this report are derived evidence.
