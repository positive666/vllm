# Additional C8 top-five logprob observation

Reviewed source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6` remains unchanged. One C8 batch of eight questions per
backend collects top-five logprobs, max3500/context4096/128 GPU blocks. This is
separate from both the original 100-question screen and expanded C1/C8 replay.
Their scores, wrong answers and FI C8 truncation remain unchanged and unresolved.

| Backend | Requests | Raw correct | Strict correct | Truncated | No marker |
|---|---|---|---|---|---|
| triton | 8 | 6 | 6 | 0 | 0 |
| flashinfer | 8 | 6 | 6 | 0 | 0 |

Parsed answers match 7/8; exact generated token sequences match
4/8. These are additional observations, not replacement scores.

| Question | Matching prefix | First divergent position (0-based) | Positions compared | Min top2 Δlogp T/FI | At divergence T/FI | T answer | FI answer |
|---|---|---|---|---|---|---|---|
| 198 | 471 | None | 471 | 0/0 | none | 320 | 320 |
| 206 | 4 | 4 | 5 | 0/0.125 | 0/0.125 | 860 | 860 |
| 209 | 454 | 454 | 455 | 0/0 | 0/0.125 | 34800 | 34800 |
| 228 | 451 | None | 451 | 0.125/0.125 | none | 1 | 1 |
| 255 | 19 | 19 | 20 | 0.125/0 | 0.125/0 | 176 | 96 |
| 285 | 139 | None | 139 | 0.75/0.625 | none | 21 | 21 |
| 292 | 253 | 253 | 254 | 0.125/0 | 0.125/0 | 75 | 75 |
| 318 | 316 | None | 316 | 0.125/0.125 | none | 123 | 123 |

Returned top-two logprobs report zero margin on at least one arm in 4/4 first-divergence pairs. This is consistent with low-margin/tie sensitivity within these observations; it does not establish an epsilon-only difference or identify the underlying cause.

The actual incoming prompt tokens match. Comparison uses the shared generated
token prefix through and including the first divergent token, whose incoming
prefix still matches. Later positions have different conditioning histories and
are excluded. For a length-only difference, only overlapping positions are used.
Matching tokens do not establish identical internal hidden/recurrent states.

Top2 Δlogp is the difference between the highest two returned log probabilities,
in natural-log units; the full audit also retains probability margins and only
shared candidate-token differences. Top-five candidates cover part of the
vocabulary: these are not full-logit errors, KL divergence or full-vocabulary
probability distances. No margin threshold is used to declare a benign tie or
attribute any answer difference to this adapter.

Logprob collection can change scheduling and trajectories. This is one C8
observation per backend without forced decode or a full-state shadow reference.
It does not prove a root cause, accuracy equivalence, or resolution of the earlier
FI C8 limitation. Audit pass means observation integrity only.

[Original raw HTTP outputs and comparison in the archive](tp2-evidence.zip) are
under `logprob-diagnostics/`: `triton.json`, `flashinfer.json`, `comparison.json`
and both launch records/full logs. [Independent audit](audit-logprobs.json)
retains the final common-prefix window, chosen tokens, top-five candidates,
missing candidates and all unrounded reported margins.
