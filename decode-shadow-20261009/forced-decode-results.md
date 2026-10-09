# Native teacher-forced incremental decode

The final V3 pair on unchanged `d8ae9e9` uses two L20 GPUs, eager synchronous TP2,
BF16 inputs, FP32 recurrent state, Marlin and FlashAttention2. The engine core runs
in process so all eight requests are enqueued before execution. Worker processes
still execute real NCCL TP2. Native V2 `TraceReplayState` commits the supplied
reference tokens; an external observer records the natural sampler before forcing.
Production files are not modified.

Each backend returns all eight exact reference prefixes and records 3,187 unique
question/step positions: eight prefill first-token observations and 3,179 actual
incremental decode positions. The audit verifies each duplicated observation on
both TP ranks. Shorter references finish at their own lengths; the longest two
are limited to 512 tokens. Actual batches start at eight and shrink thereafter.

The recorded request/batch schedules match. All eight recorded prefill top-five
logits, logsumexp and forced-token logits match. These are observed prefill values,
not a comparison of every vocabulary logit, hidden state, pool page or GPU padding
shape. Full source/producer-helper hashes, original prompts, committed token
histories, model input token/position and actual returned token IDs are checked.

Natural raw argmax and actual sampler choices differ between backends at exactly
three observed decode positions. All other observed natural choices match; each
natural sampler ID equals its backend's raw argmax. Steps below are zero-based.

| Question | Observed step | Natural token ID T / FI | Top-two raw-logit gap T / FI |
|---|---:|---|---|
| 206 | 4 | 5821 / 1324 | 0.125 / 0.0 |
| 209 | 454 | 279 / 328 | 0.125 / 0.125 |
| 255 | 229 | 198 / 271 | 0.0 / 0.125 |

Question 209 has a positive top-two gap in both arms, so these differences cannot
all be described as tie-breaking. The small margins are consistent with numerical
sensitivity under a shared forced history. They do not establish the root cause
of prior free-generation answer changes or long-output truncation, nor prove that
the differences are harmless. This is one paired observation, not a repeatability,
accuracy or serving-performance measurement. Forced token equality is expected
by construction and must never be reported as accuracy equivalence.

The earlier V2 pair is retained: both arms pass integrity and have 11 natural
choice differences, but initial prefill batch sizes are 1 versus 2, schedules
differ, and all eight recorded prefill values differ. It does not isolate the
decode adapter. The final V3 result improves the comparison; it does not erase V2.

The complete traces, frozen helpers and exact returned tokens are in the archive.
`run-records/force-audit-v3.json` and `run-records/force-audit-v2.json` provide
per-question counts, candidate probabilities and the separately scoped checks.
AI assistance was used.
