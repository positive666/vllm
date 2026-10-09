# Independent evidence and wording review

Reviewed production head: `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`. This review changes no production code and runs no GPU work.

## Local replay claims are supported

The full replay report SHA256 matches `b9845fb320a4847d52dad8fcc80b07c75e075eaaefb94bdfecf3432e9f36e633`. Its frozen helper SHA matches the available `replay_capture.py`: `f8bc6de59f821b9846acd2b63ba221cb1e247ce8e97dec2a5f1e920e57db9e4a`.

- There are 248 sampled actual-decode captures: 124 per TP rank, covering 48 GDN layers; actual batch sizes are 8/7/3/2.
- All 248 FI replays reproduce the captured active outputs and all captured poststate pages exactly. This is FI replay fidelity, not bitwise equality between Triton and FI. T/F output is exact in 58/248 cases; active state is exact in 0/248 cases.
- Both backends pass the frozen rounded FP64-reference checks and the retained unrounded mathematical comparisons. Pointwise tolerance is `atol=rtol=0.01`; aggregate relative-L2 must be strictly below `0.01`.
- Maximum T/F aggregate relative-L2 is `6.37911411434036e-5` for active output and `3.8897870142536134e-7` for active state. These are ratios, corresponding to approximately 0.00638% and 0.0000389%. Maxima are independently taken across cases; they need not identify one common worst case.
- The unrounded FP64 output comparison has maximum absolute error about `0.01497` and maximum relative-L2 about `0.0019714` (0.1971%). Do not substitute the much smaller T/F error for mathematical-reference error. The output is BF16, and the rounded reference comparison is separately retained.
- All 48 checkpoint `dt_bias` tensors are BF16. Their promotion to FP32 preserves the stored values exactly. This rules out recovering additional bias precision for this checkpoint; it does not establish identical arithmetic for every possible bias dtype or checkpoint.

Each backend starts each capture from the same captured FI prestate. This checks a local update; it does not propagate an independent Triton state history. Original per-tensor strides and pointer alignment modulo 256 are reconstructed. Absolute addresses, original aliases, all uncaptured pool pages and original storage-gap contents are outside scope. Captured null and one sampled unused page remain unchanged, and synthetic storage guards pass. There were no actual padding rows, so the new captures add no padding-row coverage.

The two representative raw snapshots intended for publication do not independently validate the other 246 captures. Publish the full hash manifest and replay reports, disclose that the complete 7,304,360,944-byte raw corpus remains on the experiment host, and preserve both failed harness attempts and corrected runs.

## Forced-v2 results are bounded and have a prefill confound

Independent aggregation of the locally downloaded raw traces finds 3,187 unique `(question, step)` observations per backend on each rank, including eight prefill-first-token observations. All eight returned token streams equal the supplied reference prefixes; maximum per-question coverage is 512 tokens. The frozen helper hashes recorded by both launches match their copied run-local helpers.

The configured V2 runner hash independently matches `git show d8ae9e9:vllm/v1/worker/gpu/model_runner.py`: `48ae64b57dae7e26599df3e2a967a5f3ac0ad91861f7b879da989da0f5c67cf3`. Both ranks recorded unsharded duplicate sampling, not separate disjoint sampler shards. The primary audit must verify both ranks rather than treating one as complete evidence for the other.

Actual natural sampler choices equal the raw argmax throughout these downloaded traces; no nonmax chosen raw logit was observed. T/F natural choices differ at 11 positions across five questions (per-question counts 1+1+3+5+1). First differing positions (zero-based) are 198:50, 206:4, 209:411, 255:81, 292:253. Questions 228/285/318 have no differing natural choices in the observed prefixes. These are observer results under deliberately forced histories, not accuracy scores or original free-generation divergence positions.

**The v2 pair does not isolate decode-backend causality.** Triton initially prefills question 198 alone; FI initially prefills questions 198/206 together. Subsequent model batch schedules also differ. All eight prefill-first-token records differ in observed top-five/logsumexp values. For question 198, the first raw top logit is 28.5 (T) versus 28.625 (FI) before incremental decode. Shared token histories alone do not mean equal initial hidden or recurrent states. Retain this pair as a valid bounded observation if its integrity audit passes, but do not attribute its later logits/choice differences solely to the GDN decode adapter. A matched prefill/scheduling rerun would improve causal interpretation.

Forced output cannot measure model accuracy, serving latency or throughput. The 512-token bound does not validate the prior 3,500-token truncation. The primary auditor's full provenance, cross-rank consistency and history checks are still required before publication; this independent read is not a substitute for that audit.

## Publication requirements

Require both final run exit markers to be zero, both eight-example `completed.json` files to exist and match their fixtures, and the final audit `integrity_pass` to be true before exporting. Distinguish integrity pass from matching batch schedules or equal prefill observations. Bind all copied files to an export manifest and archive SHA; do not silently accept absent forced-run files through optional globs.

No high-confidence GitHub-token, authorization-token or private-key pattern was found in the locally available text artifacts inspected here. The SSH session helper and credentials must remain outside the export. This scan covers files available at review time, not an archive that has not yet been built; the final archive still needs inspection.

Retain the prior expanded-budget C8 negative: FI 6/8 strict versus Triton 7/8, with one FI truncation. The newer eager capture's 7/8 and forced-token streams do not replace this result. Earlier TP2 throughput changes (~0.38%/0.42%) are descriptive, and C8 TPOT was 1.59% worse with mixed launch signs. The current validation adds no performance result.

No observed local replay contract failure justifies a production-code change. Keep FI opt-in and `auto` on Triton. Do not claim general accuracy equivalence, a proven root cause, significant serving speedup or merge readiness.

## Suggested concise addition to the existing PR performance comment

> Added real-decode validation on the same d8 source: 248 sampled TP2 eager captures from both L20 ranks. FI exactly reproduces captured active outputs/poststate pages; both backends pass the frozen FP64-reference checks (maximum T/F relative-L2: output `6.38e-5`, state `3.89e-7`). Each replay resets to the captured prestate; this is not sequence-level accuracy equivalence. An additional native teacher-forced probe covers up to 512 tokens per question, but this pair has different prefill batching/logits, so its choice differences do not isolate the decode adapter. Earlier C8 answer/truncation differences remain unresolved; FI stays opt-in. No production-code change is supported by these observations. AI assistance was used. [Evidence](IMMUTABLE_EVIDENCE_URL)

Only publish the teacher-forced clause after the final integrity audit passes; replace it with the result of any subsequently completed matched rerun rather than silently dropping this pair's limitation.

## Suggested evidence README structure

1. Source, runtime, model configuration and exact experiment boundary; unchanged production head and AI assistance.
2. Capture selection/counts and local replay results; clear FI fidelity versus T/F tolerance distinction.
3. Frozen mathematical conventions and BF16-checkpoint bias result.
4. Native teacher-forced observations, actual prefill/batch comparison and audit status; no output-quality or timing score.
5. Retained failures, original C8 negative and unvalidated long-generation/hardware workloads.
6. Reproduction commands, immutable file/archive hashes, exported representatives and remote full-corpus scope.

## Final matched-prefill V3 review

The V3 pair supersedes the V2 pair for the reported matched-history comparison; V2 remains retained with its prefill confound. Independently reading both V3 raw traces confirms 3,187 unique `(question, step)` observations per backend and per rank: eight prefill-first-token observations and 3,179 real incremental-decode observations. Both ranks' duplicate observations agree on token histories and recorded numerical values. Returned token streams exactly match the supplied reference prefixes, with each question ending at its own reference length or the 512-token limit.

All recorded request batches, scheduled token counts and input positions match between arms. Initial actual batch size is eight. All eight prefill observations have equal recorded top-five IDs/values, logsumexp and forced-token logits. This establishes equality of those observations; it does not establish bitwise equality of full-vocabulary logits, hidden states, recurrent states or GPU padding shapes.

Exactly three natural sampler choices differ between T/F in this bounded pair. Raw argmax and the actual sampler agree throughout both traces, and the same three positions differ under either definition:

| Question | Zero-based position | Actual batch size | Top-two raw-logit gap T/F |
|---|---:|---:|---:|
| 206 | 4 | 8 | 0.125 / 0 |
| 209 | 454 | 3 | 0.125 / 0.125 |
| 255 | 229 | 7 | 0 / 0.125 |

The other five questions have no natural-choice flips in their observed prefixes. Do not call all three events exact ties: question 209 has a positive top-two gap on both arms. These observations support low-margin numerical sensitivity under matched token histories and recorded batching, but they do not prove an adapter defect, harmlessness, or the root cause of the earlier wrong answers/truncation. No separate repeated pair has established repeatability. Both backends follow deliberately supplied histories after every observation, so the forced completions are not model accuracy measurements.

The primary V3 audit reports `integrity_pass=true`. This independent review verifies the SHA/size of all 41 Triton and 42 FI files bound by its `input_files`, with no mismatch. The 4 live executed helper hashes match run-local frozen snapshots. The frozen driver SHA is `3eca92db168e57e407fb29f5dcdd00f6f6e03b6da7c4bb57a2c50a50881bc30b`; `force-audit-v3.json` SHA is `5c2d2b28720157b43a62145ca850c7434e7823ddfd60c91b0f0c5a72a24c74ea`. A copied audit helper exists only in one arm's all-helper manifest because it was added between launches; it was not executed by either model arm and the audit retains this distinction.

The 248-capture local replay conclusion remains unchanged. No new production-code contract failure is demonstrated. The prior C8 answer/truncation differences, long-generation coverage and general performance claims remain unresolved. The final exported archive still requires byte/hash and secrets inspection before publication.

### Replacement concise PR comment addition after final archive inspection

> Added same-head TP2 validation on two L20s: 248 real-decode snapshots reproduce FI's captured output/state exactly, and both backends pass the frozen FP64-reference checks. A native teacher-forced comparison covers 3,179 decode positions with matched recorded request batches and identical observed prefill values. Three natural token choices differ at low margins; this bounded probe does not establish accuracy equivalence or resolve the retained C8 answer/truncation differences. FI remains opt-in, and these results provide no evidence-based production-code fix or new performance claim. AI assistance was used. [Evidence](IMMUTABLE_EVIDENCE_URL)

The evidence README should show the three raw-logit gaps above, the 512-token bound, V2's retained confound, and the exact scope of observed prefill equality. The brief public comment can link those details rather than reproduce every metric.
