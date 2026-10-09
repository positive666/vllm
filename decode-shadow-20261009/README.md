# Additional decode validation for PR #60403

Reviewed production source remains `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`.
No production-code change was made. No local replay contract failure was found,
but final model-quality equivalence remains unestablished. FI remains opt-in and
`auto` remains Triton.

- [Real decode capture/replay](capture-results.md): 248 actual TP2 snapshots,
  124 per rank, all 48 GDN layers. FI replay exactly reproduces captured output
  and poststate; both backends pass frozen FP64-reference checks. T/F output and
  state are not bitwise equivalent. Each case resets to captured FI prestate.
- [Matched-history decode observation](forced-decode-results.md): actual batch
  schedules and recorded prefill values match in the final pair; 3,179 decode
  positions include three differing natural sampler choices at small margins.
  Forced outputs are not quality results.
- [Independent review](independent-review.md) and
  [auditor tamper rejection](audit-guard-verification-v2.json): five invalid-input
  cases rejected; real run input bytes remain unchanged.
- All 48 checkpoint bias tensors are BF16; FP32 promotion preserves their stored
  values exactly. This checkpoint does not test a native FP32-bias checkpoint.

Retained earlier expanded C8 quality result: Triton 7/8 versus FI 6/8 strict,
including one FI 3,500-token truncation. Those output/truncation differences remain
unresolved. The new eager/instrumented 512-token observation does not close that
issue or validate CUDA-graph serving quality. It adds no performance measurement.
Prior TP1/TP2 timings and limitations remain in `../tp2-followup-20261008`.

The runtime is v0.29.0-cu129, Torch2.13/cu129, FI0.7.0.post1, CuTeDSL4.8 and
Triton3.7.1, with reused native binaries. The Qwen3.8-labelled FP8 checkpoint
declares `Qwen3_5ForConditionalGeneration` (text-only). H20, other model/TP shapes,
DP+EP, SM80 and a fresh full native build are outside this added validation.

`decode-shadow-evidence.zip` contains **315 files**,
**40,064,406 bytes**, SHA256 **`c44bbe5ea7d32a0265c1e0dbe09ea4a17ae15647da9f1d5dd4d59c2bb10e1519`**.
[Export manifest](export-manifest.json) binds every member except itself;
[receipt](export-receipt.json) binds that manifest and archive.

The complete **7,304,360,944-byte / 248-snapshot raw corpus stays on the experiment
host** and is bound by `capture-and-replay/snapshot-manifest.json`. The archive
exports only two representative raw `.pt` snapshots, from different ranks/layers/
calls. Those two alone cannot independently validate the remaining 246 snapshots.
All replay reports, capture traces, paired teacher-force traces, frozen producer
helpers, logs, exits and host telemetry are exported. Original aliasing, absolute
addresses, uncaptured pool pages and original padding-gap contents are unobserved;
synthetic guards and captured null/unused pages pass. No actual padding rows were
seen in the new capture run.

The archive preserves failed attempts: missing readonly dependency mount, the
same-process/disk FI bias-cache collision in diagnostic ablation, tokenizer plan
adaptation, and a V1 hook that did not apply to the actual V2 runner and was rejected
by its forced-token audit. These failures are not successful model validations.
An isolated BF16-bias diagnostic succeeds for two snapshots only. Current
production always uses FP32 bias; no FI dependency/cache source was changed.

AI assistance was used. The human submitter's earlier review/testing confirmation
applies to the unchanged production patch. [Reproduction commands](commands.md).
