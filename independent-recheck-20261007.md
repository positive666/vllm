# Independent recheck before submission

The supported opt-in integration is coherent. Independent source and raw-data
audits found no new material code defect or incorrectly calculated benefit.
This is a backend integration, with no established meaningful L20 serving gain.

- The eight source/test hashes and exact review.patch bytes match the tested
  revision, main base `3e182185aa5d143b0e69c44607f58a0cc55f3971`.
- Native null slot 0 and negative padding map to FI -1 while other kernels keep
  their original indices. Preallocated buffers belong to the metadata builder
  and cache group, preserving graph pointers. FP32 indexed-pool trajectories
  preserve inactive/null slots and page padding exactly; BF16 SSM is rejected.
- BF16 checkpoint bias promotion is exact. FP32 bias stabilizes this adapter's
  signature, without fixing FI's upstream shared-cache dtype collision.
  FP32-bias checkpoints may change parameter precision outside the tested model.
- Independent micro latency reductions from raw JSON: warm B1/B8/B32
  24.55268/13.52045/2.09770%; cold B1/B8/B32
  14.16039/-0.91254/1.29773%. These are latency reductions, not throughput gains.
  The per-layer boundary excludes projection/convolution and measures remapping
  separately per builder/group; full serving includes actual remapping.
- Serving records confirm four distinct ABBA launches, matching source/runtime,
  raw log hashes and 21,845 actual KV tokens. All 768 measured and 64 warmup
  requests completed. Independent paired-median throughput changes are
  +0.279458% C1 and +0.113280% C8; C8 pair two is -0.207993%. Two launches per
  backend establish no significance or reason to change auto from Triton.
- Independently parsing output text gives GSM8K raw T/FI 96/95, completed and
  marked correct 95/95, token-exact 63/100, and only question 209's final number
  differing. The baseline truncates; FI's completed answer is wrong. Accuracy
  equivalence is not established. The reports now disclose the preliminary
  1,024-token Triton run and fix 1,750 before both final runs.
- Final targeted pass count is 31 adapter/mixed +63 metadata +1 config =95.
  Native MTP's pre-final diagnostic and exact baseline both have 7 passed,
  7 skipped and 2 failed; this is no new passing native MTP coverage.
- Current main `5281e49908960f490c8faa789eee221d4ab5001d` is 61 commits ahead
  of the tested base. GitHub's compare reports no change to the three production
  files or affected GDN kernel/metadata suites; unrelated config tests changed.
  Neither this comparison nor L20 results establish a current-main native
  build, CuTeDSL 4.8, H20, ROCm or TP>1 runtime result.

Only evidence wording changed in this recheck; source and tested harnesses
remain frozen. The original final ZIP is retained unchanged; publish the revised
evidence as a separately named archive. The human submitter's review/testing
confirmation applies to this unchanged eight-file patch.
