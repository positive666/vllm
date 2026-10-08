# #60403 review-fix checklist audit — 2026-10-08

Audited `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6` against published parent
`c3424cc55d5258a3d5f104d817d6c6ef850db67f`: four files, +122/-104. No confirmed
code defect found. New human review/test confirmation remains pending.
Same-head serving was pending at the static audit and subsequently passed the
independent evidence audit. No source edits, GPU runs or publication by this auditor.

Read `.agents/skills/pr-checklist/SKILL.md`, the PR template, MRV2 async guidance,
and all paginated conversation/review/inline comments (including replies).

## 1. Design fit

- Changes stay inside the opt-in GDN adapter and existing tests; no runner,
  scheduler or native-kernel changes. Auto/Triton and prefill/spec dispatch remain
  unchanged; the explicit FI override is logged.
- The HV%8 constructor rejection is replaced by aligned a/b staging only when
  HV%8!=0. Forced contiguous clones fix the real BF16 packed-view 8-byte offset,
  including B1 where `contiguous()` can retain the original view. Usual HV%8=0
  layouts retain their views. The condition is a Python head-count constant;
  there is no hot `data_ptr()` branch or GPU-derived host decision.
- FI mixed-batch early return requires `split_non_spec`, which requires no spec
  masks. Conv updates, FI decode output/state, and prefill SSM writeback precede
  the tail copy; the return replaces only the final cat/output merge.
- No new explicit sync, GPU host read or CPU transfer is present. Clones add
  allocation/copy work on the smaller-head path; production runs inside the
  registered attention custom op and graph capture records that work.
  `VLLM_GPU_SYNC_CHECK=error` was not run: the sync assessment is static, not a
  runtime certification.
- SM80+ follows the ordinary cp.async/BF16 kernel and official DSL architecture
  support; actual hardware testing remains L20/SM89. K=V128 remains this
  adapter's support scope, rather than a claim about every FI configuration.

## 2. Testing and validation

- Downloaded summary records **104 passed, zero failures/errors/skips**:
  adapter/loading/indexed/graph 28, mixed-batch 12, metadata 63, config 1.
  The independent evidence audit supplies exact-source/hash verification.
- HV4 covers B1/B4, packed gate offsets, inference-mode Parameters, unused/null
  slots and page padding. Graph tests now mutate a/b and index maps after capture,
  catching stale staging. Mixed tests compare full output and both state caches.
- Fresh export probes retain the expected raw-FI failures: default exporter
  rejects grad-enabled Parameters; TVM-FFI exposes the misaligned a pointer.
  The adapter passes both probes with unchanged tolerances and Parameter flags.
  This supports retaining detach and adding aligned staging.
- CI includes these files through B200 `kernels/fla` and H200 MIG
  `kernels/mamba` jobs (`.buildkite/test_areas/kernels.yaml:214,513`); metadata is
  in H100 `v1/attention` and config in the engine job. AMD/CPU cannot validate
  the CUDA FI path. YAML inclusion is not evidence of an executed upstream run;
  FI availability/skips and the contributor gate must remain disclosed.
- Same-head serving retry passed 96 measured and 32 warmup requests after
  supplying the missing generated FlashMLA wrapper from main's official pinned
  build inputs. The independent evidence audit verifies raw requests and hashes;
  initial setup failures remain separately preserved.

## 3. Code quality and style

Local pre-commit/manual mypy-3.12 passed; the committed diff passes whitespace
checks. The pinned API stays lazy, selects `backend="flashinfer"` explicitly,
and removes signature/version compatibility probing. Errors now identify
individual unsupported settings; comments explain alignment and the default
DLPack exporter. Existing suites/helpers are reused, without new benchmarks in
tests. The small import helper remains the existing lazy-import test seam.
Commit includes human attribution, AI co-author and sign-off trailers.

## 4. PR contents and remaining items

Before publishing this follow-up:

1. Obtain human confirmation of **every changed line in this new four-file
   patch** and relevant tests; the earlier confirmation covered the parent patch.
2. Publish the prepared template's Overview/Claims/Validation/Details after
   human confirmation, retaining the alignment root cause, narrow-path staging
   cost, detach probe, 104-test/serving result, SM80 static-versus-L20 measured
   scope, AI assistance and CI limitations.
4. Keep older native supplemental, microbenchmark, ABBA and GSM8K evidence
   attached to their original heads/environments. Do not transfer those scores
   to this patch or describe successful smoke as accuracy/throughput evidence.

No new performance measurement establishes a serving speedup or an HV4 staging
cost. SM80 hardware, full CUDA13 source builds, H20 and multi-GPU TP remain
unvalidated here; these are explicit limits, not newly completed checks.
