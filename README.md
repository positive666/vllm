# FlashInfer GDN takeover: L20 evidence

This branch contains supporting evidence for the reviewer-invited successor to
vllm-project/vllm#41966, following the invitation in
[vllm-project/vllm#53463](https://github.com/vllm-project/vllm/pull/53463#issuecomment-6006969716).
AI assistance was used; the human submitter confirmed source review and testing.

[Latest main/CuTeDSL 4.8 recheck (2026-10-08)](recheck-20261008/README.md):
95 core tests and 38 supplemental native tests passed; 96 serving smoke
requests and 32 warmups succeeded. This adds current Python-source/DSL coverage
and a separately built CUDA12.9 MTP component, not a full CUDA13 native build.
The original results and archive below remain unchanged with their original
source/runtime provenance.

[Independent recheck](independent-recheck-20261007.md),
[microbenchmark](final-micro-results.md), [serving](final-serving-results.md),
[quality](final-quality-results.md), [reproduction commands](review-commands.md).

Download [the complete evidence ZIP](gdn-l20-evidence-rechecked.zip).
SHA256: `6b60d68c1d134176be2d5284308d5b0832b2023b522d25df0f1566c08b59085a`.
Its evidence-manifest.json verifies every included file, including raw results,
final test logs, the exact eight-file patch and the tested source hashes.

The supported feature is opt-in FP32-state FlashInfer decode. L20 kernel latency
improves in several measured shapes, but full-model throughput is essentially
unchanged (+0.279% C1 / +0.113% C8; two launches per backend). Auto stays Triton.
GSM8K raw scores are Triton/FI 96/95, completed and marked correct 95/95;
this 100-question screen does not establish accuracy equivalence.

The revised archive corrects wording about the preliminary 1024-token quality
run and documents the bias/cache limitation. Production source, measured
harnesses and raw results are unchanged. Validation uses base
3e182185aa5d143b0e69c44607f58a0cc55f3971 Python sources with the disclosed reused
CUDA12.9 native binary and CuTeDSL 4.7.1. That archive does not claim current-main
or CuTeDSL 4.8 coverage; see the separately scoped latest recheck above. Full
CUDA13 native, H20, ROCm and TP>1 coverage remain unclaimed; native MTP baseline
failures are included separately and are not counted as passing coverage.
