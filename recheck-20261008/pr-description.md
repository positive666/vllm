## Overview

Add opt-in FlashInfer pooled-state GDN ordinary decode for Qwen hybrid models,
including mixed prefill/decode batches. `auto` stays Triton; prefill and MTP
selection remain unchanged. Explicit FI requires CUDA SM89+, BF16 inputs,
FP32 SSM state, K=V=128 and value-head counts divisible by eight per TP rank.
Packed QKV uses views and supplied output; null indices use reusable graph
buffers per metadata builder/cache group; bias promotion happens at construction.

This follows the [reviewer's invitation](https://github.com/vllm-project/vllm/pull/53463#issuecomment-6006969716)
to succeed the still-open Draft #41966. It uses the current metadata/CustomOp
interfaces and adds mixed-batch, indexed-state and graph validation. #53645
instead integrates the SM120 whole-layer kernel. Duplicate checks and the
takeover explanation are linked in the [original evidence](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007).

```bash
vllm serve MODEL --mamba-ssm-cache-dtype float32 \
  --kernel-config '{"gdn_decode_backend":"flashinfer"}'
```

## Validation

Rebased without semantic changes onto main `8352b242` (2026-10-08), preserving
the new ROCm AITER prefill path. Tested/pushed head: `c3424cc55d5258a3d5f104d817d6c6ef850db67f`.
Torch 2.13.0+cu129, FI 0.7.0.post1 and actual CuTeDSL **4.8.0**, with a fresh
FI JIT workspace and checked wheel RECORD hashes.

- **95 core tests passed:** adapter/real bias loading 19, mixed batches 12,
  metadata 63, configuration 1. Local pre-commit and manual mypy-3.12 passed.
- **38 supplemental native tests passed:** 16 MTP model-path and 22 direct
  head-ratio cases, no skips. Pure-MTP tests assert a real native call.
  The unchanged official CUDA source was built separately for SM89 with
  NVCC12.9; this does **not** cover the full default CUDA13 build. The original
  reused binary's missing-op diagnostic remains 7 passed/7 skipped/2 failed.
- Ten FP32 configurations passed 128-step graph trajectories with unchanged
  tolerances and bitwise-preserved inactive/null state and page padding.
- One launch per backend: **96 smoke requests +32 warmups succeeded**,
  C1/C8, 512 input/64 output tokens, matched 21,845 KV tokens. This short smoke
  is not a new throughput or accuracy evaluation.
  Paired measured outputs match 24/24 at C1 and 11/24 at C8 (47.53% token-position
  agreement); C8 also varies within each backend, so no accuracy equivalence
  or attribution of the difference to this PR is established.

Latest L20 production GDNDecode + identical gated RMSNorm, H=16/HV=48/K=V=128;
median of three alternating round medians, microseconds:

| Batch | State cache | Triton | FI | FI latency reduction |
|---:|---|---:|---:|---:|
| 1 | warm | 6.354 | 4.766 | 24.99% |
| 8 | warm | 19.920 | 17.132 | 14.00% |
| 32 | warm | 200.918 | 193.073 | 3.90% |
| 1 | rotating/cold | 12.142 | 10.433 | 14.08% |
| 8 | rotating/cold | 74.690 | 75.315 | -0.84% |
| 32 | rotating/cold | 311.909 | 307.629 | 1.37% |

CUDA graph events; projection/convolution/allocation/compilation are excluded.
Remapping is measured separately at 2.040-2.092 us **per builder/group update**;
serving includes actual mappings. These are kernel measurements, not whole-model
throughput gains.

The [earlier ABBA serving and GSM8K evidence](https://github.com/positive666/vllm/blob/codex/gdn-flashinfer-evidence-20261007/README.md)
uses base `3e182185` / head `b2b93acd`, DSL4.7.1: 768 measured/64 warmup requests,
paired throughput +0.279% C1 / +0.113% C8, with -0.208% in the second C8 pair.
The seed-42, zero-shot 100-question screen (1750-token final budget) scores raw
96/95 and completed/marked 95/95; tokens match 63/100. Triton's #209 truncation
gets fallback credit, while FI completes incorrectly. This establishes neither
significant serving improvement nor accuracy equivalence. Those scores were
not rerun or transferred to the new environment.

[New commands, raw logs, hashes and audit](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/recheck-20261008).
Existing pytest suites reproduce the core checks; the supplemental build and
launcher are included separately in the archive.

## Limits

The model folder is labelled Qwen3.8-27B-FP8; its architecture is
`Qwen3_5ForConditionalGeneration`, run text-only with Marlin and FP32 state.
All 48 BF16 checkpoint biases promote exactly; FP32-bias checkpoints remain
outside the model evaluation. FI BF16-state padding can update reserved slot 0,
so that state dtype is rejected. FP32 bias stabilizes this adapter's signature,
but does not fix FI's bias-dtype cache key or validate externally populated caches.
Full CUDA13 native builds, H20, ROCm and multi-GPU TP remain untested.

Upstream pre-commit is skipped by the contributor gate (three merged PRs,
four required); ReadTheDocs is cancelled before building. Local results are
documented separately. AI assistance was used. The human submitter confirmed
review and testing of the patch; its eight-file added/deleted lines are unchanged.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill.
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Reuses current CustomOp/metadata interfaces; automatic selection is unchanged.
- [x] **Testing and Validation:** Tests, model evaluation and CI constraints are documented with raw evidence.
- [x] **Code Quality and Style:** Local pre-commit and manual type checks passed.
- [x] **Pull Request Contents:** Includes motivation, duplicate-work explanation, evidence and limitations.

</details>
