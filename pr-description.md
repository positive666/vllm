<!-- markdownlint-disable -->

## Overview

Add an opt-in FlashInfer pooled-state GDN decode backend for Qwen hybrid models, including ordinary decode in mixed prefill/decode batches.

This follows the [reviewer's invitation to open a fresh PR](https://github.com/vllm-project/vllm/pull/53463#issuecomment-6006969716), succeeding the still-open Draft [#41966](https://github.com/vllm-project/vllm/pull/41966) with current-main metadata and `GDNDecode(CustomOp)`. Its predecessor head remains unchanged. [#53645](https://github.com/vllm-project/vllm/pull/53645) instead integrates the SM120 whole-layer `gdn_fused_decode_step`; this change covers the post-convolution recurrent update. Fresh duplicate checks are recorded with the evidence.

## Claims

- `kernel_config.gdn_decode_backend={auto,triton,flashinfer}` selects ordinary decode. `auto` remains Triton; prefill and speculative/MTP selection are unchanged.
- Explicit FI requires CUDA SM89+, BF16 inputs, FP32 SSM state, 128-dimensional heads and value-head counts divisible by eight per TP rank. Unsupported configurations fail during initialization.
- Packed QKV uses views and caller-provided output; FP32 `dt_bias` is allocated at construction, with no per-forward bias cast.
- Null-slot remapping uses reusable graph buffers per metadata build/cache-group update, shared across that group's layers.

```bash
vllm serve MODEL --mamba-ssm-cache-dtype float32 \
  --kernel-config '{"gdn_decode_backend":"flashinfer"}'
```

## Validation

**95 targeted tests passed:** adapter/real bias loading 19, full metadata 63, mixed batches 12 and configuration 1. Ten FP32 cases passed 128-step graph trajectories with unchanged tolerances and bitwise-preserved inactive/null state and page padding. Pre-commit, manual mypy 3.12 and diff checks passed. Existing MTP diagnostics before the final bias fix and unchanged-main baseline both yield 7 passed/7 skipped/2 failed: the reused native binary lacks the fused operator. No new passing native MTP coverage is claimed.

L20, TP1, H=16/HV=48/K=V=128: production `GDNDecode` plus identical gated RMSNorm, median of three alternating round medians, microseconds:

| Batch | State cache | Triton | FI | FI latency reduction |
|---:|---|---:|---:|---:|
| 1 | warm | 6.438 | 4.858 | 24.55% |
| 8 | warm | 19.952 | 17.254 | 13.52% |
| 32 | warm | 236.298 | 231.341 | 2.10% |
| 1 | rotating/cold | 12.128 | 10.411 | 14.16% |
| 8 | rotating/cold | 74.517 | 75.197 | -0.91% |
| 32 | rotating/cold | 311.189 | 307.151 | 1.30% |

CUDA graph events replace unavailable CUPTI. Projection/convolution/allocation/compilation are excluded. Remapping costs 2.128–2.179 microseconds per builder/group update, measured separately; serving includes all actual mappings.

Four ABBA launches, two per backend: **768 measured/64 warmup requests succeeded**, with matched source/runtime/native, FP32 state, Marlin and 21,845 actual KV tokens. Paired median throughput changes are **+0.279% C1 / +0.113% C8**; C8's second pair regresses **0.208%**. These descriptive results establish no significant serving improvement or basis for changing `auto`.

The fixed seed-42, zero-shot 100-question GSM8K screen uses a 1,750-token budget, set before either final run after four truncations in a preliminary 1,024-token Triton run. Raw Triton/FI scores are **96/95**; completed, final-marker correct scores are **95/95**. Token sequences match on **63/100**. The sole changed parsed answer is #209: Triton truncates without a marker and receives fallback credit for trailing 145; FI completes incorrectly with 34800. This bounded screen does not establish accuracy equivalence.

Core reproduction commands:

```bash
uv run --no-project .venv/bin/python -m pytest -q \
  tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py \
  -k 'gdn_decode or flashinfer_decode or gdn_layer'
uv run --no-project .venv/bin/python -m pytest -q \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/kernels/mamba/test_gdn_forward_core_split.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py
uv run --no-project .venv/bin/python -m pytest -q tests/test_config.py -k gdn_decode
```

## Details

The local folder label is `Qwen3.8-27B-FP8`; config architecture is `Qwen3_5ForConditionalGeneration`, run text-only. All 48 checkpoint biases are BF16 and promote exactly. FP32-bias checkpoints can retain more precision than native BF16 parameters and are outside this evaluation. FI BF16-state padding updates reserved slot 0, so that state dtype is rejected. FP32 bias stabilizes this adapter's signature; it does not fix FI's missing bias-dtype cache key or validate external BF16-populated shared caches.

Provenance: main `3e182185aa5d143b0e69c44607f58a0cc55f3971` Python sources; Torch 2.13.0+cu129, official FI 0.7.0.post1 JIT, reused `0.30.1rc1.dev143+g29468dde8.cu129` native extension and CuTeDSL 4.7.1. Current-main native builds, main's DSL 4.8 pin and H20 remain untested.

Full commands, raw outputs and SHA256 manifests: [evidence ZIP](https://github.com/positive666/vllm/blob/codex/gdn-flashinfer-evidence-20261007/gdn-l20-evidence-rechecked.zip), [micro](https://github.com/positive666/vllm/blob/codex/gdn-flashinfer-evidence-20261007/final-micro-results.md), [serving](https://github.com/positive666/vllm/blob/codex/gdn-flashinfer-evidence-20261007/final-serving-results.md), [quality](https://github.com/positive666/vllm/blob/codex/gdn-flashinfer-evidence-20261007/final-quality-results.md) and [independent recheck](https://github.com/positive666/vllm/blob/codex/gdn-flashinfer-evidence-20261007/independent-recheck-20261007.md). AI assistance was used. The human submitter confirmed review and testing of this patch before submission.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
