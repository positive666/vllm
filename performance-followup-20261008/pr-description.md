## Overview

Opt-in FlashInfer GDN ordinary decode, including mixed batches. `auto` remains
Triton; prefill/MTP selection is unchanged.
Review follow-up: `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`, on main `8352b242`.

## Claims

- Lazy pinned API, separate errors, SM80+ guard, explicit-override logging.
- Conditional gate clones replace HV%8 rejection; retain DLPack bias detachment.
- Copy only prefill tail; preserve the FI-written decode prefix.

```bash
vllm serve MODEL --mamba-ssm-cache-dtype float32 \
  --kernel-config '{"gdn_decode_backend":"flashinfer"}'
```

## Validation

```bash
uv run --offline --no-project .venv/bin/python -m pytest \
  tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py \
  tests/kernels/mamba/test_gdn_forward_core_split.py \
  -k 'gdn_decode or flashinfer_decode or gdn_layer or forward_core_split'
uv run --offline --no-project .venv/bin/python -m pytest \
  tests/v1/attention/test_gdn_metadata_builder.py
uv run --offline --no-project .venv/bin/python -m pytest \
  tests/test_config.py -k gdn_decode
```

L20/SM89, Torch 2.13/cu129, FI 0.7.0.post1, CuTeDSL4.8:
**104 passed (40+63+1), no skips**; pre-commit/mypy pass.

**d8 smoke passed:** 96 measured+32 warmups (48+16/backend),
512/64 tokens, C1/C8, FP32 state/Marlin/FA2, identical 21,845-KV-token capacity.
C1 paired sequences match 24/24; C8 match 12/24, token positions 833/1536
(54.23%). C8 varies within backends (Triton/FI: 3/2 variants); differences
cannot be attributed to this PR or establish accuracy equivalence.
Initial missing-FlashMLA-wrapper failure (exit1) retained; official pinned
CMake generation enabled retry (exit0).

[New commands, raw logs, hashes and independent audit](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/review-20261008).

**d8 GSM8K screen:** both backends score **96/100** under raw and strict completed-answer scoring; no truncation, unparsed answers or missing final markers. 200/200 HTTP requests succeeded. Final answers match 100/100, token sequences 62/100; no correctness flips. Same frozen zero-shot subset, C8, T=0, seed42, thinking off, 1750-token budget; one run per backend. This bounded screen does not establish statistical accuracy equivalence. [Results, raw logs and reproduction commands](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/quality-followup-20261008).

## Details

BF16 inputs, FP32 state, K=V=128, grouped heads. Four-head BF16 `a`
has an 8-byte offset; FI requires 16-byte alignment. Default DLPack rejects
grad-enabled Parameters under `inference_mode`.
FP32 bias does not fix FI's cache key.

The [invited #53463 takeover](https://github.com/vllm-project/vllm/pull/53463#issuecomment-6006969716)
succeeds Draft #41966; #53645 targets SM120.
[Duplicate checks and prior evidence](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007).
[Earlier measurements](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/recheck-20261008) used another head/runtime.

**Current-head L20 performance (`d8ae9e9`):**

- GDNDecode + RMSNorm GPU latency: warm-state B1/B8 **-23.87%/-14.44%**; rotated-state B1 **-14.16%**, B8 **+0.94%**. Synthetic HV4 has no measured benefit. Actual timer: CUDA graph events (CUPTI fallback).
- Serving 512/128 tokens, C1/C8, ABBA: **+0.23%/+0.26% throughput**; C8 TPOT and C1 TTFT remain inconclusive. All **384 measured +64 warmup** requests succeeded. Only two independent launches per backend: small descriptive changes, not a statistically established/general serving speedup.

[Raw data, commands and independent audits](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/performance-followup-20261008).

Qwen3.8-labelled model: `Qwen3_5ForConditionalGeneration`, text-only.
Reused native binary.
SM80, CUDA13, H20, ROCm, TP, FP32-bias checkpoints and external FI caches
remain untested. Prior CI was gated/cancelled.

AI assistance was used. The human submitter confirmed line-by-line review
and relevant testing of this four-file follow-up before publication.

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
