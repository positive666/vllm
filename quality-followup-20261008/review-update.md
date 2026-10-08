Updated in `d8ae9e9`; thanks @gau-nernst. This addresses each review point:

- Removed signature probing, use the pinned lazy FI API with explicit `backend="flashinfer"`, and shortened the config docstring. Split validation errors and log the explicit override of `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`. Mixed batches copy only the prefill tail, preserving the FI-written decode prefix.
- Removed the HV%8 rejection. Packed H=2/HV=4 `a` has an 8-byte offset versus FI's 16-byte alignment, including B1. Conditional contiguous a/b clones fix this; indexed-state and graph-replay tests cover changing gates and mappings.
- Kept `.detach()`: the default DLPack exporter rejects grad-enabled Parameters under `inference_mode`. A fresh TVM-FFI probe instead reaches gate misalignment; the adapter passes both modes.
- Changed the guard to SM80+, based on the kernel requirements; actual hardware validation remains L20/SM89. K=V128 is this adapter's current support scope, not an FI API requirement.
- `auto` intentionally stays Triton. FI accepts BF16 bias; this adapter retains FP32 bias for a consistent compiled signature, without claiming to fix FI's cache key.

L20: **104 tests passed without skips**, plus **96 smoke requests and 32 warmups succeeded**; pre-commit/mypy passed. C8 differs within and across backends; this establishes no accuracy equivalence or performance gain.

Same-head GSM8K screen: **96/100 for both backends**, with no truncation, unparsed answers or correctness flips; 200/200 HTTP requests succeeded. Final answers match 100/100, token sequences 62/100. Fixed zero-shot 100-question subset, C8, T=0, seed42, thinking off, 1750-token budget; one run per backend, no statistical equivalence claim.

[New quality results, raw evidence and commands](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/quality-followup-20261008); [focused tests and parameter probes](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/review-20261008). Native binaries were reused; SM80 hardware/full CUDA13 builds remain unvalidated. AI assistance was used; the submitter confirmed review and testing of the code update.
