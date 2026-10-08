Updated in `d8ae9e9`; thanks @gau-nernst for the review.

- Removed signature probing and shortened the docstring; use the pinned lazy FI API with explicit backend selection. Split validation errors, log the explicit override, and copy only the mixed-batch prefill tail.
- Removed the HV%8 rejection. Real packed H=2/HV=4 gates expose an 8-byte `a` offset versus FI's 16-byte alignment, including B1. Conditional aligned a/b clones fix this; indexed-state and graph-replay tests cover changing gates and mappings.
- Kept `.detach()`: the default DLPack exporter rejects grad-enabled Parameters even under `inference_mode`. The fresh TVM-FFI probe instead reaches the gate-alignment error; the adapter passes both modes.
- Changed the guard to SM80+, with actual hardware validation limited to L20/SM89. K=V128 remains this adapter's support scope. `auto` stays Triton; FP32 bias stabilizes the adapter signature, not an FI dtype requirement or a shared-cache fix.

L20: **104 tests passed, no skips**, plus **96 smoke requests +32 warmups succeeded**; local pre-commit/mypy passed. C8 outputs also vary within each backend, so no accuracy equivalence or new performance gain is claimed. The native binary was reused; full CUDA13 and SM80 hardware remain unvalidated.

[Commands, preserved failed attempts and independent audit](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/review-20261008). AI assistance was used; the human submitter confirmed review and relevant testing before this update.
