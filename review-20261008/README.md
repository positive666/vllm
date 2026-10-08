# #60403 review follow-up validation — 2026-10-08

Tested source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`, parent
`c3424cc55d5258a3d5f104d817d6c6ef850db67f`, main base `8352b242`.
The four-file review delta is in `review.patch`; source archive and five relevant
file hashes are recorded in `source-manifest.json`.

## Results

| Check | Result |
|---|---|
| Adapter, bias loading, indexed state and graph replay | 28 passed |
| Mixed prefill/decode | 12 passed |
| Metadata | 63 passed |
| Configuration | 1 passed |
| Core total | **104 passed, zero failures/errors/skips** |
| Parameter/export probes | Adapter passed in both fresh-process modes |
| Explicit override message | Two constructions, one message |
| Model serving retry | **96 measured +32 warmup requests succeeded** |

L20/SM89, Torch 2.13.0+cu129, FlashInfer 0.7.0.post1, CuTeDSL 4.8.0.
Imported official wheel files match their RECORD hashes. The native library is
reused from the previous environment, SHA256
`3efc749cc7d32a0177376cdabc19d18f406ca2f5a002f933a48997882536d6d8`;
this is not a complete CUDA13 rebuild. Local pre-commit and manual mypy-3.12
checks passed separately.

## Review fixes and constraints

- Use the pinned FI API lazily with explicit `backend="flashinfer"`, without
  compatibility/signature probing. Separate unsupported-configuration errors
  and report an explicit FI override of packed Triton decode disablement.
- Remove the HV%8 constructor rejection. Packed BF16 gates at HV4 put `a` at an
  8-byte offset, while FI requires 16-byte pointer alignment. Contiguous clones
  stage a/b only for HV%8!=0, including B1 where `contiguous()` may retain the
  offset view. Tests mutate gate values and index maps after CUDA graph capture.
  No new HV4 staging-cost benchmark was run.
- Retain `.detach()` on bias Parameters: the default DLPack exporter rejects
  grad-enabled Parameters even in inference mode. With TVM FFI enabled, the raw
  probe instead reaches the gate alignment error. Both repaired adapter calls
  pass the original output/state assertions; input Parameter flags remain true.
- Mixed FI batches copy only the prefill tail into the final output, preserving
  the decode prefix FI already wrote. `auto` stays Triton. K=V128 remains the
  adapter support scope; SM80 support has static evidence but no hardware run.

## Model smoke scope

One successful launch per backend; each ran 48 measured and 16 warmup requests,
512 input/64 output tokens, C1/C8 over three measured rounds. Both used FP32
SSM state, Marlin, FlashAttention2, TP1, prefix caching disabled, and the same
21,845-token KV capacity. The folder labelled Qwen3.8-27B-FP8 declares
`Qwen3_5ForConditionalGeneration` and was served text-only.

C1 paired output sequences match 24/24. C8 sequences match 12/24, with
833/1536 token positions matching (54.23%). At C8, Triton and FI also produce
three and two distinct sequences respectively within their own runs. This
short protocol establishes compatibility, not accuracy equivalence or an
attribution of output differences to the PR. No new throughput, GSM8K or
kernel-gain measurement was performed; prior measurements remain attached
to their original source heads and environments.

## Preserved failures and reproduction

The raw archive retains the initial missing native-extension setup failures in
`results/attempt1`, and nine HV4 gate-alignment failures in `results/attempt2`
before aligned staging was added. Final core results use the new source hashes.

The first final model launches failed before reaching GDN because the generated
FlashMLA Python interface was absent. `prepare_flashmla.py` reproduces main's
CMake wrapper generation using pinned upstream `0eee43b12f034b657133cf2afca6a72ebb6efccf`.
Its generated SHA256 matches the installed official interface. No FlashMLA GPU
kernel is used on SM89. The original `pipeline.exit=1` remains unchanged;
`serving-retry.exit=0` and attempt2 launch records describe the successful retries.

Exact subprocess commands are in `results/recheck-summary.json` and the serving
launch JSONs. `run_pipeline.sh`, `run_serving_retry.sh`, parameter/log probes and
the smoke clients are included. Invoke scripts through `uv` with the prepared
virtual environment, preserving the recorded mounts and runtime versions.
`audit_results.py` recomputes JUnit counts, hashes, request/token checks and
paired output agreement; `result-audit.json` records the verified summary.

`run_review.py` retains an inherited description mentioning a native MTP
diagnostic; its actual task list has no such diagnostic. Benchmark help is only
an import/CLI check, not a measurement. SM80/H20, ROCm, multi-GPU TP, FP32-bias
checkpoints, externally populated FI caches and full CUDA13 builds remain
outside this validation. AI assistance was used.


## Publication confirmation

The human submitting operator confirmed review of every changed line and execution of the relevant tests on 2026-10-08 before publication.

The pending human review wording in `checklist-audit.md` records the earlier audit snapshot; that requirement has now been confirmed for this four-file follow-up patch. See `publication-confirmation.json`.

The ZIP and its manifest retain the original audit snapshot unchanged. `evidence-manifest.json` describes the ZIP contents; `SHA256SUMS` describes the files in this publication directory, including this README and the confirmation.
