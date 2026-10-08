# #60403: main/CuTeDSL 4.8 recheck (2026-10-08)

The published PR head is `c3424cc55d5258a3d5f104d817d6c6ef850db67f`, based on
main `8352b2427f704652da1910f0d53f3fdcc2fa2b0d`. The original eight-file patch
cherry-picked cleanly: added/deleted lines and the FI adapter AST are unchanged,
and current ROCm AITER prefill code is retained. See `rebase-comparison.json`.
AI assistance was used; the human submitter had confirmed review/testing of
the patch before publication.

Actual runtime: L20/SM89, Torch 2.13.0+cu129, Triton 3.7.1, official FI
0.7.0.post1, CuTeDSL **4.8.0** (including cu12/cu13 package providers). FI used
a fresh workspace. Imported FI/DSL files matched wheel RECORD hashes. Python
source hashes match the reviewed archive; reused native binary metadata is
recorded separately and does not identify the Python source revision.

- **95 core tests passed:** 19 adapter/loading, 12 mixed-batch, 63 metadata,
  one config; local pre-commit and manual mypy-3.12 passed.
- **38 supplemental native tests passed:** 16 MTP model-path tests (including
  two actual-native-call assertions) and 22 direct-kernel cases, no skips.
  The unchanged official kernel and a minimal official registration fragment
  were built with **NVCC12.9.86 for SM89**, using an owned Torch shadow header.
  This covers that component's GPU execution, **not the default full CUDA13
  build**. The ordinary runtime still uses the disclosed old native binary.
- Keeping only that old binary produces the retained MTP diagnostic
  **7 passed / 7 skipped / 2 failed**. Runner overall=false is intentional and
  preserved; those cases are not relabelled as passing native coverage.
- Ten FP32 cases passed **128-step** graph trajectories with unchanged
  tolerances, unchanged inactive/null state and page padding.
- One Triton and one FlashInfer serving launch passed **96 smoke requests +32 warmups**,
  all with 512 input/64 output tokens at C1/C8. Both use FP32 SSM, Marlin, FA2,
  64 overridden blocks and **21,845 KV tokens verified in each server log**.
  This short smoke is not a performance or accuracy evaluation.
  Paired measured output streams match 24/24 at C1 and 11/24 at C8; C8 token
  positions match 730/1536 (47.53%). C8 also varies within each backend, so this
  observation neither establishes accuracy equivalence nor isolates a PR effect.

Current production GDNDecode plus identical gated RMSNorm, H=16/HV=48/K=V=128,
median of three alternating round medians, microseconds:

| Batch | State cache | Triton | FI | FI latency reduction |
|---:|---|---:|---:|---:|
| 1 | warm | 6.354 | 4.766 | 24.99% |
| 8 | warm | 19.920 | 17.132 | 14.00% |
| 32 | warm | 200.918 | 193.073 | 3.90% |
| 1 | rotating/cold | 12.142 | 10.433 | 14.08% |
| 8 | rotating/cold | 74.690 | 75.315 | -0.84% |
| 32 | rotating/cold | 311.909 | 307.629 | 1.37% |

Explicit CUDA graph-event timing; projection, convolution, allocation and
compilation are excluded. Mapping is 2.040-2.092us per builder/cache-group
update, separate from the table. Warm-state traffic is not HBM throughput;
cold samples average a rotation group. Differences from old absolute timings
are not attributed to the patch or DSL upgrade. **Auto remains Triton.**

The earlier four-launch ABBA benchmark and 100-question GSM8K screen remain
separate evidence at base `3e182185` / tested head `b2b93acd`, DSL4.7.1. They
were not repeated or transferred to this environment. No significant serving
gain or accuracy equivalence is established. H20, multi-GPU TP, ROCm and a
full native CUDA13 build remain untested.

The first attempt's offline-config and pytest-hook setup failures are retained
under `results/attempt1`. The first two serving launches started successfully,
but the old performance client rejected eight requests before issuing any
completion request (its minimum is 32). A separately named smoke client keeps
the token checks and records its compatibility-only purpose; the original
performance protocol is unchanged. Supervisor shutdown messages after that
parameter rejection are not attributed to a kernel or PR failure.

Upstream pre-commit was skipped by the contributor gate (three merged PRs,
four required); the ReadTheDocs badge is SUCCESS but the actual build was
cancelled after pre-run-check exit183. See [CI details](ci-status.md).

See [the independent audit](independent-recheck.md), `micro-summary.json`,
`smoke-summary.json`, and the [evidence archive](evidence.zip), including raw
outputs, both successful and failed setup logs, reproduction scripts, the
exact patch, native source/registration/build manifest and SHA256 manifest.

Core reproduction:

```bash
uv run --no-project .venv/bin/python -m pytest -q \
  tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py \
  tests/kernels/mamba/test_gdn_forward_core_split.py \
  -k 'gdn_decode or flashinfer_decode or gdn_layer or forward_core_split'
uv run --no-project .venv/bin/python -m pytest -q \
  tests/v1/attention/test_gdn_metadata_builder.py
uv run --no-project .venv/bin/python -m pytest -q tests/test_config.py -k gdn_decode
```

For the supplemental build/launcher, see `native-supplemental/README.md` in
the archive. Compilation and actual test invocation are retained in the logs.
The original environment's missing-op diagnostic is also reproducible with
the unmodified MTP pytest suite without loading the supplemental library.

The owned experiment container was removed after completion; source, caches
and results remain in the owned artifact directory. No other service was
restarted or removed. GPUs 4, 6 and 7 were idle again at 2026-10-08 04:44:30 UTC.
