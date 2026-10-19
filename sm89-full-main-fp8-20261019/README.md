# #61823: full-main FP8 TP2 validation

Two L20s, full main `3ddc6702a8264da4cf2428aa5ca2e267d84a8841` versus the same source plus `candidate.patch`. The patch still adds 11 dispatch lines and three test cases. This baseline was pinned on October 16; it is not a claim to test every later main commit.

## Serving results

Fresh processes in A/B/B/A order; 512 fixed input tokens, 128 forced output tokens, C1/C2/C8. Each start used eight warmup requests per concurrency, then three rounds of 16 requests. All **576 measured requests succeeded**. Values below are the median of each start's three round values, then the median of the two starts per arm.

| Concurrency | Output tok/s, base → candidate | Change | TPOT p50 ms, base → candidate | Change |
|---|---:|---:|---:|---:|
| 1 | 44.678 → 44.808 | +0.29% | 21.177 → 21.116 | −0.29% |
| 2 | 78.724 → 78.958 | +0.30% | 23.390 → 23.312 | −0.33% |
| 8 | 228.602 → 228.539 | −0.03% | 27.351 → 27.902 | +2.01% |

Throughput ranges across starts: C1 base 44.656–44.700, candidate 44.794–44.822; C2 base 78.696–78.752, candidate 78.951–78.965; C8 base 228.580–228.624, candidate 228.535–228.543. C8 round TPOT medians vary roughly 27.3–28.5 ms in both arms. Its +2.01% aggregate must not be omitted or described as demonstrated latency neutrality. Two starts per arm do not establish statistical significance or a material/general serving speedup.

TPOT uses streamed token timestamps and `(last-first)/(128-1)`. It is a client estimate, not kernel latency or an ITL distribution. Throughput includes the complete fixed-length requests and is not peak saturated throughput.

## Correctness and dispatch

- Existing GPU reference suite: **20 passed, no skips**, 22.46 s.
- Actual baseline/candidate wrappers on this CUDA 13 runtime: **78 shapes bit-exact** for outputs and full state storage; **six cases × 128 changing-input CUDA Graph steps** bit-exact, including active small batches and threshold/padding cases. Inactive slots and page padding preserved; FP64 reference checks passed.
- Every serving start: **16/16 arithmetic questions correct, no truncation**. This is a smoke screen, not a full accuracy benchmark.
- Model outputs are **not bit-exact across starts**: matched baseline/candidate requests agree 73/288; baseline versus baseline agrees 1/144, candidate versus candidate 28/144. Same-baseline C1 runs diverge at the first generated token, before the changed decode kernel. This establishes a baseline repeatability limitation; it does not establish the full cause or prove model-wide numerical equivalence. All four quality runs share 15/16 exact texts; the remaining response differs in whitespace and has the same correct answer.
- Separate diagnostic workers confirm both TP ranks use H=8/Hv=24, K=V=128, BF16 QKV, FP32 state. Candidate B1/B2 selects BV4; B4/B8 retains BV32. Both arms use `MarlinFP8ScaledMMLinearKernel` for their FP8 linear layers.

Diagnostic C1 target-kernel median: approximately **5.12 → 3.65 µs** on each rank. Baseline GDN contributes about 0.75–0.77% of summed kernel durations across the short traces. These traces include prefill and 16-token decode, have profiler overhead, and may include overlapping communication; these sums are neither pure-decode wall-clock shares nor an Amdahl bound. C8 traces also contain smaller tail batches, so C8 is not a completely unchanged execution path.

## Environment and checkpoint

The local checkpoint label is `Qwen3.8-27B-FP8`; its actual architecture is `Qwen3_5ForConditionalGeneration`. It contains block FP8 E4M3 weights (128×128 blocks). This run uses **FP8 weights with Marlin/BF16 computation**, not native FP8 GEMM. `model-manifest.json` identifies the weights, tokenizer, configuration and chat template by SHA256; no official model revision is inferred from the directory label.

Torch 2.14.0+cu130, Triton 3.8.0, FlashInfer 0.7.2, Transformers 5.19.0; CUDA Graph and compilation enabled. TP2, max length 2048, max sequences 8, max batched tokens 1024, memory utilization 0.85, BF16 compute/FP32 SSM, FLASH_ATTN v2, seed 42, prefix caching and speculative decoding disabled. Packed FLA decode enabled, Triton GDN prefill. Dedicated test GPUs on a shared host; clocks were 2520 MHz at ready/after-request snapshots. Full host isolation and continuous clock monitoring were not used.

Both complete source trees share identical native/generated artifacts from the official `dd9e07a25` wheel. That commit is two commits behind the baseline; only XPU Docker and DeepSeek Python files differ, with no native build input changes. All 7,807 tracked source files were verified; only the two intended files differ. The cached native wheel was verified against its RECORD (5,486 files) and repacked locally; its ZIP bytes differ from the official archive. Runtime and source manifests record provenance. CUDA uses a container-private 580.178.04 compatibility library with the unchanged 550.144.03 host driver.

## Reproduction and data

Use an isolated Linux environment with two L20s. Mount the pinned checkpoint read-only at `/model`, this directory at `/work`, and place complete main/candidate checkouts at `/work/source/baseline` and `/work/source/candidate`. Apply `candidate.patch` only to the candidate. Install locked dependencies into `/work/runtime/.venv` with `uv`, and the official matching precompiled wheel. `helpers/install_source.sh`, `sync_native.py` and `probe_runtime.py` document the editable install, identical-native-file setup and runtime/source checks; adapt the recorded wheel/compatibility-library paths to your installation. The source verifier expects the upstream main archive at `/work/main-source.tar.gz`.

```bash
export PYTHONPATH=/work/source/candidate
export LD_LIBRARY_PATH=/work/compat13/usr/local/cuda-13.0/compat
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p /work/results
CUDA_VISIBLE_DEVICES=0 uv run --offline --no-project /work/runtime/.venv/bin/python -m pytest --confcutdir=/work/source/candidate/tests/kernels/fla /work/source/candidate/tests/kernels/fla/test_fused_recurrent_packed_decode.py -v
uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/run_fp8_tp2.py --smoke
uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/run_fp8_tp2.py
uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/analyze_tp2.py
uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/run_fp8_tp2.py --diagnostic
uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/analyze_diagnostic.py
CUDA_VISIBLE_DEVICES=0 uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/wrapper_exact.py
```

The serving harness refuses existing output files. Place the supplied model manifest at `/work/results/model-manifest.json`; its timestamps are specific to the measured mount, so regenerate/verify file hashes and stat metadata on a new copy before use. Do not point the harness at existing benchmark results. Full server commands are recorded in the raw JSON.

`raw-results.zip` includes all request records, fixture, per-start metadata, quality responses, observed launches and CUDA kernel events extracted from all 12 profiler traces. CPU trace metadata is omitted; full original traces remain in the local experiment archive. `summary.json`, `output-stability.json`, `diagnostic-summary.json` and `profile-kernel-summary.json` provide readable aggregates. The separate older cold-cache dataset in the PR is not merged with this serving result.

AI assistance was used for preparation, validation and analysis.
