# Human review and reproduction

Review `review.patch` against main
`3e182185aa5d143b0e69c44607f58a0cc55f3971`. Its SHA256 is
`cca1d2ddf75bd0d6128f81ba985c3af335cadbb7cc8dcda0ac335437db8b7f8d`;
`review-patch-manifest.json` records the eight source/test hashes.

Run from the checkout root in an already configured CUDA vLLM environment,
using its virtualenv. The focused adapter selector includes the three real
layer-construction/sharded-loading regressions.

```bash
uv run --no-project .venv/bin/python -m pytest -q \
  tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py \
  -k 'gdn_decode or flashinfer_decode or gdn_layer'
uv run --no-project .venv/bin/python -m pytest -q \
  tests/v1/attention/test_gdn_metadata_builder.py
uv run --no-project .venv/bin/python -m pytest -q \
  tests/kernels/mamba/test_gdn_forward_core_split.py
uv run --no-project .venv/bin/python -m pytest -q \
  tests/test_config.py -k gdn_decode
uv run --no-project .venv/bin/python -m pytest -q \
  tests/kernels/mamba/test_gdn_fused_mtp.py
```

Recorded results: adapter 19 passed, metadata 63 passed, mixed 12 passed,
configuration 1 passed. MTP produced 7 passed / 7 skipped / 2 failed on both
the changed source and unchanged main with the reused CUDA 12.9 native runtime;
the two failures require the absent fused post-convolution native op. This does
not replace testing with a native extension built from the target main revision.

```bash
uv run --no-project .venv/bin/python -m pre_commit run --files \
  tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py \
  tests/kernels/mamba/test_gdn_forward_core_split.py \
  tests/kernels/mamba/test_gdn_fused_mtp.py \
  tests/test_config.py \
  tests/v1/attention/test_gdn_metadata_builder.py \
  vllm/config/kernel.py \
  vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
  vllm/v1/attention/backends/gdn_attn.py
uv run --no-project .venv/bin/python -m pre_commit run mypy-3.12 \
  --hook-stage manual --files \
  vllm/config/kernel.py \
  vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
  vllm/v1/attention/backends/gdn_attn.py
git diff --check
```

The opt-in serving configuration is:

```bash
vllm serve MODEL --mamba-ssm-cache-dtype float32 \
  --kernel-config '{"gdn_decode_backend":"flashinfer"}'
```

Final serving: 768 measured and 64 warmup requests succeeded, with four audited
fresh launches and equal 21,845-token KV capacity. C1/C8 paired throughput
changes are +0.279%/+0.113%; C8's second pair regresses 0.208%. No significance
or meaningful serving improvement is established. The local model folder is
`Qwen3.8-27B-FP8`, architecture `Qwen3_5ForConditionalGeneration`, text-only.

The experiment server command, with the checkpoint mounted at `/model`, was:

```bash
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  -m vllm.entrypoints.openai.api_server --model /model \
  --served-model-name qwen-fp8 --host 127.0.0.1 --port 8000 \
  --language-model-only --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --mamba-ssm-cache-dtype float32 --seed 42 \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
  --kernel-config '{"gdn_decode_backend":"triton","linear_backend":"marlin"}' \
  --num-gpu-blocks-override 64
```

Change only `gdn_decode_backend` between `triton` and `flashinfer`, using fresh
processes in A1/B1/B2/A2 order. Keep the shared input fixture. Container-local
paths below correspond to scripts and data in the evidence package.

```bash
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/http_bench_client.py bench --model qwen-fp8 \
  --fixture /results/final/http-fixture.json --arm triton-a1 \
  --requests 32 --warmup-requests 8 \
  --server-evidence /results/final/serve-final-triton-a1.json \
  --output-dir /results/final/http
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/quality_http.py --arm triton --model qwen-fp8 \
  --dataset /cache/gsm8k-test.jsonl --samples 100 --max-tokens 1750 \
  --concurrency 8 --server-evidence /results/final/serve-final-triton-a1.json \
  --output /results/final/quality-triton.json
```

Run the quality command once per backend on its first launch, changing only
the arm, evidence record and output path (FI uses B1). Seed 42, temperature 0,
thinking disabled and the matched
100-question subset are fixed in the client. Raw T/FI correctness is 96/95;
completed/final-marker correctness is 95/95; token sequences match on 63/100.
The sole changed answer is question 209: Triton truncates without a marker and
receives fallback credit for trailing 145; FI completes incorrectly with 34800.
This screen does not show statistical accuracy equivalence.

Human review of every changed line and execution of the relevant final tests
are still required. Public evidence links remain to be added before submission.
