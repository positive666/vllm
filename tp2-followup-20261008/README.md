# TP2 follow-up for PR #60403

Reviewed source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6` is unchanged. Two L20 GPUs use TP2, global
H16/HV48/K128/V128 and local H8/HV24/K128/V128, BF16 inputs and FP32 state.

Expanded-budget C8 diagnostic: FI **6/8** strict correct versus Triton **7/8**, with **1 FI truncation(s)**. FI question 209: parsed 34800, gold 145, stop, 1355 tokens; FI question 255: parsed 20, gold 192, length, 3500 tokens. The retained FI C8 long-generation/output differences are an unresolved limitation in this bounded diagnostic, despite the larger budget. The default `auto` path remains Triton; FI remains opt-in. Integrity audit success does not establish accuracy equivalence or suitability as a general replacement.

- [Serving results](serving-results.md): four independent ABBA launches,
  per-launch medians, paired effects and negative results.
- [Quality/correctness results](quality-results.md): two 100-question model
  screens, every answer flip, 12 distributed rank/cases and a separate matched
  expanded-budget diagnostic. Original scores/truncations are preserved.
- [Reviewer alignment](review-alignment.md): completed d8 feedback and limits.
- [Additional logprob observation](additional-logprob-results.md): shared-prefix top-five distributions through first divergence only.


`protocol.json` freezes 64 blocks,
21,845 KV tokens and the selected UUIDs after
both compatibility/quality launches succeed. Audits independently verify
source, runtime, model, fixture and logs and recompute HTTP/model results.
`tp2-evidence.zip` contains 129 original files with SHA256
`775cf833e113209b0d649bc1873c9b06d1d9d8b6ed0a54f18a400019e0120b44`. `archive-manifest.json` hashes every member.

The served Qwen3.8-labelled FP8 checkpoint declares
Qwen3_5ForConditionalGeneration. This is text-only inference; native libraries
are reused. Local correctness, bounded model quality and serving measurements
have separate conclusions. No accuracy equivalence or universal speedup is claimed.
There is no TP2 microbenchmark; prior TP1 GPU timings remain separately scoped.
The optional DeepEP import probe is unavailable; DP+EP remains unvalidated.
No production source changes were made. AI assistance was used.

Reproduce only in a fresh authorized two-GPU experiment namespace with the
same isolated image/runtime/model. Old outputs intentionally cannot be overwritten.
Preserve initial idle-GPU/process/container identity, topology and five-second
host PID/clock snapshots separately using `host_monitor_tp2.sh` on the host.
The record retains those original snapshots; sampling has bounded coverage.

```sh
export PYTHONPATH=/source HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=/results/tp2-followup-20261008
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/runtime_probe.py --source-head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 \
  --output "$root/runtime-probe.json"
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  /artifacts/tp2_gdn_correctness.py \
  --harness /reference-artifacts/benchmark_current_gdn.py \
  --source-manifest /artifacts/performance-source.json \
  --output-dir "$root/correctness"
bash /artifacts/run_tp2_preflight.sh
# Each preflight also invokes /reference-artifacts/quality_http.py:
# --dataset /oldcache/gsm8k-test.jsonl --samples 100 --concurrency 8
# --max-tokens 1750 --timeout 300 --server-evidence <own TP2 launch>
# Separate matched C1/C8 diagnostic; keeps the original 100-item screen unchanged.
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/tp2_diagnostic_supervisor.py --backend triton
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/tp2_diagnostic_supervisor.py --backend flashinfer
# Audit primary/diagnostic source, quality and capacity before freezing.
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/freeze_tp2_after_diagnostics.py
bash /artifacts/run_tp2_performance.sh
# run_tp2_performance.sh collects original remote round-file-times.json.
```

Preserve `distributed-correctness.exit`, `preflight.exit` and `pipeline.exit`
from the real executions. The immutable archive requires zero for each.

Run the independent `run_tp2_logprobs_host.sh` on the prepared host after ABBA, with the matching owned container/host identity. Preserve its separate PID/clock snapshots and run.exit; this launches two C8×8 top-five observations, never a performance round.

```sh
bash run_tp2_logprobs_host.sh OWNED_CONTAINER /results/tp2-followup-20261008
```

Collect host telemetry before archiving. Package on the prepared runtime:

```sh
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/package_tp2.py archive --root "$root" \
  --archive /results/tp2-evidence.zip
```

After transporting the archive, use a uv-managed interpreter to verify/extract
and independently audit the unchanged raw files. Retain the source manifest,
frozen correctness benchmark, helpers, original quality harness, model config
and full cached GSM8K test dataset with their hashes.

```sh
uv run --no-project .venv/bin/python package_tp2.py extract \
  --root raw --archive tp2-evidence.zip --sha256 775cf833e113209b0d649bc1873c9b06d1d9d8b6ed0a54f18a400019e0120b44
uv run --no-project .venv/bin/python audit_tp2_quality_correctness.py \
  --results raw --source-manifest performance-source.json \
  --correctness-harness benchmark_current_gdn.py \
  --correctness-helper tp2_gdn_correctness.py --quality-harness quality_http.py \
  --supervisor tp2_supervisor.py --dataset gsm8k-test.jsonl \
  --model-config model-config.json --output audit-quality-correctness.json
uv run --no-project .venv/bin/python audit_tp2.py \
  --results raw --source-manifest performance-source.json \
  --output audit-tp2.json
uv run --no-project .venv/bin/python audit_tp2_diagnostics.py \
  --results raw --output audit-diagnostics.json
uv run --no-project .venv/bin/python audit_tp2_logprobs.py \
  --results raw --api-source /path/to/unchanged/d8/source \
  --output audit-logprobs.json

uv run --no-project .venv/bin/python finalize_tp2_evidence.py \
  --archive tp2-evidence.zip --output-dir ready
```

The archive includes all successful/failing statuses actually recorded; reports
are derived from immutable raw data. Approval/publication and GPU cleanup are
handled separately and are not implied by generated text.
