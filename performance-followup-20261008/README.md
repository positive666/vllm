# Exact-head performance follow-up

PR #60403 source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`,
2026-10-08, NVIDIA L20 SM89. No production source changes in this follow-up.

The earlier measurements used another source/runtime and do not validate this
head. This experiment binds the current source hashes and runtime before timing.
Source manifest covers five changed/validated source and test files; the fresh
runtime probe independently checks the three production files and package RECORD
integrity. All launches share the same immutable source, fixture, model config,
runtime and normalized command.

`micro-results.md` reports the production GDNDecode GPU boundary with gated
RMSNorm, including HV4 copies. It distinguishes warm state and rotated-state
conditions, records the actual event-timer fallback, and retains negative results.

`serving-results.md` reports the exact-head ABBA results: throughput +0.23% (C1)
and +0.26% (C8), with no established C8 TPOT or C1 TTFT improvement. These are
small, bounded descriptive gains from only two independent launches per backend.
Both independent audits pass; all requests succeed. The raw bundle is
`performance-evidence.zip`; extract it into `raw/` for the audit commands below.

The independently frozen serving protocol is 512 input/128 output tokens,
temperature zero, seed42, EOS ignored, min_tokens128, prefix caching disabled,
C1/C8, 16 measured requests per round, three rounds, eight warmup requests per
concurrency. Four independent model processes run Triton/FI/FI/Triton (ABBA),
384 measured plus 64 warmup requests. Startup and warmup are excluded from the
reported measured timings. Closed-loop concurrency and loopback HTTP client are
included in the serving boundary. One fixed prompt and one checkpoint are tested.

The Qwen3.8-labelled FP8 model has architecture Qwen3_5ForConditionalGeneration;
this is a text-only TP1 run. It has 48 GDN layers, H16/HV48/K128/V128.
Both arms use BF16 execution, FP32 SSM state, Marlin linear backend, FlashAttention2,
64 GPU blocks and verified 21,845-token KV capacity. Native libraries are reused
from the existing image; a complete CUDA13 build, SM80 hardware, H20, TP and wider models
are not validated by these measurements.

Interpretation uses per-launch medians/ranges, two independent launches per
backend, and the two paired effects. The observed-range rule is a descriptive
heuristic, not a confidence interval or significance test. Request count does not
equal independent experiment count. No statistical equivalence or universally
optimal implementation claim is made.

`run_performance.sh` invokes the fresh probe, micro correctness/timing and all
four model processes sequentially. Server and client scripts preserve commands,
source hashes, protocol/fixture hashes, streamed token IDs, usage and timing.
`audit_performance.py` independently checks raw SSE-derived timings, full token
counts, hashes, launch order, command normalization, logs and recomputed summaries.

`gpu-telemetry.csv` uses five-second sampling. `round-file-times.json` retains
remote original round file mtimes collected before download. Derived round UTC
windows approximate JSON-write end minus measured wall time; they are not
original absolute request instrumentation. `audit_telemetry.py` reports sampled
frequency/temperature/power ranges by launch and concurrency; it does not prove
absence of short-lived throttling or infer thermal causality.

Reproduction in the prepared, isolated image:

```sh
bash /artifacts/run_performance.sh
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/collect_round_times.py
```

Re-audit extracted raw results with the provided helpers and a uv-managed Python:

```sh
uv run --no-project .venv/bin/python audit_performance.py \
  --results raw --source-manifest performance-source.json --output re-audit.json
uv run --no-project .venv/bin/python audit_telemetry.py \
  --results raw --output re-telemetry-audit.json
```

Use a fresh artifact/cache namespace on a new execution; scripts intentionally
refuse to overwrite prior launches and results. All helper scripts are experiment
artifacts, not additions to the vLLM source PR. AI assistance was used.
