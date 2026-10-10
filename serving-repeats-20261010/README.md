# PR #60403: complete TP2 HTTP serving replication

Unchanged source d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6. Qwen3.8-27B-FP8 on two L20s (physical GPUs 4/6), TP2, CUDA graphs, FP32 recurrent state, Marlin linear backend and FA2. This compares Triton and explicit FlashInfer at the same PR revision.

All eight launches completed: four independent processes/backend, frozen ABBA BAAB order. Each had C1/C8, three measured rounds of 16 requests and eight warmups/concurrency: 768 measured +128 warmup requests, all successful. Same 512-token synthetic prompt and 128 generated tokens (min_tokens128, ignore_eos, greedy/seed42). Startup and warmup excluded; native model/API uninstrumented; no output-token sequence supplied.

Client-observed whole-serving measurements include model work, TP communication, API and loopback HTTP/client overhead; external networking is excluded. TTFT is send-to-first nonempty text event. Token-ID TTFT is separate. TPOT is first-to-last token-ID event span/127, not pure ITL if events coalesce. Request latency includes terminal usage/DONE. Throughput uses successful output tokens/full measured-round wall time.

Values are medians of four launch values, each the median of three round metrics. Delta=(FI/Triton-1)*100: positive throughput is better; negative latency is better. Paired ranges use the four predeclared pairs. Per-round p95 has 16 requests and is descriptive.

| C | Metric | Triton | FlashInfer | FI change | Four paired changes: min .. max |
|---|---|---:|---:|---:|---:|
| 1 | output_tokens_per_second | 44.531 | 44.708 | +0.397% | +0.309% .. +0.427% |
| 1 | ttft_ms.p50 | 173.037 | 172.925 | -0.065% | -0.636% .. +0.277% |
| 1 | ttft_ms.p95 | 174.172 | 173.870 | -0.173% | -0.880% .. +0.245% |
| 1 | token_ttft_ms.p50 | 173.037 | 172.925 | -0.065% | -0.636% .. +0.277% |
| 1 | tpot_ms.p50 | 21.262 | 21.176 | -0.408% | -0.437% .. -0.332% |
| 1 | tpot_ms.p95 | 21.265 | 21.178 | -0.406% | -0.432% .. -0.327% |
| 1 | request_ms.p50 | 2873.814 | 2862.502 | -0.394% | -0.421% .. -0.298% |
| 1 | request_ms.p95 | 2874.937 | 2863.374 | -0.402% | -0.437% .. -0.299% |
| 8 | output_tokens_per_second | 228.542 | 229.656 | +0.487% | +0.304% .. +0.635% |
| 8 | ttft_ms.p50 | 848.096 | 844.290 | -0.449% | -4.226% .. -0.263% |
| 8 | ttft_ms.p95 | 1343.581 | 1339.395 | -0.312% | -1.393% .. -0.228% |
| 8 | token_ttft_ms.p50 | 848.096 | 844.290 | -0.449% | -4.226% .. -0.263% |
| 8 | tpot_ms.p50 | 27.333 | 27.211 | -0.446% | -0.623% .. +3.635% |
| 8 | tpot_ms.p95 | 33.086 | 32.933 | -0.464% | -0.574% .. -0.318% |
| 8 | request_ms.p50 | 4427.626 | 4407.540 | -0.454% | -0.599% .. -0.328% |
| 8 | request_ms.p95 | 4814.390 | 4793.919 | -0.425% | -0.545% .. -0.069% |

serving-summary-v1.json retains per-launch/round values and all paired changes. These are bounded descriptive replications on a shared host; no significance, population confidence bound or general speedup claim. C8 is a short two-wave load, not prolonged saturation. Other lengths/concurrency, natural answer lengths, H20 and other models are not established. Previous graph quality/length differences and late-prefix ownership REVIEW_REQUIRED remain unresolved. Fixed token counts do not establish quality equivalence. FI stays opt-in and the PR stays Draft.

Metrics: SERVING_RECORDS_AND_METRICS_PASS; client overlap: CLIENT_CONCURRENCY_PASS across 64 rounds. Resources: REVIEW_REQUIRED, counts {"compute_observations": 876, "matched_same_snapshot": 874, "partial_frames": 0, "process_snapshots": 528, "unknown_uuid": 0, "unmatched": 2}. Drain gate: REVIEW_REQUIRED, 42 observations, 2 retained unmatched. These verdicts are separate; later idle snapshots never erase earlier unmatched records.

Five-second host samples align approximately to measured UTC windows. SM clocks/temperature/utilization/memory are retained per arm/concurrency/GPU. Clocks/power/fans were not changed or locked; sequential sampling cannot exclude transients or shared-host CPU contention. Memory clocks and power were not continuously sampled.

Both interrupted attempts remain in history/ and are not pooled. v1 completed 2 launches then matrix exit 1 at the inter-launch guard; its failing NVML sample was not recorded, so exact cause is unproven. v2 completed 3 launches then stopped after B2: at 04:29:57Z NVML still listed PID 216348 but later docker-top did not; the preceding 04:29:55Z observation owned that PID. This supports an exit/sampling race, not a failed model request. Both cleanup commands exited 0 and both GPUs were idle afterward.

v3 changes only measurement-external draining: at most 40 observations, require two consecutive zero-compute-PID/<=16MiB samples before another launch. Any occupancy blocks launching; persistent occupancy times out. Unmatched records remain unmatched. Four CPU mocks cover owned release, transient unmatched release, persistent foreign and persistent owned occupancy. Model supervisor, HTTP client and runtime probe are byte-identical across attempts.

Frozen protocol/producer hashes and shell commands reproduce the workload. Model metadata/index hashes were verified; full weight files were not rehashed. Native libraries were reused; the known caught optional DeepEP import warning is retained. This is not a fresh CUDA13 rebuild. GPU-operation speedup percentages from earlier experiments are not substituted for whole-serving gains.

Public request/token/metric records are unchanged. Only RFC1918 IPv4 digits in logs are masked, preserving byte lengths. raw-public-map.json binds raw/public full logs and recorded prefixes at exact offsets. Original audits use raw hashes. Public rechecks use a separately named auditor for mapped prefixes and must reproduce identical numerical comparisons; private original addresses are not included in the mapping.

After extracting results-public-v1.tar.gz into this directory, CPU checks use uv and an explicit venv Python:

~~~bash
uv run --offline --no-project <venv-python> -X utf8 audit_serving_public.py --root . --output rechecked-serving.json
uv run --offline --no-project <venv-python> -X utf8 resource_audit.py --results results --manifest producer-manifest.json --expected-owner gdn60403-serving-repeats-v3-20261010 --output rechecked-resources.json
uv run --offline --no-project <venv-python> -X utf8 audit_idle_wait.py --root . --output rechecked-idle.json
uv run --offline --no-project <venv-python> -X utf8 audit_telemetry.py --root . --output rechecked-telemetry.json
uv run --offline --no-project <venv-python> -X utf8 audit_client_load.py --root . --output rechecked-concurrency.json
~~~

AI assistance was used. Evidence only; production source is unchanged.

The two monitor-unmatched rows share timestamp 05:05:50Z, near FI b3's last measured response (05:05:50.397321Z). Second-resolution host stamps cannot establish that these rows fall outside the measured interval. The PIDs occur in another owned snapshot, consistent with an exit/visibility race, but the monitor verdict remains REVIEW_REQUIRED. Separately, two drain snapshots after FI b2/b4 retain unmatched PIDs; both waits subsequently recorded two idle samples. Measured-window SM samples were 2520 MHz, with observed temperatures 55–86 C; unsampled transients remain possible.
