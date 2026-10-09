# TP2 unforced long-generation follow-up

All eight planned launches completed with exit 0 and eight outputs each. The final quality audit passes its integrity checks. The production source remains `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`; this experiment changes no production code. AI assistance was used. This is targeted diagnostic evidence, not a general accuracy, performance or merge-readiness result.

## Protocol

Two L20 GPUs, TP2, BF16 execution and FP32 SSM state, using the same FP8 checkpoint with architecture `Qwen3_5ForConditionalGeneration`. The eight selected GSM8K indices are `[198,206,209,228,255,285,292,318]`: targets 209/255 and six controls.

Every launch queues all eight fixed prompt-token sequences before generation through the synchronous offline in-process engine. Temperature is 0, seed 42, thinking and prefix caching are disabled, context length is 4,096, output budget is 3,500, and GPU blocks are 128. Marlin and FlashAttention2 are unchanged. Native trace replay is disabled, no generated histories are supplied, EOS is respected, and no API logprobs are collected.

Initial order: Triton/eager, FI/eager, Triton/graph, FI/graph. Repeats reverse backend order within each mode: FI/eager, Triton/eager, FI/graph, Triton/graph. Every launch uses a fresh interpreter and isolated owned cache. Native binaries were reused; this is not a fresh full CUDA build.

This synchronous offline protocol differs from the prior HTTP concurrency-8 experiment, whose asynchronous scheduling is inferred from its captured configuration and source defaults. Matching initial submission does not force subsequent natural histories or batch schedules to remain equal. The original HTTP results remain retained evidence.

## Outputs and termination

Strict correctness requires the parsed gold answer, normal stop, no truncation and a `####` answer marker. All six controls are strict-correct in each launch. Counts are reported separately; repeated selected questions are not independent population-accuracy samples.

| Run | Strict correct | Truncated | Missing marker | Native dispatch counts |
|---|---:|---:|---:|---|
| Triton eager 1 | 7/8 | 0 | 0 | NONE 3396; graph 0 |
| FI eager 1 | 7/8 | 1 | 1 | NONE 3500; graph 0 |
| Triton graph 1 | 7/8 | 0 | 0 | FULL 2338; NONE 1 |
| FI graph 1 | 6/8 | 0 | 0 | FULL 2706; NONE 1 |
| FI eager 2 | 7/8 | 1 | 1 | NONE 3500; graph 0 |
| Triton eager 2 | 7/8 | 0 | 0 | NONE 3396; graph 0 |
| FI graph 2 | 7/8 | 0 | 0 | FULL 1524; NONE 1 |
| Triton graph 2 | 8/8 | 0 | 0 | FULL 1639; NONE 1 |

The dispatch counts combine native logged intervals and pending native stats. They establish observed graph execution for graph cells and no graph dispatch for eager cells. They are not timing measurements or counts of output tokens.

| Run | Question 209: gold 145 | Tokens / finish | Question 255: gold 192 | Tokens / finish |
|---|---|---:|---|---:|
| Triton eager 1/2 | 145, correct | 3,396 / stop | 176, wrong | 1,217 / stop |
| FI eager 1/2 | 145, correct | 1,715 / stop | 10, wrong; no marker | 3,500 / length |
| Triton graph 1 | 145, correct | 1,027 / stop | 176, wrong | 2,339 / stop |
| FI graph 1 | 34800, wrong | 2,707 / stop | 176, wrong | 2,475 / stop |
| FI graph 2 | 34800, wrong | 1,525 / stop | 192, correct | 359 / stop |
| Triton graph 2 | 145, correct | 1,640 / stop | 192, correct | 359 / stop |

Equal 7/8 scores conceal different failure and termination behavior: FI eager question 255 hits the token cap without an answer marker; Triton eager question 255 is wrong but stops normally. Output token lengths are not latency.

## Independent repeat comparison

| Backend / mode | Identical output-token histories | Changed indices | Complete eight-example arrays |
|---|---:|---|---|
| Triton eager | 8/8 | none | identical |
| FI eager | 8/8 | none | identical |
| Triton graph | 6/8 | 209, 255 | different |
| FI graph | 5/8 | 206, 209, 255 | different |

The eager comparisons include prompts, output IDs, raw text, scoring and termination fields. FI eager's repeated example-array SHA is `807c4d1e48dad6e28b9b25a1939199e0f229f9c18067682a0f1175c0a753cef5`. Its question-255 truncation repeats in both independent launches.

Graph-repeat variation occurs in both backends: Triton scores 7/8 then 8/8; FI scores 6/8 then 7/8. The repeated FI eager truncation establishes that CUDA graph replay is not necessary for this symptom under this protocol. These observations do not identify its cause, establish determinism generally, or rule out graph effects elsewhere.

## Integrity, ownership and scope

`quality-final8-v1.json` binds producer/source/runtime/input/log records and passes integrity checks. All eight launch and shutdown records are retained; no launch process failure is reported. The manifest and archive receipt preserve the frozen evidence bytes and helper provenance.

`ownership-final8-v1.json` passes its record-integrity checks, with 688 same-sample ownership confirmations, two unmatched observations and zero unknown-owner observations. `strict_same_sample_ownership_pass` remains false. The two candidate spawn/exit races are possible explanations, not proof of ownership at those instants; sequential telemetry is not atomic and cannot exclude unobserved short-lived activity or PID reuse.

At `2026-10-09T10:38:45Z`, the recorded post-matrix query shows GPUs 6/7 at 1 MiB and 0% utilization, with no reported compute PID. The owned long-generation container was still running at this checkpoint. This is a point-in-time idle observation, not a final container-cleanup claim.

The separate `container-cleanup.txt` receipt at `2026-10-09T10:44:35Z` records the same owned container ID in the exited state after it was stopped, and GPUs 6/7 at 1 MiB and 0% utilization. This does not claim removal of its files or current availability after that checkpoint.

The earlier HTTP C8 FI 6/8 versus Triton 7/8 and FI truncation remain unresolved evidence. Passing 248 captured local updates and the bounded 512-token teacher-forced comparison do not erase them. This unforced follow-up adds no performance result and supplies no adapter/kernel quality root cause. It cannot establish general accuracy equivalence, hardware limitation, H20/SM80 coverage, other-model coverage, BF16-state behavior, MTP, DP+EP or merge readiness.

Keep FI opt-in and `auto` on Triton. Further localization must establish a reproducible violated output/state/index/alignment contract before proposing a production fix. Matched-history replay is a numerical diagnostic; it is not a substitute for this unforced quality test.

## Reproduction and files

The frozen producer is `helpers/long_free_driver.py`; matrix scripts record actual invocations. Use a new run name because existing output/cache directories are rejected. The owned-container command has this form:

```bash
docker exec -e PYTHONPATH=/source gdn60403-long-20261009 \
  timeout 1800 uv run --offline --no-project \
  /cache/gdn-runtime/bin/python /long-helpers/long_free_driver.py \
  --backend flashinfer --mode eager --run flashinfer-eager-2
```

Each run directory retains complete outputs, fixture/reference/source bindings, launch/runtime/sampling/shutdown records and its producer-helper snapshot. Root files retain all eight exit markers, full logs, native graph observations, host/PID telemetry, matrix/auditor logs and exits, and the post-matrix idle query. `helpers/` retains scorer, producer, auditors, monitor and CPU contracts. The package file manifest records SHA-256 and byte size for every payload file; the ZIP includes the same manifest. Draft PR text and credential-bearing remote access helpers are excluded.

## Public observation copies

Private network addresses in public log/runtime observation copies are explicitly redacted. `public-redactions.json` maps original raw SHA/size to each public redacted SHA/size. Original auditor log digests still identify the retained raw records; redacted copies do not claim original-byte identity. Source/helper/math records are preserved unchanged. The unredacted unpublished candidate is retained locally and is not for publication. The original integrity audits ran against remote raw records. Public redacted copies cannot rerun every raw consumer provenance gate byte for byte; the consumer acceptance rules are unchanged.
