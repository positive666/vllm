# Additional real-decode replay results

Reviewed source: `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`. No production source changes were made.

An instrumented TP2 eager run supplied 248 real ordinary-decode snapshots: 124 per rank, covering 48 GDN layers. Actual batch sizes 8/7/3/2 contributed 206/14/14/14 snapshots. These are sampled local updates, replayed independently on an L20 using the same captured QKV, gates, parameter values and prestate. Active page mappings are preserved through compact indices; original tensor strides and pointer alignment modulo 256 are restored.

All 248 FlashInfer replays exactly reproduce captured active output and all captured poststate pages. Both backends pass the frozen checks against the FP64 mathematical reference rounded to the production output/state dtypes: pointwise `atol=rtol=0.01` and aggregate relative-L2 strictly below `0.01`.

| Triton鈥揊lashInfer comparison | Maximum absolute error | Maximum aggregate relative-L2 |
|---|---:|---:|
| Active output | 1.52587890625e-5 | 6.37911411434036e-5 |
| Active state | 3.814697265625e-6 | 3.8897870142536134e-7 |

Each column is an independently computed maximum across captures. Relative-L2 values are ratios, not percentages. Captured null and sampled unused state pages remain exactly unchanged; reconstructed storage guards also pass. There are no actual padding rows, so this run adds no padding-row coverage. All 48 checkpoint `dt_bias` tensors are BF16; promotion to FP32 preserves their values exactly.

No failed local contract check was observed, so these results provide no evidence-based reason for a production-code change. Each replay resets to the captured FI prestate; it does not propagate a separate Triton state trajectory between snapshots. The results do not establish accuracy equivalence, identify the cause of earlier C8 answer/truncation differences, or replace those retained negative observations. Eager capture and serialization can alter scheduling and trajectories; the snapshot capture/replay alone is neither teacher forcing, full-state sequence replay nor a performance measurement. Other original pool pages, original storage-gap contents, absolute addresses and alias relationships are outside the replay scope.

Evidence: `replay-full-v2.json`, SHA256 `b9845fb320a4847d52dad8fcc80b07c75e075eaaefb94bdfecf3432e9f36e633`; `replay-full-v2-summary.json`; `model-bias-probe.json`. AI assistance was used.

