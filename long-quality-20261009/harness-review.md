# Independent long-free harness review

Read-only review; no GPU execution or production edits. Reviewed producer hashes:

- long_free_driver.py: dd61dc9c2924472a8065daff38e7e2a5af2774f770bc18fb2014ecc66e52fb8f
- long_free_common.py: 9fe3403eb45386cbf116fff22fb199930b7a7332c9e84abe4821d6f75fe59af3

No blocking driver issue found in these versions. Generation uses public LLM.generate and SamplingParams with native trace replay disabled, no supplied output tokens, ignore_eos false and no logprob collection. The engine is required to be InprocClient; inputs are the exact eight reconstructed chat-template prompt ID sequences, verified again against returned prompt IDs. Source manifest, reference, tokenizer/config files, four producer/scorer files and sampled runtime fields are bound in artifacts. Scoring reuses the SHA-locked prior parser and preserves raw correctness, completed correctness, strict marker/stop correctness and truncation separately. SamplingParams struct-field serialization was corrected before this reviewed version. The actual trace_replay.py source path and InprocClient.shutdown(timeout=30) signature were verified.

Publication still requires actual zero exit markers, complete eight-output records, source/prompt/config/hash audit and GPU release. A launch that fails after generation or during cleanup is retained as failed. The external supervisor supplies bounded process timeout; the public collective_rpc reads are separately bounded. This review does not replace that runtime audit.

Graph coverage must be established from native graph execution stats/log tables, including earlier flushed intervals and final pending stats. Capture descriptors or configured mode alone prove capture/eligibility, not that user generation replayed a graph. Verify graph execution with positive native generation counts and eager without graph replay; retain any fallback as failed graph coverage. These untimed quality cells add no serving performance evidence.

## Original expanded HTTP scheduling comparison

Inspected frozen tp2-evidence.zip entries diagnostics/serve-triton.json, diagnostics/serve-flashinfer.json and their logs. Both original workers log Using V2 Model Runner. Both original launches use enforce_eager false, FULL_AND_PIECEWISE graph configuration, maximum capture size 64, and no explicit async-scheduling override. Neither original log contains an async-scheduling-disable warning. The unchanged source has SchedulerConfig.async_scheduling None and VllmConfig enables async scheduling by default unless incompatible pooling/speculative/executor/ROCm/PP conditions apply. The original CUDA TP2/no-spec/V2/mp setup follows the enabled default. This is a source-derived inference because original artifacts do not serialize the final scheduler configuration.

The new driver explicitly uses synchronous scheduling plus offline in-process all-at-once submission, with native cudagraph metrics enabled. It is a controlled diagnostic rather than literal reproduction of the original asynchronous HTTP concurrency-8 workload. Even identical initial queuing does not force later batch schedules to match after natural token histories or completion lengths diverge. If the synchronous matrix is clean, a production/default-async scheduling follow-up remains necessary before claiming the earlier HTTP long-generation symptom resolved. Keep the original FI C8 6/8 versus Triton 7/8 and FI truncation as retained unresolved evidence.

No production fix is supported by the passing local replay evidence alone. Keep FI opt-in, auto on Triton and Draft; target a concrete reproducible contract failure before editing source.
