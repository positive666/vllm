# Late-prefix GDN numerical diagnostics

Generated from completed real records. Forced histories are not model accuracy or serving performance results. The source is unchanged d8; prior unforced truncation remains unresolved.

`summary.json` reports all 672 updates per captured arm, exact own-production fidelity and frozen-reference math. `exceptions.json` lists each observed failed check. Capture initial condition NOGO remains NOGO. Full-attention KV, complete hidden states, original aliasing/absolute addresses and full state pools are unobserved.

`snapshot-manifest.json` preserves SHA/size for all 1,344 raw binaries retained remotely. Only rank0/layer0/step3499 from each arm is included as an explicit representative. All sidecars, trace/producer/runtime/helper records and supplied audit/host/cleanup receipts are retained. No padding-row coverage is inferred from synthetic guards. AI assistance was used.

## Raw audit and public-copy scope

`public-redactions.json` maps each changed observation record's raw SHA/size to its public redacted SHA/size. Private network addresses are removed from public logs/runtime observations. Auditors ran on the original remote raw records; recorded consumer SHA bindings continue to identify those raw files. Public redacted copies cannot rerun every raw consumer provenance gate byte for byte, and consumer acceptance rules are unchanged. Source, tensor and mathematical evidence remain exact. Generic source/reference localhost defaults and public RFC network definitions are not host observations.

The original pair status REVIEW_REQUIRED and any same-sample ownership exception remain explicit despite passing numerical replays. See supplied `retained-attempts.md` and attempt records for the pre-model environment-gate failure; it did not execute model kernels and the stopped container was not OOM-killed. The successful attempt added the missing read-only runtime-addons mount with the same frozen helpers, source and reference.

## Local replay and propagated native drift

The 1,344 own-production updates reproduce exact outputs and all compact captured poststate. Both kernels pass the unchanged FP64 allclose (atol=0.01, rtol=0.01) and relative-L2<1% checks from each capture's prestate. On one common captured prestate, maximum T/F output relative L2 is 2.86844e-4 and state relative L2 is 4.31375e-7. Maximum output relative L2 against unrounded FP64 math is 0.30543%; absolute error about 0.0152 satisfies the combined atol+rtol rule, not an absolute-only 0.01 bound.

Separately propagated native arms share the eight full-vocabulary prefill-logit hashes and 768 initial active GDN recurrent prestates. Their later observed prestate/poststate relative-L2 drift reaches about 1.65213%/1.65483%, and layer-output drift about 3.52642%. These are propagated-context comparisons, not single-update replay errors; the local 1% math threshold cannot judge model quality from them. Convolution state, full-attention KV and complete hidden states were not observed. Actual bias dtype remains T BF16/F FP32 even though FP32-compared parameter values match, so reduction order alone is not established as the cause. Forced histories do not validate natural EOS, accuracy or performance, and the unforced truncation remains unresolved. See support/independent-replay-review.json and its summary for all metrics.

## Final owned cleanup checkpoint

support/final-cleanup-receipt.json binds the original eight cleanup records. The expected container ID and owner label match before/after; the sleep-infinity container was actively stopped, is exited/PID0 with OOMKilled=false, and cleanup command exit 0. Its exit 137 is not model OOM. At 2026-10-09T13:32:57Z the recorded GPU 6/7 checkpoint is 1 MiB/0% with no compute-process rows. This is a recorded checkpoint, not a claim of current availability or container removal. The pair's REVIEW_REQUIRED/one earlier unmatched ownership observation is retained.

The original public-late-v1 CPU-export ZIP is preserved separately. support/public-late-v1-local-review.json records its original SHA/size and manifest. Raw cleanup records remain local/remote; any public address redaction is explicitly mapped in public-redactions.json.

The completed v1 mathematical/package audit receipt is included as `support/public-late-v1-independent-review.json`. Its bindings refer to the preserved v1 export; final v2 audits are retained separately to avoid a manifest/self-audit cycle.
