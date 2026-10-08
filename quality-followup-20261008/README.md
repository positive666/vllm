# PR #60403 model-quality follow-up

This adds a new-head quality screen to the existing review validation. Source
remains `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`; no PR code changed.
The human submitter already confirmed review and relevant testing of that patch.
AI assistance was used to run and audit this evidence update.

Both backends score **96/100**, including strict completed-answer scoring;
all 200 HTTP requests succeed. Final answers match 100/100, exact token
sequences 62/100. No correctness gains or losses, truncation, missing final
markers, or unparsed answers were observed in this fixed subset. Read
[quality-results.md](quality-results.md) for all four common incorrect answers,
the scoring rules and limits. Equal counts do not prove accuracy equivalence.

Protocol: frozen 100-question GSM8K zero-shot subset, selection seed42, T=0,
request seed42, thinking disabled, C8, 1750 output tokens. One run and one
successful launch per backend. L20/SM89, TP1, BF16 input, FP32 SSM, Marlin,
FA2, prefix caching off, 64 blocks and 21,845 actual KV tokens in both arms.
This is a quality screen; timing is not a performance benchmark.

The reused runtime/native binaries and generated dependency wrapper are
documented in [the earlier review evidence](../review-20261008/README.md).
The runtime probe is the earlier snapshot (three production files); both new
supervisors rehash all five declared source/test files before launching.
Neither model process reruns the runtime probe. SM80, H20, TP and full CUDA13
source builds remain unvalidated. No new throughput/kernel gain is claimed.

## Reproduce

Use the earlier documented container/runtime mounts, including `/source`,
`/model`, `/cache/gdn-runtime`, `/oldcache` (cached dataset) and `/results`.
Place the archived scripts under `/artifacts` and source-manifest.json at
`/artifacts/quality-followup-source.json`. Start with an unused results directory;
the supervisor refuses to overwrite previous runs.

```bash
mkdir -p /results/quality-followup-20261008
bash /artifacts/run_quality.sh
# Or one arm at a time:
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/quality_supervisor.py --backend triton
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/quality_supervisor.py --backend flashinfer
```

`quality-evidence.zip` preserves raw per-question text/tokens, supervisor
records/logs, client logs, exit files, the frozen dataset and executed harness.
`audit.json` and `quality-results.md` are independently recomputed evidence,
not edits to those raw files. The old evaluation contributes only the frozen
question/gold fixture, never its scores. `audit_quality.py` can be rerun using
its CLI paths (`--help`), the complete cached dataset and the linked prior
fixture. SHA256SUMS covers every published file except itself.

The submitter deleted the previous completed-review comment before this
reassessment. The older pending-work comment is historical; review-update.md
is the replacement prepared after this new screen. No unexecuted diagnostics
are included. The experiment container was stopped and GPU7 released.
