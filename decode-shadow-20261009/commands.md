# Reproduction boundary and commands

Use a new authorized experiment namespace, the exact d8 source and the matching
model/runtime/reference data. `helpers/create_container.sh` records the original
mounts and two selected GPU UUIDs; substitute authorized local paths/GPUs rather
than rerunning against an occupied host. It uses a readonly source/model,
isolated writable cache/results, private SHM and no host ports. Frozen prior
reference/source manifests are supplied by the linked earlier evidence package.
Completed experiment paths and output files intentionally cannot be overwritten.
The commands below reproduce the probes in a fresh namespace; all Python uses uv.

```sh
# Snapshot capture (copies actual decode inputs and output/state; eager only).
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /shadow-helpers/shadow_supervisor.py --backend flashinfer --run fi-c8-v2

# A fresh FI workspace avoids sharing diagnostic bias specializations.
export PYTHONPATH=/source
export FLASHINFER_WORKSPACE_BASE=/cache/replay-actual-v2
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /shadow-helpers/replay_capture.py \
  --captures-dir /results/fi-c8-v2/captures \
  --source-manifest /artifacts/performance-source.json \
  --bias-mode actual-only --device cuda:0 \
  --output /results/fi-c8-v2/replay-full-v2.json

# Final driver enables native replay and disables engine-core multiprocessing.
# Each arm uses a separate cache/process and shuts down its owned workers.
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /shadow-helpers/force_driver.py --backend triton --run force-triton-v3
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /shadow-helpers/force_driver.py --backend flashinfer --run force-flashinfer-v3
uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /shadow-helpers/audit_force.py \
  --triton /results/force-triton-v3 --flashinfer /results/force-flashinfer-v3 \
  --reference /oldresults/tp2-followup-20261008/diagnostics/triton.json \
  --source-manifest /artifacts/performance-source.json --source-root /source \
  --helpers-dir /shadow-helpers --output /results/force-audit-v3.json
```

Frozen run-local helpers in the archive reproduce the earlier V2/V1 attempts;
the top-level final driver is the improved V3 configuration. Do not silently
replace those frozen versions when re-auditing history. The audit uses each run's
recorded producer hashes and rejects altered prompts, input/history, missing
steps, invalid natural sampler logits and modified frozen helpers. It reports
batch/prefill equality separately from integrity success. `--helpers-dir` only
records current live helper hashes and does not override the frozen snapshots.

The full 248-capture replay requires the remote full corpus; the public archive
only supplies two representative snapshots. Original logs retain exact failures,
runtime warnings and cleanup shutdown messages. No clean-CI, fresh native-build,
optional DeepEP or unrestricted hardware-coverage claim is made.
