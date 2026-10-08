#!/bin/bash
set -euo pipefail
export PYTHONPATH=/source HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=${TP2_RESULTS_ROOT:-/results/tp2-followup-20261008}
test -f "$root/protocol.json"
test ! -e "$root/pipeline.exit"
test ! -e "$root/pipeline-started.utc"
trap 'code=$?; date -u +%FT%TZ > "$root/pipeline-finished.utc"; echo "$code" > "$root/pipeline.exit"; if [[ -n "${telemetry_pid:-}" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi' EXIT
date -u +%FT%TZ > "$root/pipeline-started.utc"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv -l 5 > "$root/gpu-telemetry.csv" 2> "$root/gpu-telemetry.stderr" &
telemetry_pid=$!
for arm in triton-a1 flashinfer-b1 flashinfer-b2 triton-a2; do
    uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/tp2_supervisor.py --phase measure --arm "$arm" --results-root "$root" > "$root/client-supervisor-$arm.log" 2>&1
done
TP2_RESULTS_ROOT="$root" uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/collect_round_times.py
