#!/bin/bash
set -euo pipefail
export PYTHONPATH=/source HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/cache/performance-micro/vllm
export TRITON_CACHE_DIR=/cache/performance-micro/triton
export CUDA_CACHE_PATH=/cache/performance-micro/cuda
root=/results/performance-followup-20261008
mkdir -p "$root"
trap 'code=$?; date -u +%FT%TZ > "$root/pipeline-finished.utc"; echo "$code" > "$root/pipeline.exit"; if [[ -n "${telemetry_pid:-}" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi' EXIT
date -u +%FT%TZ > "$root/pipeline-started.utc"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv -l 5 > "$root/gpu-telemetry.csv" 2> "$root/gpu-telemetry.stderr" &
telemetry_pid=$!
uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/runtime_probe.py --source-head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 --output "$root/runtime-probe.json" > "$root/runtime-probe.log" 2>&1
uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/benchmark_current_gdn.py --source-head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 --runtime-probe "$root/runtime-probe.json" --head-shapes 16:48,2:4 --batches 1,8 --dtypes float32 --index-dtypes int32 --steps 128 --rounds 3 --repeat-ms 100 --l2-multiple 5 --timer cupti --output "$root/micro-current.json" > "$root/micro-current.log" 2>&1
for arm in triton-a1 flashinfer-b1 flashinfer-b2 triton-a2; do
    uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/performance_supervisor.py --arm "$arm" > "$root/client-supervisor-$arm.log" 2>&1
done
