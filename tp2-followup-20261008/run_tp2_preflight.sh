#!/bin/bash
set -euo pipefail
export PYTHONPATH=/source HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/cache/tp2-preflight-probe/vllm
export TRITON_CACHE_DIR=/cache/tp2-preflight-probe/triton
export CUDA_CACHE_PATH=/cache/tp2-preflight-probe/cuda
root=${TP2_RESULTS_ROOT:-/results/tp2-followup-20261008}
blocks=${TP2_GPU_BLOCKS:-64}
mkdir -p "$root"
test ! -e "$root/preflight.exit"
test ! -e "$root/preflight-started.utc"
trap 'code=$?; date -u +%FT%TZ > "$root/preflight-finished.utc"; echo "$code" > "$root/preflight.exit"; if [[ -n "${telemetry_pid:-}" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi' EXIT
date -u +%FT%TZ > "$root/preflight-started.utc"
nvidia-smi --query-gpu=uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version --format=csv > "$root/gpu-inventory.csv"
nvidia-smi topo -m > "$root/container-gpu-topology.txt" 2>&1
set +e
nvidia-smi topo -p2p r > "$root/container-gpu-topology-p2p-read.txt" 2>&1
echo "$?" > "$root/container-gpu-topology-p2p-read.exit"
nvidia-smi topo -p2p w > "$root/container-gpu-topology-p2p-write.txt" 2>&1
echo "$?" > "$root/container-gpu-topology-p2p-write.exit"
set -e
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv -l 5 > "$root/preflight-gpu-telemetry.csv" 2> "$root/preflight-gpu-telemetry.stderr" &
telemetry_pid=$!
if [[ ! -e "$root/runtime-probe.json" ]]; then
    uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/runtime_probe.py --source-head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 --output "$root/runtime-probe.json" > "$root/runtime-probe.log" 2>&1
fi
for arm in triton-compat flashinfer-compat; do
    uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/tp2_supervisor.py --phase preflight --arm "$arm" --gpu-blocks "$blocks" --results-root "$root" > "$root/client-supervisor-$arm.log" 2>&1
done
echo "TP2 compatibility and quality probes completed; audit and freeze protocol before ABBA."
