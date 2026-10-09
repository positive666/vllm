#!/usr/bin/env bash
set -euo pipefail
root=/data/appadmin/agent-artifacts/gdn60403-shadow-20261009/results
run=${1:?pass the unique run name}
[[ "$run" =~ ^[a-z0-9-]+$ ]]
out=$root/$run-host.csv
test ! -e "$out"
printf 'utc,index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate\n' > "$out"
for ((sample=0; sample<720; sample++)); do
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  nvidia-smi -i GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e,GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675 \
    --query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate \
    --format=csv,noheader | while IFS= read -r line; do printf '%s,%s\n' "$stamp" "$line"; done >> "$out"
  printf '%s\n' "$stamp" >> "$root/$run-processes.log"
  nvidia-smi -i GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e,GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675 \
    --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader >> "$root/$run-processes.log"
  docker top gdn60403-shadow-20261009 -eo pid >> "$root/$run-processes.log"
  if test -f "$root/$run.exit"; then break; fi
  sleep 5
done
