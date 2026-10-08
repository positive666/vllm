#!/bin/bash
set -euo pipefail
root=/data/appadmin/agent-artifacts/gdn60403-review-20261008/results/tp2-followup-20261008
end=$((SECONDS + 7200))
while (( SECONDS < end )); do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw --format=csv,noheader
    printf 'COMPUTE_PROCESSES\n'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    printf 'OWNED_CONTAINER_PIDS\n'
    docker top gdn60403-tp2-20261008 -eo pid,comm
    printf 'ROUND_END\n'
    [[ -e "$root/pipeline.exit" ]] && break
    sleep 5
done
