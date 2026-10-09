#!/usr/bin/env bash
set -euo pipefail
root=/data/appadmin/agent-artifacts/gdn60403-late-v3-20261009/results
run=${1:?pass unique run name}
name=gdn60403-late-v3-20261009
owner_id=b8942b9047ed79523b6cb99ba3f5bcc1983857b3b074e31773cd4f43d5968a3c
[[ "$run" =~ ^[a-z0-9-]+$ ]]
test ! -e "$root/$run-host.csv"
printf 'utc,index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate\n' > "$root/$run-host.csv"
previous=' '
for ((sample=0; sample<720; sample++)); do
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  nvidia-smi -i GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e,GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675 \
    --query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate \
    --format=csv,noheader | while IFS= read -r line; do printf '%s,%s\n' "$stamp" "$line"; done >> "$root/$run-host.csv"
  compute=$(nvidia-smi -i GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e,GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675 \
    --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader)
  owners=$(docker top gdn60403-late-v3-20261009 -eo pid)
  printf '%s\n%s\n%s\n' "$stamp" "$compute" "$owners" >> "$root/$run-processes.log"
  current=" $(printf '%s\n' "$owners" | awk 'NR>1 {print $1}' | tr '\n' ' ') "
  while IFS=, read -r uuid pid memory; do
    [[ "$uuid" == GPU-* ]] || continue
    pid=${pid//[[:space:]]/}
    if [[ "$current" != *" $pid "* && "$previous" != *" $pid "* ]]; then
      printf '%s,%s,%s\n' "$stamp" "$uuid" "$pid" > "$root/$run-unknown-pid.txt"
      test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id"
      test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
      docker stop -t 30 "$name" >> "$root/$run-ownership-stop.log" 2>&1
      exit 1
    fi
  done <<< "$compute"
  previous=$current
  if test -f "$root/$run.exit"; then break; fi
  sleep 5
done
