#!/usr/bin/env bash
set -euo pipefail
name=$1
owner_id=$2
root=$3
deadline=$((SECONDS + 900))
unmatched=0
while test "$SECONDS" -lt "$deadline"; do
  test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id"
  test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
  stamp=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
  identity=$(docker inspect "$owner_id" --format '{{.Id}}|{{index .Config.Labels "codex.owner"}}|{{.State.Running}}')
  printf '%s|%s\n' "$stamp" "$identity" >> "$root/host-identities.log"
  nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47 --query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,pstate --format=csv,noheader | while IFS= read -r row; do printf '%s,%s\n' "$stamp" "$row"; done >> "$root/host-gpus.csv"
  compute=$(nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
  owners=$(docker top "$name" -eo pid | tail -n +2)
  printf '%s\n%s\nOWNED\n%s\n' "$stamp" "$compute" "$owners" >> "$root/host-processes.log"
  failed=false
  while IFS=',' read -r uuid pid mem; do
    test -n "$uuid" || continue
    pid=$(echo "$pid" | tr -d ' ')
    if ! echo "$owners" | awk '{$1=$1; print}' | grep -Fxq "$pid"; then failed=true; fi
  done <<< "$compute"
  if test "$failed" = true; then unmatched=$((unmatched + 1)); else unmatched=0; fi
  if test "$unmatched" -ge 2; then
    printf '%s\n' "$stamp" > "$root/repeated-unmatched-owner.txt"
    test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id"
    test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
    docker stop -t 30 "$owner_id" >> "$root/monitor-stop.log" 2>&1
    exit 2
  fi
  sleep 5
done
exit 3
