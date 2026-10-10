#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-serving-repeats-v3-20261010
name=gdn60403-serving-repeats-v3-20261010
root=$base/results
test "$(realpath "$base")" = "$base"
test ! -e "$root/matrix.exit"
owner_id=$(cat "$root/container-id.txt")
owned() {
  test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id" &&
  test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
}
idle() {
  local phase=initial attempt stable=0 memory compute owners busy foreign used uuid pid extra stamp
  if test "$#" -gt 0; then phase=$1; fi
  for attempt in $(seq 1 40); do
    owned || { printf '%s\n' 'Idle wait: owner changed' >&2; return 1; }
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    memory=$(nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47,GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e --query-gpu=uuid,memory.used --format=csv,noheader,nounits) || return 1
    compute=$(nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47,GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader) || return 1
    owners=$(docker top "$name" -eo pid | tail -n +2) || return 1
    busy=false; foreign=false
    while IFS=',' read -r uuid used; do
      used=$(printf '%s' "$used" | tr -d ' ')
      [[ "$used" =~ ^[0-9]+$ ]] || return 1
      if test "$used" -gt 16; then busy=true; fi
    done <<< "$memory"
    while IFS=',' read -r uuid pid extra; do
      test -n "$uuid" || continue
      busy=true
      pid=$(printf '%s' "$pid" | tr -d ' ')
      if ! printf '%s\n' "$owners" | awk '{$1=$1; print}' | grep -Fxq "$pid"; then foreign=true; fi
    done <<< "$compute"
    jq -nc --arg timestamp "$stamp" --arg phase "$phase" --argjson attempt "$attempt" --arg container_id "$owner_id" --arg memory "$memory" --arg compute "$compute" --arg owned_pids "$owners" --argjson busy "$busy" --argjson unmatched "$foreign" '{timestamp:$timestamp,phase:$phase,attempt:$attempt,container_id:$container_id,memory_csv:$memory,compute_csv:$compute,owned_pids:$owned_pids,busy:$busy,unmatched:$unmatched}' >> "$root/idle-wait.jsonl"
    if test "$busy" = false; then stable=$((stable+1)); else stable=0; fi
    if test "$stable" -ge 2; then return 0; fi
    sleep 1
  done
  printf '%s\n' 'Idle wait timed out after 40 bounded observations' >&2
  return 1
}
monitor_pid=''
finish() {
  result=$?
  trap - EXIT
  set +e
  printf '%s\n' "$result" > "$root/matrix.exit"
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null
    printf '%s\n' "$?" > "$root/monitor-final.exit"
  fi
  if owned; then
    docker inspect "$name" --format '{{json .State}}' > "$root/cleanup-before.json"
    docker inspect "$owner_id" --format '[{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},"Config":{"Labels":{{json .Config.Labels}}},"State":{{json .State}}}]' > "$root/container-inspect-before.json"
    docker stop -t 30 "$owner_id" > "$root/cleanup-stop.log" 2>&1
    printf '%s\n' "$?" > "$root/cleanup-stop.exit"
    docker inspect "$name" --format '{{json .State}}' > "$root/cleanup-after.json"
    docker inspect "$owner_id" --format '[{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},"Config":{"Labels":{{json .Config.Labels}}},"State":{{json .State}}}]' > "$root/container-inspect-after.json"
  fi
  nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47,GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$root/final-gpus.csv"
  nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47,GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e --query-compute-apps=gpu_uuid,pid,used_memory --format=csv > "$root/final-processes.csv"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$root/matrix-completed.txt"
  exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
owned
test "$(docker inspect "$name" --format '{{.State.Running}}')" = true
idle
bash "$base/monitor_host.sh" "$name" "$owner_id" "$root" &
monitor_pid=$!
for arm in triton-a1 flashinfer-b1 flashinfer-b2 triton-a2 flashinfer-b3 triton-a3 triton-a4 flashinfer-b4; do
  owned
  idle "before-$arm"
  test ! -e "$root/serve-$arm.exit"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$root/serve-$arm-started.txt"
  set +e
  timeout --foreground -k 30s 900s docker exec \
    -e PYTHONPATH=/source \
    -e PYTHONDONTWRITEBYTECODE=1 "$name" \
    uv run --offline --no-project /cache/gdn-runtime/bin/python \
    /artifacts/tp2_supervisor.py --phase measure --arm "$arm" --results-root /results \
    > "$root/serve-$arm-supervisor.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$root/serve-$arm.exit"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$root/serve-$arm-completed.txt"
  test "$result" -eq 0
  idle "after-$arm"
done
