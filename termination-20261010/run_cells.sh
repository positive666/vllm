#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-termination-20261010
name=gdn60403-termination-20261010
root=$base/results
test "$(realpath "$base")" = "$base"
test ! -e "$root/matrix.exit"
owner_id=$(cat "$root/container-id.txt")
owned() {
  test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id" &&
  test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
}
idle() {
  for uuid in GPU-4af84168-6d29-6128-fee3-1b146cce9b47 GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e; do
    used=$(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
    test "$used" -le 16 || return 1
    test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)" || return 1
  done
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
for arm in flashinfer triton; do
  owned
  idle
  test ! -e "$root/budget-$arm.exit"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$root/budget-$arm-started.txt"
  set +e
  timeout --foreground -k 30s 1200s docker exec \
    -e PYTHONPATH=/long-helpers:/source \
    -e PYTHONDONTWRITEBYTECODE=1 "$name" \
    uv run --offline --no-project /cache/gdn-runtime/bin/python \
    /long-helpers/long_free_driver.py --backend "$arm" --mode eager --run "budget-$arm" \
    > "$root/budget-$arm.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$root/budget-$arm.exit"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$root/budget-$arm-completed.txt"
  test "$result" -eq 0
  idle
done
