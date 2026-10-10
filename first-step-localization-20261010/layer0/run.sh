#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-rounding-20261010
name=gdn60403-rounding-20261010
root=$base/results
test "$(realpath "$base")" = "$base"
test ! -e "$root/run.exit"
owner_id=$(cat "$root/container-id.txt")
owned() {
 test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id"
 test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
}
monitor_pid=''
finish() {
 result=$?
 trap - EXIT
 set +e
 printf '%s\n' "$result" > "$root/run.exit"
 if test -n "$monitor_pid"; then kill "$monitor_pid"; wait "$monitor_pid"; fi
 if owned; then
  docker stop -t 20 "$owner_id" > "$root/cleanup.log" 2>&1
  printf '%s\n' "$?" > "$root/cleanup.exit"
 fi
 date -u +%FT%T.%NZ > "$root/completed.txt"
 nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$root/final-gpus.csv"
 nvidia-smi -i GPU-4af84168-6d29-6128-fee3-1b146cce9b47 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv > "$root/final-processes.csv"
 exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
owned
date -u +%FT%T.%NZ > "$root/started.txt"
bash "$base/monitor_host.sh" "$name" "$owner_id" "$root" &
monitor_pid=$!
set +e
timeout --foreground -k 30s 600s docker exec -e PYTHONPATH=/source -e PYTHONDONTWRITEBYTECODE=1 "$name" uv run --offline --no-project /cache/gdn-runtime/bin/python /rounding/localize.py > "$root/numeric.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$root/docker-exec.exit"
test ! -e "$root/repeated-unmatched-owner.txt"
exit "$result"
