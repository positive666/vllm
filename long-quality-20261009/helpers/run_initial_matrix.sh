#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-long-20261009
root=$base/results
name=gdn60403-long-20261009
test "$(realpath "$base")" = "$base"
test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
test ! -e "$root/initial-matrix.exit"
trap 'result=$?; printf "%s\n" "$result" > "$root/initial-matrix.exit"' EXIT

for ((poll=0; poll<420; poll++)); do
  if test -f "$root/triton-eager-1.exit"; then break; fi
  sleep 5
done
test "$(cat "$root/triton-eager-1.exit")" = 0
test -f "$root/triton-eager-1/completed.json"
test -f "$root/triton-eager-1/shutdown.json"

for item in flashinfer:eager triton:graph flashinfer:graph; do
  backend=${item%%:*}
  mode=${item##*:}
  run=$backend-$mode-1
  test ! -e "$root/$run"
  test ! -e "$root/$run.exit"
  for uuid in GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675; do
    used=$(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
    test "$used" -lt 16
    test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
  done
  printf '%s starting %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run"
  bash "$base/helpers/monitor_host.sh" "$run" > "$root/$run-monitor.log" 2>&1 &
  set +e
  docker exec -e PYTHONPATH=/source "$name" timeout 1800 uv run --offline --no-project \
    /cache/gdn-runtime/bin/python /long-helpers/long_free_driver.py \
    --backend "$backend" --mode "$mode" --run "$run" > "$root/$run.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$root/$run.exit"
  printf '%s completed %s exit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run" "$result"
  test "$result" = 0
  test -f "$root/$run/completed.json"
  test -f "$root/$run/shutdown.json"
done
