#!/usr/bin/env bash
set -euo pipefail
name=gdn60403-late-v3-20261009
expected_id=b8942b9047ed79523b6cb99ba3f5bcc1983857b3b074e31773cd4f43d5968a3c
root=/data/appadmin/agent-artifacts/gdn60403-late-v3-20261009/results
test "$(docker inspect --format '{{.Id}}' "$name")" = "$expected_id"
test "$(docker inspect --format '{{index .Config.Labels "codex.owner"}}' "$name")" = "$name"
test ! -e "$root/final-cleanup-before.json"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$root/final-cleanup-start.txt"
docker inspect "$name" > "$root/final-cleanup-before.json"
docker stop --time 10 "$name" > "$root/final-cleanup-stop.log"
docker inspect "$name" > "$root/final-cleanup-after.json"
test "$(docker inspect --format '{{.State.Running}}' "$name")" = false
nvidia-smi -i 6,7 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$root/final-cleanup-gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | awk '/GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e|GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675/' > "$root/final-cleanup-processes.csv"
test ! -s "$root/final-cleanup-processes.csv"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$root/final-cleanup-completed.txt"
printf '0\n' > "$root/final-cleanup.exit"
