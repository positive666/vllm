#!/usr/bin/env bash
set -euo pipefail
base=/validation/gdn60403-rebase-20261010
name=gdn60403-rebase-20261010
root=$base/results
test "$(realpath "$base")" = "$base"
test ! -e "$root/matrix.exit"
owner_id=$(cat "$root/container-id.txt")
owned() {
  test "$(docker inspect "$name" --format '{{.Id}}')" = "$owner_id" &&
  test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}')" = "$name"
}
idle() {
  for uuid in GPU-A GPU-B; do
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
  nvidia-smi -i GPU-A,GPU-B --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$root/final-gpus.csv"
  nvidia-smi -i GPU-A,GPU-B --query-compute-apps=gpu_uuid,pid,used_memory --format=csv > "$root/final-processes.csv"
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
run_case() {
  label=$1
  shift
  owned
  idle
  test ! -e "$root/$label.exit"
  date -u +%FT%TZ > "$root/$label-started.txt"
  set +e
  timeout --foreground -k 30s 1800s docker exec -w /source \
    -e PYTHONPATH=/long-helpers:/source -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_CACHE_ROOT=/cache/focused/vllm -e TRITON_CACHE_DIR=/cache/focused/triton \
    -e CUDA_CACHE_PATH=/cache/focused/cuda -e FLASHINFER_WORKSPACE_BASE=/cache/focused/flashinfer \
    "$name" uv run --offline --no-project /cache/gdn-runtime/bin/python "$@" > "$root/$label.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$root/$label.exit"
  date -u +%FT%TZ > "$root/$label-completed.txt"
  test "$result" -eq 0
  idle
}
run_case focused /artifacts/run_review.py --source-head abf17c7c071b1b6bdd81ca9756894998e5e8b32d --results /results/focused
run_case tp2-correctness -m torch.distributed.run --standalone --nproc-per-node=2 /artifacts/tp2_gdn_correctness.py --harness /artifacts/benchmark_current_gdn.py --source-manifest /artifacts/performance-source.json --output-dir /results/tp2-correctness
for mode in graph eager; do
  for arm in triton flashinfer; do
    run_case "$arm-$mode" /long-helpers/long_free_driver.py --backend "$arm" --mode "$mode" --run "$arm-$mode"
  done
done
