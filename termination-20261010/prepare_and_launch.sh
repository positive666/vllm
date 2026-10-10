#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-termination-20261010
name=gdn60403-termination-20261010
cd "$base"
test "$(realpath "$base")" = "$base"
test ! -e results/preparation.exit
handed_off=false
finish() {
  code=$?
  trap - EXIT
  set +e
  printf '%s\n' "$code" > results/preparation.exit
  if test "$handed_off" = false && test -f results/container-id.txt; then
    cid=$(cat results/container-id.txt)
    if test "$(docker inspect "$name" --format '{{.Id}}' 2>/dev/null)" = "$cid" && test "$(docker inspect "$name" --format '{{index .Config.Labels "codex.owner"}}' 2>/dev/null)" = "$name"; then
      docker stop -t 30 "$cid" > results/preflight-cleanup.log 2>&1
      printf '%s\n' "$?" > results/preflight-cleanup.exit
    fi
  fi
  exit "$code"
}
trap finish EXIT
tar -xzf producer-v1.tar.gz
for f in create_container.sh run_cells.sh monitor_host.sh launch_matrix.sh; do bash -n "$f"; done
sha256sum helpers/*.py create_container.sh run_cells.sh monitor_host.sh launch_matrix.sh > results/uploaded-sha256.txt
bash create_container.sh > results/create.log 2>&1
printf '0\n' > results/create.exit
for arm in flashinfer triton; do
  set +e
  timeout --foreground -k 10s 180s docker exec -e PYTHONPATH=/long-helpers:/source -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= "$name" uv run --offline --no-project /cache/gdn-runtime/bin/python /long-helpers/long_free_driver.py --backend "$arm" --mode eager --run "plan-$arm" --plan-only > "results/plan-$arm.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" > "results/plan-$arm.exit"
  test "$code" -eq 0
done
printf '0\n' > results/preflight.exit
bash launch_matrix.sh
handed_off=true
