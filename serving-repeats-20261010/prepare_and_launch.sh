#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-serving-repeats-v3-20261010
name=gdn60403-serving-repeats-v3-20261010
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
cp protocol.json results/protocol.json
cp http-fixture.json results/http-fixture.json
set +e
timeout --foreground -k 15s 180s docker exec -e PYTHONPATH=/source -e PYTHONDONTWRITEBYTECODE=1 "$name" uv run --offline --no-project /cache/gdn-runtime/bin/python /artifacts/runtime_probe.py --output /results/runtime-probe.json --source-head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 > results/runtime-probe.log 2>&1
code=$?
set -e
printf '%s\n' "$code" > results/runtime-probe.exit
test "$code" -eq 0
printf '0\n' > results/preflight.exit
bash launch_matrix.sh
handed_off=true
