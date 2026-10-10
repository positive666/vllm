#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-tp2shape-20261010
cd "$base"
test "$(realpath "$base")" = "$base"
test ! -e results/launcher.pid
tar -xzf producer-v1.tar.gz
for file in create_container.sh run_cells.sh monitor_host.sh; do bash -n "$file"; done
sha256sum helpers/benchmark_current_gdn.py create_container.sh run_cells.sh monitor_host.sh > results/uploaded-sha256.txt
bash create_container.sh > results/create.log 2>&1
printf '0\n' > results/create.exit
nohup bash run_cells.sh > results/matrix.log 2>&1 < /dev/null &
printf '%s\n' "$!" > results/launcher.pid
cat results/launcher.pid
