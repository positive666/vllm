#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-divergence-v2-20261010
test "$(cat "$base/results/preflight.exit")" = 0
test ! -e "$base/results/launcher.pid"
test ! -e "$base/results/matrix.exit"
nohup bash "$base/run_cells.sh" > "$base/results/matrix.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$base/results/launcher.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$base/results/launcher-started.txt"
kill -0 "$pid"
cat "$base/results/launcher.pid"
