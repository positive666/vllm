#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-serving-repeats-v3-20261010
cd "$base"
test ! -e mock-wait-proof
mkdir -p mock-wait-proof/bin
awk '/^idle\(\) \{/{active=1} active{print} active && /^}/{exit}' run_cells.sh > mock-wait-proof/idle_fn.sh
cat > mock-wait-proof/bin/nvidia-smi <<'MOCK'
#!/bin/bash
set -eu
if [[ "$*" == *--query-gpu=* ]]; then
  n=0; if test -f "$WAIT_TEST_STATE"; then n=$(cat "$WAIT_TEST_STATE"); fi
  n=$((n+1)); printf '%s\n' "$n" > "$WAIT_TEST_STATE"
  used=1
  if { test "$WAIT_TEST_CASE" != own_then_idle && test "$WAIT_TEST_CASE" != unknown_then_idle; } || test "$n" -eq 1; then used=17249; fi
  printf 'GPU-4af84168-6d29-6128-fee3-1b146cce9b47, %s\nGPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e, %s\n' "$used" "$used"
else
  n=$(cat "$WAIT_TEST_STATE")
  if test "$WAIT_TEST_CASE" = foreign || { test "$WAIT_TEST_CASE" = unknown_then_idle && test "$n" -eq 1; }; then printf '%s\n' 'GPU-4af84168-6d29-6128-fee3-1b146cce9b47, 99, 17234 MiB'
  elif test "$WAIT_TEST_CASE" = busy || test "$n" -eq 1; then printf '%s\n' 'GPU-4af84168-6d29-6128-fee3-1b146cce9b47, 11, 17234 MiB'; fi
fi
MOCK
printf '#!/bin/bash\nprintf "PID\\n11\\n"\n' > mock-wait-proof/bin/docker
printf '#!/bin/bash\nexit 0\n' > mock-wait-proof/bin/sleep
chmod +x mock-wait-proof/bin/*
export PATH="$base/mock-wait-proof/bin:$PATH"
test "$(command -v nvidia-smi)" = "$base/mock-wait-proof/bin/nvidia-smi"
test "$(command -v docker)" = "$base/mock-wait-proof/bin/docker"
for mode in own_then_idle unknown_then_idle foreign busy; do
  mkdir "mock-wait-proof/$mode"
  export WAIT_TEST_CASE="$mode" WAIT_TEST_STATE="$base/mock-wait-proof/$mode/counter"
  set +e
  bash -c 'set -euo pipefail; source "$1"; root=$2; name=cpu-only; owner_id=cpu-only; owned() { return 0; }; idle cpu-only' _ "$base/mock-wait-proof/idle_fn.sh" "$base/mock-wait-proof/$mode" > "mock-wait-proof/$mode/output.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" > "mock-wait-proof/$mode/actual.exit"
  count=$(wc -l < "mock-wait-proof/$mode/idle-wait.jsonl")
  case "$mode" in
    own_then_idle) test "$code" -eq 0; test "$count" -eq 3; jq -s -e '.[0].busy and (.[1].busy|not) and (.[2].busy|not)' "mock-wait-proof/$mode/idle-wait.jsonl" >/dev/null;;
    unknown_then_idle) test "$code" -eq 0; test "$count" -eq 3; jq -s -e '.[0].unmatched and (.[1].busy|not) and (.[2].busy|not)' "mock-wait-proof/$mode/idle-wait.jsonl" >/dev/null;;
    foreign) test "$code" -eq 1; test "$count" -eq 40; jq -e '.unmatched' "mock-wait-proof/$mode/idle-wait.jsonl" >/dev/null;;
    busy) test "$code" -eq 1; test "$count" -eq 40;;
  esac
  printf '%s actual_exit=%s observations=%s\n' "$mode" "$code" "$count"
done
printf '%s\n' 'CPU_MOCK_IDLE_GATE_PASS; no real GPU or docker calls'
