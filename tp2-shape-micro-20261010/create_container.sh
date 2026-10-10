#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-tp2shape-20261010
name=gdn60403-tp2shape-20261010
reference=/data/appadmin/agent-artifacts/gdn60403-late-v2-20261009
test "$(realpath "$base")" = "$base"
test -f "$base/helpers/benchmark_current_gdn.py"
test -f "$reference/reference/reference-lock.json"
if docker inspect "$name" >/dev/null 2>&1; then exit 1; fi
for uuid in GPU-4af84168-6d29-6128-fee3-1b146cce9b47 GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e; do
  used=$(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$used" -le 16
  test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
done
test "$(df --output=avail -B1 "$base" | tail -n 1)" -gt 53687091200
test ! -e "$base/results/container-id.txt"
docker run -d --name "$name" --label codex.owner="$name" \
  --gpus '"device=GPU-4af84168-6d29-6128-fee3-1b146cce9b47,GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e"' \
  --shm-size=8g --entrypoint /bin/bash \
  -v /data/models/Qwen3.8-27B-FP8:/model:ro \
  -v "$base/cache":/cache \
  -v /data/appadmin/agent-artifacts/gdn60403-recheck-20261008/cache/gdn-runtime:/cache/gdn-runtime:ro \
  -v /data/appadmin/agent-artifacts/gdn-flashinfer-takeover-20261006/cache/runtime-addons:/oldcache/runtime-addons:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/cache:/seed-cache:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/scripts:/reference-artifacts:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/scripts/tp2:/artifacts:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/source:/source:ro \
  -v /data/chengrui4/agent-artifacts/workspaces/pr53463-20260927:/study:ro \
  -v "$reference/helpers":/late-helpers:ro \
  -v "$reference/reference":/short-reference:ro \
  -v "$base/helpers":/long-helpers:ro \
  -v "$base/reference":/natural-reference:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/results:/oldresults:ro \
  -v "$base/results":/results \
  sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2 \
  -c 'sleep infinity' > "$base/results/container-id.txt"
docker inspect "$name" --format '{{json .State}}' > "$base/results/container-initial-state.json"
docker inspect "$name" --format '[{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},"Config":{"Labels":{{json .Config.Labels}}},"State":{{json .State}}}]' > "$base/results/container-inspect-initial.json"
cat "$base/results/container-id.txt"
