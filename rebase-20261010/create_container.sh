#!/usr/bin/env bash
set -euo pipefail
base=/validation/gdn60403-rebase-20261010
name=gdn60403-rebase-20261010
test "$(realpath "$base")" = "$base"
test -f "$base/helpers/long_free_driver.py"
if docker inspect "$name" >/dev/null 2>&1; then exit 1; fi
for uuid in GPU-A GPU-B; do
  used=$(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$used" -le 16
  test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
done
test "$(df --output=avail -B1 "$base" | tail -n 1)" -gt 53687091200
test ! -e "$base/results/container-id.txt"
docker run -d --name "$name" --label codex.owner="$name" \
  --gpus '"device=GPU-A,GPU-B"' \
  --shm-size=8g --entrypoint /bin/bash \
  -v /data/models/Qwen3.8-27B-FP8:/model:ro \
  -v "$base/cache":/cache \
  -v /validation/gdn60403-recheck-20261008/cache/gdn-runtime:/cache/gdn-runtime:ro \
  -v /validation/gdn-flashinfer-takeover-20261006/cache/runtime-addons:/oldcache/runtime-addons:ro \
  -v /validation/gdn60403-review-20261008/cache:/seed-cache:ro \
  -v /validation/gdn60403-review-20261008/scripts:/reference-artifacts:ro \
  -v "$base/helpers":/artifacts:ro \
  -v "$base/source":/source:ro \
  -v /validation/workspaces/pr53463-20260927:/study:ro \
  -v "$base/helpers":/long-helpers:ro \
  -v /validation/gdn60403-review-20261008/results:/oldresults:ro \
  -v "$base/results":/results \
  sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2 \
  -c 'sleep infinity' > "$base/results/container-id.txt"
docker inspect "$name" --format '{{json .State}}' > "$base/results/container-initial-state.json"
docker inspect "$name" --format '[{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},"Config":{"Labels":{{json .Config.Labels}}},"State":{{json .State}}}]' > "$base/results/container-inspect-initial.json"
cat "$base/results/container-id.txt"
