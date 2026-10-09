#!/usr/bin/env bash
set -euo pipefail
base=/data/appadmin/agent-artifacts/gdn60403-shadow-20261009
name=gdn60403-shadow-20261009
test "$(realpath "$base")" = "$base"
test -d "$base/helpers"
if docker inspect "$name" >/dev/null 2>&1; then
  echo 'Refusing to replace an existing container' >&2
  exit 1
fi
for uuid in GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675; do
  used=$(nvidia-smi -i "$uuid" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$used" -lt 16
  test -z "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)"
done
mkdir -p "$base/cache" "$base/results"
docker run -d --name "$name" --label codex.owner="$name" \
  --gpus '"device=GPU-a2be9350-e0b5-d8b0-9844-51c35161cd3e,GPU-b7783a01-c440-29bd-ccb8-98ca7acf6675"' \
  --shm-size=8g --entrypoint /bin/bash \
  -v /data/models/Qwen3.8-27B-FP8:/model:ro \
  -v "$base/cache":/cache \
  -v /data/appadmin/agent-artifacts/gdn60403-recheck-20261008/cache/gdn-runtime:/cache/gdn-runtime:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/cache:/seed-cache:ro \
  -v /data/appadmin/agent-artifacts/gdn-flashinfer-takeover-20261006/cache:/oldcache:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/scripts:/reference-artifacts:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/scripts/tp2:/artifacts:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/source:/source:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-review-20261008/results:/oldresults:ro \
  -v /data/appadmin/agent-artifacts/gdn60403-recheck-20261008/cache:/previous-cache:ro \
  -v /data/chengrui4/agent-artifacts/workspaces/pr53463-20260927:/study:ro \
  -v "$base/helpers":/shadow-helpers:ro \
  -v "$base/results":/results \
  vllm/vllm-openai:v0.29.0-cu129 -c 'sleep infinity'
