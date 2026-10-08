#!/usr/bin/env bash
# Invoke with nohup only after the authorized formal TP2 runner exits zero.
set -euo pipefail
experiment_container=${1:?pass the existing authorized experiment container}
experiment_results=${2:-/results/tp2-followup-20261008}
docker exec "$experiment_container" test ! -e "$experiment_results/logprob-diagnostics/run.exit"
docker exec "$experiment_container" mkdir -p "$experiment_results/logprob-diagnostics"
closeout() {
  experiment_exit=$?
  trap - EXIT
  docker exec "$experiment_container" sh -c 'printf "%s\n" "$1" > "$2/logprob-diagnostics/run.exit"' \
    -- "$experiment_exit" "$experiment_results"
  exit "$experiment_exit"
}
trap closeout EXIT
for experiment_arm in triton flashinfer; do
  docker exec -w /source -e PYTHONPATH=/source "$experiment_container" \
    uv run --offline --no-project /cache/gdn-runtime/bin/python \
    /artifacts/tp2_logprobs_supervisor.py \
    --backend "$experiment_arm" --results-root "$experiment_results"
done
docker exec -w /source -e PYTHONPATH=/source "$experiment_container" \
  uv run --offline --no-project /cache/gdn-runtime/bin/python \
  /artifacts/tp2_top_logprobs.py compare \
  --triton "$experiment_results/logprob-diagnostics/triton.json" \
  --flashinfer "$experiment_results/logprob-diagnostics/flashinfer.json" \
  --harness /reference-artifacts/quality_http.py \
  --output "$experiment_results/logprob-diagnostics/comparison.json"
