#!/usr/bin/env bash
set -euo pipefail
trap 'printf "%s\n" "$?" > /work/results/source-install.exit' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export UV_CACHE_DIR=/work/cache/uv
export PATH=/work/runtime/.venv/bin:$PATH
export UV_HTTP_TIMEOUT=120
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION='/work/cache/vllm-0.32.0rc1.dev274+gdd9e07a25-cp38-abi3-manylinux_2_28_x86_64.whl'
export SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0rc1.dev276+g3ddc6702a
export LD_LIBRARY_PATH=/work/compat13/usr/local/cuda-13.0/compat
cd /work/source/candidate
uv pip install --offline --python /work/runtime/.venv/bin/python -e . --no-deps --no-build-isolation
# Both full source trees use precisely the same compiled/generated artifacts.
# Copy only files absent from upstream, plus generated version metadata.
uv run --offline --no-project /work/runtime/.venv/bin/python /work/helpers/sync_native.py
for arm in baseline candidate; do
  PYTHONPATH=/work/source/$arm uv run --offline --no-project /work/runtime/.venv/bin/python \
    /work/helpers/probe_runtime.py > /work/results/$arm-runtime.json
done
uv pip freeze --python /work/runtime/.venv/bin/python > /work/results/installed-packages.txt
printf '%s\n' ready > /work/results/full-main.ready
