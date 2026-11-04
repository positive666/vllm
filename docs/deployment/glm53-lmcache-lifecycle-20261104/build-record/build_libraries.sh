#!/usr/bin/env bash
set -Eeuo pipefail
cd /build
mkdir -p results source/lmcache wheels cache/uv
trap 'code=$?; printf "%s\n" "$code" > /build/results/libraries.exit' EXIT
uv=/opt/glm53-image-gn/bin/uv
py=/opt/glm53-image-gn/.venv/bin/python
export MAX_JOBS=8 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0 CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/build/cache/uv
"$uv" pip install --python "$py" --no-deps --offline /build/build-deps/*.whl
tar -xf lmcache-source.tar -C source/lmcache
cp context/release-pyproject.toml source/lmcache/pyproject.toml
cp context/release-version.py source/lmcache/lmcache/_version.py
"$uv" run --no-project --offline --no-sync --python "$py" "$py" -B /build/context/build_lmcache_native.py > results/lmcache-build.log 2>&1
date -u +%FT%TZ > results/LMCACHE_BUILD_COMPLETE
