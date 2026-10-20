#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export UV_PYTHON_DOWNLOADS=never PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
if test ! -e /work/.venv; then
  uv venv --offline --system-site-packages --python /usr/bin/python3.12 /work/.venv
fi
task_python=/work/.venv/bin/python
test -x "$task_python"
case "$LMC_MODE" in
  server)
    exec uv run --no-project --offline --no-sync --python "$task_python" \
      "$task_python" -B -m lmcache.cli.main server "$@"
    ;;
  model)
    exec uv run --no-project --offline --no-sync --python "$task_python" \
      "$task_python" -B -m vllm.entrypoints.cli.main serve "$@" \
      --kv-transfer-config "$(cat /config/kv-transfer-config.json)"
    ;;
  *) exit 64 ;;
esac
