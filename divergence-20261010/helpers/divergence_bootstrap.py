"""Install one combined deferred hook in each diagnostic process."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

_INSTALLED = False
_CONFIG_SHA = None
_MODULES = {
    "vllm.v1.worker.gpu.model_runner",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.model_executor.layers.attention.attention",
}
_DONE = set()


def _patch(module):
    if module.__name__ in _DONE:
        raise RuntimeError("Combined diagnostic module hook executed twice")
    import short_force
    import divergence_capture
    short_force.patch(module)
    divergence_capture.patch(module)
    _DONE.add(module.__name__)


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        method = getattr(self.original, "create_module", None)
        return method(spec) if method is not None else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise RuntimeError("Target module has no normal concrete source loader")
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    """Allow sitecustomize and explicit driver calls without duplicate patches."""
    global _INSTALLED, _CONFIG_SHA
    if os.environ.get("SHORT_FORCE_ACTIVE") != "1":
        raise RuntimeError("Short diagnostic activation is absent")
    config = Path(os.environ["SHORT_FORCE_CONFIG"])
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    if _INSTALLED:
        if config_sha != _CONFIG_SHA:
            raise RuntimeError("Diagnostic config changed after hook installation")
        return
    import short_force
    short_force.install()
    # Fail closed when an import already constructed model operators.
    if any(name in sys.modules for name in _MODULES):
        raise RuntimeError("Install the diagnostic before importing model modules")
    sys.meta_path.insert(0, _Finder())
    _CONFIG_SHA, _INSTALLED = config_sha, True


if os.environ.get("SHORT_FORCE_ACTIVE") == "1":
    install()
