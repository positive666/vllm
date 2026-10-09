"""Activate the artifact-only observer in spawned workers."""
import os

if os.environ.get("SHORT_FORCE_ACTIVE") == "1":
    import divergence_bootstrap
