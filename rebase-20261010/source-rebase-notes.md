# Source rebase

- Preserved old head: d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6, local backup branch codex/gdn60403-pre-rebase-20261010.
- Frozen upstream: a98247ab4db686ee03c66d5feb3c761e52a2f8ab.
- Rebased commits: 84f5674, 97c3e80 (both range-diff identical to the original two patches).
- Compatibility commit: abf17c7, test-only Qwen3NextConfig import moved to Transformers.
- All eight PR paths pass applicable pre-commit hooks, including mypy and test tethering; Windows requires PYTHONUTF8=1.
- New upstream metadata adds two prefix_match_unit=None cases; focused total is 106 instead of 104.
- No production logic or numerical tolerances changed by this rebase update.
- Old serving and quality measurements remain bound to d8ae9e9. New validation is not a repeated performance measurement.
- The submitter previously confirmed review/testing of d8; that confirmation is not represented as human review of the new import-only follow-up.
- AI assistance used.
