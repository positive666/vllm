# #60403 CI status at c3424cc (2026-10-08)

The PR head was updated to `c3424cc55d5258a3d5f104d817d6c6ef850db67f`, based
on main `8352b2427f704652da1910f0d53f3fdcc2fa2b0d`. The eight-file patch was
cherry-picked without conflicts or semantic edits. Local pre-commit and the
manual mypy-3.12 check passed; the source worktree remained clean.

[GitHub pre-run-check](https://github.com/vllm-project/vllm/actions/runs/37728203515/job/113151080408)
failed the contributor gate: the author has three merged PRs, below the four-PR
threshold, and no CI authorization label. Actual pre-commit was skipped. This
gate failure is separate from source/test failures. No label request or bypass
was performed.

GitHub reports the [ReadTheDocs check](https://app.readthedocs.org/projects/vllm/builds/35008083/)
as SUCCESS, but the official build API records **cancelled, success=false**.
Checkout of c3424cc succeeded; `bash docs/pre_run_check.sh` exited **183**.
Dependency installation and the documentation build did not run. The official
v2/v3 API snapshots are retained beside this report.

DCO passed. This does not claim upstream kernel CI, documentation compilation,
or a full native CUDA13 build passed.
