# PR #60403 rebase checklist

Source: abf17c7c071b1b6bdd81ca9756894998e5e8b32d; base: a98247ab4db686ee03c66d5feb3c761e52a2f8ab.

## Design fit
Original two patches are unchanged by rebase (range-diff '='). The only extra commit updates the regression test to upstream Transformers' Qwen3NextConfig after upstream removed the internal module. No production adapter redesign, default change, or model-runner hook is added. auto remains Triton; explicit FI stays opt-in. SM80 remains untested.

## Previous feedback
Fetched all conversation, review and inline-comment pages. No new reviewer feedback. Existing changes retain lazy pinned FI API, split validation errors, override log, conditional HV alignment staging and mixed-batch tail-only copy. Detached parameter views remain supported by actual grad-enabled DLPack evidence; bias FP32 specialization is not claimed to fix external cache keys.

## Testing and CI
All applicable pre-commit hooks passed with PYTHONUTF8=1, including mypy for the oldest supported Python and Buildkite test tethering. First GBK failures are environment errors retained separately. Focused 106/106 (40 adapter/mixed, 65 metadata, 1 config) and 12/12 TP2 rank/cases passed without skipped cases. All four model arms completed (32 completions), each 7/8; FI eager retains one q255 truncation. Raw audit passed; no accuracy-equivalence claim. See result-audit.json and validation-plan-v2.json. No new test file or synthetic benchmark added. Model evaluation preserves the existing eight-item fixture and scorer, parameters and tolerance gates. Native CUDA binaries are reused; this is not a full build of latest main.

## Code and PR contents
Diff remains eight files, plus only a test import compatibility change relative to the old patches. Existing invited takeover and duplicate-work references remain relevant; no new PR. Preserve AI attribution, human review scope, Draft status, old-head benchmark association, shared-host audit caveats and unresolved graph/natural-generation quality differences. Rebase validation does not establish a numerical fix or broader speedup.
