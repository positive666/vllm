# PR #60403: native termination with a larger output budget

Production head: d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 (unchanged).
Qwen3.8-27B-FP8, two NVIDIA L20s, TP2, eager/synchronous, original eight
prompts, greedy sampling and original EOS behavior. Fresh Triton and FlashInfer
processes changed only max_tokens from 3500 to 8192 and max_model_len from
4096 to 16384. No trace replay, logits processors or forced output tokens.
Source, runtime/native-library and model-metadata bindings passed before runs.
The larger context also changes reported effective KV capacity (58254 to
87381 tokens); old output prefixes still match exactly. No serving timing
comparison is made from these diagnostic runs.

Both runs exited 0. Every prior output token was reproduced for all eight
requests in each backend. FI q255 reproduces its complete 3500-token prefix,
then reaches EOS 248046 at output token 7778, returning #### 176. Triton q255
remains exactly 1217 tokens, EOS 248046, #### 176. Other FI completions and
all Triton completions remain byte-for-byte identical in returned token IDs.

This identifies the earlier q255 cutoff as exhaustion of the configured output
budget on an otherwise terminating generation path. It is evidence against
ignored EOS for this reproduction; it does not identify which numerical
operation produced the earlier decode divergence. Increasing the budget is a
diagnostic, not a production-code fix or a performance improvement.

Both backends score 7/8 using the unchanged strict scorer. The q255 prompt says
both ten stalls and twenty stalls; its gold answer is 192. Both backend answers
remain wrong against that gold. Neither the prompt nor score was altered.
Earlier C8 graph differences remain unresolved; these eight cases do not
establish population quality equivalence or serving performance.

The independent CPU output audit passed. Resource audit: 166 snapshots,
303/303 compute observations owned, no unmatched/unknown/partial records;
cleanup exited 0 and both GPUs were released. Input SHA maps and actual exit
records are retained. Private IPv4 text is redacted with raw/public hash mapping.
Original audit input hashes describe raw files; public rechecks retain their
own input hashes. Full raw files remain local.

Commands are in prepare_and_launch.sh and run_cells.sh. CPU verification uses
uv with an explicit virtual-environment Python:

    uv run --offline --no-project <venv-python> -X utf8 audit/termination_audit.py --root . --output rechecked-output.json
    uv run --offline --no-project <venv-python> -X utf8 audit/resource_audit.py --results results --manifest resource-producer-manifest.json --expected-owner gdn60403-termination-20261010 --output rechecked-resources.json

AI assistance was used.
