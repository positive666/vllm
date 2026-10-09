The module isolates the unresolved long-generation diagnostic outcome across
eager/graph and Triton/FlashInfer execution on the same TP2 model and eight
selected prompts. Its input contract is the reviewed source manifest, frozen
question/gold/reference file, model configuration and tokenizer; its output is
actual unforced tokens/text, termination reasons, independent scores and native
graph dispatch evidence. It refuses existing output/cache directories and
preserves failures.

The failures guarded against are silently changing prompts/source/sampling,
counting supplied histories as accuracy, treating graph configuration as replay
proof, losing truncations, or publishing token/score evidence whose bytes no
longer match. The cheapest useful checks are standard-library CPU contracts for
strict scoring at length termination and rejection of changed token/score
evidence. Real graph activation and generation require the four GPU arms; the
audit must pass before publishing their results.

The offline synchronous all-eight request protocol is a controlled diagnostic
and differs from the retained original HTTP experiment. It does not replace
that experiment, estimate population accuracy, measure serving latency, or
identify a kernel root cause by itself.
