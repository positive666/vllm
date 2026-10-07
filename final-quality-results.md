# Final matched Qwen FP8 model-quality screen

On the same 100 GSM8K questions, Triton scores 96/100 under the raw final-number
parser and FI scores 95/100. Both have 95 completed, correctly parsed answers
with a final marker. The raw difference is question 209: Triton's unfinished
output happens to end with the gold number, whereas FI produces a completed
wrong answer. Preserve both outcomes. Equal conservative counts do not show
accuracy equivalence.

| Backend | Raw parsed correct | Completed and correct | Final-marker, untruncated correct | Truncated | Unparsed | Missing marker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton A1 | 96/100 | 95/100 | 95/100 | 1 | 0 | 1 |
| FI B1 | 95/100 | 95/100 | 95/100 | 0 | 0 | 0 |

Question/gold pairs, question-fixture hash, full dataset hash, protocol and
client revision are matched. Both use temperature 0, seed 42, thinking
disabled, C8, a fixed 1,750-token budget and zero-shot prompts. A preliminary
Triton run at 1,024 tokens truncated four outputs. The budget was increased
to 1,750 before either final backend run; both final runs used that same
budget, with no subsequent prompt, example or budget changes. Preliminary
results are excluded from the final comparison. This is
one quality run per backend on a bounded subset, with no confidence interval
or statistical-equivalence claim.

Exact output-token sequences match on **63/100** questions; 37 differ. Only
one parsed final number changes. Raw correctness flips: **one Triton-to-FI
loss, zero gains**. Completed-answer and final-marker/untruncated correctness
flips: **zero**. Common incorrect question IDs are 93, 255, 1161 and 1309.
The count of matching final answers does not erase the token-output changes.

## Changed answer and all truncated outcomes

| Question | Gold | Triton A1 | FI B1 |
| ---: | ---: | --- | --- |
| 209 | 145 | 1,750 tokens, `length`, no final marker; fallback number 145, raw correct; incomplete | Completed answer 34800 with final marker; incorrect |

This is the only truncated question in either arm. Its output first diverges
at token index **480 (zero-based)**, where the reasoning begins to phrase the
problem's interpretation differently. The raw JSON preserves both full
outputs and token IDs; the summary retains token/text context around the
first difference. Do not count the truncated baseline as a completed validated
answer or relabel FI's completed wrong answer as correct.

## Evidence and runtime

The local folder is labeled `Qwen3.8-27B-FP8`; its config identifies
`Qwen3_5ForConditionalGeneration` / `qwen3_5`, used in text-only mode.
These results do not establish an official Qwen3.8 checkpoint identity.

Raw quality JSON:

- `final/quality-triton.json`, SHA256
  `6a75093d52a8e2aa9bbff105115c0c11011d121522388ce9c18be31e38d3f1c4`.
- `final/quality-flashinfer.json`, SHA256
  `7d913eeec8eb910274e112d8df356610dc48c02c176711c4ba92b91de1070218`.

`final/serving-summary.json` independently recomputes these outcomes and retains
the conservative counts and every truncation outcome. Its four-launch serving
comparison is complete and audited; see `final-serving-results.md` for the
descriptive performance results and limits.

All four completed records and startup logs verify equal final production hashes,
equal runtime and native-extension hashes, 64 KV blocks and **21,845 actual KV
tokens** in every process. Raw log hashes match their completed records. The
native extension is the identified reused CUDA12.9 binary; H20/SM90 and a native
build of current main remain outside this validation.
