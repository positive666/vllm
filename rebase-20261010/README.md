# PR #60403 rebase validation

Tested source: abf17c7c071b1b6bdd81ca9756894998e5e8b32d on a98247ab4db686ee03c66d5feb3c761e52a2f8ab. The two original commits have identical range-diff patches; the extra commit only moves the regression test config import to Transformers. No production algorithm/tolerance changes.

## Results

- 106 focused tests pass, no skips: 40 adapter/mixed, 65 metadata, 1 config. Upstream added two metadata cases relative to the earlier 104-case run.
- 12 TP2 NCCL rank/cases pass, with indexed/padded state and 128-step changing-input CUDA Graph replay.
- All applicable pre-commit hooks pass, including mypy and Buildkite test tethering (Windows PYTHONUTF8=1).
- Four fresh model processes complete, 32/32 generation requests. L20 x2, TP2, Qwen3.8-27B-FP8 (Qwen3_5ForConditionalGeneration), unchanged eight-question diagnostic fixture and scorer, temperature 0, seed 42, 3500 output-token budget.

| Mode | Triton strict | FI strict | Triton / FI truncations |
|---|---:|---:|---:|
| Graph | 7/8 | 7/8 | 0 / 0 |
| Eager | 7/8 | 7/8 | 0 / 1 |

Graph equal totals hide different wrong answers: Triton q209=34800 (gold 145), FI q255=176 (gold 192). FI eager q255 still truncates at 3500 tokens without an answer marker; Triton reaches EOS at 1217 tokens and answers 176. This update does not fix the prior quality/trajectory issue. One launch per backend/mode cannot establish quality equivalence or remove previously observed graph variability. Draft and opt-in status remain appropriate.

## Evidence and reproduction

run_cells_v2.sh records the exact sequential commands: run_review_v2.py, torch.distributed.run with tp2_gdn_correctness.py, then long_free_driver.py for graph/eager x triton/flashinfer. All Python uses uv and the prepared virtualenv interpreter. Source hashes and archive SHA256 are in source-manifest.json; source is reproducible from the public PR commit. helpers/ includes the frozen drivers and scorer.

First focused attempt failed because the isolated offline HF cache was absent. The retry copied only the public Qwen3.5-0.8B cache and set HF_HOME; no assertions were relaxed. Original failure logs and cleanup are retained under results/, successful focused/TP2 logs under results/v2/, model records under results/<backend>-<mode>/. Other setup failures are in retained-setup-attempts.json.

The raw audit (audit_rebase.py, result-audit.json) verifies JUnit counts, log/helper/source hashes, runtime TP2/backend/graph state, token hashes/counts, frozen-scorer results and successful shutdown. GPU monitor: 263 samples, 374 matched compute rows, 0 unmatched. Sampling is not continuous ownership proof. Matrix/cleanup exit 0; both GPUs return to 1 MiB with no compute processes.

## Limits

Reused native cu129 binaries and existing pinned runtime; not a fresh whole-project build. Native upstream changes inspected (CPU, ROCm, HiSparse) are outside this selected CUDA Qwen path. Optional DeepEP remains unavailable. SM80/H20, MTP, DP+EP and broader quality remain unvalidated. No new performance measurement: earlier kernel/serving numbers belong to d8ae9e9, and its resource-audit caveats remain.

Public records sanitize host paths, private IPs, GPU UUIDs and container identifiers. raw-public-map.json records hashes before and after sanitization; raw audit ran before transformation. Public copies with changed bytes intentionally do not satisfy raw embedded hashes. Unredacted records are retained locally.

AI assistance used; the prior human review confirmation applied to d8. The import-only follow-up is separately visible in import-compatibility.patch.
