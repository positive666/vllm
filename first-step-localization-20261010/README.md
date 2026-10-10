# PR #60403: first-decode numerical localization

AI-assisted diagnostic of unchanged source `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`, Qwen3.8-27B-FP8, L20. This records new rank-local replay and CPU analysis of the [previous three-arm TP2 eager captures](https://github.com/positive666/vllm/tree/db4f2c60e09db38650a78880d008069c8d27ee16/divergence-20261010), not a fresh natural-generation or serving benchmark.

## Findings

- First decode step (output index1), layer0: both ranks have identical target inputs/prestate and bit-identical BF16 target outputs between arms B/C. Their FP32 poststates differ by at most 1.9073486328125e-6, relative L2 about 3.05–3.07e-8.
- Scanning all48 GDN layers × two ranks at that step identifies the earliest target output difference at **zero-based layer1, rank1**. Target QKV, gates, parameters and prestate still match exactly. Two of3072 BF16 outputs differ; maximum absolute difference1.1920928955078125e-7, relative L2 2.719961170594186e-6. Poststate maximum difference7.450580596923828e-8.
- At head7/value8, the FP64 result is -2.9265877743230805e-5, extremely close to a BF16 rounding midpoint. FI equals the nearest BF16 reference value; Triton selects the adjacent value. Thus a different FI output is not itself evidence of a wrong FI answer.
- At head1/value57, the result is near zero: Triton2.864908310584724e-11, FI3.0468072509393096e-11, high-precision reference2.7318277518867224e-11. This difference is not just one BF16 ULP. Triton is closer to the mathematical reference at this element. Neither backend is uniformly more accurate.
- Independent 80-digit Decimal arithmetic confirms the two FP64 values to absolute error below2.1e-21.
- The next GDN layer (layer2) already has differing target QKV/gates in both captured ranks. This is consistent with propagation of small arithmetic differences, not proof that these two elements alone cause the later token/answer divergence.

## Executed validation

Three selected rank/layer captures × two backends × B8/B1 × three repetitions = **36 independent reset updates**, all completed. Every replay matched its own captured target output and poststate exactly, including the B1 ablation. Null and sampled unused synthetic pages were unchanged. All output/state comparisons passed the existing atol0.01, rtol0.01 and relative-L2<0.01 criteria; tolerances were not loosened. All96 layer/rank scan records and negatives are retained.

Both GPU jobs and cleanup exited0. Five-second sequential host samples observed no compute row during the short first job and one matched compute row in the second; no unmatched row was observed. These sparse samples are not continuous ownership proof. Final GPU4 observation was1MiB,0% utilization,no compute PID. Other GPUs were untouched. The known optional DeepEP import traceback was caught; it is retained in logs and did not prevent this non-EP replay.

CPU command (executed with original local folder names):

`uv run --offline --no-project <venv-python> audit_localization.py --root <rounding-localization-v2-20261010> --v1 <rounding-localization-20261010>`

For this published layout, use `--root first-output --v1 layer0` after placing copies of the top-level selected-captures archive beside first-output/results. The script writes a new cpu-audit.json and refuses overwriting it. It uses CPU torch and Decimal; no GPU/model weights are needed.

## Files and reproduction limits

The frozen GPU producers, container setup, actual run commands/logs, source hashes, full reports and independent CPU audit are retained. GPU replay used `uv run --offline --no-project /cache/gdn-runtime/bin/python /rounding/localize.py` with source/captures mounted read-only. Containers reused the pinned runtime/native libraries; no fresh CUDA build.

selected-captures.tar.gz contains all six original tensor binaries/sidecars used in the36 replays. Its SHA256 is64d98642b2674031d0cf1aa775cee1afb89be8f14a237d7eb95cbe4bfd4821f4. The complete96-case scan needs the original192 tensor binaries; only its reports/hash bindings and the selected six tensors are published here. Published GPU scripts retain their original mount paths and the second script performs that full scan before selecting the replay. CPU verification of the selected results is self-contained.

Only the target recurrent page was captured. Replays preserve original batch inputs/strides, but synthesize other active pages as zero and use a contiguous pool; original full-pool stride, absolute addresses and aliases are not reconstructed. Exact target reproduction bounds this limitation; it does not prove whole-pool or TP-collective behavior. B1 is an explicit batch-shape ablation.

These results support a small arithmetic divergence at the first differing GDN output. They do not identify a particular intrinsic/reduction/FMA as the cause, establish quality equivalence, explain graph-mode q209, or close the q255 long-trajectory issue. No production fix is justified by this result alone. FI stays opt-in, auto stays Triton, PR stays Draft pending reviewer judgment on the remaining quality scope.
