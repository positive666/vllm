# PR #60403: bounded decode-divergence diagnostic

Unchanged production head `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6` ran
Qwen3.8-27B-FP8 on two L20s, TP2, eager/synchronous. Fresh processes used the
same eight prompts: A native Triton; B FlashInfer initialization/FP32 bias with
unmodified native Triton decode; C native FlashInfer. All GPU arms exited 0.
The independent resource audit passed: 31/31 compute observations matched,
zero unmatched/unknown samples, owned container stopped.

Native replay fixed outputs 0–18; output 19 remained natural, followed by public
abort. Original max_tokens=3500, min_tokens=0 and EOS behavior were retained
using a diagnostic hook bypassing private trace-termination normalization only
for these eight requests, after ordinary generation/tokenizer updates.
Effective EOS IDs: 248044/248046. This intervention is not an accuracy evaluation.

The CPU audit passed exact observed-start checks: 1,792 prompt-KV/conv/recurrent
records and 16 full-vocabulary prefill hashes per arm, matching schedules,
912 decode calls per rank. A/B had zero numerical drift across all 288 captures
and exact complete logits at steps 18/19. Bias values matched after FP32
conversion despite different stored dtypes. For q255's first divergence,
A/B chose 71072 (` stalls`), C chose 43659 (` cows`): candidate margin 0.125 → 0.

This supports a bounded decode-implementation contribution, without identifying
a particular operation/integration bug, the earlier 3,500-token termination
cause, model quality equivalence or speedup. Later tensors propagated separately;
other requests also had fixed histories. Later KV/whole hidden states were not
compared.

Preserved failures: initial native normalization changed budget/EOS; consumer v1
expected an absent launch key; the first CPU load probe failed before constructor
execution. Main math/resource exits are 0, while that combined CPU attempt's
orchestration exit remains 1. A separate six-case CPU probe exited 0: real
constructor/sharded-loader paths preserved BF16 fixture values exactly, with
no CUDA initialization. Its synthetic nonrepresentable FP32 fixture differed
by at most 0.00150001049; this does not explain the actual BF16-value divergence.

Included: reports, frozen sources/helpers/consumers, CPU receipts, all actual
metadata/sidecars, all 32 three-arm logit cases (96 vectors), and three snapshots
limited to rank0/layer0/step19.
Only private IPv4 text is redacted, with raw/public SHA maps. The approximately
2.8-GiB full snapshot set is omitted; complete tensor-audit reproduction needs
original or regenerated snapshots. AI assistance was used.
