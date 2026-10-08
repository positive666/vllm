# Reviewer alignment on the unchanged d8 implementation

The earlier [reply](https://github.com/vllm-project/vllm/pull/60403#issuecomment-6055210576)
still describes these changes as pending at c3424cc. The reviewed d8 source
completes its five items; update that reply with completed actions and evidence.

1. Removed HV%8 rejection. Packed HV4 gates are cloned when needed for pointer
   alignment, including the B1 offset-view case. Existing synthetic small-head
   timings retain their unfavorable results; TP2 H8/HV24 is separately validated.
   CUDA eligibility is SM80+; BF16 inputs, FP32 state and K=V=128 remain the
   adapter scope, not a claim of universal FI requirements. SM80 was not run.
2. Removed signature/API probing; the lazy direct import uses the pinned FI API
   with explicit `backend="flashinfer"`. Configuration documentation is shorter.
3. FI-only FP32 bias is an adapter specialization choice, not a claim that FI
   rejects BF16 bias, that FP32 is optimal or that it fixes external shared caches.
4. Split configuration errors and added once-only logging for explicit FI
   overriding the disabled packed flag. `.detach()` remains because the default
   DLPack exporter rejects grad-enabled model Parameters; this is retained with
   evidence rather than silently claiming the reviewer-requested deletion.
5. Mixed batches copy only the prefill tail after state updates and preserve decode
   prefix, removing the redundant concatenation/copy.

The new evidence does not change these source decisions. It adds real TP2
execution, rank-local correctness, a bounded model-quality screen and full
serving measurements. Maintenance review/acceptance and official CI remain
separate; generated reports do not assert that either is complete.
The FI C8 expanded-budget diagnostic retains long-generation truncation and
changed target answers. That limitation remains unresolved; keep FI opt-in and
the default Triton path. Audit pass status is not an accuracy or ready-to-merge claim.

For an earlier reply that promised follow-up work, replace future-tense claims
with the completed evidence link and actual source head/results. Preserve the
limits and answer flips. Avoid another pending-work comment or claims that
short successful requests prove model accuracy.
