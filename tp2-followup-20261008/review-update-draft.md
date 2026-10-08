Unchanged `d8ae9e9`, two-L20 TP2, 512/128-token ABBA ([evidence](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/tp2-followup-20261008)):

| C | Output tok/s T→FI (change) | TTFT p50 ms T→FI | TPOT p50 ms T→FI |
|---|---|---|---|
| 1 | 44.541→44.709 (+0.38%) | 172.635→172.731 | 21.264→21.177 |
| 8 | 228.492→229.451 (+0.42%) | 847.345→827.464 | 27.345→27.780 |

FI C8 TPOT is **1.59% slower**; paired signs differ. All **384 measured +64 warmups** meet the token protocol. Two launches per backend: descriptive results, no general speedup. Both NCCL ranks pass **12 rank/cases** at H8/HV24, including 128 changing-input steps.

Original GSM8K strict: **T 95/100, FI 96/100**; FI truncation/absent-marker **1/1**. Expanded strict: C1 **T 7/8 / FI 7/8**, C8 **T 7/8 / FI 6/8**; FI C8 truncations **1**. **FI C8 output/truncation differences remain unresolved.** [Detailed quality tables](quality-results.md) retain all negative outcomes. Integrity audits do not establish accuracy equivalence; FI remains opt-in, `auto` Triton. No TP2 microbenchmark; prior TP1 GPU timings keep their separate boundary. [Additional top-five observation](additional-logprob-results.md) uses shared prefixes through first divergence; margins are not causal proof.
