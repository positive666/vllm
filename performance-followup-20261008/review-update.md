Added exact-head L20 performance measurements for `d8ae9e9` ([raw data, commands and independent audits](https://github.com/positive666/vllm/tree/codex/gdn-flashinfer-evidence-20261007/performance-followup-20261008)).

- GDNDecode + RMSNorm GPU latency: warm-state B1/B8 **-23.87%/-14.44%**; rotated-state B1 **-14.16%**, B8 **+0.94%**. Synthetic HV4 shows no benefit. The actual timer is CUDA graph events (CUPTI fallback).
- Serving, 512/128 tokens, C1/C8, ABBA: **+0.23%/+0.26% throughput**; C8 TPOT and C1 TTFT remain inconclusive. All **384 measured +64 warmup** requests succeeded.

These are small descriptive serving changes from two independent launches per backend, not a statistically established or general serving speedup. The PR description now separates micro and end-to-end results.
