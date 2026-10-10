# TP2 serving replication plan

Source: d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6, unchanged.

Eight fresh processes use ABBA BAAB order on physical L20 GPUs 4/6.
Four paired replications/backend; pair identifiers are frozen in protocol.json.
Native HTTP API, TP2, FP8 model/BF16 execution, FP32 SSM state, Marlin, FA2,
CUDA graphs enabled, max_model_len=2048, max_num_seqs=32,
max_num_batched_tokens=1024, KV blocks=64 (21845 effective tokens), no prefix
caching. Per-backend old compilation caches are copied into private new caches.
Startup and warmup are excluded. No model runner instrumentation or supplied output-token sequence.

Each launch has C1/C8, 3 measured rounds of 16 requests each, plus 8 warmup
requests per concurrency. All use the same frozen 512-input-token fixture and
128 output tokens, min_tokens=128/ignore_eos=true/temperature=0/seed=42.
This measures fixed-workload serving, not answer quality or natural-EOS user
latency. It does not resolve the earlier differing answer lengths/quality.
There are 768 measured plus 128 warmup requests if every run completes.

TTFT: send to first nonempty text SSE event. Token-ID TTFT is retained too.
TPOT: first-to-last token-ID SSE span divided by 127; this is not pure ITL
when events contain multiple tokens. Request latency covers the complete HTTP
request through terminal usage/DONE. Throughput counts all successful output
tokens divided by the complete measured round wall time. Loopback HTTP, new
connection per request and closed-loop scheduling are included.

Compare the median of three round metrics per launch, then the median/range
of four launch medians and four predeclared paired percentage changes. Keep
failed/slow runs. No confidence interval/significance/general serving-gain claim.
C8 has only two request waves per round; it is a short closed-loop load, not
a prolonged saturation benchmark. Per-round p95 uses 16 requests and is
descriptive. Requests within one process are not independent launch samples.

Audit metrics independently from raw request times/counts; bind actual exit
records, logs, source, model metadata, runtimes, helpers, fixture and protocol.
Separately audit sampled process ownership/cleanup and clock/temperature
windows. Partial supervision stays visible. AI assistance was used.
