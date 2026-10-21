# 最小验收协议

服务配置不是验收结果。以下使用独立合成请求，不需要业务正文或生产日志。
Flash TP8 的实际验收结果已写入 `results.json`；FP8 TP16 的 C1 结果单独记录在
[fp8-tp16-results.json](fp8-tp16-results.json)，并包含已完成的一波 C32 功能验收。
FP8 C32 的延迟未满足生产体验目标，不能将功能通过写成性能达标。
以下原始步骤中的 Mamba/640-token 几何属于 Flash；FP8 使用后述独立几何。

## 基础与缓存实际命中

1. 记录固定 image/config digest、源码/补丁哈希、模型配置和 tokenizer 文件哈希。
   检查模型 health，以及 LMCache `/healthcheck` 和 `/status`。
2. 使用当前 chat template 在服务 `/tokenize` 校准 1K、32K 合成输入。
   每条都检查 SSE `[DONE]`、usage、正确答案、TTFT-any 和首个可见正文时间。
   TTFT-any 包含思考输出；两种时间分别报告。
3. 请求完成并排空后，只清空 GPU 前缀缓存：

```bash
curl -fsS -X POST \
  'http://127.0.0.1:28001/reset_prefix_cache?reset_running_requests=false&reset_external=false'
```

   HTTP 200 还不够，返回 JSON 的 `success` 必须为 `true`。
   在异步传输仍持有源块时可短时间重试；不要把失败当作已驱逐。
4. 再次发送相同 32K prompt。逐请求要求：

- 请求完成计数仅增加 1，前后 running/waiting 为零；
- `vllm:prefix_cache_hits_total` 增量为零；
- `vllm:external_prefix_cache_hits_total` 增量大于零；
- LMCache HTTP `/metrics` 的
     `lmcache_mp_num_chunks_loaded_total` 增量大于零；
- 答案、SSE 和实际输入 token 数正确。

外部 lookup 命中但加载失败后重算，不满足上述加载要求。
记录实际复用 token 数；相同 prompt 再次请求不等于已经证明全部 32K full-hit。
主动清空 GPU APC 也不等于自然淘汰条件下的吞吐量测试。

## 前缀依赖与分支污染

仅返回算术答案 5 不能证明恢复出的前缀状态正确，补充一个短 passkey case：

1. A 的早段放随机合成 keyA，之后填充 inert words；末尾只问当前 authoritative
   key，问题中不能重复 key。按服务 tokenizer 校准到 32768 tokens。
2. 先冷请求 A，再清空 GPU APC、保留外部缓存并重复 A。两次都必须返回 keyA。
3. 在 16K 之后、32K 之前的填充区插入 `AUTHORITATIVE UPDATE`，
   令当前 key 变为不同的 keyB；保持原 header 和末尾问题不变，得到 B。
   记录服务实际 token IDs、LCP 和 SHA；B 可比 A 多至 128 tokens。
4. 再清 GPU APC，B 必须返回 keyB，GPU hits 为零，并实际从外部加载。
   本配置在请求前已核实 block size 为 640、Mamba 模式为 `align`，8 个 worker
   均协商为 640；调度预算为 16384、long threshold 为 0。因此本次分支预先
   确定的精确 checkpoint 为 16000 tokens，不能在看到结果后静默更改预期。
   几何证明的 SHA-256 记录在 `results.json`。
5. 再清 GPU APC、重新请求 A，必须回到 keyA；外部覆盖量还应与步骤 2 中
   A 的实际覆盖量一致。此反向重访用于捕捉 B 覆盖 A 的状态或生命周期问题。

每一步均保留完成加载计数。一次合成 case 通过，不代表广泛模型准确率、缓存服务
故障切换、完整 1M 上下文或 C32 体验已经验收。

## FP8 TP16 双节点 C1

使用同一镜像、两个节点各一个本地 MP 服务，TP16、MTP3、`max-num-seqs=32`。
每节点缓存配置为 256 GiB；合计 512 GiB 是物理分配量，不代表去重后的独立容量。
没有 Mooncake L2 或跨主机共享池。

本轮已按同一 A→A→B→A 顺序实测。实际 MLA-only、非混合状态几何为
64-token scheduler block、256-token chunk、Mamba 模式 `none`。A 为 32768 tokens，
B 为 32799 tokens，实际 LCP 为 24565 tokens；预先确定分支外部命中为 24320。
完整 A 的外部命中指标为 32767，不能把客户端指标改写为 32768。

每次完成 GPU-only reset 后，按单请求计数增量和实际 usage 检查 GPU APC 为零、
两端加载均完成；重复 A 和回访 A 要求每端 8 个实例各加载 128 个 chunk，
B 要求各加载 95 个 chunk。容器所属实例集合须与对应 MP 注册集合及完成加载集合一致。
物理 worker/rank 集合与缓存实例集合独立核对，不宣称逐物理 rank→实例的精确 join。

四阶段的答案、SSE、stop 和 usage 一致性均已通过。冷 32K TTFT-any 仍为
19.344 秒；不同生成长度的正文和端到端差异不能全部归于缓存。该 C1 结果不代替
FP8 C32 性能、完整 256K 上下文或故障恢复验收。

## FP8 TP16 单波 C32

在同一服务实例上，先单独执行一条 seed，再执行 24×1K + 8×32K 的 C32 波次，
输出预算为每请求 2048 tokens。每行使用独立的合成 passkey；显式预热仅覆盖一条
32K 长请求，其他 31 条使用新前缀。清空 GPU APC、保留外部缓存后分别检查 seed
与波次计数，不能把单请求的 success delta 门槛直接套到 32 条请求。

本次 32/32 的 HTTP、答案、SSE、stop、usage 和 fixture 校验通过，完成计数增量为 32；
GPU APC 为零，仅一条请求实际复用 32767 tokens。两端各 8 个实例完成 128 个 chunk
加载，KV 工作及 session/prefetch 当前计数归零，抢占为零。此有界排空不代表一般
生命周期、TTL 或故障恢复已通过。

波次 154.594 秒，TTFT-any P50/P95 为 13.204/128.30865 秒，首个正文为
147.477/152.75085 秒，最大输出间隔 4.578 秒。功能通过，生产体验不达标；不能
把长冷前缀与 decode 并行时的等待和停顿描述为已经解决。结果仅为单波描述性统计，
不作为稳定性能或生产 SLA。

## 源码回归

公开包校验器使用 `regex`，应在已安装候选镜像运行依赖的虚拟环境中执行，
不是零依赖的系统 Python 脚本。从本目录执行以下命令，按实际位置替换虚拟环境路径：

```bash
uv run --no-project --offline --python .venv/bin/python \
  .venv/bin/python -B validation/verify_bundle.py
```

在安装好候选依赖的虚拟环境内，运行现有相关测试；如基础设施不覆盖相应 GPU
路径，明确记录本地模型验证范围和 CI 缺口。不要用 mock 合约测试代替 CUDA IPC
远端完成、状态语义或推理数值验收。

vLLM 使用已有的核心测试和连接器边界测试，例如：

```bash
uv run --no-project --offline --no-sync --python .venv/bin/python \
  .venv/bin/python -m pytest -q \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
  tests/v1/core/test_prefix_caching.py
```

已完成的源码、Flash、FP8 C1 和 C32 结果按各自范围记录于公开摘要；新增阶段只在实际
执行后补入命令、通过/失败/跳过数及模型结果，不预填尚未通过的验收。
