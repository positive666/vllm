# 最小验收协议

服务配置不是验收结果。以下使用独立合成请求，不需要业务正文或生产日志。
当前结果仍待写入 `results.json`，不能把以前纯 B8 的结果当作新增缓存层已通过。

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

具体执行命令、通过/失败/跳过数及模型结果由实际验收记录补入，不预填成功数字。
