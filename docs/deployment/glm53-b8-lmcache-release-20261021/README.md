# GLM-5.3：固定 B8 基线与 LMCache 验证

此目录记录固定源码、依赖和补丁的实验候选镜像，已发布并通过独立匿名 registry 核验。
本轮没有据此替换生产服务；`production=false`。

```text
ghcr.io/positive666/model@sha256:cd85ebf238eb3c980d8e61ae7891b341446d5a3cd4d2e45f9a9b6e539083c172
```

配置使用上述 digest；版本 tag 是
`glm53-b8e9001bf956-lmc2a34fa8-v4-e8f751e2-20261021`。

下述部署示例适用于单机 8 张 H20 96 GB、Flash TP8、MTP3、本机 LMCache 256 GiB。
保留已完成的 Flash 验证；新增同一镜像的 GLM-5.3-FP8 双节点 TP16 缓存 C1 验证。
FP8 的 A→A→B→A 外部前缀恢复及一波 C32 功能验收已实际通过；并发延迟未达到生产体验目标。
`max-num-seqs=32` 是配置上限，不能据此称生产并发体验已经达标。

## 固定组成与来源

完整版本、提交和补丁哈希见 [versions.json](versions.json)。

| 部件 | 固定版本或提交 |
| --- | --- |
| vLLM | `b8e9001bf9568c801985889deff34d3b9fe6d0b0` |
| LMCache | `2a34fa829b3c902c5b6393740239f6ff65bd0ff8` |
| Torch / Triton | `2.14.0+cu130` / `3.8.0` |
| CUDA 基座 / nvcc、ptxas | `13.0.3` / `13.0.88` |
| FlashInfer / DeepGEMM | `0.7.2` / `2.8.1` |
| Rust 构建工具链 | `1.95.0` |

11 个 vLLM 原生库来自 `e10ecb6bd65b94e7f281bd3f9d85fd440d7dd72b`
的源码编译产物。已比较 401 项原生源码、构建和编译输入，确认与固定 B8 的
字节及文件模式相同，因此逐字节沿用这些库。B8 的完整 Python wheel 和变更后的
Rust 重新构建。**这不是重新编译全部 B8 CUDA 目标的镜像，也不是未修改的官方镜像。**

后续公共依赖层和 LMCache 四个原生模块另行构建、检查。公共 34 项哈希锁是对
已验证基座的增量，不能单独重建完整环境，也不包含模型权重、环境快照或凭据。

## 补丁与上游工作

| 文件 | 用途 | 应用顺序 |
| --- | --- | --- |
| `0001-vllm-mamba-prefill-handoff.patch` | GPU 稀疏保留不应抑制有效完整 prefill 块的外部状态交接 | 对固定 B8 |
| `0002-lmcache-exact-boundary.patch` | 精确状态边界、独立交接任务及 worker 完成前的源块/事件保留 | 对固定 LMCache |
| `0003-lmcache-ordinary-store-lifetime.patch` | 保留每个普通 store 的 future/event；未知完成状态时失败关闭 | 在补丁 2 之后 |

vLLM 修复对应既有 [#62479](https://github.com/vllm-project/vllm/pull/62479)；
本轮更新待提交者人工审阅后推送。
LMCache [#4844](https://github.com/LMCache/LMCache/pull/4844)、
[#4836](https://github.com/LMCache/LMCache/pull/4836) 和
[#4888](https://github.com/LMCache/LMCache/issues/4888) 已跟踪相关所有权和完成语义。
所检查的 #4836 提交尚未封闭普通 store 的 heartbeat 失联路径，补丁 3 保留此本地增量。
此包不另开重复 PR，也不把尚未合入的工作称为上游已包含。

补丁 3 在已提交 store 但无法确认完成、或失去服务健康状态时停止引擎，防止提前释放
源块和事件。它没有实现 LMCache 服务热重启的透明恢复或完整故障切换。

从各自固定源码根目录执行 `git apply --check`，再按上述顺序 `git apply`。
安装使用 `uv` 和显式虚拟环境；不要把公共增量锁直接安装进主机系统 Python。
原生模块必须与固定 Torch/CUDA ABI 一起构建，不能复用旧 Torch 版本的 LMCache wheel。

## 最小部署

两个容器使用同一个最终验证镜像，分别运行 vLLM 与 LMCache。它们通过本机 CUDA IPC
读写 GPU KV 和 pinned DRAM；ZMQ 控制连接也只绑定 loopback。
此方案没有 Mooncake L2，没有跨节点共享池，不需要为本机缓存额外开放 RDMA/GDR 设备。

1. 复制 `deployment/image.env.example` 为 `image.env`，保留已核验的镜像 digest，
   填写本地 Flash 权重绝对路径。
2. 确认还有额外 256 GiB 可锁页内存及服务开销；参考验收门槛为
   `MemAvailable >= 320 GiB`。已有缓存池也计入当前用量，不能当作可用内存。
3. 使用 Docker Compose 的 NVIDIA GPU reservation 支持启动：

```bash
cd deployment
docker compose --env-file image.env config
docker compose --env-file image.env up -d
curl -fsS http://127.0.0.1:28056/healthcheck
curl -fsS http://127.0.0.1:28056/status
curl -fsS http://127.0.0.1:28001/health
```

`/healthcheck` 应返回 `{"status":"healthy"}`；`/status` 的顶层
`is_healthy` 还应为 `true`。模型注册后记录协商出的实际缓存 chunk/group
几何。配置的 `--chunk-size 256` 是最小值，不能把它当作最后使用的物理块跨度。

256 GiB 是上述单机示例的新 LMCache 池容量，使用 eager pinned DRAM。
双节点 FP8 实验配置为每节点一个 256 GiB 本地 MP 池，合计物理容量 512 GiB。
这不等于去重后的有效容量 512 GiB，也不是两个实例共享的跨节点缓存池。
上述 Flash 单机示例的上下文上限为 1M；FP8 双节点本轮实际配置为 256K
（262144 tokens）。短请求与 32K 验收不等于已验证相应完整上下文负载。

默认模型 API 只监听本机。`VLLM_SERVER_DEV_MODE=1` 用于下述实验中的 GPU
前缀缓存清空；把此实验配置接到其他入口之前应另行制定服务管理和访问配置。

## 验证状态

按 [验证协议](validation/README.md) 验证实际外部加载、分支和反向重访。
公开结果保存在 [validation/results.json](validation/results.json)。
相同 prompt 再次请求只能按实际计数称为“外部前缀复用”；没有证明全部 32K full-hit。

当前候选镜像已通过 8 GPU 可见性、四个 LMCache 原生模块导入、11 个 vLLM 原生库
字节一致性及 1 MiB 主机/设备往返检查。Linux 已安装源码回归共 577 项通过，
0 失败、0 跳过，详情见 [CPU 回归聚合](validation/cpu-regression.json)。

### Flash TP8 历史验证

单请求合成验证中，32K 冷前缀、相同 prompt 复用和分支请求均返回正确答案：

| C1 阶段 | TTFT-any 秒 | 外部复用 tokens | 已完成加载 chunks |
| --- | --- | --- | --- |
| 32K 冷前缀 | 7.632299 | 0 | 0 |
| 相同 32K prompt | 0.444183 | 32000 | 408 |
| 分支 | 1.432924 | 16000 | 208 |

这些请求的 GPU APC 命中均为零。前缀依赖的 passkey A→A→B→A 验证也全部正确，
分支后反向重访恢复 keyA；外部复用量依次为 0、32000、16000、32000。
这是单次观察，预热和 JIT 状态可能不同，不能当作稳定性能估计；TTFT-any 包含思考。

另完成一波 C32（24×1K + 8×32K，保留外部缓存）：32/32 的答案、SSE、
usage 和 fixture 哈希正确，GPU APC 命中为零。实际仅一条长请求复用 32000 tokens，
另外 31 条的 cached usage 均为零。
TTFT-any 的 P50/P95 为 13.791197/18.223267 秒，首个可见正文为
21.570524/21.753740 秒；队列最大 29、运行最大 27、抢占为零。
全局 device-stream stores 提交/完成为 536/536，retrieves 为 8/8，传输在途为零。
这不等于逐请求生命周期或故障注入已通过，也不代表生产 SLA。

首次晚期排空连续 12 次观察到一条 session 元数据，其余锁、临时缓冲、任务和传输在途
均为零。默认 TTL 清理线程随后自动删除该元数据；保留原门槛的复查连续两次全部归零，
GPU/服务健康检查通过，有效缓存保留。原失败记录仍保留，元数据的来源和触发路径
尚未追踪，不能归因为 C32，也不能据此称即时 END 清理或故障恢复已全面验证。
镜像上传成功；匿名读取 tag 与 digest 得到相同 manifest 原始字节，config digest、
`linux/amd64` 及 44 个有序 rootfs diffIDs 与本地逐项一致（39 个不同 blob）。
见 [registry 核验摘要](validation/registry-verification.json)。这不等于生产上线验收。

### FP8 TP16 双节点验证

同一已发布镜像在 2×8 张 H20 96 GB 上运行单个 TP16 实例、MTP3、配置上限 C32。
FP8 本轮实际配置为 `max-model-len=262144`、`max-num-batched-tokens=8192`、
`long-prefill-token-threshold=2048`、`max-num-seqs=32`，开启 EP。
KV 的 CLI 参数为 `--kv-cache-dtype fp8`；后端可能规范化为 `fp8_ds_mla`，
这里不将 CLI 值当作服务报告的最终 dtype。
每个节点使用自己的 LMCache MP 服务，经本机 CUDA IPC 恢复该节点的 GPU KV；
本轮没有接入 Mooncake L2 或跨节点共享缓存池。实际结果见
[FP8 验收摘要](validation/fp8-tp16-results.json)。

一次前缀依赖的合成 A→A→B→A 验证全部通过，四阶段的 HTTP 200、SSE `[DONE]`、
正确 passkey、stop、usage 一致性均通过；每次只清空 GPU APC，外部缓存保留。

| FP8 C1 阶段 | 输入 tokens | TTFT-any 秒 | 首个正文秒 | 外部复用 tokens | 每端已完成加载 chunks |
| --- | --- | --- | --- | --- | --- |
| 冷 A | 32768 | 19.344 | 23.984 | 0 | 0 |
| 重复 A | 32768 | 0.312 | 6.718 | 32767 | 1024 |
| 分支 B | 32799 | 5.766 | 6.703 | 24320 | 760 |
| 回访 A | 32768 | 0.562 | 5.859 | 32767 | 1024 |

分支的实际公共前缀为 24565 tokens，按已核实的 256-token chunk 对齐后复用 24320。
三次恢复中，两端各 8 个已注册实例分别完成 128、95、128 个 chunk 加载，
GPU APC 命中始终为零。worker/rank 集合与缓存注册集合分别核对，并验证容器对应的
实例集合一致；没有证明逐物理 rank 与缓存实例的一一对应。

这些是单次功能结果。冷 32K 的 TTFT-any 仍为 19.344 秒，不能称所有性能问题已解决。
TTFT-any 包含思考；四阶段的生成长度不同，不能把正文或端到端时间差异全部归因于缓存。
另完成一波 FP8 C32：24×1K + 8×32K，单独冷 seed 后仅一条长请求显式预热。
32/32 的答案、SSE、stop、usage 和 fixture 均正确，总生成 11214 tokens；
GPU APC 为零，外部复用 32767 tokens，每端 8 个实例各完成 128 个 chunk 加载。
两端当前 KV 工作、锁、临时缓冲、prefetch 任务及 active sessions 均排空，抢占为零。

| FP8 C32 输入组 | TTFT-any P50 / P95 秒 | 首个正文 P50 / P95 秒 |
| --- | --- | --- |
| 全部 32 条 | 13.204 / 128.30865 | 147.477 / 152.75085 |
| 24 条短输入 | 9.219 / 25.25215 | 146.2035 / 151.8353 |
| 8 条长输入 | 90.8675 / 133.68655 | 148.8985 / 153.85345 |

唯一命中缓存的长请求（index 100）虽复用 32767 tokens，TTFT-any 仍为 65.188 秒、
首个正文 151.079 秒，生成 434 tokens，其中思考 422 tokens。不能将 C1 热请求的
0.312 秒外推到 C32；混合并发中的竞争仍可淹没缓存带来的首输出收益。

波次耗时 154.594 秒，观察到队列最大 21、运行最大 32，最大输出间隔 4.578 秒。
**功能通过，但这些延迟不满足生产体验目标。** 冷长输入与持续生成并行时的等待和停顿
仍突出；本波次没有 KV 抢占，不能将其描述为已经解决所有性能问题。
这是单次混合波次，不是隔离归因实验；Flash 与 FP8 的生成长度和负载不同，不能直接
当作受控性能对比。完整 256K 上下文、一般生命周期/TTL 及故障恢复仍未验收。
服务运行时源码、镜像和补丁位未改变，此次仅补充验证材料，无需重新上传镜像。

代码和测试准备使用了 AI 辅助。上游 PR 更新仍需提交者逐行理解、审查并运行相关测试。
