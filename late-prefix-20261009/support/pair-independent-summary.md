只读复核完成。总体状态保持 **REVIEW_REQUIRED**，`strict_same_sample_ownership_pass=false`，没有改为总 PASS。

- 两个实际 native arm 的下载 metadata SHA 与审计输入绑定一致；逐 token 重算实际返回历史，均与锁定的 FI eager1 参考一致。每个 arm 为 7,378 个位置，其中 7,370 个真实 decode；两个 TP rank 各记录完整 7,378 个物理位置，执行 3,500 个 batch。
- 原始 d8 source、冻结 helper、模型和 runtime/native 文件绑定一致。实际配置为同两张 L20、TP2/NCCL 2.29.7、同步 eager、prefix cache 关闭、FP32 recurrent state、BF16、Marlin/FA2、seed42；没有 CUDA graph 或 sampling sharding。
- 全部八个 full-vocabulary prefill logit SHA 一致；实际调度、execution context 和首个 native choice 一致。第一步 48 GDN 层 × 两个 rank × 八个请求的 **768 对 active prestate 精确相等**。冻结远端 CPU consumer 已审计每个 arm 的 672 个 snapshot，共 1,344 个；本地没有重新下载并读取这些二进制。
- 固定 FI 历史下，backend native choice 共出现 **15 个差异位置**：控制题 206 为 1 个（首次 step4），209 为 5 个（首次 step454），255 为 9 个（首次 step19）。FI 的 native choice 与全部参考 token 相同；这是有条件的观察，不能当作准确率或自然停止验证。
- 原始 PID telemetry 重算得到 228 次 GPU compute PID 观察，227 次同采样确认，0 个未知 PID，1 次未同采样确认。F 在 12:11:36Z 的 GPU7 PID206320、550MiB，在前一个 12:11:30Z owned-container 样本中存在；当前样本没有该 PID。保留“可能退出采样竞态”，不能据此补成同一瞬间的所有权证明。

**READY_for_bounded_local_replay** 仅表示这批冻结捕获输入可以继续做 kernel 原样 replay 和共同 prestate 的数学检查。它不是整体通过、模型质量通过、性能收益或 decode 因果结论。full-attention KV 没有被观察；单个 ownership 未确认项仍需明确保留。

详细数值、实际 metadata SHA、原始所有权重算和限制见 [pair-independent-review.json](pair-independent-review.json)。没有 GPU 执行、生产代码改动、helper 修改或发布。
