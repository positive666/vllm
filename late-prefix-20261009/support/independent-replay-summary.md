两路各 672 个实际捕获样本的独立报告审阅通过。Triton 与 FlashInfer 自身生产输出和全部 compact captured poststate 均逐位复现，1,344/1,344；两路合计 2,688 个后端 invariant record 均通过。source/runtime/helper/reference/capture/exit receipt 绑定完整。这里审阅全部 JSON 指标和绑定，未在本地重新读取 raw PT 或执行 GPU 数学计算。

| 捕获来源 | 自身生产逐位复现 | 同 prestate 的 T/F output 最大相对 L2 | 同 prestate 的 T/F state 最大相对 L2 |
| --- | ---: | ---: | ---: |
| Triton（实际 BF16 bias） | 672/672 | 1.16839e-4 | 4.26638e-7 |
| FlashInfer（实际 FP32 bias） | 672/672 | 2.86844e-4 | 4.31375e-7 |

全部单次更新均满足固定 FP64 参考界限：allclose(atol=0.01, rtol=0.01)，并且全张量相对 L2 严格小于 1%。T 捕获的 output 相对 FP64 数学值最大相对 L2 为 0.30543%，F 捕获为 0.27844%。数学参考包含 BF16 输出舍入的影响；最大绝对误差约 0.0152 不单独违反同时带 rtol 的 allclose。

同历史 native 配对中的传播漂移是另一个指标。初始 8 份全词表 prefill logits 与 768 份 first-decode active GDN prestates 精确相同。两 rank 的 step1/layer0 QKV/a/b/A_log 和按 FP32 比较的 bias 数值也都相同；最早不等 input 出现在 step1/layer1。所有 672 对 A_log 和 bias 的数值始终相同，但实际 bias dtype 特化不同，不能只归因于归约顺序。

独立传播 prestate 最大相对 L2 为 1.65213%，poststate 为 1.65483%（rank1/layer60/step1750/q255）；QKV 最大相对 L2 为 3.09188%，生产层输出为 3.52642%（rank1/layer49/step1750）。这些 drift 数字不属于固定 prestate 的单次更新误差，也不能直接使用局部 replay 的 1% 阈值判模型质量失败或通过。

实际 incoming A_log 与 dt_bias 的 requires_grad 在每路 672/672 个样本中均为 true。该证据支持此次模型路径的 detach 输入契约说明，不能推广为所有模型均会失败。

覆盖限于 eager、非 speculative、pure decode 的 batch 8/1。未执行 padding 或 masked/null request；只验证 sentinel slot0、采样的 unused page 和 synthetic storage gaps 不被修改。未观察 full-attention KV，局部 replay 是单设备 rank-local 更新，没有测准确率、自然 EOS 或性能。

结论：已覆盖的 kernel 单次更新正确且可忠实复现；native 传播存在差异，前述自由生成截断/答案变化仍需独立评价。保留 opt-in、Draft 和 outer REVIEW_REQUIRED（native host strict ownership 有一个 possible exit race），本结果不能为模型质量等价或收益因果背书。

机器可读全量审阅与每项 max 的 rank/layer/step 位置见 independent-replay-review.json。
