q255 两次 eager 自然生成均在 **0-based index19，即第20个生成token** 首次分叉：前19个 actual IDs 完全相同，T选71072，FI选43659。分叉位置与 locked-FI-history 的 native step19 一致。实际文本标题分别走向 “initial number of stalls and cows” 与 “initial number of cows”；单token词表解码等待正确模型 tokenizer，未猜词。

题面同时写 “Ten stalls” 和 “twenty stalls”。冻结gold192按10栏计算；按20栏且每栏原有20头则是176。T eager两次全文均1217tokens，stop并给####176；FI两次全文均3500tokens，length且无####，不断重审这两种解释并在句中截断。FI原parser取出的10不是完整答案。

FI最常见8/16/32/64-token n-gram分别出现7/3/2/1次；检查周期1..128时，最长相邻精确重复只有2个相同token，末尾无精确周期段。记录显示语义层面的反复推理，未显示简单固定token循环。这不能据此证明state/writeback错误，也不能排除隐藏状态或写回问题。graph1两方先共享400tokens后分叉且都答176；graph2两方完整359tokens一致并答192。

最小下一步是两个fresh TP2 eager的**自然20-token窗口**，使用原8 prompts、真实sampler且不加载/注入forced histories、不替换choice。先核对实际20IDs与原eager前缀，再在q255 steps18/19记录真实GPU history/inputposition、全局request/row/execution join、两候选raw/post-logit与argmax margin，并以实际GDN pageindices关联compact SSM/conv pre/poststate。任何prefix或schedule偏差保留为结果；20-token窗口不验证EOS、长输出质量、性能或完整因果链。

本review只读取既有证据；未执行GPU、修改冻结记录或发送GitHub消息。JSON保留全部8run SHA、actual ID前缀、结束/重复指标及最小方案。
