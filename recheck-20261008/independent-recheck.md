# #60403 本轮独立核验

在本轮固定 main `8352b242` 上干净接入原八文件补丁，核验的源码为 `c3424cc55d5258a3d5f104d817d6c6ef850db67f`。对归档实际字节、rebase manifest、运行时三项生产源码SHA和补充构建源码SHA交叉核对一致；原补丁新增CustomOp的语义保持不变。

- 核心suite逐case计数为 **31 adapter/mixed +63 metadata +1 config =95通过**，各exit=0、无失败/跳过。
- 保留原stable二进制时，MTP诊断仍是 **7通过/7跳过/2失败，exit=1**；runner的两个overall字段仍为false，没有抹去或计为全通过。
- 独立加载官方源码的补充native库后，逐setup/call/teardown核对 **16模型路径+22直接kernel=38通过**，无失败/跳过。SiLU、sigmoid的两项pure-MTP均通过；未改测试源码，其existing `wraps=fused_op`和`call_count==1`断言保证实际native选路，fallback案例则要求0调用。launcher不替换Python op或放宽容差。
- 补充库实际使用 **NVCC12.9.86、SM89、CUDA12.9 Torch**，只构建官方MTP组件和最小注册片段；不是CUDA13编译，也没有覆盖默认main CUDA13 CMake完整构建/选路。构建使用仓库已有stable-string修补的owned shadow头文件，安装头文件未改。其BF16 state覆盖是native MTP组件的能力，不表示FI ordinary decode支持BF16 state。

真实导入的CuTeDSL为 **4.8.0**，FI为0.7.0.post1；本probe覆盖的FI四个关键文件和DSL两个入口RECORD hash/size均匹配，失败数0。DSL两个入口另与下载的4.8 core wheel字节一致。CUDA仍12.9，stable native仍复用旧SHA `3efc749c…`。`vllm._C`模块缺失被保留为可选模块诊断；实际导入的是`vllm._C_stable_libtorch`。旧wheel metadata的`0.30.1rc1.dev143+g29468dde8.cu129`和源码版本警告不代表本轮源码head；归档没有.git，head通过本地归档/文件SHA链确认，而非服务器现场git观察。

以下只按本轮 `results/micro-dsl48.json` 的三轮中位数重算，单位微秒。范围为实际GDNDecode加相同gated RMSNorm，排除投影、卷积、分配和编译；正百分比表示FI延迟更低。

| B | 状态缓存 | Triton us | FI us | FI延迟下降 |
| ---: | --- | ---: | ---: | ---: |
| 1 | warm | 6.354 | 4.766 | 24.99% |
| 8 | warm | 19.920 | 17.132 | 14.00% |
| 32 | warm | 200.918 | 193.073 | 3.90% |
| 1 | cold | 12.142 | 10.433 | 14.08% |
| 8 | cold | 74.690 | 75.315 | -0.84% |
| 32 | cold | 311.909 | 307.629 | 1.37% |

10个FP32-state配置均通过128步graph轨迹：int32/int64、packed/page stride、0/负padding、changing state-slot映射。首步FP64参考和轨迹维持原pointwise atol=rtol=1e-2、relative L2<1%，零pointwise超限；inactive/null0及page padding始终bitwise不变。轨迹最大relative L2：raw **0.013475%**、norm **0.015861%**、state **0.000011865%**。

本轮显式使用CUDA graph events（timer=graph），没有CUPTI测量，也不是CUPTI自动fallback。warm每图16次边界调用；cold每图一个161/21/6状态轮换组，工作集483/504/576MiB，样本是轮换组均值，不能解释为单次尾延迟。与旧轮的graph分组和运行时不同，不将绝对时间变化归因于PR或DSL。

索引映射B1/B8/B32分别为 **2.092/2.040/2.046us，每builder/cache-group一次remap**，供该group各层复用；多个group可分别remap。raw的“per-model-step/shared by all layers”标签只指这次测得的单个remap，不代表整个模型全层只做一次。本轮cold B8仍慢0.84%，不能声称FI全场景更优。

单L20、TP1、三轮局部测量仍不足以更改auto=Triton；H20、多GPUTP和完整CUDA13 native构建尚未覆盖。

原始micro SHA256：`6af2657d3b78beef72e15bac08d157ae39a9b280e502043889d0d816584446c4`；计算结果：`micro-summary.json`。补充native库SHA256：`4c2fa6d18f9622f34b1637caa186b2d62bf314fc9a3413051f172fb6c2b9a42b`；补充二进制已随本轮results下载；独立重算其895,328字节SHA，与构建及两测试launcher记录一致。


## 本轮 serving smoke

独立核验下载归档的128文件+24目录（152 entries），全部文件字节与本地解包一致；JUnit XML计数亦与上述JSON一致。归档SHA256：`170c065f6db2a2fb7ef61b5451fc86e5fe6a02fef6fb152158ddd505aac5e010`。

仅选 `serve-final-{triton-a1,flashinfer-b1}-attempt2.json`。每臂C1/C8各3×8 measured、各8 warmup，共 **96 measured+32 warmup，128/128成功**。逐请求HTTP200、DONE、finish_reason=length、usage512/64/576、64个输出token IDs、输入fixture及文本/ID hash均核对一致；aggregate与16份分轮原始文件逐项一致，服务日志各64次completion HTTP200。两臂相同源码SHA/runtime记录/旧stable native指纹，命令仅decode backend不同；实际日志确认FP32 SSM、Marlin、FA2、KV block override64，以及相同 **21,845-token实际容量**。runtime记录是两臂共用的启动前probe，未误称每个server进程内重新探测。补充MTP库不用于本普通decode smoke。

以下按相同repeat/request index配对，只描述输出agreement。所有请求使用同一个固定合成fixture。

| measured并发 | 每臂成功请求 | 整串token IDs相同 | 相同位置token IDs |
| ---: | ---: | ---: | ---: |
| C1 | 24/24 | 24/24（100%） | 1536/1536（100%） |
| C8 | 24/24 | 11/24（45.83%） | 730/1536（47.526%） |

warmup C1为8/8整串、512/512位置相同；C8为2/8整串、181/512（35.352%）位置相同。measured C8在Triton自身也有3种输出，FI自身2种；两臂调度没有对齐，不能据这些agreement推断准确率或PR特有根因，更不能称输出等价。两臂各只启动一次、每轮8请求，没有ABBA；原始HTTP计时不作性能收益结论。**旧100题quality未重跑**，旧轮+0.279%/+0.113%及96/95不迁移到新环境。

初次smoke失败记录仍保留：旧benchmark要求每轮≥32请求，传8时在completion请求前拒绝；初次Triton仅创建fixture时已有一次/tokenize请求，两臂均0次completion。随后由本supervisor发送owned SIGTERM，server_exit均0，不将其归因为PR/Engine崩溃。首轮tests的HF本地fixture配置缺失及native launcher的pluggy hook参数错误保留在`attempt1/`，不混入上述passing用例。原`.so`诊断的2失败和runner overall=false也仍保留。最终smoke中的版本模块/NCCL发现警告、warmup sampler JIT及停止过程警告均未抹去。

核验结果见`smoke-summary.json`。在限定为opt-in后端、局部kernel收益和兼容性smoke的表述下，未发现新增收益算术/公平性阻塞；本轮不支持模型精度等价、全场景提速或完整CUDA13验证的声明。
