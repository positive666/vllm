# #60403 d8ae9e900 独立证据核验

独立读取 `final-download/results` 的逐请求记录、JUnit XML、服务日志和参数探针；没有依赖 `result-audit.json` 的统计结论，也未修改原始数据。

本轮绑定源码 `d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6`、固定main基线 `8352b242`。`source-review.tar.gz` 的SHA为 `3cdf0fefb45a3b4cc9241110bfba2449b571cf95f68cebb1ee69e2e690e0e5da`，声明的5个源码/测试文件逐一重hash匹配；runtime记录、两个探针及两臂服务的相关源码SHA均一致。

| 核验项 | 独立结果 |
| --- | --- |
| 核心测试 | 40 adapter/split +63 metadata +1 config = **104 passed**，无失败、错误或跳过，exit均0 |
| serving retry | 每臂48 measured +16 warmup，共 **96+32请求，128/128成功** |
| 每个HTTP请求 | HTTP200、DONE、length结束；usage为512/64/576；输入fixture、64个输出token IDs、文本/IDs hash均有效 |
| 服务配置 | 两臂命令仅GDN backend不同；实际日志确认FP32 SSM、Marlin、FA2、KV blocks override64、**21,845-token容量** |
| override日志 | 两次构造，覆盖环境开关的INFO消息实际打印1次；三项相关exit均0 |

aggregate与16份分轮JSON逐项相同，服务日志各64次completion HTTP200，log SHA与服务evidence一致。Triton日志SHA为 `7b99f6fe973cc84ff45654ba2aeb7b555f8fed5824edb1888d766c9a1d801a0b`；FI为 `607ea40293bb9a12838dadc572fc85e7ba8990159410b21f87668235065718e7`。fixture文件SHA为 `2fc2a4d643b0abea735b5d83888585eec25eb5fda2919b7f964add44834a350d`；协议SHA为 `ecac8fe0e02430bca5de49bb03839b5fb2e9319327ada419fb349767fcf92595`，两个backend使用同一fixture/client/protocol。服务中的runtime记录是共用的启动前probe，不是分别在两个server进程内重新探测。

按相同repeat/request index比较measured输出：

| 并发 | 整串token IDs相同 | 相同位置token IDs |
| ---: | ---: | ---: |
| C1 | 24/24（100%） | 1536/1536（100%） |
| C8 | 12/24（50%） | 833/1536（54.2318%） |

所有请求使用同一合成fixture。measured C8在Triton自身有3种输出，FI自身2种；两臂调度未对齐。这是输出agreement描述，**不证明精度等价或PR特有根因**。warmup C8为4/8整串、305/512位置相同；未混入measured统计。

参数探针使用真实、仍需梯度的FP32 `A_log`/`dt_bias` Parameter，B1/H2/HV4、BF16 packed gates、FP32 state及 `inference_mode`。`CUTE_DSL_ENABLE_TVM_FFI=0` 时raw FI在DLPack导出处报需梯度的 `BufferError`；设为1时raw FI报a参数的16-byte对齐 `ValueError`（a指针偏移8字节），不能把两种失败都称为梯度导出错误。当前adapter在两个进程均重置状态后通过原helper断言，输出最大绝对误差0，容差未放宽；参数仍需梯度。runtime的 `ENABLE_TVM_FFI` 属性不可获取，开关依据记录的进程环境和实际错误路径；FI为0.7.0.post1、DSL为4.8.0、Torch为2.13.0+cu129。

另从官方固定FlashMLA commit `0eee43b12f034b657133cf2afca6a72ebb6efccf` 重新读取21,843字节源码并在内存执行归档CMake的替换：上游SHA `76eb66a7922531f402ce6fadd51e01aab19b84bff063d2b33dfbaa68f8915e26`、生成wrapper SHA `75b470a05ac3baa9f80df664fa5bccdb52fba6016caed7a85f03b2c7725b0510` 均与生成记录匹配。该接口用于补齐依赖导入，不能称为FlashMLA GPU验证。[官方固定源码](https://raw.githubusercontent.com/deepseek-ai/FlashMLA/0eee43b12f034b657133cf2afca6a72ebb6efccf/flash_mla/flash_mla_interface.py)

原始 `pipeline.exit=1`、初次两臂启动失败及attempt1/2失败均保留；与旧download共有的89份非pytest-cache文件字节完全相同。成功的是 **`serving-retry.exit=0` 及两臂attempt2 exit0**，不把原pipeline改称通过。

本轮只有单L20/SM89/TP1、每backend一次成功启动和短合成smoke。没有重跑性能benchmark或100题GSM8K；未验证SM80实际硬件、H20、多GPU或完整CUDA13构建。native仍复用已有二进制，旧轮性能/质量数字不能移作新head验证结果。
