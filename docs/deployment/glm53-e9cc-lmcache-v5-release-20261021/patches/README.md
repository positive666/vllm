# 固定补丁来源

补丁文件及SHA见上级`versions.json`。两种镜像的vLLM运行字节分别验收，不能互相替代。

| 顺序 | 目标与来源 | 保留范围 |
| --- | --- | --- |
| 1 | vLLM既有[#62479](https://github.com/vllm-project/vllm/pull/62479) | 仅Flash profile：完整prefill边界状态交接；FP8 profile不含该运行覆盖。 |
| 2 | 固定LMCache 2a34；与[#4844](https://github.com/LMCache/LMCache/pull/4844)协调 | exact-boundary、独立state-store和源块/事件保留。 |
| 3 | 补丁2之后；与[#4836](https://github.com/LMCache/LMCache/pull/4836)协调 | ordinary/lazy store全部future/event完成、未知DMA完成时失败关闭。 |
| 4 | 补丁3之后，第三方[#5938](https://github.com/LMCache/LMCache/pull/5938) | scheduler/metadata/worker协同，抑制跨step取消时的重复finished_sending。 |

PR #5938 作者是**amd-lalithnc**，固定 head 为 `0b16807a16eb6d101c99d903a2afd77fcc718336`。
这里保存的是在已验v4之上的三个运行文件移植增量，不是第三方PR完整四文件diff，也不记为我们的新贡献。
版本身份、dist-info和RECORD随v5打包更新，四份LMCache原生模块没有重新编译。

不要再次叠加相同#5938，或把旧GLM/HMA/MTP/APC等已由固定官方基线包含的workaround重复应用。
提供的补丁用于来源审阅，不包含完整镜像重建锁或自动安装流程；应用前须在各自固定源码上执行`git apply --check`。
本包不新开或修改上游PR；PR远端状态未在本包生成时重新查询。

AI参与了准备和验证。上游提交者仍须理解维护全部变更、逐行审阅、运行相关测试，披露AI辅助、重复工作检查和实际模型评估。
