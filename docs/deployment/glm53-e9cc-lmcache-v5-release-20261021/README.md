# GLM-5.3：固定MAIN e9cc与LMCache v5验收

此目录记录两个已发布并通过独立匿名registry核验的实验镜像，`production=false`。
两个精确镜像分别完成模型级验证；config digest和registry manifest digest分别记录，不混用。

```text
ghcr.io/positive666/model@sha256:57fdef18ef5a65f9766621af870a530324635f179b3e7cf44b8f48b491b1e4c8
ghcr.io/positive666/model@sha256:fa5f5c6d7dc9fed587d311e594457fb61dca9fcc066a11e5dc70656af4cf1da3
```

Flash首次push本身exit0，但紧随的匿名核验遇到URLError，原整体失败收据保留。
后续独立verify-only已核同一digest与config/46层，不把首次失败改标成功，也未为核验再次推送。

| profile | 实际镜像config | 已验范围 |
| --- | --- | --- |
| FP8 TP16 | `12e65dc97ab78d320689aad552f8e232ab74227f5ac1055e333c76b1aba76d45` | 2×8 H20、MTP3、C1 A/A/B/A、一波32并发、首输出后断流恢复。无vLLM私有运行覆盖。 |
| Flash TP8 | `536d9bb5dd69adcb9306d90c4abbc592579bbd4efdbf263559eb049b2728eb5f` | 8 H20、MTP3、C1及前缀相关passkey A/A/B/A；增加既有#62479 producer。没有本轮Flash C32。 |

此Git分支只承载发布记录、参数摘要和补丁包（docs-only），在旧发布分支上新增此目录以保留旧资料。
它不是最新MAIN源码checkout，也不把旧Git分支base当作本轮运行代码；运行时基线来自下列固定官方源码及wheel。

固定来源是官方vLLM MAIN `e9cc6244a0095ba6525230b49a8ecdc2cf7b0f01`，使用该提交的官方wheel及14个原生文件/Rust，未用B8内核冒充新MAIN。
LMCache是`2a34fa8`加保留修复和第三方#5938的v5，不等于已验证最新移动dev。原生库身份、补丁SHA及归属见[versions.json](versions.json)和[patches说明](patches/README.md)。

部署参数见[FP8 profile](profiles/fp8-tp16.json)与[Flash profile](profiles/flash-tp8.json)。FP8每节点一个256GiB本地MP池，物理合计512GiB；各自通过本机CUDA IPC服务本机8卡，并非共享跨机内存池，也未接Mooncake L2。
Flash使用一个256GiB本地池。配置32并发、256K或1M上限不等于已验证生产32人体验或完整上下文负载。
此包记录已验配置与来源，不提供完整镜像重建锁或可直接连接任何既有服务的部署自动化。

FP8单请求32K冷/完整恢复/分支/逆向回访的首任意输出为18.265/0.375/5.672/0.313秒，外部命中0/32767/24320/32767 token。
但混合C32虽32/32功能正确，首任意输出p95为120.641秒、首正文p95为136.660秒；唯一warm长请求仍需90.266秒才开始输出。
冷长预填充与decode竞争仍明显，缓存未解决全部并发等待。首任意输出包含思考，不能当作正文已返回。

Flash C1与passkey共8条均正确，完整恢复32000 token、分支16000 token；现场chunk640与checkpoint16000由API/启动日志/MP布局/源码共同推导，没有从命中倒推。
这些是单并发合成回归；本轮没有Flash C32、加载中取消、故障注入或长期TTL验收。FP8取消也仅覆盖首输出后断流的一例，不直接证明内部abort ACK或finished_sending恰好一次。
逐物理rank与缓存IID没有证明一一映射；Flash仅保存总完成加载量，不声称逐请求逐IID DMA。不能宣称所有性能更快、所有内核无NaN、普遍无泄漏或生产SLO。

公开数值与边界见[validation/results.json](validation/results.json)。原始业务正文、主机地址/用户名、容器ID、主机目录、环境快照和凭据不在公共包内。
原取消R1的不适用完成计数门槛失败记录在私有归档保留，未改标通过。旧B8/v4公共目录和旧发布digest保持原值，不能把旧结果改名为新MAIN结果。

验证环境须已安装 `regex`；可复用已准备的仓库 lint 或 private venv。

静态核对使用仓库要求的uv和显式venv，例如从本目录运行：

```bash
uv run --no-project --offline --no-sync --python /path/to/.venv/bin/python /path/to/.venv/bin/python validation/verify_bundle.py --require-published
```

该命令只检查公开包的哈希、独立profile身份和脱敏范围，不发网络请求、不部署模型、不推送镜像或Git，也不替代人工审阅和生产验收。
