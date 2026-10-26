
# 02 软件身份、依赖与提取说明

本流程是纯验证流程提取，没有产品代码开发或 PR。本次只编写新目录的五份文档，不改动 `ascend_project` 下的原验证材料，也不操作远端服务。场景配置与测试方法来源是 [pool_64g_a3_profiling](../../ascend_project/pool_64g_a3_profiling)；其中 `profiler_128k_1_archive` 的名称仅用作来源定位。通用 DP 启动工具直接引用 vllm-ascend 代码仓，接口与配置见 03 第 3 节。

## 1. 需要固定的软件与模型

| 对象 | 已知起点 | 执行前需要记录/确认 |
| --- | --- | --- |
| 容器 | P `wl_1011`；D `wl_1011_new` | 两端镜像标签、Image ID/digest、创建参数、挂载、设备映射；原材料未给齐这些身份。 |
| 驱动、CANN | 来源启动入口 `source /usr/local/Ascend/cann/set_env.sh` | 驱动版本、CANN 实际路径与版本、通信库实际加载路径。 |
| vLLM / vllm-ascend | 来源已安装环境，含 DSpark 与 DSV4 专用参数 | 配对 commit/包版本、源码改动、CLI 对 `dspark`、indexer dtype 和额外配置的支持。 |
| 通用 DP launcher | vllm-ascend `examples/external_online_dp/launch_online_dp.py` | 本次核对的代码仓 commit 为 `06b685f346ad525027d03cffe80d58a491a13081`；远端固定同接口的工具版本。该版本号只标识引用工具，不能代填来源容器的框架版本。 |
| memcache / memfabric / UBS IO | 来源的 MetaService、DistributedObjectStore 和 `device_sdma` | 精确版本/构建、Python 模块和原生库路径，分层指标名称与语义。不能用其他任务的版本号补填。 |
| Mooncake NPU 包 | 供 MooncakeHybridConnector 使用 | 包版本与加载库；P/D 拓扑兼容性。 |
| 模型 | `/mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci` | config、tokenizer、量化文件与权重清单；两端内容身份一致。此处的参数不适用于任意同名权重。 |
| 测试 Python | `/mnt/share/l30041859/venv/ais_bench/bin/python` | AISBench、NumPy、PyArrow、torch、transformers、pandas、requests 等版本及导入。 |
| AISBench 前缀工具 | 独立整理和提供，本流程只保留客户端配置与单条命令 | 来源工具 README 记录上游 `rayn-zzz/aisbench_auto_tools_prefix`、commit `ef2b6d8653d57487b622b0dc9db458f8a1e8dec6`；执行时固定统一工具包的实际版本。 |
| 测试数据 | 工具工作目录下 `GSM8K.jsonl`，每行含 `question` | 需从原容器工具目录或可信的内网副本取得并核对哈希；本来源未携带此文件。 |
| PD 代理实现 | 原容器 `/home/w00985460/script/vllm/load_balance_proxy_server_example.py` | 本来源仅有启动脚本，未携带代理 Python 原件；需从原环境取得、核对 CLI 与 PD 协议，并固定哈希。 |

上述缺项限制从空机重建；操作说明以保留了这些依赖的对应容器为起点。需要新建容器时，先取得精确镜像和原创建信息，再制定对应创建命令，不能把不完整的通用 `docker run` 当成已验证环境。

## 2. 现场身份检查

在各自宿主机执行，容器变量按本机角色设置。命令用于取得必要条件，不表示已经在本次整理中执行：

```bash
CONTAINER=wl_1011  # D 节点改为 wl_1011_new
npu-smi info
docker inspect --format '{{.Image}} {{json .Config.Image}} {{json .Mounts}} {{json .HostConfig.Devices}}' "$CONTAINER"
docker exec "$CONTAINER" python3 -m pip show vllm vllm-ascend torch torch-npu memcache-hybrid memfabric-hybrid mooncake-transfer-engine-npu
docker exec "$CONTAINER" readlink -f /usr/local/Ascend/cann
```

包名随安装方式可能不同；`pip show` 缺项时从相应模块、源码 commit 和原生库补查，不能以命令运行完成表示版本齐备。对源码工作区同时核对 commit 和未提交差异；对运行中 worker 核对 `/proc/<pid>/maps` 中的通信与池化库。

## 3. 提取取舍

- 03 收录来源的 SSD 配置、vLLM 接入配置、单服务启动入口和必要的池化指标工具。DP launcher 引用 vllm-ascend examples，只写工具位置、版本和参数；AISBench 只保留配置与一条命令。
- 正式脚本全文保留来源参数。通过导出 `TASK_ROOT` 可以在独立运行目录部署；固定 IP、网卡、模型和库路径必须在部署前逐项核对。
- 三份池化配置保留全部有效项，去掉历史调参和成绩备注；不改变有效参数，源文件也不修改。
- 原 `bringup_ssd.sh` 中的广泛进程清理、测试扫点中的自动重启/删除数据集和原 loop 准备脚本不作为正式操作入口。04 给出显式的准备、启动与停止顺序，原文件完整保留在源目录。
- 原 `read_tier_hits.sh` 只适合快速查看累计值，缺指标时可能输出零；05 要求先核对指标存在，并对原始快照做差。03 补充一个离线差值工具，缺指标、序列变化或计数器下降时直接报错；它是本次新增辅助脚本，未做远端实测。
- 旧工具 README 内的 135 地址、并发 49、DP4 等说明与本来源选定入口不一致；本流程以 03 的客户端配置、测试命令及 P/D 启动脚本的实际参数为准。
- 原任务定制 DP launcher 保留在原位置；本流程改为在 P/D 各自工作目录使用仓库 launcher 和名为 `run_dp_template.sh` 的场景模板链接。单 rank 脚本仅更新参数来源注释，服务参数不变；不使用定制版的模板、日志或错峰 CLI 选项。此替换尚未进行目标机器复验。
- 不引入 profiling 采集、性能对照或历史成绩。必要的流程局限会在相应章节直接说明。

## 4. 行为变化与恢复边界

本次没有给 vLLM、连接器或原生库增加补丁。SSD 配置中的 `force_new_disk=true` 是首次初始化本任务专用测试盘的选择；如果要保留数据，须改为 false 并核对版本行为。正常命中检查期间不重启 Meta/standalone，也不重新初始化盘。

退出流程按 04 停止本轮拥有的进程，恢复本轮调整的大页等资源。原任务容器、代码、配置和日志均保留；新运行配置的修改只作用于独立运行目录。

## 5. 文内脚本来源身份

下表由本次提取的原文件生成，用于定位脚本来源；它不是某次远端运行的输入身份。整理后的部署变更需要另行核对。

| 来源相对路径（`03_verification_scripts/` 下） | SHA256 |
| --- | --- |
| `config/mmc-meta-ssd.conf` | `b2095a669670bc5590d136e86e6742d98c4c1db8f4a9f80bff55a0e19921a8e1` |
| `config/mmc-local-standalone-ssd.conf` | `56a0836f5f351b0085043806c042bcfaeca83404a4ffd20f7087b3d1ae16750c` |
| `config/mmc-local.conf` | `3c56d06304bf646d0ae9f80d9a7c66a6b94a48410c378be014994ced5d7ae5a5` |
| `launch/meta_ssd.sh` | `c8ab3d5dcfa2206acbcbf2562a2b57d9ae7880861b58d2800648a0f188cad58f` |
| `launch/standalone_ssd.sh` | `4dec69ba53747104bf5179952986901e511a78c66bf33a8ee3bbe56cb5e9da65` |
| `launch/memcache_standalone.py` | `23400db0fa7ec69ba005728e07183c28f64caf0813dcd55920207e5de593c7b4` |
| `launch/p_rank.sh` | `686307e8e7eac80ab2a02b7fb9d3d623f6311485bfdb8ca048023dafdea5fe4a` |
| `launch/d_rank.sh` | `8fb94049bc8d432b6401d7f31777127d28afc3ca0b6d292d8c590f5ba8b3d6d3` |
| `launch/proxy.sh` | `7249060fa1faee018d0a0c2e9f353750bccfe54ce73c961772e21151d8f2e201` |
| `tests/aisbench/config.py` | `f5ad2f9e4e7bbe6f43948cfaa7fe9e26cdc4f05f4fd94a22bcc87e8dee2cb84d` |
| `tests/read_tier_hits.sh` | `70c7e241fbeba1c574743cfdd5fc6a48faed94a98728bc1ff27427a7c1dd99d6` |