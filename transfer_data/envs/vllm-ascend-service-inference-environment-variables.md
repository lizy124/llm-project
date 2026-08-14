# vLLM 服务启动与推理环境变量（`docs/source`）

- 扫描目录：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source`
- 扫描文档文件：**143**
- 识别到含服务启动或推理命令的文档：**90**
- 启动/推理相关章节：**305**
- 服务启动、推理及运行时配置主表变量：**73**
- 当前源码可用变量：**70**
- 文档遗留、当前源码不支持：**3**

> 本文不是 `docs/source` 环境变量全集。主表以 `vllm serve`、API server、服务脚本、分布式启动、离线推理或推理客户端启动链路为主，并补充 additional config 中直接影响服务运行时的迁移变量。镜像名、模型路径、端口占位等 Shell 辅助变量被排除。

## 分类统计

| 分类 | 数量 | 占比 |
|---|---:|---:|
| vLLM Ascend 产品配置 | 7 | 9.6% |
| Ascend/CANN/HCCL 与 NPU 运行时 | 20 | 27.4% |
| 分布式启动与并行环境 | 10 | 13.7% |
| KV Transfer、PD 分离与存储后端 | 2 | 2.7% |
| 上游 vLLM/PyTorch/模型生态 | 21 | 28.8% |
| 系统运行环境 | 4 | 5.5% |
| 其他服务运行变量 | 6 | 8.2% |
| 文档遗留/当前源码不支持 | 3 | 4.1% |
| **合计** | **73** | **100.0%** |

## 必要性统计

| 必要性 | 数量 | 解释 |
|---|---:|---|
| 场景必需 | 14 | 在对应多节点、Ray、DP、Mooncake/RFork/Netloader 等场景中缺失会导致启动链路不完整。 |
| 场景配置 | 9 | 用于选择设备、模型来源、外部库路径或特定部署资源；是否需要取决于环境。 |
| 可选调优/调试 | 47 | 通常存在默认行为，主要改变性能、超时、内存、日志或特性开关。 |
| 不应使用 | 3 | 文档仍出现，但当前代码没有注册或消费；设置后不会启用文档描述的功能。 |

## 设置方式统计

同一变量可能通过多种方式进入服务环境。

| 设置方式 | 变量数 |
|---|---:|
| export | 70 |
| Python os.environ | 11 |
| Docker -e/--env | 2 |
| 行内进程赋值 | 2 |
| 源码 os.getenv | 1 |

## 源码与文档不一致项

以下变量仅能证明“文档写过”，不能证明当前 vLLM Ascend 支持。判定依据是当前 `vllm_ascend/envs.py`、全源码读取点和 Git 历史。

| 变量 | 当前状态 | 历史证据与结论 |
|---|---|---|
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | **文档遗留，当前源码不支持** | 曾用于 v0.9.1 sampler TopK/TopP patch；提交 830332ebf（2025-07-09，Clean up v0.9.1 code）已从 envs.py、patch 和测试删除。当前 GLM4.x 文档为遗留配置。 |
| `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED` | **文档遗留，当前源码不支持** | 仅存在于提交 e3636c7eb（2025-08-05，明确标注 0.9.1 only）的兼容实现；该提交不是当前 main 的祖先，当前源码没有注册或读取该变量。large_scale_ep.md 为旧文档迁移残留。 |
| `VLLM_DP_SIZE_LOCAL` | **文档遗留，当前源码不支持** | 与 VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED 一同由提交 e3636c7eb（2025-08-05，明确标注 0.9.1 only）加入，用于旧版 external DP 的本机 DP size；当前 main 的上游 vLLM 与 vLLM Ascend 源码均未注册或读取。large_scale_ep.md 中的设置属于旧兼容实现残留。 |

## 必要性说明

- **基础必需**：启动方式或硬件拓扑明确要求；未配置可能无法发现设备或建立通信。
- **场景必需**：仅在多节点、Ray、PD 分离、Mooncake/UCM 等特定部署中必需。
- **可选调优/调试**：不配置通常仍可启动，但会改变性能、内存、日志、profiling 或特性路径。

本文不机械地把每个 `export` 都标成“必需”。具体必要性应结合对应文档场景和启动命令判断。

## 分类明细

`说明` 简要描述变量的作用、归属或当前兼容状态；每个变量最多展示 8 个文档位置。

### vLLM Ascend 产品配置（7）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `DYNAMIC_EPLB` | 可选调优/调试 | export | 启用动态专家并行负载均衡（EPLB）。 | `docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:257` |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 可选调优/调试 | export | 启用均衡调度；迁移期变量，推荐使用 scheduler_config.enable_balance_scheduling。 | `docs/source/tutorials/models/DeepSeek-R1.md:143; docs/source/tutorials/models/DeepSeek-R1.md:246; docs/source/tutorials/models/DeepSeek-R1.md:292; docs/source/tutorials/models/DeepSeek-V3.1.md:154; docs/source/tutorials/models/DeepSeek-V3.1.md:267; docs/source/tutorials/models/DeepSeek-V3.1.md:320; docs/source/tutorials/models/GLM4.x.md:133; docs/source/tutorials/models/GLM4.x.md:183` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 可选调优/调试 | export | 启用 FlashComm1 通信优化；迁移期变量，推荐使用 enable_flashcomm1。 | `docs/source/developer_guide/Design_Documents/context_parallel.md:80; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:78; docs/source/tutorials/features/suffix_speculative_decoding.md:86; docs/source/tutorials/models/DeepSeek-V3.1.md:453; docs/source/tutorials/models/DeepSeek-V3.1.md:528; docs/source/tutorials/models/DeepSeek-V3.2.md:136; docs/source/tutorials/models/DeepSeek-V3.2.md:189; docs/source/tutorials/models/DeepSeek-V3.2.md:236` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | 可选调优/调试 | export | 控制 Fused MC2 融合通信计算路径；推荐使用 enable_fused_mc2。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:79; docs/source/tutorials/models/DeepSeek-V4-Pro.md:596; docs/source/tutorials/models/DeepSeek-V4-Pro.md:667; docs/source/tutorials/models/GLM4.x.md:524; docs/source/tutorials/models/GLM4.x.md:593; docs/source/tutorials/models/GLM5.2.md:144; docs/source/tutorials/models/GLM5.2.md:221; docs/source/tutorials/models/GLM5.2.md:275` |
| `VLLM_ASCEND_ENABLE_MLAPO` | 可选调优/调试 | export | 启用 MLAPO 优化；迁移期变量，推荐使用 enable_mlapo。 | `docs/source/tutorials/models/DeepSeek-V3.2.md:134; docs/source/tutorials/models/DeepSeek-V3.2.md:187; docs/source/tutorials/models/DeepSeek-V3.2.md:234; docs/source/tutorials/models/DeepSeek-V3.2.md:285; docs/source/tutorials/models/DeepSeek-V3.2.md:336; docs/source/tutorials/models/GLM5.2.md:1075; docs/source/tutorials/models/GLM5.md:1065; docs/source/tutorials/models/GLM5.md:1134` |
| `VLLM_ASCEND_ENABLE_NZ` | 可选调优/调试 | Python os.environ, export | 控制权重 NZ 格式转换策略；迁移期变量，推荐使用 weight_nz_mode。 | `docs/source/tutorials/models/DeepSeekOCR2.md:127; docs/source/tutorials/models/GLM5.2.md:1258; docs/source/tutorials/models/GLM5.2.md:1324; docs/source/tutorials/models/GLM5.2.md:1390; docs/source/tutorials/models/GLM5.2.md:1463; docs/source/user_guide/feature_guide/sleep_mode.md:135; docs/source/user_guide/feature_guide/sleep_mode.md:98` |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | 可选调优/调试 | 源码 os.getenv | 启用按块转置 KV Cache 的融合算子；推荐使用 enable_transpose_kv_cache_by_block。 | `docs/source/user_guide/configuration/additional_config.md:25` |

### Ascend/CANN/HCCL 与 NPU 运行时（20）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `ASCEND_A3_ENABLE` | 场景配置 | export | 启用面向 Atlas A3 硬件的运行路径或相关优化。 | `docs/source/tutorials/models/DeepSeek-V3.2.md:421; docs/source/tutorials/models/DeepSeek-V3.2.md:494; docs/source/tutorials/models/DeepSeek-V3.2.md:568; docs/source/tutorials/models/DeepSeek-V3.2.md:642; docs/source/tutorials/models/GLM4.x.md:393; docs/source/tutorials/models/GLM4.x.md:456; docs/source/tutorials/models/GLM4.x.md:518; docs/source/tutorials/models/GLM4.x.md:587` |
| `ASCEND_AGGREGATE_ENABLE` | 可选调优/调试 | export | 控制 Ascend 通信传输中的聚合能力。 | `docs/source/tutorials/models/DeepSeek-V3.2.md:418; docs/source/tutorials/models/DeepSeek-V3.2.md:491; docs/source/tutorials/models/DeepSeek-V3.2.md:565; docs/source/tutorials/models/DeepSeek-V3.2.md:639; docs/source/tutorials/models/GLM4.x.md:390; docs/source/tutorials/models/GLM4.x.md:453; docs/source/tutorials/models/GLM4.x.md:515; docs/source/tutorials/models/GLM4.x.md:584` |
| `ASCEND_CONNECT_TIMEOUT` | 可选调优/调试 | export | 设置 Ascend 一侧通信建立连接的超时时间。 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:167; docs/source/tutorials/models/DeepSeek-V4-Pro.md:243; docs/source/tutorials/models/MiniMax-M3.md:212; docs/source/tutorials/models/MiniMax-M3.md:255; docs/source/tutorials/models/MiniMax-M3.md:301; docs/source/tutorials/models/MiniMax-M3.md:345; docs/source/user_guide/feature_guide/kv_pool.md:175; docs/source/user_guide/feature_guide/kv_pool.md:250` |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | 场景配置 | export | 允许通信或 KV 传输使用 Fabric Memory。 | `docs/source/user_guide/feature_guide/kv_pool.md:242; docs/source/user_guide/feature_guide/kv_pool.md:363` |
| `ASCEND_LAUNCH_BLOCKING` | 可选调优/调试 | export | 强制 Ascend 算子同步执行，主要用于定位异步报错。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:76` |
| `ASCEND_RT_VISIBLE_DEVICES` | 场景配置 | export | 指定当前进程可见的 Ascend NPU 设备。 | `docs/source/developer_guide/contribution/doc_writing.md:240; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:280; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:335; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:390; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:445; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:503; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:558; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:613` |
| `ASCEND_TRANSFER_TIMEOUT` | 可选调优/调试 | export | 设置 Ascend 数据传输操作的超时时间。 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:168; docs/source/tutorials/models/DeepSeek-V4-Pro.md:244; docs/source/tutorials/models/MiniMax-M3.md:213; docs/source/tutorials/models/MiniMax-M3.md:256; docs/source/tutorials/models/MiniMax-M3.md:302; docs/source/tutorials/models/MiniMax-M3.md:346; docs/source/user_guide/feature_guide/kv_pool.md:178; docs/source/user_guide/feature_guide/kv_pool.md:251` |
| `ASCEND_TRANSPORT_PRINT` | 可选调优/调试 | export | 控制 Ascend Transport 层的日志输出。 | `docs/source/tutorials/models/DeepSeek-V3.2.md:419; docs/source/tutorials/models/DeepSeek-V3.2.md:492; docs/source/tutorials/models/DeepSeek-V3.2.md:566; docs/source/tutorials/models/DeepSeek-V3.2.md:640; docs/source/tutorials/models/GLM4.x.md:391; docs/source/tutorials/models/GLM4.x.md:454; docs/source/tutorials/models/GLM4.x.md:516; docs/source/tutorials/models/GLM4.x.md:585` |
| `CPU_AFFINITY_CONF` | 可选调优/调试 | export | 配置进程或线程的 CPU 亲和性，减少调度抖动。 | `docs/source/tutorials/models/GLM5.2.md:873; docs/source/tutorials/models/GLM5.2.md:923; docs/source/tutorials/models/PaddleOCR-VL.md:106` |
| `HCCL_BUFFSIZE` | 可选调优/调试 | Python os.environ, export | 设置 HCCL 通信缓冲区大小。 | `docs/source/developer_guide/contribution/doc_writing.md:239; docs/source/developer_guide/contribution/doc_writing.md:90; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:71; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:276; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:331; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:386; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:441; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:499` |
| `HCCL_CONNECT_TIMEOUT` | 可选调优/调试 | export | 设置 HCCL 建立通信连接的超时时间。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:83; docs/source/tutorials/models/DeepSeek-V3.1.md:441; docs/source/tutorials/models/DeepSeek-V3.1.md:516; docs/source/tutorials/models/DeepSeek-V3.1.md:591; docs/source/tutorials/models/DeepSeek-V3.1.md:664; docs/source/tutorials/models/DeepSeek-V3.2.md:288; docs/source/tutorials/models/DeepSeek-V3.2.md:339; docs/source/tutorials/models/DeepSeek-V4-Flash.md:496` |
| `HCCL_EXEC_TIMEOUT` | 可选调优/调试 | export | 设置 HCCL 集合通信任务的执行超时时间。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:82; docs/source/tutorials/models/DeepSeek-V3.1.md:440; docs/source/tutorials/models/DeepSeek-V3.1.md:515; docs/source/tutorials/models/DeepSeek-V3.1.md:590; docs/source/tutorials/models/DeepSeek-V3.1.md:663; docs/source/tutorials/models/DeepSeek-V4-Flash.md:495; docs/source/tutorials/models/DeepSeek-V4-Flash.md:563; docs/source/tutorials/models/DeepSeek-V4-Flash.md:643` |
| `HCCL_IF_IP` | 场景必需 | export | 指定 HCCL 通信使用的本机 IP 地址。 | `docs/source/developer_guide/contribution/doc_writing.md:221; docs/source/developer_guide/contribution/doc_writing.md:234; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:269; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:324; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:379; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:434; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:492; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:547` |
| `HCCL_INTRA_PCIE_ENABLE` | 可选调优/调试 | export | 控制节点内 HCCL 是否使用 PCIe 通信链路。 | `docs/source/tutorials/models/DeepSeek-R1.md:247; docs/source/tutorials/models/DeepSeek-R1.md:293; docs/source/tutorials/models/DeepSeek-V3.1.md:268; docs/source/tutorials/models/DeepSeek-V3.1.md:321; docs/source/tutorials/models/DeepSeek-V3.2.md:289; docs/source/tutorials/models/DeepSeek-V3.2.md:340; docs/source/tutorials/models/Kimi-K2.5.md:257; docs/source/tutorials/models/Kimi-K2.5.md:325` |
| `HCCL_INTRA_ROCE_ENABLE` | 可选调优/调试 | export | 控制节点内 HCCL 是否使用 RoCE 通信链路。 | `docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:217; docs/source/tutorials/models/DeepSeek-R1.md:248; docs/source/tutorials/models/DeepSeek-R1.md:294; docs/source/tutorials/models/DeepSeek-V3.1.md:269; docs/source/tutorials/models/DeepSeek-V3.1.md:322; docs/source/tutorials/models/DeepSeek-V3.2.md:290; docs/source/tutorials/models/DeepSeek-V3.2.md:341; docs/source/tutorials/models/GLM5.2.md:1084` |
| `HCCL_OP_EXPANSION_MODE` | 可选调优/调试 | Python os.environ, export | 控制 HCCL 通信算子的展开或执行模式。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:73; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:278; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:333; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:388; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:443; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:501; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:556; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:611` |
| `HCCL_RDMA_TIMEOUT` | 可选调优/调试 | export | 设置 HCCL RDMA 通信的超时时间。 | `docs/source/user_guide/feature_guide/kv_pool.md:249; docs/source/user_guide/feature_guide/kv_pool.md:370` |
| `HCCL_SOCKET_IFNAME` | 场景必需 | export | 指定 HCCL Socket 通信使用的网卡。 | `docs/source/developer_guide/contribution/doc_writing.md:224; docs/source/developer_guide/contribution/doc_writing.md:237; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:272; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:327; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:382; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:437; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:495; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:550` |
| `HCCL_TRANSFER_TIMEOUT` | 可选调优/调试 | export | 设置 HCCL 数据传输的超时时间。 | `docs/source/tutorials/models/GLM5.2.md:1263; docs/source/tutorials/models/GLM5.2.md:1329; docs/source/tutorials/models/GLM5.2.md:136; docs/source/tutorials/models/GLM5.2.md:1395; docs/source/tutorials/models/GLM5.2.md:1468; docs/source/tutorials/models/GLM5.2.md:213; docs/source/tutorials/models/GLM5.2.md:267; docs/source/tutorials/models/GLM5.2.md:452` |
| `TASK_QUEUE_ENABLE` | 可选调优/调试 | Python os.environ, export | 控制 Ascend 任务队列机制，用于调整算子下发方式。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:75; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:277; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:332; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:387; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:442; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:500; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:555; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:610` |

### 分布式启动与并行环境（10）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `GLOO_SOCKET_IFNAME` | 场景必需 | export | 指定 Gloo 分布式通信使用的网卡。 | `docs/source/developer_guide/contribution/doc_writing.md:222; docs/source/developer_guide/contribution/doc_writing.md:235; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:270; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:325; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:380; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:435; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:493; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:548` |
| `OMP_NUM_THREADS` | 可选调优/调试 | export | 设置 OpenMP 并行区域使用的线程数。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:70; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:274; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:329; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:384; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:439; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:497; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:552; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:607` |
| `OMP_PROC_BIND` | 可选调优/调试 | Python os.environ, export | 控制 OpenMP 线程是否绑定到固定 CPU 核。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:68; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:273; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:328; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:383; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:438; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:496; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:551; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:606` |
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | 场景必需 | export | 阻止 Ray 自动改写 Ascend NPU 可见设备列表。 | `docs/source/tutorials/features/ray.md:115; docs/source/tutorials/features/ray.md:127` |
| `TP_SOCKET_IFNAME` | 场景必需 | export | 指定张量并行 Socket 通信使用的网卡。 | `docs/source/developer_guide/contribution/doc_writing.md:223; docs/source/developer_guide/contribution/doc_writing.md:236; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:271; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:326; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:381; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:436; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:494; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:549` |
| `VLLM_DP_MASTER_IP` | 场景必需 | export | 指定数据并行协调节点的 IP 地址。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:119; docs/source/user_guide/feature_guide/large_scale_ep.md:183` |
| `VLLM_DP_MASTER_PORT` | 场景必需 | export | 指定数据并行协调节点的监听端口。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:120; docs/source/user_guide/feature_guide/large_scale_ep.md:184` |
| `VLLM_DP_RANK` | 场景必需 | export | 指定当前进程在全局数据并行组中的 rank。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:122; docs/source/user_guide/feature_guide/large_scale_ep.md:186` |
| `VLLM_DP_RANK_LOCAL` | 场景必需 | export | 指定当前进程在本机数据并行组中的 local rank。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:121; docs/source/user_guide/feature_guide/large_scale_ep.md:185` |
| `VLLM_DP_SIZE` | 场景必需 | export | 指定全局数据并行实例总数。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:118; docs/source/user_guide/feature_guide/large_scale_ep.md:182` |

### KV Transfer、PD 分离与存储后端（2）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `MMC_LOCAL_CONFIG_PATH` | 可选调优/调试 | export | 指定 MMC 本地配置文件路径。 | `docs/source/user_guide/feature_guide/kv_pool.md:562; docs/source/user_guide/feature_guide/kv_pool.md:684` |
| `MOONCAKE_CONFIG_PATH` | 场景必需 | export | 指定 Mooncake KV Transfer 的配置文件路径。 | `docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:215; docs/source/tutorials/models/GLM5.2.md:1083; docs/source/tutorials/models/GLM5.2.md:990; docs/source/user_guide/feature_guide/kv_pool.md:238; docs/source/user_guide/feature_guide/kv_pool.md:358; docs/source/user_guide/feature_guide/ucm_deployment.md:257; docs/source/user_guide/feature_guide/ucm_deployment.md:305; docs/source/user_guide/feature_guide/ucm_deployment.md:527` |

### 上游 vLLM/PyTorch/模型生态（21）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `HF_DATASETS_CACHE` | 可选调优/调试 | Python os.environ | 指定 Hugging Face Datasets 的本地缓存目录。 | `docs/source/tutorials/models/Qwen3-Embedding.md:222; docs/source/tutorials/models/Qwen3-Reranker.md:251; docs/source/tutorials/models/Qwen3-VL-Embedding.md:226; docs/source/tutorials/models/Qwen3-VL-Reranker.md:256` |
| `HF_ENDPOINT` | 可选调优/调试 | Python os.environ, export | 指定 Hugging Face Hub 的访问端点或镜像站。 | `docs/source/developer_guide/evaluation/using_lm_eval.md:115; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:177; docs/source/tutorials/models/Qwen3-Embedding.md:223; docs/source/tutorials/models/Qwen3-Reranker.md:252; docs/source/tutorials/models/Qwen3-VL-Embedding.md:227; docs/source/tutorials/models/Qwen3-VL-Reranker.md:257` |
| `HF_HOME` | 可选调优/调试 | export | 指定 Hugging Face 模型、数据集等内容的缓存根目录。 | `docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:73` |
| `PYTORCH_NPU_ALLOC_CONF` | 可选调优/调试 | Docker -e/--env, Python os.environ, export | 配置 PyTorch NPU 显存分配器及内存管理策略。 | `docs/source/developer_guide/evaluation/using_ais_bench.md:30; docs/source/developer_guide/evaluation/using_evalscope.md:28; docs/source/developer_guide/evaluation/using_lm_eval.md:30; docs/source/developer_guide/evaluation/using_opencompass.md:28; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:69; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:275; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:330; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:385` |
| `TOKENIZERS_PARALLELISM` | 可选调优/调试 | export | 控制 Hugging Face Tokenizers 是否启用内部并行。 | `docs/source/tutorials/models/DeepSeekOCR2.md:128; docs/source/tutorials/models/DeepSeekOCR2.md:131` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | 可选调优/调试 | export | 允许配置超过模型声明值的最大上下文长度。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:77; docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:103; docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:56` |
| `VLLM_BATCH_INVARIANT` | 可选调优/调试 | Python os.environ, export, 行内进程赋值 | 启用批次不变性相关行为，减少批次组成对结果的影响。 | `docs/source/user_guide/feature_guide/batch_invariance.md:38; docs/source/user_guide/feature_guide/batch_invariance.md:46; docs/source/user_guide/feature_guide/flash_attention.md:102; docs/source/user_guide/feature_guide/flash_attention.md:70` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 可选调优/调试 | export | 设置等待 vLLM 引擎完成初始化的超时时间。 | `docs/source/tutorials/models/DeepSeek-V4-Pro.md:158; docs/source/tutorials/models/DeepSeek-V4-Pro.md:234; docs/source/tutorials/models/GLM5.2.md:874; docs/source/tutorials/models/GLM5.2.md:924; docs/source/tutorials/models/MiniMax-M3.md:210; docs/source/tutorials/models/MiniMax-M3.md:253; docs/source/tutorials/models/MiniMax-M3.md:299; docs/source/tutorials/models/MiniMax-M3.md:343` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 可选调优/调试 | export | 设置 worker 执行一次模型计算的超时时间。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:81; docs/source/tutorials/models/DeepSeek-V3.1.md:439; docs/source/tutorials/models/DeepSeek-V3.1.md:514; docs/source/tutorials/models/DeepSeek-V3.1.md:589; docs/source/tutorials/models/DeepSeek-V3.1.md:662; docs/source/tutorials/models/DeepSeek-V4-Flash.md:494; docs/source/tutorials/models/DeepSeek-V4-Flash.md:562; docs/source/tutorials/models/DeepSeek-V4-Flash.md:642` |
| `VLLM_HOST_IP` | 场景必需 | export | 指定当前 vLLM 实例对其他节点可达的主机 IP。 | `docs/source/tutorials/models/GLM5.2.md:1067` |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | 可选调优/调试 | export | 设置 Mooncake 中止或清理请求的等待超时时间。 | `docs/source/tutorials/models/DeepSeek-V3.2.md:423; docs/source/tutorials/models/DeepSeek-V3.2.md:496; docs/source/tutorials/models/DeepSeek-V3.2.md:570; docs/source/tutorials/models/DeepSeek-V3.2.md:644; docs/source/tutorials/models/GLM4.x.md:520; docs/source/tutorials/models/GLM4.x.md:589; docs/source/tutorials/models/GLM5.2.md:1401; docs/source/tutorials/models/GLM5.2.md:1474` |
| `VLLM_PP_LAYER_PARTITION` | 可选调优/调试 | export | 自定义流水线并行各 stage 的模型层划分。 | `docs/source/user_guide/feature_guide/pipeline_parallel.md:192` |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | 可选调优/调试 | export | 设置前缀缓存保留或周期性清理的时间间隔。 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:237; docs/source/tutorials/models/DeepSeek-V4-Flash.md:281; docs/source/tutorials/models/DeepSeek-V4-Flash.md:505; docs/source/tutorials/models/DeepSeek-V4-Flash.md:653` |
| `VLLM_RPC_TIMEOUT` | 可选调优/调试 | export | 设置 vLLM 进程间 RPC 调用的超时时间。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:80; docs/source/tutorials/models/DeepSeek-V3.1.md:438; docs/source/tutorials/models/DeepSeek-V3.1.md:513; docs/source/tutorials/models/DeepSeek-V3.1.md:588; docs/source/tutorials/models/DeepSeek-V3.1.md:661; docs/source/tutorials/models/DeepSeek-V4-Flash.md:493; docs/source/tutorials/models/DeepSeek-V4-Flash.md:561; docs/source/tutorials/models/DeepSeek-V4-Flash.md:641` |
| `VLLM_SERVER_DEV_MODE` | 可选调优/调试 | export | 启用服务端开发模式，开放调试或开发用途的接口。 | `docs/source/user_guide/feature_guide/sleep_mode.md:132` |
| `VLLM_SLEEP_WHEN_IDLE` | 可选调优/调试 | 行内进程赋值 | 让空闲 worker 进入休眠，以减少资源占用或传输干扰。 | `docs/source/user_guide/feature_guide/netloader.md:57` |
| `VLLM_TORCH_PROFILER_WITH_STACK` | 可选调优/调试 | export | 控制 PyTorch Profiler 是否记录调用栈。 | `docs/source/tutorials/models/InternVL3.5.md:105; docs/source/tutorials/models/InternVL3.5.md:148; docs/source/tutorials/models/Qwen3-235B-A22B.md:410; docs/source/tutorials/models/Qwen3-235B-A22B.md:474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:330; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:410; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:492` |
| `VLLM_USE_MODELSCOPE` | 场景配置 | Docker -e/--env, Python os.environ, export | 让 vLLM 优先通过 ModelScope 下载或加载模型。 | `docs/source/developer_guide/evaluation/using_ais_bench.md:29; docs/source/developer_guide/evaluation/using_evalscope.md:27; docs/source/developer_guide/evaluation/using_lm_eval.md:29; docs/source/developer_guide/evaluation/using_opencompass.md:27; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:107; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:149; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:169; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:224` |
| `VLLM_USE_V1` | 可选调优/调试 | export | 控制是否使用 vLLM V1 引擎。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:74; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:279; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:334; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:389; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:444; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:502; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:557; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:612` |
| `VLLM_USE_V2_MODEL_RUNNER` | 可选调优/调试 | export | 控制是否使用上游 vLLM Model Runner V2。 | `docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:86` |
| `VLLM_WORKER_MULTIPROC_METHOD` | 可选调优/调试 | Python os.environ, export | 指定 vLLM worker 多进程的启动方式，如 spawn。 | `docs/source/tutorials/models/GLM5.2.md:1267; docs/source/tutorials/models/GLM5.2.md:1333; docs/source/tutorials/models/GLM5.2.md:1399; docs/source/tutorials/models/GLM5.2.md:1472; docs/source/user_guide/feature_guide/large_scale_ep.md:131; docs/source/user_guide/feature_guide/large_scale_ep.md:195; docs/source/user_guide/feature_guide/sleep_mode.md:133; docs/source/user_guide/feature_guide/sleep_mode.md:97` |

### 系统运行环境（4）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `LD_LIBRARY_PATH` | 场景配置 | export | 指定运行时搜索动态链接库的目录。 | `docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:213; docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md:140; docs/source/tutorials/models/DeepSeek-V3.1.md:451; docs/source/tutorials/models/DeepSeek-V3.1.md:526; docs/source/tutorials/models/DeepSeek-V3.1.md:601; docs/source/tutorials/models/DeepSeek-V3.1.md:674; docs/source/tutorials/models/GLM4.x.md:396; docs/source/tutorials/models/GLM4.x.md:459` |
| `LD_PRELOAD` | 可选调优/调试 | export | 在进程启动时优先加载指定动态库。 | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:72; docs/source/tutorials/models/DeepSeek-V4-Flash.md:154; docs/source/tutorials/models/DeepSeek-V4-Flash.md:197; docs/source/tutorials/models/DeepSeek-V4-Flash.md:233; docs/source/tutorials/models/DeepSeek-V4-Flash.md:277; docs/source/tutorials/models/DeepSeek-V4-Flash.md:503; docs/source/tutorials/models/DeepSeek-V4-Flash.md:558; docs/source/tutorials/models/DeepSeek-V4-Flash.md:651` |
| `PYTHONHASHSEED` | 可选调优/调试 | export | 固定 Python 哈希随机种子，提高多进程行为的可复现性。 | `docs/source/tutorials/models/GLM5.2.md:1082; docs/source/tutorials/models/GLM5.2.md:989; docs/source/tutorials/models/MiniMax-M2.md:300; docs/source/tutorials/models/MiniMax-M2.md:362; docs/source/user_guide/feature_guide/kv_pool.md:237; docs/source/user_guide/feature_guide/kv_pool.md:360; docs/source/user_guide/feature_guide/kv_pool.md:596; docs/source/user_guide/feature_guide/kv_pool.md:708` |
| `PYTHONPATH` | 场景配置 | export | 向 Python 模块搜索路径中添加目录。 | `docs/source/user_guide/feature_guide/kv_pool.md:236; docs/source/user_guide/feature_guide/kv_pool.md:357; docs/source/user_guide/feature_guide/ucm_deployment.md:256; docs/source/user_guide/feature_guide/ucm_deployment.md:304; docs/source/user_guide/feature_guide/ucm_deployment.md:526; docs/source/user_guide/feature_guide/ucm_deployment.md:646` |

### 其他服务运行变量（6）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `ACL_OP_INIT_MODE` | 可选调优/调试 | export | 控制 ACL 算子的初始化方式，通常用于调整算子加载和执行行为。 | `docs/source/tutorials/models/DeepSeek-V3.2.md:420; docs/source/tutorials/models/DeepSeek-V3.2.md:493; docs/source/tutorials/models/DeepSeek-V3.2.md:567; docs/source/tutorials/models/DeepSeek-V3.2.md:641; docs/source/tutorials/models/DeepSeek-V4-Pro.md:157; docs/source/tutorials/models/DeepSeek-V4-Pro.md:233; docs/source/tutorials/models/GLM4.x.md:392; docs/source/tutorials/models/GLM4.x.md:455` |
| `NETLOADER_CONFIG` | 场景必需 | export | 向 Netloader 模型加载器传递源端地址和设备等配置。 | `docs/source/user_guide/feature_guide/netloader.md:68` |
| `NPU_MEMORY_FRACTION` | 场景配置 | export | 限制或调整进程可使用的 NPU 显存比例。 | `docs/source/tutorials/models/gpt-oss-120b.md:104` |
| `RFORK_CONFIG` | 场景必需 | export | 向 RFork 模型加载器传递共享权重或实例启动配置。 | `docs/source/user_guide/feature_guide/rfork.md:124` |
| `TIKTOKEN_ENCODINGS_BASE` | 场景配置 | export | 指定 tiktoken 编码文件的本地目录或下载地址。 | `docs/source/tutorials/models/gpt-oss-120b.md:110` |
| `USE_MODELSCOPE_HUB` | 场景配置 | export | 让相关评测或模型工具从 ModelScope Hub 获取资源。 | `docs/source/developer_guide/evaluation/using_lm_eval.md:116` |

### 文档遗留/当前源码不支持（3）

| 变量 | 必要性 | 设置方式 | 说明 | 文档位置（示例） |
|---|---|---|---|---|
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | 不应使用 | export | 旧版 TopK/TopP 采样优化开关；当前源码已删除，不应使用。 | `docs/source/tutorials/models/GLM4.x.md:134; docs/source/tutorials/models/GLM4.x.md:184; docs/source/tutorials/models/GLM4.x.md:234; docs/source/tutorials/models/GLM4.x.md:394; docs/source/tutorials/models/GLM4.x.md:457; docs/source/tutorials/models/GLM4.x.md:523; docs/source/tutorials/models/GLM4.x.md:592` |
| `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED` | 不应使用 | export | 旧版 external DP 负载均衡开关；仅用于 0.9.1 兼容实现，当前不支持。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:132; docs/source/user_guide/feature_guide/large_scale_ep.md:196` |
| `VLLM_DP_SIZE_LOCAL` | 不应使用 | export | 旧版 external DP 的本机并行实例数；仅用于 0.9.1 兼容实现，当前不支持。 | `docs/source/user_guide/feature_guide/large_scale_ep.md:123; docs/source/user_guide/feature_guide/large_scale_ep.md:187` |

## 已排除的启动脚本辅助变量

这些名称出现在启动流程附近，但主要用于拼接镜像、模型路径、容器名、请求长度或端口，不是服务进程的配置接口。

| 辅助变量 | 文档位置（示例） |
|---|---|
| `DEVICE` | docs/source/developer_guide/evaluation/using_ais_bench.md:13; docs/source/developer_guide/evaluation/using_evalscope.md:11; docs/source/developer_guide/evaluation/using_lm_eval.md:13; docs/source/developer_guide/evaluation/using_opencompass.md:11; docs/source/quick_start.md:128; docs/source/quick_start.md:159 |
| `DEVICE0` | docs/source/quick_start.md:216; docs/source/quick_start.md:97 |
| `DEVICE1` | docs/source/quick_start.md:217; docs/source/quick_start.md:98 |
| `ENDPOINT` | docs/source/user_guide/deployment_guide/using_volcano_kthena.md:390 |
| `IFNAME` | docs/source/tutorials/models/DeepSeek-V4-Pro.md:147; docs/source/tutorials/models/DeepSeek-V4-Pro.md:223; docs/source/tutorials/models/MiniMax-M3.md:205; docs/source/tutorials/models/MiniMax-M3.md:248; docs/source/tutorials/models/MiniMax-M3.md:294; docs/source/tutorials/models/MiniMax-M3.md:338 |
| `IMAGE` | docs/source/developer_guide/evaluation/using_ais_bench.md:15; docs/source/developer_guide/evaluation/using_evalscope.md:13; docs/source/developer_guide/evaluation/using_lm_eval.md:15; docs/source/developer_guide/evaluation/using_opencompass.md:13; docs/source/quick_start.md:102; docs/source/quick_start.md:132 |
| `IP_ADDRESS` | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:320; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:400; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:482 |
| `MASTER_IP` | docs/source/developer_guide/contribution/doc_writing.md:150 |
| `MASTER_IP_ADDRESS` | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:399; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:481 |
| `MODEL` | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:394 |
| `MODEL_PATH` | docs/source/tutorials/models/Gemma4.md:82; docs/source/tutorials/models/Gemma4.md:98; docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:74; docs/source/tutorials/models/Hy3-preview.md:81; docs/source/tutorials/models/LLaVA-OneVision-Qwen2-0.5B-OV.md:57; docs/source/tutorials/models/PaddleOCR-VL.md:104 |
| `NAME` | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:23; docs/source/tutorials/models/DeepSeekOCR2.md:42; docs/source/tutorials/models/DeepSeekOCR2.md:76; docs/source/tutorials/models/Hy3-preview.md:35; docs/source/tutorials/models/MiniMax-M3.md:44 |
| `NETWORK_CARD_NAME` | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:321; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:401; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:483 |
| `PROFILING_SYMBOLS_PATH` | docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:163 |
| `SERVER_PORT` | docs/source/developer_guide/contribution/doc_writing.md:151; docs/source/developer_guide/contribution/doc_writing.md:91 |
| `SERVICE_PROF_CONFIG_PATH` | docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:162 |
| `SOC_VERSION` | docs/source/faqs.md:277; docs/source/faqs.md:280 |

## 口径与限制

1. 以文档中的实际启动链路为准，不扩展收录外部 vLLM/CANN 文档中可能支持但本仓库文档未使用的变量。
2. `export` 会影响后续子进程，因此即使它与 `vllm serve` 分处相邻代码块，只要属于同一部署流程也纳入。
3. Docker `-e/--env`、Kubernetes/Ray 启动所注入的变量属于服务环境；Docker 镜像构建参数不属于本文范围。
4. `curl` 通常不需要环境变量；只有推理客户端自身明确读取或继承的变量才纳入。
5. 文档示例可能包含可选性能参数。是否必须设置应以对应硬件、模型和部署拓扑为准。
