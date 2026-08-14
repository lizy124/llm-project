# vLLM Ascend `docs/source` 环境变量清单

- 扫描范围：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source` 下的 Git 跟踪文件（共 344 个）。
- 文档中发现的唯一环境变量名称：**152**。
- 纳入形式：`export`、命令行行内进程赋值、Docker `-e/--env`、Python 环境 API，以及明确写成“环境变量”的文档说明。`${VAR}`/`$VAR` 只有在同名变量已被上述方式确认时才作为补充证据。
- 示例中的 `IMAGE`、`DEVICE`、`TAG` 等也会记录，但单独归入“文档示例辅助变量”，不表示它们是 vLLM Ascend 产品配置。

## 分类统计

| 分类 | 数量 | 占比 | 说明 |
|---|---:|---:|---|
| vLLM Ascend 产品配置 | 17 | 11.2% | |
| Ascend/CANN/HCCL 与 NPU 运行时 | 29 | 19.1% | |
| 分布式启动与网络 | 7 | 4.6% | |
| KV Transfer 与外部存储 | 12 | 7.9% | |
| 上游 vLLM/PyTorch/模型生态 | 28 | 18.4% | |
| 系统与通用运行环境 | 6 | 3.9% | |
| 文档示例辅助变量 | 20 | 13.2% | |
| 模型与评测示例变量 | 6 | 3.9% | |
| CI/构建/发布辅助变量 | 4 | 2.6% | |
| 文档中的其他环境变量 | 23 | 15.1% | |
| **合计** | **152** | **100.0%** | 唯一变量名称，分类互斥 |

## 使用形式统计

同一变量可能有多种使用形式，因此本表不要求合计等于唯一变量总数。

| 使用形式 | 变量数 |
|---|---:|
| export | 119 |
| inline assignment | 43 |
| 正文明确提及 | 39 |
| shell variable reference | 30 |
| Python environment API | 12 |
| docker -e/--env | 6 |

## 文档目录分布

| 文档目录 | 变量出现次数（去重变量-目录组合） |
|---|---:|
| `user_guide` | 76 |
| `tutorials` | 60 |
| `developer_guide` | 41 |
| `locale` | 41 |
| `faqs.md` | 10 |
| `installation.md` | 8 |
| `quick_start.md` | 5 |
| `community` | 1 |
| `_templates` | 1 |

## 分类明细

位置使用 `docs/source` 下的相对路径和行号。每个变量最多列出 6 个代表位置；完整命中可通过仓库搜索复核。

### vLLM Ascend 产品配置（17）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `DYNAMIC_EPLB` | export, inline assignment | docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:257 |
| `MSMONITOR_USE_DAEMON` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:376; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:379; docs/source/user_guide/configuration/additional_config.md:87 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | Python environment API, export, inline assignment, 正文明确提及 | docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md:360; docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md:41; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/balance_schedule_refactor.po:493; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/balance_schedule_refactor.po:676; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/balance_schedule_refactor.po:83; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/GLM5.2.po:357 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:401; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:405; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:741; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:745; docs/source/user_guide/release_notes.md:252 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | export, inline assignment, 正文明确提及 | docs/source/developer_guide/Design_Documents/context_parallel.md:80; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:367; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:371; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:78; docs/source/tutorials/features/suffix_speculative_decoding.md:86; docs/source/tutorials/models/DeepSeek-V3.1.md:453 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | export, inline assignment, 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:413; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:79; docs/source/tutorials/models/DeepSeek-V4-Pro.md:596; docs/source/tutorials/models/DeepSeek-V4-Pro.md:667; docs/source/tutorials/models/GLM4.x.md:524; docs/source/tutorials/models/GLM4.x.md:593 |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2873 |
| `VLLM_ASCEND_ENABLE_MLAPO` | export, 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:384; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:388; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2174; docs/source/tutorials/models/DeepSeek-V3.2.md:134; docs/source/tutorials/models/DeepSeek-V3.2.md:187; docs/source/tutorials/models/DeepSeek-V3.2.md:234 |
| `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2645; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2672; docs/source/user_guide/release_notes.md:1490 |
| `VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2646; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2874; docs/source/user_guide/release_notes.md:1491 |
| `VLLM_ASCEND_ENABLE_NZ` | Python environment API, export, inline assignment, 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:392; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:396; docs/source/tutorials/models/DeepSeekOCR2.md:127; docs/source/tutorials/models/GLM5.2.md:1258; docs/source/tutorials/models/GLM5.2.md:1324; docs/source/tutorials/models/GLM5.2.md:1390 |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | export, inline assignment | docs/source/tutorials/models/GLM4.x.md:134; docs/source/tutorials/models/GLM4.x.md:184; docs/source/tutorials/models/GLM4.x.md:234; docs/source/tutorials/models/GLM4.x.md:394; docs/source/tutorials/models/GLM4.x.md:457; docs/source/tutorials/models/GLM4.x.md:523 |
| `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2366; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:3051; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:3061; docs/source/user_guide/release_notes.md:1773 |
| `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:132; docs/source/user_guide/feature_guide/large_scale_ep.md:196 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:418; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po:422; docs/source/user_guide/configuration/additional_config.md:91 |
| `VLLM_ASCEND_MLA_PA` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2872 |
| `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/profile_execute_duration.po:45; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/profile_execute_duration.po:47 |

### Ascend/CANN/HCCL 与 NPU 运行时（29）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `ASCEND_A3_ENABLE` | export | docs/source/tutorials/models/DeepSeek-V3.2.md:421; docs/source/tutorials/models/DeepSeek-V3.2.md:494; docs/source/tutorials/models/DeepSeek-V3.2.md:568; docs/source/tutorials/models/DeepSeek-V3.2.md:642; docs/source/tutorials/models/GLM4.x.md:393; docs/source/tutorials/models/GLM4.x.md:456 |
| `ASCEND_AGGREGATE_ENABLE` | export | docs/source/tutorials/models/DeepSeek-V3.2.md:418; docs/source/tutorials/models/DeepSeek-V3.2.md:491; docs/source/tutorials/models/DeepSeek-V3.2.md:565; docs/source/tutorials/models/DeepSeek-V3.2.md:639; docs/source/tutorials/models/GLM4.x.md:390; docs/source/tutorials/models/GLM4.x.md:453 |
| `ASCEND_CONNECT_TIMEOUT` | export | docs/source/tutorials/models/DeepSeek-V4-Pro.md:167; docs/source/tutorials/models/DeepSeek-V4-Pro.md:243; docs/source/tutorials/models/MiniMax-M3.md:212; docs/source/tutorials/models/MiniMax-M3.md:255; docs/source/tutorials/models/MiniMax-M3.md:301; docs/source/tutorials/models/MiniMax-M3.md:345 |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | export | docs/source/user_guide/feature_guide/kv_pool.md:1221; docs/source/user_guide/feature_guide/kv_pool.md:159; docs/source/user_guide/feature_guide/kv_pool.md:242; docs/source/user_guide/feature_guide/kv_pool.md:363 |
| `ASCEND_GLOBAL_RESOURCE_CONFIG` | export | docs/source/user_guide/feature_guide/kv_pool.md:1229 |
| `ASCEND_LAUNCH_BLOCKING` | export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:76 |
| `ASCEND_RT_VISIBLE_DEVICES` | docker -e/--env, export, inline assignment, 正文明确提及 | docs/source/developer_guide/contribution/doc_writing.md:240; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/features/ray.po:92; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/features/ray.po:97; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:280; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:335; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:390 |
| `ASCEND_TOOLKIT_HOME` | export | docs/source/installation.md:114; docs/source/installation.md:279 |
| `ASCEND_TOTAL_MEMORY_GB` | docker -e/--env | docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:40 |
| `ASCEND_TRANSFER_TIMEOUT` | export, inline assignment | docs/source/tutorials/models/DeepSeek-V4-Pro.md:168; docs/source/tutorials/models/DeepSeek-V4-Pro.md:244; docs/source/tutorials/models/MiniMax-M3.md:213; docs/source/tutorials/models/MiniMax-M3.md:256; docs/source/tutorials/models/MiniMax-M3.md:302; docs/source/tutorials/models/MiniMax-M3.md:346 |
| `ASCEND_TRANSPORT_PRINT` | export | docs/source/tutorials/models/DeepSeek-V3.2.md:419; docs/source/tutorials/models/DeepSeek-V3.2.md:492; docs/source/tutorials/models/DeepSeek-V3.2.md:566; docs/source/tutorials/models/DeepSeek-V3.2.md:640; docs/source/tutorials/models/GLM4.x.md:391; docs/source/tutorials/models/GLM4.x.md:454 |
| `ASCEND_VISIBLE_DEVICES` | docker -e/--env | docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:38 |
| `ATB_LLM_LCOC_ENABLE` | export | docs/source/faqs.md:199 |
| `ATB_MATMUL_SHUFFLE_K_ENABLE` | export | docs/source/faqs.md:198 |
| `CPU_AFFINITY_CONF` | export | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:128; docs/source/tutorials/models/GLM5.2.md:873; docs/source/tutorials/models/GLM5.2.md:923; docs/source/tutorials/models/PaddleOCR-VL.md:106 |
| `HCCL_BUFFSIZE` | Python environment API, export, inline assignment, 正文明确提及 | docs/source/developer_guide/contribution/doc_writing.md:239; docs/source/developer_guide/contribution/doc_writing.md:90; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:3841; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:3848; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:71; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:276 |
| `HCCL_CONNECT_TIMEOUT` | export, inline assignment | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:83; docs/source/tutorials/models/DeepSeek-V3.1.md:441; docs/source/tutorials/models/DeepSeek-V3.1.md:516; docs/source/tutorials/models/DeepSeek-V3.1.md:591; docs/source/tutorials/models/DeepSeek-V3.1.md:664; docs/source/tutorials/models/DeepSeek-V3.2.md:288 |
| `HCCL_DETERMINISTIC` | export | docs/source/faqs.md:197 |
| `HCCL_EXEC_TIMEOUT` | export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:82; docs/source/tutorials/models/DeepSeek-V3.1.md:440; docs/source/tutorials/models/DeepSeek-V3.1.md:515; docs/source/tutorials/models/DeepSeek-V3.1.md:590; docs/source/tutorials/models/DeepSeek-V3.1.md:663; docs/source/tutorials/models/DeepSeek-V4-Flash.md:495 |
| `HCCL_IF_IP` | export | docs/source/developer_guide/contribution/doc_writing.md:221; docs/source/developer_guide/contribution/doc_writing.md:234; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:269; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:324; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:379; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:434 |
| `HCCL_INTRA_PCIE_ENABLE` | export | docs/source/tutorials/models/DeepSeek-R1.md:247; docs/source/tutorials/models/DeepSeek-R1.md:293; docs/source/tutorials/models/DeepSeek-V3.1.md:268; docs/source/tutorials/models/DeepSeek-V3.1.md:321; docs/source/tutorials/models/DeepSeek-V3.2.md:289; docs/source/tutorials/models/DeepSeek-V3.2.md:340 |
| `HCCL_INTRA_ROCE_ENABLE` | export, inline assignment | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:217; docs/source/tutorials/models/DeepSeek-R1.md:248; docs/source/tutorials/models/DeepSeek-R1.md:294; docs/source/tutorials/models/DeepSeek-V3.1.md:269; docs/source/tutorials/models/DeepSeek-V3.1.md:322; docs/source/tutorials/models/DeepSeek-V3.2.md:290 |
| `HCCL_OP_EXPANSION_MODE` | Python environment API, export, inline assignment | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:140; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:73; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:278; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:333; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:388; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:443 |
| `HCCL_RDMA_TIMEOUT` | export | docs/source/user_guide/feature_guide/kv_pool.md:170; docs/source/user_guide/feature_guide/kv_pool.md:249; docs/source/user_guide/feature_guide/kv_pool.md:370 |
| `HCCL_SOCKET_IFNAME` | export | docs/source/developer_guide/contribution/doc_writing.md:224; docs/source/developer_guide/contribution/doc_writing.md:237; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:272; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:327; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:382; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:437 |
| `HCCL_TRANSFER_TIMEOUT` | export | docs/source/tutorials/models/GLM5.2.md:1263; docs/source/tutorials/models/GLM5.2.md:1329; docs/source/tutorials/models/GLM5.2.md:136; docs/source/tutorials/models/GLM5.2.md:1395; docs/source/tutorials/models/GLM5.2.md:1468; docs/source/tutorials/models/GLM5.2.md:213 |
| `LCCL_DETERMINISTIC` | export | docs/source/faqs.md:196 |
| `PYTORCH_NPU_ALLOC_CONF` | Python environment API, docker -e/--env, export, inline assignment, 正文明确提及 | docs/source/developer_guide/evaluation/using_ais_bench.md:30; docs/source/developer_guide/evaluation/using_evalscope.md:28; docs/source/developer_guide/evaluation/using_lm_eval.md:175; docs/source/developer_guide/evaluation/using_lm_eval.md:30; docs/source/developer_guide/evaluation/using_opencompass.md:28; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:111 |
| `TASK_QUEUE_ENABLE` | Python environment API, export, inline assignment | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:125; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:75; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:277; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:332; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:387; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:442 |

### 分布式启动与网络（7）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `GLOO_SOCKET_IFNAME` | export | docs/source/developer_guide/contribution/doc_writing.md:222; docs/source/developer_guide/contribution/doc_writing.md:235; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:270; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:325; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:380; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:435 |
| `TP_SOCKET_IFNAME` | export | docs/source/developer_guide/contribution/doc_writing.md:223; docs/source/developer_guide/contribution/doc_writing.md:236; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:271; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:326; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:381; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:436 |
| `VLLM_DP_MASTER_IP` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:119; docs/source/user_guide/feature_guide/large_scale_ep.md:183 |
| `VLLM_DP_MASTER_PORT` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:120; docs/source/user_guide/feature_guide/large_scale_ep.md:184 |
| `VLLM_DP_RANK` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:122; docs/source/user_guide/feature_guide/large_scale_ep.md:186 |
| `VLLM_DP_RANK_LOCAL` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:121; docs/source/user_guide/feature_guide/large_scale_ep.md:185 |
| `VLLM_DP_SIZE` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:118; docs/source/user_guide/feature_guide/large_scale_ep.md:182 |

### KV Transfer 与外部存储（12）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `DATASYSTEM_CLIENT_LOG_DIR` | export, shell variable reference | docs/source/user_guide/feature_guide/kv_pool.md:948; docs/source/user_guide/feature_guide/kv_pool.md:950 |
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | export | docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:199 |
| `MMC_LOCAL_CONFIG_PATH` | export, inline assignment | docs/source/user_guide/feature_guide/kv_pool.md:562; docs/source/user_guide/feature_guide/kv_pool.md:684; docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:111 |
| `MMC_META_CONFIG_PATH` | export, inline assignment | docs/source/user_guide/feature_guide/kv_pool.md:533; docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:110 |
| `MOONCAKE_CONFIG_PATH` | export | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:215; docs/source/tutorials/models/GLM5.2.md:1083; docs/source/tutorials/models/GLM5.2.md:990; docs/source/user_guide/feature_guide/kv_pool.md:155; docs/source/user_guide/feature_guide/kv_pool.md:238; docs/source/user_guide/feature_guide/kv_pool.md:358 |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | export | docs/source/user_guide/feature_guide/kv_pool.md:475 |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | export | docs/source/user_guide/feature_guide/kv_pool.md:474 |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | export | docs/source/user_guide/feature_guide/kv_pool.md:1222; docs/source/user_guide/feature_guide/kv_pool.md:1247; docs/source/user_guide/feature_guide/kv_pool.md:476 |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | export | docs/source/user_guide/feature_guide/kv_pool.md:473 |
| `NETLOADER_CONFIG` | export, shell variable reference | docs/source/user_guide/feature_guide/netloader.md:68; docs/source/user_guide/feature_guide/netloader.md:77 |
| `RFORK_CONFIG` | export, shell variable reference | docs/source/user_guide/feature_guide/rfork.md:124; docs/source/user_guide/feature_guide/rfork.md:135 |
| `YR_CONFIG_PATH` | export | docs/source/user_guide/feature_guide/kv_pool.md:947 |

### 上游 vLLM/PyTorch/模型生态（28）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `HF_DATASETS_CACHE` | Python environment API | docs/source/tutorials/models/Qwen3-Embedding.md:222; docs/source/tutorials/models/Qwen3-Reranker.md:251; docs/source/tutorials/models/Qwen3-VL-Embedding.md:226; docs/source/tutorials/models/Qwen3-VL-Reranker.md:256 |
| `HF_DATASETS_OFFLINE` | export | docs/source/developer_guide/evaluation/using_lm_eval.md:227 |
| `HF_ENDPOINT` | Python environment API, export, inline assignment | docs/source/developer_guide/evaluation/using_lm_eval.md:115; docs/source/developer_guide/evaluation/using_lm_eval.md:186; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:177; docs/source/tutorials/models/Qwen3-Embedding.md:223; docs/source/tutorials/models/Qwen3-Reranker.md:252; docs/source/tutorials/models/Qwen3-VL-Embedding.md:227 |
| `HF_HOME` | export | docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:73 |
| `TOKENIZERS_PARALLELISM` | export, inline assignment | docs/source/tutorials/models/DeepSeekOCR2.md:128; docs/source/tutorials/models/DeepSeekOCR2.md:131 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | export, inline assignment | docs/source/developer_guide/contribution/testing.md:152; docs/source/installation.md:280 |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | export, inline assignment | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:77; docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:103; docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:56 |
| `VLLM_BATCH_INVARIANT` | Python environment API, export, inline assignment, 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/batch_invariance.po:75; docs/source/user_guide/feature_guide/batch_invariance.md:35; docs/source/user_guide/feature_guide/batch_invariance.md:38; docs/source/user_guide/feature_guide/batch_invariance.md:45; docs/source/user_guide/feature_guide/batch_invariance.md:79; docs/source/user_guide/feature_guide/flash_attention.md:102 |
| `VLLM_DP_SIZE_LOCAL` | export | docs/source/user_guide/feature_guide/large_scale_ep.md:123; docs/source/user_guide/feature_guide/large_scale_ep.md:187 |
| `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2576; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:3050; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:3060; docs/source/user_guide/release_notes.md:1772 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | export, inline assignment | docs/source/tutorials/models/DeepSeek-V4-Pro.md:158; docs/source/tutorials/models/DeepSeek-V4-Pro.md:234; docs/source/tutorials/models/GLM5.2.md:874; docs/source/tutorials/models/GLM5.2.md:924; docs/source/tutorials/models/MiniMax-M3.md:210; docs/source/tutorials/models/MiniMax-M3.md:253 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:81; docs/source/tutorials/models/DeepSeek-V3.1.md:439; docs/source/tutorials/models/DeepSeek-V3.1.md:514; docs/source/tutorials/models/DeepSeek-V3.1.md:589; docs/source/tutorials/models/DeepSeek-V3.1.md:662; docs/source/tutorials/models/DeepSeek-V4-Flash.md:494 |
| `VLLM_HOST_IP` | export | docs/source/tutorials/models/GLM5.2.md:1067 |
| `VLLM_LLMDD_RPC_PORT` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2644; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2671; docs/source/user_guide/release_notes.md:1489 |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | export, inline assignment | docs/source/tutorials/models/DeepSeek-V3.2.md:423; docs/source/tutorials/models/DeepSeek-V3.2.md:496; docs/source/tutorials/models/DeepSeek-V3.2.md:570; docs/source/tutorials/models/DeepSeek-V3.2.md:644; docs/source/tutorials/models/GLM4.x.md:520; docs/source/tutorials/models/GLM4.x.md:589 |
| `VLLM_PP_LAYER_PARTITION` | export, inline assignment | docs/source/user_guide/feature_guide/pipeline_parallel.md:192; docs/source/user_guide/feature_guide/pipeline_parallel.md:299 |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | export, inline assignment | docs/source/tutorials/models/DeepSeek-V4-Flash.md:237; docs/source/tutorials/models/DeepSeek-V4-Flash.md:281; docs/source/tutorials/models/DeepSeek-V4-Flash.md:505; docs/source/tutorials/models/DeepSeek-V4-Flash.md:653 |
| `VLLM_RPC_TIMEOUT` | export, inline assignment | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:80; docs/source/tutorials/models/DeepSeek-V3.1.md:438; docs/source/tutorials/models/DeepSeek-V3.1.md:513; docs/source/tutorials/models/DeepSeek-V3.1.md:588; docs/source/tutorials/models/DeepSeek-V3.1.md:661; docs/source/tutorials/models/DeepSeek-V4-Flash.md:493 |
| `VLLM_SERVER_DEV_MODE` | export | docs/source/user_guide/feature_guide/sleep_mode.md:132 |
| `VLLM_SLEEP_WHEN_IDLE` | inline assignment | docs/source/user_guide/feature_guide/netloader.md:56 |
| `VLLM_TARGET_DEVICE` | docker -e/--env, inline assignment | docs/source/developer_guide/contribution/testing.md:47; docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:41 |
| `VLLM_TORCH_PROFILER_DIR` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2094 |
| `VLLM_TORCH_PROFILER_WITH_STACK` | export | docs/source/tutorials/models/InternVL3.5.md:105; docs/source/tutorials/models/InternVL3.5.md:148; docs/source/tutorials/models/Qwen3-235B-A22B.md:410; docs/source/tutorials/models/Qwen3-235B-A22B.md:474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:330; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:410 |
| `VLLM_USE_MODELSCOPE` | Python environment API, docker -e/--env, export, inline assignment | docs/source/developer_guide/contribution/testing.md:195; docs/source/developer_guide/contribution/testing.md:198; docs/source/developer_guide/contribution/testing.md:201; docs/source/developer_guide/contribution/testing.md:209; docs/source/developer_guide/contribution/testing.md:212; docs/source/developer_guide/contribution/testing.md:215 |
| `VLLM_USE_V1` | export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:74; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:279; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:334; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:389; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:444; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:502 |
| `VLLM_USE_V2_MODEL_RUNNER` | export | docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:86 |
| `VLLM_VERSION` | 正文明确提及 | docs/source/community/versioning_policy.md:182; docs/source/developer_guide/Design_Documents/patch.md:75; docs/source/faqs.md:135; docs/source/locale/zh_CN/LC_MESSAGES/community/versioning_policy.po:166; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/patch.po:107; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/Design_Documents/patch.po:110 |
| `VLLM_WORKER_MULTIPROC_METHOD` | Python environment API, export | docs/source/tutorials/models/GLM5.2.md:1267; docs/source/tutorials/models/GLM5.2.md:1333; docs/source/tutorials/models/GLM5.2.md:1399; docs/source/tutorials/models/GLM5.2.md:1472; docs/source/user_guide/feature_guide/large_scale_ep.md:131; docs/source/user_guide/feature_guide/large_scale_ep.md:195 |

### 系统与通用运行环境（6）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `LD_LIBRARY_PATH` | export, inline assignment, shell variable reference | docs/source/developer_guide/contribution/testing.md:151; docs/source/developer_guide/contribution/testing.md:52; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/GLM5.po:503; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/GLM5.po:508; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:213; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:214 |
| `LD_PRELOAD` | export, inline assignment, shell variable reference | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:78; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:95; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:96; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:72; docs/source/tutorials/models/DeepSeek-V3.1.md:145; docs/source/tutorials/models/DeepSeek-V3.1.md:257 |
| `OMP_NUM_THREADS` | export, inline assignment | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:70; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:274; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:329; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:384; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:439; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:497 |
| `OMP_PROC_BIND` | Python environment API, export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:68; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:273; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:328; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:383; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:438; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:496 |
| `PYTHONHASHSEED` | export | docs/source/tutorials/models/GLM5.2.md:1082; docs/source/tutorials/models/GLM5.2.md:989; docs/source/tutorials/models/MiniMax-M2.md:300; docs/source/tutorials/models/MiniMax-M2.md:362; docs/source/user_guide/feature_guide/kv_pool.md:153; docs/source/user_guide/feature_guide/kv_pool.md:237 |
| `PYTHONPATH` | export, shell variable reference | docs/source/user_guide/feature_guide/kv_pool.md:154; docs/source/user_guide/feature_guide/kv_pool.md:236; docs/source/user_guide/feature_guide/kv_pool.md:357; docs/source/user_guide/feature_guide/ucm_deployment.md:256; docs/source/user_guide/feature_guide/ucm_deployment.md:304; docs/source/user_guide/feature_guide/ucm_deployment.md:526 |

### 文档示例辅助变量（20）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `CONFIG_BASE_PATH` | export | docs/source/developer_guide/contribution/multi_node_test.md:402; docs/source/developer_guide/contribution/multi_node_test.md:415; docs/source/developer_guide/contribution/multi_node_test.md:489; docs/source/developer_guide/contribution/multi_node_test.md:502 |
| `CONFIG_YAML_PATH` | export | docs/source/developer_guide/contribution/multi_node_test.md:401; docs/source/developer_guide/contribution/multi_node_test.md:414; docs/source/developer_guide/contribution/multi_node_test.md:490; docs/source/developer_guide/contribution/multi_node_test.md:503; docs/source/developer_guide/contribution/testing.md:247 |
| `DEVICE` | export, shell variable reference | docs/source/developer_guide/contribution/testing.md:65; docs/source/developer_guide/contribution/testing.md:71; docs/source/developer_guide/evaluation/using_ais_bench.md:13; docs/source/developer_guide/evaluation/using_ais_bench.md:19; docs/source/developer_guide/evaluation/using_evalscope.md:11; docs/source/developer_guide/evaluation/using_evalscope.md:17 |
| `IMAGE` | export, inline assignment, shell variable reference | docs/source/_templates/Model-Deployment-Tutorial-Template.md:89; docs/source/_templates/Model-Deployment-Tutorial-Template.md:96; docs/source/_templates/Model-Deployment-Tutorial-Template.zh.md:89; docs/source/_templates/Model-Deployment-Tutorial-Template.zh.md:96; docs/source/_templates/template-supplement.md:119; docs/source/_templates/template-supplement.md:127 |
| `IMAGE_NAME` | export, shell variable reference | docs/source/faqs.md:246; docs/source/faqs.md:251; docs/source/faqs.md:252 |
| `IMAGE_TAG` | export, shell variable reference | docs/source/faqs.md:245; docs/source/faqs.md:246; docs/source/faqs.md:248; docs/source/faqs.md:252 |
| `IP_ADDRESS` | export, shell variable reference | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:320; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:322; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:336; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:343; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:400; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:402 |
| `MASTER_IP` | export, shell variable reference | docs/source/developer_guide/contribution/doc_writing.md:125; docs/source/developer_guide/contribution/doc_writing.md:150; docs/source/developer_guide/contribution/doc_writing.md:159 |
| `MASTER_IP_ADDRESS` | export, shell variable reference | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:399; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:424; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:481; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:506 |
| `MODEL` | export, inline assignment, shell variable reference | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:394; docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:395; docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:400 |
| `MODEL_PATH` | export, inline assignment, shell variable reference, 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/PaddleOCR-VL.po:122; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/PaddleOCR-VL.po:126; docs/source/tutorials/models/Gemma4.md:100; docs/source/tutorials/models/Gemma4.md:82; docs/source/tutorials/models/Gemma4.md:84; docs/source/tutorials/models/Gemma4.md:98 |
| `NAME` | export, inline assignment, shell variable reference | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:23; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:26; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:84; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:90; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:128; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:133 |
| `NETWORK_CARD_NAME` | export, shell variable reference | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:321; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:323; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:324; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:325; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:401; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:403 |
| `PROFILING_SYMBOLS_PATH` | export, 正文明确提及 | docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:157; docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:163; docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:171; docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:262; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/service_profiling_guide.po:101; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/service_profiling_guide.po:184 |
| `SAVE_PATH` | export, shell variable reference | docs/source/user_guide/feature_guide/quantization.md:49; docs/source/user_guide/feature_guide/quantization.md:53 |
| `SERVER_PORT` | export, inline assignment, shell variable reference | docs/source/developer_guide/contribution/doc_writing.md:112; docs/source/developer_guide/contribution/doc_writing.md:121; docs/source/developer_guide/contribution/doc_writing.md:151; docs/source/developer_guide/contribution/doc_writing.md:155; docs/source/developer_guide/contribution/doc_writing.md:208; docs/source/developer_guide/contribution/doc_writing.md:60 |
| `SERVICE_PROF_CONFIG_PATH` | export, 正文明确提及 | docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:157; docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:162; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/service_profiling_guide.po:85 |
| `SRC_WORKSPACE` | export, shell variable reference | docs/source/developer_guide/contribution/testing.md:35; docs/source/developer_guide/contribution/testing.md:36; docs/source/developer_guide/contribution/testing.md:37; docs/source/developer_guide/contribution/testing.md:46; docs/source/developer_guide/contribution/testing.md:51 |
| `TAG` | inline assignment, shell variable reference | docs/source/faqs.md:39; docs/source/faqs.md:41; docs/source/faqs.md:43; docs/source/faqs.md:55; docs/source/faqs.md:58; docs/source/faqs.md:66 |
| `WORKSPACE` | export, shell variable reference | docs/source/developer_guide/contribution/multi_node_test.md:399; docs/source/developer_guide/contribution/multi_node_test.md:405; docs/source/developer_guide/contribution/multi_node_test.md:412; docs/source/developer_guide/contribution/multi_node_test.md:418; docs/source/developer_guide/contribution/multi_node_test.md:487; docs/source/developer_guide/contribution/multi_node_test.md:493 |

### 模型与评测示例变量（6）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `ACL_OP_INIT_MODE` | export, inline assignment | docs/source/tutorials/models/DeepSeek-V3.2.md:420; docs/source/tutorials/models/DeepSeek-V3.2.md:493; docs/source/tutorials/models/DeepSeek-V3.2.md:567; docs/source/tutorials/models/DeepSeek-V3.2.md:641; docs/source/tutorials/models/DeepSeek-V4-Pro.md:157; docs/source/tutorials/models/DeepSeek-V4-Pro.md:233 |
| `DATASET_SOURCE` | export | docs/source/developer_guide/evaluation/using_opencompass.md:65 |
| `IFNAME` | export, shell variable reference | docs/source/tutorials/models/DeepSeek-V4-Pro.md:147; docs/source/tutorials/models/DeepSeek-V4-Pro.md:148; docs/source/tutorials/models/DeepSeek-V4-Pro.md:149; docs/source/tutorials/models/DeepSeek-V4-Pro.md:150; docs/source/tutorials/models/DeepSeek-V4-Pro.md:223; docs/source/tutorials/models/DeepSeek-V4-Pro.md:224 |
| `NPU_MEMORY_FRACTION` | export | docs/source/tutorials/models/gpt-oss-120b.md:104 |
| `TIKTOKEN_ENCODINGS_BASE` | export, inline assignment | docs/source/tutorials/models/gpt-oss-120b.md:110; docs/source/tutorials/models/gpt-oss-120b.md:88 |
| `USE_MODELSCOPE_HUB` | export | docs/source/developer_guide/evaluation/using_lm_eval.md:116; docs/source/developer_guide/evaluation/using_lm_eval.md:187 |

### CI/构建/发布辅助变量（4）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `AIS_BENCH_TAG` | export, shell variable reference | docs/source/developer_guide/contribution/multi_node_test.md:379; docs/source/developer_guide/contribution/multi_node_test.md:383; docs/source/developer_guide/contribution/multi_node_test.md:462; docs/source/developer_guide/contribution/multi_node_test.md:466 |
| `AIS_BENCH_URL` | export, shell variable reference | docs/source/developer_guide/contribution/multi_node_test.md:380; docs/source/developer_guide/contribution/multi_node_test.md:383; docs/source/developer_guide/contribution/multi_node_test.md:463; docs/source/developer_guide/contribution/multi_node_test.md:466 |
| `BENCHMARK_HOME` | export, shell variable reference | docs/source/developer_guide/contribution/multi_node_test.md:381; docs/source/developer_guide/contribution/multi_node_test.md:383; docs/source/developer_guide/contribution/multi_node_test.md:384; docs/source/developer_guide/contribution/multi_node_test.md:464; docs/source/developer_guide/contribution/multi_node_test.md:466; docs/source/developer_guide/contribution/multi_node_test.md:467 |
| `PIP_EXTRA_INDEX_URL` | export | docs/source/developer_guide/contribution/testing.md:32 |

### 文档中的其他环境变量（23）

| 变量 | 使用形式 | 文档位置（示例） |
|---|---|---|
| `AscendConfig` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:836; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:839; docs/source/user_guide/release_notes.md:320 |
| `COMPILE_CUSTOM_KERNELS` | export, 正文明确提及 | docs/source/installation.md:281; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2339 |
| `DEVICE0` | export, shell variable reference | docs/source/quick_start.md:106; docs/source/quick_start.md:216; docs/source/quick_start.md:225; docs/source/quick_start.md:97 |
| `DEVICE1` | export, shell variable reference | docs/source/quick_start.md:107; docs/source/quick_start.md:217; docs/source/quick_start.md:226; docs/source/quick_start.md:98 |
| `DEVICE_LIST` | inline assignment, shell variable reference | docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:29; docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:38; docs/source/user_guide/feature_guide/lmcache_ascend_deployment.md:39 |
| `ENDPOINT` | export, shell variable reference | docs/source/user_guide/deployment_guide/using_volcano_kthena.md:390; docs/source/user_guide/deployment_guide/using_volcano_kthena.md:392 |
| `EXTERNAL_DP_LOG_DIR` | export | docs/source/developer_guide/contribution/multi_node_test.md:538 |
| `HCCN_PATH` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2643; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2670; docs/source/user_guide/release_notes.md:1488 |
| `INSTALLER_DOWNLOAD_URL` | inline assignment | docs/source/installation.md:176 |
| `IS_PR_TEST` | export | docs/source/developer_guide/contribution/multi_node_test.md:400; docs/source/developer_guide/contribution/multi_node_test.md:413; docs/source/developer_guide/contribution/multi_node_test.md:488; docs/source/developer_guide/contribution/multi_node_test.md:501 |
| `InvalidVersion` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/community/versioning_policy.po:170 |
| `LOCAL_MEDIA_PATH` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/Qwen2.5-Omni.po:106; docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/Qwen2.5-Omni.po:111 |
| `LWS_WORKER_INDEX` | export | docs/source/developer_guide/contribution/multi_node_test.md:403; docs/source/developer_guide/contribution/multi_node_test.md:416; docs/source/developer_guide/contribution/multi_node_test.md:491; docs/source/developer_guide/contribution/multi_node_test.md:504 |
| `MOE_ALL2ALL_BUFFER` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:2673 |
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | export | docs/source/tutorials/features/ray.md:115; docs/source/tutorials/features/ray.md:127 |
| `ROCE` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/kv_pool.po:1446; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/kv_pool.po:1452; docs/source/user_guide/feature_guide/kv_pool.md:1080 |
| `SOC_VERSION` | export, inline assignment, 正文明确提及 | docs/source/developer_guide/contribution/testing.md:55; docs/source/faqs.md:271; docs/source/faqs.md:274; docs/source/faqs.md:277; docs/source/faqs.md:280; docs/source/installation.md:282 |
| `USE_OPTIMIZED_MODEL` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:1814; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/release_notes.po:981 |
| `eplb_config` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/expert_parallelism_load_balancer.po:87; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/expert_parallelism_load_balancer.po:88; docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:161 |
| `expert_map_path` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/eplb_swift_balancer.po:151; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/eplb_swift_balancer.po:155; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/expert_parallelism_load_balancer.po:57; docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/expert_parallelism_load_balancer.po:60; docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:168 |
| `false` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/kv_pool.po:1410 |
| `torch_npu` | 正文明确提及 | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:105; docs/source/locale/zh_CN/LC_MESSAGES/developer_guide/performance_and_debug/optimization_and_tuning.po:60 |
| `true` | 正文明确提及 | docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/kv_pool.po:859; docs/source/user_guide/feature_guide/kv_pool.md:449 |

## 说明

1. 本文只统计 `docs/source`，不包含仓库源码、测试、CI 或其他目录中的独立命中。
2. 文档中的环境变量可能只是外部组件（CANN/HCCL/PyTorch/Hugging Face）的配置，仓库不一定读取它们。
3. `${PWD}`、`${HOME}` 等系统变量在文档中被引用时也保留，因为它们确实参与了示例命令的环境传播；普通 Shell 局部变量若没有环境变量语义则不纳入。
4. 变量实际作用、默认值和源码读取位置请结合主清单 `vllm-ascend-environment-variables.md` 一起查看。
