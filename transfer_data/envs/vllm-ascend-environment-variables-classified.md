# vLLM Ascend 环境变量分类视图

- 仓库版本：`8f89feb9d`
- 唯一名称总数：**378**
- 产品中央配置：**19**
- 被 Python/C/C++ 直接访问：**152**
- 名称疑似敏感（Token/Key/Secret/Password）：**18**

> 统计对象是“仓库中具有环境变量语义的唯一名称”，并不表示这些名称全部是 vLLM Ascend 的公开配置。总数包含运行时环境变量、CI/Kubernetes `env`、测试输入，以及单列标注的 Docker `ARG`。

## 如何阅读

- 部署和调优：优先看第 1 至第 5 类。
- 编译和制作镜像：看第 6 类。
- 排查宿主机或动态库问题：看第 7 类。
- 开发者和仓库维护者：看第 8、9、11 类。
- 第 10 类通常只是示例中的临时占位变量，不应当当作产品接口。

每个变量只分配一个“主分类”，因此分类数量可以直接相加得到总数；范围和机制统计允许一个变量重复计入。

## 主分类统计

| 主分类 | 数量 | 占比 | 定位 |
|---|---:|---:|---|
| 01. vLLM Ascend 产品配置 | 13 | 3.4% | 仓库自身的功能开关、优化策略和兼容行为。部署用户应优先阅读本节。 |
| 02. Ascend、CANN 与通信运行时 | 25 | 6.6% | 由 Ascend/CANN/HCCL/LCCL 等运行时解释的设备、通信、日志和调试配置。 |
| 03. 分布式启动与进程拓扑 | 17 | 4.5% | 由 torchrun、Ray、vLLM DP 或集群启动器注入的 rank、world size、地址和端口。 |
| 04. KV Transfer、存储与解耦后端 | 26 | 6.9% | Mooncake、MemFabric、Memcache、YuanRong、LMCache、RFork 等后端配置。 |
| 05. 上游 vLLM、PyTorch 与模型生态 | 37 | 9.8% | 由上游 vLLM、PyTorch、Hugging Face、ModelScope 或模型工具解释的配置。 |
| 06. 构建、编译、安装与 Docker 构建 | 60 | 15.9% | wheel、CMake、自定义算子、编译器、依赖和 Docker build 参数。Docker ARG 会明确标注。 |
| 07. 系统与通用运行环境 | 12 | 3.2% | 操作系统、动态链接器、Python、OpenMP 和代理等通用进程环境。 |
| 08. 测试、基准与模型用例 | 29 | 7.7% | 只用于 UT/E2E、nightly、性能基准或特定模型用例的输入。 |
| 09. CI、发布、凭据与仓库自动化 | 131 | 34.7% | GitHub Actions、镜像/制品发布、缓存、翻译、PR bot 和流水线传值。 |
| 10. 文档与示例命令辅助变量 | 23 | 6.1% | 文档为了复用命令而声明的占位路径、镜像、设备、端口等，不是产品配置接口。 |
| 11. 仓库工具与其他辅助变量 | 5 | 1.3% | 独立工具、诊断脚本或无法归入以上领域的仓库内部环境输入。 |
| **合计** | **378** | **100.0%** | 互斥分类 |

## 出现范围统计

同一变量可在多个范围出现，因此本表合计会大于唯一名称总数。

| 出现范围 | 唯一变量数 | 占总数比例 |
|---|---:|---:|
| CI/发布 | 170 | 45.0% |
| 文档/示例 | 121 | 32.0% |
| 测试 | 68 | 18.0% |
| 构建/打包 | 55 | 14.6% |
| 运行时代码 | 44 | 11.6% |
| 工具/辅助 | 20 | 5.3% |

## 使用机制统计

机制之间存在交集。例如一个变量既可能由文档 `export`，也可能被 Python 直接读取。Docker `ARG` 是构建参数，不等于容器运行时环境变量。

| 使用机制 | 唯一变量数 | 占总数比例 |
|---|---:|---:|
| YAML env | 160 | 42.3% |
| Python/C/C++ API | 152 | 40.2% |
| Shell export | 123 | 32.5% |
| Docker ARG | 30 | 7.9% |
| Docker ENV | 2 | 0.5% |

## 分类明细

`范围` 表示变量在仓库中出现的区域；`机制` 表示仓库如何把它作为环境变量使用。代表位置均为仓库相对路径。

### 01. vLLM Ascend 产品配置（13）

仓库自身的功能开关、优化策略和兼容行为。部署用户应优先阅读本节。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `DYNAMIC_EPLB` | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `tests/ut/eplb/core/a2/test_eplb_utils.py:48` |
| `EXPERT_MAP_RECORD` | 运行时代码 | Python/C/C++ API | `vllm_ascend/ascend_config.py:911` |
| `MSMONITOR_USE_DAEMON` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:74` |
| `TRITON_ALL_BLOCKS_PARALLEL` | 运行时代码 | Python/C/C++ API | `vllm_ascend/ops/rotary_embedding.py:480` |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md:360` |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:102` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/envs.py:72` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/envs.py:92` |
| `VLLM_ASCEND_ENABLE_MLAPO` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/envs.py:79` |
| `VLLM_ASCEND_ENABLE_NZ` | 工具/辅助, 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `AGENTS.md:56` |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM4.x.md:134` |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:98` |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | 运行时代码 | Python/C/C++ API | `vllm_ascend/platform.py:32` |

### 02. Ascend、CANN 与通信运行时（25）

由 Ascend/CANN/HCCL/LCCL 等运行时解释的设备、通信、日志和调试配置。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `ACL_OP_INIT_MODE` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:871` |
| `ASCEND_AGGREGATE_ENABLE` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:986` |
| `ASCEND_GLOBAL_LOG_LEVEL` | 构建/打包 | Python/C/C++ API | `csrc/cmake/scripts/utest/gen_coverage.py:220` |
| `ASCEND_LAUNCH_BLOCKING` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/platform.py:1115` |
| `ASCEND_LOCAL_COMM_RES` | 运行时代码 | Python/C/C++ API | `vllm_ascend/utils.py:575` |
| `ASCEND_LOG_PREFIX` | CI/发布 | YAML env | `.github/workflows/schedule_nightly_test_310p.yaml:90` |
| `ASCEND_RT_VISIBLE_DEVICES` | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `examples/offline_disaggregated_prefill_npu.py:41` |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 构建/打包 | Python/C/C++ API | `csrc/cmake/scripts/util/ascendc_bin_param_build.py:373` |
| `ASCEND_TRANSPORT_PRINT` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:987` |
| `CPU_AFFINITY_CONF` | 文档/示例 | Shell export | `docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:128` |
| `HCCL_BUFFSIZE` | CI/发布, 文档/示例 | Python/C/C++ API, Shell export, YAML env | `docs/source/user_guide/feature_guide/graph_mode.md:276` |
| `HCCL_CONNECT_TIMEOUT` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:83` |
| `HCCL_DETERMINISTIC` | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `tests/ut/test_batch_invariant.py:37` |
| `HCCL_EXEC_TIMEOUT` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:82` |
| `HCCL_IF_IP` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/doc_writing.md:221` |
| `HCCL_INTRA_PCIE_ENABLE` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/utils.py:1034` |
| `HCCL_INTRA_ROCE_ENABLE` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/utils.py:1034` |
| `HCCL_OP_EXPANSION_MODE` | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `docs/source/user_guide/feature_guide/graph_mode.md:279` |
| `HCCL_RDMA_RETRY_CNT` | 运行时代码 | Python/C/C++ API | `vllm_ascend/distributed/kv_transfer/utils/utils.py:60` |
| `HCCL_RDMA_TIMEOUT` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/distributed/kv_transfer/utils/utils.py:59` |
| `HCCL_SOCKET_IFNAME` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/doc_writing.md:224` |
| `HCCL_SO_PATH` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:61` |
| `HCCL_TRANSFER_TIMEOUT` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:136` |
| `LCCL_DETERMINISTIC` | 测试, 运行时代码 | Python/C/C++ API | `tests/ut/test_batch_invariant.py:38` |
| `TASK_QUEUE_ENABLE` | 文档/示例 | Python/C/C++ API, Shell export | `docs/source/user_guide/feature_guide/graph_mode.md:277` |

### 03. 分布式启动与进程拓扑（17）

由 torchrun、Ray、vLLM DP 或集群启动器注入的 rank、world size、地址和端口。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `GLOO_SOCKET_IFNAME` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/doc_writing.md:222` |
| `LOCAL_RANK` | 文档/示例 | Python/C/C++ API | `examples/offline_external_launcher.py:152` |
| `LOCAL_WORLD_SIZE` | 运行时代码 | Python/C/C++ API | `vllm_ascend/compilation/compiler_interface.py:101` |
| `LWS_LEADER_ADDRESS` | 测试 | Python/C/C++ API | `tests/e2e/nightly/multi_node/scripts/utils.py:81` |
| `LWS_WORKER_INDEX` | 工具/辅助, 文档/示例, 测试 | Python/C/C++ API, Shell export | `tests/e2e/conftest.py:328` |
| `MASTER_ADDR` | 文档/示例 | Python/C/C++ API | `examples/offline_external_launcher.py:149` |
| `MASTER_PORT` | 文档/示例 | Python/C/C++ API | `examples/offline_external_launcher.py:150` |
| `RANK` | 文档/示例 | Python/C/C++ API | `examples/offline_external_launcher.py:151` |
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | CI/发布, 文档/示例 | Shell export, YAML env | `docs/source/tutorials/features/ray.md:115` |
| `TP_SOCKET_IFNAME` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/doc_writing.md:223` |
| `VLLM_DP_MASTER_IP` | 文档/示例, 测试 | Python/C/C++ API | `examples/offline_data_parallel.py:123` |
| `VLLM_DP_MASTER_PORT` | 文档/示例, 测试 | Python/C/C++ API | `examples/offline_data_parallel.py:124` |
| `VLLM_DP_RANK` | 文档/示例, 测试 | Python/C/C++ API | `examples/offline_data_parallel.py:120` |
| `VLLM_DP_RANK_LOCAL` | 文档/示例, 测试 | Python/C/C++ API | `examples/offline_data_parallel.py:121` |
| `VLLM_DP_SIZE` | 文档/示例, 测试 | Python/C/C++ API | `examples/offline_data_parallel.py:122` |
| `VLLM_HOST_IP` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:1067` |
| `WORLD_SIZE` | 文档/示例 | Python/C/C++ API | `examples/offline_external_launcher.py:153` |

### 04. KV Transfer、存储与解耦后端（26）

Mooncake、MemFabric、Memcache、YuanRong、LMCache、RFork 等后端配置。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `ASCEND_CONNECT_TIMEOUT` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:175` |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:71` |
| `ASCEND_GLOBAL_RESOURCE_CONFIG` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:1229` |
| `ASCEND_TRANSFER_TIMEOUT` | 工具/辅助, 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `tools/test_memfabric_pd_read.py:406` |
| `DATASYSTEM_CLIENT_LOG_DIR` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:948` |
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:199` |
| `MMC_LOCAL_CONFIG_PATH` | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py:17` |
| `MMC_META_CONFIG_PATH` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:533` |
| `MOONCAKE_CONFIG_PATH` | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `tests/ut/distributed/ascend_store/test_backend.py:182` |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 运行时代码 | Python/C/C++ API | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:297` |
| `MOONCAKE_MASTER` | 运行时代码 | Python/C/C++ API | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:296` |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:475` |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:474` |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:476` |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:473` |
| `NETLOADER_CONFIG` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/netloader.md:68` |
| `OPENLIBING_SECRET` | CI/发布, 工具/辅助 | Python/C/C++ API, YAML env | `tools/upload_to_openlibing.py:217` |
| `RFORK_CONFIG` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/rfork.md:124` |
| `RFORK_MOCK_ALLOC_POLICY` | 文档/示例 | Python/C/C++ API | `examples/rfork/rfork_planner.py:69` |
| `RFORK_MOCK_DEFAULT_RESOURCE_POINTS` | 文档/示例 | Python/C/C++ API | `examples/rfork/rfork_planner.py:68` |
| `RFORK_MOCK_HEARTBEAT_SWEEP_SEC` | 文档/示例 | Python/C/C++ API | `examples/rfork/rfork_planner.py:67` |
| `RFORK_MOCK_HEARTBEAT_TTL_SEC` | 文档/示例 | Python/C/C++ API | `examples/rfork/rfork_planner.py:66` |
| `RFORK_MOCK_HOST` | 文档/示例 | Python/C/C++ API | `examples/rfork/rfork_planner.py:64` |
| `RFORK_MOCK_PORT` | 文档/示例 | Python/C/C++ API | `examples/rfork/rfork_planner.py:65` |
| `RFORK_SEED_TIMEOUT_SEC` | 测试 | Python/C/C++ API | `tests/ut/model_loader/rfork/test_rfork_loader.py:48` |
| `YR_CONFIG_PATH` | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | `tests/ut/distributed/test_yuanrong_backend.py:162` |

### 05. 上游 vLLM、PyTorch 与模型生态（37）

由上游 vLLM、PyTorch、Hugging Face、ModelScope 或模型工具解释的配置。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `HF_DATASETS_CACHE` | 文档/示例 | Python/C/C++ API | `docs/source/tutorials/models/Qwen3-Embedding.md:222` |
| `HF_DATASETS_OFFLINE` | CI/发布, 文档/示例 | Shell export, YAML env | `docs/source/developer_guide/evaluation/using_lm_eval.md:227` |
| `HF_ENDPOINT` | 文档/示例 | Python/C/C++ API, Shell export | `docs/source/tutorials/models/Qwen3-Embedding.md:223` |
| `HF_HOME` | 文档/示例 | Shell export | `docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:73` |
| `HF_HUB_OFFLINE` | CI/发布, 测试 | Python/C/C++ API, Shell export, YAML env | `tests/e2e/pull_request/four_card/spec_decode/test_mtp_step3p5.py:33` |
| `MODELSCOPE_HUB_FILE_LOCK` | 测试 | Shell export | `tests/e2e/run_doctests.sh:25` |
| `PYTORCH_NPU_ALLOC_CONF` | CI/发布, 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export, YAML env | `docs/source/user_guide/feature_guide/graph_mode.md:280` |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | CI/发布, 文档/示例 | Shell export, YAML env | `docs/source/installation.md:280` |
| `TORCH_EXTENSIONS_ALWAYS_BUILD` | 运行时代码 | Python/C/C++ API | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:374` |
| `TORCH_VERSION` | CI/发布, 测试 | Python/C/C++ API, YAML env | `tests/e2e/models/test_asr_eval_correctness.py:48` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | 文档/示例, 测试 | Python/C/C++ API | `examples/rl/rlhf_http_npu_ipc.py:53` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:77` |
| `VLLM_BATCH_INVARIANT` | 文档/示例, 测试 | Python/C/C++ API, Shell export | `docs/source/user_guide/feature_guide/batch_invariance.md:79` |
| `VLLM_CI_RUNNER` | CI/发布, 测试 | Python/C/C++ API, YAML env | `tests/e2e/nightly/multi_node/external_dp/scripts/utils.py:210` |
| `VLLM_DISABLE_COMPILE_CACHE` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/eight_card/test_minimax_m3.py:46` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | CI/发布, 文档/示例, 测试 | Shell export, YAML env | `docs/source/tutorials/models/GLM5.2.md:874` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:81` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:109` |
| `VLLM_LOGGING_LEVEL` | CI/发布, 文档/示例, 测试 | Shell export, YAML env | `examples/external_online_dp/run_dp_template.sh:5` |
| `VLLM_MAX_MODEL_LEN` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:110` |
| `VLLM_MAX_PROMPT` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:152` |
| `VLLM_MIN_PROMPT` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:151` |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:1401` |
| `VLLM_NEEDLE_BATCH_SIZE` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:108` |
| `VLLM_NEEDLE_MAX_TOKENS` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:159` |
| `VLLM_NEEDLE_TEMPERATURE` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:157` |
| `VLLM_NEEDLE_TOP_P` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:158` |
| `VLLM_NEEDLE_TRIALS` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:149` |
| `VLLM_PP_LAYER_PARTITION` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/pipeline_parallel.md:192` |
| `VLLM_RPC_TIMEOUT` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:80` |
| `VLLM_TORCH_PROFILER_WITH_STACK` | 文档/示例 | Shell export | `docs/source/tutorials/models/Qwen3-235B-A22B.md:410` |
| `VLLM_TP_SIZE` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:112` |
| `VLLM_USE_MODELSCOPE` | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | `docs/source/user_guide/feature_guide/sleep_mode.md:96` |
| `VLLM_USE_V1` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:74` |
| `VLLM_USE_V2_MODEL_RUNNER` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:86` |
| `VLLM_VERSION` | CI/发布, 工具/辅助, 测试, 运行时代码 | Python/C/C++ API, YAML env | `tests/e2e/models/test_asr_eval_correctness.py:43` |
| `VLLM_WORKER_MULTIPROC_METHOD` | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | `docs/source/user_guide/feature_guide/sleep_mode.md:97` |

### 06. 构建、编译、安装与 Docker 构建（60）

wheel、CMake、自定义算子、编译器、依赖和 Docker build 参数。Docker ARG 会明确标注。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `ACLNN_INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:411` |
| `AIS_BENCH_TAG` | 文档/示例, 构建/打包 | Docker ARG, Shell export | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:25` |
| `AIS_BENCH_URL` | 文档/示例, 构建/打包 | Docker ARG, Shell export | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:26` |
| `ASCEND_CUSTOM_OPP_PATH` | 构建/打包 | Python/C/C++ API | `csrc/aclnn_torch_adapter/op_api_common.h:194` |
| `ASCEND_HOME_PATH` | 构建/打包, 测试, 运行时代码 | Python/C/C++ API | `csrc/cmake/scripts/util/ascendc_impl_build.py:192` |
| `ASCEND_OPP_PATH` | 构建/打包 | Python/C/C++ API | `csrc/aclnn_torch_adapter/op_api_common.h:216` |
| `ASCEND_TOOLKIT_HOME` | 文档/示例 | Shell export | `docs/source/installation.md:279` |
| `BASE_IMAGE` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:19` |
| `BASE_OS` | 构建/打包 | Docker ARG | `Dockerfile:19` |
| `BASE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:324` |
| `BISHENG_REAL_PATH` | 构建/打包 | Python/C/C++ API | `csrc/cmake/scripts/util/ascendc_impl_build.py:177` |
| `BUILD_BUILTIN_OPP` | 构建/打包 | Python/C/C++ API | `csrc/cmake/scripts/util/ascendc_impl_build.py:368` |
| `BUILD_KERNEL_SRC` | 构建/打包 | Python/C/C++ API | `csrc/cmake/scripts/util/ascendc_impl_build.py:145` |
| `BUILD_PATH` | 构建/打包 | Shell export | `csrc/build.sh:328` |
| `BUILD_TYPE` | CI/发布, 构建/打包 | Docker ARG, YAML env | `Dockerfile:82` |
| `CANN_QUAY_URL` | 构建/打包 | Docker ARG | `Dockerfile:17` |
| `CANN_VERSION` | CI/发布, 构建/打包, 测试 | Docker ARG, Python/C/C++ API, YAML env | `Dockerfile:18` |
| `CC` | 运行时代码 | Python/C/C++ API | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:385` |
| `CMAKE_BUILD_TYPE` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:37` |
| `COMPILER_INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:412` |
| `COMPILE_CUSTOM_KERNELS` | CI/发布, 文档/示例, 构建/打包, 运行时代码 | Docker ARG, Python/C/C++ API, Shell export, YAML env | `Dockerfile:60` |
| `CXX` | 运行时代码 | Python/C/C++ API | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:384` |
| `CXX_COMPILER` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:46` |
| `C_COMPILER` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:49` |
| `DAILY_DEPS_MODE` | 构建/打包 | Docker ARG | `Dockerfile:91` |
| `DEBIAN_FRONTEND` | 构建/打包 | Docker ENV | `.github/workflows/dockerfiles/Dockerfile.buildwheel.310p:23` |
| `EAGER_INCLUDE_OPP_ACLNNOP_PATH` | 构建/打包 | Shell export | `csrc/build.sh:422` |
| `EAGER_LIBRARY_OPP_PATH` | 构建/打包 | Shell export | `csrc/build.sh:417` |
| `EAGER_LIBRARY_PATH` | 构建/打包 | Shell export | `csrc/build.sh:418` |
| `FETCHCONTENT_BASE_DIR` | 构建/打包 | Python/C/C++ API | `setup.py:304` |
| `GE_INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:414` |
| `GITEE_TOKEN` | CI/发布, 构建/打包 | Docker ARG, YAML env | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:28` |
| `GITEE_USERNAME` | CI/发布, 构建/打包 | Docker ARG, YAML env | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:27` |
| `GRAPH_INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:413` |
| `GRAPH_LIBRARY_PATH` | 构建/打包 | Shell export | `csrc/build.sh:420` |
| `GRAPH_LIBRARY_STUB_PATH` | 构建/打包 | Shell export | `csrc/build.sh:419` |
| `INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:410` |
| `INC_INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:415` |
| `LINUX_INCLUDE_PATH` | 构建/打包 | Shell export | `csrc/build.sh:416` |
| `MAX_JOBS` | CI/发布, 运行时代码 | Python/C/C++ API, Shell export, YAML env | `vllm_ascend/envs.py:34` |
| `MEMCACHE_DATE` | 构建/打包 | Docker ARG | `Dockerfile:84` |
| `MEMCACHE_VERSION` | 构建/打包 | Docker ARG | `Dockerfile:83` |
| `MEMFABRIC_DATE` | 构建/打包 | Docker ARG | `Dockerfile:86` |
| `MEMFABRIC_VERSION` | 构建/打包 | Docker ARG | `Dockerfile:85` |
| `MOONCAKE_TAG` | 构建/打包 | Docker ARG | `Dockerfile:27` |
| `OS_MARK` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:20` |
| `PIP_INDEX_URL` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:23` |
| `PY_VERSION` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.buildwheel.310p:17` |
| `SOC_VERSION` | CI/发布, 文档/示例, 构建/打包, 运行时代码 | Docker ARG, Docker ENV, Python/C/C++ API, Shell export, YAML env | `.github/workflows/dockerfiles/Dockerfile.buildwheel.310p:19` |
| `TAG_SUFFIX` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:21` |
| `TARGETARCH` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.lint:19` |
| `TORCH_NPU_DATE` | 构建/打包 | Docker ARG | `Dockerfile:88` |
| `TORCH_NPU_VERSION` | CI/发布, 构建/打包, 测试 | Docker ARG, Python/C/C++ API, YAML env | `Dockerfile:87` |
| `TRITON_ASCEND_PACKAGE_VERSION` | 构建/打包 | Docker ARG | `Dockerfile:90` |
| `TRITON_ASCEND_VERSION` | 构建/打包 | Docker ARG | `Dockerfile:89` |
| `VERBOSE` | 运行时代码 | Python/C/C++ API | `vllm_ascend/envs.py:55` |
| `VLLM_ASCEND_BRANCH` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.nightly.310p:17` |
| `VLLM_COMMIT` | CI/发布, 构建/打包, 测试 | Docker ARG, Python/C/C++ API, YAML env | `.github/workflows/dockerfiles/Dockerfile.lint:30` |
| `VLLM_REPO` | 构建/打包 | Docker ARG | `.github/workflows/dockerfiles/Dockerfile.lint:26` |
| `VLLM_TAG` | 构建/打包 | Docker ARG | `Dockerfile:44` |

### 07. 系统与通用运行环境（12）

操作系统、动态链接器、Python、OpenMP 和代理等通用进程环境。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `CONDA_EXE` | 工具/辅助 | Python/C/C++ API | `collect_env.py:132` |
| `FORCE_COLOR` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:171` |
| `HOME` | 构建/打包 | Python/C/C++ API | `csrc/scripts/package/common/py/pkg_parser.py:409` |
| `LD_LIBRARY_PATH` | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export | `tests/e2e/conftest.py:244` |
| `LD_PRELOAD` | 文档/示例, 测试 | Python/C/C++ API, Shell export | `tests/e2e/pull_request/eight_card/test_minimax_m3.py:70` |
| `OMP_NUM_THREADS` | 文档/示例, 测试 | Python/C/C++ API, Shell export | `tests/e2e/pull_request/eight_card/test_minimax_m3.py:44` |
| `OMP_PROC_BIND` | 文档/示例 | Python/C/C++ API, Shell export | `docs/source/user_guide/feature_guide/graph_mode.md:278` |
| `PYTHONHASHSEED` | 文档/示例 | Shell export | `docs/source/tutorials/models/GLM5.2.md:989` |
| `PYTHONPATH` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/kv_pool.md:154` |
| `SYSTEMROOT` | 工具/辅助 | Python/C/C++ API | `collect_env.py:216` |
| `TERM` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:172` |
| `TOKENIZERS_PARALLELISM` | 文档/示例 | Shell export | `docs/source/tutorials/models/DeepSeekOCR2.md:128` |

### 08. 测试、基准与模型用例（29）

只用于 UT/E2E、nightly、性能基准或特定模型用例的输入。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `BENCHMARK_JOB_NAME` | CI/发布, 测试 | Python/C/C++ API, YAML env | `tests/e2e/nightly/multi_node/external_dp/scripts/utils.py:235` |
| `BISECT_ARGS_JSON` | CI/发布 | YAML env | `.github/workflows/pr_nightly_command.yml:362` |
| `BISECT_BAD_COMMIT` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:248` |
| `BISECT_BARRIER_TIMEOUT` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:251` |
| `BISECT_CONFIG_BASE_PATH` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:255` |
| `BISECT_COORD_DIR` | 工具/辅助 | Python/C/C++ API | `tools/bisect/config.py:60` |
| `BISECT_FAIL_CONFIRM_RETRIES` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:249` |
| `BISECT_FORCE_INITIAL_BUILD` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:254` |
| `BISECT_GOOD_COMMIT` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:247` |
| `BISECT_GOOD_TABLE` | 工具/辅助 | Python/C/C++ API | `tools/bisect/config.py:49` |
| `BISECT_NO_VERIFY_BAD` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:253` |
| `BISECT_NO_VERIFY_GOOD` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:252` |
| `BISECT_TRIAL_TIMEOUT` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:250` |
| `BISECT_WORK_DIR` | 工具/辅助 | Python/C/C++ API | `tools/bisect/config.py:56` |
| `EXTERNAL_DP_MAX_WAIT_SECONDS` | 测试 | Python/C/C++ API | `tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py:108` |
| `GLOG_minloglevel` | 测试 | Shell export | `tests/e2e/nightly/multi_node/scripts/run.sh:65` |
| `LOG_PREFIX` | 测试 | Python/C/C++ API | `tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py:94` |
| `MINIMAX_M3_DECODE_BOUNDARY_REPEATS` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py:1277` |
| `MINIMAX_M3_DECODE_BOUNDARY_ROUNDS` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py:1273` |
| `MINIMAX_M3_MODEL_PATH` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/eight_card/test_minimax_m3.py:30` |
| `MINIMAX_M3_SPARSE_BACKEND` | 测试 | Python/C/C++ API | `tests/e2e/conftest.py:100` |
| `MODEL_ARGS` | 测试 | Shell export | `tests/e2e/models/report_template.md:13` |
| `QWEN3_MRV2_EPLB_MODEL_PATH` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/four_card/test_qwen3_mrv2_eplb.py:13` |
| `SFA_V1_PRECISION_METRICS_LOG` | 测试 | Python/C/C++ API | `tests/ut/attention/a2/test_sfa_v1_precision.py:32` |
| `VLLM_ASCEND_REF` | 工具/辅助, 测试 | Python/C/C++ API | `tests/e2e/nightly/multi_node/external_dp/scripts/utils.py:219` |
| `VLLM_TEST_MODEL` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/rlhf/conftest.py:40` |
| `VLLM_TEST_SEED` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:145` |
| `VLLM_TEST_TP_SIZE` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_batch_invariant.py:229` |
| `WEIGHT_TRANSFER_TEST_MODEL` | 测试 | Python/C/C++ API | `tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py:53` |

### 09. CI、发布、凭据与仓库自动化（131）

GitHub Actions、镜像/制品发布、缓存、翻译、PR bot 和流水线传值。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `A2_CACHE_HIT` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/_ensure_csrc_cache.yaml:211` |
| `A3_CACHE_HIT` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/_ensure_csrc_cache.yaml:212` |
| `ACTOR` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:54` |
| `ALL_ARGS` | CI/发布 | YAML env | `.github/workflows/pr_e2e_command.yml:142` |
| `AUROGON_API_PREFIX` | CI/发布 | YAML env | `.github/workflows/_manual-hitest.yaml:56` |
| `AWS_ACCESS_KEY_ID` | CI/发布 | YAML env | `.github/workflows/push_build_precommit_cache.yaml:52` |
| `AWS_REGION` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:162` |
| `AWS_S3_FORCE_PATH_STYLE` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:161` |
| `AWS_SECRET_ACCESS_KEY` | CI/发布 | YAML env | `.github/workflows/push_build_precommit_cache.yaml:53` |
| `BASE_BRANCH` | CI/发布 | YAML env | `.github/workflows/pr_revert_command.yml:149` |
| `BASE_REF` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:116` |
| `BASE_REPOSITORY` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:115` |
| `BASE_SHA` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:117` |
| `BRANCH` | CI/发布 | YAML env | `.github/workflows/schedule_update_estimated_times.yaml:214` |
| `BUFFER_MINUTES` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:84` |
| `CACHE_REPO` | CI/发布 | YAML env | `.github/workflows/_schedule_image_build.yaml:146` |
| `CANN_IMAGE` | CI/发布 | YAML env | `.github/workflows/schedule_nightly_test_a2.yaml:95` |
| `CHANGED_DOCS` | CI/发布 | YAML env | `.github/workflows/schedule_doc_linkcheck.yaml:60` |
| `CHECKOUT_SOURCE_SHA` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:344` |
| `COMMAND` | CI/发布 | YAML env | `.github/workflows/pr_nightly_command.yml:118` |
| `CONFIG_BASE_PATH` | CI/发布, 工具/辅助, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | `tests/e2e/nightly/multi_node/internal_dp/scripts/utils.py:14` |
| `CONFIG_YAML_PATH` | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | `tests/e2e/nightly/multi_node/scripts/utils.py:46` |
| `CONTEXT_PREFIX` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:79` |
| `CSRC_CACHE_HIT` | CI/发布 | YAML env | `.github/workflows/_selected_tests.yaml:229` |
| `DEEPSEEK_API_KEY` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/scripts/po_translate.py:508` |
| `ENABLE_COVERAGE` | CI/发布 | YAML env | `.github/workflows/_selected_tests.yaml:281` |
| `EVENT_BASE_SHA` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:216` |
| `FILE_COUNT` | CI/发布 | YAML env | `.github/workflows/schedule_doc_translate.yaml:169` |
| `FILE_LIST` | CI/发布 | YAML env | `.github/workflows/schedule_doc_translate.yaml:168` |
| `FILTER` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:78` |
| `FIXED_BASE_SHA` | CI/发布 | YAML env | `.github/workflows/_selected_tests.yaml:145` |
| `GH_REPO` | CI/发布 | YAML env | `.github/workflows/bot_pr_create.yaml:69` |
| `GH_TOKEN` | CI/发布, 工具/辅助 | Python/C/C++ API, YAML env | `.agents/skills/vllm-ascend-release/scripts/fetch_commits.py:21` |
| `GITHUB_ACTIONS` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/run_suite.py:61` |
| `GITHUB_OUTPUT` | CI/发布 | Python/C/C++ API | `.github/workflows/_ensure_csrc_cache.yaml:243` |
| `GITHUB_REPO` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:174` |
| `GITHUB_RUN_ID` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/run_suite.py:328` |
| `GITHUB_SHA` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/run_suite.py:327` |
| `GITHUB_TOKEN` | CI/发布, 工具/辅助 | Python/C/C++ API, YAML env | `.agents/skills/vllm-ascend-release/scripts/fetch_commits.py:21` |
| `GONOSUMDB` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:176` |
| `GOOD_TABLE` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_single_node.yaml:424` |
| `GOPROXY` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:175` |
| `HAS_READY_A5_LABEL` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:513` |
| `HAS_READY_LABEL` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:512` |
| `HEAD_FORK` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:58` |
| `HITEST_APIG_APPCODE` | CI/发布 | YAML env | `.github/workflows/_manual-hitest.yaml:57` |
| `HITEST_KEY` | CI/发布 | YAML env | `.github/workflows/_manual-hitest.yaml:58` |
| `HITEST_SECRET` | CI/发布 | YAML env | `.github/workflows/_manual-hitest.yaml:59` |
| `HW_TOKEN` | CI/发布 | YAML env | `.github/workflows/_nightly_image_build.yaml:100` |
| `HW_TOKEN_DAILY` | CI/发布 | YAML env | `.github/workflows/_nightly_image_build.yaml:109` |
| `HW_USERNAME` | CI/发布 | YAML env | `.github/workflows/_nightly_image_build.yaml:99` |
| `HW_USERNAME_DAILY` | CI/发布 | YAML env | `.github/workflows/_nightly_image_build.yaml:108` |
| `INPUT_DOC_VERSIONS` | CI/发布 | YAML env | `.github/workflows/labeled_doctest.yaml:147` |
| `IS_A3_560T` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/resolve_nightly_tests.py:74` |
| `KUBECONFIG` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:164` |
| `MAIN2MAIN_CASES_FILE` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:178` |
| `MAIN2MAIN_IMAGE_TAG` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:171` |
| `MAIN2MAIN_KEEP_BRANCH` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:176` |
| `MAIN2MAIN_LOG_HELPERS` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:179` |
| `MAIN2MAIN_MODEL` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:45` |
| `MAIN2MAIN_WORKSPACE` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:177` |
| `MATRIX_FILE` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/scripts/resolve_nightly_tests.py:121` |
| `MATRIX_JSON` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:77` |
| `MATRIX_OUTPUTS` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/scripts/resolve_nightly_tests.py:124` |
| `MAX_PARALLEL` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:81` |
| `MERGE_COMMIT` | CI/发布 | YAML env | `.github/workflows/pr_revert_command.yml:148` |
| `NAMESPACE` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:165` |
| `NIGHTLY_MATRIX` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/resolve_nightly_tests.py:67` |
| `NODE_VERSION` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:274` |
| `OBS_ACCESS_KEY` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/_selected_tests.yaml:486` |
| `OBS_SECRET_KEY` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/_selected_tests.yaml:487` |
| `OUTPUT_DIR` | CI/发布 | YAML env | `.github/workflows/schedule_release_code_and_wheel.yml:157` |
| `OUTPUT_JSON` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/detect_po_changes.py:549` |
| `P310_CACHE_HIT` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/_ensure_csrc_cache.yaml:213` |
| `PIP_EXTRA_INDEX_URL` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_single_node.yaml:207` |
| `POD_WAIT_MINUTES` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:83` |
| `PRE_COMMIT_COLOR` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:170` |
| `PRE_COMMIT_HOME` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:169` |
| `PROJECT_TOML` | CI/发布 | YAML env | `.github/workflows/schedule_release_code_and_wheel.yml:156` |
| `PR_AUTHOR` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:55` |
| `PR_BASE` | CI/发布 | YAML env | `.github/workflows/schedule_doc_translate.yaml:172` |
| `PR_BASE_REF` | CI/发布 | YAML env | `.github/workflows/pr_e2e_command.yml:140` |
| `PR_BASE_SHA` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:131` |
| `PR_BODY` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:129` |
| `PR_HEAD` | CI/发布 | YAML env | `.github/workflows/schedule_doc_translate.yaml:171` |
| `PR_HEAD_SHA` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:132` |
| `PR_NUMBER` | CI/发布 | YAML env | `.github/workflows/_manual-hitest.yaml:60` |
| `PR_SHA` | CI/发布 | YAML env | `.github/workflows/pr_e2e_command.yml:139` |
| `PR_TITLE` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:128` |
| `PR_URL` | CI/发布 | YAML env | `.github/workflows/pr_nightly_command.yml:117` |
| `PUSH_TO_GITHUB` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:173` |
| `QUAY_DAILY_PASSWORD` | CI/发布 | YAML env | `.github/workflows/_nightly_image_build.yaml:116` |
| `QUAY_REPO` | CI/发布 | YAML env | `.github/workflows/_schedule_image_build.yaml:143` |
| `QUAY_TEMP_REPO` | CI/发布 | YAML env | `.github/workflows/_schedule_image_build.yaml:144` |
| `REPO` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:56` |
| `RUNS_ON_RUNNER_NAME` | CI/发布 | YAML env | `.github/workflows/push_build_precommit_cache.yaml:51` |
| `RUNS_ON_S3_BUCKET_CACHE` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:158` |
| `RUNS_ON_S3_BUCKET_ENDPOINT` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:159` |
| `RUNS_ON_S3_FORCE_PATH_STYLE` | CI/发布 | YAML env | `.github/workflows/pr_test.yaml:160` |
| `RUN_ID` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:80` |
| `RUN_LINK` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:413` |
| `RUN_URL` | CI/发布 | YAML env | `.github/workflows/schedule_doc_translate.yaml:170` |
| `SCHEDULE_TAG_PATTERN` | CI/发布 | YAML env | `.github/workflows/_schedule_image_build.yaml:529` |
| `SELECTED_TARGET_IDS` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/pr_test.yaml:326` |
| `SELECTED_TEST_GROUPS` | CI/发布 | YAML env | `.github/workflows/_selected_tests.yaml:373` |
| `SHELLCHECK_OPTS` | CI/发布, 工具/辅助 | Shell export, YAML env | `format.sh:39` |
| `SOURCE_REPOSITORY` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:114` |
| `SUFFIX` | CI/发布 | YAML env | `.github/workflows/_schedule_image_build.yaml:530` |
| `TAGS` | CI/发布 | YAML env | `.github/workflows/_schedule_image_build.yaml:527` |
| `TARGETS` | CI/发布 | Python/C/C++ API, YAML env | `.github/workflows/_ensure_csrc_cache.yaml:215` |
| `TARGET_BRANCH` | CI/发布 | YAML env | `.github/workflows/pr_cherry_pick_command.yml:78` |
| `TARGET_COMMIT` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:57` |
| `TARGET_IDS` | CI/发布 | YAML env | `.github/workflows/_ensure_csrc_cache.yaml:152` |
| `TESTCASE_TIMEOUT_MINUTES` | CI/发布 | YAML env | `.github/workflows/_nightly_wait_for_pods_ready.yaml:82` |
| `TEST_CASES` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/resolve_nightly_tests.py:76` |
| `TEST_NAME` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:410` |
| `TEST_PATH` | CI/发布 | YAML env | `.github/workflows/_e2e_nightly_multi_node.yaml:411` |
| `UPSTREAM_FETCH_REF` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:342` |
| `UPSTREAM_REBASE_SHA` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:343` |
| `UPSTREAM_REPO` | CI/发布 | YAML env | `.github/workflows/schedule_main2main.yaml:44` |
| `UV_EXTRA_INDEX_URL` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:67` |
| `UV_HTTP_TIMEOUT` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:70` |
| `UV_INDEX_STRATEGY` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:68` |
| `UV_INDEX_URL` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:66` |
| `UV_INSECURE_HOST` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:69` |
| `UV_NO_CACHE` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:71` |
| `UV_SYSTEM_PYTHON` | CI/发布 | YAML env | `.github/workflows/_build_csrc_cache.yaml:72` |
| `VLLM_ASCEND_COMMIT` | CI/发布, 测试 | Python/C/C++ API, YAML env | `tests/e2e/models/test_asr_eval_correctness.py:46` |
| `VLLM_ASCEND_VERSION` | CI/发布, 测试 | Python/C/C++ API, YAML env | `tests/e2e/models/test_asr_eval_correctness.py:45` |
| `WEEKLY_MATRIX` | CI/发布 | Python/C/C++ API | `.github/workflows/scripts/resolve_nightly_tests.py:67` |
| `WHEEL_FILE` | CI/发布 | YAML env | `.github/workflows/schedule_release_code_and_wheel.yml:155` |

### 10. 文档与示例命令辅助变量（23）

文档为了复用命令而声明的占位路径、镜像、设备、端口等，不是产品配置接口。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `DATASET_SOURCE` | 文档/示例 | Shell export | `docs/source/developer_guide/evaluation/using_opencompass.md:65` |
| `DEVICE` | 文档/示例 | Shell export | `docs/source/developer_guide/evaluation/using_ais_bench.md:13` |
| `ENDPOINT` | 文档/示例 | Shell export | `docs/source/user_guide/deployment_guide/using_volcano_kthena.md:390` |
| `IMAGE` | CI/发布, 文档/示例 | Shell export, YAML env | `docs/source/_templates/template-supplement.md:119` |
| `IP_ADDRESS` | 文档/示例 | Shell export | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:320` |
| `IS_PR_TEST` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/multi_node_test.md:400` |
| `MASTER_IP` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/doc_writing.md:150` |
| `MASTER_IP_ADDRESS` | 文档/示例 | Shell export | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:399` |
| `MM_IMAGE_PATH` | 工具/辅助 | Python/C/C++ API | `tools/send_mm_request.py:37` |
| `MODEL` | 文档/示例 | Shell export | `docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:394` |
| `MODEL_PATH` | 文档/示例 | Shell export | `docs/source/tutorials/models/Gemma4.md:82` |
| `NAME` | 文档/示例 | Shell export | `docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:23` |
| `NETWORK_CARD_NAME` | 文档/示例 | Shell export | `docs/source/tutorials/models/Qwen3.5-397B-A17B.md:321` |
| `NPU_MEMORY_FRACTION` | 文档/示例 | Shell export | `docs/source/tutorials/models/gpt-oss-120b.md:104` |
| `OPENAI_API_KEY` | 文档/示例 | Python/C/C++ API | `examples/disaggregated_encoder/disagg_epd_proxy.py:650` |
| `PHYSICAL_DEVICES` | 文档/示例 | Shell export | `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md:26` |
| `PROFILING_SYMBOLS_PATH` | 文档/示例 | Shell export | `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:163` |
| `SAVE_PATH` | 文档/示例 | Shell export | `docs/source/user_guide/feature_guide/quantization.md:49` |
| `SERVER_PORT` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/doc_writing.md:91` |
| `SERVICE_PROF_CONFIG_PATH` | 文档/示例 | Shell export | `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:162` |
| `TIKTOKEN_ENCODINGS_BASE` | 文档/示例 | Shell export | `docs/source/tutorials/models/gpt-oss-120b.md:88` |
| `USE_MODELSCOPE_HUB` | 文档/示例 | Shell export | `docs/source/developer_guide/evaluation/using_lm_eval.md:116` |
| `WORKSPACE` | 文档/示例 | Shell export | `docs/source/developer_guide/contribution/multi_node_test.md:399` |

### 11. 仓库工具与其他辅助变量（5）

独立工具、诊断脚本或无法归入以上领域的仓库内部环境输入。

| 变量 | 范围 | 机制 | 代表位置 |
|---|---|---|---|
| `BENCHMARK_HOME` | 工具/辅助, 文档/示例, 测试 | Python/C/C++ API, Shell export | `tools/aisbench.py:32` |
| `DOCS_IS_RELEASE` | 工具/辅助 | Shell export | `tools/rtd_build.sh:39` |
| `DOCS_LANG` | 工具/辅助, 文档/示例 | Python/C/C++ API, Shell export | `docs/hooks/nav_titles.py:266` |
| `EXTERNAL_DP_LOG_DIR` | 文档/示例, 测试 | Python/C/C++ API, Shell export | `tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py:107` |
| `READTHEDOCS_VERSION_TYPE` | 工具/辅助 | Python/C/C++ API | `tools/set_release_flag.py:10` |

## 口径说明

1. 主分类按变量的主要消费者和生命周期确定，而不是只按名称前缀。`WORLD_SIZE`、`RANK` 等归入分布式启动与进程拓扑。
2. `VLLM_ASCEND_*` 归入产品配置；但 `MAX_JOBS`、`CMAKE_BUILD_TYPE` 等即使登记在 `vllm_ascend/envs.py`，仍按用途归入构建类。
3. `ASCEND_*`、`HCCL_*` 通常由 CANN/HCCL 解释；仓库可能读取、设置或仅在部署文档中推荐。
4. 只在 GitHub Actions `env` 中出现的名称归入 CI/发布类，不视为用户运行 vLLM Ascend 时需要配置的变量。
5. Docker `ARG` 仅在镜像构建期存在；同时转成 `ENV`、被导出或被代码读取时，才兼具环境变量语义。
6. 完整的多位置证据和中央 19 项默认值说明见 `vllm-ascend-environment-variables.md`。
