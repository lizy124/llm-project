# vLLM Ascend 环境变量清单

- 仓库版本：`8f89feb9d`
- 扫描范围：Git 跟踪文件（Python/C/C++、Shell、CMake、Dockerfile、GitHub Actions、测试、示例、文档）。
- 纳入规则：明确作为环境变量读取、写入、删除、`export`、Docker `ARG/ENV`、`-e/--env`、CI/Kubernetes `env` 的名称。CMake 函数变量和 Shell 普通局部变量不纳入。
- `ARG` 是镜像构建参数，只有通过 `ENV` 或命令传递后才进入运行时环境；文档示例和测试变量不代表生产默认配置。

## 统计

去重后共 **378** 个变量。范围列说明变量在仓库的实际出现区域，一个变量可同时出现在多个区域。

## 中央运行时配置

以下 19 个变量在 `vllm_ascend/envs.py` 的 `env_variables` 中集中定义并惰性读取：

| 变量 | 默认值/解析 | 作用 |
|---|---|---|
| `MAX_JOBS` | 未设置（使用全部 CPU 核） | 构建 wheel 时的最大并行编译线程数。 |
| `CMAKE_BUILD_TYPE` | Release | CMake 构建类型：Release、Debug 或 RelWithDebugInfo。 |
| `COMPILE_CUSTOM_KERNELS` | 1/True | 是否编译自定义算子；无 NPU 的 UT 环境才建议关闭。 |
| `CXX_COMPILER` | 未设置（系统默认） | C++ 编译器路径/命令。 |
| `C_COMPILER` | 未设置（系统默认） | C 编译器路径/命令。 |
| `SOC_VERSION` | 未设置（通过 npu-smi 探测） | 构建目标 Ascend SoC 型号。 |
| `VERBOSE` | 0/False | 是否输出详细编译日志。 |
| `ASCEND_HOME_PATH` | /usr/local/Ascend/ascend-toolkit/latest（调用方回退） | CANN toolkit 根目录。 |
| `HCCL_SO_PATH` | libhccl.so（调用方回退） | pyHCCL 通信后端加载的 HCCL 动态库。 |
| `VLLM_VERSION` | 未设置 | 源码安装/开发场景覆盖兼容检查所用的 vLLM 版本。 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 0/False | 启用 FlashComm1 张量并行通信优化；已废弃，改用 additional_config。 |
| `MSMONITOR_USE_DAEMON` | 0/False | 启用 msMonitor daemon 性能监控。 |
| `VLLM_ASCEND_ENABLE_MLAPO` | 1/True | 启用 DeepSeek W8A8 的 MLAPO 优化，会额外占用 NPU 内存。 |
| `VLLM_ASCEND_ENABLE_NZ` | 1 | 权重 FRACTAL_NZ 转换策略：0 关闭、1 仅量化、2 尽可能启用。 |
| `DYNAMIC_EPLB` | false | 动态专家并行负载均衡开关（按小写字符串读取）。 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | 0 | 允许使用 dispatch_ffn_combine/mega_moe 融合 MC2 路径。 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 0/False | 均衡调度开关；已废弃，改用 additional_config。 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | 1/True | 启用 transpose_kv_cache_by_block 融合算子。 |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | 未设置（自动探测） | KV cache offload 的 aclrtMemcpyBatchAsync：1 强制开、0 强制关。 |

## 全量索引

`位置` 使用 `仓库相对路径:行号`；为控制篇幅，每种发现机制最多列出 4 个代表位置。

| 变量 | 类别 | 范围 | 发现机制 | 位置（示例） |
|---|---|---|---|---|
| `A2_CACHE_HIT` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/_ensure_csrc_cache.yaml:211; .github/workflows/_ensure_csrc_cache.yaml:201 |
| `A3_CACHE_HIT` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/_ensure_csrc_cache.yaml:212; .github/workflows/_ensure_csrc_cache.yaml:202 |
| `ACLNN_INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:411 |
| `ACL_OP_INIT_MODE` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:871; docs/source/tutorials/models/GLM5.2.md:921; docs/source/tutorials/models/GLM5.2.md:992; docs/source/tutorials/models/GLM5.2.md:1085 |
| `ACTOR` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:54; .github/workflows/pr_e2e_command.yml:64; .github/workflows/pr_nightly_command.yml:77; .github/workflows/pr_rerun_command.yml:56 |
| `AIS_BENCH_TAG` | 仓库工具/文档专用 | 文档/示例, 构建/打包 | Docker ARG, Shell export | .github/workflows/dockerfiles/Dockerfile.nightly.310p:25; .github/workflows/dockerfiles/Dockerfile.nightly.a2:25; .github/workflows/dockerfiles/Dockerfile.nightly.a3:25; .github/workflows/dockerfiles/Dockerfile.nightly.a5:25; docs/source/developer_guide/contribution/multi_node_test.md:379; docs/source/developer_guide/contribution/multi_node_test.md:462 |
| `AIS_BENCH_URL` | 仓库工具/文档专用 | 文档/示例, 构建/打包 | Docker ARG, Shell export | .github/workflows/dockerfiles/Dockerfile.nightly.310p:26; .github/workflows/dockerfiles/Dockerfile.nightly.a2:26; .github/workflows/dockerfiles/Dockerfile.nightly.a3:26; .github/workflows/dockerfiles/Dockerfile.nightly.a5:26; docs/source/developer_guide/contribution/multi_node_test.md:380; docs/source/developer_guide/contribution/multi_node_test.md:463 |
| `ALL_ARGS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_e2e_command.yml:142; .github/workflows/pr_e2e_command.yml:156; .github/workflows/pr_nightly_command.yml:116 |
| `ASCEND_AGGREGATE_ENABLE` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:986 |
| `ASCEND_CONNECT_TIMEOUT` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:175; docs/source/user_guide/feature_guide/kv_pool.md:250; docs/source/user_guide/feature_guide/kv_pool.md:371 |
| `ASCEND_CUSTOM_OPP_PATH` | Ascend/通信/KV 组件 | 构建/打包 | Python/C/C++ API | csrc/aclnn_torch_adapter/op_api_common.h:194; csrc/cmake/scripts/examples/get_opapi_abs_path.py:35 |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | Ascend/通信/KV 组件 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:71; docs/source/user_guide/feature_guide/kv_pool.md:159; docs/source/user_guide/feature_guide/kv_pool.md:242; docs/source/user_guide/feature_guide/kv_pool.md:363; docs/source/user_guide/feature_guide/kv_pool.md:1221 |
| `ASCEND_GLOBAL_LOG_LEVEL` | Ascend/通信/KV 组件 | 构建/打包 | Python/C/C++ API | csrc/cmake/scripts/utest/gen_coverage.py:220; csrc/cmake/scripts/util/ascendc_bin_param_build.py:372 |
| `ASCEND_GLOBAL_RESOURCE_CONFIG` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:1229 |
| `ASCEND_HOME_PATH` | Ascend/通信/KV 组件 | 构建/打包, 测试, 运行时代码 | Python/C/C++ API | csrc/cmake/scripts/util/ascendc_impl_build.py:192; setup.py:61; tests/e2e/nightly/310p/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule_v310.py:12; vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:375 |
| `ASCEND_LAUNCH_BLOCKING` | Ascend/通信/KV 组件 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/platform.py:1115; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:76; examples/run_dp_server.sh:12 |
| `ASCEND_LOCAL_COMM_RES` | Ascend/通信/KV 组件 | 运行时代码 | Python/C/C++ API | vllm_ascend/utils.py:575 |
| `ASCEND_LOG_PREFIX` | Ascend/通信/KV 组件 | CI/发布 | YAML env | .github/workflows/schedule_nightly_test_310p.yaml:90; .github/workflows/schedule_nightly_test_a2.yaml:96; .github/workflows/schedule_nightly_test_a3.yaml:95; .github/workflows/schedule_nightly_test_a3_560t.yaml:96 |
| `ASCEND_OPP_PATH` | Ascend/通信/KV 组件 | 构建/打包 | Python/C/C++ API | csrc/aclnn_torch_adapter/op_api_common.h:216; csrc/cmake/scripts/examples/get_opapi_abs_path.py:55; csrc/cmake/scripts/examples/get_opapi_abs_path.py:92; csrc/cmake/scripts/util/ascendc_impl_build.py:281 |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend/通信/KV 组件 | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | examples/offline_disaggregated_prefill_npu.py:41; examples/offline_disaggregated_prefill_npu.py:88; examples/rl/rlhf_async_new_apis.py:70; tests/e2e/conftest.py:900; docs/source/developer_guide/contribution/doc_writing.md:240; docs/source/tutorials/features/ray.md:116; docs/source/tutorials/features/ray.md:128; docs/source/tutorials/features/suffix_speculative_decoding.md:80 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Ascend/通信/KV 组件 | 构建/打包 | Python/C/C++ API | csrc/cmake/scripts/util/ascendc_bin_param_build.py:373 |
| `ASCEND_TOOLKIT_HOME` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/installation.md:279 |
| `ASCEND_TRANSFER_TIMEOUT` | Ascend/通信/KV 组件 | 工具/辅助, 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | tools/test_memfabric_pd_read.py:406; vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:2004; vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py:1513; vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:1133; docs/source/user_guide/feature_guide/kv_pool.md:178; docs/source/user_guide/feature_guide/kv_pool.md:251; docs/source/user_guide/feature_guide/kv_pool.md:372 |
| `ASCEND_TRANSPORT_PRINT` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:987 |
| `AUROGON_API_PREFIX` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_manual-hitest.yaml:56 |
| `AWS_ACCESS_KEY_ID` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/push_build_precommit_cache.yaml:52 |
| `AWS_REGION` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:162; .github/workflows/push_build_precommit_cache.yaml:50 |
| `AWS_S3_FORCE_PATH_STYLE` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:161; .github/workflows/push_build_precommit_cache.yaml:49 |
| `AWS_SECRET_ACCESS_KEY` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/push_build_precommit_cache.yaml:53 |
| `BASE_BRANCH` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_revert_command.yml:149 |
| `BASE_IMAGE` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.nightly.310p:19; .github/workflows/dockerfiles/Dockerfile.nightly.a2:19; .github/workflows/dockerfiles/Dockerfile.nightly.a3:19; .github/workflows/dockerfiles/Dockerfile.nightly.a5:19 |
| `BASE_OS` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | Dockerfile:19; Dockerfile.310p:20; Dockerfile.310p.openEuler:20; Dockerfile.a3:20 |
| `BASE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:324 |
| `BASE_REF` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:116; .github/workflows/_ensure_csrc_cache.yaml:102; .github/workflows/_selected_tests.yaml:144; .github/workflows/_selected_tests_upstream.yaml:134 |
| `BASE_REPOSITORY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:115; .github/workflows/_ensure_csrc_cache.yaml:101 |
| `BASE_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:117; .github/workflows/_ensure_csrc_cache.yaml:103 |
| `BENCHMARK_HOME` | 仓库工具/文档专用 | 工具/辅助, 文档/示例, 测试 | Python/C/C++ API, Shell export | tools/aisbench.py:32; docs/source/developer_guide/contribution/multi_node_test.md:381; docs/source/developer_guide/contribution/multi_node_test.md:464; tests/e2e/nightly/multi_node/scripts/run.sh:60 |
| `BENCHMARK_JOB_NAME` | 仓库工具/文档专用 | CI/发布, 测试 | Python/C/C++ API, YAML env | tests/e2e/nightly/multi_node/external_dp/scripts/utils.py:235; tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py:151; tests/e2e/nightly/single_node/models/scripts/test_single_node.py:449; .github/workflows/_e2e_nightly_single_node.yaml:310; .github/workflows/_e2e_nightly_single_node_560t.yaml:308 |
| `BISECT_ARGS_JSON` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_nightly_command.yml:362; .github/workflows/pr_nightly_command.yml:405; .github/workflows/pr_nightly_command.yml:448; .github/workflows/pr_nightly_command.yml:614 |
| `BISECT_BAD_COMMIT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:248; .github/workflows/_e2e_nightly_multi_node_560t.yaml:271; .github/workflows/_e2e_nightly_single_node.yaml:426; .github/workflows/_e2e_nightly_single_node_560t.yaml:414 |
| `BISECT_BARRIER_TIMEOUT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:251; .github/workflows/_e2e_nightly_multi_node_560t.yaml:274; .github/workflows/_e2e_nightly_single_node.yaml:430; .github/workflows/_e2e_nightly_single_node_560t.yaml:418 |
| `BISECT_CONFIG_BASE_PATH` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:255; .github/workflows/_e2e_nightly_multi_node_560t.yaml:278; .github/workflows/_e2e_nightly_single_node.yaml:434; .github/workflows/_e2e_nightly_single_node_560t.yaml:422 |
| `BISECT_COORD_DIR` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | tools/bisect/config.py:60 |
| `BISECT_FAIL_CONFIRM_RETRIES` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:249; .github/workflows/_e2e_nightly_multi_node_560t.yaml:272; .github/workflows/_e2e_nightly_single_node.yaml:428; .github/workflows/_e2e_nightly_single_node_560t.yaml:416 |
| `BISECT_FORCE_INITIAL_BUILD` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:254; .github/workflows/_e2e_nightly_multi_node_560t.yaml:277; .github/workflows/_e2e_nightly_single_node.yaml:433; .github/workflows/_e2e_nightly_single_node_560t.yaml:421 |
| `BISECT_GOOD_COMMIT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:247; .github/workflows/_e2e_nightly_multi_node_560t.yaml:270; .github/workflows/_e2e_nightly_single_node.yaml:427; .github/workflows/_e2e_nightly_single_node_560t.yaml:415 |
| `BISECT_GOOD_TABLE` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | tools/bisect/config.py:49 |
| `BISECT_NO_VERIFY_BAD` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:253; .github/workflows/_e2e_nightly_multi_node_560t.yaml:276; .github/workflows/_e2e_nightly_single_node.yaml:432; .github/workflows/_e2e_nightly_single_node_560t.yaml:420 |
| `BISECT_NO_VERIFY_GOOD` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:252; .github/workflows/_e2e_nightly_multi_node_560t.yaml:275; .github/workflows/_e2e_nightly_single_node.yaml:431; .github/workflows/_e2e_nightly_single_node_560t.yaml:419 |
| `BISECT_TRIAL_TIMEOUT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:250; .github/workflows/_e2e_nightly_multi_node_560t.yaml:273; .github/workflows/_e2e_nightly_single_node.yaml:429; .github/workflows/_e2e_nightly_single_node_560t.yaml:417 |
| `BISECT_WORK_DIR` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | tools/bisect/config.py:56 |
| `BISHENG_REAL_PATH` | 仓库工具/文档专用 | 构建/打包 | Python/C/C++ API | csrc/cmake/scripts/util/ascendc_impl_build.py:177 |
| `BRANCH` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_update_estimated_times.yaml:214 |
| `BUFFER_MINUTES` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:84 |
| `BUILD_BUILTIN_OPP` | 仓库工具/文档专用 | 构建/打包 | Python/C/C++ API | csrc/cmake/scripts/util/ascendc_impl_build.py:368; csrc/cmake/scripts/util/ascendc_impl_build.py:663 |
| `BUILD_KERNEL_SRC` | 仓库工具/文档专用 | 构建/打包 | Python/C/C++ API | csrc/cmake/scripts/util/ascendc_impl_build.py:145; csrc/cmake/scripts/util/ascendc_impl_build.py:585 |
| `BUILD_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:328 |
| `BUILD_TYPE` | 仓库工具/文档专用 | CI/发布, 构建/打包 | Docker ARG, YAML env | Dockerfile:82; Dockerfile.310p:113; Dockerfile.310p.openEuler:109; Dockerfile.a3:85; .github/workflows/_schedule_image_build.yaml:528 |
| `CACHE_REPO` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_schedule_image_build.yaml:146 |
| `CANN_IMAGE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_nightly_test_a2.yaml:95; .github/workflows/schedule_weekly_test_a2.yaml:91 |
| `CANN_QUAY_URL` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | Dockerfile:17; Dockerfile.310p:17; Dockerfile.310p.openEuler:17; Dockerfile.a3:17 |
| `CANN_VERSION` | 仓库工具/文档专用 | CI/发布, 构建/打包, 测试 | Docker ARG, Python/C/C++ API, YAML env | Dockerfile:18; Dockerfile.310p:19; Dockerfile.310p.openEuler:19; Dockerfile.a3:19; tests/e2e/models/test_asr_eval_correctness.py:47; tests/e2e/models/test_lm_eval_correctness.py:32; tests/e2e/models/test_rm_eval_correctness.py:39; .github/workflows/_e2e_nightly_single_node_models.yaml:229 |
| `CC` | 系统/分布式通用 | 运行时代码 | Python/C/C++ API | vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:385 |
| `CHANGED_DOCS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_doc_linkcheck.yaml:60 |
| `CHECKOUT_SOURCE_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:344 |
| `CMAKE_BUILD_TYPE` | 仓库工具/文档专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:37 |
| `COMMAND` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_nightly_command.yml:118 |
| `COMPILER_INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:412 |
| `COMPILE_CUSTOM_KERNELS` | 仓库工具/文档专用 | CI/发布, 文档/示例, 构建/打包, 运行时代码 | Docker ARG, Python/C/C++ API, Shell export, YAML env | Dockerfile:60; Dockerfile.310p:55; Dockerfile.310p.openEuler:54; Dockerfile.a3:63; vllm_ascend/envs.py:43; docs/source/installation.md:281; .github/workflows/_selected_tests.yaml:265 |
| `CONDA_EXE` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | collect_env.py:132 |
| `CONFIG_BASE_PATH` | 仓库工具/文档专用 | CI/发布, 工具/辅助, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | tests/e2e/nightly/multi_node/internal_dp/scripts/utils.py:14; tests/e2e/nightly/multi_node/scripts/utils.py:50; tests/e2e/nightly/scripts/result_postprocess.py:65; tests/e2e/nightly/single_node/models/scripts/single_node_config.py:10; docs/source/developer_guide/contribution/multi_node_test.md:402; docs/source/developer_guide/contribution/multi_node_test.md:415; docs/source/developer_guide/contribution/multi_node_test.md:489; docs/source/developer_guide/contribution/multi_node_test.md:502; .github/workflows/_e2e_nightly_multi_node.yaml:412; .github/workflows/_e2e_nightly_multi_node_560t.yaml:450; .github/workflows/_e2e_nightly_single_node.yaml:309; .github/workflows/_e2e_nightly_single_node.yaml:332 |
| `CONFIG_YAML_PATH` | 仓库工具/文档专用 | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | tests/e2e/nightly/multi_node/scripts/utils.py:46; tests/e2e/nightly/scripts/result_postprocess.py:72; tests/e2e/nightly/single_node/models/scripts/single_node_config.py:142; docs/source/developer_guide/contribution/multi_node_test.md:401; docs/source/developer_guide/contribution/multi_node_test.md:414; docs/source/developer_guide/contribution/multi_node_test.md:490; docs/source/developer_guide/contribution/multi_node_test.md:503; .github/workflows/_e2e_nightly_single_node.yaml:308; .github/workflows/_e2e_nightly_single_node_560t.yaml:306 |
| `CONTEXT_PREFIX` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:79 |
| `CPU_AFFINITY_CONF` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:128; docs/source/tutorials/models/GLM5.2.md:873; docs/source/tutorials/models/GLM5.2.md:923 |
| `CSRC_CACHE_HIT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_selected_tests.yaml:229; .github/workflows/_selected_tests_upstream.yaml:214; .github/workflows/schedule_e2e_upstream_test.yaml:168; .github/workflows/schedule_e2e_upstream_test.yaml:306 |
| `CXX` | 系统/分布式通用 | 运行时代码 | Python/C/C++ API | vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:384 |
| `CXX_COMPILER` | 仓库工具/文档专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:46 |
| `C_COMPILER` | 仓库工具/文档专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:49 |
| `DAILY_DEPS_MODE` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | Dockerfile:91; Dockerfile.310p:116; Dockerfile.310p.openEuler:112; Dockerfile.a3:94 |
| `DATASET_SOURCE` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/evaluation/using_opencompass.md:65 |
| `DATASYSTEM_CLIENT_LOG_DIR` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:948 |
| `DEBIAN_FRONTEND` | 仓库工具/文档专用 | 构建/打包 | Docker ENV | .github/workflows/dockerfiles/Dockerfile.buildwheel.310p:23; .github/workflows/dockerfiles/Dockerfile.buildwheel.a2:23; .github/workflows/dockerfiles/Dockerfile.buildwheel.a3:23; .github/workflows/dockerfiles/Dockerfile.nightly.310p:31 |
| `DEEPSEEK_API_KEY` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/scripts/po_translate.py:508; .github/workflows/scripts/po_translate.py:526; .github/workflows/schedule_doc_translate.yaml:104 |
| `DEVICE` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/evaluation/using_ais_bench.md:13; docs/source/developer_guide/evaluation/using_evalscope.md:11; docs/source/developer_guide/evaluation/using_lm_eval.md:13; docs/source/developer_guide/evaluation/using_lm_eval.md:158 |
| `DOCS_IS_RELEASE` | 仓库工具/文档专用 | 工具/辅助 | Shell export | tools/rtd_build.sh:39 |
| `DOCS_LANG` | 仓库工具/文档专用 | 工具/辅助, 文档/示例 | Python/C/C++ API, Shell export | docs/hooks/nav_titles.py:266; docs/hooks/nav_titles.py:273; tools/rtd_build.sh:33 |
| `DYNAMIC_EPLB` | 仓库工具/文档专用 | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | tests/ut/eplb/core/a2/test_eplb_utils.py:48; tests/ut/eplb/core/a2/test_eplb_utils.py:52; vllm_ascend/ascend_config.py:910; vllm_ascend/envs.py:86; docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:257 |
| `EAGER_INCLUDE_OPP_ACLNNOP_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:422 |
| `EAGER_LIBRARY_OPP_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:417 |
| `EAGER_LIBRARY_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:418 |
| `ENABLE_COVERAGE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_selected_tests.yaml:281; .github/workflows/_selected_tests.yaml:304 |
| `ENDPOINT` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/user_guide/deployment_guide/using_volcano_kthena.md:390 |
| `EVENT_BASE_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:216 |
| `EXPERT_MAP_RECORD` | 仓库工具/文档专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/ascend_config.py:911; vllm_ascend/patch/platform/__init__.py:36; vllm_ascend/platform.py:780 |
| `EXTERNAL_DP_LOG_DIR` | 仓库工具/文档专用 | 文档/示例, 测试 | Python/C/C++ API, Shell export | tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py:107; docs/source/developer_guide/contribution/multi_node_test.md:538 |
| `EXTERNAL_DP_MAX_WAIT_SECONDS` | 仓库工具/文档专用 | 测试 | Python/C/C++ API | tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py:108 |
| `FETCHCONTENT_BASE_DIR` | 仓库工具/文档专用 | 构建/打包 | Python/C/C++ API | setup.py:304 |
| `FILE_COUNT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_doc_translate.yaml:169 |
| `FILE_LIST` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_doc_translate.yaml:168 |
| `FILTER` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:78 |
| `FIXED_BASE_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_selected_tests.yaml:145; .github/workflows/_selected_tests_upstream.yaml:135 |
| `FORCE_COLOR` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:171; .github/workflows/push_build_precommit_cache.yaml:83 |
| `GE_INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:414 |
| `GH_REPO` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/bot_pr_create.yaml:69 |
| `GH_TOKEN` | CI/凭据/包管理 | CI/发布, 工具/辅助 | Python/C/C++ API, YAML env | .agents/skills/vllm-ascend-release/scripts/fetch_commits.py:21; .github/workflows/_e2e_nightly_multi_node_560t.yaml:380; .github/workflows/pr_cherry_pick_command.yml:125; .github/workflows/pr_e2e_command.yml:63; .github/workflows/pr_nightly_command.yml:76 |
| `GITEE_TOKEN` | 仓库工具/文档专用 | CI/发布, 构建/打包 | Docker ARG, YAML env | .github/workflows/dockerfiles/Dockerfile.nightly.310p:28; .github/workflows/dockerfiles/Dockerfile.nightly.a2:28; .github/workflows/dockerfiles/Dockerfile.nightly.a3:28; .github/workflows/dockerfiles/Dockerfile.nightly.a5:28; .github/workflows/_nightly_image_build.yaml:124 |
| `GITEE_USERNAME` | 仓库工具/文档专用 | CI/发布, 构建/打包 | Docker ARG, YAML env | .github/workflows/dockerfiles/Dockerfile.nightly.310p:27; .github/workflows/dockerfiles/Dockerfile.nightly.a2:27; .github/workflows/dockerfiles/Dockerfile.nightly.a3:27; .github/workflows/dockerfiles/Dockerfile.nightly.a5:27; .github/workflows/_nightly_image_build.yaml:123 |
| `GITHUB_ACTIONS` | CI/凭据/包管理 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/run_suite.py:61; .github/workflows/scripts/run_suite.py:66; .github/workflows/scripts/run_suite.py:71 |
| `GITHUB_OUTPUT` | CI/凭据/包管理 | CI/发布 | Python/C/C++ API | .github/workflows/_ensure_csrc_cache.yaml:243; .github/workflows/pr_test.yaml:328; .github/workflows/scripts/assemble_coverage.py:134; .github/workflows/scripts/resolve_csrc_cache_targets.py:54 |
| `GITHUB_REPO` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:174 |
| `GITHUB_RUN_ID` | CI/凭据/包管理 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/run_suite.py:328 |
| `GITHUB_SHA` | CI/凭据/包管理 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/run_suite.py:327 |
| `GITHUB_TOKEN` | CI/凭据/包管理 | CI/发布, 工具/辅助 | Python/C/C++ API, YAML env | .agents/skills/vllm-ascend-release/scripts/fetch_commits.py:21; .github/workflows/bot_pr_create.yaml:47; .github/workflows/bot_pr_create.yaml:68; .github/workflows/bot_pr_create.yaml:151; .github/workflows/pr_cherry_pick_command.yml:53 |
| `GLOG_minloglevel` | 仓库工具/文档专用 | 测试 | Shell export | tests/e2e/nightly/multi_node/scripts/run.sh:65 |
| `GLOO_SOCKET_IFNAME` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/doc_writing.md:222; docs/source/developer_guide/contribution/doc_writing.md:235; docs/source/tutorials/features/ray.md:113; docs/source/tutorials/features/ray.md:125 |
| `GONOSUMDB` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:176; .github/workflows/schedule_main2main.yaml:165 |
| `GOOD_TABLE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_single_node.yaml:424; .github/workflows/_e2e_nightly_single_node_560t.yaml:412 |
| `GOPROXY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:175; .github/workflows/schedule_main2main.yaml:164 |
| `GRAPH_INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:413 |
| `GRAPH_LIBRARY_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:420 |
| `GRAPH_LIBRARY_STUB_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:419 |
| `HAS_READY_A5_LABEL` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:513 |
| `HAS_READY_LABEL` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:512 |
| `HCCL_BUFFSIZE` | Ascend/通信/KV 组件 | CI/发布, 文档/示例 | Python/C/C++ API, Shell export, YAML env | docs/source/user_guide/feature_guide/graph_mode.md:276; docs/source/developer_guide/contribution/doc_writing.md:90; docs/source/developer_guide/contribution/doc_writing.md:239; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:71; docs/source/tutorials/models/DeepSeek-V3.2.md:133; .github/workflows/schedule_main2main.yaml:152 |
| `HCCL_CONNECT_TIMEOUT` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:83; docs/source/tutorials/models/GLM5.2.md:138; docs/source/tutorials/models/GLM5.2.md:215; docs/source/tutorials/models/GLM5.2.md:269 |
| `HCCL_DETERMINISTIC` | Ascend/通信/KV 组件 | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | tests/ut/test_batch_invariant.py:37; vllm_ascend/batch_invariant.py:82; examples/external_online_dp/run_dp_template.sh:9 |
| `HCCL_EXEC_TIMEOUT` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:82; docs/source/tutorials/models/GLM5.2.md:137; docs/source/tutorials/models/GLM5.2.md:214; docs/source/tutorials/models/GLM5.2.md:268 |
| `HCCL_IF_IP` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/doc_writing.md:221; docs/source/developer_guide/contribution/doc_writing.md:234; docs/source/tutorials/features/ray.md:112; docs/source/tutorials/features/ray.md:124 |
| `HCCL_INTRA_PCIE_ENABLE` | Ascend/通信/KV 组件 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/utils.py:1034; docs/source/tutorials/models/MiniMax-M2.md:225 |
| `HCCL_INTRA_ROCE_ENABLE` | Ascend/通信/KV 组件 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/utils.py:1034; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:217; docs/source/tutorials/models/GLM5.2.md:991; docs/source/tutorials/models/GLM5.2.md:1084; docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:71 |
| `HCCL_OP_EXPANSION_MODE` | Ascend/通信/KV 组件 | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | docs/source/user_guide/feature_guide/graph_mode.md:279; tests/e2e/pull_request/eight_card/test_minimax_m3.py:43; tests/e2e/pull_request/two_card/test_shared_expert_dp.py:22; vllm_ascend/device/device_op.py:1122; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:140; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:73; docs/source/tutorials/features/suffix_speculative_decoding.md:84; docs/source/tutorials/models/DeepSeek-R1.md:137 |
| `HCCL_RDMA_RETRY_CNT` | Ascend/通信/KV 组件 | 运行时代码 | Python/C/C++ API | vllm_ascend/distributed/kv_transfer/utils/utils.py:60 |
| `HCCL_RDMA_TIMEOUT` | Ascend/通信/KV 组件 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/distributed/kv_transfer/utils/utils.py:59; docs/source/user_guide/feature_guide/kv_pool.md:170; docs/source/user_guide/feature_guide/kv_pool.md:249; docs/source/user_guide/feature_guide/kv_pool.md:370 |
| `HCCL_SOCKET_IFNAME` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/doc_writing.md:224; docs/source/developer_guide/contribution/doc_writing.md:237; docs/source/tutorials/models/DeepSeek-R1.md:142; docs/source/tutorials/models/DeepSeek-V3.1.md:153 |
| `HCCL_SO_PATH` | Ascend/通信/KV 组件 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:61 |
| `HCCL_TRANSFER_TIMEOUT` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:136; docs/source/tutorials/models/GLM5.2.md:213; docs/source/tutorials/models/GLM5.2.md:267; docs/source/tutorials/models/GLM5.2.md:1263 |
| `HEAD_FORK` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:58; .github/workflows/schedule_main2main.yaml:175 |
| `HF_DATASETS_CACHE` | 仓库工具/文档专用 | 文档/示例 | Python/C/C++ API | docs/source/tutorials/models/Qwen3-Embedding.md:222; docs/source/tutorials/models/Qwen3-Reranker.md:251; docs/source/tutorials/models/Qwen3-VL-Embedding.md:226; docs/source/tutorials/models/Qwen3-VL-Reranker.md:256 |
| `HF_DATASETS_OFFLINE` | 仓库工具/文档专用 | CI/发布, 文档/示例 | Shell export, YAML env | docs/source/developer_guide/evaluation/using_lm_eval.md:227; .github/workflows/_e2e_nightly_single_node_models.yaml:221 |
| `HF_ENDPOINT` | 仓库工具/文档专用 | 文档/示例 | Python/C/C++ API, Shell export | docs/source/tutorials/models/Qwen3-Embedding.md:223; docs/source/tutorials/models/Qwen3-Reranker.md:252; docs/source/tutorials/models/Qwen3-VL-Embedding.md:227; docs/source/tutorials/models/Qwen3-VL-Reranker.md:257; docs/source/developer_guide/evaluation/using_lm_eval.md:115; docs/source/developer_guide/evaluation/using_lm_eval.md:186; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:177 |
| `HF_HOME` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:73 |
| `HF_HUB_OFFLINE` | 仓库工具/文档专用 | CI/发布, 测试 | Python/C/C++ API, Shell export, YAML env | tests/e2e/pull_request/four_card/spec_decode/test_mtp_step3p5.py:33; tests/e2e/nightly/multi_node/scripts/run.sh:67; tests/e2e/run_doctests.sh:26; .github/workflows/_e2e_nightly_single_node.yaml:156; .github/workflows/_e2e_nightly_single_node_560t.yaml:155; .github/workflows/_e2e_nightly_single_node_models.yaml:223; .github/workflows/_selected_tests.yaml:97 |
| `HITEST_APIG_APPCODE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_manual-hitest.yaml:57 |
| `HITEST_KEY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_manual-hitest.yaml:58 |
| `HITEST_SECRET` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_manual-hitest.yaml:59 |
| `HOME` | 系统/分布式通用 | 构建/打包 | Python/C/C++ API | csrc/scripts/package/common/py/pkg_parser.py:409 |
| `HW_TOKEN` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_image_build.yaml:100 |
| `HW_TOKEN_DAILY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_image_build.yaml:109; .github/workflows/_nightly_image_build.yaml:240 |
| `HW_USERNAME` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_image_build.yaml:99 |
| `HW_USERNAME_DAILY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_image_build.yaml:108; .github/workflows/_nightly_image_build.yaml:239 |
| `IMAGE` | 仓库工具/文档专用 | CI/发布, 文档/示例 | Shell export, YAML env | docs/source/_templates/template-supplement.md:119; docs/source/_templates/template-supplement.md:127; docs/source/_templates/template-supplement.zh.md:119; docs/source/_templates/template-supplement.zh.md:127; .github/workflows/_schedule_image_build.yaml:526; .github/workflows/_schedule_image_build.yaml:600 |
| `INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:410 |
| `INC_INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:415 |
| `INPUT_DOC_VERSIONS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/labeled_doctest.yaml:147 |
| `IP_ADDRESS` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:320; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:400; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:482 |
| `IS_A3_560T` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/resolve_nightly_tests.py:74 |
| `IS_PR_TEST` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/multi_node_test.md:400; docs/source/developer_guide/contribution/multi_node_test.md:413; docs/source/developer_guide/contribution/multi_node_test.md:488; docs/source/developer_guide/contribution/multi_node_test.md:501 |
| `KUBECONFIG` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:164; .github/workflows/_e2e_nightly_multi_node_560t.yaml:191; .github/workflows/schedule_nightly_test_a2.yaml:264; .github/workflows/schedule_nightly_test_a3.yaml:427 |
| `LCCL_DETERMINISTIC` | Ascend/通信/KV 组件 | 测试, 运行时代码 | Python/C/C++ API | tests/ut/test_batch_invariant.py:38; vllm_ascend/batch_invariant.py:83 |
| `LD_LIBRARY_PATH` | 系统/分布式通用 | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export | tests/e2e/conftest.py:244; tests/e2e/conftest.py:246; tests/e2e/pull_request/four_card/test_deepseek_v3_2_w8a8_pruning.py:59; .github/workflows/_e2e_nightly_single_node.yaml:292; .github/workflows/_e2e_nightly_single_node.yaml:312; .github/workflows/_e2e_nightly_single_node_560t.yaml:290; .github/workflows/_e2e_nightly_single_node_560t.yaml:310 |
| `LD_PRELOAD` | 系统/分布式通用 | 文档/示例, 测试 | Python/C/C++ API, Shell export | tests/e2e/pull_request/eight_card/test_minimax_m3.py:70; tests/e2e/pull_request/eight_card/test_minimax_m3.py:71; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:78; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:96; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:72; docs/source/tutorials/models/Kimi-K2.5.md:132 |
| `LINUX_INCLUDE_PATH` | 仓库工具/文档专用 | 构建/打包 | Shell export | csrc/build.sh:416 |
| `LOCAL_RANK` | 系统/分布式通用 | 文档/示例 | Python/C/C++ API | examples/offline_external_launcher.py:152; examples/offline_weight_load.py:156 |
| `LOCAL_WORLD_SIZE` | 系统/分布式通用 | 运行时代码 | Python/C/C++ API | vllm_ascend/compilation/compiler_interface.py:101 |
| `LOG_PREFIX` | 仓库工具/文档专用 | 测试 | Python/C/C++ API | tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py:94 |
| `LWS_LEADER_ADDRESS` | 仓库工具/文档专用 | 测试 | Python/C/C++ API | tests/e2e/nightly/multi_node/scripts/utils.py:81 |
| `LWS_WORKER_INDEX` | 仓库工具/文档专用 | 工具/辅助, 文档/示例, 测试 | Python/C/C++ API, Shell export | tests/e2e/conftest.py:328; tests/e2e/nightly/multi_node/external_dp/scripts/external_dp_config.py:160; tests/e2e/nightly/multi_node/scripts/utils.py:175; tools/bisect/auto_bisect.py:323; docs/source/developer_guide/contribution/multi_node_test.md:403; docs/source/developer_guide/contribution/multi_node_test.md:416; docs/source/developer_guide/contribution/multi_node_test.md:491; docs/source/developer_guide/contribution/multi_node_test.md:504 |
| `MAIN2MAIN_CASES_FILE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:178 |
| `MAIN2MAIN_IMAGE_TAG` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:171 |
| `MAIN2MAIN_KEEP_BRANCH` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:176 |
| `MAIN2MAIN_LOG_HELPERS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:179 |
| `MAIN2MAIN_MODEL` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:45 |
| `MAIN2MAIN_WORKSPACE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:177 |
| `MASTER_ADDR` | 系统/分布式通用 | 文档/示例 | Python/C/C++ API | examples/offline_external_launcher.py:149; examples/offline_weight_load.py:153 |
| `MASTER_IP` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/doc_writing.md:150 |
| `MASTER_IP_ADDRESS` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:399; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:481 |
| `MASTER_PORT` | 系统/分布式通用 | 文档/示例 | Python/C/C++ API | examples/offline_external_launcher.py:150; examples/offline_weight_load.py:154 |
| `MATRIX_FILE` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/scripts/resolve_nightly_tests.py:121; .github/workflows/schedule_nightly_test_310p.yaml:168; .github/workflows/schedule_nightly_test_a2.yaml:174; .github/workflows/schedule_nightly_test_a2.yaml:352; .github/workflows/schedule_nightly_test_a3.yaml:173 |
| `MATRIX_JSON` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:77 |
| `MATRIX_OUTPUTS` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/scripts/resolve_nightly_tests.py:124; .github/workflows/schedule_nightly_test_310p.yaml:169; .github/workflows/schedule_nightly_test_a2.yaml:175; .github/workflows/schedule_nightly_test_a2.yaml:353; .github/workflows/schedule_nightly_test_a3.yaml:174 |
| `MAX_JOBS` | 仓库工具/文档专用 | CI/发布, 运行时代码 | Python/C/C++ API, Shell export, YAML env | vllm_ascend/envs.py:34; .github/workflows/_selected_tests.yaml:230; .github/workflows/_selected_tests_upstream.yaml:215; .github/workflows/schedule_e2e_upstream_test.yaml:169; .github/workflows/schedule_e2e_upstream_test.yaml:307; .github/workflows/_build_csrc_cache.yaml:170; .github/workflows/_e2e_nightly_single_node.yaml:206; .github/workflows/_e2e_nightly_single_node_560t.yaml:205; .github/workflows/_e2e_nightly_single_node_models.yaml:131 |
| `MAX_PARALLEL` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:81 |
| `MEMCACHE_DATE` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | Dockerfile:84; Dockerfile.a3:87; Dockerfile.a3.openEuler:82; Dockerfile.a5:79 |
| `MEMCACHE_VERSION` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | Dockerfile:83; Dockerfile.a3:86; Dockerfile.a3.openEuler:81; Dockerfile.a5:78 |
| `MEMFABRIC_DATE` | Ascend/通信/KV 组件 | 构建/打包 | Docker ARG | Dockerfile:86; Dockerfile.a3:89; Dockerfile.a3.openEuler:84; Dockerfile.a5:81 |
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:199 |
| `MEMFABRIC_VERSION` | Ascend/通信/KV 组件 | 构建/打包 | Docker ARG | Dockerfile:85; Dockerfile.a3:88; Dockerfile.a3.openEuler:83; Dockerfile.a5:80 |
| `MERGE_COMMIT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_revert_command.yml:148 |
| `MINIMAX_M3_DECODE_BOUNDARY_REPEATS` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py:1277 |
| `MINIMAX_M3_DECODE_BOUNDARY_ROUNDS` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py:1273 |
| `MINIMAX_M3_MODEL_PATH` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/e2e/pull_request/eight_card/test_minimax_m3.py:30 |
| `MINIMAX_M3_SPARSE_BACKEND` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/e2e/conftest.py:100 |
| `MMC_LOCAL_CONFIG_PATH` | Ascend/通信/KV 组件 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py:17; docs/source/user_guide/feature_guide/kv_pool.md:562; docs/source/user_guide/feature_guide/kv_pool.md:684; docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:111 |
| `MMC_META_CONFIG_PATH` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:533; docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md:110 |
| `MM_IMAGE_PATH` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | tools/send_mm_request.py:37; tools/send_mm_request.py:51 |
| `MODEL` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:394 |
| `MODELSCOPE_HUB_FILE_LOCK` | 仓库工具/文档专用 | 测试 | Shell export | tests/e2e/run_doctests.sh:25 |
| `MODEL_ARGS` | 仓库工具/文档专用 | 测试 | Shell export | tests/e2e/models/report_template.md:13 |
| `MODEL_PATH` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/Gemma4.md:82; docs/source/tutorials/models/Gemma4.md:98; docs/source/tutorials/models/Hunyuan-A13B-Instruct.md:74; docs/source/tutorials/models/Hy3-preview.md:81 |
| `MOONCAKE_CONFIG_PATH` | Ascend/通信/KV 组件 | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | tests/ut/distributed/ascend_store/test_backend.py:182; vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:319; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:215; docs/source/tutorials/models/GLM5.2.md:990; docs/source/tutorials/models/GLM5.2.md:1083; docs/source/user_guide/feature_guide/kv_pool.md:155 |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | Ascend/通信/KV 组件 | 运行时代码 | Python/C/C++ API | vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:297 |
| `MOONCAKE_MASTER` | Ascend/通信/KV 组件 | 运行时代码 | Python/C/C++ API | vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:296 |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:475 |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:474 |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:476; docs/source/user_guide/feature_guide/kv_pool.md:1222; docs/source/user_guide/feature_guide/kv_pool.md:1247 |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | Ascend/通信/KV 组件 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:473 |
| `MOONCAKE_TAG` | Ascend/通信/KV 组件 | 构建/打包 | Docker ARG | Dockerfile:27; Dockerfile.a3:30; Dockerfile.a3.openEuler:30; Dockerfile.a5:30 |
| `MSMONITOR_USE_DAEMON` | 仓库工具/文档专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:74 |
| `NAME` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:23; docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:84; docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:128; docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md:65 |
| `NAMESPACE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:165; .github/workflows/_e2e_nightly_multi_node_560t.yaml:192 |
| `NETLOADER_CONFIG` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/netloader.md:68 |
| `NETWORK_CARD_NAME` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/Qwen3.5-397B-A17B.md:321; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:401; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:483 |
| `NIGHTLY_MATRIX` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/resolve_nightly_tests.py:67 |
| `NODE_VERSION` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:274 |
| `NPU_MEMORY_FRACTION` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/gpt-oss-120b.md:104 |
| `OBS_ACCESS_KEY` | CI/凭据/包管理 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/_selected_tests.yaml:486; .github/workflows/schedule_release_code_and_wheel.yml:386; .github/workflows/_selected_tests.yaml:355; .github/workflows/schedule_release_code_and_wheel.yml:376 |
| `OBS_SECRET_KEY` | CI/凭据/包管理 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/_selected_tests.yaml:487; .github/workflows/schedule_release_code_and_wheel.yml:387; .github/workflows/_selected_tests.yaml:356; .github/workflows/schedule_release_code_and_wheel.yml:377 |
| `OMP_NUM_THREADS` | 系统/分布式通用 | 文档/示例, 测试 | Python/C/C++ API, Shell export | tests/e2e/pull_request/eight_card/test_minimax_m3.py:44; tests/e2e/pull_request/four_card/test_graph_mode.py:616; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:70; docs/source/tutorials/models/DeepSeek-V3.2.md:131; docs/source/tutorials/models/GLM4.x.md:130; docs/source/tutorials/models/GLM5.2.md:140 |
| `OMP_PROC_BIND` | 系统/分布式通用 | 文档/示例 | Python/C/C++ API, Shell export | docs/source/user_guide/feature_guide/graph_mode.md:278; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:68; docs/source/tutorials/models/DeepSeek-V3.2.md:130; docs/source/tutorials/models/GLM4.x.md:129; docs/source/tutorials/models/GLM5.2.md:139 |
| `OPENAI_API_KEY` | 仓库工具/文档专用 | 文档/示例 | Python/C/C++ API | examples/disaggregated_encoder/disagg_epd_proxy.py:650; examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:413; examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:441; examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:629 |
| `OPENLIBING_SECRET` | 仓库工具/文档专用 | CI/发布, 工具/辅助 | Python/C/C++ API, YAML env | tools/upload_to_openlibing.py:217; .github/workflows/_e2e_nightly_multi_node.yaml:166; .github/workflows/_e2e_nightly_multi_node_560t.yaml:193; .github/workflows/_e2e_nightly_single_node.yaml:164; .github/workflows/_e2e_nightly_single_node_560t.yaml:163 |
| `OS_MARK` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.nightly.310p:20; .github/workflows/dockerfiles/Dockerfile.nightly.a2:20; .github/workflows/dockerfiles/Dockerfile.nightly.a3:20; .github/workflows/dockerfiles/Dockerfile.nightly.a5:20 |
| `OUTPUT_DIR` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_release_code_and_wheel.yml:157; .github/workflows/schedule_release_code_and_wheel.yml:244; .github/workflows/schedule_release_code_and_wheel.yml:325 |
| `OUTPUT_JSON` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/detect_po_changes.py:549; .github/workflows/scripts/po_translate.py:507 |
| `P310_CACHE_HIT` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/_ensure_csrc_cache.yaml:213; .github/workflows/_ensure_csrc_cache.yaml:203 |
| `PHYSICAL_DEVICES` | 仓库工具/文档专用 | 文档/示例 | Shell export | examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md:26; examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md:92 |
| `PIP_EXTRA_INDEX_URL` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_single_node.yaml:207; .github/workflows/_e2e_nightly_single_node_560t.yaml:206; .github/workflows/_e2e_nightly_single_node_models.yaml:132 |
| `PIP_INDEX_URL` | CI/凭据/包管理 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.nightly.310p:23; .github/workflows/dockerfiles/Dockerfile.nightly.a2:23; .github/workflows/dockerfiles/Dockerfile.nightly.a3:23; .github/workflows/dockerfiles/Dockerfile.nightly.a5:23 |
| `POD_WAIT_MINUTES` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:83 |
| `PRE_COMMIT_COLOR` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:170; .github/workflows/push_build_precommit_cache.yaml:82 |
| `PRE_COMMIT_HOME` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:169; .github/workflows/push_build_precommit_cache.yaml:81; .github/workflows/schedule_main2main.yaml:161 |
| `PROFILING_SYMBOLS_PATH` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:163 |
| `PROJECT_TOML` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_release_code_and_wheel.yml:156; .github/workflows/schedule_release_code_and_wheel.yml:243; .github/workflows/schedule_release_code_and_wheel.yml:324 |
| `PR_AUTHOR` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:55; .github/workflows/pr_cherry_pick_command.yml:130; .github/workflows/pr_e2e_command.yml:65; .github/workflows/pr_rerun_command.yml:57 |
| `PR_BASE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_doc_translate.yaml:172 |
| `PR_BASE_REF` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_e2e_command.yml:140 |
| `PR_BASE_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:131; .github/workflows/pr_e2e_command.yml:141 |
| `PR_BODY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:129; .github/workflows/pr_revert_command.yml:146 |
| `PR_HEAD` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_doc_translate.yaml:171 |
| `PR_HEAD_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:132 |
| `PR_NUMBER` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_manual-hitest.yaml:60; .github/workflows/pr_cherry_pick_command.yml:127; .github/workflows/pr_revert_command.yml:79; .github/workflows/pr_revert_command.yml:144 |
| `PR_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_e2e_command.yml:139; .github/workflows/pr_rerun_command.yml:96 |
| `PR_TITLE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:128; .github/workflows/pr_revert_command.yml:145 |
| `PR_URL` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_nightly_command.yml:117 |
| `PUSH_TO_GITHUB` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:173 |
| `PYTHONHASHSEED` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:989; docs/source/tutorials/models/GLM5.2.md:1082; docs/source/tutorials/models/MiniMax-M2.md:300; docs/source/tutorials/models/MiniMax-M2.md:362 |
| `PYTHONPATH` | 系统/分布式通用 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/kv_pool.md:154; docs/source/user_guide/feature_guide/kv_pool.md:236; docs/source/user_guide/feature_guide/kv_pool.md:357; docs/source/user_guide/feature_guide/ucm_deployment.md:256 |
| `PYTORCH_NPU_ALLOC_CONF` | vLLM/PyTorch 上游 | CI/发布, 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export, YAML env | docs/source/user_guide/feature_guide/graph_mode.md:280; tests/e2e/pull_request/eight_card/test_minimax_m3.py:45; tests/e2e/pull_request/four_card/test_deepseek_v4.py:26; vllm_ascend/device_allocator/camem.py:152; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:111; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:118; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:69; docs/source/tutorials/models/DeepSeek-R1.md:144; .github/workflows/_selected_tests_upstream.yaml:234; .github/workflows/schedule_e2e_upstream_test.yaml:192; .github/workflows/schedule_e2e_upstream_test.yaml:326; .github/workflows/schedule_e2e_upstream_test.yaml:457 |
| `PY_VERSION` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.buildwheel.310p:17; .github/workflows/dockerfiles/Dockerfile.buildwheel.a2:17; .github/workflows/dockerfiles/Dockerfile.buildwheel.a3:17 |
| `QUAY_DAILY_PASSWORD` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_nightly_image_build.yaml:116 |
| `QUAY_REPO` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_schedule_image_build.yaml:143 |
| `QUAY_TEMP_REPO` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_schedule_image_build.yaml:144 |
| `QWEN3_MRV2_EPLB_MODEL_PATH` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/e2e/pull_request/four_card/test_qwen3_mrv2_eplb.py:13 |
| `RANK` | 系统/分布式通用 | 文档/示例 | Python/C/C++ API | examples/offline_external_launcher.py:151; examples/offline_weight_load.py:155 |
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | 仓库工具/文档专用 | CI/发布, 文档/示例 | Shell export, YAML env | docs/source/tutorials/features/ray.md:115; docs/source/tutorials/features/ray.md:127; .github/workflows/schedule_main2main.yaml:449 |
| `READTHEDOCS_VERSION_TYPE` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | tools/set_release_flag.py:10 |
| `REPO` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:56; .github/workflows/pr_cherry_pick_command.yml:134; .github/workflows/pr_e2e_command.yml:66; .github/workflows/pr_nightly_command.yml:78 |
| `RFORK_CONFIG` | 测试/模型专用 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/rfork.md:124 |
| `RFORK_MOCK_ALLOC_POLICY` | 测试/模型专用 | 文档/示例 | Python/C/C++ API | examples/rfork/rfork_planner.py:69 |
| `RFORK_MOCK_DEFAULT_RESOURCE_POINTS` | 测试/模型专用 | 文档/示例 | Python/C/C++ API | examples/rfork/rfork_planner.py:68 |
| `RFORK_MOCK_HEARTBEAT_SWEEP_SEC` | 测试/模型专用 | 文档/示例 | Python/C/C++ API | examples/rfork/rfork_planner.py:67 |
| `RFORK_MOCK_HEARTBEAT_TTL_SEC` | 测试/模型专用 | 文档/示例 | Python/C/C++ API | examples/rfork/rfork_planner.py:66 |
| `RFORK_MOCK_HOST` | 测试/模型专用 | 文档/示例 | Python/C/C++ API | examples/rfork/rfork_planner.py:64 |
| `RFORK_MOCK_PORT` | 测试/模型专用 | 文档/示例 | Python/C/C++ API | examples/rfork/rfork_planner.py:65 |
| `RFORK_SEED_TIMEOUT_SEC` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/ut/model_loader/rfork/test_rfork_loader.py:48; tests/ut/model_loader/rfork/test_rfork_loader.py:63 |
| `RUNS_ON_RUNNER_NAME` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/push_build_precommit_cache.yaml:51 |
| `RUNS_ON_S3_BUCKET_CACHE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:158; .github/workflows/push_build_precommit_cache.yaml:46 |
| `RUNS_ON_S3_BUCKET_ENDPOINT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:159; .github/workflows/push_build_precommit_cache.yaml:47 |
| `RUNS_ON_S3_FORCE_PATH_STYLE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:160; .github/workflows/push_build_precommit_cache.yaml:48 |
| `RUN_ID` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:80 |
| `RUN_LINK` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:413; .github/workflows/_e2e_nightly_multi_node_560t.yaml:451; .github/workflows/_e2e_nightly_single_node.yaml:333; .github/workflows/_e2e_nightly_single_node_560t.yaml:331 |
| `RUN_URL` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_doc_translate.yaml:170; .github/workflows/schedule_update_estimated_times.yaml:213 |
| `SAVE_PATH` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/quantization.md:49 |
| `SCHEDULE_TAG_PATTERN` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_schedule_image_build.yaml:529 |
| `SELECTED_TARGET_IDS` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/pr_test.yaml:326; .github/workflows/pr_test.yaml:320 |
| `SELECTED_TEST_GROUPS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_selected_tests.yaml:373; .github/workflows/_selected_tests.yaml:398 |
| `SERVER_PORT` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/doc_writing.md:91; docs/source/developer_guide/contribution/doc_writing.md:151 |
| `SERVICE_PROF_CONFIG_PATH` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:162 |
| `SFA_V1_PRECISION_METRICS_LOG` | 仓库工具/文档专用 | 测试 | Python/C/C++ API | tests/ut/attention/a2/test_sfa_v1_precision.py:32 |
| `SHELLCHECK_OPTS` | 仓库工具/文档专用 | CI/发布, 工具/辅助 | Shell export, YAML env | format.sh:39; tools/actionlint.sh:21; .github/workflows/pr_test.yaml:173; .github/workflows/push_build_precommit_cache.yaml:85 |
| `SOC_VERSION` | 仓库工具/文档专用 | CI/发布, 文档/示例, 构建/打包, 运行时代码 | Docker ARG, Docker ENV, Python/C/C++ API, Shell export, YAML env | .github/workflows/dockerfiles/Dockerfile.buildwheel.310p:19; .github/workflows/dockerfiles/Dockerfile.buildwheel.a2:19; .github/workflows/dockerfiles/Dockerfile.buildwheel.a3:19; Dockerfile:59; .github/workflows/dockerfiles/Dockerfile.buildwheel.310p:24; .github/workflows/dockerfiles/Dockerfile.buildwheel.a2:24; .github/workflows/dockerfiles/Dockerfile.buildwheel.a3:24; Dockerfile:62; vllm_ascend/envs.py:53; docs/source/faqs.md:271; docs/source/faqs.md:274; docs/source/faqs.md:277; docs/source/faqs.md:280; .github/workflows/_build_csrc_cache.yaml:169; .github/workflows/_selected_tests.yaml:264; .github/workflows/schedule_release_code_and_wheel.yml:77 |
| `SOURCE_REPOSITORY` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:114; .github/workflows/_ensure_csrc_cache.yaml:100 |
| `SUFFIX` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_schedule_image_build.yaml:530 |
| `SYSTEMROOT` | 仓库工具/文档专用 | 工具/辅助 | Python/C/C++ API | collect_env.py:216 |
| `TAGS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_schedule_image_build.yaml:527 |
| `TAG_SUFFIX` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.nightly.310p:21; .github/workflows/dockerfiles/Dockerfile.nightly.a2:21; .github/workflows/dockerfiles/Dockerfile.nightly.a3:21; .github/workflows/dockerfiles/Dockerfile.nightly.a5:21 |
| `TARGETARCH` | 仓库工具/文档专用 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.lint:19 |
| `TARGETS` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API, YAML env | .github/workflows/_ensure_csrc_cache.yaml:215; .github/workflows/_ensure_csrc_cache.yaml:204 |
| `TARGET_BRANCH` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_cherry_pick_command.yml:78; .github/workflows/pr_cherry_pick_command.yml:133 |
| `TARGET_COMMIT` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:57; .github/workflows/schedule_main2main.yaml:172 |
| `TARGET_IDS` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_ensure_csrc_cache.yaml:152 |
| `TASK_QUEUE_ENABLE` | 仓库工具/文档专用 | 文档/示例 | Python/C/C++ API, Shell export | docs/source/user_guide/feature_guide/graph_mode.md:277; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:125; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:75; docs/source/tutorials/features/suffix_speculative_decoding.md:82; docs/source/tutorials/models/DeepSeekOCR2.md:130 |
| `TERM` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/pr_test.yaml:172; .github/workflows/push_build_precommit_cache.yaml:84 |
| `TESTCASE_TIMEOUT_MINUTES` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_nightly_wait_for_pods_ready.yaml:82 |
| `TEST_CASES` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/resolve_nightly_tests.py:76 |
| `TEST_NAME` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:410; .github/workflows/_e2e_nightly_multi_node_560t.yaml:448; .github/workflows/_e2e_nightly_single_node.yaml:330; .github/workflows/_e2e_nightly_single_node_560t.yaml:328 |
| `TEST_PATH` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/_e2e_nightly_multi_node.yaml:411; .github/workflows/_e2e_nightly_multi_node_560t.yaml:449; .github/workflows/_e2e_nightly_single_node.yaml:331; .github/workflows/_e2e_nightly_single_node_560t.yaml:329 |
| `TIKTOKEN_ENCODINGS_BASE` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/gpt-oss-120b.md:88; docs/source/tutorials/models/gpt-oss-120b.md:110 |
| `TOKENIZERS_PARALLELISM` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/tutorials/models/DeepSeekOCR2.md:128; docs/source/tutorials/models/DeepSeekOCR2.md:131 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | vLLM/PyTorch 上游 | CI/发布, 文档/示例 | Shell export, YAML env | docs/source/installation.md:280; .github/workflows/_selected_tests.yaml:303; .github/workflows/_selected_tests_upstream.yaml:236 |
| `TORCH_EXTENSIONS_ALWAYS_BUILD` | vLLM/PyTorch 上游 | 运行时代码 | Python/C/C++ API | vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:374; vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py:383 |
| `TORCH_NPU_DATE` | vLLM/PyTorch 上游 | 构建/打包 | Docker ARG | Dockerfile:88; Dockerfile.310p:115; Dockerfile.310p.openEuler:111; Dockerfile.a3:91 |
| `TORCH_NPU_VERSION` | vLLM/PyTorch 上游 | CI/发布, 构建/打包, 测试 | Docker ARG, Python/C/C++ API, YAML env | Dockerfile:87; Dockerfile.310p:114; Dockerfile.310p.openEuler:110; Dockerfile.a3:90; tests/e2e/models/test_asr_eval_correctness.py:49; tests/e2e/models/test_lm_eval_correctness.py:34; tests/e2e/models/test_rm_eval_correctness.py:41; .github/workflows/_e2e_nightly_single_node_models.yaml:231 |
| `TORCH_VERSION` | vLLM/PyTorch 上游 | CI/发布, 测试 | Python/C/C++ API, YAML env | tests/e2e/models/test_asr_eval_correctness.py:48; tests/e2e/models/test_lm_eval_correctness.py:33; tests/e2e/models/test_rm_eval_correctness.py:40; .github/workflows/_e2e_nightly_single_node_models.yaml:230 |
| `TP_SOCKET_IFNAME` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/doc_writing.md:223; docs/source/developer_guide/contribution/doc_writing.md:236; docs/source/tutorials/features/ray.md:114; docs/source/tutorials/features/ray.md:126 |
| `TRITON_ALL_BLOCKS_PARALLEL` | vLLM/PyTorch 上游 | 运行时代码 | Python/C/C++ API | vllm_ascend/ops/rotary_embedding.py:480; vllm_ascend/ops/rotary_embedding.py:481 |
| `TRITON_ASCEND_PACKAGE_VERSION` | vLLM/PyTorch 上游 | 构建/打包 | Docker ARG | Dockerfile:90; Dockerfile.a3:93; Dockerfile.a3.openEuler:88; Dockerfile.a5:85 |
| `TRITON_ASCEND_VERSION` | vLLM/PyTorch 上游 | 构建/打包 | Docker ARG | Dockerfile:89; Dockerfile.a3:92; Dockerfile.a3.openEuler:87; Dockerfile.a5:84 |
| `UPSTREAM_FETCH_REF` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:342 |
| `UPSTREAM_REBASE_SHA` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:343 |
| `UPSTREAM_REPO` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_main2main.yaml:44 |
| `USE_MODELSCOPE_HUB` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/evaluation/using_lm_eval.md:116; docs/source/developer_guide/evaluation/using_lm_eval.md:187 |
| `UV_EXTRA_INDEX_URL` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:67; .github/workflows/_e2e_nightly_single_node.yaml:159; .github/workflows/_e2e_nightly_single_node_560t.yaml:158; .github/workflows/_e2e_nightly_single_node_models.yaml:88 |
| `UV_HTTP_TIMEOUT` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:70; .github/workflows/_selected_tests.yaml:103; .github/workflows/_selected_tests_upstream.yaml:85; .github/workflows/schedule_e2e_upstream_test.yaml:41 |
| `UV_INDEX_STRATEGY` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:68; .github/workflows/_e2e_nightly_single_node.yaml:160; .github/workflows/_e2e_nightly_single_node_560t.yaml:159; .github/workflows/_e2e_nightly_single_node_models.yaml:89 |
| `UV_INDEX_URL` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:66; .github/workflows/_e2e_nightly_single_node.yaml:158; .github/workflows/_e2e_nightly_single_node_560t.yaml:157; .github/workflows/_e2e_nightly_single_node_models.yaml:87 |
| `UV_INSECURE_HOST` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:69; .github/workflows/_selected_tests.yaml:102; .github/workflows/_selected_tests_upstream.yaml:84; .github/workflows/schedule_e2e_upstream_test.yaml:40 |
| `UV_NO_CACHE` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:71; .github/workflows/_e2e_nightly_single_node.yaml:161; .github/workflows/_e2e_nightly_single_node_560t.yaml:160; .github/workflows/_e2e_nightly_single_node_models.yaml:90 |
| `UV_SYSTEM_PYTHON` | CI/凭据/包管理 | CI/发布 | YAML env | .github/workflows/_build_csrc_cache.yaml:72; .github/workflows/_e2e_nightly_single_node.yaml:162; .github/workflows/_e2e_nightly_single_node_560t.yaml:161; .github/workflows/_e2e_nightly_single_node_models.yaml:91 |
| `VERBOSE` | 仓库工具/文档专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:55 |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API | examples/rl/rlhf_http_npu_ipc.py:53; tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py:128 |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:77; docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:56; docs/source/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md:103 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | vLLM Ascend 专用 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | docs/source/developer_guide/Design_Documents/balance_schedule_refactor.md:360; vllm_ascend/envs.py:95; docs/source/tutorials/models/DeepSeek-R1.md:143; docs/source/tutorials/models/DeepSeek-V3.1.md:154; docs/source/tutorials/models/GLM4.x.md:133; docs/source/tutorials/models/GLM5.md:554 |
| `VLLM_ASCEND_BRANCH` | vLLM Ascend 专用 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.nightly.310p:17; .github/workflows/dockerfiles/Dockerfile.nightly.a2:17; .github/workflows/dockerfiles/Dockerfile.nightly.a3:17; .github/workflows/dockerfiles/Dockerfile.nightly.a5:17 |
| `VLLM_ASCEND_COMMIT` | vLLM Ascend 专用 | CI/发布, 测试 | Python/C/C++ API, YAML env | tests/e2e/models/test_asr_eval_correctness.py:46; tests/e2e/models/test_lm_eval_correctness.py:31; tests/e2e/models/test_rm_eval_correctness.py:38; .github/workflows/_e2e_nightly_single_node_models.yaml:228 |
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | vLLM Ascend 专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:102 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | vLLM Ascend 专用 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/envs.py:72; docs/source/developer_guide/Design_Documents/context_parallel.md:80; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:78; docs/source/tutorials/features/suffix_speculative_decoding.md:86; docs/source/tutorials/models/DeepSeek-V3.2.md:136 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | vLLM Ascend 专用 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/envs.py:92; docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:79; docs/source/tutorials/models/GLM5.2.md:144; docs/source/tutorials/models/GLM5.2.md:221; docs/source/tutorials/models/GLM5.2.md:275 |
| `VLLM_ASCEND_ENABLE_MLAPO` | vLLM Ascend 专用 | 文档/示例, 运行时代码 | Python/C/C++ API, Shell export | vllm_ascend/envs.py:79; docs/source/tutorials/models/DeepSeek-V3.2.md:134; docs/source/tutorials/models/GLM5.2.md:1075; docs/source/tutorials/models/GLM5.md:555; docs/source/tutorials/models/GLM5.md:604 |
| `VLLM_ASCEND_ENABLE_NZ` | vLLM Ascend 专用 | 工具/辅助, 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | AGENTS.md:56; docs/source/user_guide/feature_guide/sleep_mode.md:98; tests/e2e/pull_request/one_card/test_xlite.py:33; tests/e2e/pull_request/two_card/test_xlite.py:31; docs/source/tutorials/models/DeepSeekOCR2.md:127; docs/source/tutorials/models/GLM5.2.md:1258; docs/source/tutorials/models/GLM5.2.md:1324; docs/source/tutorials/models/GLM5.2.md:1390 |
| `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | vLLM Ascend 专用 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM4.x.md:134 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | vLLM Ascend 专用 | 运行时代码 | Python/C/C++ API | vllm_ascend/envs.py:98 |
| `VLLM_ASCEND_REF` | vLLM Ascend 专用 | 工具/辅助, 测试 | Python/C/C++ API | tests/e2e/nightly/multi_node/external_dp/scripts/utils.py:219; tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py:145; tools/bisect/auto_bisect.py:308 |
| `VLLM_ASCEND_VERSION` | vLLM Ascend 专用 | CI/发布, 测试 | Python/C/C++ API, YAML env | tests/e2e/models/test_asr_eval_correctness.py:45; tests/e2e/models/test_lm_eval_correctness.py:30; tests/e2e/models/test_rm_eval_correctness.py:37; tests/e2e/nightly/single_node/models/scripts/test_single_node.py:441; .github/workflows/_e2e_nightly_single_node.yaml:334; .github/workflows/_e2e_nightly_single_node_560t.yaml:332; .github/workflows/_e2e_nightly_single_node_models.yaml:227 |
| `VLLM_BATCH_INVARIANT` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API, Shell export | docs/source/user_guide/feature_guide/batch_invariance.md:79; docs/source/user_guide/feature_guide/flash_attention.md:102; tests/e2e/pull_request/one_card/test_attention_fa3.py:18; tests/e2e/pull_request/one_card/test_batch_invariant.py:29; docs/source/user_guide/feature_guide/batch_invariance.md:38 |
| `VLLM_CI_RUNNER` | vLLM/PyTorch 上游 | CI/发布, 测试 | Python/C/C++ API, YAML env | tests/e2e/nightly/multi_node/external_dp/scripts/utils.py:210; tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py:132; tests/e2e/nightly/single_node/models/scripts/test_single_node.py:426; .github/workflows/_e2e_nightly_single_node.yaml:290; .github/workflows/_e2e_nightly_single_node.yaml:307; .github/workflows/_e2e_nightly_single_node_560t.yaml:288; .github/workflows/_e2e_nightly_single_node_560t.yaml:305 |
| `VLLM_COMMIT` | vLLM/PyTorch 上游 | CI/发布, 构建/打包, 测试 | Docker ARG, Python/C/C++ API, YAML env | .github/workflows/dockerfiles/Dockerfile.lint:30; Dockerfile:45; Dockerfile.310p:40; Dockerfile.310p.openEuler:39; tests/e2e/models/test_asr_eval_correctness.py:44; tests/e2e/models/test_lm_eval_correctness.py:29; tests/e2e/models/test_rm_eval_correctness.py:36; .github/workflows/_e2e_nightly_single_node_models.yaml:226 |
| `VLLM_DISABLE_COMPILE_CACHE` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/eight_card/test_minimax_m3.py:46; tests/e2e/pull_request/two_card/lora/test_qwen35_densemodel_lora_tp.py:11 |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | vLLM/PyTorch 上游 | 运行时代码 | Python/C/C++ API | vllm_ascend/platform.py:32 |
| `VLLM_DP_MASTER_IP` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API | examples/offline_data_parallel.py:123; tests/e2e/conftest.py:894 |
| `VLLM_DP_MASTER_PORT` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API | examples/offline_data_parallel.py:124; tests/e2e/conftest.py:895 |
| `VLLM_DP_RANK` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API | examples/offline_data_parallel.py:120; tests/e2e/conftest.py:891 |
| `VLLM_DP_RANK_LOCAL` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API | examples/offline_data_parallel.py:121; tests/e2e/conftest.py:892 |
| `VLLM_DP_SIZE` | vLLM/PyTorch 上游 | 文档/示例, 测试 | Python/C/C++ API | examples/offline_data_parallel.py:122; tests/e2e/conftest.py:893 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | vLLM/PyTorch 上游 | CI/发布, 文档/示例, 测试 | Shell export, YAML env | docs/source/tutorials/models/GLM5.2.md:874; docs/source/tutorials/models/GLM5.2.md:924; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:318; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:397; .github/workflows/_e2e_nightly_single_node.yaml:163; .github/workflows/_e2e_nightly_single_node_560t.yaml:162 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:81; docs/source/tutorials/models/GLM5.2.md:865; docs/source/tutorials/models/GLM5.2.md:915 |
| `VLLM_GPU_MEMORY_UTILIZATION` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:109; tests/e2e/pull_request/one_card/test_batch_invariant.py:226 |
| `VLLM_HOST_IP` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:1067 |
| `VLLM_LOGGING_LEVEL` | vLLM/PyTorch 上游 | CI/发布, 文档/示例, 测试 | Shell export, YAML env | examples/external_online_dp/run_dp_template.sh:5; tests/e2e/nightly/multi_node/scripts/run.sh:63; .github/workflows/_selected_tests.yaml:95; .github/workflows/_selected_tests_upstream.yaml:101; .github/workflows/schedule_e2e_upstream_test.yaml:75; .github/workflows/schedule_e2e_upstream_test.yaml:217 |
| `VLLM_MAX_MODEL_LEN` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:110 |
| `VLLM_MAX_PROMPT` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:152; tests/e2e/pull_request/one_card/test_batch_invariant.py:518 |
| `VLLM_MIN_PROMPT` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:151; tests/e2e/pull_request/one_card/test_batch_invariant.py:517 |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/models/GLM5.2.md:1401; docs/source/tutorials/models/GLM5.2.md:1474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:319; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:398 |
| `VLLM_NEEDLE_BATCH_SIZE` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:108; tests/e2e/pull_request/one_card/test_batch_invariant.py:150 |
| `VLLM_NEEDLE_MAX_TOKENS` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:159 |
| `VLLM_NEEDLE_TEMPERATURE` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:157 |
| `VLLM_NEEDLE_TOP_P` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:158 |
| `VLLM_NEEDLE_TRIALS` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:149 |
| `VLLM_PP_LAYER_PARTITION` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/pipeline_parallel.md:192; docs/source/user_guide/feature_guide/pipeline_parallel.md:299 |
| `VLLM_REPO` | vLLM/PyTorch 上游 | 构建/打包 | Docker ARG | .github/workflows/dockerfiles/Dockerfile.lint:26; Dockerfile:43; Dockerfile.310p:38; Dockerfile.310p.openEuler:37 |
| `VLLM_RPC_TIMEOUT` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:80; docs/source/tutorials/models/GLM5.2.md:864; docs/source/tutorials/models/GLM5.2.md:914 |
| `VLLM_TAG` | vLLM/PyTorch 上游 | 构建/打包 | Docker ARG | Dockerfile:44; Dockerfile.310p:39; Dockerfile.310p.openEuler:38; Dockerfile.a3:47 |
| `VLLM_TEST_MODEL` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/rlhf/conftest.py:40 |
| `VLLM_TEST_SEED` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:145; tests/e2e/pull_request/one_card/test_batch_invariant.py:238; tests/e2e/pull_request/one_card/test_batch_invariant.py:493 |
| `VLLM_TEST_TP_SIZE` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:229; tests/e2e/pull_request/one_card/test_batch_invariant.py:240; tests/e2e/pull_request/one_card/test_batch_invariant.py:496 |
| `VLLM_TORCH_PROFILER_WITH_STACK` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/models/Qwen3-235B-A22B.md:410; docs/source/tutorials/models/Qwen3-235B-A22B.md:474; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:330; docs/source/tutorials/models/Qwen3.5-397B-A17B.md:410 |
| `VLLM_TP_SIZE` | vLLM/PyTorch 上游 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_batch_invariant.py:112; tests/e2e/pull_request/one_card/test_batch_invariant.py:429 |
| `VLLM_USE_MODELSCOPE` | vLLM/PyTorch 上游 | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | docs/source/user_guide/feature_guide/sleep_mode.md:96; examples/eplb/eplb_strategy.py:9; examples/offline_data_parallel.py:67; examples/offline_disaggregated_prefill_npu.py:24; docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:55; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:98; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:107; docs/source/developer_guide/performance_and_debug/performance_benchmark.md:149; .github/workflows/_e2e_nightly_single_node.yaml:157; .github/workflows/_e2e_nightly_single_node.yaml:289; .github/workflows/_e2e_nightly_single_node.yaml:306; .github/workflows/_e2e_nightly_single_node_560t.yaml:156 |
| `VLLM_USE_V1` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md:74; docs/source/tutorials/models/DeepSeek-V3.2.md:132; docs/source/tutorials/models/DeepSeekOCR2.md:126; docs/source/tutorials/models/GLM5.2.md:983 |
| `VLLM_USE_V2_MODEL_RUNNER` | vLLM/PyTorch 上游 | 文档/示例 | Shell export | docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md:86 |
| `VLLM_VERSION` | vLLM/PyTorch 上游 | CI/发布, 工具/辅助, 测试, 运行时代码 | Python/C/C++ API, YAML env | tests/e2e/models/test_asr_eval_correctness.py:43; tests/e2e/models/test_lm_eval_correctness.py:28; tests/e2e/models/test_rm_eval_correctness.py:35; tools/bisect/vllm_compat.py:42; .github/workflows/_e2e_nightly_single_node_models.yaml:225 |
| `VLLM_WORKER_MULTIPROC_METHOD` | vLLM/PyTorch 上游 | CI/发布, 文档/示例, 测试 | Python/C/C++ API, Shell export, YAML env | docs/source/user_guide/feature_guide/sleep_mode.md:97; examples/eplb/eplb_strategy.py:10; examples/offline_data_parallel.py:68; examples/offline_disaggregated_prefill_npu.py:25; docs/source/tutorials/models/GLM5.2.md:1267; docs/source/tutorials/models/GLM5.2.md:1333; docs/source/tutorials/models/GLM5.2.md:1399; docs/source/tutorials/models/GLM5.2.md:1472; .github/workflows/_e2e_nightly_single_node.yaml:288; .github/workflows/_e2e_nightly_single_node.yaml:305; .github/workflows/_e2e_nightly_single_node_560t.yaml:286; .github/workflows/_e2e_nightly_single_node_560t.yaml:303 |
| `WEEKLY_MATRIX` | 仓库工具/文档专用 | CI/发布 | Python/C/C++ API | .github/workflows/scripts/resolve_nightly_tests.py:67 |
| `WEIGHT_TRANSFER_TEST_MODEL` | 测试/模型专用 | 测试 | Python/C/C++ API | tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py:53; tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py:92 |
| `WHEEL_FILE` | 仓库工具/文档专用 | CI/发布 | YAML env | .github/workflows/schedule_release_code_and_wheel.yml:155; .github/workflows/schedule_release_code_and_wheel.yml:242; .github/workflows/schedule_release_code_and_wheel.yml:323 |
| `WORKSPACE` | 仓库工具/文档专用 | 文档/示例 | Shell export | docs/source/developer_guide/contribution/multi_node_test.md:399; docs/source/developer_guide/contribution/multi_node_test.md:412; docs/source/developer_guide/contribution/multi_node_test.md:487; docs/source/developer_guide/contribution/multi_node_test.md:500 |
| `WORLD_SIZE` | 系统/分布式通用 | 文档/示例 | Python/C/C++ API | examples/offline_external_launcher.py:153; examples/offline_weight_load.py:157 |
| `YR_CONFIG_PATH` | Ascend/通信/KV 组件 | 文档/示例, 测试, 运行时代码 | Python/C/C++ API, Shell export | tests/ut/distributed/test_yuanrong_backend.py:162; tests/ut/distributed/test_yuanrong_backend.py:180; tests/ut/distributed/test_yuanrong_backend.py:215; vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py:44; docs/source/user_guide/feature_guide/kv_pool.md:947 |

## 解释与使用建议

1. 产品运行时优先查看 `vllm_ascend/envs.py` 及 `vllm_ascend/` 中的直接读取点；直接读取但未集中登记的变量属于历史兼容或外部组件接口。
2. `ASCEND_*`、`HCCL_*`、`MOONCAKE_*` 等多数由 CANN/HCCL/Mooncake 等外部组件解释，仓库通常只读取或透传。
3. CI、发布、凭据和测试变量只在相应自动化流程中生效；其中 `*_TOKEN`、`*_KEY`、`*_SECRET` 可能包含敏感信息，文档不记录实际值。
4. 变量生效时机取决于读取点：安装/编译期、Python import/初始化期、worker 启动期或请求运行期；修改环境后通常需要重新启动进程。
5. 本清单不把上游 vLLM、PyTorch、CANN 的全部可用变量扩展纳入，只记录本仓库明确出现或传递的变量。
