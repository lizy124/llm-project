# kv_pool 优化点 → 社区 Issue 草稿索引

> 来源：[kv_pool_优化点系统性梳理.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_优化点系统性梳理.md)
> 参考格式：vllm-ascend issue #13745 / #13746 / #13747（测试任务型 issue）
> 目标仓库：`vllm-project/vllm-ascend`，代码路径 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 验收人：@赵鹏博
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 发布日期：2026-08-11（2026-08-12 新增 kv-35）
> 说明：本文档列出全部 35 个候选 issue，后续从中挑选 10 个正式提交。

## 通用约定（所有 issue 共享）

- **代码基线**：vllm-ascend 最新 `main`
- **硬件**：Ascend NPU（提交时注明型号 + 卡数 + TP/CP 配置）
- **验收人**：@赵鹏博
- **关联任务池**：#9079
- **交付件通用要求**：
  - PR + 设计说明（改动动机 / 方案 / 风险）
  - 补充或更新 `tests/ut/distributed/ascend_store/` 下对应单测
  - 性能类 issue 需附改动前后对比数据（吞吐 / 延迟 / 调用次数）
  - 发现新问题需提 follow-up issue 并回链本 issue
- **回归红线**：现有单测全绿；精度与功能不退化（greedy / non-greedy 输出一致）

## 候选清单

### 性能维度（Perf，19 个）

| 编号 | 文件 | 标题 | 严重程度 | 建议优先级 |
|------|------|------|----------|-----------|
| kv-01 | [issue_kv-01_P01_MLA_read_dedup.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-01_P01_MLA_read_dedup.md) | [Perf] MLA 读侧去重：rank0 取 + TP broadcast | 高 | P1（已有方案） |
| kv-02 | [issue_kv-02_P02_key_string_vectorize.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-02_P02_key_string_vectorize.md) | [Perf] Key 字符串生成向量化/下沉 | 高 | P1 |
| kv-03 | [issue_kv-03_P03_nested_loop_keys.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-03_P03_nested_loop_keys.md) | [Perf] `_generate_store_query_keys` 5 层嵌套循环优化 | 中 | P2 |
| kv-04 | [issue_kv-04_P04_tp_mismatch_dup_lookup.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-04_P04_tp_mismatch_dup_lookup.md) | [Perf] TP mismatch 路径重复 lookup 消除 | 中 | P3 |
| kv-05 | [issue_kv-05_P05_gva_alloc_loop.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-05_P05_gva_alloc_loop.md) | [Perf] GVA 分配循环对象构造批量化 | 低 | P3 |
| kv-06 | [issue_kv-06_P06_non_gva_prefetch.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-06_P06_non_gva_prefetch.md) | [Perf] 非-GVA layerwise prefetch 默认值与 submit 逻辑 | 高 | P1/P2 |
| kv-07 | [issue_kv-07_P07_non_layerwise_io_merge.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-07_P07_non_layerwise_io_merge.md) | [Perf] 非-layerwise I/O 跨 request 合并 | 高 | P1 |
| kv-08 | [issue_kv-08_P08_gva_meta_rpc_merge.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-08_P08_gva_meta_rpc_merge.md) | [Perf] GVA 元数据 RPC 跨 (request,group) 合并 | 高 | P1 |
| kv-09 | [issue_kv-09_P09_tolist_eliminate.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-09_P09_tolist_eliminate.md) | [Perf] 消除 `batch_copy` 前的 `.tolist()` | 高 | P1 |
| kv-10 | [issue_kv-10_P10_block_hash_to_str_dup.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-10_P10_block_hash_to_str_dup.md) | [Perf] `block_hash_to_str` 重复转换 3 次 | 中 | P2 |
| kv-11 | [issue_kv-11_P11_from_request_tracker_rebuild.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-11_P11_from_request_tracker_rebuild.md) | [Perf] `from_request_tracker` 增量 buffer 替代全量重建 | 中高 | P1/P2 |
| kv-12 | [issue_kv-12_P12_handle_stored_request_double_rebuild.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-12_P12_handle_stored_request_double_rebuild.md) | [Perf] `_handle_stored_request` 双重建+三遍遍历 | 中 | P2 |
| kv-13 | [issue_kv-13_P13_non_layerwise_wait_for_save.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-13_P13_non_layerwise_wait_for_save.md) | [Perf] 非-layerwise `wait_for_save` 异步化 | 中 | P2 |
| kv-14 | [issue_kv-14_P14_last_layer_save_sync.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-14_P14_last_layer_save_sync.md) | [Perf] layerwise 最后一层 save 同步等待推迟 | 中 | P2 |
| kv-15 | [issue_kv-15_P15_zmq_lookup_no_timeout.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-15_P15_zmq_lookup_no_timeout.md) | [Perf] ZMQ Lookup RPC 超时与批合并 | 中 | P2 |
| kv-16 | [issue_kv-16_P16_gate_over_sync.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-16_P16_gate_over_sync.md) | [Perf] gate 对非复用预取层过度同步 | 中 | P2 |
| kv-17 | [issue_kv-17_P17_zmq_lookup_full_hashes.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-17_P17_zmq_lookup_full_hashes.md) | [Perf] ZMQ lookup 只发后缀 hashes | 中高 | P1 |
| kv-18 | [issue_kv-18_P18_lookup_key_expand_replace.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-18_P18_lookup_key_expand_replace.md) | [Perf] lookup key 展开避免字符串 replace | 中 | P1 |
| kv-19 | [issue_kv-19_P19_mooncake_get_pythonize.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-19_P19_mooncake_get_pythonize.md) | [Perf] Mooncake get 返回值降 Python 化 | 中高 | P1 |

### 结构维度（Refactor，5 个）

| 编号 | 文件 | 标题 | 严重程度 | 建议优先级 |
|------|------|------|----------|-----------|
| kv-20 | [issue_kv-20_S1_giant_file_split.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-20_S1_giant_file_split.md) | [Refactor] 巨文件拆分（pool_worker/kv_transfer/config_data/scheduler） | 高 | P1 |
| kv-21 | [issue_kv-21_S2_scheduler_worker_init_dup.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-21_S2_scheduler_worker_init_dup.md) | [Refactor] scheduler/worker 初始化逻辑去重 | 高 | P1 |
| kv-22 | [issue_kv-22_S3_to_string_dup.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-22_S3_to_string_dup.md) | [Refactor] `to_string` 重复实现统一 | 低 | P3 |
| kv-23 | [issue_kv-23_S4_chunked_token_db_split.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-23_S4_chunked_token_db_split.md) | [Refactor] `ChunkedTokenDatabase` 职责拆分 | 中 | P3 |
| kv-24 | [issue_kv-24_S5_config_schema.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-24_S5_config_schema.md) | [Refactor] 配置项集中 schema（KVPoolConfig） | 高 | P0 |

### 正确性维度（Correctness，6 个）

| 编号 | 文件 | 标题 | 严重程度 | 建议优先级 |
|------|------|------|----------|-----------|
| kv-25 | [issue_kv-25_C1_transfer_thread_exception.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-25_C1_transfer_thread_exception.md) | [Correctness] 传输线程异常路径资源清理 | 高 | P0 |
| kv-26 | [issue_kv-26_C1-1_backend_put_failure.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-26_C1-1_backend_put_failure.md) | [Correctness] backend put 失败向上层传播 | 高 | P0 |
| kv-27 | [issue_kv-27_C1-2_zmq_lookup_failover.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-27_C1-2_zmq_lookup_failover.md) | [Correctness] ZMQ lookup server/client 失效保护 | 高 | P0 |
| kv-28 | [issue_kv-28_C2_multi_group_failure.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-28_C2_multi_group_failure.md) | [Correctness] 多 group 加载失败错误传播策略 | 中 | P2 |
| kv-29 | [issue_kv-29_C3_invalid_block_ids_lock.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-29_C3_invalid_block_ids_lock.md) | [Correctness] `_invalid_block_ids` 锁保护范围审计 | 中 | P2 |
| kv-30 | [issue_kv-30_C4_iter_token_chunks_boundary.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-30_C4_iter_token_chunks_boundary.md) | [Correctness] `_iter_token_chunks` 边界条件测试 | 低 | P3 |

### 扩展性维度（Ext，4 个）

| 编号 | 文件 | 标题 | 严重程度 | 建议优先级 |
|------|------|------|----------|-----------|
| kv-31 | [issue_kv-31_E1_backend_abstraction_split.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-31_E1_backend_abstraction_split.md) | [Ext] Backend 抽象基类拆分（Backend / GVABackend） | 中 | P2 |
| kv-32 | [issue_kv-32_E2_backend_map_hardcode.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-32_E2_backend_map_hardcode.md) | [Ext] backend_map 支持外部注册 | 低 | P3 |
| kv-33 | [issue_kv-33_E3_high_risk_test_coverage.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-33_E3_high_risk_test_coverage.md) | [Ext] 高风险路径测试覆盖补强 | 中高 | P1 |
| kv-34 | [issue_kv-34_E4_connector_no_base.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-34_E4_connector_no_base.md) | [Ext] Connector 公共基类评估 | 低 | P3 |

### 架构维度（Arch，1 个）★ 重点

| 编号 | 文件 | 标题 | 严重程度 | 建议优先级 |
|------|------|------|----------|-----------|
| kv-35 | [issue_kv-35_P0_transfer_ipc_gil_release.md](file:///D:/lzy/project/kv_pool/llm-project/draft/issues_10/issue_kv-35_P0_transfer_ipc_gil_release.md) | [Arch/Perf] 传输路径改 IPC：消除 GIL 瓶颈，实现真并行传输 | 高 | **P0（重点）** |

> kv-35 是本次新增的重点 issue，方向锁定为改 IPC（不接受"不划算"作为放弃理由，代价转为设计中需攻克的难点）。依据 [kv_pool_线程存取与GIL分析.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_线程存取与GIL分析.md) 第三、四章。分三阶段推进：设计验证 → 原型对比 → 全路径落地。

## 依赖关系（挑选时注意）

- kv-24（S5 配置集中）是 kv-21（S2 初始化去重）的前置
- kv-33（E3 补测试）是 kv-20（S1 拆分）、kv-07/kv-08（I/O 合并重构）的前置
- kv-02（key 向量化）应与 kv-20（config_data 拆分）中的 keys.py 协同
- kv-09（消除 .tolist()）依赖 C++ 扩展支持 buffer protocol，需跨团队协调
- kv-06（prefetch 默认值）独立，可立即见效
- kv-07 与 kv-08 模式相同，可同步推进
- **kv-35（改 IPC）依赖 kv-24（配置 schema）**：IPC 相关配置需纳入 `KVPoolConfig`
- **kv-35 协同 kv-25/kv-26/kv-27**：子进程异常清理与失败传播复用其机制
- kv-35 与 kv-02/kv-09 不冲突：IPC 化后 Python 层优化在子进程内依然有益

## 建议挑选思路

若希望覆盖"高收益 + 风险可控"，可优先从以下挑选 10 个：
**kv-35（改 IPC，重点）**、kv-25 / kv-26 / kv-27（正确性 P0 三连）、kv-24（配置 schema）、kv-07 / kv-08（I/O 合并双连）、kv-06（prefetch 重磅）、kv-17（ZMQ payload）、kv-33（测试补强）。

> kv-35 因改造范围大、分三阶段，建议单独排期，阶段 1 设计文档可与上述其他 issue 并行推进。
