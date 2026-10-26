# 09. 指标、测试地图与排障

基线见 [00](../00_source_baseline.md)。本篇把源码结论连接到可观察结果与现有测试；列出的测试是覆盖入口，不表示本次已运行。

## 1. 当前指标能回答什么

实现为 `metrics.py::AscendStoreKVConnectorStats` / `AscendStorePromMetrics`。

| Prometheus 名称 | 类型 | 当前含义 |
|---|---|---|
| `vllm:ascend_store_load_get_duration_seconds` | histogram | per-worker 非 layerwise `Backend.get` 调用耗时，秒 |
| `vllm:ascend_store_load_get_keys_total` | counter | 上述已完成 get 调用传入的 key 数 |
| `vllm:ascend_store_delayed_release_requests` | gauge | 保留的统计接口；当前主链没有实际设置点 |
| `vllm:ascend_store_delayed_release_blocks` | gauge | 同上，不能据名称认定当前在走 delayed-free |

同步 bulk get 在 Worker 记录，异步 bulk recv 通过 callback 记录。`get_stats()` 加锁取走当前统计并换新对象，后续经 connector、runner output 和聚合进入 metrics。

当前 `record_operation()` 只接收 `load_get`。因此：
- 不是端到端 TTFT，也不含 Scheduler lookup、排队和模型 forward 的全部时间。
- 不能代表 layerwise GVA/range load。
- keys total 不是成功恢复的 token 数，也不是传输字节。
- 多 Worker 的调用样本可能都被聚合，不能直接当“请求次数”。
- delayed release gauges 仍有定义和测试，但本快照运行代码无 `set_delayed_release()` 调用；不能用其零值证明所有保存已完成。

## 2. 分清三条上行证据

| 通道 | 代表内容 | 用途 |
|---|---|---|
| transfer results | receiving / failed receiving / sending | 推进或失败处理 |
| worker metadata | Mamba completed event counts | 所有 rank 完成后释放特殊引用 |
| KV events / stats | BlockStored 等、get 统计 | 缓存事件与性能观测 |

`BlockStored` 不是 `finished_recving`。KV events 的多 Worker 聚合保留共同事件，也不能拿单 Worker event 推断所有 rank 数据齐全。

## 3. 主线测试地图

以下路径均相对 `vllm-ascend/`。

| 行为主题 | 测试入口 |
|---|---|
| connector role、bind/start、Mamba/失败结果 | [test_ascend_store_connector.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_ascend_store_connector.py) |
| hit/token/allocation/decode metadata | [test_pool_scheduler.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_pool_scheduler.py) |
| 地址布局、task、load/save 分支 | [test_pool_worker.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_pool_worker.py) |
| key、hash/group、snapshot | [test_metadata.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_metadata.py) |
| 线程、copy/put、失败、队列 | [test_kv_transfer.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_kv_transfer.py) |
| SDK wrapper 与结果契约 | [test_backend.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_backend.py) |
| hybrid 可达性/交集 | [test_coordinator.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_coordinator.py) |
| layerwise keys 与物理 reuse | [test_layerwise_keys.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_layerwise_keys.py)、[test_layerwise_cache_layout.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py) |
| Mooncake session/range / hybrid | [test_mooncake_layerwise.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_mooncake_layerwise.py)、[test_mooncake_hybrid.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_mooncake_hybrid.py) |
| PP partition / namespace / 空末层 | [test_mooncake_pipeline.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_mooncake_pipeline.py) |
| reuse 发布与尾部快照 | [test_reuse_publication_fence.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_reuse_publication_fence.py)、[test_reuse_snapshot_blocks.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_reuse_snapshot_blocks.py)、[test_reuse_snapshot_identity.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_reuse_snapshot_identity.py) |
| 请求全链、private state | [test_request_lifecycle.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_request_lifecycle.py) |
| DSpark companion / lease / collision | [test_dspark_prefix_cache.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_dspark_prefix_cache.py) |
| 统计聚合/Prometheus | [test_metrics.py](../../../../../../vllm-ascend/tests/ut/distributed/ascend_store/test_metrics.py) |
| 传输失败跨组件路径 | [test_kv_transfer_failures.py](../../../../../../vllm-ascend/tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py) |
| layerwise 生命周期 | [test_ascend_store_layerwise_lifecycle.py](../../../../../../vllm-ascend/tests/ut/kv_transfer/test_ascend_store_layerwise_lifecycle.py) |

特别值得跟读的行为：
- 普通保存：`test_stale_store_job_cannot_mutate_reused_request_id`，旧 job 不能污染复用 request id 的新状态。
- PP：`test_scheduler_block_size_rewrite_preserves_remote_hits`，防止 Scheduler/Worker key namespace 分裂。
- PP：`test_empty_last_layer_waits_for_earlier_put`，空末层不能提前放行。
- DSpark：`test_prefix_restore_refreshes_addresses_under_lease_before_copy`，lease 下重新取地址。
- DSpark：`test_incomplete_other_writer_object_is_not_overwritten_or_published`，不篡改其他 writer。
- reuse：`test_fence_order_and_scope`，保护正确的等待范围与顺序。

## 4. 本地 offload 与组合测试

| 主题 | 测试入口 |
|---|---|
| native layout / NPU copy | [test_native_cpu_offload.py](../../../../../../vllm-ascend/tests/ut/kv_offload/test_native_cpu_offload.py)、[test_npu_offload_spec.py](../../../../../../vllm-ascend/tests/ut/kv_offload/test_npu_offload_spec.py) |
| simple offload | [test_simple_kv_offload.py](../../../../../../vllm-ascend/tests/ut/simple_kv_offload/test_simple_kv_offload.py) |
| 抢占 CPU KV / partial / Mamba | [test_preempt_offload_connector.py](../../../../../../vllm-ascend/tests/ut/kv_offload/test_preempt_offload_connector.py) |
| 抢占 Scheduler | [test_recompute_scheduler.py](../../../../../../vllm-ascend/tests/ut/core/test_recompute_scheduler.py) |
| MultiConnector slot 与 hook | [test_ascend_multi_connector.py](../../../../../../vllm-ascend/tests/ut/kv_offload/test_ascend_multi_connector.py) |
| Mamba load/copy 时序 | [test_mamba_kv_connector.py](../../../../../../vllm-ascend/tests/ut/worker/test_mamba_kv_connector.py) |
| KVPP + pool / DSpark e2e | [test_kvpp_pool.py](../../../../../../vllm-ascend/tests/e2e/cases/features/kv_pool/test_kvpp_pool.py)、[test_kvpp_dspark.py](../../../../../../vllm-ascend/tests/e2e/cases/features/kv_pool/test_kvpp_dspark.py) |

UT 的 mock 可以验证参数与生命周期，不能单独证明真实 SDK、NPU stream、通信和输出精度正确。e2e 入口也不意味着覆盖所有 topology 与失败组合。

## 5. 症状 → 最短定位路径

| 症状 | 优先检查 |
|---|---|
| 始终零命中 | consumer load 门控→granularity→key sample/namespace→readability→日志是否查询异常 |
| PP=1 正常，PP>1 miss | 真 partition、cache block 归一化、all-stage keys、空 projected group |
| hybrid 只有部分模型命中 | cache family、reachable mask、private/cacheable groups、hash scale |
| 命中后输出异常 | group/block id/stride→Mamba copy→发布完整性→失效结果是否处理 |
| 请求结束后偶发脏写 | released ids 是否齐全；handle_preemptions 是否在 zero/forward 前；所有旧 chunks 是否 drain |
| layerwise 空末层卡住/提前完成 | final group vs final model layer、空 task event、earlier queue drain |
| reuse 下旧 token 丢失 | old partial load 是否先于新 alloc；本地 hash 是否误当共享层驻留 |
| session/lease 越积越多 | retry/terminal owner、get_end、异常 revoke、shutdown 顺序 |
| DSpark target 命中但 draft 错 | joint provider、companion namespace/page size、lease 下 GVA、partial tail |
| 指标零或不随 layerwise 变化 | 指标是否覆盖该数据面；stats 是否有调用点 |
| 性能比 bulk 更差 | 每层过度 drain、prefetch 窗口、RPC 数、reuse 实际流量、SSD rewarm |

## 6. 如何验证一次后续代码改动

1. 根据修改层选择上述最小行为测试，优先复用现有测试。
2. 对 key/layout 改动检查 Scheduler 与 Worker 生成身份是否一致。
3. 对生命周期改动验证正常、失败、空任务、抢占/结束、teardown。
4. 对真实异步/SDK 改动补相应环境下的运行证据，记录 SDK/两仓 commit、模式和 topology。
5. 对输出或性能影响分别记录精度/结果对照与 TTFT/TPOT/吞吐；get histogram 只能解释其中一段。

本次仅更新 Markdown，完成的是源码、符号和链接核对；未启动后端服务、运行 NPU UT/e2e 或进行性能/精度测试。
