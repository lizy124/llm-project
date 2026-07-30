# AscendStore 重构前梳理

本文档只梳理目录：`/home/lizhongyang/refactor/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`。或者 `D:\lzy\project\kv_pool\code\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store`

目标：在后续重构前明确 AscendStore KV pool connector 的现有模块、职责边界、关键数据流、vLLM 接口适配点和高风险区域。本文不再展开 `kv_transfer` 目录下其它 connector。

## 1. 总览

`kv_pool/ascend_store` 是 vLLM Ascend 侧外部 KV pool/store 集成层，主要负责：

- 在 scheduler 侧查询外部 KV pool 命中，并把命中结果转成 vLLM 可调度的 external token load 需求。
- 在 worker 侧注册 NPU KV cache buffer，并调用后端执行 KV load/save。
- 支持 Mooncake Store、memcache、YuanRong 三类后端。
- 支持非 layerwise 同步 load、非 layerwise 异步 load/save、layerwise key 模式、memcache GVA layerwise 模式。
- 支持 hybrid KV cache groups、Mamba align state、Sliding Window Attention、压缩 KV、Eagle/retention mask、TP mismatch、PP partition adaptor、KV events。
- 通过 vLLM `KVConnectorBase_V1` 和 `SupportsHMA` 接入 scheduler/worker 生命周期。

该目录内部不是单一 store client，而是一个完整 connector 子系统：

- connector 门面：`ascend_store_connector.py`
- scheduler 状态机：`pool_scheduler.py`
- worker 状态机：`pool_worker.py`
- key schema、metadata 和 request tracking：`config_data.py`
- external cache hit correctness 协调：`coordinator.py`
- transfer threads 和 layerwise batch builder：`kv_transfer.py`
- backend 抽象与实现：`backend/`

## 2. 文件树

```text
ascend_store/
  __init__.py
  ascend_store_connector.py
  config_data.py
  coordinator.py
  kv_transfer.py
  pool_scheduler.py
  pool_worker.py
  backend/
    __init__.py
    backend.py
    memcache_backend.py
    mooncake_backend.py
    yuanrong_backend.py
```

## 3. Connector 入口

### `ascend_store_connector.py`

核心类：`AscendStoreConnector(KVConnectorBase_V1, SupportsHMA)`。

职责：作为 vLLM connector 门面，按 role 创建 scheduler 或 worker，并转发 vLLM hook。

初始化关键点：

- `kv_role` 来自 `vllm_config.kv_transfer_config.kv_role`。
- `use_layerwise` 来自 `kv_connector_extra_config["use_layerwise"]`。
- `backend` 默认是 `mooncake`，转小写后保存在 `backend_name`。
- `use_gva_layerwise = use_layerwise and backend == "memcache"`。
- scheduler role 创建 `KVPoolScheduler`。
- worker role 创建 `KVPoolWorker`。
- 非 layerwise 且 global rank 0 时启动 `LookupKeyServer`，给 scheduler 查询 pool hit。
- 注册名 `MooncakeConnectorStoreV1` 仍兼容，但会提示推荐改用 `AscendStoreConnector`。

关键 vLLM hook：

- scheduler side：
  - `get_num_new_matched_tokens()`
  - `update_state_after_alloc()`
  - `build_connector_meta()`
  - `request_finished()`
  - `request_finished_all_groups()`
  - `update_connector_output()`
  - `take_events()`
  - `bind_gpu_block_pool()`
- worker side：
  - `register_kv_caches()`
  - `start_load_kv()`
  - `wait_for_layer_load()`
  - `save_kv_layer()`
  - `wait_for_save()`
  - `get_finished()`
  - `get_block_ids_with_load_errors()`
  - `get_kv_connector_kv_cache_events()`
  - `build_connector_worker_meta()`

特殊适配点：

- `requires_piecewise_for_cudagraph()`：layerwise 模式要求 PIECEWISE CUDA graph。
- `set_xfer_handshake_metadata_pp_aware()`：AscendStore 不依赖 P/D handshake，PP 信息编码在 pool key 中，因此该 hook 是 no-op。
- `update_connector_output()`：除了转发 scheduler 状态，还聚合 worker 上报的 `AscendStoreKVEvents`。
- `get_finished()`：在真实 forward step 后调用 `ensure_store_initialized()`，用于 lazy init backend。

### `LookupKeyServer`

职责：worker rank0 上的 ZMQ REP 服务。

数据流：

```text
scheduler LookupKeyClient.lookup()
  -> ZMQ IPC path
  -> LookupKeyServer process_request()
  -> KVPoolWorker.lookup_scheduler()
  -> backend.exists()
  -> 返回 longest hit tokens
```

IPC path 由 `get_zmq_rpc_path_lookup()` 构造：

```text
ipc://{VLLM_RPC_BASE_PATH}/lookup_rpc_port_{lookup_rpc_port}_dp_rank{dp_rank}
```

兼容旧配置：如果没有 `lookup_rpc_port` 但有 `mooncake_rpc_port`，继续使用旧字段并打 warning。

## 4. Backend 抽象

### `backend/backend.py`

核心抽象：`Backend`。

必需接口：

- `set_device(device_id)`
- `register_buffer(ptrs, lengths)`
- `exists(keys)`
- `put(keys, ptrs, sizes)`
- `get(keys, ptrs, sizes)`

可选接口：

- `batch_get_key_info(keys)`
- `batch_alloc(keys, sizes)`
- `batch_add_lease(keys, ttl_ms)`
- `batch_remove_lease(keys)`
- `batch_write_finish(keys)`

默认 `create_scheduler_client()` 直接创建 backend 实例。scheduler client 通常只用于 `exists()` 或 key info 查询，不注册 NPU KV buffer。

### `backend/__init__.py`

`backend_map`：

| backend 名 | 实现 |
| --- | --- |
| `mooncake` | `MooncakeBackend` |
| `memcache` | `MemcacheBackend` |
| `yuanrong` | `YuanrongBackend` |

### `backend/mooncake_backend.py`

核心类：`MooncakeBackend`。

职责：包装 `mooncake.store.MooncakeDistributedStore`。

配置来源：`MOONCAKE_CONFIG_PATH`。

关键点：

- `MooncakeStoreConfig` 支持 metadata server、global segment size、local buffer size、protocol、device name、master server、SSD offload。
- `ASCEND_ENABLE_USE_FABRIC_MEM=1` 时走 fabric memory 路径。
- 非 fabric memory 时复用 `global_te` 的 Mooncake `TransferEngine`。
- `register_buffer()` 在非 fabric memory 下把 NPU KV cache buffer 注册到 Mooncake memory。
- `put()` 调 `batch_put_from_multi_buffers`。
- `get()` 调 `batch_get_into_multi_buffers`。
- scheduler client 不贡献实际 KV cache memory。
- 部分初始化支持 lazy init，尤其用于压缩场景延后 store 初始化。

### `backend/memcache_backend.py`

核心类：`MemcacheBackend`。

职责：包装 `memcache_hybrid.DistributedObjectStore`。

关键点：

- 支持 lazy init。
- A2 设备存在特殊 buffer register 路径。
- `exists()` 调 `batch_is_exist`。
- `put()` 调 `batch_put_from_layers(... COPY_L2G)`。
- `get()` 调 `batch_get_into_layers(... COPY_G2L)`。
- GVA layerwise 路径依赖 `batch_alloc()`、`batch_get_key_info()`、`batch_add_lease()`、`batch_remove_lease()`、`batch_write_finish()`、`batch_copy()`。
- `batch_alloc()` 与 `batch_copy()` 依赖同进程 `gvaBlobTracker`，因此 GVA allocation 必须在 worker 侧执行。

### `backend/yuanrong_backend.py`

核心类：`YuanrongBackend`。

职责：包装 `yr.datasystem.HeteroClient`。

配置来源：

- `DS_WORKER_ADDR`
- `DS_ENABLE_EXCLUSIVE_CONNECTION`
- `DS_ENABLE_REMOTE_H2D`

关键点：

- `YuanrongHelper.normalize_keys()` 会清理过长或非法 key，并追加 hash suffix。
- `exists()` 调 `exist`。
- `put()` 调 `mset_d2h`。
- `get()` 调 `mget_h2d`。
- 支持大批量 key 分片。
- 错误返回模型与 Mooncake/memcache 不完全一致。

## 5. Key schema 与 metadata

### `config_data.py`

这是 AscendStore 的 key schema、metadata 和 request tracking 核心文件。

核心类型：

- `TPMismatchInfo` / `infer_tp_mismatch_info()`：判断 P/D TP 不一致时能否按 head sub-key 拆分。
- `KeyMetadata`：key 的模型、rank、group、cache role、cache family 元数据。
- `PoolKey`：非 layerwise KV pool chunk key。
- `LayerPoolKey`：layerwise key，在 `PoolKey` 基础上增加 `layer_id`。
- `ChunkedTokenDatabase`：把 token range、block hash、KV group、cache family、block id 转成 key 和 NPU addr/size。
- `LoadSpec`：scheduler 侧记录 vLLM HBM hit 与 KV pool hit 的差异。
- `RequestTracker`：scheduler 侧跨 step 追踪 request token length、allocated blocks、已保存 token、GVA key、Mamba speculative blocks。
- `ReqMeta`：scheduler 发给 worker 的单请求 load/save metadata。
- `AscendConnectorMetadata`：一个 scheduler step 的所有 `ReqMeta`，以及 unfinished、preempted、loading、delayed-free 请求集合。
- layerwise metadata：`LayerBatchReqMeta`、`LayerBlockRange`、`SharedBlockData`、`LayerTransferTask`、`LayerLoadTask`、`LayerMultiBlockReqMeta`。
- `AscendStoreKVConnectorWorkerMetadata`：worker -> scheduler 的 completed event 聚合，用于 Mamba/state block 延迟释放。

### key 字段

非 layerwise `PoolKey.to_string()` 字段：

```text
model
@pcp{pcp_rank}
@dcp{dcp_rank}
@head_or_tp_rank:{head_or_tp_rank}
@pp_rank:{pp_rank}
@group:{kv_cache_group_id}
@cache_role:{cache_role}
@cache_family:{cache_family}
@{chunk_hash}
```

layerwise `LayerPoolKey.to_string()` 字段：

```text
model
@pcp{pcp_rank}
@dcp{dcp_rank}
@head_or_tp_rank:{head_or_tp_rank}
@group:{kv_cache_group_id}
@cache_role:{cache_role}
@cache_family:{cache_family}
@layer_id:{layer_id}
@{chunk_hash}
```

memcache GVA layerwise 另有兼容 key：

```text
single group: model@{block_hash}@{head_or_tp_rank}
multi group:  model@{group_id}@{block_hash}@{head_or_tp_rank}
```

重构注意：key schema 是外部缓存兼容边界。字段、顺序、rank 语义、group 语义、cache family 粒度变化都会导致旧缓存不可读、错误命中或重复写失败。

### `ChunkedTokenDatabase`

职责：统一 key 生成和 addr/size 计算。

关键能力：

- `set_group_buffers()`：写入 group 级 base addr、block len、block stride、cache family、num layers。
- `process_token_key_strings()`：按 token_len/block_hash 生成 key string。
- `process_token_key_strings_with_block_ids()`：同时生成 key string 和本地 block id。
- `prepare_value()`：按 token range 与 block id 计算非 layerwise addr/size。
- `prepare_value_layer()`：按 layer_id 计算 layerwise addr/size。
- `decode_adaptor_prefill_pp()`：consumer 写回 pool 时按 prefill PP partition 拆 key/addr/size。
- `load_mask()` / `store_mask()` / `mask_allows_chunk()`：接入 `AscendStoreCoordinator`。

### block 粒度

AscendStore 同时存在多种粒度：

- `original_block_size`：模型 KV cache spec 的原始 block size。
- `grouped_block_size`：乘以 `pcp_size * dcp_size` 后的 group block size。
- `hash_block_size`：来自 `cache_config.hash_block_size` 或 `prefix_match_unit`，再乘以 CP scale。
- `cache_transfer_granularity`：多个 group/cache family 粒度的 LCM。
- `effective_group_block_size`：group block size 乘以 compress/cache family ratio。

这些粒度共同影响 key 数量、命中 token、保存边界、load/store mask、partial chunk 是否丢弃。

## 6. Scheduler 状态机

### `pool_scheduler.py`

核心类：`KVPoolScheduler`。

职责：scheduler 侧查外部 pool 命中，构造 worker metadata，管理 async load/save 生命周期和 block 延迟释放。

初始化关注点：

- role：`kv_producer` / `kv_consumer` / `kv_both`
- `consumer_is_to_load`
- `consumer_is_to_put`
- `load_async`
- `save_decode_cache`
- `use_layerwise`
- `backend`
- `use_gva_layerwise`
- `use_hybrid`
- `use_compress`
- `mamba_group_ids`
- `num_swa_blocks`
- `discard_partial_chunks`
- `cache_transfer_granularity`
- `tp_mismatch`
- backend scheduler client
- ZMQ `LookupKeyClient`

关键状态：

- `load_specs`：request id -> `LoadSpec`。
- `_request_trackers`：request id -> `RequestTracker`。
- `_unfinished_requests` / `_unfinished_request_ids`：跨 step 未完成请求。
- `_preempted_req_ids`：被抢占请求集合。
- `_loading_req_ids`：async load 中的请求集合。
- `_delayed_free_req_ids`：async save 完成前需要延迟释放 blocks 的请求集合。
- `sending_event_id` / `sending_blocks` / `sending_events`：Mamba/state block 引用保持与 worker 完成事件聚合。
- `_block_pool`：vLLM GPU block pool，用于 touch/free delayed blocks。

### `get_num_new_matched_tokens()`

职责：判断外部 KV pool 是否命中，并返回需要 vLLM 额外分配的 external tokens。

主要路径：

- `kv_consumer` 且未开启 `consumer_is_to_load`：直接返回 `(0, False)`。
- retention interval 开启且 prompt 太短时跳过非 layerwise lookup。
- memcache GVA layerwise：scheduler 侧调用 `batch_get_key_info()` 检查 per-rank GVA key。
- 普通 layerwise：生成包含所有 layer 的 key 后直接查 backend。
- 非 layerwise：通过 `LookupKeyClient` 请求 worker rank0 的 `LookupKeyServer`，由 worker 查 store。
- 命中完整请求时裁掉最后一个 token，避免 decoder 不计算最后 token。
- 生成 `LoadSpec(vllm_cached_tokens, kvpool_cached_tokens, can_load, kvpool_store_skip_tokens)`。
- 返回 `(need_to_allocate, load_async and not use_layerwise)`。

### `update_state_after_alloc()`

职责：vLLM 为 external tokens 分配 block 后，记录本地 block ids 并激活 `LoadSpec.can_load`。

关键点：

- 无 external tokens 时，普通模式不 load；layerwise 可因已有 pool hit 强制 load。
- async load 且非 layerwise 时把 req id 放入 `_loading_req_ids`。
- 校验 `num_external_tokens == kvpool_cached_tokens - vllm_cached_tokens`。

### `build_connector_meta()`

职责：把 scheduler step 的请求转换成 `AscendConnectorMetadata`。

处理顺序：

1. 清理 finished request 的 tracker、unfinished、preempt、loading 状态。
2. 清理 preempted request，并从 delayed-free/loading 中移除。
3. 创建 `AscendConnectorMetadata`，携带 unfinished、preempted、loading、delayed-free 集合。
4. 处理 `scheduled_new_reqs`，创建新 `RequestTracker` 和 `ReqMeta`。
5. 处理 `scheduled_cached_reqs`：
   - preempted cached request 重新建 tracker。
   - running cached request 更新 token 和 block ids。
   - decode 阶段默认不 save，除非 `save_decode_cache=True`。
6. 处理不在当前 scheduled new/cached 中但有 async load 的 request。
7. 对 Mamba/state save 调 `touch_sending_mamba_blocks()` 保持 block 引用。

### finished/free 生命周期

- `request_finished()`：非 HMA 路径判断是否延迟释放 blocks。
- `request_finished_all_groups()`：HMA/hybrid 路径，先按 SWA 裁剪 blocks，再判断 delayed free。
- layerwise 模式不会 delayed free，因为 layerwise 没有 request-level sending event，延迟释放会造成泄漏。
- `update_connector_output()` 聚合 worker 上报的 completed events。所有 worker 完成后释放被 touch 的 Mamba/state blocks。
- `update_finished_sending()` 从 `_delayed_free_req_ids` 移除已完成 send。
- `update_finished_recving()` 从 `_loading_req_ids` 移除已完成 async load。

## 7. Worker 状态机

### `pool_worker.py`

核心类：`KVPoolWorker`。

职责：worker 侧注册 KV cache buffer、初始化 backend、执行 load/save、启动 transfer threads、提供 lookup 服务。

初始化拆分：

- `_init_parallelism_info()`：local rank、TP/PP/PCP/DCP、MLA/sparse、model name。
- `_init_kv_transfer_config()`：role、backend、layerwise、hybrid、Mamba、block 粒度、async load、layerwise limits。
- `_init_key_head_config()`：head_or_tp_rank、put_step、my_key_index、TP mismatch。
- `_init_metadata()`：构造 `KeyMetadata`、`ChunkedTokenDatabase`、`AscendStoreCoordinator`。
- `_init_backend()`：按 backend_map 创建实际 backend。
- `_init_kv_events()`：启用 KV events。
- `_init_state_vars()`：transfer thread、invalid blocks、GVA cache。
- `_init_layerwise_config()`：physical layer 到 KV group/layer index 的映射、prefetch 配置、layer events。

### `register_kv_caches()`

职责：注册 worker KV cache buffer 并启动 transfer threads。

关键点：

- 遍历 `kv_caches`，收集每个 tensor 的 base addr、block len、block stride。
- 合并底层 storage register regions，降低 backend buffer register 数量。
- hybrid 场景按 KV cache group 推导 group 级 addr/len/stride/num_layers/cache_family。
- 单 group 且实际 KV cache 层数包含 MTP/spec decode draft layer 时，会更新 `num_layers`。
- 把 group buffer 信息写入 `ChunkedTokenDatabase`。
- TP mismatch 时计算 strided I/O 所需的 `per_token_bytes`、`sub_size_bytes`。
- 调 backend `register_buffer(ptrs, lengths)`。
- 调 `_start_kv_transfer_threads()`。

### `_start_kv_transfer_threads()`

线程选择矩阵：

| 模式 | 条件 | 发送线程 | 接收线程 |
| --- | --- | --- | --- |
| 非 layerwise save | `kv_producer`/`kv_both` 或 `consumer_is_to_put` | `KVCacheStoreSendingThread` | 无 |
| 非 layerwise async load | `load_async=True` | 可选 sending thread | `KVCacheStoreRecvingThread` |
| layerwise key | `use_layerwise=True` 且非 memcache GVA | `KVCacheStoreKeyLayerSendingThread` | `KVCacheStoreKeyLayerRecvingThread` |
| layerwise memcache GVA | `use_layerwise=True backend=memcache` | `KVCacheStoreLayerSendingThread` | `KVCacheStoreLayerRecvingThread` |

### `start_load_kv()`

职责：消费 `AscendConnectorMetadata`。

非 layerwise：

- 遍历 `ReqMeta`。
- 没有 `load_spec` 或 `can_load=False` 时跳过。
- async load：提交给 `KVCacheStoreRecvingThread.add_request()`。
- sync load：生成 key/addr/size/block_id，按 TP rank 做 circular shift 后调用 backend `get()`。
- backend 返回失败时调用 `record_failed_blocks()`。
- 单 group 会上报 invalid block ids 给 vLLM；hybrid group 失败只记录 error，避免 scheduler 接收到 partial group block 后崩溃。

layerwise：

- 重置 current layer、prefetch 状态和 attention gate。
- 调 `process_layer_data()` 预构建所有 layer 的 load/save task。

### layerwise 处理

核心方法：

- `_process_save_for_layer_batch()`：按 layer/group 建 save block ranges。
- `_alloc_gvas_for_save()`：memcache GVA 模式在 worker 侧为 save blocks 调 `batch_alloc()`。
- `_prepare_load_gvas()`：memcache GVA 模式在 worker 侧调 `batch_get_key_info()` 并 `batch_add_lease()`。
- `_build_shared_save_data()` / `_build_shared_load_data()`：为 GVA path 预构建跨层共享 block id/GVA arrays；为 key path 缓存 token->key 结果。
- `_process_load_for_layer_batch()`：按 layer/group 建 load block ranges。
- `_submit_ready_layer_loads()`：按 `layerwise_prefetch_layers` 提交预取 load。
- `wait_for_layer_load()`：attention 前等待当前层 load 完成。
- `save_kv_layer()`：attention 后提交当前层 save，最后一层等待 save 完成并清理 events。

GVA layerwise 的重要约束：

- scheduler 只生成 block keys，不分配 GVA。
- worker 在 save 前 per-rank 调 `batch_alloc()`。
- worker 维护 `_allocated_gvas`，避免重复分配同一个 key，因为 `batch_alloc()` 对已存在 key 非幂等。
- load 前必须 `batch_get_key_info()` 获取 GVA，并 `batch_add_lease()` 让同进程 tracker 可被 `batch_copy(G2L)` 使用。
- load 最后一层需要释放 read lease；save 最后一层需要 `batch_write_finish()`。

### TP mismatch

相关方法：

- `_make_sub_key_str()`
- `_build_strided_addrs()`
- `_build_tp_mismatch_keys_and_addrs()`
- `_load_kv_tp_mismatch()`
- `_store_kv_tp_mismatch()`

约束：

- 不支持 sparse KV layout。
- 不支持 layerwise。
- 不支持 hybrid KV cache layout。
- 仅在单 dense KV group 下按 effective TP rank 和 head sub-key 拆分。
- 每个 sub-key 使用 strided addr，按 token 粒度传输 head slice。

### lookup

worker 提供两个 lookup 入口：

- `lookup()`：worker 本地查询。
- `lookup_scheduler()`：rank0 lookup server 给 scheduler 用，会 expand 所有 PP/TP rank key。

关键点：

- hybrid 且可用 `AscendStoreCoordinator` 时，优先使用 coordinator 复用 vLLM cache manager 的 longest hit 逻辑。
- 普通连续 KV group 使用 `find_all_continuous_hit_positions()`。
- Mamba align state group 使用 `find_all_discontinuous_hit_positions()`，允许 null block 不连续命中。
- 多 group 最终取 hit positions 交集里的最大值。

### finished/error/event

- `get_finished()`：清理 preempted/stale finished，返回 done_sending/done_recving。
- layerwise 模式会清空 send finished 但不通过 request-level delayed free 返回 done_sending。
- async load 模式只返回 still-loading 集合中的 done_recving。
- `get_block_ids_with_load_errors()`：返回并清空 `_invalid_block_ids`。
- `get_kv_events()`：从 sending thread 收集 `BlockStored`。
- `build_connector_worker_meta()`：Mamba 非 layerwise save 完成后上报 completed event。

## 8. Transfer threads

### `kv_transfer.py`

职责：AscendStore 的实际 transfer threads 和 layerwise batch builder。

核心类：

- `LayerBatchBuilder`
- `KVTransferThread`
- `KVCacheStoreSendingThread`
- `KVCacheStoreRecvingThread`
- `KVCacheStoreKeyLayerSendingThread`
- `KVCacheStoreKeyLayerRecvingThread`
- `KVCacheStoreLayerSendingThread`
- `KVCacheStoreLayerRecvingThread`

### `LayerBatchBuilder`

职责：为 memcache GVA layerwise 模式预计算批量 transfer 数据。

关键能力：

- 按 layer/group 构造 addr array、size array、GVA array。
- 支持跨层共享 block ids/GVAs。
- `_batch_copy_with_limits()` 支持按 `layerwise_max_transfer_blocks` 和 `layerwise_max_transfer_bytes` 分包。

### `KVTransferThread`

所有 transfer thread 的基类。

职责：

- 管理后台 queue。
- 管理 finished request 集合。
- 管理 stored request ref count。
- 管理 KV events。
- 封装 thread ready event 和 request add/clear/discard 行为。

### 非 layerwise transfer

`KVCacheStoreSendingThread`：

- 按 request/group/token chunk 生成 key/addr/size。
- 先查 `exists()`，只写 missing keys。
- 支持 store mask、load hit skip、consumer 写回 prefill PP adaptor。
- 支持 TP mismatch 的 store 路径。
- 支持 KV events。
- 对 Mamba save 完成后记录 completed events。

`KVCacheStoreRecvingThread`：

- 按 `LoadSpec` 生成 key/addr/size。
- 调 backend `get()`。
- 记录失败 block。
- async load 完成后放入 finished requests。

### layerwise key 模式

`KVCacheStoreKeyLayerSendingThread`：

- 每层生成 `LayerPoolKey`。
- 调 backend `put()` 保存当前 layer KV。
- 支持 layer-level finished event。

`KVCacheStoreKeyLayerRecvingThread`：

- 每层生成 `LayerPoolKey`。
- 调 backend `get()` 加载当前 layer KV。
- attention 前通过 `wait_for_layer_load()` 等待当前 layer。

### layerwise memcache GVA 模式

`KVCacheStoreLayerSendingThread`：

- 使用已分配 GVA 和本地 KV cache addr。
- 调 `batch_copy(COPY_L2G)`。
- 最后一层调用 `batch_write_finish()`。
- 使用 layer save finished events 协调 `save_kv_layer()`。

`KVCacheStoreLayerRecvingThread`：

- 使用 `batch_get_key_info()` + `batch_add_lease()` 准备好的 GVA。
- 调 `batch_copy(COPY_G2L)`。
- 支持 H2D stagger 和 batch copy limits。
- load 最后一层释放 read lease。
- 通过 attention gate 降低 load/attention 竞争风险。

### `record_failed_blocks()`

职责：把 backend 返回码映射成失败 block ids。

重构注意：各 backend 的失败语义不统一。当前调用侧需要知道 Mooncake、memcache、YuanRong 的返回形态，抽象层尚未完全统一。

## 9. Cache hit coordinator

### `coordinator.py`

核心类：

- `ExternalCachedBlockPool`
- `AscendStoreCoordinator`

职责：用 vLLM cache manager 的匹配逻辑判断 external KV pool 的 longest cache hit，而不是简单按 key 连续存在判断。

关键能力：

- 把外部 pool key existence 包装成 vLLM block pool duck type。
- 复用 `SingleTypeKVCacheManager.find_longest_cache_hit()`。
- 支持 hybrid KV groups、Mamba、SWA、Eagle、retention interval、压缩 cache family。
- 生成 load/store/lookup mask，控制哪些 chunk 允许读写查询。
- 兼容 vLLM registry、旧 `spec_manager_map`，压缩场景优先 `CompressAttentionManager`。

关键方法：

- `find_longest_cache_hit()`
- `load_mask()`
- `store_mask()`
- `lookup_mask()`
- `_get_manager_class()`
- `_reachable_block_mask()`

重构注意：这里承担 correctness。绕开 coordinator 做简单 key existence，在 hybrid/Mamba/SWA/压缩/retention 场景容易误判。

## 10. 主要数据流

### 10.1 非 layerwise lookup + sync load

```text
scheduler get_num_new_matched_tokens()
  -> LookupKeyClient.lookup()
  -> worker rank0 LookupKeyServer
  -> KVPoolWorker.lookup_scheduler()
  -> backend.exists()
  -> longest hit tokens
scheduler update_state_after_alloc()
  -> LoadSpec.can_load = True
scheduler build_connector_meta()
  -> AscendConnectorMetadata[ReqMeta]
worker start_load_kv()
  -> ChunkedTokenDatabase.process_token_key_strings_with_block_ids()
  -> prepare_value()
  -> backend.get()
  -> record_failed_blocks()
```

### 10.2 非 layerwise async load

```text
scheduler get_num_new_matched_tokens()
  -> returns (need_to_allocate, True)
scheduler update_state_after_alloc()
  -> _loading_req_ids.add(req_id)
scheduler build_connector_meta()
  -> meta.loading_req_ids
worker start_load_kv()
  -> KVCacheStoreRecvingThread.add_request(req_meta)
worker get_finished()
  -> done_recving for meta.loading_req_ids
scheduler update_finished_recving()
  -> _loading_req_ids.remove(done_recving)
```

### 10.3 非 layerwise save + delayed free

```text
scheduler build_connector_meta()
  -> ReqMeta(can_save=True)
worker wait_for_save()
  -> record current NPU event
  -> KVCacheStoreSendingThread.add_stored_request(req_id)
  -> KVCacheStoreSendingThread.add_request(req_meta)
request_finished_all_groups()
  -> if async save pending, _delayed_free_req_ids.add(req_id)
worker get_finished()
  -> done_sending for meta.delayed_free_req_ids
scheduler update_finished_sending()
  -> _delayed_free_req_ids.remove(done_sending)
  -> vLLM can free delayed blocks
```

### 10.4 layerwise key 模式

```text
scheduler build_connector_meta()
  -> ReqMeta load/save specs
worker start_load_kv()
  -> process_layer_data()
  -> build layer_load_tasks/layer_save_tasks
attention before each layer
  -> wait_for_layer_load()
  -> KVCacheStoreKeyLayerRecvingThread backend.get()
attention after each layer
  -> save_kv_layer()
  -> KVCacheStoreKeyLayerSendingThread backend.put()
last layer
  -> wait save finished events
```

### 10.5 layerwise memcache GVA 模式

```text
scheduler get_num_new_matched_tokens()
  -> batch_get_key_info() checks GVA existence
scheduler build_connector_meta()
  -> scheduler only carries keys and block metadata
worker start_load_kv()
  -> process_layer_data()
  -> _alloc_gvas_for_save() with batch_alloc()
  -> _prepare_load_gvas() with batch_get_key_info() + batch_add_lease()
  -> build shared GVA/load/save data
each layer
  -> wait_for_layer_load() submits/awaits batch_copy(COPY_G2L)
  -> save_kv_layer() submits batch_copy(COPY_L2G)
last layer
  -> load releases read lease
  -> save calls batch_write_finish()
```

### 10.6 hybrid/Mamba lookup

```text
scheduler LookupKeyClient.lookup()
  -> worker lookup_scheduler(include_all_ranks=True)
  -> _lookup_with_coordinator()
  -> expand PP/TP key variants
  -> backend.exists()
  -> ExternalCachedBlockPool
  -> AscendStoreCoordinator.find_longest_cache_hit()
  -> final hit length
```

## 11. vLLM 接口适配点

### `KVConnectorBase_V1`

AscendStore 实现的 vLLM hook 覆盖 scheduler 和 worker 生命周期：

- match：`get_num_new_matched_tokens()`
- block allocation：`update_state_after_alloc()`
- metadata：`build_connector_meta()` / `start_load_kv()`
- layerwise：`wait_for_layer_load()` / `save_kv_layer()`
- save：`wait_for_save()`
- completion：`get_finished()` / `update_connector_output()`
- error：`get_block_ids_with_load_errors()`
- events：`get_kv_connector_kv_cache_events()` / `take_events()`

### `SupportsHMA`

AscendStore 支持 HMA/hybrid KV cache manager，关键入口是：

- `request_finished_all_groups()`
- `KVCacheBlocks.get_block_ids()` 返回 grouped block ids
- `ChunkedTokenDatabase` 的 group-aware key/address 逻辑
- `AscendStoreCoordinator` 的 multi-group longest hit

### `KVCacheConfig` / `KVCacheSpec`

影响：

- KV group 数量。
- 每 group block size。
- Mamba group 判断。
- Sliding Window block 裁剪。
- 压缩 ratio/cache family。
- layer_names 到 physical layer 的映射。

### vLLM block pool

`KVPoolScheduler.bind_gpu_block_pool()` 绑定 vLLM GPU block pool：

- async save delayed free。
- Mamba/state save 时 `touch_sending_mamba_blocks()`。
- worker 完成事件聚合后释放 blocks。

### vLLM KV events

- worker sending thread 生成 `BlockStored`。
- connector worker side 包装成 `AscendStoreKVEvents(num_workers=1)`。
- connector scheduler side 聚合多个 worker common events。
- `take_events()` 返回聚合后的 KV cache events。

## 12. 配置项清单

常见 `kv_connector_extra_config`：

| 配置 | 含义 |
| --- | --- |
| `backend` | `mooncake` / `memcache` / `yuanrong`，默认 `mooncake` |
| `use_layerwise` | 是否启用 layerwise load/save |
| `load_async` | 非 layerwise load 是否异步 |
| `consumer_is_to_load` | consumer role 是否从 pool load |
| `consumer_is_to_put` | consumer role 是否也向 pool put |
| `save_decode_cache` | decode 阶段是否继续保存 KV |
| `discard_partial_chunks` | 是否只传完整 transfer granularity chunk |
| `lookup_rpc_port` | scheduler-worker lookup IPC path 后缀 |
| `mooncake_rpc_port` | lookup rpc 旧字段，兼容但推荐迁移 |
| `prefill_tp_size` | consumer 侧推断 peer TP mismatch |
| `decode_tp_size` | producer 侧推断 peer TP mismatch |
| `prefill_pp_size` | consumer 写回 prefill PP partition 数 |
| `prefill_pp_layer_partition` | consumer 写回时 PP layer partition |
| `layerwise_prefetch_layers` | layerwise load 预取层数 |
| `layerwise_max_transfer_blocks` | GVA layerwise batch copy 最大 blocks |
| `layerwise_max_transfer_bytes` | GVA layerwise batch copy 最大 bytes |
| `h2d_stagger_us` | GVA layerwise H2D stagger |

相关环境变量：

| 环境变量 | 作用 |
| --- | --- |
| `MOONCAKE_CONFIG_PATH` | Mooncake Store 配置路径 |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | Mooncake backend 是否使用 fabric memory |
| `DS_WORKER_ADDR` | YuanRong worker 地址 |
| `DS_ENABLE_EXCLUSIVE_CONNECTION` | YuanRong exclusive connection |
| `DS_ENABLE_REMOTE_H2D` | YuanRong remote H2D |
| `VLLM_RPC_BASE_PATH` | ZMQ IPC base path |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | prefix cache retention interval |

## 13. 重构高风险点

1. AscendStore key schema 是缓存兼容边界。`PoolKey`、`LayerPoolKey`、GVA key 三套格式都要保持兼容，不能隐式改字段、顺序或 rank 语义。

2. `cache_transfer_granularity` 由 group block size、CP scale、hash block size、cache family ratio 共同决定。命中、保存、load、mask、partial chunk 都依赖它，容易出现 off-by-one。

3. `KVPoolScheduler.get_num_new_matched_tokens()` 同时处理普通 lookup、layerwise lookup、GVA lookup、retention、full prompt hit 裁剪和 async load 返回语义。拆分时必须保持 vLLM scheduler contract。

4. `KVPoolScheduler.build_connector_meta()` 是 request 生命周期中心。new/cached/preempted/async-load request 处理顺序不能随意调整。

5. `request_finished_all_groups()` 与 `_delayed_free_req_ids` 是 async save 资源生命周期关键点。漏报 done_sending 会 block 泄漏，过早 done 会释放仍在写 store 的 blocks。

6. layerwise 模式不能走 request-level delayed free。当前代码显式 free now，否则会因为没有 request-level sending event 而泄漏。

7. memcache GVA layerwise 的 GVA allocation 必须在 worker 侧。移到 scheduler 会破坏 memcache 同进程 `gvaBlobTracker` 假设。

8. `_allocated_gvas` 是防重复 `batch_alloc()` 的关键状态。重复 alloc 已存在 key 可能返回 duplicated object 且不注册 blob，后续 batch_copy 会失败。

9. GVA load 前必须 `batch_get_key_info()` + `batch_add_lease()`。少掉 lease 会导致同进程 tracker 不可见或对象不可读。

10. hybrid group load 失败不能上报 partial group block ids。当前逻辑只在单 group 上报 invalid blocks，hybrid 只打 error，避免 scheduler 因 group 不一致崩溃。

11. `AscendStoreCoordinator` 承担 hybrid/Mamba/SWA/压缩场景 correctness。绕开它做连续 `exists()` 判断会误判 longest hit。

12. Mamba align state 允许不连续命中，和普通 attention 连续命中逻辑不同。`find_all_discontinuous_hit_positions()` 与 null block skip 不能误删。

13. TP mismatch 只支持单 dense KV group，且不支持 sparse、layerwise、hybrid。重构时要保留这些显式限制。

14. TP mismatch 的 strided I/O 按 token 和 head slice 构造 addr/size，依赖 KV cache layout `[num_block, block_size, num_kv_head_per_local_rank, head_dim]`。KV layout 改动会直接破坏读写。

15. PP partition adaptor 只在 consumer 写回 pool 且 `consumer_is_to_put` 时生效，依赖 `prefill_pp_layer_partition` 与模型层数一致。

16. `LookupKeyClient`/`LookupKeyServer` 的 IPC path 使用 dp rank 和 rpc port，端口字段兼容逻辑不能破坏旧配置。

17. backend 错误模型不统一。Mooncake、memcache、YuanRong 的返回码/failed keys 语义不同，当前调用侧仍有后端知识。

18. backend lazy init 与 `ensure_store_initialized()` 绑定真实 forward step。提前初始化可能影响启动时序，过晚初始化可能让第一次 load/save 失败。

19. buffer registration 依赖底层 storage region 合并。KV cache tensor view/storage 变化会影响注册 ptr/length，进而影响 backend 访问合法性。

20. layerwise prefetch、attention gate、NPU event 同步都是并发正确性边界。优化等待点可能带来 load/attention/save 数据竞争。

21. KV events 聚合只保留多 worker common events。调整 events 聚合语义会影响上层 KV event consumer。

22. vLLM 版本兼容点包括 `hash_block_size` vs `prefix_match_unit`、KVCacheSpec registry、CompressAttentionManager 查找等，升级 vLLM 时应专项测试。

## 14. 建议的重构切入顺序

1. 先冻结外部接口和兼容边界：注册名、`kv_connector_extra_config`、环境变量、backend 名称、key schema、metadata 字段、vLLM hook 返回语义。

2. 先拆纯逻辑，后动线程：优先抽出 key schema、block 粒度计算、cache family 推断、lookup key expansion、TP mismatch key/addr 构造。

3. 收敛 metadata 模型：把 `LoadSpec`、`RequestTracker`、`ReqMeta` 中 save/load/GVA/layerwise 字段分组，减少可选字段交叉污染。

4. 明确 scheduler 状态机：把 new request、cached request、preempted request、async load request 的状态转换拆成可测试小函数。

5. 抽象 transfer lifecycle：统一 request ref count、done_sending、done_recving、delayed free、failed blocks、completed worker events 的生命周期模型。

6. 保留并加强 `AscendStoreCoordinator`：所有 hybrid/Mamba/SWA/压缩 lookup/load/store mask 继续走 coordinator，先补测试再调整实现。

7. 分离 backend adapter 与错误模型：为 `exists/get/put/batch_*` 定义统一结果对象，避免 thread 层直接解释不同 backend 返回码。

8. 独立整理 memcache GVA layerwise：把 GVA allocation、lease、batch_copy、write_finish 的状态机单独封装，明确同进程约束。

9. 最后再动线程模型：在纯逻辑、metadata、backend adapter 稳定后，再考虑合并 key layerwise/GVA layerwise thread 的公共部分。

## 15. 建议测试矩阵

最小专项测试应覆盖：

| 维度 | 场景 |
| --- | --- |
| backend | mooncake / memcache / yuanrong |
| role | producer / consumer / both |
| load | sync load / async load / no load |
| save | prompt save / decode save / consumer_is_to_put |
| layerwise | off / key layerwise / memcache GVA layerwise |
| KV layout | single group / hybrid groups / Mamba align state / SWA / compressed KV |
| block 粒度 | hash_block_size != block_size / discard_partial_chunks on/off / full prompt hit |
| parallel | TP / PP / PCP / DCP / TP mismatch |
| failure | missing key / partial backend failure / invalid GVA / lease failure |
| lifecycle | preempted req / finished req / delayed free / async load finished / worker completed events |
| events | KV events enabled / disabled / multi-worker aggregation |

## 16. 后续文档建议

后续重构可以继续补充：

- `ascend_store_config_matrix.md`：配置项、默认值、兼容字段、支持 backend/mode。
- `ascend_store_key_schema.md`：PoolKey、LayerPoolKey、GVA key 的版本化与兼容策略。
- `ascend_store_lifecycle.md`：request 从 scheduler lookup 到 worker load/save、done/free 的状态机。
- `ascend_store_test_matrix.md`：backend、parallel、hybrid、layerwise、failure case 的测试矩阵。
