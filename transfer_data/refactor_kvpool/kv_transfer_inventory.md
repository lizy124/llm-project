# kv_transfer 重构前梳理

本文档梳理目录：`D:\lzy\code\pd_pool_mtp\vllm-ascend\vllm_ascend\distributed\kv_transfer`。

目标：在后续重构前明确该目录现有模块、职责边界、关键数据流、vLLM 接口适配点和高风险区域。

## 1. 总览

`kv_transfer` 是 vLLM Ascend 侧 KV cache 传输与外部 KV pool 集成层，主要负责：

- disaggregated prefill/decode 场景下跨 worker 或跨 engine 传输 KV cache。
- 对接 Mooncake P2P transfer engine。
- 对接外部 KV pool/store 后端，包括 Mooncake Store、memcache、YuanRong。
- 支持 layerwise 推送/加载、异步 load/save、hybrid KV cache groups、Mamba、Sliding Window Attention、压缩 KV、TP/PP/CP 不一致等场景。
- 支持 CPU offload，包括简单 offload 和 preempt 后 recompute/offload 恢复。
- 向 vLLM `KVConnectorFactory` 注册 Ascend 自定义 connector。

该目录不是单一实现，而是多套 connector 和后端的集合：

- P2P Mooncake：`kv_p2p/`
- KV pool / AscendStore：`kv_pool/ascend_store/`
- CPU offload：`kv_pool/simple_cpu_offload/`、`kv_pool/recompute_cpu_offload/`
- 第三方或上游适配：`kv_pool/ucm_connector.py`、`kv_pool/lmcache_ascend_connector.py`
- 公共工具：`utils/`
- 多 connector 聚合：`ascend_multi_connector.py`

## 2. 文件树

```text
kv_transfer/
  __init__.py
  ascend_multi_connector.py
  kv_p2p/
    __init__.py
    mooncake_connector.py
    mooncake_hybrid_connector.py
    mooncake_layerwise_connector.py
  kv_pool/
    __init__.py
    lmcache_ascend_connector.py
    ucm_connector.py
    simple_cpu_offload/
      __init__.py
      simple_cpu_offload_connector.py
    recompute_cpu_offload/
      __init__.py
      metadata.py
      manager.py
      worker.py
      recompute_cpu_offload_connector.py
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
  utils/
    __init__.py
    mooncake_transfer_engine.py
    utils.py
```

## 3. 注册入口

### `__init__.py`

核心函数：`register_connector()`。

职责：向 vLLM `KVConnectorFactory` 注册 Ascend connector，并覆盖部分上游 connector 名称。

注册关系：

| 注册名 | 实现 |
| --- | --- |
| `MultiConnector` | `AscendMultiConnector` |
| `MooncakeConnectorV1` | `kv_p2p.mooncake_connector.MooncakeConnector` |
| `MooncakeHybridConnector` | `kv_p2p.mooncake_hybrid_connector.MooncakeConnector` |
| `MooncakeConnectorStoreV1` | `kv_pool.ascend_store.ascend_store_connector.AscendStoreConnector` |
| `AscendStoreConnector` | `kv_pool.ascend_store.ascend_store_connector.AscendStoreConnector` |
| `MooncakeLayerwiseConnector` | `kv_p2p.mooncake_layerwise_connector.MooncakeLayerwiseConnector` |
| `UCMConnector` | `kv_pool.ucm_connector.UCMConnectorV1` |
| `LMCacheAscendConnector` | 上游 `LMCacheConnectorV1`，额外加载 `lmcache_ascend` |
| `SimpleCPUOffloadConnector` | Ascend NPU worker 版 simple CPU offload |
| `RecomputeCPUOffloadConnector` | `RecomputeCPUOffloadConnectorV1` |

## 4. 多 connector 聚合

### `ascend_multi_connector.py`

核心类：`AscendMultiConnector(MultiConnector, SupportsHMA)`。

职责：在 vLLM `MultiConnector` 基础上补 Ascend 侧特殊逻辑。

关键方法：

- `__init__()`：检查所有子 connector 是否支持 HMA。如果 scheduler 启用 hybrid KV cache manager，则要求所有子 connector 都支持 HMA。
- `update_state_after_alloc()`：只把真实 blocks 传给命中的 connector 和 `MooncakeLayerwiseConnector`，其他 connector 传空 blocks，避免重复加载或错误更新状态。
- `get_num_new_matched_tokens()`：优先处理有 preempted request 的 connector，尤其是 `RecomputeCPUOffloadConnector`。原因是 recompute offload 可能持有 unhashed partial block，普通 prefix-cache connector 无法恢复这类 block。
- `update_state_before_preempt()`：广播 preempt hook 给支持该方法的子 connector。
- `request_finished_all_groups()`：在 HMA 多 KV group 场景聚合多个子 connector 的 async save 状态，并要求最多一个 connector 产出 `kv_transfer_params`。

重构注意：这里是多个 connector 生命周期协调点。修改 connector 排序、matched token 返回语义或 async save 聚合逻辑，都可能影响 preempt 恢复和 block 释放。

## 5. P2P Mooncake

### 5.1 `kv_p2p/mooncake_connector.py`

这是 Mooncake P2P pull 模式的主实现，文件内同时包含数据结构、scheduler 端、worker 端、发送线程、接收线程和工具函数。

#### 核心数据结构

- `MooncakeAgentMetadata`：远端 worker 暴露的元数据，包括 engine id、Mooncake rpc port、KV group 到 layer index 映射、block size、KV cache base addr、block len、stride、local ip、handshake port。
- `ReqMeta`：单请求传输元数据，包括本地/远端 block ids、远端 host/port/engine id、远端并行配置、num prompt blocks、remote block size 等。
- `GroupPull`：描述某个 KV group 如何从远端拉取，包括 group id、remote TP offset、需要 pull 的份数、prefill PP rank、是否该 group 最后一段。
- `GroupTransferInfo`：group 粒度传输信息，包括 tokens per block、SWA window blocks、是否 state group。
- `KVCacheTaskTracker`：管理发送/接收完成请求、延迟释放请求和超时强制释放。
- `SizedDict`：限制大小的 LRU-ish 字典，用于缓存远端 metadata/socket 相关状态。

#### `MooncakeConnector`

vLLM connector 门面类，继承 `KVConnectorBase_V1` 和 `SupportsHMA`。

- scheduler role 创建 `MooncakeConnectorScheduler`。
- worker role 创建 `MooncakeConnectorWorker`。
- 对 vLLM scheduler/worker hook 做转发。

主要转发 hook：

- scheduler side：`get_num_new_matched_tokens()`、`update_state_after_alloc()`、`build_connector_meta()`、`request_finished()`、`request_finished_all_groups()`、`set_xfer_handshake_metadata()`、`set_xfer_handshake_metadata_pp_aware()`。
- worker side：`register_kv_caches()`、`start_load_kv()`、`get_finished()`、`get_block_ids_with_load_errors()`、`get_handshake_metadata()`。

#### `MooncakeConnectorScheduler`

职责：scheduler 侧判断 remote prefill/decode、生成 worker metadata、在请求完成时生成反向 `kv_transfer_params`。

关键状态：

- `_reqs_need_recv`：D 侧已分配 blocks、需要 worker 拉取 KV 的请求。
- `_reqs_need_send`：P 侧等待远端完成接收后再释放的请求。
- `_reqs_in_batch`：本 batch 参与 connector 的请求。
- `multi_nodes_meta_mapping`：跨节点 worker handshake metadata。
- `kv_cache_groups`、`group_transfer_info`：hybrid groups、Mamba、压缩/SWA 裁剪依据。

关键方法：

- `get_num_new_matched_tokens()`：
  - `do_remote_prefill`：返回可从远端异步加载的 token 数。
  - `do_remote_decode` 且 Mamba/压缩需要 truncate：P 侧裁掉最后一个 prompt token，让 decoder 重新计算最后 token。
- `update_state_after_alloc()`：D 侧分配 KV blocks 后，把 request、本地 block ids、remote params 记到 `_reqs_need_recv`。
- `build_connector_meta()`：把 `_reqs_need_recv` 转成 `MooncakeConnectorMetadata` 传给 worker。
- `request_finished()` / `request_finished_all_groups()`：P 侧请求完成后生成给远端 decode/prefill 的 `kv_transfer_params`，包括 remote block ids、remote engine id、host、port、PCP/DCP/TP 信息、remote block size 等。
- `set_xfer_handshake_metadata_from_workers()`：收集 worker 上报的 host/port/engine id，用于跨 worker 映射。

#### `MooncakeConnectorWorker`

职责：worker 侧注册 KV cache buffer、初始化 Mooncake TransferEngine、启动发送/接收线程、根据 scheduler metadata 执行拉取。

关键方法：

- `register_kv_caches()`：
  - 遍历每层 KV cache tensor。
  - 收集 base addr、block len、stride、block size scale。
  - 注册 Mooncake memory buffer。
  - 构造 `MooncakeAgentMetadata`。
  - 根据 `kv_role` 启动 sending/recving thread。
- `_get_kv_split_metadata()`：最复杂的拆分逻辑。按 TP/PP/PCP/DCP、remote/local block size ratio、prefix hit、HMA、Mamba、压缩、SFA replicate-K 计算需要拉取的远端端口、本地 blocks、远端 blocks。
- `_get_group_pulls_metadata()`：为每个远端端口生成 `GroupPull` 列表。
- `start_load_kv()`：消费 `MooncakeConnectorMetadata`，调用接收线程 `add_request()`。
- `get_finished()`：返回接收/发送完成请求。
- `get_block_ids_with_load_errors()`：返回拉取失败的本地 block ids。

#### `KVCacheSendingThread`

职责：producer/P 侧暴露 metadata，并接收 D 侧传输完成信号。

工作方式：

- ZMQ ROUTER 监听 `GET_META_MSG`：返回编码后的 `MooncakeAgentMetadata`。
- ZMQ ROUTER 监听 `DONE_RECVING_MSG`：更新 `KVCacheTaskTracker`，达到远端端口完成计数后标记请求发送完成。
- 通过 `delayed_free_requests` 避免远端未完成时过早释放 KV blocks。

#### `KVCacheRecvingThread`

职责：consumer/D 侧主动从远端读取 KV cache。

关键流程：

1. `add_request()` 把 request 放入队列。
2. `_submit_request()` 按远端 peer 排队，避免单个 busy peer 长时间占用 executor。
3. `_handle_request()` 执行传输，并在 finally 中发送 done signal 释放远端资源。
4. `_get_remote_metadata()` 通过 ZMQ 向远端获取 `MooncakeAgentMetadata`。
5. `_transfer_kv_cache_all_groups()` 生成 `src_list`、`dst_list`、`length_list`。
6. 调用 `TransferEngine.batch_transfer_sync_read()` 从远端读到本地 NPU KV cache。
7. 根据需要执行 KV reformat：GQA head 拼接、NZ 格式、HMA hybrid linear、Mamba state 特殊布局。

关键 reformat 方法：

- `reformat_kv_cache_hybrid_linear_torch()`
- `reformat_kv_cache_with_fused_op()`
- `reformat_kv_cache()`
- `_cat_kv_cache()`
- `_nz_kv_cache()`
- `_append_mamba_transfer_meta()`

#### 关键数据流

```text
vLLM scheduler
  -> MooncakeConnector.get_num_new_matched_tokens()
  -> MooncakeConnector.update_state_after_alloc()
  -> MooncakeConnector.build_connector_meta()
worker model runner
  -> bind connector metadata
  -> MooncakeConnector.start_load_kv()
  -> MooncakeConnectorWorker.start_load_kv()
  -> _get_kv_split_metadata()
  -> KVCacheRecvingThread.add_request()
  -> _get_remote_metadata() over ZMQ
  -> TransferEngine.batch_transfer_sync_read()
  -> reformat_kv_cache*()
  -> get_finished()
scheduler
  -> request_finished_all_groups()
  -> 生成 kv_transfer_params 给远端
```

producer 侧释放流程：

```text
D worker transfer done
  -> KVCacheRecvingThread._send_done_recv_signal()
P worker KVCacheSendingThread receives DONE_RECVING_MSG
  -> KVCacheTaskTracker.update_done_task_count()
vLLM get_finished()
  -> blocks can be freed after async send completed
```

### 5.2 `kv_p2p/mooncake_hybrid_connector.py`

注册名：`MooncakeHybridConnector`。

类名同样是 `MooncakeConnector`，但实现是较旧或更窄的 hybrid 版本。

特点：

- 支持 hybrid KV groups、Mamba、压缩、SWA 裁剪。
- `MooncakeConnectorScheduler` 限制 `pcp_size * dcp_size == 1`，CP 支持比 `mooncake_connector.py` 少。
- worker 侧同样注册 KV cache 后启动 sending/recving thread。
- `KVCacheRecvingThread._transfer_kv_cache()` 处理普通 KV，`_transfer_kv_cache_all_groups()` 处理 hybrid group。

重构注意：该文件与 `mooncake_connector.py` 有较多重复，但能力不完全等价。合并前必须确认生产配置是否仍使用 `MooncakeHybridConnector`，以及哪些场景只在该实现里被验证过。

### 5.3 `kv_p2p/mooncake_layerwise_connector.py`

职责：Mooncake layerwise 推送型 P/D connector。与普通 Mooncake pull 模式不同，它在 producer forward 每层时通过 `save_kv_layer()` 推送当前层 KV。

核心类：

- `MooncakeLayerwiseConnector(KVConnectorBase_V1, SupportsHMA)`
- `MooncakeLayerwiseConnectorScheduler`
- `MooncakeLayerwiseConnectorWorker`
- `KVCacheSendingLayerThread`
- `KVCacheRecvingLayerThread`

scheduler 侧：

- consumer/D：
  - `get_num_new_matched_tokens()` 对 remote prefill 返回异步加载 token 数。
  - `update_state_after_alloc()` 登记需要接收的 request，并向 metaserver POST `kv_transfer_params` 通知 producer。
- producer/P：
  - 对 `do_remote_decode` 请求维护 `_reqs_need_send_layerwise`。
  - `build_connector_meta()` 根据 `SchedulerOutput` 计算当前 step 新增可发送 token 和 `chunk_finish`。

worker 侧：

- `register_kv_caches()`：构建每层 `LayerMetadata`，注册 Mooncake buffer，启动发送/接收线程。
- `start_load_kv()`：
  - consumer 记录 request map。
  - producer 计算 CP/TP/head 映射，并把 request 转成具体远端 host/port/block ids。
- `save_kv_layer()`：attention layer 保存阶段触发，按 layer 把 KV 加入 sending thread。
- 支持 pd_head_ratio 重分片、KV quant、C8 quant、NZ 转换。
- `get_block_ids_with_load_errors()`：返回 layerwise 传输失败 blocks。

layerwise 数据流：

```text
D scheduler remote prefill request
  -> update_state_after_alloc()
  -> POST metaserver with remote decode params
P scheduler receives do_remote_decode
  -> build_connector_meta() computes chunk send metadata
P worker start_load_kv()
  -> calculate port/block mappings
P attention layer forward
  -> save_kv_layer(layer_name, kv_layer, attn_metadata)
  -> KVCacheSendingLayerThread.batch_transfer_sync_write()
D worker KVCacheRecvingLayerThread receives DONE_SENDING_MSG
  -> mark done recving
```

## 6. KV pool / AscendStore

### 6.1 `kv_pool/lmcache_ascend_connector.py`

职责：加载 `lmcache_ascend` 扩展，然后导出上游 `LMCacheConnectorV1`。

它自身不是完整 connector 实现，主要是 Ascend 环境初始化适配。

### 6.2 `kv_pool/ucm_connector.py`

核心类：`UCMConnectorV1(KVConnectorBase_V1, SupportsHMA)`。

职责：包装 `ucm.integration.vllm.ucm_connector.UCMConnector`，补齐 vLLM 保留 hook 转发。

关键点：

- `_get_ucm_delegate_for()`：从 UCM dispatcher 中找到真实 delegate。
- `_call_ucm_reserved_hook()`：当 UCM dispatcher 没显式声明 vLLM 保留 hook 时，转发到内部 connector。
- 透传 worker hooks：`register_kv_caches()`、`start_load_kv()`、`wait_for_layer_load()`、`save_kv_layer()`、`wait_for_save()`、`get_block_ids_with_load_errors()`。
- 透传 scheduler hooks：`get_num_new_matched_tokens()`、`update_state_after_alloc()`、`build_connector_meta()`、`request_finished_all_groups()`、`update_connector_output()`。
- 支持 metrics、KV events、handshake metadata、reset cache 等 vLLM 扩展接口。

### 6.3 `kv_pool/ascend_store/backend/backend.py`

核心抽象：`Backend`。

必需方法：

- `set_device()`
- `register_buffer()`
- `exists()`
- `put()`
- `get()`

可选批量/lease 方法：

- `batch_get_key_info()`
- `batch_alloc()`
- `batch_add_lease()`
- `batch_remove_lease()`
- `batch_write_finish()`

`create_scheduler_client()` 默认直接创建 backend 实例。

### 6.4 `kv_pool/ascend_store/backend/__init__.py`

`backend_map`：

| backend 名 | 实现 |
| --- | --- |
| `mooncake` | `MooncakeBackend` |
| `memcache` | `MemcacheBackend` |
| `yuanrong` | `YuanrongBackend` |

### 6.5 `kv_pool/ascend_store/backend/memcache_backend.py`

核心类：`MemcacheBackend`。

职责：包装 `memcache_hybrid.DistributedObjectStore`。

特点：

- lazy init。
- A2 设备有特殊 buffer register 路径。
- `exists()` 调 `batch_is_exist`。
- `put()` 调 `batch_put_from_layers(... COPY_L2G)`。
- `get()` 调 `batch_get_into_layers(... COPY_G2L)`。
- 支持 GVA/layerwise 所需的 `batch_alloc()`、lease、`batch_write_finish()`。

### 6.6 `kv_pool/ascend_store/backend/mooncake_backend.py`

核心类：`MooncakeBackend`。

职责：包装 `mooncake.store.MooncakeDistributedStore`。

配置来源：`MOONCAKE_CONFIG_PATH`。

特点：

- `MooncakeStoreConfig` 支持 metadata server、segment size、protocol、SSD offload。
- `ASCEND_ENABLE_USE_FABRIC_MEM=1` 时走 fabric memory 路径。
- 非 fabric mem 时复用 `global_te` 的 TransferEngine。
- `register_buffer()` 在非 fabric mem 下注册 Mooncake memory。
- `put()` 调 `batch_put_from_multi_buffers`。
- `get()` 调 `batch_get_into_multi_buffers`。
- scheduler client 不贡献 memory。

### 6.7 `kv_pool/ascend_store/backend/yuanrong_backend.py`

核心类：`YuanrongBackend`。

职责：包装 `yr.datasystem` 的 `HeteroClient`。

配置来源：

- `DS_WORKER_ADDR`
- `DS_ENABLE_EXCLUSIVE_CONNECTION`
- `DS_ENABLE_REMOTE_H2D`

关键点：

- `YuanrongHelper.normalize_keys()`：清理过长或非法 key，并附 hash suffix。
- `put()` 调 `mset_d2h`。
- `get()` 调 `mget_h2d`。
- `exists()` 调 `exist`。
- 支持大批量 key 分片。

### 6.8 `kv_pool/ascend_store/config_data.py`

这是 AscendStore 的 key schema、metadata 和 request tracking 核心文件。

核心类型：

- `TPMismatchInfo` / `infer_tp_mismatch_info()`：判断 P/D TP 不一致时是否能拆成 sub-key。
- `KeyMetadata`：key 元数据。
- `PoolKey`：KV pool chunk key。
- `LayerPoolKey`：layerwise key。
- `ChunkedTokenDatabase`：把 token/block/hash/group 转成 key 与 value addr/size。
- `RequestTracker`：scheduler 侧跨 step 追踪 request token length、block ids、已保存 token、GVA 信息、Mamba speculative block。
- `ReqMeta`：worker 执行 load/save 的单请求 metadata。
- `AscendConnectorMetadata`：本 step 所有 `ReqMeta` 加 unfinished/preempt/loading/delayed_free 集合。
- layerwise 相关：`LayerBatchReqMeta`、`LayerBlockRange`、`SharedBlockData`、`LayerTransferTask`、`LayerLoadTask`、`LayerMultiBlockReqMeta`。
- `AscendStoreKVConnectorWorkerMetadata`：worker 完成事件聚合，用于 scheduler 释放延迟 blocks。

key 字段大致包含：

```text
model / head_or_tp_rank / pcp_rank / dcp_rank / pp_rank / kv_cache_group_id / cache_role / cache_family / chunk_hash
```

关键方法：

- `PoolKey.split_layers()`：生成 layerwise key。
- `infer_cache_family_from_ratio()` / `infer_group_cache_families()`：压缩 KV group 的 key 粒度。
- `ChunkedTokenDatabase.process_token_key_strings_with_block_ids()`：同时产出 key 和本地 block id。
- `ChunkedTokenDatabase.prepare_value()` / `prepare_value_layer()`：根据 block id、start/end 计算 NPU addr/size。
- `ChunkedTokenDatabase.store_mask()` / `load_mask()`：接入 `AscendStoreCoordinator`。
- `ChunkedTokenDatabase.decode_adaptor_prefill_pp()`：consumer 侧写回 pool 时按 prefill PP partition 拆 key/addr/size。

重构注意：key schema 是缓存兼容边界。字段、顺序、粒度变化会造成旧缓存不可读或错误命中。

### 6.9 `kv_pool/ascend_store/coordinator.py`

核心类：`AscendStoreCoordinator`。

职责：用 vLLM cache manager 的匹配逻辑判断 external KV pool 的 longest cache hit，而不是简单按 key 连续存在判断。

关键类：

- `ExternalCachedBlockPool`：把外部 pool key existence 包装成 vLLM block pool duck type。
- `AscendStoreCoordinator`：针对 hybrid KV groups、压缩、SWA/Eagle，复用 vLLM `SingleTypeKVCacheManager.find_longest_cache_hit()`。

关键方法：

- `find_longest_cache_hit()`：返回每 group mask 和最终 hit length。
- `load_mask()` / `store_mask()` / `lookup_mask()`：生成哪些 chunk 允许 load、store、lookup。
- `_get_manager_class()`：兼容 vLLM registry、旧 `spec_manager_map`，压缩场景优先 `CompressAttentionManager`。

重构注意：这里承担 correctness。绕开它做简单 key existence 会在 hybrid/Mamba/SWA/压缩场景误判。

### 6.10 `kv_pool/ascend_store/pool_scheduler.py`

核心类：`KVPoolScheduler`。

职责：AscendStore scheduler 端。负责查外部 pool 命中、构造 load/save metadata、管理 async save/load 生命周期和 block 延迟释放。

初始化关注点：

- role：`kv_producer` / `kv_consumer` / `kv_both`
- `use_layerwise`
- backend
- `load_async`
- `consumer_is_to_load` / `consumer_is_to_put`
- `save_decode_cache`
- hybrid KV groups、Mamba groups、SWA blocks
- hash block size、cache transfer granularity

关键方法：

- `get_num_new_matched_tokens()`：
  - consumer 可 load 时查询 external KV pool。
  - 非 layerwise 通过 `LookupKeyClient` 向 rank0 worker 的 `LookupKeyServer` 查 pool hit。
  - layerwise memcache GVA 通过 `batch_get_key_info()` 查 GVA。
  - 命中后生成 `LoadSpec`，返回需要分配的 token 数和是否 async load。
- `update_state_after_alloc()`：
  - 记录 `_unfinished_requests`。
  - 若存在 `LoadSpec`，标记可 load。
  - async load 加入 `_loading_req_ids`。
- `build_connector_meta()`：
  - 清理 finished/preempted 状态。
  - 处理 scheduled new req、cached req、preempted cached req、async load req。
  - 构造 `ReqMeta` 并加入 `AscendConnectorMetadata`。
- `request_finished()` / `request_finished_all_groups()`：
  - 非 layerwise 且有保存任务时延迟释放 blocks。
  - hybrid 场景按 SWA 裁剪 block ids。
- `update_connector_output()`：聚合 worker 上报 completed event，所有 worker 完成后释放 Mamba/state 延迟 blocks。
- `bind_gpu_block_pool()`：让 scheduler touch/free vLLM GPU block pool。

辅助类：

- `LookupKeyClient`：ZMQ REQ 发送 token_len、KV group ids、HBM hit tokens、block hashes 给 worker rank0 查询。
- `get_zmq_rpc_path_lookup()`：构造 IPC path，支持 `lookup_rpc_port`，兼容旧 `mooncake_rpc_port`。

### 6.11 `kv_pool/ascend_store/pool_worker.py`

核心类：`KVPoolWorker`。

职责：AscendStore worker 端。负责注册 KV cache buffer、初始化 backend、执行 load/save、启动 transfer threads、提供 lookup 服务。

初始化拆分：

- `_init_parallelism_info()`
- `_init_kv_transfer_config()`
- `_init_key_head_config()`
- `_init_metadata()`
- `_init_backend()`
- `_init_layerwise_config()`

关键方法：

- `register_kv_caches()`：
  - 解析 KV cache tensor base addr、block len、stride。
  - 合并 storage register regions。
  - 对 hybrid group 生成 group 级 addr/len/stride/num_layers。
  - 设置 `ChunkedTokenDatabase` group buffers。
  - 注册 backend buffer。
  - 启动 transfer threads。
- `_start_kv_transfer_threads()`：
  - 非 layerwise save：`KVCacheStoreSendingThread`
  - 非 layerwise async load：`KVCacheStoreRecvingThread`
  - layerwise key path：`KVCacheStoreKeyLayerSendingThread`、`KVCacheStoreKeyLayerRecvingThread`
  - layerwise memcache GVA path：`KVCacheStoreLayerSendingThread`、`KVCacheStoreLayerRecvingThread`
- `start_load_kv()`：
  - layerwise：`process_layer_data()` 预构建每层 load/save task。
  - 非 layerwise sync load：直接组 key/addr/size 调 backend `get()`。
  - 非 layerwise async load：丢给 `kv_recv_thread.add_request()`。
- `wait_for_layer_load()`：layerwise 每层 attention 前提交/等待对应 load task。
- `save_kv_layer()`：layerwise 每层 forward 后提交 save task。
- `wait_for_save()`：非 layerwise save，给 sending thread 提交 `ReqMeta`。
- `lookup_scheduler()`：给 scheduler lookup server 使用，跨所有 TP/PP/CP rank key 判断最长可命中 token。
- `lookup()`：worker 本地查询。
- `_load_kv_tp_mismatch()` / `_store_kv_tp_mismatch()`：P/D TP 不一致时用 strided addr 按 head sub-key 拆分读写。
- `get_finished()`：收集 send/recv 完成请求，处理 preempted/stale finished。
- `get_block_ids_with_load_errors()`：返回并清空失败 block ids。
- `build_connector_worker_meta()`：Mamba 发送完成事件上报 scheduler。

设计要点：

- rank0 `LookupKeyServer` 让 scheduler 不直接连接所有 store worker。
- layerwise GVA 模式把 GVA allocation 放在 worker，因为 memcache `batch_alloc` 与 `batch_copy` 依赖同进程 tracker。
- hybrid load 失败不直接上报 partial group block，避免 scheduler 因 group 不一致崩溃。

### 6.12 `kv_pool/ascend_store/kv_transfer.py`

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

`LayerBatchBuilder`：

- 为 GVA layerwise 模式预计算跨层共享 block ids/GVAs。
- 按 layer 生成 addr/size/gva arrays。
- `_batch_copy_with_limits()` 支持按 max blocks/bytes 分包。

`KVTransferThread`：

- 所有 store transfer thread 基类。
- 管理 queue、finished requests、stored request ref count、KV events。

非 layerwise：

- `KVCacheStoreSendingThread`：
  - `_handle_stored_request()` 按 group/token chunk 生成 key。
  - 先 `exists()`，缺失才 `put()`。
  - 支持 store mask、load hit skip、consumer 写回 prefill PP adaptor、KV events。
- `KVCacheStoreRecvingThread`：
  - 按 load spec 生成 key/addr/size。
  - 调 backend `get()`。
  - 记录失败 block。

layerwise key 模式：

- `KVCacheStoreKeyLayerSendingThread`
- `KVCacheStoreKeyLayerRecvingThread`

layerwise memcache GVA 模式：

- `KVCacheStoreLayerSendingThread`
- `KVCacheStoreLayerRecvingThread`
- 使用 `store.batch_copy()` L2G/G2L。
- save 最后一层调用 `batch_write_finish()`。
- load 最后一层释放 read lease。

辅助函数：

- `record_failed_blocks()`：根据 backend ret code 标记失败 block ids。

### 6.13 `kv_pool/ascend_store/ascend_store_connector.py`

核心类：`AscendStoreConnector(KVConnectorBase_V1, SupportsHMA)`。

职责：vLLM connector 门面。按 role 创建 `KVPoolScheduler` 或 `KVPoolWorker`，并在非 layerwise worker rank0 启动 `LookupKeyServer`。

关键方法：

- `requires_piecewise_for_cudagraph()`：layerwise 模式要求 PIECEWISE CUDA graph。
- scheduler hooks：
  - `get_num_new_matched_tokens()`
  - `update_state_after_alloc()`
  - `build_connector_meta()`
  - `request_finished_all_groups()`
  - `update_connector_output()`
  - `take_events()`
- worker hooks：
  - `register_kv_caches()`
  - `start_load_kv()`
  - `wait_for_layer_load()`
  - `save_kv_layer()`
  - `wait_for_save()`
  - `get_finished()`
  - `get_block_ids_with_load_errors()`
  - `build_connector_worker_meta()`

辅助类：

- `AscendStoreKVEvents`：包装 vLLM `KVEventAggregator`，聚合多 worker common KV events。
- `LookupKeyServer`：ZMQ REP 服务，接收 scheduler lookup 请求，调用 `KVPoolWorker.lookup_scheduler()` 返回命中 token 数。

AscendStore 非 layerwise 数据流：

```text
scheduler get_num_new_matched_tokens()
  -> LookupKeyClient.lookup()
  -> worker rank0 LookupKeyServer
  -> KVPoolWorker.lookup_scheduler()
  -> backend.exists()
scheduler update_state_after_alloc()
  -> LoadSpec.can_load = True
scheduler build_connector_meta()
  -> AscendConnectorMetadata[ReqMeta]
worker start_load_kv()
  -> sync backend.get() or KVCacheStoreRecvingThread async get()
worker wait_for_save()
  -> KVCacheStoreSendingThread
  -> exists missing keys
  -> backend.put()
worker get_finished()
  -> done_sending/done_recving
scheduler update_connector_output()/request_finished_all_groups()
  -> release delayed blocks
```

AscendStore layerwise 数据流：

```text
worker start_load_kv()
  -> KVPoolWorker.process_layer_data()
  -> build layer_load_tasks/layer_save_tasks
each layer attention
  -> wait_for_layer_load()
  -> KVCacheStoreKeyLayerRecvingThread or KVCacheStoreLayerRecvingThread
  -> save_kv_layer()
  -> KVCacheStoreKeyLayerSendingThread or KVCacheStoreLayerSendingThread
final layer
  -> set finished request / release leases / batch_write_finish
```

## 7. CPU offload

### 7.1 `kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py`

核心类：`AscendSimpleCPUOffloadConnector(SimpleCPUOffloadConnector)`。

职责：复用上游 simple CPU offload connector 的 scheduler 和大部分逻辑，只在 worker role 下替换为 Ascend NPU worker。

特点：

- 使用 Ascend NPU 的 memcpy、stream、event 能力做 CPU offload。
- 文件职责很窄，是上游 connector 的 Ascend worker 适配层。

### 7.2 `kv_pool/recompute_cpu_offload/metadata.py`

核心数据类：

- `RecomputeCPUOffloadMetadata`：scheduler -> worker 的 D2H/H2D transfer 任务。
  - `need_flush`
  - `preempt_store_event`
  - `preempt_store_gpu_blocks`
  - `preempt_store_cpu_blocks`
  - `preempt_load_event`
  - `preempt_load_gpu_blocks`
  - `preempt_load_cpu_blocks`
  - `preempt_load_event_to_reqs`
- `RecomputeCPUOffloadWorkerMetadata`：worker -> scheduler 的已完成 store event 聚合数据。

### 7.3 `kv_pool/recompute_cpu_offload/manager.py`

核心类：

- `RecomputeCPUOffloadScheduler`
- `PreemptedRequestState`

职责：scheduler 侧管理 preempt 后的 CPU block、store/load event 和 request 恢复状态。

关键点：

- 使用 vLLM `KVCacheCoordinator` 和 CPU `BlockPool` 管理 offload block。
- `bind_gpu_block_pool()`：绑定 GPU block pool，以便 touch/free GPU blocks。
- `update_state_before_preempt()`：抢占前创建 `PreemptedRequestState`，为已计算 block 分配 CPU block。
- `_create_preempt_state()`：区分 hashed full block 和 unhashed partial block，可选 prefix caching 共享 CPU blocks。
- `build_connector_meta()`：生成本 step D2H store 和 H2D load 任务。
- `get_num_new_matched_tokens()`：恢复 preempted request 时返回可从 CPU 恢复的 token 数；未 ready 时返回 `(None, False)` 阻止继续调度。
- `update_state_after_alloc()`：新 GPU blocks 分配后准备 H2D load mapping。
- `update_connector_output()`：聚合 worker 完成 store events，所有 worker 完成后将 CPU block 置 ready。
- `request_finished_all_groups()` / `reset_cache()` / `take_events()`：生命周期与事件管理。

### 7.4 `kv_pool/recompute_cpu_offload/worker.py`

核心类：`RecomputeCPUOffloadWorker`。

职责：worker 侧执行 D2H/H2D copy。

关键方法：

- `register_kv_caches()`：将唯一 GPU KV cache tensor flatten，按 CPU 容量分配 pinned CPU tensors。
- `handle_preemptions()`：抢占发生时同步 D2H，避免 GPU block 被 scheduler 复用前数据丢失。
- `start_load_kv()`：提交 H2D 恢复。
- `wait_for_layer_load()`：当前 stream 等待 load stream。
- `_submit_transfer()`：按 block id 遍历所有 KV tensors 做 CPU/NPU copy。
- `get_finished()`：返回完成恢复的 request ids。
- `build_connector_worker_meta()`：上报完成的 store events。

### 7.5 `kv_pool/recompute_cpu_offload/recompute_cpu_offload_connector.py`

核心类：`RecomputeCPUOffloadConnectorV1(KVConnectorBase_V1, SupportsHMA)`。

职责：vLLM connector 门面。

配置来源：`kv_connector_extra_config`：

- `cpu_bytes_to_use`
- `cpu_bytes_to_use_per_rank`
- `enable_offload_prefix_caching`

按 role 创建：

- scheduler role：`RecomputeCPUOffloadScheduler`
- worker role：`RecomputeCPUOffloadWorker`

暴露 hooks：

- `bind_connector_metadata()`
- `handle_preemptions()`
- `start_load_kv()`
- `wait_for_layer_load()`
- `get_finished()`
- `build_connector_worker_meta()`
- `update_connector_output()`
- `update_state_before_preempt()`
- `has_preempted_request()`
- `get_num_new_matched_tokens()`
- `update_state_after_alloc()`
- `request_finished_all_groups()`

Recompute CPU offload 数据流：

```text
scheduler preempt
  -> AscendMultiConnector.update_state_before_preempt()
  -> RecomputeCPUOffloadScheduler.update_state_before_preempt()
  -> allocate CPU blocks, prepare D2H store
worker handle_preemptions()
  -> RecomputeCPUOffloadWorker._submit_transfer(D2H, sync=True)
worker build_connector_worker_meta()
  -> scheduler update_connector_output()
  -> state.ready = True
request resumes
  -> get_num_new_matched_tokens()
  -> update_state_after_alloc()
  -> worker start_load_kv() H2D
  -> wait_for_layer_load()
  -> get_finished() returns restored req ids
```

## 8. 公共工具

### 8.1 `utils/mooncake_transfer_engine.py`

核心类：`GlobalTE`。

职责：维护进程级 Mooncake `TransferEngine` 单例。

关键方法：

- `get_transfer_engine()`：初始化 `TransferEngine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)`。
- `register_buffer()`：一次性注册 ptr/size，带 lock 防并发。

模块级对象：`global_te`。

使用方：

- Mooncake P2P connector。
- Mooncake Store backend。

重构注意：`global_te` 是进程级单例，多 connector、多模型、多次重载时可能存在状态污染或 buffer 注册生命周期问题。

### 8.2 `utils/utils.py`

职责：并行映射、KV 重排、内存 region 注册和超时计算工具。

主要能力：

并行映射：

- `parallel_info`
- `get_cp_group()`
- `context_parallel_parameters_check()`
- `get_tp_rank_head_mapping()`
- `get_head_group_mapping()`
- `get_local_remote_block_port_mappings()`
- `get_transfer_mappings()`

KV 重排：

- `kv_alltoall_and_rearrange()`
- `alltoall_and_rearrange()`
- `rearrange_output()`

内存注册：

- `align_memory()`
- `RegisterRange`
- `RegisterRegions`
- `iter_kv_cache_tensors()`
- `collect_storage_merged_register_regions()`
- `validate_register_region_count()`

超时：

- `get_transfer_timeout_value()`：由 `ASCEND_TRANSFER_TIMEOUT` 或 HCCL RDMA timeout/retry 推导。

## 9. vLLM 接口适配点

### 9.1 `KVConnectorFactory`

所有 Ascend connector 通过 `register_connector()` 动态注册或覆盖上游注册名。

### 9.2 `KVConnectorBase_V1`

scheduler side 常用 hook：

- `get_num_new_matched_tokens()`
- `update_state_after_alloc()`
- `build_connector_meta()`
- `request_finished()`
- `request_finished_all_groups()`
- `update_connector_output()`
- `take_events()`

worker side 常用 hook：

- `register_kv_caches()`
- `bind_connector_metadata()`
- `start_load_kv()`
- `wait_for_layer_load()`
- `save_kv_layer()`
- `wait_for_save()`
- `get_finished()`
- `get_block_ids_with_load_errors()`
- `build_connector_worker_meta()`

### 9.3 `SupportsHMA`

声明支持 hybrid memory / hybrid KV cache group 的 connector：

- `AscendMultiConnector`
- `MooncakeConnector`
- `MooncakeLayerwiseConnector`
- `AscendStoreConnector`
- `UCMConnectorV1`
- `RecomputeCPUOffloadConnectorV1`

关键入口：`request_finished_all_groups()`。

### 9.4 `Request.kv_transfer_params`

P/D disaggregated prefill/decode 的跨 engine 参数载体。

Mooncake P2P 依赖字段包括：

- `do_remote_prefill`
- `do_remote_decode`
- `remote_block_ids`
- `remote_engine_id`
- `remote_request_id`
- `remote_host`
- `remote_port`
- `remote_pcp_size`
- `remote_dcp_size`
- `remote_ptp_size`
- `remote_multi_nodes_meta_mapping`
- `num_prompt_blocks`
- `remote_block_size`

### 9.5 `SchedulerOutput`

AscendStore 和 layerwise Mooncake 从 scheduler output 中读取：

- `scheduled_new_reqs`
- `scheduled_cached_reqs`
- `num_scheduled_tokens`
- `preempted_req_ids`
- `finished_req_ids`

用于构造 connector metadata。

### 9.6 `KVCacheBlocks`

常用方法：

- `get_block_ids()`
- `get_unhashed_block_ids_all_groups()`
- `new_empty()`

用于获取本地分配 blocks 或给未命中 connector 传空 blocks。

### 9.7 `KVCacheConfig` / `KVCacheSpec`

大量逻辑围绕以下结构展开：

- `KVCacheConfig.kv_cache_groups`
- `FullAttentionSpec`
- `MambaSpec`
- `MLAAttentionSpec`
- `SlidingWindowSpec`
- `UniformTypeKVCacheSpecs`

影响 key 粒度、block 裁剪、layer 分组、是否需要 HMA、是否需要 Mamba state 特殊处理。

### 9.8 vLLM block pool

AscendStore 和 Recompute CPU offload 通过 `bind_gpu_block_pool()` touch/free blocks，实现 async save 或 preempt 恢复时延迟释放。

### 9.9 vLLM KV events

- AscendStore 生成 `BlockStored`，通过 `KVConnectorKVEvents` / `KVEventAggregator` 聚合。
- Recompute CPU offload 复用 CPU block pool events。

### 9.10 handshake metadata

Mooncake 通过以下 hook 建立跨 worker host/port/engine 映射：

- `get_handshake_metadata()`
- `set_xfer_handshake_metadata()`
- `set_xfer_handshake_metadata_pp_aware()`

AscendStore 不依赖 P/D handshake，PP 等信息编码进 pool key。

### 9.11 graph mode

`AscendStoreConnector.requires_piecewise_for_cudagraph()` 在 layerwise 模式要求 PIECEWISE CUDA graph。

## 10. 主要横向数据流对比

### Mooncake P2P pull

- D 侧 scheduler 根据 `kv_transfer_params` 判断 remote prefill。
- D 侧 worker 主动通过 Mooncake `batch_transfer_sync_read()` 从 P 侧读 KV。
- P 侧 worker 只暴露 metadata 和接收 done signal。
- 适合一次性拉取完整 prompt KV 或 group 粒度 KV。

### Mooncake layerwise push

- P 侧 producer 在每层 attention forward 后 `save_kv_layer()` 推送当前层 KV。
- D 侧 consumer 监听完成信号。
- 适合边算边传，降低等待完整 prefill 结束的延迟。

### AscendStore KV pool

- scheduler 查外部 pool hit。
- worker 从 pool load 或 save 到 pool。
- backend 可为 Mooncake Store、memcache、YuanRong。
- 支持 key 模式和 memcache GVA layerwise 模式。

### Recompute CPU offload

- 抢占时将 GPU KV blocks D2H 保存到 CPU pinned memory。
- 请求恢复时 H2D 复制回 GPU。
- 解决 preempt 后 KV cache 恢复问题，尤其是 unhashed partial block。

## 11. 重构高风险点

1. `mooncake_connector.py` 和 `mooncake_hybrid_connector.py` 重复度高但能力不完全等价。合并前必须确认注册名使用场景和覆盖能力差异。

2. `MooncakeConnectorWorker._get_kv_split_metadata()` 是 P2P 传输的复杂核心，耦合 TP、PP、PCP、DCP、block size ratio、prefix hit、HMA、Mamba、SFA replicate-K。任何改动都需要组合测试。

3. ZMQ 协议和端口偏移逻辑是隐式契约。`handshake_port = base + rank-derived-offset` 对 DP/TP/PP/CP 排布敏感。

4. `global_te` 是进程级单例，buffer 注册只有一次性语义。多 connector、多模型、多次重载场景可能有状态污染。

5. 内存注册依赖 2MB 对齐和 HCCL region 数限制。KV cache tensor storage/view 变化会影响 `collect_storage_merged_register_regions()` 和 register region 合并结果。

6. AscendStore key schema 是外部缓存兼容边界。字段、顺序、hash 粒度、cache family 变化都会影响旧缓存读取和命中正确性。

7. Hybrid/压缩/Mamba 下有多套 block 粒度：原始 block size、grouped block size、hash block size、effective cache family block size、cache transfer granularity。容易出现 off-by-one 或 partial block 裁剪错误。

8. Layerwise GVA memcache 路径依赖 `batch_alloc()`、`batch_add_lease()`、`batch_copy()`、`batch_write_finish()` 的同进程状态，不能简单把 allocation 移到 scheduler。

9. `get_block_ids_with_load_errors()` 在单 group 和 hybrid group 行为不同。hybrid 失败目前避免上报 partial group block，避免 scheduler 因 group 不一致崩溃。

10. Recompute CPU offload 保存 unhashed partial block，因此 `AscendMultiConnector.get_num_new_matched_tokens()` 必须优先检查 preempted request。调整 MultiConnector 排序会影响恢复正确性。

11. `request_finished_all_groups()` 与 async save/delayed free 是资源生命周期关键点。漏报 `done_sending` 可能 block 泄漏，过早完成可能提前释放仍在远端读取的 blocks。

12. 版本兼容点较多，包括 vLLM 版本判断、`prefix_match_unit/hash_block_size`、KVCacheSpecRegistry、request id suffix 处理。升级 vLLM 时应专项测试。

13. 后端返回码语义不统一：Mooncake 常见负数失败，memcache 非 0 失败，YuanRong 返回 failed keys 映射。抽象层没有完全统一错误模型。

14. NPU stream/event 同步有保守同步点，例如 GQA 场景 `torch.npu.synchronize()`。优化同步可能提升性能，但风险是 async copy 或 reformat 数据竞争。

15. P/D TP mismatch、PP partition、CP rank/head mapping 分散在多个工具和 worker 方法中。重构时应先抽出可测试纯函数，再改线程和 connector 生命周期。

## 12. 建议的重构切入顺序

1. 先冻结接口和行为：整理所有 connector 注册名、配置项、必需 hook、metadata 字段，避免重构时破坏外部配置兼容。

2. 先拆纯逻辑，后动线程：优先把 key 生成、block 裁剪、TP/CP/head mapping、group transfer metadata 拆成纯函数或小类，并补针对性单测。

3. 抽象 transfer lifecycle：把 request tracking、done signal、delayed free、failed blocks 收敛成统一生命周期模型，再分别接 Mooncake/Store/CPU offload。

4. 统一 backend 错误模型：为 `exists/get/put` 定义统一结果对象，避免每个 thread 手动解释 ret code。

5. 合并 Mooncake 普通/hybrid 前先做能力矩阵：列出 CP/HMA/Mamba/SWA/压缩/PP/TP mismatch 组合，确认两份实现的差异和当前生产使用情况。

6. 对 AscendStore key schema 做版本化：后续如需改 key 字段或粒度，建议显式加 schema version，而不是隐式改变字符串格式。

7. 对全局状态做生命周期边界：`global_te`、ZMQ sockets、backend clients、buffer registration 应明确 init/reuse/close/reset 语义。

## 13. 后续文档建议

本文件是 inventory。后续重构可以继续补充：

- `kv_transfer_refactor_plan.md`：目标架构与阶段计划。
- `kv_transfer_config_matrix.md`：connector 注册名、配置项、支持能力矩阵。
- `kv_transfer_test_matrix.md`：P/D、TP/PP/CP、HMA、Mamba、SWA、压缩、layerwise、backend 的测试矩阵。
- `kv_transfer_lifecycle.md`：request 从 scheduler 到 worker、load/save、done/free 的完整状态机。
