# AscendStore KV Pool 代码逻辑概览

源码范围：`code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`

这套代码实现的是 vLLM Ascend 侧的外部 KV Pool Connector：把本地 HBM 中的 KV Cache 按 token/block 粒度写入外部存储，并在后续请求命中相同 prefix 时，从外部 KV Pool 拉回 KV Cache，减少重复 prefill 计算。

## 1. 最外层入口：AscendStoreConnector

`ascend_store_connector.py` 是 vLLM KV Connector 的入口封装。

- scheduler 角色创建 `KVPoolScheduler`，负责命中查询、生成传输元数据、管理请求状态。
- worker 角色创建 `KVPoolWorker`，负责注册 KV cache 内存、实际执行 load/save。
- 非 layerwise 且 rank 0 的 worker 会启动 `LookupKeyServer`，scheduler 侧通过 ZMQ RPC 查询外部 KV Pool 命中长度。
- `use_layerwise=True` 时要求 piecewise graph 模式，因为 load/save 按层穿插在 forward 过程中。

典型调用分层：

```text
vLLM scheduler side
  -> AscendStoreConnector.get_num_new_matched_tokens()
  -> KVPoolScheduler 查询外部命中
  -> AscendStoreConnector.build_connector_meta()
  -> 生成 AscendConnectorMetadata / ReqMeta

vLLM worker side
  -> AscendStoreConnector.register_kv_caches()
  -> KVPoolWorker 记录 KV cache 地址与大小
  -> AscendStoreConnector.start_load_kv()
  -> KVPoolWorker 根据 metadata 加载 KV
  -> AscendStoreConnector.wait_for_save()/save_kv_layer()
  -> KVPoolWorker 异步或逐层保存 KV
```

## 2. 后端抽象：Backend 与三种 store

`backend/backend.py` 定义统一接口：

- `exists(keys)`：查询 key 是否存在。
- `put(keys, addrs, sizes)`：把本地地址中的 KV 写入外部存储。
- `get(keys, addrs, sizes)`：把外部 KV 读回本地地址。
- `register_buffer(ptrs, lengths)`：把本地 KV cache 内存注册给后端。
- memcache 额外支持 `batch_alloc`、`batch_get_key_info`、lease、`batch_write_finish`，主要服务于 GVA layerwise 路径。

`backend/__init__.py` 中的 `backend_map` 支持三种后端：

| backend | 文件 | 主要特点 |
| --- | --- | --- |
| `mooncake` | `mooncake_backend.py` | 使用 MooncakeDistributedStore，支持普通 batch put/get，也支持 SSD offload 配置。 |
| `memcache` | `memcache_backend.py` | 使用 memcache_hybrid，支持普通 key I/O，也支持 GVA 分配、lease、batch_copy 逐层传输。 |
| `yuanrong` | `yuanrong_backend.py` | 使用 openyuanrong datasystem，负责 key 规范化和异构 H2D/D2H 传输。 |

三者都被 `KVPoolWorker` 和 `KVPoolScheduler` 通过 `backend_map` 动态导入，不直接写死具体实现。

## 3. Key 与元数据：config_data.py

`config_data.py` 是整个模块的数据模型中心。

### 3.1 KV Pool key 的组成

`KeyMetadata` 描述 key 的命名维度：

- 模型名 `model_name`
- TP/head 维度 `head_or_tp_rank`
- prefill context parallel rank：`pcp_rank`
- decode context parallel rank：`dcp_rank`
- pipeline parallel rank：`pp_rank`
- KV cache group：`kv_cache_group_id`
- cache role：默认 `kv`
- cache family：默认 `default`，压缩 cache 可能是 `c2`、`c4` 等

`PoolKey.to_string()` 会把这些字段和 chunk hash 拼成稳定字符串。layerwise 模式下，`PoolKey.split_layers(num_layers)` 会生成每层独立的 `LayerPoolKey`。

简化后的 key 形态：

```text
model@pcpX@dcpY@head_or_tp_rank:Z@pp_rank:P@group:G@cache_role:kv@cache_family:C@<chunk_hash>

layerwise key 额外包含：
@layer_id:L
```

### 3.2 ChunkedTokenDatabase

`ChunkedTokenDatabase` 负责把“token 范围 + block hash + block id”转换成“外部 store key + 本地内存地址 + size”。

核心职责：

- 根据 block/hash 粒度把 token 切成 chunk。
- 根据 cache family 处理压缩模型的有效粒度，例如 `c2` 表示外部 key 粒度是基础 block 的 2 倍。
- 根据 KV cache group 选择对应 block size、base address、block length、stride。
- 生成普通 key 或 layerwise key。
- `prepare_value()` 根据 block id 算出本地 KV cache 的地址列表和 size 列表。
- `decode_adaptor_prefill_pp()` 在 consumer 写回 prefill PP 场景下，把一个 key 拆成多个 PP rank 的 key。

可以理解为：它是“逻辑 token/block”到“后端传输参数”的翻译层。

### 3.3 请求级元数据

- `LoadSpec`：记录本地 vLLM 已缓存多少 token、KV Pool 命中多少 token、是否允许 load。
- `RequestTracker`：scheduler 侧持续跟踪请求的 token 长度、已分配 block、已保存 token 数等。
- `ReqMeta`：每个调度 step 发送给 worker 的请求级传输计划，包含 save 区间、load spec、block hashes、block ids、事件、layerwise 所需 GVA 信息等。
- `AscendConnectorMetadata`：一个 step 中所有 `ReqMeta` 的容器，同时携带 preempted/loading/delayed-free 请求集合。

## 4. Scheduler 侧：KVPoolScheduler

`pool_scheduler.py` 负责决定“这个请求能从外部 KV Pool 命中多少 token，以及本 step 要让 worker 做什么”。

### 4.1 命中查询

入口是 `get_num_new_matched_tokens(request, num_computed_tokens)`：

1. 如果当前角色不允许 load，直接返回 0。
2. 按 `cache_transfer_granularity` 对 prompt 长度取整，避免保存/加载不完整 chunk。
3. 根据模式选择查询方式：
   - 普通非 layerwise：通过 `LookupKeyClient` 走 ZMQ 请求 worker rank 0 的 `LookupKeyServer`。
   - key-based layerwise：scheduler 直接查询所有 layer key 是否存在。
   - memcache GVA layerwise：通过 `batch_get_key_info` 查询所有 rank 的 GVA 是否有效。
4. 得到外部命中 token 数后，和本地已计算 token 数比较，计算还需要分配并加载多少 token。
5. 生成 `LoadSpec`，后续 `build_connector_meta()` 会塞进 `ReqMeta`。

### 4.2 构建传输 metadata

`build_connector_meta(scheduler_output)` 会遍历本 step 调度结果：

- 新请求：创建新的 `RequestTracker`，记录已分配 block 和 prompt token。
- cached/running 请求：更新 tracker 的 token 数和新增 block。
- async load 请求：如果之前已有 `LoadSpec` 但当时没被调度，也会补一条 load 元数据。
- preempted/finished 请求：清理 tracker 和状态集合。

最后返回 `AscendConnectorMetadata`，worker 侧据此执行 load/save。

### 4.3 block 生命周期保护

非 layerwise 保存可能是异步的。请求 finished 时，如果 KV 还在保存中，scheduler 会通过 `_delayed_free_req_ids` 延迟释放 block。mamba/hybrid 场景还会用 `touch_sending_mamba_blocks()` 保持 block 引用，等 worker 上报完成事件后再 free。

## 5. Worker 侧：KVPoolWorker

`pool_worker.py` 是 worker 侧主类，负责实际传输。

### 5.1 初始化阶段

初始化时会完成：

- 读取模型并行信息：TP、PP、PCP、DCP rank/size。
- 判断 KV cache 形态：普通、hybrid、mamba、sparse、MLA、压缩 cache。
- 推导 block size：原始 block size、CP 放大后的 grouped block size、hash block size、cache transfer granularity。
- 推导 key 的 rank 维度：当 `num_kv_head < tp_size` 时，多个 TP rank 可能共享同一份 KV，需要 `put_step` 降低重复写入。
- 动态创建 backend 实例。
- 准备 layerwise 配置和线程状态。

### 5.2 注册 KV cache

`register_kv_caches(kv_caches)` 是 worker 真正拿到本地 KV cache tensor 后的关键步骤：

1. 计算每个 cache tensor 的 base address、block length、block stride、region length。
2. 对共享 storage 的 tensor 合并注册范围。
3. hybrid 场景下做 2MB 对齐修正。
4. 按 KV cache group 记录每组 cache 的地址与层数。
5. 调用 backend `register_buffer()`。
6. 启动传输线程。

从这一刻开始，后端就知道本地 KV cache 内存区域，后续 `get/put/batch_copy` 可以直接按地址搬运数据。

### 5.3 普通非 layerwise load/save

普通路径中，`start_load_kv()` 会检查每个 `ReqMeta.load_spec`：

- 如果可 load：根据 block hash 生成 key，根据 block id 生成本地地址，调用 backend `get()`。
- 如果 `load_async=True`：把请求交给 `KVCacheStoreRecvingThread` 异步执行。
- 如果 load 失败：单组 KV cache 会把失败 block id 记录到 `_invalid_block_ids`，让 scheduler 后续回退重算；hybrid 多组场景避免部分 group 失败导致不一致，只记录错误。

save 则通过 `wait_for_save()` 把可保存请求交给 `KVCacheStoreSendingThread`：

- 跳过已经从 KV Pool 命中的 prefix，避免重复写。
- 查询 key 是否已存在，只保存缺失 chunk。
- 同步 NPU event，确保 forward 写 KV 已完成。
- 调用 backend `put()`。
- 可选生成 KV cache event。

## 6. 逐层传输：layerwise 模式

`use_layerwise=True` 时，KV load/save 不再等完整 forward 结束，而是按 layer 编排：

```text
start_load_kv()
  -> process_layer_data()
      -> 为每层构造 save task
      -> 为每层构造 load task

forward 每到一层：
  -> wait_for_layer_load(layer)
  -> attention/compute
  -> save_kv_layer(layer)
```

它的目的通常是把外部 KV 传输和模型逐层计算重叠起来，减少等待。

### 6.1 key-based layerwise

适用于非 memcache GVA 路径。每个 `(chunk, layer)` 都有独立 key：

- save：`KVCacheStoreKeyLayerSendingThread`
- load：`KVCacheStoreKeyLayerRecvingThread`

该路径直接用 backend `put/get` 传输每层 KV 的地址和 size。

### 6.2 memcache GVA layerwise

当 `backend=memcache` 且 `use_layerwise=True` 时，使用 GVA 路径：

- 保存前 `_alloc_gvas_for_save()` 为每个 block 分配外部 GVA。
- `LayerBatchBuilder` 把 block ids/GVAs 转成每层的本地地址、size、GVA 数组。
- 保存线程 `KVCacheStoreLayerSendingThread` 调用 memcache 的 `batch_copy(..., direction=0)`，即 L2G。
- 加载前 `_prepare_load_gvas()` 查询 GVA 并加 read lease。
- 加载线程 `KVCacheStoreLayerRecvingThread` 调用 `batch_copy(..., direction=1)`，即 G2L。
- 最后一层加载完成后释放 lease。

这条路径绕过普通 key-value get/put 的多 buffer 语义，更像“先分配远端 blob/GVA，再按层批量拷贝”。

## 7. hybrid / mamba / 压缩 cache 的协调

`coordinator.py` 负责 hybrid cache 场景下的命中与 mask 计算，尤其是多 KV cache group、sliding window、mamba、压缩 cache、Eagle 等组合。

核心思想：不同 group 的可命中长度可能不一样，最终只能使用所有 group 都一致可达的 prefix。

- `ExternalCachedBlockPool` 把外部 key 存在性伪装成 vLLM `BlockPool`。
- `AscendStoreCoordinator.find_longest_cache_hit()` 调用 vLLM 对应 cache manager 的 `find_longest_cache_hit()`，得到每个 group 的 mask 和最终 hit length。
- `store_mask()` 决定哪些 block 需要保存。
- `load_mask()` 决定哪些 block 可以加载。
- 对压缩 cache，外部 key 粒度使用 `block_size * compress_ratio`，但本地传输地址仍落在 cache-domain block 上。

可以把 coordinator 理解为“多 group cache 一致性裁判”。

## 8. TP mismatch 支持

当 prefill 和 decode 的 TP size 不一致，并且满足非 MLA、非 hybrid、KV heads 可整除等条件时，会启用 TP mismatch 路径。

普通 key 只按本 rank 的 `head_or_tp_rank` 存取；TP mismatch 下，一个本地 rank 可能需要拆成多个 effective rank 子 key：

- `_build_tp_mismatch_keys_and_addrs()` 为每个 chunk 生成多个 sub-key。
- `_build_strided_addrs()` 按 token 生成 strided 地址，只传本 rank 中某个 head slice。
- load/save 分别走 `_load_kv_tp_mismatch()` 和 `_store_kv_tp_mismatch()`。

该逻辑目前不支持 sparse、layerwise、hybrid。

## 9. 线程模型

`kv_transfer.py` 定义所有传输线程。

基础类 `KVTransferThread`：

- 每个线程有请求队列。
- run 时设置 NPU device。
- 统一维护 finished request 集合。
- 提供 `lookup()`、KV event 收集、batch copy 分包等公共方法。

主要子类：

| 类 | 用途 |
| --- | --- |
| `KVCacheStoreSendingThread` | 普通非 layerwise 异步保存。 |
| `KVCacheStoreRecvingThread` | 普通非 layerwise 异步加载。 |
| `KVCacheStoreKeyLayerSendingThread` | key-based layerwise 保存。 |
| `KVCacheStoreKeyLayerRecvingThread` | key-based layerwise 加载。 |
| `KVCacheStoreLayerSendingThread` | memcache GVA layerwise 保存。 |
| `KVCacheStoreLayerRecvingThread` | memcache GVA layerwise 加载。 |

线程之间主要靠 `threading.Event` 协调：load 完成事件、save 完成事件、NPU save event，以及 attention compute start gate。

## 10. 一条完整请求的简化链路

### 10.1 KV Pool 命中加载

```text
1. Scheduler 收到请求
2. KVPoolScheduler.get_num_new_matched_tokens()
   - 查询外部 KV Pool key 是否存在
   - 得到 kvpool_cached_tokens
   - 生成 LoadSpec
3. vLLM 为 external tokens 分配本地 KV blocks
4. KVPoolScheduler.update_state_after_alloc()
   - 记录 block ids
   - 标记 LoadSpec.can_load=True
5. KVPoolScheduler.build_connector_meta()
   - 生成 ReqMeta
6. Worker AscendStoreConnector.start_load_kv()
7. KVPoolWorker 根据 ReqMeta 生成 key/addrs/sizes
8. backend.get() 或 batch_copy(G2L) 把 KV 写回本地 HBM
9. forward 继续执行未命中的 token
```

### 10.2 KV Pool 保存

```text
1. Scheduler build_connector_meta() 为请求生成可 save 的 ReqMeta
2. Worker wait_for_save() 或 save_kv_layer()
3. 发送线程按 chunk 生成 key
4. 查询已存在 key，跳过已有 chunk
5. 根据 block id 算出本地 KV cache 地址和大小
6. 等待 NPU event，确保 KV 写入完成
7. backend.put() 或 batch_copy(L2G) 写入外部 KV Pool
8. 保存完成后通知 scheduler，允许释放相关 block
```

## 11. 关键配置开关

常见 extra_config：

- `backend`：`mooncake` / `memcache` / `yuanrong`。
- `use_layerwise`：启用逐层 load/save。
- `load_async`：普通非 layerwise 加载是否异步。
- `consumer_is_to_load`：consumer 角色是否允许 load。
- `consumer_is_to_put`：consumer 角色是否也写入 KV Pool。
- `discard_partial_chunks`：是否只传完整 chunk。
- `save_decode_cache`：是否保存 decode 阶段增量 KV。
- `prefill_tp_size` / `decode_tp_size`：用于推断 TP mismatch。
- `layerwise_prefetch_layers`：layerwise 首层可预取层数。
- `layerwise_max_transfer_blocks` / `layerwise_max_transfer_bytes`：限制 layerwise batch copy 分包大小。
- `h2d_stagger_us`：layerwise load 时按 TP/layer 做 H2D 提交错峰。

后端环境变量示例：

- Mooncake：`MOONCAKE_CONFIG_PATH`、`MOONCAKE_MASTER`、`MOONCAKE_GLOBAL_SEGMENT_SIZE`、`ASCEND_ENABLE_USE_FABRIC_MEM`。
- Memcache：`MMC_LOCAL_CONFIG_PATH`。
- Yuanrong：`DS_WORKER_ADDR`、`DS_ENABLE_EXCLUSIVE_CONNECTION`、`DS_ENABLE_REMOTE_H2D`。

## 12. 文件职责速查

| 文件 | 职责 |
| --- | --- |
| `ascend_store_connector.py` | 对接 vLLM KVConnectorBase_V1，区分 scheduler/worker 角色，聚合 KV events，启动 lookup server。 |
| `pool_scheduler.py` | scheduler 侧命中查询、LoadSpec 生成、ReqMeta 构建、请求生命周期和 delayed free 管理。 |
| `pool_worker.py` | worker 侧初始化、KV cache 注册、load/save 调度、layerwise 任务编排、TP mismatch、GVA 管理。 |
| `kv_transfer.py` | 实际传输线程、普通 get/put、layerwise key/GVA 传输、batch_copy 分包和完成状态管理。 |
| `config_data.py` | key 格式、token chunk 切分、地址/size 生成、请求 metadata 数据结构。 |
| `coordinator.py` | hybrid cache 命中、store/load mask、多 group 一致性、压缩 cache 粒度协调。 |
| `backend/backend.py` | 后端抽象接口。 |
| `backend/mooncake_backend.py` | Mooncake 后端实现。 |
| `backend/memcache_backend.py` | Memcache 后端实现。 |
| `backend/yuanrong_backend.py` | Yuanrong 后端实现。 |

## 13. 阅读建议

如果要继续深入，建议按这个顺序读：

1. `ascend_store_connector.py`：先看 vLLM 调用入口。
2. `pool_scheduler.py` 的 `get_num_new_matched_tokens()` 和 `build_connector_meta()`：理解 scheduler 何时决定 load/save。
3. `config_data.py` 的 `PoolKey`、`ChunkedTokenDatabase`、`ReqMeta`：理解 key 和地址是怎么生成的。
4. `pool_worker.py` 的 `register_kv_caches()`、`start_load_kv()`、`wait_for_save()`：理解 worker 侧主流程。
5. `kv_transfer.py` 的普通发送/接收线程：理解实际 put/get。
6. 如果关注 DSV4/hybrid，再看 `coordinator.py` 和 `pool_worker.py` 的 layerwise/GVA 分支。

一句话总结：`AscendStoreConnector` 负责接入 vLLM，`KVPoolScheduler` 负责“该不该传、传多少”，`KVPoolWorker` 负责“怎么传”，`ChunkedTokenDatabase` 负责“key 和地址怎么映射”，`Backend` 负责“最终写到哪个外部存储”。
