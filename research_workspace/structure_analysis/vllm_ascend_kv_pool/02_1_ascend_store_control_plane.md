# 02_1. Connector、Scheduler 与 Coordinator 如何形成控制面？

源码位置：

- `ascend_store/ascend_store_connector.py`
- `ascend_store/pool_scheduler.py`
- `ascend_store/coordinator.py`

本文关注 `ascend_store` 的控制面：它如何接入 vLLM connector 生命周期，如何查询外部命中，如何把命中长度与本地 block allocation 连接起来，以及 hybrid cache group 为什么需要独立的 coordinator。

---

## 1. 控制面的边界

`AscendStoreConnector` 实现 vLLM `KVConnectorBase_V1`，但它本身不是命中算法或数据传输实现。构造时根据 `KVConnectorRole` 只建立一侧对象：

```text
KVConnectorRole.SCHEDULER
  -> KVPoolScheduler

KVConnectorRole.WORKER
  -> KVPoolWorker
  -> 非 layerwise 且 rank 0 时可启动 LookupKeyServer
```

Scheduler 侧拥有 `Request`、`KVCacheBlocks`、`SchedulerOutput` 和本地 block pool；Worker 侧拥有真实 tensor、NPU 地址、stream 和 backend client。connector 负责保持 vLLM 接口不变，并把调用转发给拥有相应资源的一侧。

Scheduler 侧接口形成一条完整控制链：

```text
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
request_finished()/request_finished_all_groups()
update_connector_output()
```

这里没有任何 KV tensor 搬运。

---

## 2. KVPoolScheduler 初始化了哪些语义？

`KVPoolScheduler.__init__()` 不只是保存配置。它提前确定后续命中与 metadata 的基本坐标系：

```text
模型与并行维度
  model_name、TP、PP、PCP、DCP、kv_role

cache group 维度
  use_hybrid、group ids、cache family、Mamba/SWA group

粒度维度
  original_block_size
  grouped_block_size = original_block_size * PCP * DCP
  hash_block_size
  lcm_block_size
  cache_transfer_granularity

运行模式
  load_async、use_layerwise、use_gva_layerwise
  consumer_is_to_load、consumer_is_to_put、save_decode_cache
```

其中最容易混淆的是三种 block size：

```text
hash_block_size
  request.block_hashes 的粒度

grouped_block_size[group]
  某 cache group 的有效 token block 粒度

cache_transfer_granularity
  所有参与 group 都能安全传输的公共粒度
```

`cache_transfer_granularity` 由 group block size 与 LCM 共同推导。外部命中最终必须向下对齐到这个粒度，否则 Scheduler 可能认为一段无法完整恢复的 KV 已经可用。

Scheduler 同时创建 backend 的 scheduler client。它只需要 lookup 能力，不注册 NPU buffer，也不执行 `put/get` 数据搬运。

---

## 3. 外部命中查询有哪几条路径？

### 3.1 普通 key 路径

`_generate_store_query_keys()` 从 block hash 展开完整存储 key。一个 hash 可能对应多个并行维度：

```text
block hash
  x PCP rank
  x DCP rank
  x head_or_tp_rank
  x PP rank（非 layerwise 查询时覆盖所有 PP）
```

`_get_store_lookup_hit_tokens()` 调用 backend `exists()`，并按 block 检查这一组 key 是否全部存在。命中必须是从查询起点开始的连续前缀：某一 block 出现明确 miss 后停止，异常状态则作为错误处理。

非 layerwise 模式可以从 `num_computed_tokens` 之后开始查；layerwise 模式必须从 block 0 查询，因为远端逐层对象与本地 prefix cache 的覆盖范围不一定一致。

### 3.2 MemCache GVA 路径

`use_layerwise=True` 且 backend 为 MemCache 时，Scheduler 不再用普通 layer key 判断命中，而通过 `_make_layerwise_gva_keys_for_hit_check()` 和 `batch_get_key_info()` 查询已分配的 GVA。

每个 group、每个 block、每个有效 head/TP rank 都必须返回有效 size，当前 block 才算命中。不同 group 得到的命中长度最终取最小值：

```text
group 0 hit tokens = 4096
group 1 hit tokens = 3072
最终可恢复范围     = 3072
```

这是因为 forward 需要同时满足所有参与 cache group，而不是任选一个 group 命中即可。

### 3.3 LookupKeyServer/Client 旁路

非 layerwise 的 rank 0 Worker 可以启动 `LookupKeyServer`，Scheduler 侧 `LookupKeyClient` 通过 ZMQ 请求 Worker 执行 lookup。这条路径用于避免 Scheduler 直接依赖某些 Worker backend 初始化状态。

它仍然只返回命中长度或命中位置，不传输 tensor。真正 load 仍发生在 Worker 数据面。

---

## 4. Coordinator 为什么独立存在？

单一 Full Attention cache 的命中通常是连续 prefix，直接检查 block key 即可。Hybrid KV cache 中，不同 group 可能对应：

- Full Attention；
- Sliding Window Attention；
- Mamba/线性注意力状态；
- Eagle speculative cache；
- 不同 compress ratio 和 block size。

这些 group 的“某 token 范围是否可恢复”并不等价。`AscendStoreCoordinator` 把外部存在性伪装成 vLLM 可理解的 `BlockPool`，复用各 `SingleTypeKVCacheManager` 的可达性算法。

### 4.1 ExternalCachedBlockPool

`ExternalCachedBlockPool` 是一个 duck-typed block pool。它不管理真实 block，只把 `(group_id, block_hash)` 是否存在映射为“present block”或 cache miss。

这样 coordinator 可以调用 vLLM 已有的 `_find_longest_cache_hit()` 语义，而不在 AscendStore 中重新实现 Full/SWA/Mamba 的命中规则。

### 4.2 find_longest_cache_hit()

coordinator 先按有效 cache spec 对 group 分组，再分别调用对应 manager 的命中算法。多个 attention group 会迭代收敛到共同可恢复的 `hit_length`。

Eagle group 需要按规则丢弃或验证额外 speculative block；简单 hybrid 和一般 hybrid 的收敛过程也不同。最终返回：

```text
masks: 每个 cache group 哪些逻辑 block 可用
hit_length: 所有参与 group 共同支持的最长 token 范围
```

### 4.3 load/store/lookup mask

同一个 token 范围在不同操作下使用不同 mask：

```text
load_mask()   当前恢复范围中哪些 group block 应写入本地 cache
store_mask()  哪些状态对后续 token 仍可达，值得存到外部
lookup_mask() lookup 时哪些 chunk 必须参与存在性检查
```

例如 SWA 的旧 block 或 Mamba 不再可达的状态，不应仅因为 hash 存在就被当成有效恢复材料。

---

## 5. get_num_new_matched_tokens() 的结果是什么？

这个接口返回的不是“外部一共存了多少 token”，而是“在当前请求状态和本地命中基础上，还能交给 vLLM allocation 的外部 token 数”。核心过程是：

```text
request.block_hashes
  -> 查询外部连续命中
  -> hybrid coordinator 求共同可达范围
  -> 处理 layerwise/Eagle/最后一个 token 等规则
  -> 按 cache_transfer_granularity 对齐
  -> 扣除 num_computed_tokens
  -> 记录 LoadSpec
```

`LoadSpec` 同时保存：

- `vllm_cached_tokens`：本地已经计算或命中的 token；
- `kvpool_cached_tokens`：外部 pool 支持恢复到的 token；
- `can_load`：Scheduler 是否允许本轮真正 load；
- `kvpool_store_skip_tokens`：保存时应跳过的已有远端前缀。

异步 load 模式下，接口还会告诉 vLLM 需要 allocation，但请求可能先进入等待远端 KV 的状态；layerwise 模式则不沿用完全相同的异步等待语义。

---

## 6. update_state_after_alloc() 为什么关键？

命中查询时还没有目标 block 地址。只有 `KVCacheManager.allocate_slots()` 完成后，`update_state_after_alloc()` 才能同时看到：

```text
Request
外部命中 token 数
KVCacheBlocks（按 group 组织）
本地 num_computed_tokens
新请求、运行中或恢复后的状态
```

它把这些信息写入 Scheduler 侧状态：

- `_unfinished_requests` 保存 request 与已分配 block ids；
- `RequestTracker` 持续记录 token 长度、各 group block ids、已保存 token 和 GVA；
- `_loading_req_ids` 标记仍在等待异步 load 的请求；
- `LoadSpec.can_load` 根据 allocation 结果决定；
- Mamba speculative block、SWA clip 和 layerwise GVA offset 在此后具备计算条件。

因此该方法的准确含义是“提交分配结果”，不是“开始传输”。

---

## 7. build_connector_meta() 如何处理不同请求？

`build_connector_meta()` 先处理 finished/preempted 集合，再构造本轮 `AscendConnectorMetadata`。请求按来源进入四条路径：

```text
scheduled_new_reqs
  -> _process_new_request()

曾被抢占、现在重新获得 block 的 cached request
  -> _process_preempted_cached_request()

持续运行并新增 token/block 的 cached request
  -> _process_running_cached_request()

没有参与本轮 forward、但外部 load 仍需执行的请求
  -> _process_async_load_request()
```

四条路径最终都尽量归一成 `ReqMeta`，但 tracker 的重建或增量更新方式不同：

- 新请求从当前 allocation 建立 tracker；
- 抢占恢复请求丢弃旧 block 账本后重建；
- 运行中请求追加 token ids 和新 block；
- 异步 load 请求以外部命中范围建立只 load 的 metadata。

`ReqMeta.from_request_tracker()` 还会推进 `num_saved_tokens`，确保已保存区间不会被每轮重复发布。

---

## 8. 请求结束、发送完成和释放

非 layerwise save 可能在 request 已结束后仍读取本地 block。`request_finished()` 因此可能返回 `delay_free_blocks=True`，并把 request id 放入 `_delayed_free_req_ids`。

```text
request finished
  -> 本地 block 仍被后台 send 使用
  -> 暂不归还 BlockPool
  -> Worker 报告 finished_sending
  -> update_finished_sending()
  -> 清除 delayed-free 状态
```

Hybrid HMA 路径通过 `request_finished_all_groups()` 处理各 group block；SWA block 还可能先裁剪。Layerwise 路径当前不使用同一 sending event 延迟释放协议，因此直接释放，buffer 复用安全由 layer event、NPU event 和 slot release 机制保证。

`update_connector_output()` 还会聚合各 Worker 的 completed event count。达到预期 Worker 数后，Scheduler 才从绑定的 `BlockPool` 释放对应 block，防止分布式 rank 尚未完成时提前复用。

---

## 9. 阅读控制面的检查表

定位命中或调度问题时，依次确认：

```text
1. 当前走普通 key、LookupKey 还是 GVA lookup？
2. hash_block_size、group block size、transfer granularity 是否一致？
3. 所有 TP/PP/PCP/DCP key 是否都命中？
4. hybrid coordinator 的共同 hit_length 和 mask 是什么？
5. LoadSpec 是否允许 load，allocation 是否成功？
6. RequestTracker 是新建、增量更新还是抢占后重建？
7. 请求结束后 block 是否仍处于 delayed-free？
```

下一篇 [02_2](02_2_ascend_store_metadata_and_layout.md) 继续解释这些状态如何被编码成 key、request metadata、layer task 和地址布局。
