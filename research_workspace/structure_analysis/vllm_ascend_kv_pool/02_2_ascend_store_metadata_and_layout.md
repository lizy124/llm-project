# 02_2. Key、Metadata 与 Cache Layout 如何描述一次传输？

源码位置：

- `ascend_store/metadata.py`
- `ascend_store/layerwise_cache_layout.py`
- `ascend_store/attention_fence.py`

本文关注 `ascend_store` 的描述层。Scheduler 和 Worker 不能共享 Python request 或 NPU tensor 指针，因此需要一组稳定的数据结构，把“外部对象是谁、本地目标在哪里、传输覆盖哪些 token/layer/group”表达清楚。

---

## 1. Metadata 不是一个单独对象

源码中的 metadata 可以按生命周期分成四层：

```text
存储身份
  KeyMetadata -> PoolKey / LayerPoolKey

Scheduler 持久状态
  LoadSpec + RequestTracker

Scheduler -> Worker 请求描述
  ReqMeta -> AscendConnectorMetadata

Worker -> 传输线程任务
  LayerBlockRange
  SharedBlockData
  LayerTransferTask / LayerLoadTask
  LayerBatchReqMeta / LayerMultiBlockReqMeta
```

这几层不能混用。`RequestTracker` 跨调度轮次存在，`ReqMeta` 只描述一次 SchedulerOutput，`LayerTransferTask` 又只描述某一物理 layer/group 的执行单元。

---

## 2. 外部 key 如何唯一标识 KV？

### 2.1 KeyMetadata

`KeyMetadata` 保存 key 前缀所需的模型和并行维度：

```text
model_name
head_or_tp_rank
pcp_rank / dcp_rank / pp_rank
kv_cache_group_id
cache_role       # kv 或 state
cache_family     # default、c1、c2、mixed 等
```

`cache_role` 避免 KV 与 Mamba/state 使用相同 chunk hash 时冲突；`cache_family` 区分不同压缩比例或 cache spec。Hybrid 模型中只用 model name 和 hash 已不足以唯一定位对象。

### 2.2 PoolKey 与 LayerPoolKey

`PoolKey` 再附加 `chunk_hash`，字符串大致为：

```text
model@pcpN@dcpN@head_or_tp_rank:N@pp_rank:N
@group:N@cache_role:kv@cache_family:default@HASH
```

`LayerPoolKey` 用 `layer_id` 代替 PP 级聚合对象中的层隐含关系：

```text
model@pcpN@dcpN@head_or_tp_rank:N
@group:N@cache_role:kv@cache_family:default@layer_id:N@HASH
```

普通路径按 chunk 存一个包含多层地址的对象；key-based layerwise 路径则把同一 `PoolKey` 用 `split_layers()` 展开为每层独立 key。

MemCache GVA layerwise 使用另一套紧凑 key，由 Worker/Scheduler 的 `_make_layerwise_gva_key*()` 构造。它标识外部 GVA slot，而不是普通 backend object。

---

## 3. ChunkedTokenDatabase 解决什么问题？

`ChunkedTokenDatabase` 是 key 模型与物理地址模型之间的核心转换器。名字容易让人误以为它存储 token，实际上它保存的是：

```text
每个 group 的 KeyMetadata 与 block size
各 group/cache_role 的 base addresses
block length 与 block stride
每个物理 layer 包含哪些 cache tensor entry
cache family、layer count 和 coordinator
```

它提供三类能力。

### 3.1 token/chunk 迭代

`_iter_token_chunks()`、`process_tokens()` 和 `process_token_key_strings*()` 把 token 区间切成外部传输 chunk，同时生成 key，并把 block id 与 chunk 对齐。

处理时要同时尊重：

- `hash_block_size` 与 group block size 的比例；
- cache transfer granularity；
- load/store mask；
- null block 是否跳过；
- partial block 和保存起点。

### 3.2 block id 到地址

`prepare_value()` 根据 block ids、base address、block stride 和 group layer entries，构造 backend 所需的地址/长度列表。逻辑上可写成：

```text
address = tensor_base
        + local_block_id * block_stride
        + layer/cache-entry offset
```

真实实现还需要处理多个 tensor entry、不同 stride、group 和 cache role，因此返回的是地址与 size 数组，而非一个连续 pointer。

`prepare_value_layer()` 是逐层版本，只展开指定 `layer_id` 的 entry。

### 3.3 hybrid mask

当绑定 `AscendStoreCoordinator` 后，`store_mask()`、`load_mask()` 和 `mask_allows_chunk()` 把 group 可达性约束应用到 key/地址生成过程。控制面决定共同 token 范围，描述层进一步决定该范围内每个 group 实际需要哪些 chunk。

---

## 4. RequestTracker 是跨轮次账本

`RequestTracker` 只存在于 Scheduler 侧，生命周期通常覆盖多轮 schedule。关键字段包括：

```text
req_id
token_len
allocated_block_ids_by_group
num_saved_tokens
token_ids / num_prompt_tokens
block_gvas_by_group / gva_block_offset / last_block_gva
mamba_group_ids / num_speculative_blocks / block_sizes
```

其中：

- `token_len` 表示 tracker 已推进到的目标长度；
- `allocated_block_ids_by_group` 是当前累计本地 block 账本；
- `num_saved_tokens` 是增量 save 的水位线；
- GVA 字段保持 layerwise 外部 slot 跨调度轮次的关联；
- Mamba 字段用于更新 speculative block 被移动或替换为 null block 后的账本。

`update()` 追加新 block，但对 Mamba align group 会先调用 `update_mamba_spec_blocks()`，把已经失效或被复用的位置写为 block id 0。这里的 0 不是普通有效 block，而是后续地址生成时需要识别的 null block。

---

## 5. ReqMeta 是一次调度的传输意图

`ReqMeta.from_request_tracker()` 从持久 tracker 生成当前 SchedulerOutput 所需的快照。它需要同时表达 load 和 save：

```text
req_id
target_token_len
save_start_token / save_end_token
block_ids_by_group
block_hashes
can_save
load_spec
kv_cache_group_ids
is_last_chunk
```

还可能携带：

```text
event_id
current_event
partial_block_index / last_block_gva
save_keys / load_keys
block ids 和 GVA 的 NumPy 快照
partial_save_gva_per_group / partial_load_gva_per_group
```

### 5.1 为什么同时有三个 token 长度？

```text
target_token_len
  本轮 forward 完成后请求总长度

save_start_token
  tracker 上一次已保存到的位置

save_end_token（兼容字段 token_len_chunk）
  本轮完整 chunk 可保存到的位置
```

若 partial chunk 默认被丢弃，`save_end_token` 会向下对齐；若 GVA layerwise 允许保存 partial block，则额外通过 partial GVA 描述尾块。`num_saved_tokens` 只有在本轮确实安排 save 时才推进。

### 5.2 LoadSpec 为什么嵌在 ReqMeta 中？

同一个请求可能一边 load 已有前缀，一边在 forward 后 save 新产生的范围。因此 `ReqMeta` 不是 load task 或 save task，而是 Worker 对本轮请求的统一输入。Worker 再根据 `load_spec`、`can_save` 和运行模式拆分方向。

---

## 6. AscendConnectorMetadata 是跨侧信封

`AscendConnectorMetadata` 被放入 `SchedulerOutput.kv_connector_metadata`。它包含：

```text
requests             本轮 ReqMeta 列表
preempted_req_ids    需要丢弃旧异步结果的请求
loading_req_ids      仍处于远端 load 生命周期的请求
delayed_free_req_ids request 已结束但 send 尚未完成
```

它传的是可序列化的计划与状态，不携带 NPU pointer。Worker 在收到它后，结合本进程已经注册的 cache 地址才能生成执行任务。

反向的 `AscendStoreKVConnectorWorkerMetadata` 保存 completed event count，并支持多个 Worker metadata 聚合。它用于 Scheduler 确认分布式发送事件是否已被所有预期 Worker 完成。

---

## 7. Worker 任务 metadata 如何逐步细化？

### 7.1 LayerBlockRange

`LayerBlockRange` 把一个 `ReqMeta` 限定到某个 group 的 `[start_block, end_block)`。同一请求在不同 group 的 effective block size 不同，因此 block range 必须按 group 重新计算。

### 7.2 LayerTransferTask

它加入：

```text
physical layer id
layer index inside group
group id
若干 LayerBlockRange
shared block data
write-finish keys
```

这是 layerwise 发送/接收线程的主要任务单位。

### 7.3 SharedBlockData

GVA 路径中，key、GVA 和 block 范围对所有 layer 基本相同，只有本地 layer address 不同。`LayerBatchBuilder.build_shared()` 因此预先计算一次：

```text
request ids
save/load keys
GVA array
每个 request 的 block offset/count
```

随后每层只用 `build_addrs()` 补充本层 HBM 地址，避免为每一层重复遍历 hash 与 block。

### 7.4 LayerBatchReqMeta 与 LayerLoadTask

`LayerBatchReqMeta` 是已经展开好的批量数组：

```text
req_ids
layer_id
addr_array / size_array / gvas_array
save_keys / load_keys
```

`LayerLoadTask` 除传输任务外，还带 `wait_for_save_layer` 和 `AttentionComputeStartGate`。这说明 layerwise load 的输入不仅是数据地址，还包含复用 buffer 前必须满足的执行顺序。

---

## 8. Layerwise layout 如何描述物理 buffer 复用？

### 8.1 基本 layout

`build_layerwise_cache_layout()` 根据配置生成：

```text
num_shared_buffers
num_prefetch_layers
independent_layers
prefetch_layer_map
storage_indices
has_layer_reuse
```

independent layer 独占 buffer；其余 layer 按 slot 轮转共享。例如 8 层、2 个共享 buffer 时，可能形成：

```text
slot A: layer 1 -> 3 -> 5 -> 7
slot B: layer 2 -> 4 -> 6
```

`prefetch_layer_map[next] = previous_owner` 表示 load `next` 前必须确保 `previous_owner` 对该物理 buffer 的 save/attention 使用已经完成。

### 8.2 按 cache spec 分桶

`build_layerwise_reuse_layout()` 先把 layer name 映射成物理 layer，再按主 cache spec 分桶。只有形状和语义兼容的 layer 才共享 slot。

一个物理 layer 可能同时有 main cache 和 indexer cache。源码要求多 spec 情况必须正好是一主一 indexer，并对可复用 spec 做额外校验，防止不同布局被错误放进同一 buffer。

### 8.3 MTP 物理层编号

`get_layerwise_physical_layer_index()` 会识别普通 `layers.N` 和 `mtp.layers.N`。MTP layer 被放在 base model layers 之后，避免与主模型的同编号 layer 冲突。

---

## 9. AttentionComputeStartGate 属于 metadata 的原因

预取接收线程不能只根据 Python 已经调用到某处就提交传输，它需要等待 compute stream 真正到达 attention 边界。`AttentionComputeStartGate` 用 condition 保存一个 NPU event：

```text
模型执行侧在提交 attention op 前 record(stream)
  -> gate 持有 event
  -> transfer task 携带 gate
  -> 接收线程 wait()
  -> event synchronize 后确认 compute stream 已到达该边界
```

它被挂在 `LayerLoadTask` 上，是任务依赖的一部分；它表示“可以在 attention 边界启动相应传输”，不表示 attention op 已执行完，也不代表 backend load 或整个 request 已完成。

---

## 10. 描述层的不变量

阅读或修改 metadata 时应保持以下不变量：

```text
同一外部对象的 key 必须包含所有影响布局的维度；
token range 必须能整除或明确描述 partial chunk；
每个 group 的 block ids、block size、mask 和地址必须使用同一 group id；
RequestTracker 不应携带 Worker 的真实 pointer；
ReqMeta 是单轮快照，不能替代跨轮次 tracker；
layer event 与 request finished 不能互换；
共享 buffer 的 owner 关系必须与 attention/save 依赖一起传递。
```

下一篇 [02_3](02_3_ascend_store_worker_pipeline.md) 继续跟踪 Worker 如何把这些描述与真实 NPU tensor 结合，生成普通或 layerwise 设备任务。
