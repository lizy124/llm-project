# 03. Metadata 与 Layout：身份、范围、地址和依赖

基线与源码入口见 [00](../00_source_baseline.md)。核心文件是 `metadata.py`、`layerwise_cache_layout.py`，执行期数组构造在 `kv_transfer.py::LayerBatchBuilder`。

## 1. 先区分五个生命周期

| 描述 | 代表类型 | 谁持有 | 生命周期 |
|---|---|---|---|
| 外部身份 | `KeyMetadata`、`PoolKey`、backend key builder | 两侧分别推导 | 模型/布局配置 |
| 请求账本 | `LoadSpec`、`RequestTracker` | Scheduler | 跨 schedule |
| 下行快照 | `ReqMeta`、`AscendConnectorMetadata` | Scheduler→Worker | 一轮计划 |
| 设备任务 | `LayerBlockRange`、`LayerTransferTask`、`LayerLoadTask` | Worker→线程 | 一层/一个批次 |
| 批量参数 | `SharedBlockData`、`LayerBatchReqMeta`、`LayerRangeReqMeta` | Builder/线程 | 当前任务执行 |

上行 `AscendStoreKVConnectorWorkerMetadata.completed_events` 是另一条完成计数通道，不能替代 load/save 计划。

## 2. Key 是外部数据布局协议

### 2.1 普通 PoolKey

`KeyMetadata` 包含 model、head/TP、PCP/DCP/PP、group、cache role、cache family；`PoolKey` 再附加 chunk hash。普通 key 大致为：

```text
model@pcp:N@dcp:N@head_or_tp_rank:N@pp_rank:N
@group:N@cache_role:kv@cache_family:default@HASH
```

`cache_role` 区分 KV/state；family 区分压缩布局。`LayerPoolKey` 增加 layer id，用于 KeyLayer 分支；它不是所有 layerwise backend 的通用格式。

### 2.2 MemCache 与 Mooncake 的专用 builder

`backend/__init__.py` 加载协议模块；`bind_layerwise_keys()` 返回统一的 `LayerwiseKeyBuilder`。Scheduler 和 Worker 绑定相同身份，查询和实际读写必须落在同一 namespace。

| 协议 | 身份重点 |
|---|---|
| MemCache GVA | 模型、group（多组时）、block hash、head/TP、PP |
| Mooncake 单 group、PP=1 | `model@hash@head`，保留对应协议格式 |
| Mooncake PP | `mooncake_pp_v3` namespace + PP stage + hash/head |
| Mooncake hybrid | `mooncake_hybrid_v1` + layout hash + group/block size + hash/head；PP 时增加 stage |

Mooncake namespace 的关键设计：
- PP 使用真实 layer partition，而非假定等分。
- Scheduler 可能已把 cache block size 改为最小 group block size，Worker 还保留 CLI 值；身份计算必须从共享的已解析 group 归一化。
- `UniformTypeKVCacheSpecs` 与 merged spec 是同一布局的不同表示，不能直接序列化整个 spec 对象做 hash。
- PP Worker 的 layer membership 是 stage-local，不能拿它与 Scheduler 的全局 membership 各自 hash。
- TP、group page size、模型/cache/speculative 配置等影响兼容性的维度要纳入协议指定身份。

结果是：key 命中率为零时，先比两侧实际 key，再查网络。两个进程“配置看起来相同”不证明生成 key 相同。

## 3. 四种粒度

| 粒度 | 描述 |
|---|---|
| `hash_block_size` | `request.block_hashes` 的粒度 |
| `grouped_block_size[group]` | 当前 group 的逻辑 token page |
| `cache_transfer_granularity` / LCM | cacheable groups 可共同恢复的范围 |
| kernel page / block stride | 真实 tensor 页形状与字节间距 |

`get_block_hashes()` 在 group page 大于 hash page 时选择对应聚合边界；不要假设 hash list 的第 i 项就是所有 group 的第 i 个 block。

推导例：hash=16 tokens，group page=64 tokens，逻辑 page 对应 4 个 hash 页。token 水位 128 对该 group 是 2 页，对另一个 16-token group 则是 8 页。

私有 state 通过 `infer_cacheable_group_ids()` 与公共 prefix 存储分离。不能为了凑齐所有 group 而发布不能独立复用的内部状态。

## 4. ChunkedTokenDatabase：描述转换器

它不保存 token 数据库实体，而保存 group block size、key 前缀、缓存 family、base addresses、block length/stride、每层 entry 范围和可达性 coordinator。

主要能力：
- `_iter_token_chunks` / `process_tokens`：将 token range 切 chunk 并生成 key。
- `process_token_key_strings_with_block_ids`：将 hash、逻辑 range 与本地 block id 对齐。
- `prepare_value` / `prepare_value_layer`：将逻辑块转成实际地址和大小。
- `store_mask` / `load_mask` / `mask_allows_chunk`：应用 group 可达性约束。

```text
local_addr(entry, block_id)
  = tensor_base(entry) + block_id * block_stride(entry)

remote_offset(layer, entry)
  = preceding_layers_payload_bytes + preceding_entries_bytes
```

两个公式的坐标系不同：前者是本地 allocation，后者是外部聚合对象。`block_len` 是有效负载字节数，`block_stride` 是本地相邻页间距，二者不保证相等。

## 5. RequestTracker 与 ReqMeta

`RequestTracker` 持久保存累计 block ids、token length、`num_saved_tokens`、prompt/event token 信息、Mamba speculative 位置、GVA 与 partial 关联。Mamba null block 以 0 表示，不能当有效 KV 页传输。

`ReqMeta.from_request_tracker()` 形成单轮快照：

| 字段 | 含义 |
|---|---|
| `target_token_len` | 本轮计算目标长度 |
| `save_start_token` / `save_end_token` | 本轮安排保存的范围 |
| `token_len_chunk` | save end 的兼容访问形式 |
| `block_ids_by_group` | 当前本地页账本 |
| `block_hashes` | 可生成外部完整 key 的身份材料 |
| `load_spec` / `can_save` | 两个方向独立门控 |
| `save_keys` / `load_keys` | GVA/会话相关 key |
| `save_block_keys` / `load_block_keys` | block-key 的逐 slot 身份 |
| `store_masks` / `load_masks` | hybrid 稀疏可达范围 |
| partial/GVA 字段 | 未完整页、旧快照与新的保存目标 |

tracker 水位表示已安排保存，不等于后端成功；非 layerwise 线程另记成功 offset，失败范围可在后续 step 重试。

Scheduler 生成的 metadata 不包含真实 NPU 地址；到 Worker 后会给 ReqMeta 附加本地 event、NumPy 快照等执行期字段。不要把“下行可序列化”和“该类所有运行时字段都跨进程”混淆。

### 5.1 完整 hash、加载与保存不能机械叠加

`from_request_tracker()` 不只按 target length 向下取整，还检查实际可用 block hashes。speculative 调度可能跨过页边界，却尚未有已提交的 request hash；这时不能把尚未发送的页记成已保存。

普通模式中，若 `load_spec.can_load=True` 且不需要 reuse partial snapshot，函数强制 skip save，避免同一批刚从池读取的 KV 又被同时安排回写。后续 cached-request step 再保存增量。reuse 的 partial 保存有不同需求，不能把该互斥规则写成对所有模式绝对成立。

推导例：transfer granularity=64，目标长度=130，有两个完整 hash 页。普通保存最多到 128；若需要物理 reuse，则还要保护 request-scoped 尾部快照，不能把 128–130 两个 token 发布成一个完整的 64-token prefix 对象。

### 5.2 Snapshot 身份与逻辑长度分离

speculative rejection 会回退逻辑 token 数。仅用回退后的长度寻找前一轮 snapshot，可能读错旧 key。LoadSpec 因此额外保留：
- `layerwise_snapshot_end_token`：实际保存过的 snapshot 身份。
- `layerwise_snapshot_block_index` / `layerwise_snapshot_start_block`：含未 hash 页的保存范围。
- `layerwise_previous_step`：区分本请求前一轮状态与其他请求的公共 prefix。

Scheduler 先捕获 previous snapshot，再建立本轮 save metadata；Worker 先保护旧 load，再分配新 snapshot。逻辑水位、snapshot 身份与公共 block hash 是三个不同概念。

### 5.3 保存 job 身份与 request id 分离

`ReqMeta.store_job_id` 标识一次具体排队的保存操作。一个 request 可有多个 job，同一个 request id 还可能被后续 generation 复用；只按 request id 接收异步回调会让旧 job 更新新请求的成功水位。

普通发送线程用 job 集合判断是否仍 live，只有有效 job 才能更新保存进度。released fence 必须等指定请求的所有 live jobs，而不是看到一次完成回调就放行。KV event token ids 同样使用稳定、有界快照，避免异步消费者读取 Scheduler 仍在增长的完整 token 列表。

## 6. AscendConnectorMetadata 的三个集合

```text
requests            当前 ReqMeta 列表
preempted_req_ids    抢占导致旧请求执行状态失效
loading_req_ids      bulk async load 正在等待的请求
released_req_ids    finished ∪ preempted；覆盖前需 fence 相关旧保存
```

当前无 `delayed_free_req_ids`。`released_req_ids` 表示 block 账本已允许复用，而 Worker 还必须在物理写入前完成保护。

## 7. 从 range 到批量数组

`LayerBlockRange` 表示 group 内 `[start_block, end_block)`，可额外带 partial block。`LayerTransferTask` 再加：
- physical layer 与 group 内 layer index。
- `use_key_major_ranges`：使用 block-key range 表达。
- `final_group_layer`：该 group 的最后实际层。
- `write_finish_keys`：GVA 到最后相关 copy 才发布。
- `shared_block_data`：同组跨层复用的逻辑行。

group 的末层不一定是模型末层；对象提交与整轮 drain 必须分别处理。

### 7.1 SharedBlockData

跨层不变的 block ids、GVA 或 block keys、request ids、load keys 只计算一次。每层 Builder 再补本地地址与外部 offset。

它仅可在当前 step 的相关层之间共享，不能跨 step 盲目保留；allocation、partial snapshot、session owner 都会变化。

### 7.2 两种批量结果

| 类型 | 数据面 | 主要字段 |
|---|---|---|
| `LayerBatchReqMeta` | GVA | 平铺 `addr_array / size_array / gvas_array` |
| `LayerRangeReqMeta` | block-key | keys、每 key 的 buffers/sizes/offsets、block ids |

range 模式下一个对象的每层 offset 由前面所有 entry payload 的累加得到；同一 key 的多段 source/size/offset 必须逐项配对。

`LayerLoadTask` 还携带 `wait_for_save_layer` 和 `attention_start_gate`：任务描述不仅包括数据，还包括允许覆盖 buffer 的依赖。

## 8. 物理 buffer 复用

`get_layerwise_reuse_config()` 通过协议的 `extract_layout_config()` 决定是否加入物理复用。当前 MemCache 可提供此配置；Mooncake block-key 返回 None。

通用 layout 字段：
- `independent_layers`：独占 buffer，默认 `[0]`，支持 `"all"` 和负索引。
- `storage_indices`：每个 slot 的 layer owners。
- `prefetch_layer_map[next] = previous_owner`。
- `num_shared_buffers` / `num_prefetch_layers`。
- `has_layer_reuse`。

默认 shared buffer 数等于层数，因此默认不发生复用。真正复用还要按兼容 cache spec 分桶；不同 spec/stride 不允许随意 alias。`apply_layerwise_kv_cache_plan()` 校验 descriptor 覆盖、边界和连续页要求，再调整物理计划。

推导例：61 层，其中 layer 0 独立，其余同 spec；8 个共享 slot 可承载其余 60 层。只按等大 KV buffer 计，61 份变为约 9 份；这不是整卡显存缩减比例，权重、workspace、通信和其他 cache 仍占内存。

## 9. 应保持的不变量

1. key 身份在 Scheduler/Worker/PP stage 间一致。
2. 每个 token range 使用对应 group 的 hash/page 粒度。
3. block id、GVA、key、size、offset 数组同步过滤/重排。
4. partial snapshot 不能冒充完整 hash 对象，也不能 full/partial 重复传输。
5. group final layer、model final layer、request last chunk 是不同边界。
6. reuse owner、save event、attention gate 与任务一起传递。

下一篇：[04 Worker Pipeline](04_worker_pipeline.md)。
