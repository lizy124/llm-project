# 04. KVPoolWorker 如何把 Metadata 变成设备任务？

源码位置：

- `ascend_store/pool_worker.py`
- `ascend_store/metadata.py`
- `ascend_store/layerwise_cache_layout.py`

本文关注 Worker 编排层。`KVPoolWorker` 一端接收 Scheduler 产生的 `ReqMeta`，另一端持有真实 NPU KV cache、backend 和传输线程；它的核心职责是把逻辑 block/key 计划落成可执行的地址任务。

---

## 1. KVPoolWorker 的职责边界

`KVPoolWorker` 不是单纯的 backend client，也不是 vLLM block allocator。它负责连接四种坐标：

```text
Scheduler 坐标：request、token range、block ids
模型坐标：layer name、cache group、K/V 或 state tensor
设备坐标：base address、block stride、NPU event
存储坐标：PoolKey、GVA、backend object
```

它完成的主要阶段是：

```text
初始化配置与 backend
  -> register_kv_caches()
  -> 建立 group/layer/address 映射
  -> register_buffer()
  -> 启动匹配当前模式的线程
  -> start_load_kv()
  -> 构造普通或 layerwise load/save 任务
  -> wait/save/get_finished
```

是否运行某个请求、分配哪些本地 block，仍由 Scheduler 和 `KVCacheManager` 决定。

---

## 2. 初始化阶段决定了哪些分支？

构造函数把初始化拆成若干方法，分别推导并行、传输、key 和 metadata 配置，再动态加载 backend。

### 2.1 并行与 key 维度

Worker 记录 TP、PP、PCP、DCP rank/size、KV heads、模型名和 `kv_role`。这些信息一方面决定 key，另一方面决定同一 KV 在 rank 之间如何切分或轮转提交。

`put_step` 用于 KV head 少于 TP rank 的 MLA 等场景：多个 TP rank 可能共享一个有效 head key，并非每个 rank 都独立发布一份对象。

### 2.2 cache group 与粒度

Worker 与 Scheduler 使用相同的：

```text
grouped_block_size
hash_block_size
lcm_block_size
cache_transfer_granularity
cache family
hybrid/Mamba/SWA group 信息
```

这是跨侧协议的一部分。若两侧对同一 group 的 block size 理解不同，即使 key 命中，也会把数据写进错误的本地范围。

### 2.3 运行模式

决定任务形态的核心开关是：

```text
use_layerwise
backend_name
use_gva_layerwise = use_layerwise and backend == memcache
load_async
consumer_is_to_put
TP mismatch
layerwise buffer reuse
```

其中 `load_async` 主要影响非 layerwise load；layerwise 模式本身已经通过每层线程与 event 异步推进。

---

## 3. register_kv_caches() 建立了什么映射？

模型初始化完成后，connector 把 `dict[layer_name, tensor or tuple[tensor]]` 交给 Worker。此时 Worker 第一次看到真实 cache。

### 3.1 推导 block 元数据

`_get_cache_block_metadata()` 根据 tensor 形状和 stride 得到：

```text
tensor_num_blocks
block_size_scale
block_len       一个外部 block 的有效字节数
block_stride    相邻 block 首地址间隔
region_len      backend 需要注册的完整内存范围
```

`block_len` 与 `block_stride` 不一定相等。非连续 layout 中，不能用 `base + block_id * block_len` 代替真实 stride。

### 3.2 去重注册存储区间

多个 tensor view 可能共享同一 underlying storage。Worker 通过 storage pointer 聚合区间，只向 backend 注册合并后的 `[start, end)`，避免重复注册同一片 NPU 内存。

Hybrid 场景还会把共享 tensor 的起始地址向下对齐到 2 MB 边界，同时断言不会越过原始 storage。这是 backend 内存注册约束，不会改变逻辑 block id。

### 3.3 建立 group 与 layer entry

`_infer_cache_group_metadata()` 按物理 layer 排序，将一个 group 中每层的一个或多个 cache tensor 展开为：

```text
group_kv_caches_base_addr[group]
group_block_len[group]
group_block_stride[group]
group_layer_cache_entry_offsets[group]
group_num_layers[group]
```

`group_layer_cache_entry_offsets` 很关键：一个物理 layer 可能有 K/V 两个 tensor，也可能有 main/indexer 等多个 entry。逐层地址生成不能假设“一层就是一个 pointer”。

这些映射随后写入 `ChunkedTokenDatabase.set_group_buffers()`。

### 3.4 注册完成才启动线程

Worker 按以下顺序完成设备数据面初始化：

```text
必要时 backend.ensure_initialized()
  -> backend.register_buffer(ptrs, lengths)
  -> _start_kv_transfer_threads()
```

线程启动发生在 cache 注册之后，因为它们持有 `ChunkedTokenDatabase`，而后者必须先具备真实地址和 stride。

---

## 4. Worker 如何选择传输线程？

`_start_kv_transfer_threads()` 的分支可归纳为：

| 模式 | Save 线程 | Load 线程 |
| --- | --- | --- |
| 非 layerwise | `KVCacheStoreSendingThread` | 仅 `load_async` 时创建 `KVCacheStoreRecvingThread` |
| key-based layerwise | `KVCacheStoreKeyLayerSendingThread` | `KVCacheStoreKeyLayerRecvingThread` |
| MemCache GVA layerwise | `KVCacheStoreLayerSendingThread` | `KVCacheStoreLayerRecvingThread` |

Save 线程只有 producer、both 或开启 `consumer_is_to_put` 时存在。非 layerwise 同步 load 不创建接收线程，而由 `start_load_kv()` 直接调用 backend。

Layerwise 模式还创建：

```text
layer_load_finished_events[layer]
layer_save_finished_events[layer]
sync_save_events[layer]          # torch.npu.Event
get_event
```

这些 event 的粒度是物理 layer，不是 request。

---

## 5. 非 layerwise load 如何执行？

`start_load_kv()` 遍历 metadata 中有可用 `LoadSpec` 的请求，先确定实际 load token 长度，再按 group 生成 key/address/size。

### 5.1 token 长度修正

当外部命中到 prompt 最后一个 token 之前，源码可能把 `kvpool_cached_tokens` 加一，使地址范围与 request 的最后边界一致。这是 vLLM 对最后 token 计算语义与完整 block 传输之间的适配，不能简单理解为“多加载一个远端 token”。

### 5.2 按 group 生成数组

每个 group 使用自己的 block ids、block size 和 load mask：

```text
process_token_key_strings_with_block_ids()
  -> start/end/key/block_id
prepare_value()
  -> address list/size list
```

Mamba align group 会跳过 null block；coordinator mask 会过滤当前恢复范围中不可达的 chunk。

### 5.3 同步与异步分流

```text
load_async = false
  -> 循环移位 key/address/size
  -> backend.get()
  -> 记录失败 block ids

load_async = true
  -> ReqMeta 入 KVCacheStoreRecvingThread
  -> 后台生成数组并 get
  -> 完成后报告 request id
```

循环移位按 TP rank 改变批量数组起点，用来错开多个 rank 同时访问相同 backend 顺序造成的热点；它不改变 key 与对应地址的配对。

---

## 6. 非 layerwise save 如何执行？

非 layerwise 的 `save_kv_layer()` 本身不会逐层提交；connector 在该模式下主要通过 `wait_for_save()` 进入 Worker 保存路径。

Worker 为需要 save 的 `ReqMeta` 记录当前 NPU event，并把请求交给 `KVCacheStoreSendingThread`。发送线程等待 event 后再读取 cache，避免 forward 尚未写完目标 block 就开始发布。

请求可能只保存 `[save_start_token, save_end_token)` 的增量区间。已有外部前缀、非完整 chunk、SWA/Mamba mask 和 `can_save` 都会缩小实际 key 集合。

发送完成后，request-level finished 状态用于解除 Scheduler 的 delayed-free，而不是表示模型请求刚刚结束。

---

## 7. Layerwise 任务如何形成？

启用 layerwise 后，`start_load_kv()` 不直接调用 backend，而是调用 `process_layer_data()`。

### 7.1 每轮使用新任务列表

Worker 为每个 step 重建：

```text
layer_save_tasks = [[] for each physical layer]
layer_load_tasks = [[] for each physical layer]
```

线程会持有列表引用并在结束时 clear。如果复用上一轮列表，迟到的 clear 可能删除新 step 的任务，导致复用 buffer 保留旧数据。因此源码明确为每轮建立新对象。

### 7.2 按 physical layer 和 group 展开

`physical_layer_to_group_layers` 把物理 layer 映射为若干 `(group_id, layer_idx_in_group)`。`process_layer_data()` 对每个组合分别调用：

```text
_process_save_for_layer_batch()
_process_load_for_layer_batch()
```

两者根据 group effective block size 形成 `LayerBlockRange` 和 `LayerTransferTask`。一个 physical layer 可对应多个 group task，不能只用全局 `current_layer` 推导 group 内索引。

### 7.3 提前构造 shared data

GVA 发送/接收路径中，`_build_shared_save_data()` 和 `_build_shared_load_data()` 按 group 只计算一次 key/GVA/block offsets，再把同一个 `SharedBlockData` 挂到该 group 的所有 layer task。

key-based layerwise save 也会缓存 `process_tokens()` 结果，避免每层重复展开相同 token chunk。

---

## 8. Layerwise load/save 如何与 forward 交错？

### 8.1 提交 load

`_submit_ready_layer_loads()` 按 `next_layer_to_submit` 和预取窗口提交 `LayerLoadTask`。若当前 layer 复用了旧 layer 的物理 slot，任务会携带：

- `wait_for_save_layer`：先等旧 owner 的 save；
- `attention_start_gate`：先等 compute stream 到达当前 attention 边界，再提交预取传输；
- external slot release waiter：GVA slot 可复用前等待外部拥有者释放。

### 8.2 等待当前 layer

`wait_for_layer_load()` 等待当前 layer 的 load event，并在循环中检查接收和必要时发送线程的异常。超时只打印并继续等，不会被当作成功。

完成后 Worker 推进 `current_layer`，并补交后续预取任务。

### 8.3 提交 save

`save_kv_layer()` 为当前 layer 记录 `sync_save_event`，再把该层的 task 列表交给发送线程。模型可以继续后续 layer，而线程在读取本层 HBM 前同步对应 NPU event。

connector 的 `wait_for_save()` 在 layerwise 模式下直接返回。逐层保存由 `save_kv_layer()` 推进；到 final layer 时，该方法会等待最后一层 save event，并检查发送线程失败。共享 slot 能否复用仍由各 layer event、NPU event 和预取依赖控制。

---

## 9. GVA layerwise 与 key-based layerwise 的本质差异

### 9.1 key-based

每个 layer/chunk 直接以 `LayerPoolKey` 访问 backend：

```text
LayerPoolKey
  -> prepare_value_layer()
  -> backend.put/get(key, local address, size)
```

外部 backend 自己管理对象存储与传输。

### 9.2 GVA-based

Worker 先为 key 查询或分配外部 GVA：

```text
save: key -> batch_alloc -> GVA
load: key -> batch_get_key_info -> GVA
```

随后传输线程执行批量 copy：

```text
save: local HBM address -> GVA
load: GVA -> local HBM address
```

保存结束还需 `batch_write_finish()` 发布结果；lease 与外部 slot 生命周期也只出现在这条路径。这里的 GVA 是外部存储地址，不是本地 vLLM block id。

---

## 10. Partial block 与 TP mismatch

### 10.1 Partial block

普通对象路径通常只保存完整 transfer chunk。共享 layerwise buffer 场景若最后一块尚未完整，却即将被下一 layer/request 覆盖，就需要为 partial save/load 单独分配或查询 GVA，并记录：

```text
partial_block_index
last_block_gva
partial_save_gva_per_group
partial_load_gva_per_group
```

partial key 与完整 hash key 不同，因为不完整内容不能冒充可复用的完整 prefix block。

### 10.2 TP mismatch

prefill 与 decode 的 TP size 不一致时，一个本地 rank 的 KV heads 可能对应多个远端 sub-key。Worker 通过 `_build_strided_addrs()` 和 `_build_tp_mismatch_keys_and_addrs()` 将每个 block 按 head slice 展开：

```text
one logical block
  -> sub-key 0 + strided addresses
  -> sub-key 1 + strided addresses
  -> ...
```

该路径要求非 MLA、非 hybrid、head 数可被有效 TP size 整除。条件不成立时不会强行启用，避免用错误的连续内存假设重排 KV heads。

---

## 11. 完成与错误如何离开 Worker？

`get_finished()` 汇总发送和接收线程的 request-level 完成集合，并结合本轮 metadata 过滤：

```text
preempted_req_ids
loading_req_ids
delayed_free_req_ids
finished_req_ids
```

接收失败的本地 block id 通过 `get_block_ids_with_load_errors()` 暴露。当前单 group 路径可以把这些 block 交给上层做 invalid-block 回退；hybrid load 失败不能简单把多 group block id 塞回单 group 回退协议，因此源码会记录错误而避免 Scheduler 崩溃。

Worker 还通过 `build_connector_worker_meta()` 返回 completed sending event count，供 Scheduler 等待所有 rank 后释放 block。

---

## 12. 阅读 Worker 的检查表

```text
1. cache registration 是否使用真实 block stride，而非假设连续？
2. layer name 是否映射到正确 physical layer 和 group 内索引？
3. 当前模式实际创建了哪一种发送/接收线程？
4. load 是同步直调 backend，还是异步进入线程？
5. task 的 token range 是否已经过 group mask 和 save watermark？
6. 共享 buffer 的上一 owner、save event、attention gate 是否齐全？
7. GVA 与 local address 的方向是否正确？
8. partial block 和 TP mismatch 是否进入了专用展开路径？
```

下一篇 [05](05_transfer_backend_storage.md) 深入存储模型、线程公共协议、六种派生线程以及三种 backend 的能力差异。
