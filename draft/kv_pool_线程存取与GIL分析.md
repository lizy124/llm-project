# kv_pool 线程存取机制与 GIL/IPC 分析

> 分析对象：`D:\lzy\project\kv_pool\code\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool`
> 整理日期：2026-08-07
> 内容：① kv_pool 总体结构梳理 → ② 存取是否通过线程的确认 → ③ GIL 锁问题与改 IPC 的可行性分析

---

## 一、kv_pool 总体结构梳理

`kv_pool` 是 vLLM-Ascend（NPU）的 KV cache 传输/卸载连接器集合，目录下有 **4 个独立的 connector 实现**，都遵循 vLLM 的 `KVConnectorBase_V1` 接口（scheduler 侧 + worker 侧拆分）：

| 子目录 | 定位 | 复杂度 |
|---|---|---|
| `ascend_store` | 核心：P/D 分离式外部 KV 池（producer 存 / consumer 取） | 最高 |
| `recompute_cpu_offload` | 抢占重算场景的 CPU KV 保存 | 中 |
| `simple_cpu_offload` | 上游 SimpleCPUOffload 的 NPU 适配 | 低 |
| `ucm_connector` | 外部 UCM 连接器的转发壳 | 低 |

### 1.1 ascend_store —— 主连接器

实现 **Prefill/Decode 分离部署**：Prefill 节点（`kv_producer`）算完 KV 写入外部池，Decode 节点（`kv_consumer`）直接从池里取，避免重算。

**入口与 Scheduler/Worker 拆分**

入口是 `AscendStoreConnector`（`ascend_store_connector.py:76`），继承 `KVConnectorBase_V1 + SupportsHMA`。构造时按 `role` 分叉：

- `KVConnectorRole.SCHEDULER` → 创建 `KVPoolScheduler`（`pool_scheduler.py:54`）
- 否则 → 创建 `KVPoolWorker`（`pool_worker.py:94`）；非 layerwise 且 rank=0 时额外起一个 `LookupKeyServer`（`ascend_store_connector.py:293`）

`AscendStoreConnector` 本身只是个分发壳：scheduler 侧方法（`get_num_new_matched_tokens` / `build_connector_meta` / `request_finished` …）转给 scheduler，worker 侧方法（`start_load_kv` / `save_kv_layer` / `wait_for_save` …）转给 worker。

**KV 池的 Key 设计（关键）**

Key 由 `PoolKey.to_string()`（`metadata.py:114`）生成，把多维并行信息全部编码进去：

```
{model_name}@pcp{pcp_rank}@dcp{dcp_rank}@head_or_tp_rank:{h}@pp_rank:{p}@group:{g}@cache_role:{r}@cache_family:{f}@{chunk_hash}
```

每个维度的含义：
- `pcp_rank` / `dcp_rank`：prefill / decode context parallel rank
- `head_or_tp_rank`：TP rank，但 MLA 等 `num_kv_head < tp_size` 时按 `put_step` 分组共享
- `pp_rank`：pipeline parallel
- `group`：hybrid 模型下多 KV cache group（如 full-attn + mamba）
- `cache_family`：compress 感知（DSV4 压缩比 `c1/c4/...`），决定 key 粒度
- `chunk_hash`：token 块的 hash

layerwise 模式下还有 `LayerPoolKey`（`metadata.py:140`），多加 `@layer_id:{n}`。

`ChunkedTokenDatabase`（`metadata.py:255`）是中央抽象：把 token 区间映射成 key，并把 key 映射到注册进来的 KV cache buffer 地址（`prepare_value` / `prepare_value_layer`）。

**可插拔后端（backend/）**

通过 `extra_config["backend"]` 选择，`backend/__init__.py` 注册了三个：

| 后端 | 类 | 特性 |
|---|---|---|
| mooncake（默认） | `MooncakeBackend`（`backend/mooncake_backend.py:62`） | RDMA ascend 协议；fabric mem；DSV4 lazy_init；SSD offload |
| memcache | `MemcacheBackend`（`backend/memcache_backend.py:41`） | GVA + lease + batch_copy；device_sdma；layerwise 专用 |
| yuanrong | `YuanrongBackend`（`backend/yuanrong_backend.py:50`） | openEuler HeteroClient，HIXL remote H2D |

`Backend`（`backend/base.py:9`）抽象基类定义 `exists / put / get / register_buffer`，memcache 专有的 `batch_alloc / batch_add_lease / batch_get_key_info / batch_write_finish` 默认抛 `NotImplementedError`。

**两种传输模式**

由 `use_layerwise` 控制：

- **模式 A：非 layerwise（整块传输）**。一个 key 代表跨所有层的整个 block。
  - `KVCacheStoreSendingThread`（`kv_transfer.py:607`）：save。按 group 走 `process_token_key_strings_with_block_ids`，先 `lookup` 跳过已存在的，再 `put` 缺失的
  - `KVCacheStoreRecvingThread`（`kv_transfer.py:895`）：async load（`load_async=True` 时）

- **模式 B：layerwise（逐层传输，与 attention 计算重叠）**。按物理层逐层传输，分两个子路径：
  - **B1. Key 路径**（mooncake 后端）：每 (block, layer) 一个 key
    - `KVCacheStoreKeyLayerSendingThread`（`kv_transfer.py:1042`）
    - `KVCacheStoreKeyLayerRecvingThread`（`kv_transfer.py:1201`）
  - **B2. GVA 路径**（memcache 后端，`use_gva_layerwise=True`）：用全局虚拟地址 + `batch_copy(L2G/G2L)` 批量拷贝
    - `KVCacheStoreLayerSendingThread`（`kv_transfer.py:1308`）
    - `KVCacheStoreLayerRecvingThread`（`kv_transfer.py:1472`）

GVA 路径最复杂，涉及：
- worker 侧 `batch_alloc`（`pool_worker.py:1193`）分配每 rank GVA（scheduler 只查存在性）
- load 前 `batch_add_lease`（`pool_worker.py:1359`）拿读租约（TTL 5min），失败重试 + invalid block 上报
- save 完成后 `batch_write_finish` 发布新 key
- 最后一个 layer 传完 `batch_remove_lease` 释放租约

**Layerwise 缓存布局（layer 复用）**

`layerwise_cache_layout.py` 处理「多个物理层共享同一个 KV buffer」（如 MTP/spec-decode 草稿层复用基础层）：
- `build_layerwise_reuse_layout`（`layerwise_cache_layout.py:189`）：按层签名分桶，相同签名的层共享 buffer
- `prefetch_layer_map`：某层的 load 必须等另一层的 save 完成（复用关系）
- `independent_layers`：不复用的层
- `apply_layerwise_kv_cache_plan`（`layerwise_cache_layout.py:266`）：改写 `kv_cache_tensors` 让多层共享物理 buffer

**Hybrid KV cache 协调器**

`AscendStoreCoordinator`（`coordinator.py:55`）处理混合 KV 布局（如 DSV4 的 full-attn + mamba/sliding-window）：
- `find_longest_cache_hit`（`coordinator.py:144`）：跨多个 group 找最长命中，兼容 eagle spec-decode 和 retention_interval
- `ExternalCachedBlockPool`（`coordinator.py:27`）：用外部 key 存在性 duck-type vLLM 的 `BlockPool`
- `store_mask` / `load_mask`：决定哪些 chunk 该存/取

**端到端流程**

- **Scheduler 侧命中检查**（`get_num_new_matched_tokens`，`pool_scheduler.py:518`）：
  1. consumer（非 `consumer_is_to_load`）直接返回 0
  2. GVA layerwise → `_get_layerwise_gva_hit_tokens`（查 `batch_get_key_info` 的 GVA 有效性）
  3. 否则 → `LookupKeyClient`（`pool_scheduler.py:1146`）通过 ZMQ RPC 问 worker 的 `LookupKeyServer`，worker 调 `lookup_scheduler`（`pool_worker.py:2330`）展开所有 TP/PP rank 的 key 查最长连续命中
  4. 生成 `LoadSpec`（`metadata.py:690`）

- **Scheduler 侧 build_connector_meta**（`pool_scheduler.py:949`）：为每个调度的请求生成 `ReqMeta`（`metadata.py:856`），区分新请求 / 抢占恢复 / 运行中缓存 / 异步加载四类处理。

- **Worker 侧 load**（`start_load_kv`，`pool_worker.py:867`）：
  - layerwise → `process_layer_data`（`pool_worker.py:1645`）
  - 非 layerwise → 遍历请求，token_database 算 keys/addrs/sizes，按 `tp_rank` 循环移位，调 `m_store.get`

- **Worker 侧 save**：
  - layerwise → `save_kv_layer`（`pool_worker.py:1715`）
  - 非 layerwise → `wait_for_save`（`pool_worker.py:1744`）

**其他重要机制**
- `attention_fence`：GVA layerwise 下，H2D/L2G 传输必须等计算流真正到 attention 边界才提交，用 NPU Event 做门控
- Mamba 块延迟释放：`touch_sending_mamba_blocks` 给正在发送的 mamba block 计数 +1，`update_connector_output` 收齐所有 worker 的 `completed_events` 后才 free
- TP mismatch：按 effective_tp_size 拆 sub-key，strided I/O（`_build_strided_addrs`，`pool_worker.py:1910`）
- partial block：最后一个不完整块，layerwise_offload 模式下用 `partial@` key 单独存取
- KV events：`AscendStoreKVEvents` 聚合多 worker 的 `BlockStored` 事件，取交集

### 1.2 其余三个连接器

- **recompute_cpu_offload**：`RecomputeCPUOffloadConnectorV1`（`recompute_cpu_offload_connector.py:43`），用于抢占重算场景。请求被抢占时把 KV 卸到 CPU，恢复时再装回。默认 8GB CPU 容量。新增 API：`update_state_before_preempt` / `has_pending_transfers` / `has_preempted_request` / `reset_cache`。
- **simple_cpu_offload**：`AscendSimpleCPUOffloadConnector`（`simple_cpu_offload_connector.py:24`），继承上游 `SimpleCPUOffloadConnector`，只把 CUDA worker 换成 NPU 原生 `SimpleCPUOffloadNPUWorker`（用 `aclrtMemcpyBatchAsync` + `torch.npu` stream/event）。单节点本地 CPU offload。
- **ucm_connector**：`UCMConnectorV1`（`ucm_connector/connector.py:37`），纯转发壳，委托给外部 `ucm.integration.vllm.UCMConnector`。用 `_get_ucm_delegate_for` 分发，未声明的保留 hook fall through 到内部 `inner_connector`。

### 1.3 核心设计要点总结

1. **Scheduler/Worker 分离**：scheduler 只做决策（命中检查、块分配跟踪、meta 构建），worker 执行实际 I/O；非 layerwise 时 scheduler 通过 ZMQ RPC 借 worker 的 store 查命中
2. **多维并行全编码进 key**：TP/PP/PCP/DCP/group/family/role，让不同 rank 互不冲突，consumer 能精确取到 producer 存的对应分片
3. **三种传输路径**：整块 / key 逐层 / GVA 逐层，按 `use_layerwise` + backend 选择，复杂度递增
4. **可插拔后端**：mooncake/memcache/yuanrong，统一 `Backend` 抽象，memcache 多出 GVA+lease 接口
5. **大量高级特性叠加**：hybrid KV、compress（DSV4）、MLA、TP mismatch、layer 复用、partial block、KV events、延迟释放

---

## 二、存取是否通过线程的确认

**结论：绝大部分存取通过后台传输线程异步完成，但有一个例外。**

### 2.1 线程的启动

线程在 `register_kv_caches`（`pool_worker.py:754`）末尾调 `_start_kv_transfer_threads`（`pool_worker.py:453`）启动，按模式选不同线程类（都在 `kv_transfer.py`）：

| 模式 | send 线程 | recv 线程 |
|---|---|---|
| 非 layerwise | `KVCacheStoreSendingThread`（`kv_transfer.py:607`） | `KVCacheStoreRecvingThread`（`kv_transfer.py:895`）（仅 `load_async`） |
| layerwise + key | `KVCacheStoreKeyLayerSendingThread`（`kv_transfer.py:1042`） | `KVCacheStoreKeyLayerRecvingThread`（`kv_transfer.py:1201`） |
| layerwise + GVA | `KVCacheStoreLayerSendingThread`（`kv_transfer.py:1308`） | `KVCacheStoreLayerRecvingThread`（`kv_transfer.py:1472`） |

所有线程都是 `queue.Queue` 驱动的 daemon 线程（`KVTransferThread.run`，`kv_transfer.py:496`），`m_store.put/get/batch_copy` 都在 `_handle_request`（`kv_transfer.py:520`）里执行。

### 2.2 主线程 ↔ 线程的协作

- **投递**：主线程 `add_request(task)` → `request_queue`
- **等待**：`threading.Event`（`layer_save_finished_events` / `layer_load_finished_events`）+ `torch.npu.Event`（`sync_save_events`，GPU 流同步）
- **完成上报**：`finished_requests` 集合 + `stored_requests` 引用计数
- **GVA layerwise 额外有** `AttentionComputeStartGate`（`attention_fence.py:27`）：传输线程等计算流到 attention 边界才开工

### 2.3 ⚠️ 例外：非 layerwise 同步 load

这是唯一**不走线程**的存取路径。在 `start_load_kv`（`pool_worker.py:867`）里（`pool_worker.py:909-971`）：

```python
if self.load_async:
    self.kv_recv_thread.add_request(request)   # 走线程
    continue
# 否则在当前 forward 线程里同步直接调：
ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
```

即 `use_layerwise=False` 且 `load_async=False` 时，load 是在 worker 主线程（forward 调用栈里）**同步直接**调 `m_store.get`，不走后台线程。

### 2.4 总结

- **save 全部走线程**（非 layerwise / key layerwise / GVA layerwise 都是 `add_request` + event 等待）
- **layerwise load 全部走线程**（含 GVA lease/批拷贝）
- **非 layerwise load**：`load_async=True` 走 `KVCacheStoreRecvingThread`；`load_async=False` 在主线程同步调 `m_store.get`（tp_mismatch 的 `_load_kv_tp_mismatch`（`pool_worker.py:1969`）也是同理）

---

## 三、GIL 锁问题与改 IPC 的可行性分析

### 3.1 前置事实：后端扩展与 GIL

mooncake/memcache/yuanrong 都是外部 C++ pybind11 扩展（`m_store.put/get/batch_copy`）。RDMA/DMA 类存储扩展按惯例会释放 GIL（用 `py::call_guard<py::gil_scoped_release>()` 或函数内显式 release），否则跨节点传输会卡死整个 Python 进程。

传输线程的工作分两类，GIL 影响完全不同：

| 工作 | 是否受 GIL | 代码位置 |
|---|---|---|
| `m_store.put/get/batch_copy/exists`（RDMA/DMA I/O） | ❌ 不受（C++ 扩展释放 GIL） | `pool_worker.py:971`、`kv_transfer.py:477` |
| key 字符串生成（`PoolKey.to_string` 拼接 + 生成器循环） | ✅ 受 | `metadata.py:114`、`metadata.py:555` |
| `_handle_stored_request` 里 for 循环构建 addrs/sizes | ✅ 受 | `kv_transfer.py:799-855` |
| `LayerBatchBuilder`（numpy 向量化） | ❌ 不受（numpy 释放 GIL） | `kv_transfer.py:101` |

**所以 GIL 真正卡的是 Python 层的 key 生成和循环，不是 I/O 本身。** 而每进程只有 1 个 send 线程 + 1 个 recv 线程（`pool_worker.py:329`），争抢不激烈。这个多线程的**目的是"I/O 与 forward 计算重叠"，不是"CPU 多核并行"** —— 而 forward 计算（NPU kernel）也释放 GIL，正好让传输线程拿 GIL 做 key 构建，是个配合关系。

### 3.2 改 IPC 能不能解决？能，但不划算

**改 IPC 的真实代价**

1. **NPU KV cache buffer 跨进程共享是硬骨头**。`register_buffer` 注册的是设备内存 `data_ptr`（`pool_worker.py:793`），`torch.Tensor.share_memory` 只对 CPU tensor 生效。新进程要么重新 mmap NPU 显存（需平台支持跨进程 NPU 句柄），要么走 H2D 中转——后者直接抹平收益。

2. **mooncake/memcache store 是按进程初始化的**。新进程要重新 `store.setup()` + `register_buffer`（`mooncake_backend.py:93`），等于多一份内存池贡献和多一次 metadata server 注册。

3. **序列化 + 往返开销**。keys（字符串列表）、addrs（`list[list[int]]`）、sizes 每次都要打包过管道；完成事件还得反向 IPC 一次。而当前 `queue.Queue.put` 是微秒级。

4. **本质上 mooncake/memcache 自己已经是跨进程共享存储**了（metadata server + 各 rank store 进程）。在它之上再叠一层进程间通信，没解决底层任何问题。

**项目里已有 IPC 的先例，但目的不同**

`LookupKeyServer/Client`（`ascend_store_connector.py:293`）用 ZMQ `ipc://` 做 scheduler↔worker 通信。但这是**跨角色**通信（scheduler 进程没持有 store，借 worker 的 store 查命中），不是为了绕开 GIL——scheduler 本来就不做 I/O。

### 3.3 更现实的优化方向

如果确实观测到 GIL 成为瓶颈（profile 出来 key 生成占比高），按性价比排序：

1. **key 生成向量化或下沉**。`PoolKey.to_string` 逐块字符串拼接是大头，可改成 numpy 批量构造（参考 `LayerBatchBuilder` 已用 numpy），或下沉到 C++ 扩展。当前代码已部分在做（`_key_prefix_cache` 缓存前缀，`metadata.py:284`）。

2. **确认后端扩展释放 GIL**。mooncake `batch_put_from_multi_buffers` / memcache `batch_copy` 若没释放，加 `py::call_guard<py::gil_scoped_release>()`。这是零成本的大收益。

3. **减少线程内 Python 循环**。`_handle_stored_request` 的 `for index, start in enumerate(starts):` 逐块构建 addr/size（`kv_transfer.py:846`）可整体向量化。

4. **极端情况下用子进程 + 共享 NPU 句柄**，但要等 PyTorch/Ascend 提供稳定的跨进程 NPU 内存共享 API，目前不成熟。

### 3.4 结论

- **GIL 在这里有影响但有限**：I/O 主体（RDMA/DMA）在 C++ 扩展里释放 GIL，卡 GIL 的是 Python 层 key 生成与循环；每进程仅 1-2 个传输线程，争抢不激烈；设计目的是 I/O 与计算重叠，不是 CPU 多核并行。
- **改 IPC 技术上可行但收益不抵代价**：NPU buffer 跨进程共享 + store 重初始化 + 序列化开销，且 mooncake 本身已是跨进程存储。
- **推荐路径**：先把 key 生成向量化、确认 C++ 扩展释放 GIL、减少线程内 Python 循环，性价比远高于改 IPC。

---

## 附：关键文件索引

| 文件 | 作用 |
|---|---|
| `ascend_store/ascend_store_connector.py` | 入口，分发 scheduler/worker |
| `ascend_store/pool_scheduler.py` | scheduler 侧：命中检查、meta 构建 |
| `ascend_store/pool_worker.py` | worker 侧：实际 I/O、线程启动、load/save |
| `ascend_store/kv_transfer.py` | 所有传输线程类 |
| `ascend_store/metadata.py` | Key 设计、ChunkedTokenDatabase、ReqMeta |
| `ascend_store/coordinator.py` | hybrid KV 命中协调 |
| `ascend_store/layerwise_cache_layout.py` | layer 复用布局 |
| `ascend_store/attention_fence.py` | GVA layerwise 计算流门控 |
| `ascend_store/backend/base.py` | 后端抽象基类 |
| `ascend_store/backend/mooncake_backend.py` | mooncake RDMA 后端 |
| `ascend_store/backend/memcache_backend.py` | memcache GVA 后端 |
| `ascend_store/backend/yuanrong_backend.py` | yuanrong 后端 |
| `recompute_cpu_offload/` | 抢占重算 CPU 卸载 |
| `simple_cpu_offload/` | 上游 simple CPU offload 的 NPU 适配 |
| `ucm_connector/` | UCM 转发壳 |
