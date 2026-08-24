# Mooncake 后端与 LMCache 的 MLA KV 读去重分析

**日期：** 2026-08-24  
**范围：** `vllm-ascend` 的 AscendStore KV Pool，重点是 Mooncake backend；对比本地 `LMCache` 和上游 vLLM/NIXL 的 replicated-KV 处理。本文只做问题论证和设计评审，不修改实现代码，也没有声称已经完成 NPU 实测。

## 1. 结论先行

### 1.1 Issue 描述的问题成立，但不是所有 KV Pool 读路径都成立

在纯 MLA、`tp_size > 1`、相同 TP 组共享同一份 KV Pool key namespace、请求命中缓存的条件下，当前 AscendStore 的写侧已经按 TP 去重，而同步读侧仍由每个 TP rank 独立调用 `m_store.get`。因此同一份逻辑 KV 会被 Mooncake 后端读 `tp_size` 次，再分别写入每个 rank 的本地 KV cache。这个重复 I/O 是真实存在的，且随 TP 增大线性放大。

问题不能简单推广到以下场景：非 MLA 的按 head 分片 KV、MLA+SSM/hybrid、TP 不一致、不同 DP/PP/CP 组、layerwise/GVA、异步接收、部分 miss，或 key namespace 本来就按 rank 分开。这些场景需要不同的 source-rank 映射，不能仅凭 `use_mla` 判断。

### 1.2 Mooncake 和 LMCache 的共同解法是“owner 读取一次，TP 内广播”，但实现抽象不同

- **共同语义：**为一个 replicated KV group 选择一个 owner；owner 访问远端存储，其他 rank 不访问后端；owner 将结果在本地 TP group 分发。
- **LMCache 的载荷：**连续的 `MemoryObj`/chunk，广播 metadata 和 `uint8` tensor，再由 peer 送入自己的 GPU cache。
- **Mooncake 的载荷：**AscendStore 的 `get(keys, addrs, sizes)` 接收每个 key 的多组 NPU 地址和长度，目标通常跨层、跨 cache entry、带 stride/padding，不是一个连续 tensor。Mooncake 原生 API 负责把对象直接 scatter 到多个目标 buffer；它没有“读到 owner 后自动参与 HCCL broadcast”的抽象。

因此可以搬用 LMCache 的**控制面原则**（owner/passive、pure MLA guard、TP group broadcast、chunk 化、统一失败状态），不能直接搬用其 `MemoryObj` 数据结构或 `retrieve_layer` 调用路径。

### 1.3 第一阶段建议

在 AscendStore worker 层新增独立的 read-plan/helper，采用“同一 TP 组 owner 只调用一次 Mooncake，按受控 byte budget 分块 pack/broadcast/unpack”的 opt-in 实现，初始只允许：

```text
pure MLA + tp_size > 1 + peer_tp_size == tp_size
use_layerwise == False
load_async == False
use_sparse == False
use_hybrid == False
put_step == tp_size
```

非满足条件继续走现有路径。默认开关先保持关闭，完成 CPU/mock 单测和 NPU TP=2/4 正确性、hang、OOM、性能验证后，再决定是否默认开启。layerwise、async、hybrid、异构 TP 不应作为第一阶段的隐式扩展。

## 2. 代码和资料基线

本分析以 `vllm-ascend` 的 `main`（`d85e6714a`）为目标基线，并核对了当前 checkout 的已有 refactor 提交。当前 checkout 为 `5b199c84b`，相对 `main` 只涉及 `memcache_backend.py`、`pool_scheduler.py`、`pool_worker.py` 的既有整理；Mooncake backend 文件和本文引用的 `get`/地址构造语义未被该提交改变。因此文中结论适用于 main 的相关后端语义，也适用于当前 checkout。

本分析基于以下本地源码：

| 项目 | 路径 | 版本/提交 | 用途 |
| --- | --- | --- | --- |
| vllm-ascend | `D:\lzy\project\kv_pool\code\vllm-ascend` | 当前工作树 `5b199c84b`，其父为 `upstream/main` `d85e6714a` | AscendStore 实现；工作树比 `main` 多一个已有 refactor 提交 |
| LMCache | `D:\lzy\project\kv_pool\code\LMCache` | `dev` / `f9addd2` | owner/passive 和 TP broadcast 参考实现 |
| Mooncake | `D:\lzy\project\kv_pool\code\Mooncake-src` | `main` / `82e2799` | `batch_get_into_multi_buffers` 和 replica 路径 |
| vLLM | `D:\lzy\project\kv_pool\code\vllm` | 本地基线见 `implementation_plan.md` | replicated KV topology、NIXL source-rank 参考 |

外部语义参考：DeepSeek-V2 MLA 论文（arXiv:2405.04434）、vLLM `TransferTopology.replicates_kv_cache()` 和 NIXL pull worker 的 pure MLA 注释。它们支持“MLA latent KV 在 TP 间 replicated”的语义判断，但不替代 AscendStore/Mooncake 的地址和生命周期验证。

## 3. AscendStore 当前调用链

### 3.1 写侧已经建立了 replicated MLA 的 key 语义

`vllm_ascend/.../ascend_store/pool_worker.py` 的初始化逻辑在 MLA 下将 `num_kv_head` 视为 1；当 `tp_size > 1` 时设置 `put_step = tp_size`，并令 `head_or_tp_rank = 0`。`_process_save_for_layer_batch`、`_alloc_gvas_for_save` 以及同步/异步发送线程均跳过 `tp_rank % put_step != 0` 的 rank。

结果是：纯 MLA 同一个 TP 组写入相同的 key namespace，只有一个 rank 向 Pool 写入。这是读去重的必要前提，但不是读去重本身。

### 3.2 同步读侧仍是每 rank 一次后端访问

`KVPoolWorker.start_load_kv` 在非 layerwise、非 async 分支中为每个请求构造：

```text
key_list, addr_list, size_list, block_id_list
```

随后按 rank 做 `_circular_shift`，调用一次 `self.m_store.get(...)`。每个 MLA TP rank 的 key 相同，`prepare_value` 只把目标地址换成本 rank 的 KV cache。因此 TP=8 时，后端收到的是 8 次相同逻辑数据的读取请求；区别仅在目标 HBM 地址和请求顺序。

`_circular_shift` 是为了避免所有 rank 同时以相同首 key 访问后端，不能直接保留在 broadcast 协议中：广播要求所有 rank 对同一个 step 使用相同的 entry/chunk 顺序和 collective 次数。

### 3.3 不能只修改 `start_load_kv`

当前还有三类读实现：

1. **同步整批读：**`start_load_kv` 直接调用 backend `get`。
2. **async 读：**`KVCacheStoreRecvingThread._handle_request` 调用 `m_store.get`。
3. **layerwise 读：**`KVCacheStoreKeyLayerRecvingThread` 按 layer 调用 `m_store.get`；GVA/MemCache layerwise 路径则使用 `batch_copy`、lease 和 layer event。

第一阶段只覆盖第一类是有意的范围约束，而不是“已经解决全部 issue”。后两类必须建立明确的跨 rank rendezvous 和 layer 生命周期协议后再做。

### 3.4 数据并非单一连续 tensor

`metadata.py::prepare_value` 为一个 Pool key 返回所有本地 layer/cache entry 的地址和 size。一个 token/block 的 `size_list` 是多个 span 的集合，实际布局可能包含：

- 多层或多 cache group；
- 不同 dtype、padding 和 stride；
- block 内的非零 offset；
- GVA layerwise 的物理 layer 重用；
- 每个 rank 独立分配的目标 HBM。

任何去重实现都必须以 span 的**逻辑顺序、字节长度和目标地址**为协议，而不能把第一个 tensor 的 `numel()` 当作总 payload，也不能假定所有 span 可用一个连续 `view` 表示。

## 4. Mooncake backend 的实际语义

### 4.1 Python wrapper 接口

`vllm_ascend/.../ascend_store/backend/mooncake_backend.py` 的接口是：

```python
get(keys: list[str], addrs: list[list[int]], sizes: list[list[int]])
```

每个 key 对应一组目标地址和长度。wrapper 调用：

```python
self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
```

`register_buffer()` 通过 `global_te.register_buffer(ptrs, lengths)` 注册 NPU buffer；地址在 Mooncake transfer engine 生命周期内必须有效。wrapper 将正的完成字节数归一化为 0，负数保留为失败码，异常返回 `None`。这意味着上层不能只判断“调用没有抛异常”，还要解释每个 key 的状态。

### 4.2 Mooncake C++ 读路径

`Mooncake-src/mooncake-store/src/real_client.cpp` 的 `batch_get_into_multi_buffers_internal()` 大体执行：

1. 查询每个 key 的 metadata 和 replica 列表；
2. 选择可用 replica（通常优先本地 MEMORY/NOF，再考虑其他类型）；
3. 校验用户提供的多 buffer 总容量与对象大小；
4. 对 memory/NOF 直接向多个目标 slice 发起传输；
5. 对 local disk、DISK/DFS 等路径，可能先读入连续临时 buffer，再 scatter 到目标地址；
6. 逐 key 返回完成或失败状态。

`client_service.cpp::Client::BatchGet` 提交 transfer future 并调用 `future.get()`，所以同步 API 正常返回时，目标 buffer 已可用于后续 pack/broadcast。Python binding 会释放 GIL，但不改变这个同步完成语义。

### 4.3 对 read dedup 设计的含义

- **去重位置应在 worker 编排层。**若让每个 rank 继续调用 Mooncake，再在调用之后广播，后端 I/O 已经重复，问题没有解决。
- **owner 可以直接读入自己的 KV cache。**不需要先把 Mooncake 对象改成一个新类型；但必须从 owner 的目标 span 构造可读取的 byte payload。
- **不能假设所有 replica 一样快或一样连续。**memory/NOF、local disk、DFS 的延迟、临时内存和失败码不同；benchmark 必须分别覆盖。
- **Mooncake 的同步返回不等于任意 NPU stream 已完成。**若 backend 在非默认 stream 提交操作，必须以实际 transfer engine API 确认 pack 所在 stream 可见，必要时插入 event/同步。
- **对象/lease 生命周期仍由 Mooncake 控制。**owner 完成读取前不能释放 lease 或让对象被 eviction；layerwise 现有 `batch_remove_lease` 的时机不能被简单挪到第一次 broadcast 后。

## 5. LMCache 的解法和边界

### 5.1 owner/passive 机制

本地 `LMCache/lmcache/v1/cache_engine.py` 初始化 `save_only_first_rank`：默认依据 `metadata.use_mla` 开启，也允许 extra config 覆盖。开启后，首 rank 保存/读取 storage，其他 TP rank 跳过后端访问。

在普通 retrieve 路径中，首 rank 调用 `storage_manager.batched_get`；随后 `_broadcast_or_receive_memory_objs`：

1. 广播 chunk 数；
2. 广播每个 chunk 的范围和 metadata；
3. 广播 `uint8` tensor payload；
4. peer 分配自己的 memory object，接收后继续执行同一个 `batched_to_gpu` 流程。

这证明了“后端读取从 TP 次降到 1，TP 内 broadcast 补齐复制”的语义方案可行。

### 5.2 LMCache 已明确的限制

- `cache_engine.py` 初始化处有 `TODO: support save_only_first_rank when use layerwise`；`retrieve_layer` 路径不能视为已经支持该去重语义。
- `compute_extra_count()`/读锁逻辑用于多个 reader 的并发保护，不是减少实际 backend read 次数的方案。
- 多进程 connector 的 `mla_only` guard 能避免 hybrid 误启用，但核心 `LMCacheEngine.save_only_first_rank` 主要依据 `metadata.use_mla`；不能把 connector 的局部 guard 当作所有集成路径的全局证明。
- `MemoryObj` 通常代表连续 chunk，LMCache 不需要处理 Mooncake `addrs/sizes` 这种任意多 span 目标布局；它的 pack/unpack 成本和 AscendStore 不等价。
- 首 rank 的定义通常是 metadata 中的 first rank（当前实现为 0）。在 PP/DP/多节点组合中，必须确认它是**本地 TP group rank 0**，而不是全局 rank 0。

### 5.3 可借鉴的不是代码，而是协议

可以直接借鉴：

- `owner/passive` 分支和 owner 只访问后端的 invariant；
- pure MLA/replicated KV guard；
- TP group 的 tensor broadcast，而不是跨 PP/DP/CP 的全局广播；
- chunk 化的 header + payload 顺序；
- peer 在 broadcast 后继续使用已有 GPU load 处理；
- owner 统一产生 hit/miss 状态，所有 rank 使用同一 invalid/recompute 结果。

不能直接复制：

- `MemoryObj` 作为 Mooncake 多地址对象的替代品；
- LMCache `retrieve_layer` 的逐层调用来覆盖 AscendStore layerwise；
- 通过增加 reader lock 或 `world_size` 折叠来代替 AscendStore key/地址协议；
- 假定 broadcast payload 与 KV cache tensor 的物理 layout 相同。

## 6. vLLM/NIXL 和业界方案对照

上游 vLLM 的 `TransferTopology.replicates_kv_cache()` 明确将 MLA 视为 replicated KV（hidden dimension 不能按普通 KV head 切分）。NIXL pull worker 的注释也说明：pure MLA 只需要一次 read，MLA+SSM 仍需按 SSM source rank 读取。

这类实现通常在**源端选择/点到点传输映射**阶段去重：决定哪个远端 rank 是 source，然后直接建立 NIXL transfer。它们不等价于 Mooncake 的“Pool object -> owner HBM 多 span -> local HCCL broadcast”，但给出两个重要约束：

1. replicated 判断必须按 KV group/spec 做，不能将整个 hybrid 请求视为 replicated；
2. source rank 是 topology 的属性，必须和本地 TP/PP/DP group 匹配，不能固定使用全局 rank 0。

## 7. 方案对比

| 方案 | 后端读次数 | 数据抽象 | 适配 Mooncake 多 span | 额外内存 | 主要风险 |
| --- | ---: | --- | --- | --- | --- |
| 当前 AscendStore | TP 次 | 直接写各 rank HBM | 原生支持 | 低 | 重复远端 I/O |
| LMCache owner/broadcast | 1 次 | 连续 MemoryObj/chunk | 需重做 pack/unpack | chunk staging | layerwise 尚未完整支持 |
| vLLM/NIXL source mapping | 每个 replicated group 1 次 | 点到点 registered buffers | 依赖 NIXL layout | 可低 | 不同于 Mooncake API |
| Mooncake owner + HCCL（建议） | 1 次 | owner 多 span + byte payload | 可适配 | 受 byte budget 限制 | HCCL 顺序、pack 开销、stream 可见性 |
| Mooncake 新增 P2P/replica fan-out | 1 次后端读，后端/传输层 fan-out | Mooncake 原生对象 | 需要 C++/协议改动 | 可能低 | 改动面最大、跨版本兼容和故障恢复复杂 |

## 8. Mooncake 场景的特殊设计约束

### 8.1 replicated 判定和 group 选择

第一阶段至少同时检查：

```text
metadata.use_mla
tp_size > 1
put_step == tp_size
peer_tp_size == tp_size
同一 Pool key namespace
不是 hybrid/SSM/sparse
```

`tp_mismatch` 不能作为唯一 guard：当前 metadata 逻辑对 MLA 有意不把它标成普通 head mismatch，但 `peer_tp_size` 仍可能不同。owner 应是**当前 local TP group 的 rank 0**，而不是通过全局 rank 计算的固定 0。

### 8.2 canonical request plan

所有 rank 必须先独立构造相同的 canonical plan：key 顺序、block id、group/layer span 数、每个 span 的 byte size 和 chunk 边界均相同。可以广播或 all-reduce 一个轻量 digest 做一致性检查；检查失败必须在进入 collective 前回退原路径或统一失败，不能让某个 rank 静默跳过 broadcast。

`_circular_shift` 只能保留在原 backend 路径。owner/passive 协议中，所有 rank 使用同一 canonical 顺序，避免 root 和 peer 对不同 chunk 调用 collective 导致 HCCL hang。

### 8.3 pack/unpack 不能破坏目标布局

推荐 payload 定义为 span 的字节串：

```text
header = request_id / plan_digest / chunk_id / span_count / status[] / byte_count
payload = span[0] bytes || span[1] bytes || ... || span[n-1] bytes
```

owner 从已完成写入的 KV cache span pack 到可复用的 `torch.uint8` staging buffer；peer 按自己的目标地址和 size unpack。对非 contiguous、stride、padding、offset 只能做显式 copy，不能假设 `tensor.view(torch.uint8)` 后就是目标逻辑顺序。需验证 NPU byte view、device copy 和 HCCL broadcast 对 dtype/对齐的要求。

### 8.4 Mooncake 完成与 stream 可见性

`BatchGet` 的 future 返回意味着 Mooncake transfer 已结束，但 owner 的 pack 可能运行在另一个 NPU stream。必须有明确 event/stream dependency：

```text
Mooncake get complete -> owner pack visible -> HCCL broadcast -> peer unpack visible -> attention
```

不能以 Python 函数返回或日志顺序代替设备同步证明。

### 8.5 replica、磁盘和临时 buffer

memory/NOF 路径可能直接写入 owner spans；DISK/DFS/local-disk 路径可能先进入连续临时 buffer，再 scatter。后端去重只减少调用次数，不会消除这些 replica 内部拷贝。测试必须记录 replica 类型、临时 buffer 峰值、checksum/对象完整性和失败码，不能只测内存 replica。

### 8.6 partial miss 和失败状态

Mooncake 返回逐 key 状态，wrapper 还可能返回 `None`。owner 必须把状态归一化后广播给所有 rank：

- hit：广播对应 payload；
- miss/短读/异常：所有 rank 将相同 block 标为 invalid，走现有 recompute；
- 空请求：所有 rank 仍完成相同协议次数，或在 collective 前一致地走空路径；
- 不能让 peer 把未填满的 staging 当成有效 KV。

失败状态本身也需要进入 broadcast，否则 owner 失败而 peer 等待 payload，会出现死锁或数据未初始化。

### 8.7 lease、eviction 和生命周期

owner 读期间对象必须保持 lease；所有 peer 完成 unpack/消费前不能提前释放。layerwise 当前在最后 layer 才集中 `batch_remove_lease`，第一阶段不应改变这一语义。若未来按 layer/block 广播，必须为每个 chunk 定义“可释放”事件，而不是以 owner 的 broadcast 返回作为全局完成。

### 8.8 async/layerwise 不能直接套 collective

每个接收线程独立处理请求时，rank 间请求到达顺序可能不同。直接在线程中调用 HCCL broadcast 会把请求 A 的 root 与请求 B 的 peer 配对，造成 hang 或数据串写。需要中心调度的 step/request sequence、TP rendezvous 或显式通信队列。

layerwise 还受 `wait_for_save`、attention start gate、layer finished event、物理 layer 重用约束；GVA/MemCache 的 `batch_copy` 不是普通 Mooncake `get` 的同义替换。因此放在第二阶段单独设计。

## 9. 推荐协议（第一阶段）

### 9.1 控制条件

增加 opt-in 配置，例如 `enable_mla_read_dedup=False`。只在第 3 节列出的纯 MLA 同步场景启用；其他情况调用现有 `m_store.get`，保证回归行为不变。

### 9.2 一次请求的步骤

1. **构造计划：**每个 rank 生成相同 key/span/block 顺序和 plan digest；不再按 rank circular shift。
2. **一致性检查：**校验 entry 数、每个 span 数、总字节数和 chunk 边界；不一致则在 collective 前统一回退。
3. **owner 读取：**local TP rank 0 调用一次 Mooncake `get`，目标为 owner 自己的 KV cache spans。
4. **状态归一化：**将 Mooncake 返回值、`None`、短读、异常转换成 per-key/per-span status；owner 记录实际完成字节数。
5. **受控打包：**按固定 byte budget（初始建议 64 MiB，必须通过 NPU 测试调整）逐 chunk pack 到复用的 `uint8` staging buffer，不按整个长请求一次性分配。
6. **广播 header：**广播 chunk 数、plan digest、span metadata、status 和 payload 字节数；所有 rank 以相同次数进入 tensor collective。
7. **广播 payload：**owner 广播 staging tensor，peer 接收到自己的 staging 后按本地 span 地址 unpack。
8. **完成和调度：**所有 rank 使用一致 invalid block 集合；成功 span 才交给 attention，失败 span 走原有 recompute；确认 device event 后再允许后续计算。
9. **生命周期：**遵循现有 lease/remove-lease 规则，确保最后一个 consumer 完成后再释放对象。

### 9.3 为什么推荐 bounded chunk

64K token、全层、多 cache group 的 payload 可能达到数百 MiB 甚至 GiB；整批 staging 会把去掉的网络 I/O 换成 HBM 峰值和 OOM。chunk 边界应尽量与 logical block 或 layer group 对齐，便于 partial miss、错误恢复和未来 layerwise 复用，但第一阶段仍保持单个同步请求内的确定顺序。

## 10. 可搬用项、不可搬用项和 gap

| 项目 | 可否搬用 | 原因/改造要求 |
| --- | --- | --- |
| owner/passive 分支 | 可以 | 只需映射到 local TP group，并保证 peer 不访问 backend |
| pure MLA guard | 可以但需加强 | 同时检查 hybrid、`put_step`、`peer_tp_size` 和 key namespace |
| LMCache metadata + uint8 broadcast | 原则可用 | 需改成 span descriptor；不能复用 MemoryObj layout 假设 |
| chunk 化和复用 staging | 可以 | 需要按 HBM byte budget 和 NPU stream 实测 |
| LMCache `retrieve_layer` | 不能直接用 | 代码明确尚未支持 save-only-first-rank layerwise |
| `compute_extra_count`/读锁 | 不能解决 | 它保护并发，不减少后端读取次数 |
| vLLM NIXL source-rank mapping | 只能借鉴语义 | NIXL 的 registered buffer/点到点传输不同于 Mooncake multi-buffer |
| Mooncake C++ 内部 replica 选择 | 无需重写 | owner 仍调用原 `get`；新增的是上层 fan-out |
| Mooncake 新增 native fan-out | 第二阶段候选 | 需要 C++、元数据、错误恢复和版本兼容，改动面最大 |

主要 gap：没有现成的 AscendStore span pack/unpack helper、没有统一的 TP rendezvous 协议、没有证明所有 backend 读完成后同一 NPU stream 可见、没有 layerwise owner 生命周期方案，也没有针对 HCCL hang/partial miss 的端到端测试。

## 11. 风险清单与缓解措施

| 风险 | 严重度 | 缓解/验收 |
| --- | --- | --- |
| rank 请求顺序或 collective 次数不一致 | 致命 | canonical plan、digest、统一 header/status；任何不一致在 collective 前回退 |
| owner get 未完成就 pack | 致命 | 以 Mooncake future + NPU event 建立依赖；故障注入验证 |
| partial miss 被当成有效 KV | 高 | 广播 per-key/per-span status；与 baseline invalid block 逐项比较 |
| staging 导致 HBM OOM | 高 | bounded chunk、复用 buffer；64K/多 block 峰值监控 |
| pack/unpack 和 HCCL 抵消 I/O 收益 | 高 | 记录 backend、pack、broadcast、unpack 分段耗时；以 TTFT/P95 决策 |
| hybrid/SSM 被错误复制 | 高 | pure MLA hard guard；混合模型回归 backend 调用次数和结果 |
| PP/DP/CP 选错 owner | 高 | 使用 local TP group rank；多组 topology NPU 测试 |
| SSD/DISK replica 临时 buffer/错误码差异 | 中高 | 按 replica 类型覆盖 hit/miss/短读/eviction |
| lease 提前释放 | 高 | 检查最后 consumer 完成事件；长请求连续命中测试 |
| async/layerwise 线程死锁 | 高 | 第一阶段关闭；第二阶段先设计 rendezvous，再实现 |
| graph/fullgraph 对动态 collective 不兼容 | 中高 | 单独做 graph mode 兼容性实验；不满足则明确禁用该开关 |

## 12. 验证计划和 go/no-go 标准

### 12.1 CPU/mock 单测

- pure MLA TP=2/4/8：owner backend 调用恰好 1 次，peer 调用 0 次；所有 rank 的 payload 和目标 bytes 逐字节一致；
- 非 MLA、TP=1、TP mismatch、hybrid、sparse、async、layerwise：调用次数和地址顺序与 baseline 一致；
- 多 layer、多 group、不同 span 数、stride/padding/offset、空请求；
- 全 hit、全 miss、部分 miss、`None`、负错误码、短读；
- digest/entry 数不一致时不进入 collective；
- chunk 边界、复用 staging、异常后下一请求不残留状态。

### 12.2 NPU 多进程

至少 TP=2/4，条件允许再做 TP=8；覆盖：

- memory/NOF 与 SSD/DISK/DFS replica；
- 冷 miss、全 hit、部分 hit、多个 block、多 layer、16K/64K prompt；
- HCCL 无 hang、无未初始化数据、无 OOM、无 stream race；
- PP=1 先验收，再验证 DP/CP/PP 组不会误跨组；
- owner/peer 目标 KV cache 与 baseline 逐字节一致，attention 输出一致。

### 12.3 性能和 go/no-go

必须同时记录 baseline 与 opt-in：`m_store.get` 次数、读取字节数、backend 时间、pack/unpack 时间、HCCL 时间、staging 峰值、TTFT、TPOT、P50/P95/P99、错误/重算率。

基本 go 条件：

1. 纯 MLA hit 的 backend read 从 TP 次降为 1；
2. 所有 rank 结果逐字节正确，无 collective hang/OOM/stream race；
3. 非 MLA/hybrid/TP mismatch 等路径行为零回归；
4. 目标长上下文下 HBM 峰值不超过预算；
5. 端到端 TTFT/P95 不劣于 baseline，且读 I/O 节省足以覆盖 pack/broadcast 开销。

若只满足第 1 条而不满足第 2～5 条，不应默认开启；应保留 opt-in 或回退原路径。

## 13. 实施前必须冻结的设计问题

1. AscendStore 使用的 TP group API 和 local rank 获取方式，是否能覆盖 DP/PP/CP 组合；
2. Mooncake transfer engine 返回后，owner KV cache 在哪个 NPU stream 上可安全读取；
3. span pack 的 canonical byte order、对齐和 dtype/stride 处理；
4. header/status 是使用 HCCL object broadcast、固定 tensor，还是现有 vLLM group wrapper；
5. partial miss 时 scheduler 的 invalid block/recompute 接口能否接受 owner 统一状态；
6. bounded chunk 的默认 byte budget 和 staging buffer 生命周期；
7. graph/fullgraph、异步请求队列和 layerwise 的明确禁用条件；
8. Mooncake lease 在同步 owner/broadcast/unpack 全链路中的最晚释放点。

这些问题没有在代码和 NPU 实验中冻结前，不应开始大范围实现或宣称问题已解决。

## 14. 最终建议

issue 的性能问题是真实的，且 Mooncake 后端本身没有提供现成的 TP fan-out；最合适的改动边界是 AscendStore worker 的编排层。LMCache 和 vLLM/NIXL 已经验证了 owner/passive 的语义方向，但它们不能替代 Mooncake 的多 span 地址、replica、lease 和 NPU stream 处理。

建议按以下顺序推进：

1. 先评审并冻结 pure MLA synchronous 协议和 local TP owner 定义；
2. 实现独立 read-plan + bounded pack/broadcast/unpack，默认 opt-in；
3. 完成 CPU/mock 与 TP=2/4 NPU 正确性和故障测试；
4. 用真实 Mooncake memory/SSD replica 做端到端性能对比；
5. 只有在结果满足 go/no-go 后，才讨论默认开启、layerwise、async 或 hybrid 扩展。
