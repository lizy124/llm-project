# kv_pool 优化点系统性梳理

> 目标：推动 `D:\lzy\project\kv_pool\code\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool` 这部分代码往更好的方向发展——性能与结构双维度。
> 整理日期：2026-08-08
> 方法：四维度并行扫描（性能/结构/正确性/扩展性）+ 三轮深度专项分析（RDMA batch 边界 / 同步等待与空闲 / 内存拷贝与分配）+ 结合已有的两份分析文档
> 代码行数统计基于当前源码快照

---

## 0. 概览

本次梳理基于对 kv_pool 全部源码的四维度扫描（性能瓶颈、代码结构、正确性风险、可扩展性），并结合已有的两份分析文档：

- [kv_pool_线程存取与GIL分析.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_线程存取与GIL分析.md)（结构梳理 + GIL/IPC 分析）
- [MLA_KV读取去重优化讨论.md](file:///D:/lzy/project/kv_pool/llm-project/draft/MLA_KV读取去重优化讨论.md)（MLA 读侧去重提案）

### 文件规模现状（ascend_store 子目录）

| 文件 | 行数 | 评估 |
|------|------|------|
| pool_worker.py | 2306 | ⚠️ 巨文件 |
| kv_transfer.py | 1535 | ⚠️ 巨文件 |
| config_data.py | 1112 | ⚠️ 巨文件 |
| pool_scheduler.py | 1106 | ⚠️ 巨文件 |
| coordinator.py | 377 | 正常 |
| mooncake_backend.py | 348 | 正常 |
| layerwise_cache_layout.py | 298 | 正常 |
| ascend_store_connector.py | 286 | 正常 |
| memcache_backend.py | 206 | 正常 |
| yuanrong_backend.py | 144 | 正常 |
| backend.py | 39 | 过小（接口过载） |

### 配置项散落统计（`extra_config.get` 调用次数）

| 文件 | 调用次数 |
|------|----------|
| pool_worker.py | 9 |
| pool_scheduler.py | 5 |
| config_data.py | 1 |
| **合计** | **15 处散落读取** |

---

## 一、性能优化点

### P1. MLA KV 读取去重【已有专项文档】

**问题**：MLA 写侧已去重（只 rank 0 写），读侧未去重（每 TP rank 各自从池子 get 一份到各自 buffer）。

**证据**：
- 写侧：[pool_worker.py:1020](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1020) `if self.tp_rank % self.put_step != 0: return`
- 读侧：[pool_worker.py:867-980](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L867-L980) 无对应跳过

**提案**：rank 0 取 + TP 组 broadcast，详见 [MLA_KV读取去重优化讨论.md](file:///D:/lzy/project/kv_pool/llm-project/draft/MLA_KV读取去重优化讨论.md)。

**严重程度**：高（TP 越大浪费越多，MLA 小数据固定开销占比高）

---

### P2. Key 字符串生成向量化/下沉【GIL 文档已提及，需深化】

**问题**：`PoolKey.to_string()` / `LayerPoolKey.to_string()` 用 f-string 逐块拼接，是 Python 层热路径主要开销。

**证据**：
- [config_data.py:114-124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L114-L124) `PoolKey.to_string()` f-string 拼接
- [config_data.py:161-171](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L161-L171) `LayerPoolKey.to_string()` 重复实现
- 已有部分优化：[config_data.py:284-306](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L284-L306) `_key_prefix_cache` 缓存前缀，但只覆盖部分路径

**优化方向**：
- 参考 `LayerBatchBuilder`（[kv_transfer.py:101](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L101)）已用 numpy 向量化的模式
- 把 key 生成整体下沉到 C++ 扩展（配合 P5 的 key 重构）
- `process_token_key_strings_with_block_ids`（[config_data.py:574-600](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L574-L600)）是主调用点，应批量化

**严重程度**：高（GIL 文档已确认这是 Python 层主要瓶颈）

---

### P3. `_generate_store_query_keys` 多层嵌套循环

**问题**：scheduler 侧生成查询 key 时有 5 层嵌套 for 循环，每个 block 都构造 `KeyMetadata` + `PoolKey` 对象。

**证据**：[pool_scheduler.py:247-283](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L283)

```python
for block_hash in block_hashes:              # 第1层
    for pcp_rank in range(self.pcp_size):     # 第2层
        for dcp_rank in range(self.dcp_size): # 第3层
            for head_or_tp_rank in range(head_or_tp_ranks):  # 第4层
                for pp_rank in pp_ranks:     # 第5层
                    pool_key = PoolKey(KeyMetadata(...), chunk_hash)  # 每次构造对象
                    if include_layers:
                        block_keys.extend(layer_key.to_string() for layer_key in pool_key.split_layers(...))
```

**优化方向**：
- `KeyMetadata` 对象应预构造并复用（同 group 的 metadata 不变）
- 用 numpy 批量生成 key 字符串数组，避免逐对象构造
- `split_layers` 逐层构造 `LayerPoolKey` 列表可改成直接批量生成字符串

**严重程度**：中（命中检查频率，但 PCP/DCP 通常为 1）

---

### P4. TP mismatch 路径的重复 lookup

**问题**：写侧 TP mismatch 路径先 `lookup()` 查已有缓存，过滤 missing keys 后再 `put`，存在重复查询。

**证据**：
- 写侧：[pool_worker.py:2005-2035](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2005-L2035) `_store_kv_tp_mismatch` 先 lookup 再 put
- 读侧：[pool_worker.py:1936-1991](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1936-L1991) `_build_tp_mismatch_keys_and_addrs` 双重循环
- 对比非 mismatch 路径也有类似模式：[kv_transfer.py:717-818](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L717-L818) `_handle_stored_request`

**优化方向**：
- 评估 lookup 是否可缓存（同一请求的 key 状态在短时间内稳定）
- 考虑 put 时直接覆盖（mooncake 支持），省掉 lookup

**严重程度**：中（仅 TP mismatch 场景）

---

### P5. GVA 分配循环的对象构造

**问题**：GVA 分配循环里反复构造 key、查询 `_allocated_gvas`、append 列表。

**证据**：[pool_worker.py:1264-1293](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1264-L1293)

**优化方向**：批量化构造 key，用 numpy 数组替代 Python list append。

**严重程度**：低（仅 GVA layerwise + memcache 场景）

---

### P6. 非-GVA layerwise 默认 prefetch=1 导致 I/O 与计算零重叠【已核实，重磅】

**问题**：layerwise 架构的核心价值是"I/O 传输与 attention 计算重叠"，但**非-GVA 路径默认 `layerwise_prefetch_layers=1`**，意味着本层 load 在 `wait_for_layer_load` 调用时才提交并立即等待——**load 与 attention 完全串行，重叠设计完全失效**。对比 GVA 路径默认 prefetch=8，差距巨大。

更细致的限制（已核实）：[pool_worker.py:1683](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1683) `submit_count = self.num_prefetch_layers if self.current_layer == 0 else 1` —— **只有第 0 层用 num_prefetch_layers，其他层固定提交 1 层**。即使配 prefetch=8，首层之后每层只提交 1 层预取。

**证据**：
- 默认值：[pool_worker.py:425](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L425) `int(self._extra_config.get("layerwise_prefetch_layers", 1))`
- 提交逻辑：[pool_worker.py:1683](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1683)
- 等待逻辑：[pool_worker.py:1691-1707](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1691-L1707) 每层 attention 前 `wait_for_layer_load` 阻塞等本层 G2L 完成

**影响**：非-GVA layerwise 模式下，每层 attention 前计算线程都阻塞等网络 I/O，NPU 计算流排空空闲。num_layers 次串行等待，等于把传输延迟加到 forward 延迟上。

**优化方向**：
- **最低成本**：默认值改 2+（GVA 路径已默认 8，非-GVA 应对齐），配合 `submit_count` 逻辑放开首层限制
- **更深**：让 `submit_count` 在所有层都遵循 `num_prefetch_layers`，而非只首层

**严重程度**：高（非-GVA layerwise 路径性能问题的根因，可能是线上最大的单点性能损失）

---

### P7. 非-layerwise 写侧/读侧 I/O 被 per-request 拆分【已核实，高价值】

**问题**：非-layerwise 模式下，**每个 request 单独调一次 `put`（+ 一次 `lookup`）和一次 `get`**，跨 request 完全不合并。一批 N 个 request、G 个 group → N×G 次 put + N×G 次 exists（写侧）/ N 次 get（读侧）。后端 API 原生支持扁平 list，本可一次完成。

**对比**：layerwise GVA 路径已正确合并——[kv_transfer.py:1419-1429](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1419-L1429) 用 `np.concatenate` 合并多 task 后一次 `batch_copy`。[kv_transfer.py:1444-1452](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1444-L1452) `batch_write_finish` 也跨 task 合并。**非-layerwise 路径没有对齐这个合并模式**。

**证据**：
- 写侧：[kv_transfer.py:809](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L809) per-(request,group) `lookup` + [kv_transfer.py:890](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L890) per-(request,group) `put`
- 读侧同步：[pool_worker.py:971](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L971) per-request `get`
- 读侧异步：[kv_transfer.py:997](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L997) per-request `get`
- 入队侧逐 request：[pool_worker.py:1749-1762](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1749-L1762)

**可合并性**：`m_store.put/get` 接受扁平 list，跨 request 拼接 key/addr/size 即可。同批 request 共享 `current_event`（[pool_worker.py:1754](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1754) 创建一次赋给所有 request），合并后 `current_event.synchronize()` 只需一次。`store_mask`/`skip` per-request 不同只影响"选哪些 block"，不影响合并后的调用。

**优化方向**：
- 写侧：跨 request 累积 keys/addrs/sizes，一次 `lookup` + 一次 `put`（参考 GVA 路径的 `batch_write_finish` 合并模式）
- 读侧：跨 request 拼接 key/addr/size，一次 `get`；`_circular_shift` 在合并后的总 list 上做
- 异步读侧：改为 drain 队列里所有就绪 request 一次性处理

**严重程度**：高（I/O 次数膨胀 = 一批 request 数 × group 数，典型 4~32 倍）

---

### P8. layerwise GVA 元数据 RPC 被 per-(request,group) 拆分【已核实】

**问题**：`_alloc_gvas_for_save` 和 `_prepare_load_gvas` 收到的是**整批 requests**，但内部 `for request × for group` 双层循环里，每个 (request, group) 单独调 `batch_alloc`/`batch_is_exist`/`batch_get_key_info`/`batch_add_lease`。

**证据**：
- save 侧 `_alloc_gvas_for_save`：[pool_worker.py:1253](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1253) per-(req,group) `batch_is_exist` + [pool_worker.py:1278](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1278) per-(req,group) `batch_alloc`
- load 侧 `_prepare_load_gvas`：[pool_worker.py:1441](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1441) per-(req,group) `batch_get_key_info` + [pool_worker.py:1466](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1466) per-(req,group) `batch_add_lease`
- partial block 单 key：[pool_worker.py:1310](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1310) `[partial_key]` 单 key alloc

**可合并性**：所有这些 RPC 都接受扁平 key list。key 命名空间全局唯一（含 group_id + block_hash + rank），跨 request 合并无冲突。`_allocated_gvas` 是进程内 dict，合并查询后按偏移拆回各 request。

**优化方向**：收集所有 (request, group) 的 keys 拼成一次调用：
- `batch_is_exist`：所有 cached_keys 一次查
- `batch_alloc`：所有 new_keys 一次 alloc（sizes 是 list，按位置对应不同 group 的 alloc_size）
- `batch_get_key_info` / `batch_add_lease`：同理

**严重程度**：高（`batch_alloc` 非幂等，固定开销大；N×G 次可压成 1~2 次）

---

### P9. `.tolist()` 在 C++ 边界前抵消 numpy 向量化收益【已核实，隐藏拷贝】

**问题**：layerwise GVA 路径用 numpy 向量化精心构建 `addr_arr/size_arr/gvas_arr`，但在传给 C++ 扩展 `batch_copy` 前，[kv_transfer.py:477-481](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L477-L481) 用 `.tolist()` 把每个 ndarray 转回 Python list——为每个元素新建一个 PyLong 对象，**完全抵消了 numpy 向量化的内存效率**，在 C++ 边界前引入一次 O(N) 的 Python 对象分配 + 拷贝。

**证据**：
```python
res = self.m_store.store.batch_copy(
    split_gvas.tolist(),    # ndarray → list[PyLong]
    split_addrs.tolist(),   # ndarray → list[PyLong]
    split_sizes.tolist(),   # ndarray → list[PyLong]
    direction,
)
```
- 调用点：[kv_transfer.py:477-481](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L477-L481)
- 每 layer save+load 各一次，3 个数组 × 2 = 6 次 tolist / forward

**优化方向**：让 `m_store.store.batch_copy`（C++ 扩展）支持 buffer protocol / `memoryview` / 直接传 ndarray，避免 `.tolist()`。若 C++ 侧改动成本高，可传 `ctypes` 数组或 `array.array`。

**严重程度**：高（每 layer 3 个数组的 Python 对象化拷贝，规模 = blocks × caches_per_layer；这是整条向量化流水线末端的"漏点"）

---

### P10. `block_hash_to_str` 对同一 hash 重复转换 3 次【已核实】

**问题**：`_alloc_gvas_for_save` 和 `_prepare_load_gvas` 里，同一个 `group_block_hashes[block_idx]` 在 candidate_keys 列表推导、while 循环、for 循环里被 `block_hash_to_str`（`.hex()`）转换 3 次，产生三个内容相同的临时 str。

**证据**：
- [pool_worker.py:1243-1268](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1243-L1268)（save 侧 3 次转换）
- [pool_worker.py:1422-1425](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1422-L1425)（load 侧同样模式）

**优化方向**：在函数入口处一次性把所需范围的 `group_block_hashes` 转成 `list[str]`，后续循环复用。

**严重程度**：中（长 prompt 下 block 数大，3 倍冗余 hex 转换 + f-string）

---

### P11. `from_request_tracker` 每 step 全量重建 numpy 数组【已核实】

**问题**：scheduler 每 step 对每个 request 调 `from_request_tracker`，把**全量** `allocated_block_ids` 用 `np.asarray` 重新拷贝成新 ndarray。长 prompt（32k token / 16 block_size = 2000 blocks）下，每步每请求 O(N) 拷贝。

**证据**：[config_data.py:1093-1100](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L1093-L1100)

```python
block_ids_np=np.asarray(tracker.allocated_block_ids, dtype=np.int64),
block_ids_by_group_np=[np.asarray(ids, dtype=np.int64) for ids in tracker.allocated_block_ids_by_group]
```

**优化方向**：在 `RequestTracker` 中维护增量 numpy buffer，`update()` 追加 block 时往预分配的 np 数组追写，避免每步全量重建。或缓存上次转换的 (长度, 数组) 只追增量。

**严重程度**：中高（scheduler 热路径，长 prompt 下显著）

---

### P12. `_handle_stored_request` 双重建 + 三遍遍历【已核实】

**问题**：非-layerwise 写侧先 append 构造 5 个列表（starts/ends/keys/block_hashes/key_block_ids），`lookup` 后再用 5 个列表推导按 `missing_indices` 过滤重建 5 个新列表，旧列表丢弃。随后又遍历一次构造 `addrs`/`sizes`。等于对同一数据遍历三次 + 10 次部分拷贝。

**证据**：[kv_transfer.py:766-890](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L766-L890)

**优化方向**：一次遍历同时完成 lookup 过滤和 addrs/sizes 构造，用 numpy mask 一次性索引。

**严重程度**：中（非-layerwise save 热路径，每 forward 每请求每 group）

---

### P13. 非-layerwise `wait_for_save` 的 `request_queue.join()` 阻塞计算线程【已核实】

**问题**：非-layerwise 模式下，`wait_for_save` 提交所有 save task 后调 `send_thread.request_queue.join()`，**阻塞计算线程直到 send 线程处理完所有 save**。这是"计算流阻塞等 I/O"的典型点。

**证据**：[pool_worker.py:1761](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1761)

```python
if current_event is not None:
    send_thread.request_queue.join()   # 阻塞直到所有 save task 处理完
```

**优化方向**：改为异步——不 join，在后续 step 的 `get_finished` 里检查 save 是否完成（delayed free 机制已部分支持，但 `wait_for_save` 仍硬 join）。

**严重程度**：中（非-layerwise 模式下每步 forward 后的主要阻塞点）

---

### P14. 最后一层 `save_kv_layer` 同步等 save 完成【已核实】

**问题**：layerwise 模式下，`save_kv_layer` 在 `current_layer == num_layers - 1` 时，`while not layer_save_finished_events[num_layers-1].wait(timeout=10)` 阻塞计算线程，直到最后一层 save（L2G）提交完成。前面所有层异步，唯独最后一层同步等待。

**证据**：[pool_worker.py:1731-1740](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1731-L1740)

**优化方向**：把等待推迟到 `get_finished` 或下一 step 开始时（类似 non-layerwise 的 delayed free），而不是在 `save_kv_layer` 里硬等。

**严重程度**：中（每步 forward 末尾固定阻塞一次）

---

### P15. ZMQ Lookup RPC 无超时、无批合并【已核实】

**问题**：非-layerwise 模式下，scheduler 对每个新请求调 `LookupKeyClient.lookup()`，同步 ZMQ REQ-REP，**无 timeout、无批合并**——同一 step 多个新请求各自串行 RPC。worker rank0 的 `LookupKeyServer` 单线程处理。layerwise 模式不走 ZMQ，scheduler 本地查。

**证据**：
- [pool_scheduler.py:1175](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1175) `resp = self.socket.recv()` 无 timeout
- [pool_scheduler.py:556](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L556) per-request 调用

**优化方向**：① 加 `socket.poll(timeout)` 超时保护；② 同 step 多请求的 block_hashes 合并成一次批量 RPC；③ 用 DEALER/多 REQ 并发。

**严重程度**：中（非-layerwise 模式下，新请求数多时累积延迟显著）

---

### P16. gate 对非复用预取层过度同步【已核实】

**问题**：layerwise 预取层（`layer_id != current_layer`）的 load task 一律携带 `attention_start_gate`，recv 线程在 `_handle_request` 里 `gate.wait()` → `event.synchronize()` 阻塞直到计算流到达 attention 边界。但对**非 buffer 复用**的预取层（`prefetch_layer_map` 无该层），不存在共享 buffer 数据竞争，gate 是多余的——load 可以立即开始。

**证据**：
- gate 附着：[pool_worker.py:1670-1672](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1670-L1672) 对所有 `layer_id != current_layer` 一律加 gate
- gate wait：[kv_transfer.py:1597-1599](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1597-L1599)
- gate 实现：[memcache_comm_fence.py:53-61](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/memcache_comm_fence.py#L53-L61) `event.synchronize()` 硬阻塞

**影响**：单 recv 线程被 gate 阻塞会卡住后续所有层的 load（recv 线程串行处理队列）。

**优化方向**：按"是否有 reuse_source"决定是否加 gate——非复用层不加 gate，立即 load；复用层保留 gate 保证 buffer 数据安全。

**严重程度**：中（GVA 复用场景 gate 是正确性需要；但非复用预取层被不必要延迟）

---

## 二、代码结构优化点

### S1. 巨文件拆分

**问题**：4 个文件超过 1000 行，职责过重。

**证据**：见概览表。具体：
- `pool_worker.py`（2306 行）：`KVPoolWorker` 承担了初始化、load、save、GVA 分配、TP mismatch、线程管理、lookup server 等多职责
- `config_data.py`（1112 行）：既放数据类（KeyMetadata/PoolKey/LayerPoolKey），又放业务逻辑（ChunkedTokenDatabase），还放 metadata（ReqMeta/RequestTracker/LoadSpec）

**优化方向**：
- `config_data.py` 拆为 `keys.py`（PoolKey 及相关）+ `token_database.py`（ChunkedTokenDatabase）+ `request_meta.py`（ReqMeta/RequestTracker/LoadSpec）
- `pool_worker.py` 按职责拆：`load.py` / `save.py` / `gva.py` / `tp_mismatch.py`，`KVPoolWorker` 只做编排
- `kv_transfer.py` 按线程类型拆分（非layerwise / key layerwise / GVA layerwise 各一文件）

**严重程度**：高（影响可维护性和后续所有改动）

---

### S2. scheduler / worker 初始化逻辑重复

**问题**：`use_mla` 判断、`num_kv_head` 计算、`put_step` 计算、backend 加载、layerwise layout 构建在 scheduler 和 worker 两边各写一遍。

**证据**：
- worker：[pool_worker.py:188-208](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L188-L208) `_init_key_head_config`
- scheduler：[pool_scheduler.py:183-193](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L183-L193) 内联在 `__init__`
- 两边都独立做 `infer_tp_mismatch_info`、backend_map 查找、layerwise layout 构建

**优化方向**：抽取 `KVPoolConfig` 共享配置类，scheduler 和 worker 都从它读取，避免逻辑漂移（一边改了另一边没改）。

**严重程度**：高（一致性风险，已有 MLA 分析中就因为两边逻辑需对照确认）

---

### S3. `to_string` 重复实现

**问题**：`PoolKey.to_string()` 和 `LayerPoolKey.to_string()` 是两段几乎相同的 f-string，只是后者多 `@layer_id`。

**证据**：[config_data.py:114-124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L114-L124) 和 [config_data.py:161-171](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L161-L171)

**优化方向**：`LayerPoolKey.to_string()` 复用父类前缀 + 追加 layer_id；或整体重构为统一 key builder（配合 P2）。

**严重程度**：低（但会增加 P2 向量化改造的维护成本）

---

### S4. ChunkedTokenDatabase 职责过重

**问题**：`ChunkedTokenDatabase` 承担了 token 分块、key prefix 缓存、buffer/stride 管理、layer cache 准备等多职责。

**证据**：[config_data.py:255-467](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L255-L467)

**优化方向**：拆分为 `TokenChunker`（token→chunk 映射）+ `KeyBuilder`（chunk→key）+ `BufferLayout`（key→addr/size）。

**严重程度**：中

---

### S5. 配置项散落，无集中 schema

**问题**：15 处 `extra_config.get` 散落在 3 个文件里，无集中定义，新增配置项时无法发现已有哪些，易冲突或重复。

**证据**：见概览表统计。典型散落点：
- [pool_worker.py:145-177](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L177)
- [pool_scheduler.py:93-105](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L93-L105)

**优化方向**：定义 `KVPoolConfig` dataclass 集中所有配置项（含默认值、类型、文档），scheduler/worker 都从它读取。这也解决 S2 的重复初始化问题。

**严重程度**：高（演进阻力大，易引入配置不一致）

---

## 三、正确性与潜在 Bug 风险

### C1. 传输线程异常路径不清理资源【已核实】

**问题**：`KVTransferThread.run()` 捕获异常后直接 `return`，设置 `_fatal_error`，但**没有调用 `_handle_request_exception`，没有 `task_done()`，也没有释放 lease / 设置 event**，可能导致：
- `task_done()` 未调用 → `queue.join()` 永久阻塞
- `layer_load_finished_events` 永不 set → 等待方死锁
- lease 未释放 → memcache 资源泄漏

**证据**：
- [kv_transfer.py:510-518](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L510-L518) 异常分支直接 return
- [kv_transfer.py:523-525](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L523-L525) `_handle_request_exception` 基类是空实现，且 run 异常分支没调用它
- 子类如 [kv_transfer.py:670-677](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L670-L677) 实现了 `_handle_request_exception` 但仅在正常路径的 `task_done` 里调用

**优化方向**：`run()` 的 except 分支应：①调用 `self._handle_request_exception(request_data)` ②尝试 set 所有相关 event ③记录失败状态供上层查询。

**严重程度**：高（触发即死锁/泄漏，虽然触发概率低）

---

### C2. 多 group 加载失败的错误传播

**问题**：layerwise 多 group 加载失败时释放 lease 并抛 RuntimeError，但注释明说多 group 不能安全回退到 per-block recomputation，失败处理不完整。

**证据**：
- [pool_worker.py:1509-1534](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1509-L1534) 抛异常
- [kv_transfer.py:923-1039](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L923-L1039) `KVCacheStoreRecvingThread._handle_request` 单 group 更新 invalid blocks，多 group 只记日志跳过

**优化方向**：多 group 场景需要明确的失败恢复策略（降级重算？整请求失败？），不能只记日志。

**严重程度**：中（多 group 场景目前较少，但 DSV4 hybrid 会增多）

---

### C3. `_invalid_block_ids` 锁保护范围

**问题**：`_invalid_block_ids` 有锁保护，但读取它的路径（如 `get_block_ids_with_load_errors`）是否都在锁内需核实。

**证据**：[pool_worker.py:145-152](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L152) 定义锁，[pool_worker.py:570-571](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L570-L571) 传给 recv 线程。

**优化方向**：审计所有 `_invalid_block_ids` 访问点，确保读写都在锁内。

**严重程度**：中

---

### C4. `_iter_token_chunks` 边界条件

**问题**：`_iter_token_chunks` 有较多边界判断（空 block_hashes、token_len<=0、block_ids 越界），虽然已有防护，但 `block_id_offset` 计算复杂，极端组合下可能越界。

**证据**：[config_data.py:482-514](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L482-L514)

**优化方向**：补充单元测试覆盖边界组合（空 hash、token_len=0、block_ids 比 hash 少等）。

**严重程度**：低（已有防护，但缺测试保障）

---

## 四、可扩展性与接口设计

### E1. Backend 抽象基类泄漏 memcache 概念

**问题**：`Backend` 基类定义了 `batch_alloc / batch_add_lease / batch_remove_lease / batch_get_key_info / batch_write_finish`，这些是 memcache GVA 专属概念，但放在基类里默认抛 `NotImplementedError`，污染了抽象。

**证据**：[backend.py:32-48](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py#L32-L48)

**优化方向**：拆分为 `Backend`（最小接口：put/get/exists/register_buffer）+ `GVABackend`（继承，增加 lease/alloc 接口）。worker 侧用 `isinstance` 或能力查询判断是否走 GVA 路径。

**严重程度**：中（新增后端时基类方法会持续膨胀）

---

### E2. backend_map 硬编码，不支持外部注册

**问题**：`backend_map` 字典硬编码三个后端，新增后端需改源码。

**证据**：[backend/__init__.py:17-30](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L17-L30)

**优化方向**：支持 entry_point 或注册函数机制，让外部包能注册新后端。

**严重程度**：低（目前后端较少，但影响第三方扩展）

---

### E3. 缺少单元测试

**问题**：扫描未发现 test 目录，关键逻辑（PoolKey / ChunkedTokenDatabase / coordinator / 边界条件）缺测试保障。

**证据**：kv_pool 目录下无 `test_*.py` 或 `tests/` 目录。

**优化方向**：优先为 `config_data.py`（key 生成、token 分块）和 `coordinator.py`（hybrid 命中）补单元测试，这两块是纯逻辑、易测试、易回归。

**严重程度**：高（重构无安全网，阻碍所有其他优化落地）

---

### E4. Connector 间无公共基类

**问题**：4 个 connector 各自实现 `KVConnectorBase_V1`，但无 kv_pool 层面的公共基类抽取公共逻辑。

**证据**：4 个 connector 入口（ascend_store / recompute_cpu_offload / simple_cpu_offload / ucm_connector）无共同父类。

**优化方向**：评估是否有可抽取的公共逻辑（如配置解析、event 聚合）。若各 connector 差异大，可不强行抽取，但应文档化各 connector 的定位和边界。

**严重程度**：低（4 个 connector 差异较大，强行抽取可能过度设计）

---

## 五、优先级与推进建议

### 优先级矩阵

按"收益 × 紧迫度 × 落地难度"排序：

| 编号 | 优化点 | 维度 | 收益 | 紧迫度 | 难度 | 建议优先级 |
|------|--------|------|------|--------|------|-----------|
| E3 | 补单元测试 | 扩展性 | 高 | 高 | 低 | **P0 先行** |
| C1 | 线程异常路径清理 | 正确性 | 高 | 高 | 中 | **P0** |
| S5 | 配置项集中 schema | 结构 | 高 | 高 | 中 | **P0** |
| **P6** | **非-GVA prefetch=1 重叠失效** | **性能** | **高** | **高** | **低（改默认值）/ 中（改 submit 逻辑）** | **P0**（改默认值即可见效）|
| **P7** | **非-layerwise I/O per-request 拆分** | **性能** | **高** | **中** | **中** | **P1** |
| **P8** | **GVA 元数据 RPC per-(req,group) 拆分** | **性能** | **高** | **中** | **中** | **P1** |
| **P9** | **.tolist() 抵消向量化收益** | **性能** | **高** | **中** | **高（需改 C++ 扩展）** | **P1** |
| S1 | 巨文件拆分 | 结构 | 高 | 中 | 高 | P1 |
| S2 | scheduler/worker 初始化去重 | 结构 | 高 | 中 | 中 | P1 |
| P1 | MLA 读侧去重 | 性能 | 高 | 中 | 中 | P1（已有方案）|
| P2 | Key 生成向量化 | 性能 | 高 | 中 | 中 | P1 |
| **P11** | **from_request_tracker 全量重建** | **性能** | **中高** | **中** | **中** | **P1** |
| E1 | Backend 抽象拆分 | 扩展性 | 中 | 中 | 中 | P2 |
| C2 | 多 group 失败传播 | 正确性 | 中 | 中 | 中 | P2 |
| P3 | 嵌套循环优化 | 性能 | 中 | 低 | 中 | P2 |
| C3 | _invalid_block_ids 锁审计 | 正确性 | 中 | 中 | 低 | P2 |
| **P10** | **block_hash_to_str 重复转换 3 次** | **性能** | **中** | **中** | **低** | **P2** |
| **P12** | **_handle_stored_request 双重建** | **性能** | **中** | **低** | **中** | **P2** |
| **P13** | **非-layerwise wait_for_save 阻塞** | **性能** | **中** | **中** | **中** | **P2** |
| **P14** | **最后一层 save 同步等待** | **性能** | **中** | **中** | **中** | **P2** |
| **P15** | **ZMQ Lookup 无超时无批合并** | **性能** | **中** | **中** | **低（加 timeout）/ 中（批合并）** | **P2** |
| **P16** | **gate 对非复用预取层过度同步** | **性能** | **中** | **低** | **中** | **P2** |
| S3/S4 | to_string 去重 / DB 拆分 | 结构 | 中 | 低 | 中 | P3 |
| P4/P5 | TP mismatch / GVA 循环优化 | 性能 | 中 | 低 | 中 | P3 |
| C4 | 边界条件测试 | 正确性 | 低 | 低 | 低 | P3 |
| E2/E4 | backend 注册 / connector 基类 | 扩展性 | 低 | 低 | 中 | P3 |

### 推进策略

**第一阶段（打好地基 + 立即见效，P0）**：
1. **E3 补测试**——为 config_data 和 coordinator 补单元测试，建立重构安全网
2. **C1 修异常路径**——修复线程异常路径的资源泄漏/死锁风险
3. **S5 配置集中**——定义 KVPoolConfig，为后续重构铺路
4. **P6 改 prefetch 默认值**——最低成本高收益，非-GVA 路径默认 1 改 2+，并评估放开 `submit_count` 首层限制

**第二阶段（性能 + 结构双推进，P1）**：
5. **S5 落地后做 S2**——scheduler/worker 共享配置，消除重复初始化
6. **P7 非-layerwise I/O 合并**——跨 request 累积 keys/addrs/sizes，一次 put/get（参考 GVA 路径的 batch_write_finish）
7. **P8 GVA 元数据 RPC 合并**——收集所有 (req,group) 的 keys 一次调用
8. **P1 MLA 读侧去重**——已有方案，先做 PoC 验证（见专项文档）
9. **P2 Key 生成向量化**——配合 S1 拆分一起做，先抽 KeyBuilder
10. **P9 消除 .tolist()**——让 C++ 扩展支持 buffer protocol（需跨团队协调 C++ 改动）
11. **P11 RequestTracker 增量 buffer**——维护增量 numpy buffer，避免每步全量重建

**第三阶段（深化，P2/P3）**：
12. S1 巨文件拆分（依赖 S5/S2 先落地）
13. P10/P12 内存拷贝优化（block_hash 缓存、双重建消除）
14. P13/P14/P16 同步等待优化（异步化 wait_for_save、推迟最后一层等待、gate 精细化）
15. P15 ZMQ 超时 + 批合并
16. E1 Backend 抽象拆分
17. 其余按需推进

### 关键依赖关系

- S5（配置集中）是 S2（初始化去重）的前置
- E3（补测试）是 S1（文件拆分）和 P7/P8（I/O 合并重构）的前置——没有测试网重构风险极高
- P2（key 向量化）和 S1（config_data 拆分）应协同——先拆 keys.py 再做向量化
- P1（MLA 去重）独立，可并行推进
- P9（消除 .tolist()）依赖 C++ 扩展支持 buffer protocol，需跨团队协调
- P6（prefetch 默认值）独立，可立即见效，无前置依赖
- P7（非-layerwise I/O 合并）与 P8（GVA 元数据合并）可同步推进，模式相同

---

## 六、与已有文档的关系

| 已有文档 | 对应本文优化点 | 关系 |
|----------|----------------|------|
| [kv_pool_线程存取与GIL分析.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_线程存取与GIL分析.md) | P2（key 向量化）、结构梳理部分 | 已有分析是本文 P2 的基础；本文补充了更多结构/正确性/扩展性维度 |
| [MLA_KV读取去重优化讨论.md](file:///D:/lzy/project/kv_pool/llm-project/draft/MLA_KV读取去重优化讨论.md) | P1（MLA 读侧去重） | 已有专项方案；本文作为索引纳入全局优先级 |

本文不重复已有文档的细节，而是把它们纳入一个**全局优化点清单 + 优先级框架**，方便统筹推进。

---

## 附录：四维度扫描的代码命中索引

### 性能维度

| 命中点 | 文件:行号 |
|--------|-----------|
| PoolKey.to_string f-string | config_data.py:[114-124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L114-L124) |
| LayerPoolKey.to_string 重复 | config_data.py:[161-171](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L161-L171) |
| _generate_store_query_keys 嵌套循环 | pool_scheduler.py:[247-283](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L283) |
| process_token_key_strings 迭代器 | config_data.py:[526-600](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L526-L600) |
| _handle_stored_request 写侧循环 | kv_transfer.py:[717-818](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L717-L818) |
| _build_tp_mismatch_keys_and_addrs | pool_worker.py:[1936-1991](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1936-L1991) |
| _store_kv_tp_mismatch 重复 lookup | pool_worker.py:[2005-2035](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2005-L2035) |
| _refresh_allocated_gvas | pool_worker.py:[1176-1191](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1176-L1191) |
| GVA 分配循环 | pool_worker.py:[1264-1293](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1264-L1293) |

### 结构维度

| 命中点 | 文件:行号 |
|--------|-----------|
| KeyMetadata/PoolKey/LayerPoolKey 混杂 | config_data.py:[71-170](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L71-L170) |
| ChunkedTokenDatabase 职责重 | config_data.py:[255-467](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L255-L467) |
| KVPoolWorker 初始化重 | pool_worker.py:[93-233](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L93-L233) |
| KVPoolScheduler 初始化重复 | pool_scheduler.py:[53-224](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L53-L224) |
| layerwise 配置解析分散 | layerwise_cache_layout.py:[62-169](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py#L62-L169) |
| Backend 抽象层 | backend.py:[8-55](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py#L8-L55) |
| MemcacheBackend 实现 | memcache_backend.py:[41-160](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L41-L160) |

### 正确性维度

| 命中点 | 文件:行号 |
|--------|-----------|
| _invalid_block_ids 锁 | pool_worker.py:[145-152](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L152) |
| key 分片除零/越界风险 | pool_worker.py:[188-208](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L188-L208) |
| 线程启动 Event 同步 | pool_worker.py:[540-574](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L540-L574) |
| KVTransferThread.run 异常路径 | kv_transfer.py:[496-518](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496-L518) |
| RecvingThread._handle_request 错误传播 | kv_transfer.py:[923-1039](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L923-L1039) |
| 同步 load 失败处理 | pool_worker.py:[971-1001](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L971-L1001) |
| layerwise 多 group 失败 | pool_worker.py:[1509-1534](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1509-L1534) |
| _iter_token_chunks 边界 | config_data.py:[482-514](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L482-L514) |

### 扩展性维度

| 命中点 | 文件:行号 |
|--------|-----------|
| Backend 基类接口过载 | backend.py:[8-55](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py#L8-L55) |
| backend_map 硬编码 | backend/__init__.py:[17-30](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L17-L30) |
| worker 配置散落 | pool_worker.py:[145-177](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L177) |
| scheduler 配置散落 | pool_scheduler.py:[93-105](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L93-L105) |
| UCMConnector 转发壳 | ucm_connector.py:[37-69](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector.py#L37-L69) |
