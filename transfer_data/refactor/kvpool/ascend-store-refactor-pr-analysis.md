# AscendStore KV Pool 重构 PR 详细分析

## 1. PR 概览

| 项目 | 内容 |
|------|------|
| 分支 | `ascend-store-refactor` → `upstream/main` |
| 规模 | 13 文件,+983 / −2,960(净减约 1,977 行) |
| 提交数 | 2 |
| 基线 | `upstream/main`(三点 diff) |

### 这个 PR 在做什么

本 PR 对 AscendStore KV pool 模块做一次**实现简化与测试瘦身**,核心是三件事:

1. **职责下沉与去重**:把原本散落在 `KVPoolWorker` / `KVPoolScheduler` / `AscendStoreCoordinator` 中的 cache family / block size 推算逻辑,统一下沉为 `metadata.py` 中的模块级纯函数(6 个),消除三处重复实现,并使这些逻辑可独立单测。同时用上游标准函数 `cdiv` 替换手写的 `_num_chunks`。

2. **GVA 分配职责迁移(行为变更)**:把 GVA(global virtual address)的预分配从 `KVPoolScheduler` 移除,改由 `KVPoolWorker` 侧的 per-rank `batch_alloc` 承接。配套删除 scheduler 上 `_allocate_gva_if_needed` 调用、`block_keys` / `last_block_key` / `_unfinished_request_ids` 等不再需要的字段与传递路径。**这是本 PR 唯一的运行时行为变更,非纯文本重构。**

3. **冗余代码清理**:删除线程子类中与基类等价的 `add_request` / `dec_stored_request` 覆盖、未使用的 thread 构造参数(`put_step` / `my_key_index` / `num_ranks_per_layer`)、`MemcacheBackend.init_store` 等无调用点的方法、connector 层冗余字段、`ReqMeta` 上未使用的 `starts` / `ends` / `sizes_per_chunk` / `kv_cache_families_by_group` 等字段。

4. **测试重构**:引入公共 mock 基础设施(`_mock_deps.py`)与工厂函数(`make_config` / `make_worker` / `start_patch`),合并细碎数据类测试,针对新提取纯函数新增聚焦单测,测试代码净减约 1,637 行。

一句话:**在不改变对外接口的前提下,通过职责下沉、GVA 迁移与冗余清理,让 AscendStore KV pool 实现更内聚、测试更轻量。**

### 提交列表

1. `11bc84f32` `[Refactor] Simplify AscendStore KV pool Implementation and Unit Tests`
   - 主体重构提交,完成上述 1–4 项变更,Co-Authored-By: Claude
2. `c6f72551f` `refactor: port AscendStore KV pool simplification`
   - 将上述重构 rebase 到最新 `upstream/main` 之上,在 rebase 过程中保留 upstream 新引入的 layerwise 与 multi-buffer 契约,并合并已 review 的测试简化与 Python 3.10 补丁清理,使 PR 可基于当前主干合入。**此提交本身不引入新功能,只做 rebase 适配。**

### 按文件变更统计(numstat)

| 文件 | + | − | 净 |
|------|---|---|----|
| `vllm_ascend/.../ascend_store/metadata.py` | 55 | 69 | −14 |
| `vllm_ascend/.../ascend_store/pool_scheduler.py` | 25 | 187 | −162 |
| `vllm_ascend/.../ascend_store/pool_worker.py` | 33 | 105 | −72 |
| `vllm_ascend/.../ascend_store/kv_transfer.py` | 2 | 67 | −65 |
| `vllm_ascend/.../ascend_store/coordinator.py` | 3 | 14 | −11 |
| `vllm_ascend/.../ascend_store/ascend_store_connector.py` | 4 | 12 | −8 |
| `vllm_ascend/.../ascend_store/backend/memcache_backend.py` | 0 | 8 | −8 |
| `tests/.../ascend_store/test_pool_worker.py` | 327 | 948 | −621 |
| `tests/.../ascend_store/test_pool_scheduler.py` | 179 | 705 | −526 |
| `tests/.../ascend_store/test_kv_transfer.py` | 104 | 261 | −157 |
| `tests/.../ascend_store/test_metadata.py` | 106 | 250 | −144 |
| `tests/.../ascend_store/test_backend.py` | 77 | 181 | −104 |
| `tests/.../ascend_store/test_ascend_store_connector.py` | 68 | 153 | −85 |

源码 7 文件净减约 340 行,测试 6 文件净减约 1,637 行。

---

## 2. 变更分类详解

### A. 辅助函数下沉到 `metadata.py`

**目标**:把散落在 `KVPoolWorker` / `KVPoolScheduler` 中的私有静态方法,提升为 `metadata.py` 中的模块级纯函数,消除重复、便于复用与单测。

**新增模块级函数**(`metadata.py` L194–L273):

| 函数 | 职责 |
|------|------|
| `uses_hybrid_kv_cache(scheduler_config, kv_cache_groups)` | 判断是否启用混合 KV cache |
| `infer_group_block_sizes(cache_block_size, kv_cache_groups, use_hybrid)` | 推断各 group 的 block size 列表 |
| `get_group_cache_family(group_cache_families, group_id)` | 按 group_id 取 cache family,越界返回 "default" |
| `get_effective_group_block_size(group_block_sizes, group_cache_families, group_id)` | 计算考虑 family 压缩比的有效 block size |
| `infer_cache_transfer_granularity(group_block_sizes, group_cache_families)` | 计算所有 group 粒度的 LCM |
| `infer_group_cache_families(kv_cache_groups, compress_ratios, hf_config)` | 根据 compress ratio 推断各 group 的 cache family |

**特征**:均为纯函数,无副作用,输入全部显式传参,不依赖 `self`,可独立单测。

**对应删除的私有方法**:
- `KVPoolWorker._get_group_family`
- `KVPoolWorker._get_group_block_size`
- `KVPoolWorker._get_effective_group_block_size`
- `KVPoolWorker._infer_cache_transfer_granularity`
- `KVPoolWorker._uses_hybrid_kv_cache`
- `KVPoolScheduler` 中对应重复实现

**调用点改写**:`pool_worker.py` / `pool_scheduler.py` 中原 `self._get_xxx()` 调用全部改为从 `metadata` 导入的模块函数;grep 确认无残留私有方法引用。

### B. `coordinator.py` 重复函数清理

**删除的本地重复实现**:
- `_cache_family_granularity(block_size, cache_family)` → 改用 `metadata.get_cache_family_granularity`
- `_num_chunks(token_len, block_size)` → 改用 `vllm.utils.math_utils.cdiv`

**效果**:消除 `coordinator.py` 与 `metadata.py` 之间 `_cache_family_granularity` 的重复定义;`_num_chunks` 是 `cdiv` 的手写版,替换为上游标准函数。

### C. `pool_scheduler.py` GVA 预分配逻辑移除 + 未使用字段清理

这是本 PR **行为相关变更最集中**的部分,不是纯文本重构。

**C.1 删除 `_allocate_gva_if_needed` 调用**(3 处:新增请求、恢复请求、chunked prefill)

删除代码块示例:
```python
# 已删除
num_blocks = num_tokens_to_compute // self.hash_block_size
has_last_block = num_tokens_to_compute % self._block_size != 0
self._allocate_gva_if_needed(
    request_tracker,
    request_real.block_hashes,
    num_blocks,
    has_last_block,
)
```

**依据**:被删代码注释明确写道 "GVA allocation moved to worker side (per-rank batch_alloc); scheduler only generates block keys here."。即 GVA 分配职责已迁移到 worker 侧,scheduler 不再需要预分配。grep 确认 `_allocate_gva_if_needed` 全代码库无残留调用。

**C.2 删除 `RequestTracker.block_keys` / `last_block_key` 字段传递**

scheduler 不再向 `RequestTracker` 传递 `block_keys`、不再读取 `request_tracker.block_keys` / `.last_block_key`。grep 确认 `.block_keys` / `.last_block_key` 字段访问全代码库无残留(注意:pool_scheduler.py 中仍存在名为 `block_keys` 的局部变量,与字段无关)。

**C.3 删除 `_unfinished_request_ids` 集合及其传递**

- 删除 `self._unfinished_request_ids` 字段维护
- 删除 `AscendConnectorMetadata` 构造参数 `unfinished_request_ids`
- 删除 `scheduler_output.finished_req_ids` 时对该集合的 `discard` 操作

grep 确认 `.unfinished_request_ids` 全代码库无残留。

**C.4 删除 chunked prefill 中的 block key 生成逻辑**

在 `update_request` 路径中,删除了基于 `hash_block_size` 计算 `new_hash_count` 并调用 `self.generate_keys` 更新 `request_tracker.block_keys` 的整段逻辑(约 18 行)。与 C.1/C.2 一致,属于 GVA/key 管理职责迁移到 worker 的配套清理。

### D. `metadata.py` 数据类字段精简

**D.1 `RequestTracker`**
- 删除 `@staticmethod from_new_request(new_request, num_tokens_to_compute)` —— 全代码库无 `RequestTracker.from_new_request` 调用

**D.2 `ReqMeta`**
- 删除字段 `kv_cache_families_by_group: list[str] | None`
- 删除字段 `starts: list[int] | None`、`ends: list[int] | None`、`sizes_per_chunk: list[list[int]] | None`
- 对应 `__post_init__` 中的赋值同步删除
- grep 确认 `.kv_cache_families_by_group`、`.sizes_per_chunk` 全代码库无残留

**D.3 `AscendConnectorMetadata`**
- 删除构造参数 `unfinished_request_ids` 及 `self.unfinished_request_ids` 字段(与 C.3 配套)

**D.4 `AscendStoreKVConnectorWorkerMetadata`**
- 删除方法 `mark_completed_events(event_id)` —— grep 确认 `.mark_completed_events` 全代码库无残留

### E. `kv_transfer.py` 线程类参数/方法精简

**E.1 删除子类重复覆盖的 `add_request`**

三个子类删除了仅做透传的 `add_request`:
- `KVCacheStoreKeyLayerSendingThread.add_request`
- `KVCacheStoreKeyLayerRecvingThread.add_request`
- `KVCacheStoreLayerSendingThread.add_request`
- `KVCacheStoreLayerRecvingThread.add_request`

这些覆盖仅执行 `self.request_queue.put(req_meta)`,与基类 `KVTransferThread.add_request` 等价,删除后行为不变。

**E.2 删除 `KVCacheStoreLayerSendingThread.dec_stored_request` 子类覆盖**

该子类覆盖与基类 `KVTransferThread.dec_stored_request`(kv_transfer.py L378)实现完全相同,删除后调用通过继承命中基类。pool_worker.py L2004 的 `send_thread.dec_stored_request(req_id)` 仍正常工作。

**E.3 Thread 构造参数精简**

`KVCacheStoreLayerSendingThread` / `KVCacheStoreLayerRecvingThread` 构造签名移除:
- `put_step`
- `my_key_index`
- `num_ranks_per_layer`

`LayerBatchBuilder` 构造签名同步移除 `my_key_index`、`num_ranks_per_layer`。

调用点(pool_worker.py L437/L458/L495)已同步不再传参,grep 确认无 `my_key_index=` / `num_ranks_per_layer=` 残留调用。

**E.4 删除 `KVCacheStoreLayerSendingThread` 未使用字段**

- `self.put_step`
- `self.stored_requests: defaultdict[str, int]`
- `self.done_task_lock: threading.Lock`
- 对应的 `add_stored_request` / `dec_stored_request` 子类方法

注意:基类 `KVTransferThread` 仍保留 `stored_requests` / `done_task_lock` / `add_stored_request` / `dec_stored_request`,子类只是去除了重复声明。

### F. `ascend_store_connector.py` 字段精简

**删除的字段/逻辑**:
- `self.backend_name` / `self.use_gva_layerwise` / `self.consumer_is_to_put`(connector 层)—— 这些字段在 `pool_worker.py` / `pool_scheduler.py` 中各自独立计算,connector 层不需要
- `self.kv_caches: dict[str, torch.Tensor] = {}` —— 未使用
- `page_size_bytes` 参数传递 —— `KVPoolScheduler` 构造不再需要

**职责归位**:`backend_name` / `use_gva_layerwise` 的计算保留在 worker(L149–150)和 scheduler(L163–164),各自独立。connector 不再持有冗余副本。

### G. `memcache_backend.py` 未使用方法删除

**删除**:
```python
def init_store(self, init_bm: bool = True):
    if self.store is not None:
        return
    self._init_bm = init_bm
    self.store = self._setup_store()
    self._store_initialized = True
    self._register_buffers_if_needed()
```

grep 确认 `.init_store(` 全代码库无残留调用。`MemcacheBackend` 的初始化通过其他路径(如 `__init__` 或 `from_config`)完成。

### H. `pool_worker.py` 死字段保留(瑕疵)

**未清理的字段**(`pool_worker.py` L204–209):

```python
self.my_key_index = (
    self.pcp_rank * self.dcp_size * (self.tp_size // self.put_step)
    + self.dcp_rank * (self.tp_size // self.put_step)
    + self.head_or_tp_rank
)
self.num_ranks_per_layer = self.pcp_size * self.dcp_size * (self.tp_size // self.put_step)
```

PR 删除了这两个字段的所有下游消费点(thread 构造、`LayerBatchBuilder` 构造),但 Worker 上的赋值保留。全代码库 grep 确认:除这两行赋值外,`self.my_key_index` / `self.num_ranks_per_layer` 再无任何读取处,已成死字段。测试中同样无引用。

这是本 PR 唯一明显的清理不彻底之处。

### I. 测试基础设施重构

**I.1 引入公共 mock 基础设施**

新增 `tests/ut/distributed/ascend_store/_mock_deps.py`:统一 mock `torch` / `torch_npu` / `vllm` 重依赖。各测试文件顶部 `import tests.ut.distributed.ascend_store._mock_deps` 复用,消除每文件重复的 mock 样板。

适用场景:纯逻辑/数据类测试(metadata、scheduler 静态方法)。对需要真实 torch 计算的测试不适用。

**I.2 提取公共工厂函数**

- `make_config(kv_role, extra_config, block_size)` —— 构造 `VllmConfig` 替身
- `make_worker(...)` —— 构造 `KVPoolWorker` 替身
- `start_patch(test, *args, **kwargs)` —— 统一 patch 生命周期管理

减少每个用例重复构造对象。

**I.3 测试类重组合并**

把细碎的数据类测试合并:
- `TestKeyMetadata` + `TestLayerPoolKey` + `TestLoadSpec` + `TestAscendConnectorMetadata` + `TestLayerMultiBlockReqMeta` → `TestCacheLayoutHelpers`
- 移除冗余/过时用例

**I.4 针对新提取的纯函数新增聚焦单测**

对应 `TestCacheLayoutHelpers` 等,覆盖原本藏在 Worker 内部、现在下沉到 `metadata.py` 的逻辑。

---

## 3. 行为等价性分析

| 变更类别 | 行为等价性 | 依据 |
|----------|-----------|------|
| A. 函数下沉 | 等价 | 纯函数搬迁,函数体未变,调用点逐一改写 |
| B. coordinator 重复函数清理 | 等价 | `get_cache_family_granularity` 与原 `_cache_family_granularity` 逻辑一致;`cdiv` 与 `_num_chunks` 语义一致 |
| C. GVA 预分配逻辑移除 | **行为迁移** | 注释明确 "moved to worker side (per-rank batch_alloc)";scheduler 不再预分配,worker 侧承接。**需确认 worker 侧 `batch_alloc` 实现完整覆盖原逻辑** |
| D. 数据类字段精简 | 等价(若 C 已验证) | 所有被删字段 grep 无残留引用;`from_new_request`、`mark_completed_events` 无调用 |
| E. 线程类参数/方法精简 | 等价 | `add_request` 覆盖与基类等价;`dec_stored_request` 子类覆盖与基类一致;构造参数删除后调用点同步 |
| F. connector 字段精简 | 等价 | 被删字段在 connector 层未使用;worker/scheduler 各自独立计算 |
| G. memcache_backend 方法删除 | 等价 | `init_store` 无调用点 |
| H. pool_worker 死字段 | 等价(但未清理) | 字段赋值保留但无读取,不影响行为 |
| I. 测试重构 | 等价 | mock + 工厂 + 合并,覆盖新提取函数 |

**关键风险点**:C 类变更不是纯重构,是 GVA 分配职责从 scheduler 迁移到 worker。PR 标题虽为 "Simplify",但此处涉及运行时路径变更,需通过集成测试验证 worker 侧 `batch_alloc` 是否完整承接。

---

## 4. 风险点

1. **GVA 分配迁移未在 PR 描述中显式说明**:C 类变更属于行为迁移,但 commit message 仅写 "Simplify Implementation and Unit Tests"。建议在 PR 描述中补充说明 GVA 分配职责迁移的背景与验证方式。

2. **死字段 `my_key_index` / `num_ranks_per_layer` 未清理**:见 §2.H。低风险,但重构不彻底,违背 AGENTS.md「显式传递依赖、避免冗余状态」原则。

3. **测试 mock 化的覆盖度**:`_mock_deps.py` mock 了 torch/torch_npu,适用于纯逻辑测试。但 `pool_worker.py` / `pool_scheduler.py` 中涉及 torch tensor 操作的路径若被 mock 掉,可能掩盖真实集成问题。需确认哪些测试仍走真实 torch 路径。

4. **`AscendConnectorMetadata.unfinished_request_ids` 删除的下游影响**:虽然 grep 在本仓库无残留,但若 vLLM 上游或其他插件读取该字段,会引发运行时错误。需确认 `AscendConnectorMetadata` 的消费方仅限本仓库。

---

## 5. 改进建议

1. **补清理死字段**:删除 `pool_worker.py` L204–209 的 `self.my_key_index` / `self.num_ranks_per_layer` 赋值。若 `self.put_step` 在删除这两处后也无引用,一并清理(需单独 grep 确认,因 L209 还用到 `self.put_step`)。

2. **PR 描述补充 GVA 迁移说明**:在 "What this PR does" 中明确说明 GVA 分配从 scheduler 迁移到 worker 的 per-rank batch_alloc,并给出验证方式(如特定模型的 layerwise + memcache backend 集成测试)。

3. **确认 `unfinished_request_ids` 无跨仓库消费**:若 `AscendConnectorMetadata` 被 vLLM 上城框架反射或序列化,需额外确认。

4. **测试分层**:在 `_mock_deps.py` 顶部注释中明确哪些测试适用 mock 路径(纯逻辑)、哪些必须走真实 torch(集成/数值),避免误用。

---

## 6. 总结评价

### 合理之处

- **函数下沉方向正确**:把 cache family / block size 相关辅助逻辑从 Worker/Scheduler 下沉到 `metadata.py` 作为纯函数,是标准的去重与可测性提升手法。
- **重复清理干净**:coordinator 与 metadata 的 `_cache_family_granularity` 重复、`_num_chunks` 与 `cdiv` 重复、子类 `add_request`/`dec_stored_request` 与基类重复,均通过 grep 验证删除无残留。
- **测试精简策略合理**:公共 mock + 工厂函数 + 测试类合并,净减约 1,637 行测试代码,同时针对新提取纯函数新增聚焦单测,方向正确。
- **构造参数精简一致**:thread / `LayerBatchBuilder` 三处构造签名与调用点同步精简,无遗漏。

### 需要改进

- **死字段 `my_key_index` / `num_ranks_per_layer` 未清理**(§2.H),低风险收尾建议补上。
- **GVA 分配迁移属行为变更,但 PR 描述未显式说明**(§4.1),建议补充。

### 整体结论

方向、手法、测试策略整体合理,净减约 2k 行且大部分变更经 grep 验证行为等价。唯一需要重点验证的是 C 类 GVA 分配迁移(行为变更,非纯重构),建议通过集成测试确认 worker 侧 `batch_alloc` 完整承接后合入;H 类死字段建议合入前顺手清理。
