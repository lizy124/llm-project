# issue #14148 修复设计方案：`_alloc_gvas_for_save` 消除 `block_hash_to_str` 重复转换

- 关联 issue：[vllm-project/vllm-ascend#14148](https://github.com/vllm-project/vllm-ascend/issues/14148)
- 前置分析：[issue_14148_analysis.md](issue_14148_analysis.md)（问题确认，2026-08-24）
- 目标仓库：`D:\lzy\project\kv_pool\code\vllm-ascend`（基于 `origin/main`）
- 工作分支：`block_hash_to_str`

## 1. 问题回顾

`_alloc_gvas_for_save`（`pool_worker.py#L1151`）中，同一 `group_block_hashes[block_idx]` 在三个代码位置各自独立调用 `block_hash_to_str`（`.hex()`）并经 `_make_layerwise_gva_key` 拼接 key：

| 位置 | 作用 | 行号 |
| --- | --- | --- |
| candidate_keys 列表推导 | 构造候选 key 供 `_refresh_allocated_gvas` 刷新缓存 | L1199-L1209 |
| while 循环 | 跳过仍在 `_allocated_gvas` 中的块，推进 `save_start_block` | L1211-L1218 |
| for 循环 | 查缓存/构造新分配 key | L1223-L1231 |

按块计每 hash 被转换 2~3 次（详见分析文档 3.4 节）。运行时 hash 恒为 bytes（`BlockHash = NewType("BlockHash", bytes)`），`.hex()` 分支必然执行，每次调用分配一个新的 64 字符临时 str；随后的 f-string 拼接再分配一个新 key str。

## 2. 目标与非目标

### 目标

1. 同一 hash 在单次 save 流程内只调用一次 `block_hash_to_str`（对齐 issue 验收标准）。
2. 生成的 key 值与改动前完全一致。
3. 保持控制流语义不变（while 推进逻辑、partial key 处理、`_allocated_gvas` 刷新时机）。

### 非目标

- 不改动 load 侧 `_prepare_load_gvas`（已是每 hash 1 次转换）。
- 不改动 `_make_layerwise_gva_key` 的 key 格式（向后兼容 PR #11585 格式依赖）。
- 不引入跨请求/跨步骤的 hash→str 缓存（避免失效与内存常驻问题，超出 issue 范围）。

## 3. 约束

- key 格式（`model@group_id@hash@rank` / `model@hash@rank`）不变。
- 现有 UT 全绿：`tests/ut/distributed/ascend_store/test_pool_worker.py`。
- PR 规模（生产代码 + UT）≤ 1000 行；本方案预计生产代码约 ±20 行、UT 约 60~80 行。
- commit 需带 sign-off（`git commit -s`）；PR 从 fork 仓库发起（AGENTS.md 要求）。

## 4. 方案对比

### 方案 A：仅缓存 hash_str

在三段逻辑前预计算 `hash_strs` 列表，三处改用 `hash_strs[idx - scan_start]`，`_make_layerwise_gva_key` 仍三处现场调用。

- 优点：满足 issue 最低要求（hex 1×）。
- 缺点：key 的 f-string 拼接仍重复 2~3 次/块；改动点与方案 B 相同，收益却不完整。

### 方案 B：预计算完整 key 列表（推荐）

三段逻辑前一次性构造 `block_keys`（每块恰好一次 `block_hash_to_str` + 一次 `_make_layerwise_gva_key`），candidate 刷新、while 跳过、for 分配三段全部按下标复用同一列表。

- 优点：hex 与 key 拼接均降到 1×；临时对象峰值下降；三段共享同一数据源，后续维护只需改一处。
- 缺点：while/for 需维护下标偏移（`idx - scan_start`），需 UT 覆盖边界。

### 方案 C：实例级 hash→str 缓存（否决）

在 worker 上挂 `dict[bytes, str]` 跨请求复用转换结果。

- 否决原因：缓存生命周期与失效管理复杂（请求结束、hash 碰撞淘汰均无必要）；issue 验收口径是「单次 save 流程内 1 次」，方案 B 已满足；引入常驻内存，收益边际。

## 5. 推荐方案（B）详细设计

### 5.1 改动位置

`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`，`_alloc_gvas_for_save` 内 group 循环体，仅涉及 L1196-L1231 区段，函数其余部分（partial key、GVA 分配、日志、padding）不动。

### 5.2 改动后代码

```python
save_start_block = request.save_start_token // effective_block_size
save_end_block = request.save_end_token // effective_block_size
if request.load_spec is not None and request.load_spec.can_load:
    ...
    save_start_block = max(save_start_block, hit_full_blocks)

scan_start = save_start_block
scan_end = min(save_end_block, len(group_block_hashes))
# Build each block key exactly once; the candidate refresh, the
# cached-prefix skip and the allocation loop below all reuse it.
block_keys = [
    self._make_layerwise_gva_key(group_id, block_hash_to_str(group_block_hashes[block_idx]))
    for block_idx in range(scan_start, scan_end)
]
self._refresh_allocated_gvas(block_keys)
# Skip blocks that are still present and readable in MemCache.
while save_start_block < save_end_block and save_start_block < len(group_block_hashes):
    key = block_keys[save_start_block - scan_start]
    if key in self._allocated_gvas:
        save_start_block += 1
    else:
        break

block_gvas: list[int] = []
new_keys: list[str] = []
new_positions: list[int] = []
for blk_idx in range(save_start_block, scan_end):
    key = block_keys[blk_idx - scan_start]
    cached = self._allocated_gvas.get(key)
    if cached is not None:
        block_gvas.append(cached)
    else:
        new_keys.append(key)
        new_positions.append(len(block_gvas))
        block_gvas.append(0)
```

变量说明：

- `scan_start`：进入 while 前的块起点（固定不变，作为下标偏移基准）。
- `scan_end`：`min(save_end_block, len(group_block_hashes))`，与原 candidate_keys 推导、for 循环的上界一致。
- `block_keys`：取代原 `candidate_keys`。列表本身即候选刷新的输入（值相同），同时作为 while/for 的 key 来源。

### 5.3 边界条件分析

| 场景 | 分析 |
| --- | --- |
| while 越界 | while 条件上界为 `save_end_block` 与 `len(group_block_hashes)` 的较小值，即 `scan_end`；迭代中 `save_start_block < scan_end` ⇒ `save_start_block - scan_start < len(block_keys)`，下标安全 |
| for 越界 | for 区间 `[save_start_block, scan_end)` ⇒ 下标 ∈ `[0, len(block_keys))`，安全 |
| 全部块已缓存 | while 推进到 `scan_end` 退出，`for range` 为空，`block_gvas` 为空，行为与原逻辑一致 |
| 全部块未缓存 | while 首轮 break（不推进），for 覆盖 `[scan_start, scan_end)` 全部块 |
| `save_start_block ≥ scan_end`（无待存块） | `block_keys` 为空列表，`_refresh_allocated_gvas` 收到空列表直接返回（其内部 `if not cached_keys: return`），while 不进入，for 为空——与原逻辑一致 |
| partial key | 走 `_make_layerwise_partial_key`（由 req_id/block_index/end_token 拼成，不含 block hash），不在本改动范围 |
| 多组模型 | group 循环每轮重新计算 `scan_start/scan_end/block_keys`，组间无共享状态，天然隔离 |

### 5.4 语义等价性论证

1. **key 值不变**：`_make_layerwise_gva_key` 是纯函数（仅依赖 `self.model_name`、`self.head_or_tp_rank`、`group_id`、hash_str，单次 group 迭代内均不变），预计算结果与原三处现场构造的值完全相同。
2. **dict 语义不变**：`key in self._allocated_gvas`、`self._allocated_gvas.get(key)` 按 str 值比较，对象身份不影响结果。
3. **控制流不变**：while 条件、for 区间、`new_keys`/`new_positions`/`block_gvas` 的装配顺序均保持原样。
4. **下游按值消费**：`new_keys` → `m_store.batch_alloc`；`all_group_save_keys` → `request.save_keys` → write_finish_keys 发布。均为按值使用，无 `is` 身份比较（已核查 `pool_worker.py` 相关消费点）。
5. **内存**：`block_keys` 持有的 str 数量与原 `candidate_keys` 相同；while/for 不再产生临时 key str，峰值临时对象减少（每块少 1~2 个 hex str + 1~2 个 key str）。

## 6. 测试计划

### 6.1 现有 UT 回归

```bash
pytest -sv tests/ut/distributed/ascend_store/test_pool_worker.py
```

服务器执行（按项目工作流：本地脚本 → scp → ssh 执行）。

### 6.2 新增 UT（`tests/ut/distributed/ascend_store/test_pool_worker.py` 内扩展）

1. **转换次数测试** `test_alloc_gvas_for_save_converts_each_hash_once`
   - monkeypatch `vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.block_hash_to_str`（模块级导入，patch 模块命名空间）包裹计数器；
   - 构造含 N 个 block 的请求走一次 `_alloc_gvas_for_save`；
   - 断言总调用次数 == N（改动前为 2N~3N）。
   - 覆盖场景：全部未缓存（while 首轮 break）、部分命中缓存（while 推进 k 块后 break）、全部命中（while 走满）。
2. **key 一致性测试** `test_alloc_gvas_for_save_keys_unchanged`
   - 固定 hash 输入，捕获 `m_store.batch_alloc` 收到的 key 序列；
   - 与改动前基线（硬编码期望值或从 main 分支生成）比对，断言逐项相等。
   - 覆盖单组与多组（`num_kv_cache_groups > 1`，key 含 group_id 段）两种格式。

UT 风格遵循 PEP 8（类间 2 空行、方法间 1 空行），复用现有测试类的 fixture/辅助构造，不新建文件。

### 6.3 冒烟测试（服务器，refactor_818 容器，8×Ascend910）

- **memcache + layerwise**（激活 `use_gva_layerwise` 路径，为本改动必测项）：save/load 全链路，验证 KV 命中与正确性。
- **mooncake（非 layerwise）**：确认 `use_gva_layerwise=False` 早退路径无回归。

### 6.4 性能验证（issue 验收标准 2）

- **转换计数 profile**：由 6.2 的计数 UT 直接给出（3× → 1×，精确可测）。
- **长 prompt 收益**：可选，构造大 block 数场景（如 8K+ blocks）对比改动前后 `_alloc_gvas_for_save` 耗时；数据记入 PR 描述。

## 7. 验收标准（对齐 issue #14148）

1. 功能正确性：生成的 key 与改动前一致（6.2.2 测试保障）；现有单测全绿。
2. 性能：save 侧 hex 转换次数从每块 2~3 次降到 1 次（6.2.1 / 6.4 保障）；长 prompt 场景收益数据（可选，记入 PR）。
3. 交付件：PR + 本设计说明 + 性能数据 + 单测。

## 8. 风险与缓解

| 风险 | 等级 | 缓解 |
| --- | --- | --- |
| 下标偏移计算错误（`scan_start` 误用为推进后的值） | 中 | UT 覆盖 while 推进的全部三类场景（6.2.1）；`scan_start` 在 while 前赋值后不再改写 |
| 与 `ascend-store-refactor` 系列分支的演进冲突 | 中 | 基于 `origin/main` 开发，PR 前 rebase 最新 main 并逐行 review（用户既有 rebase 规范） |
| `_refresh_allocated_gvas` 输入语义被误改 | 低 | 传入的 `block_keys` 与原 `candidate_keys` 值完全相同（5.4.1），无需改函数本身 |
| 现有 UT 依赖 candidate_keys 命名 | 低 | 已确认 `test_pool_worker.py` 通过公开行为（batch_alloc 参数等）断言，不引用局部变量名 |

## 9. 实施步骤

1. 基于 `origin/main` 创建分支 `block_hash_to_str`（工作区当前干净，直接 `git checkout -b block_hash_to_str`）。
2. 按 5.2 修改 `pool_worker.py`。
3. 按 6.2 扩展 UT（本地编写，服务器执行验证）。
4. 本地/服务器跑全量 `test_pool_worker.py` 回归。
5. 服务器冒烟（memcache layerwise + mooncake），测试记录存独立目录。
6. `git commit -s`（Conventional Commits：`perf(kv_pool): convert each block hash to string once in _alloc_gvas_for_save`），从 fork 推送并创建 PR，关联 issue #14148。

## 10. 交付件清单

- [ ] 生产代码改动：`pool_worker.py`（`_alloc_gvas_for_save`，约 ±20 行）
- [ ] UT：转换次数测试 + key 一致性测试
- [ ] 本设计文档随 PR 附带（或摘要入 PR 描述）
- [ ] 性能数据：hex 调用计数对比（before/after）
- [ ] 冒烟记录：memcache layerwise + mooncake
