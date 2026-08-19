# PR #14465 合入评估报告

## 一、PR 概述

| 项 | 值 |
|---|---|
| PR | https://github.com/vllm-project/vllm-ascend/pull/14465 |
| 分支 | `ascend-store-refactor-1` |
| Head | `2ef35ee6cb9cdd508e4e76bcbcdcd322163c33bc` |
| Base | `1a144d6c3`（upstream/main, [CI] Optimize PR test selection #14298） |
| 规模 | 12 files changed, +1029/-2942 |
| 目标 | 从原 PR #13160 拆分的第一部分：公共 helper 提取、死代码清理、UT 重构 + review 回归修复 |

## 二、提交结构与 DCO

| # | SHA | Subject | Diff 行数 | DCO |
|---|---|---|---|---|
| 1 | `816ad0022` | refactor(kv_pool): extract metadata helpers to module-level functions | 635 | ✓ |
| 2 | `4b923ec54` | refactor(kv_pool): remove redundant state and simplify unit tests | 4928 | ✓ |
| 3 | `2ef35ee6c` | fix(kv_pool): preserve lookup and block-size fallback behavior | 89 | ✓ |

提交分层清晰：先提取 helper（commit 1），再清理死代码和简化 UT（commit 2），最后补回归修复（commit 3）。DCO 3/3 通过。

## 三、逐文件代码审查

### 生产代码（6 文件）

#### 3.1 ascend_store_connector.py（-16 行）

**改动**：
- 移除 `self.backend_name`、`self.use_gva_layerwise` 属性——这两个值在需要时由 `pool_scheduler`/`pool_worker` 各自计算（backend 选择是 scheduler/worker 的职责，不是 connector 的）。
- 移除 `self.kv_caches: dict[str, torch.Tensor] = {}`——未使用的空 dict。
- 移除 `page_size_bytes` 参数——scheduler 自己从 `kv_cache_config` 获取。

**结论**：纯冗余状态移除，无行为变化。

#### 3.2 coordinator.py（-17 行）

**改动**：
- `_cache_family_granularity` → 从 metadata.py 导入 `get_cache_family_granularity`（函数体完全一致，只是从局部私有变为模块级共享）。
- `_num_chunks(token_len, block_size)` → `cdiv(token_len, block_size)`——两者实现完全一致：`(a + b - 1) // b`，cdiv 是 vllm 标准工具函数。
- 移除 `self.group_block_sizes = group_block_sizes`——只存储不读取。

**结论**：去重 + 使用标准库函数，语义完全一致。

#### 3.3 kv_transfer.py（-69 行）

**改动**：
- `stored_requests` 和 `done_task_lock` 从 `KVCacheStoreSendingThread` 和 `KVCacheStoreLayerSendingThread` 两个子类移到父类 `KVTransferThread`（[kv_transfer.py:326-328](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L326)）。所有方法（`add_stored_request`、`dec_stored_request`、`is_stored_request`、`get_stored_request`、`delete_finished_stored_request`）也在父类统一定义。已验证父类确实有这些属性和方法。
- 移除 `add_request` 的 4 个子类 override——每个 override 只做 `self.request_queue.put(req_meta)`，与父类完全一致。
- 修正 `add_request` 返回类型：`-> torch.Tensor` → `-> None`（方法体只做 `put()` 返回 None，旧签名是错误的）。
- 移除 `LayerBatchBuilder` 的 `my_key_index`、`num_ranks_per_layer`、`num_layers` 参数——存储但从不读取。
- 移除 `LayerMultiBlockReqMeta` 导入——类型已删除。

**结论**：消除子类间的代码重复 + 修正错误的返回类型签名 + 移除未使用的构造参数。无行为变化。

#### 3.4 metadata.py（+129 行）

**新增 5 个模块级函数**（从 scheduler/worker 的私有方法提取）：

| 函数 | 来源 | 验证 |
|---|---|---|
| `uses_hybrid_kv_cache` | `KVPoolScheduler._uses_hybrid_kv_cache` / `KVPoolWorker._uses_hybrid_kv_cache` | 逻辑一致 |
| `infer_group_block_sizes` | 同上 | 逻辑一致 |
| `get_group_cache_family` | `_get_group_family` | 逻辑一致 |
| `get_effective_group_block_size` | `_get_effective_group_block_size` | **保留越界 fallback**（commit 3 修复） |
| `infer_cache_transfer_granularity` | `_infer_cache_transfer_granularity` | 见下方等价性分析 |

**`infer_cache_transfer_granularity` 等价性分析**：

原始版本种子为 `lcm_block_size`：
```python
granularities = [self.lcm_block_size]  # = lcm(*grouped_block_size)
for group_id in ...:
    granularities.append(block_size * ratio)
return math.lcm(*granularities)  # = lcm(lcm(b0,b1,...), b0*r0, b1*r1, ...)
```

提取版本无种子：
```python
return math.lcm(*(b0*r0, b1*r1, ...))
```

因 `ratio >= 1`，`b_i` 是 `b_i * ratio_i` 的因子，故 `lcm(b0, b1, ..., b0*r0, b1*r1, ...) = lcm(b0*r0, b1*r1, ...)`。种子冗余，去除后结果一致。

**移除的死代码**：
- `ChunkedTokenDatabase.prepare_block_info`——无调用方。
- `RequestTracker.from_new_request`——无调用方。
- `AscendStoreKVConnectorWorkerMetadata.mark_completed_events`——`aggregate` 方法已处理。
- `RequestTracker` 的 `block_keys`、`starts`、`ends`、`sizes_per_chunk`、`last_block_key` 字段——GVA 分配移到 worker 后这些字段不再被 scheduler 设置。
- `ReqMeta` 的 `kv_cache_families_by_group`、`starts`、`ends`、`sizes_per_chunk` 字段——传入但不读取。
- `AscendConnectorMetadata` 的 `unfinished_request_ids` 参数——`_unfinished_request_ids` set 被移除后不再需要。

**结论**：helper 提取逻辑正确，死代码移除有据可查，`get_effective_group_block_size` fallback 已恢复。

#### 3.5 pool_scheduler.py（-192 行）

**改动**：
- 所有私有 helper 方法替换为 metadata.py 的模块级函数调用。
- 移除 `page_size_bytes` 参数和 `isinstance(kv_cache_config, int)` 兼容 hack。
- 移除 `use_compress`、`need_truncate` 属性——`need_truncate` 仅在 `_infer_swa_blocks` 中对 MambaSpec 设置 True，但该属性从不被外部读取。
- 移除 `keys_per_block_hash` 计算——存储但从不读取。
- 移除 `_unfinished_request_ids` set——仅用于传入 `AscendConnectorMetadata`，而该参数已移除。
- 移除 `_allocate_gva_if_needed` 方法及其 3 处调用——GVA 分配在 worker 侧（`batch_alloc`），scheduler 只生成 block keys，而 `block_keys`/`last_block_key` 字段已移除。
- 移除 `generate_keys` 方法——仅被 `_allocate_gva_if_needed` 调用。
- 移除 `if self.use_layerwise: self._discard_partial_chunks = ...`——上方 2 行已设置同一默认值，此分支冗余。

**结论**：大量死状态和死方法移除，GVA 分配逻辑统一到 worker 侧。无行为变化。

#### 3.6 pool_worker.py（-108 行）

**改动**：
- 与 scheduler 相同的 helper 替换。
- 移除 `lcm_block_size` 属性——仅被 `_infer_cache_transfer_granularity` 使用，提取后不需要。
- 移除 `layerwise_retrievers: list[Any] = []`——在 `start_load_kv` 中初始化但从不使用。
- `_uses_mamba_kv_cache`：`any([...])` → `any(...)`——生成器表达式，语义一致。
- `_init_kv_events`：简化为 `bool(kv_event_config and ...)`——语义一致。

**结论**：与 scheduler 对称的 helper 替换 + 死状态移除。无行为变化。

### UT 代码（6 文件）

| 文件 | 改动行数 | 审查结论 |
|---|---|---|
| test_metadata.py | 475 | 移除 trivial 字段赋值测试（`TestKeyMetadata.test_fields`），新增 `TestCacheLayoutHelpers` 覆盖 5 个提取的 helper（含 fallback） |
| test_pool_worker.py | 1540 | 新增 `test_lookup_all_cached`/`test_lookup_partial`/`test_lookup_exception`（review2.md 修复）+ `test_process_save_for_layer_batch_skip_zero_range`（零长度区间） |
| test_pool_scheduler.py | 1049 | 适配 `AscendConnectorMetadata` 构造函数变化（移除 `unfinished_request_ids`） |
| test_kv_transfer.py | 509 | 适配 `LayerMultiBlockReqMeta` 移除和 `add_request` 签名变化 |
| test_backend.py | 299 | 适配构造函数参数移除 |
| test_ascend_store_connector.py | 289 | 适配 connector 构造函数简化 |

UT 重构策略正确：移除的是"构造函数存取字段"类的 trivial 测试，新增的是对提取 helper 的行为测试和 review2.md 要求的回归测试。

## 四、review2.md 问题修复验证

| # | review2.md 问题 | 修复 commit | 验证方式 | 结果 |
|---|---|---|---|---|
| 1 | `get_effective_group_block_size` 越界从 fallback 变为 IndexError | `2ef35ee6c` | 代码审查 + UT `get_effective_group_block_size([16,32], ["c1","c1"], 5) == 16` | ✓ |
| 2 | `KVPoolWorker.lookup()` 全命中/部分命中/异常测试缺失 | `2ef35ee6c` | UT `test_lookup_all_cached`(→32) + `test_lookup_partial`(→16) + `test_lookup_exception`(→0) | ✓ |
| 3 | 零长度保存区间测试缺失 | `2ef35ee6c` | UT `test_process_save_for_layer_batch_skip_zero_range`（start==end → 0 tasks） | ✓ |
| 4 | 构造参数/属性/方法被删除 | `4b923ec54` | 代码审查确认全部为死代码/死状态（存储不读取/无调用方） | ✓（可接受） |
| 5 | 缺少 Signed-off-by | 全部 3 commits | `git log --format='%B' | grep Signed-off-by` 计数 3/3 | ✓ |

## 五、验证结果汇总

| 阶段 | 结果 | 环境 |
|---|---|---|
| 静态检查（diff --check + compileall） | ✓ | refactor_818 |
| UT | 258 passed, 0 failed | refactor_818（真实 torch/vLLM） |
| 等价性（与 PR #13160 修复后 tree 一致） | ✓ tree hash 一致 | git diff --quiet rc=0 |
| mooncake non-layerwise smoke | ✓（4.02s→1.47s 重复前缀命中） | refactor_818, Qwen3-32B, 8 NPU |
| memcache + layerwise smoke | ✓（2.11s→1.75s 重复前缀命中） | refactor_818, Qwen3-32B, 4 NPU |
| layerwise 路径激活确认 | ✓（`layerwise config: num_layers=64`） | refactor_818 |
| DSV4 MLA + layerwise | 不兼容（`layerwise_cache_layout.py:221`，不在 refactor-1 改动范围） | refactor_818 |

## 六、风险评估

### 6.1 已消除的风险

| 风险 | 消除方式 |
|---|---|
| helper 提取引入逻辑偏差 | 逐函数比对原始私有方法与提取的模块级函数，逻辑一致 |
| `infer_cache_transfer_granularity` 去种子导致结果变化 | 数学证明：`lcm(b0,b1,...,b0*r0,...) = lcm(b0*r0,...)` 因 `b_i | b_i*r_i` |
| `stored_requests`/`done_task_lock` 移除导致丢失 | 确认父类 `KVTransferThread` 已统一定义 |
| `lookup` 行为变化 | 审查源码确认 all-hit/partial/exception 三个路径正确 |
| 越界 fallback 丢失 | commit 3 恢复 + UT 覆盖 |

### 6.2 可接受的风险

| 风险 | 评估 |
|---|---|
| 内部 API 签名变化（构造参数/属性删除） | 所有变化均为 kv_pool 内部模块间调用，非对外 API。影响范围仅限 refactor-2 的 4 个文件，且 refactor-2 以 refactor-1 为基线。 |
| DSV4 MLA + layerwise 不兼容 | `layerwise_cache_layout.py` 不在 refactor-1 改动范围（属于 refactor-2）。refactor-1 验证了 layerwise 路径可达性（Qwen3-32B PASS）。 |
| ruff 未安装 | compileall 已覆盖语法正确性。ruff 是风格检查，非阻断项。 |

### 6.3 未发现的风险

- 无并发安全引入：`stored_requests`/`done_task_lock` 的锁机制不变，只是定义位置从子类移到父类。
- 无接口契约破坏：`add_request` 返回类型修正（`-> torch.Tensor` → `-> None`）是 bug fix，不是破坏——方法从未返回过 tensor。
- 无隐藏依赖：所有移除的字段/方法已确认无调用方（通过 grep 验证）。

## 七、合入判断

### 判断依据

1. **代码逻辑正确**：6 个生产文件的改动可归纳为三类——helper 提取（语义不变）、死代码移除（无调用方）、冗余状态清理（存储不读取）。逐文件审查未发现逻辑错误。

2. **review2.md 问题已全部修复**：5 个问题中 4 个在代码层面修复，1 个（DCO）在提交流程层面修复。修复有 UT 覆盖。

3. **验证充分**：
   - UT 258 passed 覆盖所有提取的 helper + 回归测试
   - mooncake + memcache 双 backend 真实环境 smoke 通过
   - layerwise 路径激活确认 + 完整 save/load 验证
   - 与原 PR #13160 tree 等价

4. **风险可控**：内部 API 变化仅影响 refactor-2（同批拆分），DSV4 兼容性问题不在本 PR 范围。

### 结论

**PR #14465 可以合入。**

该 PR 是纯重构（提取公共 helper、清理死代码、简化 UT），不改变运行时行为。代码审查和真实环境验证均已通过，review2.md 指出的所有问题已修复并有 UT 覆盖。剩余的非阻断项（内部 API 签名变化、DSV4 MLA 兼容性）不影响本 PR 的独立合入。

---

*报告日期：2026-08-18*
*审查环境：refactor_818 容器（8×Ascend910），vllm-ascend `2ef35ee6c`，vllm `0.27.1+empty`*
