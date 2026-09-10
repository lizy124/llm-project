# issue #14148 修复验证报告

- 关联 issue：[vllm-project/vllm-ascend#14148](https://github.com/vllm-project/vllm-ascend/issues/14148)
- 被验证改动：方案 B（[issue_14148_design.md](issue_14148_design.md)）在 `_alloc_gvas_for_save` 中对 `block_hash_to_str` 去重，分支 `block_hash_to_str`
- 报告日期：2026-08-24（v2：UT 审核后删 1 个平凡测试、补 1 个偏移盲区测试，并完成变异验证）
- 验证环境：服务器 192.168.13.165，容器 `refactor_810`（Python 3.12.13 + pytest 8.3.2），测试目录 `/tmp/va_14148`
- 过程原始记录：[test_record/README.md](test_record/README.md)（含运行脚本、diff；关键日志结论已内嵌，原始日志已清理）

---

## 1. 验证结论总览

| # | 关注点 | 结论 | 证据位置 |
| --- | --- | --- | --- |
| C1 | 问题是否真实存在 | **确认真实存在** | §3；[analysis.md](issue_14148_analysis.md) |
| C2 | 方案 B 实现语义是否等价（key 不变、控制流不变） | **等价，无回归** | §4.1、§5.1 |
| C3 | 改动后是否引入回归 | **无回归，80 passed**（v2 复跑确认） | §5.1；[test_record/README.md](test_record/README.md) |
| C4 | 新增测试是否有效（能抓住缺陷） | **有效，基线 4 failed + 变异验证 1 failed** | §5.2、§5.6；[test_record/README.md](test_record/README.md) |
| C5 | 冗余消除是否达成（issue 验收标准 2） | **达成**：每块 2.25 次 → 1 次 | §5.3 |
| C6 | 边界场景（含新增 3 类 + 非零偏移）是否全部正确 | **全部正确，用例全覆盖** | §5.4 |
| C7 | 是否暴露并澄清了原逻辑语义 | **澄清一处**：跳过块 GVA 在传输数组为 0 | §5.5 |

---

## 2. 被验证的问题与方案

### 2.1 问题（来自 issue #14148 验收口径）

`_alloc_gvas_for_save`（`pool_worker.py#L1151`）中，同一 `group_block_hashes[idx]` 在三处各自独立调用 `block_hash_to_str` 并拼接 key：

| 位置 | 作用 | 改动前行号 |
| --- | --- | --- |
| candidate_keys 列表推导 | 构造候选 key 供 `_refresh_allocated_gvas` | L1199-L1209 |
| while 循环 | 跳过仍缓存的块 | L1211-L1218 |
| for 循环 | 查缓存/新分配 | L1223-L1231 |

运行时 hash 恒为 `bytes`（`BlockHash = NewType("BlockHash", bytes)`），`.hex()` 分支必然执行，每次调用分配 1 个新的 64 字符临时 str。见 [block_hash_to_str 分析](issue_14148_analysis.md)。

### 2.2 方案 B（已实施）

在 candidate 刷新 / while 跳过 / for 分配三段逻辑前，一次性构造 `block_keys`（每块恰好 1 次 `block_hash_to_str` + 1 次 `_make_layerwise_gva_key`），三处按下标 `idx - scan_start` 复用。独立基准偏移 `scan_start` 在 while 前赋值后不再改写。代码改动见 [pool_worker.py.diff](test_record/pool_worker.py.diff)（+12/-15）。

---

## 3. 结论 C1：问题真实存在

- 三处转换位置确认：candidate_keys、while、for（§2.1），彼此无结果复用，见 [analysis.md §3](../block_hash_to_str/issue_14148_analysis.md)。
- 基线实测（改造前逻辑 + 新计数测试）：单组 4 块全未缓存场景，`block_hash_to_str` 实际被调用 **9 次**而非期望的 4 次，即每块约 **2.25 次**，与 [analysis.md §3.4](../block_hash_to_str/issue_14148_analysis.md) 预测的"每块 2~3 次"吻合。
- load 侧对比：`_prepare_load_gvas` 仅 1 次列表推导构造 keys，后续整体复用，印证 save 侧是冗余点。

**子结论：冗余真实存在于 save 路径，位置、次数与 issue 描述及分析一致。**

---

## 4. 方案 B 实现审查结论

### 4.1 结论 C2：语义等价

| 维度 | 论证 | 状态 |
| --- | --- | --- |
| key 值 | `_make_layerwise_gva_key` 为纯函数（依赖 `model_name`/`head_or_tp_rank`/`group_id`/hash_str，单次 group 迭代内均不变），预计算与原三处现场构造完全同值 | ✅ |
| dict 语义 | `key in` 与 `.get(key)` 按 str 值比较，对象身份不影响 | ✅ |
| 控制流 | while 条件、for 区间、`new_keys`/`new_positions`/`block_gvas` 装配顺序均保持 | ✅ |
| 下游按值消费 | `batch_alloc`、`save_keys`→`write_finish_keys` 均按值使用，无 `is` 身份比较（已核查） | ✅ |
| 变量演进 | while 推进后的 `save_start_block` 仍是下游 padding `full_gvas[save_start_block + i]` 的基准；设计不等价替代该变量（`scan_start` 仅作下标偏移基准） | ✅ |

### 4.2 边界条件（C6）审查

设计文档 §5.3 覆盖全部场景（含复核补充的 3 类），逐一以 UT 验证：

| 场景 | 用例 | 结果 |
| --- | --- | --- |
| 全未缓存（while 首轮 break） | `..._converts_each_hash_once` | ✅ |
| 部分命中（while 推进 k 块后 break） | `..._with_cached_prefix` | ✅ |
| 全命中（while 走满） | `..._all_cached` | ✅ |
| `save_start_block ≥ scan_end` / 严格大于（load 抬升越界） | `..._skips_scan_when_start_block_lifted_beyond_end` | ✅ |
| 非零 `scan_start` 且区间非空（部分抬升，v2 新增） | `..._with_partial_lift_keeps_key_offsets_aligned` | ✅ |
| `group_block_hashes` 为空 | （v2 删除：与抬升越界用例下游路径重合的平凡用例） | 路径已被上一行覆盖 |
| 重复 hash（key 重复） | `..._duplicate_hashes` | ✅ |
| 多组模型（key 含 group_id 段） | `..._keys_unchanged_multi_group` | ✅ |
| partial key | 走 `_make_layerwise_partial_key`，含 req_id/block_index/end_token，不含 block hash | 不在改动范围，已有旧用例覆盖 |

**子结论：方案 B 实现与设计一致，key 不变、控制流与变量演进不变，语义等价成立；边界场景全部经 UT 实测通过。**

---

## 5. 测试执行与数据

### 5.1 C3：改动后全量回归 —— 80 passed

运行命令：

```bash
cd /tmp/va_14148
PYTHONPATH=/tmp/vllm_14148 python3 -m pytest tests/ut/distributed/ascend_store/test_pool_worker.py -v
```

改动后全量 UT 记录见 [test_record/README.md](test_record/README.md)（§6.1，已含 v2 尾部结论，原始日志已清理）。v2（UT 审核改进后复跑）尾部：

```
================= 80 passed, 14 warnings in 1.68s ================
```

覆盖 6 个既有测试类 + **7 个新增用例**（73 既有 + 7 新增），**无失败、无报错**（14 条 warnings 均为上游 torch `jit.script_method` 弃用告警，与本次改动无关）。

> 早期版本报告曾写"8 个新增用例"，经复核实际为 7 个，已在此更正；v1（初版）亦为 80 passed（当时含 1 个后经审核删除的平凡用例 `..._empty_hashes`，随后补入等价值的非零偏移用例 `..._with_partial_lift_keeps_key_offsets_aligned`，总数不变）。

### 5.2 C4：改动前基线 + 新 UT —— 计数测试失败（证明测试有效）

将新增 UT 与 **main 版本**的 `pool_worker.py` 组合，仅跑计数相关用例：

```bash
python3 -m pytest tests/ut/distributed/ascend_store/test_pool_worker.py -k converts_each_hash -v
```

改动前基线记录见 [test_record/README.md](test_record/README.md)（§6.2，已含隐含结论，原始日志已清理）。尾部：

```
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once_all_cached
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once_duplicate_hashes
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once_with_cached_prefix
======== 4 failed, 1 passed, 75 deselected, 14 warnings in 0.31s ========
```

失败断言展示实际超量转换：

```
E  - ['h0', 'h1', 'h2', 'h3', 'h0', 'h1', 'h2', 'h2', 'h3']   # 实际 9 次
E  + ['h0', 'h1', 'h2', 'h3']                                  # 期望 4 次
```

- 唯一 passed 项为 `..._empty_hashes`：空 hash 列表场景改动前后转换次数均为 0，符合预期（该用例因与抬升越界用例下游路径重合、属平凡测试，已在 v2 UT 审核中删除，见 §5.6）。
- **测试有效性成立**：若无改动，新测试在旧逻辑下失败；改动后全部通过，证明测试精确刻画了"每 hash 1 次转换"的不变式。

### 5.3 C5：降耗量化 —— 冗余消除达成

| 度量 | 改动前 | 改动后 | 变化 |
| --- | --- | --- | --- |
| 每块 `block_hash_to_str` 调用次数 | ~2.25 次（4 块 9 次） | 1 次（4 块 4 次） | **-56%** |
| 每块 hex `.hex()` 临时 str 分配 | 2~3 个 | 1 个 | 减少 1~2 个对象/块 |
| 每块 key f-string 拼接临时 str | 2~3 个 | 1 个 | 减少 1~2 个对象/块 |

长 prompt 场景（block 数大时）收益随块数线性放大；临时对象峰值下降改善 GC 压力。**满足 issue #14148 验收标准 2（单次 save 流程内每 hash 仅转换 1 次）。**

### 5.4 C6 补充：边界场景实测数据

- 抬升越界：转换计数为 0，`batch_alloc` 不被调用 —— 与预期一致。
- **非零偏移（v2 新增）**：`hit_full_blocks=1` 抬升 `scan_start=1`、保存块 1..3 时，计数恰为 3（仅 h1/h2/h3），`batch_alloc` 收到 key 序列 `[k(h1), k(h2), k(h3)]`，传输数组 `[0, 303, 304, 0]`——证明 `blk_idx - scan_start` 偏移在真实非零场景下取值正确。
- 重复 hash：计数仍 == N（逐位置计数，重复 hash 各计 1 次），`batch_alloc` 收到重复 key 序列与基线一致，无行为变化。
- 跳过的缓存块（while 推进）不进入 for 循环，不触发 `batch_alloc`。

### 5.5 C7：澄清的原逻辑语义

实施与断言中发现：**while 跳过的缓存块，其 GVA 在传输数组 `block_gvas_by_group_np` 中保持 0**，GVA 复用通过 `_allocated_gvas` 内部字典完成，而非写入传输数组。这是既有行为，改动保持一致。初版断言误写 `[101,102,303,304]`，据代码语义修正为 `[0,0,303,304]`。**该澄清写入 [design.md §11](../block_hash_to_str/issue_14148_design.md) 与 [test_record](test_record/README.md) 备查。**

### 5.6 C4 补充：变异验证（v2 新增）—— 偏移盲区测试有效

UT 审核发现初版 7 个用例的 `scan_start` 均为 0 或区间为空：若实现漏写 `- scan_start` 偏移（本重构最典型笔误），所有用例仍会通过（`scan_start=0` 时下标恒等）。为此补入 `..._with_partial_lift_keeps_key_offsets_aligned` 并做变异测试：

```bash
# 服务器上临时注入缺陷：去掉 for 循环的 - scan_start 偏移
sed -i 's/key = block_keys\[blk_idx - scan_start\]/key = block_keys[blk_idx]/' pool_worker.py
python3 -m pytest ... -k partial_lift
# 结果：FAILED ..._with_partial_lift_keeps_key_offsets_aligned (IndexError: list index out of range)
#       1 failed, 79 passed
```

变异体被新用例**立即捕获**（非零 `scan_start` 下 `blk_idx` 越界 `block_keys`），恢复正确代码后全量 80 passed。证明偏移盲区已有效堵住，新用例具备真实缺陷检出能力。

---

## 6. 验证限制与说明

| 项 | 说明 |
| --- | --- |
| 测试范围 | 仅 UT（`test_pool_worker.py`）。未含服务器冒烟（memcache + layerwise、mooncake）——属待办 |
| 覆盖率方案 | 容器内 vllm `568afb3a13` 较旧，用本地 `58d3918e` Python 包 `PYTHONPATH=/tmp/vllm_14148` 覆盖；已验证仓库无 `.so` 编译扩展，纯 Python 覆盖可靠 |
| 性能数据口径 | 以 `block_hash_to_str` 调用计数为准（精确确定性度量）；未单独采集 wall-clock（长 prompt 收益可由块数线性外推） |
| 日志可追溯 | 服务器留存 `/home/lizhongyang/tmp/va_14148_results_20260824/`（独立目录未覆盖）；本地关键日志结论内嵌于 `test_record/README.md`，原始 `.log` 已清理 |

---

## 7. 待办（下阶段，不属于本次验证范围）

- [ ] 服务器冒烟：memcache + layerwise（激活 `use_gva_layerwise` 路径，必测）、mooncake（非 layerwise）回归，计划容器 `refactor_818`（8×Ascend910）
- [ ] `git commit -s`，Conventional Commits 消息 `perf(kv_pool): convert each block hash to string once in _alloc_gvas_for_save`
- [ ] 从 fork 推送并创建 PR，关联 issue #14148；冒烟记录存独立服务器目录

---

## 8. 最终验证结论

1. **问题真实存在**：save 路径同一 hash 在 candidate/while/for 三处独立转换，基线实测 2.25 次/块。
2. **方案正确**：方案 B 实现与设计等价，key 不变、控制流与 `save_start_block` 演进不变，无任何语义偏移。
3. **无回归**：改动后全量 UT **80 passed**。
4. **测试有效**：改动前基线下新计数测试 **4 failed / 1 passed**，证明测试精确刻画"每 hash 1 次"约束；v2 变异验证进一步证明非零偏移用例能捕获漏写 `- scan_start` 的典型笔误（1 failed）。
5. **冗余消除达成**：`block_hash_to_str` 每块从 **~2.25 → 1 次（-56%）**，临时对象减少，满足 issue #14148 验收标准 2。
6. **边界完备**：含复核新增的 3 类边界（严格大于抬升、空 hash、重复 hash）+ 非零偏移场景全部经测试验证；v2 审核删除 1 个平凡用例（`empty_hashes`）、补入 1 个高价值用例（`partial_lift`），UT 总数 7 个，无"为写而写"的注水用例。
7. **语义澄清**：明确跳过块 GVA 在传输数组为 0 的既有语义，并写入文档供维护。

**总体：本次修复实现正确、测试充分且经变异验证、目标达成，可进入冒烟与提交流程。**