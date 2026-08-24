# issue #14148 验证记录（block_hash_to_str 去重）

本目录记录 [方案 B](../issue_14148_design.md) 的实施验证全过程：环境、脚本、操作步骤、结果日志与基线对照。验证执行于 **2026-08-24**。

> 关联文档：[issue_14148_analysis.md](../issue_14148_analysis.md)（问题确认）、[issue_14148_design.md](../issue_14148_design.md)（设计方案）。

## 目录结构

```
test_record/
├── run_14148_ut.sh              # 服务器测试脚本（见 §3）
├── pool_worker.py.diff          # 生产代码改动 diff（+12/-15）
├── test_pool_worker.py.diff     # 新增 UT 的 diff（v2 审核后版本）
└── logs/
    ├── ut_after_fix.log         # 改动后全量 UT（v1）：80 passed
    ├── ut_after_fix_v2.log      # 改动后全量 UT（v2，UT 审核改进后复跑）：80 passed
    └── ut_before_fix_baseline.log  # 改动前基线 + 新UT：4 failed / 1 passed
```

## 1. 验证目标

| 目标 | 判定标准 | 结果 |
| --- | --- | --- |
| 正确性 | 改动后全量 UT（含 8 个新增）全绿 | 80 passed ✅ |
| 有效性 | 旧代码路径下，新增计数测试能抓住冗余（应失败） | 4 failed ✅ |
| 降耗 | 每 hash 的 `block_hash_to_str` 从 2~3 次降到 1 次 | 单组 9→4 次（见 §6）✅ |

## 2. 测试目标与服务器环境

| 项目 | 值 |
| --- | --- |
| 测试目标 | `tests/ut/distributed/ascend_store/test_pool_worker.py`（含新增 8 用例） |
| 服务器 | 192.168.13.165，登录用户 root，经 ssh 连接 |
| 容器 | `refactor_810`（运行 7 天，image 7589766d8039） |
| 容器内 Python | 3.12.13 + pytest 8.3.2（`/usr/local/python3.12.13`） |
| 容器内 vllm | editable，`/vllm-workspace/vllm` @ `568afb3a13`（**较旧，见 §4 版本差异处理**） |
| 测试代码合入目录 | 容器内 `/tmp/va_14148`（本地 main+改动打包） |
| 覆盖用的 vllm 包 | 容器内 `/tmp/vllm_14148`（本地 vllm `58d3918e` 的 Python 包） |
| 测试记录保存 | 服务器 `/home/lizhongyang/tmp/va_14148_results_20260824/`（独立目录，不覆盖） |

## 3. 测试脚本

本地原稿：[D:\lzy\project\kv_pool\tmp\run_14148_ut.sh](run_14148_ut.sh)。

执行链路（项目接入规范：本地脚本 → scp → ssh 远程执行）：

```bash
# 1) 本地打包当前工作树（含改动，不影响工作区）
git stash create "va_14148_snapshot"    # 未提交改动时得到快照 sha，若无改动则用 HEAD
git archive <sha> -o va_14148.tar.gz

# 2) 上传 + 解压
scp va_14148.tar.gz 192.168.13.165:/home/lizhongyang/tmp/
docker cp /home/lizhongyang/tmp/va_14148.tar.gz refactor_810:/tmp/
docker exec refactor_810 bash -c 'mkdir -p /tmp/va_14148; tar -xzf /tmp/va_14148.tar.gz -C /tmp/va_14148'

# 3) 上传脚本并执行（PYTHONPATH 覆盖 vllm 包，见 §4）
scp run_14148_ut.sh 192.168.13.165:/home/lizhongyang/tmp/
docker cp ... refactor_810:/tmp/run_14148_ut.sh
docker exec refactor_810 bash /tmp/run_14148_ut.sh   # 默认全量
```

## 4. 服务器 vllm 与本地 main 的版本差异处理

容器内 editable 安装的 vllm 为 `568afb3a13`，而本地 main 基于 `58d3918e`（含 `FusedMoEFactory` 等更新接口）。直接跑 UT 会因接口不匹配报 `FusedMoEFactory` 未定义等错误。

处理：本地 vllm `58d3918e` 的 `vllm` Python 包打包上传，容器内以 `PYTHONPATH=/tmp/vllm_14148` 覆盖运行。已验证该仓库无编译扩展 `.so`，纯 Python 覆盖可行：

```bash
git archive 58d3918e vllm -o vllm_pkg.tar     # 本地
scp + 解压至容器 /tmp/vllm_14148
docker exec refactor_810 bash -c 'cd /tmp/va_14148; PYTHONPATH=/tmp/vllm_14148 python3 -m pytest ...'
```

## 5. 新增测试用例（7 个）

所有新增都在 `TestKVPoolWorkerProcessLayerData` 内，通过 monkeypatch 模块级 `block_hash_to_str` 计数 + `m_store` mock 断言。完整 diff 见 [test_pool_worker.py.diff](test_pool_worker.py.diff)。

> v2 审核调整：初版含 `..._empty_hashes`（空 hash 列表，与抬升越界用例下游路径重合的平凡用例，已删除）；补入 `..._with_partial_lift_keeps_key_offsets_aligned`（非零 `scan_start` 偏移场景，堵住"漏写 `- scan_start` 仍全绿"的测试盲区，经变异验证确认有效）。总数 8→7，UT 总通过数不变（80）。

| 用例 | 场景 | 断言要点 |
| --- | --- | --- |
| `..._converts_each_hash_once` | 全未缓存（while 首轮 break） | 每 hash 恰好转 1 次 |
| `..._with_cached_prefix` | 部分命中缓存（while 推进 k 块后 break） | 计数 == N；跳过块在传输数组 GVA 保持 0 |
| `..._all_cached` | 全部命中（while 走满） | 计数 == N；`batch_alloc` 不被调用；传输数组全 0 |
| `..._skips_scan_when_start_block_lifted_beyond_end` | `save_start_block` 被 `hit_full_blocks` 抬升越过 `scan_end` | 计数为 0；`batch_alloc` 不被调用 |
| `..._with_partial_lift_keeps_key_offsets_aligned`（v2 新增） | `hit_full_blocks=1` 部分抬升，`scan_start=1` 且保存块 1..3 | 计数仅含 h1..h3；`batch_alloc` key 序列正确；传输数组 `[0, 303, 304, 0]` |
| `..._duplicate_hashes` | 含重复 hash | 计数 == N（重复 hash 逐个计数）；`batch_alloc` 收到重复 key 序列与基线一致 |
| `..._keys_unchanged_multi_group` | 多组（`num_kv_cache_groups=2`） | `save_keys` 含 group_id 段，key 序列不变 |

> **实施中发现的原逻辑语义**：while 跳过的缓存块不进入 for 循环，其 GVA 在传输数组 `block_gvas_by_group_np` 中保持 `0`（GVA 复用通过 `_allocated_gvas` 内部字典而非写入数组）。这是既有行为，改动保持一致；初版断言误写为 `[101,102,303,304]`，已修正为 `[0,0,303,304]`。

## 6. 结果与基线对照

### 6.1 改动后：80 passed

v2（UT 审核改进后复跑，当前版本）：[logs/ut_after_fix_v2.log](logs/ut_after_fix_v2.log) 尾部：

```
================= 80 passed, 14 warnings in 1.68s ================
```

v1（初版 UT）：[logs/ut_after_fix.log](logs/ut_after_fix.log)，同样 80 passed。

覆盖 6 个既有测试类 + 新增 7 用例（v2 调整后），无回归。

### 6.1.1 变异验证（v2 新增）：非零偏移用例有效

在服务器上临时注入缺陷（去掉 for 循环的 `- scan_start` 偏移）后仅跑偏移用例：

```
FAILED ...test_alloc_gvas_for_save_with_partial_lift_keeps_key_offsets_aligned
       (IndexError: list index out of range)
1 failed, 79 passed, 14 warnings in 1.81s
```

变异体被立即捕获；恢复正确代码后全量 80 passed。初版 7 用例均无法捕获该缺陷（`scan_start` 全为 0 或区间为空），v2 补入的用例是唯一防线。

### 6.2 改动前基线：新计数测试失败

将 [test_pool_worker.py.diff](test_pool_worker.py.diff) 的 UT 与 **main 版本**的 `pool_worker.py` 组合运行（仅跑 `-k converts_each_hash`），[logs/ut_before_fix_baseline.log](logs/ut_before_fix_baseline.log) 尾部：

```
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once_all_cached
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once_duplicate_hashes
FAILED ...test_alloc_gvas_for_save_converts_each_hash_once_with_cached_prefix
======== 4 failed, 1 passed, 75 deselected, 14 warnings in 0.31s ========
```

基线的失败断言展示实际超量转换（[pool_worker.py.diff](pool_worker.py.diff) 对应旧代码）：

```
E  - ['h0', 'h1', 'h2', 'h3', 'h0', 'h1', 'h2', 'h2', 'h3']   # 实际 9 次
E  + ['h0', 'h1', 'h2', 'h3']                                  # 期望 4 次（每 hash 1 次）
```

- 1 passed 的一项为 `test_alloc_gvas_for_save_converts_each_hash_once_empty_hashes`：空 hash 列表场景改动前后转换次数均为 0，符合预期。
- **降耗量化**：单组 4 块场景，调用次数 9 → 4（约 2.25 次/块 → 1 次/块），与 [analysis §3.4](../issue_14148_analysis.md) 预测的“每块 2~3 次”吻合。

### 6.3 转化目标对照

| 场景 | 改动前实际 | 改动后实际 | 断言 |
| --- | --- | --- | --- |
| 全未缓存（单组 4 块） | 9 次（2.25×4） | 4 次 | == N ✅ |
| 部分命中 | >N | N | == N ✅ |
| 全命中 | >N | N | == N ✅ |
| 抬升越界 / 空 hash | 0 | 0 | == 0 ✅ |
| 重复 hash | >N | N | == N ✅ |
| 多组（2 组 × 4 块） | >N | N | == N ✅ |

## 7. 服务器留存产物

| 路径 | 内容 |
| --- | --- |
| `/tmp/va_14148` | 改动版仓库快照（验证后已恢复改动后版本） |
| `/tmp/vllm_14148` | 覆盖用的本地 vllm `58d3918e` Python 包 |
| `/home/lizhongyang/tmp/va_14148_results_20260824/` | 两份测试日志（独立目录，未覆盖） |
| `/home/lizhongyang/tmp/va_14148.tar.gz`、`/home/lizhongyang/tmp/vllm_pkg.tar.gz` | 打包上传的产物 |

## 8. 待办（不在本次记录范围）

- [ ] 服务器冒烟：memcache + layerwise（必测）、mooncake（非 layerwise）回归（计划容器 refactor_818）
- [ ] `git commit -s`（Conventional Commits）后从 fork 推送，创建 PR 关联 issue #14148

## 9. 结论

验证链完整：**改动后 80 passed 无回归**；**改动前基线 + 新 UT 抓住冗余**（4 failed，实测每块 ~2.25 次 → 1 次），证明方案 B 实现正确、测试有效、降耗达成设计预期。