# PR #13160 拆分记录

## 1. 文档目的

本文记录 vLLM Ascend PR [#13160](https://github.com/vllm-project/vllm-ascend/pull/13160) 的拆分背景、目标、分析过程、实际分支关系、验证结果和后续操作要求。

记录日期：2026-08-18。

最终结论：原 PR 已拆成两个前后依赖的分支。第一部分承担公共 helper 提取、低风险清理和 UT 重构；第二部分只保留 backend、MLA 和实际调用入口相关的高风险改动。review 发现原 PR 中 block-size 越界 fallback 被意外丢失，因此该修复已同步到两个拆分分支和原 `ascend-store-refactor` 基线。修复后的两个分支合并结果与修复后的基线完全一致，但相对未修复的 PR #13160 多出该行为修复和回归测试。

## 2. 背景

PR #13160 是 AscendStore KV Pool 的一次大规模简化和重构。rebase 到最新 `upstream/main` 后，原 PR 的总体规模为：

```text
13 files changed, 983 insertions(+), 2960 deletions(-)
```

虽然 PR 的主要意图是重构，但它覆盖 scheduler、worker、metadata、传输线程、backend 和大量单元测试。完整验证需要考虑多种模型和运行方式，例如：

- DeepSeek-V4
- Qwen3.5-27B
- Qwen3-32B
- GLM-5.2
- layerwise 开启和关闭
- PD 分离和 PD 混部
- 不同 backend、KV cache family、hybrid KV cache 和 MLA 路径

如果把所有改动保留在一个 PR 中，公共清理也必须等待整个模型和特性矩阵完成后才能提交、评审和合入，验证周期过长，问题定位也不够清晰。

## 3. 拆分目标

拆分需要同时满足以下约束：

1. 第一部分能够独立构建、运行和评审，不能处于代码只改了一半的状态。
2. 第二部分必须基于第一部分开发。
3. 两部分叠加后的最终文件内容必须与修复后的原 PR 基线完全等价；如果原 PR 本身存在已确认回归，则以修复后的基线为准。
4. 第一部分尽量只包含公共结构、接口整理、死代码清理和低风险重构。
5. 模型相关、backend 生命周期、执行入口和特性开关相关改动留在第二部分。
6. 第一部分验证通过后可以先提交，不必等待第二部分的大规模模型验证。
7. 拆分按行为风险进行，不要求两个 PR 的代码行数接近。

## 4. 拆分前的基线处理

拆分前先将 `main` 和原功能分支都 rebase 到最新 `upstream/main`，避免在过期基线上切分。

基线信息：

| 对象 | Commit |
| --- | --- |
| `main` / `origin/main` / `upstream/main` | `1a144d6c386a9879d3172b8ce236715655fab60f` |
| PR #13160 原始 Head | `309b69959e681a82fa8444b4bb8edae804489fb8` |
| `ascend-store-refactor`（修复后） | `8ad33de37e84ecaaf60335d3f9fc34467d2734e7` |

rebase 后使用 `git range-diff` 检查，原 PR 的两个补丁内容未发生变化，只改写了 commit ID。随后同步了 `origin/main` 和 `origin/ascend-store-refactor`。

## 5. 第一次保守拆分

最初采用了非常保守的边界，只把公共 metadata helper 提取放入第一部分：

```text
commit 9d2b1431196c2617deeb6d8a5c2a8d779b75dae4
refactor(kv_pool): extract metadata helpers to module-level functions

7 files changed, 160 insertions(+), 222 deletions(-)
```

该提交提取了以下 module-level 纯函数：

- `uses_hybrid_kv_cache`
- `infer_group_block_sizes`
- `get_group_cache_family`
- `get_effective_group_block_size`
- `infer_cache_transfer_granularity`

同时消除了 `KVPoolScheduler`、`KVPoolWorker` 和 `AscendStoreCoordinator` 中对应的重复实现。

这个边界非常干净，PR [#14465](https://github.com/vllm-project/vllm-ascend/pull/14465) 最初就是这一个提交。但它只占原 PR 的很小一部分，没有充分减轻第二部分的评审压力。

## 6. 为什么继续扩大第一部分

对第一次拆分后的剩余差异重新统计后发现：

| 类型 | 增加 | 删除 |
| --- | ---: | ---: |
| 单元测试 | 844 | 2470 |
| 生产代码 | 44 | 333 |

剩余删除中约 88% 是单元测试重构，不是模型执行逻辑。生产代码的许多删除也属于以下类型：

- 只有写入、没有任何读取的状态字段
- 与基类实现完全相同的 override
- 已经没有调用者的方法
- 仅转发参数、但参数从未参与计算的构造参数
- 重复的配置读取和等价表达式

因此，`+160/-222` 并不表示只有这些代码不需要完整模型矩阵。行数不能直接代表风险。真正需要严格模型和特性组合验证的改动可能只有几十行，但它们位于 backend 初始化、MLA 判断或实际调用入口上，风险密度更高。

## 7. 扩大第一部分时的分析方法

扩大第一部分不是简单按文件或行数移动，而是逐项证明改动不改变运行行为。

### 7.1 零引用检查

使用 `rg` 检查字段和方法在生产代码及测试中的全部引用，确认以下内容没有生产读取者，或只在已经删除的链路内自循环：

- `block_keys` / `last_block_key` 相关遗留状态
- `_unfinished_request_ids`
- `need_truncate`
- `keys_per_block_hash`
- `kv_cache_families_by_group`
- `starts` / `ends` / `sizes_per_chunk`
- `layerwise_retrievers`
- `prepare_block_info`
- `RequestTracker.from_new_request`
- `mark_completed_events`

这类内容可以作为死代码删除，不需要模型矩阵来证明行为正确，但仍需要 UT、静态检查和基础 smoke 验证。

### 7.2 重复 override 检查

检查多个传输线程子类的 `add_request`、`add_stored_request` 和 `dec_stored_request`。确认其实现与基类一致后，删除子类 override，由基类统一提供行为。

同时删除 `LayerBatchBuilder` 和 layerwise 线程中只保存但从未使用的构造参数，并同步修改所有调用点和 UT。

### 7.3 测试重构检查

六个 AscendStore UT 文件的大量变化是 mock、fixture、factory 和重复用例整理。测试重构与上述死代码清理一起进入第一部分，但与 backend、MLA 和实际执行入口直接绑定的断言保留原行为，避免第一部分提前包含第二部分语义。

### 7.4 明确保留的风险边界

以下改动没有放进第一部分：

- 删除 `MemcacheBackend.init_store`
- worker backend 初始化接口和错误处理整理
- scheduler backend 类加载方式整理
- `batch_is_exist` 到统一 `exists` 调用入口的切换
- scheduler 和 worker 的 MLA 判定表达式调整

这些改动代码量不大，但涉及 backend 生命周期、实际调用入口和模型特定路径，统一放在第二部分验证。

## 8. 最终分支结构

最终提交关系如下（提交 hash 已因修复和 DCO sign-off 更新）：

```text
1a144d6c3  upstream/main
    |
    +-- 816ad0022  提取公共 metadata helpers
          |
          +-- 4b923ec54  删除冗余状态并重构 UT
                |
                +-- 2ef35ee6c  恢复 fallback 并补回关键回归测试
                      |          ascend-store-refactor-1
                      |
                      +-- 25329b86c  backend 和模型特定路径整理
                           ascend-store-refactor-2
```

远程分支：

| 分支 | Head | 作用 |
| --- | --- | --- |
| `origin/ascend-store-refactor-1` | `2ef35ee6cb9cdd508e4e76bcbcdcd322163c33bc` | 第一部分，PR #14465 的 Head |
| `origin/ascend-store-refactor-2` | `25329b86c38809a37bcf2ec497abe7459e754d09` | 第二部分，基于第一部分 |
| `origin/ascend-store-refactor` | `8ad33de37e84ecaaf60335d3f9fc34467d2734e7` | 修复后的原 PR 基线 |

## 9. 第一部分内容

第一部分由三个提交组成：

```text
816ad0022 refactor(kv_pool): extract metadata helpers to module-level functions
4b923ec54 refactor(kv_pool): remove redundant state and simplify unit tests
2ef35ee6c fix(kv_pool): preserve lookup and block-size fallback behavior
```

总体规模：

```text
12 files changed, 1029 insertions(+), 2942 deletions(-)
```

第一部分包含：

- 公共 cache family、block size 和 transfer granularity helper
- scheduler、worker 和 coordinator 的重复逻辑消除
- 无读取状态、未使用字段和死方法删除
- 传输线程重复 override 删除
- 未使用构造参数及调用点清理
- connector 中未使用状态清理
- 六个 AscendStore UT 文件的 fixture、mock、factory 和重复用例重构

第一部分不包含 `memcache_backend.py` 的行为变化，也保留了原 backend 初始化、MLA 判断和 scheduler 查询入口。

线上 PR #14465 已更新到 `2ef35ee6c`，并与 `origin/ascend-store-refactor-1` 完全一致。

review 修复还包括：

- 恢复 `get_effective_group_block_size()` 越界时回退到第 0 组 block size；
- 增加 helper 合法/越界测试；
- 恢复 `KVPoolWorker.lookup()` 全命中、部分命中和 backend 异常测试；
- 恢复零长度保存区间测试。

## 10. 第二部分内容

第二部分只有一个独有提交：

```text
25329b86c refactor(kv_pool): simplify backend and model-specific paths
```

相对第一部分的规模：

```text
4 files changed, 18 insertions(+), 42 deletions(-)
```

涉及文件：

- `tests/ut/distributed/ascend_store/test_pool_scheduler.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

第二部分包含：

- 删除旧的 `MemcacheBackend.init_store` 入口
- 简化 scheduler 和 worker 的 backend 查找、类加载和初始化参数
- 将 scheduler 的存在性查询切换到统一 `exists` 接口
- 简化 scheduler 和 worker 的 `use_mla` 判断
- 更新对应 scheduler UT

第二个远程分支已经准备好，但截至本文记录时尚未创建第二个 upstream PR。

## 11. 修复后的等价性验证

拆分的核心要求不是提交历史相同，而是两部分叠加后的文件树与修复后的原 PR 基线完全一致。

验证结果：

```text
ascend-store-refactor-2 tree:
a2e713465aa29cdd587081d7edb09c52b42dbe5a

ascend-store-refactor tree:
a2e713465aa29cdd587081d7edb09c52b42dbe5a
```

同时执行：

```bash
git diff --quiet ascend-store-refactor-2 ascend-store-refactor
```

退出码为 `0`，说明最终文件内容完全一致。

修复前线上 PR #13160 的 Head 已核对为：

```text
PR #13160 原始 Head:  309b69959e681a82fa8444b4bb8edae804489fb8
修复后基线:          8ad33de37e84ecaaf60335d3f9fc34467d2734e7
```

因此，第二部分叠加第一部分后，与修复后的 `ascend-store-refactor` 基线一致；相对未修复的线上 PR #13160，差异仅为本次 review 修复及其回归测试。

## 12. 已执行的验证

已完成：

- `main` 和原 PR 分支 rebase 到同一个最新 `upstream/main`
- rebase 前后使用 `git range-diff` 检查补丁内容
- 第一部分和第二部分均执行 `git diff --check`
- AscendStore 生产代码及 UT 执行 Python `compileall`
- 检查第二部分确实以第一部分为 ancestor
- 比较最终 tree hash
- 比较最终分支与原 PR 的完整文件 diff
- 刷新并核对 PR #14465 和 PR #13160 的线上 Head

当前 mock 依赖环境下，metadata 和 pool_worker 定向测试结果为 `101 passed, 48 subtests passed`。不包含 layerwise 文件的 AscendStore 测试结果为 `229 passed, 113 subtests`，另有 2 个既有 coordinator 测试因 mock 的 `_find_longest_cache_hit` 返回值形状不匹配失败。layerwise 和跨目录 Mooncake 测试仍需要真实 `torch/vLLM` 环境，当前 Windows 环境无法收集这些测试。

静态检查已通过：`ruff`、`compileall` 和 `git diff --check`。

三个拆分提交和同步到原基线的修复提交均包含 `Signed-off-by`。

## 13. 建议的验证节奏

### 13.1 第一部分

第一部分不要求等待完整模型矩阵，但仍然需要严格验证，建议至少包括：

1. 六个 AscendStore UT 文件全部通过。
2. AscendStore connector 基础启动和一次 save/load smoke 测试。
3. layerwise 关闭时的基础 PD mix 或单实例路径。
4. 检查传输线程退出、请求完成状态和日志中无 traceback。

第一部分验证目标是证明公共 helper、死代码删除、继承关系和测试重构没有引入回归。

### 13.2 第二部分

第二部分需要针对风险点扩大验证，建议覆盖：

- DeepSeek-V4：layerwise 开启和关闭
- Qwen3.5-27B：非 layerwise 基础路径
- Qwen3-32B：非 layerwise 基础路径
- GLM-5.2：非 layerwise 基础路径
- PD 分离和 PD 混部
- Memcache backend 初始化、buffer 注册和存在性查询
- MLA 与非 MLA
- hybrid KV cache、不同 cache family 和 block size
- 如适用，layerwise offload/buffer reuse

第二部分虽然只有 `+18/-42`，但不能因为行数少而降低验证要求。

## 14. 后续操作注意事项

1. PR #14465 对应第一部分，Head 必须保持为 `origin/ascend-store-refactor-1`。
2. 第二部分必须以 `ascend-store-refactor-1` 为 base 查看差异，不能直接以 `main` 查看，否则会再次看到第一部分的全部改动。
3. 如果 upstream 尚不存在第一部分分支，第二个 upstream PR 应等待 PR #14465 合入后再基于新的 `upstream/main` rebase，或者先在 fork 中以第一部分分支作为临时 base 评审。
4. 第一部分发生 rebase 或补充修复后，第二部分也必须同步 rebase，并重新执行 tree 等价性检查。
5. `ascend-store-refactor` 已同步 review 修复，作为修复后的等价性基准，不应删除。
6. 不要用提交数量或 `git rev-list` 判断完全等价；拆分会改变提交历史，应以 tree hash 和完整 `git diff` 为准。

## 15. 最终结果

此次拆分实现了以下结果：

- 第一部分从最初的 `+160/-222` 扩大到 `+1029/-2942`，承担了原 PR 中绝大多数公共清理、测试重构和 review 回归修复。
- 第二部分相对第一部分仍为 `+18/-42`，只保留 backend、MLA 和实际调用入口相关改动。
- 第一部分已通过 PR #14465 提交并同步到远端。
- 第二部分远程分支已准备好，并严格基于第一部分。
- 两部分叠加后的 Git tree 与修复后的 `ascend-store-refactor` 完全一致；相对原始 PR #13160 的额外差异是已确认的 fallback 行为修复和对应回归测试。
- 修复已同步到 `ascend-store-refactor`，因此原 PR 基线和拆分后的最终代码重新保持一致。
- 验证节奏已经解耦：第一部分可先完成 UT 和基础 smoke，第二部分再执行完整模型及特性矩阵。
