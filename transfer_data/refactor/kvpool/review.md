# AscendStore Refactor 第一部分严格审核记录

## 1. 审核目的

本文严格审核 `ascend-store-refactor-1` 是否真正达到以下拆分目标：

- 承接公共 helper 提取、死代码清理、重复逻辑删除和 UT 重构。
- 虽然改动量较大，但不改变受支持的运行行为。
- 能够独立构建、评审和验证。
- 可以先完成 UT 与基础 smoke，然后独立合入，不必等待第二部分的完整模型矩阵。
- 第二部分继续承接 backend 生命周期、MLA、实际调用入口和模型特定路径等高风险改动。

记录日期：2026-08-18。

本次审核是代码审查，不修改生产代码、测试或分支历史。

## 2. 审核对象

审核时的分支关系：

```text
1a144d6c3  main / upstream/main
    |
    +-- 9d2b14311  extract metadata helpers
          |
          +-- ce77a3b68  remove redundant state and simplify unit tests
                |          ascend-store-refactor-1
                |
                +-- 1a29b1e4f  backend and model-specific paths
                           ascend-store-refactor-2
```

第一部分相对 `main` 的规模：

```text
12 files changed, 966 insertions(+), 2919 deletions(-)
```

生产代码涉及：

- `ascend_store_connector.py`
- `coordinator.py`
- `kv_transfer.py`
- `metadata.py`
- `pool_scheduler.py`
- `pool_worker.py`

测试代码涉及六个 AscendStore UT 文件。

## 3. 总体结论

审核结论：**拆分方向正确，但当前 `ascend-store-refactor-1` 还不能严格宣称“无行为变化、可以直接独立合入”。建议当前状态为 `Needs changes`。**

从风险隔离角度看，拆分已经基本达到预期：backend 生命周期、`exists` 调用切换、MLA 判定和 backend 初始化整理都保留在第二部分；第一部分的大多数生产代码变更确实属于 helper 提取、无读取状态删除、重复实现删除和内部接口清理。

但是审核发现以下问题：

1. 公共 block-size helper 改变了原有且被 UT 明确定义的越界行为。
2. UT 重构删除了若干核心执行路径的真实覆盖，不只是合并重复用例。
3. 多个构造签名、属性和方法发生可观察变化，因此只能称为“仓库内受支持正常路径基本不变”，不能称为 API 和所有输入行为完全等价。
4. 两个提交都缺少仓库要求的 `Signed-off-by`。
5. 当前仅完成轻量 mock UT 和静态检查，尚未完成目标中要求的基础 save/load smoke。

修复上述问题后，第一部分可以达到按风险独立评审和先行合入的目的。

## 4. 审核方法

本次执行了以下检查：

1. 阅读仓库 `AGENTS.md`，确认测试、代码风格和提交要求。
2. 检查 `main..ascend-store-refactor-1` 的全部生产代码和测试差异。
3. 分别检查 `9d2b14311` 和 `ce77a3b68`，避免两个提交相互掩盖语义变化。
4. 使用 `git grep` 和 `rg` 检查被删除字段、方法、构造参数及全部仓库内调用点。
5. 核对 helper 提取前后的输入、输出、fallback 和异常行为。
6. 对照 UT 重构前后的具体测试场景，而不只比较文件行数或测试函数数量。
7. 运行可用的 AscendStore UT、跨目录消费者测试、`ruff`、`compileall` 和 `git diff --check`。
8. 检查第一、第二部分的祖先关系以及拆分后的最终 tree 等价性。
9. 检查提交消息是否满足仓库 DCO 要求。

## 5. 主要发现

### 5.1 阻断：block-size helper 改变了原有越界行为

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
get_effective_group_block_size()
```

当前实现直接访问：

```python
group_block_sizes[group_id]
```

提取前，scheduler 和 worker 的 `_get_group_block_size()` 都具有以下 fallback：

```python
if group_id >= len(self.grouped_block_size):
    return self.grouped_block_size[0]
return self.grouped_block_size[group_id]
```

因此对于：

```text
group_block_sizes = [16, 32]
group_id = 5
```

原行为返回 `16`，新行为抛出 `IndexError`。

这不是理论推断。原 scheduler 和 worker 都有专门 UT，明确断言越界时回退到第 0 组。`9d2b14311` 在提取 helper 时把新测试改成了“预期抛出 `IndexError`”，随后 `ce77a3b68` 又删除了这个 helper 测试。

由此可以确定：

- helper 提取不是对全部原有输入行为等价。
- 测试曾主动接受新异常语义，而不是单纯迁移旧断言。
- 最终测试重构又移除了记录该变化的用例。

正常内部路径中的 `group_id` 通常来自合法的 group 枚举，因此实际触发概率较低。但一旦 request metadata、layer layout 或 group 数量出现不一致，新代码会从容错 fallback 变成直接崩溃。

审核意见：

- 恢复第 0 组 fallback。
- 为 `get_effective_group_block_size()` 保留合法 group 和越界 group 两类直接 UT。
- 除非计划明确改变容错策略，否则不要在纯 helper 提取 PR 中引入该异常语义。

### 5.2 重要：UT 重构删除了核心 worker lookup 覆盖

原 `test_pool_worker.py` 分别覆盖：

- `KVPoolWorker.lookup()` 非 layerwise 全命中。
- `KVPoolWorker.lookup()` 非 layerwise 部分命中。
- backend `exists` 抛异常时 `lookup()` 返回 `0`。
- `lookup()` layerwise 全命中。
- `lookup_scheduler()` 的全命中、部分命中、异常、layerwise 和多 TP 路径。

当前测试保留了 `lookup_scheduler()` 的主要用例，并将 layerwise 全命中合并为同时调用 `lookup()` 和 `lookup_scheduler()`。但是 `lookup()` 自身的非 layerwise 全命中、部分命中和异常路径不再被覆盖。

`lookup()` 和 `lookup_scheduler()` 不是同一个实现，不能用后者的测试完全替代前者。`lookup()` 包含 coordinator 路径、key 构造、连续命中边界、align-state 处理以及异常回退，是实际 worker 关键路径。

此外，原 `_process_save_for_layer_batch()` 的“保存区间长度为 0 时不创建任务”用例也被删除，当前只保留了 `can_save=False` 的跳过场景。

审核意见：

- 增加一个紧凑的表驱动测试，恢复 `KVPoolWorker.lookup()` 的 all-hit、partial-hit 和 exception 三个场景。
- 保留 layerwise 场景。
- 恢复零长度保存区间用例。
- UT 去重可以减少 fixture 和重复初始化，但不应减少关键分支和错误路径的行为断言。

### 5.3 边界问题：第一部分包含可观察的内部 API 变化

仓库内调用点均已同步，未发现当前仓库生产调用遗漏。但是以下内容不是严格意义上的“完全无行为变化”。

#### `KVPoolScheduler`

- 删除 `page_size_bytes` 构造参数。
- 删除第三个位置参数为整数时按旧 `page_size_bytes` 处理的兼容逻辑。
- 删除 `page_size_bytes`、`need_truncate`、`keys_per_block_hash` 等实例属性。
- 删除 `generate_keys()`。

#### `ChunkedTokenDatabase` 和 metadata

- `ChunkedTokenDatabase` 删除 `use_hybrid` 构造参数。
- `RequestTracker` 删除多个构造参数和属性。
- 删除 `RequestTracker.from_new_request()`。
- `ReqMeta` 删除 `kv_cache_families_by_group`、`starts`、`ends` 和 `sizes_per_chunk` 等参数与属性。
- `AscendConnectorMetadata` 删除第一个 `unfinished_request_ids` 参数，导致后续位置参数整体前移。
- 删除 `mark_completed_events()` 和 `prepare_block_info()`。

#### transfer thread 和 builder

- `LayerBatchBuilder` 删除 `my_key_index`、`num_ranks_per_layer` 等构造参数和状态。
- layerwise 发送、接收线程删除多个构造参数。
- 多个 `add_request()`、`add_stored_request()` 和 `dec_stored_request()` override 被删除。
- `KVCacheStoreLayerSendingThread.dec_stored_request()` 原 override 只修改计数并隐式返回 `None`；现在继承基类后会返回剩余计数。
- 删除 `KVCacheStoreSendingThread.get_stored_requests_snapshot()`。

#### connector 和 coordinator

- connector 删除 `backend_name`、`use_gva_layerwise` 和空 `kv_caches` 属性。
- coordinator 删除未读取的 `group_block_sizes` 属性。

这些符号在当前仓库内大多只有写入、定义或旧 UT 引用，因此作为内部清理风险较低。但是：

- out-of-tree 调用者可能依赖构造签名或公开命名方法。
- 反射、调试、序列化和混合版本场景可以观察到属性变化。
- `dec_stored_request()` 返回值确实发生变化，不能表述为 override 与基类完全相同。

审核意见：

- 如果这些类明确属于不稳定内部 API，可以保留清理，但 PR 描述应写成“仓库内受支持执行路径无预期变化”。
- 如果要求 API 级兼容，应保留兼容参数、方法或过渡 wrapper。
- 不要使用“所有接口和行为完全等价”的表述。

### 5.4 流程阻断：提交缺少 DCO sign-off

仓库 `AGENTS.md` 要求所有提交使用 `git commit -s`，包含：

```text
Signed-off-by: Name <email>
```

审核时以下提交均没有 `Signed-off-by`：

```text
9d2b14311 refactor(kv_pool): extract metadata helpers to module-level functions
ce77a3b68 refactor(kv_pool): remove redundant state and simplify unit tests
```

这可能直接阻塞 upstream PR 合入。

审核意见：

1. 修复代码和测试后 amend 两个提交并补签。
2. 更新 `ascend-store-refactor-1`。
3. 将 `ascend-store-refactor-2` rebase 到新的第一部分 Head。
4. 重新验证第二部分以第一部分为 ancestor。
5. 重新比较最终 tree hash 和完整 diff。

### 5.5 验证缺口：尚未完成基础 smoke

本机没有真实 `torch`、vLLM 和 Ascend 运行环境。通过仓库提供的 `_mock_deps` 可以运行 CPU 逻辑 UT，但不能验证：

- connector 真实启动。
- backend 真实初始化。
- 一次完整 save/load。
- 传输线程在真实 backend 下的启动、完成和退出。
- NPU buffer 注册和数据正确性。
- 日志中是否出现异步 traceback。

因此当前可以证明第一部分能够通过轻量逻辑测试和静态检查，但还不能证明已经达到“UT + 基础 smoke 后独立合入”的最终条件。

## 6. 逐文件生产代码判断

### 6.1 `ascend_store_connector.py`

主要变更：

- 复用 `extra_config` 局部变量。
- 删除没有仓库内读取者的 connector 属性。
- 不再计算并传递 scheduler `page_size_bytes`。

仓库内主路径判断：基本等价。`page_size_bytes` 在 scheduler 中只保存和记录日志，没有参与计算。

保留意见：构造接口和实例可观察属性发生变化，不属于 API 完全等价。

### 6.2 `coordinator.py`

主要变更：

- `_cache_family_granularity()` 统一到 metadata helper。
- `_num_chunks()` 替换为 `cdiv()`。
- 删除没有读取者的 `group_block_sizes` 属性。

对于正常非负 token 长度，原 `(token_len + block_size - 1) // block_size` 与 `cdiv()` 等价。没有发现正常路径结果变化。

### 6.3 `kv_transfer.py`

主要变更：

- 删除只保存但从未参与计算的 builder/thread 构造参数。
- 删除与基类副作用相同的 queue 和 stored-request override。
- 删除无调用者的 snapshot 方法。

仓库内正常副作用大体等价。

保留意见：

- layer sender 的 `dec_stored_request()` 返回值与旧 override 不同。
- 多个类构造签名改变。
- 删除的方法对 out-of-tree 调用者可见。

### 6.4 `metadata.py`

主要变更：

- 新增公共 cache layout helper。
- 删除无读取字段和死方法。
- 缩减 metadata 构造参数和序列化状态。

大部分 helper 是原实现的直接移动。唯一明确的结果级不等价是 `get_effective_group_block_size()` 的越界 block-size fallback 丢失。

### 6.5 `pool_scheduler.py`

主要变更：

- cache layout 推断改用 metadata helper。
- 删除只写不读状态。
- 删除 scheduler 端 GVA key 生成状态。
- 删除无实际用途的构造参数和日志。

对 `block_keys`、`last_block_key`、`_unfinished_request_ids`、`need_truncate` 和 `keys_per_block_hash` 的全仓库引用检查表明，它们没有生产读取者，或只在已经删除的自循环链路中使用。

仓库内合法 group ID 的主路径基本等价。越界 group ID 行为受 5.1 问题影响。

### 6.6 `pool_worker.py`

主要变更：

- cache layout 推断改用 metadata helper。
- 删除 worker 自身的重复 helper。
- 删除未读取的 `lcm_block_size` 和 `layerwise_retrievers`。
- 调整 layerwise builder/thread 构造调用。

合法 group ID 下 helper 结果与原实现一致。`lcm_block_size` 原本只用于 worker 内旧 granularity helper 和日志，新 granularity 公式对正常组配置等价。

越界 group ID 同样受 5.1 问题影响。

## 7. UT 重构判断

### 7.1 `test_ascend_store_connector.py`

主要是将重复 fixture 和相似断言合并到同一用例或 `subTest`。未发现明确的核心路径丢失。

### 7.2 `test_backend.py`

大量独立测试被合并为表驱动测试。合法值、非法值、SSD 支持检测、put/get 成功、错误码和异常路径仍然存在。

少数配置字段的逐字段断言变少，但整体 backend 行为覆盖仍在。属于可以接受的去重，但未来应避免继续压缩关键配置映射断言。

### 7.3 `test_kv_transfer.py`

原 `LayerMultiBlockReqMeta` layerwise 发送和接收测试类整体处于 `@unittest.skip` 状态。新测试删除这些失效用例，并增加基于当前 `LayerTransferTask` / `LayerBatchBuilder` 的实际构建、过滤、去重和 offset 测试。

这一部分不是简单削弱，实际改善了可执行覆盖。

### 7.4 `test_metadata.py`

删除了部分 dataclass 字段、hash 和简单 getter 的重复断言，并增加公共 helper 测试。

问题是最终 `test_layout_policy()` 过度合并，而且没有直接覆盖 `get_effective_group_block_size()`。第一提交中存在的合法/越界 helper 测试在第二提交中消失，掩盖了实际语义变化。

### 7.5 `test_pool_scheduler.py`

多数 early-return、lookup、事件完成和状态更新用例通过循环或 `subTest` 合并，主要场景仍然存在。

原 block-size 越界 fallback 用例被移除，这是明确问题。

### 7.6 `test_pool_worker.py`

大量重复 worker 初始化被抽到 helper，TP mismatch 和 async/layerwise 场景得到明显压缩。

明确缺口：

- `KVPoolWorker.lookup()` 非 layerwise all-hit。
- `KVPoolWorker.lookup()` 非 layerwise partial-hit。
- `KVPoolWorker.lookup()` backend exception。
- layerwise save 的零长度区间。
- block-size 越界 fallback。

其中前三项属于核心执行路径，应恢复。

## 8. 已完成验证

### 8.1 分支和 tree

- `ascend-store-refactor-1` 是 `ascend-store-refactor-2` 的 ancestor。
- `ascend-store-refactor-2` 与原 `ascend-store-refactor` tree hash 相同。
- 最终完整 diff 为空。
- 当前工作树干净，并与 `origin/ascend-store-refactor-1` 一致。

### 8.2 静态检查

以下检查通过：

```text
git diff --check main..ascend-store-refactor-1
python -m compileall
ruff check
```

`ruff` 覆盖全部 6 个生产文件和 6 个 AscendStore UT 文件。

### 8.3 单元测试

由于本机没有真实 `torch` 和 vLLM，测试通过仓库自带的 `_mock_deps` 预载轻量依赖，并让 pytest fixture 正常运行。

六个 AscendStore UT 文件结果：

```text
220 passed, 113 subtests passed
```

调用 `ChunkedTokenDatabase` 的跨目录消费者：

```text
tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py
1 passed
```

这些结果证明当前仓库内调用签名已经同步，轻量逻辑测试可执行，但不能替代真实 backend/NPU smoke。

### 8.4 未验证项

- 线上 PR #14465 的当前 CI 状态未核对，本机没有 `gh` 命令。
- 未运行真实 torch/vLLM 环境下的完整 pytest。
- 未运行 connector save/load smoke。
- 未运行 NPU 模型或特性矩阵。

## 9. 目标符合度判定

| 目标 | 当前判定 | 说明 |
| --- | --- | --- |
| 第一部分可独立查看和构建 | 基本满足 | 静态检查和轻量 UT 通过 |
| 公共 helper 提取 | 部分满足 | 越界 fallback 发生语义变化 |
| 死代码和重复逻辑清理 | 满足 | 仓库内零引用检查成立 |
| 受支持正常运行路径不变 | 基本满足 | 未发现合法 group 配置下的结果变化 |
| 所有原有输入行为完全等价 | 不满足 | 越界 group 从 fallback 变为异常 |
| API 完全兼容 | 不满足 | 多个签名、方法、属性被删除或改变 |
| UT 重构不减少关键覆盖 | 不满足 | worker lookup 等核心场景丢失 |
| 可凭 UT 先行评审 | 部分满足 | UT 通过，但关键覆盖需补回 |
| 基础 smoke 后独立合入 | 尚未满足 | smoke 未执行 |
| 高风险 backend/MLA 改动留在第二部分 | 满足 | 风险边界总体正确 |
| 两部分叠加与原 PR tree 等价 | 满足 | tree hash 和完整 diff 已验证 |
| 满足 upstream 提交规范 | 不满足 | 缺少 `Signed-off-by` |

## 10. 合入前必须处理

建议按以下顺序处理：

1. 修复 `get_effective_group_block_size()` 的越界 fallback，使其与原 scheduler/worker 行为一致。
2. 增加 helper 合法 group 和越界 group 的直接回归测试。
3. 恢复 `KVPoolWorker.lookup()` 非 layerwise all-hit、partial-hit 和 exception 覆盖。
4. 恢复 layerwise save 零长度区间测试。
5. 明确内部 API 策略：要么恢复兼容入口，要么在 PR 描述中承认内部接口清理，不再使用“所有行为完全不变”的表述。
6. 为两个提交补充 `Signed-off-by`。
7. 重跑六个 AscendStore UT 和跨目录 Mooncake 测试。
8. 在真实环境完成 connector 启动及一次 save/load smoke。
9. 检查传输线程完成、退出和日志中无 traceback。
10. 更新第一部分远程分支和 PR #14465 Head。
11. rebase 第二部分到新的第一部分 Head。
12. 重新验证 ancestor、tree hash 和完整 diff。

## 11. 建议的 PR 表述

不建议使用：

```text
This PR has no behavior or interface changes.
```

在修复越界语义后，建议使用更准确的表述：

```text
This PR extracts shared cache-layout helpers, removes state and methods
that have no in-repository readers or callers, deduplicates transfer-thread
implementations, and restructures AscendStore unit tests. No behavior change
is intended for supported in-repository execution paths. Backend lifecycle,
backend query entry points, MLA decisions, and model-specific behavior remain
in the follow-up PR.
```

同时应在测试部分明确列出：

- 六个 AscendStore UT 文件。
- 跨目录 Mooncake consumer UT。
- connector save/load smoke。
- 使用的 layerwise、PD 或单实例基础路径。

## 12. 最终意见

这次拆分的核心思路是正确的：按行为风险而不是代码行数拆分，第一部分吸收大量公共清理和 UT 重构，第二部分只保留少量但风险密度高的 backend、MLA 和实际调用入口改动。这能够显著降低第二部分的评审与模型验证压力。

当前问题不在整体拆分策略，而在第一部分对“无行为变化”的证明还不够严格。block-size fallback 是一个真实语义差异，关键 worker lookup 测试也确实被删减；再加上 DCO 和基础 smoke 尚未完成，目前不应直接给出“可以独立合入”的最终结论。

完成第 10 节的处理后，第一部分可以达到预期目标，并且其验证范围能够与第二部分的完整模型和特性矩阵真正解耦。
