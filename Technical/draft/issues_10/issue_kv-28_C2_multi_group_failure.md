# [Correctness] hybrid KV load 失败不得使用残缺 KV 进入 forward

> 编号：kv-28 | 维度：Correctness | 严重程度：高 | 建议优先级：P0
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

普通 non-layerwise hybrid load 会把多个 group 的 keys 合并后调用 backend `get`。当部分 key 失败或 backend 返回 `None` 时，single-group 会报告 invalid block，而 multi-group 分支只记录日志后继续。代码中没有在同步返回前建立失败信号，async recv 还会把 request 标为正常 finished，因此存在残缺 KV 继续进入 forward 的可达风险。是否已经在特定模型/机器上表现为错误输出，需要故障注入确认；修复不应等待线上错误复现。

GVA layerwise 已在 metadata/lease 失败时 rollback 并在 forward 前抛错，不存在相同的“只记日志”问题。当前上游 hybrid block manager 也不能安全使用 per-block recomputation，因此不能简单把所有失败 block 塞入 `_invalid_block_ids`。

## 修复前复现与状态验证

1. 在 sync/async non-layerwise hybrid 路径注入某一 group 的单 key 失败、整批 `None` 和 backend exception，记录 backend 已写入地址、connector output、request 状态以及 attention/forward 是否被调用。
2. 用至少一个真实 hybrid 模型配置在目标 Ascend 环境验证失败传播；若后端故障难以稳定制造，可使用可控 fake backend，但必须保留一条真实配置的状态机测试。
3. 复现结果用于选择近期 fail-fast 的落点，不用于否定安全不变量：任何 group 未确认加载完成时，受影响 request 都不得进入 forward。

## 任务

1. 近期修复：同步 non-layerwise multi-group get 任一项失败即 fail-fast，保证受影响 request 不进入 attention/forward。
2. async recv 将 request 标记为失败并通过 worker/model-runner 可观察通道结束等待，禁止调用正常 `set_finished_request`。
3. 在缺少 request-level connector API 时，明确终止本次 engine step；不能使用残缺 KV 继续计算。
4. 完整方案向 vLLM connector output 增加 failed request IDs 或 grouped invalid-block 描述，使默认 `kv_load_failure_policy=fail` 只终止受影响请求。
5. 只有上游 hybrid block manager 明确支持后，才开放 recompute；当前不承诺 per-block fallback。
6. 保持 GVA layerwise 的 lease rollback 和 pre-forward raise，并与 kv-25 的 fatal cleanup 集成。

## 验收标准

### 1. 功能正确性
- sync non-layerwise 某 group 部分失败时，受影响 request 不进入 forward
- async non-layerwise 失败 request 不被标记正常完成，也不滞留在 `WAITING_FOR_REMOTE_KVS`
- backend 返回 `None`、per-key 部分失败和异常均走失败传播
- 其他成功 request 不被误标或错误终止
- GVA layerwise 继续在 forward 前 fail-fast，并释放已取得 lease

### 2. 回归保护
- 分别覆盖 sync non-layerwise、async non-layerwise 和 GVA layerwise
- 覆盖 multi-group 某一 group 失败、部分 key 失败、backend `None` 和多个 request 混合成功/失败
- 现有单测全绿

### 3. 交付件
- PR + request-level 失败传播设计 + 上游接口说明（如涉及）+ 单测

## 证据

- sync load：`pool_worker.py:880-1017`
- GVA preparation/rollback：`pool_worker.py:1450-1540`
- async recv：`kv_transfer.py:923-1039`
- hybrid failure policy：`platform.py:944-950`
- 上游：vLLM `scheduler.py:_update_requests_with_invalid_blocks`
- 关键提交：`ccceb970b`、`ebb4dbac3`

## 重点关注

- fail-fast event 不能被等待方误读为成功，需与 kv-25 的终止式失败协议一致
- 当前优先目标是禁止残缺 KV 进入 forward，不把 recompute 作为既定解法
- 如果修改上游 connector output，需保持 single-group 和非 Ascend connector 兼容

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号、卡数和 hybrid 配置）
- 关联任务池：#9079
- 验收人：@赵鹏博
