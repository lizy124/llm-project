# AscendStore 社区任务难度 Top 3

> 代码基线：vllm-ascend `main@d5e9816065ede613327d93908f87fee9f5c47128`

## Top 3 清单

| 排名 | 任务 | 难度 | 核心挑战 | 当前可发布形态 |
|---:|---|---:|---|---|
| 1 | [kv-25：transfer thread 终止式失败协议](issue_kv-25_C1_transfer_thread_exception.md) | 5/5 | 跨基类、六个 transfer 子类、queue accounting、失败传播及 event/counter/lease cleanup | 拆成 terminal core 与 per-subclass cleanup |
| 2 | [kv-28：hybrid KV load 失败传播](issue_kv-28_C2_multi_group_failure.md) | 4.5/5 | 同时覆盖 sync/async/hybrid 状态，完整 request 隔离还需要上游 vLLM 协议 | 拆成近期 fail-stop 与长期 request-level 协议 |
| 3 | [kv-07：non-layerwise backend batching](issue_kv-07_P07_non_layerwise_io_merge.md) | 4/5 | descriptor、重复 key、逐项回填、三 backend 契约和真实 NPU 性能验证 | 先发布 Phase 0；Go 后按路径另开实现任务 |

## 1. kv-25：transfer thread 终止式失败协议

### 已闭合问题

- `set_device()` 位于 ready signal 和异常捕获之前，初始化异常会让 creator 永久等待。
- handler fatal 后线程直接退出，但仍允许继续入队，排队项可能永远得不到处理。
- 基类没有统一调用异常 cleanup hook；async finished polling 也没有统一传播 thread fatal。
- normal send、async recv、key layerwise 和 GVA layerwise 对 queue、event、counter、lease 的所有权不同，不能使用一套无差别 cleanup。

### 建议拆分

#### kv-25A：startup 与 terminal core

- startup 无论成功失败都唤醒 creator，并传播原始异常。
- ready wait 使用有界等待。
- 保存首次 fatal 原因，fatal 后停止消费并拒绝新增任务。
- base run 调用统一异常 hook；等待方醒来后必须重新检查 fatal。
- 明确定义排队项是取消、失败返回还是由 owner 重建，不能只修正 `unfinished_tasks`。

#### kv-25B：per-subclass ownership cleanup

- 分别覆盖 normal send、async recv、key layerwise send/recv、GVA layerwise send/recv。
- 对实际拥有的 request state、event、counter 和 lease 做 exactly-once、幂等清理。
- cleanup 不得覆盖首次 fatal 原因，也不能把 event set 解释成成功。

### 验收重点

- startup failure 有界返回，不永久等待。
- fatal 后不处理第二个任务，也不接受新任务。
- 当前项和排队项的 queue accounting 最终闭合。
- 所有等待方得到失败而非成功完成。
- GVA 中途失败不遗留 lease；cleanup 重复调用结果一致。

## 2. kv-28：hybrid KV load 失败传播

### 已闭合问题

- sync non-layerwise multi-group load 部分失败时只记录日志，然后正常返回到 forward 路径。
- async multi-group load 失败后仍调用正常 `set_finished_request`。
- single-group 可以报告 invalid block，当前 hybrid scheduler 却不能安全消费跨 group 不一致的 per-block fallback。
- GVA layerwise 已对 multi-group metadata/lease 失败执行 rollback 并抛错，应作为回归基线。

### 建议拆分

#### kv-28A：近期 fail-stop

- sync multi-group 任一必要项未确认成功时，在 forward 前失败。
- async request 进入明确的失败完成状态，禁止调用正常 finished。
- 当前缺少 request-level API 时允许终止整个 engine step，首要目标是零错误计算。

#### kv-28B：request-level failure 协议

- 与上游 vLLM 定义 failed request IDs 或等价 connector output。
- 只终止受影响 request，其他成功 request 正常执行。
- 当前不支持 hybrid per-block recompute，不把 recompute 作为既定验收方案。

### 与 kv-25 的边界

- 单 request miss/部分失败属于 request outcome，不应自动杀死共享 recv thread。
- thread 或 backend 系统性不可用才进入 kv-25 terminal state。
- kv-28A 的 sync fail-stop 可独立实施；async failure channel 需与 kv-25A 的等待方/fatal 语义对齐。

### 验收重点

- sync/async 的部分失败、整批 `None` 和 backend exception 均不会进入正常 forward/finished。
- kv-28A 验收整步安全失败，不要求当前接口完成请求隔离。
- kv-28B 单独验收失败 request 隔离和其他 request 正常推进。
- GVA layerwise rollback 行为保持不变。

## 3. kv-07：non-layerwise backend batching

### 已闭合事实与未知项

当前 sync load、async load 和 save 确实按 request 调用支持列表输入的 backend 接口，因此存在跨 request 扩批机会。但当前代码和测试不能证明：

- 真实 workload 中单 step 经常包含多个可合并 request；
- backend 固定调用开销足以覆盖 descriptor、去重和回填成本；
- async 微批不会恶化单请求尾延迟；
- TTFT 或吞吐存在端到端净收益。

所以本任务首先是性能验证，不是已经成立的性能优化。

### Phase 0

- 分别统计 sync load、async load、save 的 step request 数、backend 调用次数、items/call、局部耗时和 queue wait。
- 按 Mooncake、MemCache、Yuanrong 确认最大 batch、重复 key 和逐项失败语义。
- 在采集数据前定义 Go/No-Go 阈值。
- No-Go 验证报告是有效交付，不为完成任务强行修改热路径。

### Go 后按路径拆分

1. 优先实现 profile 收益最明确的单一路径，不一次性重写三条路径。
2. sync load 已有逐项 get result，可独立推进。
3. save batching 依赖未入选 Top 3 的 `kv-26` normalized put result。
4. async 微批依赖 kv-25A 的 terminal/cancel 语义，并必须设置数量和等待时间上限。

### 验收重点

- 合并前后每个 request 的 key、地址、失败 block、完成状态和 event 一致。
- 重复 key、multi-group、mask、partial block、取消和 fatal 均能逐项回填。
- 报告 backend 调用次数、batch 构建耗时、TTFT、吞吐和尾延迟。
- 只有端到端数据达到预设 Go 条件时才启用实现。

## 实际实施顺序

难度排名不等于排期。建议顺序为：

1. `kv-28A`：先关闭残缺 KV 进入 forward 的正确性风险。
2. `kv-25A`：建立 startup 和 terminal core。
3. `kv-07 Phase 0`：可与前两项并行采集数据，不修改热路径。
4. `kv-25B`：按子类完成资源 cleanup。
5. `kv-28B`：推进上游 request-level failure 协议。
6. `kv-07 Go path`：仅实现数据证明有价值的路径；save 等待 `kv-26`，async 等待 `kv-25A`。

## 最终记录

本目录按完整实现难度确定的 Top 3 为：

`kv-25 > kv-28 > kv-07`

其中 `kv-25` 和 `kv-28` 是已由当前代码证明但必须拆分的可靠性/正确性任务；`kv-07` 是需要先通过 Phase 0 的条件性性能任务。
