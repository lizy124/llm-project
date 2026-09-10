# vLLM-Ascend AscendStore 社区任务发布稿（Top 3）

> 目标仓库：`vllm-project/vllm-ascend`
> 代码范围：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 审核基线：`main`，提交 `d5e9816065ede613327d93908f87fee9f5c47128`（2026-08-15）
> 上游协议基线：`vllm-project/vllm`，提交 `58d3918e3ea0a544ffedadad2ba84559e9c51d8f`
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 验收人：赵鹏博（GitHub handle 待发布前本人确认）

## 审核原则

本稿按当前源码重新核对三个候选任务，不沿用旧文档中的审核结论，也不把战略价值、事故优先级和实施难度混为同一种排名。

- `kv-25`、`kv-28` 的可靠性或正确性缺陷已由当前控制流闭合证明，可以作为修复任务直接发布。
- `kv-07` 的逐 request backend 调用和跨 request 扩批机会已由代码确认，但端到端性能收益尚未闭合；可以发布，但必须把 profiling 和 Go/No-Go 判断纳入同一个任务。
- 三项分别作为三个完整社区任务发布，不再拆成 A/B 或另开 Phase 0 子任务。任务内部允许按状态机、路径或验证顺序分阶段提交，但最终按本稿的完整范围统一验收。
- 难度顺序为 `kv-25 > kv-28 > kv-07`。该顺序只表示完成完整任务的实施和验收难度，不表示修复优先级。

## 审核总表

| 难度排名 | 任务 | 当前代码是否闭合 | 发布方式 | 修复优先级 | 难度 |
|---:|---|---|---|---:|---:|
| 1 | `kv-25` transfer thread 终止式失败协议 | 是：startup 永久等待和运行期 terminal protocol 缺口均可由代码证明 | 直接发布 | P1 | 5/5 |
| 2 | `kv-28` hybrid KV load 失败传播 | 是：sync/async multi-group 失败可被当成正常 load 完成 | 跨仓发布，协议确认后编码 | P0 | 4.5/5 |
| 3 | `kv-07` non-layerwise backend batching | 部分闭合：扩批机会成立，端到端收益未证明 | 数据驱动发布 | P2 | 4/5 |

## 通用约定

- 代码基线以认领任务时最新 `main` 为准；若实现已经变化，贡献者应先在 issue 中回报并确认范围。
- 三项均围绕 AscendStore KV pool，不顺带重构无关 connector、backend 或 scheduler；`kv-28` 为闭合 hybrid failure outcome 所必需的上游 `KVConnectorOutput` 与 scheduler 改动明确属于任务范围。
- 现有单测必须全绿；新增或更新 `tests/ut/distributed/ascend_store/` 下的针对性测试。
- 涉及 NPU event、stream、GVA lease 或真实 backend 契约时，注明 NPU 型号、卡数、backend 版本及 TP/CP/PP、layerwise、hybrid 等关键配置。
- 故障测试应使用可控异常注入和 event/barrier，不依赖低概率压力循环；性能测试必须给出工作负载、预热、样本数、统计口径和原始数据。
- 交付件至少包括 PR、设计或测量说明、针对性测试、风险和回退方式。若涉及上游 vLLM 接口，应同时给出接口说明或 companion PR。

---

## 任务 1：[Reliability] 建立 transfer thread 终止式失败协议

> 编号：`kv-25` | 严重程度：高 | 修复优先级：P1 | 难度：5/5
>
> 审核状态：问题已闭合，作为一个完整可靠性任务直接发布。

### 已确认问题

`KVTransferThread.run()` 在进入 handler 的异常保护前调用 `set_device()`，成功后才设置 ready event；worker 在五个启动位置直接执行无超时 `wait()`。因此初始化异常时 creator 永久等待是确定性缺陷。

线程进入运行期后也没有完整的 terminal protocol：handler 抛出异常时，基类只保存 `_fatal_error` 后退出；`add_request()` 不检查 fatal 状态，线程退出后仍可继续入队；non-layerwise save 仍可能在 `request_queue.join()` 等待永远不会处理的排队项；async non-layerwise finished polling 也没有统一检查 recv thread fatal。

普通 send、async recv、key-layerwise 和 GVA-layerwise 对 queue item、request state、NPU event、counter 和 lease 的所有权不同。当前问题是缺少统一的 terminal fatal 状态和按实际所有权执行的清理协议，不是“所有 handler 异常都会以相同方式挂死”，也不是把所有 request-level 或逐 key backend 失败升级为 thread fatal。

代码证据：

- [kv_transfer.py:347](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L347)：`add_request()` 未拒绝 fatal 后的新任务。
- [kv_transfer.py:496](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496)：`set_device()`、ready signal 和 handler 异常捕获的顺序。
- [kv_transfer.py:510](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L510)：运行期异常只记录 fatal 后退出。
- [kv_transfer.py:523](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L523)：基类定义了异常 cleanup hook，但 `run()` 当前没有调用。
- [pool_worker.py:488](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L488)：worker 五个无界 ready wait 的首处；其余位于 `505`、`544`、`562`、`577`。
- [pool_worker.py:577](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L577)：non-layerwise async recv 的无界 ready wait。
- [pool_worker.py:456](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L456)：多个 transfer threads 在同一 startup 方法中按顺序启动，后启动失败会留下已启动 sibling。
- [pool_worker.py:1767](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1767)：普通 save 的 counter 注册与 queue 入队是两个独立操作。
- [pool_worker.py:1771](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1771)：save 路径等待 queue accounting 闭合。
- [pool_worker.py:2099](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2099)：async finished polling 直接读取完成集合，未先传播 recv thread fatal。

### 任务范围

1. 为 `KVTransferThread` 建立统一的终止式失败状态：保存首次 fatal 原因，线程停止消费新任务，fatal 后新增任务被明确拒绝，等待方能够查询并重新抛出原始失败。
2. 将 `set_device()` 等 startup 操作纳入异常处理；无论成功失败都结束 creator 的 ready 等待，并为 ready wait 增加有界超时和明确的异常传播。
3. 将 `_start_kv_transfer_threads()` 定义为 all-or-nothing startup transaction。任一后启动线程失败时，停止并有界回收此前已经启动的 sibling threads，清理已发布的 worker 引用，且不得设置 `_transfer_threads_started=True`。
4. ready wait 超时必须同时取消对应 startup generation。若 `set_device()` 在 creator 超时后迟到成功，该线程只能退出和清理，不能再进入 queue 消费循环；无法立即停止的 daemon thread 也必须与后续重建实例隔离。
5. 用同一状态锁或等价原子协议线性化 `add_request()` 与 RUNNING -> TERMINAL 转换。每个并发提交只能有一个结果：在 RUNNING 状态被接受并最终完成/取消，或在 TERMINAL 状态被拒绝；不得出现“调用方认为已接受，但 terminal cleanup 看不到”的任务。
6. 将 request counter/event 注册与 queue 入队定义为一个提交事务，覆盖普通 save 和 layerwise 的同类两步操作。入队失败或 fatal 竞态必须回滚已注册状态；成功入队后则由正常处理或 terminal cleanup exactly-once 结算。
7. `run()` 捕获 terminal fatal 后调用统一异常 hook。定义当前 queue item 和尚未处理 queue item 的所有权；对实际进入 queue accounting 的项保证 exactly-once `task_done()`。取消或排空未处理项时，同时完成 terminal failure、counter/event 和 owner 通知，不能只把 `unfinished_tasks` 改为零。
8. 对普通 send、async recv、key-layerwise send/recv、GVA-layerwise send/recv 分别注入能够逃逸到 base `run()` 的 terminal fatal。每条路径只清理自己实际取得的 request state、event、counter 和 GVA lease，清理必须幂等且不得覆盖首次 fatal 原因。
9. event 只表示等待结束。所有被 event 唤醒的等待方必须重新检查失败，不能把 `event.set()` 解释为传输成功。
10. async non-layerwise finished polling 必须把 thread fatal 传播到 worker/model runner/engine，不能让请求永久停留在 `WAITING_FOR_REMOTE_KVS`。
11. 保持现有 fatal 后停止处理后续任务的语义，不恢复“记录异常后继续消费 queue”的旧行为。
12. 明确普通 sender 在 thread 层可见异常的分类：可恢复的 request-level 失败继续完成本地生命周期；能够证明 thread/backend 系统性不可用的异常必须重新抛给 base `run()` 进入 terminal protocol，不能被 blanket catch 后伪装成正常 thread 状态。backend wrapper 已吞掉的异常、`put()` 的逐 key `AVAILABLE/FAILED/UNKNOWN` 结果和远端写入可观测性属于 `kv-26`，不由本任务推测或补造。

### 验收标准

- startup failure 在配置的时间上限内返回 creator，并保留原始异常；不得永久等待 ready event。
- 第二个或后续线程 startup 失败时，先前成功启动的 sibling threads 全部停止或进入不可消费的隔离状态，worker 不保留半初始化线程组。
- ready timeout 后模拟迟到的 `set_device()` 成功，原线程不得消费任何任务；随后重试 startup 不受旧线程或旧 generation 干扰。
- fatal 后不处理第二个排队任务，也不接受新任务。
- 用 event/barrier 控制 `add_request()` 与 fatal transition 的两种交错，证明每个任务要么被拒绝，要么出现在正常完成或 terminal cancel 集合中，不丢失、不重复。
- 在 counter/event 注册后、queue put 前注入 fatal 和入队异常，普通与 layerwise 路径均不得遗留 counter、event owner 或无法 join 的 queue 状态。
- 当前项和未处理项的 queue accounting 最终闭合，`queue.join()` 不等待永远不会执行的任务。
- 当普通 send、async recv、key-layerwise 或 GVA-layerwise 实际进入 terminal fatal 时，对应等待方都能观察到同一个 fatal outcome，不能仅因 event/finished signal 被置位就解释为成功。
- NPU event、per-request counter、transfer state 和本 step 已取得的 GVA lease 按实际所有权 exactly-once 清理；重复调用 cleanup 结果一致。
- 保留现有 fatal-stop 回归测试，并新增 startup failure、queue cancel/drain、各类 event/counter、lease cleanup、fatal 后拒绝入队和 async finished polling 测试。
- 普通 sender 故障注入同时证明两条分支：request-level/per-key 失败不会被错误提升为 terminal fatal；thread 层可见的系统性异常不会被吞掉，能够进入统一 terminal protocol。backend 结果契约仍由 `kv-26` 验收。
- 在目标 Ascend 环境至少完成一次包含 NPU event 和 GVA lease 的异常注入验证；无法硬件验证的路径必须在 PR 中明确列为未验收，不得以 mock 结果替代。

### 交付件

- PR + transfer thread 失败状态机说明 + 各子类资源所有权矩阵 + 单测与目标环境验证记录。

---

## 任务 2：[Correctness] hybrid KV load 失败不得使用残缺 KV 进入 forward

> 编号：`kv-28` | 严重程度：高 | 修复优先级：P0 | 难度：4.5/5
>
> 审核状态：问题已闭合，可发布为一个完整的跨 vLLM/vllm-ascend 正确性任务；认领者需同时承担上游协议和 AscendStore 实现，并在编码前完成 sync 终止语义与多 worker failure 聚合语义评审。

### 已确认问题

同步 non-layerwise hybrid load 会在单个 request 内合并多个 KV group 的 keys 后调用 backend `get()`。当返回部分失败或 `None` 时，single-group 分支会报告 invalid block，multi-group 分支却只记录日志，然后从 forward 前的 load hook 正常返回。当前控制流没有其他 non-layerwise load barrier，因此残缺 KV 进入 forward 是代码可达风险，不是概念推测。

async recv 在 multi-group load 失败后只记录日志，随后通过 `set_finished_request(req_id)` 报告传输结束，却没有同时报告 load failure。这里的问题不是发送 finished signal：上游 vLLM 明确要求异步加载即使失败也必须由 `get_finished()` 完成 transfer completion handshake；真正缺失的是与 finished 正交、且不晚于同一轮上报的 failure outcome。

sync/async non-layerwise 都未校验 backend `get()` 返回数量；`record_failed_blocks()` 使用 `zip(block_ids, ret_codes)`，短返回会静默忽略尾部 block。尤其当短返回中的已有 result 全为成功时，当前代码不会进入失败分支，仍可能把未确认加载的 block 当作成功。

当前 `KVConnectorOutput` 为 load completion 提供 `finished_recving`，为 load failure 提供 block-level `invalid_block_ids`，但没有 request/group-level failure 字段。上游 invalid-block 处理又显式假设单 KV group，不能安全消费 hybrid grouped failure。因此 async hybrid 失败不能仅复用 `_invalid_block_ids`，需要在本任务中补齐上游可消费的 request-level failure outcome。

GVA-layerwise 目前只覆盖了 backend 正常返回完整列表时的 invalid GVA 和非零 lease result：multi-group 分支会释放已记录的 lease 并在 forward 前抛错。但这不是完整正确基线。`batch_get_key_info()` 未校验返回长度，`zip()` 对短返回会静默截断；metadata exception、lease exception、lease 短返回以及 partial lease retry 的异常，也没有统一释放当前及先前 group 已取得或可能取得的 lease。

代码证据：

- [pool_worker.py:980](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L980)：sync non-layerwise backend get 和返回结果处理。
- [pool_worker.py:988](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L988)：sync multi-group 失败分支只记录日志。
- [ascend_store_connector.py:210](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L210)：load 是 forward 前 worker hook。
- [kv_transfer.py:1006](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1006)：async multi-group 失败只记录日志。
- [kv_transfer.py:1037](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1037)：async 失败后只有 transfer completion，没有 failure outcome。
- [kv_transfer.py:1655](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1655)：`record_failed_blocks()` 未验证 block/result 数量一致，内部 `zip()` 会截断短返回。
- [vLLM base.py:397](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L397)：异步 load 失败仍必须由 `get_finished()` 报告，失败结果不得晚于同一轮。
- [vLLM outputs.py:226](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/v1/outputs.py#L226)：`finished_recving` 与 `invalid_block_ids` 是正交字段，当前没有 failed request/group 字段。
- [vLLM outputs.py:240](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/v1/outputs.py#L240)：`is_empty()` 必须识别新增 failure 字段，否则 failure-only output 会被丢弃。
- [vLLM model-runner mixin.py:95](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/v1/worker/kv_connector_model_runner_mixin.py#L95)：一条 worker output 路径在 `try/finally` 之前调用 `start_load_kv()`。
- [vLLM GPU kv_connector.py:83](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/v1/worker/gpu/kv_connector.py#L83)：另一条 worker output 路径独立构造 `KVConnectorOutput`。
- [vLLM utils.py:53](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/distributed/kv_transfer/kv_connector/utils.py#L53)：`KVOutputAggregator` 按 expected worker count 聚合 finished，并单独 union invalid blocks。
- [vLLM scheduler.py:2805](https://github.com/vllm-project/vllm/blob/58d3918e3ea0a544ffedadad2ba84559e9c51d8f/vllm/v1/core/sched/scheduler.py#L2805)：invalid-block 处理当前解包单个 KV group，并标有 hybrid allocator TODO。
- [platform.py:944](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/platform.py#L944)：Ascend 对 `kv_load_failure_policy` 的校验以 `fail` 为默认值，并限制 hybrid recompute。
- [recompute_scheduler.py:996](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/core/recompute_scheduler.py#L996)：Ascend recompute scheduler 有独立的 connector output 消费路径。
- [worker.py:686](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/worker/worker.py#L686)：非末级 PP rank 当前只按 finished 字段决定是否透传 connector output。
- [pool_worker.py:1447](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1447)：GVA metadata 使用未校验长度的 `zip()`。
- [pool_worker.py:1473](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1473)：lease 返回长度异常直接抛出，外围没有统一 rollback。
- [pool_worker.py:1520](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1520)：仅已识别 invalid result 的 multi-group 路径会释放已记录 lease 并抛错。

### 任务范围

1. 对 sync/async non-layerwise hybrid 路径注入单 key 失败、某一 group 失败、整批 `None`、返回长度异常和 backend exception，确认 request 状态、connector output、已写入地址以及 attention/forward 是否执行。
2. 同步 multi-group load 任一必要项未确认成功时必须在当前 attention/forward 前停止，禁止只记录日志后正常返回。编码前必须由 vLLM/vllm-ascend maintainer 在 issue 中确认一种最终语义：
   - **engine-fatal**：异常终止 worker/engine，验证进程退出且 connector、metadata 和 thread state 不会被复用；或
   - **recoverable step abort**：即使 `start_load_kv()` 抛错也执行约定的 connector finalization 和 metadata 清理，并证明下一 step 可以继续。
3. 不接受“直接从当前 `start_load_kv()` 抛异常”作为独立完成条件，因为该调用位于 context manager 的 `try/finally` 之前，会跳过 `wait_for_save()`、`get_finished()` 和 `clear_connector_metadata()`。当前任务不要求在同一 batch 内移除单个 sync request 后继续 forward。
4. async recv 失败时仍必须完成 transfer completion handshake：由当前 `set_finished_request()` 或等价机制使 request 在 `get_finished()` 中出现；failure outcome 可以按评审后的协议提前上报或与最终 completion 同轮上报，但不得晚于该 request 的最终 completion，也不能让请求滞留在 `WAITING_FOR_REMOTE_KVS`。
5. 为上游 `KVConnectorOutput` 增加 failed-receiving request IDs 或等价的 request/group-level failure outcome，并闭合所有生产和消费路径：两条 worker output 采集、`KVConnectorOutput.is_empty()`、多 TP/worker `KVOutputAggregator`、默认 scheduler、Ascend recompute scheduler，以及非末级 PP output 透传。该 vLLM companion change 是完成本任务的必需交付。
6. 多 worker 聚合必须定义 failure 与 expected `finished_recving` count 的时序。允许提前发布 failure 或暂存至 completion 齐全后发布，但必须保证跨 step failure 不丢失、不重复消费；若提前失败 request，迟到 completion 仍能安全完成资源回收；若延迟 failure，aggregator 必须持久保存先到状态。
7. 默认 `kv_load_failure_policy=fail` 下只终止受影响的 async request，不把 hybrid group 失败塞入当前只支持单 group 的 `invalid_block_ids` 路径。只有上游 hybrid block manager 明确支持 grouped invalid blocks 后才能开放 recompute；本任务不以 recompute 作为验收目标。
8. GVA-layerwise 必须验证 `batch_get_key_info()`、`batch_add_lease()` 和 partial lease retry 的返回数量。短返回、超长返回、空返回、异常和非零结果均不得静默生成可进入 forward 的不完整 GVA 列表。
9. GVA-layerwise 任一 group 准备失败时，尝试回滚当前 request 在当前及先前 group 已取得的 lease。rollback 成功时结果为 RELEASED；backend remove 失败或返回状态不足以确认时结果为 UNKNOWN。两种结果都必须保留原始 load error，并单独记录 cleanup error。
10. cleanup exactly-once 和幂等要求限定在本地状态机：同一 cleanup action 不重复提交。只有 backend 明确保证重复 remove 幂等时，才能把 backend 侧重复调用列为验收要求。
11. 单 request miss 或部分失败属于 request outcome；只有共享 thread/backend 的系统性不可用才进入 `kv-25` 的 terminal state。`kv-25` 尚未合入不得成为继续正常 forward 或省略 completion handshake 的理由。

本任务的完成边界是：sync hybrid 失败在 forward 前按评审选定的 engine-fatal 或 recoverable 语义闭合生命周期；async hybrid 的 failure outcome 贯穿 worker、聚合、PP 和 scheduler，并最终终止受影响 request；GVA hybrid 的 metadata/lease 异常不进入 forward，且 lease cleanup 得到 RELEASED 或 UNKNOWN 的明确结果。sync batch 内的单 request 隔离和 hybrid per-block recompute 不属于本任务必交付范围。

### 验收标准

- sync non-layerwise 任一 KV group 部分失败、整批 `None`、返回长度异常或 backend exception 时，当前 model execution step 在 attention/forward 前有界失败。engine-fatal 方案验收进程终止且残留状态不复用；recoverable 方案验收 connector finalization、metadata 清理和下一 step 成功执行。
- async non-layerwise 失败 request 具备 transfer completion 和 request-level failure 两个正交结果；scheduler 不执行残缺 KV 请求，不将其永久留在 `WAITING_FOR_REMOTE_KVS`，也不因该请求误终止同批其他成功的 async request。
- 两条 worker output 路径、failure-only `is_empty()`、TP/worker expected-count 聚合、默认/recompute scheduler 和 PP 非末级透传均有针对性测试。
- 多 worker 测试覆盖 failure 先到、completion 先到、同轮到达、重复 worker 通知和跨 step 到达；最终 failure 恰好消费一次，completion count 和资源释放闭合。
- 测试明确断言 `finished_recving` 与 failure outcome 可以同时包含同一 req ID，也允许评审协议规定的 failure 提前上报；任何情况下都不能把 transfer completion 断言为 load success。
- single-group 现有 invalid-block 语义不回归；hybrid failure 不进入上游当前的单-group invalid-block 解包路径。
- GVA-layerwise 覆盖 metadata 短/超长/空返回、metadata exception、lease 短/超长返回、lease exception、partial retry 返回异常、非零 lease result，以及第二个 group 失败时第一个 group 已获 lease 的回滚。
- GVA rollback 成功时验证当前和先前 group 的 lease 均已释放；rollback 失败时验证原始 load error 保留、cleanup error 可观察且 lease outcome 为 UNKNOWN，不声称零遗留。
- 本地 cleanup 状态机对每个 lease action exactly-once；backend 未声明 remove 幂等时不得通过重复 remove 伪造幂等验收。
- 覆盖 sync、async、GVA-layerwise、multi-group 某一 group 失败、per-key 部分失败、backend `None`、backend exception、返回数量异常和多 request 混合结果。
- 至少用一个真实 hybrid 模型配置在 Ascend 环境验证失败传播；backend 故障可由可控 fake 注入，但必须保留真实配置的状态机验证。
- 上游协议测试覆盖旧 connector 未设置新字段时的默认兼容性，以及 single-group、非 hybrid 和其他 connector 不回归。

### 交付件

- vllm-ascend PR + 必需的 vLLM companion PR + completion/failure 状态与 GVA lease 所有权说明 + 两仓针对性测试 + Ascend hybrid 验证记录。

---

## 任务 3：[Perf] 单次 connector step 内批处理 non-layerwise backend 调用

> 编号：`kv-07` | 严重程度：中 | 优先级：P2 | 难度：4/5
>
> 审核状态：当前逐 request 调用和扩批机会已确认，可以作为一个完整性能任务发布；端到端收益必须由本任务的数据决定。

### 已确认事实

当前三条 non-layerwise 路径都按 request 调用 backend：sync load 在 request 循环内构造批量 key/address/size 后调用一次 `get()`；async load 每个 request 单独入队，recv handler 一次只处理一个 `ReqMeta`；save 在一次 connector metadata 中逐 request 入队，sender 再逐 request/group 调用 `exists()` 和 `put()`。

Mooncake、MemCache 和 YuanRong backend 的 `get/put/exists` 接口形状均接受列表，因此跨 request 扩大 batch 在接口层面可行。与此同时，普通 non-layerwise `PoolKey` 不包含 request ID，相同前缀的不同 request 可能生成重复 key；每个 request 又有独立 block ID、mask、event、完成状态和失败结果，必须保存逐项归属并精确回填。

当前代码只能证明调用形态和 batching opportunity，不能证明生产 step 中经常存在多个可合并 request，也不能证明 backend 固定调用成本足以覆盖 descriptor、去重、结果回填和 async 等待成本。因此减少调用次数不能直接等价为 TTFT 或吞吐收益。

代码证据：

- [pool_worker.py:888](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L888)：sync load 位于 request 循环内。
- [pool_worker.py:919](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L919)：async load 按 request 入队。
- [pool_worker.py:980](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L980)：sync load 每个 request 调用 backend get。
- [pool_worker.py:1758](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1758)：save 在 connector step 内逐 request 入队。
- [kv_transfer.py:809](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L809)：sender 在每个 request/group 内查询 exists。
- [kv_transfer.py:890](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L890)：sender 在每个 request/group 内调用 put。
- [kv_transfer.py:923](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L923)：async recv handler 一次处理一个 request。
- [metadata.py:74](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L74)：普通 key metadata 的字段不包含 request ID。
- [metadata.py:95](https://github.com/vllm-project/vllm-ascend/blob/d5e9816065ede613327d93908f87fee9f5c47128/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L95)：`PoolKey` 只组合 key metadata 与 chunk hash。

### 任务范围

1. 在同一个任务内先建立性能基线：分别统计 sync load、async load 和 save 的 step request 数、backend 调用次数、items/call、调用耗时及 queue wait，并在改代码前定义 Go/No-Go 阈值。
2. 在采集正式结果前，由 maintainer 在 issue 中确认至少一个参考 workload、目标 backend/NPU 配置和 Go/No-Go 阈值。贡献者可以补充其他 workload，但不能只选择有利样本替代参考 workload。
3. 逐 backend 确认最大 batch、线程安全、重复 key 语义和 per-item 失败返回能力。实现只覆盖数据证明有价值且 backend 契约足够清晰的路径，不要求为了完成任务强行改写三条路径。
4. 为进入实现的路径建立 transfer descriptor，记录每一项所属 request/group、key、block ID、address、size、mask、event 和结果回填位置。
5. sync load 只合并当前 connector step 已知的 requests，并把长度匹配的 backend per-item result 精确映射回 request/block；不得改变原有 per-request circular-shift 和 multi-group 语义。
6. save 路径可以去重 exists 查询，但必须定义重复 key 的 put 及逐 request 完成语义；若 backend 不能提供可靠的逐项 put 结果，save batching 不进入实现范围，也不能把整个 batch 统一标记成功。
7. backend 抛异常、返回 `None` 或结果数量不匹配时，不允许猜测单项归属；按已确认的 backend 契约把整个合并 batch 标为 FAILED 或 UNKNOWN，并逐 request 结束等待。只有长度匹配且具有 per-item 语义的返回才能做到“只影响对应 request/block”。
8. async recv 若进入实现，只允许有数量和等待时间上限的 bounded batch；不得无限 drain queue，也不得破坏 request 取消、preemption、fatal 和尾延迟语义。
9. profiling 完成后必须依据预先定义的阈值选择一个最终验收出口：达到阈值走 Go 验收并实现有收益的路径；未达到阈值走 No-Go 验收并保持现有实现。两者属于同一个社区任务，不另拆 issue。

### 验收标准

以下共同材料是两种出口都必须提供的：

- 改动前基线包含目标 workload 中每 step 可合并 request 数量分布、backend 调用次数、items/call、调用耗时、queue wait、TTFT、吞吐和尾延迟，不能只展示构造出来的大 batch 微基准。
- 明确工作负载、请求长度与并发、预热、样本数、统计口径、原始数据、NPU 型号和 backend 版本。
- 本任务中的 batch size 1/4/8/16 指同一 connector step 合并的 request 数；同时报告每个 backend call 的实际 item 数量分布，避免把 request 数与 key/item 数混用。
- 记录 Mooncake、MemCache、YuanRong 的最大 batch、线程安全、重复 key 和逐项失败能力；未实测的 backend 必须标为未验证，不能由接口形状推断运行时语义。
- 在查看优化结果前，由 maintainer 在 issue 中确认参考 workload、Go/No-Go 数值阈值及选择理由；结果出来后不得为通过验收临时修改阈值。

#### No-Go 验收出口

- 数据表明目标 workload 大多数 step 无可合并 request、backend 固定调用开销不显著、batch 构建成本抵消收益，或 async 等待使尾延迟超过阈值。
- 在任务 issue comment 中提交 Markdown 基线、成本归因、阈值对比和 No-Go 结论，并附可下载的原始数据或 benchmark artifact；文档 PR 可选，不作为 No-Go 验收前置。No-Go 不修改生产热路径，也不要求“合并前后结果一致”或 batch 故障测试。
- 若为采集数据加入临时 instrumentation，默认不得随 No-Go 结果合入；确需保留时必须证明开销可忽略并单独说明用途。

#### Go 验收出口

- 只实现数据达到 Go 阈值的具体路径；不要求无收益路径为了覆盖标题而一并改写。
- 合并前后每个 request 的 key、目标地址、失败 block、完成状态和 KV event 一致。
- 对进入实现且实际适用的路径覆盖重复 full-block key、partial block、multi-group、store/load mask、skip-null、TP mismatch、request cancel、preemption 和 thread fatal；不要求不可达配置为满足清单而补造测试。
- backend 返回长度匹配的 per-item 部分失败只影响对应 request/block；backend exception、`None` 或长度异常按整批 FAILED/UNKNOWN 传播，不能误标整批成功。
- 对实现路径提供 batch size 1/4/8/16 requests 的 backend 调用次数、实际 items/call、descriptor 构建耗时、queue wait、TTFT、吞吐和尾延迟，并在 maintainer 确认的参考 workload 上达到预设端到端阈值。
- batch size 1 和低并发场景无不可接受回归；async batching 同时满足最大数量和最大等待时间限制。
- 至少选择一个公开支持的 backend 和 Ascend NPU 配置完成端到端验证，并明确其他 backend 是实测、mock 验证还是未覆盖。

### 交付件

- 一个社区任务下统一交付。No-Go：性能基线、成本分析和结论报告。Go：性能基线与结论报告 + PR + descriptor/结果回填设计 + 单测 + 端到端性能数据。

## 发布前硬检查

以下事项必须由发布人在实际创建 GitHub issue 当天完成；本次离线代码审核不代表这些动态状态已经确认：

- 在 `vllm-project/vllm-ascend` 和 `vllm-project/vllm` 实时搜索 open/closed issues 与 PR，确认三个任务没有被新提交覆盖或重复认领。
- 对 `kv-28` 明确注明 [vllm-ascend#9701](https://github.com/vllm-project/vllm-ascend/pull/9701) 和 [vllm-ascend#13116](https://github.com/vllm-project/vllm-ascend/pull/13116) 是相邻的 single-group/async failure 已完成工作；本任务补的是 hybrid request/group-level failure outcome 及其完整消费链路，不能把既有改动重复包装成新任务。
- 向赵鹏博本人确认可用的 GitHub handle，再替换页首验收人；确认前不得用中文姓名构造 mention，也不得仅凭本地 commit author 猜测账号。
- 在 GitHub issue 预览中逐项抽查固定 SHA 代码链接，确认两个仓库的文件和行号均可打开并与正文证据一致。

## 最终发布建议

三项均可作为独立社区任务发布，保持以下三个任务编号，不再拆成子 issue：

1. `kv-25`：[Reliability] 建立 transfer thread 终止式失败协议。
2. `kv-28`：[Correctness] hybrid KV load 失败不得使用残缺 KV 进入 forward。
3. `kv-07`：[Perf] 单次 connector step 内批处理 non-layerwise backend 调用。

修复优先级应按正确性风险理解为 `kv-28 (P0) > kv-25 (P1) > kv-07 (P2)`；完整任务难度按用户指定顺序为：

`kv-25 (5/5) > kv-28 (4.5/5) > kv-07 (4/5)`

其中 `kv-25` 和 `kv-28` 是当前代码已闭合的可靠性/正确性问题；`kv-07` 是当前实现中已确认的批处理机会，但是否形成实际性能提升仍由任务内测量结果决定。
