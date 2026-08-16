# AscendStore Top 5 独立源码审核

> 审核日期：2026-08-16
>
> 源码仓库：`D:\lzy\project\kv_pool\code\vllm-ascend`
>
> 审核分支：`main`
>
> 审核提交：`d5e9816065ede613327d93908f87fee9f5c47128`
>
> 审核对象：`kv-28、kv-25、kv-26、kv-27、kv-07` 及其余 5 个候选的横向比较

## 1. 审核原则

本报告没有采用 `CODE_AUDIT.md`、`VALUE_RANKING.md` 或 `TOP5.md` 的结论作为证据。上述文件只用于定位待核对主张；所有判断均重新从当前提交的生产代码、调用链和测试出发。

证据分为三类：

1. **代码闭合证明**：当前源码能够证明可达路径、错误传播或接口缺口，不需要性能数据才能确认问题存在。
2. **外部契约待确认**：本仓库只能看到 wrapper，真实 backend SDK/服务的返回码、幂等性或 lease 语义仍需按实际版本确认。
3. **机器数据待确认**：源码只能证明存在减少调用或报文的机会，不能证明 Ascend NPU 上的 TTFT、吞吐或尾延迟收益。

审核同时分开评价五个维度：

- 对 AscendStore 池化能力的战略价值；
- 当前故障的紧迫度；
- 实际实施顺序；
- 任务边界和依赖是否准确；
- 社区贡献者能否独立实现并验收。

## 2. 总结论

### 2.1 Top 5 成员是否合理

**四个可靠性任务应保留，`kv-07` 只能作为有条件的第五名。**

- `kv-28`、`kv-25`、`kv-26`、`kv-27` 均有当前代码可以闭合证明的真实问题。
- `kv-07` 只能证明 batching opportunity，不能证明 batching 已经或必然提升性能。
- 其余五个候选中，没有一个在“部署无关、直接提升池化能力”这一目标下能够凭现有证据明确替代 `kv-07`。
- 因此，若 Top 5 表示“4 个必须修复的基础问题 + 1 个最值得优先验证的通用性能方向”，当前成员集合可以保留。
- 若 Top 5 表示“5 个已经证明有收益、现在即可完整实施的优化任务”，当前集合不成立，因为 `kv-07` 尚未越过 Phase 0。

### 2.2 当前排名是否合理

当前 `kv-28 → kv-25 → kv-26 → kv-27 → kv-07` 把战略价值、故障紧迫度和任务可落地性放进了同一个序号，容易被误读。

独立审核后的战略价值判断为：

`kv-28 > kv-25 ≈ kv-27 > kv-26 > kv-07（条件性）`

其中 `kv-27` 不应排在 `kv-26` 之后：

- `kv-27` 位于所有 non-layerwise 远端命中查询的同步控制面，一次 server 异常即可永久阻塞 scheduler。
- `kv-26` 的确定影响主要是 save 结果不可观测和 KV event 虚报；普通 save 失败通常在后续表现为 miss，不直接破坏当前 producer 的 forward。
- `kv-26` 所声称的“backend 状态真实可知”还受 Yuanrong 和整批异常缺少逐项结果的限制，最多只能可靠标记为 `UNKNOWN`。

按当前事故紧迫度，应为：

`kv-28 ≈ kv-27 > kv-25 > kv-26 > kv-07`

### 2.3 社区任务可发布性

| 任务 | 问题真实性 | 当前草稿能否独立闭环 | 审核意见 |
|---|---|---|---|
| `kv-28` | 闭合成立 | 否 | 拆成近期 fail-stop 与长期 request-level 上游协议 |
| `kv-25` | startup 缺陷闭合；运行期协议缺口成立 | 否 | 拆成通用 terminal state 与各子类资源清理 |
| `kv-26` | 闭合成立 | 有条件 | 收窄为 tri-state put 结果和 event 准确性；外部 SDK 契约为前置条件 |
| `kv-27` | 闭合成立 | 是 | 五项中最适合直接发布；补全 context/thread 关闭验收 |
| `kv-07` | 调用形态成立，收益未证实 | Phase 0 可以 | 只能以 benchmark/Go-No-Go 任务发布，Go 后再开实现 PR |

## 3. 逐项审核

### 3.1 kv-28：hybrid KV load 失败传播

#### 源码能够证明的事实

同步 non-layerwise load 在单个 request 内合并多个 group 的 key 后调用一次 backend `get`。当返回部分失败或 `None` 时：

- single-group 会把失败 block 放入 `_invalid_block_ids`；
- multi-group 只记录日志，然后从 `start_load_kv` 正常返回。

证据见 [pool_worker.py#L980](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L980) 和 [pool_worker.py#L986](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L986)。`start_load_kv` 是 forward 前的 worker hook，正常返回后没有其他 non-layerwise load barrier，见 [ascend_store_connector.py#L210](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L210)。因此“残缺 KV 仍可进入 forward”不是推测，而是可达控制流。

async recv 的问题更直接：multi-group 失败同样只记日志，随后仍调用 `set_finished_request(req_id)`，见 [kv_transfer.py#L997](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L997) 和 [kv_transfer.py#L1037](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1037)。scheduler 收到 finished receiving 后会移除 loading 状态，见 [pool_scheduler.py#L1151](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1151)。因此 async 路径会把失败 load 当成正常完成。

GVA layerwise 不具有相同缺陷。其 multi-group metadata/lease 失败会释放已取得 lease 并抛出异常，见 [pool_worker.py#L1520](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1520)。该路径应保留为回归基线，不应作为本问题的主要修复对象。

#### 当前任务边界的问题

草稿同时要求：

- 近期在缺少 request-level API 时终止整个 engine step；
- 长期增加 failed request IDs，只终止受影响 request；
- 验收时其他成功 request 不被误终止。

这三点不能在当前接口下由一个独立 PR 同时满足。整步 fail-stop 能保证不错误计算，但不能满足 request 隔离；request 隔离又需要 vLLM connector output/scheduler 的上游接口支持。

另一个边界问题是 async 失败不应天然等同于 transfer thread fatal。单个 request 的 backend miss/部分失败应进入 request outcome；只有线程或 backend 系统性不可用才应进入 `kv-25` 的 terminal state。否则一个 request 的 load 失败会杀死整个共享 recv thread。

#### 建议拆分

1. **kv-28A：vllm-ascend 近期安全修复**。sync multi-group 失败在 forward 前明确 fail-stop；async 增加失败完成状态，绝不调用正常 finished。允许整步失败，验收重点是零错误计算。
2. **kv-28B：request-level load failure 协议**。与上游 vLLM 定义 failed request IDs 或等价结果，只终止受影响 request；单独验收请求隔离和调度状态恢复。

#### 独立结论

- 问题真实性：**闭合成立**。
- 战略价值：**五项最高**，它决定 hybrid 池化结果是否可信。
- 社区可交付性：**当前草稿不可一次闭环，拆分后可交付**。

### 3.2 kv-25：transfer thread 终止式失败协议

#### 源码能够证明的事实

startup hang 是确定性缺陷。`KVTransferThread.run` 在进入任何异常捕获前调用 `set_device()`，成功后才设置 ready event，见 [kv_transfer.py#L496](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496)。worker 在四类启动路径上直接无超时 `wait()`，见 [pool_worker.py#L488](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L488)、[pool_worker.py#L505](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L505)、[pool_worker.py#L544](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L544) 和 [pool_worker.py#L562](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L562)。`set_device()` 抛错时 creator 会永久等待。

运行期 terminal state 也不完整：

- `run` 捕获 handler 异常后只保存 `_fatal_error` 并退出，见 [kv_transfer.py#L503](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L503)。
- 基类定义了 `_handle_request_exception`，但 `run` 从未调用它，见 [kv_transfer.py#L523](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L523)。
- `add_request` 不检查 fatal 状态，线程退出后仍可继续入队，见 [kv_transfer.py#L347](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L347)。
- non-layerwise save 使用 `request_queue.join()`，见 [pool_worker.py#L1770](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1770)。fatal 后遗留的排队项永远不会被 `task_done`，可造成永久等待。
- async non-layerwise `get_finished` 不调用 recv thread 的 `raise_if_failed`，见 [pool_worker.py#L2095](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2095)。请求可能长期停在 `WAITING_FOR_REMOTE_KVS`。
- GVA recv 只在 final layer 释放 lease，见 [kv_transfer.py#L1633](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1633)。中途 fatal 没有统一 cleanup。

但不能把上述事实扩大成“所有 handler 异常都会挂死”。普通 sender 和 async recv 已分别通过 `finally` 对当前项调用 `task_done`，见 [kv_transfer.py#L697](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L697) 和 [kv_transfer.py#L1038](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1038)；layerwise worker wait 也会周期检查 recv fatal，见 [pool_worker.py#L1701](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1701)。实际遗留资源随子类和异常点不同。

#### 当前任务边界的问题

该任务同时覆盖：

- startup handshake；
- 通用 fatal 状态和拒绝入队；
- queue drain/cancel；
- 普通 send、async recv、key layerwise、GVA layerwise；
- NPU event、per-request counter 和 MemCache lease cleanup；
- model runner/engine 的失败传播。

这是一个跨基类、六个子类、worker 和 scheduler 的并发状态机改造，不适合作为一个普通社区 issue 一次性交付。尤其“排空 queue”不能只做 `get_nowait + task_done`；每个被取消项还必须得到明确的 request failure、counter/event 清理和 owner 通知。

#### 建议拆分

1. **kv-25A：thread startup 与 terminal core**。startup outcome、有限等待、首次 fatal 原因、fatal 后拒绝入队、base run 调用异常钩子、等待方统一检查失败。
2. **kv-25B：per-subclass ownership cleanup**。按 normal send、async recv、key layerwise、GVA layerwise 分矩阵验证 queue item、request state、event、counter 和 lease；可继续拆成普通路径与 GVA 路径两个 PR。

#### 独立结论

- 问题真实性：**startup 完全闭合；运行期统一协议缺口成立**。
- 战略价值：**很高**，但当前文案把多个任务包装成一个“统一基础设施”。
- 社区可交付性：**原范围过大，必须拆分**。

### 3.3 kv-26：Backend.put per-key 结果

#### 源码能够证明的事实

`Backend.put` 没有返回契约，见 [backend/base.py#L50](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L50)。三个实现均隐式返回 `None`：

- Mooncake 已获得 batch 返回码，但只记录负值，见 [mooncake_backend.py#L189](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L189)。
- MemCache 已获得 batch 返回码，但只记录非零值，见 [memcache_backend.py#L210](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L210)。
- Yuanrong wrapper 没有捕获任何逐项结果，见 [yuanrong_backend.py#L147](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py#L147)。

三个 wrapper 还会吞掉 binding exception，只记日志后正常返回。sender 在调用 `put` 前构造所有 `BlockStored`，调用后不检查结果便发布，见 [kv_transfer.py#L846](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L846)、[kv_transfer.py#L890](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L890) 和 [kv_transfer.py#L891](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L891)。因此 KV event 虚报是可达事实。

本地 GPU block 生命周期与远端持久化结果必须分开。sender 在 handler 结束后更新 stored counter/finished 状态，见 [kv_transfer.py#L697](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L697)。远端失败不应让 producer 永久占用本地 block，这一任务边界是正确的。

#### 当前任务边界的问题

“让 backend 状态真实可知”表述过强。当前源码只能证明 Mooncake/MemCache wrapper 丢失了已有返回码，不能证明：

- 各 SDK 中非零码是否表示失败、重复对象还是对象已经可用；
- batch exception 前是否已有部分 key 成功；
- Yuanrong 的实际 binding 是否存在未使用的逐项结果；
- post-write `exists` 是否具有足够强的一致性语义。

因此统一契约可以保证的是“每个输入 key 都有 `AVAILABLE/FAILED/UNKNOWN` 结果”，而不是“每个 key 的真实状态都一定已知”。对于无法逐项判断的整批异常，正确结果是全部或部分 `UNKNOWN`，不能伪造 `FAILED` 或 `AVAILABLE`。

草稿中的 backend health/circuit breaker 属于后续能力，不是修复虚假 event 所必需，应从本 issue 的必交付范围移除。验收项“后续 lookup 对失败 key 表现为 miss”也不稳定：对象可能在竞争写入中已存在，backend 可能最终一致，或 exception 前已经完成部分写入。验收应改为 normalized result、event 过滤和本地生命周期正确。

#### 建议边界

- 固定三种 SDK/服务版本并记录原始语义。
- `put` 返回与输入等长的 normalized result；无证据时使用 `UNKNOWN`。
- 仅 `AVAILABLE` 发布 `BlockStored`。
- 保存完成通知继续释放本地 block。
- health/circuit breaker 另立后续 issue。

#### 独立结论

- 问题真实性：**闭合成立**。
- 战略价值：**中高**，主要提升 event 真实性、可观测性和安全 batching，不是当前模型正确性 P0。
- 社区可交付性：**代码边界可控，但验收必须能访问真实 backend 版本**。

### 3.4 kv-27：ZMQ lookup timeout 与恢复

#### 源码能够证明的事实

所有 non-layerwise scheduler lookup 都同步调用 `LookupKeyClient.lookup`，见 [pool_scheduler.py#L565](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L565)。client 发送 multipart 后直接阻塞 `recv()`，没有 timeout、poll 或异常恢复，见 [pool_scheduler.py#L1168](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1168)。

server daemon 在 receive、frame decode、字段访问、backend lookup 和 response conversion 外层没有逐请求 `try/except`，见 [ascend_store_connector.py#L312](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L312)。其中任一异常都会终止 daemon thread；已发出请求的 client 将永久等待。这一因果链不依赖机器数据。

降级为 remote miss 的边界也能由调用链证明。调用时已有 HBM 命中通过 `num_computed_tokens` 传入，远端 lookup 只计算额外命中，见 [pool_scheduler.py#L570](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L570)。远端失败返回 0 只放弃新增 remote hit，不会删除已有 HBM prefix。

REQ/REP timeout 后不能直接复用原 REQ socket 发送下一请求，这是协议状态机约束；重建 socket 的任务方向正确。server 在 receive 成功后则必须完成一次 success/error send，才能继续使用原 REP socket。

#### 当前任务边界的问题

任务主体清晰、局部且可用 CPU 上的真实 pyzmq 测试验收，不依赖 NPU。需要补充的边界是生命周期：

- `LookupKeyClient.close` 只关闭 socket，没有终止 context，见 [pool_scheduler.py#L1190](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1190)。
- `LookupKeyServer.close` 只关闭 socket，没有设置 `running=False`、等待 thread 退出或关闭 context，见 [ascend_store_connector.py#L338](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L338)。

“重复重建无 FD/context 泄漏”必须覆盖 socket、context 和 daemon thread 三者，而不只是 socket。

#### 依赖判断

`kv-27` 与 `KVTransferThread` 没有代码依赖。client/server 是独立的 ZMQ 控制面类，因此 `TOP5.md` 中 `kv-27 → kv-25 transfer failure lifecycle` 的箭头不成立。它也不应等待 typed config parser；可先使用局部校验的 timeout 默认值。

#### 独立结论

- 问题真实性：**完全闭合**。
- 战略价值：**很高**，覆盖所有 non-layerwise backend 的命中控制面。
- 社区可交付性：**五项最好**，任务边界最清楚，可在无 NPU 环境完成大部分状态机测试。

### 3.5 kv-07：non-layerwise backend batching

#### 源码能够证明的事实

三个路径确实按 request 调用 backend：

- sync load 在 `for request` 内构造 key/address/size 并调用一次 `get`，见 [pool_worker.py#L888](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L888) 和 [pool_worker.py#L980](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L980)。
- async load 为每个 request 单独入队，见 [pool_worker.py#L918](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L918)，recv handler 一次只处理一个 `ReqMeta`，见 [kv_transfer.py#L923](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L923)。
- save 在一次 connector metadata 中逐 request 入队，见 [pool_worker.py#L1758](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1758)，sender 再逐 request/group 调用 `exists` 和 `put`，见 [kv_transfer.py#L741](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L741)。

三个 backend 的 `get/put/exists` 接口都接受列表，所以聚合多个 request 在接口形状上可行。普通 `PoolKey` 不含 request ID，见 [metadata.py#L94](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L94)，跨 request 重复 key 和逐项归属确实需要 descriptor 处理。

#### 源码不能证明的内容

当前仓库没有以下证据：

- 生产 step 中可合并 request 数的分布；
- backend 单次调用的固定成本占比；
- 扩大已有 batch 后是否提升 backend 吞吐；
- descriptor 构建、去重和结果回填成本；
- async 微批等待对单请求尾延迟的影响；
- TTFT 或吞吐的端到端净收益。

该改动只确定能减少调用次数，不确定能减少传输数据量。当前每个 request 本身已经把多个 key 作为一批交给 backend；跨 request 扩批是否有价值取决于 backend 固定开销和实际 workload。

#### 当前任务边界和依赖

Phase 0 可以独立执行，不依赖其他四项。Go 后的实现依赖应按路径区分：

- sync load 已有 per-key `get` 结果，不硬依赖 `kv-25/26`。
- save 要准确回填批量结果，硬依赖 `kv-26` 的 normalized put result。
- async 微批涉及 queue 等待、取消和 fatal，宜在 `kv-25A` 的 terminal core 后实施。

因此 `TOP5.md` 中笼统的 `kv-26 → kv-07` 只对 save 成立；“kv-07 必须等待 kv-25/26”对 Phase 0 和 sync load 都不成立。

当前 issue 已允许只实现 profile 证明有价值的一条路径，这是正确收窄。但它仍要求社区贡献者具备目标 NPU、真实 backend 和代表性 workload；没有这些资源，只能完成合成 benchmark，不能完成端到端验收。

#### 独立结论

- 问题真实性：**batching opportunity 成立，性能收益未闭合**。
- 战略价值：**条件性中高**；作为通用性能候选覆盖面最广，但不能写成已证明优化。
- 社区可交付性：**Phase 0 可独立发布；实现任务应在 Go 后另开**。

## 4. 与其余五个候选的横向比较

| 候选 | 独立源码结论 | 是否应替代当前 Top 5 |
|---|---|---|
| `kv-01` MLA TP read dedup | MLA 时 `num_kv_head=1`，同一 TP 组的 `head_or_tp_rank` 均为 0，而每个 rank 都执行完整 backend `get`；重复调用形态成立。leader+broadcast 的成本、KV 布局适配和 collective 失败一致性均未验证 | 不作通用替代；MLA 主导部署中可能高于 `kv-07` |
| `kv-08` GVA metadata RPC aggregation | `_alloc_gvas_for_save`、`_prepare_load_gvas` 均在 request/group 循环内调用 metadata API，聚合机会成立；但仅覆盖 MemCache GVA layerwise，且受非幂等 alloc 和 lease 契约约束 | 不替代，范围更窄、外部依赖更多 |
| `kv-24` typed config | scheduler、worker、connector 存在分散读取和不一致 coercion；但 layerwise layout 已有 typed parser 和 direct/MultiConnector helper。属于确定的工程治理，不是已闭合的池化故障 | 若目标是“最容易交付的五项”可考虑；若目标是“最大池化能力价值”不替代 |
| `kv-17` offset-aware lookup | client 确实发送完整 hashes；worker 的 chunk/grouped hash 依赖绝对 token 位置，suffix 必须携带 offset。payload/编解码是否为 TTFT 主项未知 | 不替代；先 benchmark，且协议演进应建立在 `kv-27` 的可靠 wire protocol 之后 |
| `kv-31` backend capability | `Backend` 确实混入 GVA API；但所有 GVA 路由均由 `use_layerwise && backend == memcache` 保护，当前三个 backend 组合下无用户可达错误路径 | 不替代，仅是未来扩展治理 |

横向结论：`kv-07` 不是已证明收益最大的第五项，只是部署无关前提下覆盖面最广的性能验证候选。`kv-01` 在 MLA + TP 为主的目标环境中可能超过它；`kv-08` 在 MemCache GVA layerwise 为主的环境中也可能超过它。Top 5 第五名必须保留部署条件和 Go/No-Go 语义。

## 5. 排名、紧迫度和实施顺序

### 5.1 战略价值

`kv-28 > kv-25 ≈ kv-27 > kv-26 > kv-07（条件性）`

- `kv-28` 决定 hybrid pool 是否会消费残缺数据。
- `kv-25` 建立共享传输执行层的失败生命周期，但当前任务需要拆分才能兑现该价值。
- `kv-27` 建立 non-layerwise lookup 的有界、可恢复控制面，实际覆盖面和确定性高于 `kv-26`。
- `kv-26` 使 save result 和 event 不再虚报，并为 save batching 提供契约。
- `kv-07` 只有机器数据达到 Go 条件后才产生实际能力增益。

### 5.2 故障紧迫度

`kv-28 ≈ kv-27 > kv-25 > kv-26 > kv-07`

- `kv-28` 可能形成静默错误计算。
- `kv-27` 可永久阻塞 scheduler。
- `kv-25` 已有确定 startup hang，运行期影响随路径变化。
- `kv-26` 主要形成虚假 event、不可观测写失败和后续 miss。
- `kv-07` 不是故障修复。

### 5.3 建议实施顺序

1. 并行实施 `kv-28A` 和 `kv-27`，分别关闭错误计算与永久阻塞。
2. 实施 `kv-25A`，建立 startup/terminal core。
3. 实施 `kv-26`，建立 normalized put result 和准确 event。
4. 实施 `kv-25B` 与 `kv-28B`，补齐子类资源清理和 request-level 上游失败协议。
5. `kv-07` 的 Phase 0 可从第一阶段并行采集数据；只有 Go 后才按 `sync load → save → async` 的实际收益选择实现路径，而不是一次性重写三条路径。

## 6. 依赖关系修正

建议使用以下关系替代当前 `TOP5.md` 的依赖图：

```text
kv-28A 近期 fail-stop ---------------------- 独立 P0
kv-27 bounded lookup ----------------------- 独立 P0

kv-25A terminal core ----> kv-25B subclass cleanup
          |                         |
          +----> kv-28 async outcome integration
          +----> kv-07 async microbatch（Go 后）

kv-26 normalized put result ----> kv-07 save batching（Go 后）

kv-07 Phase 0 ------------------------------ 独立，可提前执行
kv-28B request-level failure -------------- 依赖上游 vLLM 协议
```

关键修正：

- `kv-27` 不依赖 `kv-25`，两者分别属于 ZMQ lookup 控制面和 transfer thread 数据面。
- `kv-28` 只有 async outcome/terminal 部分需要与 `kv-25` 协调；load 原子性本身不是 `kv-25` 的子任务。
- `kv-26` 只阻塞 `kv-07` 的 save batching，不阻塞 Phase 0 或 sync load batching。
- `kv-07` 不应作为一个三路径同时交付的大改造。

## 7. 测试与证据边界

本次在 Windows、无真实 `torch/vllm/NPU/backend SDK` 环境中，使用该目录自带 `_mock_deps.py` 和已有临时 pytest/numpy/pyzmq 依赖执行了 AscendStore 相关 UT。

聚焦的五个主要测试文件结果为：

- `293 passed`
- `10 skipped`
- `1 failed`

唯一失败是测试 mock 缺少 `torch.ones`，发生在 `test_as_cache_tuple_list`，不是被审核路径的生产代码失败。加入 metadata/coordinator 测试后有 `350 passed`；另外两个 coordinator 用例因 mock 的上游 `_find_longest_cache_hit` 返回形状不完整而失败。完整目录 collection 还会因 layerwise 测试缺少 `vllm.utils.torch_utils` mock 而停止。

排除上述唯一的 mock-only `torch.ones` 用例后，五个主要测试文件复跑结果为 `293 passed, 10 skipped, 1 deselected`。

这些结果只能说明现有轻量 UT 的当前断言大部分通过，不能替代以下验证：

- hybrid sync/async 故障注入是否阻断真实 forward；
- NPU event、stream 和 GVA lease 的异常 cleanup；
- 三种 backend 实际版本的 put 返回语义；
- ZMQ timeout/rebuild 的真实 socket 状态机测试；
- `kv-07` 的目标硬件性能数据。

现有测试也没有覆盖本报告识别出的主要缺口：startup failure、fatal 后拒绝入队/排队项取消、hybrid async 失败状态、put 失败时 event 过滤、ZMQ server 异常后的 client 恢复和 batching 性能基线。

## 8. 对 TOP5.md 的最终审核意见

当前 Top 5 **成员集合可有条件保留，但排名、依赖图和任务边界不能直接定稿**。

必须修正的内容：

1. 明确这是“4 个已证明基础问题 + 1 个条件性性能验证”，不能把 `kv-07` 写成已证明收益。
2. 不再使用一个序号同时表达战略价值、事故优先级和实施顺序。
3. 将 `kv-27` 在战略/实施判断中提升到 `kv-26` 之前，至少不能让读者按当前 3/4 名直接排期。
4. 删除 `kv-27 → kv-25` 的伪依赖。
5. 将 `kv-28` 拆成近期 fail-stop 与长期 request-level 协议。
6. 将 `kv-25` 拆成 terminal core 与 per-subclass cleanup。
7. 将 `kv-26` 的目标从“所有状态真实可知”收窄为 tri-state normalized result；去掉 circuit breaker 必交付项。
8. 将 `kv-07` 固定为 Phase 0 issue；Go 后按具体路径另开实现任务。

## 9. 五项任务难度排序

以下按“完成任务完整 Go 路径并通过真实环境验收”的难度，从难到易排序；不是价值或优先级排序。

| 难度排名 | 任务 | 难度 | 主要原因 |
|---:|---|---:|---|
| 1 | `kv-25` | 5/5 | 跨基类、六个 transfer 子类、queue accounting、并发 terminal state、NPU event、request counter 和 MemCache lease；异常 cleanup 需 exactly-once 且不能覆盖首个 fatal |
| 2 | `kv-28` | 4.5/5 | 同时涉及 sync/async/hybrid/GVA 回归与 scheduler 状态；完整 request 隔离需要上游 vLLM connector 协议，存在跨仓协作 |
| 3 | `kv-07` | 4/5 | Go 路径需要 descriptor、重复 key、逐项回填、三 backend 契约、取消/fatal 语义和真实 NPU 性能验证；仅 Phase 0 时约为 2.5/5 |
| 4 | `kv-26` | 3.5/5 | 要改公共 backend 契约和三个实现，必须验证实际 SDK 的重复对象、部分失败和异常语义；代码改动集中但外部验证成本高 |
| 5 | `kv-27` | 3/5 | ZMQ REQ/REP 状态机需要严谨处理 timeout、typed error、socket/context/thread 重建和协议校验，但改动局部，大部分测试无需 NPU |
