# issues_10：面向 AscendStore 池化能力的价值排序

> 代码基线：vllm-ascend `main@d5e9816065ede613327d93908f87fee9f5c47128`
>
> 排序日期：2026-08-16
>
> 代码真实性与逐项证据见 [CODE_AUDIT.md](CODE_AUDIT.md)。

## 1. 这里所说的“价值”

本排序只回答一个问题：**该任务对提升 vllm-ascend AscendStore KV 池化能力本身有多大意义？**

它不等同于事故严重度，也不等同于短期修复排期。评价重点是：

1. 是否补齐池化数据面或控制面的核心能力，而不只是修一处局部代码。
2. 是否覆盖更多 backend、模型类型、同步/异步模式和 load/save 路径。
3. 是否让池化从“可以运行”走向“结果可信、故障可控、可以规模化”。
4. 是否为其他池化优化提供必要契约或基础设施。
5. 性能收益是否具有足够大的理论上限，以及能否在目标机器上验证。

因此，正确性任务仍可能排在前面，但原因不是它属于 P0，而是缺少它会让某类池化能力本身不可信。反过来，纯重构即使代码上成立，如果不改善当前池化能力，也会排在后面。

## 2. 池化价值总排序

| 排名 | 任务 | 对池化的核心意义 | 价值判断 |
|---:|---|---|---|
| 1 | [kv-28 hybrid KV load failure](issue_kv-28_C2_multi_group_failure.md) | 建立 multi-group/hybrid 池化的原子加载边界，禁止残缺 KV 被消费 | 核心能力，必须具备 |
| 2 | [kv-25 transfer fatal protocol](issue_kv-25_C1_transfer_thread_exception.md) | 建立池化传输线程统一的启动、失败、取消和资源清理生命周期 | 核心基础设施 |
| 3 | [kv-26 Backend.put result](issue_kv-26_C1-1_backend_put_failure.md) | 让池化写入结果、KV event 和 backend 状态真实可知，并支撑安全批处理 | 核心后端契约 |
| 4 | [kv-27 ZMQ lookup recovery](issue_kv-27_C1-2_zmq_lookup_failover.md) | 让 scheduler-worker 池化 lookup 控制面从永久阻塞变为可恢复服务 | 核心控制面可靠性 |
| 5 | [kv-07 non-layerwise batching](issue_kv-07_P07_non_layerwise_io_merge.md) | 系统性降低普通池化 load/save 的 backend 调用次数，覆盖面最广的性能候选 | 高潜力数据面优化，先验证 |
| 6 | [kv-01 MLA TP read dedup](issue_kv-01_P01_MLA_read_dedup.md) | 降低 MLA TP 内对共享池的重复读取和后端带宽放大 | 高潜力场景优化，先验证 |
| 7 | [kv-08 GVA metadata RPC aggregation](issue_kv-08_P08_gva_meta_rpc_merge.md) | 提升 MemCache GVA layerwise 模式的元数据扩展能力 | 中等价值的窄模式优化，先验证 |
| 8 | [kv-24 typed config parser](issue_kv-24_S5_config_schema.md) | 统一池化配置语义，降低多角色默认值和类型漂移 | 支撑性工程价值 |
| 9 | [kv-17 offset-aware lookup](issue_kv-17_P17_zmq_lookup_full_hashes.md) | 减少长上下文 lookup 的 hash 编码和 IPC payload | 局部性能价值，证据不足 |
| 10 | [kv-31 backend capability model](issue_kv-31_E1_backend_abstraction_split.md) | 为未来新增 GVA backend 提供更清晰的类型边界 | 未来扩展价值，当前可延后 |

最终顺序：

`kv-28 → kv-25 → kv-26 → kv-27 → kv-07 → kv-01 → kv-08 → kv-24 → kv-17 → kv-31`

## 3. 为什么这样排

### 1. kv-28：决定 hybrid 池化是否可信

池化最基本的不变量是：一次声明成功的 KV load 必须得到可完整消费的 KV。当前 multi-group non-layerwise 路径中，某些 key/group 失败后只记录日志，async 路径还会正常 finished。

这不是普通错误处理优化，而是在补齐 hybrid 池化的事务边界：要么相关 groups 全部可用，要么 request 明确失败。没有该边界，hybrid 模型即使性能再高，也不能称为可靠的池化能力。因此它对池化本身的价值最高。

### 2. kv-25：池化传输执行层的生命周期基础

AscendStore 的 load/save 依赖多个后台 transfer thread。当前 startup、fatal、queue accounting、等待方唤醒和 lease/event cleanup 没有统一协议。

它的价值不只在修复一个 hang，而是给同步、异步、layerwise、GVA 等路径建立共同的失败生命周期。后续做 batching、异步加载和 request-level failure 时，都需要依赖这个基础，所以战略价值高于单点的 ZMQ timeout。

### 3. kv-26：池化后端必须能说明“到底存成了什么”

三个 backend 的普通 `put` 当前都没有向 sender 返回可信的 per-key 结果，sender 因而可能发布虚假 `BlockStored`。一个池系统如果不知道哪些对象真正可用，就无法提供可信 event、健康状态、失败指标，也无法安全合并多个 request 的写入。

该任务会形成 `AVAILABLE/FAILED/UNKNOWN` 之类的统一结果契约，直接支撑 kv-07 batching 和后续 backend health/circuit breaker。它是池化后端抽象的实际语义补全，不只是可观测性美化。

### 4. kv-27：池化 lookup 控制面必须可恢复

池化命中查询位于 scheduler 与 worker 之间。当前同步 REQ 无 timeout，server 单次异常可能让 client 永久等待。修复后，lookup 才具有 bounded latency、typed error 和恢复能力。

它是当前必须修的 P0，但从池化架构价值看，它主要补强一条明确的控制面 RPC；覆盖面和后续杠杆略低于 transfer lifecycle 与 backend result contract，所以在本价值排序中列第四。这不代表短期修复排期必须晚于 kv-25/kv-26。

### 5. kv-07：最有系统意义的性能候选

kv-07 同时涉及同步 load、async load 和普通 save，并覆盖 Mooncake、MemCache、Yuanrong 的 non-layerwise 路径。若生产 workload 中单 step 经常有多个 requests，它可以减少 backend 固定调用开销，改善吞吐和 TTFT，是 4 个性能任务中覆盖面最广的一个。

它没有进入前四，是因为实际可合并度尚未由机器数据证明，而且安全实现依赖 kv-25 的取消/fatal 语义和 kv-26 的 per-item 结果。应先 profile 三条路径，只实现数据证明有价值的路径。

### 6. kv-01：对 MLA 池化后端压力有直接价值

MLA 下，同一 TP 组的 ranks 会查询相同 key 集，每个 rank 又各自调用 backend `get`。如果 backend 没有在更低层有效合并，请求量和带宽压力可能随 TP 放大。

这是非常直接的池化流量优化，尤其对 MLA 长上下文模型有意义。但它只覆盖 MLA、non-layerwise、同步读取，并需要引入 collective broadcast 和全 rank 错误一致性，因此整体价值低于跨路径的 kv-07。目标机器若证明读取成本随 TP 明显放大，可将它提升到第五。

### 7. kv-08：提升 GVA layerwise 的元数据规模能力

GVA prepare 接收一批 requests，却仍在 request/group 循环中调用 metadata API。聚合后可能减少 MemCache RPC、allocation 查询和 lease 操作的固定成本。

它确实优化池化，但只适用于 MemCache GVA layerwise，且 `batch_alloc` 非幂等、lease 语义由外部服务版本决定。覆盖范围和实施确定性都低于 kv-07/kv-01，所以排第七。若真实环境显示 metadata RPC 是 layerwise TTFT 主项，可以上调。

### 8. kv-24：提高池化可运维性，不直接提升数据面

统一 typed config 能减少 scheduler、worker、connector 和 layout 的默认值/类型漂移，也能给 lookup timeout 等新参数提供稳定入口。它对长期维护和配置安全有明确价值。

但它既不直接减少数据搬运/RPC，也不修复当前已确认的池化结果错误。现有代码还有局部 typed parser 可复用，因此更适合作为支撑性治理，排在核心能力和高潜力性能候选之后。

### 9. kv-17：优化的是报文局部开销，不是池化主链路

完整 hashes 的编码和 ZMQ 发送确实可以缩减，但它只优化 lookup payload。为了正确发送 suffix，还需要 absolute offset、版本兼容、错误校验以及 hybrid/non-hybrid 语义适配。

在没有数据证明 `.hex()`、msgpack 和 IPC 是 lookup/TTFT 主要开销前，这项改造对整个池化系统的杠杆较小。更合适的形态是先做 benchmark；数据不显著时不应进入完整协议开发。

### 10. kv-31：当前更多是代码架构价值

拆分 `Backend` 与 `GVABackend` capability 可以让未来新增 backend 更清晰。但当前代码已经通过 `backend_name == "memcache"` 选择 GVA 路径，固定 backend 组合下没有可达的错误配置。

它对当前池化性能、正确性和可用性都没有直接改善。若近期没有新增 GVA backend 的计划，不值得单独占用一个高优先级任务；可在新增 backend 或相关模块重构时合并完成。

## 4. 哪些真正值得独立投入

### 核心池化任务：建议保留并投入

- `kv-28`：hybrid load 原子性和失败传播。
- `kv-25`：transfer 生命周期与资源所有权。
- `kv-26`：backend 写入结果与 event 真实性。
- `kv-27`：lookup 控制面恢复。
- `kv-07`：普通池化数据面的系统性 batching，先做 Phase 0。

这 5 项分别覆盖 hybrid 正确性、执行层、后端契约、控制面和数据面效率，合在一起对池化系统最有整体意义。

### 重要场景优化：由部署形态决定

- `kv-01`：部署以 MLA + TP + non-layerwise 为主时价值较高。
- `kv-08`：部署以 MemCache GVA layerwise 为主时价值较高。

二者不是普适任务，但在对应生产配置中可能高于 kv-07，必须由目标环境决定。

### 支撑性任务：可以做，但不应抢核心资源

- `kv-24`：配置治理。

### 独立立项价值较弱

- `kv-17`：先 benchmark；没有明确 TTFT 收益则不开发协议。
- `kv-31`：没有近期 backend 扩展时，建议合并到未来重构，不单独推进。

所以，严格回答“这 10 个是否都同样有价值”：**不是**。前 5 个对池化系统有明显的独立价值；`kv-01`、`kv-08` 是部署相关价值；`kv-24` 是支撑价值；`kv-17`、`kv-31` 当前独立投入价值最低。

## 5. 价值排序与实施排期的区别

价值排序不意味着必须严格串行。按当前风险，实际实施仍建议：

1. 先处理 `kv-28` 和 `kv-27`，立即关闭错误计算与永久阻塞。
2. 推进 `kv-25`、`kv-26`，建立 transfer 和 backend 契约。
3. 完成 `kv-07`、`kv-01`、`kv-08` 的 Phase 0，按数据决定后续投入。
4. `kv-24` 可与新增配置需求一起渐进落地。
5. `kv-17`、`kv-31` 在 benchmark 或扩展需求出现前保持候选状态。

这里 `kv-27` 的实施排期早于它在架构价值中的第四名，是因为当前永久阻塞风险紧迫；这正是“修复优先级”和“池化战略价值”需要分开的原因。

## 6. 最终结论

从优化 vllm-ascend 池化能力的意义看，最重要的不是把 10 个任务按类别排队，而是先补齐四个池化核心边界：hybrid load 原子性、transfer 生命周期、backend 写入结果、lookup 可恢复性；然后用 kv-07 优化通用数据面。

若资源有限，只优先保留 5 个任务，建议是：

`kv-28、kv-25、kv-26、kv-27、kv-07`

如果主要服务 MLA 模型，再加入 `kv-01`；如果主要使用 MemCache GVA layerwise，再加入 `kv-08`。`kv-17` 和 `kv-31` 当前不应与这些核心任务争夺同等资源。
