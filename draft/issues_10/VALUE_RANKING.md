# issues_10 任务价值排序

> 排序基线：vllm-ascend `main@d5e9816065ede613327d93908f87fee9f5c47128`
>
> 排序日期：2026-08-16
>
> 代码证据与逐项真实性审计见 [CODE_AUDIT.md](CODE_AUDIT.md)。

## 1. 排序原则

本排序评价的是“现在投入开发资源是否值得”，不是按 issue 编号，也不只看标题中的 P0/P1/P2。

排序依次考虑：

1. **用户结果风险**：错误输出或错误 KV 状态优先于可恢复失败；永久阻塞优先于性能和代码整洁。
2. **证据确定性**：当前代码可以闭合证明的缺陷，优先于尚未经过目标机器验证的性能假设。
3. **影响范围**：影响多 backend、多请求模式或公共生命周期的任务优先于窄配置任务。
4. **依赖和杠杆价值**：能为后续任务提供失败契约、配置入口或状态传播基础的任务适当前移。
5. **投入产出和实施风险**：收益相近时，优先边界清楚、容易回归验证的任务；协议改造、collective 和外部 lease 语义会降低当前排序。

性能任务的排序是“先做 Phase 0 验证”的价值排序，不代表已经批准完整实现。机器数据不支持时，应以 No-Go 报告收口。

## 2. 最终排序

| 排名 | 任务 | 价值层级 | 为什么排在这里 | 当前建议 |
|---:|---|---|---|---|
| 1 | [kv-28 hybrid KV load failure](issue_kv-28_C2_multi_group_failure.md) | S：必须优先 | 存在残缺 multi-group KV 被当作加载完成并进入 forward 的风险，直接触及模型输出正确性 | 立即做近期 fail-fast；再设计 request-level 失败通道 |
| 2 | [kv-27 ZMQ lookup recovery](issue_kv-27_C1-2_zmq_lookup_failover.md) | S：必须优先 | client 无 timeout，server 异常可导致永久等待；故障链由代码完整证明 | 立即修复 timeout、typed reply 和 socket 重建 |
| 3 | [kv-25 transfer fatal protocol](issue_kv-25_C1_transfer_thread_exception.md) | S：必须优先 | startup 可无界等待，运行期缺少统一 queue/request/resource 终止协议；还是 kv-28 async 传播的基础 | 先做故障矩阵，再统一 startup/fatal/cleanup 状态机 |
| 4 | [kv-26 Backend.put result](issue_kv-26_C1-1_backend_put_failure.md) | A：高价值 | 三种 backend 的失败信息在 wrapper 边界丢失，可能发布虚假 `BlockStored`，且阻碍后续安全 batching | 先核对真实 backend 返回语义，再统一 per-key 结果 |
| 5 | [kv-24 typed config parser](issue_kv-24_S5_config_schema.md) | A：基础治理 | 26 处配置读取分散在多个角色，统一入口能降低默认值/类型漂移，并承接 timeout 等可靠性参数 | 复用现有 layerwise parser/helper，渐进式收敛；不阻塞 kv-27 |
| 6 | [kv-01 MLA TP read dedup](issue_kv-01_P01_MLA_read_dedup.md) | B：条件性高价值 | MLA TP 内重复查询相同 key 的调用形态明确，理论节省可随 TP 放大；但 broadcast 成本和 hang 风险未知 | 先测独立 `get` 与 broadcast 成本，达到 Go 条件再做 PoC |
| 7 | [kv-07 non-layerwise batching](issue_kv-07_P07_non_layerwise_io_merge.md) | B：条件性价值 | 覆盖同步 load、async load 和 save，潜在范围较广；但真实 step 内可合并 request 数尚未证明，状态回填复杂 | 先 profile 三条路径；优先实现有数据支持的单一路径，并复用 kv-26 |
| 8 | [kv-08 GVA metadata RPC aggregation](issue_kv-08_P08_gva_meta_rpc_merge.md) | B：条件性价值 | request/group 内多次 metadata RPC 的代码形态存在，但仅覆盖 MemCache GVA layerwise，且 lease 语义依赖外部契约 | 先验证 RPC 占比和 lease 语义，只聚合被证明安全的操作 |
| 9 | [kv-17 offset-aware lookup](issue_kv-17_P17_zmq_lookup_full_hashes.md) | B：低优先条件项 | 全量 hashes 确实被发送，但编码/IPC 是否为 TTFT 瓶颈未知；改 wire protocol 的兼容成本较高 | 先测 payload、编解码、IPC 和 TTFT；没有显著收益则不改协议 |
| 10 | [kv-31 backend capability model](issue_kv-31_E1_backend_abstraction_split.md) | C：可延后 | 基类混入 GVA 方法是真实设计问题，但当前固定 backend 路由已避免用户可达错误组合，没有直接生产故障 | 等新增 backend/GVA 扩展需求或相关代码重构时顺带完成 |

## 3. 逐项排序理由

### 1. kv-28：最高价值，先阻断错误计算

同步 non-layerwise multi-group load 部分失败时只记录日志，async 路径还会调用正常 finished。相比可见的异常或 cache miss，使用残缺 KV 继续计算具有更高风险，因为它可能产生无法立即识别的错误输出。

该任务应拆成两步：先用最小 fail-fast 保证任何 group 未确认成功时不进入 forward；再推动 request-level failed IDs，使失败只终止受影响请求。即使目标机器暂时不能稳定复现错误 token，也不能否定代码中缺少阻断信号这一事实。

### 2. kv-27：消除永久阻塞

同步 REQ 在 `recv` 上没有 timeout，而 server handler 没有逐请求异常保护。一个坏报文或 handler 异常即可让 server 不回复，client 随后永久等待。这是明确、易触发且影响请求/engine 可用性的 P0 风险。

它排在 kv-28 之后，是因为 kv-28 可能静默影响模型结果；kv-27 通常表现为显式卡住。两项都应进入第一批，实际开发可以并行。timeout 数值必须按目标机器正常延迟分布确定，但不能以“尚未确定数值”为理由推迟建立有界等待。

### 3. kv-25：公共失败协议，影响面广

`set_device` 在异常捕获和 ready signal 之前，creator 又无界等待；运行期 fatal 也没有统一的 queue accounting、后续任务拒绝和子类资源清理协议。它不仅修复自身 hang/资源状态，还决定 kv-28 async failure 等路径如何可靠通知等待方。

该任务复杂度高于 kv-27，必须先按子类建立故障矩阵，避免重复清理已有 `finally` 覆盖的资源。因此排第三，而不是与两个边界更明确的 P0 任务争抢第一落点。

### 4. kv-26：修复虚假成功并为 batching 建立契约

Mooncake、MemCache、Yuanrong 普通 `put` 当前都不把逐 key 结果交给 sender；sender 无条件发布已构造的 stored events。它影响 KV event 准确性、故障可观测性，并让 kv-07 无法安全回填批量结果。

保存失败一般在未来表现为 cache miss，不像 kv-28 那样直接消费残缺 KV，所以排在前三个正确性/可用性任务之后。开发前必须验证实际 backend 的重复对象、部分失败和无逐 key 返回语义；无法确认的结果应为 `UNKNOWN`，不能伪造成功。

### 5. kv-24：确定性较高的基础治理

配置读取散布在 connector、scheduler、worker、metadata 和 layout，统一 schema 能减少角色间类型、默认值和兼容规则漂移，也给 kv-27 timeout 等新字段提供稳定入口。其价值是广泛而长期的，且不依赖性能猜测。

它不是当前用户可见 P0，也不能自动解决 scheduler/worker 初始化重复，因此放在可靠性任务之后。实现应复用已有 `_parse_int_config` 和 direct/MultiConnector helper；kv-27 可以先使用局部校验默认值，不能等待本任务。

### 6. kv-01：证据较强的性能 PoC

MLA 下同一 TP 组会生成相同后端 key 集，而每个 rank 都调用 `get`，重复调用事实比其余性能候选更确定，并可能随 TP=2/4/8 放大。

它仍然只是第六：leader+broadcast 会引入 collective、stream/event 和全 rank 错误一致性，可能比独立 backend 读取更慢，处理不当还会 hang。先做机器基线；只有 backend 读取成本显著且 broadcast 有净收益时才实现/启用。

### 7. kv-07：覆盖面广，但 workload 机会未知

三个 non-layerwise 路径都存在 per-request backend 调用，若单 step 内经常有多个 request，batching 可能同时减少调用次数和参数重建。但代码不能证明生产 workload 中大多数 step 的 batch size 大于 1，async 微批还可能增加尾延迟。

它排在 kv-01 之后，是因为实际可合并度尚未证明，且依赖健全的 per-item result/cancel/fatal 语义。完成 kv-26 后，优先选择 profile 证明最有价值的一条路径，不要求同步 load、async load、save 一次性全部重写。

### 8. kv-08：窄场景中的潜在收益

GVA 准备入口接收一批 requests，却在 request/group 循环内调用 metadata API，RPC 聚合机会真实。但它只影响 MemCache GVA layerwise 路径，覆盖面比 kv-01/kv-07 窄；`batch_alloc` 非幂等，lease 是否可去重还必须由实际 MemCache 版本确认。

如果 Phase 0 显示 metadata RPC 是 layerwise TTFT 的主要组成，可将其提升到第六或第七；若 RPC 占比低或 lease 契约不明确，应保持当前实现。

### 9. kv-17：事实成立，但改协议性价比最低

client 发送完整 hash 列表是确定事实，suffix 方案也确实需要 absolute offset 才能保持 grouped-hash 语义。不过任务需要 wire protocol 版本化、兼容、坏 offset 校验以及 hybrid/non-hybrid 双路径改造。

在没有数据证明 `.hex()`、msgpack 和 IPC 是 lookup/TTFT 瓶颈前，这种协议复杂度不值得优先承担。只有长上下文、高 HBM 前缀命中场景显示明确端到端收益时才应上调。

### 10. kv-31：合理，但当前最不紧迫

将 GVA/lease 方法从最小 `Backend` 中拆出，有助于未来增加 backend 和做类型窄化。但当前代码用 `backend_name == "memcache"` 选择 GVA 路径，固定的三个 backend 中没有用户可直接配置出的错误组合。

因此它是维护性和未来扩展保护，不应抢占正确性、可用性或已验证性能工作的资源。最合适的时机是新增 backend、显式 GVA mode 或相关模块重构时一起完成。

## 4. 建议实施批次

### 第一批：立即处理

1. `kv-28`：先提交最小 fail-fast，阻止残缺 KV 进入 forward。
2. `kv-27`：建立 timeout、typed response 和 socket 恢复。
3. `kv-25`：补齐 startup/fatal/cleanup 状态机。

第一批可以并行，但 failure signal、event 和等待方语义需要共同评审，避免三个任务定义互相冲突的失败状态。

### 第二批：补齐基础契约

4. `kv-26`：统一 backend save 结果和 event 语义。
5. `kv-24`：统一配置解析，但不阻塞前四项修复。

### 第三批：只先做 Phase 0

6. `kv-01`
7. `kv-07`
8. `kv-08`
9. `kv-17`

四项先并行收集基线数据，再按目标环境实际收益重新排序。Phase 0 后只推进达到各自 Go 条件的任务；排名不是继续投入的承诺。

### 第四批：随扩展需求实施

10. `kv-31`

## 5. 动态调整规则

以下证据出现时，可以调整第 6 至第 9 名的顺序：

- `kv-01`：重复 backend `get` 耗时随 TP 明显放大，且 broadcast 后端到端收益稳定，可上调。
- `kv-07`：生产 step 中 batch size 经常大于 1，backend 固定调用开销显著，可上调；若多数为 1，则 No-Go。
- `kv-08`：metadata RPC 占 GVA 准备或 TTFT 的比例显著，且 lease 契约允许安全聚合，可上调。
- `kv-17`：长上下文下 payload 编解码/IPC 是 lookup 延迟主要组成，并有可测 TTFT 收益，可上调。

没有上述机器数据时，保持当前排序。任何性能任务都不能越过尚未关闭的 `kv-28`、`kv-27` 和 `kv-25` 去争夺同一批核心开发资源。

## 6. 一句话结论

最值得先做的是 `kv-28 → kv-27 → kv-25 → kv-26`：它们处理已经由代码证明的正确性、永久阻塞、线程失控和虚假保存成功。随后做 `kv-24` 的基础治理；4 个性能任务只先做 Phase 0，并按机器数据决定是否继续；`kv-31` 最适合随未来 backend 扩展一起完成。
