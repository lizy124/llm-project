# AscendStore 社区任务严格审核

## 审核基线

- 代码仓库：`D:\lzy\project\kv_pool\code\vllm-ascend`
- 分支：`main`
- 提交：`d5e9816065ede613327d93908f87fee9f5c47128`
- 审核原则：以当前源码、实际调用链、后端接口和现有测试为准；不能由源码证明的性能结论均不作为事实。
- 处置含义："删除"是指从拟发布社区任务池移除，不代表立即删除原始 Markdown 文件。

## 第一组：kv-01 至 kv-07

### 总结

| 任务 | 事实准确性 | 实际价值 | 处置 |
| --- | --- | --- | --- |
| kv-01 MLA 读侧去重 | 部分正确，混淆 layerwise 与 non-layerwise 路径 | 中高，但必须实测 | 大幅修改后发布 |
| kv-02 Key 字符串向量化/下沉 | 主要诉求已实现 | 很低 | 删除 |
| kv-03 五层嵌套 key 循环 | 循环存在，但生产调用前提不成立 | 很低 | 删除 |
| kv-04 TP mismatch 重复 lookup | 核心问题判断错误 | 低且有正确性风险 | 删除 |
| kv-05 GVA 分配循环批量化 | 当前代码已经批量处理 | 很低 | 删除 |
| kv-06 非 GVA layerwise prefetch | 状态机解释错误，且场景不受支持 | 不成立 | 删除 |
| kv-07 non-layerwise 跨 request I/O 合并 | 主要现象真实，部分前提错误 | 中高 | 大幅修改后发布 |

### kv-01：大幅修改后发布

MLA 下 `num_kv_head=1`，普通同步读取中每个 TP rank 都会独立调用后端 `get`，重复读取现象真实。但原描述存在以下错误：

- "写侧只 rank0 写"只符合 layerwise 路径；普通路径实际按 `put_step` 对 block 分片，各 rank 写不同 block。
- 文档使用 layerwise 写侧和 non-layerwise 读侧相互论证，调用路径混淆。
- MLA 会禁用 TP mismatch，因此原验收项“TP mismatch 不破坏 MLA 去重”没有意义。
- 后端读取与 HCCL broadcast 的实际成本没有性能数据，不能预设必然收益。

建议改成“MLA non-layerwise 读取的 TP 内后端访问去重 PoC”。先验证 `leader get + TP broadcast` 的正确性和性能，再决定是否合入；layerwise/GVA 另行评估。

关键源码：`pool_worker.py:195-205, 888-980, 1026-1031`，`kv_transfer.py:788-797`，`metadata.py:36-58`。

### kv-02：删除

当前 non-layerwise 热路径已经缓存不可变 key 前缀，并由 `process_token_key_strings_with_block_ids` 直接生成字符串，不再分配 `PoolKey`。提交 `ee618ae8f`（`#12814`）已经完成直接序列化、前缀缓存和 miss-path 优化。

残留的 `PoolKey.to_string()` 不能证明是主要热点；`LayerBatchBuilder` 的 NumPy 优化针对数值地址数组，不是字符串；C++ 下沉也没有 profile 或扩展边界依据。剩余内容不适合作为独立社区任务。

关键源码：`metadata.py:284-306, 555-600`，`kv_transfer.py:40-90`。

### kv-03：删除

`_generate_store_query_keys` 的五层循环确实存在，但各维度对应必须生成的不同 PCP/DCP/TP/PP key，输出基数无法靠“向量化”消除。更关键的是，该函数只在 `use_layerwise=True` 且非 GVA 时进入；当前正式支持的 layerwise 模式要求 memcache，实际走 GVA 查询并绕过该函数。

若未来支持非 GVA layerwise，应重新建立功能任务，不能把当前未支持路径中的对象构造直接定义为性能问题。

关键源码：`pool_scheduler.py:250-286, 549-566`；文档：`docs/source/user_guide/feature_guide/layerwise_kv_pool.md:81-84, 233-236`。

### kv-04：删除

TP mismatch 写侧只有一次 `lookup`，随后只写 missing keys，不存在描述暗示的重复查询。读侧双层循环是 `chunk x sub-key` 的必要展开，每个 sub-key 对应不同 effective rank 和 strided address，并非重复遍历。

统一 Backend 接口没有“put 可安全覆盖已有 key”的契约；缓存 lookup 结果还会引入并发可见性和过期问题。原任务既没有正确的问题定义，也有数据一致性风险。

关键源码：`pool_worker.py:1919-1976, 2014-2044`，`backend/base.py:50-56`。

### kv-05：删除

当前实现已经批量构造 candidate keys、一次 `batch_is_exist` 刷新缓存，并汇总 missing keys 后一次 `batch_alloc`。剩余循环负责字典查询以及把返回 GVA 映射回原位置，NumPy 不能替代字符串字典操作，后端接口最终也需要 Python key list。

最多存在局部字符串重复构造，不足以发布独立性能任务。

关键源码：`pool_worker.py:1184-1199, 1251-1301`。

### kv-06：删除

`submit_count` 在第 0 层填充预取窗口，之后每推进一层补交一层，这是保持固定深度的滑动窗口，不是“首层之后退化为只预取 1 层”。此外，当前正式支持的 layerwise 后端只有 memcache/GVA；任务讨论的 non-GVA layerwise 属于未支持配置。

可以另行开展 prefetch 深度基准测试，但不能保留为当前源码修改任务。

关键源码：`pool_worker.py:410-428, 1672-1716`，`layerwise_cache_layout.py:17-20, 128-134`；文档：`layerwise_kv_pool.md:81-84, 204-210, 233-236`。

### kv-07：大幅修改后发布

同步读和异步读确实按 request 调用一次 `get`；普通写线程也按 request、group 分别执行 `lookup + put`，所以存在跨 request 批处理机会。

原描述中“跨 request key 全局唯一、无冲突”错误：key 不含 request ID，相同前缀请求会生成重复 key。实现必须处理 key 去重、结果回填、部分失败隔离、逐 request 完成状态、批量上限和微批等待时延。

建议改成“non-layerwise 单次 connector step 内的后端调用批处理”，先处理同一步已经可见的 requests，不直接无限 drain 异步队列。优先级先定 P2，取得并发性能数据后再决定是否升为 P1。

关键源码：`pool_worker.py:888-1017, 1753-1771`，`kv_transfer.py:496-515, 679-715, 741-890, 923-1039`。

## 第二组：kv-08 至 kv-14

### 总结

| 任务 | 事实准确性 | 实际价值 | 处置 |
| --- | --- | --- | --- |
| kv-08 GVA 元数据 RPC 合并 | RPC 粒度问题真实，但 key 唯一性和 lease 语义描述错误 | 中高，需硬件实测 | 大幅修改后发布 |
| kv-09 消除 `batch_copy` 前 `.tolist()` | 转换存在，但性能结论和修改仓库边界不成立 | 未证明 | 删除 |
| kv-10 `block_hash_to_str` 重复转换 | 局部冗余存在，次数描述不准确 | 很低 | 删除，最多并入 kv-08 |
| kv-11 `from_request_tracker` 增量 buffer | 全量转换存在，但调用频率和收益被夸大 | 低且有快照正确性风险 | 删除 |
| kv-12 `_handle_stored_request` 双重建 | 遍历存在，但原方案无法实现 | 很低 | 删除，最多并入 kv-07 |
| kv-13 non-layerwise `wait_for_save` 异步化 | 阻塞存在，但同步是已恢复的正确性屏障 | 有潜在价值但原方案危险 | 删除当前任务；如需继续应另立 RFC |
| kv-14 最后一层 save 等待推迟 | 阻塞存在，但承担发布和 buffer 生命周期语义 | 原方案不成立 | 删除当前任务；如需继续应另立 RFC |

### kv-08：大幅修改后发布

`_alloc_gvas_for_save` 和 `_prepare_load_gvas` 确实接收整批 requests，却在 `request x group` 循环内分别调用 `batch_is_exist`、`batch_alloc`、`batch_get_key_info` 和 `batch_add_lease`。这些是 MemCache 元数据调用，减少调用次数具有实际价值。

原描述有以下关键错误和遗漏：

- full-block key 不含 request ID，相同前缀的不同请求会生成相同 key；“跨 request 无冲突”错误。partial key 才包含 request ID。
- `batch_alloc` 不是固定 N×G 次：只有存在 new keys 时才调用，partial block 还可能产生额外的单 key alloc。
- `batch_alloc` 非幂等，合并前必须去重 full keys，并把同一 GVA 正确映射回多个 request/group。
- lease 不能未经验证直接按 key 去重。当前每个 request/group 分别 acquire，最终释放路径会聚合 key；必须确认 MemCache lease 是幂等状态还是引用计数语义。
- partial lease 有 `MMC_UNMATCHED_STATE` 重试；multi-group 失败还需要释放已经取得的 lease 并整体中止。这些语义不能因批处理丢失。

建议改成“单次 `process_layer_data` 批次内的 GVA 元数据 RPC 聚合”，采用“先构造描述符和回填位置，再批量 RPC，最后按位置回填”的两阶段实现。优先级应先定 P2，并要求给出 metadata RPC 次数、准备阶段耗时和 TTFT 数据，而不是预设 P1。

关键源码：`pool_worker.py:1184-1365, 1367-1569, 1651-1670`，`kv_transfer.py:1633-1640`。

### kv-09：删除

`_batch_copy_with_limits` 确实在每个 transfer packet 上把三个 NumPy 数组调用 `.tolist()` 后传给 `store.batch_copy`，因此存在 O(N) Python 整数对象化。但原任务不能据此成立：

- “完全抵消 NumPy 向量化”没有依据。地址、GVA、size 的批量计算仍然避免了 Python 逐元素算术，最终参数转换不等于抵消全部收益。
- GVA layerwise 当前只支持 MemCache；任务要求兼容 Mooncake 的 `batch_copy` 与实际调用路径不符。
- `batch_copy` 是外部 MemCache Python/C++ binding 的方法，本仓库没有其实现，无法在 vllm-ascend PR 内直接增加 buffer protocol 支持。
- `ctypes`、`memoryview` 或 `array.array` 是否可用完全取决于外部 binding 签名，源码没有依据。

如果硬件 profile 证明这里是显著热点，应在 MemCache 所属仓库建立零拷贝 API 任务，再在 vllm-ascend 做适配；不应作为当前社区任务发布。

关键源码：`kv_transfer.py:435-486`；仓库内 `batch_copy` 搜索结果只有调用点和测试 mock，没有实现。

### kv-10：删除，最多并入 kv-08

save 路径确有重复 `block_hash_to_str`，但并非每个 hash 都固定转换三次：candidate range 转换一次；前导 cached block 在 while 中再转换一次；未缓存范围在 for 中再转换一次，只有 while 停止位置可能达到三次。

提前生成字符串列表可以做局部清理，但 `block_hash_to_str` 对 bytes 只是 `.hex()`，对已有 str 直接返回。没有 profile 能证明它是独立热点，单独发布和要求端到端性能数据都不合理。若实施 kv-08，可在同一重构中顺手消除。

关键源码：`pool_worker.py:1251-1277`，`metadata.py:674-675`。

### kv-11：删除

`ReqMeta.from_request_tracker` 的确会把 block ID 和 GVA list 转成 NumPy 数组，group 0 还同时出现在 `block_ids_np` 与 `block_ids_by_group_np[0]`，存在重复转换。但原描述仍不足以成为社区性能任务：

- `from_request_tracker` 只在需要生成 `ReqMeta` 时运行；多条路径会因无需 save/load 提前返回，并非无条件“每 step 每 request”。
- 2000 个 int64 约 16 KiB；“scheduler 热路径显著”没有 profile 支持。
- `RequestTracker.update_mamba_spec_blocks` 会原地覆盖旧 block ID，不只有追加语义。
- `ReqMeta` 是交给 worker/异步线程的当前 step 快照。复用可变的预分配数组可能让旧 metadata 观察到后续 step 的修改，原任务没有定义 copy-on-write、版本或所有权规则。

可以在普通代码维护中复用 group 0 的一次转换，但不建议发布“增量 buffer”社区任务。

关键源码：`metadata.py:812-853, 993-1104`，`pool_scheduler.py:850-908`。

### kv-12：删除，最多并入 kv-07

当前写路径确实先构造 key 相关列表，lookup 后按 missing indices 过滤，再为 missing keys 构造 addrs/sizes。但“一次遍历同时完成 lookup 过滤”在逻辑上不可行：后端 lookup 前必须先生成完整 key list，返回后才能知道 missing indices。

用 NumPy mask 处理 Python 字符串和嵌套地址列表会引入 object array 和额外转换，不一定比列表索引更快。当前列表过滤只是本地 O(N) 操作，通常低于后端 lookup/put 成本，且没有 profile 数据。若实施 kv-07 的跨 request batching，可在重构时直接按 missing index 生成最终参数，不需要独立任务。

关键源码：`kv_transfer.py:717-890`。

### kv-13：删除当前任务；如需继续应另立 RFC

`wait_for_save` 的 `request_queue.join()` 确实会等待所有普通 save 完成，但这不是未处理的旧同步点。提交 `d6d93c9c3`（`#13120`，`Restore synchronous AscendStore save`）明确恢复了该屏障，此前异步保存与 DeepSeek V4 prefix hit 长度回归、block 生命周期竞态相关。

该屏障当前保证：

- queued save 不会在 scheduler 已释放或复用 KV blocks 后继续读取旧地址；
- 返回前数据对后续 lookup 具有确定可见性，而不只是“最终会持久化”；
- 完成通知不会因跨 step 累积而错误地允许过早释放。

因此原任务“去掉 join，依赖 get_finished/delayed free”已经被实际历史证明不充分。若要继续，只能新建带已知回归复现、跨 worker 完成协议、block 所有权和可见性定义的架构 RFC，不能发布为直接编码任务。

关键源码：`pool_worker.py:1753-1771`，`pool_scheduler.py:1086-1153`；关键提交：`d6d93c9c3`。

### kv-14：删除当前任务；如需继续应另立 RFC

最后一层等待真实存在，但它不只是“可以随意后移的等待”：

- GVA save thread 在最后有效任务上执行 `batch_write_finish`，完成 MemCache blob 发布后才设置 layer save finished event。
- 等待最后一层 event 也利用发送队列的顺序性，确认此前 layer 的 save 已处理。
- scheduler 对 layerwise 请求明确不启用 delayed free；源码注明 layerwise 没有对应 sending event，延迟释放会泄漏，因此请求结束后 block 可以立即释放。
- 把等待移到 worker 的下一 step 开始仍可能太晚，因为 scheduler 可能已经对新请求执行远端 lookup，需要上一批数据已经可见。

要移除该等待，必须先设计跨 scheduler/worker 的 layerwise 完成通知、block 延迟释放和发布可见性协议。原任务没有这些前置设计，不适合社区直接实现。

关键源码：`pool_worker.py:1724-1751`，`kv_transfer.py:1432-1459`，`pool_scheduler.py:1086-1125`；相关生命周期提交：`0a637d9c0`（`#12478`）。

## 第三组：kv-15 至 kv-21

### 总结

| 任务 | 事实准确性 | 实际价值 | 处置 |
| --- | --- | --- | --- |
| kv-15 ZMQ lookup 超时与批合并 | 阻塞和逐请求 RPC 真实，但把正确性、批处理和传输架构混为一项 | 超时价值高；批合并价值未量化且受上游逐请求接口限制 | 删除当前任务；超时并入 kv-27，批合并另立 RFC |
| kv-16 非复用预取层移除 gate | 对 gate 目的的判断错误 | 原方案有资源争用和时序回归风险 | 删除 |
| kv-17 ZMQ lookup 只发后缀 hashes | 完整 payload 真实，但“worker 均会跳过前缀”错误 | 中等潜在价值，需长上下文实测 | 大幅修改后发布，降为 P2 |
| kv-18 lookup key 展开避免字符串替换 | 字符串重建存在，但源码并非普通 `replace`，热点未证实 | 很低 | 删除，最多并入 kv-17/普通维护 |
| kv-19 Mooncake get 降 Python 化 | 转 list 和扫描真实，但忽略返回契约及失败定位用途 | 未证实，且主要受外部 binding 约束 | 删除 |
| kv-20 巨文件拆分 | 大文件仍存在，但文件名、行数和拆分基线已过期 | 有维护价值，当前任务范围过大且高风险 | 删除当前任务；需要时按职责另立小任务 |
| kv-21 scheduler/worker 初始化去重 | 仅部分推导重复；backend 和 layout 的“重复”多数是角色差异或已共享 helper | 中等维护价值 | 大幅修改后发布，或并入配置重构 |

### kv-15：删除当前任务；超时并入 kv-27，批合并另立 RFC

原任务的基础现象大体真实：

- non-layerwise scheduler 在 `get_num_new_matched_tokens` 中按 request 同步调用 `LookupKeyClient.lookup`；client 使用 REQ socket 执行 `send_multipart` 后直接阻塞 `recv`。
- 当前匹配的 vLLM 提交 `58d3918e3e` 中，`make_zmq_socket` 只配置 buffer、HWM、identity、linger 和 bind/connect，没有接收超时。
- `LookupKeyServer` 只有一个 daemon thread，以 REP socket 串行执行 decode、backend lookup 和 reply，循环外没有逐请求异常保护。

但当前任务不能按原描述发布：

- timeout 是失效保护，不是性能优化，且与 kv-27 的 client poll、server 异常保护、socket 重建完全重叠。REQ 超时后不能只补一个 `poll`：REQ socket 仍处于等待 reply 的状态，重试或下一次 send 前通常还需要关闭并重建 socket。
- vLLM connector 接口是逐 request 的同步 `get_num_new_matched_tokens(request, num_computed_tokens)`。scheduler 在遍历 waiting requests 时立即需要返回值，决定该 request 是否可调度；AscendStore 在该回调内部看不到“本 step 的全部新请求”。因此“N 次直接变 1 次”不是当前 `LookupKeyClient` 内部加一个 list 就能完成的局部修改。
- 接口允许返回 `None` 让请求稍后重试，但要利用它做微批，仍需定义收集窗口、何时发送、结果缓存、下一轮唤醒、公平性以及空调度 step 的推进方式。原任务没有这些协议。
- DEALER/ROUTER 会改变请求关联、并发和故障恢复模型，应作为架构备选评估，不能与一个确定性的 REQ/REP 修复共用验收标准。

处置建议：删除 kv-15 当前混合任务。timeout 和 server 失效保护统一归入 kv-27；只有在逐请求 RPC profile 证明占用显著后，再建立“lookup 异步/微批协议 RFC”，先验证不修改 vLLM connector API 能否可靠推进，否则明确需要上游接口扩展。

关键源码：`pool_scheduler.py:525-590, 1156-1188`，`ascend_store_connector.py:293-336`；上游契约：vLLM `kv_connector/v1/base.py:466`、`v1/core/sched/scheduler.py:787`；匹配 vLLM 提交：`58d3918e3e`。

### kv-16：删除

原任务把 `attention_start_gate` 解释成“只保护 reuse_source 对应的复用 buffer”，这与源码定义不符。复用安全已经由另一套同步负责：当 `wait_for_save_layer` 非空时，接收线程等待该层 `layer_save_finished_events`，再同步 `sync_save_events`，之后才复用本地 HBM buffer。

`attention_start_gate` 的独立目的写在 `AttentionComputeStartGate` 文档中：attention worker 在提交 attention op 前记录 compute-stream event，MemCache worker 等待该 event 完成后再提交 H2D/L2G，使传输从真实 attention 边界开始。它约束的是计算与通信的提交时序和资源竞争，不以 layer 是否复用 buffer 为条件。

当前代码只给“有实际 transfer task 且不是 current layer”的预取任务附 gate；空任务会提前完成。原方案按 `reuse_source` 移除 gate，会让独立层的 H2D 在 attention 之前开始，改变有意设计的 compute/transfer 重叠时序，不能由“没有共享 buffer 数据竞争”推出正确或更快。提交 `11e95c105` 的说明也明确记录了该行为用于让 layerwise load timing 对齐 attention boundary。

若未来 profiler 表明单 recv thread 队头阻塞有问题，应研究独立传输队列、优先级或通信限流，而不是用 buffer reuse 条件替代 attention 边界条件。原任务问题定义错误，删除。

关键源码：`pool_worker.py:1672-1716`，`kv_transfer.py:1545-1621`，`attention_fence.py:27-69`；相关提交：`11e95c105`。

### kv-17：大幅修改后发布，优先级降为 P2

client 确实会把全部 `request.block_hashes` 逐个 `.hex()`，再用 msgpack 编码并通过 ZMQ 发送。对于 hash block 很小、prompt 很长且本地 HBM 已命中大段前缀的请求，这会产生可避免的编码和 IPC payload，问题有实际意义。

原描述遗漏了决定正确性的分支差异：

- 只有 hybrid `AscendStoreCoordinator` 路径使用 `hbm_hit_tokens`。它先从完整 hashes 生成 grouped hashes，把 HBM 前缀加入 `exists`，再从对齐后的 `lookup_start` 查询后缀。
- 普通 non-hybrid fallback 当前完全不使用 `hbm_hit_tokens`，仍从 token 0 构造全部 keys，并根据完整 exists 序列计算 prefix hit。原文“worker 侧按 hbm_hit_tokens 跳过前缀”并不普遍成立。
- `_iter_token_chunks` 假设传入 hash 列表从 token 0 开始，以 `chunk_id` 计算 token 位置。直接传 `block_hashes[k:]` 会把后缀误当成前缀，并改变 grouped block 选择、cache-family ratio、lookup mask 和返回 hit 长度。
- grouped hash 取一个大 block 内最后一个 chained fine-grained hash。切片起点必须至少按 `hash_block_size` 和各 group effective block size 定义清楚，不能只用 `hbm_hit_tokens // block_size` 对所有 group 共用一个简单下标。

建议把任务改为“设计 offset-aware lookup wire protocol，避免发送已确认的 HBM 前缀 hashes”：请求显式携带 `hash_start_token`/hash index 与 suffix hashes；worker 的普通和 hybrid 路径都按绝对 token 位置构造 key；已确认 HBM 前缀直接作为命中基线；保持 multi-group、compress/cache-family、coordinator mask、TP/PP key 展开和 partial block 语义。协议还应带版本或保持旧报文兼容。

验收必须增加：non-hybrid 与 hybrid 两条路径，group block size 大于 hash block size，多个 cache family，`hbm_hit_tokens` 为 0/非对齐/全长，以及与完整报文逐项对照。性能优先级先定 P2，提供实际 payload bytes、msgpack 编解码耗时和 16K/64K 端到端数据后再判断是否升到 P1。

关键源码：`pool_scheduler.py:565-575, 1168-1187`，`ascend_store_connector.py:312-325`，`pool_worker.py:2245-2330, 2339-2390`，`metadata.py:468-524, 641-671`。

### kv-18：删除，最多并入 kv-17 或普通维护

`_expand_lookup_keys_by_rank` 确实为每个 `(pp_rank, tp_rank, key)` 重建字符串两次，但任务表述并不精确：当前 `_replace_key_field` 使用 `find` 定位 marker 后做切片拼接，不是调用普通字符串 `replace`。最终仍必须生成 `PP x effective TP x key` 个不同的 backend keys，能省掉的只是每个输出 key 的一次中间字符串。

当前 metadata 已缓存当前 rank 的 `_get_key_prefix`；若要优化 all-rank lookup，可增加按 `(group_id, cache_role, cache_family, pp_rank, head_or_tp_rank)` 生成/缓存 prefix 的 helper，然后直接拼接 chunk hash。但配置在 worker 生命周期内并不动态变化，原文强调的 cache invalidation 也不是现实难点。

这里没有 profile，且同一路径紧接 backend `exists`，不能把两次本地字符串复制直接定为 P1。它也不应再依赖已判定删除的 kv-02。删除独立任务；若 kv-17 重做 lookup 协议和 key 构造，可一并清理并用微基准记录局部收益。

关键源码：`pool_worker.py:2219-2243, 2372-2382`，`metadata.py:281-306`。

### kv-19：删除

Mooncake 路径确实执行 `res_list = list(res)`，随后提取负失败码并遍历一次把正值归零。但这些步骤不是无意义的“Python 化”：

- AscendStore 的调用方需要每个 key 对应的状态，才能由 `record_failed_blocks(block_ids, ret)` 精确标记加载失败的 block。返回一个聚合或紧凑总状态会破坏当前 Backend 契约和错误恢复。
- Mooncake binding 的成功返回值可能为正数，AscendStore 约定 `0` 为成功、非零为失败，因此必须归一化；提交 `ebb4dbac3` 正是为“load failure block ids”增加返回列表和正值归零，不是偶然冗余。
- MemCache binding 已返回 0-success 状态，所以只扫描失败项并原样返回。两者的处理差异来自外部 API 契约不同，不能据此认定 Mooncake 多做了一层可删除工作。
- 本仓库只有 `batch_get_into_multi_buffers` 的调用点，没有 Mooncake C++ binding 实现。让后端“直接返回紧凑结构”首先是外部依赖 API 任务，无法在当前仓库内按原验收完成。

本地可以在普通维护中把失败收集和归一化合并为一次循环，但仍是 O(N)，也仍要产生可索引结果；没有 profile 能证明它具有 P1 或端到端价值。删除该社区任务。如 Mooncake binding profile 证明对象转换占比显著，应在 binding 所属仓库设计 buffer/typed result API，并保持逐 key 失败映射。

关键源码：`backend/mooncake_backend.py:223-256`，`backend/memcache_backend.py:174-198`，`pool_worker.py:940-1009, 1978-2008`，`kv_transfer.py:970-1015`；语义提交：`ebb4dbac3`。

### kv-20：删除当前任务；需要时按职责另立小任务

“巨文件”现象仍存在，当前行数为：`pool_worker.py` 2497、`kv_transfer.py` 1669、`metadata.py` 1255、`pool_scheduler.py` 1209，维护性问题客观存在。但原任务基线已经过期：

- `config_data.py` 已在提交 `11e95c105` 中重命名为 `metadata.py`，backend base、attention fence、connector 等也已重新组织；原文文件名和行数均不对应当前分支。
- 原方案同时拆四个核心模块，涉及初始化、同步/异步 load/save、GVA 生命周期、TP mismatch 和多类线程，无法作为一个“纯结构、零语义”的易审查社区 PR。
- `pool_worker.py` 和 `metadata.py` 在近期仍因 hybrid、MTP、sparse、buffer reuse、coordinator 等功能频繁变化。一次跨模块搬移会制造大量冲突，并提高遗漏隐式状态和线程依赖的风险。
- `metadata.py -> keys/token_database/request_meta` 与 kv-23 重复；配置部分又与 kv-24/kv-21 重叠。当前任务不是清晰的独立交付单元。

建议删除当前大任务。若维护者确认边界，应按一个职责、一个 PR 重建小任务，例如只移动 ZMQ lookup client/server，或只提取不依赖 worker 状态的 key layout derivation；每项先列出依赖方向、内部 API 和针对性测试，不使用“每文件行数”作为主要验收指标。

关键源码规模：`pool_worker.py`、`kv_transfer.py`、`metadata.py`、`pool_scheduler.py`；关键重构提交：`11e95c105`。

### kv-21：大幅修改后发布，或并入配置重构

原任务只在一部分初始化逻辑上成立。scheduler 和 worker 确实分别推导 `use_mla`、`num_kv_head`、`put_step`、`num_layers`，scheduler 还依赖这些值计算 `keys_per_block_hash`；源码甚至有“Keep this in sync with pool_worker.py”注释，存在漂移风险。

但不能按原范围宣称所有初始化都有两份实现：

- TP mismatch 算法已经集中在 `infer_tp_mismatch_info`，两侧只是用各自可见的 role/config 调用它。
- layerwise layout 算法已经集中在 `build_layerwise_cache_layout`/`build_layerwise_reuse_layout`；scheduler 和 worker 位于不同进程且各自需要结果，调用两次不是两份算法实现。
- backend registry `backend_map` 已共享，但 scheduler 创建 scheduler client，worker 创建真实 data backend，并传入 lazy-init 等 worker 参数。对象构造不能合成一个共享实例。
- `KVPoolConfig` 名称和集中读取 `extra_config` 的方案与 kv-24 直接重叠；kv-24 甚至声称会解决 kv-21，两个原任务不应并行发布。

若保留独立任务，建议改名为“抽取 rank-invariant KV key layout 派生配置”，仅建立经过单测的纯 helper/dataclass，统一推导 `use_mla`、effective key head count、`put_step`、rank count、num layer keys 和 `keys_per_block_hash`。runtime rank、backend 对象、thread/event、scheduler request 状态仍由各角色初始化。还应增加同一组 VllmConfig 输入下 scheduler/worker 派生结果一致的参数化测试，覆盖 MLA、GQA `num_kv_head < tp_size`、PCP/DCP/PP、layerwise 和 hybrid。

该任务是维护性重构，不是“严重程度高/P1”；建议 P2 或 P3。若 kv-24 经后续审核保留，则把上述派生值和一致性测试并入 kv-24，删除 kv-21，避免两个 `KVPoolConfig` 任务重叠。

关键源码：`pool_worker.py:97-124, 126-143, 191-245, 306-323`，`pool_scheduler.py:55-81, 156-223`，`metadata.py:36-69`，`layerwise_cache_layout.py`。

## 验证限制

本审核完成了当前源码、调用链、现有测试代码和相关提交历史的静态核查。运行时单测未执行：系统 Python 没有可用的 `pytest`，已有临时依赖目录在当前沙箱中不可读取。本文仅修改审核文档，没有修改 vllm-ascend 代码。
