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

## 第四组：kv-22 至 kv-28

### 总结

| 任务 | 事实准确性 | 实际价值 | 处置 |
| --- | --- | --- | --- |
| kv-22 `to_string` 重复实现统一 | 两类 key 的差异描述基本正确，但只是少量稳定代码 | 很低，抽象后更容易掩盖 key schema 差异 | 删除 |
| kv-23 `ChunkedTokenDatabase` 职责拆分 | 多职责真实，文件名和范围已过期 | 有维护意义，但没有独立问题和可验证收益 | 删除当前任务；需要时渐进提取 |
| kv-24 配置项集中 schema | 配置散落真实，原统计和前置关系不准确 | 中等维护价值 | 大幅修改后发布，降为 P2 |
| kv-25 传输线程异常清理 | fatal 后未清理真实，但三类后果和原修法部分错误 | 高，需建立终止式失败协议 | 大幅修改后发布，调整为 P1 |
| kv-26 backend put 失败传播 | put 失败只记录日志真实，但不是模型正确性 P0，原恢复目标错误 | 中等可靠性和事件准确性价值 | 大幅修改后发布，改为 Reliability P2 |
| kv-27 ZMQ lookup 失效保护 | server/client 无保护、可能永久阻塞均真实 | 高 | 修改后发布，保留 P0 |
| kv-28 多 group 加载失败传播 | 普通 hybrid load 会只记日志；layerwise GVA 已 fail-fast | 高，存在残缺 KV 进入 forward 的风险 | 大幅修改后发布，提升为 P0 |

### kv-22：删除

`PoolKey.to_string` 和 `LayerPoolKey.to_string` 的确有共同字段；后者增加 `layer_id`、省略 `pp_rank`。两者的 `__hash__` 也有相同差异。原文对源码事实的识别基本正确，但这不构成值得发布的社区任务：

- 只有两段约 10 行的 schema 序列化代码，当前输出字段顺序一眼可见；提取带 `include_pp_rank`、`layer_id` 等分支的 builder 并不会降低实质复杂度。
- key 字符串是后端数据兼容边界。把两种有意不同的 schema 隐藏到一个通用 builder，反而更容易在修改其中一种时意外改变另一种。
- `LayerPoolKey` 的 dataclass equality 仍包含完整 `KeyMetadata`，而自定义 hash 省略 `pp_rank`；任务只要求“hash 行为不变”，却没有识别 equality/hash 与字符串 schema 并非同一套字段。贸然统一可能扩大行为面。
- 当前已有 PoolKey、LayerPoolKey 字符串和 hash 单测；不存在 bug、重复不一致或维护事故。任务依赖的 kv-02 也已判定删除。

保留显式实现更清楚。若未来真的引入第三种 key schema，可在具体 PR 中抽取不可变公共 prefix helper，不需要独立社区任务。

关键源码：`metadata.py:73-171`；现有测试：`test_metadata.py:52-108`。

### kv-23：删除当前任务；需要时渐进提取

`ChunkedTokenDatabase` 当前位于 `metadata.py:255-624`，确实同时持有 group buffer layout、key prefix cache、token chunk 迭代和地址计算逻辑。但原任务仍不适合直接发布：

- 原证据指向已不存在的 `config_data.py:255-467`；当前类约 370 行，并已加入 hybrid group、cache family、coordinator mask、block-id offset、sharding和 lazy grouped hash 等新语义，拆分边界比原描述复杂。
- `TokenChunker` 不能只做 token 映射：chunk 大小取决于 group block size、hash block size 和 cache-family ratio，并受 coordinator mask、block ID 裁剪和 shard 条件影响。
- `KeyBuilder` 依赖 group metadata/cache family；`BufferLayout` 又依赖 chunk 的绝对 token 起止位置和 group buffers。按原三个对象拆分会产生大量交叉参数或共享可变状态，未证明比现状更容易维护。
- 当前 `test_metadata.py` 已对 grouped hashes、multi-group、mask、block-id offset、prepare_value 和 PP adaptor 建立直接测试。没有 bug、循环依赖或近期变更证据要求立即做一次大搬移。

近期提交 `e68c19be6` 已采用更稳妥的方式，把无状态的 cache family/block-size 推导提取为 module-level pure functions。后续应继续按这种方式，在真实调用方需要复用时逐个提取纯函数，而不是预先建立三个大对象。删除当前任务；它也不应再作为已删除 kv-20/kv-02 的子任务。

关键源码：`metadata.py:255-624`；测试：`test_metadata.py:111-296`；相关渐进重构：`e68c19be6`。

### kv-24：大幅修改后发布，优先级降为 P2

AscendStore 自有配置缺少统一解析入口的问题真实。当前目录有 24 处 `extra_config.get` 和 2 处 `get_from_extra_config`，分布在 connector、scheduler、worker、metadata 和 layerwise layout，并非原文所说的“3 个文件 15 处”。vLLM 的 `KVTransferConfig` 只把 `kv_connector_extra_config` 定义为 `dict[str, Any]`，`get_from_extra_config` 也只是普通字典读取，没有 AscendStore 字段级校验。

但原任务需要修正以下范围和假设：

- `config_data.py` 已改名为 `metadata.py`；配置入口还包括 `ascend_store_connector.py`、`layerwise_cache_layout.py` 和 lookup path。
- 默认值并不全是静态字段。例如 `discard_partial_chunks` 在普通和 layerwise 路径有上下文差异，layerwise reuse layout 的 prefetch 默认值又取决于 shared buffer 数量。
- 文档和历史配置允许部分整数使用字符串形式，例如 `lookup_rpc_port: "0"`。schema 必须定义兼容的 coercion、范围校验和错误信息，不能仅把当前值放进 dataclass。
- `MultiConnector` 的 AscendStore 配置嵌在 `connectors` 子项中，attention/layout 初始化当前通过 `get_gva_layerwise_config` 专门定位。parser 必须覆盖 direct connector 和 MultiConnector 两种来源。
- 未识别字段需要保留或明确报错策略，因为 backend/后续版本可能消费额外键；“所有 `.get` 全部消失”不是合理验收指标。
- schema 只解决输入解析，不会自动解决 kv-21 的 rank/key layout 派生重复。

建议改成“AscendStore connector-owned extra config 的 typed parser”：集中字段名、兼容默认值、类型转换和约束；保留原始 extra dict 供 backend/未知扩展使用；direct/MultiConnector 共用提取逻辑；scheduler/worker 从同一 immutable parsed config 读取。将 kv-21 中 rank-invariant 派生值作为可选第二阶段，不把两项强行绑定。

这是可维护性和配置错误前移任务，不是 P0。建议 P2，并要求参数化测试覆盖缺省值、字符串整数、非法范围、deprecated `mooncake_rpc_port`、MultiConnector、layerwise 与 non-layerwise 差异，以及现有文档示例兼容性。

关键源码：`pool_worker.py:145-180, 263-281, 428`，`pool_scheduler.py:93-140, 169, 1194-1209`，`ascend_store_connector.py:79-95`，`layerwise_cache_layout.py:70-154`；上游：vLLM `config/kv_transfer.py:23-72, 120-121`。

### kv-25：大幅修改后发布，优先级调整为 P1

基础缺陷真实：`KVTransferThread.run` 从 queue 取出任务后，若 `_handle_request` 抛出未捕获异常，只记录 `_fatal_error` 并退出。当前任务没有 `task_done`，后续队列项也留在 `unfinished_tasks`；基类 `_handle_request_exception` 不执行任何操作，且只有普通 sending thread 实现了 override。

不过原文的调用链和修法不准确：

- 普通 `KVCacheStoreSendingThread._handle_request` 自己用 `try/finally` 调用 `task_done`，并捕获 store 异常，因此普通 `wait_for_save -> queue.join` 并不会按原文描述因常规 backend put 异常永久挂起。
- layerwise key/GVA 线程的异常可以逃到 `run`，但 `wait_for_layer_load` 和最后一层 save wait 每 10 秒调用 `raise_if_failed`，通常会抛错退出，不是永远等待。
- GVA read lease 有 5 分钟 TTL，但异常发生在 final layer 前确实不会主动释放，属于有界资源泄漏。
- `run` 中 `m_store.set_device()` 位于 try 之外；若线程初始化失败，`ready_event` 永远不 set，而 worker 创建线程后无超时等待，存在真正的启动永久挂起。
- async non-layerwise recv thread fatal 后，`KVPoolWorker.get_finished` 不调用 `raise_if_failed`，scheduler 可能一直等待 remote KV 完成。
- 提交 `7201c97a6` 有意删除了旧的“异常后调用 handler 并继续循环”，改为 fatal 后终止线程。简单恢复旧行为会在 transfer state、buffer ownership 已不确定时继续处理后续任务，不安全。

正确任务应定义“终止式失败协议”，而不是笼统 set 所有完成 event：

- 初始化阶段必须无论成功失败都唤醒 creator，并把初始化异常传回；creator 不能无界等待 ready event。
- 当前任务必须 exactly-once `task_done`；线程终止时还要取消/排空未处理任务，清理 per-request counters，并保证任何 queue join 不会等待永远不会处理的项。
- 各子类实现幂等 cleanup：GVA recv 释放本 step 已取得的全部 lease；layer events/get_event 只用于唤醒，等待方在 event 返回后必须再次 `raise_if_failed`，不能把 event set 解释为成功。
- async `get_finished` 路径必须检查 thread fatal 并向 model runner/engine 传播，不能留下 `WAITING_FOR_REMOTE_KVS` 请求。
- fatal 后不继续消费新任务；后续 `add_request` 应拒绝，或由 owner 重建完整 thread/backend 状态。

现有 `test_fatal_error_stops_before_next_queued_task` 明确锁定“fatal 后不处理第二项”，新测试应在保留该原则下验证 queue accounting、startup failure、每类 layer event、lease cleanup 和 async finished polling。问题重要，但原文大幅夸大部分路径且修法可能掩盖失败，建议 P1 而非直接按现有 P0 描述发布。

关键源码：`kv_transfer.py:311-399, 496-525, 670-715, 1042-1652`，`pool_worker.py:456-578, 1701-1771, 2078-2107`；关键历史：`7201c97a6`。

### kv-26：大幅修改后发布，改为 Reliability P2

Mooncake 和 MemCache `put` 确实吞掉 batch 返回失败和异常，只记录日志并返回 `None`；Yuanrong 同样捕获异常后不传播。Backend 抽象也没有 put 返回契约。普通 sending thread 因而无法判断哪些 key 真正写入，仍会完成 request bookkeeping；启用 KV events 时还会在 `put` 返回后发布所有预构造的 `BlockStored` events。

但原任务把缓存保存失败错误地定义成推理正确性 P0：

- `finished_sending` 的首要语义是异步读取 GPU blocks 已结束、scheduler 可以释放 blocks，不等同于“所有远端 key 已持久化”。put 失败后拒绝发送完成通知，会导致 block 无法释放或请求长期挂住。
- `_invalid_block_ids` 是 consumer load 失败后定位当前 scheduler block 的通道。producer save 失败时计算结果已经完成，把 producer block 放入该集合既不能修复远端写入，也可能错误终止当前请求。
- 后续 lookup 会再次检查 key 是否存在；缺失 key 通常退化为 cache miss 和重算。源码不能证明会“命中空数据或 stale data”。content-addressed full-block key 即使已存在，也代表同一前缀内容。
- GVA layerwise save 不走这两个 `put`：它检查 `batch_copy` 和 `batch_write_finish` 返回码并在失败时抛错，不能与普通 put 路径混为一谈。

仍然存在真实的可靠性问题：错误只能靠日志发现，部分成功无法精确统计，KV event 会误报失败 key 已存储。建议把任务改为“统一 Backend.put per-key 结果并保证 save observability”：Mooncake、MemCache、Yuanrong 都返回与 keys 等长的标准状态；sender 对成功 key 生成 events，对失败 key 记录结构化计数/错误码并可触发 backend health/circuit-breaker；无论持久化成功与否，都在本地读取结束后完成 block 生命周期。系统性异常是否要标记 connector unhealthy 应另有明确策略，不能复用 load invalid-block 通道。

这属于 Reliability/Observability P2。验收应覆盖全成功、部分失败、binding 异常、三种 backend、事件只包含成功 key，以及发送完成仍能释放 blocks。

关键源码：`backend/base.py:50-56`，`backend/mooncake_backend.py:189-221`，`backend/memcache_backend.py:210-238`，`backend/yuanrong_backend.py:147-160`，`kv_transfer.py:679-715, 807-892`。

### kv-27：修改后发布，保留 P0

该任务的核心问题成立。Lookup client 使用同步 REQ socket，在 send 后直接 `recv`；匹配的 vLLM `make_zmq_socket` 没有设置 receive timeout。server daemon thread 的循环没有逐请求异常保护，任一 decode、字段访问、worker lookup 或 response conversion 异常都会结束线程，client 可永久阻塞。现有测试只覆盖正常 request framing 和 close，没有 timeout、坏报文或 server failure。

原方案需要补齐 ZMQ 状态机语义：

- REQ 在发送后超时仍处于等待 reply 状态，不能在同一 socket 上直接重试或发送下一请求；超时/协议错误后必须 `close(linger=0)` 并重建 socket。
- REP 在成功 receive 后必须 send 一次才能接收下一条。server 的逐请求异常若发生在 receive 之后，应发送带类型的 error reply；只 catch 后 continue 会触发 EFSM。若 receive/socket 本身失败，则需要重建 server socket或明确终止整个 connector。
- 当前 response 只是任意长度 bytes 经 `int.from_bytes`，无法区分合法 hit=0 与错误。应定义版本化 success/error response，并校验 frame 数、4-byte 数值字段、group/hash 解码类型、hit 范围和 response 长度。
- lookup 是只读操作，可以做有限重试；重试耗尽后返回“remote miss”通常是安全降级，因为上层仍保留本地 HBM hit，只是不采用额外远端前缀。必须记录限频告警/指标，不能静默吞错。
- timeout 配置可由 kv-24 的 parser 管理，但 kv-27 不应因此被阻塞；本任务可先提供带校验的本地默认值。

建议保留 P0，修改为完整的 REQ/REP recovery 任务。验收除 server 异常、client timeout、坏报文外，还应验证“超时后下一次 lookup 可成功”、error reply 后 server 继续服务、socket 重建无 FD/context 泄漏，以及 hit=0 不与协议错误混淆。

关键源码：`pool_scheduler.py:1156-1191`，`ascend_store_connector.py:293-339`；匹配上游 helper：vLLM `58d3918e3e` 的 `utils/network_utils.py:make_zmq_socket`。

### kv-28：大幅修改后发布，优先级提升为 P0

原任务识别到的风险真实，但混合了两条不同路径：

- 普通 non-layerwise 同步 load 在 `pool_worker.py` 中合并多 group keys 后调用 backend `get`；返回部分失败或 `None` 时，single-group 会上报 invalid block，multi-group 只记录日志并继续返回。forward 随后可能使用只加载了部分 group 的 KV。
- async non-layerwise recv thread 有同样分支，记录日志后仍 `set_finished_request`。scheduler 会认为远端加载完成。
- GVA layerwise 路径已经不同：`_prepare_load_gvas` 发现 multi-group GVA/lease 失败时释放此前 group 的 lease 并抛 `RuntimeError`，现有单测验证它在 forward 前停止。因此原文“layerwise 多 group 只记日志”不正确。

“直接把所有失败 block 放入 `_invalid_block_ids`”当前也不可行。匹配 vLLM scheduler 的 `_update_requests_with_invalid_blocks` 仍执行 `(req_block_ids,) = get_block_ids(req_id)`，并带有 hybrid allocator TODO；提交 `ccceb970b` 正是为避免这里对 multi-group 崩溃而加入日志绕过。Ascend platform 也明确禁止 hybrid 配置 `kv_load_failure_policy="recompute"`。所以原任务列出的 per-block recomputation 或“标记整组 invalid”并非当前可用方案。

建议改为“hybrid KV load failure 不得进入 forward，并建立 request-level fail 传播”：

- 近期修复：普通同步 multi-group get 任一 key 失败即 fail-fast；async recv 将 request 标记为失败并通过 worker/model-runner 可观察通道结束等待，不能 set 正常 finished 后继续计算。若暂时没有 request-level connector API，宁可明确终止本次 engine step，也不能使用残缺 KV。
- 完整方案：向 vLLM connector output 增加 failed request IDs 或 grouped invalid-block 描述，使默认 `kv_load_failure_policy=fail` 能只终止受影响请求；只有上游 hybrid block manager 支持后，才开放 recompute。
- 保持现有 GVA layerwise 的 lease rollback 和 pre-forward raise，并补异常路径与 kv-25 的线程终止清理集成。

验收必须分别覆盖 sync non-layerwise、async non-layerwise 和 GVA layerwise；覆盖某个 group 部分失败、backend 返回 `None`、失败 request 不进入 attention/forward、async request 不滞留在 `WAITING_FOR_REMOTE_KVS`，以及其他请求不被误标。由于当前行为可能让残缺 KV 参与模型计算，建议提升到 P0。

关键源码：`pool_worker.py:880-1017, 1450-1540`，`kv_transfer.py:923-1039`，`platform.py:944-950`；匹配上游：vLLM `scheduler.py:_update_requests_with_invalid_blocks`；关键提交：`ccceb970b`、`ebb4dbac3`。

## 第五组：kv-29 至 kv-35

### 总结

| 任务 | 事实准确性 | 实际价值 | 处置 |
| --- | --- | --- | --- |
| kv-29 `_invalid_block_ids` 锁审计 | 确有两处同步 load 写入未加锁，但原文没有证明实际并发竞态 | 低；统一状态访问规则有预防价值 | 修改后发布，降为 P3 |
| kv-30 `_iter_token_chunks` 边界测试 | 文件名和“当前缺测试”均已过期，核心边界已有直接测试 | 很低；剩余组合多为非法输入或现有逻辑的重复验证 | 删除 |
| kv-31 Backend/GVABackend 拆分 | 可选 GVA 方法集中在基类属实，但当前没有靠捕获异常判断能力 | 低到中；可改善类型和初始化期校验 | 修改后发布，降为 P3 |
| kv-32 backend 外部注册 | 三个 backend 硬编码属实，但不存在稳定的第三方 backend 契约或需求证据 | 很低，当前会过早承诺内部插件 API | 删除 |
| kv-33 高风险路径测试补强 | 部分故障测试仍缺，但原文严重低估现有覆盖并与修复任务重复 | 测试有价值，独立的大而泛任务没有价值 | 删除当前任务；测试并入对应缺陷任务 |
| kv-34 Connector 公共基类评估 | “4 个 connector 各自实现同一基类”不正确，公共接口已由 vLLM 提供 | 很低，未发现可复用的共同实现 | 删除 |
| kv-35 传输路径全面改 IPC | 线程、GIL、并行能力和收益结论均未被源码或 profile 证明 | 当前方案风险极高，P0 和预设解法均不成立 | 删除当前任务；有数据后另立 P2 benchmark/RFC |

### kv-29：修改后发布，优先级降为 P3

当前 `_invalid_block_ids` 的大部分并发访问已经正确使用同一把锁：async recv thread 在两类 get 失败分支中加锁更新；GVA load 和 TP-mismatch load 的失败更新也加锁；`get_block_ids_with_load_errors` 则在锁内 copy 后 clear。确实遗漏的是普通 non-layerwise 同步 load 的两个分支：backend 返回部分失败和返回 `None` 时直接调用 `self._invalid_block_ids.update(...)`。

原任务把该遗漏直接描述成 recv thread 与读取方并发导致漏报，但当前调用链不能支持这个结论：

- 普通同步分支只在 `load_async=False` 时执行；`load_async=True` 在提交给 recv thread 后立即 `continue`，不会再执行这两处同步写入。
- 固定版本 vLLM 在 model runner 的 connector 生命周期中先调用 `start_load_kv`，forward 结束后才调用 `get_block_ids_with_load_errors`。同步模式下二者处于同一调用线程，没有已证实的 read/write 竞态。
- async recv thread 才是真正可能与 post-forward 读取并发的 writer，而它已经使用 worker 传入的同一 set 和 lock。
- “重复”不是 set 的并发后果；真正需要保护的是 update 与 copy+clear 的原子边界，避免未来新增并发调用方时丢失刚写入的 ID。

因此应把任务改成一个明确的小修复：“统一 `_invalid_block_ids` 的生产/消费锁协议”。将两处同步写入纳入锁保护，或集中为私有 `_record_invalid_block_ids`/`_take_invalid_block_ids` helper；注明集合和锁必须成对共享，禁止直接替换集合对象。不要声称已经复现中等级并发事故，也不要用不稳定的高频 stress test 作为唯一验收。

验收应使用确定性测试覆盖 partial failure、`None` failure、async writer 与 copy-and-clear 的原子交接，以及读取后只清除已返回快照。该任务规模和当前风险均为 P3。它不解决 kv-28 的 hybrid request-level failure：hybrid 分支目前有意不写 invalid block set。

关键源码：`pool_worker.py:150-151, 918-1003, 1515-1524, 1718-1722, 2000-2008`，`kv_transfer.py:895-920, 997-1021`；固定 vLLM 调用链：`58d3918e3e` 的 `kv_connector_model_runner_mixin.py:_get_kv_connector_output`。

### kv-30：删除

原任务依据的是旧文件 `config_data.py`，当前实现位于 `metadata.py:468-524`。更关键的是，“当前缺测试”的前提已经失效。`test_metadata.py` 已直接覆盖：

- 空 block hashes 返回空结果；
- token_len 小于 hashes 覆盖范围时只生成有效前缀；
- block IDs 只保留滑窗尾部时，通过 `block_id_offset` 把它们映射到最后几个逻辑 chunk；
- hash block size 与 group block size 不同、multi-group/cache-family、mask、filter 后 pre-shard 等组合；
- direct string key 与 `PoolKey` 语义一致。

实现本身先用 `min(len(grouped_hashes), cdiv(token_len, effective_block_size))` 限定逻辑块数；`token_len <= 0` 明确产生 0 个块；访问 `block_ids` 前也检查计算出的 index 上下界。因此原文列举的空 hashes、零/负 token_len、block IDs 较少不会造成越界。block IDs 较多时只消费与有效逻辑 chunk 对应的前部 ID，这与 scheduler 可能预分配更多 blocks 的输入形态一致，不应擅自报错。

提交 `ee618ae8f` 引入统一 chunk iterator 时已经随实现添加和更新了大批 metadata/transfer 测试。现在为负 token_len 或“block IDs 比 hashes 多”再建独立社区任务，只是在测试非法或已由边界公式直接决定的输入，且原文没有给出它们的业务契约。删除该任务；未来修改 iterator 时可在对应 PR 中补具体回归用例。

关键源码：`metadata.py:468-524`；现有测试：`test_metadata.py:122-238`；引入提交：`ee618ae8f`。

### kv-31：修改后发布，优先级降为 P3

源码事实部分成立：`Backend` 除通用 `set_device/register_buffer/exists/put/get` 外，还带有 `batch_get_key_info`、`batch_alloc`、`batch_add_lease`、`batch_remove_lease`、`batch_write_finish` 五个默认抛 `NotImplementedError` 的方法；只有 `MemcacheBackend` 实现这些 GVA/lease 能力。`batch_is_exist` 则是通用 exists 的批量别名，Mooncake、Memcache、Yuanrong 都会使用，不应归入 GVA 专属接口。

但原任务对当前行为和改法描述得过于简单：

- worker 并没有通过“调用后捕获 `NotImplementedError`”判断 GVA 能力。worker 和 scheduler 都由 `use_layerwise and backend_name == "memcache"` 选择 GVA 路径，随后直接调用这些方法。
- GVA 能力同时存在于 scheduler hit lookup、worker 侧 allocation/lease，以及 send/recv thread 的 write-finish/lease release，不只是“worker 侧用一次 `isinstance`”。
- `create_scheduler_client`、lazy initialization 和 scheduler-only backend 实例也是 backend 契约的一部分。只移动五个方法而不验证 scheduler client 的能力，仍会把错误延后到请求处理阶段。
- `isinstance` 不应散落在热路径。正确做法是在 scheduler/worker 初始化时一次性验证所选 mode 所需的 capability，并保存经过窄化的 backend 引用。

可以把任务改为“显式建模 AscendStore backend capability”：保留最小 `Backend`；引入内部 `GVABackend` ABC 或 runtime-checkable protocol；Memcache 实现该能力；GVA mode 在初始化期 fail-fast，普通 mode 不要求这些方法。验收覆盖三个内置 backend 的类型关系、普通模式行为不变、把非 GVA backend 配成 GVA mode 时给出明确配置错误，以及 scheduler client 和 worker backend 都通过能力校验。

这是内部类型边界清理，不是已发生的正确性问题，也没有新 backend 被当前接口阻塞的证据。建议 P3，而不是 P2；它与 kv-26 的 put 结果契约可以独立进行。

关键源码：`backend/base.py:9-56`，`backend/memcache_backend.py:41-238`，`pool_scheduler.py:169-180, 344-390`，`pool_worker.py:145-156, 306-323, 1184-1535`，`kv_transfer.py:1308-1652`。

### kv-32：删除

`backend_map` 的确只包含 Mooncake、Memcache、Yuanrong，worker 和 scheduler 根据 map 中的 module/class 字符串动态 import。但“字典硬编码”本身不等于应该公开第三方注册机制：

- 当前用户文档明确只列出这三个 backend，没有声称 AscendStore backend 是公共插件 API。
- 一个 backend 不只需实现 `put/get/exists`。它必须接受当前构造参数和 lazy-init 行为，提供 scheduler client，处理设备选择和 buffer 注册，并按模式满足 GVA、错误码、线程安全及进程生命周期契约；这些契约目前没有稳定文档。
- scheduler 和每个 worker 分别创建 backend 实例。运行时注册函数若只在某一进程执行，会造成 scheduler/worker registry 不一致；entry point discovery 又引入安装、版本兼容、冲突命名和不可信代码加载策略。
- vLLM 已通过 `KVConnectorFactory` 提供 connector 级扩展边界。真正独立的第三方存储方案可以先作为完整 connector 集成，不必把 AscendStore 的内部 backend 层立即公共化。

没有外部 backend 请求、候选实现或重复改 map 的维护证据。为未来假设需求发布 entry-point 任务会过早冻结内部 API。删除；出现第四个实际 backend 时，应先随该实现补充显式 backend spec/capability 和兼容测试，再依据真实重复决定是否建立注册机制。

关键源码：`backend/__init__.py:17-30`，`pool_worker.py:306-323`，`pool_scheduler.py:169-180`；connector 注册边界：`vllm_ascend/distributed/kv_transfer/__init__.py:21-89`；用户文档：`docs/source/user_guide/feature_guide/kv_pool.md:32-38`。

### kv-33：删除当前任务；测试并入对应缺陷任务

“高风险路径需要测试”方向没有错，但当前 issue 过于宽泛且明显落后于代码。`tests/ut/distributed/ascend_store` 当前有 8 个测试模块和约 383 个 test functions，不是只具备少量 happy-path 覆盖：

- `KVTransferThread` 已有 fatal 后停止处理下一任务和 `raise_if_failed` 测试；普通 send/recv 的 queue `task_done` 也有异常测试。kv-25 真正缺少的是 startup failure、fatal queue cancellation、等待方二次检查和 lease cleanup，这些必须与失败协议实现一起测试。
- GVA 已覆盖 `batch_alloc`、`batch_get_key_info`、lease failure/retry/release、`batch_write_finish` 失败、partial block、eviction 以及 multi-group failure 在 forward 前停止。原文笼统的“GVA 元数据批量路径待补强”没有指出仍未覆盖的行为。
- ZMQ client 目前只覆盖正常 framing 和 close，timeout/recovery 的确缺失；但它正是 kv-27 修复的必要验收，不能脱离协议和 socket 状态机实现先写一套空泛测试。
- layerwise/non-layerwise、同步等待和 async finished 已有多组测试。普通 hybrid partial failure 的缺口属于 kv-28；单独测试不能修复当前错误行为。

测试不是这些正确性任务的可选前置，而是每个修复 PR 的验收组成部分。把四类不同状态机合并成一个 P1“覆盖率提升”任务，没有基线阈值、故障语义或完成边界，还会让测试与随后修改的实现重复返工。删除 kv-33，将 startup/fatal、ZMQ recovery、ordinary hybrid fail-fast 等测试分别写入 kv-25、kv-27、kv-28 的验收标准；纯重构只需在自己的 PR 中保持并按行为风险补测试。

关键测试：`test_kv_transfer.py:136-294, 570-587, 900-934`，`test_pool_worker.py:679-701, 1123-1205, 1468-1970`，`test_pool_scheduler.py:569-608, 1245-1321`，`test_ascend_store_connector.py:248-478`。

### kv-34：删除

原任务的基础统计不准确。当前 `AscendStoreConnector`、`RecomputeCPUOffloadConnectorV1` 和 `UCMConnectorV1` 直接继承 `KVConnectorBase_V1`；`AscendSimpleCPUOffloadConnector` 则继承上游 `SimpleCPUOffloadConnector`，只替换为 NPU worker。所有 connector 已经共享 vLLM 的公共接口基类，并共同使用 `SupportsHMA` 等能力接口；“没有共同父类”不成立。

也没有值得下沉到 kv_pool 公共基类的共同实现：

- AscendStore 是 scheduler/worker 分离的远端 KV pool，拥有 lookup RPC、backend、KV events 和 layerwise 状态机。
- Recompute CPU offload 管理抢占、GPU block pool 和 host capacity。
- Simple CPU offload 是上游 connector 的薄 NPU 适配层。
- UCM 是外部引擎的委托/兼容壳，并包含保留 hook 的动态转发。

配置解析和 event 聚合在这些 connector 中既没有相同输入 schema，也没有相同生命周期。原 issue 允许“最终不抽取，只交一份评估报告”，缺少可验证的软件交付目标，不适合作为社区开发任务。各 connector 已有独立 feature/design 文档；未来只有在出现真实重复实现时，才应在具体 PR 内提取窄 helper，而不是预设一个 Ascend kv_pool connector 基类。

关键源码：`ascend_store/ascend_store_connector.py:42-288`，`recompute_cpu_offload/recompute_cpu_offload_connector.py:43-243`，`simple_cpu_offload/simple_cpu_offload_connector.py:24-54`，`ucm_connector/connector.py:37-307`，`vllm_ascend/distributed/kv_transfer/__init__.py:21-89`。

### kv-35：删除当前任务；有性能证据后另立 P2 benchmark/RFC

当前 P0 任务不能发布。它把“存在 Python 代码”直接推导为“GIL 已阻止 I/O 与 forward 重叠”，没有 profiler、吞吐/延迟对比、线程时间线或底层扩展的 GIL 证据；同时多项源码事实错误或已过期：

- 并非 save/load 全部走 daemon thread。非 layerwise 且 `load_async=False` 时，`start_load_kv` 在 model-runner 线程中构造 key/address 并同步调用 backend `get`。
- “每进程一对线程”不等于部署只有一对线程或不能多 rank 并行。每个 worker/rank 都有自己的 `KVPoolWorker` 和 transfer thread；任务没有证明单 rank queue 是吞吐瓶颈，也没有证明 backend 支持同一 rank 多路并发。
- 非 layerwise 热路径已由 `ee618ae8f` 改为缓存 immutable key prefix 并直接生成字符串，不再为每块构造 `PoolKey`。Python iterator 和 address loop 仍存在，但源码无法给出它们占 forward 时间的比例。
- 任务自己承认 Mooncake/Memcache/Yuanrong 扩展是否释放 GIL“无证据”。在确认底层调用前，不能把整个架构改造建立在 GIL 已锁住 I/O 的假设上。
- 它引用的 `kv_pool_线程存取与GIL分析.md` 实际结论是先 profile、确认扩展 GIL、减少 Python 循环；该文档明确不支持直接改 IPC。

当前仓库已经存在 NPU IPC 的可行性先例，原任务对此也已过期：weight-transfer 使用 `reduce_tensor` 生成 NPU IPC handle，并由 `torch_npu.multiprocessing.reductions.rebuild_npu_tensor` 在同一物理 NPU 的另一进程重建 tensor。这个证据只说明“共享 NPU storage 值得实验”，不能直接证明 KV transfer 子进程方案成立：

- weight-transfer 为保证 storage 生命周期保留强引用，并在发送后执行 barrier/`torch.npu.synchronize()`；动态 KV buffer 必须另外定义长期 handle ownership、block 重用同步和子进程退出后的 allocator 引用清理。
- layerwise 当前依赖父进程创建的 `torch.npu.Event`、attention gate、cache buffer reuse 和每层完成事件。现有源码没有跨进程 NPU event 交接实现。
- Memcache 明确要求 `batch_get_key_info`/`batch_add_lease` 与 `batch_copy` 共享进程内 `gvaBlobTracker`。若把 I/O 放到子进程，GVA metadata 准备也必须整体迁移，不能只把现有 thread queue 替换成 IPC queue。
- backend 需要在子进程设置 device、初始化并注册重建后的 buffer；Mooncake lazy init、Memcache tracker、Yuanrong client、scheduler lookup 与子进程重启后的状态恢复都必须逐后端验证。
- keys/hashes/request metadata 和完成状态仍需跨 IPC。pickle/msgpack 自身也在 Python 中编码，不能在没有原型数据时宣称“彻底脱离 GIL”或收益至少 10%。
- 子进程崩溃不会自动带来安全降级。进行中的 NPU 操作、lease、block release、async request 状态和父进程持有的旧 IPC handle 都需要新的失败协议；kv-25 的线程内清理不能直接复用为跨进程恢复。

更根本的问题是优先级和任务形态：P0 应对应已证实的严重正确性或可用性问题，而这里是一个未建立基线、预先锁定解法、横跨三 backend/三 transfer mode/vLLM 生命周期的大型研究项目。阶段 2 又允许因“解锁新能力”绕过性能阈值进入全量实现，使验收无法客观阻止无收益架构落地。它也不适合交给普通社区贡献者一次完成。

删除当前任务。只有线上 profile 或可复现实验确认 Python/GIL 占据显著关键路径后，才另立 P2 性能调查/RFC：先分别测 non-layerwise sync/async、key layerwise、GVA layerwise 和三个 backend；从对应 C/C++ binding 或采样时间线确认 GIL 行为；比较下沉 key/address 构建、批量 API 与 IPC。若 IPC 仍有优势，再单独做“non-layerwise + 单 backend”的 NPU IPC handle 原型，明确 spawn/device、tensor lifetime、stream synchronization、backend ownership、故障传播和回退条件。原型数据达成事先定义的端到端指标后，才拆分后续实现任务，不能现在承诺全路径 P0 落地。

关键源码：`pool_worker.py:456-578, 864-1003, 1367-1376`，`kv_transfer.py:607-715, 717-890, 895-1039, 1042-1652`，`metadata.py:275-313, 468-624`，`attention_fence.py:27-81`；NPU IPC 先例：`distributed/weight_transfer/npu_ipc_engine.py:112-220, 337-398`、`packed_tensor.py:201-323`；关键优化：`ee618ae8f`。

## 验证限制

本审核完成了当前源码、固定版本 vLLM 调用链、现有测试代码和相关提交历史的静态核查。运行时单测未执行：系统 Python 没有可用的 `pytest`，已有临时依赖目录在当前沙箱中不可读取。kv-35 没有提供硬件 profile，且 Mooncake/Memcache/Yuanrong binding 源码不在本仓库，因此不能据此证明底层调用是否释放 GIL 或量化 IPC 收益。本文仅修改审核文档，没有修改 vllm-ascend 代码。
