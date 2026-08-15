# vLLM-Ascend KV Pool 社区任务严格审核与发布稿（v3，10 项）

> 目标仓库：`vllm-project/vllm-ascend`
> 代码范围：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 审核基线：`main`，提交 `d5e9816065ede613327d93908f87fee9f5c47128`（2026-08-15）
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 验收人：@赵鹏博

## 审核原则

本版逐项核对当前代码、公开文档和现有单测，不沿用仅来自旧草稿的因果结论。

- **直接发布**：问题或冗余已由当前代码确认，任务边界和验收结果明确。
- **条件发布**：只能发布为 PoC、profiling 或 investigation，不能承诺一定有端到端收益。
- **RFC 发布**：先交付证据和设计评审，不授权直接做全路径重构。
- **替换后发布**：`v2` 的问题定义不成立或已过期，本版用已确认问题替换。

性能任务必须同时给出改动前后数据和无收益时的结论。不得用调用次数下降代替端到端收益，也不得为了满足标题而保留无收益改动。

## 审核总表

| # | v2 结论 | v3 处置 | 优先级 | 是否可发布 |
|---|---|---|---|---|
| 1 | 读侧存在重复 I/O，但“非-layerwise 写侧仅 rank 0 写”已过期 | 改为 put-step 共享 KV 读侧去重 PoC | P2 | 条件发布 |
| 2 | 有线程和 Python 工作，不等于已证明 GIL 瓶颈；直接全量 IPC 过度预设方案 | 改为 GIL/IPC 可行性 RFC | P2 | RFC 发布 |
| 3 | 非-GVA 路径存在，但不在公开支持范围；`submit_count=1` 不是窗口失效 | 改为 layerwise 支持合同与默认值对齐 | P1 | 替换后发布 |
| 4 | 完整 hash 列表仍通过 ZMQ 发送，事实成立 | 收紧 offset、编码和回归验收 | P2 | 直接发布 |
| 5 | 嵌套循环和对象构造存在，但热点程度未证明 | 保留为基准驱动的窄优化 | P3 | 条件发布 |
| 6 | `put()` 内再次查询的断言没有代码证据，原任务因果不成立 | 替换为 ZMQ lookup 失效保护 | P0 | 替换后发布 |
| 7 | 同一 block hash 在 GVA save 路径重复转换，事实成立 | 降为微优化，按调用次数和微基准验收 | P3 | 直接发布 |
| 8 | gate wait 存在，但与 buffer 复用同步被错误归为同一机制 | 改为 attention gate 调度调查 | P3 | 条件发布 |
| 9 | getter 已加锁；同步 load 的两处 writer 未遵循同一锁协议 | 改成明确访问点修复 | P1 | 直接发布 |
| 10 | 多个所谓缺口已有测试，原清单过期 | 聚焦 GVA 分配失败契约和剩余缺口 | P1 | 替换后发布 |

## 通用约定

- 代码基线以认领任务时最新 `main` 为准；若实现已变化，先在 issue 中回报并重新确认范围。
- 现有单测必须全绿；功能改动需覆盖 greedy/non-greedy，涉及硬件路径时注明 NPU 型号、卡数及 TP/CP/PP 配置。
- 性能数据必须包含工作负载、配置、预热、样本数、统计口径和原始数据；至少报告延迟与吞吐，不能只报告局部函数耗时。
- 新增或更新 `tests/ut/distributed/ascend_store/` 下的针对性测试。
- 交付件为 PR、设计/测量说明、风险与回退方式。调查任务允许以“无可复现收益，不建议改代码”作为有效结论。

---

## 任务 1：[Perf/PoC] put-step 共享 KV 的读侧去重可行性验证

> 审核状态：条件发布，不应写成已确认的性能修复。

### 已确认事实

当 `num_kv_head < tp_size` 时，多个 TP rank 通过 `put_step` 共享同一 `head_or_tp_rank` key；MLA 是该场景的典型例子。同步和异步非-layerwise load 当前仍由各 rank 把同一 key 对应的数据读入各自本地 buffer。

但 `v2` 所称“写侧只有 rank 0 写”对当前非-layerwise 路径不再成立：`KVCacheStoreSendingThread` 在适用条件下使用 `shard_rank=tp_rank % put_step` 对 block 预分片。layerwise save 才有同组非首 rank 直接跳过的逻辑。

代码证据：

- `pool_worker.py:201-211`：`put_step` 和共享 key rank 的计算。
- `kv_transfer.py:788-797`：非-layerwise save 的 block 预分片。
- `pool_worker.py:950-980`、`kv_transfer.py:970-997`：同步/异步 load 各 rank 调用 backend get。
- `pool_worker.py:1026-1030`：layerwise save 的首 rank 规则。

### 任务范围

先做 PoC，比较以下方案，不预设 rank 0 broadcast 一定最优：

1. 当前每 rank 独立 get。
2. 同一 put-step group 由一个 rank get，再 broadcast。
3. 各 rank 分片 get，再 all-gather。

首阶段只选择一个公开支持、时序清晰的路径；异步 load、GVA layerwise 和 TP mismatch 需分别说明是否纳入，不能一次性混改。

### 验收标准

- 每个 rank 最终 KV 内容逐字节一致，缺失 block、失败状态和 fallback 在 group 内一致传播。
- 不破坏非 MLA、`put_step=1`、TP mismatch、hybrid group 和 partial block 行为。
- TP=2/4/8 下报告 backend get 次数、传输字节、collective 时间、TTFT、吞吐及高并发结果。
- 若 collective 使延迟或吞吐无改善，提交测量报告并关闭，不要求合入复杂实现。

---

## 任务 2：[Arch/RFC] 传输执行隔离与 GIL/IPC 可行性验证

> 审核状态：RFC 发布；不得以本任务直接要求三条路径全部迁移到子进程。

### 已确认事实

`ascend_store` 使用 `KVTransferThread`、`queue.Queue` 和 `threading.Event` 驱动异步 save/load，部分 key、地址和状态处理在 Python 中执行。非-layerwise 且 `load_async=false` 时仍有主线程同步 get 的例外。

当前没有 profiler 证明 GIL 是端到端瓶颈；backend 的 C/C++ 扩展调用是否释放 GIL 也未在本仓库得到证明。独立进程还涉及 NPU device pointer 注册、NPU event/stream、backend 实例、故障传播及 IPC 序列化，不能从“存在 Python 线程”直接推出“IPC 必然更快”。

### 任务范围

阶段 1 只交付证据和 RFC：

- 用 profiler 区分 Python/GIL、backend I/O、NPU 等待和队列等待占比。
- 核实所用 backend 扩展是否释放 GIL。
- 选择一条公开支持路径做最小原型，验证 device memory 注册和完成事件的跨进程语义。
- 对比线程、子进程、减少 Python 热路径三类方案的收益和复杂度。

只有评审确认瓶颈和原型收益后，才另开实现任务扩展范围。

### 验收标准

- 提供可复现的 GIL/CPU contention 证据，而不是只展示线程存在。
- RFC 明确 buffer 所有权、进程生命周期、异常传播、超时、资源回收和回退路径。
- 原型报告包含 IPC 序列化成本、TTFT、吞吐和 CPU 使用率。
- 不把当前未公开支持的非-GVA key-layerwise 作为第一阶段必选范围。

---

## 任务 3：[Docs/Correctness] 对齐 layerwise 支持范围与 prefetch 默认值

> 审核状态：替换后可直接发布；`v2` 的非-GVA性能修复任务不发布。

### 已确认问题

公开文档明确 layerwise 只支持 Memcache：`docs/source/user_guide/feature_guide/layerwise_kv_pool.md:40-42,81-84,235-236`。worker 又将 `use_layerwise + backend=memcache` 判定为 GVA，因此当前公开支持的 layerwise 都是 GVA。

非-GVA key-layerwise 代码虽然可达，但 Mooncake/YuanRong layerwise 超出公开支持合同。与此同时，文档把 `layerwise_prefetch_layers` 默认值写为 1，而 GVA 实现未配置时使用 `min(num_shared_buffers, 8)`。这是可确认的合同不一致。

此外，`pool_worker.py:1693-1699` 的“初始提交 N、之后每层补 1”是正常滑动窗口维护，不能再写成 bug。

### 任务范围

- 明确非-GVA key-layerwise 的状态：正式支持、实验性保留或配置拒绝。
- 使配置校验、公开文档和测试与该决定一致。
- 对齐 `layerwise_prefetch_layers` 的文档默认值与 GVA 实际计算规则；如选择改代码而不是改文档，必须说明兼容性和性能依据。
- 补充 N=1/2/4/8、空 task 层和窗口推进测试。

### 验收标准

- 文档中不存在“只支持 Memcache”与实际允许其他 backend 的无说明冲突。
- 用户能够从错误信息或文档明确知道非-GVA layerwise 是否受支持。
- 默认值描述与运行时值一致，并有单测锁定。
- 不以提高非-GVA 默认窗口作为本任务的预设交付结果。

---

## 任务 4：[Perf] ZMQ lookup 仅发送待查询 hash 后缀

> 审核状态：直接发布，但端到端收益仍需测量。

### 已确认问题

非-layerwise scheduler 调用 `LookupKeyClient.lookup()` 时仍传入完整 `request.block_hashes`（`pool_scheduler.py:568-575`）。client 对所有 hash 执行 `.hex()` 并编码后发送（`pool_scheduler.py:1168-1187`），server 再把完整列表传给 `lookup_scheduler()`（`ascend_store_connector.py:314-324`）。coordinator 分支会消费 `hbm_hit_tokens` 来确定查询起点；普通分支仍从完整 hash 列表构造 key。无论走哪一分支，ZMQ payload 都包含已经由 HBM 覆盖的前缀。

### 任务范围

调整 scheduler、client 和 server 协议，只编码并发送实际待查询的 hash 后缀。实现方式可使用起始 block offset 或等价协议，但必须保留完整 prompt 中的绝对 token 位置语义。

### 验收标准

- `hbm_hit_tokens=0`、完整 HBM 命中、非 block 对齐、partial chunk、Eagle 和多 KV group 的返回 token 数与改动前一致。
- 测试断言实际发送的 hash 数量和 frame 内容，不只断言最终命中数。
- 报告 16K/64K prompt 的序列化字节数、encode/decode 时间和 scheduler lookup 延迟。
- payload 下降是必要条件；若端到端无可测收益，应如实报告，不夸大优先级。

---

## 任务 5：[Perf] scheduler lookup key 生成热路径优化

> 审核状态：条件发布，先基准后改动。

### 已确认事实

`pool_scheduler.py:250-283` 的 `_generate_store_query_keys()` 按 block hash、PCP、DCP、head/TP 和 PP 组合生成 key，并为每个组合构造 `KeyMetadata`、`PoolKey` 后调用 `to_string()`。对象和循环开销存在，但 PCP/DCP 常为 1，当前没有数据证明它是显著端到端热点。

### 任务范围

- 先建立独立微基准和 scheduler profile，覆盖典型及放大的 TP/PP/PCP/DCP 组合。
- 在输出协议不变的前提下减少对象构造、重复常量格式化或 Python 循环开销。
- 不预设 NumPy 字符串向量化；实现应以可读性、实际收益和分配量为依据。

### 验收标准

- 新旧输出必须按顺序逐项完全相等，不能只比较集合；key 顺序对应 backend 返回位置。
- 覆盖 include_layers、多 group、cache family、字符串/bytes hash 和不同并行组合。
- 报告单次耗时、对象/内存分配、scheduler lookup 总耗时及端到端影响。
- 无稳定收益时不合入增加复杂度的实现。

---

## 任务 6：[Correctness] ZMQ lookup RPC 超时、异常与 socket 恢复

> 审核状态：替换后直接发布；替代 `v2` 中不成立的 TP mismatch 重复查询任务。

### 原任务为何删除

`_store_kv_tp_mismatch()` 确实先 lookup 再写 missing keys（`pool_worker.py:2014-2044`），但 Memcache、Mooncake 和 YuanRong 的 Python `put()` 都直接调用底层写接口，没有第二次 `exists/lookup`。不能把未知的 backend 内部行为写成已确认的“双重查询”。`_build_tp_mismatch_keys_and_addrs()` 的 block × sub-key 循环也在生成不同 key/address 布局，不是可直接合并的重复遍历。

### 已确认问题

`LookupKeyClient` 使用同步 REQ socket，`socket.recv()` 没有 timeout（`pool_scheduler.py:1186`）。`LookupKeyServer` 的 `recv_multipart -> decode -> lookup -> send` 循环没有异常保护（`ascend_store_connector.py:312-333`）。server 退出、报文损坏或 lookup 异常都可能让 scheduler 永久阻塞；REQ socket 超时后还需要重建才能恢复正确状态机。

### 任务范围

- 定义可配置的请求超时和降级语义，优先保证 scheduler 不永久阻塞。
- server 对可恢复的解码/lookup 异常返回结构化错误，不因单个请求静默退出线程。
- client 在 timeout、断连或 REQ 状态损坏后关闭并重建 socket。
- 明确“按未命中继续重算”和“抛出可恢复错误”的选择及可观测性。

### 验收标准

- server 无响应时，client 在配置上限内返回或抛出约定异常，不永久等待。
- 覆盖 server lookup 异常、畸形 frame、recv timeout、socket 重建后下一次成功请求。
- 错误路径有带限日志/指标，不泄露完整 hash payload。
- 正常路径命中语义和任务 4 的裁剪协议均无回归。

---

## 任务 7：[Perf] GVA save 路径复用 block hash 字符串

> 审核状态：直接发布；定位为 P3 微优化。

### 已确认问题

`_alloc_gvas_for_save()` 在 candidate key、跳过已分配前缀和分配循环中，可能对同一 `group_block_hashes[block_idx]` 重复调用 `block_hash_to_str()`（`pool_worker.py:1251-1277`）。load 准备路径已采用一次转换后复用的形态（`pool_worker.py:1428-1430`）。

### 任务范围

在单次 request/group save 准备过程中缓存或预先生成 hash 字符串，使每个参与处理的 hash 最多转换一次，同时保持 key 顺序和分配语义不变。

### 验收标准

- 用 mock/spy 断言每个 hash 的转换次数及最终 ordered keys。
- 覆盖已分配前缀、部分命中、新分配、partial block 和 multi-group。
- 提供大 block 数量下的函数微基准；不要求承诺可测的端到端收益。
- 不引入跨请求无界缓存或延长大对象生命周期。

---

## 任务 8：[Perf/Investigation] layerwise attention gate 调度粒度与队列等待验证

> 审核状态：条件发布；原“非复用层过度同步”bug 不成立。

### 已确认事实

非当前层且有实际 load task 时会附加 `attention_start_gate`；GVA 和 key-layerwise recv 线程都会在传输前等待该 gate。单 recv 线程因此可能停在队头。

但 `prefetch_layer_map -> wait_for_save_layer` 保护共享 buffer owner 复用；`attention_start_gate` 则等待 compute stream 到达 attention 边界。二者是独立同步机制。`prefetch_layer_map` 中没有某层，也不能证明该层拥有独立 buffer，因为共享 slot 的首个 owner 同样不在 map key 中。

当前还没有证据表明 gate 后存在已满足执行条件、可以安全越过队头的任务。同一批预取通常共享同一个 attention gate，因此“去 gate 必然减少 head-of-line blocking”没有成立。

### 任务范围

- 增加低开销时间线或 profiler 标记，区分 attention gate、save-owner wait、backend transfer 和 queue wait。
- 构造多窗口、多请求、buffer reuse/independent layer 场景，确认是否存在可运行任务被较早 gate 阻塞。
- 只有证明存在安全且有收益的调度机会后，才提出 per-gate queue、就绪队列或其他方案。

### 验收标准

- 报告 runnable task 被阻塞的具体时间线，不能只报告 `gate.wait()` 有耗时。
- 任何实现都必须保留共享 buffer owner、`sync_save_events` 和 attention boundary 的正确性。
- 不允许用 `layer_id not in prefetch_layer_map` 作为移除 attention gate 的充分条件。
- 给出 TTFT/吞吐、队列等待和 NPU 时间线；无证据时以调查报告结束。

---

## 任务 9：[Correctness] `_invalid_block_ids` 统一锁协议

> 审核状态：修正表述后直接发布。

### 已确认问题

`get_block_ids_with_load_errors()` 已经在同一把锁内完成 copy 和 clear（`pool_worker.py:1718-1722`），所以 `v2` 对 getter 的怀疑不成立。异步 recv、TP mismatch 和 GVA 失败写入也使用该锁。

但是非-layerwise 同步 load 的两个失败分支在 `pool_worker.py:987` 和 `pool_worker.py:1002` 直接调用 `_invalid_block_ids.update()`，没有持有 `_invalid_block_ids_lock`。该集合同时被传给后台 recv 线程，混用加锁和无锁访问使现有锁不能建立完整互斥协议。

### 任务范围

- 枚举 ascend_store 内该集合的所有读写点，统一通过锁或封装方法访问。
- 修复同步 load 的无锁 writer，避免后续新增访问点再次绕过协议。
- 不把其他 connector 中同名但独立拥有的集合机械纳入本任务。

### 验收标准

- ascend_store 共享集合的所有读、update、copy 和 clear 均遵循同一锁协议。
- 并发测试覆盖同步失败写入、recv 线程失败写入和 getter copy+clear；每个已提交错误恰好在某次快照中出现，不丢失、不重复。
- 测试使用事件/barrier 控制交错，不依赖低概率压力循环。
- 热路径锁区只包含集合操作，不把 backend I/O 放进锁内。

---

## 任务 10：[Correctness/Test] GVA 分配失败契约与回归覆盖

> 审核状态：替换后发布；不再使用泛化的“高风险路径测试覆盖补强”。

### 原任务为何改写

当前测试已经覆盖多项 `v2` 所列缺口：

- `test_fatal_error_stops_before_next_queued_task` 覆盖传输线程 fatal error 传播。
- `test_write_finish_failure_does_not_complete_layer` 覆盖 GVA write-finish 失败。
- `test_empty_reuse_gate_waits_for_non_saving_rank_compute` 覆盖空复用任务同步。
- `test_multi_group_load_failure_stops_before_forward` 覆盖 multi-group load 失败和 lease 清理。

旧文档提到的 `test_config_data.py` 也已不存在。原任务按整类路径声称“缺测试”会让贡献者重复已有工作。

### 当前可确认缺口

`_alloc_gvas_for_save()` 会处理 `batch_alloc()` 的非正 GVA，但当前没有针对混合成功/失败、返回列表长度异常和 partial GVA 分配失败的完整契约测试。代码使用 `zip(new_positions, new_keys, new_gvas)`；若 backend 返回数量不足，多出的 key 会被静默留为 0，必须明确这是允许跳过、应立即失败，还是应回滚已成功分配。

### 任务范围

- 定义 `batch_alloc(keys, sizes)` 的返回长度、逐项状态和部分失败契约。
- 为全失败、部分失败、短返回、异常抛出和 partial key 分配失败补充测试。
- 若现状与契约不符，允许做最小生产代码修复；不扩展成无关模块的覆盖率工程。

### 验收标准

- 失败或缺失的 GVA 绝不进入 batch copy、save keys 或“已分配”缓存。
- 部分成功时明确保留还是回滚，并测试 request/group 间状态不会串扰。
- 异常不会错误标记 layer/request 完成，也不会留下不可释放 lease 或永久缓存项。
- 提交前列出现有相关测试，证明新增场景不是重复 happy path。

## 最终发布建议

可直接作为实现/修复任务发布：3、4、6、7、9、10。

只能按证据驱动的 PoC、RFC 或 investigation 发布：1、2、5、8。它们的标题和验收中必须保留“无收益/不实施也是有效结论”，不能改回确定性 bug 或收益承诺。

`v2` 中以下因果应明确废弃：

- 非-layerwise MLA 写侧当前“只有 rank 0 写”。
- IPC 可以直接“消除已确认的 GIL 瓶颈”。
- 非-GVA layerwise 是当前公开支持场景，且后续补交 1 层导致窗口失效。
- TP mismatch 的 backend `put()` 已确认会再次查询 key。
- `prefetch_layer_map` 可以决定 attention gate 是否多余。
- `_invalid_block_ids` getter 未加锁。
- 任务 10 所列高风险场景普遍缺少测试。
