# vLLM-Ascend KV Pool 外部开发者任务发布（v2，10 项）

> 目标仓库：`vllm-project/vllm-ascend`
> 代码路径：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 验收人：@赵鹏博
> 发布日期：2026-08-12

## 通用约定

- **代码基线**：vllm-ascend 最新 `main`
- **硬件**：Ascend NPU（提交时注明型号 + 卡数 + TP/CP/PP 配置）
- **方法开放**：本清单只描述问题与期望目标，**不限定实现方法**。开发者可自主选择方案，PR 描述中说明动机、方案、风险即可。
- **回归红线**：现有单测全绿；精度与功能不退化（greedy / non-greedy 输出一致）
- **交付件通用要求**：PR + 设计说明（改动动机 / 方案 / 风险 / 回退路径）+ 补充或更新 `tests/ut/distributed/ascend_store/` 下对应单测；性能类任务需附改动前后对比数据；发现新问题需提 follow-up issue 并回链本任务

## 任务清单

| # | 维度 | 优先级 | 标题 |
|---|------|--------|------|
| 1 | Perf | P1 | MLA 读侧去重 |
| 2 | Arch/Perf | P0 | 传输路径改 IPC，消除 GIL 瓶颈 |
| 3 | Perf | P1 | 非-GVA layerwise prefetch 行为修正 |
| 4 | Perf | P1 | ZMQ lookup payload 裁剪 |
| 5 | Perf | P2 | key 生成嵌套循环开销 |
| 6 | Perf | P3 | TP mismatch 路径重复查询 |
| 7 | Perf | P2 | `block_hash_to_str` 重复转换 |
| 8 | Perf | P2 | layerwise gate 对非复用层过度同步 |
| 9 | Correctness | P2 | `_invalid_block_ids` 锁保护范围审计 |
| 10 | Ext | P1 | 高风险路径测试覆盖补强 |

---

## 任务 1：[Perf] MLA 读侧去重

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

MLA 写侧已做去重（仅 rank 0 写入 KV 池，见 `_store_kv` 内 `tp_rank % put_step` 判断），但读侧未对齐：每个 TP rank 各自从 KV 池取同一份数据到各自 buffer（`start_load_kv` 读路径）。TP 越大冗余越多，且 MLA 数据量小，固定开销占比高。

### 任务

实现 MLA 读侧去重：仅一个 rank 从 KV 池取数，再分发到其余 TP rank，消除多 rank 重复 I/O。

* 模型：MLA 架构（如 GLM-5.2 等）
* 路径：worker 侧 `start_load_kv` 读路径
* 数据分发方式（broadcast / scatter / 其他）由开发者自定，需保证各 rank 最终拿到的数据与去重前逐字节一致

### 验收标准

#### 1. 功能正确性

* 去重后各 TP rank 拿到的数据与去重前逐字节一致
* TP mismatch 路径不破坏去重逻辑
* MLA / 非 MLA 模型均无回归

#### 2. 性能验证

* TP=2/4/8 下改动前后 get 调用次数与端到端延迟对比
* 长序列（16K/64K）下的收益数据

#### 3. 交付件

* PR + 设计说明 + 单测 + 性能数据

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数 + TP）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* `put_step` 与读侧 step 的语义对齐

### 任务周期

* 发布：2026-08-12
* 回收：2026-09-30

---

## 任务 2：[Arch/Perf] 传输路径改 IPC，消除 GIL 瓶颈

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

`ascend_store` 的 save/load 当前通过进程内 daemon 线程完成（`KVCacheStore*Sending/RecvingThread`），主线程与 send/recv 线程之间用 `queue.Queue` + `threading.Event` 协作。每进程最多 1 个 send 线程 + 1 个 recv 线程（`_start_kv_transfer_threads` 在三种模式间二选一）。部分热路径在 Python 层执行：`PoolKey.to_string`、`_handle_stored_request` 内 addrs/sizes 构建、key 生成器循环，与 forward 的 Python 层代码共享同一进程的 GIL。

随着多 rank 并行传输、独立崩溃隔离、NPU stream/event 跨进程协作等需求出现，单进程多线程架构在扩展性上存在瓶颈。

### 任务

将传输路径由"进程内多线程 + queue.Queue"改造为"独立传输子进程 + IPC"，使 key 构建、buffer 调度、I/O 提交在独立子进程执行，不再与 worker 主进程共享 GIL。

* 范围：三种传输模式（非 layerwise / key layerwise / GVA layerwise）
* 改造范围较大，建议分阶段推进，阶段划分与方法由开发者自定

### 验收标准

#### 1. 功能正确性

* 三种传输模式功能与改造前一致
* greedy / non-greedy 输出一致（精度无回归）
* 子进程异常不拖垮 worker 主进程，且能被重启或降级回进程内线程模式

#### 2. 性能验证

* 附改造前后对比数据：吞吐 / 延迟 / key 构建 CPU 占比 / IPC 开销

#### 3. 交付件

* 设计文档（含方案选型、风险清单、回退预案）
* 原型对比报告
* PR + 设计说明 + 单测（覆盖子进程启停、异常恢复、跨进程 buffer 共享、IPC 通道断连）

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数 + TP/CP/PP 配置）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* NPU KV cache buffer 跨进程共享是核心难点
* layerwise 模式的 NPU Event 跨进程语义需重新设计（`AttentionComputeStartGate`）
* 需保留降级路径，平台不支持或子进程启动失败时能回退到进程内线程模式

### 任务周期

* 发布：2026-08-12
* 回收：2026-10-31（长线任务，分阶段交付）

---

## 任务 3：[Perf] 非-GVA layerwise prefetch 行为修正

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

layerwise 架构的核心价值是"I/O 传输与 attention 计算重叠"，但非-GVA 路径存在两个问题：(1) 默认 `layerwise_prefetch_layers=1`，预取窗口过小；(2) `submit_count` 仅在第 0 层使用 `num_prefetch_layers`、其余层固定提交 1 层，导致后续层的预取窗口无法维持。两者叠加使非-GVA layerwise 模式下 load 与 attention 实际串行，重叠设计失效。对比 GVA 路径默认 prefetch=8，差距明显。

相关代码：`layerwise_prefetch_layers` 默认值、`_submit_ready_layer_loads` 内 `submit_count` 提交逻辑、`wait_for_layer_load` 等待逻辑。

### 任务

修正非-GVA layerwise 模式下的预取提交逻辑，使预取窗口在所有层都能有效维持，实现 I/O 与 attention 重叠。

* 路径：非-GVA layerwise load 提交与等待逻辑
* 修正点：默认值与 `submit_count` 在非第 0 层的提交策略
* 具体默认值与提交策略由开发者自定，需通过性能数据验证窗口选择合理

### 验收标准

#### 1. 功能正确性

* 改动后非-GVA layerwise 模式输出不变（精度无回归）
* 不引入死锁 / 资源越界

#### 2. 性能验证

* NPU 计算流利用率提升（profiler 数据）
* 端到端延迟下降曲线（prefetch=1/2/4/8 对比）
* 不同 num_layers / 序列长度下的收益

#### 3. 交付件

* PR + 设计说明 + 性能数据曲线 + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数 + TP）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* 预取窗口放大需评估 HBM / buffer 占用上限

### 任务周期

* 发布：2026-08-12
* 回收：2026-09-30

---

## 任务 4：[Perf] ZMQ lookup payload 裁剪

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

scheduler 调用 `LookupKeyClient.lookup()` 时把完整的 `request.block_hashes` 传给 client，再由 worker 侧用 `hbm_hit_tokens` 跳过前缀。对长 prompt，这放大了 IPC payload 和 msgpack 编解码成本。

相关代码：scheduler 侧 `lookup` 调用、`LookupKeyClient.lookup` 编码、worker 侧 `hbm_hit_tokens` 跳过前缀逻辑。

### 任务

减少 ZMQ lookup 的 IPC payload 与编解码开销。

* 路径：scheduler → ZMQ client → worker lookup server
* 目标：只发送待查询部分，不传冗余前缀

### 验收标准

#### 1. 功能正确性

* 裁剪后命中结果与发完整 hashes 一致
* 边界（hbm_hit_tokens = 0 / = 全长）处理正确
* 现有单测全绿

#### 2. 性能验证

* 长 prompt（16K/64K）下 IPC payload 大小下降
* lookup 延迟下降

#### 3. 交付件

* PR + 设计说明 + 性能数据 + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* offset 语义需 scheduler 与 worker 双向对齐

### 任务周期

* 发布：2026-08-12
* 回收：2026-09-30

---

## 任务 5：[Perf] key 生成嵌套循环开销

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

scheduler 侧生成查询 key 时有 5 层嵌套 for 循环（block_hash × pcp × dcp × head_or_tp × pp），每个 block 都构造 `KeyMetadata` + `PoolKey` 对象再逐层 `to_string`。命中检查频率受多维并行配置累乘影响。

相关代码：`_generate_store_query_keys`。

### 任务

降低 key 生成的 Python 层开销。

* 路径：scheduler 侧 `_generate_store_query_keys`
* 目标：消除逐对象构造与逐层 `to_string` 的开销

### 验收标准

#### 1. 功能正确性

* 生成的 key 集合与改动前完全一致（集合相等）
* 现有单测全绿

#### 2. 性能验证

* 该函数耗时下降（profile 对比）
* 多 PP / 多 head_or_tp 组合下的收益

#### 3. 交付件

* PR + 设计说明 + 性能数据 + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* PCP/DCP 通常为 1，收益主要来自 head_or_tp / pp 维度

### 任务周期

* 发布：2026-08-12
* 回收：2026-10-31

---

## 任务 6：[Perf] TP mismatch 路径重复查询

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

写侧 TP mismatch 路径在 `_store_kv_tp_mismatch` 中先 `lookup()` 查已有缓存以过滤 missing keys，过滤后再对 missing keys 调用 `put`——但 `put` 内部会再次查询 store 是否已存在同一 key，导致同一批 keys 被 lookup 查询两次。读侧 `_build_tp_mismatch_keys_and_addrs` 用双重循环构建 keys/addrs，外层按 block、内层按并行维度，存在可合并的冗余遍历。仅 TP mismatch 场景触发。

相关代码：`_store_kv_tp_mismatch`（写侧先 lookup 再 put，put 内二次查询）、`_build_tp_mismatch_keys_and_addrs`（读侧双重循环）。

### 任务

消除 TP mismatch 路径的二次查询与冗余遍历。

* 范围：仅 TP mismatch 场景的读 / 写路径
* 目标：同一批 keys 在单次 save/load 流程内只查一次，读侧循环合并

### 验收标准

#### 1. 功能正确性

* TP mismatch 场景输出与改动前一致
* 现有单测全绿

#### 2. 性能验证

* TP mismatch 场景下 lookup/put 调用次数下降
* 端到端延迟对比

#### 3. 交付件

* PR + 设计说明 + 性能数据 + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* "直接覆盖"语义需确认各 backend（mooncake / memcache）行为一致

### 任务周期

* 发布：2026-08-12
* 回收：2026-10-31

---

## 任务 7：[Perf] `block_hash_to_str` 重复转换

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

`_alloc_gvas_for_save`（save 侧）里，同一个 `group_block_hashes[block_idx]` 在 candidate_keys 列表推导、while 循环、for 循环里被 `block_hash_to_str`（`.hex()`）转换 3 次，产生三个内容相同的临时 str。load 侧 key 构造模式相同但只转换 1 次，无此冗余。

相关代码：`_alloc_gvas_for_save`（save 侧 3 次转换）、`_prepare_load_gvas`（load 侧仅 1 次，对比）。

### 任务

消除 save 侧 `block_hash_to_str` 的重复转换。

* 路径：`_alloc_gvas_for_save`
* 目标：同一 hash 在单次 save 流程内只转换一次

### 验收标准

#### 1. 功能正确性

* 生成的 key 与改动前一致
* 现有单测全绿

#### 2. 性能验证

* save 侧 hex 转换次数从 3× 降到 1×（profile 对比）
* 长 prompt（block 数大）下收益

#### 3. 交付件

* PR + 设计说明 + 性能数据 + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* 无

### 任务周期

* 发布：2026-08-12
* 回收：2026-10-31

---

## 任务 8：[Perf] layerwise gate 对非复用层过度同步

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

layerwise 预取层（`layer_id != current_layer`）的 load task 一律携带 `attention_start_gate`，recv 线程在 `_handle_request` 里 `gate.wait()` 阻塞直到计算流到达 attention 边界。`prefetch_layer_map` 定义了预取层与计算层之间的 buffer 复用关系（某预取层会覆盖某计算层正在使用的 buffer 时，需 gate 同步）。但当前 gate 附着不看 `prefetch_layer_map`，对所有预取层一律加 gate——对不在 map 中的预取层（无 buffer 复用），gate 是多余的，load 本可立即开始。单 recv 线程被这种多余 gate 阻塞会卡住后续所有层的 load（recv 线程串行处理队列）。

相关代码：`_submit_ready_layer_loads` 内 `submit_layer_load` 的 gate 附着条件、layerwise `_handle_request` 内 `gate.wait()`、`AttentionComputeStartGate.wait`（硬阻塞）、`prefetch_layer_map`（buffer 复用关系定义）。

### 任务

让 gate 仅在确有数据竞争时生效，非复用层不阻塞。

* 路径：layerwise recv 线程的 gate 附着与等待
* 目标：非复用预取层立即 load，复用层保留 gate

### 验收标准

#### 1. 功能正确性

* 复用层 gate 行为不变（buffer 数据安全）
* 非复用层 load 立即开始，无数据竞争
* 现有单测全绿

#### 2. 性能验证

* recv 线程被 gate 阻塞时间下降（profiler 时间线）
* 预取层就绪更早，attention 前等待减少

#### 3. 交付件

* PR + 设计说明 + 性能数据 + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* GVA 复用场景 gate 是正确性需要，不能去掉

### 任务周期

* 发布：2026-08-12
* 回收：2026-10-31

---

## 任务 9：[Correctness] `_invalid_block_ids` 锁保护范围审计

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

`_invalid_block_ids` 有锁保护，但读取它的路径（如 `get_block_ids_with_load_errors`）是否都在锁内未核实。若存在锁外读取，可能与 recv 线程的并发写入产生竞态，导致返回不一致的 invalid 集合（漏报或重复）。

相关代码：`_init_kv_transfer_config` 内 `_invalid_block_ids` 与 `_invalid_block_ids_lock` 定义、`_start_kv_transfer_threads` 将其传给 recv 线程。

### 任务

确保 `_invalid_block_ids` 所有访问点都在锁保护内（或通过不可变快照返回），消除竞态。

* 范围：所有 `_invalid_block_ids` 读写访问点
* 目标：并发场景下 `get_block_ids_with_load_errors` 返回一致结果

### 验收标准

#### 1. 功能正
* 所有 `_invalid_block_ids` 读写均在锁保护内（或通过不可变快照返回）
* 并发场景下 `get_block_ids_with_load_errors` 返回一致结果

#### 2. 回归保护

* 现有单测全绿
* 新增并发读写单测（多线程同时 load 失败 + 查询）

#### 3. 交付件

* PR + 审计清单（每个访问点的锁状态） + 单测

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* `get_block_ids_with_load_errors` 是否被计算流热路径调用（若是，锁竞争需评估）

### 任务周期

* 发布：2026-08-12
* 回收：2026-10-31

---

## 任务 10：[Ext] 高风险路径测试覆盖补强

> 关联任务池：#9079 | 验收人：@赵鹏博

### 背景

`tests/ut/distributed/ascend_store/` 下已有 `test_config_data.py` / `test_pool_worker.py` / `test_pool_scheduler.py` / `test_kv_transfer.py` / `test_coordinator.py` 等，但对高风险路径的覆盖仍不均衡。异常路径、GVA 批量元数据、multi-group 失败恢复、异步 load/save 状态机仍需补强。

待补强路径与期望场景：
* `KVCacheTransferThread.run` 线程异常分支：线程内 I/O 异常是否被正确捕获、是否传播到主流程、是否被静默吞掉
* `_alloc_gvas_for_save` / `_prepare_load_gvas` GVA 元数据批量：批量分配的部分失败回滚、group_block_hashes 边界（空 / 单 / 满）
* `LookupKeyClient.lookup` 及 ZMQ lookup server：server 无响应 / 连接断开 / 超长 payload 的编解码异常
* layerwise `_handle_request` 同步等待与失败传播：gate.wait 超时、load 失败后 attention 是否被错误放行、失败标记是否一致

### 任务

优先补"会卡死、会吞错、会错标完成"的高风险路径测试，而不是只补纯逻辑函数的 happy path。

* 范围：上述高风险路径
* 目标：异常注入 / 并发 / 失败传播的针对性测试覆盖

### 验收标准

#### 1. 功能正确性

* 新增测试不改变生产行为
* 现有测试全绿

#### 2. 覆盖指标

* 上述高风险路径有针对性测试（异常注入 / 并发 / 失败传播）
* 覆盖率报告显示相关路径覆盖提升

#### 3. 交付件

* PR + 新增测试文件 + 覆盖率对比

### 环境约定

* vllm-ascend：最新 main
* 硬件：Ascend NPU（注明型号 + 卡数）
* 关联任务池：#9079
* 验收人：@赵鹏博

### 重点关注

* 建议尽早落地，为后续重构提供安全网

### 任务周期

* 发布：2026-08-12
* 回收：2026-09-30
