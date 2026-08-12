# vLLM-Ascend KV Pool 外部开发者任务发布（v1，10 项）

> 目标仓库：`vllm-project/vllm-ascend`
> 代码路径：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 验收人：@赵鹏博
> 发布日期：2026-08-12
> 回收日期：2026-09-30（部分长线任务延至 2026-10-31，见各任务说明）

## 通用约定

- **代码基线**：vllm-ascend 最新 `main`
- **硬件**：Ascend NPU（提交时注明型号 + 卡数 + TP/CP/PP 配置）
- **方法开放**：本清单只描述问题与期望目标，**不限定实现方法**。开发者可自主选择方案，PR 描述中说明动机、方案、风险即可。
- **交付件通用要求**：
  - PR + 设计说明（改动动机 / 方案 / 风险 / 回退路径）
  - 补充或更新 `tests/ut/distributed/ascend_store/` 下对应单测
  - 性能类任务需附改动前后对比数据（吞吐 / 延迟 / 调用次数 / profiler 数据）
  - 发现新问题需提 follow-up issue 并回链本任务
- **回归红线**：现有单测全绿；精度与功能不退化（greedy / non-greedy 输出一致）

## 任务清单（10 项）

| # | 编号 | 维度 | 优先级 | 标题 |
|---|------|------|--------|------|
| 1 | kv-01 | Perf | P1 | MLA 读侧去重 |
| 2 | kv-35 | Arch/Perf | P0 | 传输路径改 IPC，消除 GIL 瓶颈 |
| 3 | kv-06 | Perf | P1 | 非-GVA layerwise prefetch 行为修正 |
| 4 | kv-17 | Perf | P1 | ZMQ lookup payload 裁剪 |
| 5 | kv-03 | Perf | P2 | key 生成嵌套循环开销 |
| 6 | kv-04 | Perf | P3 | TP mismatch 路径重复查询 |
| 7 | kv-10 | Perf | P2 | `block_hash_to_str` 重复转换 |
| 8 | kv-16 | Perf | P2 | layerwise gate 对非复用层过度同步 |
| 9 | kv-29 | Correctness | P2 | `_invalid_block_ids` 锁保护范围审计 |
| 10 | kv-33 | Ext | P1 | 高风险路径测试覆盖补强 |

> 选择原则：除 kv-01（已有方案）与 kv-35（重点架构项）外，其余 8 项均**相互独立、改动局部、收益明确**，可并行推进、互不阻塞。

---

## kv-01 [Perf] MLA 读侧去重

**优先级**：P1（已有方案，可直接落地）

### 问题

MLA 写侧已做去重（仅 rank 0 写），但读侧未对齐：每个 TP rank 各自从 KV 池取一份到各自 buffer。TP 越大冗余越多，且 MLA 数据量小，固定开销占比高。这是 MLA 模型读侧的主要浪费点。

### 证据

- 写侧去重：[pool_worker.py:1020](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1020)（`if self.tp_rank % self.put_step != 0: return`）
- 读侧无对应跳过：[pool_worker.py:867-980](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L867-L980)
- 专项方案参考：[MLA_KV读取去重优化讨论.md](file:///D:/lzy/project/kv_pool/llm-project/draft/MLA_KV读取去重优化讨论.md)

### 期望目标

实现 MLA 读侧去重，对齐写侧语义。验收要求：

- **功能**：去重后各 TP rank 拿到的数据与去重前逐字节一致；TP mismatch 路径不破坏去重；MLA / 非 MLA 模型均无回归
- **性能**：给出 TP=2/4/8 下 get 调用次数与端到端延迟对比；长序列（16K/64K）下的收益数据
- **交付**：PR + 设计说明 + 单测 + 性能数据

实现方式（broadcast 路径、collective 复用、step 语义对齐等）由开发者自定。

---

## kv-35 [Arch/Perf] 传输路径改 IPC，消除 GIL 瓶颈

**优先级**：P0（重点）

### 问题

`ascend_store` 的 save/load 全部走进程内 daemon 线程（`KVCacheStore*Sending/RecvingThread`），主线程与 send/recv 线程之间用 `queue.Queue` + `threading.Event` 协作。该架构的根本限制：

- 每进程仅 1 个 send 线程 + 1 个 recv 线程，无法横向扩展
- GIL 卡住 Python 层热路径：`PoolKey.to_string` 拼接、`_handle_stored_request` 内 addrs/sizes 径环构建、key 生成器循环均在 Python 层，无法与 forward 计算真正并行
- 即便后端 C 扩展释放 GIL，Python 层 key 构建仍串行

设计目标是"I/O 与 forward 重叠"，但实际被 GIL 拖累成"key 构建与 forward 串行"。长期看，单进程多线程架构无法支撑多 rank 并行传输、独立崩溃隔离、与 NPU stream/event 跨进程协作等新能力。

### 证据

- 传输线程类与启动点：[kv_transfer.py:496-1643](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496-L1643)、[pool_worker.py:453](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L453)
- 每进程仅 1 send + 1 recv：[pool_worker.py:328-329](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L328-L329)
- GIL 受影响的 Python 层热路径：[metadata.py:114](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L114)、[kv_transfer.py:846-855](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L846-L855)
- buffer 注册（跨进程共享难点）：[pool_worker.py:793](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L793)、[pool_worker.py:864](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L864)
- store 初始化（子进程重置代价）：[mooncake_backend.py:124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L124)
- 已有 IPC 先例（ZMQ，但目的不同）：[pool_scheduler.py:1199](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1199)
- 跨进程 event 协作难点：[attention_fence.py:27](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py#L27)
- 完整分析依据：[kv_pool_线程存取与GIL分析.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_线程存取与GIL分析.md)

### 期望目标

将传输路径由"进程内多线程 + queue.Queue"改造为"独立传输子进程 + IPC"，使 key 构建、buffer 调度、I/O 提交彻底脱离 worker 主进程的 GIL。

本任务改造范围大，建议分阶段推进（设计验证 → 原型对比 → 全路径落地），阶段划分与方法由开发者自定，但需满足：

- **功能**：三种传输模式（非 layerwise / key layerwise / GVA layerwise）功能与改造前一致；精度无回归；子进程异常不拖垮 worker 主进程且可恢复或降级
- **性能**：附改造前后对比数据（吞吐 / 延迟 / key 构建 CPU 占比 / IPC 开销），证明收益；若未达预期需产出 follow-up 报告
- **正确性边界**：NPU KV cache buffer 跨进程共享、store 子进程重初始化、IPC 序列化开销、layerwise NPU Event 跨进程语义、子进程崩溃时的 KV 一致性——这些是改造中必须解决的问题，方案需在设计中明确
- **降级回退**：保留进程内线程模式作为 fallback，平台不支持或子进程启动失败时自动降级
- **交付**：设计文档 + 原型对比报告 + PR + 单测

IPC 通道选型、序列化方案、子进程生命周期管理、跨进程 NPU 句柄方案等均由开发者自定。

---

## kv-06 [Perf] 非-GVA layerwise prefetch 行为修正

**优先级**：P1（可能是线上最大单点性能损失）

### 问题

layerwise 架构的核心价值是"I/O 传输与 attention 计算重叠"，但非-GVA 路径默认 `layerwise_prefetch_layers=1`，且 `submit_count` 仅在第 0 层使用 `num_prefetch_layers`、其余层固定提交 1 层。结果是非-GVA layerwise 模式下 load 与 attention 完全串行，每层 attention 前计算线程都阻塞等网络 I/O，NPU 计算流排空空闲，重叠设计失效。对比 GVA 路径默认 prefetch=8，差距巨大。

### 证据

- 默认值：[pool_worker.py:425](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L425)
- 提交逻辑：[pool_worker.py:1683](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1683)
- 等待逻辑：[pool_worker.py:1691-1707](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1691-L1707)

### 期望目标

修正非-GVA layerwise 模式下预取与计算的重叠关系，使 layerwise 设计真正生效。验收要求：

- **功能**：输出不变（精度无回归）；不引入死锁 / 资源越界
- **性能**：给出 NPU 计算流利用率提升数据（profiler）+ 端到端延迟下降曲线（prefetch=1/2/4/8 对比）+ 不同 num_layers / 序列长度下的收益
- **交付**：PR + 设计说明 + 性能数据 + 单测

预取窗口放大需评估 HBM / buffer 占用上限。具体默认值与 submit 策略由开发者自定。

---

## kv-17 [Perf] ZMQ lookup payload 裁剪

**优先级**：P1

### 问题

scheduler 调用 `LookupKeyClient.lookup()` 时把完整的 `request.block_hashes` 传给 client，再由 worker 侧用 `hbm_hit_tokens` 跳过前缀。对长 prompt，这放大了 IPC payload 和 msgpack 编解码成本。

### 证据

- scheduler 调用：[pool_scheduler.py:565-572](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L565-L572)
- client 编码：[pool_scheduler.py:1158-1178](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1158-L1178)
- worker 侧按 `hbm_hit_tokens` 跳过前缀：[pool_worker.py:2273-2288](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2273-L2288)

### 期望目标

减少 ZMQ lookup 的 IPC payload 与编解码开销。验收要求：

- **功能**：裁剪后命中结果与发完整 hashes 一致；边界（hbm_hit_tokens = 0 / = 全长）处理正确；现有单测全绿
- **性能**：长 prompt（16K/64K）下 IPC payload 大小下降 + lookup 延迟下降
- **交付**：PR + 设计说明 + 性能数据 + 单测

裁剪方式（offset + slice、只发后缀、或其他）由开发者自定，但 offset 语义需 scheduler 与 worker 双向对齐。

---

## kv-03 [Perf] key 生成嵌套循环开销

**优先级**：P2

### 问题

scheduler 侧生成查询 key 时有 5 层嵌套 for 循环（block_hash × pcp × dcp × head_or_tp × pp），每个 block 都构造 `KeyMetadata` + `PoolKey` 对象再逐层 `to_string`。命中检查频率受多维并行配置累乘影响。

### 证据

- [pool_scheduler.py:247-283](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L283)

### 期望目标

降低 key 生成的 Python 层开销。验收要求：

- **功能**：生成的 key 集合与改动前完全一致（集合相等）；现有单测全绿
- **性能**：该函数耗时下降（profile 对比）；多 PP / 多 head_or_tp 组合下的收益
- **交付**：PR + 设计说明 + 性能数据 + 单测

优化方式（对象复用、向量化、下沉等）由开发者自定。PCP/DCP 通常为 1，收益主要来自 head_or_tp / pp 维度。

---

## kv-04 [Perf] TP mismatch 路径重复查询

**优先级**：P3

### 问题

写侧 TP mismatch 路径先 `lookup()` 查已有缓存、过滤 missing keys 后再 `put`，存在重复查询；读侧 `_build_tp_mismatch_keys_and_addrs` 也是双重循环。仅 TP mismatch 场景触发。

### 证据

- 写侧先 lookup 再 put：[pool_worker.py:2005-2035](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2005-L2035)
- 读侧双重循环：[pool_worker.py:1936-1967](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1936-L1967)
- 非 mismatch 对比：[kv_transfer.py:717-818](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L717-L818)

### 期望目标

消除 TP mismatch 路径的重复查询与冗余循环。验收要求：

- **功能**：TP mismatch 场景输出与改动前一致；现有单测全绿
- **性能**：TP mismatch 场景下 lookup/put 调用次数下降 + 端到端延迟对比
- **交付**：PR + 设计说明 + 性能数据 + 单测

具体消除方式由开发者自定。"直接覆盖"语义需确认各 backend（mooncake / memcache）行为一致。

---

## kv-10 [Perf] `block_hash_to_str` 重复转换

**优先级**：P2

### 问题

`_alloc_gvas_for_save`（save 侧）里，同一个 `group_block_hashes[block_idx]` 在 candidate_keys 列表推导、while 循环、for 循环里被 `block_hash_to_str`（`.hex()`）转换 3 次，产生三个内容相同的临时 str。load 侧 key 构造模式相同但只转换 1 次，无此冗余。

### 证据

- save 侧 3 次转换：[pool_worker.py:1243-1268](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1243-L1268)
- load 侧仅 1 次（对比）：[pool_worker.py:1422-1425](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1422-L1425)

### 期望目标

消除 save 侧 `block_hash_to_str` 的重复转换。验收要求：

- **功能**：生成的 key 与改动前一致；现有单测全绿
- **性能**：save 侧 hex 转换次数从 3× 降到 1×（profile 对比）；长 prompt（block 数大）下收益
- **交付**：PR + 设计说明 + 性能数据 + 单测

具体消除方式由开发者自定。

---

## kv-16 [Perf] layerwise gate 对非复用层过度同步

**优先级**：P2

### 问题

layerwise 预取层（`layer_id != current_layer`）的 load task 一律携带 `attention_start_gate`，recv 线程在 `_handle_request` 里 `gate.wait()` 阻塞直到计算流到达 attention 边界。但对**非 buffer 复用**的预取层（`prefetch_layer_map` 无该层），不存在共享 buffer 数据竞争，gate 是多余的——load 可以立即开始。单 recv 线程被 gate 阻塞会卡住后续所有层的 load（recv 线程串行处理队列）。

### 证据

- gate 附着条件：[pool_worker.py:1670-1672](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1670-L1672)
- gate wait：[kv_transfer.py:1597-1599](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1597-L1599)
- gate 实现（硬阻塞）：[memcache_comm_fence.py:53-61](file:///D:/lzy/project/kv_pool/code/vllm-ascend/memcache_comm_fence.py#L53-L61)

### 期望目标

让 gate 仅在确有数据竞争时生效，非复用层不阻塞。验收要求：

- **功能**：复用层 gate 行为不变（buffer 数据安全）；非复用层 load 立即开始，无数据竞争；现有单测全绿
- **性能**：recv 线程被 gate 阻塞时间下降（profiler 时间线）；预取层就绪更早
- **交付**：PR + 设计说明 + 性能数据 + 单测

具体判断方式由开发者自定。GVA 复用场景 gate 是正确性需要，不能去掉。

---

## kv-29 [Correctness] `_invalid_block_ids` 锁保护范围审计

**优先级**：P2

### 问题

`_invalid_block_ids` 有锁保护，但读取它的路径（如 `get_block_ids_with_load_errors`）是否都在锁内未核实。若存在锁外读取，可能与 recv 线程的并发写入产生竞态，导致返回不一致的 invalid 集合（漏报或重复）。

### 证据

- 锁定义：[pool_worker.py:145-152](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L152)
- 传给 recv 线程：[pool_worker.py:570-571](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L570-L571)

### 期望目标

确保 `_invalid_block_ids` 所有访问点都在锁保护内（或通过不可变快照返回），消除竞态。验收要求：

- **功能**：所有读写均在锁保护内（或快照返回）；并发场景下 `get_block_ids_with_load_errors` 返回一致结果
- **回归**：现有单测全绿；新增并发读写单测（多线程同时 load 失败 + 查询）
- **交付**：PR + 审计清单（每个访问点的锁状态） + 单测

具体修复方式（锁内读 / 快照副本 / 锁粒度调整）由开发者自定。若 `get_block_ids_with_load_errors` 被计算流热路径调用，需评估锁竞争。

---

## kv-33 [Ext] 高风险路径测试覆盖补强

**优先级**：P1（是其他重构任务的安全网，建议尽早落地）

### 问题

`kv_pool` 不是"没有测试"，`tests/ut/distributed/ascend_store/` 下已有 `test_config_data.py` / `test_pool_worker.py` / `test_pool_scheduler.py` / `test_kv_transfer.py` / `test_coordinator.py` 等，但**对高风险路径的覆盖仍不均衡**。从源码复核看，异常路径、GVA 批量元数据、multi-group 失败恢复、异步 load/save 状态机仍需补强。

### 证据

- `tests/ut/distributed/ascend_store/` 下已存在相关单测文件
- 待补强路径：
  - [kv_transfer.py:496-518](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496-L518)（线程异常分支）
  - [pool_worker.py:1359-1534](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1359-L1534)（GVA 元数据批量）
  - [pool_scheduler.py:1146-1178](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1146-L1178)、[ascend_store_connector.py:293-340](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L293-L340)（ZMQ lookup）
  - [pool_worker.py:1670-1763](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1670-L1763)（同步等待与失败传播）

### 期望目标

优先补"会卡死、会吞错、会错标完成"的高风险路径测试，而不是只补纯逻辑函数的 happy path。验收要求：

- **功能**：新增测试不改变生产行为；现有测试全绿
- **覆盖**：上述高风险路径有针对性测试（异常注入 / 并发 / 失败传播）；覆盖率报告显示相关路径覆盖提升
- **交付**：PR + 新增测试文件 + 覆盖率对比

具体测试设计与覆盖范围由开发者自定。本任务是后续传输路径重构（如 kv-35）的安全网，建议尽早落地。

---

## 附录：任务独立性说明

- 除 kv-35（架构改造，分阶段）外，其余 9 项**相互独立、改动局部、收益明确**，可并行推进、互不阻塞
- kv-35 阶段 3 全路径落地时，其余 9 项应已合入，避免改动区域冲突
- kv-33（测试补强）建议作为第一批落地项，为所有后续改动铺安全网
- 各任务发现新问题需提 follow-up issue 并回链本任务

## 环境约定

- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数 + TP/CP/PP 配置）
- 关联任务池：#9079
- 验收人：@赵鹏博
