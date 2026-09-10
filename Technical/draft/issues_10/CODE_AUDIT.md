# issues_10 严格代码审计报告

> 修订状态：本报告先审计了原任务表述；随后已于 2026-08-16 按审计结论修订 10 份任务原文。修订包括性能任务 Phase 0/Go-No-Go、后端/机器未知项验证，以及 kv-07、kv-24、kv-31 的事实边界。下文保留原始问题判定和代码证据，便于追溯修订原因。

## 1. 审计结论

本目录中的 10 个任务没有一个是完全凭空构造的，但它们并不都属于同一种“问题”：

- `kv-25`、`kv-27`、`kv-28` 是当前代码中可以闭合证明的正确性或可靠性缺陷。
- `kv-26` 是当前代码中可以闭合证明的保存结果不可观测、KV event 可能失真的可靠性缺陷，但通常不是模型正确性 P0。
- `kv-01`、`kv-07`、`kv-08`、`kv-17` 所描述的当前调用形态真实存在；它们是否能带来端到端性能收益，仓库内没有 benchmark 或硬件数据可以证明，因此只能作为 PoC/性能验证任务成立。
- `kv-24` 的“缺少统一解析入口”成立，但仓库并非完全没有 typed parsing；layerwise layout 已有局部类型转换、范围检查和 direct/MultiConnector 提取逻辑。
- `kv-31` 的接口混杂事实成立，但当前路由已经用 `backend_name == "memcache"` 限定 GVA 路径，仓库中没有一个用户可直接选择的“GVA + 非 GVA backend”组合。它是低优先级的类型边界清理，不是已发生的生产故障。

因此，当前 10 个任务可以继续保留，但 `kv-07`、`kv-24`、`kv-31` 的事实表述需要按本报告收窄；4 个性能任务不得把“调用次数可以减少”写成“性能已经得到提升”。

## 2. 基线、范围和判定标准

- 目标仓库：`D:\lzy\project\kv_pool\code\vllm-ascend`
- 分支：`main`
- 审计提交：`d5e9816065ede613327d93908f87fee9f5c47128`
- 提交时间：`2026-08-15 21:09:37 +0800`
- 审计对象：本目录的 10 份 `issue_kv-*.md`
- 审计方法：任务原文逐条拆解，沿 scheduler、connector、worker、transfer thread、backend 和单测反向核对；只把当前提交中能定位到的代码事实作为“存在”的证据。

本文使用以下判定：

| 判定 | 含义 |
|---|---|
| 成立 | 当前代码存在可达路径，任务核心事实和因果链均能由代码闭合证明 |
| 成立，但收益未证实 | 当前调用形态存在，但性能收益必须由 NPU/后端实测证明 |
| 部分成立/需修正文案 | 有真实代码事实，但任务中的适用范围、现状或因果描述有需要修正之处 |

静态代码能够证明控制流、接口契约和错误传播；它不能证明真实 MemCache 服务的 lease 引用计数语义，也不能证明 NPU 上的 TTFT、吞吐或 broadcast 成本。本文不会用推测替代这两类证据。

## 3. 总体判定

| 任务 | 真实性与准确性 | 建议优先级 | 审计结论 |
|---|---|---:|---|
| kv-01 MLA TP read dedup | 成立，但收益未证实 | P2 | MLA TP 内确实会生成相同 key 集并由每个 rank 各自 `get`；leader+broadcast 是否更快未知 |
| kv-07 non-layerwise batching | 部分成立，收益未证实 | P2 | per-request backend 调用存在；“partial key 包含 request ID”只适用于 GVA layerwise partial key，不适用于普通 non-layerwise key |
| kv-08 GVA metadata RPC aggregation | 成立，但收益未证实 | P2 | 批量入口内部仍按 request/group 发 metadata RPC；聚合安全边界真实，收益未知 |
| kv-17 offset-aware lookup | 成立，但收益未证实 | P2 | client 发送完整 hashes，且直接切 suffix 会破坏绝对 token 位置语义；payload/TTFT 收益未知 |
| kv-24 typed config parser | 成立，需补充现状限定 | P2 | 缺少统一 schema；但已有局部 typed parser 和 direct/MultiConnector 提取 helper |
| kv-25 transfer fatal protocol | 成立 | P1 | startup 可永久等 ready，fatal 后当前项/排队项/资源没有统一终止协议 |
| kv-26 Backend.put result | 成立 | P2 | 三个 backend 的普通 `put` 均不向 sender 返回 per-key 结果，event 可错误宣告 stored |
| kv-27 ZMQ recovery | 成立 | P0 | REQ 无 timeout，server 无逐请求异常保护；一次异常即可造成永久等待 |
| kv-28 hybrid load failure | 成立 | P0 | multi-group 部分 load 失败只记日志，async 还会正常 finished，存在残缺 KV 进入 forward 的路径 |
| kv-31 backend capability | 部分成立，维护性任务 | P3 | 基类确实混入 GVA 方法；但当前固定 backend 路由已避免可配置的错误组合 |

## 4. 逐项证据与逻辑闭环

### 4.1 kv-01：MLA non-layerwise TP 读去重 PoC

**判定：现象成立，性能收益未证实，P2 合理。**

代码事实：

1. MLA 初始化时强制 `num_kv_head = 1`。当 `tp_size > 1` 时，`put_step = tp_size`，同一 TP 组内各 rank 的 `head_or_tp_rank = tp_rank // tp_size = 0`，见 [pool_worker.py#L191](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L191)。
2. 普通 key 包含 `head_or_tp_rank` 而不包含本地 TP rank；因此同一 TP 组的 MLA rank 在其余维度相同时生成同一组 key，见 [metadata.py#L94](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L94)。
3. non-layerwise 同步 load 在每个 worker 进程中遍历 requests，构造本地地址后直接调用一次 `self.m_store.get(...)`，没有 leader/rank 跳过条件，见 [pool_worker.py#L888](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L888) 和 [pool_worker.py#L968](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L968)。
4. 写侧不能类比成“只有 rank 0 写”：sender 通过 `shard_rank=self.tp_rank % self.put_step` 和 `shard_size=self.put_step` 分片，见 [kv_transfer.py#L788](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L788)。

逻辑闭环：同一 MLA TP 组生成相同后端 key 集，而每个 rank 都执行后端 `get`，所以“存在 TP 内重复后端读取”成立。各 rank 的目标 NPU 地址不同，因此 leader 读完还必须引入 collective broadcast、stream/event 同步和全 rank 一致的失败协议；代码不能证明这些新增成本低于重复读取。

现有测试只验证 MLA 初始化把 `num_kv_head` 设为 1，且主要初始化用例不是 TP=2/4/8 的端到端 load，见 [test_pool_worker.py#L273](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_pool_worker.py#L273)。另有测试明确关闭 MLA 的 TP mismatch 路径，见 [test_pool_worker.py#L2105](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_pool_worker.py#L2105)。因此任务按“可开关 PoC + 硬件性能门槛”发布是准确的；不得预先宣称 leader+broadcast 更快。

### 4.2 kv-07：non-layerwise backend batching

**判定：主体成立，但一处 key 语义表述错误；性能收益未证实，P2 合理。**

代码事实：

1. 同步 load 对每个 request 分别构造列表并调用 `get`，见 [pool_worker.py#L888](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L888)。
2. async load 在 worker 侧逐 request 入队，见 [pool_worker.py#L918](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L918)；recv thread 的一次 `_handle_request` 只处理一个 `ReqMeta` 并调用一次 `get`，见 [kv_transfer.py#L923](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L923)。
3. save 同样逐 request 入队，sender 再按 request/group 做 `exists` 和 `put`，见 [pool_worker.py#L1758](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1758) 和 [kv_transfer.py#L807](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L807)。三个 backend 的普通接口本身均接受 key/address/size 列表。
4. 普通 non-layerwise `PoolKey` 的字段和序列化结果没有 `req_id`，见 [metadata.py#L94](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L94)。当前明确包含 `request.req_id` 的 partial key 是 GVA layerwise 专用 `_make_layerwise_partial_key`，见 [pool_worker.py#L1175](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1175)。

逻辑闭环：单个 connector step 中存在合并多个 request 后端调用的结构性机会，且接口能承载列表，所以任务主体真实。跨 request 合并后，重复 key、各自 block IDs、mask、circular shift、event、取消和完成状态仍必须逐项回填；这也是任务中 descriptor 设计要求成立的原因。

必须修正的文案：当前任务“full-block key 可跨 request 重复，partial key 才包含 request ID”不能作为普通 non-layerwise 的通用事实。准确表述应为：“普通 `PoolKey` 不含 request ID；仅 GVA layerwise 的 request-scoped partial key 含 request ID。”

仓库没有 batch size 1/4/8/16 的 backend 调用次数、尾延迟和 TTFT 数据，所以这里只能确认 batching opportunity，不能确认收益。

### 4.3 kv-08：GVA metadata RPC 聚合

**判定：调用形态和安全约束成立，性能收益及 lease 去重语义未证实，P2 合理。**

代码事实：

1. `_alloc_gvas_for_save(requests)` 和 `_prepare_load_gvas(requests)` 虽然接收 request 列表，但内部仍是 `for request` 再 `for group_id`，见 [pool_worker.py#L1201](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1201) 和 [pool_worker.py#L1367](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1367)。
2. `batch_alloc`、`batch_get_key_info`、`batch_add_lease` 等调用位于上述循环内部，见 [pool_worker.py#L1286](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1286)、[pool_worker.py#L1447](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1447) 和 [pool_worker.py#L1472](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1472)。
3. 源码明确记录 `batch_alloc` 非幂等，重复对象会返回 `MMC_DUPLICATED_OBJECT`，见 [pool_worker.py#L335](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L335)。full GVA key 不含 request ID，而 partial GVA key 包含 request ID，因此两类 key 不能采用同一去重规则。
4. `batch_get_key_info` 的结果直接与 keys、block indices 做 `zip`，未先校验长度；短结果会被静默截断，见 [pool_worker.py#L1447](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1447)。相反，lease 返回长度已经显式校验，见 [pool_worker.py#L1472](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1472)。
5. multi-group lease 失败已有释放已获得 lease 并抛错的路径，见 [pool_worker.py#L1520](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1520)。聚合实现必须保持该语义。

逻辑闭环：当前批量入口没有跨 request/group 聚合 metadata RPC，故减少调用次数的机会存在；重复 full key、非幂等 allocation、partial key 和 rollback 使得“直接拼接全部列表”不正确。任务对 descriptor、返回长度检查和 rollback 的要求都有代码依据。

仓库只暴露 binding 调用，无法从本仓库证明 MemCache lease 是否按 key 引用计数、重复 add/remove 是否对称。因此 `batch_add_lease` 能否去重必须以 MemCache 契约或实验为准。实际 RPC 降幅和 TTFT 收益同样尚未证明。

### 4.4 kv-17：offset-aware lookup 后缀协议

**判定：现状和 offset 必要性成立，性能收益未证实，P2 合理。**

代码事实：

1. scheduler 调用 lookup 时传入 `request.block_hashes` 全量列表和 `hbm_hit_tokens`，见 [pool_scheduler.py#L565](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L565)。
2. client 对传入列表的每个 hash 执行 `.hex()`，编码全部 hash frames 后发送，见 [pool_scheduler.py#L1168](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1168)。当前 wire format 没有 `hash_start_token` 或等价 offset。
3. hybrid/coordinator lookup 会用 `hbm_hit_tokens` 计算 lookup 起点，见 [pool_worker.py#L2248](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2248)；non-hybrid fallback 仍把完整 hashes 交给从 token 0 构建 key 的路径，见 [pool_worker.py#L2339](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2339)。
4. token chunk 的 start/end 来自 hash 在列表中的索引，见 [metadata.py#L468](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L468)；更大 effective block 的 grouped hash 又选择每组末端的细粒度 hash，见 [metadata.py#L641](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L641)。

逻辑闭环：全量 payload 的事实成立。如果只发送 `block_hashes[k:]` 而不传绝对 offset，worker 会把 suffix 的第一个 hash 当作 token 0 对应 hash，从而改变 chunk 边界、grouped hash 和最终 key。因而任务提出版本化 offset-aware 协议不是过度设计，而是缩短 payload 时保持语义所必需。

但仓库没有 16K/64K prompt 下的 payload、编码时间、IPC 延迟和 TTFT 数据。现阶段只能证明“发送了可缩短的数据”，不能证明这部分是性能瓶颈。

### 4.5 kv-24：AscendStore typed config parser

**判定：统一解析入口缺失成立，但必须承认已有局部 parser；P2 可保留。**

代码事实：

1. 对 `extra_config.get`/`get_from_extra_config` 的静态搜索在 AscendStore 包内命中 26 个读取表达式，分布于 connector、scheduler、worker、metadata 和 layerwise layout，例如 [ascend_store_connector.py#L83](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L83)、[pool_scheduler.py#L94](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L94)、[pool_worker.py#L145](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145) 和 [metadata.py#L49](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L49)。没有一个统一、不可变的 parsed config 被这些角色共同消费。
2. 仓库已有 `get_gva_layerwise_config`，能够从 direct connector 或 `MultiConnector.connectors` 中提取 GVA layerwise extra config，见 [layerwise_cache_layout.py#L70](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py#L70)。
3. layerwise layout 已有 `_parse_int_config`、bool 拒绝、字符串整数转换、下界和 layer index 范围检查，见 [layerwise_cache_layout.py#L105](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py#L105) 和 [layerwise_cache_layout.py#L114](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py#L114)。
4. `discard_partial_chunks` 的两个当前分支都使用默认值 `True`，见 [pool_scheduler.py#L136](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L136)。它是上下文读取点，但当前代码不能支持“两个模式默认值已经不一致”的说法。

逻辑闭环：配置读取分散、转换规则不统一的维护问题真实存在；统一 schema 可以减少 scheduler/worker 漂移。不过“AscendStore 完全没有 typed parser”是错误表述，正确范围是“缺少覆盖 connector-owned 字段并由各角色共享的统一 parser”。现有 layerwise parser 和 direct/MultiConnector helper 应被复用或迁移，而不是被重复实现。

该任务主要降低维护风险，并不修复当前已证明的用户可见故障。P2 是否合适取决于它是否作为 kv-27 timeout 等后续配置的稳定入口；kv-27 本身不应等待本任务完成。

### 4.6 kv-25：transfer thread 终止式失败协议

**判定：成立，P1 合理。**

代码事实：

1. `KVTransferThread.run` 在进入 `try` 之前调用 `self.m_store.set_device()`，随后才 set ready event，见 [kv_transfer.py#L496](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496)。如果 `set_device` 抛错，ready event 不会被设置。
2. worker 创建 transfer threads 后直接 `ready_event.wait()`，没有 timeout 或 startup exception 通道，见 [pool_worker.py#L456](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L456)。因此 startup failure 可以闭合推导为 creator 永久等待。
3. `_handle_request` 抛错时，基类只保存 `_fatal_error` 并 return，未调用 `_handle_request_exception`，也没有统一处理当前任务的 `task_done` 和排队任务，见 [kv_transfer.py#L503](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L503)。`add_request` 也不检查 fatal 状态，见 [kv_transfer.py#L347](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L347)。
4. 不能泛化为“所有异常都会挂死”：普通 sender 和 async recv 自己有 `finally: task_done()`，例如 [kv_transfer.py#L697](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L697) 和 [kv_transfer.py#L1037](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1037)；layerwise wait 也会轮询 `raise_if_failed`，见 [pool_worker.py#L1701](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1701)。
5. async non-layerwise 的 `get_finished` 只读取 finished sets，没有调用 `raise_if_failed`，见 [pool_worker.py#L2078](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2078)。GVA lease 又主要在最终 layer 完成时释放，见 [kv_transfer.py#L1633](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1633)，中途 fatal 需要显式 cleanup。

逻辑闭环：startup 的无界等待是确定性缺陷；运行期 fatal 后，是否遗留 queue accounting、request state 或 lease 取决于具体子类路径，基类没有统一保证。任务原文已正确区分“普通路径有局部 finally”和“仍缺完整终止协议”，没有夸大成所有异常都会死锁。

现有 [test_kv_transfer.py#L208](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_kv_transfer.py#L208) 只验证 fatal 后不继续处理第二个任务，没有验证 startup failure、unfinished task 归零、排队项取消、lease cleanup、fatal 后拒绝入队和 async polling 传播。任务列出的补测方向与真实缺口一致。

### 4.7 kv-26：Backend.put per-key 结果契约

**判定：成立，属于可靠性、事件准确性和可观测性问题，P2 合理。**

代码事实：

1. 抽象基类的 `put` 没有返回类型或结果语义，见 [backend/base.py#L50](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L50)。
2. Mooncake 检查 binding 返回码但只记录日志，异常也只记录日志，函数最终隐式返回 `None`，见 [mooncake_backend.py#L189](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L189)。MemCache 行为相同，见 [memcache_backend.py#L210](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L210)。Yuanrong 捕获异常后同样不返回结果，见 [yuanrong_backend.py#L147](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py#L147)。
3. sender 在调用 `put` 之前构造全部 `BlockStored`，忽略 `put` 返回值，并在调用后发布全部 events，见 [kv_transfer.py#L829](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L829) 和 [kv_transfer.py#L888](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L888)。因此失败 key 可以被宣告为 stored。
4. scheduler 的 delayed-free 流程需要 send completion 来释放本地 blocks，见 [pool_scheduler.py#L1086](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1086)。远端保存失败不能简单阻止本地生命周期结束，也不应复用 load-side invalid block 状态。

逻辑闭环：backend 已经观察到 per-key/raw batch 失败，但信息在 backend 边界被丢弃；sender 因而无法精确生成 event。这是现有代码中的确定问题。任务把“本地读取完成”和“远端持久化成功”拆开是正确的。

现有 backend 测试反而固化了“出错只记日志、不抛错”的行为，例如 [test_backend.py#L329](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_backend.py#L329)、[test_backend.py#L457](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_backend.py#L457) 和 [test_backend.py#L619](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_backend.py#L619)，没有 sender event 与 per-key 结果一致性测试。

需要补充的契约细节：某些 backend 的“对象已存在/重复写”原始返回码可能仍表示 key 已可用。统一结果不能只做 `raw_code == 0`，而应明确区分“key 最终可用/已存储”和“写入不可用”，同时保留 raw backend code 供观测。

### 4.8 kv-27：ZMQ REQ/REP timeout 与恢复

**判定：成立，是可导致永久阻塞的可用性缺陷，P0 合理。**

代码事实：

1. client 创建 REQ socket 后同步 `send_multipart`，紧接着无 timeout 地阻塞 `recv()`，见 [pool_scheduler.py#L1156](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1156)。
2. server daemon loop 从 `recv_multipart`、frame 下标访问、msgpack decode、worker lookup 到 response conversion 全部没有逐请求 `try/except`，见 [ascend_store_connector.py#L293](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L293)。任一异常都会结束 daemon thread，且已 receive 的 REP 请求得不到 reply。
3. response 没有版本、状态或固定长度校验；client 对任意 bytes 直接 `int.from_bytes`。空 response 会被解释为 0，与合法 remote miss 不可区分，超长 response 也不会被拒绝，见 [pool_scheduler.py#L1185](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1185)。
4. 本仓库固定的上游 vLLM 提交记录在 [.github/vllm-main-verified.commit](../../../code/vllm-ascend/.github/vllm-main-verified.commit)。对该提交 `58d3918e...` 的 `make_zmq_socket` 核对显示 helper 只按参数设置 `LINGER`，没有设置 `RCVTIMEO`/`SNDTIMEO`，因此不能依赖 helper 隐式提供 timeout。

逻辑闭环：server 一次解析或处理异常即可不发送 reply 并退出；client 没有 timeout，会永久阻塞。即使只给 `recv` 增加 timeout，REQ socket 在等待 reply 后也不能直接再次 send，否则触发 ZMQ REQ/REP 状态机错误；所以任务要求 timeout 后 `linger=0` 关闭并重建 socket 是必要条件。server 在 receive 后失败也必须发 typed error reply，或重建/终止其 REP socket。

当前 [test_pool_scheduler.py#L569](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_pool_scheduler.py#L569) 只覆盖正常 framing/response 和 close；没有直接的 `LookupKeyServer` 异常、timeout、坏报文、EFSM 恢复测试。有限重试耗尽后返回 remote miss 不会抹掉已有本地 HBM 命中，因而任务所述降级边界合理。

### 4.9 kv-28：hybrid KV load 失败传播

**判定：成立，是可能使用残缺 KV 的正确性缺陷，P0 合理。**

代码事实：

1. 同步 non-layerwise load 把多个 group 的 keys 合并为一次 `get`。当返回部分失败或 `None` 时，single-group 更新 `_invalid_block_ids`，但 multi-group 仅记录 error 后继续返回，见 [pool_worker.py#L980](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L980)。
2. async recv 的 multi-group 分支也只记录 error，随后无条件调用正常的 `set_finished_request(req_id)`，见 [kv_transfer.py#L997](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L997) 和 [kv_transfer.py#L1037](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1037)。
3. 当前 connector output 暴露 invalid block IDs 和正常 finished request sets，但没有 request-level failed set。上述 multi-group 分支刻意不报告 per-block invalid，以避免 scheduler 在 group 状态不一致时崩溃，却没有替代失败通道。
4. GVA layerwise 已采用另一种正确语义：multi-group 中一旦发现 invalid GVA/lease，就释放本 request 已获取的 leases 并抛 `RuntimeError`，见 [pool_worker.py#L1520](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1520)。现有单测也验证 forward 前抛错，见 [test_pool_worker.py#L1891](../../../code/vllm-ascend/tests/ut/distributed/ascend_store/test_pool_worker.py#L1891)。
5. platform 明确拒绝 hybrid 模型使用 `kv_load_failure_policy=recompute`，见 [platform.py#L944](../../../code/vllm-ascend/vllm_ascend/platform.py#L944)，所以不能把失败 block 塞回普通 recompute 通道作为现成修复。

逻辑闭环：部分 group 的 KV 已写入目标地址、另一部分失败，函数却正常结束；async 还把 request 标记为正常完成。上层因此缺少阻止 affected request 进入 attention/forward 的信号。任务的核心风险和“近期先 fail-fast、长期增加 request-level failure channel”的边界都符合代码现状。

这里需要与 kv-26 区分：kv-28 是 load 后可能消费残缺 KV 的模型正确性问题；kv-26 是 save 结果和 event 不准确，失败通常表现为未来 cache miss。

### 4.10 kv-31：backend capability model

**判定：接口混杂成立，但当前故障场景并未成为可达用户配置；作为 P3 维护任务保留。**

代码事实：

1. 最小 `Backend` 基类除通用 API 外还定义了 5 个默认抛 `NotImplementedError` 的 GVA/lease 方法，见 [backend/base.py#L35](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L35)。
2. 当前只有 `MemcacheBackend` 实现这些方法，见 [memcache_backend.py#L142](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L142)。`batch_is_exist` 则只是所有 backend 都可用的 `exists` 别名，应继续属于通用能力，见 [backend/base.py#L32](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L32)。
3. worker 和 scheduler 都将 GVA layerwise 路径定义为 `use_layerwise and backend_name == "memcache"`，见 [pool_worker.py#L145](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145) 和 [pool_scheduler.py#L169](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L169)。Mooncake/Yuanrong 的 layerwise 配置走 key-based layerwise 路径，而不是调用 GVA 方法。
4. backend 注册表目前固定为 Mooncake、MemCache、Yuanrong 三类，见 [backend/__init__.py#L17](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L17)。没有独立于 backend 名称的 `gva_mode=true` 用户开关。

逻辑闭环：基类被 GVA 专有方法污染是真实的设计问题，显式 capability/protocol 能改善扩展边界。但“用户配置 GVA mode + 非 GVA backend，直到首个请求才报错”在当前固定路由中没有可达配置，不能被描述成已经发生的正确性缺陷。

建议把验收重点改为：验证三种 backend 的 protocol/type 关系；MemCache GVA 路径在初始化时通过 capability narrowing；未来新增 backend 或显式 GVA mode 时能 fail-fast。不要为了测试一个当前不存在的配置组合而制造新的用户选项。P3 与其实际价值相符。

## 5. 跨任务关系与实施顺序

建议顺序仍以风险而非编号排列：

1. `kv-28`：先阻断残缺 hybrid KV 进入 forward。近期可先让本 engine step fail-fast，长期再设计 request-level failure output。
2. `kv-27`：消除 lookup 永久阻塞；timeout 可先用经过校验的默认值，不依赖 `kv-24`。
3. `kv-25`：统一 thread startup/fatal、queue accounting 和资源清理。`kv-28` 的 async 失败信号应与该协议对齐。
4. `kv-26`：统一普通 save 的 per-key 结果和 event 语义。它会给 `kv-07` 的批处理回填提供必要契约。
5. `kv-24`：集中配置解析，为 timeout 等后续字段提供稳定入口，同时保留现有局部 parser 的兼容语义。
6. `kv-01`、`kv-07`、`kv-08`、`kv-17`：以 profile/PoC 推进，不在数据之前承诺性能收益。
7. `kv-31`：独立的低优先级接口清理，不阻塞前述修复。

这里不是硬依赖图：`kv-28`、`kv-27`、`kv-25` 都可以先做最小正确性修复；`kv-07` 若先于 `kv-26` 实施，就必须在自身补出等价的 per-item 结果映射，不能把整个 batch 统一标成功。

## 6. 单测执行与证据边界

在该提交上，使用临时依赖目录执行了 AscendStore 相关纯 Python 单测：

```powershell
$env:PYTHONPATH='D:\lzy\project\kv_pool\tmp\.test-deps;D:\lzy\project\kv_pool\code\vllm-ascend'
python -m pytest -q -p no:cacheprovider `
  --confcutdir=tests\ut\distributed\ascend_store `
  -k 'not test_as_cache_tuple_list' `
  tests\ut\distributed\ascend_store\test_kv_transfer.py `
  tests\ut\distributed\ascend_store\test_backend.py `
  tests\ut\distributed\ascend_store\test_pool_scheduler.py `
  tests\ut\distributed\ascend_store\test_pool_worker.py `
  tests\ut\distributed\ascend_store\test_ascend_store_connector.py `
  tests\ut\distributed\ascend_store\test_metadata.py
```

结果：`345 passed, 10 skipped, 1 deselected in 0.97s`。

限制说明：

- `test_as_cache_tuple_list` 需要 `torch.ones`，当前轻量 mock 环境不提供该成员，因此主动 deselect；这不是产品测试失败。
- `test_layerwise_cache_layout.py` 的正常 conftest 依赖真实 PyTorch，当前环境无法收集；这不是产品测试失败。
- 当前机器没有 Ascend NPU 和真实 Mooncake/MemCache/Yuanrong 服务，因此没有执行硬件端到端、真实 RPC、collective broadcast、lease 引用计数或性能测试。
- 上述限制不影响 `kv-25`、`kv-27`、`kv-28` 的静态控制流结论，但意味着 4 个性能任务仍必须以目标硬件数据验收。

## 7. 修订落实状态

- `kv-07` 已改为：普通 non-layerwise `PoolKey` 不含 request ID；只有 GVA layerwise request-scoped partial key 明确包含 request ID。
- `kv-24` 已写明现有 layerwise 局部 typed parser 和 direct/MultiConnector helper，任务目标改为复用现有逻辑并统一剩余 connector-owned 字段。
- `kv-31` 已明确为未来扩展和类型边界清理，不再声称当前存在可配置的“GVA + 非 GVA backend”生产故障。
- 4 个性能任务已增加 Phase 0、Go/No-Go 阈值和 No-Go 交付方式；数据不支持时允许保持当前代码。
- `kv-25`、`kv-27`、`kv-28` 已增加故障注入/机器状态验证；`kv-26` 已要求按实际 backend 版本确认返回码，并区分 `AVAILABLE/FAILED/UNKNOWN`。

综合判断：这 10 个任务整体上是基于真实代码演进出来的任务集，其中 4 个是当前缺陷、4 个是待实测的性能机会、2 个是维护性/扩展性改造。按上述修正后，任务集与 `vllm-ascend@d5e9816` 的代码现状能够形成完整、可复核的逻辑闭环。
