# MLA KV Pool 读去重：现状分析与可落地设计

**最终审核修订说明（2026-08-24）：**本版是对目录内分析的最终事实审计结果。已明确：框架级协议不等于所有 backend 默认支持；Mooncake 非 fabric-memory 才调用 `global_te.register_buffer`；当前 wrapper 无法向框架提供精确 bytes-written；普通同步 get 的对象生命周期不等同于 GVA/layerwise lease；step-level collective 必须先证明所有 TP rank（包括空 metadata）都会进入；backend read batch 与 broadcast chunk 不得混用。本文没有进行源码改动或 NPU 实测。

**日期：** 2026-08-24  
**范围：** `vllm-ascend` AscendStore KV Pool 框架，以及 Mooncake、LMCache、MemCache、YuanRong backend。  
**性质：** 设计分析和实施方案；本文不修改实现代码，不把候选分支视为已合入能力，也没有进行 NPU 实测。

## 0. 执行结论

### 0.1 是否可做

可以做，而且应在 KV Pool **框架层**做，而不是在 Mooncake backend 内部单独实现。当前代码已经具备第一阶段的主要条件：框架知道 MLA/TP/`put_step`/KV group/Pool key/目标 HBM span；写侧已经按 TP 去重；同步读入口明确；vLLM 有 TP group broadcast；backend 有统一 `get(keys, addrs, sizes)`。

第一阶段的核心路径是：owner 对每个 key 只调用一次 backend，直接写入自己的 KV cache；所有 backend read 完成后，再把成功的多 span 按受控 broadcast chunk 打包为 `uint8` staging，TP 内 broadcast，peer 解包到本地 KV cache。backend read batch 和 broadcast chunk 是两个层次，不能因为广播需要分块而重复读取同一个 key。

### 0.2 第一阶段边界

只在以下条件全部满足时启用：`use_mla=True`、`tp_size>1`、`put_step==tp_size`、`peer_tp_size==tp_size`、`tp_mismatch=False`、同一 local TP group、非 layerwise、非 async、非 sparse、非 hybrid、所有 load group 都确认是 replicated MLA，并且 backend 有已验证的同步完成/device-target/per-key-status capability adapter。第一阶段只 enable 已验证的 Mooncake 同步 adapter；其他情况继续旧的逐 rank backend read。

### 0.3 当前的 gap

当前不是“本质不能做”，而是缺少 replicated-read 协议：replicated group 判定、owner/passive、canonical read plan、一致性校验、多 span payload、bounded staging、统一 status/invalid 映射、backend completion/stream 语义、async/layerwise rendezvous。这些可以补齐，但在补齐前不能默认覆盖复杂路径。**“框架级”表示复制语义和协议放在 AscendStore 编排层，不表示所有 backend 自动具备安全条件。** 当前 Backend base 没有正式的 completion capability；第一阶段只能对已验证、具备 adapter 的 backend opt-in，其他 backend 必须保持旧路径。

### 0.4 推荐决策

进入同步纯 MLA 的 opt-in 实现，默认关闭；通过逐字节正确性、HCCL hang/OOM/stream race、非 MLA 回归、backend 调用次数和端到端性能验收后，再考虑默认开启及后续路径。

---

# 第一部分：现状分析

## 1. 基线和证据

本分析核对了：

- `D:\lzy\project\kv_pool\code\vllm-ascend`：目标 `main`=`d85e6714a`；当前 checkout=`5b199c84b`，多出的已有 refactor 未改变本文所依赖的 Mooncake backend 和同步读语义；
- `D:\lzy\project\kv_pool\code\vllm`：TP group、replicated KV、NIXL source-rank 参考；
- `D:\lzy\project\kv_pool\code\LMCache`：`dev/f9addd2`，`save_only_first_rank` 和 chunk broadcast；
- `D:\lzy\project\kv_pool\code\Mooncake-src`：`main/82e2799`，multi-buffer get、replica 和同步完成路径。

已有的 [implementation_plan.md](D:\lzy\project\kv_pool\llm-project\draft\mla_load\implementation_plan.md) 和 [mooncake_lmcache_mla_read_dedup_analysis.md](D:\lzy\project\kv_pool\llm-project\draft\mla_load\mooncake_lmcache_mla_read_dedup_analysis.md) 保存了 issue、源码和业界资料的详细证据；本文将其收敛为框架级结论和实施设计。

Issue #14140 的核心命题是：MLA latent KV 在 TP 内 replicated，写侧只保存一次，读侧却每个 TP rank 分别从 Pool 读取同一数据，造成随 TP 增长的重复 I/O。该命题已被当前 key/rank 和调用链源码直接支持；端到端收益仍必须 NPU 实测。

## 2. 当前框架调用链

主要链路是：scheduler 生成 `SchedulerOutput`，`KVPoolScheduler.build_connector_meta` 转为 `AscendConnectorMetadata.requests`，每个 worker 的 `AscendStoreConnector.start_load_kv` 调用 `KVPoolWorker.start_load_kv`，最后调用 backend 的 `get(keys, addrs, sizes)`。

`KVPoolWorker` 已掌握 `use_mla`、TP/PP/PCP/DCP rank、`put_step`、`peer_tp_size`、`tp_mismatch`、KV group/spec、block size、cache family、每个 layer/cache entry 的 base address/length/stride，以及 request 的 key/hash/block id/load mask。这些信息足够在框架层构造 replicated read plan。

但是 `Backend.get` 只表达“把数据写入指定多 span 地址并返回结果码”，没有正式表达同步完成、目标 stream、标准错误码、连续 payload 或跨 rank fan-out。现有 wrapper 的 0/非 0/`None` 约定对逐 rank read 足够，对 owner broadcast 需要更严格的 framework normalization；因此需要 capability/result adapter，不能因方案位于框架层就默认 Mooncake、MemCache、YuanRong 都能直接启用。

## 3. MLA 语义和写侧证据

DeepSeek MLA 的 compressed latent KV 与普通按 KV head 分片不同；vLLM 的 MLA 实现和 `TransferTopology.replicates_kv_cache()` 都支持 MLA replicated 语义，NIXL 也明确 pure MLA 只需一个 source，MLA+SSM 仍需读取 SSM source。

`pool_worker.py::_init_key_head_config` 在 MLA 下将 `num_kv_head` 视为 1；TP 大于 1 时 `put_step=tp_size`、`head_or_tp_rank=0`。`_process_save_for_layer_batch`、`_alloc_gvas_for_save` 和发送线程均跳过 `tp_rank % put_step != 0` 的 rank。因此纯 MLA 同一 TP group 的写侧 key namespace 确实已经去重。

写侧只解决 Pool 中存一份；读侧当前仍要填充每个 rank 的本地 HBM，所以没有额外协议时自然会每 rank 调一次 backend。

## 4. 当前读路径和范围

### 4.1 同步非 layerwise

`KVPoolWorker.start_load_kv` 在非 layerwise、非 async 分支中构造 key、`addr_list`、`size_list`、block id，按 `tp_rank` circular shift 后调用 `m_store.get`，再用 `record_failed_blocks` 处理返回码。MLA 各 rank 的 key 通常相同，差异主要是本地 HBM 地址和 shift；TP=8 时会读取 8 次。

这是第一阶段的明确切入点：调用同步、状态由当前 step 驱动、目标 span 已在 worker 中可见。

### 4.2 Async

`KVCacheStoreRecvingThread` 每 rank 独立 queue/thread，在 `_handle_request` 调 backend。当前没有跨 rank sequence；直接在此处调用 broadcast 可能将 rank 0 的 request A 与 rank 1 的 request B 配对，导致 HCCL hang 或数据错配。必须先增加 rendezvous，第一阶段不改。

### 4.3 Layerwise 和 GVA

Key-based layerwise 还受 `wait_for_save`、attention start gate、layer finished event、最后 layer completion 约束。MemCache/GVA path 还涉及 `batch_alloc`、lease、`batch_copy`、physical layer reuse 和 write-finish。它们不是同步 `get` 的简单变体，必须后续单独设计。

### 4.4 TP mismatch、hybrid 和特殊 group

MLA 的 `tp_mismatch` 标志不能单独证明 peer TP 相同，必须额外检查 `peer_tp_size==tp_size`。MLA+SSM/hybrid 可能同时包含 replicated MLA group 和按 rank 分片的 SSM group，不能只用 `use_mla` 复制整个 request。第一阶段排除 hybrid、sparse、压缩/DSV4；后续按 KV group/spec 分类。

## 5. 地址和布局约束

`metadata.py::prepare_value` 为一个 key 返回跨 layer/cache entry 的多个地址和 size，可能有 K/V tuple、padding、stride、physical layer reuse、多个 group。一个 key 不是一个连续 tensor，不能用一个 `numel()` 或一个指针表示 payload。

owner/peer 的裸 HBM 地址以及 local block id 可能不同，因此 canonical plan 只能比较 request 顺序、key/hash、group/layer 逻辑顺序、span 数、span size 和 chunk 边界；状态广播按 canonical entry index，peer 再映射自己的 block id。

从 cache storage 构造 byte view 还要处理子视图 offset、共享 storage、stride、注册区域边界和 NPU `uint8` view/copy 支持。建议显式 span copy，并在单测/debug 校验边界，不能假设 cache contiguous。

## 6. Backend、LMCache 和上游方案

### 6.1 Mooncake

Ascend wrapper 调 `batch_get_into_multi_buffers(keys, addrs, sizes)`。在非 fabric-memory 初始化路径中，`register_buffer` 通过 `global_te.register_buffer(ptrs, lengths)` 注册 KV cache 的 device memory；fabric-memory 路径跳过该注册，使用另一套统一内存地址语义。之后传给 Mooncake 的每个 span 必须符合对应模式的地址和生命周期约束，不能把临时 staging 或已释放 storage 的地址当作 backend 目标。Mooncake C++ 查询 metadata/replica，选择 memory/NOF/local disk/DISK/DFS，检查多 buffer 总容量，memory 可能直接写多个 slice，disk/DFS 可能先写连续临时 buffer 再 scatter，最后返回逐 key 状态。`Client::BatchGet` 等待 transfer future 后返回，Python binding 虽然会释放 GIL，但不改变同步 API 的完成等待语义，因此可作为 owner read；但 pack 所在 NPU stream 的可见性仍需确认。

Mooncake wrapper 会把正的完成字节数归一化为 0，把负值保留为失败码，异常返回 `None`。当前框架可以检查 `None`、返回列表长度和每 key 状态，但不能从 wrapper 得到精确 bytes-written，也不能可靠识别“正数但少于预期”的短读。同模型、同 cache 配置下，put/get 都由对应的 `prepare_value` 布局生成，`sum(size_list)==object_size` 是合理预期，但当前接口没有运行时验证。Mooncake C++ 只要求 `dst_total_size >= object total_size`；若跨角色/配置差异使 framework 提供的总容量大于对象实际大小，未覆盖的尾部可能保留旧数据。因此第一阶段必须增加布局不变量断言，或扩展 adapter/result 暴露实际对象大小/完成字节数。completion event/stream 要求也应通过 capability/result adapter 表达，而不是在 dedup helper 中猜测。

Mooncake 没有现成 TP fan-out。应该保留其 backend get，把 owner 结果交给 framework pack/broadcast；不要让 Mooncake backend 自己判断 MLA。SSD/DFS 临时 buffer、replica 延迟、对象大小与目标容量、错误码必须纳入测试；显式 lease 主要属于 GVA/layerwise 等后续路径。

### 6.2 LMCache

LMCache `save_only_first_rank` 已实现 owner/passive：首 rank 读取 storage，广播 chunk count、metadata 和 `uint8` payload，peer 接收后继续 GPU load。可参考 owner/passive、pure MLA guard、TP group、chunk header/payload、统一 status；不能直接复制连续 `MemoryObj`、world size 折叠、`retrieve_layer` 或其 object 生命周期。LMCache 的 chunk/object 粒度不等于固定的 byte budget，也不自动保证长上下文 staging 峰值受控；AscendStore 必须自行定义 `chunk_bytes`、处理超大 entry 和 buffer reuse。LMCache 自己也没有完整支持 layerwise save-only-first-rank。

### 6.3 MemCache/YuanRong

普通同步多 buffer get 可以作为 framework owner-read 的候选 backend，但要分别确认目标写入完成、stream 可见性、timeout 后内容和 partial write。GVA/layerwise 不直接纳入。

### 6.4 vLLM/NIXL 和候选提交

NIXL 的 source-rank/registered-buffer mapping 证明 replicated read-once 语义可行，但它是点到点传输，不等价于 Mooncake multi-buffer 加 owner HBM 加 HCCL。`origin/gil_branch` 的 `552327040` 已做过 owner get、byte view、pack、broadcast、unpack 原型，证明当前 worker 架构可以承载实现；但它缺少 bounded chunk、完整 hybrid/peer-TP guard、plan digest、异常/stream/partial-miss 协议和 NPU 验证，不应直接 cherry-pick。

## 7. 中转 buffer 是否必要

当前路径是 backend 直接写每个 rank 的 KV cache。若使用已有 `GroupCoordinator.broadcast(tensor)`，多 span 目标必须转成连续或可描述 payload。因此当前通用方案实际需要某种中间数据表示，推荐受 byte budget 限制、可复用的 NPU `uint8` staging；不需要整请求一次性复制。

理论替代方案是每个 span 单独 broadcast、新增 arbitrary-span collective，或 backend native fan-out/P2P。它们不属于当前通用框架能力，前两者 collective/实现复杂度高，后者需要 backend 协议和多 rank target registration。第一阶段选择 bounded staging。

---

# 第二部分：可落地方案

## 8. 设计目标和安全不变量

第一阶段目标：pure MLA hit 的 backend read 从 TP 次降到 1；所有 rank 的目标 KV 逐字节等价；backend 不实现 MLA 判定；复用 key schema、scheduler hit 和 invalid/recompute；不满足条件时旧路径零回归；staging 受预算限制；所有阶段可观测、可回滚。

非目标：不改 Mooncake C++、不改 key schema、不第一阶段覆盖 async/layerwise/GVA/hybrid/异构 TP、不承诺 broadcast 一定更快、不在 collective 已开始后 rank-local fallback。

必须保持：每个 canonical entry 最多一次 owner read；peer 不调 backend；所有 rank 使用相同 entry/chunk 顺序和 tensor shape；失败 entry 统一 invalid；guard 关闭时旧路径完全不变；backend completion 到 attention 的 event 顺序明确。

## 9. 复制判定和 owner

建议增加 `enable_mla_read_dedup=False`，同时检查 `use_mla`、`tp_size>1`、`put_step==tp_size`、`peer_tp_size==tp_size`、`tp_mismatch=False`、非 layerwise/async/sparse/hybrid、所有 group 为 replicated MLA、backend capability 声明同步目标写入和可接受的 bytes/object-size 语义。压缩/DSV4 初版建议关闭；未知或未验证 capability 必须回退。

使用 vLLM `get_tp_group()` 的 `GroupCoordinator`；`src=0` 是 local TP group rank 0，不是 global rank 0。不要跨 DP/PP/CP/DCP 广播；key 中已有的 rank/partition 字段继续参与 namespace。后续 hybrid 要增加 `REPLICATED_MLA/SHARDED_KV/STATEFUL_SSM/UNKNOWN` 的 group classifier。

## 10. 新增 framework helper

建议新增 `ascend_store/mla_read_dedup.py`，而不是把协议全部塞入 `start_load_kv`。

`ReadPlanEntry` 至少保存：request index、req id、group id、canonical entry index、key、span sizes、token range、replication class，以及仅本 rank 使用的 local addresses/local block id。

`ReadPlan` 保存 step sequence、request/entry 顺序、总 bytes、`backend_batches` 和广播 chunk 描述以及 canonical digest。backend batch 保证每个 key 的全部 spans 只读一次；广播 chunk 只描述已完成 owner spans 的 byte-budget 切分。digest 包含 request 顺序、key/hash、group/layer 顺序、span 数/size、广播 chunk 边界、replication class；不包含裸地址和 local block id。

`BackendReadCapabilities` 建议至少表达：multi-buffer get、get 是否 blocking、返回时目标是否写完、是否有 per-key status、是否能提供 exact bytes-written/object size、是否支持 device target、是否需要显式 stream sync。短期用 adapter 只为已验证 backend 声明这些能力，未知 backend 默认关闭；长期再考虑把它做成 Backend 的可选接口。

framework 统一 status：`SUCCESS`、`MISSING`、`BACKEND_ERROR`、`SHORT_READ`、`PLAN_ERROR`、`COLLECTIVE_ERROR`；保留 backend 原始码用于诊断。

## 11. 同步第一阶段协议

### 11.1 Step-level preflight

step-level preflight 需要一个当前代码尚未自动保证的调用不变量：所有 TP rank 每个 step 都进入固定次数的 control collective，包括 `metadata.requests` 为空、request 没有可加载块和所有 load group 为空的情况。当前 `start_load_kv` 对空 `metadata.requests` 会直接返回，因此不能仅在其后追加 all-gather 就声称协议安全。实施前必须通过调用方源码和多进程实测证明 scheduler 调用同构；更稳妥的实现是把固定 shape 的 control collective 前移到空请求返回之前。feature/config 也必须跨 TP 一致。

满足该前置条件后，所有 rank 对当前 step 构造 local plan（包含 no-load/no-op 的 eligibility），以固定 shape 交换 eligibility、request/entry count、总 bytes 和 digest。任一 rank 不 eligible 或 digest 不一致，全组统一走旧逐 rank get；旧路径可以继续 circular shift。计划构造异常也必须表现为 preflight 的 ineligible，而不是某个 rank 直接退出。若无法保证所有 rank 进入 preflight，第一阶段不得启用 step-level collective，应继续旧路径并把该不变量列为阻塞验证项。

### 11.2 Owner read 和 status

owner 以 `BackendReadBatch` 为单位只用自己的 key/address/size 调一次 backend；同一 key 的所有 spans 必须在该次读取中提供，不能按广播 chunk 重复调用 backend。严格校验 `None`、返回长度和错误码；当前 Mooncake wrapper 不能提供精确 bytes-written，若需短读判定必须由 adapter 提供，否则先以 `sum(size_list)==object_size` 的不变量约束。异常转为全失败或按 entry 的统一 status，不能在通知 peer 前直接退出。

先 broadcast 固定长度 header 和按 canonical entry index 排列的 status，再广播 payload。header 至少包含 protocol version、step sequence、request/chunk index、entry count、payload bytes、digest 摘要和 flags。不能广播 owner 的 local block id。

### 11.3 Bounded pack/broadcast/unpack

owner backend batch 完成后，owner 才按 canonical 顺序把成功 spans pack 到可复用 NPU `uint8` staging；失败 entry 不进入 payload。随后从已完成的 spans 生成 `BroadcastChunk`，peer 按 status 和自己的 span descriptors unpack。建议 `mla_read_dedup_chunk_bytes` 初始为 64 MiB，必须按 NPU 结果调整；不按整请求分配。一个 entry 大于 budget 时必须完整 backend read 一次，再拆成多个 broadcast chunk，不能把同一 key 作为多个 backend get。

chunk 优先按 logical block/layer group 边界切分；超大单 entry 需要 span split descriptor 或独立大 chunk。零 payload 时所有 rank 按 header 一致跳过，若 HCCL 不支持零长度则统一发送 1-byte dummy。

### 11.4 Invalid block

每个 rank 用自己的 local block id list 和广播的 canonical status 调用现有 `record_failed_blocks`。owner 的 block id 不能广播，因为各 rank allocator 可能不同。成功 entry 才能交给 attention，失败 entry 统一走 recompute。

## 12. Stream、lease 和错误状态机

必须保证：backend get complete -> owner pack 可见 -> payload broadcast complete -> peer unpack complete -> attention/load gate。不能用 Python 返回或日志顺序替代 NPU event/stream 证明。

普通同步 backend 只需保证对象在 `get` 返回前有效；成功返回后数据已复制到 owner HBM，远端对象 eviction 不会影响 owner 的 payload 或 peer 的本地副本，因此不要求 backend lease 覆盖 peer unpack/attention。仍需保证本地 KV cache block 在 unpack/attention 前不被 allocator 重用。显式 GVA/layerwise/MemCache lease 另行设计，可能必须覆盖 batch-copy/write-finish 和 attention gate；不能把普通同步 Mooncake 的结论套到这些路径。

状态机为：eligible -> plan build -> preflight；mismatch 回旧路径，match 进入 owner read -> status broadcast；全失败走 invalid/recompute，部分/成功走 payload broadcast -> peer unpack -> complete。preflight 失败可统一回退；collective 已开始后不能 rank-local fallback，collective error 应标记 dedup unhealthy 并按进程级错误处理。

## 13. 复杂路径的后续设计

Async 需要 `TPReadRendezvous`：主 worker 为 step 分配 sequence，各线程 enqueue 同一 sequence/request/digest，rendezvous 等待同一 TP group 全部到达后才执行 owner read/broadcast；还要处理 cancel、preempt、timeout、线程异常和 event 传递。

Key-based layerwise 可以按 `(physical layer, group, chunk)` 去重，但必须保留 `wait_for_save`、attention gate、layer event、物理 layer reuse 和最后 layer completion。GVA/MemCache 还需定义 owner GVA、peer GVA、lease、batch_copy/write-finish 关系。

Hybrid 必须 group-level 混合新旧路径；TP mismatch 必须重新设计 source/sub-key/strided payload mapping；两者都不是简单开关扩展。

## 14. 性能模型和风险

逻辑 payload 为 B 时，旧路径外部 read 约为 TP*B；新路径外部 read 约为 B，但新增 owner pack、约 (TP-1)*B 的 TP broadcast 和 peer unpack，另有 bounded staging。远端网络/SSD backend 更可能获益，本地高速 memory backend 可能被 HCCL/pack 抵消，必须分项测量。

最高风险是 collective 顺序不一致导致 hang；其次是 staging OOM、owner get 未完成就 pack、对象大小与目标容量不一致导致旧尾部被广播、partial miss 传播错误、hybrid 误复制、PP/DP/CP 选错 owner、本地 block 过早重用、GVA/layerwise lease 提前释放和 backend 失败码差异。缓解措施分别是 preflight 调用不变量和 digest、bounded reusable staging、event dependency、exact-size 不变量/adapter、canonical status、hard guard、local TP group、生命周期测试和 capability adapter。

## 15. 测试与 go/no-go

CPU/mock 必须覆盖 MLA TP2/4/8、TP1、非 MLA、put_step=1、peer TP mismatch、hybrid/sparse/compression、async/layerwise/GVA、multi-layer/group、stride/padding/offset/shared storage、owner 一次/peer 零次、空 metadata 的 preflight、feature 不一致、digest mismatch、全成功/全 miss/partial miss/None/异常/返回列表长度错误、`sum(size_list)` 大于对象大小、adapter 报告短读、zero payload、超大单 entry 拆为多个广播 chunk、chunk reuse 和 local block id 不同。

NPU 至少 TP2/4，条件允许 TP8；比较 baseline/opt-in 每个 rank 的目标 span 和 attention 输出，覆盖 Mooncake memory/NOF/SSD/DISK/DFS/local disk、冷 miss/全 hit/partial hit、16K/64K、多 block/layer、PP/DP/CP group、owner exception、HCCL timeout、OOM 和 stream race。

Go 条件必须全部满足：backend read 从 TP 次降到 1；逐字节正确；无 hang/OOM/stream race/未初始化数据；invalid block 与 baseline 一致；非 MLA/hybrid/TP mismatch 零回归；HBM peak 在预算内；目标 TTFT/P95 不劣于 baseline。只达到调用次数下降不能默认开启。

## 16. 实施拆分、配置和回滚

**Phase 0：**只加 counters，测每 rank get 次数/keys/bytes/backend latency、group/layout、候选 plan 和 HBM/stream；先 TP2/4、16K/64K 确认问题规模。

**Phase 1：**先只 enable 已验证的 Mooncake 同步路径，增加最小 capability/result adapter；新增 `mla_read_dedup.py`，`pool_worker.py` 只负责 guard、plan 调度和 invalid 接入；增加 helper/worker 单测、配置说明和结构化 counters。建议 `enable_mla_read_dedup=false`、chunk bytes 初始 64 MiB。第一阶段不改 key schema、scheduler hit、非 MLA transfer thread 和 layerwise thread。

**Phase 2：**将 capability 正式抽象到 Backend base，逐个验证和 enable MemCache/YuanRong，再做 group classifier、hybrid/compression。**Phase 3：**async rendezvous、layerwise/GVA/lease。**Phase 4：**只有当 HCCL/pack 成本过高，再评估 Mooncake native fan-out/P2P。

回滚只需关闭 feature flag，回到逐 rank `m_store.get`，不改 Pool 数据和 key。collective 已开始后的异常不做 rank-local fallback。

## 17. 实施前必须冻结的问题

1. 所有 TP rank 是否每 step 都调用 `start_load_kv`，包括 metadata 为空；
2. scheduler metadata 的 request 顺序和 load decision 是否跨 TP 一致；
3. Ascend `GroupCoordinator.broadcast` 的 stream/timeout 语义；
4. 各 backend 返回时目标 HBM 是否可被 pack stream 读取；
5. partial failure 是否可能写入部分 span；
6. canonical status 能否复用 `record_failed_blocks`；
7. MTP/physical layer reuse 的真实 span 顺序；
8. staging budget、reuse 和 OOM 行为；
9. graph/fullgraph 对动态 collective 的限制；
10. 普通同步对象有效期、本地 KV block 最早可重用点，以及 GVA/layerwise lease 最晚释放点；
11. `kv_role`、prefill PP partition 对 local TP owner 的影响；
12. compression/Eagle/speculative 是否第一阶段显式关闭；
13. collective error 后的进程恢复和后续禁用；
14. capability 先用 adapter 还是正式加入 Backend base。

## 18. 第一阶段的具体代码落点

建议将第一阶段拆成下面几个可独立评审的组件，而不是在 `start_load_kv` 中继续堆叠逻辑：

| 组件 | 建议位置 | 责任 | 不负责的事情 |
| --- | --- | --- | --- |
| eligibility | `pool_worker.py` 的小型 guard/helper | 判断 pure MLA、TP、group、backend capability | 不判断具体 backend replica |
| plan builder | `mla_read_dedup.py` | 从现有 `prepare_value`/request metadata 生成 canonical entries 和本地地址 | 不执行 backend 或 collective |
| plan agreement | `mla_read_dedup.py` | 计算 digest、固定次数 control collective、决定旧路径/新路径 | 不在 mismatch 后局部回退 |
| owner reader | `mla_read_dedup.py` | owner 调 `Backend.get`、归一化返回码、确认 completion | 不修改 scheduler 状态 |
| span codec | `mla_read_dedup.py` | cache span -> staging、staging -> local span、chunk budget | 不解释 key 命中语义 |
| status adapter | `pool_worker.py` 或 helper | canonical status -> local block id -> `record_failed_blocks` | 不传播 owner 物理 block id |
| capability adapter | `backend/` 或 `base.py` 可选方法 | 声明同步、device target、per-key status、stream 要求 | 不实现 owner/passive |

同步调用形态应接近：

```text
entries = build_read_plan(requests)
decision = agree_plan(entries)
if decision == LEGACY:
    legacy_start_load(entries)
else:
    for batch in plan.backend_batches:
        result = owner_backend_read(batch)  # each key read once; passive ranks do not call backend
        broadcast_header_and_status(result)
        for chunk in make_broadcast_chunks(result.successful_spans):
            broadcast_payload(chunk)
            unpack_and_apply_local_status(chunk)
```

这里的 `legacy_start_load` 应复用当前实现，确保开关关闭或 preflight mismatch 时 key/address 顺序和错误处理不变。`owner_backend_read` 不应在 framework helper 中以 backend 名称分支；Mooncake/MemCache/YuanRong 的差异来自 capability/result adapter。第一阶段只有 Mooncake adapter 被验证并 enable，不代表其他 backend 自动支持。

### 18.1 单测的最小断言

第一阶段 PR 至少应有以下可审计断言：

1. TP=N 的纯 MLA hit 中，owner mock `get.call_count == 1`，每个 passive mock `get.call_count == 0`；
2. owner/peer 使用不同 HBM 地址和 block id 时，pack/unpack 后每个 local span 逐字节相等；
3. canonical key/span-size 不同会让所有 rank 选择 legacy，不会只有一个 rank 进入 payload collective；
4. owner 返回 `None`、短结果或异常时，所有 rank 收到相同 canonical failure status，且 invalid block 集合按各自 local block id 计算；
5. chunk payload 大于 budget 时不会分配整请求 buffer，staging allocation 峰值受上限约束；单个 entry 大于 budget 时 backend 仍只读取该 key 一次，随后产生多个广播 chunk；
6. 在相同输入下关闭开关时，原有 backend 调用次数、circular shift 顺序和结果码完全不变。

### 18.2 关键源码索引

实现前应以这些位置为准重新核对，而不是依赖文档中的推测：

- `vllm_ascend/.../ascend_store/pool_worker.py`：`_init_key_head_config`、`register_kv_caches`、`start_load_kv`、`_load_kv_tp_mismatch`；
- `vllm_ascend/.../ascend_store/metadata.py`：`prepare_value`、`prepare_value_layer`、`infer_tp_mismatch_info`、`ReqMeta`、`AscendConnectorMetadata`；
- `vllm_ascend/.../ascend_store/kv_transfer.py`：`KVCacheStoreRecvingThread`、`KVCacheStoreKeyLayerRecvingThread`、`KVCacheStoreLayerRecvingThread`；
- `vllm_ascend/.../ascend_store/backend/base.py`：统一 backend contract；
- `vllm_ascend/.../ascend_store/backend/mooncake_backend.py`：`register_buffer`、`get`、Mooncake result normalization；
- `vllm_ascend/.../ascend_store/backend/memcache_backend.py`、`yuanrong_backend.py`：其他 backend 的同步/多 buffer 差异；
- `vllm/vllm/distributed/parallel_state.py`：`GroupCoordinator.broadcast` 和 local group rank；
- `LMCache/lmcache/v1/cache_engine.py`：`save_only_first_rank`、`_broadcast_or_receive_memory_objs`、`retrieve_layer` 限制；
- `Mooncake-src/mooncake-store/src/real_client.cpp`、`client_service.cpp`：multi-buffer replica 路径和 future completion。

## 19. 最终判断

当前 `vllm-ascend` 可以实现 **同步纯 MLA 的框架级读去重**。最稳妥的实现是 framework owner/passive + bounded span pack/broadcast/unpack；Mooncake 只是第一个 backend 适配对象，LMCache 提供控制面参考，vLLM/NIXL 提供 replicated/source-rank 语义参考。

当前不能把 async、layerwise、GVA、hybrid、TP mismatch 视为已覆盖；它们需要后续 rendezvous、group classifier、lease 和 capability 设计。第一阶段在正确性、资源和性能验收通过前，保持 opt-in，不应宣称 issue 已完全解决。
