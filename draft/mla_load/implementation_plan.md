# vLLM-Ascend Issue #14140：MLA KV Pool 读侧去重计划

**最终审核修订说明（2026-08-24）：**本版已按当前 `vllm-ascend`/LMCache/Mooncake 源码重新核对关键事实。文档区分了 Mooncake fabric-memory 与非 fabric-memory 注册路径、wrapper 丢失精确完成字节数的限制、普通同步读取与 GVA/layerwise lease 的不同生命周期，以及 backend read batch 与 broadcast chunk 的边界。方案仍是设计计划，未声称已完成实现或 NPU 验证。

## 0. 文档目的与当前结论

- 本文只做问题论证和实施计划；本轮不修改 `vllm-ascend` / `vllm` 代码。
- 目标代码基线：`vllm-ascend` 的 `upstream/main`，提交 `d85e6714a`（当前工作树分支另有一个与本问题无关的 refactor 提交 `5b199c84b`；相关读写逻辑未改变）。
- `vllm` 参考基线：当前本地 detached HEAD `58d3918e3e`。
- GitHub issue：[#14140](https://github.com/vllm-project/vllm-ascend/issues/14140)，标题为“[Contribution] 任务 #1：[Perf] MLA 读侧去重”，正文抓取时间为 2026-08-24；页面未显示额外评论。

**初步判断：问题成立，但适用条件是“MLA、TP 大于 1、KV Pool 命中且走 AscendStore 非 layerwise 同步读路径”。** 当前实现确实存在重复的后端读取；是否能改善端到端延迟，必须通过 NPU 实测，因为去重会增加一次 TP 内 HCCL broadcast 和本地 pack/unpack。

## 1. Issue 内容复述

Issue 的核心断言是：MLA 写侧在 `put_step` 组内只由一个 TP rank 写入 KV Pool，读侧却由每个 TP rank 独立从 KV Pool 取相同数据；TP 越大，重复 I/O 越多。任务要求只取一次，再在 TP 内分发，并覆盖：

1. 最终各 rank 的数据逐字节一致；
2. TP mismatch 路径不被破坏；
3. MLA / 非 MLA 不回归；
4. TP=2/4/8、16K/64K 长上下文下给出调用次数和性能数据。

## 2. 代码证据：为什么说问题成立

### 2.1 写侧已经按 `put_step` 去重

文件：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

- `_init_key_head_config`：MLA 将 `num_kv_head` 固定为 1；当 `tp_size > 1` 时得到 `put_step = tp_size`、`head_or_tp_rank = tp_rank // tp_size = 0`。
- `_process_save_for_layer_batch` 和 `_alloc_gvas_for_save`：都有 `if self.tp_rank % self.put_step != 0: return`，同一 `put_step` 组的非首 rank 不写 Pool。
- 同步写线程 `KVCacheStoreSendingThread` 也将 `put_step` 传入，并按同一语义切分/跳过写入。

因此，纯 MLA、同一 TP 组内的 rank 共享同一个 Pool key namespace；这不是推测，而是当前 key/rank 计算的直接结果。

### 2.2 同步读侧没有对应去重

`KVPoolWorker.start_load_kv` 的非 layerwise、非 async 分支会对每个请求：

1. 构造 `key_list`、`addr_list`、`size_list`；
2. 根据本 rank 做 circular shift；
3. 直接执行 `self.m_store.get(key_list_c, addr_list_c, size_list_c)`。

MLA 的各 rank 由于 `head_or_tp_rank` 都是 0，key 内容相同；`prepare_value` 只把目标地址换成本 rank 的本地 KV cache。因此 TP=8 时，后端会收到 8 次相同数据读取，每次写入一个等价的本地 cache。

### 2.3 还有两个读实现需要单独处理

- `load_async=True`：`KVCacheStoreRecvingThread._handle_request` 中每个 rank 都调用 `m_store.get`。
- `use_layerwise=True`：key-based `KVCacheStoreKeyLayerRecvingThread` 每个 rank 都调用 `m_store.get`；MemCache/GVA 路径则是每个 rank 自己做 `batch_copy`。它们不是 `start_load_kv` 的同一同步调用栈，不能用同步方案直接覆盖。

### 2.4 TP mismatch 不能只看 `self.tp_mismatch`

`metadata.py::infer_tp_mismatch_info` 对 MLA 使用 `not use_mla` 作为启用条件，所以 MLA 的不同 TP 配置目前会保留 `tp_mismatch=False`，但 `peer_tp_size` 仍可能与本地不同。初版去重必须额外要求 `peer_tp_size == tp_size`，否则会把“尚未证明的异构 TP 语义”误认为普通复制场景。

### 2.5 地址和数据布局证据

`metadata.py::prepare_value` 对每个 Pool key 返回所有本地 layer/cache entry 的地址和大小。也就是说，一个 token/block 的 `size_list` 不是单一小 tensor，而是跨层的多个 span；长上下文一次性拼成 payload 会有很高的额外显存峰值。

## 3. 业界/上游对照

### 3.1 DeepSeek MLA 的语义

- 论文：[DeepSeek-V2, arXiv:2405.04434](https://arxiv.org/abs/2405.04434)。MLA 用每个 token 的 compressed latent KV 表示历史，decode 路径具有 MQA 式的数据访问特征。
- vLLM 实现：[mla_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py) 的文档和初始化均明确 `num_kv_heads = 1`。这支持“MLA latent KV 在 TP 间是复制数据，而不是按普通 KV head 分片”的判断。

### 3.2 vLLM 已有同类“复制 KV 只读一次”抽象

- [kv_connector/utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/utils.py)：`TransferTopology.replicates_kv_cache()` 明确将 MLA 和 `num_kv_heads < tp_size` 视为 replicated KV。
- [nixl/tp_mapping.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py)：MLA 映射只选择一个远端 source rank；非 MLA 则按 head/TP 映射多个 source rank。
- [nixl/pull_worker.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py)：注释和实现均说明“Pure MLA reads once because its cache is replicated”；混合 MLA+SSM 仍需为 SSM 的 source ranks 分别读取。
- [parallel_state.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py)：`GroupCoordinator.broadcast` 是现有 TP 内设备 tensor broadcast 封装，可作为通信接口参考。

这说明业界已有相同的**语义解决方案**：先识别 replicated region，再选择一个 source；但上游 NIXL 是点到点内存读映射，不等价于 AscendStore 的“Pool -> owner HBM -> TP broadcast”，所以不能直接照搬代码。

### 3.3 本地未合入候选提交

`origin/gil_branch` 上的 `552327040`（2026-08-12，标题 `Implement MLA read dedup for KV pool load`）已经尝试：rank 0 调 `m_store.get`，把地址 span pack 成 `uint8` payload，broadcast 后由 peer unpack，并添加同步路径单测。

该提交可复用的方向：复用 `kv_caches` storage 建立 byte view、owner/peer 分支、广播返回码。

该提交不能直接合入的原因：

1. feature flag 默认关闭，不能证明 issue 已解决；
2. 只覆盖非 layerwise、非 async、非 sparse 的同步路径；
3. 一次性分配整个请求的 payload。按 MLA 每层 latent KV、全层数和 64K token 估算，额外 HBM 可能达到数百 MB 至数 GB；
4. 没有按 KV group 识别 replicated region，MLA+SSM/hybrid 可能错误复制 SSM 状态；
5. 只以 `tp_mismatch` 作为 guard，没有排除 `peer_tp_size != tp_size`；
6. 对 backend 返回长度、collective 顺序、owner 读取异常后的数据有效性缺少完整协议校验。

### 3.4 LMCache 的现成解法

本地浅克隆的 LMCache `dev` HEAD 为 `f9addd2`。它已经实现了与本问题相同的 owner/passive 模式，核心开关是 `save_only_first_rank`：

- `lmcache/v1/cache_engine.py` 在 MLA 下默认启用 `save_only_first_rank`（可由 extra config 覆盖）；非首 rank 的 `store` 直接跳过，普通 `retrieve` 中也跳过本地 `storage_manager` 读取。
- 首 rank 执行一次 `storage_manager.batched_get`（同步或 async 处理后的统一入口），然后通过 vLLM 的 TP group 广播 chunk 数、每个 chunk 的 metadata 和 `uint8` tensor；其他 rank 分配本地 tensor 接收后，再走同一个 `batched_to_gpu`。
- `lmcache/v1/token_database.py` 在该模式下把 key 的逻辑 `world_size` 折叠为 1；`lmcache/integration/vllm/lmcache_mp_connector_0180.py` 也明确按 `world_size // tp_size`、`rank // tp_size` 消除 MLA 中 TP 对 KV 布局的影响。
- `lmcache/integration/vllm/utils.py::mla_only` 在多进程 connector 路径明确要求模型是纯 MLA；发现 hybrid/SSM 后该 connector 不启用这一复制语义。需要注意，核心 `LMCacheEngine.save_only_first_rank` 主要依据 `metadata.use_mla`，所以不能把 connector 级 guard 直接当成所有集成路径的全局保证。

因此，LMCache 证明了这个问题存在成熟的语义解法：后端读取次数从 TP 次降为 1，代价是 TP 内 broadcast。它与 AscendStore 候选提交的“整批地址 span 打包”不同，LMCache 按 memory object/chunk 粒度广播；但这不等于 LMCache 已经提供固定 byte budget 或长上下文 staging 峰值上限，AscendStore 仍需自己定义 `chunk_bytes` 并复用 buffer。

LMCache 也暴露了需要保留的边界：`retrieve_layer` 路径仍有显式的“暂不支持 save_only_first_rank”注释并逐层访问 storage；首 rank 目前固定为 `metadata.first_rank = 0`，PP/多节点/DP 组合必须确认它和 local TP rank 0 的关系；multi-process connector 对 MLA+PP 也有额外限制。因此不能直接复制 LMCache 代码，但应把其 owner/passive、TP-group broadcast、pure-MLA guard 和 chunk 级传输作为 AscendStore 第一阶段的主要参考。

## 4. 建议的实施边界

### 4.1 第一阶段：只做可证明、安全的同步纯 MLA

初版只在以下条件全部满足时启用：

- `use_mla=True`，`tp_size > 1`，`put_step == tp_size`；
- `peer_tp_size == tp_size`；
- `use_layerwise=False`、`load_async=False`、`use_sparse=False`、`use_hybrid=False`；
- 只处理确认属于 replicated MLA KV group 的请求；
- 使用标准 local TP group，source 为 TP-local rank 0。
- backend 具备已验证的同步完成、device target 和 per-key status adapter；第一阶段先只 enable Mooncake，其他 backend 逐个验证后再打开。

默认采用 `kv_connector_extra_config` 的 opt-in 开关，例如 `enable_mla_read_dedup=False`，先完成线上验证再讨论默认打开。非满足条件继续走原始 `m_store.get`，不改变 key schema 或非 MLA 行为。复制判定和 fan-out 位于框架层，但这不表示所有 backend 默认支持；未知 capability 必须回退。

### 4.2 读取协议

建议新增一个小型、可单测的 read-plan/helper，而不是把逻辑全部塞进 `start_load_kv`。必须把 backend 读取和后续广播切块分成两个层次：

1. **preflight 前置条件：**所有 TP rank 必须在每个 step 进入固定次数的 control collective，即使本 rank 的 `metadata.requests` 为空或没有可加载请求；feature/config 也必须在 TP 内一致。当前 `start_load_kv` 对空 requests 会提前返回，这个不变量尚未由源码自动保证，实施前必须通过调用方源码/实测证明，或先调整入口。若无法证明，不能直接加入 step-level collective。
2. **canonical plan：**各 rank 按相同请求顺序构造 `(key, spans, sizes, block_id)` 计划，并以 eligibility、entry 数、总字节数、span 数和 digest 做固定形状 preflight。任一 rank 不 eligible 或 digest 不一致时，全组统一走旧路径；不能由单个 rank 静默跳过 collective。初版不使用按 rank 的 circular shift，避免 broadcast 的源/目标顺序不一致。
3. **BackendReadBatch：**将每个 key/entry 的全部目标 spans 放入确定的 backend read batch；每个 key 在 owner 上只调用一次 `get`。这里的 batch 边界与后续广播 chunk 无关，不能因为 payload 超过预算而对同一 key 重复调用 backend。
4. **完成状态：**owner 调一次 backend `get`，检查返回是否为 `None`、返回列表长度及每 key 状态。当前 Mooncake wrapper 把正的实际完成字节数归一化为 0，因此不能仅靠当前返回值证明精确 bytes-written 或“短读”；若需要该校验，必须扩展 adapter/result，或先证明 `sum(size_list)` 与对象大小严格相等。Mooncake 只要求目标总容量不小于对象大小，若 size 总和更大而尾部未写入，pack 可能带入旧数据。
5. **BroadcastChunk：**backend read 全部完成后，才从 owner KV cache 的成功 spans 按固定上限（建议可配置，初始 64 MiB，需 NPU 实测）切分广播 payload。使用复用的 `torch.uint8` staging buffer，禁止按整个请求一次性分配；一个大 entry 只能“完整 backend read 一次、随后拆成多个 broadcast chunk”，不能把同一 key 分成多次 backend get。
6. 广播逐 entry/逐 block 的失败码，所有 rank 使用与原逻辑相同的 invalid-block/recompute 处理；失败 span 不得被当成有效 KV 使用。
7. backend `get` 完成后必须确认目标 HBM 写入对 pack 可见；需根据各 backend 实际 API 确认是否需要 NPU stream/event 同步。
8. 空请求、零字节请求、部分 backend miss、跨多个 cache group 都必须由 preflight 统一决定协议路径，并让所有 rank 调用相同次数的 collective，不能出现 rank 分歧导致 HCCL hang。

### 4.3 后续阶段（不阻塞第一阶段）

- **Async**：不能直接在每个接收线程里无序调用 collective。需要按 step/request sequence 建立 TP rendezvous，或由主 worker 统一调度 owner/peer 任务；在此之前保持原始 async 路径。
- **Layerwise key path**：可按 layer/block 做 owner read + broadcast，但必须保留 `wait_for_save`、attention start gate 和 layer finished event 语义。
- **Layerwise GVA/MemCache path**：需要先决定 lease/GVA 的 owner 语义，再设计跨 rank 分发；不能简单把 `batch_copy` 替换成普通 `m_store.get`。普通同步 backend 的对象在 `get` 返回前必须有效，成功复制到 owner HBM 后通常不需要让 backend lease 覆盖 peer unpack/attention；跨 layer 的 GVA/lease 则可能需要覆盖到写完成和 attention gate。
- **MLA+SSM/hybrid**：应按 group/spec 分类，只对 replicated MLA group 去重，SSM/非复制 group 仍按 source-rank 映射读取；这与上游 NIXL 的 hybrid 处理一致。
- **异构 TP**：先保留原路径，待有明确的本地 TP group 与 Pool key 映射证明后再支持。

## 5. 代码改动候选范围

第一阶段预计涉及：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
  - 复制场景判定；
  - load entry 构建复用；
  - bounded pack/broadcast/unpack；
  - backend 返回码和 invalid block 处理复用。
- 可能新增 `.../ascend_store/mla_read_dedup.py`，承载地址 span、read plan 和协议校验，降低 worker 复杂度。
- `tests/ut/distributed/ascend_store/test_pool_worker.py`、必要时新增 helper 单测。
- `backend/mooncake_backend.py` 或独立 adapter：声明同步完成/stream/status 能力；若无法先证明 exact-size 不变量，则扩展 Mooncake adapter 保留原始 bytes-written/object size，而不是继续只返回归一化的 0。
- 配置/文档：说明 opt-in 开关、限制条件、回退方式和监控字段。

第一阶段不改：`metadata.py` 的 key 格式、scheduler hit 判定、非 MLA transfer thread、layerwise 线程。backend 公共 `get` 接口是否保持不变取决于 exact-size 验证：若能证明 `sum(size_list)==object_size`，可用 Mooncake 专用 capability adapter 而不改公共接口；否则必须扩展 Mooncake result/adapter，不能牺牲数据完整性来维持接口不变。

## 6. 测试计划

### 6.1 单元测试（CPU/mocked dependencies）

1. 判定矩阵：MLA TP2/4/8 命中；TP1；非 MLA；`put_step=1`；`peer_tp_size != tp_size`；sparse/hybrid/layerwise/async；开关关闭。
2. 地址 view：跨层多个 storage、非零 offset、不同 span 大小、stride/padding；pack -> unpack 后逐字节相等。
3. owner/peer：owner 只调用一次 backend；peer 不调用 backend；所有 rank 得到相同 status。
4. 空/部分 miss/`None`/错误长度：不死循环、不越界，并生成正确 invalid block 集合。
5. collective 协议：先完成固定次数的 control preflight；entry 数或 payload 大小不一致时在 payload collective 前由全组统一选择 legacy；验证不会让单个 rank 静默跳过 broadcast。
6. 保留并运行现有 `test_pool_worker.py`、`test_kv_transfer.py`、backend 测试，确认非 MLA 调用次数和地址顺序不变。

### 6.2 NPU 多进程正确性

- TP=2、4（具备条件时 TP=8），使用实际 Ascend TP group 和 fake/in-memory backend；owner 写入确定字节模式，比较每个 rank 的 KV cache 目标 span 与 baseline 逐字节一致。
- 覆盖同步 hit、partial miss、空 load、多个 block、多个 layer、PP=1；再单独验证 CP/DP 不会错误跨 group broadcast。
- 记录 HCCL hang、NPU OOM、stream race；任何一个 rank 进入/退出 collective 不一致都视为阻断问题。

### 6.3 回归与降级

- MLA 开关关闭时结果与当前 main 完全一致；
- 非 MLA、TP mismatch、layerwise、async、sparse/hybrid 仍走旧路径；
- backend 异常后 scheduler 能收到原有 invalid block 并触发 recompute；
- 进程重启、连续请求和 prefix-cache 命中/未命中不残留上一请求 payload。

## 7. 性能验证方案

每组测试同时跑 baseline（开关关闭）和 opt-in（开关打开），至少记录：

- 配置：NPU 型号、卡数、vLLM/vLLM-Ascend commit、backend、TP/CP/PP、block size、dtype、模型；
- TP=2/4/8；prompt 16K/64K；冷 miss、全 hit、部分 hit；并发 1/中高并发；
- `m_store.get` 调用次数、keys 数、backend 读取字节数；
- TP broadcast 字节数、pack/unpack 时间、HCCL 时间、额外峰值 HBM；
- TTFT、TPOT/吞吐、P50/P95/P99、错误/重算率。

预期的可验证不变量：纯 MLA hit 时 backend `get` 从 TP 次降为 1，外部 Pool 读取字节从 TP 倍降为 1 倍；新增的 TP broadcast 字节约为 `(TP-1) * payload_bytes`。端到端是否变快不能预先承诺，必须以实测决定默认开关。

建议的 go/no-go：逐字节正确、无 collective hang/OOM、backend get 去重达到预期；非 MLA 回归为零；在目标场景 TTFT/TPOT 至少不劣于 baseline，且长上下文的额外峰值 HBM 在设定上限内。

## 8. 风险与回滚

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 各 rank 请求序列不一致 | HCCL hang | 只在已证明同步场景启用；增加一致性协议和超时/错误日志；异步路径先不改 |
| 整批 payload 额外占用 HBM | 64K OOM | bounded staging buffer、按 byte budget 分块、复用 buffer |
| backend get 未完成就 pack | 广播旧数据 | 明确 backend/NPU stream 完成语义，必要时 event synchronize |
| partial miss 传播不正确 | 错误 KV 或错误重算 | 广播逐 entry status，沿用 `record_failed_blocks` |
| MLA+SSM 误复制 | 结果错误 | 初版排除 hybrid；后续按 group 做 replication classifier |
| 不同 TP/PP/CP 组误用 source=0 | 数据错位 | 只使用 local TP group；校验 group world size/rank；异构 TP 初版回退 |
| HCCL broadcast 成本超过 Pool I/O 节省 | 性能倒退 | 记录分项耗时；保留 opt-in 开关和快速回退 |

回滚方式：将 `enable_mla_read_dedup` 设为 `False` 即回到现有 `m_store.get` 路径；不需要清理 Pool 数据，也不需要变更 key 版本。

## 9. 交付物和评审门槛

1. 设计说明：复制判定、collective 协议、内存上限、异常/回退、为何不覆盖 async/layerwise/hybrid。
2. 实现 PR：仅包含第一阶段范围，带配置说明和结构化 debug/perf counters。
3. 单测和 NPU 多进程正确性报告。
4. TP=2/4/8、16K/64K 的 baseline/opt-in 性能表和原始日志。
5. 评审前必须明确：是否接受初版只覆盖同步纯 MLA、是否默认关闭 feature flag、允许的额外 HBM 上限、目标 NPU/模型及性能门槛。

## 10. 建议执行顺序

1. 先在当前 main 上实现只读的 baseline counters，并跑 TP=2/4 的同步纯 MLA，确认重复 get 和实际 byte layout。
2. 评审并冻结复制判定与 bounded broadcast 协议。
3. 实现 opt-in 同步路径和 CPU/mock 单测。
4. 在 NPU TP=2/4 上做逐字节和故障注入验证，再扩展 TP=8。
5. 输出 16K/64K 性能结果；根据数据决定默认开关和是否进入 async/layerwise 第二阶段。
