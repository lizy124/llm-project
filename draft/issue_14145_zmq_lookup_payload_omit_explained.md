# vLLM Ascend Issue #14145：ZMQ KV Lookup Payload 裁剪详解

> 这份文档面向对 vLLM、KV cache 和 AscendStore 不熟悉的读者，解释任务为什么出现、代码原来怎么工作、这组提交改了什么、请求在进程之间如何流动，以及当前实现还缺什么验证。
>
> 文档对应的源代码分支是：E:\lizy\code\vllm-project\vllm-ascend-zmq-lookup-payload-omit，分支名为 test/zmq-lookup-payload-omit。本文所在的 llm-project 是知识库，不是 vLLM Ascend 的源代码仓库。

## 先说结论

Issue #14145 的目标很具体：当 scheduler 已经知道 prompt 的前缀 KV cache 在本地 HBM 中命中时，发送给 worker lookup server 的 ZMQ 请求不应该再次携带这段前缀的 block hash，只发送仍然需要查询外部 AscendStore 的后缀 hash。

这能减少长 prompt 的：

1. Python 对 hash 列表的处理量；
2. msgpack 编码和解码量；
3. ZMQ multipart 消息的 payload 大小；
4. worker 侧重复生成和检查已经确定命中的前缀 key 的工作。

这不是简单的列表切片。因为 worker 和 coordinator 仍然需要知道这些后缀 hash 在原始 prompt 中的绝对位置，所以实现同时引入了：

- FULL 和 SUFFIX 两种 lookup hash 模式；
- 一项新的 lookup_hash_mode 配置；
- ZMQ 请求中的 mode frame；
- token_offset，用于把后缀查询重新放回原始 token 坐标；
- HBMCachedBlockHashList，用逻辑 marker 保留 HBM 前缀而不传输真实 hash。

当前分支已经实现了主要代码路径和单元测试，但 Issue 原始要求的性能数据、设计交付件以及真实端到端协议兼容性验证仍然不足。仓库中的严格审查报告因此给出过 REQUEST CHANGES 判断。默认模式仍是 FULL，所以同版本内未显式启用优化时，lookup 算法保持旧语义；但新增 mode frame 本身改变了 wire protocol，不能把“默认 FULL”误认为“新旧进程天然兼容”。

## 1. 先建立几个基本概念

### 1.1 vLLM 在处理什么

用户发送一段 prompt。模型在处理 prompt 时，会为每个 token 计算 attention 所需的 Key/Value 状态，也就是 KV cache。后续生成 token 时，模型可以复用已经计算好的 KV，而不是从第一个 token 重新计算。

如果两个请求共享相同的 prompt 前缀，后一个请求还可以复用前一个请求留下的 prefix cache。这是长上下文推理中很重要的优化。

### 1.2 HBM、外部 KV pool 和 AscendStore

- HBM：High Bandwidth Memory，Ascend NPU 上的高速设备内存。当前 worker 正在运行的模型直接使用这里的 KV cache。
- 外部 KV pool：放不下在本地 HBM 中的 KV，或者需要跨进程、跨节点复用的 KV，可以放到外部存储池。
- AscendStore：vLLM Ascend 使用的外部 KV 存储后端。代码通过 key 的存在性来判断某段 KV 是否在远端可用。

因此，一个请求的 KV 命中可能分成两段：

    token 0 ---------------- token 31 | token 32 -------- token 63
              本地 HBM 命中          |       需要查询远端 AscendStore

本任务只优化第二段查询请求的表达方式，不改变 HBM 中已经命中的事实。

### 1.3 block hash 是什么

vLLM 不会为每个 token 单独发送一条 KV 查询。它把 token 按固定大小分成 block，并为每个 block 计算 hash。可以抽象成：

    block_hashes = [h0, h1, h2, h3]
    hash_block_size = 16 tokens

    h0 -> token [0, 16)
    h1 -> token [16, 32)
    h2 -> token [32, 48)
    h3 -> token [48, 64)

实际代码中的 BlockHash 来自 vLLM 的 vllm.v1.core.kv_cache_utils。hash 是查找 KV 的索引，不是 KV 数据本身。传输 hash 的目的，是让 worker 能构造 AscendStore key 并调用 exists()。

### 1.4 scheduler、worker 和 lookup server

在这个任务里，“scheduler”和“worker”不是同一个职责：

- scheduler 管理请求生命周期，知道当前请求已经在 HBM 中计算到多少 token，并决定要不要向外部 KV pool 查找。
- worker 真正持有 NPU、外部 store client 和 token key 生成逻辑，负责把 hash 转成远端 key 并查询。
- lookup server 是 worker 进程内的一个 ZMQ REP 服务线程。scheduler 进程中的 client 通过 IPC 连接它。

调用链可以理解为：

    请求进入
       |
       v
    vLLM scheduler
       |  request.block_hashes, hbm_hit_tokens
       |  ZMQ REQ / IPC
       v
    LookupKeyClient
       |
       v
    LookupKeyServer
       |
       v
    KVPoolWorker
       |
       +---------------------------+
       |                           |
       v                           v
    Coordinator 路径          legacy fallback 路径
    多 group / mask            生成 key -> m_store.exists()
       |                           |
       +-------------+-------------+
                     v
              返回连续命中 token 数
                     |
                     v
              scheduler 继续调度

### 1.5 hbm_hit_tokens 的含义

hbm_hit_tokens 表示当前请求前缀已经在本地 HBM 中命中的 token 数。例如：

    token_len = 64
    hbm_hit_tokens = 32

意思是 token [0, 32) 已经本地命中，远端 lookup 的有效起点是 token 32。

它不是“远端命中了多少 token”。worker 的 lookup 最终会把本地 HBM 命中和远端命中合并，返回整个连续前缀的命中长度：

    HBM 命中 32 + 远端连续命中 16 = 最终命中 48

## 2. 任务的前因后果

### 2.1 原始 Issue 要解决什么

GitHub Issue 是 vllm-ascend#14145，标题为：

    [Contribution] 任务 #4：[Perf] ZMQ lookup payload 裁剪

原始背景是：scheduler 调用 LookupKeyClient.lookup() 时，把完整的 request.block_hashes 传给 client。worker 后面虽然会根据 hbm_hit_tokens 跳过前缀，但完整前缀已经完成了编码、进程间传输和解码。

对于 16K、64K 甚至更长的 prompt，这会造成不必要的 payload 和编解码开销。Issue 的验收标准要求：

1. 裁剪后命中结果必须与发送完整 hash 时一致；
2. hbm_hit_tokens = 0 和命中整个请求等边界必须正确；
3. 16K/64K 长 prompt 要有 payload 大小和 lookup 延迟数据；
4. 要有设计说明、性能数据和单测；
5. 必须特别说明 offset 语义，保证 scheduler 与 worker 对齐。

### 2.2 为什么不能只写一行切片

表面上可以写成：

    lookup_hashes = request.block_hashes[hbm_hit_tokens // hash_block_size:]

但 worker 原有的很多逻辑默认 hash 列表从 token 0 开始。直接传 [h2, h3] 会产生两个风险：

1. h2 可能被误认为对应 token 0，而不是 token 32；
2. coordinator 根据 hash 列表下标计算命中长度时，会丢失 HBM 前缀的逻辑位置。

所以真正的改动必须同时处理：

    传输层：只发送 suffix hash
    计算层：保留 suffix 的绝对 token offset
    命中层：把 HBM prefix 继续当成已命中的逻辑 block
    协议层：让 server 知道 client 发的是 full 还是 suffix
    兼容层：FULL 仍然保留为默认行为

## 3. 修改前的完整流程

下面用一个 64 token 的请求说明旧逻辑。

### 3.1 scheduler 侧

请求有：

    block_hashes = [h0, h1, h2, h3]
    token_len = 64
    num_computed_tokens = 32

旧代码把完整 request.block_hashes 传入 client.lookup()。

### 3.2 client 编码和发送

旧版请求大致包含：

    frame 0: token_len
    frame 1: kv_cache_group_ids
    frame 2: hbm_hit_tokens
    frame 3...: h0, h1, h2, h3

hash 会先转成十六进制字符串，再经过 msgpack 编码，然后通过 ZMQ multipart message 发送。

### 3.3 worker 侧

worker 收到完整 hash 后，用 hbm_hit_tokens 决定从哪一个 token/block 开始查询：

    已经传过来的 h0、h1：不需要真正访问远端
    真正需要访问远端的：h2、h3

但 h0、h1 已经付出了：

    hash -> hex string -> msgpack -> ZMQ IPC -> msgpack decode

这就是本任务要消除的冗余。

## 4. 修改后的整体设计

### 4.1 两种 lookup 模式

在 vllm_ascend/.../ascend_store/metadata.py:257-261 中新增：

    class LookupHashMode(str, Enum):
        FULL = "full"
        SUFFIX = "suffix"

语义如下：

| 模式 | client 发送的 hash | hbm_hit_tokens 的作用 | 适用含义 |
|---|---|---|---|
| FULL | 完整 hash 列表 | worker 根据它跳过 HBM 前缀 | 原有行为，默认模式 |
| SUFFIX | HBM 前缀之后的 hash | worker 必须把 HBM 前缀补回逻辑命中结果 | 新的 payload 优化模式 |

默认值是 FULL，在 scheduler 初始化处读取：

    vllm_config.kv_transfer_config.kv_connector_extra_config["lookup_hash_mode"]

没有配置时使用 full；配置了非法值会抛出明确的 ValueError。

注意：默认值只说明 lookup 算法的默认语义，不代表新旧版本的 ZMQ frame 布局可以互通。因为当前 client 无论 FULL 还是 SUFFIX 都会发送新的 mode frame。

### 4.2 scheduler 如何裁剪

在 pool_scheduler.py:583-598，SUFFIX 模式执行：

    hbm_cached_hashes = num_computed_tokens // self.hash_block_size
    lookup_block_hashes = request.block_hashes[hbm_cached_hashes:]

例如：

    完整列表： [h0, h1, h2, h3]
    hbm_hit_tokens = 32
    hash_block_size = 16
    切掉 2 个 hash
    发送列表： [h2, h3]

代码同时要求：

    num_computed_tokens % hash_block_size == 0

这是因为当前 SUFFIX 协议按完整 hash block 切分，不能表达只命中半个 hash block 的情况。

### 4.3 新的 ZMQ frame 布局

LookupKeyClient.lookup() 位于 pool_scheduler.py:1191-1214。当前代码把 mode 插在 HBM 命中计数之后：

    frame 0: token_len，4 字节大端整数
    frame 1: kv_cache_group_ids 的 msgpack frame
    frame 2: hbm_hit_tokens，4 字节大端整数
    frame 3: lookup_hash_mode，即 full 或 suffix 的 msgpack frame
    frame 4...: hash 字符串的 msgpack frames

client 的关键步骤是：

    hash_strs = [h.hex() for h in block_hashes]
    hash_frames = self.encoder.encode(hash_strs)
    kv_group_frames = self.encoder.encode(kv_cache_group_ids)
    lookup_mode_frames = self.encoder.encode(lookup_hash_mode.value)

server 在 ascend_store_connector.py:315-330 按相同顺序解码，然后把 mode 继续传给 pool_worker.lookup_scheduler()。

这是一个必须两端同时升级的协议变更。旧 server 会把 full 或 suffix 当成第一个 hash payload；旧 client 发来的第一个 hash又会被新 server 当成 mode 并尝试转换为 LookupHashMode。当前代码没有自动完成跨版本 client/server 协商。

### 4.4 worker fallback 路径：如何恢复绝对位置

当没有使用 coordinator，worker 会走 legacy fallback。lookup_scheduler() 在 SUFFIX 模式下计算：

    lookup_token_len = token_len - hbm_hit_tokens
    token_offset = hbm_hit_tokens

然后调用 _build_lookup_keys()，并传入 token_offset。

_build_lookup_keys() 在 pool_worker.py:2116-2141 中把每个 key 的起止位置加上 token_offset：

    suffix 内部看到的区间： [0, 16), [16, 32)
    加上 offset=32 后：     [32, 48), [48, 64)

这样远端 key 仍然对应原始 prompt 中的正确位置。

fallback 的命中结果还必须把 HBM 前缀保留下来：

- 后缀全部命中：返回 64；
- 后缀第一个命中、第二个 miss：返回 48；
- 后缀第一个就 miss：返回 32；
- 没有任何后缀 hash：只要 HBM 命中 32，仍返回 32。

### 4.5 coordinator 路径：为什么需要 marker list

coordinator 负责更复杂的命中判断，例如：

- 多个 KV cache group 的交集；
- hybrid cache 的不同 block size；
- lookup mask；
- 不同 cache family 的粒度；
- TP/PP rank 扩展后的 key。

如果 SUFFIX 模式只把 [h2, h3] 交给 coordinator，那么 coordinator 会以为列表的第一个元素在 token 0，绝对位置就错了。

因此 coordinator.py:28-77 增加了 HBMCachedBlockHashList。它的逻辑形态是：

    HBMCachedBlockHashList(
        block_hashes=[h2, h3],
        num_hbm_cached_hashes=2,
    )

    逻辑上等价于：
    [_HBM_MARKER, _HBM_MARKER, h2, h3]

但前两个元素不存储真实 h0、h1，只返回一个专用 marker。这样同时满足两点：

1. 列表长度和下标仍然从原始 block 0 开始；
2. 不需要重新传输或重新构造 HBM 前缀的真实 hash。

ExternalCachedBlockPool.get_cached_block() 看到 marker 时，直接把它视为所有 group 都已经存在；看到真实后缀 hash 时，才检查 group 和 hash 的存在集合。

这使 coordinator 能继续用原有的 find_longest_cache_hit() 算法计算“从 token 0 开始的连续命中长度”。

## 5. 用例演示

假设：

    token_len = 64
    hash_block_size = 16
    block_hashes = [h0, h1, h2, h3]
    hbm_hit_tokens = 32

### 5.1 FULL 模式

    发送： [h0, h1, h2, h3]
    worker 逻辑：h0、h1 视为 HBM 已命中，只查询 h2、h3

如果远端 h2 命中、h3 未命中，最终返回：

    32 + 16 = 48 tokens

### 5.2 SUFFIX 模式，后缀全部命中

    发送： [h2, h3]
    逻辑列表： [HBM_MARKER, HBM_MARKER, h2, h3]
    远端：h2、h3 都存在
    最终返回：64 tokens

### 5.3 SUFFIX 模式，第一个后缀就 miss

    发送： [h2, h3]
    HBM 前缀：32 tokens
    远端：h2 miss，h3 即使存在也不能越过 h2 计入连续前缀
    最终返回：32 tokens

### 5.4 SUFFIX 模式，后缀为空

当 hbm_hit_tokens == token_len 时，scheduler 可能发送空的后缀列表。worker 不能把空列表解释成“完全没有命中”，而应该保留 HBM 结果：

    后缀列表：[]
    HBM 命中：64
    最终返回：64

当前 fallback 代码在没有 keys 且 hbm_hit_tokens > 0 时，会构造 hbm_hit_tokens 作为该 group 的命中边界；coordinator 路径则通过 marker list 表达同一语义。

### 5.5 多 group 的交集

如果一个请求有两个 KV cache group：

    group 0: h2、h3 都存在
    group 1: h2 存在、h3 不存在

最终只能认为 h2 对两个 group 都命中，所以结果是：

    HBM 32 + h2 16 = 48 tokens

不能因为 group 0 命中了 h3，就把整体结果算成 64。当前 coordinator 和 fallback 测试都覆盖了这种“按 group 求共同前缀”的场景。

### 5.6 非对齐的 HBM 命中

若：

    hbm_hit_tokens = 24
    hash_block_size = 16

当前 SUFFIX 模式会拒绝该请求，因为 24 不是完整 hash block 的边界。这个限制不是数学上绝对必要，而是当前协议没有定义“半个 block 如何表达”。产品上需要明确选择：

- 在配置或调度层禁止非对齐 SUFFIX；
- 非对齐时自动回退 FULL；
- 或扩展协议以支持更细粒度 offset。

## 6. 代码文件地图

以下路径相对于源代码仓库 vllm-ascend-zmq-lookup-payload-omit。

| 文件 | 职责 | 本任务的变化 |
|---|---|---|
| vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py | KV pool 的类型和 metadata | 定义 LookupHashMode.FULL/SUFFIX |
| .../pool_scheduler.py | scheduler 侧 KV lookup、ZMQ client | 读取配置、裁剪 suffix、编码 mode frame |
| .../ascend_store_connector.py | worker 进程内 lookup server | 解码 mode frame，转发给 worker |
| .../pool_worker.py | 生成外部 key、查询 AscendStore、合并命中 | 支持 offset、FULL/SUFFIX、fallback 和 coordinator 两路 |
| .../coordinator.py | 多 group/hybrid/cache mask 命中协调 | 增加 HBM marker list，保留绝对 hash 索引 |
| tests/ut/distributed/ascend_store/test_pool_scheduler.py | scheduler 单测 | 配置、裁剪、边界、wire frame |
| tests/ut/distributed/ascend_store/test_ascend_store_connector.py | server 单测 | mode 解码和 worker 转发 |
| tests/ut/distributed/ascend_store/test_pool_worker.py | worker 单测 | offset、远端全命中/部分命中/miss、多 group |
| tests/ut/distributed/ascend_store/test_coordinator.py | coordinator 单测 | marker list 和 HBM/远端命中合并 |

### 6.1 scheduler 侧的重要位置

pool_scheduler.py:157-166：从 kv_connector_extra_config 读取 lookup_hash_mode，默认 full。

pool_scheduler.py:583-598：SUFFIX 模式按 num_computed_tokens // hash_block_size 切掉 HBM 前缀 hash，并将 mode 和 hbm_hit_tokens 一起发送。

pool_scheduler.py:1191-1214：LookupKeyClient.lookup() 将 token 长度、group、HBM 命中数、mode 和 hash 编码成 multipart request。

### 6.2 server 侧的重要位置

ascend_store_connector.py:315-339：LookupKeyServer 读取 multipart frames：

    token_len = ...
    kv_group_ids = ...
    hbm_hit_tokens = ...
    lookup_hash_mode = LookupHashMode(...)
    hashes_str = ...

随后调用 pool_worker.lookup_scheduler()，并把计算出的命中 token 数编码为 4 字节 response。

### 6.3 worker 侧的重要位置

pool_worker.py:2116-2141：_build_lookup_keys() 生成 key，同时把 token_offset 加到 start/end。

pool_worker.py:2248-2362：_lookup_with_coordinator()：

1. 校验 HBM 命中范围和 SUFFIX 对齐；
2. SUFFIX 时构造 HBMCachedBlockHashList；
3. 按 group 生成后缀 key；
4. 查询 m_store.exists()；
5. 让 coordinator 计算连续命中长度。

pool_worker.py:2364-2501：lookup_scheduler()：

1. 先尝试 coordinator；
2. coordinator 不适用时走 fallback；
3. SUFFIX 将 token 长度改为后缀长度，并设置绝对 offset；
4. 合并每个 group 的命中位置；
5. 异常时记录错误并返回 0，避免 lookup 线程直接把上层请求卡死。

### 6.4 coordinator 侧的重要位置

coordinator.py:35-77 的 HBMCachedBlockHashList 是本任务最容易被忽略的设计点。它不是普通的 Python list，而是一个支持绝对索引的惰性 Sequence。

coordinator.py:80-107 的 ExternalCachedBlockPool 将 HBM marker 视为所有相关 group 已命中，将真实 suffix hash 的存在性委托给外部 store 结果。

## 7. 单元测试覆盖了什么

当前分支新增了大量测试，主要覆盖以下行为。

### 7.1 scheduler 测试

test_pool_scheduler.py 覆盖：

- 默认 mode 是 FULL；
- 配置 lookup_hash_mode = suffix 后使用 SUFFIX；
- 非法配置会报错；
- HBM 命中为 0 时不丢任何 hash；
- HBM 命中时只传后缀 hash；
- 非 hash block 对齐时拒绝 SUFFIX；
- mode 被编码到预期的 wire frame 位置。

### 7.2 server 测试

test_ascend_store_connector.py 验证 server 能从 frames 解出 suffix，并将 LookupHashMode.SUFFIX 传到 lookup_scheduler()。

### 7.3 coordinator 测试

test_coordinator.py 验证：

- marker prefix 和真实 suffix 组成正确的逻辑列表；
- 正向/负向索引和切片保持合理行为；
- HBM prefix 与远端 hit 能合并；
- 第一个远端 block miss 时，HBM prefix 不会丢失；
- 多 group 只能取共同命中前缀。

### 7.4 worker 测试

test_pool_worker.py 覆盖：

- coordinator 路径只查询 suffix key；
- HBMCachedBlockHashList 保持 HBM marker 和 suffix 的绝对位置；
- _build_lookup_keys() 能正确加入 token_offset；
- fallback 路径的远端全命中、部分命中、首个 miss、空 suffix；
- 多 group 的交集和共同 HBM 边界。

## 8. 当前实现的优点和限制

### 8.1 已经做对的地方

1. 优化开关是显式的：默认 FULL，只有配置 SUFFIX 才改变 payload 形态。
2. scheduler 和 worker 都知道 mode：不是只在 client 侧切片，服务端也能区分 suffix 语义。
3. 绝对 offset 被保留下来：fallback 生成 key 时不会把 suffix 错当成从 token 0 开始。
4. coordinator 使用逻辑 marker：既避免传输 HBM 前缀，又不破坏原有 hash 索引。
5. 多 group 语义有测试：不会把某一个 group 的更长命中错误地当成所有 group 的命中。
6. 边界有专门测试：包括 hbm_hit_tokens = 0、空 suffix、首个 suffix miss 和非对齐输入。

### 8.2 仍然需要谨慎的地方

#### 8.2.1 新旧 client/server 的 wire protocol 不兼容

新 client 总是插入 mode frame，新 server 总是按新位置读取 mode。因此：

    旧 client -> 新 server：旧 hash frame 可能被当成 mode
    新 client -> 旧 server：full/suffix 可能被当成第一个 hash

FULL 默认值只能保持 lookup 算法的旧语义，不能自动保持 frame 布局兼容。真实部署如果存在滚动升级、混合镜像或不同进程版本，需要额外的协议版本、能力协商、双格式解析或明确的升级顺序。

#### 8.2.2 SUFFIX 依赖 hash block 对齐

当前代码在 scheduler 和 worker 两端都使用 assert 检查对齐。生产服务通常不应该把普通运行时输入错误变成未捕获的 AssertionError，因此需要产品层决定：

- 配置启动时校验；
- 非对齐自动回退 FULL；
- 或把“不能使用 SUFFIX”变成显式可观测的降级。

#### 8.2.3 性能证据还没有补齐

Issue 要求 16K/64K prompt 的：

- payload bytes；
- hex/msgpack 编解码耗时；
- ZMQ lookup 延迟；
- 端到端 TTFT；
- Ascend NPU 型号、卡数、TP/CP/PP 配置。

当前提交主要是代码和单测，没有提供这些硬件和端到端数据。因此只能说“理论上减少 payload”，不能据此宣称 TTFT 已经改善。

#### 8.2.4 测试大多不是真实 socket round-trip

部分 wire 测试 mock 了 MsgpackEncoder、MsgpackDecoder 和 socket。它们能验证 frame 组织顺序，但不能完全发现：

- 真实 msgpack multipart 解码差异；
- 空 hash 列表的实际 frame 行为；
- 新旧协议互通问题；
- malformed mode 对 server 线程生命周期的影响。

至少还应有一个 in-process ZMQ REQ/REP 测试，使用真实 codec，并覆盖合法、旧格式、坏 mode、空 suffix。

#### 8.2.5 现有严格审查报告要结合当前源代码阅读

知识库中已有：

    draft/issue_14145_zmq_lookup_payload_omit_strict_review.md

该报告对当时审查的 4 个提交给出 REQUEST CHANGES，重点指出 wire 兼容、性能交付件、协议测试和提交规范等问题。阅读当前源代码时应注意：lookup_scheduler() 当前已经包含“空 key 保留 HBM 命中”和在 group_hits 中插入 HBM 边界的逻辑，说明代码在审查过程中有演进。无论如何，静态代码和 mock 单测都不能替代真实端到端验证。

## 9. 与 vLLM 整体架构的关系

### 9.1 这个任务不改模型 forward

它不改变 Transformer 的 attention、sampling 或模型权重。它位于 vLLM 的 KV transfer / KV pool 控制面：

    模型计算结果和 KV cache 的生成方式：不变
    请求前缀是否已在 HBM：不变
    外部 KV 是否存在：不变
    查询外部 KV 时发送哪些 hash：改变

因此，优化目标是控制面通信开销，而不是改变模型数值结果。

### 9.2 为什么有 KV cache group

简单模型可以把所有 attention KV 看成一种 cache。但 hybrid 模型可能同时存在不同的：

- attention 类型；
- block size；
- 压缩比例；
- cache family；
- layer/state 组织方式。

vLLM 将这些可以用同一套规则管理的 cache 单元划成 group。一个请求的命中结果必须在相关 group 之间取共同前缀，而不是只看某一个 group。

这也是为什么本任务不能只做字符串层面的 payload 裁剪：worker 必须继续按 group、cache family、mask、rank 和绝对 token 位置生成 key。

### 9.3 layerwise 与本任务的边界

本任务的 ZMQ lookup 主要针对非 layerwise 的 scheduler-worker lookup。lookup_scheduler() 和 coordinator 仍然保留 use_layerwise 分支，但当前 vLLM Ascend 对某些 hybrid、多 group、layerwise 组合有明确限制。

不要把“SUFFIX 支持”理解成“所有 layerwise/hybrid 组合都已经支持”。应分别查看：

- use_layerwise 是否开启；
- KV group 是否完整覆盖；
- cache family 和 block size 是否一致；
- coordinator 是否适用于当前 group 布局。

## 10. 推荐的验证顺序

如果要继续把这个任务做成可合入、可部署的版本，建议按以下顺序验证。

### 第一步：静态检查和单测

在 vllm-ascend-zmq-lookup-payload-omit 源码仓库中：

    pytest -q tests/ut/distributed/ascend_store/test_pool_scheduler.py
    pytest -q tests/ut/distributed/ascend_store/test_ascend_store_connector.py
    pytest -q tests/ut/distributed/ascend_store/test_pool_worker.py
    pytest -q tests/ut/distributed/ascend_store/test_coordinator.py

实际运行需要 vLLM Ascend 的依赖环境；仅有知识库仓库不能代替这些依赖。

### 第二步：真实 codec 和 in-process ZMQ

建立一对真实 zmq.REQ/zmq.REP socket，至少测试：

    1. FULL client -> FULL server
    2. SUFFIX client -> SUFFIX server
    3. 空 suffix
    4. 非法 mode
    5. 旧 frame -> 新 server
    6. 新 frame -> 旧 server

明确决定旧格式的行为是接受、拒绝并返回 typed error，还是必须在部署层禁止混合版本。

### 第三步：边界回归

至少覆盖以下矩阵：

| HBM 命中 | 远端后缀 | 期望结果 |
|---:|---|---:|
| 0 | 全部命中 | 完整 token_len |
| 0 | 首个 miss | 0 |
| 中间边界 | 全部命中 | token_len |
| 中间边界 | 首个 suffix miss | hbm_hit_tokens |
| 中间边界 | 部分连续命中 | HBM + 连续 suffix |
| 全长 | 空 suffix | token_len |
| 任意 | 多 group 不一致 | 各 group 共同前缀 |

同时覆盖 non-hybrid、hybrid、不同 effective block size、lookup mask、TP/PP key expansion。

### 第四步：性能基线

在实际 Ascend NPU 上固定记录：

    模型、量化方式、NPU 型号、卡数
    TP/CP/PP/DP 配置
    prompt 长度：16K、64K
    HBM prefix 命中比例：0%、50%、接近 100%
    FULL/SUFFIX 的 hash 数量和字节数
    hex/msgpack 编解码耗时
    ZMQ 往返延迟 p50/p95/p99
    端到端 TTFT

只有当 payload 缩减能在真实 workload 中带来可测的 lookup 或 TTFT 收益时，才有理由把 SUFFIX 作为默认模式或扩大适用范围。

## 11. 读代码的最短路线

对 vLLM 不熟的人可以按这个顺序阅读：

1. 先读本文第 1、2、3 节，理解 HBM prefix、外部 store 和完整 hash 的关系；
2. 看 metadata.py:257-261，理解 FULL/SUFFIX 是什么；
3. 看 pool_scheduler.py:583-598，确认 scheduler 如何切 suffix；
4. 看 pool_scheduler.py:1191-1214，确认 ZMQ frame；
5. 看 ascend_store_connector.py:315-339，确认 server 如何解码；
6. 看 pool_worker.py:2364-2501，先理解 fallback，再看 coordinator 调用；
7. 看 pool_worker.py:2248-2362，理解多 group 和 marker list；
8. 最后阅读 4 个测试文件，将每个边界映射回代码。

可以把整个任务压缩成一句话：

> scheduler 已经知道前 N 个 token 的 KV 在 HBM 中，所以只把第 N 个 token 之后的 hash 发给 worker；worker 必须用绝对 offset 和 HBM marker 把这个“短消息”还原成原来从 token 0 开始的逻辑命中问题。

## 12. 术语速查表

| 术语 | 含义 |
|---|---|
| token | 模型处理的基本文本单元 |
| prompt | 用户输入、需要先做 prefill 的 token 序列 |
| KV cache | attention 复用的 Key/Value 中间状态 |
| prefix cache | 可被后续请求复用的 prompt 前缀 KV |
| HBM | Ascend NPU 的高速设备内存 |
| block | 一段固定大小的 token/KV 管理单元 |
| block hash | 用来索引一段 KV 的 hash |
| hbm_hit_tokens | 本地 HBM 已命中的前缀 token 数 |
| AscendStore | 外部 KV 存储后端 |
| KV pool | 管理外部 KV 存取和查找的组件 |
| KV cache group | 使用同一套 cache 管理规则的一组 cache 单元 |
| layerwise | 按模型 layer 粒度传输或查找 KV |
| coordinator | 处理多 group、mask、cache family 等命中组合的协调器 |
| ZMQ REQ/REP | client/server 请求-响应 socket 模式 |
| multipart frame | 一次 ZMQ 消息中的多个独立字节帧 |
| FULL | 发送完整 hash 列表的旧语义 |
| SUFFIX | 只发送 HBM 前缀之后 hash 的新语义 |
| token_offset | 把 suffix 的局部位置还原到原始 token 坐标的偏移 |
| HBM marker | 代表“这个逻辑 block 已在 HBM”的惰性占位 hash |

## 最终判断

Issue #14145 的核心思路是合理的：对已经在 HBM 命中的前缀，不再通过 ZMQ 重复传输 hash，而是只发送远端真正需要查询的 suffix。当前实现已经把这个思路贯穿 scheduler、协议、worker fallback 和 coordinator 四层，并配套了较多边界单测。

但它目前应被理解为“已完成主要实现的实验性优化分支”，而不是已经由性能和兼容性证据证明完备的生产协议。合入前最重要的工作不是继续扩大代码范围，而是补齐：

1. 新旧 client/server 的协议策略；
2. 非对齐 HBM 命中的降级策略；
3. 真实 msgpack + ZMQ round-trip 测试；
4. 16K/64K 长 prompt 的硬件性能数据；
5. 设计说明和部署升级说明。

