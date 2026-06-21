# KVPoolScheduler 初始化属性作用说明

## 问题

`D:\lzy\project\kv_pool\llm-project\tmp_doc\documents\pool\03_Scheduler端_调度决策.md` 中，`KVPoolScheduler.__init__` 里的这些属性具体有什么作用？为什么需要这些属性？

示例代码：

```python
class KVPoolScheduler:
    def __init__(self, vllm_config, use_layerwise, kv_cache_config=None):
        # 1. 读取模型配置
        self.use_hybrid = self._uses_hybrid_kv_cache(...)  # 是否混合 KV Cache 组
        self.compress_ratios = ...  # 压缩比率（如 DeepSeek V4 的 c4/c128）
        
        # 2. 计算 block 大小
        self.original_block_size = self._infer_group_block_sizes(...)  # 原始 block_size
        cp_scale = self.pcp_size * self.dcp_size  # Context Parallel 缩放
        self.grouped_block_size = [bs * cp_scale for bs in self.original_block_size]
        self.hash_block_size = ...  # hash 用的 block 大小
        self.lcm_block_size = math.lcm(*self.grouped_block_size)  # 最小公倍数
        
        # 3. 计算传输粒度
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        # 例如：DeepSeek V4 压缩组 c1=128, c4=512, c128=16384
        # 最终 granularity = lcm(128, 512, 16384) = 16384
        
        # 4. 创建 LookupKeyClient（ZMQ 客户端，用于查询 Worker 端的池命中情况）
        self.client = LookupKeyClient(vllm_config)
        
        # 5. 状态追踪
        self.load_specs: dict[str, LoadSpec] = {}       # 请求 → 加载规格
        self._request_trackers: dict[str, RequestTracker] = {}  # 请求追踪器
        self._unfinished_requests: dict[str, ...] = {}  # 未完成的请求
```

相关源码：

- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`

## 总体理解

这些 `__init__` 属性可以理解成三类东西：

1. **静态配置**：这个模型的 KV Cache 长什么样、有没有压缩、有没有 hybrid group、block 多大。
2. **调度计算参数**：按什么粒度查池、按什么粒度传输、hash 怎么对齐。
3. **运行时状态账本**：某个 request 命中了多少外部 KV、已经分配了哪些本地 block、哪些请求还没结束、哪些 block 需要延迟释放。

核心原因是：`KVPoolScheduler` 本身不搬 KV Tensor，它负责在 vLLM Scheduler 侧做决策，并把“该加载/该保存什么 KV”的元数据传给 Worker。

一句话概括：

> `KVPoolScheduler.__init__` 里这些属性的作用，是把“模型 KV Cache 的几何结构”和“请求运行时的保存/加载状态”都准备好，这样 Scheduler 后续才能正确回答：这个请求命中了多少外部 KV、要分配多少本地 block、哪些 KV 要加载、哪些 KV 要保存、什么时候可以释放 block。

## 1. `use_layerwise`

```python
self.use_layerwise = use_layerwise
```

表示当前是否使用 **layerwise 模式**，也就是按 layer 分层处理 KV 的模式。

作用：

- 决定 load 是否可以异步。
- 如果 `use_layerwise=True`，当前代码不支持 hybrid KV cache groups。

代码里有类似限制：

```python
if self.use_layerwise and len(self.kv_cache_group_ids) > 1:
    raise NotImplementedError(...)
```

为什么需要：

因为 layerwise 模式和普通 multi-block / group 模式的元数据组织方式不同。Scheduler 必须提前知道当前模式，否则后面构造 `ReqMeta`、load/save 行为会不一致。

## 2. `kv_cache_config`

```python
self.kv_cache_config = kv_cache_config
```

这是 vLLM 给出的 KV Cache 配置。

它里面包含：

- 有几个 KV cache group；
- 每组 block size 是多少；
- 每组是 full attention、sliding window、mamba 还是别的类型；
- hybrid KV cache 是否存在。

例如 `_infer_group_block_sizes()` 会从这里读取每个 group 的 block size。

为什么需要：

普通模型可能只有一组 KV Cache；但 DeepSeek V4 这类模型可能有不同压缩比、不同结构的 KV group。Scheduler 如果不知道 group 信息，就没法判断：

- 查询哪些 group；
- 每个 group 用多大的 block；
- 保存/加载时如何给 Worker 传 block IDs。

## 3. `hf_config` / `compress_ratios` / `use_compress`

```python
hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
self.hf_config = hf_text_config or hf_config
self.compress_ratios = ...
self.use_compress = self.compress_ratios is not None
```

这几个属性用于判断模型是否有 **压缩 KV Cache**。

例如 DeepSeek V4 可能有：

- c1
- c4
- c128

这些不是普通的 block size，而是不同 KV cache family 的压缩比例。

`config_data.py` 里通过 `infer_cache_family_from_ratio()` 把压缩比变成 family 名称：

```python
1 -> c1
4 -> c4
128 -> c128
```

为什么需要：

外部 KV Pool 不能只知道“这是第几个 token 的 KV”，还必须知道这是哪种 KV family。否则 c1/c4/c128 的 KV 会混在一起，key 会冲突，或者查出来的 KV 形态不匹配。

## 4. `use_hybrid`

```python
self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
```

它表示当前模型是否使用 **hybrid KV cache manager**。

判断逻辑大致是：

```python
return len(kv_cache_config.kv_cache_groups) > 1 and any(
    not isinstance(group.kv_cache_spec, FullAttentionSpec)
    for group in kv_cache_config.kv_cache_groups
)
```

也就是说：

- 只有一个 KV group：不是 hybrid；
- 多个 group，但全是普通 full attention：也不一定算 hybrid；
- 多个 group，且存在非 FullAttentionSpec，例如 SlidingWindow / Mamba / 压缩组：算 hybrid。

为什么需要：

hybrid 模型里，不同 KV group 的：

- block size 可能不同；
- 生命周期可能不同；
- 保存粒度可能不同；
- hash 粒度可能不同；
- Worker 侧读取地址方式也可能不同。

所以 Scheduler 必须从一开始就知道是不是 hybrid。

## 5. `kv_cache_group_ids`

```python
self.kv_cache_group_ids = (
    list(range(len(kv_cache_config.kv_cache_groups)))
    if kv_cache_config is not None and self.use_hybrid
    else [0]
)
```

表示 Scheduler 需要处理哪些 KV cache group。

普通模型：

```python
[0]
```

hybrid 模型：

```python
[0, 1, 2, ...]
```

为什么需要：

后面查外部 KV Pool 时，会把 group ids 发给 Worker：

```python
self.client.lookup(
    token_len,
    request.block_hashes,
    self.kv_cache_group_ids,
)
```

也就是说，Scheduler 查询池命中时，不是只问“这个 prompt 命中了吗”，而是问：

> 对这些 KV cache group，这个 prompt 的哪些 chunk 已经在外部池里？

## 6. `kv_cache_group_families`

```python
self.kv_cache_group_families = self._infer_group_families()
```

它记录每个 group 属于什么 cache family，例如：

```text
group 0 -> c1
group 1 -> c4
group 2 -> c128
```

为什么需要：

外部 KV Pool 的 key 里包含：

```python
kv_cache_group_id
cache_role
cache_family
chunk_hash
```

如果没有 `cache_family`，那同一个 token chunk 的 c1/c4/c128 KV 可能会生成相同 key，导致错误复用。

## 7. `need_truncate`

```python
self.need_truncate = self.use_compress
```

这个属性表示某些场景下需要做 token/block 截断处理。

后面 `_infer_swa_blocks()` 里，如果发现 MambaSpec，也会设置：

```python
self.need_truncate = True
```

为什么需要：

压缩 KV、Sliding Window、Mamba 这类缓存不是“完整 prompt 全量 KV 一直保留”的简单模型。某些 group 只保留窗口内 KV，或者保存的是压缩状态，所以保存/加载时可能要截断到有效范围。

## 8. `num_swa_blocks`

```python
self.num_swa_blocks = self._infer_swa_blocks()
```

这个属性记录 sliding window attention 每个 group 需要保留多少 block。

计算逻辑大致是：

```python
cdiv(first_spec.sliding_window, first_spec.block_size) + 1
```

为什么需要：

Sliding Window Attention 不需要保存从第 0 个 token 到当前 token 的全部 KV，只需要窗口范围内的 block。

所以请求结束释放 block 时，会用：

```python
self.get_sw_clipped_blocks(block_ids)
```

避免把已经无效或不需要的窗口外 block 也当成有效 KV 保存。

## 9. `kv_role` / `consumer_is_to_load` / `consumer_is_to_put`

```python
self.kv_role = vllm_config.kv_transfer_config.kv_role
self.consumer_is_to_load = ...
self.consumer_is_to_put = ...
```

这三个是角色控制。

`kv_role` 可能表示当前节点是：

- producer；
- consumer；
- both；
- 或某种 KV transfer 角色。

`consumer_is_to_load` 控制 consumer 是否从 KV Pool 加载。

在 `get_num_new_matched_tokens()` 里：

```python
if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
    return 0, False
```

`consumer_is_to_put` 控制 consumer 是否把 KV 保存回池。

在 `build_connector_meta()` 里：

```python
force_skip_save = self.kv_role == "kv_consumer" and not self.consumer_is_to_put
```

为什么需要：

不是所有节点都既读池又写池。比如某些架构里：

- prefill 节点负责写 KV；
- decode 节点负责读 KV；
- consumer 只消费，不回写；
- producer 只生产，不加载。

这些策略必须在 Scheduler 侧提前决定，否则会错误地构造 load/save 元数据。

## 10. `load_async`

```python
self.load_async = ...
```

表示是否允许异步加载外部 KV。

在 `get_num_new_matched_tokens()` 返回值里：

```python
return need_to_allocate, self.load_async and not self.use_layerwise
```

这个返回值会告诉 vLLM Scheduler：

> 这部分外部命中的 token 是否可以异步加载。

为什么需要：

如果异步加载，Scheduler 可以先分配 block，然后 Worker 后台把外部 KV 搬进本地 KV cache。这样可以减少阻塞。

但 layerwise 模式下当前不走这个异步路径，所以要加 `not self.use_layerwise`。

## 11. `client = LookupKeyClient(vllm_config)`

```python
self.client = LookupKeyClient(vllm_config)
```

这是 Scheduler 侧的 ZMQ 客户端。

它负责向 Worker 侧的 lookup server 查询：

> 某个请求的 prompt，外部 KV Pool 已经命中了多少 token？

`LookupKeyClient.lookup()` 会发送：

- token_len；
- kv_cache_group_ids；
- block_hashes。

为什么需要：

真正知道 KV Pool 里有什么的是 Worker / KV Pool 侧。Scheduler 自己不直接访问外部 KV 存储，所以需要 RPC 查询。

这一步发生在：

```python
get_num_new_matched_tokens()
```

## 12. `load_specs`

```python
self.load_specs: dict[str, LoadSpec] = {}
```

这是 request_id 到 `LoadSpec` 的映射。

`LoadSpec` 定义大致是：

```python
@dataclass
class LoadSpec:
    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
```

含义：

- `vllm_cached_tokens`：vLLM 本地已经算好的 token 数；
- `kvpool_cached_tokens`：外部 KV Pool 命中的 token 数；
- `can_load`：是否已经可以真正执行 load。

创建时：

```python
self.load_specs[request.request_id] = LoadSpec(
    vllm_cached_tokens=num_computed_tokens,
    kvpool_cached_tokens=num_external_hit_tokens,
    can_load=False,
)
```

为什么一开始 `can_load=False`？

因为这时 Scheduler 只是知道“外部池命中了多少 token”，但还没有给这些 token 分配本地 KV block。没有本地 block，就没有地方把外部 KV 加载进来。

等 `update_state_after_alloc()` 分配完 block 后，才会设置：

```python
self.load_specs[request.request_id].can_load = True
```

所以 `load_specs` 是连接这三个阶段的桥：

```text
查池命中 get_num_new_matched_tokens()
    ↓
记录 LoadSpec，但 can_load=False
    ↓
vLLM 分配本地 block
    ↓
update_state_after_alloc() 设置 can_load=True
    ↓
build_connector_meta() 把 load_spec 交给 Worker
```

## 13. `pcp_size` / `dcp_size`

```python
self.pcp_size = ...
self.dcp_size = ...
```

分别是：

- `prefill_context_parallel_size`
- `decode_context_parallel_size`

也就是 prefill/decode 的 Context Parallel 并行规模。

后面计算：

```python
cp_scale = self.pcp_size * self.dcp_size
self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
```

为什么需要：

Context Parallel 会把上下文 token 按并行维度切分。对外部 KV Pool 来说，一个逻辑 chunk 需要和 CP 切分方式对齐。

所以原始 block size 不能直接拿来做外部 KV chunk 粒度，需要乘上 CP scale。

简单理解：

```text
原始 block_size = 单个 CP 分片看到的 block 粒度
grouped_block_size = 外部 KV Pool 需要统一对齐的全局 block 粒度
```

## 14. `mamba_group_ids`

```python
self.mamba_group_ids = self._infer_mamba_groups()
```

记录哪些 KV cache group 是 MambaSpec。

为什么需要：

Mamba / state cache 的生命周期和普通 attention KV 不完全一样。代码里有专门的逻辑保护 Mamba blocks：

```python
touch_sending_mamba_blocks()
```

如果某些 mamba block 正在异步发送到外部 KV store，不能马上释放，否则 Worker 还没存完，block 就被复用了。

## 15. `original_block_size`

```python
self.original_block_size = self._infer_group_block_sizes(vllm_config, kv_cache_config)
```

表示每个 KV cache group 原始的 block size。

普通模型：

```python
[block_size]
```

hybrid 模型：

```python
[group0_block_size, group1_block_size, group2_block_size]
```

为什么需要：

后面构造 `ReqMeta` 时会传给 Worker：

```python
original_block_size=self.original_block_size
```

Worker 需要知道每个 group 原始 block size，才能把 token 范围映射到对应的本地 block IDs 和内存地址。

## 16. `grouped_block_size`

```python
self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
```

这是乘了 CP scale 之后的 block size。

为什么需要：

外部 KV Pool 的 key/hash/传输粒度要按全局上下文对齐，而不是只按单卡/单 CP rank 的局部 block 对齐。

例如：

```text
original_block_size = 128
pcp_size = 2
dcp_size = 1
grouped_block_size = 256
```

也就是说，对外部池来说，一个完整可复用 chunk 至少要覆盖 CP 组合后的逻辑范围。

## 17. `hash_block_size`

```python
requested_hash_block_size = vllm_config.cache_config.hash_block_size
self.hash_block_size = (...) * cp_scale
```

这是用于 block hash 的粒度。

代码还要求：

```python
assert group_block_size % self.hash_block_size == 0
```

为什么需要：

vLLM 原生 `block_hashes` 可能是按较小 block 粒度生成的。但外部 KV Pool 的 cache chunk 可能更大。

例如：

```text
hash_block_size = 128
grouped_block_size = 512
```

那就需要把 4 个 hash block 重新组合成一个 grouped hash。

`config_data.py` 里有：

```python
get_block_hashes(...)
```

它会在 group_block_size 大于 hash_block_size 时，把多个小 hash 重新 hash 成一个大 chunk hash。

## 18. `_block_size`

```python
self._block_size = self.grouped_block_size[0]
```

这是兼容单 group 路径的默认 block size。

为什么需要：

很多旧逻辑默认只有一个 KV group，所以 `_block_size` 保留第一组 grouped block size，方便单组路径使用。

## 19. `lcm_block_size`

```python
self.lcm_block_size = math.lcm(*self.grouped_block_size)
```

这是所有 group block size 的最小公倍数。

为什么需要：

hybrid 模型里不同 group 的 block size 可能不一样。

例如：

```text
group 0 block = 128
group 1 block = 256
group 2 block = 512
```

如果保存/加载粒度不是它们的公倍数，就可能出现：

- group 0 正好切完整 block；
- group 1 切半个 block；
- group 2 还没凑够一个 block。

这样无法安全生成一致的外部 KV chunk。

所以需要 `lcm_block_size` 作为基础对齐单位。

## 20. `cache_transfer_granularity`

```python
self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
```

这是最重要的属性之一。

计算逻辑：

```python
granularities = [self.lcm_block_size]
for group_id in self.kv_cache_group_ids:
    granularities.append(
        get_cache_family_granularity(
            self._get_group_block_size(group_id),
            self._get_group_family(self.kv_cache_group_families, group_id),
        )
    )
return math.lcm(*granularities)
```

`get_cache_family_granularity()` 是：

```python
return block_size * infer_cache_family_ratio(cache_family)
```

所以如果某组是 c128：

```text
block_size = 128
family ratio = 128
granularity = 128 * 128 = 16384
```

为什么需要：

这个属性决定外部 KV Pool 保存/加载的最小安全 token 粒度。

它用于：

1. 查询前截断 token 长度：

```python
token_len = self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
```

2. 判断请求是否太短：

```python
if token_len < self.cache_transfer_granularity:
    return 0, False
```

3. 构造 `ReqMeta` 时决定保存多少 token：

```python
num_tokens_to_save = input_token_len // cache_transfer_granularity * cache_transfer_granularity
```

也就是说，Scheduler 不会随便保存一个 100 token、200 token 的片段，而是尽量保存完整、对齐、所有 group 都能解释的 chunk。

## 21. `_request_trackers`

```python
self._request_trackers: dict[str, RequestTracker] = {}
```

这是 request_id 到 `RequestTracker` 的映射。

`RequestTracker` 大致记录某个请求当前：

- 已调度到多少 token；
- 分配了哪些 block；
- 已经保存了多少 token 到外部池；
- token_ids 是什么。

为什么需要：

vLLM 的一个请求不是一次性完成的。它可能经历：

```text
prefill chunk 1
prefill chunk 2
decode step 1
decode step 2
...
```

Scheduler 每一步都要知道：

- 这个请求之前保存到哪里了；
- 新增了哪些 block；
- 这次是否跨过了新的保存边界；
- 是否需要生成新的 `ReqMeta`。

所以 `_request_trackers` 是跨调度 step 的请求状态。

## 22. `_preempted_req_ids`

```python
self._preempted_req_ids: set[str] = set()
```

记录被 preempt 的请求。

在 `build_connector_meta()` 里，如果请求被抢占，会清理 tracker 和 unfinished 状态。

为什么需要：

被抢占的请求后面可能 resume。恢复时不能简单沿用旧 block 映射，因为 block 可能已经释放/重分配。Scheduler 需要知道它是 resumed request，然后重新构造 `RequestTracker` 和 `ReqMeta`。

## 23. `_discard_partial_chunks`

```python
self._discard_partial_chunks = ...
```

表示是否丢弃不完整 chunk。

默认是 `True`。

如果为 True：

```python
token_len = floor_to_cache_transfer_granularity(prompt_len)
```

也就是说，只处理完整 chunk。

为什么需要：

外部 KV Pool 复用最怕“半个 chunk”。如果保存了不完整 chunk，后续 lookup/hash/block 映射可能变复杂，甚至无法和其他 group 对齐。

所以默认只保存/加载完整对齐 chunk。

## 24. `_unfinished_requests` / `_unfinished_request_ids`

```python
self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
self._unfinished_request_ids: set[str] = set()
```

`update_state_after_alloc()` 里会写入：

```python
self._unfinished_requests[request.request_id] = (request, local_block_ids)
self._unfinished_request_ids.add(request.request_id)
```

为什么需要：

这些属性保存“还没结束的请求”的 request 对象和 block IDs。

后面 `build_connector_meta()` 需要用它们：

```python
request_tuple = self._unfinished_requests.get(request.req_id)
request_real = request_tuple[0]
```

原因是：SchedulerOutput 里的一些数据可能是轻量化的新请求数据，不一定保留完整 `Request` 对象；但构造 KV Pool metadata 时需要：

- `prompt_token_ids`
- `block_hashes`
- `all_token_ids`
- 已分配 block ids

所以要在分配后把这些状态保存起来。

## 25. `_block_pool`

```python
self._block_pool: BlockPool | None = None
```

后面通过：

```python
bind_gpu_block_pool()
```

绑定真实的 GPU block pool。

为什么需要：

异步保存 Mamba blocks 时，不能让 block 太早释放。代码里会：

```python
self._block_pool.touch(...)
```

等 Worker 汇报保存完成后，再：

```python
self._block_pool.free_blocks(...)
```

也就是说，`_block_pool` 是为了控制 block 生命周期，防止异步保存期间 block 被复用。

## 26. `sending_event_id` / `sending_blocks` / `sending_events`

```python
self.sending_event_id = 0
self.sending_blocks: dict[int, list[int]] = {}
self.sending_events: dict[int, int] = {}
```

这些用于异步保存事件追踪，尤其是 Mamba/hybrid 相关 block。

逻辑在：

```python
touch_sending_mamba_blocks()
```

保存时：

- 分配 event_id；
- 记录这个 event 涉及哪些 block；
- touch 这些 block，防止释放。

Worker 完成后通过 `update_connector_output()` 回报。

为什么需要：

分布式场景下，不是一个 Worker 完成就能释放 block。需要等所有相关 Worker 都完成保存后，Scheduler 才能释放这些 block。

## 27. `_expected_worker_count`

```python
self._expected_worker_count = vllm_config.parallel_config.world_size
```

表示需要等待多少 Worker 回报完成。

为什么需要：

如果 world size 是 8，那某个异步保存事件可能要等 8 个 Worker 都完成，才能认为这个 event 完成。

否则某些 rank 还没保存完，Scheduler 就释放 block，会造成数据错误。

## 调用链总结

`KVPoolScheduler` 是一个有状态的调度器，不是一个纯函数。

它要跨多个阶段保存状态：

```text
初始化阶段：
    读取模型结构、KV cache group、压缩比、并行配置
    ↓
调度前：
    用 block hash 查询外部 KV Pool 命中多少 token
    ↓
分配后：
    记录本地 block ids，允许 load
    ↓
构造 metadata：
    把 load/save 信息交给 Worker
    ↓
请求运行中：
    追踪已经保存了多少 token、是否跨过新 chunk 边界
    ↓
请求结束：
    判断 block 能否释放，必要时延迟释放
```

这些属性分别解决的问题：

```text
模型 KV 形态是什么？
    use_hybrid / compress_ratios / kv_cache_group_ids / kv_cache_group_families

每个 cache chunk 应该多大？
    original_block_size / grouped_block_size / hash_block_size / lcm_block_size / cache_transfer_granularity

当前请求能不能从外部池加载？
    client / load_specs / kv_role / consumer_is_to_load / load_async

当前请求要不要保存到外部池？
    _request_trackers / _discard_partial_chunks / consumer_is_to_put

请求跨多个调度 step 怎么追踪？
    _unfinished_requests / _unfinished_request_ids / _preempted_req_ids

异步保存期间 block 怎么防止提前释放？
    _block_pool / sending_event_id / sending_blocks / sending_events / _expected_worker_count
```

## 简短结论

`KVPoolScheduler.__init__` 不是简单保存几个配置，而是在为后续完整的 KV Pool 调度流程建立“坐标系”和“账本”：

- 坐标系：模型有哪些 KV group、每组 block 多大、压缩比是多少、hash 和传输粒度怎么对齐。
- 账本：每个请求命中了多少外部 KV、本地已经算了多少、是否已经分配 block、是否可以加载、已经保存了多少、请求是否结束或被抢占。

没有这些属性，Scheduler 就无法安全地完成外部 KV Cache 的查找、加载、保存和释放控制。
