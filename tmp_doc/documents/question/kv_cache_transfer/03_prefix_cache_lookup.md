# 03. 本地 prefix cache 如何命中？

源码位置：

- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/kv_cache_utils.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`

本文梳理 vLLM V1 中本地 prefix cache 的命中链路：`Request.block_hashes` 如何生成，Scheduler 什么时候调用 `KVCacheManager.get_computed_blocks()`，命中结果如何变成 `num_computed_tokens`，以及为什么 prompt 全命中时仍然要保留最后 token 的 forward。

---

## 1. 一句话回答

本地 prefix cache 命中发生在 Scheduler 为 `WAITING` 请求首次安排本地 prefill 之前；如果 connector 选择异步加载远端 KV，这次调度会先分配/登记 blocks 并把请求放到 `WAITING_FOR_REMOTE_KVS`，收包完成后在后续调度收尾逻辑中回到 `WAITING` 继续。

核心链路是：

```text
EngineCore 初始化 request_block_hasher
  → Request 创建时生成 full-block block_hashes
  → 可复用的 full KV block 被登记到 BlockPool 的 hash 表
  → Scheduler.schedule() 调度 WAITING 请求
  → KVCacheManager.get_computed_blocks(request)
  → KVCacheCoordinator.find_longest_cache_hit(...)
  → BlockPool.get_cached_block(block_hash, group_ids)
  → 返回 new_computed_blocks + num_new_local_computed_tokens
  → Scheduler 用命中 token 数减少本轮 num_new_tokens
  → allocate_slots() 复用命中的 KV blocks，并为剩余 token 分配新 blocks
  → SchedulerOutput 把 block_ids / num_computed_tokens 发给 Worker
```

所以：

```text
prefix cache 不是 Worker 在 forward 前临时查；
而是 Scheduler 在分配 KV blocks 前先查本地 BlockPool 的 block hash 表。
```

---

## 2. 本文要回答的问题

```text
Scheduler 什么时候查本地 prefix cache？
get_computed_blocks() 返回什么？
block hash 从哪里来？
本地命中如何影响 num_new_tokens？
new_computed_blocks 后续怎么被复用？
为什么 full hit 不能总是直接 decode？
本地 prefix cache 和 external KV cache 如何衔接？
Worker 侧看到的状态是什么？
```

---

## 3. 最小主链路

先看最小路径：

```text
Request.block_hashes
  → KVCacheManager.get_computed_blocks(request)
      → max_cache_hit_length = request.num_tokens - 1
      → coordinator.find_longest_cache_hit(block_hashes, max_cache_hit_length)
          → manager.find_longest_cache_hit(...)
              → block_pool.get_cached_block(block_hash, kv_cache_group_ids)
      → KVCacheBlocks + num_new_local_computed_tokens
  → Scheduler 计算 num_computed_tokens
  → num_new_tokens = request.num_tokens - num_computed_tokens
  → KVCacheManager.allocate_slots(...)
  → request.status 更新
  → request.num_computed_tokens = num_computed_tokens
  → SchedulerOutput.scheduled_new_reqs / scheduled_cached_reqs
```

可以压缩成一句：

```text
block hash 找到已缓存 full blocks；命中的 block 数换算成已计算 token 数；Scheduler 只调度剩余未计算 token。
```

---

## 4. block hash 从哪里来

### 4.1 EngineCore 创建 request_block_hasher

`EngineCore` 初始化时，如果启用了 prefix caching，或者配置了 KV connector，就会创建 `request_block_hasher`。

关键代码位置：`code/vllm/vllm/v1/engine/core.py:210`

```python
self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)

    self.request_block_hasher = get_request_block_hasher(
        hash_block_size, caching_hash_fn
    )
```

位置：`code/vllm/vllm/v1/engine/core.py:210` 到 `code/vllm/vllm/v1/engine/core.py:219`

含义：

```text
prefix cache / KV connector 需要按 token block 计算 hash；
EngineCore 先根据 hash_block_size 和 hash 算法构造一个 block_hasher；
后续每个 Request 都带着这个 hasher。
```

### 4.2 Request 创建时注入 block_hasher

新请求进入 EngineCore 后，会转成 V1 内部 `Request`：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`code/vllm/vllm/v1/engine/core.py:867`

`Request.from_engine_core_request()` 会把 `block_hasher` 传给 `Request.__init__()`。

位置：`code/vllm/vllm/v1/request.py:198` 到 `code/vllm/vllm/v1/request.py:217`

### 4.3 Request 内部保存 block_hashes

`Request` 里有两个关键字段：

```python
self.block_hashes: list[BlockHash] = []
self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
self.update_block_hashes()
```

位置：`code/vllm/vllm/v1/request.py:179` 到 `code/vllm/vllm/v1/request.py:184`

`update_block_hashes()` 只负责增量补齐新产生的 full block hash：

```python
def update_block_hashes(self) -> None:
    """Compute block hashes for any new full blocks and append them."""
    if self._block_hasher is not None:
        self.block_hashes.extend(self._block_hasher(self))
```

位置：`code/vllm/vllm/v1/request.py:237` 到 `code/vllm/vllm/v1/request.py:240`

注意：

```text
block_hashes 只覆盖“完整 block”；
最后不足一个 block 的 tail token 不会生成 block hash，也不能被本地 prefix cache 直接命中。
```

---

## 5. block hash 如何计算

### 5.1 get_request_block_hasher

block hasher 定义在：`code/vllm/vllm/v1/core/kv_cache_utils.py:659`

核心逻辑：

```python
start_token_idx = len(request.block_hashes) * block_size
num_tokens = request.num_tokens

if start_token_idx + block_size > num_tokens:
    return []
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:667` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:673`

含义：

```text
从当前已有 block_hashes 后面开始；
只有新凑满一个 hash_block_size 的 token block，才继续计算 hash；
没有新 full block 就返回空列表。
```

### 5.2 hash 是链式的

每个 block 的 hash 不只包含当前 block token ids，还包含前一个 block 的 hash。

```python
prev_block_hash_value = (
    request.block_hashes[-1] if request.block_hashes else None
)
...
block_hash = hash_block_tokens(
    caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:683` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:702`

`hash_block_tokens()` 中：

```python
return BlockHash(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:563` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:590`

这意味着：

```text
第 N 个 block hash 依赖第 N-1 个 block hash；
所以 block_hashes 是 prefix chain；
一旦某个 block miss，后续 block 即使 token 内容局部相同，也不能作为同一个 prefix 命中。
```

### 5.3 extra_keys 会参与 hash

如果请求包含多模态、LoRA、cache_salt 或 prompt_embeds，hash 会加入额外 key。

相关函数：`generate_block_hash_extra_keys()`

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:525` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:560`

额外 key 来源：

```text
多模态输入：mm identifier + 在 block 内的 offset
LoRA：LoRA name
cache_salt：只加在第一个 block
prompt_embeds：当前 block embedding 数据的 sha256
```

所以两个请求要命中同一个本地 prefix cache，不只是 token ids 一样，还要满足：

```text
相同 token prefix；
相同 hash 链；
相同相关多模态位置 / LoRA / cache_salt / prompt_embeds key；
相同 KV cache group 语义。
```

---

## 6. 已计算 block 什么时候进入 prefix cache

本地 prefix cache 的查找依赖 `BlockPool.cached_block_hash_to_block`。

`BlockPool` 初始化时会创建这个 hash 表：

```python
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

位置：`code/vllm/vllm/v1/core/block_pool.py:170` 到 `code/vllm/vllm/v1/core/block_pool.py:171`

当 `KVCacheManager.allocate_slots()` 或 connector 收包完成后调用 `cache_blocks()` 时，会把当前请求中已经拥有 KV、可复用的 full blocks 继续下钻到 `BlockPool.cache_full_blocks()`，由它把 block hash 写到 block 元数据和 hash 表中。

入口：`code/vllm/vllm/v1/core/kv_cache_manager.py:442` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:456`

下钻入口：`code/vllm/vllm/v1/core/block_pool.py:211`

核心代码：

```python
block_hash_with_group_id = make_block_hash_with_group_id(
    block_hash, kv_cache_group_id
)
blk.block_hash = block_hash_with_group_id
self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

位置：`code/vllm/vllm/v1/core/block_pool.py:276` 到 `code/vllm/vllm/v1/core/block_pool.py:281`

这里有几个关键点：

```text
1. 只有 full block 会被登记到本地 prefix cache；
2. null block 或被 block_mask 标记为不可命中的 block 会跳过；
3. cache key 不是单纯 block_hash，而是 block_hash + kv_cache_group_id。
```

也就是说，不同 KV cache group 即使 token hash 相同，也会分开查找；SWA、Mamba 等稀疏/混合场景也可能只登记某些可作为 replay 边界的 blocks。

---

## 7. Scheduler 什么时候查本地 prefix cache

### 7.1 WAITING 请求首次本地计算前查

Scheduler 的 `schedule()` 先调度已经在 `running` 队列里的请求，再调度 `waiting / skipped_waiting` 请求。

本地 prefix cache 查询发生在调度 WAITING 请求这一段；多数情况下它会把请求推进到 `RUNNING`，但如果 connector 选择异步加载远端 KV，请求会先进入 `WAITING_FOR_REMOTE_KVS`，完成收包后仍在这一轮收尾逻辑中回到 `WAITING`，下一轮再继续调度：

```python
# Get already-cached tokens.
if request.num_computed_tokens == 0:
    ...
    new_computed_blocks, num_new_local_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:671` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:775`

这说明：

```text
只有 request.num_computed_tokens == 0 时才查本地 prefix cache；
典型场景是新请求第一次从 waiting 开始本地 prefill；
如果请求已经跑过一部分 prefill，或者 async remote KV load 完成后已经带着 num_computed_tokens 回到 WAITING，Scheduler 会走 `request.num_computed_tokens > 0` 分支，不会再次调用 `get_computed_blocks()`。
```

### 7.2 running 请求不会走 get_computed_blocks

`running` 阶段的请求已经有自己的 KV block 分配和 `num_computed_tokens`，Scheduler 直接根据差值算本轮要跑多少 token：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:462` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:466`

然后直接 `allocate_slots()`。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:521` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:528`

因此：

```text
running decode / running chunked prefill 不重新做 prefix lookup；
它们沿用 request 当前已计算进度继续推进。
```

### 7.3 preempted 请求为什么可能重新查

请求被 preempt 时，Scheduler 会释放 blocks，并把状态放回 waiting：

```python
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1117` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1118`

因此 preempted 请求再次从 waiting 被调度时，满足：

```text
request.num_computed_tokens == 0
```

会重新走 `get_computed_blocks()`。

这也是 prefix cache 对抢占恢复有意义的原因：

```text
请求自己的 blocks 被释放了；
但如果 full blocks 仍在 prefix cache hash 表中，就可以重新命中并复用。
```

---

## 8. KVCacheManager.get_computed_blocks() 做什么

入口：`code/vllm/vllm/v1/core/kv_cache_manager.py:202`

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
```

它返回两个东西：

```text
new_computed_blocks：本次新命中的本地 KV blocks；
num_new_local_computed_tokens：这些 blocks 覆盖的 token 数。
```

### 8.1 先判断是否允许读 prefix cache

```python
if not self.enable_caching or request.skip_reading_prefix_cache:
    return self.empty_kv_cache_blocks, 0
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:214` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:219`

会跳过本地 prefix cache 的情况：

```text
全局没有启用 prefix caching；
请求被标记为 skip_reading_prefix_cache。
```

`Request.skip_reading_prefix_cache` 来自 sampling / pooling params：

```python
if self.sampling_params is not None and self.sampling_params.skip_reading_prefix_cache is not None:
    return self.sampling_params.skip_reading_prefix_cache
elif self.pooling_params is not None and self.pooling_params.skip_reading_prefix_cache is not None:
    return self.pooling_params.skip_reading_prefix_cache
return False
```

位置：`code/vllm/vllm/v1/request.py:266` 到 `code/vllm/vllm/v1/request.py:277`

默认值在参数校验阶段设置：generation 请求带 `prompt_logprobs` 时会默认跳过读取 prefix cache，pooling 的 `token_embed` / `token_classify` 也会默认跳过；普通 generation / pooling 请求通常不跳过。

位置：`code/vllm/vllm/sampling_params.py:481` 到 `code/vllm/vllm/sampling_params.py:485`

位置：`code/vllm/vllm/pooling_params.py:124` 到 `code/vllm/vllm/pooling_params.py:131`

### 8.2 full hit 也最多按 request.num_tokens - 1 查

`get_computed_blocks()` 最重要的一行是：

```python
max_cache_hit_length = request.num_tokens - 1
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:221` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:227`

源码注释说明：

```text
When all tokens hit the cache, we must recompute the last token to obtain logits.
```

含义：

```text
即使 prompt 的所有 full blocks 都在本地 cache 里，Scheduler 也不能把全部 prompt token 都当作已完成；
对于 generation 请求，至少要保留最后一个 prompt token 参与一次 forward，才能得到 next-token logits。
```

更细一点：

```text
max_cache_hit_length = prompt_length - 1；
命中 token 数还必须按 block size / scheduler alignment 对齐；
因此可能不只是保留最后 1 个 token forward，而是保留最后一个 block 参与 forward。
```

这就是文档标题里“full hit 为什么仍可能需要保留最后 token 对应区间 forward”的答案。

### 8.3 调用 coordinator 查最长命中

```python
computed_blocks, num_new_computed_tokens = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:228` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:232`

最后包装成 `KVCacheBlocks`：

```python
return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:242`

---

## 9. find_longest_cache_hit 如何查

### 9.1 Coordinator 负责跨 KV cache group 收敛

入口：`code/vllm/vllm/v1/core/kv_cache_coordinator.py:621`

```python
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
```

返回：

```text
hit_blocks_by_group：每个 KV cache group 命中的 block 列表；
hit_length：最终可作为已计算 prefix 的 token 数。
```

Coordinator 的核心职责不是单纯查一个 hash 表，而是协调多种 attention / KV cache group：

```text
FullAttention
SlidingWindow
ChunkedLocalAttention
Mamba
Hybrid KV cache groups
EAGLE / MTP 需要额外 drop 的场景
不同 block_size / hash_block_size 的转换
scheduler_block_size 对齐
num_uncached_common_prefix_tokens 统计
```

它采用迭代收敛逻辑：

```text
先给一个候选 hit_length；
每个 attention group 根据自己的规则查最长可用命中；
如果某个 group 只能支持更短命中，就缩短 hit_length；
必要时重新检查，直到不再变短。
```

位置：`code/vllm/vllm/v1/core/kv_cache_coordinator.py:667` 到 `code/vllm/vllm/v1/core/kv_cache_coordinator.py:731`

### 9.2 FullAttentionManager 的普通前缀查找

普通 full attention 的查找最直观。

入口：`code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:540`

核心逻辑：

```python
max_num_blocks = max_length // block_size
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(
        block_hash, kv_cache_group_ids
    ):
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)
    else:
        break
```

位置：`code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:561` 到 `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:575`

特点：

```text
从第一个 block 开始按顺序查；
遇到第一个 miss 就停止；
因为 block hash 是链式 prefix hash，后续 block 不可能构成合法连续 prefix hit。
```

### 9.3 BlockPool.get_cached_block 真正查 hash 表

入口：`code/vllm/vllm/v1/core/block_pool.py:184`

```python
def get_cached_block(
    self, block_hash: BlockHash, kv_cache_group_ids: list[int]
) -> list[KVCacheBlock] | None:
```

它会对每个 group id 拼接 key：

```python
block_hash_with_group_id = make_block_hash_with_group_id(
    block_hash, group_id
)
block = self.cached_block_hash_to_block.get_one_block(
    block_hash_with_group_id
)
if not block:
    return None
cached_blocks.append(block)
```

位置：`code/vllm/vllm/v1/core/block_pool.py:198` 到 `code/vllm/vllm/v1/core/block_pool.py:209`

含义：

```text
一个 block hash 必须在所有目标 kv_cache_group_ids 下都能找到 block；
只要某个 group miss，这个 block 对当前 attention group 就是 miss。
```

### 9.4 命中长度如何换算成 token 数

Coordinator 在每个 manager 返回 blocks 后，用：

```python
_new_hit_length = len(hit_blocks[0]) * spec.block_size
```

位置：`code/vllm/vllm/v1/core/kv_cache_coordinator.py:700`

所以普通情况下：

```text
num_new_local_computed_tokens = 命中 block 数 × KV cache spec block_size
```

但要注意：

```text
Hybrid / sliding window / chunked local / mamba 可能返回 null block 或不同 group 的 block 列表；
最终 hit_length 以 coordinator 收敛后的 token 数为准，不应只按某一个普通 list 粗暴理解。
```

---

## 10. 命中后如何影响 Scheduler 的 num_new_tokens

### 10.1 Scheduler 合并本地和外部命中

在 WAITING 调度中，本地命中后可能继续查 external KV connector：

```python
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:744` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:747`

如果没有 connector，则：

```text
num_external_computed_tokens = 0
num_computed_tokens = num_new_local_computed_tokens
```

如果有 connector，则本地命中会作为参数传给 connector：

```python
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:721` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:727`

这表示：

```text
本地 GPU prefix cache 先查；
external KV connector 再基于“本地已经命中的 token 数”继续判断远端还能补多少 KV。
```

### 10.2 num_new_tokens 只计算剩余 token

如果不是 async remote KV load，Scheduler 会用：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:781` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:795`

如果是 async remote KV load，则这一步 `num_new_tokens = 0`，只先为外部 computed tokens 分配/登记 slots，等待 connector 收包完成后再继续本地计算。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:781` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:785`

于是：

```text
本地 prefix cache 命中越多，num_computed_tokens 越大；
num_new_tokens 越小；
本轮 forward 的 prefill token 越少。
```

再经过：

```text
long_prefill_token_threshold
chunked prefill token_budget
encoder input 调度
mamba block 对齐
lookahead / spec decode
```

最终才得到本轮真正能调度的 `num_new_tokens`。

### 10.3 示例

假设：

```text
prompt = 10000 tokens
block_size = 16
本地 prefix cache 命中 = 3008 tokens
external KV 命中 = 0 tokens
token_budget 足够
```

则：

```text
num_new_local_computed_tokens = 3008
num_computed_tokens = 3008
num_new_tokens = 10000 - 3008 = 6992
```

Scheduler 本轮只需要为后 6992 个 token 安排 prefill forward。

如果 full prompt 都可命中：

```text
prompt = 10000 tokens
max_cache_hit_length = 9999
block_size = 16
最大可对齐命中 = 9984 tokens
num_new_tokens = 10000 - 9984 = 16
```

实际效果通常是：

```text
看起来 full hit；
但为了最后 token logits，仍保留最后一个 block 做 forward。
```

---

## 11. new_computed_blocks 后续怎么被复用

本地命中后，Scheduler 调用 `allocate_slots()` 时会把命中信息传进去：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_external_computed_tokens=num_external_computed_tokens,
    ...
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:873` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:885`

`allocate_slots()` 的注释把 token 区间分成：

```text
< comp > | < new_comp > | < ext_comp > | < new > | < lookahead >
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:290` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:322`

其中：

```text
comp：request 之前已经计算过的 token；
new_comp：本轮刚从本地 prefix cache 命中的 token；
ext_comp：本轮从 external connector 命中的 token；
new：本轮需要实际 forward 的 token；
lookahead：spec decode 预留 token。
```

### 11.1 先计算本地已计算 token 总数

```python
num_local_computed_tokens = (
    request.num_computed_tokens + num_new_computed_tokens
)
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:353` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:361`

对新 waiting 请求来说，`request.num_computed_tokens` 通常是 0，所以：

```text
num_local_computed_tokens = num_new_local_computed_tokens
```

### 11.2 把命中的 computed blocks 挂到请求上

如果有本地命中或 external 命中：

```python
self.coordinator.allocate_new_computed_blocks(
    request_id=request.request_id,
    new_computed_blocks=new_computed_block_list,
    num_local_computed_tokens=num_local_computed_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:422` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:433`

`allocate_new_computed_blocks()` 内部会先对所有 group 的本地命中 blocks 执行 `add_local_computed_blocks()`，再为 external computed tokens 分配 blocks，避免某个 group 的 external allocation 先发生时把另一个 group 尚未 touch 的命中 block 驱逐。

位置：`code/vllm/vllm/v1/core/kv_cache_coordinator.py:186` 到 `code/vllm/vllm/v1/core/kv_cache_coordinator.py:230`

含义：

```text
本地命中的 blocks 不需要重新分配和重算；
但要加入当前 request 的 block table / req_to_blocks；
加入前会 touch 命中的 blocks，更新引用和 LRU 访问状态，避免仍被当前请求使用时被回收；
这样 Worker 后续 attention 能直接引用这些已有 KV blocks。
```

### 11.3 再为需要 forward 的 token 分配新 blocks

```python
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,
    num_tokens_main_model,
    num_encoder_tokens,
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:435` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:440`

所以 request 的 block table 会包含：

```text
已命中的本地 cached blocks；
external KV 对应的 blocks；
本轮新分配、需要 forward 写入的 blocks；
可能还有 lookahead blocks。
```

随后如果 `enable_caching` 开启且不是 `delay_cache_blocks`，`allocate_slots()` 会把 `total_computed_tokens + num_new_tokens`（再按 `request.num_tokens` 截断）范围内的 full blocks 写入本地 prefix cache；async remote KV load 场景则延迟到收包完成后再 cache。

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:442` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:456`

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2350` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2380`

---

## 12. SchedulerOutput 里如何体现 prefix cache 命中

### 12.1 新请求通过 NewRequestData 发给 Worker

新请求第一次被调度时，会进入 `scheduled_new_reqs`。

Scheduler 构造：

```python
NewRequestData.from_request(
    req, req_to_new_blocks[req.request_id].get_block_ids()
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1024` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1028`

`NewRequestData` 里包含：

```python
block_ids: tuple[list[int], ...]
num_computed_tokens: int
```

位置：`code/vllm/vllm/v1/core/sched/output.py:30` 到 `code/vllm/vllm/v1/core/sched/output.py:65`

这两个字段就是 Worker 侧理解 prefix cache 的关键：

```text
block_ids：当前请求已经拥有/分配的 KV block table；
num_computed_tokens：其中前多少 token 已经可以视为 computed。
```

### 12.2 已缓存请求通过 CachedRequestData 发 diff

已经在 Worker 缓存过的请求，会进入 `scheduled_cached_reqs`。

其中也有：

```python
new_block_ids: list[tuple[list[int], ...] | None]
num_computed_tokens: list[int]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:111` 到 `code/vllm/vllm/v1/core/sched/output.py:126`

这表示：

```text
Worker 不需要每轮重新接收完整请求；
Scheduler 只发新增 block ids 和新的 num_computed_tokens。
```

### 12.3 Scheduler 会先设置 request.num_computed_tokens

WAITING 请求成功调度后：

```python
request.num_computed_tokens = num_computed_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:959`

随后 `_update_after_schedule()` 会把本轮真正调度的 token 数也加上：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1138` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1142`

所以有两个阶段：

```text
schedule waiting 时：先把 prefix cache / external KV 命中的 token 计入 request.num_computed_tokens；
update_after_schedule 时：再把本轮安排 forward 的 token 也预推进到 request.num_computed_tokens。
```

为什么要预推进？源码注释说明：

```text
允许这个 prefill 请求在下一轮 schedule 中立刻继续被调度；
如果后续 spec tokens 被拒绝，再在 update_from_output 中修正。
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1128` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1137`

---

## 13. Worker / ModelRunner 侧看到什么

Worker 不会重新查本地 prefix cache。它消费的是 SchedulerOutput 中已经算好的状态。

新请求进入 Worker 侧 `CachedRequestState` 时，会带上：

```python
block_ids=new_req_data.block_ids,
num_computed_tokens=new_req_data.num_computed_tokens,
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1224` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1237`

然后加入 `InputBatch` 时，会写入：

```python
self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
self.block_table.add_row(request.block_ids, req_index)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:353` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:378`

所以 Worker 侧看到的是：

```text
这个请求已经有一组 block_ids；
前 num_computed_tokens 个 token 不需要重新作为 query token 计算；
本轮只处理 SchedulerOutput.num_scheduled_tokens 指定的新 token。
```

换句话说：

```text
prefix cache 命中结果已经被 Scheduler 折叠进 block table 和 num_computed_tokens；
ModelRunner 只按调度计划准备 input_ids / positions / slot_mapping / attention metadata。
```

---

## 14. 本地 prefix cache 和 external KV cache 的关系

这篇文档重点是本地 prefix cache，但源码里 WAITING 调度实际是两级命中：

```text
1. 本地 GPU prefix cache：KVCacheManager.get_computed_blocks()
2. 外部 KV connector：connector.get_num_new_matched_tokens(request, local_hits)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:671` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:747`

本地命中先发生，external KV 后发生。

原因是：

```text
如果本地已经有前 3000 token 的 KV，就不应该再从远端重复加载这 3000 token；
connector 只需要从本地命中之后的位置继续判断远端是否还能补 KV。
```

统计字段也能看出这个关系：

```python
connector_prefix_cache_queries = (
    request.num_tokens - num_new_local_computed_tokens
)
connector_prefix_cache_hits = num_external_computed_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:739` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:742`

含义：

```text
external connector 的查询范围 = 总 token 数 - 本地已命中 token 数；
external hits 只统计本地之后补上的 token。
```

---

## 15. 为什么 full hit 仍然需要保留最后 token 的 forward

这是 prefix cache 里最容易误解的点。

### 15.1 生成需要 next-token logits

对于 generation 请求，prompt prefill 的最后一步要产出下一 token 的 logits。

如果 Scheduler 认为 prompt 全部 token 都已经 computed，并且本轮 `num_new_tokens = 0`，那 Worker 没有任何 token forward，也就没有 logits 可以采样。

因此 vLLM 在本地 prefix cache lookup 时强制：

```text
最多命中 request.num_tokens - 1。
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:221` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:227`

### 15.2 为什么可能让整个 block 参与 forward

虽然语义上只需要保留最后一个 token 做 forward，但源码注释也说：

```text
This can trigger recomputation of an entire block, rather than just the single last token,
because allocate_slots() requires num_computed_tokens to be block-size aligned.
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:223` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:226`

原因：

```text
prefix cache 命中以 full block 为单位；
num_computed_tokens 需要满足 block-size / scheduler alignment；
如果 prompt_length - 1 落在某个 block 内部，命中长度会向下对齐到上一个完整 block 边界。
```

所以：

```text
full prompt cached ≠ 本轮 0 token forward；
更准确是：最多复用到最后一个需要 logits 的 token 之前、且满足 block / scheduler 对齐的边界。
```

---

## 16. 常见例子

### 16.1 无命中

```text
prompt = 10000 tokens
local hit = 0
external hit = 0
```

则：

```text
num_computed_tokens = 0
num_new_tokens = 10000
本轮从 prompt 开头开始 prefill
```

### 16.2 部分本地命中

```text
prompt = 10000 tokens
local hit = 3008 tokens
external hit = 0
```

则：

```text
num_new_local_computed_tokens = 3008
num_computed_tokens = 3008
num_new_tokens = 6992
本轮只 prefill 后 6992 token
```

### 16.3 本地 + external 命中

```text
prompt = 10000 tokens
local hit = 3008 tokens
external hit = 4096 tokens
```

则：

```text
num_computed_tokens = 3008 + 4096 = 7104
num_new_tokens = 2896
本地 KV blocks 直接复用；
external KV blocks 由 connector 负责 load / recv；
剩余 2896 token 本轮或后续 chunked prefill 计算。
```

### 16.4 full prompt cache

```text
prompt = 10000 tokens
block_size = 16
所有 block 都在 prefix cache
```

`get_computed_blocks()` 仍然设置：

```text
max_cache_hit_length = 9999
```

最终可命中通常向下对齐到：

```text
9984 tokens
```

于是：

```text
num_new_tokens = 16
最后一个 block 参与本轮 forward，用来得到 next-token logits。
```

---

## 17. 容易疑惑的点

### 17.1 prefix cache 查的是 token 还是 block？

查的是 block hash。

```text
Request 根据 token 生成 full-block block_hashes；
BlockPool 根据 block_hash + group_id 找 KVCacheBlock；
命中 block 数再换算成 token 数。
```

### 17.2 为什么只命中连续 prefix，不能跳着命中？

因为 block hash 是链式的：

```text
当前 block hash = hash(parent_block_hash, current_block_tokens, extra_keys)
```

如果前一个 block miss，后面的 block 即使 token 内容类似，也不能证明它属于同一个 prefix chain。

### 17.3 为什么 `get_computed_blocks()` 只返回 full blocks？

因为 KV cache 的复用和 block table 都以 block 为基本单位。

```text
不足一个 block 的 tail token 没有 block hash；
也不会进入本地 prefix cache hash 表。
```

### 17.4 `num_new_local_computed_tokens` 和 `request.num_computed_tokens` 是一回事吗？

不是。

```text
num_new_local_computed_tokens：本轮 prefix lookup 新命中的本地 token 数；
request.num_computed_tokens：Scheduler 维护的请求当前已计算进度。
```

新 waiting 请求里二者可能相等；running 请求中 `request.num_computed_tokens` 会继续随每轮调度推进。

### 17.5 为什么 `get_computed_blocks()` 只在 `request.num_computed_tokens == 0` 时调用？

因为 prefix cache lookup 用于确定请求进入执行时的初始 computed prefix。

请求已经进入 running 后，Scheduler 维护它自己的 computed 进度和 block table，不需要每轮重复查 prefix cache。

### 17.6 prefix cache 命中后是否一定不占用 block 引用？

不是。

命中的 cached blocks 会被挂到当前 request 的 block table，并增加引用 / 访问状态，避免仍在使用时被回收。

相关逻辑在 `BlockPool.touch()`：`code/vllm/vllm/v1/core/block_pool.py:402`

### 17.7 prefix cache 命中是否一定等于性能提升？

通常会减少 prefill forward token 数，但还受这些因素影响：

```text
token_budget / chunked prefill；
block-size 对齐；
full hit 保留最后 block forward；
sliding window / mamba / hybrid attention 约束；
external KV load 是否异步；
KV block 分配是否有足够空间。
```

---

## 18. 总结

本地 prefix cache 命中链路可以记成：

```text
Request.block_hashes
  → BlockPool.cached_block_hash_to_block
  → KVCacheManager.get_computed_blocks()
  → find_longest_cache_hit()
  → num_new_local_computed_tokens
  → num_computed_tokens
  → num_new_tokens = request.num_tokens - num_computed_tokens
  → allocate_slots() 复用命中 blocks + 分配剩余 blocks
  → SchedulerOutput 把 block_ids / num_computed_tokens 发给 Worker
```

最关键的边界是：

```text
本地 prefix cache 只复用 full-block KV；
Scheduler 只在 waiting 请求初次调度或 preempted 后恢复时查；
命中结果先影响 Scheduler 的 token 预算和 KV block 分配；
Worker 不再查 cache，只消费 SchedulerOutput 中的 block table 和 computed token 数；
即使 full hit，也要保留最后 token 的 forward 来生成 logits，实际常表现为最后一个 block 参与本轮 forward。
```

如果只记一句话：

```text
本地 prefix cache lookup 是 Scheduler 分配 KV blocks 前的一次“已计算 prefix 对账”：用 Request.block_hashes 找到已有 KV blocks，把命中部分折算进 num_computed_tokens，从而让本轮 prefill 只计算剩余 token。
```
