# 01. KVCacheManager 在 vLLM V1 里负责什么？

源码位置：

- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/core/block_pool.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/kv_cache_interface.py`

本问题关注：`KVCacheManager` 在 vLLM V1 里到底处于哪一层、负责什么、不负责什么；它如何配合 Scheduler 查询 prefix cache、分配 KV blocks、释放 block、记录 KV cache events；它和 `BlockPool`、KV cache coordinator、Worker / ModelRunner 侧物理 KV cache tensor 的职责边界是什么。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本篇按“先定角色，再走主链路，再拆关键接口，最后总结边界”的方式梳理 `KVCacheManager`。

要回答的问题分成 10 组：

```text
1. KVCacheManager 是哪一层？
2. 它由谁创建？初始化时拿到哪些配置？
3. 它和 Scheduler / BlockPool / coordinator 的关系是什么？
4. 它如何查询本地 prefix cache 命中？
5. 它如何为 running / waiting 请求分配 KV block？
6. 外部 KV Cache / KV Connector 命中如何进入 allocate_slots()？
7. 它如何释放、缓存、重置 KV block？
8. KVCacheBlocks 是什么，为什么要作为接口对象？
9. 它和 Worker / ModelRunner 侧物理 KV cache 的边界是什么？
10. 从 KVPool 视角看，它处在哪些关键插入点？
```

阅读顺序建议：

```text
01_kv_cache_manager_role.md
  → 02_block_pool_and_block_lifecycle.md
  → 03_prefix_cache_lookup.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 09_deferred_free_and_async_safety.md
```

本篇重点讲定位和总链路，不把 `BlockPool`、external KV load、deferred free 展开到最细；后续专题再分别细拆。

---

## 1. 一句话回答

`KVCacheManager` 是 **Scheduler 侧的 KV block 管理入口**。

它负责：

```text
1. 根据 KVCacheConfig 创建底层 KV cache coordinator；
2. 暴露 prefix cache 查询接口；
3. 暴露 block 分配接口 allocate_slots()；
4. 根据 request 进度维护请求到 KV blocks 的逻辑映射；
5. 把本地 prefix cache 命中 block、外部 KV 命中 token、新计算 token、lookahead token 统一放进 block 布局；
6. 在请求结束、抢占、reset 时释放或弹出 block；
7. 在 prefix caching 开启时把完整 block 写入 prefix cache；
8. 提供 block ids 给 SchedulerOutput / KV Connector / Worker；
9. 通过 `take_new_block_ids()` 把新分配的 attention block ids 暴露给 Scheduler，供 Worker 在使用前清零；
10. 提供 KV cache usage、prefix cache stats、KV cache events。
```

它不负责：

```text
1. 不负责 deciding 哪些请求本轮执行，这是 Scheduler 的事；
2. 不负责 token budget，这是 Scheduler 的事；
3. 不负责真正 GPU KV tensor 的分配和读写，这是 Worker / ModelRunner 的事；
4. 不负责 attention metadata、slot mapping、block table 张量化，这是 ModelRunner 的事；
5. 不负责外部 KV 传输协议本身，这是 KV Connector 的事；
6. 不负责 sampled token、stop、logprobs、RequestOutput，这是 Scheduler / OutputProcessor 的事。
```

可以把它理解成：

```text
Scheduler 决定“这个请求本轮要推进多少 token”；
KVCacheManager 决定“这些 token 需要哪些逻辑 KV blocks”；
BlockPool 提供“可用 block 资源”；
Worker / ModelRunner 把 block ids 变成真正的 KV cache tensor 读写。
```

---

## 2. 一句话总览链路

普通 waiting 请求第一次调度时：

```text
Scheduler.schedule()
  → KVCacheManager.get_computed_blocks(request)
  → 得到本地 prefix cache 命中 blocks / tokens
  → 可选：KV Connector 查询外部 KV 命中
  → KVCacheManager.allocate_slots(...)
  → 得到本轮新增 KVCacheBlocks
  → SchedulerOutput 携带 block ids
  → Worker / ModelRunner 使用 block ids 构造 block table / slot mapping
```

running 请求继续推进时：

```text
Scheduler.schedule()
  → 算出 num_new_tokens
  → KVCacheManager.allocate_slots(request, num_new_tokens, lookahead...)
  → 成功则本轮可调度
  → 失败则 Scheduler 触发 preemption
```

请求结束或抢占时：

```text
Scheduler
  → KVCacheManager.free(request)
  或 pop_blocks_for_free(request)
  → BlockPool 释放 / 延迟释放 / 保留 prefix cache blocks
```

---

## 3. KVCacheManager 在 Scheduler 初始化时创建

`KVCacheManager` 不是外层 Engine 创建的，也不是 Worker 创建的。

它在 `Scheduler.__init__()` 中创建：

```python
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=vllm_config.max_in_flight_tokens,
    enable_caching=self.cache_config.enable_prefix_caching,
    use_eagle=self.use_eagle,
    log_stats=self.log_stats,
    enable_kv_cache_events=self.enable_kv_cache_events,
    dcp_world_size=self.dcp_world_size,
    pcp_world_size=self.pcp_world_size,
    scheduler_block_size=self.block_size,
    hash_block_size=hash_block_size,
    metrics_collector=self.kv_metrics_collector,
    watermark=self.scheduler_config.watermark,
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:264`

紧接着，如果有 KV Connector，会把 GPU block pool 绑定给 connector：

```python
if self.connector is not None:
    self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:281`

这说明：

```text
KVCacheManager 是 Scheduler 内部组件；
BlockPool 是 KVCacheManager 内部暴露出来的底层 block 池；
KV Connector 需要知道 block pool，才能在外部 KV transfer 场景中理解 block 资源。
```

---

## 4. KVCacheManager 初始化时做什么

`KVCacheManager.__init__()` 的入口在：

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        ...
    ) -> None:
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:114`

初始化时最关键的是创建 coordinator：

```python
self.coordinator = get_kv_cache_coordinator(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=max_in_flight_tokens,
    use_eagle=self.use_eagle,
    enable_caching=self.enable_caching,
    enable_kv_cache_events=enable_kv_cache_events,
    dcp_world_size=dcp_world_size,
    pcp_world_size=pcp_world_size,
    scheduler_block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
    metrics_collector=self.metrics_collector,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:148`

然后保存：

```python
self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
self.block_pool = self.coordinator.block_pool
self.kv_cache_config = kv_cache_config
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:161`

这里有三个重点：

```text
1. KVCacheManager 自己不直接实现所有 block 分配细节；
   它把复杂布局交给 coordinator。

2. coordinator 持有 block_pool；
   KVCacheManager 暴露 self.block_pool 供 Scheduler / Connector 使用。

3. 支持多个 KV cache groups；
   所以返回 block 时外层是 tuple[group]，不是单纯 list[block]。
```

---

## 5. coordinator 是什么角色

`KVCacheManager` 很多方法只是薄封装，真正复杂逻辑在 coordinator。

例如：

```python
self.coordinator.find_longest_cache_hit(...)
self.coordinator.get_num_blocks_to_allocate(...)
self.coordinator.allocate_new_computed_blocks(...)
self.coordinator.allocate_new_blocks(...)
self.coordinator.cache_blocks(...)
self.coordinator.free(...)
self.coordinator.remove_skipped_blocks(...)
self.coordinator.get_blocks(...)
```

可以这样理解：

```text
KVCacheManager：Scheduler 面向 KV cache 的稳定接口。
coordinator：根据 KVCacheConfig 处理具体 block 布局、group、hybrid、sliding window、Mamba、DCP / PCP 等差异。
BlockPool：真正管理 KVCacheBlock 对象和 prefix cache hash map。
```

为什么需要 coordinator？

因为 vLLM V1 的 KV cache 不一定是单一形态：

```text
普通 full attention；
sliding window attention；
hybrid attention；
Mamba / attention 混合；
多 KV cache group；
DCP / PCP；
EAGLE / spec decode lookahead；
encoder-decoder cross-attention blocks。
```

这些差异不适合都塞进 Scheduler，所以 Scheduler 只调用 `KVCacheManager`。

---

## 6. KVCacheBlocks 是什么

`KVCacheBlocks` 是 `KVCacheManager` 返回给 Scheduler 的 block 包装对象。

定义：

```python
@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:29`

注释已经说明它的定位：

```text
KVCacheBlocks 是 Scheduler 和 KVCacheManager 之间的接口对象，
用于隐藏 KVCacheManager 内部数据结构。
```

它为什么是：

```python
tuple[Sequence[KVCacheBlock], ...]
```

而不是简单：

```python
list[KVCacheBlock]
```

因为它要支持多个 KV cache group：

```text
blocks[i][j]
  → 第 i 个 kv_cache_group 的第 j 个 block
```

后续如果不同 group 有不同 block size，也不需要改 Scheduler 的外层接口。

### 6.1 转成 block ids

Scheduler / Worker 更常用的是 block id，不是 Python 对象。

`KVCacheBlocks` 提供：

```python
def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:73`

它把：

```text
KVCacheBlock 对象
```

转换成：

```text
block_ids: tuple[list[int], ...]
```

这些 block ids 后续会进入 `SchedulerOutput`，再被 Worker / ModelRunner 用于 block table / slot mapping。

### 6.2 empty_kv_cache_blocks

初始化时，`KVCacheManager` 预构造了一个空 block 对象：

```python
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:182`

原因是减少频繁创建空对象带来的 GC 开销。

---

## 7. get_computed_blocks：查询本地 prefix cache

waiting 请求第一次调度时，Scheduler 会先查本地 prefix cache。

Scheduler 调用位置：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:762`

`KVCacheManager.get_computed_blocks()` 定义：

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:207`

它返回两个东西：

```text
1. new_computed_blocks：本地 prefix cache 命中的 KV blocks；
2. num_new_computed_tokens：本地 prefix cache 命中的 token 数。
```

### 7.1 什么时候不查 prefix cache

如果 prefix caching 关闭，或者请求明确跳过读 prefix cache：

```python
if not self.enable_caching or request.skip_reading_prefix_cache:
    return self.empty_kv_cache_blocks, 0
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:223`

`request.skip_reading_prefix_cache` 常见于一些不能直接复用 prefix cache 的请求，比如需要 prompt logprobs 或特定 pooling 场景。

### 7.2 为什么 full hit 也要留最后一个 token

源码注释说明：

```python
# NOTE: When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:226`

所以：

```python
max_cache_hit_length = request.num_tokens - 1
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:232`

意思是：

```text
即使整个 prompt 都在 prefix cache 里，decode 下一个 token 之前，也通常需要重新计算最后一个 token 来拿 logits。
```

因此本地 prefix cache 命中最多到：

```text
request.num_tokens - 1
```

### 7.3 真正查找 longest cache hit

```python
computed_blocks, num_new_computed_tokens = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:233`

这里依赖：

```text
request.block_hashes
```

也就是请求 token 分块后的 block hash。

如果开启 KV cache events 且请求的 `kv_cache_report_mode` 为 `full`，复用的 prefix cache blocks 也会通过 `emit_cached_block_events()` 补发 `BlockStored` 事件，便于外部消费者拿到完整缓存状态。

最终返回：

```python
return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:267`

---

## 8. allocate_slots：KVCacheManager 最核心的方法

`allocate_slots()` 是 `KVCacheManager` 最关键的方法。

定义：

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
    has_scheduled_reqs: bool = True,
) -> KVCacheBlocks | None:
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:269`

它的返回值：

```text
KVCacheBlocks：分配成功，本轮新增 / 需要传给 Worker 的 blocks；
None：分配失败，Scheduler 需要跳过或抢占。
```

---

## 9. allocate_slots 的参数含义

### 9.1 request

当前正在调度的请求。

`KVCacheManager` 通过 request 拿到：

```text
request_id；
status；
num_computed_tokens；
num_tokens；
block_hashes；
all_token_ids；
num_preemptions；
```

### 9.2 num_new_tokens

本轮本地要计算的 token 数。

对于 running 请求，Scheduler 先算：

```text
num_new_tokens = 目标 token 数 - 已计算 token 数
```

然后调用：

```python
self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:572`

对于 waiting 请求，`num_new_tokens` 通常是：

```text
request.num_tokens - num_computed_tokens
```

再受 token_budget / chunked prefill / encoder / Mamba 限制。

### 9.3 num_new_computed_tokens / new_computed_blocks

这两个来自本地 prefix cache 命中：

```text
num_new_computed_tokens：本地命中的 token 数；
new_computed_blocks：本地命中的 blocks。
```

Scheduler waiting 阶段先查：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

然后传进：

```python
num_new_computed_tokens=num_new_local_computed_tokens,
new_computed_blocks=new_computed_blocks,
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:945`

### 9.4 num_external_computed_tokens

这是外部 KV Connector 命中的 token 数。

Scheduler 中：

```python
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)
...
num_external_computed_tokens = ext_tokens
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:775`

然后传入 `allocate_slots()`：

```python
num_external_computed_tokens=num_external_computed_tokens,
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:948`

含义是：

```text
这些 token 的 KV 不在 vLLM 本地 prefix cache 中，
但 connector 认为外部 KV cache / KVPool 中有，
因此需要为它们在本地分配 block 槽位，后续 load 进来。
```

### 9.5 delay_cache_blocks

```python
delay_cache_blocks: bool = False
```

源码注释：

```text
used by P/D when allocating blocks used in a KV transfer
which will complete in a future step.
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:297`

Scheduler 在 async load 远端 KV 时传：

```python
delay_cache_blocks=load_kv_async
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:949`

意思是：

```text
这些 block 现在先分配出来给外部 KV load 使用，
但 KV 数据还没真正 load 完，暂时不要把它们当作已经可用于 prefix cache 的本地 cached blocks。
```

### 9.6 num_lookahead_tokens

用于 spec decode / EAGLE 等需要提前分配 lookahead slots 的场景。

```text
num_new_tokens：本轮真实要计算 / 验证的 token；
num_lookahead_tokens：为后续 speculative tokens 预留的 KV slots。
```

### 9.7 num_encoder_tokens

用于 encoder-decoder cross-attention block 分配。

源码注释：

```text
For decoder-only models, this should be 0.
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:300`

### 9.8 full_sequence_must_fit / reserved_blocks / watermark

这些是 admission 控制相关参数：

```text
full_sequence_must_fit：要求完整序列能放下才接纳；
reserved_blocks：为其他 in-flight 请求保留 block；
watermark：waiting / preempted request 接纳时保留一定空闲 block 水位。
```

其中 `reserved_blocks` 在 async KV load 中很重要，避免一个异步 load 把其他正在 prefill 的请求需要的 block 吃光。

---

## 10. allocate_slots 的 block 布局

源码注释给出布局：

```text
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
----------------------------------------------------------------------
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:315`

可以解释成：

| 区段 | 含义 |
|---|---|
| `comp` | 请求之前已经计算过、已经持有的 tokens |
| `new_comp` | 本轮新查到的本地 prefix cache 命中 tokens |
| `ext_comp` | 外部 KV cache 命中、需要 load 的 tokens |
| `new` | 本轮要本地 forward 计算的 tokens |
| `lookahead` | spec decode / EAGLE 预留 tokens |

注意：

```text
to be computed = new + lookahead
```

但：

```text
to be allocated = ext_comp + new + lookahead
```

因为 external computed tokens 虽然不用本地 forward，但它们的 KV 需要被 load 到本地 block，所以也要分配槽位。

---

## 11. allocate_slots 的三阶段

源码注释说明：

```text
The allocation has three stages:
- Free unnecessary blocks in `comp` and check if we have sufficient free blocks.
- Handle prefix tokens (`comp + new_comp + ext_comp`).
- Allocate new blocks for tokens to be computed (`new + lookahead`).
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:353`

实际流程可以拆成：

### 11.1 计算本地已计算和总已计算 tokens

```python
num_local_computed_tokens = (
    request.num_computed_tokens + num_new_computed_tokens
)
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:380`

这里区分：

```text
local computed：vLLM 本地已经有 KV 的 token；
total computed：本地 + 外部命中的 token。
```

### 11.2 可选 full sequence admission 检查

如果要求完整序列必须能放下：

```python
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:397`

这一步是 admission gate，会按完整序列、watermark 和 coordinator 的 `apply_admission_cap=True` 检查容量，用于避免 chunked prefill 只看第一块就过度接纳请求。

### 11.3 删除不再需要的 skipped blocks

```python
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens),
    num_prompt_tokens=request.num_prompt_tokens,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:429`

例如 sliding window attention 中，窗口外的旧 block 可以从请求 blocks 中替换成 null block 或释放引用。

源码注释强调：

```text
Should call this function before allocating new blocks to reduce the number of evicted blocks.
```

也就是说，先按已处理且安全提交的 token 边界清理无用 block，再决定是否有足够空闲 block；这里会扣除 `request.num_in_flight_tokens`，避免异步执行或 spec 回退仍可能读取的 block 被过早释放。

### 11.4 计算还需要分配多少 block

```python
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=request.request_id,
    num_tokens=num_tokens_need_slot,
    new_computed_blocks=new_computed_block_list,
    num_encoder_tokens=num_encoder_tokens,
    total_computed_tokens=num_local_computed_tokens
    + num_external_computed_tokens,
    num_local_computed_tokens=num_local_computed_tokens,
    num_tokens_main_model=num_tokens_main_model,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:435`

这里会综合：

```text
已有 blocks；
本地 prefix hit blocks；
外部 KV hit tokens；
本轮 new tokens；
lookahead tokens；
encoder tokens；
sliding window / hybrid 规则。
```

### 11.5 检查 free blocks 是否够

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:448`

如果返回 `None`：

```text
running 请求：Scheduler 可能抢占其他 running 请求；
waiting 请求：Scheduler 通常停止 waiting 调度或跳过。
```

### 11.6 把 prefix hit / external hit 接到请求 blocks 上

```python
if (
    new_computed_block_list is not self.empty_kv_cache_blocks.blocks
    or num_external_computed_tokens > 0
):
    self.coordinator.allocate_new_computed_blocks(
        request_id=request.request_id,
        new_computed_blocks=new_computed_block_list,
        num_local_computed_tokens=num_local_computed_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
    )
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:454`

这一步处理的是：

```text
new_comp：本地 prefix cache 命中 blocks，需要增加引用或接入请求 block list；
ext_comp：外部 KV 命中 tokens，需要为本地 load 分配相应 block。
```

### 11.7 为 new + lookahead 分配新 blocks

```python
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,
    num_tokens_main_model,
    num_encoder_tokens,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:467`

这是真正为本轮新增计算 token / lookahead / encoder tokens 分配 block 的地方。

### 11.8 是否立即 cache blocks

如果 prefix caching 关闭，或者是异步 KV load 需要延迟 cache：

```python
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:476`

否则会把已经可以安全缓存的 block 写入 prefix cache：

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:484`

注意这里会 cap 到 `request.num_tokens`，避免把可能被拒绝的 draft tokens 提前 cache。

---

## 12. running 请求如何使用 KVCacheManager

running 请求已经在 `self.running` 中，通常已经持有部分 KV blocks。

Scheduler 先算本轮还要推进多少 token，然后调用：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:572`

如果分配成功：

```python
scheduled_running_reqs.append(request)
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:622`

如果分配失败：

```text
Scheduler 会抢占 running 队列中优先级最低或队尾请求，释放它的 KV blocks，再重试 allocate_slots()。
```

所以对于 running 请求：

```text
KVCacheManager 决定“是否有足够 block 继续推进”；
Scheduler 决定“不够时抢占谁”。
```

---

## 13. waiting 请求如何使用 KVCacheManager

waiting 请求进入 running 前，调用链更复杂。

### 13.1 先查本地 prefix cache

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:762`

### 13.2 再查外部 KV Connector

如果有 connector：

```python
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:775`

### 13.3 合并 computed tokens

```python
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:797`

这里的含义是：

```text
本地 prefix cache 命中的 token
+ 外部 KV cache 命中的 token
= Scheduler 认为不需要本轮本地 forward 的 token
```

### 13.4 调用 allocate_slots()

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:942`

这是 waiting 请求从“还没进入模型执行流”到“拥有本地 KV block 布局”的关键一步。

### 13.5 分配后通知 connector

如果有 KV Connector：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:970`

这一步很关键：

```text
KVCacheManager 负责分配 block；
Connector 需要知道这些 block，才能生成外部 KV load metadata。
```

---

## 14. get_blocks / get_block_ids：把逻辑 blocks 暴露给上层

`KVCacheManager` 提供：

```python
def get_blocks(self, request_id: str) -> KVCacheBlocks:
    return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:618`

以及：

```python
def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
    return self.get_blocks(request_id).get_block_ids()
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:622`

Scheduler 会在 waiting 请求调度成功后记录：

```python
req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
    request_id
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1022`

这里特意使用 `get_blocks(request_id)` 取请求当前完整 block 布局，而不是只使用 `allocate_slots()` 返回的新 blocks；因为首次进入 running 的请求需要把 prefix hit、external hit 和本轮新分配 block 的完整布局一起发给 Worker。

这些 blocks 后续会用于构造：

```text
scheduled_new_reqs；
scheduled_cached_reqs；
SchedulerOutput；
KV Connector metadata；
Worker 侧 block table。
```

---

## 15. cache_blocks：把已完成 tokens 写入 prefix cache

`KVCacheManager.cache_blocks()`：

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    if self.enable_caching:
        self.coordinator.cache_blocks(request, num_computed_tokens)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:646`

它的作用是：

```text
把请求中已经完整计算、可复用的 full blocks 写入 prefix cache。
```

`allocate_slots()` 成功后，如果不是 `delay_cache_blocks`，也会调用：

```python
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:488`

异步外部 KV load 完成后，Scheduler 也会在恢复请求时调用相关 cache 流程。这个会在 `05_external_kv_load_flow.md` 里细讲。

---

## 16. free / pop_blocks_for_free：释放请求 block

请求结束或抢占时，需要释放它持有的 blocks。

简单释放接口：

```python
def free(self, request: Request) -> None:
    self.coordinator.free(request.request_id)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:492`

但在异步调度、KV transfer、deferred free 场景下，Scheduler 可能不能立刻把 block 归还给 pool。

因此还有：

```python
def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
    return self.coordinator.pop_blocks_for_free(request.request_id)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:521`

它的含义是：

```text
先把 request 的 block bookkeeping 弹出来，
但不马上归还给 block pool，
由 Scheduler 判断安全时机后再释放。
```

这和 `deferred free` 关系很大，后续在 `09_deferred_free_and_async_safety.md` 细讲。

---

## 17. remove_skipped_blocks：释放不再参与 attention 的 block

`remove_skipped_blocks()`：

```python
def remove_skipped_blocks(
    self,
    request_id: str,
    processed_computed_tokens: int,
    num_prompt_tokens: int | None = None,
) -> None:
    self.coordinator.remove_skipped_blocks(
        request_id, processed_computed_tokens, num_prompt_tokens
    )
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:502`

它用于：

```text
按已经处理且安全提交的 computed-token 边界，把请求里后续 attention 不再需要的 block 移除或替换成 null block；`num_prompt_tokens` 用于 R-SWA gap eviction。
```

典型场景：

```text
sliding window attention；
某些 hybrid attention；
外部 KV transfer 后部分 blocks 不再需要本地保留；
P/D 场景中移除 skipped blocks。
```

`allocate_slots()` 中会先调用它，再分配新 blocks：

```python
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens),
    num_prompt_tokens=request.num_prompt_tokens,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:429`

原因是：

```text
先释放或标记无用 block，可以减少后续分配时的 eviction 压力。
```

---

## 18. reset_prefix_cache / evict_blocks / take_new_block_ids / take_kv_cache_block_copies / take_events

### 18.1 reset_prefix_cache

```python
def reset_prefix_cache(self) -> bool:
    if not self.block_pool.reset_prefix_cache():
        return False
    if self.log_stats:
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.reset = True
    return True
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:542`

它负责清空 prefix cache hash 状态。

注意：

```text
只有在没有活跃 blocks 时才能安全 reset prefix cache。
```

这由 `BlockPool.reset_prefix_cache()` 检查。

### 18.2 evict_blocks

```python
def evict_blocks(self, block_ids: set[int]) -> None:
    self.block_pool.evict_blocks(block_ids)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:534`

这个接口常用于外部 KV load 失败或 connector 报告 invalid blocks 时，从 prefix cache hash table 中移除对应 block。

### 18.3 take_new_block_ids

```python
def take_new_block_ids(self) -> list[int]:
    ids: list[int] = []
    for mgr in self.coordinator.single_type_managers:
        ids.extend(mgr.take_new_block_ids())
    return ids
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:663`

Scheduler 在构造 `SchedulerOutput` 前会调用它，把新分配、需要 zero 的 attention block ids 放入 `SchedulerOutput.new_block_ids_to_zero`。Worker / ModelRunner 在 `_update_states()` 里真正执行设备侧清零。

### 18.4 take_kv_cache_block_copies

```python
def take_kv_cache_block_copies(
    self,
) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:670`

它会 drain coordinator 中 pending 的 CoW block copy，返回给 `SchedulerOutput.kv_cache_block_copies`，同时返回需要暂时 retain 的 source / destination blocks，避免 copy 执行前被释放。

### 18.5 take_events

```python
def take_events(self) -> list[KVCacheEvent]:
    events = self.block_pool.take_events()
    ...
    return events
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:592`

它把 block pool 的 KV cache events 取出，并为 `BlockStored` 事件补充 KV cache spec kind / sliding window metadata。

这些 events 可用于观测：

```text
BlockStored；
BlockRemoved；
AllBlocksCleared；
```

---

## 19. usage / stats：KVCacheManager 的统计能力

`KVCacheManager` 提供 usage：

```python
@property
def usage(self) -> float:
    return self.block_pool.get_usage()
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:186`

含义：

```text
当前 KV block pool 的使用率。
```

prefix cache stats：

```python
def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
    if not self.log_stats:
        return None
    stats = self.prefix_cache_stats
    self.prefix_cache_stats = PrefixCacheStats()
    return stats
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:195`

`get_computed_blocks()` 中会记录 prefix cache 命中：

```python
self.prefix_cache_stats.record(
    num_tokens=request.num_tokens,
    num_hits=num_new_computed_tokens,
    preempted=request.num_preemptions > 0,
)
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:261`

所以：

```text
KVCacheManager 不只管理 block，也负责 prefix cache hit 统计和 KV cache usage 观测入口。
```

---

## 20. BlockPool 的位置：KVCacheManager 下面的资源池

`BlockPool` 定义在：`vllm/vllm/v1/core/block_pool.py:143`

源码注释：

```python
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks.
    ...
    """
```

它负责：

```text
1. 创建所有 KVCacheBlock 对象；
2. 维护 free_block_queue；
3. 维护 prefix cache hash → block 的映射；
4. 分配新 blocks；
5. touch 命中的 cached blocks；
6. free blocks；
7. evict prefix cache blocks；
8. reset prefix cache；
9. 生成 KV cache events。
```

初始化时会创建所有 blocks：

```python
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)
]
```

位置：`vllm/vllm/v1/core/block_pool.py:175`

并创建 free queue：

```python
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```

位置：`vllm/vllm/v1/core/block_pool.py:181`

再创建 prefix cache map：

```python
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

位置：`vllm/vllm/v1/core/block_pool.py:184`

可以这样理解：

```text
KVCacheManager 是 Scheduler 面向 KV cache 的 API；
BlockPool 是底层 block 对象池和 prefix cache hash 表。
```

---

## 21. KVCacheManager 和 Worker / ModelRunner 的边界

这是最容易混淆的地方。

### 21.1 KVCacheManager 管逻辑 block

`KVCacheManager` 管的是：

```text
request_id → KVCacheBlock list；
block_id；
block hash；
free / cached / referenced block 状态；
prefix cache 命中；
哪些 block 分配给哪个请求。
```

它不直接持有：

```text
GPU 上的 key tensor；
GPU 上的 value tensor；
attention backend metadata；
slot mapping tensor；
block table tensor。
```

### 21.2 ModelRunner 管物理 KV cache 和 attention 使用

Worker / ModelRunner 侧负责：

```text
初始化 KV cache tensors；
根据 SchedulerOutput 中的 block ids 更新 InputBatch.block_table；
构造 slot mapping；
构造 attention metadata；
在 forward 中让 attention backend 读写物理 KV cache。
```

所以边界是：

```text
KVCacheManager：这个请求拥有哪些 block ids。
ModelRunner：这些 block ids 在本轮 forward 中如何映射到物理 KV cache tensor。
```

---

## 22. KVCacheManager 和 KV Connector 的边界

KV Connector 负责外部 KV Cache / KVPool 交互。

Scheduler 侧典型链路：

```text
connector.get_num_new_matched_tokens()
  → 告诉 Scheduler 外部命中多少 token

KVCacheManager.allocate_slots(... num_external_computed_tokens ...)
  → 为这些外部命中 token 分配本地 block 槽位

connector.update_state_after_alloc(request, kv_cache_manager.get_blocks(...), ...)
  → connector 根据本地 block 布局生成 load metadata
```

边界是：

```text
KV Connector 决定“外部有没有 KV、要不要 load / save”；
KVCacheManager 决定“本地要给这些 KV 分配哪些 block”。
```

KVCacheManager 不知道外部 KV 的传输协议；它只接收：

```text
num_external_computed_tokens
```

以及通过 Scheduler 被要求：

```text
为这些 token 分配本地 slots。
```

---

## 23. 一个完整例子：无外部 KV 的普通 prefill

假设：

```text
prompt = 10000 tokens
本地 prefix cache hit = 3000 tokens
外部 KV Connector = None
本轮 token_budget 足够
```

调度过程：

```text
1. Scheduler 取 waiting request。
2. 调用 KVCacheManager.get_computed_blocks(request)。
3. 返回：
   num_new_local_computed_tokens = 3000
   new_computed_blocks = prefix cache 命中的 blocks
4. Scheduler 计算：
   num_computed_tokens = 3000
   num_new_tokens = 10000 - 3000 = 7000
5. 调用 KVCacheManager.allocate_slots(
     num_new_tokens=7000,
     num_new_computed_tokens=3000,
     new_computed_blocks=...
   )。
6. KVCacheManager：
   - 把命中的 prefix blocks 接到 request；
   - 为剩余 7000 tokens 分配新 blocks；
   - cache 已完成 full blocks；
   - 返回 new_blocks。
7. Scheduler 把请求放入 running，并把 block ids 写入 SchedulerOutput。
```

本轮真正 forward 的 token 是：

```text
7000 tokens
```

但本地请求 block 布局覆盖的是：

```text
3000 cached tokens + 7000 new tokens
```

---

## 24. 一个完整例子：有外部 KV 命中

假设：

```text
prompt = 10000 tokens
本地 prefix cache hit = 3000 tokens
外部 KVPool hit = 5000 tokens
剩余需要本地 forward = 2000 tokens
```

Scheduler 先查本地：

```text
num_new_local_computed_tokens = 3000
```

再查 connector：

```text
num_external_computed_tokens = 5000
```

合并：

```text
num_computed_tokens = 3000 + 5000 = 8000
num_new_tokens = 10000 - 8000 = 2000
```

调用 `allocate_slots()` 时：

```text
new_comp = 3000 本地 prefix cache blocks
ext_comp = 5000 外部 KV tokens，需要本地 block slots
new = 2000 本轮 forward tokens
```

KVCacheManager 要做三件事：

```text
1. 接入本地 prefix cache hit blocks；
2. 为外部 KV 命中的 5000 tokens 分配本地 block 槽位；
3. 为剩余 2000 tokens 分配本地 forward 写入的 block 槽位。
```

然后 Scheduler 通知 connector：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

这样 connector 才知道：

```text
外部 KVPool 的 KV 应该 load 到本地哪些 block ids。
```

---

## 25. 一个完整例子：running decode 分配失败触发抢占

假设 running 请求本轮 decode 需要 1 个 token，但 KV block pool 不够。

Scheduler 调用：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

如果返回 `None`：

```text
KVCacheManager 只表达“当前没有足够 block”；
它不决定抢占谁。
```

随后 Scheduler：

```text
1. 按 priority 或 running 队尾选择 preempted_req；
2. 调用 _preempt_request(preempted_req, ...)
3. 释放被抢占请求的 blocks；
4. 重试 allocate_slots()。
```

所以抢占边界是：

```text
KVCacheManager：判断 block 是否够；
Scheduler：决定怎么抢占和状态迁移。
```

---

## 26. 容易疑惑的点

### 26.1 KVCacheManager 是不是 KV cache 本身？

不是。

```text
KVCacheManager 管的是逻辑 block 分配和映射；
真正的 KV cache tensor 在 Worker / ModelRunner 侧。
```

### 26.2 KVCacheManager 是不是 BlockPool？

不是。

```text
KVCacheManager 是上层管理接口；
BlockPool 是底层 KVCacheBlock 对象池和 prefix cache hash map。
```

### 26.3 KVCacheManager 是否决定本轮调度多少 token？

不决定。

```text
num_new_tokens 由 Scheduler 根据 request 状态、token_budget、chunked prefill、encoder、Mamba 等条件算好；
KVCacheManager 只根据这个数量判断和分配 block。
```

### 26.4 allocate_slots 返回 None 是不是请求失败？

不是。

```text
返回 None 只表示当前 block 不够或 admission 不通过；
Scheduler 可以抢占、跳过、等待下一轮，或者停止 waiting 调度。
```

### 26.5 prefix cache 命中的 token 是否还会 forward？

通常不会。

```text
prefix cache 命中的 token 被计入 num_computed_tokens，
本轮 forward 只计算剩余 num_new_tokens。
```

但 full hit 仍通常会重算最后一个 token 来获得 logits。

### 26.6 外部 KV 命中的 token 是否需要本地 block？

需要。

```text
外部 KV 命中表示不用本地 forward 计算这些 token，
但要把外部 KV load 到本地 KV cache block，
所以 allocate_slots() 仍要为 ext_comp 分配本地 slots。
```

### 26.7 delay_cache_blocks 是不是不分配 block？

不是。

```text
delay_cache_blocks=True 表示先分配 block，
但不要立即把这些 block 作为本地 prefix cache 命中项缓存起来，
因为远端 KV load 可能还没完成。
```

---

## 27. 从“回答问题”的角度总结

如果问：

```text
KVCacheManager 在 vLLM V1 里负责什么？
```

可以回答：

```text
KVCacheManager 是 Scheduler 侧的 KV block 管理入口。
它负责查询本地 prefix cache 命中、根据本轮调度 token 数为请求分配 KV blocks、处理本地 prefix cache blocks 和外部 KV 命中 tokens 的 block 布局、释放或延迟释放请求 blocks、记录 prefix cache stats 和 KV cache events。
它不持有真正的 GPU KV tensor，也不执行 KV load / save；真正的物理 KV cache 在 Worker / ModelRunner 侧，外部 KV 传输由 KV Connector 负责。
```

如果问：

```text
KVCacheManager 和 Scheduler 是什么关系？
```

可以回答：

```text
Scheduler 决定本轮哪些请求执行、每个请求执行多少 token；KVCacheManager 根据 Scheduler 给出的 num_new_tokens、本地 prefix cache 命中、外部 KV 命中、lookahead token 等信息，判断 KV block 是否够并分配对应 block。分配失败时，KVCacheManager 返回 None，由 Scheduler 决定抢占、跳过或停止调度。
```

如果问：

```text
KVCacheManager 和 Worker / ModelRunner 是什么关系？
```

可以回答：

```text
KVCacheManager 管理的是 Scheduler 侧的逻辑 block 分配和 block id；Worker / ModelRunner 根据 SchedulerOutput 中的 block ids 构造 block table、slot mapping 和 attention metadata，并在真实 GPU KV cache tensor 上读写 KV。
```

---

## 28. 最关键流程图

```text
Scheduler.schedule()
  │
  ├─ running 请求
  │    ├─ Scheduler 计算 num_new_tokens
  │    ├─ KVCacheManager.allocate_slots()
  │    │    ├─ remove_skipped_blocks()
  │    │    ├─ get_num_blocks_to_allocate()
  │    │    ├─ 检查 free blocks / watermark / reserved_blocks
  │    │    ├─ allocate_new_blocks()
  │    │    └─ cache_blocks()
  │    ├─ 成功：记录 req_to_new_blocks / num_scheduled_tokens
  │    └─ 失败：Scheduler 抢占其他 running request
  │
  ├─ waiting 请求
  │    ├─ KVCacheManager.get_computed_blocks()
  │    │    └─ 查询本地 prefix cache
  │    ├─ KV Connector get_num_new_matched_tokens()
  │    │    └─ 查询外部 KV cache / KVPool
  │    ├─ Scheduler 计算 num_computed_tokens / num_new_tokens
  │    ├─ KVCacheManager.allocate_slots()
  │    │    ├─ 接入本地 prefix hit blocks
  │    │    ├─ 为 external KV hit tokens 分配本地 slots
  │    │    ├─ 为本轮 forward tokens 分配 blocks
  │    │    └─ 可选 delay_cache_blocks
  │    ├─ KV Connector update_state_after_alloc()
  │    └─ 请求进入 running 或 WAITING_FOR_REMOTE_KVS
  │
  ├─ SchedulerOutput
  │    └─ 携带 block ids / kv_connector_metadata
  │
  └─ Worker / ModelRunner
       ├─ 根据 block ids 构造 block table / slot mapping
       ├─ 执行 KV load / model forward
       └─ 返回 ModelRunnerOutput
```

---

## 29. 最关键对象关系

```text
Request
  Scheduler 管理的请求对象，带有 request_id、num_tokens、num_computed_tokens、block_hashes 等状态。

KVCacheManager
  Scheduler 侧 KV block 管理入口。

KVCacheBlocks
  KVCacheManager 返回给 Scheduler 的 block 包装对象，隐藏内部多 group 结构。

KVCacheBlock
  单个逻辑 KV block 对象，带 block_id、block_hash、ref_cnt 等状态。

BlockPool
  底层 KVCacheBlock 资源池和 prefix cache hash map。

coordinator
  处理具体 KV cache group、sliding window、hybrid、Mamba、DCP / PCP 等布局细节。

SchedulerOutput
  把 block ids 和本轮调度计划传给 Worker / ModelRunner。

ModelRunner
  使用 block ids 构造 block table / slot mapping，并在物理 KV cache tensor 上执行 forward。
```

---

## 30. 和后续专题的关系

本篇回答的是 `KVCacheManager` 的总定位。

后续专题继续拆：

```text
02_block_pool_and_block_lifecycle.md
  详细解释 BlockPool、KVCacheBlock、ref_cnt、free queue、prefix cache hash map。

03_prefix_cache_lookup.md
  详细解释 get_computed_blocks()、block_hashes、full hit 重算最后一个 token。

04_scheduler_kv_connector_flow.md
  详细解释 Scheduler 如何把外部 KV hit 合并进 num_computed_tokens，并通知 connector。

05_external_kv_load_flow.md
  详细解释 load_kv_async、WAITING_FOR_REMOTE_KVS、finished_recving。

09_deferred_free_and_async_safety.md
  详细解释 pop_blocks_for_free()、deferred_frees、异步安全释放。
```

最终最小心智模型：

```text
KVCacheManager = Scheduler 侧的逻辑 KV block 管家：查 prefix cache、分配 block、释放 block、产出 block ids；真正的 KV tensor 读写在 Worker / ModelRunner，外部 KV 传输在 KV Connector。
```
