# 05. prefix cache / 外部 KV cache 命中了多少 token？

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：Scheduler 在把 waiting 请求调度进 running 之前，如何判断本地 prefix cache 命中了多少 token、外部 KV cache 又额外命中了多少 token，以及这些命中结果如何影响本轮要计算的 `num_new_tokens`、KV block 分配和 KV Connector metadata。

一句话概括：

```text
waiting 请求第一次被调度时，Scheduler 先查本地 prefix cache，
再把本地命中 token 数传给 KV Connector 查询外部额外命中，
最终用：

num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens

表示已经可复用的 token 数，
再用：

num_new_tokens = request.num_tokens - num_computed_tokens

表示本轮还需要本地模型实际计算的 token 数。
```

---

## 1. 一句话回答

prefix cache / 外部 KV cache 命中逻辑发生在 waiting 请求调度阶段。

入口条件是：

```python
if request.num_computed_tokens == 0:
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:724`

也就是说，只有请求还没有任何已计算进度时，Scheduler 才会执行这一轮本地 prefix cache 和外部 KV cache 查询。

核心流程是：

```text
waiting request
  → 查询本地 prefix cache
  → 得到 num_new_local_computed_tokens
  → 如果有 KV Connector，查询外部 KV cache
  → 得到 num_external_computed_tokens 和 load_kv_async
  → 合并为 num_computed_tokens
  → 用 request.num_tokens - num_computed_tokens 计算剩余要 prefill 的 token
  → 分配 KV block
  → 必要时通知 connector 构造远端 KV load metadata
```

核心源码是：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)

ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)

num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

位置：`scheduler.py:761`、`scheduler.py:775`、`scheduler.py:797`

最终效果是：

```text
cache 命中越多，本轮要实际计算的 num_new_tokens 越少；
如果外部 KV 支持 async load，本轮甚至可以 num_new_tokens = 0，只发起远端 KV 加载。
```

---

## 2. 这个逻辑只在 waiting 调度阶段发生

running 请求已经进入执行流，它本轮要继续计算多少 token，主要看：

```python
request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
```

而 prefix cache / external KV cache 查询主要发生在 waiting 请求第一次进入 running 之前。

源码位置在 waiting 调度阶段：

```python
# Get already-cached tokens.
if request.num_computed_tokens == 0:
```

位置：`scheduler.py:723`

这里的含义是：

```text
只有当请求还没有已计算 token 时，才需要查询 prefix cache / external KV cache；
如果 request.num_computed_tokens 已经大于 0，说明这个请求已经有可用进度，
例如 async KV recv 完成后重新回来，就不再重复查询。
```

对应的 else 分支是：

```python
else:
    # KVTransfer: WAITING reqs have num_computed_tokens > 0
    # after async KV recvs are completed.
    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
    num_new_local_computed_tokens = 0
    num_computed_tokens = request.num_computed_tokens
```

位置：`scheduler.py:823`

也就是说：

```text
request.num_computed_tokens == 0：
  第一次调度，查询本地 / 外部 cache。

request.num_computed_tokens > 0：
  通常是 KVTransfer / async recv 完成后的请求，复用已有进度，不重复查 cache。
```

---

## 3. 本地 prefix cache 查询入口

普通本地 prefix cache 查询是：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

位置：`scheduler.py:761`

`KVCacheManager.get_computed_blocks()` 的定义是：

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:207`

它返回两个东西：

| 返回值 | 含义 |
|---|---|
| `new_computed_blocks` | 本地 prefix cache 命中的 KV blocks |
| `num_new_local_computed_tokens` | 本地 prefix cache 命中的 token 数 |

函数注释说：

```python
# A tuple containing:
# - A list of blocks that are computed for the request.
# - The number of computed tokens.
```

位置：`kv_cache_manager.py:215`

因此，本地 prefix cache 的回答是：

```text
这个请求的前缀中，有多少 token 的 KV 已经在本地 vLLM KV Cache 里，可以直接复用。
```

---

## 4. 什么时候不会查本地 prefix cache

`get_computed_blocks()` 里有一个前置判断：

```python
if not self.enable_caching or request.skip_reading_prefix_cache:
    return self.empty_kv_cache_blocks, 0
```

位置：`kv_cache_manager.py:223`

也就是说，以下情况本地 prefix cache 命中直接为 0：

```text
1. prefix caching 没启用；
2. request 被标记为 skip_reading_prefix_cache。
```

源码注释解释了 `skip_reading_prefix_cache` 的典型来源：

```python
# which happens when the request requires prompt logprobs
# or calls a pooling model with all pooling
```

位置：`kv_cache_manager.py:221`

所以即使 KV Cache 里可能存在相同前缀，如果该请求不能读 prefix cache，也会返回：

```text
new_computed_blocks = empty
num_new_local_computed_tokens = 0
```

---

## 5. 为什么 full cache hit 也要重算最后一个 token

本地 prefix cache 查询时，会设置：

```python
max_cache_hit_length = request.num_tokens - 1
```

位置：`kv_cache_manager.py:232`

注释解释：

```python
# When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
```

位置：`kv_cache_manager.py:226`

意思是：

```text
即使请求的全部 prompt KV 都在 cache 里，
Scheduler 也不能直接认为本轮完全不用算。
为了采样下一个 token，需要至少重新计算最后一个 token 的 logits。
```

所以本地 prefix cache 的最大命中长度通常是：

```text
request.num_tokens - 1
```

而不是：

```text
request.num_tokens
```

这也解释了为什么：

```text
一个看起来 100% 命中的 prompt，仍可能需要本地计算 1 个 token 或一个 block 对齐后的 token 范围。
```

源码注释还提到，由于 `allocate_slots()` 要求 `num_computed_tokens` block-size aligned，最后一个 token 的重算可能触发整个 block 的重算。

---

## 6. 本地 prefix cache 如何找最长命中

普通路径中，`get_computed_blocks()` 调用：

```python
computed_blocks, num_new_computed_tokens = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
```

位置：`kv_cache_manager.py:233`

这里的输入是：

| 参数 | 含义 |
|---|---|
| `request.block_hashes` | 请求 token 按 block 形成的 hash，用于 prefix cache 查找 |
| `max_cache_hit_length` | 最多允许命中的 token 长度，通常是 `request.num_tokens - 1` |

输出是：

```text
从请求开头开始，最长连续可复用的本地 KV blocks，
以及它们对应的 token 数。
```

注意是“prefix” cache，所以它关心的是从开头连续命中，而不是任意中间片段命中。

---

## 7. prefix cache stats 如何记录

如果开启 stats：

```python
if self.log_stats:
    self.prefix_cache_stats.record(
        num_tokens=request.num_tokens,
        num_hits=num_new_computed_tokens,
        preempted=request.num_preemptions > 0,
    )
```

位置：`kv_cache_manager.py:259`

Scheduler 侧在 hybrid 特殊路径中也会记录类似统计：

```python
self.kv_cache_manager.prefix_cache_stats.record(
    num_tokens=request.num_tokens,
    num_hits=num_new_local_computed_tokens,
    preempted=request.num_preemptions > 0,
)
```

位置：`scheduler.py:753`

这里统计的是：

```text
本地 prefix cache 查询了多少 token，命中了多少 token，
以及这个请求是否是被抢占后重新调度的请求。
```

`preempted` 很重要，因为被抢占请求重新调度时，可能通过 prefix cache 复用之前已经计算并缓存的 blocks。

如果启用了 KV Connector，并且之前算完的 KV 已经被外部 KV 池 / 远端 KV 系统保存，PREEMPTED 请求重新进入 waiting 调度时也会继续走 `get_num_new_matched_tokens()`。这时 connector 可以返回 `num_external_computed_tokens`，必要时触发 `load_kv_async=True`，先把远端 KV 加载回本地 blocks，再恢复调度。

因此抢占后的恢复不是只能重算：

```text
PREEMPTED / num_computed_tokens = 0
  → 查询本地 prefix cache
  → 查询外部 KV Connector
  → 命中则复用或异步加载 KV
  → 未命中部分再重新 prefill
```

---

## 8. Hybrid + Mamba + Connector 的特殊本地命中路径

Scheduler 中有一个特殊分支：

```python
if (
    self.connector is not None
    and self.has_mamba_layers
    and isinstance(
        self.kv_cache_manager.coordinator,
        HybridKVCacheCoordinator,
    )
):
    computed, per_group_hits = (
        self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
            request.block_hashes,
            request.num_tokens - 1,
        )
    )
```

位置：`scheduler.py:726`

这个路径用于 Hybrid + Mamba 模型，并且配置了 KV Connector 的场景。

这里不是只返回一个全局命中长度，而是返回每个 KV cache group 的命中：

```python
computed, per_group_hits
```

然后：

```python
num_new_local_computed_tokens = max(per_group_hits)
```

位置：`scheduler.py:752`

源码注释解释：

```python
# For Mamba hybrid models,
# num_new_local_computed_tokens should be the FA hit
# length. This value is passed to the connector's
# get_num_new_matched_tokens which computes:
# external = total - local_computed.
```

位置：`scheduler.py:743`

这里的关键点是：

```text
Hybrid/Mamba 下，不同 group 的 cache 命中可能不同；
Scheduler 要给 connector 一个本地已经命中的 token 数，
connector 再基于它计算外部还需要补多少。
```

新 commit 的注释还明确了这个值取 FA hit length 的原因：避免重新传输 D-side 已经缓存的 FA blocks；Mamba state 仍由 worker 侧 `_apply_prefix_caching` 无条件处理。

这属于特殊模型路径，普通 decoder-only full attention 模型一般走 `get_computed_blocks()`。

---

## 9. Mamba 下的 uncached common prefix hint

在 Hybrid / Mamba 场景中，Scheduler 还会读取一个 hint：

```python
if self.has_mamba_layers:
    num_uncached_common_prefix_tokens = getattr(
        self.kv_cache_manager.coordinator,
        "num_uncached_common_prefix_tokens",
        0,
    )
```

位置：`scheduler.py:766`

这个值后续传给：

```python
self._mamba_block_aligned_split(..., num_uncached_common_prefix_tokens)
```

位置：`scheduler.py:904`

它不是外部 KV 命中数，而是 Mamba / Marconi-style APC 逻辑使用的提示：

```text
如果存在足够长的尚未缓存公共前缀，
Scheduler 可以把本轮 chunk 切到公共前缀边界，
优先把公共前缀缓存起来。
```

因此它属于 prefix cache 调度优化的辅助信息。

---

## 10. 外部 KV cache 查询入口

如果配置了 KV Connector，Scheduler 会在本地 prefix cache 之后查询外部 KV cache：

```python
if self.connector is not None:
    ext_tokens, load_kv_async = (
        self.connector.get_num_new_matched_tokens(
            request, num_new_local_computed_tokens
        )
    )
```

位置：`scheduler.py:774`

这个接口返回两个值：

| 返回值 | 含义 |
|---|---|
| `ext_tokens` | 外部 KV cache 额外命中的 token 数；如果是 `None` 表示暂时无法确定 |
| `load_kv_async` | 是否通过异步方式从外部加载这些 KV |

注意，传给 connector 的第二个参数是：

```text
num_new_local_computed_tokens
```

也就是本地 prefix cache 已经命中的 token 数。

这非常关键：

```text
外部 KV cache 不应该重复计算本地已经命中的部分；
它返回的是相对本地命中之外的额外命中。
```

---

## 11. 外部命中数为什么叫 `num_external_computed_tokens`

Scheduler 拿到 connector 返回值后：

```python
num_external_computed_tokens = ext_tokens
```

位置：`scheduler.py:789`

之后合并：

```python
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

位置：`scheduler.py:797`

这里把外部 KV cache 命中的 token 也叫 computed tokens，因为从 Scheduler 的角度看：

```text
这些 token 的 KV 已经存在，不需要本轮本地模型 forward 再计算。
```

但它们和本地 prefix cache 有区别：

| 类型 | KV 在哪里 | 本轮是否需要本地计算 | 是否需要本地 block |
|---|---|---|---|
| 本地 prefix cache hit | 本地 vLLM KV Cache | 不需要 | 已有 cached blocks |
| 外部 KV cache hit | 外部 KV 系统 / CPU / 远端 | 不需要重新计算 | 需要分配本地 blocks 接收或映射 |
| 未命中 token | 没有 KV | 需要本地 forward | 需要新 blocks |

所以外部 KV 命中节省的是模型计算，但通常仍然会涉及本地 KV block 分配和 KV 传输。

---

## 12. 外部 KV 查询可能返回 `None`

connector 可以返回：

```python
ext_tokens is None
```

Scheduler 对应处理是：

```python
if ext_tokens is None:
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:781`

源码注释说：

```python
# the KVConnector couldn't determine
# the number of matched tokens.
```

位置：`scheduler.py:782`

含义：

```text
connector 暂时还不能回答这个请求外部命中了多少 token；
Scheduler 不会猜测，也不会阻塞整个 waiting 队列；
而是把请求临时放入 skipped_waiting，后续再问。
```

这类请求不一定是 blocked status。它可能仍然是：

```python
RequestStatus.WAITING
```

只是当前本轮因为外部 KV 查询不确定而被跳过。

---

## 13. 外部 KV 查询统计

如果 connector 返回了确定结果，Scheduler 会记录外部查询统计需要的两个值：

```python
connector_prefix_cache_queries = (
    request.num_tokens - num_new_local_computed_tokens
)
connector_prefix_cache_hits = num_external_computed_tokens
```

位置：`scheduler.py:791`

含义：

```text
外部 KV cache 查询范围 = request.num_tokens - 本地已命中 token 数；
外部 KV cache 命中数 = connector 返回的额外命中 token 数。
```

分配 block 成功后，如果统计对象存在，会记录：

```python
self.connector_prefix_cache_stats.record(
    num_tokens=connector_prefix_cache_queries,
    num_hits=connector_prefix_cache_hits,
    preempted=request.num_preemptions > 0,
)
```

位置：`scheduler.py:979`

这说明外部 KV cache 的统计是在：

```text
查询得到确定命中数，并且本轮 block 分配成功后记录。
```

---

## 14. 一个例子：本地与外部命中如何合并

假设：

```text
request.num_tokens = 10000
本地 prefix cache 命中 = 3000
外部 KV cache 总共可覆盖到 8000
```

Scheduler 先得到：

```text
num_new_local_computed_tokens = 3000
```

然后调用 connector：

```python
get_num_new_matched_tokens(request, 3000)
```

connector 应返回“相对本地额外命中”：

```text
num_external_computed_tokens = 5000
```

最终：

```text
num_computed_tokens = 3000 + 5000 = 8000
num_new_tokens = 10000 - 8000 = 2000
```

也就是说，本轮只需要本地 prefill 剩下 2000 token。

---

## 15. `num_computed_tokens` 必须不超过请求 token 数

合并本地和外部命中后，有一个断言：

```python
assert num_computed_tokens <= request.num_tokens
```

位置：`scheduler.py:800`

这保证：

```text
本地命中 + 外部命中，不能超过请求当前已有 token 总数。
```

如果 connector 返回的是“总命中数”而不是“外部额外命中数”，就可能导致：

```text
num_new_local_computed_tokens + num_external_computed_tokens > request.num_tokens
```

因此 connector 接口语义必须清楚：

```text
get_num_new_matched_tokens(request, local_hit)
返回的是 local_hit 之后额外可用的 token 数。
```

---

## 16. prefill stats 如何记录 local / external 命中

如果请求带有 prefill stats：

```python
if request.prefill_stats is not None:
    assert num_computed_tokens <= request.num_prompt_tokens
    request.prefill_stats.set(
        num_prompt_tokens=request.num_prompt_tokens,
        num_local_cached_tokens=num_new_local_computed_tokens,
        num_external_cached_tokens=num_external_computed_tokens,
    )
```

位置：`scheduler.py:814`

这说明 Scheduler 会把三类信息记录到请求统计里：

```text
prompt 总 token 数；
本地 prefix cache 命中的 token 数；
外部 KV cache 命中的 token 数。
```

注释特别说明：

```python
# Track first scheduled prefill, not post-preemption repeat prefills
```

位置：`scheduler.py:814`

也就是说，这里的 prefill stats 更关注第一次进入 prefill 时的 cache 命中，而不是每次被抢占后重复 prefill 的命中。

---

## 17. 命中结果如何影响本轮 `num_new_tokens`

普通 waiting 请求会用：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`scheduler.py:847`

所以：

```text
本地 / 外部命中越多，num_computed_tokens 越大；
num_computed_tokens 越大，本轮要实际计算的 num_new_tokens 越小。
```

几个例子：

```text
prompt = 10000
无缓存命中：
  num_computed_tokens = 0
  num_new_tokens = 10000

本地命中 3000：
  num_computed_tokens = 3000
  num_new_tokens = 7000

本地命中 3000，外部额外命中 5000：
  num_computed_tokens = 8000
  num_new_tokens = 2000
```

这就是 KVPool / 外部 KV cache 对 Scheduler 的核心价值：

```text
把原本要消耗 GPU forward 的 token，转换成已经 computed 的 token。
```

---

## 18. external KV async load 时 `num_new_tokens = 0`

如果 connector 返回：

```text
load_kv_async = True
```

Scheduler 会走特殊路径：

```python
if load_kv_async:
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
```

位置：`scheduler.py:834`

含义：

```text
本轮不做本地模型 forward；
本轮只负责为外部 KV load 分配本地 block，
并让 connector / worker 后续把远端 KV 加载进来。
```

这时虽然不消耗 token budget 做计算，但仍然需要分配 KV blocks。

新 commit 里 async KV load 还会临时禁用 speculative lookahead slots，避免本地和远端 block 计数不一致；并在分配前用 `_inflight_prefill_reserved_blocks()` 计算其他 in-flight prefill 需要保留的 blocks，作为 `reserved_blocks` 传入分配逻辑，避免 async load 占满可用空间。

后续分配时会传：

```python
delay_cache_blocks=load_kv_async
```

位置：`scheduler.py:949`

表示：

```text
这些 blocks 是给异步 KV transfer 用的，暂时延迟正式 cache。
```

---

## 19. external KV async load 不会直接进入 running

async load 分配 block 成功后：

```python
request = request_queue.pop_request()
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens
    self._inflight_prefills.add(request)
    continue
```

位置：`scheduler.py:985`

也就是说：

```text
外部 KV async load 请求不会 append 到 self.running；
它会进入 WAITING_FOR_REMOTE_KVS，并被放回 skipped_waiting。
```

它的状态迁移是：

```text
WAITING
  → 本地/外部 cache 查询
  → load_kv_async=True
  → allocate_slots(delay_cache_blocks=True)
  → WAITING_FOR_REMOTE_KVS
  → skipped_waiting
```

Worker 完成加载后，会在后续输出中报告 `finished_recving`，Scheduler 再把请求状态恢复成 `WAITING` 或 `PREEMPTED`。这里恢复的是 `request.status`；队列位置不一定立即搬回 `self.waiting`，可能仍从 `skipped_waiting` 队头继续尝试调度。

---

## 20. async KV load 完成后如何更新命中结果

当 async KV load 完成后，Scheduler 会调用：

```python
_update_waiting_for_remote_kv(request)
```

位置：`scheduler.py:2492`

这个函数里会真正 cache blocks：

```python
self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
```

位置：`scheduler.py:2517`

如果完整 prompt 都从远端命中，还要回退一个 token：

```python
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

位置：`scheduler.py:2521`

原因和本地 full prefix hit 一样：

```text
即使远端 KV 覆盖了整个 prompt，
为了采样下一个 token，也需要重新计算最后一个 token 的 logits。
```

如果 KV load 失败，则会根据已成功加载的 token 数修正 `request.num_computed_tokens`，并只缓存有效部分。

---

## 21. 命中结果如何传入 `allocate_slots()`

无论是本地命中还是外部命中，最终都要参与 KV block 分配。

waiting 请求调用：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    ...
)
```

位置：`scheduler.py:942`

这些参数的含义是：

| 参数 | 含义 |
|---|---|
| `num_new_tokens` | 本轮需要本地模型实际计算的 token 数 |
| `num_new_computed_tokens` | 本地 prefix cache 命中的 token 数 |
| `new_computed_blocks` | 本地 prefix cache 命中的 KV blocks |
| `num_external_computed_tokens` | 外部 KV cache 额外命中的 token 数 |
| `delay_cache_blocks` | 是否异步加载外部 KV，延迟正式 cache blocks |

所以 `allocate_slots()` 不是只为未命中 token 分配 block。它还要把：

```text
本地已命中的 blocks 接到请求上；
为外部命中的 token 分配接收 blocks；
为本轮新计算 token 分配新 blocks。
```

---

## 22. local / external blocks 在 KVCacheManager 里的处理

`KVCacheManager.allocate_slots()` 会把命中信息交给 coordinator。

coordinator 中有：

```python
def allocate_new_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
```

位置：`vllm/vllm/v1/core/kv_cache_coordinator.py:192`

这个函数会先处理本地 cache-hit blocks：

```python
manager.add_local_computed_blocks(...)
```

位置：`kv_cache_coordinator.py:223`

如果外部命中数大于 0，再分配外部 computed tokens 对应的 blocks：

```python
if num_external_computed_tokens > 0:
    for manager in self.single_type_managers:
        manager.allocate_external_computed_blocks(...)
```

位置：`kv_cache_coordinator.py:230`

注释说明这是 two-phase allocation：

```text
先 touch 每个 group 的本地 cache-hit blocks，
再为每个 group 分配 external blocks，
避免某个 group 分配 external blocks 时把另一个 group 还没 touch 的 cache-hit blocks 驱逐掉。
```

这说明 local prefix hit 和 external KV hit 在 block 层面不是简单加数字，而是要非常小心地维护 block 引用和驱逐顺序。

---

## 23. 本地命中 blocks 如何加入请求

单个 KV cache manager 里，本地命中处理函数是：

```python
def add_local_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: Sequence[KVCacheBlock],
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
```

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:223`

它会做几件事：

```text
1. 计算 sliding window / Mamba 等场景中需要跳过的 token/block；
2. touch 本地命中的 blocks，避免被 evict；
3. 对 skipped blocks 填 null block；
4. 把剩余本地命中 blocks 接到 req_blocks；
5. 更新 num_cached_block，避免后续重复 cache；
6. 如果命中落在 block 中间，记录 partial-hit 请求，后续 allocate_new_blocks 会用私有 CoW block 重定向共享尾块。
```

关键代码包括：

```python
self.block_pool.touch(new_computed_blocks)
req_blocks.extend([self._null_block] * num_skipped_blocks)
req_blocks.extend(new_computed_blocks)
self.num_cached_block[request_id] = len(req_blocks)
```

位置：`single_type_kv_cache_manager.py:258`

所以本地 prefix cache hit 不是复制 KV，而是：

```text
让当前请求引用已有 cache blocks，并增加/维护这些 blocks 的引用状态。
```

---

## 24. 外部命中 token 如何分配 blocks

外部 KV 命中的 token 不在本地 KV block pool 里，因此需要为它们分配新的本地 blocks。

函数是：

```python
def allocate_external_computed_blocks(
    self,
    request_id: str,
    num_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> None:
```

位置：`single_type_kv_cache_manager.py:282`

它会计算总 computed tokens：

```python
num_total_computed_tokens = (
    num_local_computed_tokens + num_external_computed_tokens
)
```

位置：`single_type_kv_cache_manager.py:300`

然后为外部 computed tokens 对应的 block 范围分配新 blocks：

```python
allocated_blocks = self.block_pool.get_new_blocks(
    cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
)
req_blocks.extend(allocated_blocks)
```

位置：`single_type_kv_cache_manager.py:314`

含义：

```text
本地 prefix hit 使用已有 cached blocks；
外部 KV hit 需要本地新 blocks 承接远端 KV；
这些 blocks 后续由 connector / worker load KV，或者在同步路径中变成可用 KV。
```

如果当前 manager 需要记录新 block id，外部分配出来的新 block id 会追加到 `new_block_ids`，后续可用于需要 zeroing 的新 block 处理：

```python
if self._record_new_block_ids:
    self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```

位置：`single_type_kv_cache_manager.py:318` 到 `single_type_kv_cache_manager.py:319`

---

## 25. 分配后为什么要调用 `connector.update_state_after_alloc()`

block 分配成功后，Scheduler 会通知 connector：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

位置：`scheduler.py:970`

这一步的作用是：

```text
Scheduler 已经为请求分配好了本地 KV blocks；
connector 现在可以知道：
  外部命中的 token 要加载到哪些本地 block 里。
```

例如 offloading connector 的接口说明中，`get_num_new_matched_tokens()` 返回的是可以在本地 computed tokens 之后继续 load 的 token 数：

```python
# Get number of new tokens that can be loaded beyond the
# num_computed_tokens.
```

位置：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:696`

而 `update_state_after_alloc()` 会根据 `blocks` 和 `num_external_tokens` 构造后续 load 需要的源 key、目标 block id 等信息。

所以外部 KV cache 的完整链路不是只返回命中数：

```text
get_num_new_matched_tokens()
  → 回答命中了多少 token、是否 async load

allocate_slots()
  → 为这些 token 分配本地 blocks

update_state_after_alloc()
  → 告诉 connector 外部 KV 应该加载到哪些 blocks

build_connector_meta()
  → 把 load metadata 放进 SchedulerOutput 给 Worker
```

---

## 26. SimpleCPUOffload 的例子

以 `SimpleCPUOffload` 为例，connector scheduler 的查询函数是：

```python
def get_num_new_matched_tokens(
    self, request: "Request", num_computed_tokens: int
) -> tuple[int | None, bool]:
```

位置：`vllm/vllm/v1/simple_kv_offload/manager.py:243`

它会根据本地已经 computed 的 token 数跳过前面的 hashes：

```python
num_skipped_hashes = num_computed_tokens // self.hash_block_size
remaining_hashes = request.block_hashes[num_skipped_hashes:]
```

位置：`manager.py:255`

然后设置最大命中长度：

```python
max_hit_len = request.num_tokens - 1 - num_computed_tokens
```

位置：`manager.py:262`

这和本地 prefix cache 的逻辑一致：即使全命中，也要保留最后 token 重新计算 logits。

如果 CPU cache 命中：

```python
cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
    remaining_hashes, max_hit_len
)
```

位置：`manager.py:265`

命中后会 pin 住 CPU blocks，避免在 `update_state_after_alloc()` 前被 LRU 回收：

```python
self.cpu_block_pool.touch(pin_blocks)
self._pending_cpu_hits[request.request_id] = (
    cpu_hit_blocks,
    hit_length,
)
return hit_length, True
```

位置：`manager.py:273`

这个例子说明：

```text
外部 KV cache 查询不仅是算命中数，
还可能临时 pin 住远端 / CPU cache blocks，
等 Scheduler 分配好 GPU blocks 后再真正生成 load 计划。
```

---

## 27. OffloadingConnector 的例子

Offloading connector 的 scheduler 接口是：

```python
def get_num_new_matched_tokens(
    self, request: Request, num_computed_tokens: int
) -> tuple[int | None, bool]:
```

位置：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:692`

注释说明返回值：

```text
- 可以在 already computed 之外额外加载的 token 数；
- 如果是 None，说明 connector 还需要更多时间判断命中；
- 如果第二个值是 True，表示 token 会异步加载。
```

对应源码：`offloading/scheduler.py:704`

这个实现里，`None` 的常见来源包括：请求还有 in-flight transfer 时直接延迟返回，或者 `_lookup()` 因 backend deferred lookup / hit blocks 正在加载而暂时无法收敛。

它会保存本地 computed token 数：

```python
req_status.num_locally_computed_tokens = num_computed_tokens
```

位置：`offloading/scheduler.py:726`

然后查询外部命中；如果请求被标记为 `skip_reading_prefix_cache`，则直接视为 0 命中：

```python
if request.skip_reading_prefix_cache:
    num_hit_tokens = 0
else:
    num_hit_tokens = self._lookup(req_status)
```

位置：`offloading/scheduler.py:729` 到 `offloading/scheduler.py:733`

最后返回：

```python
return num_hit_tokens, bool(num_hit_tokens)
```

位置：`offloading/scheduler.py:747`

也就是说，在这个实现里，只要有外部命中，就会走 async load。

---

## 28. full prompt 外部命中为什么还要回退一个 token

外部 async load 完成后，如果：

```python
request.num_computed_tokens == request.num_tokens
```

会执行：

```python
request.num_computed_tokens = request.num_tokens - 1
```

位置：`scheduler.py:2521`

这和本地 prefix cache 的：

```python
max_cache_hit_length = request.num_tokens - 1
```

位置：`kv_cache_manager.py:232`

是一致的。

原因都是：

```text
为了生成下一个 token，模型需要最后一个输入位置的 logits；
即使 KV 全部命中，也必须重新计算最后一个 token。
```

所以：

```text
full hit 不等于本地 forward 完全为 0；
在进入真正 decode 前，通常还需要至少补算最后一个 token。
```

---

## 29. 对 token budget 的影响

cache 命中最终影响的是本轮 `num_new_tokens`。

普通路径：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`scheduler.py:847`

所以：

```text
本地 prefix cache hit：减少本轮需要 prefill 的 token；
外部 KV cache hit：进一步减少本轮需要 prefill 的 token；
async external KV load：本轮本地计算 token 可以是 0。
```

但要注意：

```text
不消耗 token budget 不等于不消耗资源。
```

外部 KV 命中仍可能消耗：

```text
本地 KV blocks；
KV transfer 带宽；
connector metadata；
异步 load 的 in-flight 状态；
reserved blocks。
```

因此，prefix / external KV cache 的收益主要是减少 GPU forward 计算量，但它仍然受 KV block 容量和 connector 状态约束。

---

## 30. 对请求状态的影响

不同命中路径对请求状态的影响不同。

### 30.1 只有本地 prefix cache / 同步外部命中

如果不是 async load，并且 block 分配成功：

```text
request.status: WAITING / PREEMPTED → RUNNING
```

进入 running 时：

```python
request.num_computed_tokens = num_computed_tokens
```

位置：`scheduler.py:1028`

然后 `_update_after_schedule()` 再加上本轮调度 token。

### 30.2 async external KV load

如果 `load_kv_async=True`：

```text
request.status: WAITING / PREEMPTED → WAITING_FOR_REMOTE_KVS
```

位置：`scheduler.py:989`

请求不会进入 running，而是放入 skipped_waiting，等待 Worker 报告 KV load 完成。

---

## 31. 对 SchedulerOutput 的影响

prefix / external KV 命中结果会影响多个输出字段。

普通进入 running 的请求会影响：

```text
num_scheduled_tokens[request_id]
req_to_new_blocks[request_id]
scheduled_new_reqs / scheduled_resumed_reqs
scheduled_cached_reqs
```

外部 KV connector 还会影响：

```python
scheduler_output.kv_connector_metadata = meta
```

位置：`scheduler.py:1168`

这个 metadata 来自：

```python
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
```

位置：`scheduler.py:1167`

作用是：

```text
把本轮需要 load / save 的 KV transfer 信息传给 Worker。
```

---

## 32. 一个完整例子：没有任何 cache 命中

假设：

```text
request.num_tokens = 10000
本地 prefix cache 命中 = 0
没有 KV Connector
```

则：

```text
num_new_local_computed_tokens = 0
num_external_computed_tokens = 0
num_computed_tokens = 0
num_new_tokens = 10000
```

如果开启 chunked prefill 且 token budget 为 4096：

```text
本轮 num_new_tokens = 4096
```

请求进入 running 后：

```text
request.num_computed_tokens = 0
_update_after_schedule 后变成 4096
```

---

## 33. 一个完整例子：本地 prefix cache 部分命中

假设：

```text
request.num_tokens = 10000
本地 prefix cache 命中 = 4096
没有外部 KV cache
```

则：

```text
num_new_local_computed_tokens = 4096
num_external_computed_tokens = 0
num_computed_tokens = 4096
num_new_tokens = 10000 - 4096 = 5904
```

如果 token budget 足够，本轮只需要计算 5904 token。

如果 token budget 只有 2048 且开启 chunked prefill：

```text
本轮只计算 2048 token；
调度后 num_computed_tokens = 4096 + 2048 = 6144。
```

---

## 34. 一个完整例子：本地 + 外部 KV 命中

假设：

```text
request.num_tokens = 16000
本地 prefix cache 命中 = 4000
外部 KV cache 额外命中 = 8000
```

则：

```text
num_computed_tokens = 4000 + 8000 = 12000
num_new_tokens = 16000 - 12000 = 4000
```

相比没有外部 KV cache：

```text
只靠本地命中：需要计算 12000 token；
本地 + 外部命中：只需要计算 4000 token。
```

这就是外部 KV cache 对 prefill 计算量的直接削减。

---

## 35. 一个完整例子：外部 KV async load

假设：

```text
request.num_tokens = 10000
本地 prefix cache 命中 = 0
外部 KV cache 额外命中 = 9999
load_kv_async = True
```

则：

```text
num_external_computed_tokens = 9999
num_computed_tokens = 9999
num_new_tokens = 0
```

Scheduler 会：

```text
1. allocate_slots(..., delay_cache_blocks=True)；
2. connector.update_state_after_alloc(...)；
3. request.status = WAITING_FOR_REMOTE_KVS；
4. request.num_computed_tokens = 9999；
5. 请求进入 skipped_waiting；
6. 本轮不进入 running。
```

等 Worker load 完成后，下一轮再恢复调度，并通常会补算最后一个 token 以产生 logits。

---

## 36. 容易疑惑的点

### 36.1 外部 KV cache 返回的是总命中数吗？

不是。

Scheduler 传给 connector 的参数是本地已经命中的 token 数：

```python
get_num_new_matched_tokens(request, num_new_local_computed_tokens)
```

connector 应返回的是：

```text
在本地命中之后，外部还能额外提供多少 token。
```

最终才相加：

```text
num_computed_tokens = local_hit + external_hit
```

### 36.2 full prefix cache hit 后为什么还要计算？

因为要采样下一个 token，需要最后一个 token 位置的 logits。

所以本地 prefix cache 查询限制为：

```text
request.num_tokens - 1
```

async 外部 KV full hit 完成后也会把：

```text
num_computed_tokens 从 request.num_tokens 回退到 request.num_tokens - 1。
```

### 36.3 外部 KV 命中是否消耗 token budget？

外部 KV 命中的 token 不需要本地 forward，因此不消耗 `num_new_tokens` 对应的计算预算。

但是它可能消耗：

```text
KV block 容量；
KV transfer 时间；
connector metadata；
异步 load 状态。
```

### 36.4 `ext_tokens is None` 和 `ext_tokens = 0` 有什么区别？

区别很大。

```text
ext_tokens = 0：
  connector 已经确定没有外部命中，可以继续按普通路径调度。

ext_tokens is None：
  connector 暂时无法确定命中数，Scheduler 会跳过该请求，后续再问。
```

### 36.5 async KV load 为什么 `num_new_tokens = 0`？

因为本轮不做本地模型 forward，只启动远端 KV 加载。

```text
num_new_tokens = 0
```

表示本轮没有本地计算 token，但仍会为 external computed tokens 分配 block。

### 36.6 被抢占请求会重新利用 prefix cache 吗？

会。

被抢占请求通常会把 `num_computed_tokens` 重置为 0，重新进入 waiting 后会再次走 prefix / external KV 查询。

如果之前计算的 blocks 已经被 cache，后续可以通过本地 prefix cache 命中复用。

---

## 37. 从“回答问题”的角度总结

如果要问：

```text
prefix cache / 外部 KV cache 命中了多少 token？
```

Scheduler 的回答是：

```python
num_new_local_computed_tokens
num_external_computed_tokens
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

其中：

```text
num_new_local_computed_tokens：
  本地 vLLM prefix cache 命中的 token 数。

num_external_computed_tokens：
  KV Connector 报告的、在本地命中之后外部额外可用的 token 数。

num_computed_tokens：
  Scheduler 认为本轮调度前已经可复用的 token 总数。
```

普通 waiting 请求剩余要计算：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

如果 `load_kv_async=True`，则：

```python
num_new_tokens = 0
request.status = WAITING_FOR_REMOTE_KVS
```

也就是说：

```text
本地 / 外部 cache 命中决定了 Scheduler 还需要计算多少 token；
async external KV load 则把“计算”变成“加载”，请求先进入 WAITING_FOR_REMOTE_KVS。
```

---

## 38. 最关键的判断公式

```text
本地 prefix cache 查询：
  new_computed_blocks, num_new_local_computed_tokens =
      kv_cache_manager.get_computed_blocks(request)

外部 KV cache 查询：
  ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(
      request,
      num_new_local_computed_tokens,
  )

外部查询不确定：
  if ext_tokens is None:
      move request to step_skipped_waiting
      continue

合并命中：
  num_external_computed_tokens = ext_tokens
  num_computed_tokens = (
      num_new_local_computed_tokens
      + num_external_computed_tokens
  )

普通本地计算路径：
  num_new_tokens = request.num_tokens - num_computed_tokens

async external KV load：
  if load_kv_async:
      num_new_tokens = 0
      allocate_slots(..., delay_cache_blocks=True)
      request.status = WAITING_FOR_REMOTE_KVS
      step_skipped_waiting.prepend_request(request)

进入 running 前：
  request.num_computed_tokens = num_computed_tokens

调度后：
  request.num_computed_tokens += num_scheduled_tokens[request_id]
```

---

## 39. 和前后问题的关系

`04_waiting_to_running.md` 解释了：

```text
waiting 请求如何经过一系列检查后进入 running。
```

本篇解释的是其中最关键的 cache 命中子流程：

```text
进入 running 前，Scheduler 如何判断已有多少 token 不需要再算。
```

接下来 `06_kv_block_allocation_and_preemption.md` 可以继续深入：

```text
拿到 prefix / external KV 命中数和 num_new_tokens 后，
KV Cache block 到底如何分配；
如果 block 不够，什么时候抢占，什么时候停止调度。
```
