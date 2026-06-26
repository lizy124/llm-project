# 04. Scheduler 如何接入 KV Connector？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\factory.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\mooncake\store\scheduler.py`

本问题关注：Scheduler 侧 KV Connector 在 vLLM V1 中如何接入调度流程。重点回答：Scheduler 什么时候创建 connector，waiting 请求如何查询外部 KV 命中，外部命中如何和本地 prefix cache 命中合并，`allocate_slots()` 后为什么要调用 `update_state_after_alloc()`，`build_connector_meta()` 如何进入 `SchedulerOutput`，以及 Worker 回传的 KV transfer 完成状态如何回到 Scheduler。

---

## 0. 梳理规划

本篇按“先定角色，再走 waiting 调度主线，再看 metadata 和回收”的方式梳理 Scheduler 侧 KV Connector。

要回答的问题分成 10 组：

```text
1. KV Connector 是什么层？Scheduler 侧和 Worker 侧为什么分开？
2. Scheduler 什么时候创建 KV Connector？
3. 新请求进入 Scheduler 时 connector 做什么？
4. waiting 请求如何查询本地 prefix cache 和外部 KV cache？
5. get_num_new_matched_tokens() 的返回值是什么意思？
6. external KV hit 如何影响 num_computed_tokens / num_new_tokens？
7. allocate_slots() 后 update_state_after_alloc() 为什么必要？
8. build_connector_meta() 如何写入 SchedulerOutput？
9. Worker 侧 finished_recving / finished_sending 如何回到 Scheduler？
10. 从 KVPool / MooncakeStore 视角看完整链路是什么？
```

阅读顺序建议：

```text
01_kv_cache_manager_role.md
  → 03_prefix_cache_lookup.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 06_external_kv_save_flow.md
  → 07_worker_kv_connector_flow.md
  → 08_invalid_blocks_and_recompute.md
```

本篇重点讲 Scheduler 侧，不深入 Worker 侧如何真正 load / save KV；Worker 侧在 `07_worker_kv_connector_flow.md` 细讲。

---

## 1. 一句话回答

Scheduler 侧 KV Connector 负责把 **外部 KV Cache / KVPool 的命中、加载、保存意图** 转成 vLLM Scheduler 能理解的调度状态和 `SchedulerOutput.kv_connector_metadata`。

它负责：

```text
1. 在新请求进入 Scheduler 时记录 connector 侧状态；
2. 在 waiting 请求调度时查询外部 KV cache 命中多少 token；
3. 告诉 Scheduler 这些外部命中 token 是否需要异步加载；
4. 在 KVCacheManager 分配本地 block 后，记录 external KV 应该 load 到哪些 block；
5. 在 schedule() 末尾构造 KVConnectorMetadata，放入 SchedulerOutput；
6. 在请求结束时决定是否需要把本地 KV save 到外部 KV cache；
7. 在 update_from_output() 阶段接收 Worker 侧 finished_recving / finished_sending / invalid blocks 状态。
```

它不负责：

```text
1. 不负责本地 prefix cache 查询，这是 KVCacheManager 的事；
2. 不负责本地 block 分配，这是 KVCacheManager / BlockPool 的事；
3. 不负责真正 GPU KV tensor load / save，这是 Worker 侧 connector 的事；
4. 不负责 token budget、waiting / running 队列、preemption，这是 Scheduler 主逻辑的事；
5. 不负责构造 attention metadata / slot mapping，这是 ModelRunner 的事。
```

可以把它理解成：

```text
Scheduler 侧 connector = 外部 KV 系统在 Scheduler 调度阶段的代理。
```

---

## 2. 一句话总览链路

waiting 请求接入外部 KV 的主线是：

```text
waiting request
  → KVCacheManager.get_computed_blocks()
  → 得到本地 prefix cache hit
  → connector.get_num_new_matched_tokens(request, local_hit_tokens)
  → 得到 external hit tokens 和 load_kv_async
  → num_computed_tokens = local_hit + external_hit
  → num_new_tokens = request.num_tokens - num_computed_tokens
  → KVCacheManager.allocate_slots(... num_external_computed_tokens ...)
  → connector.update_state_after_alloc(request, blocks, external_hit)
  → connector.build_connector_meta(scheduler_output)
  → SchedulerOutput.kv_connector_metadata
  → Worker / ModelRunner 执行 KV load / save
  → ModelRunnerOutput.kv_connector_output
  → Scheduler.update_from_output()
```

---

## 3. KV Connector 为什么分 Scheduler 侧和 Worker 侧

`KVConnectorFactory.create_connector()` 的注释说明 v1 connector 明确分成两个角色：

```python
# Scheduler connector:
# - Co-locate with scheduler process
# - Should only be used inside the Scheduler class
# Worker connector:
# - Co-locate with worker process
# - Should only be used inside the forward context & attention layer
# We build separately to enforce strict separation
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:67`

角色枚举定义：

```python
class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:124`

这说明：

```text
Scheduler 侧 connector：参与调度决策，构造 metadata。
Worker 侧 connector：消费 metadata，真正执行 KV load / save。
```

二者通过：

```text
KVConnectorMetadata
```

通信。

---

## 4. KVConnectorMetadata 是什么

抽象定义：

```python
class KVConnectorMetadata(ABC):
    """
    Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:141`

它的作用是：

```text
把 Scheduler 侧 connector 在本轮 schedule() 中整理出的 load / save / preemption / block mapping 信息，传给 Worker 侧 connector。
```

`SchedulerOutput` 中有字段：

```python
# KV Cache Connector metadata.
kv_connector_metadata: KVConnectorMetadata | None = None
```

位置：`code/vllm/vllm/v1/core/sched/output.py:232`

所以：

```text
SchedulerOutput 不只告诉 Worker 跑哪些 token，也携带 KV transfer 本轮要做什么。
```

---

## 5. Scheduler 初始化时创建 connector

`Scheduler.__init__()` 中：

```python
self.connector = None
self.connector_prefix_cache_stats: PrefixCacheStats | None = None
self.recompute_kv_load_failures = True
self.defer_block_free = False
kv_transfer_config = self.vllm_config.kv_transfer_config
if kv_transfer_config is not None:
    assert not self.is_encoder_decoder, (
        "Encoder-decoder models are not currently supported with KV connectors"
    )
    self.connector = KVConnectorFactory.create_connector(
        config=self.vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=self.kv_cache_config,
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:123`

几个关键点：

```text
1. 只有配置了 kv_transfer_config 才创建 connector；
2. Scheduler 侧创建 role=KVConnectorRole.SCHEDULER 的 connector；
3. 当前 encoder-decoder 模型不支持 KV connector；
4. connector 使用 vllm_config 和 kv_cache_config 初始化。
```

---

## 6. create_connector 如何选具体实现

工厂入口：

```python
KVConnectorFactory.create_connector(
    config,
    role,
    kv_cache_config,
)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:43`

它从配置里拿：

```python
kv_transfer_config = config.kv_transfer_config
connector_cls = cls.get_connector_class(kv_transfer_config)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:49`

connector 名来自：

```python
connector_name = kv_transfer_config.kv_connector
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:99`

支持两类加载方式：

```text
1. kv_connector_module_path 指定外部模块路径；
2. 内置 registry 中注册的 connector 名称。
```

内置注册包括：

```text
ExampleConnector
LMCacheConnectorV1
LMCacheMPConnector
NixlConnector
NixlPullConnector
NixlPushConnector
MultiConnector
OffloadingConnector
MooncakeConnector
MooncakeStoreConnector
FlexKVConnectorV1
SimpleCPUOffloadConnector
HF3FSKVConnector
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:152`

因此：

```text
Scheduler 不关心具体是 Mooncake、NIXL、LMCache、Offloading 还是别的 connector；
它只依赖 KVConnectorBase_V1 的统一接口。
```

---

## 7. HMA 检查和 block_pool 绑定

`create_connector()` 还会检查 connector 是否支持 HMA：

```python
hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
if hma_enabled and not cls.supports_hma_config(kv_transfer_config):
    raise ValueError(...)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:55`

Scheduler 创建 `KVCacheManager` 后，会把 block pool 绑定给 connector：

```python
if self.connector is not None:
    self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:270`

抽象接口说明：

```python
def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
    """
    Bind the GPU block pool to the connector for per-GPU block status tracking.
    For example, inc/dec ref counts, or iterate over the prefix cache blocks.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:443`

这表示 connector 有时需要观察或维护本地 block 状态，例如：

```text
外部 offloading；
异步 save；
block ref count 保护；
prefix cache block 遍历；
invalid block evict。
```

---

## 8. KV load failure policy 和 deferred free

Scheduler 初始化 connector 时还处理两个重要运行策略。

### 8.1 load failure policy

```python
kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:142`

含义：

```text
外部 KV load 失败时，是回退重算，还是直接失败请求。
```

这会在 `08_invalid_blocks_and_recompute.md` 里细讲。

### 8.2 deferred free

```python
multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
    self.defer_block_free = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:149`

源码注释解释：

```text
With overlapping batches (async scheduling or PP), a step may still be writing a freed request's KV blocks.
A consumer KV Connector can reallocate and fill those blocks via a load that isn't ordered against that write, so defer freeing them.
```

也就是说：

```text
如果有多个 in-flight batch，并且当前是 KV consumer，block 不能随便立即复用，否则可能和异步 KV load / forward 写入产生竞态。
```

这会在 `09_deferred_free_and_async_safety.md` 里细讲。

---

## 9. 新请求进入 Scheduler 时通知 connector

新请求入口在 `Scheduler.add_request()`。

当 request id 不存在，是全新请求时：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
if self.connector is not None:
    self.connector.on_new_request(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1976`

抽象接口：

```python
def on_new_request(self, request: "Request") -> None:
    """Called by the scheduler when a new request is added.

    Connectors can override this to inspect the request and perform
    bookkeeping. The default implementation is a no-op.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:524`

这一步的含义是：

```text
Scheduler 把请求放进 waiting 队列后，connector 可以提前读取 request 信息，初始化外部 KV lookup / save / load 所需状态。
```

不是所有 connector 都需要重写它；默认是 no-op。

---

## 10. waiting 阶段：先查本地 prefix cache

KV Connector 只参与 waiting 请求调度阶段，准确地说是：

```text
request.num_computed_tokens == 0
```

的初始调度路径。

Scheduler 先查本地 prefix cache：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:709`

这一步得到：

```text
num_new_local_computed_tokens：vLLM 本地 prefix cache 已经命中的 token 数。
new_computed_blocks：本地 prefix cache 命中的 block。
```

之后才进入外部 KV Connector 查询。

为什么先查本地？

```text
因为本地 prefix cache 已经在 GPU block pool 里，通常比外部 KV load 更直接；
外部 connector 只需要返回“本地之外额外命中了多少 token”。
```

---

## 11. waiting 阶段：查询外部 KV cache

如果配置了 connector：

```python
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request, num_new_local_computed_tokens
    )
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:723`

抽象接口定义：

```python
def get_num_new_matched_tokens(
    self,
    request: "Request",
    num_computed_tokens: int,
) -> tuple[int | None, bool]:
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:453`

接口注释说明它返回：

```text
1. 外部 KV cache 中，超过 num_computed_tokens 之外还能加载的新 token 数；
2. 是否异步加载 external KV tokens。
```

更准确地说：

```text
num_computed_tokens 参数 = 本地 prefix cache 已命中的 token 数。
返回的 ext_tokens = 外部 KV cache 在本地命中之外额外命中的 token 数。
```

---

## 12. get_num_new_matched_tokens 的返回语义

返回类型：

```python
tuple[int | None, bool]
```

### 12.1 返回 `(0, False)`

表示：

```text
外部 KV cache 没有额外命中；
本轮正常走本地 prefill / decode；
```

### 12.2 返回 `(N, False)`，N > 0

表示：

```text
外部 KV cache 额外命中了 N 个 token；
这 N 个 token 可以作为外部已计算 tokens；
KV load 可能在本轮 forward 路径中同步或由 Worker 侧 metadata 处理；
Scheduler 不把请求放入 WAITING_FOR_REMOTE_KVS。
```

具体同步 / 异步行为取决于 connector 实现。

### 12.3 返回 `(N, True)`，N > 0

表示：

```text
外部 KV cache 额外命中了 N 个 token；
这些 KV 会在 scheduler step 之间异步加载；
本轮不做本地 forward；
请求进入 WAITING_FOR_REMOTE_KVS。
```

Scheduler 对这个分支有专门处理：

```python
if load_kv_async:
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:781`

### 12.4 返回 `(None, bool)`

接口注释说明：

```text
If None, it means that the connector needs more time to determine the number of matched tokens, and the scheduler should query for this request again later.
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:469`

Scheduler 处理方式：

```python
if ext_tokens is None:
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:729`

也就是：

```text
本轮暂时跳过该请求，放入 skipped_waiting，下一轮再问 connector。
```

---

## 13. external hit 如何和 local hit 合并

connector 返回后：

```python
num_external_computed_tokens = ext_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:737`

然后合并：

```python
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
assert num_computed_tokens <= request.num_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:744`

含义是：

```text
Scheduler 把本地 prefix cache 命中和外部 KV cache 命中统一看作“本轮不需要本地 forward 的 token”。
```

例如：

```text
prompt = 10000 tokens
local prefix hit = 3000 tokens
external KV hit beyond local = 5000 tokens
num_computed_tokens = 3000 + 5000 = 8000
num_new_tokens = 10000 - 8000 = 2000
```

注意：

```text
external hit 不是总命中数，而是本地命中之外的新增命中数。
```

有些 connector 内部可能先查到“外部总命中到多少 token”，然后减掉传入的 `num_computed_tokens`，返回新增部分。

---

## 14. connector prefix cache stats

Scheduler 会记录 connector 查询统计：

```python
connector_prefix_cache_queries = (
    request.num_tokens - num_new_local_computed_tokens
)
connector_prefix_cache_hits = num_external_computed_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:739`

分配成功后，如果启用了 stats：

```python
self.connector_prefix_cache_stats.record(
    num_tokens=connector_prefix_cache_queries,
    num_hits=connector_prefix_cache_hits,
    preempted=request.num_preemptions > 0,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:910`

这说明 Scheduler 分开统计：

```text
本地 prefix cache hit；
外部 connector prefix cache hit。
```

---

## 15. load_kv_async=True 时为什么本轮 num_new_tokens=0

当 connector 返回 `load_kv_async=True`：

```python
if load_kv_async:
    # KVTransfer: loading remote KV, do not allocate for new work.
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:781`

意思是：

```text
本轮只是为外部 KV load 分配本地 block，并发起或安排 load；
不做本地模型 forward。
```

为什么？

因为：

```text
请求后续 token 依赖这些外部 KV；
如果 KV 还没 load 完，不能越过它继续 forward；
所以本轮先进入 WAITING_FOR_REMOTE_KVS，等 Worker 报告 finished_recving 后再恢复调度。
```

此时 `allocate_slots()` 仍会被调用，因为 external KV 需要本地 block 槽位：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    ...
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    ...
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:873`

---

## 16. async load 时 reserved_blocks 的作用

如果是 async load：

```python
reserved_blocks = self._inflight_prefill_reserved_blocks()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:871`

源码注释：

```text
An async load holds its blocks for the whole transfer with no forward progress and isn't preemptible here.
Admit it only if it fits in (free - other in-flight reservations), to avoid deadlock and predictable preemptions.
```

意思是：

```text
异步 load 会长期占住 block，但本轮不产生 forward 进展；
如果它把剩余 block 全占了，其他 in-flight prefill 可能无法完成；
所以调度 async load 时要为其他正在进行的 prefill 预留 blocks。
```

这也是 `KVCacheManager.allocate_slots()` 中 `reserved_blocks` 参数的来源。

---

## 17. allocate_slots 后为什么必须 update_state_after_alloc

分配成功后：

```python
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request,
        self.kv_cache_manager.get_blocks(request_id),
        num_external_computed_tokens,
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:900`

抽象接口说明：

```python
def update_state_after_alloc(
    self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
):
    """
    Update KVConnector state after block allocation.
    ...
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:488`

它为什么必须在 `allocate_slots()` 后？

因为 `get_num_new_matched_tokens()` 只知道：

```text
外部 KV 命中了多少 token。
```

但不知道：

```text
这些外部 KV 应该 load 到本地哪些 block ids。
```

只有 `KVCacheManager.allocate_slots()` 成功后，才有完整本地 block 布局：

```text
local prefix blocks + external load blocks + new compute blocks + lookahead blocks
```

所以 connector 需要在这一步拿到：

```python
self.kv_cache_manager.get_blocks(request_id)
```

用来生成后续 Worker load / save metadata。

---

## 18. update_state_after_alloc 可能被调用两次

抽象接口注释提到：

```text
If get_num_new_matched_tokens previously returned True for a request,
this function may be called twice for that same request - first when blocks
are allocated for the connector tokens to be asynchronously loaded into,
and second when any additional blocks are allocated, after the load/transfer
is complete.
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:495`

对应场景：

```text
第一次：load_kv_async=True，本轮只为 external KV 分配 block，进入 WAITING_FOR_REMOTE_KVS。
第二次：KV load 完成后请求恢复调度，再为后续本地 forward token 分配 blocks。
```

因此 connector 侧必须能处理同一个 request 的多阶段状态。

---

## 19. waiting 请求调度成功后的状态分支

`update_state_after_alloc()` 后，Scheduler 从 waiting queue 取出请求：

```python
request = request_queue.pop_request()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:916`

### 19.1 load_kv_async=True

```python
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:917`

含义：

```text
请求暂时不进入 running；
它持有为外部 KV load 分配好的 blocks；
状态变为 WAITING_FOR_REMOTE_KVS；
放入 skipped_waiting，等待 Worker 侧 finished_recving。
```

### 19.2 load_kv_async=False

请求会正常进入 running，并进入 `scheduled_new_reqs` 或 `scheduled_resumed_reqs`。

含义：

```text
本轮可以继续走模型 forward；
如果有 external KV 命中，对应 load 信息会通过 kv_connector_metadata 交给 Worker。
```

---

## 20. build_connector_meta 何时调用

`SchedulerOutput` 先构造出来：

```python
scheduler_output = SchedulerOutput(...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1057`

然后：

```python
if self.connector is not None:
    meta = self._build_kv_connector_meta(self.connector, scheduler_output)
    scheduler_output.kv_connector_metadata = meta
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1080`

`_build_kv_connector_meta()` 很薄：

```python
def _build_kv_connector_meta(
    self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
) -> KVConnectorMetadata:
    return connector.build_connector_meta(scheduler_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1100`

源码注释说明这个函数有三个目的：

```text
1. Plan the KV cache store
2. Wrap up all the KV cache load / save ops into an opaque object
3. Clear the internal states of the connector
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1076`

因此：

```text
build_connector_meta() 是 Scheduler 侧 connector 把本轮内部状态“打包成给 Worker 的 metadata”的收口点。
```

---

## 21. build_connector_meta 的约束

抽象接口说明：

```python
def build_connector_meta(
    self, scheduler_output: SchedulerOutput
) -> KVConnectorMetadata:
    """
    Build the connector metadata for this step.

    This function should NOT modify fields in the scheduler_output.
    Also, calling this function will reset the state of the connector.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:509`

这有两个关键约束：

```text
1. 不应该修改 scheduler_output 的原有调度字段；
2. 调用后 connector 内部本轮临时状态会被清空或消费。
```

也就是说：

```text
connector 在前面 get_num_new_matched_tokens() / update_state_after_alloc() 积累状态；
build_connector_meta() 把这些状态打包出去，并准备进入下一轮。
```

---

## 22. MooncakeStoreScheduler 的具体例子

`MooncakeStoreScheduler` 是一个很贴近 KVPool / AscendStore 场景的 Scheduler 侧实现。

文件：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`

### 22.1 查询外部 KV 命中

```python
num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:85`

如果 full hit，会留最后一个 token 给本地重算：

```python
if num_external_hit_tokens == request.num_tokens:
    num_external_hit_tokens = max(
        0,
        (request.num_tokens - 1) // self._block_size * self._block_size,
    )
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:87`

然后算新增命中：

```python
if num_external_hit_tokens < num_computed_tokens:
    need_to_allocate = 0
else:
    need_to_allocate = num_external_hit_tokens - num_computed_tokens
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:95`

返回：

```python
return need_to_allocate, self.load_async
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:117`

这正好体现了接口语义：

```text
外部系统可能知道“总共命中到哪个 token”；
但 Scheduler 要的是“本地 prefix cache 之外还新增命中了多少 token”。
```

### 22.2 分配后记录 block ids

```python
if num_external_tokens > 0:
    local_block_ids = blocks.get_block_ids()

self._unfinished_requests[request.request_id] = (request, local_block_ids)
self._unfinished_request_ids.add(request.request_id)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:126`

如果 request 有 load spec，还会标记可以 load：

```python
self.load_specs[request.request_id].can_load = True
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:152`

这说明 `update_state_after_alloc()` 的核心价值是：

```text
拿到本地 block ids，把“外部 KV 命中”转成“外部 KV 应该 load 到这些本地 block”。
```

### 22.3 构造 metadata

```python
meta = MooncakeStoreConnectorMetadata(
    self._unfinished_request_ids,
    preempted_ids,
)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:175`

然后遍历 `scheduler_output.scheduled_new_reqs`，为每个新请求构造 `ReqMeta`：

```python
for request in scheduler_output.scheduled_new_reqs:
    load_spec = self.load_specs.pop(request.req_id, None)
    ...
    req_meta = ReqMeta.from_request_tracker(...)
    if req_meta is not None:
        meta.add_request(req_meta)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:181`

也就是说：

```text
SchedulerOutput 提供本轮调度了哪些请求、调度了多少 token、block ids 是什么；
connector 自己之前记录 external hit / load specs；
build_connector_meta() 把两边合并成 Worker 能执行的 metadata。
```

---

## 23. ExampleConnector 的简单例子

`ExampleConnector` 的 scheduler 侧实现也能说明接口语义。

### 23.1 get_num_new_matched_tokens

```python
if not self._found_match_for_request(request):
    return 0, False
...
num_tokens_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)
return num_tokens_to_check - num_computed_tokens, False
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:276`

这里同样做了：

```text
外部命中 token 数 - 本地已计算 token 数 = connector 返回的新增 external tokens。
```

### 23.2 update_state_after_alloc

```python
if num_external_tokens > 0:
    self._requests_need_load[request.request_id] = request
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:297`

### 23.3 build_connector_meta

```python
if new_req.req_id in self._requests_need_load:
    meta.add_request(
        token_ids=token_ids,
        block_ids=new_req.block_ids[0],
        block_size=self._block_size,
        is_store=False,
        mm_hashes=mm_hashes,
    )
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:318`

这里能看出 metadata 至少要告诉 Worker：

```text
token_ids；
block_ids；
block_size；
是 load 还是 store；
多模态 hash 等额外 key。
```

具体字段因 connector 实现不同而不同。

---

## 24. request_finished：请求结束时 connector 如何接入

Scheduler 在 `_free_request()` 中会调用：

```python
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2052`

`_connector_finished()` 中：

```python
if self.connector is None:
    return False, None

self.kv_cache_manager.remove_skipped_blocks(
    request_id=request.request_id,
    total_computed_tokens=request.num_computed_tokens,
)

block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2308`

然后根据 connector 是否支持 HMA：

```python
if not isinstance(self.connector, SupportsHMA):
    assert len(self.kv_cache_config.kv_cache_groups) == 1
    return self.connector.request_finished(request, block_ids[0])

return self.connector.request_finished_all_groups(request, block_ids)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2320`

含义是：

```text
请求结束后，Scheduler 把该请求最终持有的 block ids 交给 connector；
connector 决定是否需要把这些 KV 保存到外部 KV cache；
如果需要异步 save，可能要求延迟释放 blocks。
```

这个主题会在 `06_external_kv_save_flow.md` 中详细展开。

---

## 25. Worker 侧输出如何回到 Scheduler

Worker / ModelRunner 返回的 `ModelRunnerOutput` 中有：

```text
kv_connector_output
```

Scheduler 在 `update_from_output()` 阶段会处理 connector output。

相关方法：

```python
def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2417`

如果 Scheduler 侧 connector 存在，会先让 connector 更新内部状态：

```python
if self.connector is not None:
    self.connector.update_connector_output(kv_connector_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2428`

抽象接口：

```python
def update_connector_output(self, connector_output: KVConnectorOutput):
    """
    Update KVConnector state from worker-side connectors output.
    """
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:532`

这一步让 Scheduler 侧 connector 有机会消费 Worker 侧 connector 输出的 metadata / stats / job 状态。

---

## 26. finished_recving 如何让 WAITING_FOR_REMOTE_KVS 恢复

Worker 侧 connector output 中如果有：

```text
finished_recving
```

Scheduler 处理：

```python
for req_id in kv_connector_output.finished_recving or ():
    assert req_id in self.requests
    req = self.requests[req_id]
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        self.finished_recving_kv_req_ids.add(req_id)
    else:
        assert RequestStatus.is_finished(req.status)
        self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2432`

之后下一轮 schedule 遍历 skipped_waiting 时：

```python
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False
    self._update_waiting_for_remote_kv(request)
    if request.num_preemptions:
        request.status = RequestStatus.PREEMPTED
    else:
        request.status = RequestStatus.WAITING
    return True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2388`

这就是异步外部 KV load 的闭环：

```text
schedule() 发起 async load
  → request.status = WAITING_FOR_REMOTE_KVS
  → Worker 完成 load
  → kv_connector_output.finished_recving
  → Scheduler 记录 finished_recving_kv_req_ids
  → 下一轮 promote 回 WAITING / PREEMPTED
  → 继续调度后续 token
```

---

## 27. _update_waiting_for_remote_kv 做什么

当异步 KV load 完成后：

```python
def _update_waiting_for_remote_kv(self, request: Request) -> None:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2350`

正常成功分支：

```python
self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2373`

含义：

```text
1. 现在外部 KV 已经 load 到本地 blocks，可以把这些 blocks 作为已缓存 blocks；
2. 如果 full prompt 都命中，仍要回退最后一个 token，用于重新计算 logits 采样下一个 token。
```

如果 load 失败：

```python
if request.request_id in self.failed_recving_kv_req_ids:
    if request.num_computed_tokens:
        self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
    else:
        self.kv_cache_manager.free(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2360`

这部分在 `08_invalid_blocks_and_recompute.md` 里细讲。

---

## 28. finished_sending 如何影响 block 释放

如果 Worker 侧 connector output 中有：

```text
finished_sending
```

Scheduler 处理：

```python
for req_id in kv_connector_output.finished_sending or ():
    assert req_id in self.requests
    self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2441`

含义：

```text
请求结束时 connector 可能要求先不要释放 blocks，因为还要异步 save KV 到外部系统；
当 Worker 报告 finished_sending 后，Scheduler 才真正释放这些 blocks。
```

这和请求结束保存 KV 的链路对应，后续在 `06_external_kv_save_flow.md` 细讲。

---

## 29. invalid blocks 和 load failure policy

Worker 侧 connector 还可能报告：

```text
invalid_block_ids
```

这些会在 `update_from_output()` 中触发：

```text
_handle_invalid_blocks()
_update_requests_with_invalid_blocks()
```

并根据：

```python
self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:143`

决定：

```text
回退 num_computed_tokens 后重新计算；
或者直接失败请求。
```

这部分在 `08_invalid_blocks_and_recompute.md` 细讲。

---

## 30. Scheduler 侧 connector 的完整时序

把上面的关键点串起来：

```text
Scheduler.__init__()
  → KVConnectorFactory.create_connector(role=SCHEDULER)
  → 创建 KVCacheManager
  → connector.bind_gpu_block_pool(block_pool)

Scheduler.add_request()
  → connector.on_new_request(request)

Scheduler.schedule() waiting 阶段
  → KVCacheManager.get_computed_blocks(request)
  → connector.get_num_new_matched_tokens(request, local_hit_tokens)
  → num_computed_tokens = local_hit + external_hit
  → 如果 load_kv_async: num_new_tokens = 0
  → KVCacheManager.allocate_slots(... external_hit ...)
  → connector.update_state_after_alloc(request, blocks, external_hit)
  → 请求进入 running 或 WAITING_FOR_REMOTE_KVS

Scheduler.schedule() 构造输出
  → SchedulerOutput(...)
  → connector.build_connector_meta(scheduler_output)
  → scheduler_output.kv_connector_metadata = meta

Worker / ModelRunner
  → Worker 侧 connector 消费 metadata
  → 执行 KV load / save
  → 返回 kv_connector_output

Scheduler.update_from_output()
  → connector.update_connector_output(kv_connector_output)
  → finished_recving → 恢复 WAITING_FOR_REMOTE_KVS
  → finished_sending → 释放延迟释放的 blocks
  → invalid blocks → recompute 或 fail

Request finished
  → _connector_finished(request)
  → connector.request_finished(...) / request_finished_all_groups(...)
  → 可能触发外部 KV save
```

---

## 31. 一个完整例子：外部 KV 部分命中，同步路径

假设：

```text
prompt = 10000 tokens
本地 prefix cache hit = 3000 tokens
外部 KVPool hit beyond local = 5000 tokens
load_kv_async = False
```

调度过程：

```text
1. Scheduler 调 KVCacheManager.get_computed_blocks()
   → local_hit = 3000

2. Scheduler 调 connector.get_num_new_matched_tokens(request, 3000)
   → ext_tokens = 5000
   → load_kv_async = False

3. Scheduler 计算：
   num_computed_tokens = 3000 + 5000 = 8000
   num_new_tokens = 10000 - 8000 = 2000

4. Scheduler 调 KVCacheManager.allocate_slots()
   → 接入 local prefix blocks
   → 为 external 5000 tokens 分配本地 slots
   → 为 new 2000 tokens 分配本地 slots

5. Scheduler 调 connector.update_state_after_alloc()
   → connector 记录 external KV 要 load 到哪些 block ids

6. Scheduler 构造 SchedulerOutput
   → connector.build_connector_meta()
   → SchedulerOutput.kv_connector_metadata

7. Worker 消费 metadata
   → load external KV
   → forward 剩余 2000 tokens
```

关键点：

```text
本轮 forward token 数是 2000；
但本地 block 布局覆盖 10000 tokens 对应的 KV 状态。
```

---

## 32. 一个完整例子：外部 KV 异步 load

假设：

```text
prompt = 10000 tokens
本地 prefix cache hit = 0
外部 KVPool hit = 8000 tokens
load_kv_async = True
```

调度过程：

```text
1. get_num_new_matched_tokens()
   → ext_tokens = 8000
   → load_kv_async = True

2. Scheduler 设置：
   num_new_tokens = 0

3. KVCacheManager.allocate_slots()
   → 为 external 8000 tokens 分配本地 block slots
   → delay_cache_blocks=True

4. connector.update_state_after_alloc()
   → 记录这些 block ids 用于 async load

5. request.status = WAITING_FOR_REMOTE_KVS
   → 放入 skipped_waiting
   → 本轮不 forward

6. build_connector_meta()
   → Worker 侧开始 async load

7. Worker 后续返回 finished_recving

8. Scheduler 下一轮：
   _try_promote_blocked_waiting_request()
   → _update_waiting_for_remote_kv()
   → request.status = WAITING
   → 后续继续调度剩余 token
```

关键点：

```text
load_kv_async=True 时，本轮 token budget 不被本地 forward 消耗；
但 KV blocks 会被分配并占用，直到 load 完成或请求释放。
```

---

## 33. 和 KVCacheManager 的边界

`KVCacheManager` 负责：

```text
本地 prefix cache 查询；
本地 block 分配；
external tokens 对应的本地 slots 分配；
block ids / KVCacheBlocks 管理；
cache_blocks / free blocks。
```

Scheduler 侧 connector 负责：

```text
外部 KV 是否命中；
命中多少 token；
是否异步加载；
分配后把 request 和 block ids 记录为 load / save 计划；
build_connector_meta() 给 Worker。
```

边界一句话：

```text
KVCacheManager 管本地 blocks，KV Connector 管外部 KV 协议，Scheduler 把二者串起来。
```

---

## 34. 和 Worker / ModelRunner 的边界

Scheduler 侧 connector 不执行真正的 KV load / save。

它只输出：

```text
SchedulerOutput.kv_connector_metadata
```

Worker / ModelRunner 侧做：

```text
bind_connector_metadata(metadata)
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
build_connector_worker_meta()
get_finished()
get_block_ids_with_load_errors()
```

这些接口定义在 `KVConnectorBase_V1` 的 Worker-side methods 中。

边界一句话：

```text
Scheduler 侧 connector 负责计划，Worker 侧 connector 负责执行。
```

---

## 35. 容易疑惑的点

### 35.1 connector.get_num_new_matched_tokens 返回的是总命中数吗？

不是。

```text
它应该返回“本地已经 computed 的 token 之外，外部还能新增命中的 token 数”。
```

如果外部系统查到总命中到 8000，本地已经命中 3000，那么返回应该是：

```text
5000
```

### 35.2 ext_tokens is None 是失败吗？

不一定。

```text
None 表示 connector 暂时无法确定命中数，Scheduler 会把请求放入 skipped_waiting，后续再查。
```

真正的 load 失败通常通过 Worker 侧 invalid blocks / failed recving 回报。

### 35.3 load_kv_async=True 是否表示不分配 block？

不是。

```text
它表示本轮不做本地 forward；
但必须先分配本地 block slots，供外部 KV load 进来。
```

### 35.4 update_state_after_alloc 为什么不能在 allocate_slots 前调用？

因为 allocate 前还不知道本地 block ids。

```text
connector 需要 block ids 才能告诉 Worker：外部 KV 应该 load 到哪里。
```

### 35.5 build_connector_meta 会不会修改 SchedulerOutput？

按照接口约束，不应该修改已有字段。

```text
它只应该根据 scheduler_output 和 connector 内部状态构造 metadata，并清理 connector 本轮状态。
```

### 35.6 KV Connector 是否只用于 load？

不是。

```text
它也可以用于 save / offload / disaggregated prefill-decode / KVPool 存储 / invalid block 上报 / stats。
```

---

## 36. 从“回答问题”的角度总结

如果问：

```text
Scheduler 如何接入 KV Connector？
```

可以回答：

```text
Scheduler 在初始化时根据 kv_transfer_config 创建 role=SCHEDULER 的 KV Connector，并在 KVCacheManager 创建后把 block_pool 绑定给 connector。新请求进入 Scheduler 时，Scheduler 调用 connector.on_new_request()。waiting 请求调度时，Scheduler 先用 KVCacheManager 查询本地 prefix cache 命中，再调用 connector.get_num_new_matched_tokens() 查询外部 KV cache 在本地命中之外额外命中了多少 token。Scheduler 把本地命中和外部命中合并为 num_computed_tokens，之后调用 KVCacheManager.allocate_slots() 为本地 prefix、外部 KV 和本轮 forward token 分配 block。分配成功后，Scheduler 调用 connector.update_state_after_alloc()，让 connector 记录 external KV 应该 load 到哪些本地 block。最后 Scheduler 调用 connector.build_connector_meta()，把本轮 KV load / save 计划封装到 SchedulerOutput.kv_connector_metadata，交给 Worker 侧 connector 执行。
```

如果问：

```text
get_num_new_matched_tokens() 返回值怎么理解？
```

可以回答：

```text
它返回两个值：第一个是外部 KV cache 在本地已 computed token 之外额外可用的 token 数；如果是 None，表示 connector 暂时无法确定，Scheduler 本轮跳过该请求。第二个布尔值表示这些外部 KV 是否需要在 scheduler step 之间异步加载；如果为 True，Scheduler 本轮只分配本地 block 并把请求置为 WAITING_FOR_REMOTE_KVS，不做本地 forward。
```

如果问：

```text
update_state_after_alloc() 为什么重要？
```

可以回答：

```text
因为 connector 查询命中时只知道外部 KV 命中了多少 token，但不知道这些 KV 应该 load 到本地哪些 block。只有 KVCacheManager.allocate_slots() 成功后，本地 block ids 才确定。update_state_after_alloc() 正是把 request、block ids 和 external token 数交给 connector，让它能在 build_connector_meta() 中生成 Worker 可执行的 KV load / save metadata。
```

---

## 37. 最关键流程图

```text
Scheduler.__init__()
  ├─ if kv_transfer_config:
  │    └─ KVConnectorFactory.create_connector(role=SCHEDULER)
  ├─ KVCacheManager(...)
  └─ connector.bind_gpu_block_pool(kv_cache_manager.block_pool)

Scheduler.add_request(request)
  ├─ waiting.enqueue(request)
  ├─ requests[request_id] = request
  └─ connector.on_new_request(request)

Scheduler.schedule() waiting 阶段
  ├─ KVCacheManager.get_computed_blocks(request)
  │    └─ local_hit_tokens
  │
  ├─ connector.get_num_new_matched_tokens(request, local_hit_tokens)
  │    ├─ ext_tokens = None
  │    │    └─ skipped_waiting，下一轮再查
  │    ├─ ext_tokens = 0
  │    │    └─ 无外部命中
  │    └─ ext_tokens > 0
  │         └─ 有外部 KV 命中
  │
  ├─ num_computed_tokens = local_hit_tokens + ext_tokens
  │
  ├─ if load_kv_async:
  │    └─ num_new_tokens = 0
  │  else:
  │    └─ num_new_tokens = request.num_tokens - num_computed_tokens
  │
  ├─ KVCacheManager.allocate_slots(
  │      num_new_computed_tokens=local_hit_tokens,
  │      num_external_computed_tokens=ext_tokens,
  │      delay_cache_blocks=load_kv_async,
  │  )
  │
  ├─ connector.update_state_after_alloc(request, blocks, ext_tokens)
  │
  ├─ if load_kv_async:
  │    └─ request.status = WAITING_FOR_REMOTE_KVS
  │  else:
  │    └─ request.status = RUNNING
  │
  └─ SchedulerOutput(...)
       └─ connector.build_connector_meta()
            └─ scheduler_output.kv_connector_metadata

Worker / ModelRunner
  └─ consume kv_connector_metadata
       ├─ load external KV
       ├─ save local KV
       └─ return kv_connector_output

Scheduler.update_from_output()
  ├─ connector.update_connector_output(kv_connector_output)
  ├─ finished_recving → promote WAITING_FOR_REMOTE_KVS later
  ├─ finished_sending → free delayed blocks
  └─ invalid blocks → recompute / fail
```

---

## 38. 最关键对象关系

```text
KVTransferConfig
  控制是否启用 KV Connector、使用哪个 connector、失败策略、consumer / producer 角色等。

KVConnectorFactory
  根据 KVTransferConfig 创建具体 connector 实现。

KVConnectorRole.SCHEDULER
  Scheduler 进程内 connector，参与调度和 metadata 构造。

KVConnectorRole.WORKER
  Worker 进程内 connector，执行实际 KV load / save。

KVConnectorBase_V1
  Scheduler-side 和 Worker-side connector 的统一抽象接口。

KVConnectorMetadata
  Scheduler connector → Worker connector 的本轮 KV transfer 计划。

KVConnectorOutput
  Worker connector → Scheduler 的 KV transfer 完成 / 失败状态。

KVCacheManager
  本地 block 分配和 prefix cache 管理。

SchedulerOutput
  携带 num_scheduled_tokens、block ids、kv_connector_metadata 等本轮执行计划。
```

---

## 39. 和后续专题的关系

本篇回答的是 Scheduler 侧 KV Connector 如何接入。

后续专题继续拆：

```text
05_external_kv_load_flow.md
  详细解释 load_kv_async、WAITING_FOR_REMOTE_KVS、finished_recving、恢复调度。

06_external_kv_save_flow.md
  详细解释 request_finished、request_finished_all_groups、finished_sending、延迟释放 blocks。

07_worker_kv_connector_flow.md
  详细解释 Worker / ModelRunner 如何消费 kv_connector_metadata 并执行 load / save。

08_invalid_blocks_and_recompute.md
  详细解释 invalid_block_ids、load failure policy、回退重算。

10_kvpool_end_to_end.md
  从 KVPool / MooncakeStore 视角串完整端到端链路。
```

最终最小心智模型：

```text
Scheduler 侧 KV Connector = 外部 KV 系统的调度代理：查命中、记录 block 分配结果、构造 metadata、接收 Worker 完成状态；真正的本地 block 管理由 KVCacheManager 完成，真正的 KV 传输由 Worker 侧 connector 完成。
```
