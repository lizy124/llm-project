# vLLM V1 Scheduler 逻辑梳理

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

本文按“由浅入深”的方式梳理 `Scheduler` 的职责、核心状态、主调度循环、输出回收、KV Connector、Encoder Cache、Spec Decode 等逻辑。

---

## 1. Scheduler 是什么

`Scheduler` 是 vLLM V1 EngineCore 里的调度中心。它不直接执行模型 forward，而是决定每一步应该让哪些请求进入模型、每个请求本步计算多少 token、要分配哪些 KV Cache block、要携带哪些辅助元数据。

它主要回答这些问题：

1. 当前有哪些请求在等待、运行、阻塞？
2. 本轮最多能调度多少 token？
3. 哪些 running 请求继续 decode / prefill？
4. 哪些 waiting 请求可以进入运行态？
5. prefix cache / 外部 KV cache 命中了多少 token？
6. KV Cache block 是否够用，不够时是否需要抢占？
7. 多模态 encoder 输入、结构化输出、投机解码等附加能力如何同步调度？
8. Worker 执行完后，如何更新请求状态、释放 block、返回输出？

核心类入口：

```python
class Scheduler(SchedulerInterface):
```

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:69`

---

## 2. Scheduler 的核心队列和状态

Scheduler 内部最重要的是三类请求容器。

### 2.1 `self.requests`

```python
self.requests: dict[str, Request] = {}
```

保存所有还没有彻底释放的请求，key 是 `request_id`。

注意：请求即使已经从 `running` / `waiting` 队列移除，也可能因为 KV Connector 还有异步发送/接收没完成，暂时仍然留在 `self.requests` 中。

源码位置：`scheduler.py:172`

### 2.2 `self.waiting`

```python
self.waiting = create_request_queue(self.policy)
```

正常等待调度的请求队列。

新请求通过 `add_request()` 进入这里，除非它处于特殊阻塞状态。

源码位置：`scheduler.py:182`

### 2.3 `self.skipped_waiting`

```python
self.skipped_waiting = create_request_queue(self.policy)
```

被临时跳过的 waiting 请求队列。常见原因：

- 等结构化输出 grammar 初始化完成；
- 等远端 KV load 完成；
- 等 streaming input 的下一段输入；
- LoRA 数量限制；
- KV Connector 暂时无法确定命中数；
- EC / Encoder cache 暂时不可用。

源码位置：`scheduler.py:184`

### 2.4 `self.running`

```python
self.running: list[Request] = []
```

已经进入模型执行流、后续可继续调度的请求列表。

源码位置：`scheduler.py:185`

### 2.5 调度策略

Scheduler 支持不同 policy：

```python
self.policy = SchedulingPolicy(self.scheduler_config.policy)
```

常见为：

- `FCFS`：先到先服务；
- `PRIORITY`：按优先级排序。

源码位置：`scheduler.py:175`

---

## 3. 初始化逻辑

`__init__()` 做的事情很多，可以分成几组。

### 3.1 读取配置

主要配置包括：

```python
self.scheduler_config = vllm_config.scheduler_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.kv_cache_config = kv_cache_config
self.parallel_config = vllm_config.parallel_config
```

源码位置：`scheduler.py:81`

其中最核心的调度限制是：

```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
self.max_num_scheduled_tokens = ...
self.max_model_len = vllm_config.model_config.max_model_len
```

含义：

- `max_num_running_reqs`：最多同时 running 的请求数；
- `max_num_scheduled_tokens`：单步最多调度 token 数；
- `max_model_len`：模型最大上下文长度。

源码位置：`scheduler.py:108`

### 3.2 创建 KV Connector

如果配置了 `kv_transfer_config`，Scheduler 会创建一个 Scheduler 侧的 KV Connector：

```python
self.connector = KVConnectorFactory.create_connector(
    config=self.vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=self.kv_cache_config,
)
```

源码位置：`scheduler.py:137`

它负责和外部 KV Cache / P-D disaggregation / offloading 交互。

Scheduler 侧主要调用：

- `connector.get_num_new_matched_tokens()`：查询外部 KV 命中；
- `connector.update_state_after_alloc()`：分配 block 后更新 connector 状态；
- `connector.build_connector_meta()`：构造给 Worker 的 KV 传输元数据；
- `connector.request_finished()`：请求结束时决定是否保存/发送 KV。

### 3.3 创建 KVCacheManager

Scheduler 自己不直接管理 block 细节，而是交给 `KVCacheManager`：

```python
self.kv_cache_manager = KVCacheManager(...)
```

源码位置：`scheduler.py:264`

它负责：

- prefix cache 查询；
- block 分配；
- block 释放；
- block cache / evict；
- common prefix 计算；
- KV cache events。

### 3.4 Encoder Cache

多模态或 encoder-decoder 模型会使用 encoder cache：

```python
self.encoder_cache_manager = EncoderDecoderCacheManager(...) 或 EncoderCacheManager(...)
```

源码位置：`scheduler.py:226`

它用于缓存 image/audio 等 encoder 输出，避免重复编码。

### 3.5 Spec Decode 状态

投机解码相关字段：

```python
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.num_lookahead_tokens = 0
self.dynamic_sd_lookup = None
```

源码位置：`scheduler.py:232`

根据 speculative config，会设置 EAGLE / draft model / DFlash / DSpark 等模式；如果配置了 `num_speculative_tokens_per_batch_size`，还会通过 `dynamic_sd_lookup` 在每轮按 batch size 选择本轮要调度的 spec token 数。

---

## 4. 最核心方法：`schedule()` 总览

`schedule()` 是整个文件最关键的函数。

源码入口：`scheduler.py:433`

它每调用一次，就生成一个 `SchedulerOutput`，交给 ModelRunner 执行。

简化流程如下：

```text
schedule()
  │
  ├─ 初始化本轮预算 token_budget
  ├─ 调度 running 请求
  │    ├─ 计算本轮要处理多少 token
  │    ├─ 处理 encoder input
  │    ├─ 分配 KV block
  │    ├─ 不够 block 时抢占其他请求
  │    └─ 记录本轮调度信息
  │
  ├─ 调度 waiting 请求
  │    ├─ 查询本地 prefix cache
  │    ├─ 查询外部 KV cache
  │    ├─ 计算还需要 prefill 的 token
  │    ├─ 分配 KV block
  │    ├─ 可能进入 WAITING_FOR_REMOTE_KVS
  │    └─ 加入 running
  │
  ├─ 组装 SchedulerOutput
  ├─ 构建 KVConnector / ECConnector metadata
  ├─ 更新调度后状态
  └─ 返回 SchedulerOutput
```

---

## 5. `schedule()` 的本轮预算

每轮调度开始时：

```python
token_budget = self.max_num_scheduled_tokens
```

源码位置：`scheduler.py:453`

如果 Scheduler 被整体 pause：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
```

源码位置：`scheduler.py:454`

所以本轮能调度多少 token，首先受 `max_num_scheduled_tokens` 限制。

另外还有 encoder budget：

```python
encoder_compute_budget = self.max_num_encoder_input_tokens
```

源码位置：`scheduler.py:460`

这表示本轮最多能处理多少 encoder input token / embed。

---

## 6. 第一阶段：调度 RUNNING 请求

Scheduler 会优先处理已经 running 的请求。

入口：

```python
while req_index < len(self.running) and token_budget > 0:
```

源码位置：`scheduler.py:479`

### 6.1 为什么先处理 running 请求

running 请求通常已经占用了 KV block。如果不继续推进它们，可能导致：

- decode 延迟变大；
- 已占用 block 长时间不释放；
- batch 中已有请求无法继续生成。

所以 Scheduler 先给 running 请求分配本轮 token 预算。

### 6.2 计算 running 请求本轮要处理的 token 数

核心公式：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

源码位置：`scheduler.py:510`

含义：

- `num_tokens_with_spec`：prompt + output + speculative tokens；
- `num_output_placeholders`：异步调度 / PP 场景中预留但还没返回的输出 token；
- `num_computed_tokens`：Scheduler 认为该请求已经安排 / 计算到的 token 位置；async / PP 下可能先于 Worker 真实返回。

差值就是本轮还需要补算的 token。当前源码还用 `num_in_flight_tokens` 记录这条乐观进度中尚未由 `update_from_output()` 回收的 token 数，但它不直接参与这个公式。

然后会受三个限制：

1. `long_prefill_token_threshold` 限制长 prefill chunk；
2. `token_budget` 限制本轮总 token；
3. `max_model_len` 限制上下文长度。

源码位置：`scheduler.py:515`、`scheduler.py:517`、`scheduler.py:521`

### 6.3 Encoder input 调度

如果请求有 encoder input：

```python
if request.has_encoder_inputs:
    ... = self._try_schedule_encoder_inputs(...)
```

源码位置：`scheduler.py:532`

这会决定本轮需要处理哪些多模态输入，以及是否需要缩短 `num_new_tokens`。

### 6.4 Mamba block 对齐

如果模型含 Mamba 层，并且 `mamba_cache_mode == "align"`，会执行：

```python
num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
```

源码位置：`scheduler.py:546`

目的是让 Mamba state 的 cache chunk 尽量按 block 边界对齐。

### 6.5 分配 KV block

running 请求本轮调度前，需要为新增 token 分配 KV block：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

源码位置：`scheduler.py:570`

如果分配成功，该请求可以被调度。

如果失败，就进入抢占逻辑。

### 6.6 KV block 不够时抢占

当 `allocate_slots()` 返回 `None`，说明 KV Cache 空间不足。

如果是 PRIORITY 策略：

```python
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

源码位置：`scheduler.py:584`

会选择优先级最低的 running 请求抢占。

如果不是 PRIORITY：

```python
preempted_req = self.running.pop()
```

源码位置：`scheduler.py:608`

通常就是 running 队尾请求。

抢占调用：

```python
self._preempt_request(preempted_req, scheduled_timestamp)
```

源码位置：`scheduler.py:611`

被抢占的请求会：

- 释放 KV blocks；
- 释放 encoder cache；
- 状态改成 `PREEMPTED`；
- `num_computed_tokens` 重置为 0；
- 放回 waiting 队列头部。

源码位置：`scheduler.py:1191`

### 6.7 running 请求调度成功后的记录

成功后会记录：

```python
scheduled_running_reqs.append(request)
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

源码位置：`scheduler.py:621`

这些信息之后会被打包到 `SchedulerOutput`。

---

## 7. 第二阶段：调度 WAITING 请求

只有在本轮没有发生抢占，并且 Scheduler 没有 pause new request 时，才会调度 waiting 请求：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

源码位置：`scheduler.py:673`

### 7.1 从哪个 waiting 队列取请求

Scheduler 有两个 waiting 队列：

- `waiting`：正常等待；
- `skipped_waiting`：之前被跳过或阻塞。

选择逻辑在：

```python
request_queue = self._select_waiting_queue_for_scheduling()
```

源码位置：`scheduler.py:684`

FCFS 下优先处理 `skipped_waiting`，因为它们之前已经等过一轮：

```python
if self.policy == SchedulingPolicy.FCFS:
    return self.skipped_waiting or self.waiting or None
```

源码位置：`scheduler.py:1924`

PRIORITY 下会比较两个队列头部请求的优先级。

### 7.2 阻塞状态提升

waiting 请求可能处于阻塞状态：

```python
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
WAITING_FOR_REMOTE_KVS
WAITING_FOR_STREAMING_REQ
```

判断逻辑：`scheduler.py:1910`

如果是阻塞状态，会尝试提升：

```python
self._try_promote_blocked_waiting_request(request)
```

源码位置：`scheduler.py:690`

比如远端 KV load 完成后，`WAITING_FOR_REMOTE_KVS` 可以恢复成 `WAITING` 或 `PREEMPTED`。这也覆盖被抢占请求的恢复：PREEMPTED 请求重新调度时会重新查本地 prefix cache 和外部 KV Connector；如果已计算 KV 已在远端 KV 池中，Scheduler 可以先触发 async load，等 Worker 加载完成后再继续进入 running。

### 7.3 LoRA 限制

如果启用了 LoRA，一轮调度里不能超过 `max_loras`：

```python
if len(scheduled_loras) == self.lora_config.max_loras
```

源码位置：`scheduler.py:705`

超过则暂时跳过该请求。

---

## 8. WAITING 请求的 prefix cache / KV cache 命中逻辑

waiting 请求第一次被调度时，`request.num_computed_tokens == 0`。

这时 Scheduler 会先查本地 prefix cache。

### 8.1 本地 prefix cache

普通路径：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

源码位置：`scheduler.py:761`

含义：

- `new_computed_blocks`：本地已经缓存命中的 block；
- `num_new_local_computed_tokens`：本地命中的 token 数。

如果是 Hybrid + Mamba，会走特殊逻辑：

```python
find_longest_cache_hit_per_group(...)
```

源码位置：`scheduler.py:734`

### 8.2 外部 KV Connector 命中

如果配置了 KV Connector，还会查外部 KV cache：

```python
ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
    request, num_new_local_computed_tokens
)
```

源码位置：`scheduler.py:775`

这里把本地已经命中的 token 数传给 connector，connector 只需要返回“外部额外可用”的 token 数。

例如：

```text
prompt 总长度 = 10000
本地 prefix cache 命中 = 3000
外部 KVPool 命中 = 8000
那么外部新增命中 = 8000 - 3000 = 5000
```

最终：

```python
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

源码位置：`scheduler.py:796`

### 8.3 外部 KV 查询失败时跳过

如果 connector 返回 `ext_tokens is None`，说明暂时无法确定命中数：

```python
if ext_tokens is None:
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

源码位置：`scheduler.py:781`

请求会被放入 skipped queue，等待后续再试。

---

## 9. WAITING 请求的 token 计算逻辑

如果不是异步 KV load，waiting 请求需要计算的 token 数是：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

源码位置：`scheduler.py:847`

这里的 `request.num_tokens` 不只是 prompt token，也可能包含 resumed request 的 output token。

然后受以下限制：

1. `long_prefill_token_threshold`；
2. `enable_chunked_prefill`；
3. `token_budget`；
4. encoder input budget；
5. Mamba block 对齐。

### 9.1 chunked prefill 开关

如果没有启用 chunked prefill，并且当前 prompt 剩余 token 超过本轮预算：

```python
if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
    break
```

源码位置：`scheduler.py:873`

这意味着：如果一个请求无法完整 prefill，就先不调度后续 waiting 请求。

如果启用了 chunked prefill，则可以只调度一部分：

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

源码位置：`scheduler.py:881`

### 9.2 异步 KV load

如果 connector 告诉 Scheduler 可以异步加载远端 KV：

```python
if load_kv_async:
    num_new_tokens = 0
```

源码位置：`scheduler.py:834`

这表示本轮不做模型计算，只分配 block 并发起远端 KV load。

之后请求状态变为：

```python
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
```

源码位置：`scheduler.py:989`

并放入 `skipped_waiting`：

```python
step_skipped_waiting.prepend_request(request)
```

源码位置：`scheduler.py:990`

---

## 10. WAITING 请求的 block 分配

waiting 请求分配 block 的调用更复杂：

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

源码位置：`scheduler.py:942`

这些参数的含义：

- `num_new_tokens`：本轮要实际计算的 token 数；
- `num_new_computed_tokens`：本地 prefix cache 命中的 token 数；
- `new_computed_blocks`：本地命中的 block；
- `num_external_computed_tokens`：外部 KV 命中的 token 数；
- `delay_cache_blocks`：异步 KV load 时，先分配但暂不正式 cache；
- `num_encoder_tokens`：encoder-decoder cross-attention 需要的 block；
- `reserved_blocks`：异步 load 场景下，为其它 in-flight prefill 保留的 block。

如果分配失败，则当前 waiting 调度停止。

源码位置：`scheduler.py:956`

---

## 11. Connector 在分配后的更新

分配 block 成功后，如果有 KV Connector：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

源码位置：`scheduler.py:969`

这一步非常关键。

对于外部 KVPool 场景，它告诉 connector：

1. 这个请求本地分配到了哪些 block；
2. 有多少 token 需要从外部 KV cache load；
3. connector 后续可以在 `build_connector_meta()` 里生成 load 元数据。

---

## 12. waiting 请求进入 running

如果不是异步 KV load，分配成功后：

```python
request = request_queue.pop_request()
self.running.append(request)
```

源码位置：`scheduler.py:985`、`scheduler.py:1008`

然后根据原状态记录到不同列表：

```python
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
```

源码位置：`scheduler.py:1013`

最后更新：

```python
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

源码位置：`scheduler.py:1027`

注意这里先把 prefix cache / external cache 命中的 token 记到 `num_computed_tokens`，后续 `_update_after_schedule()` 会再加上本轮实际调度的 token 数。

---

## 13. SchedulerOutput 是什么

`schedule()` 最终会构建 `SchedulerOutput`：

```python
scheduler_output = SchedulerOutput(...)
```

源码位置：`scheduler.py:1142`

里面包含：

- `scheduled_new_reqs`：本轮新进入模型的请求；
- `scheduled_cached_reqs`：已经在 running batch 中、继续推进的请求；
- `num_scheduled_tokens`：每个请求本轮调度 token 数；
- `total_num_scheduled_tokens`：本轮总 token 数；
- `scheduled_spec_decode_tokens`：本轮投机解码 token；
- `scheduled_encoder_inputs`：本轮 encoder 输入；
- `num_common_prefix_blocks`：running 请求共同 prefix block 数；
- `preempted_req_ids`：本轮被抢占的请求；
- `finished_req_ids`：上轮到本轮之间完成的请求；
- `kv_connector_metadata`：KV Connector 元数据；
- `ec_connector_metadata`：Encoder Cache Connector 元数据；
- `new_block_ids_to_zero`：需要清零的新 KV block；
- `kv_cache_block_copies`：本轮需要执行的 KV cache block copy；
- `num_spec_tokens_to_schedule`：动态 spec decode 下本轮建议的 spec token 数。

---

## 14. cached request 数据构造

对于已经在 running batch 里的请求，Scheduler 构造 `CachedRequestData`：

```python
cached_reqs_data = self._make_cached_request_data(...)
```

源码位置：`scheduler.py:1104`

入口：`scheduler.py:1308`

它会输出：

- `req_ids`：本轮继续执行的请求 ID；
- `new_token_ids`：PP 场景下可能需要传回的 token；
- `new_block_ids`：本轮新增 block；
- `all_token_ids`：必要时传完整 token；
- `num_computed_tokens`：调度前已计算 token 数；
- `num_output_tokens`：已有输出 token 数；
- `resumed_req_ids`：本轮恢复的请求。

`resumed_req_ids` 的作用是区分“已有 request state，但旧 KV block table 已失效”的恢复请求。旧 model runner 看到这些请求时会用新的 `block_ids` 替换旧 block table，而不是像普通 running 请求一样追加；这样可以复用 prompt、output tokens、sampling/spec/grammar 等仍有效的 request state，只重建 KV cache 相关状态。V2 model runner 路径下，resumed 请求会合并进 `scheduled_new_reqs`，按重新下发完整 request 数据处理。

这里有一个细节：

```python
if req_id not in self.prev_step_scheduled_req_ids:
    all_token_ids[req_id] = req.all_token_ids.copy()
```

源码位置：`scheduler.py:1347`

如果请求上一轮已经在 ModelRunner 的 persistent batch 中，就不需要重复传完整 token ids。

---

## 15. 调度后状态更新 `_update_after_schedule()`

`schedule()` 构造完输出后，会调用：

```python
self._update_after_schedule(scheduler_output)
```

源码位置：`scheduler.py:1182`

入口：`scheduler.py:1215`

核心逻辑：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

源码位置：`scheduler.py:1228`

也就是说：

1. `SchedulerOutput` 里保留的是调度前状态和本轮调度量；
2. Scheduler 内部状态会立刻向前推进；
3. 如果后续 Worker 输出发现投机 token 被拒绝，再在 `update_from_output()` 里回退。

同时更新：

```python
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```

源码位置：`scheduler.py:1233`

如果请求已经不再是 prefill chunk，就从 `_inflight_prefills` 移除。

最后清空 `finished_req_ids`：

```python
self.finished_req_ids = set()
```

源码位置：`scheduler.py:1262`

注意：不能直接 clear 老 set，因为 `SchedulerOutput` 还引用着旧的 finished set。

---

## 16. Worker 输出回收：`update_from_output()`

`schedule()` 只负责“发任务”，`update_from_output()` 负责“收结果”。

入口：`scheduler.py:1551`

输入：

```python
scheduler_output: SchedulerOutput
model_runner_output: ModelRunnerOutput
```

返回：

```python
dict[int, EngineCoreOutputs]
```

key 是 client index，value 是该 client 本轮要收到的输出。

整体流程：

```text
update_from_output()
  │
  ├─ 处理 deferred free
  ├─ 处理 KV load 失败 block
  ├─ 遍历本轮调度的 req_id
  │    ├─ 回收 num_in_flight_tokens
  │    ├─ 取 sampled token
  │    ├─ 处理 spec decode 接受/拒绝
  │    ├─ 释放已用 encoder input
  │    ├─ append 输出 token
  │    ├─ 检查 stop 条件
  │    ├─ 处理 structured output grammar（必要时先 trim reasoning token）
  │    ├─ 请求结束则 free request，并返回 KV / EC transfer params
  │    └─ 构造 EngineCoreOutput
  │
  ├─ 从 running / waiting 移除已停止请求
  ├─ 更新 KV Connector 完成状态
  ├─ 发布 KV cache events
  ├─ 附加 stats
  └─ 返回 EngineCoreOutputs
```

---

## 17. sampled token 处理

每个本轮调度的请求都会从 `model_runner_output` 里取生成 token：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
```

源码位置：`scheduler.py:1632`

对于 prefill chunk，通常没有 sampled token；对于 decode，则会返回新生成 token。

### 17.1 append 输出并检查 stop

```python
new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
```

源码位置：`scheduler.py:1684`

内部逻辑：

```python
request.append_output_token_ids(output_token_id)
stopped = check_stop(request, self.max_model_len)
```

源码位置：`scheduler.py:1961`

如果遇到 stop，会裁剪多余 token：

```python
del new_token_ids[num_new:]
```

源码位置：`scheduler.py:1968`

---

## 18. 投机解码 Spec Decode 逻辑

### 18.1 调度阶段

running 请求如果带有 `request.spec_token_ids`，会先计算本轮可调度的 spec token 数，必要时裁剪后再记录：

```python
num_scheduled_spec_tokens = (
    num_new_tokens
    + request.num_computed_tokens
    - request.num_tokens
    - request.num_output_placeholders
)
if num_scheduled_spec_tokens > 0:
    spec_token_ids = request.spec_token_ids
    if len(spec_token_ids) > num_scheduled_spec_tokens:
        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
```

源码位置：`scheduler.py:631`

waiting 请求刚进入 decode、且需要保持固定 spec decode 图形时，也可能直接把 `[-1] * self.num_spec_tokens` 放入 `scheduled_spec_decode_tokens`，作为本轮 spec padding / placeholder。

源码位置：`scheduler.py:1029`

### 18.2 输出阶段

Worker 返回后，Scheduler 会在非 stale async frame 的情况下计算接受了多少 draft token：

```python
if (
    scheduled_spec_token_ids
    and (generated_token_ids or self.num_sampled_tokens_per_step == 0)
    and request.async_tokens_to_discard == 0
):
    num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    num_rejected = num_draft_tokens - num_accepted
```

源码位置：`scheduler.py:1642`

被拒绝的 token 要从 `num_computed_tokens` 回退：

```python
request.num_computed_tokens -= num_rejected
```

源码位置：`scheduler.py:1656`

如果 async scheduling 中有 output placeholders，也要同步扣掉：

```python
request.num_output_placeholders -= num_rejected
```

源码位置：`scheduler.py:1660`

### 18.3 draft token 更新

外部 draft model 生成的 token 通过：

```python
update_draft_token_ids()
```

写回 request。

源码位置：`scheduler.py:2005`

如果结构化输出开启，会先用 grammar 过滤 draft token；async output 中的 spec placeholder 也会通过 `update_draft_token_ids_in_output()` 裁剪、validate，并用 `-1` pad 回原长度，记录 `num_invalid_spec_tokens`。

---

## 19. 结构化输出 Structured Output

Scheduler 会为结构化输出请求生成 grammar bitmask：

```python
get_grammar_bitmask()
```

源码位置：`scheduler.py:1527`

只对非 prefill chunk 的结构化输出请求生效：

```python
req.use_structured_output and not req.is_prefill_chunk
```

源码位置：`scheduler.py:1539`

生成 token 后，还会推进 grammar 状态；如果输出块里包含 reasoning content，会先裁掉 reasoning 部分，只把 grammar token 交给 grammar：

```python
advance_token_ids = self.structured_output_manager.trim_reasoning_for_advance(
    request, new_token_ids
)
if advance_token_ids and not grammar.accept_tokens(req_id, advance_token_ids):
    ...
```

源码位置：`scheduler.py:1701`

如果 grammar 拒绝已经生成的 token，会把请求标记为 error：

```python
request.status = RequestStatus.FINISHED_ERROR
request.resumable = False
```

源码位置：`scheduler.py:1715`

---

## 20. Encoder / 多模态输入调度

入口：

```python
_try_schedule_encoder_inputs(...)
```

源码位置：`scheduler.py:1367`

它的目标是判断本轮 decoder token 范围内，哪些 encoder input 必须先计算。

### 20.1 查找本轮覆盖的 mm feature

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
)
```

源码位置：`scheduler.py:1409`

即：本轮 decoder 要处理的 token 范围，如果覆盖了 image/audio placeholder，就必须安排 encoder。

### 20.2 已缓存则跳过

```python
if self.encoder_cache_manager.check_and_update_cache(request, i):
    continue
```

源码位置：`scheduler.py:1449`

### 20.3 不允许切分多模态输入时回退

如果 `disable_chunked_mm_input` 开启，并且本轮只覆盖了 mm input 的一部分，会把 `num_new_tokens` 回退到 mm input 之前：

```python
num_new_tokens = max(0, start_pos - (...))
```

源码位置：`scheduler.py:1457`

### 20.4 encoder cache / budget 不够时回退

如果 encoder cache 或 encoder compute budget 不够：

```python
if not self.encoder_cache_manager.can_allocate(...):
```

源码位置：`scheduler.py:1470`

则只调度到该 encoder input 之前，或者本轮不调度该请求。

### 20.5 ECConnector 外部 encoder cache

如果配置了 ECConnector，并且远端有该 encoder item：

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
    external_load_encoder_input.append(i)
```

源码位置：`scheduler.py:1507`

表示本轮不是本地计算 encoder，而是从远端加载 encoder cache。

---

## 21. 请求结束与释放

### 21.1 外部主动结束请求

API server 或客户端断开时，会调用：

```python
finish_requests(request_ids, finished_status)
```

源码位置：`scheduler.py:2093`

它会：

1. 从 running / waiting / skipped_waiting 中移除请求；
2. 设置 finished status；
3. 调用 `_free_request()` 释放资源。

### 21.2 正常生成停止

在 `update_from_output()` 中，如果 `check_stop()` 判断请求结束：

```python
finished = self._handle_stopped_request(request)
if finished:
    kv_transfer_params, ec_transfer_params = self._free_request(request)
```

源码位置：`scheduler.py:1764`

### 21.3 `_free_request()`

```python
self._inflight_prefills.discard(request)
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
if self.ec_connector is not None:
    ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
    connector_delay_free_blocks |= ec_delay_free
self.encoder_cache_manager.free(request)
self.finished_req_ids.add(request_id)
```

源码位置：`scheduler.py:2156`

如果 connector 要异步发送 KV，可能不会立即释放 block：

```python
delay_free_blocks |= connector_delay_free_blocks
if not delay_free_blocks:
    self._free_blocks(request)
```

源码位置：`scheduler.py:2179`

---

## 22. KV Connector 相关逻辑

Scheduler 文件底部专门有一组 KV Connector 方法。

入口位置：`scheduler.py:2424`

### 22.1 请求结束时通知 connector

```python
_connector_finished(request)
```

源码位置：`scheduler.py:2434`

它会先按 processed-token basis 移除 sliding window 等场景中已经不需要的 skipped blocks：

```python
self.kv_cache_manager.remove_skipped_blocks(
    request_id=request.request_id,
    processed_computed_tokens=max(
        0, request.num_computed_tokens - request.num_in_flight_tokens
    ),
    num_prompt_tokens=request.num_prompt_tokens,
)
```

源码位置：`scheduler.py:2448`

然后取已计算 token 范围对应的 block ids，调用 connector：

```python
block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
    request_id=request.request_id,
    num_computed_tokens=request.num_computed_tokens,
)
return self.connector.request_finished_all_groups(request, block_ids)
```

源码位置：`scheduler.py:2469`

对于 AscendStore / KVPool 类 connector，这里通常会决定是否把请求的 KV 保存到外部池。

### 22.2 异步远端 KV load 完成后恢复请求

当 Worker 报告远端 KV 接收完成后：

```python
self.finished_recving_kv_req_ids.add(req_id)
```

源码位置：`scheduler.py:2574`

下次调度 waiting 队列时，会在：

```python
_try_promote_blocked_waiting_request()
```

里恢复。

源码位置：`scheduler.py:2526`

真正更新状态的是：

```python
_update_waiting_for_remote_kv(request)
```

源码位置：`scheduler.py:2492`

成功时：

```python
self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
```

源码位置：`scheduler.py:2517`

如果 full prompt 都从远端命中，需要回退一个 token：

```python
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

源码位置：`scheduler.py:2521`

原因：decode 阶段至少需要重新计算最后一个 token，才能采样下一个 token。

### 22.3 KV load 失败处理

如果 Worker 返回 invalid block ids：

```python
failed_kv_load_req_ids = self._handle_invalid_blocks(...)
```

源码位置：`scheduler.py:1579`

核心逻辑在：

```python
_update_requests_with_invalid_blocks()
```

源码位置：`scheduler.py:2588`

它会扫描请求使用的 block，如果某个 externally loaded block 失败：

```python
request.num_computed_tokens = idx * self.block_size
```

源码位置：`scheduler.py:2665`

也就是把请求回退到第一个失败 block 之前，后面 token 重新计算。

当前实现会分别扫描 `skipped_waiting` 中的 async KV load 请求和 `running` 中的 sync load 请求；recompute 策略下 async 失败请求会进入 `failed_recving_kv_req_ids`，等 remote load 完成后由 `_update_waiting_for_remote_kv()` 缓存有效前缀或释放已分配 blocks。如果配置为 failure policy = fail，则直接失败请求。

---

## 23. pause / reset 逻辑

### 23.1 pause state

Scheduler 有三种 pause 状态，定义来自 `PauseState`：

- `UNPAUSED`：正常；
- `PAUSED_NEW`：不接新请求，只跑已有 running；
- `PAUSED_ALL`：全部暂停。

`schedule()` 开头如果是 `PAUSED_ALL`，本轮 token budget 直接为 0。

源码位置：`scheduler.py:454`

`get_num_unfinished_requests()` 也会根据 pause 状态返回不同数量：

源码位置：`scheduler.py:2238`

### 23.2 reset prefix cache

```python
reset_prefix_cache(reset_running_requests=False, reset_connector=False)
```

源码位置：`scheduler.py:2275`

如果 `reset_running_requests=True`，会先抢占所有 running 请求：

```python
while self.running:
    request = self.running.pop()
    self._preempt_request(request, timestamp)
```

源码位置：`scheduler.py:2293`

同时，对于 async scheduling 中还没回来的 output frame，会把当时的 `num_output_placeholders` 转成 `async_tokens_to_discard`，并清零 placeholder，避免 stale frame 回来后继续 append token。

然后重置 KV prefix cache。

如果 `reset_connector=True`，还会重置外部 connector cache。

---

## 24. Mamba block 对齐逻辑

入口：

```python
_mamba_block_aligned_split(...)
```

源码位置：`scheduler.py:347`

这个函数只在：

```python
self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
```

时使用。

源码位置：`scheduler.py:297`

它会尽量让 prefill chunk 按 block size 对齐：

```python
num_new_tokens = num_new_tokens // block_size * block_size
```

源码位置：`scheduler.py:374`

原因是 Mamba state cache 对 chunk 边界更敏感，如果 chunk 不对齐，可能导致 Mamba cache miss 或无法缓存。

同时也有 Marconi cache admission 优化：如果发现未缓存 common prefix 足够长，可以只调度 common prefix 长度，先把公共前缀缓存起来。

源码位置：`scheduler.py:422`

---

## 25. deferred free：异步场景下延迟释放 block

如果使用 KV consumer 且存在多个 in-flight batches：

```python
self.defer_block_free = True
```

源码位置：`scheduler.py:151`

原因：异步调度 / pipeline parallel 下，一个 batch 的 GPU 写操作可能还没完成，如果此时把 block 释放并复用，另一个 KV load 可能写入同一 block，产生竞态。

释放逻辑：

```python
_free_request_blocks()
```

源码位置：`scheduler.py:2197`

如果 block 对应的 scheduled step 还没被 `update_from_output()` 确认完成，就放进：

```python
self.deferred_frees
```

源码位置：`scheduler.py:2210`

在后续 `update_from_output()` 开头：

```python
self._drain_deferred_frees()
```

源码位置：`scheduler.py:1567`

确认安全后再真正归还 block pool。

---

## 26. add_request：新请求入口

新请求进入 Scheduler 的入口：

```python
add_request(request)
```

源码位置：`scheduler.py:2069`

如果 request id 已存在，说明这是 streaming input 的后续 chunk：

```python
existing = self.requests.get(request.request_id)
```

源码位置：`scheduler.py:2070`

它会更新已有 session，而不是创建新请求。

如果是全新请求：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
```

源码位置：`scheduler.py:2086`

如果配置了 KV Connector，还会通知 connector：

```python
self.connector.on_new_request(request)
```

源码位置：`scheduler.py:2088`

---

## 27. 完整生命周期

一个普通文本请求的生命周期如下：

```text
add_request()
  │
  ▼
进入 waiting 队列
  │
  ▼
schedule() 调度 waiting
  │
  ├─ 查本地 prefix cache
  ├─ 查外部 KV cache（如果有 connector）
  ├─ 分配 KV blocks
  ├─ 加入 running
  └─ 生成 SchedulerOutput
  │
  ▼
ModelRunner 执行 forward
  │
  ▼
update_from_output()
  │
  ├─ append sampled token
  ├─ 检查 stop
  ├─ 未结束：继续留在 running
  └─ 结束：_free_request()
          ├─ 通知 connector
          ├─ 释放 encoder cache
          ├─ 释放或延迟释放 KV blocks
          └─ 从 self.requests 删除，或等待 connector async send 完成后删除
```

如果是异步外部 KV load 请求，中间会多一段：

```text
waiting 请求
  │
  ├─ connector 命中外部 KV
  ├─ allocate_slots(delay_cache_blocks=True)
  ├─ 状态变为 WAITING_FOR_REMOTE_KVS
  └─ 放入 skipped_waiting
       │
       ▼
Worker 完成 KV load
       │
       ▼
update_from_output() 记录 finished_recving
       │
       ▼
下一轮 schedule()
       │
       ├─ _try_promote_blocked_waiting_request()
       ├─ _update_waiting_for_remote_kv()
       └─ 重新进入 WAITING / PREEMPTED 后继续调度
```

---

## 28. 从 KVPool 视角看 Scheduler 的关键插入点

如果关注你前面看的 KVPool / AscendStore，这个 Scheduler 文件中最关键的是以下几处。

### 28.1 查询外部 KV 命中

```python
ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
    request, num_new_local_computed_tokens
)
```

源码位置：`scheduler.py:775`

这里会进入 KVPoolScheduler 的 `get_num_new_matched_tokens()`。

### 28.2 分配 block 后通知 connector

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

源码位置：`scheduler.py:969`

这里会进入 KVPoolScheduler 的 `update_state_after_alloc()`。

### 28.3 构造 connector metadata

```python
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

源码位置：`scheduler.py:1166`

这里会进入 KVPoolScheduler 的 `build_connector_meta()`。

### 28.4 请求结束时保存 KV

```python
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
```

源码位置：`scheduler.py:2162`

这里最终会进入 connector 的 `request_finished()` / `request_finished_all_groups()`，用于决定 KV 是否保存到外部池。

---

## 29. 关键数据结构关系

### 29.1 Request

Scheduler 操作的核心对象。重要字段包括：

- `request_id`：请求 ID；
- `status`：请求状态；
- `prompt_token_ids`：prompt tokens；
- `_all_token_ids` / `all_token_ids`：prompt + output；
- `num_tokens`：当前总 token 数；
- `num_prompt_tokens`：prompt 长度；
- `num_computed_tokens`：Scheduler 认为已推进的计算进度；
- `num_in_flight_tokens`：乐观推进中尚未由 `update_from_output()` 回收的 token 数；
- `spec_token_ids`：投机 token；
- `num_output_placeholders`：异步/PP 占位输出；
- `async_tokens_to_discard`：reset / force-preempt 后待丢弃的 stale async output frame 数；
- `kv_transfer_params` / `ec_transfer_params`：KV / Encoder Cache connector 返回上层的 transfer 参数；
- `block_hashes`：prefix cache / KV connector 查找用 hash；
- `mm_features`：多模态输入；
- `streaming_queue`：streaming input 后续 chunk。

### 29.2 SchedulerOutput

Scheduler 发给 Worker / ModelRunner 的一轮执行计划，同时也是 `Scheduler.update_from_output()` 回收阶段的对账凭证。

它不是最终输出，而是“本轮该怎么跑，以及回收时如何对账”。

### 29.3 ModelRunnerOutput

Worker / ModelRunner 执行完后返回的结果。

包含 sampled token、logprobs、pooling output、KV connector output、routing experts 等。

### 29.4 EngineCoreOutput

Scheduler 把 `ModelRunnerOutput` 消化后，返回给上层客户端的输出。

---

## 30. 一句话总结

`Scheduler` 的本质是一个“带 KV Cache 管理能力的 token 预算分配器”。它每一轮先推进 running 请求，再尝试接纳 waiting 请求；每个请求调度前先查本地/外部缓存，再分配 KV block，必要时抢占；Worker 返回后再更新请求状态、处理 stop/spec/grammar/KV transfer，并释放资源。

最核心的主链路是：

```text
add_request
  → waiting
  → schedule
  → prefix/KV cache lookup
  → allocate_slots
  → SchedulerOutput
  → ModelRunner forward
  → update_from_output
  → in-flight 回收 / append token / spec / grammar / KV&EC transfer / stop / free
```
