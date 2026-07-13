# 04. SchedulerOutput 在 EngineCore 中如何流动？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：`SchedulerOutput` 是如何由 Scheduler 生成，并在 EngineCore 中传给 Worker / ModelRunner，最后又回到 Scheduler 做结果对账的。

---

## 1. 一句话回答

`SchedulerOutput` 是 Scheduler 为某一轮 `EngineCore.step()` 生成的执行计划。

主线是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → ModelRunner / Worker forward
  → ModelRunnerOutput
  → Scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
```

一句话：

```text
SchedulerOutput 是这一轮 Worker 应该怎么执行的计划书；
EngineCore 负责把它交给 model_executor，
并在 Worker 返回结果后，把同一个 SchedulerOutput 和 ModelRunnerOutput 一起交回 Scheduler 对账。
```

所以 `SchedulerOutput` 在 EngineCore 中有两个作用：

```text
1. 向下：告诉 Worker / ModelRunner 本轮要执行什么；
2. 向上：告诉 Scheduler.update_from_output() 本轮之前发出去的计划是什么。
```

---

## 2. SchedulerOutput 是谁生成的

`SchedulerOutput` 由 `Scheduler.schedule()` 生成。

在 `EngineCore.step()` 中，调度入口是：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
```

位置：`vllm/vllm/v1/engine/core.py:499`

也就是说，EngineCore 不自己决定本轮哪些请求执行，而是把这个问题交给 Scheduler。

Scheduler 会根据当前内部状态决定：

```text
哪些 waiting 请求可以进入 running；
哪些 running 请求继续 decode / prefill；
每个请求本轮调度多少 token；
需要哪些 KV cache block；
需要哪些 encoder input；
是否带 spec decode tokens；
哪些请求已经 finished，需要通知 Worker 清理缓存；
是否需要 KV Connector / EC Connector metadata。
```

这些信息被打包到 `SchedulerOutput`。

`SchedulerOutput` 的定义在：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]
    preempted_req_ids: set[str] | None = None
    has_structured_output_requests: bool = False
    pending_structured_output_tokens: bool = False
    num_invalid_spec_tokens: dict[str, int] | None = None
    kv_connector_metadata: KVConnectorMetadata | None = None
    ec_connector_metadata: ECConnectorMetadata | None = None
    new_block_ids_to_zero: list[int] | None = None
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None
    num_spec_tokens_to_schedule: int = 0
```

位置：`vllm/vllm/v1/core/sched/output.py:182` 到 `vllm/vllm/v1/core/sched/output.py:250`

可以把它理解成：

```text
SchedulerOutput = Scheduler 发给 Worker 的一轮 batch 执行描述。
```

其中部分字段也供 EngineCore 和 Scheduler 回收阶段使用，例如 structured output / deferred sampling 相关字段。

注意，`Scheduler.schedule()` 生成 `SchedulerOutput` 后，Scheduler 内部通常已经做了乐观状态更新。

例如它会认为：

```text
这些 token 已经被调度出去了；
这些请求的 num_computed_tokens 可以先推进；
下一轮不要重复调度同一批 token。
```

后面 `Scheduler.update_from_output()` 会再用 Worker 的真实结果修正这些状态。

---

## 3. EngineCore 拿到 SchedulerOutput 后做什么

EngineCore 拿到 `SchedulerOutput` 后，第一件事是把它交给 `model_executor`：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:500`

这里体现了 EngineCore 和 Worker 的边界：

```text
EngineCore 不解析 SchedulerOutput 的所有字段，
也不直接执行模型 forward；
它把 SchedulerOutput 作为一轮执行计划交给 model_executor。
```

`model_executor` 再把这个计划传给实际的 Worker / ModelRunner。

普通 step 中的完整顺序是：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
...
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:499` 到 `vllm/vllm/v1/engine/core.py:515`

这说明 `SchedulerOutput` 在 EngineCore 中会经历三次关键使用：

```text
1. execute_model(scheduler_output)：
   发给 Worker / ModelRunner 执行。

2. get_grammar_bitmask(scheduler_output)：
   根据本轮调度请求计算结构化输出需要的 grammar bitmask。

3. update_from_output(scheduler_output, model_output)：
   和 Worker 返回结果一起交给 Scheduler 做状态更新。
```

也就是说：

```text
SchedulerOutput 不是一次性发给 Worker 就结束；
它还要在 Worker 返回后参与结果回收。
```

---

## 4. SchedulerOutput 中哪些字段影响 Worker 执行

`SchedulerOutput` 里的字段大致可以分成几类。

### 4.1 新请求数据：scheduled_new_reqs

```python
scheduled_new_reqs: list[NewRequestData]
```

位置：`vllm/vllm/v1/core/sched/output.py:187`

这是第一次被调度的请求。

Worker 侧还没有缓存这些请求的完整数据，所以需要把请求信息发过去，包括：

```text
req_id
prompt_token_ids
mm_features
sampling_params / pooling_params
block_ids
num_computed_tokens
lora_request
prompt_embeds
prompt_is_token_ids
```

对应定义：

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: torch.Tensor | None = None
    prompt_is_token_ids: list[bool] | None = None
    prefill_token_ids: list[int] | None = None
```

位置：`vllm/vllm/v1/core/sched/output.py:32` 到 `vllm/vllm/v1/core/sched/output.py:46`

可以理解为：

```text
scheduled_new_reqs 是 Worker 首次认识某个请求时需要的完整请求资料。
```

---

### 4.2 已缓存请求增量：scheduled_cached_reqs

```python
scheduled_cached_reqs: CachedRequestData
```

位置：`vllm/vllm/v1/core/sched/output.py:191`

对于之前已经调度过的请求，Worker 侧已经缓存了请求的静态信息，不需要每轮重复发送完整请求。

所以这里只发送增量信息，例如：

```text
req_ids
resumed_req_ids
new_token_ids
all_token_ids
new_block_ids
num_computed_tokens
num_output_tokens
```

对应定义：

```python
@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_token_ids: list[list[int]]
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
```

位置：`vllm/vllm/v1/core/sched/output.py:113` 到 `vllm/vllm/v1/core/sched/output.py:128`

可以理解为：

```text
scheduled_cached_reqs 是 Worker 已认识请求后的轻量更新包。
```

这样可以减少 EngineCore / Worker 之间的通信成本，尤其在多进程或分布式执行时更重要。

---

### 4.3 本轮 token 数：num_scheduled_tokens

```python
num_scheduled_tokens: dict[str, int]
total_num_scheduled_tokens: int
```

位置：`vllm/vllm/v1/core/sched/output.py:193` 到 `vllm/vllm/v1/core/sched/output.py:198`

这两个字段说明本轮到底安排了多少 token：

```text
num_scheduled_tokens：每个请求本轮调度多少 token；
total_num_scheduled_tokens：所有请求本轮调度 token 总数。
```

Worker / ModelRunner 会根据这些信息准备 batch、slot mapping、attention metadata、KV 写入位置等。

EngineCore 自己也使用 `total_num_scheduled_tokens` 判断本轮是否真正执行了模型：

```python
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:517`

所以：

```text
total_num_scheduled_tokens > 0：本轮有实际 token 计算；
total_num_scheduled_tokens == 0：可能只是清理 finished 请求或空调度。
```

---

### 4.4 Spec decode tokens：scheduled_spec_decode_tokens

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
```

位置：`vllm/vllm/v1/core/sched/output.py:202`

如果请求使用 speculative decoding，Scheduler 会把本轮要验证或参与执行的 draft tokens 放进这里。

它的含义是：

```text
req_id -> spec_token_ids
```

没有 spec decode tokens 的请求不会出现在这个字典里。

Worker / ModelRunner 会根据这些 draft tokens 执行对应的验证或计算。

后续 `Scheduler.update_from_output()` 会根据 Worker 返回的真实采样结果判断：

```text
哪些 spec tokens 被接受；
哪些 spec tokens 被拒绝；
需要如何修正请求的 token 进度和输出。
```

所以这个字段既影响 Worker 本轮怎么执行，也影响后续 Scheduler 如何对账。

---

### 4.5 Encoder inputs：scheduled_encoder_inputs

```python
scheduled_encoder_inputs: dict[str, list[int]]
```

位置：`vllm/vllm/v1/core/sched/output.py:206`

这个字段用于多模态或 encoder-decoder 相关输入。

源码注释举例：

```text
如果某个请求有 [0, 1]，可能表示当前 step 需要处理该请求的第 0 张和第 1 张图片。
```

也就是说：

```text
scheduled_encoder_inputs 告诉 Worker 本轮哪些 encoder input 需要处理。
```

Worker 执行完成后，相关 encoder cache 的释放也会通过 Scheduler / EngineCore 输出链路继续处理。

---

### 4.6 Prefix / KV / EC 相关元信息

`SchedulerOutput` 还包含一些缓存和 connector 相关字段：

```python
num_common_prefix_blocks: list[int]
kv_connector_metadata: KVConnectorMetadata | None = None
ec_connector_metadata: ECConnectorMetadata | None = None
new_block_ids_to_zero: list[int] | None = None
kv_cache_block_copies: list[KVCacheBlockCopy] | None = None
```

位置：`vllm/vllm/v1/core/sched/output.py:207` 到 `vllm/vllm/v1/core/sched/output.py:209`、`vllm/vllm/v1/core/sched/output.py:234` 到 `vllm/vllm/v1/core/sched/output.py:246`

含义分别是：

```text
num_common_prefix_blocks：
  本轮请求在各个 KV cache group 中共享的 prefix block 数，可用于 cascade attention。

kv_connector_metadata：
  KV transfer / KV connector 需要的元信息。

ec_connector_metadata：
  Encoder cache connector 需要的元信息。

new_block_ids_to_zero：
  本轮新分配、需要 Worker 先清零的 block IDs，避免旧数据污染计算。

kv_cache_block_copies：
  新 block 清零后、forward 前需要应用的 KV cache CoW copy 列表。
```

这些字段都不是 EngineCore 自己执行，而是随 `SchedulerOutput` 下发给 Worker / ModelRunner。

---

### 4.7 async / structured / dynamic spec 相关字段

`SchedulerOutput` 还包含一些用于异步调度、结构化输出和动态 speculative decoding 的字段：

```python
preempted_req_ids: set[str] | None = None
has_structured_output_requests: bool = False
pending_structured_output_tokens: bool = False
num_invalid_spec_tokens: dict[str, int] | None = None
num_spec_tokens_to_schedule: int = 0
```

位置：`vllm/vllm/v1/core/sched/output.py:219` 到 `vllm/vllm/v1/core/sched/output.py:250`

含义可以概括为：

```text
preempted_req_ids：
  v2 model runner 使用的本轮被抢占请求集合。

has_structured_output_requests / pending_structured_output_tokens：
  async scheduling / batch queue 场景下判断 grammar bitmask 是否能立即计算。

num_invalid_spec_tokens：
  structured output 过滤 draft tokens 后，用于调整 spec decode 接受率统计。

num_spec_tokens_to_schedule：
  dynamic speculative decoding 下，Scheduler 选出的下一步 spec token 数。
```

---

### 4.8 finished_req_ids 和 free_encoder_mm_hashes

```python
finished_req_ids: set[str]
free_encoder_mm_hashes: list[str]
```

位置：`vllm/vllm/v1/core/sched/output.py:211` 到 `vllm/vllm/v1/core/sched/output.py:217`

这两个字段用于通知 Worker 侧清理缓存：

```text
finished_req_ids：
  上一轮到当前轮之间已经结束的请求 ID，Worker 可以释放这些请求的缓存状态。

free_encoder_mm_hashes：
  可以从 encoder cache 中释放的多模态 encoder output 标识。
```

注意，这类字段不一定表示本轮要执行新的 token。

也就是说，某个 `SchedulerOutput` 即使 `total_num_scheduled_tokens == 0`，也可能携带清理类信息。

---

## 5. 空调度如何处理

空调度要分两种情况看。

### 5.1 Scheduler 没有任何请求

`EngineCore.step()` 开头先检查：

```python
if not self.scheduler.has_requests():
    return {}, False
```

位置：`vllm/vllm/v1/engine/core.py:495` 到 `vllm/vllm/v1/engine/core.py:498`

这种情况下 EngineCore 不会调用 `Scheduler.schedule()`，也不会生成 `SchedulerOutput`。

含义是：

```text
Scheduler 里已经没有未完成请求，也没有需要清理的请求状态；
本轮 EngineCore 什么都不做。
```

返回值是：

```text
({}, False)
```

---

### 5.2 Scheduler 还有请求，但本轮没有实际 token 执行

另一种情况是：

```text
scheduler.has_requests() 为 True，
但 scheduler_output.total_num_scheduled_tokens == 0。
```

这时仍然会生成 `SchedulerOutput`，并进入后续流程。

原因是 Scheduler 可能还有一些非 token 计算类工作要处理，例如：

```text
通知 Worker 清理 finished_req_ids；
释放 encoder cache；
处理 connector metadata；
处理已经 finished 但还没完全从 batch / Worker 状态中移除的请求。
```

EngineCore 判断本轮是否真正执行模型时用的是：

```python
scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:517`

因此：

```text
has_requests() 决定要不要进入 schedule；
total_num_scheduled_tokens 决定这一轮是否算作真正执行了模型 token。
```

### 5.3 batch queue 模式下的空输出

如果启用了 batch queue，`step_with_batch_queue()` 对空输出有额外处理。

源码注释说明：

```python
"""Schedule and execute batches with the batch queue.
Note that if nothing to output in this step, None is returned.
"""
```

位置：`vllm/vllm/v1/engine/core.py:528` 到 `vllm/vllm/v1/engine/core.py:532`

它会优先填充 batch queue：

```text
如果新 batch 被 schedule 进队列，
但还不需要阻塞等待 Worker 输出，
这一轮可以先返回 None。
```

对应代码：

```python
if len(batch_queue) < self.batch_queue_size and (
    model_executed or self.scheduler.has_requests()
):
    return None, model_executed
```

位置：`vllm/vllm/v1/engine/core.py:585` 到 `vllm/vllm/v1/engine/core.py:590`

所以 batch queue 模式中，`SchedulerOutput` 可能先进入队列，稍后等 Worker 结果回来时再和 `ModelRunnerOutput` 一起传给 `Scheduler.update_from_output()`。

这一点体现了 `SchedulerOutput` 的另一个重要作用：

```text
它是异步 / batch queue 场景下回收 Worker 输出时的对账凭证。
```

---

## 6. 和 update_from_output 的关系

`SchedulerOutput` 发给 Worker 后，不能丢。

Worker 返回的是 `ModelRunnerOutput`，它描述真实执行结果，例如：

```text
sampled_token_ids
logprobs
prompt_logprobs_dict
pooler_output
kv_connector_output
cudagraph_stats
```

但 Worker 输出本身并不足以让 Scheduler 完整更新状态。

Scheduler 还需要知道：

```text
这一轮原本调度了哪些请求；
每个请求原本调度了多少 token；
哪些 spec tokens 是这轮发出去的；
哪些 encoder inputs 是这轮处理的；
哪些请求是本轮需要清理的 finished 请求；
哪些 connector metadata 和这轮执行相关。
```

这些信息都在 `SchedulerOutput` 里。

所以 EngineCore 在 Worker 返回后调用：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:513` 到 `vllm/vllm/v1/engine/core.py:515`

它把：

```text
SchedulerOutput：本轮计划
ModelRunnerOutput：本轮结果
```

合并交给 Scheduler。

这就是：

```text
schedule() 是发任务；
execute_model() 是执行任务；
update_from_output() 是收结果并对账。
```

更详细的 `update_from_output()` 内部逻辑，可以继续看：

```text
../scheduler/08_update_after_worker_output.md
```

---

## 7. 总结

从 EngineCore 视角看，`SchedulerOutput` 的流动路径是：

```text
Scheduler.schedule()
  → 生成 SchedulerOutput
  → EngineCore 拿到执行计划
  → model_executor.execute_model(scheduler_output)
  → Worker / ModelRunner 根据计划 forward / sample / 处理 encoder / connector
  → Worker 返回 ModelRunnerOutput
  → EngineCore 调 Scheduler.update_from_output(scheduler_output, model_output)
  → Scheduler 更新请求状态并生成 EngineCoreOutputs
```

关键点是：

```text
SchedulerOutput 不是最终输出；
它是 Scheduler 发给 Worker 的执行计划。

EngineCore 不负责解释所有调度细节；
它负责把 SchedulerOutput 从 Scheduler 传给 Worker，
再把 SchedulerOutput 和 Worker 结果一起传回 Scheduler。

SchedulerOutput 既是执行计划，
也是 update_from_output 阶段的对账依据。
```

所以如果要回答：

```text
SchedulerOutput 在 EngineCore 中如何流动？
```

可以概括为：

```text
SchedulerOutput 由 Scheduler.schedule() 生成，
EngineCore 将它传给 model_executor / Worker 执行，
Worker 返回 ModelRunnerOutput 后，
EngineCore 再把原 SchedulerOutput 和 ModelRunnerOutput 一起交给 Scheduler.update_from_output()，
最终得到 EngineCoreOutputs 返回上层。
```
