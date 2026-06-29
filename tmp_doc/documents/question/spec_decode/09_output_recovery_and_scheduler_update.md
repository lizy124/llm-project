# 09. Scheduler 如何回收 spec decode 输出？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\outputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\__init__.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\output_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\logprobs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\rejection_sampler.py`

本问题关注：`RejectionSampler` 产出的 accepted / recovered / bonus tokens 如何经过 `ModelRunnerOutput` 回到 `Scheduler.update_from_output()`，Scheduler 如何据此修正 `num_computed_tokens` / `num_output_placeholders`、更新 `Request.output_token_ids`、推进 structured output grammar、切分 logprobs / routed experts，并最终构造 `EngineCoreOutput` 给前端 `OutputProcessor`。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录文档风格，本篇按“先讲输出形态，再走 Scheduler 回收主链路，最后落到 OutputProcessor 和例子”的方式梳理。

要回答的问题分成 13 组：

```text
1. RejectionSampler 输出到 Scheduler 前经历了什么？
2. ModelRunnerOutput 中哪些字段是 Scheduler 回收 spec decode 的依据？
3. EngineCore 何时调用 Scheduler.update_from_output()？
4. Scheduler.update_from_output() 如何找到 request 对应的 batch index？
5. generated_token_ids 和 scheduled_spec_decode_tokens 如何计算 accepted / rejected？
6. 为什么 rejected tokens 要回滚 num_computed_tokens / num_output_placeholders？
7. accepted / recovered / bonus tokens 如何写入 Request.output_token_ids？
8. stop / finish_reason / stop_reason 如何在多 token 输出中处理？
9. structured output grammar.accept_tokens() 如何推进状态？
10. logprobs / prompt_logprobs 如何随 spec decode 多 token 输出对齐？
11. routed experts 为什么在 spec decode 下取 scheduled range 的开头？
12. OutputProcessor 为什么基本不关心 token 来自普通 decode 还是 spec decode？
13. async scheduling / batch queue / KV connector 对输出回收有什么影响？
```

阅读顺序建议：

```text
06_rejection_sampler_flow.md
  → 07_kv_cache_and_num_computed_tokens.md
  → 08_structured_output_interaction.md
  → 09_output_recovery_and_scheduler_update.md
```

本篇不重复展开 `RejectionSampler` 的概率算法，也不重复展开 `SpecDecodeMetadata` 的 logits row 布局；重点是输出回收和状态落账。

---

## 1. 一句话回答

Spec decode 的输出回收不是“把 sampler token append 到 request”这么简单。

Scheduler 收到的是 `ModelRunnerOutput.sampled_token_ids` 中已经过滤后的真实输出 token list：

```text
accepted draft prefix
  + recovered token（某个 draft 被拒绝时）

或

all accepted draft tokens
  + bonus token（所有 draft 都接受时）
```

Scheduler 做的核心事情是：

```text
1. 用 req_id_to_index 找到每个请求在 ModelRunnerOutput 中的位置；
2. 取 generated_token_ids；
3. 结合 SchedulerOutput.scheduled_spec_decode_tokens 计算 accepted / rejected；
4. 用 rejected 数回滚 num_computed_tokens / async placeholders；
5. 把 generated_token_ids 写入 Request.output_token_ids；
6. 做 stop / grammar / logprobs / routed experts / cleanup；
7. 构造 EngineCoreOutput 交给 OutputProcessor。
```

一句话压缩：

```text
Scheduler.update_from_output() 是 spec decode 从“target 验证结果”变成“正式请求状态和用户输出”的落账点。
```

---

## 2. 整体链路

完整链路可以看成：

```text
GPUModelRunner.sample_tokens()
  → _sample(logits, spec_decode_metadata)
      → RejectionSampler.forward()
          → SamplerOutput.sampled_token_ids
             shape = [batch_size, max_spec_len + 1]
             padding = -1
  → _bookkeeping_sync()
      → RejectionSampler.parse_output()
      → valid_sampled_token_ids: list[list[int]]
      → ModelRunnerOutput.sampled_token_ids

EngineCore.step()
  → Scheduler.update_from_output(scheduler_output, model_output)
      → accepted / rejected 统计
      → Request.num_computed_tokens 回滚
      → Request.output_token_ids append
      → EngineCoreOutput

OutputProcessor.process_outputs()
  → detokenizer.update(new_token_ids)
  → logprobs_processor.update_from_output()
  → RequestOutput / CompletionOutput
```

关键边界：

```text
RejectionSampler：决定哪些 token 是真实输出。
ModelRunnerOutput：把真实输出 token list 交给 Scheduler。
Scheduler：把真实输出落到 Request 状态。
OutputProcessor：把已经落账的 token 转成用户可见输出。
```

---

## 3. RejectionSampler 输出在进入 Scheduler 前已经被过滤

`RejectionSampler.forward()` 返回的是 `SamplerOutput`：

```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
```

位置：`code/vllm/vllm/v1/outputs.py:185` 到 `code/vllm/vllm/v1/outputs.py:192`

spec decode 下，`sampled_token_ids` 的形状是：

```text
[batch_size, max_spec_len + 1]
```

其中：

```text
真实 token id：accepted / recovered / bonus
-1：padding，或者 rejected 后不再使用的位置
```

在 Scheduler 看到之前，Worker 侧 `_bookkeeping_sync()` 会先做过滤。

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3601`

关键逻辑：

```python
max_gen_len = sampled_token_ids.shape[-1]
if max_gen_len == 1:
    valid_sampled_token_ids = self._to_list(sampled_token_ids)
else:
    valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
        sampled_token_ids,
        self.input_batch.vocab_size,
        discard_sampled_tokens_req_indices,
        logprobs_tensors=logprobs_tensors,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3656` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3674`

也就是说：

```text
Scheduler.update_from_output() 一般不再看到 -1 padding。

它看到的是每个请求本轮真实有效的 generated_token_ids。
```

---

## 4. ModelRunnerOutput 是 Scheduler 的回收凭证

`ModelRunnerOutput` 定义在：`code/vllm/vllm/v1/outputs.py:234`

核心字段：

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    logprobs: LogprobsLists | None = None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(default_factory=dict)
    pooler_output: list[torch.Tensor | None] | None = None
    kv_connector_output: KVConnectorOutput | None = None
    ec_connector_output: ECConnectorOutput | None = None
    num_nans_in_logits: dict[str, int] | None = None
    cudagraph_stats: CUDAGraphStat | None = None
    routed_experts: RoutedExpertsLists | None = None
```

位置：`code/vllm/vllm/v1/outputs.py:234` 到后续字段定义

和 spec decode 输出回收最相关的是：

| 字段 | 用途 |
|---|---|
| `req_ids` | ModelRunner 输出时的 batch 请求顺序 |
| `req_id_to_index` | Scheduler 用 request id 查 batch index |
| `sampled_token_ids` | 每个请求本轮真实生成 token list，spec decode 下长度可大于 1 |
| `logprobs` | 与生成 token 对齐的 sample logprobs |
| `prompt_logprobs_dict` | 每个请求的 prompt logprobs |
| `kv_connector_output` | KV connector 回传，包括 invalid blocks / transfer stats |
| `routed_experts` | MoE routed experts 输出，spec decode 下切片方式特殊 |

构造位置在 `sample_tokens()` 收尾：

```python
output = ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    kv_connector_output=kv_connector_output,
    ec_connector_output=ec_connector_output,
    ...
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4610` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4617`

这里有两个重要点：

```text
1. sampled_token_ids 已经是 Python list[list[int]]，不是 GPU tensor。
2. req_id_to_index 是输出 snapshot，避免 async scheduling 下 input_batch 后续变化影响回收。
```

---

## 5. async scheduling 下 ModelRunnerOutput 何时填 sampled_token_ids

同步路径中，`_bookkeeping_sync()` 直接把 `valid_sampled_token_ids` 填进 `ModelRunnerOutput`。

async scheduling 下，`_bookkeeping_sync()` 先不阻塞做完整 D2H：

```python
else:
    valid_sampled_token_ids = []
    invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
    ...
    self.input_batch.prev_sampled_token_ids = sampled_token_ids
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3675` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3691`

随后 `sample_tokens()` 包装：

```python
AsyncGPUModelRunnerOutput(
    model_runner_output=output,
    sampled_token_ids=sampler_output.sampled_token_ids,
    logprobs_tensors=sampler_output.logprobs_tensors,
    invalid_req_indices=invalid_req_indices,
    ...
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4663` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4671`

真正回填发生在 `AsyncGPUModelRunnerOutput.get_output()`：

```python
if max_gen_len == 1:
    valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
else:
    valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(...)

output.sampled_token_ids = valid_sampled_token_ids
output.logprobs = logprobs_lists
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:293` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:310`

所以：

```text
无论 sync 还是 async，Scheduler.update_from_output() 最终拿到的 ModelRunnerOutput 都已经是可消费的 CPU 结构。
```

---

## 6. EngineCore 何时调用 update_from_output

同步 `EngineCore.step()` 中：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)

engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`code/vllm/vllm/v1/engine/core.py:479` 到 `code/vllm/vllm/v1/engine/core.py:508`

这说明：

```text
1. SchedulerOutput 是本轮调度计划；
2. ModelRunnerOutput 是本轮执行结果；
3. update_from_output() 用这两个对象做对账。
```

batch queue 路径也一样，模型输出回来后调用：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`code/vllm/vllm/v1/engine/core.py:602` 到 `code/vllm/vllm/v1/engine/core.py:607`

---

## 7. update_from_output 的入口和输入拆包

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

开头拆出：

```python
sampled_token_ids = model_runner_output.sampled_token_ids
logprobs = model_runner_output.logprobs
prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
num_scheduled_tokens = scheduler_output.num_scheduled_tokens
pooler_outputs = model_runner_output.pooler_output
num_nans_in_logits = model_runner_output.num_nans_in_logits
kv_connector_output = model_runner_output.kv_connector_output
cudagraph_stats = model_runner_output.cudagraph_stats
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1468` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1475`

这些字段后面分别用于：

```text
sampled_token_ids：输出 token 回收。
logprobs：生成 token logprobs 切片。
prompt_logprobs_dict：prompt logprobs 回传。
num_scheduled_tokens：遍历本轮调度请求、计算 routed experts offsets。
pooler_outputs：pooling 请求输出。
kv_connector_output：KV transfer / invalid block 处理。
cudagraph_stats / perf_stats：统计输出。
```

---

## 8. update_from_output 先处理 deferred frees 和 KV invalid blocks

### 8.1 deferred block free

如果启用了延迟释放 KV blocks：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1477` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1481`

含义：

```text
到 update_from_output 时，本轮及更早 GPU writes 已完成，
可以安全把之前 deferred 的 blocks 还回池子。
```

spec decode 一轮可能写多个 KV slots，因此 async / KV connector 下尤其需要这个 fence。

### 8.2 invalid KV blocks

如果 KV connector 报告外部 KV 加载失败：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(
        kv_connector_output.invalid_block_ids,
        num_scheduled_tokens,
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1490` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1498`

这不是 spec decode 专属，但和输出回收顺序有关：

```text
如果请求的 KV load 失败，它可能被跳过本轮 token 回收，
转而触发 recompute 或 finish error。
```

---

## 9. Scheduler 如何定位每个请求的 generated_token_ids

`update_from_output()` 遍历本轮调度请求：

```python
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
    ...
    req_index = model_runner_output.req_id_to_index[req_id]
    generated_token_ids = (
        sampled_token_ids[req_index] if sampled_token_ids else []
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1526` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1545`

这里不要直接假设 Scheduler 的 dict 顺序等于 Worker batch 顺序。

正确对齐依赖：

```text
req_id → model_runner_output.req_id_to_index → sampled_token_ids[index]
```

这就是 `ModelRunnerOutput.req_id_to_index` 必须存在的原因。

如果 request 已经在执行过程中被 abort / finish：

```python
request = self.requests.get(req_id)
if request is None or request.is_finished():
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1531` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1540`

也就是说：

```text
模型可能已经返回了某个请求的 token，
但如果请求执行期间被 abort，Scheduler 不再把这些 token 落账。
```

---

## 10. scheduled_spec_decode_tokens 是 accepted / rejected 统计的另一半

取完 `generated_token_ids` 后，Scheduler 取本轮调度的 draft tokens：

```python
scheduled_spec_token_ids = (
    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1547` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1549`

两个对象的语义不同：

| 对象 | 来源 | 含义 |
|---|---|---|
| `scheduled_spec_token_ids` | SchedulerOutput | 本轮交给 target model 验证的 draft tokens |
| `generated_token_ids` | ModelRunnerOutput | RejectionSampler 后真正输出的 tokens |

可以理解为：

```text
scheduled_spec_token_ids：本轮候选。
generated_token_ids：候选验证后的结果。
```

Scheduler 的回收逻辑就是用这两个 list 做对账。

---

## 11. accepted / rejected 数如何计算

如果本轮有 scheduled spec tokens，并且有输出：

```python
if scheduled_spec_token_ids and (
    generated_token_ids or self.num_sampled_tokens_per_step == 0
):
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_sampled = self.num_sampled_tokens_per_step
    num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    num_rejected = num_draft_tokens - num_accepted
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1550` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1556`

通常 `num_sampled_tokens_per_step = 1`，所以：

```text
len(generated_token_ids)
  = accepted draft token 数
    + 1 个 recovered / bonus / normal sampled token
```

因此：

```text
num_accepted = max(len(generated_token_ids) - 1, 0)
num_rejected = len(scheduled_spec_token_ids) - num_accepted
```

为什么要减 `num_sampled`？

```text
因为 generated_token_ids 的最后一个 token 通常不是“被接受的 draft token”：

- draft 被拒绝时，它是 recovered / replacement token；
- draft 全接受时，它是 bonus token；
- 无 draft 或特殊场景下，它是普通 sampled token。
```

---

## 12. 为什么 Scheduler 不需要知道每个 token 的标签

`generated_token_ids` 里没有显式标记：

```text
这个 token 是 accepted draft；
这个 token 是 recovered；
这个 token 是 bonus。
```

Scheduler 只根据长度推断 accepted / rejected 数。

原因是 RejectionSampler 的输出形态有固定规则：

```text
部分接受：
  [accepted draft prefix..., recovered token]

全部接受：
  [all accepted draft tokens..., bonus token]
```

所以只要知道：

```text
本轮验证了几个 draft tokens；
最终输出了几个 tokens；
每步正常 sample 几个 tokens；
```

就能得到：

```text
accepted draft 数；
rejected draft 数。
```

这是一种紧凑协议：

```text
RejectionSampler 不传 accepted bitmap；
Scheduler 用输出长度和 scheduled draft 长度恢复统计。
```

---

## 13. rejected tokens 如何回滚 num_computed_tokens

Scheduler 在 `_update_after_schedule()` 中已经乐观推进过：

```text
request.num_computed_tokens += num_scheduled_tokens[req_id]
```

所以如果 draft 被拒绝，必须回滚。

`update_from_output()` 中：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1557` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1563`

含义：

```text
被拒绝的 draft tokens 虽然本轮 target forward 可能计算过，
但它们不能成为真实上下文进度，
所以下一轮 Scheduler 不能从这些 rejected token 后面继续。
```

这和 `07_kv_cache_and_num_computed_tokens.md` 的核心原则一致：

```text
draft token 可以被计算，
但只有 accepted / finalized token 才能推进真实进度。
```

---

## 14. async placeholders 也要按 rejected 数回滚

async scheduling 下，`num_output_placeholders` 也可能包含 spec tokens 的占位。

因此：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1564` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1567`

原因：

```text
Scheduler 可能提前为下一步输出占住位置，
但 rejected draft tokens 不会成为真实输出，
因此对应 placeholder 也必须释放 / 回退。
```

否则会导致：

```text
1. request.num_tokens_with_spec / num_computed_tokens 计算偏大；
2. 后续 positions 偏移；
3. structured output / encoder cache free 判断错误；
4. async spec decode 下一轮继续基于错误 token 边界调度。
```

---

## 15. spec decode stats 如何更新

Scheduler 会在回滚后更新 speculative decoding 统计：

```python
spec_decoding_stats = self.make_spec_decoding_stats(
    spec_decoding_stats,
    num_draft_tokens=num_draft_tokens,
    num_accepted_tokens=num_accepted,
    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
    request_id=req_id,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1568` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1574`

统计输入包括：

```text
num_draft_tokens：本轮验证 draft 总数；
num_accepted_tokens：target 接受的 draft 数；
num_invalid_spec_tokens：structured output 等提前判定非法的 draft 数。
```

`num_invalid_spec_tokens` 的意义：

```text
把 grammar 裁掉的无效 draft 和 target rejection 区分开，
避免 acceptance rate 统计被结构化输出误伤。
```

---

## 16. generated_token_ids 如何正式写入 Request

统计和回滚之后：

```python
new_token_ids = generated_token_ids
...
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1580` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1592`

`_update_request_with_output()` 定义在：`code/vllm/vllm/v1/core/sched/scheduler.py:1848`

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]
        break
return new_token_ids, stopped
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1848` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1864`

这一步非常关键：

```text
scheduled_spec_decode_tokens 本身不会直接 append。
只有 generated_token_ids 会进入 Request.output_token_ids。
```

所以：

```text
accepted draft tokens：会进入 output，因为它们在 generated_token_ids 中。
recovered token：会进入 output，因为它替代 rejected draft。
bonus token：会进入 output，因为它是 all accepted 后的额外 target token。
rejected draft tokens：不会进入 output，因为 parse_output 后它们不在 generated_token_ids 中。
```

---

## 17. 多 token 输出下 stop 如何处理

spec decode 一轮可能返回多个 tokens。

`_update_request_with_output()` 会逐个 append 并逐个检查 stop：

```text
for token in new_token_ids:
  append token
  check_stop()
  if stopped:
    trim 后续 token
    break
```

这意味着：

```text
如果 generated_token_ids = [A, B, C, D]，
但 B 后触发 stop，
Scheduler 会把 C/D 从本轮 new_token_ids 中裁掉。
```

裁剪发生在：

```python
del new_token_ids[num_new:]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1861` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1863`

这样可以保证：

```text
1. Request.output_token_ids 不包含 stop 之后多余 token；
2. EngineCoreOutput.new_token_ids 和真正提交的 token 一致；
3. OutputProcessor 不需要再理解 spec decode 多 token 裁剪。
```

---

## 18. finish_reason 和 stop_reason 在哪里确定

如果 `_update_request_with_output()` 返回 `stopped=True`：

```python
if stopped:
    finish_reason = request.get_finished_reason()
    finished = self._handle_stopped_request(request)
    if finished:
        kv_transfer_params = self._free_request(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1655` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1663`

`EngineCoreOutput` 里会带：

```python
finish_reason=finish_reason,
stop_reason=request.stop_reason,
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1690` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1697`

也就是说：

```text
stop / length / repetition 等结束原因在 Scheduler 侧确定；
OutputProcessor 只把这些字段转换成最终 RequestOutput / CompletionOutput。
```

注意 stop string 的 detokenization 检查还会在 OutputProcessor 侧补充处理，见后文。

---

## 19. structured output grammar 在输出回收后真正推进

如果请求使用 structured output，Scheduler 会在 token 已经写入 request 后推进 grammar：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    struct_output_request = request.structured_output_request
    assert struct_output_request is not None
    assert struct_output_request.grammar is not None
    if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
        logger.error(...)
        request.status = RequestStatus.FINISHED_ERROR
        request.resumable = False
        stopped = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1598` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1614`

这里推进的是最终 `new_token_ids`，包括：

```text
accepted draft tokens；
recovered token；
bonus token。
```

不包括：

```text
rejected draft tokens。
```

这和 grammar 的状态语义一致：

```text
只有真正成为输出的 token 才能改变 grammar 状态。
```

如果 grammar 拒绝这些已经采样出来的 tokens，说明前面的 `validate_tokens()` / `grammar_bitmask()` / logits row 对齐出现不一致，vLLM 会把请求置为 `FINISHED_ERROR`。

---

## 20. sample logprobs 如何与 spec decode 多 token 输出对齐

如果请求需要 sample logprobs：

```python
if (
    request.sampling_params is not None
    and request.sampling_params.num_logprobs is not None
    and logprobs
):
    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1669` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1675`

`LogprobsLists` 支持 spec decode 的不同请求不同输出长度：

```python
class LogprobsLists(NamedTuple):
    logprob_token_ids: np.ndarray
    logprobs: np.ndarray
    sampled_token_ranks: np.ndarray
    cu_num_generated_tokens: list[int] | None = None
```

位置：`code/vllm/vllm/v1/outputs.py:27` 到 `code/vllm/vllm/v1/outputs.py:38`

`slice_request(req_index, num_positions)` 会按请求切出当前请求的 logprobs。

在 spec decode 下，重点是：

```text
len(new_token_ids) 可能大于 1；
logprobs 也必须按实际有效 tokens 数切片；
如果 stop 裁掉了后续 token，logprobs 也只切裁剪后的长度。
```

---

## 21. prompt logprobs 如何回传

Scheduler 从 `ModelRunnerOutput.prompt_logprobs_dict` 取当前请求的 prompt logprobs：

```python
prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1680` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1681`

然后放进 `EngineCoreOutput`：

```python
new_prompt_logprobs_tensors=prompt_logprobs_tensors,
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1694` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1696`

prompt logprobs 和 spec decode 输出 token 的关系较弱：

```text
prompt logprobs 属于 prefill / prompt 位置；
spec decode 主要影响 generated token logprobs 的多 token 对齐。
```

---

## 22. routed experts 在 spec decode 下为什么取 scheduled range 开头

如果开启 routed experts 返回，Scheduler 先保存 batch routing data：

```python
if model_runner_output.routed_experts is not None:
    re = model_runner_output.routed_experts
    self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
    routing_data = re.routing_data.astype(...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1500` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1513`

随后为每个请求构造 offset：

```python
for rid in model_runner_output.req_ids:
    routing_offsets[rid] = offset
    offset += num_scheduled_tokens[rid]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1514` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1519`

普通 decode 和 spec decode 的切片不同：

```python
if scheduled_spec_token_ids:
    # Spec decode: accepted tokens at the START of
    # the scheduled range, rejected at the end.
    routed_experts = routing_data[
        req_offset : req_offset + len(new_token_ids)
    ]
else:
    # Normal decode / re-prefill: token(s) at the END.
    routed_experts = routing_data[end - len(new_token_ids) : end]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1645` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1653`

原因：

```text
普通 decode / re-prefill：本轮输出 token 通常对应 scheduled range 的末尾。

spec decode：accepted prefix 位于 scheduled range 的开头，
rejected draft tokens 位于后面，真实输出要取前缀。
```

这是 spec decode 影响输出回收的一个细节：

```text
不只是 token 数变了，输出 token 对应 scheduled range 中的位置也变了。
```

---

## 23. EngineCoreOutput 是 Scheduler 给前端的正式输出单位

`EngineCoreOutput` 定义在：`code/vllm/vllm/v1/engine/__init__.py:173`

关键字段：

```python
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None
    pooling_output: torch.Tensor | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    kv_transfer_params: dict[str, Any] | None = None
    routed_experts: np.ndarray | None = None
    num_nans_in_logits: int = 0
```

位置：`code/vllm/vllm/v1/engine/__init__.py:173` 到 `code/vllm/vllm/v1/engine/__init__.py:199`

Scheduler 构造位置：

```python
outputs[request.client_index].append(
    EngineCoreOutput(
        request_id=req_id,
        new_token_ids=new_token_ids,
        finish_reason=finish_reason,
        new_logprobs=new_logprobs,
        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
        pooling_output=pooler_output,
        stop_reason=request.stop_reason,
        ...
    )
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1688` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1704`

注意：

```text
EngineCoreOutput.new_token_ids 已经是 Scheduler 确认后的 token。

它不再携带 scheduled_spec_decode_tokens，
也不再携带 accepted / rejected 的中间信息。
```

---

## 24. 什么时候不会产生 EngineCoreOutput

Scheduler 只有在请求有实际输出或状态变化时才 append：

```python
if (
    new_token_ids
    or pooler_output is not None
    or kv_transfer_params
    or stopped
):
    outputs[request.client_index].append(EngineCoreOutput(...))
else:
    assert not prompt_logprobs_tensors
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1680` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1708`

因此：

```text
partial prefill 没有生成 token 时，不会给用户发 partial output；
spec decode 如果生成了 token，一次可能在一个 EngineCoreOutput 中携带多个 new_token_ids。
```

---

## 25. EngineCoreOutputs 按 client_index 分组返回

`update_from_output()` 最后把每个 client 的 outputs 包装成：

```python
engine_core_outputs = {
    client_index: EngineCoreOutputs(outputs=outs)
    for client_index, outs in outputs.items()
}
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1769` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1774`

`EngineCoreOutputs` 定义：

```python
class EngineCoreOutputs:
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
```

位置：`code/vllm/vllm/v1/engine/__init__.py:218` 到 `code/vllm/vllm/v1/engine/__init__.py:235`

所以回收后输出层级是：

```text
client_index
  → EngineCoreOutputs
      → list[EngineCoreOutput]
          → request_id + new_token_ids + finish/logprobs/metadata
```

---

## 26. OutputProcessor 不需要理解 spec decode 细节

前端 `OutputProcessor` 入口：`code/vllm/vllm/v1/engine/output_processor.py:576`

```python
def process_outputs(
    self,
    engine_core_outputs: list[EngineCoreOutput],
    ...
) -> OutputProcessorOutput:
```

它在循环里取：

```python
new_token_ids = engine_core_output.new_token_ids
pooling_output = engine_core_output.pooling_output
finish_reason = engine_core_output.finish_reason
stop_reason = engine_core_output.stop_reason
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:618` 到 `code/vllm/vllm/v1/engine/output_processor.py:622`

然后对 generation 请求：

```python
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
...
req_state.logprobs_processor.update_from_output(engine_core_output)
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:635` 到 `code/vllm/vllm/v1/engine/output_processor.py:648`

这说明：

```text
OutputProcessor 只看到 Scheduler 已经确认过的 new_token_ids。
它不关心这些 token 是普通 sampler 生成的，还是 spec decode 接受 / recover / bonus 得到的。
```

---

## 27. OutputProcessor 如何处理多 token new_token_ids

`detokenizer.update(new_token_ids, ...)` 可以一次接收多个 token ids。

之后构造输出：

```python
request_output := req_state.make_request_output(
    new_token_ids,
    pooling_output,
    finish_reason,
    stop_reason,
    kv_transfer_params,
)
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:650` 到 `code/vllm/vllm/v1/engine/output_processor.py:657`

`make_request_output()` 中会根据 output kind 决定返回 delta 还是全量。

对于 completion：

```python
output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:319`

`_new_completion_output()` 中：

```python
text = self.detokenizer.get_next_output_text(finished, delta)
if not delta:
    token_ids = self.detokenizer.output_token_ids
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:376` 到 `code/vllm/vllm/v1/engine/output_processor.py:390`

所以 spec decode 对前端表现为：

```text
一次 EngineCoreOutput 可能带多个 new_token_ids；
DELTA 模式返回这一批新增 token 对应文本；
非 DELTA 模式返回累计 token / 文本。
```

---

## 28. OutputProcessor 侧 logprobs 如何更新

`OutputProcessor` 调用：

```python
req_state.logprobs_processor.update_from_output(engine_core_output)
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:646` 到 `code/vllm/vllm/v1/engine/output_processor.py:648`

`LogprobsProcessor.update_from_output()` 很薄：

```python
def update_from_output(self, output: EngineCoreOutput) -> None:
    if output.new_logprobs is not None:
        self._update_sample_logprobs(output.new_logprobs)
    if output.new_prompt_logprobs_tensors is not None:
        self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
```

位置：`code/vllm/vllm/v1/engine/logprobs.py:348` 到 `code/vllm/vllm/v1/engine/logprobs.py:352`

`_update_sample_logprobs()` 的 docstring 特别提到：

```text
Outer lists are only of len > 1 if EngineCore made
>1 tokens in prior step (e.g. in spec decoding).
```

位置：`code/vllm/vllm/v1/engine/logprobs.py:69` 到 `code/vllm/vllm/v1/engine/logprobs.py:78`

这说明 OutputProcessor 的 logprobs 处理已经支持 spec decode 多 token 输出。

---

## 29. stop string 仍在 OutputProcessor 侧做文本级检查

Scheduler 的 `check_stop()` 处理 token 级 / 长度等停止逻辑。

OutputProcessor 还会在 detokenize 后检查 stop string：

```python
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
if stop_string:
    finish_reason = FinishReason.STOP
    stop_reason = stop_string
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:635` 到 `code/vllm/vllm/v1/engine/output_processor.py:644`

这说明：

```text
Scheduler 确认 token 序列；
OutputProcessor 可以基于 detokenized text 进一步确认 stop string。
```

即使 spec decode 一次返回多个 tokens，也会作为一个连续 token 序列交给 detokenizer。

---

## 30. 下一轮 draft tokens 的写回发生在 post_step

同步 scheduling 下，输出回收之后，EngineCore 的 `post_step()` 会拿下一轮 draft tokens：

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`code/vllm/vllm/v1/engine/core.py:510` 到 `code/vllm/vllm/v1/engine/core.py:517`

这和 `update_from_output()` 是两个方向：

```text
update_from_output():
  ModelRunnerOutput.sampled_token_ids
    → Request.output_token_ids

update_draft_token_ids():
  DraftTokenIds
    → Request.spec_token_ids
```

所以完整闭环是：

```text
本轮输出落账
  → request.output_token_ids 更新
  → drafter 基于新上下文生成下一轮 draft
  → request.spec_token_ids 更新
  → 下一轮 schedule 消费 draft
```

---

## 31. batch queue / structured output 下的 deferred draft 修正

batch queue 路径中，如果采样被 deferred，EngineCore 会在计算 grammar bitmask 前先修正 scheduled spec tokens：

```python
if deferred_scheduler_output:
    if self.check_for_draft_tokens:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )
    grammar_output = self.scheduler.get_grammar_bitmask(
        deferred_scheduler_output
    )
    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
```

位置：`code/vllm/vllm/v1/engine/core.py:612` 到 `code/vllm/vllm/v1/engine/core.py:630`

`update_draft_token_ids_in_output()` 不是写 `Request.spec_token_ids`，而是直接修正已经生成的 `SchedulerOutput.scheduled_spec_decode_tokens`。

关键逻辑：

```python
orig_num_spec_tokens = len(placeholder_spec_tokens)
del spec_token_ids[orig_num_spec_tokens:]
...
spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
...
num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
if num_invalid_tokens:
    spec_token_ids.extend([-1] * num_invalid_tokens)
    num_invalid_spec_tokens[req_id] = num_invalid_tokens

sched_spec_tokens[req_id] = spec_token_ids
scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1917` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1953`

这对输出回收的影响是：

```text
num_invalid_spec_tokens 会在 update_from_output() 中传给 make_spec_decoding_stats，
用来修正 acceptance 统计。
```

---

## 32. 示例：全部 draft 被接受

假设本轮：

```text
scheduled_spec_token_ids = [A, B, C]
bonus token = D
RejectionSampler raw output = [A, B, C, D]
```

`parse_output()` 后：

```text
generated_token_ids = [A, B, C, D]
```

Scheduler 计算：

```text
num_draft_tokens = 3
num_sampled = 1
num_accepted = len([A,B,C,D]) - 1 = 3
num_rejected = 3 - 3 = 0
```

落账：

```text
num_computed_tokens 不回滚；
Request.output_token_ids append [A, B, C, D]；
grammar.accept_tokens([A, B, C, D])；
EngineCoreOutput.new_token_ids = [A, B, C, D]。
```

语义：

```text
A/B/C 是 accepted draft；
D 是 bonus token；
四个 token 都成为正式输出。
```

---

## 33. 示例：部分接受后拒绝

假设本轮：

```text
scheduled_spec_token_ids = [A, B, C]
A/B accepted
C rejected
recovered token = X
RejectionSampler raw output = [A, B, X, -1]
```

`parse_output()` 后：

```text
generated_token_ids = [A, B, X]
```

Scheduler 计算：

```text
num_draft_tokens = 3
num_sampled = 1
num_accepted = len([A,B,X]) - 1 = 2
num_rejected = 3 - 2 = 1
```

落账：

```text
request.num_computed_tokens -= 1；
request.num_output_placeholders -= 1（如果 async placeholder 存在）；
Request.output_token_ids append [A, B, X]；
EngineCoreOutput.new_token_ids = [A, B, X]。
```

语义：

```text
A/B 是 accepted draft；
C 不进入输出；
X 是 target / residual distribution 采出的替代 token。
```

---

## 34. 示例：第一个 draft 就拒绝

假设本轮：

```text
scheduled_spec_token_ids = [A, B, C]
A rejected
recovered token = X
RejectionSampler raw output = [X, -1, -1, -1]
```

`parse_output()` 后：

```text
generated_token_ids = [X]
```

Scheduler 计算：

```text
num_accepted = len([X]) - 1 = 0
num_rejected = 3 - 0 = 3
```

落账：

```text
request.num_computed_tokens -= 3；
Request.output_token_ids append [X]；
rejected draft A/B/C 都不会进入 output。
```

这说明：

```text
即使 target model 本轮验证了多个 draft 位置，
只要第一个 draft 被拒绝，真实输出也只包含 recovered token。
```

---

## 35. 示例：spec decode 输出中途触发 stop

假设：

```text
generated_token_ids = [A, B, C, D]
B 触发 stop
```

`_update_request_with_output()` 行为：

```text
append A，未 stop；
append B，触发 stop；
裁掉 C/D。
```

最终：

```text
new_token_ids = [A, B]
EngineCoreOutput.new_token_ids = [A, B]
finish_reason = request.get_finished_reason()
```

注意：

```text
accepted / rejected 数的统计发生在裁剪前，
输出给用户的 new_token_ids 是裁剪后的正式 token。
```

这通常没问题，因为 stop 后请求结束，后续 token 不再对用户可见；请求清理和 KV 释放由 finish 逻辑处理。

---

## 36. 示例：mixed batch 中无 draft 请求

如果整个 batch 有 spec decode metadata，但某个请求本轮没有 draft tokens：

```text
scheduled_spec_token_ids = None 或 []
generated_token_ids = [T]
```

Scheduler 不进入 accepted / rejected 统计分支：

```text
不计算 num_accepted / num_rejected；
不回滚 num_computed_tokens；
直接 append [T]。
```

这让 mixed batch 可以统一走 `ModelRunnerOutput.sampled_token_ids`，但只有有 `scheduled_spec_decode_tokens[req_id]` 的请求才执行 spec decode 回滚逻辑。

---

## 37. KV connector 输出回收和 spec decode 的关系

`ModelRunnerOutput.kv_connector_output` 在 `update_from_output()` 中有几类作用：

```text
1. invalid_block_ids：触发 computed token 回退 / recompute / error；
2. transfer finished：更新 scheduler-side connector 状态；
3. kv_connector_stats：合并到 Scheduler stats；
4. kv_transfer_params：请求结束时随 EngineCoreOutput 回前端。
```

相关位置：

```text
invalid blocks：scheduler.py:1490 到 scheduler.py:1498
_update_from_kv_xfer_finished：scheduler.py:1731 到 scheduler.py:1733
kv_connector_stats：scheduler.py:1735 到 scheduler.py:1750
kv_transfer_params 输出：scheduler.py:1660 到 scheduler.py:1701
```

和 spec decode 的共同点：

```text
它们都会影响 num_computed_tokens / KV block 生命周期。
```

不同点：

```text
spec decode 回滚来自 target 对 draft 的接受 / 拒绝；
KV invalid 回退来自外部 KV load failure。
```

---

## 38. Scheduler 与 OutputProcessor 的职责边界

### 38.1 Scheduler 负责

```text
读取 ModelRunnerOutput；
按 request id 对齐 batch index；
计算 accepted / rejected；
回滚 num_computed_tokens / placeholders；
append Request.output_token_ids；
检查 stop / finish；
推进 structured output grammar；
切分 logprobs / routed experts；
清理 finished request / KV blocks；
构造 EngineCoreOutput。
```

### 38.2 OutputProcessor 负责

```text
接收 EngineCoreOutput；
detokenize new_token_ids；
处理 stop string；
累积 sample / prompt logprobs；
组装 RequestOutput / CompletionOutput；
处理 streaming / queue / final-only 输出策略。
```

边界一句话：

```text
Scheduler 消化 spec decode 语义，OutputProcessor 消费普通的 confirmed token stream。
```

---

## 39. 容易混淆的点

### 39.1 `sampled_token_ids` 是 draft tokens 吗？

不是。

`ModelRunnerOutput.sampled_token_ids` 是 RejectionSampler 后的真实输出：

```text
accepted draft tokens + recovered / bonus token。
```

原始待验证 draft tokens 在：

```text
SchedulerOutput.scheduled_spec_decode_tokens。
```

### 39.2 rejected draft 会出现在 Scheduler 的 generated_token_ids 里吗？

不会。

rejected 后的位置在 `SamplerOutput.sampled_token_ids` 中是 `-1` padding，`RejectionSampler.parse_output()` 会过滤掉。

### 39.3 Scheduler 是否知道哪个 token 是 bonus？

没有显式标签。

Scheduler 通过：

```text
len(generated_token_ids) - num_sampled_tokens_per_step
```

推断 accepted draft 数。

### 39.4 为什么不直接让 RejectionSampler 更新 Request？

因为 RejectionSampler 在 Worker / ModelRunner 侧，只负责采样和验证。

Request 的权威状态在 Scheduler 侧，包括：

```text
output_token_ids；
num_computed_tokens；
finish status；
KV block lifecycle；
grammar 状态。
```

所以必须由 Scheduler 统一落账。

### 39.5 OutputProcessor 会区分 accepted / recovered / bonus 吗？

不会。

到 OutputProcessor 时，这些差异已经被 Scheduler 消化成：

```text
EngineCoreOutput.new_token_ids
finish_reason
stop_reason
new_logprobs
```

### 39.6 logprobs 为什么要 slice_request(req_index, len(new_token_ids))？

因为 spec decode 下每个请求本轮输出 token 数可能不同。

`len(new_token_ids)` 是当前请求真正提交给用户的 token 数，也是 logprobs 应该对应的位置数。

---

## 40. 从“回答问题”的角度总结

如果要问：

```text
Scheduler 如何回收 spec decode 输出？
```

可以回答：

```text
ModelRunner 侧 RejectionSampler 会把 target model 验证后的结果整理成每个请求的 generated_token_ids：它只包含 accepted draft tokens，以及拒绝时的 recovered token 或全接受时的 bonus token。padding 和 rejected draft tokens 会在 RejectionSampler.parse_output() 阶段过滤掉，最终进入 ModelRunnerOutput.sampled_token_ids。

EngineCore 拿到 ModelRunnerOutput 后调用 Scheduler.update_from_output(scheduler_output, model_output)。Scheduler 对每个本轮调度的 request，用 model_runner_output.req_id_to_index 找到 generated_token_ids，再读取 scheduler_output.scheduled_spec_decode_tokens 中本轮验证的 draft tokens。若存在 spec tokens，则用 len(generated_token_ids) - num_sampled_tokens_per_step 推出 accepted draft 数，用 scheduled draft 总数减 accepted 数得到 rejected 数，并把 rejected 数从 request.num_computed_tokens 和 async num_output_placeholders 中减掉。

随后 Scheduler 把 generated_token_ids 逐个 append 到 Request.output_token_ids，并在每个 token 后检查 stop。如果中途 stop，会裁掉后续 token。对 structured output 请求，Scheduler 用最终 new_token_ids 调用 grammar.accept_tokens() 推进 grammar 状态。最后 Scheduler 切分 logprobs / prompt logprobs / routed experts，构造 EngineCoreOutput。OutputProcessor 只消费 EngineCoreOutput.new_token_ids，做 detokenize、stop string 检查、logprobs 累积和 RequestOutput 组装，不再关心 token 来自普通 decode 还是 spec decode。
```

职责关系可以概括为：

```text
SchedulerOutput.scheduled_spec_decode_tokens：本轮验证了哪些 draft。
ModelRunnerOutput.sampled_token_ids：验证后真正输出哪些 token。
Scheduler.update_from_output()：根据两者计算 accepted/rejected 并落账。
EngineCoreOutput.new_token_ids：已经确认、可交给前端的 token stream。
OutputProcessor：把 confirmed token stream 转成用户输出。
```

---

## 41. 最关键流程图

```text
Worker / ModelRunner

RejectionSampler.forward()
  → SamplerOutput.sampled_token_ids
      [accepted..., recovered/bonus, -1 padding]
  → _bookkeeping_sync()
      ├─ if max_gen_len == 1:
      │    _to_list(sampled_token_ids)
      └─ else:
           RejectionSampler.parse_output()
             → filter -1
             → valid_sampled_token_ids
  → ModelRunnerOutput(
       req_id_to_index,
       sampled_token_ids=valid_sampled_token_ids,
       logprobs,
       prompt_logprobs_dict,
       ...
    )
```

```text
EngineCore

scheduler_output = Scheduler.schedule()
model_output = model_executor.execute_model/sample_tokens(...)
Scheduler.update_from_output(scheduler_output, model_output)
```

```text
Scheduler.update_from_output()

for req_id in scheduler_output.num_scheduled_tokens:
  req_index = model_runner_output.req_id_to_index[req_id]
  generated_token_ids = sampled_token_ids[req_index]
  scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)

  if scheduled_spec_token_ids:
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    num_rejected = num_draft_tokens - num_accepted
    request.num_computed_tokens -= num_rejected
    request.num_output_placeholders -= num_rejected

  new_token_ids = generated_token_ids
  new_token_ids, stopped = _update_request_with_output(request, new_token_ids)

  if structured output:
    grammar.accept_tokens(req_id, new_token_ids)

  new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
  routed_experts = routing slice
  EngineCoreOutput(new_token_ids, finish_reason, logprobs, ...)
```

```text
OutputProcessor

EngineCoreOutput.new_token_ids
  → detokenizer.update()
  → LogprobsProcessor.update_from_output()
  → RequestOutput / CompletionOutput
```

---

## 42. 最关键对象关系

```text
SchedulerOutput.scheduled_spec_decode_tokens
  本轮从 Request.spec_token_ids 消费出来、交给 target model 验证的 draft tokens。

ModelRunnerOutput.sampled_token_ids
  RejectionSampler 验证后的真实输出 tokens，已过滤 -1 padding。

ModelRunnerOutput.req_id_to_index
  Scheduler 从 req_id 找回 sampled_token_ids row 的唯一可靠映射。

Request.num_computed_tokens
  schedule 后乐观推进；update_from_output 后按 rejected draft tokens 回滚。

Request.num_output_placeholders
  async scheduling 下的输出占位；rejected draft tokens 也要从这里回滚。

Request.output_token_ids
  只 append generated_token_ids，不 append rejected draft。

EngineCoreOutput.new_token_ids
  Scheduler 确认后的本轮新增 tokens，是 OutputProcessor 的输入。

LogprobsLists.cu_num_generated_tokens
  spec decode 下对齐不同请求不同 generated token 数。

StructuredOutputGrammar
  bitmask 阶段临时推进并 rollback；update_from_output 阶段对最终 new_token_ids 永久 accept。
```

---

## 43. 最小心智模型

如果只记一条主线，可以记：

```text
scheduled_spec_decode_tokens 是“本轮猜了什么”，
sampled_token_ids 是“target 验证后真正输出什么”，
Scheduler.update_from_output() 用二者的长度差算 rejected，
先回滚计算进度，再提交真实输出。
```

再压缩成一句话：

```text
Spec decode 输出回收的核心是：过滤 rejected，统计 accepted，回滚进度，提交 confirmed tokens。
```