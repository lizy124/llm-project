# 05. ModelRunner / Worker 执行阶段如何接入？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：EngineCore 如何把 `SchedulerOutput` 交给 `model_executor`，Executor / Worker 如何接入 ModelRunner，ModelRunner 如何完成 forward / sample，并返回 `ModelRunnerOutput`。

---

## 1. 一句话回答

`EngineCore` 不直接实现模型 forward。

它只在一轮 `step()` 中调用：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:491`

然后等待 Worker / ModelRunner 返回：

```python
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:497` 到 `vllm/vllm/v1/engine/core.py:499`

主线是：

```text
EngineCore.step()
  → model_executor.execute_model(scheduler_output)
  → Executor.collective_rpc("execute_model")
  → Worker.execute_model(scheduler_output)
  → ModelRunner.execute_model(scheduler_output)
  → 模型 forward / pooling / logits
  → sample_tokens() 采样
  → ModelRunnerOutput
  → EngineCore
  → Scheduler.update_from_output()
```

一句话：

```text
EngineCore 是模型执行的发起者；
Executor / Worker 是执行调用的承载层；
ModelRunner 才是真正准备输入、调用模型 forward、采样并构造 ModelRunnerOutput 的地方。
```

---

## 2. EngineCore 如何调用 Worker 执行

普通 `EngineCore.step()` 中，Scheduler 先生成一轮执行计划：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
```

位置：`vllm/vllm/v1/engine/core.py:490`

然后 EngineCore 把这个计划交给 `model_executor`：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:491`

注意这里传的是完整 `SchedulerOutput`。

这意味着 Worker / ModelRunner 会从 `SchedulerOutput` 中读取：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
scheduled_encoder_inputs
num_common_prefix_blocks
kv_connector_metadata
ec_connector_metadata
finished_req_ids
free_encoder_mm_hashes
```

EngineCore 自己不解释这些字段的细节。

它只负责：

```text
1. 从 Scheduler 拿到 SchedulerOutput；
2. 把 SchedulerOutput 交给 model_executor；
3. 等待 ModelRunnerOutput；
4. 把 SchedulerOutput + ModelRunnerOutput 交回 Scheduler。
```

所以 EngineCore 到 Worker 的边界可以概括为：

```text
EngineCore：调度执行闭环的控制层；
model_executor：跨 Worker 的调用层；
Worker / ModelRunner：模型执行层。
```

---

## 3. Executor 执行路径

`model_executor` 是 Executor 抽象。

Executor 的职责是：

```text
在一个或多个 Worker 上执行模型相关方法。
```

抽象类注释写得很直接：

```python
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
```

位置：`vllm/vllm/v1/executor/abstract.py:37` 到 `vllm/vllm/v1/executor/abstract.py:42`

### 3.1 Executor.execute_model()

Executor 对外暴露的执行入口是：

```python
def execute_model(
    self, scheduler_output: SchedulerOutput, non_block: bool = False
) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
    output = self.collective_rpc(
        "execute_model", args=(scheduler_output,), non_block=non_block
    )
    return output[0]
```

位置：`vllm/vllm/v1/executor/abstract.py:221` 到 `vllm/vllm/v1/executor/abstract.py:227`

这里的关键点是：

```text
Executor.execute_model() 本质上是一次 collective_rpc；
它调用 Worker 上的 execute_model(scheduler_output)。
```

也就是说：

```text
EngineCore 不直接拿 Worker；
EngineCore 只调用 model_executor；
model_executor 再负责把调用分发到 Worker。
```

### 3.2 UniProcExecutor 路径

单进程 Executor 中，`execute_model()` 会调用：

```python
output = self.collective_rpc(
    "execute_model",
    args=(scheduler_output,),
    non_block=non_block,
    single_value=True,
)
```

位置：`vllm/vllm/v1/executor/uniproc_executor.py:108` 到 `vllm/vllm/v1/executor/uniproc_executor.py:116`

`collective_rpc()` 对 driver worker 直接执行方法：

```python
result = run_method(self.driver_worker, method, args, kwargs)
```

位置：`vllm/vllm/v1/executor/uniproc_executor.py:91` 到 `vllm/vllm/v1/executor/uniproc_executor.py:99`

所以 UniProc 路径可以理解为：

```text
EngineCore
  → UniProcExecutor.execute_model()
  → run_method(driver_worker, "execute_model")
  → GPUWorker.execute_model()
```

### 3.3 分布式 / 多 Worker 路径

在多进程、Ray、外部 launcher 等模式下，Executor 仍然保持同样的上层接口：

```text
model_executor.execute_model(scheduler_output)
```

不同的是 `collective_rpc()` 的实现会把请求分发到多个 Worker。

对 EngineCore 来说，这些差异被 `model_executor` 屏蔽。

因此 EngineCore 看到的是统一模型：

```text
输入：SchedulerOutput
输出：ModelRunnerOutput | None | Future[ModelRunnerOutput | None]
```

它不需要关心底层是：

```text
单 GPU；
多 GPU；
tensor parallel；
pipeline parallel；
Ray；
多进程 Worker。
```

---

## 4. Worker 如何接入 ModelRunner

以 GPU Worker 为例，Worker 的 `execute_model()` 是 EngineCore 到真正 ModelRunner 的桥。

入口是：

```python
@torch.inference_mode()
def execute_model(
    self, scheduler_output: "SchedulerOutput"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:835` 到 `vllm/vllm/v1/worker/gpu_worker.py:838`

它会先判断本轮是否有真实 forward：

```python
forward_pass = scheduler_output.total_num_scheduled_tokens > 0
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:845` 到 `vllm/vllm/v1/worker/gpu_worker.py:847`

然后调用 ModelRunner：

```python
output = self.model_runner.execute_model(
    scheduler_output, intermediate_tensors
)
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:895` 到 `vllm/vllm/v1/worker/gpu_worker.py:898`

所以 Worker 在这里主要做几类事情：

```text
处理 pipeline parallel 的中间 tensor 收发；
处理 profiler annotation；
调用 model_runner.execute_model()；
必要时调用 pooling；
把 ModelRunnerOutput / AsyncModelRunnerOutput / None 返回给 Executor。
```

在 PP 场景下，如果当前 rank 不是最后一段，ModelRunner 可能返回 `IntermediateTensors`，Worker 会把它发送给下一段：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(...)
return None
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:917` 到 `vllm/vllm/v1/worker/gpu_worker.py:924`

这说明：

```text
不是每个 Worker 都直接产出 ModelRunnerOutput；
最终输出通常由最后一个 pipeline stage 或聚合路径产生。
```

---

## 5. ModelRunner 执行路径

GPU ModelRunner 的入口是：

```python
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4047` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4051`

这里才是真正执行模型前后处理的核心路径。

### 5.1 更新 Worker 侧 batch 状态

ModelRunner 首先会根据 `SchedulerOutput` 更新持久 batch 状态：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4088` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4089`

这一步会把 Scheduler 本轮发来的计划合并到 Worker 本地状态中，例如：

```text
新请求加入 input batch；
已有请求追加 token / block 信息；
移除 finished 请求；
更新 KV block 映射；
处理 spec decode / encoder input 相关状态。
```

也就是说，Worker 不只是被动执行一个 batch，它还维护一份和 Scheduler 对齐的请求执行状态。

### 5.2 空执行处理

ModelRunner 会取出本轮 token 总数：

```python
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4083`

如果没有任何 token 要执行：

```python
if not num_scheduled_tokens:
    ...
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4099` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4115`

这对应 EngineCore 文档里的空调度情况：

```text
Scheduler 可能还有清理 / connector 工作，
但本轮没有真正 token forward。
```

这种情况下 Worker 可以返回空的 `ModelRunnerOutput`，或者只返回 KV connector 相关输出。

### 5.3 准备输入和 attention metadata

如果本轮确实有 token 要执行，ModelRunner 会准备输入。

关键步骤包括：

```python
logits_indices, spec_decode_metadata = self._prepare_inputs(
    scheduler_output,
    num_scheduled_tokens_np,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4131` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4134`

以及构造 attention metadata：

```python
attn_metadata, spec_decode_common_attn_metadata = (
    self._build_attention_metadata(...)
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4258` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4272`

再执行通用预处理：

```python
(
    input_ids,
    inputs_embeds,
    positions,
    intermediate_tensors,
    model_kwargs,
    ec_connector_output,
) = self._preprocess(
    scheduler_output, num_tokens_padded, intermediate_tensors
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4274` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4283`

可以理解为：

```text
SchedulerOutput 只是调度计划；
ModelRunner 要把计划转换成模型真正需要的张量输入、position、attention metadata、KV slot mapping 等。
```

### 5.4 执行模型 forward

真正模型 forward 发生在：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4323` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4329`

forward 外层会设置 forward context：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    ...
    slot_mapping=slot_mappings,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4305` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4316`

这说明模型 forward 需要的不只是 token ids，还包括：

```text
attention metadata；
KV slot mapping；
CUDA graph / ubatch 信息；
pipeline parallel intermediate tensors；
encoder / multimodal 预处理结果；
connector 相关上下文。
```

这些都由 ModelRunner 根据 `SchedulerOutput` 和 Worker 本地状态准备。

### 5.5 forward 后不一定立刻返回 ModelRunnerOutput

普通 generation 模型在 forward 后会得到 hidden states，再计算 logits：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4357` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4358`

然后 ModelRunner 会保存一份临时状态：

```python
self.execute_model_state = ExecuteModelState(...)
self.kv_connector_output = kv_connector_output
return None
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4389` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4408`

这解释了 EngineCore 中为什么有：

```python
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:498` 到 `vllm/vllm/v1/engine/core.py:499`

含义是：

```text
execute_model() 可能只完成 forward 和 logits 准备；
真正 sampling 需要等 EngineCore 从 Scheduler 拿到 grammar bitmask 后，再调用 sample_tokens()。
```

这对结构化输出尤其重要，因为 grammar bitmask 需要 Scheduler 根据本轮请求状态计算。

---

## 6. sample_tokens 如何返回 ModelRunnerOutput

如果 `execute_model()` 返回 `None`，EngineCore 会调用：

```python
model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:498` 到 `vllm/vllm/v1/engine/core.py:499`

Executor 同样通过 RPC 调 Worker：

```python
output = self.collective_rpc(
    "sample_tokens", args=(grammar_output,), non_block=non_block
)
return output[0]
```

位置：`vllm/vllm/v1/executor/abstract.py:241` 到 `vllm/vllm/v1/executor/abstract.py:247`

GPU Worker 直接转给 ModelRunner：

```python
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
    return self.model_runner.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:829` 到 `vllm/vllm/v1/worker/gpu_worker.py:833`

ModelRunner 的 `sample_tokens()` 会：

```text
取出 execute_model_state；
应用 structured output grammar bitmask；
执行 sampling；
更新 Worker 侧 batch 状态；
处理 speculative decoding draft tokens；
收集 logprobs / prompt_logprobs；
收集 KV / EC connector output；
构造 ModelRunnerOutput。
```

其中 grammar bitmask 应用发生在：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4455` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4459`

采样发生在：

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4461` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4462`

最后构造 `ModelRunnerOutput`：

```python
output = ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    kv_connector_output=kv_connector_output,
    ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
    num_nans_in_logits=num_nans_in_logits,
    cudagraph_stats=cudagraph_stats,
    routed_experts=None,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4612` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4626`

所以 generation 模型的常见路径其实是：

```text
execute_model()
  → forward / logits
  → return None

sample_tokens(grammar_output)
  → grammar mask
  → sampling
  → ModelRunnerOutput
```

---

## 7. ModelRunnerOutput 包含什么

`ModelRunnerOutput` 定义在：

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

位置：`vllm/vllm/v1/outputs.py:233` 到 `vllm/vllm/v1/outputs.py:281`

核心字段可以这样理解：

```text
req_ids：
  本轮有输出的请求 ID 列表。

req_id_to_index：
  req_id 到输出 batch index 的映射，Scheduler 用它找对应输出。

sampled_token_ids：
  每个请求本轮采样出的 token。

logprobs：
  生成 token 的 logprobs。

prompt_logprobs_dict：
  prompt token 的 logprobs。

pooler_output：
  pooling / embedding 类模型的输出。

kv_connector_output：
  KV transfer 相关结果，例如 finished_sending / finished_recving / invalid blocks。

ec_connector_output：
  encoder cache connector 相关结果。

num_nans_in_logits：
  logits 中 NaN 统计。

cudagraph_stats：
  CUDA graph 执行统计。

routed_experts：
  MoE routed experts 信息，如果启用相关返回。
```

注意：

```text
ModelRunnerOutput 还不是最终用户输出；
它是 Worker 返回给 Scheduler 的模型执行结果。
```

EngineCore 拿到它后，还要调用：

```python
self.scheduler.update_from_output(scheduler_output, model_output)
```

位置：`vllm/vllm/v1/engine/core.py:504` 到 `vllm/vllm/v1/engine/core.py:506`

由 Scheduler 生成 `EngineCoreOutputs`。

---

## 8. pooling 模型和 encoder-only 输出

不是所有任务都会走 generation sampling。

如果是 pooling 模型，ModelRunner 在 forward 后可能直接返回 pooling output：

```python
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4348` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4355`

这种情况下输出会体现在：

```text
ModelRunnerOutput.pooler_output
```

而不是 `sampled_token_ids`。

对于某些 encoder cache / EC transfer 场景，ModelRunner 也可能只执行 encoder 相关工作，然后返回空 encoder 输出：

```python
self._execute_mm_encoder(scheduler_output)
return make_empty_encoder_model_runner_output(scheduler_output)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4091` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4097`

所以 Worker 执行阶段不只服务于文本生成，还覆盖：

```text
generation；
pooling / embedding；
多模态 encoder；
KV / EC connector；
pipeline parallel 中间 tensor 传递。
```

---

## 9. 执行异常如何处理

异常可能发生在 Executor、Worker、ModelRunner 任一层。

### 9.1 UniProcExecutor 会把异常放进 Future

单进程非阻塞调用中，`collective_rpc()` 会捕获异常并写入 Future：

```python
try:
    result = run_method(self.driver_worker, method, args, kwargs)
    ...
except Exception as e:
    future = Future[Any]()
    future.set_exception(e)
return future
```

位置：`vllm/vllm/v1/executor/uniproc_executor.py:97` 到 `vllm/vllm/v1/executor/uniproc_executor.py:106`

并且 `execute_model()` 会在非阻塞模式下尽早暴露已完成 Future 的异常：

```python
if non_block and output.done():
    output.result()
```

位置：`vllm/vllm/v1/executor/uniproc_executor.py:117` 到 `vllm/vllm/v1/executor/uniproc_executor.py:120`

### 9.2 EngineCore 在 future.result() 处感知执行失败

EngineCore 等待 Worker 输出时调用：

```python
model_output = future.result()
```

位置：`vllm/vllm/v1/engine/core.py:497`

如果 Worker / ModelRunner 执行失败，异常会在这里抛出。

外层还有错误日志上下文：

```python
with (
    self.log_error_detail(scheduler_output),
    self.log_iteration_details(scheduler_output),
):
    model_output = future.result()
```

位置：`vllm/vllm/v1/engine/core.py:493` 到 `vllm/vllm/v1/engine/core.py:497`

它能把当前 `SchedulerOutput` 相关信息带进错误日志，便于定位是哪一轮调度导致异常。

### 9.3 sample_tokens 状态错误

ModelRunner 内部也会检查执行顺序。

例如 GPU ModelRunner 的 `execute_model()` 开头有：

```python
if self.execute_model_state is not None:
    raise RuntimeError(
        "State error: sample_tokens() must be called "
        "after execute_model() returns None."
    )
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4052` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4056`

这说明 `execute_model()` 和 `sample_tokens()` 有严格配对关系：

```text
如果 execute_model() 返回 None，后续必须调用 sample_tokens() 消费临时状态；
否则下一次 execute_model() 会发现状态未清理并报错。
```

---

## 10. 总结

Worker 执行阶段可以按三层理解：

```text
EngineCore
  负责控制一轮执行闭环：schedule → execute → update。

Executor
  负责把 execute_model / sample_tokens 调用分发到一个或多个 Worker。

Worker / ModelRunner
  负责根据 SchedulerOutput 准备输入、执行 forward、采样 / pooling、构造 ModelRunnerOutput。
```

完整链路是：

```text
EngineCore.step()
  → scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → Executor.collective_rpc("execute_model")
  → Worker.execute_model(scheduler_output)
  → ModelRunner.execute_model(scheduler_output)
  → _update_states()
  → _prepare_inputs() / _build_attention_metadata() / _preprocess()
  → _model_forward()
  → compute_logits / pooling / intermediate tensors
  → sample_tokens(grammar_output)  # generation 常见路径
  → ModelRunnerOutput
  → EngineCore
  → scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
```

如果要回答：

```text
ModelRunner / Worker 执行阶段如何接入？
```

可以概括为：

```text
EngineCore 把 Scheduler.schedule() 生成的 SchedulerOutput 传给 model_executor.execute_model()；
Executor 通过 collective_rpc 调用 Worker.execute_model()；
Worker 再调用 ModelRunner.execute_model()，由 ModelRunner 根据 SchedulerOutput 准备 batch、KV slot、attention metadata 并执行模型 forward。
对于 generation，forward 后通常返回 None，EngineCore 再调用 sample_tokens(grammar_output) 完成采样并拿到 ModelRunnerOutput。
最后 EngineCore 把 SchedulerOutput 和 ModelRunnerOutput 一起交给 Scheduler.update_from_output()，生成 EngineCoreOutputs。
```
