# 07. 模型 forward 和 logits 在哪里发生？

源码位置：

- `code/vllm/vllm\v1\worker\gpu_model_runner.py`
- `code/vllm/vllm\v1\worker\gpu\model_runner.py`
- `code/vllm/vllm\v1\worker\gpu_worker.py`
- `code/vllm/vllm\forward_context.py`

本问题关注：`SchedulerOutput` 已经同步到 `InputBatch`，输入和 attention metadata 也准备好之后，模型真正在哪里 forward；forward 外层的 `set_forward_context()` 提供了什么；generation 模型的 logits 在哪里产生；pooling / embedding 类模型如何直接产生 pooling 输出；Pipeline Parallel 下为什么有些 rank 返回 `IntermediateTensors` 而不是 logits。

---

## 1. 一句话回答

模型真正 forward 发生在 `GPUModelRunner._model_forward()` 中，最终调用的是：

```python
self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`gpu_model_runner.py:3757` 到 `gpu_model_runner.py:3787`

forward 之后：

```text
非最后 PP rank：
  返回 IntermediateTensors，发给下一个 pipeline stage。

最后 PP rank + pooling model：
  调用 pooler，生成 pooling output。

最后 PP rank + generation model：
  取 logits_indices 对应 hidden states，调用 model.compute_logits()，得到 logits。
```

主链路可以记为：

```text
_prepare_inputs()
  → _build_attention_metadata()
  → set_forward_context(...)
  → _model_forward()
  → hidden_states / IntermediateTensors
  → compute_logits() 或 _pool()
  → sample_tokens() / ModelRunnerOutput
```

---

## 2. forward 在 execute_model 的哪一段

`GPUModelRunner.execute_model()` 的主体链路中，forward 发生在输入准备之后、后处理之前。

大致结构是：

```text
_update_states(scheduler_output)
  → _prepare_inputs(...)
  → _determine_batch_execution_and_padding(...)
  → _get_slot_mappings(...)
  → _build_attention_metadata(...)
  → _preprocess(...)
  → set_forward_context(...)
  → _model_forward(...)
  → postprocess hidden_states / logits / pooling
```

真正调用位置在：`gpu_model_runner.py:4297` 到 `gpu_model_runner.py:4326`

核心代码：

```python
with (
    set_forward_context(...),
    record_function_or_nullcontext("gpu_model_runner: forward"),
    self.maybe_get_kv_connector_output(...),
):
    model_output = self._model_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )
```

位置：`gpu_model_runner.py:4302` 到 `gpu_model_runner.py:4326`

这说明：

```text
forward 不是裸调用模型；
它发生在 forward context、profiling range、KV connector context 包裹之内。
```

---

## 3. _model_forward() 的职责

定义：

```python
def _model_forward(
    self,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs: dict[str, Any],
) -> Any:
```

位置：`gpu_model_runner.py:3757`

实现非常直接：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`gpu_model_runner.py:3781` 到 `gpu_model_runner.py:3787`

它的定位是：

```text
只负责调用模型 forward；
不负责 batch 状态更新；
不负责 attention metadata 构造；
不负责采样；
不负责 Scheduler 状态回收。
```

源码注释也说明：

```text
这个方法可以被子类覆盖，方便只检查模型执行部分，而不是整个 execute_model。
```

位置：`gpu_model_runner.py:3765` 到 `gpu_model_runner.py:3770`

---

## 4. forward 的输入来自哪里

`_model_forward()` 的输入不是直接来自 Scheduler，而是由前面的准备阶段构造出来。

### 4.1 input_ids / positions / inputs_embeds

来自 `_preprocess()`：

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

位置：`gpu_model_runner.py:4271` 到 `gpu_model_runner.py:4280`

其中：

```text
input_ids：
  本轮要执行的 token ids。

positions：
  每个 token 对应的位置。

inputs_embeds：
  多模态或 prompt_embeds 场景下可能替代 input_ids。

intermediate_tensors：
  Pipeline Parallel 非首 rank 接收到的上一 stage 输出。

model_kwargs：
  模型特定参数、多模态参数、encoder-decoder 参数等。
```

### 4.2 attention metadata

虽然 attention metadata 不作为 `_model_forward()` 的显式参数传入，但会通过 `set_forward_context()` 注入当前 forward 上下文。

也就是说 attention backend 在模型内部执行 attention 时，可以从 forward context 里拿到当前 batch 的 metadata。

---

## 5. set_forward_context() 做什么

forward 外层包着：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    ubatch_slices=ubatch_slices_padded,
    slot_mapping=slot_mappings,
    skip_compiled=has_encoder_input,
)
```

位置：`gpu_model_runner.py:4303` 到 `gpu_model_runner.py:4313`

这里传进去的关键信息包括：

```text
attn_metadata：attention backend 使用的元数据；
num_tokens：本轮 padded 后 token 数；
num_tokens_across_dp：DP 场景下各 rank token 信息；
cudagraph_runtime_mode：CUDA graph / eager / piecewise 模式；
batch_descriptor：batch 执行描述；
ubatch_slices：microbatch 切分；
slot_mapping：token 写入 KV cache 的位置；
skip_compiled：是否跳过编译路径。
```

这说明：

```text
模型 forward 依赖的不只是 input_ids，
还依赖当前 forward context 中的 attention / KV cache / compile / batch 运行时信息。
```

---

## 6. forward 和 attention backend 的关系

`_build_attention_metadata()` 先构造 attention backend 需要的 metadata：

```python
attn_metadata, spec_decode_common_attn_metadata = (
    self._build_attention_metadata(...)
)
```

位置：`gpu_model_runner.py:4255` 到 `gpu_model_runner.py:4268`

然后 `set_forward_context()` 把 `attn_metadata` 放入当前 forward 上下文。

因此关系是：

```text
SchedulerOutput / InputBatch / block table
  → slot_mappings
  → attention metadata
  → set_forward_context
  → model forward 内部 attention layer 使用
```

forward 本身不会重新决定：

```text
哪些 token 属于哪个请求；
KV cache 写到哪个 slot；
block table 是什么；
prefix / cascade attention 怎么处理。
```

这些都已经由前面的输入准备和 attention metadata 阶段决定。

---

## 7. forward 返回什么

`model_output` 的形态取决于模型类型和 PP rank。

### 7.1 最后 PP rank：hidden_states

普通 generation 模型在最后 PP rank 通常返回：

```text
hidden_states: torch.Tensor
```

如果启用辅助 hidden state 输出，例如 EAGLE 3，则返回：

```text
(hidden_states, aux_hidden_states)
```

处理逻辑：

```python
if self.use_aux_hidden_state_outputs:
    hidden_states, aux_hidden_states = model_output
else:
    hidden_states = model_output
    aux_hidden_states = None
```

位置：`gpu_model_runner.py:4328` 到 `gpu_model_runner.py:4335`

### 7.2 非最后 PP rank：IntermediateTensors

如果当前 rank 不是最后一个 pipeline stage：

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`gpu_model_runner.py:4337` 到 `gpu_model_runner.py:4343`

这说明非最后 PP rank 不会计算 logits，而是把中间 hidden states 交给下一个 PP stage。

### 7.3 pooling model：pooling output

如果是 pooling 模型：

```python
if self.is_pooling_model:
    return self._pool(...)
```

位置：`gpu_model_runner.py:4345` 到 `gpu_model_runner.py:4352`

pooling 模型不会走 token sampling 的主路径。

---

## 8. logits 在哪里产生

generation 模型的 logits 产生于：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`gpu_model_runner.py:4354` 到 `gpu_model_runner.py:4355`

关键点：

```text
不是所有 hidden_states 都拿去算 logits；
只对 logits_indices 指定的位置计算 logits。
```

`logits_indices` 由前面的 `_prepare_inputs()` 产生：

```python
logits_indices, spec_decode_metadata = self._prepare_inputs(...)
```

位置：`gpu_model_runner.py:4128` 到 `gpu_model_runner.py:4131`

这表示：

```text
prefill / chunked prefill / decode / spec decode 中，
哪些位置需要 logits，是输入准备阶段决定的。
```

---

## 9. 为什么只对 logits_indices 算 logits

LLM forward 会产生一批 hidden states，但不是每个 token 位置都需要立刻计算 logits。

常见情况：

```text
prefill：
  可能只需要最后一个位置的 logits 用来采样下一个 token；
  如果请求 prompt_logprobs，则还要额外处理 prompt 位置 logits。

chunked prefill：
  非最后 prefill chunk 可能不需要采样 logits。

decode：
  通常每个请求只需要当前 decode 位置的 logits。

spec decode：
  target model 可能需要验证多个 draft token 位置。
```

因此：

```text
hidden_states[logits_indices]
  → 只截取真正需要算 logits 的 hidden states
  → model.compute_logits()
```

这可以减少不必要的 logits 计算和显存/带宽开销。

---

## 10. logits 之后为什么不立刻返回 ModelRunnerOutput

在 generation 路径中，计算完 logits 后，并不直接返回 `ModelRunnerOutput`。

而是保存临时状态：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
)
self.kv_connector_output = kv_connector_output
return None
```

位置：`gpu_model_runner.py:4386` 到 `gpu_model_runner.py:4405`

原因是：

```text
logits 只是采样前的中间结果；
还需要 grammar bitmask、sampler、logprobs、spec decode、bookkeeping；
这些在 sample_tokens() 里完成。
```

所以 generation 路径是：

```text
execute_model()
  → forward
  → compute_logits
  → 保存 ExecuteModelState
  → return None

sample_tokens(grammar_output)
  → apply_grammar_bitmask
  → _sample(logits)
  → bookkeeping
  → ModelRunnerOutput
```

---

## 11. pooling 输出在哪里产生

pooling 路径走 `_pool()`。

入口：

```python
def _pool(
    self,
    hidden_states: torch.Tensor,
    num_scheduled_tokens: int,
    num_scheduled_tokens_np: np.ndarray,
    kv_connector_output: KVConnectorOutput | None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
```

位置：`gpu_model_runner.py:3342`

### 11.1 构造 pooling metadata

```python
pooling_metadata = self.input_batch.get_pooling_metadata()
pooling_metadata.build_pooling_cursor(...)
```

位置：`gpu_model_runner.py:3357` 到 `gpu_model_runner.py:3363`

这一步告诉 pooler：

```text
每个请求有哪些 token；
本轮处理了多少 token；
sequence length / prompt length 是多少；
哪些请求已经可以产出 pooling 结果。
```

### 11.2 调用模型 pooler

```python
model = cast(VllmModelForPooling, self.model)
raw_pooler_output = model.pooler(
    hidden_states=hidden_states,
    pooling_metadata=pooling_metadata,
)
```

位置：`gpu_model_runner.py:3365` 到 `gpu_model_runner.py:3368`

这就是 pooling 输出产生的位置。

### 11.3 构造 ModelRunnerOutput

```python
model_runner_output = ModelRunnerOutput(
    req_ids=self.input_batch.req_ids.copy(),
    req_id_to_index=self.input_batch.req_id_to_index.copy(),
    kv_connector_output=kv_connector_output,
)
```

位置：`gpu_model_runner.py:3381` 到 `gpu_model_runner.py:3385`

如果有完成的 pooling output，则填入 `pooler_output`；CUDA 场景下可能包装成 `AsyncGPUPoolingModelRunnerOutput`。

位置：`gpu_model_runner.py:3387` 到 `gpu_model_runner.py:3405`

---

## 12. Pipeline Parallel 下 forward / logits 的分工

Pipeline Parallel 会把模型层切到多个 rank。

### 12.1 非 first PP rank 的输入

在 Worker 层，如果当前 rank 不是 first PP rank，会先接收上一 stage 的 intermediate tensors：

```python
tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(...)
intermediate_tensors = AsyncIntermediateTensors(...)
```

位置：`gpu_worker.py:853` 到 `gpu_worker.py:865`

这说明：

```text
first PP rank 输入 token ids / embeddings；
后续 PP rank 输入 IntermediateTensors。
```

### 12.2 非最后 PP rank 的输出

在 `GPUModelRunner.execute_model()` 中，非最后 PP rank 返回：

```text
IntermediateTensors
```

位置：`gpu_model_runner.py:4337` 到 `gpu_model_runner.py:4343`

然后 `GPUWorker.execute_model()` 会把它异步发送到下一 PP stage：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
return None
```

位置：`gpu_worker.py:889` 到 `gpu_worker.py:896`

### 12.3 最后 PP rank 才算 logits / pooling

只有最后 PP rank 拿到最终 hidden states 后，才会：

```text
compute_logits()
或
_pool()
```

因此 PP 场景可以记为：

```text
PP rank 0...N-2：
  forward partial layers → IntermediateTensors

PP rank N-1：
  forward final layers → hidden_states → logits / pooling
```

---

## 13. broadcast_pp_output 特殊路径

代码里还有一个较少见的分支：

```python
if self.broadcast_pp_output:
```

位置：`gpu_model_runner.py:4356`

它主要用于 `external_launcher` + PP 场景。

普通路径下，非最后 PP rank 不会拿 logits；但 `broadcast_pp_output` 路径会让最后 PP rank 计算 logits 后广播给其他 rank。

核心逻辑：

```python
if not get_pp_group().is_last_rank:
    send_tensor_dict(...)
    logits = None
else:
    logits = self.model.compute_logits(sample_hidden_states)

broadcasted = get_pp_group().broadcast_tensor_dict(...)
logits = broadcasted["logits"]
```

位置：`gpu_model_runner.py:4356` 到 `gpu_model_runner.py:4384`

这不是普通 serving 主路径，但说明 vLLM 为外部 launcher / torchrun 类场景保留了同步 logits 的机制。

---

## 14. V2 ModelRunner 对照

V2 路径在：`vllm/v1/worker/gpu/model_runner.py`

它的总体思路一致，但组织方式不同。

### 14.1 execute_model 中直接准备 batch / attention / model_inputs

V2 `execute_model()` 会：

```text
finish_requests / free_states / add_requests / update_requests
prepare_inputs
prepare_attn
构造 model_inputs
set_forward_context
调用 model 或 cudagraph_manager
保存 ExecuteModelState
```

位置：`gpu/model_runner.py:1101` 到 `gpu/model_runner.py:1323`

### 14.2 eager / piecewise / full cudagraph

V2 明确区分：

```text
FULL：cudagraph_manager.run_fullgraph(batch_desc)
PIECEWISE：cudagraph_manager.run_pw_graph(self.model, model_inputs)
NONE：self.model(**model_inputs)
```

位置：`gpu/model_runner.py:1256` 到 `gpu/model_runner.py:1294`

### 14.3 logits 在 sample 阶段计算

V2 的 sampling 里会：

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`gpu/model_runner.py:1042` 到 `gpu/model_runner.py:1046`

和 V1 的区别是组织位置不同，但核心仍然是：

```text
hidden_states + logits_indices → compute_logits()
```

---

## 15. forward 和 CUDA graph / compile 的关系

在 V1 里，执行前会调用：

```python
_determine_batch_execution_and_padding(...)
```

位置：`gpu_model_runner.py:4143` 到 `gpu_model_runner.py:4156`

它会决定：

```text
cudagraph_mode
batch_desc
should_ubatch
num_tokens_across_dp
cudagraph_stats
```

后续 `set_forward_context()` 把这些信息传入 forward context。

因此：

```text
同样是 model forward，实际可能走 eager、compiled、CUDA graph replay、piecewise graph 等不同执行形态。
```

但从主链路视角看，都可以归纳为：

```text
准备 batch descriptor
  → 设置 forward context
  → 调模型 forward
```

---

## 16. forward 和 KV connector 的关系

forward 外层还有：

```python
self.maybe_get_kv_connector_output(
    scheduler_output,
    defer_finalize=defer_kv_connector_finalize,
) as kv_connector_output
```

位置：`gpu_model_runner.py:4315` 到 `gpu_model_runner.py:4318`

这说明：

```text
KV connector 可以在 forward 前后参与 KV save / load / metadata 处理；
forward 产生的 KV cache 状态可能需要被 connector 输出带回 Scheduler。
```

如果启用 speculative decoding，还会：

```python
defer_kv_connector_finalize = self.speculative_config is not None
```

位置：`gpu_model_runner.py:4297` 到 `gpu_model_runner.py:4301`

原因是 draft model 也可能需要保存 KV，因此 connector finalize 要延后到 draft model 之后。

---

## 17. forward 后的状态保存：ExecuteModelState

generation 路径最终会保存：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
)
```

位置：`gpu_model_runner.py:4386` 到 `gpu_model_runner.py:4397`

这说明 execute_model 和 sample_tokens 之间通过 `ExecuteModelState` 传递：

```text
SchedulerOutput
logits
spec decode metadata
hidden states
aux hidden states
EC connector output
CUDA graph stats
slot mappings
```

也就是说：

```text
execute_model() 负责 forward / logits；
sample_tokens() 负责 grammar / sampling / bookkeeping / output。
```

---

## 18. 容易疑惑的点

### 18.1 forward 是在 Executor 里发生的吗？

不是。

Executor 只负责分发 `execute_model()`；真正模型 forward 在 `GPUModelRunner._model_forward()` 中。

### 18.2 logits 是模型 forward 直接返回的吗？

通常不是。

模型 forward 返回的是 hidden states；logits 通过：

```python
self.model.compute_logits(sample_hidden_states)
```

单独计算。

### 18.3 每个 token 的 hidden state 都会算 logits 吗？

不是。

只对 `logits_indices` 指定位置算 logits。

### 18.4 pooling 模型会走 sample_tokens 吗？

一般不会。

pooling 模型 forward 后调用 `_pool()` 或 V2 的 `pool()`，直接生成 pooling output。

### 18.5 非最后 PP rank 为什么没有 logits？

因为它只执行模型的一部分层，输出是 `IntermediateTensors`，最终 logits 只能在最后 PP rank 拿到完整 hidden states 后计算。

### 18.6 set_forward_context 是不是 attention 的输入参数？

它不是传给 `self.model()` 的显式参数，但会设置当前 forward 上下文，attention backend 和编译/graph 运行时会依赖里面的 metadata。

---

## 19. 总结

模型 forward 和 logits 的完整关系是：

```text
InputBatch / SchedulerOutput
  → _prepare_inputs()
  → _build_attention_metadata()
  → set_forward_context(attn_metadata, slot_mapping, batch_desc, ...)
  → _model_forward()
  → hidden_states / IntermediateTensors
  → compute_logits(hidden_states[logits_indices])
  → ExecuteModelState
  → sample_tokens()
  → ModelRunnerOutput
```

如果是 pooling 模型：

```text
hidden_states
  → model.pooler(hidden_states, pooling_metadata)
  → pooler_output
  → ModelRunnerOutput
```

如果是 Pipeline Parallel：

```text
非最后 PP rank：
  model forward → IntermediateTensors → send to next stage

最后 PP rank：
  model forward → hidden_states → logits / pooling
```

一句话压缩：

```text
forward 产出 hidden states，logits 是最后 PP rank 基于 logits_indices 对 hidden states 单独计算出来的；pooling 模型则绕过 logits / sampling，直接用 pooler 生成输出。
```
