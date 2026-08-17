# 08. Sampler / Output 和 CUDA graph 的边界在哪里？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\cudagraph_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\outputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\output_processor.py`

本问题关注：CUDA graph capture 通常覆盖哪些 forward 计算；`compute_logits()`、`Sampler`、`logprobs`、`prompt_logprobs`、pooling output、异步 D2H copy、`OutputProcessor` 和 `Scheduler.update_from_output()` 是否在 graph 内；以及为什么这些输出侧逻辑大多需要留在 graph 边界之外。

---

## 1. 一句话回答

CUDA graph 的核心边界在 `GPUModelRunner._model_forward()` / V2 `model(**model_inputs)` 这一段：

```text
graph 内：
  model backbone forward
  attention / KV cache update / compiled piecewise 子图
  → 产出 hidden_states 或 IntermediateTensors

通常 graph 外：
  hidden_states[logits_indices]
  compute_logits()
  grammar bitmask
  Sampler / rejection sampler
  logprobs gather / prompt_logprobs
  bookkeeping / spec draft proposal
  ModelRunnerOutput / AsyncGPUModelRunnerOutput
  Scheduler.update_from_output()
  OutputProcessor / detokenize / RequestOutput
```

更短地说：

```text
CUDA graph 优化的是 GPU forward 路径；采样和输出回收是动态状态机逻辑，通常在 graph 边界之外完成。
```

---

## 2. 最小主链路

从 EngineCore 一轮 step 看，执行顺序是：

```text
EngineCore.step()
  → scheduler.schedule()
  → model_executor.execute_model(scheduler_output, non_block=True)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → future.result()
  → 如果 model_output is None：model_executor.sample_tokens(grammar_output)
  → scheduler.update_from_output(scheduler_output, model_output)
```

位置：`code/vllm/vllm/v1/engine/core.py:479` 到 `code/vllm/vllm/v1/engine/core.py:508`

其中生成模型的 ModelRunner 内部可以拆成：

```text
execute_model()
  → _update_states()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(... cudagraph_runtime_mode ...)
  → _model_forward()          # CUDA graph / compile wrapper 的主要覆盖区
  → compute_logits()          # 在 forward 之后
  → 保存 ExecuteModelState
  → return None

sample_tokens(grammar_output)
  → apply_grammar_bitmask()
  → _sample()
      → Sampler / RejectionSampler
  → _update_states_after_model_execute()
  → spec decode draft proposal
  → _bookkeeping_sync()
  → ModelRunnerOutput
  → AsyncGPUModelRunnerOutput（可选）
```

位置：

- `execute_model()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`
- forward context：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4302` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`
- forward 后 compute logits：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`
- 暂存 `ExecuteModelState`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4397`
- `sample_tokens()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4682`

---

## 3. CUDA graph 的“边界线”在哪里

最重要的边界线是这段 `with set_forward_context(...): self._model_forward(...)`。

```python
with (
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
    ),
    record_function_or_nullcontext("gpu_model_runner: forward"),
    self.maybe_get_kv_connector_output(...),
):
    model_output = self._model_forward(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4302` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

`_model_forward()` 本身只是：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3787`

因此 CUDA graph 的控制面来自 `ForwardContext`：

```text
cudagraph_runtime_mode
batch_descriptor
attn_metadata
slot_mapping
ubatch_slices
skip_compiled
```

真正 capture / replay 由模型外层 wrapper、compiled 子图或 V2 `CudaGraphManager` 完成，不是 sampler 或 output 层完成。

---

## 4. FULL / PIECEWISE 下 graph 覆盖范围的差异

### 4.1 FULL graph

FULL CUDA graph 目标是覆盖整个 model forward：

```text
_model_forward()
  → self.model(...)
      → embedding / layers / attention / KV update / MLP / norm
  → hidden_states
```

它通常包含 attention，因此需要 padded attention metadata、固定 `BatchDescriptor`、稳定 input buffer 地址。

但即使是 FULL：

```text
compute_logits()
Sampler
ModelRunnerOutput
OutputProcessor
Scheduler.update_from_output()
```

也不属于这段 `_model_forward()`，因此不在 FULL graph 主边界内。

### 4.2 PIECEWISE graph

PIECEWISE CUDA graph 通常覆盖 compiled 后的部分子图：

```text
model forward
  → attention / splitting op 可能在 graph 外
  → FFN / norm / linear 等 compiled piecewise graph 可能 replay
  → hidden_states
```

它更兼容动态 attention metadata，但覆盖范围更窄。

### 4.3 V2 ModelRunner 对照

V2 `vllm/v1/worker/gpu/model_runner.py` 把 graph 分支写得更显式：

```text
if batch_desc.cg_mode == FULL:
  cudagraph_manager.run_fullgraph(batch_desc)
elif batch_desc.cg_mode == PIECEWISE:
  cudagraph_manager.run_pw_graph(self.model, model_inputs)
else:
  self.model(**model_inputs)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1256` 到 `code/vllm/vllm/v1/worker/gpu/model_runner.py:1294`

V2 的 `sample()` 仍然在 forward 之后单独执行：

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
...
sampler_output = self.sampler(logits, input_batch)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1038` 到 `code/vllm/vllm/v1/worker/gpu/model_runner.py:1070`

这说明新旧组织方式不同，但边界一致：

```text
CUDA graph / compile 主要围绕 model forward；sample/logits/output 是 forward 后阶段。
```

---

## 5. compute_logits 是否在 CUDA graph 内

在 V1 GPUModelRunner 常见路径中，`compute_logits()` 不在 `_model_forward()` 的 graph 边界内。

forward 结束后才执行：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

这意味着：

```text
CUDA graph replay 返回 hidden_states；
然后 Python 继续做 hidden_states 索引和 lm head / logits 投影。
```

为什么 logits 放在 forward 外面？

```text
1. 不是所有 hidden states 都需要 logits；只对 logits_indices 指定位置算。
2. prefill / chunked prefill / decode / spec decode 的 logits 需求不同。
3. 用户是否请求 prompt_logprobs / sampled logprobs 会影响后续处理。
4. lm head 输出 vocab 维很大，和 sampler/logprobs 绑定更紧。
5. forward 编译图通常以 hidden_states 为稳定输出边界。
```

`logits_indices` 在输入准备阶段产生：

```text
_prepare_inputs(...)
  → logits_indices
  → 后续 hidden_states[logits_indices]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4128` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4131`

所以可以记成：

```text
CUDA graph：tokens → hidden_states
CUDA graph 外：hidden_states[logits_indices] → logits
```

---

## 6. Sampler 是否在 CUDA graph 内

一般不在。

V1 的 sampler 调用发生在 `sample_tokens()` 里，而 `sample_tokens()` 是 `execute_model()` 返回 `None` 之后由 EngineCore 继续调用的阶段。

```python
with record_function_or_nullcontext("gpu_model_runner: sample"):
    sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4458` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4459`

`_sample()` 会根据是否有 speculative decoding 选择普通 sampler 或 rejection sampler：

```text
无 spec_decode_metadata：
  self.sampler(logits, sampling_metadata)

有 spec_decode_metadata：
  self.rejection_sampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3570` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3599`

普通 `Sampler` 定义在：`code/vllm/vllm/v1/sample/sampler.py:20`

它的流程包括：

```text
1. 根据 logprobs_mode 计算 raw_logprobs 或保留 raw_logits
2. logits 转 float32
3. allowed token ids mask
4. bad words mask
5. 非 argmax-invariant logits processors
6. repetition / frequency / presence penalties
7. greedy 或 random sampling
8. top-k / top-p / min-p
9. gather logprobs / ranks
10. 返回 SamplerOutput
```

位置：`code/vllm/vllm/v1/sample/sampler.py:20` 到 `code/vllm/vllm/v1/sample/sampler.py:58`

这些步骤虽然很多在 GPU tensor 上执行，但它们依赖动态 `SamplingMetadata`、每请求采样参数、随机数生成器、logprobs 需求、bad words、penalties、structured output 等，因此不是 model forward CUDA graph 的一部分。

---

## 7. grammar bitmask 为什么在 graph 外

结构化输出的 grammar bitmask 在 `sample_tokens()` 开头应用：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

而 `grammar_output` 是 EngineCore 在 model forward 进行时由 Scheduler 准备的：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`code/vllm/vllm/v1/engine/core.py:490` 到 `code/vllm/vllm/v1/engine/core.py:500`

这说明 grammar 的设计就是：

```text
forward / logits 先产生候选分布；
Scheduler 同步准备动态 grammar mask；
sample_tokens() 在采样前修改 logits；
然后 sampler 只从合法 token 空间采样。
```

它不适合进入 model forward graph，原因是：

```text
- 每个 request 的 grammar 状态动态变化；
- grammar mask 的 token 集合每步不同；
- Scheduler 需要根据 request 状态生成 bitmask；
- 它作用在 logits 上，而 logits 本身已经是 forward 后产物。
```

---

## 8. logprobs 是否在 CUDA graph 内

通常不在 model forward CUDA graph 内。

logprobs 可以分成两类：

```text
sampled token logprobs：
  采样阶段围绕 logits / sampled token 计算。

prompt logprobs：
  bookkeeping 阶段基于 hidden_states / prompt 位置再组织。
```

### 8.1 sampled token logprobs

`Sampler.forward()` 中，如果请求了 logprobs：

```python
raw_logprobs = self.compute_logprobs(logits)
```

位置：`code/vllm/vllm/v1/sample/sampler.py:84` 到 `code/vllm/vllm/v1/sample/sampler.py:94`

`compute_logprobs()` 是：

```python
return logits.log_softmax(dim=-1, dtype=torch.float32)
```

位置：`code/vllm/vllm/v1/sample/sampler.py:304` 到 `code/vllm/vllm/v1/sample/sampler.py:306`

随后根据需求 gather：

```text
num_logprobs：
  gather top-k logprobs + sampled token rank

logprob_token_ids：
  gather 指定 token ids 的 logprobs
```

位置：

- `gather_specific_token_logprobs()`：`code/vllm/vllm/v1/sample/sampler.py:151` 到 `code/vllm/vllm/v1/sample/sampler.py:225`
- `gather_logprobs()`：`code/vllm/vllm/v1/sample/sampler.py:309` 到 `code/vllm/vllm/v1/sample/sampler.py:356`

这些都属于 sampler 逻辑，不属于 `_model_forward()` graph。

### 8.2 prompt logprobs

V1 GPUModelRunner 在 bookkeeping 阶段计算 prompt logprobs：

```python
prompt_logprobs_dict = self._get_prompt_logprobs_dict(
    hidden_states[:num_scheduled_tokens],
    scheduler_output.num_scheduled_tokens,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3726` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3730`

bookkeeping 本身发生在 `sample_tokens()`：

```text
_bookkeeping_sync(...)
  → num_nans_in_logits
  → logprobs_lists
  → valid_sampled_token_ids
  → prompt_logprobs_dict
  → req_ids_output_copy
  → invalid_req_indices
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4574` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4589`

因此：

```text
prompt logprobs 依赖 output 组织和请求状态；
不属于 cudagraph replay 的固定 forward 子图。
```

---

## 9. RejectionSampler / spec decode 和 graph 的关系

Speculative decoding 更说明 sampler/output 不适合放进 forward graph。

在 `sample_tokens()` 中：

```text
1. 先用 target logits 采样或 rejection sample；
2. 更新状态；
3. 根据 sampled tokens / hidden states / metadata 生成下一轮 draft tokens；
4. draft tokens 可能用 GPU tokens，也可能等 CPU bookkeeping 后再生成；
5. 如果不适合 drafter，要清零 draft tokens，避免调度旧 draft。
```

关键位置：

- `_sample()` spec 分支：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3586` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3599`
- 清理旧 draft 状态：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4474` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4479`
- `propose_draft_token_ids()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4481` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4496`
- 各 spec decode 分支：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4497` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4573`

这部分不只是 tensor kernel：

```text
它会读取 request 状态；
判断 drafter max length；
决定是否用 EAGLE / draft model / ngram GPU；
决定 draft proposal 在 bookkeeping 前还是后执行；
更新 Worker 侧缓存；
复制 draft tokens 到 CPU；
影响下一轮 Scheduler 调度。
```

这些动态控制流不适合塞进固定 CUDA graph。

---

## 10. ModelRunnerOutput 是否在 CUDA graph 内

不在。

`ModelRunnerOutput` 是 Python dataclass，定义在：

`code/vllm/vllm/v1/outputs.py:233`

核心字段包括：

```text
req_ids
req_id_to_index
sampled_token_ids
logprobs
prompt_logprobs_dict
pooler_output
kv_connector_output
ec_connector_output
num_nans_in_logits
cudagraph_stats
routed_experts
```

位置：`code/vllm/vllm/v1/outputs.py:233` 到 `code/vllm/vllm/v1/outputs.py:281`

生成模型在 `sample_tokens()` 末尾构造：

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

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4609` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4623`

这一步明显是 graph 外 Python 对象构造。

`cudagraph_stats` 只是把本轮 graph dispatch 统计带出去：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

统计对象在 `_determine_batch_execution_and_padding()` 中创建：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3914`

所以：

```text
ModelRunnerOutput 可以携带 cudagraph_stats，
但 ModelRunnerOutput 本身不在 cudagraph 内。
```

---

## 11. AsyncGPUModelRunnerOutput 和 CUDA graph 的关系

`AsyncGPUModelRunnerOutput` 也不在 CUDA graph 内，它是输出侧 D2H copy 优化。

定义位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:238`

它做的事是：

```text
1. 保存 GPU sampled_token_ids / logprobs_tensors / routed_experts 引用；
2. 在 async_output_copy_stream 上等待默认 stream；
3. 发起 sampled_token_ids.to("cpu", non_blocking=True)；
4. 发起 logprobs_tensors.to_cpu_nonblocking()；
5. 发起 routed_experts.to_cpu_nonblocking()；
6. record event；
7. get_output() 时 synchronize event；
8. 转成 list / numpy 并回填 ModelRunnerOutput。
```

位置：

- 构造函数：`code/vllm/vllm/v1/worker/gpu_model_runner.py:239` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:280`
- `get_output()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:282` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:316`

`sample_tokens()` 中只有在 async scheduling 时才包装：

```python
async_output = AsyncGPUModelRunnerOutput(
    model_runner_output=output,
    sampled_token_ids=sampler_output.sampled_token_ids,
    logprobs_tensors=sampler_output.logprobs_tensors,
    invalid_req_indices=invalid_req_indices,
    async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
    vocab_size=self.input_batch.vocab_size,
    routed_experts=routed_experts_snapshot,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4663` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4671`

它和 CUDA graph 的关系可以概括为：

```text
CUDA graph replay 优化 forward kernel launch；
AsyncGPUModelRunnerOutput 优化输出 tensor 的 D2H copy 等待；
两者都减少同步/调度开销，但发生在不同边界。
```

### 11.1 为什么 async output copy 不能简单放进 graph

因为它要处理的是：

```text
- sampled_token_ids 的 CPU list 化；
- logprobs tensor 的 CPU numpy 化；
- invalid request 清空；
- routed experts 快照；
- Python dataclass 回填；
- event 同步和生命周期管理。
```

这些是 output transport / Python object 逻辑，不是 model forward CUDA graph。

---

## 12. pooling 输出是否在 CUDA graph 内

pooling 模型比 generation 特殊：它不走 token sampler，但 pooling output 也通常在 forward graph 边界之后处理。

V1 路径中，forward 结束后如果是 pooling model：

```python
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4345` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4352`

`_pool()` 做三件事：

```text
1. 构造 pooling_metadata；
2. 调用 model.pooler(hidden_states, pooling_metadata)；
3. 构造 ModelRunnerOutput，并把 pooler output 拷到 CPU 或包装成 AsyncGPUPoolingModelRunnerOutput。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3342` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3405`

核心代码：

```python
raw_pooler_output = model.pooler(
    hidden_states=hidden_states,
    pooling_metadata=pooling_metadata,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3365` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3368`

如果是 CUDA-like 平台并且有 finished pooling output，会返回：

```text
AsyncGPUPoolingModelRunnerOutput
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3400` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3405`

所以 pooling 可以理解为：

```text
CUDA graph 可能覆盖 backbone forward → hidden_states；
pooler / pooling metadata / CPU output copy 是 forward 后输出路径。
```

---

## 13. OutputProcessor 是否在 CUDA graph 内

完全不在。

`OutputProcessor` 位于 Engine 层，处理的是 `Scheduler.update_from_output()` 后得到的 `EngineCoreOutput`，把它转换成最终 `RequestOutput` / `PoolingRequestOutput`。

入口：

```python
def process_outputs(
    self,
    engine_core_outputs: list[EngineCoreOutput],
    ...
) -> OutputProcessorOutput:
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:576`

它做的事包括：

```text
1. 更新统计信息；
2. detokenize token ids；
3. 检查 stop string；
4. 更新 sample logprobs / prompt logprobs；
5. 构造 CompletionOutput / PoolingOutput；
6. 构造 RequestOutput / PoolingRequestOutput；
7. 放入 AsyncLLM queue 或返回给 LLMEngine；
8. 清理 finished request state；
9. 必要时向 EngineCore 发 abort。
```

位置：`code/vllm/vllm/v1/engine/output_processor.py:576` 到 `code/vllm/vllm/v1/engine/output_processor.py:693`

这部分是 CPU / Python / tokenizer / request state 逻辑，和 CUDA graph 没有执行边界上的重叠。

---

## 14. Scheduler.update_from_output 是否在 CUDA graph 内

不在。

`Scheduler.update_from_output(scheduler_output, model_output)` 在 EngineCore 拿到 `ModelRunnerOutput` 后执行：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`code/vllm/vllm/v1/engine/core.py:504` 到 `code/vllm/vllm/v1/engine/core.py:506`

Scheduler update 负责：

```text
- 把 sampled tokens 写回 request 状态；
- 处理 stop / finish；
- 处理 spec decode accepted tokens；
- 释放 KV blocks；
- 处理 prompt_logprobs / logprobs 输出；
- 生成 EngineCoreOutputs；
- 更新 draft token ids / connector 信息。
```

这显然不属于 model forward CUDA graph。

从协议角度看：

```text
SchedulerOutput：Scheduler → Worker 的计划
ModelRunnerOutput：Worker → Scheduler 的对账凭证
EngineCoreOutput：Scheduler → OutputProcessor 的请求级输出
RequestOutput：OutputProcessor → 用户侧输出
```

CUDA graph 只覆盖其中 Worker 内部 forward 的 GPU 执行片段。

---

## 15. 为什么 output 不适合 capture

可以从“动态性”角度理解。

### 15.1 request 数和输出长度动态

生成输出不是固定 shape：

```text
普通 decode：每个请求通常 1 token；
spec decode：每个请求可能接受不同数量 token；
jump decoding：每轮生成 token 数可能不同；
chunked prefill：某些请求本轮不应该产出 token；
pooling：只有 finished 的请求产出 pooling output。
```

因此 `ModelRunnerOutput.sampled_token_ids` 是：

```text
list[list[int]]
```

位置：`code/vllm/vllm/v1/outputs.py:240` 到 `code/vllm/vllm/v1/outputs.py:244`

它不是固定 GPU tensor 语义的最终用户输出。

### 15.2 采样参数按请求动态

Sampler 依赖：

```text
temperature
top_p / top_k
min_p
allowed_token_ids
bad_words
penalties
logit processors
random generators
logprobs_mode
logprob_token_ids
```

这些都来自 `SamplingMetadata`，按 batch / request 动态变化。

### 15.3 logprobs 返回结构动态

logprobs 可能是：

```text
None
Top-K + sampled token
全 vocab unsorted logprobs
指定 token ids logprobs
prompt logprobs dict
```

对应输出字段包括：

```text
LogprobsTensors
LogprobsLists
prompt_logprobs_dict: dict[str, LogprobsTensors | None]
```

位置：`code/vllm/vllm/v1/outputs.py:27` 到 `code/vllm/vllm/v1/outputs.py:70`

这类结构组织不适合 capture 成固定 graph。

### 15.4 CPU 状态必须立即更新

采样后要更新 Worker 侧状态：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4461` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4463`

bookkeeping 还会更新：

```text
input_batch.token_ids_cpu
input_batch.num_tokens_no_spec
req_state.output_token_ids
prev_sampled_token_ids
prev_req_id_to_index
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3693` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3725`

这些是下一轮调度和采样的状态基础。

### 15.5 Python 对象生命周期动态

输出阶段构造和传递的是：

```text
ModelRunnerOutput
AsyncGPUModelRunnerOutput
EngineCoreOutput
RequestOutput
CompletionOutput
PoolingRequestOutput
```

这些对象携带 request id、finish reason、stop reason、metrics、trace、queue 等 Python 状态，不是 CUDA graph 的目标。

---

## 16. graph 内外的对象流

可以按对象流记：

```text
SchedulerOutput
  → execute_model()
      → InputBatch / attention metadata / model inputs
      → CUDA graph / compiled forward
          → hidden_states 或 IntermediateTensors
      → compute_logits
      → ExecuteModelState
  → sample_tokens(grammar_output)
      → SamplerOutput      # GPU tensor 形式 sampled ids / logprobs
      → bookkeeping
      → ModelRunnerOutput  # Scheduler 可消费的 Python 结构
  → Scheduler.update_from_output()
      → EngineCoreOutput   # 每个 request 的核心输出
  → OutputProcessor.process_outputs()
      → RequestOutput / PoolingRequestOutput
```

对应 graph 边界可以画成：

```text
                 ┌──────────── CUDA graph / compile wrapper ────────────┐
input buffers →  │ model forward / attention / KV update / hidden states │  → hidden_states
                 └───────────────────────────────────────────────────────┘

hidden_states[logits_indices]
  → compute_logits
  → grammar bitmask
  → sampler / rejection sampler
  → logprobs / prompt_logprobs
  → ModelRunnerOutput
  → Scheduler.update_from_output
  → OutputProcessor
```

---

## 17. 容易疑惑的点

### 17.1 CUDA graph 是否覆盖整个 `execute_model()`？

不是。

`execute_model()` 包含大量 graph 外逻辑：

```text
_update_states
_prepare_inputs
_determine_batch_execution_and_padding
_get_slot_mappings
_build_attention_metadata
_preprocess
compute_logits
ExecuteModelState 保存
```

CUDA graph 主要覆盖其中的 `_model_forward()` / `self.model(...)`。

### 17.2 FULL graph 是否包含 sampler？

通常不包含。

FULL 指的是 full model forward，不是 full engine step。

```text
FULL graph ≈ 整个模型 forward graph；
不是 Scheduler + sampling + output processor 的全链路 graph。
```

### 17.3 compute_logits 为什么也在 graph 外？

因为 vLLM 把模型 backbone forward 的稳定输出边界放在 hidden states，之后只对 `logits_indices` 位置计算 logits，并根据采样/logprobs需求动态处理。

### 17.4 sampler 虽然是 torch module，为什么不 capture？

`Sampler` 确实是 `nn.Module`，但它不是模型 `self.model(...)` forward wrapper 的一部分；它由 `sample_tokens()` 在 forward 后单独调用，并依赖动态 sampling metadata 和 output state。

### 17.5 async output copy 是不是 CUDA graph？

不是。

它使用独立 CUDA stream 和 event 做 D2H copy overlap，优化的是输出拷贝等待，不是 graph replay。

### 17.6 pooling model 会不会走 sampler？

不会走 token sampler。

pooling 路径是：

```text
forward → hidden_states → _pool() → pooler_output → ModelRunnerOutput
```

但 `_pool()` 和 output copy 仍然是 forward 后逻辑。

### 17.7 `cudagraph_stats` 在输出里，是否说明 output 在 graph 内？

不是。

`cudagraph_stats` 只是记录本轮 forward dispatch 的统计，并随 `ModelRunnerOutput` 带回 Scheduler。

---

## 18. 最小伪代码

### 18.1 generation 路径

```text
# execute_model()
_update_states(scheduler_output)
logits_indices, spec_decode_metadata = _prepare_inputs(...)

cudagraph_mode, batch_desc, ..., cudagraph_stats = \
    _determine_batch_execution_and_padding(...)

attn_metadata = _build_attention_metadata(...)
input_ids, inputs_embeds, positions, model_kwargs = _preprocess(...)

with set_forward_context(
    attn_metadata,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    slot_mapping=slot_mappings,
):
    hidden_states = self.model(...)       # graph / compiled forward 边界

sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)

self.execute_model_state = ExecuteModelState(..., logits, hidden_states, ...)
return None
```

### 18.2 sample_tokens 路径

```text
# sample_tokens(grammar_output)
state = self.execute_model_state
self.execute_model_state = None

if grammar_output is not None:
    apply_grammar_bitmask(..., logits)

sampler_output = _sample(logits, spec_decode_metadata)
_update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

# spec decode draft proposal / bookkeeping / logprobs / prompt logprobs
num_nans, logprobs, valid_ids, prompt_logprobs, req_ids, req_map, invalid = \
    _bookkeeping_sync(...)

output = ModelRunnerOutput(
    req_ids=req_ids,
    sampled_token_ids=valid_ids,
    logprobs=logprobs,
    prompt_logprobs_dict=prompt_logprobs,
    cudagraph_stats=cudagraph_stats,
)

if use_async_scheduling:
    return AsyncGPUModelRunnerOutput(output, sampler_output.sampled_token_ids, ...)
return output
```

### 18.3 pooling 路径

```text
with set_forward_context(...):
    hidden_states = self.model(...)       # graph / compiled forward 边界

if is_pooling_model:
    pooling_metadata = input_batch.get_pooling_metadata()
    raw_pooler_output = model.pooler(hidden_states, pooling_metadata)
    return ModelRunnerOutput(pooler_output=copy_to_cpu_or_async_wrapper(...))
```

---

## 19. 总结

Sampler / Output 与 CUDA graph 的边界可以压缩成：

```text
CUDA graph / compile wrapper：
  负责让 model forward 以稳定 shape、稳定 buffer、稳定 attention metadata 执行，产出 hidden_states。

forward 后 GPU 计算：
  compute_logits、grammar mask、sampler、logprobs、rejection sampler 继续使用 GPU tensor，
  但通常不属于 model forward CUDA graph。

输出回收：
  bookkeeping、D2H copy、ModelRunnerOutput、Scheduler.update_from_output、OutputProcessor
  是动态 Python / CPU 状态机逻辑，位于 graph 外。
```

一句话压缩：

```text
vLLM 的 CUDA graph 边界不是“一轮请求执行”，而是“模型 forward”；sampler 和 output 仍然留在 graph 外，用动态状态机处理采样约束、logprobs、spec decode、CPU 状态更新和最终 RequestOutput。
```
