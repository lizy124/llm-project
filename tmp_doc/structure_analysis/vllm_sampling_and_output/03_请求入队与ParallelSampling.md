# 03 请求入队与 Parallel Sampling

本篇梳理 `SamplingParams` 进入 V1 engine 后如何被校验、补全、包装成 `EngineCoreRequest` 和内部 `Request`，以及 `n > 1` 时如何拆成 parallel sampling 的 child request。

## 1. 从 AsyncLLM/LLMEngine 开始

常见入口：

- 异步：`AsyncLLM.add_request()`，`code/vllm/vllm/v1/engine/async_llm.py:280`
- 同步：`LLMEngine.add_request()`，`code/vllm/vllm/v1/engine/llm_engine.py:218`

基本链路：

```text
AsyncLLM.generate()
  ↓
AsyncLLM.add_request()
  ↓
InputProcessor.process_inputs()
  ↓
EngineCoreRequest
  ↓
OutputProcessor.add_request()
  ↓
EngineCoreClient.add_request_async()
  ↓
EngineCore.preprocess_add_request()
  ↓
Request.from_engine_core_request()
  ↓
Scheduler.add_request()
```

## 2. InputProcessor 的职责

文件：`code/vllm/vllm/v1/engine/input_processor.py`

主入口：`code/vllm/vllm/v1/engine/input_processor.py:242`

职责：

1. 校验 sampling/pooling 参数；
2. 校验 LoRA；
3. 对 prompt、tokens、多模态输入做 preprocess；
4. 校验 prompt 长度；
5. 补全 `SamplingParams`；
6. 构造 `EngineCoreRequest`。

## 3. 参数校验

`InputProcessor._validate_params()`：`code/vllm/vllm/v1/engine/input_processor.py:82`

当 params 是 `SamplingParams` 时：

- 确认模型支持 generation；
- 调用 `params.verify(model_config, speculative_config, structured_outputs_config, tokenizer)`；
- 校验 `thinking_token_budget` 是否允许。

这一步把纯参数级校验推进到模型/engine 级校验。

## 4. process_inputs() 关键流程

入口：`code/vllm/vllm/v1/engine/input_processor.py:242`

### 4.1 校验参数与 LoRA

锚点：`code/vllm/vllm/v1/engine/input_processor.py:256`

先校验 params，再校验 LoRA 请求，避免无效请求进入 engine core。

### 4.2 输入预处理

锚点：`code/vllm/vllm/v1/engine/input_processor.py:269`

如果 prompt 已经是 `EngineInput`，直接使用；否则走 `InputPreprocessor.preprocess()`。

这一步会把外部 prompt 变成 engine 可理解的 token ids、多模态 features、encoder/decoder input 等。

### 4.3 平台与模型输入校验

锚点：`code/vllm/vllm/v1/engine/input_processor.py:296`

包括平台能力、模型输入格式、prompt 长度等。

### 4.4 拆 encoder / decoder 输入

锚点：`code/vllm/vllm/v1/engine/input_processor.py:298`

encoder-decoder 或多模态模型需要把不同输入拆开保存。

### 4.5 clone 并补全 SamplingParams

锚点：`code/vllm/vllm/v1/engine/input_processor.py:313`

处理 generation 请求时：

1. `sampling_params = params.clone()`；
2. 如果 `max_tokens is None`，设为 `max_model_len - prompt_len`；
3. `update_from_generation_config()` 注入 EOS / generation config；
4. `update_from_tokenizer()` 处理 bad words tokenization。

因此，协议层传进来的 `SamplingParams` 和 engine core 里真正使用的对象不是同一个引用，而是补全后的 clone。

## 5. EngineCoreRequest

定义：`code/vllm/vllm/v1/engine/__init__.py:86`

主要字段：

- `request_id`；
- `prompt_token_ids`；
- `prompt_embeds`；
- `mm_features`；
- `sampling_params`；
- `pooling_params`；
- `arrival_time`；
- `lora_request`；
- `cache_salt`；
- `priority`；
- `trace_headers`；
- `external_req_id`；
- reasoning 状态。

`params` property：`code/vllm/vllm/v1/engine/__init__.py:137`

它返回 `sampling_params` 或 `pooling_params`，让上层代码统一访问 generation/pooling 参数。

## 6. AsyncLLM.add_request() 中的 request id 处理

入口：`code/vllm/vllm/v1/engine/async_llm.py:280`

关键步骤：

1. 调 `input_processor.process_inputs()` 得到 `EngineCoreRequest`：`code/vllm/vllm/v1/engine/async_llm.py:349`
2. 调 `assign_request_id()`：`code/vllm/vllm/v1/engine/async_llm.py:368`
3. 外部 request id 保存为 `external_req_id`；
4. 内部 request id 可能加随机后缀，避免冲突；
5. 后续 `OutputProcessor` 会把内部 id 映射回外部 id。

## 7. n == 1：普通请求路径

当 `params.n == 1`：`code/vllm/vllm/v1/engine/async_llm.py:381`

直接调用 `_add_request()`：

```text
AsyncLLM.add_request()
  ↓
AsyncLLM._add_request()
  ↓
OutputProcessor.add_request()
  ↓
EngineCoreClient.add_request_async()
```

`OutputProcessor.add_request()` 会注册 frontend request state 和 collector。

## 8. n > 1：Parallel Sampling

当 `params.n > 1`：`code/vllm/vllm/v1/engine/async_llm.py:388`

V1 不把一个 request 内部扩成多个 sequence，而是创建 `ParentRequest`，再 fan-out 多个 child request。

## 9. ParentRequest

文件：`code/vllm/vllm/v1/engine/parallel_sampling.py`

类定义：`code/vllm/vllm/v1/engine/parallel_sampling.py:13`

### 9.1 保存内容

初始化相关锚点：`code/vllm/vllm/v1/engine/parallel_sampling.py:36`

保存：

- parent request id；
- external request id；
- parent sampling params；
- child request 输出状态；
- child 到 parent 的聚合信息。

### 9.2 child sampling params

`_get_child_sampling_params(index)`：`code/vllm/vllm/v1/engine/parallel_sampling.py:52`

逻辑：

1. shallow copy parent `SamplingParams`；
2. child 的 `n` 强制设为 1；
3. 如果 parent `seed is None`，所有 child 可复用一个 cached child params；
4. 如果 parent 有 seed，每个 child 使用 `seed + index`。

这样保证：

- 对外仍然是一个请求返回 n 个候选；
- 对 engine/scheduler 来说是多个独立请求；
- seed 场景下可复现且每条候选随机流不同。

### 9.3 child request id

`get_child_info(index)`：`code/vllm/vllm/v1/engine/parallel_sampling.py:83`

child request id 形如：

```text
{index}_{parent_request_id}
```

返回 child id 和 child sampling params。

### 9.4 输出聚合

`get_outputs()`：`code/vllm/vllm/v1/engine/parallel_sampling.py:100`

不同 output kind 下行为不同：

- `FINAL_ONLY`：等所有 child 最终输出完成后再聚合成 parent output；
- streaming/DELTA：child 产生输出后可以直接转成 parent 的增量输出；
- 每个 child 的输出 index 对应最终 `CompletionOutput.index`。

## 10. EngineCoreRequest 到内部 Request

### 10.1 EngineCore.preprocess_add_request()

入口：`code/vllm/vllm/v1/engine/core.py:853`

核心调用：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

锚点：`code/vllm/vllm/v1/engine/core.py:867`

### 10.2 Request.from_engine_core_request()

入口：`code/vllm/vllm/v1/request.py:197`

它会把 `EngineCoreRequest.sampling_params` 原样带入内部 `Request`。

### 10.3 Request.__init__()

锚点：`code/vllm/vllm/v1/request.py:80`

关键处理：

- 保存 `self.sampling_params`；
- 根据 `StructuredOutputsParams` 构造 `StructuredOutputRequest`；
- generation request 设置 `self.max_tokens = sampling_params.max_tokens`；
- 如果有 structured output，初始状态可能进入 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`；
- 从 `sampling_params.extra_args` 提取 `kv_transfer_params`。

## 11. Request 与 sampling 参数的关系

进入 scheduler 后，内部 `Request` 才是运行态对象。它会携带：

- prompt token ids；
- output token ids；
- sampling params；
- structured output request；
- status；
- stop reason；
- max tokens；
- KV transfer 参数；
- request metrics。

采样参数后续会被 worker input batch 读取，变成 GPU 侧 sampling state。

## 12. SamplingParams 到 SamplingMetadata / GPU state

旧路径中：

- `SamplingMetadata` 定义：`code/vllm/vllm/v1/sample/metadata.py:14`
- `gpu_input_batch.add_request()` 提取 sampling params：`code/vllm/vllm/v1/worker/gpu_input_batch.py:335`
- `_make_sampling_metadata()`：`code/vllm/vllm/v1/worker/gpu_input_batch.py:831`

新 GPU worker 路径中：

- sampling state 位于 `code/vllm/vllm/v1/worker/gpu/sample/states.py:17`
- logit bias state 位于 `code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:15`
- penalties state 位于 `code/vllm/vllm/v1/worker/gpu/sample/penalties.py:14`
- bad words state 位于 `code/vllm/vllm/v1/worker/gpu/sample/bad_words.py:15`

两条路径的共同点是：`Request.sampling_params` 会被拆成 batch/state 级字段，供 sampler 高效使用。

## 13. 本阶段小结

```text
外部请求参数
  -> SamplingParams
  -> InputProcessor clone + verify + 补 max_tokens/generation config/tokenizer 信息
  -> EngineCoreRequest
  -> Request
  -> 如果 n > 1，则 parent fan-out 多个 child Request
  -> Scheduler/Worker 读取 Request.sampling_params 进入实际采样
```

理解重点：`SamplingParams` 是语义对象，`Request` 是运行态对象，`ParentRequest` 是 `n > 1` 的前台聚合对象。
