# 01 采样与输出总览

本篇先把 vLLM sampling 与 output 的整体边界讲清楚：哪里只是参数，哪里真的采样，哪里只是内部输出，哪里才是用户可见输出。

## 1. 分层视角

```text
协议/API 层
  - OpenAI Completion / Chat / Responses schema
  - 把请求参数转成 SamplingParams

AsyncLLM / LLMEngine 前台层
  - InputProcessor 校验与补全 SamplingParams
  - OutputProcessor 转换 EngineCoreOutputs 到 RequestOutput
  - 维护每个请求的 queue / state / detokenizer / logprobs processor

EngineCore / Scheduler 内核层
  - Scheduler 决定本 step 调度哪些 token
  - 维护 Request 状态、stop 状态、KV block、structured output FSM
  - 把 ModelRunnerOutput 整理成 EngineCoreOutputs

Worker / GPUModelRunner 执行层
  - forward 模型得到 hidden states
  - compute_logits 得到 vocab logits
  - grammar bitmask / penalty / top-p / top-k / temperature 等真正影响采样
  - 生成 token ids 和 logprobs

Serving 输出层
  - OpenAI completion/chat streaming 或 non-streaming response
  - reasoning / tool calls / finish_reason / usage 组装
```

## 2. 端到端主链路

```text
HTTP / Python API request
  ↓
CompletionRequest.to_sampling_params()
ChatCompletionRequest.to_sampling_params()
ResponsesRequest.to_sampling_params()
  ↓
SamplingParams
  ↓
AsyncLLM.add_request() / LLMEngine.add_request()
  ↓
InputProcessor.process_inputs()
  ↓
EngineCoreRequest
  ↓
EngineCore.preprocess_add_request()
  ↓
Request.from_engine_core_request()
  ↓
Scheduler.add_request()
  ↓
Scheduler.schedule()
  ↓
GPUModelRunner.execute_model()
  ↓
GPUModelRunner.sample_tokens()
  ↓
GPUModelRunner.sample()
  ↓
Sampler / RejectionSampler
  ↓
ModelRunnerOutput
  ↓
Scheduler.update_from_output()
  ↓
EngineCoreOutputs
  ↓
OutputProcessor.process_outputs()
  ↓
RequestOutput / CompletionOutput
  ↓
OpenAI response / Python API result
```

关键入口：

- `AsyncLLM.add_request()`：`code/vllm/vllm/v1/engine/async_llm.py:280`
- `InputProcessor.process_inputs()`：`code/vllm/vllm/v1/engine/input_processor.py:242`
- `EngineCore.step()`：`code/vllm/vllm/v1/engine/core.py:479`
- `GPUModelRunner.sample()`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1038`
- `Scheduler.update_from_output()`：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`
- `OutputProcessor.process_outputs()`：`code/vllm/vllm/v1/engine/output_processor.py:576`

## 3. 三类输出不要混淆

### 3.1 ModelRunnerOutput：worker/model runner 输出

定义：`code/vllm/vllm/v1/outputs.py:234`

它表示 worker 执行一个 step 后交给 scheduler 的底层结果，主要包含：

- `req_ids`：当前 batch 请求顺序；
- `req_id_to_index`：request id 到 batch index 映射；
- `sampled_token_ids`：每个请求本 step 新采样出的 token；
- `logprobs`：采样 token 的 logprobs；
- `prompt_logprobs_dict`：prompt logprobs；
- `pooler_output`：embedding/classification/scoring 类输出；
- `kv_connector_output`：KV transfer 相关输出；
- `num_nans_in_logits`：logits NaN 统计；
- `routed_experts`：MoE routed expert 信息。

它不是用户可见输出。

### 3.2 EngineCoreOutputs：engine core 到前台的内部输出

单请求输出 `EngineCoreOutput` 定义在 `code/vllm/vllm/v1/engine/__init__.py:173`。

批量输出 `EngineCoreOutputs` 定义在 `code/vllm/vllm/v1/engine/__init__.py:218`。

它在 scheduler 中由 `update_from_output()` 生成，包含：

- `new_token_ids`；
- `new_logprobs`；
- `new_prompt_logprobs_tensors`；
- `pooling_output`；
- `finish_reason`；
- `stop_reason`；
- `prefill_stats`；
- `kv_transfer_params`；
- `routed_experts`。

它比 `ModelRunnerOutput` 更接近用户输出，但仍是内部结构。

### 3.3 RequestOutput：用户可见输出

`CompletionOutput` 定义在 `code/vllm/vllm/outputs.py:21`。

`RequestOutput` 定义在 `code/vllm/vllm/outputs.py:85`。

这是 LLM/AsyncLLM/OpenAI serving 看到的输出，包含：

- prompt；
- prompt token ids；
- prompt logprobs；
- 一个或多个 `CompletionOutput`；
- detokenized text；
- output token ids；
- cumulative logprob；
- finish reason；
- stop reason；
- metrics；
- cached token 数；
- KV transfer 参数。

## 4. sampling 与 output 的关键分工

| 层 | 是否真正采样 | 是否 detokenize | 是否判断 stop | 是否用户可见 |
|---|---:|---:|---:|---:|
| OpenAI protocol | 否 | 否 | 否 | 否 |
| SamplingParams | 否 | 否 | 只保存参数 | 否 |
| Scheduler | 否 | 否 | stop token / length / repetition | 否 |
| GPU Sampler | 是 | 否 | 通过 min tokens / stop ids mask 间接影响 | 否 |
| OutputProcessor | 否 | 是 | stop string | 接近最终 |
| OpenAI serving | 否 | 否 | tool call / response finish 展示 | 是 |

## 5. 为什么 stop 分散在多个地方

vLLM 的 stop 逻辑不是一个函数全部完成，而是按信息可见性分布：

1. token 级 stop 在 scheduler：
   - EOS；
   - stop token ids；
   - max tokens；
   - max model len；
   - repetition detection。

2. 字符串 stop 在 frontend：
   - 必须 detokenize 后才知道；
   - 由 `IncrementalDetokenizer` 和 `OutputProcessor` 处理。

3. OpenAI serving 层只负责把 finish reason / stop reason 映射到 response schema。

相关锚点：

- token/length/repetition stop：`code/vllm/vllm/v1/core/sched/utils.py:94`
- stop string 检测：`code/vllm/vllm/v1/engine/detokenizer.py:309`
- OutputProcessor 处理 stop string：`code/vllm/vllm/v1/engine/output_processor.py:639`

## 6. structured output 的位置

structured output 的核心不是“输出后检查 JSON 是否合法”，而是在采样前构造 grammar bitmask，直接把不允许 token 的 logits 置为 `-inf`。

```text
Scheduler.get_grammar_bitmask()
  ↓
StructuredOutputManager.grammar_bitmask()
  ↓
GrammarOutput
  ↓
GPUModelRunner.sample()
  ↓
StructuredOutputsWorker.apply_grammar_bitmask()
  ↓
Sampler/RejectionSampler
```

关键锚点：

- scheduler 获取 grammar bitmask：`code/vllm/vllm/v1/core/sched/scheduler.py:1439`
- structured output manager：`code/vllm/vllm/v1/structured_output/__init__.py:204`
- GPU 应用 bitmask：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:23`
- runner 调用位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1046`

## 7. V1 新旧 GPU 采样路径

当前仓库里可以看到两套相关路径：

1. 新 GPU worker 路径：
   - `code/vllm/vllm/v1/worker/gpu/model_runner.py`
   - `code/vllm/vllm/v1/worker/gpu/sample/*`
   - `code/vllm/vllm/v1/worker/gpu/spec_decode/*`

2. 旧/兼容路径：
   - `code/vllm/vllm/v1/worker/gpu_model_runner.py`
   - `code/vllm/vllm/v1/sample/sampler.py`
   - `code/vllm/vllm/v1/sample/metadata.py`

理解时要注意：

- 新路径更倾向把 sampling state 放在 GPU resident state 中；
- 旧路径更明显使用 `SamplingMetadata`；
- 二者都表达同一类采样语义，但实现组织方式不同。
