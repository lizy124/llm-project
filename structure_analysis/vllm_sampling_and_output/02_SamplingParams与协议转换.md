# 02 SamplingParams 与协议转换

本篇梳理外部请求如何变成 vLLM 内部的 `SamplingParams`，以及 `SamplingParams` 自身字段、默认值、校验、输出模式和结构化输出参数。

## 1. SamplingParams 的定位

`SamplingParams` 定义在 `code/vllm/vllm/sampling_params.py:199`。

它是 generation 请求最核心的采样配置对象。OpenAI completion、chat completion、responses，以及离线 Python API 最终都会把采样相关参数归一到它。

它不执行采样，只描述采样行为。

## 2. 辅助类型

### 2.1 SamplingType

定义：`code/vllm/vllm/sampling_params.py:64`

取值：

- `GREEDY`：贪心解码；
- `RANDOM`：随机采样；
- `RANDOM_SEED`：带显式 seed 的随机采样。

推导逻辑在 `SamplingParams.sampling_type`：`code/vllm/vllm/sampling_params.py:679`

规则：

```text
temperature < 1e-5 -> GREEDY
seed is not None   -> RANDOM_SEED
otherwise          -> RANDOM
```

### 2.2 StructuredOutputsParams

定义：`code/vllm/vllm/sampling_params.py:71`

支持：

- `json`：JSON schema；
- `regex`：正则；
- `choice`：固定候选；
- `grammar`：文法；
- `json_object`：OpenAI JSON object 模式；
- `structural_tag`：结构标签；
- `disable_any_whitespace`；
- `disable_additional_properties`；
- `whitespace_pattern`。

关键约束：同一个 `StructuredOutputsParams` 只能启用一种 constraint，不能同时指定 JSON schema、regex、choice 等多种约束。

### 2.3 RepetitionDetectionParams

定义：`code/vllm/vllm/sampling_params.py:145`

用于重复 n-gram/pattern 检测，字段包括：

- `max_pattern_size`；
- `min_pattern_size`；
- `min_count`。

当 scheduler 检测到重复模式时，finish reason 会走 repetition。

### 2.4 RequestOutputKind

定义：`code/vllm/vllm/sampling_params.py:182`

三种输出模式：

| 模式 | 含义 | 常见场景 |
|---|---|---|
| `CUMULATIVE` | 每次返回截至当前的完整输出 | 部分内部/兼容路径 |
| `DELTA` | 每次只返回增量 | streaming |
| `FINAL_ONLY` | 只返回最终输出 | non-streaming OpenAI 请求 |

## 3. SamplingParams 主要字段

类定义：`code/vllm/vllm/sampling_params.py:199`

### 3.1 候选数量

- `n`：同一个 prompt 返回多少条候选输出。

锚点：`code/vllm/vllm/sampling_params.py:213`

V1 中 `n > 1` 通常会被 `ParentRequest` 拆成多个 child request，而不是一个 request 内直接维护多个 sequence。

### 3.2 penalty 类字段

锚点：`code/vllm/vllm/sampling_params.py:224`

- `presence_penalty`：token 出现过就惩罚，鼓励生成新 token；
- `frequency_penalty`：按出现频率惩罚；
- `repetition_penalty`：对 prompt + generated token 中重复 token 做惩罚。

### 3.3 随机采样字段

锚点：`code/vllm/vllm/sampling_params.py:236`

- `temperature`：温度。0 或极小值会进入 greedy；
- `top_p`：nucleus sampling；
- `top_k`：只保留 top-k token，0 或 -1 表示不限制；
- `min_p`：相对最大概率的最小概率阈值；
- `seed`：随机种子。

### 3.4 stop 字段

锚点：`code/vllm/vllm/sampling_params.py:252`

- `stop`：字符串 stop；
- `stop_token_ids`：token id stop；
- `ignore_eos`：是否忽略 EOS；
- `include_stop_str_in_output`：输出中是否保留 stop 字符串。

注意：

- `stop_token_ids` 和 EOS 可以在 scheduler/token 级检查；
- `stop` 字符串必须在 detokenize 后由 frontend 检查。

### 3.5 长度字段

锚点：`code/vllm/vllm/sampling_params.py:262`

- `max_tokens`：每条输出最多生成 token 数；
- `min_tokens`：至少生成多少 token 后才允许 EOS/stop 生效。

### 3.6 logprobs 字段

锚点：`code/vllm/vllm/sampling_params.py:267`

- `logprobs`：输出 token 的 logprobs 数量；
- `prompt_logprobs`：prompt token 的 logprobs 数量；
- `logprob_token_ids`：只返回指定 token ids 的 logprobs。

特殊点：

- `logprobs=True` 会归一成 `1`；
- `logprobs=-1` 表示全 vocab；
- 指定 `prompt_logprobs` 时通常会跳过 prefix cache 的部分读取优化。

### 3.7 detokenize 与特殊 token

锚点：`code/vllm/vllm/sampling_params.py:290`

- `detokenize`；
- `skip_special_tokens`；
- `spaces_between_special_tokens`。

### 3.8 logits processor / 结构化输出相关

锚点：`code/vllm/vllm/sampling_params.py:316`

- `structured_outputs`；
- `logit_bias`；
- `allowed_token_ids`；
- `bad_words`；
- `extra_args`；
- `thinking_token_budget`；
- `repetition_detection`。

这些字段后续会进入 GPU sampling state、structured output manager 或 scheduler stop 逻辑。

## 4. 构造与规范化

### 4.1 from_optional()

入口：`code/vllm/vllm/sampling_params.py:355`

这是 OpenAI 协议转换中最常用的构造方式。它会：

- 把 `None` 转为默认值；
- 处理 `logit_bias` key 的类型和值域；
- 调用 dataclass 初始化和 `__post_init__()`。

### 4.2 __post_init__()

入口：`code/vllm/vllm/sampling_params.py:429`

关键规范化：

1. 极小但非零 temperature 会被提升到内部下限；
2. `seed == -1` 转为 `None`；
3. `stop` 统一转 list；
4. `stop_token_ids` / `bad_words` 默认空 list；
5. `logprobs=True` / `prompt_logprobs=True` 转为 `1`；
6. greedy 模式强制：
   - `top_p = 1.0`
   - `top_k = 0`
   - `min_p = 0.0`
7. 初始化 `_all_stop_token_ids`；
8. prompt logprobs 相关请求会影响 prefix cache 读取策略。

### 4.3 _verify_args()

入口：`code/vllm/vllm/sampling_params.py:487`

主要校验：

- `n >= 1`，且不超过环境限制；
- penalty 范围；
- `temperature` 范围；
- `top_p` 范围；
- `top_k` 合法值；
- `max_tokens` / `min_tokens`；
- `logprobs` / `prompt_logprobs`；
- stop / bad_words 不能是空字符串。

### 4.4 verify()

入口：`code/vllm/vllm/sampling_params.py:717`

这是引擎侧结合模型配置的二次校验，会结合：

- `model_config`；
- `speculative_config`；
- `structured_outputs_config`；
- tokenizer。

检查内容包括 logprobs、logit bias、allowed token ids、spec decode、structured outputs 等。

## 5. Completion 请求转换

文件：

- `code/vllm/vllm/entrypoints/openai/completion/api_router.py`
- `code/vllm/vllm/entrypoints/openai/completion/protocol.py`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py`

### 5.1 HTTP 入口

`POST /v1/completions` 绑定到 `create_completion()`：`code/vllm/vllm/entrypoints/openai/completion/api_router.py:34`

随后调用 `OpenAIServingCompletion.create_completion()`：`code/vllm/vllm/entrypoints/openai/completion/api_router.py:54`

### 5.2 CompletionRequest

定义：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:45`

包含 OpenAI 标准字段和 vLLM 扩展字段：

- 标准：`frequency_penalty`、`logit_bias`、`logprobs`、`max_tokens`、`n`、`presence_penalty`、`seed`、`stop`、`stream`、`temperature`、`top_p`；
- 扩展：`top_k`、`min_p`、`repetition_penalty`、`stop_token_ids`、`include_stop_str_in_output`、`ignore_eos`、`min_tokens`、`prompt_logprobs`、`allowed_token_ids`；
- 结构化输出：`response_format`、`structured_outputs`；
- 其他：`truncate_prompt_tokens`、`kv_transfer_params`、`vllm_xargs`、`repetition_detection`、`thinking_token_budget`。

### 5.3 to_sampling_params()

入口：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:244`

关键步骤：

1. 请求字段优先，缺省回退 generation config / vLLM 默认值；
2. `echo=True` 且未显式给 `prompt_logprobs` 时，用 `logprobs` 作为 `prompt_logprobs`；
3. `response_format` 转成 `StructuredOutputsParams`；
4. `kv_transfer_params` 放入 `SamplingParams.extra_args["kv_transfer_params"]`；
5. 调用 `SamplingParams.from_optional()`；
6. streaming 时 `output_kind=DELTA`，非 streaming 时 `output_kind=FINAL_ONLY`。

相关锚点：

- 参数默认值处理：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:252`
- echo/prompt logprobs：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:275`
- response_format：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:281`
- extra_args：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:313`
- SamplingParams 构造：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:317`
- output kind：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:337`

## 6. Chat Completion 请求转换

文件：

- `code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py`
- `code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py`

### 6.1 HTTP 入口

`POST /v1/chat/completions`：`code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:40`

调用 serving：`code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:61`

### 6.2 ChatCompletionRequest

定义：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:186`

重点字段：

- OpenAI 标准：`messages`、`frequency_penalty`、`logit_bias`、`logprobs`、`top_logprobs`、`max_tokens`、`max_completion_tokens`、`n`、`presence_penalty`、`response_format`、`seed`、`stop`、`stream`、`temperature`、`top_p`、`tools`、`tool_choice`；
- vLLM 扩展：`top_k`、`min_p`、`repetition_penalty`、`stop_token_ids`、`include_stop_str_in_output`、`ignore_eos`、`min_tokens`、`prompt_logprobs`、`allowed_token_ids`、`bad_words`；
- reasoning：`reasoning_effort`、`thinking_token_budget`、`include_reasoning`；
- render：`chat_template`、`chat_template_kwargs`、`add_generation_prompt`、`continue_final_message`、`mm_processor_kwargs`；
- structured output：`structured_outputs`、`response_format`。

### 6.3 to_sampling_params()

入口：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:549`

关键逻辑：

1. temperature/top_p/top_k/min_p/repetition_penalty 使用请求值优先，缺省回退默认采样参数；
2. `prompt_logprobs` 缺省且 `echo=True` 时使用 `top_logprobs`；
3. `response_format` 转 `StructuredOutputsParams`；
4. chat 的 `logprobs` 是布尔开关，实际传给 `SamplingParams.logprobs` 的是 `top_logprobs if logprobs else None`；
5. chat 会额外传 `bad_words`、`thinking_token_budget`、`allowed_token_ids`、`repetition_detection`；
6. streaming -> `DELTA`，non-streaming -> `FINAL_ONLY`。

相关锚点：

- 入口：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:549`
- 默认值：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:554`
- prompt logprobs：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:577`
- response_format：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:581`
- logprobs 转换：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:629`
- output kind：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:637`
- 扩展字段：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:640`

## 7. Responses API 转换

`ResponsesRequest.to_sampling_params()`：`code/vllm/vllm/entrypoints/openai/responses/protocol.py:342`

支持：

- temperature；
- top_p；
- top_k；
- presence/frequency/repetition penalty；
- text.format 到 structured outputs；
- stop；
- output kind。

serving 调用位置：`code/vllm/vllm/entrypoints/openai/responses/serving.py:436`

## 8. 参数转换后的重要变化

外部请求转成 `SamplingParams` 后，还没最终定型。后续 `InputProcessor.process_inputs()` 会继续：

- clone sampling params；
- 如果 `max_tokens is None`，根据 `max_model_len - prompt_len` 补齐；
- 调用 `update_from_generation_config()` 注入 EOS / generation config；
- 调用 `update_from_tokenizer()` 处理 bad words tokenization。

入口：`code/vllm/vllm/v1/engine/input_processor.py:313`
