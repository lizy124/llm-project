# 04. InputProcessor 如何把用户输入转成 EngineCoreRequest？

源码位置：`vllm/vllm/v1/engine/input_processor.py`

本问题关注：`InputProcessor` 在外层 Engine 中的位置，以及它如何把用户侧的 prompt / EngineInput / params / LoRA / 多模态输入处理成 EngineCore 能接收的 `EngineCoreRequest`。

---

## 1. 一句话回答

`InputProcessor` 属于外层 Engine 的输入预处理层。

它的核心职责是：

```text
用户输入 / PromptType / EngineInput
  → 参数校验
  → prompt 预处理 / tokenization / embeds 识别
  → encoder-decoder 输入拆分
  → 多模态特征整理
  → SamplingParams / PoolingParams 复制和补全
  → EngineCoreRequest
```

一句话：

```text
InputProcessor 不负责调度和执行模型；
它负责把外部输入规范化为 EngineCoreRequest，让 EngineCore 后续能把它转换成 Scheduler 内部 Request。
```

完整链路是：

```text
LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
```

---

## 2. InputProcessor 位于哪一层

`InputProcessor` 在外层 Engine 初始化时创建。

同步 `LLMEngine` 中：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:93` 到 `vllm/vllm/v1/engine/llm_engine.py:94`

异步 `AsyncLLM` 中也持有 `InputProcessor`，请求进入时同样会调用：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:360`

所以层次关系是：

```text
LLMEngine / AsyncLLM：
  面向用户输入，持有 InputProcessor。

InputProcessor：
  负责把用户输入转成 EngineCoreRequest。

EngineCore：
  不直接处理用户原始 prompt，而是接收 EngineCoreRequest。

Scheduler：
  不接收 EngineCoreRequest，而是接收 EngineCore 转换后的内部 Request。
```

也就是说：

```text
InputProcessor 是 EngineCore 外面的输入边界层。
```

---

## 3. InputProcessor 初始化时保存哪些上下文

`InputProcessor.__init__()` 入口：

```python
class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
```

位置：`vllm/vllm/v1/engine/input_processor.py:36` 到 `vllm/vllm/v1/engine/input_processor.py:43`

初始化时会保存多类配置：

```python
self.vllm_config = vllm_config
self.model_config = model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.scheduler_config = vllm_config.scheduler_config
self.speculative_config = vllm_config.speculative_config
self.structured_outputs_config = vllm_config.structured_outputs_config
self.observability_config = vllm_config.observability_config
self.use_v2_model_runner = vllm_config.use_v2_model_runner
```

位置：`vllm/vllm/v1/engine/input_processor.py:44` 到 `vllm/vllm/v1/engine/input_processor.py:52`

这些配置会用于后续：

```text
参数校验；
模型输入长度校验；
SamplingParams.verify；
structured output 参数校验；
speculative decoding 参数校验；
LoRA 输入校验；
多模态输入预算校验；
V2 ModelRunner 限制校验。
```

### 3.1 renderer

初始化时会保存 renderer：

```python
self.renderer = renderer or renderer_from_config(vllm_config)
```

位置：`vllm/vllm/v1/engine/input_processor.py:56`

renderer 提供：

```text
tokenizer；
EOS token id；
多模态 processor cache；
render_cmpl / render_chat 输出的 EngineInput 支持。
```

`InputProcessor.tokenizer` 也来自 renderer：

```python
@property
def tokenizer(self) -> TokenizerLike | None:
    return self.renderer.tokenizer
```

位置：`vllm/vllm/v1/engine/input_processor.py:75` 到 `vllm/vllm/v1/engine/input_processor.py:77`

### 3.2 多模态预算

如果模型支持多模态输入：

```python
self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
self.mm_encoder_cache_size = 0
self.skip_prompt_length_check = False
if self.supports_mm_inputs:
    mm_budget = MultiModalBudget(vllm_config, mm_registry)
    self.mm_encoder_cache_size = mm_budget.encoder_cache_size
    self.skip_prompt_length_check = (
        mm_budget.processor.info.skip_prompt_length_check
    )
    mm_budget.reset_cache()  # Not used anymore
```

位置：`vllm/vllm/v1/engine/input_processor.py:58` 到 `vllm/vllm/v1/engine/input_processor.py:67`

这说明 `InputProcessor` 会提前知道：

```text
当前模型是否支持多模态；
多模态 encoder cache size；
是否跳过 prompt length check。
```

这些信息后续用于校验图像 / 音频 / 视频等多模态 placeholder 的 embedding token 数是否超过预算。

### 3.3 InputPreprocessor

初始化最后创建 `InputPreprocessor`：

```python
self.input_preprocessor = InputPreprocessor(
    vllm_config,
    renderer=renderer,
    mm_registry=mm_registry,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:69` 到 `vllm/vllm/v1/engine/input_processor.py:73`

它用于兼容 raw prompt 输入：

```text
raw PromptType
  → InputPreprocessor.preprocess()
  → EngineInput
```

不过源码中已经提示 raw prompt 传入 `InputProcessor` 的用法会被废弃，推荐用户传入 renderer 处理后的结果。

---

## 4. process_inputs() 总流程

`InputProcessor` 的核心入口是：

```python
def process_inputs(
    self,
    request_id: str,
    prompt: PromptType | EngineInput,
    params: SamplingParams | PoolingParams,
    supported_tasks: tuple[SupportedTask, ...],
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
    resumable: bool = False,
) -> EngineCoreRequest:
```

位置：`vllm/vllm/v1/engine/input_processor.py:242` 到 `vllm/vllm/v1/engine/input_processor.py:255`

主流程可以概括为：

```text
process_inputs()
  → _validate_params(params, supported_tasks)
  → _validate_lora(lora_request)
  → 校验 data_parallel_rank
  → 如果 prompt 已经是 EngineInput，直接使用
  → 否则用 InputPreprocessor.preprocess() 处理 raw prompt
  → current_platform.validate_request(processed_inputs, params)
  → split_enc_dec_input(processed_inputs)
  → _validate_model_inputs(encoder_inputs, decoder_inputs)
  → 从 decoder_inputs 取 prompt_token_ids / prompt_embeds / prompt_is_token_ids
  → clone 并补全 SamplingParams 或 PoolingParams
  → 如果是 multimodal，整理 mm_features
  → return EngineCoreRequest(...)
```

可以理解为：

```text
process_inputs() 的输出只有一个核心对象：EngineCoreRequest。
```

---

## 5. 第一步：校验 SamplingParams / PoolingParams

`process_inputs()` 一开始先调用：

```python
self._validate_params(params, supported_tasks)
self._validate_lora(lora_request)
```

位置：`vllm/vllm/v1/engine/input_processor.py:256` 到 `vllm/vllm/v1/engine/input_processor.py:257`

### 5.1 SamplingParams 校验

如果是 generation 请求：

```python
if isinstance(params, SamplingParams):
    supported_generation_tasks = [
        task for task in supported_tasks if task in GENERATION_TASKS
    ]
    if not supported_generation_tasks:
        raise ValueError("This model does not support generation")

    params.verify(
        self.model_config,
        self.speculative_config,
        self.structured_outputs_config,
        self.tokenizer,
    )
```

位置：`vllm/vllm/v1/engine/input_processor.py:88` 到 `vllm/vllm/v1/engine/input_processor.py:100`

这一步确认：

```text
当前模型支持 generation；
SamplingParams 和 model_config 兼容；
speculative decoding 参数合法；
structured output 参数合法；
tokenizer 相关参数合法。
```

### 5.2 thinking_token_budget 限制

如果设置了 `thinking_token_budget`，还会额外检查 reasoning config 和 V2 ModelRunner：

```python
if params.thinking_token_budget is not None:
    if (
        self.vllm_config.reasoning_config is None
        or not self.vllm_config.reasoning_config.enabled
    ):
        raise ValueError(...)
    if self.use_v2_model_runner:
        raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:102` 到 `vllm/vllm/v1/engine/input_processor.py:117`

含义是：

```text
thinking_token_budget 只有在 reasoning_config 启用时可用；
并且当前不支持 V2 ModelRunner。
```

### 5.3 PoolingParams 校验

如果是 pooling 请求：

```python
elif isinstance(params, PoolingParams):
    supported_pooling_tasks = [
        task for task in supported_tasks if task in POOLING_TASKS
    ]
    if not supported_pooling_tasks:
        raise ValueError("This model does not support pooling")
```

位置：`vllm/vllm/v1/engine/input_processor.py:118` 到 `vllm/vllm/v1/engine/input_processor.py:123`

如果 `params.task` 没有设置，会按模型支持任务自动补默认值：

```python
if params.task is None:
    if "token_embed" in supported_pooling_tasks:
        params.task = "token_embed"
    elif "token_classify" in supported_pooling_tasks:
        params.task = "token_classify"
    elif "plugin" in supported_pooling_tasks:
        params.task = "plugin"
```

位置：`vllm/vllm/v1/engine/input_processor.py:125` 到 `vllm/vllm/v1/engine/input_processor.py:131`

然后校验 task 是否支持并调用：

```python
params.verify(self.model_config)
```

位置：`vllm/vllm/v1/engine/input_processor.py:133` 到 `vllm/vllm/v1/engine/input_processor.py:139`

### 5.4 params 类型必须二选一

如果既不是 `SamplingParams`，也不是 `PoolingParams`：

```python
raise TypeError(
    f"params must be either SamplingParams or PoolingParams, "
    f"but got {type(params).__name__}"
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:140` 到 `vllm/vllm/v1/engine/input_processor.py:144`

所以 `InputProcessor` 明确把请求分成两类：

```text
generation request：SamplingParams
pooling request：PoolingParams
```

---

## 6. 第二步：校验 LoRA 请求

LoRA 校验入口：

```python
def _validate_lora(self, lora_request: LoRARequest | None) -> None:
```

位置：`vllm/vllm/v1/engine/input_processor.py:146`

如果请求没带 LoRA，直接返回：

```python
if lora_request is None:
    return
```

位置：`vllm/vllm/v1/engine/input_processor.py:147` 到 `vllm/vllm/v1/engine/input_processor.py:148`

如果请求带了 LoRA，但引擎没有启用 LoRA：

```python
if not self.lora_config:
    raise ValueError(
        f"Got lora_request {lora_request} but LoRA is not enabled!"
    )
```

位置：`vllm/vllm/v1/engine/input_processor.py:150` 到 `vllm/vllm/v1/engine/input_processor.py:154`

如果有 tokenizer，还会提示不同 LoRA 使用不同 tokenizer 的支持已废弃：

```python
logger.warning_once(
    "vLLM has deprecated support for supporting different "
    "tokenizers for different LoRAs. ..."
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:156` 到 `vllm/vllm/v1/engine/input_processor.py:163`

这说明：

```text
InputProcessor 只校验 LoRA 请求是否合法；
真正加载 / 卸载 / pin LoRA adapter 是 EngineCore / Worker 侧能力。
```

---

## 7. 第三步：校验 data_parallel_rank

`process_inputs()` 会校验请求指定的 `data_parallel_rank`：

```python
parallel_config = self.vllm_config.parallel_config
dp_size = parallel_config.data_parallel_size
dp_local_size = parallel_config.data_parallel_size_local
num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
    raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:259` 到 `vllm/vllm/v1/engine/input_processor.py:267`

含义是：

```text
如果调用方显式指定 data_parallel_rank，InputProcessor 会先确认它在当前可用 rank 范围内。
```

这个字段后续会进入 `EngineCoreRequest.data_parallel_rank`，供 EngineCore / DP 相关逻辑使用。

---

## 8. 第四步：区分 EngineInput 和 raw prompt

`InputProcessor` 接收的 `prompt` 可能是两类：

```text
1. 已经处理好的 EngineInput；
2. 原始 PromptType。
```

### 8.1 已经是 EngineInput

判断条件是：

```python
if isinstance(prompt, dict) and "type" in prompt:
```

位置：`vllm/vllm/v1/engine/input_processor.py:269`

这种情况下直接使用：

```python
processed_inputs: EngineInput = prompt
```

位置：`vllm/vllm/v1/engine/input_processor.py:280`

如果同时传了 `tokenization_kwargs`，会警告：

```python
logger.warning_once(
    "Passing tokenization_kwargs to InputProcessor is deprecated "
    "and will be removed in v0.18. You should instead pass "
    "them to Renderer.render_cmpl() or Renderer.render_chat()."
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:270` 到 `vllm/vllm/v1/engine/input_processor.py:275`

如果没有 `arrival_time`，会从 `prompt` 中取，取不到则用当前时间：

```python
if arrival_time is None:
    arrival_time = prompt.get("arrival_time", time.time())
```

位置：`vllm/vllm/v1/engine/input_processor.py:277` 到 `vllm/vllm/v1/engine/input_processor.py:278`

### 8.2 raw prompt

如果不是 EngineInput，会走兼容路径：

```python
logger.warning_once(
    "Passing raw prompts to InputProcessor is deprecated "
    "and will be removed in v0.18. You should instead pass "
    "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:282` 到 `vllm/vllm/v1/engine/input_processor.py:286`

然后设置 arrival time：

```python
if arrival_time is None:
    arrival_time = time.time()
```

位置：`vllm/vllm/v1/engine/input_processor.py:288` 到 `vllm/vllm/v1/engine/input_processor.py:289`

并调用：

```python
processed_inputs = self.input_preprocessor.preprocess(
    prompt,
    tokenization_kwargs=tokenization_kwargs,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:291` 到 `vllm/vllm/v1/engine/input_processor.py:294`

因此 raw prompt 的处理链路是：

```text
raw PromptType
  → InputPreprocessor.preprocess()
  → EngineInput
  → 后续统一处理
```

---

## 9. 第五步：平台级请求校验

拿到 `processed_inputs` 后，会先调用：

```python
current_platform.validate_request(processed_inputs, params)
```

位置：`vllm/vllm/v1/engine/input_processor.py:296`

这一步把平台相关限制提前挡住。

可以理解为：

```text
InputProcessor 不只做模型参数校验，也会让当前平台检查这个 request 是否可运行。
```

具体校验细节取决于 `current_platform` 的实现。

---

## 10. 第六步：拆分 encoder / decoder 输入

接着调用：

```python
encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
self._validate_model_inputs(encoder_inputs, decoder_inputs)
```

位置：`vllm/vllm/v1/engine/input_processor.py:298` 到 `vllm/vllm/v1/engine/input_processor.py:299`

含义是：

```text
EngineInput 可能同时包含 encoder 输入和 decoder 输入；
InputProcessor 会先拆开，再分别校验。
```

对于 decoder-only 模型，通常关注 decoder input。

对于 encoder-decoder 或多模态模型，encoder input / decoder input 都可能参与后续调度。

拆分后的输入会进入 `_validate_model_inputs()`。

---

## 11. 第七步：校验模型输入长度和 token id

模型输入校验入口：

```python
def _validate_model_inputs(
    self,
    encoder_input: SingletonInput | None,
    decoder_input: SingletonInput,
):
    if encoder_input is not None:
        self._validate_model_input(encoder_input, prompt_type="encoder")

    self._validate_model_input(decoder_input, prompt_type="decoder")
```

位置：`vllm/vllm/v1/engine/input_processor.py:486` 到 `vllm/vllm/v1/engine/input_processor.py:494`

### 11.1 prompt 长度校验

单个输入校验里先计算 prompt length：

```python
prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
self._validate_prompt_len(prompt_len, prompt_type)
```

位置：`vllm/vllm/v1/engine/input_processor.py:451` 到 `vllm/vllm/v1/engine/input_processor.py:452`

`_validate_prompt_len()` 会处理三类情况。

第一，某些多模态 processor 可以跳过长度校验：

```python
if self.skip_prompt_length_check:
    return
```

位置：`vllm/vllm/v1/engine/input_processor.py:392` 到 `vllm/vllm/v1/engine/input_processor.py:393`

第二，decoder prompt 不能为空：

```python
if prompt_len == 0 and prompt_type == "decoder":
    raise ValueError(f"The {prompt_type} prompt cannot be empty")
```

位置：`vllm/vllm/v1/engine/input_processor.py:395` 到 `vllm/vllm/v1/engine/input_processor.py:396`

第三，长度不能超过对应上限：

```python
max_prompt_len = (
    model_config.max_model_len
    if prompt_type == "decoder"
    else self.mm_encoder_cache_size
)
if prompt_len > max_prompt_len:
    raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:398` 到 `vllm/vllm/v1/engine/input_processor.py:422`

对于 decoder generation 请求，如果 prompt 刚好等于 `max_model_len`，也会报错：

```python
elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
    raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:423` 到 `vllm/vllm/v1/engine/input_processor.py:432`

原因是 generation 至少还要生成一个 token：

```text
decoder prompt length == max_model_len
  → 没有空间生成 output token
  → 对 generate runner 不合法
```

### 11.2 多模态 encoder cache size 校验

如果输入类型是 multimodal：

```python
if prompt_input["type"] == "multimodal":
    decoder_mm_positions = prompt_input["mm_placeholders"]
    for modality, mm_positions in decoder_mm_positions.items():
        for mm_position in mm_positions:
            num_embeds = mm_position.get_num_embeds()
            if num_embeds > self.mm_encoder_cache_size:
                raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:454` 到 `vllm/vllm/v1/engine/input_processor.py:467`

这一步检查单个多模态 item 的 embedding token 数是否超过预分配 encoder cache size。

### 11.3 token id 是否越界

如果输入包含 token ids，并且 tokenizer 存在：

```python
if prompt_ids and tokenizer is not None:
    max_input_id = max(prompt_ids, default=0)
    model_vocab_size = model_config.get_vocab_size()
    if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
        raise ValueError(f"Token id {max_input_id} is out of vocabulary")
```

位置：`vllm/vllm/v1/engine/input_processor.py:469` 到 `vllm/vllm/v1/engine/input_processor.py:484`

这里使用的是：

```text
max(tokenizer.max_token_id, model_vocab_size - 1)
```

原因在注释中说明：某些模型的 tokenizer vocab 和 model vocab 不完全一致，例如 Qwen3 或多模态 placeholder tokens。

---

## 12. 第八步：提取 decoder token ids / embeds

校验完成后，`process_inputs()` 从 `decoder_inputs` 里取出实际送入 EngineCore 的 prompt 表示。

如果 decoder 输入是 embeds：

```python
if decoder_inputs["type"] == "embeds":
    prompt_embeds = decoder_inputs["prompt_embeds"]
    prompt_token_ids = decoder_inputs.get("prompt_token_ids")
    prompt_is_token_ids = decoder_inputs.get("is_token_ids")
```

位置：`vllm/vllm/v1/engine/input_processor.py:302` 到 `vllm/vllm/v1/engine/input_processor.py:305`

否则使用 token ids：

```python
else:
    prompt_token_ids = decoder_inputs["prompt_token_ids"]
    prompt_embeds = None
    prompt_is_token_ids = None
```

位置：`vllm/vllm/v1/engine/input_processor.py:306` 到 `vllm/vllm/v1/engine/input_processor.py:309`

这里的三种字段含义是：

```text
prompt_token_ids：
  token id 形式的 prompt。

prompt_embeds：
  预计算 embedding 形式的 prompt。

prompt_is_token_ids：
  mixed-mode 输入中每个位置是 token id 还是 prompt embed 的 mask。
```

这些字段会原样进入 `EngineCoreRequest`。

---

## 13. 第九步：复制并补全 SamplingParams / PoolingParams

`InputProcessor` 不直接复用用户传入的 params，而是 clone。

### 13.1 SamplingParams

如果是 `SamplingParams`：

```python
sampling_params = params.clone()
```

位置：`vllm/vllm/v1/engine/input_processor.py:313` 到 `vllm/vllm/v1/engine/input_processor.py:315`

如果用户没有设置 `max_tokens`，会补成模型剩余长度：

```python
if sampling_params.max_tokens is None:
    seq_len = length_from_prompt_token_ids_or_embeds(
        prompt_token_ids, prompt_embeds
    )
    sampling_params.max_tokens = self.model_config.max_model_len - seq_len
```

位置：`vllm/vllm/v1/engine/input_processor.py:317` 到 `vllm/vllm/v1/engine/input_processor.py:321`

然后用 generation config 和 tokenizer 更新参数：

```python
sampling_params.update_from_generation_config(
    self.generation_config_fields,
    self.renderer.get_eos_token_id(),
)
if self.tokenizer is not None:
    sampling_params.update_from_tokenizer(self.tokenizer)
```

位置：`vllm/vllm/v1/engine/input_processor.py:323` 到 `vllm/vllm/v1/engine/input_processor.py:328`

所以 SamplingParams 在进入 EngineCore 前已经完成：

```text
clone；
max_tokens 默认值补全；
generation_config 合并；
tokenizer 相关 stop/eos 信息更新。
```

### 13.2 PoolingParams

如果是 pooling 请求：

```python
pooling_params = params.clone()
```

位置：`vllm/vllm/v1/engine/input_processor.py:329` 到 `vllm/vllm/v1/engine/input_processor.py:330`

所以 EngineCoreRequest 中只会有一个参数分支非空：

```text
sampling_params != None, pooling_params == None
或
sampling_params == None, pooling_params != None
```

---

## 14. 第十步：整理多模态 mm_features

如果 decoder 输入是 multimodal：

```python
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]
    decoder_mm_positions = decoder_inputs["mm_placeholders"]
    decoder_mm_hashes = decoder_inputs["mm_hashes"]
```

位置：`vllm/vllm/v1/engine/input_processor.py:335` 到 `vllm/vllm/v1/engine/input_processor.py:338`

### 14.1 校验 mm_hashes

先确保所有 hash leaf 都是字符串：

```python
if not all(
    isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
):
    raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:340` 到 `vllm/vllm/v1/engine/input_processor.py:347`

这说明多模态 cache / identifier 依赖稳定的字符串 hash。

### 14.2 按输入序列位置排序多模态项

然后按多模态 placeholder 在输入序列中的位置排序：

```python
sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)
```

位置：`vllm/vllm/v1/engine/input_processor.py:352`

注释说明：

```python
# Merge and flatten multimodal placeholders, hashes and inputs
# from dictionaries to lists, and sort them by each item's position
# in the input sequence.
```

位置：`vllm/vllm/v1/engine/input_processor.py:349` 到 `vllm/vllm/v1/engine/input_processor.py:351`

也就是说，多模态输入原本可能按 modality 分组：

```text
image: [...]
audio: [...]
video: [...]
```

但 EngineCoreRequest 需要的是按序列位置展开后的 `mm_features` 列表。

### 14.3 构造 MultiModalFeatureSpec

每个多模态 item 会变成一个 `MultiModalFeatureSpec`：

```python
mm_features.append(
    MultiModalFeatureSpec(
        data=decoder_mm_inputs[modality][idx],
        modality=modality,
        identifier=self._get_mm_identifier(
            base_mm_hash,
            lora_request,
        ),
        mm_position=decoder_mm_positions[modality][idx],
        mm_hash=base_mm_hash,
    )
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:357` 到 `vllm/vllm/v1/engine/input_processor.py:367`

字段含义可以理解为：

| 字段 | 含义 |
|---|---|
| `data` | 多模态实际数据或预处理结果 |
| `modality` | image / audio / video 等模态名 |
| `identifier` | 用于多模态缓存识别的 id |
| `mm_position` | 多模态 placeholder 在 token 序列中的位置 |
| `mm_hash` | 原始多模态 hash |

### 14.4 LoRA 会影响多模态 identifier

`identifier` 来自：

```python
self._get_mm_identifier(base_mm_hash, lora_request)
```

位置：`vllm/vllm/v1/engine/input_processor.py:361` 到 `vllm/vllm/v1/engine/input_processor.py:364`

如果启用了 `enable_tower_connector_lora`，多模态 embedding 会随 LoRA 改变，所以 identifier 要带上 LoRA 名：

```python
if (
    lora_request is None
    or self.lora_config is None
    or not self.lora_config.enable_tower_connector_lora
):
    return mm_hash
return f"{lora_request.lora_name}:{mm_hash}"
```

位置：`vllm/vllm/v1/engine/input_processor.py:175` 到 `vllm/vllm/v1/engine/input_processor.py:181`

含义是：

```text
同一个图片 / 多模态输入，在不同 LoRA 下可能产生不同 embedding；
因此 cache key 不能只看 mm_hash，还要区分 LoRA。
```

---

## 15. 最后一步：构造 EngineCoreRequest

`process_inputs()` 最后返回：

```python
return EngineCoreRequest(
    request_id=request_id,
    prompt_token_ids=prompt_token_ids,
    prompt_embeds=prompt_embeds,
    prompt_is_token_ids=prompt_is_token_ids,
    mm_features=mm_features,
    sampling_params=sampling_params,
    pooling_params=pooling_params,
    arrival_time=arrival_time,
    lora_request=lora_request,
    cache_salt=decoder_inputs.get("cache_salt"),
    priority=priority,
    data_parallel_rank=data_parallel_rank,
    trace_headers=trace_headers,
    resumable=resumable,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:370` 到 `vllm/vllm/v1/engine/input_processor.py:385`

这就是 `InputProcessor` 的核心产物。

它不会创建 Scheduler 内部的 `Request`。

它只创建跨 Engine / EngineCore 边界的：

```text
EngineCoreRequest
```

---

## 16. EngineCoreRequest 包含什么

`EngineCoreRequest` 定义在：

```python
class EngineCoreRequest(msgspec.Struct, ...):
```

位置：`vllm/vllm/v1/engine/__init__.py:88` 到 `vllm/vllm/v1/engine/__init__.py:93`

主要字段包括：

```python
request_id: str
prompt_token_ids: list[int] | None
mm_features: list[MultiModalFeatureSpec] | None
sampling_params: SamplingParams | None
pooling_params: PoolingParams | None
arrival_time: float
lora_request: LoRARequest | None
cache_salt: str | None
data_parallel_rank: int | None
prompt_embeds: torch.Tensor | None = None
prompt_is_token_ids: list[bool] | None = None
client_index: int = 0
current_wave: int = 0
priority: int = 0
trace_headers: Mapping[str, str] | None = None
resumable: bool = False
external_req_id: str | None = None
reasoning_ended: bool | None = None
reasoning_parser_kwargs: dict[str, Any] | None = None
abort_immediately: bool = False
```

位置：`vllm/vllm/v1/engine/__init__.py:94` 到 `vllm/vllm/v1/engine/__init__.py:137`

可以分成几类：

```text
输入内容：
  prompt_token_ids / prompt_embeds / prompt_is_token_ids / mm_features

请求参数：
  sampling_params / pooling_params

请求元信息：
  request_id / external_req_id / arrival_time / priority / trace_headers

并行和分布式：
  data_parallel_rank / current_wave / client_index

扩展能力：
  lora_request / cache_salt / resumable / reasoning_* / abort_immediately
```

---

## 17. assign_request_id：把外部 id 转成内部 id

`InputProcessor` 还有一个重要静态方法：

```python
@staticmethod
def assign_request_id(request: EngineCoreRequest):
```

位置：`vllm/vllm/v1/engine/input_processor.py:222` 到 `vllm/vllm/v1/engine/input_processor.py:223`

注释说明：

```python
"""Replace the externally supplied request ID with an internal request ID
that adds 8 random characters in order to ensure uniqueness.
"""
```

位置：`vllm/vllm/v1/engine/input_processor.py:223` 到 `vllm/vllm/v1/engine/input_processor.py:226`

逻辑是：

```python
if request.external_req_id is not None:
    raise ValueError(...)
request.external_req_id = request.request_id
if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
    logger.warning_once(...)
else:
    request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

位置：`vllm/vllm/v1/engine/input_processor.py:227` 到 `vllm/vllm/v1/engine/input_processor.py:240`

所以外部 id 和内部 id 的关系是：

```text
调用方传入：
  request_id = 用户 id

assign_request_id() 后：
  external_req_id = 用户 id
  request_id = 用户 id + 随机后缀
```

这样做的原因是：

```text
内部 Scheduler / EngineCore 需要唯一 request_id；
用户可能重复传入相同 id；
OutputProcessor 和 abort 仍然需要保留用户可见的 external_req_id。
```

注意：

```text
process_inputs() 本身不会自动调用 assign_request_id()；
LLMEngine / AsyncLLM 在 process_inputs() 后调用它。
```

同步路径位置：`vllm/vllm/v1/engine/llm_engine.py:263`

异步路径位置：`vllm/vllm/v1/engine/async_llm.py:368`

---

## 18. InputProcessor 和 LLMEngine.add_request() 的关系

同步 `LLMEngine.add_request()` 中，如果用户传入的不是 `EngineCoreRequest`，会调用：

```python
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    supported_tasks=self.get_supported_tasks(),
    arrival_time=arrival_time,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    trace_headers=trace_headers,
    priority=priority,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250` 到 `vllm/vllm/v1/engine/llm_engine.py:260`

之后：

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:263`

然后才会：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:274` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

同步请求入口主线是：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
```

这说明：

```text
InputProcessor 只负责输入转换；
请求真正进入 EngineCore 是 EngineCoreClient.add_request() 完成的。
```

---

## 19. InputProcessor 和 AsyncLLM.add_request() 的关系

异步普通请求中，逻辑也类似：

```python
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    supported_tasks=await self.get_supported_tasks(),
    arrival_time=arrival_time,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    trace_headers=trace_headers,
    priority=priority,
    data_parallel_rank=data_parallel_rank,
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:360`

异步路径额外可能设置 reasoning 字段：

```python
if reasoning_ended is not None:
    request.reasoning_ended = reasoning_ended
if reasoning_parser_kwargs is not None:
    request.reasoning_parser_kwargs = reasoning_parser_kwargs
```

位置：`vllm/vllm/v1/engine/async_llm.py:363` 到 `vllm/vllm/v1/engine/async_llm.py:366`

然后同样调用：

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:368`

最后由 `_add_request()` 注册输出状态并异步送入 EngineCore：

```python
self.output_processor.add_request(request, prompt, parent_req, index, queue)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:408` 到 `vllm/vllm/v1/engine/async_llm.py:412`

所以同步和异步普通请求的输入处理一致：

```text
都先走 InputProcessor.process_inputs()；
都得到 EngineCoreRequest；
都由外层 Engine 调 assign_request_id()；
差异在后续 output collector 和 EngineCoreClient 调用方式。
```

---

## 20. streaming input 中 InputProcessor 如何使用

异步 streaming input 是一个特殊路径。

入口在 `AsyncLLM.add_request()`：

```python
if isinstance(prompt, AsyncGenerator):
    ...
    return await self._add_streaming_input_request(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:316` 到 `vllm/vllm/v1/engine/async_llm.py:331`

### 20.1 先创建 final request

`_add_streaming_input_request()` 会先创建一个 final request，用于输入流结束信号：

```python
final_req = self.input_processor.process_inputs(
    request_id=request_id,
    prompt=TokensPrompt(prompt_token_ids=[0]),
    params=sampling_params,
    **inputs,
)
self.input_processor.assign_request_id(final_req)
internal_req_id = final_req.request_id
```

位置：`vllm/vllm/v1/engine/async_llm.py:447` 到 `vllm/vllm/v1/engine/async_llm.py:454`

这一步得到一个内部 request id，用于整个 streaming session。

### 20.2 每个 input chunk 都重新 process_inputs

每收到一个 chunk：

```python
req = self.input_processor.process_inputs(
    request_id=internal_req_id,
    prompt=input_chunk.prompt,
    params=sp,
    resumable=True,
    **inputs,
)
req.external_req_id = request_id
```

位置：`vllm/vllm/v1/engine/async_llm.py:468` 到 `vllm/vllm/v1/engine/async_llm.py:475`

关键点：

```text
每个 chunk 都会经过 InputProcessor.process_inputs()；
request_id 使用同一个 internal_req_id；
resumable=True；
external_req_id 手动设置成用户原始 request_id。
```

如果 chunk 使用 prompt embeds，会报错：

```python
if req.prompt_embeds is not None:
    raise ValueError(
        "prompt_embeds not supported for streaming inputs"
    )
```

位置：`vllm/vllm/v1/engine/async_llm.py:476` 到 `vllm/vllm/v1/engine/async_llm.py:479`

### 20.3 输入流结束时发送 final request

输入流结束时：

```python
await self._add_request(final_req, None, None, 0, queue)
```

位置：`vllm/vllm/v1/engine/async_llm.py:491` 到 `vllm/vllm/v1/engine/async_llm.py:495`

所以 streaming input 的主线是：

```text
AsyncGenerator input
  → final_req = process_inputs(dummy token)
  → assign_request_id(final_req)
  → internal_req_id
  → 每个 chunk:
       process_inputs(request_id=internal_req_id, resumable=True)
       req.external_req_id = 用户 request_id
       EngineCoreClient.add_request_async(req)
  → 输入结束:
       发送 final_req
```

---

## 21. InputProcessor 和 EngineCore.preprocess_add_request() 的边界

`InputProcessor` 的输出是 `EngineCoreRequest`。

真正转换成 Scheduler 内部 `Request` 的地方是 EngineCore：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:867`

`Request.from_engine_core_request()` 会把 `EngineCoreRequest` 字段复制进内部 `Request`：

```python
return cls(
    request_id=request.request_id,
    client_index=request.client_index,
    prompt_token_ids=request.prompt_token_ids,
    prompt_embeds=request.prompt_embeds,
    prompt_is_token_ids=request.prompt_is_token_ids,
    mm_features=request.mm_features,
    sampling_params=request.sampling_params,
    pooling_params=request.pooling_params,
    arrival_time=request.arrival_time,
    lora_request=request.lora_request,
    cache_salt=request.cache_salt,
    priority=request.priority,
    trace_headers=request.trace_headers,
    block_hasher=block_hasher,
    resumable=request.resumable,
    reasoning_ended=request.reasoning_ended,
    reasoning_parser_kwargs=request.reasoning_parser_kwargs,
    abort_immediately=request.abort_immediately,
)
```

位置：`vllm/vllm/v1/request.py:203` 到 `vllm/vllm/v1/request.py:222`

注意：内部 `Request` 会再初始化很多调度状态，例如：

```text
Request.status
structured_output_request
kv_transfer_params
num_prompt_tokens
output_token_ids
num_output_placeholders
spec_token_ids
num_computed_tokens
block_hashes
skip_reading_prefix_cache
streaming_queue
abort_immediately
```

相关位置：`vllm/vllm/v1/request.py:81` 到 `vllm/vllm/v1/request.py:195`

所以边界很清楚：

```text
InputProcessor：
  外部输入 → EngineCoreRequest

EngineCore.preprocess_add_request：
  EngineCoreRequest → Request

Scheduler：
  Request → waiting / running / skipped_waiting 状态机
```

---

## 22. Structured output 在 InputProcessor 中做了什么

`InputProcessor` 本身不编译 grammar，也不推进 grammar。

它只在 `SamplingParams.verify()` 时参与 structured output 参数校验：

```python
params.verify(
    self.model_config,
    self.speculative_config,
    self.structured_outputs_config,
    self.tokenizer,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:95` 到 `vllm/vllm/v1/engine/input_processor.py:100`

真正的 structured output request 创建发生在内部 `Request` 初始化：

```python
self.structured_output_request = StructuredOutputRequest.from_sampling_params(
    sampling_params
)
```

位置：`vllm/vllm/v1/request.py:87` 到 `vllm/vllm/v1/request.py:89`

如果有 structured output request，会设置初始状态：

```python
if self.structured_output_request is not None:
    self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

位置：`vllm/vllm/v1/request.py:111` 到 `vllm/vllm/v1/request.py:112`

grammar 初始化发生在 EngineCore：

```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
```

位置：`vllm/vllm/v1/engine/core.py:868` 到 `vllm/vllm/v1/engine/core.py:874`

因此：

```text
InputProcessor 只校验 structured output 参数；
Request 初始化创建 structured_output_request；
EngineCore 初始化 grammar；
Scheduler 在 grammar ready 前把请求放入 skipped_waiting。
```

---

## 23. prompt_embeds / mixed-mode 输入的处理边界

`InputProcessor` 支持 decoder 输入是 embeds：

```python
prompt_embeds = decoder_inputs["prompt_embeds"]
prompt_token_ids = decoder_inputs.get("prompt_token_ids")
prompt_is_token_ids = decoder_inputs.get("is_token_ids")
```

位置：`vllm/vllm/v1/engine/input_processor.py:302` 到 `vllm/vllm/v1/engine/input_processor.py:305`

这些字段进入 `EngineCoreRequest` 后，会继续传入内部 `Request`：

```python
prompt_embeds=request.prompt_embeds,
prompt_is_token_ids=request.prompt_is_token_ids,
```

位置：`vllm/vllm/v1/request.py:207` 到 `vllm/vllm/v1/request.py:208`

内部 `Request` 注释说明 `prompt_is_token_ids` 用于 mixed-mode：

```python
# Per-position mask used in mixed-mode (chat completion with
# prompt_embeds). `None` except when both `prompt_token_ids` and
# `prompt_embeds` are set and their positions are interleaved.
```

位置：`vllm/vllm/v1/request.py:123` 到 `vllm/vllm/v1/request.py:126`

所以 InputProcessor 的职责是：

```text
识别输入是 token ids、embeds，还是 mixed-mode；
把相关字段放入 EngineCoreRequest；
不负责后续 block hash / KV cache / model runner 如何使用 embeds。
```

---

## 24. 多模态 cache 注入：inject_into_mm_cache()

除了 `process_inputs()`，InputProcessor 还有一个多模态 cache 辅助方法：

```python
def inject_into_mm_cache(
    self,
    mm_hashes: dict[str, list[str]],
    mm_kwargs: dict[str, list],
) -> None:
```

位置：`vllm/vllm/v1/engine/input_processor.py:183` 到 `vllm/vllm/v1/engine/input_processor.py:187`

注释说明它用于：

```text
当 mm_kwargs 已经被外部 HF processor 处理过时，
把这些预处理结果注入 renderer 的 mm_processor_cache，
避免后续相同图片 / 多模态输入重复处理，
并保证 cache hit rate 指标准确。
```

核心逻辑是：

```python
cache = self.renderer.mm_processor_cache
if cache is None:
    return
...
items[i], _ = cache.get_and_update_item(
    (items[i], []),
    mm_hash,
)
...
self.renderer.update_mm_cache_stats()
```

位置：`vllm/vllm/v1/engine/input_processor.py:199` 到 `vllm/vllm/v1/engine/input_processor.py:215`

这不是普通请求的主路径，但说明 `InputProcessor` 也承担了外层多模态输入缓存的一部分维护工作。

---

## 25. InputProcessor 不负责什么

`InputProcessor` 只负责输入侧规范化，不负责以下事情。

### 25.1 不负责输出 detokenize

输出 token 转文本发生在 `OutputProcessor`：

```text
EngineCoreOutput.new_token_ids
  → OutputProcessor.process_outputs()
  → IncrementalDetokenizer.update()
  → RequestOutput.text
```

`InputProcessor` 只负责输入，不处理输出。

### 25.2 不负责 Scheduler 调度

它不会维护：

```text
waiting
running
skipped_waiting
token_budget
KV block allocation
preemption
```

这些属于 Scheduler。

### 25.3 不负责 Worker forward

它不会调用模型，也不会调用 `model_executor.execute_model()`。

模型执行发生在：

```text
EngineCore.step()
  → model_executor.execute_model()
  → Worker / ModelRunner
```

### 25.4 不负责 EngineCoreRequest → Request

这个转换发生在：

```text
EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
```

不是 `InputProcessor`。

### 25.5 不负责 prefix cache / block hash 查询

`InputProcessor` 会把 `cache_salt` 放进 `EngineCoreRequest`，但不会计算 prefix cache 命中。

block hashes 和 prefix cache 相关逻辑发生在内部 `Request` / Scheduler：

```text
Request.update_block_hashes()
Scheduler waiting 调度阶段查询 prefix cache / external KV cache
```

### 25.6 不负责 structured output grammar 编译和推进

它只参与参数校验。

真正 grammar 初始化和推进发生在：

```text
EngineCore.structured_output_manager.grammar_init()
Scheduler.get_grammar_bitmask()
Scheduler.update_from_output() 中 grammar.accept_tokens()
```

---

## 26. 输入对象转换完整流程图

```text
用户请求
  │
  ├─ request_id
  ├─ prompt: PromptType | EngineInput
  ├─ params: SamplingParams | PoolingParams
  ├─ lora_request
  ├─ trace_headers
  ├─ priority
  ├─ data_parallel_rank
  └─ resumable

InputProcessor.process_inputs()
  │
  ├─ _validate_params()
  │    ├─ SamplingParams.verify()
  │    └─ PoolingParams.verify()
  │
  ├─ _validate_lora()
  │
  ├─ validate data_parallel_rank
  │
  ├─ prompt already EngineInput ?
  │    ├─ yes: processed_inputs = prompt
  │    └─ no: InputPreprocessor.preprocess(prompt)
  │
  ├─ current_platform.validate_request()
  │
  ├─ split_enc_dec_input()
  │    ├─ encoder_inputs
  │    └─ decoder_inputs
  │
  ├─ _validate_model_inputs()
  │    ├─ prompt length
  │    ├─ multimodal encoder cache size
  │    └─ token id vocab range
  │
  ├─ extract decoder prompt
  │    ├─ prompt_token_ids
  │    ├─ prompt_embeds
  │    └─ prompt_is_token_ids
  │
  ├─ clone params
  │    ├─ SamplingParams: max_tokens / generation_config / tokenizer
  │    └─ PoolingParams: clone
  │
  ├─ multimodal ?
  │    └─ build mm_features sorted by position
  │
  └─ EngineCoreRequest
```

---

## 27. EngineCoreRequest 后续流向图

`InputProcessor` 结束后，请求并没有进入 Scheduler。

后续主线是：

```text
EngineCoreRequest
  │
  ├─ LLMEngine / AsyncLLM
  │    ├─ InputProcessor.assign_request_id()
  │    ├─ OutputProcessor.add_request()
  │    └─ EngineCoreClient.add_request()
  │
  ├─ EngineCoreClient
  │    ├─ InprocClient: 直接调用 EngineCore.preprocess_add_request()
  │    └─ MPClient: 序列化发送 ADD 消息
  │
  ├─ EngineCore.preprocess_add_request()
  │    ├─ mm_receiver_cache 处理
  │    ├─ Request.from_engine_core_request()
  │    └─ structured output grammar_init()
  │
  ├─ EngineCore.add_request()
  │
  └─ Scheduler.add_request()
       ├─ self.requests[request_id] = Request
       ├─ self.waiting
       └─ self.skipped_waiting
```

所以关键对象边界是：

```text
用户输入
  → EngineInput
  → EngineCoreRequest
  → Request
  → Scheduler 队列状态
```

---

## 28. 一个完整例子：普通文本 generation 请求

假设用户传入：

```text
request_id = "req-a"
prompt = "hello"
params = SamplingParams(max_tokens=None)
lora_request = None
```

流程是：

```text
1. _validate_params() 确认模型支持 generation；
2. SamplingParams.verify() 校验参数；
3. _validate_lora(None) 直接返回；
4. prompt 不是 EngineInput，走 InputPreprocessor.preprocess()；
5. 得到 decoder_inputs["prompt_token_ids"]；
6. 校验 prompt length 和 token id 范围；
7. clone SamplingParams；
8. 因为 max_tokens=None，补成 max_model_len - prompt_len；
9. 用 generation config 和 tokenizer 更新 SamplingParams；
10. 无多模态，mm_features=None；
11. 返回 EngineCoreRequest。
```

返回对象大致包含：

```text
EngineCoreRequest(
  request_id="req-a",
  prompt_token_ids=[...],
  prompt_embeds=None,
  mm_features=None,
  sampling_params=cloned_and_updated_params,
  pooling_params=None,
  arrival_time=...,
  priority=0,
)
```

随后外层 Engine 调用：

```text
assign_request_id()
  → external_req_id="req-a"
  → request_id="req-a-xxxxxxxx"
```

---

## 29. 一个完整例子：Pooling 请求

假设用户传入：

```text
params = PoolingParams(task=None)
```

`InputProcessor` 会：

```text
1. 检查模型是否支持 POOLING_TASKS；
2. 如果 task=None，按支持列表自动选择 token_embed / token_classify / plugin；
3. 检查 task 是否在 supported_pooling_tasks；
4. params.verify(model_config)；
5. clone PoolingParams；
6. 构造 EngineCoreRequest(pooling_params=cloned_params, sampling_params=None)。
```

后续内部 `Request` 初始化时：

```python
if pooling_params is not None:
    # Pooling models.
    self.max_tokens = 1
```

位置：`vllm/vllm/v1/request.py:104` 到 `vllm/vllm/v1/request.py:106`

这说明 pooling 请求不是自回归生成 token，而是执行一次 pooling 相关输出。

---

## 30. 一个完整例子：多模态请求

假设 decoder input 是 multimodal，并包含两个图片 placeholder。

InputProcessor 会：

```text
1. 从 decoder_inputs 取 mm_kwargs / mm_placeholders / mm_hashes；
2. 校验所有 mm_hash leaf 都是字符串；
3. 用 argsort_mm_positions() 按输入序列位置排序；
4. 每个多模态 item 构造成 MultiModalFeatureSpec；
5. 如果 LoRA 会影响 tower connector，多模态 identifier 加上 lora_name；
6. 把 mm_features 放入 EngineCoreRequest。
```

结果是：

```text
EngineCoreRequest.mm_features = [
  MultiModalFeatureSpec(... position 更靠前 ...),
  MultiModalFeatureSpec(... position 更靠后 ...),
]
```

后续 EngineCore / Scheduler / Worker 会基于这些 `mm_features` 处理 encoder input、encoder cache 和多模态 embedding。

---

## 31. 容易疑惑的点

### 31.1 InputProcessor 输出的是 Request 吗？

不是。

```text
InputProcessor 输出 EngineCoreRequest；
EngineCore.preprocess_add_request() 才把 EngineCoreRequest 转成内部 Request。
```

转换位置：`vllm/vllm/v1/request.py:198` 到 `vllm/vllm/v1/request.py:222`

### 31.2 InputProcessor 会不会直接进入 Scheduler？

不会。

主线是：

```text
InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Scheduler.add_request()
```

### 31.3 process_inputs() 会不会自动随机化 request_id？

不会。

`process_inputs()` 返回的 `EngineCoreRequest.request_id` 仍是传入的 request_id。

随机化发生在外层 Engine 调用：

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:263`、`vllm/vllm/v1/engine/async_llm.py:368`

### 31.4 为什么要 clone SamplingParams / PoolingParams？

因为 `InputProcessor` 会补全或更新 params，例如：

```text
SamplingParams.max_tokens；
generation_config 字段；
tokenizer 相关 EOS / stop 信息；
PoolingParams.task 默认值。
```

clone 后可以避免直接修改调用方传入的原始对象。

### 31.5 InputProcessor 是否负责 stop string？

不负责输出侧 stop string 检查。

它只在 `SamplingParams.verify()` 和 tokenizer 更新阶段处理参数合法性。

真正根据生成文本检查 stop string 的地方是 `OutputProcessor` 的 detokenizer。

### 31.6 InputProcessor 是否处理 structured output grammar？

不直接处理 grammar 编译。

它只把 structured output 相关参数交给 `SamplingParams.verify()` 校验。

后续：

```text
Request.from_engine_core_request()
  → StructuredOutputRequest.from_sampling_params()
EngineCore.preprocess_add_request()
  → grammar_init()
Scheduler
  → 等 grammar ready 后再调度
```

### 31.7 EngineInput 和 raw prompt 有什么区别？

`EngineInput` 已经是 renderer / preprocessor 处理过的结构，包含 `type` 字段。

raw prompt 需要：

```text
InputPreprocessor.preprocess()
```

源码提示 raw prompt 直接传给 InputProcessor 的路径会被废弃，推荐使用：

```text
Renderer.render_cmpl()
Renderer.render_chat()
```

### 31.8 多模态 mm_features 的顺序重要吗？

重要。

InputProcessor 会用：

```python
argsort_mm_positions(decoder_mm_positions)
```

位置：`vllm/vllm/v1/engine/input_processor.py:352`

把多模态项按它们在输入序列中的位置排序，保证后续 encoder / decoder 对齐。

### 31.9 InputProcessor 校验 prompt 长度时看哪个上限？

decoder 输入看：

```text
model_config.max_model_len
```

encoder 输入看：

```text
mm_encoder_cache_size
```

对应代码：`vllm/vllm/v1/engine/input_processor.py:398` 到 `vllm/vllm/v1/engine/input_processor.py:403`

### 31.10 streaming input 为什么每个 chunk 都 process_inputs？

因为每个 chunk 也是一段新的输入，需要做相同的参数、prompt、多模态、平台和长度校验。

但它们共享同一个 internal request id：

```text
request_id=internal_req_id
resumable=True
```

这样 Scheduler 能把它识别为同一个 streaming session 的后续输入。

---

## 32. 从“回答问题”的角度总结

如果要问：

```text
InputProcessor 如何把用户输入转成 EngineCoreRequest？
```

可以回答：

```text
InputProcessor 是 LLMEngine / AsyncLLM 持有的输入预处理组件。
它先校验 SamplingParams / PoolingParams、LoRA 和 data_parallel_rank，
再把 raw prompt 或 EngineInput 统一成 processed_inputs，
通过 split_enc_dec_input() 拆分 encoder / decoder 输入，
校验 prompt 长度、token id 范围和多模态 encoder cache size，
然后提取 prompt_token_ids / prompt_embeds / prompt_is_token_ids，
clone 并补全 SamplingParams 或 PoolingParams，
最后把多模态输入整理成按序列位置排序的 mm_features，
构造 EngineCoreRequest 返回给外层 Engine。
```

核心公式是：

```text
process_inputs()
  = validate(params, lora, dp_rank)
  + normalize(prompt → EngineInput)
  + validate(model inputs)
  + clone/update params
  + collect mm_features
  + EngineCoreRequest
```

后续外层 Engine 会：

```text
InputProcessor.assign_request_id()
  → external_req_id / internal request_id
OutputProcessor.add_request()
  → 注册输出状态
EngineCoreClient.add_request()
  → 送入 EngineCore
```

---

## 33. 最关键流程图

```text
InputProcessor.process_inputs()

_validate_params(params, supported_tasks)
  ├─ SamplingParams:
  │    ├─ model supports generation?
  │    ├─ params.verify(model/spec/structured/tokenizer)
  │    └─ thinking_token_budget checks
  └─ PoolingParams:
       ├─ model supports pooling?
       ├─ fill default task
       └─ params.verify(model_config)

_validate_lora(lora_request)
  └─ LoRA request requires lora_config

validate data_parallel_rank

normalize prompt
  ├─ EngineInput: use directly
  └─ raw PromptType: InputPreprocessor.preprocess()

current_platform.validate_request(processed_inputs, params)

split_enc_dec_input(processed_inputs)
  ├─ encoder_inputs
  └─ decoder_inputs

_validate_model_inputs()
  ├─ prompt length
  ├─ multimodal encoder cache size
  └─ token id vocab range

extract decoder prompt
  ├─ prompt_token_ids
  ├─ prompt_embeds
  └─ prompt_is_token_ids

clone/update params
  ├─ SamplingParams.clone()
  │    ├─ fill max_tokens
  │    ├─ update_from_generation_config()
  │    └─ update_from_tokenizer()
  └─ PoolingParams.clone()

multimodal input
  ├─ validate mm_hashes
  ├─ argsort_mm_positions()
  └─ MultiModalFeatureSpec(...)

return EngineCoreRequest(...)
```

```text
外层后续：

EngineCoreRequest
  → assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
```

---

## 34. 和其它文档的关系

`02_llm_engine_sync.md` 解释的是：

```text
同步 LLMEngine 如何作为外层 Engine 连接 InputProcessor、EngineCoreClient 和 OutputProcessor。
```

本篇聚焦其中的输入侧：

```text
InputProcessor.process_inputs()
```

`engine_core/02_request_entry.md` 解释的是：

```text
EngineCoreRequest 如何进入 EngineCore，再变成 Scheduler 内部 Request。
```

本篇解释的是它前一段：

```text
用户输入如何先变成 EngineCoreRequest。
```

`scheduler/01_request_states.md` 解释的是：

```text
Request 进入 Scheduler 后如何处在 waiting / running / skipped_waiting 状态。
```

本篇强调：

```text
InputProcessor 不维护这些状态；
它只产生 EngineCoreRequest，状态机从 EngineCore 转成 Request 后才开始。
```
