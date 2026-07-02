# 02. 多模态用户输入如何进入 vLLM？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\entrypoints\openai\chat_completion\serving.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\entrypoints\openai\engine\serving.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\entrypoints\chat_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\entrypoints\offline_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\renderers\base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\renderers\hf.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\inputs\llm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\inputs\engine.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\inputs\preprocess.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\input_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\__init__.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\request.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core.py`

本问题关注：用户通过 OpenAI-compatible API、offline LLM 或已经渲染好的 `EngineInput` 提供 image / audio / video / prompt embeds 后，这些输入如何经过 chat parser、renderer、`InputProcessor`，最终变成 `EngineCoreRequest.prompt_token_ids`、`EngineCoreRequest.prompt_embeds`、`EngineCoreRequest.prompt_is_token_ids` 和 `EngineCoreRequest.mm_features`。

---

## 1. 一句话回答

多模态请求进入 vLLM V1 的核心路径是：

```text
外部请求
  → OpenAI entrypoint / offline LLM
  → chat parser 或 PromptType 标准化
  → Renderer.render_chat() / render_cmpl()
  → EngineInput
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → Request.from_engine_core_request()
  → Scheduler waiting queue
```

其中多模态数据分成两部分：

```text
文本侧：
  prompt / messages 被 chat template 和 tokenizer 转成 prompt_token_ids。
  多模态位置会在 token 序列中留下 placeholder token span。

数据侧：
  image / audio / video / prompt_embeds 被多模态 processor 转成 mm_kwargs、mm_hashes、mm_placeholders。
  InputProcessor 再把它们按 placeholder 位置排序，压成 mm_features。
```

所以一句话压缩：

```text
renderer 负责把用户输入渲染成 EngineInput；
InputProcessor 负责把 EngineInput 收敛成 EngineCoreRequest；
EngineCore / Scheduler 之后只看到 token 序列和按位置排列的 mm_features。
```

---

## 2. 最小主链路

以 OpenAI Chat Completions 为例，主链路是：

```text
ChatCompletionRequest.messages
  → OpenAIServingChat.create_chat_completion()
  → render_chat_request()
  → openai_serving_render.render_chat(request)
  → BaseRenderer.render_chat_async()
      → render_messages_async()
      → tokenize_prompts_async()
      → process_for_engine_async()
      → _process_multimodal_async() / _process_embeds()
  → engine_inputs: list[EngineInput]
  → engine_client.generate(engine_input, sampling_params, request_id, ...)
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → engine_core.add_request_async(request)
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → scheduler.add_request(request)
```

关键源码位置：

```text
chat_completion/serving.py:267
  result = await self.render_chat_request(request)

chat_completion/serving.py:358
  generator = self.engine_client.generate(engine_input, sampling_params, sub_request_id, ...)

async_llm.py:349
  request = self.input_processor.process_inputs(...)

input_processor.py:370
  return EngineCoreRequest(...)

async_llm.py:368
  self.input_processor.assign_request_id(request)

core.py:867
  req = Request.from_engine_core_request(request, self.request_block_hasher)

core.py:403
  self.scheduler.add_request(request)
```

这条链路里有两个重要分界点：

```text
EngineInput：
  已完成渲染和多模态预处理，是 engine client 可以接收的输入格式。

EngineCoreRequest：
  已完成参数校验、prompt 字段拆分和 mm_features 扁平化，是发给 EngineCore 的请求格式。
```

---

## 3. 外部输入有哪些形式

vLLM 允许请求从几个层级进入。

### 3.1 OpenAI Chat Completions

用户请求通常是：

```text
messages: [
  {
    role: "user",
    content: [
      {type: "text", text: "..."},
      {type: "image_url", image_url: ...},
      {type: "input_audio", input_audio: ...},
      {type: "video_url", video_url: ...},
      {type: "prompt_embeds", data: ...},
    ],
  }
]
```

content part 类型在 `chat_utils.py` 中解析。

支持的关键类型包括：

```text
text / input_text / output_text / refusal / thinking
image_url / input_image / image_pil / image_embeds
audio_url / input_audio / audio_embeds
video_url
prompt_embeds
tool_reference
```

对应源码：`chat_utils.py:1454` 到 `chat_utils.py:1558`、`chat_utils.py:1627` 到 `chat_utils.py:1735`

解析时发生两件事：

```text
1. 文本 part 直接进入 ConversationMessage.content；
2. 多模态 part 被 mm_parser 解析、下载或解码，并向文本里插入 placeholder。
```

例如：

```python
elif part_type in ("image_url", "input_image"):
    mm_parser.parse_image(str_content, uuid)
    modality = "image"
elif part_type == "input_audio":
    mm_parser.parse_input_audio(dict_content, uuid)
    modality = "audio"
elif part_type == "video_url":
    mm_parser.parse_video(str_content, uuid)
    modality = "video"
elif part_type == "prompt_embeds":
    mm_parser.parse_prompt_embeds(cast(str, content))
    modality = "prompt_embeds"
```

位置：`chat_utils.py:1678` 到 `chat_utils.py:1706`

最终 `parse_chat_messages()` 返回：

```text
conversation：给 chat template 使用的消息结构；
mm_data：按 modality 聚合后的真实多模态数据；
mm_uuids：每个多模态 item 的 uuid / hash 标识来源。
```

位置：`chat_utils.py:1863` 到 `chat_utils.py:1899`

### 3.2 OpenAI Completion / Embeddings / Pooling

Completion 请求的 `prompt` 可以是文本、token ids，也可以携带 `prompt_embeds`。

`CompletionRequest` 中有：

```python
prompt_embeds: bytes | list[bytes] | None = None
```

位置：`completion/protocol.py:98`

对 pooling / embeddings 类入口来说，最终也会走到同一类 `EngineInput` / `EngineCoreRequest`，只是 `params` 是 `PoolingParams`，`EngineCoreRequest.pooling_params` 非空，`sampling_params` 为空。

### 3.3 Offline LLM

offline LLM 侧不是 HTTP request，而是 Python API 直接传入 `PromptType` 或已经渲染好的 `EngineInput`。

入口在 `offline_utils.py`：

```text
_preprocess_cmpl_one() / _preprocess_chat_one()
  → _render_and_add_requests()
  → _add_request()
  → llm_engine.add_request(request_id, prompt, params, ...)
```

位置：`offline_utils.py:523` 到 `offline_utils.py:565`

`PromptType` 定义在 `inputs/llm.py`，可以是：

```text
str / TextPrompt
list[int] / TokensPrompt
EmbedsPrompt
EncoderDecoderPrompt
```

位置：`inputs/llm.py:140` 到 `inputs/llm.py:215`

多模态离线输入一般通过 prompt dict 携带：

```python
{
    "prompt": "...",
    "multi_modal_data": {"image": image, "audio": audio, "video": video},
    "mm_processor_kwargs": {...},
    "multi_modal_uuids": {...},
}
```

这些字段来自 `_PromptOptions`：

```python
multi_modal_data: NotRequired[MultiModalDataDict | None]
mm_processor_kwargs: NotRequired[dict[str, Any] | None]
multi_modal_uuids: NotRequired[MultiModalUUIDDict]
cache_salt: NotRequired[str]
```

位置：`inputs/llm.py:64` 到 `inputs/llm.py:96`

### 3.4 已渲染 EngineInput

更底层的调用方可以直接传 `EngineInput`。

`InputProcessor.process_inputs()` 判断：

```python
if isinstance(prompt, dict) and "type" in prompt:
    processed_inputs: EngineInput = prompt
else:
    processed_inputs = self.input_preprocessor.preprocess(prompt, ...)
```

位置：`input_processor.py:269` 到 `input_processor.py:294`

也就是说：

```text
带 type 字段的 dict 被认为已经是 EngineInput；
原始 PromptType 才会经过 InputPreprocessor.preprocess()。
```

---

## 4. PromptType 和 EngineInput 的区别

`PromptType` 是用户 API 层输入。

定义在：`inputs/llm.py:215`

```python
PromptType: TypeAlias = DecoderOnlyPrompt | EncoderDecoderPrompt
```

它能表达：

```text
文本 prompt；
已 tokenized prompt；
prompt_embeds；
encoder-decoder prompt；
附带 multi_modal_data 的 prompt dict。
```

`EngineInput` 是 renderer / preprocessor 后的 engine 输入。

定义在：`inputs/engine.py:264`

```python
EngineInput: TypeAlias = DecoderOnlyEngineInput | EncoderDecoderInput
```

核心子类型有：

```text
TokensInput:
  type="token"
  prompt_token_ids

EmbedsInput:
  type="embeds"
  prompt_embeds
  prompt_token_ids?      # mixed-mode 时存在
  is_token_ids?          # mixed-mode mask

MultiModalInput:
  type="multimodal"
  prompt_token_ids
  mm_kwargs
  mm_hashes
  mm_placeholders

EncoderDecoderInput:
  type="enc_dec"
  encoder_prompt
  decoder_prompt
```

位置：`inputs/engine.py:29` 到 `inputs/engine.py:264`

因此可以把输入演进理解为：

```text
PromptType：用户说了什么，可能还是文本 / URL / PIL / bytes / tensor。
EngineInput：renderer 已经把它转成 token ids、processor kwargs、placeholder ranges。
EngineCoreRequest：V1 engine core 可以排队和调度的请求对象。
```

---

## 5. OpenAI Chat 如何把 content part 变成 mm_data

Chat parser 的中心对象是 `BaseMultiModalItemTracker` 和 `BaseMultiModalContentParser`。

`BaseMultiModalItemTracker.add()` 做三件事：

```text
1. 校验每种 modality 的数量是否超过 limit-mm-per-prompt；
2. 把 item 暂存到 _items_by_modality；
3. 返回模型专用 placeholder string。
```

位置：`chat_utils.py:586` 到 `chat_utils.py:651`

例如 image：

```python
def parse_image(self, image_url: str | None, uuid: str | None = None) -> None:
    image = self._connector.fetch_image(image_url) if image_url else None
    placeholder = self._tracker.add("image", (image, uuid))
    self._add_placeholder("image", placeholder)
```

位置：`chat_utils.py:955` 到 `chat_utils.py:959`

audio：

```python
def parse_input_audio(self, input_audio: InputAudio | None, uuid: str | None = None) -> None:
    if input_audio:
        audio_data = input_audio.get("data", "")
        audio_format = input_audio.get("format", "")
        if audio_data:
            audio_url = f"data:audio/{audio_format};base64,{audio_data}"
        else:
            audio_url = None
    else:
        audio_url = None

    return self.parse_audio(audio_url, uuid)
```

位置：`chat_utils.py:1025` 到 `chat_utils.py:1039`

video：

```python
def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
    video = self._connector.fetch_video(...)
    placeholder = self._tracker.add("video", (video, uuid))
    self._add_placeholder("video", placeholder)
```

位置：`chat_utils.py:1041` 到 `chat_utils.py:1052`

`prompt_embeds` 比较特殊：

```python
tensor = safe_load_prompt_embeds(self.model_config, data.encode())
self._tracker.add("prompt_embeds", (tensor, None))
self._add_placeholder("prompt_embeds", PROMPT_EMBEDS_PLACEHOLDER_TOKEN)
```

位置：`chat_utils.py:941` 到 `chat_utils.py:953`

它不会走普通 HF 多模态 processor 的校验逻辑，而是把预计算 embedding tensor 当成一种特殊 modality 暂存。

最后 `resolve_items()` 把 tracker 中的 item 变成：

```text
mm_data: MultiModalDataDict | None
mm_uuids: MultiModalUUIDDict | None
```

位置：`chat_utils.py:792` 到 `chat_utils.py:814`

---

## 6. Renderer 如何生成 EngineInput

`BaseRenderer.render_chat()` / `render_chat_async()` 是 OpenAI Chat 到 `EngineInput` 的关键阶段。

同步版本主流程：

```python
rendered = [self.render_messages(conversation, chat_params) for conversation in conversations]
...
tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)
self._apply_prompt_extras(tok_prompts, prompt_extras)
eng_prompts = [
    self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
    for prompt in tok_prompts
]
return out_conversations, eng_prompts
```

位置：`renderers/base.py:978` 到 `renderers/base.py:1012`

异步版本同理：

```python
tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)
...
eng_prompts = await asyncio.gather(
    *(self.process_for_engine_async(p, arrival_time, ...) for p in tok_prompts)
)
```

位置：`renderers/base.py:1014` 到 `renderers/base.py:1052`

`process_for_engine()` 再分流：

```python
if "encoder_prompt" in prompt:
    engine_input = self._process_enc_dec(prompt, ...)
else:
    engine_input = self._process_singleton(prompt, ...)

engine_input["arrival_time"] = arrival_time
```

位置：`renderers/base.py:888` 到 `renderers/base.py:903`

`_process_singleton()` 再分流：

```python
if "prompt_embeds" in prompt:
    return self._process_embeds(prompt)
return self._process_tokens(prompt, skip_mm_cache=skip_mm_cache)
```

位置：`renderers/base.py:811` 到 `renderers/base.py:820`

所以 renderer 层有三条主要产物路径：

```text
纯文本 / 纯 token：
  TokensInput(type="token", prompt_token_ids=[...])

多模态：
  MultiModalInput(type="multimodal", prompt_token_ids=[...], mm_kwargs=..., mm_hashes=..., mm_placeholders=...)

纯 prompt_embeds：
  EmbedsInput(type="embeds", prompt_embeds=tensor, ...)
```

---

## 7. 多模态 processor 产出什么

`BaseRenderer._process_multimodal()` 是从 `multi_modal_data` 到 `MultiModalInput` 的核心函数。

源码：

```python
mm_data_items = mm_processor.info.parse_mm_data(mm_data)
mm_uuid_items = parse_mm_uuids(mm_uuids)
mm_uuid_items = self._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, mm_req_id)

mm_processor_inputs = MMProcessorInputs(
    prompt,
    mm_data_items,
    mm_uuid_items,
    hf_processor_mm_kwargs=mm_processor_kwargs or {},
    tokenization_kwargs=tokenization_kwargs or {},
)

mm_inputs = mm_processor.apply(mm_processor_inputs, mm_timing_ctx)
return mm_inputs
```

位置：`renderers/base.py:681` 到 `renderers/base.py:720`

返回的 `mm_inputs` 类型是 `MultiModalInput`，字段定义在 `inputs/engine.py`：

```python
class MultiModalInput(_InputOptions):
    type: Literal["multimodal"]
    prompt_token_ids: list[int]
    prompt: NotRequired[str]
    mm_kwargs: MultiModalKwargsOptionalItems
    mm_hashes: MultiModalHashes
    mm_placeholders: MultiModalPlaceholders
```

位置：`inputs/engine.py:126` 到 `inputs/engine.py:149`

这几个字段分别表示：

| 字段 | 含义 |
|---|---|
| `prompt_token_ids` | 已经包含多模态 placeholder token span 的 token 序列 |
| `mm_kwargs` | 后续模型 / encoder 需要的多模态张量或 processor 输出 |
| `mm_hashes` | 每个多模态 item 的 cache 标识 |
| `mm_placeholders` | 每个多模态 item 在 `prompt_token_ids` 中占据的 token 范围 |

这里最重要的是：

```text
placeholder 已经进入 prompt_token_ids；
真实多模态数据没有混进 token ids，而是保存在 mm_kwargs；
两者靠 mm_placeholders 建立位置关系。
```

---

## 8. InputProcessor 如何构造 EngineCoreRequest

`InputProcessor.process_inputs()` 是 `EngineInput` 到 `EngineCoreRequest` 的核心。

入口签名：

```python
def process_inputs(
    self,
    request_id: str,
    prompt: PromptType | EngineInput,
    params: SamplingParams | PoolingParams,
    supported_tasks: tuple[SupportedTask, ...],
    ...
) -> EngineCoreRequest:
```

位置：`input_processor.py:242` 到 `input_processor.py:255`

### 8.1 先判断输入是否已经渲染

```python
if isinstance(prompt, dict) and "type" in prompt:
    processed_inputs: EngineInput = prompt
else:
    processed_inputs = self.input_preprocessor.preprocess(prompt, ...)
```

位置：`input_processor.py:269` 到 `input_processor.py:294`

这解释了为什么 OpenAI serving 先 render，而 raw offline prompt 也能直接传：

```text
OpenAI path：通常传入已经 render 好的 EngineInput。
Raw PromptType path：InputProcessor 内部再用 InputPreprocessor 兜底 preprocess。
```

### 8.2 再拆 encoder / decoder

```python
encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
self._validate_model_inputs(encoder_inputs, decoder_inputs)
```

位置：`input_processor.py:298` 到 `input_processor.py:299`

`split_enc_dec_input()` 逻辑很简单：

```python
if inputs["type"] == "enc_dec":
    return inputs["encoder_prompt"], inputs["decoder_prompt"]
return None, inputs
```

位置：`inputs/engine.py:365` 到 `inputs/engine.py:371`

### 8.3 再抽取 token / embeds 字段

如果 decoder 是 pure embeds：

```python
if decoder_inputs["type"] == "embeds":
    prompt_embeds = decoder_inputs["prompt_embeds"]
    prompt_token_ids = decoder_inputs.get("prompt_token_ids")
    prompt_is_token_ids = decoder_inputs.get("is_token_ids")
else:
    prompt_token_ids = decoder_inputs["prompt_token_ids"]
    prompt_embeds = None
    prompt_is_token_ids = None
```

位置：`input_processor.py:301` 到 `input_processor.py:309`

因此：

```text
纯 token / multimodal：prompt_token_ids 一定来自 decoder_inputs。
纯 embeds：prompt_embeds 是主输入，prompt_token_ids 可以不存在。
mixed prompt_embeds：prompt_embeds、prompt_token_ids、prompt_is_token_ids 同时存在。
```

### 8.4 再处理 SamplingParams / PoolingParams

生成请求：

```python
sampling_params = params.clone()
if sampling_params.max_tokens is None:
    seq_len = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)
    sampling_params.max_tokens = self.model_config.max_model_len - seq_len
sampling_params.update_from_generation_config(...)
sampling_params.update_from_tokenizer(...)
```

位置：`input_processor.py:311` 到 `input_processor.py:328`

pooling 请求：

```python
pooling_params = params.clone()
```

位置：`input_processor.py:329` 到 `input_processor.py:330`

这说明多模态请求并不会绕过普通参数校验和 max token 计算，只是 prompt 长度可能来自 token ids 或 embeds。

---

## 9. mm_features 是如何生成的

如果 decoder 输入是 `MultiModalInput`：

```python
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]
    decoder_mm_positions = decoder_inputs["mm_placeholders"]
    decoder_mm_hashes = decoder_inputs["mm_hashes"]
```

位置：`input_processor.py:335` 到 `input_processor.py:338`

接着校验 `mm_hashes`：

```python
if not all(isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)):
    raise ValueError(...)
```

位置：`input_processor.py:340` 到 `input_processor.py:347`

然后按多模态 item 在 token 序列中的位置排序：

```python
sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)
```

位置：`input_processor.py:352`

最后构造 `MultiModalFeatureSpec`：

```python
mm_features = []
for modality, idx in sorted_mm_idxs:
    base_mm_hash = decoder_mm_hashes[modality][idx]
    mm_features.append(
        MultiModalFeatureSpec(
            data=decoder_mm_inputs[modality][idx],
            modality=modality,
            identifier=self._get_mm_identifier(base_mm_hash, lora_request),
            mm_position=decoder_mm_positions[modality][idx],
            mm_hash=base_mm_hash,
        )
    )
```

位置：`input_processor.py:354` 到 `input_processor.py:368`

这里的语义非常关键：

```text
mm_kwargs / mm_hashes / mm_placeholders 是按 modality 分组的 dict；
mm_features 是按 prompt 中出现顺序排列的 list；
每个 MultiModalFeatureSpec 同时携带 data、modality、identifier、mm_position、mm_hash。
```

也就是说，进入 EngineCore 后，多模态数据不再主要按 `image/audio/video` 字典组织，而是变成：

```text
按 prompt token 位置排序的一串 feature spec。
```

这对后面的 scheduler / model runner 很重要，因为它们需要知道每个多模态 encoder input 对应 prompt 的哪段 placeholder。

---

## 10. EngineCoreRequest 字段映射

`EngineCoreRequest` 定义在：`v1/engine/__init__.py:86`

核心字段：

```python
class EngineCoreRequest(msgspec.Struct, ...):
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

位置：`v1/engine/__init__.py:86` 到 `v1/engine/__init__.py:135`

`InputProcessor.process_inputs()` 构造它时的字段映射是：

| EngineCoreRequest 字段 | 来源 |
|---|---|
| `request_id` | OpenAI serving / offline LLM 生成的外部 request id |
| `prompt_token_ids` | `TokensInput` / `MultiModalInput` / mixed `EmbedsInput` |
| `prompt_embeds` | pure embeds 或 mixed prompt embeds |
| `prompt_is_token_ids` | mixed prompt embeds 的 per-position mask |
| `mm_features` | 从 `mm_kwargs + mm_hashes + mm_placeholders` 排序压平得到 |
| `sampling_params` | `SamplingParams.clone()` 后补齐 generation config/tokenizer 信息 |
| `pooling_params` | `PoolingParams.clone()` |
| `arrival_time` | renderer 设置或 process_inputs 当前时间 |
| `lora_request` | OpenAI model adapter / offline 参数 |
| `cache_salt` | prompt / EngineInput 中的 prefix cache salt |
| `priority` | OpenAI request priority 或调用参数 |
| `data_parallel_rank` | header 或调用参数 |
| `trace_headers` | HTTP tracing headers |
| `resumable` | streaming input / resumable 请求标记 |

构造位置：`input_processor.py:370` 到 `input_processor.py:385`

---

## 11. request_id 和 external_req_id

OpenAI serving 先生成用户可见的 request id，例如 chat：

```python
request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
```

位置：`chat_completion/serving.py:273` 到 `chat_completion/serving.py:275`

随后传给：

```python
self.engine_client.generate(engine_input, sampling_params, sub_request_id, ...)
```

位置：`chat_completion/serving.py:358` 到 `chat_completion/serving.py:366`

`InputProcessor.process_inputs()` 先把它放进 `EngineCoreRequest.request_id`。

之后 `AsyncLLM.add_request()` 调用：

```python
self.input_processor.assign_request_id(request)
```

位置：`async_llm.py:368`

`assign_request_id()` 会做：

```python
request.external_req_id = request.request_id
request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

位置：`input_processor.py:222` 到 `input_processor.py:240`

这说明：

```text
external_req_id：保存用户 / API 层 request id，用于输出和 external abort；
request_id：内部唯一 id，默认追加随机 8 字符，避免重复 id 干扰调度和缓存状态。
```

---

## 12. prompt_embeds 的两种路径

`prompt_embeds` 有两种不同语义。

### 12.1 纯 EmbedsPrompt

用户直接传：

```python
{"prompt_embeds": tensor}
```

`BaseRenderer._process_embeds()` 会校验：

```python
if not self.model_config.enable_prompt_embeds:
    raise ValueError("You must set `--enable-prompt-embeds` to input `prompt_embeds`.")
```

位置：`renderers/base.py:753` 到 `renderers/base.py:757`

然后把 tensor squeeze 到二维、搬到 CPU，并构造：

```python
return embeds_input(
    prompt_embeds=prompt_embeds,
    cache_salt=prompt.get("cache_salt"),
    prompt_token_ids=prompt.get("prompt_token_ids"),
    is_token_ids=prompt.get("prompt_is_token_ids"),
)
```

位置：`renderers/base.py:759` 到 `renderers/base.py:781`

进入 `InputProcessor` 后：

```text
prompt_embeds → EngineCoreRequest.prompt_embeds
prompt_token_ids? → EngineCoreRequest.prompt_token_ids
is_token_ids? → EngineCoreRequest.prompt_is_token_ids
```

### 12.2 Chat content part 中的 prompt_embeds

Chat 请求中也可以有：

```text
{type: "prompt_embeds", data: base64_tensor}
```

parser 会：

```text
1. 解码 base64 tensor；
2. 把 tensor 存进 tracker 的 prompt_embeds modality；
3. 在文本中放入 PROMPT_EMBEDS_PLACEHOLDER_TOKEN。
```

位置：`chat_utils.py:941` 到 `chat_utils.py:953`、`chat_utils.py:1722` 到 `chat_utils.py:1735`

之后 renderer 会把这个 sentinel 扩展为对应长度的 placeholder token span。`renderers/hf.py` 中的 mixed mode 注释说明：

```text
_process_multimodal 已经处理了 pre-expanded token ids；
后续会定位 prompt_embeds span，
并把 prompt_embeds entries 加进 mm_kwargs、mm_hashes、mm_placeholders。
```

位置：`renderers/hf.py:1201` 到 `renderers/hf.py:1245`

因此 chat content part 的 `prompt_embeds` 更像一种特殊 multimodal item：

```text
它的位置通过 placeholder token 表达；
它的数据进入 mm_kwargs / mm_features；
后续模型 forward 时再按 mask / placeholder 位置替换 embedding。
```

---

## 13. EngineCoreRequest 进入 Scheduler 前发生什么

`AsyncLLM.add_request()` 在得到 `EngineCoreRequest` 后会：

```text
1. assign_request_id() 生成内部 id；
2. output_processor.add_request() 建立输出队列；
3. engine_core.add_request_async(request) 发给 EngineCore。
```

位置：`async_llm.py:368` 到 `async_llm.py:412`

EngineCore 侧先做 `preprocess_add_request()`：

```python
if self.mm_receiver_cache is not None and request.mm_features:
    request.mm_features = self.mm_receiver_cache.get_and_update_features(
        request.mm_features
    )

req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`core.py:853` 到 `core.py:867`

`Request.from_engine_core_request()` 把字段转进调度层的 `Request`：

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
    ...
)
```

位置：`request.py:197` 到 `request.py:222`

`Request.__init__()` 保存：

```python
self.prompt_token_ids = prompt_token_ids
self.prompt_embeds = prompt_embeds
self.prompt_is_token_ids = prompt_is_token_ids
self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)
self.mm_features = mm_features or []
```

位置：`request.py:121` 到 `request.py:157`

最后 EngineCore 入队：

```python
self.scheduler.add_request(request)
```

位置：`core.py:403`

这说明 scheduler 收到的已经不是原始 OpenAI 请求，也不是 `MultiModalInput` 字典，而是 V1 内部 `Request`：

```text
Request.prompt_token_ids
Request.prompt_embeds
Request.prompt_is_token_ids
Request.mm_features
Request.sampling_params / pooling_params
```

---

## 14. 数据形态总览

从用户输入到调度层，可以按数据形态分 5 层。

```text
1. API 层
   ChatCompletionRequest.messages
   CompletionRequest.prompt / prompt_embeds
   Offline PromptType

2. Chat parser / prompt parser 层
   ConversationMessage
   mm_data
   mm_uuids
   placeholder text

3. Renderer 层
   TokensInput
   EmbedsInput
   MultiModalInput
   EncoderDecoderInput

4. Engine client 层
   EngineCoreRequest

5. Scheduler 层
   Request
```

关键字段流向：

```text
image_url / image_pil / image_embeds
  → mm_data["image"]
  → MultiModalInput.mm_kwargs["image"]
  → MultiModalFeatureSpec(modality="image")
  → Request.mm_features

audio_url / input_audio / audio_embeds
  → mm_data["audio"]
  → MultiModalInput.mm_kwargs["audio"]
  → MultiModalFeatureSpec(modality="audio")
  → Request.mm_features

video_url
  → mm_data["video"] 或 vision_chunk
  → MultiModalInput.mm_kwargs["video" / "vision_chunk"]
  → MultiModalFeatureSpec(...)
  → Request.mm_features

text prompt
  → chat template rendered prompt
  → tokenizer
  → prompt_token_ids
  → EngineCoreRequest.prompt_token_ids
  → Request.prompt_token_ids

prompt_embeds
  → EmbedsInput.prompt_embeds 或 prompt_embeds modality
  → EngineCoreRequest.prompt_embeds / mm_features
  → Request.prompt_embeds / mm_features
```

---

## 15. 容易混淆的点

### 15.1 mm_placeholders 不直接进入 EngineCoreRequest

`MultiModalInput` 有 `mm_placeholders`：

```text
modality -> list[PlaceholderRange]
```

但 `EngineCoreRequest` 没有单独的 `mm_placeholders` 字段。

它被合并进：

```text
MultiModalFeatureSpec.mm_position
```

对应构造位置：`input_processor.py:358` 到 `input_processor.py:366`

### 15.2 mm_kwargs 不直接进入 EngineCoreRequest

`mm_kwargs` 中的每个 item 会进入：

```text
MultiModalFeatureSpec.data
```

所以 EngineCore 之后看到的是 `mm_features`，不是原始 `mm_kwargs` 字典。

### 15.3 mm_hashes 有两层含义

`MultiModalInput.mm_hashes` 是按 modality 分组的 hash 字典。

进入 `MultiModalFeatureSpec` 后：

```text
mm_hash：原始 hash；
identifier：用于 cache 的 identifier，可能受 LoRA 影响。
```

如果启用 `enable_tower_connector_lora`，`identifier` 会拼上 LoRA name：

```python
return f"{lora_request.lora_name}:{mm_hash}"
```

位置：`input_processor.py:165` 到 `input_processor.py:181`

### 15.4 prompt_token_ids 里包含 placeholder token

多模态 item 的位置不是靠 side channel 猜出来的，而是由 tokenizer 后的 placeholder span 明确记录。

```text
prompt_token_ids：包含 placeholder token span；
mm_features[i].mm_position：指出该 feature 对应哪段 span；
mm_features[i].data：对应真实多模态数据或 processor 输出。
```

### 15.5 encoder-decoder 多模态会被重新拆分

`build_enc_dec_input()` 会把 encoder 多模态输入改造成：

```text
encoder_prompt：TokensInput
decoder_prompt：MultiModalInput
```

位置：`inputs/engine.py:315` 到 `inputs/engine.py:362`

因此即使是 encoder-decoder 模型，`InputProcessor.process_inputs()` 最终仍然从 decoder side 的 `MultiModalInput` 抽取 `mm_features`。

---

## 16. 本篇结论

多模态请求入口不是直接把 image/audio/video 塞进 scheduler，而是先经过 renderer 收敛成统一的 engine 输入。

最关键的转换是：

```text
OpenAI content part / Offline multi_modal_data
  → mm_data + placeholder text
  → MultiModalInput(prompt_token_ids, mm_kwargs, mm_hashes, mm_placeholders)
  → EngineCoreRequest(prompt_token_ids, mm_features)
  → Request(prompt_token_ids, mm_features)
```

从这个点开始，后续 scheduler / worker / model runner 只需要处理两个问题：

```text
1. prompt_token_ids 中哪些 token 要执行；
2. mm_features 中哪些 encoder input 要在对应 placeholder span 上注入。
```

后续专题可以继续顺着 `Request.mm_features` 往下看：

```text
Request.mm_features
  → Scheduler.scheduled_encoder_inputs
  → NewRequestData.mm_features
  → CachedRequestState.mm_features
  → GPUModelRunner._prepare_inputs()
  → encoder cache / inputs_embeds
  → model forward
```
