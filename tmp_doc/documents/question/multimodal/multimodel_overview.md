# vLLM V1 Multimodal 逻辑梳理

源码位置：

- `code/vllm/vllm/config/multimodal.py`
- `code/vllm/vllm/multimodal/`
- `code/vllm/vllm/assets/`
- `code/vllm/vllm/inputs/preprocess.py`
- `code/vllm/vllm/renderers/base.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/models/`

本文按“先定边界，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式，梳理 vLLM V1 里多模态输入从用户请求进入系统、被 processor 转成特征、被 Scheduler 调度 encoder、在 ModelRunner 执行 encoder 并合并 `inputs_embeds`，最后进入普通 forward / sampling / output 的完整链路。

它和 `executor_worker_model_runner` 目录的总览文档一样，不是逐个 helper 函数的字典式说明，而是先回答几个最关键的问题：

```text
1. Multimodal 子系统在 vLLM V1 中是哪一层？负责什么？
2. image / audio / video / embeds 如何从 entrypoints 进入 EngineCoreRequest？
3. parser / processor / mapper / registry 各自负责什么？
4. placeholder token 如何和 prompt token、encoder output 对齐？
5. Scheduler 为什么要调度 encoder input？encoder budget 和 encoder cache 怎么工作？
6. ModelRunner 如何执行 multimodal encoder，并把结果拼进 inputs_embeds？
7. 多模态模型类需要提供哪些接口？
8. 多模态链路和 KV cache、LoRA、spec decode、parallel、EC connector 的关系是什么？
```

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本文按“先定角色，再走主链路，再拆关键阶段”的顺序组织。

要回答的问题分成 10 组：

```text
1. Multimodal 是哪一层？它不是哪一层？
2. 用户请求中的多模态数据如何进入 vLLM？
3. MultiModalRegistry / DataParser / Processor / Mapper 如何协作？
4. MultiModalFeatureSpec 如何承载一个多模态 item？
5. placeholder / mm_position / mm_hash / identifier 分别是什么？
6. Scheduler 如何根据 token window、encoder budget、encoder cache 调度多模态 encoder？
7. EncoderCacheManager 和 GPUModelRunner.encoder_cache 有什么区别？
8. GPUModelRunner 如何执行 _execute_mm_encoder() / _gather_mm_embeddings() / _preprocess()？
9. 多模态模型类需要提供 embed_multimodal() / embed_input_ids() 等哪些接口？
10. 多模态链路如何影响 chunked prefill、prompt_embeds、LoRA、KV / EC connector 和输出链路？
```

阅读顺序建议：

```text
multimodel_overview.md
  → 01_multimodal_role.md
  → 02_multimodal_request_entry.md
  → 03_parser_processor_mapper.md
  → 04_placeholders_and_prompt_alignment.md
  → 05_feature_spec_and_cache.md
  → 06_encoder_budget_and_scheduler.md
  → 07_encoder_cache_lifecycle.md
  → 08_model_runner_mm_encoder_flow.md
  → 09_multimodal_model_interfaces.md
  → 10_multimodal_runtime_interactions.md
```

如果只想先抓住一条主线，可以先读总览，再读 `01`、`06`、`08`、`09`。

---

## 1. Multimodal 在 vLLM V1 里是什么

`Multimodal` 是 vLLM V1 中把非文本输入接入语言模型执行链路的输入准备、encoder 调度、encoder output 缓存和 embedding 拼接系统。

它负责：

```text
- 判断模型是否支持多模态输入；
- 解析 image / audio / video / vision_chunk / embeds 等原始数据；
- 调用模型注册的 processor，把原始 media 转成模型需要的 mm_kwargs；
- 生成 placeholder、mm_position、mm_hash、identifier；
- 构造 MultiModalFeatureSpec，并放入 EngineCoreRequest / Request；
- 为 Scheduler 提供 encoder token budget 和 encoder cache size；
- 让 Scheduler 决定本轮哪些 encoder input 必须可用；
- 在 ModelRunner 侧执行 model.embed_multimodal()；
- 缓存 encoder output；
- 根据 placeholder 位置把 encoder output 拼回 inputs_embeds；
- 最后汇入普通 decoder forward、logits、sampling 和 output 链路。
```

它不负责：

```text
- 不负责用户请求队列调度；
- 不负责 decoder KV block 分配和抢占；
- 不负责 attention backend 的主链路执行；
- 不负责 token sampling；
- 不负责 detokenize；
- 不负责 OpenAI response 协议包装；
- 不负责模型权重加载；
- 不负责把 ModelRunnerOutput 转成 RequestOutput。
```

一句话记忆：

```text
Multimodal 负责“把非文本输入变成模型 forward 能消费的 embeddings，并把这件事接入调度、缓存和执行链路”。
```

---

## 2. 一句话总览链路

最小主链路是：

```text
用户请求 prompt + multi_modal_data
  → entrypoints / renderer
  → MultiModalDataParser
  → model-specific MultiModalProcessor
  → MultiModalInput(prompt_token_ids, mm_kwargs, mm_placeholders, mm_hashes)
  → InputProcessor.process_inputs()
  → MultiModalFeatureSpec
  → EngineCoreRequest.mm_features
  → Request.mm_features
  → Scheduler.schedule()
  → Scheduler._try_schedule_encoder_inputs()
  → EncoderCacheManager
  → SchedulerOutput.scheduled_encoder_inputs
  → Executor / Worker
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._build_attention_metadata()
  → GPUModelRunner._preprocess()
  → GPUModelRunner._execute_mm_encoder()
  → model.embed_multimodal(**mm_kwargs)
  → GPUModelRunner.encoder_cache[identifier]
  → GPUModelRunner._gather_mm_embeddings()
  → model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
  → inputs_embeds
  → GPUModelRunner._model_forward()
  → logits / pooling
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```

对应源码主入口：

- `code/vllm/vllm/renderers/base.py:682`
- `code/vllm/vllm/v1/engine/input_processor.py:242`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1279`
- `code/vllm/vllm/v1/core/sched/output.py:204`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py:17`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

如果把链路再压缩一层，可以记成：

```text
原始 media
  → processor outputs
  → MultiModalFeatureSpec
  → Scheduler encoder input 调度
  → ModelRunner encoder 执行
  → inputs_embeds 拼接
  → 普通模型 forward / sampling
```

---

## 3. 这条链路为什么要拆成多层

多模态链路贯穿输入、调度、执行和模型实现。如果全部放在一个组件里，会把 CPU 侧解析、Scheduler 侧预算、GPU 侧 encoder 执行和模型特定逻辑混在一起。

### 3.1 Renderer / Processor 适合放“原始输入解析”

Renderer 和 processor 处理的是用户输入形态：

```text
- OpenAI chat content parts；
- offline API 的 multi_modal_data；
- image URL / base64 / PIL / ndarray / tensor；
- audio array / tensor / sampling rate；
- video frames / metadata；
- prompt_embeds / precomputed embeds；
- tokenizer 和 HF processor 的协作；
- processor cache 和跨进程传输。
```

这些逻辑和 Scheduler 的 token budget 无关，也不应该进入 GPU forward 主链路。

### 3.2 InputProcessor 适合放“请求状态标准化”

InputProcessor 的职责是把 renderer / processor 的输出变成 EngineCore 可理解的请求对象。

对多模态来说，它最关键的事情是：

```text
- 按 placeholder 位置排序多模态 item；
- 构造 MultiModalFeatureSpec；
- 处理 mm_hash 和 identifier；
- 把 mm_features 放进 EngineCoreRequest。
```

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:242`

### 3.3 Scheduler 适合放“是否需要执行 encoder 的决策”

Scheduler 不解析图片，也不跑 encoder。

它只回答一个问题：

```text
这一轮 decoder token window 是否需要某些多模态 encoder output，如果需要，预算和 cache 是否允许？
```

因此多模态在 Scheduler 里的核心字段是：

```text
SchedulerOutput.scheduled_encoder_inputs
SchedulerOutput.free_encoder_mm_hashes
```

源码位置：

- `code/vllm/vllm/v1/core/sched/output.py:204`
- `code/vllm/vllm/v1/core/sched/output.py:215`

### 3.4 ModelRunner 适合放“encoder 执行和 embedding 拼接”

ModelRunner 持有模型、device、batch 状态和执行缓存，因此它适合处理：

```text
- 从 SchedulerOutput 找到本轮要跑的 encoder input；
- 调用 model.embed_multimodal()；
- 保存 GPU 侧 encoder output；
- 根据 mm_position 切片；
- 生成 is_mm_embed mask；
- 调用 model.embed_input_ids()；
- 生成 inputs_embeds；
- 继续普通 forward。
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

---

## 4. 核心角色总览

### 4.1 MultiModalConfig

`MultiModalConfig` 控制多模态能力开关、输入限制、processor cache、encoder 执行策略等。

源码位置：`code/vllm/vllm/config/multimodal.py:74`

关键字段包括：

```text
language_model_only
limit_per_prompt
enable_mm_embeds
media_io_kwargs
mm_processor_kwargs
mm_processor_cache_gb
mm_processor_cache_type
mm_tensor_ipc
mm_encoder_only
mm_encoder_tp_mode
mm_encoder_attn_backend
mm_encoder_attn_dtype
skip_mm_profiling
video_pruning_rate
```

其中要特别区分两类配置：

```text
processor 侧配置：影响 media decode、HF processor、CPU 侧 tensor 构造和缓存；
encoder 侧配置：影响 multimodal encoder 的执行图、TP、attention backend 和 dtype。
```

### 4.2 MultiModalRegistry

`MultiModalRegistry` 是模型类到多模态 processor 的注册中心。

源码位置：`code/vllm/vllm/multimodal/registry.py:98`

它负责：

```text
- 判断模型是否支持多模态输入；
- 注册模型对应的 processor / info / dummy inputs；
- 创建 processor；
- 创建 processor cache；
- 给 MultiModalBudget 提供模型相关的最大 token 信息。
```

多模态模型能不能处理 image / audio / video，不是靠通用层硬编码，而是靠模型类注册 processor 和 processing info。

### 4.3 MultiModalDataParser

`MultiModalDataParser` 把用户传入的 `MultiModalDataDict` 标准化成内部 data items。

源码位置：`code/vllm/vllm/multimodal/parse.py:489`

内置支持的 modality 包括：

```text
audio
image
video
vision_chunk
```

它还会识别 embedding 输入。对于 embeddings，后续可以走 embedding-only / prompt_embeds 路径，不需要真正执行视觉塔或音频塔 encoder。

### 4.4 Renderer / InputPreprocessor

Renderer 是外部输入进入多模态 processor 的重要入口。

源码位置：`code/vllm/vllm/renderers/base.py:682`

主流程是：

```text
Renderer._process_multimodal()
  → get_mm_processor()
  → processor.info.parse_mm_data(mm_data)
  → parse_mm_uuids(mm_uuids)
  → processor.apply(MMProcessorInputs)
  → MultiModalInput
```

`InputPreprocessor._process_multimodal()` 只是薄封装，会转发给 renderer。

源码位置：`code/vllm/vllm/inputs/preprocess.py:90`

### 4.5 InputProcessor

InputProcessor 把 `MultiModalInput` 整理成 EngineCore 使用的请求对象。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:242`

它会：

```text
- 判断当前模型是否 supports_mm_inputs；
- 初始化 MultiModalBudget；
- 从 decoder_inputs 中取出 mm_kwargs / mm_placeholders / mm_hashes；
- 按 placeholder 位置排序；
- 构造 MultiModalFeatureSpec；
- 填入 EngineCoreRequest.mm_features。
```

### 4.6 Scheduler / EncoderCacheManager

Scheduler 决定哪些 encoder input 本轮必须可用。

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1279`

`EncoderCacheManager` 管理 Scheduler 侧的 encoder output 逻辑账本。

源码位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

它不存 tensor，只记录：

```text
- 哪些 identifier 已缓存；
- 哪些 request 正在引用某个 encoder output；
- 哪些缓存 entry 已经可释放；
- 哪些 hash 需要通知 worker 删除物理 tensor。
```

### 4.7 GPUModelRunner

GPUModelRunner 是多模态 encoder 执行和 embedding 拼接的落地点。

关键源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:492`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

它负责：

```text
- 保存 CachedRequestState.mm_features；
- 释放 Scheduler 要求释放的 encoder cache；
- 从 scheduled_encoder_inputs 中取出本轮 encoder item；
- 调用 model.embed_multimodal()；
- 保存 encoder_cache[identifier]；
- 按 mm_position gather embedding；
- 调用 model.embed_input_ids()；
- 生成 inputs_embeds；
- 继续普通 forward / logits / sampling。
```

### 4.8 Multimodal model class

具体模型类负责真正的模型语义。

常见接口包括：

```text
embed_multimodal(**mm_kwargs)
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
get_num_mm_encoder_tokens(...)
get_num_mm_connector_tokens(...)
get_mm_mapping()
```

通用层不理解每个模型的视觉塔、音频塔、projector、connector 细节，只约定输入输出接口。

---

## 5. 从用户输入到 EngineCoreRequest

多模态请求通常来自两类入口：

```text
OpenAI / serving entrypoints：chat content parts、image_url、audio、video 等；
offline API：prompt + multi_modal_data / multi_modal_uuids / mm_processor_kwargs。
```

进入 engine 之前，Renderer 会把它们变成 `MultiModalInput`。

### 5.1 Renderer 产出什么

`MultiModalInput` 通常包含：

```text
prompt_token_ids
mm_kwargs
mm_placeholders
mm_hashes
```

含义是：

```text
prompt_token_ids：包含多模态 placeholder 的 token 序列；
mm_kwargs：模型 processor 输出，后续传给 embed_multimodal()；
mm_placeholders：每个多模态 item 在 prompt token 里的位置范围；
mm_hashes：processor/cache/encoder 复用所需的 hash。
```

### 5.2 InputProcessor 构造 mm_features

InputProcessor 在 `process_inputs()` 中会把多模态字典压平成列表。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:242`

核心动作是：

```text
1. 从 decoder_inputs 中取出 mm_kwargs、mm_placeholders、mm_hashes；
2. 使用 argsort_mm_positions() 按 placeholder 出现顺序排序；
3. 对每个多模态 item 构造 MultiModalFeatureSpec；
4. 把列表写入 EngineCoreRequest.mm_features。
```

这样做的原因是：

```text
Scheduler 和 ModelRunner 后续都按 request.mm_features 的索引来引用多模态 item。
```

### 5.3 Request 保存 mm_features

`Request` 初始化参数包含 `mm_features`。

源码位置：

- `code/vllm/vllm/v1/request.py:70`
- `code/vllm/vllm/v1/request.py:157`

也就是说，多模态特征不是临时输入，而是 request 生命周期状态的一部分。

---

## 6. MultiModalFeatureSpec 是什么

`MultiModalFeatureSpec` 是 V1 多模态链路中最核心的数据结构。

源码位置：`code/vllm/vllm/multimodal/inputs.py:302`

它代表一个多模态 item。

字段包括：

```text
data: MultiModalKwargsItem | None
modality: str
identifier: str
mm_position: PlaceholderRange
mm_hash: str | None
```

字段含义：

| 字段 | 含义 |
|---|---|
| `data` | 该 item 的 processor 输出，即模型 encoder 需要的 kwargs；缓存命中时可以是 `None` |
| `modality` | 输入模态，例如 image / audio / video / prompt_embeds |
| `identifier` | encoder output cache 的 key，tower connector LoRA 场景下可能带 LoRA 前缀 |
| `mm_position` | placeholder 在 prompt token 序列中的位置和长度 |
| `mm_hash` | processor output cache 的 hash，通常不带 LoRA 前缀 |

一个请求如果包含 2 张图和 1 段音频，就会有 3 个 `MultiModalFeatureSpec`。

需要注意：

```text
MultiModalFeatureSpec.data 可以为 None。
```

这表示 processor 输出可能已经缓存，或者不需要跨 API server / engine / worker 重新传输 tensor payload。但只要 identifier 和 mm_position 还在，Scheduler 和 ModelRunner 仍然能判断和使用对应 encoder output。

---

## 7. placeholder、mm_position、mm_hash、identifier

这几个概念很容易混淆。

### 7.1 placeholder

多模态 decoder-only VLM 通常会把 image / audio / video 映射为 prompt token 序列中的一段 placeholder。

这段 placeholder 的作用是：

```text
先在 token 序列里占住位置，后续在 ModelRunner 中用 encoder embeddings 替换对应 token embeddings。
```

### 7.2 mm_position

`mm_position` 是 `PlaceholderRange`，描述某个多模态 item 对应的 placeholder 范围。

它至少表达：

```text
offset：placeholder 起始 token 位置；
length：placeholder token 区间长度；
get_num_embeds()：实际对应多少 encoder embeddings；
get_embeds_indices_in_range()：把本轮 token window 映射到 embedding slice。
```

这点很重要：

```text
placeholder token 数量和 encoder embedding 数量不一定简单一一对应。
```

在动态分辨率视觉模型、video pruning、M-RoPE / XD-RoPE、prompt_embeds、audio-in-video 等场景中，都必须通过 `PlaceholderRange` 做精确映射。

### 7.3 mm_hash

`mm_hash` 更偏 processor/cache 层，通常表示原始多模态 item 或 processor output 的复用 key。

它主要影响：

```text
- processor cache；
- 多进程 IPC 中是否需要重新传输处理结果；
- 和 identifier 的基础对应关系。
```

### 7.4 identifier

`identifier` 是 encoder output cache 的 key。

在普通场景下，它可能和 `mm_hash` 相同。

但在 tower connector LoRA 场景下，同一张图片在不同 LoRA 下的 encoder output 可能不同，因此 identifier 可能带上 LoRA 名字。

相关入口：`code/vllm/vllm/v1/engine/input_processor.py:165`

区别可以记成：

```text
mm_hash：processor output 复用 key；
identifier：encoder output 复用 key。
```

---

## 8. Scheduler 如何调度 multimodal encoder

Scheduler 不运行 encoder，但它必须保证 decoder forward 触达某个多模态 placeholder 前，对应 encoder output 已经可用。

核心入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1279`

### 8.1 Scheduler 初始化时的多模态预算

如果模型支持多模态输入，Scheduler 会创建 `MultiModalBudget`，并据此设置：

```text
max_num_encoder_input_tokens
encoder_cache_size
encoder_cache_manager
```

相关逻辑在 scheduler 初始化阶段。

其中 `encoder_cache_manager` 对 decoder-only 多模态模型通常是 `EncoderCacheManager`，对 encoder-decoder 模型可能是 `EncoderDecoderCacheManager`。

### 8.2 调度条件

`_try_schedule_encoder_inputs()` 会判断一个 encoder input 是否需要本轮调度。

条件可以概括为：

```text
- 该多模态 item 的 placeholder output tokens 和本轮 token window 有重叠；
- 对应 encoder output 尚未在 encoder cache 中命中；
- 对应 output 不在远端 encoder cache 中；
- encoder compute budget 足够；
- encoder cache 有空间保存 output；
- chunked multimodal input 策略允许本轮覆盖方式。
```

### 8.3 token window 决定是否触达多模态 item

Scheduler 会用 `get_mm_features_in_window()` 找到本轮 token 范围涉及哪些 `mm_features`。

逻辑可以理解为：

```text
num_computed_tokens 到 num_computed_tokens + num_new_tokens
这一段 decoder token 如果覆盖某个 placeholder，
就必须保证该 placeholder 对应 encoder output 已经可用。
```

### 8.4 budget 或 cache 不够时会截断本轮 token

如果 encoder compute budget 不够，或者 encoder cache 没有空间，Scheduler 不会让 decoder forward 硬跨过尚未准备好的多模态 placeholder。

它会缩短本轮 `num_new_tokens`，常见结果是：

```text
本轮最多算到 placeholder 前面，等后续轮次有 encoder budget / cache 后再继续。
```

这也是多模态和 chunked prefill 强耦合的原因。

### 8.5 SchedulerOutput 传递调度结果

多模态相关的两个核心输出字段是：

```text
scheduled_encoder_inputs: dict[str, list[int]]
free_encoder_mm_hashes: list[str]
```

源码位置：

- `code/vllm/vllm/v1/core/sched/output.py:204`
- `code/vllm/vllm/v1/core/sched/output.py:215`

其中：

```text
scheduled_encoder_inputs：req_id -> request.mm_features 中需要本轮执行 encoder 的索引列表；
free_encoder_mm_hashes：Scheduler 侧逻辑 cache 已驱逐，需要 worker 删除物理 tensor 的 identifier/hash 列表。
```

注意：

```text
scheduled_encoder_inputs 里存的不是 hash，而是 mm_features 的索引。
```

---

## 9. EncoderCacheManager 和 GPUModelRunner.encoder_cache

多模态有两个容易混淆的 cache：processor cache 和 encoder cache；encoder cache 内部又分逻辑账本和物理 tensor。

### 9.1 processor cache

processor cache 在 `MultiModalRegistry` 相关工厂中创建。

它缓存的是：

```text
Renderer / MultiModalProcessor 处理后的 processor outputs / mm_kwargs。
```

主要目的：

```text
避免重复 decode 图片、处理音频、抽帧、调用 HF processor、构造 CPU 侧 tensor。
```

控制字段包括：

```text
mm_processor_cache_gb
mm_processor_cache_type
mm_shm_cache_max_object_size_mb
mm_tensor_ipc
```

### 9.2 EncoderCacheManager

`EncoderCacheManager` 是 Scheduler 侧逻辑账本。

源码位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

它管理的是：

```text
- 哪些 encoder output 已经逻辑缓存；
- 每个 cached output 被哪些 request 引用；
- 哪些 output 已经没有 request 引用，因而 freeable；
- cache 空间是否足够；
- 需要通知 worker 删除哪些物理 tensor。
```

它不存储 GPU tensor。

### 9.3 GPUModelRunner.encoder_cache

`GPUModelRunner.encoder_cache` 是 Worker / ModelRunner 侧物理 cache。

它存的是：

```text
model.embed_multimodal() 的输出 tensor。
```

写入位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`

读取位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`

二者边界：

```text
EncoderCacheManager：Scheduler 侧决定哪些 output 应该存在；
GPUModelRunner.encoder_cache：Worker 侧真正保存 output tensor。
```

如果 Scheduler 发送 `free_encoder_mm_hashes`，ModelRunner 会删除对应物理 cache entry。

---

## 10. ModelRunner 中的多模态执行链路

GPUModelRunner 是多模态真正落地的地方。

### 10.1 初始化状态

初始化阶段会判断模型是否支持多模态：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:492`

关键状态包括：

```text
self.supports_mm_inputs
self.mm_registry
self.mm_budget
self.encoder_cache
self.inputs_embeds
self.uses_mrope
self.uses_xdrope_dim
```

多模态模型通常需要 `inputs_embeds` buffer，因为 forward 前要把文本 token embedding 和多模态 embedding 合并。

### 10.2 _update_states()

当 `SchedulerOutput` 进入 ModelRunner，`_update_states()` 会把新请求或恢复请求的 `mm_features` 保存到 `CachedRequestState`。

它还会处理：

```text
- finished request 清理；
- input batch 更新；
- free_encoder_mm_hashes 对应物理 cache 删除；
- M-RoPE / XD-RoPE 相关位置状态；
- prompt_embeds / multimodal state 保存。
```

这一步保证后续 `_execute_mm_encoder()` 可以通过 `req_state.mm_features` 找到每个多模态 item。

### 10.3 _batch_mm_inputs_from_scheduler()

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`

它从 `SchedulerOutput.scheduled_encoder_inputs` 中取出本轮要执行的 encoder input：

```text
for req_id, encoder_input_ids in scheduled_encoder_inputs:
  req_state = self.requests[req_id]
  for mm_input_id in encoder_input_ids:
    mm_feature = req_state.mm_features[mm_input_id]
```

输出三类信息：

```text
mm_hashes：encoder cache key，也就是 MultiModalFeatureSpec.identifier；
mm_kwargs：传给模型 embed_multimodal() 的 processor 输出；
mm_lora_refs：tower connector LoRA 场景需要的请求和 placeholder 位置。
```

如果 `mm_feature.data is None`，说明 processor payload 不需要本轮传递或已缓存，相关逻辑会跳过直接 payload 处理。

### 10.4 _execute_mm_encoder()

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`

这是多模态 encoder 的实际执行入口。

它处理几类情况：

```text
1. prompt_embeds：输入已经是 embedding，直接放入 encoder_cache；
2. 普通 image / audio / video：按 modality 分组和 batch；
3. encoder CUDA graph：如果可用，走 cudagraph manager；
4. 普通路径：调用 model.embed_multimodal(**mm_kwargs_batch)；
5. 输出检查：确认 encoder output 数量和 item 数量一致；
6. 写入 self.encoder_cache[identifier]；
7. 需要时保存到 EC connector。
```

普通多模态路径的核心调用是：

```text
model.embed_multimodal(**mm_kwargs_batch)
```

这一步是模型特定的，通用层只负责调用约定接口。

### 10.5 _gather_mm_embeddings()

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`

执行完 encoder 后，还不能直接 forward，因为本轮 batch 可能只计算 prompt 的一部分，也可能 batch 顺序已经被 `_prepare_inputs()` 重排。

`_gather_mm_embeddings()` 会：

```text
- 按当前 input_batch.req_ids 遍历 batch；
- 根据每个请求的 num_computed_tokens 和 num_scheduled_tokens 找到本轮 token window；
- 找到 window 覆盖的 mm_features；
- 根据 mm_position 计算 encoder output 的 slice；
- 从 encoder_cache[identifier] 取出对应 tensor；
- 生成 mm_embeds；
- 生成 is_mm_embed mask。
```

关键点：

```text
_gather_mm_embeddings() 必须在 _prepare_inputs() 之后执行，
因为 _prepare_inputs() 可能重排 batch，
多模态 embedding 必须和当前 batch row 顺序对齐。
```

### 10.6 _preprocess()

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

`_preprocess()` 是模型 forward 前准备 `input_ids / inputs_embeds / model_kwargs` 的地方。

多模态路径大致是：

```text
if supports_mm_inputs and is_first_rank and not is_encoder_decoder:
  _execute_mm_encoder(scheduler_output)
  mm_embeds, is_mm_embed = _gather_mm_embeddings(scheduler_output)
  inputs_embeds_scheduled = model.embed_input_ids(
      input_ids,
      multimodal_embeddings=mm_embeds,
      is_multimodal=is_mm_embed,
  )
  self.inputs_embeds.copy_(inputs_embeds_scheduled)
  input_ids, inputs_embeds = _prepare_mm_inputs(...)
```

这一步完成了最关键的转换：

```text
prompt token ids + placeholder + encoder output
  → inputs_embeds
```

之后的 `_model_forward()` 不需要理解用户原始图片或音频。

---

## 11. decoder-only 和 encoder-decoder 的区别

多模态链路在 decoder-only VLM 和 encoder-decoder 模型中不完全一样。

### 11.1 decoder-only VLM

典型图文模型走这条路径：

```text
prompt 中有 image/audio/video placeholder
  → 执行 multimodal encoder
  → 从 encoder_cache gather embeddings
  → embed_input_ids() 替换 placeholder 对应位置
  → forward 吃 inputs_embeds
```

也就是说，encoder output 被拼进 decoder 输入 embedding 序列。

### 11.2 encoder-decoder 模型

encoder-decoder 模型更接近：

```text
执行 encoder
  → 得到 encoder_outputs
  → 作为 model_kwargs 传给 decoder
```

它通常不做 prompt replacement。

在 `_preprocess()` 中，encoder-decoder 有单独分支，会把 `_execute_mm_encoder()` 的输出作为 `encoder_outputs` 放入 `model_kwargs`。

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

可以记成：

```text
Decoder-only VLM：encoder output 拼进 inputs_embeds；
Encoder-decoder：encoder output 作为 encoder_outputs 传给 decoder。
```

---

## 12. prompt_embeds / embedding-only 路径

`prompt_embeds` 或预计算多模态 embeddings 是特殊但重要的路径。

它的特点是：

```text
输入已经在 embedding space 中，不需要再执行 image/audio/video encoder。
```

但它不是完全绕过多模态系统。

它仍然可能进入：

```text
MultiModalFeatureSpec
SchedulerOutput.scheduled_encoder_inputs
GPUModelRunner.encoder_cache
_gather_mm_embeddings()
model.embed_input_ids()
```

区别只是 `_execute_mm_encoder()` 对 `prompt_embeds` 会直接把 tensor 放进 encoder cache，而不是调用 `model.embed_multimodal()`。

这个设计让 embedding-only 输入复用同一套 placeholder、scheduler、cache、gather 和 merge 逻辑。

---

## 13. 多模态和 chunked prefill

多模态输入常常对应很长的 placeholder，尤其是高分辨率图片和视频。

在 chunked prefill 下，一个请求的 prompt 可能被分多轮执行。

Scheduler 必须保证：

```text
如果本轮 token window 覆盖某个多模态 placeholder，
对应 encoder output 必须在本轮 forward 前可用。
```

因此 `_try_schedule_encoder_inputs()` 会结合：

```text
num_computed_tokens
num_new_tokens
mm_position.offset
mm_position.length
encoder_compute_budget
encoder_cache_size
disable_chunked_mm_input
```

来决定：

```text
- 本轮是否调度某个 encoder input；
- 是否允许只覆盖多模态 placeholder 的一部分；
- 是否因为 budget/cache 不够而缩短 num_new_tokens。
```

这说明多模态不是简单的“prefill 前先跑一遍 encoder”。它和 V1 的 token 级调度、chunked prefill、encoder cache 都是耦合的。

---

## 14. 多模态和 KV cache 的关系

多模态 encoder cache 和 decoder KV cache 不是同一个东西。

### 14.1 decoder KV cache

```text
对象：decoder self-attention 的 key/value；
管理者：KVCacheManager / Scheduler / Worker KV tensors；
粒度：block；
作用：加速 autoregressive decode；
使用位置：attention backend。
```

### 14.2 multimodal encoder cache

```text
对象：image/audio/video/prompt_embeds 的 encoder output embeddings；
逻辑管理者：EncoderCacheManager；
物理存储：GPUModelRunner.encoder_cache；
粒度：单个 multimodal item；
作用：避免重复执行 multimodal encoder，并支持 placeholder embedding 拼接；
使用位置：ModelRunner._gather_mm_embeddings() / _preprocess()。
```

关键区别：

```text
KV cache 是 decoder 历史 token 的 attention 状态；
encoder cache 是非文本输入经过 encoder 后的 embedding 结果。
```

两者都影响显存和调度，但职责完全不同。

---

## 15. 多模态和 LoRA / EC connector / parallel

### 15.1 LoRA

普通 LoRA 主要影响语言模型权重。

但 tower connector LoRA 可能影响多模态 encoder 或 connector 输出，因此 `identifier` 可能需要包含 LoRA 信息，避免不同 LoRA 下复用同一个 encoder output。

相关入口：`code/vllm/vllm/v1/engine/input_processor.py:165`

可以记成：

```text
如果 LoRA 会改变 multimodal encoder / connector 输出，encoder cache key 必须区分 LoRA。
```

### 15.2 EC connector

EC connector 用于远端 encoder cache 交互。

Scheduler 调度 encoder input 时会考虑：

```text
- 本地 encoder cache 是否命中；
- 远端 encoder cache 是否已有对应 output；
- worker 是否需要保存或加载 encoder cache output。
```

ModelRunner 侧 `_execute_mm_encoder()` 写入本地 `encoder_cache` 后，也可能调用相关 connector 保存 output。

### 15.3 Tensor parallel / pipeline parallel

多模态模型仍然运行在 vLLM 的 TP / PP 框架中。

需要注意：

```text
- multimodal encoder 可能有自己的 TP 策略，例如 mm_encoder_tp_mode；
- _preprocess() 中多模态 encoder 通常只在 first rank 等特定位置执行；
- PP 场景下还要处理 intermediate_tensors；
- forward 主链路仍然由 ModelRunner 统一组织。
```

多模态没有绕开 executor / worker / model_runner 分层。

---

## 16. 多模态和输出链路

多模态通常只影响输入准备和 prefill 阶段。

对于 generation，后续仍然是普通链路：

```text
inputs_embeds / input_ids
  → model.forward()
  → hidden_states
  → compute_logits()
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor / RequestOutput
```

也就是说，多模态输入不会让输出对象变成另一套体系。

对上层来说，图文请求最终仍然产生 token、logprobs、finish reason 等普通 generation 输出。

---

## 17. 关键数据结构关系

### 17.1 `MultiModalConfig`

多模态行为配置，包括输入限制、processor 参数、processor cache、encoder TP、encoder attention、embedding-only 等。

### 17.2 `MultiModalRegistry`

模型类到 processor / processing info / dummy inputs 的注册中心。

### 17.3 `MultiModalDataParser`

把原始 image / audio / video / embeddings 解析成内部 data items。

### 17.4 `MultiModalInput`

renderer / processor 的输出，通常包含 `prompt_token_ids`、`mm_kwargs`、`mm_placeholders`、`mm_hashes`。

### 17.5 `MultiModalFeatureSpec`

单个多模态 item 在 V1 engine / scheduler / worker 链路中的核心描述对象。

### 17.6 `PlaceholderRange`

描述 placeholder 在 prompt token 序列中的范围，以及 token window 到 encoder embedding slice 的映射。

### 17.7 `EngineCoreRequest.mm_features`

InputProcessor 输出给 EngineCore 的多模态特征列表。

### 17.8 `Request.mm_features`

Scheduler 持有的请求级多模态状态。

### 17.9 `SchedulerOutput.scheduled_encoder_inputs`

本轮需要 worker 执行的 encoder input 索引。

### 17.10 `SchedulerOutput.free_encoder_mm_hashes`

本轮需要 worker 释放的 encoder output cache key。

### 17.11 `EncoderCacheManager`

Scheduler 侧 encoder output 逻辑缓存账本。

### 17.12 `GPUModelRunner.encoder_cache`

Worker 侧 encoder output tensor cache。

### 17.13 `model.embed_multimodal()`

模型特定多模态 encoder 执行入口。

### 17.14 `model.embed_input_ids()`

把文本 token embedding 和多模态 embedding 合并成 `inputs_embeds` 的入口。

---

## 18. 端到端例子：一张图 + 文本

假设一个 decoder-only VLM 请求包含一张图片和一段文本。

完整链路可以写成：

```text
1. 用户传入：
   prompt + multi_modal_data={"image": ...}

2. Renderer._process_multimodal()
   → parse_mm_data()
   → mm_processor.apply()
   → 得到 prompt_token_ids / mm_kwargs / mm_placeholders / mm_hashes

3. InputProcessor.process_inputs()
   → 按 placeholder 位置排序
   → 构造 MultiModalFeatureSpec(data, modality, identifier, mm_position, mm_hash)
   → EngineCoreRequest.mm_features

4. EngineCore / Request
   → 保存 request.mm_features

5. Scheduler.schedule()
   → 根据本轮 token window 找到涉及的 mm_feature
   → 检查 EncoderCacheManager 是否命中
   → 检查 encoder_compute_budget 和 encoder cache 空间
   → SchedulerOutput.scheduled_encoder_inputs[req_id] = [0]

6. Executor / Worker
   → 把 SchedulerOutput 发给 GPUModelRunner

7. GPUModelRunner._update_states()
   → new_req_data.mm_features 保存到 CachedRequestState
   → 处理 free_encoder_mm_hashes

8. GPUModelRunner._preprocess()
   → _execute_mm_encoder()
      → model.embed_multimodal(**mm_kwargs)
      → encoder_cache[identifier] = image_embedding
   → _gather_mm_embeddings()
      → 根据 mm_position 从 encoder_cache 切片
      → 生成 mm_embeds 和 is_mm_embed
   → model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)
   → 得到 inputs_embeds

9. GPUModelRunner._model_forward()
   → 用 inputs_embeds / positions / attention metadata 执行模型

10. generation 后续
   → compute_logits()
   → sample_tokens()
   → ModelRunnerOutput
   → Scheduler.update_from_output()
```

这条链路里，多模态只改变 forward 输入准备，不改变采样和输出处理的基本框架。

---

## 19. 容易混淆的点

### 19.1 Multimodal 是不是一个模型执行器？

不是。

```text
模型执行器是 Worker / ModelRunner；
Multimodal 是输入处理、encoder 辅助、embedding 拼接和缓存机制。
```

### 19.2 `MultiModalFeatureSpec.data` 是不是一定有 tensor？

不是。

```text
data 可以为 None，表示 processor 输出已经缓存或不需要通过 IPC 重新发送。
```

### 19.3 `mm_hash` 和 `identifier` 是不是同一个东西？

不总是。

```text
mm_hash：processor cache key，通常不带 LoRA 前缀；
identifier：encoder cache key，tower connector LoRA 场景下可能带 LoRA 前缀。
```

### 19.4 `scheduled_encoder_inputs` 里存的是 hash 吗？

不是。

```text
scheduled_encoder_inputs: req_id -> list[input_id]
```

这里的 `input_id` 是 `Request.mm_features` 中的索引。

### 19.5 encoder cache hit 后还会执行 `_execute_mm_encoder()` 吗？

通常不会对该 item 执行。

Scheduler 命中 `EncoderCacheManager.check_and_update_cache()` 后，不会把该 item 加入 `scheduled_encoder_inputs`。

### 19.6 prompt_embeds 是否绕过所有多模态机制？

不是。

它跳过真正的 multimodal encoder，但仍然复用：

```text
MultiModalFeatureSpec
scheduled_encoder_inputs
encoder_cache
_gather_mm_embeddings
embed_input_ids
```

### 19.7 多模态最终会改变输出对象类型吗？

通常不会。

对于 generation，最终仍然走：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → RequestOutput
```

---

## 20. 和已有专题的关系

多模态链路需要和以下专题联动阅读：

```text
../engine/04_input_processor.md
  解释外部输入如何变成 EngineCoreRequest。

../engine_core/02_request_entry.md
  解释 EngineCoreRequest 如何进入 EngineCore / Scheduler。

../scheduler/07_auxiliary_scheduling_features.md
  解释 encoder input、grammar、spec decode 等辅助调度能力。

../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
  解释 ModelRunner `_preprocess()` 如何合并多模态输入。

../executor_worker_model_runner/07_model_forward_and_logits.md
  解释最终 forward 和 logits 位置。

../config_and_model_loading/03_model_config_and_hf_config.md
  解释 ModelConfig 如何识别模型能力。

../config_and_model_loading/05_model_registry_and_arch_resolution.md
  解释模型 registry 如何解析多模态模型类。
```

---

## 21. 推荐阅读路线

### 21.1 快速建立全局印象

```text
multimodel_overview.md
  → 01_multimodal_role.md
  → 08_model_runner_mm_encoder_flow.md
```

### 21.2 按输入处理链路阅读

```text
multimodel_overview.md
  → 02_multimodal_request_entry.md
  → 03_parser_processor_mapper.md
  → 04_placeholders_and_prompt_alignment.md
  → 05_feature_spec_and_cache.md
```

### 21.3 按调度和缓存链路阅读

```text
multimodel_overview.md
  → 06_encoder_budget_and_scheduler.md
  → 07_encoder_cache_lifecycle.md
  → ../scheduler/07_auxiliary_scheduling_features.md
```

### 21.4 按执行和模型接口阅读

```text
multimodel_overview.md
  → 08_model_runner_mm_encoder_flow.md
  → 09_multimodal_model_interfaces.md
  → 10_multimodal_runtime_interactions.md
```

---

## 22. 文档定位

```text
multimodel_overview.md：
  总览主文档，适合快速建立多模态链路全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读多模态输入解析、placeholder 对齐、feature/cache、scheduler、encoder cache、ModelRunner 和模型接口。
```

---

## 23. 一句话总结

vLLM V1 的多模态链路可以概括为：

```text
Multimodal = 非文本输入处理 + placeholder 对齐 + encoder input 调度 + encoder output 缓存 + inputs_embeds 拼接。
```

最核心的主线是：

```text
multi_modal_data
  → Renderer / MultiModalProcessor
  → MultiModalInput
  → InputProcessor
  → MultiModalFeatureSpec
  → Request.mm_features
  → Scheduler._try_schedule_encoder_inputs()
  → SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._execute_mm_encoder()
  → GPUModelRunner.encoder_cache
  → GPUModelRunner._gather_mm_embeddings()
  → model.embed_input_ids()
  → inputs_embeds
  → model.forward()
  → logits / sampling / output
```

再压缩成一句话：

```text
多模态不是另一套生成引擎，而是在普通 vLLM V1 执行链路前，把 image / audio / video / embeds 转成 decoder forward 可以消费的 embeddings。
```
