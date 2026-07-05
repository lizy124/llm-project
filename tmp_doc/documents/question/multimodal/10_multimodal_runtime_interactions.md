# 10. Multimodal 如何影响输出、KV cache、并行和高级能力？

源码位置：

- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py`
- `code/vllm/vllm/v1/worker/gpu/mm/encoder_runner.py`
- `code/vllm/vllm/v1/worker/gpu/mm/encoder_cache.py`
- `code/vllm/vllm/multimodal/inputs.py`
- `code/vllm/vllm/distributed/`
- `code/vllm/vllm/lora/`
- `code/vllm/vllm/v1/spec_decode/`

本问题关注：多模态链路不是一条孤立支线。它从输入处理阶段进入 `Request.mm_features`，在 Scheduler 阶段影响 encoder input 调度和 encoder cache，在 ModelRunner 阶段影响 `_preprocess()` 和 `inputs_embeds`，然后重新汇入普通 forward、sampling、pooling、OutputProcessor、KV cache、parallelism、LoRA、spec decode、structured output、KV/EC transfer 等运行时能力。

本文按“先定边界，再走主链路，再拆交互点”的方式梳理。

---

## 0. 梳理规划

这篇不是重新解释多模态 processor、placeholder、encoder cache 的每个细节，而是回答多模态进入运行时之后会影响哪些系统边界。

要回答的问题分成 8 组：

```text
1. 多模态运行时的最小主链路是什么？
2. 多模态输入会不会改变 RequestOutput / PoolingRequestOutput？
3. Processor cache、Encoder cache、Decoder KV cache、Prefix cache 分别缓存什么？
4. Scheduler 如何同时调度 decoder token 和 encoder input？
5. ModelRunner 如何把多模态 embedding 合并进普通 forward？
6. TP / PP / DP / CP / EP 场景下，多模态链路的边界在哪里？
7. LoRA、spec decode、structured output、KV transfer、EC transfer、CUDA graph 如何和多模态交互？
8. 哪些能力只是“被多模态输入影响”，哪些能力真正“理解多模态”？
```

阅读顺序建议：

```text
10_multimodal_runtime_interactions.md
  → 05_feature_spec_and_cache.md
  → 06_encoder_budget_and_scheduler.md
  → 07_encoder_cache_lifecycle.md
  → 08_model_runner_mm_encoder_flow.md
  → 09_multimodal_model_interfaces.md
```

如果只想抓住运行时边界，可以先记住这一句：

```text
多模态改变的是输入准备和 encoder cache，输出、sampling、decoder KV cache、structured output 的主职责仍然是文本/hidden-state 运行时职责。
```

---

## 1. 一句话总览

多模态请求在 vLLM V1 里通常走这样的路径：

```text
media / prompt
  → renderer / InputPreprocessor
  → MultiModalFeatureSpec(mm_kwargs + placeholder + hash)
  → EngineCoreRequest / Request.mm_features
  → Scheduler.schedule()
      → scheduled_encoder_inputs
      → EncoderCacheManager allocate / free
      → KVCacheManager allocate decoder slots
  → GPUModelRunner.execute_model()
      → _update_states()
      → _prepare_inputs()
      → _build_attention_metadata()
      → _preprocess()
          → _execute_mm_encoder()
          → _gather_mm_embeddings()
          → embed_input_ids(..., multimodal_embeddings=..., is_multimodal=...)
      → _model_forward(input_ids 或 inputs_embeds)
      → logits / pooler output
  → sample_tokens() 或 _pool()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

关键边界是：

```text
多模态在 forward 前被“降维”为 inputs_embeds 或 encoder_outputs；
forward 后的 logits、pooling、sampling、stop、detokenize、RequestOutput 仍然走通用输出链路。
```

---

## 2. 多模态进入运行时的形态

### 2.1 输入处理阶段生成 `MultiModalFeatureSpec`

入口在 `InputProcessor.process_inputs()`。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:242`

当 `decoder_inputs["type"] == "multimodal"` 时，InputProcessor 会把 renderer / preprocessor 产生的三类信息合并起来：

```text
- decoder_mm_inputs：已经处理好的 mm kwargs；
- decoder_mm_positions：多模态 placeholder 在 prompt 中的位置；
- decoder_mm_hashes：用于 processor cache / encoder cache 的 hash。
```

然后按 placeholder 位置排序，构造 `MultiModalFeatureSpec`。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:332`

结构含义：

```text
MultiModalFeatureSpec
  data        ：该多模态 item 的 processed kwargs；缓存命中或跨进程省传时可以是 None
  modality    ：image / audio / video / prompt_embeds 等
  identifier  ：encoder cache 的 key；启用 tower/connector LoRA 时会带 LoRA 前缀
  mm_position ：placeholder 的 offset、length、is_embed mask
  mm_hash     ：processor cache 的原始 hash；不带 LoRA 前缀
```

对应定义：`code/vllm/vllm/multimodal/inputs.py:301`

一句话：

```text
运行时不直接围绕 image/audio/video 原始对象调度，而是围绕 MultiModalFeatureSpec 调度。
```

### 2.2 `Request` 把多模态视为 encoder inputs

`Request` 初始化时保存 `mm_features`。

源码位置：`code/vllm/vllm/v1/request.py:156`

它提供三个和后续调度强相关的属性：

```text
num_encoder_inputs = len(mm_features)
has_encoder_inputs = num_encoder_inputs > 0
get_num_encoder_embeds(input_id) = mm_features[input_id].mm_position.get_num_embeds()
```

源码位置：`code/vllm/vllm/v1/request.py:258`

这说明 Scheduler 并不关心“这是图片还是音频”，它主要关心：

```text
- 这个 request 有没有 encoder inputs；
- 每个 encoder input 占多少 encoder embedding cache 空间；
- 当前 decoder token window 是否覆盖某个多模态 placeholder；
- 该 item 的 encoder output 是否已经在 cache 中。
```

---

## 3. 输出链路：多模态不改变输出对象类型

### 3.1 generation 请求仍然输出 `RequestOutput`

多模态 generation 请求最终仍然是文本 generation：

```text
mm encoder / inputs_embeds
  → decoder forward
  → logits
  → sampler
  → sampled token ids
  → detokenizer
  → CompletionOutput
  → RequestOutput
```

`OutputProcessor.process_outputs()` 只消费 `EngineCoreOutput`，核心字段仍然是：

```text
new_token_ids
pooling_output
finish_reason
stop_reason
kv_transfer_params
routed_experts
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:576`

如果 `pooling_output is None`，它会走普通文本输出路径：

```text
1. detokenizer.update(new_token_ids, ...)
2. logprobs_processor.update_from_output(...)
3. RequestState.make_request_output(...)
4. CompletionOutput / RequestOutput
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:635`

所以：

```text
输入里有 image/audio/video，不代表输出对象会变成 MultimodalOutput。
对 API 上层来说，它仍然是文本 completion，只是 prompt 的一部分来自多模态 embedding。
```

### 3.2 pooling / embedding 请求仍然输出 `PoolingRequestOutput`

如果模型 runner type 是 pooling，ModelRunner 会走 `_pool()`，最终在 `ModelRunnerOutput.pooler_output` 中回传 CPU 侧 pooling tensor。

`OutputProcessor` 判断 `pooling_output is not None` 后，会构造 `PoolingOutput` 和 `PoolingRequestOutput`。

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:312`
- `code/vllm/vllm/v1/engine/output_processor.py:347`
- `code/vllm/vllm/v1/engine/output_processor.py:413`

链路可以记成：

```text
multimodal inputs
  → inputs_embeds / encoder_outputs
  → pooling model forward
  → pooler output tensor
  → PoolingOutput
  → PoolingRequestOutput
```

输出对象仍然只表达“模型输出的文本或向量”，不表达“输入的媒体对象”。

### 3.3 `ModelRunnerOutput` 只多了 EC connector 状态

`ModelRunnerOutput` 的核心字段仍然是 token / logprobs / pooler output / connector output。

源码位置：`code/vllm/vllm/v1/outputs.py:231`

其中和多模态直接相关的是：

```text
ec_connector_output: ECConnectorOutput | None
```

源码位置：`code/vllm/vllm/v1/outputs.py:224`

这个字段表示 encoder cache transfer 的完成状态：

```text
finished_sending
finished_recving
```

它不是用户可见的“多模态输出”，而是 Scheduler 和 connector 做状态闭环的运行时信号。

---

## 4. Cache 关系：四类 cache 不要混在一起

多模态运行时最容易混淆的是 cache。这里要分成四类：

```text
Processor cache：缓存 media processor 输出。
Encoder cache：缓存多模态 encoder output / embedding。
Decoder KV cache：缓存 decoder self-attention KV。
Prefix cache：复用 decoder token prefix 对应的 KV blocks。
```

### 4.1 Processor cache

Processor cache 位于输入处理更前面，服务对象是 media processor 输出。

`InputProcessor.inject_into_mm_cache()` 可以把外部已经预处理过的 mm kwargs 注入 processor cache，避免后续重复 processor 处理。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:183`

它缓存的是：

```text
raw media / external processed tensor
  → HF processor / multimodal processor 后的 mm_kwargs
```

它不缓存：

```text
- 模型 encoder output；
- decoder attention KV；
- sampled tokens；
- RequestOutput。
```

### 4.2 Encoder cache

Encoder cache 由 Scheduler 侧的 `EncoderCacheManager` 管状态，由 Worker / ModelRunner 侧的 `encoder_cache` 管实际 tensor。

Scheduler 侧：

```text
EncoderCacheManager
  cached: mm_hash -> set(req_id)
  request_cached_ids: req_id -> set(input_id)
  freeable: mm_hash -> num_encoder_embeds
  freed: list[mm_hash]
```

源码位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

Worker 侧在 `GPUModelRunner` 中保存：

```text
self.encoder_cache: dict[str, torch.Tensor]
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:533`

新版拆分辅助类里也有同样的概念：

```text
EncoderCache.mm_features: req_id -> list[MultiModalFeatureSpec]
EncoderCache.encoder_outputs: mm_hash -> torch.Tensor
```

源码位置：`code/vllm/vllm/v1/worker/gpu/mm/encoder_cache.py:8`

这类 cache 缓存的是：

```text
image/audio/video/prompt_embeds item
  → model.embed_multimodal(...) 或 passthrough prompt_embeds
  → encoder output tensor
```

### 4.3 Decoder KV cache

Decoder KV cache 由 `KVCacheManager` 管理 block 分配、prefix hit、block 生命周期。

源码位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:110`

它缓存的是 decoder self-attention 的 key/value：

```text
decoder token positions
  → attention layer K/V blocks
```

多模态会间接影响 decoder KV cache，因为 placeholder token / multimodal token 会占 prompt 位置，进而影响：

```text
- request.num_tokens；
- block_hashes；
- prefix cache hit length；
- num_computed_tokens；
- allocate_slots() 的 token 数；
- attention metadata 的 seq_lens / block table / slot mapping。
```

但 decoder KV cache 不保存图片 embedding 本身。

### 4.4 Prefix cache

Prefix cache 是 `KVCacheManager.get_computed_blocks()` / coordinator 找到的 decoder KV cache hit。

源码位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:202`

它按 request block hashes 查找最长 prefix hit：

```text
request.block_hashes
  → find_longest_cache_hit(...)
  → computed KV blocks + num_computed_tokens
```

多模态对 prefix cache 的影响主要来自两个方面：

```text
1. placeholder token 是 prompt token 序列的一部分，会影响 block hash；
2. 启用 tower/connector LoRA 时，InputProcessor 会把 LoRA 信息加进 mm identifier，避免 encoder cache 错误命中。
```

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:165`

注意：

```text
prefix cache 命中 decoder KV，不等于 encoder cache 命中。
一个 request 可以 decoder prefix cache 命中，但对应 mm encoder output 仍需确认是否在 encoder cache / EC connector 中。
```

---

## 5. Scheduler：同时调度 decoder token 和 encoder input

### 5.1 Scheduler 初始化 encoder budget 和 cache manager

Scheduler 初始化时会判断模型是否支持 multimodal inputs。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:199`

如果支持，则构造 `MultiModalBudget`，并计算：

```text
max_num_encoder_input_tokens
encoder_cache_size
encoder_cache_manager
```

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:217`

对应 budget 计算在：

```text
compute_mm_encoder_budget(...)
  → encoder_compute_budget
  → encoder_cache_size
```

源码位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269`

这里的关键点是：

```text
Scheduler 不执行 encoder；Scheduler 只决定本轮哪些 encoder input 可以被执行或复用。
```

### 5.2 `SchedulerOutput` 携带多模态调度结果

`SchedulerOutput` 中多模态相关字段主要是：

```text
scheduled_encoder_inputs: dict[str, list[int]]
free_encoder_mm_hashes: list[str]
ec_connector_metadata: ECConnectorMetadata | None
```

源码位置：`code/vllm/vllm/v1/core/sched/output.py:180`

含义是：

```text
scheduled_encoder_inputs：
  req_id -> 本轮需要处理的 encoder input 下标。

free_encoder_mm_hashes：
  worker 侧可以从 encoder_cache 删除的 mm_hash。

ec_connector_metadata：
  encoder cache transfer 需要的 connector 元数据。
```

### 5.3 `_try_schedule_encoder_inputs()` 决定本轮 encoder input

Scheduler 在调度 running / waiting request 时，如果 `request.has_encoder_inputs`，会调用 `_try_schedule_encoder_inputs()`。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:480`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1279`

这个函数的判断条件是：

```text
一个 encoder input 会被本轮调度，当且仅当：
- 它的 placeholder range 和本轮要计算的 decoder token window 有重叠；
- 它的 encoder output 还没有在 encoder cache 中；
- 它不在 remote encoder cache 中，或者需要通过 EC connector 加载；
- encoder compute budget 足够；
- encoder cache 空间足够。
```

源码注释对应：`code/vllm/vllm/v1/core/sched/scheduler.py:1291`

如果当前 encoder input 因 budget/cache 不能调度，Scheduler 会缩短 `num_new_tokens`，让 decoder token 只推进到该多模态 placeholder 之前。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1382`

这就是多模态对调度最直接的影响：

```text
decoder token budget 不再只受 max_num_scheduled_tokens / KV blocks 影响，
还会被 encoder compute budget 和 encoder cache capacity 截断。
```

### 5.4 Scheduler 分配 encoder cache，但不保存 tensor

当 `encoder_inputs_to_schedule` 不为空时，Scheduler 会：

```text
scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
encoder_cache_manager.allocate(request, input_id)
ec_connector.update_state_after_alloc(request, input_id)  # 如果启用 EC connector
```

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:599`

这一步只是状态分配：

```text
Scheduler 预留 encoder cache 位置；
ModelRunner 后续真正执行 encoder，并把 tensor 放进 worker 侧 encoder_cache。
```

### 5.5 encoder cache 释放通过 `free_encoder_mm_hashes` 通知 worker

当 encoder cache manager 决定淘汰某些无引用 entry 时，会把 mm_hash 放进 `freed`。

源码位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:174`

每轮 SchedulerOutput 会携带：

```text
free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes()
```

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1071`

ModelRunner 在 `_update_states()` 中执行实际删除：

```text
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1158`

这说明 encoder cache 的生命周期是两段式的：

```text
Scheduler：决定引用、空间、淘汰名单。
Worker：保存和删除真实 GPU tensor。
```

---

## 6. ModelRunner：多模态在 `_preprocess()` 汇入 forward

### 6.1 多模态执行发生在 forward 前

`GPUModelRunner.execute_model()` 的主线是：

```text
_update_states()
_prepare_inputs()
_determine_batch_execution_and_padding()
_build_attention_metadata()
_preprocess()
_model_forward()
postprocess logits / pooling
sample_tokens()
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`

多模态真正接入的位置是 `_preprocess()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

### 6.2 decoder-only 多模态模型：统一转成 `inputs_embeds`

当满足：

```text
supports_mm_inputs
is_first_rank
not is_encoder_decoder
```

`_preprocess()` 会：

```text
1. maybe_get_ec_connector_output(...)
2. _execute_mm_encoder(scheduler_output)
3. _gather_mm_embeddings(scheduler_output)
4. model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)
5. 写入 self.inputs_embeds
6. _prepare_mm_inputs(num_input_tokens)
7. model_kwargs 合并 _init_model_kwargs() 和 _extract_mm_kwargs()
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`

注释里说得很清楚：

```text
为了统一 token ids 和 soft tokens，多模态模型总是使用 embeddings 作为输入，即使其中一部分输入是普通文本 token。
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3456`

所以普通文本模型和多模态模型的 forward 输入差异是：

```text
文本模型：
  input_ids + positions → model forward

decoder-only 多模态模型：
  inputs_embeds + positions → model forward
```

### 6.3 `_execute_mm_encoder()` 执行或注入 encoder output

`_execute_mm_encoder()` 先从 `SchedulerOutput.scheduled_encoder_inputs` 收集本轮要执行的 item。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2866`

它会把每个 item 转成：

```text
mm_hashes: list[str]
mm_kwargs: list[(modality, MultiModalKwargsItem)]
mm_lora_refs: list[(req_id, PlaceholderRange)]
```

然后分三类处理：

```text
prompt_embeds：
  不跑 encoder，直接把 embedding tensor 放进 encoder_cache。

普通 image/audio/video：
  按 modality batch 后调用 model.embed_multimodal(**mm_kwargs_batch)。

需要特殊处理的视频 / pruning 场景：
  可能拆成 micro-batch 或顺序执行，降低峰值内存。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2899`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3013`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3032`

执行后会按 `mm_hash` 写入 worker 侧 encoder cache：

```text
self.encoder_cache[mm_hash] = output
maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3092`

### 6.4 `_gather_mm_embeddings()` 把 encoder output 对齐回 placeholder window

`_gather_mm_embeddings()` 遍历当前 persistent batch 的 request 顺序，用 `num_computed_tokens` 和本轮 `num_scheduled_tokens` 算出每个 request 本轮 token window。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`

它用 `get_mm_features_in_window()` 找出和本轮 window 重叠的 `mm_features`，再用 `PlaceholderRange.get_embeds_indices_in_range()` 找出需要取 encoder output 的哪一段。

关键步骤：

```text
mm_feature.mm_position.offset / length
  → 当前 chunk 与 placeholder 的交集
  → curr_embeds_start / curr_embeds_end
  → encoder_cache[mm_hash][...]
  → mm_embeds.append(...)
  → is_mm_embed 对应位置置 True
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3123`

如果 `is_embed` mask 存在，则不是 placeholder range 内所有位置都替换成 embedding，而是只替换 mask 标记的位置。

源码位置：`code/vllm/vllm/multimodal/inputs.py:141`

### 6.5 encoder-decoder 多模态模型：传 `encoder_outputs`

如果是 encoder-decoder 模型，处理方式更接近传统 seq2seq：

```text
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3552`

区别是：

```text
decoder-only multimodal：
  把 mm encoder output splice 进 decoder input embeddings。

encoder-decoder：
  encoder output 作为 model_kwargs["encoder_outputs"] 传给 decoder。
```

---

## 7. 和 KV cache / attention metadata 的关系

### 7.1 Scheduler 同时分配 decoder KV blocks

在 schedule 阶段，encoder input 调度完成后，Scheduler 仍然要为 decoder token 分配 KV slots。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:521`

主调用是：

```text
kv_cache_manager.allocate_slots(
  request,
  num_new_tokens,
  num_lookahead_tokens=...,  # spec decode
)
```

多模态对这里的影响是：

```text
- 如果 encoder input 不能处理，num_new_tokens 会被截断；
- placeholder token 算在 decoder 序列位置中；
- 多模态 prompt 仍然要占用 decoder attention 的位置和 KV block；
- encoder cache 命中不等于 decoder KV cache 命中。
```

### 7.2 ModelRunner 构造 attention metadata 时已经看不到“媒体对象”

`GPUModelRunner.execute_model()` 在 `_preprocess()` 前先 `_prepare_inputs()` 和 `_build_attention_metadata()`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4128`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`

attention backend 看到的是：

```text
- positions
- seq_lens
- query_start_loc
- block table
- slot mapping
- cascade attention prefix lengths
- spec decode metadata
```

它不直接处理 image/audio/video；多模态只通过 token positions、placeholder 长度、inputs_embeds 和 encoder_outputs 间接影响 attention。

### 7.3 encoder cache 和 decoder KV cache 的生命周期不同

encoder cache：

```text
key = mm_hash / identifier
value = encoder output tensor
释放 = Scheduler 生成 free_encoder_mm_hashes，Worker pop encoder_cache
```

decoder KV cache：

```text
key = token block hash / request blocks
value = attention KV blocks
释放 = Scheduler / KVCacheManager 管 block 生命周期，Worker 用 slot mapping 写入物理 block
```

所以不要把下面两件事混为一谈：

```text
- 图片 encoder output 是否已缓存；
- 文本/placeholder prefix 对应的 decoder KV 是否已缓存。
```

---

## 8. 和并行的关系

### 8.1 Tensor Parallel：模型内部并行，多模态以模型接口为边界

TP 下，多模态 encoder、projector、language model 是否参与 tensor parallel，取决于具体模型实现和 layer 分布。

运行时层看到的边界是：

```text
model.embed_multimodal(**mm_kwargs_batch)
model.embed_input_ids(input_ids, multimodal_embeddings=..., is_multimodal=...)
model.compute_logits(hidden_states)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3085`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3472`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

也就是说：

```text
TP 是模型层和 distributed state 的职责；
多模态 runtime 只保证把 mm kwargs / mm embeddings 送到正确的模型接口。
```

### 8.2 Pipeline Parallel：首个 stage 处理输入，多模态主要在 first rank

`_preprocess()` 里明确检查 `get_pp_group().is_first_rank`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3440`

decoder-only 多模态输入只在 first pipeline rank 上执行 encoder / embedding splice：

```text
supports_mm_inputs and is_first_rank and not is_encoder_decoder
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`

非 first rank 不重新处理多模态输入，而是接收前一 stage 的 intermediate tensors：

```text
if is_first_rank:
    intermediate_tensors = None
else:
    intermediate_tensors = sync_and_gather_intermediate_tensors(...)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3544`

所以 PP 下的边界是：

```text
first PP rank：
  media encoder / inputs_embeds / input embedding preparation

middle / last PP ranks：
  intermediate tensors + hidden states + logits / sampling output
```

### 8.3 Data Parallel：请求路由决定 cache 命中范围

DP 下，每个 engine / rank 负责一部分请求。多模态 processor cache、encoder cache、decoder KV cache 的命中都和请求被路由到哪个 engine 有关。

输入阶段支持指定 `data_parallel_rank`，并校验 rank 范围。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:259`

ModelRunner 还会在 batch 执行中协调 DP batch 信息。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:197`

需要记住：

```text
DP 不会天然让每个 rank 共享本地 encoder_cache。
如果要跨 engine / rank 复用 encoder output，需要 EC connector 这类 transfer 机制。
```

### 8.4 Context Parallel / DCP / PCP：多模态表现为序列位置和 embedding

context parallel 关心的是 token 序列如何被切分、KV 如何分片、attention metadata 如何构造。

多模态进入这个层面时已经主要表现为：

```text
- placeholder 占据的 token positions；
- inputs_embeds 中的 multimodal embedding；
- encoder-decoder 的 encoder_outputs；
- attention metadata 的 seq_lens / slot mapping。
```

源码里相关状态包括：

```text
decode_context_parallel_size
prefill_context_parallel_size
get_dcp_group()
get_total_cp_world_size()
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:166`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:44`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:197`

### 8.5 Expert Parallel / MoE：多模态不改变 routed experts 输出收集位置

如果模型包含 MoE，routing 数据由 `RoutedExpertsCapturer` / routed experts 相关逻辑收集，最终放入 `ModelRunnerOutput.routed_experts`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:60`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4625`
- `code/vllm/vllm/v1/outputs.py:272`

多模态输入会影响 hidden states，从而可能影响专家路由结果；但 routed experts 的采集和回传仍是通用 MoE 运行时能力。

---

## 9. 和 LoRA 的关系

### 9.1 LoRA 会影响 encoder cache key

`InputProcessor._get_mm_identifier()` 专门处理这个问题。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:165`

逻辑是：

```text
如果未启用 tower/connector LoRA：
  identifier = mm_hash

如果启用 tower/connector LoRA 且请求带 LoRA：
  identifier = f"{lora_name}:{mm_hash}"
```

原因是：

```text
同一张图片在不同 LoRA adapter 下，tower / connector encoder output 可能不同。
如果 encoder cache key 仍只用原始 mm_hash，就可能错误复用别的 LoRA 下的 encoder output。
```

### 9.2 ModelRunner 会为 multimodal encoder 单独设置 LoRA mapping

`_execute_mm_encoder()` 中，如果 LoRA manager 支持 tower / connector LoRA，会为 encoder batch 构造独立的 mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2941`

这里区分两类：

```text
TOWER LoRA：
  作用于 multimodal tower encoder token。

CONNECTOR LoRA：
  作用于 tower output 到 language embedding 空间之间的 connector/projector。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2966`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2979`

关键点：

```text
主 decoder batch 的 LoRA mapping 和 multimodal encoder batch 的 LoRA mapping 不是同一个形状。
因为 encoder batch 按多模态 item 组织，而 decoder batch 按 request/token 组织。
```

---

## 10. 和 Spec Decode 的关系

### 10.1 spec decode 调度的是 decoder token，不是 media encoder

SchedulerOutput 同时有：

```text
scheduled_spec_decode_tokens
scheduled_encoder_inputs
```

源码位置：`code/vllm/vllm/v1/core/sched/output.py:197`

这两个字段职责不同：

```text
scheduled_spec_decode_tokens：
  draft / target 之间的 speculative token。

scheduled_encoder_inputs：
  当前 decoder token window 覆盖到的多模态 encoder input。
```

Scheduler 调度 running request 时会分别处理：

```text
_try_schedule_encoder_inputs(...)
然后处理 request.spec_token_ids
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:480`
- `code/vllm/vllm/v1/core/sched/scheduler.py:581`

### 10.2 多模态必须先满足 encoder cache，spec decode 才能安全推进

`_try_schedule_encoder_inputs()` 如果发现 encoder input 无法调度，会把 `num_new_tokens` 截断到 encoder input 前面，或者直接置 0。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1382`

这对 spec decode 很关键：

```text
不能让 decoder / drafter 越过一个尚未准备好的 multimodal placeholder。
否则 _gather_mm_embeddings() 会在 encoder_cache 中找不到对应 mm_hash。
```

ModelRunner 里确实会 assert：

```text
encoder_output = self.encoder_cache.get(mm_hash, None)
assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3149`

### 10.3 draft proposer 发生在 sampling 后

`sample_tokens()` 的顺序是：

```text
1. apply_grammar_bitmask(...)
2. _sample(logits, spec_decode_metadata)
3. _update_states_after_model_execute(...)
4. propose_draft_token_ids(...)
5. _bookkeeping_sync(...)
6. ModelRunnerOutput(...)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`

所以 spec decode 的草稿 token 逻辑看到的是：

```text
已经完成 multimodal embedding splice 和 target model forward 后的 hidden states / logits。
```

它不是一个独立的多模态 processor。

### 10.4 draft model 是否支持多模态要单独判断

从 runtime 边界看，多模态输入在 target model forward 前已经处理完成；但如果 speculative config 使用 draft model / EAGLE / MTP 等 proposer，是否能正确处理含多模态 prompt 的请求，需要看对应 proposer 和模型能力。

源码入口包括：

```text
vllm/v1/spec_decode/draft_model.py
vllm/v1/spec_decode/eagle.py
vllm/v1/spec_decode/step3p5.py
vllm/v1/spec_decode/gemma4.py
```

在 `GPUModelRunner.sample_tokens()` 里，draft proposer 消费的是 sampled token、hidden states、sample hidden states、spec decode metadata、attention metadata 和 slot mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4481`

一句话：

```text
多模态影响 spec decode 的前提输入和序列位置；spec decode 本身仍然是 decoder token 级能力。
```

---

## 11. 和 Structured Output 的关系

Structured output 的 grammar bitmask 在 `sample_tokens()` 里应用到 logits。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452`

调用是：

```text
apply_grammar_bitmask(
  scheduler_output,
  grammar_output,
  input_batch,
  logits,
)
```

这说明 structured output 的约束位置是：

```text
multimodal inputs → model forward → logits → grammar bitmask → sampler
```

它不约束图片内容本身，也不改变多模态 encoder output。

所以：

```text
多模态输入可以影响模型 logits；
structured output 只在 logits 层限制可采样 token。
```

---

## 12. 和 KV transfer / EC transfer 的关系

### 12.1 KV transfer 是 decoder KV cache transfer

`KVConnectorOutput` 记录的是 decoder KV connector 的状态。

源码位置：`code/vllm/vllm/v1/outputs.py:196`

ModelRunner 在 forward 外层通过 `maybe_get_kv_connector_output()` 管理 KV connector。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4315`

这类 transfer 的对象是：

```text
decoder KV blocks
```

它不传输 multimodal encoder output。

### 12.2 EC transfer 是 encoder cache transfer

EC connector 是专门给 encoder cache 用的。

`ECConnectorModelRunnerMixin` 提供两类方法：

```text
maybe_save_ec_to_connector(encoder_cache, mm_hash)
maybe_get_ec_connector_output(scheduler_output, encoder_cache, ...)
```

源码位置：`code/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py:24`

consumer 会在上下文里 load encoder cache：

```text
if ec_connector.is_consumer:
    ec_connector.start_load_caches(encoder_cache, **kwargs)
```

源码位置：`code/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py:67`

结束时回传：

```text
finished_sending
finished_recving
```

源码位置：`code/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py:74`

### 12.3 KV transfer 和 EC transfer 是两条平行链路

可以这样区分：

```text
KV transfer：
  decoder token KV blocks
  → 与 prefix cache / P-D disaggregation / offload 相关

EC transfer：
  multimodal encoder outputs
  → 与 mm_hash / encoder cache / image/audio/video embedding 相关
```

在 `ModelRunnerOutput` 里也分成两个字段：

```text
kv_connector_output
ec_connector_output
```

源码位置：`code/vllm/vllm/v1/outputs.py:262`

---

## 13. 和 CUDA graph / compilation 的关系

### 13.1 多模态倾向于让输入 shape 更动态

多模态输入的动态性来自：

```text
- 图片分辨率不同；
- 视频帧数不同；
- audio 长度不同；
- placeholder length 不同；
- encoder output feature_size 可能不同；
- 多模态 item 数量不同；
- pruning / dynamic resolution 可能改变实际 embedding 数量。
```

因此 `_execute_mm_encoder()` 中会处理 fixed feature size 和 dynamic feature size 两种输出形态。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3062`

### 13.2 encoder CUDA graph 是单独优化点

`_execute_mm_encoder()` 中，如果 `encoder_cudagraph_manager` 支持当前 modality，会优先尝试用 encoder CUDA graph 执行。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3073`

如果不支持或返回 None，则回退到：

```text
model.embed_multimodal(**mm_kwargs_batch)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3085`

### 13.3 encoder-decoder 首次 encoder 输入会 skip compiled

ModelRunner 在设置 forward context 时会判断：

```text
has_encoder_input = model_config.is_encoder_decoder and num_encoder_reqs > 0
skip_compiled = has_encoder_input
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4290`

原因是 encoder-decoder 模型第一步带 encoder input，纯 decode step 不带 encoder input，编译形态不同。

### 13.4 多模态模型通常用 `inputs_embeds`，会影响 graph capture 边界

文本模型为了性能通常直接把 `input_ids` 送进模型，让 embedding layer 包在 CUDA graph 内。

多模态模型为了合并 soft tokens，通常在 `_preprocess()` 里先构造 `inputs_embeds`，再 forward。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3456`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3527`

这就是为什么多模态和 prompt_embeds 场景经常对 CUDA graph / compilation 更敏感。

---

## 14. 和 pooling / late interaction 的关系

Pooling 模型在 request state 初始化时会读取并应用 pooling params。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1215`

ModelRunner forward 后，如果 `self.is_pooling_model`，不会走 logits + sampler，而是直接 `_pool()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4345`

所以多模态 pooling 的链路是：

```text
mm inputs
  → encoder / inputs_embeds
  → model forward hidden states
  → pooler
  → ModelRunnerOutput.pooler_output
  → PoolingRequestOutput
```

如果 late interaction runner 被启用，请求完成或注册时会维护额外状态，但它仍然属于 pooling / retrieval 类输出逻辑，不改变多模态 encoder 的基本边界。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:535`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1138`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1239`

---

## 15. 关键交互总表

```text
输入处理：
  多模态 raw input / processed input
    → MultiModalFeatureSpec
    → Request.mm_features

Scheduler：
  mm_features
    → _try_schedule_encoder_inputs()
    → scheduled_encoder_inputs
    → EncoderCacheManager.allocate/free
    → SchedulerOutput.free_encoder_mm_hashes

ModelRunner preprocess：
  scheduled_encoder_inputs
    → _execute_mm_encoder()
    → encoder_cache[mm_hash]
    → _gather_mm_embeddings()
    → inputs_embeds / encoder_outputs

Forward：
  inputs_embeds 或 encoder_outputs
    → model forward
    → hidden states
    → logits 或 pooler output

Sampling / pooling：
  logits
    → grammar bitmask / sampler / spec decode
    → sampled token ids

Output：
  sampled token ids / pooler output
    → ModelRunnerOutput
    → EngineCoreOutput
    → RequestOutput / PoolingRequestOutput

Cache：
  processor cache：mm_kwargs
  encoder cache：mm_hash -> encoder output
  decoder KV cache：token positions -> KV blocks
  prefix cache：block hashes -> computed KV blocks

Transfer：
  KV connector：decoder KV blocks
  EC connector：encoder cache tensors
```

---

## 16. 常见误区

### 16.1 “多模态输出”不是特殊输出类型

大多数多模态 generation 请求最终仍然输出文本 token。

```text
图片问答、音频转写、视频理解：
  输入是多模态；输出通常仍然是 token stream。
```

除非模型 runner 是 pooling，否则不会因为输入包含图片而产生新的输出对象类型。

### 16.2 encoder cache 命中不等于 prefix cache 命中

```text
encoder cache 命中：
  这张图 / 这段音频的 encoder output 已经算过。

prefix cache 命中：
  decoder prompt prefix 的 KV blocks 已经算过。
```

它们的 key、value、生命周期、transfer 机制都不同。

### 16.3 `mm_hash` 和 `identifier` 不总是同一个东西

`mm_hash` 是原始 processor cache hash。

`identifier` 是 encoder cache key，在 tower/connector LoRA 场景可能加 LoRA 前缀。

源码位置：`code/vllm/vllm/v1/engine/input_processor.py:165`

### 16.4 Scheduler 不执行 encoder

Scheduler 只生成：

```text
scheduled_encoder_inputs
free_encoder_mm_hashes
ec_connector_metadata
```

真正的 encoder forward 在 ModelRunner 的 `_execute_mm_encoder()`。

### 16.5 structured output 不约束多模态 encoder

structured output 约束 logits 采样，不约束图片/audio/video 的 encoder output。

---

## 17. 推荐源码阅读路线

### 17.1 从输入到 Request

```text
vllm/v1/engine/input_processor.py
  → process_inputs()
  → _get_mm_identifier()
  → MultiModalFeatureSpec
  → EngineCoreRequest
  → Request.mm_features
```

### 17.2 从 Scheduler 到 encoder cache

```text
vllm/v1/core/sched/scheduler.py
  → Scheduler.__init__()
  → schedule()
  → _try_schedule_encoder_inputs()
  → EncoderCacheManager.allocate()
  → SchedulerOutput.scheduled_encoder_inputs
```

### 17.3 从 ModelRunner 到 forward

```text
vllm/v1/worker/gpu_model_runner.py
  → execute_model()
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _execute_mm_encoder()
  → _gather_mm_embeddings()
  → _model_forward()
```

### 17.4 从 forward 到输出

```text
vllm/v1/worker/gpu_model_runner.py
  → logits / _pool()
  → sample_tokens()
  → ModelRunnerOutput

vllm/v1/core/sched/scheduler.py
  → update_from_output()

vllm/v1/engine/output_processor.py
  → process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 18. 一句话总结

多模态在 vLLM V1 运行时的核心影响点是：

```text
输入侧：把 media 变成 MultiModalFeatureSpec；
调度侧：增加 encoder input budget / encoder cache / scheduled_encoder_inputs；
执行侧：在 _preprocess() 中执行或复用 mm encoder output，并合并成 inputs_embeds 或 encoder_outputs；
输出侧：重新回到普通 logits / pooling / sampling / OutputProcessor 链路。
```

最重要的边界是：

```text
多模态不是一套独立输出系统；
它是输入准备、encoder cache 和模型 forward 前处理对通用生成/池化运行时的一次扩展。
```
