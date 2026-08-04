# 08. ModelRunner 如何执行多模态 encoder？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/multimodal/inputs.py`
- `code/vllm/vllm/multimodal/utils.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/models/`

本文关注：Scheduler 已经把本轮需要处理的多模态 encoder input 放进 `SchedulerOutput.scheduled_encoder_inputs` 后，`GPUModelRunner` 如何在 `_preprocess()` 阶段执行 multimodal encoder，如何复用 encoder cache，如何按 placeholder span 收集多模态 embedding，并最终把它和文本 token embedding 合并成 `inputs_embeds` 进入统一的模型 forward。

它不是逐个模型文件的视觉塔实现说明，而是回答执行层里最关键的问题：

```text
1. SchedulerOutput 里和多模态 encoder 相关的字段是什么？
2. ModelRunner 什么时候执行 multimodal encoder？
3. _execute_mm_encoder() 的输入来自哪里，输出缓存在哪里？
4. _gather_mm_embeddings() 如何把 encoder output 对齐到本轮 token window？
5. inputs_embeds 是怎么由 text embedding + mm embedding 合并出来的？
6. prompt_embeds、encoder-decoder、pipeline parallel、LoRA、EC connector 分别怎么接入？
```

---

## 0. 梳理规划

本文按“先定边界，再走主链路，再拆关键阶段，最后总结特殊分支”的顺序组织。

要回答的问题分成 8 组：

```text
1. 多模态 encoder flow 的边界是什么？
2. SchedulerOutput 怎样描述本轮需要 encode 的多模态 item？
3. Worker 侧请求状态里如何保存 mm_features？
4. _preprocess() 如何选择 text-only / multimodal / prompt_embeds / encoder-decoder 路径？
5. _execute_mm_encoder() 如何 batch 多模态输入、调用模型 encoder、写入 encoder cache？
6. _gather_mm_embeddings() 如何按 placeholder range 和本轮 token window 收集 embedding？
7. embed_input_ids() 如何把文本 token embedding 和多模态 embedding merge 成 inputs_embeds？
8. prompt_embeds、LoRA、EC connector、M-RoPE、PP 场景有什么特殊处理？
```

阅读顺序建议：

```text
08_model_runner_mm_encoder_flow.md
  → 03_multimodal_scheduler_encoder_cache.md
  → 04_multimodal_request_mm_features.md
  → ../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
  → ../executor_worker_model_runner/07_model_forward_and_logits.md
```

如果只想抓住主线，先看第 2、3、4、5、6 节。

---

## 1. 一句话回答

```text
GPUModelRunner 在模型 forward 前的 _preprocess() 阶段处理多模态输入：
先根据 SchedulerOutput.scheduled_encoder_inputs 找到本轮要 encode 的 mm_features，
执行或复用 multimodal encoder output，写入 encoder_cache；
再按每个 request 的 placeholder range 和本轮 token window 收集 mm embeddings；
最后调用模型的 embed_input_ids() 把 text embedding 与 mm embeddings 合并成 inputs_embeds，
然后走统一的 _model_forward()。
```

也可以压缩成：

```text
scheduled_encoder_inputs
  → _execute_mm_encoder()
  → encoder_cache[mm_hash]
  → _gather_mm_embeddings()
  → model.embed_input_ids(..., multimodal_embeddings, is_multimodal)
  → inputs_embeds
  → _model_forward()
```

---

## 2. 入口总览

多模态 encoder 不在 `execute_model()` 一开始就跑，而是在输入张量和 attention metadata 准备完成后、真正模型 forward 前的 `_preprocess()` 里跑。

最小主链路是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput.scheduled_encoder_inputs
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      → _update_states()
      → _prepare_inputs()
      → _build_attention_metadata()
      → _preprocess()
          → maybe_get_ec_connector_output(...)
          → _execute_mm_encoder(scheduler_output)
              → _batch_mm_inputs_from_scheduler(...)
              → group_and_batch_mm_kwargs(...)
              → model.embed_multimodal(...)
              → encoder_cache[mm_hash] = encoder_output
          → _gather_mm_embeddings(scheduler_output)
              → get_mm_features_in_window(...)
              → encoder_cache[mm_hash]
              → is_mm_embed mask
          → model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )
          → inputs_embeds
      → _model_forward(input_ids=None 或 input_ids + inputs_embeds, ...)
```

对应关键源码入口：

- `code/vllm/vllm/v1/core/sched/output.py:183`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1367`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1162`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2913`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2956`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3165`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3462`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3476`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`
- `code/vllm/vllm/model_executor/models/interfaces.py:160`

---

## 3. SchedulerOutput 传了什么

### 3.1 `scheduled_encoder_inputs`

`SchedulerOutput` 里和本问题最直接相关的是：

源码位置：`code/vllm/vllm/v1/core/sched/output.py:203`

```text
scheduled_encoder_inputs: dict[str, list[int]]
```

它的含义是：

```text
req_id -> 本轮需要处理的 encoder input 下标列表
```

例如：

```text
scheduled_encoder_inputs = {
  "req-1": [0, 1],
  "req-7": [0],
}
```

表示：

```text
- req-1 的第 0 个、第 1 个多模态 item 需要在本轮跑 encoder；
- req-7 的第 0 个多模态 item 需要在本轮跑 encoder。
```

注意这里传的是下标，不是图像 tensor 本身。

真正的多模态数据在 worker 侧缓存的 request state 里：

```text
self.requests[req_id].mm_features[mm_input_id]
```

### 3.2 `free_encoder_mm_hashes`

另一个相关字段是：

```text
free_encoder_mm_hashes: list[str]
```

它表示 Scheduler 认为某些 encoder output 已经不需要了，worker 可以从 GPU 侧 `encoder_cache` 释放。

`GPUModelRunner._update_states()` 会在每轮开始时处理它：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1199`

```text
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

也就是说：

```text
Scheduler 管 encoder cache 的生命周期决策；
ModelRunner 管 GPU 侧 encoder output 的实际存取。
```

### 3.3 `NewRequestData.mm_features`

新请求第一次被调度时，`SchedulerOutput.scheduled_new_reqs` 会携带 `NewRequestData`。

源码位置：`code/vllm/vllm/v1/core/sched/output.py:32`

其中包含：

```text
mm_features: list[MultiModalFeatureSpec]
prompt_embeds: torch.Tensor | None
prompt_is_token_ids: list[bool] | None
```

`GPUModelRunner._update_states()` 会把这些内容保存成 worker 侧的 `CachedRequestState`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1265`

核心关系是：

```text
NewRequestData.mm_features
  → CachedRequestState.mm_features
  → self.requests[req_id].mm_features
  → _execute_mm_encoder() / _gather_mm_embeddings()
```

---

## 4. MultiModalFeatureSpec 是什么

`MultiModalFeatureSpec` 描述一个请求里的一个多模态 item。

源码位置：`code/vllm/vllm/multimodal/inputs.py:301`

核心字段是：

```text
data: MultiModalKwargsItem | None
modality: str
identifier: str
mm_position: PlaceholderRange
mm_hash: str | None
```

可以这样理解：

```text
MultiModalFeatureSpec =
  一个多模态 item 的数据 + 类型 + cache key + 在 prompt 里的 placeholder 位置
```

### 4.1 `data`

`data` 是已经经过 processor 处理后、可以传给模型 `embed_multimodal()` 的 kwargs item。

如果 `data is None`，说明该 item 可能已经在缓存中，或者不需要在当前进程间重复传输。

`_execute_mm_encoder()` 会跳过 `data is None` 的 item：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2945`

```text
if mm_feature.data is None:
    continue
```

### 4.2 `modality`

`modality` 表示输入类型，例如：

```text
image / video / audio / prompt_embeds
```

`_execute_mm_encoder()` 会按 modality 分组 batch，以便同类输入一起进入对应模型 encoder。

### 4.3 `identifier`

`identifier` 是 encoder output cache 的 key。

`_execute_mm_encoder()` 最终会执行：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3157`

```text
self.encoder_cache[mm_hash] = output
```

这里的 `mm_hash` 来自 `mm_feature.identifier`。

### 4.4 `mm_position`

`mm_position` 是 `PlaceholderRange`。

源码位置：`code/vllm/vllm/multimodal/inputs.py:118`

它描述多模态 placeholder 在 prompt token 序列里的位置：

```text
offset: int
length: int
is_embed: torch.Tensor | None
```

例如：

```text
Prompt tokens:
  [<image_0_placeholder_0>, ..., <image_0_placeholder_335>, "What", "is", "this"]

PlaceholderRange:
  offset = 0
  length = 336
```

如果 `is_embed` 不为空，说明 placeholder span 内不是每个位置都要替换成 embedding；`get_embeds_indices_in_range()` 会把 token window 映射到 encoder output 的 embedding window。

源码位置：`code/vllm/vllm/multimodal/inputs.py:158`

---

## 5. Scheduler 如何决定本轮 encode 哪些 item

Scheduler 的 `_try_schedule_encoder_inputs()` 负责根据本轮 token window、encoder cache、encoder compute budget 来决定哪些多模态 item 要进入 `scheduled_encoder_inputs`。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1367`

它的核心输入是：

```text
request.mm_features
num_computed_tokens
num_new_tokens
encoder_compute_budget
```

主逻辑是：

```text
1. 如果本轮没有新 token，或请求没有 encoder input，直接返回空列表。
2. 用 get_mm_features_in_window() 找到和本轮 token window 重叠的 mm_features。
3. 对每个 mm_feature：
   - encoder-decoder 且 encoder 已经处理过，则跳过；
   - 如果 encoder cache 已经有结果，则跳过；
   - 如果 disable_chunked_mm_input 且本轮只覆盖 item 的一部分，则回退 token 数；
   - 如果 encoder cache 空间或 encoder compute budget 不够，则停止或回退；
   - 否则把该 item 下标加入 encoder_inputs_to_schedule。
4. 返回 encoder_inputs_to_schedule 和修正后的 num_new_tokens。
```

关键点：

```text
Scheduler 不是“看到多模态请求就一定跑 encoder”。
它只会在本轮 token window 需要这个 encoder output，且 cache / budget 允许时，才把 item 下标放进 scheduled_encoder_inputs。
```

`get_mm_features_in_window()` 的作用是按 token window 找重叠的多模态 placeholder。

源码位置：`code/vllm/vllm/multimodal/utils.py:114`

```text
get_mm_features_in_window(mm_features, start, end) -> (lo, hi)
```

它假设 `mm_features` 已经按 `mm_position.offset` 排序，并返回与 `[start, end)` 有重叠的 feature 下标范围。

---

## 6. `_preprocess()` 的分支选择

`_preprocess()` 是多模态 encoder flow 的总入口。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3476`

它先拿到三个关键信息：

```text
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
is_first_rank = get_pp_group().is_first_rank
is_encoder_decoder = self.model_config.is_encoder_decoder
```

然后根据模型能力和并行位置选择路径。

### 6.1 decoder-only multimodal 路径

条件是：

```text
self.supports_mm_inputs and is_first_rank and not is_encoder_decoder
```

这就是典型 VLM decoder-only 模型的路径，例如文本 token 序列中混入 image placeholder，然后把 image embedding merge 到 `inputs_embeds`。

执行顺序是：

```text
with maybe_get_ec_connector_output(...):
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

inputs_embeds_scheduled = self.model.embed_input_ids(
    input_ids,
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
model_kwargs = {
    **self._init_model_kwargs(),
    **self._extract_mm_kwargs(scheduler_output),
}
```

这里最重要的结论是：

```text
多模态模型默认走 inputs_embeds，而不是只把 input_ids 传给模型。
```

原因是多模态 embedding 已经不再是普通 vocab id 能表达的内容，必须在 forward 前把 token embedding 和 mm embedding 合并好。

### 6.2 prompt_embeds-only 路径

条件是：

```text
self.enable_prompt_embeds and is_first_rank
```

这条路径不是普通 multimodal encoder，而是请求里直接给了预计算 prompt embeddings。

它会：

```text
- 找出哪些位置仍然是 token ids；
- 对 token-id 位置调用 model.embed_input_ids()；
- 保留已经在 self.inputs_embeds 里的 prompt embeds；
- input_ids = None；
- inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]。
```

### 6.3 text-only 路径

普通文本模型走：

```text
input_ids = self.input_ids.gpu[:num_input_tokens]
inputs_embeds = None
model_kwargs = self._init_model_kwargs()
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3579`

这也是为什么 vLLM 不把所有请求都统一转成 embedding：

```text
text-only 路径保留 input_ids，可以让 embedding layer 进入 CUDA graph，性能更好。
```

### 6.4 encoder-decoder 路径

encoder-decoder 模型在 `_preprocess()` 后半段单独处理：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3605`

```text
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

它和 decoder-only multimodal 的差异是：

```text
decoder-only multimodal：
  encoder output 会替换 prompt placeholder，形成 inputs_embeds。

encoder-decoder：
  encoder output 不替换 decoder token embedding，
  而是作为 encoder_outputs 传给 decoder cross-attention。
```

---

## 7. `_execute_mm_encoder()` 如何执行 encoder

`_execute_mm_encoder()` 是真正运行多模态 encoder 的函数。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2956`

它可以拆成 6 步。

### 7.1 从 SchedulerOutput 批量取出本轮 item

第一步调用：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2913`

```text
_batch_mm_inputs_from_scheduler(scheduler_output)
```

这个函数遍历：

```text
scheduler_output.scheduled_encoder_inputs.items()
```

然后对每个 `(req_id, mm_input_id)` 取：

```text
req_state = self.requests[req_id]
mm_feature = req_state.mm_features[mm_input_id]
```

最终产出三组列表：

```text
mm_hashes: list[str]
mm_kwargs: list[tuple[str, MultiModalKwargsItem]]
mm_lora_refs: list[tuple[str, PlaceholderRange]]
```

含义分别是：

```text
mm_hashes：encoder output cache key；
mm_kwargs：真正传给模型 embed_multimodal() 的数据；
mm_lora_refs：为多模态 tower / connector LoRA 构造 mapping 所需的 request 和位置引用。
```

如果没有需要处理的 kwargs：

```text
if not mm_kwargs:
    return []
```

### 7.2 特判 `prompt_embeds`

`prompt_embeds` 是 passthrough modality。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2966`

它已经处在模型 embedding space 里，不需要再跑 vision/audio encoder。

处理方式是：

```text
pe_tensor = mm_kwargs[i][1]["embedding"].data
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
maybe_save_ec_to_connector(...)
```

然后从 `mm_kwargs / mm_hashes / mm_lora_refs` 里过滤掉这些 item。

这一步的意义是：

```text
即使 prompt_embeds 不跑 encoder，也把它放进 encoder_cache，
这样后面的 _gather_mm_embeddings() 可以复用同一套 is_mm_embed / placeholder splice 逻辑。
```

### 7.3 准备多模态 LoRA mapping

如果启用了 LoRA，并且 LoRA manager 支持 tower / connector LoRA，`_execute_mm_encoder()` 会根据 `mm_lora_refs` 构造 encoder 侧 LoRA mapping。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3008`

关键点是：

```text
主 batch 的 token mapping 不能直接复用到 encoder batch，
因为 encoder batch 是按多模态 item 分组的，和 decoder token batch 结构不同。
```

所以这里会单独构造：

```text
LoRAMappingType.TOWER
LoRAMappingType.CONNECTOR
```

其中 token 数量来自模型接口：

```text
model.get_num_mm_encoder_tokens(...)
model.get_num_mm_connector_tokens(...)
```

### 7.4 按 modality group + batch

真正执行 encoder 前，会调用：

```text
group_and_batch_mm_kwargs(mm_kwargs, device=self.device, pin_memory=PIN_MEMORY)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3080`

原因是：

```text
不同 modality 的输入字段、shape、processor 输出格式不同，不能无脑拼成一个 batch。
同 modality 的 item 可以尽量 batch，提高 encoder 吞吐。
```

当前实现还保留一个重要约束：

```text
如果同一个 batch 中出现多个 modality，或顺序发生变化，runner 会分组处理以保持输出顺序。
```

因为后续 `encoder_outputs` 需要和 `mm_hashes` 按顺序 zip 写入 cache。

### 7.5 调用 `model.embed_multimodal()`

普通路径下，encoder 调用是：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3150`

```text
batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

模型接口定义在：

源码位置：`code/vllm/vllm/model_executor/models/interfaces.py:160`

```text
embed_multimodal(**kwargs) -> MultiModalEmbeddings
```

它要求返回的 embedding 顺序和 prompt 中多模态 item 出现顺序一致。

返回类型可以是：

源码位置：`code/vllm/vllm/model_executor/models/interfaces.py:63`

```text
- list / tuple of 2D tensors；
- 或单个 3D tensor，batch 维对应多个 item。
```

这说明：

```text
GPUModelRunner 不理解每个模型的 vision tower / audio tower 细节。
它只负责把 kwargs batch 交给模型的统一接口 embed_multimodal()。
具体怎么从 pixel_values / video / audio 变成 hidden_size embedding，是模型类自己的职责。
```

### 7.6 写入 encoder cache

最后 `_execute_mm_encoder()` 会检查输出数量，然后写入 cache：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3152`

```text
sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
encoder_outputs.extend(batch_outputs)

for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

这里的 cache 是 worker / GPUModelRunner 侧状态：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:564`

```text
self.encoder_cache: dict[str, torch.Tensor] = {}
```

它保存的是：

```text
mm_hash -> encoder output embedding tensor
```

---

## 8. `_gather_mm_embeddings()` 如何对齐 placeholder

`_execute_mm_encoder()` 只是把每个多模态 item 的完整 encoder output 算出来并缓存。

真正决定“本轮 forward 哪些 token 位置要替换成多模态 embedding”的，是 `_gather_mm_embeddings()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3165`

它返回：

```text
mm_embeds: list[torch.Tensor]
is_mm_embed: torch.Tensor
```

### 8.1 `mm_embeds`

`mm_embeds` 是本轮需要 merge 到输入序列里的多模态 embedding 列表。

它不是简单地返回每个 item 的完整 encoder output，而是会根据本轮 token window 截取。

### 8.2 `is_mm_embed`

`is_mm_embed` 是一个 CPU bool tensor，长度等于：

```text
scheduler_output.total_num_scheduled_tokens
```

它表示本轮扁平化 token batch 中哪些位置应该用多模态 embedding 覆盖文本 embedding。

后面调用 `model.embed_input_ids()` 时会传入：

```text
is_multimodal=is_mm_embed
```

### 8.3 按 request 遍历本轮 batch

函数按 `self.input_batch.req_ids` 遍历本轮 persistent batch：

```text
for req_id in self.input_batch.req_ids:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    req_state = self.requests[req_id]
    num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens
```

这里 `num_computed_tokens` 是这个 request 当前已经处理到 prompt / output 序列的哪个位置。

结合本轮 token 数，可以得到 request 内部的 token window：

```text
[num_computed_tokens, num_computed_tokens + num_scheduled_tokens)
```

### 8.4 找到和本轮 window 重叠的 mm_features

调用：

```text
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_scheduled_tokens,
)
```

这一步避免扫描所有多模态 item。

### 8.5 计算当前 item 在本轮 window 里的相对范围

对每个重叠的 `mm_feature`：

```text
pos_info = mm_feature.mm_position
start_pos = pos_info.offset
num_encoder_tokens = pos_info.length

start_idx = max(num_computed_tokens - start_pos, 0)
end_idx = min(
    num_computed_tokens - start_pos + num_scheduled_tokens,
    num_encoder_tokens,
)
```

含义是：

```text
start_idx / end_idx 是本轮 token window 与该 placeholder span 的重叠部分，
并且是相对于 placeholder 起点的下标。
```

举例：

```text
placeholder: offset=100, length=10，对应 token [100, 110)
本轮 window: [106, 114)

start_idx = 6
end_idx = 10

说明本轮只需要该多模态 item 的后 4 个 placeholder token 对应的 embedding。
```

### 8.6 `is_embed` 的作用

如果 `PlaceholderRange.is_embed is None`，说明 placeholder span 里的每个位置都对应一个 embedding。

此时：

```text
mm_embeds_item = encoder_output[start_idx:end_idx]
is_mm_embed[...] = True
```

如果 `is_embed` 不为空，说明 placeholder span 里只有部分位置需要 embedding。

这时会调用：

源码位置：`code/vllm/vllm/multimodal/inputs.py:158`

```text
curr_embeds_start, curr_embeds_end = pos_info.get_embeds_indices_in_range(
    start_idx,
    end_idx,
)
```

然后截取：

```text
mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
is_mm_embed[...] |= is_embed
```

这解决了一个重要问题：

```text
placeholder token window 和 encoder output embedding window 不一定是一一等长映射。
```

### 8.7 从 encoder_cache 取结果

`_gather_mm_embeddings()` 不直接运行 encoder，而是要求 encoder output 已经存在：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3217`

```text
mm_hash = mm_feature.identifier
encoder_output = self.encoder_cache.get(mm_hash, None)
if encoder_output is None:
    if start_pos >= req_state.num_computed_tokens + num_scheduled_tokens:
        continue
    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")
```

所以多模态主线有一个强约束：

```text
如果本轮 token window 需要某个 mm embedding，
那么这个 mm_hash 必须已经由本轮 _execute_mm_encoder() 写入 cache，
或者由之前步骤 / EC connector 提前放进 cache。
```

### 8.8 生成扁平 batch 的 mask

`req_start_idx` 记录当前 request 在本轮扁平化 batch 中的起点。

```text
req_start_pos = req_start_idx + start_pos - num_computed_tokens
```

然后设置：

```text
is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = True
```

最终 `is_mm_embed` 的坐标系不是 request 内部坐标，而是：

```text
本轮所有 scheduled tokens 拼成的 flat token batch 坐标。
```

这正好和 `self.input_ids.gpu[:num_scheduled_tokens]` 对齐。

---

## 9. `embed_input_ids()` 如何 merge embedding

多模态 decoder-only 路径里，`_preprocess()` 会调用：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3538`

```text
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

模型协议定义在：

源码位置：`code/vllm/vllm/model_executor/models/interfaces.py:380`

`SupportsMultiModal.embed_input_ids()` 做三件事：

```text
1. 先调用 language model 的 embed_input_ids，把普通 input_ids 变成 text embeddings；
2. 如果模型存在 out-of-vocab multimodal tokens，会先把 multimodal 位置 mask 成 0，避免 embedding lookup 越界；
3. 调用 _merge_multimodal_embeddings()，用 multimodal_embeddings 覆盖 is_multimodal=True 的位置。
```

对应源码逻辑是：

```text
inputs_embeds = self._embed_text_input_ids(
    input_ids,
    self.get_language_model().embed_input_ids,
    is_multimodal=is_multimodal,
)

if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
    return inputs_embeds

return _merge_multimodal_embeddings(
    inputs_embeds=inputs_embeds,
    multimodal_embeddings=multimodal_embeddings,
    is_multimodal=_require_is_multimodal(is_multimodal),
)
```

这一步之后，ModelRunner 把结果放到预分配 buffer：

```text
self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
```

然后 `_prepare_mm_inputs()` 返回真正传给 forward 的输入：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3462`

```text
if self.model.requires_raw_input_tokens:
    input_ids = self.input_ids.gpu[:num_tokens]
else:
    input_ids = None

inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
return input_ids, inputs_embeds
```

也就是说，多模态 forward 的常见形态是：

```text
_model_forward(
  input_ids=None,
  inputs_embeds=<text + mm merged embeddings>,
  positions=...,
  ...
)
```

但如果模型声明 `requires_raw_input_tokens=True`，则会同时传 `input_ids` 和 `inputs_embeds`。

---

## 10. 和普通文本 forward 的关系

可以把普通文本和多模态请求对比如下：

```text
普通文本请求：
  input_ids
    → _model_forward(input_ids=input_ids, inputs_embeds=None)
    → 模型内部 embedding layer
    → hidden states

多模态 decoder-only 请求：
  input_ids + mm_features
    → _execute_mm_encoder()
    → encoder_cache
    → _gather_mm_embeddings()
    → model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
    → inputs_embeds
    → _model_forward(input_ids=None 或 raw input_ids, inputs_embeds=inputs_embeds)
    → hidden states

encoder-decoder 多模态请求：
  scheduled_encoder_inputs
    → _execute_mm_encoder()
    → encoder_outputs
    → _model_forward(..., model_kwargs["encoder_outputs"] = encoder_outputs)
```

汇合点是：

```text
_model_forward()
```

区别主要发生在 `_preprocess()`。

---

## 11. prompt_embeds 和 multimodal embeddings 如何共存

`prompt_embeds` 有两种相关形态，需要分开看。

### 11.1 作为 modality 的 `prompt_embeds`

在 `_execute_mm_encoder()` 中，`modality == "prompt_embeds"` 被当作 passthrough modality。

它不会跑 encoder，而是直接：

```text
self.encoder_cache[mm_hash] = pe_tensor.to(self.device)
```

然后后续 `_gather_mm_embeddings()` 按普通 mm feature 处理它。

这保证了：

```text
prompt_embeds 可以和 image / video / audio 走同一套 placeholder splice 逻辑。
```

### 11.2 请求级 `prompt_embeds`

如果 `enable_prompt_embeds` 打开，并且 `self.input_batch.req_prompt_embeds` 不为空，`_preprocess()` 会避免覆盖这些已给定 embedding。

多模态路径下的特殊处理是：

```text
is_token_ids = self.is_token_ids.gpu[:num_scheduled_tokens]
safe_input_ids = torch.where(is_token_ids, self.input_ids.gpu[:num_scheduled_tokens], 0)
inputs_embeds_scheduled = self.model.embed_input_ids(
    safe_input_ids,
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
self.inputs_embeds.gpu[:num_scheduled_tokens] = torch.where(
    is_token_ids.unsqueeze(-1),
    inputs_embeds_scheduled,
    target,
)
```

含义是：

```text
- token-id 位置：重新 embed，并允许 mm embedding 覆盖；
- prompt-embed 位置：保留 self.inputs_embeds 里已有的 embedding；
- safe_input_ids 把非 token-id 位置置 0，避免 embedding lookup 读到非法 id。
```

---

## 12. EC connector 与 encoder cache

`_preprocess()` 的 decoder-only multimodal 路径包了一层：

```text
with self.maybe_get_ec_connector_output(
    scheduler_output,
    encoder_cache=self.encoder_cache,
) as ec_connector_output:
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

这说明 encoder cache 不只可以由本地 `_execute_mm_encoder()` 生成，还可以和外部 EC connector 协作。

从 ModelRunner 视角看，约束仍然很简单：

```text
_gather_mm_embeddings() 只认 self.encoder_cache。
只要进入 gather 前 encoder_cache[mm_hash] 已经存在，后续 merge 逻辑不关心它来自本地 encoder、prompt_embeds passthrough，还是外部 connector。
```

写入本地 encoder output 后也会调用：

```text
maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

表示本地算出的 encoder output 可以被保存到外部 encoder cache connector。

---

## 13. M-RoPE / XD-RoPE 与多模态 pruning

`_gather_mm_embeddings()` 里有一段多模态 pruning + M-RoPE 的特殊处理：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3248`

```text
if self.is_multimodal_pruning_enabled and self.uses_mrope:
    mm_embeds_req, new_mrope_positions, new_delta = (
        self.model.recompute_mrope_positions(...)
    )
    req_state.mrope_positions.copy_(new_mrope_positions)
    req_state.mrope_position_delta = new_delta
```

原因是：

```text
多模态 pruning 可能改变实际进入 language model 的视觉 token 数；
原始按未裁剪 placeholder 算出的 M-RoPE positions 需要同步修正。
```

修正后会重新计算并拷贝到 GPU：

```text
self._calc_mrope_positions(scheduler_output)
self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)
```

`XD-RoPE` 也预留了类似同步分支。

---

## 14. Pipeline parallel 场景

`_preprocess()` 只在 PP first rank 处理多模态输入：

```text
is_first_rank = get_pp_group().is_first_rank
```

decoder-only multimodal 路径条件明确要求：

```text
self.supports_mm_inputs and is_first_rank and not is_encoder_decoder
```

非 first rank 不重新跑多模态 encoder，也不重新构造 inputs_embeds，而是接收上一 pipeline stage 的 intermediate tensors：

```text
assert intermediate_tensors is not None
intermediate_tensors = self.sync_and_gather_intermediate_tensors(
    num_input_tokens,
    intermediate_tensors,
    True,
)
```

这符合 PP 的职责划分：

```text
first rank：负责 token / embedding 输入入口，包括 multimodal merge；
后续 rank：消费上一 stage 的 hidden states / intermediate tensors。
```

---

## 15. 为什么 `_execute_mm_encoder()` 和 `_gather_mm_embeddings()` 分开

这两个函数分开是必要的。

### 15.1 `_execute_mm_encoder()` 面向 item

它处理的是：

```text
- 本轮哪些多模态 item 要 encode；
- 如何按 modality batch；
- 如何调用模型 tower / connector；
- 如何写入 encoder_cache。
```

它的单位是：

```text
多模态 item
```

### 15.2 `_gather_mm_embeddings()` 面向 token window

它处理的是：

```text
- 本轮每个 request schedule 了哪些 token；
- 这些 token 和哪些 placeholder range 重叠；
- 每个 placeholder range 需要 encoder output 的哪一段；
- flat token batch 中哪些位置要被 mm embedding 替换。
```

它的单位是：

```text
本轮 scheduled token window
```

这就是为什么 encoder output 要先进入 `encoder_cache`：

```text
item 级 encoder 计算 和 token-window 级 embedding splice 不是同一个维度。
```

---

## 16. 关键不变量

梳理代码时可以抓住这些不变量。

### 16.1 `scheduled_encoder_inputs` 只携带下标

```text
SchedulerOutput.scheduled_encoder_inputs 不携带 tensor 数据，
它只告诉 worker：本轮哪些 req_id 的哪些 mm_features 需要处理。
```

### 16.2 `mm_feature.identifier` 是 encoder cache key

```text
_execute_mm_encoder() 写 encoder_cache[identifier]；
_gather_mm_embeddings() 读 encoder_cache[identifier]。
```

### 16.3 `PlaceholderRange` 决定 splice 坐标

```text
mm_position.offset / length 决定 placeholder 在 token 序列中的位置；
is_embed 决定 placeholder 内哪些位置真的消费 encoder embedding。
```

### 16.4 `is_mm_embed` 坐标系是 flat scheduled token batch

它和下面这个张量对齐：

```text
self.input_ids.gpu[:scheduler_output.total_num_scheduled_tokens]
```

不是单个 request 内的坐标。

### 16.5 decoder-only multimodal 最终主要走 `inputs_embeds`

```text
text-only：input_ids → forward
multimodal：input_ids → embed_input_ids(...mm...) → inputs_embeds → forward
```

### 16.6 encoder-decoder 不做 placeholder replacement

```text
encoder-decoder 的 encoder output 作为 encoder_outputs 传入 decoder，
不会 merge 到 decoder input embeddings 里。
```

---

## 17. 典型例子

假设一个请求的 prompt 里有一张图：

```text
tokens:
  ["<image>", "<image>", "<image>", "<image>", "What", "is", "this"]

mm_features[0]:
  modality = "image"
  identifier = "img-hash-xxx"
  mm_position = PlaceholderRange(offset=0, length=4)
```

第一轮 prefill schedule 了 7 个 token。

Scheduler 会看到本轮 window `[0, 7)` 覆盖 image placeholder，于是：

```text
scheduled_encoder_inputs = {req_id: [0]}
```

ModelRunner 执行：

```text
_execute_mm_encoder():
  req_state.mm_features[0]
    → image kwargs
    → model.embed_multimodal(...)
    → encoder_cache["img-hash-xxx"] = image_embeds

_gather_mm_embeddings():
  window [0, 7) 与 placeholder [0, 4) 重叠
    → 取 encoder_cache["img-hash-xxx"][0:4]
    → is_mm_embed[0:4] = True

embed_input_ids():
  先得到 7 个 text embeddings
  再用 4 个 image embeddings 覆盖 is_mm_embed=True 的位置

_model_forward():
  inputs_embeds = [image_embeds(4), text_embeds(3)]
```

如果是 chunked prefill，本轮只 schedule `[0, 2)`，那么：

```text
_gather_mm_embeddings() 只取 image_embeds[0:2]
is_mm_embed 只标记本轮 flat batch 的前 2 个位置。
```

后续轮次 schedule `[2, 7)` 时，会继续从 `encoder_cache` 取剩余 image embeddings，而不需要重新跑 encoder。

---

## 18. 常见问题

### 18.1 为什么 scheduler schedule 的是 encoder input 下标，而不是 embedding？

因为 scheduler 负责资源决策，不负责 GPU 上的模型计算。

```text
Scheduler：决定哪些 item 需要 encode，管理 encoder budget/cache 生命周期；
ModelRunner：持有模型和 GPU cache，实际运行 encoder 并 merge embeddings。
```

### 18.2 为什么 `_gather_mm_embeddings()` 可能 cache miss？

如果本轮 token window 需要某个 placeholder embedding，但 `encoder_cache[mm_hash]` 不存在，当前实现会允许 drafter 的 +1 look-ahead 跳过尚未编码的后续 feature；否则抛出：

```text
RuntimeError(f"Encoder cache miss for {mm_hash}.")
```

通常说明：

```text
- Scheduler 没有把对应 item 放进 scheduled_encoder_inputs；
- encoder cache 被提前释放；
- external EC connector 没有把需要的 item 放回本地 cache；
- chunked / speculative / prefix cache 相关的 token window 计算出了错。
```

### 18.3 为什么 prompt_embeds 要写入 encoder_cache？

因为后续 gather / merge 逻辑统一从 `encoder_cache` 取 embedding。

```text
prompt_embeds 写入 encoder_cache 后，
就可以像 image/audio/video encoder output 一样按 placeholder range 被 splice。
```

### 18.4 为什么 text-only 不也统一转成 inputs_embeds？

性能原因。

text-only 保留 `input_ids`，模型 embedding layer 可以留在 CUDA graph 内。

多模态必须提前 merge embedding，所以只能走 `inputs_embeds` 或 `input_ids + inputs_embeds` 的 forward 形态。

### 18.5 encoder output 的顺序为什么重要？

`_execute_mm_encoder()` 最后是：

```text
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
```

所以 `embed_multimodal()` 的输出顺序必须和输入 item 顺序一致。

模型接口也明确要求：

```text
returned multimodal embeddings must be in the same order as their corresponding multimodal data item in the input prompt.
```

---

## 19. 和其它模块的关系

### 19.1 和 Scheduler 的关系

```text
Scheduler 决定：
  - 本轮 token window；
  - 哪些 encoder input 要 schedule；
  - encoder compute budget 是否足够；
  - encoder cache 何时释放。

ModelRunner 执行：
  - 根据 scheduled_encoder_inputs 找到 mm_features；
  - 运行 embed_multimodal；
  - 管理 GPU 侧 encoder_cache；
  - 把 mm embeddings merge 到 inputs_embeds。
```

### 19.2 和 InputBatch 的关系

`InputBatch` 维护本轮 request 顺序、token ids、prompt embeds、is_token_ids、LoRA mapping 等 batch 状态。

`_prepare_inputs()` 会先把这些状态整理成 flat token batch。

多模态 gather 必须发生在 `_prepare_inputs()` 之后，因为：

```text
_prepare_inputs() 可能重排 batch；
_gather_mm_embeddings() 需要按最终 input_batch.req_ids 顺序构造 flat is_mm_embed mask。
```

源码注释也明确写了这一点：

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3497`

```text
_prepare_inputs may reorder the batch, so we must gather multi modal outputs after that to ensure the correct order
```

### 19.3 和模型类的关系

模型类负责实现：

```text
embed_multimodal(**kwargs)
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
get_num_mm_encoder_tokens(...)
get_num_mm_connector_tokens(...)
```

`GPUModelRunner` 只依赖 `SupportsMultiModal` 协议，不硬编码具体模型的视觉塔结构。

---

## 20. 一句话总结

`GPUModelRunner` 的多模态 encoder flow 本质上是把“item 级多模态 encoder 计算”和“token-window 级 embedding 替换”分成两步：先根据 `scheduled_encoder_inputs` 执行或复用 encoder output，写入 `encoder_cache`；再根据 `PlaceholderRange` 和本轮 scheduled token window 收集 `mm_embeds` 与 `is_mm_embed`，调用 `embed_input_ids()` 合并成 `inputs_embeds`，最后进入统一的 `_model_forward()`。

最核心的主线是：

```text
SchedulerOutput.scheduled_encoder_inputs
  → self.requests[req_id].mm_features[mm_input_id]
  → _execute_mm_encoder()
  → model.embed_multimodal(...)
  → encoder_cache[mm_hash]
  → _gather_mm_embeddings()
  → mm_embeds + is_mm_embed
  → model.embed_input_ids(...)
  → inputs_embeds
  → _model_forward()
```
