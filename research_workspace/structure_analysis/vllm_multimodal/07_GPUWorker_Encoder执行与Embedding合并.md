# 07 GPUWorker、Encoder 执行与 Embedding 合并

本篇梳理 scheduler 下发多模态 encoder 任务后，GPU worker 如何组装多模态 batch、执行 encoder、缓存 encoder output，并把多模态 embedding 合入语言模型输入。

## 1. 总体链路

```text
SchedulerOutput.scheduled_encoder_inputs
  ↓
GPUModelRunner._batch_mm_inputs_from_scheduler()
  ↓
取出 req_state.mm_features[mm_input_id]
  ↓
组装 mm_kwargs batch / mm_hashes / LoRA 信息
  ↓
GPUModelRunner._execute_mm_encoder()
  ↓
model.embed_multimodal(**mm_kwargs_batch)
  ↓
GPU encoder cache 保存输出
  ↓
GPUModelRunner._gather_mm_embeddings()
  ↓
model.embed_input_ids(input_ids, multimodal_embeddings=..., is_multimodal=...)
  ↓
语言模型 forward
```

关键位置：

- request GPU 状态：`code/vllm/vllm/v1/worker/gpu_input_batch.py:33`
- 添加请求状态：`code/vllm/vllm/v1/worker/gpu_input_batch.py:335`
- GPU runner 构造 request state：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1224`
- 更新已有请求：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1566`
- 组 encoder batch：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`
- 执行 encoder：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- gather embedding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`
- 合入 input embedding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`

## 2. GPU 侧请求状态

`CachedRequestState` 保存每个请求在 GPU worker 侧的状态。

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:33`。

多模态相关字段包括：

- `mm_features`；
- `prompt_embeds`；
- `prompt_is_token_ids`。

当 `InputBatch.add_request(...)` 添加请求时，会把这些字段保存到 batch 状态中：`code/vllm/vllm/v1/worker/gpu_input_batch.py:335`。

这说明 engine 侧整理出的 `Request.mm_features` 会原样进入 GPU worker。

## 3. `_batch_mm_inputs_from_scheduler()`

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`。

它读取 scheduler output：

```text
scheduled_encoder_inputs: dict[request_id, list[mm_input_id]]
```

然后对每个 `request_id` 和 `mm_input_id`：

1. 找到 GPU 侧 `req_state`；
2. 取 `req_state.mm_features[mm_input_id]`；
3. 收集该 item 的 `data`、`modality`、`identifier`、`mm_hash`；
4. 组装 batched `mm_kwargs`；
5. 处理多模态 LoRA 映射。

scheduler 下发的是“哪个 item 要算”，worker 才真正拿到对应 image/audio/video kwargs。

## 4. `_execute_mm_encoder()`

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`。

它负责执行多模态 encoder。

主路径是调用模型接口：

```python
model.embed_multimodal(**mm_kwargs_batch)
```

抽象辅助文件中也有同样结构：

- `prepare_mm_inputs`：`code/vllm/vllm/v1/worker/gpu/mm/encoder_runner.py:34`
- `execute_mm_encoder`：`code/vllm/vllm/v1/worker/gpu/mm/encoder_runner.py:50`

对于普通 decoder-only 多模态模型，`embed_multimodal` 会返回 image/audio/video 的 embeddings。

## 5. video 的特殊处理

video 可能不总是能和 image 一样常规 batch 化，因为不同视频长度、帧采样、grid、processor 输出可能差异更大。

`_execute_mm_encoder()` 中针对 video 有更保守的 sequential/micro-batch 处理逻辑。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`。

文档中可以概括为：vLLM 在 worker 层统一抽象为 `embed_multimodal`，但对 video 这类高维变长输入保留了特殊执行路径。

## 6. prompt_embeds / embedding-only 旁路

有些输入已经是 embedding，不需要执行 tower encoder。

这类输入仍会进入统一的 `mm_features` / cache / gather 框架，但 worker 可以直接把 embedding 放入 GPU encoder cache，而不是调用 vision/audio tower。

相关字段：

- `EngineCoreRequest.prompt_embeds`：`code/vllm/vllm/v1/engine/__init__.py:86`
- `CachedRequestState.prompt_embeds`：`code/vllm/vllm/v1/worker/gpu_input_batch.py:33`
- `_execute_mm_encoder()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`

## 7. GPU encoder cache

GPU 侧真实 tensor cache 在：`code/vllm/vllm/v1/worker/gpu/mm/encoder_cache.py:8`。

它保存：

```text
mm_features
encoder_outputs
```

和 scheduler 侧 `EncoderCacheManager` 的区别：

- scheduler cache 管 key、token 数、引用和释放；
- GPU cache 管真实 encoder output tensor。

当 scheduler output 带来 `free_encoder_mm_hashes` 时，worker 会据此删除不再需要的 GPU encoder outputs。

## 8. `_gather_mm_embeddings()`

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`。

它根据当前 step 的 token window，从 GPU encoder cache 中取出需要写回的多模态 embeddings。

关键点：

- 一个多模态 item 的 encoder output 可能很长；
- 当前 step 只覆盖 prompt 的某个 token window；
- gather 需要根据 `mm_position.offset/length` 切出本 step 对应部分；
- gather 后得到的 `mm_embeds` 与 `is_mm_embed` mask 会传给 embedding 层。

所以 encoder output 不是一次性无条件塞入模型，而是按调度 window 分段对齐。

## 9. `embed_input_ids()` 合并 embedding

调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`。

抽象辅助：`code/vllm/vllm/v1/worker/gpu/mm/encoder_runner.py:134`。

调用形态：

```python
model.embed_input_ids(
    input_ids,
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

默认实现会：

1. 先对普通 token 做文本 embedding；
2. 找出 `is_multimodal` 为 true 的位置；
3. 用多模态 embeddings 覆写这些位置；
4. 返回最终 inputs_embeds。

公共合并逻辑：`code/vllm/vllm/model_executor/models/utils.py:479`。

默认接口实现：`code/vllm/vllm/model_executor/models/vision.py:374`。

## 10. decoder-only 与 encoder-decoder 差异

### 10.1 decoder-only 主路径

大部分 VLM 走：

```text
placeholder token embeddings
  ↓
被 multimodal embeddings 覆写
  ↓
整段 inputs_embeds 进入 decoder-only LM
```

例如 LLaVA、Qwen2-VL、Gemma3-MM、Pixtral 等。

### 10.2 encoder-decoder 路径

Whisper 等模型可能把音频 encoder output 作为 encoder side output，传给 decoder cross-attention，而不是覆写 decoder token embedding。

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3552`。

模型实现参考：`code/vllm/vllm/model_executor/models/whisper.py:598`。

## 11. 多模态 LoRA 与位置编码辅助

GPU 多模态辅助目录还有几个重要文件：

### 11.1 LoRA

文件：`code/vllm/vllm/v1/worker/gpu/mm/lora.py:13`。

作用：为多模态 tower/connector 构造 LoRA 映射。多模态 encoder token 的 LoRA 映射不一定等同于纯文本 token 的 LoRA mapping。

### 11.2 M-RoPE / XD-RoPE

文件：`code/vllm/vllm/v1/worker/gpu/mm/rope.py`。

关键函数：

- `get_mrope_input_positions()`：`code/vllm/vllm/v1/worker/gpu/mm/rope.py:14`
- `get_xdrope_input_positions()`：`code/vllm/vllm/v1/worker/gpu/mm/rope.py:62`
- 相关位置准备：`code/vllm/vllm/v1/worker/gpu/mm/rope.py:117`

这说明 `mm_features` 不只用于 encoder 执行，也参与多维位置编码准备。

## 12. 常见问题定位

### 12.1 scheduler 已下发但 encoder 没执行

检查：

```text
scheduled_encoder_inputs 是否为空
request_id 是否仍在 GPU input batch
mm_input_id 是否存在于 req_state.mm_features
encoder cache 是否已经命中
embedding-only 是否绕过 tower
```

### 12.2 encoder 输出和 placeholder 对不上

检查：

```text
mm_position offset/length
模型 get_num_mm_encoder_tokens / connector tokens
_gather_mm_embeddings 的 window 切片
embed_input_ids 的 is_multimodal mask
```

### 12.3 多模态 embedding merge 后 shape 错误

检查：

```text
mm_embeds_flat 长度
is_multimodal true 的数量
hidden size 是否一致
connector/projector 输出维度
模型语言部分 embedding dim
```

## 13. 一句话总结

GPU worker 根据 scheduler 下发的多模态 item id，从 `req_state.mm_features` 取出 processor 输出，调用模型 `embed_multimodal` 得到 encoder embeddings，写入 GPU encoder cache；每个 decode/prefill step 再按 token window gather 对应 embedding，并通过 `embed_input_ids` 覆写 placeholder token embedding，使多模态信息进入语言模型。
