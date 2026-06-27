# 08. ModelRunner 如何执行多模态 encoder？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/`

本问题关注：Scheduler 已经把本轮要处理的 encoder input 放进 `SchedulerOutput` 后，ModelRunner 如何在 `_preprocess()` 阶段执行 multimodal encoder，如何收集多模态 embedding，并最终与文本 token embedding 合并为 `inputs_embeds`。

---

## 1. 一句话回答占位

```text
ModelRunner 在模型 forward 前的 preprocess 阶段处理多模态输入：
先执行或复用 mm encoder output，再把多模态 embedding 合并到 inputs_embeds，最后走统一的 model forward。
```

---

## 2. 主链路占位

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
      → maybe_get_ec_connector_output()
      → _execute_mm_encoder()
      → _gather_mm_embeddings()
      → 合并 token embedding 和 mm embedding
  → _model_forward(input_ids=None 或 inputs_embeds=...)
```

---

## 3. 需要解释的问题占位

```text
- _preprocess() 什么时候选择 input_ids，什么时候选择 inputs_embeds？
- _execute_mm_encoder() 输入来自哪里？
- mm encoder output 如何和 placeholder span 对齐？
- prompt_embeds 和 multimodal embeddings 如何同时存在？
- encoder-decoder 模型和 decoder-only multimodal 模型有什么差异？
- PP 场景下非首 rank 如何处理多模态输入？
```

---

## 4. 和普通文本 forward 的关系占位

```text
文本请求：
  input_ids + positions → model forward

多模态请求：
  input_ids / token embeddings + mm embeddings → inputs_embeds + positions → model forward

区别在 preprocess，汇合点在 _model_forward。
```

---

## 5. 后续待补源码证据

占位：补充 `_preprocess()`、`_execute_mm_encoder()`、`_gather_mm_embeddings()`、模型接口调用源码位置。
