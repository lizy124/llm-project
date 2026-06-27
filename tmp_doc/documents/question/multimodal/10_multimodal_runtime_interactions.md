# 10. Multimodal 如何影响输出、KV cache、并行和高级能力？

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/distributed/`
- `code/vllm/vllm/lora/`
- `code/vllm/vllm/v1/spec_decode/`

本问题关注：多模态链路不是孤立存在的，它会和输出、KV cache、encoder cache、parallelism、LoRA、spec decode、KV transfer 等运行时能力发生交互。本篇先建立这些边界。

---

## 1. 一句话回答占位

```text
多模态主要影响输入准备、encoder cache 和模型 forward，
最终输出仍然走普通 sampling / pooling / OutputProcessor；
它和 decoder KV cache、parallelism、LoRA、spec decode 等能力有交点，但职责边界不同。
```

---

## 2. 和输出链路的关系占位

```text
generation 多模态请求：
  mm encoder / inputs_embeds → decoder forward → logits → sampling → RequestOutput

pooling / embedding 多模态请求：
  mm encoder / inputs_embeds → model pooler → PoolingRequestOutput / EmbeddingOutput
```

需要说明：

```text
输出对象不会因为输入是 image/audio/video 而变成特殊多模态输出；
多数情况下仍然是文本 completion 或 pooling/embedding tensor。
```

---

## 3. 和 cache 的关系占位

```text
Processor cache：
  缓存 media processor 输出。

Encoder cache：
  缓存多模态 encoder output。

Decoder KV cache：
  缓存 decoder self-attention KV。

Prefix cache：
  主要作用于 token prefix / decoder KV，可受多模态 placeholder 和 prompt token 影响。
```

---

## 4. 和并行的关系占位

```text
TP：
  多模态模型层可能参与 tensor parallel。

PP：
  通常首个 pipeline stage 处理输入 embedding / mm encoder，后续 stage 处理 intermediate tensors。

DP：
  多个 engine / rank 分担请求，多模态 processor cache / encoder cache 需要按请求路由。

EP：
  如果多模态模型包含 MoE，路由专家输出仍走 routed experts 链路。
```

---

## 5. 和高级能力的关系占位

```text
LoRA：
  多模态模型可能只对 language model 部分或部分 projector 支持 LoRA。

Spec decode：
  draft / target model 是否都支持多模态输入需要单独判断。

KV transfer：
  decoder KV transfer 和 encoder cache transfer 是两类能力。

Structured output：
  多模态输入不改变 grammar bitmask 的采样约束位置。

Compilation / CUDA graph：
  多模态 encoder input 可能导致动态 shape，影响 graph capture / skip compiled。
```

---

## 6. 后续待补源码证据

占位：补充 output processor、多模态 pooling、KV/EC connector、PP/TP 场景、多模态 LoRA / spec decode 限制的源码证据。
