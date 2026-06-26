# 08. 模型层和执行接口如何衔接 ModelRunner？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/model_executor/layers/logits_processor.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：模型被加载出来后，需要向 ModelRunner 暴露哪些执行接口，例如 `forward()`、`compute_logits()`、`pooler()`，以及 embedding、LM head、attention、MoE、LoRA、quantization layer 如何参与执行。

---

## 1. 一句话回答占位

占位：后续补充 vLLM 模型不是只要能 `forward`，还要符合 ModelRunner 对 generation / pooling / embedding 等任务的接口约定。

```text
ModelRunner._model_forward()
  → model.forward(input_ids / inputs_embeds / positions / model_kwargs)
  → hidden_states / IntermediateTensors
  → model.compute_logits(hidden_states[logits_indices])
  → sampler
```

Pooling 路径：

```text
model.forward()
  → hidden_states
  → model.pooler(hidden_states, pooling_metadata)
  → PoolingRequestOutput
```

---

## 2. 典型模型接口占位

```text
forward：
  执行 transformer / encoder / decoder 主体。

compute_logits：
  基于 hidden states 和 LM head 计算 logits。

pooler：
  pooling / embedding / classification / reward 类任务使用。

load_weights：
  模型类自定义权重映射和加载逻辑。

make_empty_intermediate_tensors：
  Pipeline Parallel 场景的中间张量接口。
```

---

## 3. layers 层占位

```text
Embedding / LM Head：
  vocab parallel embedding 和 parallel lm head。

Attention：
  接入 attention backend、KV cache、ForwardContext。

MLP / MoE：
  dense MLP 或 expert parallel MoE。

Norm / Activation：
  RMSNorm、LayerNorm、SiluAndMul 等。

Quantized layers：
  由 quantization config 决定具体实现。

LoRA layers：
  在启用 LoRA 时包装或替换原 layer。
```

---

## 4. 后续待补源码证据

占位：补充典型模型类、`compute_logits()`、`load_weights()`、attention layer、parallel embedding / lm head 的源码位置。
