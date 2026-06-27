# vLLM Model Architectures 逻辑梳理

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/config.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/tasks.py`
- `code/vllm/vllm/transformers_utils/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本文按“先定边界，再走模型识别和构造，再拆 forward 接口、layer 组件、权重加载和扩展点”的方式，梳理 vLLM 中模型架构适配机制。

vLLM 支持很多模型，但它们不是各自完全独立的一套执行链路，而是共享一组约定：

```text
ModelConfig / HF config
  → architecture name
  → model registry
  → vLLM model class
  → shared layers
      → Attention
      → MLP
      → MoE
      → RMSNorm / LayerNorm
      → RotaryEmbedding
      → VocabParallelEmbedding / LMHead
  → forward(input_ids, positions, intermediate_tensors, **kwargs)
  → hidden states / logits / pooler output
  → weight loading mapping
```

---

## 0. 梳理规划

本目录要回答的问题分成 13 组：

```text
1. model architectures 在 vLLM 中处于哪一层？
2. model registry 如何根据 HF architecture 选择 vLLM model class？
3. ModelConfig 如何判断 task、runner_type、模型能力？
4. 一个 vLLM model class 是如何构造 layers 的？
5. ModelRunner 对 model forward 接口有什么约定？
6. Attention / MLP / Norm / RoPE 等基础 blocks 如何复用？
7. Embedding、LM head、logits processor 如何接入？
8. MoE 模型架构如何组织 experts、router 和 fused MoE？
9. Pooling / embedding / rerank 模型如何区别于生成模型？
10. 多模态模型如何组织 vision encoder、projector、M-RoPE 和 inputs_embeds？
11. 权重加载如何处理模型命名、fused layer、TP 切分和 checkpoint 差异？
12. Quantization、LoRA、parallelism 如何 hook 到模型架构中？
13. 新增一个模型架构需要实现哪些接口和检查项？
```

阅读顺序建议：

```text
model_architectures_overview.md
  → 01_model_architecture_role.md
  → 02_model_registry_and_resolution.md
  → 03_model_config_and_task_detection.md
  → 04_model_class_construction.md
  → 05_forward_interface_contract.md
  → 06_attention_mlp_norm_blocks.md
  → 07_embedding_and_lm_head.md
  → 08_moe_model_architectures.md
  → 09_pooling_embedding_rerank_models.md
  → 10_multimodal_model_architectures.md
  → 11_weight_loading_and_name_mapping.md
  → 12_quant_lora_parallelism_hooks.md
  → 13_add_new_model_checklist.md
```

---

## 1. 一句话回答

vLLM 的 model architecture 机制负责把“外部模型配置和 checkpoint”适配成“vLLM 执行层能统一调用的 model class”。

```text
HF config / architecture name
  → vLLM registry 找到 model class
  → model class 构造 vLLM layers
  → model_loader 加载 checkpoint
  → ModelRunner 用统一 forward 接口调用
  → 输出 hidden states / logits / pooling output
```

这层解决的是：

```text
不同模型长得不一样，
但执行层希望用统一方式准备输入、调用 forward、拿到输出。
```

---

## 2. 总体流程图

```text
EngineArgs / model path
  → ModelConfig
      → 读取 HF config
      → architecture names
      → task / runner_type / dtype / max_model_len
  → ModelRegistry
      → architecture name → vLLM model class
  → model_loader
      → construct model class
      → load weights
  → Worker / ModelRunner
      → input_ids / positions / inputs_embeds
      → model.forward(...)
  → model class
      → embedding
      → decoder layers
      → attention / MLP / MoE / norm
      → final norm
      → hidden states
  → generation model
      → compute_logits()
  → pooling model
      → pooler(hidden_states)
```

---

## 3. 模型架构层要统一哪些差异

```text
结构差异：
  Llama / Qwen / Mistral / DeepSeek / Mixtral / Gemma / encoder-only / encoder-decoder 等。

attention 差异：
  MHA / MQA / GQA / MLA / sliding window / cross attention。

MLP 差异：
  gated MLP / SwiGLU / GeLU / fused gate_up_proj。

MoE 差异：
  router、experts、shared experts、expert parallel、fused MoE。

位置编码差异：
  RoPE、M-RoPE、ALiBi、XD-RoPE、sliding window position。

输出差异：
  generation logits、pooling embedding、classification score、token-level output。

加载差异：
  权重名、fused layer、checkpoint shard、quantization、LoRA target modules。
```

---

## 4. 和其他专题的关系

```text
config_and_model_loading：
  解释 ModelConfig 和 model_loader 如何触发模型架构选择。

executor_worker_model_runner：
  解释 ModelRunner 如何统一调用不同模型的 forward。

attention：
  模型架构决定 Attention layer 的 head 数、MLA、sliding window 等参数。

parallelism：
  模型 layer 创建时要选择 TP / PP 兼容的 parallel layers。

quantization：
  模型 layer 创建时要挂接 quant_method。

lora_and_adapters：
  模型架构决定 LoRA target modules 如何匹配。

multimodal：
  多模态模型架构要接入 vision encoder / projector / M-RoPE。

sampling_and_output：
  generation model 输出 logits，pooling model 输出 pooler_output。
```

---

## 5. 后续专题占位

```text
01_model_architecture_role.md：
  定义模型架构层的职责、边界，以及它和执行层 / 配置层的关系。

02_model_registry_and_resolution.md：
  梳理 registry 如何根据 HF architectures 字段解析到 vLLM model class。

03_model_config_and_task_detection.md：
  梳理 ModelConfig 如何识别 task、runner_type、支持能力、dtype 和模型长度。

04_model_class_construction.md：
  梳理一个 vLLM model class 如何构造 embedding、layers、norm、lm_head / pooler。

05_forward_interface_contract.md：
  梳理 ModelRunner 对 model.forward、compute_logits、pooler、load_weights 的接口约定。

06_attention_mlp_norm_blocks.md：
  梳理 Attention、MLP、Norm、RoPE 等基础 block 如何在不同模型中复用。

07_embedding_and_lm_head.md：
  梳理 vocab parallel embedding、lm_head、tie weights、logits processor 的关系。

08_moe_model_architectures.md：
  梳理 MoE 模型中的 router、experts、fused MoE、shared experts 和 expert parallel。

09_pooling_embedding_rerank_models.md：
  梳理 embedding / classify / score / rerank 模型如何接入 model architecture。

10_multimodal_model_architectures.md：
  梳理 vision-language 模型中的 vision tower、projector、placeholder、M-RoPE 和 inputs_embeds。

11_weight_loading_and_name_mapping.md：
  梳理模型权重加载、fused layer 映射、TP 切分、checkpoint 命名差异。

12_quant_lora_parallelism_hooks.md：
  梳理模型架构如何为 quantization、LoRA、TP/PP/EP 提供 hook。

13_add_new_model_checklist.md：
  梳理新增模型架构时需要实现和验证的接口清单。
```

---

## 6. 一句话总结

model architectures 是 vLLM 把“各种外部模型”变成“统一执行接口”的适配层：

```text
它屏蔽模型结构差异，
复用 vLLM 的 attention / linear / MoE / embedding / norm 组件，
并让 Worker / ModelRunner 可以用统一方式执行不同模型。
```
