# Model Architectures 文档目录

本目录用于梳理 vLLM 中模型架构适配机制。

模型架构不是单个模型文件的问题，而是横跨：

```text
model config
  → task detection
  → model registry
  → architecture class resolution
  → model class construction
  → layer 组件复用
  → weight loading mapping
  → forward interface
  → logits / pooling output
```

它要回答的核心问题是：

```text
vLLM 如何识别一个 HuggingFace 模型应该用哪个 model class，
每个 model class 如何复用 attention / MLP / MoE / embedding / norm / logits processor 等组件，
forward 接口如何与 ModelRunner 对齐，
以及新增模型架构时需要实现哪些约定。
```

---

## 阅读顺序建议

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

如果只想先抓主线，可以先读：

```text
model_architectures_overview.md
  → 02_model_registry_and_resolution.md
  → 04_model_class_construction.md
  → 05_forward_interface_contract.md
  → 11_weight_loading_and_name_mapping.md
```

---

## 文档定位

```text
model_architectures_overview.md：
  总览主文档，建立 vLLM 模型架构适配机制的全局图。

01-13：
  按问题拆开的专题文档，后续逐篇补源码细节。
```
