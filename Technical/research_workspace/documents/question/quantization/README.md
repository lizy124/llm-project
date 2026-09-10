# Quantization 文档目录

本目录用于梳理 vLLM 中量化相关机制。

量化不是单一功能点，而是横跨：

```text
用户配置
  → QuantizationConfig
  → model config / load config
  → model registry / layer 创建
  → weight loader
  → quantized linear / MoE / attention / KV cache
  → kernel backend
  → runtime forward
  → 精度、显存、吞吐和兼容性
```

它要回答的核心问题是：

```text
vLLM 如何根据配置识别量化方式，
如何加载量化权重，
如何把普通 layer 替换成量化 layer，
如何选择对应 kernel，
以及量化如何影响 attention、KV cache、MoE、LoRA、并行和输出精度。
```

---

## 阅读顺序建议

```text
quantization_overview.md
  → 01_quantization_role.md
  → 02_quantization_config.md
  → 03_weight_loading_and_param_mapping.md
  → 04_quantized_linear_layers.md
  → 05_weight_only_quantization.md
  → 06_activation_and_dynamic_quantization.md
  → 07_kv_cache_quantization.md
  → 08_attention_backend_interaction.md
  → 09_moe_quantization.md
  → 10_lora_and_quantization.md
  → 11_parallelism_and_quantization.md
  → 12_accuracy_performance_tradeoffs.md
  → 13_limitations_and_debugging.md
```

如果只想先抓主线，可以先读：

```text
quantization_overview.md
  → 02_quantization_config.md
  → 03_weight_loading_and_param_mapping.md
  → 04_quantized_linear_layers.md
  → 07_kv_cache_quantization.md
  → 08_attention_backend_interaction.md
```

---

## 文档定位

```text
quantization_overview.md：
  总览主文档，建立 vLLM 量化机制的全局图。

01-13：
  按问题拆开的专题文档，后续逐篇补源码细节。
```
