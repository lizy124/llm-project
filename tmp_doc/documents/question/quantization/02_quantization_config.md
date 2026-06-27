# 02. 用户量化配置如何变成 QuantizationConfig？

源码位置：

- `code/vllm/vllm/config.py`
- `code/vllm/vllm/config/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/platforms/`

本问题关注：用户指定的 `quantization`、`kv_cache_dtype`、load format 等配置如何进入模型配置和运行时。

---

## 1. 一句话回答

量化配置从用户参数进入 `VllmConfig`，再被拆到 `ModelConfig`、`CacheConfig`、`LoadConfig` 和具体 layer 的 `QuantizationConfig` 中。

```text
EngineArgs / CLI
  → VllmConfig
  → ModelConfig.quantization
  → QuantizationConfig
  → layer.quant_method
```

---

## 2. 配置来源占位

后续补充：

```text
- CLI 参数；
- EngineArgs；
- model config 中的 quantization_config；
- checkpoint 自带量化元数据；
- load format；
- kv_cache_dtype；
- platform 默认能力。
```

---

## 3. 需要区分的配置

```text
weight quantization：
  例如 awq / gptq / fp8 / marlin 等。

kv_cache_dtype：
  例如 fp8 / auto / int8 per-token-head 等。

activation quantization：
  可能由 quantization method 或模型配置决定。

load format：
  决定 checkpoint 如何被读取，不完全等同于 quantization method。
```

---

## 4. 校验占位

```text
需要检查：

- 当前平台是否支持该量化方式；
- 当前模型结构是否支持；
- head size / group size / hidden size 是否满足 kernel 约束；
- dtype 是否兼容；
- LoRA / MoE / attention backend 是否兼容。
```

---

## 5. 一句话总结

```text
量化配置不是只决定权重 dtype，而是会一路影响模型加载、layer 创建和 kernel 选择。
```
