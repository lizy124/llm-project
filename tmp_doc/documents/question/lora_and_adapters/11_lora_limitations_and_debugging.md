# 11. LoRA 有哪些限制和调试入口？

源码位置：

- `code/vllm/vllm/lora/`
- `code/vllm/vllm/config.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/executor/`

本问题关注：LoRA 常见不支持场景、错误、fallback 和调试路径。

---

## 1. 一句话回答

LoRA 问题通常来自 adapter 配置、target module 映射、rank 限制、并行切分、量化兼容或动态加载状态不一致。

---

## 2. 常见限制占位

```text
- 模型未启用 LoRA；
- adapter rank 超过 max_lora_rank；
- target module 不存在或不支持；
- fused layer 映射失败；
- 量化方式不支持 LoRA；
- TP / PP 下权重切分不匹配；
- adapter 数量超过 max_loras；
- LoRA path / config / safetensors 缺失；
- CUDA graph 与动态 LoRA mapping 冲突。
```

---

## 3. 调试入口占位

```text
1. 检查 VllmConfig 中 LoRA 是否启用。
2. 检查 LoRARequest 的 name / id / path。
3. 检查 adapter_config.json。
4. 检查 target_modules 是否匹配 vLLM 模型层名。
5. 检查 worker list_loras。
6. 检查 InputBatch active LoRA mapping。
7. 检查 LoRA layer 是否被注入。
8. 检查量化 / 并行组合是否支持。
```

---

## 4. 常见报错分类占位

```text
配置错误：
  LoRA 未启用、rank 超限、max_loras 超限。

加载错误：
  path 不存在、adapter config 缺失、权重名不匹配。

运行时错误：
  active LoRA 未加载、mapping 不一致、kernel 不支持。

输出异常：
  target module 未注入、adapter 权重未生效、dtype 不匹配。
```

---

## 5. 一句话总结

```text
调试 LoRA 要沿着“请求 LoRARequest → manager 加载状态 → layer 注入 → active mapping → forward 输出”逐层定位。
```
