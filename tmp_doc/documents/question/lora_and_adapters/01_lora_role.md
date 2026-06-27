# 01. LoRA / adapter 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/lora/`
- `code/vllm/vllm/adapter_commons/`
- `code/vllm/vllm/config.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：LoRA 在 vLLM 中的职责、边界，以及它和 base model、量化、sampling 的关系。

---

## 1. 一句话回答

LoRA 是在 base model 上动态叠加的低秩 adapter 权重，用于让不同请求在共享同一个 base model 的同时使用不同的增量能力。

```text
base model：
  常驻、共享、一次加载。

LoRA adapter：
  可动态加载、卸载、缓存、pin；
  请求级选择；
  forward 时叠加到对应 layer。
```

---

## 2. LoRA 解决什么问题

占位：

```text
- 一个服务同时支持多个微调任务；
- 避免为每个任务加载一份完整模型；
- 让请求级选择 adapter；
- 支持 batch 内混合多个 adapter；
- 支持运行时 add / remove adapter。
```

---

## 3. LoRA 不负责什么

```text
- 不负责调度 token budget；
- 不直接改变 sampling 规则；
- 不直接改变 KV cache 分配方式；
- 不是 tokenizer 或 prompt template；
- 不等同于完整模型热切换。
```

---

## 4. 一句话总结

```text
LoRA 是 vLLM 中请求级可切换的增量权重机制。
```
