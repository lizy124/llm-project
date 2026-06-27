# 07. LoRA 权重如何加载和映射？

源码位置：

- `code/vllm/vllm/lora/models.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/model_executor/models/`

本问题关注：LoRA checkpoint 中的权重如何按 target modules 映射到 vLLM 模型 layer。

---

## 1. 一句话回答

LoRA 权重加载的核心是把 checkpoint 中的 adapter 权重名，映射到 vLLM 内部模型 layer，并处理 rank、alpha、scaling 和 fused layer 的名字差异。

---

## 2. LoRA checkpoint 内容占位

```text
- adapter_config.json；
- rank / r；
- lora_alpha；
- target_modules；
- lora_A；
- lora_B；
- bias；
- modules_to_save；
- embedding / lm_head adapter 权重。
```

---

## 3. 权重映射占位

需要梳理：

```text
- checkpoint module name 到 vLLM layer name；
- fused QKV / gate_up_proj 如何映射；
- TP 下权重如何切分；
- embedding / lm_head 如何处理 vocab parallel；
- 不支持 target module 如何报错；
- rank 超过 max_lora_rank 如何处理。
```

---

## 4. 加载流程占位

```text
add_lora(request)
  → manager 检查 cache
  → 读取 adapter config
  → 读取 adapter weights
  → 映射到 LoRA model
  → 上传到 GPU / LoRA slots
  → 标记 loaded
```

---

## 5. 一句话总结

```text
LoRA 权重加载的关键，是让外部 adapter checkpoint 的命名和 vLLM 内部 fused/parallel layer 对齐。
```
