# 05. LoRA layer 如何注入 Linear / Embedding / LM head？

源码位置：

- `code/vllm/vllm/lora/layers.py`
- `code/vllm/vllm/lora/models.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/lora/punica_wrapper/`

本问题关注：LoRA adapter 如何挂到模型 layer 上，并在 forward 中叠加低秩 delta。

---

## 1. 一句话回答

LoRA layer 会包装或增强原始 Linear / Embedding / LM head，在 base output 基础上额外计算：

```text
LoRA delta = x @ A @ B * scaling
output = base_output + LoRA delta
```

---

## 2. 目标模块占位

需要梳理：

```text
- q_proj / k_proj / v_proj / o_proj；
- gate_proj / up_proj / down_proj；
- fused qkv_proj；
- fused gate_up_proj；
- embedding；
- lm_head；
- 模型特定 target modules 映射。
```

---

## 3. 注入流程占位

```text
加载 base model
  → 根据 LoRAConfig / target modules 找到可替换 layer
  → 替换为 LoRA-aware layer
  → manager 加载 adapter 权重
  → forward 时按 active mapping 选择 adapter
```

---

## 4. punica wrapper 占位

```text
punica wrapper 负责高效执行 batch mixed LoRA：

- 多 adapter；
- 多 request；
- token 到 LoRA id mapping；
- LoRA A / B 权重选择；
- fused kernel / batched matmul。
```

---

## 5. 一句话总结

```text
LoRA layer 是 base model 和 adapter 权重真正相遇的地方。
```
