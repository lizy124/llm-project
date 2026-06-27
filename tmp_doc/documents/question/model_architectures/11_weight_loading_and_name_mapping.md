# 11. 模型权重如何加载和做名称映射？

源码位置：

- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/parameter.py`
- `code/vllm/vllm/model_executor/layers/linear.py`

本问题关注：model class 如何实现 `load_weights()`，以及 checkpoint 权重名如何映射到 vLLM 内部 fused / parallel layer。

---

## 1. 一句话回答

权重加载负责把 checkpoint 中的 tensor 名称和形状，映射到 vLLM model class 中的参数，并处理 fused layer、TP 切分、量化和特殊权重。

---

## 2. 主链路占位

```text
model_loader
  → checkpoint iterator
  → model.load_weights(weights)
  → name mapping
  → parameter weight_loader
  → shard / slice / pack
  → param.data
```

---

## 3. 常见映射问题

```text
- q_proj / k_proj / v_proj → fused qkv_proj；
- gate_proj / up_proj → fused gate_up_proj；
- embedding / lm_head tie weights；
- tensor parallel shard；
- quantized qweight / scales / zero points；
- MoE expert weights；
- vision encoder / projector weights；
- LoRA target modules。
```

---

## 4. 容易混淆点占位

```text
1. checkpoint 名称不一定等于 vLLM 参数名称。
2. fused layer 会把多个 checkpoint tensor 合并到一个参数。
3. TP 下只加载本 rank 需要的 shard。
4. 量化权重的 scale / zero point 也要加载。
5. 某些模型需要跳过或重命名权重。
```

---

## 5. 一句话总结

```text
load_weights 是模型架构适配 checkpoint 格式的关键接口。
```
