# 03. 量化权重如何加载和映射？

源码位置：

- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/parameter.py`

本问题关注：量化 checkpoint 中的 weight、scale、zero point、group size 等如何映射到 vLLM 参数。

---

## 1. 一句话回答

量化权重加载不是简单 `state_dict[name] = tensor`，而是要处理命名映射、tensor parallel 切分、packed weight、scale / zero point 以及量化方法专属 loader。

---

## 2. 主链路占位

```text
model_loader.load_weights()
  → checkpoint iterator
  → weight name mapping
  → model.load_weights()
  → parameter loader
  → quant_method 对应参数
      → qweight
      → scales
      → zero_points
      → g_idx / group metadata
      → packed params
```

---

## 3. 需要梳理的对象

```text
- Parameter / PackedParameter；
- weight_loader；
- shard_id / loaded_weight；
- column parallel / row parallel 权重切分；
- fused QKV / gate_up_proj 的名字映射；
- checkpoint quantization metadata；
- safetensors / pt / npc / bitsandbytes 等加载格式。
```

---

## 4. 容易混淆点占位

```text
1. quantization method 和 checkpoint load format 不是一回事。
2. scale / zero point 也是模型参数的一部分。
3. fused layer 的权重名经常和 checkpoint 名称不一一对应。
4. TP 下 qweight 和 scales 都可能需要切分。
5. 有些 kernel 要求 packed weight 的特殊布局。
```

---

## 5. 一句话总结

```text
量化权重加载的核心，是把 checkpoint 的低 bit 表示转换成 vLLM layer 和 kernel 能消费的参数布局。
```
