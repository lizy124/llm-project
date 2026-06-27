# 04. QuantizedLinear 如何替代普通 Linear？

源码位置：

- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`

本问题关注：Linear layer 如何根据量化配置创建量化参数，并在 forward 中调用量化 kernel。

---

## 1. 一句话回答

vLLM 的量化 Linear 通常不是换一个模型结构，而是在 Linear layer 内部挂接 `quant_method`。

```text
Linear layer 初始化
  → 根据 QuantizationConfig 选择 quant_method
  → quant_method.create_weights()
  → weight loader 加载 qweight / scales
  → forward 调 quant_method.apply()
```

---

## 2. Linear 层占位

需要梳理：

```text
- ColumnParallelLinear；
- RowParallelLinear；
- QKVParallelLinear；
- MergedColumnParallelLinear；
- ReplicatedLinear；
- vocab parallel embedding / lm_head 是否支持量化。
```

---

## 3. quant_method 接口占位

```text
create_weights：
  创建 layer 所需的量化参数。

apply：
  forward 时执行量化计算。

process_weights_after_loading：
  权重加载后做 packing / scale 处理。
```

---

## 4. forward 占位

```text
input activation
  → optional activation quant
  → quantized matmul
  → bias / all-reduce / gather
  → output activation
```

---

## 5. 一句话总结

```text
QuantizedLinear 的关键不是替换模型语义，而是替换权重表示和 matmul 实现。
```
