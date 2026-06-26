# 06. 权重加载和量化如何接入？

源码位置：

- `code/vllm/vllm/model_executor/model_loader/weight_utils.py`
- `code/vllm/vllm/model_executor/model_loader/loader.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/config/quantization.py`
- `code/vllm/vllm/model_executor/models/`

本问题关注：模型实例创建后，vLLM 如何查找权重文件、迭代权重 tensor、映射到模型参数，以及量化配置如何影响权重加载、layer 创建和 kernel 选择。

---

## 1. 一句话回答占位

占位：后续补充权重加载是 `ModelLoader` 把外部 checkpoint tensor 映射到 vLLM 模型参数的过程，量化会改变参数解释和 layer 实现。

```text
weight files
  → weights iterator
  → model.load_weights()
  → parameter mapping / shard loading
  → quantization method / packed weights
  → ready model
```

---

## 2. 权重格式占位

```text
- safetensors；
- PyTorch bin；
- sharded checkpoint；
- tensorizer；
- bitsandbytes；
- GGUF / 其他特殊格式；
- dummy weights；
- remote / local cached weights。
```

---

## 3. 量化接入点占位

```text
QuantizationConfig / quantization method：
  决定使用哪种量化方案。

模型 layer：
  根据 quantization method 创建对应 Linear / MoE / KV cache scale 等实现。

权重加载：
  读取 packed weights、scale、zero point、group size 等量化参数。

Attention / kernel：
  某些量化会影响 attention backend、KV cache dtype 或 custom op。
```

---

## 4. 后续待补源码证据

占位：补充 `safetensors_weights_iterator()`、模型 `load_weights()`、量化 config 创建、典型量化 layer。
