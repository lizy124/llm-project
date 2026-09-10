# 01. Quantization 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/config/quantization.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/cache.py`
- `code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/v1/attention/`

本问题只关注：量化在 vLLM 中的职责边界。具体配置解析、权重加载、Linear、KV cache、attention backend、MoE、LoRA、并行和调试分别在后续专题展开。

---

## 1. 一句话回答

量化在 vLLM 中负责把“用户 / checkpoint 描述的低精度格式”翻译成“layer 参数布局、加载规则、后处理流程和 runtime kernel 调用”。

它不是单一开关，而是一套跨模块协议：

```text
配置层：
  识别量化方法和用户配置。

参数层：
  决定 layer 上注册哪些 qweight / scale / zero point / packed 参数。

加载层：
  决定 checkpoint tensor 如何映射到 vLLM 参数。

后处理层：
  把加载后的参数转成 kernel-ready layout。

执行层：
  通过 quant_method.apply() 或 attention backend 调用低 bit kernel。
```

---

## 2. 量化负责哪些事

### 2.1 识别量化方法

量化负责识别：

```text
--quantization
--quantization-config
HF config.json quantization_config
compression_config
额外量化配置文件
online quantization shorthand
```

并把它们变成 vLLM 内部可用的：

```text
ModelConfig.quantization
VllmConfig.quant_config
QuantizationConfig 子类实例
```

具体见：`02_quantization_config.md`。

### 2.2 为 layer 选择 quant_method

量化负责让不同 layer 拿到自己的执行方法：

```text
LinearBase
  → LinearMethodBase / QuantizeMethodBase

RoutedExperts / FusedMoE
  → FusedMoEMethodBase

Attention
  → KV cache scale method / backend 相关 scale 处理

Embedding / LM head
  → embedding-aware quant method
```

核心分界是：

```python
quant_config.get_quant_method(layer, prefix)
```

### 2.3 创建量化参数

量化负责决定一个 layer 上应该有哪些参数，例如：

```text
weight
qweight
qzeros
scales
g_idx
weight_scale
input_scale
block_scale
k_scale / v_scale / q_scale / prob_scale
bnb_quant_state
```

这些参数通常还携带：

```text
weight_loader
input_dim / output_dim
packed_dim / packed_factor
TP shard 信息
```

具体见：`03_weight_loading_and_param_mapping.md`、`04_quantized_linear_layers.md`。

### 2.4 加载后处理

很多量化权重加载后不能直接执行，量化负责在：

```python
process_weights_after_loading()
```

阶段做：

```text
repack
transpose
scale 合并 / 修正
online quantize
kernel layout 转换
临时参数删除
Attention KV scale finalize
```

### 2.5 runtime kernel 调用

量化负责把 forward 接到对应 kernel：

```text
Linear.forward()
  → quant_method.apply()
  → GPTQ / AWQ / FP8 / INT8 / FP4 / Marlin / CUTLASS / Triton kernel

MoE forward
  → quant_method.apply() / apply_monolithic()
  → fused expert kernel

Attention forward
  → backend cache update / attention kernel
  → 量化 KV cache 读写和 dequant / fused compute
```

---

## 3. 量化不负责哪些事

量化不直接负责：

```text
1. 请求调度；
2. token budget；
3. prefix cache 命中决策；
4. KV block 分配和回收；
5. tokenizer / detokenizer；
6. sampling 策略；
7. 模型结构定义；
8. executor / worker 通信拓扑；
9. checkpoint 下载；
10. LoRA adapter 的请求级调度。
```

但量化会对这些模块产生约束。例如：

```text
- 某个量化 kernel 不支持当前 GPU，模型无法启动；
- KV cache dtype 会影响 attention backend 选择；
- 量化权重的 scale / packed layout 会影响 TP shard；
- 某些量化方法和 LoRA / CUDA graph / speculative decoding 存在兼容限制。
```

所以量化不是调度模块，但会限制执行模块能怎么跑。

---

## 4. 权重量化、激活量化、KV cache 量化的边界

### 4.1 权重量化

权重量化作用于静态模型参数。

```text
关注对象：
  Linear / Embedding / LM head / MoE expert weight。

典型参数：
  qweight / scales / qzeros / g_idx / packed weight。

典型问题：
  checkpoint 怎么加载，TP 怎么切，kernel layout 怎么 repack。
```

专题：`03_weight_loading_and_param_mapping.md`、`04_quantized_linear_layers.md`、`05_weight_only_quantization.md`。

### 4.2 激活量化

激活量化作用于 forward 中的输入 activation。

```text
关注对象：
  x / activation scale / per-token scale / block scale。

典型问题：
  scale 是静态还是动态，kernel 是否支持 W8A8 / W4A8 / FP8 activation。
```

专题：`06_activation_and_dynamic_quantization.md`。

### 4.3 KV cache 量化

KV cache 量化作用于推理过程中产生的历史 key/value cache。

```text
关注对象：
  cache_dtype / KVQuantMode / k_scale / v_scale / KVCacheSpec / page size。

典型问题：
  K/V 写 cache 时如何量化，读 cache 时如何 dequant，backend 是否支持。
```

专题：`07_kv_cache_quantization.md`。

### 4.4 backend / kernel 支持

backend 支持决定某个量化格式能否实际运行。

```text
关注对象：
  GPU capability / dtype / head_size / block_size / layout / backend feature。

典型问题：
  FlashAttention、FlashInfer、Triton、TurboQuant、MoE kernel 哪个能跑。
```

专题：`08_attention_backend_interaction.md`、`09_moe_quantization.md`、`13_limitations_and_debugging.md`。

---

## 5. 和其他模块的关系

```text
config：
  识别和校验量化方法，构造 QuantizationConfig。

model loader：
  读取 checkpoint，把 tensor 交给量化参数的 weight_loader。

layer：
  持有 quant_method，调用 create_weights / apply。

parameter：
  表示 qweight、scale、zero point 等量化参数及其 shard 规则。

kernel：
  真正执行低 bit GEMM、MoE 或 attention。

attention：
  处理 KV cache dtype、scale、layout 和 backend 支持。

parallelism：
  处理量化参数在 TP / PP / EP 下的切分和缺失层。

LoRA：
  叠加在量化 base model 上，需要额外兼容检查。
```

---

## 6. 本文与后续文档的边界

```text
本文：
  定义职责边界和概念分层。

02_quantization_config.md：
  展开配置如何变成 QuantizationConfig。

03_weight_loading_and_param_mapping.md：
  展开 checkpoint tensor 如何进入量化参数。

04/05/06：
  展开 Linear、weight-only、activation quantization。

07：
  展开 KV cache quantization 内部机制。

08：
  展开 KV cache 量化如何影响 attention backend 选择。

09/10/11/12/13：
  展开 MoE、LoRA、并行、性能精度、限制调试。
```

---

## 7. 容易混淆的点

### 7.1 `quantization` 是不是等于 `quant_config`？

不是。

```text
quantization：
  方法名字符串。

quant_config：
  解析后的 QuantizationConfig 对象。
```

### 7.2 `quantization_config` 是不是只有一种含义？

不是。

```text
HF config 里的 quantization_config：
  checkpoint 元数据。

用户传入的 --quantization-config：
  主要是在线量化配置。
```

### 7.3 KV cache 量化是不是权重量化？

不是。

```text
权重量化：
  处理静态模型参数。

KV cache 量化：
  处理推理时产生的历史 K/V cache。
```

### 7.4 量化是否一定提升性能？

不一定。

它可能降低显存、提升吞吐，但也可能带来：

```text
精度损失；
启动 repack 成本；
kernel fallback；
backend 不支持；
CUDA graph / LoRA / 并行限制。
```

---

## 8. 一句话总结

```text
量化在 vLLM 中负责把低精度模型或低精度 runtime cache 接入 layer、loader、parameter、backend 和 kernel；它不负责调度或模型语义，但会深刻影响执行层能否高效运行。
```
