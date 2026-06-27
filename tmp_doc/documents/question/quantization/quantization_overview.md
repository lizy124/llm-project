# vLLM Quantization 逻辑梳理

源码位置：

- `code/vllm/vllm/config.py`
- `code/vllm/vllm/config/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/platforms/`

本文按“先定边界，再走配置到运行时主链路，再拆权重、KV cache、kernel、并行和精度性能影响”的方式，梳理 vLLM 中的量化机制。

量化不是一个孤立模块。它会影响：

```text
- 模型配置解析；
- 权重加载方式；
- Linear / MoE / Attention layer 的创建；
- kernel backend 选择；
- KV cache dtype / scale / layout；
- tensor parallel / pipeline parallel 下的权重切分；
- LoRA、spec decode、compile、CUDA graph 等运行时能力；
- 显存占用、吞吐、延迟和输出精度。
```

---

## 0. 梳理规划

本目录要回答的问题分成 13 组：

```text
1. 量化在 vLLM 中处于哪一层？解决什么问题？
2. 用户配置如何变成 QuantizationConfig？
3. 量化权重如何加载、映射、切分和反序列化？
4. QuantizedLinear 如何替代普通 Linear？
5. weight-only 量化如何影响 GEMM / kernel？
6. activation / dynamic quantization 如何参与 forward？
7. KV cache quantization 和 weight quantization 有什么区别？
8. 量化如何影响 attention backend 选择？
9. MoE / fused MoE 量化如何处理 expert 权重？
10. LoRA 和量化权重如何共存？
11. tensor parallel / pipeline parallel 下量化权重如何切分？
12. 精度、显存和性能之间如何取舍？
13. 常见限制、报错和调试入口有哪些？
```

阅读顺序建议：

```text
quantization_overview.md
  → 01_quantization_role.md
  → 02_quantization_config.md
  → 03_weight_loading_and_param_mapping.md
  → 04_quantized_linear_layers.md
  → 05_weight_only_quantization.md
  → 06_activation_and_dynamic_quantization.md
  → 07_kv_cache_quantization.md
  → 08_attention_backend_interaction.md
  → 09_moe_quantization.md
  → 10_lora_and_quantization.md
  → 11_parallelism_and_quantization.md
  → 12_accuracy_performance_tradeoffs.md
  → 13_limitations_and_debugging.md
```

---

## 1. 一句话回答

vLLM 的量化机制本质上是：

```text
用更低 bit 或特殊格式保存 / 计算权重、激活或 KV cache，
再通过 QuantizationConfig、量化 layer、weight loader 和专用 kernel，
在尽量保持精度的前提下降低显存、提升吞吐或支持更大模型。
```

最小主线是：

```text
用户指定 quantization / kv_cache_dtype
  → config 解析成量化配置
  → model loader 读取量化权重和 scale / zero point
  → model layer 创建时选择 quant_method
  → quant_method 创建量化参数
  → weight loader 填充参数
  → forward 调用量化 kernel
  → attention / MoE / Linear / KV cache 按对应 dtype 和 layout 执行
```

---

## 2. 量化的几个层次

```text
权重量化：
  例如 GPTQ / AWQ / Marlin / FP8 / INT8 / INT4 等，主要影响模型参数存储和 Linear / MoE kernel。

激活量化：
  forward 中对 activation 做动态或静态量化，影响输入到 GEMM 的数值格式。

KV cache 量化：
  历史 K/V cache 使用 FP8、INT8 per-token-head、NVFP4 等格式，主要影响 attention backend 和 KV cache layout。

kernel 量化：
  某些量化格式需要专用 matmul / fused MoE / attention kernel。

加载格式量化：
  checkpoint 中的权重、scale、zero point、group size 等如何映射到 vLLM 参数。
```

---

## 3. 总体流程图

```text
EngineArgs / CLI / API 配置
  → VllmConfig
  → ModelConfig / CacheConfig / LoadConfig
  → QuantizationConfig / kv_cache_dtype
  → model registry 创建模型
  → layer 创建
      → LinearBase / QuantizedLinear
      → FusedMoE
      → Attention / KV cache spec
  → quant_method.create_weights()
      → 创建 weight / scale / zero point / packed params
  → model_loader.load_weights()
      → checkpoint tensor
      → parameter name mapping
      → tensor parallel slicing
      → quantized param loader
  → model forward
      → quant_method.apply()
      → quantized GEMM / fused kernel
      → attention backend 使用量化 KV cache
  → logits / output
```

---

## 4. 和其他专题的关系

```text
config_and_model_loading：
  解释量化配置如何被解析，以及模型加载如何选择 quantization。

executor_worker_model_runner：
  解释量化模型如何进入 worker，并在 forward 中执行。

attention：
  解释 KV cache quantization 和 attention backend 的关系。

parallelism：
  解释量化权重在 TP / PP / EP 下如何切分和通信。

sampling_and_output：
  通常不直接关心量化，但 logits 精度和数值稳定性会间接受影响。

lora_and_adapters：
  解释量化 base model 与 LoRA adapter 如何共存。
```

---

## 5. 后续专题占位

```text
01_quantization_role.md：
  定义量化在 vLLM 中的边界，区分权重量化、激活量化、KV cache 量化和 kernel 支持。

02_quantization_config.md：
  梳理用户配置如何转成 QuantizationConfig、CacheConfig 和 backend 限制。

03_weight_loading_and_param_mapping.md：
  梳理量化 checkpoint 中 weight / scale / zero point 如何映射到 vLLM 参数。

04_quantized_linear_layers.md：
  梳理 Linear layer 如何通过 quant_method 创建权重并调用量化 kernel。

05_weight_only_quantization.md：
  梳理 GPTQ / AWQ / Marlin / INT4 / INT8 / FP8 等 weight-only 路径。

06_activation_and_dynamic_quantization.md：
  梳理 activation quant、dynamic quant、per-token scaling 等运行时量化。

07_kv_cache_quantization.md：
  梳理 KV cache dtype、scale、layout、KVQuantMode 与 attention backend 的关系。

08_attention_backend_interaction.md：
  梳理量化如何影响 FlashAttention / FlashInfer / Triton / MLA backend 选择。

09_moe_quantization.md：
  梳理 fused MoE、expert 权重、routing 和量化 kernel 的关系。

10_lora_and_quantization.md：
  梳理量化 base model 与 LoRA adapter 的共存方式和限制。

11_parallelism_and_quantization.md：
  梳理 TP / PP / EP 下量化权重、scale、group size 的切分。

12_accuracy_performance_tradeoffs.md：
  梳理显存、吞吐、延迟、精度和数值稳定性的取舍。

13_limitations_and_debugging.md：
  梳理常见不支持场景、错误信息、fallback 和调试入口。
```

---

## 6. 一句话总结

量化在 vLLM 中不是单个开关，而是一条贯穿配置、加载、layer、kernel 和 runtime 的执行链：

```text
配置决定量化方式，
loader 负责读入量化权重，
layer 持有量化参数，
kernel 执行量化计算，
attention / KV cache / MoE / parallelism 决定它能不能高效正确地跑。
```
