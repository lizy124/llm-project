# vLLM Quantization 总览

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/config/vllm.py`
- `vllm/vllm/config/quantization.py`
- `vllm/vllm/config/cache.py`
- `vllm/vllm/model_executor/model_loader/weight_utils.py`
- `vllm/vllm/model_executor/model_loader/utils.py`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/v1/attention/`
- `vllm/vllm/v1/kv_cache_interface.py`

本文只建立 vLLM 量化机制的全局地图：量化从哪里进入、经过哪些对象、最后落到哪些 layer / kernel。具体细节下沉到 `01` 到 `13` 的专题文档，避免在总览里重复展开。

---

## 1. 一句话回答

vLLM 的量化不是一个单独开关，而是一套从“配置”贯穿到“参数布局”和“kernel dispatch”的协议。

最小主线是：

```text
用户参数 / checkpoint metadata
  → ModelConfig 识别量化方法
  → VllmConfig 构造 QuantizationConfig
  → 模型 layer 接收 quant_config
  → layer 获取 quant_method
  → quant_method.create_weights()
  → checkpoint 权重加载
  → process_weights_after_loading()
  → quant_method.apply() / attention backend
  → quantized kernel 执行
```

这里有三个核心对象：

```text
ModelConfig.quantization：
  量化方法名，例如 awq / gptq / fp8 / compressed-tensors / online。

VllmConfig.quant_config：
  解析后的 QuantizationConfig 对象，描述当前量化方法如何作用到 layer。

layer.quant_method：
  单个 layer 的执行对象，负责参数创建、加载后处理和 forward。
```

---

## 2. 量化机制分成哪些主线

本目录把 vLLM 量化拆成六条主线：

```text
1. 配置解析
   --quantization / --quantization-config / HF quantization_config / load_format / kv_cache_dtype。

2. 权重加载和参数映射
   checkpoint 中 qweight / scales / qzeros / g_idx 如何映射到 vLLM 参数。

3. Linear / MoE 权重量化
   LinearBase、RoutedExperts 如何通过 quant_method 创建参数并调用 kernel。

4. activation / dynamic quantization
   forward 时 activation scale、per-token scale、block scale 如何进入 kernel。

5. KV cache quantization
   cache_dtype、KVQuantMode、KVCacheSpec、scale、page size、slot layout。

6. backend / 并行 / LoRA / 调试限制
   attention backend、TP/PP/EP、LoRA、性能精度和常见错误。
```

总览只保留这些主线之间的关系；每条主线的源码细节由对应专题展开。

---

## 3. 全局对象关系

```text
EngineArgs / CLI / API
  ├─ quantization
  ├─ quantization_config
  ├─ load_format
  └─ kv_cache_dtype
       ↓
ModelConfig
  ├─ quantization                # 方法名
  ├─ quantization_config         # 用户侧在线量化配置
  └─ hf_config.quantization_config / compression_config
       ↓
VllmConfig
  └─ quant_config                # QuantizationConfig 实例
       ↓
Model / Layer
  ├─ LinearBase.quant_method
  ├─ RoutedExperts.quant_method
  ├─ Attention quant scales
  └─ Embedding / LM head quant method
       ↓
ModelLoader / weight_loader
  └─ checkpoint tensor → vLLM Parameter
       ↓
process_weights_after_loading()
  └─ repack / transpose / online quant / scale finalize
       ↓
forward
  ├─ quant_method.apply()
  ├─ fused MoE kernel
  └─ attention backend / KV cache kernel
```

这个图里最重要的边界是：

```text
配置阶段：决定“用不用量化、用哪种量化”。
初始化阶段：决定“这个 layer 有哪些量化参数”。
加载阶段：决定“checkpoint tensor 如何落到本 rank 参数”。
后处理阶段：决定“参数如何变成 kernel-ready layout”。
执行阶段：决定“forward 调哪个 quantized kernel”。
```

---

## 4. 术语表

```text
quantization：
  用户或 checkpoint 给出的量化方法名。

quantization_config：
  两种语境：HF checkpoint 元数据，或用户侧在线量化配置。

QuantizationConfig：
  vLLM 内部的量化配置对象，负责从全局配置为具体 layer 选择 quant_method。

quant_config：
  VllmConfig 上保存的 QuantizationConfig 实例。

QuantizeMethodBase / quant_method：
  绑定到 layer 的执行对象，负责 create_weights / process_weights_after_loading / apply。

cache_dtype / kv_cache_dtype：
  KV cache 存储格式，属于 cache 配置线，不等同于权重量化方法。

KVQuantMode：
  从 kv_cache_dtype 派生出的 kernel 侧枚举，用于区分 NONE / FP8 / per-token-head / NVFP4。

attention backend：
  Attention 的运行时实现，负责 KV cache shape、metadata build、prefill/decode kernel。
```

---

## 5. 各专题文档分工

### 5.1 边界和配置

```text
01_quantization_role.md
  只回答“量化在 vLLM 中负责什么、不负责什么”，不展开每条源码细节。

02_quantization_config.md
  只回答“用户配置 / checkpoint metadata 如何变成 QuantizationConfig”。
```

### 5.2 权重和 layer

```text
03_weight_loading_and_param_mapping.md
  专注 checkpoint 参数、weight_loader、packed mapping、TP shard。

04_quantized_linear_layers.md
  专注 LinearBase、LinearMethod、create_weights、apply。

05_weight_only_quantization.md
  专注 GPTQ / AWQ / Marlin / INT4 / INT8 / FP8 weight-only 路径。

06_activation_and_dynamic_quantization.md
  专注 activation quant、dynamic quant、per-token / per-block scale。
```

### 5.3 KV cache 和 backend

```text
07_kv_cache_quantization.md
  专注 KV cache 的 dtype、scale、KVQuantMode、KVCacheSpec、page size、cache layout。
  只说明 backend 是 KV cache 的消费者，不展开 backend 选择策略。

08_attention_backend_interaction.md
  专注 backend selection：AttentionSelectorConfig、validate_configuration、FlashAttention / FlashInfer / Triton / TurboQuant 支持矩阵。
  只引用 KV cache dtype / scale 作为输入，不重复解释 KV cache 分配机制。
```

### 5.4 MoE、LoRA、并行和调试

```text
09_moe_quantization.md
  专注 fused MoE、RoutedExperts、expert weight、MoE kernel。

10_lora_and_quantization.md
  专注 LoRA adapter 与量化 base model 的共存边界。

11_parallelism_and_quantization.md
  专注 TP / PP / EP 下量化参数、scale、group size 的切分。

12_accuracy_performance_tradeoffs.md
  专注显存、吞吐、延迟、精度和数值稳定性取舍。

13_limitations_and_debugging.md
  专注不支持场景、报错、fallback 和排查入口。
```

---

## 6. 阅读顺序

推荐顺序：

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

如果只想先抓主线：

```text
quantization_overview.md
  → 01_quantization_role.md
  → 02_quantization_config.md
  → 03_weight_loading_and_param_mapping.md
  → 04_quantized_linear_layers.md
  → 07_kv_cache_quantization.md
  → 08_attention_backend_interaction.md
```

---

## 7. 常见误区

### 7.1 `--quantization` 和 `--kv-cache-dtype` 是一回事吗？

不是。

```text
--quantization：
  主要控制权重 / activation / MoE 等 layer 量化方法。

--kv-cache-dtype：
  控制推理时历史 K/V cache 的存储格式。
```

### 7.2 `load_format` 是量化方法吗？

不是。

```text
load_format：
  决定怎么读 checkpoint 文件。

quantization：
  决定读出来的权重如何解释、如何后处理、forward 用哪个 kernel。
```

### 7.3 checkpoint 里的 `quantization_config` 一定等于最终运行方法吗？

不一定。

vLLM 会通过 `override_quantization_method()` 把 checkpoint 存储格式映射到更合适的 runtime backend，例如 Marlin、ModelOpt 或模型专用量化实现。

### 7.4 FP8 权重量化和 FP8 KV cache 是一回事吗？

不是。

```text
FP8 权重量化：
  静态模型参数低精度存储和 GEMM 计算。

FP8 KV cache：
  runtime 产生的历史 key/value 低精度存储和 attention 读取。
```

---

## 8. 总结

```text
QuantizationConfig 负责全局策略；
quant_method 负责单个 layer 的参数和执行；
weight_loader 负责 checkpoint tensor 映射；
process_weights_after_loading 负责 kernel layout；
CacheConfig / KVQuantMode 负责 KV cache 量化这条独立路径；
attention backend 负责消费 KV cache dtype / scale / layout 并选择合适 kernel。
```

如果只记一句话：

```text
vLLM 量化是一套把 checkpoint / 用户配置翻译成 layer 参数布局和 runtime kernel 调用的协议。
```
