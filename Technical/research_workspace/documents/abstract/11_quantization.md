# 11 quantization 背诵文档

## 1. 专题定位

`quantization` 讲的是 vLLM 如何把用户配置或 checkpoint 里的量化信息，转成模型 layer 的参数布局和运行时 kernel 调用。

它不是一个单独开关。

一句话：

```text
vLLM 的量化是一套从配置、权重加载、参数布局到 kernel dispatch 的跨层协议。
```

## 2. 最小心智模型

主链路：

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

要背住：

```text
QuantizationConfig 负责全局策略，quant_method 负责单个 layer 的参数创建、加载后处理和 forward 执行。
```

## 3. 三个核心对象

### ModelConfig.quantization

量化方法名。

例如：

```text
awq
gptq
fp8
compressed-tensors
bitsandbytes
online
```

它回答：

```text
这个模型用哪种量化方法。
```

### VllmConfig.quant_config

解析后的 `QuantizationConfig` 对象。

它回答：

```text
当前量化方法如何作用到具体 layer。
```

### layer.quant_method

单个 layer 的量化执行对象。

它负责：

```text
create_weights()
process_weights_after_loading()
apply()
```

一句话：

```text
quantization 是方法名，quant_config 是全局配置对象，quant_method 是 layer 级执行对象。
```

## 4. 量化机制的六条主线

```text
1. 配置解析：
   --quantization / HF quantization_config / load_format / kv_cache_dtype。

2. 权重加载和参数映射：
   checkpoint 中 qweight / scales / qzeros / g_idx 如何映射到 vLLM 参数。

3. Linear / MoE 权重量化：
   LinearBase、RoutedExperts 如何通过 quant_method 创建参数并调用 kernel。

4. activation / dynamic quantization：
   forward 时 activation scale、per-token scale、block scale 如何进入 kernel。

5. KV cache quantization：
   kv_cache_dtype、KVQuantMode、KVCacheSpec、scale、page size、slot layout。

6. backend / 并行 / LoRA / 调试限制：
   attention backend、TP/PP/EP、LoRA、性能精度和常见错误。
```

## 5. 全局对象关系

```text
EngineArgs / CLI / API
  ├─ quantization
  ├─ quantization_config
  ├─ load_format
  └─ kv_cache_dtype
       ↓
ModelConfig
  ├─ quantization
  ├─ quantization_config
  └─ hf_config.quantization_config / compression_config
       ↓
VllmConfig
  └─ quant_config
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

## 6. 阶段边界

量化链路可以按阶段背：

```text
配置阶段：
  决定用不用量化、用哪种量化。

初始化阶段：
  决定这个 layer 有哪些量化参数。

加载阶段：
  checkpoint tensor 如何落到本 rank 参数。

后处理阶段：
  参数如何变成 kernel-ready layout。

执行阶段：
  forward 调哪个 quantized kernel。
```

## 7. 术语表

```text
quantization：
  用户或 checkpoint 给出的量化方法名。

quantization_config：
  可能指 HF checkpoint 元数据，也可能指用户侧在线量化配置。

QuantizationConfig：
  vLLM 内部量化配置对象，负责给 layer 选择 quant_method。

quant_config：
  VllmConfig 上保存的 QuantizationConfig 实例。

QuantizeMethodBase / quant_method：
  绑定到 layer 的执行对象，负责 create_weights / post-load / apply。

kv_cache_dtype：
  KV cache 存储格式，属于 cache 配置线，不等同于权重量化。

KVQuantMode：
  kernel 侧 KV cache 量化枚举，例如 NONE / FP8 / per-token-head / NVFP4。
```

## 8. 配置如何进入

量化配置来源：

```text
CLI / Python API 的 quantization
用户传入 quantization_config
HF config.quantization_config
HF compression_config
load_format 特殊值
kv_cache_dtype
```

`ModelConfig` 会：

```text
读取 HF config
读取 checkpoint quantization metadata
和用户 quantization 参数合并 / 校验
调用 _verify_quantization()
```

`VllmConfig` 最终保存：

```text
quant_config
```

## 9. load_format 和 quantization 的关系

`load_format` 不是量化方法。

```text
load_format：决定怎么读 checkpoint 文件。
quantization：决定读出来的权重如何解释、如何后处理、forward 用哪个 kernel。
```

特殊例子：

```text
bitsandbytes 量化会反过来影响 load_format。
```

因此：

```text
load_format 解决 IO / loader 问题。
quantization 解决参数语义 / kernel 问题。
```

## 10. 权重量化主线

权重量化主要作用于：

```text
Linear layer
QKV projection
MLP projection
LM head
MoE experts
```

主链路：

```text
LinearBase 初始化
  → quant_config.get_quant_method(layer, prefix)
  → layer.quant_method
  → quant_method.create_weights()
  → loader 加载 qweight / scale / zero point
  → process_weights_after_loading()
  → quant_method.apply(x, weight)
  → quantized matmul kernel
```

一句话：

```text
量化 layer 不是普通 Parameter 加一个标志，而是由 quant_method 创建和解释自己的权重参数。
```

## 11. weight-only quantization

weight-only 量化常见于：

```text
GPTQ
AWQ
INT4 / INT8
Marlin
compressed-tensors
FP8 weight-only
```

特点：

```text
权重低精度存储。
activation 多数仍用 fp16 / bf16。
forward 时用特定 kernel 解包 / dequant / matmul。
```

需要处理：

```text
qweight
scales
zero points
group size
packed layout
TP shard offset
kernel-specific repack
```

## 12. activation / dynamic quantization

activation quantization 不只改权重。

它还会在 forward 时处理：

```text
activation scale
per-token scale
per-channel scale
per-block scale
dynamic scale compute
scaled mm
```

区别：

```text
weight-only：主要参数在加载阶段准备。
dynamic activation quant：forward 时还要根据输入动态计算或应用 scale。
```

## 13. KV cache quantization

KV cache 量化是独立主线。

它由：

```text
kv_cache_dtype
CacheConfig
KVQuantMode
KVCacheSpec
attention backend
```

共同决定。

它控制：

```text
历史 key/value 用什么 dtype 存储。
写入 KV cache 时如何 scale / quantize。
attention 读取 KV cache 时如何 dequant / 使用 scale。
backend 是否支持该 KV dtype。
```

重要区分：

```text
FP8 权重量化：模型参数低精度。
FP8 KV cache：runtime 产生的历史 K/V 低精度。
```

这两者不是一回事。

## 14. attention backend 与量化

attention backend 会受量化影响：

```text
kv_cache_dtype
KV scale
head_size
block_size
use_mla
FP8 / NVFP4 支持
backend validate_configuration()
```

有些 backend 支持某种 KV cache dtype，有些不支持。

因此量化可能导致：

```text
backend 选择变化
fallback
报不支持
CUDA graph 降级
```

## 15. MoE 量化

MoE 量化作用于：

```text
router 后的 expert weights
fused MoE kernel
grouped GEMM
shared expert
expert parallel 通信后的 local expert compute
```

要处理：

```text
每个 expert 的 qweight / scale
expert weight layout
top-k routing 后 token-expert 分组
EP 下 expert 分布
quantized grouped GEMM backend
```

MoE 量化比普通 Linear 更复杂，因为它同时受：

```text
router
token dispatch
expert layout
EP all2all
kernel group size
```

影响。

## 16. LoRA 与量化

LoRA 通常叠加在量化 base layer 上。

典型语义：

```text
base output = quantized_base_layer(x)
lora delta = x @ A @ B * scaling
output = base output + lora delta
```

要注意：

```text
base weight 可以是 GPTQ / AWQ / FP8 / compressed。
LoRA A/B 通常按 lora_dtype 加载。
LoRA wrapper 需要理解 base layer 的设备、并行和权重布局。
```

LoRA 不一定能和所有量化 kernel 任意组合。

## 17. 并行与量化

并行会影响量化参数。

### TP

```text
权重按 rank shard。
scale / zero point / group size 也要按 shard 对齐。
QKV / gate_up fused weight 要处理 shard_id。
```

### PP

```text
每个 PP stage 只加载本 stage layers 的量化权重。
missing layer 不应加载参数。
```

### EP

```text
expert weight 分布在 expert ranks。
量化 expert kernel 要和 EP token dispatch 对齐。
```

一句话：

```text
量化参数不是全局一整块加载，而要按当前 rank 的模型分片加载和解释。
```

## 18. process_weights_after_loading

权重加载后还需要后处理。

可能包括：

```text
repack
transpose
scale finalize
online quant
weight layout conversion
kernel-specific format conversion
attention layer post-load processing
```

原因：

```text
checkpoint 存储格式不一定等于 runtime kernel 最优格式。
```

## 19. online quantization

online quantization 指加载普通或中间格式权重后，在 vLLM 内部转换成量化格式。

可能流程：

```text
加载 fp16 / bf16 权重
  → quant_method.finalize_layerwise_processing()
  → 生成 qweight / scales
  → 转成 kernel-ready layout
```

这类路径通常对内存、初始化时间、精度和 kernel 支持有更多限制。

## 20. 精度 / 性能 / 显存取舍

量化的目标通常是：

```text
降低显存
提高吞吐
降低带宽压力
让更大模型装进设备
```

代价可能是：

```text
精度下降
kernel 兼容性下降
部分 shape 变慢
额外 scale / dequant 开销
CUDA graph 支持受限
LoRA / MoE / parallel 组合限制
```

一句话：

```text
量化不是一定更快，它取决于模型结构、硬件、batch 形态、kernel 和并行方式。
```

## 21. 常见误区

### --quantization 和 --kv-cache-dtype 是一回事吗

不是。

```text
--quantization：控制权重 / activation / MoE layer 量化。
--kv-cache-dtype：控制 runtime 历史 K/V cache 存储格式。
```

### load_format 是量化方法吗

不是。

```text
load_format 决定怎么读文件。
quantization 决定如何解释和执行权重。
```

### checkpoint quantization_config 一定等于 runtime 方法吗

不一定。

vLLM 可能把 checkpoint 存储格式映射到更合适的 runtime backend。

例如：

```text
某些 GPTQ / AWQ checkpoint 可以运行时选择 Marlin / ModelOpt / 专用 kernel。
```

### FP8 权重和 FP8 KV cache 是一回事吗

不是。

```text
FP8 权重：模型参数低精度。
FP8 KV cache：推理过程中产生的 key/value 低精度。
```

## 22. 调试入口

量化问题常看：

```text
ModelConfig.quantization
VllmConfig.quant_config
LoadConfig.load_format
layer.quant_method
qweight / scale / zero point shape
TP shard offset
process_weights_after_loading 是否执行
selected backend / fallback
attention backend validate_configuration
KV cache dtype support
```

常见错误：

```text
量化方法不支持当前模型结构。
checkpoint 权重名和 runtime 参数名不匹配。
scale shape 不对。
TP shard 切分不对。
backend 不支持当前 dtype / head_size / block_size。
LoRA 和量化组合不支持。
CUDA graph 因量化路径降级。
```

## 23. 与其他专题的关系

```text
config_and_model_loading：量化配置如何进入 ModelConfig / VllmConfig。
model_architectures：模型 layer 如何接收 quant_config。
operators：quantized matmul / MoE / KV cache kernel 如何执行。
attention：KV cache dtype 和 attention backend 支持矩阵。
parallelism：TP / PP / EP 下量化权重和 scale 如何切分。
lora_and_adapters：LoRA delta 如何叠加在量化 base layer 上。
compilation_and_cuda_graph：量化 kernel 是否支持 compile / graph。
```

## 24. 背诵总结

背这一段：

```text
vLLM 的量化不是一个开关，而是一套跨配置、模型层、权重加载和 kernel 的协议。用户参数或 checkpoint metadata 先进入 ModelConfig.quantization，再生成 VllmConfig.quant_config；模型 layer 构造时根据 QuantizationConfig 获取自己的 quant_method，quant_method.create_weights 创建 qweight、scale、zero point 等参数；loader 把 checkpoint tensor 写入这些参数，process_weights_after_loading 把它们转成 kernel-ready layout；forward 时 quant_method.apply 或 attention / MoE backend 调用量化 kernel。权重量化、activation 量化和 KV cache 量化是不同主线，必须分别看配置、参数、backend 和并行切分。
```
