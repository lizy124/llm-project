# vLLM Quantization 逻辑梳理

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\config\model.py`
- `E:\lizy\code\vllm-project\vllm\vllm\config\vllm.py`
- `E:\lizy\code\vllm-project\vllm\vllm\config\quantization.py`
- `E:\lizy\code\vllm-project\vllm\vllm\config\cache.py`
- `E:\lizy\code\vllm-project\vllm\vllm\platforms\interface.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\weight_utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\base_loader.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\__init__.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\base_config.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\kv_cache.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\linear.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\routed_experts.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\`

本文按“配置入口 → quant config 解析 → layer 接入 → 权重加载 → 后处理 → runtime kernel → KV cache / MoE / 并行限制”的顺序，梳理 vLLM 中量化机制的整体架构。

量化不是一个孤立模块。它会影响：

```text
- 用户配置和 HF checkpoint config 的解析；
- 模型初始化时 Linear / MoE / Attention layer 的参数创建；
- checkpoint 中 qweight / scales / zero point / g_idx 的加载；
- tensor parallel 下 packed weight 和 scale 的切分；
- process_weights_after_loading 阶段的 repack / transpose / online quant；
- forward 阶段调用的 GEMM / fused MoE / attention kernel；
- KV cache dtype、scale 和 attention backend 行为；
- 平台能力、activation dtype、CUDA graph、LoRA、spec decode 等兼容性。
```

---

## 1. 一句话回答

vLLM 的量化机制，本质上是一条从“配置”贯穿到“kernel”的执行链。

最小主线是：

```text
用户指定 --quantization / --quantization-config / --kv-cache-dtype
  → ModelConfig 校验量化方法
  → VllmConfig 构造 QuantizationConfig
  → 模型初始化时把 quant_config 传给 layer
  → Linear / MoE / Attention 创建 quant_method
  → quant_method.create_weights() 注册量化参数
  → ModelLoader 读取 checkpoint 并调用 param.weight_loader
  → process_weights_after_loading() 转换 kernel layout
  → forward 调用 quant_method.apply()
  → 专用 GEMM / MoE / Attention kernel 执行
```

所以：

```text
QuantizationConfig 决定“用哪种量化”；
quant_method 决定“这个 layer 怎么建参数、怎么执行”；
weight_loader 决定“checkpoint tensor 怎么映射到本 rank 参数”；
kernel 决定“runtime 怎么真正计算”。
```

---

## 2. 量化在 vLLM 中分成哪些层次

从功能边界看，vLLM 里至少有五类量化相关逻辑。

```text
1. 权重量化
   例如 GPTQ / AWQ / FP8 / MXFP4 / NVFP4 / INT8 / bitsandbytes。
   主要影响 Linear / Embedding / MoE 的 weight 参数和 GEMM kernel。

2. 激活量化
   forward 时对 activation 做动态或静态量化。
   主要影响 x 输入 GEMM 前的 scale、per-token scale、block scale 等。

3. KV cache 量化
   K/V cache 使用 fp8、fp8_e5m2、per-token-head scale 等格式。
   主要影响 Attention layer 的 cache 写入、读取、scale 和 backend。

4. checkpoint / loader 层量化
   checkpoint 里可能保存 qweight、scales、qzeros、g_idx、quant_state。
   loader 要把它们映射成 vLLM 参数。

5. kernel 层量化
   同一种 checkpoint 表示，最终可能走 Marlin、Cutlass、DeepGemm、Triton、TorchAO、BNB 等不同 backend。
```

这几层经常同时出现。

例如 GPTQ：

```text
checkpoint 中是 int4 packed qweight + scales + qzeros + g_idx；
layer 上注册 PackedvLLMParameter；
加载时做 TP shard 和 packed offset 调整；
加载后 kernel 可能 repack；
forward 时 quant_method.apply() 调 Marlin / MP linear kernel。
```

---

## 3. 源码主链路总图

从配置到运行时，可以画成：

```text
EngineArgs / CLI / API
  → ModelConfig.quantization
  → ModelConfig.quantization_config
  → CacheConfig.cache_dtype
  → ModelConfig._verify_quantization()
      → QUANTIZATION_METHODS
      → override_quantization_method(...)
      → current_platform.verify_quantization(...)
  → VllmConfig._get_quantization_config()
      → get_quant_config(model_config, load_config)
      → quant_config.get_supported_act_dtypes()
      → quant_config.get_min_capability()
      → quant_config.maybe_update_config(...)
  → initialize_model(...)
      → model receives vllm_config.quant_config
      → LinearBase / RoutedExperts / Attention initialize
  → quant_config.get_quant_method(layer, prefix)
      → quant_method
  → quant_method.create_weights(...)
      → weight / qweight / scales / qzeros / g_idx / scale params
  → ModelLoader.load_weights(...)
      → checkpoint iterator
      → model.load_weights(...)
      → param.weight_loader(...)
  → process_weights_after_loading(...)
      → quant_method.process_weights_after_loading(...)
      → Attention.process_weights_after_loading(...)
  → forward
      → Linear.forward() / RoutedExperts.forward_*()
      → quant_method.apply(...)
      → quantized kernel
```

这个图里最重要的分界是：

```text
配置阶段：决定有没有量化、用什么量化；
初始化阶段：创建什么参数；
加载阶段：checkpoint tensor 如何填入参数；
后处理阶段：参数如何变成 kernel-ready layout；
执行阶段：quant_method.apply 调哪个 kernel。
```

---

## 4. 支持的量化方法在哪里注册

量化方法列表在：`quantization/__init__.py`。

`QuantizationMethods` 包含：

```text
awq / auto_awq / awq_marlin
auto_gptq / gptq / gptq_marlin
fp8 / fbgemm_fp8 / fp_quant
modelopt / modelopt_fp4 / modelopt_mxfp8 / modelopt_mixed
compressed-tensors
bitsandbytes
experts_int8
quark
moe_wna16
torchao
inc
mxfp4 / gpt_oss_mxfp4
deepseek_v4_fp8
humming
online
fp8_per_tensor / fp8_per_block / fp8_per_channel / int8_per_channel_weight_only / mxfp8
```

位置：`quantization/__init__.py:12` 到 `quantization/__init__.py:47`

字符串到 Config 类的映射在：

```python
def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
```

位置：`quantization/__init__.py:108` 到 `quantization/__init__.py:182`

例如：

```text
awq / awq_marlin / auto_awq
  → AutoAWQConfig

auto_gptq / gptq / gptq_marlin
  → AutoGPTQConfig

fp8
  → Fp8Config

compressed-tensors
  → CompressedTensorsConfig

bitsandbytes
  → BitsAndBytesConfig

torchao
  → TorchAOConfig

mxfp4 / gpt_oss_mxfp4
  → Mxfp4Config / GptOssMxfp4Config

online / fp8_per_tensor / fp8_per_block / fp8_per_channel / mxfp8
  → OnlineQuantizationConfig 或 checkpoint 专属 config
```

自定义量化方法可以通过：

```python
register_quantization_config("my_quant")
```

注册。

位置：`quantization/__init__.py:58` 到 `quantization/__init__.py:105`

---

## 5. QuantizationConfig 是什么

所有量化配置类继承 `QuantizationConfig`。

位置：`base_config.py:77`

它定义了几个关键接口：

```text
get_name()
  返回量化方法名字。

get_supported_act_dtypes()
  返回支持的 activation dtype，例如 float16 / bfloat16。

get_min_capability()
  返回最低 GPU capability 要求。

get_config_filenames()
  返回要从模型目录查找的量化配置文件名。

from_config(config)
  从 HF quantization_config / 量化 json 构造配置。

override_quantization_method(...)
  根据 checkpoint config 自动纠正用户指定或 HF 标注的量化方法。

get_quant_method(layer, prefix)
  给某个 layer 返回实际 QuantizeMethodBase。

maybe_update_config(...)
  在拿到模型名 / HF config 后补充或修正配置。

get_cache_scale_mapper()
  提供 KV cache scale 名字映射。
```

位置：`base_config.py:85` 到 `base_config.py:212`

其中最关键的是：

```python
def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None:
```

位置：`base_config.py:157` 到 `base_config.py:170`

它决定：

```text
同一个 quant_config 下，哪些 layer 量化；
哪些 layer 跳过；
Linear 用什么 method；
MoE 用什么 method；
Attention KV scale 是否需要额外 method。
```

---

## 6. QuantizeMethodBase 是什么

`QuantizeMethodBase` 是每种量化运行时方法的基类。

位置：`base_config.py:19`

它有三个核心接口：

```text
create_weights(layer, ...)
  在 layer 上注册权重参数。

apply(layer, ...)
  forward 时执行量化计算。

process_weights_after_loading(layer)
  权重加载后做 repack / transpose / online quant / kernel 预处理。
```

位置：`base_config.py:27` 到 `base_config.py:62`

这三个函数对应量化生命周期的三个阶段：

```text
初始化：create_weights
加载后：process_weights_after_loading
运行时：apply
```

有些方法还会实现：

```text
embedding()
tie_weights()
```

用于 embedding / tied weights 场景。

---

## 7. ModelConfig 如何识别和校验量化

`ModelConfig._verify_quantization()` 是模型配置阶段的入口。

位置：`config/model.py:970`

它做几件事：

```text
1. 如果用户传了 quantization，先转成 QuantizationMethods；
2. 从 HF config 读取 quantization_config；
3. 调各量化 Config 的 override_quantization_method()；
4. 如果用户指定和 checkpoint 标注不一致，报错；
5. 检查 quantization 是否在 QUANTIZATION_METHODS 中；
6. 调 current_platform.verify_quantization() 做平台校验；
7. 拒绝默认禁用的 deprecated quantization。
```

核心位置：

```text
config/model.py:970 到 config/model.py:1072
```

特别注意 override 顺序。

代码里把这些方法放进 `overrides`：

```text
auto_gptq / gptq / gptq_marlin
auto_awq / awq / awq_marlin
inc
moe_wna16
modelopt / modelopt_fp4 / modelopt_mxfp8 / modelopt_mixed
mxfp4 / gpt_oss_mxfp4
deepseek_v4_fp8
humming
```

位置：`config/model.py:983` 到 `config/model.py:1002`

原因是：

```text
HF checkpoint 的 quant_method 字段不一定刚好等于 vLLM 运行时要用的实现；
例如 GPTQ checkpoint 可能要优先匹配 gptq_marlin / auto_gptq 等具体路径。
```

---

## 8. VllmConfig 如何真正构造 quant_config

`VllmConfig._get_quantization_config()` 是把 `model_config.quantization` 变成 `vllm_config.quant_config` 的位置。

位置：`config/vllm.py:609` 到 `config/vllm.py:642`

主逻辑是：

```text
if model_config.quantization is not None:
  quant_config = get_quant_config(model_config, load_config)
  检查 GPU capability >= quant_config.get_min_capability()
  检查 model_config.dtype in quant_config.get_supported_act_dtypes()
  quant_config.maybe_update_config(model_name, hf_config)
  return quant_config
else:
  return None
```

也就是说：

```text
ModelConfig 阶段负责“识别和校验量化名字”；
VllmConfig 阶段负责“构造具体 QuantizationConfig 对象并校验硬件/dtype”。
```

---

## 9. get_quant_config 如何找配置

`get_quant_config()` 在 `model_loader/weight_utils.py`。

位置：`weight_utils.py:240` 到 `weight_utils.py:394`

它按优先级寻找量化配置：

```text
1. model_config.hf_config.quantization_config
2. hf_config.text_config.quantization_config
3. hf_config.compression_config  # compressed-tensors
4. model_config.hf_overrides["quantization_config_file"]
5. model_config.hf_overrides["quantization_config_dict_json"]
6. model_config.quantization_config  # online quantization
7. bitsandbytes inflight 默认空配置
8. 模型目录中的量化 config json 文件
9. quant_cls.from_config(config)
```

这说明：

```text
同样是 --quantization，配置来源可能来自 HF config、额外 json、用户 override、online quantization args 或 checkpoint 专属文件。
```

compressed-tensors 还有一个特殊处理：

```text
把 total_num_heads / total_num_kv_heads 写入 hf_quant_config，
用于 TP-aware loading attention head scales。
```

位置：`weight_utils.py:257` 到 `weight_utils.py:273`

---

## 10. online quantization 是什么入口

online quantization 不是读取已经量化好的 checkpoint，而是在加载 fp16 / bf16 权重时在线量化。

用户侧配置在：`config/quantization.py`。

核心对象：

```text
QuantSpec
  - weight: QuantKey
  - activation: QuantKey

QuantizationConfigArgs
  - linear: QuantSpec | None
  - moe: QuantSpec | None
  - ignore: list[str]
```

位置：`config/quantization.py:63` 到 `config/quantization.py:110`

CLI shorthand 包括：

```text
fp8_per_tensor
fp8_per_block
fp8_per_channel
mxfp8
int8_per_channel_weight_only
online
```

位置：`config/quantization.py:112` 到 `config/quantization.py:144`

解析入口：

```python
def resolve_quantization_config(quantization, quantization_config)
```

位置：`config/quantization.py:147` 到 `config/quantization.py:183`

它的含义是：

```text
--quantization fp8_per_tensor
  → 展开成 QuantizationConfigArgs(linear=..., moe=...)

--quantization online --quantization-config {...}
  → 使用用户给的 linear / moe 量化规格
```

在线量化最终会走 `OnlineQuantizationConfig`，并在 `process_weights_after_loading()` 阶段把加载进来的浮点权重量化成目标格式。

---

## 11. Linear layer 如何接入量化

所有主要 Linear 都继承 `LinearBase`。

位置：`linear.py:228`

初始化时：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`linear.py:268` 到 `linear.py:274`

然后具体 Linear 调用：

```python
self.quant_method.create_weights(...)
```

例如：

```text
ReplicatedLinear：linear.py:338
ColumnParallelLinear：linear.py:461
RowParallelLinear：linear.py:1565
```

forward 时：

```python
output = self.quant_method.apply(self, x, bias)
```

位置：

```text
ReplicatedLinear：linear.py:378
ColumnParallelLinear：linear.py:555
RowParallelLinear：linear.py:1644
```

因此 Linear 的量化接入点非常清晰：

```text
LinearBase.__init__ 选 quant_method；
具体 Linear.__init__ 调 create_weights；
具体 Linear.forward 调 apply。
```

---

## 12. 未量化 Linear 也是一个 quant_method

未量化路径不是特殊 if-else 分散在 forward 中，而是 `UnquantizedLinearMethod`。

位置：`linear.py:179`

它也实现：

```text
create_weights()
  → 注册 ModelWeightParameter

process_weights_after_loading()
  → CPU 场景可能 dispatch CPU GEMM

apply()
  → dispatch_unquantized_gemm()
```

位置：`linear.py:182` 到 `linear.py:225`

这让 vLLM 的 Linear 抽象保持统一：

```text
无量化：UnquantizedLinearMethod
有量化：Fp8LinearMethod / AutoGPTQLinearMethod / AutoAWQMarlinLinearMethod / ...
```

forward 不需要知道自己是不是量化，只调用：

```text
self.quant_method.apply(...)
```

---

## 13. quant_method.create_weights 创建什么

不同量化方法注册的参数不同。

典型例子：

```text
未量化：
  weight: ModelWeightParameter

GPTQ：
  qweight: PackedvLLMParameter
  g_idx: RowvLLMParameter
  scales: ChannelQuantScaleParameter / GroupQuantScaleParameter
  qzeros: PackedColumnParameter / PackedvLLMParameter

AWQ：
  qweight: PackedvLLMParameter
  qzeros: PackedvLLMParameter
  scales: GroupQuantScaleParameter

FP8：
  weight
  weight_scale 或 weight_scale_inv
  input_scale 可选

bitsandbytes：
  qweight + bnb_quant_state / offsets 等参数属性

MoE 量化：
  w13 / w2 或 expert-packed 权重、scale、zero point、group metadata
```

这些参数通常不是普通 `torch.nn.Parameter`，而是带有 loader 语义的 vLLM 参数对象。

例如：

```text
ModelWeightParameter
PackedvLLMParameter
PackedColumnParameter
GroupQuantScaleParameter
ChannelQuantScaleParameter
PerTensorScaleParameter
BlockQuantScaleParameter
```

它们会记录：

```text
input_dim / output_dim
packed_dim / packed_factor
TP rank / TP size
weight_loader
```

这部分在 `03_weight_loading_and_param_mapping.md` 里展开。

---

## 14. 权重加载和量化的关系

模型加载主入口是：

```python
BaseModelLoader.load_model()
```

位置：`model_loader/base_loader.py:42`

顺序是：

```text
1. initialize_model(...)
2. self.load_weights(model, model_config)
3. 如果是 online quant，finalize_layerwise_processing(...)
4. process_weights_after_loading(model, model_config, target_device)
5. model.eval()
```

位置：`model_loader/base_loader.py:53` 到 `model_loader/base_loader.py:82`

这说明：

```text
量化参数是在模型初始化时创建的；
checkpoint 加载只是填充这些参数；
加载后还要统一调用后处理；
online quantization 的 finalize 发生在普通 post-load 之前。
```

默认 loader 会做：

```text
DefaultModelLoader._get_weights_iterator()
  → 根据 safetensors / pt / npcache / fastsafetensors 等产生 (name, tensor)
DefaultModelLoader.load_weights()
  → model.load_weights(iterator)
```

位置：

```text
default_loader.py:244
default_loader.py:415
default_loader.py:427
```

真正把 checkpoint tensor 映射到参数的是：

```text
model.load_weights()
param.weight_loader(...)
```

而不是 iterator 本身。

---

## 15. process_weights_after_loading 做什么

加载后处理入口：

```python
def process_weights_after_loading(model, model_config, target_device)
```

位置：`model_loader/utils.py:100`

它做两轮遍历。

第一轮处理所有有 `quant_method` 的模块：

```python
quant_method = getattr(module, "quant_method", None)
if isinstance(quant_method, QuantizeMethodBase):
    quant_method.process_weights_after_loading(module)
```

位置：`model_loader/utils.py:103` 到 `model_loader/utils.py:115`

第二轮处理 attention / MLA / MM encoder attention：

```text
Attention / MLAAttention / MMEncoderAttention
  → module.process_weights_after_loading(model_config.dtype)
```

位置：`model_loader/utils.py:117` 到 `model_loader/utils.py:127`

后处理常见工作：

```text
- AWQ 转换 packing order；
- GPTQ / Marlin 做 kernel repack；
- FP8 根据 checkpoint 状态做 transpose / requantize / scale 合并；
- online quant 把浮点权重量化成目标格式；
- TorchAO 设置 reload attrs；
- Attention 处理 KV cache scale。
```

---

## 16. runtime forward 如何调用量化 kernel

Linear forward 不直接判断 GPTQ / AWQ / FP8。

它只做：

```text
bias 准备
  → self.quant_method.apply(self, x, bias)
  → TP gather / all-reduce
```

Column parallel：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
```

位置：`linear.py:552` 到 `linear.py:562`

Row parallel：

```python
output_parallel = self.quant_method.apply(self, input_parallel, bias_)
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

位置：`linear.py:1640` 到 `linear.py:1649`

所以 runtime 的核心边界是：

```text
Linear 负责 TP 输入/输出协议；
quant_method.apply 负责量化 GEMM；
kernel 负责真正 matmul / dequant / scale / fused epilogue。
```

---

## 17. MoE 如何接入量化

MoE 主要在 `RoutedExperts` 中接入量化。

初始化时：

```python
self.quant_method = self._get_quant_method(self.layer_name, self.quant_config, self.moe_config)
```

位置：`routed_experts.py:114` 到 `routed_experts.py:118`

如果量化配置没有给 MoE method，则 fallback 到：

```python
UnquantizedFusedMoEMethod(moe_config)
```

位置：`routed_experts.py:180` 到 `routed_experts.py:196`

然后：

```python
self.quant_method.create_weights(layer=self, **moe_quant_params)
```

位置：`routed_experts.py:150` 到 `routed_experts.py:168`

执行时有两类路径。

模块化 MoE kernel：

```python
return self.quant_method.apply(
    layer=self,
    x=x,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    ...
)
```

位置：`routed_experts.py:1053` 到 `routed_experts.py:1088`

单体 MoE kernel：

```python
return self.quant_method.apply_monolithic(
    layer=self,
    x=x,
    router_logits=router_logits,
    input_ids=input_ids,
)
```

位置：`routed_experts.py:1090` 到 `routed_experts.py:1119`

MoE 量化额外关注：

```text
- expert 权重可能按 EP / TP / DP 切分；
- gate_up_proj 常 fused 成 w13；
- down_proj 常是 w2；
- expert parallel load balancing 可能要求量化 method 支持 EPLB；
- 不同 kernel 可能是 modular 或 monolithic。
```

---

## 18. KV cache 量化是另一条主线

KV cache 量化不等同于权重量化。

用户入口是 `CacheConfig.cache_dtype`，也就是 CLI 中的 `--kv-cache-dtype`。

位置：`config/cache.py:75` 到 `config/cache.py:82`

相关配置：

```text
cache_dtype
  auto / fp8 / fp8_e4m3 / fp8_e5m2 / fp8_inc / per-token-head 等。

calculate_kv_scales
  已废弃；以前控制是否动态计算 k_scale / v_scale。

kv_cache_dtype_skip_layers
  指定某些层跳过 KV cache 量化。
```

位置：`config/cache.py:110` 到 `config/cache.py:117`

校验和提示在：

```text
config/cache.py:268 到 config/cache.py:286
```

如果是 per-token-head scales：

```text
runtime kernel 动态计算 scale。
```

如果是普通 quantized KV cache：

```text
可能使用 checkpoint 中加载的 k_scale / v_scale；
如果没有 scale，则默认 1.0，可能有精度损失。
```

---

## 19. Attention 里的 KV scale 参数

KV cache scale 由 `BaseKVCacheMethod` 创建。

位置：`quantization/kv_cache.py:42`

它会在 Attention layer 上创建：

```text
q_scale
k_scale
v_scale
prob_scale
```

位置：`quantization/kv_cache.py:57` 到 `quantization/kv_cache.py:70`

这些 scale 初始化为 `-1.0`，表示 invalid sentinel。

加载后处理：

```text
1. per-token-head KV cache：忽略 checkpoint scales，runtime 动态算；
2. 非量化 KV cache：强制 k/v scale 为 1.0；
3. 量化 KV cache 且不动态计算：优先使用 checkpoint k_scale / v_scale；
4. 如果只加载到旧的 kv_scale，则复制成 k_scale / v_scale；
5. 如果都没有，则默认 1.0。
```

位置：`quantization/kv_cache.py:74` 到 `quantization/kv_cache.py:120`

这说明：

```text
权重量化走 Linear / MoE quant_method；
KV cache 量化走 Attention scale + attention backend；
两者配置和生命周期相关，但不是同一个参数体系。
```

---

## 20. 平台和 dtype 校验在哪里发生

平台能力有两层校验。

### 20.1 平台支持哪些量化方法

平台接口有：

```python
supported_quantization: list[str] = []
```

位置：`platforms/interface.py:170`

校验函数：

```python
def verify_quantization(cls, quant: str) -> None:
    if cls.supported_quantization and quant not in cls.supported_quantization:
        raise ValueError(...)
```

位置：`platforms/interface.py:824` 到 `platforms/interface.py:831`

`ModelConfig._verify_quantization()` 会调用它。

位置：`config/model.py:1050` 到 `config/model.py:1057`

### 20.2 当前 GPU capability / dtype 是否满足 quant_config

`VllmConfig._get_quantization_config()` 会检查：

```text
current_platform.get_device_capability() >= quant_config.get_min_capability()
model_config.dtype in quant_config.get_supported_act_dtypes()
```

位置：`config/vllm.py:619` 到 `config/vllm.py:636`

所以量化方法能否跑取决于：

```text
量化方法是否在 registry；
平台是否声明支持；
GPU capability 是否足够；
activation dtype 是否支持；
kernel backend 是否可用。
```

---

## 21. load_format 和 quantization 的区别

这两个概念经常混淆。

```text
load_format
  决定 checkpoint 用哪个 ModelLoader 读取。
  例如 hf / safetensors / pt / bitsandbytes / sharded_state / tensorizer。

quantization
  决定 layer 使用哪个 QuantizationConfig / quant_method / kernel。
  例如 gptq / awq / fp8 / compressed-tensors / torchao。
```

典型组合：

```text
load_format=safetensors + quantization=gptq
  → DefaultModelLoader 读 safetensors；
  → AutoGPTQConfig 创建 GPTQ 参数和 kernel。

load_format=bitsandbytes + quantization=bitsandbytes
  → BitsAndBytesModelLoader 读 / 在线量化 BNB 权重；
  → BitsAndBytesConfig 创建对应 layer method。

load_format=sharded_state + quantization=fp8
  → ShardedStateLoader 读本 rank shard；
  → Fp8Config 决定 layer 参数和 kernel。
```

详细权重映射见：`03_weight_loading_and_param_mapping.md`。

---

## 22. 量化如何影响并行

量化权重在并行下不能只按普通 float weight 理解。

Tensor Parallel 下要处理：

```text
ColumnParallelLinear：按 output dim 切；
RowParallelLinear：按 input dim 切；
QKVParallelLinear：q/k/v 各自有 shard_id；
GQA/MQA：K/V head 可能复制；
MergedColumnParallelLinear：gate/up 写入 fused 参数不同段；
Packed weight：offset / size 要按 packed_factor 修正；
Block scale：按 block 数切，不按原始 hidden size 切。
```

Pipeline Parallel 下要处理：

```text
不在当前 PP stage 的参数可能是 missing layer；
load_weights 需要跳过 is_pp_missing_parameter；
非最后 PP rank 不产生 logits，但量化 layer forward 仍在本 stage 执行。
```

Expert Parallel / MoE 下要处理：

```text
每个 rank 只加载本地 expert；
EPLB 可能重排 expert；
量化 method 必须声明是否支持对应 MoE 并行策略。
```

这部分在 `11_parallelism_and_quantization.md` 中展开。

---

## 23. 量化如何影响运行时性能和精度

量化通常希望获得：

```text
更低权重显存；
更低 KV cache 显存；
更高吞吐；
更大 batch / 更长上下文；
更好的带宽利用率。
```

但代价包括：

```text
精度损失；
kernel 支持范围变窄；
某些 GPU capability 不支持；
activation dtype 受限；
scale 缺失时精度下降；
加载后 repack 可能增加启动时间和峰值显存；
某些方法和 LoRA / CUDA graph / speculative decoding / MoE 并行存在限制。
```

从源码结构看，性能和精度主要由这些点决定：

```text
QuantizationConfig 参数：group_size / activation_scheme / block_size / quant_type；
weight_loader 映射：scale / zero point 是否正确切分；
process_weights_after_loading：是否转成 kernel 最佳 layout；
quant_method.apply：最终调用哪个 backend；
CacheConfig.cache_dtype：KV cache 是否量化以及 scale 如何来；
平台能力：是否有对应硬件指令和高性能 kernel。
```

---

## 24. 本目录后续专题阅读顺序

本目录的问题可以按以下顺序阅读：

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

各文件分工：

```text
01_quantization_role.md
  定义量化在 vLLM 中的边界，区分权重量化、激活量化、KV cache 量化和 kernel 支持。

02_quantization_config.md
  梳理用户配置、HF config、QuantizationConfig、online quantization 的解析链路。

03_weight_loading_and_param_mapping.md
  梳理量化 checkpoint 中 weight / scale / zero point 如何映射到 vLLM 参数。

04_quantized_linear_layers.md
  梳理 LinearBase 如何通过 quant_method 创建权重并调用量化 kernel。

05_weight_only_quantization.md
  梳理 GPTQ / AWQ / Marlin / INT4 / INT8 / FP8 等 weight-only 路径。

06_activation_and_dynamic_quantization.md
  梳理 activation quant、dynamic quant、per-token scaling 等运行时量化。

07_kv_cache_quantization.md
  梳理 KV cache dtype、scale、layout 和 attention backend 的关系。

08_attention_backend_interaction.md
  梳理量化如何影响 FlashAttention / FlashInfer / Triton / MLA backend 选择。

09_moe_quantization.md
  梳理 fused MoE、expert 权重、routing 和量化 kernel 的关系。

10_lora_and_quantization.md
  梳理量化 base model 与 LoRA adapter 的共存方式和限制。

11_parallelism_and_quantization.md
  梳理 TP / PP / EP 下量化权重、scale、group size 的切分。

12_accuracy_performance_tradeoffs.md
  梳理显存、吞吐、延迟、精度和数值稳定性的取舍。

13_limitations_and_debugging.md
  梳理常见不支持场景、错误信息、fallback 和调试入口。
```

---

## 25. 常见误区

### 25.1 quantization 是不是只影响加载？

不是。

它同时影响：

```text
配置校验；
参数创建；
权重加载；
后处理；
forward kernel；
KV cache scale；
MoE expert kernel；
平台限制。
```

### 25.2 checkpoint 里有 quantization_config 就一定按它跑吗？

不一定。

vLLM 会通过 `override_quantization_method()` 重新判断 checkpoint 最适合的运行时方法，并检查用户传入的 `--quantization` 是否冲突。

### 25.3 FP8 权重量化和 FP8 KV cache 是一回事吗？

不是。

FP8 权重量化由 `Fp8Config` / `Fp8LinearMethod` 等控制；FP8 KV cache 由 `CacheConfig.cache_dtype`、Attention scale 和 attention backend 控制。

### 25.4 量化参数都是普通 weight 吗？

不是。

很多量化参数是：

```text
qweight
qzeros
scales
g_idx
weight_scale
weight_scale_inv
input_scale
k_scale / v_scale
bnb_quant_state
```

它们有不同的切分和后处理逻辑。

### 25.5 quant_method.apply 一定只是 dequant + matmul 吗？

不是。

不同 kernel 可能融合：

```text
activation quantization；
scale 应用；
zero point；
expert routing；
SwiGLU；
transpose / packed layout；
all-reduce 前后的 epilogue。
```

### 25.6 online quantization 和预量化 checkpoint 有什么区别？

预量化 checkpoint：

```text
checkpoint 已有 qweight / scales / qzeros；
loader 主要负责映射和 repack。
```

online quantization：

```text
checkpoint 通常是 fp16 / bf16；
加载后在 vLLM 内部把权重量化成目标格式。
```

---

## 26. 总结

vLLM 的量化可以压缩成一条完整链路：

```text
配置入口
  → 量化方法识别
  → QuantizationConfig 构造
  → layer 获取 quant_method
  → quant_method.create_weights()
  → checkpoint weight_loader 填参
  → quant_method.process_weights_after_loading()
  → quant_method.apply()
  → kernel 执行
```

如果只记住一句话：

```text
量化在 vLLM 中不是一个开关，而是一套贯穿配置、加载、参数布局、后处理和 kernel dispatch 的协议。
```

再压缩成最小心智模型：

```text
QuantizationConfig 负责选择策略；
QuantizeMethod 负责 layer 生命周期；
Parameter / weight_loader 负责权重映射；
process_weights_after_loading 负责 kernel layout；
apply 负责 runtime 计算；
CacheConfig 负责 KV cache 量化这条独立但相关的路径。
```
