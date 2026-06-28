# 05. Weight-only 量化如何工作？

源码位置：

- `vllm/vllm/config/vllm.py`
- `vllm/vllm/model_executor/model_loader/weight_utils.py`
- `vllm/vllm/model_executor/model_loader/utils.py`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/parameter.py`
- `vllm/vllm/model_executor/kernels/linear/`
- `vllm/vllm/model_executor/layers/fused_moe/`

本问题关注：GPTQ、AWQ、Marlin、INT4、INT8、FP8、bitsandbytes、compressed-tensors、MoE WNA16 等权重量化路径如何接入 vLLM；量化配置如何被解析；量化后的参数如何创建、加载、TP 切分和 repack；forward 时如何从 `LinearBase.forward()` 或 `RoutedExperts.forward_*()` 进入对应 kernel。

---

## 1. 一句话回答

Weight-only 量化在 vLLM 里不是一个单独的 forward 分支，而是一套 **QuantizationConfig → QuantizeMethod → 参数创建 / 权重加载 / post-load repack / kernel apply** 的协议。

主链路是：

```text
ModelConfig.quantization / checkpoint quantization_config
  → VllmConfig._get_quantization_config()
  → weight_utils.get_quant_config()
  → get_quantization_config(name).from_config(...)
  → initialize_model(...)
  → configure_quant_config(...)
  → LinearBase / RoutedExperts 初始化时调用 quant_config.get_quant_method(...)
  → quant_method.create_weights(...)
  → checkpoint weight_loader 填充 qweight / scales / qzeros / g_idx / weight_scale
  → process_weights_after_loading(...)
  → quant_method.process_weights_after_loading(...)
  → Linear.forward() / RoutedExperts.forward_*()
  → quant_method.apply(...)
  → Marlin / Cutlass / Triton / torch / bitsandbytes / fused MoE kernel
```

所以可以把它理解成：

```text
配置决定用哪种量化方法；
量化方法决定创建哪些参数；
参数类和 weight_loader 负责 TP 切分与 packed layout；
post-load 阶段把 checkpoint 格式转换成 kernel 需要的格式；
forward 阶段只调用 quant_method.apply()。
```

---

## 2. 最小心智模型

### 2.1 Linear 层

Linear 层里最关键的对象是：

```text
LinearBase.quant_method
```

初始化时：

```text
quant_config is None
  → UnquantizedLinearMethod

quant_config exists
  → quant_config.get_quant_method(layer, prefix)
  → AutoGPTQLinearMethod / AutoAWQLinearMethod / Fp8LinearMethod / ...
```

位置：`vllm/vllm/model_executor/layers/linear.py:269`

forward 时：

```python
output = self.quant_method.apply(self, input_, bias)
```

位置：`vllm/vllm/model_executor/layers/linear.py:555`

也就是说，Linear 层本身不关心 GPTQ、AWQ、FP8 的细节；它只持有一个 `quant_method`。

### 2.2 MoE 层

MoE 的对应对象是：

```text
RoutedExperts.quant_method
```

初始化时：

```text
RoutedExperts
  → quant_config.get_quant_method(self, prefix)
  → AutoGPTQMoEMethod / AutoAWQMoEMethod / Fp8MoEMethod / MoeWNA16Method / ...
  → quant_method.create_weights(...)
```

位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:114`

执行时：

```text
forward_modular(...)
  → quant_method.apply(...)

forward_monolithic(...)
  → quant_method.apply_monolithic(...)
```

位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:1053` 和 `vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:1090`

---

## 3. 配置从哪里来

量化配置入口在 `VllmConfig`：

```python
def _get_quantization_config(model_config, load_config)
```

位置：`vllm/vllm/config/vllm.py:609`

它会：

```text
1. 如果 model_config.quantization 不为空，调用 get_quant_config();
2. 检查当前设备 capability 是否满足 quant_config.get_min_capability();
3. 检查 model_config.dtype 是否在 quant_config.get_supported_act_dtypes();
4. 调用 quant_config.maybe_update_config(...);
5. 返回 quant_config。
```

对应源码：`vllm/vllm/config/vllm.py:615` 到 `vllm/vllm/config/vllm.py:641`

这说明量化在模型真正创建前就已经被解析，并且会提前做硬件和 activation dtype 校验。

---

## 4. get_quant_config() 如何解析 checkpoint 配置

入口：

```python
def get_quant_config(model_config, load_config) -> QuantizationConfig
```

位置：`vllm/vllm/model_executor/model_loader/weight_utils.py:240`

核心流程：

```text
model_config.quantization
  → get_quantization_config(model_config.quantization)
  → 得到 QuantizationConfig 子类
  → 从 HF config 读取 quantization_config / compression_config
  → quant_cls.from_config(hf_quant_config)
```

位置：`vllm/vllm/model_executor/model_loader/weight_utils.py:245` 到 `vllm/vllm/model_executor/model_loader/weight_utils.py:291`

### 4.1 常见配置来源

```text
1. config.json 里的 quantization_config；
2. text_config.quantization_config；
3. compression_config，也就是 compressed-tensors；
4. hf_overrides 里的 quantization_config_file；
5. hf_overrides 里的 quantization_config_dict_json；
6. CLI / API 传入的 online quantization_config；
7. bitsandbytes 的 inflight 量化空配置。
```

### 4.2 compressed-tensors 的特殊处理

如果是 compressed-tensors，并且有 `config_groups`，vLLM 会把：

```text
total_num_heads
total_num_kv_heads
```

写入 quant config，用于 KV cache scale 的 TP-aware 加载。

位置：`vllm/vllm/model_executor/model_loader/weight_utils.py:257` 到 `vllm/vllm/model_executor/model_loader/weight_utils.py:273`

---

## 5. quantization 方法注册表

量化方法名在：`vllm/vllm/model_executor/layers/quantization/__init__.py:12`

典型名字包括：

```text
awq / auto_awq / awq_marlin
gptq / auto_gptq / gptq_marlin
fp8 / fbgemm_fp8 / fp_quant
compressed-tensors
bitsandbytes
experts_int8
quark
moe_wna16
torchao
inc
mxfp4 / gpt_oss_mxfp4
online
fp8_per_tensor / fp8_per_block / fp8_per_channel
int8_per_channel_weight_only
mxfp8
```

映射入口：

```python
def get_quantization_config(quantization: str) -> type[QuantizationConfig]
```

位置：`vllm/vllm/model_executor/layers/quantization/__init__.py:108`

它把方法名映射到具体类，例如：

```text
"gptq" / "auto_gptq" / "gptq_marlin"
  → AutoGPTQConfig

"awq" / "auto_awq" / "awq_marlin"
  → AutoAWQConfig

"fp8"
  → Fp8Config

"compressed-tensors"
  → CompressedTensorsConfig

"bitsandbytes"
  → BitsAndBytesConfig

"moe_wna16"
  → MoeWNA16Config
```

位置：`vllm/vllm/model_executor/layers/quantization/__init__.py:140` 到 `vllm/vllm/model_executor/layers/quantization/__init__.py:170`

---

## 6. QuantizationConfig / QuantizeMethod 的抽象边界

### 6.1 QuantizationConfig 负责什么

抽象定义：`vllm/vllm/model_executor/layers/quantization/base_config.py:77`

核心接口：

```text
get_name()
get_supported_act_dtypes()
get_min_capability()
get_config_filenames()
from_config(config)
get_quant_method(layer, prefix)
maybe_update_config(...)
apply_vllm_mapper(...)
```

其中最重要的是：

```python
def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None
```

位置：`vllm/vllm/model_executor/layers/quantization/base_config.py:157`

它决定某个具体 layer 应该使用哪种执行方法。

### 6.2 QuantizeMethod 负责什么

抽象定义：`vllm/vllm/model_executor/layers/quantization/base_config.py:19`

核心接口：

```text
create_weights(layer, ...)
apply(layer, ...)
process_weights_after_loading(layer)
```

含义是：

```text
create_weights：
  在 layer 上注册 qweight / scales / qzeros / g_idx / input_scale 等参数。

process_weights_after_loading：
  checkpoint 权重加载完后，把权重转成 kernel 需要的布局。

apply：
  forward 热路径，调用具体 kernel。
```

---

## 7. 模型初始化时如何把 quant_config 传给层

模型初始化入口：

```python
initialize_model(vllm_config)
```

位置：`vllm/vllm/model_executor/model_loader/utils.py:40`

在创建模型前：

```python
if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)
```

位置：`vllm/vllm/model_executor/model_loader/utils.py:54`

`configure_quant_config()` 会把模型类里的：

```text
hf_to_vllm_mapper
packed_modules_mapping
```

传给 quant config。

位置：`vllm/vllm/model_executor/model_loader/utils.py:274` 到 `vllm/vllm/model_executor/model_loader/utils.py:295`

这一步很重要，因为很多 checkpoint 的量化配置是按 HF 模块名写的，而 vLLM 模型会把 QKV、gate/up 等模块 fuse 起来。

---

## 8. LinearBase 如何接入量化

`LinearBase` 是所有 linear 层的基类。

位置：`vllm/vllm/model_executor/layers/linear.py:228`

初始化时：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`vllm/vllm/model_executor/layers/linear.py:269` 到 `vllm/vllm/model_executor/layers/linear.py:274`

然后不同 Linear 子类调用：

```python
self.quant_method.create_weights(...)
```

例如：

```text
ReplicatedLinear：linear.py:338
ColumnParallelLinear：linear.py:461
RowParallelLinear：linear.py:1565
```

forward 时统一走：

```text
ColumnParallelLinear.forward()
  → quant_method.apply(self, input_, bias)
  → 如果 gather_output，做 tensor_model_parallel_all_gather

RowParallelLinear.forward()
  → 必要时切 input
  → quant_method.apply(self, input_parallel, bias)
  → 如果 reduce_results，做 tensor_model_parallel_all_reduce
```

对应位置：`vllm/vllm/model_executor/layers/linear.py:548` 和 `vllm/vllm/model_executor/layers/linear.py:1628`

这意味着 TP 的通信仍由 Linear 子类控制，而低 bit GEMM 由量化 method 控制。

---

## 9. TP 切分和 packed layout 为什么复杂

量化权重经常不是普通 `[out, in]` 浮点矩阵，而是：

```text
qweight：低 bit 打包后的整数张量；
scales：per-tensor / per-channel / per-group / per-block scale；
qzeros：zero point，也可能被 packed；
g_idx：GPTQ act-order / desc_act 场景下的 group index；
input_scale：activation static quant 场景下的输入 scale。
```

vLLM 用一组 Parameter 子类处理这些布局：

```text
ModelWeightParameter
PackedvLLMParameter
PackedColumnParameter
GroupQuantScaleParameter
ChannelQuantScaleParameter
PerTensorScaleParameter
BlockQuantScaleParameter
RowvLLMParameter
```

位置：`vllm/vllm/model_executor/parameter.py:18`

关键能力：

```text
1. 根据 input_dim / output_dim 做 TP narrow；
2. 根据 packed_factor 调整 shard_size / shard_offset；
3. 对 QKV / MergedColumn 的 shard_id 做局部加载；
4. 对 per-tensor scale 处理 fused logical matrix；
5. 对 block scale 处理 block 维度。
```

### 9.1 packed_factor

比如 INT4 packed 到 int32：

```text
pack_factor = 32 // 4 = 8
```

一个 int32 里放 8 个 4-bit 权重。

当 TP 切输出或输入时，不能直接按原始浮点矩阵的维度切，必须考虑 packed 后维度缩小。

对应逻辑：`vllm/vllm/model_executor/parameter.py:353` 到 `vllm/vllm/model_executor/parameter.py:394`

### 9.2 Linear 层加载时的特殊调整

`linear.py` 里也有一些量化专用 shard 调整：

```text
adjust_marlin_shard()
adjust_block_scale_shard()
adjust_bitsandbytes_4bit_shard()
adjust_scalar_to_fused_array()
```

位置：`vllm/vllm/model_executor/layers/linear.py:70` 到 `vllm/vllm/model_executor/layers/linear.py:135`

这些函数解决的是：

```text
checkpoint 的逻辑权重维度
  和
vLLM 当前 rank 上实际注册的 packed 参数维度
```

不一致的问题。

---

## 10. 加载后处理 process_weights_after_loading

checkpoint 权重加载完后，vLLM 会遍历所有模块：

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(module)
```

位置：`vllm/vllm/model_executor/model_loader/utils.py:100` 到 `vllm/vllm/model_executor/model_loader/utils.py:113`

这一步通常做：

```text
1. AWQ packed order 转换；
2. GPTQ / AWQ / FP8 权重 repack；
3. Marlin / Cutlass / Triton 需要的 layout 转换；
4. FP8 在线量化；
5. MoE 权重转换成 fused MoE kernel 格式；
6. 替换 layer 上的参数对象；
7. 初始化 kernel workspace / moe_quant_config。
```

所以 weight-only 量化的关键不只在 forward，而是在加载完成后的格式准备。

---

## 11. GPTQ 路径

源码：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py`

### 11.1 配置

配置类：

```python
class AutoGPTQConfig(QuantizationConfig)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:97`

从 config 读取：

```text
bits
group_size
desc_act
sym
lm_head
dynamic
modules_in_block_to_quantize
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:195` 到 `vllm/vllm/model_executor/layers/quantization/auto_gptq.py:216`

支持的核心组合：

```text
INT4 symmetric → scalar_types.uint4b8
INT8 symmetric → scalar_types.uint8b128
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:100`

### 11.2 Linear method

Linear 方法：

```python
class AutoGPTQLinearMethod(LinearMethodBase)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:304`

创建的参数：

```text
qweight：PackedvLLMParameter，通常 int32 packed；
g_idx：RowvLLMParameter，用于 desc_act / act-order；
scales：ChannelQuantScaleParameter 或 GroupQuantScaleParameter；
qzeros：PackedColumnParameter 或 PackedvLLMParameter。
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:378` 到 `vllm/vllm/model_executor/layers/quantization/auto_gptq.py:443`

### 11.3 kernel 选择

GPTQ Linear 会构造：

```python
MPLinearLayerConfig(...)
```

然后调用：

```python
choose_mp_linear_kernel(mp_linear_kernel_config)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:339` 到 `vllm/vllm/model_executor/layers/quantization/auto_gptq.py:352`

这会在 Marlin、Machete、Cutlass、Triton、Exllama、CPU 等 mixed precision kernel 中选择可用实现。

kernel 列表在：`vllm/vllm/model_executor/kernels/linear/__init__.py:353`

### 11.4 post-load 和 forward

加载后：

```python
self.kernel.process_weights_after_loading(layer)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:453`

forward：

```python
return self.kernel.apply_weights(layer, x, bias)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:456` 到 `vllm/vllm/model_executor/layers/quantization/auto_gptq.py:462`

也就是说 GPTQ 的热路径不是 Python 解包权重，而是把已经准备好的 packed 权重交给选中的 kernel。

### 11.5 desc_act / g_idx

GPTQ 的 `desc_act` 会影响：

```text
has_g_idx=True / False
scales 是否在 TP rank 上重复
w2 scales 是否完整加载
MoE kernel 是否需要 g_idx_sort_indices
```

这也是 GPTQ 比普通 per-group INT4 更复杂的原因。

---

## 12. AWQ 路径

源码：`vllm/vllm/model_executor/layers/quantization/auto_awq.py`

### 12.1 配置

配置类：

```python
class AutoAWQConfig(QuantizationConfig)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:170`

从 config 读取：

```text
w_bit / bits
q_group_size / group_size
zero_point
lm_head
modules_to_not_convert
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:237` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:256`

当前 AutoAWQ 主路径支持：

```text
4-bit AWQ weight-only
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:177`

### 12.2 方法选择

`get_quant_method()` 会根据平台和 kernel 支持选择：

```text
XPU
  → AutoAWQXPULinearMethod

CPU
  → AutoAWQMarlinLinearMethod，实际可选 CPUWNA16LinearKernel

CUDA + Marlin 支持 + 非 batch invariant
  → AutoAWQMarlinLinearMethod

否则
  → AutoAWQLinearMethod，走 AWQ Triton/custom op 路径
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:284` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:332`

### 12.3 普通 AWQ Linear

普通 AWQ 创建：

```text
qweight：PackedvLLMParameter
qzeros：PackedvLLMParameter
scales：GroupQuantScaleParameter
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:821` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:900`

forward 时：

```text
如果 token 数较大或 batch invariant：
  awq_dequantize(qweight, scales, qzeros)
  torch.matmul(...)

否则：
  ops.awq_gemm(...)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:921` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:939`

### 12.4 AWQ Marlin

AWQ checkpoint 的 packed order 和标准 GPTQ-like 格式不同。

vLLM 在 post-load 阶段调用：

```python
_convert_awq_to_standard_format(layer, "qweight", "qzeros", size_bits)
self.kernel.process_weights_after_loading(layer)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:525` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:533`

转换原因：

```text
AWQ checkpoint：
  qweight 沿 output dim packed，并且 int32 内部 bit 顺序是 AWQ 特有顺序。

Marlin / MPLinearKernel：
  期望更标准的 GPTQ-like packed 格式，通常沿 input dim packed。
```

转换函数位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:92`

---

## 13. Marlin 在这里扮演什么角色

Marlin 不是一个独立的“配置入口”，而是一类 kernel / layout 后端。

它常出现在：

```text
GPTQ Marlin
AWQ Marlin
FP8 Marlin
MoE WNA16 Marlin
MXFP4 / MXFP8 / NVFP4 Marlin
```

在 Linear 路径里，Marlin 通常通过：

```text
choose_mp_linear_kernel(...)
init_fp8_linear_kernel(...)
init_mxfp4_linear_kernel(...)
init_mxfp8_linear_kernel(...)
init_nvfp4_linear_kernel(...)
```

被选中。

位置：`vllm/vllm/model_executor/kernels/linear/__init__.py:531`、`vllm/vllm/model_executor/kernels/linear/__init__.py:640`

### 13.1 Marlin 的关键要求

Marlin 关心：

```text
1. quant_type，例如 uint4 / uint4b8 / uint8b128 / fp8；
2. group_size；
3. 是否有 zero point；
4. activation dtype；
5. 矩阵 shape 是否能被 tile 支持；
6. packed_dim / packed_factor；
7. post-load 后的 weight layout。
```

因此 vLLM 经常在 `process_weights_after_loading()` 中把 checkpoint 权重转为 Marlin 需要的布局。

---

## 14. FP8 路径

源码：`vllm/vllm/model_executor/layers/quantization/fp8.py`

### 14.1 配置

配置类：

```python
class Fp8Config(QuantizationConfig)
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:99`

关键字段：

```text
is_checkpoint_fp8_serialized：checkpoint 是否已经是 FP8；
activation_scheme：static / dynamic；
ignored_layers：跳过量化的层；
weight_block_size：是否 block-wise FP8；
store_dtype：部分 MoE / MXFP4 场景使用。
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:102` 到 `vllm/vllm/model_executor/layers/quantization/fp8.py:137`

### 14.2 FP8 不一定是纯 weight-only

FP8 路径有两类：

```text
W8A16 / weight-only FP8：
  权重 FP8，activation 仍是 bf16/fp16，或者由 kernel 内部处理。

W8A8：
  权重和 activation 都参与 FP8 scaled mm，activation scale 可以 static 或 dynamic。
```

所以本问题虽然聚焦 weight-only，但 FP8 在 vLLM 中实际覆盖了 weight-only 和 weight+activation 两种语义。

### 14.3 offline FP8 和 online FP8

`get_quant_method()` 里：

```text
如果 checkpoint 不是 fp8 serialized：
  → Fp8PerTensorOnlineLinearMethod
  → 加载 fp16/bf16 后在线量化

如果 checkpoint 已经是 fp8 serialized：
  → Fp8LinearMethod
  → 直接加载 FP8 weight / scale
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:179` 到 `vllm/vllm/model_executor/layers/quantization/fp8.py:200`

### 14.4 FP8 Linear 参数

`Fp8LinearMethod.create_weights()` 会注册：

```text
weight：FP8 weight parameter；
weight_scale：per-tensor scale；
weight_scale_inv：block-wise scale 场景；
input_scale：static activation scale 场景。
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:322` 到 `vllm/vllm/model_executor/layers/quantization/fp8.py:386`

然后调用：

```python
init_fp8_linear_kernel(...)
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:387`

kernel 选择入口：`vllm/vllm/model_executor/kernels/linear/__init__.py:531`

可能选择：

```text
MarlinFP8ScaledMMLinearKernel
FlashInferFP8ScaledMMLinearKernel
CutlassFP8ScaledMMLinearKernel
TritonFp8BlockScaledMMKernel
DeepGemmFp8BlockScaledMMKernel
Torch FP8 fallback
ROCm / AITER / XPU 相关 kernel
```

### 14.5 FP8 post-load

加载后处理会：

```text
1. 如果走 Marlin，可能转置 weight 并调用 Marlin post-load；
2. 如果 checkpoint 不是 serialized FP8，把 fp16/bf16 weight 量化成 FP8；
3. 如果是 per-tensor fused module，合并 / max scale；
4. 如果 static activation scale，整理 input_scale；
5. 最后调用 fp8_linear.process_weights_after_loading(layer)。
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:398` 到 `vllm/vllm/model_executor/layers/quantization/fp8.py:444`

forward：

```python
return self.fp8_linear.apply_weights(layer, x, bias)
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:446` 到 `vllm/vllm/model_executor/layers/quantization/fp8.py:489`

---

## 15. bitsandbytes 路径

源码：`vllm/vllm/model_executor/layers/quantization/bitsandbytes.py`

### 15.1 配置

配置类：

```python
class BitsAndBytesConfig(QuantizationConfig)
```

位置：`vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:49`

关键字段：

```text
load_in_8bit
load_in_4bit
bnb_4bit_compute_dtype
bnb_4bit_quant_storage
bnb_4bit_quant_type
bnb_4bit_use_double_quant
llm_int8_skip_modules
llm_int8_threshold
```

位置：`vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:55` 到 `vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:79`

### 15.2 Linear 参数

8bit 路径创建：

```text
Int8Params，dtype=torch.int8
```

4bit 路径创建：

```text
BitsAndBytesWeightParameter，dtype=torch.uint8
```

位置：`vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:225` 到 `vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:269`

### 15.3 forward

```text
8bit：
  bitsandbytes.matmul(...)

4bit：
  bitsandbytes.matmul_4bit(...)
```

位置：`vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:291` 和 `vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:361`

BNB 的特点是：

```text
1. 更依赖 bitsandbytes 自身的量化 state；
2. 4bit 权重是 flat packed tensor；
3. TP 切分有专门的 adjust_bitsandbytes_4bit_shard；
4. MoE 4bit 目前会在 apply 前 dequant 成普通 w13 / w2，再调用 fused_experts。
```

MoE 4bit dequant 位置：`vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:594`

---

## 16. compressed-tensors 路径

源码：`vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`

### 16.1 配置模型

配置类：

```python
class CompressedTensorsConfig(QuantizationConfig)
```

位置：`vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:80`

compressed-tensors 不是单一格式，而是从 `config_groups` 中解析 target 和 scheme：

```text
config_groups
  → targets
  → weights QuantizationArgs
  → input_activations QuantizationArgs
  → output_activations QuantizationArgs
  → format
  → target_scheme_map
```

位置：`vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:297` 到 `vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:367`

### 16.2 scheme 选择

`CompressedTensorsConfig._get_scheme_from_parts()` 会根据权重和 activation 配置选择：

```text
CompressedTensorsWNA16：INT4/INT8 weight-only，activation A16；
CompressedTensorsW8A16Fp8：FP8 weight-only / A16；
CompressedTensorsW8A8Fp8：FP8 weight + FP8 activation；
CompressedTensorsW8A8Int8：INT8 weight + INT8 activation；
CompressedTensorsW4A8Int：INT4 weight + INT8 activation；
CompressedTensorsW4A8Fp8：INT4/FP8 mixed；
CompressedTensorsW4A4Mxfp4 / W4A4Fp4；
CompressedTensorsW8A8Mxfp8；
CompressedTensorsWNA8O8Int。
```

位置：`vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:692` 到 `vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:813`

### 16.3 Linear 执行

Linear method 很薄：

```text
create_weights(...)
  → layer.scheme.create_weights(...)

process_weights_after_loading(...)
  → layer.scheme.process_weights_after_loading(layer)

apply(...)
  → layer.scheme.apply_weights(layer, x, bias)
```

位置：`vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:910` 到 `vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:958`

所以 compressed-tensors 的复杂性在 scheme 子类，而不是顶层 Linear method。

---

## 17. MoE weight-only 量化路径

MoE 不走普通 `LinearBase`，而是走：

```text
FusedMoE(...)
  → RoutedExperts(...)
  → quant_config.get_quant_method(RoutedExperts, prefix)
  → FusedMoEMethodBase 子类
```

入口：`vllm/vllm/model_executor/layers/fused_moe/layer.py:103`

`RoutedExperts` 创建后会调用：

```python
self.quant_method.create_weights(layer=self, **moe_quant_params)
```

位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:168`

### 17.1 MoE 的典型权重名

MoE expert 权重通常按 fused 方式组织：

```text
w13：gate_proj + up_proj 的 fused 权重；
w2：down_proj 权重。
```

对应 checkpoint shard：

```text
w1：gate_proj
w3：up_proj
w2：down_proj
```

`RoutedExperts.weight_loader()` 会根据 shard_id 和 expert_id 把 checkpoint 权重加载到本 rank 的本地 expert 参数中。

位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:571`

### 17.2 AutoGPTQMoE

GPTQ MoE 方法：

```python
class AutoGPTQMoEMethod(FusedMoEMethodBase)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:465`

它创建：

```text
w13_qweight / w2_qweight
w13_scales / w2_scales
w13_qzeros / w2_qzeros
w13_g_idx / w2_g_idx
w13_g_idx_sort_indices / w2_g_idx_sort_indices
workspace / moe_kernel
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:492` 到 `vllm/vllm/model_executor/layers/quantization/auto_gptq.py:650`

post-load 时会调用：

```python
convert_to_wna16_moe_kernel_format(...)
make_wna16_moe_kernel(...)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:651` 到 `vllm/vllm/model_executor/layers/quantization/auto_gptq.py:750`

### 17.3 AutoAWQMoE

AWQ MoE 方法：

```python
class AutoAWQMoEMethod(FusedMoEMethodBase)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:544`

它只支持 4bit，并创建：

```text
w13_qweight / w2_qweight
w13_scales / w2_scales
w13_qzeros / w2_qzeros
workspace
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:562` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:661`

post-load 同样转换成 WNA16 MoE kernel 格式。

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:663` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:741`

### 17.4 MoeWNA16 fallback

`MoeWNA16Config` 是 MoE W4A16 / W8A16 的通用路径。

位置：`vllm/vllm/model_executor/layers/quantization/moe_wna16.py:34`

它支持：

```text
gptq：4bit / 8bit，要求 desc_act 为 False；
awq：4bit；
```

位置：`vllm/vllm/model_executor/layers/quantization/moe_wna16.py:134` 到 `vllm/vllm/model_executor/layers/quantization/moe_wna16.py:156`

当 GPTQ/AWQ MoE 不满足 Marlin kernel 的 shape 要求时，会 fallback 到 MoeWNA16。

GPTQ fallback 位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:243`

AWQ fallback 位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:342`

---

## 18. scale / zero point / group size 怎么理解

### 18.1 scale 粒度

vLLM 常见 scale 粒度：

```text
per-tensor：
  一个逻辑矩阵一个 scale，常见于部分 FP8 / fused QKV scale。

per-channel：
  通常沿输出 channel 有 scale。

per-group：
  每 group_size 个输入通道共享 scale，GPTQ / AWQ / WNA16 常见。

per-block：
  二维 block scale，例如 FP8 block-wise 128x128。
```

对应 Parameter 类型：

```text
PerTensorScaleParameter
ChannelQuantScaleParameter
GroupQuantScaleParameter
BlockQuantScaleParameter
```

位置：`vllm/vllm/model_executor/parameter.py:242`、`vllm/vllm/model_executor/parameter.py:251`、`vllm/vllm/model_executor/parameter.py:260`、`vllm/vllm/model_executor/parameter.py:397`

### 18.2 zero point

zero point 通常出现在 asymmetric INT 量化里，例如：

```text
AWQ zero_point=True
GPTQ sym=False
MoE WNA16 has_zp=True
```

但不是所有 kernel 都需要或支持 zero point。

例如 GPTQ 的 `TYPE_MAP` 只接受 symmetric：

```text
(4, True)
(8, True)
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_gptq.py:100`

而 AWQ 配置显式保存：

```text
zero_point
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:192`

### 18.3 group_size 与 TP

很多 weight-only 量化要求：

```text
input_size_per_partition % group_size == 0
```

如果 TP 太大，单 rank 上的 input partition 可能不能整除 group_size，加载或 kernel 会报错。

AWQ 明确检查：

```text
input_size_per_partition % group_size == 0
output_size_per_partition % pack_factor == 0
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:843` 到 `vllm/vllm/model_executor/layers/quantization/auto_awq.py:856`

MoE WNA16 会在必要时缩小 group_size，并记录 `group_size_div_factor`，后续 repeat scale / qzeros。

位置：`vllm/vllm/model_executor/layers/quantization/moe_wna16.py:216` 到 `vllm/vllm/model_executor/layers/quantization/moe_wna16.py:224`

---

## 19. weight-only 与在线量化的区别

### 19.1 离线量化 checkpoint

GPTQ、AWQ、compressed-tensors、部分 FP8 通常是离线量化：

```text
checkpoint 已经包含 qweight / scales / qzeros / g_idx 等；
vLLM 只负责加载、切分、repack、调用 kernel。
```

### 19.2 加载时在线量化

部分路径会在加载后把 fp16/bf16 权重量化成低 bit：

```text
fp8 非 serialized checkpoint
  → Fp8PerTensorOnlineLinearMethod

Fp8OnlineMoEMethod
  → uses_meta_device=True
  → 加载后 ops.scaled_fp8_quant(...)
```

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:189` 和 `vllm/vllm/model_executor/layers/quantization/fp8.py:857`

这类路径的特点是：

```text
checkpoint 本身不是低 bit；
加载时或加载后转换；
峰值内存和 post-load 时间会受影响；
但用户不需要提前准备量化 checkpoint。
```

---

## 20. kernel 选择关系

### 20.1 mixed precision / WNA16 kernel

GPTQ、AWQ Marlin 等 WNA16 路径通常走：

```python
choose_mp_linear_kernel(MPLinearLayerConfig)
```

位置：`vllm/vllm/model_executor/kernels/linear/__init__.py:640`

候选 kernel 包括：

```text
CUDA：CutlassW4A8、Machete、AllSpark、Marlin、Humming、Conch、Exllama、TritonW4A16
ROCm：RDNA3W4A16、TritonW4A16、Conch、Exllama
XPU：XPUW4A8Int、XPUwNa16
CPU：Dynamic4bit、ZentorchWNA16、CPUWNA16
```

位置：`vllm/vllm/model_executor/kernels/linear/__init__.py:353` 到 `vllm/vllm/model_executor/kernels/linear/__init__.py:379`

### 20.2 FP8 scaled-mm kernel

FP8 路径通常走：

```python
init_fp8_linear_kernel(...)
```

位置：`vllm/vllm/model_executor/kernels/linear/__init__.py:531`

候选 kernel 包括：

```text
MarlinFP8ScaledMMLinearKernel
FlashInferFP8ScaledMMLinearKernel
CutlassFP8ScaledMMLinearKernel
Torch FP8 fallback
AITER / ROCm / XPU / Triton / DeepGemm block kernel
```

### 20.3 backend 过滤

如果用户设置：

```text
--linear-backend cutlass / marlin / triton / torch / aiter / machete / ...
```

kernel 选择会被 `_filter_kernels_by_backend()` 限制。

位置：`vllm/vllm/model_executor/kernels/linear/__init__.py:183` 到 `vllm/vllm/model_executor/kernels/linear/__init__.py:272`

---

## 21. CUDA graph / torch.compile 关系

量化 Linear / MoE 的 forward 最终仍然是 PyTorch module forward 的一部分，因此会被 ModelRunner 的 forward context、CUDA graph、compile 路径包裹。

它本身需要满足：

```text
1. apply() 热路径 shape 稳定；
2. kernel 支持当前 dtype / device / shape；
3. 参数 layout 在 capture 前已经 post-load 完成；
4. 某些 batch-invariant 模式会避开非确定性 kernel。
```

例如 AWQ Marlin 在 batch invariant 模式下不会被选择：

```text
not envs.VLLM_BATCH_INVARIANT
```

位置：`vllm/vllm/model_executor/layers/quantization/auto_awq.py:309`

FP8 在 batch invariant 模式下也有直接 FP8 / BF16 fallback 的特殊分支。

位置：`vllm/vllm/model_executor/layers/quantization/fp8.py:452` 到 `vllm/vllm/model_executor/layers/quantization/fp8.py:489`

---

## 22. 常见格式对照

```text
GPTQ：
  qweight / scales / qzeros / g_idx
  INT4 或 INT8，常见 per-group scale，desc_act 影响 g_idx。

AWQ：
  qweight / scales / qzeros
  INT4，checkpoint packed order 特殊，Marlin 路径需要转换。

Marlin：
  kernel/layout 后端，不是单独 checkpoint 标准。
  要求 weight layout、group size、zero point、tile shape 与 kernel 匹配。

FP8：
  weight / weight_scale / input_scale / weight_scale_inv
  支持 serialized FP8，也支持加载时在线量化。

bitsandbytes：
  使用 BNB 自己的 quant state 和 matmul/matmul_4bit。
  4bit 权重常是 flat packed tensor。

compressed-tensors：
  根据 config_groups 动态选择 scheme。
  可覆盖 WNA16、W8A8、W4A8、FP8、MXFP4/MXFP8、KV cache scale 等。

MoE WNA16：
  w13_qweight / w2_qweight / w13_scales / w2_scales / qzeros / g_idx。
  需要转换到 fused MoE kernel 格式。
```

---

## 23. 容易混淆的点

### 23.1 Weight-only 是不是 activation 完全不参与量化？

不一定。

严格的 weight-only 是：

```text
W4A16 / W8A16 / WFP8A16
```

但 vLLM 的量化模块也包含 W8A8、W4A8、FP8 dynamic activation 等路径。

所以看源码时要区分：

```text
quantization 模块
  不等于
全部都是 weight-only。
```

### 23.2 GPTQ 和 gptq_marlin 是两个完全不同实现吗？

在当前路径里，`gptq`、`auto_gptq`、`gptq_marlin` 都映射到 `AutoGPTQConfig`。

位置：`vllm/vllm/model_executor/layers/quantization/__init__.py:151`

真正使用哪个 kernel，由 `choose_mp_linear_kernel()` 根据平台、shape、quant config、backend 设置选择。

### 23.3 AWQ 是否总是 Marlin？

不是。

AWQ 可能走：

```text
AutoAWQMarlinLinearMethod
AutoAWQLinearMethod
AutoAWQXPULinearMethod
CPU WNA16 kernel
```

取决于平台、batch invariant、Marlin 支持和 layer shape。

### 23.4 qweight 为什么经常不是 int4 dtype？

PyTorch 没有通用的 int4 Parameter dtype。

所以常见做法是：

```text
INT4 packed into int32 或 uint8；
通过 packed_factor 表示一个元素里塞了多少低 bit 权重。
```

例如 GPTQ / AWQ 常用 int32 packed，bitsandbytes 4bit 常用 uint8 packed。

### 23.5 为什么 post-load 还要改参数？

因为 checkpoint 格式和 runtime kernel 格式经常不同。

例如：

```text
AWQ checkpoint packed order
  → 标准 GPTQ-like packed order
  → Marlin kernel layout
```

又例如：

```text
FP8 checkpoint weight + scale
  → 合并 logical shards 的 scale
  → 转置 / block scale reshape
  → scaled-mm kernel layout
```

### 23.6 为什么 TP 会影响量化是否可用？

因为低 bit 权重和 scale 有额外对齐要求：

```text
output_size_per_partition 要能被 pack_factor 整除；
input_size_per_partition 要能被 group_size 整除；
block scale 要能按 block_n / block_k 对齐；
MoE expert shard 要能被 kernel tile 支持。
```

TP 越大，每个 rank 的局部维度越小，越容易触发不对齐。

---

## 24. 总结

Weight-only 量化在 vLLM 中的完整关系可以压缩成：

```text
quantization_config
  → QuantizationConfig.from_config()
  → quant_config.get_quant_method(layer, prefix)
  → quant_method.create_weights()
  → weight_loader 按 TP / fused module / packed layout 加载 checkpoint
  → quant_method.process_weights_after_loading()
  → kernel-specific layout / repack / online quant
  → quant_method.apply()
  → selected linear or fused MoE kernel
```

如果只记一句话：

```text
vLLM 的 weight-only 量化难点不是“把权重变小”，而是让 checkpoint 里的低 bit 权重、scale、zero point、TP 切分、fused module 和实际 kernel layout 精确对齐。
```

再压缩成最小心智模型：

```text
QuantizationConfig 选方法；
QuantizeMethod 建参数和跑 kernel；
Parameter / weight_loader 处理切分；
process_weights_after_loading 处理格式转换；
Linear / RoutedExperts 只负责把 forward 委托给 quant_method.apply()。
```
