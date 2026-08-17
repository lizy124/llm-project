# 05. 量化算子如何参与权重加载和 forward？

源码位置：

- `code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/model_executor/parameter.py`
- `code/vllm/vllm/_custom_ops.py`
- `code/vllm/vllm/model_executor/layers/quantization/utils/`
- `code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/`
- `code/vllm/vllm/model_executor/layers/quantization/online/`
- `code/vllm/csrc/quantization/`
- `code/vllm/csrc/moe/`

本问题关注：GPTQ、AWQ、FP8、INT8、Marlin、CUTLASS、compressed-tensors、online quant 等量化路径，如何在模型构造时替代普通 dense linear 的权重创建和 forward 计算；权重加载时如何保存 packed weight、scale、zero point、group metadata；运行时如何调用 backend-specific matmul kernel；以及这些量化算子和 TP、MoE、LoRA、CUDA Graph、KV cache 的边界。

---

## 0. 梳理规划

本篇按“先定抽象，再走 linear 主链路，再拆典型量化后端，最后总结边界和排查”的方式组织。

要回答的问题分成 12 组：

```text
1. vLLM 的量化算子处于哪一层？
2. QuantizationConfig / QuantizeMethodBase / LinearMethodBase 各自负责什么？
3. LinearBase 如何根据 quant_config 选择 quant_method？
4. create_weights() 如何创建 packed weight、scale、zero point 等参数？
5. weight_loader / process_weights_after_loading() 如何处理 checkpoint 权重？
6. forward() 如何从普通 GEMM 切换到 quantized kernel？
7. GPTQ / AWQ / FP8 / INT8 / Marlin / CUTLASS 路径有什么差异？
8. weight-only quantization 和 activation quantization 有什么区别？
9. MoE / embedding / KV cache 的量化路径和 linear 有什么不同？
10. TP / LoRA / CUDA Graph / torch.compile 会如何影响量化算子？
11. fallback 和硬件能力判断在哪里发生？
12. 量化相关问题如何 debug？
```

---

## 1. 一句话回答

量化算子把普通 linear / MoE expert GEMM 从：

```text
activation fp16/bf16 × weight fp16/bf16 → output
```

替换为：

```text
activation fp16/bf16/int8/fp8
  × packed / quantized weight
  + scale / zero point / group metadata
  → backend-specific GEMM kernel
  → output activation
```

在 vLLM 里，这个替换不是 ModelRunner 每轮临时判断，而是在 layer 构造时由 `QuantizationConfig.get_quant_method()` 选择 `QuantizeMethodBase` 子类，并由 `create_weights()` 创建该量化方法所需的参数；forward 时 layer 统一调用 `quant_method.apply()`。

最小主线是：

```text
ModelConfig / LoadConfig / HF quant config
  → QuantizationConfig
  → layer init receives quant_config
  → quant_config.get_quant_method(layer, prefix)
  → quant_method.create_weights(...)
  → weight_loader loads qweight / scales / zeros / metadata
  → process_weights_after_loading()
  → layer.forward()
  → quant_method.apply(layer, x, bias)
  → _custom_ops / CUTLASS / Marlin / Triton / torch fallback
  → output tensor
```

一句话记忆：

```text
量化配置决定 layer 用什么权重布局，quant_method 决定怎么加载和怎么算，backend kernel 决定最终性能和硬件约束。
```

---

## 2. 量化算子在 vLLM 执行链路中的位置

量化算子不直接和用户请求、Scheduler 或 OutputProcessor 交互。

它位于模型 layer 内部：

```text
GPUModelRunner._model_forward()
  → model.forward()
  → decoder layer
  → attention / MLP / MoE layer
  → LinearBase / FusedMoE / Embedding
  → quant_method.apply()
  → native op / backend kernel
```

对应职责边界：

```text
Scheduler：
  决定本轮哪些 token / request 被执行。

ModelRunner：
  准备 input_ids / positions / attention metadata / KV cache。

Model layer：
  组织 attention、MLP、MoE 等模型语义。

Quantization method：
  创建量化参数，加载权重，选择量化 kernel。

Native kernel：
  真正执行 packed weight matmul / scaled mm / grouped GEMM。
```

因此量化算子是“模型层内部的计算替换机制”，不是独立调度系统。

---

## 3. 核心抽象：QuantizationConfig 和 QuantizeMethodBase

### 3.1 QuantizeMethodBase

定义位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:19`

它是所有量化 method 的基类。

核心接口：

```python
class QuantizeMethodBase(ABC):
    uses_meta_device: bool = False

    def create_weights(self, layer, *weight_args, **extra_weight_attrs): ...
    def apply(self, layer, *args, **kwargs) -> torch.Tensor: ...
    def embedding(self, layer, *args, **kwargs) -> torch.Tensor: ...
    def tie_weights(self, layer, *args, **kwargs): ...
    def process_weights_after_loading(self, layer) -> None: ...
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:19` 到 `code/vllm/vllm/model_executor/layers/quantization/base_config.py:62`

关键职责：

```text
create_weights()：
  在 layer 上注册该量化方法需要的 Parameter，例如 qweight、qzeros、scales、g_idx、input_scale。

apply()：
  forward 时读取 layer 上的量化参数，调用对应 kernel。

process_weights_after_loading()：
  checkpoint 权重加载后做 repack、transpose、scale 合并、backend 初始化等后处理。

embedding()/tie_weights()：
  embedding 或 tied weight 场景的可选接口。
```

`uses_meta_device` 用于 online quantization，表示先在 meta device 创建权重，再在加载后逐层量化，降低加载峰值显存。

### 3.2 QuantizationConfig

定义位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:77`

它是量化配置基类。

关键接口：

```python
class QuantizationConfig(ABC):
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    def get_min_capability(cls) -> int: ...
    def get_config_filenames() -> list[str]: ...
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig: ...
    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None: ...
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:77` 到 `code/vllm/vllm/model_executor/layers/quantization/base_config.py:170`

它负责：

```text
- 解析 checkpoint / HF quant config；
- 声明支持的 activation dtype；
- 声明最低 GPU capability；
- 判断某个 layer 是否适用该量化方法；
- 返回具体 QuantizeMethodBase 实例；
- 必要时提供 KV cache scale mapper 或模型名修正。
```

### 3.3 QuantizationConfig 和 QuantizeMethodBase 的关系

可以记成：

```text
QuantizationConfig：
  这是“这个模型用什么量化方案”的全局配置。

QuantizeMethodBase：
  这是“这个具体 layer 如何创建权重、加载权重、执行 forward”的局部策略。
```

同一个 `QuantizationConfig` 可以根据 layer 类型返回不同 method，例如：

```text
Linear layer → Linear quant method
Embedding layer → Embedding quant method
FusedMoE layer → FusedMoE quant method
不支持的 layer → None
```

---

## 4. LinearBase 如何接入量化

`LinearMethodBase` 定义在：`code/vllm/vllm/model_executor/layers/linear.py:138`

它继承 `QuantizeMethodBase`，专门描述 linear 层的量化方法。

核心接口：

```python
class LinearMethodBase(QuantizeMethodBase):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ): ...

    def apply(self, layer, x, bias=None) -> torch.Tensor: ...
```

### 4.1 LinearBase 选择 quant_method

`LinearBase.__init__()` 中：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:265` 到 `code/vllm/vllm/model_executor/layers/linear.py:274`

这说明：

```text
Linear 层本身不关心 GPTQ / AWQ / FP8 细节；
它只保存一个 quant_method，后续 create_weights / forward 都委托给它。
```

### 4.2 create_weights() 在 layer 初始化时执行

以 `ReplicatedLinear` 为例：

```python
self.quant_method.create_weights(
    self,
    self.input_size,
    self.output_partition_sizes,
    self.input_size,
    self.output_size,
    self.params_dtype,
    weight_loader=self.weight_loader,
)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:338` 到 `code/vllm/vllm/model_executor/layers/linear.py:346`

对于 ColumnParallelLinear、RowParallelLinear、QKVParallelLinear、MergedColumnParallelLinear 等，也会用各自的 input/output partition size 调用同一套 method 接口。

### 4.3 forward() 统一调用 apply()

`ReplicatedLinear.forward()` 中：

```python
output = self.quant_method.apply(self, x, bias)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:372` 到 `code/vllm/vllm/model_executor/layers/linear.py:383`

这就是量化算子接入 forward 的核心位置。

对上层模型来说：

```text
self.qkv_proj(hidden_states)
self.gate_up_proj(hidden_states)
self.down_proj(hidden_states)
```

仍然是普通 layer 调用；内部由 `quant_method.apply()` 决定执行普通 GEMM 还是量化 GEMM。

---

## 5. 权重加载：create_weights、weight_loader、process_weights_after_loading

量化路径不是只改 forward kernel，还要改变权重参数形态。

### 5.1 普通未量化权重

`UnquantizedLinearMethod.create_weights()` 创建：

```text
weight: [output_partition, input_partition]
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:179` 到 `code/vllm/vllm/model_executor/layers/linear.py:209`

forward 时：

```python
return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:217` 到 `code/vllm/vllm/model_executor/layers/linear.py:225`

### 5.2 量化权重会创建更多参数

不同量化方法会注册不同参数，例如：

```text
qweight / weight：packed or fp8 weight
qzeros / zeros：zero point
scales / weight_scale / weight_scale_inv：weight scale
input_scale：activation static scale
g_idx：group index
bias：可选 bias
workspace / metadata：backend-specific buffer
```

这些参数通常通过 `set_weight_attrs()` 设置：

```text
input_dim / output_dim
packed_dim
weight_loader
shard_id
scale_type
```

以便 AutoWeightsLoader 或 model-specific weight_loader 正确把 checkpoint tensor 写入对应 shard。

### 5.3 weight_loader 负责把 checkpoint tensor 放到参数里

linear 层的 `weight_loader` 会处理：

```text
- tensor parallel shard；
- QKV / gate_up 等 fused module 的 shard_id；
- scale / zero point 的 shape；
- packed weight 的布局；
- checkpoint 名称和 vLLM 参数名映射。
```

对于量化方法，`weight_loader` 往往要配合 Parameter 子类和 attrs 才能正确切分。

### 5.4 process_weights_after_loading()

`process_weights_after_loading()` 是量化方法常用的后处理入口。

它可能做：

```text
- transpose weight 到 kernel 需要的 K/N layout；
- repack GPTQ / AWQ weight 到 Marlin layout；
- 合并或重算 scale；
- 把 checkpoint fp16 权重量化成 fp8/int8；
- 初始化 backend kernel wrapper；
- 检查硬件能力和 shape 对齐。
```

例如 FP8 路径中，`process_weights_after_loading()` 会处理 Marlin layout、block quant、per-tensor strategy、input_scale 等。

源码位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:398` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:444`

---

## 6. forward：量化 linear 如何调用 kernel

forward 的统一入口是：

```text
LinearBase.forward()
  → quant_method.apply(layer, x, bias)
```

不同量化 method 的 `apply()` 会调用不同 backend。

### 6.1 GPTQ

GPTQ 配置和方法在：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py`

关键入口：

- `AutoGPTQConfig`: `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:96`
- `get_quant_method()`: `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:239`
- `create_weights()`: `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:323`
- `apply()`: `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:455`

典型参数：

```text
qweight
qzeros
scales
g_idx
bits
group_size
```

底层 op 入口包括：

```text
_custom_ops.gptq_gemm
_custom_ops.gptq_shuffle
_custom_ops.gptq_marlin_repack
_custom_ops.marlin_gemm
```

相关位置：

- `code/vllm/vllm/_custom_ops.py:615`
- `code/vllm/vllm/_custom_ops.py:655`
- `code/vllm/vllm/_custom_ops.py:1210`
- `code/vllm/vllm/_custom_ops.py:1327`

### 6.2 AWQ

AWQ 配置在：`code/vllm/vllm/model_executor/layers/quantization/awq.py`

关键入口：

- `AWQConfig`: `code/vllm/vllm/model_executor/layers/quantization/awq.py:34`
- `get_quant_method()`: `code/vllm/vllm/model_executor/layers/quantization/awq.py:97`
- `create_weights()`: `code/vllm/vllm/model_executor/layers/quantization/awq.py:182`
- `apply()`: `code/vllm/vllm/model_executor/layers/quantization/awq.py:262`

底层 op 包括：

```text
_custom_ops.awq_dequantize
_custom_ops.awq_gemm
_custom_ops.awq_marlin_repack
_custom_ops.marlin_gemm
```

相关位置：

- `code/vllm/vllm/_custom_ops.py:548`
- `code/vllm/vllm/_custom_ops.py:582`
- `code/vllm/vllm/_custom_ops.py:1244`
- `code/vllm/vllm/_custom_ops.py:1327`

### 6.3 AWQ Marlin

AWQ Marlin 在：`code/vllm/vllm/model_executor/layers/quantization/awq_marlin.py`

关键入口：

- `AWQMarlinConfig`: `code/vllm/vllm/model_executor/layers/quantization/awq_marlin.py:167`
- `get_quant_method()`: `code/vllm/vllm/model_executor/layers/quantization/awq_marlin.py:279`
- `create_weights()`: `code/vllm/vllm/model_executor/layers/quantization/awq_marlin.py:404`
- `apply()`: `code/vllm/vllm/model_executor/layers/quantization/awq_marlin.py:503`

它的核心是把 AWQ packed weight 转成 Marlin kernel 高效使用的布局。

### 6.4 FP8

FP8 配置在：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:99`

关键入口：

- `Fp8Config`: `code/vllm/vllm/model_executor/layers/quantization/fp8.py:99`
- `get_quant_method()`: `code/vllm/vllm/model_executor/layers/quantization/fp8.py:179`
- `create_weights()`: `code/vllm/vllm/model_executor/layers/quantization/fp8.py:322`
- `apply()`: `code/vllm/vllm/model_executor/layers/quantization/fp8.py:446`

FP8 路径会根据配置和硬件选择：

```text
- per-tensor fp8；
- block fp8；
- static activation scale；
- dynamic per-token activation scale；
- MarlinFP8ScaledMMLinearKernel；
- CutlassFP8ScaledMMLinearKernel；
- torch._scaled_mm / fallback。
```

`create_weights()` 会创建：

```text
weight
weight_scale 或 weight_scale_inv
input_scale（如果静态 activation quant）
```

并初始化：

```text
self.fp8_linear = init_fp8_linear_kernel(...)
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:387` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:396`

底层 op 包括：

```text
_custom_ops.cutlass_scaled_mm
_custom_ops.cutlass_scaled_mm_azp
_custom_ops.scaled_fp8_quant
```

位置：

- `code/vllm/vllm/_custom_ops.py:813`
- `code/vllm/vllm/_custom_ops.py:864`
- `code/vllm/vllm/_custom_ops.py:1955`

### 6.5 INT8 / W8A8

INT8 相关路径分散在：

```text
quantization/experts_int8.py
quantization/online/int8.py
quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
quantization/utils/int8_utils.py
quantization/utils/w8a8_utils.py
```

典型形态：

```text
weight int8 + activation int8/fp16/bf16 + scales
```

底层 op 包括：

```text
_custom_ops.scaled_int8_quant
CUTLASS scaled mm
compressed-tensors triton_scaled_mm
```

位置：`code/vllm/vllm/_custom_ops.py:2109`

### 6.6 compressed-tensors

Compressed Tensors 是一套更通用的 quantization scheme 入口。

关键文件：

```text
code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/
code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/
```

它会根据 checkpoint 中的 scheme 创建对应 linear / embedding / MoE 方法，例如：

```text
W4A16
W4A8 FP8 / INT8
W8A8 FP8 / INT8
MXFP4 / NVFP4 / MXFP8
```

### 6.7 online quantization

online quantization 相关文件：

```text
code/vllm/vllm/model_executor/layers/quantization/online/base.py
code/vllm/vllm/model_executor/layers/quantization/online/fp8.py
code/vllm/vllm/model_executor/layers/quantization/online/int8.py
code/vllm/vllm/model_executor/layers/quantization/online/mxfp8.py
```

它和 checkpoint 已经量化不同：

```text
checkpoint 可能仍是 fp16/bf16，加载后 vLLM 在本地按 layer 做量化。
```

这时 `uses_meta_device` 和 `process_weights_after_loading()` 的意义更明显：降低加载峰值内存，并在权重可用后生成量化布局。

---

## 7. weight-only quantization 和 activation quantization

### 7.1 weight-only

weight-only 量化只量化权重。

典型：

```text
activation: fp16/bf16
weight: int4/int8 packed
scale/zero point: per-group or per-channel
```

常见方法：

```text
GPTQ
AWQ
Marlin W4A16
WNA16
```

优点：

```text
- 显著降低权重显存；
- activation 不需要每轮动态量化；
- 对 serving runtime 动态 batch 友好。
```

代价：

```text
- matmul kernel 必须支持 packed weight 解码；
- group size / layout / hardware capability 限制较多。
```

### 7.2 weight + activation quantization

activation 也量化，例如 W8A8、FP8、W4A8。

典型：

```text
activation: fp8/int8/mxfp8
weight: fp8/int8/int4
scale: input_scale + weight_scale
```

关键差异：

```text
- 每轮 forward 可能需要对 activation 做 dynamic quant；
- 需要 static 或 dynamic input_scale；
- CUDA Graph 下 activation quant kernel 也要满足 shape / address 稳定；
- 对硬件 backend 要求更高。
```

FP8 中就会区分：

```text
static activation scale
dynamic per-token activation scale
dynamic per-tensor activation scale
block quant activation scale
```

相关逻辑见：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:301` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:320`

---

## 8. 量化和 MoE 的关系

MoE 的 expert 权重也可以量化，但路径和普通 linear 不完全一样。

普通 linear 是：

```text
LinearBase
  → LinearMethodBase.create_weights()
  → LinearMethodBase.apply()
```

MoE 是：

```text
FusedMoE / RoutedExperts
  → FusedMoEMethodBase.create_weights()
  → router top-k / dispatch
  → quant_method.apply() or apply_monolithic()
  → grouped GEMM / fused expert kernel
```

相关基类：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py:31`

MoE 量化方法需要处理：

```text
- w13 / gate_up_proj merged weight；
- w2 / down_proj weight；
- expert 维度；
- expert map / EP placement；
- topk ids / weights；
- grouped GEMM layout；
- quantized expert scale / zero point；
- combine / reduce 是否 kernel 内完成。
```

典型文件：

```text
quantization/experts_int8.py
quantization/moe_wna16.py
quantization/compressed_tensors/compressed_tensors_moe/
fused_moe/experts/cutlass_moe.py
fused_moe/experts/deep_gemm_moe.py
fused_moe/experts/flashinfer_cutlass_moe.py
```

---

## 9. 量化和 KV cache 的关系

KV cache 量化和 weight quantization 不是同一层。

相关文件：`code/vllm/vllm/model_executor/layers/quantization/kv_cache.py`

KV cache 量化关注：

```text
- attention K/V cache 的 dtype；
- KV cache scale；
- cache write / read kernel；
- attention backend 如何读取量化 KV。
```

weight quantization 关注：

```text
- linear / MoE 权重布局；
- matmul kernel；
- scale / zero point；
- activation quantization。
```

两者可能同时启用，但调试时要分开：

```text
输出异常可能来自 quantized linear，
也可能来自 quantized KV cache / attention backend。
```

---

## 10. 量化和 TP / PP / DP

### 10.1 Tensor Parallel

TP 会影响 linear weight 的 partition：

```text
ColumnParallelLinear：按 output dim 切；
RowParallelLinear：按 input dim 切；
QKVParallelLinear：按 Q/K/V shard 切；
MergedColumnParallelLinear：多个 logical weight 合并后切。
```

量化方法的 `create_weights()` 必须根据：

```text
input_size_per_partition
output_partition_sizes
input_size
output_size
```

创建正确 shard 的 packed weight / scale。

如果 group size、pack factor、block size 和 TP shard 不对齐，就会导致加载失败或 kernel 不支持。

### 10.2 Pipeline Parallel

PP 不直接改变单个 layer 的量化方法，但每个 PP rank 只加载自己 stage 的层。

量化权重加载要保证：

```text
只加载本 rank 持有的 layer 参数；
prefix / weight name mapping 正确；
process_weights_after_loading 在本地 shard 上执行。
```

### 10.3 Data Parallel / Expert Parallel

DP 对普通 dense linear 影响较小，但对 MoE quantization 影响明显。

MoE + EP 需要：

```text
expert map；
local experts；
expert weight placement；
All2All / grouped GEMM layout；
quantized expert scale 与 expert id 对齐。
```

---

## 11. 量化和 LoRA

LoRA 通常假设 base layer 输出保持标准 activation dtype。

量化 base weight + LoRA 组合时要注意：

```text
- base linear 走 quant_method.apply()；
- LoRA delta 通常单独计算并加到输出；
- 某些量化方法不支持 LoRA；
- CUDA graph key 可能需要区分 active LoRA 数；
- MoE + LoRA 还有 routed experts / custom op 约束。
```

对于 CUDA Graph，LoRA active adapter 数会进入 `BatchDescriptor`：

```text
has_lora
num_active_loras
```

如果 LoRA case 未 capture，可能 fallback。

---

## 12. 量化和 CUDA Graph / torch.compile

量化算子对 CUDA Graph / compile 的影响主要有三类。

### 12.1 kernel 是否 graph-safe

某些 backend kernel 可以稳定 capture / replay，某些路径会因为 workspace、dynamic metadata 或 unsupported op fallback。

例如：

```text
CUTLASS scaled mm 通常更适合固定 shape；
某些 Triton JIT 路径可能有首次 compile 开销；
Marlin repack 应发生在加载后，不应在 forward 中动态做。
```

### 12.2 shape / metadata 是否稳定

activation quantization 可能引入额外 scale tensor：

```text
input_scale
per-token dynamic scale
block scale
```

这些 tensor 的 shape 和地址也会影响 graph replay。

### 12.3 compiled graph 是否支持 custom op

vLLM 的 custom op / torch.ops 通常会作为 opaque op 进入 compile graph。

如果 op 没有 fake impl、shape 函数或 compile 支持不足，可能导致 graph break 或 fallback。

---

## 13. backend selection 和硬件能力

每个量化配置会声明最低硬件能力：

```python
QuantizationConfig.get_min_capability()
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:95`

例如不同 backend 可能要求：

```text
- Ampere / Hopper；
- CUDA-like platform；
- ROCm 特定路径；
- CPU fallback；
- CUTLASS fp8 support；
- Marlin supported shape / dtype；
- XPU 特定 op。
```

FP8 路径会根据支持情况选择不同 kernel：

```text
CutlassFP8ScaledMMLinearKernel
MarlinFP8ScaledMMLinearKernel
torch / fallback scaled mm
```

相关判断和 op：

- `code/vllm/vllm/_custom_ops.py:805`
- `code/vllm/vllm/_custom_ops.py:809`
- `code/vllm/vllm/_custom_ops.py:813`

如果硬件不支持，vLLM 通常会：

```text
- 配置阶段报错；
- 选择另一个 backend；
- fallback 到 torch / unquantized / dequantized path；
- 对某些组合直接不支持。
```

---

## 14. 常见问题和排查

### 14.1 权重 shape 不匹配

常见原因：

```text
- TP shard 切分和 checkpoint shape 不一致；
- QKV / gate_up fused module 的 shard_id 不对；
- scale / zero point shape 没有按 logical weights 对齐；
- block quant weight_block_size 不满足 layer shape；
- group_size 和 hidden dim 不整除。
```

排查：

```text
1. 看 layer prefix；
2. 看 quant_config.get_quant_method() 是否返回预期 method；
3. 看 create_weights() 注册的参数 shape；
4. 看 weight_loader 处理 shard_id / packed_dim 的逻辑；
5. 看 process_weights_after_loading() 是否改了 weight layout。
```

### 14.2 输出精度异常

可能原因：

```text
- scale / zero point 加载错；
- activation scale 静态/动态策略不匹配；
- checkpoint 不是预期格式；
- Marlin / CUTLASS repack 错；
- FP8 per-tensor vs per-channel scale 混用；
- KV cache quantization 和 linear quantization 混淆。
```

排查方法：

```text
- 对比 unquantized 输出；
- 关闭 CUDA graph；
- 切换 backend；
- 检查 loaded qweight / scale 范围；
- 检查 process_weights_after_loading() 后的参数 dtype / shape。
```

### 14.3 性能不达预期

可能原因：

```text
- batch 太小，quant kernel launch overhead 高；
- shape 不满足 Marlin / CUTLASS 最优路径；
- activation quant dynamic scale 成本高；
- 发生 dequant + bf16 GEMM fallback；
- Triton / CUTLASS 首次 JIT；
- MoE token imbalance；
- CUDA graph fallback 到 NONE。
```

### 14.4 CUDA Graph 下异常

优先检查：

```text
- 量化 op 是否支持 graph capture；
- activation scale tensor shape / address 是否稳定；
- quant kernel 是否在 forward 中动态分配 workspace；
- LoRA active count 是否导致 graph key 变化；
- cudagraph_metrics runtime_mode 是否为预期。
```

---

## 15. 端到端例子：FP8 Linear

以 FP8 linear 为例，完整链路是：

```text
1. 模型配置启用 fp8 quantization。

2. LinearBase 初始化：
   quant_config.get_quant_method(layer, prefix)
   → 返回 Fp8LinearMethod。

3. create_weights()：
   → 创建 layer.weight；
   → 创建 weight_scale 或 weight_scale_inv；
   → 如果 act_q_static，创建 input_scale；
   → init_fp8_linear_kernel(...) 选择 backend。

4. checkpoint 加载：
   → weight_loader 写入 weight / scale。

5. process_weights_after_loading()：
   → 必要时 transpose；
   → 处理 per-tensor / block scale；
   → Marlin / CUTLASS 后端准备。

6. forward：
   Linear.forward(x)
     → quant_method.apply(layer, x, bias)
     → fp8_linear.apply_weights(...)
     → cutlass_scaled_mm / marlin / fallback
     → output。
```

对应源码锚点：

- `code/vllm/vllm/model_executor/layers/linear.py:265`
- `code/vllm/vllm/model_executor/layers/linear.py:338`
- `code/vllm/vllm/model_executor/layers/linear.py:372`
- `code/vllm/vllm/model_executor/layers/quantization/fp8.py:322`
- `code/vllm/vllm/model_executor/layers/quantization/fp8.py:398`
- `code/vllm/vllm/model_executor/layers/quantization/fp8.py:446`

---

## 16. 最关键对象关系

```text
QuantizationConfig
  全局量化配置，解析 checkpoint / HF config，决定 layer 对应 method。

QuantizeMethodBase
  layer 级量化策略接口，定义 create_weights / apply / process_weights_after_loading。

LinearMethodBase
  linear 层专用 quant method 基类。

LinearBase
  保存 quant_method，统一在 forward 中调用 quant_method.apply()。

ModelWeightParameter / BasevLLMParameter
  带 weight_loader 和 TP metadata 的参数对象。

_custom_ops.py
  Python 到 native op 的桥，例如 gptq_gemm、awq_gemm、marlin_gemm、cutlass_scaled_mm、scaled_fp8_quant。

FusedMoEMethodBase
  MoE expert 权重量化和 fused grouped GEMM 的量化策略接口。

KV cache quantization
  独立于 weight quantization，影响 attention KV cache 读写。
```

---

## 17. 最小心智模型

如果只记一条主线，可以记：

```text
quant_config 不是 forward 时临时分支，
而是在 layer 构造阶段把 layer 的参数形态和 apply() 策略换掉；
权重加载后再做 repack / scale 处理，
forward 时 layer 仍然像普通 Linear 一样被调用，
但内部已经变成 quant_method.apply() 调用专用量化 kernel。
```

再压缩成一句话：

```text
vLLM 的量化算子 = 量化配置选择 method + method 创建量化参数 + 权重加载后处理 + apply 调用 backend GEMM。
```
