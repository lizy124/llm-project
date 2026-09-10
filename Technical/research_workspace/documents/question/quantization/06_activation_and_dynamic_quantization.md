# 06. Activation / Dynamic quantization 如何参与 forward？

源码位置：

- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/fp8.py`
- `code/vllm/vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `code/vllm/vllm/model_executor/layers/quantization/online/fp8.py`
- `code/vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py`
- `code/vllm/vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py`
- `code/vllm/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`
- `code/vllm/vllm/model_executor/layers/fusion/quant_activation.py`
- `code/vllm/vllm/compilation/passes/fusion/act_quant_fusion.py`
- `code/vllm/vllm/compilation/passes/fusion/rms_quant_fusion.py`
- `code/vllm/vllm/compilation/passes/fusion/attn_quant_fusion.py`
- `code/vllm/vllm/_custom_ops.py`
- `vllm/csrc/libtorch_stable/quantization/`

本问题关注：activation quantization、dynamic quantization、per-token scaling、per-token-group / block scaling 这些运行时量化机制如何参与 forward；它们和 weight-only quantization 的区别；scale 在哪里计算；量化后的 activation 如何进入 scaled GEMM；以及 torch.compile / CUDA graph / fusion 对这条链路有什么影响。

---

## 1. 一句话回答

Activation / dynamic quantization 不是只改变权重存储格式，而是在 forward 过程中把当前 activation 也量化成 FP8 / INT8 / FP4 等低精度张量，并把运行时计算出的 activation scale 一起传给后续 GEMM 或 fused kernel。

最典型的 FP8 linear 链路是：

```text
LinearBase.forward(x)
  → quant_method.apply(layer, x, bias)
  → Fp8LinearMethod.apply()
  → fp8_linear.apply_weights(layer, x, bias)
  → QuantFP8(x, input_scale 或 None)
  → ops.scaled_fp8_quant(...)
  → dynamic/static FP8 quant kernel
  → cutlass_scaled_mm(A=quantized_x, B=fp8_weight, scale_a=x_scale, scale_b=w_scale)
  → output dtype hidden states
```

所以：

```text
weight quantization：
  主要决定 weight 如何存、如何加载、weight_scale 如何保存。

activation quantization：
  决定每次 forward 的输入 x 是否要先量化，以及 input_scale 如何得到。

static activation quantization：
  input_scale 来自 checkpoint / 参数，forward 直接使用。

dynamic activation quantization：
  input_scale 由当前 forward 的 x 现场计算，常见是 per-tensor / per-token / per-token-group。
```

---

## 2. 最小主链路

从 linear 层看，所有量化方法最终都挂在 `LinearBase.forward()` 的 `quant_method.apply()` 上。

入口：`linear.py:372`

```python
output = self.quant_method.apply(self, x, bias)
```

对于 FP8 linear，常见链路是：

```text
LinearBase.forward()
  → Fp8LinearMethod.apply()
  → FP8ScaledMMLinearKernel.apply_weights()
  → QuantFP8.forward_cuda() / forward_native()
  → ops.scaled_fp8_quant()
  → _C.dynamic_per_token_scaled_fp8_quant
     或 _C.dynamic_scaled_fp8_quant
     或 _C.static_scaled_fp8_quant
  → ops.cutlass_scaled_mm()
```

关键文件：

```text
linear.py
  LinearBase.forward()

fp8.py
  Fp8LinearMethod.create_weights()
  Fp8LinearMethod.process_weights_after_loading()
  Fp8LinearMethod.apply()

ScaledMMLinearKernel.py
  FP8ScaledMMLinearKernel.apply_weights()

input_quant_fp8.py
  QuantFP8

_custom_ops.py
  scaled_fp8_quant()
  scaled_int8_quant()

cutlass.py
  CutlassFP8ScaledMMLinearKernel.apply_scaled_mm()
```

---

## 3. Activation quantization 和 weight-only 的区别

Weight-only 量化可以理解成：

```text
加载前 / 加载时：
  weight_fp16/bf16
    → quantized_weight + weight_scale

forward 时：
  activation 仍然保持 fp16/bf16
    → 用特殊 kernel 读取 quantized_weight
    → 内部反量化或低精度 GEMM
```

Activation quantization 则变成：

```text
forward 时：
  activation_fp16/bf16
    → 计算 input_scale
    → quantized_activation_fp8/int8
    → scaled GEMM 使用 input_scale + weight_scale
```

区别在于：

```text
1. weight-only 的主要成本在加载 / 权重处理阶段；
2. activation quant 的成本在每次 forward；
3. dynamic activation quant 的 scale 和当前 batch/token 内容有关；
4. scaled GEMM 必须同时知道 activation scale 和 weight scale；
5. dynamic scale 的粒度会影响精度、性能和 kernel 选择。
```

---

## 4. static / dynamic activation scale 的核心差异

### 4.1 static activation scale

Static activation quantization 的 input scale 不是 forward 现场算的。

在 FP8 checkpoint 路径里：

```text
Fp8LinearMethod.create_weights()
  → 如果 activation_scheme == "static"
  → 创建 layer.input_scale
```

位置：`fp8.py:381` 到 `fp8.py:385`

forward 时：

```text
QuantFP8(static=True)
  → scale 参数必须存在
  → ops.scaled_fp8_quant(input, scale=layer.input_scale)
  → _C.static_scaled_fp8_quant(...)
```

也就是说：

```text
static：scale 是参数 / checkpoint 信息；
dynamic：scale 是当前 activation 的统计结果。
```

### 4.2 dynamic activation scale

Dynamic activation quantization 的 input scale 来自当前输入 `x`。

非 group FP8 路径中：

```text
scale is None
  → dynamic quantization
```

位置：`_custom_ops.py:1939`

如果是 dynamic per-token：

```python
scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
```

位置：`_custom_ops.py:1940` 到 `_custom_ops.py:1944`

如果是 dynamic per-tensor：

```python
scale = torch.empty(1, device=input.device, dtype=torch.float32)
torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
```

位置：`_custom_ops.py:1945` 到 `_custom_ops.py:1947`

因此：

```text
dynamic per-tensor：
  整个 activation 矩阵一个 scale。

dynamic per-token：
  每个 token / row 一个 scale。

dynamic per-token-group：
  每个 token 的 hidden 维再按 group 切分，每个 group 一个 scale。
```

---

## 5. vLLM 如何描述 scale 粒度

核心定义在 `quant_utils.py`。

### 5.1 GroupShape

`GroupShape` 用 `(row, col)` 描述 scale 覆盖的 activation / weight 分组形状。

位置：`quant_utils.py:44` 到 `quant_utils.py:70`

常见值：

```text
GroupShape.PER_TENSOR = (-1, -1)
  整个矩阵一个 scale。

GroupShape.PER_TOKEN = (1, -1)
  每个 token / row 一个 scale。

GroupShape.PER_CHANNEL = (-1, 1)
  每个 channel 一个 scale，常用于 weight scale。

GroupShape(1, 128)
  每个 token 每 128 个 hidden 元素一个 scale。
```

### 5.2 QuantKey

`QuantKey` 是 vLLM 对“量化格式 + scale 类型”的统一描述。

位置：`quant_utils.py:99` 起

它包含：

```text
dtype：
  量化后的数据类型，例如 fp8 / fp4 / int8。

scale：
  scale 的 dtype、static/dynamic、group_shape。

scale2：
  二级 scale，用于 NVFP4 / MXFP 等格式。

symmetric：
  是否对称量化。
```

常见 FP8 key：

```text
kFp8StaticTensorSym
  FP8 + static + per-tensor scale。

kFp8DynamicTensorSym
  FP8 + dynamic + per-tensor scale。

kFp8DynamicTokenSym
  FP8 + dynamic + per-token scale。

kFp8Dynamic128Sym
  FP8 + dynamic + per-token-group，每组 128 hidden。

kFp8Dynamic64Sym
  FP8 + dynamic + per-token-group，每组 64 hidden。
```

### 5.3 用户配置名

用户侧可见名称在 `config/quantization.py`：

位置：`config/quantization.py:23` 到 `config/quantization.py:34`

```text
fp8_per_tensor_static
fp8_per_tensor_dynamic
fp8_per_token
fp8_per_channel_static
fp8_per_block_static
fp8_per_block_dynamic
mxfp8
mxfp4
int8_per_channel_static
```

这些名称最终会解析成 `QuantKey`，再影响 linear / MoE 的 kernel 选择。

---

## 6. FP8 LinearMethod 如何决定 activation quant key

FP8 linear 的关键类是：

```text
Fp8LinearMethod
```

位置：`fp8.py:267`

它支持：

```text
- FP8 checkpoint weight；
- static weight scale；
- static activation scale；
- dynamic activation scale；
- block FP8 weight；
- online FP8 weight quantization 场景。
```

### 6.1 非 block FP8

非 block FP8 中：

```python
self.weight_quant_key = kFp8StaticTensorSym
```

然后根据 activation scheme 和平台能力决定 activation quant key：

```text
activation_scheme == "static"
  → kFp8StaticTensorSym

activation_scheme == "dynamic" 且 cutlass_fp8_supported()
  → kFp8DynamicTokenSym

activation_scheme == "dynamic" 且不支持 cutlass fp8
  → kFp8DynamicTensorSym
```

位置：`fp8.py:312` 到 `fp8.py:320`

含义是：

```text
能走 Cutlass FP8 时，vLLM 倾向用 dynamic per-token activation quant；
不能走时，可能退到 dynamic per-tensor。
```

### 6.2 block FP8

如果 `weight_block_size is not None`：

```text
weight_quant_key:
  static block scale，例如 GroupShape(128, 128)

activation_quant_key:
  dynamic per-token-group，例如 GroupShape(1, 128)
```

位置：`fp8.py:301` 到 `fp8.py:311`

这类路径常见于 DeepSeek 风格的 block FP8 / W8A8：

```text
activation：每个 token，每 128 hidden 一个 scale；
weight：每个 block 一个 scale；
GEMM：block-scaled FP8 kernel。
```

### 6.3 static activation scale 参数

如果 `activation_scheme == "static"`：

```python
scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
layer.register_parameter("input_scale", scale)
```

位置：`fp8.py:381` 到 `fp8.py:385`

如果是 dynamic activation quant：

```text
layer.input_scale = None
```

这会让后续 `QuantFP8` 看到 `scale=None`，从而进入 dynamic scale 计算。

---

## 7. FP8 forward 里 activation quant 发生在哪里

`Fp8LinearMethod.apply()` 本身很薄：

位置：`fp8.py:446` 到 `fp8.py:489`

普通路径是：

```python
return self.fp8_linear.apply_weights(layer, x, bias)
```

真正做 activation quant 的是 kernel 层：

```text
FP8ScaledMMLinearKernel.apply_weights()
```

位置：`ScaledMMLinearKernel.py:135` 到 `ScaledMMLinearKernel.py:170`

核心逻辑：

```text
1. 取出 weight、weight_scale、input_scale、input_scale_ub；
2. 判断 x 是否已经是 QuantizedActivation；
3. 如果不是，展平成 2D；
4. 调 QuantFP8(x_2d, x_s, x_s_ub)；
5. 得到 x_2d_q 和 x_s；
6. 调 apply_scaled_mm(A=x_2d_q, B=w, As=x_s, Bs=w_s)。
```

对应代码结构：

```python
x_2d = x_data.view(-1, x_data.shape[-1])
...
if qa is None:
    x_2d_q, x_s = self.quant_fp8(x_2d, x_s, x_s_ub)
return self.apply_scaled_mm(A=x_2d_q, B=w, As=x_s, Bs=w_s, ...)
```

位置：`ScaledMMLinearKernel.py:155` 到 `ScaledMMLinearKernel.py:170`

因此：

```text
LinearMethod 决定用哪种 quant kernel；
ScaledMMLinearKernel 决定 forward 里是否量化 activation；
QuantFP8 负责把 x 变成低精度张量和 scale；
Cutlass / Triton / Marlin / DeepGEMM kernel 负责实际 GEMM。
```

---

## 8. QuantFP8 做了什么

`QuantFP8` 是 activation FP8 quantization 的 Python 层 CustomOp wrapper。

位置：`input_quant_fp8.py:29`

它支持：

```text
- static quantization；
- dynamic quantization；
- per-tensor scale；
- per-token scale；
- per-channel scale；
- per-token-group scale；
- CUDA / ROCm / XPU / native fallback。
```

### 8.1 初始化参数

位置：`input_quant_fp8.py:38` 到 `input_quant_fp8.py:70`

关键参数：

```text
static：
  True 表示 scale 外部提供；False 表示 forward 动态计算。

group_shape：
  scale 粒度，例如 PER_TENSOR / PER_TOKEN / GroupShape(1, 128)。

num_token_padding：
  某些 downstream kernel 需要 token 维 padding。

column_major_scales / tma_aligned_scales：
  block scale 布局相关，给 DeepGEMM / TMA 路径使用。

use_ue8m0：
  是否使用 UE8M0 scale 表示，常见于 DeepGEMM。
```

### 8.2 dynamic non-group 限制

非 group dynamic quant 只支持：

```text
PER_TOKEN
PER_TENSOR
```

对应断言：`input_quant_fp8.py:78` 到 `input_quant_fp8.py:82`

原因是：

```text
dynamic per-channel 对 activation forward 不常用；
per-channel 更常见于 weight scale；
activation dynamic 主要按整个 tensor 或 token 行统计。
```

### 8.3 CUDA 路径

`forward_cuda()` 的主要分支：

```text
如果是 dynamic group quant：
  → fp8_utils.per_token_group_quant_fp8(...)

否则：
  → ops.scaled_fp8_quant(...)
```

位置：`input_quant_fp8.py:84` 到 `input_quant_fp8.py:133`

所以可以分成两类：

```text
non-group：
  per-tensor / per-token，走 _custom_ops.scaled_fp8_quant。

group：
  per-token-group，走 fp8_utils.per_token_group_quant_fp8 或 DeepGEMM packed kernel。
```

### 8.4 native fallback 展示了 scale 公式

`forward_native()` 里能直接看到 dynamic scale 的数学逻辑。

位置：`input_quant_fp8.py:184` 到 `input_quant_fp8.py:231`

per-token：

```python
x_max, _ = x.abs().max(dim=-1)
x_max = x_max.unsqueeze(-1).to(torch.float32)
```

per-tensor：

```python
x_max = x.abs().max().unsqueeze(-1).to(torch.float32)
```

scale：

```python
scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
```

量化：

```python
out = x.float() * scale.reciprocal()
out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
```

因此 dynamic activation quant 的本质是：

```text
当前 activation 的 absmax
  → 除以 FP8 最大可表示值
  → 得到 scale
  → x / scale
  → clamp 到 FP8 范围
  → cast 成 FP8
```

---

## 9. `_custom_ops.scaled_fp8_quant()` 的三种路径

入口：`_custom_ops.py:1885`

函数签名：

```python
def scaled_fp8_quant(
    input,
    scale=None,
    num_token_padding=None,
    scale_ub=None,
    use_per_token_if_dynamic=False,
    output=None,
    group_shape=None,
):
```

它根据 `scale` 是否存在决定 static / dynamic。

### 9.1 dynamic per-token

条件：

```text
scale is None
use_per_token_if_dynamic == True
```

行为：

```text
分配 scale shape = [M, 1]
调用 _C.dynamic_per_token_scaled_fp8_quant
```

位置：`_custom_ops.py:1939` 到 `_custom_ops.py:1944`

其中 `M` 是 flatten 后的 token 数。

### 9.2 dynamic per-tensor

条件：

```text
scale is None
use_per_token_if_dynamic == False
```

行为：

```text
分配 scale shape = [1]
调用 _C.dynamic_scaled_fp8_quant
```

位置：`_custom_ops.py:1945` 到 `_custom_ops.py:1947`

### 9.3 static

条件：

```text
scale is not None
```

行为：

```text
调用 _C.static_scaled_fp8_quant(output, input, scale, group_shape)
```

位置：`_custom_ops.py:1948` 到 `_custom_ops.py:1949`

### 9.4 scale shape 语义

函数注释里明确说明：

```text
0D 或 [1]：
  per-tensor scale。

1D：
  需要 group_shape 区分 per-channel / per-token。

2D：
  group scaling，例如 [M, N/128]。
```

位置：`_custom_ops.py:1903` 到 `_custom_ops.py:1920`

---

## 10. INT8 activation quantization 对照

INT8 的入口是：

```text
_custom_ops.scaled_int8_quant()
```

位置：`_custom_ops.py:2039`

它支持：

```text
static per-tensor INT8 activation quant；
dynamic per-token INT8 activation quant；
symmetric / asymmetric INT8。
```

### 10.1 static INT8

如果传入 `scale`：

```python
torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
```

位置：`_custom_ops.py:2060` 到 `_custom_ops.py:2066`

这表示：

```text
scale 外部给定；
当前 input 不需要动态统计 absmax。
```

### 10.2 dynamic per-token INT8

如果 `scale is None`：

```python
input_scales = torch.empty((input.numel() // input.shape[-1], 1), ...)
torch.ops._C.dynamic_scaled_int8_quant(output, input.contiguous(), input_scales, input_azp)
```

位置：`_custom_ops.py:2068` 到 `_custom_ops.py:2075`

所以 INT8 dynamic activation quant 默认是：

```text
每个 token / row 一个 scale。
```

### 10.3 INT8 scaled GEMM

Cutlass INT8 kernel 中：

```text
x_q, x_s, x_zp = ops.scaled_int8_quant(...)
```

然后：

```text
symmetric：
  ops.cutlass_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s)

asymmetric：
  ops.cutlass_scaled_mm_azp(..., azp=x_zp, azp_adj=azp_adj)
```

位置：`cutlass.py:121` 到 `cutlass.py:153`

这和 FP8 的主线一致：

```text
activation quant → activation scale → scaled GEMM
```

---

## 11. per-token / per-tensor / per-channel / per-token-group 怎么理解

### 11.1 per-tensor

```text
整个 activation 矩阵一个 scale。
```

形状通常是：

```text
[1]
```

优点：

```text
scale 少；
kernel 参数简单；
额外开销低。
```

缺点：

```text
一个异常大值会影响整个 tensor 的量化精度。
```

### 11.2 per-token

```text
每个 token / row 一个 scale。
```

如果 activation flatten 后是 `[M, N]`：

```text
scale shape = [M, 1]
```

优点：

```text
每个 token 独立适配动态范围；
精度通常优于 per-tensor；
适合 decode / prefill 混合场景。
```

缺点：

```text
每行都要做 absmax reduce；
scale 数量更多；
GEMM 需要支持 row-wise scale。
```

### 11.3 per-channel

```text
每个 channel 一个 scale。
```

在 vLLM 里更常见于 weight scale。

例如：

```text
weight 每个输出 channel 一个 scale；
GEMM 时按 channel 广播。
```

对于 activation dynamic quant，vLLM 的非 group dynamic 路径不支持 per-channel。

### 11.4 per-token-group / block

```text
每个 token 的 hidden 维再按 group 切分。
```

例如 hidden size 为 `N`，group size 为 128：

```text
activation shape = [M, N]
scale shape      = [M, N / 128]
```

优点：

```text
比 per-token 更细；
能减少某个 hidden group 的 outlier 对其他 group 的影响；
适合 block-scaled FP8 / DeepGEMM / DeepSeek 风格 W8A8。
```

缺点：

```text
scale 更多；
scale layout 更复杂；
kernel 必须原生支持 block scale。
```

---

## 12. per-token-group FP8 如何参与 forward

per-token-group 对应 `GroupShape(1, group_size)`。

在 `QuantFP8.forward_cuda()` 中：

```text
self.is_group_quant and not self.static
  → fp8_utils.per_token_group_quant_fp8(...)
```

位置：`input_quant_fp8.py:105` 到 `input_quant_fp8.py:115`

native fallback 的逻辑是：

```text
1. 把 x reshape 成 [-1, num_groups, group_size]；
2. 每个 group 内做 absmax；
3. scale = absmax / FP8_MAX；
4. x_grouped / scale；
5. clamp；
6. cast 到 FP8；
7. scale reshape 回 [tokens, num_groups]。
```

位置：`input_quant_fp8.py:233` 到 `input_quant_fp8.py:266`

简化链路：

```text
x: [tokens, hidden]
  → view [tokens, hidden/group_size, group_size]
  → absmax per group
  → scale [tokens, hidden/group_size]
  → quantized_x [tokens, hidden]
  → block-scaled GEMM
```

如果启用 DeepGEMM / UE8M0 scale，还可能走：

```text
per_token_group_quant_fp8_packed_for_deepgemm
```

位置：`input_quant_fp8.py:93` 到 `input_quant_fp8.py:103`

这类路径不只是“量化数值”，还涉及 scale 的物理布局、转置布局、TMA 对齐和 packed scale 表示。

---

## 13. scaled GEMM 如何消费 activation scale

以 Cutlass FP8 为例。

### 13.1 apply_weights 的输入输出

`FP8ScaledMMLinearKernel.apply_weights()` 负责把原始 activation 或 `QuantizedActivation` 转成 scaled GEMM 输入。

它最终调用：

```python
self.apply_scaled_mm(
    A=x_2d_q,
    B=w,
    out_dtype=out_dtype,
    As=x_s,
    Bs=w_s,
    bias=bias,
    output_shape=output_shape,
)
```

位置：`ScaledMMLinearKernel.py:162` 到 `ScaledMMLinearKernel.py:170`

这里：

```text
A：量化后的 activation；
B：量化后的 weight；
As：activation scale；
Bs：weight scale；
out_dtype：输出恢复到的 dtype，例如 fp16 / bf16。
```

### 13.2 Cutlass FP8 scaled mm

Cutlass FP8 的实现中：

```python
output = ops.cutlass_scaled_mm(
    A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
)
```

位置：`cutlass.py:265` 到 `cutlass.py:267`

也就是说 scaled GEMM 的语义是：

```text
output ≈ (A_fp8 * As) @ (B_fp8 * Bs)
```

但实际不会简单地先 dequant 成大 tensor 再 matmul，而是由 kernel 在 GEMM 内部按 scale 解释低精度输入。

### 13.3 输出是否需要 dequant

对 linear 来说，输出通常不是继续保持 FP8，而是回到模型主 dtype：

```text
out_dtype = orig_dtype 或配置指定 dtype
```

位置：`ScaledMMLinearKernel.py:156` 到 `ScaledMMLinearKernel.py:157`

所以主链路是：

```text
fp16/bf16 activation
  → FP8 activation + scale
  → scaled GEMM
  → fp16/bf16 hidden states
```

如果下游也需要量化，通常会在下一个 quantized linear 前再次量化，或者通过 fusion 直接产出 `QuantizedActivation`。

---

## 14. QuantizedActivation：避免重复量化的协议

`QuantizedActivation` 定义在：

```text
model_executor/layers/fusion/quant_activation.py
```

它表示：

```text
上游 fused kernel 已经把 activation 量化好了，
linear 层可以直接消费，不需要再 QuantFP8 一次。
```

字段：

```text
data：
  量化后的 activation。

scale：
  activation scale。

orig_dtype：
  原始 dtype，用于决定输出 dtype。

orig_shape：
  原始 shape，用于恢复 output shape。

quant_key：
  描述 data + scale 的量化格式。
```

位置：`quant_activation.py:18` 到 `quant_activation.py:35`

### 14.1 producer / consumer 如何对接

consumer 侧 linear kernel 暴露自己能吃的 activation quant key：

```python
kernel.input_quant_key()
```

再通过：

```python
expose_input_quant_key(layer, kernel)
```

挂到 layer 上。

位置：`quant_activation.py:38` 到 `quant_activation.py:52`

消费时：

```python
qa = as_quantized_activation(x, self.input_quant_key())
```

位置：`ScaledMMLinearKernel.py:145`

如果 key 匹配：

```text
直接使用 qa.data 和 qa.scale；
跳过 QuantFP8。
```

如果 key 不匹配：

```text
直接 assert 失败，避免错误的量化格式被静默使用。
```

位置：`quant_activation.py:55` 到 `quant_activation.py:71`

### 14.2 为什么重要

没有 `QuantizedActivation` 时：

```text
上游 activation op
  → 输出 fp16/bf16
  → linear 内部再 quant
```

有融合时：

```text
上游 fused activation+quant op
  → 输出 QuantizedActivation(data, scale)
  → linear 直接 scaled GEMM
```

这可以减少：

```text
1. 中间 fp16/bf16 tensor 写回；
2. 额外 quant kernel launch；
3. 重复 scale 计算；
4. 内存带宽压力。
```

---

## 15. Activation + quant fusion

vLLM 的 torch.compile fusion pass 会把一些常见模式融合成单个 kernel。

### 15.1 SiLU and Mul + quant

文件：`act_quant_fusion.py`

它匹配：

```text
silu_and_mul(input)
  → quant_fp8(...)
```

替换成：

```text
silu_and_mul_quant
或 silu_and_mul_per_block_quant
或 silu_and_mul_nvfp4_quant
```

关键映射：`act_quant_fusion.py:33` 到 `act_quant_fusion.py:45`

```text
kFp8StaticTensorSym
  → _C.silu_and_mul_quant

kFp8Dynamic128Sym / kFp8Dynamic64Sym
  → _C.silu_and_mul_per_block_quant

kNvfp4Dynamic
  → _C.silu_and_mul_nvfp4_quant
```

含义：

```text
MLP 中常见的 activation function + quant 可以合成一个 kernel，
直接产出低精度 activation 和 scale。
```

### 15.2 RMSNorm + quant

文件：`rms_quant_fusion.py`

它把：

```text
rms_norm(input)
  → quant_fp8(...)
```

或：

```text
fused_add_rms_norm(input, residual)
  → quant_fp8(...)
```

替换成：

```text
rms_norm_static_fp8_quant
fused_add_rms_norm_static_fp8_quant
rms_norm_dynamic_per_token_quant
rms_norm_per_block_quant
```

关键映射：`rms_quant_fusion.py:89` 到 `rms_quant_fusion.py:143`

这类 fusion 很重要，因为很多模型结构是：

```text
RMSNorm
  → Linear(QKV / MLP projection)
```

如果 linear 需要量化 activation，那么 RMSNorm 输出马上就会被 quantize。

融合后：

```text
RMSNorm + absmax + scale + quant
```

可以在一个 fused op 里完成。

### 15.3 Attention output + quant

文件：`attn_quant_fusion.py`

它处理 attention output 后接 quant 的场景。

典型思路是：

```text
attention output
  → static FP8 quant
```

可以把 quant scale 下沉到 attention op 的 output quant 参数中，减少 attention 后单独的 quant kernel。

---

## 16. torch.compile 与 CUDA graph 的关系

Activation quantization 和 CUDA graph 是两个层面的事情。

### 16.1 quantization 的语义层

Activation quantization 决定：

```text
1. 当前 x 是否要量化；
2. scale 是 static 还是 dynamic；
3. scale 粒度是什么；
4. 用哪个 quant kernel；
5. scaled GEMM 如何消费 scale。
```

这属于模型 forward 内部算子语义。

### 16.2 torch.compile / Inductor fusion 层

Fusion pass 决定：

```text
RMSNorm + quant 是否合并；
SiLU/Mul + quant 是否合并；
attention output + quant 是否合并；
mutating custom op 如何 functionalize；
```

例如 `act_quant_fusion.py` 和 `rms_quant_fusion.py` 中都使用：

```text
auto_functionalized(...)
```

让带 output 参数的 custom op 可以被 Inductor pattern replacement 处理。

### 16.3 CUDA graph 层

CUDA graph 主要负责：

```text
捕获 / replay 已确定形状和执行路径的 forward；
减少 CPU launch overhead；
配合 batch padding / cudagraph runtime mode。
```

它不改变 activation quantization 的数学语义。

可以理解为：

```text
activation quantization：
  这一步算什么。

torch.compile fusion：
  哪些算子合并成一个 kernel。

CUDA graph：
  已经确定的 kernel 序列如何 replay。
```

### 16.4 dynamic quant 和 CUDA graph 是否冲突

一般不冲突。

原因是 dynamic quant 的 scale 虽然每次 forward 数值不同，但：

```text
scale tensor 的 shape 通常由 batch/token padding 后的形状决定；
CUDA graph replay 允许 tensor 内容变化；
只要形状、内存地址、执行路径满足 capture 要求即可。
```

真正需要注意的是：

```text
1. dynamic per-token scale 的 shape 与 token 数有关；
2. vLLM 会通过 batch padding / cudagraph mode 管理形状；
3. 某些 padding 或 native fallback 可能生成额外 kernel；
4. compile fusion 会尽量减少独立 quant kernel。
```

`QuantFP8.forward_native()` 里也提到：

```text
padding 在 compilation 中可能产生额外 Triton kernel，
编译场景通常不使用 padding。
```

位置：`input_quant_fp8.py:223` 到 `input_quant_fp8.py:226`

---

## 17. online FP8 和 activation quant 的关系

Online FP8 指的是：

```text
checkpoint 里的 weight 不是 FP8，
加载时把 fp16/bf16 weight 量化成 FP8。
```

典型类：

```text
Fp8PerTensorOnlineLinearMethod
Fp8PerBlockOnlineLinearMethod
```

文件：`online/fp8.py`

### 17.1 online per-tensor FP8

`Fp8PerTensorOnlineLinearMethod` 加载 fp16/bf16 weight，然后在 `process_weights_after_loading()` 中：

```python
qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
```

位置：`online/fp8.py:157` 到 `online/fp8.py:162`

这一步是 weight 的 online quant，不是每次 forward 的 activation quant。

forward 时仍然会：

```text
activation_fp16/bf16
  → dynamic activation quant
  → scaled GEMM
```

### 17.2 online per-block FP8

`Fp8PerBlockOnlineLinearMethod` 使用：

```text
weight_quant_key：static block scale
activation_quant_key：dynamic per-token-group
```

位置：`online/fp8.py:203` 到 `online/fp8.py:216`

所以它的 forward 仍然属于：

```text
W8A8：weight FP8 + activation FP8
```

只是 weight 的 FP8 化发生在加载阶段，而不是 checkpoint 原生就是 FP8。

---

## 18. MoE 中 activation quantization 的差异

MoE FP8 路径也会设置 activation key。

`Fp8MoEMethod` 中：

```text
block_quant：
  weight_key = kFp8Static128BlockSym
  activation_key = kFp8Dynamic128Sym

非 block + static activation：
  activation_key = kFp8StaticTensorSym

非 block + dynamic activation：
  activation_key = kFp8DynamicTensorSym
```

位置：`fp8.py:514` 到 `fp8.py:524`

MoE 和普通 linear 的差别在于：

```text
1. activation 还要按 expert routing 分发；
2. grouped GEMM / fused MoE backend 会消费 quantized activation；
3. scale 可能要按 expert / token group 配合 kernel layout；
4. block FP8 MoE 更依赖 backend 对 per-token-group scale 的原生支持。
```

但主线仍然是：

```text
activation
  → quantized activation + activation scale
  → quantized expert weight + weight scale
  → fused MoE / grouped GEMM
```

---

## 19. dynamic quant 的额外开销在哪里

Dynamic activation quantization 的额外开销主要来自：

```text
1. absmax reduce：
   per-tensor 做一次全局 reduce；
   per-token 每行做 reduce；
   per-token-group 每行每组做 reduce。

2. scale tensor 写入：
   dynamic scale 需要写出给 GEMM 使用。

3. quantized activation 写入：
   x 需要转换成 FP8 / INT8 格式。

4. kernel launch：
   如果没有 fusion，quant 是 GEMM 前的独立 kernel。

5. scale layout 转换：
   block scale 可能要 column-major、TMA-aligned、packed UE8M0。
```

但是它也能节省：

```text
1. GEMM 输入带宽；
2. GEMM 计算资源；
3. weight bandwidth；
4. 某些 fused path 的中间 tensor 写回。
```

所以性能取决于：

```text
quant kernel 开销
  vs
scaled GEMM / fused kernel 的收益
```

常见优化方向：

```text
1. 用 per-token 替代 per-tensor 提升精度；
2. 用 per-token-group 提升 block FP8 精度；
3. 用 fusion 减少 quant kernel launch；
4. 用 CUDA graph 减少 launch overhead；
5. 用 DeepGEMM / Cutlass 原生支持 scale layout。
```

---

## 20. activation scale 在 forward 中的生命周期

以 dynamic per-token FP8 linear 为例：

```text
1. 输入 x 进入 LinearBase.forward()。
2. LinearBase 调 quant_method.apply()。
3. Fp8LinearMethod 把工作交给 fp8_linear.apply_weights()。
4. apply_weights 把 x flatten 成 [M, hidden]。
5. QuantFP8 看到 input_scale=None。
6. QuantFP8 调 ops.scaled_fp8_quant(..., use_per_token_if_dynamic=True)。
7. scaled_fp8_quant 分配 x_scale: [M, 1]。
8. dynamic_per_token_scaled_fp8_quant 计算每行 absmax 和 scale，并写出 quantized_x。
9. Cutlass scaled_mm 消费 quantized_x、x_scale、weight_fp8、weight_scale。
10. 输出恢复成模型主 dtype。
```

如果是 static activation FP8：

```text
1. layer.input_scale 从 checkpoint / 参数加载。
2. QuantFP8 看到 scale 不为 None。
3. static_scaled_fp8_quant 使用这个 scale 量化 x。
4. scaled_mm 使用同一个 input_scale。
```

如果是 per-token-group：

```text
1. QuantFP8 识别 group_shape = (1, group_size)。
2. per_token_group_quant_fp8 对每个 token/group 计算 scale。
3. 输出 scale shape = [M, hidden/group_size]。
4. block-scaled GEMM 使用 group scale。
```

---

## 21. 容易疑惑的点

### 21.1 activation quantization 是不是只在模型加载时发生？

不是。

权重量化通常在 checkpoint / 加载 / process_weights_after_loading 阶段发生。

activation quantization 发生在 forward 中，尤其 dynamic activation quant 每次 forward 都要根据当前输入重新计算 scale。

### 21.2 dynamic quantization 是不是一定 per-token？

不是。

vLLM 中常见 dynamic activation scale 有：

```text
per-tensor：
  一个 scale。

per-token：
  每行一个 scale。

per-token-group：
  每行每组一个 scale。
```

具体用哪个由 `QuantKey`、平台能力和 kernel backend 决定。

### 21.3 per-channel 和 per-token 有什么区别？

```text
per-token：
  按 activation 的 row / token 维度切。

per-channel：
  按 channel / output 维度切，常见于 weight。
```

activation dynamic quant 中，vLLM 非 group 路径主要支持 per-tensor 和 per-token。

### 21.4 dynamic scale 会不会影响 CUDA graph？

scale 的数值每次变不影响 CUDA graph replay。

需要稳定的是：

```text
shape、内存地址、执行路径、kernel 参数结构。
```

vLLM 通过 batch padding、cudagraph runtime mode 和 compile path 管理这些约束。

### 21.5 output 会保持 FP8 吗？

普通 linear 输出一般不会保持 FP8，而是回到模型主 dtype。

典型链路是：

```text
bf16/fp16 x
  → fp8 x + scale
  → scaled GEMM
  → bf16/fp16 output
```

下一个 quantized linear 前，如果需要，activation 会再次被量化。

### 21.6 fusion 后 linear 还会再 quant 一次吗？

如果上游 fused op 产出 `QuantizedActivation`，并且 `quant_key` 和 linear kernel 的 `input_quant_key()` 匹配，linear 会直接消费 `data` 和 `scale`，不会重复量化。

如果没有 `QuantizedActivation`，linear 内部仍然会调用 `QuantFP8`。

### 21.7 static activation scale 和 static weight scale 是一回事吗？

不是。

```text
weight_scale：
  描述 quantized weight 如何恢复数值。

input_scale / activation scale：
  描述 quantized activation 如何恢复数值。
```

scaled GEMM 同时需要：

```text
scale_a = activation scale
scale_b = weight scale
```

---

## 22. 调试时应该看哪些位置

### 22.1 看 linear 是否走量化

先看：

```text
linear.py:372
  LinearBase.forward()
```

然后看具体 layer 的：

```text
self.quant_method
```

如果是 FP8：

```text
Fp8LinearMethod.apply()
```

位置：`fp8.py:446`

### 22.2 看 activation key 怎么选

看：

```text
Fp8LinearMethod.__init__()
```

位置：`fp8.py:280` 到 `fp8.py:320`

重点变量：

```text
self.act_q_static
self.block_quant
self.activation_quant_key
self.weight_quant_key
```

### 22.3 看 scale 是否 dynamic

看 `QuantFP8` 调用时传入的 `scale`：

```text
scale is None
  → dynamic

scale is not None
  → static
```

核心位置：

```text
input_quant_fp8.py:117
_custom_ops.py:1939
```

### 22.4 看 per-token 还是 per-tensor

看：

```text
use_per_token_if_dynamic
```

位置：`input_quant_fp8.py:64`、`_custom_ops.py:1890`

如果为 True：

```text
scale shape = [M, 1]
_C.dynamic_per_token_scaled_fp8_quant
```

如果为 False：

```text
scale shape = [1]
_C.dynamic_scaled_fp8_quant
```

### 22.5 看是否被 fusion 了

看 compile pass：

```text
act_quant_fusion.py
rms_quant_fusion.py
attn_quant_fusion.py
```

如果图中出现：

```text
silu_and_mul_quant
rms_norm_dynamic_per_token_quant
rms_norm_per_block_quant
```

说明 activation op 和 quant 已经合并。

---

## 23. 总结

Activation / dynamic quantization 的完整心智模型是：

```text
LinearBase.forward(x)
  → quant_method.apply()
  → 根据 QuantKey / activation_scheme 选择 activation quant 粒度
  → 如果 x 不是 QuantizedActivation，则 QuantFP8 / scaled_int8_quant 现场量化
  → dynamic 路径根据当前 x 计算 scale
  → static 路径使用 layer.input_scale
  → scaled GEMM 同时消费 activation scale 和 weight scale
  → 输出回到模型主 dtype
```

按 scale 粒度看：

```text
per-tensor：
  一个 scale，开销低，精度相对弱。

per-token：
  每 token 一个 scale，精度更好，是 FP8 dynamic activation 常见路径。

per-token-group：
  每 token 每 hidden group 一个 scale，适合 block FP8 / DeepGEMM。

per-channel：
  更常用于 weight scale。
```

按执行优化看：

```text
没有 fusion：
  activation op → quant kernel → scaled GEMM。

有 fusion：
  RMSNorm / SiLU / Attention output 和 quant 合并，直接产出低精度 activation + scale。

CUDA graph：
  replay 已确定的 quant + GEMM kernel 序列，不改变 quant 语义。
```

一句话压缩：

```text
Activation quantization 把量化从“加载时的权重问题”扩展成“每次 forward 的运行时问题”；dynamic quantization 的核心就是在 forward 中根据当前 activation 计算 scale，并把 quantized activation 与 scale 一起交给 scaled GEMM 或 fused kernel。
```
