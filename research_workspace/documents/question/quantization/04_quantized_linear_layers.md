# 04. QuantizedLinear 如何替代普通 Linear？

源码位置：

- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/vllm/vllm/model_executor/layers/quantization/fp8.py`
- `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py`
- `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`

本问题关注：vLLM 里的 `ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear`、`MergedColumnParallelLinear`、`ReplicatedLinear` 等 Linear 层，如何在不改变模型 forward 语义的前提下，根据 `QuantizationConfig` 创建量化参数，并在 forward 中调用量化 kernel。

---

## 1. 一句话回答

vLLM 的 QuantizedLinear 通常不是把模型里的 Linear 替换成另一个独立类，而是在同一套 Linear layer 内部挂接一个 `quant_method`。

主链路是：

```text
模型构造 Linear layer
  → LinearBase 根据 quant_config 选择 quant_method
  → quant_method.create_weights() 注册 weight / qweight / scales / qzeros 等参数
  → weight_loader 按 TP / fused shard / packed dim 加载 checkpoint 权重
  → model_loader.process_weights_after_loading()
  → quant_method.process_weights_after_loading() 做 repack / transpose / scale 合并 / kernel 初始化
  → forward 调 quant_method.apply()
  → bias / all-gather / all-reduce 等并行后处理
```

所以：

```text
Linear layer 负责模型结构、张量并行切分和 bias/reduce/gather；
quant_method 负责权重表示、量化参数创建、后加载处理和 GEMM kernel 调用。
```

---

## 2. 最小心智模型

普通 Linear 的核心是：

```text
output = x @ weight.T + bias
```

vLLM 量化 Linear 的核心变成：

```text
output = quant_method.apply(layer, x, bias)
```

其中 `layer` 上可能不再只有一个 `weight`，而是会有不同量化格式需要的参数，例如：

```text
FP8：
  weight
  weight_scale / weight_scale_inv
  input_scale

AWQ / GPTQ：
  qweight
  qzeros
  scales
  g_idx  # GPTQ desc_act / act_order 场景
```

但调用 Linear 的上层模型通常不需要关心这些细节。

---

## 3. QuantizationConfig 和 QuantizeMethodBase 的边界

抽象接口在：`base_config.py`

### 3.1 QuantizationConfig 负责选择方法

`QuantizationConfig` 的关键接口是：

```python
def get_quant_method(
    self, layer: torch.nn.Module, prefix: str
) -> QuantizeMethodBase | None:
```

位置：`base_config.py:157` 到 `base_config.py:170`

它根据：

```text
layer 类型；
prefix 名称；
ignored_layers / modules_to_not_convert；
平台能力；
checkpoint 量化格式；
是否 lm_head_quantized；
是否 MoE / Attention / Embedding。
```

返回一个具体的 `QuantizeMethodBase` 子类。

### 3.2 QuantizeMethodBase 负责层内量化行为

`QuantizeMethodBase` 定义了两个必需方法：

```python
def create_weights(self, layer, *weight_args, **extra_weight_attrs)
def apply(self, layer, *args, **kwargs) -> torch.Tensor
```

位置：`base_config.py:27` 到 `base_config.py:41`

还有几个可选方法：

```python
def embedding(...)
def tie_weights(...)
def process_weights_after_loading(...)
```

位置：`base_config.py:43` 到 `base_config.py:62`

可以理解为：

```text
create_weights：
  定义这个 layer 需要哪些参数以及参数形状。

apply：
  forward 时实际执行 GEMM / dequant / quantized kernel。

process_weights_after_loading：
  checkpoint 权重加载后，做运行时需要的格式转换。

embedding / tie_weights：
  给 embedding / lm_head 这种非标准 Linear 调用路径使用。
```

---

## 4. LinearBase 如何选择 quant_method

入口在：`linear.py:228`

`LinearBase.__init__()` 中的关键逻辑是：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`linear.py:269` 到 `linear.py:274`

这说明：

```text
1. 没有 quant_config：使用 UnquantizedLinearMethod；
2. 有 quant_config：让具体 QuantizationConfig 根据 layer/prefix 返回方法；
3. Linear 层本身不硬编码 AWQ、GPTQ、FP8 等格式。
```

也就是说，`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` 的类名不变，真正变化的是：

```text
self.quant_method
```

---

## 5. LinearMethodBase 的接口

`LinearMethodBase` 继承自 `QuantizeMethodBase`。

位置：`linear.py:138`

核心接口：

```python
def create_weights(
    self,
    layer,
    input_size_per_partition,
    output_partition_sizes,
    input_size,
    output_size,
    params_dtype,
    **extra_weight_attrs,
)
```

位置：`linear.py:141` 到 `linear.py:165`

这些参数非常关键：

```text
input_size_per_partition：
  当前 TP rank 上的输入维度。

output_partition_sizes：
  当前 TP rank 上每个逻辑矩阵的输出宽度。
  QKV / MLP fused 场景会有多个逻辑分片。

input_size / output_size：
  全局逻辑矩阵大小。

params_dtype：
  权重或 scale 默认 dtype。

extra_weight_attrs：
  主要携带 weight_loader，以及参数加载所需属性。
```

`apply()` 的接口更直接：

```python
def apply(self, layer, x, bias=None) -> torch.Tensor
```

位置：`linear.py:167` 到 `linear.py:176`

---

## 6. 未量化路径也是 quant_method

未量化并不是 Linear 里写死 `torch.matmul`，而是走 `UnquantizedLinearMethod`。

### 6.1 create_weights

```python
weight = ModelWeightParameter(
    data=torch.empty(
        sum(output_partition_sizes),
        input_size_per_partition,
        dtype=params_dtype,
    ),
    input_dim=1,
    output_dim=0,
    weight_loader=weight_loader,
)
layer.register_parameter("weight", weight)
```

位置：`linear.py:182` 到 `linear.py:209`

它创建普通权重：

```text
weight shape = [sum(output_partition_sizes), input_size_per_partition]
```

### 6.2 apply

```python
return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

位置：`linear.py:217` 到 `linear.py:225`

因此未量化路径和量化路径的抽象完全一致：

```text
create_weights()
apply()
process_weights_after_loading()
```

这也是 vLLM 能把各种 Linear 层统一起来的原因。

---

## 7. ColumnParallelLinear 如何接入量化

`ColumnParallelLinear` 表示按输出维度切分权重。

位置：`linear.py:392`

### 7.1 初始化分片大小

```python
self.input_size_per_partition = input_size
self.output_size_per_partition = divide(output_size, self.tp_size)
self.output_partition_sizes = [self.output_size_per_partition]
```

位置：`linear.py:435` 到 `linear.py:439`

如果是 QKV 或 MergedColumn，则会把多个逻辑矩阵的输出大小分别记录下来：

```python
if hasattr(self, "output_sizes"):
    self.output_partition_sizes = [
        divide(output_size, self.tp_size) for output_size in self.output_sizes
    ]
```

位置：`linear.py:441` 到 `linear.py:444`

### 7.2 创建量化权重

```python
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=self.input_size_per_partition,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    params_dtype=self.params_dtype,
    weight_loader=(
        self.weight_loader_v2
        if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
        else self.weight_loader
    ),
)
```

位置：`linear.py:461` 到 `linear.py:473`

这里的关键点是：

```text
ColumnParallelLinear 不知道 qweight/scales/qzeros 怎么创建；
它只把分片信息和 weight_loader 交给 quant_method。
```

### 7.3 forward

```python
output_parallel = self.quant_method.apply(self, input_, bias)

if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

位置：`linear.py:548` 到 `linear.py:566`

所以 ColumnParallel 的 forward 可以拆成：

```text
本 rank 上的 quantized GEMM
  → 可选 all-gather 拼回完整输出
  → 可选返回 bias
```

---

## 8. RowParallelLinear 如何接入量化

`RowParallelLinear` 表示按输入维度切分权重。

位置：`linear.py:1491`

### 8.1 初始化分片大小

```python
self.input_size_per_partition = divide(input_size, self.tp_size)
self.output_size_per_partition = output_size
self.output_partition_sizes = [output_size]
```

位置：`linear.py:1543` 到 `linear.py:1548`

### 8.2 创建量化权重

```python
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=self.input_size_per_partition,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    params_dtype=self.params_dtype,
    weight_loader=(
        self.weight_loader_v2
        if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
        else self.weight_loader
    ),
)
```

位置：`linear.py:1565` 到 `linear.py:1577`

### 8.3 forward

RowParallel 先确保输入也是按 TP 切好的：

```python
if self.input_is_parallel:
    input_parallel = input_
else:
    split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
    input_parallel = split_input[self.tp_rank].contiguous()
```

位置：`linear.py:1628` 到 `linear.py:1639`

然后执行量化 GEMM：

```python
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
output_parallel = self.quant_method.apply(self, input_parallel, bias_)
```

位置：`linear.py:1640` 到 `linear.py:1644`

最后按需 all-reduce：

```python
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

位置：`linear.py:1646` 到 `linear.py:1650`

所以 RowParallel 的 forward 是：

```text
输入按 hidden dim 切分
  → 本 rank 上 quantized GEMM
  → 可选 all-reduce 汇总各 rank 部分结果
  → 可选返回 bias
```

---

## 9. ReplicatedLinear 如何接入量化

`ReplicatedLinear` 不做 TP 切分，每个 rank 持有完整权重。

位置：`linear.py:287`

创建权重时传入的是完整尺寸：

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

位置：`linear.py:338` 到 `linear.py:346`

forward 也最简单：

```python
output = self.quant_method.apply(self, x, bias)
```

位置：`linear.py:372` 到 `linear.py:383`

它可以理解为：

```text
没有 tensor parallel 切分的 QuantizedLinear。
```

---

## 10. QKVParallelLinear 的特殊性

`QKVParallelLinear` 继承自 `ColumnParallelLinear`。

位置：`linear.py:914`

它特殊在于一个物理 Linear 中包含三个逻辑矩阵：

```text
q_proj
k_proj
v_proj
```

### 10.1 output_sizes

初始化时会设置：

```python
self.output_sizes = [
    self.num_heads * self.head_size * tp_size,
    self.num_kv_heads * self.head_size * tp_size,
    self.num_kv_heads * self.v_head_size * tp_size,
]
```

位置：`linear.py:980` 到 `linear.py:984`

然后交给 `ColumnParallelLinear.__init__()`，最终变成：

```text
output_partition_sizes = [q_width_per_rank, k_width_per_rank, v_width_per_rank]
```

### 10.2 weight_loader 识别 q/k/v shard

QKV 的 checkpoint 可能是分开的：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

也可能已经 fused 成：

```text
qkv_proj.weight
```

因此 `QKVParallelLinear.weight_loader()` 支持 `loaded_shard_id`：

```text
"q"
"k"
"v"
None  # checkpoint 已经 fused
```

位置：`linear.py:1125` 到 `linear.py:1303`

量化参数加载时还要处理：

```text
packed_dim == output_dim：
  shard_size / shard_offset 要除以 packed_factor。

BlockQuantScaleParameter：
  scale 的 shard 要按 block size 调整。

bitsandbytes 4bit：
  offset/size 要按量化后张量形状换算。

KV heads 少于 TP size：
  K/V shard_rank 使用 num_kv_head_replicas 处理复制。
```

### 10.3 对量化方法的意义

对 `quant_method.create_weights()` 来说，QKV 只是收到多个 `output_partition_sizes`。

因此 FP8 / GPTQ / AWQ 等方法可以创建：

```text
一个 fused qweight；
或一组能表达 q/k/v logical width 的 scale；
或 per logical matrix 的 scale array。
```

---

## 11. MergedColumnParallelLinear 的特殊性

`MergedColumnParallelLinear` 也继承自 `ColumnParallelLinear`。

位置：`linear.py:577`

常见用途是 MLP 的 fused gate/up projection：

```text
gate_proj + up_proj
```

初始化时传入：

```python
self.output_sizes = output_sizes
```

位置：`linear.py:603` 到 `linear.py:633`

它的 `weight_loader()` 支持：

```text
loaded_shard_id = 0 / 1 / ...：
  checkpoint 分开存 gate_proj、up_proj。

loaded_shard_id = tuple：
  checkpoint 中多个连续 shard 一起加载。

loaded_shard_id = None：
  checkpoint 本身已经 fused。
```

位置：`linear.py:662` 到 `linear.py:805`

量化场景下，MergedColumn 的 loader 同样要处理：

```text
PackedColumnParameter / PackedvLLMParameter：按 packed_factor 调整 shard；
BlockQuantScaleParameter：按 weight_block_size 调整 scale shard；
PerTensorScaleParameter：把单个 scalar scale 填进 fused array；
bitsandbytes 4bit：按量化后形状换算 offset/size。
```

这就是为什么 `create_weights()` 需要 `output_partition_sizes` 而不是只给一个总输出大小。

---

## 12. 权重加载为什么要知道 input_dim / output_dim / packed_dim

vLLM 的量化参数通常不是裸 `torch.nn.Parameter`，而是带属性的参数类，例如：

```text
ModelWeightParameter
PackedvLLMParameter
PackedColumnParameter
RowvLLMParameter
GroupQuantScaleParameter
ChannelQuantScaleParameter
BlockQuantScaleParameter
PerTensorScaleParameter
```

这些参数上会带：

```text
input_dim：
  哪个维度对应输入 hidden dim。

output_dim：
  哪个维度对应输出 hidden dim。

packed_dim：
  哪个维度被 int4/int8 packing 压缩。

packed_factor：
  一个 int32 中打包多少个低 bit 元素。

weight_loader：
  checkpoint tensor 应该如何切给当前参数。
```

原因是：

```text
TP 切分发生在 input/output 维度；
量化 packing 也发生在 input/output 维度；
scale 的维度有时和 weight 不一致；
fused QKV / gate_up 还会多一层 logical shard。
```

所以加载量化权重时不能简单按原始 tensor copy，必须让 layer 的 `weight_loader` 根据这些属性切片。

---

## 13. process_weights_after_loading 在哪里调用

权重加载完成后，统一入口在：`model_loader/utils.py:100`

核心逻辑：

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
```

位置：`model_loader/utils.py:100` 到 `model_loader/utils.py:115`

这一步发生在：

```text
模型已构造；
checkpoint 权重已 copy 到参数；
真正推理前。
```

它的作用包括：

```text
- 把 checkpoint 格式转换成 kernel 需要的运行时格式；
- 对权重 transpose；
- 对 int4/int8 权重 repack；
- 合并或规范化 scales；
- 初始化 kernel workspace；
- 替换 layer 上的 Parameter。
```

这一步是理解量化 Linear 的关键：

```text
create_weights() 创建的是“可加载 checkpoint 的参数形态”；
process_weights_after_loading() 转成“适合运行时 kernel 的参数形态”。
```

---

## 14. FP8 Linear 例子

FP8 实现在：`fp8.py`

### 14.1 get_quant_method

`Fp8Config.get_quant_method()` 对 Linear 层的逻辑是：

```python
if isinstance(layer, LinearBase):
    if is_layer_skipped(...):
        return UnquantizedLinearMethod()
    if not self.is_checkpoint_fp8_serialized:
        return Fp8PerTensorOnlineLinearMethod()
    else:
        return Fp8LinearMethod(self)
```

位置：`fp8.py:179` 到 `fp8.py:200`

含义：

```text
FP8 checkpoint 已经序列化：
  加载 FP8 权重和 scale。

checkpoint 不是 FP8：
  使用 online quantization，在加载后把 FP16/BF16 权重量化成 FP8。

ignored layer：
  回退 UnquantizedLinearMethod。
```

### 14.2 create_weights

`Fp8LinearMethod.create_weights()` 会创建：

```text
weight：FP8 weight parameter；
weight_scale 或 weight_scale_inv：权重量化 scale；
input_scale：静态 activation scale 场景才有；
fp8_linear：根据 activation_quant_key / weight_quant_key 选择 kernel。
```

核心位置：`fp8.py:322` 到 `fp8.py:395`

其中 block-wise FP8 会使用：

```text
BlockQuantScaleParameter
weight_scale_inv
weight_block_size
```

per-tensor FP8 会使用：

```text
PerTensorScaleParameter
weight_scale
```

### 14.3 process_weights_after_loading

`Fp8LinearMethod.process_weights_after_loading()` 会处理：

```text
Marlin 路径：
  可能把 weight 转置成 kernel 需要的形态，再调用 fp8_linear.process_weights_after_loading。

非 block quant：
  对 fused module 的多 scale 做规整；
  必要时把 checkpoint 权重转成统一 tensor strategy；
  weight 转置；
  replace_parameter 更新 layer.weight / layer.weight_scale。

最后：
  调用 self.fp8_linear.process_weights_after_loading(layer)。
```

位置：`fp8.py:398` 到 `fp8.py:444`

### 14.4 apply

```python
return self.fp8_linear.apply_weights(layer, x, bias)
```

位置：`fp8.py:446` 到 `fp8.py:489`

特殊地，`VLLM_BATCH_INVARIANT` 下可能优先走直接 FP8 路径或 BF16 dequant fallback。

---

## 15. AWQ Linear 例子

AWQ 实现在：`auto_awq.py`

### 15.1 get_quant_method

`AutoAWQConfig.get_quant_method()` 支持：

```text
LinearBase；
ParallelLMHead 且 lm_head_quantized=True；
RoutedExperts。
```

位置：`auto_awq.py:284` 到 `auto_awq.py:357`

对 Linear 层，它会根据平台选择：

```text
XPU：AutoAWQXPULinearMethod；
CPU：AutoAWQMarlinLinearMethod；
CUDA + Marlin 可用：AutoAWQMarlinLinearMethod；
否则：AutoAWQLinearMethod。
```

位置：`auto_awq.py:287` 到 `auto_awq.py:332`

### 15.2 AWQ 参数形态

基础 AWQ 权重创建逻辑在 `BaseAWQLinearMethod.create_weights()`。

它创建：

```text
qweight：int32 packed weight；
qzeros：int32 packed zero points；
scales：group quant scales。
```

位置：`auto_awq.py:827` 到 `auto_awq.py:900`

参数形状大致是：

```text
qweight:
  [input_size_per_partition, output_size_per_partition / pack_factor]

qzeros:
  [num_groups, output_size_per_partition / pack_factor]

scales:
  [num_groups, output_size_per_partition]
```

### 15.3 AutoAWQLinearMethod apply

普通 AWQ Triton 路径：

```python
if num_tokens >= 256 or VLLM_BATCH_INVARIANT:
    out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
    out = torch.matmul(reshaped_x, out)
else:
    out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
```

位置：`auto_awq.py:915` 到 `auto_awq.py:939`

含义：

```text
token 较多时：先 dequant 再 matmul 可能更快；
token 较少时：直接 AWQ GEMM kernel；
最后按需加 bias 并 reshape 回原 batch 形状。
```

### 15.4 AutoAWQMarlinLinearMethod

Marlin 路径会先创建 AWQ checkpoint 形态的参数：

```text
qweight packed along output dim；
qzeros packed along output dim；
scales。
```

位置：`auto_awq.py:436` 到 `auto_awq.py:524`

加载后再转换格式：

```python
_convert_awq_to_standard_format(
    layer, "qweight", "qzeros", self.quant_config.quant_type.size_bits
)
self.kernel.process_weights_after_loading(layer)
```

位置：`auto_awq.py:525` 到 `auto_awq.py:533`

原因是：

```text
AWQ checkpoint 的 int4 packing 顺序和 kernel 期望格式不同；
AWQ checkpoint qweight 沿 output dim 打包；
MPLinearKernel / Marlin 更偏向 GPTQ-like 标准格式。
```

forward 时：

```python
return self.kernel.apply_weights(layer, x, bias)
```

位置：`auto_awq.py:535` 到 `auto_awq.py:541`

---

## 16. GPTQ Linear 例子

GPTQ 实现在：`auto_gptq.py`

### 16.1 get_quant_method

`AutoGPTQConfig.get_quant_method()` 会：

```text
RoutedExperts：选择 MoE GPTQ / WNA16 方法；
其他 Linear：通过 get_linear_quant_method(...) 选择 AutoGPTQLinearMethod 或跳过。
```

位置：`auto_gptq.py:240` 到 `auto_gptq.py:268`

GPTQ 还支持 dynamic 配置，可以按 regex 对模块做：

```text
正向匹配：覆盖 bits / group_size 等；
负向匹配：跳过某些模块量化。
```

位置：`auto_gptq.py:121` 到 `auto_gptq.py:145`

### 16.2 create_weights

`AutoGPTQLinearMethod.create_weights()` 创建：

```text
qweight：packed quantized weight；
g_idx：activation order / desc_act 用的 group index；
scales：group/channel scales；
qzeros：zero points。
```

位置：`auto_gptq.py:324` 到 `auto_gptq.py:451`

它还会根据 layer 是否 row parallel 决定 scale 是否在 TP rank 间复制：

```python
is_row_parallel = input_size != input_size_per_partition
...
if marlin_repeat_scales_on_all_ranks(...):
    scales_and_zp_input_dim = None
else:
    scales_and_zp_input_dim = 0
```

位置：`auto_gptq.py:334` 到 `auto_gptq.py:377`

### 16.3 kernel 选择

GPTQ 使用：

```python
mp_linear_kernel_config = MPLinearLayerConfig(...)
kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
```

位置：`auto_gptq.py:339` 到 `auto_gptq.py:352`

这说明 GPTQ 不是固定一个 kernel，而是根据：

```text
weight_type；
activation dtype；
group_size；
zero_points；
has_g_idx；
full / partition weight shape；
平台能力。
```

选择合适的 mixed-precision linear kernel。

### 16.4 process 和 apply

加载后处理：

```python
self.kernel.process_weights_after_loading(layer)
```

位置：`auto_gptq.py:453` 到 `auto_gptq.py:454`

forward：

```python
return self.kernel.apply_weights(layer, x, bias)
```

位置：`auto_gptq.py:456` 到 `auto_gptq.py:462`

---

## 17. Embedding 和 LM Head 是否支持量化

Embedding / LM Head 在：`vocab_parallel_embedding.py`

### 17.1 VocabParallelEmbedding

`VocabParallelEmbedding` 初始化时同样会问 quant_config：

```python
quant_method = None
if quant_config is not None:
    quant_method = quant_config.get_quant_method(self, prefix=prefix)
if quant_method is None:
    quant_method = UnquantizedEmbeddingMethod()
```

位置：`vocab_parallel_embedding.py:276` 到 `vocab_parallel_embedding.py:280`

但真正的 embedding layer 要求 quant method 实现 `embedding()`：

```python
if is_embedding_layer and not quant_method_implements_embedding:
    raise NotImplementedError(...)
```

位置：`vocab_parallel_embedding.py:282` 到 `vocab_parallel_embedding.py:293`

forward 时调用：

```python
output_parallel = self.quant_method.embedding(self, masked_input.long())
```

位置：`vocab_parallel_embedding.py:472` 到 `vocab_parallel_embedding.py:492`

所以：

```text
Embedding 是否能量化，取决于 quant_method 是否实现 embedding()。
```

### 17.2 ParallelLMHead

`ParallelLMHead` 继承 `VocabParallelEmbedding`。

位置：`vocab_parallel_embedding.py:503`

它自己的 forward 不用于普通模型 forward：

```python
def forward(self, input_):
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`vocab_parallel_embedding.py:562` 到 `vocab_parallel_embedding.py:564`

但它可以通过 quant_config 创建量化权重，并在采样 / logits 相关路径中被使用。

例如 AWQ / GPTQ 都有：

```text
lm_head_quantized=True 时，ParallelLMHead 也可以返回对应 quant_method。
```

AWQ 位置：`auto_awq.py:287` 到 `auto_awq.py:289`

GPTQ 配置中也有 `lm_head_quantized` 字段。

位置：`auto_gptq.py:112` 到 `auto_gptq.py:154`

---

## 18. forward 阶段的完整链路

以 Transformer MLP 或 Attention projection 为例，上层模型看到的还是普通调用：

```text
hidden_states
  → qkv_proj(hidden_states)
  → o_proj(attn_output)
  → gate_up_proj(hidden_states)
  → down_proj(activation)
```

进入任意 Linear layer 后，实际链路是：

```text
Linear.forward(x)
  → 准备 bias
  → quant_method.apply(layer, x, bias)
  → ColumnParallel：可选 all-gather
  → RowParallel：可选 all-reduce
  → skip_bias_add：可选返回 bias 给后续融合算子
```

也就是说，量化只替换：

```text
矩阵乘法内部实现和权重表示。
```

不替换：

```text
模型层级结构；
Attention / MLP 的调用关系；
Tensor Parallel 的通信语义；
bias add / skip_bias_add 协议。
```

---

## 19. 为什么量化不会改变 Tensor Parallel 语义

以 ColumnParallel 为例：

```text
原始：
  每个 rank 持有 A_i，计算 X @ A_i。

量化：
  每个 rank 持有 quantized(A_i)，计算 quantized_gemm(X, quantized(A_i))。

通信：
  gather_output=True 时仍然 all-gather。
```

以 RowParallel 为例：

```text
原始：
  每个 rank 持有 A_i，输入 X_i，计算 X_i @ A_i。

量化：
  每个 rank 持有 quantized(A_i)，计算 quantized_gemm(X_i, quantized(A_i))。

通信：
  reduce_results=True 时仍然 all-reduce。
```

因此 TP 语义在 Linear 类中保持不变；量化方法只负责本 rank 的局部 GEMM。

---

## 20. ignored_layers / modules_to_not_convert 如何回退未量化

很多量化配置允许跳过部分模块。

例如 FP8：

```python
if is_layer_skipped(...):
    return UnquantizedLinearMethod()
```

位置：`fp8.py:182` 到 `fp8.py:188`

AWQ：

```python
if is_layer_skipped(...):
    return UnquantizedLinearMethod()
```

位置：`auto_awq.py:290` 到 `auto_awq.py:296`

GPTQ 则可以通过 dynamic 负向匹配跳过模块。

位置：`auto_gptq.py:121` 到 `auto_gptq.py:145`

这说明同一个模型里可以混合：

```text
部分 Linear 量化；
部分 Linear 保持 BF16/FP16；
部分 MoE / lm_head 使用特殊量化；
Attention KV cache 使用另一套量化方法。
```

---

## 21. 容易疑惑的点

### 21.1 vLLM 里有一个叫 QuantizedLinear 的类吗？

主路径里通常不是。

vLLM 使用的是：

```text
ColumnParallelLinear / RowParallelLinear / QKVParallelLinear / MergedColumnParallelLinear / ReplicatedLinear
  + quant_method
```

因此“QuantizedLinear”更像一种运行状态，而不是单独替换成某个固定类。

### 21.2 create_weights 创建的是最终 kernel 格式吗？

不一定。

很多方法的 `create_weights()` 创建的是方便 checkpoint loader 加载的参数形态。

真正适合 kernel 的格式可能要在：

```text
process_weights_after_loading()
```

中 repack / transpose / replace。

### 21.3 为什么 QKV / gate_up 的 scale 有时是数组？

因为一个物理 Linear 里包含多个逻辑矩阵。

例如：

```text
QKV：q/k/v 三个逻辑矩阵；
MLP：gate/up 两个逻辑矩阵。
```

per-tensor scale 在 fused module 中可能需要保存成：

```text
[scale_q, scale_k, scale_v]
或
[scale_gate, scale_up]
```

### 21.4 为什么 packed 权重要调整 shard offset？

因为 int4/int8 权重会把多个元素打包进一个 int32。

如果原始输出维度按 `N` 切分，而 packed tensor 维度是 `N / pack_factor`，那么加载 shard 时必须同步除以 `pack_factor`。

相关逻辑出现在：

```text
QKVParallelLinear.weight_loader
MergedColumnParallelLinear.weight_loader
```

### 21.5 bias 是 quant_method 处理还是 Linear 处理？

两者都有参与。

Linear 决定传不传 bias：

```text
skip_bias_add=True：不传 bias，返回 output_bias；
RowParallel TP>1：只有 rank 0 在 GEMM 中加 bias，避免重复加；
```

quant_method.apply 接收 `bias`，具体 kernel 可以选择融合 bias add。

### 21.6 lm_head 一定量化吗？

不一定。

很多配置需要显式 `lm_head_quantized=True`，否则 lm_head 可能保持未量化。

### 21.7 Embedding 一定能量化吗？

不一定。

只有 quant method 实现了 `embedding()`，`VocabParallelEmbedding` 才能作为真正 embedding layer 使用该方法。

---

## 22. 总结

QuantizedLinear 的完整关系可以压缩成：

```text
QuantizationConfig
  → get_quant_method(layer, prefix)
  → LinearBase.quant_method
  → quant_method.create_weights(layer, shard sizes, weight_loader)
  → checkpoint weight_loader 按 TP/fused/packed 规则加载
  → quant_method.process_weights_after_loading(layer)
  → Linear.forward()
  → quant_method.apply(layer, x, bias)
  → TP gather/reduce + bias 协议
```

如果只记一句话：

```text
vLLM 的 QuantizedLinear 不是替换模型语义，而是让同一套并行 Linear 层通过 quant_method 替换权重表示、加载后处理和 GEMM kernel。
```

再压缩成最小心智模型：

```text
Linear 层管结构和并行；
QuantizationConfig 管选择；
quant_method 管参数和 kernel；
weight_loader 管 checkpoint 到当前 rank 的切片；
process_weights_after_loading 管运行时格式转换；
apply 管真正量化矩阵乘。
```
