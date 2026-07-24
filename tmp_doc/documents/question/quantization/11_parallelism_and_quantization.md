# 11. 并行场景下量化权重如何切分？

源码位置：

- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/parameter.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/model_executor/model_loader/weight_utils.py`
- `code/vllm/vllm/model_executor/models/utils.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/distributed/parallel_state.py`
- `code/vllm/vllm/distributed/utils.py`

本问题关注：tensor parallel、pipeline parallel、expert parallel、data parallel / decode-context / prefill-context 并行下，量化权重、scale、zero point、g_idx、input scale、block scale、packed layout 和 fused MoE expert 权重如何切分、加载和转换；为什么并行量化的难点不只是切 `qweight`，而是让 checkpoint 格式、vLLM 参数布局、TP/EP/PP 分片和 kernel runtime layout 全部对齐。

---

## 1. 一句话回答

并行场景下，vLLM 的量化权重切分不是由某个量化后端单独完成的，而是由 **并行层的 shape 语义 + vLLM Parameter 的 shard loader + quant_method 的参数创建 / post-load 转换 + 分布式并行组** 共同完成。

最小主链路是：

```text
ParallelConfig / distributed groups
  → 模型按 PP 切 layer 范围
  → LinearBase / RoutedExperts 根据 quant_config 选择 quant_method
  → Column / Row / QKV / MergedColumn / RoutedExperts 计算本 rank 逻辑 shape
  → quant_method.create_weights(...) 注册 qweight / scales / qzeros / g_idx / input_scale
  → weight_loader / weight_loader_v2 按 TP / fused shard / packed_factor / block size 加载 checkpoint
  → process_weights_after_loading(...) 做 repack / transpose / kernel layout 转换
  → forward 时 quant_method.apply(...) 调用 kernel
  → TP all-gather / all-reduce 或 EP expert_map / all2all 完成并行协作
```

所以可以把它理解成：

```text
PP 决定“当前 rank 有哪些层”；
TP 决定“当前 rank 拿矩阵哪一段”；
EP 决定“当前 rank 拿哪些 expert”；
量化参数类决定“低 bit packed 后怎么切”；
量化 method 决定“切好的参数如何变成 kernel 需要的格式”。
```

---

## 2. 最小心智模型

并行量化要同时对齐四个坐标系：

```text
1. checkpoint 坐标系：
   HF / GPTQ / AWQ / compressed-tensors / bitsandbytes 文件里权重怎么命名、怎么 pack。

2. vLLM layer 坐标系：
   QKV、gate_up_proj、w13/w2 等 fused module 在 vLLM 里怎么组织。

3. parallel rank 坐标系：
   TP rank / PP rank / EP rank / DP rank 当前负责哪一段权重或哪些 expert。

4. kernel runtime 坐标系：
   Marlin / Cutlass / Triton / FP8 scaled-mm / fused MoE kernel 需要什么 layout。
```

可以记成：

```text
checkpoint weight
  → vLLM fused layer name
  → local rank shard
  → packed tensor slice
  → post-load kernel layout
```

如果其中任意一步不一致，就会出现：

```text
shape mismatch
scale 数量不对
group_size 不整除
qzeros 对不上 qweight
Marlin tile 不对齐
MoE expert_map 找不到本地 expert
```

---

## 3. 并行组如何建立

并行组初始化在：

```python
def initialize_model_parallel(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    decode_context_model_parallel_size,
    backend,
)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1694`

注释里的布局顺序是：

```text
ExternalDP x DP x PP x PCP x TP
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1760`

其中：

```text
TP：tensor parallel，切单层矩阵；
PP：pipeline parallel，切 transformer layer 范围；
DP：data parallel，复制模型，分摊请求；
PCP / DCP：context parallel 相关维度；
EP：expert parallel，只对 MoE expert 分布有意义。
```

常用 group getter：

```text
get_tp_group()
get_pp_group()
get_dp_group()
get_ep_group()
get_pcp_group()
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1349` 到 `code/vllm/vllm/distributed/parallel_state.py:1405`

---

## 4. PP 先决定当前 rank 有哪些层

Pipeline Parallel 不直接改变某个 Linear 的 qweight 切法，它先决定：

```text
当前 rank 是否拥有这个 layer。
```

### 4.1 layer 范围计算

`ModelConfig.get_layers_start_end_indices()` 会根据 rank、TP size、PP size 计算 PP rank：

```python
pp_rank = (
    parallel_config.rank // parallel_config.tensor_parallel_size
) % parallel_config.pipeline_parallel_size
start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
```

位置：`code/vllm/vllm/config/model.py:1282` 到 `code/vllm/vllm/config/model.py:1295`

`get_pp_indices()` 会尽量均匀分配 hidden layers。

位置：`code/vllm/vllm/distributed/utils.py:109`

### 4.2 make_layers 如何处理缺失层

模型通常调用：

```python
make_layers(num_hidden_layers, layer_fn, prefix)
```

位置：`code/vllm/vllm/model_executor/models/utils.py:640`

它会：

```text
1. 计算 start_layer / end_layer；
2. start_layer 之前放 PPMissingLayer；
3. start_layer 到 end_layer 之间创建真实 layer；
4. end_layer 之后放 PPMissingLayer。
```

核心代码位置：`code/vllm/vllm/model_executor/models/utils.py:660` 到 `code/vllm/vllm/model_executor/models/utils.py:670`

这意味着：

```text
PP rank 只会为自己负责的层创建真实 Linear / Attention / MLP / MoE；
只有真实层才会创建 qweight / scales / qzeros；
missing layer 不会加载量化权重。
```

### 4.3 权重加载如何跳过 PP missing 参数

工具函数：

```python
is_pp_missing_parameter(name, model)
```

位置：`code/vllm/vllm/model_executor/models/utils.py:697`

它根据 `PPMissingLayer` / `StageMissingLayer` 判断某个参数是否属于当前 PP rank 缺失的层。

所以 PP 对量化的直接影响是：

```text
checkpoint 里有全模型量化权重，
但当前 PP rank 只实例化和加载自己 stage 的量化参数。
```

---

## 5. TP 是单层量化切分的核心

Tensor Parallel 直接影响 Linear 权重矩阵：

```text
ColumnParallelLinear：
  沿输出维切分 weight。

RowParallelLinear：
  沿输入维切分 weight。

QKVParallelLinear：
  Q/K/V fused，按 head 维度切分，KV head 可能复制。

MergedColumnParallelLinear：
  gate/up 或多个 column-parallel 逻辑矩阵 fused 后一起切分。
```

这些层都继承 `LinearBase`。

位置：`code/vllm/vllm/model_executor/layers/linear.py:228`

`LinearBase` 初始化时选择量化 method：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:269` 到 `code/vllm/vllm/model_executor/layers/linear.py:274`

然后每个并行层把自己的本地 shape 传给：

```python
self.quant_method.create_weights(...)
```

也就是说：

```text
TP 切分先体现在 create_weights 的参数里；
量化 method 根据这些 shape 注册本 rank 的量化参数。
```

---

## 6. ColumnParallelLinear 如何切量化权重

源码：`code/vllm/vllm/model_executor/layers/linear.py:394`

Column Parallel 的语义是：

```text
Y = X A
A 沿输出维切分：A = [A_1, A_2, ...]
每个 TP rank 持有一段 output columns。
```

初始化时：

```text
input_size_per_partition = input_size
output_size_per_partition = output_size / tp_size
output_partition_sizes = [output_size_per_partition]
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:434` 到 `code/vllm/vllm/model_executor/layers/linear.py:443`

创建量化权重时传入：

```python
self.quant_method.create_weights(
    input_size_per_partition=self.input_size_per_partition,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    ...
)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:461` 到 `code/vllm/vllm/model_executor/layers/linear.py:473`

### 6.1 ColumnParallel weight_loader

普通 loader 会根据 `output_dim` 从 checkpoint 里 narrow 当前 TP rank 的输出 shard：

```text
shard_size = param_data.shape[output_dim]
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:517` 到 `code/vllm/vllm/model_executor/layers/linear.py:538`

v2 loader 则委托给参数对象：

```python
param.load_column_parallel_weight(loaded_weight=loaded_weight)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:540` 到 `code/vllm/vllm/model_executor/layers/linear.py:546`

### 6.2 ColumnParallel forward

forward 时先做本地 GEMM：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
```

如果 `gather_output=True`，再 all-gather。

位置：`code/vllm/vllm/model_executor/layers/linear.py:548` 到 `code/vllm/vllm/model_executor/layers/linear.py:566`

所以量化 kernel 只处理本 rank 的输出 shard；跨 rank 合并仍由 ColumnParallelLinear 管。

---

## 7. RowParallelLinear 如何切量化权重

源码：`code/vllm/vllm/model_executor/layers/linear.py:1493`

Row Parallel 的语义是：

```text
Y = X A
A 沿输入维切分，X 也按输入维切；
每个 TP rank 计算局部结果，最后 all-reduce。
```

初始化时：

```text
input_size_per_partition = input_size / tp_size
output_size_per_partition = output_size
output_partition_sizes = [output_size]
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1543` 到 `code/vllm/vllm/model_executor/layers/linear.py:1548`

创建量化权重：

```python
self.quant_method.create_weights(
    input_size_per_partition=self.input_size_per_partition,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    ...
)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1565` 到 `code/vllm/vllm/model_executor/layers/linear.py:1577`

### 7.1 RowParallel weight_loader

普通 loader 根据 `input_dim` narrow 当前 TP rank 的输入 shard：

```text
shard_size = param_data.shape[input_dim]
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1597` 到 `code/vllm/vllm/model_executor/layers/linear.py:1617`

v2 loader 调用：

```python
param.load_row_parallel_weight(loaded_weight=loaded_weight)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1619` 到 `code/vllm/vllm/model_executor/layers/linear.py:1626`

### 7.2 RowParallel forward

forward 时：

```text
如果 input_is_parallel=False：
  先把 input 沿最后一维 split，取当前 tp_rank 的 input shard。

quant_method.apply(...)
  → 本地量化 GEMM。

如果 reduce_results=True：
  tensor_model_parallel_all_reduce(output_parallel)。
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1628` 到 `code/vllm/vllm/model_executor/layers/linear.py:1654`

---

## 8. MergedColumnParallelLinear 如何处理 fused 权重

源码：`code/vllm/vllm/model_executor/layers/linear.py:577`

它用于多个 column-parallel 线性层 fused 到一个矩阵的场景，例如：

```text
gate_up_proj = [gate_proj | up_proj]
```

关键字段：

```text
self.output_sizes：每个逻辑矩阵的全局输出维度；
self.output_partition_sizes：每个逻辑矩阵在当前 TP rank 的输出维度。
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:617` 到 `code/vllm/vllm/model_executor/layers/linear.py:633`

### 8.1 shard_id

加载时 `loaded_shard_id` 可以是：

```text
None：checkpoint 已经是 fused 权重；
int：单个逻辑 shard；
tuple[int, ...]：连续多个逻辑 shard。
```

校验逻辑位置：`code/vllm/vllm/model_executor/layers/linear.py:635`

### 8.2 量化特殊处理

MergedColumn 的 weight loader 会处理：

```text
BlockQuantScaleParameter：
  调整 block scale 的 shard offset / size。

packed_dim == output_dim：
  根据 packed_factor 调整 shard size / offset。

Marlin：
  调整 marlin tile size。

bitsandbytes 4bit：
  通过 adjust_bitsandbytes_4bit_shard 重新计算 packed offset。

PerTensorScaleParameter：
  scalar scale 加载到 fused array 的指定逻辑槽位。
```

对应位置：`code/vllm/vllm/model_executor/layers/linear.py:708` 到 `code/vllm/vllm/model_executor/layers/linear.py:805`

v2 loader 版本位置：`code/vllm/vllm/model_executor/layers/linear.py:847`

---

## 9. QKVParallelLinear 如何处理 Q/K/V fused 权重

源码：`code/vllm/vllm/model_executor/layers/linear.py:914`

QKVParallelLinear 把 attention 的 Q、K、V projection fused 成一个 column-parallel 矩阵。

关键逻辑：

```text
Q head 按 TP 切；
KV head 如果总数小于 TP size，则复制，保证每个 rank 至少有一个 KV head；
output_sizes = [q_proj, k_proj, v_proj] 的全局 fused 尺寸。
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:966` 到 `code/vllm/vllm/model_executor/layers/linear.py:984`

### 9.1 loaded_shard_id

QKV 的 `loaded_shard_id` 必须是：

```text
q
k
v
None
```

校验位置：`code/vllm/vllm/model_executor/layers/linear.py:999`

### 9.2 KV head replication

加载 q shard 时：

```text
shard_rank = tp_rank
```

加载 k / v shard 时：

```text
shard_rank = tp_rank // num_kv_head_replicas
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1278` 到 `code/vllm/vllm/model_executor/layers/linear.py:1286`

这解释了为什么 QKV 量化权重不能简单按输出维平均切：KV head 可能是复制而不是均分。

### 9.3 QKV 的量化特殊处理

和 MergedColumn 类似，QKV loader 也处理：

```text
block scale shard；
packed_dim / packed_factor；
Marlin shard；
bitsandbytes 4bit packed shard；
per-tensor scale 的 q/k/v 槽位加载。
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1167` 到 `code/vllm/vllm/model_executor/layers/linear.py:1303`

---

## 10. Parameter 类是真正的 shard 执行层

量化 method 创建参数时，通常不会直接用普通 `torch.nn.Parameter`，而是用 vLLM 自定义 Parameter。

源码：`code/vllm/vllm/model_executor/parameter.py`

核心类：

```text
BasevLLMParameter
ModelWeightParameter
RowvLLMParameter
PackedvLLMParameter
PackedColumnParameter
GroupQuantScaleParameter
ChannelQuantScaleParameter
PerTensorScaleParameter
BlockQuantScaleParameter
```

位置：`code/vllm/vllm/model_executor/parameter.py:18`

### 10.1 BasevLLMParameter

它保存：

```text
_weight_loader
tp_rank
tp_size
```

位置：`code/vllm/vllm/model_executor/parameter.py:32` 到 `code/vllm/vllm/model_executor/parameter.py:67`

也就是说 Parameter 本身知道当前 TP rank。

### 10.2 Column 参数切分

`_ColumnvLLMParameter.load_column_parallel_weight()`：

```text
shard_size = self.data.shape[output_dim]
loaded_weight = loaded_weight.narrow(output_dim, tp_rank * shard_size, shard_size)
copy_ 到本地参数
```

位置：`code/vllm/vllm/model_executor/parameter.py:148` 到 `code/vllm/vllm/model_executor/parameter.py:154`

### 10.3 Row 参数切分

`RowvLLMParameter.load_row_parallel_weight()`：

```text
shard_size = self.data.shape[input_dim]
loaded_weight = loaded_weight.narrow(input_dim, tp_rank * shard_size, shard_size)
copy_ 到本地参数
```

位置：`code/vllm/vllm/model_executor/parameter.py:220` 到 `code/vllm/vllm/model_executor/parameter.py:230`

### 10.4 Packed 参数切分

`PackedvLLMParameter` 和 `PackedColumnParameter` 保存：

```text
packed_factor
packed_dim
marlin_tile_size
```

位置：`code/vllm/vllm/model_executor/parameter.py:313` 和 `code/vllm/vllm/model_executor/parameter.py:353`

packed shard 调整：

```python
shard_size = round(shard_size // packed_factor)
shard_offset = round(shard_offset // packed_factor)
if marlin_tile_size is not None:
    shard_size *= marlin_tile_size
    shard_offset *= marlin_tile_size
```

位置：`code/vllm/vllm/model_executor/parameter.py:606` 到 `code/vllm/vllm/model_executor/parameter.py:618`

这就是 INT4 / INT8 packed 权重不能按原始浮点维度直接切的原因。

### 10.5 PerTensorScaleParameter

Per-tensor scale 在 fused QKV / gate_up 场景里通常是一个数组：

```text
QKV：3 个 scale 槽位；
MergedColumn：N 个逻辑矩阵 scale 槽位。
```

`PerTensorScaleParameter._load_into_shard_id()` 会按 shard_id 写入对应槽位。

位置：`code/vllm/vllm/model_executor/parameter.py:291` 到 `code/vllm/vllm/model_executor/parameter.py:310`

### 10.6 BlockQuantScaleParameter

Block scale 不是普通 row/column scale，而是二维 block scale。

对应类型：`code/vllm/vllm/model_executor/parameter.py:397`

Linear loader 里遇到它时会调用：

```python
adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:82`

逻辑是按 block_n 把输出 shard 映射到 scale shard。

---

## 11. GPTQ 与 TP / packed / scale 的关系

源码：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py`

### 11.1 GPTQ Linear 创建哪些参数

`AutoGPTQLinearMethod.create_weights()` 创建：

```text
qweight：PackedvLLMParameter
  shape = [input_size_per_partition / pack_factor, output_size_per_partition]
  packed_dim = input dim

g_idx：RowvLLMParameter
  shape = [input_size_per_partition]

scales：ChannelQuantScaleParameter 或 GroupQuantScaleParameter
qzeros：PackedColumnParameter 或 PackedvLLMParameter
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:324` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:443`

### 11.2 RowParallel 对 scale 的影响

GPTQ 会判断：

```python
is_row_parallel = input_size != input_size_per_partition
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:335`

然后通过：

```python
marlin_repeat_scales_on_all_ranks(desc_act, group_size, is_row_parallel)
```

决定 scale / zp 的 input_dim：

```text
scales_and_zp_input_dim = None
  → scale 在每个 rank 重复加载。

scales_and_zp_input_dim = 0
  → scale 按 input group 维度切分。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:364` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:376`

这说明 GPTQ 的 scale 切法不只取决于矩阵是 Row 还是 Column，还取决于 `desc_act` 和 `group_size`。

### 11.3 GPTQ kernel config

GPTQ 会构造：

```python
MPLinearLayerConfig(
    full_weight_shape=(input_size, output_size),
    partition_weight_shape=(input_size_per_partition, output_size_per_partition),
    weight_type=quant_type,
    group_size=group_size,
    zero_points=False,
    has_g_idx=desc_act,
)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:339` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:350`

然后 `choose_mp_linear_kernel()` 根据这个本地 partition 选择 Marlin / Machete / Cutlass / Triton / CPU 等 kernel。

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:352`

---

## 12. AWQ 与 TP / packed order 的关系

源码：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py`

### 12.1 AWQ Linear 参数

`BaseAWQLinearMethod.create_weights()` 创建：

```text
qweight：PackedvLLMParameter
  shape = [input_size_per_partition, output_size_per_partition / pack_factor]
  packed_dim = output dim

qzeros：PackedvLLMParameter
  shape = [num_groups, output_size_per_partition / pack_factor]

scales：GroupQuantScaleParameter
  shape = [num_groups, output_size_per_partition]
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:821` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:900`

### 12.2 AWQ 的对齐检查

AWQ 明确要求：

```text
input_size_per_partition % group_size == 0
output_size_per_partition % pack_factor == 0
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:843` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:856`

这意味着 TP size 过大时，单 rank 上的 input/output partition 可能不满足 group / pack 对齐。

### 12.3 AWQ Marlin post-load 转换

AWQ checkpoint 的 packing 顺序特殊：

```text
standard order：0,1,2,3,4,5,6,7
AWQ order：0,4,1,5,2,6,3,7
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:72` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:76`

Marlin 路径在加载后调用：

```python
_convert_awq_to_standard_format(layer, "qweight", "qzeros", size_bits)
self.kernel.process_weights_after_loading(layer)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:525` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:533`

这一步会把：

```text
AWQ packed along output dim
  → 标准 bit order
  → GPTQ-like packed along input dim
  → kernel layout
```

所以 AWQ 的 TP 切分必须和 post-load repack 一起理解。

---

## 13. FP8 与 per-tensor / block scale 的关系

源码：`code/vllm/vllm/model_executor/layers/quantization/fp8.py`

### 13.1 FP8 Linear 参数

`Fp8LinearMethod.create_weights()` 创建：

```text
weight：FP8 weight parameter
weight_scale：per-tensor scale
weight_scale_inv：block quant scale
input_scale：static activation scale
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:322` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:386`

### 13.2 per-tensor FP8 scale

非 block quant 时：

```python
create_fp8_scale_parameter(
    PerTensorScaleParameter,
    output_partition_sizes,
    input_size_per_partition,
    None,
    weight_loader,
)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:358` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:366`

这让 fused QKV / MLP 的多个逻辑矩阵 scale 能正确映射到本地 shard。

### 13.3 block-wise FP8 scale

block quant 时：

```python
create_fp8_scale_parameter(
    BlockQuantScaleParameter,
    output_partition_sizes,
    input_size_per_partition,
    weight_block_size,
    weight_loader,
)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:370` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:379`

并且会调用：

```python
validate_fp8_block_shape(...)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:340` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:350`

这说明 block-wise FP8 对 TP 后的局部 shape 有额外 block 对齐要求。

### 13.4 FP8 post-load

加载后处理会根据 kernel 和 block quant 做：

```text
Marlin：可能转置 weight 并调用 Marlin post-load；
非 serialized FP8：加载后在线量化；
fused module：合并 logical shard scale；
static input scale：整理 input_scale；
block quant：保留 block scale layout。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:398` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:444`

---

## 14. bitsandbytes 与并行切分

源码：`code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py`

### 14.1 Linear 参数

8bit 路径：

```text
Int8Params
shape = [sum(output_partition_sizes), input_size_per_partition]
```

4bit 路径：

```text
BitsAndBytesWeightParameter
shape = [total_size / quant_ratio, 1]
dtype = uint8
```

位置：`code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:225` 到 `code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:269`

### 14.2 BNB 4bit 的特殊 shard

因为 BNB 4bit 是 flat packed tensor，不能按普通二维矩阵直接 narrow。

vLLM 用：

```python
adjust_bitsandbytes_4bit_shard(param, shard_offsets, loaded_shard_id)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:94`

它根据原始 logical offset 和 packed 后总长度，重新计算：

```text
quantized_offset
quantized_size
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:104` 到 `code/vllm/vllm/model_executor/layers/linear.py:107`

### 14.3 BNB forward

8bit 调：

```text
bitsandbytes.matmul
```

4bit 调：

```text
bitsandbytes.matmul_4bit
```

位置：`code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:291` 和 `code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:361`

所以 BNB 的量化切分更多依赖 BNB quant_state / offset，而不是 vLLM 的通用 packed int32 layout。

---

## 15. compressed-tensors 与并行切分

源码：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`

### 15.1 scheme 决定参数布局

`CompressedTensorsConfig` 会根据 `config_groups` 选择 scheme。

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:80`

Linear method 本身很薄：

```text
create_weights(...)
  → layer.scheme.create_weights(...)

process_weights_after_loading(...)
  → layer.scheme.process_weights_after_loading(layer)

apply(...)
  → layer.scheme.apply_weights(layer, x, bias)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:910` 到 `code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:958`

这意味着 compressed-tensors 的并行切分规则主要由具体 scheme 决定。

### 15.2 supported scheme 与并行的关系

compressed-tensors 可以选择：

```text
WNA16：INT4/INT8 weight-only，A16 activation；
W8A16Fp8：FP8 weight-only；
W8A8Fp8：FP8 weight + activation；
W8A8Int8：INT8 weight + activation；
W4A8Int / W4A8Fp8；
MXFP4 / MXFP8；
WNA8O8Int；
Embedding WNA16；
MoE compressed-tensors method。
```

scheme 选择位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:692` 到 `code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:813`

### 15.3 KV cache scale 的 TP-aware 加载

compressed-tensors 还支持 KV cache scale。

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:961`

当 strategy 是 `ATTN_HEAD` 时：

```text
q_scale：先把 Q heads reduce 到 KV head 粒度；
k/v_scale：如果 KV heads 均分给 TP rank，就 narrow；
如果 KV heads 少于 TP size，就按 replica 关系选择 shard_rank。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:1053` 到 `code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:1099`

这和 QKVParallelLinear 里的 KV head replication 是同一个问题：KV head 数可能小于 TP size。

---

## 16. MoE 并行下量化权重如何组织

MoE 不是普通 Linear 层，而是：

```text
FusedMoE
  → Router
  → RoutedExperts
  → MoERunner
```

入口：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:103`

### 16.1 FusedMoEParallelConfig

`make_parallel_config()` 会把：

```text
tp_size
dp_size
pcp_size
is_sequence_parallel
parallel_config
```

转换成 `FusedMoEParallelConfig`。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:43`

FusedMoE 初始化时调用：

```python
moe_parallel_config = make_parallel_config(...)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:216`

### 16.2 ExpertMapManager 决定本地 expert

`ExpertMapManager` 负责 EP 下 expert 的 global → local 映射。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:152`

核心函数：

```python
determine_expert_map(ep_size, ep_rank, global_num_experts, ...)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:22`

它返回：

```text
local_num_experts：当前 EP rank 持有多少 expert；
expert_map：global expert id → local expert id，不在本 rank 的是 -1；
expert_mask：部分 AITER kernel 使用。
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:45` 到 `code/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:58`

如果 `ep_size == 1`：

```text
所有 expert 都在本 rank，expert_map=None。
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:62` 到 `code/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:64`

---

## 17. RoutedExperts 如何创建量化 expert 参数

源码：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:44`

初始化时：

```python
self.quant_method = self._get_quant_method(
    self.layer_name,
    self.quant_config,
    self.moe_config,
)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:114`

然后量化方法可以调整 hidden / intermediate size：

```python
self.hidden_size, self.intermediate_size_per_partition = (
    self.quant_method.maybe_roundup_sizes(...)
)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:122` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:133`

这一步对 MXFP4、block quant、某些 kernel tile 对齐很重要。

### 17.1 create_weights 参数

RoutedExperts 传给 quant_method 的关键参数：

```text
num_experts = moe_config.num_local_experts
hidden_size
unpadded_hidden_size
intermediate_size_per_partition
params_dtype
weight_loader = self.weight_loader
global_num_experts
intermediate_size_full（部分 WNA16 / act-order 需要）
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:150` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:168`

这说明 MoE 量化参数的第一维通常是：

```text
本 rank 的 local_num_experts
```

而不是 global_num_experts。

### 17.2 EPLB 支持检查

如果启用 EPLB：

```python
if enable_eplb and not quant_method.supports_eplb:
    raise NotImplementedError
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:135` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:148`

这说明不是所有量化 MoE kernel 都支持 expert load balancing。

---

## 18. MoE weight_loader 如何按 expert 和 shard 加载

`RoutedExperts.weight_loader()` 是 MoE expert 权重加载的核心。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:571`

它接收：

```text
param
loaded_weight
weight_name
shard_id：w1 / w2 / w3
expert_id
return_success
```

MoE 的 shard 语义：

```text
w1：gate_proj
w3：up_proj
w2：down_proj
w13：w1 + w3 fused 后的本地参数
```

### 18.1 global expert 到 local expert

加载时先做：

```python
expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:592` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:603`

如果这个 expert 不属于当前 rank：

```text
expert_id == -1
  → 跳过加载
```

这就是 EP 下量化 expert 权重只加载本地 expert 的核心。

### 18.2 shard_dim

MoE 默认 shard 维度：

```text
w1：0
gate / up 的 intermediate 维度

w3：0
up 的 intermediate 维度

w2：1
down_proj 的 intermediate 输入维度
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:628` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:672`

如果参数是 transposed，则 shard_dim 会翻转。

### 18.3 scale / zero point / g_idx 的加载

MoE loader 会根据 weight_name 和 param.quant_method 分流：

```text
input_scale：单值或 global scale；
g_idx：按 w1/w2/w3 的规则加载；
scale / zero / offset：按 TENSOR / CHANNEL / GROUP / BLOCK scale 规则加载；
weight：按模型权重或 group scale 规则加载；
bitsandbytes 4bit：特殊 flat packed 加载。
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:633` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:851`

---

## 19. MoE load_weights 和 expert_mapping

如果模型用 `RoutedExperts.load_weights()`，入口是：

```python
def load_weights(self, weights)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:853`

它依赖 `expert_mapping`：

```text
(param_name, weight_name, expert_id, shard_id)
```

生成 helper：

```python
make_expert_params_mapping(...)
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:905`

### 19.1 fused 3D expert 权重

如果 checkpoint 里的 loaded_weight 是 3D tensor：

```text
表示多个 experts 或 fused experts；
w1/w3 可能需要 chunk(2, dim=1) 拆 gate/up；
然后逐 expert 调 weight_loader。
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:869` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:895`

### 19.2 redundant experts / EPLB

`make_expert_params_mapping()` 会处理 physical expert id 到 logical expert id 的映射。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:937` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:974`

这保证 EPLB / redundant expert 场景下 checkpoint 里的 logical expert 权重能加载到当前物理 expert 位置。

---

## 20. GPTQ / AWQ MoE 的并行量化

### 20.1 GPTQ MoE

`AutoGPTQMoEMethod` 创建：

```text
w13_qweight / w2_qweight
w13_scales / w2_scales
w13_qzeros / w2_qzeros
w13_g_idx / w2_g_idx
w13_g_idx_sort_indices / w2_g_idx_sort_indices
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:492` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:650`

加载后调用：

```python
convert_to_wna16_moe_kernel_format(...)
make_wna16_moe_kernel(...)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:651` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:750`

### 20.2 AWQ MoE

`AutoAWQMoEMethod` 创建：

```text
w13_qweight / w2_qweight
w13_scales / w2_scales
w13_qzeros / w2_qzeros
workspace
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:562` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:661`

post-load 也会转换到 WNA16 MoE kernel 格式。

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:663` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:741`

### 20.3 MoeWNA16 fallback

当 GPTQ / AWQ MoE 不满足 Marlin support 时，会 fallback 到 `MoeWNA16Config`。

GPTQ fallback：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:243`

AWQ fallback：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:342`

`MoeWNA16Method` 会根据 group_size 调整：

```text
如果 hidden_size 或 intermediate_size_per_partition 不能整除 group_size，
就不断把 group_size 除以 2，直到满足要求。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/moe_wna16.py:216` 到 `code/vllm/vllm/model_executor/layers/quantization/moe_wna16.py:224`

---

## 21. FusedMoEQuantConfig 如何表达量化参数

MoE 的 modular kernel 使用：

```python
FusedMoEQuantConfig
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/config.py:214`

它包含四个描述：

```text
a1：第一个 GEMM 的 activation quant；
a2：第二个 GEMM 的 activation quant；
w1：w13 / gate_up 的 weight quant；
w2：down_proj 的 weight quant。
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/config.py:248` 到 `code/vllm/vllm/model_executor/layers/fused_moe/config.py:252`

每个 `FusedMoEQuantDesc` 可描述：

```text
dtype
GroupShape
scale
alpha_or_gscale
zero point
bias
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/config.py:174` 到 `code/vllm/vllm/model_executor/layers/fused_moe/config.py:208`

这解释了为什么 MoE 量化的 scale / zero point 不只是普通 Linear 的 `scales`：MoE kernel 需要分别知道两个 GEMM 的 activation / weight 量化信息。

---

## 22. DP 对量化权重的影响

Data Parallel 通常不切单层权重。

它的语义是：

```text
每个 DP replica 拥有一份完整模型分片；
在这个 replica 内部仍可能有 TP / PP / EP；
量化权重在不同 DP rank 上复制；
请求和 batch 在 DP 维度上分摊。
```

并行组布局里 DP 在 PP/TP 外侧。

位置：`code/vllm/vllm/distributed/parallel_state.py:1760` 到 `code/vllm/vllm/distributed/parallel_state.py:1775`

因此文档中讨论“量化权重如何切分”时，真正影响权重 shard 的通常是：

```text
TP：切矩阵；
PP：切层；
EP：切 expert；
DP：复制这些 shard。
```

---

## 23. PCP / context parallel 对量化权重的影响

PCP / DCP 主要影响上下文或 token 执行分组，不是 weight shard 的核心维度。

在 `FusedMoE` 中，`make_parallel_config()` 会把 `pcp_size` 放入 `FusedMoEParallelConfig`。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/layer.py:43` 到 `code/vllm/vllm/model_executor/layers/fused_moe/layer.py:69`

但对单个 Linear 的量化参数创建来说，关键仍是：

```text
input_size_per_partition
output_partition_sizes
tp_rank / tp_size
```

也就是说：

```text
PCP 影响执行组织；
TP/PP/EP 才直接决定量化权重的存放位置。
```

---

## 24. lm_head / embedding 的特殊性

并不是所有量化参数都来自普通 Linear。

compressed-tensors 会对：

```text
ParallelLMHead
VocabParallelEmbedding
```

做特殊处理。

位置：`code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:178` 到 `code/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:203`

其中：

```text
ParallelLMHead 被当作 linear 类路径处理；
真正 embedding lookup 只支持 weight-only INT group/channel WNA16。
```

PP 场景下，embedding / lm_head 常出现在首尾 stage，因此要结合 `PPMissingLayer` 判断当前 rank 是否拥有这些参数。

---

## 25. group size / pack factor / block size 的约束

并行量化最容易出错的是三个对齐：

### 25.1 group_size

per-group quant 要求局部输入维度能按 group size 分组：

```text
input_size_per_partition % group_size == 0
```

AWQ 显式检查：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:843`

MoE WNA16 会动态缩小 group_size：`code/vllm/vllm/model_executor/layers/quantization/moe_wna16.py:216`

### 25.2 pack_factor

INT4 / INT8 packed 权重要求局部 shard 能被 pack factor 对齐：

```text
INT4 in int32：pack_factor = 8
INT8 in int32：pack_factor = 4
```

GPTQ：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:151`

AWQ：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:192`

Packed 参数调整：`code/vllm/vllm/model_executor/parameter.py:606`

### 25.3 block size

FP8 block quant、MXFP4/NVFP4 等 block scheme 要求 block_n / block_k 对齐。

FP8 block shape 校验：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:340`

MoE FP8 block quant 也会检查 intermediate size 与 block size：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:550` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:575`

---

## 26. process_weights_after_loading 是并行量化的收口点

加载 checkpoint 后，vLLM 会遍历模块：

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(module)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:100` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:113`

它通常做：

```text
1. 把 checkpoint shard 转成 kernel shard；
2. 把 AWQ packed order 转成标准 order；
3. 把 FP8 fused scale 合并或转置；
4. 把 MoE w13/w2 转成 fused MoE kernel layout；
5. 创建 Marlin / Cutlass / Triton workspace；
6. 替换 layer 上的参数对象。
```

所以：

```text
weight_loader 解决“加载到哪”；
process_weights_after_loading 解决“运行时怎么放”。
```

---

## 27. forward 时并行和量化如何配合

### 27.1 ColumnParallelLinear

```text
quant_method.apply()
  → 本地 output shard
  → optional all-gather
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:548` 到 `code/vllm/vllm/model_executor/layers/linear.py:566`

### 27.2 RowParallelLinear

```text
input split
  → quant_method.apply()
  → 本地 partial output
  → optional all-reduce
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1628` 到 `code/vllm/vllm/model_executor/layers/linear.py:1654`

### 27.3 RoutedExperts

```text
router 选 topk expert
  → expert_map 把 global expert 映射到 local expert
  → quant_method.apply() / apply_monolithic()
  → fused MoE kernel 使用本地 w13/w2/qweight/scales
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:1053` 到 `code/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:1120`

---

## 28. 容易混淆的点

### 28.1 PP 会不会切单个 qweight？

通常不会。

PP 切的是 layer 范围：

```text
当前 PP rank 有这个 layer：
  再按 TP / EP 创建和加载这个 layer 的量化权重。

当前 PP rank 没有这个 layer：
  PPMissingLayer，占位，不加载该层量化参数。
```

### 28.2 TP 是不是只切 qweight？

不是。

TP 还要切或复制：

```text
scales
qzeros
g_idx
input_scale
weight_scale_inv
block scales
fused module 的 per-tensor scale array
```

### 28.3 scale 一定和 qweight 同维度切吗？

不是。

例如：

```text
per-tensor scale：可能按 logical shard_id 写槽位；
per-group scale：按 input group 切；
per-channel scale：按 output channel 切；
block scale：按 block_n / block_k 映射；
GPTQ desc_act：某些 RowParallel 场景会复制 scale 到所有 rank。
```

### 28.4 QKV 的 K/V 为什么不是简单 TP 切？

因为 KV head 数可能小于 TP size。

这时每个 rank 至少要有一个 KV head，因此 K/V head 会复制。

位置：`code/vllm/vllm/config/model.py:1265` 到 `code/vllm/vllm/config/model.py:1270`

QKV loader 中 K/V 使用：

```text
shard_rank = tp_rank // num_kv_head_replicas
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:1281` 到 `code/vllm/vllm/model_executor/layers/linear.py:1286`

### 28.5 EP 是不是等价于 TP？

不是。

TP 切一个矩阵的维度；EP 切 expert 集合。

```text
TP：每个 rank 拿同一个矩阵的一段；
EP：每个 rank 拿不同 expert 的完整或 TP-sharded 权重。
```

MoE 场景下通常是：

```text
EP 决定 local expert；
TP 决定每个 local expert 内部 w13/w2 的矩阵 shard。
```

### 28.6 DP 会不会切量化权重？

通常不会。

DP 复制模型 shard，每个 DP replica 内部再有自己的 TP / PP / EP 切分。

### 28.7 为什么 post-load 后参数还会变？

因为 checkpoint layout 不一定等于 kernel layout。

典型例子：

```text
AWQ checkpoint packed order
  → 标准 packed order
  → Marlin layout

GPTQ / AWQ MoE
  → w13/w2 qweight/scales/qzeros
  → WNA16 fused MoE kernel format

FP8
  → checkpoint FP8 weight/scale
  → transpose / merge scale / block scale layout
  → scaled-mm kernel format
```

---

## 29. 调试并行量化问题看哪里

如果遇到并行量化加载失败，可以按这个顺序看：

```text
1. 当前 PP rank 是否应该拥有这个 layer？
   models/utils.py: make_layers / PPMissingLayer

2. 这个 layer 是 Column、Row、QKV 还是 MergedColumn？
   linear.py 中 input_size_per_partition / output_partition_sizes

3. quant_method.create_weights 创建的参数 shape 是什么？
   auto_gptq.py / auto_awq.py / fp8.py / compressed_tensors scheme

4. 参数是否 packed？packed_dim / packed_factor 是什么？
   parameter.py: PackedvLLMParameter / PackedColumnParameter

5. scale 类型是什么？
   PerTensor / Channel / Group / Block

6. weight_loader 是普通 loader 还是 weight_loader_v2？
   linear.py: WEIGHT_LOADER_V2_SUPPORTED

7. process_weights_after_loading 是否改变了 layout？
   model_loader/utils.py + 具体 quant method

8. MoE 是否有 EP / expert_map / EPLB？
   fused_moe/expert_map_manager.py + routed_experts.py
```

---

## 30. 总结

并行量化的完整链路可以压缩成：

```text
PP：
  get_pp_indices / make_layers
  → 决定当前 rank 有哪些真实 layer

TP：
  Column / Row / QKV / MergedColumn
  → 决定每个 Linear 的本地 input/output shard

Parameter：
  Packed / Scale / Block / PerTensor 参数类
  → 决定 qweight、scale、qzeros、g_idx 如何 narrow / 写槽位

QuantMethod：
  GPTQ / AWQ / FP8 / BNB / compressed-tensors
  → 创建低 bit 参数，并在 post-load 转成 kernel layout

EP：
  ExpertMapManager / RoutedExperts
  → 决定本 rank 持有哪些 expert，以及 w1/w2/w3 如何加载到 w13/w2

Kernel：
  Marlin / Cutlass / Triton / FP8 scaled-mm / fused MoE
  → 使用本 rank 已对齐的量化参数执行 forward
```

如果只记一句话：

```text
vLLM 并行量化的核心，是让 PP 的层归属、TP 的矩阵分片、EP 的 expert 分布、低 bit packed layout、scale/zero point/g_idx 的切分规则，以及 kernel 的运行时格式全部一致。
```

再压缩成最小心智模型：

```text
PP 切层，TP 切矩阵，EP 切 expert，DP 复制 shard；
Parameter 负责切 tensor，QuantMethod 负责创建和转换量化格式，kernel 只消费已经对齐好的本地量化参数。
```
