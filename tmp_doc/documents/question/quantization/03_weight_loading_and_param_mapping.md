# 03. 量化权重如何加载和映射？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\__init__.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\base_loader.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\default_loader.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\sharded_state_loader.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\bitsandbytes_loader.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\weight_utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\llama.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\linear.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\auto_gptq.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\auto_awq.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\fp8.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\parameter.py`

本问题关注：量化 checkpoint 中的 `weight`、`qweight`、`scales`、`qzeros`、`zero_point`、`g_idx`、FP8 scale、group / block metadata 等，如何经过 checkpoint iterator、模型命名映射、tensor parallel 切分、fused layer 拆分和量化方法专属 `weight_loader`，最终进入 vLLM layer 和 kernel 可消费的参数布局。

---

## 1. 一句话回答

量化权重加载不是把 checkpoint tensor 直接按名字塞进 `state_dict`。

它的主链路是：

```text
LoadConfig.load_format
  → 选择 ModelLoader
  → 初始化带 quant_method 的模型结构
  → quant_method.create_weights() 注册 qweight / scales / qzeros / g_idx 等参数
  → checkpoint iterator 流式产出 (name, tensor)
  → model.load_weights() 做名字映射和 fused layer 映射
  → param.weight_loader(...) 做 TP 切分、packed offset 调整、scale 映射
  → process_weights_after_loading() 转成 kernel 最终布局
```

所以量化权重加载的核心不是“文件格式解析”，而是：

```text
checkpoint 里的低 bit 表示
  → vLLM 参数对象
  → TP 本地 shard
  → kernel 需要的 packed / scale / zero point 布局
```

---

## 2. 最小主链路

以默认 HuggingFace / safetensors / pt 加载路径为例，主链路可以压缩成：

```text
BaseModelLoader.load_model()
  → initialize_model(...)
      → 构造具体模型
      → 构造 Linear / QKV / MLP 层
      → quant_method.create_weights(...)
  → DefaultModelLoader.load_weights(model, model_config)
      → get_all_weights()
      → _get_weights_iterator()
      → model.load_weights(iterator)
          → 名字映射
          → fused q/k/v、gate/up 映射
          → param.weight_loader(param, loaded_weight, shard_id?)
  → finalize_layerwise_processing()  # online quant 场景
  → process_weights_after_loading()
      → quant_method.process_weights_after_loading(layer)
      → 转换成 kernel 最终格式
```

对应源码：

```text
base_loader.py:42
base_loader.py:64
base_loader.py:77
base_loader.py:80

default_loader.py:244

default_loader.py:415

default_loader.py:427
```

几个关键点：

```text
1. 模型结构先初始化，参数形状先由 quant_method 决定；
2. checkpoint iterator 只是流式产出原始 tensor；
3. 真正决定“这个 tensor 放到哪里”的是 model.load_weights() 和 param.weight_loader；
4. 很多量化格式还要在加载后再做一次 kernel layout 转换。
```

---

## 3. load_format 和 quantization 不是一回事

vLLM 里至少有两个容易混淆的配置维度。

### 3.1 load_format：权重从哪里、用什么 loader 读

`LoadFormats` 定义在 `model_loader/__init__.py`。

常见值包括：

```text
auto
hf
safetensors
fastsafetensors
instanttensor
pt
npcache
bitsandbytes
sharded_state
runai_streamer
runai_streamer_sharded
tensorizer
```

位置：`model_loader/__init__.py:33` 到 `model_loader/__init__.py:49`

它决定使用哪个 `ModelLoader`：

```text
hf / safetensors / pt / npcache / fastsafetensors
  → DefaultModelLoader

bitsandbytes
  → BitsAndBytesModelLoader

sharded_state / runai_streamer_sharded
  → ShardedStateLoader

tensorizer
  → TensorizerLoader
```

位置：`model_loader/__init__.py:50` 到 `model_loader/__init__.py:66`

### 3.2 quantization：模型层注册什么参数、forward 用什么 kernel

`model_config.quantization` / `quant_config` 决定每个 linear layer 的 `quant_method`。

例如：

```text
AutoGPTQLinearMethod
  → qweight / g_idx / scales / qzeros

AutoAWQMarlinLinearMethod
  → qweight / qzeros / scales

Fp8LinearMethod
  → weight / weight_scale 或 weight_scale_inv / input_scale

BitsAndBytes
  → 特殊 qweight iterator + quant_state 绑定
```

也就是说：

```text
load_format 解决“怎么读 checkpoint”；
quantization 解决“读出来的 tensor 应该按什么量化参数和 kernel 布局解释”。
```

---

## 4. 模型初始化时，量化参数已经注册好了

量化参数不是 checkpoint 到来时临时创建的。

在 `ColumnParallelLinear` 和 `RowParallelLinear` 初始化时，会调用：

```python
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=...,
    output_partition_sizes=...,
    input_size=...,
    output_size=...,
    params_dtype=...,
    weight_loader=...
)
```

位置：`linear.py:461` 到 `linear.py:473`

Row parallel 也类似：

位置：`linear.py:1565` 到 `linear.py:1577`

这一步会根据量化方法注册不同参数。

例如 GPTQ：

```text
qweight: PackedvLLMParameter
  - dtype=int32
  - packed_dim=0
  - packed_factor=pack_factor
  - input_dim=0
  - output_dim=1

g_idx: RowvLLMParameter
  - input_dim=0

scales: ChannelQuantScaleParameter 或 GroupQuantScaleParameter

qzeros: PackedColumnParameter 或 PackedvLLMParameter
```

位置：`auto_gptq.py:324` 到 `auto_gptq.py:443`

例如 AWQ：

```text
qweight: PackedvLLMParameter
  - AWQ checkpoint 中沿 output dim pack
  - packed_dim=1

qzeros: PackedvLLMParameter
scales: GroupQuantScaleParameter
```

位置：`auto_awq.py:473` 到 `auto_awq.py:516`

例如 FP8：

```text
非 block FP8：
  weight
  weight_scale: PerTensorScaleParameter
  input_scale: 可选，静态 activation scale

block FP8：
  weight
  weight_scale_inv: BlockQuantScaleParameter
```

位置：`fp8.py:352` 到 `fp8.py:385`

这解释了一个重要现象：

```text
checkpoint 里的 scales / qzeros / g_idx 不是普通附属 metadata，
它们在 vLLM 里通常就是注册到 layer 上的 Parameter。
```

---

## 5. checkpoint iterator 只负责产出 name 和 tensor

默认加载器中，`_get_weights_iterator()` 根据文件类型选择不同 iterator。

核心分支：

```text
npcache
  → np_cache_weights_iterator

safetensors
  → safetensors_weights_iterator
  → fastsafetensors_weights_iterator
  → instanttensor_weights_iterator
  → multi_thread_safetensors_weights_iterator

pt / bin
  → pt_weights_iterator
  → multi_thread_pt_weights_iterator
```

位置：`default_loader.py:244` 到 `default_loader.py:319`

然后 `DefaultModelLoader.load_weights()` 调用：

```python
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`default_loader.py:415` 到 `default_loader.py:427`

这里要注意：

```text
iterator 产出的还是 checkpoint 原始名字；
后续 model.load_weights() 才做模型结构相关的名字映射；
param.weight_loader 才做 TP shard、packed shard、scale shard 的真实拷贝。
```

---

## 6. AutoWeightsLoader：通用名字递归加载

部分模型使用 `AutoWeightsLoader` 自动递归加载权重。

它的职责是：

```text
1. 按 `.` 拆分 checkpoint name；
2. 递归匹配 child module；
3. 找到当前 module 的 parameter / buffer；
4. 如果参数有 weight_loader，就调用 param.weight_loader；
5. 否则使用 default_weight_loader。
```

关键代码：

```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, weight_data)
```

位置：`models/utils.py:231` 到 `models/utils.py:232`

递归加载 child module / child param 的位置：

```text
models/utils.py:268
models/utils.py:300
models/utils.py:318
```

它适合名字一一对应的模型。

但很多 LLM 不完全依赖它，因为 attention 和 MLP 常常在 vLLM 里是 fused layer，而 checkpoint 里是拆开的 `q_proj` / `k_proj` / `v_proj` 或 `gate_proj` / `up_proj`。

---

## 7. LLaMA 的 fused 名字映射

以 LLaMA 为例，checkpoint 中常见名字是：

```text
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
mlp.gate_proj.weight
mlp.up_proj.weight
```

但 vLLM layer 里通常是：

```text
self_attn.qkv_proj.weight
mlp.gate_up_proj.weight
```

所以 `LlamaModel.load_weights()` 里有显式映射：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

位置：`models/llama.py:433` 到 `models/llama.py:441`

加载时：

```text
1. 如果 name 命中 q_proj/k_proj/v_proj/gate_proj/up_proj；
2. 把 name 替换成 vLLM fused 参数名；
3. 找到 params_dict[name]；
4. 调用 param.weight_loader(param, loaded_weight, shard_id)。
```

核心调用：

```python
param = params_dict[name]
weight_loader = param.weight_loader
weight_loader(param, loaded_weight, shard_id)
```

位置：`models/llama.py:456` 到 `models/llama.py:470`

因此：

```text
shard_id='q'/'k'/'v' 表示 checkpoint 的 q/k/v shard；
shard_id=0/1 表示 gate/up shard；
同一个 vLLM fused parameter 会被多个 checkpoint tensor 分段写入。
```

`packed_modules_mapping` 也记录了这种关系：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

位置：`models/llama.py:489` 到 `models/llama.py:492`

---

## 8. 参数对象如何表达“切哪一维”

量化加载的关键是 vLLM 自定义参数类。

入口是 `BasevLLMParameter`：

```text
BasevLLMParameter
  - 保存 weight_loader
  - 保存 tp_rank / tp_size
  - 提供 load_column_parallel_weight
  - 提供 load_row_parallel_weight
  - 提供 load_merged_column_weight
  - 提供 load_qkv_weight
```

位置：`parameter.py:32` 到 `parameter.py:120`

几个重要子类：

```text
ModelWeightParameter
  普通 linear weight，可 column / row parallel。

GroupQuantScaleParameter
  group-wise quant scales，可按 input_dim / output_dim 切。

ChannelQuantScaleParameter
  channel-wise scales，等价于 column parallel scale。

PerTensorScaleParameter
  per-tensor scale，fused QKV / gate_up 中按 shard_id 写入 scale 数组。

PackedColumnParameter
  packed on disk，只支持 column parallel。

PackedvLLMParameter
  packed on disk，同时支持 row / column parallel。

BlockQuantScaleParameter
  block-wise quant scale，按 block 维度切，不是简单按原始 hidden size 切。
```

位置：

```text
parameter.py:233
parameter.py:242
parameter.py:251
parameter.py:260
parameter.py:313
parameter.py:353
parameter.py:397
```

这些参数类把两类信息编码进参数本身：

```text
1. 这个 tensor 是沿 input_dim 还是 output_dim 切；
2. 这个 tensor 是否 packed，packed_factor 和 packed_dim 是什么。
```

---

## 9. ColumnParallelLinear 如何加载权重

Column parallel 的含义是：

```text
weight 的 output dimension 被 TP 切分。
```

普通 loader 逻辑：

```python
if output_dim is not None and not is_sharded_weight:
    shard_size = param_data.shape[output_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

位置：`linear.py:517` 到 `linear.py:538`

V2 参数 loader 逻辑更直接：

```python
param.load_column_parallel_weight(loaded_weight=loaded_weight)
```

位置：`linear.py:540` 到 `linear.py:546`

在 `parameter.py` 里真正切片：

```python
shard_size = self.data.shape[self.output_dim]
loaded_weight = loaded_weight.narrow(
    self.output_dim, self.tp_rank * shard_size, shard_size
)
self.data.copy_(loaded_weight)
```

位置：`parameter.py:148` 到 `parameter.py:154`

因此 column parallel 的心智模型是：

```text
checkpoint full output dim
  → 按 tp_rank 取本地 output shard
  → copy 到本 rank 的 param.data
```

---

## 10. RowParallelLinear 如何加载权重

Row parallel 的含义是：

```text
weight 的 input dimension 被 TP 切分。
```

普通 loader 逻辑：

```python
if input_dim is not None and not is_sharded_weight:
    shard_size = param_data.shape[input_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
```

位置：`linear.py:1597` 到 `linear.py:1617`

V2 参数 loader：

```python
param.load_row_parallel_weight(loaded_weight=loaded_weight)
```

位置：`linear.py:1619` 到 `linear.py:1626`

对应参数类：

```python
shard_size = self.data.shape[self.input_dim]
loaded_weight = loaded_weight.narrow(
    self.input_dim, self.tp_rank * shard_size, shard_size
)
self.data.copy_(loaded_weight)
```

位置：`parameter.py:220` 到 `parameter.py:230`

因此 row parallel 的心智模型是：

```text
checkpoint full input dim
  → 按 tp_rank 取本地 input shard
  → copy 到本 rank 的 param.data
```

---

## 11. MergedColumnParallelLinear：gate/up 这类 fused MLP 如何加载

`MergedColumnParallelLinear` 用在类似 `gate_up_proj` 的 fused MLP。

它的输出由多个逻辑矩阵拼接：

```text
gate_up_proj = [gate_proj | up_proj]
```

加载时有两种情况。

### 11.1 checkpoint 是拆开的 gate_proj / up_proj

模型 `load_weights()` 会把：

```text
gate_proj → gate_up_proj, shard_id=0
up_proj   → gate_up_proj, shard_id=1
```

然后 `MergedColumnParallelLinear.weight_loader()` 根据 `loaded_shard_id` 计算 fused 参数中的位置：

```python
shard_offset = sum(self.output_sizes[:loaded_shard_id])
shard_size = self.output_sizes[loaded_shard_id]
shard_offset //= self.tp_size
shard_size //= self.tp_size
```

位置：`linear.py:745` 到 `linear.py:750`

接着再考虑量化 packed 情况：

```python
if packed_dim == output_dim:
    shard_size = round(shard_size // param.packed_factor)
    shard_offset = round(shard_offset // param.packed_factor)
    shard_size, shard_offset = adjust_marlin_shard(...)
```

位置：`linear.py:758` 到 `linear.py:768`

最后拷贝到 fused param 对应区间：

```python
param_data = param_data.narrow(output_dim, shard_offset, shard_size)
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
param_data.copy_(loaded_weight)
```

位置：`linear.py:785` 到 `linear.py:805`

### 11.2 checkpoint 已经是 fused 的 gate_up_proj

有些模型 checkpoint 里已经保存 fused tensor。

这时没有 `shard_id`，loader 会自己把 checkpoint tensor 拆成多个逻辑 shard，再递归调用 shard loader：

```text
_load_fused_module_from_checkpoint()
  → 按 output_sizes 拆 gate/up
  → 对 packed 参数调整 shard_size / shard_offset
  → weight_loader_v2(..., shard_id)
```

位置：`linear.py:807` 到 `linear.py:845`

这解释了为什么：

```text
同样叫 gate_up_proj，checkpoint 可能是拆开的，也可能已经 fused；
vLLM loader 两种都要处理。
```

---

## 12. QKVParallelLinear：q/k/v 映射、GQA/MQA 和 TP 复制

`QKVParallelLinear` 用在 attention 的 QKV projection。

vLLM 中本地输出布局是：

```text
[q | k | v]
```

但 q、k、v 的切分规则不完全相同。

初始化时会计算：

```text
num_heads：本 rank 的 query heads
num_kv_heads：本 rank 的 kv heads
num_kv_head_replicas：当 TP 数大于 KV heads 时，KV head 在多个 rank 上复制
output_sizes：全局意义上的 q/k/v 输出大小
```

位置：`linear.py:958` 到 `linear.py:984`

### 12.1 q/k/v 在 fused 参数里的偏移

V2 loader 中：

```python
shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
shard_size = self._get_shard_size_mapping(loaded_shard_id)
```

位置：`linear.py:1104` 到 `linear.py:1108`

对应映射：

```text
q offset = 0
k offset = num_heads * head_size
v offset = (num_heads + num_kv_heads) * head_size
```

位置：`linear.py:1011` 到 `linear.py:1027`

### 12.2 Q 和 KV 的 TP rank 选择不同

参数类里的 QKV 加载逻辑：

```python
shard_id_int = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
```

位置：`parameter.py:193` 到 `parameter.py:198`

`QKVParallelLinear.weight_loader()` 里也有同样语义：

```python
if loaded_shard_id == "q":
    shard_rank = self.tp_rank
else:
    shard_rank = self.tp_rank // self.num_kv_head_replicas
```

位置：`linear.py:1278` 到 `linear.py:1286`

含义是：

```text
Q heads 通常按 TP rank 切；
K/V heads 在 GQA/MQA 下可能少于 TP rank；
当 tp_size > total_num_kv_heads 时，K/V 会在多个 TP rank 上复制。
```

### 12.3 checkpoint 已经 fused 的 qkv_proj

如果 checkpoint 直接给 `qkv_proj`，没有 `q/k/v shard_id`，loader 会拆：

```text
_load_fused_module_from_checkpoint()
  → q shard
  → k shard
  → v shard
  → weight_loader_v2(..., shard_id='q'/'k'/'v')
```

位置：`linear.py:1029` 到 `linear.py:1077`

---

## 13. packed weight 为什么要调整 shard offset

GPTQ / AWQ / Marlin 等低 bit 权重通常不是一个元素对应一个权重值。

例如 int4 常见做法是：

```text
多个 4-bit 值 pack 到一个 int32 中。
```

所以如果原始逻辑 shard 是：

```text
shard_offset = 4096
shard_size = 4096
```

实际 checkpoint tensor 的 packed 维度上可能要变成：

```text
packed_offset = shard_offset / pack_factor
packed_size = shard_size / pack_factor
```

在 vLLM 中，这个信息由参数对象记录：

```text
PackedColumnParameter / PackedvLLMParameter
  - packed_factor
  - packed_dim
  - marlin_tile_size
```

位置：`parameter.py:313` 到 `parameter.py:394`

通用调整入口：

```python
adjust_shard_indexes_for_packing(...)
```

位置：`parameter.py:344` 到 `parameter.py:350`

在 fused MLP 和 QKV loader 中，都会检查：

```python
if packed_dim == output_dim:
    shard_size = shard_size // param.packed_factor
    shard_offset = shard_offset // param.packed_factor
```

位置：

```text
linear.py:720 到 linear.py:727
linear.py:761 到 linear.py:768
linear.py:1179 到 linear.py:1186
linear.py:1241 到 linear.py:1249
```

因此 packed 参数加载时要同时回答三个问题：

```text
1. 逻辑 shard 在 fused layer 的哪一段？
2. TP 本 rank 应该取 checkpoint 的哪一段？
3. 如果这一维 packed 了，offset / size 要除以 pack_factor 或按 Marlin tile 修正。
```

---

## 14. scale / zero point / g_idx 如何映射

量化 checkpoint 中的 scale / zero point / group index 也是参数加载主链路的一部分。

### 14.1 group-wise scales

GPTQ 中，如果 scales / qzeros 需要按 input group 切，会使用：

```text
GroupQuantScaleParameter(output_dim=1, input_dim=0)
PackedvLLMParameter(input_dim=0, output_dim=1, packed_dim=1)
```

位置：`auto_gptq.py:428` 到 `auto_gptq.py:438`

这意味着：

```text
scales 既可能随 output dim column shard；
也可能随 input group row shard；
取决于 desc_act / group_size / row_parallel 等条件。
```

GPTQ 里有一段显式判断：

```text
marlin_repeat_scales_on_all_ranks(...)
  → scales_and_zp_input_dim = None，所有 TP rank 重复 scales
否则
  → scales_and_zp_input_dim = 0，scales 按 input groups 切分
```

位置：`auto_gptq.py:364` 到 `auto_gptq.py:377`

### 14.2 per-tensor scales

FP8 per-tensor scale 常用 `PerTensorScaleParameter`。

它的特点是：

```text
fused QKV 有 q/k/v 三个逻辑矩阵，scale 参数可以是长度为 3 的数组；
fused gate_up 有 gate/up 两个逻辑矩阵，scale 参数可以是长度为 2 的数组；
checkpoint 里的 scalar scale 会按 shard_id 写入对应槽位。
```

核心逻辑：

```python
param_data = param_data[shard_id]
param_data.copy_(loaded_weight)
```

位置：`parameter.py:291` 到 `parameter.py:310`

对于 checkpoint 已经 fused 且只有一个 scale 的情况，loader 会把同一个 scale 填到所有逻辑 shard 槽位。

QKV：

位置：`linear.py:1085` 到 `linear.py:1096`

Merged MLP：

位置：`linear.py:853` 到 `linear.py:871`

### 14.3 qzeros / zero_point

GPTQ / AWQ 的 zero point 常以 packed int32 形式存在，例如：

```text
qzeros: PackedColumnParameter 或 PackedvLLMParameter
```

位置：

```text
auto_gptq.py:402 到 auto_gptq.py:438
auto_awq.py:490 到 auto_awq.py:516
```

它们会走和 qweight 类似的 packed offset 调整，只是形状维度代表的是 group 和 output pack。

### 14.4 g_idx

GPTQ 的 activation order / desc_act 需要 `g_idx`：

```text
g_idx: RowvLLMParameter(input_dim=0)
```

位置：`auto_gptq.py:392` 到 `auto_gptq.py:400`

因此 TP 下它按 input dimension 切分。

---

## 15. FP8 KV cache scale 名字重映射

FP8 checkpoint 里经常有 attention KV scale，名字可能和 vLLM 期望不一致。

LLaMA 加载时遇到：

```python
if "scale" in name or "zero_point" in name:
    name = maybe_remap_kv_scale_name(name, params_dict)
```

位置：`models/llama.py:451` 到 `models/llama.py:455`

`maybe_remap_kv_scale_name()` 支持多种历史 / 外部格式，例如：

```text
.kv_scale
.self_attn.k_proj.k_scale
.self_attn.v_proj.v_scale
.self_attn.qkv_proj.k_scale
.self_attn.qkv_proj.v_scale
```

并把它们重映射到 vLLM attention 参数期望的名字。

位置：`weight_utils.py:1341` 到 `weight_utils.py:1399`

特殊点：

```text
旧格式 kv_scale 会被当成 k_scale，并复制 / 映射到新格式；
如果 remapped name 在 params_dict 中不存在，则跳过该 scale。
```

这说明：

```text
scale 名字映射不只发生在 linear weight；
attention KV cache scale 也有专门兼容逻辑。
```

---

## 16. AWQ：加载后还要转换 packing 格式

AWQ 的一个典型特点是：

```text
checkpoint 中 qweight 按 AWQ 格式保存，packed along output dim；
kernel 最终更希望使用 GPTQ-like 标准格式，packed along input dim。
```

AWQ 参数创建：

```text
qweight:
  shape = [input_size_per_partition, output_size_per_partition / pack_factor]
  packed_dim = 1

qzeros:
  shape = [num_groups, output_size_per_partition / pack_factor]
  packed_dim = 1

scales:
  shape = [num_groups, output_size_per_partition]
```

位置：`auto_awq.py:473` 到 `auto_awq.py:516`

加载后处理：

```python
_convert_awq_to_standard_format(
    layer, "qweight", "qzeros", self.quant_config.quant_type.size_bits
)
self.kernel.process_weights_after_loading(layer)
```

位置：`auto_awq.py:525` 到 `auto_awq.py:533`

因此 AWQ 的完整链路是：

```text
checkpoint AWQ packed layout
  → 先按 checkpoint layout 加进 qweight / qzeros / scales
  → process_weights_after_loading()
  → 转为 kernel 统一布局
```

---

## 17. GPTQ：qweight、qzeros、scales、g_idx 的加载

GPTQ / AutoGPTQ 在 vLLM 中通常走 Marlin 相关 kernel。

参数创建时会先构造 kernel config：

```text
full_weight_shape=(input_size, output_size)
partition_weight_shape=(input_size_per_partition, output_size_per_partition)
weight_type=quant_type
group_size=group_size
zero_points=False
has_g_idx=desc_act
```

位置：`auto_gptq.py:339` 到 `auto_gptq.py:350`

然后注册：

```text
qweight
  PackedvLLMParameter
  input_dim=0, output_dim=1, packed_dim=0

g_idx
  RowvLLMParameter
  input_dim=0

scales
  ChannelQuantScaleParameter 或 GroupQuantScaleParameter

qzeros
  PackedColumnParameter 或 PackedvLLMParameter
```

位置：`auto_gptq.py:378` 到 `auto_gptq.py:443`

加载后：

```python
self.kernel.process_weights_after_loading(layer)
```

位置：`auto_gptq.py:453` 到 `auto_gptq.py:454`

这里的重点是：

```text
GPTQ 的 qweight packed_dim 是 input 维；
AWQ 的 qweight 初始 packed_dim 是 output 维；
二者 checkpoint layout 不同，不能只看“都是 int4 qweight”。
```

---

## 18. FP8：weight scale 和 block scale 的加载

FP8 既可能是 per-tensor scale，也可能是 block-wise scale。

### 18.1 per-tensor FP8

非 block FP8 注册：

```text
weight
weight_scale: PerTensorScaleParameter
input_scale: 可选
```

位置：`fp8.py:352` 到 `fp8.py:366`

如果 checkpoint 不是已经序列化好的 FP8，后处理会执行量化 / scale 合并：

```text
process_fp8_weight_tensor_strategy(...)
replace_parameter(layer, "weight", weight.data)
replace_parameter(layer, "weight_scale", weight_scale.data)
```

位置：`fp8.py:416` 到 `fp8.py:437`

### 18.2 block FP8

block FP8 注册：

```text
weight_scale_inv: BlockQuantScaleParameter
```

位置：`fp8.py:367` 到 `fp8.py:379`

它的 shard 不是按原始 output size 简单除以 TP，而是要按 block shape 修正。

在 fused MLP / QKV loader 中可以看到：

```python
if isinstance(param, BlockQuantScaleParameter):
    shard_size, shard_offset = adjust_block_scale_shard(...)
```

位置：

```text
linear.py:714 到 linear.py:718
linear.py:752 到 linear.py:756
linear.py:1060 到 linear.py:1064
linear.py:1110 到 linear.py:1114
```

因此 block FP8 的关键是：

```text
scale tensor 的切分单位是 block，不是原始 hidden dimension。
```

---

## 19. BitsAndBytes：特殊 loader，不只是普通 iterator

`bitsandbytes` 的加载路径使用 `BitsAndBytesModelLoader`，不是普通 `DefaultModelLoader`。

主入口：

```python
qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(...)
loaded_weights = model.load_weights(qweight_iterator)
...
stacked_quant_state_dict = self._stack_quantization_states(...)
self._bind_quant_states_to_params(model, stacked_quant_state_dict)
```

位置：`bitsandbytes_loader.py:804` 到 `bitsandbytes_loader.py:836`

它的特点：

```text
1. 可能边加载边量化 HF 权重；
2. 也可能读取预量化的 bitsandbytes state；
3. qweight 仍然走 model.load_weights()；
4. quant_state 不只是普通参数，还要后续绑定到 param attrs；
5. fused QKV / gate_up 的 quant_state 也要 stack / fuse。
```

绑定 quant state 时会设置：

```text
bnb_quant_state
bnb_shard_offsets
matmul_state  # 8bit 场景
```

位置：`bitsandbytes_loader.py:780` 到 `bitsandbytes_loader.py:802`

并且 bitsandbytes 参数常带有：

```text
use_bitsandbytes_4bit=True
is_sharded_weight=True
```

这会影响 `linear.py` 中 loader 是否再次 narrow。

例如：

```python
is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
```

位置：

```text
linear.py:520 到 linear.py:525
linear.py:1598 到 linear.py:1604
```

含义是：

```text
bitsandbytes loader 可能已经给了本地 shard；
linear weight_loader 不能再按 TP 重复切一次。
```

---

## 20. sharded_state：预切分 checkpoint 的特殊路径

`ShardedStateLoader` 用于已经按 TP rank 保存的 checkpoint。

它不是走 checkpoint full tensor → param.weight_loader 切分的逻辑，而是：

```text
1. 根据当前 TP rank 找 model-rank-{rank}-part-* 文件；
2. 读出本 rank 的 safetensors；
3. 直接按 key 找 model.state_dict()[key]；
4. copy 到本地参数。
```

核心代码：

```python
rank = get_tensor_model_parallel_rank()
pattern = os.path.join(local_model_path, self.pattern.format(rank=rank, part="*"))
...
state_dict = self._filter_subtensors(model.state_dict())
for key, tensor in self.iterate_over_files(filepaths):
    param_data = state_dict[key].data
    param_data.copy_(tensor)
```

位置：`sharded_state_loader.py:110` 到 `sharded_state_loader.py:162`

所以：

```text
DefaultModelLoader：加载 full checkpoint，weight_loader 负责本地切分；
ShardedStateLoader：checkpoint 已经是本地 shard，直接 copy。
```

---

## 21. 严格加载检查为什么量化模型通常不默认开启

`DefaultModelLoader.load_weights()` 最后会决定是否跟踪 missing weights。

逻辑是：

```python
default_enable_weights_track = (
    model_config.quantization is None and loaded_weights is not None
)
```

位置：`default_loader.py:434` 到 `default_loader.py:438`

也就是说默认只对非量化模型开启严格 tracking。

原因可以理解为：

```text
量化模型中有些参数可能来自：
- checkpoint tensor；
- online quantization；
- process_weights_after_loading 转换；
- kernel 后处理生成 / 替换；
- bitsandbytes quant_state 绑定；
- 兼容性 remap 后跳过的旧 scale。
```

因此不能简单用“所有 named_parameters 都必须直接从 checkpoint 命中一次”来判断。

---

## 22. 一个完整例子：LLaMA GPTQ q_proj.weight 如何进入 qkv_proj.qweight

以 GPTQ LLaMA 的 `q_proj` 为例，可以按以下链路理解。

### 22.1 初始化阶段

```text
LlamaAttention
  → QKVParallelLinear(..., quant_config=AutoGPTQConfig)
  → AutoGPTQLinearMethod.create_weights()
  → 在 qkv_proj 上注册：
      qweight
      g_idx
      scales
      qzeros
```

其中 `qweight` 是：

```text
PackedvLLMParameter(input_dim=0, output_dim=1, packed_dim=0)
```

位置：`auto_gptq.py:378` 到 `auto_gptq.py:390`

### 22.2 checkpoint iterator 阶段

iterator 产出类似：

```text
model.layers.0.self_attn.q_proj.qweight
model.layers.0.self_attn.q_proj.scales
model.layers.0.self_attn.q_proj.qzeros
model.layers.0.self_attn.q_proj.g_idx
```

### 22.3 模型 load_weights 阶段

LLaMA 映射：

```text
.q_proj → .qkv_proj, shard_id='q'
```

位置：`models/llama.py:433` 到 `models/llama.py:470`

所以 name 变成：

```text
model.layers.0.self_attn.qkv_proj.qweight
```

然后调用：

```text
qkv_proj.qweight.weight_loader(param, loaded_weight, shard_id='q')
```

### 22.4 QKV loader 阶段

`QKVParallelLinear` 计算 q shard 在 fused 参数中的位置：

```text
q offset = 0
q size = num_heads * head_size
```

位置：`linear.py:1011` 到 `linear.py:1027`

如果参数是 packed 或 block scale，会调整 offset / size。

然后调用参数自身：

```python
param.load_qkv_weight(
    loaded_weight=loaded_weight,
    shard_id="q",
    shard_offset=shard_offset,
    shard_size=shard_size,
    num_heads=self.num_kv_head_replicas,
)
```

位置：`linear.py:1116` 到 `linear.py:1123`

### 22.5 参数 copy 阶段

`PackedvLLMParameter` 继承 `_ColumnvLLMParameter.load_qkv_weight()`。

它会：

```text
1. 如果 packed_dim == output_dim，先按 pack_factor 调整 shard_size / offset；
2. 在 param.data 的 output_dim 上 narrow 到 q 的位置；
3. 在 checkpoint loaded_weight 的 output_dim 上取本 TP rank 的 q shard；
4. copy 到 param.data。
```

位置：`parameter.py:178` 到 `parameter.py:201`

最终结果是：

```text
checkpoint 的 q_proj.qweight
  → vLLM qkv_proj.qweight 的 q 段
  → 当前 TP rank 的本地 packed shard
```

---

## 23. 一个完整例子：AWQ gate_proj/up_proj 如何进入 gate_up_proj

AWQ MLP 的 checkpoint 可能有：

```text
mlp.gate_proj.qweight
mlp.up_proj.qweight
mlp.gate_proj.qzeros
mlp.up_proj.qzeros
mlp.gate_proj.scales
mlp.up_proj.scales
```

LLaMA 映射：

```text
gate_proj → gate_up_proj, shard_id=0
up_proj   → gate_up_proj, shard_id=1
```

位置：`models/llama.py:433` 到 `models/llama.py:470`

`MergedColumnParallelLinear` 根据 shard_id 写入 fused 参数不同段：

```text
shard_id=0 → gate 段
shard_id=1 → up 段
```

位置：`linear.py:745` 到 `linear.py:805`

AWQ 的 `qweight/qzeros` 是 packed 参数，loader 会根据 `packed_dim=1` 对 output 维 offset 和 size 做调整。

加载后还会做：

```text
_convert_awq_to_standard_format()
  → AWQ packed output layout
  → GPTQ-like packed input layout
```

位置：`auto_awq.py:525` 到 `auto_awq.py:533`

所以 AWQ gate/up 的完整心智模型是：

```text
gate_proj/up_proj checkpoint tensors
  → gate_up_proj fused parameter 的不同逻辑段
  → TP 本地 output shard
  → AWQ packed layout copy
  → loading 后转换成 kernel 标准 layout
```

---

## 24. 容易疑惑的点

### 24.1 量化 checkpoint 的 scale 是 metadata 吗？

不是简单 metadata。

在 vLLM 中，很多 scale / qzeros / g_idx 都是注册到 layer 上的 `Parameter`，只是不参与训练。

### 24.2 为什么同一个 qkv_proj 参数会被加载三次？

因为 checkpoint 可能是拆开的：

```text
q_proj
k_proj
v_proj
```

而 vLLM 运行时为了 kernel 和性能使用 fused：

```text
qkv_proj = [q | k | v]
```

所以同一个 fused parameter 会按 shard_id 写入不同区间。

### 24.3 为什么 TP 下 K/V 的 shard_rank 不是 tp_rank？

GQA / MQA 下 KV heads 可能少于 query heads。

当 `tp_size > total_num_kv_heads` 时，K/V heads 会复制到多个 rank，所以 K/V 使用：

```text
tp_rank // num_kv_head_replicas
```

而 Q 使用：

```text
tp_rank
```

### 24.4 为什么 packed qweight 不能直接按原始 hidden size 切？

因为 checkpoint tensor 的某个维度已经被 pack。

如果 `pack_factor=8`，逻辑上 8192 个 int4 权重在 packed tensor 中可能只对应 1024 个 int32 元素。

所以 shard offset / size 必须按 `packed_factor` 和 Marlin tile 规则修正。

### 24.5 为什么有些 scale 是 0 维 scalar？

例如 AutoFP8 checkpoint 里的 per-tensor scale 可能是 scalar。

loader 会把 0-D tensor reshape 成 `[1]`，或者在 fused layer 中按 shard_id 写入 scale 数组。

相关位置：

```text
linear.py:532 到 linear.py:535
linear.py:541 到 linear.py:546
parameter.py:303 到 parameter.py:310
```

### 24.6 为什么量化模型默认不做严格 missing weight 检查？

因为量化参数可能由 checkpoint、online quant、后处理、kernel 转换、quant_state 绑定共同产生。

简单的“每个参数必须 checkpoint 命中一次”容易误报。

### 24.7 load_format=bitsandbytes 和 quantization=bitsandbytes 是同一个概念吗？

不是完全同一个层面。

`load_format=bitsandbytes` 选择 `BitsAndBytesModelLoader`，负责 qweight iterator 和 quant_state 绑定。

`quantization` / `quant_config` 决定模型层如何创建 BNB 参数、forward 使用什么量化实现。

### 24.8 sharded_state 还会走 TP 切分吗？

通常不会走同样的 full checkpoint 切分逻辑。

`sharded_state` 假设 checkpoint 已经按 rank 保存，本 rank 直接读取自己的 shard 并 copy 到 `state_dict`。

---

## 25. 总结

量化权重加载可以压缩成一条链：

```text
LoadConfig
  → ModelLoader
  → checkpoint iterator
  → model.load_weights()
  → name remap / fused mapping
  → param.weight_loader()
  → TP shard / packed shard / scale shard
  → process_weights_after_loading()
  → kernel-ready layout
```

如果只记住一个心智模型：

```text
checkpoint 的 tensor 名字描述“来源矩阵”；
vLLM 的参数名描述“运行时 fused layer”；
shard_id 描述“写入 fused layer 的哪一段”；
BasevLLMParameter 描述“按哪一维、按什么 packing 规则切”；
quant_method 描述“这些参数最终要喂给哪个 kernel”。
```

再压缩成一句话：

```text
量化权重加载的本质，是把 checkpoint 中分散的低 bit 权重、scale、zero point 和 group 信息，映射成当前 TP rank 上 fused linear kernel 能直接消费的本地 packed 参数。
```
