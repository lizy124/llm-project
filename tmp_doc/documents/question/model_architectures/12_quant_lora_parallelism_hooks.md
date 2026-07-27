# 12. Quantization、LoRA、Parallelism 如何 hook 到模型架构？

源码位置：

- `code/vllm/vllm/model_executor/models/llama.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/models/utils.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/lora/lora_model.py`
- `code/vllm/vllm/lora/model_manager.py`
- `code/vllm/vllm/lora/utils.py`
- `code/vllm/vllm/lora/layers/`
- `code/vllm/vllm/distributed/`

本问题关注：一个具体模型架构文件，例如 `llama.py`，并不会自己完整实现量化、LoRA、Tensor Parallel、Pipeline Parallel、Expert Parallel 的全部细节；它主要通过 layer 选型、`quant_config` 传递、`prefix` 命名、权重加载映射、接口标记和 forward 形态，把这些通用能力 hook 进模型结构。

---

## 1. 一句话回答

vLLM 的模型架构层通过四类 hook 接入扩展能力：

```text
1. 构造 layer 时传入 quant_config / prefix；
2. 使用 vLLM 提供的并行层，而不是裸 torch.nn.Linear；
3. 在模型类上声明 SupportsLoRA / SupportsPP 等接口与映射；
4. 在 load_weights() 中把 HF checkpoint 名称映射到 vLLM 的 fused / sharded 参数。
```

以 LLaMA 为例，主线是：

```text
LlamaForCausalLM
  → SupportsLoRA / SupportsPP
  → packed_modules_mapping / embedding_modules
  → LlamaModel.make_layers(...)
  → LlamaDecoderLayer
  → LlamaAttention / LlamaMLP
  → QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear
  → quant_method.create_weights()
  → load_weights() 的 shard_id 映射
  → quant_method.apply() / LoRA wrapper / TP collective / PP IntermediateTensors
```

因此：

```text
模型架构负责“把结构摆对、名字对齐、接口声明清楚”；
量化、LoRA、并行的重逻辑主要在通用 layer、loader、manager、distributed runtime 中完成。
```

---

## 2. 先看 LLaMA 这个具体例子

`llama.py` 是最适合观察 hook 的模型之一。

### 2.1 Attention 里的 hook

源码位置：`code/vllm/vllm/model_executor/models/llama.py:122`

`LlamaAttention.__init__()` 里没有使用普通 `nn.Linear`，而是使用：

```python
self.qkv_proj = QKVParallelLinear(..., quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
self.o_proj = RowParallelLinear(..., quant_config=quant_config, prefix=f"{prefix}.o_proj")
self.attn = Attention(..., quant_config=quant_config, prefix=f"{prefix}.attn")
```

源码位置：

- `code/vllm/vllm/model_executor/models/llama.py:162`
- `code/vllm/vllm/model_executor/models/llama.py:172`
- `code/vllm/vllm/model_executor/models/llama.py:218`

这三处分别接入：

```text
QKVParallelLinear：
  Q/K/V fused projection + Tensor Parallel + quantized linear hook。

RowParallelLinear：
  output projection + Tensor Parallel all-reduce + quantized linear hook。

Attention：
  attention backend、KV cache、KV cache scale、attention quantization 相关 hook。
```

### 2.2 MLP 里的 hook

源码位置：`code/vllm/vllm/model_executor/models/llama.py:79`

`LlamaMLP.__init__()` 使用：

```python
self.gate_up_proj = MergedColumnParallelLinear(..., quant_config=quant_config)
self.down_proj = RowParallelLinear(..., quant_config=quant_config)
```

源码位置：

- `code/vllm/vllm/model_executor/models/llama.py:92`
- `code/vllm/vllm/model_executor/models/llama.py:100`

这里的含义是：

```text
gate_proj + up_proj 在运行时融合成 gate_up_proj；
Column Parallel 负责按输出维切分；
Row Parallel 负责按输入维切分并 all-reduce；
quant_config 会继续传到每个并行线性层内部。
```

### 2.3 顶层模型里的 hook

源码位置：`code/vllm/vllm/model_executor/models/llama.py:446`

`LlamaForCausalLM` 继承：

```python
class LlamaForCausalLM(..., SupportsLoRA, SupportsPP, ...):
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:446`

并声明：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

embedding_modules = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:455` 到 `code/vllm/vllm/model_executor/models/llama.py:463`

这些字段是 LoRA 和权重加载最关键的模型侧声明之一：

```text
hf_to_vllm_mapper：
  告诉 AutoWeightsLoader 如何把 HF 的 q_proj / gate_proj 等名称映射到 qkv_proj / gate_up_proj，并携带 shard_id。

packed_modules_mapping：
  告诉 LoRA / adapter 工具：运行时的 fused module 对应 PEFT 里的哪些子模块。

embedding_modules：
  告诉 LoRA：输入 embedding 和输出 lm_head 可以作为特殊 embedding LoRA 目标处理。
```

---

## 3. quantization hook 的整体链路

量化不是模型文件里写一堆 `if GPTQ / AWQ / FP8`，而是抽象成 `QuantizationConfig` 和 `QuantizeMethodBase`。

核心接口位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:20`

### 3.1 两个核心抽象

`QuantizeMethodBase` 定义每种量化方法必须实现：

```text
create_weights(layer, ...)：
  在 layer 上创建该量化方法需要的参数，例如 weight、qweight、scales、zero_points。

apply(layer, x, bias)：
  forward 时用对应 kernel 或计算路径执行线性层。

process_weights_after_loading(layer)：
  权重加载后做 repack、transpose、online quantize、kernel 格式转换等后处理。
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:20`

`QuantizationConfig` 定义每种量化配置必须实现：

```text
get_name()
get_supported_act_dtypes()
get_min_capability()
get_config_filenames()
from_config()
get_quant_method(layer, prefix)
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:87`

其中最关键的是：

```python
def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
```

源码位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:180`

这表示：

```text
同一个 quant_config 可以根据 layer 类型和 prefix，决定这个具体模块是否量化、使用哪种量化实现。
```

### 3.2 LinearBase 如何接入 quant_config

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:228`

所有 vLLM linear layer 都继承 `LinearBase`。

`LinearBase.__init__()` 的关键逻辑是：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:269`

这就是量化 hook 的入口：

```text
模型传 quant_config
  → layer 调 quant_config.get_quant_method(self, prefix)
  → 得到具体 quant_method
  → 后续创建参数、加载权重、forward 都委托 quant_method。
```

如果没有量化，`UnquantizedLinearMethod` 也走同一套接口，只是创建普通 weight 并执行普通 GEMM。

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:179`

### 3.3 create_weights：量化参数在哪里创建

以 `ColumnParallelLinear` 为例。

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:394`

构造时会调用：

```python
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=self.input_size_per_partition,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    params_dtype=self.params_dtype,
    weight_loader=...
)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:461`

这一步的意义是：

```text
layer 本身不关心 GPTQ/AWQ/FP8 具体需要哪些 tensor；
它只把分片后的维度、全局维度、dtype、weight_loader 交给 quant_method；
quant_method 决定注册普通 weight、qweight、scales、zeros、packed 参数等。
```

同样的模式也出现在：

```text
ReplicatedLinear
ColumnParallelLinear
MergedColumnParallelLinear
QKVParallelLinear
RowParallelLinear
VocabParallelEmbedding
Attention / MLA / MMEncoderAttention 的量化相关参数
```

### 3.4 apply：forward 时如何走量化 kernel

以 `ColumnParallelLinear.forward()` 为例：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:555`

以 `RowParallelLinear.forward()` 为例：

```python
output_parallel = self.quant_method.apply(self, input_parallel, bias_)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:1644`

因此 forward 链路是：

```text
模型 forward
  → qkv_proj / gate_up_proj / down_proj 等 layer.forward()
  → quant_method.apply(layer, input, bias)
  → 具体量化 kernel / 反量化 GEMM / unquantized GEMM
  → TP gather 或 all-reduce
```

这就是为什么模型架构不需要知道具体量化 kernel。

### 3.5 weight_loader：量化权重如何和 TP / fused module 对齐

量化权重经常不是普通二维矩阵，可能有：

```text
qweight
scales
zero_points
packed_factor
block scale
Marlin tile
bitsandbytes 4bit shard
per-tensor scale array
```

因此 vLLM 的并行 layer 在 `weight_loader` 里不仅做普通切片，还要处理量化 packing 后的 offset / size。

典型逻辑包括：

```text
- output_dim / input_dim 决定沿哪一维切 shard；
- tp_rank 决定当前 rank 加载哪一段；
- packed_dim / packed_factor 修正量化打包后的切片大小；
- BlockQuantScaleParameter 根据 block size 修正 scale 切片；
- QKVParallelLinear 按 q / k / v 的 shard_id 计算 offset；
- MergedColumnParallelLinear 按 gate / up 等 shard_id 计算 offset。
```

源码位置：

- `code/vllm/vllm/model_executor/layers/linear.py:517`
- `code/vllm/vllm/model_executor/layers/linear.py:662`
- `code/vllm/vllm/model_executor/layers/linear.py:1078`
- `code/vllm/vllm/model_executor/layers/linear.py:1125`
- `code/vllm/vllm/model_executor/layers/linear.py:1597`

一句话：

```text
weight_loader 是“checkpoint tensor → 当前 rank 的 vLLM 参数”的落点，也是量化、融合、TP 三者交汇最密集的地方。
```

### 3.6 process_weights_after_loading：加载后的量化后处理

模型加载完成后，vLLM 会遍历所有 module：

```python
quant_method = getattr(module, "quant_method", None)
if isinstance(quant_method, QuantizeMethodBase):
    quant_method.process_weights_after_loading(module)
```

源码位置：`code/vllm/vllm/model_executor/model_loader/utils.py:101` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:127`

这一步用于：

```text
- repack 权重到 kernel 需要的布局；
- online quantization；
- 删除临时 scale 参数；
- 初始化 CPU / GPU 特定 dispatch；
- 为 reload / offload 保留必要元数据。
```

随后 Attention / MLA / MM encoder attention 也会执行自己的 post-load 逻辑，HPC module 也会在这里统一处理：

```python
module.process_weights_after_loading(model_config.dtype)
```

源码位置：`code/vllm/vllm/model_executor/model_loader/utils.py:118` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:136`

因此量化完整生命周期是：

```text
VllmConfig.quant_config
  → 模型 __init__ 传入 quant_config
  → LinearBase / Attention 获取 quant_method
  → quant_method.create_weights()
  → load_weights() / weight_loader 加载 checkpoint
  → quant_method.process_weights_after_loading()
  → forward 中 quant_method.apply()
```

---

## 4. LoRA hook 的整体链路

LoRA 的核心不是模型 forward 手写 `x + BAx`，而是运行时把支持 LoRA 的 layer 替换为 LoRA wrapper。

入口在 `LoRAModelManager`。

源码位置：`code/vllm/vllm/lora/model_manager.py:71`

### 4.1 模型如何声明支持 LoRA

模型类需要满足 `SupportsLoRA`。

接口位置：`code/vllm/vllm/model_executor/models/interfaces.py:538`

关键字段：

```text
supports_lora = True
is_3d_moe_weight
is_non_gated_moe
embedding_modules
packed_modules_mapping
lora_skip_prefixes
lora_manager
```

以 LLaMA 为例：

```python
class LlamaForCausalLM(..., SupportsLoRA, ...):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:446`

这告诉 LoRA 系统：

```text
- 这个模型允许被 LoRA manager 包装；
- qkv_proj 是运行时 fused 层，但 adapter 里可能叫 q_proj / k_proj / v_proj；
- gate_up_proj 是运行时 fused 层，但 adapter 里可能叫 gate_proj / up_proj；
- embedding 和 lm_head 需要特殊处理。
```

### 4.2 LoRA manager 如何发现可替换模块

`LoRAModelManager.__init__()` 会先计算：

```python
self.supported_lora_modules = get_supported_lora_modules(self.model)
self.packed_modules_mapping = process_packed_modules_mapping(self.model, ...)
self._create_lora_modules()
```

源码位置：

- `code/vllm/vllm/lora/model_manager.py:96`
- `code/vllm/vllm/lora/model_manager.py:131`
- `code/vllm/vllm/lora/model_manager.py:139`

`get_supported_lora_modules()` 会扫描模型：

```text
- embedding_modules 中声明的模块；
- 所有 LinearBase 子类；
- MoERunner；
```

源码位置：`code/vllm/vllm/lora/utils.py:219`

这意味着：

```text
模型只要使用 vLLM 的 LinearBase 系列层，大多数 linear LoRA target 就能自动被发现。
```

### 4.3 LoRA wrapper 如何替换原 layer

`_create_lora_modules()` 会遍历 `model.named_modules()`：

```text
1. 跳过 PPMissingLayer；
2. 判断模块名是否匹配 supported_lora_modules；
3. 判断是否被 LoRAConfig.target_modules 允许；
4. 根据 module 类型选择对应 LoRA wrapper；
5. replace_submodule() 替换原模块；
6. register_module() 记录到 manager；
7. set_mapping() 注入 punica_wrapper。
```

源码位置：`code/vllm/vllm/lora/model_manager.py:382`

真正选择 wrapper 的函数是：

```python
from_layer(module, max_loras, lora_config, packed_moduled_lst, self.model.config)
```

源码位置：`code/vllm/vllm/lora/utils.py:106`

`from_layer()` 会按顺序尝试多个 LoRA wrapper：

```text
VocabParallelEmbeddingWithLoRA
ColumnParallelLinearWithLoRA
MergedColumnParallelLinearWithLoRA
QKVParallelLinearWithLoRA
MergedQKVParallelLinearWithLoRA
RowParallelLinearWithLoRA
ReplicatedLinearWithLoRA
LogitsProcessorWithLoRA
ColumnParallelLinearWithShardedLoRA
QKVParallelLinearWithShardedLoRA
MergedColumnParallelLinearWithShardedLoRA
RowParallelLinearWithShardedLoRA
FusedMoEWithLoRA
FusedMoE3DWithLoRA
```

源码位置：`code/vllm/vllm/lora/utils.py:76`

每个 wrapper 通过 `can_replace_layer()` 判断自己是否能替换当前 layer。

源码位置：`code/vllm/vllm/lora/layers/base.py:69`

### 4.4 LoRA wrapper 如何保留 base quantization

以 linear LoRA wrapper 为例。

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:70`

`BaseLinearLayerWithLoRA` 持有原始 layer：

```python
self.base_layer = base_layer
self.tp_size = self.base_layer.tp_size
self.tp_rank = self.base_layer.tp_rank
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:71`

forward 时先走 base layer 的量化方法：

```python
output = self._get_quant_method().apply(self.base_layer, x, bias)
return self._apply_lora_to_output(x, output)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:204`

这点非常关键：

```text
LoRA wrapper 不会绕开量化；
它先调用 base_layer.quant_method.apply() 得到 base output，
再用 LoRA A/B 对 output 做增量修正。
```

所以 LoRA 和量化可以组合：

```text
quantized base weight forward
  + active LoRA delta
  → final output
```

### 4.5 LoRA 权重如何按 TP 切分

`create_lora_weights()` 会根据 base layer 类型决定 LoRA A/B buffer 的形状。

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:100`

核心规则：

```text
ReplicatedLinear：
  A/B 都按完整形态保存。

ColumnParallelLinear：
  B 的输出维跟随 column parallel 的本地 output；
  fully_sharded_loras 时 A 的 rank 维也会按 TP 切。

RowParallelLinear：
  A 输入侧匹配当前输入分片；
  fully_sharded_loras 时 B 输出侧也会按 TP 切。
```

实际拷贝 adapter 时：

```python
if self.tp_size > 1:
    lora_a = self.slice_lora_a(lora_a)
    lora_b = self.slice_lora_b(lora_b)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:175`

这说明：

```text
LoRA wrapper 必须知道 base layer 的 TP rank / TP size，否则 adapter 权重无法正确对齐当前 rank。
```

### 4.6 LoRA adapter 如何加载和激活

LoRA checkpoint 先被加载为 `LoRAModel`。

源码位置：`code/vllm/vllm/lora/lora_model.py:60`

`LoRAModel.from_local_checkpoint()` 支持：

```text
adapter_model.safetensors
adapter_model.bin
adapter_model.pt
tensorizer adapter
```

源码位置：`code/vllm/vllm/lora/lora_model.py:166`

加载时会：

```text
- 校验 checkpoint 里是否出现非预期 target module；
- 解析 lora_A / lora_B 名称；
- 应用 weights_mapper；
- 跳过 base embedding weights；
- 按需跳过 lora_skip_prefixes；
- MoE + EP 时跳过非本 rank expert。
```

源码位置：`code/vllm/vllm/lora/lora_model.py:212`

真正激活时：

```python
module.set_lora(index, module_lora.lora_a, module_lora.lora_b)
```

源码位置：`code/vllm/vllm/lora/model_manager.py:292`

含义是：

```text
LoRA adapter 先注册在 CPU / manager cache；
激活时拷贝到某个 GPU LoRA slot；
batch 运行时通过 mapping 决定每个 token / request 使用哪个 LoRA slot。
```

### 4.7 LoRA mapping 如何作用到 forward

`LoRAModelManager.set_adapter_mapping()` 会把当前 batch 的 LoRA 映射写入 Punica wrapper：

```python
punica_wrapper.update_metadata(mapping, self.lora_index_to_id, self.lora_slots + 1, self.vocab_size)
```

源码位置：`code/vllm/vllm/lora/model_manager.py:351`

LoRA layer forward 时会调用：

```python
self.punica_wrapper.add_lora_linear(output, x, self.lora_a_stacked, self.lora_b_stacked, ...)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:227`

因此运行时链路是：

```text
Scheduler / ModelRunner 知道每个 request 的 LoRA id
  → LoRAMapping
  → LoRAModelManager.set_adapter_mapping()
  → PunicaWrapper.update_metadata()
  → LoRA wrapper forward
  → add_lora_linear 按 token / request 应用对应 adapter
```

### 4.8 fused module 的 LoRA 特殊点

LLaMA 里 `qkv_proj` 和 `gate_up_proj` 都是 fused runtime module。

但是 LoRA adapter 常常按 unfused 名称保存：

```text
q_proj / k_proj / v_proj
gate_proj / up_proj
```

所以 manager 会用 `packed_modules_mapping` 把多个 LoRA 子模块合并为一个 runtime module 的 packed LoRA。

源码位置：`code/vllm/vllm/lora/model_manager.py:718`

合并逻辑在：

```python
_create_merged_loras_inplace()
```

源码位置：`code/vllm/vllm/lora/model_manager.py:731`

这就是为什么模型类必须声明：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

否则 LoRA 无法知道 adapter 里的多个子 target 应该合并到哪个 fused runtime layer。

---

## 5. Tensor Parallel hook

Tensor Parallel 主要通过 vLLM 的并行 layer 接入，而不是模型 forward 手写 collective。

### 5.1 ColumnParallelLinear

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:394`

含义：

```text
Y = X A + b
A 按输出维切分：A = [A1, A2, ..., Ap]
每个 TP rank 只持有一段输出列。
```

构造时：

```python
self.output_size_per_partition = divide(output_size, self.tp_size)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:438`

forward 时：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:548`

常见用途：

```text
qkv_proj
gate_up_proj
部分 fused projection
```

### 5.2 RowParallelLinear

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:1491`

含义：

```text
Y = X A + b
A 按输入维切分；如果输入还没切，则先 split input；
每个 TP rank 算 partial output，最后 all-reduce。
```

构造时：

```python
self.input_size_per_partition = divide(input_size, self.tp_size)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:1546`

forward 时：

```python
if self.input_is_parallel:
    input_parallel = input_
else:
    input_parallel = split_tensor_along_last_dim(...)[self.tp_rank]

output_parallel = self.quant_method.apply(self, input_parallel, bias_)

if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:1628`

常见用途：

```text
o_proj
down_proj
```

### 5.3 QKVParallelLinear

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:914`

`QKVParallelLinear` 是 attention Q/K/V 的 fused + TP 版本。

它处理两个复杂点：

```text
1. q / k / v 在同一个 weight 中融合；
2. GQA / MQA 下 KV heads 可能少于 TP size，需要复制 KV heads。
```

构造时会计算：

```python
self.num_heads = divide(self.total_num_heads, tp_size)
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:966`

weight loading 时，`loaded_shard_id` 可以是：

```text
q
k
v
```

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:999`

这让模型加载可以把 HF 的：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

加载到 vLLM 的：

```text
qkv_proj.weight
```

### 5.4 MergedColumnParallelLinear

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:577`

它用于 MLP 中多个 column-parallel projection 的融合。

LLaMA 里对应：

```text
gate_proj + up_proj → gate_up_proj
```

加载时 `loaded_shard_id` 可以是：

```text
0：gate_proj
1：up_proj
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:345` 到 `code/vllm/vllm/model_executor/models/llama.py:354`

这样可以让运行时使用 fused GEMM，但 checkpoint 仍然按原始 HF 名称加载。

---

## 6. Pipeline Parallel hook

Pipeline Parallel 主要通过三类东西接入模型架构：

```text
1. SupportsPP 接口；
2. make_layers() 按 PP rank 创建本地层和 PPMissingLayer；
3. forward() 接收 / 返回 IntermediateTensors。
```

### 6.1 SupportsPP 接口

接口位置：`code/vllm/vllm/model_executor/models/interfaces.py:617`

模型需要：

```text
supports_pp = True
make_empty_intermediate_tensors(batch_size, dtype, device)
forward(..., intermediate_tensors=...)
```

`supports_pp()` 还会检查 forward 是否接受 `intermediate_tensors` 参数。

源码位置：`code/vllm/vllm/model_executor/models/interfaces.py:686`

### 6.2 make_layers 如何切分层

源码位置：`code/vllm/vllm/model_executor/models/utils.py:640`

`make_layers()` 会根据 PP rank 计算：

```python
start_layer, end_layer = get_pp_indices(num_hidden_layers, pp_rank, pp_world_size)
```

然后构造：

```text
[0, start_layer)         → PPMissingLayer
[start_layer, end_layer) → 当前 rank 真实 layers
[end_layer, N)           → PPMissingLayer
```

源码位置：`code/vllm/vllm/model_executor/models/utils.py:660`

这带来两个结果：

```text
- 每个 PP rank 只初始化自己负责的 transformer blocks；
- 模型的 module name 仍然保持完整层号，方便权重加载、LoRA、调试和映射。
```

### 6.3 PPMissingLayer 的意义

源码位置：`code/vllm/vllm/model_executor/models/utils.py:627`

`PPMissingLayer` 是一个占位层：

```python
def forward(self, *args, **kwargs):
    return args[0] if args else next(iter(kwargs.values()))
```

源码位置：`code/vllm/vllm/model_executor/models/utils.py:635`

它的作用是：

```text
- 让模型结构保持完整；
- 避免不属于当前 PP rank 的层分配真实权重；
- 在 LoRA manager 遍历模块时可以明确跳过；
- 在权重加载时可以跳过 missing parameter。
```

LoRA manager 会跳过它：

```python
if isinstance(module, PPMissingLayer):
    continue
```

源码位置：`code/vllm/vllm/lora/model_manager.py:392`

权重加载也会跳过缺失参数：

```python
if is_pp_missing_parameter(name, self):
    continue
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:464`

### 6.4 LLaMA forward 如何适配 PP

源码位置：`code/vllm/vllm/model_executor/models/llama.py:400`

PP first rank：

```python
if get_pp_group().is_first_rank:
    hidden_states = self.embed_input_ids(input_ids)
    residual = None
```

非 first rank：

```python
else:
    assert intermediate_tensors is not None
    hidden_states = intermediate_tensors["hidden_states"]
    residual = intermediate_tensors["residual"]
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:408` 到 `code/vllm/vllm/model_executor/models/llama.py:417`

非 last rank 返回：

```python
return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:430` 到 `code/vllm/vllm/model_executor/models/llama.py:433`

last rank 才执行最终 norm 并返回 hidden states：

```python
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:435` 到 `code/vllm/vllm/model_executor/models/llama.py:439`

因此 PP 模型 forward 的结构是：

```text
PP rank 0：
  input_ids / inputs_embeds → local layers → IntermediateTensors

PP rank 1...N-2：
  IntermediateTensors → local layers → IntermediateTensors

PP rank N-1：
  IntermediateTensors → local layers → norm → hidden_states → logits / pooling
```

### 6.5 lm_head / embed_tokens / norm 的 PP hook

LLaMA 中：

```python
if get_pp_group().is_first_rank:
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:373` 到 `code/vllm/vllm/model_executor/models/llama.py:382`

```python
if get_pp_group().is_last_rank:
    self.norm = RMSNorm(...)
else:
    self.norm = PPMissingLayer()
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:388` 到 `code/vllm/vllm/model_executor/models/llama.py:391`

```python
if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)
else:
    self.lm_head = PPMissingLayer()
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:484` 到 `code/vllm/vllm/model_executor/models/llama.py:499`

这说明：

```text
embedding 通常在 first PP rank；如果 tied word embeddings 且当前是 last rank，也会创建 embed_tokens 供 lm_head 共享；
norm / lm_head / logits_processor 通常只在 last PP rank；
中间 rank 只处理 transformer blocks 和 IntermediateTensors。
```

---

## 7. Expert Parallel / MoE hook

Expert Parallel 主要不是普通 dense 模型的路径，而是在 MoE 层和 LoRA MoE wrapper 中体现。

### 7.1 模型侧需要提供 MoE 映射

LoRA 对 MoE 的 packed mapping 来自：

```python
process_packed_modules_mapping(model)
```

源码位置：`code/vllm/vllm/lora/utils.py:371`

如果模型包含 `MoERunner`，它会要求模型能提供 expert mapping：

```python
if moe_packed_mapping := get_moe_expert_mapping(model):
    packed_modules_mapping = get_packed_modules_mapping(model)
    packed_modules_mapping["experts"] = [...]
else:
    raise AttributeError("To support LoRA for MoE model, 'get_expert_mapping' must be implemented")
```

源码位置：`code/vllm/vllm/lora/utils.py:374`

这表示：

```text
MoE 模型如果要支持 LoRA，需要告诉 vLLM：
checkpoint / adapter 中的 expert 权重名如何映射到运行时 fused expert module。
```

### 7.2 LoRA + EP 的加载优化

`LoRAModel.from_local_checkpoint()` 支持 `moe_ep_spec`。

源码位置：`code/vllm/vllm/lora/lora_model.py:166`

如果启用 EP，加载 safetensors 时可以跳过非本 rank 的 expert：

```python
if moe_ep_spec is not None and _is_remote_expert_key(module, moe_ep_spec):
    continue
```

源码位置：`code/vllm/vllm/lora/lora_model.py:273`

`LoRAModelManager` 会构造这个 spec：

```python
return MoEEPLoadSpec(ep_rank=module.ep_rank, local_num_experts=module.local_num_experts, global_num_experts=module.global_num_experts)
```

源码位置：`code/vllm/vllm/lora/model_manager.py:1091`

这说明：

```text
EP 不只是 forward 路径的并行；
LoRA adapter 加载阶段也会根据 EP rank 跳过非本地 expert，减少 CPU 内存和拷贝成本。
```

### 7.3 LoRA + MoE 的 2D / 3D 格式

`LoRAModelManager` 里有两类 MoE LoRA wrapper：

```text
FusedMoEWithLoRA
FusedMoE3DWithLoRA
```

源码位置：`code/vllm/vllm/lora/utils.py:93`

相关逻辑包括：

```text
- is_3d_moe_weight：模型声明是否使用 3D MoE weight layout；
- enable_mixed_moe_lora_format：允许 2D / 3D adapter 混用；
- _stack_moe_lora_weights()：把 3D MoE LoRA 权重整理到 wrapper 需要的格式；
- _convert_3d_to_2d_moe_lora()：混合模式下把 3D adapter 转成 2D pack layout；
- _slice_moe_lora_ep()：EP 下把 LoRA tensor 切到本 rank local experts。
```

源码位置：

- `code/vllm/vllm/lora/model_manager.py:120`
- `code/vllm/vllm/lora/model_manager.py:824`
- `code/vllm/vllm/lora/model_manager.py:919`
- `code/vllm/vllm/lora/model_manager.py:1012`

一句话：

```text
MoE / EP 的 hook 比 dense layer 多一层 expert mapping，因为 adapter 名称、运行时 expert 布局、本 rank local experts 三者必须对齐。
```

---

## 8. 权重加载如何把架构、量化、并行、LoRA 串起来

### 8.1 LLaMA 的 load_weights 映射

`LlamaModel` 现在通过 `hf_to_vllm_mapper` 声明 stacked 名称映射：

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_stacked={
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
)
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:345` 到 `code/vllm/vllm/model_executor/models/llama.py:354`

加载时交给 `AutoWeightsLoader` 应用 mapper：

```python
loader = AutoWeightsLoader(self)
return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:441` 到 `code/vllm/vllm/model_executor/models/llama.py:443`

这做了几件事：

```text
HF checkpoint name：model.layers.0.self_attn.q_proj.weight
  → vLLM runtime param：model.layers.0.self_attn.qkv_proj.weight
  → shard_id="q"
  → QKVParallelLinear.weight_loader(..., loaded_shard_id="q")
  → 当前 TP rank 只取自己负责的 q shard。
```

MLP 同理：

```text
HF checkpoint gate_proj / up_proj
  → vLLM runtime gate_up_proj
  → shard_id=0 / 1
  → MergedColumnParallelLinear.weight_loader()
```

### 8.2 为什么 prefix 很重要

模型构造 layer 时传入 `prefix`：

```python
prefix=f"{prefix}.self_attn.qkv_proj"
prefix=f"{prefix}.mlp.gate_up_proj"
prefix=maybe_prefix(prefix, "lm_head")
```

源码位置：

- `code/vllm/vllm/model_executor/models/llama.py:169`
- `code/vllm/vllm/model_executor/models/llama.py:98`
- `code/vllm/vllm/model_executor/models/llama.py:489`

`prefix` 影响：

```text
- quant_config.get_quant_method(layer, prefix) 判断该层是否量化；
- layer 参数名和 checkpoint 名称对齐；
- LoRA module_name 匹配；
- packed_modules_mapping 的父子模块合并；
- debug / tracing / compile static context 中的层名。
```

所以模型架构文件里的 `prefix` 不是装饰性字符串，而是各种 hook 对齐的主键之一。

### 8.3 AutoWeightsLoader 和模型自定义 loader

`LlamaForCausalLM.load_weights()` 使用：

```python
loader = AutoWeightsLoader(self, skip_prefixes=(['lm_head.'] if tie_word_embeddings else None))
return loader.load_weights(weights)
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:535` 到 `code/vllm/vllm/model_executor/models/llama.py:540`

而 `LlamaModel.load_weights()` 展示了 `WeightsMapper.orig_to_new_stacked` + `AutoWeightsLoader` 的模式。

这些方式本质都依赖参数或子模块上的 `weight_loader`：

```text
模型 loader 负责把 checkpoint name 映射到 vLLM param name；
具体 param 如何按 TP / quant / fused shard 加载，由 param.weight_loader 决定。
```

---

## 9. hook 之间的组合关系

### 9.1 quantization + TP

组合点在并行 linear layer：

```text
ColumnParallelLinear / RowParallelLinear / QKVParallelLinear
  → 根据 TP rank / TP size 决定本地参数形状
  → quant_method.create_weights() 按本地形状创建量化参数
  → weight_loader 按 TP + quant packing 切 checkpoint
  → quant_method.apply() 做本地 GEMM
  → all_gather 或 all_reduce 完成 TP 通信
```

也就是说：

```text
TP 先决定“本 rank 持有哪片矩阵”；
量化方法决定“这片矩阵以什么格式保存和计算”。
```

### 9.2 quantization + LoRA

组合点在 LoRA wrapper：

```text
BaseLinearLayerWithLoRA
  → 持有 base_layer
  → base_layer.quant_method.apply(base_layer, x, bias)
  → punica_wrapper.add_lora_linear(...)
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:204`

因此：

```text
base weight 可以是 GPTQ / AWQ / FP8 / compressed-tensors；
LoRA delta 通常以自己的 A/B buffer 保存；
forward 时先算量化 base output，再叠加 LoRA output。
```

### 9.3 LoRA + TP

组合点在 LoRA wrapper 的切分函数：

```text
slice_lora_a()
slice_lora_b()
fully_sharded_loras
base_layer.tp_rank / tp_size
```

源码位置：`code/vllm/vllm/lora/layers/base_linear.py:100`

不同 base layer 的切法不同：

```text
ColumnParallel：重点切 LoRA B 的输出维；
RowParallel：重点切 LoRA A 的输入维；
QKV / MergedColumn：还要按 q/k/v 或 gate/up 子 slice 处理；
Embedding / lm_head：还要处理 vocab parallel 和 extra vocab。
```

### 9.4 LoRA + PP

组合点在 `PPMissingLayer` 和 manager 遍历：

```text
- 每个 PP rank 只包装自己真实存在的模块；
- PPMissingLayer 会被跳过；
- 非 last rank 没有真实 lm_head，不会包装 lm_head LoRA；
- 非 first rank 没有真实 embed_tokens，不会包装 embedding LoRA。
```

源码位置：`code/vllm/vllm/lora/model_manager.py:392`

因此：

```text
LoRA adapter 在每个 PP rank 上只加载与本 rank layer 对应的权重。
```

### 9.5 TP + PP

组合方式是：

```text
PP 决定当前 rank 负责哪些 layers；
TP 决定这些 layers 内部每个 linear weight 如何切分。
```

在 LLaMA 中表现为：

```text
make_layers()：
  先按 PP 切 transformer block 范围。

每个真实 LlamaDecoderLayer：
  内部 qkv_proj / o_proj / gate_up_proj / down_proj 再按 TP 切权重。
```

### 9.6 quantization + Attention / KV cache

Attention 层也接收 `quant_config`：

```python
self.attn = Attention(..., quant_config=quant_config, ...)
```

源码位置：`code/vllm/vllm/model_executor/models/llama.py:218`

加载后 attention 会执行：

```python
module.process_weights_after_loading(model_config.dtype)
```

源码位置：`code/vllm/vllm/model_executor/model_loader/utils.py:118` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:127`

这主要服务：

```text
- KV cache scale；
- attention backend 需要的 post-load 参数；
- MLA / MM encoder attention 的特殊权重处理；
- 某些量化方法对 attention 相关 scale 的初始化。
```

---

## 10. 一个完整的初始化到 forward 链路

以量化 + LoRA + TP + PP 的 LLaMA serving 为例，可以串成：

```text
initialize_model(vllm_config)
  → set_current_vllm_config(vllm_config, ...)
  → LlamaForCausalLM(vllm_config, prefix)
  → LlamaModel(...)
  → make_layers(num_hidden_layers, layer_fn, prefix="model.layers")
  → 当前 PP rank 创建真实 LlamaDecoderLayer，其余位置 PPMissingLayer
  → LlamaAttention / LlamaMLP 构造 vLLM parallel linear layers
  → LinearBase 从 quant_config 获取 quant_method
  → quant_method.create_weights() 创建本 rank 的量化参数
  → model loader 加载 checkpoint
  → q/k/v、gate/up 名称映射到 qkv_proj、gate_up_proj
  → weight_loader 根据 TP rank、shard_id、quant packing 写入参数
  → process_weights_after_loading() 做量化后处理
  → create_lora_manager() 替换支持 LoRA 的真实模块
  → add_adapter() / activate_adapter() 加载 adapter 到 LoRA slots
  → set_adapter_mapping() 设置当前 batch 的 LoRA id 映射
  → model forward
  → quant_method.apply() 计算 base output
  → LoRA wrapper 叠加 adapter delta
  → TP collective 汇总必要输出
  → PP 非 last rank 返回 IntermediateTensors
  → PP last rank 返回 hidden_states
  → compute_logits() / pooler
```

这条链路解释了为什么模型架构里的每个 hook 都必须对齐：

```text
prefix 错了：quant / loader / LoRA 名称可能对不上；
packed_modules_mapping 错了：fused layer 的 adapter 或 checkpoint shard 可能合并失败；
PP missing 处理错了：某些 rank 会尝试加载或包装不存在的层；
TP shard 处理错了：权重形状、KV heads、LoRA A/B 切片会不一致；
quant_method 返回错了：create_weights / apply / post-load 会不匹配。
```

---

## 11. 架构文件需要做什么，不需要做什么

### 11.1 需要做的事

一个模型架构要正确接入这些能力，通常需要：

```text
1. 使用 vLLM 的 layer：
   QKVParallelLinear、MergedColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、ParallelLMHead、Attention 等。

2. 正确传递 quant_config：
   每个可量化 linear / embedding / attention 层都要传入 quant_config。

3. 正确传递 prefix：
   prefix 要和 checkpoint 名称、runtime module name、LoRA target name 保持可映射。

4. 声明 SupportsLoRA：
   包括 packed_modules_mapping、embedding_modules、必要时的 MoE expert mapping / skip prefixes。

5. 声明 SupportsPP：
   包括 make_empty_intermediate_tensors，forward 支持 intermediate_tensors。

6. 使用 make_layers：
   让 PP rank 只创建本地真实层，其余用 PPMissingLayer。

7. 实现 load_weights 映射：
   把 HF checkpoint 的 unfused 权重名映射到 vLLM runtime fused module，并传 shard_id。

8. 实现 compute_logits / pooler：
   last PP rank 上基于 hidden states 产生 logits 或 pooling output。
```

### 11.2 不需要做的事

模型架构通常不需要：

```text
- 手写 GPTQ / AWQ / FP8 kernel；
- 手写 LoRA A/B batch kernel；
- 手写 TP all-reduce / all-gather 的全部细节；
- 手写 PP 通信；
- 手写 LoRA adapter LRU cache；
- 手写量化权重 post-load repack；
- 在每个模型里重复实现通用 fused MoE LoRA 逻辑。
```

这些由通用组件负责。

---

## 12. 容易疑惑的点

### 12.1 quant_config 是模型级还是 layer 级？

入口是模型级 `vllm_config.quant_config`，但真正生效是 layer 级。

```text
同一个 quant_config 会被传给多个 layer；
每个 layer 再调用 get_quant_method(layer, prefix)；
量化配置可以根据 layer 类型和 prefix 做选择。
```

### 12.2 LoRA 是模型 forward 里显式调用的吗？

通常不是。

LoRA manager 会把原始 layer 替换成 LoRA wrapper。

模型 forward 仍然调用：

```text
self.qkv_proj(...)
self.gate_up_proj(...)
self.down_proj(...)
```

只是这些模块已经被替换为支持 LoRA 的 wrapper。

### 12.3 LoRA 会不会绕过量化？

不会。

linear LoRA wrapper 会先调用：

```text
base_layer.quant_method.apply(base_layer, x, bias)
```

再叠加 LoRA delta。

### 12.4 packed_modules_mapping 是给量化用的吗？

主要不是。

它主要给 LoRA 和模型权重映射使用，用来说明：

```text
运行时 fused module 名称 → adapter / checkpoint 中的 unfused 子模块名称
```

但 fused module 的 weight_loader 同时也会处理量化 packed 参数，所以在实际加载链路里它们会交汇。

### 12.5 PP 下为什么还保留完整 layers 列表？

因为完整层号对权重名、LoRA target、调试和映射都很重要。

`make_layers()` 用 `PPMissingLayer` 占位，使每个 rank 的 module tree 仍然能对应全局层号。

### 12.6 PP 下每个 rank 都有 lm_head 吗？

通常不是。

LLaMA 里只有 last PP rank 创建真实 `lm_head`，其他 rank 是 `PPMissingLayer`。

### 12.7 TP 和 PP 谁先切？

逻辑上：

```text
PP 先决定当前 rank 有哪些层；
TP 再决定这些层内部的矩阵如何分片。
```

### 12.8 QKVParallelLinear 为什么要处理 KV head replication？

GQA / MQA 模型里 KV heads 可能少于 TP size。

当 `tp_size >= total_num_kv_heads` 时，KV heads 无法继续均分，只能在多个 TP rank 上复制。

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:968`

### 12.9 为什么量化权重加载要处理 packed_factor？

量化 checkpoint 里的权重可能把多个低 bit 值打包到一个存储单元里。

因此原始 shard size / offset 不能直接用于 qweight，需要按 `packed_factor` 修正。

源码位置：`code/vllm/vllm/model_executor/layers/linear.py:720`

### 12.10 为什么 prefix 对 LoRA 和量化都重要？

因为它同时参与：

```text
quant_config.get_quant_method(layer, prefix)
module.named_modules() 的 LoRA 匹配
checkpoint name 到 runtime param name 的映射
compile / tracing / debug 的层名标识
```

---

## 13. 推荐阅读路线

### 13.1 先看模型侧 hook

```text
model_executor/models/llama.py
  → LlamaForCausalLM
  → packed_modules_mapping / embedding_modules
  → LlamaModel.make_layers
  → LlamaAttention / LlamaMLP
  → load_weights
```

### 13.2 再看量化 hook

```text
model_executor/layers/quantization/base_config.py
  → QuantizationConfig / QuantizeMethodBase
  → model_executor/layers/linear.py
  → LinearBase
  → ColumnParallelLinear / RowParallelLinear / QKVParallelLinear
  → model_executor/model_loader/utils.py
```

### 13.3 再看 LoRA hook

```text
model_executor/models/interfaces.py
  → SupportsLoRA
  → lora/model_manager.py
  → lora/utils.py
  → lora/layers/base_linear.py
  → lora/lora_model.py
```

### 13.4 最后看并行 hook

```text
model_executor/models/interfaces.py
  → SupportsPP
  → model_executor/models/utils.py
  → make_layers / PPMissingLayer
  → distributed/parallel_state.py
  → model_executor/layers/linear.py
```

---

## 14. 总结

vLLM 的模型架构层不是把量化、LoRA、并行都硬编码进每个模型，而是通过一组稳定 hook 把模型结构接到通用运行时：

```text
quantization：
  quant_config → quant_method → create_weights / weight_loader / process_weights_after_loading / apply

LoRA：
  SupportsLoRA → packed_modules_mapping / embedding_modules → LoRAModelManager → LoRA wrapper → Punica mapping

Tensor Parallel：
  QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear → TP shard → gather / all-reduce

Pipeline Parallel：
  SupportsPP → make_layers → PPMissingLayer → IntermediateTensors

Expert Parallel：
  MoE expert mapping → local expert slicing → FusedMoE LoRA wrapper / EP-aware loading
```

压缩成一句话：

```text
模型架构负责把“模型语义上的层”映射成“vLLM 运行时可量化、可 LoRA、可 TP/PP/EP 的层”；真正的 kernel、adapter 管理、分布式通信和权重后处理都由通用组件接管。
```
