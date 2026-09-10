# 11. 模型权重如何加载和做名称映射？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_model_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\model_loader\`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\parameter.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\linear.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\fused_moe\`

本问题关注：model class 如何实现 `load_weights()`，checkpoint 权重名如何映射到 vLLM 内部 fused / parallel / quantized layer，以及为什么同一个 checkpoint tensor 在 vLLM 中可能进入不同形态的参数。

---

## 1. 一句话回答

vLLM 的权重加载不是简单的 `state_dict[name] -> param[name]`。

它实际做的是：

```text
checkpoint tensor
  → 权重文件迭代器
  → model.load_weights()
  → 名称映射 / fused mapping / skip rules
  → parameter.weight_loader
  → TP shard / packed slice / quant scale / expert mapping
  → 当前 rank 的 param.data
```

所以：

```text
model.load_weights() 负责“把外部 checkpoint 名称解释成 vLLM 模型参数”；
parameter.weight_loader 负责“把这个 tensor 的正确切片写入当前 rank 参数”。
```

---

## 2. 最小主链路

从 GPU Worker 角度看，模型加载发生在 Worker 生命周期中的 `load_model()` 阶段。

主链路可以压缩成：

```text
GPUWorker.load_model()
  → GPUModelRunner.load_model()
  → get_model_loader(load_config)
  → model_loader.load_model(vllm_config, model_config)
  → initialize_model(vllm_config)
  → model_loader.load_weights(model, model_config)
  → model.load_weights(weights)
  → name mapping / skip / packed mapping
  → param.weight_loader(param, loaded_weight, shard_id)
  → param.data.copy_() 或加载当前 rank shard
  → process_weights_after_loading(model, vllm_config)
```

其中最关键的边界是：

```text
ModelLoader：负责选择权重来源、下载 / 迭代 checkpoint、创建模型；
Model class：负责解释 checkpoint 名称和 vLLM 参数名称的关系；
Parameter / Layer：负责真正切片、拼接、解包、写入参数。
```

---

## 3. 加载发生在生命周期的哪一步

在 Worker 生命周期里，顺序大致是：

```text
init_device()
  → 创建 ModelRunner
load_model()
  → 创建模型结构
  → 加载权重
initialize_from_config()
  → 初始化 KV cache
compile_or_warm_up_model()
  → warmup / CUDA graph / compile
execute_model()
  → 正常推理
```

也就是说：

```text
权重加载早于 KV cache 初始化，
也早于 warmup / cudagraph capture。
```

原因是：

```text
1. warmup 需要真实模型参数；
2. memory profiling 需要知道 weights 占用；
3. KV cache 能分多少依赖权重加载后的显存状态；
4. TP / PP / quantization 都需要在模型构造和加载阶段确定。
```

---

## 4. get_model_loader() 如何选择加载器

入口在 `model_executor/model_loader/__init__.py`。

核心函数：

```python
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
```

它会根据 `load_config.load_format` 选择不同 loader。

常见 loader 包括：

```text
DefaultModelLoader：
  默认 HF / safetensors / pt 权重加载路径。

BitsAndBytesModelLoader：
  bitsandbytes 量化模型加载路径。

ShardedStateLoader：
  vLLM 保存的 sharded state 加载路径。

TensorizerLoader：
  tensorizer 格式加载路径。

RunaiModelStreamerLoader：
  流式加载路径。

ModelExpressModelLoader：
  ModelExpress 格式加载路径。

DummyModelLoader：
  dummy weights / profile / 测试场景。
```

所以 `load_format` 决定的是：

```text
权重从哪里来，怎么迭代，是否需要特殊格式适配。
```

但它不直接决定：

```text
q_proj 如何进 qkv_proj；
gate_proj 如何进 gate_up_proj；
TP shard 如何切；
MoE expert 如何映射。
```

这些通常在 model class 和 parameter loader 层完成。

---

## 5. BaseModelLoader.load_model() 的模板流程

`BaseModelLoader.load_model()` 是一个模板方法。

它做的事情可以理解为：

```text
1. 创建空模型结构；
2. 调用具体 loader 加载权重；
3. 对在线量化做 finalize；
4. 对模型做权重加载后的处理；
5. 返回 eval() 模型。
```

简化后是：

```text
initialize_model(vllm_config)
  → self.load_weights(model, model_config)
  → finalize_layerwise_processing(model, model_config)  # online quant
  → process_weights_after_loading(model, model_config, target_device)
  → model.eval()
```

这里有一个关键点：

```text
initialize_model() 只创建参数对象，
load_weights() 才把 checkpoint tensor 写进去。
```

因此阅读模型结构时会看到很多 vLLM 自定义 layer，例如：

```text
VocabParallelEmbedding
QKVParallelLinear
MergedColumnParallelLinear
RowParallelLinear
ReplicatedLinear
RoutedExperts / FusedMoE
```

这些 layer 创建出的参数并不一定和 checkpoint 名称一一对应。

---

## 6. DefaultModelLoader.load_weights() 做什么

默认加载器的职责是把 checkpoint 文件变成 `(name, tensor)` 迭代器，然后交给模型。

主流程是：

```text
DefaultModelLoader.load_weights(model, model_config)
  → 准备 maybe_download / revision / fall_back_to_pt
  → 初始化可选 EP weight filter
  → get_all_weights(model_config, model)
  → model.load_weights(weights)
  → 记录 loaded_weights
  → 按 enable_weights_track 决定是否检查 missing / unexpected weights
```

其中：

```text
weights: Iterable[tuple[str, torch.Tensor]]
```

每个元素代表 checkpoint 中的一个 tensor。

此时 tensor name 仍然通常是 checkpoint 原名，例如：

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.embed_tokens.weight
lm_head.weight
```

这些名字不一定存在于 vLLM 模型参数里。

---

## 7. model.load_weights() 的职责边界

每个 vLLM model class 通常都会实现：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

它返回：

```text
set[str]
```

表示成功加载的 vLLM 参数名集合。

它的职责是：

```text
1. 遍历 checkpoint 权重；
2. 跳过不需要加载的权重；
3. 把 checkpoint 名称映射成 vLLM 内部参数名；
4. 为 fused 参数传入 shard_id；
5. 调用参数上的 weight_loader；
6. 返回 loaded_params。
```

一个典型手写逻辑是：

```text
for name, loaded_weight in weights:
  if name should be skipped:
    continue

  for param_name, weight_name, shard_id in stacked_params_mapping:
    if weight_name in name:
      name = name.replace(weight_name, param_name)
      param = params_dict[name]
      weight_loader = param.weight_loader
      weight_loader(param, loaded_weight, shard_id)
      break
  else:
    param = params_dict[name]
    weight_loader = param.weight_loader
    weight_loader(param, loaded_weight)
```

这就是很多模型里 `q_proj/k_proj/v_proj`、`gate_proj/up_proj` 映射的来源。

---

## 8. 为什么需要名称映射

因为 HuggingFace checkpoint 的模块结构通常是“训练友好”的，而 vLLM 的模块结构是“推理友好”的。

常见差异包括：

```text
HF checkpoint：
  q_proj.weight
  k_proj.weight
  v_proj.weight

vLLM：
  qkv_proj.weight
```

```text
HF checkpoint：
  gate_proj.weight
  up_proj.weight

vLLM：
  gate_up_proj.weight
```

```text
HF checkpoint：
  experts.0.gate_proj.weight
  experts.0.up_proj.weight
  experts.0.down_proj.weight

vLLM：
  experts.w13_weight
  experts.w2_weight
```

vLLM 这样做的原因是：

```text
1. 减少推理时 kernel 调用次数；
2. 适配 fused GEMM / fused MoE kernel；
3. 让 TP shard 更直接；
4. 适配量化权重的 packed 布局；
5. 统一不同模型架构的 serving 运行时。
```

---

## 9. AutoWeightsLoader：自动递归加载

有些模型不手写完整 `load_weights()`，而是使用 `AutoWeightsLoader`。

典型形式是：

```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(self, skip_prefixes=[...])
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

`AutoWeightsLoader` 位于 `model_executor/models/utils.py`。

它的核心思路是：

```text
checkpoint name
  → 可选 WeightsMapper 重写，并可把 stacked shard_id 写到 tensor.shard_id
  → 按 . 分段递归查找 module / parameter
  → 如果子模块有 load_weights()，下放给子模块
  → 如果参数有 weight_loader，调用参数自定义 loader
  → MergedColumnParallelLinear / QKVParallelLinear 等子模块可读取 tensor.shard_id
  → 否则使用 default_weight_loader
```

它适合：

```text
1. checkpoint 名称和 vLLM 模型结构大体一致；
2. 名称可通过 mapper 修正，包括 stacked qkv / gate_up 映射；
3. 子模块 / 参数已经定义好了自己的加载逻辑；
4. 复杂 MoE、特殊多模态前缀或 tied 权重仍可能需要外层手写过滤。
```

它不等于完全“无映射”。

即使用 AutoWeightsLoader，也可能通过：

```text
skip_prefixes
ignore_unexpected_prefixes
WeightsMapper
param.weight_loader
```

完成跳过、重命名和特殊加载。

---

## 10. WeightsMapper：checkpoint 名称重写

`WeightsMapper` 也是 `model_executor/models/utils.py` 里的工具。

它负责在自动加载前重写 checkpoint 名称。

常见能力包括：

```text
orig_to_new_prefix
orig_to_new_substr
orig_to_new_regex
orig_to_new_suffix
orig_to_new_renaming
orig_to_new_stacked：把原始 q/k/v 或 gate/up 名称映射到 fused 名称，并携带 shard_id
```

可以把它理解为：

```text
checkpoint name 进入参数查找前的“名字归一化层”。
```

例如某些模型 checkpoint 名称可能多一层：

```text
transformer.h.0...
```

而 vLLM 模型内部是：

```text
model.layers.0...
```

这类差异就可以通过 mapper 处理。

需要注意：

```text
WeightsMapper 不负责 tensor 切片、融合、量化解包；
但 orig_to_new_stacked 可以在改名时把 shard_id 挂到 loaded tensor 上。
```

真正写入 tensor 仍然要靠参数或子模块的 `weight_loader`。

---

## 11. packed_modules_mapping 是什么

很多 model class 上会定义：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

它表达的是：

```text
vLLM 内部一个 packed / fused module，
对应 checkpoint 里的多个原始 module。
```

典型映射：

```text
q_proj + k_proj + v_proj → qkv_proj
gate_proj + up_proj → gate_up_proj
```

它常被用于：

```text
1. 量化配置识别 fused module；
2. LoRA / adapter 相关逻辑把外部 target module 对齐到 fused module；
3. 模型保存 / 加载 / 工具层理解 packed 模块关系；
4. 判断某个原始 module 是否属于 fused 参数。
```

容易误解的是：

```text
packed_modules_mapping 本身通常不是最终 copy tensor 的代码。
```

实际 copy 可能发生在：

```text
1. model.load_weights() 手写 name.replace + shard_id；
2. AutoWeightsLoader 递归调用参数 weight_loader；
3. quantization method 创建的特殊参数 loader；
4. MoE expert 子模块的 load_weights()。
```

---

## 12. q_proj / k_proj / v_proj 如何进入 qkv_proj

以 decoder-only transformer 为例，HF checkpoint 常见结构是：

```text
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
```

vLLM 内部通常是：

```text
self_attn.qkv_proj.weight
```

加载时会把三份 checkpoint tensor 写入同一个参数的不同 shard。

逻辑可以表示为：

```text
checkpoint: q_proj.weight
  → vLLM param: qkv_proj.weight
  → shard_id = "q"

checkpoint: k_proj.weight
  → vLLM param: qkv_proj.weight
  → shard_id = "k"

checkpoint: v_proj.weight
  → vLLM param: qkv_proj.weight
  → shard_id = "v"
```

主过程是：

```text
name.replace("q_proj", "qkv_proj")
param = params_dict["...qkv_proj.weight"]
param.weight_loader(param, loaded_weight, "q")
```

对于 `QKVParallelLinear`，`weight_loader` 还会处理：

```text
1. 当前 TP rank 应该加载哪个 shard；
2. q/k/v 在 fused weight 中的 offset；
3. GQA / MQA 下 KV head 是否需要复制或切片；
4. 输出维度 packed 后的形状；
5. 量化权重的 packed_dim / packed_factor。
```

所以不要把 qkv 映射理解成简单拼接。

更准确地说：

```text
model.load_weights() 决定“这是 q shard / k shard / v shard”；
QKVParallelLinear 的参数 loader 决定“这个 shard 在当前 rank 的 fused tensor 中写到哪里”。
```

---

## 13. gate_proj / up_proj 如何进入 gate_up_proj

MLP 中常见 checkpoint 结构是：

```text
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

vLLM 内部通常是：

```text
mlp.gate_up_proj.weight
mlp.down_proj.weight
```

其中：

```text
gate_proj + up_proj → gate_up_proj
down_proj 保持单独 RowParallelLinear
```

加载映射可以表示为：

```text
checkpoint: gate_proj.weight
  → vLLM param: gate_up_proj.weight
  → shard_id = 0

checkpoint: up_proj.weight
  → vLLM param: gate_up_proj.weight
  → shard_id = 1
```

`MergedColumnParallelLinear` 会知道：

```text
1. gate 和 up 在 fused 输出维度上的位置；
2. 每个 TP rank 只加载自己负责的 output shard；
3. 量化 packed weight 需要怎样对齐；
4. bias / scale / zero point 等附属参数如何加载。
```

因此：

```text
qkv 的 shard_id 通常是 "q" / "k" / "v"；
gate_up 的 shard_id 通常是 0 / 1。
```

不要把不同 layer 的 `shard_id` 当成统一语义。

---

## 14. TP shard 在哪里发生

Tensor Parallel 下，每个 rank 只加载当前 rank 需要的权重切片。

这个切片通常不是 model class 手写完成的，而是在 parameter / layer 的 `weight_loader` 中完成。

典型参数类型包括：

```text
ModelWeightParameter：普通模型权重；
ColumnParallelLinear 参数：按输出维度切；
RowParallelLinear 参数：按输入维度切；
QKVParallelLinear 参数：按 q/k/v + TP 规则切；
MergedColumnParallelLinear 参数：按 merged output shard 切；
VocabParallelEmbedding 参数：按 vocab shard 切；
PackedvLLMParameter：按 packed 维度和 packed factor 切；
BlockQuantScaleParameter：按 block scale 布局切。
```

可以把职责分成两层：

```text
model.load_weights():
  把 checkpoint name 映射成 param name，并传 shard_id。

param.weight_loader():
  根据 TP rank、param metadata、quant metadata，把 tensor 的正确部分写入 param.data。
```

例如：

```text
checkpoint q_proj.weight 是完整 q projection；
当前 rank 只需要 q projection 的某个 output shard；
weight_loader 会 slice loaded_weight 后写入 qkv_proj.weight 的 q 区域。
```

所以阅读 `load_weights()` 时，如果只看到：

```python
weight_loader(param, loaded_weight, shard_id)
```

并不表示整块 `loaded_weight` 都 copy 进来了。

---

## 15. parameter.py 中的参数类型

`model_executor/parameter.py` 定义了 vLLM 自己的参数类型。

这些参数继承自 `torch.nn.Parameter`，但带有额外属性，例如：

```text
input_dim
output_dim
packed_dim
packed_factor
weight_loader
marlin_tile_size
tp_rank / tp_size
```

常见类型可以这样理解：

```text
BasevLLMParameter：
  vLLM 参数基类，保存加载相关 metadata 和当前 TP 信息。

ModelWeightParameter：
  普通模型权重，组合 column / row parallel 加载能力。

RowvLLMParameter：
  RowParallelLinear 相关参数，按 input dim shard。

PackedColumnParameter：
  packed-on-disk 且只走 column parallel 的参数。

PackedvLLMParameter：
  packed / quantized 参数，加载时要考虑 packed_dim、packed_factor 和 marlin tile。

PerTensorScaleParameter：
  fused linear 的 per-tensor scale 参数，可按 shard_id 写入对应 scale。

ChannelQuantScaleParameter / GroupQuantScaleParameter：
  channel-wise / group-wise quant scale 参数。

BlockQuantScaleParameter：
  block quantization scale 参数，按 block 布局加载。

SharedWeightParameter：
  支持多个 partition 共享底层 tensor 的特殊参数。
```

这些参数存在的意义是：

```text
把“当前参数应该如何从 checkpoint tensor 中取数据”这件事，
从 model.load_weights() 下沉到参数本身。
```

---

## 16. default_weight_loader 只是最简单路径

最简单的加载器是默认权重加载：

```text
检查 shape
  → param.data.copy_(loaded_weight)
```

它适合：

```text
1. 参数名完全一致；
2. 参数形状完全一致；
3. 不需要 TP shard；
4. 不需要 fused 写入；
5. 不需要量化 packed 处理。
```

但大模型 serving 中大量参数都不是这个情况。

因此更常见的是：

```text
LinearBase / ParallelLinear 自己注册 weight_loader；
Quantization method 创建特殊参数和 loader；
MoE module 自己实现 expert weight_loader；
Embedding / lm_head 使用 vocab parallel loader。
```

也就是说：

```text
default_weight_loader 是兜底，不是主角。
```

---

## 17. Linear 层的加载逻辑

`model_executor/layers/linear.py` 是理解权重加载最重要的文件之一。

常见 layer：

```text
ReplicatedLinear：
  每个 rank 都有完整权重。

ColumnParallelLinear：
  按输出维度切分权重。

MergedColumnParallelLinear：
  多个 ColumnParallelLinear 融合后再按输出维度切分。

RowParallelLinear：
  按输入维度切分权重。

QKVParallelLinear：
  q/k/v 融合，并处理 GQA / MQA / TP 下的 head 分布。
```

加载时的核心问题是：

```text
checkpoint tensor 的维度，
和当前 rank 的 param.data 维度，
通常不一样。
```

因此 `weight_loader` 要计算：

```text
1. loaded_weight 的 shard_size；
2. 当前 rank 的 shard_offset；
3. fused 参数中目标区域 offset；
4. 是否需要 narrow / slice；
5. 是否需要处理 packed factor；
6. 是否需要处理 bitsandbytes / marlin / block quant scale。
```

---

## 18. QKVParallelLinear 的特殊性

`QKVParallelLinear` 不只是把三个矩阵拼起来。

它还要处理 attention head 的并行分布。

典型配置包括：

```text
num_heads：总 query heads；
num_kv_heads：总 key/value heads；
head_size：每个 head 的维度；
tensor_parallel_size：TP world size；
```

在 MHA / GQA / MQA 下，情况不同：

```text
MHA：
  num_kv_heads == num_heads，q/k/v head 数相同。

GQA：
  num_kv_heads < num_heads，多个 q head 共享一组 kv head。

MQA：
  num_kv_heads 更少，甚至接近 1。
```

因此 q/k/v 的 shard 规则不是完全一样。

加载时要回答：

```text
当前 TP rank 拿哪些 q heads？
当前 TP rank 拿哪些 kv heads？
如果 kv heads 少于 TP ranks，是否复制？
q/k/v 在 fused qkv_proj.weight 中的 offset 是多少？
```

所以文档中看到：

```text
q_proj / k_proj / v_proj → qkv_proj
```

只是名称层面的最小心智模型。

完整心智模型应该是：

```text
q/k/v checkpoint tensor
  → 手写 load_weights 或 WeightsMapper.orig_to_new_stacked 标记 shard_id
  → QKVParallelLinear 依据 TP + head layout 计算本 rank slice
  → 写入 qkv_proj fused tensor 对应区域
```

---

## 19. Embedding 和 lm_head 的加载

Embedding / lm_head 有两个常见特殊点。

### 19.1 Vocab parallel

vLLM 通常使用：

```text
VocabParallelEmbedding
ParallelLMHead
```

这意味着 vocab 维度可能被 TP 切分。

加载时要处理：

```text
1. 当前 rank 对应 vocab range；
2. padding vocab size；
3. org_vocab_size 和 padded_vocab_size；
4. added tokens；
5. tied embedding 下 lm_head 是否共享 embed_tokens。
```

### 19.2 tie_word_embeddings

很多 causal LM 会设置：

```text
tie_word_embeddings = True
```

此时：

```text
lm_head.weight 和 embed_tokens.weight 共享同一份权重。
```

vLLM 中常见做法是：

```text
1. 构造 lm_head；
2. 调用 tie_weights 或直接令 lm_head 指向 embed_tokens；
3. load_weights 时跳过 lm_head.weight；
4. 避免同一份权重重复加载。
```

所以看到 `lm_head.weight` 被 skip，不一定是 bug。

它可能表示：

```text
lm_head 已经和 embedding tied，
加载 embed_tokens.weight 就够了。
```

---

## 20. 量化权重如何参与加载

量化模型的权重加载比 fp16 / bf16 更复杂。

原因是 checkpoint 中可能不再只有：

```text
weight
bias
```

还会有：

```text
qweight
scales
zeros
g_idx
qzeros
input_scale
weight_scale
kv_cache_scale
```

并且这些 tensor 可能是 packed 布局。

vLLM 的处理方式是：

```text
1. quantization config 决定每个 layer 使用的 quant method；
2. quant method 在 create_weights() 中创建特殊 Parameter；
3. 参数带有 packed_dim / packed_factor / weight_loader；
4. load_weights() 名称映射后仍调用 param.weight_loader；
5. loader 根据 packed 规则加载 qweight / scales / zeros。
```

这意味着：

```text
量化权重不是先还原成 fp16 再加载；
很多时候是直接加载到推理 kernel 需要的 packed 参数格式。
```

---

## 21. 量化名称映射的特殊点

量化配置还需要知道模型里的 fused module。

例如：

```text
q_proj / k_proj / v_proj → qkv_proj
gate_proj / up_proj → gate_up_proj
```

如果量化后端不知道这个关系，就可能错误地查找：

```text
q_proj.qweight
k_proj.qweight
v_proj.qweight
```

但 vLLM 实际参数可能是：

```text
qkv_proj.qweight
qkv_proj.scales
gate_up_proj.qweight
gate_up_proj.scales
```

因此模型构造和量化配置初始化阶段会使用模型的：

```text
hf_to_vllm_mapper
packed_modules_mapping
```

让 quant config 看到 HF 名称到 vLLM fused module 的关系。

这样量化后端才能理解：

```text
HF 名称、vLLM 名称、fused module、量化参数名称之间的关系。
```

---

## 22. KV cache scale 也是权重吗

有些量化方法会把 KV cache scale 存在 checkpoint 里。

例如可能出现：

```text
self_attn.k_scale
self_attn.v_scale
self_attn.kv_scale
```

这些不是普通模型矩阵，但加载阶段仍可能作为模型参数或 buffer 处理。

相关逻辑通常通过：

```text
QuantizationConfig.get_cache_scale_mapper()
AutoWeightsLoader 的 mapper
模型 / attention layer 的 scale 参数
```

完成名称适配。

容易混淆的是：

```text
KV cache scale 不是 KV cache 本身。
```

它表示：

```text
量化 KV cache 读写时需要的缩放因子，
加载时和权重一起进入模型，
运行时才影响 KV cache 的 quant/dequant。
```

---

## 23. MoE expert 权重如何映射

MoE 模型的 checkpoint 通常按 expert 展开：

```text
layers.0.mlp.experts.0.gate_proj.weight
layers.0.mlp.experts.0.up_proj.weight
layers.0.mlp.experts.0.down_proj.weight
layers.0.mlp.experts.1.gate_proj.weight
...
```

而 vLLM 的 fused MoE 通常会组织成：

```text
experts.w13_weight
experts.w2_weight
```

其中：

```text
w13_weight：gate / up fused 权重；
w2_weight：down 权重。
```

映射关系可以理解为：

```text
gate_proj / w1 → w13_weight 的 w1 区域
up_proj / w3   → w13_weight 的 w3 区域
down_proj / w2 → w2_weight
```

MoE 加载还要处理 expert id：

```text
checkpoint expert id
  → global expert id
  → 当前 EP rank 是否拥有这个 expert
  → local / physical expert id
  → fused expert tensor 中的位置
```

因此 expert 权重加载通常不只是名称替换。

还需要考虑：

```text
1. Expert Parallelism；
2. EPLB 重新布局；
3. 当前 rank 是否跳过非本地 expert；
4. fused MoE kernel 需要的 weight layout；
5. 量化 MoE 权重的 scale / zero point。
```

---

## 24. DefaultModelLoader 和 MoE expert filtering

在 Expert Parallel 场景下，如果 `enable_ep_weight_filter` 开启且没有启用 EPLB，默认加载器可能会提前过滤非本 rank 的 expert 权重。

原因是：

```text
checkpoint 中包含所有 experts，
但当前 rank 通常只需要其中一部分；
如果启用 EPLB，冗余 physical expert slot 可能还需要其他 logical expert 权重，加载器会跳过提前过滤。
```

如果不提前过滤，会导致：

```text
1. 不必要的磁盘 / CPU / GPU 传输；
2. params_dict 找不到当前 rank 不存在的 expert；
3. loaded_weights 检查误判；
4. 大 MoE 模型加载开销过高。
```

所以 MoE 权重加载要分两层看：

```text
ModelLoader 层：
  在满足 EP filter 条件时，可能过滤非本地 expert checkpoint tensor。

MoE module 层：
  把本地 expert tensor 写入 fused expert 参数。
```

这也是为什么：

```text
checkpoint expert index 不一定等于当前 rank 参数里的 expert index。
```

---

## 25. 多模态 / projector 权重

多模态模型还会有 vision tower、projector、resampler 等模块。

这些模块的 checkpoint 名称可能来自不同来源，例如：

```text
language_model.model.layers...
vision_tower.vision_model...
multi_modal_projector...
mm_projector...
visual...
```

vLLM 模型加载时可能需要：

```text
1. 给 language model 子模块加前缀或去前缀；
2. 跳过 HF 中 serving 不需要的模块；
3. 把 vision encoder 权重交给对应子模块；
4. 把 projector 权重映射到 vLLM 内部多模态 connector；
5. 处理 processor / config 中的 hidden size 差异。
```

这类模型通常更依赖：

```text
WeightsMapper
AutoWeightsLoader
子模块 load_weights()
ignore_unexpected_prefixes
```

所以多模态权重加载的核心问题不是 fused qkv，而是：

```text
不同子模型的命名空间如何对齐。
```

---

## 26. PP 场景下权重如何加载

Pipeline Parallel 会把层切到不同 PP rank。

这意味着当前 rank 只拥有一部分 transformer layers。

常见策略是：

```text
1. 构造模型时只创建当前 PP rank 负责的层；
2. load_weights 遍历 checkpoint 时跳过不属于当前 PP rank 的层；
3. first / last PP rank 额外负责 embedding / lm_head；
4. 中间 PP rank 只加载自己的 decoder layers。
```

所以 PP 下会看到一些权重被跳过。

这通常不是缺失，而是：

```text
该层属于其他 pipeline stage。
```

从职责看：

```text
PP 决定“哪些层存在于当前 rank”；
TP 决定“某一层的参数在当前 rank 上取哪一片”。
```

两者不要混淆。

---

## 27. loaded_weights 集合有什么用

`model.load_weights()` 通常返回 `loaded_params`。

它的作用是：

```text
告诉 ModelLoader 哪些 vLLM 参数已经从 checkpoint 成功加载。
```

随后 loader 可以检查：

```text
1. 是否有必要参数没加载；
2. 是否有 checkpoint 中的权重没人消费；
3. tied / skipped / ignored 权重是否合理；
4. quantization 相关参数是否缺失；
5. PP / EP 下跳过的参数是否符合预期。
```

但这个集合也有边界：

```text
一个 fused 参数可能由多个 checkpoint tensor 共同加载；
一个 checkpoint tensor 也可能因为 tied / skip 不进入 loaded_params；
量化参数可能有多个附属 tensor。
```

因此 `loaded_weights` 是 sanity check 的依据，不是 checkpoint 到参数的一一映射表。

---

## 28. 跳过权重的常见原因

`load_weights()` 中经常能看到各种 skip rule。

常见原因包括：

```text
lm_head.weight：
  tie_word_embeddings=True 时跳过。

rotary_emb.inv_freq：
  RoPE 频率可能运行时生成，不从 checkpoint 加载。

bias：
  某些 vLLM layer 不使用 bias，或 checkpoint bias 与架构不匹配。

vision / projector extra keys：
  HF checkpoint 中存在训练或 processor 相关额外权重。

PP 非本地层：
  当前 pipeline stage 不拥有这些 layer。

EP 非本地 expert：
  当前 expert parallel rank 不拥有这些 expert。

quant method 不需要的 tensor：
  某些 scale / zero point 由运行时重新计算或由其他参数承载。
```

因此看到 skip 不要立刻理解成加载失败。

要先判断：

```text
这个权重是否应该存在于当前 rank 的 vLLM 模型结构中？
```

---

## 29. checkpoint 已经 fused 的情况

并不是所有 checkpoint 都是 HF 原始拆分格式。

有些 checkpoint 可能已经是 fused 格式，例如：

```text
qkv_proj.weight
gate_up_proj.weight
```

这时 `load_weights()` 可能不会传 `shard_id`。

可以理解为两种路径：

```text
拆分 checkpoint：
  q_proj / k_proj / v_proj
  → qkv_proj + shard_id

fused checkpoint：
  qkv_proj
  → qkv_proj + shard_id=None
```

参数 loader 需要同时支持这两种情况。

因此：

```text
loaded_shard_id is None
```

通常表示：

```text
checkpoint tensor 已经对应整个 fused 参数，
不需要再按 q/k/v 或 gate/up 单独写入。
```

但仍然可能需要 TP slice。

---

## 30. 保存格式和加载格式的差异

vLLM 可以加载 HF checkpoint，也可以加载 vLLM 自己保存的 sharded state。

两者差异是：

```text
HF checkpoint：
  更接近训练框架命名；
  经常需要 qkv / gate_up / expert 名称映射；
  tensor 通常是全局完整权重。

vLLM sharded state：
  更接近 vLLM 内部参数命名；
  可能已经按 TP / PP / EP 保存；
  加载时映射规则不同。
```

因此不要假设所有 `load_format` 都走完全一样的名称映射。

`DefaultModelLoader` 关注 HF 风格权重；
`ShardedStateLoader` 更关注 vLLM 自己保存的 rank-local state。

---

## 31. 一个典型 decoder-only 模型的加载流程

以 Llama / Qwen2 类模型为例，可以把流程理解为：

```text
1. ModelLoader 创建模型结构
   - embed_tokens
   - decoder layers
   - qkv_proj
   - o_proj
   - gate_up_proj
   - down_proj
   - norm
   - lm_head

2. ModelLoader 迭代 checkpoint
   - model.embed_tokens.weight
   - model.layers.0.self_attn.q_proj.weight
   - model.layers.0.self_attn.k_proj.weight
   - model.layers.0.self_attn.v_proj.weight
   - model.layers.0.self_attn.o_proj.weight
   - model.layers.0.mlp.gate_proj.weight
   - model.layers.0.mlp.up_proj.weight
   - model.layers.0.mlp.down_proj.weight
   - model.norm.weight
   - lm_head.weight

3. model.load_weights 做名称映射
   - q_proj/k_proj/v_proj → qkv_proj
   - gate_proj/up_proj → gate_up_proj
   - tied lm_head → skip 或 tie

4. param.weight_loader 做参数级加载
   - qkv 当前 rank shard
   - gate_up 当前 rank shard
   - o_proj row shard
   - down_proj row shard
   - embedding vocab shard
   - norm replicated

5. 加载后处理
   - quant finalize
   - process_weights_after_loading
   - model.eval()
```

这个流程就是 vLLM 权重加载的基础心智模型。

---

## 32. 名称映射和 LoRA target modules 的关系

LoRA 的 target modules 通常写的是逻辑模块名，例如：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

但 vLLM 内部推理模块可能是：

```text
qkv_proj
gate_up_proj
```

因此 LoRA 适配时也需要理解 packed module mapping。

否则会出现：

```text
用户指定 q_proj，
但 vLLM 模型里没有单独 q_proj module。
```

vLLM 需要把 LoRA 的目标模块映射到 fused module 的对应 shard。

这和 checkpoint 权重加载的核心问题类似：

```text
外部生态使用 HF 名称，
vLLM 内部使用 serving 优化后的 fused 名称。
```

---

## 33. 权重加载和 forward 的边界

权重加载阶段解决的是：

```text
参数 tensor 在当前 rank 上是什么。
```

forward 阶段解决的是：

```text
给定 input_ids / positions / attention metadata，如何用这些参数计算 hidden states / logits。
```

两者的边界是：

```text
load_model 完成后，模型参数已经就位；
execute_model 不再做 checkpoint 名称映射；
forward 只使用已经加载好的 vLLM 内部参数。
```

因此：

```text
q_proj/k_proj/v_proj 的名称映射只发生在加载阶段；
forward 中看到的是 qkv_proj 这个 fused layer。
```

---

## 34. 容易疑惑的点

### 34.1 checkpoint 名称一定等于 vLLM 参数名吗？

不一定。

HF checkpoint 名称经常是：

```text
q_proj / k_proj / v_proj / gate_proj / up_proj
```

vLLM 参数名可能是：

```text
qkv_proj / gate_up_proj
```

所以需要 `load_weights()` 做名称映射。

### 34.2 packed_modules_mapping 会直接 copy 权重吗？

通常不会。

它更多是声明：

```text
哪些 HF module 被 vLLM packed 到一个 module 里。
```

真正 copy 权重的是：

```text
param.weight_loader
```

### 34.3 shard_id 的含义统一吗？

不统一。

例如：

```text
qkv："q" / "k" / "v"
gate_up：0 / 1
MoE："w1" / "w2" / "w3" 或 expert-specific id
```

要结合具体 layer 的 `weight_loader` 理解。

### 34.4 TP 切片是在 model.load_weights() 里做的吗？

通常不是。

`model.load_weights()` 多数只做名称映射和传递 `shard_id`。

真正 TP slice 通常在：

```text
ColumnParallelLinear / RowParallelLinear / QKVParallelLinear / VocabParallelEmbedding 的 weight_loader
```

### 34.5 为什么 tied lm_head 被跳过？

因为：

```text
lm_head.weight 和 embed_tokens.weight 共享。
```

加载 embedding 后，lm_head 已经有对应权重，不需要重复加载。

### 34.6 量化权重能按 fp16 规则理解吗？

不能。

量化参数可能是：

```text
packed qweight
scales
zeros
block scales
g_idx
```

加载时要考虑 packed_dim、packed_factor、tile layout 和 quant method。

### 34.7 MoE expert id 等于 checkpoint 里的 id 吗？

不一定。

EP / EPLB 下可能存在：

```text
checkpoint expert id
  → global expert id
  → physical expert id
  → local expert id
```

当前 rank 只加载自己拥有的 expert。

### 34.8 load_weights 返回的 loaded_params 是完整映射表吗？

不是。

它只是已加载参数集合。

一个参数可能由多个 checkpoint tensor 共同加载，一个 checkpoint tensor 也可能因为 tied / skip 不进入集合。

---

## 35. 排查权重加载问题的顺序

如果遇到权重加载报错，可以按这个顺序查。

### 35.1 先看 loader 类型

确认：

```text
load_format 是什么？
使用的是 DefaultModelLoader 还是 BitsAndBytes / ShardedState / Tensorizer？
checkpoint 是 HF 格式还是 vLLM sharded state？
```

### 35.2 再看模型 load_weights

重点找：

```text
stacked_params_mapping
packed_modules_mapping
skip_prefixes
hf_to_vllm_mapper
ignore_unexpected_prefixes
```

判断 checkpoint name 是否应该被替换或跳过。

### 35.3 再看参数是否存在

如果报 `params_dict[name]` 找不到，说明：

```text
名称映射后的 vLLM 参数名不存在。
```

常见原因：

```text
1. checkpoint 架构和 model class 不匹配；
2. mapper 少了一层前缀；
3. PP rank 跳过逻辑不正确；
4. tied lm_head 没有正确 skip；
5. 多模态子模块名称不一致。
```

### 35.4 再看 shape mismatch

如果参数存在但 shape 不匹配，重点看：

```text
1. TP size 是否和 checkpoint 预期一致；
2. qkv / gate_up 是否 fused；
3. hidden size / intermediate size 是否匹配；
4. vocab size 是否 padded；
5. quantization packed_factor 是否正确；
6. num_kv_heads / GQA 配置是否正确。
```

### 35.5 最后看 rank-local 特殊逻辑

分布式场景还要看：

```text
PP：当前 rank 是否拥有该 layer；
TP：当前 rank 应该加载哪一片；
EP：当前 rank 是否拥有该 expert；
DP：各 DP rank 是否各自加载同样模型 shard。
```

---

## 36. 总结

vLLM 权重加载可以压缩成：

```text
GPUModelRunner.load_model()
  → get_model_loader()
  → BaseModelLoader.load_model()
  → initialize_model()
  → DefaultModelLoader.get_all_weights()
  → model.load_weights(weights)
  → name mapping / skip / packed mapping
  → param.weight_loader()
  → TP / PP / EP / quant / fused 写入
  → process_weights_after_loading()
```

如果只记住一句话：

```text
load_weights() 是 checkpoint 格式和 vLLM 推理模型结构之间的适配层；真正的数据切片和写入通常下沉到 parameter.weight_loader。
```

再压缩成最小心智模型：

```text
checkpoint 名称是外部训练格式；
vLLM 参数名是内部推理格式；
model.load_weights 负责翻译名称；
parameter.weight_loader 负责加载当前 rank 需要的 tensor 部分；
fused、TP、量化、MoE 都是在这两层之间协作完成的。
```
