# 04. 一个 vLLM model class 是如何构造的？

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/model_executor/model_loader/`
- `vllm/vllm/model_executor/models/`
- `vllm/vllm/model_executor/layers/`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：vLLM 如何从 HuggingFace config 的 `architectures` 解析出真正使用的 vLLM model class；`ModelRunner.load_model()` 如何调用 model loader 完成实例化；一个典型 `*ForCausalLM` 类如何根据 `VllmConfig / HF config` 构造 embedding、decoder layers、attention、MLP、norm、lm_head、logits processor；pooling / embedding / classification 模型如何由 adapter 包装；TP / PP / quantization / KV cache / weight loading 如何插入模型类构造过程。

---

## 1. 一句话回答

一个 vLLM model class 的构造可以分成两层：

```text
外层选择和加载：
  HF config.architectures
    → ModelConfig.architectures / runner_type / convert_type
    → ModelRegistry.resolve_model_cls()
    → initialize_model()
    → model_class(vllm_config, prefix)
    → load_weights()

内层模型搭图：
  vllm_config.model_config.hf_config
    → embedding
    → layers[0..N]
        → attention(qkv + rope + Attention backend + KV cache spec)
        → mlp / moe
        → norms
    → final norm
    → lm_head + logits_processor
    → load_weights / compute_logits / forward
```

一句话压缩：

```text
model class construction 是把 HF config 和 vLLM 运行配置翻译成一个可并行、可量化、可加载权重、可被 ModelRunner forward 的 PyTorch module graph。
```

---

## 2. 构造链路总览

以普通 generation 模型为例，完整主链路是：

```text
GPUWorker.load_model()
  → GPUModelRunner.load_model()
  → get_model_loader(load_config)
  → BaseModelLoader.load_model(vllm_config, model_config)
      → initialize_model(vllm_config, model_config)
          → get_model_architecture(model_config)
              → ModelRegistry.resolve_model_cls(hf_config.architectures)
          → model_class(vllm_config=vllm_config, prefix=prefix)
              → __init__() 内部按 config 搭建 module graph
      → loader.load_weights(model, model_config)
      → process_weights_after_loading()
      → model.eval()
```

关键位置：

- `GPUModelRunner.load_model()` 调 `get_model_loader()` 和 `model_loader.load_model()`：`vllm/vllm/v1/worker/gpu_model_runner.py:5231`
- `get_model_loader()` 根据 `load_format` 选择 loader：`vllm/vllm/model_executor/model_loader/__init__.py:122`
- `BaseModelLoader.load_model()` 负责 `initialize_model → load_weights → postprocess → eval`：`vllm/vllm/model_executor/model_loader/base_loader.py:42`
- `initialize_model()` 负责选类并调用 `model_class(vllm_config, prefix)`：`vllm/vllm/model_executor/model_loader/utils.py:42`

---

## 3. ModelConfig 先解析模型能力和运行类型

模型类不是直接从文件名硬编码出来的，而是先由 `ModelConfig` 解析 HF config。

初始化阶段会拿到：

```python
architectures = self.architectures
registry = self.registry
is_generative_model = registry.is_text_generation_model(architectures, self)
is_pooling_model = registry.is_pooling_model(architectures, self)

self.runner_type = self._get_runner_type(...)
self.convert_type = self._get_convert_type(...)
```

位置：`vllm/vllm/config/model.py:578`

这一步决定三个关键信息：

```text
architectures：
  HF config / architecture convertor 给出的候选架构名。

runner_type：
  当前实例以 generate、pooling 还是 draft 方式运行。

convert_type：
  是否需要把原始 generation model 包装成 embedding / classify model。
```

`ModelConfig.registry` 返回的是全局 `ModelRegistry`：

```python
@property
def registry(self):
    return me_models.ModelRegistry
```

位置：`vllm/vllm/config/model.py:827`

如果 `runner_type == "pooling"`，还会补齐 `PoolerConfig` 的默认 pooling 类型：

```text
runner_type == pooling
  → 初始化 PoolerConfig
  → 使用 model_info.default_seq_pooling_type
  → 使用 model_info.default_tok_pooling_type
```

位置：`vllm/vllm/config/model.py:640`

---

## 4. registry 如何把 HF architecture 映射到 vLLM 类

vLLM 在 `models/registry.py` 里维护了 architecture 名到实现类的映射。

例如文本生成模型映射表里：

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM")
```

位置：`vllm/vllm/model_executor/models/registry.py:143`

这表示：

```text
HF config.architectures = ["LlamaForCausalLM"]
  → import vllm.model_executor.models.llama
  → 取 LlamaForCausalLM 类
```

registry 不只是一个字典，它还处理：

```text
1. in-tree vLLM model；
2. lazy import，避免主进程过早初始化 CUDA；
3. Transformers backend fallback；
4. Terratorch backend；
5. runner / convert 默认匹配；
6. 模型接口能力检查，例如是否支持 generation / pooling / multimodal / PP。
```

`resolve_model_cls()` 的核心逻辑是：

```text
如果 model_impl == transformers：
  尝试解析 Transformers backend class。

否则：
  如果架构不在 vLLM registry 且 model_impl == auto：
    可能 fallback 到 Transformers backend。

遍历 HF architectures：
  normalized_arch = _normalize_arch(arch, model_config)
  model_cls = _try_load_model_cls(normalized_arch)
  成功则返回。

仍失败：
  抛出 unsupported architecture。
```

位置：`vllm/vllm/model_executor/models/registry.py:1253`

所以 registry 的定位是：

```text
registry 解决“这个 HF architecture 应该用哪个 Python class 实现”。
```

---

## 5. initialize_model：真正实例化 model class

`initialize_model()` 是模型类实例化的核心入口。

核心代码：

```python
if model_class is None:
    model_class, _ = get_model_architecture(model_config)

if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)

with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
    model = model_class(vllm_config=vllm_config, prefix=prefix)
    record_metadata_for_reloading(model)
    return model
```

位置：`vllm/vllm/model_executor/model_loader/utils.py:42`

这里有几个重要点：

```text
1. 新式 vLLM model class 必须接受 vllm_config 和 prefix；
2. quant_config 会在实例化前绑定到模型类 / layer 构造逻辑；
3. set_current_vllm_config 会建立构造期上下文，Attention 等层会从这里取当前 VllmConfig；
4. prefix 会贯穿模块命名、权重名匹配、量化方法选择、static_forward_context 注册。
```

因此，vLLM 的模型构造不是：

```text
model_class(config)
```

而是：

```text
model_class(vllm_config=vllm_config, prefix=prefix)
```

因为构造时不只需要 HF config，还需要：

```text
cache_config；
parallel_config；
quant_config；
load_config；
compilation_config；
lora_config；
scheduler_config；
pooling / multimodal / speculative config。
```

---

## 6. loader：实例化之后再加载权重

`BaseModelLoader.load_model()` 的职责可以拆成：

```text
1. 根据 load_config / device_config 确定 load_device；
2. 设置默认 torch dtype；
3. 在目标 device 上 initialize_model()；
4. 打印可选模型结构；
5. load_weights()；
6. online quantization 后处理；
7. process_weights_after_loading()；
8. 返回 model.eval()。
```

位置：`vllm/vllm/model_executor/model_loader/base_loader.py:42`

默认 loader 的权重加载入口是：

```python
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`vllm/vllm/model_executor/model_loader/default_loader.py:427`

这说明：

```text
模型类 __init__ 只负责搭 graph 和创建参数；
权重数据由 loader 读取 checkpoint 后交给 model.load_weights() 填入。
```

权重加载过程会处理：

```text
safetensors / pt / npcache / mistral / fastsafetensors 等格式；
远程或本地 checkpoint；
sharded checkpoint；
EP expert weight filter；
packed qkv / gate_up_proj 参数映射；
PP missing layer 跳过；
量化权重后处理。
```

---

## 7. 典型 generation 模型结构：LlamaForCausalLM

以 `LlamaForCausalLM` 为例，它是 vLLM decoder-only generation model 的典型结构。

类定义：

```python
class LlamaForCausalLM(
    LocalArgmaxMixin,
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    SupportsQuant,
):
```

位置：`vllm/vllm/model_executor/models/llama.py:446`

它的构造分两层：

```text
LlamaForCausalLM
  → self.model = LlamaModel(...)
  → self.lm_head = ParallelLMHead(...)
  → self.logits_processor = LogitsProcessor(...)
```

对应代码：

```python
self.model = self._init_model(...)

if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)
    if config.tie_word_embeddings:
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
    self.logits_processor = LogitsProcessor(...)
else:
    self.lm_head = PPMissingLayer()
```

位置：`vllm/vllm/model_executor/models/llama.py:478`

它的 forward 很薄：

```python
model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
return model_output
```

位置：`vllm/vllm/model_executor/models/llama.py:516`

logits 不在 `forward()` 中直接产生，而是通过：

```python
logits = self.logits_processor(self.lm_head, hidden_states)
```

位置：`vllm/vllm/model_executor/models/llama.py:528`

所以 generation 模型外壳可以记为：

```text
forward()：
  input_ids / positions / intermediate_tensors / inputs_embeds
    → backbone hidden_states

compute_logits()：
  selected hidden_states
    → lm_head + logits_processor
    → logits
```

这和执行层文档里的 `ModelRunner` 逻辑一致：`ModelRunner` 先 `_model_forward()` 得到 hidden states，再对 `logits_indices` 选出的 hidden states 调 `model.compute_logits()`。

---

## 8. Backbone：LlamaModel 如何构造 embedding、layers、norm

`LlamaModel` 是真正的 decoder backbone。

类定义带 `@support_torch_compile`：

```python
@support_torch_compile(...)
class LlamaModel(nn.Module, EagleModelMixin):
```

位置：`vllm/vllm/model_executor/models/llama.py:334`

构造主体：

```python
config = vllm_config.model_config.hf_config
quant_config = vllm_config.quant_config

self.vocab_size = config.vocab_size

if get_pp_group().is_first_rank or (
    config.tie_word_embeddings and get_pp_group().is_last_rank
):
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()

self.start_layer, self.end_layer, self.layers = make_layers(...)

if get_pp_group().is_last_rank:
    self.norm = RMSNorm(...)
else:
    self.norm = PPMissingLayer()
```

位置：`vllm/vllm/model_executor/models/llama.py:356`

这体现了三个关键设计：

```text
1. embedding 通常只在 first PP rank 存在；
2. decoder layers 通过 make_layers() 按 PP rank 切分；
3. final norm 通常只在 last PP rank 存在。
```

backbone 的结构可以画成：

```text
LlamaModel
  ├─ embed_tokens: VocabParallelEmbedding 或 PPMissingLayer
  ├─ layers: ModuleList[PPMissingLayer / LlamaDecoderLayer / PPMissingLayer]
  ├─ norm: RMSNorm 或 PPMissingLayer
  └─ make_empty_intermediate_tensors
```

---

## 9. Pipeline Parallel 如何影响 model class 构造

PP 不只影响运行时通信，也直接影响模型类构造。

`make_layers()` 会根据当前 PP rank 计算本 rank 负责的层范围：

```python
start_layer, end_layer = get_pp_indices(
    num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
)
```

位置：`vllm/vllm/model_executor/models/utils.py:687`

然后构造：

```python
modules = torch.nn.ModuleList(
    [PPMissingLayer() for _ in range(start_layer)]
    + get_offloader().wrap_modules(
        layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
    )
    + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
)
```

位置：`vllm/vllm/model_executor/models/utils.py:711`

也就是说每个 PP rank 的 `self.layers` 长度仍然等于总层数，但不属于本 rank 的层会变成 `PPMissingLayer`。

`PPMissingLayer` 是一个占位层：

```python
class PPMissingLayer(torch.nn.Identity):
    def forward(self, *args, **kwargs):
        return args[0] if args else next(iter(kwargs.values()))
```

位置：`vllm/vllm/model_executor/models/utils.py:674`

这样做的好处是：

```text
1. 模块名和层号保持全局一致；
2. 权重加载时可以判断 missing layer 并跳过；
3. PP rank 可以只实例化自己负责的重层；
4. forward 逻辑仍可通过 start_layer / end_layer 只遍历本 rank 层。
```

`LlamaModel.forward()` 中也按 PP rank 分支：

```text
first PP rank：
  input_ids / inputs_embeds → embed_tokens → hidden_states

middle PP rank：
  intermediate_tensors["hidden_states"] / ["residual"] → 本 rank layers

non-last PP rank：
  return IntermediateTensors({"hidden_states", "residual"})

last PP rank：
  norm(hidden_states, residual) → final hidden_states
```

位置：`vllm/vllm/model_executor/models/llama.py:400`

所以 PP 场景下，model class 构造出来的是：

```text
同一个逻辑模型的 rank-local shard。
```

---

## 10. Decoder layer 如何由 config 构造

`LlamaDecoderLayer` 是单层 transformer block。

构造入口：

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", ...):
```

位置：`vllm/vllm/model_executor/models/llama.py:248`

它从 `vllm_config` 取：

```text
config = vllm_config.model_config.hf_config；
cache_config = vllm_config.cache_config；
quant_config = self.get_quant_config(vllm_config)。
```

位置：`vllm/vllm/model_executor/models/llama.py:258`

然后构造：

```text
LlamaDecoderLayer
  ├─ self_attn: LlamaAttention
  ├─ mlp: LlamaMLP
  ├─ input_layernorm: RMSNorm
  └─ post_attention_layernorm: RMSNorm
```

关键字段来自 HF config：

```text
hidden_size；
num_attention_heads；
num_key_value_heads；
max_position_embeddings；
intermediate_size；
hidden_act；
rms_norm_eps；
attention_bias / qkv_bias / mlp_bias；
is_causal / sliding_window / layer_types。
```

构造 attention：

```python
self.self_attn = attn_layer_type(
    config=config,
    hidden_size=self.hidden_size,
    num_heads=config.num_attention_heads,
    num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
    max_position_embeddings=max_position_embeddings,
    quant_config=quant_config,
    cache_config=cache_config,
    prefix=f"{prefix}.self_attn",
    attn_type=attn_type,
)
```

位置：`vllm/vllm/model_executor/models/llama.py:282`

构造 MLP 和 norm：

```python
self.mlp = LlamaMLP(...)
self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

位置：`vllm/vllm/model_executor/models/llama.py:297`

---

## 11. Attention 子图如何构造

`LlamaAttention` 把 attention 拆成：

```text
qkv projection；
o projection；
rotary embedding；
Attention runtime layer。
```

构造入口：`vllm/vllm/model_executor/models/llama.py:122`

关键逻辑：

```python
tp_size = get_tensor_model_parallel_world_size()
self.total_num_heads = num_heads
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self._init_rotary_emb(config, quant_config=quant_config)
self.attn = Attention(...)
```

位置：`vllm/vllm/model_executor/models/llama.py:140`

这里体现 TP 对 attention 构造的影响：

```text
num_heads 会按 tensor parallel size 切分；
num_kv_heads 如果小于 TP size，则在多个 rank 上复制；
QKVParallelLinear / RowParallelLinear 负责张量并行线性层；
Attention 层保存 KV cache 相关元信息和 backend impl。
```

RoPE 通过 `get_rope()` 构造：

```python
self.rotary_emb = get_rope(
    self.head_dim,
    max_position=self.max_position_embeddings,
    rope_parameters=getattr(config, "rope_parameters", None),
    is_neox_style=is_neox_style,
)
```

位置：`vllm/vllm/model_executor/models/llama.py:233`

attention forward 则是：

```text
hidden_states
  → qkv_proj
  → split q / k / v
  → rotary_emb(positions, q, k)
  → Attention(q, k, v)
  → o_proj
```

位置：`vllm/vllm/model_executor/models/llama.py:221`

---

## 12. Attention 层和 KV cache / backend 的关系

`vllm.model_executor.layers.attention.Attention` 是 runtime attention 层。

类定义：

```python
class Attention(nn.Module, AttentionLayerBase):
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:221`

构造时会确定：

```text
num_heads / num_kv_heads / head_size；
KV cache dtype；
sliding window；
attention backend；
backend impl；
static_forward_context 注册；
KV cache quantization；
KV sharing target；
layer_name / prefix。
```

backend 选择发生在：

```python
self.attn_backend = get_attn_backend(...)
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:348`

Attention 会把自己注册进 compile config 的 `static_forward_context`：

```python
if prefix in compilation_config.static_forward_context:
    raise ValueError(f"Duplicate layer name: {prefix}")
compilation_config.static_forward_context[prefix] = self
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:440`

这一步很重要，因为后续 ModelRunner / attention backend 会按 layer name 绑定 KV cache、查 attention layer、构造 KV cache spec。

Attention 的 forward 注释说明：

```text
attention metadata 不是显式传给 Attention.forward()；
而是由 ModelRunner.execute_model() 通过 set_forward_context() 设置；
Attention 内部通过 get_forward_context().attn_metadata 使用。
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:485`

KV cache 规格由 `get_kv_cache_spec()` 返回：

```text
encoder-only / encoder attention：返回 None；
sliding window decoder attention：返回 SlidingWindowSpec；
普通 decoder attention：返回 FullAttentionSpec；
特殊量化 cache：返回对应 spec。
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:616`

所以 attention 构造的意义不只是创建一个 PyTorch layer，它还定义了：

```text
这个 layer 是否需要 KV cache；
KV cache 每个 block 多大；
KV head 数和 head dim；
使用哪个 attention backend；
是否 sliding window；
是否 KV quantization；
运行时 layer_name 如何和 forward context / KV cache 绑定。
```

---

## 13. MLP 子图如何构造

`LlamaMLP` 是典型 SwiGLU MLP。

构造结构：

```text
LlamaMLP
  ├─ gate_up_proj: MergedColumnParallelLinear
  ├─ down_proj: RowParallelLinear
  └─ act_fn: SiluAndMul
```

位置：`vllm/vllm/model_executor/models/llama.py:79`

代码：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size] * 2,
    bias=bias,
    quant_config=quant_config,
    prefix=f"{prefix}.gate_up_proj",
)
self.down_proj = RowParallelLinear(
    input_size=intermediate_size,
    output_size=hidden_size,
    bias=bias,
    quant_config=quant_config,
    prefix=f"{prefix}.down_proj",
)
self.act_fn = SiluAndMul()
```

位置：`vllm/vllm/model_executor/models/llama.py:92`

这里也能看到 vLLM 模型类的典型写法：

```text
HF 的 gate_proj / up_proj 在 vLLM 里合并成 gate_up_proj；
ColumnParallel / RowParallel 负责 tensor parallel；
quant_config 会传到每个线性层；
prefix 用于权重加载和量化方法选择。
```

---

## 14. Embedding 和 LM head 如何构造

vLLM 使用 `VocabParallelEmbedding` 和 `ParallelLMHead` 处理词表维度并行。

`VocabParallelEmbedding` 定义：

```python
class VocabParallelEmbedding(PluggableLayer):
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:198`

构造时会：

```text
1. 根据 tp_rank / tp_size 计算当前 rank 的 vocab shard；
2. 对原始 vocab 和 added vocab 分别 padding；
3. 从 quant_config 获取 embedding quant method；
4. 创建当前 rank 的 embedding weight；
5. 注册 weight_loader。
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:239`

forward 时：

```text
TP > 1：
  mask 不属于本 rank 的 token ids；
  本 rank 做 embedding；
  对 masked 位置置 0；
  tensor_model_parallel_all_reduce 汇总。

TP == 1：
  直接 embedding。
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:472`

`ParallelLMHead` 继承 `VocabParallelEmbedding`：

```python
class ParallelLMHead(VocabParallelEmbedding):
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:505`

它的权重用于 sampler / logits processor，不应该被当作普通 forward layer 调用：

```python
def forward(self, input_):
    del input_
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:559`

这解释了为什么 generation 模型里是：

```text
hidden_states
  → model.compute_logits()
  → logits_processor(lm_head, hidden_states)
```

而不是：

```text
hidden_states
  → lm_head(hidden_states)
```

---

## 15. Weight loading 如何适配 vLLM 的模块结构

模型类通常自己实现 `load_weights()`。

`LlamaForCausalLM.load_weights()` 使用 `AutoWeightsLoader`：

```python
loader = AutoWeightsLoader(
    self,
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
)
return loader.load_weights(weights)
```

位置：`vllm/vllm/model_executor/models/llama.py:535`

`LlamaModel.hf_to_vllm_mapper` 中还有 packed 参数映射，`LlamaModel.load_weights()` 会把它传给 `AutoWeightsLoader`：

```text
.qkv_proj      ← .q_proj / .k_proj / .v_proj
.gate_up_proj  ← .gate_proj / .up_proj
```

位置：`vllm/vllm/model_executor/models/llama.py:344`

这解决了 HF checkpoint 和 vLLM 模块结构不完全一致的问题：

```text
HF 权重名：
  model.layers.0.self_attn.q_proj.weight
  model.layers.0.self_attn.k_proj.weight
  model.layers.0.self_attn.v_proj.weight

vLLM 参数：
  model.layers.0.self_attn.qkv_proj.weight
```

加载时还会跳过 PP 不属于本 rank 的 missing module：`AutoWeightsLoader._load_module()` 遇到 `StageMissingLayer / PPMissingLayer` 会直接返回；通用的 `is_pp_missing_parameter()` 也会根据 missing layer 前缀判断参数是否缺失。

位置：`vllm/vllm/model_executor/models/utils.py:318`、`vllm/vllm/model_executor/models/utils.py:744`

所以权重加载不是简单 `load_state_dict()`，而是：

```text
checkpoint iterator
  → weight name remap / packed shard map
  → TP shard loader
  → PP missing layer skip
  → quant weight loader
  → strict missing check / postprocess
```

---

## 16. Pooling / embedding model 如何构造

有些模型本身就是 pooling model；也有些 generation model 可以通过 adapter 转成 pooling model。

`get_model_architecture()` 里会根据 `model_config.convert_type` 包装：

```python
if convert_type == "embed":
    model_cls = as_embedding_model(model_cls)
elif convert_type == "classify":
    model_cls = as_seq_cls_model(model_cls)
```

位置：`vllm/vllm/model_executor/model_loader/utils.py:193`

`as_embedding_model()` 会基于原始模型动态创建一个 subclass：

```text
原始 LlamaForCausalLM
  → ModelForPooling
  → ModelForEmbedding
```

位置：`vllm/vllm/model_executor/models/adapters.py:230`

`ModelForPooling.__init__()` 有一个关键技巧：

```python
with no_init_weights(
    self,
    lambda mod: StageMissingLayer("output", mod),
    targets=(LogitsProcessor, ParallelLMHead),
):
    super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
```

位置：`vllm/vllm/model_executor/models/adapters.py:136`

这表示：

```text
pooling 模型仍可复用 generation backbone；
但 lm_head / logits_processor 这类输出头会被 StageMissingLayer 替代；
然后额外创建 pooler。
```

pooler 初始化：

```python
if not pooler:
    pooler = self._init_pooler(vllm_config, prefix=prefix)
self.pooler = pooler
```

位置：`vllm/vllm/model_executor/models/adapters.py:156`

embedding adapter 创建的是：

```python
return DispatchPooler.for_embedding(pooler_config)
```

位置：`vllm/vllm/model_executor/models/adapters.py:248`

sequence classification adapter 会额外创建 `score` head：

```python
self.score = ReplicatedLinear(...)
return DispatchPooler.for_seq_cls(pooler_config, classifier=self.score)
```

位置：`vllm/vllm/model_executor/models/adapters.py:321`

因此 pooling 模型构造可以记为：

```text
原始生成模型类
  → 去掉 lm_head / logits processor
  → 保留 backbone forward
  → 增加 pooler
  → is_pooling_model = True
```

---

## 17. 多模态模型的构造形态

多模态模型不是单一模板，但通常遵循：

```text
MultimodalForConditionalGeneration
  ├─ vision tower / audio encoder / image encoder
  ├─ projector / resampler / merger
  ├─ language model backbone
  ├─ lm_head / logits_processor
  └─ multimodal input mapper / dummy input builder / processor info
```

和纯文本模型相比，多模态模型构造额外关注：

```text
1. model_info.supports_multimodal；
2. MultiModalConfig 初始化；
3. MULTIMODAL_REGISTRY 是否支持该模型；
4. encoder / projector 的 dtype、TP、cache、processor；
5. forward 时如何把 multimodal embeddings 合并进 text embeddings；
6. get_language_model() / get_multimodal_embeddings() 等接口。
```

`ModelConfig` 会根据 `model_info.supports_multimodal` 初始化 `MultiModalConfig`：

位置：`vllm/vllm/config/model.py:682`

从 model class construction 的角度看，多模态模型本质是：

```text
额外构造一个或多个 encoder，再把 encoder 输出投影并注入 language model backbone。
```

---

## 18. 构造时注入的配置清单

一个 vLLM model class 构造时通常会消费这些配置。

### 18.1 来自 HF config

```text
vocab_size；
hidden_size；
num_hidden_layers；
num_attention_heads；
num_key_value_heads；
head_dim；
intermediate_size；
hidden_act；
rms_norm_eps / layer_norm_eps；
max_position_embeddings；
rope_parameters / rope_scaling；
sliding_window / layer_types；
tie_word_embeddings；
attention_bias / qkv_bias / mlp_bias；
model_type / architectures。
```

### 18.2 来自 vLLM config

```text
model_config：runner、convert、dtype、max_model_len、pooler_config、multimodal_config；
cache_config：block_size、cache_dtype、sliding_window、prefix caching、KV quant；
parallel_config：TP / PP / DP / EP；
quant_config：weight quant、activation quant、KV cache quant；
compilation_config：torch.compile、static_forward_context、CUDA graph；
lora_config：LoRA embedding / packed module 支持；
load_config：load_format、download_dir、weight loading strategy；
scheduler_config：batch 上限会间接影响 runner / profiling；
speculative_config：draft / Eagle / MTP 相关构造。
```

### 18.3 来自构造期上下文

```text
prefix：模块名、权重名前缀、量化 method lookup、attention layer_name；
current_vllm_config：Attention 等层从上下文获取完整 VllmConfig；
当前 distributed rank：TP rank、PP rank、DP rank、EP rank；
当前默认 dtype / target device。
```

---

## 19. Prefix 为什么重要

vLLM 模型类大量传递 `prefix`：

```text
LlamaForCausalLM(prefix="")
  → LlamaModel(prefix="model")
      → layers(prefix="model.layers")
          → layer 0(prefix="model.layers.0")
              → self_attn(prefix="model.layers.0.self_attn")
                  → qkv_proj(prefix="model.layers.0.self_attn.qkv_proj")
                  → attn(prefix="model.layers.0.self_attn.attn")
              → mlp(prefix="model.layers.0.mlp")
  → lm_head(prefix="lm_head")
```

它影响：

```text
1. checkpoint 权重名匹配；
2. quant_config.get_quant_method(layer, prefix=prefix)；
3. Attention.layer_name；
4. compilation_config.static_forward_context 的 key；
5. KV cache spec 中 attention layer 的名称；
6. LoRA / packed modules / reload metadata。
```

所以 prefix 不是日志用字符串，而是 vLLM 模型构造中的结构性标识。

---

## 20. 构造出的 model class 需要提供哪些接口

一个可被 vLLM ModelRunner 使用的 generation model 通常需要：

```text
__init__(vllm_config: VllmConfig, prefix: str = "")；
forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs)；
compute_logits(hidden_states)；
load_weights(weights)；
make_empty_intermediate_tensors；
```

常见能力接口 / mixin：

```text
SupportsLoRA：支持 LoRA 注入；
SupportsPP：支持 pipeline parallel；
SupportsEagle / SupportsEagle3：支持 speculative / Eagle；
VllmModelForPooling：支持 pooling output；
SupportsMultiModal：支持 multimodal inputs；
SupportsCrossEncoding：支持 cross encoder / classification；
IsHybrid / HasInnerState：支持 hybrid / stateful 模型。
```

从执行层看，最关键的是：

```text
ModelRunner._model_forward()
  → self.model(...)

ModelRunner.compute logits
  → self.model.compute_logits(sample_hidden_states)

pooling path
  → self.model.pooler(hidden_states, pooling_metadata)
```

---

## 21. 一个完整例子：LlamaForCausalLM 构造过程

假设 HF config 是：

```text
architectures = ["LlamaForCausalLM"]
runner = auto
tensor_parallel_size = 2
pipeline_parallel_size = 1
```

构造链路：

```text
1. ModelConfig 读取 HF config
   → architectures = ["LlamaForCausalLM"]
   → runner_type = generate
   → convert_type = none

2. get_model_architecture()
   → ModelRegistry.resolve_model_cls(["LlamaForCausalLM"])
   → 返回 vllm.model_executor.models.llama.LlamaForCausalLM

3. initialize_model()
   → 设置 current_vllm_config
   → 调 LlamaForCausalLM(vllm_config, prefix="")

4. LlamaForCausalLM.__init__()
   → self.model = LlamaModel(prefix="model")
   → self.lm_head = ParallelLMHead(prefix="lm_head")
   → self.logits_processor = LogitsProcessor(...)

5. LlamaModel.__init__()
   → self.embed_tokens = VocabParallelEmbedding(...)
   → self.layers = make_layers(num_hidden_layers, layer_fn, "model.layers")
   → self.norm = RMSNorm(...)

6. 每个 LlamaDecoderLayer.__init__()
   → self_attn = LlamaAttention(...)
   → mlp = LlamaMLP(...)
   → input_layernorm / post_attention_layernorm

7. LlamaAttention.__init__()
   → qkv_proj = QKVParallelLinear(...)
   → o_proj = RowParallelLinear(...)
   → rotary_emb = get_rope(...)
   → attn = Attention(...)

8. Attention.__init__()
   → 选择 attention backend
   → 注册 static_forward_context[prefix]
   → 初始化 KV cache quant 属性
   → 定义 get_kv_cache_spec()

9. DefaultModelLoader.load_weights()
   → model.load_weights(weights_iterator)
   → packed qkv / gate_up 映射
   → TP shard loading
   → postprocess weights

10. 返回 model.eval()
```

最终得到的结构大致是：

```text
LlamaForCausalLM
  ├─ model: LlamaModel
  │   ├─ embed_tokens: VocabParallelEmbedding
  │   ├─ layers: ModuleList
  │   │   ├─ LlamaDecoderLayer
  │   │   │   ├─ self_attn: LlamaAttention
  │   │   │   │   ├─ qkv_proj: QKVParallelLinear
  │   │   │   │   ├─ o_proj: RowParallelLinear
  │   │   │   │   ├─ rotary_emb
  │   │   │   │   └─ attn: Attention
  │   │   │   ├─ mlp: LlamaMLP
  │   │   │   ├─ input_layernorm: RMSNorm
  │   │   │   └─ post_attention_layernorm: RMSNorm
  │   │   └─ ...
  │   └─ norm: RMSNorm
  ├─ lm_head: ParallelLMHead
  └─ logits_processor: LogitsProcessor
```

---

## 22. 一个完整例子：PP=2 时构造过程有什么不同

假设 `num_hidden_layers = 32`，`pipeline_parallel_size = 2`。

rank 0：

```text
embed_tokens = VocabParallelEmbedding
layers[0..15] = LlamaDecoderLayer
layers[16..31] = PPMissingLayer
norm = PPMissingLayer
lm_head = PPMissingLayer
```

rank 1：

```text
embed_tokens = PPMissingLayer
layers[0..15] = PPMissingLayer
layers[16..31] = LlamaDecoderLayer
norm = RMSNorm
lm_head = ParallelLMHead
```

forward 时：

```text
rank 0：
  input_ids → embedding → layers 0..15 → IntermediateTensors

rank 1：
  IntermediateTensors → layers 16..31 → norm → hidden_states → logits
```

权重加载时：

```text
rank 0 只加载 embedding 和 layers 0..15；
rank 1 只加载 layers 16..31、norm、lm_head；
PPMissingLayer 对应参数会被 is_pp_missing_parameter() 跳过。
```

---

## 23. 一个完整例子：把 generation model 转成 embedding model

假设用户指定：

```text
--runner pooling
--convert embed
```

或者 vLLM 自动判断该模型应该以 pooling 运行。

构造链路：

```text
get_model_architecture()
  → resolve_model_cls() 得到原始 LlamaForCausalLM
  → convert_type == embed
  → as_embedding_model(LlamaForCausalLM)
  → 返回动态类 ModelForEmbedding
```

实例化时：

```text
ModelForEmbedding.__init__()
  → 调原始 LlamaForCausalLM.__init__()
      但用 StageMissingLayer 替代 lm_head / logits_processor
  → self.pooler = DispatchPooler.for_embedding(pooler_config)
```

运行时：

```text
forward() 仍然返回 hidden_states；
ModelRunner 检测 is_pooling_model；
调用 model.pooler(hidden_states, pooling_metadata)；
返回 ModelRunnerOutput(pooler_output=...)。
```

所以 embedding model 不是完全重写一个模型，而是：

```text
复用 backbone，替换输出头。
```

---

## 24. 常见模型类构造模式

### 24.1 Decoder-only CausalLM

```text
ForCausalLM
  → decoder backbone
      → token embedding
      → decoder layers
      → final norm
  → lm_head
  → logits_processor
```

代表：Llama、Qwen、Mistral、Gemma、Phi 等。

### 24.2 MoE CausalLM

```text
ForCausalLM
  → backbone
      → attention
      → MoE layer / shared experts / routed experts
  → lm_head
```

额外关注：

```text
expert_parallel；
expert placement；
EPLB；
EP weight filter；
routed experts 输出。
```

### 24.3 Hybrid / Mamba model

```text
ForCausalLM
  → layers
      → attention layer
      → mamba / linear attention / state-space layer
      → mlp / moe
```

额外关注：

```text
inner state；
Mamba cache spec；
attention-free / hybrid model info；
state preprocess / postprocess。
```

### 24.4 Pooling / embedding / classify model

```text
ModelForPooling
  → backbone
  → pooler
  → 可选 classifier score head
```

额外关注：

```text
PoolerConfig；
seq_pooling_type / tok_pooling_type；
是否复用 generation backbone；
lm_head 是否替换为 StageMissingLayer。
```

### 24.5 Multimodal model

```text
ForConditionalGeneration
  → vision/audio encoder
  → projector / merger
  → language model backbone
  → lm_head / pooler
```

额外关注：

```text
multimodal registry；
processor info；
dummy inputs；
placeholder token；
encoder cache；
mm encoder TP mode。
```

---

## 25. 容易疑惑的点

### 25.1 vLLM model class 是 HuggingFace model class 吗？

通常不是。

HF config 里的 `architectures` 只是选择依据，vLLM 会通过 `ModelRegistry` 映射到自己的实现类。

例如：

```text
HF architecture: LlamaForCausalLM
vLLM class: vllm.model_executor.models.llama.LlamaForCausalLM
```

### 25.2 为什么不直接 `AutoModelForCausalLM.from_pretrained()`？

因为 vLLM 需要自己的执行结构：

```text
TP / PP sharded layers；
KV cache aware Attention；
prefix / static_forward_context；
packed QKV / gate_up projections；
custom logits processor；
custom weight loading；
CUDA graph / torch.compile 友好结构；
LoRA / quantization / speculative / pooling adapters。
```

Transformers backend 只是 fallback 或特定实现路径，不是 vLLM 主路径。

### 25.3 `forward()` 为什么不直接返回 logits？

vLLM generation model 通常：

```text
forward() 返回 hidden_states；
ModelRunner 根据 logits_indices 选择需要采样的位置；
compute_logits() 才计算 logits。
```

这样可以避免对不需要采样的位置计算完整 vocab logits。

### 25.4 为什么构造函数要传 `prefix`？

因为 prefix 同时用于：

```text
权重名；
量化配置匹配；
attention layer name；
KV cache spec key；
static_forward_context；
LoRA / reload metadata。
```

没有稳定 prefix，权重加载、KV cache 绑定和编译上下文都会出问题。

### 25.5 PP 下为什么模型里有 PPMissingLayer？

为了保持全局模块名和层号一致，同时避免每个 PP rank 都实例化全量层。

它使得：

```text
rank-local module graph
  仍保留全局层编号；
  但只真实构造本 rank 负责的层。
```

### 25.6 pooling 模型是不是完全不同的模型？

不一定。

很多 pooling / embedding / classification 模型是由 generation model 动态包装而来：

```text
generation backbone
  - lm_head / logits_processor
  + pooler / score head
```

### 25.7 Attention 的 KV cache 是构造时就分配好的吗？

不是。

Attention 构造时定义 KV cache 需求和 layer_name，并注册到 `static_forward_context`；真实 KV cache tensor 通常由 ModelRunner 在 `initialize_kv_cache()` 阶段根据 `get_kv_cache_spec()` 分配和绑定。

---

## 26. 从“回答问题”的角度总结

如果要问：

```text
一个 vLLM model class 是如何构造的？
```

可以回答：

```text
vLLM 先通过 ModelConfig 读取 HF config.architectures，并结合 runner_type、convert_type、model_impl 让 ModelRegistry 解析出实际使用的 vLLM model class。
随后 ModelLoader 在目标 device / dtype 上调用 initialize_model()，也就是执行 model_class(vllm_config, prefix)。
模型类内部再根据 HF config 和 VllmConfig 构造 embedding、decoder layers、attention、MLP、norm、lm_head 或 pooler。
构造过程中会注入 TP / PP / quantization / cache / compilation / LoRA / multimodal 等运行配置；attention 层会注册 static_forward_context 并定义 KV cache spec；PP 场景下非本 rank 的层用 PPMissingLayer 占位。
最后 loader 读取 checkpoint，通过 model.load_weights() 做权重名映射、TP shard 加载、PP missing 参数跳过和量化后处理，返回可执行的 model.eval()。
```

最小主线：

```text
HF config.architectures
  → ModelConfig.runner_type / convert_type
  → ModelRegistry.resolve_model_cls()
  → initialize_model()
  → model_class(vllm_config, prefix)
  → embedding / layers / norm / lm_head-or-pooler
  → load_weights()
  → ModelRunner.forward / compute_logits / pooler
```

---

## 27. 最关键流程图

```text
ModelConfig
  ├─ get_config() / hf_config
  ├─ model_arch_config.architectures
  ├─ ModelRegistry.inspect_model_cls()
  ├─ runner_type
  └─ convert_type
        │
        ▼
GPUModelRunner.load_model()
  → get_model_loader(load_config)
  → BaseModelLoader.load_model()
        │
        ├─ initialize_model()
        │    ├─ get_model_architecture()
        │    │    └─ ModelRegistry.resolve_model_cls()
        │    ├─ configure_quant_config()
        │    ├─ set_current_vllm_config()
        │    └─ model_class(vllm_config, prefix)
        │
        ├─ model.__init__()
        │    ├─ read hf_config / quant_config / cache_config
        │    ├─ build embedding
        │    ├─ make_layers()
        │    │    ├─ PPMissingLayer before start_layer
        │    │    ├─ real decoder layers for this PP rank
        │    │    └─ PPMissingLayer after end_layer
        │    ├─ build attention / mlp / norms
        │    ├─ build final norm
        │    └─ build lm_head + logits_processor or pooler
        │
        ├─ load_weights()
        │    ├─ checkpoint iterator
        │    ├─ name remap / packed params
        │    ├─ TP shard copy
        │    ├─ PP missing skip
        │    └─ quant postprocess
        │
        └─ model.eval()
```

---

## 28. 最关键对象关系

```text
ModelConfig
  决定 architecture、runner_type、convert_type、dtype、pooler/multimodal 配置。

ModelRegistry
  把 HF architecture 名解析成 vLLM Python class。

BaseModelLoader / DefaultModelLoader
  负责实例化 model、读取 checkpoint、加载权重、后处理。

VllmConfig
  是 model class 构造时的总配置入口，包含 model/cache/parallel/quant/load/compile 等配置。

prefix
  是模块名、权重名、attention layer_name、KV cache key、量化匹配的统一命名线索。

LlamaForCausalLM / *ForCausalLM
  generation 模型外壳，包含 backbone、lm_head、logits_processor、compute_logits。

LlamaModel / backbone
  负责 embedding、layers、final norm 和 PP intermediate tensors。

Attention
  定义 attention backend、KV cache spec、layer_name 和运行时 forward 入口。

PPMissingLayer / StageMissingLayer
  表示当前模型实例中逻辑存在但本 rank / 当前任务不需要真实构造的模块。

AutoWeightsLoader
  把 checkpoint 权重加载进 vLLM 自定义 module graph。
```

---

## 29. 一句话总结

```text
vLLM model class construction 不是简单创建一个 HF 模型，而是先解析 architecture，再选择 vLLM 实现类，并在构造函数里把 HF config、并行策略、KV cache、attention backend、量化、pooling / multimodal adapter 和权重加载规则一起编织成可高性能推理的 rank-local module graph。
```
