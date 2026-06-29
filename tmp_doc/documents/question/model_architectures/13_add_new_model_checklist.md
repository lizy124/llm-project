# 13. 新增一个模型架构需要实现什么？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/models/utils.py`
- `code/vllm/vllm/model_executor/models/config.py`
- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/tests/models/registry.py`
- `code/vllm/tests/models/`
- `code/vllm/docs/contributing/model/`
- `code/vllm/docs/models/supported_models.md`

本问题关注：向 vLLM 新增一个模型架构时，到底需要实现哪些类、方法、注册项、权重加载逻辑、并行/量化/多模态适配，以及如何验证它真的能被 vLLM 加载和执行。

---

## 1. 一句话回答

新增模型架构的核心不是“把 HuggingFace 的 forward 复制进来”，而是：

```text
HF config.architectures
  → registry.py 中的架构名映射
  → vLLM model class 初始化
  → forward(input_ids, positions, intermediate_tensors, inputs_embeds)
  → vLLM Attention / Linear / Embedding / LogitsProcessor
  → load_weights() 把 HF checkpoint 映射到 vLLM 参数
  → tests/models/registry.py 提供 HF 样例
  → correctness / TP / PP / quant / LoRA / multimodal 等验证
```

最小可运行目标是：

```text
1. vLLM 能根据 config.json 里的 architectures 找到你的类；
2. initialize_model() 能实例化模型；
3. loader 能调用 model.load_weights() 成功加载权重；
4. GPUModelRunner 能调用 forward 并得到 hidden states；
5. generation 模型能通过 compute_logits() 得到 logits；
6. pooling / multimodal / hybrid / mamba 等模型按各自协议补齐额外接口。
```

---

## 2. 最小主链路

新增模型接入后的运行链路可以记成：

```text
config.json
  → architectures: ["YourModelForCausalLM"]
  → ModelRegistry.resolve_model_cls()
  → initialize_model(vllm_config, prefix="")
  → YourModelForCausalLM(vllm_config=..., prefix="")
  → loader.load_weights()
  → model.load_weights(weights)
  → Worker / GPUModelRunner 持有 self.model
  → self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
  → hidden_states / IntermediateTensors
  → model.compute_logits(hidden_states[logits_indices])
  → sampler / ModelRunnerOutput
```

对应源码入口：

```text
registry.py:
  _VLLM_MODELS / ModelRegistry
  resolve_model_cls()
  inspect_model_cls()

model_loader/utils.py:
  initialize_model()
  get_model_architecture()

model_loader/default_loader.py:
  load_weights()
  model.load_weights(...)

v1/worker/gpu_model_runner.py:
  _model_forward()
  execute_model()
  sample_tokens()
```

关键边界是：

```text
registry 只负责找到类；
model_loader 负责实例化和加载权重；
model class 负责 forward / compute_logits / load_weights；
ModelRunner 负责准备输入、attention metadata、调用模型和采样。
```

---

## 3. 实现清单总览

如果只看任务清单，新增一个普通 decoder-only generation 模型通常要做：

```text
1. 确认 HF config 中的 architectures 名称。
2. 选择相似模型作为模板，例如 llama.py / qwen2.py / mistral.py。
3. 在 vllm/model_executor/models/ 下新增或复用模型文件。
4. 实现 Attention / MLP / DecoderLayer / BaseModel / ForCausalLM。
5. 所有 vLLM 子模块构造时传入正确 prefix。
6. 使用 vLLM 的并行层替换 torch.nn.Linear / Embedding。
7. 使用 vLLM Attention 层，不在模型里手写 KV cache 管理。
8. 实现 forward(input_ids, positions, intermediate_tensors, inputs_embeds)。
9. generation 模型实现 compute_logits(hidden_states)。
10. 实现 load_weights(weights)。
11. 如果支持 TP，处理 QKV / gate_up / LM head / embedding 分片。
12. 如果支持 PP，继承 SupportsPP，使用 make_layers 和 IntermediateTensors。
13. 如果支持 LoRA，继承 SupportsLoRA，补 packed_modules_mapping / embedding_modules。
14. 如果是 pooling / reward / classification，接入对应 adapter 或 pooler。
15. 如果是 multimodal，继承 SupportsMultiModal 并实现处理信息、dummy inputs、processor 注册。
16. 如果是 Mamba / hybrid / attention-free，补 interfaces 和 runtime config。
17. 在 registry.py 的正确模型类别字典中注册 architecture。
18. 在 tests/models/registry.py 添加 HF example model。
19. 必要时在 models/config.py 添加 VerifyAndUpdateConfig。
20. 更新 docs/models/supported_models.md。
21. 添加或复用 correctness tests。
22. 跑模型加载、最小 generate、HF 对齐、TP/PP/量化/LoRA 等验证。
```

一句话压缩：

```text
模型代码、注册映射、权重加载、运行协议、测试样例、文档列表六件事必须一起完成。
```

---

## 4. 第一步：确认 HF architecture name

vLLM 默认从 HuggingFace config 里读取：

```json
{
  "architectures": ["YourModelForCausalLM"]
}
```

这个字符串就是注册表里的 key。

例如：

```text
"LlamaForCausalLM"
  → registry.py: _TEXT_GENERATION_MODELS["LlamaForCausalLM"]
  → ("llama", "LlamaForCausalLM")
  → vllm.model_executor.models.llama.LlamaForCausalLM

"Qwen2ForCausalLM"
  → registry.py: _TEXT_GENERATION_MODELS["Qwen2ForCausalLM"]
  → ("qwen2", "Qwen2ForCausalLM")
  → vllm.model_executor.models.qwen2.Qwen2ForCausalLM
```

如果一个 HF repo 的 config 中有多个 architecture：

```text
architectures: ["A", "B"]
```

vLLM 会按顺序尝试解析，直到找到能 inspect / load 的模型类。

需要确认：

```text
1. config.json 里的 architectures 是否稳定；
2. Transformers 里的 class name 是否和 config 一致；
3. 是否需要 trust_remote_code；
4. 是否已有其他 architecture 可以复用现有 vLLM 实现；
5. 是否是 generation / embedding / rerank / reward / classification / multimodal / speculative model。
```

常见情况：

```text
情况 A：新模型只是 Llama 变体
  → 可能不需要新增模型文件，只需 registry 映射到 LlamaForCausalLM。

情况 B：权重命名相似但结构略有差异
  → 可继承或复用已有模型文件，少量改 attention / mlp / config。

情况 C：结构完全不同
  → 新增独立 models/xxx.py。

情况 D：Transformers backend 已兼容
  → 可能先走 Transformers backend，而不是 in-tree vLLM native model。
```

---

## 5. registry.py 要改哪里

注册表在：

```text
vllm/model_executor/models/registry.py
```

内置模型主要分组：

```text
_TEXT_GENERATION_MODELS
_EMBEDDING_MODELS
_LATE_INTERACTION_MODELS
_REWARD_MODELS
_TOKEN_CLASSIFICATION_MODELS
_SEQUENCE_CLASSIFICATION_MODELS
_MULTIMODAL_MODELS
_SPECULATIVE_DECODING_MODELS
_TRANSFORMERS_SUPPORTED_MODELS
_TRANSFORMERS_BACKEND_MODELS
```

最后合并为：

```python
_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_LATE_INTERACTION_MODELS,
    **_REWARD_MODELS,
    **_TOKEN_CLASSIFICATION_MODELS,
    **_SEQUENCE_CLASSIFICATION_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
    **_TRANSFORMERS_SUPPORTED_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}
```

位置：`registry.py:692` 到 `registry.py:703`

注册项格式是：

```python
"HFArchitectureName": ("module_name", "VllmClassName")
```

例如：

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM")
"Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM")
```

位置：`registry.py:152`、`registry.py:198`

含义是：

```text
module_name = llama
  → vllm.model_executor.models.llama

class_name = LlamaForCausalLM
  → getattr(module, "LlamaForCausalLM")
```

如果 module_name 以 `vllm.` 开头，则会按完整模块路径解析：

```text
"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")
```

`_resolve_module_name()` 会处理这个逻辑。

位置：`registry.py:1368` 到 `registry.py:1374`

---

## 6. registry 的解析流程

vLLM 不是直接 import 所有模型，而是懒加载。

初始化注册表：

```python
ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=_resolve_module_name(mod_relname),
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)
```

位置：`registry.py:1377` 到 `registry.py:1385`

解析模型类时：

```text
ModelConfig.architectures
  → ModelRegistry.resolve_model_cls(architectures, model_config)
  → _normalize_arch()
  → _try_load_model_cls()
  → importlib.import_module(module_name)
  → getattr(module, class_name)
```

位置：`registry.py:1225` 到 `registry.py:1277`

inspect 模型能力时：

```text
ModelRegistry.inspect_model_cls()
  → _try_inspect_model_cls()
  → _ModelInfo.from_model_cls(model_cls)
  → 判断是否 generation / pooling / multimodal / PP / hybrid 等
```

位置：`registry.py:1173` 到 `registry.py:1223`

`_ModelInfo` 会读取这些能力：

```text
is_text_generation_model
is_pooling_model
attn_type
supports_multimodal
supports_pp
has_inner_state
is_attention_free
is_hybrid
supports_transcription
```

位置：`registry.py:745` 到 `registry.py:795`

所以新增模型类的继承和 class attributes 会直接影响：

```text
调度选择；
attention backend 选择；
是否允许 PP；
是否按 pooling model 处理；
是否需要多模态处理；
是否有 Mamba state；
是否可以作为 transcription 模型。
```

---

## 7. 模型文件的基本结构

普通 decoder-only 模型一般拆成五层：

```text
YourMLP
YourAttention
YourDecoderLayer
YourModel
YourForCausalLM
```

参考 `llama.py`：

```text
LlamaMLP
LlamaAttention
LlamaDecoderLayer
LlamaModel
LlamaForCausalLM
```

参考 `qwen2.py`：

```text
Qwen2MLP
Qwen2Attention
Qwen2DecoderLayer
Qwen2Model
Qwen2ForCausalLM
```

推荐结构：

```python
class YourMLP(nn.Module):
    ...

class YourAttention(nn.Module):
    ...

class YourDecoderLayer(nn.Module):
    ...

@support_torch_compile(...)
class YourModel(nn.Module):
    ...

class YourForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    ...
```

其中：

```text
YourModel：
  负责 embeddings、decoder layers、final norm、PP 中间张量。

YourForCausalLM：
  负责 lm_head、logits_processor、compute_logits、load_weights、对外 forward。
```

不要把所有逻辑堆在一个类里。vLLM 的并行、加载、LoRA、multimodal wrapper 都依赖这种清晰分层。

---

## 8. __init__ 的要求：prefix 必须贯穿

vLLM 要求模型内部 vLLM 模块构造时带 `prefix`。

原因：

```text
1. Attention layer 需要唯一 layer name；
2. 非均匀量化需要按 prefix 匹配量化配置；
3. 权重加载需要 prefix 和 state_dict 名称对齐；
4. LoRA / packed module mapping 也依赖模块路径。
```

典型写法：

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=self.total_num_heads,
    total_num_kv_heads=self.total_num_kv_heads,
    quant_config=quant_config,
    prefix=f"{prefix}.qkv_proj",
)

self.attn = Attention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    cache_config=cache_config,
    quant_config=quant_config,
    prefix=f"{prefix}.attn",
)
```

Llama 的 attention 初始化就是这个模式。

位置：`llama.py:165` 到 `llama.py:222`

常见错误：

```text
错误 1：直接使用 torch.nn.Linear，不传 prefix。
错误 2：多个 Attention 共享同一个 prefix。
错误 3：prefix 和 checkpoint 名称不一致，导致 load_weights 映射复杂化。
错误 4：外层 prefix 为空时生成 ".model" 这类错误名字。
```

推荐使用：

```python
maybe_prefix(prefix, "model")
```

位置：`utils.py:724` 到 `utils.py:734`

---

## 9. vLLM Linear / Embedding 层怎么选

新增模型通常不要直接用 HuggingFace 里的 `nn.Linear` 和 `nn.Embedding`。

常用替换：

```text
torch.nn.Embedding
  → VocabParallelEmbedding

LM head
  → ParallelLMHead

q_proj / k_proj / v_proj
  → QKVParallelLinear

gate_proj / up_proj
  → MergedColumnParallelLinear

down_proj / o_proj
  → RowParallelLinear

普通 column split
  → ColumnParallelLinear

不切分但需要统一接口
  → ReplicatedLinear
```

Llama / Qwen2 的典型组合：

```text
Attention:
  QKVParallelLinear
  RowParallelLinear

MLP:
  MergedColumnParallelLinear
  RowParallelLinear

Embedding:
  VocabParallelEmbedding

LM head:
  ParallelLMHead
```

这样可以同时获得：

```text
Tensor Parallel；
量化参数注入；
packed weight loading；
LoRA 模块识别；
统一参数命名。
```

如果模型结构特殊，例如：

```text
QKV 未融合；
q/k/v head_dim 不一致；
MLA；
MoE；
Mamba；
Linear Attention；
多 Query group；
shared expert；
```

需要优先找类似模型参考，而不是强行套 Llama 模板。

---

## 10. Attention 层怎么接入

vLLM 的模型 forward 中通常不显式传 attention metadata。

attention metadata 来自：

```text
GPUModelRunner._build_attention_metadata()
  → set_forward_context(attn_metadata, ...)
  → 模型内部 Attention layer 从 forward context 使用 metadata
```

因此模型内部只需要：

```python
attn_output = self.attn(q, k, v)
```

不需要自己传：

```text
block_table
slot_mapping
kv_cache
attn_metadata
cache_position
past_key_values
```

Llama attention 的核心：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

位置：`llama.py:224` 到 `llama.py:234`

这说明：

```text
模型层只负责 q/k/v、RoPE、调用 Attention；
KV cache 写入位置、prefix cache、chunked prefill、decode metadata 都由外层 ModelRunner 和 Attention backend 管。
```

---

## 11. RoPE / position 处理

模型 forward 会收到：

```python
positions: torch.Tensor
```

普通 decoder-only 模型通常直接把它传给 rotary embedding：

```python
q, k = self.rotary_emb(positions, q, k)
```

RoPE 初始化通常用：

```python
self.rotary_emb = get_rope(
    self.head_dim,
    max_position=max_position_embeddings,
    rope_parameters=getattr(config, "rope_parameters", None),
    is_neox_style=True,
)
```

位置：`llama.py:236` 到 `llama.py:248`

Qwen2 还会处理：

```text
set_default_rope_theta(config, default_theta=1000000)
dual_chunk_attention_config
qk_norm
```

位置：`qwen2.py:239` 到 `qwen2.py:279`

需要检查：

```text
1. config 中是 rope_scaling 还是 rope_parameters；
2. 是否需要 default rope theta；
3. 是否是 neox style；
4. 是否支持 partial rotary；
5. 是否有 M-RoPE，多维 positions；
6. 是否有 dual chunk attention；
7. 是否有 interleaved sliding window；
8. max_position_embeddings 是否需要修正。
```

如果 position 形状不是一维，例如多模态 M-RoPE，`support_torch_compile` 的 dynamic_arg_dims 也要跟着改。

Qwen2 的注释：

```text
positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
otherwise (seq_len, ).
```

位置：`qwen2.py:315` 到 `qwen2.py:323`

---

## 12. BaseModel.forward 的标准形态

普通 decoder-only base model 的 forward 建议长这样：

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    ...
```

关键点：

```text
input_ids：
  本 step 执行的 token ids，已经是 flattened tokens。

positions：
  每个 token 的位置，也按 flattened tokens 对齐。

inputs_embeds：
  多模态、prompt embeds、embedding input 场景可能直接传 embedding。

intermediate_tensors：
  Pipeline Parallel 非首 rank 接收上一 stage 输出。
```

LlamaModel 的处理方式：

```text
如果是 first PP rank：
  inputs_embeds 优先；否则 embed_input_ids(input_ids)。

如果不是 first PP rank：
  从 intermediate_tensors 里取 hidden_states / residual。

遍历本 rank 负责的 layers。

如果不是 last PP rank：
  返回 IntermediateTensors。

如果是 last PP rank：
  final norm 后返回 hidden_states。
```

位置：`llama.py:392` 到 `llama.py:431`

对应心智模型：

```text
PP rank 0:
  input_ids / inputs_embeds → embeddings → partial layers → IntermediateTensors

PP middle:
  IntermediateTensors → partial layers → IntermediateTensors

PP last:
  IntermediateTensors → final layers → norm → hidden_states
```

---

## 13. ForCausalLM.forward 和 compute_logits

外层 `*ForCausalLM` 通常只做两件事：

```text
1. forward 调 base model，返回 hidden_states 或 IntermediateTensors；
2. compute_logits 用 lm_head + logits_processor 把 hidden_states 转 logits。
```

LlamaForCausalLM：

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    model_output = self.model(
        input_ids, positions, intermediate_tensors, inputs_embeds
    )
    return model_output
```

位置：`llama.py:550` 到 `llama.py:560`

logits：

```python
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

位置：`llama.py:562` 到 `llama.py:567`

注意：

```text
模型 forward 通常不返回 logits；
GPUModelRunner 会先取 hidden_states[logits_indices]；
然后调用 model.compute_logits(sample_hidden_states)。
```

这和执行层文档中的 forward / logits 主链路对应。

---

## 14. load_weights 的职责

`load_weights()` 是新增模型最容易出错的部分。

模型加载链路中，loader 最终会调用：

```python
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`default_loader.py:415` 到 `default_loader.py:427`

`load_weights()` 的输入是：

```text
Iterable[tuple[str, torch.Tensor]]
```

每个元素对应 checkpoint 里的一个 weight name 和 tensor。

它需要返回：

```text
set[str]
```

表示成功加载的参数名。

普通情况可以用：

```python
loader = AutoWeightsLoader(
    self,
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
)
return loader.load_weights(weights)
```

位置：`llama.py:569` 到 `llama.py:574`

但是如果 checkpoint 名字和 vLLM 参数名字不一致，就要自定义映射。

---

## 15. QKV / gate_up fused 权重加载

vLLM 常把 HF 中分开的参数融合成一个并行层：

```text
HF checkpoint:
  q_proj.weight
  k_proj.weight
  v_proj.weight

vLLM parameter:
  qkv_proj.weight
```

```text
HF checkpoint:
  gate_proj.weight
  up_proj.weight

vLLM parameter:
  gate_up_proj.weight
```

典型映射：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

位置：`llama.py:433` 到 `llama.py:441`

加载时核心逻辑：

```text
1. 遍历 checkpoint weight；
2. 如果命中 q_proj/k_proj/v_proj，替换成 qkv_proj；
3. 根据 shard_id = q/k/v 调对应 param.weight_loader；
4. 如果命中 gate_proj/up_proj，替换成 gate_up_proj；
5. 根据 shard_id = 0/1 加载到 merged column parallel 参数；
6. 其他参数走 default_weight_loader。
```

位置：`llama.py:444` 到 `llama.py:482`

检查点：

```text
- q/k/v 的输出维度顺序是否和 QKVParallelLinear 一致；
- GQA / MQA 下 KV heads 是否正确切分或复制；
- gate/up 的顺序是否和激活函数实现一致；
- bias 是否存在；
- GPTQ / AWQ / FP8 的 scale / zero_point 是否需要重命名；
- tied embeddings 时 lm_head 是否跳过。
```

---

## 16. PP 下 load_weights 要跳过缺失层

Pipeline Parallel 下，每个 rank 只持有一部分层。

其他层会是：

```text
PPMissingLayer
StageMissingLayer
```

创建层时用：

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
    prefix=f"{prefix}.layers",
)
```

位置：`llama.py:375` 到 `llama.py:379`

`make_layers()` 会根据当前 PP rank 创建：

```text
前面缺失层：PPMissingLayer
当前 rank 负责的层：真实 layer
后面缺失层：PPMissingLayer
```

位置：`utils.py:640` 到 `utils.py:672`

所以 load_weights 时必须跳过当前 rank 不存在的参数：

```python
if is_pp_missing_parameter(name, self):
    continue
```

位置：`llama.py:464` 到 `llama.py:477`

否则 PP 启动时会出现：

```text
某些 rank 找不到参数；
某些 rank 试图加载不属于自己的 layer；
named_parameters 和 checkpoint 不一致。
```

---

## 17. tied weights / lm_head 的处理

如果 HF config：

```text
tie_word_embeddings = true
```

通常代表：

```text
lm_head.weight 和 embed_tokens.weight 共享。
```

vLLM 中常见处理：

```python
if config.tie_word_embeddings:
    self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
```

或：

```python
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens
```

Llama 位置：`llama.py:518` 到 `llama.py:527`

Qwen2 位置：`qwen2.py:509` 到 `qwen2.py:520`

load_weights 时通常跳过：

```python
skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None)
```

位置：`llama.py:569` 到 `llama.py:574`

检查点：

```text
1. checkpoint 是否真的包含 lm_head.weight；
2. tied 情况是否重复加载；
3. PP last rank 是否需要 lm_head；
4. first rank 是否需要 embed_tokens；
5. vocab size padding 后的权重 shape 是否兼容。
```

---

## 18. Tensor Parallel 支持检查

如果模型要支持 TP，需要检查：

```text
Attention:
  total_num_heads % tp_size == 0
  KV heads >= TP size 时按 TP 分片
  KV heads < TP size 时按 TP 复制

MLP:
  gate/up column parallel
  down row parallel

Embedding:
  vocab parallel embedding
  lm_head parallel

Output:
  row parallel 层是否 reduce_results
  logits_processor 是否接 ParallelLMHead
```

Llama attention 中的 KV head 逻辑：

```text
如果 total_num_kv_heads >= tp_size：
  KV heads 跨 TP 分片。

否则：
  KV heads 在多个 TP rank 上复制。
```

位置：`llama.py:143` 到 `llama.py:157`

典型问题：

```text
1. num_attention_heads 不能整除 TP size；
2. num_key_value_heads 复制逻辑遗漏；
3. head_dim 不是 hidden_size / num_heads；
4. q_size / kv_size split 错；
5. RowParallelLinear 不 reduce 导致输出未 all-reduce；
6. 量化 loader 没按 shard_id 加载。
```

---

## 19. Pipeline Parallel 支持检查

如果模型支持 PP，外层类通常继承：

```python
SupportsPP
```

位置：`interfaces.py:617` 到 `interfaces.py:719`

还需要：

```text
1. forward 接受 intermediate_tensors 参数；
2. 非 first rank 从 intermediate_tensors 取输入；
3. 非 last rank 返回 IntermediateTensors；
4. last rank 返回 hidden_states；
5. 实现 make_empty_intermediate_tensors；
6. embedding / norm / lm_head 在非对应 rank 用 PPMissingLayer；
7. load_weights 跳过 PP missing parameter。
```

LlamaModel 中：

```text
embed_tokens:
  first rank 或 tied embedding 的 last rank 才创建真实层；否则 PPMissingLayer。

layers:
  make_layers 切分。

norm:
  last rank 才创建真实 norm；否则 PPMissingLayer。

lm_head:
  last rank 才创建真实 lm_head；否则 PPMissingLayer。
```

位置：`llama.py:365` 到 `llama.py:386`，`llama.py:518` 到 `llama.py:533`

PP forward 返回：

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

位置：`llama.py:422` 到 `llama.py:425`

---

## 20. LoRA 支持检查

如果模型支持 LoRA，需要继承：

```python
SupportsLoRA
```

位置：`interfaces.py:538` 到 `interfaces.py:615`

并提供：

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

Llama 示例：

位置：`llama.py:486` 到 `llama.py:499`

这些字段告诉 LoRA：

```text
HF adapter 里的 q_proj/k_proj/v_proj 应该打到 vLLM 的 qkv_proj；
HF adapter 里的 gate_proj/up_proj 应该打到 gate_up_proj；
embedding 和 lm_head 对应输入/输出 embedding 模块。
```

如果模型是 MoE，还要关注：

```text
is_3d_moe_weight
is_non_gated_moe
lora_skip_prefixes
```

不要只因为模型有 Linear 就宣称支持 LoRA。必须确认 adapter 命名、packed mapping 和目标模块一致。

---

## 21. Quantization 支持检查

量化支持依赖两部分：

```text
1. 初始化时把 quant_config 传给 vLLM layer；
2. load_weights 时正确加载 quantized weight / scale / zero point。
```

初始化示例：

```python
quant_config = vllm_config.quant_config

self.qkv_proj = QKVParallelLinear(..., quant_config=quant_config, ...)
self.down_proj = RowParallelLinear(..., quant_config=quant_config, ...)
self.embed_tokens = VocabParallelEmbedding(..., quant_config=quant_config)
self.lm_head = ParallelLMHead(..., quant_config=quant_config)
```

权重加载中常见处理：

```python
if "scale" in name or "zero_point" in name:
    name = maybe_remap_kv_scale_name(name, params_dict)
    if name is None:
        continue
```

位置：`llama.py:451` 到 `llama.py:455`

`AutoWeightsLoader` 也会根据 quant_config 的 cache scale mapper 处理映射。

位置：`utils.py:348` 到 `utils.py:377`

检查点：

```text
- FP8 KV scale 名称是否匹配；
- GPTQ extra bias 是否要跳过；
- AWQ / GPTQ / bitsandbytes loader 是否需要特殊逻辑；
- fused QKV 的 scale 是否对应 shard；
- lm_head 是否量化；
- reload from high precision weights 是否可用。
```

---

## 22. Pooling / embedding / reward / classification 模型

不是所有模型都是 generation。

registry 中有多个类别：

```text
_EMBEDDING_MODELS
_LATE_INTERACTION_MODELS
_REWARD_MODELS
_TOKEN_CLASSIFICATION_MODELS
_SEQUENCE_CLASSIFICATION_MODELS
```

位置：`registry.py:220` 到 `registry.py:335`

这类模型通常不走：

```text
hidden_states → compute_logits → sampler
```

而是走：

```text
hidden_states → pooler / classifier / scorer → ModelRunnerOutput.pooler_output
```

常见做法是使用 adapters：

```python
from .adapters import as_embedding_model, as_seq_cls_model

class LlamaBidirectionalForSequenceClassification(
    as_seq_cls_model(LlamaForCausalLM)
):
    pass

class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    pass
```

位置：`llama.py:577` 到 `llama.py:586`

这类模型要额外检查：

```text
1. registry 放到正确类别；
2. interfaces_base 能识别为 pooling model；
3. pooler config 默认值是否正确；
4. attention 是否 causal / bidirectional；
5. output shape 是否符合 embeddings / score / classification；
6. tests 使用 pooling / rerank / classification 的测试工具。
```

---

## 23. Multimodal 模型额外要做什么

多模态模型必须继承：

```python
SupportsMultiModal
```

位置：`interfaces.py:95` 到 `interfaces.py:411`

核心要求：

```text
1. get_placeholder_str()
2. embed_multimodal()
3. get_language_model()
4. _mark_language_model()
5. _mark_tower_model()
6. 多模态 processor / processing info / dummy inputs
7. placeholder token 和 multimodal embeddings 合并逻辑
```

官方 contributing 文档要求：

```text
- language model 组件放在 _mark_language_model 里初始化；
- vision/audio/tower 组件放在 _mark_tower_model 里初始化；
- forward 中移除多模态 embedding 生成逻辑；
- embed_multimodal 返回 3D tensor 或 2D tensor list/tuple；
- text embedding 和 multimodal merge 通常由 SupportsMultiModal.embed_input_ids 默认实现处理。
```

多模态模型的主链路是：

```text
processor / input mapper
  → placeholder ranges
  → embed_multimodal(**kwargs)
  → embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
  → language_model(..., inputs_embeds=...)
  → hidden_states
```

常见验证：

```text
- text + image/audio/video；
- token ids + multimodal data；
- cached multimodal data；
- 多图 / 多视频限制；
- dummy input 是否覆盖最坏显存；
- mm encoder only / tower LoRA / connector LoRA。
```

---

## 24. Mamba / Hybrid / Attention-free 模型额外要做什么

如果模型不是纯 attention decoder，要看它属于哪类：

```text
Attention-free：
  例如 Mamba，只使用 recurrent state，不使用 KV attention。

Hybrid：
  attention + mamba / linear attention / short conv 混合。

Mamba-like：
  有原地更新 state，而不是像 KV cache 那样 append。
```

相关 interfaces：

```text
HasInnerState
IsAttentionFree
IsHybrid
```

位置：`interfaces.py:737` 到 `interfaces.py:844`

这类模型通常要实现：

```text
1. state shape / dtype 计算；
2. state copy function；
3. attention backend 或 mamba backend；
4. prefix caching 行为；
5. MODELS_CONFIG_MAP 默认配置；
6. torch compile / CUDA graph custom op；
7. block size / page size 对齐。
```

`models/config.py` 中有 Mamba / Hybrid 相关默认配置。

位置：`config.py:404` 起，`config.py:671` 起

普通 Llama/Qwen 类模型不需要这些，但如果模型有 recurrent state，不补这些会导致：

```text
KV cache 管理错误；
prefix cache 行为错误；
CUDA graph 捕获失败；
state copy / preemption 异常；
调度器无法正确估算 cache。
```

---

## 25. models/config.py 什么时候要改

`vllm/model_executor/models/config.py` 提供模型级 config 修正入口。

它定义：

```python
class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config):
        return

    @staticmethod
    def verify_and_update_model_config(model_config):
        return
```

位置：`config.py:17` 到 `config.py:24`

最后通过：

```python
MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {...}
```

位置：`config.py:671`

适合放这里的逻辑：

```text
1. 修正 HF config 中 vLLM 需要的字段；
2. 根据模型强制 attention backend；
3. 设置默认 CUDA graph capture size；
4. 禁用某些不兼容优化；
5. 修正 quantization_config；
6. 设置 pooling 默认行为；
7. 为 Mamba / hybrid 模型设置 runtime defaults。
```

不适合放这里的逻辑：

```text
- 权重名映射；
- forward 里的 shape hack；
- 大量模型实现逻辑；
- 和用户显式参数冲突的强制覆盖。
```

判断标准：

```text
如果这个修正是模型启动前就能从 config 判断出来的运行时默认值，可以放 models/config.py。
如果这个修正依赖具体 weight tensor 或 forward 输入，应放模型实现或 loader。
```

---

## 26. Transformers backend 和 native vLLM backend 怎么选

registry 中有：

```text
_TRANSFORMERS_SUPPORTED_MODELS
_TRANSFORMERS_BACKEND_MODELS
```

位置：`registry.py:647` 到 `registry.py:690`

如果模型的 Transformers 实现已兼容 vLLM Transformers backend，可以先通过这条路径接入。

解析流程中会尝试：

```text
model_config.model_impl == "transformers"
  → _try_resolve_transformers()

model_config.model_impl == "auto"
  → native registry 不命中时 fallback 到 transformers impl
```

位置：`registry.py:1077` 到 `registry.py:1145`

适合 Transformers backend 的情况：

```text
1. 模型较新，先要能跑；
2. native vLLM 高性能实现还没写；
3. Transformers model class 已声明 backend compatible；
4. 用户愿意接受性能/功能限制。
```

适合 native vLLM implementation 的情况：

```text
1. 要支持高性能 serving；
2. 要支持 TP / PP；
3. 要支持 vLLM attention backend / CUDA graph；
4. 要支持量化、LoRA、prefix cache、spec decode；
5. 模型会长期维护在 vLLM 中。
```

---

## 27. tests/models/registry.py 必须更新

`registry.py` 文件顶部明确写着：

```text
Whenever you add an architecture to this page, please also update
tests/models/registry.py with example HuggingFace models for it.
```

位置：`registry.py:3` 到 `registry.py:6`

测试 registry 的作用是：

```text
为每个 architecture 提供一个 HF repo；
让 CI 能加载 dummy weights；
验证模型类可以初始化；
验证 registry 和 supported models 不脱节。
```

测试样例结构：

```python
"Qwen2ForCausalLM": _HfExamplesInfo(
    "Qwen/Qwen2-0.5B-Instruct",
    extras={
        "2.5": "Qwen/Qwen2.5-0.5B-Instruct",
        "2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    },
)
```

位置：`tests/models/registry.py:504` 到 `tests/models/registry.py:510`

`_HfExamplesInfo` 支持：

```text
default
extras
tokenizer
tokenizer_mode
speculative_model
speculative_method
min_transformers_version
max_transformers_version
require_embed_inputs
dtype
enforce_eager
enable_prefix_caching
is_available_online
trust_remote_code
hf_overrides
max_model_len
max_num_batched_tokens
revision
max_num_seqs
use_original_num_layers
```

位置：`tests/models/registry.py:16` 到 `tests/models/registry.py:121`

选择 example model 时优先：

```text
1. 小模型；
2. 公开可下载；
3. 不需要特殊 license gate；
4. Transformers 版本要求明确；
5. 能覆盖真实 architecture；
6. CI 不容易 OOM；
7. 如需 trust_remote_code，要显式标记。
```

---

## 28. correctness tests 怎么加

最低要求是：

```text
tests/models/registry.py 有 example model，模型能用 dummy weights 初始化。
```

更可靠的验证要加 correctness tests。

根据模型类型选择目录：

```text
文本生成：
  tests/models/language/generation/

PPL：
  tests/models/language/generation_ppl_test/

Embedding / pooling：
  tests/models/language/pooling/

Multimodal generation：
  tests/models/multimodal/generation/

Multimodal pooling：
  tests/models/multimodal/pooling/

Multimodal processing：
  tests/models/multimodal/processing/

Quantization：
  tests/models/quantization/ 或 tests/quantization/

LoRA：
  tests/lora/
```

官方 tests 文档把 correctness 分成：

```text
Generative models:
  check_outputs_equal
  check_logprobs_close

Pooling models:
  cosine similarity

Multimodal processing:
  text + data / tokens + data / cached data 等输入组合一致性
```

位置：`docs/contributing/model/tests.md:27` 到 `docs/contributing/model/tests.md:57`

新增模型至少建议跑：

```text
1. 架构注册和 dummy load；
2. 单卡 generate；
3. vLLM vs HF 输出或 logprobs 对齐；
4. eager + CUDA graph；
5. prefix caching；
6. chunked prefill；
7. TP；
8. PP；
9. 量化；
10. LoRA；
11. 长上下文；
12. 多模态输入组合。
```

不是每个模型都要支持所有项，但不支持的能力要明确原因。

---

## 29. docs/models/supported_models.md 要更新

新增内置模型后，要更新 supported models 文档。

入口：

```text
vllm/docs/models/supported_models.md
```

常见 section：

```text
List of Text-only Language Models
List of Multimodal Language Models
Embedding Models
Reward Models
```

例如：

```text
LlamaForCausalLM
Qwen2ForCausalLM
```

位置：`supported_models.md:348` 起，`supported_models.md:434`，`supported_models.md:469`

表格通常要说明：

```text
Architecture name；
Model family；
Example HF models；
是否支持 LoRA；
是否支持 PP / TP / 其他能力。
```

如果只是 out-of-tree plugin，不一定更新内置 supported models，但应该在插件文档中说明注册方式。

---

## 30. 外部插件注册方式

如果不想改 vLLM 源码，可以用插件注册外部模型。

注册代码：

```python
def register():
    from vllm import ModelRegistry
    from your_code import YourModelForCausalLM

    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

如果 import 会初始化 CUDA，推荐懒加载字符串：

```python
def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "YourModelForCausalLM",
        "your_code:YourModelForCausalLM",
    )
```

`register_model()` 支持两种形式：

```text
model_cls 是 nn.Module class；
model_cls 是 "<module>:<class>" 字符串。
```

位置：`registry.py:986` 到 `registry.py:1030`

插件适合：

```text
1. 私有模型；
2. 实验性模型；
3. 还不适合进入 vLLM 主仓库；
4. 需要和特定业务包一起发布。
```

但插件模型仍然必须遵守同样的模型接口：

```text
forward；
compute_logits 或 pooler；
load_weights；
SupportsMultiModal / SupportsPP / SupportsLoRA 等协议。
```

---

## 31. 最小 decoder-only 模型骨架

一个简化骨架如下：

```python
class YourModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: YourDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        ...
```

外层：

```python
class YourForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = YourModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

真实实现要补：

```text
PP first/last rank 判断；
PPMissingLayer；
custom load_weights；
RoPE；
attention；
MLP；
tied embeddings；
quant / LoRA / MoE / multimodal 特殊逻辑。
```

---

## 32. 从 HF modeling 迁移时要删除什么

HuggingFace training/inference 通用实现里有很多 vLLM 不需要的内容。

通常要删或改：

```text
training loss；
labels；
past_key_values 输入输出；
use_cache；
attention_mask；
position_ids 的 batch x seq 形态；
return_dict；
output_attentions；
output_hidden_states 的 HF 结构；
BaseModelOutputWithPast；
CausalLMOutputWithPast；
_prepare_decoder_attention_mask；
generate 相关 helper；
cache_position；
```

vLLM 需要的是：

```text
flattened input_ids；
flattened positions；
inputs_embeds 可选；
IntermediateTensors 可选；
返回 hidden_states 或 IntermediateTensors；
logits 由 compute_logits 单独算。
```

不要把 HF 的 attention_mask / past_key_values 直接搬进来，因为 vLLM 的 KV cache 和 attention metadata 已经由 ModelRunner / Attention backend 管理。

---

## 33. 常见特殊结构检查

### 33.1 Sliding window

检查：

```text
config.sliding_window；
config.layer_types；
max_window_layers；
per_layer_sliding_window；
interleaved sliding window 是否被支持。
```

LlamaAttention 会按 layer_types 判断某层是否 sliding，并把 sliding_window 传给 Attention。

位置：`llama.py:185` 到 `llama.py:220`

### 33.2 MLA

检查：

```text
q/k/v 不是标准 head layout；
KV cache 存储的 latent 形态；
attention backend 是否支持；
load_weights 是否有特殊拆分。
```

建议参考 DeepSeek 系列。

### 33.3 MoE

检查：

```text
expert parallel；
routed experts；
shared experts；
EPLB；
MoE quant；
LoRA 是否支持 MoE 权重；
routed_experts 输出是否需要回传。
```

相关协议：`MixtureOfExperts`。

位置：`interfaces.py:847` 起

### 33.4 Speculative / MTP / EAGLE

检查：

```text
_SPECULATIVE_DECODING_MODELS 注册；
SupportsEagle / SupportsEagle3；
aux hidden states；
MTP layers；
draft model quant config；
load_weights 是否识别自己的 lm_head / embed_tokens。
```

LlamaForCausalLM 继承了：

```text
SupportsEagle
SupportsEagle3
```

位置：`llama.py:486` 到 `llama.py:488`

### 33.5 Encoder-decoder / ASR / transcription

检查：

```text
是否属于 _MULTIMODAL_MODELS；
是否实现 SupportsTranscription；
encoder inputs 如何进入 model_kwargs；
是否支持 transcription only；
processor / tokenizer 是否特殊。
```

---

## 34. 最小验证命令思路

在 vLLM 仓库里，通常从小到大验证：

```text
1. registry / model loading 单测；
2. 单模型 correctness test；
3. 单卡 smoke generate；
4. TP；
5. PP；
6. 量化；
7. LoRA；
8. 多模态 processing / generation；
9. 文档和 lint。
```

项目说明要求 Python 命令通过 uv / .venv 执行。

典型命令形态：

```bash
.venv/bin/python -m pytest tests/models/registry.py -v
.venv/bin/python -m pytest tests/models/language/generation/test_common.py -v
.venv/bin/python -m pytest tests/models/language/generation/test_qwen.py -v
.venv/bin/python -m pytest tests/models/multimodal/generation/test_common.py -v
```

实际要根据新增模型所在测试文件选择。

如果模型很大，要用：

```text
_HfExamplesInfo.max_model_len
_HfExamplesInfo.max_num_batched_tokens
_HfExamplesInfo.max_num_seqs
_HfExamplesInfo.dtype
_HfExamplesInfo.enforce_eager
_HfExamplesInfo.use_original_num_layers
```

避免 CI OOM 或 Transformers 版本不兼容。

---

## 35. PR 级最终检查清单

提交前逐项确认：

```text
模型注册：
  - registry.py 加了 architecture；
  - 放在正确类别；
  - 字母顺序符合该 section 习惯；
  - module / class 名能 import；
  - tests/models/registry.py 同步 example。

模型实现：
  - __init__ 接受 vllm_config 和 prefix；
  - 所有 vLLM layer 都有 prefix；
  - forward 签名符合 vLLM；
  - forward 返回 hidden_states / IntermediateTensors；
  - compute_logits 正确；
  - load_weights 返回 loaded set；
  - tied embeddings 正确；
  - PP missing parameter 正确跳过。

并行和性能：
  - TP head split 正确；
  - PP first/last rank 正确；
  - quant_config 传入所有相关层；
  - support_torch_compile dynamic_arg_dims 合理；
  - CUDA graph / eager 均能跑。

扩展能力：
  - SupportsLoRA 字段完整；
  - SupportsPP 条件完整；
  - SupportsMultiModal 处理完整；
  - Mamba / hybrid state 完整；
  - pooling / reward / classification adapter 正确。

测试文档：
  - registry dummy load 通过；
  - correctness test 覆盖主要路径；
  - supported_models.md 更新；
  - 如需 trust_remote_code / transformers 版本限制已标注；
  - 不支持的能力有说明。
```

---

## 36. 容易疑惑的点

### 36.1 新增模型是不是只要改 registry.py？

不是。

registry 只让 vLLM 找到类。真正能运行还需要：

```text
模型 class；
forward 协议；
compute_logits 或 pooler；
load_weights；
测试样例；
必要的 config 修正；
文档更新。
```

### 36.2 forward 里要不要返回 logits？

普通 generation 模型不要。

vLLM 的主路径是：

```text
forward → hidden_states
GPUModelRunner 取 hidden_states[logits_indices]
compute_logits → logits
sample_tokens → ModelRunnerOutput
```

### 36.3 attention_mask / past_key_values 怎么处理？

通常不从 HF 代码里保留。

vLLM 使用：

```text
SchedulerOutput
  → slot_mapping / block table
  → attention metadata
  → set_forward_context
  → Attention(q, k, v)
```

模型内部不直接管理 HF 风格的 past_key_values。

### 36.4 load_weights 能不能直接 strict load state_dict？

通常不能。

原因：

```text
vLLM 融合了 QKV / gate_up；
TP 会分片；
PP 只加载部分层；
量化权重有 scale / zero_point；
tied weights 要跳过重复项；
参数名可能和 HF 不一致。
```

### 36.5 支持 TP 是否自动支持 PP？

不是。

TP 主要靠并行 linear / embedding 层；PP 需要：

```text
make_layers；
IntermediateTensors；
PPMissingLayer；
make_empty_intermediate_tensors；
forward first/last rank 分支；
load_weights 跳过 missing parameter。
```

### 36.6 支持 LoRA 是否只要继承 SupportsLoRA？

不是。

还要保证：

```text
packed_modules_mapping 正确；
embedding_modules 正确；
目标模块命名和 adapter 权重命名一致；
MoE / multimodal tower / connector 场景额外处理。
```

### 36.7 多模态模型是不是在 forward 里处理图片？

vLLM 推荐把多模态 embedding 生成放到：

```text
embed_multimodal()
```

文本 embedding 和多模态 embedding 合并通常由：

```text
SupportsMultiModal.embed_input_ids()
```

默认实现处理。

### 36.8 tests/models/registry.py 为什么必须改？

因为 vLLM 的 registry 测试需要知道每个 architecture 对应哪个 HF repo，用来做 dummy load 和兼容性检查。

registry.py 顶部也明确要求新增 architecture 时同步更新。

---

## 37. 总结

新增模型架构可以压缩成一条主链路：

```text
确认 HF architecture
  → 选择相似 vLLM 模型模板
  → 实现 model / attention / mlp / layer
  → 接入 vLLM parallel layers 和 Attention
  → 实现 forward 协议
  → 实现 compute_logits 或 pooler
  → 实现 load_weights 映射
  → 补 SupportsPP / SupportsLoRA / SupportsMultiModal / Mamba 等协议
  → registry.py 注册
  → tests/models/registry.py 提供 HF example
  → correctness / TP / PP / quant / LoRA / multimodal 验证
  → supported_models.md 更新
```

如果只记住一句话：

```text
新增模型不是写一个 PyTorch forward，而是把一个 HF architecture 接入 vLLM 的注册、加载、并行、attention、输出和测试协议。
```

最小心智模型：

```text
registry 找类；
loader 建模并加载权重；
model forward 产 hidden states；
compute_logits / pooler 产任务输出；
ModelRunner 负责 batch、attention metadata、KV cache 和 sampling；
tests/models/registry.py 保证这个 architecture 能被 CI 找到和加载。
```
