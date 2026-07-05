# vLLM Model Architectures 逻辑梳理

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\model.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\tasks.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\registry.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\interfaces.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\interfaces_base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\adapters.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\module_mapping.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\default_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\llama.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`

本文按“先定边界，再走模型识别和构造，再拆 forward 接口、layer 组件、权重加载和扩展点”的方式，梳理 vLLM 中 model architecture 适配机制。

vLLM 支持很多模型，但它们不是各自完全独立的一套执行链路。模型架构层的职责，是把外部 HF config / checkpoint / architecture name 适配成 vLLM 执行层能统一调用的 `nn.Module`：

```text
ModelConfig / HF config
  → hf_config.architectures
  → ModelRegistry.inspect_model_cls() / resolve_model_cls()
  → vLLM model class
  → initialize_model(vllm_config, prefix)
  → model class 构造 shared layers
      → VocabParallelEmbedding / ParallelLMHead
      → Attention / MLAAttention / MMEncoderAttention
      → MLP / MoE / FusedMoE
      → RMSNorm / LayerNorm
      → RotaryEmbedding / M-RoPE / XD-RoPE
  → model.load_weights(...)
  → ModelRunner 调 model.forward(...)
  → hidden states / logits / pooler output
```

---

## 0. 梳理规划

本目录要回答的问题分成 13 组：

```text
1. model architectures 在 vLLM 中处于哪一层？
2. model registry 如何根据 HF architecture 选择 vLLM model class？
3. ModelConfig 如何判断 task、runner_type、模型能力？
4. 一个 vLLM model class 是如何构造 layers 的？
5. ModelRunner 对 model forward 接口有什么约定？
6. Attention / MLP / Norm / RoPE 等基础 blocks 如何复用？
7. Embedding、LM head、logits processor 如何接入？
8. MoE 模型架构如何组织 experts、router 和 fused MoE？
9. Pooling / embedding / rerank 模型如何区别于生成模型？
10. 多模态模型如何组织 vision encoder、projector、M-RoPE 和 inputs_embeds？
11. 权重加载如何处理模型命名、fused layer、TP 切分和 checkpoint 差异？
12. Quantization、LoRA、parallelism 如何 hook 到模型架构中？
13. 新增一个模型架构需要实现哪些接口和检查项？
```

阅读顺序建议：

```text
model_architectures_overview.md
  → 01_model_architecture_role.md
  → 02_model_registry_and_resolution.md
  → 03_model_config_and_task_detection.md
  → 04_model_class_construction.md
  → 05_forward_interface_contract.md
  → 06_attention_mlp_norm_blocks.md
  → 07_embedding_and_lm_head.md
  → 08_moe_model_architectures.md
  → 09_pooling_embedding_rerank_models.md
  → 10_multimodal_model_architectures.md
  → 11_weight_loading_and_name_mapping.md
  → 12_quant_lora_parallelism_hooks.md
  → 13_add_new_model_checklist.md
```

---

## 1. 一句话回答

vLLM 的 model architecture 机制负责把“外部模型配置和 checkpoint”适配成“vLLM 执行层能统一调用的 model class”。

主链路是：

```text
EngineArgs / model path
  → ModelConfig
      → 读取 HF config
      → 得到 hf_config.architectures
      → 判断 runner_type / convert_type / dtype / max_model_len / multimodal_config
  → ModelRegistry
      → architecture name → model info / model class
  → model_loader.initialize_model()
      → model_class(vllm_config=vllm_config, prefix=prefix)
  → model_loader.load_weights()
      → model.load_weights(weights_iterator)
  → GPUModelRunner._model_forward()
      → self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
  → generation model
      → model.compute_logits(hidden_states)
  → pooling model
      → pooler(hidden_states)
```

这层解决的是：

```text
不同模型结构、checkpoint 命名、输出任务、并行方式和特殊能力都不一样；
但 Worker / ModelRunner 希望用统一方式准备输入、调用 forward、拿到 hidden states / logits / pooling output。
```

---

## 2. 总体流程图

完整一点可以写成：

```text
用户传入 model / task / runner / convert / dtype / quant / lora / parallel 参数
  → ModelConfig.__post_init__
      → get_config(...)
      → get_hf_text_config(...)
      → get_model_arch_config()
      → architectures
      → registry.is_text_generation_model(...)
      → registry.is_pooling_model(...)
      → _get_runner_type(...)
      → _get_convert_type(...)
      → registry.inspect_model_cls(...)
      → _model_info / _architecture
      → dtype / max_model_len / pooler_config / multimodal_config / tokenizer_mode

  → model_loader.get_model_architecture(model_config)
      → registry.resolve_model_cls(architectures, model_config)
      → 如果 convert_type=embed/classify，则 as_embedding_model / as_seq_cls_model

  → initialize_model(vllm_config)
      → configure_quant_config(...)
      → model_class(vllm_config=vllm_config, prefix=prefix)
      → 构造 embedding / decoder layers / norm / lm_head / pooler / multimodal tower

  → default_loader.load_weights(...)
      → model.load_weights(iterator)
      → 模型类处理 fused weight、TP shard、PP missing layer、tie weights、name mapping

  → GPUModelRunner.execute_model(...)
      → _update_states()
      → _prepare_inputs()
      → _build_attention_metadata()
      → _preprocess()
      → _model_forward()

  → model.forward(...)
      → inner model / decoder layers
      → hidden_states 或 IntermediateTensors

  → 后处理
      → generation: compute_logits()
      → pooling: pooler()
      → multimodal / encoder-decoder / PP / LoRA / quant hooks 按能力接入
```

---

## 3. model architecture 层处于哪一层

可以把 vLLM 的模型执行拆成几层：

```text
配置层：
  ModelConfig / ParallelConfig / CacheConfig / LoRAConfig / QuantConfig
  决定模型类型、运行任务、dtype、长度、并行和功能开关。

注册层：
  ModelRegistry
  根据 HF architectures 找到 vLLM model class，并 inspect 这个 class 支持哪些能力。

模型架构层：
  vllm/model_executor/models/*.py
  定义每个模型如何构造 embedding、layers、attention、MLP、norm、lm_head、pooler、tower，以及 forward / load_weights。

基础 layer 层：
  vllm/model_executor/layers/*
  提供可复用的 Attention、Linear、Embedding、MoE、Norm、RoPE、LogitsProcessor 等组件。

执行层：
  Worker / GPUModelRunner
  不关心具体 Llama / Qwen / DeepSeek 细节，只按统一 forward 契约调用 model。
```

model architecture 层的边界是：

```text
它负责“模型结构适配”；
不负责调度、不负责 KV block 分配、不负责 sampling 策略；
但它必须提供足够的结构信息和接口，让执行层、loader、LoRA、quant、parallelism 可以 hook 进来。
```

---

## 4. `ModelConfig` 如何驱动架构选择

### 4.1 先拿 HF config 和 architectures

`ModelConfig` 初始化时会读取 HF config，然后得到：

```python
self.hf_config = hf_config
self.hf_text_config = get_hf_text_config(self.hf_config)
self.model_arch_config = self.get_model_arch_config()
architectures = self.architectures
registry = self.registry
```

位置：`model.py:540` 到 `model.py:558`

`architectures` 通常来自 `hf_config.architectures`，例如：

```text
LlamaForCausalLM
Qwen2ForCausalLM
MixtralForCausalLM
Qwen2VLForConditionalGeneration
BertModel
```

这个字段是 registry 解析模型类的入口。

### 4.2 判断 generation / pooling 能力

`ModelConfig` 会先问 registry：

```python
is_generative_model = registry.is_text_generation_model(architectures, self)
is_pooling_model = registry.is_pooling_model(architectures, self)
```

位置：`model.py:557` 到 `model.py:560`

这两个判断最终来自 `ModelRegistry.inspect_model_cls()` 返回的 `_ModelInfo`。

### 4.3 决定 runner_type 和 convert_type

随后：

```python
self.runner_type = self._get_runner_type(architectures, self.runner, self.convert)
self.convert_type = self._get_convert_type(architectures, self.runner_type, self.convert)
```

位置：`model.py:562` 到 `model.py:567`

这里的含义是：

```text
runner_type：
  本次用 generate / pooling / draft 等哪类 runner 执行。

convert_type：
  如果原始模型类不是目标任务形态，是否通过 adapter 转成 embed / classify。
```

例如：

```text
CausalLM + --runner generate
  → runner_type=generate, convert_type=none

CausalLM + --runner pooling --convert embed
  → runner_type=pooling, convert_type=embed

原生 embedding model
  → runner_type=pooling, convert_type=none
```

如果 runner 和模型能力不匹配，`ModelConfig` 会在这里直接报错。相关检查见 `model.py:569` 到 `model.py:591`。

### 4.4 inspect model class，固化 `_model_info`

`ModelConfig` 会尽早 inspect model class：

```python
model_info, arch = registry.inspect_model_cls(architectures, self)
self._model_info = model_info
self._architecture = arch
logger.info("Resolved architecture: %s", arch)
```

位置：`model.py:593` 到 `model.py:598`

这一步很重要，因为后续很多配置都依赖 `_model_info`：

```text
pooling 默认 pooling type；
multimodal_config 是否初始化；
mm_encoder_tp_mode 是否支持 data；
pipeline parallel 是否支持；
attention type / score type；
transcription / Mamba / hybrid / noops 等能力。
```

### 4.5 multimodal / pooling 等配置也在这里挂上

如果 `runner_type == "pooling"`，会初始化 `PoolerConfig`，并从 `_model_info` 拿默认 pooling type：

位置：`model.py:620` 到 `model.py:637`

如果 `_model_info.supports_multimodal` 为 True，则初始化 `MultiModalConfig`：

位置：`model.py:663` 到 `model.py:701`

所以 architecture 选择不是只为了构造 model class，它还会反过来影响配置对象本身。

---

## 5. `ModelRegistry` 如何解析 architecture

### 5.1 `_ModelInfo` 是模型类能力摘要

`registry.py` 中 `_ModelInfo` 记录模型类的关键能力：

```python
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    is_pooling_model: bool
    attn_type: AttnTypeStr
    default_seq_pooling_type: SequencePoolingType
    default_tok_pooling_type: TokenPoolingType
    score_type: ScoreType
    supports_multimodal: bool
    supports_multimodal_raw_input_only: bool
    requires_raw_input_tokens: bool
    supports_multimodal_encoder_tp_data: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    has_noops: bool
    supports_mamba_prefix_caching: bool
    supports_transcription: bool
    supports_transcription_only: bool
```

位置：`registry.py:747` 到 `registry.py:767`

`from_model_cls()` 会调用一组接口判断函数：

```python
is_text_generation_model(model)
is_pooling_model(model)
get_attn_type(model)
get_score_type(model)
supports_multimodal(model)
supports_pp(model)
supports_transcription(model)
...
```

位置：`registry.py:768` 到 `registry.py:796`

这说明 vLLM 不是只靠类名判断模型类型，而是把模型类的协议能力统一 inspect 成 `_ModelInfo`。

### 5.2 registered / lazy registered

registry 中有两种注册形态：

```text
_RegisteredModel：
  model class 已经 import，直接保存 model_cls 和 interfaces。

_LazyRegisteredModel：
  只保存 module_name / class_name；
  inspect 时再 import，并把 _ModelInfo 缓存到磁盘。
```

位置：`registry.py:799` 到 `registry.py:936`

lazy inspect 会在子进程中执行，避免主进程初始化 CUDA：

```python
mi = _run_in_subprocess(lambda: _ModelInfo.from_model_cls(self.load_model_cls()))
```

位置：`registry.py:933` 到 `registry.py:936`

### 5.3 resolve 有三条主要路径

`inspect_model_cls()` 和 `resolve_model_cls()` 的解析顺序基本一致。

第一类：强制 Transformers backend：

```text
model_impl == "transformers"
  → _try_resolve_transformers(architectures[0], model_config)
  → _try_inspect_model_cls / _try_load_model_cls
```

位置：`registry.py:1184` 到 `registry.py:1190`，以及 `registry.py:1236` 到 `registry.py:1242`

第二类：Terratorch：

```text
model_impl == "terratorch"
  → arch = "Terratorch"
```

位置：`registry.py:1191` 到 `registry.py:1193`，以及 `registry.py:1243` 到 `registry.py:1247`

第三类：默认 vLLM / auto：

```text
for arch in architectures:
  normalized_arch = _normalize_arch(arch, model_config)
  model_info / model_cls = _try_inspect_model_cls / _try_load_model_cls(normalized_arch)
```

位置：`registry.py:1207` 到 `registry.py:1211`，以及 `registry.py:1261` 到 `registry.py:1265`

如果 vLLM 没有实现，并且 `model_impl == "auto"`，会尝试 fallback 到 Transformers backend：

位置：`registry.py:1195` 到 `registry.py:1205`，以及 `registry.py:1267` 到 `registry.py:1276`。

### 5.4 `_normalize_arch()` 处理 architecture defaults

某些 architecture 可以通过默认规则转换成基础模型再 adapter：

```python
match = try_match_architecture_defaults(
    architecture,
    runner_type=getattr(model_config, "runner_type", None),
    convert_type=getattr(model_config, "convert_type", None),
)
```

位置：`registry.py:1148` 到 `registry.py:1172`

这和 `convert_type=embed/classify` 配合，用于把同一个基础模型类复用到 pooling / classify 等任务。

---

## 6. `get_model_architecture()` 和模型构造

### 6.1 `get_model_architecture()` 是 loader 侧入口

`model_loader/utils.py` 中：

```python
architectures = getattr(model_config.hf_config, "architectures", None) or []

model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

位置：`model_loader/utils.py:183` 到 `model_loader/utils.py:191`

如果 `convert_type` 不是 none，会包一层 adapter：

```python
if convert_type == "embed":
    model_cls = as_embedding_model(model_cls)
elif convert_type == "classify":
    model_cls = as_seq_cls_model(model_cls)
```

位置：`model_loader/utils.py:203` 到 `model_loader/utils.py:213`

结果会按 model / convert / runner / trust_remote_code / model_impl / architectures 做缓存：

位置：`model_loader/utils.py:218` 到 `model_loader/utils.py:234`。

### 6.2 `initialize_model()` 要求新式 model class 接收 `vllm_config` 和 `prefix`

初始化模型的主函数：

```python
def initialize_model(vllm_config, *, prefix="", model_class=None, model_config=None):
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    if "vllm_config" in all_params and "prefix" in all_params:
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            record_metadata_for_reloading(model)
            return model
```

位置：`model_loader/utils.py:40` 到 `model_loader/utils.py:64`

这是新增模型架构时最基础的构造契约：

```python
class XxxForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...
```

旧式 model class 仍有兼容路径，但源码会 warning，提示应迁移到 `vllm_config` / `prefix` 风格。位置：`model_loader/utils.py:66` 到 `model_loader/utils.py:97`。

### 6.3 loader 只调用统一 `load_weights()`

默认 loader 加载权重时：

```python
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`default_loader.py:381` 到 `default_loader.py:395`

所以 checkpoint 命名差异、fused QKV/gate_up、PP missing layer、tie lm_head、quant scale 名称等，主要由模型类自己的 `load_weights()` 或 `AutoWeightsLoader` 配置处理。

---

## 7. 一个典型 model class 长什么样：Llama

Llama 是最典型的 decoder-only generation 模型，可以把它当成阅读大多数 CausalLM 架构的模板。

### 7.1 inner model：embedding / decoder layers / norm

`LlamaModel` 构造：

```python
self.embed_tokens = VocabParallelEmbedding(...)
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
    prefix=f"{prefix}.layers",
)
self.norm = RMSNorm(...)
```

位置：`llama.py:347` 到 `llama.py:387`

它还考虑 pipeline parallel：

```text
first PP rank：
  持有 embed_tokens。

last PP rank：
  持有 norm。

其他 rank：
  对不属于本 rank 的模块放 PPMissingLayer。
```

位置：`llama.py:365` 到 `llama.py:383`

### 7.2 inner model forward

`LlamaModel.forward()` 的统一签名：

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
```

位置：`llama.py:392` 到 `llama.py:399`

first rank 从 `input_ids` 或 `inputs_embeds` 得到 hidden states：

```python
if inputs_embeds is not None:
    hidden_states = inputs_embeds
else:
    hidden_states = self.embed_input_ids(input_ids)
```

位置：`llama.py:400` 到 `llama.py:405`

中间 rank 从 `IntermediateTensors` 接收上一段 pipeline 输出：

位置：`llama.py:406` 到 `llama.py:410`。

如果不是最后 PP rank，返回 `IntermediateTensors`：

```python
return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
```

位置：`llama.py:422` 到 `llama.py:425`

最后 rank 才做 final norm 并返回 hidden states：

位置：`llama.py:427` 到 `llama.py:431`。

### 7.3 outer model：CausalLM 包装

`LlamaForCausalLM` 声明能力：

```python
class LlamaForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
```

位置：`llama.py:486` 到 `llama.py:488`

它声明 LoRA / packed mapping：

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

位置：`llama.py:489` 到 `llama.py:498`

构造时：

```text
self.model = LlamaModel(...)
self.lm_head = ParallelLMHead(...)
self.logits_processor = LogitsProcessor(...)
```

位置：`llama.py:500` 到 `llama.py:537`

### 7.4 outer model forward / logits / weights

`forward()` 只委托 inner model：

```python
model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
return model_output
```

位置：`llama.py:550` 到 `llama.py:560`

`compute_logits()` 委托 logits processor：

```python
logits = self.logits_processor(self.lm_head, hidden_states)
return logits
```

位置：`llama.py:562` 到 `llama.py:567`

`load_weights()` 使用 `AutoWeightsLoader`：

```python
loader = AutoWeightsLoader(
    self,
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
)
return loader.load_weights(weights)
```

位置：`llama.py:569` 到 `llama.py:574`

inner `LlamaModel.load_weights()` 则展示了手写 fused mapping 的模式：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

位置：`llama.py:433` 到 `llama.py:483`

---

## 8. ModelRunner 对模型 forward 的统一契约

从 `GPUModelRunner._model_forward()` 看，执行层最终只做统一调用：

```python
self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

这个约定要求 model class 至少支持：

```text
input_ids：
  文本 token ids。多模态或 prompt embeds 路径下可能为 None。

positions：
  本轮 token 的 position ids。可能是一维，也可能是 M-RoPE / XD-RoPE 的多维位置。

intermediate_tensors：
  pipeline parallel 非 first rank 的输入，或非 last rank 的输出。

inputs_embeds：
  多模态 / prompt embeddings 路径下直接传入 embedding。

**model_kwargs：
  encoder_outputs、cross attention 输入、multimodal kwargs、Mamba state、特殊模型参数等。
```

对 generation 模型，还需要：

```text
compute_logits(hidden_states) -> logits | None
```

对 pooling / embedding / rerank 模型，还需要：

```text
pooler 或 pooling adapter 输出 PoolerOutput / embedding / scores。
```

对权重加载，还需要：

```text
load_weights(weights: Iterable[tuple[str, Tensor]]) -> set[str] | None
```

---

## 9. 模型架构层要统一哪些差异

### 9.1 结构差异

```text
Decoder-only：
  Llama / Qwen / Mistral / Gemma / DeepSeek 等。

Encoder-decoder：
  Whisper / speech-to-text 等。

Encoder-only / bidirectional：
  embedding / rerank / classify / BERT-like 模型。

Multimodal：
  LLaVA / Qwen2-VL / Phi vision / audio-video 模型。

MoE：
  Mixtral / DeepSeek-MoE / other routed expert 模型。

Attention-free / hybrid：
  Mamba / hybrid attention 模型。
```

模型类通过不同 inner blocks 组合这些结构，但对执行层暴露统一接口。

### 9.2 attention 差异

模型架构决定：

```text
MHA / MQA / GQA；
MLA；
sliding window；
cross attention；
attention-free 或 hybrid 层；
encoder attention / multimodal encoder attention；
head_dim、num_heads、num_kv_heads、rope 参数。
```

`_ModelInfo.attn_type`、`ModelConfig` 中的 sliding window / max len / backend 选择，以及各模型 layer 构造的 `Attention` 参数共同决定最终 attention 行为。

### 9.3 MLP / MoE 差异

模型层里常见差异：

```text
普通 MLP：
  gate_up_proj + down_proj / fused SwiGLU。

MoE：
  router + routed experts + shared experts；
  fused MoE kernel；
  expert parallel；
  2D / 3D expert weight layout。
```

这些差异通常被封装在模型自己的 decoder layer 和 `model_executor/layers/fused_moe` 等基础组件里。

### 9.4 位置编码差异

```text
RoPE：普通 decoder-only 模型主流路径。
M-RoPE：视觉语言模型按 text/image/video grid 计算多维位置。
XD-RoPE：部分模型使用额外维度位置。
ALiBi / absolute position / sliding window：由具体模型适配。
```

模型类或接口会通过 `supports_mrope`、`supports_xdrope`、`get_mrope_input_positions()` 等暴露特殊位置需求。

### 9.5 输出差异

```text
generation：
  hidden_states → compute_logits() → sampling。

pooling / embedding：
  hidden_states → pooler → embedding output。

classify / score / rerank：
  hidden_states 或 pair input → score output。

transcription：
  audio encoder / decoder 或 special output path。
```

registry 的 `_ModelInfo.is_text_generation_model`、`is_pooling_model`、`score_type` 会把这些能力向配置层暴露。

### 9.6 加载差异

```text
HF checkpoint 可能是 unfused q/k/v；
vLLM runtime 可能是 fused qkv_proj。

HF checkpoint 可能有 gate_proj / up_proj；
vLLM runtime 可能是 gate_up_proj。

TP 下参数需要按 shard_id 加载；
PP 下当前 rank 没有的参数要跳过；
quantized checkpoint 可能有 scale / zero_point / packed weight；
tie_word_embeddings 时 lm_head 可能不单独加载。
```

这些差异通常在 `load_weights()`、`AutoWeightsLoader`、`WeightsMapper`、`packed_modules_mapping` 中处理。

---

## 10. model interfaces 是能力开关

`interfaces.py` 中定义了很多 Protocol / mixin 风格接口。它们的共同作用是：

```text
模型类通过继承或属性声明能力；
registry inspect 时把能力写入 _ModelInfo；
配置层和执行层按能力启用对应路径。
```

常见接口包括：

```text
SupportsLoRA：
  允许 LoRA manager 注入 layer。

SupportsPP：
  允许 pipeline parallel。

SupportsMultiModal：
  允许 multimodal processor / encoder / embedding 合并路径。

SupportsQuant：
  允许模型类自定义 quant method 处理。

SupportsTranscription：
  语音转文本模型能力。

SupportsEagle / SupportsEagle3：
  speculative decoding 相关能力。

SupportsMambaPrefixCaching：
  Mamba prefix caching。

SupportsMRoPE / SupportsXDRoPE / SupportsEncoderCudaGraph：
  多模态位置和 encoder cudagraph 相关能力。
```

对应定义分布在 `interfaces.py`，例如：

```text
SupportsMultiModal：interfaces.py:94 开始
SupportsLoRA：interfaces.py:538 开始
SupportsPP：interfaces.py:616 开始
SupportsQuant：interfaces.py:997 开始
SupportsTranscription：interfaces.py:1075 开始
SupportsEagle：interfaces.py:1256 开始
SupportsMRoPE：interfaces.py:1445 开始
SupportsEncoderCudaGraph：interfaces.py:1544 开始
```

这些接口不是文档标签，而是运行时判断的依据。例如 `_ModelInfo.from_model_cls()` 会调用 `supports_multimodal(model)`、`supports_pp(model)` 等函数。

---

## 11. 和其他专题的关系

```text
config_and_model_loading：
  解释 ModelConfig 和 model_loader 如何触发模型架构选择。

executor_worker_model_runner：
  解释 ModelRunner 如何统一调用不同模型的 forward。

attention：
  模型架构决定 Attention layer 的 head 数、MLA、sliding window、RoPE 等参数。

parallelism：
  模型 layer 创建时要选择 TP / PP / EP 兼容的 parallel layers。

quantization：
  模型 layer 创建时会接入 quant_config / quant_method；loader 后处理 quant weights。

lora_and_adapters：
  模型架构通过 SupportsLoRA、embedding_modules、packed_modules_mapping 决定 LoRA target modules。

multimodal：
  多模态模型架构通过 SupportsMultiModal、processor registry、vision/audio tower、projector、M-RoPE 接入。

sampling_and_output：
  generation model 输出 logits，pooling model 输出 pooler_output。
```

---

## 12. 新增一个 model architecture 的最小检查项

### 12.1 注册和构造

```text
1. 在 registry 中能通过 HF architectures 解析到模型类。
2. 模型类构造函数支持：__init__(*, vllm_config: VllmConfig, prefix: str = "")。
3. 使用 vLLM shared layers，而不是直接复用 HF module forward。
4. 正确传递 prefix，保证参数名、量化、LoRA、loader 都能定位模块。
```

### 12.2 forward 契约

```text
1. forward 接受 input_ids、positions、intermediate_tensors、inputs_embeds。
2. first PP rank 支持从 input_ids 或 inputs_embeds 生成 hidden_states。
3. 非 first PP rank 支持从 IntermediateTensors 恢复 hidden_states。
4. 非 last PP rank 返回 IntermediateTensors。
5. last PP rank 返回 hidden_states 或任务需要的输出。
```

### 12.3 输出契约

```text
Generation：
  - 实现 compute_logits(hidden_states)。
  - 构造 lm_head 和 logits_processor。

Pooling / embedding / classify：
  - 声明 pooling 能力或使用 adapter 转换。
  - 提供 pooler / score 输出路径。

Multimodal：
  - 继承 SupportsMultiModal。
  - 注册 MULTIMODAL_REGISTRY processor。
  - 实现 embed_multimodal / placeholder / embedding 合并相关接口。
```

### 12.4 权重加载

```text
1. 实现 load_weights(weights)。
2. 处理 q/k/v、gate/up 等 checkpoint 到 runtime fused 参数的映射。
3. TP 参数调用对应 param.weight_loader，并传 shard_id。
4. PP missing parameter 跳过。
5. tie_word_embeddings 时跳过重复 lm_head。
6. 如有 HF/vLLM 前缀差异，提供 WeightsMapper / hf_to_vllm_mapper。
```

### 12.5 扩展能力

```text
LoRA：
  - 继承 SupportsLoRA。
  - 提供 packed_modules_mapping / embedding_modules。

PP：
  - 继承 SupportsPP。
  - 使用 make_layers / PPMissingLayer / IntermediateTensors。

Quant：
  - 使用支持 quant_config 的 vLLM layers。
  - 必要时实现 SupportsQuant / get_quant_config。

MoE：
  - 使用 MoERunner / fused MoE。
  - 提供 expert mapping / packed mapping。

特殊位置编码：
  - 正确接入 RoPE / M-RoPE / XD-RoPE。
```

---

## 13. 后续专题说明

```text
01_model_architecture_role.md：
  定义模型架构层的职责、边界，以及它和执行层 / 配置层的关系。

02_model_registry_and_resolution.md：
  梳理 registry 如何根据 HF architectures 字段解析到 vLLM model class。

03_model_config_and_task_detection.md：
  梳理 ModelConfig 如何识别 task、runner_type、支持能力、dtype 和模型长度。

04_model_class_construction.md：
  梳理一个 vLLM model class 如何构造 embedding、layers、norm、lm_head / pooler。

05_forward_interface_contract.md：
  梳理 ModelRunner 对 model.forward、compute_logits、pooler、load_weights 的接口约定。

06_attention_mlp_norm_blocks.md：
  梳理 Attention、MLP、Norm、RoPE 等基础 block 如何在不同模型中复用。

07_embedding_and_lm_head.md：
  梳理 vocab parallel embedding、lm_head、tie weights、logits processor 的关系。

08_moe_model_architectures.md：
  梳理 MoE 模型中的 router、experts、fused MoE、shared experts 和 expert parallel。

09_pooling_embedding_rerank_models.md：
  梳理 embedding / classify / score / rerank 模型如何接入 model architecture。

10_multimodal_model_architectures.md：
  梳理 vision-language 模型中的 vision tower、projector、placeholder、M-RoPE 和 inputs_embeds。

11_weight_loading_and_name_mapping.md：
  梳理模型权重加载、fused layer 映射、TP 切分、checkpoint 命名差异。

12_quant_lora_parallelism_hooks.md：
  梳理模型架构如何为 quantization、LoRA、TP/PP/EP 提供 hook。

13_add_new_model_checklist.md：
  梳理新增模型架构时需要实现和验证的接口清单。
```

---

## 14. 一句话总结

model architectures 是 vLLM 把“各种外部模型”变成“统一执行接口”的适配层：

```text
它从 HF architecture name 出发，
通过 registry 解析模型类和能力，
通过 model class 复用 vLLM shared layers 构造实际网络，
通过 load_weights 适配 checkpoint 命名和并行切分，
通过统一 forward / compute_logits / pooler 契约交给 ModelRunner 执行。
```

最终目标是：

```text
让 Llama、Qwen、DeepSeek、Mixtral、BERT、Whisper、LLaVA、Qwen2-VL 等结构不同的模型，
都能落到同一套 vLLM 执行、调度、KV cache、quant、LoRA、parallelism 和输出处理框架里。
```
