# 01. Model architecture 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/interfaces_base.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：`model architecture` 在 vLLM 中到底是哪一层；它如何把 HuggingFace config 中的 `architectures` 解析成 vLLM 可执行的 `nn.Module`；它需要实现哪些统一接口；它和 `ModelConfig`、`ModelRegistry`、`ModelLoader`、`ModelRunner`、attention / linear / logits / pooler layer 的职责边界是什么。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本文按“先定角色，再走主链路，再拆接口和边界”的方式梳理 `model architecture`。

要回答的问题分成 10 组：

```text
1. Model architecture 是哪一层？
2. 它负责什么，不负责什么？
3. HF config.architectures 如何映射到 vLLM model class？
4. ModelRegistry 在其中做什么？
5. ModelLoader 如何实例化架构类并加载权重？
6. 一个架构类需要实现哪些统一接口？
7. 以 LlamaForCausalLM 为例，架构类内部如何组织 embedding / decoder / norm / lm_head？
8. forward、compute_logits、load_weights 分别承担什么？
9. TP / PP / quant / LoRA / multimodal / pooling 如何挂接到架构层？
10. 它和 ModelRunner、Scheduler、Executor、layer 层的边界是什么？
```

阅读顺序建议：

```text
01_model_architecture_role.md
  → registry.py
  → model_loader/utils.py
  → model_loader/default_loader.py
  → models/interfaces_base.py
  → models/interfaces.py
  → models/llama.py
  → v1/worker/gpu_model_runner.py 的 forward / logits 部分
```

---

## 1. 一句话回答

`Model architecture` 是 **具体模型结构到 vLLM 执行框架之间的适配层**。

它负责把一个外部模型结构，例如：

```text
LlamaForCausalLM
Qwen2ForCausalLM
DeepseekV2ForCausalLM
BertModel
CLIPEmbeddingModel
各种多模态 ForConditionalGeneration
```

实现成 vLLM 可以统一加载、统一 forward、统一计算 logits / pooling、统一参与并行和量化的 `torch.nn.Module`。

一句话压缩：

```text
Model architecture 负责“模型长什么样、权重怎么装、forward 怎么跑、hidden states 怎么变成 logits 或 pooling output”。
```

---

## 2. 它负责什么

一个 vLLM model architecture 通常负责：

```text
- 根据 VllmConfig / HF config 构造模型模块；
- 创建 embedding、decoder layers、final norm、lm_head、pooler；
- 使用 vLLM 的 Attention、Linear、Embedding、LogitsProcessor、MoE、RoPE 等基础 layer；
- 处理 tensor parallel 下的参数切分和输出规约；
- 处理 pipeline parallel 下的层切分、PPMissingLayer、IntermediateTensors；
- 实现 forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)；
- 对 generation 模型实现 compute_logits(hidden_states)；
- 对 pooling 模型暴露 pooler；
- 实现 load_weights(weights)，把 HF checkpoint 权重映射到 vLLM 参数；
- 声明 LoRA、PP、multimodal、Mamba、EAGLE、quant、transcription 等能力标签；
- 通过 packed_modules_mapping / embedding_modules 等字段告诉 LoRA、quant、weight loader 如何处理特殊模块。
```

从执行视角看，它是 `ModelRunner._model_forward()` 最终调用的对象：

```text
GPUModelRunner._model_forward()
  → self.model(...)
  → 具体 architecture.forward(...)
```

---

## 3. 它不负责什么

`model architecture` 不负责执行系统的上层调度和请求生命周期。

它不负责：

```text
- 不负责接收 HTTP / OpenAI API 请求；
- 不负责 tokenizer / detokenizer 主流程；
- 不负责 Scheduler 的 waiting / running 队列；
- 不负责决定本轮调度哪些请求；
- 不负责 token budget；
- 不负责逻辑 KV block 分配、抢占、释放；
- 不负责 Executor 的多进程 / Ray / RPC 分发；
- 不负责 Worker 的 device 初始化和生命周期；
- 不负责构造 InputBatch；
- 不负责构造 attention metadata；
- 不负责 grammar bitmask、sampler 策略和最终 RequestOutput 组装。
```

这些分别属于：

```text
Scheduler：调度和逻辑 KV block 管理；
Executor：分发到 worker；
Worker：设备生命周期；
ModelRunner：batch 状态、输入张量、attention metadata、forward 编排、sampling；
OutputProcessor：最终用户输出。
```

---

## 4. 一句话主链路

从配置到实际 forward，可以把 model architecture 放在这条链路里：

```text
HF config.architectures
  → ModelConfig.model_arch_config.architectures
  → ModelRegistry.inspect_model_cls() / resolve_model_cls()
  → get_model_architecture(model_config)
  → initialize_model(vllm_config)
  → model_class(vllm_config, prefix)
  → DefaultModelLoader.load_weights(model)
  → model.load_weights(weights)
  → GPUModelRunner.model = model
  → GPUModelRunner._model_forward()
  → model.forward(...)
  → model.compute_logits(...) 或 model.pooler(...)
```

它说明：

```text
Model architecture 不是单独运行的；
它先被配置层选中，再被 loader 实例化和加载权重，最后被 ModelRunner 放进一轮 batch 执行链路里调用。
```

---

## 5. ModelConfig 如何看待 architecture

`ModelConfig` 会从 HF config 中抽取模型架构相关信息。

关键属性：

```python
@property
def architectures(self) -> list[str]:
    return self.model_arch_config.architectures
```

位置：`code/vllm/vllm/config/model.py:811` 到 `code/vllm/vllm/config/model.py:812`

而 `model_arch_config` 来自：

```python
def get_model_arch_config(self) -> ModelArchitectureConfig:
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
        self.hf_config.model_type, ModelArchConfigConvertorBase
    )
    convertor = convertor_cls(self.hf_config, self.hf_text_config)
    return convertor.convert()
```

位置：`code/vllm/vllm/config/model.py:728` 到 `code/vllm/vllm/config/model.py:735`

`ModelArchitectureConfig` 保存的是 vLLM runtime 需要的架构信息：

```text
architectures
model_type
text_model_type
hidden_size
num_hidden_layers
num_attention_heads
head_size
vocab_size
num_kv_heads
num_experts
quantization_config
is_deepseek_mla
is_mm_prefix_lm
max_model_len 信息
```

位置：`code/vllm/vllm/config/model_arch.py:13` 到 `code/vllm/vllm/config/model_arch.py:60`

这说明：

```text
ModelConfig 不实现模型结构；
它负责解析、验证、缓存“应该用哪个架构类”和“这个架构有什么运行特征”。
```

---

## 6. ModelRegistry 的角色

`ModelRegistry` 是 architecture 名字到 vLLM model class 的注册表。

入口：`code/vllm/vllm/model_executor/models/registry.py:1377`

它由 `_VLLM_MODELS` 构造：

```text
architecture name
  → module name
  → class name
```

最终形成：

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

位置：`code/vllm/vllm/model_executor/models/registry.py:1377` 到 `code/vllm/vllm/model_executor/models/registry.py:1385`

### 6.1 为什么是 lazy registered

`_LazyRegisteredModel` 保存的是：

```text
module_name
class_name
```

它不会在 registry 初始化时立刻 import 所有模型文件。

原因是：

```text
避免 import 模型模块时过早初始化 CUDA；
避免主进程一次性加载所有模型实现；
允许 inspect model class 时放到子进程里完成。
```

相关逻辑：

```python
# Performed in another process to avoid initializing CUDA
mi = _run_in_subprocess(
    lambda: _ModelInfo.from_model_cls(self.load_model_cls())
)
```

位置：`code/vllm/vllm/model_executor/models/registry.py:932` 到 `code/vllm/vllm/model_executor/models/registry.py:935`

### 6.2 inspect_model_cls() 做什么

`inspect_model_cls()` 不实例化模型，而是检查模型类的能力。

返回 `_ModelInfo`，包括：

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
supports_mamba_prefix_caching
默认 pooling 类型
score_type
```

位置：`code/vllm/vllm/model_executor/models/registry.py:745` 到 `code/vllm/vllm/model_executor/models/registry.py:795`

`ModelConfig` 会用它来决定：

```text
runner_type 是否合法；
pooling config 默认值；
是否启用 multimodal config；
是否支持 PP；
是否是 attention-free / hybrid / transcription model。
```

### 6.3 resolve_model_cls() 做什么

`resolve_model_cls()` 真正返回要实例化的 `model_cls`。

核心逻辑：

```text
1. 如果指定 model_impl="transformers"，优先解析 Transformers backend；
2. 如果 model_impl="terratorch"，解析 Terratorch；
3. 如果 auto 且 vLLM 没有实现，可 fallback 到 Transformers backend；
4. 遍历 config.architectures，查找 vLLM registry 中的实现；
5. 支持 normalize architecture，用默认映射处理 embedding/classify 等变体；
6. 全部失败则报 unsupported architecture。
```

入口：`code/vllm/vllm/model_executor/models/registry.py:1225`

---

## 7. get_model_architecture() 的角色

`get_model_architecture(model_config)` 是 loader 侧选择模型类的统一入口。

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:218`

主逻辑：

```python
architectures = getattr(model_config.hf_config, "architectures", None) or []

model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:183` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:191`

然后根据 `convert_type` 可能把生成模型适配成 pooling 模型：

```python
if convert_type == "embed":
    model_cls = as_embedding_model(model_cls)
elif convert_type == "classify":
    model_cls = as_seq_cls_model(model_cls)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:203` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:212`

所以这里有两个层次：

```text
arch：HF / registry 解析出来的原始架构名字；
model_cls：最终要实例化的 vLLM class，可能已经被 adapter 包装过。
```

---

## 8. initialize_model() 如何实例化 architecture

模型实例化入口是：

```python
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:40`

如果没有传 `model_class`，会调用：

```python
model_class, _ = get_model_architecture(model_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:51` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:52`

新式 vLLM 模型类应该支持：

```python
model_class(vllm_config=vllm_config, prefix=prefix)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:57` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:64`

也就是说，一个新架构类的标准构造签名是：

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

如果是老式 out-of-tree model class，vLLM 会尝试兼容 `config`、`cache_config`、`quant_config`、`lora_config` 等参数，但已经是降级路径。

---

## 9. ModelLoader 和 architecture 的关系

以默认 loader 为例，`DefaultModelLoader` 负责找到权重文件，并把权重 iterator 交给模型自己的 `load_weights()`。

核心调用：

```python
loaded_weights = model.load_weights(
    self.get_all_weights(model_config, model)
)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:427`

这说明权重加载分两层：

```text
DefaultModelLoader：
  负责找文件、下载、选择 safetensors/bin/pt/npcache、生成 (name, tensor) iterator。

model architecture.load_weights()：
  负责把 checkpoint 里的参数名映射到当前架构的参数，处理 fused qkv、gate_up、PP missing layer、tie weights 等模型结构差异。
```

加载后，loader 还会做：

```text
- track weights loading；
- quantization postprocess；
- attention weight postprocess；
- torchao reload metadata。
```

相关位置：`code/vllm/vllm/model_executor/model_loader/utils.py:100` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:132`

---

## 10. vLLM model class 的基础接口

基础接口定义在：`code/vllm/vllm/model_executor/models/interfaces_base.py`

### 10.1 所有 vLLM model 都应满足 VllmModel

`VllmModel` 要求：

```python
def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None

def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor

def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:46` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:57`

实际模型常常会扩展 forward 参数，例如：

```python
forward(
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    **model_kwargs,
)
```

这是为了支持 PP、多模态、encoder-decoder、prompt embeds 等场景。

### 10.2 生成模型需要 compute_logits

`VllmModelForTextGeneration` 要求：

```python
def compute_logits(self, hidden_states) -> Tensor | None
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:113` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:122`

它的语义是：

```text
forward 返回 hidden states；
compute_logits 负责把需要采样位置的 hidden states 投影成 vocab logits。
```

### 10.3 pooling 模型需要 pooler

`VllmModelForPooling` 要求模型：

```text
is_pooling_model = True
pooler: Pooler
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:147` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:212`

pooling 模型 forward 后不会走 `compute_logits()` 和 sampler，而是由 ModelRunner 调用：

```text
model.pooler(hidden_states, pooling_metadata)
```

---

## 11. 扩展能力接口

扩展接口定义在：`code/vllm/vllm/model_executor/models/interfaces.py`

常见能力包括：

```text
SupportsLoRA：声明支持 LoRA，提供 embedding_modules / packed_modules_mapping；
SupportsPP：声明支持 pipeline parallel，提供 make_empty_intermediate_tensors 和 PP forward 约定；
SupportsMultiModal：声明支持多模态，提供 embed_multimodal / embed_input_ids / get_language_model；
SupportsQuant：声明支持 quantization，提供 packed mapping / hf_to_vllm_mapper；
HasInnerState：例如 Mamba / Jamba 这类有内部 state 的模型；
IsAttentionFree：无 attention 但有固定状态的模型；
IsHybrid：同时有 attention 和 mamba block 的模型；
MixtureOfExperts：MoE 模型暴露 expert metadata / EPLB 接口；
SupportsMRoPE / SupportsXDRoPE：多维位置编码；
SupportsTranscription / SupportsRealtime：语音模型能力；
SupportsEagle / SupportsEagle3：speculative decoding 辅助 hidden state 能力。
```

这些接口本质上是：

```text
用 class attribute / Protocol 告诉 ModelConfig、ModelRegistry、ModelRunner、LoRA、quant、multimodal registry：这个模型有哪些特殊能力。
```

---

## 12. 以 LlamaForCausalLM 为例看 architecture 内部结构

`LlamaForCausalLM` 是一个典型 generation architecture。

位置：`code/vllm/vllm/model_executor/models/llama.py:486`

类定义：

```python
class LlamaForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
```

位置：`code/vllm/vllm/model_executor/models/llama.py:486` 到 `code/vllm/vllm/model_executor/models/llama.py:488`

这说明它声明支持：

```text
LoRA
Pipeline Parallel
EAGLE / EAGLE3 speculative decoding
LocalArgmaxMixin draft token 辅助逻辑
```

### 12.1 外层 LlamaForCausalLM

外层负责：

```text
- 持有 self.model = LlamaModel；
- 在最后 PP rank 创建 lm_head；
- 创建 LogitsProcessor；
- 暴露 embed_input_ids；
- 实现 forward；
- 实现 compute_logits；
- 实现 load_weights。
```

关键字段：

```python
self.model = self._init_model(...)

if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)
    self.logits_processor = LogitsProcessor(...)
else:
    self.lm_head = PPMissingLayer()
```

位置：`code/vllm/vllm/model_executor/models/llama.py:512` 到 `code/vllm/vllm/model_executor/models/llama.py:533`

### 12.2 内层 LlamaModel

`LlamaModel` 负责主干网络：

```text
embed_tokens
layers
norm
make_empty_intermediate_tensors
```

位置：`code/vllm/vllm/model_executor/models/llama.py:347`

构造时会：

```python
self.embed_tokens = VocabParallelEmbedding(...)
self.start_layer, self.end_layer, self.layers = make_layers(...)
self.norm = RMSNorm(...)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:365` 到 `code/vllm/vllm/model_executor/models/llama.py:383`

### 12.3 LlamaDecoderLayer

每层 decoder 包含：

```text
self_attn
mlp
input_layernorm
post_attention_layernorm
```

位置：`code/vllm/vllm/model_executor/models/llama.py:251` 到 `code/vllm/vllm/model_executor/models/llama.py:334`

它使用的是 vLLM layer：

```text
QKVParallelLinear
RowParallelLinear
MergedColumnParallelLinear
Attention
RMSNorm
SiluAndMul
RoPE
```

这说明 architecture 层不是自己手写所有 kernel，而是把 vLLM 的 layer 组合成具体模型结构。

---

## 13. architecture.forward() 做什么

以 Llama 为例，外层 `LlamaForCausalLM.forward()` 很薄：

```python
model_output = self.model(
    input_ids, positions, intermediate_tensors, inputs_embeds
)
return model_output
```

位置：`code/vllm/vllm/model_executor/models/llama.py:550` 到 `code/vllm/vllm/model_executor/models/llama.py:560`

真正主干 forward 在 `LlamaModel.forward()`。

它做的事情是：

```text
1. 如果是 first PP rank：
   input_ids / inputs_embeds → hidden_states；

2. 如果不是 first PP rank：
   从 intermediate_tensors 取 hidden_states / residual；

3. 遍历当前 PP rank 负责的 decoder layers；

4. 如果不是 last PP rank：
   返回 IntermediateTensors；

5. 如果是 last PP rank：
   过 final norm，返回 hidden_states。
```

对应代码位置：`code/vllm/vllm/model_executor/models/llama.py:392` 到 `code/vllm/vllm/model_executor/models/llama.py:431`

最关键的 PP 分支是：

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

位置：`code/vllm/vllm/model_executor/models/llama.py:422` 到 `code/vllm/vllm/model_executor/models/llama.py:425`

因此：

```text
architecture.forward() 的输出不一定是 logits；
通常是 hidden_states，PP 非 last rank 时是 IntermediateTensors。
```

---

## 14. compute_logits() 做什么

Llama 的 `compute_logits()`：

```python
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

位置：`code/vllm/vllm/model_executor/models/llama.py:562` 到 `code/vllm/vllm/model_executor/models/llama.py:567`

它说明：

```text
lm_head 不在 forward 里直接调用；
forward 先产出 hidden states；
ModelRunner 根据 logits_indices 截取 sample_hidden_states；
再调用 model.compute_logits(sample_hidden_states)。
```

ModelRunner 侧调用：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

这个拆分很重要：

```text
不是每个 forward 输出位置都需要 logits；
只对采样、prompt logprobs、spec decode 需要的位置计算 logits。
```

---

## 15. load_weights() 做什么

权重文件里的参数名通常来自 HF 模型；vLLM 模型内部为了性能会使用 fused / parallel layer。

因此 `load_weights()` 需要做名称和 shard 映射。

`LlamaModel.load_weights()` 里定义了：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

位置：`code/vllm/vllm/model_executor/models/llama.py:433` 到 `code/vllm/vllm/model_executor/models/llama.py:441`

含义是：

```text
HF checkpoint 里的 q_proj / k_proj / v_proj
  → vLLM 里的 qkv_proj 不同 shard；

HF checkpoint 里的 gate_proj / up_proj
  → vLLM 里的 gate_up_proj 不同 shard。
```

外层 `LlamaForCausalLM.load_weights()` 使用：

```python
loader = AutoWeightsLoader(
    self,
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
)
return loader.load_weights(weights)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:569` 到 `code/vllm/vllm/model_executor/models/llama.py:574`

这说明 architecture 层需要理解：

```text
自己的参数命名；
HF 权重命名；
哪些权重要跳过；
哪些参数是 fused 后的 shard；
PP rank 缺失参数如何跳过；
tie word embeddings 如何处理。
```

---

## 16. TP 在 architecture 层怎么体现

Tensor Parallel 不是由 Scheduler 决定的，而是在模型 layer 构造时体现。

以 Llama attention 为例：

```python
tp_size = get_tensor_model_parallel_world_size()
self.total_num_heads = num_heads
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:143` 到 `code/vllm/vllm/model_executor/models/llama.py:156`

然后使用 TP-aware layer：

```text
QKVParallelLinear
RowParallelLinear
MergedColumnParallelLinear
VocabParallelEmbedding
ParallelLMHead
```

这说明：

```text
architecture 层负责根据并行配置构造“已经切分好的模型”；
执行时每个 rank 只持有自己负责的参数 shard。
```

---

## 17. PP 在 architecture 层怎么体现

Pipeline Parallel 需要架构类配合层切分。

典型做法：

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
    prefix=f"{prefix}.layers",
)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:375` 到 `code/vllm/vllm/model_executor/models/llama.py:379`

同时非本 rank 负责的模块用 `PPMissingLayer()` 占位：

```text
非 first rank：embed_tokens = PPMissingLayer()
非 last rank：norm / lm_head = PPMissingLayer()
```

相关位置：

- `code/vllm/vllm/model_executor/models/llama.py:365` 到 `code/vllm/vllm/model_executor/models/llama.py:383`
- `code/vllm/vllm/model_executor/models/llama.py:518` 到 `code/vllm/vllm/model_executor/models/llama.py:533`

PP 支持还要求模型暴露：

```text
supports_pp = True
make_empty_intermediate_tensors(...)
forward(..., intermediate_tensors=...)
```

接口位置：`code/vllm/vllm/model_executor/models/interfaces.py:617` 到 `code/vllm/vllm/model_executor/models/interfaces.py:735`

因此 PP 的边界是：

```text
architecture：定义每个 PP rank 持有哪些层，forward 如何接收/返回 IntermediateTensors；
Worker / distributed：负责 rank 间发送和接收 IntermediateTensors；
ModelRunner：负责把 intermediate_tensors 传给 model.forward，并处理非 last rank 返回值。
```

---

## 18. quant / LoRA 在 architecture 层怎么体现

### 18.1 Quantization

架构类通常会把 `vllm_config.quant_config` 传给底层 layer：

```python
quant_config = vllm_config.quant_config
QKVParallelLinear(..., quant_config=quant_config)
RowParallelLinear(..., quant_config=quant_config)
ParallelLMHead(..., quant_config=quant_config)
```

这样量化逻辑由 layer / quant method 执行，但 architecture 负责把配置接进去。

`SupportsQuant` 还允许模型声明：

```text
hf_to_vllm_mapper
packed_modules_mapping
quant_config
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:999` 到 `code/vllm/vllm/model_executor/models/interfaces.py:1038`

### 18.2 LoRA

LoRA 需要架构类告诉 vLLM 哪些模块可注入：

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

位置：`code/vllm/vllm/model_executor/models/llama.py:489` 到 `code/vllm/vllm/model_executor/models/llama.py:498`

接口位置：`code/vllm/vllm/model_executor/models/interfaces.py:538` 到 `code/vllm/vllm/model_executor/models/interfaces.py:614`

所以：

```text
LoRA 的 runtime 管理在 ModelRunner / LoRA manager；
但“这个架构哪些模块能挂 LoRA、HF 名字如何对应 vLLM fused 名字”由 architecture 声明。
```

---

## 19. multimodal 在 architecture 层怎么体现

多模态模型一般继承 / 实现 `SupportsMultiModal`。

接口要求模型提供：

```text
embed_multimodal(**kwargs)：把图片/音频/视频等输入编码成 embeddings；
embed_input_ids(...)：把 text token ids 和 multimodal embeddings 合并；
get_language_model()：返回底层语言模型；
get_placeholder_str(...)：定义 prompt 中的占位文本；
可能还包括 M-RoPE / pruning / encoder cudagraph 等能力。
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:95` 到 `code/vllm/vllm/model_executor/models/interfaces.py:410`

ModelConfig 会根据 registry inspect 结果初始化 multimodal config：

```python
if self._model_info.supports_multimodal:
    self.multimodal_config = MultiModalConfig(...)
```

位置：`code/vllm/vllm/config/model.py:663` 到 `code/vllm/vllm/config/model.py:701`

ModelRunner 在 `_preprocess()` / encoder 阶段调用多模态逻辑，architecture 负责具体怎么把多模态输入变成语言模型可消费的 embedding。

边界是：

```text
ModelRunner：决定本轮哪些多模态输入要执行、何时执行 encoder、如何放进 model_kwargs；
architecture：定义这个模型的视觉/音频 tower、connector、placeholder、embedding 合并方式。
```

---

## 20. pooling / embedding architecture 怎么体现

Pooling 模型可以是原生 pooling 架构，也可以由 generation 模型 adapter 转换而来。

转换入口：

```text
as_embedding_model(model_cls)
as_seq_cls_model(model_cls)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:203` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:212`

转换后的类会：

```text
is_pooling_model = True
初始化 self.pooler
保留或复用原模型 forward
把 hidden states 交给 pooler 处理
```

原生 pooling 模型也会直接设置：

```text
is_pooling_model = True
self.pooler = ...
```

ModelConfig 会根据 `_ModelInfo.default_seq_pooling_type`、`default_tok_pooling_type` 初始化 `PoolerConfig`。

位置：`code/vllm/vllm/config/model.py:620` 到 `code/vllm/vllm/config/model.py:637`

ModelRunner 侧如果发现是 pooling model，会在 forward 后直接：

```text
hidden_states → model.pooler(hidden_states, pooling_metadata) → ModelRunnerOutput(pooler_output)
```

而不是走 logits / sampler。

---

## 21. architecture 和 ModelRunner 的边界

这是最容易混淆的地方。

### 21.1 ModelRunner 负责

```text
- 维护 CachedRequestState / InputBatch；
- 消费 SchedulerOutput；
- 准备 input_ids / positions / inputs_embeds；
- 构造 slot mapping / block table；
- 构造 attention metadata；
- 设置 forward context；
- 调用 model.forward；
- 根据 logits_indices 选择 hidden states；
- 调用 model.compute_logits；
- 调用 model.pooler；
- 调用 sampler；
- 构造 ModelRunnerOutput。
```

### 21.2 Model architecture 负责

```text
- 定义模型有哪些层；
- 定义每层如何 forward；
- 使用 vLLM layer 和并行 layer；
- 定义 input_ids / inputs_embeds / intermediate_tensors 如何进入模型；
- 定义 PP 中间张量如何返回；
- 定义 hidden_states 如何变成 logits；
- 定义权重如何从 checkpoint 加载；
- 声明 LoRA / PP / multimodal / quant / pooling 等能力。
```

边界一句话：

```text
ModelRunner 编排“这一轮 batch 怎么跑”；architecture 定义“模型本身怎么计算”。
```

---

## 22. architecture 和 layer 层的边界

architecture 不是 kernel 层。

以 Llama 为例：

```text
architecture：LlamaForCausalLM / LlamaModel / LlamaDecoderLayer；
layer：Attention / QKVParallelLinear / RowParallelLinear / RMSNorm / ParallelLMHead / LogitsProcessor；
kernel / backend：attention backend、quant method、CUDA kernel、Triton kernel 等。
```

architecture 做的是组合：

```text
把 vLLM 已有的高性能 layer 组合成某个具体模型结构。
```

layer 做的是计算：

```text
矩阵乘、attention、norm、activation、logits projection、pooling。
```

所以新增一个模型架构时，理想情况是：

```text
尽量复用已有 layer；
只在 architecture 层写模型结构和权重映射；
除非模型真的需要新算子，才新增 layer / kernel。
```

---

## 23. architecture 和 Scheduler / Executor / Worker 的边界

### 23.1 和 Scheduler

Scheduler 不知道具体模型层结构。

Scheduler 只关心：

```text
max_model_len
是否 encoder-decoder / multimodal
KV cache block 数
每个请求本轮调度多少 token
```

architecture 不参与 waiting / running 队列决策。

### 23.2 和 Executor

Executor 不直接接触 architecture。

Executor 只通过 Worker 分发：

```text
execute_model
sample_tokens
load_model
initialize_from_config
profile / sleep / wake_up
```

### 23.3 和 Worker

Worker 负责设备生命周期，最终调用：

```text
GPUModelRunner.load_model()
  → model_loader.load_model(...)
  → initialize_model(...)
  → architecture class 实例化
```

Worker 不实现模型结构。

---

## 24. 一个完整例子：普通 Llama generation 请求

把前面的信息串起来，一个 Llama 请求的大致链路是：

```text
1. HF config.architectures = ["LlamaForCausalLM"]

2. ModelConfig.architectures 读取到 LlamaForCausalLM

3. ModelRegistry.resolve_model_cls()
   → 找到 vllm.model_executor.models.llama:LlamaForCausalLM

4. initialize_model()
   → LlamaForCausalLM(vllm_config, prefix)
   → 内部创建 LlamaModel / layers / lm_head / logits_processor

5. DefaultModelLoader.load_weights()
   → 读取 safetensors / bin 权重
   → 调用 model.load_weights(weights)
   → q_proj/k_proj/v_proj 映射到 qkv_proj
   → gate_proj/up_proj 映射到 gate_up_proj

6. GPUModelRunner.execute_model()
   → 准备 input_ids / positions / attention metadata
   → set_forward_context(...)
   → self.model(...)

7. LlamaForCausalLM.forward()
   → LlamaModel.forward()
   → embedding / decoder layers / norm
   → hidden_states

8. GPUModelRunner
   → hidden_states[logits_indices]
   → model.compute_logits(sample_hidden_states)
   → logits

9. sample_tokens()
   → sampler
   → ModelRunnerOutput
```

这条链说明：

```text
architecture 是“模型结构和权重语义”的核心，
但它是在 ModelRunner 的 batch 执行编排下被调用。
```

---

## 25. 新增一个 model architecture 通常要做什么

如果要给 vLLM 增加一个新模型架构，通常需要：

```text
1. 在 vllm/model_executor/models/ 下新增或扩展模型文件；
2. 实现 nn.Module，并使用新式 __init__(vllm_config, prefix)；
3. 实现 embed_input_ids()；
4. 实现 forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)；
5. generation 模型实现 compute_logits()；
6. pooling 模型设置 is_pooling_model=True 并提供 pooler；
7. 实现 load_weights() 或使用 AutoWeightsLoader；
8. 如果支持 PP，继承 SupportsPP，并处理 make_layers / PPMissingLayer / IntermediateTensors；
9. 如果支持 LoRA，声明 packed_modules_mapping / embedding_modules；
10. 如果支持 multimodal，实现 SupportsMultiModal 所需方法；
11. 如果有特殊量化或 HF/vLLM 名字差异，提供 mapper / packed mapping；
12. 在 ModelRegistry 对应映射里注册 architecture name 到 model class。
```

最小要求可以记为：

```text
构造模型结构 + forward + compute_logits/pooler + load_weights + registry 注册。
```

---

## 26. 容易疑惑的点

### 26.1 architecture 是不是 HuggingFace 的 architecture？

不是完全等同。

HF config 里的 `architectures` 是名字来源；vLLM 的 `ModelRegistry` 会把这个名字解析成 vLLM 自己实现的 model class。

例如：

```text
HF: LlamaForCausalLM
vLLM: vllm.model_executor.models.llama.LlamaForCausalLM
```

名字相同，但实现不是直接使用 Transformers 的 PyTorch 实现，除非走 Transformers backend fallback。

### 26.2 architecture.forward() 会直接返回 logits 吗？

通常不会。

vLLM generation 模型通常：

```text
forward() → hidden_states
compute_logits(hidden_states) → logits
```

logits 的计算由 ModelRunner 在合适的位置触发。

### 26.3 为什么要把 compute_logits 拆出来？

因为不是所有 token 位置都需要 logits。

ModelRunner 会根据 `logits_indices` 只截取需要采样或算 logprobs 的 hidden states，再调用 `compute_logits()`。

### 26.4 load_weights 为什么放在 architecture 里？

因为只有具体架构知道：

```text
HF 权重名和 vLLM 参数名如何对应；
哪些参数被 fused；
哪些参数被 TP/PP 切分；
tie embeddings 时哪些权重要跳过；
特殊模型有没有额外 secondary weights。
```

### 26.5 Attention metadata 是 architecture 构造的吗？

不是。

Attention metadata 由 ModelRunner 构造，并通过 `set_forward_context()` 注入当前 forward context。

architecture 里的 `Attention` layer 在执行时使用这些上下文，但不负责从 SchedulerOutput 构造 metadata。

### 26.6 KV cache 是 architecture 管的吗？

不是完整管理。

```text
Scheduler：逻辑 block 分配；
ModelRunner：物理 KV cache tensor、slot mapping、attention metadata；
Architecture / Attention layer：在 forward 中读写当前 layer 对应的 KV cache。
```

### 26.7 PP 是 ModelRunner 做还是 architecture 做？

两者都参与：

```text
architecture：切层、PPMissingLayer、IntermediateTensors；
Worker / distributed：PP rank 间通信；
ModelRunner：传入 intermediate_tensors，处理非 last rank 返回值。
```

---

## 27. 最关键流程图

```text
配置解析阶段
  HF config
    → ModelArchitectureConfig
    → ModelConfig.architectures
    → ModelRegistry.inspect_model_cls()
    → runner_type / convert_type / multimodal / pooling / PP 能力确认

模型加载阶段
  get_model_architecture()
    → ModelRegistry.resolve_model_cls()
    → 可选 as_embedding_model / as_seq_cls_model
    → initialize_model()
    → model_class(vllm_config, prefix)
    → DefaultModelLoader.load_weights()
    → model.load_weights(weights)

执行阶段
  GPUModelRunner.execute_model()
    → _prepare_inputs()
    → _build_attention_metadata()
    → set_forward_context()
    → _model_forward()
    → architecture.forward()
    → hidden_states / IntermediateTensors
    → architecture.compute_logits() 或 architecture.pooler()
    → sample_tokens() / ModelRunnerOutput
```

---

## 28. 最关键对象关系

```text
ModelArchitectureConfig
  从 HF config 中抽取 vLLM runtime 关心的架构元信息。

ModelConfig
  保存模型配置，解析 runner / convert / dtype / multimodal / pooling 等策略。

ModelRegistry
  把 architecture 名字解析成 vLLM model class，并 inspect 模型能力。

model_class
  具体 architecture 类，例如 LlamaForCausalLM。

ModelLoader
  找到权重文件，并调用 model.load_weights()。

ModelRunner
  持有 model 实例，准备 batch 输入并调用 model.forward() / compute_logits() / pooler()。

vLLM layers
  Attention、Linear、Embedding、Norm、MoE、LogitsProcessor 等基础构件。
```

---

## 29. 从“回答问题”的角度总结

如果要问：

```text
Model architecture 在 vLLM 中负责什么？
```

可以回答：

```text
Model architecture 负责把一个具体模型结构实现成 vLLM 可执行的 nn.Module。

它根据 VllmConfig / HF config 构造 embedding、decoder layers、attention、MLP、norm、lm_head 或 pooler，
使用 vLLM 的并行和高性能 layer，
实现统一的 forward、compute_logits / pooler、load_weights 接口，
并通过 Protocol / class attributes 声明 LoRA、PP、multimodal、quant、pooling、Mamba、MoE 等能力。

它不负责调度请求、分配 KV block、构造 InputBatch、构造 attention metadata、采样策略或最终用户输出；
这些由 Scheduler、ModelRunner、Worker、Executor 和 OutputProcessor 完成。
```

职责关系可以压缩成：

```text
Scheduler 决定这轮跑什么；
ModelRunner 决定这轮 batch 怎么喂给模型；
Model architecture 定义模型本身怎么计算；
Layer / backend 执行具体算子；
Sampler / OutputProcessor 处理 token 和用户输出。
```

---

## 30. 一句话总结

```text
模型架构层是外部模型结构和 vLLM 执行框架之间的适配层：它定义模型结构、权重映射和 forward/logits/pooler 接口，让 ModelRunner 可以用统一方式执行不同模型。
```
