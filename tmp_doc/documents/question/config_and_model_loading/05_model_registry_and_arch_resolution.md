# 05. model registry 如何解析模型类？

源码位置：

- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/interfaces_base.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/model_executor/models/adapters.py`

本问题关注：vLLM 如何根据 Hugging Face config 中的 `architectures`，找到对应的 vLLM 模型实现类；`ModelConfig` 为什么会提前 inspect 模型能力；`get_model_architecture()` 和 `initialize_model()` 如何把 registry 解析结果变成模型实例；以及 generation、pooling、embedding、多模态、Transformers fallback、adapter convert、pipeline parallel 等能力是如何判断出来的。

---

## 1. 一句话回答

`ModelRegistry` 是 vLLM 的 **HF architecture 名称到 vLLM 模型类的解析层**。

它负责回答：

```text
HF config.architectures 里的字符串
  → vLLM 是否支持
  → 对应哪个 Python 模型类
  → 这个模型类支持哪些能力
  → 是否需要 fallback 到 Transformers backend
  → 是否需要 adapter 转成 embedding / classification
```

最小链路是：

```text
HF config.json
  → hf_config.architectures
  → ModelConfig.architectures
  → ModelRegistry.inspect_model_cls()
  → ModelConfig._model_info / runner_type / convert_type
  → get_model_architecture()
  → ModelRegistry.resolve_model_cls()
  → as_embedding_model / as_seq_cls_model 可选包装
  → initialize_model()
  → model_class(vllm_config, prefix)
  → model instance
```

一句话压缩：

```text
ModelRegistry 把“模型配置里写的 architecture 名字”解析成“vLLM 实际要实例化的模型类和能力信息”。
```

---

## 2. 为什么需要 registry

Hugging Face 模型配置通常会有：

```json
{
  "architectures": ["LlamaForCausalLM"]
}
```

但 vLLM 不能直接拿这个字符串去实例化 HF 的 `LlamaForCausalLM`。

原因是 vLLM 需要自己的模型实现：

```text
- forward 接口要适配 vLLM 的 input_ids / positions / inputs_embeds / intermediate_tensors；
- attention 层要接入 vLLM 的 paged attention / KV cache；
- 权重要支持 vLLM 的 weight loader、packed modules、quantization；
- generation 模型要暴露 compute_logits；
- pooling 模型要暴露 pooler；
- 多模态模型要暴露 multimodal processor / embed_multimodal / get_language_model；
- PP / LoRA / Mamba / transcription 等能力需要通过接口声明。
```

所以 registry 的输入不是“模型路径”，而是：

```text
architecture 字符串 + ModelConfig
```

输出是：

```text
vLLM model class + 实际命中的 architecture 名称
```

---

## 3. registry 的入口在哪里

核心对象在：`code/vllm/vllm/model_executor/models/registry.py:979`

```python
@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)
```

全局实例在：`code/vllm/vllm/model_executor/models/registry.py:1378`

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

也就是说，vLLM 启动时会构造一个全局 `ModelRegistry`，其中的 key 是 HF config 里的 architecture 名字，value 是一个可延迟导入的模型类引用。

例如：`code/vllm/vllm/model_executor/models/registry.py:153`

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM")
```

这表示：

```text
HF architecture: LlamaForCausalLM
vLLM module:     vllm.model_executor.models.llama
vLLM class:      LlamaForCausalLM
```

注意：key 和 class name 不一定相同。

例如：`code/vllm/vllm/model_executor/models/registry.py:75`

```python
"AquilaModel": ("llama", "LlamaForCausalLM")
```

这表示 Aquila 的 HF architecture 会复用 vLLM 的 Llama 实现。

---

## 4. registry 里有哪些模型表

`registry.py` 不是只有一个平铺字典，而是先按任务类别拆表，再合并成 `_VLLM_MODELS`。

主要表包括：

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

位置：`code/vllm/vllm/model_executor/models/registry.py:71` 到 `code/vllm/vllm/model_executor/models/registry.py:696`

合并位置：`code/vllm/vllm/model_executor/models/registry.py:698`

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

这说明 registry 的第一层分类是“默认支持哪些 architecture 名称”，但真正能力判断不只靠这些表名。

例如一个模型可能同时出现在 generation 和 embedding 相关表里，最终还要看模型类是否满足接口。

---

## 5. architecture 名称从哪里来

### 5.1 常规来源：HF config

`ModelConfig.__post_init__()` 会调用 `get_config()` 读取 Hugging Face config：

```python
hf_config = get_config(...)
self.hf_config = hf_config
self.hf_text_config = get_hf_text_config(self.hf_config)
self.model_arch_config = self.get_model_arch_config()
```

位置：`code/vllm/vllm/config/model.py:534` 到 `code/vllm/vllm/config/model.py:548`

随后：

```python
architectures = self.architectures
registry = self.registry
```

位置：`code/vllm/vllm/config/model.py:557` 到 `code/vllm/vllm/config/model.py:558`

`ModelConfig.architectures` 属性返回：

```python
return self.model_arch_config.architectures
```

位置：`code/vllm/vllm/config/model.py:811` 到 `code/vllm/vllm/config/model.py:812`

所以更准确地说，vLLM 不是直接到处读 `hf_config.architectures`，而是先通过 `ModelArchitectureConfig` 做一层归一化，再从 `ModelConfig.architectures` 访问。

### 5.2 VllmConfig 可以覆写 architectures

`VllmConfig.with_hf_config()` 支持传入新的 `architectures`：

```python
if architectures is not None:
    hf_config = copy.deepcopy(hf_config)
    hf_config.architectures = architectures
```

位置：`code/vllm/vllm/config/vllm.py:667` 到 `code/vllm/vllm/config/vllm.py:673`

如果 `hf_config.architectures` 是 `None`，还会尝试从 Transformers 的 causal LM mapping 推导：

```python
elif hf_config.architectures is None:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )
    if hf_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        hf_config = copy.deepcopy(hf_config)
        hf_config.architectures = [
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[hf_config.model_type]
        ]
```

位置：`code/vllm/vllm/config/vllm.py:674` 到 `code/vllm/vllm/config/vllm.py:683`

这解释了一个现象：

```text
即使原始 config.json 没有 architectures，vLLM 也可能在配置阶段补出一个 architecture。
```

### 5.3 speculative decoding 也会改 architectures

draft / MTP 场景会覆写 `hf_config.architectures`。

例如：`code/vllm/vllm/config/speculative.py:319` 到 `code/vllm/vllm/config/speculative.py:321`

```python
hf_config.update(
    {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
)
```

这表示 target model 和 draft model 可能走不同 architecture 解析路径。

---

## 6. ModelConfig 为什么先 inspect 一次

在模型真正加载之前，`ModelConfig.__post_init__()` 会先问 registry：

```python
is_generative_model = registry.is_text_generation_model(architectures, self)
is_pooling_model = registry.is_pooling_model(architectures, self)
```

位置：`code/vllm/vllm/config/model.py:557` 到 `code/vllm/vllm/config/model.py:560`

然后决定：

```text
runner_type
convert_type
pooler_config
multimodal_config
dtype
attn_type
prefix caching / chunked prefill 能力
PP 是否支持
```

更关键的是，它会保存一次完整的模型能力信息：

```python
model_info, arch = registry.inspect_model_cls(architectures, self)
self._model_info = model_info
self._architecture = arch
logger.info("Resolved architecture: %s", arch)
```

位置：`code/vllm/vllm/config/model.py:593` 到 `code/vllm/vllm/config/model.py:598`

这一步不是创建模型实例，而是：

```text
导入或检查模型类，得到模型类的能力声明。
```

因此，模型加载前就能知道：

```text
- 是 generation 还是 pooling；
- 是否支持 multimodal；
- 是否支持 PP；
- attention 类型是什么；
- pooling 默认策略是什么；
- 是否 attention-free / hybrid / has inner state；
- 是否支持 transcription。
```

---

## 7. _ModelInfo：registry 检查出来的能力包

`_ModelInfo` 定义在：`code/vllm/vllm/model_executor/models/registry.py:746`

核心字段：

```python
@dataclass(frozen=True)
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

位置：`code/vllm/vllm/model_executor/models/registry.py:746` 到 `code/vllm/vllm/model_executor/models/registry.py:766`

生成逻辑在：`code/vllm/vllm/model_executor/models/registry.py:768`

```python
@staticmethod
def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":
    return _ModelInfo(
        architecture=model.__name__,
        is_text_generation_model=is_text_generation_model(model),
        is_pooling_model=is_pooling_model(model),
        default_seq_pooling_type=get_default_seq_pooling_type(model),
        default_tok_pooling_type=get_default_tok_pooling_type(model),
        attn_type=get_attn_type(model),
        score_type=get_score_type(model),
        supports_multimodal=supports_multimodal(model),
        supports_pp=supports_pp(model),
        has_inner_state=has_inner_state(model),
        is_attention_free=is_attention_free(model),
        is_hybrid=is_hybrid(model),
        ...
    )
```

也就是说，能力判断来自模型类本身的接口和类属性，而不是单纯来自 `_TEXT_GENERATION_MODELS` 这类表。

---

## 8. 模型类需要满足什么接口

### 8.1 所有 vLLM 模型的基础接口

基础接口在：`code/vllm/vllm/model_executor/models/interfaces_base.py:46`

```python
@runtime_checkable
class VllmModel(Protocol[T_co]):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...
```

检查逻辑：`code/vllm/vllm/model_executor/models/interfaces_base.py:103`

```python
def is_vllm_model(model):
    return (
        _check_vllm_model_init(model)
        and _check_vllm_model_embed_input_ids(model)
        and _check_vllm_model_forward(model)
    )
```

因此，一个类被 registry 找到还不够，它还必须像 vLLM 模型一样工作。

### 8.2 generation 模型接口

generation 模型接口在：`code/vllm/vllm/model_executor/models/interfaces_base.py:113`

```python
@runtime_checkable
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    def compute_logits(self, hidden_states: T) -> T | None:
        ...
```

判断函数：`code/vllm/vllm/model_executor/models/interfaces_base.py:135`

```python
def is_text_generation_model(model):
    if not is_vllm_model(model):
        return False
    return isinstance(model, VllmModelForTextGeneration)
```

所以 generation 模型除了 `forward()`，还要能从 hidden states 计算 logits。

这就是 `ModelRunner` 后面可以调用：

```text
self.model.compute_logits(sample_hidden_states)
```

的前提。

### 8.3 pooling 模型接口

pooling 模型接口在：`code/vllm/vllm/model_executor/models/interfaces_base.py:147`

```python
@runtime_checkable
class VllmModelForPooling(VllmModel[T_co], Protocol[T_co]):
    is_pooling_model: ClassVar[Literal[True]] = True
    default_seq_pooling_type: ClassVar[SequencePoolingType] = "LAST"
    default_tok_pooling_type: ClassVar[TokenPoolingType] = "ALL"
    attn_type: ClassVar[AttnTypeStr] = "decoder"
    score_type: ClassVar[ScoreType] = "bi-encoder"
    pooler: Pooler
```

判断函数：`code/vllm/vllm/model_executor/models/interfaces_base.py:223`

```python
def is_pooling_model(model):
    if not is_vllm_model(model):
        return False
    return getattr(model, "is_pooling_model", False)
```

这解释了为什么 pooling 能力通常是类属性 / adapter 添加出来的，而不是靠类名包含 `Embedding`。

### 8.4 多模态接口

多模态接口在：`code/vllm/vllm/model_executor/models/interfaces.py:94`

```python
@runtime_checkable
class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True
    supports_multimodal_raw_input_only: ClassVar[bool] = False
    supports_encoder_tp_data: ClassVar[bool] = False
    requires_raw_input_tokens: ClassVar[bool] = False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None: ...
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings: ...
    def get_language_model(self) -> VllmModel: ...
```

判断函数：`code/vllm/vllm/model_executor/models/interfaces.py:459`

```python
def supports_multimodal(model):
    return getattr(model, "supports_multimodal", False)
```

所以多模态模型的判定关键是：

```text
模型类声明 supports_multimodal=True，并实现 multimodal 相关接口。
```

---

## 9. LazyRegisteredModel：为什么 registry 要延迟导入

registry 的 value 通常不是已经导入的 class，而是 `_LazyRegisteredModel`。

定义位置：`code/vllm/vllm/model_executor/models/registry.py:832`

```python
@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    module_name: str
    class_name: str
```

真正加载类时：`code/vllm/vllm/model_executor/models/registry.py:947`

```python
def load_model_cls(self) -> type[nn.Module]:
    mod = importlib.import_module(self.module_name)
    return getattr(mod, self.class_name)
```

为什么要 lazy？

注释在 `register_model()` 里说得很直接：`code/vllm/vllm/model_executor/models/registry.py:992` 到 `code/vllm/vllm/model_executor/models/registry.py:1001`

```text
避免导入模型时初始化 CUDA，进而触发 fork subprocess 中的 CUDA re-initialize 问题。
```

更关键的是 inspect 路径。

`_LazyRegisteredModel.inspect_model_cls()` 会在另一个进程里检查模型类：

```python
# Performed in another process to avoid initializing CUDA
mi = _run_in_subprocess(
    lambda: _ModelInfo.from_model_cls(self.load_model_cls())
)
```

位置：`code/vllm/vllm/model_executor/models/registry.py:933` 到 `code/vllm/vllm/model_executor/models/registry.py:936`

这表示：

```text
ModelConfig 阶段需要知道模型能力，
但又不想在主进程里过早导入模型模块并碰到 CUDA 初始化副作用，
所以用 subprocess inspect。
```

---

## 10. inspect_model_cls 的解析顺序

入口：`code/vllm/vllm/model_executor/models/registry.py:1174`

```python
def inspect_model_cls(
    self,
    architectures: str | list[str],
    model_config: ModelConfig,
) -> tuple[_ModelInfo, str]:
```

这一步只返回能力信息和命中的 architecture，不返回模型类本身。

主流程可以概括为：

```text
1. architectures 是字符串则转成 list；
2. 空 architectures 直接报错；
3. 如果 model_impl == "transformers"，强制走 Transformers backend 解析；
4. 如果 model_impl == "terratorch"，强制解析 Terratorch；
5. 如果 model_impl == "auto" 且所有 arch 都不在 vLLM registry，先尝试 Transformers fallback；
6. 遍历 architectures，逐个 normalize 后 inspect；
7. 如果还失败，model_impl == "auto" 再尝试一次 Transformers fallback；
8. 全部失败则 _raise_for_unsupported()。
```

对应代码：`code/vllm/vllm/model_executor/models/registry.py:1179` 到 `code/vllm/vllm/model_executor/models/registry.py:1224`

这个顺序说明：

```text
vLLM 优先用自己的模型实现；
在 auto 模式下，如果没有 in-tree 实现，可以 fallback 到 Transformers backend；
在 transformers 模式下，则强制使用 Transformers backend。
```

---

## 11. resolve_model_cls 的解析顺序

真正加载模型类时，入口是：`code/vllm/vllm/model_executor/models/registry.py:1226`

```python
def resolve_model_cls(
    self,
    architectures: str | list[str],
    model_config: ModelConfig,
) -> tuple[type[nn.Module], str]:
```

它和 `inspect_model_cls()` 的结构基本一致，只是把：

```text
_try_inspect_model_cls()
```

换成：

```text
_try_load_model_cls()
```

关键流程：

```text
1. model_impl == "transformers"：先 _try_resolve_transformers()，再 load backend class；
2. model_impl == "terratorch"：加载 Terratorch；
3. model_impl == "auto" 且 registry 没有命中：尝试 Transformers fallback；
4. 遍历 architectures：_normalize_arch() 后尝试加载 vLLM 模型类；
5. 再尝试一次 Transformers fallback；
6. 失败则 _raise_for_unsupported()。
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1231` 到 `code/vllm/vllm/model_executor/models/registry.py:1278`

因此：

```text
inspect_model_cls() 用于配置阶段拿能力信息；
resolve_model_cls() 用于加载阶段拿真正 Python class。
```

---

## 12. _normalize_arch：为什么 architecture 会被改写

入口：`code/vllm/vllm/model_executor/models/registry.py:1148`

```python
def _normalize_arch(
    self,
    architecture: str,
    model_config: ModelConfig,
) -> str:
```

如果 architecture 已经在 registry 里，直接返回。

否则会尝试根据后缀匹配默认 runner / convert：

```python
match = try_match_architecture_defaults(
    architecture,
    runner_type=getattr(model_config, "runner_type", None),
    convert_type=getattr(model_config, "convert_type", None),
)
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1157` 到 `code/vllm/vllm/model_executor/models/registry.py:1162`

后缀规则在：`code/vllm/vllm/config/model.py:1887`

```python
_SUFFIX_TO_DEFAULTS = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForTokenClassification", ("pooling", "classify")),
    ...
    ("Model", ("pooling", "embed")),
]
```

位置：`code/vllm/vllm/config/model.py:1887` 到 `code/vllm/vllm/config/model.py:1907`

如果匹配成功，registry 会尝试把当前 suffix 替换成其它 suffix，寻找一个已经注册的 base architecture：

```python
for repl_suffix, _ in iter_architecture_defaults():
    base_arch = architecture.replace(suffix, repl_suffix)
    if base_arch in self.models:
        return base_arch
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1167` 到 `code/vllm/vllm/model_executor/models/registry.py:1170`

这个逻辑主要服务于：

```text
同一底座模型在 generation / embedding / classification 任务之间转换。
```

例如某个 `FooModel` 没有直接注册，但 `FooForCausalLM` 注册了，且当前 runner/convert 允许适配，那么 registry 可以先找到 base arch，再交给 adapter 包装。

---

## 13. Transformers backend fallback

`ModelConfig.model_impl` 定义在：`code/vllm/vllm/config/model.py:299`

```python
model_impl: str | ModelImpl = "auto"
```

含义：

```text
auto          优先 vLLM 实现，没有则 fallback 到 Transformers backend；
vllm          只用 vLLM 实现；
transformers  强制使用 Transformers backend；
terratorch    强制使用 TerraTorch。
```

registry fallback 的核心函数是：`code/vllm/vllm/model_executor/models/registry.py:1078`

```python
def _try_resolve_transformers(
    self,
    architecture: str,
    model_config: ModelConfig,
) -> str | None:
```

它做几件事：

```text
1. 如果 architecture 已经是 Transformers backend wrapper 类，直接返回；
2. 读取 hf_config.auto_map；
3. 根据 trust_remote_code 尝试加载 dynamic module；
4. 在 transformers 包里查找对应 architecture；
5. 检查 model_module.is_backend_compatible()；
6. 返回 ModelConfig._get_transformers_backend_cls()。
```

返回的不是原始 HF architecture，而是 vLLM 的 wrapper 类名。

`_get_transformers_backend_cls()` 在：`code/vllm/vllm/config/model.py:773`

它根据模型形态生成：

```text
TransformersForCausalLM
TransformersMoEForCausalLM
TransformersMultiModalForCausalLM
TransformersMultiModalMoEForCausalLM
TransformersEmbeddingModel
TransformersForSequenceClassification
...
```

例如：

```python
cls = "Transformers"
cls += "MultiModal" if self.hf_config != self.hf_text_config else ""
cls += "MoE" if self.is_moe else ""
...
cls += "ForCausalLM"
```

位置：`code/vllm/vllm/config/model.py:776` 到 `code/vllm/vllm/config/model.py:797`

如果 fallback 发生在 `get_model_architecture()` 中，vLLM 会提示：

```python
logger.warning_once(
    "%s has no vLLM implementation, falling back to Transformers "
    "implementation. Some features may not be supported and "
    "performance may not be optimal.",
    arch,
)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:193` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:201`

---

## 14. get_model_architecture：加载阶段的模型类解析入口

模型加载阶段的入口在：`code/vllm/vllm/model_executor/model_loader/utils.py:218`

```python
def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
```

它有一层缓存：

```python
key = hash(
    (
        model_config.model,
        model_config.convert_type,
        model_config.runner_type,
        model_config.trust_remote_code,
        model_config.model_impl,
        tuple(getattr(model_config.hf_config, "architectures", None) or []),
    )
)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:219` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:227`

真正解析在 `_get_model_architecture()`：`code/vllm/vllm/model_executor/model_loader/utils.py:183`

```python
architectures = getattr(model_config.hf_config, "architectures", None) or []

model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:183` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:191`

注意这里取的是：

```text
model_config.hf_config.architectures
```

而不是 `model_config.architectures`。

通常二者已经保持一致，但概念上：

```text
ModelConfig.__post_init__ 阶段用 model_arch_config.architectures 做能力推导；
模型加载阶段用 hf_config.architectures 做最终 class 解析。
```

---

## 15. convert_type：模型类解析后还可能被 adapter 包装

`get_model_architecture()` 拿到 `model_cls` 后，还会根据 `model_config.convert_type` 做包装。

代码：`code/vllm/vllm/model_executor/model_loader/utils.py:203` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:213`

```python
convert_type = model_config.convert_type
if convert_type == "none":
    pass
elif convert_type == "embed":
    logger.debug_once("Converting to embedding model.")
    model_cls = as_embedding_model(model_cls)
elif convert_type == "classify":
    logger.debug_once("Converting to sequence classification model.")
    model_cls = as_seq_cls_model(model_cls)
```

这说明：

```text
registry.resolve_model_cls() 返回的是 base class；
get_model_architecture() 返回的才是最终要实例化的 class。
```

例如一个 decoder-only LM 可以通过 adapter 变成 embedding model。

adapter 位置：`code/vllm/vllm/model_executor/models/adapters.py:136`

```python
class ModelForPooling(orig_cls, VllmModelForPooling):
    is_pooling_model = True
```

这类 adapter 会给原模型增加 pooling 接口和类属性，使其满足 `VllmModelForPooling`。

---

## 16. initialize_model：模型类如何变成实例

模型实例化入口：`code/vllm/vllm/model_executor/model_loader/utils.py:40`

```python
@instrument(span_name="Initialize model")
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
```

核心逻辑：

```python
if model_config is None:
    model_config = vllm_config.model_config
if model_class is None:
    model_class, _ = get_model_architecture(model_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:49` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:52`

如果有 quant config，会先配置 quant：

```python
if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:54` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:55`

新式模型类要求 `__init__` 支持：

```text
vllm_config
prefix
```

代码：`code/vllm/vllm/model_executor/model_loader/utils.py:57` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:64`

```python
signatures = inspect.signature(model_class.__init__)
all_params = [param.name for param in signatures.parameters.values()]
if "vllm_config" in all_params and "prefix" in all_params:
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(vllm_config=vllm_config, prefix=prefix)
        record_metadata_for_reloading(model)
        return model
```

如果是旧式 out-of-tree 模型类，vLLM 会尝试猜参数：

```text
prefix
config
cache_config
quant_config
lora_config
scheduler_config
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:66` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:97`

所以完整加载可以记成：

```text
resolve class
  → optional adapter
  → optional quant config patch
  → model_class(vllm_config, prefix)
  → record reload metadata
```

---

## 17. registry 和 ModelConfig 的职责边界

### 17.1 registry 负责

```text
- 保存 architecture → module/class 的映射；
- 延迟导入模型类；
- inspect 模型类能力；
- resolve 真正的模型类；
- 处理 Transformers backend fallback；
- 处理 unsupported / removed / plugin 模型报错；
- 支持外部 register_model()。
```

### 17.2 ModelConfig 负责

```text
- 加载 HF config；
- 归一化 architectures；
- 决定 runner_type；
- 决定 convert_type；
- 保存 _model_info 和 _architecture；
- 根据 _model_info 初始化 pooling / multimodal / dtype / attn_type 等配置；
- 校验 PP、prefix caching、chunked prefill 等后续能力。
```

### 17.3 model_loader 负责

```text
- 调用 registry.resolve_model_cls()；
- 根据 convert_type 包装模型类；
- 配置 quantization；
- 实例化模型对象；
- 加载后处理权重。
```

边界一句话：

```text
registry 找类和能力，ModelConfig 用能力决定配置，model_loader 用类创建模型实例。
```

---

## 18. unsupported architecture 如何报错

如果所有解析路径都失败，会进入 `_raise_for_unsupported()`。

入口：`code/vllm/vllm/model_executor/models/registry.py:1033`

它会先区分三类情况。

### 18.1 architecture 在 registry 里但 inspect/load 失败

```python
if any(arch in all_supported_archs for arch in architectures):
    raise ValueError(
        f"Model architectures {architectures} failed "
        "to be inspected. Please check the logs for more details."
    )
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1036` 到 `code/vllm/vllm/model_executor/models/registry.py:1040`

含义：

```text
不是不支持，而是导入或 inspect 过程中异常了。
```

### 18.2 以前支持但现在移除

表：`code/vllm/vllm/model_executor/models/registry.py:717`

```python
_PREVIOUSLY_SUPPORTED_MODELS = {
    "MotifForCausalLM": "0.10.2",
    "Phi3SmallForCausalLM": "0.9.2",
    ...
}
```

报错会提示最后支持的 vLLM 版本。

### 18.3 已迁移到 out-of-tree plugin

表：`code/vllm/vllm/model_executor/models/registry.py:738`

```python
_OOT_SUPPORTED_MODELS = {
    "BartModel": "https://github.com/vllm-project/bart-plugin",
    ...
}
```

报错会提示安装插件。

### 18.4 完全不支持

最后报错：`code/vllm/vllm/model_executor/models/registry.py:1061`

```python
raise ValueError(
    f"Model architectures {architectures} are not supported for now. "
    f"Supported architectures: {all_supported_archs}"
)
```

---

## 19. 外部模型如何注册

`ModelRegistry.register_model()` 支持注册 out-of-tree 模型。

位置：`code/vllm/vllm/model_executor/models/registry.py:987`

```python
def register_model(
    self,
    model_arch: str,
    model_cls: type[nn.Module] | str,
) -> None:
```

`model_cls` 支持两种形式：

```text
1. 直接传 torch.nn.Module 子类；
2. 传 "<module>:<class>" 字符串，延迟导入。
```

如果传字符串：

```python
model = _LazyRegisteredModel(*split_str)
```

如果传 class：

```python
model = _RegisteredModel.from_model_cls(model_cls)
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1015` 到 `code/vllm/vllm/model_executor/models/registry.py:1023`

推荐 out-of-tree 模型用字符串形式，因为它保持 lazy import，能减少 CUDA 初始化副作用。

---

## 20. 和模型文件本身的关系

以 Llama 为例，registry 只负责找到：

```text
vllm.model_executor.models.llama.LlamaForCausalLM
```

但模型类本身还要提供 vLLM 执行需要的能力，例如：

```text
- __init__(vllm_config, prefix)
- embed_input_ids()
- forward(input_ids, positions, ...)
- compute_logits()
- load_weights()
- packed_modules_mapping
- hf_to_vllm_mapper
```

registry 不解释这些方法的内部实现，只做：

```text
找到类、检查接口、返回类。
```

后续模型执行链路中，`GPUModelRunner.load_model()` 会通过 model loader 创建模型实例。

在 ModelRunner 侧：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5163` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5166`

```python
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config, model_config=self.model_config
)
```

具体 loader 内部最终会走到 `initialize_model()`。

---

## 21. 一个完整例子：LlamaForCausalLM

假设 HF config 里是：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama"
}
```

解析链路：

```text
1. ModelConfig 读取 hf_config；
2. ModelConfig.architectures 得到 ["LlamaForCausalLM"]；
3. registry.is_text_generation_model() 触发 inspect；
4. registry 在 _VLLM_MODELS 中找到：
   "LlamaForCausalLM" → ("llama", "LlamaForCausalLM")；
5. _resolve_module_name("llama") 得到：
   vllm.model_executor.models.llama；
6. _LazyRegisteredModel.inspect_model_cls() 在 subprocess 中导入类；
7. _ModelInfo.from_model_cls() 发现它是 text generation model；
8. ModelConfig.runner_type 通常解析为 generate；
9. get_model_architecture() 调用 resolve_model_cls()；
10. 返回 vLLM 的 LlamaForCausalLM class；
11. convert_type == "none"，不包装；
12. initialize_model() 调用 LlamaForCausalLM(vllm_config, prefix)。
```

最小流程图：

```text
"LlamaForCausalLM"
  → registry.models["LlamaForCausalLM"]
  → _LazyRegisteredModel("vllm.model_executor.models.llama", "LlamaForCausalLM")
  → import module
  → class LlamaForCausalLM
  → initialize_model()
  → LlamaForCausalLM(vllm_config, prefix)
```

---

## 22. 一个完整例子：pooling / embedding 转换

假设某个模型原本是 causal LM，但用户指定：

```text
--runner pooling
```

或 vLLM 根据 Sentence Transformers 配置推断出 pooling。

`ModelConfig._get_runner_type()` 会决定 runner：

```python
if runner != "auto":
    return runner
...
runner_type = self._get_default_runner_type(architectures)
```

位置：`code/vllm/vllm/config/model.py:894` 到 `code/vllm/vllm/config/model.py:916`

`ModelConfig._get_convert_type()` 决定 convert：

```python
if convert != "auto":
    return convert
convert_type = self._get_default_convert_type(architectures, runner_type)
```

位置：`code/vllm/vllm/config/model.py:949` 到 `code/vllm/vllm/config/model.py:968`

如果最后得到：

```text
runner_type = pooling
convert_type = embed
```

那么 `get_model_architecture()` 会：

```python
model_cls = as_embedding_model(model_cls)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:206` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:208`

这时最终实例化的不是原始 class，而是 adapter 生成的 class。

所以 pooling 路径里有两种情况：

```text
1. 模型类本身就是 pooling model；
2. generation/base model 被 adapter 包装成 pooling model。
```

---

## 23. 一个完整例子：Transformers fallback

假设某个 architecture 不在 `_VLLM_MODELS`，并且：

```text
model_impl = "auto"
```

registry 会尝试：

```python
arch = self._try_resolve_transformers(architectures[0], model_config)
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1195` 到 `code/vllm/vllm/model_executor/models/registry.py:1205`

如果 Transformers 模块可用并且 `is_backend_compatible()` 返回 True，则返回 wrapper architecture，例如：

```text
TransformersForCausalLM
TransformersMultiModalForCausalLM
TransformersEmbeddingModel
```

然后 registry 再从 `_TRANSFORMERS_BACKEND_MODELS` 中找到对应 vLLM wrapper class。

这条路径可以理解为：

```text
HF 原生模型类
  → Transformers backend compatibility check
  → vLLM Transformers wrapper class
  → initialize_model()
```

它不是直接把 HF 模型类当作 vLLM 模型类使用，而是通过 vLLM 的 Transformers backend wrapper 接起来。

---

## 24. architecture 和实际 class name 的区别

registry 返回的是：

```python
tuple[type[nn.Module], str]
```

第二个 `str` 是命中的 architecture 名称。

需要区分三种名字：

```text
HF config 里的 architecture：
  例如 "AquilaModel"

registry 归一化后用于查表的 architecture：
  可能还是 "AquilaModel"，也可能变成某个 base arch

实际加载的 Python class name：
  例如 "LlamaForCausalLM"
```

`_ModelInfo.architecture` 存的是模型类的 `__name__`：

```python
architecture=model.__name__
```

位置：`code/vllm/vllm/model_executor/models/registry.py:771`

而 registry 返回的 `arch` 可能是原始 architecture。

例如：

```text
HF architecture: AquilaModel
实际 class:      LlamaForCausalLM
```

这就是为什么排查问题时不能只看 config.json 的 `architectures`，还要看日志里的：

```text
Resolved architecture: ...
```

以及最终 `get_model_architecture()` 返回的 class。

---

## 25. registry 解析对后续执行有什么影响

解析结果会影响很多后续配置和执行分支。

### 25.1 runner_type

`ModelConfig._get_default_runner_type()` 会根据 registry 判断：

```python
if registry.is_pooling_model(architectures, self):
    return "pooling"
if registry.is_text_generation_model(architectures, self):
    return "generate"
```

位置：`code/vllm/vllm/config/model.py:880` 到 `code/vllm/vllm/config/model.py:885`

### 25.2 pooler_config

如果 `runner_type == "pooling"`，会初始化 `PoolerConfig`，并从 `_model_info` 读取默认 pooling 类型：

```python
default_seq_pooling_type = self._model_info.default_seq_pooling_type
default_tok_pooling_type = self._model_info.default_tok_pooling_type
```

位置：`code/vllm/vllm/config/model.py:620` 到 `code/vllm/vllm/config/model.py:637`

### 25.3 multimodal_config

如果模型支持多模态：

```python
if self._model_info.supports_multimodal:
    ...
    self.multimodal_config = MultiModalConfig(...)
```

位置：`code/vllm/vllm/config/model.py:663` 到 `code/vllm/vllm/config/model.py:702`

### 25.4 pipeline parallel 校验

PP 校验会调用：

```python
if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
    self.architectures, self
):
    raise NotImplementedError(...)
```

位置：`code/vllm/vllm/config/model.py:1173` 到 `code/vllm/vllm/config/model.py:1180`

### 25.5 attention 类型

`ModelConfig.attn_type` 会结合 `_model_info` 和 config 判断：

```python
elif self.is_hybrid:
    return "hybrid"
elif self.is_attention_free:
    return "attention_free"
elif self.is_encoder_decoder:
    return "encoder_decoder"
else:
    return "decoder"
```

位置：`code/vllm/vllm/config/model.py:1726` 到 `code/vllm/vllm/config/model.py:1743`

这些都会影响后面的 KV cache、attention backend、prefix caching、chunked prefill 和 ModelRunner 执行路径。

---

## 26. 容易混淆的点

### 26.1 registry 是否直接实例化模型？

不是。

registry 只返回：

```text
模型类 + architecture 名称
```

真正实例化发生在：

```text
initialize_model()
```

### 26.2 inspect_model_cls 和 resolve_model_cls 有什么区别？

```text
inspect_model_cls：配置阶段，拿 _ModelInfo，不创建模型实例；
resolve_model_cls：加载阶段，拿 Python class，准备实例化。
```

### 26.3 _TEXT_GENERATION_MODELS 里的模型一定是 generation 吗？

通常是，但最终判断以模型类接口为准。

registry 会调用：

```text
is_text_generation_model(model_cls)
is_pooling_model(model_cls)
supports_multimodal(model_cls)
```

这些函数检查的是类本身是否满足接口或声明属性。

### 26.4 `architectures[0]` 一定是最终用的模型吗？

不一定。

vLLM 会遍历 architectures，并可能：

```text
- normalize architecture；
- fallback 到 Transformers backend；
- 使用另一个 vLLM class 实现该 architecture；
- 通过 adapter 包装 class。
```

### 26.5 `model_impl="transformers"` 是什么意思？

它不是“直接使用 HF 原生类跑 vLLM 执行链路”，而是：

```text
强制 registry 解析到 vLLM 的 Transformers backend wrapper class。
```

### 26.6 为什么 inspect 要开子进程？

为了避免 inspect 模型类时在主进程导入模型模块并初始化 CUDA。

相关代码明确写着：

```python
# Performed in another process to avoid initializing CUDA
```

位置：`code/vllm/vllm/model_executor/models/registry.py:933`

---

## 27. 从“回答问题”的角度总结

如果要问：

```text
model registry 如何解析模型类？
```

可以回答：

```text
vLLM 先从 HF config / ModelArchitectureConfig 得到 architectures，
然后 ModelConfig 在初始化阶段通过 ModelRegistry.inspect_model_cls() 导入或检查模型类，得到 _ModelInfo，
用它决定 runner_type、convert_type、pooler_config、multimodal_config、attention 类型和并行能力。

真正加载模型时，model_loader.get_model_architecture() 再调用 ModelRegistry.resolve_model_cls()，
把 architectures 解析成具体 Python 模型类。
如果 model_impl=auto 且没有 vLLM in-tree 实现，registry 会尝试 Transformers backend fallback；
如果 convert_type 是 embed 或 classify，model_loader 会用 adapter 把 base class 包装成 pooling / classification class。
最后 initialize_model() 调用 model_class(vllm_config, prefix) 创建模型实例。
```

职责关系可以概括为：

```text
HF config：提供 architecture 名称；
ModelRegistry：把 architecture 名称解析成模型类和能力信息；
ModelConfig：使用能力信息决定运行配置；
model_loader：把模型类包装、量化配置、实例化；
ModelRunner：持有最终模型实例并执行 forward / logits / pooling / sampling。
```

---

## 28. 最关键流程图

```text
config.json
  └─ architectures: ["..."]
       ↓
ModelConfig.__post_init__()
  ├─ get_config()
  ├─ get_model_arch_config()
  ├─ architectures = self.architectures
  ├─ registry.is_text_generation_model()
  ├─ registry.is_pooling_model()
  ├─ _get_runner_type()
  ├─ _get_convert_type()
  └─ registry.inspect_model_cls()
       ├─ _normalize_arch()
       ├─ _try_resolve_transformers() 可选
       ├─ _LazyRegisteredModel.inspect_model_cls()
       │    └─ subprocess import model class
       └─ _ModelInfo.from_model_cls()
            ├─ is_text_generation_model
            ├─ is_pooling_model
            ├─ supports_multimodal
            ├─ supports_pp
            ├─ attn_type
            └─ score / pooling / transcription / hybrid flags

加载模型时：

GPUModelRunner.load_model()
  → model_loader.load_model()
  → initialize_model()
       ├─ get_model_architecture()
       │    ├─ registry.resolve_model_cls()
       │    │    ├─ _normalize_arch()
       │    │    ├─ _try_resolve_transformers() 可选
       │    │    └─ _LazyRegisteredModel.load_model_cls()
       │    ├─ as_embedding_model() 可选
       │    └─ as_seq_cls_model() 可选
       ├─ configure_quant_config() 可选
       └─ model_class(vllm_config, prefix)
            ↓
          model instance
```

---

## 29. 最小心智模型

可以把整个 registry 机制记成三层：

```text
第一层：名字解析
HF architecture string → registry dict → module/class

第二层：能力检查
model class → _ModelInfo → runner / pooling / multimodal / PP / attention flags

第三层：实例化准备
model class → optional adapter → optional quant config → model_class(vllm_config, prefix)
```

最终一句话：

```text
ModelRegistry 不执行模型，它决定 vLLM 应该用哪个模型类，以及这个类能以什么方式被执行。
```
