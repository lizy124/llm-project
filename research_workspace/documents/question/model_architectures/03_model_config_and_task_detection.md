# 03. ModelConfig 如何识别 task 和模型能力？

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/config/scheduler.py`
- `vllm/vllm/tasks.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/model_executor/models/interfaces_base.py`
- `vllm/vllm/model_executor/models/interfaces.py`
- `vllm/vllm/model_executor/models/adapters.py`
- `vllm/vllm/engine/arg_utils.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/pool/pooling_runner.py`

本问题关注：`ModelConfig` 如何把 HuggingFace config、模型 architecture、用户传入的 `--runner / --convert`、registry 里的模型接口能力组合起来，判断模型是 generation、pooling / embedding / classification、multimodal、transcription 还是 draft 相关形态，并进一步影响 Scheduler、Worker、ModelRunner 的执行链路。

---

## 1. 一句话回答

`ModelConfig` 不是简单读取一个 `task` 字段，而是通过 **runner 类型 + convert 类型 + registry 能力探测 + pooler task** 共同决定模型能力。

核心链路是：

```text
用户参数 / 默认值
  → ModelConfig.__post_init__()
  → 加载 HF config，拿到 architectures
  → registry 判断是否是 generation / pooling / multimodal 模型
  → 解析 runner_type
  → 解析 convert_type
  → registry.inspect_model_cls() 得到 _ModelInfo
  → 初始化 pooler_config / multimodal_config
  → Scheduler / ModelRunner 按 runner_type 走不同执行路径
```

一句话记忆：

```text
ModelConfig 负责把“这个模型是什么”和“这次要按什么形态运行”统一成 runner_type、convert_type、_model_info、pooler_config、multimodal_config。
```

---

## 2. 先区分三个概念：task、runner、convert

### 2.1 task：API / pooling 子任务层面的概念

`vllm/vllm/tasks.py:5` 定义 generation 任务：

```python
GenerationTask = Literal["generate", "transcription", "realtime"]
```

`vllm/vllm/tasks.py:8` 定义 pooling 任务：

```python
PoolingTask = Literal[
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
    "embed&token_classify",
]
```

也就是说，`task` 更像上层能力名：

```text
generate：文本生成；
transcription / realtime：语音或实时输入相关 generation；
embed：句向量 / embedding；
classify：序列分类 / rerank / cross-encoder；
token_embed：token 级 embedding / late interaction；
token_classify：token 分类；
plugin：自定义 pooling 任务。
```

### 2.2 runner_type：引擎实际启动哪类 runner

`RunnerType` 在 `vllm/vllm/config/scheduler.py:21`：

```python
RunnerType = Literal["generate", "pooling", "draft"]
```

它比 `task` 更粗：

```text
runner_type=generate：
  走生成模型路径，forward 后 compute_logits，再采样。

runner_type=pooling：
  走 pooling 模型路径，forward 后 pooler 聚合 hidden states。

runner_type=draft：
  用于投机解码 draft 模型相关路径。
```

`SchedulerConfig` 也保存这个字段：`vllm/vllm/config/scheduler.py:46`

```python
runner_type: RunnerType = "generate"
```

### 2.3 convert_type：把已有模型适配成另一种形态

`ModelConfig` 中相关类型在 `vllm/vllm/config/model.py:83` 到 `vllm/vllm/config/model.py:85`：

```python
RunnerOption = Literal["auto", RunnerType]
ConvertType = Literal["none", "embed", "classify"]
ConvertOption = Literal["auto", ConvertType]
```

允许的 runner / convert 组合在 `vllm/vllm/config/model.py:95`：

```python
_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],
    "pooling": ["embed", "classify"],
    "draft": [],
}
```

这说明：

```text
- generate runner 当前不靠 convert 适配；
- pooling runner 可以通过 embed / classify adapter 把 generation backbone 包成 pooling 模型；
- draft runner 是独立 runner 类型，不是 pooling 子任务。
```

---

## 3. ModelConfig 初始化时的主链路

入口是 `ModelConfig.__post_init__()`，位置：`vllm/vllm/config/model.py:483`

和 task / capability 直接相关的主线可以压缩成：

```text
1. 修正 model / tokenizer / revision；
2. get_config() 加载 HuggingFace config；
3. get_hf_text_config() 拿到 text config；
4. get_model_arch_config() 提取 architecture config；
5. self.architectures 得到模型 architecture 列表；
6. registry.is_text_generation_model() 判断 generation 能力；
7. registry.is_pooling_model() 判断 pooling 能力；
8. _get_runner_type() 解析 runner_type；
9. _get_convert_type() 解析 convert_type；
10. 校验 runner_type 和模型能力是否兼容；
11. registry.inspect_model_cls() 得到 _ModelInfo；
12. 按 runner_type 初始化 pooler_config；
13. 解析 dtype、max_model_len；
14. 按 _ModelInfo.supports_multimodal 初始化 multimodal_config；
15. 继续解析 quantization、cuda graph 等执行配置。
```

对应代码集中在 `vllm/vllm/config/model.py:555` 到 `vllm/vllm/config/model.py:721`。

关键片段是：

```python
architectures = self.architectures
registry = self.registry
is_generative_model = registry.is_text_generation_model(architectures, self)
is_pooling_model = registry.is_pooling_model(architectures, self)

self.runner_type = self._get_runner_type(
    architectures, self.runner, self.convert
)
self.convert_type = self._get_convert_type(
    architectures, self.runner_type, self.convert
)
```

位置：`vllm/vllm/config/model.py:578` 到 `vllm/vllm/config/model.py:588`

这说明：

```text
ModelConfig 会先用 registry 粗判模型能力，再解析 runner_type / convert_type。
runner_type 决定执行大类，convert_type 决定是否需要模型 adapter。
```

---

## 4. generation / pooling 能力从哪里来

### 4.1 registry 不是只看 architecture 名字

registry 中 `_ModelInfo` 记录模型类能力，位置：`vllm/vllm/model_executor/models/registry.py:755`

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

这些字段来自模型类接口探测，构造位置：`vllm/vllm/model_executor/models/registry.py:777`

```python
return _ModelInfo(
    architecture=model.__name__,
    is_text_generation_model=is_text_generation_model(model),
    is_pooling_model=is_pooling_model(model),
    ...
    supports_multimodal=supports_multimodal(model),
    ...
    supports_transcription=supports_transcription(model),
)
```

因此 registry 判断能力时，既会用 `architectures` 找到模型类，也会 inspect 模型类是否实现了对应接口。

### 4.2 generation 模型需要 compute_logits

`VllmModelForTextGeneration` 在 `vllm/vllm/model_executor/models/interfaces_base.py:113`：

```python
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    """The interface required for all generative models in vLLM."""

    def compute_logits(
        self,
        hidden_states: T,
    ) -> T | None:
        """Return `None` if TP rank > 0."""
```

判断函数在 `vllm/vllm/model_executor/models/interfaces_base.py:135`：

```python
def is_text_generation_model(...):
    if not is_vllm_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, VllmModelForTextGeneration)

    return isinstance(model, VllmModelForTextGeneration)
```

所以 generation 能力的本质是：

```text
模型是合法 vLLM model，且实现 VllmModelForTextGeneration 协议，也就是 forward 后能 compute_logits。
```

### 4.3 pooling 模型需要 is_pooling_model 标记和 pooler

`VllmModelForPooling` 在 `vllm/vllm/model_executor/models/interfaces_base.py:147`：

```python
class VllmModelForPooling(VllmModel[T_co], Protocol[T_co]):
    is_pooling_model: ClassVar[Literal[True]] = True
    default_seq_pooling_type: ClassVar[SequencePoolingType] = "LAST"
    default_tok_pooling_type: ClassVar[TokenPoolingType] = "ALL"
    attn_type: ClassVar[AttnTypeStr] = "decoder"
    score_type: ClassVar[ScoreType] = "bi-encoder"
    pooler: Pooler
```

判断函数在 `vllm/vllm/model_executor/models/interfaces_base.py:223`：

```python
def is_pooling_model(...):
    if not is_vllm_model(model):
        return False

    return getattr(model, "is_pooling_model", False)
```

所以 pooling 能力的本质是：

```text
模型是合法 vLLM model，且类上有 is_pooling_model=True，并且具备 pooler 语义。
```

---

## 5. runner_type 如何自动推断

### 5.1 用户显式指定时直接使用

`_get_runner_type()` 在 `vllm/vllm/config/model.py:940`：

```python
def _get_runner_type(...):
    if runner != "auto":
        return runner
```

也就是说：

```text
--runner generate → 直接使用 generate；
--runner pooling  → 直接使用 pooling；
--runner draft    → 直接使用 draft；
--runner auto     → 再走推断逻辑。
```

### 5.2 auto + convert=auto/none：按模型默认能力推断

`_get_runner_type()` 的核心逻辑在 `vllm/vllm/config/model.py:949`：

```python
if convert in {"auto", "none"}:
    runner_type = self._get_default_runner_type(architectures)
else:
    runner_type = "pooling"
```

含义是：

```text
- 如果用户没有强制 convert，runner_type 由 architecture / registry 默认能力决定；
- 如果用户指定 --convert embed/classify，就说明想走 pooling adapter，所以 runner_type=pooling。
```

### 5.3 默认 runner 的优先级

`_get_default_runner_type()` 在 `vllm/vllm/config/model.py:916`。

它的判断顺序是：

```text
1. 如果模型目录存在 Sentence Transformers pooling config → pooling；
2. 如果 architecture 是 vLLM 已注册模型：
   - registry.is_pooling_model(...) 为真 → pooling；
   - registry.is_text_generation_model(...) 为真 → generate；
3. 如果 architecture 后缀命中默认规则 → 使用后缀规则；
4. 否则默认 generate。
```

对应代码：`vllm/vllm/config/model.py:922` 到 `vllm/vllm/config/model.py:938`

```python
if get_pooling_config(self.model, self.revision):
    return "pooling"

for arch in architectures:
    if arch in registry.get_supported_archs():
        if registry.is_pooling_model(architectures, self):
            return "pooling"
        if registry.is_text_generation_model(architectures, self):
            return "generate"

    match = try_match_architecture_defaults(arch)
    if match:
        _, (runner_type, _) = match
        return runner_type

return "generate"
```

### 5.4 architecture 后缀默认规则

默认后缀规则在 `vllm/vllm/config/model.py:1940`：

```python
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForTokenClassification", ("pooling", "classify")),
    ("ForAudioClassification", ("pooling", "classify")),
    ("ForImageClassification", ("pooling", "classify")),
    ("ForVideoClassification", ("pooling", "classify")),
    ("ClassificationModel", ("pooling", "classify")),
    ("ForRewardModeling", ("pooling", "embed")),
    ("RewardModel", ("pooling", "embed")),
    ("Model", ("pooling", "embed")),
]
```

可以记成：

```text
*ForCausalLM / *ForConditionalGeneration / *ChatModel / *LMHeadModel
  → generate

*EmbeddingModel / *ForTextEncoding / *Model / *RewardModel
  → pooling + embed

*ForSequenceClassification / *ForTokenClassification / *ClassificationModel
  → pooling + classify
```

---

## 6. convert_type 如何自动推断

### 6.1 用户显式指定时直接使用

`_get_convert_type()` 在 `vllm/vllm/config/model.py:995`：

```python
if convert != "auto":
    return convert
```

也就是说：

```text
--convert none     → 不适配；
--convert embed    → 适配成 embedding/pooling；
--convert classify → 适配成 classification/pooling；
--convert auto     → 由 ModelConfig 推断。
```

### 6.2 默认 convert 的判断顺序

`_get_default_convert_type()` 在 `vllm/vllm/config/model.py:964`。

它的逻辑是：

```text
1. 如果模型已注册，且 runner_type=generate，模型本身支持 generation → none；
2. 如果模型已注册，且 runner_type=pooling，模型本身支持 pooling → none；
3. 如果 architecture 后缀命中默认规则 → 使用后缀规则里的 convert_type；
4. 如果 runner_type=pooling 但前面都没命中 → embed；
5. 否则 → none。
```

对应代码：`vllm/vllm/config/model.py:971` 到 `vllm/vllm/config/model.py:993`

```python
if runner_type == "generate" and registry.is_text_generation_model(...):
    return "none"
if runner_type == "pooling" and registry.is_pooling_model(...):
    return "none"

match = try_match_architecture_defaults(arch, runner_type=runner_type)
if match:
    _, (_, convert_type) = match
    return convert_type

if runner_type == "pooling":
    return "embed"

return "none"
```

### 6.3 convert 的实际作用：创建 pooling adapter

adapter 在 `vllm/vllm/model_executor/models/adapters.py`。

`as_embedding_model()` 位置：`vllm/vllm/model_executor/models/adapters.py:230`

```python
def as_embedding_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support embeddings.
    """
```

它会基于原始生成模型创建 `ModelForEmbedding`，并通过 `DispatchPooler.for_embedding(pooler_config)` 初始化 pooler。

关键位置：`vllm/vllm/model_executor/models/adapters.py:248` 到 `vllm/vllm/model_executor/models/adapters.py:258`

```python
class ModelForEmbedding(_create_pooling_model_cls(cls)):
    def _init_pooler(...):
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        return DispatchPooler.for_embedding(pooler_config)
```

所以 `--convert embed` 的含义不是改 HF config，而是：

```text
把已有 vLLM 模型类包成一个 pooling model class，去掉 LM head 语义，增加 pooler 输出。
```

---

## 7. runner_type 和能力校验

`ModelConfig` 解析出 `runner_type` / `convert_type` 后，会立即校验是否兼容。

位置：`vllm/vllm/config/model.py:590` 到 `vllm/vllm/config/model.py:612`

### 7.1 纯 pooling 模型不能按 generate 跑

```python
if (
    is_pooling_model
    and not is_generative_model
    and self.runner_type in ("draft", "generate")
):
    raise ValueError(
        f"Embedding models do not support `--runner {self.runner_type}`. "
        "Use `--runner pooling` or `--runner auto` for embedding models."
    )
```

含义：

```text
如果一个模型只有 pooling 能力，没有 generation 能力，不能强行 --runner generate 或 draft。
```

### 7.2 generate runner 必须有 generation 能力

```python
if self.runner_type == "generate" and not is_generative_model:
    generate_converts = _RUNNER_CONVERTS["generate"]
    if self.convert_type not in generate_converts:
        raise ValueError("This model does not support `--runner generate`.")
```

由于 `_RUNNER_CONVERTS["generate"]` 为空，实际含义是：

```text
当前没有“把非生成模型 convert 成 generate 模型”的路径。
```

### 7.3 pooling runner 可以依赖 pooling 能力或 convert

```python
if self.runner_type == "pooling" and not is_pooling_model:
    pooling_converts = _RUNNER_CONVERTS["pooling"]
    if self.convert_type not in pooling_converts:
        raise ValueError(...)
```

含义：

```text
如果模型本身不是 pooling 模型，仍然可以通过 --convert embed/classify 适配成 pooling 模型；
如果既不是 pooling 模型，也没有合法 convert，就报错。
```

---

## 8. pooling task 如何确定

`runner_type=pooling` 只说明走 pooling 执行路径，但具体是 `embed`、`classify`、`token_embed` 还是 `token_classify`，还要看 `PoolerConfig.task` 和模型支持任务。

入口是 `ModelConfig.get_pooling_task()`，位置：`vllm/vllm/config/model.py:1539`

```python
def get_pooling_task(
    self, supported_tasks: tuple[SupportedTask, ...]
) -> PoolingTask | None:
```

判断顺序是：

```text
1. 如果没有 pooler_config → None；
2. 如果用户显式设置 pooler_config.task：
   - 在 supported_tasks 中 → 使用它；
   - 不在 supported_tasks 中 → 报 Unsupported task；
3. 如果 supported_tasks 支持 token_classify，且 architecture 名中有 ForTokenClassification → token_classify；
4. 否则按优先级选择第一个支持项：
   embed&token_classify
   embed
   classify
   token_embed
   token_classify
   plugin
5. 都不支持 → None。
```

对应优先级在 `vllm/vllm/config/model.py:1561` 到 `vllm/vllm/config/model.py:1571`：

```python
priority: list[PoolingTask] = [
    "embed&token_classify",
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
]
for task in priority:
    if task in supported_tasks:
        return task
```

在 V1 的 `PoolingRunner` 中，目前支持面更窄。`get_supported_tasks()` 位置：`vllm/vllm/v1/worker/gpu/pool/pooling_runner.py:23`

```python
@staticmethod
def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
    if not is_pooling_model(model):
        return []
    assert "embed" in model.pooler.get_supported_tasks()
    return ["embed"]
```

这说明：

```text
ModelConfig 层知道 pooling task 的完整分类；
具体执行后端可能只实现其中一部分能力，V1 PoolingRunner 当前只返回 embed。
```

---

## 9. multimodal 能力如何识别

multimodal 不是 `runner_type`，而是模型类能力。

模型类通过 `SupportsMultiModal` 协议声明能力，位置：`vllm/vllm/model_executor/models/interfaces.py:100`

```python
class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True
    supports_multimodal_raw_input_only: ClassVar[bool] = False
    supports_encoder_tp_data: ClassVar[bool] = False
    requires_raw_input_tokens: ClassVar[bool] = False
```

registry inspect 时会写入 `_ModelInfo.supports_multimodal` 等字段：`vllm/vllm/model_executor/models/registry.py:787` 到 `vllm/vllm/model_executor/models/registry.py:793`

```python
supports_multimodal=supports_multimodal(model),
supports_multimodal_raw_input_only=supports_multimodal_raw_input_only(model),
requires_raw_input_tokens=requires_raw_input_tokens(model),
supports_multimodal_encoder_tp_data=supports_multimodal_encoder_tp_data(model),
```

`ModelConfig` 中初始化 multimodal config 的条件是：`vllm/vllm/config/model.py:683`

```python
if self._model_info.supports_multimodal:
    ...
    self.multimodal_config = MultiModalConfig(**mm_config_kwargs)
```

所以 multimodal 判断链路是：

```text
模型类实现 SupportsMultiModal / 注册 processor
  → registry.inspect_model_cls() 得到 supports_multimodal=True
  → ModelConfig 创建 MultiModalConfig
  → renderer / InputProcessor / ModelRunner 开启多模态预处理与 encoder 相关路径
```

注意：

```text
multimodal 可以和 generate 共存，也可以和 pooling 共存；
它不是 generate/pooling 的替代 runner_type，而是输入处理和模型 forward 的附加能力。
```

---

## 10. tokenizer、dtype、max_model_len、quantization 和 task 的关系

这些字段不直接决定 task，但会影响“这个 runner 能不能稳定跑”。

### 10.1 tokenizer_mode

`ModelConfig` 会在 architecture 确定后，为特殊模型设置默认 tokenizer mode。

位置：`vllm/vllm/config/model.py:621` 到 `vllm/vllm/config/model.py:637`

```text
model_impl=terratorch    → terratorch
MoonshotKimiaForCausalLM → kimi_audio
DeepseekV32ForCausalLM   → deepseek_v32
DeepseekV4ForCausalLM    → deepseek_v4
```

这不是 task detection，但会影响输入如何被 tokenizer / renderer 处理。

### 10.2 dtype

`dtype` 在 runner_type 确定后解析：`vllm/vllm/config/model.py:658`

```python
self.dtype: torch.dtype = _get_and_verify_dtype(
    self.model,
    self.hf_config,
    self.dtype,
    is_pooling_model=self.runner_type == "pooling",
    revision=self.revision,
    config_format=self.config_format,
)
```

重点是：

```text
is_pooling_model 参数来自 runner_type == "pooling"，所以 dtype 校验会感知当前是否按 pooling 模型运行。
```

### 10.3 max_model_len

`max_model_len` 在 config 和 runner 能力确定后解析：`vllm/vllm/config/model.py:675` 到 `vllm/vllm/config/model.py:676`

```python
self.original_max_model_len = self.max_model_len
self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
```

它不决定模型是 generate 还是 pooling，但会影响：

```text
- Scheduler token budget；
- KV cache 规划；
- chunked prefill / prefix caching 是否可用；
- 多模态 encoder token budget。
```

### 10.4 quantization

量化校验在 `ModelConfig.__post_init__()` 末尾触发：`vllm/vllm/config/model.py:744`

```python
self._verify_quantization()
```

它主要决定权重加载和 kernel 支持，不直接决定 `runner_type`。

### 10.5 trust_remote_code

`trust_remote_code` 会传给 `get_config()`：`vllm/vllm/config/model.py:555` 到 `vllm/vllm/config/model.py:564`

```python
hf_config = get_config(
    self.hf_config_path or self.model,
    self.trust_remote_code,
    self.revision,
    self.code_revision,
    self.config_format,
    ...
)
```

它影响能否加载远端自定义 config / architecture。间接地，它会影响：

```text
HF config 能不能加载；
architectures 能不能解析；
registry / transformers fallback 能不能找到模型类。
```

---

## 11. 对 Scheduler 和 Engine 参数的影响

`runner_type` 会继续传给 `SchedulerConfig`，使调度侧知道当前模型形态。

`SchedulerConfig.runner_type` 位置：`vllm/vllm/config/scheduler.py:46`

```python
runner_type: RunnerType = "generate"
```

在 engine args 默认值里，`runner_type` 会影响 chunked prefill 和 prefix caching 的默认行为。

位置：`vllm/vllm/engine/arg_utils.py:2497`

```python
def _set_default_chunked_prefill_and_prefix_caching_args(
    self, model_config: ModelConfig
) -> None:
    default_chunked_prefill = model_config.is_chunked_prefill_supported
    default_prefix_caching = (
        model_config.is_prefix_caching_supported and not model_config.is_hybrid
    )
```

其中对 pooling 的特殊 warning 在 `vllm/vllm/engine/arg_utils.py:2525` 和 `vllm/vllm/engine/arg_utils.py:2543`：

```python
elif (
    model_config.runner_type == "pooling"
    and self.enable_chunked_prefill
    and not default_chunked_prefill
):
    logger.warning_once(...)
```

```python
elif (
    model_config.runner_type == "pooling"
    and self.enable_prefix_caching
    and not default_prefix_caching
):
    logger.warning_once(...)
```

这说明：

```text
runner_type 不只是模型加载参数，也会影响 scheduler / cache / prefill 相关默认策略。
```

---

## 12. 对 ModelRunner 执行链路的影响

执行层最关键的分叉在 `GPUModelRunner.execute_model()`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4097` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4458`

### 12.1 共同前半段

无论 generate 还是 pooling，前半段都是：

```text
_update_states(scheduler_output)
  → _prepare_inputs(...)
  → _determine_batch_execution_and_padding(...)
  → _get_slot_mappings(...)
  → _build_attention_metadata(...)
  → _preprocess(...)
  → set_forward_context(...)
  → maybe_get_kv_connector_output(...)
  → _model_forward(...)
```

也就是说：

```text
runner_type 不是决定“要不要 forward”，而是决定 forward 后如何解释 hidden states。
```

### 12.2 pooling 分支

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4405`

```python
if self.is_pooling_model:
    # Return the pooling output.
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

含义：

```text
pooling 模型 forward 后不进入 sampler，直接把 hidden_states 交给 pooler，产出 embedding / pooling output。
```

### 12.3 generation 分支

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4414`

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

之后才会进入 `sample_tokens()`。

含义：

```text
generation 模型 forward 后只取需要采样位置的 hidden states，调用 compute_logits 得到 logits，再由 sampler 采样 token。
```

### 12.4 Pipeline Parallel 分支

如果当前不是最后一个 PP rank，会直接返回 `IntermediateTensors`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4397` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4403`

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

所以完整执行分叉可以记成：

```text
_model_forward()
  → 非最后 PP rank
      → IntermediateTensors
  → 最后 PP rank + pooling model
      → _pool(hidden_states)
  → 最后 PP rank + generation model
      → compute_logits(sample_hidden_states)
      → sample_tokens()
```

---

## 13. 典型场景串起来看

### 13.1 普通 CausalLM

```text
architectures = ["LlamaForCausalLM"]
  → 后缀命中 ForCausalLM
  → runner_type=generate
  → convert_type=none
  → registry 确认实现 compute_logits
  → ModelRunner forward
  → compute_logits
  → sampler
```

### 13.2 Sentence Transformers embedding 模型

```text
模型目录存在 pooling config
  → _get_default_runner_type() 优先返回 pooling
  → runner_type=pooling
  → 初始化 pooler_config
  → forward 后 _pool
  → 输出 embedding / pooling result
```

### 13.3 CausalLM 强制转 embedding

```text
用户传 --runner pooling --convert embed
  → runner_type=pooling
  → convert_type=embed
  → adapters.as_embedding_model() 包装原模型类
  → 初始化 DispatchPooler.for_embedding
  → forward 后 _pool
```

### 13.4 SequenceClassification 模型

```text
architectures = ["XxxForSequenceClassification"]
  → 后缀命中 ForSequenceClassification
  → runner_type=pooling
  → convert_type=classify
  → pooling task 倾向 classify / token_classify
  → forward 后走 pooler / classification 输出路径
```

### 13.5 多模态生成模型

```text
模型类 supports_multimodal=True
  → registry._ModelInfo.supports_multimodal=True
  → ModelConfig 初始化 MultiModalConfig
  → runner_type 仍可能是 generate
  → renderer / InputProcessor / ModelRunner 处理 image/video/audio 等输入
  → forward 后 compute_logits / sampler
```

---

## 14. 常见误区

### 14.1 误区一：ModelConfig 直接根据 `task` 字段判断所有能力

更准确的说法是：

```text
当前核心判断字段是 runner / convert / pooler_config.task，
ModelConfig 通过 registry 和 architecture 推断 runner_type / convert_type，
再在 pooling 场景下解析具体 PoolingTask。
```

### 14.2 误区二：architecture 后缀就是唯一依据

后缀规则只是 fallback / default。

更优先的是：

```text
- Sentence Transformers pooling config；
- vLLM registry 中模型类实际实现的接口；
- 用户显式指定的 --runner / --convert。
```

### 14.3 误区三：multimodal 是一种 runner_type

multimodal 不是 runner_type。

```text
runner_type 决定 generate / pooling / draft；
multimodal_config 决定输入预处理和 multimodal encoder 路径。
```

### 14.4 误区四：pooling 就等于 embed

ModelConfig 层面 pooling 包含：

```text
embed、classify、token_embed、token_classify、plugin、embed&token_classify
```

但具体执行后端可能只实现一部分。比如 V1 `PoolingRunner.get_supported_tasks()` 当前只返回 `embed`。

---

## 15. 一句话总览图

```text
HF config / architectures
  → ModelRegistry
      → is_text_generation_model
      → is_pooling_model
      → supports_multimodal
      → supports_transcription
      → default pooling / score / attn type
  → ModelConfig
      → runner_type: generate / pooling / draft
      → convert_type: none / embed / classify
      → pooler_config.task: embed / classify / token_embed / token_classify / plugin
      → multimodal_config: enabled / disabled
  → EngineArgs / SchedulerConfig
      → chunked prefill / prefix caching / scheduler runner_type
  → Worker / ModelRunner
      → forward
      → pooling: _pool
      → generate: compute_logits + sample_tokens
```

---

## 16. 一句话总结

```text
ModelConfig 的 task detection 本质上是“模型能力探测 + runner 解析 + pooling 子任务选择”：
registry 判断模型能不能生成、能不能 pooling、是否多模态；
runner_type 决定执行大分支；
convert_type 决定是否把生成模型适配成 pooling；
pooler_config.task 决定 pooling 内部的具体任务。
```
