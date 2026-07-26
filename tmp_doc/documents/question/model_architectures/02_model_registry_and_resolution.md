# 02. Model registry 如何解析模型架构？

源码位置：

- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/model_executor/models/interfaces_base.py`
- `vllm/vllm/model_executor/models/interfaces.py`
- `vllm/vllm/model_executor/models/adapters.py`
- `vllm/vllm/model_executor/model_loader/utils.py`
- `vllm/vllm/config/model.py`
- `vllm/vllm/transformers_utils/dynamic_module.py`
- `vllm/vllm/platforms/interface.py`

这个问题关注：vLLM 如何从 HuggingFace config 里的 `architectures` 字段，找到 vLLM 内部真正可执行的 model class；registry 里为什么有多张模型表；`model_impl=auto / vllm / transformers / terratorch` 会如何影响解析；runner / convert 如何参与架构归一化；lazy import、子进程 inspect、能力探测、Transformers fallback 和错误提示分别发生在哪一层。

---

## 1. 一句话回答

`ModelRegistry` 是 **HF architecture name 到 vLLM model class 的解析表 + 能力探测器 + fallback 决策点**。

它负责：

```text
1. 根据 HF config.architectures 找到 vLLM 支持的 architecture；
2. 把 architecture 映射到实际 Python module 和 class；
3. 延迟 import 模型类，避免主进程过早初始化 CUDA；
4. inspect 模型类支持 generate / pooling / multimodal / PP 等能力；
5. 根据 model_impl 决定是否回退到 Transformers backend；
6. 给 ModelConfig / model_loader 提供最终可实例化的 model class。
```

它不负责：

```text
1. 下载模型权重；
2. 加载 checkpoint tensor；
3. 创建 Worker / ModelRunner；
4. 执行 forward；
5. 决定模型内部 layer 如何计算。
```

可以把它理解成：

```text
HF config 说明“这个 checkpoint 声称自己是什么架构”；
ModelRegistry 判断“vLLM 应该用哪个实现类来跑它”；
model_loader 再用这个类创建真正的 nn.Module。
```

---

## 2. 一句话总览链路

```text
用户传入 model 路径 / HF repo id
  → ModelConfig.__post_init__()
  → get_config() 读取 HF config
  → get_model_arch_config()
  → ModelConfig.architectures
  → ModelRegistry.inspect_model_cls()
  → 得到 _ModelInfo + 实际 architecture
  → ModelConfig 决定 runner_type / convert_type / multimodal_config 等
  → model_loader.get_model_architecture()
  → ModelRegistry.resolve_model_cls()
  → 必要时 as_embedding_model() / as_seq_cls_model()
  → initialize_model()
  → model_class(vllm_config=..., prefix=...)
```

最核心的两次 registry 调用是：

```text
初始化配置阶段：inspect_model_cls() 只拿模型能力信息；
真正建模阶段：resolve_model_cls() 加载并返回模型类。
```

---

## 3. architecture 名称从哪里来

入口在 `ModelConfig.__post_init__()`。

关键步骤：

```text
get_config()
  → self.hf_config
  → get_hf_text_config()
  → self.hf_text_config
  → get_model_arch_config()
  → self.model_arch_config
  → self.architectures
```

对应源码：`vllm/config/model.py:555` 到 `vllm/config/model.py:569`

`ModelConfig.architectures` 是一个 property：

```python
@property
def architectures(self) -> list[str]:
    return self.model_arch_config.architectures
```

位置：`vllm/config/model.py:857`

所以 registry 接收的不是直接裸读 `hf_config.architectures` 的结果，而是经过 `ModelArchitectureConfig` 归一后的 architecture 列表。

这点很重要，因为有些模型的 text config、multimodal config、量化 config、nested config 会先在 config 层被整理一次。

---

## 4. Registry 的核心数据结构

核心文件是：`vllm/model_executor/models/registry.py`

### 4.1 多张模型表

registry 顶部先按任务 / 形态维护多张字典：

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

位置：`registry.py:71` 到 `registry.py:686`

每个条目的形态是：

```python
"HFArchitectureName": ("module_name", "VllmClassName")
```

例如：

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM")
"Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration")
"BertModel": ("bert", "BertEmbeddingModel")
```

含义是：

```text
HF config.architectures 里的名字
  → vllm.model_executor.models.<module_name>
  → 该模块里的 <VllmClassName>
```

### 4.2 合并成 _VLLM_MODELS

这些表最终合并成一张总表：

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

位置：`registry.py:688` 到 `registry.py:699`

这说明 registry 的“支持列表”不是只服务文本生成，而是覆盖：

```text
generate / embedding / rerank / reward / classify / multimodal / speculative / transformers backend
```

### 4.3 创建全局 ModelRegistry

底部创建全局 registry：

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

位置：`registry.py:1405` 到 `registry.py:1413`

这里有两个关键点：

```text
1. 全部内置模型默认都是 _LazyRegisteredModel；
2. registry 记录的是“怎么 import”，不是启动时立即 import 所有模型。
```

---

## 5. module name 如何解析

registry entry 里的 module name 有两种形式。

### 5.1 普通短名

例如：

```python
"llama"
"qwen2"
"bert"
```

会被 `_resolve_module_name()` 转成：

```text
vllm.model_executor.models.llama
vllm.model_executor.models.qwen2
vllm.model_executor.models.bert
```

位置：`registry.py:1396` 到 `registry.py:1402`

### 5.2 完整模块名

有些模型不在传统 flat layout 下，而是直接写完整模块名：

```python
"vllm.models.deepseek_v4"
"vllm.models.minimax_m3"
```

如果 `mod_relname.startswith("vllm.")`，registry 会原样使用。

这说明新模型可以不局限于：

```text
vllm/model_executor/models/<name>.py
```

也可以放到新的硬件隔离 / 独立包布局中。

---

## 6. _RegisteredModel 与 _LazyRegisteredModel

registry 内部不是直接存 class，而是存 `_BaseRegisteredModel`。

抽象接口：

```python
class _BaseRegisteredModel(ABC):
    def inspect_model_cls(self) -> _ModelInfo: ...
    def load_model_cls(self) -> type[nn.Module]: ...
```

位置：`registry.py:808` 到 `registry.py:815`

### 6.1 _RegisteredModel：已经 import 的 class

`_RegisteredModel` 用于外部显式注册一个已经 import 的 PyTorch class。

```text
model_cls 是 nn.Module 子类
  → _ModelInfo.from_model_cls(model_cls)
  → 直接保存 model_cls
```

位置：`registry.py:818` 到 `registry.py:838`

### 6.2 _LazyRegisteredModel：延迟 import 的 class

内置表默认使用 `_LazyRegisteredModel`。

它只保存：

```text
module_name
class_name
```

真正加载时才执行：

```python
mod = importlib.import_module(self.module_name)
return getattr(mod, self.class_name)
```

位置：`registry.py:974` 到 `registry.py:976`

为什么要 lazy import？

```text
1. 避免 import 所有模型模块带来启动成本；
2. 避免某些模型模块 import 时触发 CUDA 初始化；
3. 支持在子进程中 inspect 模型接口，降低主进程副作用。
```

---

## 7. _ModelInfo：registry 不只返回 class，还返回能力

`_ModelInfo` 是 registry 对模型类的能力摘要。

字段包括：

```text
architecture
is_text_generation_model
is_pooling_model
attn_type
default_seq_pooling_type
default_tok_pooling_type
score_type
supports_multimodal
supports_multimodal_raw_input_only
requires_raw_input_tokens
supports_multimodal_encoder_tp_data
supports_pp
has_inner_state
is_attention_free
is_hybrid
has_noops
supports_mamba_prefix_caching
supports_transcription
supports_transcription_only
```

位置：`registry.py:756` 到 `registry.py:775`

`_ModelInfo.from_model_cls()` 会调用 `interfaces_base.py` 和 `interfaces.py` 里的探测函数：

```text
is_text_generation_model()
is_pooling_model()
get_attn_type()
get_default_seq_pooling_type()
get_score_type()
supports_multimodal()
supports_pp()
is_hybrid()
...
```

位置：`registry.py:777` 到 `registry.py:805`

这说明 registry 的作用不是“字符串查表”这么简单。

它还会回答：

```text
这个模型能不能 generate？
这个模型是不是 pooling 模型？
这个模型支不支持多模态？
这个模型支不支持 pipeline parallel？
这个模型是不是 attention-free / hybrid / mamba 类模型？
```

这些能力会反过来影响 `ModelConfig`、并行配置、worker、runner 和输入处理。

---

## 8. 为什么 inspect 要跑子进程

`_LazyRegisteredModel.inspect_model_cls()` 不直接在主进程里 import + inspect，而是：

```python
mi = _run_in_subprocess(
    lambda: _ModelInfo.from_model_cls(self.load_model_cls())
)
```

位置：`registry.py:960` 到 `registry.py:963`

注释写得很明确：

```text
Performed in another process to avoid initializing CUDA
```

也就是：

```text
inspect 模型类可能触发模块 import；
模块 import 可能初始化 CUDA；
主进程如果过早初始化 CUDA，后续 fork / worker 初始化可能出问题；
所以 vLLM 把 inspect 放到隔离子进程里做。
```

子进程执行入口是：

```text
python -m vllm.model_executor.models.registry
```

位置：`registry.py:701` 到 `registry.py:705`

`_run_in_subprocess()` 通过 `cloudpickle` 把 lambda 和输出路径传给子进程，子进程执行 `_run()` 后把结果 pickle 回临时文件。

位置：`registry.py:1418` 到 `registry.py:1445`

---

## 9. inspect 结果缓存

为了避免每次都 fork 子进程 inspect，`_LazyRegisteredModel` 会把 `_ModelInfo` 缓存到：

```text
<VLLM_CACHE_ROOT>/modelinfos/<module-class>.json
```

关键逻辑：

```text
1. 先按 vllm.model_executor.models.* 的传统路径定位模块文件；
2. 对 vllm.models.* 等完整模块名，用 importlib.util.find_spec() 定位实际文件；
3. 计算文件 hash；
4. 如果缓存里的 hash 相同，直接读取 _ModelInfo；
5. 如果缓存缺失或 stale，重新子进程 inspect；
6. inspect 成功后写回缓存。
```

对应源码：

```text
_get_cache_dir()               registry.py:851
_load_modelinfo_from_cache()   registry.py:877
_save_modelinfo_to_cache()     registry.py:909
inspect_model_cls()            registry.py:927
```

这个设计让 registry 同时满足：

```text
安全：避免主进程 CUDA 副作用；
性能：避免每次启动都重复 inspect；
正确性：模型文件 hash 变化后缓存自动失效。
```

---

## 10. ModelConfig 初始化阶段如何使用 registry

`ModelConfig.__post_init__()` 里，读取完 config 后立刻使用 registry。

### 10.1 判断 generate / pooling 能力

```python
architectures = self.architectures
registry = self.registry
is_generative_model = registry.is_text_generation_model(architectures, self)
is_pooling_model = registry.is_pooling_model(architectures, self)
```

位置：`vllm/config/model.py:578` 到 `vllm/config/model.py:581`

这一步用于后面判断：

```text
--runner generate 是否合法；
--runner pooling 是否合法；
--convert 是否需要把模型适配成 pooling / classify。
```

### 10.2 决定 runner_type / convert_type

```python
self.runner_type = self._get_runner_type(
    architectures, self.runner, self.convert
)
self.convert_type = self._get_convert_type(
    architectures, self.runner_type, self.convert
)
```

位置：`vllm/config/model.py:583` 到 `vllm/config/model.py:588`

如果用户传的是 `--runner auto` / `--convert auto`，ModelConfig 会结合 registry 能力和 architecture suffix 默认规则来推断。

### 10.3 缓存 _model_info 和 _architecture

```python
model_info, arch = registry.inspect_model_cls(architectures, self)
self._model_info = model_info
self._architecture = arch
logger.info("Resolved architecture: %s", arch)
```

位置：`vllm/config/model.py:614` 到 `vllm/config/model.py:619`

这一步的结果后续会被大量使用：

```text
self._model_info.supports_multimodal
self._model_info.supports_multimodal_encoder_tp_data
self._model_info.default_seq_pooling_type
self._model_info.default_tok_pooling_type
self._architecture
```

例如多模态配置初始化依赖：`vllm/config/model.py:683` 到 `vllm/config/model.py:721`

---

## 11. runner_type 和 convert_type 的默认规则

registry 解析不是孤立的，它会和 runner / convert 互相影响。

### 11.1 suffix 默认规则

`config/model.py` 里维护了一张 suffix 表：

```text
ForCausalLM                 → generate / none
ForConditionalGeneration    → generate / none
ChatModel                   → generate / none
LMHeadModel                 → generate / none
ForTextEncoding             → pooling / embed
EmbeddingModel              → pooling / embed
ForSequenceClassification   → pooling / classify
ForTokenClassification      → pooling / classify
ForAudioClassification      → pooling / classify
ForImageClassification      → pooling / classify
ForVideoClassification      → pooling / classify
ClassificationModel         → pooling / classify
ForRewardModeling           → pooling / embed
RewardModel                 → pooling / embed
Model                       → pooling / embed
```

位置：`vllm/config/model.py:1937` 到 `vllm/config/model.py:1957`

`try_match_architecture_defaults()` 会根据 architecture suffix 匹配默认 runner / convert。

位置：`vllm/config/model.py:1964` 到 `vllm/config/model.py:1981`

### 11.2 registry._normalize_arch()

有些模型只有基础架构被注册，但用户希望按 pooling/classify 方式使用。

`_normalize_arch()` 会尝试：

```text
当前 architecture 不在 registry 中；
但它的 suffix 能匹配 runner_type / convert_type；
则把 suffix 替换成其他默认 suffix；
如果替换后的 base_arch 在 registry 中，就用 base_arch 查询。
```

位置：`registry.py:1175` 到 `registry.py:1199`

可以理解成：

```text
SomeModelForSequenceClassification
  → 尝试找到 SomeModel / SomeModelForCausalLM 等 base arch
  → 再通过 adapter 转成 classify / embedding 形态
```

这就是 `architecture`、`runner_type`、`convert_type` 三者会互相影响的原因。

---

## 12. inspect_model_cls() 的完整解析顺序

入口：`_ModelRegistry.inspect_model_cls()`

位置：`registry.py:1201` 到 `registry.py:1251`

流程可以拆成 6 步。

### 12.1 参数归一

```text
architectures 是 str → 转成 list[str]
architectures 为空 → 抛 ValueError
```

位置：`registry.py:1206` 到 `registry.py:1209`

### 12.2 强制 Transformers backend

如果：

```text
model_config.model_impl == "transformers"
```

则先调用：

```text
_try_resolve_transformers(architectures[0], model_config)
```

如果能解析到 transformers backend class，再 inspect 这个 backend class。

位置：`registry.py:1211` 到 `registry.py:1217`

### 12.3 Terratorch 特例

如果：

```text
model_config.model_impl == "terratorch"
```

直接 inspect：

```text
Terratorch
```

位置：`registry.py:1218` 到 `registry.py:1220`

### 12.4 auto 模式下先尝试 Transformers fallback

如果所有 architecture 都不在 vLLM registry 中，并且：

```text
model_impl == "auto"
convert_type == "none"
```

则尝试 `_try_resolve_transformers()`。

位置：`registry.py:1222` 到 `registry.py:1232`

这里的语义是：

```text
vLLM 没有内置实现，且当前不需要先转换成 pooling/classify，
那就尝试用 Transformers backend 包一层。
```

### 12.5 遍历 architecture 列表

```python
for arch in architectures:
    normalized_arch = self._normalize_arch(arch, model_config)
    model_info = self._try_inspect_model_cls(normalized_arch)
    if model_info is not None:
        return (model_info, arch)
```

位置：`registry.py:1234` 到 `registry.py:1238`

注意返回的是：

```text
(model_info, arch)
```

这里的 `arch` 是原始 architecture，不一定是 normalized_arch。

### 12.6 auto 模式下再次 Transformers fallback

如果前面还没找到，且：

```text
model_impl == "auto"
```

会再次尝试 `_try_resolve_transformers()`。

位置：`registry.py:1240` 到 `registry.py:1249`

最后仍失败则进入 `_raise_for_unsupported()`。

---

## 13. resolve_model_cls() 的完整解析顺序

入口：`_ModelRegistry.resolve_model_cls()`

位置：`registry.py:1253` 到 `registry.py:1305`

它和 `inspect_model_cls()` 几乎同构，区别是：

```text
inspect_model_cls() 返回 _ModelInfo；
resolve_model_cls() 返回真正的 type[nn.Module]。
```

流程：

```text
1. architecture 参数归一；
2. model_impl == transformers 时强制走 Transformers backend；
3. model_impl == terratorch 时强制走 Terratorch；
4. model_impl == auto 且没有 vLLM 注册项时，先尝试 Transformers fallback；
5. 遍历 architectures，先 _normalize_arch()，再 _try_load_model_cls()；
6. auto 模式下最后再尝试一次 Transformers fallback；
7. 仍失败则 _raise_for_unsupported()。
```

真正 import class 的地方是：

```python
return _try_load_model_cls(model_arch, self.models[model_arch])
```

位置：`registry.py:1093` 到 `registry.py:1097`

`_try_load_model_cls()` 内部会先做平台校验：

```python
current_platform.verify_model_arch(model_arch)
```

位置：`registry.py:979` 到 `registry.py:991`

这意味着即使 registry 表里有某个模型，当前平台也可能拒绝加载它。

---

## 14. Transformers backend fallback 是怎么做的

入口：`_try_resolve_transformers()`

位置：`registry.py:1105` 到 `registry.py:1173`

它解决的是：

```text
当前 architecture 没有 vLLM 原生实现时，能不能用 Transformers backend 包起来跑。
```

### 14.1 已经是 backend class

如果 architecture 本身就在：

```text
_TRANSFORMERS_BACKEND_MODELS
```

直接返回这个 architecture。

位置：`registry.py:1110` 到 `registry.py:1111`

例如：

```text
TransformersForCausalLM
TransformersMoEForCausalLM
TransformersMultiModalForCausalLM
TransformersMultiModalMoEForCausalLM
TransformersEmbeddingModel
TransformersMoEEmbeddingModel
TransformersMultiModalEmbeddingModel
TransformersForSequenceClassification
TransformersMoEForSequenceClassification
TransformersMultiModalForSequenceClassification
```

### 14.2 处理 auto_map 和 trust_remote_code

如果 HF config 里有：

```json
"auto_map": {
  "AutoConfig": "...",
  "AutoModel": "...",
  "AutoModelForCausalLM": "..."
}
```

registry 会先尝试加载 `AutoConfig` / `AutoModel` 对应的动态模块。

位置：`registry.py:1113` 到 `registry.py:1135`

加载动态模块时会传入：

```text
model_config.model
model_config.revision
model_config.code_revision
model_config.trust_remote_code
```

也就是说：

```text
trust_remote_code 影响的是 Transformers 动态模块解析；
它不是让 vLLM 自动相信任意 architecture 都有原生实现。
```

### 14.3 找 Transformers 中的 architecture

registry 会先查：

```python
getattr(transformers, architecture, None)
```

位置：`registry.py:1137`

如果找不到，再从 `auto_map` 中找 `AutoModel*` 动态模块。

位置：`registry.py:1140` 到 `registry.py:1151`

如果仍找不到，并且用户显式指定了：

```text
model_impl == "transformers"
```

就抛出清晰错误。

位置：`registry.py:1153` 到 `registry.py:1162`

### 14.4 检查 backend 兼容性

即使找到了 Transformers 模型类，还会检查：

```python
model_module.is_backend_compatible()
```

位置：`registry.py:1164` 到 `registry.py:1171`

通过后返回：

```python
model_config._get_transformers_backend_cls()
```

位置：`registry.py:1173`

也就是说最终跑的不是 Transformers 原类本身，而是 vLLM 的 backend wrapper class，例如：

```text
TransformersForCausalLM
TransformersMoEForCausalLM
TransformersMultiModalForCausalLM
TransformersMultiModalMoEForCausalLM
TransformersEmbeddingModel
TransformersMoEEmbeddingModel
TransformersMultiModalEmbeddingModel
TransformersForSequenceClassification
TransformersMoEForSequenceClassification
TransformersMultiModalForSequenceClassification
```

---

## 15. model_impl 对解析的影响

`model_impl` 决定 registry 的容错策略。

### 15.1 model_impl="vllm"

语义：

```text
只接受 vLLM 原生实现。
```

如果 registry 表里找不到对应 architecture，不会 fallback 到 Transformers backend。

### 15.2 model_impl="transformers"

语义：

```text
强制使用 Transformers backend。
```

解析时会优先 `_try_resolve_transformers()`，并最终返回 vLLM 的 Transformers wrapper class。

### 15.3 model_impl="auto"

语义：

```text
优先 vLLM 原生实现；
没有原生实现时，尝试 Transformers backend；
如果 fallback 成功，会打印性能/功能可能不完整的 warning。
```

warning 位置在 model_loader：`vllm/model_executor/model_loader/utils.py:203` 到 `vllm/model_executor/model_loader/utils.py:211`

### 15.4 model_impl="terratorch"

语义：

```text
强制使用 Terratorch 特例模型。
```

registry 会直接解析到 `Terratorch`。

---

## 16. model_loader 如何接住 registry 的结果

真正创建模型之前，model_loader 会调用：

```python
model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

位置：`vllm/model_executor/model_loader/utils.py:196` 到 `vllm/model_executor/model_loader/utils.py:201`

然后根据 `convert_type` 可能做适配：

```python
if convert_type == "embed":
    model_cls = as_embedding_model(model_cls)
elif convert_type == "classify":
    model_cls = as_seq_cls_model(model_cls)
```

位置：`vllm/model_executor/model_loader/utils.py:213` 到 `vllm/model_executor/model_loader/utils.py:223`

最后返回：

```python
return model_cls, arch
```

位置：`vllm/model_executor/model_loader/utils.py:225`

### 16.1 结果会被缓存

`get_model_architecture()` 会按以下字段 hash 缓存解析结果：

```text
model
convert_type
runner_type
trust_remote_code
model_impl
hf_config.architectures
```

位置：`vllm/model_executor/model_loader/utils.py:228` 到 `vllm/model_executor/model_loader/utils.py:238`

### 16.2 initialize_model() 构造模型

`initialize_model()` 如果没有显式传 `model_class`，会调用：

```python
model_class, _ = get_model_architecture(model_config)
```

位置：`vllm/model_executor/model_loader/utils.py:50` 到 `vllm/model_executor/model_loader/utils.py:53`

新式模型类应该支持：

```python
model_class(vllm_config=vllm_config, prefix=prefix)
```

位置：`vllm/model_executor/model_loader/utils.py:58` 到 `vllm/model_executor/model_loader/utils.py:64`

如果是旧式 out-of-tree 模型类，vLLM 会 warning 并尝试用老参数名兼容。

位置：`vllm/model_executor/model_loader/utils.py:67` 到 `vllm/model_executor/model_loader/utils.py:98`

---

## 17. 外部模型如何注册

`_ModelRegistry.register_model()` 支持外部注册。

入口：`registry.py:1014` 到 `registry.py:1058`

支持两种形式：

```text
1. 直接传 nn.Module 子类；
2. 传 "<module>:<class>" 字符串，让 registry lazy import。
```

字符串形式更推荐用于避免 import 副作用：

```python
ModelRegistry.register_model(
    "MyModelForCausalLM",
    "my_package.my_model:MyModelForCausalLM",
)
```

如果 architecture 已经存在，新的注册会覆盖旧的注册项，并记录 debug 日志。

位置：`registry.py:1034` 到 `registry.py:1040`

当前 `ModelConfig` 还提供了 `model_class_overrides` 开发调试入口。每次访问 `model_config.registry` 时会先调用 `_maybe_register_model_class_overrides()`，把 `{"Architecture": "module:class"}` 形式的覆盖项注册到进程内的全局 `ModelRegistry`；该 guard 是进程本地的，worker 进程会各自注册一次。

位置：`vllm/config/model.py:283` 到 `vllm/config/model.py:289`，`vllm/config/model.py:826` 到 `vllm/config/model.py:854`

---

## 18. 不支持模型时如何报错

如果所有解析路径都失败，会进入 `_raise_for_unsupported()`。

位置：`registry.py:1060` 到 `registry.py:1091`

它会区分三类情况。

### 18.1 registry 里有，但 inspect/load 失败

如果 architecture 在 supported archs 里，但 inspect 失败：

```text
Model architectures [...] failed to be inspected.
Please check the logs for more details.
```

位置：`registry.py:1063` 到 `registry.py:1067`

### 18.2 以前支持，现在移除了

`_PREVIOUSLY_SUPPORTED_MODELS` 记录被移除的 architecture 和最后支持版本。

位置：`registry.py:707` 到 `registry.py:745`

错误会提示：

```text
这个 architecture 支持到 vX.Y.Z，若要使用请安装旧版 vLLM。
```

位置：`registry.py:1069` 到 `registry.py:1078`

### 18.3 已迁移到 out-of-tree plugin

`_OOT_SUPPORTED_MODELS` 记录已迁移到插件的模型。

位置：`registry.py:747` 到 `registry.py:752`

例如：

```text
BartModel → https://github.com/vllm-project/bart-plugin
```

错误会提示安装对应 plugin。

位置：`registry.py:1079` 到 `registry.py:1086`

### 18.4 完全不支持

最后才抛：

```text
Model architectures [...] are not supported for now.
Supported architectures: ...
```

位置：`registry.py:1088` 到 `registry.py:1091`

---

## 19. HF model_type 和 architectures 的区别

这是最容易混淆的点。

```text
model_type：通常描述 config 类型，例如 llama / qwen2 / bert；
architectures：通常描述模型类名，例如 LlamaForCausalLM / Qwen2ForCausalLM / BertModel。
```

在 vLLM 里：

```text
ModelRegistry 主要按 architectures 解析 model class；
ModelArchConfigConvertor 主要按 model_type 转换 config 结构；
_get_and_verify_dtype / quantization / max_len 等逻辑也会读 model_type。
```

所以：

```text
model_type 决定“怎么理解 config”；
architectures 决定“用哪个模型实现类”。
```

例如一个 Llama 家族模型可能有：

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"]
}
```

registry 查的是：

```text
LlamaForCausalLM
```

不是：

```text
llama
```

---

## 20. 多 architecture fallback 怎么理解

HF config 的 `architectures` 是 list。

registry 对它的处理是：

```text
按顺序遍历 architectures；
每个 architecture 先尝试 normalize；
能 inspect/load 成功就返回；
全部失败才走 fallback 或 unsupported。
```

位置：`registry.py:1234` 到 `registry.py:1238`，`registry.py:1288` 到 `registry.py:1292`

因此多个 architecture 的语义是：

```text
优先使用列表里靠前的 architecture；
后面的 architecture 是候选 fallback；
但 fallback 是否可用仍取决于 registry 是否支持和当前 model_impl 策略。
```

---

## 21. 一个 LlamaForCausalLM 的例子

假设 HF config 里有：

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"]
}
```

解析链路是：

```text
ModelConfig.architectures
  → ["LlamaForCausalLM"]
  → ModelRegistry.inspect_model_cls()
  → _VLLM_MODELS["LlamaForCausalLM"]
  → ("llama", "LlamaForCausalLM")
  → _resolve_module_name("llama")
  → vllm.model_executor.models.llama
  → _LazyRegisteredModel.inspect_model_cls()
  → 子进程 import vllm.model_executor.models.llama
  → getattr(module, "LlamaForCausalLM")
  → _ModelInfo.from_model_cls()
  → ModelConfig 缓存 _model_info / _architecture
```

真正创建模型时：

```text
model_loader.get_model_architecture()
  → ModelRegistry.resolve_model_cls()
  → import vllm.model_executor.models.llama
  → getattr(..., "LlamaForCausalLM")
  → initialize_model()
  → LlamaForCausalLM(vllm_config=..., prefix=...)
```

---

## 22. 一个 Transformers fallback 的例子

假设 HF config 里有一个 vLLM 原生表里没有的 architecture：

```json
{
  "architectures": ["SomeNewForCausalLM"]
}
```

并且用户使用默认：

```text
model_impl = "auto"
```

解析链路是：

```text
ModelRegistry 发现 SomeNewForCausalLM 不在 _VLLM_MODELS；
_try_resolve_transformers("SomeNewForCausalLM", model_config)；
先查 transformers.SomeNewForCausalLM；
找不到则查 hf_config.auto_map；
如果动态模块可加载且 is_backend_compatible() 为 True；
返回 model_config._get_transformers_backend_cls()；
最终用 vLLM 的 TransformersForCausalLM wrapper 跑。
```

如果用户显式设置：

```text
model_impl = "vllm"
```

则不会走这个 fallback。

---

## 23. Registry 和平台校验的关系

加载模型类时，`_try_load_model_cls()` 会调用：

```python
current_platform.verify_model_arch(model_arch)
```

位置：`registry.py:984` 到 `registry.py:987`

平台接口里也会解析模型类，用于平台能力判断。

位置：`vllm/platforms/interface.py:944`

这意味着：

```text
registry 表示 vLLM 代码层面知道这个 architecture；
platform.verify_model_arch() 表示当前硬件/平台是否允许它；
二者都通过，模型类才会真正返回给 loader。
```

---

## 24. 容易混淆点

### 24.1 registry 选的是 vLLM class，不是 HF transformers class

即使 architecture 名字来自 HF config，registry 返回的通常是 vLLM 自己实现的类：

```text
HF: LlamaForCausalLM
vLLM: vllm.model_executor.models.llama.LlamaForCausalLM
```

除非走 Transformers backend fallback，否则不会直接实例化 transformers 原类。

### 24.2 trust_remote_code 不等于“强制支持”

`trust_remote_code=True` 只允许从 HF repo 动态加载 Python 代码。

它仍然需要满足：

```text
1. auto_map 能找到对应动态模块；
2. Transformers 模型类 backend-compatible；
3. model_impl 策略允许 fallback；
4. vLLM wrapper 支持该类需要的能力。
```

### 24.3 inspect_model_cls 和 resolve_model_cls 不是重复劳动

```text
inspect_model_cls：配置阶段使用，拿能力信息；
resolve_model_cls：建模阶段使用，拿真正 class。
```

前者可以走缓存和子进程，后者会真正 import class 给 loader。

### 24.4 architecture 和实际返回 arch 可能不同

`_normalize_arch()` 可能用 normalized architecture 查表，但返回给上层的是原始 `arch`。

这会影响日志里的：

```text
Resolved architecture: ...
```

也会影响后续 `model_config.architecture` 的含义。

### 24.5 generate / pooling 是接口能力，不只是名字后缀

后缀规则只是默认推断。

最终模型是否支持 generate / pooling，还是由：

```text
is_text_generation_model(model_cls)
is_pooling_model(model_cls)
```

这些接口探测函数决定。

---

## 25. 如果要新增一个 vLLM 原生模型

最小路径是：

```text
1. 在 vllm/model_executor/models/ 下实现模型类，或放到 vllm.models.<name>；
2. 模型类继承 / 实现 vLLM 期望的接口；
3. 构造函数支持 vllm_config 和 prefix；
4. 在 registry.py 对应任务表里加入 architecture → (module, class)；
5. 如果支持多模态 / PP / pooling / transcription 等能力，实现对应接口标记；
6. 更新 tests/models/registry.py 里的示例模型。
```

源码文件顶部也提醒：新增 architecture 时要同步更新测试 registry。

位置：`registry.py:3` 到 `registry.py:6`

---

## 26. 一句话总结

```text
ModelRegistry 是 vLLM 模型加载链路中的“架构解析与能力确认层”：
它把 HF config 里的 architecture name 转成 vLLM 可执行 model class，
同时决定是否 fallback 到 Transformers backend，并把模型能力暴露给 ModelConfig 和 model_loader。
```

最终主线可以压缩成：

```text
HF config.architectures
  → ModelConfig.architectures
  → ModelRegistry.inspect_model_cls()  # 配置阶段：确认能力
  → ModelConfig.runner_type / convert_type / multimodal_config
  → ModelRegistry.resolve_model_cls()  # 加载阶段：返回 class
  → adapters.as_embedding_model / as_seq_cls_model  # 可选
  → initialize_model()
  → vLLM model instance
```
