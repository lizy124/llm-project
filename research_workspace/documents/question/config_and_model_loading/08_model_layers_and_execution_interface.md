# 08. 模型层和执行接口如何衔接 ModelRunner？

源码位置：

- `code/vllm/vllm/model_executor/models/interfaces_base.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/adapters.py`
- `code/vllm/vllm/model_executor/models/llama.py`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/logits_processor.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/model_executor/layers/pooler/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：模型被加载出来后，需要向 `ModelRunner` 暴露哪些执行接口，例如 `forward()`、`compute_logits()`、`pooler()`、`load_weights()`、`embed_input_ids()`、`make_empty_intermediate_tensors()`，以及 embedding、LM head、attention、MLP、MoE、LoRA、quantization layer 如何参与执行。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的文档风格，本篇按“先定接口契约，再走 ModelRunner 调用链，再拆典型模型和底层 layer，最后总结边界”的方式梳理模型层执行接口。

要回答的问题分成 12 组：

```text
1. vLLM 模型类必须暴露哪些基础接口？
2. generation 模型和 pooling 模型的接口差异是什么？
3. registry 如何通过接口判断模型能力？
4. loader 如何实例化模型并调用 load_weights()？
5. ModelRunner 如何调用 model.forward()？
6. generation 路径如何从 hidden_states 到 logits 再到 sampler？
7. pooling 路径如何从 hidden_states 到 pooler_output？
8. 典型 Llama 模型如何组织 embedding、layers、norm、lm_head？
9. Attention layer 如何通过 ForwardContext 接入 KV cache 和 attention metadata？
10. embedding / LM head / logits processor 如何支持 TP 和 vocab sharding？
11. LoRA、PP、多模态、quantization 如何通过接口插入？
12. 写一个 vLLM 模型实现时最小需要实现什么？
```

阅读顺序建议：

```text
03_model_config_and_hf_config.md
  → 05_model_registry_and_arch_resolution.md
  → 07_worker_load_model_flow.md
  → 08_model_layers_and_execution_interface.md
  → executor_worker_model_runner/07_model_forward_and_logits.md
  → executor_worker_model_runner/08_sampling_and_model_runner_output.md
```

本篇重点讲模型层和执行接口，不展开每个 attention backend kernel、每种 quantization backend 或每个模型文件的细节。

---

## 1. 一句话回答

vLLM 模型类不是只要能 `forward()` 就够了，它必须满足 `ModelRunner`、`ModelLoader`、`ModelRegistry` 共同依赖的一组接口契约。

最基础接口是：

```text
__init__(vllm_config, prefix="")
embed_input_ids(input_ids)
forward(input_ids, positions, ...)
load_weights(weights)
```

对于 generation 模型，还需要：

```text
compute_logits(hidden_states)
```

对于 pooling / embedding / classify / reward 模型，还需要：

```text
pooler(hidden_states, pooling_metadata)
is_pooling_model = True
```

对于 pipeline parallel，还需要：

```text
supports_pp = True
make_empty_intermediate_tensors(batch_size, dtype, device)
forward(..., intermediate_tensors=...)
```

最小 generation 主线是：

```text
GPUModelRunner._model_forward()
  → model.forward(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
  → hidden_states / IntermediateTensors
  → hidden_states[logits_indices]
  → model.compute_logits(sample_hidden_states)
  → GPUModelRunner.sample_tokens()
  → Sampler
```

Pooling 主线是：

```text
GPUModelRunner._model_forward()
  → model.forward(...)
  → hidden_states
  → model.pooler(hidden_states, pooling_metadata)
  → ModelRunnerOutput(pooler_output=...)
```

一句话压缩：

```text
ModelRunner 负责准备 batch 和上下文，模型类负责按 vLLM 接口消费 input_ids / positions / metadata 并产出 hidden states、logits 或 pooler output。
```

---

## 2. 模型接口在整体链路中的位置

模型类不是独立运行的，它被几个组件共同消费：

```text
ModelRegistry
  → inspect 模型类，判断 generation / pooling / multimodal / PP / LoRA 等能力

ModelLoader
  → instantiate model
  → model.load_weights(weights)
  → process_weights_after_loading()

ModelRunner
  → model.forward(...)
  → model.compute_logits(...)
  → model.pooler(...)

Attention backend / ForwardContext
  → attention layer 在 forward 中读取 attn_metadata / kv_cache / slot_mapping
```

核心关系是：

```text
HF config.architectures
  → ModelRegistry.resolve_model_cls()
  → vLLM model class
  → initialize_model(vllm_config)
  → model instance
  → model.load_weights()
  → GPUModelRunner._model_forward()
```

所以本篇的“执行接口”不是一个 Python 抽象基类问题，而是贯穿配置、加载、执行、采样、pooling 的运行时契约。

---

## 3. VllmModel：所有模型的基础接口

基础接口定义在 `interfaces_base.py`：

```python
@runtime_checkable
class VllmModel(Protocol[T_co]):
    """The interface required for all models in vLLM."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to `input_ids`."""
        ...

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:46` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:57`

vLLM 会检查三个点：

```text
__init__ 是否支持 vllm_config；
是否有 embed_input_ids；
forward 是否支持 input_ids 和 positions 关键字。
```

检查函数：

```python
def is_vllm_model(model):
    return (
        _check_vllm_model_init(model)
        and _check_vllm_model_embed_input_ids(model)
        and _check_vllm_model_forward(model)
    )
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:103` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:110`

其中 `_check_vllm_model_forward()` 会检查：

```python
vllm_kws = ("input_ids", "positions")
missing_kws = tuple(kw for kw in vllm_kws if not supports_kw(model_forward, kw))
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:76` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:92`

因此，vLLM 新式模型类的最小签名应该是：

```python
class MyModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        ...
```

---

## 4. generation 模型接口：compute_logits

文本生成模型扩展 `VllmModelForTextGeneration`：

```python
@runtime_checkable
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    """The interface required for all generative models in vLLM."""

    def compute_logits(
        self,
        hidden_states: T,
    ) -> T | None:
        """Return `None` if TP rank > 0."""
        ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:113` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:122`

判断函数：

```python
def is_text_generation_model(model):
    if not is_vllm_model(model):
        return False
    return isinstance(model, VllmModelForTextGeneration)
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:125` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:144`

注意注释中的 `Return None if TP rank > 0`。

这和 `LogitsProcessor` 的 TP gather 逻辑有关：部分平台 / gather 模式下，只有某些 rank 需要返回完整 logits，其他 rank 可能返回 `None`。ModelRunner 会处理这种情况。

---

## 5. pooling 模型接口：pooler 和 is_pooling_model

Pooling 模型扩展 `VllmModelForPooling`：

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

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:147` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:212`

判断 pooling 模型的逻辑很直接：

```python
def is_pooling_model(model):
    if not is_vllm_model(model):
        return False
    return getattr(model, "is_pooling_model", False)
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:215` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:229`

也就是说：

```text
pooling 模型必须先满足基础 VllmModel 接口；
然后通过 is_pooling_model=True 声明自己支持 pooling；
真正 pooling 逻辑放在 model.pooler。
```

默认 pooling 类型可以通过 decorator 设置：

```python
def default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL"):
    ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:235` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:247`

attention 类型也可以通过 decorator 设置：

```python
def attn_type(attn_type: AttnTypeStr):
    ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces_base.py:262` 到 `code/vllm/vllm/model_executor/models/interfaces_base.py:269`

---

## 6. registry 如何用接口判断模型能力

`ModelRegistry` inspect 模型类后，会形成 `_ModelInfo`，其中的能力字段来自接口判断函数。

`registry.py` 导入的判断函数包括：

```python
from .interfaces_base import (
    get_attn_type,
    get_default_seq_pooling_type,
    get_default_tok_pooling_type,
    get_score_type,
    is_pooling_model,
    is_text_generation_model,
)
```

位置：`code/vllm/vllm/model_executor/models/registry.py:60` 到 `code/vllm/vllm/model_executor/models/registry.py:67`

`ModelRegistry.is_text_generation_model()`：

```python
def is_text_generation_model(self, architectures, model_config) -> bool:
    model_cls, _ = self.inspect_model_cls(architectures, model_config)
    return model_cls.is_text_generation_model
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1307` 到 `code/vllm/vllm/model_executor/models/registry.py:1313`

`ModelRegistry.is_pooling_model()`：

```python
def is_pooling_model(self, architectures, model_config) -> bool:
    model_cls, _ = self.inspect_model_cls(architectures, model_config)
    return model_cls.is_pooling_model
```

位置：`code/vllm/vllm/model_executor/models/registry.py:1315` 到 `code/vllm/vllm/model_executor/models/registry.py:1321`

因此：

```text
ModelConfig 不直接猜模型能不能 generate / pooling；
它通过 registry inspect 模型类，再读模型类暴露的接口能力。
```

---

## 7. loader 如何实例化模型

模型加载主入口是 `BaseModelLoader.load_model()`：

```python
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )

    self.load_weights(model, model_config)
    ...
    process_weights_after_loading(model, model_config, target_device)

return model.eval()
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:42` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:82`

这里有几个关键点：

```text
模型初始化时默认 torch dtype 已被设为 model_config.dtype；
模型会直接创建在 load_device / target_device 上；
权重加载完成后会做 quantization / attention postprocess；
返回的是 eval() 模式模型。
```

`initialize_model()` 负责选择模型类并构造实例：

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

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:41` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:65`

这就是为什么新式模型类必须支持：

```text
__init__(*, vllm_config, prefix="")
```

---

## 8. get_model_architecture 如何处理 convert_type

`initialize_model()` 调用 `get_model_architecture(model_config)`。

内部逻辑：

```python
model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)

convert_type = model_config.convert_type
if convert_type == "none":
    pass
elif convert_type == "embed":
    model_cls = as_embedding_model(model_cls)
elif convert_type == "classify":
    model_cls = as_seq_cls_model(model_cls)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:193` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:225`

这说明 pooling / embedding / classify 不一定有独立模型文件。

vLLM 可以把原本的 generation 模型类包装成 pooling 模型类：

```text
as_embedding_model(LlamaForCausalLM)
  → LlamaForEmbedding

as_seq_cls_model(LlamaForCausalLM)
  → LlamaForSequenceClassification
```

典型例子在 `llama.py` 末尾：

```python
class LlamaBidirectionalForSequenceClassification(as_seq_cls_model(LlamaForCausalLM)):
    pass

class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    pass
```

位置：`code/vllm/vllm/model_executor/models/llama.py:543` 到 `code/vllm/vllm/model_executor/models/llama.py:552`

---

## 9. model.load_weights 是权重加载接口

默认 loader 最终调用：

```python
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:414` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:428`

这表示模型类需要实现：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    ...
```

返回的 `set[str]` 用于严格检查哪些参数已经加载。

如果开启 weights tracking：

```python
weights_to_load = {name for name, _ in model.named_parameters()}
weights_not_loaded = weights_to_load - loaded_weights
if weights_not_loaded:
    raise ValueError(...)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:434` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:445`

典型模型可以自己处理 packed weights，也可以使用 `AutoWeightsLoader`。

Llama 外层使用 `AutoWeightsLoader`：

```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
    )
    return loader.load_weights(weights)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:535` 到 `code/vllm/vllm/model_executor/models/llama.py:540`

Llama 内层 `LlamaModel` 则用 `WeightsMapper` 描述 packed mapping，并在 `load_weights()` 中交给 `AutoWeightsLoader`：

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

位置：`code/vllm/vllm/model_executor/models/llama.py:344` 到 `code/vllm/vllm/model_executor/models/llama.py:354`；`load_weights()` 位置：`code/vllm/vllm/model_executor/models/llama.py:441` 到 `code/vllm/vllm/model_executor/models/llama.py:443`

这对应 vLLM layer 里把 Q/K/V 或 gate/up 合并成单个并行参数的实现。

---

## 10. 权重加载后的 postprocess

权重加载后，loader 调用：

```python
process_weights_after_loading(model, model_config, target_device)
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:75` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:82`

它先遍历所有模块，处理 quantization method：

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:101` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:117`

然后处理 attention layer：

```python
for _, module in model.named_modules():
    if isinstance(module, (Attention, MLAAttention, MMEncoderAttention)) \
       and hasattr(module, "process_weights_after_loading"):
        with device_loading_context(module, target_device):
            module.process_weights_after_loading(model_config.dtype)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:118` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:127`

随后还会处理依赖 `process_weights_after_loading()` 的 `HpcModule`，避免 DummyModelLoader / reload 场景绕过模型自身 `load_weights()` 时漏掉 HPC 模块后处理。

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:129` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:136`

所以模型类初始化出来的 layer 不能只是普通 PyTorch layer；很多 layer 带有：

```text
quant_method；
weight_loader；
process_weights_after_loading；
参数 shard metadata。
```

这些会在加载阶段被 loader 使用。

---

## 11. ModelRunner 如何调用 model.forward

`GPUModelRunner._model_forward()` 是 ModelRunner 和模型类的直接接口：

```python
def _model_forward(
    self,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs: dict[str, Any],
) -> Any:
    return self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3840`

调用发生在 `execute_model()` 中，并被 `set_forward_context()` 包住：

```python
with (
    set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_mode,
        batch_descriptor=batch_desc,
        ubatch_slices=ubatch_slices_padded,
        slot_mapping=slot_mappings,
        skip_compiled=has_encoder_input,
    ) as kv_connector_output,
):
    model_output = self._model_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4362` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4386`

这说明模型 forward 本身只收到 `input_ids`、`positions`、`inputs_embeds` 等显式参数。

attention 所需的：

```text
attention metadata；
KV cache；
slot mapping；
CUDA graph mode；
batch descriptor；
```

主要通过 `ForwardContext` 传给底层 attention layer。

---

## 12. generation 路径：hidden states 到 logits

ModelRunner 拿到 forward 结果后，会在 generation 路径取需要计算 logits 的 hidden states：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4414` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4415`

另一个分支也会调用同样接口：

```python
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4434`

prompt logprobs 路径也会调用：

```python
logits = self.model.compute_logits(prompt_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5611` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5617`

因此，generation 模型的职责边界是：

```text
forward()：返回 hidden states；
compute_logits()：把 hidden states 转成 vocab logits；
ModelRunner / Sampler：根据 logits 做 grammar mask、采样、logprobs。
```

模型类不负责 sampler，也不负责生成 `ModelRunnerOutput`。

---

## 13. pooling 路径：hidden states 到 pooler output

Pooling 路径由 `GPUModelRunner._pool()` 处理。

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3392`

它先构造 pooling metadata：

```python
pooling_metadata = self.input_batch.get_pooling_metadata()
pooling_metadata.build_pooling_cursor(
    num_scheduled_tokens_np,
    seq_lens_cpu,
    device=hidden_states.device,
    query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3407` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3413`

然后调用模型的 `pooler`：

```python
model = cast(VllmModelForPooling, self.model)
raw_pooler_output: PoolerOutput = model.pooler(
    hidden_states=hidden_states, pooling_metadata=pooling_metadata
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3415` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3418`

后续 `_pool()` 负责：

```text
判断哪些请求 finished；
late interaction postprocess；
构造 ModelRunnerOutput；
把 pooler output 复制回 CPU 或包装成 AsyncGPUPoolingModelRunnerOutput。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3420` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3455`

所以 pooling 模型类只需要提供 `pooler`，而不需要自己处理 request id、finished mask、异步复制或输出对象。

---

## 14. Pooler 接口和 DispatchPooler

Pooler 抽象接口：

```python
class Pooler(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        ...

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        ...
```

位置：`code/vllm/vllm/model_executor/layers/pooler/abstract.py:16` 到 `code/vllm/vllm/model_executor/layers/pooler/abstract.py:36`

常见的 `DispatchPooler` 会按 task 分发到子 pooler：

```python
class DispatchPooler(Pooler):
    """Dispatches calls to a sub-pooler based on the pooling task."""
```

位置：`code/vllm/vllm/model_executor/layers/pooler/special.py:25` 到 `code/vllm/vllm/model_executor/layers/pooler/special.py:26`

embedding pooler：

```python
@classmethod
def for_embedding(cls, pooler_config):
    return cls({
        "token_embed": pooler_for_token_embed(pooler_config),
        "embed": pooler_for_embed(pooler_config),
    })
```

位置：`code/vllm/vllm/model_executor/layers/pooler/special.py:28` 到 `code/vllm/vllm/model_executor/layers/pooler/special.py:35`

sequence classification pooler：

```python
@classmethod
def for_seq_cls(cls, pooler_config, pooling=None, classifier=None):
    return cls({
        "token_classify": pooler_for_token_classify(...),
        "classify": pooler_for_classify(...),
    })
```

位置：`code/vllm/vllm/model_executor/layers/pooler/special.py:37` 到 `code/vllm/vllm/model_executor/layers/pooler/special.py:58`

`forward()` 内部按 `pooling_metadata.tasks` 分组：

```python
for task, group in groupby(pooling_metadata.tasks):
    pooler = poolers_by_task.get(task)
    group_metadata = pooling_metadata[offset : offset + num_items]
    group_output = pooler(group_hidden_states, group_metadata)
    outputs.extend(group_output)
```

位置：`code/vllm/vllm/model_executor/layers/pooler/special.py:78` 到 `code/vllm/vllm/model_executor/layers/pooler/special.py:133`

这解释了一个 batch 中不同 pooling task 如何由同一个 model.pooler 分发处理。

---

## 15. as_embedding_model / as_seq_cls_model 如何把生成模型改成 pooling 模型

`adapters.py` 提供了模型类包装器。

核心函数 `_create_pooling_model_cls()`：

```python
class ModelForPooling(orig_cls, VllmModelForPooling):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix="", **kwargs):
        with no_init_weights(
            self,
            lambda mod: StageMissingLayer("output", mod),
            targets=(LogitsProcessor, ParallelLMHead),
        ):
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        pooler = getattr(self, "pooler", None)
        ...
        if not pooler:
            pooler = self._init_pooler(vllm_config, prefix=prefix)
        self.pooler = pooler
```

位置：`code/vllm/vllm/model_executor/models/adapters.py:129` 到 `code/vllm/vllm/model_executor/models/adapters.py:227`

这里的关键点是：

```text
继承原始 generation model class；
通过 VllmModelForPooling 声明 pooling 能力；
初始化时跳过 LogitsProcessor 和 ParallelLMHead；
创建 pooler；
load_weights 支持从 ForCausalLM 或 Model 风格权重映射。
```

embedding adapter：

```python
class ModelForEmbedding(_create_pooling_model_cls(cls)):
    def _init_pooler(self, vllm_config, prefix=""):
        pooler_config = vllm_config.model_config.pooler_config
        return DispatchPooler.for_embedding(pooler_config)
```

位置：`code/vllm/vllm/model_executor/models/adapters.py:230` 到 `code/vllm/vllm/model_executor/models/adapters.py:261`

sequence classification adapter：

```python
class ModelForSequenceClassification(
    _create_pooling_model_cls(cls), SupportsCrossEncoding
):
    def _init_pooler(...):
        self.score = ReplicatedLinear(...)
        return DispatchPooler.for_seq_cls(pooler_config, classifier=self.score)
```

位置：`code/vllm/vllm/model_executor/models/adapters.py:264` 到 `code/vllm/vllm/model_executor/models/adapters.py:369`

所以 pooling 能力可能来自：

```text
模型类原生实现；
adapters.py 动态包装；
ModelConfig.convert_type 决定是否包装。
```

---

## 16. 典型模型结构：LlamaForCausalLM 外层

`LlamaForCausalLM` 是典型 generation 模型。

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

位置：`code/vllm/vllm/model_executor/models/llama.py:446` 到 `code/vllm/vllm/model_executor/models/llama.py:454`

它声明 LoRA packed modules：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

位置：`code/vllm/vllm/model_executor/models/llama.py:455` 到 `code/vllm/vllm/model_executor/models/llama.py:460`

LoRA embedding modules：

```python
embedding_modules = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}
```

位置：`code/vllm/vllm/model_executor/models/llama.py:461` 到 `code/vllm/vllm/model_executor/models/llama.py:464`

初始化时创建三大部分：

```text
self.model：LlamaModel 主干；
self.lm_head：ParallelLMHead；
self.logits_processor：LogitsProcessor。
```

代码：

```python
self.model = self._init_model(...)

if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)
    if config.tie_word_embeddings:
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
    self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
else:
    self.lm_head = PPMissingLayer()
```

位置：`code/vllm/vllm/model_executor/models/llama.py:478` 到 `code/vllm/vllm/model_executor/models/llama.py:503`

外层暴露接口：

```python
def embed_input_ids(self, input_ids):
    return self.model.embed_input_ids(input_ids)

def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
    return model_output

def compute_logits(self, hidden_states):
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

位置：`code/vllm/vllm/model_executor/models/llama.py:513` 到 `code/vllm/vllm/model_executor/models/llama.py:533`

这就是 vLLM generation 模型最典型的外层形态。

---

## 17. 典型模型结构：LlamaModel 主干

`LlamaModel` 是真正执行 transformer block 的主干。

初始化时创建：

```text
embed_tokens：VocabParallelEmbedding；
layers：按 PP 切分后的 LlamaDecoderLayer 列表；
norm：最后一个 PP rank 才有；
make_empty_intermediate_tensors：PP profiling / 中间张量接口。
```

代码：

```python
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

self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
    ["hidden_states", "residual"], config.hidden_size
)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:356` 到 `code/vllm/vllm/model_executor/models/llama.py:395`

`embed_input_ids()`：

```python
def embed_input_ids(self, input_ids):
    return self.embed_tokens(input_ids)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:397` 到 `code/vllm/vllm/model_executor/models/llama.py:398`

`forward()` 处理 PP：

```python
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)
    residual = None
else:
    assert intermediate_tensors is not None
    hidden_states = intermediate_tensors["hidden_states"]
    residual = intermediate_tensors["residual"]
```

位置：`code/vllm/vllm/model_executor/models/llama.py:400` 到 `code/vllm/vllm/model_executor/models/llama.py:418`

逐层执行：

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(
        positions, hidden_states, residual, **extra_layer_kwargs
    )
```

位置：`code/vllm/vllm/model_executor/models/llama.py:419` 到 `code/vllm/vllm/model_executor/models/llama.py:428`

非 last PP rank 返回中间张量：

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

位置：`code/vllm/vllm/model_executor/models/llama.py:430` 到 `code/vllm/vllm/model_executor/models/llama.py:433`

last rank 做 norm 并返回 hidden states：

```python
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```

位置：`code/vllm/vllm/model_executor/models/llama.py:435` 到 `code/vllm/vllm/model_executor/models/llama.py:439`

---

## 18. 典型 decoder layer：Attention + MLP + Norm

`LlamaDecoderLayer` 初始化：

```text
self.self_attn = LlamaAttention(...)
self.mlp = LlamaMLP(...)
self.input_layernorm = RMSNorm(...)
self.post_attention_layernorm = RMSNorm(...)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:248` 到 `code/vllm/vllm/model_executor/models/llama.py:331`

forward 结构：

```python
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
return hidden_states, residual
```

位置：`code/vllm/vllm/model_executor/models/llama.py:310` 到 `code/vllm/vllm/model_executor/models/llama.py:327`

这就是典型 transformer block：

```text
RMSNorm
  → self attention
  → RMSNorm
  → MLP
  → residual
```

但注意：attention 不是普通 PyTorch attention，它会进入 vLLM 的 `Attention` layer，通过 forward context 获取 KV cache 和 metadata。

---

## 19. Attention layer 如何接入 KV cache 和 metadata

Llama attention 初始化时创建：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.rotary_emb = get_rope(...)
self.attn = Attention(...)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:162` 到 `code/vllm/vllm/model_executor/models/llama.py:219`

Llama attention forward：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
return output
```

位置：`code/vllm/vllm/model_executor/models/llama.py:221` 到 `code/vllm/vllm/model_executor/models/llama.py:231`

`Attention.forward()` 的注释说明：

```text
KV cache 存在 Attention layer 中；
attention metadata 由 ModelRunner.execute_model() 中的 set_forward_context 设置；
Attention.forward() 通过 get_forward_context().attn_metadata 访问。
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:485` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:504`

Attention 初始化时选择 backend：

```python
self.attn_backend = get_attn_backend(
    head_size,
    dtype,
    kv_cache_dtype,
    use_mla=False,
    has_sink=self.has_sink,
    use_mm_prefix=self.use_mm_prefix,
    use_per_head_quant_scales=use_per_head_quant_scales,
    attn_type=attn_type,
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:349` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:358`

并把自己注册到 static forward context：

```python
compilation_config.static_forward_context[prefix] = self
self.attn_type = attn_type
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:440` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:444`

这就是 ModelRunner 可以按 layer name 绑定 KV cache 和 slot mapping 的基础。

---

## 20. Attention.forward 内部怎么用 ForwardContext

`Attention.forward()` 会把 query/key/value reshape，然后调用统一 attention op：

```python
query = query.view(-1, self.num_heads, self.head_size)
output = output.view(-1, self.num_heads, self.head_size_v)
key = key.view(-1, self.num_kv_heads, self.head_size)
value = value.view(-1, self.num_kv_heads, self.head_size_v)
...
unified_attention_with_output(
    query,
    key,
    value,
    output,
    self.layer_name,
    kv_cache_dummy_dep=kv_cache_dummy_dep,
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:530` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:579`

`unified_attention_with_output()` 通过 layer name 取上下文：

```python
attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

self.impl.forward(
    self,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=output,
    ...
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:813` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:840`

`get_attention_context()` 从 `ForwardContext` 取：

```python
forward_context = get_forward_context()
attn_metadata_raw = forward_context.attn_metadata
attn_layer = forward_context.no_compile_layers[layer_name]
kv_cache = attn_layer.kv_cache
slot_mapping = forward_context.slot_mapping
layer_slot_mapping = slot_mapping.get(layer_name)
return attn_metadata, attn_layer, kv_cache, layer_slot_mapping
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:726` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:766`

这条链路说明：

```text
模型 forward 不显式传 attn_metadata / kv_cache；
ModelRunner 通过 set_forward_context 设置全局 forward context；
Attention layer 在执行时按 layer_name 取 metadata、kv_cache、slot_mapping。
```

---

## 21. Attention layer 如何报告 KV cache spec

ModelRunner 初始化 KV cache 前，会从模型 attention layers 查询 KV cache spec。

Attention layer 提供：

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
    block_size = vllm_config.cache_config.block_size
    if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return None
    assert self.attn_type == AttentionType.DECODER
    quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
    if self.sliding_window is not None:
        return SlidingWindowSpec(...)
    else:
        return FullAttentionSpec(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:616` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:688`

因此 attention layer 不只是 forward 计算层，它还参与：

```text
KV cache 规格上报；
KV cache dtype / quant mode；
sliding window spec；
attention backend 选择；
KV scale postprocess。
```

---

## 22. Embedding：VocabParallelEmbedding

`VocabParallelEmbedding` 是 vocab 维度切分的 embedding layer。

类定义：

```python
@PluggableLayer.register("vocab_parallel_embedding")
class VocabParallelEmbedding(PluggableLayer):
    """Embedding parallelized in the vocabulary dimension."""
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:196` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:237`

初始化时会：

```text
计算 TP rank / TP size；
pad vocab size；
把原始 vocab 和 LoRA added vocab 分开 sharding；
选择 quant_method；
创建 shard weight；
设置 weight_loader。
```

关键位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:239` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:325`

forward：

```python
if self.tp_size > 1:
    masked_input, input_mask = get_masked_input_and_mask(...)
else:
    masked_input = input_

output_parallel = self.quant_method.embedding(self, masked_input.long())

if self.tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

output = tensor_model_parallel_all_reduce(output_parallel)
return output
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:472` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:492`

这说明 embedding 层负责处理 TP vocab shard：

```text
每个 rank 只保存 vocab 的一部分；
不属于本 rank 的 token 被 mask；
embedding 后 all-reduce 合并结果。
```

---

## 23. LM Head：ParallelLMHead

`ParallelLMHead` 继承 `VocabParallelEmbedding`：

```python
@PluggableLayer.register("parallel_lm_head")
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head."""
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:503` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:521`

它可以带 bias：

```python
if bias:
    self.bias = Parameter(...)
    set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
else:
    self.register_parameter("bias", None)
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:523` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:553`

它支持 tie weights：

```python
def tie_weights(self, embed_tokens: VocabParallelEmbedding):
    return self.quant_method.tie_weights(self, embed_tokens)
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:555` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:557`

但它的 forward 被禁止：

```python
def forward(self, input_):
    del input_
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:559` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:561`

这点很重要：

```text
LM head 不是在 model.forward() 里直接调用；
它被 model.compute_logits() 通过 LogitsProcessor 使用。
```

---

## 24. LogitsProcessor：hidden states 到 logits

`LogitsProcessor` 定义：

```python
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata."""
```

位置：`code/vllm/vllm/model_executor/layers/logits_processor.py:17` 到 `code/vllm/vllm/model_executor/layers/logits_processor.py:28`

forward：

```python
def forward(self, lm_head, hidden_states, embedding_bias=None):
    if self.logits_as_input:
        logits = hidden_states
    else:
        logits = self._get_logits(hidden_states, lm_head, embedding_bias)
    if logits is not None:
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits *= self.scale
    return logits
```

位置：`code/vllm/vllm/model_executor/layers/logits_processor.py:54` 到 `code/vllm/vllm/model_executor/layers/logits_processor.py:73`

`_get_logits()`：

```python
logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
logits = self._gather_logits(logits)
if logits is not None:
    logits = logits[..., : self.org_vocab_size]
return logits
```

位置：`code/vllm/vllm/model_executor/layers/logits_processor.py:89` 到 `code/vllm/vllm/model_executor/layers/logits_processor.py:104`

TP gather：

```python
if self.use_all_gather:
    logits = tensor_model_parallel_all_gather(logits)
else:
    logits = tensor_model_parallel_gather(logits)
```

位置：`code/vllm/vllm/model_executor/layers/logits_processor.py:75` 到 `code/vllm/vllm/model_executor/layers/logits_processor.py:87`

因此 `compute_logits()` 的典型实现是：

```python
def compute_logits(self, hidden_states):
    return self.logits_processor(self.lm_head, hidden_states)
```

Llama 位置：`code/vllm/vllm/model_executor/models/llama.py:528` 到 `code/vllm/vllm/model_executor/models/llama.py:533`

---

## 25. Linear / MLP / packed modules

Llama MLP 使用两个 vLLM 并行 linear：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size] * 2,
    quant_config=quant_config,
    prefix=f"{prefix}.gate_up_proj",
)
self.down_proj = RowParallelLinear(
    input_size=intermediate_size,
    output_size=hidden_size,
    quant_config=quant_config,
    prefix=f"{prefix}.down_proj",
)
self.act_fn = SiluAndMul()
```

位置：`code/vllm/vllm/model_executor/models/llama.py:79` 到 `code/vllm/vllm/model_executor/models/llama.py:119`

forward：

```python
x, _ = self.gate_up_proj(x)
x = self.act_fn(x)
x, _ = self.down_proj(x)
return x
```

位置：`code/vllm/vllm/model_executor/models/llama.py:115` 到 `code/vllm/vllm/model_executor/models/llama.py:119`

这里的 `MergedColumnParallelLinear` 把 gate/up 两个 HF 参数合并到一个 vLLM 参数里。

对应权重加载需要 packed mapping：

```text
gate_proj → gate_up_proj shard 0
up_proj → gate_up_proj shard 1
q_proj/k_proj/v_proj → qkv_proj q/k/v shard
```

Llama mapping 位置：`code/vllm/vllm/model_executor/models/llama.py:344` 到 `code/vllm/vllm/model_executor/models/llama.py:354`

这解释了为什么模型类的 `load_weights()` 是接口的一部分：HF checkpoint 参数名和 vLLM fused/packed layer 参数名往往不一一对应。

---

## 26. Quantization 如何插入模型层

模型层一般在初始化时把 `quant_config` 传给 vLLM layer：

```python
quant_config = vllm_config.quant_config
...
self.qkv_proj = QKVParallelLinear(..., quant_config=quant_config, ...)
self.o_proj = RowParallelLinear(..., quant_config=quant_config, ...)
self.gate_up_proj = MergedColumnParallelLinear(..., quant_config=quant_config, ...)
self.lm_head = ParallelLMHead(..., quant_config=quant_config, ...)
```

Llama 相关位置：

- `code/vllm/vllm/model_executor/models/llama.py:162` 到 `code/vllm/vllm/model_executor/models/llama.py:180`
- `code/vllm/vllm/model_executor/models/llama.py:92` 到 `code/vllm/vllm/model_executor/models/llama.py:107`
- `code/vllm/vllm/model_executor/models/llama.py:484` 到 `code/vllm/vllm/model_executor/models/llama.py:490`

模型初始化时，loader 还会配置 quant config：

```python
if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:55` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:56`

权重加载后，quant method 会做 postprocess：

```python
quant_method.process_weights_after_loading(module)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:101` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:117`

所以 quantization 的插入点有三处：

```text
1. ModelConfig / VllmConfig 决定 quant_config；
2. 模型 layer 初始化时传入 quant_config；
3. 权重加载后 quant_method 做 repack / scale / kernel format postprocess。
```

---

## 27. LoRA 如何通过模型接口声明能力

LoRA 支持定义在 `interfaces.py`：

```python
class SupportsLoRA(Protocol):
    supports_lora: ClassVar[Literal[True]] = True
    is_3d_moe_weight: ClassVar[bool] = False
    is_non_gated_moe: ClassVar[bool] = False
    embedding_modules: ClassVar[dict[str, str]] = {}
    packed_modules_mapping: dict[str, list[str]] = {}
    lora_skip_prefixes: ClassVar[list[str]] = []
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:544` 到 `code/vllm/vllm/model_executor/models/interfaces.py:564`

判断函数会检查：

```text
是否设置 supports_lora；
是否有 packed_modules_mapping；
是否有 embedding_modules。
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:567` 到 `code/vllm/vllm/model_executor/models/interfaces.py:620`

Llama 声明：

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

位置：`code/vllm/vllm/model_executor/models/llama.py:446` 到 `code/vllm/vllm/model_executor/models/llama.py:464`

这说明 LoRA 不靠 ModelRunner 猜模块名，而是模型类声明：

```text
哪些 fused module 对应哪些原始 HF module；
哪些 embedding module 是输入 / 输出 embedding；
是否跳过某些 prefix。
```

---

## 28. Pipeline Parallel 接口

PP 支持定义在 `SupportsPP`：

```python
class SupportsPP(Protocol):
    supports_pp: ClassVar[Literal[True]] = True

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        ...

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:623` 到 `code/vllm/vllm/model_executor/models/interfaces.py:659`

`supports_pp()` 还会检查 forward 是否接受 `intermediate_tensors`：

```python
return supports_kw(model_forward, "intermediate_tensors")
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:735` 到 `code/vllm/vllm/model_executor/models/interfaces.py:740`

Llama 外层声明 `SupportsPP`：

```python
class LlamaForCausalLM(..., SupportsPP, ...):
```

位置：`code/vllm/vllm/model_executor/models/llama.py:446` 到 `code/vllm/vllm/model_executor/models/llama.py:454`

Llama 主干创建：

```python
self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
    ["hidden_states", "residual"], config.hidden_size
)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:393` 到 `code/vllm/vllm/model_executor/models/llama.py:395`

外层暴露：

```python
self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
```

位置：`code/vllm/vllm/model_executor/models/llama.py:501` 到 `code/vllm/vllm/model_executor/models/llama.py:503`

forward 中非 last rank 返回 `IntermediateTensors`，last rank 返回 hidden states。

位置：`code/vllm/vllm/model_executor/models/llama.py:400` 到 `code/vllm/vllm/model_executor/models/llama.py:439`

---

## 29. 多模态模型接口

多模态支持定义在 `SupportsMultiModal`：

```python
class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True
    supports_multimodal_raw_input_only: ClassVar[bool] = False
    supports_encoder_tp_data: ClassVar[bool] = False
    requires_raw_input_tokens: ClassVar[bool] = False

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        ...

    def get_language_model(self) -> VllmModel:
        ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:100` 到 `code/vllm/vllm/model_executor/models/interfaces.py:219`

多模态模型的 `embed_input_ids()` 支持额外参数：

```python
def embed_input_ids(
    self,
    input_ids: Tensor,
    multimodal_embeddings: MultiModalEmbeddings | None = None,
    *,
    is_multimodal: Tensor | None = None,
) -> Tensor:
    inputs_embeds = self._embed_text_input_ids(...)
    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds
    return _merge_multimodal_embeddings(...)
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:349` 到 `code/vllm/vllm/model_executor/models/interfaces.py:415`

关键含义：

```text
多模态 encoder / processor 产生 multimodal embeddings；
模型通过 embed_input_ids 把文本 embedding 和多模态 embedding 合并；
如果有 out-of-vocab multimodal token，会先 mask input_ids 再做 text embedding。
```

ModelRunner 的 `_preprocess()` 会准备 `inputs_embeds`，最终传给模型 forward：

```text
inputs_embeds → model.forward(..., inputs_embeds=inputs_embeds)
```

---

## 30. Mamba / hybrid / attention-free 能力接口

`interfaces.py` 还定义了一些用于非标准 transformer 的能力标记。

Attention-free：

```python
class IsAttentionFree(Protocol):
    is_attention_free: ClassVar[Literal[True]] = True
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:769` 到 `code/vllm/vllm/model_executor/models/interfaces.py:793`

Hybrid：

```python
class IsHybrid(Protocol):
    is_hybrid: ClassVar[Literal[True]] = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        ...

    @classmethod
    def get_mamba_state_copy_func(cls):
        ...
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:796` 到 `code/vllm/vllm/model_executor/models/interfaces.py:850`

Inner state：

```python
class HasInnerState(Protocol):
    has_inner_state: ClassVar[Literal[True]] = True
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:743` 到 `code/vllm/vllm/model_executor/models/interfaces.py:766`

这些能力会被 registry inspect 进 `_ModelInfo`，再被 `ModelConfig` 和 `ModelRunner` 用于：

```text
选择 attention / mamba cache；
计算 layer block type；
初始化 Mamba backend；
判断是否需要 inner state；
决定 prefix caching 支持方式。
```

---

## 31. MoE 模型接口和 routed experts

MoE 能力接口定义在 `MixtureOfExperts`：

```python
class MixtureOfExperts(Protocol):
    expert_weights: MutableSequence[Sequence[Tensor]]
    num_moe_layers: int
    num_expert_groups: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    num_routed_experts: int
    num_shared_experts: int
    num_redundant_experts: int
    moe_layers: Iterable["MoERunner"]
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:853` 之后

虽然不同 MoE 模型实现差异很大，但接口思想一致：

```text
模型类 / layer 暴露专家权重组织方式；
parallel config / expert parallel 根据这些信息加载本 rank 所需专家；
ModelRunner 可收集 routed experts 信息。
```

默认 loader 中也有 expert parallel weight filter：

```python
self._init_ep_weight_filter(model_config)
loaded_weights = model.load_weights(...)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:414` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:428`

这说明 MoE 不只是 forward 层逻辑，还影响权重加载时“本 rank 应该加载哪些 expert”。

---

## 32. 模型类和 ModelRunner 的职责边界

### ModelRunner 负责

```text
维护 InputBatch；
准备 input_ids / positions / inputs_embeds；
构造 attention metadata；
构造 slot mapping；
设置 ForwardContext；
调用 model.forward；
选择 logits_indices；
调用 model.compute_logits；
调用 model.pooler；
采样 token；
构造 ModelRunnerOutput。
```

### 模型类负责

```text
定义模型层结构；
实现 embed_input_ids；
实现 forward；
实现 compute_logits 或 pooler；
实现 load_weights；
声明 LoRA / PP / multimodal / Mamba / pooling 能力；
使用 vLLM layer 参与 TP、PP、quantization、KV cache 和 attention backend。
```

边界一句话：

```text
ModelRunner 编排一次 batch 怎么跑；模型类定义这一批 token 进模型后每一层怎么计算。
```

---

## 33. 一个完整例子：普通 generation 请求

以 Llama generation 为例：

```text
1. ModelLoader
   → initialize_model()
   → LlamaForCausalLM(vllm_config)
   → LlamaModel + ParallelLMHead + LogitsProcessor
   → model.load_weights()
   → process_weights_after_loading()

2. ModelRunner.execute_model()
   → _prepare_inputs()
   → input_ids / positions / logits_indices
   → _build_attention_metadata()
   → set_forward_context(attn_metadata, slot_mapping, ...)

3. _model_forward()
   → LlamaForCausalLM.forward()
   → LlamaModel.forward()
   → embed_tokens(input_ids)
   → for each LlamaDecoderLayer:
       RMSNorm
       QKVParallelLinear
       rotary_emb
       Attention.forward()
         → ForwardContext
         → KV cache update
         → attention backend
       RowParallelLinear
       MergedColumnParallelLinear / MLP
   → final RMSNorm
   → hidden_states

4. logits
   → hidden_states[logits_indices]
   → LlamaForCausalLM.compute_logits()
   → LogitsProcessor(lm_head, hidden_states)
   → ParallelLMHead.quant_method.apply()
   → TP gather logits

5. sampling
   → ModelRunner 保存 execute_model_state
   → sample_tokens()
   → grammar mask / sampler / logprobs
   → ModelRunnerOutput
```

这条链路里，模型类参与第 1、3、4 步；ModelRunner 参与第 2、5 步，并负责把它们串起来。

---

## 34. 一个完整例子：pooling / embedding 请求

如果 `ModelConfig.runner_type == "pooling"` 且 `convert_type == "embed"`：

```text
1. get_model_architecture()
   → registry.resolve_model_cls()
   → as_embedding_model(model_cls)

2. initialize_model()
   → ModelForEmbedding(vllm_config)
   → 继承原 generation model
   → 跳过 LogitsProcessor / ParallelLMHead
   → 初始化 DispatchPooler.for_embedding(pooler_config)

3. ModelRunner.execute_model()
   → _model_forward()
   → forward 返回 hidden_states

4. ModelRunner._pool()
   → build pooling_metadata
   → model.pooler(hidden_states, pooling_metadata)
   → DispatchPooler 按 task 分发 embed / token_embed
   → ModelRunnerOutput(pooler_output=...)
```

这说明 pooling 路径并不经过：

```text
compute_logits；
sampler；
sample_tokens。
```

它直接从 hidden states 进入 pooler。

---

## 35. 一个完整例子：Pipeline Parallel

PP 下模型类需要支持：

```text
每个 rank 只创建自己负责的层；
first rank 从 input_ids / inputs_embeds 开始；
middle rank 从 intermediate_tensors 开始；
non-last rank 返回 IntermediateTensors；
last rank 返回 hidden_states；
只有 last rank 创建 lm_head / logits_processor。
```

Llama 里的体现：

```text
LlamaModel:
  make_layers() 决定 start_layer / end_layer；
  first rank 有 embed_tokens；
  last rank 有 norm；
  non-last rank 返回 IntermediateTensors。

LlamaForCausalLM:
  last rank 有 ParallelLMHead 和 LogitsProcessor；
  non-last rank 用 PPMissingLayer。
```

相关位置：

- `code/vllm/vllm/model_executor/models/llama.py:365` 到 `code/vllm/vllm/model_executor/models/llama.py:395`
- `code/vllm/vllm/model_executor/models/llama.py:400` 到 `code/vllm/vllm/model_executor/models/llama.py:439`
- `code/vllm/vllm/model_executor/models/llama.py:484` 到 `code/vllm/vllm/model_executor/models/llama.py:503`

ModelRunner 侧如果不是 last PP rank，会返回 `IntermediateTensors` 给下一个 stage；last rank 才进入 logits / pooling。

---

## 36. 容易疑惑的点

### 36.1 model.forward() 为什么不直接返回 logits？

因为 vLLM 需要只对部分位置计算 logits。

```text
forward 返回所有本轮 token 的 hidden states；
ModelRunner 根据 logits_indices 只取需要采样 / logprobs 的位置；
compute_logits 只对这些位置计算 vocab logits。
```

这样可以减少不必要的 LM head 计算。

### 36.2 LM head 为什么 forward 会报错？

`ParallelLMHead.forward()` 明确抛错：

```python
raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:559` 到 `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:561`

它的权重由 `LogitsProcessor` 使用，而不是当作普通 layer forward。

### 36.3 attention metadata 为什么不传进 model.forward 参数？

因为每个 attention layer 都需要自己的 metadata / KV cache / slot mapping，而这些信息由 ModelRunner 构造后放进 `ForwardContext`。

```text
ModelRunner.set_forward_context(...)
  → Attention.forward()
  → get_forward_context()
  → get_attention_context(layer_name)
```

这避免了在每个模型 forward 签名中显式传一大堆 attention backend 内部参数。

### 36.4 load_weights 为什么属于模型类？

因为 HF checkpoint 参数名和 vLLM 模型层参数名经常不同。

典型差异：

```text
HF: q_proj / k_proj / v_proj
vLLM: qkv_proj

HF: gate_proj / up_proj
vLLM: gate_up_proj
```

模型类最清楚自己的 packed module 映射和跳过规则，所以由模型类实现 `load_weights()`。

### 36.5 pooling 模型一定有独立模型文件吗？

不是。

很多 pooling 模型由 `as_embedding_model()` 或 `as_seq_cls_model()` 包装已有 generation 模型类得到。

```text
原模型负责 forward hidden states；
adapter 负责补 pooler 和 pooling 能力声明。
```

### 36.6 多模态模型的 forward 输入是什么？

多数情况下 ModelRunner 会先准备 `inputs_embeds`，模型 forward 接收：

```text
input_ids=None 或 raw input_ids；
inputs_embeds=文本 embedding 和多模态 embedding 合并后的张量；
positions；
model_kwargs。
```

是否还需要 raw input ids 取决于：

```python
requires_raw_input_tokens
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:125` 到 `code/vllm/vllm/model_executor/models/interfaces.py:129`

---

## 37. 写一个 vLLM 模型类时的最小清单

### Generation 模型最小清单

```text
1. __init__(*, vllm_config, prefix="")
2. embed_input_ids(input_ids)
3. forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs)
4. compute_logits(hidden_states)
5. load_weights(weights)
6. 使用 vLLM layer 创建 embedding / attention / MLP / lm_head
7. 如果支持 PP，声明 SupportsPP 并实现 make_empty_intermediate_tensors
8. 如果支持 LoRA，声明 SupportsLoRA 并提供 packed_modules_mapping / embedding_modules
```

### Pooling 模型最小清单

```text
1. 满足基础 VllmModel 接口
2. is_pooling_model=True
3. pooler: Pooler
4. pooler.get_supported_tasks()
5. pooler.forward(hidden_states, pooling_metadata)
6. load_weights(weights)
```

### 多模态模型额外清单

```text
1. supports_multimodal=True
2. embed_multimodal(**kwargs)
3. get_language_model()
4. embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=...)
5. 根据需要声明 requires_raw_input_tokens / supports_encoder_tp_data
```

---

## 38. 从“回答问题”的角度总结

如果要问：

```text
模型层和执行接口如何衔接 ModelRunner？
```

可以回答：

```text
vLLM 模型类通过一组接口和 ModelRunner 衔接。基础模型必须支持 __init__(vllm_config, prefix)、embed_input_ids() 和 forward(input_ids, positions)。generation 模型还必须支持 compute_logits(hidden_states)，pooling 模型则通过 is_pooling_model=True 和 pooler(hidden_states, pooling_metadata) 暴露 embedding / classify / reward 等任务能力。

ModelLoader 先通过 ModelRegistry 和 ModelConfig 解析模型类，然后 initialize_model() 构造模型实例，调用 model.load_weights() 加载 checkpoint，再执行 quantization 和 attention postprocess。执行时，GPUModelRunner 先准备 input_ids、positions、inputs_embeds、attention metadata 和 slot mapping，用 set_forward_context() 设置 ForwardContext，再调用 model.forward()。模型内部的 Attention layer 通过 ForwardContext 获取 KV cache 和 attention metadata。forward 返回 hidden states 后，generation 路径由 ModelRunner 调用 model.compute_logits()，pooling 路径由 ModelRunner 调用 model.pooler()。
```

职责关系可以概括为：

```text
ModelRegistry：解析模型类和能力；
ModelLoader：实例化模型并加载权重；
ModelRunner：准备 batch、上下文和调用模型接口；
Model class：定义层结构、forward、logits/pooler、权重映射和能力声明；
Layer modules：实现 TP/PP/attention/KV cache/quantization/LoRA 等底层执行细节。
```

---

## 39. 最关键流程图

```text
ModelConfig
  → architectures / runner_type / convert_type
  → ModelRegistry.resolve_model_cls()
  → model_cls
      │
      ├─ if convert_type="embed"
      │    └─ as_embedding_model(model_cls)
      │
      ├─ if convert_type="classify"
      │    └─ as_seq_cls_model(model_cls)
      │
      └─ else use original model_cls

BaseModelLoader.load_model()
  ├─ set_default_torch_dtype(model_config.dtype)
  ├─ initialize_model(vllm_config)
  │    ├─ configure_quant_config()
  │    ├─ set_current_vllm_config()
  │    └─ model_cls(vllm_config=vllm_config, prefix=prefix)
  │
  ├─ model.load_weights(weights)
  ├─ finalize_layerwise_processing()       # online quantization
  ├─ process_weights_after_loading()
  │    ├─ quant_method.process_weights_after_loading()
  │    └─ Attention.process_weights_after_loading()
  └─ model.eval()

GPUModelRunner.execute_model()
  ├─ _prepare_inputs()
  │    ├─ input_ids
  │    ├─ positions
  │    ├─ inputs_embeds
  │    └─ logits_indices
  │
  ├─ _build_attention_metadata()
  ├─ _get_slot_mappings()
  ├─ set_forward_context(attn_metadata, slot_mapping, ...)
  │
  ├─ _model_forward()
  │    └─ model.forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
  │         ├─ embed_input_ids() / inputs_embeds
  │         ├─ transformer layers
  │         │    ├─ attention layer
  │         │    │    ├─ get_forward_context()
  │         │    │    ├─ KV cache update
  │         │    │    └─ attention backend forward
  │         │    └─ MLP / MoE / norm
  │         └─ hidden_states or IntermediateTensors
  │
  ├─ if non-last PP rank:
  │    └─ return IntermediateTensors
  │
  ├─ if pooling model:
  │    ├─ build PoolingMetadata
  │    ├─ model.pooler(hidden_states, pooling_metadata)
  │    └─ ModelRunnerOutput(pooler_output)
  │
  └─ if generation model:
       ├─ sample_hidden_states = hidden_states[logits_indices]
       ├─ model.compute_logits(sample_hidden_states)
       │    └─ LogitsProcessor(lm_head, sample_hidden_states)
       ├─ save execute_model_state
       └─ sample_tokens()
            ├─ grammar bitmask
            ├─ sampler
            └─ ModelRunnerOutput(sampled_token_ids)
```

---

## 40. 最关键对象关系

```text
VllmModel
  所有模型的基础协议：__init__、embed_input_ids、forward。

VllmModelForTextGeneration
  generation 协议：compute_logits。

VllmModelForPooling
  pooling 协议：is_pooling_model、pooler、默认 pooling 类型。

SupportsLoRA
  LoRA 能力声明：packed_modules_mapping、embedding_modules。

SupportsPP
  pipeline parallel 能力声明：make_empty_intermediate_tensors、intermediate_tensors forward。

SupportsMultiModal
  多模态能力声明：embed_multimodal、get_language_model、合并 multimodal embeddings。

ModelRegistry
  根据 architecture inspect 模型类，生成能力信息。

BaseModelLoader / DefaultModelLoader
  实例化模型、调用 load_weights、执行 postprocess。

GPUModelRunner
  准备输入和上下文，调用 forward / compute_logits / pooler。

Attention
  vLLM attention layer，按 layer_name 从 ForwardContext 获取 metadata、KV cache、slot mapping。

VocabParallelEmbedding
  vocab 维度 TP 切分的 embedding layer。

ParallelLMHead
  vocab 维度 TP 切分的 LM head 权重层，由 LogitsProcessor 使用。

LogitsProcessor
  hidden states → logits，负责 LM head apply、TP gather、scale、soft cap。

Pooler / DispatchPooler
  hidden states → embedding / classify / token outputs。
```

---

## 41. 和前后专题的关系

本篇回答的是模型层和执行接口如何接上 ModelRunner。

相关专题关系：

```text
03_model_config_and_hf_config.md
  解释 ModelConfig 如何决定 architecture、runner_type、convert_type、dtype、max_model_len。

05_model_registry_and_arch_resolution.md
  继续细拆 ModelRegistry 如何从 architecture 找到模型类和能力。

06_weight_loading_and_quantization.md
  继续细拆 load_weights、weight_loader、quantization config、postprocess。

07_worker_load_model_flow.md
  继续细拆 Worker / ModelRunner 如何调用 loader 完成加载。

executor_worker_model_runner/07_model_forward_and_logits.md
  继续细拆 ModelRunner forward / logits 路径。

executor_worker_model_runner/08_sampling_and_model_runner_output.md
  继续细拆 logits 后的 sampling 和输出。
```

最终最小心智模型：

```text
vLLM 模型类 = vLLM 接口协议 + 模型层结构 + 权重加载映射 + 能力声明。
ModelRunner 不理解每个模型的内部层细节；它只依赖 forward / compute_logits / pooler 这些稳定接口，把 batch、attention metadata、KV cache 和输出处理串起来。
```
