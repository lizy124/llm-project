# 05. ModelRunner 对 model forward 有什么接口约定？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_model_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\interfaces_base.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\interfaces.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\llama.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\adapters.py`
- `E:\lizy\code\vllm-project\vllm\vllm\forward_context.py`

本问题关注：vLLM 里模型架构很多，为什么 `GPUModelRunner` 可以用几乎同一套调用方式执行它们；模型 class 至少要提供哪些方法；`forward()` 的显式参数和隐式上下文分别是什么；generation、pooling、Pipeline Parallel、多模态、LoRA 等能力如何挂到同一个模型对象上。

---

## 1. 一句话回答

`ModelRunner` 对模型的核心约定是：模型对象要像一个 vLLM 执行单元一样，接受统一的 forward 参数，返回 hidden states 或 pipeline 中间结果；如果是生成模型，还要提供 `compute_logits()`；如果是 pooling 模型，还要提供 `pooler`。

主调用形态是：

```python
self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3784` 到 `vllm/v1/worker/gpu_model_runner.py:3790`

可以压缩成：

```text
input_ids / inputs_embeds
positions
intermediate_tensors
**model_kwargs
forward context(attention metadata / slot mapping / graph mode / DP metadata ...)
  → hidden_states / IntermediateTensors / (hidden_states, aux_hidden_states)
  → compute_logits() 或 pooler()
```

也就是说：

```text
forward 只负责把本轮 token 跑成 hidden states；
logits、pooling、sampling、Scheduler 状态更新都在 forward 外层完成。
```

---

## 2. 最基础的 VllmModel 接口

基础接口定义在 `interfaces_base.py`。

```python
class VllmModel(Protocol[T_co]):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...
```

位置：`vllm/model_executor/models/interfaces_base.py:46` 到 `vllm/model_executor/models/interfaces_base.py:56`

因此一个能被 vLLM 识别的模型，最基础要满足：

```text
构造函数接受 vllm_config；
提供 embed_input_ids(input_ids)；
forward 至少能接受 input_ids 和 positions。
```

校验逻辑也在同一个文件：

```text
is_vllm_model(model)
  → _check_vllm_model_init()
  → _check_vllm_model_embed_input_ids()
  → _check_vllm_model_forward()
```

其中 `_check_vllm_model_forward()` 会检查 `forward` 是否支持 `input_ids`、`positions` 这两个 vLLM 关键字。

位置：`vllm/model_executor/models/interfaces_base.py:76` 到 `vllm/model_executor/models/interfaces_base.py:92`

---

## 3. ModelRunner 实际调用的 forward 签名

基础 `VllmModel` 只描述最小约定；真实 serving 路径中，`GPUModelRunner._model_forward()` 调用的是更宽的签名：

```python
def _model_forward(
    self,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs: dict[str, Any],
) -> Any:
```

位置：`vllm/v1/worker/gpu_model_runner.py:3760` 到 `vllm/v1/worker/gpu_model_runner.py:3767`

它只做一件事：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3784` 到 `vllm/v1/worker/gpu_model_runner.py:3790`

因此模型实现最好遵守下面这个实际契约：

```text
forward(
  input_ids: Tensor | None,
  positions: Tensor,
  intermediate_tensors: IntermediateTensors | None = None,
  inputs_embeds: Tensor | None = None,
  **model_kwargs,
) -> Tensor | IntermediateTensors | tuple[Tensor, ...]
```

注意：

```text
基础协议只要求 input_ids / positions；
ModelRunner 实际会额外传 intermediate_tensors、inputs_embeds、model_kwargs；
支持 PP、多模态、encoder-decoder、特定架构参数的模型必须能接住这些参数。
```

---

## 4. Llama 是最典型的实现例子

以 `LlamaForCausalLM` 为例，它的外层 `forward()` 基本只是把参数转给内部 `LlamaModel`：

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

位置：`vllm/model_executor/models/llama.py:550` 到 `vllm/model_executor/models/llama.py:560`

真正处理 PP 输入、embedding、decoder layer、最终 norm 的是 `LlamaModel.forward()`：

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

位置：`vllm/model_executor/models/llama.py:392` 到 `vllm/model_executor/models/llama.py:399`

这就是很多 vLLM 模型的典型分层：

```text
ForCausalLM wrapper：
  负责 lm_head、logits_processor、load_weights、能力标记。

Backbone model：
  负责 embedding、layers、norm、PP 中间张量。
```

---

## 5. input_ids 和 inputs_embeds 的约定

`input_ids` 和 `inputs_embeds` 是二选一或互补关系。

在 Llama 中，首个 PP rank 会这样处理：

```python
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)
    residual = None
```

位置：`vllm/model_executor/models/llama.py:400` 到 `vllm/model_executor/models/llama.py:405`

含义是：

```text
普通文本生成：
  input_ids → embed_input_ids() → hidden_states。

多模态 / prompt embeds / 已预处理 embedding：
  inputs_embeds 直接作为 hidden_states。
```

所以模型必须保证：

```text
当 inputs_embeds 不为空时，不要再强制依赖 input_ids 做 embedding；
当 inputs_embeds 为空时，能通过 embed_input_ids(input_ids) 得到输入 embedding。
```

`embed_input_ids()` 也是基础 `VllmModel` 的一部分。

位置：`vllm/model_executor/models/interfaces_base.py:52` 到 `vllm/model_executor/models/interfaces_base.py:53`

---

## 6. positions 的约定

`positions` 是每个 token 的位置张量，`ModelRunner` 在 forward 前准备好，模型内部通常把它传给 attention / rotary embedding。

在 Llama decoder layer 中：

```python
hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
```

位置：`vllm/model_executor/models/llama.py:325`

因此：

```text
positions 不是模型自己从 input_ids 长度重新推导的；
它由 runner 根据当前 batch、prefill/decode、prefix cache、chunked prefill、多模态位置等信息准备。
```

对普通 decoder-only 模型，`positions` 通常是一维 token position；对 M-RoPE / XD-RoPE / 多模态模型，位置形态可能由模型能力接口进一步定制。

例如 M-RoPE 能力接口要求模型提供：

```python
def get_mrope_input_positions(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]: ...
```

位置：`vllm/model_executor/models/interfaces.py:1459` 到 `vllm/model_executor/models/interfaces.py:1463`

---

## 7. intermediate_tensors 的约定

`intermediate_tensors` 是 Pipeline Parallel 的接口。

`SupportsPP` 协议定义：

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

位置：`vllm/model_executor/models/interfaces.py:617` 到 `vllm/model_executor/models/interfaces.py:653`

检查函数不只看 `supports_pp=True`，还会检查 `forward` 是否真的支持 `intermediate_tensors` 关键字：

```python
return supports_kw(model_forward, "intermediate_tensors")
```

位置：`vllm/model_executor/models/interfaces.py:729` 到 `vllm/model_executor/models/interfaces.py:735`

Llama 的处理方式是：

```text
first PP rank：
  input_ids / inputs_embeds → hidden_states。

非 first PP rank：
  从 intermediate_tensors["hidden_states"] 和 ["residual"] 恢复输入。

非 last PP rank：
  返回 IntermediateTensors。

last PP rank：
  返回最终 hidden_states。
```

对应代码：

```python
else:
    assert intermediate_tensors is not None
    hidden_states = intermediate_tensors["hidden_states"]
    residual = intermediate_tensors["residual"]
```

位置：`vllm/model_executor/models/llama.py:406` 到 `vllm/model_executor/models/llama.py:409`

非最后 rank 返回：

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

位置：`vllm/model_executor/models/llama.py:422` 到 `vllm/model_executor/models/llama.py:425`

这说明：

```text
PP 模型的 forward 返回类型不是固定 Tensor；
在非最后 PP rank，它必须返回 IntermediateTensors，供下一个 pipeline stage 使用。
```

---

## 8. model_kwargs 的约定

`ModelRunner` 会把 `_preprocess()` 产生的 `model_kwargs` 透传给模型：

```python
(
    input_ids,
    inputs_embeds,
    positions,
    intermediate_tensors,
    model_kwargs,
    ec_connector_output,
) = self._preprocess(
    scheduler_output, num_tokens_padded, intermediate_tensors
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4274` 到 `vllm/v1/worker/gpu_model_runner.py:4283`

随后：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4323` 到 `vllm/v1/worker/gpu_model_runner.py:4329`

这些 `model_kwargs` 可能包括：

```text
多模态相关输入；
encoder-decoder 的 encoder outputs / encoder input；
token_type_ids；
模型特定的位置、mask、cache 或 rope 参数；
某些架构透传到 layer 的额外参数。
```

所以模型实现有两种常见写法：

```text
外层 wrapper 明确列出自己支持的额外参数；
内部 backbone / layer 使用 **extra_layer_kwargs 接收并继续透传。
```

Llama 的 backbone 就接受 `**extra_layer_kwargs`，再传给每一层：

```python
hidden_states, residual = layer(
    positions, hidden_states, residual, **extra_layer_kwargs
)
```

位置：`vllm/model_executor/models/llama.py:415` 到 `vllm/model_executor/models/llama.py:417`

---

## 9. attention metadata 不在 forward 参数里

一个容易误解的点是：attention metadata 并不是作为 `self.model(..., attn_metadata=...)` 传进去的。

在 `execute_model()` 中，runner 先构造 attention metadata，然后用 `set_forward_context()` 包住模型 forward：

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
    ),
    ...
):
    model_output = self._model_forward(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4305` 到 `vllm/v1/worker/gpu_model_runner.py:4329`

`ForwardContext` 中保存的信息包括：

```text
attn_metadata；
slot_mapping；
dp_metadata；
cudagraph_runtime_mode；
batch_descriptor；
ubatch_slices；
skip_compiled；
no_compile_layers；
平台相关 additional_kwargs。
```

定义位置：`vllm/forward_context.py:128` 到 `vllm/forward_context.py:180`

这说明：

```text
模型 forward 的显式参数只描述 token / position / embedding / PP 输入；
attention backend、KV cache slot、CUDA graph、DP/MoE 等运行时信息通过 forward context 隐式读取。
```

---

## 10. forward 返回值的约定

`GPUModelRunner` 对 `model_output` 的解释取决于模型类型和 PP rank。

### 10.1 普通 generation 模型

最后 PP rank 上，普通生成模型返回：

```text
hidden_states: torch.Tensor
```

runner 会把它当成 hidden states：

```python
hidden_states = model_output
aux_hidden_states = None
```

位置：`vllm/v1/worker/gpu_model_runner.py:4335` 到 `vllm/v1/worker/gpu_model_runner.py:4338`

### 10.2 EAGLE 3 辅助 hidden states

如果启用辅助 hidden state 输出：

```python
hidden_states, aux_hidden_states = model_output
```

位置：`vllm/v1/worker/gpu_model_runner.py:4332` 到 `vllm/v1/worker/gpu_model_runner.py:4334`

Llama backbone 里也能看到这种返回：

```python
if len(aux_hidden_states) > 0:
    return hidden_states, aux_hidden_states
return hidden_states
```

位置：`vllm/model_executor/models/llama.py:429` 到 `vllm/model_executor/models/llama.py:431`

### 10.3 非最后 PP rank

非最后 PP rank 必须返回 `IntermediateTensors`：

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`vllm/v1/worker/gpu_model_runner.py:4340` 到 `vllm/v1/worker/gpu_model_runner.py:4346`

### 10.4 pooling 模型

pooling 模型的 forward 通常仍返回 hidden states；runner 再调用 `_pool()`：

```python
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

位置：`vllm/v1/worker/gpu_model_runner.py:4348` 到 `vllm/v1/worker/gpu_model_runner.py:4355`

---

## 11. compute_logits 是 generation 模型的额外接口

基础模型只需要 forward；生成模型还必须提供 `compute_logits()`。

协议定义：

```python
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    def compute_logits(
        self,
        hidden_states: T,
    ) -> T | None:
        ...
```

位置：`vllm/model_executor/models/interfaces_base.py:113` 到 `vllm/model_executor/models/interfaces_base.py:122`

Llama 的实现是：

```python
def compute_logits(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

位置：`vllm/model_executor/models/llama.py:562` 到 `vllm/model_executor/models/llama.py:567`

runner 在 forward 之后才调用它：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4357` 到 `vllm/v1/worker/gpu_model_runner.py:4358`

这说明：

```text
generation 模型的 forward 不直接返回 logits；
forward 返回 hidden_states；
runner 根据 logits_indices 选出需要采样/打分的位置；
再调用 compute_logits(hidden_states[logits_indices])。
```

这样做的好处是：

```text
prefill 时不必对所有 token 都算 logits；
chunked prefill 可以跳过不需要输出的位置；
spec decode 可以只对验证位置算 logits；
prompt logprobs 可以按需要额外计算 prompt 位置 logits。
```

---

## 12. pooling 模型的额外接口

pooling 模型通过 `VllmModelForPooling` 协议识别。

关键字段是：

```python
class VllmModelForPooling(VllmModel[T_co], Protocol[T_co]):
    is_pooling_model: ClassVar[Literal[True]] = True
    pooler: Pooler
```

位置：`vllm/model_executor/models/interfaces_base.py:147` 到 `vllm/model_executor/models/interfaces_base.py:212`

`is_pooling_model(model)` 会先确认它是 vLLM 模型，再检查 `is_pooling_model` 标记。

位置：`vllm/model_executor/models/interfaces_base.py:223` 到 `vllm/model_executor/models/interfaces_base.py:229`

很多 embedding / classification 模型不是重新写一整套模型，而是通过 adapter 包装已有 causal LM class：

```python
class ModelForPooling(orig_cls, VllmModelForPooling):
    is_pooling_model = True
    ...
    self.pooler = pooler
```

位置：`vllm/model_executor/models/adapters.py:136` 到 `vllm/model_executor/models/adapters.py:167`

因此 pooling 路径可以记为：

```text
model.forward(...)
  → hidden_states
  → model.pooler(hidden_states, pooling_metadata)
  → ModelRunnerOutput(pooler_output=...)
```

这里的重点是：

```text
pooling 模型仍复用 forward interface；
区别在于 forward 后不调用 compute_logits()，而是调用 pooler。
```

---

## 13. 多模态模型的 forward 约定

多模态能力由 `SupportsMultiModal` 描述。

关键字段和方法包括：

```text
supports_multimodal = True
embed_multimodal(**kwargs)
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=...)
get_language_model()
```

位置：`vllm/model_executor/models/interfaces.py:95` 到 `vllm/model_executor/models/interfaces.py:410`

多模态模型通常遵守这个流程：

```text
多模态 processor / runner 准备 mm kwargs；
模型或 runner 生成 multimodal embeddings；
embed_input_ids() 把 text embeddings 和 multimodal embeddings 合并；
forward 接收 inputs_embeds 或 multimodal kwargs；
语言模型 backbone 继续按普通 hidden_states 路径执行。
```

`SupportsMultiModal.embed_input_ids()` 的默认实现会先拿文本 embedding：

```python
inputs_embeds = self._embed_text_input_ids(
    input_ids,
    self.get_language_model().embed_input_ids,
    is_multimodal=is_multimodal,
)
```

位置：`vllm/model_executor/models/interfaces.py:397` 到 `vllm/model_executor/models/interfaces.py:401`

然后把多模态 embedding scatter 到对应位置：

```python
return _merge_multimodal_embeddings(
    inputs_embeds=inputs_embeds,
    multimodal_embeddings=multimodal_embeddings,
    is_multimodal=_require_is_multimodal(is_multimodal),
)
```

位置：`vllm/model_executor/models/interfaces.py:406` 到 `vllm/model_executor/models/interfaces.py:410`

所以多模态并没有破坏 ModelRunner 的统一调用方式：

```text
差异被吸收到 inputs_embeds / model_kwargs / embed_input_ids 中；
最终进入语言模型时仍是 positions + hidden_states 的执行形态。
```

---

## 14. LoRA 对 forward contract 的影响

LoRA 不是通过改变 `forward()` 主签名接入的，而是通过模型能力标记和模块映射接入。

`SupportsLoRA` 协议定义：

```text
supports_lora = True
embedding_modules
packed_modules_mapping
lora_skip_prefixes
lora_manager
```

位置：`vllm/model_executor/models/interfaces.py:538` 到 `vllm/model_executor/models/interfaces.py:559`

LlamaForCausalLM 声明：

```python
class LlamaForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
```

位置：`vllm/model_executor/models/llama.py:486` 到 `vllm/model_executor/models/llama.py:498`

这说明：

```text
LoRA 改的是模型内部模块和权重加载 / adapter 激活；
ModelRunner 调 forward 的参数形态不需要因为 LoRA 改变。
```

---

## 15. load_weights 也是模型必须配合的接口

虽然本问题重点是 forward，但模型架构要能被 vLLM 加载，还必须提供或继承合适的 `load_weights()`。

LlamaForCausalLM：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
    )
    return loader.load_weights(weights)
```

位置：`vllm/model_executor/models/llama.py:569` 到 `vllm/model_executor/models/llama.py:574`

LlamaModel 内部则展示了更细的权重映射逻辑，例如把 HF 的 `q_proj/k_proj/v_proj` 映射到 vLLM 的 packed `qkv_proj`。

位置：`vllm/model_executor/models/llama.py:433` 到 `vllm/model_executor/models/llama.py:483`

所以完整的模型接入不是只有 forward：

```text
构造模型结构；
load_weights 加载 / 映射权重；
embed_input_ids 支持输入 embedding；
forward 产生 hidden states；
compute_logits 或 pooler 产生任务输出。
```

---

## 16. ModelRunner 不关心模型内部层结构

从 runner 的角度看，它不会关心：

```text
模型有多少层；
attention 是 LlamaAttention、QwenAttention 还是其他实现；
MLP 是否是 MoE；
是否用了 RMSNorm / LayerNorm；
权重是否 packed；
是否支持多模态 tower；
是否开启 LoRA。
```

runner 只关心：

```text
这个对象能不能 self.model(...统一参数...)；
返回值是不是当前阶段需要的类型；
如果是 generation，能不能 compute_logits()；
如果是 pooling，能不能 pooler()；
如果是 PP，能不能处理 intermediate_tensors。
```

这就是 forward interface contract 的价值。

---

## 17. forward 前后完整链路

结合 `execute_model()`，完整关系是：

```text
SchedulerOutput
  → _update_states()
  → _prepare_inputs()
      input_ids / inputs_embeds / positions / model_kwargs
  → _build_attention_metadata()
      attn_metadata / slot_mappings
  → set_forward_context(...)
      attention / KV slot / CUDA graph / DP metadata
  → _model_forward(...)
      self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
  → hidden_states / IntermediateTensors
  → compute_logits() 或 _pool()
  → ExecuteModelState 或 ModelRunnerOutput
```

对应关键位置：

```text
_prepare_inputs / _preprocess：
  vllm/v1/worker/gpu_model_runner.py:4274 到 4283

set_forward_context + _model_forward：
  vllm/v1/worker/gpu_model_runner.py:4305 到 4329

postprocess hidden_states / PP / pooling / logits：
  vllm/v1/worker/gpu_model_runner.py:4331 到 4358
```

---

## 18. 写一个新模型时要对齐哪些接口

如果要新增一个 decoder-only generation 模型，至少要对齐：

```text
__init__(vllm_config, prefix="")；
embed_input_ids(input_ids)；
forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs)；
compute_logits(hidden_states)；
load_weights(weights)。
```

如果支持 Pipeline Parallel，还要：

```text
继承 / 满足 SupportsPP；
设置 supports_pp=True；
提供 make_empty_intermediate_tensors；
forward 支持 intermediate_tensors；
非最后 PP rank 返回 IntermediateTensors。
```

如果支持 pooling，还要：

```text
设置 is_pooling_model=True；
提供 pooler；
forward 返回可供 pooler 使用的 hidden_states。
```

如果支持多模态，还要：

```text
满足 SupportsMultiModal；
实现 embed_multimodal；
正确合并 text / multimodal embeddings；
必要时实现 get_language_model、M-RoPE / XD-RoPE 位置接口。
```

如果支持 LoRA，还要：

```text
满足 SupportsLoRA；
声明 packed_modules_mapping；
声明 embedding_modules；
保证被 LoRA patch 的模块命名和权重映射一致。
```

---

## 19. 容易疑惑的点

### 19.1 forward 要返回 logits 吗？

通常不要。

generation 模型 forward 返回 hidden states，logits 由 `compute_logits(hidden_states[logits_indices])` 单独计算。

### 19.2 attention metadata 为什么不在 forward 参数里？

因为 vLLM 把它放在 `ForwardContext` 里，attention backend 在内部通过当前 forward context 读取。

### 19.3 inputs_embeds 和 input_ids 谁优先？

通常 `inputs_embeds` 优先。只要 `inputs_embeds` 不为空，首个 PP rank 可以直接把它作为 hidden states，不再调用 `embed_input_ids(input_ids)`。

### 19.4 PP 非首 rank 还需要 input_ids 吗？

通常不需要。非首 rank 从 `intermediate_tensors` 取上一 stage 的 hidden states / residual。

### 19.5 pooling 模型是不是单独一套 forward？

不是。pooling 模型通常复用同样的 forward 产出 hidden states，然后额外调用 `pooler`。

### 19.6 LoRA 会改变 forward 签名吗？

一般不会。LoRA 通过模块替换、权重映射、adapter 状态影响模型内部计算，而不是改变 runner 调用模型的参数列表。

---

## 20. 总结

ModelRunner 和模型架构之间的契约可以概括为：

```text
ModelRunner 负责：
  调度 batch、准备 input_ids / positions / inputs_embeds、构造 attention metadata、设置 forward context、调用 forward、处理 logits / pooling / sampling。

Model class 负责：
  接受统一 forward 参数、按自身架构执行 embedding 和 layers、处理 PP intermediate_tensors、返回 hidden_states 或 IntermediateTensors、提供 compute_logits / pooler / load_weights 等任务接口。
```

最终链路是：

```text
self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
  → hidden_states / IntermediateTensors
  → generation: compute_logits(hidden_states[logits_indices])
  → pooling: pooler(hidden_states, pooling_metadata)
```

一句话压缩：

```text
forward interface contract 把各种模型架构统一成“输入 token/embedding + positions + runtime context → hidden states”的执行对象，ModelRunner 只依赖这个契约完成 serving 主链路。
```
