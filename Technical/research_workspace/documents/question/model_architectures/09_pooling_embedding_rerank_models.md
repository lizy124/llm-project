# 09. Pooling / Embedding / Rerank 模型如何接入？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\interfaces_base.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\interfaces.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\pooler\abstract.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\pooler\seqwise\poolers.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\pooler\seqwise\methods.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\pooler\seqwise\heads.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\pooler\tokwise\poolers.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\pooler\tokwise\methods.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\pool\metadata.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_input_batch.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_model_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\engine\core.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\engine\output_processor.py`
- `E:\lizy\code\vllm-project\vllm\vllm\pooling_params.py`
- `E:\lizy\code\vllm-project\vllm\vllm\outputs.py`
- `E:\lizy\code\vllm-project\vllm\vllm\tasks.py`
- `E:\lizy\code\vllm-project\vllm\vllm\entrypoints\pooling\scoring\io_processor.py`
- `E:\lizy\code\vllm-project\vllm\vllm\entrypoints\pooling\scoring\serving.py`

本问题关注：Pooling / Embedding / Classify / Token Embedding / Rerank 这类非自回归生成任务，如何在 vLLM 模型架构层接入 `pooler`，如何复用主模型 `forward` 得到 hidden states，如何通过 `PoolingMetadata` 和 `PoolingParams` 控制池化行为，以及 score / rerank API 如何映射到 `embed`、`classify`、`token_embed` 三类 pooling task。

---

## 0. 梳理规划

本篇按“先定任务类型，再看模型接口，再看 pooler 层，再走执行链路，最后看 score / rerank API”的方式梳理。

要回答的问题分成 12 组：

```text
1. pooling 模型和 generation 模型的根本区别是什么？
2. vLLM 支持哪些 pooling task？
3. 模型类如何声明自己是 pooling model？
4. Pooler 抽象接口是什么？
5. seq-wise pooling 如何做 embedding / classify？
6. token-wise pooling 如何做 token embedding / token classify？
7. PoolingParams 如何表达任务参数？
8. PoolingMetadata / PoolingCursor 如何把 batch hidden states 切回请求？
9. GPUModelRunner 如何在 forward 后走 _pool 而不是 compute_logits / sample_tokens？
10. ModelRunnerOutput.pooler_output 如何变成 PoolingRequestOutput？
11. Score / Rerank API 如何选择 bi-encoder、cross-encoder、late-interaction？
12. 新增一个 pooling / rerank 模型时要实现哪些点？
```

阅读顺序建议：

```text
08_multimodal_models.md
  → 09_pooling_embedding_rerank_models.md
  → 10_model_registry_and_loading.md
```

本篇重点放在“模型架构如何接入 pooler”和“执行层如何分流 pooling 输出”，不会展开每个具体模型文件的全部结构。

---

## 1. 一句话回答

Pooling / embedding / rerank 模型在 vLLM 里不是走：

```text
hidden states → lm_head / compute_logits → sampler → token ids
```

而是走：

```text
hidden states → model.pooler(hidden_states, pooling_metadata) → pooler_output
```

也就是说：

```text
生成模型把 hidden states 转成下一个 token；
pooling 模型把 hidden states 转成向量、分类概率、token 向量或相关性分数。
```

最小主线是：

```text
Pooling API / LLM.embed / LLM.classify / score / rerank
  → PoolingParams(task=...)
  → EngineCoreRequest(pooling_params=...)
  → SchedulerOutput
  → GPUModelRunner._model_forward()
  → GPUModelRunner._pool()
  → model.pooler(hidden_states, PoolingMetadata)
  → ModelRunnerOutput(pooler_output=...)
  → Scheduler.update_from_output()
  → OutputProcessor
  → PoolingRequestOutput / EmbeddingRequestOutput / ClassificationRequestOutput / ScoringRequestOutput
```

一句话压缩：

```text
Pooling 模型复用 vLLM 的调度、KV cache、attention 和 forward 链路，只把 forward 后处理从 logits/sampling 换成 pooler。
```

---

## 2. Pooling task 类型

vLLM 用 `PoolingTask` 表达 pooling 模型支持的任务：

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

位置：`vllm/vllm/tasks.py:8` 到 `vllm/vllm/tasks.py:15`

常见任务含义：

| task | 输出形态 | 典型用途 |
|---|---|---|
| `embed` | `[hidden_size]` 向量 | 文本 / 多模态 embedding，bi-encoder 检索 |
| `classify` | `[num_labels]` 概率或 logits | 分类、cross-encoder 打分、rerank |
| `token_embed` | `[seq_len, hidden_size]` token 向量 | ColBERT / late interaction |
| `token_classify` | `[seq_len, num_labels]` token 分类 | NER、token-level 分类 |
| `plugin` | 插件自定义 | 外部 IO processor / 自定义输出 |

score API 还用 `SCORE_TYPE_MAP` 把 pooling task 映射成 scoring 类型：

```python
SCORE_TYPE_MAP: dict[PoolingTask, ScoreType] = {
    "embed": "bi-encoder",
    "classify": "cross-encoder",
    "token_embed": "late-interaction",
}
```

位置：`vllm/vllm/tasks.py:18` 到 `vllm/vllm/tasks.py:23`

所以：

```text
embed          → bi-encoder score：分别编码 query / doc，再算 cosine similarity；
classify       → cross-encoder score：query-doc 拼成一条输入，模型直接输出分数；
token_embed    → late-interaction score：分别输出 token embeddings，再算 MaxSim。
```

---

## 3. 模型如何声明自己是 pooling model

pooling 模型需要满足 `VllmModelForPooling` 协议。

核心接口是：

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

位置：`vllm/vllm/model_executor/models/interfaces_base.py:148` 到 `vllm/vllm/model_executor/models/interfaces_base.py:212`

这说明 pooling 模型和普通 vLLM 模型一样需要：

```text
__init__(vllm_config, prefix)
embed_input_ids(input_ids)
forward(input_ids, positions, ...)
```

但额外需要：

```text
is_pooling_model = True
pooler: Pooler
```

判断函数是：

```python
def is_pooling_model(model: type[object] | object) -> ...:
    if not is_vllm_model(model):
        return False
    return getattr(model, "is_pooling_model", False)
```

位置：`vllm/vllm/model_executor/models/interfaces_base.py:223` 到 `vllm/vllm/model_executor/models/interfaces_base.py:229`

因此，模型架构接入 pooling 的关键不是实现 `compute_logits()`，而是提供 `pooler`。

---

## 4. Pooler 抽象接口

所有 pooler 都继承 `Pooler`：

```python
class Pooler(nn.Module, ABC):
    """The interface required for all poolers used in pooling models in vLLM."""

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

位置：`vllm/vllm/model_executor/layers/pooler/abstract.py:16` 到 `vllm/vllm/model_executor/layers/pooler/abstract.py:36`

这个接口说明三件事：

```text
1. pooler 自己声明支持哪些 task；
2. pooler 可以修改 PoolingParams，比如要求提供 prompt token ids；
3. pooler 的输入是 hidden states + PoolingMetadata，不是 logits。
```

`get_pooling_updates()` 的返回类型是 `PoolingParamsUpdate`：

```python
@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = False

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids
```

位置：`vllm/vllm/model_executor/layers/pooler/common.py:18` 到 `vllm/vllm/model_executor/layers/pooler/common.py:29`

`PoolingParamsUpdate` 还支持用 `__or__` 合并多个 pooler/method 的更新结果，便于 `SequencePooler` / `TokenPooler` 汇总底层 pooling method 的需求。

典型用途是 `StepPool`：它需要根据 `step_tag_id` 在原始 token ids 里筛选位置，所以会设置 `requires_token_ids=True`。

---

## 5. Pooling 模型和 generation 模型的结构区别

普通生成模型结构是：

```text
input_ids / inputs_embeds
  → backbone / decoder / encoder
  → hidden states
  → lm_head / compute_logits
  → sampler
  → sampled_token_ids
```

pooling 模型结构是：

```text
input_ids / inputs_embeds
  → backbone / encoder / decoder
  → hidden states
  → pooler
      ├─ sequence pooling: CLS / LAST / MEAN
      ├─ token pooling: ALL / STEP
      └─ head: embedding / classifier / token embedding / token classifier
  → pooler_output
```

关键区别：

```text
generation model：需要 VllmModelForTextGeneration.compute_logits()；
pooling model：需要 VllmModelForPooling.pooler。
```

在 `GPUModelRunner.execute_model()` 里，这个差异表现为 forward 后分支：

```python
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )

sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4405` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4415`

所以 pooling 模型不进入 `sample_tokens()`，也不需要 grammar bitmask、temperature、top_p、top_k 等采样逻辑。

---

## 6. SequencePooler：面向整段输入的 pooling

`SequencePooler` 用于把每个请求的一整段 hidden states 聚合成一个向量，再接 embedding 或 classifier head。

定义：

```python
class SequencePooler(Pooler):
    def __init__(
        self,
        pooling: SequencePoolingMethod | SequencePoolingFn,
        head: SequencePoolerHead | SequencePoolingHeadFn,
    ) -> None:
        self.pooling = pooling
        self.head = head

    def forward(self, hidden_states, pooling_metadata):
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/poolers.py:44` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/poolers.py:95`

它分两步：

```text
1. pooling method：从 token hidden states 中抽取或聚合每个 sequence 的表示；
2. head：把 sequence 表示变成 embedding 或 classification 输出。
```

### 6.1 支持的 sequence pooling method

基础抽象：

```python
class SequencePoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify"}
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:21` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:34`

内置方法包括：

| 方法 | 实现类 | 含义 |
|---|---|---|
| `CLS` | `CLSPool` | 取每个 sequence 的第一个 token hidden state |
| `LAST` | `LastPool` | 取每个 sequence 的最后一个 token hidden state |
| `MEAN` | `MeanPool` | 对每个 sequence 的所有 token hidden states 求平均 |

`CLSPool`：

```python
return hidden_states[pooling_cursor.first_token_indices_gpu]
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:37` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:48`

`LastPool`：

```python
return hidden_states[pooling_cursor.last_token_indices_gpu]
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:50` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:57`

`MeanPool` 会用 `prompt_lens` 构造 segment ids，对同一请求的 token hidden states 做 `index_add_` 后除以长度。

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:60` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:106`

注意：

```text
CLS / MEAN 不支持 partial prefill；
如果 chunked prefill 只处理了部分 prompt，CLSPool / MeanPool 会抛错。
```

对应检查：

```python
if pooling_cursor.is_partial_prefill():
    raise RuntimeError("partial prefill is not supported with ... pooling")
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:43` 和 `vllm/vllm/model_executor/layers/pooler/seqwise/methods.py:67`

### 6.2 embedding head

`pooler_for_embed()` 创建 embedding pooler：

```python
head = EmbeddingPoolerHead(
    head_dtype=model_config.head_dtype,
    projector=_load_st_projector(model_config),
    activation=PoolerNormalize(),
)
return SequencePooler(pooling=pooling, head=head)
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/poolers.py:98` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/poolers.py:109`

`EmbeddingPoolerHead` 做：

```text
1. 可选转换 head_dtype；
2. 可选应用 SentenceTransformers projector；
3. 支持 Matryoshka dimensions 截断；
4. 可选 normalize；
5. 返回 embedding tensor。
```

关键代码：

```python
if self.projector is not None:
    embeddings = self.projector(pooled_data)
else:
    embeddings = pooled_data

if any(d is not None for d in dimensions_list):
    embeddings = embeddings[..., :d]

if self.activation is not None:
    embeddings = self.activation(embeddings)
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/heads.py:75` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/heads.py:117`

### 6.3 classify head

`pooler_for_classify()` 创建分类 pooler：

```python
head = ClassifierPoolerHead(
    head_dtype=model_config.head_dtype,
    classifier=classifier,
    logit_mean=model_config.pooler_config.logit_mean,
    logit_sigma=model_config.pooler_config.logit_sigma,
    activation=resolve_classifier_act_fn(...),
)
return SequencePooler(pooling=pooling, head=head)
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/poolers.py:112` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/poolers.py:138`

`ClassifierPoolerHead` 做：

```text
1. 可选 classifier projection；
2. 可选 logit_mean / logit_sigma 校准；
3. 可选 activation，比如 sigmoid / softmax；
4. 返回分类概率或分数向量。
```

关键代码：

```python
if self.classifier is not None:
    logits = self.classifier(pooled_data)
else:
    logits = pooled_data

if self.logit_mean is not None:
    logits = logits - self.logit_mean
if self.logit_sigma is not None:
    logits = logits / self.logit_sigma

if self.activation is not None:
    logits = self.activation(logits)
```

位置：`vllm/vllm/model_executor/layers/pooler/seqwise/heads.py:170` 到 `vllm/vllm/model_executor/layers/pooler/seqwise/heads.py:196`

---

## 7. TokenPooler：面向 token 级输出的 pooling

`TokenPooler` 用于保留每个请求的 token hidden states，或者按 token 位置筛选后再接 token embedding / token classifier head。

定义：

```python
class TokenPooler(Pooler):
    def __init__(
        self,
        pooling: TokenPoolingMethod | TokenPoolingFn,
        head: TokenPoolerHead | TokenPoolingHeadFn | None = None,
    ) -> None:
        self.pooling = pooling
        self.head = head

    def forward(self, hidden_states, pooling_metadata):
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        if self.head is not None:
            pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data
```

位置：`vllm/vllm/model_executor/layers/pooler/tokwise/poolers.py:48` 到 `vllm/vllm/model_executor/layers/pooler/tokwise/poolers.py:98`

### 7.1 ALL pooling

`AllPool` 按每个请求本轮 scheduled token 数，把扁平 hidden states 拆成 list：

```python
hidden_states_lst = list(
    torch.split(hidden_states, pooling_cursor.num_scheduled_tokens_cpu.tolist())
)
```

位置：`vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:47` 到 `vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:58`

如果没有 chunked prefill，直接返回每个请求的 token hidden states：

```text
hidden_states: [sum_tokens, hidden_size]
  → [req0_tokens, req1_tokens, ...]
```

如果开启 chunked prefill，`AllPool` 会把每个 chunk 的 hidden states 暂存在 `PoolingStates.hidden_states_cache`，只有当请求完成 prefill 时才拼接输出：

```python
p.hidden_states_cache.append(hs_chunk)
...
if finished:
    output_list.append(torch.concat(hidden_states_cache, dim=0))
    p.clean()
else:
    output_list.append(None)
```

位置：`vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:63` 到 `vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:83`

这也是 `_pool()` 中允许 `pooler_output` 为 `None` 的原因。

### 7.2 STEP pooling

`StepPool` 继承 `AllPool`，但会根据 token id 筛选位置：

```python
class StepPool(AllPool):
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)
```

位置：`vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:86` 到 `vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:88`

核心逻辑：

```python
if returned_token_ids is not None and len(returned_token_ids) > 0:
    data = data[:, returned_token_ids]

if step_tag_id is not None:
    idx_cpu = (token_id_cpu == step_tag_id).nonzero(as_tuple=True)[0]
    idx = idx_cpu.to(data.device, non_blocking=True)
    data = data[idx]
```

位置：`vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:112` 到 `vllm/vllm/model_executor/layers/pooler/tokwise/methods.py:119`

所以 STEP pooling 适合这类模型：

```text
模型在特定 step tag token 位置输出有意义的 embedding / score，
pooler 只返回这些位置的 hidden states 或指定维度。
```

---

## 8. PoolingParams：请求侧如何控制 pooler

`PoolingParams` 是 pooling 请求的参数对象。

核心字段：

```python
class PoolingParams(msgspec.Struct, omit_defaults=True, array_like=True):
    use_activation: bool | None = None
    dimensions: int | None = None
    step_tag_id: int | None = None
    returned_token_ids: list[int] | None = None

    task: PoolingTask | None = None
    requires_token_ids: bool = False
    skip_reading_prefix_cache: bool | None = None
    late_interaction_params: LateInteractionParams | None = None
    extra_kwargs: dict[str, Any] | None = None
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY
```

位置：`vllm/vllm/pooling_params.py:37` 到 `vllm/vllm/pooling_params.py:70`

字段含义：

| 字段 | 含义 |
|---|---|
| `task` | 当前请求是 `embed` / `classify` / `token_embed` / `token_classify` 等 |
| `use_activation` | 是否应用 pooler head 的 normalize / sigmoid / softmax 等激活 |
| `dimensions` | embedding 输出维度截断，主要用于 Matryoshka embedding |
| `step_tag_id` | STEP pooling 中用于筛选 token 位置的标记 token id |
| `returned_token_ids` | STEP pooling 中只返回指定 token 维度 |
| `requires_token_ids` | pooler 是否需要原始 prompt token ids |
| `skip_reading_prefix_cache` | 是否跳过 prefix cache 读取 |
| `late_interaction_params` | worker 侧 late-interaction scoring 的 query/doc 缓存参数 |
| `extra_kwargs` | 模型或 IO processor 的额外参数，例如压缩后的 token type ids |
| `output_kind` | pooling 强制 `FINAL_ONLY` |

`verify()` 会做几类校验：

```text
1. plugin task 跳过常规参数校验，并默认 skip_reading_prefix_cache=True；
2. 合并 PoolerConfig 默认参数；
3. 给 use_activation 设置默认值；
4. 校验 dimensions 是否符合 Matryoshka 配置；
5. 校验 task 是否允许当前参数；
6. token-level pooling 默认 skip_reading_prefix_cache=True。
```

关键位置：`vllm/vllm/pooling_params.py:89` 到 `vllm/vllm/pooling_params.py:214`

注意 token-level pooling 的 prefix cache 策略：

```python
if self.task in ["token_embed", "token_classify"]:
    self.skip_reading_prefix_cache = True
else:
    self.skip_reading_prefix_cache = False
```

位置：`vllm/vllm/pooling_params.py:124` 到 `vllm/vllm/pooling_params.py:131`

原因是：

```text
token-level 输出要求完整 token hidden states；
如果读取 prefix cache 跳过了部分 prefill token，就无法返回完整 token-level output。
```

---

## 9. PoolingMetadata：pooler 如何知道每个请求的边界

模型 forward 输出的 hidden states 通常是按 batch token 扁平排列的：

```text
hidden_states.shape = [sum(num_scheduled_tokens), hidden_size]
```

pooler 需要知道：

```text
每个请求从哪里开始；
每个请求到哪里结束；
每个请求 prompt 长度；
每个请求本轮 scheduled 了多少 token；
哪些请求已经完成；
是否需要 prompt token ids；
每个请求的 PoolingParams / PoolingStates。
```

这些由 `PoolingMetadata` 和 `PoolingCursor` 表达。

`PoolingCursor`：

```python
@dataclass
class PoolingCursor:
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor
```

位置：`vllm/vllm/v1/pool/metadata.py:13` 到 `vllm/vllm/v1/pool/metadata.py:20`

辅助方法：

```python
def is_partial_prefill(self) -> bool:
    return not torch.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)

def is_finished(self) -> torch.Tensor:
    return self.prompt_lens_cpu == self.seq_lens_cpu
```

位置：`vllm/vllm/v1/pool/metadata.py:30` 到 `vllm/vllm/v1/pool/metadata.py:34`

`PoolingMetadata`：

```python
@dataclass
class PoolingMetadata:
    prompt_lens: torch.Tensor
    prompt_token_ids: torch.Tensor | None
    prompt_token_ids_cpu: torch.Tensor | None
    pooling_params: list[PoolingParams]
    pooling_states: list[PoolingStates]
    pooling_cursor: PoolingCursor | None = None
```

位置：`vllm/vllm/v1/pool/metadata.py:46` 到 `vllm/vllm/v1/pool/metadata.py:55`

初始化时 `PoolingMetadata.__post_init__()` 会从每个 `PoolingParams.task` 派生 `tasks`，并校验每个 pooling request 都已经设置 task。

`build_pooling_cursor()` 根据 `num_scheduled_tokens_np` 和 `query_start_loc_gpu` 构建 token 边界：

```python
self.pooling_cursor = PoolingCursor(
    first_token_indices_gpu=cumsum[:n_seq],
    last_token_indices_gpu=cumsum[1:] - 1,
    prompt_lens_cpu=prompt_lens,
    seq_lens_cpu=seq_lens_cpu,
    num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,
)
```

位置：`vllm/vllm/v1/pool/metadata.py:116` 到 `vllm/vllm/v1/pool/metadata.py:158`

可以这样理解：

```text
PoolingMetadata = pooler 的 batch 说明书；
PoolingCursor = 扁平 hidden states 到每个请求的切片索引。
```

---

## 10. InputBatch 如何维护 pooling 请求状态

`CachedRequestState` 中有 pooling 专用字段：

```python
pooling_params: PoolingParams | None = None
pooling_states: PoolingStates | None = None
```

位置：`vllm/vllm/v1/worker/gpu_input_batch.py:63` 到 `vllm/vllm/v1/worker/gpu_input_batch.py:65`

初始化时，如果请求带 `pooling_params`，就创建 `PoolingStates`：

```python
if self.pooling_params is not None:
    self.pooling_states = PoolingStates()
```

位置：`vllm/vllm/v1/worker/gpu_input_batch.py:72` 到 `vllm/vllm/v1/worker/gpu_input_batch.py:73`

`InputBatch` 维护两个 dict：

```python
self.pooling_params: dict[str, PoolingParams] = {}
self.pooling_states: dict[str, PoolingStates] = {}
```

位置：`vllm/vllm/v1/worker/gpu_input_batch.py:291` 到 `vllm/vllm/v1/worker/gpu_input_batch.py:293`

添加 pooling request 时：

```python
self.pooling_params[req_id] = pooling_params
self.pooling_states[req_id] = pooling_states
self.logits_processing_needs_token_ids[req_index] = (
    pooling_params.requires_token_ids
)
```

位置：`vllm/vllm/v1/worker/gpu_input_batch.py:456` 到 `vllm/vllm/v1/worker/gpu_input_batch.py:464`

构造 `PoolingMetadata` 的入口：

```python
def get_pooling_metadata(self) -> PoolingMetadata:
    pooling_params = self.get_pooling_params()
    pooling_states = self.get_pooling_states()
    prompt_token_ids_cpu = None
    if any(p.requires_token_ids for p in pooling_params):
        prompt_token_ids_cpu = self._make_prompt_token_ids_cpu_tensor()

    return PoolingMetadata(
        prompt_lens=self.num_prompt_tokens_cpu_tensor[: self.num_reqs].clone(),
        prompt_token_ids=self.sampling_metadata.prompt_token_ids,
        prompt_token_ids_cpu=prompt_token_ids_cpu,
        pooling_params=pooling_params,
        pooling_states=pooling_states,
    )
```

位置：`vllm/vllm/v1/worker/gpu_input_batch.py:947` 到 `vllm/vllm/v1/worker/gpu_input_batch.py:960`

这里复用了 `sampling_metadata.prompt_token_ids`，但含义不是采样，而是为了让 pooler 可以拿到 prompt token ids。

---

## 11. GPUModelRunner 如何识别 pooling 模型

初始化时：

```python
self.is_pooling_model = model_config.runner_type == "pooling"
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:480`

创建 `InputBatch` 时会传入：

```python
is_pooling_model=self.is_pooling_model
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:696`

这会影响 `InputBatch` 的行为：

```text
pooling request 不维护采样参数；
remove / condense / refresh_metadata 时跳过 generation 专用采样状态；
get_pooling_metadata() 返回 pooler 所需 metadata。
```

新增请求进入 `_update_states()` 时，如果是 pooling model，会根据模型 pooler 更新请求参数：

```python
if self.is_pooling_model:
    assert pooling_params is not None
    task = pooling_params.task
    assert task is not None, "You did not set `task` in the API"

    model = cast(VllmModelForPooling, self.get_model())
    to_update = model.pooler.get_pooling_updates(task)
    to_update.apply(pooling_params)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:1256` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:1263`

这一步很关键：

```text
API 层只告诉请求 task 和参数；
模型自己的 pooler 可以补充执行层需要的参数，比如 requires_token_ids。
```

---

## 12. _pool：pooling 模型的核心执行分支

`GPUModelRunner._pool()` 是 forward 后处理的主入口。

入口：

```python
def _pool(
    self,
    hidden_states: torch.Tensor,
    num_scheduled_tokens: int,
    num_scheduled_tokens_np: np.ndarray,
    kv_connector_output: KVConnectorOutput | None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3392` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3398`

主流程：

```text
1. 校验 batch 内请求都必须是 pooling request；
2. 截断 padding 后的 hidden_states；
3. 从 InputBatch 构造 PoolingMetadata；
4. build_pooling_cursor() 建立每个请求的 token 边界；
5. 调用 model.pooler(hidden_states, pooling_metadata)；
6. late_interaction_runner 做可选后处理；
7. 构造 ModelRunnerOutput(req_ids, req_id_to_index, pooler_output)；
8. CPU/XPU 同步返回，CUDA 场景用 AsyncGPUPoolingModelRunnerOutput 异步拷贝。
```

关键代码：

```python
hidden_states = hidden_states[:num_scheduled_tokens]
seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]

pooling_metadata = self.input_batch.get_pooling_metadata()
pooling_metadata.build_pooling_cursor(
    num_scheduled_tokens_np,
    seq_lens_cpu,
    device=hidden_states.device,
    query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
)

model = cast(VllmModelForPooling, self.model)
raw_pooler_output: PoolerOutput = model.pooler(
    hidden_states=hidden_states, pooling_metadata=pooling_metadata
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3404` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3418`

然后判断哪些请求已经完成：

```python
finished_mask = [
    seq_len == prompt_len
    for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
]
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3420` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3423`

如果 pooler 还没有完整输出，返回占位：

```python
if raw_pooler_output is None or not any(finished_mask):
    model_runner_output.pooler_output = [None] * num_reqs
    return model_runner_output
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3437` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:3439`

这主要服务于：

```text
chunked prefill + token-level output：前几个 chunk 只能缓存 hidden states，最后一个 chunk 才能返回完整 token output。
```

---

## 13. EngineCore 为什么 pooling 不走 sample_tokens

`EngineCore.step()` 对普通 generation 的流程是：

```text
execute_model()
  → 如果 model_output is None
  → sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:488` 到 `vllm/vllm/v1/engine/core.py:508`

而 pooling 模型在 `GPUModelRunner.execute_model()` 中直接返回 `ModelRunnerOutput`，不会返回 `None` 等待采样。

在 batch queue 模式里，这个分支更明显：

```python
if self.is_pooling_model or not model_executed:
    # No sampling required (no requests scheduled).
    future = cast(Future[ModelRunnerOutput], exec_future)
else:
    ... sample_tokens(...)
```

位置：`vllm/vllm/v1/engine/core.py:555` 到 `vllm/vllm/v1/engine/core.py:568`

所以 pooling 模型的执行链路是：

```text
Scheduler.schedule()
  → model_executor.execute_model()
  → GPUModelRunner.execute_model()
  → _model_forward()
  → _pool()
  → ModelRunnerOutput(pooler_output)
  → Scheduler.update_from_output()
```

没有：

```text
compute_logits
apply_grammar_bitmask
sampler
sampled_token_ids
```

---

## 14. ModelRunnerOutput.pooler_output 到用户输出

`ModelRunnerOutput` 中 pooling 结果字段是：

```python
pooler_output: list[torch.Tensor | None] | None = None
```

位置：`vllm/vllm/v1/outputs.py:259` 到 `vllm/vllm/v1/outputs.py:260`

它不是最终用户输出。

从执行层返回后，Scheduler 会把它转成 `EngineCoreOutput.pooling_output`，然后 `OutputProcessor` 构造 `PoolingRequestOutput`。

`RequestState.make_request_output()` 中：

```python
if pooling_output is not None:
    return self._new_request_output(
        external_req_id,
        [self._new_pooling_output(pooling_output)],
        finished,
    )
```

位置：`vllm/vllm/v1/engine/output_processor.py:313` 到 `vllm/vllm/v1/engine/output_processor.py:318`

`_new_pooling_output()`：

```python
def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
    return PoolingOutput(data=pooling_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:420` 到 `vllm/vllm/v1/engine/output_processor.py:421`

最终 `PoolingRequestOutput` 定义为：

```python
class PoolingRequestOutput(Generic[_O]):
    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
    ):
        ...
```

位置：`vllm/vllm/outputs.py:204` 到 `vllm/vllm/outputs.py:229`

不同 API 再把基础 `PoolingRequestOutput` 转成更具体的输出：

```text
EmbeddingRequestOutput
ClassificationRequestOutput
ScoringRequestOutput
```

对应位置：

```text
EmbeddingOutput / EmbeddingRequestOutput：vllm/vllm/outputs.py:240 到 vllm/vllm/outputs.py:276
ClassificationOutput / ClassificationRequestOutput：vllm/vllm/outputs.py:279 到 vllm/vllm/outputs.py:316
ScoringOutput / ScoringRequestOutput：vllm/vllm/outputs.py:319 到 vllm/vllm/outputs.py:353
```

---

## 15. embedding 输出如何形成

embedding 请求通常走：

```text
PoolingParams(task="embed")
  → pooler_for_embed()
  → SequencePooler(pooling=CLS/LAST/MEAN, head=EmbeddingPoolerHead)
  → PoolingOutput(data=tensor[hidden_size])
  → EmbeddingOutput(embedding=list[float])
```

`EmbeddingOutput.from_base()` 要求输出是一维向量：

```python
pooled_data = pooling_output.data
if pooled_data.ndim != 1:
    raise ValueError("pooled_data should be a 1-D embedding vector")
return EmbeddingOutput(pooled_data.tolist())
```

位置：`vllm/vllm/outputs.py:251` 到 `vllm/vllm/outputs.py:257`

因此 embedding 模型的最终输出约束是：

```text
每个请求一个 1-D tensor。
```

如果启用了 Matryoshka dimensions：

```text
EmbeddingPoolerHead 会先截断维度，再 normalize。
```

---

## 16. classification 输出如何形成

classification 请求通常走：

```text
PoolingParams(task="classify")
  → pooler_for_classify()
  → SequencePooler(pooling=CLS/LAST/MEAN, head=ClassifierPoolerHead)
  → PoolingOutput(data=tensor[num_classes])
  → ClassificationOutput(probs=list[float])
```

`ClassificationOutput.from_base()` 要求输出是一维概率向量：

```python
pooled_data = pooling_output.data
if pooled_data.ndim != 1:
    raise ValueError("pooled_data should be a 1-D probability vector")
return ClassificationOutput(pooled_data.tolist())
```

位置：`vllm/vllm/outputs.py:290` 到 `vllm/vllm/outputs.py:297`

如果它用于 score / rerank，通常要求最终能 squeeze 成标量。

---

## 17. score 输出如何形成

score / rerank 最终使用 `ScoringRequestOutput`。

`ScoringOutput.from_base()` 支持两类输入：

```text
classify task：输出形状是 (num_classes)，且 num_classes == 1；
embed task：输出已经是一个 scalar similarity score。
```

代码：

```python
pooled_data = pooling_output.data.squeeze()
if pooled_data.ndim != 0:
    raise ValueError("pooled_data should be a scalar score")
return ScoringOutput(pooled_data.item())
```

位置：`vllm/vllm/outputs.py:329` 到 `vllm/vllm/outputs.py:338`

所以 score / rerank 的最终输出必须是：

```text
每个 query-doc pair 一个 scalar score。
```

这个 scalar 可以来自：

```text
bi-encoder：API 层对两个 embedding 做 cosine similarity；
cross-encoder：模型 classify head 直接输出一个分数；
late-interaction：API 层或 worker 侧对 token embeddings 做 MaxSim。
```

---

## 18. Score / Rerank API 如何选择执行模式

`ServingScores` 初始化时根据模型支持的 pooling task 决定 IO processor：

```python
pooling_task = engine_client.model_config.get_pooling_task(supported_tasks)
score_type = SCORE_TYPE_MAP.get(pooling_task, None)
self.io_processor_name: str = score_type
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:46` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:50`

如果是 late-interaction 并启用 flash late interaction：

```python
self.enable_flash_late_interaction = (
    self.io_processor_name == "late-interaction"
    and enable_flash_late_interaction
)

if self.enable_flash_late_interaction:
    self.io_processor_name = "flash-late-interaction"
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:51` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:58`

Jina ranking 模型有特殊分支：

```python
if engine_client.model_config.architecture == "JinaForRanking":
    self.io_processor_name = "jina-reranking-scoring"
    self.enable_flash_late_interaction = False
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:60` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:62`

最终可选 IO processors：

```python
ScoringIOProcessors = {
    "bi-encoder": BiEncoderIOProcessor,
    "late-interaction": LateInteractionIOProcessor,
    "jina-reranking-scoring": JinaRankingIOProcessor,
    "flash-late-interaction": FlashLateInteractionIOProcessor,
    "cross-encoder": CrossEncoderIOProcessor,
}
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:793` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:802`

---

## 19. Bi-encoder score / rerank 链路

bi-encoder 对应：

```text
pooling_task = "embed"
score_type = "bi-encoder"
```

预处理时，把 query 和 document 分别作为独立 prompt：

```python
data_1 = score_data_to_prompts(scoring_data.data_1, "query", self.model_config)
data_2 = score_data_to_prompts(scoring_data.data_2, "document", self.model_config)
return self._preprocess_cmpl_offline(prompts=data_1 + data_2, ...)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:246` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:259`

后处理时，把前 `n_queries` 个输出当 query embeddings，后面的当 doc embeddings：

```python
emb_data_1 = outputs[:n_queries]
emb_data_2 = outputs[n_queries:]
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:261` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:264`

然后计算 cosine similarity：

```python
pair_score = F.cosine_similarity(
    emb_1.outputs.data.float(), emb_2.outputs.data.float(), dim=0
)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:270` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:272`

最终重新包装成 `PoolingRequestOutput(outputs=pair_score)`。

所以 bi-encoder rerank 的本质是：

```text
query → embedding
 doc  → embedding
 score = cosine_similarity(query_embedding, doc_embedding)
```

优点：query / doc 可以独立编码、易缓存；缺点：交互弱。

---

## 20. Cross-encoder score / rerank 链路

cross-encoder 对应：

```text
pooling_task = "classify"
score_type = "cross-encoder"
```

预处理时，每个 query-doc pair 被拼成一条输入：

```python
prompt_inputs = tokenizer(text=prompt_1, text_pair=prompt_2, **local_kwargs)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:589` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:591`

如果模型支持 score template，则由模型类提供 prompt 模板：

```python
full_prompt = self.model.get_score_template(prompt_1, prompt_2)
prompt_inputs = tokenizer(full_prompt, **local_kwargs)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:561` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:568`

`SupportsScoreTemplate` 协议定义为：

```python
class SupportsScoreTemplate(Protocol):
    supports_score_template: ClassVar[Literal[True]] = True

    @classmethod
    def get_score_template(cls, query: str, document: str) -> str | None:
        ...

    @classmethod
    def post_process_tokens(cls, prompt: TokensPrompt) -> None:
        ...
```

位置：`vllm/vllm/model_executor/models/interfaces.py:494` 到 `vllm/vllm/model_executor/models/interfaces.py:519`

cross-encoder 的核心是：

```text
(query, document) → single prompt → model forward → classify pooler → scalar score
```

优点：query 和 document 在模型内部充分交互；缺点：每个 query-doc pair 都要完整 forward。

---

## 21. Late-interaction score / rerank 链路

late-interaction 对应：

```text
pooling_task = "token_embed"
score_type = "late-interaction"
```

普通 late-interaction 后处理：

```python
q_emb = emb_1.outputs.data
d_emb = emb_2.outputs.data
maxsim_score = compute_maxsim_score(q_emb, d_emb)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:311` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:318`

典型公式是：

```text
score(q, d) = sum_i max_j sim(q_i, d_j)
```

也就是每个 query token 找最相似的 document token，再求和。

vLLM 还支持 worker 侧 flash late interaction：

```text
stage 1：encode queries and cache token embeddings on workers；
stage 2：encode docs and return scalar scores from workers。
```

入口：

```python
async def flash_late_interaction(self, *args, **kwargs) -> Response:
    ctx = await self._init_ctx(...)
    await self._preprocessing_async(...)
    await self._flash_late_interaction_encode_queries(ctx)
    await self._flash_late_interaction_encode_docs(ctx)
    return await self._postprocessing_async(...)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:191` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:200`

query 阶段会设置：

```python
pooling_params.late_interaction_params = build_late_interaction_query_params(
    query_key=query_keys[i],
    query_uses=query_uses[i],
)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:214` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:223`

doc 阶段会设置：

```python
pooling_params.late_interaction_params = build_late_interaction_doc_params(
    query_key=query_keys[query_idx]
)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:258` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:265`

`LateInteractionParams` 定义为：

```python
class LateInteractionParams(msgspec.Struct, omit_defaults=True, array_like=True):
    mode: str
    query_key: str
    query_uses: int | None = None
```

位置：`vllm/vllm/pooling_params.py:17` 到 `vllm/vllm/pooling_params.py:35`

含义是：

```text
mode="cache_query"：把 query token embeddings 缓存在 worker；
mode="score_doc"：用当前 doc token embeddings 和缓存 query 算分；
query_key：定位同一个 query；
query_uses：query cache 可以被几个 doc 使用。
```

---

## 22. Rerank 响应如何生成

`ServingScores._build_response()` 区分 score 和 rerank：

```python
if isinstance(ctx.request, ScoreRequest):
    return self._request_output_to_score_response(...)
elif isinstance(ctx.request, RerankRequest):
    return self._request_output_to_rerank_response(...)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:83` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:99`

rerank 响应构造时，每个 document 对应一个 score：

```python
classify_res = ScoringRequestOutput.from_base(final_res)
...
result = RerankResult(
    index=idx,
    document=rerank_document,
    relevance_score=classify_res.outputs.score,
)
results.append(result)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:151` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:167`

随后按分数降序排序，并截断 `top_n`：

```python
results.sort(key=lambda x: x.relevance_score, reverse=True)
if top_n < len(documents):
    results = results[:top_n]
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:172` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:174`

因此 rerank 只是 score API 的排序包装：

```text
score(query, doc_i) for each doc_i
  → sort by relevance_score desc
  → return top_n results
```

---

## 23. JinaForRanking 的特殊处理

`JinaForRanking` 在 scoring serving 里单独识别：

```python
if engine_client.model_config.architecture == "JinaForRanking":
    self.io_processor_name = "jina-reranking-scoring"
```

位置：`vllm/vllm/entrypoints/pooling/scoring/serving.py:60` 到 `vllm/vllm/entrypoints/pooling/scoring/serving.py:62`

它的 IO processor 会把多个 docs 放进一个特殊 prompt：

```python
prompt = (
    f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
    f"Rank the passages based on their relevance to query: {query}\n"
)
...
prompt += "\n".join(doc_prompts) + "\n"
prompt += f"<query>\n{query}{query_emb_token}\n</query>"
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:700` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:714`

后处理时：

```python
embeds = outputs[i].outputs.data.float()
query_embeds = embeds[-1]
doc_embeds = embeds[:-1]
scores = F.cosine_similarity(query_embeds, doc_embeds)
```

位置：`vllm/vllm/entrypoints/pooling/scoring/io_processor.py:770` 到 `vllm/vllm/entrypoints/pooling/scoring/io_processor.py:778`

它比较特殊：

```text
同一个 prompt 中包含多个 passage 和 query；
模型输出多个 doc embedding + 一个 query embedding；
后处理用 query embedding 和各 doc embedding 算 cosine similarity。
```

---

## 24. 新增 pooling 模型时要实现什么

如果要给一个模型架构接入 embedding / rerank，通常需要做这些事：

```text
1. 模型类满足 VllmModel 基础接口；
2. 设置 is_pooling_model = True，或继承相关 pooling interface；
3. 定义 pooler: Pooler；
4. 根据任务选择 pooler_for_embed / pooler_for_classify / pooler_for_token_embed / pooler_for_token_classify；
5. 如有特殊 pooling 位置，设置 default_seq_pooling_type / default_tok_pooling_type；
6. 如用于 score API，设置 score_type 或继承 SupportsCrossEncoding / SupportsLateInteraction；
7. 如 cross-encoder 需要特殊 prompt，实现 SupportsScoreTemplate；
8. 如需要 token ids，pooler.get_pooling_updates() 返回 requires_token_ids=True；
9. 确保 forward 返回 hidden states，而不是 logits。
```

最小结构类似：

```text
class XxxEmbeddingModel(...):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"

    def __init__(...):
        self.model = XxxBackbone(...)
        self.pooler = pooler_for_embed(pooler_config)

    def forward(...):
        return self.model(...)
```

如果是 cross-encoder reranker：

```text
class XxxReranker(..., SupportsCrossEncoding):
    is_pooling_model = True
    score_type = "cross-encoder"
    pooler = pooler_for_classify(...)
```

如果是 ColBERT / late-interaction：

```text
class XxxColBERT(..., SupportsLateInteraction):
    is_pooling_model = True
    score_type = "late-interaction"
    pooler = pooler_for_token_embed(...)
```

---

## 25. 和 prefix cache / chunked prefill 的关系

Pooling 模型仍然复用 vLLM 的调度和 KV cache 体系。

但是 token-level pooling 有一个特殊点：

```text
如果请求需要返回每个 token 的输出，就不能只返回未命中 prefix cache 的那部分 hidden states。
```

因此 `PoolingParams._merge_default_parameters()` 中：

```python
if self.task in ["token_embed", "token_classify"]:
    self.skip_reading_prefix_cache = True
else:
    self.skip_reading_prefix_cache = False
```

位置：`vllm/vllm/pooling_params.py:124` 到 `vllm/vllm/pooling_params.py:131`

chunked prefill 下，token-level `AllPool` 会用 `PoolingStates.hidden_states_cache` 跨 chunk 累积 hidden states，直到整个 prompt 完成才输出。

这说明：

```text
pooling 模型不是完全绕开 KV cache / prefix cache；
而是根据输出形态选择是否能安全复用 prefix cache。
```

---

## 26. 容易疑惑的点

### 26.1 pooling model 还会不会执行 attention？

会。

```text
Pooling 模型仍然执行 backbone forward；
如果模型结构有 attention，仍然会构造 attention metadata、使用 KV cache；
区别只在 forward 后不计算 logits / 不采样。
```

### 26.2 embedding 是不是直接取最后一层 hidden state？

不一定。

```text
取哪个位置由 seq_pooling_type 决定：CLS / LAST / MEAN；
取完后还可能经过 projector、dimension 截断和 normalize。
```

### 26.3 rerank 是不是一种独立模型 runner？

不是。

```text
rerank 是 API / IO processor 层的任务形态；
底层仍然是 pooling model runner；
具体可以走 embed、classify 或 token_embed。
```

### 26.4 score 和 classify 有什么关系？

cross-encoder score 通常就是 classify task 的单标量输出。

```text
classify 输出 [num_labels]；
score 要求 squeeze 后是 scalar；
如果 num_labels=1，就能作为相关性分数。
```

### 26.5 token_embed 为什么常常跳过 prefix cache？

因为 token_embed 要返回完整 token 序列的 hidden states。

```text
prefix cache 命中会跳过已缓存 token 的 forward；
这样就拿不到这些 token 的 hidden states；
所以 token_embed / token_classify 默认 skip_reading_prefix_cache=True。
```

### 26.6 PoolingRequestOutput 是最终 OpenAI 响应吗？

不是。

```text
PoolingRequestOutput 是 vLLM 内部统一 pooling 输出；
OpenAI embed / classify / score / rerank serving 层会再转换成各自协议响应。
```

---

## 27. 从“回答问题”的角度总结

如果要问：

```text
Pooling / Embedding / Rerank 模型如何接入 vLLM？
```

可以回答：

```text
它们通过 VllmModelForPooling 协议接入。
模型类仍然实现 vLLM 基础 forward，forward 输出 hidden states，
但不实现 generation 路径里的 compute_logits + sampler，
而是在模型实例上提供 pooler。

pooler 接收 hidden_states 和 PoolingMetadata，
根据 PoolingParams.task 选择 embed、classify、token_embed 或 token_classify 逻辑，
把 hidden states 转成 embedding vector、classification vector、token-level tensor 或 score。

执行层中，GPUModelRunner 在 _model_forward() 后发现 is_pooling_model，
就调用 _pool() 构造 PoolingMetadata、执行 model.pooler，
并把结果放进 ModelRunnerOutput.pooler_output。
EngineCore 对 pooling 模型不调用 sample_tokens，
OutputProcessor 再把 pooler_output 包装成 PoolingRequestOutput，
最后由 embedding/classify/scoring/rerank API 转成协议响应。
```

职责关系可以概括为：

```text
模型架构：提供 backbone forward 和 pooler；
Pooler：定义 hidden states 如何转成任务输出；
PoolingParams：表达请求任务和输出参数；
PoolingMetadata：提供 batch 内请求边界和 token 信息；
GPUModelRunner：执行 forward，并在 pooling 分支调用 pooler；
OutputProcessor：把 pooler_output 包装成 PoolingRequestOutput；
Scoring IOProcessor：把 score/rerank 映射成 embed/classify/token_embed 任务并做后处理。
```

---

## 28. 最关键流程图

```text
用户请求
  ├─ embed / classify / token_embed / token_classify
  └─ score / rerank
       │
       ├─ bi-encoder          → PoolingParams(task="embed")
       ├─ cross-encoder       → PoolingParams(task="classify")
       └─ late-interaction    → PoolingParams(task="token_embed")

EngineCoreRequest(pooling_params)
  → Request
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      │
      ├─ _update_states()
      │    ├─ model.pooler.get_pooling_updates(task)
      │    ├─ CachedRequestState(pooling_params)
      │    └─ InputBatch.pooling_params / pooling_states
      │
      ├─ _prepare_inputs()
      ├─ _build_attention_metadata()
      ├─ _preprocess()
      ├─ _model_forward()
      │    └─ hidden_states
      │
      └─ if is_pooling_model:
           └─ _pool()
                ├─ InputBatch.get_pooling_metadata()
                ├─ PoolingMetadata.build_pooling_cursor()
                ├─ model.pooler(hidden_states, pooling_metadata)
                ├─ late_interaction_runner.postprocess_pooler_output()
                └─ ModelRunnerOutput(pooler_output)

Scheduler.update_from_output()
  → EngineCoreOutput(pooling_output)
  → OutputProcessor
  → PoolingRequestOutput
  → API response
      ├─ EmbeddingRequestOutput
      ├─ ClassificationRequestOutput
      ├─ ScoringRequestOutput
      └─ RerankResponse
```

---

## 29. 最关键对象关系

```text
PoolingTask
  字符串任务类型：embed / classify / token_embed / token_classify。

PoolingParams
  请求级参数：task、dimensions、use_activation、step_tag_id、late_interaction_params。

PoolingParamsUpdate
  pooler 反向告诉执行层需要什么额外输入，例如 requires_token_ids。

VllmModelForPooling
  pooling 模型协议：基础 vLLM forward + pooler。

Pooler
  hidden_states → pooling output 的抽象接口。

SequencePooler
  CLS / LAST / MEAN + embedding/classifier head。

TokenPooler
  ALL / STEP + token embedding/token classifier head。

PoolingMetadata
  pooler 的 batch metadata，包含 prompt_lens、pooling_params、pooling_states、token ids。

PoolingCursor
  把扁平 hidden_states 切回每个请求的 first / last / len 信息。

PoolingStates
  chunked prefill 下缓存 token hidden states。

ModelRunnerOutput.pooler_output
  worker 返回给 scheduler 的 batch 级 pooling tensor 列表。

PoolingRequestOutput
  OutputProcessor 生成的请求级 pooling 输出。
```

---

## 30. 最小心智模型

```text
Pooling / Embedding / Rerank 接入 vLLM 的关键点不是新建一套执行引擎，
而是在现有 ModelRunner forward 链路后，把 logits/sampling 分支替换为 pooler 分支。
```

最终可以记成：

```text
generation:
  hidden states → compute_logits → sampler → token ids

embedding:
  hidden states → sequence pooler → embedding vector

classification / cross-encoder rerank:
  hidden states → sequence pooler → classifier head → scalar / class probs

token embedding / late interaction:
  hidden states → token pooler → token embeddings → MaxSim score
```
