# 11. Pooling / Embedding / Rerank 输出如何区别于生成式输出？

源码位置：

- `code/vllm/vllm/tasks.py`
- `code/vllm/vllm/pooling_params.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/utils.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/pool/metadata.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/entrypoints/pooling/offline.py`
- `code/vllm/vllm/entrypoints/pooling/base/serving.py`
- `code/vllm/vllm/entrypoints/pooling/embed/protocol.py`
- `code/vllm/vllm/entrypoints/pooling/classify/protocol.py`
- `code/vllm/vllm/entrypoints/pooling/scoring/protocol.py`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py`
- `code/vllm/vllm/entrypoints/pooling/classify/serving.py`
- `code/vllm/vllm/entrypoints/pooling/scoring/serving.py`

本问题关注：embedding、pooling、classification、score、rerank、token-level embedding / classification 等非生成任务如何绕开普通 token sampling 路径，并最终返回用户可见输出。

---

## 1. 一句话回答

非生成式任务通常不需要 sampler。

生成式请求在模型 forward 后走：

```text
hidden states
  → logits
  → sampler
  → sampled token ids
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler append token / stop check
  → OutputProcessor detokenize
  → RequestOutput / CompletionOutput
```

pooling / embedding / rerank 请求在模型 forward 后走：

```text
hidden states
  → model.pooler(...)
  → pooler output tensor
  → ModelRunnerOutput.pooler_output
  → Scheduler 看到 pooler_output 后完成请求
  → OutputProcessor 包装成 PoolingRequestOutput
  → entrypoint 再转换成 Embedding / Classification / Scoring / Rerank 响应
```

如果只记住一句话：

```text
pooling / embedding 共享模型执行前半段，但在 hidden states 之后从 generation sampling 主线分叉。
```

---

## 2. 非生成式任务有哪些

### 2.1 Pooling task 类型

vLLM 的 pooling task 定义在 `tasks.py`。

源码位置：

- `code/vllm/vllm/tasks.py:8`
- `code/vllm/vllm/tasks.py:16`

常见任务包括：

```text
embed：
  句向量 / embedding。

classify：
  序列级分类，常用于 cross-encoder 分类或打分。

token_embed：
  token-level embedding。

token_classify：
  token-level classification。

plugin：
  插件式 pooling 任务。

embed&token_classify：
  同时需要 embedding 和 token classification 的复合任务。
```

score / rerank 语义上通常也走 pooling / classify，只是上层 API 进一步解释输出。

源码中还定义了 score task 到架构语义的映射：

```text
embed          → bi-encoder
token_embed    → late-interaction
classify       → cross-encoder
```

源码位置：

- `code/vllm/vllm/tasks.py:18`
- `code/vllm/vllm/tasks.py:23`

### 2.2 PoolingParams 是非生成式请求的参数对象

生成式请求使用 `SamplingParams`。

非生成式请求使用 `PoolingParams`。

`PoolingParams` 定义在：

- `code/vllm/vllm/pooling_params.py:37`

核心字段包括：

```text
use_activation：
  是否应用 pooler activation。

dimensions：
  embedding 输出维度，常用于 Matryoshka embedding。

step_tag_id：
  某些 step/tag 相关模型使用。

returned_token_ids：
  token-level 输出时指定返回哪些 token 位置。

task：
  embed / classify / token_embed / token_classify 等任务名。

requires_token_ids：
  pooler 是否需要 token ids。

skip_reading_prefix_cache：
  是否跳过读取 prefix cache。

late_interaction_params：
  late interaction / rerank 相关参数。

extra_kwargs：
  扩展参数，例如 cross-encoder 的 token_type_ids。

output_kind：
  输出模式。pooling 强制 FINAL_ONLY。
```

源码位置：

- `code/vllm/vllm/pooling_params.py:37`
- `code/vllm/vllm/pooling_params.py:70`

### 2.3 PoolingParams 的任务参数白名单

不同 task 允许的参数不同。

源码位置：

- `code/vllm/vllm/pooling_params.py:76`
- `code/vllm/vllm/pooling_params.py:83`

大致是：

```text
embed：
  dimensions, use_activation

classify：
  use_activation

token_embed：
  dimensions, use_activation

token_classify：
  use_activation
```

`PoolingParams.verify()` 会合并模型 pooler 默认配置、设置默认参数，并校验 task 和参数是否合法。

源码位置：

- `code/vllm/vllm/pooling_params.py:89`
- `code/vllm/vllm/pooling_params.py:107`

### 2.4 token-level task 为什么会跳过 prefix cache

`token_embed` / `token_classify` 默认会设置 `skip_reading_prefix_cache=True`。

源码位置：

- `code/vllm/vllm/pooling_params.py:124`
- `code/vllm/vllm/pooling_params.py:131`

原因是：

```text
token-level 输出要求返回 prompt 内多个 token 位置的 hidden states / logits / embedding。

如果读取 prefix cache，前缀 token 可能不重新 forward，
那么 worker 当前 step 里拿不到这些 token 对应的 hidden states，
输出长度和位置就可能不完整。
```

### 2.5 pooling 强制 FINAL_ONLY

pooling 输出强制使用 `RequestOutputKind.FINAL_ONLY`。

源码位置：

- `code/vllm/vllm/pooling_params.py:230`
- `code/vllm/vllm/pooling_params.py:235`

这和生成式 streaming 不同：

```text
生成式请求：
  可以每生成一个或多个 token streaming 输出。

pooling 请求：
  通常要等整个 prompt 计算完并完成 pooler 后，一次性返回最终 tensor。
```

---

## 3. 上层 API 如何生成 PoolingParams

### 3.1 Embedding API

Embedding 请求会转换成：

```text
PoolingParams(task="embed", dimensions=..., use_activation=...)
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/embed/protocol.py:39`
- `code/vllm/vllm/entrypoints/pooling/embed/protocol.py:44`
- `code/vllm/vllm/entrypoints/pooling/embed/protocol.py:75`
- `code/vllm/vllm/entrypoints/pooling/embed/protocol.py:80`

### 3.2 Classification API

Classification 请求会转换成：

```text
PoolingParams(task="classify", use_activation=...)
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/classify/protocol.py:31`
- `code/vllm/vllm/entrypoints/pooling/classify/protocol.py:35`
- `code/vllm/vllm/entrypoints/pooling/classify/protocol.py:44`
- `code/vllm/vllm/entrypoints/pooling/classify/protocol.py:48`

### 3.3 Score / rerank API

score / rerank 默认走 cross-encoder 分类任务，即：

```text
PoolingParams(task="classify")
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/scoring/protocol.py:77`
- `code/vllm/vllm/entrypoints/pooling/scoring/protocol.py:81`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:445`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:447`

rerank 的输入会被组装成 query-document pair：

```text
RerankRequest.query      → 左侧输入
RerankRequest.documents  → 右侧输入列表
每个 query-doc pair      → 一个 engine prompt / pooling request
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:357`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:367`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:426`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:474`

cross-encoder 的 `token_type_ids` 会压缩到：

```text
PoolingParams.extra_kwargs["compressed_token_type_ids"]
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:463`
- `code/vllm/vllm/entrypoints/pooling/scoring/io_processor.py:467`

worker 侧后续会把它展开成模型 forward 需要的 `token_type_ids`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1024`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1063`

### 3.4 serving 层统一调用 encode

pooling serving 层会统一准备 pooling params，然后调用 engine 的 encode 路径。

源码位置：

- `code/vllm/vllm/entrypoints/pooling/base/serving.py:111`
- `code/vllm/vllm/entrypoints/pooling/base/serving.py:118`
- `code/vllm/vllm/entrypoints/pooling/base/serving.py:139`
- `code/vllm/vllm/entrypoints/pooling/base/serving.py:175`

可以理解为：

```text
embedding / classify / score / rerank API 的差异，
大部分在 entrypoint 层转成不同 task 的 PoolingParams；
进入 EngineCore 后，它们都走 pooling 请求主链路。
```

---

## 4. PoolingParams 如何进入 Request

### 4.1 InputProcessor 校验 pooling params

V1 `InputProcessor._validate_params()` 对 pooling 请求会做：

```text
- 确认模型支持 pooling；
- 自动补默认 task；
- 校验 task 是否在模型支持列表内；
- 调用 params.verify()。
```

源码位置：

- `code/vllm/vllm/v1/engine/input_processor.py:118`
- `code/vllm/vllm/v1/engine/input_processor.py:139`

### 4.2 process_inputs clone PoolingParams

`InputProcessor.process_inputs()` 对非 `SamplingParams` 的参数会 clone：

```text
pooling_params = params.clone()
```

源码位置：

- `code/vllm/vllm/v1/engine/input_processor.py:311`
- `code/vllm/vllm/v1/engine/input_processor.py:330`

随后构造 `EngineCoreRequest`，其中携带：

```text
pooling_params=pooling_params
```

源码位置：

- `code/vllm/vllm/v1/engine/input_processor.py:370`
- `code/vllm/vllm/v1/engine/input_processor.py:385`

`EngineCoreRequest` 同时有：

```text
sampling_params
pooling_params
```

并通过 `params` 属性在无 sampling 时返回 pooling params。

源码位置：

- `code/vllm/vllm/v1/engine/__init__.py:94`
- `code/vllm/vllm/v1/engine/__init__.py:145`

### 4.3 Request 保存 pooling_params

`Request.from_engine_core_request()` 会把 `EngineCoreRequest.pooling_params` 传入运行时 `Request`。

源码位置：

- `code/vllm/vllm/v1/request.py:197`
- `code/vllm/vllm/v1/request.py:222`

`Request.__init__()` 保存：

```text
self.pooling_params
```

源码位置：

- `code/vllm/vllm/v1/request.py:81`
- `code/vllm/vllm/v1/request.py:119`

对 pooling 请求来说：

```text
self.max_tokens = 1
```

它不是说 pooling 真的要生成一个 token，而是让 request 生命周期和调度框架可以用统一字段表达“这个请求最多只需要一个最终输出”。

源码位置：

- `code/vllm/vllm/v1/request.py:107`
- `code/vllm/vllm/v1/request.py:119`

`Request.skip_reading_prefix_cache` 也会读取 `PoolingParams.skip_reading_prefix_cache`。

源码位置：

- `code/vllm/vllm/v1/request.py:266`
- `code/vllm/vllm/v1/request.py:277`

---

## 5. Scheduler 如何处理 pooling 请求

### 5.1 pooling 请求仍然走 Scheduler 主链路

Scheduler 并不会为 pooling 单独开一套完全不同的执行器。

它仍然会：

```text
- 放入 waiting / running 队列；
- 查询 prefix cache；
- 计算本轮 num_new_tokens；
- 分配 KV cache slots；
- 构造 SchedulerOutput；
- 下发给 Worker / ModelRunner。
```

V1 Scheduler 的基本目标是让：

```text
request.num_computed_tokens 追上 request.num_tokens_with_spec
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:388`
- `code/vllm/vllm/v1/core/sched/scheduler.py:399`

### 5.2 pooling 请求仍会查询 prefix cache

WAITING 请求，包括 pooling 请求，仍然会查 prefix cache。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:672`
- `code/vllm/vllm/v1/core/sched/scheduler.py:712`

但如果 `PoolingParams.skip_reading_prefix_cache=True`，则 KVCacheManager 会跳过读取 prefix cache。

源码位置：

- `code/vllm/vllm/v1/core/kv_cache_manager.py:214`
- `code/vllm/vllm/v1/core/kv_cache_manager.py:219`

这对 token-level pooling 很重要，因为 token-level 输出通常要求完整 hidden states。

### 5.3 pooling 请求是否支持 chunked prefill

Scheduler 计算本轮要调度的新 token 数：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:792`
- `code/vllm/vllm/v1/core/sched/scheduler.py:812`

源码注释说明：pooling 请求如果要 chunked prefill，必须显式启用 `enable_chunked_prefill`，否则 token budget 不够时会停止调度。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:801`
- `code/vllm/vllm/v1/core/sched/scheduler.py:809`

原因是：

```text
pooling 输出通常要等整个 prompt 的 hidden states / pooling 结果完整后才能返回。

如果没有启用 chunked prefill，Scheduler 不应该把一个无法完整计算的 pooling prompt 切成半截执行。
```

### 5.4 pooling 请求仍分配 KV cache slots

pooling 请求仍然会分配 KV cache slots。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:874`
- `code/vllm/vllm/v1/core/sched/scheduler.py:884`

调度成功后，同样进入 `RUNNING`，并更新 `num_computed_tokens`。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:940`
- `code/vllm/vllm/v1/core/sched/scheduler.py:963`

这说明：

```text
pooling 不是绕过 KV cache / attention 的轻量旁路。

它仍然通过正常 transformer forward 得到 hidden states，
只是 hidden states 后面接的是 pooler，不是 lm_head + sampler。
```

### 5.5 NewRequestData 下发 pooling_params

新请求下发给 worker 时，`NewRequestData` 会携带：

```text
pooling_params
block_ids
prompt_token_ids
```

源码位置：

- `code/vllm/vllm/v1/core/sched/output.py:31`
- `code/vllm/vllm/v1/core/sched/output.py:64`

---

## 6. Worker / InputBatch 如何保存 pooling 状态

### 6.1 GPUModelRunner 判断是否 pooling model

`GPUModelRunner` 通过：

```text
model_config.runner_type == "pooling"
```

判断当前模型是否是 pooling runner。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:441`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:453`

### 6.2 新请求进入 worker 时应用 pooling params

新请求进入 worker 时，如果是 pooling model：

```text
1. 要求 pooling_params 不为空；
2. 要求 pooling_params.task 已设置；
3. 通过 model.pooler.get_pooling_updates(task).apply(pooling_params)
   把 task 相关默认值 / pooler 更新应用到 params。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1203`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1222`

这一步对应生成式请求里的 sampling params 缓存，但 pooling 还要让模型 pooler 根据 task 做参数更新。

### 6.3 InputBatch 保存 pooling params 和 pooling states

`InputBatch` 会保存每个请求的 pooling params 和 pooling states。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:937`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:943`

这些状态用于后续构造 `PoolingMetadata`。

---

## 7. PoolingMetadata 是什么

### 7.1 构造 PoolingMetadata

`InputBatch` 构造 `PoolingMetadata` 时会收集：

```text
- 当前 batch 的 pooling params；
- 当前 batch 的 pooling states；
- prompt lens；
- 如果任一 params 需要 token ids，则额外构造 CPU prompt token ids tensor。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:945`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:958`

`PoolingMetadata` 要求每个 pooling param 都有 task。

源码位置：

- `code/vllm/vllm/v1/pool/metadata.py:57`
- `code/vllm/vllm/v1/pool/metadata.py:72`

### 7.2 PoolingMetadata 解决什么问题

pooler 需要知道：

```text
- 每个请求的 prompt 长度；
- 当前 step 里哪些 hidden states 属于哪个请求；
- 每个请求应该取 first token、last token、所有 token，还是特定 token span；
- 每个请求的 task 和 pooling 参数；
- token-level 输出是否需要 prompt token ids。
```

这不是 `SamplingMetadata` 能表达的内容，所以 pooling 有独立的 `PoolingMetadata`。

### 7.3 Pooling cursor

`PoolingMetadata.build_pooling_cursor()` 会根据：

```text
num_scheduled_tokens
seq_lens
query_start_loc
```

构造每个请求在当前 hidden states 中的 first / last token 索引。

源码位置：

- `code/vllm/vllm/v1/pool/metadata.py:116`
- `code/vllm/vllm/v1/pool/metadata.py:158`

它的作用可以理解为：

```text
告诉 pooler：
  这个 batch 中每个请求的 pooling 位置在哪里。
```

---

## 8. hidden states 后如何分叉：_pool vs compute_logits/sample

### 8.1 forward 前半段仍然一样

pooling 请求在 forward 前仍然会准备：

```text
- input ids / positions；
- slot mapping；
- attention metadata；
- forward context；
- multimodal / encoder / token_type_ids 等 model kwargs。
```

GPUModelRunner 仍会构造 slot mapping 和 attention metadata。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4247`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4256`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4258`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4271`

随后仍在 `set_forward_context(attn_metadata, ...)` 下执行模型 forward。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4300`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4329`

### 8.2 forward 后拿到 hidden states

模型 forward 后得到 `hidden_states`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4331`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4338`

这里是 generation 和 pooling 的真正分叉点。

### 8.3 pooling model 直接走 _pool

如果是 pooling model：

```text
return self._pool(...)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4348`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

生成式模型才会继续：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4357`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4358`

因此最关键的分叉是：

```text
_model_forward()
  → hidden_states
      → pooling model：_pool(hidden_states)
      → generation model：compute_logits(hidden_states[logits_indices]) → sampler
```

---

## 9. _pool 如何生成 ModelRunnerOutput.pooler_output

### 9.1 PoolerOutput 类型

worker 内部的 `PoolerOutput` 类型定义为：

```text
torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]
```

源码位置：

- `code/vllm/vllm/v1/outputs.py:180`
- `code/vllm/vllm/v1/outputs.py:182`

`ModelRunnerOutput.pooler_output` 是按请求展开后的：

```text
list[torch.Tensor | None] | None
```

源码位置：

- `code/vllm/vllm/v1/outputs.py:231`
- `code/vllm/vllm/v1/outputs.py:260`

### 9.2 _pool 主流程

`_pool()` 中会：

```text
1. 截取本 step 的 hidden states；
2. 构造 PoolingMetadata；
3. 构造 pooling cursor；
4. 调用 model.pooler(hidden_states, pooling_metadata)；
5. 判断哪些请求已经完成；
6. 对 late-interaction 等场景做 postprocess；
7. 只返回已完成请求的 pooler output，未完成请求返回 None；
8. 将输出拷到 CPU，CUDA-like 平台可走异步 D2H。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3358`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3361`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3367`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3369`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3372`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3374`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3377`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3378`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3383`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3391`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3393`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3395`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3402`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3404`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3409`

### 9.3 为什么 pooler_output 里会有 None

`_pool()` 会用：

```text
seq_len == prompt_len
```

判断请求是否已经完成 pooling。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3374`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3377`

如果当前 step 只是 chunked prefill 的中间段，请求还没有完整 prompt hidden states，则这个请求的 pooler output 是 `None`。

所以：

```text
pooler_output is None：
  当前请求还没完成，不应该返回最终 pooling 输出。

pooler_output is not None：
  当前请求已经完整计算，可以被 Scheduler 标记 finished。
```

### 9.4 异步拷贝到 CPU

CUDA-like 平台会返回 `AsyncGPUPoolingModelRunnerOutput`，异步把 pooler output 从 GPU 拷到 CPU。

只会拷贝 `finished_mask=True` 的输出，未完成项保持 `None`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:322`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:364`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3404`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3409`

---

## 10. Scheduler.update_from_output 如何处理 pooler_output

### 10.1 从 ModelRunnerOutput 取 pooler_output

`Scheduler.update_from_output()` 会从 `ModelRunnerOutput` 里取：

```text
pooler_outputs = model_runner_output.pooler_output
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1468`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1474`

每个 request 通过 `req_id_to_index` 找到自己在 batch 中的位置。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1543`

### 10.2 生成式和 pooling 的回收分支

生成式输出从：

```text
sampled_token_ids
```

取新 token。

pooling 输出从：

```text
pooler_outputs[req_index]
```

取 tensor。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1543`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1584`

如果有 `new_token_ids`，走生成式更新：

```text
_update_request_with_output()
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1589`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1593`

如果没有新 token，但请求是 pooling 且 `pooler_output is not None`，则直接完成请求：

```text
FINISHED_STOPPED
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1594`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1597`

这说明 pooling 的完成条件不是 EOS / stop token，而是：

```text
pooler_output 已经产出。
```

### 10.3 EngineCoreOutput 携带 pooling_output

Scheduler 构造 `EngineCoreOutput` 时会把 pooling tensor 放入：

```text
pooling_output
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1683`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1698`

`EngineCoreOutput` 同时有生成式字段和 pooling 字段：

```text
new_token_ids
pooling_output
```

源码位置：

- `code/vllm/vllm/v1/engine/__init__.py:175`
- `code/vllm/vllm/v1/engine/__init__.py:188`

### 10.4 stop check 不处理 pooling

生成式 stop 检查函数 `check_stop()` 明确断言不是 pooling 请求。

源码位置：

- `code/vllm/vllm/v1/core/sched/utils.py:94`
- `code/vllm/vllm/v1/core/sched/utils.py:97`

这也说明：

```text
pooling 请求不靠 stop token / EOS / max_tokens 结束，
而靠 pooler_output 是否产生来结束。
```

---

## 11. OutputProcessor 如何返回 PoolingRequestOutput

### 11.1 RequestState 对 pooling 请求不创建 detokenizer

对 sampling 请求，`RequestState.from_new_request()` 会创建 detokenizer 和 logprobs processor。

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:222`
- `code/vllm/vllm/v1/engine/output_processor.py:237`

对 pooling 请求：

```text
logprobs_processor = None
detokenizer = None
output_kind = request.pooling_params.output_kind
```

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:238`
- `code/vllm/vllm/v1/engine/output_processor.py:247`

原因很直接：

```text
pooling 输出是 tensor，不是 token id 序列；
不需要 detokenize，也不需要 generation logprobs。
```

### 11.2 process_outputs 跳过 detokenize / logprobs

`process_outputs()` 会从 `EngineCoreOutput` 中取：

```text
pooling_output
```

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:618`
- `code/vllm/vllm/v1/engine/output_processor.py:620`

只有当 `pooling_output is None` 时，才走生成式的 detokenize 和 logprobs 更新。

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:635`
- `code/vllm/vllm/v1/engine/output_processor.py:649`

如果 `pooling_output is not None`，后面会构造 pooling request output。

### 11.3 make_request_output 构造 PoolingRequestOutput

`make_request_output()` 中，如果 `pooling_output is not None`，会创建：

```text
PoolingOutput(data=pooling_output)
PoolingRequestOutput(...)
```

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:312`
- `code/vllm/vllm/v1/engine/output_processor.py:317`
- `code/vllm/vllm/v1/engine/output_processor.py:333`
- `code/vllm/vllm/v1/engine/output_processor.py:355`

对外的 `PoolingOutput` 只是包装一个 tensor：

```text
data: torch.Tensor
```

源码位置：

- `code/vllm/vllm/outputs.py:66`
- `code/vllm/vllm/outputs.py:82`

`PoolingRequestOutput` 定义在：

- `code/vllm/vllm/outputs.py:204`
- `code/vllm/vllm/outputs.py:228`

它包含：

```text
request_id
outputs
num_cached_tokens
prompt_token_ids
finished
```

---

## 12. embedding / classification / score / rerank 最终返回形态

### 12.1 LLM.encode 的基础返回

离线 `LLM.encode()` 的基础返回类型是：

```text
list[PoolingRequestOutput]
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/offline.py:120`
- `code/vllm/vllm/entrypoints/pooling/offline.py:136`

它是所有 pooling 任务的底座。

### 12.2 EmbeddingRequestOutput

`LLM.embed()` 会调用：

```text
encode(..., pooling_task="embed")
```

再把 `PoolingRequestOutput` 转成 `EmbeddingRequestOutput`。

源码位置：

- `code/vllm/vllm/entrypoints/pooling/offline.py:199`
- `code/vllm/vllm/entrypoints/pooling/offline.py:242`

`EmbeddingOutput.from_base()` 要求 pooling tensor 是 1-D，并转成：

```text
list[float]
```

源码位置：

- `code/vllm/vllm/outputs.py:240`
- `code/vllm/vllm/outputs.py:257`

### 12.3 ClassificationRequestOutput

`LLM.classify()` 会调用：

```text
encode(..., pooling_task="classify")
```

再转换成 `ClassificationRequestOutput`。

源码位置：

- `code/vllm/vllm/entrypoints/pooling/offline.py:244`
- `code/vllm/vllm/entrypoints/pooling/offline.py:287`

`ClassificationOutput.from_base()` 要求 pooling tensor 是 1-D probability / logit vector，并转成：

```text
list[float]
```

源码位置：

- `code/vllm/vllm/outputs.py:279`
- `code/vllm/vllm/outputs.py:297`

HTTP classification serving 会：

```text
1. 从 final_res.outputs 转 ClassificationOutput；
2. 取 np.argmax(probs) 作为 predicted index；
3. 返回 label / probs / num_classes。
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/classify/serving.py:36`
- `code/vllm/vllm/entrypoints/pooling/classify/serving.py:72`

### 12.4 ScoringRequestOutput

`ScoringOutput.from_base()` 会对 pooling tensor 做 `squeeze()`，要求最终是标量，然后返回 Python `float`。

源码位置：

- `code/vllm/vllm/outputs.py:319`
- `code/vllm/vllm/outputs.py:338`

`ScoringRequestOutput.from_base()` 包装 score 输出。

源码位置：

- `code/vllm/vllm/outputs.py:344`
- `code/vllm/vllm/outputs.py:353`

HTTP score response 会将每个 `PoolingRequestOutput` 转成 `ScoringRequestOutput`，并写入：

```text
ScoreResponseData.score
```

源码位置：

- `code/vllm/vllm/entrypoints/pooling/scoring/serving.py:101`
- `code/vllm/vllm/entrypoints/pooling/scoring/serving.py:136`

### 12.5 RerankResponse

rerank response 复用 scoring 的 pooling 结果：

```text
每个 document 对应一个 relevance_score。
```

最终会按 `relevance_score` 降序排序，并按 `top_n` 截断。

源码位置：

- `code/vllm/vllm/entrypoints/pooling/scoring/serving.py:138`
- `code/vllm/vllm/entrypoints/pooling/scoring/serving.py:184`

所以 rerank 不是一个完全独立的底层执行链路，而是：

```text
query-doc pairs
  → 多个 pooling/classify requests
  → score tensors
  → relevance_score 排序
  → rerank response
```

### 12.6 token-level 输出

`token_embed` / `token_classify` 在参数层受支持。

源码位置：

- `code/vllm/vllm/pooling_params.py:79`
- `code/vllm/vllm/pooling_params.py:83`

这类输出通常保留为基础 `PoolingRequestOutput.outputs.data` 的 token 维 tensor，不像 embedding / classification / scoring 那样总是转换成单个向量或标量。

---

## 13. 与生成式 sampling 输出链路的核心差异

### 13.1 Worker 输出字段不同

生成式：

```text
ModelRunnerOutput.sampled_token_ids
ModelRunnerOutput.logprobs
ModelRunnerOutput.prompt_logprobs_dict
```

pooling：

```text
ModelRunnerOutput.pooler_output
```

源码位置：

- `code/vllm/vllm/v1/outputs.py:240`
- `code/vllm/vllm/v1/outputs.py:260`

### 13.2 GPUModelRunner 分叉不同

生成式：

```text
hidden_states[logits_indices]
  → compute_logits()
  → sample_tokens()
```

pooling：

```text
hidden_states
  → _pool()
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4348`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4358`

### 13.3 Scheduler 停止条件不同

生成式：

```text
sampled token
  → _update_request_with_output()
  → stop / eos / max_tokens / stop string 等逻辑
```

pooling：

```text
pooler_output is not None
  → FINISHED_STOPPED
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1589`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1597`

### 13.4 OutputProcessor 不同

生成式：

```text
token ids
  → detokenize
  → logprobs / prompt logprobs 格式化
  → RequestOutput / CompletionOutput
```

pooling：

```text
pooling_output tensor
  → PoolingOutput
  → PoolingRequestOutput
```

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:635`
- `code/vllm/vllm/v1/engine/output_processor.py:649`
- `code/vllm/vllm/v1/engine/output_processor.py:312`
- `code/vllm/vllm/v1/engine/output_processor.py:317`

### 13.5 输出模式不同

生成式可以 streaming / delta 输出。

pooling 强制 `FINAL_ONLY`。

源码位置：

- `code/vllm/vllm/pooling_params.py:230`
- `code/vllm/vllm/pooling_params.py:235`

---

## 14. KV cache / attention 是否仍参与

结论：普通 decoder / transformer pooling 模型仍然会走 KV cache、prefix cache、attention metadata 和模型 forward。

它不是跳过 transformer，只是不进入 generation sampler。

### 14.1 Scheduler 仍查 prefix cache

Scheduler 仍会查询 prefix cache。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:672`
- `code/vllm/vllm/v1/core/sched/scheduler.py:712`

KVCacheManager 在关闭 prefix caching 或 request 标记 `skip_reading_prefix_cache` 时才不查。

源码位置：

- `code/vllm/vllm/v1/core/kv_cache_manager.py:214`
- `code/vllm/vllm/v1/core/kv_cache_manager.py:219`

### 14.2 即使 prefix cache 命中，也可能重算最后 token

KVCacheManager 注释说明，即使全部命中 prefix cache，也会重算最后 token。

源码位置：

- `code/vllm/vllm/v1/core/kv_cache_manager.py:221`
- `code/vllm/vllm/v1/core/kv_cache_manager.py:230`

原注释主要从 generation logits 的角度解释，但 pooling 场景同样沿用这套机制。

### 14.3 pooling 请求仍分配 KV slots

pooling 请求仍分配 KV cache slots。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:874`
- `code/vllm/vllm/v1/core/sched/scheduler.py:884`

KV slots 分配逻辑会综合：

```text
- 已计算 token；
- 本地 prefix cache 命中 token；
- external KV connector 命中 token；
- 新计算 token；
- lookahead token。
```

源码位置：

- `code/vllm/vllm/v1/core/kv_cache_manager.py:244`
- `code/vllm/vllm/v1/core/kv_cache_manager.py:339`

### 14.4 ModelRunner 仍构造 attention metadata

GPUModelRunner 仍然构造 attention slot mappings 和 attention metadata。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4247`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4256`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4258`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4271`

也就是说：

```text
pooling 请求的前半段仍是正常模型 forward；
区别在于 forward 后从 hidden states 中做 pooling，而不是从 logits 中采样 token。
```

---

## 15. 整体流程图

### 15.1 embedding / pooling 主链路

```text
用户调用 embed / classify / score / rerank
  → entrypoint 构造 PoolingParams(task=...)
  → InputProcessor._validate_params()
      → 校验模型支持 pooling
      → params.verify()
  → EngineCoreRequest.pooling_params
  → Request.pooling_params
  → Scheduler.schedule()
      → prefix cache lookup
      → KV slot allocation
      → SchedulerOutput.scheduled_new_reqs
  → GPUModelRunner._update_states()
      → 应用 model.pooler task updates
      → InputBatch 保存 pooling params / states
  → GPUModelRunner._prepare_inputs()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → set_forward_context(...)
  → _model_forward()
      → hidden_states
  → _pool(hidden_states)
      → PoolingMetadata
      → model.pooler(...)
      → pooler_output 或 None
  → ModelRunnerOutput.pooler_output
  → Scheduler.update_from_output()
      → pooler_output is not None 时 finish request
      → EngineCoreOutput.pooling_output
  → OutputProcessor
      → PoolingOutput
      → PoolingRequestOutput
  → entrypoint 转成 Embedding / Classification / Scoring / Rerank response
```

### 15.2 和 generation 对比

```text
共同前半段：

Scheduler
  → Worker / ModelRunner
  → input_ids / positions / attention metadata
  → model forward
  → hidden_states

分叉：

生成式：
  hidden_states[logits_indices]
    → compute_logits
    → sampler
    → sampled token ids
    → detokenize text

pooling：
  hidden_states
    → model.pooler
    → pooling tensor
    → PoolingRequestOutput
```

---

## 16. 最终可以记成一张表

| 阶段 | 生成式请求 | pooling / embedding / rerank 请求 |
|---|---|---|
| 参数对象 | `SamplingParams` | `PoolingParams` |
| EngineCoreRequest 字段 | `sampling_params` | `pooling_params` |
| Request 字段 | `Request.sampling_params` | `Request.pooling_params` |
| Scheduler 目标 | prefill + decode，持续生成 token | 计算完整 prompt，得到 pooler output |
| KV cache | 使用 | 通常仍使用 |
| attention metadata | 使用 | 通常仍使用 |
| forward 后 | `compute_logits()` | `_pool()` |
| worker 输出 | `sampled_token_ids` / `logprobs` | `pooler_output` |
| Scheduler 完成条件 | stop / eos / max_tokens / abort 等 | `pooler_output is not None` |
| OutputProcessor | detokenize + RequestOutput | PoolingOutput + PoolingRequestOutput |
| streaming | 支持 | 强制 FINAL_ONLY |
| 上层转换 | Completion / Chat chunk | Embedding / Classification / Score / Rerank |

---

## 17. 容易混淆的点

### 17.1 pooling output 是 sampled token 吗？

不是。

```text
pooling output 是模型 hidden states 经 pooler 后得到的 tensor，
不是从 vocab logits 中采样出来的 token id。
```

### 17.2 embedding 请求是否需要 detokenize？

通常不需要。

```text
embedding / classification / score 返回的是 tensor / vector / scalar，
OutputProcessor 不会像生成式请求那样把 token ids 转成 text。
```

### 17.3 pooling 请求是否完全不走 Scheduler？

不是。

pooling 请求仍然通过：

```text
Scheduler → Worker → ModelRunner → model forward
```

只是 forward 后不走 logits / sampler。

### 17.4 pooling 请求是否完全不需要 KV cache？

不是。

普通 transformer pooling 模型仍可能使用：

```text
KV cache
prefix cache
block table
slot mapping
attention metadata
```

具体是否使用还取决于模型结构、runner、prefix cache 配置和 task 参数。

### 17.5 pooling 请求是否可以 streaming？

通常不可以。

`PoolingParams` 强制 `FINAL_ONLY`，因为 pooling 输出要等完整结果产生后才能返回。

### 17.6 score / rerank 是新的底层模型执行链路吗？

通常不是。

```text
score / rerank 多数是在 entrypoint 层把 query-doc pair 转成 pooling/classify 请求，
底层仍然走 PoolingParams → pooler_output → PoolingRequestOutput。
```

### 17.7 token_embed / token_classify 为什么要注意 prefix cache？

因为 token-level 输出需要 token 维 hidden states。

如果前缀被 prefix cache 命中而不重新 forward，就可能无法返回完整 token-level 输出，所以默认会跳过读取 prefix cache。

---

## 18. 一句话总结

pooling / embedding / rerank 是 vLLM 推理链路中的“非生成式后半段”：

```text
它们共享 Scheduler、KV cache、attention metadata 和 model forward，
但在 hidden states 之后不进入 logits / sampler，
而是进入 model.pooler，产出 tensor，再由 OutputProcessor 包装成 PoolingRequestOutput。
```

如果只记住一句话，就是：

```text
生成式输出回答“下一个 token 是什么”，pooling 输出回答“这段输入的向量、类别或分数是什么”。
```
