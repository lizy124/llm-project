# 02. SamplingParams 和 SamplingMetadata 如何进入 sampler？

源码位置：

- `code/vllm/vllm/sampling_params.py`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/engine/parallel_sampling.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/utils.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/metadata.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `code/vllm/vllm/v1/sample/ops/logprobs.py`
- `code/vllm/vllm/v1/sample/ops/penalties.py`
- `code/vllm/vllm/v1/sample/logits_processor/builtin.py`
- `code/vllm/vllm/v1/sample/logits_processor/__init__.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/structured_output/request.py`
- `code/vllm/vllm/v1/structured_output/utils.py`
- `code/vllm/vllm/v1/engine/output_processor.py`

本问题关注：用户传入的 sampling 参数如何从请求层进入 worker 侧 batch，并最终被 sampler 消费；同时区分哪些字段在 sampler 前生效，哪些字段在 Scheduler / OutputProcessor 阶段生效。

---

## 1. 一句话回答

`SamplingParams` 是用户级采样配置，描述“这个请求想怎么生成”；`SamplingMetadata` 是 worker / sampler 侧按当前 batch 整理出的执行态，描述“这一轮 batch 里每一行 logits 应该怎么采”。

核心链路是：

```text
用户传入 SamplingParams
  → InputProcessor 校验、clone、补齐 generation config / tokenizer 信息
  → EngineCoreRequest.sampling_params
  → Request.sampling_params
  → Scheduler 首次调度时放入 NewRequestData
  → GPUModelRunner._update_states()
  → CachedRequestState.sampling_params
  → InputBatch.add_request()
      → 拆成 temperature / top_p / top_k / penalties / generators / logprobs 等列式状态
  → InputBatch._make_sampling_metadata()
      → SamplingMetadata
  → GPUModelRunner.sample_tokens()
      → grammar bitmask 先 mask logits
      → _sample(logits, spec_decode_metadata)
  → Sampler / RejectionSampler
      → SamplerOutput
```

如果压缩成一句话：

```text
SamplingParams 是 request 级配置，SamplingMetadata 是 batch 级执行视图，sampler 只消费后者。
```

---

## 2. 为什么不能直接把 SamplingParams 丢给 sampler

`SamplingParams` 是“单个请求”的配置，但 sampler 面对的是“本轮 batch 的 logits”。

同一个 batch 里可能同时存在：

```text
- greedy 请求和 random 请求；
- 不同 temperature / top_p / top_k 的请求；
- 有 seed 和没有 seed 的请求；
- 需要 logprobs 和不需要 logprobs 的请求；
- 有 repetition / presence / frequency penalty 的请求；
- 有 allowed_token_ids / bad_words / logits processors 的请求；
- structured output 请求和普通请求；
- spec decode 请求和普通 decode 请求；
- prefill chunk、decode、bonus token 等不同采样位置。
```

所以 worker 侧需要把 request 级配置整理成 batch 级张量和索引结构：

```text
SamplingParams：
  面向请求，字段丰富，适合校验和表达用户意图。

InputBatch sampling state：
  面向当前活跃 batch，适合随请求增删 / swap / condense 维护。

SamplingMetadata：
  面向 sampler，尽量只传本轮实际需要的 GPU tensor / dict / mask。
```

这也是为什么 `SamplingMetadata` 不是 `SamplingParams` 的简单 list。

---

## 3. SamplingParams 包含什么

`SamplingParams` 定义在 `code/vllm/vllm/sampling_params.py:199`。

它是用户级生成配置，大致可以分成几组。

### 3.1 采样数量和并行采样

```text
n：
  每个 prompt 要生成几个 output。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:213`

注意：V1 中 `n > 1` 不会直接让一个 request 在 worker 内部一次性带 `n` 个分支采样，而是在 engine 层被 fan out 成多个 child request，每个 child 的 `n` 改成 1。

相关位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:268`
- `code/vllm/vllm/v1/engine/llm_engine.py:279`
- `code/vllm/vllm/v1/engine/parallel_sampling.py:72`

### 3.2 随机采样参数

```text
temperature：
  控制分布平滑程度；接近 0 时走 greedy。

top_p：
  nucleus sampling。

top_k：
  只在 top-k token 中采样；0 / -1 可表示禁用。

min_p：
  根据最大概率相对阈值过滤低概率 token。

seed：
  请求级随机种子。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:236`
- `code/vllm/vllm/sampling_params.py:240`
- `code/vllm/vllm/sampling_params.py:243`
- `code/vllm/vllm/sampling_params.py:246`
- `code/vllm/vllm/sampling_params.py:250`

### 3.3 penalty 参数

```text
presence_penalty：
  出现过的 token 统一惩罚。

frequency_penalty：
  按出现频次惩罚。

repetition_penalty：
  对重复 token 进行缩放式惩罚。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:224`
- `code/vllm/vllm/sampling_params.py:228`
- `code/vllm/vllm/sampling_params.py:232`

这几类参数最终需要 prompt token ids 和 output token ids 参与计算，所以 worker 侧不是只传 penalty 数值，还要在 `SamplingMetadata` 中按需传 token ids。

### 3.4 停止条件

```text
stop：
  stop strings。

stop_token_ids：
  stop token ids。

ignore_eos：
  是否忽略 EOS。

max_tokens：
  最大生成 token 数。

min_tokens：
  最小生成 token 数。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:252`
- `code/vllm/vllm/sampling_params.py:255`
- `code/vllm/vllm/sampling_params.py:259`
- `code/vllm/vllm/sampling_params.py:262`
- `code/vllm/vllm/sampling_params.py:264`

这里要区分：

```text
stop strings：
  主要在 detokenize / OutputProcessor 阶段检查。

stop_token_ids / EOS：
  主要在 Scheduler 的 stop 检查中根据最后生成 token 判断。

min_tokens：
  需要在 sampler 前通过 logits processor 屏蔽 EOS / stop token。
```

### 3.5 logprobs 相关

```text
logprobs：
  返回生成 token 位置的 top logprobs。

prompt_logprobs：
  返回 prompt token 位置的 logprobs。

logprob_token_ids：
  指定额外要返回 logprob 的 token ids。

flat_logprobs：
  控制 logprobs 返回形态。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:267`
- `code/vllm/vllm/sampling_params.py:275`
- `code/vllm/vllm/sampling_params.py:278`
- `code/vllm/vllm/sampling_params.py:284`

### 3.6 logits processors / structured output 相关

```text
structured_outputs：
  guided decoding / grammar / schema 等约束。

logit_bias：
  对指定 token 加 bias。

allowed_token_ids：
  限制候选 token 集合。

bad_words：
  禁止生成某些 token 序列。

extra_args：
  扩展参数。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:316`
- `code/vllm/vllm/sampling_params.py:318`
- `code/vllm/vllm/sampling_params.py:321`
- `code/vllm/vllm/sampling_params.py:337`
- `code/vllm/vllm/sampling_params.py:324`

---

## 4. SamplingParams 的归一化和校验

### 4.1 `__post_init__` 做基础归一化

`SamplingParams.__post_init__()` 会把用户输入规范成内部更容易处理的形式。

典型逻辑：

```text
- 极小但非零 temperature 会被抬高到安全值；
- seed == -1 会转成 None；
- stop=None 转成 []；
- stop 是字符串时转成单元素 list；
- stop_token_ids=None 转成 []；
- bad_words=None 转成 []；
- logprobs=True 转成 1；
- prompt_logprobs=True 转成 1；
- 有 stop string 且不 include_stop_str_in_output 时，设置 output_text_buffer_length；
- greedy 场景下强制 top_p=1、top_k=0、min_p=0，并禁止 n > 1；
- stop_token_ids 汇总进 _all_stop_token_ids；
- prompt_logprobs 默认会导致 skip_reading_prefix_cache=True。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:429`
- `code/vllm/vllm/sampling_params.py:440`
- `code/vllm/vllm/sampling_params.py:447`
- `code/vllm/vllm/sampling_params.py:452`
- `code/vllm/vllm/sampling_params.py:455`
- `code/vllm/vllm/sampling_params.py:458`
- `code/vllm/vllm/sampling_params.py:464`
- `code/vllm/vllm/sampling_params.py:471`
- `code/vllm/vllm/sampling_params.py:479`
- `code/vllm/vllm/sampling_params.py:481`
- `code/vllm/vllm/sampling_params.py:604`

这里有一个关键点：

```text
prompt_logprobs 会影响 prefix cache 行为。

原因是如果直接复用 prefix cache，prompt 前缀部分可能不重新 forward，
那么就无法返回完整 prompt_logprobs。
```

### 4.2 `_verify_args` 做字段范围校验

`_verify_args()` 主要检查参数自身是否合法。

典型校验：

```text
- n >= 1，且不超过环境变量限制；
- presence/frequency penalty 在 [-2, 2]；
- repetition penalty 有限且 > 0；
- temperature 有限、非负、<= 2；
- top_p 在 (0, 1]；
- top_k 允许 0 或 -1 表示禁用；
- min_p 在 [0, 1]；
- max_tokens / min_tokens 合法；
- logprobs / prompt_logprobs 合法；
- stop_token_ids、stop strings、bad_words 合法。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:487`
- `code/vllm/vllm/sampling_params.py:499`
- `code/vllm/vllm/sampling_params.py:507`
- `code/vllm/vllm/sampling_params.py:517`
- `code/vllm/vllm/sampling_params.py:535`
- `code/vllm/vllm/sampling_params.py:542`
- `code/vllm/vllm/sampling_params.py:550`
- `code/vllm/vllm/sampling_params.py:552`
- `code/vllm/vllm/sampling_params.py:567`
- `code/vllm/vllm/sampling_params.py:573`
- `code/vllm/vllm/sampling_params.py:584`
- `code/vllm/vllm/sampling_params.py:589`
- `code/vllm/vllm/sampling_params.py:597`

### 4.3 `verify()` 做模型相关校验

`SamplingParams.verify()` 是进入 engine 前的模型级校验入口。

它会调用：

```text
_validate_logprobs
_validate_logit_bias
_validate_logits_processors
_validate_allowed_token_ids
_validate_spec_decode
_validate_structured_outputs
```

源码位置：

- `code/vllm/vllm/sampling_params.py:717`

典型逻辑：

```text
logprobs=-1 / prompt_logprobs=-1：
  展开成 vocab size，再和 model_config.max_logprobs 比较。

logprob_token_ids：
  校验长度上限、token id 范围，以及是否和 logprobs 数量匹配。

spec decode：
  当前不支持部分采样配置，例如 min_p / logit_bias。

structured outputs：
  校验 tokenizer、backend、grammar / schema，并在 auto backend 下选择实现。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:733`
- `code/vllm/vllm/sampling_params.py:750`
- `code/vllm/vllm/sampling_params.py:848`
- `code/vllm/vllm/sampling_params.py:862`

---

## 5. SamplingParams 如何进入 Request

### 5.1 LLMEngine.add_request 接收 SamplingParams

V1 前端入口 `LLMEngine.add_request()` 接收：

```text
params: SamplingParams | PoolingParams
```

源码位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:218`

如果是原始 prompt 输入，会先进入 `InputProcessor.process_inputs()`。

源码位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:250`
- `code/vllm/vllm/v1/engine/input_processor.py:242`

### 5.2 InputProcessor 校验和补齐 SamplingParams

`InputProcessor._validate_params()` 会对 `SamplingParams` 调用 `params.verify(...)`。

源码位置：

- `code/vllm/vllm/v1/engine/input_processor.py:82`
- `code/vllm/vllm/v1/engine/input_processor.py:95`

`process_inputs()` 还会：

```text
1. clone SamplingParams，避免直接修改用户原对象；
2. 补齐 max_tokens=None 的情况；
3. 应用 generation config，例如 EOS / stop token；
4. 应用 tokenizer 信息，例如 bad words tokenization；
5. 构造 EngineCoreRequest。
```

源码位置：

- `code/vllm/vllm/v1/engine/input_processor.py:313`
- `code/vllm/vllm/v1/engine/input_processor.py:323`
- `code/vllm/vllm/v1/engine/input_processor.py:327`
- `code/vllm/vllm/v1/engine/input_processor.py:370`

`EngineCoreRequest` 中直接包含 `sampling_params`。

源码位置：

- `code/vllm/vllm/v1/engine/__init__.py:88`

### 5.3 EngineCoreRequest 转成 scheduler 内部 Request

`EngineCore.preprocess_add_request()` 将 `EngineCoreRequest` 转成 scheduler 内部 `Request`。

源码位置：

- `code/vllm/vllm/v1/engine/core.py:853`

`Request.from_engine_core_request()` 透传 `sampling_params`。

源码位置：

- `code/vllm/vllm/v1/request.py:197`

`Request.__init__()` 中保存为：

```text
self.sampling_params
```

并基于它构造 structured output request、设置 `max_tokens` 和初始状态。

源码位置：

- `code/vllm/vllm/v1/request.py:59`
- `code/vllm/vllm/v1/request.py:84`
- `code/vllm/vllm/v1/request.py:87`
- `code/vllm/vllm/v1/request.py:107`

---

## 6. n > 1 和 seed 如何处理

### 6.1 n == 1：普通请求

如果 `params.n == 1`，`LLMEngine.add_request()` 直接把请求加入 EngineCore。

源码位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:268`

### 6.2 n > 1：fan out 成多个 child request

如果 `n > 1`，V1 会通过 `ParentRequest` 把一个用户请求拆成多个 child request。

关键点：

```text
- 每个 child request 的 sampling_params.n 改为 1；
- 如果原始 seed 非空，每个 child 的 seed = seed + index；
- 后续 worker / sampler 看到的是多个普通 request，而不是一个 n>1 request。
```

源码位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:279`
- `code/vllm/vllm/v1/engine/parallel_sampling.py:68`
- `code/vllm/vllm/v1/engine/parallel_sampling.py:72`
- `code/vllm/vllm/v1/engine/parallel_sampling.py:78`

这说明：

```text
parallel sampling 的“多路输出”在 engine 层拆请求，
不是在 sampler 里让单个 request 一次返回 n 个 token 分支。
```

---

## 7. Scheduler 如何把 SamplingParams 下发给 Worker

Scheduler 内部保存 `Request` 对象。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:171`

每轮 `Scheduler.schedule()` 会生成 `SchedulerOutput`。

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:388`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1059`

首次调度的新请求通过 `NewRequestData` 下发完整数据，其中包含：

```text
sampling_params=request.sampling_params
```

源码位置：

- `code/vllm/vllm/v1/core/sched/output.py:30`
- `code/vllm/vllm/v1/core/sched/output.py:47`
- `code/vllm/vllm/v1/core/sched/output.py:181`

已缓存请求后续走 `scheduled_cached_reqs`，不会每轮重复发送完整 `SamplingParams`。

这条边界很重要：

```text
Scheduler 是 SamplingParams 的长期持有者；
Worker 只在新请求首次进入执行层时接收并缓存它。
```

---

## 8. Worker / InputBatch 如何保存 per-request sampling 参数

### 8.1 CachedRequestState 保存原始 SamplingParams

worker 侧 `CachedRequestState` 保存：

```text
sampling_params
generator
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:34`

### 8.2 GPUModelRunner._update_states 处理新请求

`GPUModelRunner._update_states()` 处理 `scheduler_output.scheduled_new_reqs` 时，会取出：

```text
new_req_data.sampling_params
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1203`

如果请求使用随机种子，会创建请求级 `torch.Generator`：

```text
torch.Generator(device=self.device).manual_seed(sampling_params.seed)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1206`

然后创建 `CachedRequestState`，保存 sampling params 和 generator。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1224`

如果请求需要 `prompt_logprobs`，还会记录到：

```text
self.num_prompt_logprobs[req_id]
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1241`

### 8.3 InputBatch.add_request 拆成列式状态

真正供 batch 执行使用的是 `InputBatch` 中的列式状态。

`InputBatch.add_request()` 会把单个请求的 `SamplingParams` 拆到 batch 结构里。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:336`

典型拆分如下：

```text
greedy / random：
  根据 sampling_params.sampling_type 放入 greedy_reqs / random_reqs。

temperature：
  写入 temperature_cpu。

top_p / top_k：
  写入 top_p_cpu / top_k_cpu，并用集合记录是否有非默认请求。

penalties：
  写入 frequency / presence / repetition penalty CPU tensor。

generator：
  按 req_index 放入 self.generators。

logprobs：
  放入 self.num_logprobs。

logprob_token_ids：
  放入 self.logprob_token_ids。

allowed_token_ids：
  懒分配并写入 mask。

bad_words_token_ids：
  保存到 req_index -> list[list[int]]。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:381`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:390`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:399`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:411`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:416`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:423`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:427`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:450`

### 8.4 删除、swap、condense 时也要同步 sampling 状态

`InputBatch` 是持久 batch，会随请求完成、抢占、重排而变化。

因此 sampling 相关状态也必须同步维护：

```text
remove_request：
  清理 greedy/random/top-p/top-k/penalty/generator/logprob/allowed/bad_words 等状态。

swap / condense：
  请求行号变化时，同步交换或搬移 sampling 状态。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:511`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:567`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:684`

这和 `InputBatch` 的职责一致：

```text
InputBatch 不只是 token_ids 的缓存，
也是 worker 侧每个请求采样状态的持久 batch 表。
```

---

## 9. SamplingMetadata 是怎么构造的

### 9.1 SamplingMetadata 字段

`SamplingMetadata` 定义在：

- `code/vllm/vllm/v1/sample/metadata.py:14`

它是 sampler 真正消费的结构，核心字段包括：

```text
temperature
all_greedy
all_random

top_p
top_k

generators

max_num_logprobs
logprob_token_ids

frequency_penalties
presence_penalties
repetition_penalties
prompt_token_ids
output_token_ids

allowed_token_ids_mask
bad_words_token_ids
logitsprocs

spec decode 相关状态
thinking budget 相关状态
```

源码位置：

- `code/vllm/vllm/v1/sample/metadata.py:16`
- `code/vllm/vllm/v1/sample/metadata.py:20`
- `code/vllm/vllm/v1/sample/metadata.py:23`
- `code/vllm/vllm/v1/sample/metadata.py:25`
- `code/vllm/vllm/v1/sample/metadata.py:28`
- `code/vllm/vllm/v1/sample/metadata.py:36`
- `code/vllm/vllm/v1/sample/metadata.py:40`
- `code/vllm/vllm/v1/sample/metadata.py:43`
- `code/vllm/vllm/v1/sample/metadata.py:46`
- `code/vllm/vllm/v1/sample/metadata.py:51`

### 9.2 InputBatch._make_sampling_metadata

`InputBatch._make_sampling_metadata()` 从 batch 内部状态构造 `SamplingMetadata`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:832`

它会做几个重要优化：

```text
全 greedy：
  不复制 temperature 到 GPU，temperature=None。

没有非默认 top_p / top_k：
  top_p / top_k 传 None。

没有 penalty：
  penalty tensors 传 None。

只有 penalty / bad_words / logits processors / thinking budget 需要 token ids 时：
  才构造 prompt_token_ids / output_token_ids。

allowed_token_ids_mask：
  按当前 batch index 重建。

logprob_token_ids：
  按当前 batch index 重建。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:832`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:840`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:845`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:861`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:878`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:896`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:906`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:915`

可以把它理解成：

```text
SamplingMetadata 是按需 materialize 的。

如果 batch 里没有某类功能，就不传对应 GPU tensor，
避免每轮无意义的数据复制和 kernel 处理。
```

### 9.3 refresh_metadata 何时发生

`InputBatch.refresh_metadata()` 会在 batch 状态变化后更新 logits processors 状态，并重建：

```text
self.sampling_metadata
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py:812`

这一步通常发生在：

```text
- 新请求加入；
- 请求结束或移除；
- batch swap / condense；
- output token ids 更新后需要刷新 penalty / logits processor 依赖状态。
```

---

## 10. GPUModelRunner 如何使用 SamplingMetadata

### 10.1 execute_model 阶段先算 logits，不立即直接采样

`GPUModelRunner.execute_model()` 是模型前向入口。

主流程包括：

```text
1. _update_states(scheduler_output)
   → 更新 InputBatch 和 sampling metadata。

2. _prepare_inputs()
   → 计算 logits_indices 和 spec_decode_metadata。

3. _model_forward()
   → 得到 hidden states。

4. 根据 logits_indices 取 sample_hidden_states。

5. compute_logits(sample_hidden_states)
   → 得到 logits。

6. 保存 ExecuteModelState。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4047`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4088`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4131`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4357`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4389`

### 10.2 EngineCore 再调用 sample_tokens

`EngineCore.step()` 中，如果 `execute_model()` 返回 `None`，会取 grammar bitmask 并调用 `sample_tokens()`。

源码位置：

- `code/vllm/vllm/v1/engine/core.py:491`
- `code/vllm/vllm/v1/engine/core.py:498`

这说明 V1 中执行可以分成两段：

```text
execute_model：
  负责 forward 和 logits。

sample_tokens：
  负责 grammar mask、sampler、bookkeeping、ModelRunnerOutput。
```

### 10.3 sample_tokens 中先处理 grammar，再采样

`GPUModelRunner.sample_tokens()` 会：

```text
1. 取出 ExecuteModelState；
2. 如果有 structured output grammar bitmask，先修改 logits；
3. 调用 _sample(logits, spec_decode_metadata)；
4. bookkeeping；
5. 组装 ModelRunnerOutput。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4426`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4455`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4461`

### 10.4 _sample 中真正把 SamplingMetadata 交给 sampler

`_sample()` 从：

```text
self.input_batch.sampling_metadata
```

取出 metadata。

普通采样：

```text
self.sampler(
    logits=logits,
    sampling_metadata=sampling_metadata,
)
```

spec decode：

```text
self.rejection_sampler(
    ...,
    sampling_metadata=sampling_metadata,
)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3573`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3589`

---

## 11. Sampler 输入输出结构

### 11.1 Sampler.forward 输入

`Sampler` 定义在：

- `code/vllm/vllm/v1/sample/sampler.py:20`

`Sampler.forward()` 输入包括：

```text
logits：
  当前需要采样的位置的 logits。

sampling_metadata：
  当前 batch 的采样参数执行态。

predict_bonus_token：
  spec decode bonus token 路径使用。

logprobs_mode_override：
  特殊路径覆盖 logprobs 模式。
```

源码位置：

- `code/vllm/vllm/v1/sample/sampler.py:72`

### 11.2 SamplerOutput

`SamplerOutput` 定义在：

- `code/vllm/vllm/v1/outputs.py:185`

主要包含：

```text
sampled_token_ids：
  普通采样时通常是 [num_requests, 1]。

logprobs_tensors：
  如果请求需要 logprobs，则携带 token ids、logprobs、rank 等。
```

相关位置：

- `code/vllm/vllm/v1/sample/sampler.py:141`
- `code/vllm/vllm/v1/outputs.py:52`

---

## 12. Sampler 内部处理顺序

`Sampler.forward()` 的核心顺序可以概括成：

```text
logits
  → 必要时先基于 raw logits 计算 raw logprobs
  → logits 转 float32
  → apply_logits_processors()
      → allowed_token_ids
      → bad_words
      → non-argmax-invariant processors，例如 min_tokens、logit_bias
      → penalties
  → sample()
      → greedy argmax
      → random 分支应用 temperature
      → argmax-invariant processors，例如 min_p
      → top-k / top-p
      → multinomial sampling
      → greedy/random 混合 batch 合并
  → gather logprobs
  → SamplerOutput
```

源码位置：

- `code/vllm/vllm/v1/sample/sampler.py:84`
- `code/vllm/vllm/v1/sample/sampler.py:95`
- `code/vllm/vllm/v1/sample/sampler.py:255`
- `code/vllm/vllm/v1/sample/sampler.py:273`
- `code/vllm/vllm/v1/sample/sampler.py:280`
- `code/vllm/vllm/v1/sample/sampler.py:285`
- `code/vllm/vllm/v1/sample/sampler.py:296`
- `code/vllm/vllm/v1/sample/sampler.py:395`
- `code/vllm/vllm/v1/sample/sampler.py:399`
- `code/vllm/vllm/v1/sample/sampler.py:403`
- `code/vllm/vllm/v1/sample/sampler.py:407`

logprobs 相关收集：

```text
logprob_token_ids：
  gather_specific_token_logprobs()

logprobs=-1：
  返回全 vocab logprobs。

普通 logprobs：
  gather_logprobs()
```

源码位置：

- `code/vllm/vllm/v1/sample/sampler.py:111`
- `code/vllm/vllm/v1/sample/sampler.py:122`
- `code/vllm/vllm/v1/sample/sampler.py:128`

---

## 13. 各类 SamplingParams 字段在哪里生效

### 13.1 temperature

处理层次：

```text
SamplingParams：
  定义、校验、greedy 归一化。

InputBatch：
  写入 temperature_cpu，并区分 greedy / random。

SamplingMetadata：
  非全 greedy 时传入 temperature tensor。

Sampler：
  random 分支中 logits / temperature。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:236`
- `code/vllm/vllm/sampling_params.py:517`
- `code/vllm/vllm/sampling_params.py:471`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:381`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:832`
- `code/vllm/vllm/v1/sample/sampler.py:228`

### 13.2 top_p / top_k

处理层次：

```text
SamplingParams：
  定义和范围校验。

InputBatch：
  保存 top_p / top_k；top_k 未启用时可设为 vocab size。

SamplingMetadata：
  只有 batch 中存在非默认 top_p / top_k 请求时才传 tensor。

Sampler：
  调用 TopKTopPSampler。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:240`
- `code/vllm/vllm/sampling_params.py:243`
- `code/vllm/vllm/sampling_params.py:535`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:390`
- `code/vllm/vllm/v1/sample/sampler.py:285`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py:345`

### 13.3 min_p

`min_p` 在 V1 中不是 `SamplingMetadata` 的顶层字段，而是内置 logits processor。

处理层次：

```text
SamplingParams：
  定义和校验。

LogitsProcessor：
  MinPLogitsProcessor。

Sampler：
  在 temperature 后、top-k/top-p 前应用 argmax-invariant processors。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:246`
- `code/vllm/vllm/sampling_params.py:550`
- `code/vllm/vllm/v1/sample/logits_processor/__init__.py:49`
- `code/vllm/vllm/v1/sample/logits_processor/builtin.py:23`
- `code/vllm/vllm/v1/sample/logits_processor/builtin.py:47`
- `code/vllm/vllm/v1/sample/sampler.py:280`

### 13.4 penalties

处理层次：

```text
SamplingParams：
  定义和范围校验。

InputBatch：
  保存 penalty 数值，并记录哪些请求有非默认 penalty。

SamplingMetadata：
  传 penalty tensor，同时按需传 prompt_token_ids / output_token_ids。

Sampler：
  apply_all_penalties()。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:224`
- `code/vllm/vllm/sampling_params.py:228`
- `code/vllm/vllm/sampling_params.py:232`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:399`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:845`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:861`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:878`
- `code/vllm/vllm/v1/sample/sampler.py:422`
- `code/vllm/vllm/v1/sample/ops/penalties.py:10`

### 13.5 seed

处理层次：

```text
SamplingParams：
  seed 字段；seed == -1 转 None；带 seed 时 sampling_type 为 RANDOM_SEED。

GPUModelRunner：
  创建 per-request torch.Generator。

InputBatch：
  按 req_index 保存 generator。

TopKTopPSampler：
  随机采样时使用 generators。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:250`
- `code/vllm/vllm/sampling_params.py:440`
- `code/vllm/vllm/sampling_params.py:679`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1206`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:411`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py:446`

### 13.6 logprobs

处理层次：

```text
SamplingParams：
  定义、校验、logprobs=-1 展开为 vocab size。

InputBatch：
  保存每个 req_id 需要的 num_logprobs。

SamplingMetadata：
  记录当前 batch 的 max_num_logprobs 和 logprob_token_ids。

Sampler：
  在 raw logits/logprobs 上 gather top-k / full vocab / specified token ids。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:267`
- `code/vllm/vllm/sampling_params.py:700`
- `code/vllm/vllm/sampling_params.py:733`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:416`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py:1121`
- `code/vllm/vllm/v1/sample/sampler.py:84`
- `code/vllm/vllm/v1/sample/sampler.py:120`

### 13.7 prompt_logprobs

`prompt_logprobs` 不等于普通 generation logprobs。

处理层次：

```text
SamplingParams：
  定义和校验；默认影响 prefix cache 读取。

GPUModelRunner：
  新请求时记录 num_prompt_logprobs。

bookkeeping：
  用 prompt hidden states 重新 compute logits，并计算 prompt token 的 logprobs。

ModelRunnerOutput：
  通过 prompt_logprobs_dict 返回。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:275`
- `code/vllm/vllm/sampling_params.py:481`
- `code/vllm/vllm/sampling_params.py:783`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1241`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5461`
- `code/vllm/vllm/v1/outputs.py:251`

### 13.8 stop / stop_token_ids / eos / min_tokens

这组参数不是全部在 sampler 中处理。

```text
stop strings：
  SamplingParams 归一化，并设置 output_text_buffer_length；
  OutputProcessor / detokenizer 检测字符串命中。

stop_token_ids / EOS：
  Scheduler check_stop() 根据最后生成 token 判断结束。

min_tokens：
  MinTokensLogitsProcessor 在达到最小长度前屏蔽 EOS / stop token。
```

源码位置：

- `code/vllm/vllm/sampling_params.py:447`
- `code/vllm/vllm/sampling_params.py:464`
- `code/vllm/vllm/v1/engine/output_processor.py:639`
- `code/vllm/vllm/v1/core/sched/utils.py:94`
- `code/vllm/vllm/v1/sample/logits_processor/builtin.py:165`

---

## 14. structured output 和 SamplingMetadata 的关系

structured output 不是 `SamplingMetadata` 的普通字段。

它的链路是：

```text
SamplingParams.structured_outputs
  → Request.structured_output_request
  → Scheduler / structured output manager 编译和维护 grammar 状态
  → Scheduler.get_grammar_bitmask()
  → EngineCore 把 grammar output 交给 sample_tokens()
  → GPUModelRunner.sample_tokens()
      → apply_grammar_bitmask(logits)
  → Sampler 在被 mask 后的 logits 上采样
  → Scheduler.update_from_output()
      → accept_tokens 推进 grammar 状态
```

源码位置：

- `code/vllm/vllm/sampling_params.py:316`
- `code/vllm/vllm/v1/request.py:87`
- `code/vllm/vllm/v1/request.py:111`
- `code/vllm/vllm/v1/structured_output/request.py:31`
- `code/vllm/vllm/v1/engine/core.py:867`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1140`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1440`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4455`
- `code/vllm/vllm/v1/structured_output/utils.py:85`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1599`

关键理解：

```text
structured output 是 sampler 前的 logits mask，
而不是 sampler 内部的一组普通 top-k/top-p 参数。
```

---

## 15. spec decode 和 SamplingMetadata 的关系

Spec decode 会改变 logits 的行含义和采样位置，但仍会复用同一份 `SamplingMetadata`。

### 15.1 spec decode metadata

Spec decode metadata 定义在：

- `code/vllm/vllm/v1/spec_decode/metadata.py:9`

它包含：

```text
- draft_token_ids
- num_draft_tokens
- cu_num_draft_tokens
- target_logits_indices
- bonus_logits_indices
- logits_indices
```

`_prepare_inputs()` 根据 `scheduler_output.scheduled_spec_decode_tokens` 判断是否进入 spec decode：

```text
普通 decode：
  logits_indices 是每个 request 最后一个 query 位置。

spec decode：
  调用 _calc_spec_decode_metadata() 构造 target / bonus / draft 相关索引。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2162`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2171`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2750`

### 15.2 普通 sampler 和 rejection sampler 共用 SamplingMetadata

普通采样走：

```text
Sampler(logits, sampling_metadata)
```

spec decode 走：

```text
RejectionSampler(..., sampling_metadata)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3573`
- `code/vllm/vllm/v1/sample/rejection_sampler.py:88`

Spec decode 中：

```text
- bonus token 仍通过普通 Sampler 采样；
- target logits 会应用 allowed token ids、penalties、bad words、min_tokens 等处理；
- temperature/top-k/top-p 会按 draft token 展开到 token 级别；
- structured output 还要验证 draft token 是否满足 grammar。
```

源码位置：

- `code/vllm/vllm/v1/sample/rejection_sampler.py:129`
- `code/vllm/vllm/v1/sample/rejection_sampler.py:285`
- `code/vllm/vllm/v1/sample/rejection_sampler.py:510`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1896`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1918`
- `code/vllm/vllm/v1/structured_output/utils.py:112`

---

## 16. 整体数据流图

把全部链路串起来：

```text
用户 SamplingParams
  → SamplingParams.__post_init__()
      → 基础归一化
  → InputProcessor._validate_params()
      → SamplingParams.verify()
  → InputProcessor.process_inputs()
      → clone
      → max_tokens 兜底
      → generation config / tokenizer 更新
  → EngineCoreRequest.sampling_params
  → Request.from_engine_core_request()
  → Request.sampling_params
  → Scheduler.schedule()
  → NewRequestData.sampling_params
  → SchedulerOutput.scheduled_new_reqs
  → GPUModelRunner._update_states()
      → CachedRequestState.sampling_params
      → seeded torch.Generator
  → InputBatch.add_request()
      → greedy/random sets
      → temperature_cpu
      → top_p_cpu / top_k_cpu
      → penalty tensors
      → generators
      → num_logprobs / logprob_token_ids
      → allowed token mask / bad words
  → InputBatch.refresh_metadata()
  → InputBatch._make_sampling_metadata()
  → SamplingMetadata
  → GPUModelRunner.execute_model()
      → hidden states
      → logits_indices
      → logits
      → ExecuteModelState
  → GPUModelRunner.sample_tokens(grammar_bitmask)
      → apply_grammar_bitmask()
      → _sample()
  → Sampler / RejectionSampler
  → SamplerOutput
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → OutputProcessor
```

---

## 17. 最终可以记成一张表

| 阶段 | 主要对象 / 函数 | 核心产物 | 作用 |
|---|---|---|---|
| 用户配置 | `SamplingParams` | request 级采样参数 | 表达用户想怎么生成 |
| 参数归一化 | `SamplingParams.__post_init__()` | 规范化字段 | 处理默认值、greedy、stop、logprobs 等 |
| 模型级校验 | `SamplingParams.verify()` | 校验后的 params | 校验 logprobs、spec decode、structured output 等 |
| 输入处理 | `InputProcessor.process_inputs()` | `EngineCoreRequest` | clone params，补 generation config / tokenizer 信息 |
| 请求状态 | `Request` | `Request.sampling_params` | Scheduler 长期持有请求级参数 |
| 首次下发 | `NewRequestData` | `sampling_params` | 新请求进入 worker 时携带完整参数 |
| worker 缓存 | `CachedRequestState` | sampling params + generator | worker 侧保存请求采样配置 |
| batch 状态 | `InputBatch.add_request()` | 列式 sampling state | 把 request 参数拆成 batch 可维护结构 |
| metadata 构造 | `_make_sampling_metadata()` | `SamplingMetadata` | sampler 真正消费的 batch 执行态 |
| 前向 | `execute_model()` | logits | 根据 logits_indices 计算采样位置 logits |
| 采样 | `sample_tokens()` / `_sample()` | `SamplerOutput` | 用 metadata 从 logits 采 token |
| 回收 | `_bookkeeping_sync()` / `ModelRunnerOutput` | worker 输出 | 传回 Scheduler 更新 request 状态 |

---

## 18. 容易混淆的点

### 18.1 SamplingParams 和 SamplingMetadata 是一回事吗？

不是。

```text
SamplingParams：
  单请求、用户级、字段完整、适合校验。

SamplingMetadata：
  当前 batch、执行级、按需 materialize、适合 sampler。
```

### 18.2 stop string 会进入 sampler 吗？

通常不会。

```text
stop string 主要在 detokenizer / OutputProcessor 阶段基于文本检测；
stop_token_ids / EOS 则由 Scheduler 根据 token id 检查；
min_tokens 才会通过 logits processor 在 sampler 前屏蔽 stop/eos token。
```

### 18.3 structured output 是 SamplingMetadata 字段吗？

不是普通字段。

```text
structured output 先由 Scheduler 生成 grammar bitmask，
再由 GPUModelRunner.sample_tokens() 在 sampler 前 mask logits。
```

### 18.4 prompt_logprobs 和 logprobs 是同一条路径吗？

不是。

```text
logprobs：
  generation token 位置，sampler 输出相关。

prompt_logprobs：
  prompt token 位置，通常需要 prompt hidden states 额外计算。
```

### 18.5 n > 1 是 sampler 一次采多个吗？

V1 里通常不是。

```text
n > 1 会在 engine 层 fan out 成多个 child request，
每个 child 的 sampling_params.n 变成 1。
```

### 18.6 spec decode 会绕过 SamplingMetadata 吗？

不会。

```text
spec decode 使用 RejectionSampler，
但仍复用当前 batch 的 SamplingMetadata，
只是额外引入 SpecDecodeMetadata 描述 draft / target / bonus 位置。
```

---

## 19. 一句话总结

`SamplingParams` 到 `SamplingMetadata` 的转换，本质是一次从“用户请求级配置”到“worker batch 级执行态”的翻译：

```text
SamplingParams 负责描述“这个请求想怎么采”，
InputBatch 负责把多个请求的采样配置维护成 batch 状态，
SamplingMetadata 负责把本轮真正需要的字段交给 sampler。
```

如果只记住一句话，就是：

```text
sampler 不直接理解用户请求，它只理解 logits + SamplingMetadata；SamplingParams 必须先经过 Engine、Scheduler、Worker、InputBatch 四层翻译。
```
