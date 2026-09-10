# vLLM V1 Spec Decode 逻辑梳理

源码位置：

- `code/vllm/vllm/config/speculative.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/spec_decode/metadata.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/structured_output/__init__.py`
- `code/vllm/vllm/v1/structured_output/utils.py`

本文按“先定边界，再走主链路，再拆关键阶段，最后总结接口和状态修正”的方式，梳理 vLLM V1 speculative decoding 的完整机制。

它和 `scheduler`、`executor_worker_model_runner`、`sampling_and_output` 都有关，但本目录专门把分散在 Scheduler、ModelRunner、Sampler、KV cache、structured output 和 output recovery 里的逻辑串成一个闭环。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的文档风格，本文不是逐个 helper 函数的字典式说明，而是先回答几个最关键的问题：

```text
1. Spec decode 在 vLLM V1 中解决什么问题？
2. 它为什么不是单纯 sampler 分支，而是一条跨层协议？
3. draft tokens 从哪里来，如何挂在 Request 状态上？
4. Scheduler 如何把 draft tokens 调度给 target model 验证？
5. ModelRunner 如何构造 spec decode forward 的输入和 logits 布局？
6. RejectionSampler 如何接受 / 拒绝 draft tokens？
7. accepted / recovered / bonus tokens 如何回到 Scheduler 并落账？
8. KV cache、num_computed_tokens、grammar、logprobs、OutputProcessor 如何保持一致？
9. async scheduling、PP、chunked prefill、KV connector 等边界如何影响 spec decode？
10. 相关专题文档之间如何阅读？
```

本目录要回答的问题分成 10 组：

```text
1. Spec decode 在 vLLM 中的角色和边界；
2. draft tokens 和 Request 状态；
3. Scheduler spec decode 调度流程；
4. SpecDecodeMetadata 的 logits 行号布局；
5. ModelRunner spec forward；
6. RejectionSampler 接受 / 拒绝流程；
7. KV cache 和 num_computed_tokens 一致性；
8. structured output / grammar 交互；
9. output recovery 和 Scheduler.update_from_output；
10. 限制、边界和容易踩坑场景。
```

阅读顺序建议：

```text
spec_decode_overview.md
  → 01_spec_decode_role.md
  → 02_draft_tokens_and_request_state.md
  → 03_scheduler_spec_decode_flow.md
  → 04_spec_decode_metadata.md
  → 05_model_runner_spec_forward.md
  → 06_rejection_sampler_flow.md
  → 07_kv_cache_and_num_computed_tokens.md
  → 08_structured_output_interaction.md
  → 09_output_recovery_and_scheduler_update.md
  → 10_limitations_and_edge_cases.md
```

如果只想先抓主线，可以先读：

```text
spec_decode_overview.md
  → 03_scheduler_spec_decode_flow.md
  → 04_spec_decode_metadata.md
  → 06_rejection_sampler_flow.md
  → 09_output_recovery_and_scheduler_update.md
```

---

## 1. 一句话回答

Spec decode 的本质是：

```text
用便宜路径先猜多个 draft tokens，
再用 target model 一次 forward 验证这些 token，
通过 rejection sampling 接受一段前缀，
拒绝时采一个 recovered token，
全部接受时补一个 bonus token，
最后把 request、KV cache、grammar、logprobs 和输出状态修正到一致。
```

它不是一个独立组件，而是一条贯穿多层的协议：

```text
SpeculativeConfig
  → Scheduler.num_spec_tokens / num_lookahead_tokens
  → Request.spec_token_ids
  → SchedulerOutput.scheduled_spec_decode_tokens
  → InputBatch.spec_token_ids / token_ids_cpu
  → SpecDecodeMetadata
  → target model forward / logits
  → RejectionSampler
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.output_token_ids / num_computed_tokens 修正
  → proposer 产生下一轮 DraftTokenIds
  → Scheduler.update_draft_token_ids()
```

最小心智模型：

```text
Spec decode = drafter 猜 token + Scheduler 调度候选 + target model 验 token + RejectionSampler 选 token + Scheduler 回写账本。
```

---

## 2. 它解决什么问题

普通 generation 每轮通常只让 target model 产生一个 token：

```text
当前上下文
  → target model forward
  → logits
  → Sampler
  → 1 个 token
```

spec decode 希望把一次 target model forward 变成“验证多个候选 token”：

```text
当前上下文
  → drafter 先猜 [d1, d2, d3, ...]
  → target model 一次 forward 验证这些 draft tokens
  → 接受尽可能长的 draft prefix
  → 如果全接受，再额外采 bonus token
```

如果 drafter 足够快、acceptance rate 足够高，target model 的一次 forward 就能推进多个 token，从而提高吞吐。

但难点也在这里：

```text
draft tokens 被猜过、调度过、forward 过，
不代表它们一定是最终输出。
```

所以系统必须区分：

```text
candidate tokens：drafter 猜出来的候选；
verified tokens：target model 验证过的位置；
accepted tokens：真正被接受的 draft prefix；
recovered token：draft 被拒绝后 target / residual 分布采出来的替代 token；
bonus token：所有 draft 都接受后额外采样的 token；
finalized tokens：写入 Request.output_token_ids 的真实输出。
```

一句话：

```text
spec decode 的收益来自“多 token 验证”，复杂度来自“只提交 accepted/finalized token”。
```

---

## 3. 和普通 decode 的关键差异

| 阶段 | 普通 decode | spec decode |
|---|---|---|
| 请求状态 | 只看 prompt + output tokens | 还要暂存 `Request.spec_token_ids` |
| Scheduler token 模型 | `num_tokens` / `num_computed_tokens` | `num_tokens_with_spec = num_tokens + len(spec_token_ids)` |
| KV allocation | 为本轮 token 分配 slot | 还要传 `num_lookahead_tokens` |
| Worker batch | 每个请求通常追加 1 个 decode token | 在真实 token 后追加 draft tokens |
| logits 布局 | 每个请求通常 1 行 logits | target verification rows + bonus rows |
| sampler | `Sampler` | `RejectionSampler`，内部仍复用普通 `Sampler` |
| 输出 token 数 | 每请求通常 1 个 | 每请求可能是 `accepted + 1` 个 |
| 状态回收 | append sampled token | 统计 accepted/rejected，回滚 rejected，append generated tokens |
| structured output | mask 当前 token logits | 还要 validate draft、为每个 spec position 生成 bitmask |
| logprobs | 每请求 1 个位置为主 | 每请求本轮输出长度可能不同 |

最关键的变化是：

```text
普通 decode：sampled token 几乎就是 output token。
spec decode：sampled_token_ids 是 accepted / recovered / bonus 后的真实输出，原始 draft 不一定输出。
```

---

## 4. 总体流程图

```text
初始化阶段

VllmConfig.speculative_config
  → SpeculativeConfig
  → Scheduler.num_spec_tokens
  → Scheduler.num_lookahead_tokens
  → GPUModelRunner.drafter
  → GPUModelRunner.rejection_sampler
```

```text
第 N 轮结束后：产生下一轮 draft

GPUModelRunner.sample_tokens()
  → target model 本轮 sampled_token_ids
  → propose_draft_token_ids(...)
  → GPUModelRunner._draft_token_ids
  → take_draft_token_ids()
  → DraftTokenIds(req_ids, draft_token_ids)
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

```text
第 N+1 轮：消费上一轮 draft

Scheduler.schedule()
  → num_new_tokens = request.num_tokens_with_spec
                     + request.num_output_placeholders
                     - request.num_computed_tokens
  → apply token budget / max_model_len / encoder / KV limits
  → KVCacheManager.allocate_slots(..., num_lookahead_tokens)
  → SchedulerOutput.scheduled_spec_decode_tokens
  → request.spec_token_ids = []
  → _update_after_schedule()
      → request.num_computed_tokens += num_scheduled_tokens
```

```text
ModelRunner target verification forward

GPUModelRunner.execute_model(scheduler_output)
  → _update_states()
      → InputBatch.update_req_spec_token_ids()
  → _prepare_inputs()
      → num_draft_tokens
      → SpecDecodeMetadata
      → logits_indices
  → _build_attention_metadata(use_spec_decode=True)
  → _model_forward()
  → hidden_states[logits_indices]
  → model.compute_logits(...)
  → ExecuteModelState(..., spec_decode_metadata, logits, hidden_states)
  → return None
```

```text
sampling / rejection / next proposal

GPUModelRunner.sample_tokens(grammar_output)
  → apply_grammar_bitmask(logits)
  → _sample(logits, spec_decode_metadata)
      → RejectionSampler.forward(...)
      → SamplerOutput.sampled_token_ids
  → _bookkeeping_sync()
      → RejectionSampler.parse_output()
      → valid_sampled_token_ids
  → propose_draft_token_ids(...)
  → ModelRunnerOutput(sampled_token_ids=valid_sampled_token_ids)
```

```text
Scheduler 输出回收

Scheduler.update_from_output(scheduler_output, model_runner_output)
  → generated_token_ids = sampled_token_ids[req_index]
  → scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens[req_id]
  → num_accepted = max(len(generated_token_ids) - num_sampled, 0)
  → num_rejected = len(scheduled_spec_token_ids) - num_accepted
  → request.num_computed_tokens -= num_rejected
  → request.num_output_placeholders -= num_rejected
  → request.append_output_token_ids(generated_token_ids)
  → grammar.accept_tokens(new_token_ids)
  → EngineCoreOutput(new_token_ids)
  → OutputProcessor
```

---

## 5. 核心对象关系

```text
SpeculativeConfig
  配置入口，决定 method、draft model、num_speculative_tokens、rejection strategy。

Request.spec_token_ids
  Scheduler 侧暂存的下一轮 draft tokens。

Request.num_tokens_with_spec
  prompt + accepted output + pending draft，是 Scheduler 统一 token 模型的关键。

SchedulerOutput.scheduled_spec_decode_tokens
  本轮真正交给 Worker / target model 验证的 draft tokens。

SchedulerOutput.num_spec_tokens_to_schedule
  dynamic speculative decoding 给下一轮 proposer 的 draft token 数 K。

InputBatch.spec_token_ids
  Worker 侧当前 batch row 上的 draft token 状态。

SpecDecodeMetadata
  本轮 spec decode 的 logits 行号说明书：draft ids、target logits indices、bonus logits indices。

RejectionSampler
  根据 draft ids、draft probs、target logits 和 sampling metadata 决定 accepted / recovered / bonus。

ModelRunnerOutput.sampled_token_ids
  RejectionSampler 后真正要交给 Scheduler 落账的 token lists。

DraftTokenIds
  ModelRunner proposer 生成、回传给 Scheduler 的下一轮 draft tokens。

EngineCoreOutput.new_token_ids
  Scheduler 确认后的本轮新增 tokens，是 OutputProcessor 的输入。
```

---

## 6. SpeculativeConfig：控制面入口

`SpeculativeConfig` 定义在：`code/vllm/vllm/config/speculative.py:75`

它描述：

```text
- 是否开启 speculative decoding；
- 使用哪种 method；
- 每轮最多 draft 多少 token；
- 是否使用 draft model；
- draft model 的并行、量化、attention backend；
- rejection sampling 方法；
- dynamic speculative decoding 策略。
```

关键字段包括：

```text
num_speculative_tokens
model
method
draft_tensor_parallel_size
quantization
moe_backend
attention_backend
max_model_len
prompt_lookup_max / prompt_lookup_min
parallel_drafting
num_speculative_tokens_per_batch_size
rejection_sample_method
draft_sample_method
```

### 6.1 method 决定 drafter 类型

`GPUModelRunner.__init__()` 会根据 speculative config 创建 drafter。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:545` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:620`

常见分支包括：

```text
custom_class
ngram
ngram_gpu
draft_model
dflash
suffix
eagle / eagle3
medusa
extract_hidden_states
Gemma4 / Step3.5 MTP 等模型特化路径
```

也就是说：

```text
SpeculativeConfig 是“启用什么 speculative 方法”的控制面；
GPUModelRunner 是“把方法落成运行时 drafter / rejection sampler”的执行面。
```

### 6.2 num_speculative_tokens 不只是性能参数

`num_speculative_tokens` 会影响：

```text
1. Scheduler 默认让 proposer 下一轮生成几个 draft tokens；
2. Scheduler.num_lookahead_tokens；
3. RejectionSampler 输出矩阵宽度 max_spec_len + 1；
4. InputBatch 预留 spec token 空间；
5. async spec decode 的 prev_num_spec_tokens / placeholders；
6. CUDA graph / padding / attention metadata 的最大 shape。
```

所以它既是性能参数，也是 shape / 状态边界参数。

### 6.3 dynamic speculative decoding

如果配置了：

```python
num_speculative_tokens_per_batch_size
```

Scheduler 会构造 `dynamic_sd_lookup`，并在每轮根据当前 scheduled batch size 设置：

```text
SchedulerOutput.num_spec_tokens_to_schedule
```

这个字段不是“本轮验证了几个 draft tokens”，而是：

```text
本轮执行后，proposer 下一轮应该最多生成几个 draft tokens。
```

---

## 7. Scheduler：spec decode 的请求账本和调度边界

Scheduler 不生成 draft tokens。

它负责：

```text
1. 保存 Request.spec_token_ids；
2. 把 spec tokens 纳入 num_tokens_with_spec；
3. 按 token budget / max_model_len / KV capacity 决定本轮能验证多少；
4. 构造 SchedulerOutput.scheduled_spec_decode_tokens；
5. 为 KV allocation 传 num_lookahead_tokens；
6. schedule 后乐观推进 num_computed_tokens；
7. update_from_output 后按 rejected tokens 回滚；
8. 把下一轮 DraftTokenIds 写回 Request.spec_token_ids。
```

### 7.1 Request 上的 spec 状态

`Request` 中的关键字段：

```python
self.num_output_placeholders = 0
self.async_tokens_to_discard = 0
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
```

位置：`code/vllm/vllm/v1/request.py:140` 到 `code/vllm/vllm/v1/request.py:153`

两个关键属性：

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)

@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

位置：`code/vllm/vllm/v1/request.py:247` 到 `code/vllm/vllm/v1/request.py:252`

含义：

```text
num_tokens：prompt + accepted output，真实 token 边界。
num_tokens_with_spec：真实 token + pending draft，调度候选边界。
num_computed_tokens：计算进度账本，schedule 后可能乐观包含 draft。
```

### 7.2 Scheduler 的统一 token 模型

`schedule()` 注释说明，Scheduler 没有单独的 prefill / decode / spec phase。

它只让：

```text
request.num_computed_tokens 追上 request.num_tokens_with_spec。
```

RUNNING 请求中会计算：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:462` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:466`

因此：

```text
如果 request.spec_token_ids 非空，num_new_tokens 可能覆盖多个 draft verification tokens。
```

### 7.3 scheduled_spec_decode_tokens 的构造

当请求带 draft tokens 时，Scheduler 计算本轮调度区间中落在 spec 区域的 token 数：

```python
num_scheduled_spec_tokens = (
    num_new_tokens
    + request.num_computed_tokens
    - request.num_tokens
    - request.num_output_placeholders
)
```

然后截断并写入：

```python
scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
request.spec_token_ids = []
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:581` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:597`

这说明：

```text
Request.spec_token_ids：跨 step 暂存的 draft tokens；
SchedulerOutput.scheduled_spec_decode_tokens：本轮真实要验证的 draft tokens；
调度后 request.spec_token_ids 清空，避免重复验证旧 draft。
```

### 7.4 num_lookahead_tokens 和 KV allocation

Scheduler 初始化时：

```python
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.num_lookahead_tokens = 0
```

如果使用 EAGLE、draft model、DFlash 等方法，会设置 lookahead：

```python
if speculative_config.use_eagle():
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.uses_draft_model():
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.use_dflash():
    self.num_lookahead_tokens = self.num_spec_tokens + 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:227` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:248`

分配 KV slots 时传入：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:521` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:528`

这说明 spec decode 不是采样层小优化，而会影响 KV block admission。

---

## 8. SchedulerOutput：Scheduler 到 Worker 的 spec 载体

`SchedulerOutput` 定义在：`code/vllm/vllm/v1/core/sched/output.py:180`

和 spec decode 直接相关的字段：

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
num_invalid_spec_tokens: dict[str, int] | None = None
num_spec_tokens_to_schedule: int = 0
```

含义：

| 字段 | 含义 |
|---|---|
| `scheduled_spec_decode_tokens` | 本轮要交给 target model 验证的 draft tokens |
| `num_invalid_spec_tokens` | structured output / deferred output 下非法 draft 数，用于统计修正 |
| `num_spec_tokens_to_schedule` | 下一轮 proposer 应生成的 draft token 数 K |

还有一些字段会被 spec decode 间接影响：

| 字段 | 影响 |
|---|---|
| `num_scheduled_tokens` | 可能包含 draft verification tokens |
| `total_num_scheduled_tokens` | spec tokens 占用 token budget |
| `scheduled_cached_reqs` | Worker 侧缓存请求 diff 需要和 spec tokens 对齐 |
| `new_block_ids_to_zero` | spec decode 分配的新 KV blocks 也可能需要 zero |
| `has_structured_output_requests` | grammar bitmask 需要知道 scheduled spec tokens |

一句话：

```text
SchedulerOutput 是 spec decode 从 Scheduler 进入 Worker / ModelRunner 的出站协议。
```

---

## 9. EngineCore：把两个闭环串起来

EngineCore 的基础执行链路仍然是：

```text
schedule → execute_model → sample_tokens → update_from_output
```

`EngineCore.step()` 中：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`code/vllm/vllm/v1/engine/core.py:479` 到 `code/vllm/vllm/v1/engine/core.py:508`

spec decode 增加的是 post-step draft 回写：

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`code/vllm/vllm/v1/engine/core.py:510` 到 `code/vllm/vllm/v1/engine/core.py:517`

因此 spec decode 有两个闭环。

### 9.1 验证闭环

```text
Request.spec_token_ids
  → Scheduler.schedule()
  → SchedulerOutput.scheduled_spec_decode_tokens
  → ModelRunner target verification forward
  → RejectionSampler
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.output_token_ids / num_computed_tokens
```

它回答：

```text
上一轮猜的 tokens，这一轮接受几个？
```

### 9.2 生成闭环

```text
本轮 sampled_token_ids / hidden_states / metadata
  → drafter.propose(...)
  → GPUModelRunner._draft_token_ids
  → take_draft_token_ids()
  → DraftTokenIds
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

它回答：

```text
下一轮要让 target model 验哪些 tokens？
```

这两个闭环首尾相接，形成持续 speculative decoding。

---

## 10. ModelRunner：把 scheduled draft 变成 target forward

`GPUModelRunner` 是 spec decode 执行层核心。

它负责：

```text
1. 创建 drafter；
2. 创建 RejectionSampler；
3. 接收 SchedulerOutput.scheduled_spec_decode_tokens；
4. 把 draft tokens 写入 InputBatch；
5. 准备 input_ids / positions / slot mapping；
6. 构造 SpecDecodeMetadata；
7. 执行 target model forward；
8. 计算 target / bonus logits；
9. 在 sample_tokens() 中调用 RejectionSampler；
10. 调 proposer 生成下一轮 DraftTokenIds。
```

### 10.1 初始化 drafter 和 rejection sampler

当开启 spec decode 且当前 rank 是 PP last rank：

```python
if self.speculative_config and get_pp_group().is_last_rank:
    ...
    self.drafter = ...
    self.rejection_sampler = RejectionSampler(
        self.sampler, self.speculative_config, self.device
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:545` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:620`

为什么是 last PP rank？

```text
只有 last rank 拿到最终 hidden states / logits，
才能执行 sampling、rejection sampling 和 draft proposal。
```

### 10.2 _update_states：scheduled spec tokens 落入 InputBatch

`execute_model()` 开头会调用：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4085`

`_update_states()` 会读取：

```python
scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
```

并调用：

```python
self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
```

`InputBatch.update_req_spec_token_ids()` 会把 spec tokens 写到真实 token 后面：

```python
start_index = self.num_tokens_no_spec[req_index]
end_token_index = start_index + num_spec_tokens
self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
self.is_token_ids[req_index, start_index:end_token_index] = True
cur_spec_token_ids.extend(spec_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:483` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:508`

这说明：

```text
spec tokens 在 Worker 侧不是另起一个 batch，
而是追加在同一个 request row 的真实 token 后面，
由后续 metadata 决定哪些位置用于 target verification。
```

### 10.3 _prepare_inputs：构造 logits_indices 和 SpecDecodeMetadata

当本轮存在 scheduled spec tokens：

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```

`_prepare_inputs()` 会按 `InputBatch.req_id_to_index` 计算每个请求的 draft 长度，再调用 `_calc_spec_decode_metadata()`。

结果包括：

```text
logits_indices：从 hidden_states 中取哪些行去 compute_logits；
SpecDecodeMetadata.target_logits_indices：哪些 logits rows 验证 draft tokens；
SpecDecodeMetadata.bonus_logits_indices：哪些 logits rows 采 bonus token。
```

也就是说：

```text
ModelRunner spec forward 的核心，是把“req_id -> draft token list”转换成“target model logits row layout”。
```

---

## 11. SpecDecodeMetadata：本轮 logits 行号说明书

`SpecDecodeMetadata` 定义在：`code/vllm/vllm/v1/spec_decode/metadata.py:9`

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor
    num_draft_tokens: list[int]
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor
```

字段含义：

| 字段 | 含义 |
|---|---|
| `draft_token_ids` | flatten 后的 draft token ids |
| `num_draft_tokens` | 每个请求本轮有几个 draft tokens |
| `cu_num_draft_tokens` | draft tokens 的前缀和，用于 flatten 索引 |
| `cu_num_sampled_tokens` | 每个请求最多输出 `draft_len + 1` 的前缀和 |
| `target_logits_indices` | logits tensor 中用于验证 draft tokens 的 rows |
| `bonus_logits_indices` | logits tensor 中用于 bonus sampling 的 rows |
| `logits_indices` | 从 full hidden states 中取哪些 rows 计算 logits |

它解决的问题是：

```text
普通 decode：每个请求通常只需要 1 个 logits row。

spec decode：每个请求需要 K 个 target verification logits
             + 1 个 bonus logits。
```

一句话：

```text
SpecDecodeMetadata 不是保存请求状态，而是保存本轮 spec decode 的 logits 行号布局。
```

---

## 12. execute_model：target model 验证 draft tokens

spec decode 的 target verification forward 仍在 `GPUModelRunner.execute_model()` 中完成。

主线是：

```text
_update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → hidden_states[logits_indices]
  → model.compute_logits()
  → ExecuteModelState(...)
  → return None
```

`ExecuteModelState` 会保存：

```text
scheduler_output
logits
spec_decode_metadata
spec_decode_common_attn_metadata
hidden_states
sample_hidden_states
aux_hidden_states
ec_connector_output
cudagraph_stats
slot_mappings
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4398`

为什么 `execute_model()` 返回 `None`？

```text
generation 模型的 forward 和 sampling 被拆成两步：
execute_model() 负责 forward / logits，
sample_tokens() 负责 grammar bitmask、Sampler / RejectionSampler、bookkeeping、draft proposal 和 ModelRunnerOutput。
```

---

## 13. sample_tokens：从 logits 到 accepted / recovered / bonus

`GPUModelRunner.sample_tokens()` 会取出 `ExecuteModelState`，先应用 grammar bitmask：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

然后调用：

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

如果 `spec_decode_metadata is None`：

```text
走普通 Sampler。
```

如果 `spec_decode_metadata` 存在：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3592` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3598`

所以分流规则很简单：

```text
没有 SpecDecodeMetadata：Sampler。
有 SpecDecodeMetadata：RejectionSampler。
```

---

## 14. RejectionSampler：接受 / 拒绝的判决器

`RejectionSampler` 定义在：`code/vllm/vllm/v1/sample/rejection_sampler.py:37`

它的输入是：

```text
SpecDecodeMetadata：draft / target / bonus logits 布局；
draft_probs：draft model 给出的概率，可为 None；
logits：target model logits；
sampling_metadata：temperature / top-k / top-p / penalties / allowed tokens 等。
```

主流程：

```text
1. 从 logits 中取 bonus_logits；
2. 用普通 Sampler 采 bonus_token_ids；
3. 从 logits 中取 target_logits；
4. 对 target_logits 应用 logits processors / penalties / allowed tokens / bad words；
5. 对 target_logits 应用 temperature / top-k / top-p；
6. 调用 rejection_sample() 接受 / 拒绝 draft tokens；
7. 如需 logprobs，收集 logprobs_tensors；
8. 返回 SamplerOutput。
```

输出规则：

```text
从左到右验证 draft tokens：
  - 接受：输出该 draft token，继续验证下一个；
  - 拒绝：输出 recovered token，后续 draft 不再输出；
  - 全部接受：输出所有 draft tokens，再追加 bonus token。
```

所以 `SamplerOutput.sampled_token_ids` 在 spec decode 下是：

```text
[batch_size, max_spec_len + 1]
```

其中：

```text
真实 token id：accepted / recovered / bonus；
-1：padding 或 rejected 后不再使用的位置。
```

---

## 15. _bookkeeping_sync：把 SamplerOutput 变成 ModelRunnerOutput

`RejectionSampler` 返回的是 GPU tensor，Scheduler 需要 Python / CPU 结构。

`GPUModelRunner._bookkeeping_sync()` 会处理：

```text
- 过滤 -1 padding；
- 生成 valid_sampled_token_ids；
- 生成 logprobs_lists；
- 生成 prompt_logprobs_dict；
- 复制 req_ids / req_id_to_index snapshot；
- 更新 Worker 侧 CachedRequestState / InputBatch token 状态。
```

当 `max_gen_len > 1` 时：

```python
valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
    sampled_token_ids,
    self.input_batch.vocab_size,
    discard_sampled_tokens_req_indices,
    logprobs_tensors=logprobs_tensors,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3656` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3674`

随后构造：

```python
ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    ...
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4610` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4617`

这意味着：

```text
Scheduler.update_from_output() 看到的 sampled_token_ids 已经是有效输出 token list，通常不再包含 -1 padding。
```

---

## 16. propose_draft_token_ids：生成下一轮候选

采样和 bookkeeping 后，ModelRunner 会生成下一轮 draft tokens。

`sample_tokens()` 内部有局部函数：

```python
def propose_draft_token_ids(sampled_token_ids):
    self._draft_token_ids = self.propose_draft_token_ids(...)
    self._copy_draft_token_ids_to_cpu(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4481` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4496`

真正入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4852`

proposer 会读取：

```python
num_spec_tokens_to_schedule = scheduler_output.num_spec_tokens_to_schedule
```

也就是说：

```text
下一轮 draft K 由 Scheduler 决定，生成动作由 ModelRunner / drafter 执行。
```

不同 proposer 使用的信息不同：

```text
ngram：使用历史 token ids 和本轮 sampled tokens；
ngram_gpu：在 GPU 上更新 token ids 并 propose；
draft model：运行小模型生成 draft；
EAGLE / EAGLE3：使用 hidden states 预测；
Medusa：使用 medusa heads；
DFlash / MTP / suffix / custom_class：走各自特化路径。
```

如果 drafter 提供 draft probabilities，会保存到：

```text
GPUModelRunner._draft_probs
GPUModelRunner._draft_prob_req_ids
```

下一轮 `_get_spec_decode_draft_probs()` 会把它们交给 `RejectionSampler`。

---

## 17. DraftTokenIds：从 Worker 回传给 Scheduler

ModelRunner 缓存下一轮 draft tokens 后，EngineCore 通过 Executor 拉取：

```text
EngineCore.post_step()
  → Executor.take_draft_token_ids()
  → Worker.take_draft_token_ids()
  → GPUModelRunner.take_draft_token_ids()
```

ModelRunner 侧：

```python
def take_draft_token_ids(self) -> DraftTokenIds | None:
    if not self.num_spec_tokens or not self._draft_token_req_ids:
        return None
    draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
    return DraftTokenIds(req_ids, draft_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4731` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4735`

Scheduler 写回：

```python
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
    ...
    if request.is_prefill_chunk:
        ...
        continue
    if self.structured_output_manager.should_advance(request):
        spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
    request.spec_token_ids = spec_token_ids
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1895` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1915`

这里有几个关键边界：

```text
1. finished request 的 draft tokens 忽略；
2. prefill chunk 不使用 draft tokens；
3. structured output 先 validate draft tokens；
4. 写入 Request.spec_token_ids 后，下一轮 schedule 才能消费。
```

---

## 18. Scheduler.update_from_output：输出回收和状态落账

`ModelRunnerOutput.sampled_token_ids` 回到 Scheduler 后，核心入口是：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

对每个本轮调度请求：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
```

如果有 scheduled spec tokens：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
request.num_output_placeholders -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1542` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1567`

然后写入真实输出：

```python
new_token_ids = generated_token_ids
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1580` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1592`

`_update_request_with_output()` 会逐个 append 并检查 stop：

```text
append token
  → check_stop
  → 如果 stop，裁掉后续 token
```

输出回收的核心原则：

```text
scheduled_spec_decode_tokens 不直接进入 output；
只有 RejectionSampler 产出的 generated_token_ids 才进入 Request.output_token_ids。
```

---

## 19. KV cache 和 num_computed_tokens 一致性

spec decode 会让 target model 计算 draft token 位置，但 rejected draft 不能成为真实上下文。

所以 vLLM 采用两阶段策略：

```text
schedule 阶段：乐观推进
  request.num_computed_tokens += num_scheduled_tokens

update 阶段：拒绝回滚
  request.num_computed_tokens -= num_rejected
```

KVCacheManager 中也明确区分：

```text
new tokens 可以包含 unverified draft tokens；
cache_blocks() 只能提交 request.num_tokens 范围内的 finalized tokens。
```

核心原则：

```text
draft tokens 可以被分配 slot、可以被 target forward 写入 KV，
但不能在未 accepted 前进入 prefix cache 或真实输出边界。
```

这避免：

```text
1. rejected draft 被后续请求复用成 prefix cache；
2. 下一轮 positions 从错误位置开始；
3. encoder cache / external KV connector 过早释放；
4. Mamba / hybrid state 使用错误 accepted token 数。
```

---

## 20. Structured output / grammar 交互

structured output 会在四个边界介入 spec decode：

```text
1. 请求进入系统时：等待 grammar ready；
2. draft tokens 写回 Scheduler 时：grammar.validate_tokens()；
3. target logits 采样前：grammar bitmask；
4. 输出回收后：grammar.accept_tokens(new_token_ids)。
```

### 20.1 draft validation

`Scheduler.update_draft_token_ids()` 中：

```python
if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

这会提前裁剪不合法 draft prefix。

### 20.2 multi-position grammar bitmask

spec decode 下，一个 structured request 可能需要：

```text
len(scheduled_spec_decode_tokens[req_id]) + 1
```

行 bitmask：

```text
draft_0 verification row
draft_1 verification row
...
bonus row
```

`StructuredOutputManager.grammar_bitmask()` 会临时 accept draft tokens 生成后续位置 mask，最后 rollback。

### 20.3 final accept_tokens

`Scheduler.update_from_output()` 中最终用 `new_token_ids` 推进 grammar：

```text
accepted draft tokens：推进；
recovered token：推进；
bonus token：推进；
rejected draft tokens：不推进。
```

一句话：

```text
structured output 在 spec decode 中不是只 mask 最后一个 logits，而是沿着“draft 可能连续接受”的路径生成多行 mask。
```

---

## 21. logprobs / routed experts / OutputProcessor

### 21.1 logprobs

spec decode 下，每个请求本轮输出 token 数可能不同。

`LogprobsLists` 支持：

```text
cu_num_generated_tokens
```

用于按请求切片。

Scheduler 回收时：

```python
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

这保证：

```text
logprobs 和最终 new_token_ids 对齐，
如果 stop 裁掉了 token，logprobs 也只取裁剪后的长度。
```

### 21.2 routed experts

spec decode 下 routed experts 切片和普通 decode 不同：

```text
普通 decode：输出 token 通常在 scheduled range 末尾；
spec decode：accepted tokens 位于 scheduled range 开头，rejected 在后面。
```

所以 Scheduler 用：

```text
routing_data[req_offset : req_offset + len(new_token_ids)]
```

而不是普通 decode 的末尾切片。

### 21.3 OutputProcessor

到 `OutputProcessor` 时，spec decode 复杂性已经被 Scheduler 消化。

它只看到：

```text
EngineCoreOutput.new_token_ids
finish_reason
stop_reason
new_logprobs
new_prompt_logprobs_tensors
```

因此：

```text
OutputProcessor 不区分 token 是普通 sample、accepted draft、recovered 还是 bonus。
它只把 confirmed token stream detokenize 成用户输出。
```

---

## 22. async scheduling / batch queue / PP 边界

### 22.1 async scheduling

同步 scheduling 下，EngineCore 在 `post_step()` 取 draft tokens。

async scheduling 下，EngineCore 注释说明：

```text
不能提前从 worker 取 draft token ids，
因此 draft token 更新更靠近 worker 侧处理。
```

ModelRunner 会维护：

```text
prev_sampled_token_ids
prev_req_id_to_index
valid_sampled_token_count_gpu
prev_num_draft_len
deferred_spec_decode_corrections
```

目的：

```text
在不阻塞 CPU D2H 输出拷贝的情况下，
让下一轮输入准备和 GPU num_computed_tokens 修正保持一致。
```

### 22.2 batch queue / deferred sampling

如果存在 deferred scheduler output，structured output 场景下必须先修正 draft tokens 再生成 grammar bitmask：

```text
take_draft_token_ids()
  → update_draft_token_ids_in_output()
  → invalid tokens pad 为 -1
  → get_grammar_bitmask()
  → sample_tokens()
```

这样保证：

```text
SchedulerOutput 的 shape / logits layout 不变，
同时 grammar bitmask 基于真实合法 draft tokens。
```

### 22.3 Pipeline Parallel

只有 last PP rank 创建 drafter / rejection sampler。

```text
非 last rank：负责中间层 forward / IntermediateTensors；
last rank：负责 logits、sampling、rejection sampling、draft proposal。
```

---

## 23. 常见方法和信息流差异

不同 speculative method 的共同点是：

```text
都要产出 draft_token_ids，
都要让 target model 验证，
都要通过 SchedulerOutput / SpecDecodeMetadata / RejectionSampler 闭环。
```

差异在于 drafter 如何提出候选：

| 方法 | draft 来源 | 是否通常有 draft_probs | 关键依赖 |
|---|---|---|---|
| ngram | 历史 token 序列匹配 | 通常无 | token_ids_cpu / sampled_token_ids |
| ngram_gpu | GPU 上 ngram proposer | 通常无 | GPU token buffer / valid sampled count |
| draft_model | 小模型生成 | 通常有 | draft model / draft_probs / sampling metadata |
| EAGLE / EAGLE3 | hidden states 预测 | 视实现而定 | target hidden states / common attention metadata |
| Medusa | medusa heads | 视实现而定 | hidden states / medusa-specific heads |
| DFlash / MTP | 模型结构特化 | 视实现而定 | aux hidden states / 特化 proposer |
| suffix / custom_class | suffix tree 或用户逻辑 | 视实现而定 | 自定义输入和 proposer 实现 |

这也是为什么 `GPUModelRunner.propose_draft_token_ids()` 需要接收很多上下文：

```text
scheduler_output
sampled_token_ids
sampling_metadata
hidden_states
sample_hidden_states
aux_hidden_states
spec_decode_metadata
spec_decode_common_attn_metadata
slot_mappings
```

不同 proposer 从中取自己需要的部分。

---

## 24. 一个完整例子：4 个 draft tokens，接受 2 个

假设第 N 轮结束后，drafter 为某请求生成：

```text
Request.spec_token_ids = [d1, d2, d3, d4]
```

### 24.1 第 N+1 轮 schedule

当前：

```text
num_tokens = 100
num_computed_tokens = 100
num_output_placeholders = 0
```

Scheduler 看到：

```text
num_tokens_with_spec = 104
num_new_tokens = 104 - 100 = 4
```

如果 token budget / KV allocation 允许：

```text
SchedulerOutput.scheduled_spec_decode_tokens[req] = [d1, d2, d3, d4]
request.spec_token_ids = []
request.num_computed_tokens 乐观推进到 104
```

### 24.2 ModelRunner forward

Worker 侧：

```text
InputBatch 把 [d1,d2,d3,d4] 写到真实 token 后；
_prepare_inputs() 构造 SpecDecodeMetadata；
target model 计算验证 d1/d2/d3/d4 的 logits 和 bonus logits。
```

### 24.3 RejectionSampler

假设：

```text
d1 accepted
d2 accepted
d3 rejected
recovered token = x
```

RejectionSampler 输出矩阵可能是：

```text
[d1, d2, x, -1, -1]
```

`parse_output()` 后：

```text
generated_token_ids = [d1, d2, x]
```

### 24.4 Scheduler update

Scheduler 计算：

```text
num_draft_tokens = 4
num_sampled = 1
num_accepted = len([d1,d2,x]) - 1 = 2
num_rejected = 4 - 2 = 2
```

回滚：

```text
request.num_computed_tokens = 104 - 2 = 102
```

正式输出：

```text
Request.output_token_ids append [d1, d2, x]
```

语义：

```text
d1/d2 成为 accepted draft 输出；
d3/d4 被拒绝，不进入输出；
x 是 target distribution 采出的真实 token；
下一轮从新上下文继续。
```

---

## 25. 各组件职责边界

### 25.1 SpeculativeConfig 负责

```text
定义是否开启 spec decode；
定义 method / model / num_speculative_tokens；
定义 draft model 配置和并行配置；
定义 rejection sampling 策略；
定义 dynamic speculative decoding 策略。
```

### 25.2 Scheduler 负责

```text
维护 Request.spec_token_ids；
用 num_tokens_with_spec 统一调度 draft tokens；
为 spec decode 传 num_lookahead_tokens；
构造 SchedulerOutput.scheduled_spec_decode_tokens；
处理 structured output 对 draft 的 validate；
根据 ModelRunnerOutput 统计 accepted / rejected；
回滚 num_computed_tokens / placeholders；
提交 Request.output_token_ids。
```

### 25.3 Executor / Worker 负责

```text
Executor：转发 execute_model / sample_tokens / take_draft_token_ids；
Worker：持有 ModelRunner，把执行和控制命令委托给 ModelRunner。
```

它们通常不理解 rejection sampling 的细节。

### 25.4 ModelRunner 负责

```text
创建 drafter 和 RejectionSampler；
把 scheduled spec tokens 写入 InputBatch；
准备 target verification forward；
构造 SpecDecodeMetadata；
执行 target model forward；
调用 RejectionSampler；
做 Worker 侧 bookkeeping；
调用 proposer 生成下一轮 draft tokens；
缓存并回传 DraftTokenIds。
```

### 25.5 RejectionSampler 负责

```text
根据 draft_token_ids、draft_probs、target logits 和 sampling metadata，
决定 accepted prefix、recovered token 和 bonus token。
```

### 25.6 OutputProcessor 负责

```text
接收 Scheduler 已确认的 EngineCoreOutput.new_token_ids；
detokenize；
处理 stop string；
更新 logprobs；
构造 RequestOutput / CompletionOutput。
```

一句话边界：

```text
Scheduler 决定“验证哪些 draft 并如何落账”，ModelRunner 决定“如何验证并继续猜”，RejectionSampler 决定“接受几个”，OutputProcessor 只处理 confirmed output stream。
```

---

## 26. 容易混淆的点

### 26.1 `spec_token_ids` 是最终输出吗？

不是。

```text
spec_token_ids 是候选 token；
只有出现在 RejectionSampler 后的 generated_token_ids 中，才会进入 output_token_ids。
```

### 26.2 `scheduled_spec_decode_tokens` 和 `_draft_token_ids` 是同一个吗？

不是。

```text
scheduled_spec_decode_tokens：本轮要验证的 draft tokens，来自 Scheduler；
_draft_token_ids：本轮执行后 proposer 生成的下一轮 draft tokens，保存在 ModelRunner。
```

### 26.3 RejectionSampler 是否完全替代普通 Sampler？

在 spec verification 阶段，是主要采样入口。

但它内部仍会调用普通 `Sampler` 来采 bonus token，并复用普通 sampling constraints / logprobs 能力。

### 26.4 draft token 被 target model forward 过，是否就算 accepted？

不是。

```text
forward 过只表示 target model 计算了验证 logits；
是否 accepted 由 RejectionSampler 决定。
```

### 26.5 rejected token 的 KV 会不会进入 prefix cache？

不会作为 finalized prefix 提交。

KV slot 可以被写入，但 `cache_blocks()` 会 cap 到 `request.num_tokens`，避免 unverified draft 进入 prefix cache。

### 26.6 为什么 generated_token_ids 长度要减 1 才是 accepted 数？

因为 spec decode 输出通常是：

```text
accepted draft prefix + recovered/bonus token
```

最后那个 token 是 target 采出来的 sampled token，不属于 accepted draft 数。

### 26.7 structured output 下 rejected draft 会推进 grammar 吗？

不会。

grammar 最终只对 `new_token_ids` 调用 `accept_tokens()`，而 rejected draft 不在 `new_token_ids` 中。

### 26.8 spec decode 是否只影响 ModelRunner？

不是。

它至少影响：

```text
Config；
Scheduler；
SchedulerOutput；
Request；
KVCacheManager；
InputBatch；
Attention metadata；
Sampler / RejectionSampler；
StructuredOutputManager；
Output recovery；
OutputProcessor 的多 token logprobs 消费。
```

---

## 27. 限制和边界地图

spec decode 的限制大致来自五层。

### 27.1 配置层

```text
num_speculative_tokens 必须有效；
draft model vocab / tokenizer 要和 target 对齐；
draft TP 只能是 1 或等于 target TP；
speculative_max_model_len 不能超过 draft / target 上限；
某些 method 只在特定 backend / ModelRunner 下支持。
```

### 27.2 调度层

```text
token budget 不足时只能验证部分 draft；
max_model_len 会限制 spec tokens，必须给 bonus token 留空间；
chunked prefill 阶段忽略 draft tokens；
preemption 会清空 stale spec_token_ids。
```

### 27.3 执行层

```text
PP 非 last rank 不做 sampling / proposal；
async scheduling 需要 GPU/CPU 状态延迟修正；
Mamba / hybrid 模型需要 num_accepted_tokens 修正 state；
CUDA graph / padding shape 要考虑 max_spec_len + 1。
```

### 27.4 采样层

```text
allowed_token_ids / bad_words / penalties 要展开到 draft-token 级；
draft_probs 可为 None，但语义不同；
logprobs 要过滤 -1 padding 并按实际 output length 对齐；
某些 custom logits processor 可能不适合 spec decode 多位置验证。
```

### 27.5 输出回收层

```text
accepted / rejected 只通过长度推断；
stop 可能裁掉本轮多 token 输出；
grammar accept_tokens 只推进最终 new_token_ids；
KV connector invalid blocks 可能额外回退 num_computed_tokens。
```

---

## 28. 推荐阅读路线

### 28.1 快速建立全局印象

```text
spec_decode_overview.md
  → 01_spec_decode_role.md
  → 02_draft_tokens_and_request_state.md
```

### 28.2 按 Scheduler 调度链路阅读

```text
02_draft_tokens_and_request_state.md
  → 03_scheduler_spec_decode_flow.md
  → 07_kv_cache_and_num_computed_tokens.md
  → 09_output_recovery_and_scheduler_update.md
```

### 28.3 按 ModelRunner / sampling 链路阅读

```text
04_spec_decode_metadata.md
  → 05_model_runner_spec_forward.md
  → 06_rejection_sampler_flow.md
  → 09_output_recovery_and_scheduler_update.md
```

### 28.4 按 grammar / structured output 阅读

```text
03_scheduler_spec_decode_flow.md
  → 08_structured_output_interaction.md
  → 09_output_recovery_and_scheduler_update.md
```

### 28.5 排查边界问题

```text
07_kv_cache_and_num_computed_tokens.md
  → 08_structured_output_interaction.md
  → 10_limitations_and_edge_cases.md
```

---

## 29. 文档定位

```text
spec_decode_overview.md：
  总览主文档，用一条端到端链路串起所有组件。

01_spec_decode_role.md：
  定义 spec decode 在 vLLM 中的职责、收益、边界，以及和普通 decode 的区别。

02_draft_tokens_and_request_state.md：
  梳理 draft tokens 从哪里来，如何保存在 Request / Scheduler / Worker 状态中。

03_scheduler_spec_decode_flow.md：
  梳理 Scheduler 如何调度 draft tokens、处理 token budget、KV allocation 和 update_from_output。

04_spec_decode_metadata.md：
  梳理 SpecDecodeMetadata 中 draft_token_ids、target_logits_indices、bonus_logits_indices 等字段。

05_model_runner_spec_forward.md：
  梳理 ModelRunner 如何准备 spec decode 输入、positions、logits_indices、attention metadata 和 target logits。

06_rejection_sampler_flow.md：
  梳理 RejectionSampler 如何接受 / 拒绝 draft tokens，以及如何采 recovered / bonus token。

07_kv_cache_and_num_computed_tokens.md：
  梳理 accepted / rejected token 如何影响 KV cache、slot mapping、num_computed_tokens 和 recompute。

08_structured_output_interaction.md：
  梳理 grammar bitmask、draft token validate、invalid spec tokens 和 structured output 状态推进。

09_output_recovery_and_scheduler_update.md：
  梳理 ModelRunnerOutput 到 Scheduler.update_from_output() 再到 OutputProcessor 的回收逻辑。

10_limitations_and_edge_cases.md：
  梳理 spec decode 与 sampling 参数、backend、chunked prefill、logprobs、grammar、KV connector 等限制。
```

---

## 30. 最关键流程图

```text
上一轮输出
  → ModelRunner.propose_draft_token_ids()
  → DraftTokenIds
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

```text
本轮调度
  → Scheduler.schedule()
      ├─ num_tokens_with_spec
      ├─ token budget / max_model_len
      ├─ KV allocate_slots(num_lookahead_tokens)
      ├─ scheduled_spec_decode_tokens
      └─ request.spec_token_ids = []
  → SchedulerOutput
```

```text
本轮执行
  → GPUModelRunner._update_states()
      └─ InputBatch.update_req_spec_token_ids()
  → GPUModelRunner._prepare_inputs()
      └─ SpecDecodeMetadata / logits_indices
  → target model forward
  → compute_logits()
  → sample_tokens()
      ├─ apply_grammar_bitmask()
      ├─ RejectionSampler.forward()
      ├─ parse_output()
      ├─ propose_draft_token_ids()
      └─ ModelRunnerOutput
```

```text
本轮回收
  → Scheduler.update_from_output()
      ├─ generated_token_ids
      ├─ scheduled_spec_token_ids
      ├─ accepted / rejected
      ├─ num_computed_tokens 回滚
      ├─ output_token_ids append
      ├─ grammar.accept_tokens()
      └─ EngineCoreOutput
  → OutputProcessor
```

---

## 31. 最小心智模型

如果只记一条主线，可以记：

```text
Scheduler 保存 drafter 猜出来的 token，
ModelRunner 让 target model 一次性验证这些 token，
RejectionSampler 决定接受几个，
Scheduler 只提交验证后的真实 token，
ModelRunner 再猜下一批 token，
形成“猜测 → 验证 → 落账 → 再猜”的循环。
```

再压缩成一句话：

```text
Spec decode 用“候选先行、target 验证、拒绝回滚、真实提交”的协议，把多 token 生成压进尽量少的 target model forward 里。
```
