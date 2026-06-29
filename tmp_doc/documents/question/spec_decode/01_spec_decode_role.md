# 01. Speculative Decoding 在 vLLM V1 里负责什么？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\speculative.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_input_batch.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\spec_decode\metadata.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\rejection_sampler.py`

本问题关注：`Speculative Decoding` 在 vLLM V1 中到底是一条怎样的跨层链路；它如何从配置开启，如何影响 Scheduler 的 token / KV block 调度，如何把 draft tokens 放进 `SchedulerOutput`，如何在 `GPUModelRunner` 中构造 drafter 和 rejection sampler，如何准备 spec decode metadata，如何执行 target model 验证和 rejection sampling，以及最终如何把下一轮 draft tokens 回传给 Scheduler。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的文档风格，本篇按“先定角色，再走主链路，再拆关键状态，最后总结边界”的方式梳理 speculative decoding。

要回答的问题分成 12 组：

```text
1. Speculative Decoding 是哪一层能力？
2. 它和 EngineCore / Scheduler / Executor / Worker / ModelRunner 的关系是什么？
3. SpeculativeConfig 如何决定方法、draft model、draft token 数？
4. Scheduler 如何调度带 spec token 的请求？
5. SchedulerOutput 里哪些字段承载 spec decode 状态？
6. ModelRunner 初始化时如何创建 drafter 和 RejectionSampler？
7. scheduled_spec_decode_tokens 如何落到 InputBatch？
8. SpecDecodeMetadata 如何决定 logits_indices / bonus_logits_indices？
9. RejectionSampler 如何接受 / 拒绝 draft tokens？
10. ModelRunner 如何提出下一轮 draft tokens？
11. EngineCore 如何通过 take_draft_token_ids() 把 draft tokens 交回 Scheduler？
12. Spec decode 和 structured output、KV cache、PP、async scheduling 有哪些交互？
```

阅读顺序建议：

```text
scheduler/spec_decode_token_states.md
  → executor_worker_model_runner/03_model_runner_role.md
  → executor_worker_model_runner/08_sampling_and_model_runner_output.md
  → spec_decode/01_spec_decode_role.md
```

本篇重点讲 speculative decoding 的总定位和跨层主链路，不把每一种 proposer 的内部算法全部展开。后续专题可以继续拆 EAGLE、Medusa、ngram、draft model、DFlash、suffix decoding 等具体方法。

---

## 1. 一句话回答

`Speculative Decoding` 是 vLLM V1 中跨 Scheduler 和 ModelRunner 的加速机制：

```text
先用 drafter 为请求提前猜一批 draft tokens，
下一轮由 target model 一次性验证这些 draft tokens，
RejectionSampler 决定接受多少个，
再把新 draft tokens 回传给 Scheduler，形成循环。
```

它不是一个单独组件，而是一条贯穿多层的协议：

```text
SpeculativeConfig
  → Scheduler.num_lookahead_tokens / scheduled_spec_decode_tokens
  → GPUModelRunner.InputBatch / SpecDecodeMetadata
  → target model forward / logits
  → RejectionSampler
  → proposer 生成下一轮 draft tokens
  → EngineCore.take_draft_token_ids()
  → Scheduler.update_draft_token_ids()
```

它负责：

```text
1. 用 draft model / ngram / EAGLE / Medusa 等方法提出候选 token；
2. 让 Scheduler 为候选 token 预留计算和 KV 空间；
3. 让 target model 一次 forward 验证多个候选 token；
4. 用 rejection sampling 决定哪些 draft tokens 被接受；
5. 把接受后的真实输出和下一轮 draft tokens 纳入请求状态。
```

它不负责：

```text
1. 替代 Scheduler 的 waiting / running 队列管理；
2. 替代 KVCacheManager 的 block 分配；
3. 替代 target model 的最终正确性；
4. 直接构造用户侧 RequestOutput；
5. 绕过采样参数、grammar、logprobs 等约束。
```

最小心智模型：

```text
Spec decode = drafter 猜 token + target model 验 token + rejection sampler 选 token + scheduler 账本回写。
```

---

## 2. 一句话总览链路

普通 generation 每轮通常只采一个 token：

```text
Scheduler.schedule()
  → target model forward
  → sample 1 token
  → Scheduler.update_from_output()
```

speculative decoding 试图把每轮变成：

```text
上一轮 proposer 产生 draft tokens
  → Scheduler.schedule() 把 draft tokens 一起排进本轮
  → target model forward 同时验证 sampled token + draft tokens
  → RejectionSampler 接受若干 draft tokens，必要时采 bonus token
  → ModelRunnerOutput.sampled_token_ids 可能包含多个 token
  → proposer 基于新状态产生下一轮 draft tokens
  → Scheduler.update_draft_token_ids()
```

展开成 vLLM V1 组件链路：

```text
EngineCore.step()
  → Scheduler.schedule()
      ├─ 使用 request.spec_token_ids
      ├─ 分配 num_lookahead_tokens
      └─ 生成 SchedulerOutput.scheduled_spec_decode_tokens
  → Executor.execute_model(scheduler_output)
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      ├─ _update_states()
      ├─ InputBatch.update_req_spec_token_ids()
      ├─ _prepare_inputs()
      ├─ _calc_spec_decode_metadata()
      ├─ _build_attention_metadata(use_spec_decode=True)
      ├─ _model_forward()
      └─ 保存 execute_model_state
  → GPUModelRunner.sample_tokens()
      ├─ _sample()
      ├─ RejectionSampler.forward()
      ├─ _update_states_after_model_execute()
      ├─ propose_draft_token_ids()
      └─ ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCore.post_step()
  → Executor.take_draft_token_ids()
  → Scheduler.update_draft_token_ids()
```

注意这里有两个 token 流：

```text
1. 本轮被验证的 draft tokens：SchedulerOutput.scheduled_spec_decode_tokens；
2. 下一轮要验证的 draft tokens：ModelRunner.take_draft_token_ids() 返回给 Scheduler。
```

---

## 3. SpeculativeConfig：配置入口

配置定义在：`code/vllm/vllm/config/speculative.py:75`

```python
class SpeculativeConfig:
    """Configuration for speculative decoding."""
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

其中最关键的是：

```text
num_speculative_tokens：每轮最多 draft 多少 token；
model：draft model / eagle head / 附加权重；
method：使用哪种 speculative 方法；
draft_model_config：post-init 后生成的 draft model 配置；
draft_parallel_config：post-init 后生成的 draft 并行配置。
```

### 3.1 method 决定 drafter 类型

在 `GPUModelRunner.__init__()` 中会根据 `self.speculative_config.method` 或 helper 方法创建 drafter。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:545` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:620`

支持的分支包括：

```text
custom_class
ngram
ngram_gpu
draft_model
gemma4_mtp
step3p5_mtp
dflash
suffix
eagle / eagle3
medusa
extract_hidden_states
```

源码上对应逻辑是：

```python
if self.speculative_config.method == "custom_class":
    ...
elif self.speculative_config.method == "ngram":
    ...
elif self.speculative_config.uses_draft_model():
    ...
elif self.speculative_config.use_ngram_gpu():
    ...
elif self.speculative_config.use_eagle():
    ...
elif self.speculative_config.method == "medusa":
    ...
```

也就是说，`SpeculativeConfig` 是 spec decode 的控制面，`GPUModelRunner` 是把控制面落成运行时对象的地方。

### 3.2 dynamic speculative decoding

`SpeculativeConfig` 还支持：

```python
num_speculative_tokens_per_batch_size
```

位置：`code/vllm/vllm/config/speculative.py:161` 到 `code/vllm/vllm/config/speculative.py:167`

它表示：

```text
根据当前 batch size 动态选择本轮应 draft 的 token 数。
```

Scheduler 初始化时会基于这个字段构造 `dynamic_sd_lookup`。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:229` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:238`

---

## 4. EngineCore：打开 spec decode 循环

EngineCore 初始化时会记录是否使用 spec decode：

```python
self.use_spec_decode = vllm_config.speculative_config is not None
self.check_for_draft_tokens = (
    self.use_spec_decode or vllm_config.model_config.is_diffusion
)
```

位置：`code/vllm/vllm/v1/engine/core.py:159` 到 `code/vllm/vllm/v1/engine/core.py:162`

这说明 EngineCore 并不直接生成 draft tokens，但它知道：

```text
如果开启 spec decode，每轮模型执行后需要检查 worker 是否产生了下一轮 draft tokens。
```

### 4.1 step() 中的主执行

`EngineCore.step()` 主链路仍然是：

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

位置：`code/vllm/vllm/v1/engine/core.py:490` 到 `code/vllm/vllm/v1/engine/core.py:506`

spec decode 没有改变 EngineCore 的基本闭环：

```text
schedule → execute → update
```

它改变的是 `SchedulerOutput` 和 `ModelRunnerOutput` 里的 token 语义：

```text
SchedulerOutput 可能带 draft tokens；
ModelRunnerOutput 每个请求可能返回多个 sampled tokens。
```

### 4.2 post_step() 中取回 draft tokens

同步 scheduling 路径下，EngineCore 在 `post_step()` 中取回下一轮 draft tokens：

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`code/vllm/vllm/v1/engine/core.py:510` 到 `code/vllm/vllm/v1/engine/core.py:517`

这一步非常关键：

```text
ModelRunner 产生 draft tokens，
EngineCore 通过 Executor 拉回来，
Scheduler 写入 request.spec_token_ids，
下一轮 schedule() 才能把它们放进 scheduled_spec_decode_tokens。
```

---

## 5. Scheduler：spec decode 的调度账本

Scheduler 初始化时会读取 speculative 配置：

```python
speculative_config = vllm_config.speculative_config
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.num_lookahead_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:227` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:230`

如果开启某些 spec 方法，会设置 lookahead KV 空间：

```python
if speculative_config.use_eagle():
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.uses_draft_model():
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.use_dflash():
    self.num_lookahead_tokens = self.num_spec_tokens + 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:239` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:248`

### 5.1 为什么需要 num_lookahead_tokens

spec decode 会让某个请求在当前真实 token 之后额外验证 draft tokens。

这要求 KV cache 提前留出空间：

```text
普通 decode：通常只需要为当前新 token 分配 slot；
spec decode：还要为可能被验证的 draft tokens 预留 lookahead slots。
```

Scheduler 分配 KV slots 时会把它传给 KVCacheManager：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:523` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:528`

所以：

```text
Spec decode 不只是采样优化，也会影响 KV block allocation。
```

### 5.2 Scheduler 的统一 token 模型

Scheduler 的注释给了一个重要心智模型：

```text
num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:389` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:397`

这意味着 Scheduler 不单独维护“decode phase / prefill phase / spec phase”。

它统一看：

```text
request.num_computed_tokens 要追上 request.num_tokens_with_spec。
```

其中：

```text
prompt_token_ids：已经给定的输入；
output_token_ids：已经确认生成的输出；
spec_token_ids：drafter 猜的、等待 target model 验证的 token。
```

---

## 6. SchedulerOutput：spec tokens 的出站载体

`SchedulerOutput` 定义在：`code/vllm/vllm/v1/core/sched/output.py:180`

和 spec decode 直接相关的字段有：

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
num_invalid_spec_tokens: dict[str, int] | None = None
num_spec_tokens_to_schedule: int = 0
```

位置：

```text
scheduled_spec_decode_tokens：output.py:197 到 output.py:200
num_invalid_spec_tokens：output.py:229 到 output.py:230
num_spec_tokens_to_schedule：output.py:243 到 output.py:245
```

字段含义：

| 字段 | 含义 |
|---|---|
| `scheduled_spec_decode_tokens` | 本轮要交给 target model 验证的 draft tokens |
| `num_invalid_spec_tokens` | structured output 等场景下用于修正 acceptance rate 的无效 token 数 |
| `num_spec_tokens_to_schedule` | dynamic speculative decoding 为下一轮 proposer 指定的 K |

### 6.1 Scheduler 如何填 scheduled_spec_decode_tokens

当 running request 上存在 `request.spec_token_ids` 时，Scheduler 会计算本轮能调度多少 spec tokens：

```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = (
        num_new_tokens
        + request.num_computed_tokens
        - request.num_tokens
        - request.num_output_placeholders
    )
    if num_scheduled_spec_tokens > 0:
        spec_token_ids = request.spec_token_ids
        if len(spec_token_ids) > num_scheduled_spec_tokens:
            spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
        scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
    request.spec_token_ids = []
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:581` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:597`

这说明：

```text
request.spec_token_ids 是 request 侧暂存的下一批 draft tokens；
scheduled_spec_decode_tokens 是本轮真正被调度出去验证的 draft tokens；
调度后 request.spec_token_ids 会清空，等待下一轮 update_draft_token_ids() 写入。
```

---

## 7. Scheduler.update_draft_token_ids()：把下一轮 draft tokens 写回请求

`EngineCore.post_step()` 取回 draft tokens 后，会调用：

```python
self.scheduler.update_draft_token_ids(draft_token_ids)
```

Scheduler 侧入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1895`

核心逻辑：

```python
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
    for req_id, spec_token_ids in zip(
        draft_token_ids.req_ids,
        draft_token_ids.draft_token_ids,
    ):
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            continue

        if request.is_prefill_chunk:
            if request.spec_token_ids:
                request.spec_token_ids = []
            continue

        if self.structured_output_manager.should_advance(request):
            metadata = request.structured_output_request
            spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
        request.spec_token_ids = spec_token_ids
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1895` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1915`

这里有几个重点：

```text
1. finished request 的 draft tokens 会被忽略；
2. prefill chunk 不使用 draft tokens；
3. structured output 会先用 grammar 校验 draft tokens；
4. 合法 draft tokens 写入 request.spec_token_ids，供下一轮 schedule() 使用。
```

所以 Scheduler 是 spec decode 的“请求账本层”：

```text
它不生成 draft tokens，
但它决定哪些 draft tokens 能进入下一轮执行计划。
```

---

## 8. GPUModelRunner 初始化：创建 drafter 和 rejection sampler

`GPUModelRunner.__init__()` 保存 speculative 配置：

```python
self.speculative_config = vllm_config.speculative_config
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:426` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:436`

如果开启 spec decode，并且当前是 last PP rank，就创建 drafter：

```python
if self.speculative_config and get_pp_group().is_last_rank:
    ...
    self.drafter = ...
    self.rejection_sampler = RejectionSampler(
        self.sampler, self.speculative_config, self.device
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:545` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:620`

### 8.1 为什么只在 last PP rank 创建

spec decode 的核心动作包括：

```text
1. 使用最终 hidden states / logits 验证 draft tokens；
2. 采样最终输出 token；
3. 基于最终采样结果提出下一轮 draft tokens。
```

这些动作通常只发生在 pipeline parallel 的 last rank。

因此代码里有条件：

```python
if self.speculative_config and get_pp_group().is_last_rank:
```

非 last PP rank 主要负责自己那段模型层的 forward，不负责最终 logits / sampling / draft proposal。

### 8.2 cached draft 状态

ModelRunner 内部维护 draft token 缓存：

```python
self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
self._draft_probs: torch.Tensor | None = None
self._draft_prob_req_ids: list[str] | None = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:834` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:837`

这些状态的含义：

```text
_draft_token_ids：proposer 生成、待回传给 Scheduler 的下一轮 draft tokens；
_draft_probs：draft model 对这些 tokens 的概率，standard rejection sampling 需要；
_draft_prob_req_ids：_draft_probs 的行对应哪些 request。
```

对于 ngram 这类没有概率分布的 proposer，`draft_probs` 可以是 `None`，rejection sampler 会走兼容逻辑。

---

## 9. _update_states：scheduled spec tokens 落到 worker 状态

SchedulerOutput 到达 ModelRunner 后，`execute_model()` 会先调用：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4085`

在 `_update_states()` 中会读取：

```python
scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1261` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1265`

然后对 running / resumed / new request 调用：

```python
self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
```

这一步的意义是：

```text
把 Scheduler 侧的 draft token 列表写入 worker-local InputBatch，
让后续 _prepare_inputs() 能把它们放进模型输入。
```

---

## 10. InputBatch.update_req_spec_token_ids()：spec token 的最终落点

入口：`code/vllm/vllm/v1/worker/gpu_input_batch.py:483`

核心逻辑：

```python
def update_req_spec_token_ids(
    self, request: CachedRequestState, scheduled_spec_tokens: dict[str, list[int]]
) -> None:
    req_id = request.req_id
    req_index = self.req_id_to_index[req_id]
    cur_spec_token_ids = self.spec_token_ids[req_index]
    cur_spec_token_ids.clear()
    spec_token_ids = scheduled_spec_tokens.get(req_id, ())
    num_spec_tokens = len(spec_token_ids)
    request.prev_num_draft_len = num_spec_tokens
    if not spec_token_ids:
        return

    start_index = self.num_tokens_no_spec[req_index]
    end_token_index = start_index + num_spec_tokens
    self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
    self.is_token_ids[req_index, start_index:end_token_index] = True
    cur_spec_token_ids.extend(spec_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:483` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:508`

字段含义：

| 字段 | 含义 |
|---|---|
| `spec_token_ids[req_index]` | 当前 batch row 上等待验证的 spec tokens |
| `prev_num_draft_len` | 上一轮 draft 长度，后续状态修正会用到 |
| `num_tokens_no_spec` | 不含 spec tokens 的真实 token 边界 |
| `token_ids_cpu` | worker 侧 CPU token 缓存，spec tokens 会追加在真实 token 后 |
| `is_token_ids` | 标记哪些位置已有 token id |

这说明：

```text
Spec tokens 在 InputBatch 里不是单独 batch，
而是追加到同一个 request row 的真实 token 后面，
由后续 metadata 决定哪些位置参与 logits / rejection sampling。
```

---

## 11. _prepare_inputs：构造 spec decode metadata

在 `execute_model()` 主流程中，ModelRunner 会调用：

```python
logits_indices, spec_decode_metadata = self._prepare_inputs(
    scheduler_output,
    num_scheduled_tokens_np,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4128` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4131`

内部会判断本轮是否使用 spec decode：

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4241`

当本轮有 scheduled spec tokens 时，会构造 `SpecDecodeMetadata`。

`SpecDecodeMetadata` 定义在：`code/vllm/vllm/v1/spec_decode/metadata.py:9`

字段包括：

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

位置：`code/vllm/vllm/v1/spec_decode/metadata.py:9` 到 `code/vllm/vllm/v1/spec_decode/metadata.py:24`

### 11.1 这些 indices 是什么

spec decode 下，target model forward 后需要两类 logits：

```text
1. target logits：用于验证每个 draft token；
2. bonus logits：当 draft tokens 都被接受时，再采一个额外 token。
```

因此 metadata 中有：

| 字段 | 含义 |
|---|---|
| `draft_token_ids` | flatten 后的 draft token ids |
| `num_draft_tokens` | 每个 request 有多少 draft tokens |
| `cu_num_draft_tokens` | draft token 前缀和，用于 flatten 索引 |
| `cu_num_sampled_tokens` | 每个请求最多输出 `draft_len + 1` 的前缀和 |
| `target_logits_indices` | 从 logits 中取用于验证 draft tokens 的位置 |
| `bonus_logits_indices` | 从 logits 中取 bonus token logits 的位置 |
| `logits_indices` | 本轮需要计算 logits 的所有 hidden state 位置 |

一句话：

```text
SpecDecodeMetadata 告诉 RejectionSampler：哪些 logits 用来验 draft token，哪些 logits 用来采 bonus token。
```

---

## 12. attention metadata：spec decode 如何影响 attention

ModelRunner 构造 attention metadata 时会传入：

```python
attn_metadata, spec_decode_common_attn_metadata = (
    self._build_attention_metadata(
        ...
        use_spec_decode=use_spec_decode,
        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
        ...
    )
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4269`

当 `use_spec_decode=True` 时，attention metadata 需要知道：

```text
每个请求本轮有多少 draft tokens；
多少 tokens 已被接受；
某些 drafter 是否需要 common attention metadata；
某些 proposer 是否需要按 attention group 获取 metadata。
```

在 hybrid / Mamba / GDN 等路径里，还会传递：

```text
num_accepted_tokens
num_decode_draft_tokens_cpu
```

这说明 spec decode 不只影响 sampler，也会影响模型 forward 所需的 metadata。

---

## 13. execute_model：target model 验证 draft tokens

`GPUModelRunner.execute_model()` 的 generation 路径仍然是：

```text
_update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → compute_logits()
  → save execute_model_state
  → return None
```

关键点是：

```text
如果 SchedulerOutput 里有 scheduled_spec_decode_tokens，
_prepare_inputs() 会把 draft tokens 纳入 input_ids / positions，
并让 logits_indices 覆盖 target / bonus 所需位置。
```

forward 后会保存：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4398`

然后返回 `None`：

```python
return None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`

这和普通 generation 一样，采样在 `sample_tokens()` 中完成。

---

## 14. sample_tokens：从 logits 到接受/拒绝结果

`GPUModelRunner.sample_tokens()` 会取出 `execute_model_state`：

```python
(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
) = self.execute_model_state
self.execute_model_state = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4436` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4450`

如果有 grammar bitmask，会先应用：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

然后采样：

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4458` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4459`

### 14.1 _sample 如何切到 RejectionSampler

当 `spec_decode_metadata` 不为空时，`_sample()` 会走 rejection sampler：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3588` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3598`

如果没有 spec metadata，则走普通 sampler。

所以：

```text
RejectionSampler 不是单独从 EngineCore 调用，
而是 ModelRunner._sample() 在 spec decode 场景下替代普通 Sampler 的路径。
```

---

## 15. RejectionSampler：接受 / 拒绝的核心

`RejectionSampler` 定义在：`code/vllm/vllm/v1/sample/rejection_sampler.py:37`

入口：

```python
def forward(
    self,
    metadata: SpecDecodeMetadata,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:88` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:96`

它的核心步骤：

```text
1. 根据 metadata.bonus_logits_indices 取 bonus logits；
2. 用普通 sampler 采 bonus_token_ids；
3. 根据 metadata.target_logits_indices 取 target logits；
4. 应用 logits processors 和采样约束；
5. 调 rejection_sample()；
6. 可选计算 accepted tokens 的 logprobs；
7. 返回 SamplerOutput。
```

源码对应：

```python
bonus_logits = logits[bonus_logits_indices]
bonus_sampler_output = self.sampler(...)
bonus_token_ids = bonus_sampler_output.sampled_token_ids
raw_target_logits = logits[target_logits_indices]
target_logits = self.apply_logits_processors(...)
target_logits = apply_sampling_constraints(...)
output_token_ids = rejection_sample(...)
return SamplerOutput(...)
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:121` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:197`

### 15.1 draft_probs 可以为空吗

可以。

`draft_probs` 的注释说明：

```text
Can be None if probabilities are not provided, which is the case for ngram spec decode.
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:101` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:105`

这说明：

```text
不同 drafter 提供的信息不一样，
RejectionSampler 需要兼容有概率和无概率两类场景。
```

### 15.2 输出为什么是二维 token 矩阵

`rejection_sample()` 返回的 `output_token_ids` 形状类似：

```text
[batch_size, max_spec_len + 1]
```

其中：

```text
前面若干位置是接受的 draft tokens；
如果全部接受，最后可能是 bonus token；
被拒绝的位置用 PLACEHOLDER_TOKEN_ID 填充。
```

后续 parse / Scheduler update 会把 placeholder 过滤掉，只保留真实生成 token。

---

## 16. sample 后状态更新：接受 token 如何回写

采样后，ModelRunner 会更新 worker 侧状态：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4461` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4463`

这一步负责：

```text
把接受的 sampled tokens 写回 InputBatch / CachedRequestState；
修正 spec decode 下被拒绝 token 对状态造成的影响；
为下一轮 input preparation 准备 prev_sampled_token_ids 等状态。
```

Scheduler 侧也会在 `update_from_output()` 中处理 spec decode 的接受 / 拒绝结果。

相关逻辑可见：`code/vllm/vllm/v1/core/sched/scheduler.py:1547` 起。

它会根据：

```text
scheduled_spec_token_ids
sampled_token_ids
num_sampled_tokens_per_step
```

计算：

```text
num_draft_tokens
num_accepted
num_rejected
```

从而修正：

```text
request.num_computed_tokens
request.output_token_ids
KV cache / routing / encoder cache 等相关状态
```

---

## 17. propose_draft_token_ids：生成下一轮 draft tokens

在 `sample_tokens()` 中，采样和状态更新之后，会进入 draft proposal。

代码里有局部函数：

```python
def propose_draft_token_ids(sampled_token_ids):
    assert spec_decode_common_attn_metadata is not None
    with record_function_or_nullcontext("gpu_model_runner: draft"):
        self._draft_token_ids = self.propose_draft_token_ids(
            scheduler_output,
            sampled_token_ids,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            slot_mappings,
        )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4481` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4495`

真正入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4852`

```python
def propose_draft_token_ids(
    self,
    scheduler_output: "SchedulerOutput",
    sampled_token_ids: torch.Tensor | list[list[int]],
    sampling_metadata: SamplingMetadata,
    ...
):
```

它会读取：

```python
num_spec_tokens_to_schedule = scheduler_output.num_spec_tokens_to_schedule
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4864` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4868`

这说明 dynamic speculative decoding 的 K 是：

```text
Scheduler 决定，ModelRunner 的 proposer 执行。
```

### 17.1 不同 proposer 的分支

`propose_draft_token_ids()` 内部会根据 `spec_config.method` 或 drafter 类型分支：

```text
ngram：基于已有 token 序列查找候选；
draft model：运行小模型生成 draft tokens；
EAGLE / EAGLE3：基于 hidden states 预测后续 token；
Medusa：使用 medusa heads；
DFlash / Gemma4 / Step3.5 MTP：使用对应模型结构或 proposer；
suffix：基于 suffix tree / prompt tree；
extract_hidden_states：抽取 hidden states。
```

如果 proposer 提供 draft probabilities，会被保存：

```python
if hasattr(self.drafter, "take_last_draft_probs"):
    draft_probs = self.drafter.take_last_draft_probs()
    if draft_probs is not None:
        self._draft_probs = draft_probs
        self._draft_prob_req_ids = self.input_batch.req_ids.copy()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5123` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5127`

这些概率会在下一轮 `_sample()` 中由 `_get_spec_decode_draft_probs()` 取出，传给 `RejectionSampler`。

---

## 18. take_draft_token_ids：从 worker 回传给 Scheduler

ModelRunner 生成 draft tokens 后，会缓存到：

```python
self._draft_token_ids
```

EngineCore 不直接访问 ModelRunner，而是通过 Executor → Worker → ModelRunner。

Worker 侧：

```python
def take_draft_token_ids(self):
    return self.model_runner.take_draft_token_ids()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:898` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:899`

ModelRunner 侧：

```python
def take_draft_token_ids(self) -> DraftTokenIds | None:
    if not self.num_spec_tokens or not self._draft_token_req_ids:
        return None
    draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
    return DraftTokenIds(req_ids, draft_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4731` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4735`

这一步完成了：

```text
GPU / worker-local draft token cache
  → CPU list[list[int]]
  → DraftTokenIds(req_ids, draft_token_ids)
  → Executor.take_draft_token_ids()
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
```

---

## 19. spec decode 的两个闭环

### 19.1 验证闭环

这一闭环验证上一轮 draft tokens：

```text
Scheduler.request.spec_token_ids
  → SchedulerOutput.scheduled_spec_decode_tokens
  → InputBatch.spec_token_ids
  → SpecDecodeMetadata
  → target model logits
  → RejectionSampler
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
```

它回答的问题是：

```text
上一轮猜的 tokens，这一轮接受几个？
```

### 19.2 生成闭环

这一闭环生成下一轮 draft tokens：

```text
sampled_token_ids / hidden_states / metadata
  → drafter.propose(...)
  → GPUModelRunner._draft_token_ids
  → take_draft_token_ids()
  → Scheduler.update_draft_token_ids()
  → request.spec_token_ids
```

它回答的问题是：

```text
下一轮要让 target model 验哪些 tokens？
```

两个闭环首尾相接，形成 spec decode 的持续加速。

---

## 20. 和普通 decode 的差异

### 普通 decode

```text
每个请求本轮通常只需要一个新 token；
ModelRunner 只需要对最后位置算 logits；
Sampler 直接采样；
Scheduler update 一个 token。
```

### Spec decode

```text
每个请求本轮可能验证多个 draft tokens；
ModelRunner 需要对 target / bonus 多个位置算 logits；
RejectionSampler 先验 draft，再决定 bonus；
Scheduler update 可能一次接受多个 token；
下一轮还要回传新的 draft tokens。
```

核心变化不是“多采几个 token”这么简单，而是：

```text
token 状态、KV slot 预留、logits indices、sampling、request 账本都要一起适配。
```

---

## 21. 和 KV cache 的关系

spec decode 会影响 KV cache，主要体现在两处。

### 21.1 Scheduler 侧预留 lookahead slots

Scheduler 调用 KVCacheManager 分配 slots 时会传入：

```python
num_lookahead_tokens=self.num_lookahead_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:523` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:528`

这用于为可能验证的 draft tokens 预留 KV 空间。

### 21.2 Rejected tokens 的状态修正

如果 draft tokens 被拒绝，不能把这些 token 当成真实已生成 token。

因此 Scheduler / ModelRunner 都要修正：

```text
num_computed_tokens
output_token_ids
block table / KV cache 使用边界
prev_num_draft_len
num_output_placeholders
```

这也是为什么 spec decode 不是纯 sampler 层功能，它必须深入 Scheduler 的请求状态账本。

---

## 22. 和 Structured Output 的关系

structured output 会影响 spec decode 的两个阶段。

### 22.1 回写 draft tokens 前 grammar validate

在 `Scheduler.update_draft_token_ids()` 中：

```python
if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1911` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1915`

这表示：

```text
不符合 grammar 的 draft tokens 不会进入下一轮 schedule。
```

### 22.2 采样前应用 grammar bitmask

在 `GPUModelRunner.sample_tokens()` 中：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

这表示：

```text
即使 draft tokens 通过了回写校验，最终 target logits 采样仍然会受 grammar 约束。
```

---

## 23. 和 Pipeline Parallel 的关系

`GPUModelRunner` 只有在 last PP rank 创建 drafter 和 rejection sampler：

```python
if self.speculative_config and get_pp_group().is_last_rank:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:545`

原因是：

```text
非 last PP rank 没有最终 logits，
也不负责 token sampling，
因此不能完成 rejection sampling 或 draft proposal。
```

在 PP 场景中：

```text
前面 rank：执行部分模型层，返回 IntermediateTensors；
last rank：计算 logits，执行 sampler / rejection sampler / proposer。
```

---

## 24. 和 async scheduling 的关系

在同步 scheduling 中：

```text
EngineCore.post_step()
  → take_draft_token_ids()
  → Scheduler.update_draft_token_ids()
```

在 async scheduling 中，EngineCore 注释说明：

```python
# When using async scheduling we can't get draft token ids in advance,
# so we update draft token ids in the worker process and don't
# need to update draft token ids here.
```

位置：`code/vllm/vllm/v1/engine/core.py:510` 到 `code/vllm/vllm/v1/engine/core.py:514`

这表示 async scheduling 下 draft token 更新路径会更靠近 worker 侧，避免 EngineCore 在错误时机提前取 draft tokens。

ModelRunner 里也有 async spec token 的更新逻辑，例如：

```python
if self.use_async_scheduling and self._draft_token_req_ids is not None:
    draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
    self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3588` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3590`

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
维护 request.spec_token_ids；
根据 num_tokens_with_spec 调度 token；
为 spec decode 预留 lookahead KV slots；
构造 SchedulerOutput.scheduled_spec_decode_tokens；
处理 structured output 对 draft tokens 的校验；
根据 ModelRunnerOutput 修正 accepted / rejected token 状态。
```

### 25.3 Executor / Worker 负责

```text
Executor：转发 execute_model / sample_tokens / take_draft_token_ids；
Worker：把这些调用委托给 ModelRunner。
```

它们不理解 rejection sampling 的细节。

### 25.4 ModelRunner 负责

```text
创建 drafter；
创建 RejectionSampler；
把 scheduled spec tokens 写入 InputBatch；
构造 SpecDecodeMetadata；
执行 target model forward；
调用 RejectionSampler；
更新 worker 侧 batch 状态；
调用 proposer 生成下一轮 draft tokens；
缓存并回传 DraftTokenIds。
```

### 25.5 RejectionSampler 负责

```text
根据 target logits、draft token ids、draft probs 和 sampling metadata，
决定每个请求接受多少 draft tokens，
以及是否采 bonus token。
```

一句话边界：

```text
Scheduler 负责“哪些 draft tokens 进入本轮”，ModelRunner 负责“如何验证并生成下一轮”，RejectionSampler 负责“接受几个”。
```

---

## 26. 一个完整例子：draft model spec decode

假设请求已经完成 prefill，上一轮 drafter 给请求生成了 4 个 draft tokens。

### 第 N 轮结束后

ModelRunner：

```text
1. sample_tokens() 得到真实 sampled_token_ids；
2. propose_draft_token_ids() 调用 draft model；
3. 得到 [t1, t2, t3, t4]；
4. 缓存在 self._draft_token_ids；
5. take_draft_token_ids() 返回 DraftTokenIds(req_ids, draft_token_ids)。
```

EngineCore / Scheduler：

```text
1. EngineCore.post_step() 调 Executor.take_draft_token_ids()；
2. Scheduler.update_draft_token_ids()；
3. request.spec_token_ids = [t1, t2, t3, t4]。
```

### 第 N+1 轮 schedule

Scheduler：

```text
1. 发现 request.spec_token_ids 非空；
2. 根据 token budget / max_model_len / KV slots 决定能调度几个；
3. 写入 SchedulerOutput.scheduled_spec_decode_tokens；
4. 清空 request.spec_token_ids。
```

### 第 N+1 轮 execute

ModelRunner：

```text
1. _update_states() 把 scheduled_spec_decode_tokens 写入 InputBatch；
2. _prepare_inputs() 构造 SpecDecodeMetadata；
3. _model_forward() 用 target model 计算 target / bonus logits；
4. sample_tokens() 调 RejectionSampler；
5. 假设接受 t1、t2，拒绝 t3；
6. ModelRunnerOutput.sampled_token_ids 返回 [t1, t2, new_token] 或对应采样结果；
7. proposer 再生成下一批 draft tokens。
```

Scheduler update：

```text
1. update_from_output() 写入接受的 tokens；
2. 修正 rejected draft tokens 对 num_computed_tokens / KV 状态的影响；
3. 请求继续进入下一轮。
```

---

## 27. 容易混淆的点

### 27.1 spec_token_ids 是最终输出吗？

不是。

```text
spec_token_ids 只是候选 token；
只有被 target model + rejection sampler 接受后，才会成为 output_token_ids。
```

### 27.2 scheduled_spec_decode_tokens 和 _draft_token_ids 是同一个东西吗？

不是。

```text
scheduled_spec_decode_tokens：本轮要验证的 draft tokens，来自 Scheduler；
_draft_token_ids：本轮执行后 proposer 生成的下一轮 draft tokens，保存在 ModelRunner。
```

### 27.3 RejectionSampler 是否替代普通 Sampler？

在 spec decode 验证阶段，是。

```text
普通 decode：Sampler(logits)；
spec decode：RejectionSampler(metadata, draft_probs, logits, sampling_metadata)。
```

但 RejectionSampler 内部仍会调用普通 `Sampler` 来采 bonus token。

### 27.4 draft model 生成的 token 一定会被接受吗？

不会。

```text
draft tokens 必须经过 target logits 验证；
不符合 target 分布、sampling 约束或 grammar 的 token 会被拒绝。
```

### 27.5 spec decode 为什么需要 bonus token？

如果所有 draft tokens 都被接受，系统还可以从最后一个位置的 target logits 再采一个 token。

这让一次 target model forward 最多产出：

```text
num_draft_tokens + 1
```

个 token。

### 27.6 spec decode 是否只影响 ModelRunner？

不是。

它至少影响：

```text
Config：SpeculativeConfig；
Scheduler：token 调度、KV lookahead、request.spec_token_ids；
SchedulerOutput：scheduled_spec_decode_tokens；
InputBatch：spec_token_ids；
Attention metadata：spec decode 相关字段；
Sampler：RejectionSampler；
EngineCore：take_draft_token_ids 回传闭环。
```

---

## 28. 从“回答问题”的角度总结

如果要问：

```text
Speculative Decoding 在 vLLM V1 里负责什么？
```

可以回答：

```text
Speculative Decoding 是 vLLM V1 的跨层生成加速协议。

它通过 SpeculativeConfig 选择 drafter 方法和每轮 draft token 数；
Scheduler 为 draft tokens 维护 request.spec_token_ids，调度时把它们放进 SchedulerOutput.scheduled_spec_decode_tokens，并为它们预留 KV lookahead slots；
GPUModelRunner 把这些 draft tokens 写入 InputBatch，构造 SpecDecodeMetadata，让 target model 一次 forward 计算 target logits 和 bonus logits；
RejectionSampler 根据 target logits、draft probs、采样参数和 grammar 约束决定接受多少 draft tokens；
随后 ModelRunner 再调用 proposer 生成下一轮 draft tokens，通过 take_draft_token_ids() 回传给 Scheduler，形成下一轮循环。
```

职责关系可以概括为：

```text
SpeculativeConfig：决定启用什么 speculative 方法；
Scheduler：管理 draft token 的请求账本和调度；
ModelRunner：执行 target 验证和下一轮 proposal；
RejectionSampler：决定接受 / 拒绝；
EngineCore / Executor / Worker：串起跨层调用和结果回传。
```

---

## 29. 最关键流程图

```text
初始化阶段

VllmConfig.speculative_config
  → SpeculativeConfig
  → Scheduler.num_spec_tokens / num_lookahead_tokens
  → GPUModelRunner.drafter
  → GPUModelRunner.rejection_sampler
```

```text
每轮执行阶段

EngineCore.step()
  → Scheduler.schedule()
      ├─ request.spec_token_ids
      ├─ allocate_slots(num_lookahead_tokens)
      ├─ scheduled_spec_decode_tokens
      └─ num_spec_tokens_to_schedule
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      ├─ _update_states()
      │    └─ InputBatch.update_req_spec_token_ids()
      ├─ _prepare_inputs()
      │    └─ SpecDecodeMetadata
      ├─ _build_attention_metadata(use_spec_decode=True)
      ├─ _model_forward()
      ├─ compute_logits()
      └─ save execute_model_state
  → GPUModelRunner.sample_tokens()
      ├─ apply_grammar_bitmask()
      ├─ _sample()
      │    └─ RejectionSampler.forward()
      ├─ _update_states_after_model_execute()
      ├─ propose_draft_token_ids()
      └─ ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCore.post_step()
      └─ take_draft_token_ids()
          └─ Scheduler.update_draft_token_ids()
              └─ request.spec_token_ids for next step
```

---

## 30. 最关键对象关系

```text
SpeculativeConfig
  spec decode 的配置入口，决定方法、draft model、K、rejection sampling 策略。

Request.spec_token_ids
  Scheduler 请求状态里暂存的下一轮 draft tokens。

SchedulerOutput.scheduled_spec_decode_tokens
  本轮真正被调度给 worker 验证的 draft tokens。

InputBatch.spec_token_ids
  worker 侧 batch row 上的 spec token 状态。

SpecDecodeMetadata
  RejectionSampler 所需的 draft token ids、target logits indices、bonus logits indices。

RejectionSampler
  用 target logits 和 draft probs 决定接受 / 拒绝。

DraftTokenIds
  ModelRunner 回传给 Scheduler 的下一轮 draft tokens。

ModelRunnerOutput.sampled_token_ids
  本轮最终被接受并输出给 Scheduler 的 token ids。
```

---

## 31. 最小心智模型

如果只记一条主线，可以记：

```text
Scheduler 保存 drafter 猜出来的 token，
ModelRunner 让 target model 一次性验证这些 token，
RejectionSampler 决定接受几个，
ModelRunner 再猜下一批 token，
EngineCore 把下一批 token 交回 Scheduler。
```

再压缩成一句话：

```text
Spec decode 用“猜测 + 验证 + 回写”的循环，把多 token 生成压进尽量少的 target model forward 里。
```
