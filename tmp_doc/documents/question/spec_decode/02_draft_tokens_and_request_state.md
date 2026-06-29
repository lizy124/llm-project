# 02. Draft tokens 从哪里来，如何挂在 request 状态上？

源码位置：

- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/spec_decode/`

本问题关注：draft tokens 的来源、在 Scheduler 侧 `Request` 上的暂存方式、如何进入 `SchedulerOutput.scheduled_spec_decode_tokens`、Worker / ModelRunner 如何把它们放进 `InputBatch`，以及 target model 验证后如何把 accepted tokens 写回真正的 request output 状态。

---

## 1. 一句话回答

Draft tokens 是 **drafter / proposer 对下一段输出的预测**，在 vLLM V1 中先暂存在 Scheduler 侧 `Request.spec_token_ids` 上。

```text
Worker / ModelRunner 产生 draft tokens
  → EngineCore 取回 DraftTokenIds
  → Scheduler.update_draft_token_ids()
  → 写入 Request.spec_token_ids
  → 下一轮 Scheduler.schedule() 把它们调度为 scheduled_spec_decode_tokens
  → Worker.update_states() 写入 InputBatch.spec_token_ids / token_ids_cpu
  → target model 验证
  → RejectionSampler 输出 accepted / replacement / bonus tokens
  → Scheduler.update_from_output() 把 accepted tokens append 到 Request.output_token_ids
```

关键边界：

```text
Request.spec_token_ids：
  只是“待验证候选 token”。

Request.output_token_ids：
  是“target model 验证后正式接受的输出 token”。
```

所以 draft token 可以挂在 request 状态上，但它还不是最终输出。

---

## 2. 总体对象关系

```text
GPUModelRunner.propose_draft_token_ids()
  → DraftTokenIds(req_ids, draft_token_ids)
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
  → Scheduler.schedule()
  → SchedulerOutput.scheduled_spec_decode_tokens
  → GPUModelRunner._update_states()
  → InputBatch.update_req_spec_token_ids()
  → SpecDecodeMetadata / RejectionSampler
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.append_output_token_ids()
```

可以按两段理解：

```text
第一段：产生 draft
  target 本轮采样结束后，drafter 预测下一轮 draft tokens。

第二段：消费 draft
  下一轮调度时，把上一轮预测的 draft tokens 带给 target model 验证。
```

---

## 3. Request 里哪些字段和 draft 有关

`Request` 定义在：`request.py:59`

和 spec decode 直接相关的字段：

```python
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
```

位置：`request.py:152` 到 `request.py:153`

另外，async scheduling 还会用到：

```python
self.num_output_placeholders = 0
self.async_tokens_to_discard = 0
```

位置：`request.py:140` 到 `request.py:142`

这些字段表达的是 Scheduler 侧的请求进度：

| 字段 | 含义 |
|---|---|
| `spec_token_ids` | 当前 request 上暂存的 draft tokens，等待下一轮 target 验证 |
| `num_computed_tokens` | Scheduler 认为已经被模型计算过的 token 数，可能临时包含待修正的 spec token |
| `num_output_placeholders` | async scheduling 下为未来输出预留的位置 |
| `async_tokens_to_discard` | async 场景下需要丢弃的 token 数 |

---

## 4. `num_tokens` 和 `num_tokens_with_spec` 的区别

`Request.num_tokens`：

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)
```

位置：`request.py:246` 到 `request.py:248`

`Request.num_tokens_with_spec`：

```python
@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

位置：`request.py:250` 到 `request.py:252`

区别是：

```text
num_tokens：
  prompt tokens + 已正式接受的 output tokens。

num_tokens_with_spec：
  prompt tokens + 已正式接受的 output tokens + 待验证 draft tokens。
```

这正是 Scheduler 能把 draft tokens 纳入调度的关键。

Scheduler 的注释也明确说明：

```text
num_tokens_with_spec =
  len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
```

位置：`scheduler.py:389` 到 `scheduler.py:398`

---

## 5. draft tokens 从哪里来

Draft tokens 不是在 Scheduler 中凭空产生的，而是在 Worker / ModelRunner 侧由不同 drafter / proposer 产生。

入口在 `GPUModelRunner.sample_tokens()` 中：

```python
def propose_draft_token_ids(sampled_token_ids):
    self._draft_token_ids = self.propose_draft_token_ids(...)
    self._copy_draft_token_ids_to_cpu(scheduler_output)
```

位置：`gpu_model_runner.py:4481` 到 `gpu_model_runner.py:4495`

也就是说：

```text
target model 本轮 forward + sample 完成
  → 得到 sampled_token_ids
  → drafter 基于 sampled token、hidden states、token_ids_cpu 等信息
  → 预测下一批 draft_token_ids
```

这里的 `sampled_token_ids` 是 target model 本轮已经采样出来的 token。drafter 用它作为下一轮预测的上下文。

---

## 6. proposer 的不同来源

统一入口是：

```python
def propose_draft_token_ids(...)
```

位置：`gpu_model_runner.py:4852`

它会根据 speculative config 走不同分支。

### 6.1 ngram proposer

```python
if spec_config.method == "ngram":
    draft_token_ids = self.drafter.propose(
        num_spec_tokens_to_schedule,
        sampled_token_ids,
        self.input_batch.num_tokens_no_spec,
        self.input_batch.token_ids_cpu,
        slot_mappings=slot_mappings,
    )
```

位置：`gpu_model_runner.py:4870` 到 `gpu_model_runner.py:4881`

`NgramProposer.propose()` 定义在：`ngram_proposer.py:135`

它的输入核心是：

```text
sampled_token_ids：本轮 target 采样结果
num_tokens_no_spec：每个请求不含 spec token 的当前长度
token_ids_cpu：请求历史 token 矩阵
num_speculative_tokens：最多要预测几个 token
```

返回值是：

```text
list[list[int]]
每个 request 一组 draft token ids。
```

如果某个请求不适合 ngram draft，会返回空列表。

### 6.2 suffix / custom_class / medusa

这些分支同样通过 proposer 返回 `draft_token_ids`：

```text
custom_class：用户自定义 proposer
suffix：suffix decoding proposer
medusa：Medusa head 基于 hidden states 预测 draft
```

相关分支位置：

- `custom_class`：`gpu_model_runner.py:4882`
- `suffix`：`gpu_model_runner.py:4928`
- `medusa`：`gpu_model_runner.py:4937`

### 6.3 EAGLE / DFlash / draft model / Gemma4

这些方法通常需要：

```text
sampled_token_ids
target_token_ids
target_positions
target_hidden_states
common_attn_metadata
sampling_metadata
```

最终调用：

```python
draft_token_ids = self.drafter.propose(...)
```

位置：`gpu_model_runner.py:5110` 到 `gpu_model_runner.py:5122`

这类 drafter 往往更接近“轻量模型或辅助 head 预测下一段 token”，而 ngram 更像“从已有上下文中查找可复用片段”。

---

## 7. DraftTokenIds 是什么

Worker 返回给 Scheduler 的 draft token 载体是 `DraftTokenIds`。

定义在：`outputs.py:310`

```python
@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]
```

位置：`outputs.py:310` 到 `outputs.py:315`

它的含义是：

```text
req_ids[i]
  对应
 draft_token_ids[i]
```

例如：

```text
DraftTokenIds(
  req_ids=["r1", "r2"],
  draft_token_ids=[
    [101, 102, 103],
    [201, 202],
  ],
)
```

表示：

```text
r1 的 draft tokens 是 [101, 102, 103]
r2 的 draft tokens 是 [201, 202]
```

---

## 8. Worker 如何把 draft tokens 暴露给 EngineCore

`GPUModelRunner` 会把 proposer 的结果暂存在：

```python
self._draft_token_ids
self._draft_token_req_ids
```

相关字段位置：`gpu_model_runner.py:834` 到 `gpu_model_runner.py:838`

然后通过：

```python
def take_draft_token_ids(self) -> DraftTokenIds | None:
```

位置：`gpu_model_runner.py:4731`

逻辑是：

```python
if not self.num_spec_tokens or not self._draft_token_req_ids:
    return None
draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
return DraftTokenIds(req_ids, draft_token_ids)
```

位置：`gpu_model_runner.py:4731` 到 `gpu_model_runner.py:4735`

这说明：

```text
ModelRunner 不直接修改 Scheduler Request。
它只提供 DraftTokenIds，由 EngineCore 转交给 Scheduler。
```

---

## 9. draft tokens 什么时候从 GPU 拷到 CPU

`_copy_draft_token_ids_to_cpu()` 负责把 GPU 上的 draft token tensor 拷到 CPU buffer。

入口：`gpu_model_runner.py:4737`

关键逻辑：

```python
if self.use_async_scheduling and not (
    scheduler_output.has_structured_output_requests
    or self.input_batch.sampling_metadata.output_token_ids
):
    return

self._draft_token_req_ids = self.input_batch.req_ids.copy()
```

位置：`gpu_model_runner.py:4743` 到 `gpu_model_runner.py:4751`

含义：

```text
同步 scheduling：
  通常需要把 draft tokens 拷回 CPU，供 Scheduler.update_draft_token_ids() 使用。

async scheduling：
  如果不需要 structured output / penalties / bad_words 等 CPU 侧信息，
  可以不急着 D2H 拷贝，避免同步开销。
```

如果确实需要拷贝，会在专门的 copy stream 上异步复制：

```python
self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens].copy_(
    draft_token_ids, non_blocking=True
)
```

位置：`gpu_model_runner.py:4762` 到 `gpu_model_runner.py:4772`

---

## 10. 同步路径：EngineCore.post_step() 写回 Scheduler Request

普通同步调度路径中，EngineCore 在每一步结束后调用 `post_step()`。

入口：`engine/core.py:510`

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`engine/core.py:510` 到 `engine/core.py:517`

这一步完成：

```text
Worker draft output
  → EngineCore.take_draft_token_ids()
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

注意这里是 `post_step()`，也就是：

```text
本轮 target model 已经执行完；
产生的 draft tokens 是给下一轮 schedule 使用的。
```

---

## 11. Scheduler.update_draft_token_ids() 如何写入 Request

入口：`scheduler.py:1895`

```python
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
```

核心流程：

```python
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

位置：`scheduler.py:1895` 到 `scheduler.py:1915`

这里有几个重要规则。

### 11.1 请求已经结束就跳过

```text
draft tokens 是异步产生或 step 后产生的。
在它们回到 Scheduler 时，请求可能已经 finished / aborted。
这种情况下直接跳过。
```

### 11.2 prefill chunk 不使用 draft tokens

```text
如果 request.is_prefill_chunk 为 True，说明请求还在 prefill/chunked prefill 阶段。
这时 draft tokens 不应该参与 decode 验证。
```

所以 Scheduler 会清空旧的 `request.spec_token_ids`。

### 11.3 structured output 会提前校验 draft tokens

如果请求有 grammar 约束：

```python
spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
```

位置：`scheduler.py:1911` 到 `scheduler.py:1915`

这意味着：

```text
不符合 grammar 的 draft tokens 可能会被提前裁掉，
避免下一轮把非法候选 token 送去 target 验证。
```

---

## 12. Scheduler.schedule() 如何消费 Request.spec_token_ids

Scheduler 的主循环入口：`scheduler.py:387`

它计算每个 running request 本轮需要调度的 token 数：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`scheduler.py:462` 到 `scheduler.py:466`

这里 `num_tokens_with_spec` 已经包含：

```text
prompt + accepted output + pending draft tokens
```

因此，如果 request 上有 `spec_token_ids`，Scheduler 会把它们纳入本轮待计算范围。

---

## 13. scheduled_spec_decode_tokens 是如何生成的

在 Scheduler 成功调度一个 running request 后，会处理 spec decode token：

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

位置：`scheduler.py:581` 到 `scheduler.py:597`

这个公式可以理解为：

```text
num_scheduled_spec_tokens
  = 本轮调度范围里落在 draft token 区间的 token 数
```

为什么要裁剪？

```text
Scheduler 可能因为 token budget、max_model_len、encoder budget、KV block 等限制，
本轮不能调度所有 draft tokens。
所以只能把本轮实际调度到的 draft 前缀放入 scheduled_spec_decode_tokens。
```

为什么调度后清空 `request.spec_token_ids`？

```text
因为这些 draft tokens 已经被转移到本轮 SchedulerOutput。
如果下一轮还需要新的 draft tokens，会由 update_draft_token_ids() 再写入。
避免重复调度旧 draft。
```

---

## 14. SchedulerOutput 中的 scheduled_spec_decode_tokens

`SchedulerOutput` 定义在：`output.py:180`

字段：

```python
# req_id -> spec_token_ids
# If a request does not have any spec decode tokens, it will not be
# included in the dictionary.
scheduled_spec_decode_tokens: dict[str, list[int]]
```

位置：`output.py:197` 到 `output.py:200`

它的含义是：

```text
本轮真正要交给 Worker / target model 验证的 draft tokens。
```

注意：

```text
Request.spec_token_ids：
  Scheduler 侧跨 step 暂存。

SchedulerOutput.scheduled_spec_decode_tokens：
  本轮已经确定要执行的 draft tokens。
```

这两个不是同一个生命周期。

---

## 15. SchedulerOutput 何时被构造

在 `schedule()` 末尾：

```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    ...
)
```

位置：`scheduler.py:1057` 到 `scheduler.py:1074`

同时，Scheduler 还会设置：

```python
num_spec_tokens_to_schedule=num_spec_tokens_to_schedule
```

位置：`scheduler.py:1050` 到 `scheduler.py:1073`

这个字段用于告诉下一次 drafter：

```text
下一轮每个请求最多应该 draft 几个 token。
```

如果开启 dynamic speculative decoding，它可能根据 batch size 动态调整。

---

## 16. schedule 后 num_computed_tokens 会先前进

Scheduler 在 `_update_after_schedule()` 中推进请求进度。

入口：`scheduler.py:1128`

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
```

位置：`scheduler.py:1138` 到 `scheduler.py:1141`

注释说明：

```text
如果某些 token，比如 spec tokens，后续被 rejected，
num_computed_tokens 会在 update_from_output 中再修正。
```

位置：`scheduler.py:1131` 到 `scheduler.py:1137`

这点非常关键：

```text
调度时先乐观认为本轮 scheduled tokens 都会被计算；
输出回收时再根据 rejected draft tokens 回退 num_computed_tokens。
```

---

## 17. Worker 如何接收 scheduled_spec_decode_tokens

Worker / ModelRunner 在 `_update_states()` 中读取：

```python
req_data = scheduler_output.scheduled_cached_reqs
scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
```

位置：`gpu_model_runner.py:1261` 到 `gpu_model_runner.py:1264`

对已经在 `InputBatch` 中的请求，会调用：

```python
self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
```

位置：`gpu_model_runner.py:1430` 到 `gpu_model_runner.py:1431`

对新加入或恢复的请求，也会在 `add_request()` 后调用：

```python
self.input_batch.add_request(request)
self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)
```

位置：`gpu_model_runner.py:1438` 到 `gpu_model_runner.py:1442`

所以 Worker 侧的状态同步顺序是：

```text
SchedulerOutput
  → CachedRequestState
  → InputBatch.add_request / update existing row
  → InputBatch.update_req_spec_token_ids()
```

---

## 18. InputBatch 中 spec token 存在哪里

`InputBatch` 初始化时有两个 spec decode 相关结构：

```python
self.num_accepted_tokens_cpu_tensor = torch.ones(...)
self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()
```

位置：`gpu_input_batch.py:238` 到 `gpu_input_batch.py:242`

以及：

```python
self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]
```

位置：`gpu_input_batch.py:284` 到 `gpu_input_batch.py:285`

它们分别用于：

```text
num_accepted_tokens_cpu：
  记录每个请求本轮接受了多少 token，hybrid / Mamba 等场景会用。

spec_token_ids：
  当前 batch 中每个 req_index 对应的 draft token 列表。
```

---

## 19. update_req_spec_token_ids() 做什么

入口：`gpu_input_batch.py:483`

```python
def update_req_spec_token_ids(
    self, request: CachedRequestState, scheduled_spec_tokens: dict[str, list[int]]
) -> None:
```

核心逻辑：

```python
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

位置：`gpu_input_batch.py:483` 到 `gpu_input_batch.py:508`

它做了四件事：

```text
1. 清空当前 req_index 上旧的 spec_token_ids；
2. 从 SchedulerOutput 取本轮 scheduled spec tokens；
3. 记录 request.prev_num_draft_len；
4. 把 draft tokens 写入 token_ids_cpu 的 output 尾部之后。
```

`start_index` 是：

```text
num_tokens_no_spec[req_index]
```

也就是：

```text
prompt + accepted output 的末尾。
```

因此 Worker 侧 token 布局变成：

```text
token_ids_cpu row:
  [prompt tokens][accepted output tokens][scheduled draft tokens]
```

---

## 20. CachedRequestState.prev_num_draft_len 的作用

`CachedRequestState` 定义在：`gpu_input_batch.py:33`

字段：

```python
# Used when both async_scheduling and spec_decode are enabled.
prev_num_draft_len: int = 0
```

位置：`gpu_input_batch.py:59` 到 `gpu_input_batch.py:60`

它记录上一轮放进 batch 的 draft token 数。

在 async scheduling + spec decode 下，Worker 可能需要先乐观推进 token 状态，再等真实采样结果回来修正。

相关逻辑在：`gpu_model_runner.py:1292`

```python
if req_state.prev_num_draft_len and self.use_async_scheduling:
    optimistic_num_accepted = req_state.prev_num_draft_len
    req_state.output_token_ids.extend([-1] * optimistic_num_accepted)
    deferred_spec_decode_corrections.append(...)
```

位置：`gpu_model_runner.py:1292` 到 `gpu_model_runner.py:1317`

含义：

```text
上一轮有 draft tokens；
async scheduling 下一轮可能已经开始准备；
Worker 先假设这些 draft 都 accepted，用 -1 占位扩展 output_token_ids；
等拿到真实 accepted 数量后再修正。
```

后续修正在：`gpu_model_runner.py:1463` 到 `gpu_model_runner.py:1493`

---

## 21. draft tokens 如何参与 target model 输入

`update_req_spec_token_ids()` 只是把 draft token 写进 CPU 侧 batch 状态。

后续 `_prepare_inputs()` 会基于：

```text
InputBatch.token_ids_cpu
InputBatch.num_tokens_no_spec
InputBatch.spec_token_ids
SchedulerOutput.num_scheduled_tokens
SpecDecodeMetadata
```

构造：

```text
input_ids
positions
logits_indices
attention metadata
spec decode metadata
```

此时 target model 看到的是：

```text
已接受上下文 + draft tokens 对应的验证位置
```

它不是“直接采纳 draft”，而是为每个 draft 位置计算 target logits，让 `RejectionSampler` 决定接受多少。

---

## 22. target 验证后，draft 如何变成 output tokens

target model 验证和采样后，`ModelRunnerOutput.sampled_token_ids` 会返回每个请求本轮真正生成 / 接受的 token。

Scheduler 在 `update_from_output()` 中处理输出。

相关片段位置：`scheduler.py:1518` 起

它先取出：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
```

位置：`scheduler.py:1542` 到 `scheduler.py:1549`

如果本轮有 scheduled draft tokens：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`scheduler.py:1550` 到 `scheduler.py:1556`

这里的含义是：

```text
generated_token_ids：
  RejectionSampler 返回的本轮有效输出 token。

num_sampled：
  target 本轮常规采样 token 数，通常是 1。

num_accepted：
  generated_token_ids 中超过常规采样 token 的部分，视为 accepted draft tokens。

num_rejected：
  本轮 scheduled draft tokens 中没有被接受的部分。
```

---

## 23. rejected draft 如何修正 num_computed_tokens

如果有 rejected draft tokens：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`scheduler.py:1557` 到 `scheduler.py:1567`

为什么要回退？

```text
Scheduler 在 _update_after_schedule() 中已经乐观推进了 num_computed_tokens。
但 rejected draft tokens 不应该算作真正前进的上下文。
所以 update_from_output() 必须把 rejected 的数量减回来。
```

这保证：

```text
Request.num_computed_tokens
KV cache 进度
Request.output_token_ids
下一轮调度范围
```

保持一致。

---

## 24. accepted tokens 如何写入 Request.output_token_ids

真正写入 output 的入口是：

```python
def _update_request_with_output(
    self, request: Request, new_token_ids: list[int]
) -> tuple[list[int], bool]:
```

位置：`scheduler.py:1848`

核心逻辑：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]
        break
```

位置：`scheduler.py:1848` 到 `scheduler.py:1864`

而 `Request.append_output_token_ids()` 会同时更新：

```text
_request._output_token_ids
_request._all_token_ids
block hashes
```

位置：`request.py:224` 到 `request.py:235`

所以 accepted draft tokens 的最终归宿是：

```text
Request.output_token_ids
Request.all_token_ids
```

不是 `Request.spec_token_ids`。

---

## 25. 为什么 scheduled draft 不等于 accepted draft

`scheduled_spec_decode_tokens` 只表示：

```text
这些 draft tokens 被送去 target model 验证。
```

它不表示：

```text
这些 draft tokens 已经被接受。
```

验证后可能出现三种情况：

```text
1. 全部接受：
   accepted draft tokens 全部进入 output；可能还有 bonus token。

2. 部分接受：
   前缀 accepted，遇到第一个 rejected 后停止；可能采样 replacement token。

3. 全部拒绝：
   draft tokens 都不进入 output；只保留 target 采样出的 replacement token。
```

因此生命周期必须区分：

```text
proposed draft
  → scheduled draft
  → accepted output
```

---

## 26. async scheduling 下有什么不同

同步路径中：

```text
本轮执行完成
  → take_draft_token_ids()
  → update_draft_token_ids()
  → 下一轮 schedule 使用 Request.spec_token_ids
```

async batch queue 路径中，EngineCore 注释明确说明：

```text
When using async scheduling we can't get draft token ids in advance,
so we update draft token ids in the worker process and don't
need to update draft token ids here.
```

位置：`engine/core.py:510` 到 `engine/core.py:513`

也就是说，async scheduling 不走普通 `post_step()` 写回 request 的节奏。

在 batch queue 中，如果 deferred scheduler output 需要等 draft tokens 才能做 grammar bitmask，会走：

```python
draft_token_ids = self.model_executor.take_draft_token_ids()
if draft_token_ids is not None:
    self.scheduler.update_draft_token_ids_in_output(
        draft_token_ids, deferred_scheduler_output
    )
```

位置：`engine/core.py:612` 到 `engine/core.py:623`

这不是写 `Request.spec_token_ids`，而是直接修正已经生成的：

```text
SchedulerOutput.scheduled_spec_decode_tokens
```

---

## 27. update_draft_token_ids_in_output() 做什么

入口：`scheduler.py:1917`

它用于 async / deferred 场景：

```python
def update_draft_token_ids_in_output(
    self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
) -> None:
```

核心逻辑：

```python
placeholder_spec_tokens = sched_spec_tokens.get(req_id)
if not placeholder_spec_tokens:
    continue

orig_num_spec_tokens = len(placeholder_spec_tokens)
del spec_token_ids[orig_num_spec_tokens:]

if self.structured_output_manager.should_advance(request):
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)

num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
if num_invalid_tokens:
    spec_token_ids.extend([-1] * num_invalid_tokens)
    num_invalid_spec_tokens[req_id] = num_invalid_tokens

sched_spec_tokens[req_id] = spec_token_ids
scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
```

位置：`scheduler.py:1917` 到 `scheduler.py:1953`

它解决的问题是：

```text
async scheduling 已经先创建了 SchedulerOutput；
但真实 draft tokens 可能稍后才从 Worker 拿到；
所以需要在 sample_tokens / grammar bitmask 前，把 SchedulerOutput 里的占位 draft tokens 修正成真实值。
```

如果 structured output 校验裁掉了一些 token，会用 `-1` padding 回原长度。

原因：

```text
SchedulerOutput 已经决定了本轮 scheduled spec token 数；
后续 tensor shape / logits layout 需要保持一致；
无效 draft 用 -1 表示，后续跳过或屏蔽。
```

---

## 28. structured output 对 draft state 的影响

structured output 会在两个地方影响 draft tokens。

### 28.1 写入 Request.spec_token_ids 前校验

在 `update_draft_token_ids()` 中：

```python
spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

位置：`scheduler.py:1911` 到 `scheduler.py:1915`

这会直接减少挂在 request 上的 draft token 数。

### 28.2 async output 修正时校验

在 `update_draft_token_ids_in_output()` 中：

```python
spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
```

位置：`scheduler.py:1940` 到 `scheduler.py:1945`

如果变短，会 padding `-1` 并记录：

```python
num_invalid_spec_tokens[req_id] = num_invalid_tokens
```

位置：`scheduler.py:1946` 到 `scheduler.py:1953`

这些 invalid token 会影响 acceptance rate 统计和 grammar bitmask 计算。

---

## 29. prefill chunk 为什么不能使用 draft tokens

在 `update_draft_token_ids()` 中：

```python
if request.is_prefill_chunk:
    if request.spec_token_ids:
        request.spec_token_ids = []
    continue
```

位置：`scheduler.py:1905` 到 `scheduler.py:1909`

原因是：

```text
spec decode 针对 decode 阶段“预测未来输出 token”；
prefill/chunked prefill 阶段还在处理 prompt 上下文；
这时 draft output token 没有稳定的验证语义。
```

所以 draft tokens 只应该在请求不再是 prefill chunk 后参与调度。

---

## 30. preemption 时 draft state 如何处理

请求被 preempt 时：

```python
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
if request.spec_token_ids:
    request.spec_token_ids = []
```

位置：`scheduler.py:1117` 到 `scheduler.py:1120`

含义：

```text
被抢占后，Scheduler 会重置 computed 进度；
旧的 draft tokens 依赖抢占前的上下文和 KV 状态；
继续使用可能不安全，所以直接清空。
```

---

## 31. draft token 生命周期例子

假设一个请求当前状态：

```text
prompt = [P1, P2]
output = [A, B]
spec_token_ids = []
num_computed_tokens = 4
```

### 31.1 本轮 target 正常采样

target 采样出：

```text
C
```

drafter 预测：

```text
[D, E, F]
```

EngineCore.post_step() 后：

```text
output = [A, B, C]
spec_token_ids = [D, E, F]
num_computed_tokens = 5
```

注意：

```text
D/E/F 还没有进入 output。
```

### 31.2 下一轮 schedule

Scheduler 看到：

```text
num_tokens = prompt + output = 2 + 3 = 5
num_tokens_with_spec = 5 + 3 = 8
num_computed_tokens = 5
```

所以本轮需要追到 8：

```text
num_new_tokens = 3
scheduled_spec_decode_tokens = [D, E, F]
request.spec_token_ids = []
```

### 31.3 target 验证后接受两个

假设 RejectionSampler 返回：

```text
[D, E, X]
```

其中：

```text
D/E 是 accepted draft prefix
F 被 rejected
X 是 replacement token
```

Scheduler.update_from_output() 后：

```text
output = [A, B, C, D, E, X]
num_computed_tokens 回退 rejected 数后保持一致
spec_token_ids = []
```

---

## 32. 和 KV cache / num_computed_tokens 的关系

draft tokens 会临时影响：

```text
num_tokens_with_spec
num_scheduled_tokens
num_computed_tokens
KV block allocation
```

但 rejected draft tokens 不能永久推进请求状态。

所以 vLLM 的策略是：

```text
调度时：
  先把 scheduled draft tokens 算进 num_scheduled_tokens，
  并推进 num_computed_tokens。

输出回收时：
  根据 RejectionSampler 的 accepted 数量，
  把 rejected token 数从 num_computed_tokens 中减掉。
```

这就是为什么 `scheduler.py:1131` 到 `scheduler.py:1137` 的注释强调：

```text
spec tokens rejected later 时，num_computed_tokens 会在 update_from_output 中调整。
```

---

## 33. 容易混淆的点

### 33.1 `spec_token_ids` 是不是输出 token？

不是。

```text
Request.spec_token_ids：待验证候选。
Request.output_token_ids：已接受输出。
```

只有 `Scheduler.update_from_output()` 调用 `_update_request_with_output()` 后，token 才会进入 `output_token_ids`。

### 33.2 `scheduled_spec_decode_tokens` 是不是一定会被接受？

不是。

它只是本轮被送去 target model 验证的 draft tokens。

### 33.3 为什么 schedule 后要清空 `request.spec_token_ids`？

因为它们已经转移到本轮 `SchedulerOutput`。

如果不清空，下一轮可能重复调度旧 draft。

### 33.4 为什么 rejected 后要回退 `num_computed_tokens`？

因为 schedule 阶段已经乐观推进了 computed 进度。

Rejected draft 不应该作为后续上下文的一部分，所以必须回退。

### 33.5 为什么 async scheduling 有 `prev_num_draft_len`？

因为 async 场景下下一轮状态准备可能早于上一轮输出完全回收。

Worker 需要先用 draft 长度做乐观占位，再根据真实 accepted token 数修正。

### 33.6 为什么 structured output 会裁剪 draft tokens？

因为 draft token 也必须满足 grammar。

不合法的 draft 如果送去验证，会浪费计算，甚至影响 grammar bitmask。

---

## 34. 从字段角度记忆

```text
Request.spec_token_ids
  Scheduler 侧 pending draft tokens。

SchedulerOutput.scheduled_spec_decode_tokens
  本轮已经调度给 Worker 验证的 draft tokens。

InputBatch.spec_token_ids
  Worker 当前 batch 中每个 req_index 的 draft tokens。

InputBatch.token_ids_cpu
  prompt + accepted output + scheduled draft 的数组化 token buffer。

CachedRequestState.prev_num_draft_len
  async spec decode 下上一轮 draft token 数，用于乐观占位和修正。

ModelRunnerOutput.sampled_token_ids
  target 验证 / RejectionSampler 后真正返回给 Scheduler 的 tokens。

Request.output_token_ids
  最终被接受并对外输出的 tokens。
```

---

## 35. 总结

Draft tokens 在 vLLM V1 中的核心生命周期是：

```text
propose：
  Worker / drafter 产生 draft tokens。

store：
  Scheduler 把 draft tokens 暂存在 Request.spec_token_ids。

schedule：
  Scheduler 把本轮可验证的 draft tokens 放入 SchedulerOutput.scheduled_spec_decode_tokens。

batch：
  Worker 把 scheduled draft tokens 写入 InputBatch.spec_token_ids 和 token_ids_cpu。

verify：
  target model + RejectionSampler 决定 accepted / rejected。

commit：
  Scheduler.update_from_output() 只把 accepted / replacement / bonus tokens 写入 Request.output_token_ids。

repair：
  rejected draft tokens 会触发 num_computed_tokens / placeholders 修正。
```

如果只记一句话：

```text
Request.spec_token_ids 是“下一轮待验证的预测”，SchedulerOutput.scheduled_spec_decode_tokens 是“本轮实际验证的预测”，Request.output_token_ids 才是“验证后正式提交的输出”。
```
