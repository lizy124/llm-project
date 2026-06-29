# 07. Spec decode 如何影响 KV cache 和 num_computed_tokens？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\kv_cache_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\request.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_input_batch.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\block_table.py`

本问题关注：speculative decoding 中，draft tokens 被 target model 验证前后，`Request.num_computed_tokens`、`Request.output_token_ids`、KV cache block 分配、slot mapping、prefix cache、external KV connector、Mamba / hybrid state 如何保持一致；为什么 vLLM 会先“乐观推进”计算进度，再在 rejected tokens 出现时回滚；以及为什么 KV cache 只缓存 finalized tokens，而不是缓存所有被猜过的 draft tokens。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录文档风格，本篇按“先定状态一致性问题，再走 Scheduler / KVCacheManager / ModelRunner 三层链路，最后总结边界和例子”的方式梳理。

要回答的问题分成 12 组：

```text
1. Spec decode 为什么会让 KV cache 和 num_computed_tokens 变复杂？
2. Scheduler 侧的 num_tokens / num_tokens_with_spec / num_computed_tokens 分别表示什么？
3. num_lookahead_tokens 如何影响 KV block 分配？
4. KVCacheManager.allocate_slots() 如何处理 new + lookahead？
5. 为什么只 cache finalized tokens，而不 cache unverified draft tokens？
6. Scheduler 为什么在 _update_after_schedule() 先乐观推进 num_computed_tokens？
7. update_from_output() 如何根据 accepted / rejected 回滚 num_computed_tokens？
8. Worker / ModelRunner 如何用 num_computed_tokens 构造 positions / seq_lens / slot_mapping？
9. async spec decode 如何在 GPU 侧修正 num_computed_tokens？
10. Mamba / hybrid 模型为什么还要维护 num_accepted_tokens？
11. preemption、deferred free、external KV invalid blocks 和 spec decode 如何交互？
12. 哪些状态是真实输出，哪些只是验证过程中的临时状态？
```

阅读顺序建议：

```text
03_scheduler_spec_decode_flow.md
  → 05_model_runner_spec_forward.md
  → 06_rejection_sampler_flow.md
  → 07_kv_cache_and_num_computed_tokens.md
  → 09_output_recovery_and_scheduler_update.md
```

本篇重点讲 KV cache 和计算进度的一致性，不重复展开 `SpecDecodeMetadata` 和 `RejectionSampler` 的内部算法。

---

## 1. 一句话回答

Spec decode 对 KV cache 和 `num_computed_tokens` 的核心影响是：

```text
Scheduler 会把 draft tokens 当成本轮要计算的 token 调度出去，
KVCacheManager 会为它们分配可写 slot，
ModelRunner 会让 target model forward 写入对应 KV，
但只有被 target model 接受的 tokens 才能推进 request 的真实输出和可缓存前缀。
```

因此 vLLM 采用两阶段状态管理：

```text
1. schedule 阶段：乐观推进
   request.num_computed_tokens += num_scheduled_tokens

2. update 阶段：拒绝回滚
   request.num_computed_tokens -= num_rejected
```

同时，KV cache manager 在缓存 blocks 时会明确避免缓存未验证的 draft tokens：

```text
new tokens 可以包含 unverified draft tokens；
但 cache_blocks() 只提交 request.num_tokens 范围内的 finalized tokens。
```

最小心智模型：

```text
spec decode 可以临时“计算”draft tokens，
但只有 accepted tokens 才能成为真实上下文进度。
```

---

## 2. 状态一致性到底要保证什么

spec decode 下至少有 5 个状态必须一致：

```text
Request.output_token_ids
  已被 target model / rejection sampler 正式接受的输出 token。

Request.spec_token_ids
  drafter 猜出的、等待下一轮验证的候选 token。

Request.num_computed_tokens
  Scheduler 认为请求已经计算到的位置，可能经过乐观推进和回滚。

KV cache blocks / slots
  target forward 实际写入过的 KV 物理空间。

InputBatch.num_computed_tokens / positions / slot_mapping
  Worker 侧本轮 forward 使用的 token 位置和 KV slot 映射。
```

如果这几个状态不一致，会出现：

```text
1. rejected draft token 被当成正式上下文；
2. 下一轮 positions 偏移，导致 logits 对不上；
3. KV cache 中无效 token 被 prefix cache 复用；
4. Mamba / hybrid recurrent state 用了错误的 accepted token 数；
5. external KV / encoder cache 过早释放或错误 recompute。
```

所以 spec decode 的难点不是“一次输出多个 token”，而是：

```text
猜过、算过、接受过、缓存过，这四个概念必须分清。
```

---

## 3. Request 侧三个长度概念

`Request` 中的关键字段在 `code/vllm/vllm/v1/request.py:140` 到 `code/vllm/vllm/v1/request.py:153`：

```python
self.num_output_placeholders = 0
self.async_tokens_to_discard = 0
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
```

两个属性：

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

| 字段 | 含义 | 是否包含 draft tokens |
|---|---|---|
| `num_tokens` | prompt + accepted output | 不包含 |
| `num_tokens_with_spec` | prompt + accepted output + pending draft | 包含 |
| `num_computed_tokens` | Scheduler/Worker 认为已经计算到的位置 | schedule 后可能临时包含 |

重要边界：

```text
num_tokens 是真实 token 边界；
num_tokens_with_spec 是调度候选边界；
num_computed_tokens 是计算进度边界。
```

spec decode 的核心就是让这三个边界在不同阶段有不同含义，但最终收敛到一致状态。

---

## 4. Scheduler 的统一 token 模型

`schedule()` 的注释明确说明：

```text
There is no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has the num_computed_tokens and num_tokens_with_spec.
num_tokens_with_spec =
  len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:387` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:398`

这意味着 Scheduler 不单独建一个“spec decode 阶段”。

它只计算：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:462` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:466`

普通 decode 下：

```text
num_tokens_with_spec == num_tokens
num_new_tokens 通常是 1
```

spec decode 下：

```text
num_tokens_with_spec = num_tokens + len(spec_token_ids)
num_new_tokens 可能覆盖多个 draft verification tokens
```

---

## 5. max_model_len 为什么要特别限制 spec decode

计算 `num_new_tokens` 后，会限制最大长度：

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:471` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:478`

注释写得很直接：

```text
This is necessary when using spec decoding.
```

原因：

```text
spec decode 一轮会验证 K 个 draft tokens，
如果 K 个都接受，还可能额外采样 bonus token；
因此必须提前给 num_sampled_tokens_per_step 留空间，
避免 positions 超过 max_model_len。
```

这也是为什么接近最大长度时，ModelRunner 可能不再生成 draft tokens，而是 zero out drafts，避免 stale drafts 污染状态。

---

## 6. num_lookahead_tokens：为 spec decode 预留 KV slot

Scheduler 初始化时：

```python
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.num_lookahead_tokens = 0
```

如果启用某些 speculative 方法：

```python
if speculative_config.use_eagle():
    self.use_eagle = True
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.uses_draft_model():
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.use_dflash():
    self.num_lookahead_tokens = self.num_spec_tokens + 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:227` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:248`

调度时传给 KVCacheManager：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:521` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:528`

含义：

```text
num_new_tokens：本轮 target model 真正要计算的 token 数；
num_lookahead_tokens：为了 spec proposer / draft verification 额外预留的未来 slots。
```

这说明 spec decode 不只影响 sampler，也会影响 KV cache admission。

---

## 7. KVCacheManager.allocate_slots 的 block layout

`allocate_slots()` 入口：`code/vllm/vllm/v1/core/kv_cache_manager.py:244`

参数中：

```python
num_lookahead_tokens: int = 0
```

注释说明：

```text
The number of speculative tokens to allocate.
This is used by spec decode proposers with kv-cache such as eagle.
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:267` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:269`

函数里的 block layout 注释非常关键：

```text
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
                                              | < to be computed > |
                                  | < to be allocated > |
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:290` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:322`

其中：

```text
comp：request.num_computed_tokens
new_comp：prefix cache 新命中的 token
ext_comp：external KV connector 命中的 token
new：本轮要计算的 token，包含未验证 draft tokens
lookahead：spec decode 额外预留 token
```

注释明确说：

```text
new = num_new_tokens, including unverified draft tokens
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:320` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:321`

这句话是本篇核心之一。

---

## 8. KV allocation 中 “要分配” 和 “可缓存” 是两回事

`allocate_slots()` 先计算要有多少 token 需要 slot：

```python
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:389` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:392`

然后基于 `num_tokens_need_slot` 计算要分配多少 blocks：

```python
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=request.request_id,
    num_tokens=num_tokens_need_slot,
    ...
    num_tokens_main_model=num_tokens_main_model,
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:404` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:412`

最后真正分配：

```python
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,
    num_tokens_main_model,
    num_encoder_tokens,
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:435` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:440`

这里的区别：

```text
num_tokens_need_slot：需要物理 slot 的范围，包含 lookahead；
num_tokens_main_model：target model 本轮实际会计算到的范围，不含 lookahead；
num_tokens_to_cache：可以提交到 prefix cache 的 finalized 范围。
```

---

## 9. 为什么只 cache finalized tokens

`allocate_slots()` 的注释明确指出：

```text
NOTE: for new tokens which include both verified and unverified draft
tokens, we only cache the verified tokens (by capping the number at
request.num_tokens).
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:324` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:326`

真正 cache 时：

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:447` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:456`

这表示：

```text
即使 target model 已经 forward 过 draft token 的位置，
只要这些 draft tokens 还没有成为 request.num_tokens 的一部分，
就不会被当成 finalized prefix 缓存。
```

为什么？

```text
1. draft tokens 可能被 rejected；
2. rejected token 后面的 KV 不能作为未来请求的 prefix cache；
3. prefix cache 只能保存真实上下文前缀；
4. request.num_tokens 是 finalized token 边界。
```

一句话：

```text
KV slot 可以为 draft token 分配，KV cache 可以被 forward 写入，但 prefix cache 不能提交 unverified draft token。
```

---

## 10. schedule 后为什么先乐观推进 num_computed_tokens

Scheduler 构造 `SchedulerOutput` 后调用：

```python
self._update_after_schedule(scheduler_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1096` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1098`

核心逻辑：

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    ...
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1138` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1147`

注释说明：

```text
If some tokens (e.g. spec tokens) are rejected later,
the number of computed tokens will be adjusted in update_from_output.
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1128` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1137`

这叫“乐观推进”：

```text
schedule 阶段先假设本轮 scheduled tokens 都完成了；
update 阶段再根据 rejection sampler 结果修正。
```

为什么要这样做？

```text
1. SchedulerOutput 必须保留原始 num_scheduled_tokens，供 Worker 准备 input ids；
2. Scheduler 可能继续调度后续 batch，需要先推进进度；
3. async / PP / batch queue 下，schedule 和 update 可能交错；
4. rejected token 数只有 ModelRunner 采样后才知道。
```

---

## 11. update_from_output 如何回滚 rejected tokens

模型返回后，Scheduler 在 `update_from_output()` 中处理 spec decode。

先取本轮输出：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = (
    sampled_token_ids[req_index] if sampled_token_ids else []
)

scheduled_spec_token_ids = (
    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1542` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1549`

如果本轮有 spec tokens：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1550` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1556`

然后回滚：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1557` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1563`

如果 async scheduling 下有 placeholders：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1564` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1567`

这就把乐观推进的计算进度修正为真实 accepted 进度。

---

## 12. accepted / rejected 数量为什么这样算

公式：

```text
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

其中：

```text
num_draft_tokens：本轮验证了几个 draft tokens；
num_sampled：正常采样 token 数，通常是 1；
generated_token_ids：RejectionSampler 最终返回的真实 tokens。
```

为什么 `len(generated_token_ids)` 要减 `num_sampled`？

因为 spec decode 输出通常是：

```text
accepted draft token prefix + replacement/bonus token
```

最后那个 replacement / bonus token 是 target model 采样出来的，不属于“accepted draft token”。

例子：

```text
scheduled draft = [d1, d2, d3, d4]
generated = [d1, d2, x]
num_sampled = 1
num_accepted = 3 - 1 = 2
num_rejected = 4 - 2 = 2
```

这表示 d1/d2 接受，d3/d4 拒绝，x 是 target 采出的真实 token。

---

## 13. output_token_ids 如何落账

回滚 `num_computed_tokens` 后，Scheduler 继续把真实输出写入 request：

```python
new_token_ids = generated_token_ids
...
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1580` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1592`

`_update_request_with_output()` 中：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1848` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1864`

这说明：

```text
只有 generated_token_ids 才会进入 output_token_ids；
scheduled_spec_decode_tokens 本身不会直接进入 output_token_ids。
```

---

## 14. Worker 侧 CachedRequestState 如何表达进度

Worker 侧 `CachedRequestState` 定义：

```python
@dataclass
class CachedRequestState:
    ...
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    ...
    prev_num_draft_len: int = 0
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:33` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:60`

它的 `num_tokens` 是：

```python
@property
def num_tokens(self) -> int:
    return self.num_prompt_tokens + len(self.output_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:74` 到 `code/vllm/vllm/v1/worker/gpu_input_batch.py:76`

Worker 侧同样区分：

```text
output_token_ids：正式输出；
spec_token_ids：InputBatch 中的待验证 tokens；
num_computed_tokens：本轮准备 positions / slot mapping 的进度基准；
prev_num_draft_len：async spec decode 修正需要。
```

---

## 15. InputBatch 如何保存 spec tokens

`InputBatch.update_req_spec_token_ids()` 入口：`code/vllm/vllm/v1/worker/gpu_input_batch.py:483`

核心逻辑：

```python
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

这里的 `num_tokens_no_spec` 是关键边界：

```text
num_tokens_no_spec 之前是真实 token；
从 num_tokens_no_spec 开始追加本轮 spec tokens。
```

也就是说：

```text
spec tokens 被放进 token_ids_cpu 供 target forward 使用，
但它们没有进入 output_token_ids 的真实边界。
```

---

## 16. ModelRunner 如何用 num_computed_tokens 生成 positions

`_prepare_inputs()` 中先 gather input ids，然后计算 positions。

CPU positions：

```python
positions_np = (
    self.input_batch.num_computed_tokens_cpu[req_indices]
    + self.query_pos.np[: cu_num_tokens[-1]]
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1920` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1924`

GPU positions：

```python
self.positions[:total_num_scheduled_tokens] = (
    self.num_computed_tokens[req_indices_gpu].to(torch.int64)
    + self.query_pos.gpu[:total_num_scheduled_tokens]
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2109` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2112`

这说明：

```text
num_computed_tokens 决定本轮 scheduled tokens 的起始 position。
```

如果 rejected tokens 没有回滚，下轮 positions 会从错误位置开始。

---

## 17. optimistic_seq_lens：forward 先按乐观长度构造 metadata

`_prepare_inputs()` 中：

```python
torch.add(
    self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
    torch.from_numpy(num_scheduled_tokens),
    out=self.optimistic_seq_lens_cpu[:num_reqs],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2010` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2019`

注释说明：

```text
Compute optimistic seq_lens (assumes all draft tokens from previous iteration accepted).
```

它用于：

```text
1. _build_attention_metadata() 的 max_seq_len；
2. discard_request_mask；
3. attention backend 的 seq_lens_cpu_upper_bound；
4. padded / CUDA graph 场景下的 metadata 上界。
```

这也是乐观推进在 Worker 侧的体现。

---

## 18. slot mapping：draft tokens 也会映射到 KV slots

`_prepare_inputs()` 中计算 slot mapping：

```python
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    self.query_start_loc.gpu[: num_reqs + 1],
    self.positions[:total_num_scheduled_tokens],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2118` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2122`

随后 `execute_model()` 取 slot mappings：

```python
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4241` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4253`

含义：

```text
本轮 target forward 会计算 draft verification tokens，
所以这些 draft tokens 也需要物理 KV slot。
```

但这不等于它们会被永久视为有效前缀：

```text
slot 可写 ≠ token finalized ≠ prefix cache committed。
```

---

## 19. async spec decode 的 GPU 侧 num_computed_tokens 修正

async spec decode 下，CPU `num_computed_tokens` 可能是乐观值。

`_prepare_inputs()` 中：

```python
if (
    self.use_async_spec_decode
    and self.valid_sampled_token_count_gpu is not None
    and prev_req_id_to_index
):
    ...
    update_num_computed_tokens_for_batch_change(
        self.num_computed_tokens,
        self.num_accepted_tokens.gpu[:num_reqs],
        self.prev_positions.gpu[:num_reqs],
        self.valid_sampled_token_count_gpu,
        self.prev_num_draft_tokens.gpu,
        cpu_values,
    )
else:
    self.num_computed_tokens[:num_reqs].copy_(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2073` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2099`

目的：

```text
在不等待 CPU 完整同步的情况下，
用上一轮 valid sampled token count 修正当前 batch 的 GPU num_computed_tokens，
保证 positions / seq_lens / slot mapping 使用正确进度。
```

这对 async scheduling 很重要，因为下一轮输入准备可能与上一轮输出拷贝重叠。

---

## 20. async scheduling 下 worker 侧的乐观 output_token_ids

`_update_states()` 中，如果上一轮有 draft length 且 async scheduling 开启：

```python
if req_state.prev_num_draft_len and self.use_async_scheduling:
    ...
    optimistic_num_accepted = req_state.prev_num_draft_len
    req_state.output_token_ids.extend([-1] * optimistic_num_accepted)
    deferred_spec_decode_corrections.append(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1292` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1317`

这表示：

```text
Worker 侧为了保持 pipeline 流水，会先用 -1 占位模拟 draft tokens 全部接受；
后续再根据真实 accepted 数修正。
```

这些 `-1` 不是用户输出，只是 worker-local bookkeeping 占位。

---

## 21. Mamba / hybrid 模型为什么需要 num_accepted_tokens

`_update_states_after_model_execute()` 专门处理 hybrid 模型：

```python
if not self.speculative_config or not self.model_config.is_hybrid:
    return

num_reqs = output_token_ids.size(0)
self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1497` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1515`

docstring 说明：

```text
This is used for MTP/EAGLE for hybrid models, as in linear attention,
only the last token's state is kept. In MTP/EAGLE, for draft tokens
the state are kept until we decide how many tokens are accepted for
each sequence, and a shifting is done during the next iteration
based on the number of accepted tokens.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1497` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1507`

含义：

```text
对于某些 hybrid / linear attention / Mamba 模型，
不是所有 draft token 的 recurrent state 都能简单保留；
必须知道每个请求接受了几个 token，
下一轮才能把 state 移到正确位置。
```

---

## 22. Mamba align 模式的 preprocess / postprocess

forward 前：

```python
if self.cache_config.mamba_cache_mode == "align":
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()
        deferred_state_corrections_fn = None
    mamba_utils.preprocess_mamba(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4198` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4216`

注释说明：

```text
preprocess_mamba reads req_state.num_computed_tokens (CPU)
to decide copy operations, so we must apply deferred corrections before it runs.
```

采样后：

```python
mamba_utils.postprocess_mamba_align_gpu(
    bufs=self._get_mamba_bufs(),
    num_reqs=num_reqs,
    num_accepted_tokens_gpu=self.num_accepted_tokens.gpu,
    num_accepted_tokens_cpu_tensor=(
        self.input_batch.num_accepted_tokens_cpu_tensor
    ),
    ...
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1517` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1533`

这条链路说明：

```text
spec decode 的 accepted token 数会直接影响 Mamba state copy / align。
```

---

## 23. preemption 时为什么清空 spec tokens

如果 KV allocation 失败，Scheduler 可能 preempt 请求。

`_preempt_request()` 中：

```python
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
if request.spec_token_ids:
    request.spec_token_ids = []
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1105` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1121`

如果被抢占请求已经在本轮 scheduled_running_reqs 中，还要移除本轮 spec 输出：

```python
scheduled_spec_decode_tokens.pop(preempted_req_id, None)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:542` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:548`

原因：

```text
preempt 后 block / computed state 都要重新建立；
旧 spec tokens 基于旧上下文，不能继续验证；
否则可能把过期 draft token 写入新 KV 轨迹。
```

---

## 24. deferred block free：避免释放仍在写的 KV blocks

KV connector + overlapping batches 下可能启用 `defer_block_free`：

```python
if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
    self.defer_block_free = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:145` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:151`

每个非空 step 会推进 fence：

```python
if self.defer_block_free and total_num_scheduled_tokens > 0:
    self.sched_step_seq += 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1091` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1094`

schedule 后记录 request 的 in-flight step：

```python
if self.defer_block_free:
    request.last_sched_seq = self.sched_step_seq
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1142` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1144`

update 时 drain：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1477` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1481`

释放逻辑：

```python
if not self.defer_block_free or request.last_sched_seq <= self.processed_step_seq:
    self.kv_cache_manager.free(request)
    return
blocks = self.kv_cache_manager.pop_blocks_for_free(request)
if blocks:
    self.deferred_frees.append((self.sched_step_seq, blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2077` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2090`

这和 spec decode 的关系：

```text
spec decode 一轮可能写入多个 token 的 KV，
async / overlapping batches 下，释放 blocks 前必须确认相关 GPU writes 已完成。
```

---

## 25. encoder cache free 为什么要考虑 placeholders

`_free_encoder_inputs()` 中有 spec decode 相关注释：

```python
elif (
    start_pos + num_tokens
    <= request.num_computed_tokens - request.num_output_placeholders
):
    # The encoder output is already processed and stored in the
    # decoder's KV cache, and progress is far enough past the
    # placeholder range that no pending draft-token rejection can
    # roll num_computed_tokens back into it.
    self.encoder_cache_manager.free_encoder_input(request, input_id)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1886` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1893`

含义：

```text
因为 rejected draft tokens 会回滚 num_computed_tokens，
释放 encoder cache 时不能只看当前乐观 num_computed_tokens；
必须扣掉 output placeholders，确保即使回滚也不会回到 encoder input 范围。
```

这说明 spec decode 的拒绝回滚会影响 encoder cache 生命周期。

---

## 26. external KV invalid blocks：另一类 num_computed_tokens 回退

如果 KV connector 返回 invalid blocks：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(
        kv_connector_output.invalid_block_ids,
        num_scheduled_tokens,
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1490` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1498`

`_update_requests_with_invalid_blocks()` 会把请求的 `num_computed_tokens` 截断到最长有效前缀：

```python
request.num_computed_tokens = idx * self.block_size
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2521` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2524`

它还会统计需要 recompute 的 tokens，并收集要 evict 的 blocks。

这不是 spec decode 专属逻辑，但和本篇主题一致：

```text
num_computed_tokens 是可回退的进度账本；
它可能因为 rejected draft tokens 回退，
也可能因为 external KV load failure 回退。
```

---

## 27. WAITING_FOR_REMOTE_KVS 恢复时的 computed tokens

异步 KV 接收完成后：

```python
if request.request_id in self.failed_recving_kv_req_ids:
    # Request had KV load failures; num_computed_tokens was already
    # updated in _update_requests_with_invalid_blocks
    if request.num_computed_tokens:
        self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
    else:
        self.kv_cache_manager.free(request)
...
else:
    self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
    if request.num_computed_tokens == request.num_tokens:
        request.num_computed_tokens = request.num_tokens - 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2350` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2381`

这里体现同一个原则：

```text
cache_blocks() 只能缓存当前确认可用的 computed prefix；
如果 full hit，为了采下一个 token，还要回退最后一个 token 重新计算 logits。
```

---

## 28. routed experts：spec decode 下输出位于 scheduled range 开头

在 `update_from_output()` 中，如果返回 routed experts：

```python
if scheduled_spec_token_ids:
    # Spec decode: accepted tokens at the START of
    # the scheduled range, rejected at the end.
    routed_experts = routing_data[
        req_offset : req_offset + len(new_token_ids)
    ]
else:
    # Normal decode / re-prefill: token(s) at the END.
    routed_experts = routing_data[end - len(new_token_ids) : end]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1645` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1653`

这和 KV/进度语义一致：

```text
spec decode scheduled range 中，accepted tokens 在前，rejected tokens 在后；
因此真实输出对应前缀，而不是普通 decode 中的末尾。
```

---

## 29. 一个完整例子：4 个 draft tokens，接受 2 个

假设请求状态：

```text
num_tokens = 100
num_computed_tokens = 100
spec_token_ids = [d1, d2, d3, d4]
num_output_placeholders = 0
num_lookahead_tokens = 4
```

### 29.1 schedule 阶段

```text
num_tokens_with_spec = 104
num_new_tokens = 104 - 100 = 4
```

KV allocation：

```text
num_tokens_main_model = 100 + 4 = 104
num_tokens_need_slot = 104 + lookahead
```

Scheduler 输出：

```text
num_scheduled_tokens[req] = 4
scheduled_spec_decode_tokens[req] = [d1,d2,d3,d4]
```

然后乐观推进：

```text
request.num_computed_tokens = 104
```

### 29.2 target forward 阶段

ModelRunner 使用 positions：

```text
positions = [100, 101, 102, 103]
```

slot mapping 为这 4 个 token 找到 KV slots。

target model forward 可能写入这 4 个位置的 KV。

### 29.3 rejection sampling 阶段

假设输出：

```text
generated_token_ids = [d1, d2, x]
```

表示：

```text
d1/d2 accepted；
d3/d4 rejected；
x 是 replacement / bonus token。
```

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
output_token_ids append [d1, d2, x]
num_tokens 变为 103
```

注意这里 `num_computed_tokens = 102` 和 `num_tokens = 103` 看起来可能短暂不同，后续调度会继续让计算进度追上真实 token 边界。

---

## 30. 为什么 rejected token 的 KV 不直接删除

spec decode 下，target forward 可能已经把 rejected draft tokens 的 KV 写入了物理 slot。

但 vLLM 的关键不在于“立即擦掉每个 rejected slot”，而在于：

```text
1. request.num_computed_tokens 回滚到 accepted 前缀；
2. 未来 schedule / positions / slot mapping 从回滚后的边界继续；
3. cache_blocks() 不提交 unverified draft tokens；
4. 后续有效 token 会覆盖或重新使用对应位置；
5. 新分配 blocks 如有需要会 zero，避免脏数据影响。
```

因此 rejected KV 的处理更像：

```text
逻辑上从请求进度中剔除，
物理上由后续 slot reuse / block lifecycle 管理。
```

这避免了每次 rejection 都做昂贵的精细 KV 删除。

---

## 31. new_block_ids_to_zero：新 block 防脏数据

Scheduler 会把新分配且需要清零的 block ids 放入：

```text
SchedulerOutput.new_block_ids_to_zero
```

Worker 侧 `_update_states()` 中：

```python
if scheduler_output.new_block_ids_to_zero:
    self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1153` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1156`

这和 spec decode 的关系：

```text
spec decode 可能使一次调度覆盖更多 KV slots；
新 block 在用于 attention / SSM 前必须避免旧 NaN 或脏数据污染。
```

---

## 32. 与 prefix cache 的边界

prefix cache 查找使用真实请求 token 序列，而不是 pending spec tokens。

`KVCacheManager.get_computed_blocks()` 中最大命中长度：

```python
max_cache_hit_length = request.num_tokens - 1
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:221` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:228`

这表示：

```text
prefix cache 的 key 只基于 finalized request tokens；
pending draft tokens 不会作为 prefix cache hit 的正式前缀。
```

再结合 `cache_blocks()` 被 cap 到 `request.num_tokens`，可以得到原则：

```text
prefix cache 只服务真实前缀，不服务猜测前缀。
```

---

## 33. 与 ModelRunner attention metadata 的边界

ModelRunner 构造 attention metadata 时，spec decode 下使用的 `seq_lens` / `positions` / `slot_mapping` 是为了本轮 target verification forward。

这些 metadata 的语义是：

```text
让本轮 forward 正确运行。
```

而 Scheduler / KVCacheManager 的缓存语义是：

```text
让未来复用的 prefix 只包含 finalized tokens。
```

所以不要混淆：

```text
attention metadata 可以覆盖 draft tokens；
prefix cache commit 不能覆盖 unverified draft tokens。
```

---

## 34. 各组件职责边界

### 34.1 Scheduler 负责

```text
用 num_tokens_with_spec 计算本轮 token 需求；
把 num_lookahead_tokens 传给 KVCacheManager；
在 _update_after_schedule() 乐观推进 num_computed_tokens；
在 update_from_output() 根据 num_rejected 回滚；
管理 preemption / deferred free / encoder cache free；
在 external KV invalid blocks 时截断 num_computed_tokens。
```

### 34.2 KVCacheManager 负责

```text
根据 comp / new_comp / ext_comp / new / lookahead 计算需要的 blocks；
为 new + lookahead 分配 slot；
只 cache request.num_tokens 范围内的 finalized tokens；
free / pop blocks；
处理 prefix cache blocks。
```

### 34.3 ModelRunner 负责

```text
把 scheduled spec tokens 写入 InputBatch；
用 num_computed_tokens 生成 positions；
用 block table 生成 slot mapping；
用 optimistic seq_lens 构造 attention metadata；
在 async spec decode 下 GPU 修正 num_computed_tokens；
在 hybrid/Mamba 下维护 num_accepted_tokens。
```

### 34.4 RejectionSampler 负责

```text
输出 sampled_token_ids；
其中 accepted draft tokens 和 replacement/bonus token 决定 Scheduler 如何计算 num_rejected。
```

一句话边界：

```text
Scheduler/KVCacheManager 管“进度和物理空间”，ModelRunner 管“本轮如何写入这些空间”，RejectionSampler 决定“哪些写入成为真实进度”。
```

---

## 35. 容易混淆的点

### 35.1 draft token forward 后是不是就算 computed？

临时算，最终不一定算。

```text
schedule 后 num_computed_tokens 会乐观加上 scheduled draft tokens；
update_from_output 后 rejected tokens 会被减掉。
```

### 35.2 draft token 的 KV 会不会被写入？

会参与本轮 target forward，因此可能写入物理 KV slot。

但：

```text
写入 slot ≠ token accepted；
写入 slot ≠ prefix cache committed。
```

### 35.3 rejected KV 是否立即逐 token 清理？

不是这个层面的主要机制。

vLLM 通过：

```text
num_computed_tokens 回滚；
prefix cache 不提交 unverified tokens；
后续 slot reuse / block lifecycle；
必要时 zero 新 block；
```

保持逻辑正确。

### 35.4 num_lookahead_tokens 是本轮验证的 draft 数吗？

不是。

```text
scheduled_spec_decode_tokens：本轮实际验证的 draft tokens；
num_lookahead_tokens：KV allocation 额外预留的 slot 数。
```

### 35.5 request.num_tokens 和 request.num_computed_tokens 谁更真实？

它们表达不同维度：

```text
num_tokens：真实 token 序列长度；
num_computed_tokens：KV/计算进度账本，可能短暂乐观或回滚。
```

最终调度目标是让 `num_computed_tokens` 追上真实需要计算的 token 边界。

### 35.6 async spec decode 为什么要 GPU 修正？

因为 CPU 侧输出拷贝和下一轮输入准备可能重叠。

```text
不能总等 CPU 知道 accepted 数；
所以用 valid_sampled_token_count_gpu 在 GPU 上修正 num_computed_tokens。
```

---

## 36. 从“回答问题”的角度总结

如果要问：

```text
Spec decode 如何影响 KV cache 和 num_computed_tokens？
```

可以回答：

```text
Spec decode 会让 Scheduler 把 pending draft tokens 纳入 num_tokens_with_spec，从而在 schedule() 中把它们作为本轮 num_new_tokens 的一部分调度出去。KVCacheManager.allocate_slots() 会把这些 new tokens 视为需要 target model 计算的 tokens，并结合 num_lookahead_tokens 为 speculative proposer 预留额外 KV slots。但 KVCacheManager 在 cache_blocks() 时会把可缓存范围 cap 到 request.num_tokens，确保未验证 draft tokens 不会进入 prefix cache。

调度完成后，Scheduler._update_after_schedule() 会先把 num_scheduled_tokens 乐观加到 request.num_computed_tokens 上；ModelRunner 使用这个进度生成 positions、seq_lens、slot_mapping，让 target model forward 能验证 draft tokens。RejectionSampler 返回后，Scheduler.update_from_output() 根据 generated_token_ids 和 scheduled_spec_decode_tokens 计算 accepted / rejected 数量，并把 rejected tokens 从 num_computed_tokens 和 async output placeholders 中减掉。只有 generated_token_ids 中真正接受或采样出来的 tokens 会进入 Request.output_token_ids。

因此 spec decode 的 KV/进度一致性原则是：可以为 draft tokens 分配和写入临时 KV slots，但只有 accepted/finalized tokens 才能推进真实请求状态和 prefix cache。
```

---

## 37. 最关键流程图

```text
上一轮 drafter
  → Request.spec_token_ids

Scheduler.schedule()
  ├─ num_new_tokens =
  │    request.num_tokens_with_spec
  │    + request.num_output_placeholders
  │    - request.num_computed_tokens
  │
  ├─ KVCacheManager.allocate_slots(
  │    num_new_tokens,
  │    num_lookahead_tokens,
  │  )
  │    ├─ new 包含 unverified draft tokens
  │    ├─ lookahead 额外预留 slots
  │    └─ cache_blocks cap 到 request.num_tokens
  │
  ├─ scheduled_spec_decode_tokens[req_id]
  └─ _update_after_schedule()
       └─ request.num_computed_tokens += num_scheduled_tokens

ModelRunner.execute_model()
  ├─ InputBatch.update_req_spec_token_ids()
  ├─ positions = num_computed_tokens + query_pos
  ├─ compute_slot_mapping(positions)
  ├─ target model forward writes KV
  └─ RejectionSampler outputs sampled_token_ids

Scheduler.update_from_output()
  ├─ num_accepted = len(generated_token_ids) - num_sampled
  ├─ num_rejected = num_draft_tokens - num_accepted
  ├─ request.num_computed_tokens -= num_rejected
  ├─ request.num_output_placeholders -= num_rejected
  └─ request.output_token_ids append generated_token_ids
```

---

## 38. 最关键对象关系

```text
Request.spec_token_ids
  待验证 draft tokens，只参与下一轮调度。

Request.num_tokens_with_spec
  调度边界：真实 tokens + pending draft tokens。

Request.num_tokens
  finalized 边界：prompt + accepted output tokens。

Request.num_computed_tokens
  计算进度账本：schedule 后乐观推进，rejection 后回滚。

KVCacheManager.allocate_slots(new, lookahead)
  为 target forward 和 speculative lookahead 分配物理 blocks。

num_tokens_to_cache
  prefix cache 提交边界，被 cap 到 request.num_tokens。

InputBatch.num_computed_tokens_cpu
  Worker 准备 positions / slot mapping 的 CPU 进度。

GPUModelRunner.num_computed_tokens
  Worker 侧 GPU 进度，async spec decode 下可能用 GPU kernel 修正。

InputBatch.num_accepted_tokens_cpu / GPUModelRunner.num_accepted_tokens
  hybrid / Mamba spec decode 中记录每个请求接受了几个 token。

slot_mapping
  本轮每个 scheduled token，包括 draft verification tokens，对应的 KV slot。
```

---

## 39. 最小心智模型

如果只记一条主线，可以记：

```text
Spec decode 会让 draft tokens 临时进入计算和 KV slot，
但不会让它们直接进入真实输出或 prefix cache；
Scheduler 先乐观推进 num_computed_tokens，
再按 rejected tokens 回滚，
保证最终只有 accepted tokens 决定真实进度。
```

再压缩成一句话：

```text
Spec decode 的 KV/进度规则是：可计算、可占 slot、不可未验证提交。
```
