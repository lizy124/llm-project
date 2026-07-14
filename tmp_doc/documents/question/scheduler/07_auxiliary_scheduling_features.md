# 07. 多模态 encoder 输入、结构化输出、投机解码等附加能力如何同步调度？

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：除了最基本的 running / waiting 调度、token budget、prefix cache、KV block 分配之外，Scheduler 还要同时处理多模态 encoder input、结构化输出 grammar、投机解码、LoRA、Mamba、DP prefill balancing、pause state、KV / EC Connector metadata 等附加能力。这些能力不是独立的一套调度器，而是嵌入在 `schedule()` 主流程的不同检查点。

一句话概括：

```text
Scheduler 的主线仍然是：
先调度 running，再调度 waiting，最后构造 SchedulerOutput。

附加能力的作用是：
在这条主线上裁剪 num_new_tokens、阻塞或跳过请求、消耗额外预算、
预留 KV blocks、生成 bitmask / connector metadata，
并在 Worker 返回后修正状态。
```

---

## 1. 一句话回答

这些附加能力的插入点可以概括为：

| 能力 | 插入位置 | 对调度的影响 |
|---|---|---|
| Encoder / 多模态输入 | running / waiting 计算 `num_new_tokens` 后 | 可能缩短 `num_new_tokens`，消耗 `encoder_compute_budget` |
| ECConnector | encoder input 调度时、SchedulerOutput 构造时 | 允许从外部 encoder cache load，生成 EC metadata |
| Structured output | waiting blocked 恢复、SchedulerOutput 后 bitmask、output 回收 | grammar 未就绪会阻塞；decode 阶段生成 bitmask；输出后推进 grammar |
| Speculative decoding | running token 计算、KV block 分配、SchedulerOutput、output 回收 | 增加 `num_tokens_with_spec` 和 lookahead blocks；被拒绝 token 需要回退 |
| LoRA | waiting 请求进入 running 前 | 限制一轮内最多同时调度多少种 LoRA |
| Mamba block alignment | running / waiting token 裁剪后 | 要求 prefill chunk 尽量 block 对齐，可能让 `num_new_tokens` 变成 0 |
| DP prefill balancing | schedule 开头、running / waiting 阶段 | 在某些 step 延后 prefill，把资源让给 decode |
| Pause state | schedule 开头、waiting 阶段入口 | `PAUSED_ALL` 停止所有调度；`PAUSED_NEW` 只推进 running |
| KV Connector metadata | SchedulerOutput 构造后 | 把 KV load / save 计划传给 Worker |

所以：

```text
这些附加能力不是 schedule() 之外的后处理，
而是在调度过程中直接影响哪些请求能跑、能跑多少 token、需要哪些额外元数据。
```

---

## 2. 附加能力不是独立调度器，而是插入主调度链路

Scheduler 的主链路仍然是：

```text
schedule()
  → 初始化 token_budget / encoder_compute_budget
  → 调度 running 请求
  → 调度 waiting 请求
  → 分配 KV blocks
  → 构造 SchedulerOutput
  → 构造 KV / EC Connector metadata
  → _update_after_schedule()
```

附加能力不是把这个流程拆成多个独立 scheduler，而是在流程中插入约束。例如：

```text
running 请求已经算出 num_new_tokens
  → 如果有 encoder input，要检查 encoder budget/cache
  → 如果是 Mamba，要做 block-aligned split
  → 如果是 spec decode，要记录 scheduled_spec_decode_tokens
  → 如果 KV block 不够，可能触发抢占
```

waiting 请求也是类似：

```text
waiting 请求被选中
  → 先尝试恢复 blocked grammar / remote KV / streaming
  → 检查 LoRA max_loras
  → 查 prefix / external KV cache
  → 检查 encoder input / ECConnector
  → 检查 Mamba alignment
  → 分配 KV blocks
  → 进入 running 或被跳过
```

因此理解 `07` 的关键是：

```text
不要把这些能力看成独立阶段，
而要看它们分别插入 schedule() 主流程的哪个位置，
以及它们改变了哪个变量或哪个输出字段。
```

---

## 3. Encoder / 多模态 input 的调度入口

多模态或 encoder-decoder 请求可能带有 encoder input，例如 image/audio/video embedding。

Scheduler 使用同一个函数处理 running 和 waiting 请求中的 encoder input：

```python
def _try_schedule_encoder_inputs(
    self,
    request: Request,
    num_computed_tokens: int,
    num_new_tokens: int,
    encoder_compute_budget: int,
    shift_computed_tokens: int = 0,
) -> tuple[list[int], int, int, list[int]]:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1367`

函数注释说明它的职责：

```python
# Determine which encoder inputs need to be scheduled in the current step,
# and update `num_new_tokens` and encoder token budget accordingly.
```

位置：`scheduler.py:1376`

也就是说，它不只是返回“要不要跑 encoder”，还可能修改：

```text
num_new_tokens
encoder_compute_budget
external_load_encoder_input
```

running 阶段调用位置：

```python
if request.has_encoder_inputs:
    (
        encoder_inputs_to_schedule,
        num_new_tokens,
        new_encoder_compute_budget,
        external_load_encoder_input,
    ) = self._try_schedule_encoder_inputs(...)
```

位置：`scheduler.py:532`

waiting 阶段也有同样调用：

```python
if request.has_encoder_inputs:
    (...)= self._try_schedule_encoder_inputs(...)
    if num_new_tokens == 0:
        break
```

位置：`scheduler.py:885`

---

## 4. Encoder 调度看的是本轮 decoder token 窗口

`_try_schedule_encoder_inputs()` 会先找出本轮 decoder token 范围覆盖了哪些多模态 feature：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
)
```

位置：`scheduler.py:1409`

含义是：

```text
如果本轮 decoder 要计算的 token 范围覆盖到了某个 image/audio/video placeholder，
Scheduler 就必须确保对应 encoder input 已经计算、缓存，或可以从外部加载。
```

对于 encoder-decoder 模型，源码强制：

```python
if self.is_encoder_decoder:
    lo = 0
```

位置：`scheduler.py:1415`

因为 encoder-decoder 模型的 encoder input 通常必须在 decoder 开始前处理。

---

## 5. encoder cache 命中时不重复计算

对于非 encoder-decoder 模型，Scheduler 会先查 encoder cache：

```python
if self.encoder_cache_manager.check_and_update_cache(request, i):
    # The encoder input is already computed and cached from a
    # previous step.
    continue
```

位置：`scheduler.py:1449`

这表示：

```text
如果某个多模态 input 的 encoder output 已经在本地 encoder cache 中，
本轮不需要重新安排 encoder 计算。
```

同一个 step 内也会避免重复安排相同 encoder input：

```python
if item_identifier in mm_hashes_to_schedule:
    continue
```

位置：`scheduler.py:1444`

所以 encoder input 的调度粒度不是简单按 request，而是按 request 中的每个 multimodal item。

---

## 6. 不允许切分多模态输入时，会回退 `num_new_tokens`

如果配置不允许 chunked multimodal input：

```python
self.scheduler_config.disable_chunked_mm_input
```

并且本轮 token 范围只覆盖了某个 mm input 的一部分，Scheduler 会把 `num_new_tokens` 回退到该 mm input 之前：

```python
num_new_tokens = max(
    0, start_pos - (num_computed_tokens + shift_computed_tokens)
)
break
```

位置：`scheduler.py:1466`

含义：

```text
不能只处理半个 image/audio/video placeholder；
如果本轮预算只够覆盖它的一部分，
那就宁可先停在它之前，等后续预算足够时再整体处理。
```

这会导致：

```text
request 明明还有 token_budget，
但 num_new_tokens 被缩短，甚至变成 0。
```

---

## 7. encoder budget / encoder cache 不足时也会裁剪 token

如果 encoder cache 没空间，或者本轮 encoder budget 不够：

```python
if not self.encoder_cache_manager.can_allocate(
    request, i, encoder_compute_budget, num_embeds_to_schedule
):
    ...
```

位置：`scheduler.py:1470`

Scheduler 会把本轮 decoder token 范围缩短到该 encoder input 之前：

```python
if num_computed_tokens + shift_computed_tokens < start_pos:
    num_new_tokens = start_pos - (
        num_computed_tokens + shift_computed_tokens
    )
else:
    num_new_tokens = 0
break
```

位置：`scheduler.py:1477`

这说明 encoder 侧资源会反向影响 decoder 侧调度：

```text
即使 token_budget 够，
只要前方必须处理的 encoder input 没法安排，
本轮 decoder token 也只能调度到该 input 之前。
```

---

## 8. encoder_compute_budget 与 token_budget 是两个预算

`schedule()` 开始时有两个预算：

```python
token_budget = self.max_num_scheduled_tokens
encoder_compute_budget = self.max_num_encoder_input_tokens
```

位置：`scheduler.py:453`、`scheduler.py:460`

区别是：

| 预算 | 限制对象 | 典型场景 |
|---|---|---|
| `token_budget` | decoder / language model token 数 | 文本 prefill、decode、spec decode |
| `encoder_compute_budget` | encoder input 的 embed / token 数 | image/audio/video encoder |

`_try_schedule_encoder_inputs()` 中，如果本轮要本地计算 encoder input，会扣 encoder budget：

```python
num_embeds_to_schedule += num_encoder_embeds
encoder_compute_budget -= num_encoder_embeds
encoder_inputs_to_schedule.append(i)
```

位置：`scheduler.py:1515`

所以：

```text
encoder input 不直接扣 token_budget，
但它会通过缩短 num_new_tokens 间接影响 decoder token 调度。
```

---

## 9. ECConnector：外部 encoder cache

如果配置了 ECConnector，并且外部有该 encoder item：

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(
    item_identifier
):
    mm_hashes_to_schedule.add(item_identifier)
    external_load_encoder_input.append(i)
    num_embeds_to_schedule += num_encoder_embeds
    continue
```

位置：`scheduler.py:1507`

这表示：

```text
该 encoder input 不需要本地 encoder compute，
而是可以从外部 encoder cache 加载。
```

注意这里仍会累加 `num_embeds_to_schedule`，用于同一轮 encoder cache 容量 / 已安排 embedding 数量核算；区别是它不加入 `encoder_inputs_to_schedule`，也不扣 `encoder_compute_budget`。

调度成功后，Scheduler 仍然要为它分配 encoder cache 槽位，并通知 ECConnector：

```python
for i in external_load_encoder_input:
    self.encoder_cache_manager.allocate(request, i)
    if self.ec_connector is not None:
        self.ec_connector.update_state_after_alloc(request, i)
```

running 阶段位置：`scheduler.py:657`

waiting 阶段位置：`scheduler.py:1046`

最后构造 SchedulerOutput 时会生成 EC metadata：

```python
ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
    scheduler_output
)
scheduler_output.ec_connector_metadata = ec_meta
```

位置：`scheduler.py:1171`

---

## 10. scheduled_encoder_inputs 如何进入 SchedulerOutput

如果本轮要本地计算 encoder input，会记录：

```python
scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
```

running 阶段位置：`scheduler.py:649`

waiting 阶段位置：`scheduler.py:1037`

最终写入：

```python
scheduler_output = SchedulerOutput(
    ...
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    ...
)
```

位置：`scheduler.py:1142`

所以 Worker 能从 `SchedulerOutput.scheduled_encoder_inputs` 知道：

```text
本轮哪些请求需要同时执行 encoder input。
```

---

## 11. Structured output：grammar 未就绪时会阻塞请求

结构化输出请求可能在 grammar 构造完成前进入 Scheduler。

这时请求状态可能是：

```python
RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

这个状态属于 blocked waiting status：

```python
return status in (
    RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
    RequestStatus.WAITING_FOR_REMOTE_KVS,
    RequestStatus.WAITING_FOR_STREAMING_REQ,
)
```

位置：`scheduler.py:1911`

如果新请求进入 Scheduler 时已经是这个状态，`_enqueue_waiting_request()` 会把它放入 `skipped_waiting`。

waiting 调度时会尝试恢复：

```python
if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
    structured_output_req = request.structured_output_request
    if not (structured_output_req and structured_output_req.grammar):
        return False
    request.status = RequestStatus.WAITING
    return True
```

位置：`scheduler.py:2543`

状态迁移是：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  → grammar not ready: 继续 skipped_waiting
  → grammar ready: WAITING
  → 后续可进入 RUNNING
```

---

## 12. Structured output：decode 阶段生成 grammar bitmask

结构化输出在真正 decode 时，需要限制下一 token 的可选集合。

Scheduler 提供：

```python
def get_grammar_bitmask(
    self, scheduler_output: SchedulerOutput
) -> GrammarOutput | None:
```

位置：`scheduler.py:1527`

如果本轮没有结构化输出请求，直接返回：

```python
if not scheduler_output.has_structured_output_requests:
    return None
```

位置：`scheduler.py:1532`

只收集满足以下条件的请求：

```python
req.use_structured_output and not req.is_prefill_chunk
```

位置：`scheduler.py:1535`

这说明：

```text
grammar bitmask 主要用于 decode 阶段；
prefill chunk 阶段不生成 bitmask。
```

最终调用：

```python
bitmask = self.structured_output_manager.grammar_bitmask(
    self.requests,
    structured_output_request_ids,
    scheduler_output.scheduled_spec_decode_tokens,
)
```

位置：`scheduler.py:1544`

注意这里还传入了 `scheduled_spec_decode_tokens`，因为结构化输出也需要和 speculative decoding 协同。

---

## 13. Structured output：输出后推进 grammar 状态

Worker 返回 token 后，在 `update_from_output()` 中：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    struct_output_request = request.structured_output_request
    ...
    advance_token_ids = (
        self.structured_output_manager.trim_reasoning_for_advance(
            request, new_token_ids
        )
    )
    if advance_token_ids and not grammar.accept_tokens(
        req_id, advance_token_ids
    ):
        request.status = RequestStatus.FINISHED_ERROR
        request.resumable = False
        stopped = True
```

位置：`scheduler.py:1693`

含义：

```text
生成 token 后，grammar 状态要向前推进；
如果输出里混有 reasoning 内容，会先裁剪出需要交给 grammar 的 token；
如果 grammar 拒绝这些 token，说明出现不符合约束的输出，
请求会被标记为 FINISHED_ERROR。
```

这一部分属于 Worker 输出回收，`08_update_after_worker_output.md` 会进一步展开。

---

## 14. Streaming request：等待后续输入时会停在 skipped_waiting

blocked waiting status 还包括：

```python
RequestStatus.WAITING_FOR_STREAMING_REQ
```

位置：`scheduler.py:1915`

它通常来自可续写的 streaming/session 请求：当前输出停止但请求仍 `resumable`，且暂时没有下一段输入时，`_handle_stopped_request()` 会把状态设为 `WAITING_FOR_STREAMING_REQ`，并通过 `_enqueue_waiting_request()` 放入 `skipped_waiting`。

```python
request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
self.num_waiting_for_streaming_input += 1
self._enqueue_waiting_request(request)
```

位置：`scheduler.py:1948`

waiting 调度遇到它时不会自行恢复：

```python
if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
    assert not request.streaming_queue
    return False
```

位置：`scheduler.py:2550`

只有后续 `add_request()` 收到同 request id 的新输入，并通过 `_update_request_as_session()` 把状态改回 `WAITING` 后，它才会重新参与调度。

---

## 15. Speculative decoding 初始化：lookahead tokens

Scheduler 初始化时会根据 speculative config 设置：

```python
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.num_lookahead_tokens = 0
self.dynamic_sd_lookup: list[int] | None = None
```

位置：`scheduler.py:234`

如果启用 EAGLE：

```python
if speculative_config.use_eagle():
    self.use_eagle = True
    self.num_lookahead_tokens = self.num_spec_tokens
```

位置：`scheduler.py:244`

如果使用 draft model：

```python
if speculative_config.uses_draft_model():
    self.num_lookahead_tokens = self.num_spec_tokens
```

位置：`scheduler.py:247`

如果使用 DFlash：

```python
self.num_lookahead_tokens = self.num_spec_tokens + 1
```

位置：`scheduler.py:249`

如果使用 DSpark：

```python
self.num_lookahead_tokens = self.num_spec_tokens
```

位置：`scheduler.py:254`

所以 spec decode 不只影响采样，还会影响 KV block 预留：

```text
num_lookahead_tokens 越大，allocate_slots() 需要预留的 KV slots 越多。
```

---

## 16. Spec decode 如何影响 running 请求 token 数

running 请求计算本轮 token 数时使用：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`scheduler.py:510`

其中：

```text
request.num_tokens_with_spec = prompt tokens + output tokens + spec token ids
```

所以 speculative decoding 会让请求本轮一次调度多个 token：

```text
普通 decode：通常 num_new_tokens = 1
spec decode：可能 num_new_tokens = draft tokens + 1
```

例如：

```text
当前 prompt + output = 105 tokens
spec_token_ids = 4
num_tokens_with_spec = 109
num_computed_tokens = 104
num_new_tokens = 109 - 104 = 5
```

这表示本轮可能验证 4 个 draft tokens，并额外采样 1 个 token。

---

## 17. Spec decode 如何影响 KV block 分配

running 阶段分配 block 时：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`scheduler.py:572`

waiting 阶段也会传：

```python
num_lookahead_tokens=effective_lookahead_tokens
```

位置：`scheduler.py:942`

`allocate_slots()` 会把 `lookahead` 算入需要 slot 的范围：

```text
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```

位置：`kv_cache_manager.py:315`

因此：

```text
即使 draft tokens 后续可能被拒绝，
Scheduler 也需要提前给它们预留 KV slots。
```

async KV load + spec lookahead 有特殊处理：

```python
limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
effective_lookahead_tokens = (
    0 if limit_lookahead_tokens else self.num_lookahead_tokens
)
```

位置：`scheduler.py:917`

原因是 async KV load 本轮不运行 forward，要等 KV transfer 完成后再分配 speculative lookahead slots，避免本地和远端 block 数不匹配。

---

## 18. Spec decode 如何进入 SchedulerOutput

running 阶段中，如果请求有 `spec_token_ids`：

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

位置：`scheduler.py:631`

含义：

```text
Scheduler 记录本轮实际要验证的 draft tokens，
写入 scheduled_spec_decode_tokens，
然后清空 request 上旧的 spec tokens。
```

最终 `SchedulerOutput` 中包含：

```python
scheduled_spec_decode_tokens=scheduled_spec_decode_tokens
```

位置：`scheduler.py:1147`

同时还有：

```python
num_spec_tokens_to_schedule=num_spec_tokens_to_schedule
```

位置：`scheduler.py:1159`

如果启用了 dynamic speculative decoding，会根据本轮 batch size 查表：

```python
if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
    num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
        len(num_scheduled_tokens)
    ]
```

位置：`scheduler.py:1137`

---

## 19. Spec token 被拒绝后如何回退

Worker 返回后，在 `update_from_output()` 中会根据实际接受的 draft token 数修正状态：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`scheduler.py:1647`

如果有 token 被拒绝：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
```

位置：`scheduler.py:1656`

如果 async scheduling 下 placeholder 也包含这些 spec token，还要同步回退：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`scheduler.py:1660`

这说明 spec decode 是乐观调度：

```text
schedule 阶段先把 draft tokens 当作可执行 token 调度出去；
Worker 返回后再根据接受 / 拒绝情况修正 num_computed_tokens。
```

---

## 20. LoRA max_loras 如何限制 waiting 请求

LoRA 限制主要发生在 waiting 阶段。

running 阶段结束后，Scheduler 先记录本轮已经调度的 LoRA：

```python
scheduled_loras = set(
    req.lora_request.lora_int_id
    for req in scheduled_running_reqs
    if req.lora_request and req.lora_request.lora_int_id > 0
)
assert len(scheduled_loras) <= self.lora_config.max_loras
```

位置：`scheduler.py:663`

waiting 阶段，如果当前请求会引入新的 LoRA，并且本轮 LoRA 种类已经达到上限：

```python
if (
    self.lora_config
    and request.lora_request
    and (
        len(scheduled_loras) == self.lora_config.max_loras
        and request.lora_request.lora_int_id not in scheduled_loras
    )
):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:705`

含义是：

```text
一轮调度中最多同时使用 max_loras 种 LoRA；
如果 waiting 请求需要新的 LoRA 类型，而本轮已经满了，
它会被临时跳过。
```

这类请求不是 blocked status，通常仍然是 `WAITING`，只是本轮被放入 `skipped_waiting`。

---

## 21. Mamba block alignment 如何裁剪 token 数

如果模型含 Mamba 层，并且 cache 模式是 align：

```python
self.need_mamba_block_aligned_split = (
    self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
)
```

位置：`scheduler.py:299`

Scheduler 会在 running / waiting 阶段调用：

```python
num_new_tokens = self._mamba_block_aligned_split(...)
```

running 阶段位置：`scheduler.py:546`

waiting 阶段位置：`scheduler.py:903`

核心目的：

```text
Mamba state cache 对 chunk 边界敏感；
为了让 Mamba state 可以被缓存，prefill chunk 要尽量按 block_size 对齐。
```

`_mamba_block_aligned_split()` 中会按 block 边界裁剪 chunk 结束位置：

```python
aligned_end = num_computed_tokens_after_sched // block_size * block_size
num_new_tokens = max(aligned_end - num_computed_tokens, 0)
```

位置：`scheduler.py:390`

如果对齐后 `num_new_tokens == 0`：

```text
running 阶段：跳过当前 request，继续看后面的 running；
waiting 阶段：break，停止 waiting 调度。
```

此外，它还包含 Marconi cache admission 优化：如果存在足够长的未缓存 common prefix，可以把 chunk 切到 common prefix 长度，优先缓存公共前缀。

---

## 22. DP prefill balancing 如何延后 prefill

`schedule()` 支持参数：

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
```

位置：`scheduler.py:433`

如果当前 step 需要 throttle，并且不是 prefill capacity bound，且 running 中存在 decode 请求：

```python
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```

位置：`scheduler.py:473`

running 阶段，如果请求是 prefill chunk：

```python
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

位置：`scheduler.py:504`

waiting 阶段，如果要提交新的本地 prefill：

```python
elif defer_prefills and request.num_computed_tokens == 0:
    break
```

位置：`scheduler.py:838`

但 async KV load 是例外：

```text
async KV load 可以启动，因为它不提交本地 prefill compute；
真正的新 prefill 计算会被延后。
```

所以 DP prefill balancing 的效果是：

```text
在某些非 cadence-aligned step 上优先保证 decode，
把 prefill chunk 或新 prefill 延后到更合适的 step。
```

---

## 23. Pause state 如何影响 running / waiting

Scheduler 有三种 pause 状态：

| 状态 | 调度行为 |
|---|---|
| `UNPAUSED` | 正常调度 running 和 waiting |
| `PAUSED_NEW` | running 可以继续推进，但 waiting 不会进入 running |
| `PAUSED_ALL` | 所有请求都暂停，本轮 `token_budget = 0` |

`PAUSED_ALL` 在 `schedule()` 开头处理：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
```

位置：`scheduler.py:453`

waiting 阶段要求：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`scheduler.py:674`

因此：

```text
PAUSED_NEW：不会把 token_budget 清零，running 仍可继续；但 waiting 阶段不执行。
PAUSED_ALL：token_budget 清零，running / waiting 都不会调度。
```

---

## 24. KV Connector metadata 如何同步给 Worker

KV Connector 的调度插入点主要有三个：

```text
1. waiting 阶段查询外部 KV 命中：
   connector.get_num_new_matched_tokens()

2. block 分配成功后通知 connector：
   connector.update_state_after_alloc()

3. SchedulerOutput 构造后生成 metadata：
   connector.build_connector_meta()
```

构造 metadata 的代码是：

```python
if self.connector is not None:
    meta = self._build_kv_connector_meta(self.connector, scheduler_output)
    scheduler_output.kv_connector_metadata = meta
```

位置：`scheduler.py:1166`

注释说明这个函数有多个目的：

```python
# 1. Plan the KV cache store
# 2. Wrap up all the KV cache load / save ops into an opaque object
# 3. Clear the internal states of the connector
```

位置：`scheduler.py:1162`

所以 SchedulerOutput 不只是告诉 Worker 哪些 token 要 forward，还会携带：

```text
外部 KV load / save 所需的元数据。
```

Worker 执行后，会通过 `KVConnectorOutput.finished_recving` / `finished_sending` 等字段反馈给 Scheduler，后续再更新请求状态或释放 blocks。

---

## 25. ECConnector metadata 如何同步给 Worker

ECConnector 类似，但处理的是 encoder cache。

SchedulerOutput 构造后：

```python
if self.ec_connector is not None:
    ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
        scheduler_output
    )
    scheduler_output.ec_connector_metadata = ec_meta
```

位置：`scheduler.py:1171`

这说明：

```text
如果本轮需要从外部 encoder cache 加载，
或者需要保存 encoder cache，
相关元数据会进入 scheduler_output.ec_connector_metadata。
```

它和 `scheduled_encoder_inputs` 的区别是：

```text
scheduled_encoder_inputs：
  表示本轮要本地计算哪些 encoder inputs。

ec_connector_metadata：
  表示本轮有哪些 encoder cache load / save 需要由 connector 处理。
```

---

## 26. SchedulerOutput 中和附加能力相关的字段

构造 `SchedulerOutput` 时，附加能力相关字段包括：

```python
SchedulerOutput(
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    num_common_prefix_blocks=num_common_prefix_blocks,
    preempted_req_ids=self.reset_preempted_req_ids,
    finished_req_ids=self.finished_req_ids,
    free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
    new_block_ids_to_zero=new_block_ids_to_zero,
    kv_cache_block_copies=pending_kv_cache_block_copies,
    num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
)
```

位置：`scheduler.py:1142`

含义：

| 字段 | 作用 |
|---|---|
| `scheduled_spec_decode_tokens` | 本轮要验证的 draft/spec tokens |
| `scheduled_encoder_inputs` | 本轮要本地处理的 encoder inputs |
| `num_common_prefix_blocks` | running 请求公共 prefix blocks，可用于 cascade attention |
| `preempted_req_ids` | 本轮需要重置的被抢占请求 id |
| `finished_req_ids` | 上一轮到本轮之间完成的请求 id |
| `free_encoder_mm_hashes` | 本轮释放的 encoder cache item |
| `new_block_ids_to_zero` | 需要清零的新 KV blocks |
| `kv_cache_block_copies` | 本轮需要随执行完成的 KV block copy 计划 |
| `num_spec_tokens_to_schedule` | 后续 draft/spec token 调度数量 |
| `kv_connector_metadata` | KV Connector load/save metadata |
| `ec_connector_metadata` | Encoder Cache Connector metadata |

这些字段共同保证：

```text
Worker 拿到的不只是“请求和 token 数”，
还包括执行这一轮 forward / load / save 所需的全部辅助信息。
```

---

## 27. 一个完整例子：多模态 waiting 请求进入 running

假设：

```text
waiting req-a:
  prompt 中包含 image placeholder
  本轮 num_new_tokens 覆盖到这个 image placeholder
  encoder_compute_budget 足够
  encoder cache 有空间
```

流程：

```text
1. waiting 阶段计算 num_new_tokens；
2. request.has_encoder_inputs 为 True；
3. 调用 _try_schedule_encoder_inputs()；
4. get_mm_features_in_window() 找到本轮覆盖的 image feature；
5. encoder cache 未命中；
6. encoder budget 足够，加入 encoder_inputs_to_schedule；
7. scheduled_encoder_inputs[req-a] = [image_index]；
8. encoder_cache_manager.allocate(req-a, image_index)；
9. req-a 正常进入 running。
```

最终 SchedulerOutput 会同时包含：

```text
num_scheduled_tokens[req-a]
scheduled_encoder_inputs[req-a]
```

Worker 因此知道：

```text
本轮不仅要处理 decoder tokens，还要先处理该 image encoder input。
```

---

## 28. 一个完整例子：结构化输出 grammar 未就绪

假设：

```text
req-b 使用 structured output；
grammar 还没有构造完成；
request.status = WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR。
```

进入 Scheduler 时：

```text
_enqueue_waiting_request(req-b)
  → 因为是 blocked waiting status
  → 放入 skipped_waiting
```

某轮 waiting 调度时：

```text
1. 从 skipped_waiting 选中 req-b；
2. _try_promote_blocked_waiting_request(req-b)；
3. structured_output_req.grammar 仍为空；
4. return False；
5. req-b 被放回 step_skipped_waiting；
6. 本轮不进入 running。
```

等 grammar ready 后：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  → WAITING
  → 后续可正常进入 running
```

---

## 29. 一个完整例子：spec decode running 请求

假设：

```text
running req-c:
  prompt + output = 105 tokens
  spec_token_ids = [a, b, c, d]
  num_computed_tokens = 104
```

则：

```text
num_tokens_with_spec = 109
num_new_tokens = 109 - 104 = 5
```

调度时：

```text
1. allocate_slots(req-c, num_new_tokens=5, num_lookahead_tokens=...)；
2. scheduled_spec_decode_tokens[req-c] = [a, b, c, d]；
3. request.spec_token_ids = []；
4. SchedulerOutput 带上 scheduled_spec_decode_tokens。
```

Worker 返回后，如果只接受了 2 个 draft token：

```text
num_rejected = 4 - 2 = 2
request.num_computed_tokens -= 2
```

这样 Scheduler 的进度会从乐观调度状态回退到真实接受状态。

---

## 30. 一个完整例子：LoRA 限制导致 waiting 跳过

假设：

```text
max_loras = 2
本轮 scheduled_loras = {lora-1, lora-2}
waiting 队头 req-d 使用 lora-3
```

waiting 阶段检查：

```text
len(scheduled_loras) == max_loras
and lora-3 not in scheduled_loras
```

成立，于是：

```text
1. req-d 从原 waiting 队列 pop；
2. 放入 step_skipped_waiting；
3. 本轮不进入 running；
4. 下一轮再尝试。
```

注意：

```text
req-d 不是 blocked status；
它只是因为本轮 LoRA 种类限制被临时跳过。
```

---

## 31. 容易疑惑的点

### 30.1 encoder input 会消耗 token_budget 吗？

不直接消耗 decoder `token_budget`。

encoder input 有自己的：

```python
encoder_compute_budget
```

但 encoder 资源不足会裁剪 `num_new_tokens`，因此会间接影响 decoder token 调度。

### 30.2 structured output 会影响 prefill 吗？

grammar bitmask 主要影响 decode 阶段。

但 grammar 未就绪时，请求会停在：

```python
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

这会影响它能否进入 running。

### 30.3 spec decode 为什么会影响 KV block 分配？

因为 spec / lookahead tokens 也需要 KV slots。

即使 draft tokens 后续可能被拒绝，Scheduler 也要先为它们预留可写位置。

### 30.4 LoRA 限制为什么只跳过 waiting？

running 请求已经在执行流中，本轮先统计 scheduled running 里的 LoRA。

waiting 请求如果会引入新的 LoRA 且超过 `max_loras`，就被本轮跳过，避免同一批次中 LoRA 种类过多。

### 30.5 Mamba 对齐为什么会让请求本轮不能调度？

如果当前 token budget 或剩余 token 不能形成合法 block-aligned chunk，`_mamba_block_aligned_split()` 可能把 `num_new_tokens` 裁成 0。

running 阶段会跳过该请求；waiting 阶段会停止 waiting 调度。

### 30.6 async KV load 算不算 prefill？

它属于 prefill 相关的资源准备，但本轮不做本地 forward。

因此在 `defer_prefills` 时，async KV load 可以启动，而新的本地 prefill compute 会被延后。

---

## 32. 从“回答问题”的角度总结

如果要问：

```text
多模态 encoder 输入、结构化输出、投机解码等附加能力如何同步调度？
```

Scheduler 的回答是：

```text
它们都嵌入在同一轮 schedule() 的 running / waiting 调度流程中。

encoder input 通过 _try_schedule_encoder_inputs() 影响 num_new_tokens 和 encoder budget；
structured output 通过 blocked status、grammar bitmask 和 accept_tokens 影响请求状态和输出合法性；
spec decode 通过 num_tokens_with_spec、lookahead blocks、scheduled_spec_decode_tokens 和回退逻辑影响调度；
LoRA、Mamba、DP prefill balancing、pause state 则分别在 waiting 接纳、token 裁剪和调度入口处限制请求是否能跑。

最后，SchedulerOutput 会把这些能力需要的 metadata 一起交给 Worker。
```

---

## 33. 最关键的插入点总图

```text
schedule()
  │
  ├─ pause state
  │    └─ PAUSED_ALL: token_budget = 0
  │
  ├─ running 阶段
  │    ├─ async / PP cadence
  │    ├─ DP prefill balancing: defer prefill chunk
  │    ├─ spec decode: num_tokens_with_spec / placeholders
  │    ├─ encoder input: _try_schedule_encoder_inputs()
  │    ├─ Mamba: _mamba_block_aligned_split()
  │    ├─ KV block allocation / preemption
  │    ├─ scheduled_spec_decode_tokens
  │    └─ scheduled_encoder_inputs
  │
  ├─ waiting 阶段
  │    ├─ blocked status promotion
  │    │    ├─ WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  │    │    ├─ WAITING_FOR_REMOTE_KVS
  │    │    └─ WAITING_FOR_STREAMING_REQ
  │    ├─ LoRA max_loras limit
  │    ├─ prefix / external KV cache
  │    ├─ async KV load
  │    ├─ encoder input: _try_schedule_encoder_inputs()
  │    ├─ Mamba: _mamba_block_aligned_split()
  │    └─ KV block allocation
  │
  ├─ SchedulerOutput
  │    ├─ scheduled_encoder_inputs
  │    ├─ scheduled_spec_decode_tokens
  │    ├─ num_spec_tokens_to_schedule
  │    ├─ kv_connector_metadata
  │    └─ ec_connector_metadata
  │
  └─ update_from_output()
       ├─ spec token accepted / rejected
       ├─ grammar.accept_tokens()
       ├─ KV connector finished_recving / finished_sending
       └─ encoder cache release / free
```

---

## 34. 最关键的判断公式

```text
encoder input：
  if request.has_encoder_inputs:
      encoder_inputs_to_schedule,
      num_new_tokens,
      encoder_compute_budget,
      external_load_encoder_input = _try_schedule_encoder_inputs(...)

structured output blocked：
  WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    → grammar ready ? WAITING : skipped_waiting

structured output bitmask：
  req.use_structured_output and not req.is_prefill_chunk

spec decode running token 数：
  num_new_tokens = (
      request.num_tokens_with_spec
      + request.num_output_placeholders
      - request.num_computed_tokens
  )

spec decode lookahead：
  allocate_slots(..., num_lookahead_tokens=self.num_lookahead_tokens)

LoRA waiting 限制：
  if len(scheduled_loras) == max_loras
     and request.lora_request.lora_int_id not in scheduled_loras:
      move to step_skipped_waiting

Mamba 对齐：
  if need_mamba_block_aligned_split:
      num_new_tokens = _mamba_block_aligned_split(...)

DP prefill balancing：
  defer_prefills = (
      throttle_prefills and not prefill_capacity_bound
  ) and any(not r.is_prefill_chunk for r in running)

pause state：
  PAUSED_ALL → token_budget = 0
  PAUSED_NEW → skip waiting stage

connector metadata：
  scheduler_output.kv_connector_metadata = connector.build_connector_meta(...)
  scheduler_output.ec_connector_metadata = ec_connector.build_connector_meta(...)
```

---

## 35. 和前后问题的关系

前几篇已经拆开了 Scheduler 的主流程：

```text
01_request_states.md：请求在哪些队列和状态中
02_token_budget.md：本轮最多能调度多少 token
03_running_decode_prefill.md：running 请求如何继续推进
04_waiting_to_running.md：waiting 请求如何进入 running
05_prefix_and_external_kv_hits.md：cache 命中如何减少计算
06_kv_block_allocation_and_preemption.md：KV block 如何分配与抢占
```

本篇解释的是：

```text
多模态、结构化输出、spec decode、LoRA、Mamba、pause、connector metadata 等横切能力，
如何嵌入上述主流程。
```

下一篇 `08_update_after_worker_output.md` 应该继续讲：

```text
Worker 执行完 SchedulerOutput 后，Scheduler 如何处理 sampled tokens、spec 接受/拒绝、grammar 状态、KV connector output、请求停止和资源释放。
```
