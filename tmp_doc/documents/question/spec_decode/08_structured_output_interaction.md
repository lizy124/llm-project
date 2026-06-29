# 08. Spec decode 如何和 structured output / grammar 交互？

源码位置：

- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/structured_output/__init__.py`
- `code/vllm/vllm/v1/structured_output/request.py`
- `code/vllm/vllm/v1/structured_output/utils.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`

本问题关注：structured output / grammar 约束如何影响 spec decode 的 draft tokens、target logits、bonus logits、accepted tokens，以及 async scheduling 下为什么需要先拿到真实 draft tokens 再计算 grammar bitmask。

---

## 1. 一句话回答

Structured output 让 spec decode 不能只按概率接受 draft tokens，还必须让每个可能输出的位置都符合 grammar 状态。

它主要介入四个位置：

```text
1. 请求进入系统时：
   编译 grammar，并让请求在 grammar ready 前不可调度。

2. draft tokens 写回 Scheduler 时：
   用 grammar.validate_tokens() 裁剪不合法 draft。

3. target logits 采样前：
   生成并应用 grammar bitmask，约束 target / bonus logits。

4. 输出回收后：
   用 grammar.accept_tokens() 推进 grammar 状态。
```

所以可以记成：

```text
structured output 约束 draft 进入、logits 采样、输出提交三个边界。
```

---

## 2. 整体链路

```text
EngineCore.preprocess_add_request()
  → StructuredOutputManager.grammar_init()
  → Request.status = WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  → Scheduler 等 grammar ready 后调度

spec decode draft 产生后：
  → Scheduler.update_draft_token_ids()
      → grammar.validate_tokens(draft_tokens)
      → Request.spec_token_ids

Scheduler.schedule()
  → SchedulerOutput.scheduled_spec_decode_tokens
  → Scheduler.get_grammar_bitmask()
      → StructuredOutputManager.grammar_bitmask(..., scheduled_spec_decode_tokens)
      → GrammarOutput

GPUModelRunner.sample_tokens(grammar_output)
  → apply_grammar_bitmask(...)
  → _sample()
      → Sampler / RejectionSampler

Scheduler.update_from_output()
  → Request.append_output_token_ids(...)
  → grammar.accept_tokens(new_token_ids)
```

---

## 3. 请求为什么会等待 grammar ready

`Request` 初始化时会从 sampling params 中提取 structured output 配置：

```python
self.structured_output_request = StructuredOutputRequest.from_sampling_params(
    sampling_params
)
```

位置：`request.py:87` 到 `request.py:89`

如果是 generation 请求，并且存在 structured output：

```python
if self.structured_output_request is not None:
    self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

位置：`request.py:107` 到 `request.py:113`

这表示：

```text
请求不能立即进入正常 WAITING / RUNNING；
必须等 grammar 编译完成。
```

`Request.use_structured_output` 是：

```python
@property
def use_structured_output(self) -> bool:
    return self.structured_output_request is not None
```

位置：`request.py:242` 到 `request.py:244`

---

## 4. grammar 在哪里初始化

`EngineCore.preprocess_add_request()` 会在输入处理线程里初始化 grammar：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
return req, request.current_wave
```

位置：`engine/core.py:853` 到 `engine/core.py:875`

注释说明：

```text
grammar_init 只在 input processing thread 调用；
grammar compilation 可以异步；
Scheduler 调度前会检查 grammar compilation 状态。
```

位置：`engine/core.py:867` 到 `engine/core.py:874`

---

## 5. StructuredOutputManager.grammar_init() 做什么

入口：`structured_output/__init__.py:115`

如果 request 没有 structured output，直接返回：

```python
if request.structured_output_request is None:
    return
```

位置：`structured_output/__init__.py:115` 到 `structured_output/__init__.py:117`

第一次需要 structured output 时，会初始化 backend：

```text
xgrammar
guidance
outlines
lm-format-enforcer
```

位置：`structured_output/__init__.py:125` 到 `structured_output/__init__.py:165`

然后编译 grammar：

```python
if self._use_async_grammar_compilation:
    grammar = self.executor.submit(self._create_grammar, request)
else:
    grammar = self._create_grammar(request)
request.structured_output_request.grammar = grammar
```

位置：`structured_output/__init__.py:167` 到 `structured_output/__init__.py:171`

因此 `grammar` 可能是：

```text
Future[StructuredOutputGrammar]
或已经编译好的 StructuredOutputGrammar
```

---

## 6. Scheduler 如何等待 grammar ready

`StructuredOutputRequest.grammar` 属性会检查 Future 是否完成。

定义在：`structured_output/request.py:42`

```python
if isinstance(self._grammar, Future):
    try:
        self._grammar = self._grammar.result(timeout=0.0001)
        self.status = RequestStatus.WAITING
    except TimeoutError:
        return False
return True
```

位置：`structured_output/request.py:42` 到 `structured_output/request.py:53`

Scheduler 尝试把 blocked waiting request 提升回可调度状态时：

```python
if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
    structured_output_req = request.structured_output_request
    if not (structured_output_req and structured_output_req.grammar):
        return False
    request.status = RequestStatus.WAITING
    return True
```

位置：`scheduler.py:2401` 到 `scheduler.py:2406`

也就是说：

```text
grammar 未 ready：请求继续 blocked；
grammar ready：请求回到 WAITING，后续才能被 Scheduler 调度。
```

---

## 7. structured output 对 draft token 的第一层约束

Spec decode 的 draft tokens 是提前猜出来的。

如果请求有 grammar 约束，vLLM 会在 draft tokens 写回 Scheduler 侧 request 时先校验它们。

入口：`scheduler.py:1895`

```python
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
```

核心逻辑：

```python
if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

位置：`scheduler.py:1911` 到 `scheduler.py:1915`

含义：

```text
如果 draft token 序列从当前 grammar 状态出发不合法，
grammar.validate_tokens() 会裁剪到合法前缀。
```

这一步发生在：

```text
Worker / drafter 产生 draft tokens
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

---

## 8. prefill chunk 为什么忽略 draft grammar 校验

`update_draft_token_ids()` 里先检查：

```python
if request.is_prefill_chunk:
    if request.spec_token_ids:
        request.spec_token_ids = []
    continue
```

位置：`scheduler.py:1905` 到 `scheduler.py:1909`

原因是：

```text
prefill / chunked prefill 阶段还在处理 prompt；
spec decode draft token 是 decode 阶段的候选输出；
此时不应该把 draft tokens 当成 grammar output 去推进或验证。
```

所以 structured output 对 draft 的校验只在请求进入 decode 语义后有效。

---

## 9. should_advance() 决定什么时候推进 grammar

`StructuredOutputManager.should_advance()` 定义在：`structured_output/__init__.py:325`

普通 structured output：

```python
if not request.use_structured_output:
    return False
...
reasoner = self._get_reasoner(request)
if reasoner is None:
    return True
```

位置：`structured_output/__init__.py:325` 到 `structured_output/__init__.py:338`

如果涉及 reasoning parser，则要看是否允许在 reasoning 阶段约束：

```python
if self.enable_in_reasoning:
    return True
```

位置：`structured_output/__init__.py:340` 到 `structured_output/__init__.py:342`

如果 reasoning 尚未结束，一般不推进 grammar；但 structural tag 有 spec decode 特例：

```python
if (
    self.vllm_config.speculative_config is not None
    and structured_req.structured_output_key[0]
    == StructuredOutputOptions.STRUCTURAL_TAG
):
    return True
```

位置：`structured_output/__init__.py:359` 到 `structured_output/__init__.py:371`

注释说明：

```text
Structural tags model phased output，
speculative decoding must run grammar.validate_tokens
on draft tokens produced immediately after that transition.
```

位置：`structured_output/__init__.py:359` 到 `structured_output/__init__.py:365`

---

## 10. SchedulerOutput 中的 structured output 标记

`SchedulerOutput` 有两个字段：

```python
has_structured_output_requests: bool = False
pending_structured_output_tokens: bool = False
```

位置：`output.py:221` 到 `output.py:227`

含义：

```text
has_structured_output_requests：
  本轮 scheduled requests 中是否有需要 structured output 的请求。

pending_structured_output_tokens：
  本轮 scheduled requests 是否还缺少计算 grammar bitmask 所需的 output tokens。
```

另一个 spec decode 相关字段：

```python
num_invalid_spec_tokens: dict[str, int] | None = None
```

位置：`output.py:229` 到 `output.py:230`

用于记录 structured output 裁掉的无效 draft tokens 数，后面调整 spec decode acceptance 统计。

---

## 11. Scheduler 何时设置 has_structured_output_requests

在 `_update_after_schedule()` 中，每个 scheduled request 调度后会更新：

```python
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
scheduler_output.has_structured_output_requests |= (
    request.use_structured_output and not request.is_prefill_chunk
)
```

位置：`scheduler.py:1145` 到 `scheduler.py:1150`

含义：

```text
只有 structured output 请求，并且不是 prefill chunk，
才需要本轮 grammar bitmask。
```

这和 draft 校验逻辑一致：grammar 约束主要作用于 decode 输出阶段。

---

## 12. Scheduler 如何生成 GrammarOutput

入口：`scheduler.py:1440`

```python
def get_grammar_bitmask(
    self, scheduler_output: SchedulerOutput
) -> GrammarOutput | None:
```

如果本轮没有 structured output 请求：

```python
if not scheduler_output.has_structured_output_requests:
    return None
```

位置：`scheduler.py:1440` 到 `scheduler.py:1445`

然后收集 scheduled requests 中需要 structured output 且不在 prefill chunk 的请求：

```python
structured_output_request_ids = [
    req_id
    for req_id in scheduler_output.num_scheduled_tokens
    if (req := self.requests.get(req_id))
    and (req.use_structured_output and not req.is_prefill_chunk)
]
```

位置：`scheduler.py:1447` 到 `scheduler.py:1452`

再调用 manager：

```python
bitmask = self.structured_output_manager.grammar_bitmask(
    self.requests,
    structured_output_request_ids,
    scheduler_output.scheduled_spec_decode_tokens,
)
return GrammarOutput(structured_output_request_ids, bitmask)
```

位置：`scheduler.py:1456` 到 `scheduler.py:1461`

---

## 13. GrammarOutput 是什么

定义在：`output.py:262`

```python
@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: npt.NDArray[np.int32]
```

位置：`output.py:262` 到 `output.py:267`

它是 Scheduler 传给 Worker / ModelRunner 的 grammar 约束结果。

注意：

```text
grammar_bitmask 只包含 structured output requests 的 rows；
不是完整 batch 顺序。
```

所以 Worker 侧还要在 `apply_grammar_bitmask()` 中按 `input_batch.req_ids` 重新对齐。

---

## 14. StructuredOutputManager.grammar_bitmask() 的核心逻辑

入口：`structured_output/__init__.py:204`

```python
def grammar_bitmask(
    self,
    requests: dict[str, Request],
    structured_output_request_ids: list[str],
    scheduled_spec_decode_tokens: dict[str, list[int]],
) -> npt.NDArray[np.int32] | None:
```

位置：`structured_output/__init__.py:204` 到 `structured_output/__init__.py:209`

如果没有 structured output requests：

```python
if not structured_output_request_ids:
    return None
```

位置：`structured_output/__init__.py:210` 到 `structured_output/__init__.py:212`

它会预分配 bitmask：

```python
self._grammar_bitmask = self.backend.allocate_token_bitmask(
    max_batch_size * (1 + max_num_spec_tokens)
)
```

位置：`structured_output/__init__.py:214` 到 `structured_output/__init__.py:226`

为什么是 `1 + max_num_spec_tokens`？

```text
每个请求最多需要：
  num_spec_tokens 个 draft verification 位置
  + 1 个 bonus / normal sampling 位置
```

---

## 15. spec decode 下为什么需要多个 bitmask row

普通 decode 中，一个请求本轮只采样一个 token，所以只需要一个 grammar bitmask row。

spec decode 中，一个请求可能有多个可能输出位置：

```text
验证 draft_0 的 target logits row
验证 draft_1 的 target logits row
...
验证 draft_k 的 target logits row
bonus token logits row
```

每个位置对应的 grammar 状态不同。

例如 grammar 当前只允许 JSON：

```text
位置 0：可能允许 '{'
位置 1：如果 draft_0 是 '{'，可能允许 '"'
位置 2：如果 draft_1 是 '"'，可能允许 key token
bonus 位置：如果所有 draft 都接受后的下一个 grammar 状态
```

所以 `grammar_bitmask()` 需要针对每个 speculative position 逐步填 mask。

---

## 16. grammar_bitmask() 如何临时推进 grammar

核心循环：

```python
req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
if self.vllm_config.model_config.is_diffusion and req_tokens:
    token_iter = req_tokens
else:
    token_iter = itertools.chain(req_tokens, (-1,))
for token in token_iter:
    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
    if token == -1:
        apply_bitmask = False
    if apply_bitmask and not grammar.is_terminated():
        accepted = grammar.accept_tokens(req_id, [token])
        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
        state_advancements += 1
    cumulative_index += 1
if state_advancements > 0:
    grammar.rollback(state_advancements)
```

位置：`structured_output/__init__.py:275` 到 `structured_output/__init__.py:294`

它的含义是：

```text
1. 先为当前位置填 bitmask；
2. 如果当前位置对应一个 scheduled draft token，就临时 accept 这个 token；
3. 进入下一个 speculative position；
4. 最后 rollback，恢复 grammar 的真实状态。
```

这很关键：

```text
生成 bitmask 时会临时推进 grammar，
但不会永久改变 request 的 grammar 状态。
```

真正永久推进发生在 Scheduler.update_from_output() 后。

---

## 17. token_iter 为什么追加 -1

非 diffusion 情况下：

```python
token_iter = itertools.chain(req_tokens, (-1,))
```

位置：`structured_output/__init__.py:281` 到 `structured_output/__init__.py:282`

`-1` 表示：

```text
bonus token / normal sampling position
```

当 token 是 `-1`：

```python
if token == -1:
    apply_bitmask = False
```

位置：`structured_output/__init__.py:285` 到 `structured_output/__init__.py:287`

含义：

```text
为 bonus 位置填 bitmask，
但不要把 -1 当成真实 token 去推进 grammar。
```

所以每个 structured request 的 bitmask rows 数是：

```text
len(scheduled_spec_decode_tokens[req_id]) + 1
```

---

## 18. fill_bitmask 和 should_fill_bitmask

`_fill_bitmasks()` 会判断是否真的填 grammar mask：

```python
if apply_bitmask and not grammar.is_terminated():
    grammar.fill_bitmask(self._grammar_bitmask, index)
else:
    self._grammar_bitmask[index].fill_(self._full_mask)
```

位置：`structured_output/__init__.py:186` 到 `structured_output/__init__.py:197`

`should_fill_bitmask()` 处理 reasoning 相关逻辑：

```python
reasoner = self._get_reasoner(request)
if reasoner is not None:
    if self.enable_in_reasoning:
        return True
    ...
    return request.structured_output_request.reasoning_ended
return True
```

位置：`structured_output/__init__.py:305` 到 `structured_output/__init__.py:323`

如果不应该应用 bitmask，会填 full mask：

```text
相当于不限制 token。
```

---

## 19. GrammarOutput 如何传到 sample_tokens

同步 step 中：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`engine/core.py:486` 到 `engine/core.py:500`

也就是说：

```text
execute_model() 负责 forward / logits；
get_grammar_bitmask() 生成 grammar mask；
sample_tokens(grammar_output) 在采样前应用 mask。
```

---

## 20. ModelRunner 如何应用 grammar bitmask

`sample_tokens()` 中：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`gpu_model_runner.py:4452` 到 `gpu_model_runner.py:4456`

注意：

```text
grammar bitmask 在 _sample() 前应用。
```

因此它会影响：

```text
普通 Sampler 的采样；
RejectionSampler 中 bonus token 采样；
RejectionSampler 中 target logits 对 draft tokens 的验证与 recovered token 采样。
```

---

## 21. apply_grammar_bitmask() 为什么要重新排序

`GrammarOutput.grammar_bitmask` 的顺序是：

```text
structured_output_request_ids 顺序
```

但 Worker 的 batch 顺序是：

```text
input_batch.req_ids 顺序
```

两者不保证一致。

`apply_grammar_bitmask()` 注释说明：

```text
The order of the requests in the bitmask is not guaranteed to be the
same as the order of the requests in the gpu runner's batch.
We need to sort the bitmask to match the order of the requests used here.
```

位置：`utils.py:104` 到 `utils.py:109`

---

## 22. apply_grammar_bitmask() 如何处理 spec offset

核心逻辑：

```python
struct_out_req_batch_indices: dict[str, int] = {}
cumulative_offset = 0
spec_tokens = scheduler_output.scheduled_spec_decode_tokens
struct_out_req_ids = set(grammar_output.structured_output_request_ids)
for batch_index, req_id in enumerate(input_batch.req_ids):
    logit_index = batch_index + cumulative_offset
    cumulative_offset += len(spec_tokens.get(req_id, ()))
    if req_id in struct_out_req_ids:
        struct_out_req_batch_indices[req_id] = logit_index
```

位置：`utils.py:110` 到 `utils.py:121`

为什么 `logit_index = batch_index + cumulative_offset`？

在 spec decode 下，logits rows 不是一请求一行，而是：

```text
每个请求：1 + num_spec_tokens 行
```

如果前面某些请求有 draft tokens，后面请求的 logits row 会向后偏移。

例如：

```text
batch reqs = [r0, r1, r2]
r0 有 2 个 spec tokens
r1 有 0 个 spec tokens
r2 有 1 个 spec token

logits rows:
r0: row 0,1,2
r1: row 3
r2: row 4,5
```

所以 `r2` 的起始 row 不是 batch index 2，而是 4。

---

## 23. apply_grammar_bitmask() 如何展开每个 structured 请求

创建完整 logits 形状的 sorted bitmask：

```python
sorted_bitmask = np.full(
    shape=(logits.shape[0], grammar_bitmask.shape[1]),
    fill_value=-1,
    dtype=grammar_bitmask.dtype,
)
```

位置：`utils.py:123` 到 `utils.py:130`

然后按 `structured_output_request_ids` 重排：

```python
cumulative_index = 0
for req_id in grammar_output.structured_output_request_ids:
    num_spec_tokens = len(spec_tokens.get(req_id, ()))
    if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
        for i in range(1 + num_spec_tokens):
            bitmask_index = logit_idx + i
            sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
            out_indices.append(bitmask_index)
    cumulative_index += 1 + num_spec_tokens
```

位置：`utils.py:131` 到 `utils.py:139`

含义：

```text
对每个 structured request：
  把它的 draft verification rows + bonus row 的 bitmask
  放到 logits tensor 对应 rows 上。
```

---

## 24. apply_grammar_bitmask() 如何调用 xgrammar

先把 numpy bitmask 转为 device tensor：

```python
grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
    logits.device, non_blocking=True
)
```

位置：`utils.py:141` 到 `utils.py:144`

如果不是所有 logits rows 都需要 mask，就传 indices：

```python
skip_out_indices = len(out_indices) == logits.shape[0]
...
index_tensor = torch.tensor(out_indices, dtype=torch.int32, device="cpu", pin_memory=pin_memory)
index_tensor = index_tensor.to(logits.device, non_blocking=True)
```

位置：`utils.py:146` 到 `utils.py:162`

GPU 情况下：

```python
xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

位置：`utils.py:151` 到 `utils.py:164`

这一步会原地修改 logits：

```text
不允许的 token logits 被 mask 掉。
```

---

## 25. grammar bitmask 会约束哪些 logits row

在 spec decode 下，`logits` 来自：

```text
hidden_states[SpecDecodeMetadata.logits_indices]
  → compute_logits()
```

它包含：

```text
target verification rows
bonus rows
```

`apply_grammar_bitmask()` 对 structured output 请求的：

```text
1 + len(scheduled_spec_decode_tokens[req_id])
```

行都应用 bitmask。

因此：

```text
- draft verification 的 target logits 会被 grammar 约束；
- all-accepted 后的 bonus logits 也会被 grammar 约束；
- 无 draft 请求的普通采样 logits 也会被 grammar 约束。
```

---

## 26. RejectionSampler 中 grammar mask 的作用

`apply_grammar_bitmask()` 在 `_sample()` 前执行。

所以进入 `RejectionSampler` 的 `logits` 已经被 mask。

`RejectionSampler.forward()` 会：

```text
1. 从 masked logits 中取 bonus_logits；
2. 用普通 Sampler 从 bonus_logits 采样 bonus token；
3. 从 masked logits 中取 target_logits；
4. 用 target_logits 验证 draft tokens；
5. 拒绝后从 target / residual distribution 采样 recovered token。
```

相关入口：`rejection_sampler.py:88`

这意味着 structured output 会影响：

```text
accepted 判断：
  target_prob 会受 grammar mask 影响。

recovered token：
  residual / target distribution 已经排除非法 token。

bonus token：
  bonus logits 已经排除非法 token。
```

---

## 27. 为什么 draft token 既要 validate，又要 bitmask

这两个步骤解决的问题不同。

### 27.1 validate_tokens

发生在 draft token 写回 Scheduler 时：

```text
Worker 预测 draft tokens
  → Scheduler.update_draft_token_ids()
  → grammar.validate_tokens()
```

作用：

```text
提前裁剪不可能合法的 draft token 前缀，避免下一轮调度无效 draft。
```

### 27.2 grammar bitmask

发生在采样前：

```text
Scheduler.get_grammar_bitmask()
  → ModelRunner.apply_grammar_bitmask()
  → Sampler / RejectionSampler
```

作用：

```text
保证 target verification、recovered token、bonus token 的采样空间合法。
```

所以：

```text
validate_tokens 管 draft token 是否值得调度；
bitmask 管 target logits 最终能采什么。
```

---

## 28. async scheduling 下为什么有 pending_structured_output_tokens

在 async batch queue 路径中，Scheduler 可能先调度下一步，但 grammar bitmask 需要上一轮真实输出 / draft 信息才能正确计算。

EngineCore 逻辑：

```python
if not scheduler_output.pending_structured_output_tokens:
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
else:
    deferred_scheduler_output = scheduler_output
```

位置：`engine/core.py:555` 到 `engine/core.py:571`

含义：

```text
如果 grammar bitmask 所需 tokens 都已齐备，立即 sample；
否则先 defer sampling，等前一个 output 回收后再处理。
```

---

## 29. async deferred sampling 中如何修正 draft tokens

当存在 deferred scheduler output：

```python
if deferred_scheduler_output:
    if self.check_for_draft_tokens:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )
    grammar_output = self.scheduler.get_grammar_bitmask(
        deferred_scheduler_output
    )
    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
```

位置：`engine/core.py:612` 到 `engine/core.py:630`

注释说明：

```text
When draft tokens are used with structured output,
validate them before computing the grammar bitmask for the deferred request.
```

位置：`engine/core.py:612` 到 `engine/core.py:620`

也就是说：

```text
async 场景下，不能直接用占位 draft tokens 生成 grammar bitmask；
必须先拿到真实 draft tokens，并过滤无效 token。
```

---

## 30. update_draft_token_ids_in_output() 做什么

入口：`scheduler.py:1917`

它不是写 `Request.spec_token_ids`，而是直接修正已经生成的：

```text
SchedulerOutput.scheduled_spec_decode_tokens
```

核心逻辑：

```python
placeholder_spec_tokens = sched_spec_tokens.get(req_id)
if not placeholder_spec_tokens:
    continue

orig_num_spec_tokens = len(placeholder_spec_tokens)
del spec_token_ids[orig_num_spec_tokens:]

if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)

num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
if num_invalid_tokens:
    spec_token_ids.extend([-1] * num_invalid_tokens)
    num_invalid_spec_tokens[req_id] = num_invalid_tokens

sched_spec_tokens[req_id] = spec_token_ids
scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
```

位置：`scheduler.py:1917` 到 `scheduler.py:1953`

它做了三件事：

```text
1. 把真实 draft tokens 裁到本轮 scheduled spec token 数；
2. 用 grammar.validate_tokens() 过滤不合法 draft；
3. 用 -1 padding 回原长度，保持本轮 logits / tensor layout 不变。
```

---

## 31. 为什么无效 draft 要 padding 为 -1

在 async deferred 场景中，`SchedulerOutput` 已经决定了本轮：

```text
num_scheduled_tokens
scheduled_spec_decode_tokens 的原始长度
input / logits layout
```

如果 grammar 校验后直接缩短 list，会破坏后续 layout 对齐。

所以代码：

```python
num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
if num_invalid_tokens:
    spec_token_ids.extend([-1] * num_invalid_tokens)
    num_invalid_spec_tokens[req_id] = num_invalid_tokens
```

位置：`scheduler.py:1946` 到 `scheduler.py:1949`

含义：

```text
无效 draft token 不再作为合法 token 使用，
但占位长度保留，
让 grammar bitmask / logits row / acceptance stats 都能对齐。
```

`num_invalid_spec_tokens` 后续用于 spec decode stats：

```python
spec_decoding_stats = self.make_spec_decoding_stats(
    spec_decoding_stats,
    num_draft_tokens=num_draft_tokens,
    num_accepted_tokens=num_accepted,
    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
    request_id=req_id,
)
```

位置：`scheduler.py:1568` 到 `scheduler.py:1574`

---

## 32. grammar 状态何时真正推进

生成 bitmask 时会临时 `accept_tokens()` 然后 `rollback()`。

真正推进发生在 Scheduler 回收输出之后。

在 `Scheduler.update_from_output()` 中：

```python
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
```

位置：`scheduler.py:1588` 到 `scheduler.py:1592`

随后：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    struct_output_request = request.structured_output_request
    assert struct_output_request is not None
    assert struct_output_request.grammar is not None
    if not struct_output_request.grammar.accept_tokens(
        req_id, new_token_ids
    ):
        logger.error(...)
        request.status = RequestStatus.FINISHED_ERROR
        request.resumable = False
        stopped = True
```

位置：`scheduler.py:1598` 到 `scheduler.py:1614`

这一步才是：

```text
accepted / recovered / bonus tokens 被正式提交后，推进 grammar FSM。
```

---

## 33. grammar.accept_tokens() 拒绝输出时会怎样

如果 grammar 拒绝已经采样出来的 `new_token_ids`：

```python
logger.error(
    "Unexpected: grammar rejected tokens %s for request %s. "
    "Terminating request.",
    new_token_ids,
    req_id,
)
request.status = RequestStatus.FINISHED_ERROR
request.resumable = False
stopped = True
```

位置：`scheduler.py:1605` 到 `scheduler.py:1614`

这属于异常情况。

理论上：

```text
validate_tokens + grammar_bitmask 应该保证 sampled tokens 合法。
```

如果最终 `accept_tokens()` 仍拒绝，说明 grammar 状态、bitmask 对齐或输出回收存在不一致，vLLM 会终止请求。

---

## 34. structured output 与 accepted / rejected draft 的关系

RejectionSampler 输出的是：

```text
accepted draft prefix + recovered token
或
all accepted draft + bonus token
```

Scheduler 回收时：

```text
new_token_ids = generated_token_ids
```

位置：`scheduler.py:1580` 到 `scheduler.py:1592`

然后 grammar 用整个 `new_token_ids` 推进：

```text
accepted draft tokens：
  推进 grammar。

recovered token：
  推进 grammar。

bonus token：
  推进 grammar。

rejected draft token：
  不在 new_token_ids 中，不推进 grammar。
```

这正好符合 spec decode 的语义：

```text
只有真正成为输出的 token 才改变 grammar 状态。
```

---

## 35. grammar bitmask 和 RejectionSampler 的顺序

顺序是：

```text
apply_grammar_bitmask(logits)
  → RejectionSampler.forward(..., logits, ...)
```

位置：

- `gpu_model_runner.py:4452` 到 `gpu_model_runner.py:4456`
- `gpu_model_runner.py:3592` 到 `gpu_model_runner.py:3598`

因此 `RejectionSampler` 看到的是已经被 grammar 约束的 logits。

这会影响：

```text
bonus_logits：
  普通 Sampler 只能从合法 token 中采 bonus。

target_logits：
  target probability / argmax 已经排除了非法 token。

recovered token：
  residual distribution 基于 masked target logits 计算。
```

---

## 36. 一个完整例子

假设当前 grammar 状态下允许输出 JSON：

```text
当前 output: ""
允许 token: "{"
```

drafter 预测：

```text
["{", "\"", "bad"]
```

### 36.1 draft 写回 Scheduler

`update_draft_token_ids()` 调用：

```text
grammar.validate_tokens(["{", "\"", "bad"])
```

如果第三个 token 不合法，得到：

```text
["{", "\""]
```

然后：

```text
Request.spec_token_ids = ["{", "\""]
```

### 36.2 生成 bitmask

Scheduler 调度这两个 draft 后：

```text
scheduled_spec_decode_tokens = ["{", "\""]
```

`grammar_bitmask()` 会生成三行 mask：

```text
row 0：当前状态允许什么，用来验证 "{"
临时 accept "{"
row 1：accept "{" 后允许什么，用来验证 "\""
临时 accept "\""
row 2：accept 两个 draft 后允许什么，用来采 bonus
rollback 2 步
```

### 36.3 采样和回收

如果 target 接受两个 draft 并采样 bonus：

```text
new_token_ids = ["{", "\"", bonus]
```

Scheduler 最终：

```text
request.append_output_token_ids(...)
grammar.accept_tokens(new_token_ids)
```

grammar 真实状态推进三步。

---

## 37. 为什么 grammar_bitmask 要 rollback

`grammar_bitmask()` 需要知道：

```text
如果 draft token 被接受，下一个位置允许什么？
```

所以它必须临时推进 grammar。

但此时 target model 还没有验证 draft，draft 可能被拒绝。

如果不 rollback，会把“尚未确认的 draft tokens”提前提交到 grammar 状态，导致：

```text
后续请求状态错误；
recovered token / bonus token 对不上 grammar；
Scheduler.update_from_output() 再 accept 时重复推进。
```

所以源码在每个 request 处理后：

```python
if state_advancements > 0:
    grammar.rollback(state_advancements)
```

位置：`structured_output/__init__.py:293` 到 `structured_output/__init__.py:294`

---

## 38. 为什么 apply_grammar_bitmask 要使用 out_indices

如果 batch 中不是所有请求都有 structured output，或者部分 logits row 不需要 mask，那么不应该对所有 rows 应用 grammar bitmask。

代码：

```python
out_indices = []
...
out_indices.append(bitmask_index)
...
skip_out_indices = len(out_indices) == logits.shape[0]
```

位置：`utils.py:123` 到 `utils.py:149`

如果 `out_indices` 覆盖全部 logits rows，就不传 indices；否则只 mask 指定 rows：

```python
xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
```

位置：`utils.py:151` 到 `utils.py:164`

这避免非 structured output 请求被错误约束。

---

## 39. invalid spec tokens 如何影响统计

在 async deferred path 中，无效 draft 会记录到：

```python
scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
```

位置：`scheduler.py:1951` 到 `scheduler.py:1953`

`Scheduler.update_from_output()` 统计 spec decode stats 时传入：

```python
num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens
```

位置：`scheduler.py:1568` 到 `scheduler.py:1574`

这样 acceptance rate 统计可以区分：

```text
target model 拒绝的 draft；
grammar 提前判定无效的 draft。
```

否则 grammar 裁掉的 token 可能会被错误算成 target rejection，影响 spec decode metrics。

---

## 40. structured output 和 ordinary decode 的区别

普通 decode：

```text
每个 structured request 只有一个 logits row；
grammar_bitmask 只需一行；
Sampler 采样一个 token；
Scheduler.accept_tokens([token]) 推进 grammar。
```

spec decode：

```text
每个 structured request 有 len(draft_tokens) + 1 个 logits rows；
grammar_bitmask 需要模拟 draft 接受路径生成多行；
RejectionSampler 可能输出多个 token；
Scheduler.accept_tokens(new_token_ids) 一次推进多个 token。
```

这就是 spec decode 与 structured output 交互的核心复杂度。

---

## 41. 容易混淆的点

### 41.1 validate_tokens 会永久推进 grammar 吗？

不会。

它只校验 / 裁剪 draft token 序列，避免不合法 draft 进入调度。

### 41.2 grammar_bitmask() 里的 accept_tokens 会永久推进 grammar 吗？

不会。

`grammar_bitmask()` 会临时推进，然后 `rollback(state_advancements)`。

### 41.3 真正推进 grammar 的地方在哪里？

在 `Scheduler.update_from_output()` 中，对最终 `new_token_ids` 调用：

```text
grammar.accept_tokens(req_id, new_token_ids)
```

### 41.4 rejected draft 会推进 grammar 吗？

不会。

Rejected draft 不在 `new_token_ids` 中。

### 41.5 bonus token 会被 grammar 约束吗？

会。

`grammar_bitmask()` 为 `req_tokens + (-1,)` 的 bonus position 生成 bitmask，`apply_grammar_bitmask()` 会应用到 bonus logits row。

### 41.6 为什么 async 下要 update_draft_token_ids_in_output()？

因为 deferred scheduler output 可能已经包含占位 draft tokens，但 grammar bitmask 必须基于真实 draft tokens 生成。

---

## 42. 总结

Spec decode 与 structured output 的完整交互可以概括为：

```text
grammar ready gate：
  请求必须等 grammar 编译完成后才能调度。

draft validation：
  draft tokens 写回 Scheduler 时先被 grammar.validate_tokens() 裁剪。

multi-position bitmask：
  每个 structured request 生成 len(draft_tokens) + 1 行 grammar bitmask。

logits masking：
  apply_grammar_bitmask() 根据 spec offset 把 bitmask 对齐到 target / bonus logits rows。

sampling：
  RejectionSampler 在 masked logits 上接受 / 拒绝 draft，并采 recovered / bonus。

state commit：
  Scheduler 只用最终 new_token_ids 推进 grammar.accept_tokens()。

async repair：
  deferred sampling 前先用真实 draft tokens 修正 SchedulerOutput，并用 -1 padding 无效 draft。
```

如果只记一句话：

```text
structured output 在 spec decode 中不是只 mask 最后一个 logits，而是要沿着“draft 可能被连续接受”的路径，为每个 draft 验证位置和 bonus 位置生成并对齐 grammar mask。
```
