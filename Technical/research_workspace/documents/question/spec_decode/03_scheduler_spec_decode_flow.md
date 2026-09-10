# 03. Scheduler 如何调度 spec decode？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/outputs.py`

本问题关注：Scheduler 在 speculative decoding 中到底做什么；它如何把 `Request.spec_token_ids` 转成 `SchedulerOutput.scheduled_spec_decode_tokens`；`token_budget`、`num_computed_tokens`、`num_tokens_with_spec`、`num_lookahead_tokens` 如何共同决定本轮验证多少 draft tokens；KV block 分配、preemption、structured output、async scheduling、batch queue 和 `update_from_output()` 又如何修正接受 / 拒绝后的请求状态。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的文档风格，本篇按“先定 Scheduler 角色，再走 schedule 主链路，再拆输出和回写，最后总结边界”的方式梳理 Scheduler 侧 spec decode。

要回答的问题分成 12 组：

```text
1. Scheduler 在 spec decode 中负责什么？
2. Request 上哪些字段表达 draft token 状态？
3. Scheduler 初始化时如何设置 num_spec_tokens / num_lookahead_tokens？
4. schedule() 如何计算 num_new_tokens？
5. request.spec_token_ids 如何变成 scheduled_spec_decode_tokens？
6. token budget / max_model_len / KV cache allocation 如何限制 spec tokens？
7. SchedulerOutput 中哪些字段和 spec decode 有关？
8. _update_after_schedule() 为什么先推进 num_computed_tokens？
9. update_from_output() 如何根据 accepted / rejected tokens 修正状态？
10. update_draft_token_ids() 如何把下一轮 draft tokens 写回 Request？
11. structured output / batch queue 为什么需要 update_draft_token_ids_in_output()？
12. spec decode 下 preemption、encoder cache、routed experts 有哪些特殊处理？
```

阅读顺序建议：

```text
01_spec_decode_role.md
  → 02_draft_tokens_and_request_state.md
  → 03_scheduler_spec_decode_flow.md
  → 04_spec_decode_metadata.md
  → 09_output_recovery_and_scheduler_update.md
```

本篇只聚焦 Scheduler 侧，不展开 `GPUModelRunner` 如何构造 `SpecDecodeMetadata` 或 `RejectionSampler` 如何采样；那些内容放在后续专题。

---

## 1. 一句话回答

Scheduler 在 spec decode 中负责把 **请求上暂存的 draft tokens** 转成本轮 target model 要验证的调度计划，并在模型返回后修正接受 / 拒绝造成的请求状态变化。

主线是：

```text
Request.spec_token_ids
  → Scheduler.schedule()
  → SchedulerOutput.scheduled_spec_decode_tokens
  → Worker / ModelRunner 验证
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.output_token_ids / num_computed_tokens 修正
```

Scheduler 不生成 draft tokens。

它负责：

```text
1. 保存 drafter 回传的 Request.spec_token_ids；
2. 按 token budget / max_model_len / KV cache capacity 决定本轮能验证多少；
3. 为 spec tokens 预留 KV lookahead slots；
4. 构造 SchedulerOutput.scheduled_spec_decode_tokens；
5. schedule 后先乐观推进 num_computed_tokens；
6. update_from_output 时按 rejected token 数回滚 num_computed_tokens / placeholders；
7. structured output 场景下过滤非法 draft tokens；
8. 把下一轮 draft tokens 写回 Request。
```

一句话压缩：

```text
Scheduler 是 spec decode 的请求账本和调度边界：它不猜 token，但决定哪些 draft tokens 能被 target model 验证，以及验证后请求状态如何落账。
```

---

## 2. Scheduler 侧核心对象关系

```text
Request
  ├─ output_token_ids       已正式接受的输出 token
  ├─ spec_token_ids         待验证 draft tokens
  ├─ num_computed_tokens    已调度/已计算进度
  ├─ num_output_placeholders async scheduling 占位
  └─ is_prefill_chunk       是否仍处于 prefill chunk

Scheduler.schedule()
  ├─ num_scheduled_tokens
  ├─ scheduled_spec_decode_tokens
  ├─ req_to_new_blocks
  └─ SchedulerOutput

Scheduler.update_from_output()
  ├─ sampled_token_ids
  ├─ scheduled_spec_decode_tokens
  ├─ num_accepted / num_rejected
  └─ Request 状态修正
```

两条闭环：

```text
消费 draft：
Request.spec_token_ids
  → scheduled_spec_decode_tokens
  → target model / rejection sampler
  → accepted tokens
  → Request.output_token_ids

产生下一轮 draft：
ModelRunner.propose_draft_token_ids()
  → DraftTokenIds
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

---

## 3. Request 上的 spec decode 状态

`Request` 定义在：`code/vllm/vllm/v1/request.py:59`

和 spec decode 直接相关的字段：

```python
self.num_output_placeholders = 0
self.async_tokens_to_discard = 0
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
```

位置：`code/vllm/vllm/v1/request.py:150` 到 `code/vllm/vllm/v1/request.py:168`

### 3.1 `spec_token_ids`

```text
Request.spec_token_ids 是 Scheduler 侧暂存的 draft tokens。
```

这些 token 来自上一轮 Worker / ModelRunner 的 proposer，经由：

```text
ModelRunner.take_draft_token_ids()
  → Executor.take_draft_token_ids()
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
```

写回 request。

重要边界：

```text
spec_token_ids 不是最终输出；
它只是等待 target model 验证的候选 token。
```

### 3.2 `num_tokens` vs `num_tokens_with_spec`

`Request.num_tokens`：

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)
```

位置：`code/vllm/vllm/v1/request.py:261` 到 `code/vllm/vllm/v1/request.py:263`

`Request.num_tokens_with_spec`：

```python
@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

位置：`code/vllm/vllm/v1/request.py:265` 到 `code/vllm/vllm/v1/request.py:267`

区别：

```text
num_tokens：prompt tokens + 已正式接受的 output tokens；
num_tokens_with_spec：num_tokens + 待验证 draft tokens。
```

Scheduler 正是通过 `num_tokens_with_spec` 把 draft tokens 纳入同一套调度算法。

---

## 4. 初始化阶段：num_spec_tokens 和 num_lookahead_tokens

Scheduler 初始化时读取 speculative 配置：

```python
speculative_config = vllm_config.speculative_config
self.use_eagle = False
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.num_lookahead_tokens = 0
self.dynamic_sd_lookup: list[int] | None = None
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:232` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:236`

如果开启 dynamic speculative decoding：

```python
if speculative_config.num_speculative_tokens_per_batch_size:
    self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:237` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:243`

不同方法会设置不同 lookahead：

```python
if speculative_config.use_eagle():
    self.use_eagle = True
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.uses_draft_model():
    self.num_lookahead_tokens = self.num_spec_tokens
if speculative_config.use_dflash():
    self.num_lookahead_tokens = self.num_spec_tokens + 1
if speculative_config.use_dspark():
    self.num_lookahead_tokens = self.num_spec_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:244` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:258`

### 4.1 `num_spec_tokens`

`num_spec_tokens` 表示配置层允许的最大 draft token 数。

它用于：

```text
1. dynamic speculative decoding 的默认 K；
2. proposer 下一轮最多生成多少 draft tokens；
3. 统计和调度输出中的 num_spec_tokens_to_schedule。
```

### 4.2 `num_lookahead_tokens`

`num_lookahead_tokens` 表示 KV cache 分配时要额外预留的 token slots。

它用于：

```text
1. 为 draft tokens 的验证预留 KV 空间；
2. 避免本轮 target model 验证多个位置时没有物理 slot；
3. 支持 EAGLE / draft model / DFlash 等需要多 token 前瞻的方式。
```

DFlash 多一个 lookahead，而 DSpark 使用正好 `num_spec_tokens` 个 lookahead。DFlash 注释说明：

```text
DFlash 使用 in-fill-style decoding，
除了每个 draft token 的 query，
还需要 last sampled token 的 query。
```

---

## 5. schedule() 的统一 token 模型

`schedule()` 入口：`code/vllm/vllm/v1/core/sched/scheduler.py:433`

开头注释非常关键：

```text
There is no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has num_computed_tokens and num_tokens_with_spec.
num_tokens_with_spec =
  len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:435` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:444`

这说明 Scheduler 没有单独设计一个 “spec decode phase”。

它只做一件事：

```text
让 request.num_computed_tokens 追上 request.num_tokens_with_spec。
```

所以普通 prefill、普通 decode、chunked prefill、prefix cache、spec decode 都统一成：

```text
本轮还差多少 token 没计算？
```

---

## 6. schedule() 开始：初始化本轮 spec 输出容器

每轮 schedule 初始化：

```python
scheduled_new_reqs: list[Request] = []
scheduled_resumed_reqs: list[Request] = []
scheduled_running_reqs: list[Request] = []
preempted_reqs: list[Request] = []

req_to_new_blocks: dict[str, KVCacheBlocks] = {}
num_scheduled_tokens: dict[str, int] = {}
token_budget = self.max_num_scheduled_tokens
...
scheduled_spec_decode_tokens: dict[str, list[int]] = {}
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:446` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:462`

`scheduled_spec_decode_tokens` 是本轮 spec decode 的核心输出容器：

```text
key：request_id；
value：本轮实际要交给 target model 验证的 draft token ids。
```

只有当请求本轮真的被调度，并且它有可验证的 spec tokens 时，才会写入这个 dict。

---

## 7. RUNNING 请求调度：计算 num_new_tokens

Scheduler 先调度 `self.running` 中的请求。

每个 request 的本轮 token 数初始计算：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:510` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:514`

这条公式非常重要。

可以展开成：

```text
num_new_tokens
  = prompt + accepted output + pending draft + async placeholders
    - 已经调度/计算过的 tokens
```

在普通 decode 下：

```text
spec_token_ids 为空，num_new_tokens 通常是 1。
```

在 spec decode 下：

```text
spec_token_ids 非空，num_new_tokens 可能包含多个待验证 draft tokens。
```

### 7.1 token budget 限制

随后会受 token budget 限制：

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:515` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:517`

如果 batch token budget 不够，本轮只能验证一部分 draft tokens。

### 7.2 max_model_len 限制

spec decode 还要避免超过模型最大长度：

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:519` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:526`

注释明确说：

```text
This is necessary when using spec decoding.
```

原因是 spec decode 一次可能验证多个 draft tokens，还可能产生 bonus token；如果不提前限制，可能让输入位置越过 `max_model_len`。

### 7.3 prefill / encoder / mamba 限制

`num_new_tokens` 还可能被这些逻辑进一步缩小：

```text
long_prefill_token_threshold；
encoder input scheduling；
EAGLE 下 encoder window shift；
Mamba block aligned split；
KV / encoder cache capacity。
```

这意味着：

```text
即使 request.spec_token_ids 有 K 个，Scheduler 也不保证本轮全部验证。
```

---

## 8. KV allocation：为 spec decode 预留 lookahead slots

计算出 `num_new_tokens` 后，Scheduler 分配 KV blocks：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:570` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:576`

这里的重点是：

```text
num_new_tokens 是本轮实际调度要计算的 tokens；
num_lookahead_tokens 是 spec decode 额外预留空间。
```

KVCacheManager 如果无法分配，会触发 preemption：

```text
1. 找到可抢占请求；
2. _preempt_request(preempted_req)；
3. 回收其 blocks；
4. 如果它已经在本轮 scheduled_running_reqs 中，还要回滚 num_scheduled_tokens / scheduled_spec_decode_tokens。
```

当已调度请求被抢占时，会清理 spec decode 输出：

```python
scheduled_spec_decode_tokens.pop(preempted_req_id, None)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:590` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:596`

这避免了：

```text
请求已被抢占，但 SchedulerOutput 里还带着它的 draft tokens。
```

---

## 9. request.spec_token_ids 如何进入 scheduled_spec_decode_tokens

请求成功分配 KV blocks 后，会记录基础调度结果：

```python
scheduled_running_reqs.append(request)
request_id = request.request_id
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:621` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:628`

然后进入 spec decode 专门逻辑：

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

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:630` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:646`

### 9.1 公式含义

```text
num_scheduled_spec_tokens
  = num_new_tokens
    + request.num_computed_tokens
    - request.num_tokens
    - request.num_output_placeholders
```

可以理解为：

```text
本轮 scheduled range 中，越过真实 token 边界之后的那部分，就是 spec tokens。
```

其中：

```text
request.num_tokens：真实 token 边界；
request.num_computed_tokens：已经计算到哪里；
num_new_tokens：本轮要推进多少；
num_output_placeholders：async scheduling 中预留但还没正式落账的输出占位。
```

只有当本轮调度范围覆盖到 spec token 区间时，`num_scheduled_spec_tokens` 才大于 0。

### 9.2 为什么要截断

```python
if len(spec_token_ids) > num_scheduled_spec_tokens:
    spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
```

这表示：

```text
Request 上可能暂存了 K 个 draft tokens，
但本轮受 token budget / max_model_len / encoder / KV 限制，
只能验证前 N 个。
```

Scheduler 只把本轮真实调度到的前 N 个写入 `scheduled_spec_decode_tokens`。

### 9.3 为什么调度后清空 request.spec_token_ids

```python
request.spec_token_ids = []
```

原因：

```text
这些 draft tokens 已经被本轮 SchedulerOutput 消费；
新的 spec tokens 会在模型执行后由 update_draft_token_ids() 写入；
如果不清空，下一轮可能重复验证旧 draft tokens。
```

---

## 10. WAITING / 新请求与 spec decode

spec decode tokens 通常来自已经执行过一轮 decode 的 running request。

新请求首次进入时：

```text
1. 还没有 target model 输出；
2. 还没有 drafter 基于该请求生成下一轮 draft tokens；
3. Request.spec_token_ids 通常为空；
4. 本轮更多是 prefill / chunked prefill。
```

因此 Scheduler 的 spec decode 逻辑主要出现在 RUNNING 请求调度段。

如果某个请求仍是 prefill chunk，`update_draft_token_ids()` 也会忽略它的 draft tokens：

```python
if request.is_prefill_chunk:
    if request.spec_token_ids:
        request.spec_token_ids = []
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2015` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2019`

这说明：

```text
draft tokens 只应该挂到已经进入 decode 验证语义的请求上，
而不是还在 chunked prefill 中的请求。
```

---

## 11. Dynamic speculative decoding：本轮给下一轮的 K

Scheduler 构造 `SchedulerOutput` 前，会计算动态 K：

```python
num_spec_tokens_to_schedule = self.num_spec_tokens
if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
    num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
        len(num_scheduled_tokens)
    ]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1135` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1140`

这个字段写入：

```python
SchedulerOutput(...,
    num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1142` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1160`

语义是：

```text
Scheduler 根据当前 batch size 决定下一轮 proposer 最多 draft 多少 token。
```

注意它不是本轮已经调度的 spec token 数。

区别：

| 字段 | 含义 |
|---|---|
| `scheduled_spec_decode_tokens` | 本轮要验证的 draft tokens |
| `num_spec_tokens_to_schedule` | 本轮执行后 proposer 为下一轮应该生成的 K |

---

## 12. SchedulerOutput 中的 spec decode 字段

`SchedulerOutput` 定义在：`code/vllm/vllm/v1/core/sched/output.py:183`

核心字段：

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
num_invalid_spec_tokens: dict[str, int] | None = None
num_spec_tokens_to_schedule: int = 0
```

位置：

```text
scheduled_spec_decode_tokens：output.py:199 到 output.py:202
num_invalid_spec_tokens：output.py:231 到 output.py:232
num_spec_tokens_to_schedule：output.py:248 到 output.py:250
```

字段含义：

| 字段 | Scheduler 侧来源 | Worker / 后续用途 |
|---|---|---|
| `scheduled_spec_decode_tokens` | 从 `Request.spec_token_ids` 截断而来 | Worker 写入 InputBatch，target model 验证 |
| `num_invalid_spec_tokens` | structured output / batch queue 修正时填入 | spec decode acceptance rate 统计修正 |
| `num_spec_tokens_to_schedule` | dynamic SD lookup 或默认 K | ModelRunner proposer 生成下一轮 draft tokens |

还有一些字段会间接受 spec decode 影响：

| 字段 | 影响 |
|---|---|
| `num_scheduled_tokens` | 可能包含要验证的 draft tokens |
| `total_num_scheduled_tokens` | spec tokens 会占用 batch token budget |
| `scheduled_cached_reqs` | 携带 cached request 的 token / block diff |
| `new_block_ids_to_zero` | spec decode 分配的新 KV blocks 也可能需要 zero |
| `has_structured_output_requests` | grammar bitmask 要知道 spec tokens |

---

## 13. _make_cached_request_data：发给 Worker 的 cached request diff

Scheduler 构造 cached request data：

```python
cached_reqs_data = self._make_cached_request_data(
    scheduled_running_reqs,
    scheduled_resumed_reqs,
    num_scheduled_tokens,
    scheduled_spec_decode_tokens,
    req_to_new_blocks,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1104` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1111`

`_make_cached_request_data()` 中一个和 spec decode 相关的细节是 PP 非 async 路径：

```python
num_tokens = num_scheduled_tokens[req_id] - len(
    spec_decode_tokens.get(req_id, ())
)
token_ids = req.all_token_ids[
    req.num_computed_tokens : req.num_computed_tokens + num_tokens
]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1324` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1342`

这说明：

```text
spec tokens 不从 req.all_token_ids 里取，
因为它们还不是正式 output；
它们通过 SchedulerOutput.scheduled_spec_decode_tokens 单独传递。
```

这也再次说明：

```text
accepted output tokens 和 pending draft tokens 在 Scheduler 账本中必须分开。
```

---

## 14. _update_after_schedule：为什么先推进 num_computed_tokens

SchedulerOutput 构造完成后，会调用：

```python
self._update_after_schedule(scheduler_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1175` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1178`

核心逻辑：

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders
    )
    scheduler_output.has_structured_output_requests |= (
        request.use_structured_output and not request.is_prefill_chunk
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1225` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1238`

注释解释了关键点：

```text
1. SchedulerOutput 必须保留本轮原始 scheduled token 数，供 Worker 决定 input ids；
2. schedule 后先推进 num_computed_tokens，允许 prefill 请求下一轮继续被调度；
3. 如果后续 spec tokens 被拒绝，再在 update_from_output() 中回滚。
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1215` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1224`

这就是 spec decode 下的“乐观记账”：

```text
schedule 阶段先假设本轮 scheduled tokens 都被计算；
update 阶段根据 rejection sampling 结果修正 rejected draft tokens。
```

---

## 15. get_grammar_bitmask：grammar 也要看 scheduled spec tokens

structured output 的 bitmask 构造入口：

```python
def get_grammar_bitmask(
    self, scheduler_output: SchedulerOutput
) -> GrammarOutput | None:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1527`

调用 grammar manager 时会传入：

```python
bitmask = self.structured_output_manager.grammar_bitmask(
    self.requests,
    structured_output_request_ids,
    scheduler_output.scheduled_spec_decode_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1544` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1549`

原因：

```text
grammar bitmask 不只约束普通下一个 token，
还需要知道本轮有哪些 draft tokens 参与验证，
否则 structured output 无法正确处理 spec decode 的多 token 候选。
```

---

## 16. update_from_output：根据 accepted / rejected 修正状态

模型执行完成后，EngineCore 调用：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`code/vllm/vllm/v1/engine/core.py:513` 到 `code/vllm/vllm/v1/engine/core.py:515`

Scheduler 入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1551`

对每个本轮调度的请求：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = (
    sampled_token_ids[req_index] if sampled_token_ids else []
)

scheduled_spec_token_ids = (
    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1632` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1639`

如果本轮有 spec tokens：

```python
if (
    scheduled_spec_token_ids
    and (generated_token_ids or self.num_sampled_tokens_per_step == 0)
    and request.async_tokens_to_discard == 0
):
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_sampled = self.num_sampled_tokens_per_step
    num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    num_rejected = num_draft_tokens - num_accepted
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1640` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1650`

### 16.1 accepted / rejected 怎么算

```text
num_draft_tokens：本轮 target model 验证了多少 draft tokens；
num_sampled：正常每步采样 token 数，通常是 1；
generated_token_ids：RejectionSampler 输出的最终 token 列表；
num_accepted = max(len(generated_token_ids) - num_sampled, 0)；
num_rejected = num_draft_tokens - num_accepted。
```

为什么要减 `num_sampled`？

```text
spec decode 输出里可能包含 bonus / replacement token，
它不属于“被接受的 draft token”数量。
```

所以：

```text
generated_token_ids 的长度 = accepted draft tokens + 普通 sampled / bonus token。
```

### 16.2 回滚 num_computed_tokens

如果有 rejected tokens：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1651` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1657`

原因：

```text
_update_after_schedule() 先乐观把所有 scheduled tokens 加进 num_computed_tokens；
但 rejected draft tokens 不能算作真实进度；
所以 update_from_output() 要减回来。
```

### 16.3 回滚 async placeholders

async scheduling 下：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1658` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1661`

因为：

```text
num_output_placeholders 也包含 scheduled spec tokens 的占位；
rejected tokens 不能继续占住 output placeholder。
```

### 16.4 统计 acceptance rate

Scheduler 会更新 spec decoding stats：

```python
spec_decoding_stats = self.make_spec_decoding_stats(
    spec_decoding_stats,
    num_draft_tokens=num_draft_tokens,
    num_accepted_tokens=num_accepted,
    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
    request_id=req_id,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1662` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1668`

这里的 `num_invalid_spec_tokens` 用于 structured output 等场景下修正统计。

---

## 17. update_from_output：正式输出 token 如何落账

在 spec decode 修正之后，Scheduler 继续处理真实输出：

```python
new_token_ids = generated_token_ids
...
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1674` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1687`

`_update_request_with_output()` 最终会把 token append 到 request 的正式输出状态。

也就是说：

```text
scheduled_spec_decode_tokens：候选；
generated_token_ids：target model + rejection sampler 后的真实输出；
_update_request_with_output()：把真实输出写入 Request.output_token_ids。
```

如果 structured output 开启，还会推进 grammar：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    ...
    if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
        request.status = RequestStatus.FINISHED_ERROR
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1693` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1717`

这表示：

```text
最终被接受的 tokens 还要推进 grammar 状态；
如果 grammar 认为异常，会终止请求。
```

---

## 18. routed experts：spec decode 输出位置不同

如果开启 routed experts 返回，Scheduler 要从 routing data 里取本轮输出 token 对应的 experts。

spec decode 下有特殊处理：

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

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1749` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1757`

差异：

```text
普通 decode：输出 token 通常对应 scheduled range 的末尾；
spec decode：accepted tokens 位于 scheduled range 的开头，rejected tokens 在末尾。
```

这说明 spec decode 不仅影响 token 数，还影响“输出 token 对应 scheduled range 中哪几行”。

---

## 19. update_draft_token_ids：把下一轮 draft tokens 写回 Request

同步 scheduling 下，EngineCore 在 `post_step()` 调用：

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`code/vllm/vllm/v1/engine/core.py:519` 到 `code/vllm/vllm/v1/engine/core.py:526`

Scheduler 入口：

```python
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2005`

核心逻辑：

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

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2005` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2025`

这一步的语义：

```text
1. finished request：忽略 draft tokens；
2. prefill chunk：清空/忽略 draft tokens；
3. structured output：先 grammar.validate_tokens()；
4. 合法 tokens 写入 Request.spec_token_ids，等待下一轮 schedule() 消费。
```

---

## 20. update_draft_token_ids_in_output：batch queue / structured output 的特殊路径

在 `EngineCore.step_with_batch_queue()` 中，如果存在 deferred scheduler output，需要先处理 draft tokens 再计算 grammar bitmask：

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

位置：`code/vllm/vllm/v1/engine/core.py:623` 到 `code/vllm/vllm/v1/engine/core.py:638`

为什么要这样？

```text
batch queue 场景下，某个 SchedulerOutput 可能已经被调度并进入 deferred 状态；
structured output 的 grammar bitmask 需要知道本轮 scheduled spec tokens；
但 draft tokens 可能刚从 worker 产生，需要先过滤/修正这个 deferred SchedulerOutput。
```

Scheduler 侧入口：

```python
def update_draft_token_ids_in_output(
    self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
) -> None:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2027`

核心逻辑：

```python
sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
for req_id, spec_token_ids in zip(...):
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

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2030` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2063`

### 20.1 为什么要 pad `-1`

如果 grammar 过滤掉部分 draft tokens，长度会变短。

但 deferred `SchedulerOutput` 中本轮已经按原 spec token 数调度了位置。

所以需要：

```text
1. 保持原 scheduled spec token 数量；
2. 用 -1 占位非法 token；
3. grammar bitmask / worker 侧可以跳过这些 invalid token；
4. 用 num_invalid_spec_tokens 修正 acceptance rate 统计。
```

这是一种“保持 shape / 位置稳定，同时标记非法 draft tokens”的设计。

---

## 21. preemption：抢占时清理 spec 状态

如果 KV allocation 失败，Scheduler 会抢占请求。

`_preempt_request()` 中会清空 spec tokens：

```python
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
if request.spec_token_ids:
    request.spec_token_ids = []
request.num_preemptions += 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1191` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1207`

原因：

```text
preempt 后请求需要重新计算 / 恢复；
旧 draft tokens 基于旧执行上下文，不能继续信任；
否则可能在恢复后验证过期候选 token。
```

如果被抢占请求已经进入本轮 scheduled list，还会从本轮输出中移除：

```python
scheduled_spec_decode_tokens.pop(preempted_req_id, None)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:590` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:596`

---

## 22. encoder cache：为什么 free 时要考虑 placeholders

`_free_encoder_inputs()` 中有一段 spec decode 相关注释：

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

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1972` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2003`

这里的关键是：

```text
spec decode 可能因为 rejected draft tokens 回滚 num_computed_tokens；
所以释放 encoder cache 时不能只看 num_computed_tokens，
还要减掉 num_output_placeholders，确保回滚不会回到 encoder input 所在范围。
```

这说明 spec decode 的 rejection 不只是 output token 层面的事，也会影响 cache 生命周期安全。

---

## 23. Scheduler 和 ModelRunner 的边界

### Scheduler 负责

```text
Request.spec_token_ids 的保存和清空；
本轮能验证多少 spec tokens 的调度决策；
token budget / max_model_len / KV allocation 限制；
SchedulerOutput.scheduled_spec_decode_tokens；
num_spec_tokens_to_schedule；
structured output 下 draft token 过滤；
accepted / rejected 后的 num_computed_tokens 修正；
正式 output token 写回 Request。
```

### ModelRunner 负责

```text
根据 scheduled_spec_decode_tokens 更新 InputBatch；
构造 SpecDecodeMetadata；
执行 target model forward；
调用 RejectionSampler；
产生 sampled_token_ids；
生成下一轮 DraftTokenIds。
```

边界一句话：

```text
Scheduler 决定“验证哪些 draft tokens 并如何记账”，ModelRunner 决定“如何验证这些 draft tokens 并继续猜下一批”。
```

---

## 24. 一个完整例子：running 请求带 4 个 draft tokens

假设某请求当前状态：

```text
num_tokens = 100
num_computed_tokens = 100
spec_token_ids = [201, 202, 203, 204]
num_output_placeholders = 0
token_budget 足够
```

### 24.1 schedule 阶段

计算：

```text
num_tokens_with_spec = 104
num_new_tokens = 104 - 100 = 4
```

如果 KV allocation 成功：

```text
num_scheduled_tokens[req_id] = 4
scheduled_spec_decode_tokens[req_id] = [201, 202, 203, 204]
request.spec_token_ids = []
```

然后 `_update_after_schedule()`：

```text
request.num_computed_tokens = 104
```

此时是乐观记账，暂时假设这 4 个 token 都被计算。

### 24.2 ModelRunner 验证

Worker 收到：

```text
scheduled_spec_decode_tokens[req_id] = [201, 202, 203, 204]
```

target model + rejection sampler 假设接受前 2 个 draft tokens，并采出 1 个 replacement / bonus token：

```text
generated_token_ids = [201, 202, 301]
```

### 24.3 update_from_output 阶段

Scheduler 计算：

```text
num_draft_tokens = 4
num_sampled = 1
num_accepted = len([201, 202, 301]) - 1 = 2
num_rejected = 4 - 2 = 2
```

回滚：

```text
request.num_computed_tokens = 104 - 2 = 102
```

正式输出：

```text
Request.output_token_ids append [201, 202, 301]
```

最终状态含义：

```text
前两个 draft tokens 被接受；
后两个 draft tokens 被拒绝；
301 是 target model 采出来的真实 token；
下一轮从新的上下文继续。
```

---

## 25. 容易混淆的点

### 25.1 `request.spec_token_ids` 和 `scheduled_spec_decode_tokens` 是同一个吗？

不是。

```text
request.spec_token_ids：跨 step 暂存在 Request 上，等待下一轮调度；
scheduled_spec_decode_tokens：本轮 SchedulerOutput 中实际要验证的 draft tokens。
```

调度后 `request.spec_token_ids` 会被清空。

### 25.2 `num_scheduled_tokens` 是否包含 spec tokens？

包含。

```text
num_scheduled_tokens 表示本轮模型要处理多少 token，
其中可能包括 prompt、普通 decode token、draft spec tokens、async placeholders。
```

具体哪些是 spec tokens，要看 `scheduled_spec_decode_tokens`。

### 25.3 为什么 schedule 后先加 num_computed_tokens？

因为 Scheduler 采用乐观记账：

```text
先认为本轮 scheduled tokens 已推进；
如果 rejection sampler 拒绝了某些 draft tokens，
update_from_output() 再把 rejected 数量减回来。
```

### 25.4 `num_spec_tokens_to_schedule` 是本轮验证数量吗？

不是。

```text
本轮验证数量：len(scheduled_spec_decode_tokens[req_id])；
下一轮 proposer 应生成数量：num_spec_tokens_to_schedule。
```

### 25.5 structured output 为什么会把 invalid tokens pad 成 -1？

因为 batch queue / deferred output 中，调度 shape 已经确定。

```text
过滤非法 draft tokens 会变短；
为了保持原 scheduled spec token 数，
用 -1 占位，并记录 num_invalid_spec_tokens。
```

### 25.6 prefill chunk 能使用 draft tokens 吗？

Scheduler 会忽略 prefill chunk 的 draft tokens。

```text
request.is_prefill_chunk 为 True 时，update_draft_token_ids() 不写入 spec_token_ids。
```

---

## 26. 从“回答问题”的角度总结

如果要问：

```text
Scheduler 如何调度 spec decode？
```

可以回答：

```text
Scheduler 把 spec decode 融入统一 token 调度模型中。

每个 Request 不只看正式 token 数 num_tokens，还看 num_tokens_with_spec，也就是正式 token 加上待验证的 spec_token_ids。schedule() 计算 num_new_tokens 时会把 spec tokens 纳入待计算范围，并受 token budget、max_model_len、encoder / Mamba 限制和 KV allocation 约束。请求成功调度后，Scheduler 会计算本轮 scheduled range 中有多少 token 落在 spec 区间，把对应前缀写入 SchedulerOutput.scheduled_spec_decode_tokens，然后清空 Request.spec_token_ids。

schedule 后 Scheduler 先乐观推进 num_computed_tokens；模型返回后，update_from_output() 根据 generated_token_ids 和 scheduled_spec_decode_tokens 计算 accepted / rejected 数量，把 rejected tokens 从 num_computed_tokens 和 async placeholders 中减掉，再把真正接受的 generated_token_ids 写入 Request.output_token_ids。下一轮 draft tokens 则通过 update_draft_token_ids() 写回 Request.spec_token_ids，形成循环。
```

职责关系可以概括为：

```text
Request.spec_token_ids：下一轮候选；
Scheduler.schedule()：决定本轮验证哪些候选；
SchedulerOutput.scheduled_spec_decode_tokens：发给 Worker 的候选；
ModelRunnerOutput.sampled_token_ids：验证后的真实输出；
Scheduler.update_from_output()：接受 / 拒绝后落账；
Scheduler.update_draft_token_ids()：写回下一轮候选。
```

---

## 27. 最关键流程图

```text
上一轮结束

ModelRunner.propose_draft_token_ids()
  → DraftTokenIds(req_ids, draft_token_ids)
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

```text
本轮 schedule

Scheduler.schedule()
  ├─ num_new_tokens =
  │    request.num_tokens_with_spec
  │    + request.num_output_placeholders
  │    - request.num_computed_tokens
  │
  ├─ apply token_budget / max_model_len / encoder / mamba limits
  │
  ├─ KVCacheManager.allocate_slots(
  │    num_new_tokens,
  │    num_lookahead_tokens=self.num_lookahead_tokens,
  │  )
  │
  ├─ if request.spec_token_ids:
  │    ├─ num_scheduled_spec_tokens =
  │    │    num_new_tokens
  │    │    + request.num_computed_tokens
  │    │    - request.num_tokens
  │    │    - request.num_output_placeholders
  │    ├─ scheduled_spec_decode_tokens[req_id] = prefix(spec_token_ids)
  │    └─ request.spec_token_ids = []
  │
  ├─ SchedulerOutput(... scheduled_spec_decode_tokens ...)
  │
  └─ _update_after_schedule()
       └─ request.num_computed_tokens += num_scheduled_tokens[req_id]
```

```text
本轮 update

Scheduler.update_from_output()
  ├─ generated_token_ids = model_runner_output.sampled_token_ids[req_index]
  ├─ scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens[req_id]
  │
  ├─ if scheduled_spec_token_ids:
  │    ├─ num_draft_tokens = len(scheduled_spec_token_ids)
  │    ├─ num_accepted = max(len(generated_token_ids) - num_sampled, 0)
  │    ├─ num_rejected = num_draft_tokens - num_accepted
  │    ├─ request.num_computed_tokens -= num_rejected
  │    └─ request.num_output_placeholders -= num_rejected
  │
  ├─ _update_request_with_output(request, generated_token_ids)
  └─ EngineCoreOutput(new_token_ids=...)
```

---

## 28. 最关键对象关系

```text
Request.spec_token_ids
  Scheduler 侧暂存的下一轮 draft token 候选。

Request.num_tokens_with_spec
  正式 tokens + pending spec tokens，是 schedule() 统一调度公式的核心。

Scheduler.num_lookahead_tokens
  KV allocation 时为 spec decode 预留的额外 token slots。

SchedulerOutput.scheduled_spec_decode_tokens
  本轮真正交给 Worker / target model 验证的 draft tokens。

SchedulerOutput.num_spec_tokens_to_schedule
  dynamic speculative decoding 给下一轮 proposer 的 draft token 数。

SchedulerOutput.num_invalid_spec_tokens
  structured output 过滤非法 draft tokens 后，用于统计修正。

ModelRunnerOutput.sampled_token_ids
  RejectionSampler 后真正被接受 / 采出的 token ids。

Request.num_computed_tokens
  schedule 后乐观推进，update_from_output 后按 rejected tokens 回滚。
```

---

## 29. 最小心智模型

如果只记一条主线，可以记：

```text
Scheduler 用 num_tokens_with_spec 把 draft tokens 当成“待计算 token”调度出去，
用 scheduled_spec_decode_tokens 告诉 Worker 哪些 token 是候选，
再用 update_from_output 根据 accepted / rejected 结果修正 num_computed_tokens 和正式输出。
```

再压缩成一句话：

```text
Scheduler 侧 spec decode 的核心是：候选入账、调度验证、乐观推进、拒绝回滚、下一轮候选回写。
```
