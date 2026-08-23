# Spec Decode 中各种 token 和状态到底是什么

源码相关文件与符号：

- `vllm/v1/request.py`：Request token 字段、spec token 字段与长度属性。
- `vllm/v1/core/sched/scheduler.py`：`num_sampled_tokens_per_step`，diffusion 场景可为 0。
- `vllm/v1/core/sched/scheduler.py`：running 请求的 `num_new_tokens` 公式。
- `vllm/v1/core/sched/scheduler.py`：running spec token 调度、裁剪与清空。
- `vllm/v1/core/sched/scheduler.py`：new decode 请求的 `[-1]` spec padding。
- `vllm/v1/core/sched/scheduler.py`：`_update_after_schedule()` 乐观推进 computed / in-flight 进度。
- `vllm/v1/core/sched/scheduler.py`：`update_from_output()` 计算 accepted / rejected 并修正状态。
- `vllm/v1/core/sched/scheduler.py`：draft token 更新与 structured output grammar validate。
- `vllm/v1/core/sched/async_scheduler.py`：async placeholder、spec placeholder 与 stale output frame 处理。

这份文档专门解释 speculative decoding / async scheduling 里容易混淆的几个概念：

```text
num_tokens
num_tokens_with_spec
spec_token_ids
scheduled_spec_decode_tokens
num_output_placeholders
num_computed_tokens
num_in_flight_tokens
num_scheduled_tokens
accepted draft tokens
rejected draft tokens
target-sampled token / replacement token / bonus token
num_stale_output_tokens / drop_stale_output
```

核心目标是回答：

```text
num_tokens_with_spec 里包含的 spec token，到底是还没调度、正在 forward、还是已经验证完？
```

---

## 1. 先给结论

一句话总结：

```text
num_tokens_with_spec 包含的是“当前挂在 request 上、准备被 Scheduler 调度给 target model 验证的 draft token”。
```

它本身不表示这些 token 已经验证通过，也不表示这些 token 已经成为真实输出。

更准确的时间线是：

```text
1. draft model 生成 draft token
   → 放到 request.spec_token_ids
   → num_tokens_with_spec 包含它们
   → 状态：待调度、待验证

2. Scheduler.schedule() 调度它们
   → 放进 scheduler_output.scheduled_spec_decode_tokens
   → request.spec_token_ids 被清空
   → 状态：已调度出去，准备由 target model forward 验证

3. Worker / target model 执行 forward
   → 一次 forward 验证整串 draft token
   → 状态：正在验证或已经验证，但 Scheduler 还没处理结果

4. Worker 返回 ModelRunnerOutput
   → generated_token_ids 表示最终接受/拒绝后的输出
   → 状态：Worker 已经验证，结果回到 Scheduler

5. Scheduler.update_from_output()
   → 先按 num_scheduled_tokens 回收 num_in_flight_tokens
   → 计算 accepted / rejected
   → append 真实输出 token
   → 回退 rejected draft 对应的 num_computed_tokens / num_output_placeholders
   → structured output 场景还会校验/推进 grammar token
   → 状态：验证结果正式落到 request
```

所以：

```text
num_tokens_with_spec 阶段的 spec token = 待调度 / 待验证 draft token。
SchedulerOutput 阶段的 spec token = 已调度 / in-flight draft token。
update_from_output 后的 token = 已接受的变成真实 output；被拒绝的被丢弃并回退进度。
```

---

## 2. `num_tokens`：真实 token 总数

源码：

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)
```


含义：

```text
num_tokens = 当前 request 里真实存在的 token 数
           = prompt token + 已确认 output token
```

它来自：

```python
self._all_token_ids
```

所以它只包含已经确认属于请求的 token。

不包含：

```text
还没验证的 draft token
还没返回的 async output placeholder
```

例子：

```text
prompt = 100 tokens
已经确认输出 = 3 tokens
num_tokens = 103
```

---

## 3. `spec_token_ids`：draft model 猜出来、准备验证的 token

源码初始化：

```python
self.spec_token_ids: list[int] = []
```


含义：

```text
spec_token_ids = draft model 预测出来的候选 token
```

这些 token 还没有被 target model 接受。

例如当前真实序列是：

```text
A
```

draft model 猜了：

```text
d1 d2 d3 d4
```

那么：

```python
request.spec_token_ids = [d1, d2, d3, d4]
```

此时它们只是候选，不是真实输出。

---

## 4. `num_tokens_with_spec`：真实 token + 当前挂着的 draft token

源码：

```python
@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```


含义：

```text
num_tokens_with_spec = 真实 token 数 + 当前待验证 draft token 数
```

例如：

```text
真实 token 数 num_tokens = 100
spec_token_ids = [d1, d2, d3, d4]
```

那么：

```text
num_tokens_with_spec = 104
```

这个字段回答的是：

```text
如果这轮要把 draft token 也送去验证，target model 需要计算到哪个位置？
```

它不是说：

```text
d1~d4 已经验证通过
```

而是说：

```text
d1~d4 当前挂在 request 上，下一轮调度时要考虑进去。
```

---

## 5. `scheduled_spec_decode_tokens`：本轮真正调度出去的 draft token

在 `schedule()` 里，如果 running 请求带有 `spec_token_ids`，Scheduler 会先算本轮真正能调度多少个 spec token，再裁剪后放入本轮输出：

```python
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
```


随后：

```python
request.spec_token_ids = []
```


这一步非常关键。

它表示：

```text
spec_token_ids 从 request 上被取走，进入 SchedulerOutput。
```

也就是说：

```text
调度前：
  draft token 在 request.spec_token_ids 里
  num_tokens_with_spec 包含它们

调度后：
  draft token 在 scheduler_output.scheduled_spec_decode_tokens 里
  request.spec_token_ids 被清空
```

所以不要把 `num_tokens_with_spec` 理解成“所有已经调度出去的 spec token”。

它更准确表示：

```text
当前还挂在 request 上、准备参与本轮调度的 spec token。
```

另外，waiting 请求刚进入 decode 且需要固定 spec decode 图形时，Scheduler 也可能直接给本轮 `scheduled_spec_decode_tokens` 填入：

```python
scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
```


这些 `-1` 是 spec padding / placeholder，不是真实 draft token id。

Worker / drafter 后续更新 draft token 时还有两条路径：

```text
update_draft_token_ids()
  → 把下一批 draft token 写回 request.spec_token_ids；prefill chunk 会忽略并清空。

update_draft_token_ids_in_output()
  → 把真实 draft token 回填到已经发出的 scheduler_output.scheduled_spec_decode_tokens；
     会裁剪到原本 scheduled 的数量，structured output 场景会先 validate_tokens()，
     不足的部分再用 -1 pad 回原长度，并记录 num_invalid_spec_tokens。
```


---

## 6. `num_scheduled_tokens`：本轮每个请求实际安排计算多少 token

在 `schedule()` 中，调度成功后会记录：

```python
num_scheduled_tokens[request_id] = num_new_tokens
```

含义：

```text
num_scheduled_tokens[req_id] = 这个请求本轮被安排计算的 token 数
```

例如：

```text
普通 decode：num_scheduled_tokens = 1
chunked prefill：num_scheduled_tokens = 2048
spec decode：num_scheduled_tokens 可能 = 1 + draft token 数
```

调度结束后，`_update_after_schedule()` 会用它推进进度：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```


注意：这里的 `num_scheduled_token` 是遍历字典时单个请求的值，不是全局总数。

`num_new_tokens` 还会被本轮两个总预算共同裁剪：

```text
token_budget：SchedulerOutput 中实际计算 token 的剩余预算；
input_budget - draft_slots：模型输入总预算扣除 speculative drafter 预留位置。
```

因此 draft token 会影响目标长度和 KV lookahead，同时还会占用
`draft_slots` 输入位置；不能只用 `token_budget` 判断请求是否能调度。

---

## 7. `num_computed_tokens`：Scheduler 认为已经安排/计算到的位置

源码初始化：

```python
self.num_in_flight_tokens = 0
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
```


它表示：

```text
Scheduler 认为这个请求已经计算到哪里了。
```

但在 async / pipeline 场景中，它不一定表示物理上 Worker 已经完成并返回。

因为 `_update_after_schedule()` 会在 Worker 返回前就提前做：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

所以：

```text
num_computed_tokens 更像是 Scheduler 侧的“已安排计算进度”。
num_in_flight_tokens 则记录这条乐观进度里还有多少 token 的 Worker 输出没被 Scheduler 回收。
```

它的作用是避免下一轮重复调度同一段 token。

如果后续发现 spec token 被拒绝，再回退；如果只是 Worker 输出回来，`update_from_output()` 会按 `num_scheduled_tokens` 先扣回 `num_in_flight_tokens`。

---

## 8. `num_output_placeholders`：已经发出去、但输出还没回来的占位

源码初始化：

```python
self.num_output_placeholders = 0
```


在 async scheduler 中，本轮调度后会增加：

```python
request.num_output_placeholders += (
    self.num_sampled_tokens_per_step + cur_num_spec_tokens
)
```


含义：

```text
num_output_placeholders = 已经异步调度出去、但 sampled token 还没返回并写入 request 的输出位置数
```

普通 decode：

```text
num_sampled_tokens_per_step = 1
cur_num_spec_tokens = 0
num_output_placeholders += 1
```

spec decode，假设 4 个 draft：

```text
num_output_placeholders += num_sampled_tokens_per_step + 4
```

普通 AR 模型中 `num_sampled_tokens_per_step = 1`，所以通常加 5；diffusion 模型中 `num_sampled_tokens_per_step = 0`，只按 spec / canvas token 数增加。

普通 AR spec decode 的 5 个占位表示：

```text
最多可能产生 4 个 accepted draft token + 1 个 target-sampled token
```

但它们还没真正写入 `_all_token_ids`。

如果 reset prefix cache 或 connector 场景在 async 输出返回前发生 preempt，Scheduler
会记录仍在途的输出 token，并清零 placeholder：

```python
request.num_stale_output_tokens = request.num_in_flight_tokens
request.num_output_placeholders = 0
```

`drop_stale_output` 决定 stale frame 是直接丢弃还是仍然交付。两种模式都会在
`update_from_output()` 中按本轮返回量递减 `num_stale_output_tokens`；交付模式
不会用 stale frame 执行 spec reject 回退，避免旧结果破坏当前账本。


---

## 9. 为什么 `num_tokens_with_spec` 和 `num_output_placeholders` 不算重复？

它们看起来都可能包含 spec token，但表示的是不同时间阶段。

### 9.1 `num_tokens_with_spec`

表示：

```text
当前 request 上挂着的 draft token，准备被调度。
```

关键词：

```text
待调度、待验证、还在 request.spec_token_ids 中
```

### 9.2 `num_output_placeholders`

表示：

```text
已经调度出去、但结果还没回来的输出占位。
```

关键词：

```text
已调度、in-flight、等待 Worker 返回
```

所以二者不是同一批状态：

```text
num_tokens_with_spec：当前准备发出去的 draft
num_output_placeholders：之前已经发出去、还没回来的占位
```

当然，在连续异步调度中，它们可能同时存在：

```text
上一批 spec decode 还没回来 → num_output_placeholders > 0
下一批 draft token 又挂到 request 上 → num_tokens_with_spec 包含新的 spec_token_ids
```

在 `AsyncScheduler` 中，下一批 `spec_token_ids` 也可能先是 `[-1]` placeholder：`_update_after_schedule()` 会根据 `scheduler_output.num_spec_tokens_to_schedule` 设置占位，真实 draft token id 后续由 Worker 侧更新。它仍然会计入 `num_tokens_with_spec`，用于继续预留下一轮验证范围。

这就是公式里二者同时出现的原因。

---

## 10. running 请求的 `num_new_tokens` 公式怎么理解

源码：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```


这个公式回答的是：

```text
这个 running 请求，本轮还需要再调度多少 token？
```

可以把它理解成：

```text
num_new_tokens = 目标应该计算到的位置 - Scheduler 已经认为计算到的位置
```

其中：

```text
目标应该计算到的位置
= request.num_tokens_with_spec + request.num_output_placeholders

Scheduler 已经认为计算到的位置
= request.num_computed_tokens
```

所以：

```text
num_new_tokens
= (真实 token + 当前新 draft token + 已经在路上的输出占位)
  - 已经安排计算过的 token
```

---

### 10.1 先看最简单的同步普通 decode

假设：

```text
当前真实序列长度 = 100
没有 draft token：spec_token_ids = []
没有异步在路上的输出：num_output_placeholders = 0
Scheduler 已经计算到 99
```

那么：

```text
num_tokens_with_spec = 100
num_output_placeholders = 0
num_computed_tokens = 99
```

代入公式：

```text
num_new_tokens = 100 + 0 - 99 = 1
```

含义：

```text
当前真实序列有 100 个 token，但模型只计算到 99，
所以本轮需要再计算 1 个 token。
```

这就是普通 decode 一步。

---

### 10.2 prefill / chunked prefill 也一样

假设 prompt 一共有 1000 个 token，已经计算了 600 个：

```text
num_tokens_with_spec = 1000
num_output_placeholders = 0
num_computed_tokens = 600
```

代入：

```text
num_new_tokens = 1000 + 0 - 600 = 400
```

含义：

```text
还剩 400 个 prompt token 没 prefill。
```

如果本轮 token budget 不够，后面还会被 `token_budget` 或 `long_prefill_token_threshold` 截断。

---

### 10.3 加上 spec token：为什么用 `num_tokens_with_spec`

假设当前真实序列长度是 100，draft model 新猜了 4 个 token：

```text
spec_token_ids = [d1, d2, d3, d4]
num_tokens_with_spec = 100 + 4 = 104
num_output_placeholders = 0
num_computed_tokens = 99
```

代入：

```text
num_new_tokens = 104 + 0 - 99 = 5
```

含义：

```text
这 5 个位置对应最后一个已确认 token + 4 个 draft，用于得到 4 个 draft 的验证结果和 1 个 target-sampled / replacement 位置。
```

所以 `num_tokens_with_spec` 的作用是：

```text
把当前新挂上的 draft token 纳入“目标要计算到的位置”。
```

如果不用 `num_tokens_with_spec`，只用 `num_tokens`，Scheduler 就看不到这 4 个 draft token，自然不会调度 target model 去验证它们。

---

### 10.4 async 普通 decode：为什么要加 `num_output_placeholders`

这是最容易混淆的地方。

假设当前真实 `_all_token_ids` 长度是 100：

```text
num_tokens = 100
spec_token_ids = []
num_tokens_with_spec = 100
```

上一轮 async decode 已经发出去了，还没返回。Scheduler 在调度后已经提前推进了：

```text
num_computed_tokens = 100
num_output_placeholders = 1
```

这里的含义是：

```text
第 100 个位置的 forward 已经在路上，回来后会产生下一个输出 token；
但这个输出 token 还没写入 _all_token_ids，所以 num_tokens 仍然是 100。
```

如果不加 placeholder，只算：

```text
num_tokens_with_spec - num_computed_tokens = 100 - 100 = 0
```

Scheduler 会误以为：

```text
目标长度已经追上了，不需要再调度。
```

但 async pipeline 可能希望继续提前调度下一步。于是公式把在路上的输出占位加回来：

```text
num_new_tokens = 100 + 1 - 100 = 1
```

含义：

```text
虽然真实 token 还没增长，但已经有 1 个输出在路上；
为了继续异步流水线，本轮还可以再调度下一步 decode。
```

所以 `num_output_placeholders` 的作用是：

```text
补偿“已经调度出去但还没写入 _all_token_ids 的输出位置”。
```

---

### 10.5 async + spec decode：两个字段为什么可能同时出现

假设上一轮 speculative decode 已经发出去，还没回来：

```text
上一轮：4 个 draft + 1 个 target-sampled token
num_output_placeholders = 5
```

同时下一批 draft token 或 async spec placeholder 又已经挂到 request 上：

```text
新 spec_token_ids = [e1, e2, e3, e4]
# 或 AsyncScheduler 先设置的 [-1, -1, -1, -1]
num_tokens_with_spec = 当前真实 token 数 + 4
```

这时：

```text
num_output_placeholders
  表示上一批已经调度出去、还没返回的输出占位。

num_tokens_with_spec
  表示当前新挂上、准备参与下一次调度的 draft token / spec placeholder。
```

它们可能都和 spec decode 有关，但不是同一批东西：

```text
num_output_placeholders = 过去已经发出去但未返回
num_tokens_with_spec = 现在准备发出去的新 draft
```

所以不是重复计算。

---

### 10.6 公式的直观图

可以画成一条时间线：

```text
真实已落到 request 的 token:
|----------------------|                     = num_tokens

当前新挂着的 draft token:
                       |----|                = len(spec_token_ids)

已经发出去但还没返回的输出占位:
                            |-----|          = num_output_placeholders

Scheduler 已认为计算到的位置:
|---------------------------|                = num_computed_tokens

本轮还要调度的部分:
                            |???|            = num_new_tokens
```

公式就是在算：

```text
目标右边界 - 已计算右边界
```

目标右边界是：

```text
真实 token + 当前 draft token + in-flight placeholder
```

已计算右边界是：

```text
num_computed_tokens
```

---

### 10.7 最终记法

可以把三个字段记成一句话：

```text
num_tokens_with_spec：现在 request 上已经有多少“真实 + 新 draft” token 需要考虑；
num_output_placeholders：过去已经发出去、但还没落到 request 上的输出占位；
num_computed_tokens：Scheduler 已经认为计算进度走到哪里；
num_in_flight_tokens：这条 computed 进度里还有多少 token 没被 output 回收，不直接参与 num_new_tokens 公式。
```

所以：

```text
num_new_tokens
= 现在要考虑的目标长度
  + 过去在路上的输出占位
  - 已经安排过的计算进度
```

一句话总结：

```text
这个公式不是在数“真实 token 有多少”，而是在对齐三个时间阶段：
已经落到 request 的 token、当前准备调度的 draft token、已经发出去但还没回来的 output placeholder，最后减掉 Scheduler 已经安排过的计算进度；`num_in_flight_tokens` 只用于回收和安全释放判断。
```

---

## 11. 一次 speculative step 到底怎么验证

假设当前真实序列最后是：

```text
A
```

draft model 给出：

```text
d1 d2 d3 d4
```

target model 通常一次 forward 处理整串：

```text
A d1 d2 d3 d4
```

逻辑上从左到右验证：

```text
A  → 验证/采样 d1 位置
d1 → 验证/采样 d2 位置
d2 → 验证/采样 d3 位置
d3 → 验证/采样 d4 位置
d4 → 采样下一个位置
```

可能结果：

```text
d1 被拒绝：
  输出 x
  x 是 A 后面重新采样的 replacement / normal sampled token

d1 接受，d2 被拒绝：
  输出 d1 x
  x 是 d1 后面重新采样的 token

d1、d2 接受，d3 被拒绝：
  输出 d1 d2 x

全部接受：
  输出 d1 d2 d3 d4 x
  x 是最后额外采样出来的 bonus token
```

所以：

```text
普通 AR spec decode 中，无论接受多少 draft，通常至少会有 1 个 target model 自己采样出来的 token。
```

这个数量在源码里由 `num_sampled_tokens_per_step` 表示；普通 AR 是 1，diffusion denoising step 可以是 0。

如果某个 draft 被拒绝，这个 token 是 replacement token。

如果全部 draft 都被接受，这个 token 才通常叫 bonus token。

---

## 12. Worker 返回后 Scheduler 如何确认接受/拒绝

在 `update_from_output()` 中：

```python
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
```

如果本轮有 spec token，且这批输出不属于需要丢弃或仅交付的 stale share，Scheduler 会计算：

```python
if (
    scheduled_spec_token_ids
    and (generated_token_ids or self.num_sampled_tokens_per_step == 0)
    and not output_is_stale
):
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_sampled = self.num_sampled_tokens_per_step
    num_accepted = max(len(generated_token_ids) - num_sampled, 0)
    num_rejected = num_draft_tokens - num_accepted
```


理解：

```text
generated_token_ids = 最终要追加到 request 的真实输出 token
num_sampled = target model 自己采样的 token 数，普通 AR 是 1
num_accepted = 输出 token 中扣掉 target-sampled token 后，剩下的 accepted draft 数
num_rejected = draft 总数 - accepted draft 数
```

如果有 rejected：

```python
request.num_computed_tokens -= num_rejected
```


async 场景还会：

```python
request.num_output_placeholders -= num_rejected
```


含义：

```text
调度时先乐观认为 draft token 都参与计算；
输出回来后，如果 draft 被拒绝，就把 rejected token 的进度回退。
如果 `scheduler_output.num_invalid_spec_tokens` 记录了 grammar validate 过滤掉的 spec token，`make_spec_decoding_stats()` 还会从统计用的 draft token 数中扣掉这些 invalid token。
```

---

## 13. 五个阶段总结表

| 阶段 | token 在哪里 | 表示什么 | 是否真实输出 |
|---|---|---|---|
| Draft 生成后 | `request.spec_token_ids` | 准备被验证的 draft token；async 下也可能先是 `[-1]` placeholder | 否 |
| schedule 前 | `num_tokens_with_spec` 包含它们 | 目标长度临时包含 draft / placeholder | 否 |
| schedule 后 | `scheduler_output.scheduled_spec_decode_tokens` | 已调度给 Worker 验证；可能被裁剪或用 `-1` padding | 否 |
| Worker forward 中 | Worker / ModelRunner 内部 | target model 正在一次 forward 验证 | 否 |
| update_from_output 后 | `request._all_token_ids` | accepted draft + target-sampled / replacement token 成为真实输出；invalid / rejected draft 不会落入 request | 是 |

---

## 14. 最容易混淆的几个点

### 14.1 `num_tokens_with_spec` 里的 spec token 是不是已经验证通过？

不是。

它只是：

```text
当前挂在 request 上、准备验证的 draft token。
```

### 14.2 `num_output_placeholders` 是不是和 `spec_token_ids` 重复？

不是。

```text
spec_token_ids：准备发出去的 draft token
num_output_placeholders：已经发出去但还没回来的输出占位
```

### 14.3 `num_computed_tokens` 是不是一定表示 Worker 已经算完？

不是。

在 async / PP 场景下，它可能只是 Scheduler 已经把这些 token 调度出去了，因此提前推进了计算进度。当前还有 `num_in_flight_tokens` 专门记录其中尚未被 Worker 输出回收的 token 数。

### 14.4 `num_in_flight_tokens` 和 `num_output_placeholders` 是不是一回事？

不是。

```text
num_in_flight_tokens：已调度但 update_from_output() 还没回收的计算 token 数。
num_output_placeholders：async 下已发出去但还没写回 request 的输出位置数。
```

### 14.5 draft token 被拒绝后怎么办？

被拒绝的 draft token 不会成为真实输出。

Scheduler 会：

```text
回退 num_computed_tokens
回退 num_output_placeholders
只把 generated_token_ids 中真实接受/采样出来的 token append 到 request
```

### 14.6 bonus token 和 replacement token 是一回事吗？

不是严格一回事。

```text
如果所有 draft 都接受，target model 在最后额外采样的 token 通常叫 bonus token。
如果某个 draft 被拒绝，target model 在拒绝位置重新采样的 token 更准确叫 replacement / normal sampled token。
```

---

## 15. 最终一句话

```text
num_tokens_with_spec 关心“当前准备验证哪些 draft token / spec placeholder”；
num_output_placeholders 关心“哪些输出位置已经异步发出去但还没回来”；
num_computed_tokens 关心“Scheduler 认为计算进度已经推进到哪里”；
num_in_flight_tokens 关心“这条乐观进度里还有多少 token 的 Worker 输出没被回收”；
num_stale_output_tokens 关心“reset / force-preempt 后哪些 stale output token 仍在途”；
drop_stale_output 决定 stale output 是交付还是丢弃；
真正哪些 draft 被接受，要等 Worker 返回后由 update_from_output() 计算 accepted/rejected 才知道。
```
