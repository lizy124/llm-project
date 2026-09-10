# vLLM Scheduler 中 token 的各种状态

源码相关文件与符号：

- `vllm/v1/request.py`：Request token 字段初始化、`append_output_token_ids()` 与 token 长度属性。
- `vllm/v1/core/sched/scheduler.py`：running 请求的 `num_new_tokens` 计算公式。
- `vllm/v1/core/sched/scheduler.py`：spec token 从 request 转移到 `SchedulerOutput`。
- `vllm/v1/core/sched/scheduler.py`：`_update_after_schedule()` 乐观推进 computed / in-flight 计数。
- `vllm/v1/core/sched/scheduler.py`：`update_from_output()` 回收 in-flight、处理 sampled token 与 spec reject。
- `vllm/v1/core/sched/scheduler.py`：`_preempt_request()` 标记 stale output，
  `update_from_output()` 负责交付或丢弃这些返回帧。
- `vllm/v1/core/sched/async_scheduler.py`：async scheduling 的 output placeholder 与 spec placeholder 更新。

本文不只解释 speculative decoding，而是从头梳理 vLLM Scheduler 里 token 的完整状态。重点解释这些字段为什么同时存在、各自代表什么、什么时候增长、什么时候回退。

---

## 1. 先建立一个总心智模型

在 Scheduler 里，token 不是只有一种状态。

同一个请求里的 token，大致会经历这些阶段：

```text
用户输入 prompt token
  → 进入 request.prompt_token_ids / request._all_token_ids
  → 等待 prefill
  → 被 Scheduler 调度
  → Worker 计算 KV
  → Scheduler 认为 num_computed_tokens 前进
  → decode 产生 output token
  → output token 写入 _output_token_ids / _all_token_ids
  → 后续继续 decode
```

如果开启 speculative decoding，还会多出 draft token：

```text
draft model 生成 spec_token_ids
  → Scheduler 把 draft token 发给 target model 验证
  → target model 一次 forward 验证整串 draft
  → 接受的 draft 变成真实 output token
  → 拒绝的 draft 被丢弃，并回退 Scheduler 的计算进度
```

如果开启 async scheduling，还会多出 placeholder：

```text
Scheduler 已经把某些 token 调度出去了
  → Worker 结果还没回来
  → _all_token_ids 还没增长
  → 但 num_computed_tokens 已经提前前进
  → num_in_flight_tokens 记录这些已调度但尚未被 Scheduler 回收的 token 数
  → num_output_placeholders 记录这些“在路上”的输出位置
```

所以理解 token 状态，最重要的是区分三件事：

```text
1. token 是否已经真实写入 request？
2. token 是否已经被 Scheduler 调度出去？
3. token 是否已经被 Worker 计算完成并被 Scheduler 消化？
```

---

## 2. 最核心的四条“长度线”

理解所有字段前，先记住四条线。

### 2.1 真实 token 线：`_all_token_ids` / `num_tokens`

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)
```

含义：

```text
request 中已经真实存在的 token 数。
```

它包括：

```text
prompt token + 已确认 output token
```

不包括：

```text
还没验证的 draft token
还没返回的 async output placeholder
```

### 2.2 Scheduler 计算进度线：`num_computed_tokens`

```python
self.num_computed_tokens = 0
```

含义：

```text
Scheduler 认为这个请求已经计算到哪个 token 位置。
```

注意，它不总是等于 Worker 物理上已经完成的 token 数。在 async / PP 场景中，Scheduler 发出 `SchedulerOutput` 后，会先把进度推进：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

也就是说：

```text
num_computed_tokens 有时表示“已经安排计算”，不一定表示“已经返回结果”。
```

### 2.3 in-flight 计算线：`num_in_flight_tokens`

```python
self.num_in_flight_tokens = 0
```

含义：

```text
已经发给 Worker / GPU step，但这一轮输出还没有回到 Scheduler、还没有被 update_from_output() 消化的 token 数。
```

它和 `num_computed_tokens` 一起在 `_update_after_schedule()` 增长；在 `update_from_output()` 遍历本轮 `num_scheduled_tokens` 时扣回：

```python
request.num_in_flight_tokens -= num_tokens_scheduled
```

所以：

```text
num_computed_tokens：调度进度线，可能乐观提前。
num_in_flight_tokens：这条乐观进度里，还有多少 token 的执行结果没被 Scheduler 收回来。
```

### 2.4 在路上的输出线：`num_output_placeholders`

```python
self.num_output_placeholders = 0
```

含义：

```text
已经异步调度出去，但 sampled token 还没返回、还没写入 _all_token_ids 的输出位置数。
```

普通 async decode 一步通常加 1。

spec decode 有 4 个 draft 时，通常加：

```text
4 个 draft + 1 个 target-sampled token = 5
```

---

## 3. Request 里常见 token 字段总览

| 字段 | 含义 | 是否真实 token | 什么时候变 |
|---|---|---|---|
| `prompt_token_ids` | prompt token 列表 | 是 | 请求创建 / streaming input 追加时 |
| `_output_token_ids` | 已确认输出 token 列表 | 是 | `update_from_output()` append 后 |
| `_all_token_ids` | prompt + 已确认 output | 是 | prompt 创建、输出 append、streaming 更新 |
| `num_prompt_tokens` | prompt 长度 | 是 | 请求创建 / streaming input 追加时 |
| `num_output_tokens` | 已确认 output 长度 | 是 | 输出 token append 后 |
| `num_tokens` | `_all_token_ids` 长度 | 是 | 随 `_all_token_ids` 自动变化 |
| `spec_token_ids` | draft model 猜的候选 token | 否 | draft model 更新后，schedule 后清空 |
| `num_tokens_with_spec` | `num_tokens + len(spec_token_ids)` | 部分真实，部分候选 | `spec_token_ids` 改变时 |
| `num_computed_tokens` | Scheduler 认为已计算到的位置 | 不是 token 列表，是进度 | schedule 后前进，spec reject / invalid KV block 后回退 |
| `num_in_flight_tokens` | 已调度但尚未被 `update_from_output()` 回收的 token 数 | 否，是 in-flight 计数 | schedule 后增加，worker output 回来后扣减 |
| `num_output_placeholders` | async in-flight 输出占位 | 否，是输出位置数量 | async schedule 后增加，output 返回后减少，spec reject 时也会回退 |
| `num_stale_output_tokens` | preempt 后仍在途的 stale output token 数 | 否，是 stale token 计数 | `_preempt_request()` 设置，`update_from_output()` 按本轮返回量递减 |
| `drop_stale_output` | 是否丢弃 stale output，而不是交付给客户端 | 否，是行为开关 | reset 或 connector 场景的 preempt 时设置 |
| `num_scheduled_tokens[req_id]` | 某轮给该请求安排的 token 数 | 不是 token 列表，是本轮工作量 | 每次 `schedule()` 生成 |
| `scheduled_spec_decode_tokens[req_id]` | 本轮发给 Worker 验证的 draft tokens | 否，待验证 | `schedule()` 生成 |
| `generated_token_ids` | Worker 返回的真实输出 token | 是，待 append | `ModelRunnerOutput` 返回后 |

---

## 4. prompt token 的状态

请求创建时，prompt token 会进入：

```python
self.prompt_token_ids = prompt_token_ids
self.num_prompt_tokens = len(prompt_token_ids)
self._all_token_ids = self.prompt_token_ids.copy()
```

此时：

```text
num_prompt_tokens = prompt 长度
num_tokens = prompt 长度
num_output_tokens = 0
num_computed_tokens = 0
```

举例：

```text
prompt 长度 = 100
```

那么初始状态：

```text
num_prompt_tokens = 100
num_tokens = 100
num_computed_tokens = 0
```

Scheduler 会在 prefill 阶段把 prompt token 分配给 Worker 计算 KV Cache。

如果 prefix cache 命中，`num_computed_tokens` 可以直接从 0 跳到命中长度；如果外部 KV cache 命中，也会把外部命中部分算进 computed token。

---

## 5. prefill 阶段 token 状态

prefill 的任务是：

```text
把 prompt token 跑过模型，生成对应 KV Cache。
```

如果 prompt 长度是 1000，且没有任何 cache 命中：

```text
num_tokens = 1000
num_computed_tokens = 0
```

running / waiting 调度时会算：

```text
num_new_tokens = 1000 - 0 = 1000
```

如果启用 chunked prefill，本轮可能只调度一部分，比如 256：

```text
num_scheduled_tokens[req_id] = 256
```

调度后 `_update_after_schedule()` 会：

```text
num_computed_tokens += 256
num_in_flight_tokens += 256
```

于是：

```text
num_computed_tokens = 256
num_in_flight_tokens = 256
```

等 Worker 输出回到 `update_from_output()`，这 256 个 token 会从 `num_in_flight_tokens` 扣回；如果只是中间 prefill chunk，通常不会 append sampled output token。

下一轮继续：

```text
num_new_tokens = 1000 - 256 = 744
```

直到：

```text
num_computed_tokens == num_tokens
```

prefill 完成后，后续进入 decode。

---

## 6. output token 的状态

decode 阶段，每次 Worker 产生 sampled token 后，Scheduler 会在 `update_from_output()` 中处理：

```python
request.append_output_token_ids(output_token_id)
```

这会让：

```text
_output_token_ids 增长
_all_token_ids 增长
num_output_tokens 增长
num_tokens 增长
```

例如：

```text
prompt = 100
已经生成 3 个 output
```

那么：

```text
num_prompt_tokens = 100
num_output_tokens = 3
num_tokens = 103
```

注意：

```text
num_tokens 是真实已经确认的 token 数。
```

如果 async 场景中 token 还没回来，它不会提前进入 `num_tokens`。

---

## 7. `num_computed_tokens` 和 `num_tokens` 为什么经常差 1

普通自回归 decode 中，常见节奏是：

```text
已有序列长度 = N
模型计算当前位置
采样出第 N+1 个 token
```

所以一轮循环中常见变化是：

```text
第 k 轮开始：
  num_tokens = 101
  num_computed_tokens = 100

schedule 后：
  num_computed_tokens = 101

Worker 返回后 append token：
  num_tokens = 102
  num_computed_tokens = 101
```

所以看起来总是：

```text
num_tokens 比 num_computed_tokens 多 1
```

这不是 bug，而是自回归生成的节奏：

```text
先计算已有最后一个 token，再采样下一个 token。
```

---

## 8. `spec_token_ids`：draft token 的状态

投机解码中，draft model 会先猜一些 token：

```text
d1 d2 d3 d4
```

它们会放入：

```python
request.spec_token_ids
```

此时这些 token：

```text
不是正式输出
没有写入 _all_token_ids
还没有被 target model 验证
```

所以它们只是：

```text
候选 token / draft token
```

---

## 9. `num_tokens_with_spec`：为什么要包含 draft token

源码：

```python
@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

它表示：

```text
真实 token + 当前挂在 request 上、准备验证的 draft token
```

例如：

```text
num_tokens = 100
spec_token_ids = [d1, d2, d3, d4]
```

那么：

```text
num_tokens_with_spec = 104
```

这并不是说 d1~d4 已经是真实输出。

它只是告诉 Scheduler：

```text
这 4 个 draft token 也要纳入本轮 target model 验证范围。
```

如果不用 `num_tokens_with_spec`，Scheduler 只看到真实长度 100，就不会知道还要验证这 4 个 draft。

---

## 10. schedule 后 draft token 去哪里

在 `schedule()` 里，如果 running 请求带有 `spec_token_ids`，会把它们放进本轮 `SchedulerOutput`：

```python
scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
```

然后清空：

```python
request.spec_token_ids = []
```

另外，waiting 请求刚进入 decode 且需要保持固定 spec decode 图形时，Scheduler 也可能直接 padding：

```python
scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
```

这类 `-1` 是 spec placeholder，不是真实 draft token id。

这一步表示：

```text
这些 draft token 已经从 request 上转移到 SchedulerOutput，准备发给 Worker 验证。
```

所以 draft token 的位置变化是：

```text
调度前：request.spec_token_ids
调度中：scheduler_output.scheduled_spec_decode_tokens
Worker 中：target model forward 验证
返回后：accepted 的进入 _all_token_ids，rejected 的丢弃
```

---

## 11. 一次 speculative step 怎么验证 draft token

假设当前真实序列最后是：

```text
A
```

draft model 猜：

```text
d1 d2 d3 d4
```

target model 通常一次 forward 处理：

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
无论接受多少 draft，通常至少会有 1 个 target model 自己采样出来的 token。
```

如果某个 draft 被拒绝，这个 token是 replacement token。

如果全部 draft 都被接受，这个 token 才通常叫 bonus token。

---

## 12. `num_output_placeholders`：async 下为什么需要占位

async scheduling 中，Scheduler 可能在 Worker 结果还没回来时，就继续下一轮调度。

因此会出现：

```text
Scheduler 已经调度了某个输出 token
Worker 还没返回 sampled token
_all_token_ids 还没增长
但 num_computed_tokens 已经提前前进
```

这时需要一个字段记录：

```text
有多少输出位置已经在路上。
```

这就是：

```python
num_output_placeholders
```

普通 async decode：

```text
num_output_placeholders += num_sampled_tokens_per_step
```

普通自回归模型里 `num_sampled_tokens_per_step = 1`，所以通常加 1。

spec decode，4 个 draft：

```text
num_output_placeholders += num_sampled_tokens_per_step + 4
```

普通 AR spec decode 通常是 `1 + 4 = 5`；diffusion 模型没有 AR bonus token，`num_sampled_tokens_per_step = 0`，只按 spec / canvas token 数增加。

这 5 个不是已经确认的真实 token，只是：

```text
已经调度出去、可能返回的输出位置。
```

---

## 13. `num_tokens_with_spec` 和 `num_output_placeholders` 是否重复

不重复。

它们描述的是不同时间阶段：

```text
num_tokens_with_spec：
  当前 request 上挂着的 draft token，准备调度。

num_output_placeholders：
  之前已经调度出去、但结果还没回来的输出占位。
```

一句话：

```text
num_tokens_with_spec 是“现在准备发出去的”；
num_output_placeholders 是“过去已经发出去但还没回来”的。
```

连续 async 调度中，它们可能同时存在。

例如：

```text
上一批 spec decode 还没回来 → num_output_placeholders > 0
下一批 draft token 又挂到 request 上 → num_tokens_with_spec 包含新的 spec_token_ids
```

在 `AsyncScheduler` 中，这里的下一批 `spec_token_ids` 可能先是 `[-1]` placeholder：`_update_after_schedule()` 会根据 `scheduler_output.num_spec_tokens_to_schedule` 设置占位，真实 draft token id 后续由 Worker 侧更新。它仍然会计入 `num_tokens_with_spec`，用于让 Scheduler 继续预留下一轮验证范围。

这不是重复，而是两批不同阶段的 token。

---

## 14. `num_scheduled_tokens`：本轮给请求安排多少工作

在 `schedule()` 中，每个被调度的请求都会记录：

```python
num_scheduled_tokens[request_id] = num_new_tokens
```

含义：

```text
这个请求本轮被安排计算多少 token。
```

例子：

```text
普通 decode：1
chunked prefill：2048
spec decode：1 + draft token 数
```

这个值会进入 `SchedulerOutput`，告诉 Worker 本轮要做多少工作。

实际值还会在 `schedule()` 中受 `token_budget` 和
`input_budget - draft_slots` 限制；`draft_slots` 是 speculative decoding
占用的输入预算，不应和 `num_scheduled_tokens` 混为同一个计数。

调度后，Scheduler 会用它推进：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

---

## 15. `num_in_flight_tokens` 和 `num_output_placeholders` 是否重复

不重复。

```text
num_in_flight_tokens：
  本轮或之前已经发给 Worker、但输出还没被 Scheduler 回收的调度 token 数。

num_output_placeholders：
  async decode/spec decode 下，预计会返回但还没有写入 request 的输出位置数。
```

prefill chunk 可能让 `num_in_flight_tokens` 增长很多，但没有 sampled output，所以 `num_output_placeholders` 可以仍然是 0。

async decode 则通常两者都会变化：`num_computed_tokens` / `num_in_flight_tokens` 记录调度出去的 forward 位置，`num_output_placeholders` 记录将来要被真实 sampled token 消耗的输出位置。

---

## 16. `num_new_tokens` 公式完整解释

running 请求中，Scheduler 用这个公式计算本轮还要调度多少 token：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

这里的“调度 token”不是指“生成多少 output token”，而是指：

```text
本轮要安排给 Worker / 主模型 forward 处理多少个 token 位置。
```

这个公式回答的是：

```text
这个 running 请求，本轮还需要让主模型继续计算多少 token 位置？
```

可以理解成：

```text
num_new_tokens = 目标应该计算到的位置 - Scheduler 已经认为计算到的位置
```

目标应该计算到的位置是：

```text
真实 token + 当前新 draft token + 已经在路上的输出占位
```

即：

```text
request.num_tokens_with_spec + request.num_output_placeholders
```

这里有两个容易混淆的点：

```text
当前新 draft token：
  draft model 已经生成了 token id，
  但主模型还没有 forward 验证它们，
  所以仍然要计入本轮主模型需要计算的位置。

num_output_placeholders：
  表示之前已经调度出去、但还没写回 request 的输出位置；
  async scheduling 为了继续流水线，需要把这些在路上的位置也算进目标长度。
```

已计算到的位置是：

```text
request.num_computed_tokens
```

所以：

```text
num_new_tokens
= (真实 token + 当前新 draft token + 已经在路上的输出占位)
  - Scheduler 已经认为安排计算过的位置
```

---

## 17. 公式例子 1：同步普通 decode

假设：

```text
当前真实序列长度 = 100
没有 draft token
没有 async placeholder
Scheduler 已经计算到 99
```

那么：

```text
num_tokens_with_spec = 100
num_output_placeholders = 0
num_computed_tokens = 99
```

代入：

```text
num_new_tokens = 100 + 0 - 99 = 1
```

含义：

```text
当前真实序列有 100 个 token，但模型只计算到 99，所以本轮需要再计算 1 个 token。
```

---

## 18. 公式例子 2：chunked prefill

prompt 一共有 1000 个 token，已经计算 600 个：

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

## 19. 公式例子 3：新 draft token 需要验证

当前真实序列长度是 100，draft model 新猜了 4 个 token：

```text
spec_token_ids = [d1, d2, d3, d4]
num_tokens_with_spec = 104
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

---

## 20. 公式例子 4：async 普通 decode 为什么要加 placeholder

假设当前真实 `_all_token_ids` 长度是 100：

```text
num_tokens = 100
spec_token_ids = []
num_tokens_with_spec = 100
```

上一轮 async decode 已经发出去了，还没返回。Scheduler 已经提前推进：

```text
num_computed_tokens = 100
num_output_placeholders = 1
```

含义：

```text
第 100 个位置的 forward 已经在路上；
它回来后会产生下一个 output token；
但这个 output token 还没写入 _all_token_ids，所以 num_tokens 仍然是 100。
```

如果不加 placeholder：

```text
num_tokens_with_spec - num_computed_tokens = 100 - 100 = 0
```

Scheduler 会误以为：

```text
目标长度已经追上了，不需要再调度。
```

但 async pipeline 可能希望继续提前调度下一步。加上 placeholder 后：

```text
num_new_tokens = 100 + 1 - 100 = 1
```

含义：

```text
虽然真实 token 还没增长，但已经有 1 个输出在路上；
为了继续异步流水线，本轮还可以再调度下一步 decode。
```

---

## 21. 公式例子 5：async + spec decode

上一轮 speculative decode 已经发出去，还没回来：

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

## 22. Worker 返回后如何确认 accepted / rejected

在 `update_from_output()` 中，Scheduler 拿到：

```text
generated_token_ids
```

这是真正要追加到 request 的输出 token。

如果本轮有 spec token：

```python
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
```

会计算：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

含义：

```text
num_draft_tokens：本轮发出去验证的 draft 数
num_sampled：target model 自己采样的 token 数，普通 AR 是 1
num_accepted：最终输出里有多少 token 是 accepted draft
num_rejected：有多少 draft 被拒绝
```

如果有 rejected：

```python
request.num_computed_tokens -= num_rejected
```

async 场景还会：

```python
request.num_output_placeholders -= num_rejected
```

因为：

```text
调度时先乐观认为 draft token 都参与计算；
输出回来后，如果 draft 被拒绝，就把 rejected token 的进度回退。
```

---

## 23. async 返回后 placeholder 怎么减少

在 async scheduler 的输出更新中：

```python
request.num_output_placeholders -= len(new_token_ids)
```

含义：

```text
Worker 返回了多少真实输出 token，就消耗多少 placeholder。
```

例如普通 async decode：

```text
num_output_placeholders = 1
返回 1 个 token
num_output_placeholders -= 1 → 0
```

spec decode，4 个 draft，全部接受：

```text
num_output_placeholders = 5
返回 5 个 token
num_output_placeholders -= 5 → 0
```

如果只接受 2 个 draft，最终返回：

```text
d1 d2 x  # 3 个真实输出 token
```

那么 spec reject 逻辑会先扣 rejected draft 数；随后 `AsyncScheduler._update_request_with_output()` 再按真实 append 的 `new_token_ids` 数量扣减。总扣减量仍是 rejected + returned tokens，最终让状态和真实输出对齐。

如果 reset prefix cache 或 connector 场景在 async 输出返回前强制 preempt，Scheduler
会把当时仍在途的计算输出记录为 stale：

```python
request.num_stale_output_tokens = request.num_in_flight_tokens
request.num_output_placeholders = 0
```

如果 `drop_stale_output` 为真，stale frame 会被直接丢弃；否则 stale token
仍可交付，但不会参与 spec reject 回退。`update_from_output()` 每次按
`num_tokens_scheduled` 消化 `num_stale_output_tokens`，避免同一批输出重复影响计数。

---

## 24. token 生命周期总图

```text
prompt token
  → prompt_token_ids
  → _all_token_ids
  → prefill scheduled
  → num_computed_tokens / num_in_flight_tokens 前进
  → Worker output 回来后 num_in_flight_tokens 扣回
  → KV cache 写入

普通 decode token
  → schedule 调度已有最后 token
  → num_computed_tokens / num_in_flight_tokens 前进
  → async 下 num_output_placeholders 增加
  → Worker forward
  → sampled token 返回
  → num_in_flight_tokens 扣回
  → append_output_token_ids
  → _output_token_ids / _all_token_ids 增长

spec draft token
  → draft model 生成，或 async scheduler 先放入 [-1] placeholder
  → request.spec_token_ids
  → num_tokens_with_spec 包含它
  → schedule 放入 scheduled_spec_decode_tokens
  → Worker target model 验证
  → accepted：进入 generated_token_ids，最终 append 到 _all_token_ids
  → rejected：丢弃，并回退 num_computed_tokens / num_output_placeholders

async placeholder
  → schedule 发出但 output 未返回
  → num_output_placeholders 增加
  → output 返回后减少
  → reset / force-preempt 时标记 num_stale_output_tokens
  → 真实 token append 到 _all_token_ids，或按 drop_stale_output 丢弃 stale frame
```

---

## 25. 最容易混淆的问题

### 25.1 `num_tokens_with_spec` 里的 spec token 是真实 token 吗？

不是。

它只是当前挂在 request 上、准备验证的 draft token。

### 25.2 `num_tokens_with_spec` 里的 spec token 是不是已经调度出去了？

在 schedule 前，还没调度。

schedule 后，它会从 `request.spec_token_ids` 转移到 `scheduler_output.scheduled_spec_decode_tokens`。

### 25.3 `num_output_placeholders` 是不是 spec token？

不是具体 token ID，而是数量。

它表示已经发出去但还没返回的输出位置，其中可能包含 spec decode 的 draft 占位。

### 25.4 `num_computed_tokens` 是不是 Worker 已经算完？

不一定。

在 async / PP 中，它可以表示 Scheduler 已经把这些 token 调度出去，因此先把进度推进。

### 25.5 `num_in_flight_tokens` 是不是 output placeholder？

不是。

`num_in_flight_tokens` 统计已经调度但还没被 Scheduler 回收的计算 token；`num_output_placeholders` 统计 async 输出位置。prefill 可以有 in-flight token 但没有 output placeholder。

### 25.6 为什么 rejected draft 要回退 `num_computed_tokens`？

因为调度时先假设 draft token 参与计算并推进了进度。如果后面发现某些 draft 被拒绝，它们不能算作真实连续进度，所以要回退。

### 25.7 bonus token 和 replacement token 是一回事吗？

不是严格一回事。

```text
所有 draft 都接受后，target model 在最后额外采样的 token，通常叫 bonus token。
某个 draft 被拒绝时，target model 在拒绝位置重新采样的 token，更准确叫 replacement / normal sampled token。
```

---

## 26. 最终总结

一句话记住：

```text
num_tokens 是已经真实落到 request 上的 token；
num_tokens_with_spec 是真实 token 加上当前准备验证的 draft token；
num_computed_tokens 是 Scheduler 认为计算进度已经推进到哪里；
num_in_flight_tokens 是这条乐观进度里尚未被 update_from_output() 回收的 token 数；
num_output_placeholders 是 async 下已经发出去但还没写回 request 的输出占位；
num_stale_output_tokens 是 reset / force-preempt 后仍在途的 stale output token 数；
drop_stale_output 决定这些 token 是交付还是丢弃；
num_scheduled_tokens 是本轮实际安排给 Worker 的工作量；
最终哪些 token 成为真实输出，要等 Worker 返回后由 update_from_output() append 和修正。
```

如果只记一个核心公式：

```text
num_new_tokens
= request.num_tokens_with_spec
  + request.num_output_placeholders
  - request.num_computed_tokens
```

它的意思是：

```text
本轮还需要调度的 token 数
= 当前真实 token + 当前待验证 draft token + 过去已发出但未返回的输出占位
  - Scheduler 已经认为安排过的计算进度
```

`num_in_flight_tokens` 不直接出现在这个公式里；它用于区分 `num_computed_tokens` 中哪些进度还没有被 Worker 输出回收，主要影响 output 回收、deferred free、connector finished 时的 processed-token 判断。
