# 02. 本轮最多能调度多少 token？

源码位置：`D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`

本问题关注：Scheduler 每一轮 `schedule()` 到底能安排多少 token，哪些配置会限制本轮调度量，token budget 是如何被 running / waiting 请求共同消耗的，以及为什么有些请求明明在队列里却不能被本轮调度。

这里的“每一轮”指的是一次 `schedule()` 调用，也就是一个 scheduler step。它不等于“一次 decode”：一轮 `schedule()` 会做一次全局批处理决策，可能同时包含多个请求；每个请求可能是在 decode、prefill、chunked prefill、spec decode，或者只是发起远端 KV async load。对单个普通 decode 请求来说，一轮通常是 decode 一个 token；但对整个 Scheduler 来说，一轮是生成一个 `SchedulerOutput`，决定本次所有请求合计要执行多少 token。

---

## 1. 一句话回答

Scheduler 每轮最多能调度的 token 数由 `self.max_num_scheduled_tokens` 决定，进入 `schedule()` 后变成本轮局部变量 `token_budget`。

```python
token_budget = self.max_num_scheduled_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:407`

之后：

```text
每成功调度一个请求，就从 token_budget 中扣掉该请求本轮的 num_new_tokens。
```

也就是说，本轮调度 token 的基本模型是：

```text
本轮总预算 token_budget
  → 先给 running 请求使用
  → 剩余预算再给 waiting 请求使用
  → 每个请求调度多少，扣多少
  → token_budget 用完，本轮停止接纳更多 token
```

---

## 2. `max_num_scheduled_tokens` 从哪里来

初始化时：

```python
self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens is not None
    else self.scheduler_config.max_num_batched_tokens
)
```

位置：`scheduler.py:108`

含义：

1. 如果显式配置了 `max_num_scheduled_tokens`，使用它；
2. 否则退回使用 `max_num_batched_tokens`。

所以 Scheduler 每轮 token 上限不是直接写死的，而是来自 scheduler config。

可以理解为：

```text
max_num_scheduled_tokens = 每一轮 SchedulerOutput 允许安排的最大 token 数
```

它限制的是“调度层面”的 token 数，不是模型最大上下文长度，也不是请求最大输出长度。

---

## 3. `token_budget` 是每轮临时预算

`schedule()` 开始时：

```python
token_budget = self.max_num_scheduled_tokens
```

位置：`scheduler.py:407`

`token_budget` 是本轮局部变量。

每轮调用 `schedule()` 都会重新初始化一次。

所以：

```text
max_num_scheduled_tokens 是配置上限；
token_budget 是当前 step 剩余可用 token 数。
```

例如：

```text
max_num_scheduled_tokens = 8192
本轮开始 token_budget = 8192
调度 req-a 1024 token 后 token_budget = 7168
调度 req-b 2048 token 后 token_budget = 5120
...
```

---

## 4. pause 状态会直接影响预算

如果 Scheduler 被设置为 `PAUSED_ALL`：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
```

位置：`scheduler.py:408`

这意味着本轮不调度任何 token。

### 4.1 三种 pause 状态

Scheduler 有三种 pause 状态：

| 状态 | 含义 | 对 token_budget 的影响 |
|---|---|---|
| `UNPAUSED` | 正常调度 | 正常使用 `max_num_scheduled_tokens` |
| `PAUSED_NEW` | 不接纳新请求，只保留 running 推进 | `schedule()` 不直接把 budget 置 0，但 unfinished 统计只看 running |
| `PAUSED_ALL` | 所有请求都暂停 | `token_budget = 0` |

`get_num_unfinished_requests()` 中也体现了 pause 状态：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    return 0
if self._pause_state == PauseState.PAUSED_NEW:
    return len(self.running)
```

位置：`scheduler.py:2106`

### 4.2 为什么 `PAUSED_NEW` 不直接把 token_budget 设为 0

`PAUSED_NEW` 的语义不是停止所有计算，而是不再接纳新的 waiting 请求。

也就是说：

```text
已有 running 请求仍可以继续推进；
新的 waiting 请求不进入 running。
```

`schedule()` 中调度 waiting 的条件是：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`scheduler.py:625`

所以 `PAUSED_NEW` 下 running 阶段仍可用 token budget，但 waiting 阶段不会执行。

---

## 5. Scheduler 的调度顺序决定预算分配优先级

`schedule()` 中先调度 running，再调度 waiting。

### 5.1 running 请求先消耗预算

running 阶段入口：

```python
while req_index < len(self.running) and token_budget > 0:
```

位置：`scheduler.py:431`

只要还有 running 请求，并且 token budget 大于 0，就尝试推进 running 请求。

调度成功后扣预算：

```python
token_budget -= num_new_tokens
```

位置：`scheduler.py:578`

### 5.2 waiting 请求使用剩余预算

waiting 阶段入口：

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
```

位置：`scheduler.py:628`

也就是说：

```text
waiting 请求只能使用 running 阶段剩下的 token_budget。
```

waiting 请求调度成功后同样扣预算：

```python
token_budget -= num_new_tokens
```

位置：`scheduler.py:957`

### 5.3 为什么 running 优先

因为 running 请求已经持有 KV block，并且通常处于 decode 或 chunked prefill 过程中。

优先推进 running 有几个好处：

1. 减少已有请求的 decode latency；
2. 让请求尽快结束，释放 KV block；
3. 避免 waiting 请求挤占 running 请求导致已运行请求长期停滞；
4. 对 async scheduling / PP 更容易保持 batch 连续性。

---

## 6. running 请求如何计算本轮 token 数

running 请求本轮想调度多少 token，核心公式是：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`scheduler.py:462`

解释：

| 字段 | 含义 |
|---|---|
| `request.num_tokens_with_spec` | 当前请求 token 总数，包含 spec decode token |
| `request.num_output_placeholders` | async / PP 场景下已经预留但还没返回的输出 token |
| `request.num_computed_tokens` | Scheduler 认为已经计算过的 token 数 |

因此：

```text
num_new_tokens = 目标应计算 token 数 - 已计算 token 数
```

这统一覆盖了：

- 普通 decode；
- chunked prefill；
- speculative decode；
- pipeline parallel / async scheduling 中的 placeholder。

---

## 7. running 请求 token 数会被哪些条件截断

running 请求算出 `num_new_tokens` 后，不一定全部调度，会继续被多重条件限制。

### 7.1 long prefill threshold

```python
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
```

位置：`scheduler.py:467`

这个配置用于限制单个长 prefill 每轮最多调度多少 token。

如果没有这个限制，长 prompt 请求可能一次吃掉整个 batch token budget，导致其他请求延迟增加。

### 7.2 本轮剩余 token budget

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

位置：`scheduler.py:469`

这是最直接的预算限制。

如果当前剩余 budget 只有 512，即使请求还需要 4096 token，本轮最多也只能安排 512 token。

### 7.3 模型最大长度 max_model_len

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

位置：`scheduler.py:473`

这里防止调度位置超过模型最大上下文长度。

为什么还要减 `self.num_sampled_tokens_per_step`？

因为 decode 通常本轮计算后还会采样一个 token，需要给采样位置留空间。普通自回归模型中：

```python
self.num_sampled_tokens_per_step = 1
```

位置：`scheduler.py:119`

扩散模型可能不采样 token，因此是 0。

### 7.4 encoder input 限制

如果请求有 encoder input：

```python
encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget, ... = self._try_schedule_encoder_inputs(...)
```

位置：`scheduler.py:484`

`_try_schedule_encoder_inputs()` 可能会缩短 `num_new_tokens`，因为本轮 decoder token 范围内如果遇到一个必须先处理的多模态输入，但 encoder budget 或 encoder cache 不够，就只能调度到那个输入之前。

### 7.5 Mamba block 对齐限制

```python
if self.need_mamba_block_aligned_split:
    num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
```

位置：`scheduler.py:498`

Mamba state cache 对 chunk 边界敏感，某些情况下 `num_new_tokens` 会被裁剪到 block 对齐长度。

如果裁剪后为 0，则本轮跳过该 running 请求。

---

## 8. waiting 请求如何计算本轮 token 数

waiting 请求有两种大情况。

### 8.1 外部 KV 异步加载：本轮不计算 token

如果 connector 返回 `load_kv_async=True`：

```python
if load_kv_async:
    num_new_tokens = 0
```

位置：`scheduler.py:781`

这表示本轮只是为远端 KV load 分配 block，不做模型 forward 计算。

这种情况下不会消耗 token budget，因为：

```text
没有本地计算 token。
```

但会占用 KV block。

### 8.2 普通 waiting 请求：剩余未计算 token

普通 waiting 请求本轮要计算：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`scheduler.py:795`

这里的 `num_computed_tokens` 包含：

```text
本地 prefix cache 命中的 token
+ 外部 KV cache 命中的 token
```

例如：

```text
prompt 长度 = 10000
本地 prefix cache 命中 = 3000
外部 KV cache 新增命中 = 4000
num_computed_tokens = 7000
num_new_tokens = 10000 - 7000 = 3000
```

Scheduler 只需要安排剩下 3000 token 的 prefill。

---

## 9. waiting 请求 token 数也会被限制

### 9.1 long prefill threshold

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

位置：`scheduler.py:796`

和 running 一样，防止单个长 prefill 独占本轮预算。

### 9.2 chunked prefill 开关

```python
if (
    not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > token_budget
):
    break
```

位置：`scheduler.py:802`

这是一个非常重要的分支。

如果关闭 chunked prefill，则一个 waiting 请求必须能在本轮完整安排它需要的 prefill token，否则停止 waiting 调度。

也就是说：

```text
关闭 chunked prefill：请求不能被切块 prefill。
```

如果请求剩余 4096 token，但本轮只剩 2048 budget，则它不能调度。

如果启用 chunked prefill，则允许部分调度：

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

位置：`scheduler.py:810`

### 9.3 encoder input 限制

waiting 请求也会调用：

```python
_try_schedule_encoder_inputs(...)
```

位置：`scheduler.py:814`

它可能把 `num_new_tokens` 降为 0。

```python
if num_new_tokens == 0:
    break
```

位置：`scheduler.py:827`

### 9.4 Mamba block 对齐限制

```python
if self.need_mamba_block_aligned_split and not load_kv_async:
    num_new_tokens = self._mamba_block_aligned_split(...)
    if num_new_tokens == 0:
        break
```

位置：`scheduler.py:832`

如果 Mamba 对齐后不能形成可调度 chunk，则 waiting 调度停止。

---

## 10. encoder_compute_budget：另一个独立预算

除了 `token_budget`，Scheduler 还有 encoder 侧预算：

```python
encoder_compute_budget = self.max_num_encoder_input_tokens
```

位置：`scheduler.py:414`

它限制本轮最多处理多少 encoder input。

初始化时：

```python
self.max_num_encoder_input_tokens = (
    mm_budget.encoder_compute_budget if mm_budget else 0
)
```

位置：`scheduler.py:217`

### 10.1 token_budget 和 encoder_compute_budget 的区别

| 预算 | 限制对象 | 典型场景 |
|---|---|---|
| `token_budget` | decoder / language model token 调度量 | 文本 prefill、decode |
| `encoder_compute_budget` | 多模态 encoder input 计算量 | image/audio/video embedding |

两者是独立预算。

一个请求可能 decoder token budget 够，但 encoder budget 不够，最终还是不能调度或只能调度到 encoder input 前面。

### 10.2 encoder budget 如何消耗

在 `_try_schedule_encoder_inputs()` 中：

```python
encoder_compute_budget -= num_encoder_embeds
```

位置：`scheduler.py:1428`

如果某个 encoder input 已经在远端 EC cache 里，则会进入 external load 路径，不一定本地消耗 encoder compute。

---

## 11. running 数量限制不是 token budget，但会影响能调度多少请求

初始化：

```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
```

位置：`scheduler.py:107`

waiting 调度时：

```python
if len(self.running) == self.max_num_running_reqs:
    break
```

位置：`scheduler.py:629`

这不是 token 数限制，而是 sequence/request 数限制。

即使 token budget 还有剩余，如果 running 请求数量已经达到上限，也不能再接纳新的 waiting 请求。

所以 Scheduler 的本轮调度量受两个维度共同限制：

```text
token 维度：max_num_scheduled_tokens / token_budget
sequence 维度：max_num_running_reqs / max_num_seqs
```

---

## 12. token_budget 如何被真正扣减

### 12.1 running 请求扣减

running 调度成功：

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

位置：`scheduler.py:577`

### 12.2 waiting 请求扣减

waiting 调度成功：

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

位置：`scheduler.py:956`

### 12.3 异步 KV load 不扣 token budget

`load_kv_async` 时：

```python
num_new_tokens = 0
```

位置：`scheduler.py:781`

然后请求进入 `WAITING_FOR_REMOTE_KVS`，不会把它作为普通本地计算加入 running。

所以它不消耗本轮 token budget，但会分配 KV block。

这点很重要：

```text
token_budget 管的是本轮模型计算 token；
KV block 分配是另一种资源约束。
```

---

## 13. 总量校验

调度完成后，Scheduler 会校验本轮总调度 token：

```python
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert token_budget >= 0
```

位置：`scheduler.py:988`

这两个 assert 保证：

1. 记录下来的总调度 token 没超过配置上限；
2. 局部剩余预算没有被扣成负数。

最终 `SchedulerOutput` 中会带上：

```python
num_scheduled_tokens=num_scheduled_tokens
total_num_scheduled_tokens=total_num_scheduled_tokens
```

位置：`scheduler.py:1060`

这就是 ModelRunner 后续知道本轮每个请求该处理多少 token 的依据。

---

## 14. `_update_after_schedule()` 为什么会影响下一轮预算判断

调度输出构造后，会调用：

```python
self._update_after_schedule(scheduler_output)
```

位置：`scheduler.py:1096`

里面会更新：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`scheduler.py:1139`

这意味着：

```text
本轮 schedule 成功后，Scheduler 立刻认为这些 token 已经进入计算进度。
```

即使 Worker 输出还没回来，下一轮调度也会基于更新后的 `num_computed_tokens` 继续做预算判断。

为什么这样设计？

因为 async scheduling / pipeline parallel 下，Scheduler 可能在上一轮结果返回前就准备下一轮。它必须先把已安排的 token 计入进度，避免重复调度同一段 token。

如果后续投机解码有 token 被拒绝，再在 `update_from_output()` 中回退。

位置：`scheduler.py:1562`

---

## 15. DP prefill balancing 对预算使用的影响

`schedule()` 支持一个参数：

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
```

位置：`scheduler.py:387`

如果开启 throttle：

```python
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```

位置：`scheduler.py:425`

含义：

```text
在某些 DP 平衡场景下，如果当前有 decode 请求，prefill 可以延后，避免 prefill 抢占 decode cadence。
```

running 中的 prefill chunk 会被跳过：

```python
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

位置：`scheduler.py:456`

waiting 中的新 prefill 也可能被延后：

```python
elif defer_prefills and request.num_computed_tokens == 0:
    break
```

位置：`scheduler.py:785`

所以 token_budget 即使还剩，也可能不用于 prefill。

这说明：

```text
有 token_budget 不代表一定会用完。
```

Scheduler 会基于延迟、平衡、cache、block 等条件决定是否使用预算。

---

## 16. 为什么本轮可能调度不到 max_num_scheduled_tokens

即使 `max_num_scheduled_tokens` 很大，本轮实际调度 token 也可能更少。

常见原因：

1. 没有足够请求；
2. running 请求暂时不可 decode；
3. waiting 请求数量达到 `max_num_running_reqs`；
4. KV Cache block 不够；
5. 关闭 chunked prefill，剩余 budget 不够完整 prefill 一个请求；
6. encoder budget 不够；
7. Mamba 对齐导致本轮不能形成合法 chunk；
8. Scheduler pause；
9. DP prefill throttling 延后 prefill；
10. 外部 KV async load 本轮只分配 block，不计算 token。

因此：

```text
max_num_scheduled_tokens 是上限，不是每轮必须跑满的目标。
```

---

## 17. 和 KVPool / 外部 KV Cache 的关系

外部 KV Cache 会影响 token budget 的使用方式。

waiting 请求先查本地和外部命中：

```python
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

位置：`scheduler.py:744`

然后只调度剩余 token：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`scheduler.py:795`

如果外部 KV 命中越多，`num_new_tokens` 越少，本轮 token budget 消耗越少。

例如：

```text
prompt = 16000 token
无缓存命中：num_new_tokens = 16000
本地/外部缓存命中 12000：num_new_tokens = 4000
外部 async load：num_new_tokens = 0，本轮只发起加载
```

所以 KVPool 对 Scheduler 的意义是：

```text
把原本需要本轮模型计算的 token 转换成已计算 token，从而节省 token_budget。
```

但外部 KV load 仍然需要 KV block 容量，因此可能不消耗 token budget，却消耗 block 资源。

---

## 18. 一个完整例子

假设：

```text
max_num_scheduled_tokens = 8192
本轮开始 token_budget = 8192
running:
  req-a 需要 decode 1 token
  req-b 是 chunked prefill，还需要 4096 token
waiting:
  req-c prompt 12000，本地/外部 cache 命中 8000，还需 4000
```

调度过程：

```text
1. 调度 running req-a：num_new_tokens = 1
   token_budget = 8191

2. 调度 running req-b：num_new_tokens = 4096
   token_budget = 4095

3. 调度 waiting req-c：num_new_tokens = min(4000, 4095) = 4000
   token_budget = 95

4. 本轮 total_num_scheduled_tokens = 1 + 4096 + 4000 = 8097
```

如果 `req-c` 没有 cache 命中，需要 12000 token：

- 开启 chunked prefill：本轮最多调度 4095；
- 关闭 chunked prefill：req-c 本轮不能调度，waiting 阶段停止。

---

## 19. 总结

Scheduler 回答“本轮最多能调度多少 token”时，核心答案是：

```text
上限 = self.max_num_scheduled_tokens
本轮剩余 = token_budget
实际调度 = sum(num_scheduled_tokens.values())
```

主流程：

```text
schedule()
  → token_budget = max_num_scheduled_tokens
  → 如果 PAUSED_ALL，token_budget = 0
  → running 请求先消耗预算
  → waiting 请求使用剩余预算
  → 每个请求调度成功后 token_budget -= num_new_tokens
  → total_num_scheduled_tokens = sum(num_scheduled_tokens)
  → assert 不超过 max_num_scheduled_tokens
  → 写入 SchedulerOutput
```

但实际能调度多少还受到这些因素影响：

```text
max_num_running_reqs
long_prefill_token_threshold
enable_chunked_prefill
max_model_len
encoder_compute_budget
Mamba block alignment
KV Cache block 是否够用
KV Connector 外部命中 / async load
pause state
DP prefill throttling
spec decode / async placeholders
```

因此，`token_budget` 是 Scheduler 的“总 token 预算”，但最终调度量是多个约束共同作用后的结果。
