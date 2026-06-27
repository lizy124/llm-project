# 03. Scheduler 如何调度 spec decode？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/spec_decode/`

本问题关注：Scheduler 如何决定本轮哪些请求带 draft tokens，如何把这些 token 放入 SchedulerOutput，以及 token budget / KV allocation 如何受影响。

---

## 1. 一句话回答

Scheduler 在 spec decode 中负责把“请求已有的 draft tokens”转换成本轮 target model 需要验证的执行计划。

```text
request draft state
  → Scheduler.schedule()
  → scheduled_spec_decode_tokens
  → SchedulerOutput
  → Worker / ModelRunner
```

---

## 2. 调度阶段要回答的问题

```text
1. 当前请求有没有 draft tokens？
2. 本轮能验证几个 draft tokens？
3. token budget 是否够？
4. KV cache slots 是否够？
5. structured output grammar 是否允许这些 draft tokens？
6. 本轮是否需要修剪 invalid draft tokens？
7. prefill / chunked prefill 场景是否允许 spec decode？
```

---

## 3. SchedulerOutput 占位

后续补充：

```text
SchedulerOutput.scheduled_spec_decode_tokens：
  表示本轮每个 request 携带的 draft tokens。

其他相关字段：
  scheduled_new_reqs
  scheduled_cached_reqs
  num_scheduled_tokens
  total_num_scheduled_tokens
  num_common_prefix_blocks
```

---

## 4. grammar 修剪占位

Spec decode 与 structured output 结合时，Scheduler 需要提前检查 draft tokens 是否满足 grammar。

```text
如果 draft token 不合法：
  trim / pad scheduled draft tokens，
  记录 invalid spec tokens，
  避免 target forward 验证不可能接受的 token。
```

---

## 5. 一句话总结

```text
Scheduler 负责把 draft tokens 变成 target model 本轮要验证的调度计划。
```
