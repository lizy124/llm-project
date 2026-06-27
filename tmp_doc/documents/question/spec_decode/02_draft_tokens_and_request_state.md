# 02. Draft tokens 从哪里来，如何挂在 request 状态上？

源码位置：

- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/spec_decode/`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：draft tokens 的来源、生命周期，以及它们如何进入 SchedulerOutput 和 ModelRunner。

---

## 1. 一句话回答

Draft tokens 是 target model 尚未正式接受的候选输出。

```text
它们可以提前挂在 request 状态上，
但只有经过 target model 验证并被 RejectionSampler 接受后，
才会成为真正的 output tokens。
```

---

## 2. draft token 生命周期占位

```text
产生 draft tokens
  → 挂到 request / scheduler 状态
  → Scheduler 本轮选择携带部分 draft tokens
  → SchedulerOutput.scheduled_spec_decode_tokens
  → ModelRunner 构造 spec decode inputs
  → target model 验证
  → RejectionSampler 接受 / 拒绝
  → Scheduler.update_from_output() 写回 accepted tokens
```

---

## 3. request 状态需要表达什么

后续补充：

```text
- 当前请求有哪些 draft tokens；
- 哪些 draft tokens 本轮被调度；
- draft tokens 和 output_token_ids 的边界；
- draft tokens 被 grammar 修剪后的状态；
- rejected draft tokens 如何丢弃；
- accepted draft tokens 如何 append 到 request。
```

---

## 4. 容易混淆点占位

```text
1. draft token 不是 final output token。
2. scheduled draft token 不代表一定会被接受。
3. target model 验证后，只有 accepted 前缀进入 output_token_ids。
4. rejected token 后可能会采样 replacement token。
5. 全部 draft 接受时可能会产生 bonus token。
```

---

## 5. 一句话总结

```text
draft tokens 是“待验证的预测”，不是“已经生成的输出”。
```
