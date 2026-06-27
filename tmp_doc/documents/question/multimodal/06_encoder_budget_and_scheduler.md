# 06. Scheduler 如何调度多模态 encoder input？

源码位置：

- `code/vllm/vllm/multimodal/encoder_budget.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/request.py`

本问题关注：多模态请求除了 decoder token budget，还可能需要 encoder budget。Scheduler 如何决定本轮哪些 encoder input 可以执行，如何把它们放进 `SchedulerOutput.scheduled_encoder_inputs`，以及 encoder cache 如何影响调度。

---

## 1. 一句话回答占位

```text
Scheduler 在调度 token 的同时，还要根据 encoder budget 和 encoder cache 状态，决定哪些多模态 encoder input 本轮执行，并通过 SchedulerOutput 交给 Worker。
```

---

## 2. 为什么需要 encoder budget 占位

```text
- 多模态 encoder 可能很重；
- 一轮 batch 不能无限处理 image/video/audio feature；
- encoder output 可能被 cache 命中而跳过；
- decoder token budget 和 encoder compute budget 需要协调。
```

---

## 3. 主链路占位

```text
Request.mm_features
  → Scheduler 检查 encoder cache
  → 估算 encoder budget
  → 选择 scheduled_encoder_inputs
  → SchedulerOutput.scheduled_encoder_inputs
  → ModelRunner._execute_mm_encoder()
```

---

## 4. 需要解释的问题占位

```text
- MultiModalBudget 如何计算？
- encoder input 如何挂在 Request 上？
- SchedulerOutput.scheduled_encoder_inputs 的结构是什么？
- encoder cache 命中时是否还需要执行 encoder？
- encoder input 和 prefill/decode 调度如何配合？
```

---

## 5. 后续待补源码证据

占位：补充 Scheduler 中 encoder input 调度、`SchedulerOutput` 字段、`MultiModalBudget` 和 `EncoderCacheManager` 的源码位置。
