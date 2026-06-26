# 08. Scheduler 如何消费 ModelRunnerOutput？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/engine/core.py`

本问题关注：`Scheduler.update_from_output()` 如何把 worker 产出的 token / logprobs / connector 状态消化回 request 状态机，并生成 `EngineCoreOutputs`。

---

## 1. 一句话回答

Scheduler 回收输出不是简单转发 token，而是一次状态机推进：

```text
ModelRunnerOutput
  → append sampled tokens
  → 更新 request 状态
  → 检查 stop condition
  → 处理 spec decode / grammar / KV connector
  → 释放资源
  → EngineCoreOutputs
```

---

## 2. update_from_output 主职责占位

```text
- 将 sampled_token_ids 写回 request.output_token_ids；
- 更新 num_computed_tokens；
- 处理 prompt logprobs / generation logprobs；
- 检查 stop token / stop string / max_tokens / EOS；
- 处理 spec decode accepted / rejected tokens；
- 推进 structured output / grammar 状态；
- 处理 finished_sending / finished_recving；
- 释放 KV blocks / encoder cache / connector metadata；
- 构造 EngineCoreOutputs。
```

---

## 3. 输入输出占位

```text
输入：
  SchedulerOutput：本轮调度了谁、调度了多少 token。
  ModelRunnerOutput：本轮 worker 实际产出了什么。

输出：
  EngineCoreOutputs：给 Engine / OutputProcessor 消费的内部输出。
```

---

## 4. request 状态推进占位

后续补充：

```text
- WAITING / RUNNING / FINISHED 状态转换；
- decode token append；
- prefill chunk 的 discard / no-sample 情况；
- spec decode 的 accepted count；
- stop reason / finish reason 设置；
- abort / preemption / recompute 相关边界。
```

---

## 5. 资源释放占位

```text
完成请求时需要处理：

- KV cache blocks；
- encoder cache；
- KV connector remote state；
- structured output state；
- parallel sampling 子请求状态；
- metrics / stats。
```

---

## 6. 一句话总结

```text
Scheduler.update_from_output() 是采样结果回到调度状态机的桥，它决定 token 是否真正成为 request 的一部分。
```
