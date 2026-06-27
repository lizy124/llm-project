# 09. Scheduler 如何回收 spec decode 输出？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/engine/__init__.py`
- `code/vllm/vllm/v1/engine/output_processor.py`

本问题关注：RejectionSampler 产出的 accepted / replacement / bonus tokens 如何通过 ModelRunnerOutput 回到 Scheduler，并最终变成用户可见输出。

---

## 1. 一句话回答

Spec decode 的输出回收不是简单 append sampler token，而是：

```text
只 append 被接受的 tokens，
丢弃或修正 rejected draft tokens，
处理 bonus / replacement token，
更新 request 状态，
再交给 OutputProcessor 做用户输出。
```

---

## 2. ModelRunnerOutput 占位

后续补充：

```text
sampled_token_ids
logprobs
prompt_logprobs_dict
spec decode accepted token 信息
req_id_to_index
```

---

## 3. Scheduler.update_from_output 占位

需要梳理：

```text
- 如何根据 req_id 找到 batch index；
- 如何从 sampled_token_ids 中取 spec decode 结果；
- 如何 append accepted tokens；
- 如何设置 stop / finish reason；
- 如何处理 grammar accept_tokens；
- 如何处理 KV connector / finished req cleanup；
- 如何构造 EngineCoreOutputs。
```

---

## 4. OutputProcessor 关系占位

OutputProcessor 通常不关心 token 是普通 decode 还是 spec decode 来的。

它看到的是 Scheduler 已经确认过的：

```text
new_token_ids
finish_reason
stop_reason
logprobs
```

也就是说：

```text
spec decode 的复杂性主要在 Scheduler 回收前被消化。
```

---

## 5. 一句话总结

```text
Scheduler.update_from_output() 是 spec decode 从“验证结果”变成“正式输出状态”的桥。
```
