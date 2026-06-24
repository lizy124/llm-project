# 07. EngineCoreOutputs 如何返回给上层？

源码位置：`vllm/v1/engine/core.py`

本问题关注：Scheduler.update_from_output() 产生的 EngineCoreOutputs 如何被 EngineCore 返回给上层 Engine / AsyncLLMEngine。

---

## 1. 一句话回答

TODO

```text
EngineCoreOutputs 是 EngineCore 一轮 step 的输出结果；
它由 Scheduler.update_from_output() 产生，
再由 EngineCore 返回给外层 Engine / AsyncLLMEngine 消费。
```

---

## 2. EngineCoreOutputs 里有什么

TODO

```text
new_token_ids
finish_reason
logprobs
pooling_output
finished_requests
scheduler_stats
```

---

## 3. 为什么按 client_index 分组

TODO

---

## 4. Engine / AsyncLLMEngine 如何消费输出

TODO

---

## 5. 没有 token 的 prefill step 是否返回输出

TODO

---

## 6. 总结

TODO
