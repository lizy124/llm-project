# 05. ModelRunner / Worker 执行阶段如何接入？

源码位置：`vllm/v1/engine/core.py`

本问题关注：EngineCore 如何把 SchedulerOutput 交给 ModelRunner / Worker，Worker 如何执行 forward，并返回 ModelRunnerOutput。

---

## 1. 一句话回答

TODO

```text
EngineCore 不直接实现模型 forward；
它把 SchedulerOutput 交给 ModelRunner / Worker，
由后者根据执行计划完成模型计算，并返回 ModelRunnerOutput。
```

---

## 2. 本地 ModelRunner 执行路径

TODO

---

## 3. Worker / Executor 执行路径

TODO

---

## 4. ModelRunnerOutput 包含什么

TODO

```text
sampled_token_ids
logprobs
pooler_output
kv_connector_output
req_id_to_index
```

---

## 5. 执行异常如何处理

TODO

---

## 6. 总结

TODO
