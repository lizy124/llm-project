# 04. SchedulerOutput 在 EngineCore 中如何流动？

源码位置：`vllm/v1/engine/core.py`

本问题关注：SchedulerOutput 是如何由 Scheduler 生成，并在 EngineCore 中传给 Worker / ModelRunner 的。

---

## 1. 一句话回答

TODO

```text
SchedulerOutput 是 Scheduler 生成的一轮执行计划；
EngineCore 拿到它后，将其交给 ModelRunner / Worker 执行，
并在 Worker 返回后再把同一个 SchedulerOutput 传回 Scheduler.update_from_output() 做对账。
```

---

## 2. SchedulerOutput 是谁生成的

TODO

---

## 3. EngineCore 拿到 SchedulerOutput 后做什么

TODO

---

## 4. SchedulerOutput 中哪些字段影响 Worker 执行

TODO

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
scheduled_encoder_inputs
scheduled_spec_decode_tokens
kv_connector_metadata
ec_connector_metadata
finished_req_ids
```

---

## 5. 空调度如何处理

TODO

---

## 6. 总结

TODO
