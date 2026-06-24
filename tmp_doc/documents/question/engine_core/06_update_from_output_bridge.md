# 06. EngineCore 如何使用 Scheduler.update_from_output()？

源码位置：`vllm/v1/engine/core.py`

本问题关注：EngineCore 在拿到 Worker 返回的 ModelRunnerOutput 后，为什么要把 SchedulerOutput 和 ModelRunnerOutput 一起交回 Scheduler.update_from_output()。

---

## 1. 一句话回答

TODO

```text
SchedulerOutput 是本轮执行计划；
ModelRunnerOutput 是 Worker 的真实执行结果；
EngineCore 把二者一起传给 Scheduler.update_from_output()，
让 Scheduler 根据真实结果更新请求状态并生成 EngineCoreOutputs。
```

---

## 2. 为什么需要 SchedulerOutput

TODO

---

## 3. 为什么需要 ModelRunnerOutput

TODO

---

## 4. update_from_output 返回什么

TODO

---

## 5. 和 scheduler/08 文档的关系

TODO

```text
这里重点讲 EngineCore 的桥接关系；
具体 update_from_output 内部逻辑见：../scheduler/08_update_after_worker_output.md
```

---

## 6. 总结

TODO
