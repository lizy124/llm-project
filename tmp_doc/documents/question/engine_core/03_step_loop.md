# 03. EngineCore.step() 如何驱动一轮执行？

源码位置：`vllm/v1/engine/core.py`

本问题关注：一次 EngineCore step 的主流程，以及 EngineCore 如何串起 Scheduler.schedule()、ModelRunner / Worker forward、Scheduler.update_from_output()。

---

## 1. 一句话回答

TODO

```text
一次 EngineCore.step() 大致是：
调用 Scheduler.schedule() 生成执行计划，
把 SchedulerOutput 发给 Worker / ModelRunner 执行，
拿回 ModelRunnerOutput，
再调用 Scheduler.update_from_output() 产生 EngineCoreOutputs。
```

---

## 2. step 主流程图

TODO

```text
EngineCore.step()
  → scheduler.schedule()
  → scheduler_output
  → model_executor / worker / model_runner execute
  → model_runner_output
  → scheduler.update_from_output(scheduler_output, model_runner_output)
  → engine_core_outputs
```

---

## 3. schedule 阶段

TODO

---

## 4. forward 执行阶段

TODO

---

## 5. update_from_output 阶段

TODO

---

## 6. step 返回值

TODO

---

## 7. 总结

TODO
