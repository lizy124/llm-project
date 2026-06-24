# vLLM V1 EngineCore 逻辑梳理

源码位置：`vllm/v1/engine/core.py`

本文用于总览 `EngineCore` 的职责、主执行循环、Scheduler / Worker 的协作关系，以及 `EngineCoreOutputs` 如何返回给上层。

---

## 1. EngineCore 是什么

TODO

一句话定位：

```text
EngineCore 是 vLLM Engine 内部的执行核心，负责编排 Scheduler 和 Worker / ModelRunner 完成一轮一轮的模型执行。
```

它主要回答：

```text
请求如何进入内部执行流？
一次 step 如何被驱动？
SchedulerOutput 如何交给 Worker？
ModelRunnerOutput 如何回到 Scheduler？
EngineCoreOutputs 如何返回给上层？
```

---

## 2. EngineCore 主链路

TODO

```text
Engine / API
  → EngineCore.add_request()
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → ModelRunner / Worker forward
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → Engine / API 返回上层
```

---

## 3. 和 Scheduler 的关系

TODO

```text
EngineCore：负责编排一轮执行闭环。
Scheduler：负责决定本轮哪些请求执行、每个请求执行多少 token、如何更新请求状态。
```

---

## 4. 和 ModelRunner / Worker 的关系

TODO

```text
EngineCore 把 SchedulerOutput 交给 ModelRunner / Worker；
ModelRunner / Worker 真正执行 forward；
执行结果以 ModelRunnerOutput 返回。
```

---

## 5. 和 Engine / AsyncLLMEngine 的关系

TODO

```text
Engine 更靠外层，负责接收请求、驱动 EngineCore step、把输出返回给上层；
EngineCore 更靠内层，负责执行闭环。
```

---

## 6. 一句话总结

TODO
