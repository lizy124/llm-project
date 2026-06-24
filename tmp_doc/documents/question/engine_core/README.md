# vLLM V1 EngineCore 问题目录

源码位置：`vllm/v1/engine/core.py`

这个目录按问题拆解 vLLM V1 `EngineCore` 的主执行流程。建议先读总览，再按 `01` 到 `08` 顺序读专题。

---

## 1. 总览文档

- [vLLM V1 EngineCore 逻辑梳理](engine_core_overview.md)

适合第一次建立全局印象。它按主链路梳理：

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

## 2. 主线专题阅读顺序

### 01. EngineCore 的定位

- [EngineCore 在 vLLM 里负责什么？](01_engine_core_role.md)

回答：

```text
Engine 和 EngineCore 是什么关系？
EngineCore 和 Scheduler / ModelRunner / Worker 是什么关系？
EngineCore 是否直接执行 forward？
EngineCore 为什么可以理解为内部执行主循环？
```

### 02. 请求入口

- [请求如何进入 EngineCore？](02_request_entry.md)

回答：

```text
外部请求如何进入 EngineCore？
add_request 做了什么？
请求什么时候交给 Scheduler？
abort / finish request 如何传入？
```

### 03. step 主循环

- [EngineCore.step() 如何驱动一轮执行？](03_step_loop.md)

回答：

```text
一次 EngineCore.step() 做了什么？
什么时候调用 Scheduler.schedule()？
什么时候调用 ModelRunner / Worker？
什么时候调用 Scheduler.update_from_output()？
step 返回什么？
```

### 04. SchedulerOutput 流转

- [SchedulerOutput 在 EngineCore 中如何流动？](04_scheduler_output_flow.md)

回答：

```text
SchedulerOutput 是谁生成的？
EngineCore 拿到 SchedulerOutput 后做什么？
哪些字段会影响 Worker 执行？
空调度如何处理？
```

### 05. Worker 执行阶段

- [ModelRunner / Worker 执行阶段如何接入？](05_worker_execution.md)

回答：

```text
EngineCore 如何把 SchedulerOutput 发给 Worker？
本地执行和多进程执行有什么差异？
ModelRunnerOutput 如何返回？
Worker 异常如何处理？
```

### 06. 输出回收桥接

- [EngineCore 如何使用 Scheduler.update_from_output()？](06_update_from_output_bridge.md)

回答：

```text
为什么 update_from_output 需要同时拿到 SchedulerOutput 和 ModelRunnerOutput？
Scheduler.update_from_output() 返回什么？
EngineCoreOutputs 是如何产生的？
```

### 07. EngineCoreOutputs 返回上层

- [EngineCoreOutputs 如何返回给上层？](07_engine_core_outputs.md)

回答：

```text
EngineCoreOutputs 里有什么？
为什么按 client_index 分组？
Engine / AsyncLLMEngine 如何消费这些输出？
finished request 如何返回？
```

### 08. 生命周期与异步能力

- [EngineCore 的异步、并发和生命周期管理](08_lifecycle_and_async.md)

回答：

```text
同步 EngineCore 和异步 EngineCore 有什么区别？
Worker 如何启动和关闭？
profile / reset / sleep / wakeup 如何接入？
异常时如何 shutdown？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立主流程

```text
engine_core_overview.md
  → 01_engine_core_role.md
  → 03_step_loop.md
  → 04_scheduler_output_flow.md
  → 05_worker_execution.md
  → 06_update_from_output_bridge.md
  → 07_engine_core_outputs.md
```

### 3.2 和 Scheduler 文档联动阅读

```text
01_engine_core_role.md
  → ../scheduler/vllm_scheduler.md
  → 03_step_loop.md
  → ../scheduler/08_update_after_worker_output.md
  → 07_engine_core_outputs.md
```

重点关注：

```text
EngineCore 负责编排一轮执行闭环；
Scheduler 负责调度决策和请求状态账本；
Worker / ModelRunner 负责真正 forward；
EngineCoreOutputs 是返回给上层 Engine 的结果。
```
