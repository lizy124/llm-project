# vLLM V1 EngineCore 问题目录

源码位置：`vllm/vllm/v1/engine/core.py`

这个目录按问题拆解 vLLM V1 `EngineCore` 的主执行流程，重点回答：请求如何进入 EngineCore、`step()` 如何驱动一轮调度和执行、`SchedulerOutput` 如何交给 Worker、`ModelRunnerOutput` 如何回到 Scheduler、`EngineCoreOutputs` 如何返回上层，以及 EngineCore 在同步 / 异步 / 多进程场景中的生命周期管理。

---

## 1. 总览文档

- [vLLM V1 EngineCore 逻辑梳理](engine_core_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
Engine / API
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → Worker / ModelRunner forward / sample
  → ModelRunnerOutput
  → Scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 2. 主线专题阅读顺序

### 01. EngineCore 的定位

- [EngineCore 在 vLLM 里负责什么？](01_engine_core_role.md)

回答：

```text
Engine 和 EngineCore 是什么关系？
EngineCore 和 EngineCoreClient 是什么关系？
EngineCore 和 Scheduler / ModelRunner / Worker 是什么关系？
EngineCore 是否直接执行 forward？
EngineCore 为什么可以理解为内部执行主循环？
```

核心结论：

```text
EngineCore 是内部执行闭环总控；
Scheduler 负责调度和状态账本；
Worker / ModelRunner 负责真正 forward / sample；
LLMEngine / AsyncLLM 负责外层输入输出。
```

---

### 02. 请求入口

- [请求如何进入 EngineCore？](02_request_entry.md)

回答：

```text
外部请求如何进入 EngineCore？
EngineCoreRequest 和 Request 有什么区别？
add_request 做了什么？
请求什么时候交给 Scheduler？
abort / finish request 如何传入？
```

核心链路：

```text
LLMEngine / AsyncLLM
  → InputProcessor
  → EngineCoreRequest
  → EngineCoreClient
  → EngineCore.preprocess_add_request()
  → Request
  → Scheduler.add_request()
```

---

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

核心链路：

```text
EngineCore.step()
  → scheduler.has_requests()
  → scheduler.schedule()
  → model_executor.execute_model()
  → scheduler.get_grammar_bitmask()
  → model_executor.sample_tokens()  # 如果 execute_model 返回 None
  → _process_aborts_queue()
  → scheduler.update_from_output()
  → EngineCoreOutputs
```

---

### 04. SchedulerOutput 流转

- [SchedulerOutput 在 EngineCore 中如何流动？](04_scheduler_output_flow.md)

回答：

```text
SchedulerOutput 是谁生成的？
EngineCore 拿到 SchedulerOutput 后做什么？
哪些字段会影响 Worker 执行？
空调度如何处理？
SchedulerOutput 为什么还要传回 update_from_output？
```

核心结论：

```text
SchedulerOutput 既是 Worker 的执行计划，
也是 Scheduler.update_from_output() 阶段的对账凭证。
```

---

### 05. Worker 执行阶段

- [ModelRunner / Worker 执行阶段如何接入？](05_worker_execution.md)

回答：

```text
EngineCore 如何把 SchedulerOutput 发给 Worker？
Executor / Worker / ModelRunner 的调用层级是什么？
本地执行和多进程执行有什么差异？
execute_model() 为什么可能返回 None？
sample_tokens() 如何生成 ModelRunnerOutput？
ModelRunnerOutput 包含什么？
Worker 异常如何处理？
```

核心链路：

```text
EngineCore
  → model_executor.execute_model(scheduler_output)
  → Executor.collective_rpc("execute_model")
  → Worker.execute_model()
  → ModelRunner.execute_model()
  → forward / logits / pooling
  → sample_tokens(grammar_output)
  → ModelRunnerOutput
```

---

### 06. 输出回收桥接

- [EngineCore 如何使用 Scheduler.update_from_output()？](06_update_from_output_bridge.md)

回答：

```text
为什么 update_from_output 需要同时拿到 SchedulerOutput 和 ModelRunnerOutput？
SchedulerOutput 在回收阶段提供什么信息？
ModelRunnerOutput 在回收阶段提供什么信息？
Scheduler.update_from_output() 返回什么？
EngineCoreOutputs 是如何产生的？
```

核心结论：

```text
SchedulerOutput 是计划账本；
ModelRunnerOutput 是真实结果；
update_from_output() 负责对账、更新请求状态并生成 EngineCoreOutputs。
```

---

### 07. EngineCoreOutputs 返回上层

- [EngineCoreOutputs 如何返回给上层？](07_engine_core_outputs.md)

回答：

```text
EngineCoreOutput 和 EngineCoreOutputs 有什么区别？
EngineCoreOutputs 里有什么？
为什么按 client_index 分组？
InprocClient 如何取输出？
EngineCoreProc / MPClient 如何通过 ZMQ 返回输出？
LLMEngine / AsyncLLM 如何消费这些输出？
没有 token 的 prefill step 是否返回输出？
EngineCoreOutputs 和 RequestOutput 有什么区别？
```

核心层级：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

### 08. 生命周期与异步能力

- [EngineCore 的异步、并发和生命周期管理](08_lifecycle_and_async.md)

回答：

```text
同步 EngineCore 和异步 / 多进程 EngineCore 有什么区别？
EngineCore 初始化时管理哪些组件？
batch queue 和 async scheduling 如何接入？
EngineCoreProc busy loop 如何工作？
input_queue / output_queue 如何驱动执行？
profile / reset / sleep / wakeup 如何接入？
abort 队列如何处理？
异常和 shutdown 如何传播？
```

核心结论：

```text
EngineCore 管理执行环境、Scheduler、model_executor 和 step_fn；
EngineCoreProc 在此基础上增加 ZMQ、后台 busy loop、输入输出线程和 shutdown 状态机。
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
engine_core_overview.md
  → 01_engine_core_role.md
  → 03_step_loop.md
  → 07_engine_core_outputs.md
```

适合先知道 EngineCore 是什么、核心 step 做什么、输出如何回上层。

---

### 3.2 按主执行链路完整阅读

```text
engine_core_overview.md
  → 01_engine_core_role.md
  → 02_request_entry.md
  → 03_step_loop.md
  → 04_scheduler_output_flow.md
  → 05_worker_execution.md
  → 06_update_from_output_bridge.md
  → 07_engine_core_outputs.md
  → 08_lifecycle_and_async.md
```

适合系统理解 EngineCore 从请求进入到输出返回的完整闭环。

---

### 3.3 重点理解 step 内部闭环

```text
03_step_loop.md
  → 04_scheduler_output_flow.md
  → 05_worker_execution.md
  → 06_update_from_output_bridge.md
```

重点关注：

```text
Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```

---

### 3.4 和 Scheduler 文档联动阅读

```text
01_engine_core_role.md
  → ../scheduler/vllm_scheduler.md
  → 04_scheduler_output_flow.md
  → ../scheduler/08_update_after_worker_output.md
  → 06_update_from_output_bridge.md
  → 07_engine_core_outputs.md
```

重点关注：

```text
EngineCore 负责编排一轮执行闭环；
Scheduler 负责调度决策和请求状态账本；
Worker / ModelRunner 负责真正 forward / sample；
EngineCoreOutputs 是返回给上层 Engine 的内部结果。
```

---

### 3.5 重点理解同步 / 异步 / 多进程

```text
01_engine_core_role.md
  → 07_engine_core_outputs.md
  → 08_lifecycle_and_async.md
```

重点关注：

```text
InprocClient：同进程直接调用 EngineCore。
SyncMPClient / AsyncMPClient：通过 ZMQ 和后台 EngineCoreProc 通信。
EngineCoreProc：EngineCore + input/output socket threads + busy loop + shutdown 状态机。
```

---

## 4. 核心概念速查

### 4.1 对象层级

```text
EngineCoreRequest：
  外层 Engine 传入 EngineCore 的请求对象。

Request：
  EngineCore 转换后交给 Scheduler 的内部请求对象。

SchedulerOutput：
  Scheduler 生成的一轮执行计划。

ModelRunnerOutput：
  Worker / ModelRunner 返回的模型执行结果。

EngineCoreOutput：
  Scheduler 生成的单请求增量输出。

EngineCoreOutputs：
  EngineCore 返回给某个 client 的一批内部输出。

RequestOutput / PoolingRequestOutput：
  OutputProcessor 转换后的用户可见输出。
```

---

### 4.2 组件职责

```text
LLMEngine / AsyncLLM：
  外层用户接口、输入处理、输出处理。

EngineCoreClient：
  屏蔽 in-process / multi-process / async 通信差异。

EngineCore：
  内部执行闭环总控。

Scheduler：
  请求队列、token budget、KV block、状态更新、资源释放。

model_executor：
  执行器抽象，负责把模型执行调用分发到 Worker。

Worker / ModelRunner：
  模型输入准备、forward、sampling、pooling、KV / encoder 实际执行。

OutputProcessor：
  detokenize、stop string、RequestOutput 构造、异步队列推送。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
EngineCore = schedule → execute → update → output 的闭环控制器。
```

展开就是：

```text
Scheduler 决定本轮怎么跑；
Worker / ModelRunner 执行本轮计划；
Scheduler 根据真实结果更新状态；
EngineCore 把输出返回给外层 Engine。
```

再压缩成一句话：

```text
EngineCore 不做调度细节，也不做模型 forward；它负责把 Scheduler 和 Worker 串成一轮一轮可持续执行的内部主循环。
```
