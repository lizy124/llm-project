# 11. Engine 和 EngineCore 的边界是什么？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/processor.py`

本问题关注：外层 `Engine` 和内部 `EngineCore` 的职责边界，避免把输入输出处理、调度、模型执行混在一起理解。

---

## 1. 一句话回答

占位。

待展开结论：

```text
Engine 是外层输入输出编排层；
EngineCore 是内部执行闭环总控。
```

---

## 2. Engine 负责什么

占位。

计划回答：

```text
接收用户/API 请求；
调用 InputProcessor；
通过 EngineCoreClient 访问 EngineCore；
调用 OutputProcessor；
维护同步或异步对外接口。
```

---

## 3. EngineCore 负责什么

占位。

计划回答：

```text
接收 EngineCoreRequest；
转成内部 Request；
调用 Scheduler.schedule()；
调用 model_executor.execute_model()；
调用 Scheduler.update_from_output()；
返回 EngineCoreOutputs。
```

---

## 4. Engine 不负责什么

占位。

计划说明：

```text
不直接管理 KV block；
不直接执行 ModelRunner forward；
不直接完成 Scheduler.update_from_output() 的状态对账。
```

---

## 5. EngineCore 不负责什么

占位。

计划说明：

```text
不直接处理用户原始 prompt；
不直接 detokenize；
不直接构造最终 RequestOutput；
不直接实现模型 forward。
```

---

## 6. 最小关系图

占位。

计划图：

```text
LLMEngine / AsyncLLM
  → InputProcessor
  → EngineCoreClient
  → EngineCore
  → Scheduler + model_executor
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput
```

---

## 7. 从“回答问题”的角度总结

占位。