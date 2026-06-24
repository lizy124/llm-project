# 07. 一个请求在外层 Engine 中如何流动？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/processor.py`
- `vllm/vllm/v1/engine/core_client.py`

本问题关注：用户请求从外层 Engine 入口到进入 EngineCore / Scheduler 的完整路径。

---

## 1. 一句话回答

占位。

待展开结论：

```text
用户请求先进入 LLMEngine / AsyncLLM，经过 InputProcessor 转成 EngineCoreRequest，再通过 EngineCoreClient 送入 EngineCore，最后由 EngineCore 转成 Request 交给 Scheduler。
```

---

## 2. 同步请求路径

占位。

计划梳理：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

---

## 3. 异步请求路径

占位。

计划梳理：

```text
AsyncLLM.generate / add_request
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request_async()
  → EngineCoreProc input queue
  → EngineCore.add_request()
  → Scheduler.add_request()
```

---

## 4. 为什么 OutputProcessor.add_request 在输出前出现

占位。

计划说明：

```text
OutputProcessor 需要提前登记 request 状态，后续才能把 EngineCoreOutput 增量合并成 RequestOutput。
```

---

## 5. abort 请求路径

占位。

计划梳理：

```text
LLMEngine / AsyncLLM abort
  → EngineCoreClient.abort_requests()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

---

## 6. 从“回答问题”的角度总结

占位。