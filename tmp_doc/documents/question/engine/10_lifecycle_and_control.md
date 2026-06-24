# 10. Engine 的 abort、profile、sleep、shutdown 如何接入？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`

本问题关注：外层 Engine 除了 add_request / step 之外，如何接入取消请求、profile、reset、sleep、wake_up、shutdown 等控制能力。

---

## 1. 一句话回答

占位。

待展开结论：

```text
外层 Engine 暴露控制类接口；
这些接口通常通过 EngineCoreClient 转发到 EngineCore，再由 EngineCore 操作 Scheduler 或 model_executor。
```

---

## 2. abort 请求

占位。

计划梳理：

```text
LLMEngine / AsyncLLM abort
  → EngineCoreClient.abort_requests()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(...)
```

---

## 3. profile / reset

占位。

计划梳理：

```text
外层 Engine utility API
  → EngineCoreClient utility
  → EngineCore.profile / reset_xxx
  → Scheduler / model_executor
```

---

## 4. sleep / wake_up

占位。

计划说明：

```text
sleep / wake_up 属于执行资源控制能力。
EngineCore 负责协调 Scheduler 暂停 / 恢复和 model_executor 显存状态切换。
```

---

## 5. shutdown

占位。

计划说明：

```text
同步路径关注 EngineCore / executor / scheduler 释放。
异步和多进程路径还需要处理后台 EngineCoreProc、socket、output handler 和异常传播。
```

---

## 6. 从“回答问题”的角度总结

占位。