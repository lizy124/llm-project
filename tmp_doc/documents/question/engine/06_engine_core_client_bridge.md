# 06. EngineCoreClient 如何连接外层 Engine 和 EngineCore？

源码位置：`vllm/vllm/v1/engine/core_client.py`

本问题关注：为什么外层 `LLMEngine` / `AsyncLLM` 通常通过 `EngineCoreClient` 访问 `EngineCore`，以及不同 client 形态如何屏蔽同进程、多进程和异步差异。

---

## 1. 一句话回答

占位。

待展开结论：

```text
EngineCoreClient 是外层 Engine 和内部 EngineCore 之间的通信桥；
它让 LLMEngine / AsyncLLM 不需要关心 EngineCore 是同进程对象还是后台进程。
```

---

## 2. EngineCoreClient 的位置

占位。

计划梳理：

```text
LLMEngine / AsyncLLM
  → EngineCoreClient
  → EngineCore / EngineCoreProc
```

---

## 3. InprocClient

占位。

计划说明：

```text
InprocClient 同进程直接创建和调用 EngineCore。
get_output() 里可以直接触发 EngineCore.step_fn()。
```

---

## 4. SyncMPClient

占位。

计划说明：

```text
SyncMPClient 通过 ZMQ 和后台 EngineCoreProc 通信。
外层仍然用同步接口 add_request / get_output。
```

---

## 5. AsyncMPClient

占位。

计划说明：

```text
AsyncMPClient 通过异步 socket / asyncio 路径和后台 EngineCoreProc 通信。
AsyncLLM 用它实现 add_request_async / get_output_async。
```

---

## 6. Client 屏蔽了哪些差异

占位。

计划回答：

```text
请求发送方式；
输出拉取方式；
abort / utility 转发；
后台进程异常传播；
多 client index 输出分发。
```

---

## 7. 从“回答问题”的角度总结

占位。