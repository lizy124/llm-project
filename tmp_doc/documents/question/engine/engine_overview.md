# vLLM V1 Engine 逻辑梳理

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/processor.py`
- `vllm/vllm/v1/engine/core_client.py`

本文用于总览 vLLM V1 外层 `Engine` 的职责、`LLMEngine` / `AsyncLLM` 的关系、输入输出处理边界，以及外层 Engine 如何通过 `EngineCoreClient` 驱动内部 `EngineCore`。

---

## 1. Engine 是什么

占位。

本节后续补充：

```text
Engine 是外层引擎体系的泛称，不一定是单一具体类。
LLMEngine 是同步形态。
AsyncLLM 是异步形态。
EngineCore 是内部执行核心，不是外层 Engine 本身。
```

---

## 2. Engine 主链路

占位。

计划梳理主线：

```text
用户 / API / LLM
  → LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore
  → Scheduler / Worker / ModelRunner
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 3. LLMEngine：同步外层 Engine

占位。

后续补充：

```text
LLMEngine 初始化 InputProcessor / OutputProcessor / EngineCoreClient。
add_request() 负责把用户输入转成 EngineCoreRequest 并送入 EngineCore。
step() 负责从 EngineCore 拉取 EngineCoreOutputs 并转成 RequestOutput。
```

---

## 4. AsyncLLM：异步外层 Engine

占位。

后续补充：

```text
AsyncLLM 和 LLMEngine 的核心边界相同。
区别在于请求和输出走 async path。
AsyncLLM 通过后台 output_handler 持续消费 EngineCoreOutputs。
```

---

## 5. InputProcessor 的位置

占位。

后续补充：

```text
InputProcessor 属于外层 Engine。
它把用户输入、prompt、sampling params、多模态输入等转换为 EngineCoreRequest。
EngineCore 不直接处理用户原始输入。
```

---

## 6. OutputProcessor 的位置

占位。

后续补充：

```text
OutputProcessor 属于外层 Engine。
它把 EngineCoreOutputs / EngineCoreOutput 转成 RequestOutput / PoolingRequestOutput。
EngineCore 不直接构造最终用户输出。
```

---

## 7. EngineCoreClient 的桥接作用

占位。

后续补充：

```text
外层 Engine 通常持有 EngineCoreClient，而不是总是直接持有 EngineCore。
EngineCoreClient 屏蔽 InprocClient / SyncMPClient / AsyncMPClient 的差异。
```

---

## 8. Engine 和 EngineCore 的边界

占位。

后续补充：

```text
Engine 负责外层输入输出和用户接口。
EngineCore 负责内部执行闭环。
Scheduler 负责调度和请求状态账本。
Worker / ModelRunner 负责真正模型执行。
```

---

## 9. 从“回答问题”的角度总结

占位。

后续形成标准回答：

```text
Engine 可以理解为 vLLM 外层引擎体系的泛称。
在 V1 中，常见具体对象是同步的 LLMEngine 和异步的 AsyncLLM。
它们负责输入处理、输出处理和对 EngineCore 的访问编排；
真正的内部调度与模型执行闭环由 EngineCore 完成。
```