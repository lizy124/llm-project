# 09. Engine 初始化时如何配置 EngineCore？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/executor/abstract.py`

本问题关注：外层 Engine 初始化时如何创建 `InputProcessor`、`OutputProcessor`、`EngineCoreClient`，以及如何把配置和 executor 选择传给内部 EngineCore。

---

## 1. 一句话回答

占位。

待展开结论：

```text
LLMEngine / AsyncLLM 初始化时会根据 vllm_config 创建输入输出处理器，并通过 EngineCoreClient.make_client / make_async_mp_client 创建通往 EngineCore 的访问路径。
```

---

## 2. vllm_config 的作用

占位。

计划说明：

```text
vllm_config 决定模型配置、调度配置、并行配置、cache 配置和执行器选择。
```

---

## 3. executor_class 如何选择

占位。

计划梳理：

```text
Executor.get_class(vllm_config)
  → 选择 uni-proc / multi-proc / distributed executor
  → 传给 EngineCoreClient / EngineCore
```

---

## 4. LLMEngine 初始化链路

占位。

计划梳理：

```text
LLMEngine.__init__
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient.make_client(...)
```

---

## 5. AsyncLLM 初始化链路

占位。

计划梳理：

```text
AsyncLLM.__init__
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient.make_async_mp_client(...)
  → output handler lifecycle
```

---

## 6. 从“回答问题”的角度总结

占位。