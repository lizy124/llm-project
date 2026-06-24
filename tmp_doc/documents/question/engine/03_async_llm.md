# 03. AsyncLLM 负责什么？

源码位置：`vllm/vllm/v1/engine/async_llm.py`

本问题关注：异步接口里的 `AsyncLLM` 如何作为外层 Engine，异步地提交请求、消费输出，并对外提供 async generator 风格的结果。

---

## 1. 一句话回答

占位。

待展开结论：

```text
AsyncLLM 是异步路径里的外层 Engine；
它和 LLMEngine 的核心边界相同，但请求提交和输出消费采用 async / background handler 方式。
```

---

## 2. 初始化阶段

占位。

计划梳理：

```text
AsyncLLM
  → 创建 InputProcessor
  → 创建 OutputProcessor
  → 创建 AsyncMPClient / EngineCoreClient
  → 准备 output_handler
```

---

## 3. 异步请求入口

占位。

计划梳理：

```text
AsyncLLM.add_request / generate
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → engine_core.add_request_async()
```

---

## 4. 后台输出处理

占位。

计划梳理：

```text
output_handler
  → engine_core.get_output_async()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutputCollector
  → async generator
```

---

## 5. AsyncLLM 和 LLMEngine 的异同

占位。

计划说明：

```text
相同：都属于外层 Engine，都使用 InputProcessor / OutputProcessor / EngineCoreClient。
不同：LLMEngine 由调用方 step 拉输出；AsyncLLM 由后台任务持续拉输出。
```

---

## 6. 从“回答问题”的角度总结

占位。