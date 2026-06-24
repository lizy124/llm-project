# 08. 一次输出如何从 EngineCore 回到用户？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/processor.py`
- `vllm/vllm/v1/engine/core_client.py`

本问题关注：`EngineCoreOutputs` 如何被外层 Engine 消费，并转换成最终用户可见输出。

---

## 1. 一句话回答

占位。

待展开结论：

```text
EngineCore 只返回内部 EngineCoreOutputs；
LLMEngine / AsyncLLM 通过 OutputProcessor 把它转换成 RequestOutput / PoolingRequestOutput。
```

---

## 2. 输出对象层级

占位。

计划梳理：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 3. 同步输出路径

占位。

计划梳理：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → list[RequestOutput]
```

---

## 4. 异步输出路径

占位。

计划梳理：

```text
AsyncLLM output_handler
  → EngineCoreClient.get_output_async()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutputCollector
  → async generator yield
```

---

## 5. EngineCoreOutputs 和 RequestOutput 的区别

占位。

计划回答：

```text
EngineCoreOutputs 是内部输出协议。
RequestOutput / PoolingRequestOutput 是用户可见输出协议。
两者之间由 OutputProcessor 转换。
```

---

## 6. 从“回答问题”的角度总结

占位。