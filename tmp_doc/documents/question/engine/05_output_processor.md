# 05. OutputProcessor 如何把 EngineCoreOutputs 转成用户输出？

源码位置：`vllm/vllm/v1/engine/processor.py`

本问题关注：`OutputProcessor` 在外层 Engine 中的位置，以及它如何把内部输出协议转换成用户可见的 `RequestOutput` / `PoolingRequestOutput`。

---

## 1. 一句话回答

占位。

待展开结论：

```text
OutputProcessor 属于外层 Engine；
它消费 EngineCoreOutputs / EngineCoreOutput，生成用户可见的 RequestOutput / PoolingRequestOutput。
```

---

## 2. OutputProcessor 位于哪一层

占位。

计划说明：

```text
LLMEngine / AsyncLLM 持有 OutputProcessor。
EngineCore 只返回内部 EngineCoreOutputs。
最终用户输出由 OutputProcessor 构造。
```

---

## 3. 输出对象转换关系

占位。

计划梳理：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 4. OutputProcessor 处理哪些内容

占位。

计划补充：

```text
detokenize；
stop string；
finish reason；
logprobs；
request output aggregation；
async collector 推送。
```

---

## 5. 同步和异步路径差异

占位。

计划说明：

```text
LLMEngine.step() 同步调用 OutputProcessor.process_outputs()。
AsyncLLM output_handler 在后台异步调用 OutputProcessor.process_outputs()。
```

---

## 6. 从“回答问题”的角度总结

占位。