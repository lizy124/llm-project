# 04. InputProcessor 如何把用户输入转成 EngineCoreRequest？

源码位置：`vllm/vllm/v1/engine/processor.py`

本问题关注：`InputProcessor` 在外层 Engine 中的位置，以及它和 `EngineCoreRequest`、`Request` 的边界。

---

## 1. 一句话回答

占位。

待展开结论：

```text
InputProcessor 属于外层 Engine；
它把用户侧输入处理成 EngineCore 能接受的 EngineCoreRequest。
```

---

## 2. InputProcessor 位于哪一层

占位。

计划说明：

```text
LLMEngine / AsyncLLM 持有 InputProcessor。
EngineCore 不直接处理用户原始 prompt。
```

---

## 3. 输入对象转换关系

占位。

计划梳理：

```text
用户输入 / EngineInput
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
```

---

## 4. InputProcessor 处理哪些内容

占位。

计划补充：

```text
prompt / prompt token ids；
sampling params / pooling params；
多模态输入；
request metadata；
结构化输出相关输入。
```

---

## 5. 容易混淆的点

占位。

计划回答：

```text
EngineCoreRequest 不是 Scheduler 内部 Request。
InputProcessor 不负责调度。
InputProcessor 不负责 detokenize 输出。
```

---

## 6. 从“回答问题”的角度总结

占位。