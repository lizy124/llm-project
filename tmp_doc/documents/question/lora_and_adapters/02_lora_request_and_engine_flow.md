# 02. LoRARequest 如何从用户请求进入执行链路？

源码位置：

- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/engine/__init__.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/output.py`

本问题关注：用户请求携带的 LoRA 信息如何进入 EngineCoreRequest、Request，并最终被 Worker 看到。

---

## 1. 一句话回答

`LoRARequest` 是请求级 adapter 选择信息，它随请求状态从 Engine 进入 Scheduler，再随 SchedulerOutput 被 Worker / ModelRunner 消费。

```text
API / LLM request
  → LoRARequest
  → EngineCoreRequest.lora_request
  → Request.lora_request
  → SchedulerOutput
  → GPUModelRunner._update_states()
```

---

## 2. LoRARequest 字段占位

后续补充：

```text
- lora_name；
- lora_int_id；
- lora_path；
- base_model_name；
- adapter metadata；
- prompt adapter 相关字段是否共用 adapter_commons。
```

---

## 3. 请求流转占位

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.schedule()
  → NewRequestData / cached request state
  → Worker._update_states()
```

---

## 4. 容易混淆点占位

```text
1. LoRARequest 只是选择 adapter，不等于 adapter 已加载。
2. adapter 加载由 LoRA manager / Worker 控制面负责。
3. Scheduler 通常不执行 LoRA forward。
4. Worker 需要根据当前 batch 的 LoRARequest 设置 active LoRA。
```

---

## 5. 一句话总结

```text
LoRARequest 是请求和 adapter 之间的绑定信息，真正执行发生在 Worker / LoRA layer。
```
