# vLLM LoRA / Adapters 逻辑梳理

源码位置：

- `code/vllm/vllm/lora/`
- `code/vllm/vllm/lora/request.py`
- `code/vllm/vllm/lora/models.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/lora/layers.py`
- `code/vllm/vllm/lora/punica_wrapper/`
- `code/vllm/vllm/adapter_commons/`
- `code/vllm/vllm/v1/engine/`
- `code/vllm/vllm/v1/core/sched/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/executor/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/config.py`

本文按“先定边界，再走请求到 worker 主链路，再拆 adapter 管理、layer 注入、batch 混合执行和生命周期”的方式，梳理 vLLM 中 LoRA / adapter 机制。

LoRA 与普通模型权重不同：

```text
base model：
  在 worker 初始化时加载，所有请求共享。

LoRA adapter：
  可以动态加载、卸载、pin，
  请求级选择，
  batch 内可混合多个 adapter，
  forward 时作为低秩 delta 叠加到 base layer 上。
```

---

## 0. 梳理规划

本目录要回答的问题分成 11 组：

```text
1. LoRA / adapter 在 vLLM 中处于哪一层？
2. LoRARequest 如何从 API / Engine 进入 Scheduler 和 Worker？
3. LoRA manager 如何加载、缓存、pin、卸载 adapter？
4. Worker / ModelRunner 如何维护当前 batch 的 active LoRA 状态？
5. LoRA layer 如何注入 Linear / Embedding / LM head？
6. 同一个 batch 中混合多个 LoRA 请求如何执行？
7. LoRA 权重如何加载、命名映射和切分？
8. LoRA 如何与量化 base model 共存？
9. LoRA 如何与 TP / PP / DP 并行交互？
10. add_lora / remove_lora / pin_lora / list_loras 生命周期如何走？
11. 常见限制、错误和调试入口有哪些？
```

阅读顺序建议：

```text
lora_and_adapters_overview.md
  → 01_lora_role.md
  → 02_lora_request_and_engine_flow.md
  → 03_lora_manager_and_cache.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 06_batch_mixed_lora_execution.md
  → 07_lora_loading_and_weight_mapping.md
  → 08_lora_and_quantization.md
  → 09_lora_and_parallelism.md
  → 10_lora_lifecycle_and_control.md
  → 11_lora_limitations_and_debugging.md
```

---

## 1. 一句话回答

vLLM 中的 LoRA 是一种“请求级可切换的模型增量权重”机制：

```text
base model 固定加载，
LoRA adapter 动态加载和缓存，
请求通过 LoRARequest 指定 adapter，
ModelRunner 在 batch 执行前激活对应 LoRA，
LoRA layer 在 forward 中把低秩 delta 叠加到 base layer 输出。
```

最小主线是：

```text
用户请求携带 LoRARequest
  → Engine / InputProcessor
  → EngineCoreRequest.lora_request
  → Request.lora_request
  → SchedulerOutput
  → GPUModelRunner._update_states()
  → InputBatch 记录每个 request 的 LoRA
  → _prepare_inputs() / set_active_loras()
  → LoRA manager 确保 adapter 已加载
  → LoRA layer / punica wrapper 执行 batch mixed LoRA
  → model forward
  → output
```

---

## 2. LoRA 机制的几个层次

```text
请求层：
  LoRARequest 表达这个请求想使用哪个 adapter。

管理层：
  LoRA manager 负责加载、缓存、pin、卸载 adapter。

状态层：
  InputBatch / ModelRunner 维护当前 batch 中每个 request 对应哪个 LoRA。

layer 层：
  LoRA-wrapped Linear / Embedding / LM head 在 forward 中叠加 delta。

kernel 层：
  punica wrapper 或其他 kernel 支持 batch 内不同 request 使用不同 LoRA。

控制面：
  Executor / Worker 暴露 add_lora、remove_lora、pin_lora、list_loras 等能力。
```

---

## 3. 总体流程图

```text
请求提交
  → lora_request
  → EngineCoreRequest
  → Request
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner._update_states()
      → 新请求保存 lora_request
      → InputBatch.add_request()
  → GPUModelRunner._prepare_inputs()
      → 根据当前 batch 设置 active LoRAs
  → model forward
      → LoRA layer 读取 active mapping
      → base output + LoRA delta
  → logits / pooling / output
```

控制面流程：

```text
Executor.add_lora()
  → Worker.add_lora()
  → LoRA manager 加载 adapter
  → adapter cache 更新

Executor.remove_lora()
  → Worker.remove_lora()
  → LoRA manager 卸载 adapter

Executor.pin_lora()
  → Worker.pin_lora()
  → adapter 标记常驻
```

---

## 4. 和其他专题的关系

```text
engine / engine_core：
  解释 LoRARequest 如何进入内部 Request。

scheduler：
  Scheduler 不执行 LoRA forward，但调度输出会携带请求状态，Worker 需要知道当前 batch 的 LoRA。

executor_worker_model_runner：
  解释 Worker / ModelRunner 如何在每轮执行前激活 LoRA。

quantization：
  解释量化 base model 与 LoRA delta 如何共存。

parallelism：
  解释 TP / PP / DP 下 LoRA 权重和 active mapping 如何对齐。

sampling_and_output：
  LoRA 通常不改变 sampling 机制，但会改变 logits 来源。
```

---

## 5. 后续专题占位

```text
01_lora_role.md：
  定义 LoRA 在 vLLM 中的职责、边界和与 base model 的关系。

02_lora_request_and_engine_flow.md：
  梳理 LoRARequest 如何从用户请求进入 EngineCoreRequest、Request 和 SchedulerOutput。

03_lora_manager_and_cache.md：
  梳理 LoRA manager 如何加载、缓存、pin、卸载 adapter。

04_worker_model_runner_lora_state.md：
  梳理 Worker / ModelRunner / InputBatch 如何维护当前 batch 的 active LoRA 状态。

05_lora_layer_injection.md：
  梳理 LoRA layer 如何包装 Linear、Embedding、LM head 并在 forward 中叠加 delta。

06_batch_mixed_lora_execution.md：
  梳理同一 batch 中不同请求使用不同 LoRA 时如何执行。

07_lora_loading_and_weight_mapping.md：
  梳理 LoRA checkpoint 的权重命名、rank、alpha、target modules 和加载映射。

08_lora_and_quantization.md：
  梳理量化 base model 与 LoRA adapter 共存方式和限制。

09_lora_and_parallelism.md：
  梳理 TP / PP / DP 下 LoRA 权重切分、广播和 rank 映射。

10_lora_lifecycle_and_control.md：
  梳理 add_lora、remove_lora、pin_lora、list_loras、shutdown 等控制接口。

11_lora_limitations_and_debugging.md：
  梳理常见限制、错误、fallback 和调试入口。
```

---

## 6. 一句话总结

LoRA 在 vLLM 中是一套“动态 adapter 执行系统”：

```text
请求选择 adapter，
worker 管理 adapter，
ModelRunner 激活 adapter，
LoRA layer 在 forward 中叠加 adapter，
控制面负责加载、卸载和查询 adapter。
```
