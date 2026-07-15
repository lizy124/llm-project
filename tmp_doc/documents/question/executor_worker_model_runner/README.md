# vLLM V1 Executor / Worker / ModelRunner 问题目录

源码位置：

- `vllm/vllm/v1/executor/abstract.py:37`：Executor 抽象接口与后端选择。
- `vllm/vllm/v1/executor/uniproc_executor.py:45`、`multiproc_executor.py:103`、`ray_executor.py:64`、`ray_executor_v2.py:219`：主要执行后端。
- `vllm/vllm/v1/worker/worker_base.py:39`、`worker_base.py:142`、`worker_base.py:153`：`WorkerBase` 抽象层与 execute_model / sample_tokens 接口。
- `vllm/vllm/v1/worker/gpu_worker.py:297`、`gpu_worker.py:1002`：GPUWorker 初始化与执行入口。
- `vllm/vllm/v1/worker/cpu_worker.py:33`、`xpu_worker.py:24`：CPU / XPU Worker 入口。
- `vllm/vllm/v1/worker/gpu_model_runner.py:445`、`gpu_model_runner.py:4097`、`gpu_model_runner.py:4483`：GPUModelRunner 主类、execute_model 与 sample_tokens。
- `vllm/vllm/v1/worker/gpu/`：GPU 输入 batch、attention、sampling、spec decode、structured output 等拆分实现。
- `vllm/vllm/v1/outputs.py:196`、`outputs.py:225`、`outputs.py:234`：`KVConnectorOutput` / `ECConnectorOutput` / `ModelRunnerOutput` 等执行层输出结构。

这个目录按问题拆解 vLLM V1 的执行层，重点回答：`Executor` 负责什么、`Worker` 负责什么、`ModelRunner` 负责什么、`SchedulerOutput` 如何进入真正的模型执行、`ModelRunnerOutput` / `AsyncModelRunnerOutput` 如何产生，以及执行层如何和 KV Cache / Encoder Cache Connector、Attention、Sampling、structured output、spec decode、生命周期管理协同。

---

## 1. 总览文档

- [vLLM V1 Executor / Worker / ModelRunner 逻辑梳理](executor_worker_model_runner_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
EngineCore
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → ModelRunner.execute_model()
  → prepare inputs / forward / logits / pooling / KV&EC connector hooks
  → sample_tokens()（部分路径 execute_model 先返回 None，再由 sample_tokens 完成采样）
  → ModelRunnerOutput / AsyncModelRunnerOutput
  → Scheduler.update_from_output()
```

---

## 2. 主线专题阅读顺序

### 01. Executor 的定位

- [Executor 在 vLLM V1 里负责什么？](01_executor_role.md)

回答：

```text
Executor 是什么层？
它和 EngineCore / Worker / ModelRunner 什么关系？
它如何选择执行后端？
它如何把执行请求分发到 Worker？
```

### 02. Worker 的定位

- [Worker 在 vLLM V1 里负责什么？](02_worker_role.md)

回答：

```text
Worker 负责什么初始化？
它如何加载模型？
它如何管理 device、KV cache、LoRA、profile、sleep / wake_up？
```

### 03. ModelRunner 的定位

- [ModelRunner 在 vLLM V1 里负责什么？](03_model_runner_role.md)

回答：

```text
ModelRunner 如何消费 SchedulerOutput？
它如何准备输入 batch？
它如何做 forward、logits、pooling、sampling？
```

### 04. execute_model 主链路

- [execute_model() 如何从 EngineCore 走到模型执行？](04_execute_model_flow.md)

回答：

```text
Executor.execute_model() 做了什么？
Worker.execute_model() 做了什么？
ModelRunner.execute_model() 做了什么？
调用层级是什么？
```

### 05. 输入批次与状态更新

- [Worker / ModelRunner 如何维护 batch 和请求状态？](05_input_batch_and_state_update.md)

回答：

```text
InputBatch 是什么？
Worker 侧状态如何更新？
请求如何进入当前 batch？
```

### 06. 输入准备与 attention metadata

- [ModelRunner 如何准备输入和 attention metadata？](06_prepare_inputs_and_attention_metadata.md)

回答：

```text
token ids、positions、slot mapping、block tables 如何准备？
attention backend 需要什么 metadata？
mm / spec / prefix cache 信息如何进入？
```

### 07. forward 与 logits

- [模型 forward 和 logits 在哪里发生？](07_model_forward_and_logits.md)

回答：

```text
真正的 forward 在哪里？
logits 如何产生？
pooling 输出在哪里产生？
```

### 08. sampling 与输出

- [sample_tokens() 如何生成 ModelRunnerOutput？](08_sampling_and_model_runner_output.md)

回答：

```text
Sampler 如何工作？
grammar / structured output 如何影响采样？
ModelRunnerOutput 包含什么？
```

### 09. KV Cache 交互

- [Worker / ModelRunner 如何使用 KV Cache？](09_worker_kv_cache_interaction.md)

回答：

```text
KV block 如何映射到请求？
prefix cache / external KV / lookahead 如何影响执行？
KV cache 和 batch 执行如何配合？
```

### 10. 生命周期与控制

- [Executor / Worker / ModelRunner 的生命周期如何管理？](10_executor_worker_lifecycle.md)

回答：

```text
load_model、init_kv_cache、profile、sleep、wake_up、shutdown 如何工作？
异常如何传播？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
executor_worker_model_runner_overview.md
  → 01_executor_role.md
  → 02_worker_role.md
  → 03_model_runner_role.md
```

### 3.2 按执行链路完整阅读

```text
executor_worker_model_runner_overview.md
  → 01_executor_role.md
  → 04_execute_model_flow.md
  → 05_input_batch_and_state_update.md
  → 06_prepare_inputs_and_attention_metadata.md
  → 07_model_forward_and_logits.md
  → 08_sampling_and_model_runner_output.md
```

### 3.3 和 Scheduler 联动阅读

```text
../scheduler/03_running_decode_prefill.md
  → ../scheduler/04_waiting_to_running.md
  → 04_execute_model_flow.md
  → 09_worker_kv_cache_interaction.md
  → ../scheduler/08_update_after_worker_output.md
```

---

## 4. 文档定位

```text
README.md：
  当前目录索引和阅读路线。

01-10：
  按问题拆开的专题文档，适合逐段精读执行层源码。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
Executor 负责分发执行，Worker 负责设备侧承载，ModelRunner 负责把调度计划变成模型 forward / sampling。
```
