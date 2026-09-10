# 01. 执行层总览

Worker / Executor 执行层是 vLLM V1 中连接“调度结果”和“模型真实执行”的中间层。

它不直接等同于模型层，也不等同于 scheduler。更准确地说：

- `EngineCore` 负责请求生命周期、调度和输出更新。
- `Executor` 负责把 EngineCore 的执行请求分发给一个或多个 worker。
- `WorkerWrapperBase` 是 executor 看到的 worker 外壳。
- 真实 `Worker` 负责设备初始化、模型加载、KV cache、执行模型。
- `ModelRunner` 负责真正的 forward、sampling、attention metadata、KV connector、LoRA 等。

## 1. 执行层在 V1 架构中的位置

核心链路：

```text
入口/API/离线 LLM
  -> LLMEngine / AsyncLLM
  -> EngineCoreClient
  -> EngineCore
  -> Scheduler
  -> Executor
  -> WorkerWrapperBase
  -> Worker
  -> ModelRunner
  -> Model / Attention / CUDA kernels
```

其中 Worker / Executor 执行层覆盖：

```text
Executor
  -> UniProcExecutor
  -> MultiprocExecutor
  -> RayDistributedExecutor
  -> RayExecutorV2

Worker wrapper
  -> WorkerWrapperBase

Worker
  -> GPU worker: vllm.v1.worker.gpu_worker.Worker
  -> CPUWorker
  -> XPUWorker

Model runner
  -> GPUModelRunner V1
  -> GPUModelRunner V2
  -> CPUModelRunner
  -> XPUModelRunner
```

关键入口文件：

- `code/vllm/vllm/v1/engine/core.py:96`
- `code/vllm/vllm/v1/engine/core.py:479`
- `code/vllm/vllm/v1/executor/abstract.py:47`
- `code/vllm/vllm/v1/worker/worker_base.py:39`
- `code/vllm/vllm/v1/worker/gpu_worker.py:117`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`

## 2. 三个最重要的边界

### 2.1 EngineCore 与 Executor 的边界

`EngineCore` 不关心 worker 是单进程、多进程还是 Ray。它只依赖 `Executor` 的统一接口：

- `get_kv_cache_specs()`
- `determine_available_memory()`
- `initialize_from_config()`
- `execute_model()`
- `sample_tokens()`
- `check_health()`
- `shutdown()`

典型执行点在 `EngineCore.step()`：

```text
scheduler_output = scheduler.schedule(...)
future = model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)
scheduler.update_from_output(scheduler_output, model_output)
```

源码位置：

- `code/vllm/vllm/v1/engine/core.py:479`
- `code/vllm/vllm/v1/engine/core.py:490`
- `code/vllm/vllm/v1/engine/core.py:491`
- `code/vllm/vllm/v1/engine/core.py:492`
- `code/vllm/vllm/v1/engine/core.py:497`
- `code/vllm/vllm/v1/engine/core.py:498`
- `code/vllm/vllm/v1/engine/core.py:504`

### 2.2 Executor 与 Worker 的边界

Executor 本质上不做模型 forward，它负责：

- 创建 worker。
- 初始化 worker。
- 分发方法调用。
- 收集结果。
- 管理 worker 失败与关闭。

Worker 才负责：

- 设备初始化。
- 分布式初始化。
- 模型加载。
- KV cache 初始化。
- 执行 forward。
- 执行 sampling。

典型边界：

- `Executor.execute_model()` 调用 worker 的 `execute_model()`。
- `Executor.sample_tokens()` 调用 worker 的 `sample_tokens()`。

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py:221`
- `code/vllm/vllm/v1/executor/abstract.py:241`
- `code/vllm/vllm/v1/worker/worker_base.py:340`
- `code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `code/vllm/vllm/v1/worker/gpu_worker.py:801`

### 2.3 Worker 与 ModelRunner 的边界

Worker 是“设备与生命周期”的主体，ModelRunner 是“模型执行”的主体。

Worker 负责：

- 建立 device。
- 建立 distributed environment。
- 创建 model runner。
- 包装 memory pool。
- 处理 PP send/recv。
- 处理 profiler、shutdown、LoRA API 转发。

ModelRunner 负责：

- 维护 batch/request 状态。
- 准备 input ids、positions、slot mapping。
- 构建 attention metadata。
- 执行模型 forward。
- 处理 logits、sampling、logprobs。
- 处理 KV connector、spec decode、LoRA active mapping。

源码位置：

- Worker 创建 model runner：`code/vllm/vllm/v1/worker/gpu_worker.py:326`
- Worker execute：`code/vllm/vllm/v1/worker/gpu_worker.py:807`
- ModelRunner execute V1：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`
- ModelRunner sample V1：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`
- ModelRunner execute V2：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`
- ModelRunner sample V2：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`

## 3. Executor 后端总览

| Executor | 资源模型 | 控制面 | 执行面 | 用途 |
|---|---|---|---|---|
| `UniProcExecutor` | 单进程单 worker | 直接函数调用 | 本进程执行 | 最简单基线 |
| `MultiprocExecutor` | 本地多进程 | MessageQueue 广播 + response MQ | 所有 worker 执行，指定 rank 回包 | 本地多 GPU |
| `RayDistributedExecutor` | Ray actor | Ray remote call | Ray compiled DAG | 旧 Ray 后端 |
| `RayExecutorV2` | Ray actor | 复用 Multiproc MQ | Ray 调度资源，vLLM 控制面 | 新 Ray 后端 |

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:45`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:110`
- `code/vllm/vllm/v1/executor/ray_executor.py:64`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:205`

## 4. Worker 类型总览

vLLM V1 中容易混淆的一点是：GPU worker 类名并不叫 `GPUWorker`，而是：

```python
vllm.v1.worker.gpu_worker.Worker
```

CPU/XPU worker 则继承它并覆盖平台相关逻辑：

- GPU/CUDA/ROCm：`Worker`
- CPU：`CPUWorker(Worker)`
- XPU：`XPUWorker(Worker)`

源码位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py:117`
- `code/vllm/vllm/v1/worker/cpu_worker.py:33`
- `code/vllm/vllm/v1/worker/xpu_worker.py:24`
- CUDA 平台默认 worker：`code/vllm/vllm/platforms/cuda.py:269`
- CPU 平台默认 worker：`code/vllm/vllm/platforms/cpu.py:153`
- XPU 平台默认 worker：`code/vllm/vllm/platforms/xpu.py:230`

## 5. 两阶段执行：execute_model 与 sample_tokens

V1 执行层一个关键设计是：`execute_model()` 不一定直接返回最终 token。

可能情况：

1. 非 generation / pooling 模型：`execute_model()` 可直接返回 `ModelRunnerOutput`。
2. generation 模型：`execute_model()` 做 forward 和 logits 准备，返回 `None`。
3. 随后 `sample_tokens(grammar_output)` 执行结构化输出约束、采样、logprobs、bookkeeping，返回 `ModelRunnerOutput`。
4. PP 非最后 stage：`execute_model()` 返回 `IntermediateTensors`，worker 发送给下一 PP stage 后返回 `None`。
5. async scheduling：返回可能被包装为 `AsyncModelRunnerOutput`，稍后再取 CPU 可见输出。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4386`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`
- `code/vllm/vllm/v1/worker/gpu_worker.py:871`
- `code/vllm/vllm/v1/worker/gpu_worker.py:882`

## 6. 执行层处理的关键能力

Worker / Executor 执行层不只是 forward 调用，还覆盖大量横切能力：

- KV cache spec 收集。
- KV cache 显存/内存容量估计。
- KV cache tensor 分配和绑定。
- Prefix cache / block table 的 worker 侧更新。
- Pipeline parallel intermediate tensors 传递。
- Tensor parallel / pipeline parallel / data parallel / expert parallel 分组初始化。
- NCCL / Gloo / device communicator 初始化。
- LoRA 管理与 active adapter 设置。
- KV transfer connector。
- Speculative decoding。
- Structured output grammar bitmask。
- CUDA graph capture。
- Profiler。
- Weight transfer。
- Worker failure handling。
- Async output copy。

这些能力大多分布在：

- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `code/vllm/vllm/distributed/parallel_state.py`
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py`

## 7. 一句话总结

vLLM V1 Worker / Executor 执行层的核心职责是：

> 把 SchedulerOutput 以统一接口分发到正确的 worker/rank，在 worker 上完成设备、KV cache、模型 forward、采样与通信，再把 ModelRunnerOutput 聚合回 EngineCore。
