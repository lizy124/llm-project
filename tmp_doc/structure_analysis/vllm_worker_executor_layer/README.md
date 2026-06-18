# vLLM Worker / Executor 执行层梳理

本文档集梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM V1 的 Worker / Executor 执行层。

核心结论：

```text
EngineCore 负责调度与编排
  ↓
Executor 负责 worker 生命周期、RPC/IPC、fan-out/fan-in
  ↓
WorkerWrapperBase 是 executor 看到的 worker 壳
  ↓
Worker / CPUWorker / XPUWorker 负责设备、模型、KV cache 生命周期
  ↓
ModelRunner 负责真正的 forward、sampling、KV/LoRA/spec decode 等执行细节
```

## 文档目录

1. `01_执行层总览.md`
   - Worker / Executor 执行层在 vLLM V1 中的位置、边界、核心对象。

2. `02_Executor抽象与后端选择.md`
   - `Executor` 抽象接口、后端选择、公共 RPC 语义、初始化职责。

3. `03_UniProcExecutor单进程执行器.md`
   - 单进程执行器的初始化、直接调用、Future 与 external launcher 变体。

4. `04_MultiprocExecutor多进程执行器.md`
   - 本地多进程执行器、`WorkerProc`、MessageQueue 控制面、输出 rank、失败处理与关闭。

5. `05_Ray执行器.md`
   - 旧 Ray executor、Ray compiled DAG、RayExecutorV2、Ray actor 与 Multiproc MQ 的差异。

6. `06_WorkerBase与WorkerWrapper.md`
   - Worker 抽象接口、Worker wrapper 的延迟构造、环境变量、插件与多模态缓存。

7. `07_GPU_CPU_XPU_Worker生命周期.md`
   - GPU/CPU/XPU worker 的设备初始化、模型加载、显存/内存 profiling、KV cache 初始化、warmup 与 shutdown。

8. `08_ModelRunner执行链路.md`
   - `execute_model()` 与 `sample_tokens()` 的两阶段执行、V1/V2 GPUModelRunner、PP intermediate tensors、async output。

9. `09_KVCache_LoRA_KVTransfer_Profiler.md`
   - KV cache spec/初始化、LoRA、KV transfer connector、profiler、weight transfer 等横切能力。

10. `10_分布式并行与通信.md`
    - rank/local_rank、DP/TP/PP/EP 分组、NCCL/device communicator、PP 通信、async scheduling。

11. `11_端到端调用链路.md`
    - 从请求进入 EngineCore，到 SchedulerOutput，经 Executor/Worker/ModelRunner，再返回 EngineCoreOutputs 的完整链路。

12. `12_关键文件地图.md`
    - 关键源码文件、类、方法、建议阅读顺序。

## 推荐阅读顺序

如果只想快速理解主链路，按以下顺序读：

1. `01_执行层总览.md`
2. `11_端到端调用链路.md`
3. `02_Executor抽象与后端选择.md`
4. `04_MultiprocExecutor多进程执行器.md`
5. `06_WorkerBase与WorkerWrapper.md`
6. `07_GPU_CPU_XPU_Worker生命周期.md`
7. `08_ModelRunner执行链路.md`

如果重点研究分布式执行，继续读：

1. `05_Ray执行器.md`
2. `10_分布式并行与通信.md`
3. `09_KVCache_LoRA_KVTransfer_Profiler.md`

## 最短主链路

```text
LLMEngine / AsyncLLM
  -> EngineCoreClient
  -> EngineCore.preprocess_add_request
  -> Scheduler.add_request

EngineCore.step
  -> Scheduler.schedule
  -> SchedulerOutput
  -> Executor.execute_model(scheduler_output)
      -> UniProc: direct call
      -> Multiproc: broadcast MQ fan-out
      -> Ray old: Ray compiled DAG
      -> Ray v2: Ray actor + Multiproc MQ
  -> WorkerWrapperBase.execute_model
  -> Worker.execute_model
  -> ModelRunner.execute_model
  -> Executor.sample_tokens(grammar_output) when execute_model returns None
  -> Worker.sample_tokens
  -> ModelRunner.sample_tokens
  -> ModelRunnerOutput
  -> Scheduler.update_from_output
  -> EngineCoreOutputs
  -> OutputProcessor
```

## 源码根目录

本文档中的源码引用均基于：

```text
D:/lzy/project/kv_pool/code/vllm
```

示例：`code/vllm/vllm/v1/executor/abstract.py:47`。
