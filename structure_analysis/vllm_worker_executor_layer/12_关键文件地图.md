# 12. 关键文件地图

本文给出 vLLM V1 Worker / Executor 执行层的关键源码地图和建议阅读顺序。

源码根目录：

```text
D:/lzy/project/kv_pool/code/vllm
```

## 1. 最推荐阅读顺序

### 第一轮：理解主链路

1. `code/vllm/vllm/v1/engine/core.py:479`
   - 看 `EngineCore.step()`，理解 scheduler -> executor -> scheduler update。

2. `code/vllm/vllm/v1/executor/abstract.py:47`
   - 看 executor 后端选择。

3. `code/vllm/vllm/v1/executor/abstract.py:221`
   - 看 `execute_model()` 抽象默认语义。

4. `code/vllm/vllm/v1/executor/uniproc_executor.py:45`
   - 从最简单 executor 理解 WorkerWrapper 调用。

5. `code/vllm/vllm/v1/worker/worker_base.py:187`
   - 看 `WorkerWrapperBase` 如何创建真实 worker。

6. `code/vllm/vllm/v1/worker/gpu_worker.py:807`
   - 看 worker 如何包裹 PP 通信并调用 model runner。

7. `code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`
   - 看 V1 ModelRunner execute。

8. `code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`
   - 看 V1 ModelRunner sample。

### 第二轮：理解多进程与分布式

1. `code/vllm/vllm/v1/executor/multiproc_executor.py:110`
   - Multiproc 初始化。

2. `code/vllm/vllm/v1/executor/multiproc_executor.py:340`
   - Multiproc collective RPC。

3. `code/vllm/vllm/v1/executor/multiproc_executor.py:593`
   - WorkerProc 生命周期。

4. `code/vllm/vllm/v1/worker/gpu_worker.py:1164`
   - worker 分布式初始化。

5. `code/vllm/vllm/distributed/parallel_state.py:1516`
   - default process group 初始化。

6. `code/vllm/vllm/distributed/parallel_state.py:1674`
   - model parallel groups。

7. `code/vllm/vllm/distributed/parallel_state.py:351`
   - GroupCoordinator。

### 第三轮：理解 Ray

1. `code/vllm/vllm/v1/executor/ray_executor.py:64`
   - 旧 Ray executor。

2. `code/vllm/vllm/v1/executor/ray_executor.py:536`
   - Ray compiled DAG。

3. `code/vllm/vllm/v1/executor/ray_utils.py:123`
   - `execute_model_ray()`。

4. `code/vllm/vllm/v1/executor/ray_executor_v2.py:205`
   - RayExecutorV2。

5. `code/vllm/vllm/v1/executor/ray_executor_v2.py:249`
   - Ray v2 初始化。

## 2. Engine / Core 相关

### `vllm/v1/engine/core.py`

关键内容：

- EngineCore 初始化 executor、KV cache、scheduler。
- 请求预处理。
- `step()` 主循环。
- EngineCoreProc busy loop。

关键位置：

- `code/vllm/vllm/v1/engine/core.py:96`：EngineCore 初始化。
- `code/vllm/vllm/v1/engine/core.py:239`：KV cache 初始化链路。
- `code/vllm/vllm/v1/engine/core.py:378`：add_request。
- `code/vllm/vllm/v1/engine/core.py:479`：step 主链路。
- `code/vllm/vllm/v1/engine/core.py:519`：batch queue 变体。
- `code/vllm/vllm/v1/engine/core.py:862`：preprocess_add_request。
- `code/vllm/vllm/v1/engine/core.py:1257`：EngineCoreProc busy loop。

### `vllm/v1/engine/core_client.py`

关键内容：

- InprocClient。
- MP/ZMQ client。
- 同步/异步输出队列。

关键位置：

- `code/vllm/vllm/v1/engine/core_client.py:289`：Inproc get_output。
- `code/vllm/vllm/v1/engine/core_client.py:297`：Inproc add_request。
- `code/vllm/vllm/v1/engine/core_client.py:849`：SyncMPClient get_output。
- `code/vllm/vllm/v1/engine/core_client.py:861`：SyncMPClient send input。
- `code/vllm/vllm/v1/engine/core_client.py:1053`：Async get_output。

### `vllm/v1/engine/llm_engine.py`

关键内容：

- 同步 LLMEngine。
- 外部 request 到 EngineCoreRequest。
- 输出处理。

关键位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:250`：process_inputs。
- `code/vllm/vllm/v1/engine/llm_engine.py:272`：add_request。
- `code/vllm/vllm/v1/engine/llm_engine.py:302`：step get_output。

## 3. Scheduler 输出相关

### `vllm/v1/core/sched/output.py`

关键数据结构：

- `NewRequestData`
- `CachedRequestData`
- `SchedulerOutput`

关键位置：

- `code/vllm/vllm/v1/core/sched/output.py:30`：NewRequestData。
- `code/vllm/vllm/v1/core/sched/output.py:111`：CachedRequestData。
- `code/vllm/vllm/v1/core/sched/output.py:180`：SchedulerOutput。

### `vllm/v1/core/sched/scheduler.py`

关键内容：

- `schedule()` 构造 SchedulerOutput。
- `update_from_output()` 用 ModelRunnerOutput 更新 Request 状态。

关键位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:387`：schedule。
- `code/vllm/vllm/v1/core/sched/scheduler.py:1057`：构造 SchedulerOutput。
- `code/vllm/vllm/v1/core/sched/scheduler.py:1463`：update_from_output。
- `code/vllm/vllm/v1/core/sched/scheduler.py:1688`：构造 EngineCoreOutput。
- `code/vllm/vllm/v1/core/sched/scheduler.py:1769`：构造 EngineCoreOutputs。

## 4. Executor 相关

### `vllm/v1/executor/abstract.py`

关键内容：

- Executor 抽象。
- 后端选择。
- collective_rpc 抽象。
- execute_model / sample_tokens 默认实现。

关键位置：

- `code/vllm/vllm/v1/executor/abstract.py:47`：get_class。
- `code/vllm/vllm/v1/executor/abstract.py:94`：Executor 初始化。
- `code/vllm/vllm/v1/executor/abstract.py:121`：initialize_from_config。
- `code/vllm/vllm/v1/executor/abstract.py:152`：collective_rpc。
- `code/vllm/vllm/v1/executor/abstract.py:221`：execute_model。
- `code/vllm/vllm/v1/executor/abstract.py:241`：sample_tokens。

### `vllm/v1/executor/uniproc_executor.py`

关键内容：

- 单进程 executor。
- 直接函数调用。
- AsyncOutputFuture。
- external launcher。

关键位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:26`：AsyncOutputFuture。
- `code/vllm/vllm/v1/executor/uniproc_executor.py:45`：UniProcExecutor。
- `code/vllm/vllm/v1/executor/uniproc_executor.py:79`：collective_rpc。
- `code/vllm/vllm/v1/executor/uniproc_executor.py:108`：execute_model。
- `code/vllm/vllm/v1/executor/uniproc_executor.py:150`：ExecutorWithExternalLauncher。

### `vllm/v1/executor/multiproc_executor.py`

关键内容：

- 本地多进程 executor。
- MessageQueue 控制面。
- WorkerProc。
- FutureWrapper。
- output rank。
- failure monitor。

关键位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:70`：FutureWrapper。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:110`：MultiprocExecutor。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:151`：broadcast MQ。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:268`：monitor。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:307`：execute_model。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:319`：sample_tokens。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:340`：collective_rpc。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:404`：shutdown。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:593`：WorkerProc。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:806`：worker busy loop。
- `code/vllm/vllm/v1/executor/multiproc_executor.py:926`：async output copy。

### `vllm/v1/executor/ray_executor.py`

关键内容：

- 旧 Ray executor。
- Ray actor 创建。
- Ray remote control plane。
- Ray compiled DAG。

关键位置：

- `code/vllm/vllm/v1/executor/ray_executor.py:64`：RayDistributedExecutor。
- `code/vllm/vllm/v1/executor/ray_executor.py:143`：初始化 Ray workers。
- `code/vllm/vllm/v1/executor/ray_executor.py:399`：execute_model。
- `code/vllm/vllm/v1/executor/ray_executor.py:418`：sample_tokens。
- `code/vllm/vllm/v1/executor/ray_executor.py:443`：_execute_dag。
- `code/vllm/vllm/v1/executor/ray_executor.py:479`：collective_rpc。
- `code/vllm/vllm/v1/executor/ray_executor.py:536`：compiled Ray DAG。

### `vllm/v1/executor/ray_executor_v2.py`

关键内容：

- 新 Ray executor。
- 继承 MultiprocExecutor。
- Ray actor 作为 WorkerProc 容器。
- MQ 控制面。
- actor failure monitor。

关键位置：

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:48`：RayWorkerHandle。
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:75`：RayWorkerProc。
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:205`：RayExecutorV2。
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:249`：初始化。
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:429`：failure monitor。
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:498`：shutdown。

### `vllm/v1/executor/ray_utils.py`

关键内容：

- Ray worker wrapper utilities。
- `execute_model_ray()`。

关键位置：

- `code/vllm/vllm/v1/executor/ray_utils.py:123`：execute_model_ray。

## 5. Worker 相关

### `vllm/v1/worker/worker_base.py`

关键内容：

- `WorkerBase` 抽象接口。
- `WorkerWrapperBase`。

关键位置：

- `code/vllm/vllm/v1/worker/worker_base.py:39`：WorkerBase。
- `code/vllm/vllm/v1/worker/worker_base.py:98`：get_kv_cache_spec。
- `code/vllm/vllm/v1/worker/worker_base.py:114`：init_device。
- `code/vllm/vllm/v1/worker/worker_base.py:138`：load_model。
- `code/vllm/vllm/v1/worker/worker_base.py:142`：execute_model。
- `code/vllm/vllm/v1/worker/worker_base.py:153`：sample_tokens。
- `code/vllm/vllm/v1/worker/worker_base.py:187`：WorkerWrapperBase。
- `code/vllm/vllm/v1/worker/worker_base.py:229`：init_worker。
- `code/vllm/vllm/v1/worker/worker_base.py:315`：initialize_from_config。
- `code/vllm/vllm/v1/worker/worker_base.py:340`：execute_model wrapper。

### `vllm/v1/worker/gpu_worker.py`

关键内容：

- GPU worker 主实现。
- device/distributed/model runner/KV cache/execution/profiler/shutdown。

关键位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py:117`：Worker 类。
- `code/vllm/vllm/v1/worker/gpu_worker.py:249`：init_device。
- `code/vllm/vllm/v1/worker/gpu_worker.py:349`：load_model。
- `code/vllm/vllm/v1/worker/gpu_worker.py:371`：determine_available_memory。
- `code/vllm/vllm/v1/worker/gpu_worker.py:526`：KV connector handshake metadata。
- `code/vllm/vllm/v1/worker/gpu_worker.py:547`：get_kv_cache_spec。
- `code/vllm/vllm/v1/worker/gpu_worker.py:562`：initialize_from_config。
- `code/vllm/vllm/v1/worker/gpu_worker.py:591`：compile_or_warm_up_model。
- `code/vllm/vllm/v1/worker/gpu_worker.py:801`：sample_tokens。
- `code/vllm/vllm/v1/worker/gpu_worker.py:807`：execute_model。
- `code/vllm/vllm/v1/worker/gpu_worker.py:901`：profile。
- `code/vllm/vllm/v1/worker/gpu_worker.py:958`：LoRA API。
- `code/vllm/vllm/v1/worker/gpu_worker.py:1141`：shutdown。
- `code/vllm/vllm/v1/worker/gpu_worker.py:1164`：init_worker_distributed_environment。

### CPU / XPU

- `code/vllm/vllm/v1/worker/cpu_worker.py:33`：CPUWorker。
- `code/vllm/vllm/v1/worker/cpu_worker.py:107`：CPU init_device。
- `code/vllm/vllm/v1/worker/cpu_worker.py:180`：CPU memory profiling。
- `code/vllm/vllm/v1/worker/xpu_worker.py:24`：XPUWorker。
- `code/vllm/vllm/v1/worker/xpu_worker.py:42`：XPU init_device。

## 6. ModelRunner 相关

### V1 GPUModelRunner

文件：`code/vllm/vllm/v1/worker/gpu_model_runner.py`

关键位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:263`：AsyncGPUModelRunnerOutput。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1090`：KV zero metadata。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`：execute_model。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`：sample_tokens。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5142`：load_model。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6227`：profile_run。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6583`：capture_model。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7220`：KV cache tensors。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`：initialize_kv_cache。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7459`：get_kv_cache_spec。

### V2 GPUModelRunner

文件：`code/vllm/vllm/v1/worker/gpu/model_runner.py`

关键位置：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:120`：类定义。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:274`：load_model。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:403`：get_kv_cache_spec。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:406`：initialize_kv_cache。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:653`：profile_run。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:690`：capture_model。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`：execute_model。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`：sample_tokens。

## 7. 横切能力文件

### LoRA

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31`：load_lora_model。
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`：set_active_loras。
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:274`：add_lora。

### KV connector

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:33`：mixin。
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:36`：no forward。
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:50`：maybe_get_kv_connector_output。
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:78`：connector context。
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:115`：uniform KV cache。

### 输出结构

- `code/vllm/vllm/v1/outputs.py:233`：ModelRunnerOutput。

## 8. Distributed 相关

### `vllm/distributed/parallel_state.py`

关键位置：

- `code/vllm/vllm/distributed/parallel_state.py:351`：GroupCoordinator。
- `code/vllm/vllm/distributed/parallel_state.py:607`：all_reduce。
- `code/vllm/vllm/distributed/parallel_state.py:1434`：split_group。
- `code/vllm/vllm/distributed/parallel_state.py:1516`：init_distributed_environment。
- `code/vllm/vllm/distributed/parallel_state.py:1674`：initialize_model_parallel。
- `code/vllm/vllm/distributed/parallel_state.py:1852`：EP group。

### Device communicators

- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:26`：CudaCommunicator。
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:254`：allreduce dispatch。
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:60`：PyNcclCommunicator。

## 9. 按问题定位文件

### 想看 executor 怎么选？

- `code/vllm/vllm/v1/executor/abstract.py:47`

### 想看 EngineCore 每步怎么跑？

- `code/vllm/vllm/v1/engine/core.py:479`

### 想看 SchedulerOutput 里面有什么？

- `code/vllm/vllm/v1/core/sched/output.py:180`

### 想看多进程怎么广播 RPC？

- `code/vllm/vllm/v1/executor/multiproc_executor.py:340`

### 想看 worker 子进程 busy loop？

- `code/vllm/vllm/v1/executor/multiproc_executor.py:806`

### 想看 GPU worker 初始化？

- `code/vllm/vllm/v1/worker/gpu_worker.py:249`

### 想看模型 forward？

- V1：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`
- V2：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`

### 想看 sampling？

- V1：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`
- V2：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`

### 想看 KV cache 初始化？

- Worker：`code/vllm/vllm/v1/worker/gpu_worker.py:562`
- V1 ModelRunner：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`
- V2 ModelRunner：`code/vllm/vllm/v1/worker/gpu/model_runner.py:406`

### 想看 PP send/recv？

- `code/vllm/vllm/v1/worker/gpu_worker.py:807`

### 想看 Ray DAG？

- `code/vllm/vllm/v1/executor/ray_executor.py:536`
- `code/vllm/vllm/v1/executor/ray_utils.py:123`

### 想看 Ray v2？

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:205`

## 10. 总结

最小理解闭环是：

```text
core.py:479
  -> abstract.py:221
  -> multiproc_executor.py:340 / uniproc_executor.py:79
  -> worker_base.py:340
  -> gpu_worker.py:807
  -> gpu_model_runner.py:4043
  -> gpu_model_runner.py:4422
  -> scheduler.py:1463
```

掌握这条链路后，再展开 Ray、KV cache、distributed groups、LoRA、KV transfer 等横切能力会更容易。
