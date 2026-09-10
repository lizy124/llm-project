# 07. 旧架构映射与阅读指南

## 1. 当前代码库的分布式主线

当前代码库中，V1 分布式主线主要由以下几部分组成：

```text
vllm/config/parallel.py
  -> vllm/v1/executor/*
  -> vllm/v1/worker/*
  -> vllm/distributed/parallel_state.py
  -> vllm/distributed/device_communicators/*
  -> vllm/distributed/kv_transfer / ec_transfer / weight_transfer / eplb / elastic_ep
```

其中：

- `vllm/distributed` 是 V1 和部分非 V1 共享的底层分布式运行时。
- `vllm/v1/executor` 是 V1 的 worker 编排和 RPC 控制面。
- `vllm/v1/worker` 是 V1 的实际执行面。

## 2. V1 与非 V1 的关系

### 2.1 共享层

这些模块基本是共享底层：

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/*`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/kv_transfer/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/ec_transfer/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/weight_transfer/**`

无论 V1 还是非 V1，只要需要 torch distributed process group、NCCL/Gloo 通信、KV transfer 等，都会依赖这些底层能力。

### 2.2 V1 专属层

这些模块是 V1 主线：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py`

### 2.3 兼容/过渡点

`Executor` 抽象底部保留了 backwards compatibility alias：

- `UniProcExecutor = _UniProcExecutor`
- `ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:371`。

`RayDistributedExecutor` 与 `RayExecutorV2` 同时存在，说明 Ray V1 和 Ray V2 仍在共存。

## 3. 老概念到 V1 的映射

| 老/通用概念 | V1 对应 |
|---|---|
| executor backend | `Executor.get_class()` |
| single process executor | `UniProcExecutor` |
| multiprocessing executor | `MultiprocExecutor` |
| Ray actor executor | `RayDistributedExecutor` / `RayExecutorV2` |
| worker wrapper | `WorkerWrapperBase` |
| GPU worker | `vllm.v1.worker.gpu_worker.Worker` |
| process group manager | `parallel_state.GroupCoordinator` |
| tensor parallel group | `get_tp_group()` |
| pipeline parallel group | `get_pp_group()` |
| data parallel group | `get_dp_group()` |
| expert parallel group | `get_ep_group()` |
| pipeline tensor p2p | `GroupCoordinator.isend_tensor_dict()` / `irecv_tensor_dict()` |
| DP synchronization | `vllm/v1/worker/dp_utils.py` |
| PP sampled token sync | `PPHandler` |
| MoE all2all | `GroupCoordinator.dispatch()` / `combine()` |
| external KV cache | KV Connector / KVTransferConfig |
| encoder cache transfer | EC Connector / ECTransferConfig |
| hot weight update | WeightTransferEngine |

## 4. 推荐阅读顺序

### 第 1 阶段：把整体主链路串起来

1. `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py`
   - 看 `ParallelConfig` 字段和 `_validate_parallel_config()`。
2. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py`
   - 看 `Executor.get_class()`、`collective_rpc()`、`execute_model()`。
3. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py`
   - 看 `WorkerBase` 与 `WorkerWrapperBase`。
4. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py`
   - 看 `init_device()`、`initialize_from_config()`、`execute_model()`。

读完后应理解：配置如何选 executor，executor 如何创建 worker，worker 如何初始化设备和执行模型。

### 第 2 阶段：理解 process group

5. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
   - 看 `GroupCoordinator`。
   - 看 `init_distributed_environment()`。
   - 看 `initialize_model_parallel()`。
6. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/base_device_communicator.py`
7. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/cuda_communicator.py`
8. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/communication_op.py`

读完后应理解：WORLD/TP/PP/DP/DCP/PCP/EP/EPLB groups 如何创建，collective 由谁执行。

### 第 3 阶段：理解 executor 后端差异

9. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py`
10. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py`
11. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py`
12. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py`

读完后应理解：UniProc、MP、Ray V1、Ray V2 的控制面差异。

### 第 4 阶段：理解运行时通信

13. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:807`
14. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py`
15. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py`
16. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/all2all.py`

读完后应理解：PP/DP/EP 运行时怎么通信。

### 第 5 阶段：理解 transfer 扩展

17. `D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py`
18. `D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py`
19. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/kv_transfer/**`
20. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/ec_transfer/**`
21. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/weight_transfer/**`
22. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/elastic_ep/**`
23. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/eplb/**`

读完后应理解：跨实例 KV/EC/weight/EP 动态扩展如何接入。

## 5. 按问题定位代码

### 5.1 “为什么选择了某个 executor backend？”

看：

- `ParallelConfig.distributed_executor_backend`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:240`
- `Executor.get_class()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:48`
- `ParallelConfig.use_ray`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:941`

### 5.2 “worker 为什么没起来？”

看：

- Multiproc：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:110`
- Ray V2：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:249`
- Worker wrapper：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:230`
- GPU Worker init：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:249`

### 5.3 “torch.distributed 初始化在哪里？”

看：

- `Worker.init_device()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:294`
- `init_distributed_environment()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1516`

### 5.4 “TP/PP/DP group 怎么切？”

看：

- `initialize_model_parallel()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1674`
- rank layout：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1749`

### 5.5 “PP 为什么卡住？”

看：

- `Worker.execute_model()` 中 recv/send：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:853`
- `PPHandler`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:51`
- `GroupCoordinator.isend_tensor_dict()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:964`
- `GroupCoordinator.irecv_tensor_dict()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1059`

### 5.6 “DP 为什么卡住？”

看：

- `coordinate_batch_across_dp()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:164`
- `_run_ar()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:36`
- `disable_nccl_for_dp_synchronization`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:224`

### 5.7 “KV transfer 为什么没生效？”

看：

- `KVTransferConfig`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:23`
- Worker 初始化：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:575`
- handshake metadata：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:526`
- scheduler connector：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:123`

### 5.8 “MoE/EP 通信在哪里？”

看：

- `enable_expert_parallel`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:162`
- `all2all_backend`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:185`
- EP group：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1850`
- `GroupCoordinator.dispatch()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1206`
- `GroupCoordinator.combine()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1228`

## 6. 最短掌握路径

如果只想最快理解 vLLM 分布式通信主链路，按以下 8 个点阅读：

1. `ParallelConfig` 字段。
2. `Executor.get_class()`。
3. `MultiprocExecutor._init_executor()`。
4. `WorkerWrapperBase.init_worker()`。
5. `Worker.init_device()`。
6. `init_distributed_environment()`。
7. `initialize_model_parallel()`。
8. `Worker.execute_model()`。

这 8 个点串起来，就是 vLLM 分布式从配置到运行时通信的主干。

## 7. 一句话总结

V1 分布式通信层的核心不是某个单文件，而是一条链：`ParallelConfig` 编译拓扑，`Executor` 编排 worker，`Worker` 初始化设备和执行模型，`parallel_state` 构建进程组，`GroupCoordinator/device communicator` 执行数据面通信，transfer/EP/EPLB 在这条链上扩展专用通信能力。
