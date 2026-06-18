# 10. 分布式并行与通信

Worker / Executor 执行层涉及两类通信：

1. 控制面通信：Executor 调 worker 方法。
   - UniProc：直接函数调用。
   - Multiproc：MessageQueue。
   - Ray old：Ray remote / Ray DAG。
   - Ray v2：Ray actor + MessageQueue。

2. 模型并行通信：worker/model runner 内部的 tensor / object 通信。
   - torch distributed process group。
   - NCCL/Gloo。
   - vLLM `GroupCoordinator`。
   - device communicator。
   - TP/PP/DP/EP groups。

本文重点梳理第二类通信，同时说明它与 executor 控制面的关系。

## 1. rank 与 local_rank

### 1.1 WorkerBase 中的 rank

WorkerBase 保存：

- `rank`：全局 distributed rank。
- `local_rank`：本地设备 index。
- `parallel_config.rank`：会写成当前 worker rank。

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:45`
- `code/vllm/vllm/v1/worker/worker_base.py:58`
- `code/vllm/vllm/v1/worker/worker_base.py:81`

### 1.2 Multiproc rank

Multiproc 中：

```text
global_rank = local_world_size * node_rank_within_dp + local_rank
is_driver_worker = rank % tensor_parallel_size == 0
```

源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:164`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:176`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:177`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:265`

### 1.3 GPU Worker 的 DP local_rank 修正

单节点 DP 且不是 Ray/external launcher 时，会把 DP local rank 纳入 local_rank：

```text
local_rank += dp_local_rank * (PP * TP)
```

目的：同一节点多个 DP rank 映射到不同 GPU 段。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:251`
- `code/vllm/vllm/v1/worker/gpu_worker.py:261`
- `code/vllm/vllm/v1/worker/gpu_worker.py:266`
- `code/vllm/vllm/v1/worker/gpu_worker.py:272`
- `code/vllm/vllm/v1/worker/gpu_worker.py:285`

## 2. world_size 与并行维度

`ParallelConfig.world_size` 基本为：

```text
world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size
```

external launcher 模式下还会乘以 data parallel size。

源码：

- `code/vllm/vllm/config/parallel.py:782`
- `code/vllm/vllm/config/parallel.py:784`
- `code/vllm/vllm/config/parallel.py:790`

常见并行维度：

- TP：Tensor Parallel。
- PP：Pipeline Parallel。
- DP：Data Parallel。
- PCP：Prefill Context Parallel。
- DCP：Decode Context Parallel。
- EP：Expert Parallel。
- EPLB：Expert Parallel Load Balancing。

## 3. distributed environment 初始化

GPU worker 在 device 初始化期间调用：

```text
init_worker_distributed_environment
  -> init_distributed_environment
  -> ensure_model_parallel_initialized
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:290`
- `code/vllm/vllm/v1/worker/gpu_worker.py:294`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1164`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1188`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1197`

为什么在内存快照前初始化：

- NCCL/Gloo 等通信 buffer 会占显存/内存。
- 这些显存不应被 KV cache 使用。
- 所以先初始化通信，再 profile 可用 KV cache 显存。

## 4. init_distributed_environment

该函数处理默认 process group 初始化。

DP / 多节点场景会扩展 rank/world_size：

```text
rank = data_parallel_rank * world_size + rank
world_size = world_size_across_dp
```

单节点 DP 使用 `data_parallel_master_ip` 和 DP init port。

多节点使用 `master_addr/master_port`。

源码：

- `code/vllm/vllm/distributed/parallel_state.py:1516`
- `code/vllm/vllm/distributed/parallel_state.py:1536`
- `code/vllm/vllm/distributed/parallel_state.py:1548`
- `code/vllm/vllm/distributed/parallel_state.py:1550`
- `code/vllm/vllm/distributed/parallel_state.py:1553`
- `code/vllm/vllm/distributed/parallel_state.py:1558`

## 5. split_group 特殊路径

开启 `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` 时，默认 PG 初始化为：

```text
"cpu:gloo,cuda:nccl"
```

并传入：

```text
device_id=torch.device(f"cuda:{local_rank}")
```

目的：

- 确保 CPU 和 device backend 都存在。
- 后续 subgroup 可以通过 `split_group` 创建。

源码：

- `code/vllm/vllm/distributed/parallel_state.py:1434`
- `code/vllm/vllm/distributed/parallel_state.py:1439`
- `code/vllm/vllm/distributed/parallel_state.py:1445`
- `code/vllm/vllm/distributed/parallel_state.py:1589`

## 6. model parallel 分组布局

`initialize_model_parallel()` 的 rank layout：

```text
ExternalDP x DP x PP x PCP x TP
```

分组方式：

- TP group：最后一维连续。
- DCP group：复用 TP 维度切分。
- PCP group：PCP 维。
- PP group：PP 维，跨 pipeline stage。
- DP group：DP 维。
- EP group：合并 `DP * PCP * TP` 作为 expert parallel group。

源码：

- `code/vllm/vllm/distributed/parallel_state.py:1674`
- `code/vllm/vllm/distributed/parallel_state.py:1740`
- `code/vllm/vllm/distributed/parallel_state.py:1749`
- `code/vllm/vllm/distributed/parallel_state.py:1757`
- `code/vllm/vllm/distributed/parallel_state.py:1774`
- `code/vllm/vllm/distributed/parallel_state.py:1796`
- `code/vllm/vllm/distributed/parallel_state.py:1815`
- `code/vllm/vllm/distributed/parallel_state.py:1833`
- `code/vllm/vllm/distributed/parallel_state.py:1850`

## 7. EP / EPLB

MoE 模型才创建 EP group。

开启 `enable_eplb` 时会创建独立 EPLB group。

为什么 EPLB group 独立：

- EP group 用于 MoE forward collectives。
- EPLB 通信用同一个 group 可能和 MoE forward collectives 死锁。
- 所以 rank 相同，但 process group 分离。

Elastic EP 使用 stateless group。

源码：

- `code/vllm/vllm/distributed/parallel_state.py:1852`
- `code/vllm/vllm/distributed/parallel_state.py:1854`
- `code/vllm/vllm/distributed/parallel_state.py:1878`
- `code/vllm/vllm/distributed/parallel_state.py:1884`

## 8. GroupCoordinator

`GroupCoordinator` 是 vLLM 对一个逻辑通信 group 的封装。

内部维护：

- `cpu_group`：通常 Gloo，用于 CPU/object/metadata 控制通信。
- `device_group`：通常 NCCL，用于 GPU tensor 通信。
- `device_communicator`：平台相关通信器，如 CUDA 下的 `CudaCommunicator`。

源码：

- `code/vllm/vllm/distributed/parallel_state.py:351`
- `code/vllm/vllm/distributed/parallel_state.py:374`
- `code/vllm/vllm/distributed/parallel_state.py:375`
- `code/vllm/vllm/distributed/parallel_state.py:455`
- `code/vllm/vllm/distributed/parallel_state.py:459`

## 9. GroupCoordinator 通信分发

`GroupCoordinator` 对外提供：

- `all_reduce`
- `all_gather`
- `reduce_scatter`
- `send_tensor_dict`
- `recv_tensor_dict`
- `isend_tensor_dict`
- `irecv_tensor_dict`

`all_reduce()` 优先走自定义 op，否则调用 device communicator。

源码：

- `code/vllm/vllm/distributed/parallel_state.py:607`
- `code/vllm/vllm/distributed/parallel_state.py:626`
- `code/vllm/vllm/distributed/parallel_state.py:631`
- `code/vllm/vllm/distributed/parallel_state.py:636`
- `code/vllm/vllm/distributed/parallel_state.py:667`

## 10. CudaCommunicator

CUDA 平台下 device communicator 创建 `PyNcclCommunicator`。

TP group 上可启用多种 allreduce 后端：

- NCCL symmetric memory。
- QuickReduce。
- FlashInfer。
- Custom allreduce。
- SymmMem。
- PyNCCL。
- fallback torch distributed all_reduce。

源码：

- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:26`
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:45`
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:74`
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:100`
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:254`
- `code/vllm/vllm/distributed/device_communicators/cuda_communicator.py:297`

## 11. PyNcclCommunicator

PyNCCL 通信器特点：

- 基于非 NCCL CPU group 分发 NCCL unique id。
- 每个 communicator 绑定唯一 device。
- 创建后执行一次小 all_reduce warmup。
- destroy 使用 daemon thread 调 `ncclCommAbort()`，避免 CUDA graph 持有 NCCL op 时主线程自死锁。

源码：

- `code/vllm/vllm/distributed/device_communicators/pynccl.py:60`
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:78`
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:111`
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:121`
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:137`
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:142`
- `code/vllm/vllm/distributed/device_communicators/pynccl.py:148`

## 12. Pipeline Parallel 执行通信

GPU worker 的 `execute_model()` 处理 PP intermediate tensors。

流程：

```text
Worker.execute_model
  -> 如果上一轮有异步 PP send，先 wait
  -> 非 first PP rank: get_pp_group().irecv_tensor_dict() 接收上游 tensors
  -> model_runner.execute_model(..., intermediate_tensors)
  -> 如果输出 IntermediateTensors 且不是最后 PP rank:
       get_pp_group().isend_tensor_dict() 发送给下一 stage
       返回 None
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `code/vllm/vllm/v1/worker/gpu_worker.py:811`
- `code/vllm/vllm/v1/worker/gpu_worker.py:853`
- `code/vllm/vllm/v1/worker/gpu_worker.py:867`
- `code/vllm/vllm/v1/worker/gpu_worker.py:882`
- `code/vllm/vllm/v1/worker/gpu_worker.py:890`

## 13. DP 与 async scheduling

V1 async scheduling 启用时，DP synchronization 的 NCCL 可能默认禁用，改走 Gloo。

原因：

- async 场景下 NCCL 多流/同步更容易出问题。
- Gloo CPU group 更安全。

相关源码：

- `code/vllm/vllm/config/vllm.py:950`
- `code/vllm/vllm/config/vllm.py:970`
- `code/vllm/vllm/config/vllm.py:1016`
- `code/vllm/vllm/config/vllm.py:1021`
- `code/vllm/vllm/config/vllm.py:1030`

DP CUDAGraph padding 同步使用 `get_dp_group().cpu_group` 执行 CPU all_reduce：

- `code/vllm/vllm/v1/worker/gpu/dp_utils.py:16`
- `code/vllm/vllm/v1/worker/gpu/dp_utils.py:32`
- `code/vllm/vllm/v1/worker/gpu/dp_utils.py:37`
- `code/vllm/vllm/v1/worker/gpu/dp_utils.py:88`
- `code/vllm/vllm/v1/worker/gpu/dp_utils.py:120`

## 14. 控制面通信与模型并行通信的区别

非常重要：Executor 的 RPC/MQ 和模型并行通信不是一回事。

### 14.1 Executor 控制面

负责调用方法：

```text
execute_model
sample_tokens
initialize_from_config
profile
shutdown
```

实现方式：

- UniProc：函数调用。
- Multiproc：MQ。
- Ray old：Ray remote / DAG。
- Ray v2：MQ over Ray actor。

### 14.2 模型并行通信

负责模型计算中的 tensor/object 通信：

- TP all_reduce。
- PP send/recv intermediate tensors。
- DP synchronization。
- EP/MoE collectives。
- KV transfer 相关通信。

实现方式：

- torch distributed。
- NCCL/Gloo。
- PyNCCL。
- vLLM custom allreduce。
- GroupCoordinator。

## 15. 关键理解

1. rank/local_rank 是贯穿 executor、worker、distributed 初始化的核心。
2. Executor 控制面只负责“调用 worker 方法”，不负责 TP/PP tensor 通信。
3. TP/PP/DP/EP 通信在 worker/model runner 内通过 distributed groups 完成。
4. PP intermediate tensors 在 worker 层显式 send/recv。
5. DP async scheduling 下可能默认使用 CPU/Gloo 同步。
6. Ray v2 复用 vLLM 的 distributed 通信，而旧 Ray executor 更依赖 Ray compiled DAG。
