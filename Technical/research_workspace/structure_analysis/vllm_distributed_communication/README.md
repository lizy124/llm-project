# vLLM 分布式通信层梳理

本目录用于系统梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的分布式通信层。

当前代码库中，分布式通信层不是单个模块，而是由配置层、executor 控制面、worker 执行面、`torch.distributed` 进程组、device communicator、pipeline/data/expert/context parallel 运行时通信，以及 KV/EC/weight transfer 等扩展通道共同组成。

## 文档目录

1. [分布式通信层总览](01_distributed_overview.md)
   - 分层架构、控制面/数据面、核心目录、主调用链。

2. [ParallelConfig 与并行拓扑](02_parallel_config_and_topology.md)
   - TP、PP、DP、PCP、DCP、EP、EPLB、Ray/MP/uni/external launcher、world size 公式与关键配置字段。

3. [Executor 与 Worker 生命周期](03_executor_worker_lifecycle.md)
   - `Executor` 抽象、UniProc、Multiproc、Ray、Ray V2、WorkerWrapper、GPU Worker 初始化和执行边界。

4. [进程组与通信原语](04_process_groups_and_collectives.md)
   - `parallel_state.py`、`GroupCoordinator`、WORLD/TP/PP/DP/DCP/PCP/EP/EPLB group、broadcast/all_reduce/send/recv/tensor_dict。

5. [运行时并行通信流程](05_runtime_parallel_flows.md)
   - PP 中间张量传输、sampled token 广播、DP batch 协调、Expert Parallel/MoE all2all、DCP/PCP 关系。

6. [KV/EC/Weight Transfer 与 Elastic EP](06_transfer_and_elastic_layers.md)
   - KV transfer、EC transfer、weight transfer、Elastic EP、EPLB 的配置、初始化、握手与运行时交互。

7. [旧架构映射与阅读指南](07_legacy_mapping_and_reading_guide.md)
   - V1 与非 V1 分布式代码关系、排查路径、推荐阅读顺序。

## 核心代码入口

### 配置层

- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py`

### Executor 层

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py`

### Worker 层

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py`

### Distributed runtime

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/communication_op.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/*`

### Transfer 扩展层

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/kv_transfer/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/ec_transfer/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/weight_transfer/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/elastic_ep/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/eplb/**`

## 一句话总览

vLLM 的分布式通信层可以理解为：`ParallelConfig` 决定并行拓扑，`Executor` 负责创建和驱动 worker，`Worker` 在各 rank 上初始化 `torch.distributed` 和模型，`parallel_state.GroupCoordinator` 统一封装进程组与 collective/p2p 通信，device communicator 提供高性能设备通信，KV/EC/Weight transfer 与 EP/EPLB 是围绕模型执行扩展出来的专用通信通道。
