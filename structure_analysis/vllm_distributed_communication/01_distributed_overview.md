# 01. vLLM 分布式通信层总览

## 1. 总体分层

vLLM 的分布式通信层可以按下面 6 层理解：

```text
配置层
  ParallelConfig / KVTransferConfig / ECTransferConfig
    ↓
Executor 控制面
  UniProcExecutor / MultiprocExecutor / RayDistributedExecutor / RayExecutorV2
    ↓
Worker 执行面
  WorkerWrapperBase / WorkerBase / GPU Worker / GPUModelRunner
    ↓
分布式运行时
  torch.distributed / parallel_state.GroupCoordinator
    ↓
设备通信实现
  DeviceCommunicator / NCCL / Gloo / custom all-reduce / all2all / SHM MQ
    ↓
扩展传输层
  KV transfer / EC transfer / Weight transfer / Elastic EP / EPLB
```

核心入口：

- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:117`：`ParallelConfig`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:37`：`Executor`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:39`：`WorkerBase`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:187`：`WorkerWrapperBase`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:117`：GPU `Worker`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:351`：`GroupCoordinator`

## 2. 控制面与数据面

vLLM 分布式通信必须区分两类通道。

### 2.1 控制面

控制面用于发命令、传调度输出、收 worker 返回结果。

典型内容：

- `SchedulerOutput`
- `GrammarOutput`
- `KVCacheConfig`
- worker method name
- health check / shutdown / sleep / wake_up
- LoRA 添加删除
- profiling 指令

对应代码：

- `Executor.collective_rpc()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:153`
- `MultiprocExecutor.collective_rpc()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:340`
- `UniProcExecutor.collective_rpc()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:79`
- `RayDistributedExecutor.collective_rpc()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py:479`

控制面要求可靠，但通常不是大 tensor 的高吞吐路径。`Executor.collective_rpc()` 的注释明确建议：该 API 只用于 control messages，data-plane communication 应单独建立，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:181`。

### 2.2 数据面

数据面用于真正高吞吐通信：

- TP all-reduce / all-gather / reduce-scatter
- PP 中间 hidden states send/recv
- PP sampled token broadcast
- DP batch 协调 all-reduce
- EP/MoE all2all dispatch/combine
- DCP/PCP context parallel 通信
- KV cache transfer
- EC cache transfer
- weight transfer

对应代码：

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/*`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:51`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:164`

## 3. 分布式相关目录

### 3.1 `vllm/config`

- `parallel.py`
  - 分布式拓扑与 executor backend 总配置。
- `kv_transfer.py`
  - KV cache 跨实例传输配置。
- `ec_transfer.py`
  - Encoder cache 跨实例传输配置。

### 3.2 `vllm/v1/executor`

- `abstract.py`
  - Executor 抽象接口。
- `uniproc_executor.py`
  - 单进程 executor。
- `multiproc_executor.py`
  - 多进程 executor，使用 message queue 控制面。
- `ray_executor.py`
  - Ray V1 executor，Ray actor / compiled DAG 风格。
- `ray_executor_v2.py`
  - Ray V2 executor，Ray 管 actor，控制面复用 MessageQueue，数据面复用 NCCL/torch.distributed。
- `ray_utils.py`
  - Ray worker 排布、FutureWrapper 等辅助。
- `ray_env_utils.py`
  - Ray 环境变量处理。
- `vllm_net_devices.py`
  - worker 网络设备设置。

### 3.3 `vllm/v1/worker`

- `worker_base.py`
  - worker 抽象与 wrapper。
- `gpu_worker.py`
  - GPU worker 主实现，分布式 init、模型加载、执行、PP send/recv、KV/weight transfer 都在这里接入。
- `gpu/pp_utils.py`
  - PP sampled token broadcast/recv。
- `dp_utils.py`
  - DP batch 协调。
- `kv_connector_model_runner_mixin.py`
  - KV connector 与 model runner 的生命周期接入。
- `ec_connector_model_runner_mixin.py`
  - EC connector 接入。

### 3.4 `vllm/distributed`

- `parallel_state.py`
  - 分布式运行时中枢。
- `communication_op.py`
  - custom collective op 注册。
- `device_communicators/*`
  - CUDA/XPU/CPU/Ray/custom all-reduce/all2all/SHM/NCCL 等通信实现。
- `kv_transfer/**`
  - KV cache transfer。
- `ec_transfer/**`
  - Encoder cache transfer。
- `weight_transfer/**`
  - 权重热更新/传输。
- `elastic_ep/**`
  - 弹性 expert parallel。
- `eplb/**`
  - expert parallel load balancing。

## 4. 主初始化链路

从 engine 创建到 worker 分布式环境初始化：

```text
EngineCore.__init__
  -> Executor.get_class(vllm_config)
  -> executor_class(vllm_config)
      -> Executor.__init__
          -> _init_executor()
              -> 创建 worker / actor / 子进程
              -> WorkerWrapperBase.init_worker()
              -> Worker.init_device()
                  -> init_worker_distributed_environment()
                      -> init_distributed_environment()
                      -> ensure_model_parallel_initialized()
                      -> ensure_ec_transfer_initialized()
                  -> 创建 GPUModelRunner
              -> Worker.load_model()
  -> EngineCore._initialize_kv_caches()
      -> executor.get_kv_cache_specs()
      -> executor.determine_available_memory()
      -> executor.initialize_from_config(kv_cache_configs)
          -> Worker.initialize_from_config()
              -> ensure_kv_transfer_initialized()
              -> model_runner.initialize_kv_cache()
              -> compile_or_warm_up_model()
```

关键位置：

- `Executor.get_class()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:48`
- `Executor.__init__()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:95`
- `Worker.init_device()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:249`
- `init_distributed_environment()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1516`
- `initialize_model_parallel()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1674`
- `Worker.initialize_from_config()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:562`

## 5. 主运行链路

一次 step 的分布式执行链路：

```text
EngineCore.step()
  -> scheduler.schedule()
  -> model_executor.execute_model(scheduler_output, non_block=True)
      -> Executor.collective_rpc("execute_model")
          -> 每个 worker 执行 Worker.execute_model(scheduler_output)
              -> 如果不是 PP first rank，irecv 上游 IntermediateTensors
              -> model_runner.execute_model(...)
              -> 如果不是 PP last rank，isend 下游 IntermediateTensors
              -> last rank 返回 ModelRunnerOutput 或 None
  -> model_executor.sample_tokens(grammar_output)  # 如果 execute_model 返回 None
  -> scheduler.update_from_output(scheduler_output, model_output)
```

关键位置：

- `Executor.execute_model()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:221`
- `MultiprocExecutor.execute_model()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:307`
- `Worker.execute_model()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:807`

## 6. 并行类型总览

### 6.1 Tensor Parallel，TP

TP 通常用于模型权重和张量计算切分，运行时依赖：

- all-reduce
- all-gather
- reduce-scatter
- all-gatherv / reduce-scatterv

Group：`get_tp_group()`。

### 6.2 Pipeline Parallel，PP

PP 把模型层切成多个 stage。

通信包括：

- stage 间 intermediate tensor send/recv。
- last stage sampled tokens 广播给前面 stage。

Group：`get_pp_group()`。

### 6.3 Data Parallel，DP

DP 是多副本并行，每个 DP rank 可以有自己的 TP/PP/PCP 组。

通信包括：

- batch 形态协调。
- padding/ubatch/cudagraph mode 同步。
- MoE EP 也会使用 DP 维度。

Group：`get_dp_group()`。

### 6.4 Prefill Context Parallel，PCP

PCP 用于 prefill context parallel。

Group：`get_pcp_group()`。

### 6.5 Decode Context Parallel，DCP

DCP 在当前实现中复用 TP 组内 GPU，要求 `tp_size % dcp_size == 0`。

Group：`get_dcp_group()`。

### 6.6 Expert Parallel，EP

EP 用于 MoE expert 分片与 all2all 通信。

Group：`get_ep_group()`。

### 6.7 EPLB

EPLB 是 expert parallel load balancing，使用与 EP 相同 rank 集合但独立 process group，避免 MoE forward collective 与 EPLB 通信互相死锁。

Group：`get_eplb_group()`。

## 7. 核心设计不变量

1. `ParallelConfig` 决定拓扑与 executor 后端。
2. Executor 负责 worker 生命周期和控制面 RPC。
3. Worker 内部负责真正初始化 `torch.distributed` 和 model parallel groups。
4. `parallel_state.GroupCoordinator` 是所有 process group 通信的统一 facade。
5. 高吞吐 tensor 通信不走 `collective_rpc`，而走 group/device communicator。
6. PP、DP、EP、KV/EC transfer 各自有专用通信路径。
7. Ray V2 的设计目标是让 Ray 只管 placement 和 actor 生命周期，通信协议尽量复用 Multiproc 的 MQ + NCCL 数据面。
