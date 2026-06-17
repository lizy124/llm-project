# 03. Executor 与 Worker 生命周期

## 1. 核心文件

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:37`：`Executor`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:45`：`UniProcExecutor`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:150`：`ExecutorWithExternalLauncher`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:103`：`MultiprocExecutor`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:554`：`WorkerProc`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py:64`：`RayDistributedExecutor`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:75`：`RayWorkerProc`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:205`：`RayExecutorV2`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:39`：`WorkerBase`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:187`：`WorkerWrapperBase`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:117`：GPU `Worker`

## 2. Executor 抽象

`Executor` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:37`。

职责：

- 选择和初始化执行后端。
- 管理 worker 生命周期。
- 提供统一的 control plane RPC。
- 触发模型执行和采样。
- 初始化 KV cache。
- 收集 worker 侧 KV cache specs 和可用显存。
- 执行 health check / shutdown / sleep / wake_up。

### 2.1 get_class()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:48`

根据 `parallel_config.distributed_executor_backend` 选择：

```text
ray -> RayDistributedExecutor 或 RayExecutorV2
mp -> MultiprocExecutor
uni -> UniProcExecutor
external_launcher -> ExecutorWithExternalLauncher
自定义字符串 -> resolve_obj_by_qualname
```

### 2.2 __init__()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:95`

它保存所有 config，然后调用抽象方法 `_init_executor()`。

这意味着不同 executor 的 worker 创建、Ray actor 创建、message queue 创建等都发生在 `_init_executor()`。

### 2.3 collective_rpc()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:153`

语义：在所有 worker 上执行某个方法。

参数：

- method：方法名或 callable。
- timeout。
- args / kwargs。
- non_block。

重要注释：该 API 推荐只传 control messages，大数据通信应走 data-plane。

### 2.4 initialize_from_config()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:118`

流程：

1. `collective_rpc("initialize_from_config", args=(kv_cache_configs,))`
2. `collective_rpc("compile_or_warm_up_model")`
3. 汇总 worker 编译时间。

worker 侧对应：

- `WorkerWrapperBase.initialize_from_config()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:315`
- `Worker.initialize_from_config()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:562`

## 3. UniProcExecutor

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:45`

适用：单进程、单 worker、本地直接调用。

### 3.1 初始化

`_init_executor()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:46`。

流程：

1. 创建 `WorkerWrapperBase(rpc_rank=0)`。
2. `_distributed_args()` 返回 init method、rank、local_rank。
3. 组装 worker kwargs。
4. 设置 worker 网络设备。
5. `driver_worker.init_worker(...)`。
6. `driver_worker.init_device()`。
7. `driver_worker.load_model()`。
8. 更新 block size backend。

### 3.2 distributed args

默认 `_distributed_args()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:71`。

- `distributed_init_method = tcp://ip:port`
- `rank = 0`
- `local_rank` 从 device 字符串解析。

### 3.3 collective_rpc

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:79`

它直接：

```text
run_method(self.driver_worker, method, args, kwargs)
```

所以 UniProc 没有真正 IPC。

### 3.4 ExternalLauncher

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:150`

设计目标：torchrun-compatible launchers 下，每个 executor 只创建一个 worker，但多个外部进程共同组成 TP group。

`_distributed_args()` 使用环境变量：

- `RANK`
- `LOCAL_RANK`
- `MASTER_ADDR`
- `MASTER_PORT`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:175`。

## 4. MultiprocExecutor

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:103`

适用：单机/多节点多进程，vLLM V1 的核心多 worker 后端。

### 4.1 初始化流程

`_init_executor()` 从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:110` 开始。

关键步骤：

1. 注册 finalizer，确保退出时 shutdown。
2. 校验：`world_size == TP × PP × PCP`。
3. `set_multiprocessing_worker_envs()`。
4. 构造 `distributed_init_method`。
5. 如果是 DP group leader，创建 `rpc_broadcast_mq`。
6. 创建 multiprocessing context。
7. 创建 shared worker lock。
8. 根据 `local_world_size` 启动本地 worker 进程。
9. `WorkerProc.wait_for_ready()` 等所有本地 worker ready。
10. 启动 worker monitor。
11. 收集 response message queues。
12. 等所有 MQ ready。
13. 初始化 `futures_queue`。
14. `_post_init_executor()`。
15. 设置 `output_rank`。

### 4.2 world size 与 local_world_size

`_get_parallel_sizes()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:249`。

它读取：

- `parallel_config.world_size`
- `parallel_config.local_world_size`
- `tensor_parallel_size`
- `pipeline_parallel_size`
- `prefill_context_parallel_size`

### 4.3 driver worker 判断

`_is_driver_worker()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:265`。

```text
rank % tensor_parallel_size == 0
```

即每个 TP group 的 rank 0 是 driver worker。

### 4.4 execute_model / sample_tokens

位置：

- `execute_model()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:307`
- `sample_tokens()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:319`

它们都调用 `collective_rpc()`，但设置：

- `unique_reply_rank=self.output_rank`
- `timeout=VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
- 可选 `kv_output_aggregator`

因为通常只有一个 rank 的输出需要返回给 engine core。

### 4.5 collective_rpc

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:340`

流程：

1. 校验 `rpc_broadcast_mq` 存在。
2. 如果 executor 已失败，直接报错。
3. 把 method 编码为字符串或 cloudpickle。
4. 入队：

```text
rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
```

5. 根据 `output_rank` 决定等待一个 response queue 还是所有 response queues。
6. 构造 `FutureWrapper`。
7. non_block 返回 future，否则立即 `future.result()`。

### 4.6 worker monitor

`start_worker_monitor()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:268`。

它监听 worker 进程 sentinel。一旦任意 worker 死亡：

1. 标记 executor failed。
2. shutdown executor。
3. 调用 failure callback。

这是多进程模式的核心健康机制。

## 5. WorkerProc

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:554`

`WorkerProc` 是子进程 worker 的封装。

关键方法：

- `_init_message_queues()`：初始化输入/输出 MQ。
- `make_worker_process()`：创建进程。
- `wait_for_ready()`：等待 worker 初始化成功。
- `worker_main()`：子进程入口。
- `worker_busy_loop()`：循环接收 RPC 并执行。
- `handle_output()`：处理输出。
- `async_output_busy_loop()`：异步输出线程。
- `shutdown()`：清理进程和 MQ。

Multiproc 的控制面结构：

```text
Executor process
  rpc_broadcast_mq  --广播-->  WorkerProc.worker_busy_loop()
  response_mqs      <--返回--  WorkerProc.handle_output()
```

数据面仍由 worker 内的 torch.distributed groups 负责。

## 6. RayDistributedExecutor

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py:64`

特点：

- worker 是 Ray actor。
- 可使用 Ray compiled DAG。
- 更偏 Ray 原生 actor + DAG 模式。

关键方法：

- `_init_executor()`：初始化 Ray executor。
- `_init_workers_ray()`：根据 placement group 创建 Ray workers。
- `collective_rpc()`：对 Ray actors 调用远程方法。
- `execute_model()`：通过 Ray 或 DAG 执行模型。
- `_compiled_ray_dag()`：构建 Ray compiled graph。
- `_execute_dag()`：执行 DAG。
- `shutdown()`：清理 Ray workers。

适合更强 Ray 编排场景，但当前代码中 Ray V2 更接近 Multiproc 的控制面设计。

## 7. RayExecutorV2

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:205`

类注释指出：RayExecutorV2 使用 MessageQueue communication，继承 MultiprocExecutor 以复用 MQ 控制面和 NCCL 数据面。

### 7.1 设计目标

```text
Ray 管 worker placement / actor lifecycle
MessageQueue 管 control plane
NCCL / torch.distributed 管 data plane
```

### 7.2 初始化流程

`_init_executor()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:249`。

关键步骤：

1. 初始化 Ray cluster。
2. 获取 placement group。
3. 校验 `world_size == TP × PP × PCP`。
4. 根据 bundle 分配 worker ranks。
5. rank 0 所在节点提供 torch distributed TCPStore 地址。
6. 创建 `rpc_broadcast_mq`。
7. 创建 RayWorkerProc actors。
8. 获取每个 actor 的 node id 与 GPU ids。
9. 按 node 分配 local_rank 和 `CUDA_VISIBLE_DEVICES`。
10. 调 actor 的 `initialize_worker()`。
11. 收集 response MQ handles。
12. 调 actor 的 `run()`。
13. 等所有 MQ ready。
14. `_post_init_executor()`。
15. 启动 Ray worker monitor。
16. 设置 output rank。

### 7.3 RayWorkerProc

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:75`

它继承/复用 Multiproc 的 WorkerProc 思路，但运行在 Ray actor 中。

关键点：

- actor 创建时轻量初始化。
- GPU id discovery 后才初始化 worker。
- `run()` 进入 worker busy loop。

### 7.4 monitor

`start_worker_monitor()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:429`。

它通过 `ray.wait()` 监听 actor 的 `run_ref`。如果任意 actor 结束，标记 executor failed 并 shutdown。

## 8. WorkerBase 与 WorkerWrapperBase

### 8.1 WorkerBase

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:39`

它定义 worker 的统一接口：

- `init_device()`
- `load_model()`
- `execute_model()`
- `sample_tokens()`
- `get_kv_cache_spec()`
- `compile_or_warm_up_model()`
- `get_cache_block_size_bytes()`
- LoRA 相关方法
- `shutdown()`

WorkerBase 保存：

- `vllm_config`
- `model_config`
- `cache_config`
- `parallel_config`
- `device_config`
- `kv_transfer_config`
- `local_rank`
- `rank`
- `distributed_init_method`
- `is_driver_worker`

### 8.2 WorkerWrapperBase

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:187`

它是 executor 和具体 worker 之间的延迟初始化包装层。

职责：

1. 保存 rpc rank / global rank。
2. 接收环境变量更新。
3. 根据 `parallel_config.worker_cls` 动态加载 worker class。
4. 注入 worker extension class。
5. 创建多模态 shared memory cache receiver。
6. 在 `set_current_vllm_config()` 上下文里实例化真实 worker。
7. `initialize_from_config()` 时按 global rank 取对应 KV cache config。

## 9. GPU Worker 生命周期

GPU Worker 定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:117`

### 9.1 构造

`__init__()` 保存：

- Elastic EP executor。
- sleep buffer。
- weight transfer engine。
- profiler。
- 是否 V2 model runner。
- PP send handles。

### 9.2 init_device()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:249`

流程：

1. 处理 Ray/NCCL 环境变量。
2. 根据 DP local rank 调整 local_rank。
3. 设置 CUDA device。
4. 检查 dtype 支持。
5. 调 `init_worker_distributed_environment()`。
6. 设置随机种子。
7. 清理 CUDA cache 并记录 memory snapshot。
8. 初始化 workspace manager。
9. 创建 GPUModelRunner V1 或 V2。
10. rank 0 记录 usage stats。

`init_worker_distributed_environment()` 内部会调用：

- `init_distributed_environment()`
- `ensure_model_parallel_initialized()`
- `ensure_ec_transfer_initialized()`

### 9.3 load_model()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:349`

在 weights memory pool 与当前 vLLM config 上下文中加载模型。

如果 `weight_transfer_config` 非空，会创建 `WeightTransferEngine`。

### 9.4 determine_available_memory()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:371`

用于 profile 可用于 KV cache 的显存。

关键点：

- 如果用户手动设置 `kv_cache_memory_bytes`，仍会跑 profile compile，但跳过自动估算。
- 否则执行 dummy forward profile。
- 估算非 KV cache 内存、activation peak、CUDA graph 内存。
- 返回可用 KV cache memory bytes。

### 9.5 initialize_from_config()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:562`

流程：

1. 写回 `cache_config.num_gpu_blocks`。
2. `ensure_kv_transfer_initialized(vllm_config, kv_cache_config)`。
3. 在 kv_cache memory pool 中 `model_runner.initialize_kv_cache(kv_cache_config)`。
4. 如果需要 routed experts capturer，初始化。
5. 如果 KV cache 需要 zeroing，初始化 zero metadata。

### 9.6 compile_or_warm_up_model()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:591`

做模型 warmup、kernel warmup、CUDA graph capture、sampler warmup、JIT monitor 激活、worker heap freeze。

### 9.7 execute_model()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:807`

核心执行路径：

1. 等待上一轮非阻塞 PP send 完成。
2. 判断是否需要 forward pass。
3. 如果启用 PP 且当前不是 first PP rank，调用 `get_pp_group().irecv_tensor_dict(...)` 接收上游 intermediate tensors。
4. 调 `model_runner.execute_model(scheduler_output, intermediate_tensors)`。
5. 如果返回 `ModelRunnerOutput`、`AsyncModelRunnerOutput` 或 `None`，直接返回。
6. 如果返回 `IntermediateTensors`，说明当前 rank 不是 PP last rank，需要 `get_pp_group().isend_tensor_dict(...)` 发给下游。
7. 返回 `None`。

### 9.8 sample_tokens()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:801`

直接调用 `model_runner.sample_tokens(grammar_output)`。

## 10. Executor / Worker 边界总结

```text
Executor 负责：
  - 创建 worker
  - 给 worker 发 control RPC
  - 收集 worker 输出
  - 健康监控与 shutdown

Worker 负责：
  - 初始化设备和 distributed groups
  - 加载模型
  - 分配 KV cache
  - 执行 forward/sample
  - 执行 PP/TP/DP/EP 数据面通信
  - 接入 KV/EC/weight transfer
```

这个边界非常重要：executor 不直接执行 tensor collective；真正的 collective 通常发生在 worker 内部或模型执行过程中。
