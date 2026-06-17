# vLLM 分布式通信层技术点问答

本文基于本目录已有文档整理，面向技术考察/面试/源码讲解，覆盖 vLLM 分布式通信层中可能被提问的关键技术点，并给出可直接回答的参考答案。

## 1. 总体架构

### Q1：vLLM 的分布式通信层可以分为哪几层？

答：可以分为 6 层：

1. 配置层：`ParallelConfig`、`KVTransferConfig`、`ECTransferConfig`，负责描述并行拓扑和扩展传输配置。
2. Executor 控制面：`UniProcExecutor`、`MultiprocExecutor`、`RayDistributedExecutor`、`RayExecutorV2`，负责创建和驱动 worker。
3. Worker 执行面：`WorkerWrapperBase`、`WorkerBase`、GPU `Worker`、`GPUModelRunner`，负责设备初始化、模型加载、forward/sample。
4. 分布式运行时：`torch.distributed` 与 `parallel_state.GroupCoordinator`，负责 process group 管理。
5. 设备通信实现：`DeviceCommunicator`、NCCL、Gloo、custom all-reduce、all2all、SHM MQ 等。
6. 扩展传输层：KV transfer、EC transfer、Weight transfer、Elastic EP、EPLB。

核心理解：`ParallelConfig` 定拓扑，`Executor` 管 worker，`Worker` 初始化 distributed 和模型，`GroupCoordinator` 统一封装通信原语，扩展传输层服务 KV/EC/权重/MoE 等高级场景。

### Q2：vLLM 分布式中的“控制面”和“数据面”有什么区别？

答：控制面用于发命令和传较小的元数据，数据面用于高吞吐 tensor 通信。

控制面典型内容：

- `SchedulerOutput`
- `KVCacheConfig`
- worker method name
- health check、shutdown、sleep、wake_up
- LoRA add/remove
- profile 指令

控制面主要通过 `Executor.collective_rpc()` 实现。

数据面典型内容：

- TP all-reduce/all-gather/reduce-scatter
- PP intermediate tensors send/recv
- PP sampled token broadcast
- DP batch shape 同步
- EP/MoE all2all dispatch/combine
- DCP/PCP context parallel 通信
- KV/EC/weight transfer

数据面主要通过 `parallel_state.GroupCoordinator`、`DeviceCommunicator`、NCCL/Gloo/custom ops 等实现。

### Q3：为什么 `Executor.collective_rpc()` 不适合大 tensor 通信？

答：`collective_rpc()` 是控制面 RPC，适合传命令、配置、调度结果和小型元数据。它通常经过 message queue、Ray RPC 或本地函数调用，设计目标是可靠地调用 worker 方法，而不是高吞吐 tensor 传输。大 tensor 通信需要低延迟、高带宽、GPU-aware 的通信路径，因此应该走 NCCL、device communicator、process group、tensor dict p2p 或 all2all 等数据面通道。

### Q4：vLLM 分布式通信层的主初始化链路是什么？

答：主链路是：

```text
EngineCore.__init__
  -> Executor.get_class(vllm_config)
  -> executor_class(vllm_config)
      -> Executor.__init__
          -> _init_executor()
              -> 创建 worker / actor / 子进程
              -> WorkerWrapperBase.init_worker()
              -> Worker.init_device()
                  -> init_distributed_environment()
                  -> ensure_model_parallel_initialized()
                  -> ensure_ec_transfer_initialized()
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

### Q5：一次 `execute_model` 的分布式运行主链路是什么？

答：一次 step 中，EngineCore 先调用 scheduler 生成 `SchedulerOutput`，然后 executor 广播执行命令到 worker：

```text
EngineCore.step()
  -> scheduler.schedule()
  -> Executor.execute_model(scheduler_output)
      -> Executor.collective_rpc("execute_model")
          -> Worker.execute_model(scheduler_output)
              -> PP 场景下接收上游 IntermediateTensors
              -> GPUModelRunner.execute_model(...)
              -> TP/EP/DP 等通信在模型执行中发生
              -> PP 场景下发送下游 IntermediateTensors
              -> last rank 返回 ModelRunnerOutput 或 None
  -> Executor.sample_tokens(grammar_output)  # 如果 execute_model 返回 None
  -> scheduler.update_from_output(...)
```

## 2. ParallelConfig 与拓扑

### Q6：`ParallelConfig` 的核心作用是什么？

答：`ParallelConfig` 是 vLLM 分布式拓扑的总配置入口。它决定：

- 使用多少 worker；
- 使用哪个 executor backend；
- TP、PP、DP、PCP、DCP、EP、EPLB 的规模；
- Ray/MP/uni/external launcher 的控制面；
- MoE expert parallel 和 all2all backend；
- DP 同步使用 NCCL 还是 Gloo；
- 多节点 master 地址、端口、node rank；
- Elastic EP、EPLB、DBO、context parallel 等高级能力。

可以把 `ParallelConfig` 理解为“分布式拓扑编译器”。

### Q7：vLLM 中 TP、PP、DP、PCP、DCP、EP 分别是什么意思？

答：

- TP，Tensor Parallel：切分模型权重和张量计算，常用 all-reduce/all-gather/reduce-scatter。
- PP，Pipeline Parallel：按层切分模型到多个 pipeline stage，stage 间传 intermediate tensors。
- DP，Data Parallel：多个模型副本并行处理请求，需要协调 batch shape、padding、CUDA graph mode。
- PCP，Prefill Context Parallel：prefill 阶段的 context parallel，参与 world size。
- DCP，Decode Context Parallel：decode 阶段的 context parallel，复用 TP group 内 GPU，不增加 world size。
- EP，Expert Parallel：MoE experts 分布到不同 rank，通过 all2all dispatch/combine。

### Q8：`world_size` 和 TP/PP/PCP 的关系是什么？

答：在当前 V1 Multiproc/Ray V2 逻辑中，executor world size 需要满足：

```text
world_size = tensor_parallel_size × pipeline_parallel_size × prefill_context_parallel_size
```

DP 不直接包含在 executor 创建 worker 的这个 world size 内，但在 `init_distributed_environment()` 中会扩展到跨 DP 的总 world size。

### Q9：DCP 为什么不增加 world size？

答：DCP，即 decode context parallel，复用 TP group 内的 GPU。代码注释明确要求 DCP size 不能超过 TP size，因为 DCP 只是在 TP group 内进一步划分 decode context，而不是额外创建 worker。因此约束是：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

如果不满足，拓扑无法正确映射。

### Q10：PCP 和 DCP 的区别是什么？

答：PCP 是 prefill context parallel，会参与 executor world size：

```text
world_size = TP × PP × PCP
```

DCP 是 decode context parallel，不增加 world size，而是在 TP group 内复用已有 GPU。

简单说：PCP 增加 worker 维度，DCP 复用 TP 维度。

### Q11：`initialize_model_parallel()` 中 rank tensor 的维度顺序是什么？

答：核心 rank tensor 是：

```text
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

维度顺序是：

```text
ExternalDP × DP × PP × PCP × TP
```

后续通过 `view`、`transpose`、`reshape` 创建不同 process group。

### Q12：TP group 是如何创建的？

答：TP group 沿最后一个 TP 维度划分：

```text
group_ranks = all_ranks.view(-1, tensor_model_parallel_size)
```

每个 group 包含同一 DP、PP、PCP 下的 TP ranks。TP group 主要用于模型层内部的 tensor parallel collective。

### Q13：PP group 是如何创建的？

答：PP group 通过交换 PP 和 TP 维度后 reshape 得到：

```text
group_ranks = all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size)
```

每个 PP group 包含同一 DP、PCP、TP 位置下的多个 pipeline stage。

### Q14：DP group 是如何创建的？

答：DP group 通过交换 DP 和 TP 维度后 reshape 得到：

```text
group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size)
```

DP group 用于推理时多个 DP replica 之间协调 batch 形态，而不是训练中的梯度同步。

### Q15：EP group 是如何创建的？

答：EP group 的 rank 集合来自同一个 PP stage 内的 DP、PCP、TP ranks：

```text
group_ranks = all_ranks.transpose(1, 2).reshape(
    -1,
    data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size,
)
```

它服务 MoE expert parallel，用于 router logits、token dispatch、expert output combine 等 all2all 通信。

### Q16：EPLB group 为什么要和 EP group 分开？

答：EPLB 使用与 EP 相同的 rank 集合，但创建独立 process group。原因是 EPLB 的负载均衡通信不应与 MoE forward pass 的 collectives 共用同一个 communicator，否则可能互相等待导致死锁。独立 group 可以隔离 EPLB 通信和正常 MoE 前向通信。

### Q17：`distributed_executor_backend` 支持哪些值？

答：常见值包括：

- `uni`：使用 `UniProcExecutor`。
- `mp`：使用 `MultiprocExecutor`。
- `ray`：使用 `RayDistributedExecutor` 或 `RayExecutorV2`。
- `external_launcher`：使用 `ExecutorWithExternalLauncher`。
- 自定义 Executor 类或类路径字符串。

选择逻辑在 `Executor.get_class()` 中完成。

### Q18：什么时候使用 `external_launcher`？

答：`external_launcher` 适合 torchrun 等外部启动器场景。每个外部进程中 executor 只创建一个 worker，多个外部进程通过环境变量 `RANK`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT` 等共同组成 TP/PP 等分布式组。

### Q19：DP 同步为什么可以选择禁用 NCCL？

答：DP batch 协调只需要同步少量整数 metadata，例如 token 数、padding 后 token 数、是否 ubatch、CUDA graph mode。默认可用 NCCL/device group；如果设置 `disable_nccl_for_dp_synchronization=True`，则改用 CPU tensor 和 Gloo group，避免某些环境下 NCCL 初始化或通信开销/兼容性问题。

### Q20：DBO/ubatch 配置会影响什么？

答：DBO/ubatch 会影响 worker 内部 workspace 数量、microbatch 切分和 DP batch 协调。启用 DBO 时，worker 会使用更多 ubatch workspace；DP 组内还要协商是否所有 rank 都允许 microbatching，只有所有 DP rank 都同意且最后一个 ubatch 不为空，才真正启用 ubatching。

## 3. Executor 与 Worker 生命周期

### Q21：Executor 抽象负责什么？

答：Executor 是 EngineCore 和 worker 之间的控制面抽象，负责：

- 选择并初始化执行后端；
- 创建 worker 或 Ray actor；
- 管理 worker 生命周期；
- 提供 `collective_rpc()`；
- 调用 `execute_model()` 和 `sample_tokens()`；
- 初始化 KV cache；
- 收集 KV cache specs 和可用显存；
- health check、shutdown、sleep、wake_up；
- LoRA add/remove/pin/list。

### Q22：`Executor.get_class()` 如何选择具体 executor？

答：它读取 `parallel_config.distributed_executor_backend`：

```text
ray -> RayDistributedExecutor 或 RayExecutorV2
mp -> MultiprocExecutor
uni -> UniProcExecutor
external_launcher -> ExecutorWithExternalLauncher
自定义字符串 -> resolve_obj_by_qualname
自定义 class -> 直接使用
```

Ray V2 是否启用取决于 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`。

### Q23：`Executor.initialize_from_config()` 做什么？

答：它分两步：

1. 通过 `collective_rpc("initialize_from_config", args=(kv_cache_configs,))` 通知所有 worker 分配 KV cache。
2. 通过 `collective_rpc("compile_or_warm_up_model")` 通知 worker 编译/warmup/cuda graph capture，并汇总 worker 编译时间。

它是 EngineCore 侧完成 KV cache 初始化和模型 warmup 的入口。

### Q24：UniProcExecutor 的特点是什么？

答：UniProcExecutor 适用于单进程、单 worker、本地直接调用。它创建一个 `WorkerWrapperBase`，初始化 worker、设备和模型。它的 `collective_rpc()` 本质是直接调用：

```text
run_method(self.driver_worker, method, args, kwargs)
```

没有真正 IPC，也没有多进程 message queue。

### Q25：MultiprocExecutor 的特点是什么？

答：MultiprocExecutor 是 V1 多 worker 的核心后端。它会启动多个本地 worker 子进程，通过 message queue 做控制面广播和响应收集。数据面仍在 worker 内部通过 torch distributed/NCCL/Gloo 等完成。

核心结构：

```text
Executor process
  rpc_broadcast_mq  --广播--> WorkerProc.worker_busy_loop()
  response_mqs      <--返回-- WorkerProc.handle_output()
```

### Q26：MultiprocExecutor 如何判断 driver worker？

答：driver worker 判断逻辑是：

```text
rank % tensor_parallel_size == 0
```

也就是每个 TP group 的 rank 0 是 driver worker。

### Q27：MultiprocExecutor 为什么通常只等待一个 output rank 的返回？

答：在多 rank 执行中，并不是所有 rank 都需要把完整输出返回给 EngineCore。比如 PP 场景下通常只有最后一个 pipeline rank 产生最终 `ModelRunnerOutput`，TP 场景下输出也可能只由某个 driver/output rank 汇总。因此 `execute_model()`/`sample_tokens()` 会设置 `unique_reply_rank=self.output_rank`，只等待一个 rank 的返回，以减少控制面开销。

### Q28：MultiprocExecutor 的 worker monitor 做什么？

答：worker monitor 监听 worker 进程 sentinel。如果任一 worker 异常死亡，monitor 会：

1. 标记 executor failed；
2. shutdown executor；
3. 调用 failure callback。

这是多进程后端的核心健康检查机制。

### Q29：RayDistributedExecutor 和 RayExecutorV2 有什么区别？

答：RayDistributedExecutor 更偏 Ray 原生 actor 和 compiled DAG 模式；RayExecutorV2 则让 Ray 负责 worker placement 和 actor 生命周期，但控制面复用 Multiproc 的 MessageQueue，数据面复用 NCCL/torch.distributed。

简化理解：

```text
RayDistributedExecutor：Ray actor / DAG 风格更强
RayExecutorV2：Ray 管资源，MQ 管控制面，NCCL 管数据面
```

### Q30：RayExecutorV2 的设计目标是什么？

答：目标是把 Ray 的职责收缩到 placement 和 actor lifecycle，而不是让 Ray 承担高频控制面/数据面通信。它通过 Ray 创建 actor、分配 GPU 和 local rank，但 worker 执行循环使用 MessageQueue，tensor 通信仍走 NCCL/torch.distributed。

### Q31：WorkerBase 定义了哪些统一接口？

答：`WorkerBase` 定义 worker 的统一接口，包括：

- `init_device()`
- `load_model()`
- `execute_model()`
- `sample_tokens()`
- `get_kv_cache_spec()`
- `compile_or_warm_up_model()`
- `get_cache_block_size_bytes()`
- LoRA 相关方法
- `shutdown()`

它还保存 `vllm_config`、`model_config`、`cache_config`、`parallel_config`、`device_config`、`local_rank`、`rank`、`distributed_init_method` 等。

### Q32：WorkerWrapperBase 的作用是什么？

答：WorkerWrapperBase 是 executor 和真实 worker 之间的延迟初始化包装层。它负责：

- 保存 rpc rank/global rank；
- 接收并设置环境变量；
- 根据 `parallel_config.worker_cls` 动态加载 worker class；
- 注入 worker extension class；
- 创建多模态 shared memory cache receiver；
- 在 `set_current_vllm_config()` 上下文中实例化真实 worker；
- `initialize_from_config()` 时按 global rank 取对应 KV cache config。

### Q33：GPU Worker 的生命周期包括哪些阶段？

答：主要阶段：

1. 构造：保存配置、Elastic EP executor、profiler、weight transfer engine 状态等。
2. `init_device()`：设置设备、初始化 distributed、创建 GPUModelRunner。
3. `load_model()`：加载模型并可初始化 weight transfer engine。
4. `determine_available_memory()`：profile 可用于 KV cache 的显存。
5. `initialize_from_config()`：分配 KV cache 并初始化 KV transfer。
6. `compile_or_warm_up_model()`：warmup、kernel warmup、CUDA graph capture。
7. `execute_model()`：执行一次 scheduler output。
8. `sample_tokens()`：执行采样。
9. `shutdown()`：清理资源。

### Q34：GPU Worker 的 `init_device()` 做什么？

答：它负责：

- 处理 Ray/NCCL 环境变量；
- 根据 DP local rank 调整 local rank；
- 设置 CUDA device；
- 检查 dtype 支持；
- 调用 `init_worker_distributed_environment()`；
- 设置随机种子；
- 清理 CUDA cache 并记录 memory snapshot；
- 初始化 workspace manager；
- 创建 GPUModelRunner V1 或 V2；
- rank 0 记录 usage stats。

### Q35：GPU Worker 的 `execute_model()` 如何处理 PP？

答：流程是：

1. 等待上一轮非阻塞 PP send 完成。
2. 如果当前 rank 不是 PP first rank，调用 `get_pp_group().irecv_tensor_dict()` 接收上游 intermediate tensors。
3. 将接收到的 tensor dict 包装为 `AsyncIntermediateTensors`。
4. 调用 `model_runner.execute_model()`。
5. 如果返回 `IntermediateTensors`，说明不是 PP last rank，调用 `get_pp_group().isend_tensor_dict()` 发给下游。
6. 如果返回 `ModelRunnerOutput`/`AsyncModelRunnerOutput`/`None`，直接返回给 executor。

## 4. Process Group 与 GroupCoordinator

### Q36：`parallel_state.py` 的核心职责是什么？

答：它是 vLLM 分布式运行时中枢，负责：

- 初始化 torch distributed；
- 创建 WORLD/TP/PP/DP/DCP/PCP/EP/EPLB groups；
- 封装 process group 为 `GroupCoordinator`；
- 提供 all-reduce、all-gather、reduce-scatter、broadcast、send/recv、tensor dict send/recv；
- 提供 message queue broadcaster；
- 提供 graph capture 上下文；
- 销毁分布式状态并释放显存。

### Q37：GroupCoordinator 解决了什么问题？

答：PyTorch `ProcessGroup` 只表示一组 rank 和 backend。`GroupCoordinator` 在此基础上统一管理：

- CPU group；
- device group；
- 当前 rank 在 group 内的位置；
- local rank 与设备；
- device communicator；
- message queue broadcaster；
- custom op collective；
- tensor dict 传输协议。

它是 vLLM 数据面的统一门面。

### Q38：GroupCoordinator 中 `rank_in_group` 和 `rank` 有什么区别？

答：`rank` 是全局 rank；`rank_in_group` 是当前 rank 在某个 group 内的局部编号。例如 PP group 里 `rank_in_group=0` 表示该 rank 是这个 PP group 的 first stage，但它的全局 rank 不一定是 0。

### Q39：`broadcast(src=...)` 中的 `src` 是全局 rank 吗？

答：不是。`GroupCoordinator.broadcast()` 中的 `src` 是 group 内 rank，内部会用：

```text
self.ranks[src]
```

转换成全局 rank，再传给 `torch.distributed.broadcast()`。

### Q40：`init_distributed_environment()` 做什么？

答：它初始化全局 torch distributed 环境与 WORLD group。主要步骤：

1. 根据 DP、多节点、external launcher、elastic EP 等调整 rank/world_size/init method。
2. 如果 torch distributed 未初始化，则调用 `torch.distributed.init_process_group()` 或 split_group 路径。
3. backend 不可用时 fallback 到 Gloo。
4. 创建 `_WORLD` GroupCoordinator。
5. 多节点 DP 下可能创建 `_INNER_DP_WORLD`。

### Q41：为什么 DP 或多节点时 rank/world_size 需要调整？

答：executor 内部的 world size 通常描述模型并行 worker 数；DP 或多节点场景下，全局 torch distributed 需要把不同 DP replica 或不同节点也纳入同一个全局 rank 空间。因此会执行类似：

```text
rank = data_parallel_rank * world_size + rank
world_size = parallel_config.world_size_across_dp
```

这样所有 DP replica 的 ranks 都能被统一管理。

### Q42：`initialize_model_parallel()` 创建哪些全局 group？

答：主要创建：

- WORLD group
- TP group
- DCP group
- PCP group
- PP group
- DP group
- EP group
- EPLB group

对应 getter 包括：`get_tp_group()`、`get_dcp_group()`、`get_pcp_group()`、`get_pp_group()`、`get_dp_group()`、`get_ep_group()`、`get_eplb_group()`。

### Q43：GroupCoordinator 的 `all_reduce()` 如何执行？

答：逻辑是：

1. group size 为 1，直接返回输入。
2. 如果启用 custom op，则调用：

```text
torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
```

3. 否则调用：

```text
self.device_communicator.all_reduce(input_)
```

### Q44：为什么 custom collective op 使用 group name 字符串？

答：Dynamo/custom op 不方便传 Python `GroupCoordinator` 对象，所以 vLLM 用 `group_name` 字符串作为桥。custom op 中通过 group name 查找全局 registry 中的 `GroupCoordinator`，再转发到实际 group 执行 collective。

### Q45：`broadcast_object()` 为什么优先使用 MQ？

答：object 通信通常是控制信息和 metadata，走共享内存/message queue broadcaster 可以减少 `torch.distributed.broadcast_object_list()` 的开销，并避免某些对象序列化/同步路径过慢。如果没有 MQ broadcaster，则 fallback 到 torch distributed object broadcast。

### Q46：`send_object()` / `recv_object()` 是如何实现的？

答：流程是：

1. pickle object；
2. 先发送 size；
3. 再发送 bytes tensor；
4. 使用 CPU group 通信。

这种路径适合小对象和元数据，不适合大 tensor。

### Q47：tensor dict 通信解决什么问题？

答：tensor dict 通信用于传输一组带 metadata 的 tensors，典型用于 PP stage 间传 intermediate tensors。它会先发送 metadata，再按 metadata 创建/复用 tensor，并分别用 CPU/GPU group 传输。它支持异步 send/recv 和部分 tensor 的 all-gather 优化。

### Q48：`isend_tensor_dict()` 和 `irecv_tensor_dict()` 返回什么？

答：`isend_tensor_dict()` 返回通信 handles，调用方后续需要 wait。

`irecv_tensor_dict()` 返回：

- `tensor_dict`
- `comm_handles`
- `comm_postprocess`

GPU Worker 会把它们包装成 `AsyncIntermediateTensors`，实现 lazy synchronization。

### Q49：`make_sibling_device_group()` 的作用是什么？

答：它创建与当前 group rank 成员相同、但使用独立 communicator 的 device process group。典型用途是 PP sampled token broadcast。原因是 sampled token broadcast 不应与 PP hidden-state p2p send/recv 共用 communicator，否则会串行阻塞甚至造成死锁风险。

### Q50：`graph_capture()` 在分布式通信中有什么用？

答：它用于 CUDA graph capture 场景，使 communicator 进入 capture context，并在指定 stream 上进行 capture。这样 collective 或通信相关操作可以和 CUDA graph 的 stream 语义对齐，避免 capture 时发生非法同步或不一致行为。

## 5. 运行时并行通信

### Q51：TP 运行时主要通信是什么？

答：TP 主要发生在模型层内部，包括：

- all-reduce：合并各 rank 的部分输出；
- all-gather：收集切分的张量；
- reduce-scatter：规约并切分结果；
- all-gatherv/reduce-scatterv：变长版本。

这些通信通常由模型层、parallel linear、attention/MLP 内部调用 `get_tp_group()` 触发。

### Q52：PP 有哪两条通信线？

答：PP 有两条通信线：

1. stage 间 intermediate tensors send/recv，用于把上一个 pipeline stage 的 hidden states 传给下一个 stage。
2. last PP rank sampled tokens 广播，用于把采样结果同步给前面 stages，以便它们更新下一步 decode 状态。

这两条线故意使用不同 communicator，避免互相阻塞。

### Q53：PP 中间张量如何接收？

答：如果当前 rank 不是 PP first rank，GPU Worker 会调用：

```text
get_pp_group().irecv_tensor_dict(
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)
```

返回 tensor dict 和通信 handles，然后包装成 `AsyncIntermediateTensors`，等模型真正访问 `.tensors` 时才等待通信完成。

### Q54：`AsyncIntermediateTensors` 的作用是什么？

答：它继承 `IntermediateTensors`，但带 lazy communication synchronization。它会在访问 `.tensors` 时才等待通信 handles 完成并执行 postprocess。好处是 PP recv 可以和部分输入准备、调度后处理等工作重叠，提高 pipeline 并行效率。

### Q55：PP 中间张量如何发送？

答：如果 `model_runner.execute_model()` 返回 `IntermediateTensors`，说明当前 rank 不是 PP last rank。GPU Worker 调用：

```text
get_pp_group().isend_tensor_dict(
    output.tensors,
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)
```

发送给下游 stage，并保存 handles 到 `_pp_send_work`。下一轮 `execute_model()` 开头会 wait 上一轮 send 完成。

### Q56：PP sampled token broadcast 为什么需要单独 sibling group？

答：因为 sampled token broadcast 和 hidden-state p2p 都发生在 PP ranks 之间。如果共用同一个 NCCL communicator，它们可能串行阻塞，甚至因调用顺序不一致导致卡住。单独 sibling group 可以隔离 sampled token broadcast 和 intermediate tensor p2p。

### Q57：PPHandler 的作用是什么？

答：`PPHandler` 管理 PP 下 sampled token 广播/接收。last PP rank 在采样后广播 sampled token ids 和 rejection info；非 last ranks 接收这些结果，并在后续 step 消费，用于更新 decode 状态。

### Q58：DP batch 协调和训练中的 DP 梯度同步一样吗？

答：不一样。vLLM 推理中的 DP 不是同步梯度，而是同步 batch 形态。它要让多个 DP rank 对 token 数、padding、ubatch、CUDA graph mode 等达成一致，避免后续 collective 或 CUDA graph shape 不匹配。

### Q59：`coordinate_batch_across_dp()` 做什么？

答：它协调 DP ranks 的 batch 形态。输入包括原始 token 数、是否允许 microbatching、padding 后 token 数、uniform decode、CUDA graph mode 等；输出包括：

- 是否启用 ubatch；
- padding 后 token 数；
- 同步后的 CUDA graph mode。

如果 `data_parallel_size == 1`，直接返回本地决策。

### Q60：DP 协调的 `_run_ar()` 同步哪些信息？

答：它构造一个 `[4, dp_size]` 的 int32 tensor：

- row 0：每个 DP rank 的原始 token 数；
- row 1：每个 DP rank 的 padded token 数；
- row 2：该 rank 是否希望 ubatch；
- row 3：该 rank 的 CUDA graph mode。

然后通过 `dist.all_reduce()` 同步。

### Q61：DP padding 什么时候需要？

答：当使用 CUDA graph 或启用 ubatching 时，需要让 DP ranks 的 token 数 padding 到一致或至少满足共同形态。否则不同 DP rank 的 shape 不一致，会导致 collective、CUDA graph replay 或后续张量操作出错。

### Q62：EP/MoE 通信的典型流程是什么？

答：MoE 的典型通信流程是：

```text
hidden states
  -> router logits
  -> dispatch_router_logits()
  -> 根据 routing 做 token dispatch
  -> 各 rank 本地 expert 计算
  -> combine() 合并 expert outputs
```

EP group 负责 dispatch/combine，实际 all2all backend 由 `ParallelConfig.all2all_backend` 和 device communicator 决定。

### Q63：`all2all_backend` 有什么作用？

答：`all2all_backend` 决定 MoE expert parallel 中 token dispatch/combine 的后端实现。例如 allgather+reduce-scatter、DeepEP 高吞吐/低延迟、MORI、NIXL、FlashInfer NVLink 等。不同 backend 对硬件拓扑、延迟、吞吐和 sequence parallel 支持不同。

### Q64：sequence parallel MoE 什么时候启用？

答：通常需要满足：

- expert parallel 开启；
- TP > 1；
- DP > 1；
- all2all backend 支持 sequence parallel。

作用是避免 TP attention 后重复计算，并改善 MoE 通信/计算布局。

### Q65：DCP/PCP 下 `cp_kv_cache_interleave_size` 是什么？

答：它表示在 DCP 或 PCP 下，KV cache 按 total CP rank 交错存储的粒度。相关概念：

```text
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

interleave size 可以是 token-level，也可以是 block-level，影响 KV cache 的分布方式。

### Q66：一次 `Worker.execute_model()` 中可能发生哪些通信？

答：可能发生：

1. 等待上一轮 PP send 完成。
2. PP 非 first rank 通过 `irecv_tensor_dict()` 接收上游 hidden states。
3. 模型执行中 TP all-reduce/all-gather/reduce-scatter。
4. MoE 层触发 EP dispatch/combine。
5. DP 环境下调用 `coordinate_batch_across_dp()`。
6. PP 非 last rank 通过 `isend_tensor_dict()` 发给下游。
7. PP last rank 采样后通过 PPHandler 广播 sampled tokens。
8. KV/EC connector 在 forward 前后执行 load/save。

## 6. KV/EC/Weight Transfer 与 Elastic EP

### Q67：KV Transfer 解决什么问题？

答：KV Transfer 用于跨 vLLM 实例传输 decoder KV cache，典型场景是 prefill/decode disaggregation。prefill 实例计算 prompt KV，decode 实例复用这些 KV，避免重复 prefill，提高吞吐和资源利用率。

### Q68：`KVTransferConfig` 的关键字段有哪些？

答：关键字段包括：

- `kv_connector`：connector 名称；
- `engine_id`：当前 engine 唯一 ID；
- `kv_buffer_device`：buffer 设备；
- `kv_buffer_size`：buffer 大小；
- `kv_role`：`kv_producer`、`kv_consumer`、`kv_both`；
- `kv_rank`：KV transfer rank；
- `kv_parallel_size`：KV transfer 并行大小；
- `kv_ip`、`kv_port`：连接地址；
- `kv_connector_extra_config`：私有配置；
- `kv_connector_module_path`：动态加载 connector；
- `enable_permute_local_kv`：HND/NHD 实验开关；
- `kv_load_failure_policy`：`recompute` 或 `fail`。

### Q69：KV Transfer 的 producer/consumer 角色是什么？

答：

- producer：产生并保存 KV cache，通常是 prefill 侧。
- consumer：加载并消费 KV cache，通常是 decode 侧。
- both：既可以生产也可以消费。

配置中 `kv_role` 可以是 `kv_producer`、`kv_consumer`、`kv_both`。

### Q70：KV Transfer 在 worker 侧什么时候初始化？

答：GPU Worker 在初始化 KV cache 前调用：

```text
ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
```

必须在 `model_runner.initialize_kv_cache(kv_cache_config)` 之前，因为 connector 需要知道 KV cache config，并在后续绑定真实 KV cache tensors。

### Q71：Scheduler 侧 KV connector 负责什么？

答：Scheduler 侧 connector 负责：

- 新请求通知 connector；
- 查询外部 KV 命中 token 数；
- 在本地分配 blocks 后更新 connector 状态；
- 构建 `kv_connector_metadata` 放入 `SchedulerOutput`；
- 处理 worker 返回的 transfer 完成/失败信息；
- 根据 load failure policy 选择 recompute 或 fail。

### Q72：Worker 侧 KV connector 负责什么？

答：Worker/model runner 侧 connector 负责：

- 注册真实 KV cache tensors；
- pre-forward load KV；
- post-forward save KV；
- 处理 preemptions；
- 汇报 finished sending/receiving；
- 汇报 invalid block ids；
- 返回 stats/events/worker metadata。

### Q73：KV connector handshake metadata 是什么？

答：它是 worker 侧 connector 提供给 scheduler 侧 connector 的握手信息。GPU Worker 返回：

```text
{(pp_rank, tp_rank): metadata}
```

EngineCore 初始化时从所有 worker 收集并合并这些 metadata，然后设置到 scheduler connector，让 scheduler 了解各 PP/TP rank 的 connector 上下文。

### Q74：KV Transfer 的运行时链路是什么？

答：简化链路：

```text
Request 进入 Scheduler
  -> connector on_new_request
  -> 查询本地 prefix cache
  -> connector.get_num_new_matched_tokens()
  -> KVCacheManager.allocate_slots(... external tokens ...)
  -> connector.update_state_after_alloc()
  -> SchedulerOutput.kv_connector_metadata
  -> Worker KVConnector.pre_forward() load KV
  -> model forward 或 no-forward
  -> Worker KVConnector.post_forward() save/finish/stats
  -> Scheduler.update_from_output() 处理完成、失败、invalid blocks
```

### Q75：`kv_load_failure_policy` 有哪两种策略？

答：

- `recompute`：load 失败时重新调度请求，重算失败 KV blocks。
- `fail`：load 失败时请求直接以 error finished。

Scheduler 中通过 `self.recompute_kv_load_failures` 决定行为。

### Q76：EC Transfer 和 KV Transfer 有什么区别？

答：KV Transfer 传输 decoder KV cache，主要服务 decoder-only 或 prefill/decode disaggregation。EC Transfer 传输 encoder cache，主要服务 encoder-decoder 或多模态场景中的 encoder outputs/cache 跨实例复用。

### Q77：Weight Transfer 用于什么？

答：Weight Transfer 用于从 trainer 或外部服务向 vLLM worker 热更新权重。典型用于在线训练/推理一体、RLHF/RL rollout、模型动态更新等场景。它支持 checkpoint format、sparse flat 权重、kernel format 直接 copy 等路径。

### Q78：Weight Transfer 的生命周期是什么？

答：

1. `init_weight_transfer_engine()`：初始化 transfer engine，NCCL backend 下可能与 trainer 建 process group。
2. `start_weight_update()`：开始一次权重更新 session。
3. `update_weights()`：接收一个权重 chunk，copy 或 broadcast 到模型参数。
4. `finish_weight_update()`：结束更新 session，必要时 finalize layerwise reload。

### Q79：Elastic EP 是什么？

答：Elastic EP 是弹性 expert parallel，用于动态扩缩 expert parallel groups。启用后，distributed init 会进入 Elastic EP world 初始化路径，model parallel group 中 DP/EP/EPLB 可能使用 stateless NCCL groups 和 TCPStore 协调。

### Q80：Elastic EP 有什么限制？

答：当前 Elastic EP 不支持 multi-node TP/PP。如果 TP/PP group 跨多节点，会报错。原因是 Elastic EP 的 stateless group 和动态扩缩设计目前主要针对局部 EP/DP 维度，不覆盖跨节点 TP/PP 的复杂通信拓扑。

### Q81：EPLB 是什么？

答：EPLB 是 expert parallel load balancing，用于 MoE expert 负载均衡。它统计 expert 负载并调整/平衡 expert 分布。它要求 expert parallel 开启，并且使用与 EP 相同 rank 集合但独立 process group，避免和 MoE forward collectives 互相死锁。

## 7. 常见排查类问题

### Q82：如果想知道为什么选择了某个 executor backend，应看哪里？

答：看三个点：

1. `ParallelConfig.distributed_executor_backend`；
2. `Executor.get_class()`；
3. `ParallelConfig.use_ray` 和相关环境变量，例如 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`。

这能解释为什么走 `uni`、`mp`、`ray`、Ray V2 或 external launcher。

### Q83：worker 没起来应该如何排查？

答：按 executor 类型看：

- Multiproc：看 `MultiprocExecutor._init_executor()` 和 `WorkerProc.wait_for_ready()`。
- Ray V2：看 `RayExecutorV2._init_executor()`、actor 创建和 `initialize_worker()`。
- Worker wrapper：看 `WorkerWrapperBase.init_worker()` 是否正确解析 worker class。
- GPU Worker：看 `Worker.init_device()` 中设备设置、distributed init、dtype 检查、GPUModelRunner 创建。

### Q84：torch.distributed 初始化在哪里？

答：worker 初始化设备时触发。路径是：

```text
Worker.init_device()
  -> init_worker_distributed_environment()
      -> init_distributed_environment()
      -> ensure_model_parallel_initialized()
```

其中 `init_distributed_environment()` 负责初始化 torch process group 和 WORLD group。

### Q85：TP/PP/DP group 怎么切，应该看哪里？

答：看 `initialize_model_parallel()`，重点看 `all_ranks` reshape 的维度：

```text
ExternalDP × DP × PP × PCP × TP
```

然后看各 group 的 `transpose/reshape` 公式。

### Q86：PP 卡住通常可能是什么原因？

答：常见原因：

- 某些 PP rank 没有进入同一个 step；
- `irecv_tensor_dict()` 和 `isend_tensor_dict()` 调用不匹配；
- PP sampled token broadcast 调用不匹配；
- sibling communicator 没有正确创建；
- 上一轮 `_pp_send_work` 未正确 wait；
- 非 last rank/last rank 对 `IntermediateTensors` 和 `ModelRunnerOutput` 的路径判断不一致。

排查文件：`gpu_worker.py`、`pp_utils.py`、`parallel_state.py` 的 tensor dict send/recv。

### Q87：DP 卡住通常可能是什么原因？

答：常见原因：

- 某些 DP rank 没有调用 `coordinate_batch_across_dp()`；
- DP ranks 使用的 CPU/GPU group 不一致；
- `disable_nccl_for_dp_synchronization` 配置不一致；
- CUDA graph mode 或 ubatch 决策不一致；
- DP rank 的 token/padding shape 差异没有同步。

排查 `vllm/v1/worker/dp_utils.py`。

### Q88：EP/MoE 卡住通常可能是什么原因？

答：常见原因：

- `enable_expert_parallel` 配置不正确；
- `all2all_backend` 不支持当前硬件或拓扑；
- EP group 没有正确创建；
- MoE dispatch/combine 调用顺序不一致；
- EPLB 与 MoE forward collectives 共用 group 导致死锁风险；
- sequence parallel MoE 的启用条件不满足。

排查 `parallel.py`、`parallel_state.py`、device communicator all2all、fused MoE 相关文件。

### Q89：KV transfer 没生效应该如何排查？

答：看：

1. `KVTransferConfig` 是否设置 `kv_connector` 和合法 `kv_role`；
2. worker 是否调用 `ensure_kv_transfer_initialized()`；
3. worker 是否返回 handshake metadata；
4. EngineCore 是否聚合 metadata 并设置到 scheduler connector；
5. Scheduler 是否创建 connector；
6. Scheduler 是否查询 external KV 命中；
7. `SchedulerOutput.kv_connector_metadata` 是否传到 worker；
8. worker pre_forward/post_forward 是否执行 load/save；
9. load failure policy 是否导致 recompute 或 fail。

### Q90：DCP/PCP 错误通常应该看什么？

答：看：

- `tp_size % dcp_size == 0` 是否满足；
- PCP 是否正确参与 `world_size = TP × PP × PCP`；
- `cp_kv_cache_interleave_size` 是否与 block size/attention 实现兼容；
- DCP/PCP group 创建是否正确；
- attention/KV cache 是否按 total CP rank 正确切分和交错。

## 8. 设计理解类问题

### Q91：为什么 vLLM 要区分 Executor 和 Worker？

答：Executor 是控制面编排层，关心 worker 创建、RPC、健康检查和输出收集；Worker 是执行面，关心设备、模型、KV cache、forward/sample 和数据面通信。这样可以让不同 executor backend 复用同一套 worker 执行逻辑，也让 UniProc/Multiproc/Ray/external launcher 只在编排层不同。

### Q92：为什么数据面通信放在 Worker 内，而不是 Executor 内？

答：数据面通信通常和 GPU tensor、模型 forward、process group、NCCL communicator 紧密绑定，必须发生在持有模型和设备上下文的 worker 进程中。Executor 只负责发控制命令，不持有完整模型执行上下文，也不适合参与高吞吐 collective。

### Q93：为什么 Ray V2 还要复用 MessageQueue，而不是全用 Ray RPC？

答：Ray RPC 更适合 actor 编排和任务调度，不适合高频、低延迟的 worker 控制循环。Ray V2 让 Ray 管 placement 和 actor 生命周期，控制面复用 Multiproc 的 MessageQueue，数据面复用 NCCL/torch.distributed，可以降低通信开销并保持与 Multiproc 一致的执行模型。

### Q94：为什么 PP sampled token 需要广播给非 last ranks？

答：PP 下只有 last rank 生成 logits 并采样，但前面 ranks 也维护下一步 decode 所需的 request/batch 状态。非 last ranks 需要知道 last rank 采样出的 token、接受/拒绝信息等，才能在下一步准备正确输入和状态，因此需要 sampled token broadcast。

### Q95：为什么 DP 推理也需要同步？

答：虽然推理没有梯度同步，但多个 DP ranks 可能仍参与共同的 batch/collective/CUDA graph 路径。如果各 rank 的 token 数、padding、ubatch、CUDA graph mode 不一致，后续 collective 或 graph replay 可能不匹配，导致错误或死锁。因此需要 DP batch 协调。

### Q96：为什么 KV Transfer 需要 Scheduler 和 Worker 两侧 connector？

答：Scheduler 负责逻辑决策：外部 KV 是否命中、需要为哪些 token 分配本地 blocks、如何处理 load 失败。Worker 负责实际数据操作：绑定真实 KV cache tensor、执行 load/save、报告完成/失败。二者职责不同，需要通过 `SchedulerOutput.kv_connector_metadata` 和 worker output 协作。

### Q97：为什么 EPLB 要求 expert parallel 开启？

答：EPLB 是 expert parallel load balancing，本质上服务 MoE experts 在 EP ranks 间的负载均衡。如果没有 expert parallel，就没有跨 rank expert 分布和负载平衡对象，因此启用 EPLB 没有意义，配置校验会禁止。

### Q98：为什么 GroupCoordinator 同时维护 CPU group 和 device group？

答：不同通信内容适合不同 backend：GPU tensor 通信应走 device group/NCCL；object、小 metadata、size header、某些 DP 同步可以走 CPU group/Gloo。GroupCoordinator 同时维护两者，可以根据通信类型选择最合适路径。

### Q99：为什么有 custom all-reduce/custom collective op？

答：custom collective op 可以在 torch compile/Dynamo/CUDA graph 场景中更稳定地表达 collective，并通过 group name 查找通信组，避免把 Python object 传入编译图。同时 custom all-reduce 也可能针对特定 GPU/NVLink 拓扑优化性能。

### Q100：vLLM 分布式通信层最核心的不变量是什么？

答：可以总结为 7 条：

1. `ParallelConfig` 决定拓扑和 executor backend。
2. Executor 负责 worker 生命周期和控制面 RPC。
3. Worker 负责设备初始化、模型加载和数据面通信。
4. `parallel_state.GroupCoordinator` 是 process group 和 collective 的统一 facade。
5. 大 tensor 通信不走 `collective_rpc`，而走 group/device communicator。
6. PP、DP、EP、KV/EC transfer 各自有专用通信路径。
7. Ray V2 尽量只让 Ray 管 placement/actor，通信协议复用 MQ + NCCL。

## 9. 可追问的代码定位题

### Q101：如果要看 executor backend 选择代码，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:48` 的 `Executor.get_class()`，以及 `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:240` 的 `distributed_executor_backend`。

### Q102：如果要看 process group 创建，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1674` 的 `initialize_model_parallel()`。

### Q103：如果要看 WORLD group 和 torch distributed 初始化，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1516` 的 `init_distributed_environment()`。

### Q104：如果要看 PP tensor p2p，看哪里？

答：看：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:853` 的接收路径；
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:889` 的发送路径；
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:964` 的 `isend_tensor_dict()`；
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1059` 的 `irecv_tensor_dict()`。

### Q105：如果要看 PP sampled token broadcast，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:51` 的 `PPHandler`。

### Q106：如果要看 DP batch 协调，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:164` 的 `coordinate_batch_across_dp()`。

### Q107：如果要看 MoE/EP dispatch/combine，看哪里？

答：看：

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1186` 的 `dispatch_router_logits()`；
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1206` 的 `dispatch()`；
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1228` 的 `combine()`。

### Q108：如果要看 KV transfer 配置，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:23` 的 `KVTransferConfig`。

### Q109：如果要看 KV transfer worker 初始化，看哪里？

答：看 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:575`，这里在初始化 KV cache 前调用 `ensure_kv_transfer_initialized()`。

### Q110：如果要看 Weight Transfer 生命周期，看哪里？

答：看 GPU Worker 中：

- `init_weight_transfer_engine()`；
- `start_weight_update()`；
- `update_weights()`；
- `finish_weight_update()`。

对应路径是 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py`。

## 10. 总结性回答模板

### Q111：请用一段话概括 vLLM 分布式通信层。

答：vLLM 的分布式通信层由配置、控制面、执行面、process group、device communicator 和扩展传输层组成。`ParallelConfig` 把 TP/PP/DP/PCP/DCP/EP/EPLB 等并行维度编译成 worker 数量、rank 拓扑和 executor backend；`Executor` 负责创建 worker 并通过 `collective_rpc` 发控制命令；`Worker` 在各 rank 上初始化 torch distributed、模型并行组和 GPUModelRunner；运行时大 tensor 通信通过 `parallel_state.GroupCoordinator` 和 device communicator 完成，包括 TP collective、PP p2p、DP batch 协调、EP all2all；KV/EC/weight transfer、Elastic EP 和 EPLB 是围绕模型执行扩展的专用通信通道。

### Q112：请用一段话说明 vLLM 中为什么要把控制面和数据面分开。

答：控制面和数据面的需求不同。控制面需要可靠地把方法调用、调度输出、配置、状态管理命令发到所有 worker，适合用 executor 的 RPC/message queue/Ray actor 实现；数据面需要传输大 tensor 或执行高频 collective，要求低延迟、高带宽、GPU-aware，必须走 NCCL/Gloo/device communicator/tensor dict p2p/all2all 等路径。分离后，vLLM 可以灵活支持 UniProc、Multiproc、Ray 等不同编排方式，同时保持高性能 tensor 通信不被控制面拖慢。

### Q113：请用一段话说明 PP 的通信设计。

答：PP 把模型层切成多个 stage，前向时非 first stage 先异步接收上游 intermediate tensors，执行本 stage 后，如果不是 last stage，则异步发送 intermediate tensors 给下游；last stage 负责生成 logits 和采样。由于前面 stages 也需要知道采样结果来更新下一步 decode 状态，last stage 还会通过 `PPHandler` 使用 sibling communicator 广播 sampled tokens。hidden-state p2p 和 sampled token broadcast 分开 communicator，是为了避免同一个 NCCL communicator 上不同通信线互相阻塞。

### Q114：请用一段话说明 DP 推理通信设计。

答：vLLM 中 DP 推理不是同步梯度，而是同步 batch 形态。多个 DP rank 可能各自处理不同请求，但为了后续 collective、CUDA graph replay、ubatching 等路径一致，需要同步每个 rank 的 token 数、padding 后 token 数、是否启用 ubatch、CUDA graph mode 等。`coordinate_batch_across_dp()` 通过 DP group 做 all-reduce 汇总这些小型 metadata，然后统一决策 padding、ubatch 和 graph mode。

### Q115：请用一段话说明 KV Transfer 的设计。

答：KV Transfer 用于跨实例复用 decoder KV cache，典型场景是 prefill/decode disaggregation。Scheduler 侧 connector 负责逻辑决策：查询外部 KV 命中、分配本地接收 blocks、把 connector metadata 放入 SchedulerOutput、处理 load 失败；Worker 侧 connector 负责真实数据操作：绑定 KV cache tensors、在 forward 前 load KV、forward 后 save KV、汇报完成/失败。二者通过 SchedulerOutput 和 ModelRunnerOutput 协作，让外部 KV cache 像本地 prefix cache 一样参与调度。
