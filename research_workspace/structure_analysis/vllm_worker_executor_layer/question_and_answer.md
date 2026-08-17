# vLLM Worker / Executor 执行层技术点问答整理

本文基于本目录已有 Worker / Executor 执行层分析文档，整理适合技术考察、代码复盘、源码提问的技术点，并给出参考答案。

## 一、执行层整体架构

### 1. vLLM V1 Worker / Executor 执行层在整体架构中处于什么位置？

Worker / Executor 执行层位于 `EngineCore/Scheduler` 和真实模型执行之间，用来把调度结果转换成设备侧模型执行。

主链路是：

```text
EngineCore
  -> Scheduler
  -> SchedulerOutput
  -> Executor
  -> WorkerWrapperBase
  -> Worker
  -> ModelRunner
  -> Model / Attention / CUDA kernels
```

`EngineCore` 负责调度与编排，`Executor` 负责 worker 生命周期和 RPC/IPC 分发，`WorkerWrapperBase` 是 executor 看到的 worker 壳，真实 `Worker` 负责设备、模型、KV cache 生命周期，`ModelRunner` 负责 forward、sampling、KV/LoRA/spec decode 等执行细节。

### 2. Worker / Executor 执行层的核心职责是什么？

核心职责是：把 `SchedulerOutput` 以统一接口分发到正确的 worker/rank，在 worker 上完成设备初始化、KV cache、模型 forward、采样与分布式通信，再把 `ModelRunnerOutput` 聚合回 `EngineCore`。

它不只是调用模型 forward，还包括：

- worker 创建与生命周期管理；
- 单进程、多进程、Ray 等 executor 后端适配；
- KV cache spec 收集、显存估算、KV tensor 分配；
- PP/TP/DP/EP 分布式通信初始化；
- LoRA、KV transfer、spec decode、structured output；
- CUDA graph capture、profiler、weight transfer；
- worker failure handling 和 shutdown。

### 3. 执行层中 Executor、WorkerWrapper、Worker、ModelRunner 如何分工？

可以分四层理解：

| 对象 | 定位 | 主要职责 |
|---|---|---|
| `Executor` | 控制层 | 选择后端、创建 worker、RPC/IPC、fan-out/fan-in、健康检查、shutdown |
| `WorkerWrapperBase` | 进程级代理 | 延迟构造真实 worker、注入环境变量、加载插件、转发方法、多模态 cache |
| `Worker` | 设备生命周期层 | 初始化 device/distributed、加载模型、profile、KV cache、PP send/recv、LoRA API |
| `ModelRunner` | 模型执行层 | batch 状态、输入准备、forward、logits、sampling、KV connector、LoRA active mapping |

一句话：Executor 管 worker，Worker 管设备和生命周期，ModelRunner 管真实模型执行。

### 4. 执行层有哪些 Executor 后端？

常见后端包括：

| Executor | 资源模型 | 控制面 | 执行面 | 用途 |
|---|---|---|---|---|
| `UniProcExecutor` | 单进程单 worker | 直接函数调用 | 本进程执行 | 最简单基线 |
| `MultiprocExecutor` | 本地多进程 | MessageQueue 广播 + response MQ | 所有 worker 执行，指定 rank 回包 | 本地多 GPU |
| `RayDistributedExecutor` | Ray actor | Ray remote call | Ray compiled DAG | 旧 Ray 后端 |
| `RayExecutorV2` | Ray actor | 复用 Multiproc MQ | Ray 调度资源，vLLM 控制面 | 新 Ray 后端 |

### 5. Worker 类型有哪些？

V1 中真实执行 worker 主要包括：

- GPU/CUDA/ROCm：`vllm.v1.worker.gpu_worker.Worker`；
- CPU：`CPUWorker(Worker)`；
- XPU：`XPUWorker(Worker)`。

注意 GPU worker 类名实际叫 `Worker`，不是 `GPUWorker`。CPUWorker 和 XPUWorker 都继承 GPU worker 的大部分逻辑，只覆盖平台相关初始化、内存 profiling、model runner、profiler 等差异。

### 6. 为什么 CPUWorker 和 XPUWorker 要继承 GPU worker？

因为三者在执行层的大部分生命周期逻辑是一致的：init worker、load model、KV cache、execute_model、sample_tokens、LoRA、profiler、shutdown 等。

CPU/XPU 只需要覆盖平台相关部分，例如 device 初始化、distributed backend、内存统计、model runner、profiler activity、部分 CUDA API 替换等。这样可以最大限度复用 V1 Worker 主逻辑。

## 二、Executor 抽象与后端选择

### 7. Executor 的核心定位是什么？

`Executor` 是 V1 执行层的上层抽象，面向 `EngineCore` 提供统一接口，屏蔽单进程、多进程、Ray 等 worker 部署方式。

它的职责不是模型计算，而是执行控制面：

- 根据配置选择执行后端；
- 初始化 worker；
- 下发 KV cache 配置；
- 对 worker 做 collective RPC；
- 执行模型调用；
- 执行采样调用；
- 健康检查；
- shutdown。

### 8. Executor.get_class 如何选择后端？

`Executor.get_class()` 根据 `parallel_config.distributed_executor_backend` 选择具体实现：

```text
backend == "ray"
  -> VLLM_USE_RAY_V2_EXECUTOR_BACKEND ? RayExecutorV2 : RayDistributedExecutor

backend == "mp"
  -> MultiprocExecutor

backend == "uni"
  -> UniProcExecutor

backend == "external_launcher"
  -> ExecutorWithExternalLauncher
```

因此 executor 后端由并行配置和环境变量共同决定。

### 9. EngineCore 初始化时如何使用 Executor？

典型顺序如下：

```text
EngineCore.__init__
  -> Executor.get_class(vllm_config)
  -> executor = executor_class(vllm_config, failure_callback)
  -> executor.get_kv_cache_specs()
  -> executor.determine_available_memory()
  -> EngineCore 计算 KVCacheConfig
  -> executor.initialize_from_config(kv_cache_configs)
  -> executor.compile_or_warm_up_model()
```

也就是说，KV cache 配置不是 worker 自己决定，而是由 EngineCore 综合 worker spec、profile 结果和配置计算后，再通过 executor 下发。

### 10. collective_rpc 是什么？

`collective_rpc()` 是 Executor 最核心的抽象，表示在 worker 集合上调用某个方法，并按 executor 后端规则收集结果。

不同实现：

- UniProc：本进程直接调用；
- Multiproc：向 broadcast MQ 写 RPC 请求，worker 从 MQ 消费；
- RayDistributedExecutor：Ray actor remote call；
- RayExecutorV2：复用 Multiproc 的 MQ 机制，但 worker 容器是 Ray actor。

### 11. Executor.execute_model 的默认语义是什么？

抽象层默认实现是对 collective RPC 的包装：

```text
Executor.execute_model(scheduler_output, non_block)
  -> collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
  -> 默认取 output[0]
```

但很多具体 executor 会覆盖它，用于处理 output rank、KV connector 输出聚合、Ray DAG、两阶段执行状态等。

### 12. Executor.sample_tokens 的默认语义是什么？

`sample_tokens()` 也是 collective RPC 包装：

```text
Executor.sample_tokens(grammar_output, non_block)
  -> collective_rpc("sample_tokens", args=(grammar_output,), non_block=non_block)
```

它通常在 `execute_model()` 返回 `None` 后被 EngineCore 调用，用于执行结构化输出约束和采样。

### 13. 多 rank 下为什么只需要 output_rank 回包？

多 rank 中所有 rank 都要执行 forward，因为 TP/PP 都参与模型计算。但最终 `ModelRunnerOutput` 通常只需要最后一个 pipeline stage 的某个 TP rank 返回。

因此 Multiproc 使用 `output_rank` 指定回包 rank，避免所有 rank 都把控制面结果返回给 executor，减少结果聚合开销。

### 14. output_rank 通常如何理解？

Multiproc 中：

```text
output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size
```

语义是：最后一个 pipeline stage 的第一个 TP rank，负责返回最终 `ModelRunnerOutput`。

### 15. Executor 与 async scheduling 有什么关系？

部分 executor 声明支持 async scheduling，例如：

- `UniProcExecutor`；
- `MultiprocExecutor`；
- `RayExecutorV2`。

async scheduling 会影响：

- `execute_model(..., non_block=True)` 更重要；
- worker 可能返回 `AsyncModelRunnerOutput`；
- Multiproc worker 内部可能启动 async output copy thread；
- DP synchronization 默认可能不用 NCCL，而改走 Gloo，降低异步多流场景风险。

## 三、UniProcExecutor

### 16. UniProcExecutor 的特点是什么？

`UniProcExecutor` 是最简单的 executor：

- 单进程；
- 单 worker；
- 不启动子进程；
- 不创建 Ray actor；
- `collective_rpc()` 实际就是本进程函数调用；
- 仍然遵守 executor 抽象接口。

它是理解 Executor/Worker 边界的最佳起点。

### 17. UniProcExecutor 初始化流程是什么？

大致流程：

```text
UniProcExecutor._init_executor
  -> 创建 WorkerWrapperBase(rpc_rank=0)
  -> 生成 distributed_init_method
  -> 构造 worker kwargs
  -> 设置网络设备环境变量
  -> driver_worker.init_worker(...)
  -> driver_worker.init_device()
  -> driver_worker.load_model()
```

即使单进程，也会通过 WorkerWrapperBase 延迟解析真实 worker class。

### 18. UniProc 为什么仍构造 distributed_init_method？

虽然 UniProc 是单 worker，但仍构造 distributed init method，是为了让单 worker 路径和多 worker 路径尽量复用同一套 Worker 分布式初始化逻辑。

这能减少单进程和多进程路径的语义差异。

### 19. UniProc 的 collective_rpc 如何工作？

同步模式可以理解为：

```text
collective_rpc(method, args, kwargs, non_block=False)
  -> run_method(self.driver_worker, method, args, kwargs)
  -> 如返回 AsyncModelRunnerOutput，则 get_output()
  -> 返回结果
```

这里没有真实 RPC，只是按方法名在 `WorkerWrapperBase` 上调用对应方法。

### 20. UniProc 中 non_block Future 有什么意义？

EngineCore 通常以非阻塞形式调用：

```text
future = model_executor.execute_model(scheduler_output, non_block=True)
```

UniProc 虽然是直接调用，但仍然需要返回 Future 风格对象，以保持 executor 接口统一。普通结果会包装为已完成 Future；`AsyncModelRunnerOutput` 会包装为 `AsyncOutputFuture`，在 `result()` 时调用 `get_output()`。

### 21. ExecutorWithExternalLauncher 是什么？

`ExecutorWithExternalLauncher` 用于外部进程启动方式，例如 torchrun 或其他 launcher 已设置好：

- `RANK`；
- `LOCAL_RANK`；
- `MASTER_ADDR`；
- `MASTER_PORT`。

它使用 `env://` 初始化，不由 vLLM executor 自己创建所有 worker 进程，每个外部进程内行为接近 UniProc。

## 四、MultiprocExecutor

### 22. MultiprocExecutor 的核心定位是什么？

`MultiprocExecutor` 是本地多进程执行器，用于单机多 worker / 多 GPU 场景。它通过 MessageQueue 做控制面广播，通过 worker response queue 收敛结果。

它负责：

- 拉起 worker 子进程；
- 建立广播 MQ 和 response MQ；
- 等待 worker ready；
- 执行 collective RPC；
- 管理 output rank；
- 支持 non-block Future；
- 监控 worker 异常退出；
- 关闭 worker 与 MQ。

### 23. MultiprocExecutor 初始化流程是什么？

大致流程：

```text
MultiprocExecutor._init_executor
  -> 计算 world_size / local_world_size / global_start_rank
  -> 创建 rpc_broadcast_mq
  -> 创建 worker response MQ
  -> 为每个 local_rank 创建 WorkerProc
  -> 启动 worker 进程
  -> 等待 worker ready
  -> 设置 driver worker / output rank
  -> 启动 monitor thread
```

### 24. Multiproc 中 rank 和 local_rank 如何理解？

```text
local_rank = 当前节点内 worker 序号
global_rank = global_start_rank + local_rank
global_start_rank = local_world_size * node_rank_within_dp
```

用途：

- `rank` 用于 distributed 全局通信；
- `local_rank` 用于本地设备选择；
- `is_driver_worker = rank % tensor_parallel_size == 0`。

### 25. Multiproc 的 MessageQueue 有哪几类？

两类核心 MQ：

1. `rpc_broadcast_mq`
   - executor 写入；
   - 所有 worker 读取；
   - 用于广播方法调用。

2. `worker_response_mq`
   - 每个 worker 一个或一组；
   - worker 写结果；
   - executor 读取结果。

跨节点 DP 内部场景中，worker 还可能通过 inner DP group 创建跨节点 MQ broadcaster。

### 26. Multiproc collective_rpc 的 fan-out / fan-in 流程是什么？

流程：

```text
collective_rpc(method, args, kwargs, output_rank)
  -> 将 (method, args, kwargs, output_rank) 写入 rpc_broadcast_mq
  -> 所有 worker 收到 RPC
  -> 所有 worker 执行方法
  -> 如果 output_rank 指定，只等待该 rank response
  -> 否则收集多个 rank response
  -> 检查 SUCCESS / FAILURE
  -> 返回结果或 FutureWrapper
```

关键点：Multiproc 是“广播调用，指定 rank 回包”。

### 27. 为什么所有 worker 都要执行同一个 RPC？

因为 TP/PP/DP 等模型并行都需要多个 worker 共同参与计算。即使只有一个 output rank 返回结果，其他 rank 也必须执行 forward、通信、KV cache 更新等操作，否则模型并行会挂起或结果错误。

### 28. MultiprocExecutor.execute_model 如何工作？

```text
MultiprocExecutor.execute_model(scheduler_output)
  -> collective_rpc("execute_model", args=(scheduler_output,), unique_reply_rank=output_rank)
  -> 若有 KV connector，可能通过 kv_output_aggregator 聚合
```

`sample_tokens()` 也走同样模式，只是调用方法变成 `sample_tokens`。

### 29. WorkerProc 是什么？

`WorkerProc` 是每个子进程的宿主，负责：

- 创建进程；
- 在子进程中构造 `WorkerWrapperBase`；
- 初始化 worker；
- 进入 busy loop；
- 消费 RPC；
- 执行 worker 方法；
- 写回 response；
- 捕获异常；
- 处理父进程死亡；
- shutdown 时销毁分布式环境。

### 30. FutureWrapper 的作用是什么？

`FutureWrapper` 用于 non-block RPC：

- 保存取结果函数；
- `result()` 时才等待 MQ response；
- 让 EngineCore 可以在模型执行期间做 grammar bitmask 等工作。

这和 EngineCore 的 `execute_model(..., non_block=True)` 配合使用。

### 31. Multiproc 的 async output copy 是什么？

async scheduling 开启时，worker 可能返回 `AsyncModelRunnerOutput`。这时 worker 不立即把最终 CPU 可见结果写 response MQ，而是放入 async output queue，由后台线程：

```text
设置当前 CUDA device
  -> 等待 GPU->CPU copy 完成
  -> 调用 get_output()
  -> 写 response MQ
```

这样可以避免主执行流阻塞在输出拷贝上。

### 32. Multiproc 如何处理 worker 失败？

主要有两类机制：

1. monitor thread 等待 worker process sentinel。任一 worker 异常退出，会设置 `is_failed=True`，shutdown executor，并触发 failure callback。
2. worker busy loop 捕获内部异常后，通过 response MQ 返回 FAILURE，executor `get_response()` 收到后抛 `RuntimeError`。

另外 worker 持有 death pipe，父进程退出时会收到 EOF 并退出。

### 33. Multiproc shutdown 流程是什么？

关闭流程：

```text
MultiprocExecutor.shutdown
  -> 关闭 worker death_writer，通知子进程退出
  -> 等待 graceful shutdown
  -> 超时 SIGTERM
  -> 再超时 SIGKILL
  -> 关闭 response MQ
  -> 关闭 broadcast MQ
```

worker 子进程关闭时会关闭 MQ、调用 `worker.shutdown()`、销毁 model parallel、销毁 distributed environment。

## 五、Ray 执行器

### 34. vLLM V1 中有哪两套 Ray executor？

两套：

1. `RayDistributedExecutor`：旧 Ray 后端，Ray actor + Ray remote control plane + Ray compiled DAG execution plane。
2. `RayExecutorV2`：新 Ray 后端，Ray actor 只负责资源调度和 worker 容器，控制面复用 Multiproc 的 MessageQueue。

两者架构差异很大，不能简单认为 Ray v2 是 Ray v1 的小改版。

### 35. RayDistributedExecutor 的职责是什么？

旧 Ray 后端负责：

- 初始化 Ray；
- 创建 Ray actor worker；
- 使用 placement group 做资源放置；
- 为每个 worker 设置 rank/local_rank；
- 通过 Ray remote call 执行控制面 RPC；
- 用 Ray compiled DAG 执行模型路径。

### 36. 旧 Ray executor 为什么要自己设置 CUDA_VISIBLE_DEVICES？

旧 Ray executor 不让 Ray 自动设置 worker 可见 GPU，而是 vLLM 根据 Ray 分配的 GPU ID 自己设置。

原因：同一节点内一个 vLLM executor 可能占用多个 GPU，每个 worker 需要看到本 executor 在该节点上的所有 GPU，然后再根据 `local_rank` 选择自己的 device。

### 37. RayDistributedExecutor 的控制面 RPC 如何工作？

控制面直接调用 Ray actor 方法：

```text
worker.execute_method.remote(method, *args, **kwargs)
```

这不是 MessageQueue，而是 Ray remote call。

### 38. 旧 Ray executor 的 Ray compiled DAG 是什么？

模型执行路径通过 Ray compiled DAG 表达：

```text
input: (SchedulerOutput, GrammarOutput)
  -> fan-out 到第一 PP stage 的 TP workers
  -> 每个 PP stage 内 TP group SPMD 执行
  -> 非最后 PP stage 输出 IntermediateTensors
  -> 下一个 PP stage 接收 IntermediateTensors
  -> 最后 stage 采样并返回 ModelRunnerOutput
```

### 39. RayDistributedExecutor 中 execute_model / sample_tokens 为什么有特殊状态机？

如果本 step 需要采样，旧 Ray executor 可能在 `execute_model()` 中缓存 `scheduler_output`，返回 `None` 或 completed none future，然后等 `sample_tokens(grammar_output)` 时再执行 DAG。

这是因为 Ray DAG 输入同时需要 `SchedulerOutput` 和 `GrammarOutput`。

### 40. Ray worker wrapper 的 execute_model_ray 做什么？

`execute_model_ray()` 是 Ray DAG 中 actor 的执行入口，输入可能是：

```text
(SchedulerOutput, GrammarOutput)
(SchedulerOutput, GrammarOutput, IntermediateTensors)
```

流程：

```text
设置 device
  -> 解包 scheduler_output / grammar_output / intermediate_tensors
  -> worker.model_runner.execute_model(...)
  -> 如果返回 IntermediateTensors，传给下一 PP stage
  -> 如果最后 rank 且 execute_model 返回 None，则调用 sample_tokens(grammar_output)
  -> 如果是 AsyncModelRunnerOutput，先 get_output()
```

### 41. RayExecutorV2 的核心思想是什么？

RayExecutorV2 的核心思想是：

> Ray 负责资源调度和 actor 容器；vLLM 继续使用 Multiproc 的 MQ 控制面和自己的 distributed 通信。

所以 Ray v2 更接近 MultiprocExecutor，只是 worker 容器从本地子进程变成 Ray actor。

### 42. RayExecutorV2 初始化流程是什么？

大致流程：

1. 初始化 Ray cluster；
2. 根据 placement group bundle 生成 rank 到 bundle/node 映射；
3. 选择 rank 0 所在节点 IP 作为 torch distributed init TCP 地址；
4. driver 创建 broadcast MQ；
5. 为每个 rank 创建 `RayWorkerProc` actor；
6. actor 查询 Ray runtime 分配的 node_id/gpu_ids；
7. 计算每个节点内 local_rank，并设置 visible devices；
8. 收集 response MQ handle；
9. 调用 actor `run.remote()` 启动 busy loop；
10. driver 与 workers 执行 MQ ready barrier。

### 43. RayWorkerProc 为什么要两阶段初始化？

Ray actor 构造时不能立刻做完整 worker 初始化，因为 CUDA device / visible devices 需要等 Ray placement 完成。

所以分两阶段：

```text
Ray actor __init__
  -> 只保存参数

initialize_worker
  -> 获取 Ray 分配的 GPU/node 信息
  -> 设置 visible devices
  -> 调 WorkerProc.__init__
  -> 初始化真实 worker
```

### 44. RayDistributedExecutor 和 RayExecutorV2 有什么区别？

| 方面 | RayDistributedExecutor | RayExecutorV2 |
|---|---|---|
| 继承关系 | 直接继承 Executor | 继承 MultiprocExecutor |
| worker 容器 | Ray actor | Ray actor |
| 控制面 | Ray remote call | Multiproc MQ |
| 执行面 | Ray compiled DAG | vLLM distributed + worker execute_model |
| PP 传递 | Ray DAG tensor transport | vLLM PP communicator |
| failure | 健康检查偏乐观 | ray.wait 监控 actor run |
| async scheduling | 非主支持路径 | 支持 |

## 六、WorkerBase 与 WorkerWrapper

### 45. WorkerBase 和 WorkerWrapperBase 有什么区别？

- `WorkerBase`：真实 worker 的抽象基类，定义硬件无关接口。
- `WorkerWrapperBase`：executor 直接持有和调用的 wrapper，负责延迟创建真实 worker，并转发生命周期方法。

Executor 通常不直接操作真实 GPU/CPU/XPU worker，而是通过 wrapper 间接调用。

### 46. WorkerBase 保存哪些关键配置？

初始化时保存：

- `model_config`；
- `cache_config`；
- `lora_config`；
- `load_config`；
- `parallel_config`；
- `scheduler_config`；
- `device_config`；
- `speculative_config`；
- `observability_config`；
- `kv_transfer_config`；
- `compilation_config`。

同时记录 `local_rank`、`rank`、`distributed_init_method`、`is_driver_worker`、`device`、`model_runner`。

### 47. WorkerBase 定义了哪些核心接口？

核心接口包括：

- `get_kv_cache_spec()`：返回模型 KV cache 规格；
- `compile_or_warm_up_model()`：编译、kernel warmup、CUDA graph capture；
- `init_device()`：初始化 device、distributed environment、model runner；
- `load_model()`：加载模型权重；
- `execute_model()`：执行一次 scheduler output；
- `sample_tokens()`：当 execute_model 返回 None 后执行采样；
- LoRA API：`add_lora`、`remove_lora`、`pin_lora`、`list_loras`。

### 48. WorkerWrapperBase 的职责是什么？

`WorkerWrapperBase` 是 executor 直接操作的进程级代理对象，职责包括：

- 延迟构造真实 worker；
- 加载插件；
- 注入每个 rank 的环境变量；
- 支持 worker extension；
- 管理多模态 receiver cache；
- 转发 `init_device`、`load_model`、`execute_model` 等方法。

### 49. WorkerWrapperBase.init_worker 的流程是什么？

流程：

```text
init_worker(all_kwargs)
  -> 根据 rpc_rank 取当前 rank kwargs
  -> 保存 vllm_config
  -> 加载 general plugins
  -> 从 parallel_config.worker_cls 解析 worker 类
  -> 如有 worker_extension_cls，动态扩展类
  -> 初始化 multimodal receiver cache
  -> set_current_vllm_config
  -> 实例化真实 worker
```

这解释了为什么 executor 初始化时通常只知道 wrapper，而不是直接知道 GPU/CPU/XPU worker。

### 50. worker_cls 是如何映射到真实 worker 的？

真实 worker class 由平台配置决定，常见映射：

- CUDA：`vllm.v1.worker.gpu_worker.Worker`；
- ROCm：`vllm.v1.worker.gpu_worker.Worker`；
- CPU：`vllm.v1.worker.cpu_worker.CPUWorker`；
- XPU：`vllm.v1.worker.xpu_worker.XPUWorker`。

### 51. WorkerWrapperBase.initialize_from_config 做什么？

Executor 计算完每个 worker 的 KV cache config 后，会调用 wrapper 的 `initialize_from_config()`。

wrapper 行为：

```text
initialize_from_config(kv_cache_configs)
  -> 根据 self.global_rank 取当前 worker 的 KVCacheConfig
  -> self.worker.initialize_from_config(kv_cache_config)
```

即 wrapper 做 rank 选择和转发，真实 KV cache 初始化由 worker/model runner 完成。

### 52. WorkerWrapperBase.execute_model 为什么不只是简单转发？

它会先应用多模态 receiver cache：

```text
execute_model(scheduler_output)
  -> _apply_mm_cache(scheduler_output)
  -> self.worker.execute_model(scheduler_output)
```

因此 wrapper 也承担了一些请求数据进入 worker 前的适配逻辑。

## 七、GPU / CPU / XPU Worker 生命周期

### 53. GPU Worker 的主要职责是什么？

GPU Worker 负责：

- 初始化 CUDA/ROCm device；
- 初始化 distributed environment；
- 创建 GPUModelRunner；
- 加载模型；
- profile 可用显存；
- 初始化 KV cache；
- 编译、warmup、CUDA graph capture；
- 执行模型；
- 执行采样；
- 处理 pipeline parallel intermediate tensors；
- 管理 LoRA API；
- 管理 KV transfer / EC transfer；
- profiler；
- weight transfer；
- shutdown。

### 54. GPU Worker.__init__ 做什么？

初始化阶段主要保存状态，不立即创建 device/model。

关键内容包括：

- 调用 `WorkerBase.__init__()`；
- 设置 float32 matmul precision；
- 创建 ElasticEPScalingExecutor；
- 初始化 sleep mode buffer；
- 初始化 weight transfer 状态；
- 保存 profiler config；
- 判断是否使用 V2 model runner；
- 初始化 PP 非阻塞 send 状态。

### 55. GPU Worker.init_device 的流程是什么？

流程：

```text
Worker.init_device
  -> 校验 device_type == cuda
  -> 清理 Ray 可能设置的 NCCL_ASYNC_ERROR_HANDLING
  -> DP 单节点场景修正 local_rank
  -> 设置 self.device = cuda:{local_rank}
  -> torch.accelerator.set_device_index
  -> 检查 dtype 支持
  -> 初始化 distributed environment
  -> 设置随机种子
  -> GC + empty_cache
  -> 记录 MemorySnapshot
  -> 计算 requested_memory
  -> 初始化 workspace manager
  -> 创建 GPUModelRunner 或 GPUModelRunnerV2
  -> rank 0 上报 usage stats
```

### 56. GPU Worker 为什么在内存快照前初始化 distributed environment？

因为 NCCL/Gloo 等通信 buffer 会占用显存，这些显存不应该被 KV cache 使用。

所以 vLLM 先初始化 distributed environment，再记录 memory snapshot 和 profile 可用 KV cache 显存，避免 KV cache 分配过量导致后续 OOM。

### 57. DP 场景下 local_rank 为什么要修正？

单节点 DP 且不是 Ray/external launcher 时，会把 DP local rank 纳入 local_rank：

```text
local_rank += dp_local_rank * (PP * TP)
```

目的是让同一节点多个 DP rank 映射到不同 GPU 段，避免多个 DP rank 选到同一组 GPU。

### 58. Worker.load_model 的流程是什么？

Worker 层主要做上下文和 memory pool 管理，实际加载交给 model runner：

```text
Worker.load_model
  -> _maybe_get_memory_pool_context(tag="weights")
  -> set_current_vllm_config
  -> 临时设置 allocator max_split_size_mb=20
  -> model_runner.load_model(load_dummy_weights=...)
  -> 如配置 weight_transfer_config，创建 weight transfer engine
```

### 59. Worker.determine_available_memory 有哪两条路径？

两条路径：

1. 显式设置 `kv_cache_memory_bytes`：仍执行 `model_runner.profile_run()`，目的不是估算显存，而是编译最大 batch token 形状，然后直接返回用户配置值。
2. 自动估算：执行 `memory_profiling` 和 `model_runner.profile_run()`，统计 torch peak、non-torch increase、weights memory、CUDA graph 估算显存，计算可用于 KV cache 的显存。

### 60. Worker.get_kv_cache_spec 做什么？

Worker 层直接转发给 model runner：

```text
Worker.get_kv_cache_spec
  -> model_runner.get_kv_cache_spec()
```

ModelRunner 会扫描模型 attention layers，跳过不需要 KV cache 的模块，对 KV sharing target layer 做映射，并返回 layer name 到 `KVCacheSpec` 的字典。

### 61. Worker.initialize_from_config 是什么阶段？

这是 KV cache 真正初始化阶段：

```text
Worker.initialize_from_config(kv_cache_config)
  -> 写回 cache_config.num_gpu_blocks
  -> ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  -> 在 kv_cache memory pool 中调用 model_runner.initialize_kv_cache(kv_cache_config)
  -> 可初始化 routed experts capturer
  -> 如需要 KV cache zeroing，初始化 zero metadata
```

### 62. 为什么 KV transfer 要在 KV cache 初始化前初始化？

因为 connector 依赖 `KVCacheConfig`，且可能影响 KV cache group/layout。例如 connector 可能偏好 uniform / cross-layer KV layout，以便更高效地跨层传输 KV block。

所以必须在 model runner 分配 KV cache tensors 前初始化 KV transfer。

### 63. Worker.compile_or_warm_up_model 做什么？

它负责模型编译、kernel warmup、CUDA graph capture 等：

- 对需要编译但非 cudagraph capture size 的 batch size 做 dummy run；
- LoRA warmup 后移除 dummy LoRA；
- kernel warmup；
- 如果不是 enforce eager，调用 `model_runner.capture_model()`；
- V2 runner warmup execute/sample 相关 Triton kernel；
- V1 last PP rank 对 sampler/pooler 做 dummy run；
- 重置随机种子；
- 激活 Triton JIT monitor；
- freeze GC heap。

### 64. Worker.execute_model 的流程是什么？

流程：

```text
Worker.execute_model(scheduler_output)
  -> 等待上一轮 PP send 完成
  -> 如启用 PP 且非 first rank，接收 intermediate tensors
  -> model_runner.execute_model(scheduler_output, intermediate_tensors)
  -> 如果输出是 ModelRunnerOutput / AsyncModelRunnerOutput / None，直接返回
  -> 如果输出是 IntermediateTensors，发送给下一 PP rank 并返回 None
```

### 65. Worker.sample_tokens 做什么？

Worker 层 sample 只是转发：

```text
Worker.sample_tokens(grammar_output)
  -> model_runner.sample_tokens(grammar_output)
```

真正采样、grammar bitmask、logprobs、bookkeeping 都在 ModelRunner 中完成。

### 66. Worker.shutdown 会清理哪些资源？

GPU worker shutdown 会清理：

- KV transfer；
- EC transfer；
- profiler；
- weight transfer engine；
- model runner。

Multiproc 子进程关闭时还会销毁 model parallel 和 distributed environment。

### 67. CPUWorker 覆盖了哪些 GPU worker 逻辑？

CPUWorker 覆盖：

- CPU NUMA/内存节点初始化；
- CPU 设备初始化；
- CPU distributed backend；
- CPUModelRunner；
- CPU 内存 profiling；
- CPU profiler；
- 不支持 sleep/wake。

CPUModelRunner 通过 wrapper/no-op 替换 CUDA 相关 API，禁用 CUDA graph、cascade attention，并使用 CPU fallback。

### 68. XPUWorker 覆盖了哪些 GPU worker 逻辑？

XPUWorker 主要覆盖：

- XPU 设备初始化；
- CCL 环境变量；
- XPU memory snapshot；
- XPUModelRunner；
- XPU profiler activity。

XPUModelRunner 通过替换 torch.cuda API 到 torch.xpu API 来复用 GPUModelRunner。

## 八、ModelRunner 执行链路

### 69. ModelRunner 的核心定位是什么？

ModelRunner 是 Worker 下方真正执行模型 forward、sampling、KV/LoRA/spec decode 等逻辑的主体。

Worker 负责生命周期和设备通信包装，ModelRunner 负责模型执行细节。

### 70. vLLM V1 当前有哪些 GPU ModelRunner 路径？

两条：

- 旧版 V1：`vllm/v1/worker/gpu_model_runner.py`；
- 新版 V2：`vllm/v1/worker/gpu/model_runner.py`。

Worker 根据 `vllm_config.use_v2_model_runner` 选择具体实现。

### 71. 为什么 generation 路径拆成 execute_model 和 sample_tokens 两阶段？

两阶段执行：

```text
execute_model(scheduler_output)
  -> 更新状态
  -> 准备输入
  -> 执行 forward
  -> 保存 logits/metadata 到 ExecuteModelState
  -> 返回 None

sample_tokens(grammar_output)
  -> 读取 ExecuteModelState
  -> 应用 grammar bitmask
  -> sampling
  -> bookkeeping
  -> 返回 ModelRunnerOutput
```

原因：

- EngineCore 可以在 forward 期间准备 structured output grammar bitmask；
- 支持 async scheduling；
- 支持 PP 场景下非最后 rank 只传 intermediate tensors；
- 支持 spec decode、KV connector finalize 等复杂后处理。

### 72. execute_model 返回 None 是错误吗？

不是。对 generation 模型来说，`execute_model()` 返回 `None` 是主路径的一部分，表示 forward/logits 已完成并保存到 `ExecuteModelState`，后续需要调用 `sample_tokens(grammar_output)` 采样并返回 `ModelRunnerOutput`。

### 73. GPUModelRunner V1 execute_model 主流程是什么？

主流程包括：

1. 检查上一次 `execute_model()` 返回 `None` 后是否已调用 `sample_tokens()`；
2. spec ngram 路径可能复制 scheduler output；
3. KV transfer 处理 preemption；
4. `_update_states(scheduler_output)` 更新持久 batch/request 状态；
5. EC transfer producer 特殊路径；
6. 无 scheduled token 时返回 empty output 或 no-forward connector；
7. 准备 input ids、positions、token indices、slot mapping、LoRA active mapping、spec decode metadata、attention metadata；
8. 判断 CUDA graph / padding / ubatch；
9. `set_forward_context(...)` 并执行 `_model_forward(...)`；
10. 非最后 PP rank 返回 `IntermediateTensors`；pooling 模型直接返回 pooling output；generation 模型保存 `ExecuteModelState` 并返回 `None`。

### 74. GPUModelRunner V1 sample_tokens 主流程是什么？

流程：

1. 如果没有 `execute_model_state`，通常是 PP 非最后 rank 或只有 KV connector output，返回 connector only output；
2. 解包并清空 `execute_model_state`；
3. 应用 structured output grammar bitmask；
4. 调用 `_sample()` 产生 sampled token；
5. 更新 ModelRunner 状态；
6. async PP 场景下广播 sampled token ids；
7. 处理 speculative decoding draft tokens、接受/拒绝；
8. bookkeeping 生成 logprobs、prompt logprobs、有效 sampled token 等；
9. 构造 `ModelRunnerOutput`；
10. 同步路径直接返回，async scheduling 路径返回 `AsyncGPUModelRunnerOutput`。

### 75. AsyncGPUModelRunnerOutput 的作用是什么？

作用是把 GPU 输出异步拷贝到 CPU，避免主执行流阻塞。

创建时在 copy stream 上发起 sampled token ids / logprobs / routed experts 的异步 copy；`get_output()` 时等待 copy event，把 tensor 转成 list，写回 `ModelRunnerOutput`，返回 CPU 可见结果。

### 76. GPUModelRunner V2 相比 V1 有什么特点？

V2 更模块化，依赖多个子模块：

- `gpu.attn_utils`；
- `gpu.kv_connector`；
- `gpu.lora_utils`；
- `gpu.model_states`；
- `gpu.sample`；
- `gpu.spec_decode`；
- `gpu.pp_utils`；
- `gpu.cudagraph_utils`。

它将 V1 中较集中的逻辑拆分到更多专门模块中，便于维护和扩展。

### 77. GPUModelRunner V2 execute_model 主流程是什么？

大致流程：

```text
execute_model
  -> update_pp_decode_requests
  -> finish_requests
  -> free_states
  -> add_requests
  -> update_requests
  -> block_tables.apply_staged_writes
  -> 如无 token: kv_connector.no_forward
  -> dispatch_cg_and_sync_dp
  -> prepare_inputs
  -> prepare_attn
  -> multimodal embedding
  -> model forward: full cudagraph or eager/piecewise
  -> 保存 ExecuteModelState
  -> 非 last PP rank 返回 IntermediateTensors
  -> last PP rank 返回 None
```

### 78. GPUModelRunner V2 sample_tokens 主流程是什么？

非 last PP rank：

```text
sample_tokens
  -> 接收 last rank 广播 sampled tokens
  -> 更新本地状态
  -> 返回 KV connector only output
```

last PP rank：

```text
sample_tokens
  -> sample
  -> PP 广播 sampled tokens
  -> prompt logprobs
  -> 构造 ModelRunnerOutput
  -> 创建 AsyncOutput
  -> postprocess_sampled 更新状态
  -> speculator propose draft tokens
  -> kv_connector.post_forward
  -> 返回 async output
```

### 79. ModelRunnerOutput 包含哪些关键字段？

关键字段包括：

- `req_ids`：输出请求顺序；
- `req_id_to_index`：请求 ID 到输出 index；
- `sampled_token_ids`：本 step 生成 token；
- `logprobs`：采样 logprobs；
- `prompt_logprobs_dict`：prompt logprobs；
- `pooler_output`：pooling 输出；
- `kv_connector_output`：KV connector 输出；
- `ec_connector_output`：EC connector 输出；
- `num_nans_in_logits`：logits NaN 统计；
- `cudagraph_stats`：CUDA graph 统计；
- `routed_experts`：MoE routed experts 数据。

### 80. Pipeline Parallel 下 intermediate tensors 如何处理？

PP 非最后 stage 不返回最终 token，而是返回 `IntermediateTensors`。

Worker 层发现后：

- 通过 PP group 发送给下一 stage；
- 当前 rank 返回 `None`。

最后 PP stage 才负责 logits 和 sampling。

## 九、KV Cache、LoRA、KV Transfer、Profiler

### 81. KV cache 生命周期在执行层如何完成？

分阶段完成：

```text
Worker.get_kv_cache_spec
  -> ModelRunner 扫描模型 attention layers
  -> 返回每层 KVCacheSpec

Worker.determine_available_memory
  -> profile_run 估算非 KV 显存
  -> 返回可用于 KV cache 的内存

EngineCore
  -> 根据 spec + available memory 计算 KVCacheConfig

Executor.initialize_from_config
  -> 下发每个 worker 的 KVCacheConfig

Worker.initialize_from_config
  -> ensure_kv_transfer_initialized
  -> ModelRunner.initialize_kv_cache
  -> 分配、reshape、bind KV cache tensors
```

### 82. ModelRunner.get_kv_cache_spec 如何得到 KVCacheSpec？

ModelRunner V1 会遍历 static forward context 中的 `AttentionLayerBase`：

- 对 KV sharing target layer，不创建自己的 KVCacheSpec，而是记录 shared mapping；
- 跳过不需要 KV cache 的模块；
- 返回 layer name 到 `KVCacheSpec` 的 dict。

V2 有对应更模块化实现。

### 83. ModelRunner.initialize_kv_cache 做什么？

V1 `initialize_kv_cache()` 主要做：

- deep copy KVCacheConfig；
- 添加 encoder-only layers；
- 处理 cross-layer KV sharing；
- 初始化 attention backend；
- 初始化 Mamba SSU backend；
- 计算 kernel block sizes；
- 初始化 metadata builders；
- 根据真实 block size 可能重建 InputBatch；
- 分配并绑定 KV cache tensor；
- spec decode extract hidden states 校验；
- 注册 KV cache 到 KV connector。

### 84. uniform / cross-layer KV layout 是什么？

KV transfer connector 可能偏好跨层连续化 KV cache。

判断条件包括：

- 存在 KV transfer group；
- connector 偏好 cross-layer blocks；
- KV cache 只有单 group/单 attention group；
- backend 支持带 layer 维度 layout。

目的：让跨层 KV cache 连续分配，更便于按 block 做高效传输。

### 85. KV cache zeroing 是什么？

部分场景需要对新分配 KV block 清零。

机制：

- `_init_kv_zero_meta()` 创建 `KVBlockZeroer`；
- `_update_states()` 遇到 `scheduler_output.new_block_ids_to_zero` 时调用 `_zero_block_ids()`。

CPUModelRunner 中 `_zero_block_ids()` 是 no-op，因为 CPU attention 对非法位置赋 `-INF`，旧 KV 不影响计算。

### 86. Worker 层 LoRA API 和真实 LoRA 逻辑在哪里？

Worker 层 LoRA API 主要是转发：

- `add_lora`；
- `remove_lora`；
- `list_loras`；
- `pin_lora`。

真实逻辑在 `lora_model_runner_mixin.py`，包括加载 LoRA manager、设置 active adapters、dummy LoRA warmup、动态 add/remove/pin/list。

### 87. LoRA 如何在执行时激活？

执行时 ModelRunner 从 `InputBatch.make_lora_inputs()` 构造 prompt/token LoRA mapping，然后设置 active adapters。

这样同一个 batch 内可以根据请求使用不同 LoRA adapter。

### 88. dummy LoRA warmup 的作用是什么？

dummy LoRA warmup 用于 warmup / CUDA graph capture，确保带 LoRA 的执行路径、kernel、图捕获形状等提前准备好。

相关能力包括：

- `maybe_setup_dummy_loras()`；
- `maybe_select_dummy_loras()`；
- `maybe_dummy_run_with_lora()`。

### 89. KV Transfer / KV Connector 解决什么问题？

KV transfer 负责跨 worker / 跨实例 / 分层存储等场景下的 KV cache 传输，例如 disaggregated prefill/decode、remote KV load/save、KV offload。

Worker 初始化入口在 `initialize_from_config()` 中，ModelRunner 通过 connector mixin 在 execute 期间进入 connector context。

### 90. KV connector handshake metadata 是什么？

Worker 可以返回 KV connector 握手 metadata：

- 无 KV transfer group：返回 None；
- connector 不需要握手 metadata：返回 None；
- 否则以 `(pp_rank, tp_rank)` 为 key 返回 metadata。

EngineCore / scheduler 可以用这些 metadata 建立 scheduler 和 worker connector 的配套关系。

### 91. KV connector execute 期做什么？

关键流程：

```text
_get_kv_connector_output
  -> bind scheduler metadata
  -> start_load_kv(get_forward_context())
  -> forward 执行期间 connector 工作
  -> finally:
       wait_for_save
       collect finished sending/receiving
       collect invalid block ids
       collect stats/events/worker meta
       clear connector metadata
```

如果没有 forward，也可能执行 `kv_connector_no_forward`，用于处理只有 KV send/recv 的场景。

### 92. spec decode 与 KV connector finalize 有什么关系？

spec decode 时 forward 阶段可能需要延迟 finalize 到 drafter 之后，因为 draft token 的生成和验证可能影响 connector 需要保存/传输的 KV 状态。

所以 connector finalize 不是简单在 forward 结束立即完成。

### 93. Profiler 在 Worker 层如何支持？

GPU worker 支持：

- torch profiler：CPU + CUDA；
- cuda profiler。

start 时会给 trace name 加 rank suffix，profile annotation 会根据 SchedulerOutput 生成 context/generation 请求数量和 token 数。

CPU profiler 是 CPU-only torch profiler；XPU profiler 使用 CPU + XPU activities。

### 94. Weight transfer 在哪里初始化和清理？

Worker 加载模型后，如果配置了 `weight_transfer_config`，会基于 model 创建 weight transfer engine。

shutdown 时清理 weight transfer engine。

### 95. 这些横切能力和 SchedulerOutput 有什么关系？

很多能力由 `SchedulerOutput` 元数据驱动：

- `finished_req_ids`：worker/model runner 清理请求状态；
- `preempted_req_ids`：KV connector 处理抢占；
- `new_block_ids`：更新 block table；
- `new_block_ids_to_zero`：KV zeroing；
- `kv_connector_metadata`：KV transfer；
- `ec_connector_metadata`：EC transfer；
- `scheduled_spec_decode_tokens`：spec decode；
- `scheduled_encoder_inputs`：多模态/encoder。

## 十、分布式并行与通信

### 96. 执行层涉及哪两类通信？

两类：

1. 控制面通信：Executor 调 worker 方法。
   - UniProc：直接函数调用；
   - Multiproc：MessageQueue；
   - Ray old：Ray remote / Ray DAG；
   - Ray v2：Ray actor + MessageQueue。

2. 模型并行通信：worker/model runner 内部 tensor/object 通信。
   - torch distributed process group；
   - NCCL/Gloo；
   - vLLM `GroupCoordinator`；
   - device communicator；
   - TP/PP/DP/EP groups。

### 97. 控制面通信和模型并行通信为什么不能混淆？

Executor 控制面只负责调用方法，例如 `execute_model`、`sample_tokens`、`initialize_from_config`、`profile`、`shutdown`。

模型并行通信负责模型计算中的 tensor/object 通信，例如 TP all-reduce、PP send/recv intermediate tensors、DP synchronization、EP/MoE collectives、KV transfer 等。

Multiproc 的 MQ 不负责 TP/PP tensor 通信；这些由 NCCL/Gloo/GroupCoordinator 等完成。

### 98. rank 和 local_rank 在执行层有什么作用？

- `rank`：全局 distributed rank，用于 process group 和模型并行通信；
- `local_rank`：本地设备 index，用于选择本节点上的设备；
- `parallel_config.rank`：会写成当前 worker rank。

rank/local_rank 贯穿 executor 初始化、worker 初始化和 distributed group 构造。

### 99. ParallelConfig.world_size 如何理解？

基本为：

```text
world_size = pipeline_parallel_size * tensor_parallel_size * prefill_context_parallel_size
```

external launcher 模式下还会乘以 data parallel size。

常见并行维度包括 TP、PP、DP、PCP、DCP、EP、EPLB。

### 100. init_distributed_environment 做什么？

它处理默认 process group 初始化。

DP / 多节点场景会扩展 rank/world_size：

```text
rank = data_parallel_rank * world_size + rank
world_size = world_size_across_dp
```

单节点 DP 使用 `data_parallel_master_ip` 和 DP init port，多节点使用 `master_addr/master_port`。

### 101. split_group 特殊路径是什么？

开启 `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` 时，默认 PG 初始化为：

```text
"cpu:gloo,cuda:nccl"
```

并传入当前 CUDA device。目的是确保 CPU 和 device backend 都存在，后续 subgroup 可以通过 `split_group` 创建。

### 102. model parallel 分组布局是什么？

`initialize_model_parallel()` 的 rank layout 是：

```text
ExternalDP x DP x PP x PCP x TP
```

分组方式：

- TP group：最后一维连续；
- DCP group：复用 TP 维度切分；
- PCP group：PCP 维；
- PP group：PP 维，跨 pipeline stage；
- DP group：DP 维；
- EP group：合并 `DP * PCP * TP` 作为 expert parallel group。

### 103. EP / EPLB 为什么要分离 group？

MoE 模型才创建 EP group。开启 `enable_eplb` 时会创建独立 EPLB group。

原因：EP group 用于 MoE forward collectives；EPLB 如果用同一个 group，可能和 MoE forward collectives 死锁。所以 EPLB group rank 相同，但 process group 分离。

### 104. GroupCoordinator 是什么？

`GroupCoordinator` 是 vLLM 对一个逻辑通信 group 的封装。

内部维护：

- `cpu_group`：通常 Gloo，用于 CPU/object/metadata 控制通信；
- `device_group`：通常 NCCL，用于 GPU tensor 通信；
- `device_communicator`：平台相关通信器，如 CUDA 下的 `CudaCommunicator`。

对外提供 all_reduce、all_gather、reduce_scatter、send/recv tensor dict 等接口。

### 105. CudaCommunicator 支持哪些 allreduce 后端？

CUDA 平台下 device communicator 创建 `PyNcclCommunicator`，TP group 上可启用：

- NCCL symmetric memory；
- QuickReduce；
- FlashInfer；
- Custom allreduce；
- SymmMem；
- PyNCCL；
- fallback torch distributed all_reduce。

### 106. PyNcclCommunicator 有哪些特点？

特点：

- 基于非 NCCL CPU group 分发 NCCL unique id；
- 每个 communicator 绑定唯一 device；
- 创建后执行一次小 all_reduce warmup；
- destroy 使用 daemon thread 调 `ncclCommAbort()`，避免 CUDA graph 持有 NCCL op 时主线程自死锁。

### 107. Pipeline Parallel 执行通信如何发生？

Worker 的 `execute_model()` 处理 PP intermediate tensors：

```text
Worker.execute_model
  -> 如果上一轮有异步 PP send，先 wait
  -> 非 first PP rank: get_pp_group().irecv_tensor_dict() 接收上游 tensors
  -> model_runner.execute_model(..., intermediate_tensors)
  -> 如果输出 IntermediateTensors 且不是最后 PP rank:
       get_pp_group().isend_tensor_dict() 发送给下一 stage
       返回 None
```

### 108. DP 与 async scheduling 有什么特殊关系？

V1 async scheduling 启用时，DP synchronization 的 NCCL 可能默认禁用，改走 Gloo。

原因是 async 场景下 NCCL 多流/同步更容易出问题，Gloo CPU group 更安全。

DP CUDA graph padding 同步也可能使用 `get_dp_group().cpu_group` 执行 CPU all_reduce。

## 十一、端到端调用链路

### 109. 从外部请求到 Request 的链路是什么？

同步 LLMEngine：

```text
外部请求
  -> InputProcessor.process_inputs
  -> EngineCoreRequest
  -> assign_request_id
  -> output_processor 注册
  -> engine_core.add_request
```

异步 AsyncLLM 类似：

```text
InputProcessor.process_inputs
  -> assign_request_id
  -> _add_request
  -> output_processor 注册
  -> engine_core.add_request_async
```

### 110. EngineCoreRequest 包含哪些关键字段？

关键字段包括：

- `request_id`；
- `prompt_token_ids`；
- `mm_features`；
- `sampling_params`；
- `pooling_params`；
- `arrival_time`；
- `lora_request`；
- `cache_salt`；
- `data_parallel_rank`；
- `prompt_embeds`；
- `client_index`；
- `current_wave`；
- `priority`；
- `trace_headers`；
- `abort_immediately`。

### 111. EngineCoreClient 有哪些模式？

主要分两类：

1. In-process client：直接调用 engine_core 方法。
2. Multiprocess/ZMQ client：通过 socket 把输入请求发给 EngineCoreProc，从 output queue 读取 `EngineCoreOutputs`。

### 112. EngineCore.preprocess_add_request 做什么？

流程：

```text
EngineCoreRequest
  -> 如有 MM cache，处理 mm_features
  -> Request.from_engine_core_request
  -> 如有 structured output，初始化 grammar
  -> 返回 (Request, current_wave)
```

然后 `EngineCore.add_request()` 将 Request 放入 Scheduler。

### 113. EngineCoreProc busy loop 如何工作？

多进程 EngineCoreProc 主循环：

```text
run_busy_loop
  -> _process_input_queue
  -> _process_engine_step
```

输入处理会处理 ADD、ABORT、UTILITY 等 client request；engine step 会调用 `step_fn()`，把 outputs 放入 output queue，并执行 `post_step(model_executed)`。

### 114. EngineCore.step 主链路是什么？

核心流程：

```text
EngineCore.step
  -> if not scheduler.has_requests(): return {}, False
  -> scheduler_output = scheduler.schedule(...)
  -> future = model_executor.execute_model(scheduler_output, non_block=True)
  -> grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
  -> model_output = future.result()
  -> if model_output is None:
       model_output = model_executor.sample_tokens(grammar_output)
  -> _process_aborts_queue()
  -> engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)
  -> return engine_core_outputs, model_executed
```

### 115. SchedulerOutput 为什么是 executor/worker 的核心输入？

因为当前 V1 主路径中没有独立的 `ExecuteModelRequest` 类。实际从 scheduler 传给 executor/worker 的对象是 `SchedulerOutput`。

它包含本 step worker 执行所需的信息：新请求、缓存请求增量、scheduled token 数、新 block ids、finished/preempted request ids、spec decode tokens、encoder inputs、KV connector metadata 等。

### 116. SchedulerOutput 中 NewRequestData 和 CachedRequestData 有什么区别？

- `NewRequestData`：新请求首次调度时发送完整数据，包括 req id、prompt token ids、mm features、sampling/pooling params、block ids、LoRA、prompt embeds、prefill token ids 等。
- `CachedRequestData`：已缓存请求只发送增量，包括 req ids、resumed req ids、new token ids、all token ids、new block ids、num computed tokens、num output tokens 等。

这样可以减少每 step 传输的数据量。

### 117. Scheduler.update_from_output 做什么？

流程：

```text
Scheduler.update_from_output(scheduler_output, model_runner_output)
  -> 解包 sampled_token_ids/logprobs/pooler/connector output
  -> 处理 KV load 失败
  -> 存 routed experts
  -> 遍历本 step 调度过的 request
  -> 找到对应 sampled tokens
  -> 处理 spec decode 接受/拒绝
  -> 更新 Request 状态
  -> 推进 grammar
  -> 判断停止条件
  -> 释放 finished request
  -> 构造 EngineCoreOutput
  -> 更新 KV connector finished 状态
  -> 按 client_index 分组为 EngineCoreOutputs
  -> 附带 finished request ids 和 scheduler stats
```

### 118. ModelRunnerOutput 会直接返回给用户吗？

不会。`ModelRunnerOutput` 是 worker/model runner 的底层输出，先交给 `Scheduler.update_from_output()` 更新请求状态并生成 `EngineCoreOutputs`，再由 OutputProcessor 转换成用户可见的 `RequestOutput`。

### 119. in-process 完整链路是什么？

```text
LLMEngine.step
  -> InprocClient.get_output
  -> EngineCore.step_fn
  -> Scheduler.schedule
  -> Executor.execute_model
  -> WorkerWrapperBase.execute_model
  -> Worker.execute_model
  -> ModelRunner.execute_model
  -> Executor.sample_tokens if needed
  -> Worker.sample_tokens
  -> ModelRunner.sample_tokens
  -> ModelRunnerOutput
  -> Scheduler.update_from_output
  -> EngineCoreOutputs
  -> OutputProcessor.process_outputs
  -> RequestOutput
```

### 120. multiproc 完整链路是什么？

```text
Frontend
  -> EngineCoreClient socket send EngineCoreRequest
  -> EngineCoreProc input queue
  -> EngineCore.preprocess_add_request
  -> Scheduler.add_request
  -> EngineCoreProc busy loop
  -> EngineCore.step
  -> Scheduler.schedule
  -> MultiprocExecutor.execute_model
  -> rpc_broadcast_mq fan-out
  -> all WorkerProc receive RPC
  -> WorkerWrapperBase.execute_model
  -> Worker/ModelRunner execute
  -> output_rank response MQ
  -> FutureWrapper.result
  -> maybe sample_tokens same pattern
  -> Scheduler.update_from_output
  -> EngineCoreProc output_queue
  -> frontend output queue
  -> OutputProcessor
```

## 十二、常见调试与定位

### 121. 想看 Executor 怎么选，应该看哪里？

看：

```text
code/vllm/vllm/v1/executor/abstract.py:47
```

这里是 `Executor.get_class()`，根据 `distributed_executor_backend` 选择 UniProc、Multiproc、Ray old、Ray v2 或 external launcher。

### 122. 想看 EngineCore 每步怎么跑，应该看哪里？

看：

```text
code/vllm/vllm/v1/engine/core.py:479
```

这里是 `EngineCore.step()`，串起 scheduler -> executor -> grammar bitmask -> sample -> scheduler update。

### 123. 想看 SchedulerOutput 里面有什么，应该看哪里？

看：

```text
code/vllm/vllm/v1/core/sched/output.py:180
```

这里定义 `SchedulerOutput` 以及相关的 `NewRequestData`、`CachedRequestData`。

### 124. 想看多进程怎么广播 RPC，应该看哪里？

看：

```text
code/vllm/vllm/v1/executor/multiproc_executor.py:340
```

这里是 Multiproc 的 `collective_rpc()`，负责写 broadcast MQ、等待 response、处理 output rank。

### 125. 想看 worker 子进程 busy loop，应该看哪里？

看：

```text
code/vllm/vllm/v1/executor/multiproc_executor.py:806
```

这里是 worker 子进程消费 RPC、执行方法、写回 response 的循环。

### 126. 想看 GPU worker 初始化，应该看哪里？

看：

```text
code/vllm/vllm/v1/worker/gpu_worker.py:249
```

这里是 GPU Worker 的 `init_device()`。

### 127. 想看模型 forward，应该看哪里？

看：

```text
V1: code/vllm/vllm/v1/worker/gpu_model_runner.py:4043
V2: code/vllm/vllm/v1/worker/gpu/model_runner.py:1102
```

### 128. 想看 sampling，应该看哪里？

看：

```text
V1: code/vllm/vllm/v1/worker/gpu_model_runner.py:4422
V2: code/vllm/vllm/v1/worker/gpu/model_runner.py:1327
```

### 129. 想看 KV cache 初始化，应该看哪里？

看：

```text
Worker: code/vllm/vllm/v1/worker/gpu_worker.py:562
V1 ModelRunner: code/vllm/vllm/v1/worker/gpu_model_runner.py:7303
V2 ModelRunner: code/vllm/vllm/v1/worker/gpu/model_runner.py:406
```

### 130. 想看 PP send/recv，应该看哪里？

看：

```text
code/vllm/vllm/v1/worker/gpu_worker.py:807
```

Worker 的 `execute_model()` 中处理 PP receive/send intermediate tensors。

### 131. 想看 Ray DAG，应该看哪里？

看：

```text
code/vllm/vllm/v1/executor/ray_executor.py:536
code/vllm/vllm/v1/executor/ray_utils.py:123
```

前者构造 Ray compiled DAG，后者是 Ray actor 中实际执行的 `execute_model_ray()`。

### 132. 想看 Ray v2，应该看哪里？

看：

```text
code/vllm/vllm/v1/executor/ray_executor_v2.py:205
```

这里是 `RayExecutorV2` 类定义和主入口。

### 133. worker 初始化失败时应该如何定位？

优先看：

- executor 后端是否选对；
- WorkerWrapperBase 是否正确解析 worker_cls；
- rank/local_rank 是否正确；
- visible devices 是否正确；
- `init_device()` 是否成功；
- distributed init method、master addr/port 是否正确；
- NCCL/Gloo 初始化是否卡住；
- ModelRunner 是否创建成功。

对应优先文件：`abstract.py`、`worker_base.py`、`gpu_worker.py`、`parallel_state.py`、具体 executor 文件。

### 134. 多进程执行卡住时应该如何定位？

优先判断：

- broadcast MQ 是否成功写入；
- 所有 WorkerProc 是否 ready；
- worker busy loop 是否消费 RPC；
- output_rank 是否正确；
- response MQ 是否有结果；
- worker 内部是否异常并返回 FAILURE；
- 是否是模型并行通信 NCCL/Gloo 卡住，而不是 MQ 卡住。

### 135. PP 场景没有输出时应该如何定位？

重点看：

- 当前 rank 是否是最后 PP stage；
- 非最后 PP rank 是否正确返回 `IntermediateTensors`；
- Worker 是否执行 `isend_tensor_dict()`；
- 下一 PP stage 是否执行 `irecv_tensor_dict()`；
- output_rank 是否指向最后 PP stage；
- sample 是否只在最后 PP rank 执行。

### 136. execute_model 返回 None 时应该如何判断是否正常？

如果是 generation 模型，返回 None 通常正常，表示 forward/logits 已完成，等待 `sample_tokens()`。

异常情况是上一次 `execute_model()` 返回 None 后没有调用 `sample_tokens()`，下一次又进入 execute，这会触发状态检查错误。

### 137. async output 相关问题如何定位？

重点看：

- 是否启用 async scheduling；
- ModelRunner 是否返回 `AsyncModelRunnerOutput`；
- Multiproc 中 async output copy thread 是否启动；
- `get_output()` 是否等待 copy event；
- response MQ 是否在 copy 完成后写回；
- CUDA device 是否在线程中正确设置。

### 138. KV cache 初始化失败如何定位？

优先看：

- `Worker.get_kv_cache_spec()` 是否返回正确 spec；
- `determine_available_memory()` 是否得到合理容量；
- EngineCore 生成的 KVCacheConfig 是否正确；
- `ensure_kv_transfer_initialized()` 是否影响 layout；
- `ModelRunner.initialize_kv_cache()` 是否分配/reshape/bind KV tensors 成功；
- attention backend 是否支持 block size / dtype / layout。

### 139. LoRA 不生效如何定位？

重点看：

- Worker LoRA API 是否调用到；
- LoRA manager 是否加载 adapter；
- `set_active_loras()` 是否根据 InputBatch 设置 active mapping；
- batch 内请求的 LoRA mapping 是否正确；
- warmup/cudagraph 是否覆盖 LoRA 路径。

### 140. 如何一句话总结 Worker / Executor 执行层？

vLLM V1 Worker / Executor 执行层通过 Executor 抽象屏蔽 UniProc、Multiproc、Ray 等部署方式，用 WorkerWrapper 管理真实 worker 的创建与转发，在 Worker 中完成设备和分布式生命周期，在 ModelRunner 中把 SchedulerOutput 转成模型 forward、采样和 ModelRunnerOutput，是连接调度器与底层模型/kernel 的关键执行桥梁。

## 十三、可继续深入追问的问题清单

1. 为什么 Executor 是控制层而不是计算层？
2. `collective_rpc()` 在 UniProc、Multiproc、Ray old、Ray v2 中分别如何实现？
3. Multiproc 为什么采用“广播 RPC + output rank 回包”的模式？
4. output_rank 为什么通常是最后 PP stage 的第一个 TP rank？
5. WorkerWrapperBase 为什么要延迟构造真实 worker？
6. GPU worker 类为什么叫 `Worker` 而不是 `GPUWorker`？
7. 为什么 UniProc 仍然要构造 distributed init method？
8. Ray old 的 compiled DAG 和 Ray v2 的 MQ 控制面有什么本质差异？
9. RayWorkerProc 为什么不能在 actor `__init__` 中直接完整初始化 worker？
10. 为什么 worker 要在 memory snapshot 前初始化 distributed environment？
11. `determine_available_memory()` 中显式 `kv_cache_memory_bytes` 为什么仍要跑 profile？
12. KV transfer 为什么会影响 KV cache layout？
13. `execute_model()` 返回 None 的正常路径和异常路径如何区分？
14. structured output grammar bitmask 为什么要放在 `execute_model` 和 `sample_tokens` 之间？
15. PP 非最后 rank 为什么不产生 `ModelRunnerOutput`？
16. AsyncGPUModelRunnerOutput 如何避免主执行流阻塞？
17. DP async scheduling 为什么可能更偏向 Gloo 同步？
18. EP 和 EPLB 为什么要用独立 process group？
19. Worker shutdown 时为什么既要清理 transfer/profiler/model runner，也要销毁 distributed environment？
20. `SchedulerOutput` 中哪些字段会影响 worker/model runner 的横切能力？
