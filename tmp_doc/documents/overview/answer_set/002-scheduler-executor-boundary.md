# Q002：Scheduler 为什么不直接管理所有 Worker，而要中间加一层 Executor？

完成度：可定位

## 问题

Scheduler 为什么不直接管理所有 Worker，而要中间加一层 Executor？

## 一句话结论

Scheduler 负责生成调度决策，并维护请求生命周期状态；Executor 负责执行后端抽象、Worker 生命周期、collective RPC 分发和结果收集。中间加 Executor，是为了让调度策略不被单进程、多进程、Ray、多机多卡和控制面接口污染。

## L1：概念边界

### 它是什么

这个问题本质上是在问 Scheduler 和 Executor 的职责边界。Scheduler 是一轮推理中“决定怎么跑”的模块，它维护请求队列、token budget、KV block 账本，并产出 `SchedulerOutput`。Executor 是 EngineCore 和 Worker 之间的执行分发层，它把 `SchedulerOutput` 和控制命令发给一个或多个 Worker，再把 `ModelRunnerOutput` 收回来。

### 它解决什么问题

Executor 解决的是“同一份调度计划如何落到不同执行后端”的问题。Worker 可能是单进程里的本地对象，也可能是多进程里的子进程，还可能是 Ray actor 或外部 launcher 管理的执行单元。如果 Scheduler 直接管理 Worker，它就必须同时处理后端选择、RPC、队列、进程生命周期、Ray DAG、Worker 异常、profile、sleep、wake_up、LoRA、health、shutdown 等非调度逻辑。

### 它不负责什么

Executor 不负责决定哪些请求本轮执行，也不负责 token budget、waiting / running 队列、请求抢占、prefix cache 命中决策或 KV block 分配策略。这些属于 Scheduler。反过来，Scheduler 不负责创建 Worker、选择执行后端、发送 collective RPC、管理 Worker 进程或 Ray actor、处理执行层 shutdown 和设备侧控制命令。

### 和相邻模块的边界

- 和 **EngineCore** 的边界：EngineCore 是内层主循环总控，负责创建 Executor 和 Scheduler，并在 `step()` 中串起 schedule、execute、update；它不直接做 Worker 侧模型执行。
- 和 **Scheduler** 的边界：Scheduler 产出 `SchedulerOutput`，并在执行后用 `ModelRunnerOutput` 更新请求状态；它不直接调度 RPC 或管理 Worker 生命周期。
- 和 **Executor** 的边界：Executor 接收 `SchedulerOutput`，选择后端实现，把执行和控制命令发给 Worker，并收集结果。
- 和 **Worker / ModelRunner** 的边界：Worker 绑定设备和执行资源，ModelRunner 把 `SchedulerOutput` 展开成输入张量、attention metadata、forward、logits、sampling 和 `ModelRunnerOutput`。

## L2：端到端链路

### 输入

这个边界问题里最关键的输入包括：

- `VllmConfig.parallel_config.distributed_executor_backend`：决定使用 uni、mp、ray 或 external launcher 等执行后端。
- EngineCore 中的请求状态：Scheduler 根据 waiting / running 请求、KV cache 状态、token budget 等生成计划。
- `SchedulerOutput`：Scheduler 交给 Executor 的一轮执行计划。
- 控制命令：profile、sleep、wake_up、add/remove LoRA、health、shutdown、reset cache 等。

### 输出

主要输出包括：

- `SchedulerOutput`：Scheduler 产出的执行计划。
- `ModelRunnerOutput`：Worker / ModelRunner 执行后返回给 Scheduler 的内部结果。
- 控制接口结果：例如 `collective_rpc()` 从所有 Worker 返回的结果列表，或某个 output rank 的唯一结果。
- 更新后的请求状态和 `EngineCoreOutputs`：Scheduler 消化 `ModelRunnerOutput` 后产出给 EngineCore。

### 主链路

```text
LLMEngine / EngineCore
  -> Executor.get_class(vllm_config)
  -> model_executor = executor_class(vllm_config)
  -> Scheduler(...)
  -> Scheduler.schedule()
  -> SchedulerOutput
  -> Executor.execute_model(scheduler_output)
  -> Executor.collective_rpc("execute_model", ...)
  -> Worker.execute_model(scheduler_output)
  -> GPUModelRunner.execute_model(...)
  -> ModelRunnerOutput
  -> Scheduler.update_from_output(scheduler_output, model_output)
  -> EngineCoreOutputs
```

控制面链路则是：

```text
LLMEngine / EngineCore control API
  -> Executor.profile / sleep / wake_up / add_lora / shutdown / health
  -> Executor.collective_rpc(method, args, kwargs)
  -> all Worker 或指定 output rank
  -> Worker / ModelRunner 实际执行控制操作
```

### 状态变化对象

- `Scheduler`：改变 waiting / running 请求状态、每轮 token 分配、KV block 分配和请求完成状态。
- `Executor`：维护执行后端、Worker 集合、sleeping 状态、failure callback、KV output aggregator 等执行控制面状态。
- `Worker`：维护 device、rank、model_runner、分布式环境、KV cache、profile、sleep / wake_up 等设备侧状态。
- `GPUModelRunner`：维护 persistent batch、input batch、cached request state、attention metadata、执行中间态和采样状态。

### 真实计算对象

真实 GPU 计算不在 Scheduler 和 Executor 中完成，而是在 Worker / ModelRunner / model / attention backend / sampler / kernel 层完成。Executor 只负责把计划和命令分发过去；Scheduler 只负责生成计划并维护请求状态流转。

## L3：源码对象

### 关键类 / 函数

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:37`：`Executor` 是 vLLM executors 的抽象基类。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:40`：Executor 可以在一个 device 上执行，也可以作为 distributed executor 在多个 device 上执行。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:47`：`Executor.get_class()` 根据配置选择具体 Executor 类。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:60`：Ray backend 选择 `RayDistributedExecutor` 或 `RayExecutorV2`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:69`：`distributed_executor_backend == "mp"` 时选择 `MultiprocExecutor`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:73`：`distributed_executor_backend == "uni"` 时选择 `UniProcExecutor`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:153`：`collective_rpc()` 是 Executor 发起 Worker 集体 RPC 的抽象接口。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:162`：源码注释明确 `collective_rpc()` 会在所有 workers 上执行 RPC call。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:182`：源码注释建议 `collective_rpc()` 只传控制消息，数据面通信应另行建立。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:221`：`Executor.execute_model()` 接收 `SchedulerOutput`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:224`：`Executor.execute_model()` 通过 `collective_rpc("execute_model")` 分发到 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:256`：`profile()` 通过 `collective_rpc("profile")` 转发到 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:276`：`shutdown()` 通过 `collective_rpc("shutdown")` 转发到 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:292`：LoRA add/remove/list/pin 等控制接口也在 Executor 层统一转发。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:318`：`sleep()` 由 Executor 维护 sleeping 状态并广播给 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:331`：`wake_up()` 由 Executor 校验 tags 并广播给 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:45`：`UniProcExecutor` 是单进程执行后端。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:79`：单进程 `collective_rpc()` 直接在 driver worker 上执行方法。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:310`：`MultiprocExecutor.execute_model()` 把 `SchedulerOutput` 交给 `collective_rpc()`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:343`：多进程 `collective_rpc()` 负责广播 RPC 并收集响应。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:377`：多进程后端把 `(method, args, kwargs, output_rank)` 放进 broadcast queue。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:390`：多进程后端从 response queue 收集 Worker 返回。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py:470`：Ray executor 也实现了 `collective_rpc()`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py:484`：Ray 后端对每个 Ray worker 调用 `execute_method.remote()`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:122`：EngineCore 创建 `model_executor`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:150`：EngineCore 在 KV cache 初始化后创建 Scheduler。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:488`：`EngineCore.step()` 是 schedule、execute、make output 的闭环入口。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:499`：EngineCore 调用 `scheduler.schedule()` 生成 `SchedulerOutput`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:500`：EngineCore 调用 `model_executor.execute_model(scheduler_output)` 执行计划。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:513`：EngineCore 把 `model_output` 交回 `scheduler.update_from_output()`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:69`：`Scheduler` 是核心调度器类。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:108`：Scheduler 保存调度约束，例如最大 running 请求数和 token budget。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:433`：`Scheduler.schedule()` 产出一轮调度结果。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:435`：源码注释说明 Scheduler 没有固定 prefill/decode phase，而是根据请求 token 进度分配 token。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:182`：`SchedulerOutput` 是一轮执行计划对象。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:39`：`WorkerBase` 是 Worker 接口，用于分离不同硬件实现。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:40`：源码注释明确 Worker 接口用于 cleanly separate implementations for different hardware。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:142`：Worker 接收 `SchedulerOutput` 执行模型。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1002`：GPU Worker 的 `execute_model()` 是设备侧执行入口。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1062`：GPU Worker 最终调用 `model_runner.execute_model()`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:4097`：`GPUModelRunner.execute_model()` 真正把 `SchedulerOutput` 变成模型执行。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/outputs.py:231`：`ModelRunnerOutput` 会被序列化并发送到 scheduler process。

### 关键字段

- `VllmConfig.parallel_config.distributed_executor_backend`：决定 Executor 后端类型，是 `uni`、`mp`、`ray`、`external_launcher` 或自定义 Executor。
- `Executor.vllm_config` / `parallel_config` / `scheduler_config` / `cache_config`：Executor 持有执行后端需要的全局配置。
- `Executor.is_sleeping` / `sleeping_tags`：Executor 层维护 sleep / wake_up 控制面状态。
- `Executor.kv_output_aggregator`：有 KV connector 输出时，用于聚合多个 Worker 的 KV 输出。
- `Scheduler.max_num_running_reqs`：Scheduler 的最大 running 请求数约束。
- `Scheduler.max_num_scheduled_tokens`：Scheduler 的每轮 token budget。
- `SchedulerOutput.num_scheduled_tokens`：每个请求本轮计划执行的 token 数。
- `SchedulerOutput.total_num_scheduled_tokens`：本轮所有请求计划执行 token 总数。
- `WorkerBase.device`：Worker 绑定的设备。
- `WorkerBase.model_runner`：Worker 持有的真实执行组件。
- `ModelRunnerOutput.sampled_token_ids`：执行层返回的采样 token。

### 状态改变方法

- `Executor.get_class()`：根据配置决定执行后端类型，隔离 backend 差异。
- `Executor.initialize_from_config()`：广播 KV cache 初始化并触发 worker 侧 compile / warmup。
- `Executor.collective_rpc()`：把某个方法调用广播到 Worker，或收集指定 output rank / 聚合后的结果。
- `Executor.execute_model()`：把 `SchedulerOutput` 分发给 Worker 执行。
- `Executor.profile()` / `sleep()` / `wake_up()` / `shutdown()`：统一转发运行期控制命令。
- `Scheduler.schedule()`：根据请求状态、token budget、KV cache 状态产出执行计划。
- `Scheduler.update_from_output()`：用执行结果推进请求状态、释放资源并产出 EngineCoreOutputs。
- `Worker.execute_model()`：设备侧执行入口，进一步调用 ModelRunner。
- `GPUModelRunner.execute_model()`：更新 worker 侧 batch 状态、准备输入并执行 forward / sampling 前半段。

### 关键配置

- `distributed_executor_backend`：最直接体现为什么需要 Executor；Scheduler 不应知道后端是本地调用、多进程队列还是 Ray actor。
- `tensor_parallel_size` / `pipeline_parallel_size` / `data_parallel_size`：影响 Worker 数量、output rank、执行结果聚合和通信路径。
- `max_concurrent_batches`：影响 EngineCore 是否使用 batch queue，也影响调度和执行的异步重叠方式。
- `scheduler_config.max_num_seqs` / `max_num_batched_tokens` / `max_num_scheduled_tokens`：属于 Scheduler 的调度约束，不属于 Executor。
- `kv_transfer_config`：Scheduler 和 Worker 都可能各自创建 KV connector，但跨 Worker 输出聚合在 Executor 层衔接。
- `lora_config` / `observability_config` / `compilation_config`：会触发 Executor 转发控制命令或 Worker 侧执行能力。

### 计划对象与结果对象

- 计划对象：`VllmConfig`、`SchedulerOutput`、`KVCacheConfig`、`ReconfigureDistributedRequest`。
- 状态对象：`Scheduler`、`KVCacheManager`、`Executor`、`WorkerBase`、`GPUModelRunner`、`InputBatch`。
- 结果对象：`ModelRunnerOutput`、`EngineCoreOutputs`、Worker control RPC 返回值。

## L4：取舍、性能与排查

### 为什么这样设计

Scheduler 和 Executor 分开，是为了让调度策略保持纯粹：Scheduler 只关心“本轮哪些请求执行多少 token、占用哪些 KV block、结束哪些请求”；Executor 关心“怎么把这个计划发到具体执行资源上”。这样同一个 Scheduler 可以复用在单进程、多进程、Ray、多机多卡等不同后端上，执行后端变化不需要改调度算法。

另一层原因是控制接口并不都来自 Scheduler。profile、sleep、wake_up、health、shutdown、LoRA、reset cache、KV handshake 等控制能力可能来自 LLMEngine、EngineCore 或上层服务，它们不是调度算法的一部分。如果没有 Executor，这些接口就会被塞进 Scheduler，导致 Scheduler 变成“调度器 + RPC 层 + Worker 管理器 + 生命周期管理器”。

### 优化了什么

- **可扩展性**：同一 Scheduler 可以对接 uni、mp、ray、external launcher 或自定义 Executor。
- **可维护性**：调度策略和执行后端代码分离，减少 Scheduler 中的 backend 分支。
- **多 GPU / 多进程支持**：Executor 统一处理 Worker 创建、RPC、output rank、KV 输出聚合、failure callback 和 shutdown。
- **控制面一致性**：profile、sleep、wake_up、LoRA、health、shutdown 都通过统一入口转发到 Worker。
- **性能演进空间**：Executor 可以针对不同后端做非阻塞执行、Ray compiled DAG、pipeline parallel batch queue 等优化，而不污染 Scheduler。

### 牺牲了什么

- **调用链更长**：`SchedulerOutput` 需要经过 Executor 才能到 Worker，排查时要跨 Scheduler、Executor、Worker、ModelRunner 多层。
- **序列化 / RPC 成本**：多进程和 Ray 后端需要传输 `SchedulerOutput`、接收 `ModelRunnerOutput`，会有队列、序列化、反序列化和同步开销。
- **状态分散**：请求状态在 Scheduler，执行状态在 Worker / ModelRunner，生命周期状态在 Executor；错误定位需要明确是哪一层的问题。
- **返回值聚合复杂度**：TP / PP / KV connector 场景下，并不是所有 Worker 都直接返回最终 `ModelRunnerOutput`，Executor 要处理 output rank 或聚合逻辑。

### 什么情况下收益不明显

- 单 GPU、单进程、低并发场景：Executor 仍然存在，但它更多是薄封装，抽象层收益没有多进程 / 多 GPU 明显。
- 后端固定且极简：如果系统永远只有一个本地 Worker，直接调用 Worker 理论上也能跑，但会失去未来扩展和统一控制面的能力。
- workload 很小：RPC / 队列 / Future 的开销可能相对更明显，不过这通常不是 vLLM 面向的主要 serving 场景。

### 常见问题与排查路径

1. 现象：Scheduler 已经产生计划，但模型没有执行。
   - 可能原因：`SchedulerOutput.total_num_scheduled_tokens == 0`，Executor future 返回空输出，Worker 提前返回 `EMPTY_MODEL_RUNNER_OUTPUT`，或 Ray / 多进程后端调用失败。
   - 排查对象：`SchedulerOutput.total_num_scheduled_tokens`、`EngineCore.step()`、`Executor.execute_model()`、Worker `execute_model()` 日志。
   - 验证方法：打印或日志记录每轮 `num_scheduled_tokens`、确认 Executor 后端类型、检查是否进入 Worker / ModelRunner。

2. 现象：多进程 / Ray 场景卡住或超时。
   - 可能原因：Worker 进程崩溃、response queue 没有返回、Ray actor 调用失败、NCCL / PP 通信阻塞。
   - 排查对象：`MultiprocExecutor.collective_rpc()` timeout、worker monitor、Ray refs、Worker traceback、NCCL 日志。
   - 验证方法：先用 `distributed_executor_backend=uni` 对照，再切到 `mp` / `ray`；缩小 TP / PP，确认是否是执行后端或通信问题。

3. 现象：Scheduler 状态正常，但输出为空或只有部分 Worker 返回。
   - 可能原因：output rank 选择错误、PP 最后一阶段未返回、KV output aggregator 配置不一致、`ModelRunnerOutput` 被延后到 `sample_tokens()`。
   - 排查对象：`unique_reply_rank` / `output_rank`、`kv_output_aggregator`、`execute_model()` 是否返回 None、`sample_tokens()` 是否被调用。
   - 验证方法：关闭 PP / KV transfer / async scheduling 做最小复现，再逐项开启。

4. 现象：profile / sleep / wake_up / LoRA 控制命令不生效。
   - 可能原因：命令没有通过 Executor 广播到所有 Worker、某个 Worker 失败、sleep tags 不匹配、LoRA 在各 Worker 上状态不一致。
   - 排查对象：`Executor.collective_rpc()` 返回值、`Executor.is_sleeping` / `sleeping_tags`、`Worker.profile()` / `sleep()` / `wake_up()` / LoRA 方法。
   - 验证方法：用 `collective_rpc("check_health")` 验证 Worker 存活，再检查每个 Worker 返回的 LoRA 列表或 profiler 状态。

### Benchmark 设计

- 指标：TTFT、TPOT / ITL、throughput tokens/s、request/s、P95 / P99 latency、CPU 调度时间、Executor RPC 时间、序列化时间、GPU 利用率、NCCL / Ray / queue 等待时间、错误率。
- 变量：`distributed_executor_backend=uni/mp/ray`、TP size、PP size、并发数、prompt length、output length、`max_num_batched_tokens`、`max_concurrent_batches`、是否开启 KV transfer / LoRA / structured output。
- 对照组：
  - 单进程 uni vs 多进程 mp。
  - mp TP=1 vs TP=2/4/8。
  - mp vs Ray。
  - PP off vs PP on。
  - KV connector off vs on。
- 预期现象：
  - 单进程低并发下 Executor 抽象开销相对更明显，但绝对值通常较小。
  - 多 GPU 场景中 Executor 的后端抽象让吞吐扩展成为可能，但通信和结果聚合会成为新瓶颈。
  - Ray / mp 后端的 RPC 与序列化成本会随 `SchedulerOutput` 和 `ModelRunnerOutput` 大小增长。
  - PP / async scheduling 可能降低 pipeline bubble，但排查链路更复杂。

## 源码证据

- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:21`：已有问题文档把 Executor / Worker / ModelRunner 的职责作为执行层首要问题。
- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:22`：已有问题文档明确要求解释它们和 EngineCore、Scheduler 的关系。
- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:72`：已有总览文档把 Executor 定义为 EngineCore 和 Worker 之间的执行分发层。
- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:79`：已有总览文档列出 Executor 根据配置选择执行后端。
- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:81`：已有总览文档说明 Executor 通过 collective RPC 分发控制命令。
- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:278`：已有总览文档说明 Scheduler 生成的是执行计划 `SchedulerOutput`。
- `D:/lzy/project/kv_pool/llm-project/tmp_doc/documents/question/executor_worker_model_runner/executor_worker_model_runner_overview.md:303`：已有总览文档列出 EngineCore 每轮 schedule、execute、sample、update 的流程。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:37`：`Executor` 抽象基类定义执行层统一接口。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:47`：`Executor.get_class()` 根据配置选择后端，是 Scheduler 不直接管理 Worker 的核心证据。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:153`：`collective_rpc()` 是 Executor 对 Worker 发起集体调用的抽象。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:162`：源码注释说明 `collective_rpc()` 在所有 Worker 上执行 RPC call。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:182`：源码注释说明 collective RPC 推荐用于控制消息，体现控制面和数据面的区分。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:221`：Executor 接收 SchedulerOutput 执行模型。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:224`：Executor 通过 collective RPC 把 `execute_model` 分发到 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/uniproc_executor.py:79`：单进程后端的 collective RPC 是本地直接调用。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:343`：多进程后端单独实现 collective RPC。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor.py:470`：Ray 后端单独实现 collective RPC。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:122`：EngineCore 先创建 `model_executor`。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:150`：EngineCore 再创建 Scheduler，说明二者是被 EngineCore 编排的相邻模块。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:499`：EngineCore 每轮先调用 Scheduler 生成计划。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:500`：EngineCore 随后把计划交给 Executor 执行。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:513`：执行结果回到 Scheduler 更新状态。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:433`：`Scheduler.schedule()` 是调度计划入口。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:435`：源码注释说明 Scheduler 的核心是按请求 token 进度分配 token，而不是管理 Worker。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:182`：`SchedulerOutput` 承载一轮计划。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:39`：Worker 接口用于分离不同硬件实现。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1002`：Worker 侧 `execute_model()` 接收计划并进入设备执行路径。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:4097`：ModelRunner 执行真实模型路径。
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/outputs.py:231`：`ModelRunnerOutput` 是跨进程回传给 scheduler process 的结果。

## 容易混淆点

- 不要把 Executor 理解成“另一个 Scheduler”。Executor 不决定请求优先级、token budget 或 KV block 分配，它只负责执行后端分发和控制面。
- 不要把 SchedulerOutput 理解成模型输出。它是计划；`ModelRunnerOutput` 才是执行层结果。
- 不要把 collective RPC 理解成 NCCL all-reduce。这里的 collective RPC 是控制层面对一组 Worker 发起方法调用；真实张量通信可能由 NCCL、Ray compiled DAG、pipeline send/recv 等机制完成。
- 不要认为单进程时 Executor 没意义。单进程时 Executor 是薄封装，但它保证上层代码和多进程 / Ray 后端共享同一接口。
- 不要把 Worker 生命周期塞给 Scheduler。Worker 初始化、load_model、KV cache 初始化、warmup、profile、sleep、shutdown 都是执行系统管理能力，不是调度策略。

## 我还不确定的点

- 当前答案基于本地 `D:/lzy/project/kv_pool/code/vllm` 源码行号整理；如果后续 vLLM commit 更新，具体行号需要重新校准。
- 不同 Executor 后端在 Ray V2、external launcher、elastic EP 等场景下还有更细的控制流差异，后续可以单独补充专题答案。
