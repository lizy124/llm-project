# 01. Executor 在 vLLM V1 里负责什么？

源码位置：

- `vllm/vllm/v1/executor/abstract.py`
- `vllm/vllm/v1/executor/uniproc_executor.py`
- `vllm/vllm/v1/executor/multiproc_executor.py`
- `vllm/vllm/v1/executor/ray_executor.py`
- `vllm/vllm/v1/executor/ray_executor_v2.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/worker/worker_base.py`

这个问题关注：`Executor` 在 vLLM V1 的执行链路里到底处于哪一层、负责什么、不负责什么；它如何根据配置选择单进程 / 多进程 / Ray 后端；它如何把 `SchedulerOutput` 送到 Worker；它如何把 Worker 的结果取回来；以及 profile、sleep、LoRA、KV transfer、distributed control 等控制面能力如何统一通过它转发。

---

## 1. 一句话回答

`Executor` 是 **EngineCore 和 Worker 之间的执行分发层**。

它负责：

```text
1. 根据配置选择执行后端；
2. 创建并管理 Worker；
3. 用 collective RPC 把控制命令发到所有 Worker；
4. 把 SchedulerOutput 分发给 Worker 执行；
5. 把 ModelRunnerOutput 收回来；
6. 转发 profile / sleep / wake_up / LoRA / health / shutdown / KV 相关控制接口。
```

它不负责：

```text
1. 调度决策（这是 Scheduler 的事）；
2. token budget / request state 管理（这是 Scheduler 的事）；
3. 模型 forward 的具体细节（这是 Worker / ModelRunner 的事）；
4. 采样算法本身（这是 ModelRunner / Sampler 的事）。
```

可以把它理解成：

```text
EngineCore 决定“这一轮要跑什么”；
Executor 决定“把这轮交给哪些 Worker 去跑”；
Worker / ModelRunner 决定“具体怎么跑”。
```

---

## 2. 一句话总览链路

```text
EngineCore
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → ModelRunner.execute_model()
  → forward / logits / pooling
  → sample_tokens()
  → ModelRunnerOutput
```

如果是控制类接口，链路通常是：

```text
EngineCore / 外层 Engine
  → Executor.collective_rpc()
  → WorkerBase / Worker
```

---

## 3. Executor 的抽象定义

核心抽象在：`vllm/v1/executor/abstract.py:37`

```python
class Executor(ABC):
    """Abstract base class for vLLM executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """
```

这里有两个关键信息：

1. `Executor` 是抽象基类，不是具体执行后端；
2. 它既可以表示单设备执行，也可以表示分布式执行。

### 3.1 初始化时拿到的配置

`Executor.__init__()` 会把 `VllmConfig` 中的关键子配置缓存到成员变量里：

```text
self.model_config
self.cache_config
self.lora_config
self.load_config
self.parallel_config
self.scheduler_config
self.device_config
self.speculative_config
self.observability_config
```

位置：`abstract.py:95` 到 `abstract.py:110`

这说明 Executor 并不是“纯转发器”，它需要理解：

```text
模型配置、缓存配置、并行配置、LoRA、投机解码、可观测性
```

这些信息会影响它怎么创建 Worker、怎么走 RPC、怎么执行和控制。

### 3.2 运行状态

Executor 还维护一些运行态：

```text
is_sleeping
sleeping_tags
kv_output_aggregator
```

位置：`abstract.py:110` 到 `abstract.py:113`

这说明它除了执行，还承担一部分 lifecycle 状态管理。

---

## 4. Executor 怎么选具体后端

入口是：

```python
Executor.get_class(vllm_config)
```

位置：`abstract.py:47`

它根据 `parallel_config.distributed_executor_backend` 选择实现。

### 4.1 选择规则

```text
- type 是 Executor 子类 → 直接使用
- "ray" → RayDistributedExecutor 或 RayExecutorV2
- "mp" → MultiprocExecutor
- "uni" → UniProcExecutor
- "external_launcher" → ExecutorWithExternalLauncher
- 其他字符串 → 按 qualname 动态解析
```

对应源码：`abstract.py:48` 到 `abstract.py:92`

### 4.2 这意味着什么

Executor 的选择不是写死的，而是由配置驱动：

```text
VllmConfig.parallel_config.distributed_executor_backend
```

因此，`Executor` 实际上是一个后端工厂接口 + 统一控制面抽象。

---

## 5. 具体 Executor 形态

### 5.1 UniProcExecutor：单进程执行

源码：`vllm/v1/executor/uniproc_executor.py`

它的特点是：

```text
- driver 和 worker 在同一进程；
- 通过 WorkerWrapperBase 直接调用；
- execute_model / sample_tokens 可直接同步或异步返回；
- supports_async_scheduling() = True
```

初始化里会：

```text
1. 创建 driver_worker；
2. 计算 distributed init 参数；
3. init_worker();
4. init_device();
5. load_model();
```

位置：`uniproc_executor.py:45` 到 `uniproc_executor.py:70`

这条路径最简单，适合单机单进程或轻量部署。

### 5.2 MultiprocExecutor：多进程执行

源码：`vllm/v1/executor/multiproc_executor.py`

它的特点是：

```text
- 每个 Worker 在独立进程中；
- 通过 MessageQueue / shared memory / pipe 做控制面通信；
- 支持 TP / PP / PCP / DP 相关并行拓扑；
- 有 worker monitor 监控子进程存活；
- supports_pp = True
```

初始化时会：

```text
1. 计算 world size / TP / PP / PCP；
2. 创建 broadcast MQ 和 response MQ；
3. 启动多个 WorkerProc；
4. 等待 worker ready；
5. 初始化 worker 后端；
6. 计算 output_rank。
```

位置：`multiproc_executor.py:103` 到 `multiproc_executor.py:260`

这是 vLLM 里最典型的多进程执行后端。

### 5.3 RayDistributedExecutor：Ray actor 后端

源码：`vllm/v1/executor/ray_executor.py`

它的特点是：

```text
- 通过 Ray actor 管理 worker；
- 用 compiled DAG / Ray 调度执行；
- 适配 Ray 生态；
- 可处理 KV connector 场景；
- uses_ray = True, supports_pp = True
```

它会在初始化时：

```text
1. 初始化 Ray cluster；
2. 创建 workers；
3. 计算是否使用 sampler；
4. 初始化 KV / EC transfer 相关状态；
5. 保存 scheduler_output 供后续 sample_tokens 使用。
```

位置：`ray_executor.py:64` 到 `ray_executor.py:99`，以及 `ray_executor.py:389` 之后

### 5.4 RayExecutorV2：基于 MessageQueue 的 Ray 方案

源码：`vllm/v1/executor/ray_executor_v2.py`

它复用 `MultiprocExecutor` 的 MQ 通信思路，但 worker 运行在 Ray actor 里。

重点特征：

```text
- 继承 MultiprocExecutor；
- Ray actor + MQ；
- 支持 async scheduling；
- 处理 placement group、bundle index、worker rank 重排；
- 面向更复杂的分布式部署。
```

位置：`ray_executor_v2.py:219` 起

### 5.5 ExecutorWithExternalLauncher

在 `uniproc_executor.py` 中定义。

它主要面向：

```text
- torchrun / external launcher 场景；
- SPMD 风格 offline inference；
- 要求 deterministic execution。
```

位置：`uniproc_executor.py:150` 到 `uniproc_executor.py:197`

---

## 6. Executor 的核心职责

从接口上看，Executor 负责以下几类事。

### 6.1 创建和初始化 Worker

典型入口：

```python
initialize_from_config(kv_cache_configs)
```

位置：`abstract.py:118`

它会：

```text
1. 调用 workers 的 initialize_from_config();
2. 调用 compile_or_warm_up_model();
3. 收集 compilation time，并回写到 config。
```

这是 Executor 进入“真正可运行状态”的关键步骤。

### 6.2 分发模型执行

核心接口：

```python
execute_model(scheduler_output, non_block=False)
```

位置：`abstract.py:210`

它把 SchedulerOutput 传给所有 worker。

在抽象实现里：

```python
output = self.collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
return output[0]
```

位置：`abstract.py:221` 到 `abstract.py:227`

说明对大多数后端来说，Executor 只关心“发给 worker 并拿回结果”。

### 6.3 采样阶段

核心接口：

```python
sample_tokens(grammar_output, non_block=False)
```

位置：`abstract.py:229` 到 `abstract.py:247`

这个接口的存在说明：

```text
execute_model() 不一定总是直接返回最终输出；
sample_tokens() 可能需要单独调用。
```

这一点在 Ray 后端尤其明显：

```text
execute_model() 先记录 scheduler_output；
sample_tokens() 再真正执行 DAG 并生成输出。
```

### 6.4 运行时控制

Executor 还统一提供：

```text
profile()
execute_dummy_batch()
save_sharded_state()
sleep()
wake_up()
shutdown()
check_health()
add_lora()
remove_lora()
pin_lora()
list_loras()
reset_mm_cache()
reset_encoder_cache()
reinitialize_distributed()
```

这些方法本质上都是把控制命令发给 Worker。

---

## 7. collective_rpc 是什么

`collective_rpc()` 是 Executor 的核心底层接口。

位置：`abstract.py:152` 到 `abstract.py:203`

它的语义是：

```text
对所有 worker 执行同一个方法，或执行同一个可序列化 callable。
```

### 7.1 参数含义

```text
method      方法名或 callable
args        位置参数
kwargs      关键字参数
timeout     超时
non_block   是否异步返回 Future
```

### 7.2 为什么它重要

因为 Executor 的大多数能力都可以归纳成：

```text
- 广播控制命令
- 收集 Worker 返回值
```

例如：

```text
initialize_from_config()
determine_available_memory()
get_kv_cache_specs()
get_supported_tasks()
shutdown()
profile()
sleep()
wake_up()
```

都建立在 `collective_rpc()` 上。

### 7.3 返回值形态

抽象层里返回 worker 结果列表；
具体实现里可能：

```text
- 返回单个值；
- 返回 Future；
- 聚合多个 worker 的输出；
- 只取 output_rank 的结果。
```

这取决于后端实现。

---

## 8. execute_model() 的主链路

### 8.1 抽象层语义

```text
EngineCore
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → ModelRunner.execute_model()
```

Executor 的工作不是做 forward，而是把一轮 `SchedulerOutput` 送到 worker。

### 8.2 不同后端的执行差异

#### UniProcExecutor

直接调用本地 worker：

```text
run_method(self.driver_worker, "execute_model", args, kwargs)
```

如果返回 `AsyncModelRunnerOutput`，还会通过 `AsyncOutputFuture` 延迟取回。

#### MultiprocExecutor

通过 MQ 广播到所有 worker，然后从响应队列收结果。

并且它支持：

```text
unique_reply_rank
kv_output_aggregator
output_rank
```

因此适合真正的分布式环境。

#### RayDistributedExecutor / RayExecutorV2

会把 execute_model 变成 Ray actor 上的分布式执行任务，可能还会在 `sample_tokens()` 阶段再取结果。

### 8.3 一个关键细节：`None`

`Worker.execute_model()` / `Executor.execute_model()` 可能返回 `None`。

这表示：

```text
当前还没有最终 ModelRunnerOutput，需要紧接着调用 sample_tokens()。
```

这个设计在异步 scheduling / structured output / pipeline 场景里很重要。

---

## 9. sample_tokens() 的角色

`sample_tokens()` 不是“可有可无的辅助函数”，而是执行链路的一部分。

### 9.1 它什么时候被调用

常见于：

```text
execute_model() 返回 None
  → 立刻调用 sample_tokens(grammar_output)
```

### 9.2 它为什么独立出来

因为有些后端会把：

```text
forward
和
sampling
```

拆成两个阶段。

这样做的原因通常是：

```text
- 支持 async scheduling
- 支持 pipeline parallel
- 支持 structured output 的 grammar 约束
- 支持 Ray / 进程间异步执行
```

### 9.3 与 grammar_output 的关系

Executor 会把：

```python
grammar_output: GrammarOutput | None
```

传给 worker。

它来自：

```text
Scheduler.get_grammar_bitmask(scheduler_output)
```

也就是说，Executor 并不自己决定 grammar；它只是把 Scheduler 的约束继续传给 Worker。

---

## 10. Executor 和 Worker 的边界

这是最容易混淆的地方。

### 10.1 Executor 负责

```text
- 管 worker 的生命周期
- 广播控制命令
- 分发 SchedulerOutput
- 收集 ModelRunnerOutput
- 聚合多 worker 结果
- 处理多进程 / Ray 通信差异
```

### 10.2 Worker 负责

```text
- 初始化 device
- 加载模型
- 管理 model_runner
- 初始化 KV cache
- 真正执行模型
- 采样
- 管理 LoRA / profiler / sleep / wake_up
```

### 10.3 ModelRunner 负责

```text
- 把 SchedulerOutput 转成 batch 输入
- 准备 attention metadata
- 执行 forward
- 生成 logits / pooling output
- sample_tokens()
- 构造 ModelRunnerOutput
```

### 10.4 这个边界如何记

```text
Executor = orchestration
Worker = device runtime
ModelRunner = model execution
```

---

## 11. 初始化链路：从配置到可执行

### 11.1 `initialize_from_config()`

位置：`abstract.py:118` 起

这一步本质上是：

```text
先把 KV cache 和模型 warmup 跑起来，
让 worker 进入真正可服务状态。
```

### 11.2 `determine_available_memory()`

位置：`abstract.py:146`

它通过 worker 计算可用显存，用于：

```text
KV cache 大小估计 / profiling / auto-fit。
```

### 11.3 `get_kv_cache_specs()`

位置：`abstract.py:149`

这是 EngineCore 初始化 KV cache 的关键输入之一。

### 11.4 `EngineCore` 如何使用它

在 `engine/core.py` 里：

```python
kv_cache_specs = self.model_executor.get_kv_cache_specs()
available_gpu_memory = self.model_executor.determine_available_memory()
self.model_executor.initialize_from_config(kv_cache_configs)
```

位置：`engine/core.py:240` 到 `engine/core.py:348`

也就是说，Executor 是 EngineCore 初始化模型执行环境的直接接口。

---

## 12. 控制面能力

### 12.1 LoRA 管理

接口：

```text
add_lora(lora_request)
remove_lora(lora_id)
pin_lora(lora_id)
list_loras()
```

位置：`abstract.py:292` 到 `abstract.py:308`

Executor 只负责转发，具体实现由 Worker / ModelRunner 完成。

### 12.2 多模态缓存

接口：

```text
reset_mm_cache()
reset_encoder_cache()
```

位置：`abstract.py:310` 到 `abstract.py:316`

这说明 Executor 也要照顾多模态和 encoder cache 的控制接口。

### 12.3 profile

接口：

```text
profile(is_start=True, profile_prefix=None)
```

位置：`abstract.py:256`

对 worker 统一开启或关闭 profiler。

### 12.4 save_sharded_state / sleep / wake_up

`save_sharded_state()` 位于 `abstract.py:259` 到 `abstract.py:268`，通过 collective RPC 转发到 worker 保存分片权重状态。

`sleep()` / `wake_up()` 位于 `abstract.py:318` 到 `abstract.py:357`

Executor 的睡眠逻辑是：

```text
1. 广播 sleep 到所有 worker；
2. 记录 sleeping tags；
3. 标记 executor 为 sleeping；
4. wake_up 时按 tag 唤醒。
```

`UniProcExecutor` 和 `Worker` 侧分别会把这个控制落到内存池、权重、KV cache 等资源层。

### 12.5 shutdown

接口：

```python
shutdown()
```

位置：`abstract.py:276`

抽象层只要求广播 shutdown；
多进程实现还要负责真正杀 worker 进程和回收队列。

---

## 13. 多进程执行里 Executor 具体做了什么

以 `MultiprocExecutor` 为例，它除了继承 Executor 的通用接口，还负责：

```text
- 创建 message queues
- 启动 worker process
- wait_for_ready
- worker health monitor
- worker_response_mq / rpc_broadcast_mq 管理
- output_rank 选择
- worker termination
```

### 13.1 worker 监控

如果 worker 异常退出，会：

```text
1. 标记 executor failed;
2. shutdown executor;
3. 调用 failure callback。
```

位置：`multiproc_executor.py:268` 到 `multiproc_executor.py:305`

### 13.2 结果收集策略

`collective_rpc()` 会：

```text
- 通过广播队列发送命令；
- 从 response queue 接收结果；
- 支持 unique_reply_rank；
- 支持 kv_output_aggregator；
- 支持 FutureWrapper 异步返回。
```

位置：`multiproc_executor.py:343` 到 `multiproc_executor.py:405`

### 13.3 为什么要有 output_rank

在 TP / PP 场景里，不是每个 worker 的结果都要回传给上层。

例如 `ModelRunnerOutput` 往往只需要从：

```text
last PP stage + first TP rank
```

取回。

所以 Executor 会在分布式拓扑中选一个输出 rank。

---

## 14. Ray 后端里的 Executor 逻辑

Ray 路径的特点和 multiprocessing 略不同。

### 14.1 RayDistributedExecutor

它强调：

```text
- Ray cluster 初始化
- actor 创建
- compiled DAG 执行
- scheduler_output 暂存
- sample_tokens() 触发真正的 DAG 执行
```

### 14.2 为什么 execute_model 和 sample_tokens 分开

因为 Ray 后端里，某些情况下：

```text
execute_model() 只是设置状态；
sample_tokens() 才真正触发 DAG 并返回结果。
```

这与 async scheduling 配合得比较紧。

### 14.3 KV connector 场景

RayExecutorV2 / RayDistributedExecutor 还要考虑：

```text
- connector 是否存在
- 是否需要聚合各 worker 的 KV 输出
- 是否只从一个 rank 取结果
```

这也是 Executor 不只是“call worker”的原因：它还要做结果收束和协议适配。

---

## 15. 和 EngineCore 的关系

### 15.1 EngineCore 怎么依赖 Executor

在 `engine/core.py` 中，EngineCore 用 Executor 做这些事：

```text
get_kv_cache_specs()
determine_available_memory()
initialize_from_config()
execute_model()
sample_tokens()
get_kv_connector_handshake_metadata()
execute_dummy_batch()
take_draft_token_ids()
```

### 15.2 这说明什么

EngineCore 本身不直接接触 worker 细节，它只认识 Executor。

也就是说：

```text
EngineCore 负责 schedule / execute / update 的闭环；
Executor 负责把 execute 这一步真正落到 worker 上。
```

---

## 16. 容易混淆的点

### 16.1 Executor 是不是模型执行器？

不是。

它不做 forward，不做 attention，不做采样算法本身。

它更像：

```text
执行层的控制面 / 运行时桥接器。
```

### 16.2 `execute_model()` 是不是一定同步返回结果？

不是。

它可能返回：

```text
- ModelRunnerOutput
- Future[ModelRunnerOutput]
- None
```

取决于后端和调度模式。

### 16.3 `sample_tokens()` 是不是总会被调用？

也不是。

只有当 `execute_model()` 没有直接产出最终结果，或者后端设计要求分离采样时，才会单独调用。

### 16.4 `collective_rpc()` 是不是只用于多进程？

不是。

它是 Executor 的统一抽象，即便单进程后端也会沿用同样的接口语义。

---

## 17. 最小心智模型

如果只记一条主线，可以记：

```text
Executor 是 vLLM V1 的执行桥：
它把 EngineCore 的调度结果送到 Worker，
把 Worker 的执行结果收回来，
并统一处理分布式、生命周期和控制面操作。
```

再压缩成一句话：

```text
Scheduler 决定跑什么，Executor 决定怎么把它发给 Worker，Worker / ModelRunner 决定真正怎么跑。
```
