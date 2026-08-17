# 02. Executor 抽象与后端选择

`Executor` 是 vLLM V1 执行层的上层抽象。它面向 `EngineCore` 提供统一接口，屏蔽单进程、多进程、Ray 等不同 worker 部署方式。

源码主文件：

- `code/vllm/vllm/v1/executor/abstract.py`

## 1. Executor 的核心定位

`Executor` 的职责不是模型计算，而是执行控制面：

- 根据配置选择执行后端。
- 初始化 worker。
- 下发 KV cache 配置。
- 对 worker 做 collective RPC。
- 执行模型调用。
- 执行采样调用。
- 健康检查。
- shutdown。

关键抽象位置：

- `code/vllm/vllm/v1/executor/abstract.py:37`
- `code/vllm/vllm/v1/executor/abstract.py:47`
- `code/vllm/vllm/v1/executor/abstract.py:94`
- `code/vllm/vllm/v1/executor/abstract.py:152`
- `code/vllm/vllm/v1/executor/abstract.py:221`
- `code/vllm/vllm/v1/executor/abstract.py:241`

## 2. 后端选择 get_class

`Executor.get_class()` 根据 `parallel_config.distributed_executor_backend` 选择具体实现。

主要分支：

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

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py:48`
- `code/vllm/vllm/v1/executor/abstract.py:61`
- `code/vllm/vllm/v1/executor/abstract.py:69`
- `code/vllm/vllm/v1/executor/abstract.py:73`

Ray v2 开关：

- `code/vllm/vllm/envs.py:854`
- `code/vllm/vllm/envs.py:856`

## 3. Executor 初始化公共逻辑

`Executor.__init__()` 负责保存配置并调用具体后端的 `_init_executor()`。

抽象流程：

```text
Executor.__init__
  -> 保存 vllm_config
  -> 保存 failure_callback
  -> 调用 self._init_executor()
```

具体 worker 创建、MQ 创建、Ray actor 创建都在各子类的 `_init_executor()` 中。

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py:94`
- `code/vllm/vllm/v1/executor/abstract.py:102`

## 4. EngineCore 初始化时如何调用 Executor

`EngineCore` 初始化时会创建 executor，并通过 executor 完成 worker 初始化之后的 KV cache 配置流程。

典型顺序：

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

相关源码：

- `code/vllm/vllm/v1/engine/core.py:96`
- `code/vllm/vllm/v1/engine/core.py:239`
- `code/vllm/vllm/v1/engine/core.py:348`

## 5. collective_rpc：最核心的抽象方法

`collective_rpc()` 是 executor 抽象的中心。

它表示：

> 在 worker 集合上调用某个方法，并按 executor 后端的规则收集结果。

不同实现：

- UniProc：本进程直接调用。
- Multiproc：向 broadcast MQ 写 RPC 请求，worker 从 MQ 消费。
- RayDistributedExecutor：Ray actor remote call。
- RayExecutorV2：复用 Multiproc 的 MQ 机制，但 worker 容器是 Ray actor。

抽象定义位置：

- `code/vllm/vllm/v1/executor/abstract.py:152`

## 6. execute_model 默认语义

抽象层的 `execute_model()` 默认只是 collective RPC 包装：

```text
Executor.execute_model(scheduler_output, non_block)
  -> collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
  -> 默认取 output[0]
```

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py:221`
- `code/vllm/vllm/v1/executor/abstract.py:227`

注意：很多具体 executor 会覆盖它，以处理：

- output rank。
- connector 输出聚合。
- Ray compiled DAG。
- `execute_model` / `sample_tokens` 两阶段状态。

## 7. sample_tokens 默认语义

`sample_tokens()` 同样是 collective RPC 包装：

```text
Executor.sample_tokens(grammar_output, non_block)
  -> collective_rpc("sample_tokens", args=(grammar_output,), non_block=non_block)
```

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py:241`
- `code/vllm/vllm/v1/executor/abstract.py:247`

## 8. initialize_from_config

KV cache 配置不是 worker 自己决定的。

典型流程：

1. Worker 暴露 KV cache spec。
2. Worker profile 可用显存/内存。
3. EngineCore 计算每个 worker 的 `KVCacheConfig`。
4. Executor 调用 `initialize_from_config(kv_cache_configs)` 下发给 worker。
5. Worker/ModelRunner 依据配置分配 KV cache tensors。

Executor 层方法：

- `code/vllm/vllm/v1/executor/abstract.py:121`
- `code/vllm/vllm/v1/executor/abstract.py:138`

Worker wrapper 侧接收：

- `code/vllm/vllm/v1/worker/worker_base.py:315`

GPU worker 初始化：

- `code/vllm/vllm/v1/worker/gpu_worker.py:562`

## 9. Executor 与 output_rank

在多 rank 场景中，不是所有 rank 都需要返回最终输出。

典型规则：

- 所有 rank 都要执行 forward，因为 TP/PP 都参与计算。
- 但最终 `ModelRunnerOutput` 通常只需要最后一个 PP stage 的某个 TP rank 返回。
- Multiproc 中使用 `output_rank` 来指定回包 rank。

Multiproc 计算：

- `output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size`

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:307`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:313`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:495`

## 10. Executor 与 async scheduling

部分 executor 声明支持 async scheduling。

支持者：

- `UniProcExecutor`
- `MultiprocExecutor`
- `RayExecutorV2`

旧 `RayDistributedExecutor` 没有同等支持声明。

相关位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:145`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:511`
- `code/vllm/vllm/config/vllm.py:950`
- `code/vllm/vllm/config/vllm.py:970`
- `code/vllm/vllm/config/vllm.py:1016`
- `code/vllm/vllm/config/vllm.py:1021`
- `code/vllm/vllm/config/vllm.py:1030`

async scheduling 影响：

- `execute_model(..., non_block=True)` 更重要。
- worker 可能返回 `AsyncModelRunnerOutput`。
- Multiproc worker 内部可能启动 async output copy thread。
- DP synchronization 默认可能禁用 NCCL，改走 Gloo，避免多流/异步场景问题。

## 11. 具体 executor 对比

| 方面 | UniProc | Multiproc | RayDistributed | RayExecutorV2 |
|---|---|---|---|---|
| worker 容器 | 本进程对象 | Python 子进程 | Ray actor | Ray actor |
| 控制面 | 直接调用 | MQ | Ray remote | MQ |
| 数据/执行面 | 本进程 | vLLM distributed | Ray compiled DAG | vLLM distributed |
| output rank | 单 rank | 支持 | DAG 输出 | 支持 |
| failure monitor | 简单 | 进程 sentinel | 偏乐观 | ray.wait 监控 actor run |
| async scheduling | 支持 | 支持 | 不作为主路径 | 支持 |

## 12. 理解 Executor 的关键点

1. Executor 是控制层，不是计算层。
2. Executor 的核心抽象是 `collective_rpc()`。
3. 不同 executor 的主要差异是 worker 部署方式和 RPC/IPC 方式。
4. Worker 真实执行模型，ModelRunner 才是 forward/sampling 主体。
5. 多 rank 下通常只从 output rank 收最终结果。
6. Ray v1 和 Ray v2 架构差异很大，不能混为一谈。
