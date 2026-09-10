# 03. UniProcExecutor 单进程执行器

`UniProcExecutor` 是 vLLM V1 最简单的 executor 实现。它只在当前进程内创建一个 worker wrapper，不涉及真实 IPC，是理解 Executor/Worker 边界的最佳起点。

源码文件：

- `code/vllm/vllm/v1/executor/uniproc_executor.py`

## 1. 核心定位

`UniProcExecutor` 的特点：

- 单进程。
- 单 worker。
- 不启动子进程。
- 不创建 Ray actor。
- `collective_rpc()` 实际就是本进程函数调用。
- 仍然遵守 executor 抽象接口，因此 EngineCore 不需要特殊处理。

类定义位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:45`

## 2. 初始化流程

`_init_executor()` 是初始化核心。

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

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:46`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:70`

### 2.1 WorkerWrapperBase 创建

UniProc 不直接实例化 GPU worker，而是先创建 `WorkerWrapperBase`。

原因：executor 统一面对 wrapper，真实 worker class 由 wrapper 根据 `parallel_config.worker_cls` 延迟解析。

相关位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:46`
- `code/vllm/vllm/v1/worker/worker_base.py:187`
- `code/vllm/vllm/v1/worker/worker_base.py:229`

### 2.2 distributed_init_method

虽然 UniProc 是单 worker，但仍然构造 distributed init method。

这保证单 worker 路径和多 worker 路径尽量复用 Worker 的分布式初始化逻辑。

相关位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:46`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1164`
- `code/vllm/vllm/distributed/parallel_state.py:1516`

### 2.3 init_worker / init_device / load_model

UniProc 初始化阶段会连续调用：

```text
init_worker
  -> wrapper 解析并构造真实 Worker

init_device
  -> Worker 初始化 device、distributed、model runner

load_model
  -> Worker/ModelRunner 加载模型权重
```

关键位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:46-70`
- `code/vllm/vllm/v1/worker/worker_base.py:229`
- `code/vllm/vllm/v1/worker/worker_base.py:321`
- `code/vllm/vllm/v1/worker/gpu_worker.py:249`
- `code/vllm/vllm/v1/worker/gpu_worker.py:349`

## 3. collective_rpc：直接函数调用

UniProc 的 `collective_rpc()` 没有真正的 RPC。

同步模式：

```text
collective_rpc(method, args, kwargs, non_block=False)
  -> run_method(self.driver_worker, method, args, kwargs)
  -> 如返回 AsyncModelRunnerOutput，则 get_output()
  -> 返回结果
```

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:79`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:91`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:92`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:95`

这里的 `run_method()` 负责按方法名在对象上调用对应方法。

## 4. 非阻塞 Future 语义

EngineCore 通常以非阻塞形式调用：

```text
future = model_executor.execute_model(scheduler_output, non_block=True)
```

UniProc 虽然是直接调用，但仍然要返回 Future 风格对象，以统一 executor 接口。

两类包装：

1. 普通结果：包装为已完成的 `Future`。
2. `AsyncModelRunnerOutput`：包装为 `AsyncOutputFuture`，在 `result()` 时调用 `get_output()`。

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:26`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:36`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:97`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:106`

## 5. execute_model

`UniProcExecutor.execute_model()` 的核心逻辑：

```text
execute_model(scheduler_output, non_block)
  -> collective_rpc("execute_model", args=(scheduler_output,), single_value=True)
  -> 如果 non_block 且 future 已完成，提前 surface 异常
  -> 返回 Future 或结果
```

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:108`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:121`

最终会调用：

```text
WorkerWrapperBase.execute_model
  -> _apply_mm_cache(scheduler_output)
  -> self.worker.execute_model(scheduler_output)
```

源码位置：

- `code/vllm/vllm/v1/worker/worker_base.py:340`
- `code/vllm/vllm/v1/worker/worker_base.py:345`

## 6. sample_tokens

`sample_tokens()` 沿用抽象父类的 RPC 包装，最终调用 worker 的 `sample_tokens()`。

关键位置：

- `code/vllm/vllm/v1/executor/abstract.py:241`
- `code/vllm/vllm/v1/worker/gpu_worker.py:801`

## 7. 支持 async scheduling

UniProc 声明支持 async scheduling：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:145`

这意味着它可以配合 EngineCore 的 batch queue / async output 路径工作。

## 8. ExecutorWithExternalLauncher

同文件还定义了 `ExecutorWithExternalLauncher`。

它用于外部进程启动方式，例如 torchrun 或其他 launcher 已经设置好：

- `RANK`
- `LOCAL_RANK`
- `MASTER_ADDR`
- `MASTER_PORT`

特点：

- 使用 `env://` 初始化。
- 不由 vLLM executor 自己创建所有 worker 进程。
- 每个外部进程内可表现得接近 UniProc。

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py:150`
- `code/vllm/vllm/v1/executor/uniproc_executor.py:197`

## 9. UniProc 的意义

UniProc 不适合解释多 rank 通信，但适合理解：

1. EngineCore 如何调用 executor。
2. Executor 如何调用 WorkerWrapper。
3. WorkerWrapper 如何调用真实 Worker。
4. Worker 如何调用 ModelRunner。
5. `execute_model()` 和 `sample_tokens()` 的基本边界。

最短链路：

```text
EngineCore.step
  -> UniProcExecutor.execute_model
  -> UniProcExecutor.collective_rpc
  -> WorkerWrapperBase.execute_model
  -> Worker.execute_model
  -> ModelRunner.execute_model
  -> maybe sample_tokens
```

## 10. 与 Multiproc/Ray 的差异

UniProc 和其他 executor 的接口相同，但省略了：

- worker 子进程。
- broadcast MQ。
- response MQ。
- Ray actor。
- output rank 选择。
- worker monitor。

因此它是“语义完整、机制最简单”的 executor。
