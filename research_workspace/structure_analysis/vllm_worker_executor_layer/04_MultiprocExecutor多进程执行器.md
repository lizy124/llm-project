# 04. MultiprocExecutor 多进程执行器

`MultiprocExecutor` 是 vLLM V1 本地多进程执行器，用于单机多 worker / 多 GPU 场景。它通过 MessageQueue 做控制面广播，通过每个 worker 的 response queue 收敛结果。

源码文件：

- `code/vllm/vllm/v1/executor/multiproc_executor.py`

## 1. 核心定位

`MultiprocExecutor` 负责：

- 拉起 worker 子进程。
- 为所有 worker 建立广播 MQ。
- 为每个 worker 建立 response MQ。
- 等待 worker 初始化 ready。
- 执行 collective RPC。
- 管理 output rank。
- 支持 non-block Future。
- 监控 worker 异常退出。
- 关闭 worker 与 MQ。

关键类与位置：

- `MultiprocExecutor`：`code/vllm/vllm/v1/executor/multiproc_executor.py:110`
- `FutureWrapper`：`code/vllm/vllm/v1/executor/multiproc_executor.py:70`
- `WorkerProc`：`code/vllm/vllm/v1/executor/multiproc_executor.py:593`

## 2. 初始化整体流程

`_init_executor()` 大致流程：

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

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:110`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:117`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:151`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:164`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:176`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:182`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:268`

## 3. rank 与 local_rank

Multiproc 中：

```text
local_rank = 当前节点内 worker 序号
global_rank = global_start_rank + local_rank
global_start_rank = local_world_size * node_rank_within_dp
```

用途：

- `rank` 用于 distributed 全局通信。
- `local_rank` 用于本地设备选择。
- `is_driver_worker = rank % tensor_parallel_size == 0`。

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:164`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:176`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:177`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:265`

## 4. 控制面 MessageQueue

Multiproc 控制面有两类 MQ：

1. `rpc_broadcast_mq`
   - executor 写入。
   - 所有 worker 读取。
   - 用于广播方法调用。

2. `worker_response_mq`
   - 每个 worker 一个或一组。
   - worker 写结果。
   - executor 读取结果。

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:151`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:370`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:374`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:561`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:564`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:573`

跨节点 DP 内部场景：

- 当 `nnodes_within_dp > 1` 时，worker 会通过 `get_inner_dp_world_group()` 创建跨节点 MQ broadcaster。

相关位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:561`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:573`

## 5. collective_rpc fan-out / fan-in

Multiproc 的 `collective_rpc()` 是核心方法。

执行流程：

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

executor 侧源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:340`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:370`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:376`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:380`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:398`

worker 侧源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:806`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:921`

## 6. output_rank

多 rank 中，通常只有一个 rank 需要回传最终输出。

`MultiprocExecutor` 中：

```text
output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size
```

语义：

- 最后一个 pipeline stage。
- 该 stage 的第一个 TP rank。
- 负责返回 `ModelRunnerOutput`。

相关源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:307`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:313`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:495`

`execute_model()` 与 `sample_tokens()` 都使用 `unique_reply_rank=self.output_rank`：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:307`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:319`

## 7. execute_model / sample_tokens

### 7.1 execute_model

```text
MultiprocExecutor.execute_model(scheduler_output)
  -> collective_rpc("execute_model", args=(scheduler_output,), unique_reply_rank=output_rank)
  -> 若有 KV connector，可能通过 kv_output_aggregator 聚合
```

源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:307`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:317`

### 7.2 sample_tokens

```text
MultiprocExecutor.sample_tokens(grammar_output)
  -> collective_rpc("sample_tokens", args=(grammar_output,), unique_reply_rank=output_rank)
```

源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:319`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:329`

## 8. WorkerProc 生命周期

`WorkerProc` 是每个子进程的宿主。

职责：

- 创建进程。
- 在子进程中构造 `WorkerWrapperBase`。
- 初始化 worker。
- 进入 busy loop。
- 消费 RPC。
- 执行 worker 方法。
- 写回 response。
- 捕获异常。
- 处理父进程死亡。
- shutdown 时销毁分布式环境。

关键源码：

- 创建：`code/vllm/vllm/v1/executor/multiproc_executor.py:593`
- 初始化：`code/vllm/vllm/v1/executor/multiproc_executor.py:636`
- ready：`code/vllm/vllm/v1/executor/multiproc_executor.py:732`
- busy loop：`code/vllm/vllm/v1/executor/multiproc_executor.py:806`
- 异常处理：`code/vllm/vllm/v1/executor/multiproc_executor.py:969`
- shutdown：`code/vllm/vllm/v1/executor/multiproc_executor.py:770`

## 9. FutureWrapper

`FutureWrapper` 用于 non-block RPC。

作用：

- 保存取结果函数。
- `result()` 时才等待 MQ response。
- 保证 EngineCore 可以在模型执行期间做 grammar bitmask 等工作。

源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:70`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:101`

## 10. async output copy

async scheduling 开启时，worker 内部可能启动 `WorkerAsyncOutputCopy` daemon thread。

流程：

```text
worker execute/sample 得到 AsyncModelRunnerOutput
  -> handle_output 不立即写 response MQ
  -> 放入 async_output_queue
  -> 后台线程设置当前 CUDA device
  -> 调用 get_output() 等待 GPU->CPU copy
  -> 写 response MQ
```

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:636`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:639`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:645`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:926`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:931`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:941`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:946`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:951`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:962`

## 11. 失败处理

### 11.1 monitor thread

Multiproc 使用 monitor thread 等待 worker process sentinel。

任意 worker 异常退出：

```text
worker died
  -> is_failed = True
  -> shutdown executor
  -> failure_callback
```

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:268`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:276`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:282`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:287`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:301`

### 11.2 worker 内部异常

worker busy loop 捕获异常后：

- 记录 stack trace。
- 通过 response MQ 返回 FAILURE。
- executor `get_response()` 收到后抛 `RuntimeError`。

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:381`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:391`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:732`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:760`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:986`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:994`

### 11.3 父进程死亡检测

worker 持有 death pipe。

当父进程退出：

```text
death pipe EOF
  -> worker 关闭 MQ
  -> worker 退出
```

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:781`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:787`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:791`

## 12. shutdown

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

源码位置：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:404`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:430`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:436`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:446`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:456`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:467`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:481`

worker 子进程关闭时：

- 关闭 MQ。
- 调用 `worker.shutdown()`。
- 销毁 model parallel。
- 销毁 distributed environment。

源码：

- `code/vllm/vllm/v1/executor/multiproc_executor.py:770`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:775`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:778`
- `code/vllm/vllm/v1/executor/multiproc_executor.py:779`

## 13. Multiproc 主链路

```text
EngineCore.step
  -> MultiprocExecutor.execute_model
  -> collective_rpc("execute_model")
  -> rpc_broadcast_mq.enqueue(method, args, kwargs, output_rank)
  -> all WorkerProc busy loop receive RPC
  -> WorkerWrapperBase.execute_model
  -> Worker.execute_model
  -> ModelRunner.execute_model
  -> output_rank worker writes response MQ
  -> FutureWrapper.result reads response
  -> maybe sample_tokens repeats same pattern
  -> EngineCore receives ModelRunnerOutput
```

## 14. 关键理解

1. Multiproc 是“广播调用，指定 rank 回包”。
2. 所有 worker 都要执行同一 RPC，因为 TP/PP 都参与计算。
3. 只有 output rank 返回最终输出，避免控制面结果爆炸。
4. 子进程中的真实执行仍然是 Worker/ModelRunner。
5. Multiproc 的控制面和分布式计算通信是两套东西：MQ 负责 RPC，NCCL/Gloo 负责模型并行通信。
