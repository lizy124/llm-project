# 05. Ray 执行器

vLLM V1 中有两套 Ray executor：

1. `RayDistributedExecutor`：旧 Ray 后端，Ray actor + Ray remote control plane + Ray compiled DAG execution plane。
2. `RayExecutorV2`：新 Ray 后端，Ray actor 只负责资源调度和 worker 容器，控制面复用 `MultiprocExecutor` 的 MessageQueue。

两者架构差异很大，不能简单认为 Ray v2 是 Ray v1 的小改版。

相关文件：

- `code/vllm/vllm/v1/executor/ray_executor.py`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py`
- `code/vllm/vllm/v1/executor/ray_utils.py`

## 1. Ray 后端选择

`Executor.get_class()` 中：

```text
backend == "ray"
  -> if VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
         RayExecutorV2
     else:
         RayDistributedExecutor
```

源码：

- `code/vllm/vllm/v1/executor/abstract.py:61`
- `code/vllm/vllm/envs.py:854`
- `code/vllm/vllm/envs.py:856`

## 2. RayDistributedExecutor 总览

`RayDistributedExecutor` 是旧 Ray 后端。

职责：

- 初始化 Ray。
- 创建 Ray actor worker。
- 使用 placement group 做资源放置。
- 为每个 worker 设置 rank/local_rank。
- 通过 Ray remote call 执行控制面 RPC。
- 用 Ray compiled DAG 执行模型路径。

类位置：

- `code/vllm/vllm/v1/executor/ray_executor.py:64`

## 3. RayDistributedExecutor 初始化

初始化涉及：

1. 初始化 Ray cluster。
2. 创建 placement group。
3. 创建 `RayWorkerWrapper` actors。
4. 获取每个 actor 的 node id / IP / GPU IDs。
5. 调整 rank 顺序。
6. 设置每个 worker 可见设备。
7. 初始化 worker。
8. load model。

关键源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:70`
- `code/vllm/vllm/v1/executor/ray_executor.py:78`
- `code/vllm/vllm/v1/executor/ray_executor.py:143`
- `code/vllm/vllm/v1/executor/ray_executor.py:189`
- `code/vllm/vllm/v1/executor/ray_executor.py:253`
- `code/vllm/vllm/v1/executor/ray_executor.py:259`

## 4. Ray actor、CUDA_VISIBLE_DEVICES 与 local_rank

旧 Ray executor 不让 Ray 自动设置 worker 可见 GPU，而是 vLLM 自己根据 Ray 分配的 GPU ID 设置。

原因：

- 同一节点内一个 vLLM executor 可能占用多个 GPU。
- 每个 worker 需要看到本 executor 在该节点上的所有 GPU。
- worker 再根据 `local_rank` 选择自己的 device。

相关源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:162`
- `code/vllm/vllm/v1/executor/ray_executor.py:261`
- `code/vllm/vllm/v1/executor/ray_executor.py:300`
- `code/vllm/vllm/v1/executor/ray_executor.py:310`
- `code/vllm/vllm/v1/executor/ray_executor.py:355`

## 5. RayDistributedExecutor 控制面 RPC

`collective_rpc()` 直接调用 Ray actor 方法：

```text
worker.execute_method.remote(method, *args, **kwargs)
```

它不是 MQ。

源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:479`
- `code/vllm/vllm/v1/executor/ray_executor.py:505`

## 6. Ray compiled DAG 执行面

旧 Ray executor 的模型执行主要通过 Ray compiled DAG。

DAG 形态：

```text
input: (SchedulerOutput, GrammarOutput)
  -> fan-out 到第一 PP stage 的 TP workers
  -> 每个 PP stage 内 TP group SPMD 执行
  -> 非最后 PP stage 输出 IntermediateTensors
  -> 下一个 PP stage 接收 IntermediateTensors
  -> 最后 stage 采样并返回 ModelRunnerOutput
```

源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:536`
- `code/vllm/vllm/v1/executor/ray_executor.py:562`
- `code/vllm/vllm/v1/executor/ray_executor.py:569`
- `code/vllm/vllm/v1/executor/ray_executor.py:580`
- `code/vllm/vllm/v1/executor/ray_executor.py:591`
- `code/vllm/vllm/v1/executor/ray_executor.py:604`
- `code/vllm/vllm/v1/executor/ray_executor.py:622`

## 7. RayDistributedExecutor execute_model / sample_tokens

旧 Ray executor 中，`execute_model()` 与 `sample_tokens()` 有特殊状态机。

### 7.1 execute_model

如果本 step 不需要采样，可以直接执行 DAG。

如果需要采样：

```text
execute_model(scheduler_output)
  -> 缓存 scheduler_output
  -> 返回 None 或 completed none future
  -> 等 sample_tokens(grammar_output) 再执行 DAG
```

源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:399`
- `code/vllm/vllm/v1/executor/ray_executor.py:410`
- `code/vllm/vllm/v1/executor/ray_executor.py:414`

### 7.2 sample_tokens

```text
sample_tokens(grammar_output)
  -> 取出缓存的 scheduler_output
  -> _execute_dag(scheduler_output, grammar_output)
```

源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:418`
- `code/vllm/vllm/v1/executor/ray_executor.py:441`

### 7.3 _execute_dag

```text
_execute_dag(scheduler_output, grammar_output)
  -> 首次构建 compiled Ray DAG
  -> forward_dag.execute((scheduler_output, grammar_output))
  -> 无 connector: ray.get(output_rank ref)
  -> 有 connector: ray.get(all refs) + kv_output_aggregator.aggregate
```

源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:443`
- `code/vllm/vllm/v1/executor/ray_executor.py:449`
- `code/vllm/vllm/v1/executor/ray_executor.py:453`
- `code/vllm/vllm/v1/executor/ray_executor.py:455`
- `code/vllm/vllm/v1/executor/ray_executor.py:467`

## 8. Ray worker wrapper execute_model_ray

Ray DAG 中每个 actor 执行 `execute_model_ray()`。

输入可能是：

```text
(SchedulerOutput, GrammarOutput)
(SchedulerOutput, GrammarOutput, IntermediateTensors)
```

执行流程：

```text
execute_model_ray(data)
  -> 设置 device
  -> 解包 scheduler_output / grammar_output / intermediate_tensors
  -> worker.model_runner.execute_model(scheduler_output, intermediate_tensors)
  -> 如果返回 IntermediateTensors，传给下一 PP stage
  -> 如果最后 rank 且 execute_model 返回 None，则调用 sample_tokens(grammar_output)
  -> 如果是 AsyncModelRunnerOutput，先 get_output()
```

源码：

- `code/vllm/vllm/v1/executor/ray_utils.py:123`
- `code/vllm/vllm/v1/executor/ray_utils.py:131`
- `code/vllm/vllm/v1/executor/ray_utils.py:135`
- `code/vllm/vllm/v1/executor/ray_utils.py:142`
- `code/vllm/vllm/v1/executor/ray_utils.py:146`
- `code/vllm/vllm/v1/executor/ray_utils.py:161`
- `code/vllm/vllm/v1/executor/ray_utils.py:168`

注意：注释中可能提到 `ExecuteModelRequest`，但当前 V1 代码实际传递的是 `SchedulerOutput` 与 `GrammarOutput`，并没有真实的 `ExecuteModelRequest` 类作为主路径数据结构。

## 9. RayDistributedExecutor 健康检查与关闭

旧 Ray executor 的健康检查偏乐观：

- `check_health()` 基本假定 Ray workers 健康。

shutdown：

- teardown compiled DAG。
- kill Ray workers。

源码：

- `code/vllm/vllm/v1/executor/ray_executor.py:99`
- `code/vllm/vllm/v1/executor/ray_executor.py:107`
- `code/vllm/vllm/v1/executor/ray_executor.py:634`

## 10. RayExecutorV2 总览

`RayExecutorV2` 继承 `MultiprocExecutor`。

核心思想：

> Ray 负责资源调度和 actor 容器；vLLM 继续使用 Multiproc 的 MQ 控制面和自己的 distributed 通信。

类位置：

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:205`

组件：

- `RayWorkerHandle`：`code/vllm/vllm/v1/executor/ray_executor_v2.py:48`
- `RayWorkerProc`：`code/vllm/vllm/v1/executor/ray_executor_v2.py:75`
- `RayExecutorV2`：`code/vllm/vllm/v1/executor/ray_executor_v2.py:205`

## 11. RayExecutorV2 初始化流程

大致 10 步：

1. 初始化 Ray cluster。
2. 根据 placement group bundle 生成 rank 到 bundle/node 映射。
3. 选择 rank 0 所在节点 IP 作为 torch distributed init TCP 地址。
4. driver 创建 broadcast MQ。
5. 为每个 rank 创建 `RayWorkerProc` actor。
6. actor 查询 Ray runtime 分配的 node_id/gpu_ids。
7. 计算每个节点内 local_rank，并设置 visible devices。
8. 收集 response MQ handle。
9. 调用 actor `run.remote()` 启动 busy loop。
10. driver 与 workers 执行 MQ ready barrier。

源码位置：

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:249`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:260`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:271`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:297`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:304`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:340`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:369`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:381`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:400`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:413`

## 12. RayWorkerProc 两阶段初始化

Ray actor 构造时不能立刻做完整 worker 初始化，因为 CUDA device / visible devices 需要等 Ray placement 完成。

因此 RayWorkerProc 分两阶段：

```text
Ray actor __init__
  -> 只保存参数

initialize_worker
  -> 获取 Ray 分配的 GPU/node 信息
  -> 设置 visible devices
  -> 调 WorkerProc.__init__
  -> 初始化真实 worker
```

源码：

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:75`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:103`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:123`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:134`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:152`

## 13. RayExecutorV2 控制面

RayExecutorV2 复用 MultiprocExecutor 的控制面：

- broadcast MQ。
- response MQ。
- output_rank。
- `FutureWrapper`。
- worker busy loop。

差异只在 worker 容器：

```text
MultiprocExecutor: local Python child process
RayExecutorV2: Ray actor running WorkerProc-like loop
```

## 14. RayExecutorV2 failure handling

Ray v2 通过 monitor thread 使用 `ray.wait()` 监控 actor 的 `run_ref`。

如果任一 actor run 结束：

```text
actor run returned unexpectedly
  -> is_failed = True
  -> shutdown()
  -> failure_callback
```

为了避免 Ray shutdown 时阻塞，monitor 使用 5 秒 timeout 轮询。

源码：

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:429`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:444`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:450`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:463`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:468`

## 15. RayExecutorV2 shutdown

shutdown 特点：

- 加锁防止重入。
- join monitor thread。
- `ray.kill()` 所有 actor。
- 关闭 broadcast MQ 和 response MQ。

源码：

- `code/vllm/vllm/v1/executor/ray_executor_v2.py:498`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:504`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:509`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:511`
- `code/vllm/vllm/v1/executor/ray_executor_v2.py:518`

## 16. 两套 Ray 后端对比

| 方面 | RayDistributedExecutor | RayExecutorV2 |
|---|---|---|
| 继承关系 | 直接继承 Executor | 继承 MultiprocExecutor |
| worker 容器 | Ray actor | Ray actor |
| 控制面 | Ray remote call | Multiproc MQ |
| 执行面 | Ray compiled DAG | vLLM distributed + worker execute_model |
| PP 传递 | Ray DAG tensor transport | vLLM PP communicator |
| failure | 健康检查偏乐观 | ray.wait 监控 actor run |
| async scheduling | 非主支持路径 | 支持 |

## 17. 关键理解

1. 旧 Ray executor 是 Ray 原生控制面 + Ray compiled DAG。
2. Ray v2 是 Ray 调度壳 + Multiproc MQ 控制面。
3. Ray v2 更接近 MultiprocExecutor 的行为。
4. 当前实际执行数据是 `SchedulerOutput` / `GrammarOutput`，不是独立的 `ExecuteModelRequest` 类。
5. Ray executor 的复杂点主要在 placement、rank/local_rank、visible devices、PP/TP DAG 或 MQ 控制面差异。
