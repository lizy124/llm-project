# 11. 端到端调用链路

本文从请求进入 vLLM V1 开始，串起 EngineCore、Scheduler、Executor、Worker、ModelRunner、Scheduler output update 的完整链路。

注意：当前 V1 代码主路径中没有独立的 `ExecuteModelRequest` 类。实际从 scheduler 传给 executor/worker 的对象是 `SchedulerOutput`，结构化输出约束通过 `GrammarOutput` 在采样阶段传递。

## 1. 请求进入 LLMEngine / AsyncLLM

### 1.1 同步 LLMEngine

`LLMEngine.add_request()`：

```text
外部请求
  -> InputProcessor.process_inputs
  -> EngineCoreRequest
  -> assign_request_id
  -> output_processor 注册
  -> engine_core.add_request
```

源码：

- `code/vllm/vllm/v1/engine/llm_engine.py:250`
- `code/vllm/vllm/v1/engine/llm_engine.py:263`
- `code/vllm/vllm/v1/engine/llm_engine.py:272`
- `code/vllm/vllm/v1/engine/llm_engine.py:276`

`LLMEngine.step()`：

```text
engine_core.get_output
  -> output_processor.process_outputs
  -> 用户可见 RequestOutput / PoolingRequestOutput
```

源码：

- `code/vllm/vllm/v1/engine/llm_engine.py:302`
- `code/vllm/vllm/v1/engine/llm_engine.py:306`
- `code/vllm/vllm/v1/engine/llm_engine.py:314`

### 1.2 异步 AsyncLLM

`AsyncLLM.add_request()` 类似：

```text
InputProcessor.process_inputs
  -> assign_request_id
  -> _add_request
  -> output_processor 注册
  -> engine_core.add_request_async
```

源码：

- `code/vllm/vllm/v1/engine/async_llm.py:349`
- `code/vllm/vllm/v1/engine/async_llm.py:368`
- `code/vllm/vllm/v1/engine/async_llm.py:381`
- `code/vllm/vllm/v1/engine/async_llm.py:408`
- `code/vllm/vllm/v1/engine/async_llm.py:411`

## 2. EngineCoreRequest

`EngineCoreRequest` 是进入 EngineCore 的请求结构。

关键字段：

- `request_id`
- `prompt_token_ids`
- `mm_features`
- `sampling_params`
- `pooling_params`
- `arrival_time`
- `lora_request`
- `cache_salt`
- `data_parallel_rank`
- `prompt_embeds`
- `client_index`
- `current_wave`
- `priority`
- `trace_headers`
- `abort_immediately`

定义：

- `code/vllm/vllm/v1/engine/__init__.py:86`
- `code/vllm/vllm/v1/engine/__init__.py:143`

## 3. EngineCoreClient

EngineCoreClient 分 in-process 和 multiprocess/ZMQ 两类。

### 3.1 InprocClient

添加请求：

```text
InprocClient.add_request
  -> engine_core.preprocess_add_request
  -> engine_core.add_request
```

取输出：

```text
InprocClient.get_output
  -> engine_core.step_fn
  -> engine_core.post_step
  -> EngineCoreOutputs
```

源码：

- `code/vllm/vllm/v1/engine/core_client.py:289`
- `code/vllm/vllm/v1/engine/core_client.py:292`
- `code/vllm/vllm/v1/engine/core_client.py:297`
- `code/vllm/vllm/v1/engine/core_client.py:299`

### 3.2 多进程 / ZMQ Client

同步 MP client：

- `get_output()` 从 outputs queue 取 `EngineCoreOutputs`。
- `_send_input()` 编码后通过 socket 发给 EngineCoreProc。

源码：

- `code/vllm/vllm/v1/engine/core_client.py:849`
- `code/vllm/vllm/v1/engine/core_client.py:859`
- `code/vllm/vllm/v1/engine/core_client.py:861`
- `code/vllm/vllm/v1/engine/core_client.py:873`

异步 MP client：

- `code/vllm/vllm/v1/engine/core_client.py:1053`
- `code/vllm/vllm/v1/engine/core_client.py:1064`

## 4. EngineCore 预处理请求

`EngineCore.preprocess_add_request()`：

```text
EngineCoreRequest
  -> 如有 MM cache，处理 mm_features
  -> Request.from_engine_core_request
  -> 如有 structured output，初始化 grammar
  -> 返回 (Request, current_wave)
```

源码：

- `code/vllm/vllm/v1/engine/core.py:862`
- `code/vllm/vllm/v1/engine/core.py:867`
- `code/vllm/vllm/v1/engine/core.py:868`
- `code/vllm/vllm/v1/engine/core.py:875`

`EngineCore.add_request()`：

```text
校验 request id
  -> 校验 pooling task
  -> scheduler.add_request(request)
  -> 如 abort_immediately，立即 abort
```

源码：

- `code/vllm/vllm/v1/engine/core.py:378`
- `code/vllm/vllm/v1/engine/core.py:384`
- `code/vllm/vllm/v1/engine/core.py:403`
- `code/vllm/vllm/v1/engine/core.py:404`

## 5. EngineCoreProc busy loop

多进程 EngineCoreProc 主循环：

```text
run_busy_loop
  -> _process_input_queue
  -> _process_engine_step
```

源码：

- `code/vllm/vllm/v1/engine/core.py:1257`
- `code/vllm/vllm/v1/engine/core.py:1264`

输入处理：

```text
_process_input_queue
  -> 无 work 时阻塞等输入
  -> 有输入则 _handle_client_request
```

源码：

- `code/vllm/vllm/v1/engine/core.py:1267`
- `code/vllm/vllm/v1/engine/core.py:1296`

client request：

- ADD：调用 `add_request`。
- ABORT：调用 `abort_requests`。
- UTILITY：执行工具方法并写 output queue。

源码：

- `code/vllm/vllm/v1/engine/core.py:1370`
- `code/vllm/vllm/v1/engine/core.py:1399`

engine step：

```text
_process_engine_step
  -> outputs, model_executed = step_fn()
  -> output_queue.put(outputs)
  -> post_step(model_executed)
```

源码：

- `code/vllm/vllm/v1/engine/core.py:1301`
- `code/vllm/vllm/v1/engine/core.py:1303`
- `code/vllm/vllm/v1/engine/core.py:1306`

## 6. EngineCore.step 主链路

核心源码：

- `code/vllm/vllm/v1/engine/core.py:479`
- `code/vllm/vllm/v1/engine/core.py:508`

流程：

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

源码点：

- `code/vllm/vllm/v1/engine/core.py:486`
- `code/vllm/vllm/v1/engine/core.py:490`
- `code/vllm/vllm/v1/engine/core.py:491`
- `code/vllm/vllm/v1/engine/core.py:492`
- `code/vllm/vllm/v1/engine/core.py:497`
- `code/vllm/vllm/v1/engine/core.py:498`
- `code/vllm/vllm/v1/engine/core.py:501`
- `code/vllm/vllm/v1/engine/core.py:504`
- `code/vllm/vllm/v1/engine/core.py:508`

## 7. step_with_batch_queue 变体

async scheduling / batch queue 路径：

- 先填 batch queue。
- `execute_model(..., non_block=True)`。
- 必要时 `sample_tokens(..., non_block=True)`。
- 等最早 future。
- 仍然通过 `scheduler.update_from_output()` 生成 outputs。

源码：

- `code/vllm/vllm/v1/engine/core.py:519`
- `code/vllm/vllm/v1/engine/core.py:547`
- `code/vllm/vllm/v1/engine/core.py:561`
- `code/vllm/vllm/v1/engine/core.py:590`
- `code/vllm/vllm/v1/engine/core.py:605`

## 8. SchedulerOutput 结构

定义文件：

- `code/vllm/vllm/v1/core/sched/output.py`

### 8.1 NewRequestData

新请求首次调度时发送完整数据：

- `req_id`
- `prompt_token_ids`
- `mm_features`
- `sampling_params`
- `pooling_params`
- `block_ids`
- `num_computed_tokens`
- `lora_request`
- `prompt_embeds`
- `prefill_token_ids`

源码：

- `code/vllm/vllm/v1/core/sched/output.py:30`
- `code/vllm/vllm/v1/core/sched/output.py:65`

### 8.2 CachedRequestData

已缓存请求只发送增量：

- `req_ids`
- `resumed_req_ids`
- `new_token_ids`
- `all_token_ids`
- `new_block_ids`
- `num_computed_tokens`
- `num_output_tokens`

源码：

- `code/vllm/vllm/v1/core/sched/output.py:111`
- `code/vllm/vllm/v1/core/sched/output.py:177`

### 8.3 SchedulerOutput

关键字段：

- `scheduled_new_reqs`
- `scheduled_cached_reqs`
- `num_scheduled_tokens`
- `total_num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `scheduled_encoder_inputs`
- `num_common_prefix_blocks`
- `finished_req_ids`
- `free_encoder_mm_hashes`
- `preempted_req_ids`
- `kv_connector_metadata`
- `ec_connector_metadata`
- `new_block_ids_to_zero`

源码：

- `code/vllm/vllm/v1/core/sched/output.py:180`
- `code/vllm/vllm/v1/core/sched/output.py:245`

## 9. Scheduler.schedule 构造 SchedulerOutput

入口：

- `code/vllm/vllm/v1/core/sched/scheduler.py:387`

主要流程：

1. 初始化本 step 临时状态。
2. 调度 RUNNING 请求。
3. 分配 KV blocks。
4. 可能 preempt 请求。
5. 调度 WAITING 请求。
6. 处理 prefix cache / KV connector cache 命中。
7. 构造 `NewRequestData`。
8. 构造 `CachedRequestData`。
9. 构造 `SchedulerOutput`。
10. 构造 KV / EC connector metadata。
11. 更新 scheduler 内部状态。

源码：

- 初始化：`code/vllm/vllm/v1/core/sched/scheduler.py:400`
- RUNNING：`code/vllm/vllm/v1/core/sched/scheduler.py:429`
- KV blocks：`code/vllm/vllm/v1/core/sched/scheduler.py:521`
- preempt：`code/vllm/vllm/v1/core/sched/scheduler.py:534`
- WAITING：`code/vllm/vllm/v1/core/sched/scheduler.py:624`
- prefix/KV connector：`code/vllm/vllm/v1/core/sched/scheduler.py:671`
- NewRequestData：`code/vllm/vllm/v1/core/sched/scheduler.py:1011`
- CachedRequestData：`code/vllm/vllm/v1/core/sched/scheduler.py:1031`
- SchedulerOutput：`code/vllm/vllm/v1/core/sched/scheduler.py:1057`
- KV metadata：`code/vllm/vllm/v1/core/sched/scheduler.py:1080`
- EC metadata：`code/vllm/vllm/v1/core/sched/scheduler.py:1085`
- update after schedule：`code/vllm/vllm/v1/core/sched/scheduler.py:1096`

## 10. Executor 分发 SchedulerOutput

抽象层：

```text
Executor.execute_model(scheduler_output)
  -> collective_rpc("execute_model", args=(scheduler_output,))
```

源码：

- `code/vllm/vllm/v1/executor/abstract.py:221`

不同后端：

- UniProc：直接调用 driver worker。
- Multiproc：broadcast MQ fan-out，output rank 回包。
- Ray old：Ray compiled DAG。
- Ray v2：Ray actor + MQ。

## 11. Worker 调用 ModelRunner

Worker wrapper：

```text
WorkerWrapperBase.execute_model
  -> _apply_mm_cache
  -> worker.execute_model
```

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:340`
- `code/vllm/vllm/v1/worker/worker_base.py:345`

GPU worker：

```text
Worker.execute_model
  -> PP recv intermediate tensors
  -> model_runner.execute_model(scheduler_output, intermediate_tensors)
  -> PP send intermediate tensors if needed
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `code/vllm/vllm/v1/worker/gpu_worker.py:853`
- `code/vllm/vllm/v1/worker/gpu_worker.py:867`
- `code/vllm/vllm/v1/worker/gpu_worker.py:882`

## 12. ModelRunner 返回 ModelRunnerOutput

V1 ModelRunner：

- execute：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`
- sample：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`

V2 ModelRunner：

- execute：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`
- sample：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`

`ModelRunnerOutput` 定义：

- `code/vllm/vllm/v1/outputs.py:233`

## 13. Scheduler.update_from_output

入口：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1463`

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

源码：

- 解包：`code/vllm/vllm/v1/core/sched/scheduler.py:1468`
- KV load fail：`code/vllm/vllm/v1/core/sched/scheduler.py:1490`
- routed experts：`code/vllm/vllm/v1/core/sched/scheduler.py:1500`
- 遍历请求：`code/vllm/vllm/v1/core/sched/scheduler.py:1521`
- 找输出：`code/vllm/vllm/v1/core/sched/scheduler.py:1542`
- spec decode：`code/vllm/vllm/v1/core/sched/scheduler.py:1547`
- 更新 request：`code/vllm/vllm/v1/core/sched/scheduler.py:1588`
- grammar：`code/vllm/vllm/v1/core/sched/scheduler.py:1598`
- stopped：`code/vllm/vllm/v1/core/sched/scheduler.py:1655`
- EngineCoreOutput：`code/vllm/vllm/v1/core/sched/scheduler.py:1688`
- finished queues：`code/vllm/vllm/v1/core/sched/scheduler.py:1710`
- KV connector finished：`code/vllm/vllm/v1/core/sched/scheduler.py:1731`
- EngineCoreOutputs：`code/vllm/vllm/v1/core/sched/scheduler.py:1769`
- finished ids：`code/vllm/vllm/v1/core/sched/scheduler.py:1776`
- stats：`code/vllm/vllm/v1/core/sched/scheduler.py:1790`
- return：`code/vllm/vllm/v1/core/sched/scheduler.py:1802`

## 14. in-process 完整链路

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

## 15. multiproc 完整链路

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

## 16. Ray compiled DAG 完整链路

```text
RayDistributedExecutor.execute_model
  -> 可能缓存 scheduler_output
  -> sample_tokens(grammar_output)
  -> _execute_dag(scheduler_output, grammar_output)
  -> Ray DAG fan-out to first PP stage TP workers
  -> execute_model_ray
  -> worker.model_runner.execute_model
  -> IntermediateTensors pass to next PP stage
  -> last PP stage sample_tokens
  -> output rank returns ModelRunnerOutput
  -> Scheduler.update_from_output
```

## 17. 最短总链路

```text
External request
  -> EngineCoreRequest
  -> Request
  -> Scheduler
  -> SchedulerOutput
  -> Executor.execute_model
  -> Worker.execute_model
  -> ModelRunner.execute_model
  -> Executor.sample_tokens
  -> Worker.sample_tokens
  -> ModelRunner.sample_tokens
  -> ModelRunnerOutput
  -> Scheduler.update_from_output
  -> EngineCoreOutputs
  -> RequestOutput
```

## 18. 关键理解

1. `SchedulerOutput` 是 executor/worker 的核心输入。
2. `GrammarOutput` 在采样阶段传递结构化输出约束。
3. `execute_model()` 返回 `None` 是 generation 主路径的一部分。
4. `ModelRunnerOutput` 不是直接返回给用户，而是先交给 scheduler 更新请求状态。
5. `EngineCoreOutputs` 才是 EngineCore 返回给前端的输出结构。
