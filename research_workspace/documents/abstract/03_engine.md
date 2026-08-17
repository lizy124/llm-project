# 03 engine 背诵文档

## 1. 专题定位

`engine` 讲的是 vLLM V1 的外层引擎体系。

它关注用户请求如何进入 vLLM，内部输出如何变成用户可见输出。

它不直接讲模型 forward，也不直接讲 token 调度。

一句话：

```text
Engine 是 vLLM 对外提供推理能力的外层体系，负责输入处理、输出处理、同步/异步编排，并通过 EngineCoreClient 驱动内部 EngineCore。
```

## 2. 最小心智模型

最小链路是：

```text
用户 / API server / LLM
  → LLMEngine 或 AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore
  → Scheduler / Worker / ModelRunner
  → EngineCoreOutputs
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

要背住：

```text
Engine 管输入输出，EngineCore 管内部执行闭环，Scheduler 管调度账本，Worker / ModelRunner 管模型计算。
```

## 3. Engine 和 EngineCore 的边界

Engine 不是 EngineCore。

```text
Engine：
  负责用户接口、输入处理、输出处理、同步/异步编排。

EngineCore：
  负责内部 schedule → execute → update → output 执行闭环。
```

一句话：

```text
LLMEngine / AsyncLLM 是外层 Engine 的具体形态；EngineCore 是它们背后的内部执行核心；EngineCoreClient 是桥。
```

## 4. Engine 总体主链路

完整请求输出链路：

```text
用户输入
  → LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → Worker / ModelRunner forward / sample
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → RequestOutput
```

这条链路分三段：

```text
输入段：用户输入 → InputProcessor → EngineCoreRequest → Request → Scheduler
执行段：Scheduler → Worker / ModelRunner → Scheduler.update_from_output
输出段：EngineCoreOutputs → OutputProcessor → RequestOutput
```

## 5. LLMEngine：同步外层 Engine

`LLMEngine` 是同步路径里的外层 Engine。

它主要提供：

```text
add_request()
step()
has_unfinished_requests()
abort_request()
profile / reset / sleep / wake_up / LoRA 控制接口
```

初始化时创建三大组件：

```text
InputProcessor：EngineInput → EngineCoreRequest
OutputProcessor：EngineCoreOutputs → RequestOutput
EngineCoreClient：连接外层 Engine 和内部 EngineCore
```

注意：

```text
LLMEngine.engine_core 字段名虽然叫 engine_core，但通常实际是 EngineCoreClient。
```

## 6. LLMEngine 请求入口

同步请求进入：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
```

如果 `SamplingParams.n > 1`：

```text
一个外部请求
  → ParentRequest
  → 多个 child request
  → 每个 child 进入 EngineCore
  → OutputProcessor 聚合回一个外部 RequestOutput
```

## 7. LLMEngine 输出入口

同步输出通过 `step()` 拉取：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
  → return
```

如果 OutputProcessor 检测到 stop string，但 EngineCore 还没结束请求：

```text
OutputProcessor
  → reqs_to_abort
  → LLMEngine 通知 EngineCore abort
```

## 8. AsyncLLM：异步外层 Engine

`AsyncLLM` 是异步路径里的外层 Engine。

它和 LLMEngine 的边界相同：

```text
InputProcessor：输入转换
EngineCoreClient：访问 EngineCore
OutputProcessor：输出转换
```

区别是：

```text
LLMEngine：调用方主动 step() 拉输出。
AsyncLLM：后台 output_handler 持续拉输出，调用方通过 async generator 消费结果。
```

## 9. AsyncLLM 请求入口

异步请求进入：

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()
  → EngineCoreProc
  → Scheduler.add_request()
```

用户消费的是：

```text
AsyncLLM.generate() 返回的 async generator
```

它不是直接消费 EngineCoreOutputs。

它消费 OutputProcessor 已经处理好的 RequestOutput。

## 10. AsyncLLM 输出入口

异步输出路径：

```text
EngineCoreProc
  → EngineCoreOutputs
  → AsyncMPClient.get_output_async()
  → AsyncLLM.output_handler
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put()
  → AsyncLLM.generate() / encode()
  → yield RequestOutput / PoolingRequestOutput
```

重要点：

```text
异步模式下 OutputProcessor 通常把结果推入 queue，而不是直接返回 request_outputs。
```

## 11. InputProcessor 的职责

`InputProcessor` 属于外层 Engine。

它负责把用户输入变成 `EngineCoreRequest`。

它会做：

```text
校验 SamplingParams / PoolingParams
校验 LoRA
校验 data_parallel_rank
预处理 raw prompt
拆分 encoder / decoder input
提取 prompt_token_ids / prompt_embeds
处理多模态 mm_features
构造 EngineCoreRequest
```

一句话：

```text
InputProcessor 负责把用户请求标准化成 EngineCore 能理解的内部请求。
```

## 12. request id 的两层含义

InputProcessor 会处理 request id。

通常有两层：

```text
external_req_id：
  用户传入的 request_id，最终对外展示。

request_id：
  vLLM 内部 request_id，用于 EngineCore / Scheduler / OutputProcessor 状态对齐。
```

默认内部 id 会追加随机后缀。

这样可以避免用户传入重复 id 导致内部冲突。

## 13. OutputProcessor 的职责

`OutputProcessor` 也属于外层 Engine。

它负责把内部输出协议变成用户输出协议。

输入：

```text
EngineCoreOutput / EngineCoreOutputs
```

输出：

```text
RequestOutput
CompletionOutput
PoolingRequestOutput
streaming queue item
```

它会做：

```text
维护 frontend RequestState
detokenize token ids
检查 stop string
处理 logprobs / prompt logprobs
构造 CompletionOutput
处理 DELTA / CUMULATIVE / FINAL_ONLY 输出模式
finished 后清理 RequestState
```

一句话：

```text
OutputProcessor 负责把 EngineCore 的 token-id 增量输出翻译成用户可见输出。
```

## 14. OutputProcessor.add_request 为什么在 EngineCore 前调用

请求进入 EngineCore 前，外层 Engine 会先调用：

```text
OutputProcessor.add_request()
```

原因：

```text
EngineCoreOutput 只带内部增量信息。
OutputProcessor 必须提前建立 RequestState，才能在输出回来时知道如何 detokenize、聚合、stream、格式化。
```

因此顺序是：

```text
OutputProcessor.add_request()
  → EngineCoreClient.add_request()
```

而不是反过来。

## 15. EngineCoreClient 的作用

外层 Engine 通常不直接持有 EngineCore。

它持有 `EngineCoreClient`。

常见实现：

```text
InprocClient：同进程 EngineCore。
SyncMPClient：同步前端 + 后台 EngineCoreProc + ZMQ。
AsyncMPClient：异步前端 + 后台 EngineCoreProc + asyncio/ZMQ。
DPAsyncMPClient / DPLBAsyncMPClient：data parallel 场景。
```

一句话：

```text
EngineCoreClient 屏蔽了 EngineCore 是同进程、后台进程、异步进程还是 DP 多副本的差异。
```

## 16. Inproc 和多进程差异

### InprocClient

```text
LLMEngine
  → InprocClient
  → EngineCore 同进程对象
```

特点：

```text
add_request 直接调用 EngineCore.preprocess_add_request()
get_output 直接调用 EngineCore.step_fn()
```

### SyncMPClient / AsyncMPClient

```text
Frontend Engine
  → ZMQ input socket
  → EngineCoreProc 后台进程
  → EngineCore busy loop
  → ZMQ output socket
  → Frontend Engine
```

特点：

```text
EngineCore 在后台进程运行，前端通过 socket 发送 ADD / ABORT / UTILITY，接收 EngineCoreOutputs。
```

## 17. Engine 负责什么

Engine 负责：

```text
1. 对外提供同步 / 异步接口。
2. 接收用户 prompt、params、request_id。
3. 调用 InputProcessor 处理输入。
4. 调用 OutputProcessor 登记输出状态。
5. 通过 EngineCoreClient 访问 EngineCore。
6. 从 EngineCoreClient 拉取 EngineCoreOutputs。
7. 调用 OutputProcessor 构造 RequestOutput。
8. 转发 abort、profile、sleep、wake_up、LoRA 等控制 API。
9. 管理 output_handler、stats、tracing 等外层生命周期。
```

## 18. Engine 不负责什么

Engine 不负责：

```text
token 级调度
waiting / running 队列管理
KV block 分配和释放
prefix cache 命中
preemption
Scheduler.update_from_output 状态对账
Worker / ModelRunner forward
采样实际执行
KV / encoder 物理缓存操作
```

这些属于：

```text
EngineCore
Scheduler
model_executor
Worker / ModelRunner
```

## 19. EngineCore 不负责什么

EngineCore 也不负责外层用户协议：

```text
用户原始 prompt 预处理
detokenize
stop string 文本级检查
最终 RequestOutput 构造
AsyncLLM streaming queue 分发
OpenAI API response 包装
```

这些属于：

```text
LLMEngine / AsyncLLM
InputProcessor
OutputProcessor
entrypoints
```

## 20. abort / stop / 控制接口

用户主动 abort：

```text
LLMEngine.abort_request() / AsyncLLM.abort()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

文本 stop string：

```text
OutputProcessor 检测 stop string
  → reqs_to_abort
  → Engine 通知 EngineCore abort
```

控制类 API：

```text
profile
reset_mm_cache
reset_prefix_cache
reset_encoder_cache
pause_generation / resume_generation
sleep / wake_up
add_lora / remove_lora / list_loras / pin_lora
collective_rpc / apply_model
```

这些一般由 Engine 转发到 EngineCore，再转发到 Scheduler / Executor / Worker。

## 21. 容易混淆的点

### Engine 不是 EngineCore

```text
Engine 是外层接口和输入输出处理。
EngineCore 是内部执行闭环。
```

### OutputProcessor 不等于 Scheduler

```text
Scheduler 管 token-id 状态机和资源账本。
OutputProcessor 管 detokenize、stop string、RequestOutput。
```

### AsyncLLM.generate 不直接读 EngineCore

```text
generate 读的是 per-request queue。
queue 里的结果来自 output_handler 和 OutputProcessor。
```

### EngineCoreOutputs 不是用户输出

```text
EngineCoreOutputs 是内部输出协议。
RequestOutput 才是用户可见输出。
```

## 22. 与其他专题的关系

```text
engine_core：解释 EngineCore 内部 schedule / execute / update 闭环。
scheduler：解释请求状态和 token / KV block 调度。
executor_worker_model_runner：解释 SchedulerOutput 如何进入模型执行。
sampling_and_output：解释 ModelRunnerOutput 到 RequestOutput 的后半段。
multimodal：InputProcessor 如何处理 mm_features。
lora_and_adapters：InputProcessor 如何校验 LoRARequest，Engine 如何转发 LoRA 控制接口。
```

## 23. 背诵总结

背这一段：

```text
vLLM V1 的 Engine 是外层推理引擎体系，具体形态包括同步 LLMEngine 和异步 AsyncLLM。它们不直接做调度和 forward，而是通过 InputProcessor 把用户输入变成 EngineCoreRequest，通过 OutputProcessor 把 EngineCoreOutputs 变成 RequestOutput，并通过 EngineCoreClient 驱动内部 EngineCore。LLMEngine 由调用方 step 拉输出；AsyncLLM 由后台 output_handler 拉输出并推入 per-request queue。Engine 的边界是管输入输出和外层 API，EngineCore / Scheduler / Worker 才负责内部执行、调度和模型计算。
```
