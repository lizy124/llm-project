# 04 engine_core 背诵文档

## 1. 专题定位

`engine_core` 讲的是 vLLM V1 内部执行核心。

它位于外层 Engine 和执行层 Worker 之间，是内部主循环的总控。

一句话：

```text
EngineCore 是 vLLM V1 的内部执行闭环总控，负责把请求交给 Scheduler，把调度计划交给 Executor，把 Worker 结果交回 Scheduler，并把 EngineCoreOutputs 返回外层 Engine。
```

## 2. 最小心智模型

最小链路是：

```text
EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → Request
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

要背住：

```text
EngineCore 不做调度细节，不做模型 forward，不做 detokenize；它编排 schedule、execute、update 三步闭环。
```

## 3. EngineCore 负责什么

EngineCore 负责：

```text
1. 接收外层已经处理好的 EngineCoreRequest。
2. 转换成 Scheduler 使用的 Request。
3. 把 Request 加入 Scheduler。
4. 创建并持有 model_executor。
5. 初始化 KV cache。
6. 创建 StructuredOutputManager。
7. 创建 Scheduler。
8. 每轮调用 Scheduler.schedule()。
9. 调用 model_executor.execute_model()。
10. 必要时调用 model_executor.sample_tokens()。
11. 调用 Scheduler.update_from_output()。
12. 返回 EngineCoreOutputs。
13. 转发 profile、reset、sleep、wake_up、LoRA 等内部控制能力。
```

## 4. EngineCore 不负责什么

EngineCore 不负责：

```text
用户原始 prompt 处理
detokenize
最终 RequestOutput 构造
token budget 具体决策
waiting / running 队列策略
KV block 抢占和释放细节
模型输入张量准备
模型 forward
采样算法本身
```

这些分别属于：

```text
InputProcessor
OutputProcessor
Scheduler
KVCacheManager
Executor / Worker / ModelRunner
Sampler
```

## 5. EngineCore.step 的核心

`EngineCore.step()` 是最重要的方法。

它可以背成：

```text
step()
  → 如果 Scheduler 没请求，返回空输出
  → scheduler.schedule()
  → model_executor.execute_model(scheduler_output, non_block=True)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → 等 Worker 返回 model_output
  → 如果 model_output is None，调用 sample_tokens(grammar_output)
  → 处理 abort queue
  → scheduler.update_from_output(scheduler_output, model_output)
  → 返回 EngineCoreOutputs
```

压缩成三步：

```text
Schedule：生成 SchedulerOutput。
Execute：交给 model_executor / Worker / ModelRunner 执行。
Make output：Scheduler.update_from_output 生成 EngineCoreOutputs。
```

## 6. 三个关键输出对象

### SchedulerOutput

```text
Scheduler 生成的本轮执行计划。
```

它告诉 Worker：

```text
本轮跑哪些请求
每个请求跑多少 token
哪些是新请求
哪些是 cached request
哪些 block id 可用
哪些 encoder input 要跑
哪些 spec tokens 要验证
哪些 connector metadata 要执行
```

### ModelRunnerOutput

```text
Worker / ModelRunner 返回的执行结果。
```

它包含：

```text
sampled_token_ids
logprobs
prompt_logprobs
pooler_output
kv_connector_output
ec_connector_output
cudagraph_stats
routed_experts
```

### EngineCoreOutputs

```text
Scheduler 消化 ModelRunnerOutput 后，返回给外层 Engine 的内部输出协议。
```

它还不是最终用户输出。

用户输出还要经过 OutputProcessor。

## 7. 请求如何进入 EngineCore

外层 Engine 先通过 InputProcessor 得到 `EngineCoreRequest`。

EngineCore 再转换：

```text
EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → 如果 structured output，grammar_init(req)
  → EngineCore.add_request()
  → Scheduler.add_request()
```

要区分：

```text
EngineCoreRequest：外层 Engine 交给 EngineCore 的请求对象。
Request：EngineCore 转换后交给 Scheduler 的内部请求对象。
```

## 8. EngineCore 和 Scheduler 的关系

Scheduler 是 EngineCore 内部的调度和状态管理组件。

EngineCore 调用 Scheduler：

```text
scheduler.schedule()
scheduler.get_grammar_bitmask()
scheduler.update_from_output()
scheduler.add_request()
scheduler.finish_requests()
scheduler.reset_prefix_cache()
```

Scheduler 负责具体判断：

```text
每个请求本轮调度多少 token
KV block 是否够
是否需要抢占
prefix cache 命中多少
spec token 接受 / 拒绝后如何回退
请求是否结束
什么时候释放 block
```

一句话：

```text
EngineCore 调用 Scheduler；Scheduler 决定调度和状态账本。
```

## 9. SchedulerOutput 在 EngineCore 中如何流动

`SchedulerOutput` 有双重作用：

```text
向下：告诉 Worker / ModelRunner 本轮怎么执行。
向上：作为 update_from_output 的对账凭证。
```

EngineCore 拿到它后做三件事：

```text
1. execute_model(scheduler_output)
2. get_grammar_bitmask(scheduler_output)
3. update_from_output(scheduler_output, model_output)
```

所以它不是发给 Worker 后就结束。

它还要和 `ModelRunnerOutput` 一起回到 Scheduler 做对账。

## 10. EngineCore 和 Executor / Worker / ModelRunner 的关系

EngineCore 不直接 forward。

它只调用：

```text
model_executor.execute_model(scheduler_output)
```

执行链路：

```text
EngineCore
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → logits / pooling / sample_tokens
  → ModelRunnerOutput
```

一句话：

```text
EngineCore 只把计划交出去；Worker / ModelRunner 才真正把计划跑进模型。
```

## 11. 为什么 execute_model 可能返回 None

对于 generation 模型，ModelRunner 可能把 forward 和 sampling 拆开。

链路是：

```text
execute_model()
  → forward
  → compute_logits
  → 缓存 ExecuteModelState
  → return None

EngineCore 看到 None
  → model_executor.sample_tokens(grammar_output)
  → ModelRunnerOutput
```

这样做是为了让 EngineCore 在 forward 和 sampling 之间插入 grammar bitmask。

要背住：

```text
execute_model 可能只完成 forward / logits；sample_tokens 才生成 ModelRunnerOutput。
```

## 12. ModelRunnerOutput 如何回到 Scheduler

EngineCore 不自己解释 ModelRunnerOutput。

它直接调用：

```text
scheduler.update_from_output(scheduler_output, model_output)
```

为什么要同时传两个对象：

```text
SchedulerOutput：本轮计划账本。
ModelRunnerOutput：真实执行结果。
```

Scheduler 会对齐：

```text
哪些请求本轮被调度
每个请求调度多少 token
每个请求实际输出哪些 token
spec decode 接受 / 拒绝多少
logprobs 属于哪个请求
pooling output 属于哪个请求
哪些资源要释放
```

一句话：

```text
schedule 是发任务，execute_model 是执行任务，update_from_output 是收结果并对账。
```

## 13. EngineCoreOutputs 如何返回上层

输出层级是：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

`EngineCoreOutput` 是单请求增量输出。

包含：

```text
request_id
new_token_ids
new_logprobs
new_prompt_logprobs_tensors
pooling_output
finish_reason
stop_reason
kv_transfer_params
ec_transfer_params
```

`EngineCoreOutputs` 是一组输出加统计信息。

包含：

```text
outputs
scheduler_stats
timestamp
utility_output
finished_requests
engine_index
```

## 14. 初始化顺序

EngineCore 初始化时大致是：

```text
EngineCore.__init__()
  → self.model_executor = executor_class(vllm_config)
  → 初始化 / profile KV cache
  → 得到 kv_cache_config
  → 创建 StructuredOutputManager
  → 创建 Scheduler
  → 设置 step_fn
```

重要顺序：

```text
先初始化 KV cache，再创建 Scheduler。
```

因为 Scheduler 需要：

```text
kv_cache_config
block size
KV cache capacity
```

## 15. KV cache 初始化在 EngineCore 中的地位

EngineCore 创建 model_executor 后，会初始化 KV cache。

大致动作：

```text
获取 KV cache specs
处理 non-causal cache 限制
profile 可用 GPU memory
生成 worker KV cache config
生成 scheduler KV cache config
更新 cache_config
调用 model_executor.initialize_from_config()
```

注意：

```text
EngineCore 不直接分配每个 Worker 的 KV tensor；它通过 model_executor / Worker 完成。
```

## 16. 同步、异步、多进程模式

外层 Engine 通过 EngineCoreClient 连接 EngineCore。

### InprocClient

```text
前端和 EngineCore 同进程
get_output() 直接调用 EngineCore.step_fn()
```

### EngineCoreProc

多进程模式下后台运行：

```text
EngineCoreProc
  → input thread 收 ZMQ 请求
  → input_queue
  → run_busy_loop()
  → _process_input_queue()
  → _process_engine_step()
  → output_queue
  → output thread 发回前端
```

后台 loop 可以背成：

```text
while running:
  process input queue
  process engine step
```

## 17. batch queue 和异步调度

普通 step 是：

```text
schedule → execute → wait result → update_from_output
```

如果启用 batch queue：

```text
先 schedule 新 batch 入队
不一定马上等输出
等最早的 future 完成后再 update_from_output
```

目的：

```text
让调度和模型执行 overlap，尤其减少 pipeline parallel 的 bubble。
```

关键区别：

```text
普通 step：SchedulerOutput 和 ModelRunnerOutput 通常在同一次 step 配对。
batch queue：SchedulerOutput 和 future 暂存在队列中，稍后再配对。
```

## 18. abort 队列

EngineCore 有 abort queue。

在 Worker 输出回来后、Scheduler update 前处理：

```text
model_output = future.result()
_process_aborts_queue()
scheduler.update_from_output(...)
```

这样可以保证：

```text
执行期间到达的取消请求，在 Scheduler 回收输出前生效。
```

## 19. utility / profile / sleep / wake_up

EngineCore 还处理控制类请求。

常见类型：

```text
ADD：添加请求
ABORT：取消请求
UTILITY：profile / reset / sleep / LoRA 等
EXECUTOR_FAILED：executor 异常
WAKEUP：唤醒 loop
```

utility 输出通过：

```text
EngineCoreOutputs.utility_output
```

返回，而不是普通请求输出列表。

## 20. sleep / wake_up 的边界

`sleep()` 通常先暂停调度，再让 executor / Worker 管理 GPU 内存。

`wake_up()` 唤醒 executor 后，只有 executor 不再 sleeping，才恢复 Scheduler。

要背住：

```text
sleep / wake_up 同时影响 Scheduler 状态和 Worker 设备状态。
```

## 21. shutdown

EngineCore shutdown 会释放：

```text
structured output backend
model_executor / Worker
Scheduler
GC freeze 状态
distributed environment / cached memory
```

多进程 EngineCoreProc 还要处理：

```text
RUNNING
REQUESTED
SHUTTING_DOWN
```

## 22. 容易混淆的点

### EngineCore 是不是 Scheduler

不是。

```text
EngineCore 编排闭环。
Scheduler 管调度策略和请求状态。
```

### EngineCore 是不是 Worker

不是。

```text
EngineCore 调用 model_executor。
Worker / ModelRunner 准备输入并执行模型。
```

### EngineCoreOutputs 是不是最终用户输出

不是。

```text
EngineCoreOutputs 是内部输出协议。
RequestOutput 才是用户可见输出。
```

### EngineCore.step 是不是一定执行模型

不一定。

```text
没有请求时返回空。
0-token connector step 也可能只推进 KV transfer。
batch queue 下当前 step 可能只回收之前 batch。
```

## 23. 与其他专题的关系

```text
engine：解释外层 LLMEngine / AsyncLLM 如何调用 EngineCore。
scheduler：解释 schedule / update_from_output 的内部状态机。
executor_worker_model_runner：解释 model_executor.execute_model 后发生什么。
sampling_and_output：解释 ModelRunnerOutput 到 RequestOutput 的后半段。
kv_cache_transfer：解释 KV connector metadata 和 output 如何穿过 EngineCore。
spec_decode：解释 draft tokens 的 post_step 回写闭环。
compilation_and_cuda_graph：解释 Worker forward 内部的性能优化路径。
```

## 24. 背诵总结

背这一段：

```text
EngineCore 是 vLLM V1 的内部执行闭环总控。外层 Engine 把 EngineCoreRequest 交给它，它先转换成 Scheduler 的 Request，再由 Scheduler 管理。每轮 EngineCore.step 调 Scheduler.schedule 生成 SchedulerOutput，把计划交给 model_executor / Worker / ModelRunner 执行，必要时再调用 sample_tokens，然后把 SchedulerOutput 和 ModelRunnerOutput 一起交给 Scheduler.update_from_output，对账请求状态、tokens、KV、spec decode 和资源释放，最终返回 EngineCoreOutputs 给外层 OutputProcessor。EngineCore 的核心是编排 schedule、execute、update，不负责具体调度策略、模型 forward 或用户输出格式。
```
