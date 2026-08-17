# 01. EngineCore 在 vLLM 里负责什么？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：EngineCore 在 vLLM V1 架构中的定位，以及它和 Engine、Scheduler、ModelRunner / Worker 的关系。

---

## 1. 一句话回答

`EngineCore` 是 vLLM V1 Engine 内部的执行核心。

源码里对它的定位非常直接：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96` 到 `vllm/vllm/v1/engine/core.py:97`

可以理解为：

```text
EngineCore 是 Engine 的内部主循环。
```

它不负责把用户输入转成请求，也不负责把 token detokenize 成最终用户输出；这些更靠外层的工作由 `LLMEngine` / `AsyncLLM` 的 `InputProcessor`、`OutputProcessor` 处理。

它也不直接实现 token 级调度细节；调度细节由 `Scheduler` 负责。

它也不直接实现模型 forward；模型执行由 `model_executor`、Worker、ModelRunner 负责。

`EngineCore` 主要负责把这些组件串起来：

```text
EngineCore
  → 接收已经处理好的 EngineCoreRequest
  → 转成内部 Request
  → 交给 Scheduler
  → 调用 Scheduler.schedule() 生成 SchedulerOutput
  → 把 SchedulerOutput 交给 model_executor / Worker 执行
  → 拿到 ModelRunnerOutput
  → 调用 Scheduler.update_from_output()
  → 得到 EngineCoreOutputs
  → 返回给外层 Engine / AsyncLLM
```

一句话：

```text
EngineCore 是 vLLM V1 的内部执行闭环总控；
Scheduler 负责决定怎么调度，Worker / ModelRunner 负责真正 forward，
EngineCore 负责把调度、执行、回收结果串成一轮一轮的 step。
```

---

## 2. Engine 和 EngineCore 的关系

这里的 `Engine` 可以理解为更靠外层的引擎对象，例如同步接口里的 `LLMEngine` 和异步接口里的 `AsyncLLM`。

### 2.1 `LLMEngine` 创建并持有 EngineCoreClient

同步 `LLMEngine` 初始化时，会创建 `InputProcessor`、`OutputProcessor`，然后创建 `EngineCoreClient`：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)

# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:93` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

这里的注释已经说明了边界：

```text
EngineCore 接收 EngineCoreRequests，返回 EngineCoreOutputs。
```

也就是说：

```text
LLMEngine 负责外层输入输出处理；
EngineCore 负责内部执行。
```

### 2.2 同步请求入口：LLMEngine.add_request()

同步 `LLMEngine.add_request()` 的主线是：

```text
用户输入 prompt / params
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
```

关键代码：

```python
request = self.input_processor.process_inputs(...)
...
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250`、`vllm/vllm/v1/engine/llm_engine.py:274` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

这说明外层 `LLMEngine` 不直接把请求塞进 Scheduler，而是通过 `EngineCoreClient` 交给 EngineCore。

### 2.3 同步 step：LLMEngine 从 EngineCore 拉输出

同步 `LLMEngine.step()` 中，第一步就是从 EngineCore 拉取输出：

```python
outputs = self.engine_core.get_output()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:304`

然后外层再把 `EngineCoreOutputs` 转成用户可见的 `RequestOutput`：

```python
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:309` 到 `vllm/vllm/v1/engine/llm_engine.py:312`

所以同步路径可以理解为：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCore.step_fn()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

### 2.4 异步 AsyncLLM 也是同样边界

异步 `AsyncLLM` 初始化时也有同样结构：

```python
self.input_processor = InputProcessor(self.vllm_config, renderer)
self.output_processor = OutputProcessor(...)
self.engine_core = EngineCoreClient.make_async_mp_client(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:134` 到 `vllm/vllm/v1/engine/async_llm.py:153`

异步请求进入时：

```python
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:412`

异步输出处理后台任务中，会不断从 EngineCore 拉输出：

```python
outputs = await engine_core.get_output_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:660`

然后再走 `OutputProcessor.process_outputs()`：

```python
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:675` 到 `vllm/vllm/v1/engine/async_llm.py:677`

因此，不管同步还是异步，层级都是：

```text
LLMEngine / AsyncLLM
  → InputProcessor：把外部输入转成 EngineCoreRequest
  → EngineCoreClient：把请求送进 EngineCore
  → EngineCore：内部执行闭环
  → EngineCoreOutputs
  → OutputProcessor：转成上层 RequestOutput
```

---

## 3. EngineCore 和 EngineCoreClient 的关系

外层 Engine 通常不是直接持有 `EngineCore`，而是持有 `EngineCoreClient`。

`EngineCoreClient` 是访问 EngineCore 的客户端抽象：

```python
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

也就是说，EngineCore 可能有两种运行方式：

```text
in-process：
  EngineCore 和外层 LLMEngine 在同一进程；
  InprocClient 直接调用 EngineCore 方法。

multi-process：
  EngineCore 在后台进程 busy loop 中运行；
  SyncMPClient / AsyncMPClient 通过 ZMQ 发送请求、接收输出。
```

### 3.1 InprocClient：直接调用 EngineCore

in-process 模式下，`InprocClient` 直接创建 EngineCore：

```python
self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/vllm/v1/engine/core_client.py:286` 到 `vllm/vllm/v1/engine/core_client.py:287`

拉输出时直接调用：

```python
outputs, model_executed = self.engine_core.step_fn()
self.engine_core.post_step(model_executed=model_executed)
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:291`

这里的两个返回值来自 `EngineCore.step()` 或 `EngineCore.step_with_batch_queue()`。

普通 `step()` 的返回类型是：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
```

位置：`vllm/vllm/v1/engine/core.py:488`

所以：

```text
outputs：
  dict[int, EngineCoreOutputs]

  key 是 client_index；
  value 是这个 client 本轮要收到的 EngineCoreOutputs。

  在 in-process 模式下，通常只有 client_index = 0，
  所以 InprocClient 最后会返回 outputs.get(0)。

model_executed：
  bool

  表示这一轮 step 是否真的执行了模型 token 计算。
  普通 step 中它来自：

    scheduler_output.total_num_scheduled_tokens > 0

  如果本轮没有任何请求，step 返回 ({}, False)。
  如果本轮只有清理 / stats / connector 等工作，但没有实际 token，
  model_executed 也可能是 False。
```

普通 `step()` 结尾是：

```python
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:517`

`InprocClient.get_output()` 会把 dict 里的 client 0 输出取出来：

```python
return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:292`

所以 in-process 拉输出可以理解为：

```text
EngineCore.step_fn()
  → dict[client_index, EngineCoreOutputs], model_executed
  → post_step(model_executed)
  → 取 client 0 的 EngineCoreOutputs
  → 返回给 LLMEngine.step()
```

`post_step(model_executed=...)` 是 step 后置钩子，主要用于 speculative decoding / diffusion 场景下同步 draft token ids。

源码是：

```python
def post_step(self, model_executed: bool) -> None:
    # When using async scheduling we can't get draft token ids in advance,
    # so we update draft token ids in the worker process and don't
    # need to update draft token ids here.
    if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`vllm/vllm/v1/engine/core.py:519` 到 `vllm/vllm/v1/engine/core.py:526`

它做的事情是：

```text
如果当前模型需要检查 draft tokens，
并且不是 async scheduling，
并且本轮确实执行了模型，
那么从 model_executor / Worker 取出本轮产生的 draft_token_ids，
再交给 Scheduler.update_draft_token_ids() 写回对应 Request。
```

这里的 draft token ids 主要用于下一轮 speculative decoding。

也就是说：

```text
step()：
  完成本轮 schedule → execute → update → output。

post_step()：
  在本轮结束后，把 Worker 侧新产生的 draft tokens 同步回 Scheduler，
  供下一轮 schedule() 使用。
```

为什么要用 `model_executed` 判断？

```text
如果本轮没有实际模型执行，就不可能产生新的 draft tokens；
这时不需要调用 take_draft_token_ids()。
```

为什么 async scheduling 不在这里处理？

```text
源码注释说明：async scheduling 时无法提前拿 draft token ids，
所以 draft token ids 会在 Worker 进程中更新，
不需要在 EngineCore.post_step() 里再更新。
```

添加请求时：

```python
req, request_wave = self.engine_core.preprocess_add_request(request)
self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

所以 in-process 模式中：

```text
EngineCoreClient 只是一个薄封装；
真正执行在同一个 EngineCore 对象里。
```

### 3.2 MPClient：通过 ZMQ 和后台 EngineCore 通信

multi-process 模式下，`MPClient` 的注释说明：

```python
MPClient: base client for multi-proc EngineCore.
    EngineCore runs in a background process busy loop, getting
    new EngineCoreRequests and returning EngineCoreOutputs

    * pushes EngineCoreRequests via input_socket
    * pulls EngineCoreOutputs via output_socket
```

位置：`vllm/vllm/v1/engine/core_client.py:467` 到 `vllm/vllm/v1/engine/core_client.py:475`

同步 MP 模式中，请求通过 `_send_input()` 发出去：

```python
self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/vllm/v1/engine/core_client.py:886` 到 `vllm/vllm/v1/engine/core_client.py:889`

异步 MP 模式中，请求通过：

```python
await self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/vllm/v1/engine/core_client.py:1121` 到 `vllm/vllm/v1/engine/core_client.py:1124`

因此：

```text
EngineCoreClient 屏蔽了 EngineCore 是同进程还是后台进程的差异；
外层 Engine 只需要调用 add_request / get_output / abort / utility 方法。
```

---

## 4. EngineCore 和 Scheduler 的关系

`Scheduler` 是 EngineCore 内部的调度组件。

EngineCore 初始化时会先创建 `model_executor`、初始化 KV cache，并在需要时调整 `scheduler_config` / `cache_config`，然后创建 Scheduler：

```python
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
...
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    include_finished_set=include_finished_set,
    log_stats=self.log_stats,
    block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
)
```

位置：`vllm/vllm/v1/engine/core.py:136` 到 `vllm/vllm/v1/engine/core.py:158`

这说明 Scheduler 不是外部 Engine 直接创建的，而是在 EngineCore 初始化模型、KV cache 之后创建。

### 4.1 为什么 Scheduler 在 KV cache 初始化后创建

EngineCore 初始化时先创建 model executor：

```python
self.model_executor = executor_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:123`

然后初始化 KV cache：

```python
kv_cache_config = self._initialize_kv_caches(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:132` 到 `vllm/vllm/v1/engine/core.py:133`

之后才创建 Scheduler。

原因是 Scheduler 需要知道 KV cache 配置，例如 block size、KV cache groups、可用 block 数等。`Scheduler` 初始化参数里就包含：

```text
kv_cache_config
block_size
hash_block_size
```

所以初始化顺序是：

```text
EngineCore
  → 创建 model_executor
  → profile / 初始化 KV cache
  → 必要时处理 non_causal / attention-free KV cache 和 auto-fit max_model_len
  → 生成 scheduler_kv_cache_config
  → 解析 scheduler_block_size / hash_block_size
  → 创建 Scheduler
```

### 4.2 请求进入 Scheduler

EngineCore 收到内部 `Request` 后，会调用：

```python
self.scheduler.add_request(request)
```

位置：`vllm/vllm/v1/engine/core.py:412`

外部 abort 则调用：

```python
self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

位置：`vllm/vllm/v1/engine/core.py:424`

所以 Scheduler 管理请求队列和请求状态：

```text
EngineCore.add_request()
  → Scheduler.add_request()
  → waiting / skipped_waiting / requests

EngineCore.abort_requests()
  → Scheduler.finish_requests()
  → 请求进入 finished / aborted 清理流程
```

### 4.3 一轮 step 中，EngineCore 调用 Scheduler 做调度和回收

EngineCore 的核心 `step()` 里，调度入口是：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
```

位置：`vllm/vllm/v1/engine/core.py:499`

Worker 执行完成后，结果回收入口是：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:512` 到 `vllm/vllm/v1/engine/core.py:515`

所以 Scheduler 在 EngineCore 中承担两件核心工作：

```text
1. schedule()：
   生成本轮执行计划 SchedulerOutput。

2. update_from_output()：
   用 Worker 返回的 ModelRunnerOutput 更新请求状态，产生 EngineCoreOutputs。
```

### 4.4 EngineCore 不关心 token 级细节

EngineCore 不直接判断：

```text
哪些请求在 waiting / running？
本轮每个请求调度多少 token？
prefix cache 命中了多少？
KV block 是否够？
是否需要抢占？
spec token 接受/拒绝后如何回退？
请求是否 finished，如何释放 block？
```

这些由 Scheduler 处理。

EngineCore 只把 Scheduler 当成一个组件：

```text
给 Scheduler 请求；
从 Scheduler 拿执行计划；
执行计划交给 Worker；
Worker 结果再交回 Scheduler。
```

---

## 5. EngineCore 和 ModelExecutor / Worker / ModelRunner 的关系

EngineCore 初始化时创建 `model_executor`：

```python
self.model_executor = executor_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:123`

这里的 `executor_class` 来自外层 Engine：

```python
executor_class=Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:152` 到 `vllm/vllm/v1/engine/llm_engine.py:154`

可以理解为：

```text
EngineCore 持有 model_executor；
model_executor 负责和实际 Worker / ModelRunner 交互；
Worker / ModelRunner 负责真正执行模型 forward、sampling、KV transfer 等。
```

### 5.1 EngineCore 通过 model_executor 初始化 KV cache 和模型执行环境

EngineCore 初始化 KV cache 时会问 model executor：

```python
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```

位置：`vllm/vllm/v1/engine/core.py:246` 到 `vllm/vllm/v1/engine/core.py:247`

如果模型有 KV cache，会通过 profiling 确定可用显存：

```python
available_gpu_memory = self.model_executor.determine_available_memory()
```

位置：`vllm/vllm/v1/engine/core.py:282` 到 `vllm/vllm/v1/engine/core.py:284`

最后初始化 Worker 侧 KV cache / warmup 模型：

```python
self.model_executor.initialize_from_config(kv_cache_configs)
```

位置：`vllm/vllm/v1/engine/core.py:320` 到 `vllm/vllm/v1/engine/core.py:321`

这说明：

```text
EngineCore 负责协调初始化；
真正模型和 KV cache 的物理执行环境在 model_executor / Worker 侧完成。
```

### 5.2 EngineCore 通过 model_executor 执行模型

普通 `step()` 中，EngineCore 把 SchedulerOutput 交给 model executor：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:500`

然后等待结果：

```python
model_output = future.result()
```

位置：`vllm/vllm/v1/engine/core.py:506`

如果 `execute_model()` 返回的 `model_output` 是 `None`，说明需要单独 sampling：

```python
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:507` 到 `vllm/vllm/v1/engine/core.py:508`

这里体现了一个重要边界：

```text
EngineCore 负责调用 execute_model / sample_tokens；
具体 forward 和 sample 的实现不在 EngineCore，而在 model_executor / Worker / ModelRunner。
```

### 5.3 EngineCore 也转发一些 Worker 侧 utility 能力

例如：

```python
self.model_executor.profile(is_start, profile_prefix)
self.model_executor.reset_mm_cache()
self.model_executor.reset_encoder_cache()
self.model_executor.sleep(level)
self.model_executor.wake_up(tags)
self.model_executor.add_lora(lora_request)
self.model_executor.remove_lora(lora_id)
self.model_executor.collective_rpc(...)
```

位置示例：

- profile：`vllm/vllm/v1/engine/core.py:671` 到 `vllm/vllm/v1/engine/core.py:672`
- reset mm cache：`vllm/vllm/v1/engine/core.py:674` 到 `vllm/vllm/v1/engine/core.py:687`
- reset encoder cache：`vllm/vllm/v1/engine/core.py:696` 到 `vllm/vllm/v1/engine/core.py:714`
- sleep / wake_up：`vllm/vllm/v1/engine/core.py:770` 到 `vllm/vllm/v1/engine/core.py:824`
- LoRA：`vllm/vllm/v1/engine/core.py:833` 到 `vllm/vllm/v1/engine/core.py:843`
- collective_rpc：`vllm/vllm/v1/engine/core.py:855` 到 `vllm/vllm/v1/engine/core.py:862`

这些能力进一步说明：

```text
EngineCore 是 Worker 执行系统的控制层，
但不是模型 forward 的具体实现层。
```

---

## 6. EngineCore 是否直接执行 forward

严格说：

```text
EngineCore 不直接执行模型 forward。
```

它只是调用：

```python
self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:500`

真正的模型 forward 在 executor / Worker / ModelRunner 内部完成。

不过从系统职责上看，EngineCore 是“发起 forward 的控制者”：

```text
Scheduler 生成 SchedulerOutput；
EngineCore 把 SchedulerOutput 交给 model_executor；
model_executor / Worker / ModelRunner 执行 forward；
EngineCore 拿回 ModelRunnerOutput；
EngineCore 再交给 Scheduler.update_from_output()。
```

所以可以区分两层说法：

```text
从控制流角度：
  EngineCore 驱动一次 forward 执行。

从计算实现角度：
  EngineCore 不实现 forward，真正执行在 Worker / ModelRunner。
```

---

## 7. EngineCore 的主循环角色

EngineCore 的主循环职责最集中体现在 `step()`。

源码注释写得很清楚：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
```

位置：`vllm/vllm/v1/engine/core.py:488` 到 `vllm/vllm/v1/engine/core.py:493`

这句 `Schedule, execute, and make output` 就是 EngineCore 的核心职责。

普通 step 主流程是：

```text
1. 检查 Scheduler 里是否还有请求；
2. 调用 Scheduler.schedule() 生成 SchedulerOutput；
3. 调用 model_executor.execute_model()；
4. 获取 grammar bitmask；
5. 等待 ModelRunnerOutput；
6. 处理执行期间进入 abort 队列的请求；
7. 调用 Scheduler.update_from_output()；
8. 返回 EngineCoreOutputs 和 model_executed 标记。
```

对应代码主线：

```python
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
...
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:495` 到 `vllm/vllm/v1/engine/core.py:517`

这条链路就是：

```text
SchedulerOutput
  → ModelRunnerOutput
  → EngineCoreOutputs
```

EngineCore 负责把这三者串起来。

---

## 8. EngineCoreProc：后台进程模式下的 EngineCore

当 EngineCore 在后台进程中运行时，使用的是 `EngineCoreProc`。

源码注释：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/vllm/v1/engine/core.py:905` 到 `vllm/vllm/v1/engine/core.py:906`

它仍然继承 `EngineCore`，但增加了 ZMQ 输入输出线程和 busy loop。

### 8.1 busy loop 主线

后台进程入口：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
```

位置：`vllm/vllm/v1/engine/core.py:1268` 到 `vllm/vllm/v1/engine/core.py:1276`

这说明后台进程中的 EngineCore 不需要外部同步调用 `step()`；它自己循环：

```text
处理输入队列
  → 如果有请求或 batch queue，就 step
  → 把输出放进 output_queue
  → 输出线程通过 socket 发回前端
```

### 8.2 input queue 处理

输入线程从 ZMQ 收请求后，会放入 `input_queue`：

```python
self.input_queue.put_nowait((request_type, request))
```

位置：`vllm/vllm/v1/engine/core.py:1595` 到 `vllm/vllm/v1/engine/core.py:1596`

busy loop 中 `_process_input_queue()` 会取出请求并分发：

```python
req = self.input_queue.get(block=block)
self._handle_client_request(*req)
```

位置：`vllm/vllm/v1/engine/core.py:1294` 到 `vllm/vllm/v1/engine/core.py:1296`

### 8.3 client request 分发

`_handle_client_request()` 根据请求类型处理：

```text
ADD：添加请求；
ABORT：取消请求；
UTILITY：profile / reset / sleep / LoRA 等工具调用；
EXECUTOR_FAILED：executor 异常；
WAKEUP：唤醒 loop。
```

关键代码：

```python
if request_type == EngineCoreRequestType.ADD:
    req, request_wave = request
    ...
    self.add_request(req, request_wave)
elif request_type == EngineCoreRequestType.ABORT:
    self.abort_requests(request)
elif request_type == EngineCoreRequestType.UTILITY:
    ...
elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
    raise RuntimeError("Executor failed.")
```

位置：`vllm/vllm/v1/engine/core.py:1381` 到 `vllm/vllm/v1/engine/core.py:1410`

### 8.4 step 输出如何返回前端

`_process_engine_step()` 调用 `step_fn()`，把输出放入 `output_queue`：

```python
outputs, model_executed = self.step_fn()
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
self.post_step(model_executed)
```

位置：`vllm/vllm/v1/engine/core.py:1309` 到 `vllm/vllm/v1/engine/core.py:1318`

输出线程再从 `output_queue` 取出，序列化后通过 ZMQ 发回前端：

```python
client_index, outputs = output
outputs.engine_index = engine_index
...
tracker = sockets[client_index].send_multipart(...)
```

位置：`vllm/vllm/v1/engine/core.py:1633` 到 `vllm/vllm/v1/engine/core.py:1657`

所以后台进程模式下：

```text
EngineCoreProc = EngineCore + ZMQ 输入输出 + busy loop + 生命周期管理。
```

---

## 9. EngineCore 的输入输出边界

### 9.1 输入：EngineCoreRequest / Request

外层 Engine 传给 EngineCoreClient 的是 `EngineCoreRequest`。

EngineCore 内部会通过：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:878`

把它转成 Scheduler 使用的内部 `Request`。

如果请求使用结构化输出，还会初始化 grammar：

```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
```

位置：`vllm/vllm/v1/engine/core.py:879` 到 `vllm/vllm/v1/engine/core.py:885`

所以输入边界是：

```text
外层 Engine：EngineCoreRequest
EngineCore 内部：Request
Scheduler：Request
```

### 9.2 输出：EngineCoreOutputs

Scheduler.update_from_output() 返回的是：

```text
dict[int, EngineCoreOutputs]
```

其中 key 是 `client_index`。

EngineCoreProc 会把这些 `(client_index, EngineCoreOutputs)` 放入 output queue；InprocClient 则取 client 0 的输出返回：

```python
return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

外层 `LLMEngine` / `AsyncLLM` 再用 `OutputProcessor` 把 `EngineCoreOutputs` 转为用户可见输出。

---

## 10. 容易疑惑的点

### 10.1 EngineCore 是不是 Scheduler？

不是。

```text
EngineCore 是执行闭环总控；
Scheduler 是 EngineCore 内部的调度和状态管理组件。
```

EngineCore 调用 Scheduler，但不替 Scheduler 做 token 级调度。

### 10.2 EngineCore 是不是 Worker？

不是。

```text
EngineCore 调用 model_executor；
model_executor / Worker / ModelRunner 负责真正模型 forward。
```

EngineCore 不直接实现模型 forward。

### 10.3 EngineCore 和 EngineCoreClient 是什么关系？

```text
EngineCore 是实际执行核心；
EngineCoreClient 是外层 Engine 访问 EngineCore 的客户端封装。
```

如果是 in-process，Client 直接调用 EngineCore。

如果是 multi-process，Client 通过 ZMQ 和后台 EngineCoreProc 通信。

### 10.4 EngineCore 和 LLMEngine / AsyncLLM 谁更靠外？

```text
LLMEngine / AsyncLLM 更靠外；
EngineCore 更靠内。
```

外层负责输入输出处理和用户接口；EngineCore 负责内部执行闭环。

### 10.5 EngineCoreOutputs 是最终用户输出吗？

不是最终用户输出。

```text
EngineCoreOutputs 是 EngineCore 返回给外层 Engine 的内部输出；
外层 OutputProcessor 还会把它转换成 RequestOutput / PoolingRequestOutput。
```

---

## 11. 从“回答问题”的角度总结

如果要问：

```text
EngineCore 在 vLLM 里负责什么？
```

可以回答：

```text
EngineCore 是 vLLM V1 Engine 的内部执行主循环。

它接收外层 Engine 处理好的 EngineCoreRequest，
转成内部 Request 后交给 Scheduler；
每轮 step 中，它调用 Scheduler.schedule() 得到 SchedulerOutput，
再把 SchedulerOutput 交给 model_executor / Worker / ModelRunner 执行，
拿到 ModelRunnerOutput 后，再调用 Scheduler.update_from_output()，
最终得到 EngineCoreOutputs 返回给上层。
```

它的角色可以概括为：

```text
EngineCore：执行闭环总控
Scheduler：调度决策和请求状态账本
model_executor / Worker / ModelRunner：真正执行模型 forward / sampling
LLMEngine / AsyncLLM：外层用户接口、输入处理、输出处理
```

---

## 12. 最关键的关系图

```text
外层同步路径：

LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → InputProcessor.assign_request_id()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()

LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCore.step_fn()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

```text
外层异步路径：

AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → InputProcessor.assign_request_id()
  → EngineCoreClient.add_request_async()
  → EngineCoreProc.process_input_sockets()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → 后台 EngineCoreProc input_queue
  → EngineCore.add_request()
  → Scheduler.add_request()

AsyncLLM._run_output_handler() 内创建的 output_handler 后台任务
  → EngineCoreClient.get_output_async()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutputCollector / async generator
```

```text
EngineCore 内部一轮 step：

EngineCore.step()
  → scheduler.has_requests()
  → scheduler.schedule()
  → model_executor.execute_model(scheduler_output, non_block=True)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → future.result()
  → model_executor.sample_tokens(grammar_output)  # 如果 model_output is None
  → _process_aborts_queue()
  → scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
```

```text
组件职责：

LLMEngine / AsyncLLM
  外层接口、输入处理、输出处理

EngineCoreClient
  隐藏 in-process / multi-process 通信差异

EngineCore
  内部执行闭环总控

Scheduler
  请求队列、token budget、KV block、状态更新

model_executor / Worker / ModelRunner
  模型 forward、sampling、KV/encoder 实际执行
```
