# vLLM V1 Engine 问题目录

源码位置：

- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/input_processor.py`
- `vllm/v1/engine/output_processor.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/request.py`

这个目录按问题拆解 vLLM V1 外层 `Engine` 体系，重点回答：`Engine` 到底指什么、`LLMEngine` 和 `AsyncLLM` 分别承担什么角色、外层 Engine 如何把用户输入转成 `EngineCoreRequest`、如何通过 `EngineCoreClient` 驱动内部 `EngineCore`、如何把 `EngineCoreOutputs` 转成用户可见输出，以及外层 Engine 和内部 EngineCore / Scheduler / Worker 的职责边界。

---

## 1. 总览文档

- [vLLM V1 Engine 逻辑梳理](engine_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
用户 / API server / LLM
  → LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output, non_block=True)
  → Scheduler.get_grammar_bitmask(...)
  → Worker / ModelRunner forward
  → ModelRunnerOutput 或 model_executor.sample_tokens(grammar_output)
  → Scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

核心分层：

```text
外层 Engine：
  LLMEngine / AsyncLLM + InputProcessor + OutputProcessor + EngineCoreClient。

内部 EngineCore：
  EngineCore + Scheduler + model_executor。

执行层：
  Executor / Worker / ModelRunner。
```

---

## 2. 主线专题阅读顺序

### 01. Engine 的定位

- [Engine 在 vLLM V1 里指什么？](01_engine_role.md)

回答：

```text
Engine 是具体类还是泛称？
LLMEngine / AsyncLLM 和 Engine 是什么关系？
Engine 和 EngineCore 是什么关系？
InputProcessor / OutputProcessor 属于哪一层？
EngineCoreClient 为什么夹在中间？
```

核心结论：

```text
Engine 是外层引擎体系的泛称；
LLMEngine 和 AsyncLLM 是两种具体形态；
EngineCore 是它们通过 EngineCoreClient 驱动的内部执行核心。
```

---

### 02. 同步 LLMEngine

- [LLMEngine 负责什么？](02_llm_engine_sync.md)

回答：

```text
LLMEngine 初始化了哪些组件？
add_request() 如何进入 EngineCore？
step() 如何拉取并处理输出？
n > 1 parallel sampling 如何 fan out？
同步控制接口如何转发给 EngineCore？
```

核心链路：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()

LLMEngine.step()
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

### 03. 异步 AsyncLLM

- [AsyncLLM 负责什么？](03_async_llm.md)

回答：

```text
AsyncLLM 和 LLMEngine 有什么相同点？
异步请求如何进入 EngineCore？
_run_output_handler() 内部的 output_handler task 如何消费 EngineCoreOutputs？
RequestOutputCollector / async generator 如何返回输出？
streaming input 如何进入 EngineCore？
AsyncMPClient 在异步路径中负责什么？
```

核心链路：

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()

AsyncLLM._run_output_handler() 内部的 output_handler task
  → EngineCoreClient.get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put()
  → generate() yield
```

---

### 04. 输入处理

- [InputProcessor 如何把用户输入转成 EngineCoreRequest？](04_input_processor.md)

回答：

```text
InputProcessor 位于哪一层？
EngineInput / PromptType 如何变成 EngineCoreRequest？
SamplingParams / PoolingParams 如何校验和补全？
request_id / external_req_id 如何处理？
prompt token ids、prompt embeds、多模态输入如何进入内部请求？
```

核心链路：

```text
用户输入 / PromptType / EngineInput
  → InputProcessor.process_inputs()
  → 参数校验 / 输入校验 / 多模态整理
  → EngineCoreRequest
```

---

### 05. 输出处理

- [OutputProcessor 如何把 EngineCoreOutputs 转成用户输出？](05_output_processor.md)

回答：

```text
OutputProcessor 位于哪一层？
EngineCoreOutput 和 RequestOutput 有什么区别？
RequestState 保存什么？
detokenize、stop string、logprobs 在哪里处理？
同步和异步输出分发有什么差异？
reqs_to_abort 从哪里来？
```

核心链路：

```text
EngineCoreOutput
  → RequestState
  → detokenizer / logprobs_processor
  → CompletionOutput / PoolingOutput
  → RequestOutput / PoolingRequestOutput
```

---

### 06. EngineCoreClient 桥接层

- [EngineCoreClient 如何连接外层 Engine 和 EngineCore？](06_engine_core_client_bridge.md)

回答：

```text
为什么 LLMEngine / AsyncLLM 通常持有 EngineCoreClient？
InprocClient、SyncMPClient、AsyncMPClient 有什么区别？
add_request / get_output / abort 如何屏蔽进程差异？
UTILITY 调用如何转发？
DPAsyncMPClient / DPLBAsyncMPClient 如何接入？
```

核心结论：

```text
EngineCoreClient 是外层 Engine 和内部 EngineCore 的通信桥；
它屏蔽同进程、多进程、同步、异步和 DP 路由差异。
```

---

### 07. 请求生命周期

- [一个请求在外层 Engine 中如何流动？](07_request_lifecycle.md)

回答：

```text
用户请求从 add_request 到 Scheduler 的完整路径是什么？
EngineCoreRequest 和 Request 有什么区别？
OutputProcessor.add_request 为什么早于输出处理？
同步 / 异步请求路径有什么不同？
streaming input 请求如何流动？
abort 请求如何同时清理输出侧和调度侧状态？
```

核心链路：

```text
用户请求
  → InputProcessor：变成 EngineCoreRequest
  → OutputProcessor：登记输出状态
  → EngineCoreClient：送入 EngineCore
  → EngineCore：变成 Request
  → Scheduler：进入调度队列
```

---

### 08. 输出生命周期

- [一次输出如何从 EngineCore 回到用户？](08_output_lifecycle.md)

回答：

```text
EngineCoreOutputs 如何被 LLMEngine.step() 消费？
AsyncLLM._run_output_handler() 内部的 output_handler task 如何分发输出？
EngineCoreOutput / EngineCoreOutputs / RequestOutput 有什么区别？
RequestOutput / PoolingRequestOutput 何时构造？
多进程输出如何通过 ZMQ 回到前端？
```

核心链路：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

### 09. 初始化与配置

- [Engine 初始化时如何配置 EngineCore？](09_initialization_and_config.md)

回答：

```text
vllm_config 如何进入 LLMEngine / AsyncLLM / EngineCore？
executor_class 如何选择？
Executor.get_class(vllm_config) 做了什么？
EngineCoreClient.make_client / make_async_mp_client 如何选择 client？
EngineCore 何时创建 executor / KV cache / Scheduler？
多进程 ready response 如何回写前端配置？
```

核心链路：

```text
EngineArgs / AsyncEngineArgs
  → VllmConfig
  → Executor.get_class(vllm_config)
  → LLMEngine / AsyncLLM
  → EngineCoreClient
  → EngineCore
      → executor_class(vllm_config)
      → KV cache init
      → Scheduler init
```

---

### 10. 生命周期与控制能力

- [Engine 的 abort、profile、sleep、shutdown 如何接入？](10_lifecycle_and_control.md)

回答：

```text
外层 Engine 如何取消请求？
profile / reset / sleep / wake_up 如何转发到 EngineCore？
pause_generation / resume_generation 如何接入 Scheduler？
LoRA / collective_rpc 如何转发到 executor？
同步和异步 shutdown 有什么差异？
```

核心模式：

```text
LLMEngine / AsyncLLM 控制 API
  → EngineCoreClient
  → EngineCore
  → Scheduler / model_executor / Worker
```

---

### 11. Engine 和 EngineCore 边界

- [Engine 和 EngineCore 的边界是什么？](11_engine_vs_engine_core_boundaries.md)

回答：

```text
Engine 和 EngineCore 分别在哪一层？
Engine 负责什么，不负责什么？
EngineCore 负责什么，不负责什么？
InputProcessor / OutputProcessor / Scheduler / Worker 分别属于哪个边界？
EngineInput、EngineCoreRequest、Request 的对象边界是什么？
EngineCoreOutputs 和 RequestOutput 的输出边界是什么？
```

核心结论：

```text
Engine 管用户侧输入输出；
EngineCore 管内部调度执行闭环。
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
engine_overview.md
  → 01_engine_role.md
  → 11_engine_vs_engine_core_boundaries.md
```

适合先搞清楚：Engine 是什么、EngineCore 是什么、两者怎么分层。

---

### 3.2 按同步 / 异步外层接口阅读

```text
engine_overview.md
  → 02_llm_engine_sync.md
  → 03_async_llm.md
```

适合重点理解：同步 `LLMEngine.step()` 和异步 `AsyncLLM._run_output_handler()` 内部 output handler task 的差异。

---

### 3.3 按完整请求链路阅读

```text
engine_overview.md
  → 01_engine_role.md
  → 04_input_processor.md
  → 07_request_lifecycle.md
  → 06_engine_core_client_bridge.md
  → ../engine_core/02_request_entry.md
  → ../engine_core/03_step_loop.md
```

适合系统理解请求如何从用户侧进入 Scheduler / Worker 执行闭环。

---

### 3.4 按完整输出链路阅读

```text
../engine_core/07_engine_core_outputs.md
  → 08_output_lifecycle.md
  → 05_output_processor.md
  → 03_async_llm.md
```

适合系统理解一次输出如何从 `ModelRunnerOutput` 变成用户可见 `RequestOutput`。

---

### 3.5 按初始化和运行形态阅读

```text
09_initialization_and_config.md
  → 06_engine_core_client_bridge.md
  → ../engine_core/08_lifecycle_and_async.md
  → 10_lifecycle_and_control.md
```

适合重点理解：in-process、多进程、异步、DP、shutdown / sleep / utility 的关系。

---

### 3.6 和 EngineCore 文档联动阅读

```text
11_engine_vs_engine_core_boundaries.md
  → ../engine_core/01_engine_core_role.md
  → ../engine_core/03_step_loop.md
  → ../engine_core/05_worker_execution.md
  → ../engine_core/07_engine_core_outputs.md
```

重点关注：

```text
Engine 负责外层输入输出；
EngineCore 负责编排 Scheduler 和 Worker；
Scheduler 负责调度账本；
Worker / ModelRunner 负责真正模型计算。
```

---

## 4. 核心概念速查

```text
Engine：
  外层引擎体系的泛称，不一定是单一具体类。

LLMEngine：
  同步 Engine 形态，负责同步 add_request / step / 同步控制 API。

AsyncLLM：
  异步 Engine 形态，负责 generate / encode / output_handler / async generator。

InputProcessor：
  把用户输入、params、多模态输入转换成 EngineCoreRequest。

EngineCoreRequest：
  外层 Engine 传给 EngineCore 的请求协议。

Request：
  EngineCore 转换后交给 Scheduler 的内部调度对象。

RequestOutputCollector：
  AsyncLLM 中每个请求的异步输出队列。

EngineCoreClient：
  屏蔽同进程 / 多进程 / 同步 / 异步 / DP 通信差异。

EngineCore：
  内部执行闭环总控。

EngineCoreProc：
  后台进程版 EngineCore，额外管理 ZMQ、input/output queue、busy loop。

Scheduler：
  请求队列、token budget、KV block、状态账本、输出回收。

model_executor：
  执行器抽象，负责把 execute_model / sample_tokens 分发到 Worker。

Worker / ModelRunner：
  模型输入准备、forward、logits、sampling、pooling、KV / encoder 实际执行。

EngineCoreOutput：
  Scheduler 生成的单请求内部增量输出。

EngineCoreOutputs：
  EngineCore 返回给某个 client 的一批内部输出。

OutputProcessor：
  把 EngineCoreOutputs 转成 RequestOutput / PoolingRequestOutput。

RequestOutput / PoolingRequestOutput：
  用户可见输出。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
Engine = InputProcessor + EngineCoreClient + OutputProcessor + 同步/异步外层接口。

EngineCore = Scheduler + model_executor 的内部执行闭环总控。
```

展开就是：

```text
用户请求
  → LLMEngine / AsyncLLM
  → InputProcessor：变成 EngineCoreRequest
  → OutputProcessor：登记输出状态
  → EngineCoreClient：送入 EngineCore
  → EngineCore：变成 Request 并交给 Scheduler
  → Scheduler / Worker：完成调度和模型执行
  → EngineCoreOutputs
  → OutputProcessor：变成用户可见输出
```

再压缩成一句话：

```text
Engine 管用户侧输入输出，EngineCore 管内部调度执行。
```
