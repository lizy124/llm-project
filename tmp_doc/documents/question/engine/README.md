# vLLM V1 Engine 问题目录

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/processor.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/__init__.py`

这个目录按问题拆解 vLLM V1 外层 `Engine` 体系，重点回答：`Engine` 到底指什么、`LLMEngine` 和 `AsyncLLM` 分别承担什么角色、它们如何通过 `InputProcessor` 把用户输入转成 `EngineCoreRequest`、如何通过 `EngineCoreClient` 驱动 `EngineCore`、以及如何通过 `OutputProcessor` 把 `EngineCoreOutputs` 转成用户可见输出。

---

## 1. 总览文档

- [vLLM V1 Engine 逻辑梳理](engine_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
用户 / API / LLM
  → LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore
  → Scheduler / Worker / ModelRunner
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
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
Engine 负责哪些外层工作？
```

核心结论：

```text
Engine 是外层引擎体系的泛称；
LLMEngine 和 AsyncLLM 是两种具体形态；
EngineCore 是它们驱动的内部执行核心。
```

---

### 02. 同步 LLMEngine

- [LLMEngine 负责什么？](02_llm_engine_sync.md)

回答：

```text
LLMEngine 初始化了哪些组件？
add_request() 如何进入 EngineCore？
step() 如何拉取并处理输出？
同步接口为什么是外层 Engine？
```

---

### 03. 异步 AsyncLLM

- [AsyncLLM 负责什么？](03_async_llm.md)

回答：

```text
AsyncLLM 和 LLMEngine 有什么相同点？
异步请求如何进入 EngineCore？
output_handler 如何消费 EngineCoreOutputs？
RequestOutputCollector / async generator 如何返回输出？
```

---

### 04. 输入处理

- [InputProcessor 如何把用户输入转成 EngineCoreRequest？](04_input_processor.md)

回答：

```text
InputProcessor 位于哪一层？
EngineInput 和 EngineCoreRequest 有什么区别？
prompt、sampling params、多模态输入如何进入内部请求？
```

---

### 05. 输出处理

- [OutputProcessor 如何把 EngineCoreOutputs 转成用户输出？](05_output_processor.md)

回答：

```text
OutputProcessor 位于哪一层？
EngineCoreOutput 和 RequestOutput 有什么区别？
detokenize、stop string、finished 状态在哪里处理？
同步和异步输出处理有什么差异？
```

---

### 06. EngineCoreClient 桥接层

- [EngineCoreClient 如何连接外层 Engine 和 EngineCore？](06_engine_core_client_bridge.md)

回答：

```text
为什么 LLMEngine / AsyncLLM 通常持有 EngineCoreClient？
InprocClient、SyncMPClient、AsyncMPClient 有什么区别？
add_request / get_output / abort 如何屏蔽进程差异？
```

---

### 07. 请求生命周期

- [一个请求在外层 Engine 中如何流动？](07_request_lifecycle.md)

回答：

```text
用户请求从 add_request 到 EngineCore 的完整路径是什么？
OutputProcessor.add_request 为什么早于输出处理？
abort / finish request 在外层如何体现？
```

---

### 08. 输出生命周期

- [一次输出如何从 EngineCore 回到用户？](08_output_lifecycle.md)

回答：

```text
EngineCoreOutputs 如何被 LLMEngine.step() 消费？
AsyncLLM output_handler 如何分发输出？
RequestOutput / PoolingRequestOutput 何时构造？
```

---

### 09. 初始化与配置

- [Engine 初始化时如何配置 EngineCore？](09_initialization_and_config.md)

回答：

```text
vllm_config 如何进入 LLMEngine / AsyncLLM？
executor_class 如何选择？
EngineCoreClient.make_client / make_async_mp_client 做了什么？
```

---

### 10. 生命周期与控制能力

- [Engine 的 abort、profile、sleep、shutdown 如何接入？](10_lifecycle_and_control.md)

回答：

```text
外层 Engine 如何取消请求？
profile / reset / sleep / wake_up 如何转发到 EngineCore？
同步和异步 shutdown 有什么差异？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
engine_overview.md
  → 01_engine_role.md
  → 02_llm_engine_sync.md
  → 03_async_llm.md
```

### 3.2 按完整请求链路阅读

```text
engine_overview.md
  → 01_engine_role.md
  → 04_input_processor.md
  → 07_request_lifecycle.md
  → 06_engine_core_client_bridge.md
  → ../engine_core/README.md
  → 08_output_lifecycle.md
  → 05_output_processor.md
```

### 3.3 和 EngineCore 文档联动阅读

```text
01_engine_role.md
  → 06_engine_core_client_bridge.md
  → ../engine_core/01_engine_core_role.md
  → ../engine_core/03_step_loop.md
  → 08_output_lifecycle.md
```

---

## 4. 核心概念速查

```text
Engine：
  外层引擎体系的泛称，不一定是单一具体类。

LLMEngine：
  同步 Engine 形态，负责同步 add_request / step。

AsyncLLM：
  异步 Engine 形态，负责异步请求、后台输出处理和 async generator。

InputProcessor：
  把用户输入转换成 EngineCoreRequest。

EngineCoreClient：
  屏蔽同进程 / 多进程 / 异步通信差异。

EngineCore：
  内部执行闭环总控。

OutputProcessor：
  把 EngineCoreOutputs 转成 RequestOutput / PoolingRequestOutput。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
Engine = 外层输入输出编排层；
EngineCore = 内层 schedule → execute → update → output 执行闭环。
```

展开就是：

```text
LLMEngine / AsyncLLM 接用户请求；
InputProcessor 转内部请求；
EngineCoreClient 送入 EngineCore；
EngineCore 完成内部执行；
OutputProcessor 转成用户可见输出。
```