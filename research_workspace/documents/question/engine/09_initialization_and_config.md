# 09. Engine 初始化时如何配置 EngineCore？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/utils.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/executor/abstract.py`

本问题关注：外层 Engine 初始化时如何创建 `InputProcessor`、`OutputProcessor`、`EngineCoreClient`，以及如何把配置和 executor 选择传给内部 EngineCore。

---

## 1. 一句话回答

vLLM V1 中，外层 Engine 初始化的核心任务是：

```text
用 vllm_config 创建外层输入输出处理器，
选择 executor_class，
再通过 EngineCoreClient 创建通往 EngineCore 的访问路径。
```

同步路径是：

```text
LLMEngine
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient.make_client(...)
      → InprocClient / SyncMPClient
      → EngineCore / EngineCoreProc
```

异步路径是：

```text
AsyncLLM
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient.make_async_mp_client(...)
      → AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
      → 后台 EngineCoreProc
```

而真正的模型执行环境、KV cache profiling、Scheduler 创建等内部初始化发生在 `EngineCore.__init__()` 中。

一句话：

```text
LLMEngine / AsyncLLM 负责外层 Engine 初始化；
EngineCoreClient 负责选择同进程或多进程访问方式；
EngineCore 负责内部 executor、KV cache、Scheduler 和执行闭环初始化。
```

---

## 2. 初始化链路总览

完整初始化可以分成四层。

### 2.1 参数层：EngineArgs / AsyncEngineArgs

外部通常先有：

```text
EngineArgs / AsyncEngineArgs
```

它们创建：

```text
VllmConfig
```

同步路径：

```python
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:170` 到 `vllm/vllm/v1/engine/llm_engine.py:172`

异步路径：

```python
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/async_llm.py:241` 到 `vllm/vllm/v1/engine/async_llm.py:243`

### 2.2 外层 Engine 层：LLMEngine / AsyncLLM

外层 Engine 使用 `vllm_config` 创建：

```text
renderer；
InputProcessor；
OutputProcessor；
EngineCoreClient；
logger / tracing / profiler；
```

### 2.3 Client 层：EngineCoreClient

`EngineCoreClient` 根据同步 / 异步、多进程 / 同进程、DP 配置决定实际 client：

```text
InprocClient；
SyncMPClient；
AsyncMPClient；
DPAsyncMPClient；
DPLBAsyncMPClient。
```

### 2.4 内部 EngineCore 层

`EngineCore` 使用传入的 `vllm_config` 和 `executor_class` 初始化：

```text
model_executor；
KV cache；
StructuredOutputManager；
Scheduler；
batch_queue；
request_block_hasher；
step_fn；
```

总图：

```text
EngineArgs / AsyncEngineArgs
  → VllmConfig
  → Executor.get_class(vllm_config)
  → LLMEngine / AsyncLLM
      → InputProcessor(vllm_config, renderer)
      → OutputProcessor(tokenizer, ...)
      → EngineCoreClient(..., vllm_config, executor_class)
          → EngineCore(vllm_config, executor_class)
              → executor_class(vllm_config)
              → initialize KV cache
              → Scheduler(...)
```

---

## 3. vllm_config 的作用

`vllm_config` 是贯穿整个初始化链路的核心配置对象。

它会被传给：

```text
LLMEngine / AsyncLLM；
InputProcessor；
OutputProcessor 的部分参数；
EngineCoreClient；
EngineCore；
Executor；
Scheduler；
Worker / ModelRunner。
```

在不同层里使用的字段不同。

### 3.1 外层 Engine 使用的配置

`LLMEngine.__init__()` 保存：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`vllm/vllm/v1/engine/llm_engine.py:62` 到 `vllm/vllm/v1/engine/llm_engine.py:64`

`AsyncLLM.__init__()` 也保存：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`vllm/vllm/v1/engine/async_llm.py:110` 到 `vllm/vllm/v1/engine/async_llm.py:112`

外层主要用这些配置做：

```text
renderer / tokenizer 初始化；
InputProcessor 初始化；
OutputProcessor 的 stream_interval / tracing 配置；
stats logger 初始化；
profile / tracing 设置；
data parallel frontend 行为控制；
```

### 3.2 InputProcessor 使用的配置

`InputProcessor.__init__()` 会拆出很多配置：

```python
self.vllm_config = vllm_config
self.model_config = model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.scheduler_config = vllm_config.scheduler_config
self.speculative_config = vllm_config.speculative_config
self.structured_outputs_config = vllm_config.structured_outputs_config
self.observability_config = vllm_config.observability_config
self.use_v2_model_runner = vllm_config.use_v2_model_runner
```

位置：`vllm/vllm/v1/engine/input_processor.py:44` 到 `vllm/vllm/v1/engine/input_processor.py:52`

它还会创建 / 保存：

```text
renderer；
MultiModalBudget 推导出的 mm_encoder_cache_size / skip_prompt_length_check；
InputPreprocessor。
```

位置：`vllm/vllm/v1/engine/input_processor.py:54` 到 `vllm/vllm/v1/engine/input_processor.py:73`

它主要用来：

```text
校验 SamplingParams / PoolingParams；
校验 LoRA；
校验 prompt 长度；
处理多模态输入；
根据 generation_config / tokenizer 更新 sampling params；
构造 EngineCoreRequest。
```

### 3.3 EngineCore 使用的配置

`EngineCore.__init__()` 里保存：

```python
self.vllm_config = vllm_config
```

位置：`vllm/vllm/v1/engine/core.py:112`

它使用 `vllm_config` 完成内部初始化：

```text
加载 engine / scheduler 层插件；
创建 model_executor；
初始化 KV cache；
创建 StructuredOutputManager；
选择 Scheduler 类；
设置 scheduler block size / hash_block_size；
必要时禁用不适用的 chunked prefill / prefix caching；
配置 KV connector handshake；
配置 batch queue / step_fn；
判断 spec decode / diffusion draft-token 检查；
创建 request block hasher；
冻结 startup heap 并启用 env cache。
```

### 3.4 Executor 使用的配置

`Executor.__init__()` 中会保存：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.load_config = vllm_config.load_config
self.parallel_config = vllm_config.parallel_config
self.scheduler_config = vllm_config.scheduler_config
self.device_config = vllm_config.device_config
self.speculative_config = vllm_config.speculative_config
self.observability_config = vllm_config.observability_config
```

位置：`vllm/vllm/v1/executor/abstract.py:94` 到 `vllm/vllm/v1/executor/abstract.py:109`

Executor 使用这些配置初始化 Worker / ModelRunner 执行环境。

---

## 4. executor_class 如何选择

外层 Engine 不直接写死使用哪个 executor，而是通过：

```python
Executor.get_class(vllm_config)
```

同步路径：

```python
executor_class=Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:151` 到 `vllm/vllm/v1/engine/llm_engine.py:154`

异步路径：

```python
executor_class=Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/async_llm.py:216` 到 `vllm/vllm/v1/engine/async_llm.py:220`

### 4.1 Executor.get_class()

入口：

```python
@staticmethod
def get_class(vllm_config: VllmConfig) -> type["Executor"]:
```

位置：`vllm/vllm/v1/executor/abstract.py:47` 到 `vllm/vllm/v1/executor/abstract.py:48`

核心依据是：

```python
parallel_config = vllm_config.parallel_config
distributed_executor_backend = parallel_config.distributed_executor_backend
```

位置：`vllm/vllm/v1/executor/abstract.py:49` 到 `vllm/vllm/v1/executor/abstract.py:51`

### 4.2 backend 到 executor_class 的映射

`Executor.get_class()` 主要映射如下：

```text
backend = "ray"
  → RayExecutorV2 或 RayDistributedExecutor

backend = "mp"
  → MultiprocExecutor

backend = "uni"
  → UniProcExecutor

backend = "external_launcher"
  → ExecutorWithExternalLauncher

backend = 自定义 Executor 子类
  → 直接使用

backend = qualname 字符串
  → resolve_obj_by_qualname(...)
```

对应源码：

```python
elif distributed_executor_backend == "ray":
    ...
elif distributed_executor_backend == "mp":
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    executor_class = MultiprocExecutor
elif distributed_executor_backend == "uni":
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    executor_class = UniProcExecutor
elif distributed_executor_backend == "external_launcher":
    executor_class = ExecutorWithExternalLauncher
elif isinstance(distributed_executor_backend, str):
    executor_class = resolve_obj_by_qualname(distributed_executor_backend)
```

位置：`vllm/vllm/v1/executor/abstract.py:60` 到 `vllm/vllm/v1/executor/abstract.py:87`

### 4.3 executor_class 什么时候真正实例化

`Executor.get_class()` 只是选类。

真正实例化发生在 `EngineCore.__init__()`：

```python
self.model_executor = executor_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:123`

所以：

```text
LLMEngine / AsyncLLM：
  选择 executor_class。

EngineCore：
  实例化 executor_class，得到 model_executor。
```

### 4.4 executor 的职责

Executor 抽象注释说明：

```python
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """
```

位置：`vllm/vllm/v1/executor/abstract.py:37` 到 `vllm/vllm/v1/executor/abstract.py:42`

也就是说：

```text
executor_class 决定 EngineCore 后面如何执行模型：
单进程、本地多进程、Ray 分布式，或外部 launcher。
```

---

## 5. LLMEngine 初始化链路

同步外层 Engine 是 `LLMEngine`。

源码定义：

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

位置：`vllm/vllm/v1/engine/llm_engine.py:48` 到 `vllm/vllm/v1/engine/llm_engine.py:49`

### 5.1 from_engine_args()

从 EngineArgs 构造时：

```python
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:170` 到 `vllm/vllm/v1/engine/llm_engine.py:172`

如果环境变量开启 V1 multiprocessing：

```python
if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
    logger.debug("Enabling multiprocessing for LLMEngine.")
    enable_multiprocessing = True
```

位置：`vllm/vllm/v1/engine/llm_engine.py:174` 到 `vllm/vllm/v1/engine/llm_engine.py:176`

最后构造：

```python
return cls(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=not engine_args.disable_log_stats,
    usage_context=usage_context,
    stat_loggers=stat_loggers,
    multiprocess_mode=enable_multiprocessing,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:178` 到 `vllm/vllm/v1/engine/llm_engine.py:186`

### 5.2 from_vllm_config()

如果已经有 `VllmConfig`：

```python
return cls(
    vllm_config=vllm_config,
    executor_class=Executor.get_class(vllm_config),
    log_stats=(not disable_log_stats),
    usage_context=usage_context,
    stat_loggers=stat_loggers,
    multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:143` 到 `vllm/vllm/v1/engine/llm_engine.py:158`

### 5.3 LLMEngine.__init__ 保存配置

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`vllm/vllm/v1/engine/llm_engine.py:62` 到 `vllm/vllm/v1/engine/llm_engine.py:64`

### 5.4 data parallel group 初始化

如果不是 multiprocess_mode，并且 data parallel size > 1，会先初始化 DP group：

```python
if (
    not multiprocess_mode
    and parallel_config.data_parallel_size > 1
    and not self.external_launcher_dp
):
    self.dp_group = parallel_config.stateless_init_dp_group()
else:
    self.dp_group = None
```

位置：`vllm/vllm/v1/engine/llm_engine.py:72` 到 `vllm/vllm/v1/engine/llm_engine.py:88`

源码注释强调：

```python
# important: init dp group before init the engine_core
# In the decoupled engine case this is handled in EngineCoreProc.
```

位置：`vllm/vllm/v1/engine/llm_engine.py:79` 到 `vllm/vllm/v1/engine/llm_engine.py:80`

这说明：

```text
同进程 LLMEngine 需要前端先初始化 DP group；
多进程解耦模式下，这件事由 EngineCoreProc 处理。
```

如果是 external launcher DP，`LLMEngine` 会在 `EngineCoreClient` 创建后复用已有 DP group 的 CPU group：

```python
if self.external_launcher_dp:
    self.dp_group = get_dp_group().cpu_group
```

位置：`vllm/vllm/v1/engine/llm_engine.py:135` 到 `vllm/vllm/v1/engine/llm_engine.py:138`

### 5.5 创建 renderer

```python
self.renderer = renderer = renderer_from_config(self.vllm_config)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:91`

renderer 提供 tokenizer 和多模态渲染相关能力。

### 5.6 创建 InputProcessor

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:93` 到 `vllm/vllm/v1/engine/llm_engine.py:94`

### 5.7 创建 OutputProcessor

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(
    renderer.tokenizer,
    log_stats=self.log_stats,
    stream_interval=self.vllm_config.scheduler_config.stream_interval,
    tracing_enabled=tracing_endpoint is not None,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:96` 到 `vllm/vllm/v1/engine/llm_engine.py:102`

### 5.8 创建 EngineCoreClient

```python
# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:104` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

注意：字段名叫 `engine_core`，但类型是 `EngineCoreClient`。

根据 `multiprocess_mode`，它可能是：

```text
InprocClient：同进程直接持有 EngineCore；
SyncMPClient：通过 ZMQ 访问后台 EngineCoreProc。
```

### 5.9 统计日志初始化

```python
if self.log_stats:
    self.logger_manager = StatLoggerManager(...)
    self.logger_manager.log_engine_initialized()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:113` 到 `vllm/vllm/v1/engine/llm_engine.py:121`

### 5.10 in-process 兼容字段

如果不是 multiprocess_mode：

```python
self.model_executor = self.engine_core.engine_core.model_executor
```

位置：`vllm/vllm/v1/engine/llm_engine.py:123` 到 `vllm/vllm/v1/engine/llm_engine.py:125`

这主要是为了 v0 兼容。

同一分支还会捕获 driver model 并注册 finalizer，用于在 engine 删除时清理 bytecode hooks 持有的模型引用：

```python
model = self._get_driver_model_for_cleanup()
if model is not None:
    self._finalizer = weakref.finalize(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:127` 到 `vllm/vllm/v1/engine/llm_engine.py:133`

它说明在 inproc 下结构是：

```text
LLMEngine.engine_core = InprocClient
InprocClient.engine_core = EngineCore
EngineCore.model_executor = executor_class(vllm_config)
```

### 5.11 清理多模态 cache

初始化末尾：

```python
self.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:140` 到 `vllm/vllm/v1/engine/llm_engine.py:141`

注释：

```python
# Don't keep the dummy data in memory
```

位置：`vllm/vllm/v1/engine/llm_engine.py:140`

### 5.12 LLMEngine 初始化总图

```text
LLMEngine.from_engine_args()
  → engine_args.create_engine_config()
  → Executor.get_class(vllm_config)
  → LLMEngine.__init__()
      → 保存 vllm_config / model_config
      → 初始化 tracing
      → 必要时初始化 DP group
      → renderer_from_config()
      → InputProcessor(vllm_config, renderer)
      → OutputProcessor(tokenizer, stream_interval, tracing)
      → EngineCoreClient.make_client(...)
          → InprocClient 或 SyncMPClient
      → StatLoggerManager
      → reset_mm_cache()
```

---

## 6. AsyncLLM 初始化链路

异步外层 Engine 是 `AsyncLLM`。

源码定义：

```python
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
```

位置：`vllm/vllm/v1/engine/async_llm.py:70` 到 `vllm/vllm/v1/engine/async_llm.py:71`

### 6.1 from_engine_args()

异步路径从 `AsyncEngineArgs` 构造：

```python
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/async_llm.py:241` 到 `vllm/vllm/v1/engine/async_llm.py:243`

然后：

```python
return cls(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_requests=engine_args.enable_log_requests,
    log_stats=not engine_args.disable_log_stats,
    start_engine_loop=start_engine_loop,
    usage_context=usage_context,
    stat_loggers=stat_loggers,
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:245` 到 `vllm/vllm/v1/engine/async_llm.py:254`

### 6.2 from_vllm_config()

```python
return cls(
    vllm_config=vllm_config,
    executor_class=Executor.get_class(vllm_config),
    start_engine_loop=start_engine_loop,
    stat_loggers=stat_loggers,
    log_requests=enable_log_requests,
    log_stats=not disable_log_stats,
    ...
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:202` 到 `vllm/vllm/v1/engine/async_llm.py:229`

### 6.3 注册配置序列化

```python
maybe_register_config_serialize_by_value()
```

位置：`vllm/vllm/v1/engine/async_llm.py:107` 到 `vllm/vllm/v1/engine/async_llm.py:108`

异步路径通常会启动后台 EngineCoreProc，因此序列化能力很重要。

### 6.4 保存配置和 tracing

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`vllm/vllm/v1/engine/async_llm.py:110` 到 `vllm/vllm/v1/engine/async_llm.py:112`

如果 tracing endpoint 存在：

```python
init_tracer("vllm.llm_engine", tracing_endpoint)
```

位置：`vllm/vllm/v1/engine/async_llm.py:114` 到 `vllm/vllm/v1/engine/async_llm.py:116`

### 6.5 创建 renderer / InputProcessor / OutputProcessor

```python
self.renderer = renderer = renderer_from_config(self.vllm_config)
```

位置：`vllm/vllm/v1/engine/async_llm.py:132`

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/vllm/v1/engine/async_llm.py:134` 到 `vllm/vllm/v1/engine/async_llm.py:135`

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:137` 到 `vllm/vllm/v1/engine/async_llm.py:143`

### 6.6 创建异步 EngineCoreClient

```python
# EngineCore (starts the engine in background process).
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_count=client_count,
    client_index=client_index,
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:145` 到 `vllm/vllm/v1/engine/async_llm.py:153`

异步路径固定走 async mp client 系列：

```text
AsyncMPClient；
DPAsyncMPClient；
DPLBAsyncMPClient。
```

### 6.7 创建 logger_manager

创建 logger 前，AsyncLLM 会先加载自定义 stat logger 插件；如果存在自定义 logger，即使入参 `log_stats=False`，也会打开 `self.log_stats`，但不启用默认 logger：

```python
custom_stat_loggers = list(stat_loggers or [])
custom_stat_loggers.extend(load_stat_logger_plugin_factories())
...
self.log_stats = log_stats or has_custom_loggers
```

位置：`vllm/vllm/v1/engine/async_llm.py:120` 到 `vllm/vllm/v1/engine/async_llm.py:130`

然后创建 `StatLoggerManager`：

```python
self.logger_manager = StatLoggerManager(
    vllm_config=vllm_config,
    engine_idxs=self.engine_core.engine_ranks_managed,
    custom_stat_loggers=custom_stat_loggers,
    enable_default_loggers=log_stats,
    client_count=client_count,
    aggregate_engine_logging=aggregate_engine_logging,
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:155` 到 `vllm/vllm/v1/engine/async_llm.py:166`

这里使用 `engine_ranks_managed`，说明异步路径可能管理多个 EngineCore rank。

### 6.8 output_handler 生命周期

初始化时：

```python
self.output_handler: asyncio.Task | None = None
try:
    asyncio.get_running_loop()
    self._run_output_handler()
except RuntimeError:
    pass
```

位置：`vllm/vllm/v1/engine/async_llm.py:170` 到 `vllm/vllm/v1/engine/async_llm.py:176`

含义：

```text
如果当前已经在 event loop 中，就立即启动 output_handler；
否则等第一次 add_request() 再启动。
```

### 6.9 profiler

如果开启 torch profiler：

```python
self.profiler = torch.profiler.profile(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:178` 到 `vllm/vllm/v1/engine/async_llm.py:200`

### 6.10 AsyncLLM 初始化总图

```text
AsyncLLM.from_engine_args()
  → engine_args.create_engine_config()
  → Executor.get_class(vllm_config)
  → AsyncLLM.__init__()
      → maybe_register_config_serialize_by_value()
      → 保存 vllm_config / model_config
      → 初始化 tracing
      → renderer_from_config()
      → InputProcessor(vllm_config, renderer)
      → OutputProcessor(tokenizer, stream_interval, tracing)
      → EngineCoreClient.make_async_mp_client(...)
          → AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
          → 后台 EngineCoreProc
      → StatLoggerManager
      → output_handler 准备 / 启动
      → frontend profiler
```

---

## 7. EngineCoreClient 如何选择运行方式

`EngineCoreClient` 是外层 Engine 和内部 EngineCore 的桥。

注释说明：

```python
EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.

Subclasses:
* InprocClient: In process EngineCore (for V0-style LLMEngine use)
* SyncMPClient: ZMQ + background proc EngineCore (for LLM)
* AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

### 7.1 make_client()

入口：

```python
def make_client(
    multiprocess_mode: bool,
    asyncio_mode: bool,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
) -> "EngineCoreClient":
```

位置：`vllm/vllm/v1/engine/core_client.py:82` 到 `vllm/vllm/v1/engine/core_client.py:89`

选择逻辑：

```python
if asyncio_mode and not multiprocess_mode:
    raise NotImplementedError(...)

if multiprocess_mode and asyncio_mode:
    return EngineCoreClient.make_async_mp_client(...)

if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(...)

return InprocClient(...)
```

位置：`vllm/vllm/v1/engine/core_client.py:90` 到 `vllm/vllm/v1/engine/core_client.py:105`

因此：

```text
同步 + 非多进程：InprocClient
同步 + 多进程：SyncMPClient
异步 + 多进程：AsyncMPClient 系列
异步 + 非多进程：不支持
```

### 7.2 make_async_mp_client()

异步 client 选择逻辑：

```python
parallel_config = vllm_config.parallel_config
...
if parallel_config.data_parallel_size > 1:
    if parallel_config.data_parallel_external_lb:
        return DPAsyncMPClient(*client_args)
    return DPLBAsyncMPClient(*client_args)
return AsyncMPClient(*client_args)
```

位置：`vllm/vllm/v1/engine/core_client.py:107` 到 `vllm/vllm/v1/engine/core_client.py:132`

所以：

```text
无 DP：AsyncMPClient
DP + external load balancer：DPAsyncMPClient
DP + internal load balancer：DPLBAsyncMPClient
```

### 7.3 InprocClient：直接创建 EngineCore

```python
self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/vllm/v1/engine/core_client.py:286` 到 `vllm/vllm/v1/engine/core_client.py:287`

也就是说 inproc 初始化时，`EngineCore.__init__()` 就在当前进程执行。

### 7.4 MPClient：启动后台 EngineCoreProc

多进程 client 初始化时会创建 ZMQ socket，并启动 core engines。

如果不是外部传入 client_addresses，会执行：

```python
addresses = get_engine_zmq_addresses(vllm_config)
...
with launch_core_engines(
    vllm_config, executor_class, log_stats, addresses
) as (engine_manager, coordinator, addresses, tensor_queue):
    self.resources.coordinator = coordinator
    self.resources.engine_manager = engine_manager
```

位置：`vllm/vllm/v1/engine/core_client.py:551` 到 `vllm/vllm/v1/engine/core_client.py:578`

然后等待每个 EngineCore ready：

```python
while identities:
    ...
    identity, payload = sync_input_socket.recv_multipart()
    identities.remove(identity)
    self._apply_ready_response(payload)
```

位置：`vllm/vllm/v1/engine/core_client.py:615` 到 `vllm/vllm/v1/engine/core_client.py:633`

这说明多进程初始化包括：

```text
创建前端 ZMQ socket；
启动后台 EngineCoreProc；
等待每个 EngineCoreProc 完成模型 / KV cache 初始化；
接收 EngineCoreReadyResponse；
把初始化后的配置同步回 frontend vllm_config。
```

---

## 8. 后台 EngineCoreProc 如何启动

多进程启动由 `CoreEngineProcManager` 或 Ray actor manager 负责。

### 8.1 CoreEngineProcManager

`CoreEngineProcManager` 注释：

```python
class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """
```

位置：`vllm/vllm/v1/engine/utils.py:120` 到 `vllm/vllm/v1/engine/utils.py:124`

它会启动 `EngineCoreProc.run_engine_core`：

```python
context.Process(
    target=EngineCoreProc.run_engine_core,
    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
    kwargs=common_kwargs
    | {"dp_rank": global_index, "local_dp_rank": local_index},
)
```

位置：`vllm/vllm/v1/engine/utils.py:163` 到 `vllm/vllm/v1/engine/utils.py:171`

其中 `common_kwargs` 包含：

```python
common_kwargs = {
    "vllm_config": vllm_config,
    "local_client": local_client,
    "handshake_address": handshake_address,
    "executor_class": executor_class,
    "log_stats": log_stats,
    "tensor_queue": tensor_queue,
}
```

位置：`vllm/vllm/v1/engine/utils.py:141` 到 `vllm/vllm/v1/engine/utils.py:148`

所以后台进程拿到的关键初始化参数也是：

```text
vllm_config；
executor_class；
log_stats；
ZMQ handshake 地址；
DP rank 信息；
```

### 8.2 CoreEngineActorManager

Ray 场景下使用 `CoreEngineActorManager`。

注释：

```python
class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """
```

位置：`vllm/vllm/v1/engine/utils.py:370` 到 `vllm/vllm/v1/engine/utils.py:377`

它会根据 DP / MoE 选择 actor class：

```python
actor_class = (
    DPMoEEngineCoreActor
    if dp_size > 1 and vllm_config.model_config.is_moe
    else EngineCoreActor
)
```

位置：`vllm/vllm/v1/engine/utils.py:396` 到 `vllm/vllm/v1/engine/utils.py:401`

Ray actor 初始化时也传入：

```text
vllm_config；
executor_class；
log_stats；
local_client；
addresses；
```

对应位置：`vllm/vllm/v1/engine/utils.py:495` 到 `vllm/vllm/v1/engine/utils.py:503`

---

## 9. EngineCore 内部初始化链路

无论是 inproc 还是多进程，最终都会创建 `EngineCore` 或 `EngineCoreProc`。

`EngineCoreProc` 继承自 `EngineCore`：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/vllm/v1/engine/core.py:905` 到 `vllm/vllm/v1/engine/core.py:906`

所以核心内部初始化都在 `EngineCore.__init__()`。

### 9.1 EngineCore.__init__ 参数

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None = None,
    include_finished_set: bool = False,
):
```

位置：`vllm/vllm/v1/engine/core.py:99` 到 `vllm/vllm/v1/engine/core.py:106`

### 9.2 创建 model_executor

```python
self.model_executor = executor_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:123`

如果有 executor fail callback：

```python
self.model_executor.register_failure_callback(executor_fail_callback)
```

位置：`vllm/vllm/v1/engine/core.py:124` 到 `vllm/vllm/v1/engine/core.py:125`

### 9.3 初始化 KV cache

```python
kv_cache_config = self._initialize_kv_caches(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:132` 到 `vllm/vllm/v1/engine/core.py:133`

`_initialize_kv_caches()` 中会：

```text
注册 KV cache specs；
从 model_executor 获取 worker 侧 KV cache specs；
如果发现 non_causal KV cache spec，禁用 chunked prefill / prefix caching；
profile 可用 GPU memory，attention-free 模型则可用 KV memory 记为 0；
生成 worker / scheduler KV cache config；
必要时更新 max_model_len，并通过 collective_rpc("update_max_model_len") 同步给 workers；
更新 cache_config.num_gpu_blocks / block_size / kv_cache_size_tokens / kv_cache_max_concurrency；
调用 model_executor.initialize_from_config(kv_cache_configs) 初始化 worker KV cache 并 warmup model。
```

关键代码：

```python
kv_cache_specs = self.model_executor.get_kv_cache_specs()
if has_kv_cache:
    available_gpu_memory = self.model_executor.determine_available_memory()
else:
    available_gpu_memory = [0] * len(kv_cache_specs)
kv_cache_configs = get_kv_cache_configs(...)
if max_model_len_after != max_model_len_before:
    self.collective_rpc("update_max_model_len", args=(max_model_len_after,))
scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
self.model_executor.initialize_from_config(kv_cache_configs)
```

位置：`vllm/vllm/v1/engine/core.py:243` 到 `vllm/vllm/v1/engine/core.py:321`

### 9.4 创建 StructuredOutputManager

```python
self.structured_output_manager = StructuredOutputManager(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:134`

### 9.5 选择并创建 Scheduler

先从配置里取 Scheduler 类：

```python
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
```

位置：`vllm/vllm/v1/engine/core.py:136` 到 `vllm/vllm/v1/engine/core.py:137`

如果模型没有 KV cache 且配置开启了 chunked prefill，会先禁用 chunked prefill：

```python
if len(kv_cache_config.kv_cache_groups) == 0:
    if vllm_config.scheduler_config.enable_chunked_prefill:
        vllm_config.scheduler_config.enable_chunked_prefill = False
```

位置：`vllm/vllm/v1/engine/core.py:139` 到 `vllm/vllm/v1/engine/core.py:144`

然后解析 block size：

```python
scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
    kv_cache_config, vllm_config
)
```

位置：`vllm/vllm/v1/engine/core.py:146` 到 `vllm/vllm/v1/engine/core.py:148`

最后创建 Scheduler：

```python
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

位置：`vllm/vllm/v1/engine/core.py:150` 到 `vllm/vllm/v1/engine/core.py:158`

注意顺序：

```text
先初始化 executor / KV cache；
再创建 Scheduler。
```

原因是 Scheduler 初始化需要 `kv_cache_config`、block size 等信息。

### 9.6 KV connector handshake

如果 Scheduler 有 KV connector，会让 executor 初始化 output aggregator：

```python
if self.scheduler.connector is not None:
    self.model_executor.init_kv_output_aggregator(self.scheduler.connector)
```

位置：`vllm/vllm/v1/engine/core.py:163` 到 `vllm/vllm/v1/engine/core.py:164`

还会从 workers 收集 KV connector handshake metadata：

```python
xfer_handshake_metadata = (
    self.model_executor.get_kv_connector_handshake_metadata()
)
...
kv_connector.set_xfer_handshake_metadata_pp_aware(content)
```

位置：`vllm/vllm/v1/engine/core.py:171` 到 `vllm/vllm/v1/engine/core.py:190`

### 9.7 batch queue

```python
self.batch_queue_size = vllm_config.max_concurrent_batches
self.batch_queue = None
if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)
```

位置：`vllm/vllm/v1/engine/core.py:192` 到 `vllm/vllm/v1/engine/core.py:202`

### 9.8 request_block_hasher

如果启用 prefix caching 或 KV connector，会创建 block hasher：

```python
if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
    caching_hash_fn = get_hash_fn_by_name(...)
    init_none_hash(caching_hash_fn)
    self.request_block_hasher = get_request_block_hasher(
        hash_block_size, caching_hash_fn
    )
```

位置：`vllm/vllm/v1/engine/core.py:210` 到 `vllm/vllm/v1/engine/core.py:219`

这个 hasher 后续会在：

```text
Request.from_engine_core_request(request, self.request_block_hasher)
```

时传入，用于 prefix cache / block hash。

### 9.9 step_fn / async_scheduling / aborts_queue

request block hasher 初始化后，EngineCore 再决定 step function：

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
self.async_scheduling = vllm_config.scheduler_config.async_scheduling
self.aborts_queue = queue.Queue[list[str]]()
```

位置：`vllm/vllm/v1/engine/core.py:221` 到 `vllm/vllm/v1/engine/core.py:226`

最后还会冻结 startup heap、挂 GC debug callback，并启用 envs cache：

```python
freeze_gc_heap()
maybe_attach_gc_debug_callback()
enable_envs_cache()
```

位置：`vllm/vllm/v1/engine/core.py:230` 到 `vllm/vllm/v1/engine/core.py:237`

### 9.10 EngineCore 初始化总图

```text
EngineCore.__init__(vllm_config, executor_class, log_stats)
  → load plugins
  → self.model_executor = executor_class(vllm_config)
  → 可选 EEP scale-up before KV init
  → _initialize_kv_caches(vllm_config)
      → get_kv_cache_specs()
      → 处理 non_causal KV cache spec
      → determine_available_memory()
      → get_kv_cache_configs()
      → 可选 collective_rpc("update_max_model_len")
      → generate_scheduler_kv_cache_config()
      → model_executor.initialize_from_config()
  → StructuredOutputManager(vllm_config)
  → Scheduler = scheduler_config.get_scheduler_cls()
  → 无 KV cache 时可禁用 chunked prefill
  → resolve_kv_cache_block_sizes()
  → Scheduler(vllm_config, kv_cache_config, structured_output_manager, ...)
  → KV connector handshake
  → mm_receiver_cache
  → batch_queue
  → request_block_hasher
  → step_fn / async_scheduling / aborts_queue
  → freeze_gc_heap() / enable_envs_cache()
```

---

## 10. 配置如何从 EngineCore 回写到 frontend

多进程初始化中，后台 EngineCore 可能在 KV cache profiling 后调整配置。

比如：

```text
max_model_len 可能因为 KV cache auto-fitting 变小；
cache_config.num_gpu_blocks 会在 EngineCore 初始化后确定；
block_size 可能被 worker 对齐调整；
kv_cache_size_tokens / kv_cache_max_concurrency 也在初始化后才知道。
```

因此 EngineCoreProc 会在 ready 阶段返回 `EngineCoreReadyResponse`。

前端 MPClient 接收后调用：

```python
self._apply_ready_response(payload)
```

位置：`vllm/vllm/v1/engine/core_client.py:631` 到 `vllm/vllm/v1/engine/core_client.py:633`

### 10.1 _apply_ready_response()

它先 decode：

```python
response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
```

位置：`vllm/vllm/v1/engine/core_client.py:719` 到 `vllm/vllm/v1/engine/core_client.py:720`

然后同步 max_model_len：

```python
vllm_config.model_config.max_model_len = min(
    vllm_config.model_config.max_model_len, response.max_model_len
)
```

位置：`vllm/vllm/v1/engine/core_client.py:721` 到 `vllm/vllm/v1/engine/core_client.py:723`

累加 num_gpu_blocks：

```python
num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
num_gpu_blocks += response.num_gpu_blocks
vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
```

位置：`vllm/vllm/v1/engine/core_client.py:727` 到 `vllm/vllm/v1/engine/core_client.py:729`

同步 block_size 和 KV cache capacity：

```python
cache_config.block_size = response.block_size
cache_config.kv_cache_size_tokens = (
    getattr(cache_config, "kv_cache_size_tokens", None)
    if getattr(cache_config, "kv_cache_size_tokens", None) is not None
    else response.kv_cache_size_tokens
)
cache_config.kv_cache_max_concurrency = (
    getattr(cache_config, "kv_cache_max_concurrency", None)
    if getattr(cache_config, "kv_cache_max_concurrency", None) is not None
    else response.kv_cache_max_concurrency
)
```

位置：`vllm/vllm/v1/engine/core_client.py:731` 到 `vllm/vllm/v1/engine/core_client.py:745`

所以多进程初始化不是单向配置下发，还包括：

```text
EngineCore 初始化后，把 profiling / cache 结果回写给 frontend；
DP 场景下 num_gpu_blocks 会累加所有 engine 的值；
kv_cache_size_tokens / kv_cache_max_concurrency 保持 per-engine cache_config_info 语义，不跨 DP 累加。
```

---

## 11. 初始化时各组件职责边界

### 11.1 LLMEngine / AsyncLLM 负责

```text
创建外层 renderer；
创建 InputProcessor；
创建 OutputProcessor；
选择 executor_class；
创建 EngineCoreClient；
初始化 stats / tracing / profiler；
启动或准备 output handler；
管理 frontend 生命周期。
```

### 11.2 EngineCoreClient 负责

```text
决定访问 EngineCore 的方式；
InprocClient 直接创建 EngineCore；
MPClient 创建 ZMQ socket；
MPClient 启动 / 连接后台 EngineCoreProc；
等待 EngineCore ready；
同步 ready response 到 frontend config；
提供 add_request / get_output / utility 统一接口。
```

### 11.3 EngineCore 负责

```text
创建 model_executor；
初始化 KV cache；
创建 StructuredOutputManager；
创建 Scheduler；
配置 batch queue / step_fn；
准备 request block hashing；
管理内部执行闭环。
```

### 11.4 Executor 负责

```text
初始化 Worker / ModelRunner；
获取 KV cache specs；
profile 可用显存；
初始化 worker 侧 KV cache；
warmup / compile model；
后续执行 execute_model / sample_tokens。
```

### 11.5 Scheduler 负责

```text
基于 EngineCore 提供的 kv_cache_config 初始化调度状态；
管理 waiting / running 请求队列；
后续执行 schedule() / update_from_output()。
```

---

## 12. 容易混淆的点

### 12.1 executor_class 是在哪里创建出来的？

在外层 Engine 构造前通过：

```text
Executor.get_class(vllm_config)
```

选出来。

### 12.2 executor_class 是在哪里实例化的？

在 `EngineCore.__init__()` 中：

```text
self.model_executor = executor_class(vllm_config)
```

外层 Engine 只是传入 class，不直接创建 executor 实例。

### 12.3 LLMEngine.engine_core 是不是一定是 EngineCore？

不是。

```text
LLMEngine.engine_core 实际是 EngineCoreClient。
```

如果是 `InprocClient`，内部才有真实 `EngineCore`。

### 12.4 AsyncLLM 是否支持 in-process EngineCore？

`EngineCoreClient.make_client()` 里明确：

```text
asyncio_mode=True 且 multiprocess_mode=False 不支持。
```

位置：`vllm/vllm/v1/engine/core_client.py:90` 到 `vllm/vllm/v1/engine/core_client.py:95`

所以异步路径使用 `AsyncMPClient` 系列。

### 12.5 Scheduler 是外层 Engine 创建的吗？

不是。

Scheduler 在 `EngineCore.__init__()` 中创建。

外层 Engine 只创建 `EngineCoreClient`。

### 12.6 KV cache 是外层 Engine 初始化的吗？

不是。

KV cache 初始化发生在 `EngineCore._initialize_kv_caches()`，并通过 executor 调用 Worker 侧初始化。

### 12.7 为什么 Scheduler 要在 KV cache 初始化之后创建？

因为 Scheduler 初始化需要：

```text
kv_cache_config；
block_size；
hash_block_size；
```

这些必须等 KV cache profiling / config 生成后才知道。

### 12.8 多进程下前端配置会不会更新？

会。

后台 EngineCoreProc 初始化后通过 `EngineCoreReadyResponse` 把 max_model_len、num_gpu_blocks、block_size 等信息回传给 MPClient，MPClient 再更新 frontend 的 `vllm_config`。

---

## 13. 最关键的关系图

### 13.1 同步 in-process 初始化

```text
LLMEngine.from_engine_args()
  → EngineArgs.create_engine_config()
  → Executor.get_class(vllm_config)
  → LLMEngine.__init__(multiprocess_mode=False)
      → renderer
      → InputProcessor
      → OutputProcessor
      → EngineCoreClient.make_client(...)
          → InprocClient
              → EngineCore(vllm_config, executor_class)
                  → model_executor = executor_class(vllm_config)
                  → KV cache init
                  → Scheduler init
```

### 13.2 同步 multi-process 初始化

```text
LLMEngine.__init__(multiprocess_mode=True)
  → renderer
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient.make_client(...)
      → SyncMPClient
          → MPClient
              → create ZMQ sockets
              → launch_core_engines(...)
                  → EngineCoreProc.run_engine_core
                      → EngineCore.__init__()
              → wait EngineCoreReadyResponse
              → _apply_ready_response()
```

### 13.3 异步初始化

```text
AsyncLLM.from_engine_args()
  → AsyncEngineArgs.create_engine_config()
  → Executor.get_class(vllm_config)
  → AsyncLLM.__init__()
      → maybe_register_config_serialize_by_value()
      → renderer
      → InputProcessor
      → OutputProcessor
      → EngineCoreClient.make_async_mp_client(...)
          → AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
              → MPClient
                  → ZMQ sockets
                  → launch_core_engines(...)
                  → wait ready response
      → output_handler 准备 / 启动
```

### 13.4 EngineCore 内部初始化

```text
EngineCore.__init__()
  → executor_class(vllm_config)
  → model_executor
  → model_executor.get_kv_cache_specs()
  → 处理 non_causal / attention-free KV cache 情况
  → model_executor.determine_available_memory()
  → get_kv_cache_configs()
  → 可选同步 auto-fit 后的 max_model_len 到 workers
  → generate_scheduler_kv_cache_config()
  → model_executor.initialize_from_config()
  → StructuredOutputManager
  → scheduler_config.get_scheduler_cls()
  → resolve_kv_cache_block_sizes()
  → Scheduler(...)
  → batch_queue / request_block_hasher / step_fn
  → aborts_queue / env cache
```

---

## 14. 从“回答问题”的角度总结

如果问：

```text
Engine 初始化时如何配置 EngineCore？
```

可以回答：

```text
外层 Engine 初始化时首先拿到 vllm_config，并通过 Executor.get_class(vllm_config) 选出 executor_class。
同步路径由 LLMEngine 初始化，异步路径由 AsyncLLM 初始化。它们都会创建 renderer、InputProcessor 和 OutputProcessor，分别负责输入转换和输出转换。

随后外层 Engine 会创建 EngineCoreClient。LLMEngine 通过 EngineCoreClient.make_client() 根据 multiprocess_mode 选择 InprocClient 或 SyncMPClient；AsyncLLM 通过 EngineCoreClient.make_async_mp_client() 选择 AsyncMPClient 或 DP 版本的 async client。

如果是 InprocClient，会在当前进程直接创建 EngineCore；如果是 MPClient，会启动后台 EngineCoreProc，并通过 ZMQ 与它通信。后台 EngineCoreProc 初始化完成后会回传 EngineCoreReadyResponse，把 max_model_len、num_gpu_blocks、block_size 等实际初始化结果同步回前端 vllm_config，其中 num_gpu_blocks 在 DP 场景下累加，KV cache capacity 字段保持 per-engine 语义。

真正的 EngineCore 内部初始化发生在 EngineCore.__init__()。它会用 executor_class(vllm_config) 创建 model_executor，profile 并初始化 KV cache；KV cache 初始化可能禁用不适用的 chunked prefill / prefix caching，也可能把 auto-fit 后的 max_model_len 同步给 workers。之后 EngineCore 再根据 scheduler_config.get_scheduler_cls() 创建 Scheduler。Scheduler 需要 KV cache 配置，所以必须在 KV cache 初始化之后创建。
```

最小心智模型：

```text
外层 Engine：准备配置、输入输出处理器和 EngineCoreClient。
EngineCoreClient：决定同进程还是多进程访问 EngineCore。
EngineCore：真正创建 executor、KV cache 和 Scheduler。
Executor：真正管理 Worker / ModelRunner 执行环境。
```
