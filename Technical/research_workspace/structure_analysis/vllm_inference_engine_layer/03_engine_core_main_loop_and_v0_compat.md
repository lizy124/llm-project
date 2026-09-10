# 03 EngineCore 主循环与 V0 兼容

本篇详细梳理 V1 `EngineCore` 的初始化、KV cache 初始化、主循环 `step()`、batch queue，以及 V0 兼容层在当前架构中的定位。

## 1. EngineCore 的定位

`EngineCore` 定义在 `code/vllm/vllm/v1/engine/core.py:96`。

它是 V1 推理引擎的“内核”，负责把调度器、执行器、KV cache、structured output、KV connector、多模态缓存等组件连接起来。

可以把它理解成：

```text
EngineCore = Scheduler + Executor + KV cache 初始化 + engine loop
```

它不直接执行模型 forward；实际 forward 由 executor 分发到 worker，再由 model runner 执行。

## 2. EngineCore 初始化流程

`EngineCore.__init__` 从 `code/vllm/vllm/v1/engine/core.py:99` 开始。

主要步骤：

```text
1. 加载插件
2. 保存 VllmConfig
3. 创建 model_executor = executor_class(vllm_config)
4. 初始化 KV cache
5. 创建 StructuredOutputManager
6. 创建 Scheduler
7. 初始化 KV connector handshake metadata
8. 设置 pipeline parallel batch queue
9. 准备 prefix caching block hasher
10. 冻结 GC heap / 启用 env cache
```

### 关键代码点

| 行号 | 作用 |
|---|---|
| `code/vllm/vllm/v1/engine/core.py:123` | `self.model_executor = executor_class(vllm_config)` |
| `code/vllm/vllm/v1/engine/core.py:133` | `_initialize_kv_caches(vllm_config)` |
| `code/vllm/vllm/v1/engine/core.py:134` | 创建 `StructuredOutputManager` |
| `code/vllm/vllm/v1/engine/core.py:137` | 从配置取 scheduler 类 |
| `code/vllm/vllm/v1/engine/core.py:150` | 创建 `Scheduler` |
| `code/vllm/vllm/v1/engine/core.py:163` | scheduler connector 与 executor KV output aggregator 连接 |
| `code/vllm/vllm/v1/engine/core.py:174` | 获取 worker 侧 KV connector handshake metadata |
| `code/vllm/vllm/v1/engine/core.py:196` | batch queue size |
| `code/vllm/vllm/v1/engine/core.py:210` | prefix caching / connector 时构造 request block hasher |

## 3. KV cache 初始化：EngineCore 的启动关键

`_initialize_kv_caches()` 定义在 `code/vllm/vllm/v1/engine/core.py:240`。

这是 vLLM 启动时最重要的阶段之一。它要决定 GPU 上能放多少 KV cache block。

### 初始化过程

```text
1. register_all_kvcache_specs(vllm_config)
2. model_executor.get_kv_cache_specs()
3. 检查是否存在 non-causal attention
   - 如果有，关闭 chunked prefill 和 prefix caching
4. model_executor.determine_available_memory()
   - worker/profile run 估算可用于 KV cache 的显存
5. get_kv_cache_configs(...)
   - 根据 spec 和可用显存生成 worker KV cache config
6. generate_scheduler_kv_cache_config(...)
   - 生成 scheduler 使用的 KVCacheConfig
7. 更新 vllm_config.cache_config.num_gpu_blocks / block_size / capacity
8. model_executor.initialize_from_config(kv_cache_configs)
   - 通知 worker 真正分配 KV cache tensor 并 warmup
```

### 为什么 KV cache 初始化要放在 EngineCore

因为 scheduler 必须知道：

- 总共有多少 block；
- 每个 block 存多少 token；
- 有哪些 KV cache group；
- 是否启用 prefix caching；
- 是否存在特殊 cache 类型，比如 sliding window、Mamba、encoder-only attention；
- worker 实际可用显存是多少。

因此，KV cache 初始化连接了三方：

```text
模型层提供 KVCacheSpec
  ↓
worker profile 可用显存
  ↓
scheduler 获得可调度的 KVCacheConfig
```

## 4. add_request：请求进入调度器

`EngineCore.add_request()` 定义在 `code/vllm/vllm/v1/engine/core.py:372`。

职责：

1. 校验 request id 必须是字符串；
2. 校验 pooling task 是否被当前模型支持；
3. 如果请求带 `kv_transfer_params` 但 scheduler 没有 connector，则打印 warning；
4. 调用 `self.scheduler.add_request(request)`；
5. 如果请求要求立即 abort，则马上走 abort。

主线很短：

```text
EngineCore.add_request(request)
  -> Scheduler.add_request(request)
```

这说明 request 是否能执行、何时执行、占用哪些 block，都不在 `EngineCore.add_request()` 决定，而是在 scheduler 的 `schedule()` 决定。

## 5. step：EngineCore 的核心主循环

`EngineCore.step()` 定义在 `code/vllm/vllm/v1/engine/core.py:479`。

它是 V1 推理引擎单步执行的核心。

### step 主流程

```text
1. 如果 scheduler 没有请求，返回空输出
2. scheduler.schedule()
   - 生成 SchedulerOutput
3. model_executor.execute_model(scheduler_output, non_block=True)
   - 异步提交 worker 执行
4. scheduler.get_grammar_bitmask(scheduler_output)
   - 结构化输出约束
5. future.result()
   - 等待模型执行返回
6. 如果 model_output is None
   - 说明 execute_model 只完成 forward，采样要单独执行
   - 调 model_executor.sample_tokens(grammar_output)
7. 处理执行期间发生的 abort
8. scheduler.update_from_output(scheduler_output, model_output)
9. 返回 EngineCoreOutputs 和 model_executed 标记
```

### 为什么 execute_model 可能返回 None

`GPUModelRunner.execute_model()` 在常见生成路径里会：

1. 准备 batch；
2. 做模型 forward；
3. 计算 logits；
4. 把中间状态保存到 `execute_model_state`；
5. 返回 `None`；
6. 后续 `sample_tokens()` 再真正采样。

这样可以让结构化输出 grammar bitmask 在 forward 后、sample 前插入。

对应代码：

- `EngineCore.step()` 中 `model_output is None` 时调用 `sample_tokens()`：`code/vllm/vllm/v1/engine/core.py:498`。
- `GPUModelRunner.execute_model()` 返回 `None`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`。
- `GPUModelRunner.sample_tokens()` 执行采样：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`。

## 6. post_step：spec decode draft tokens 更新

`EngineCore.post_step()` 在 `code/vllm/vllm/v1/engine/core.py:510`。

它主要处理 speculative decoding 的 draft token 更新。

逻辑：

```text
如果开启 spec decode，且不是 async scheduling，且本步执行了模型：
  draft_token_ids = model_executor.take_draft_token_ids()
  scheduler.update_draft_token_ids(draft_token_ids)
```

这说明 draft token 是 worker/model runner 侧产生，然后回传给 scheduler，用于下一轮调度。

## 7. batch queue：pipeline parallel 的异步批队列

`step_with_batch_queue()` 从 `code/vllm/vllm/v1/engine/core.py:519` 开始。

当 `max_concurrent_batches > 1` 时，EngineCore 会启用 batch queue。它的目的主要是 pipeline parallel 场景下减少 pipeline bubble。

### 普通 step 与 batch queue step 的区别

普通 `step()`：

```text
schedule -> execute -> wait -> sample -> update scheduler -> output
```

batch queue：

```text
1. 如果 batch queue 未满，优先 schedule 新 batch 并提交执行
2. 如果没有新 batch 或 queue 满，再等待最早的 batch 完成
3. 完成后 update scheduler
```

也就是说 batch queue 允许多个 batch in-flight。

### batch queue 的影响

- 能提升 pipeline parallel 利用率；
- 但 scheduler/worker/KV connector 必须处理 overlapping batches；
- KV block free 可能需要 defer，因为一个 batch 还在写 KV，另一个 batch 可能已经想复用 block。

这也是 scheduler 中 `defer_block_free` 存在的原因之一。

## 8. EngineCoreProc：后台进程中的 EngineCore

`EngineCoreProc` 定义在 `code/vllm/vllm/v1/engine/core.py:894`。

它继承 `EngineCore`，负责在后台进程里运行 engine loop，并处理 ZMQ/socket 输入输出。

核心职责：

- 进程启动和 handshake；
- input socket/output socket；
- busy loop；
- utility request；
- shutdown；
- pause/resume；
- client request 转发到 EngineCore 方法。

关键方法：

| 方法 | 作用 |
|---|---|
| `run_engine_core()` | 后台进程启动入口 |
| `startup_handshake()` | 启动握手 |
| `run_busy_loop()` | 主循环 |
| `_process_input_queue()` | 处理输入请求 |
| `_process_engine_step()` | 推进一步 engine step |
| `_handle_client_request()` | 处理 client 请求 |
| `process_input_sockets()` | ZMQ 输入处理 |
| `process_output_sockets()` | ZMQ 输出处理 |

## 9. DPEngineCoreProc：Data Parallel 特化

`DPEngineCoreProc` 定义在 `code/vllm/vllm/v1/engine/core.py:1741`。

它针对 data parallel 做了额外处理：

- 多 DP engine 的同步；
- request wave；
- global unfinished request 判断；
- prefill throttling；
- 分布式重初始化。

对于单机单卡/普通多卡理解，可以先跳过；调试 DP/elastic EP 时再看。

## 10. V0 兼容层定位

旧引擎相关文件主要在：

```text
vllm/engine/llm_engine.py
vllm/engine/async_llm_engine.py
vllm/engine/arg_utils.py
```

当前理解上可以这样看：

```text
V0 API/旧接口语义
  ↓
EngineCoreClient / Executor
  ↓
V1 EngineCore 主链路
```

V0 层重点是兼容用户已有调用方式，而不是当前最核心的调度执行逻辑。阅读时建议先读 V1，再回头看 V0 如何包一层。

## 11. EngineCore 的关键输入输出

### 输入

EngineCore 主要接收：

- `EngineCoreRequest`：新增请求；
- abort request ids；
- utility RPC，比如 sleep/wake/reset/profile；
- scheduler pause/resume；
- LoRA add/remove/pin；
- distributed reconfigure。

### 输出

EngineCore 主要输出：

- `EngineCoreOutputs`：给 OutputProcessor；
- scheduler stats；
- finished request ids；
- error/abort outputs。

## 12. 本层阅读重点

阅读 `EngineCore` 时，建议重点抓三条线：

### 线 1：启动线

```text
__init__
  -> executor_class(vllm_config)
  -> _initialize_kv_caches
  -> Scheduler(...)
```

### 线 2：执行线

```text
step
  -> scheduler.schedule
  -> executor.execute_model
  -> executor.sample_tokens
  -> scheduler.update_from_output
```

### 线 3：进程通信线

```text
EngineCoreClient
  -> AsyncMPClient/SyncMPClient
  -> EngineCoreProc
  -> process_input_sockets/process_output_sockets
```

只要这三条线清楚，EngineCore 的大部分代码就能归位。
