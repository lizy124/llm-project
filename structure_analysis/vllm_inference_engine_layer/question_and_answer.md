# vLLM 推理引擎层技术点问答整理

本文基于本目录已有的 vLLM 推理引擎层结构分析文档，整理适合技术考察、代码阅读复盘、面试提问的技术点，并给出可直接回答的参考答案。

## 一、整体架构与主链路

### 1. vLLM 的“推理引擎层”具体指什么？

vLLM 的推理引擎层不是单个目录，而是一条跨目录的运行链路。它从 API 入口开始，经由前台异步引擎、EngineCore、Scheduler、KV Cache 管理、Executor/Worker、GPUModelRunner、模型层 Attention，最终落到底层 CUDA/C++ kernel。

核心链路可以概括为：

```text
entrypoints/openai
  -> v1/engine/async_llm.py
  -> v1/engine/core_client.py
  -> v1/engine/core.py
  -> v1/core/sched/scheduler.py
  -> v1/executor/*
  -> v1/worker/gpu_worker.py
  -> v1/worker/gpu_model_runner.py
  -> model_executor/*
  -> model_executor/layers/attention/* + v1/attention/*
  -> csrc/*
```

一句话：API 层负责接请求，AsyncLLM 做前台异步封装，EngineCore 驱动推理主循环，Scheduler 决定每一步算哪些 token 和如何使用 KV block，Executor/Worker 把调度结果发到设备，GPUModelRunner 组织 batch 和模型 forward，Attention backend 调用底层 kernel 完成高性能推理。

### 2. vLLM 推理引擎层可以分成哪几层？

可以分成 7 层：

| 层级 | 主要路径 | 核心职责 |
|---|---|---|
| API/入口层 | `vllm/entrypoints/` | OpenAI API、CLI、server、请求解析 |
| 前台引擎层 | `vllm/v1/engine/async_llm.py` | 异步接口、输入处理、输出收集、与 EngineCore 通信 |
| 引擎核心层 | `vllm/v1/engine/core.py` | 创建 executor、初始化 KV cache、驱动 scheduler/executor 主循环 |
| 调度层 | `vllm/v1/core/sched/` | request 状态机、token budget、prefill/decode、preemption、spec decode |
| KV 管理层 | `vllm/v1/core/kv_cache_*`, `block_pool.py` | KV block 分配、复用、回收、prefix cache、connector |
| 执行层 | `vllm/v1/executor/`, `vllm/v1/worker/` | 单进程/多进程/Ray 执行，worker 设备初始化与模型调用 |
| 模型与 kernel 层 | `model_executor/`, `v1/attention/`, `csrc/` | 模型 forward、attention backend、paged attention、CUDA/C++ kernel |

### 3. 一次 OpenAI Chat/Completion 请求在 vLLM 内部如何流转？

一次请求大体流程如下：

```text
1. OpenAI API server 接收 HTTP 请求
2. AsyncLLM.generate()/add_request()
3. InputProcessor 把用户输入转成 EngineCoreRequest
4. EngineCoreClient 把请求送到 EngineCore
5. EngineCore.add_request() 加入 Scheduler
6. EngineCore.step() 驱动一次调度与执行
7. Scheduler.schedule() 选出本 step 要执行的 request/token/block
8. Executor.execute_model() 把 SchedulerOutput 发给 worker
9. Worker.execute_model() 调 GPUModelRunner
10. GPUModelRunner 准备 input_ids / positions / block table / attention metadata
11. model_executor 中具体模型 forward
12. Attention 层读取 forward context 和 KV cache，调用 backend
13. csrc CUDA/C++ kernel 执行 paged attention/cache ops/all-reduce/MoE 等
14. GPUModelRunner 计算 logits 并 sample
15. Scheduler.update_from_output() 更新 request 状态
16. OutputProcessor 转成 RequestOutput
17. API server 流式或一次性返回给客户端
```

### 4. vLLM 推理引擎中最核心的几个类有哪些？

核心类包括：

| 类 | 文件 | 作用 |
|---|---|---|
| `AsyncLLM` | `vllm/v1/engine/async_llm.py` | 前台异步引擎封装，面向 API 层 |
| `InputProcessor` | `vllm/v1/engine/input_processor.py` | 把用户输入变成 EngineCoreRequest |
| `OutputProcessor` | `vllm/v1/engine/output_processor.py` | 把 EngineCoreOutputs 转成用户输出 |
| `EngineCoreClient` | `vllm/v1/engine/core_client.py` | 前台与 EngineCore 的通信抽象 |
| `EngineCore` | `vllm/v1/engine/core.py` | V1 推理内核主循环 |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py` | request/token/KV 资源调度器 |
| `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` | KV block 分配、复用和释放 |
| `BlockPool` | `vllm/v1/core/block_pool.py` | 底层 block 池 |
| `Executor` | `vllm/v1/executor/abstract.py` | 执行后端抽象 |
| `Worker` | `vllm/v1/worker/gpu_worker.py` | GPU worker，负责设备与模型 runner |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py` | GPU 上组织 batch 和模型 forward 的核心类 |
| `AttentionBackend` | `vllm/v1/attention/backend.py` | Attention backend 能力抽象 |
| `Attention` | `vllm/model_executor/layers/attention/attention.py` | 模型层 attention 模块 |

### 5. V0 和 V1 推理引擎的关系是什么？

当前仓库中，V1 是主推理链路。V0 主要保留旧接口和旧语义，例如 `LLMEngine.add_request()`、`LLMEngine.step()`、`AsyncLLMEngine` 等。

可以这样理解：

```text
V0 API/旧接口语义
  -> EngineCoreClient / Executor
  -> V1 EngineCore 主链路
```

所以阅读时建议先看 V1 主链路，再回头看 V0 如何做兼容包装。

### 6. 为什么说 vLLM 的推理执行不是简单的 PyTorch forward？

因为 vLLM 在模型 forward 之前会做大量系统级准备：调度 request、分配 KV block、构造 block table、生成 slot mapping、构造 attention metadata、处理 prefix cache/spec decode/KV connector/CUDA graph/pipeline parallel 等。

模型里的 Attention 也不是普通 PyTorch attention，而是通过 forward context 获取当前 batch 的 KV cache、slot mapping、metadata，然后调用 backend 和底层 kernel 执行 paged attention。

## 二、请求生命周期与前台异步引擎

### 7. API/入口层的职责是什么？

入口层通常在 `vllm/entrypoints/` 和 `vllm/entrypoints/openai/`，职责包括：

1. 启动 FastAPI server 或 CLI；
2. 解析 OpenAI-compatible 请求；
3. 将 HTTP schema 转换成 vLLM 内部参数；
4. 构造 `AsyncEngineArgs` / `VllmConfig`；
5. 创建 `AsyncLLM`；
6. 调用 `AsyncLLM.generate()`、`encode()` 等接口；
7. 把结果包装成 OpenAI-compatible response。

入口层不负责实际调度和模型执行。

### 8. AsyncLLM 的职责是什么？

`AsyncLLM` 是前台异步引擎，面向 API 层提供异步生成接口。它负责：

1. 保存 `VllmConfig`；
2. 创建 tokenizer/renderer 相关组件；
3. 创建 `InputProcessor`；
4. 创建 `OutputProcessor`；
5. 通过 `EngineCoreClient` 创建与 EngineCore 的通信客户端；
6. 启动 output handler；
7. 提供 `generate()`、`add_request()`、`abort()`、`encode()`、`sleep()`、`wake_up()` 等接口。

### 9. InputProcessor 做了什么？

`InputProcessor` 把外部用户输入转换成 engine core 可调度的 `EngineCoreRequest`。

典型处理包括：

- 校验 sampling params / pooling params；
- 校验 LoRA；
- 处理 prompt、tokens、multi-modal 输入；
- 处理 tokenizer 相关逻辑；
- 校验 prompt 长度；
- 分配或确认 request id；
- 生成 `EngineCoreRequest`。

### 10. EngineCoreClient 为什么需要存在？

因为 `AsyncLLM` 通常运行在 API/server 进程中，而 `EngineCore` 可能运行在后台进程或多个进程中。`EngineCoreClient` 抽象了前台与 EngineCore 的通信方式，使上层不需要关心是 in-process、multi-process、sync、async 还是 data parallel async。

常见 client 包括：

- `InprocClient`；
- `MPClient`；
- `SyncMPClient`；
- `AsyncMPClient`；
- `DPAsyncMPClient`。

### 11. OutputProcessor 的职责是什么？

`OutputProcessor` 把 EngineCore 内部输出转换成用户可见的输出。

它维护 request state 和 collector，负责：

1. 注册 request state；
2. 处理 `EngineCoreOutputs`；
3. 更新 streaming request state；
4. 判断请求完成并清理；
5. 处理 abort；
6. 传播错误；
7. 更新统计信息；
8. 生成 `RequestOutput` 或 `PoolingRequestOutput`。

### 12. ModelRunnerOutput、EngineCoreOutputs、RequestOutput 有什么区别？

三者位于不同抽象层：

1. `ModelRunnerOutput`：worker/model runner 产生的底层输出，包含 sampled tokens、logprobs、KV connector output 等。
2. `EngineCoreOutputs`：engine core/scheduler 整理后的内部输出，按 request 返回。
3. `RequestOutput`：API/user 可见输出，是最终给用户或 OpenAI-compatible API 的结果。

不能把 `GPUModelRunner` 的输出直接理解成最终用户输出。

### 13. 一个 generation 请求的状态如何演化？

典型状态链路：

```text
客户端请求
  -> AsyncLLM.generate()
  -> InputProcessor.process_inputs()
  -> EngineCoreRequest
  -> OutputProcessor.add_request() 注册 collector
  -> EngineCoreClient.add_request_async()
  -> EngineCore.preprocess_add_request()
  -> Request 对象
  -> Scheduler.add_request()
  -> WAITING
  -> Scheduler.schedule()
  -> RUNNING
  -> Worker/GPUModelRunner 执行
  -> ModelRunnerOutput
  -> Scheduler.update_from_output()
  -> EngineCoreOutputs
  -> OutputProcessor.process_outputs()
  -> RequestOutput
  -> 客户端收到 token / final output
```

## 三、EngineCore 主循环与启动流程

### 14. EngineCore 的定位是什么？

`EngineCore` 是 V1 推理引擎的内核，可以理解为：

```text
EngineCore = Scheduler + Executor + KV cache 初始化 + engine loop
```

它负责连接 scheduler、executor、KV cache、structured output、KV connector、多模态缓存等组件，但不直接做模型 forward。实际 forward 由 executor 分发到 worker，再由 GPUModelRunner 执行。

### 15. EngineCore 初始化阶段做了哪些事情？

`EngineCore.__init__` 主要步骤：

1. 加载插件；
2. 保存 `VllmConfig`；
3. 创建 `model_executor = executor_class(vllm_config)`；
4. 初始化 KV cache；
5. 创建 `StructuredOutputManager`；
6. 创建 `Scheduler`；
7. 初始化 KV connector handshake metadata；
8. 设置 pipeline parallel batch queue；
9. 准备 prefix caching block hasher；
10. 冻结 GC heap / 启用 env cache。

### 16. 为什么 KV cache 初始化要放在 EngineCore 中？

因为 scheduler 必须知道可调度的 KV 资源，而这些资源依赖三方信息：

```text
模型层提供 KVCacheSpec
  -> worker profile 可用显存
  -> scheduler 获得 KVCacheConfig
```

EngineCore 需要协调模型层、worker 和 scheduler：模型层声明每层需要什么 KV cache；worker 通过 profile 判断显存能放多少 KV block；scheduler 根据最终 KVCacheConfig 做资源调度。

### 17. EngineCore._initialize_kv_caches 的流程是什么？

核心流程：

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

### 18. EngineCore.add_request 做了什么？

`EngineCore.add_request()` 的主线很短：

```text
EngineCore.add_request(request)
  -> Scheduler.add_request(request)
```

它主要做 request id 校验、pooling task 支持校验、KV transfer 参数检查，然后把请求交给 scheduler。请求是否能执行、何时执行、占用哪些 block，都是 scheduler 在 `schedule()` 中决定的。

### 19. EngineCore.step 的核心流程是什么？

`EngineCore.step()` 是 V1 推理引擎单步执行的核心：

```text
1. 如果 scheduler 没有请求，返回空输出
2. scheduler.schedule() 生成 SchedulerOutput
3. model_executor.execute_model(scheduler_output, non_block=True)
4. scheduler.get_grammar_bitmask(scheduler_output)
5. future.result() 等待模型执行返回
6. 如果 model_output is None，调用 model_executor.sample_tokens(grammar_output)
7. 处理执行期间发生的 abort
8. scheduler.update_from_output(scheduler_output, model_output)
9. 返回 EngineCoreOutputs 和 model_executed 标记
```

### 20. 为什么 execute_model 可能返回 None？

在常见生成路径中，`GPUModelRunner.execute_model()` 只完成模型 forward 和 logits 计算，然后把中间状态保存到 `execute_model_state`，并返回 `None`。

随后 `EngineCore.step()` 会调用：

```text
Executor.sample_tokens(grammar_output)
```

这样可以让结构化输出 grammar bitmask 插在 forward 之后、sample 之前。

### 21. EngineCore.post_step 的作用是什么？

`EngineCore.post_step()` 主要处理 speculative decoding 的 draft token 更新。

逻辑是：如果开启 spec decode，且不是 async scheduling，并且本步执行了模型，则从 executor/model runner 获取 draft token ids，然后调用 scheduler 更新到对应 request，供下一轮调度使用。

### 22. batch queue 是什么？为什么 pipeline parallel 需要它？

当 `max_concurrent_batches > 1` 时，EngineCore 会启用 batch queue。普通 step 是：

```text
schedule -> execute -> wait -> sample -> update scheduler -> output
```

batch queue 则允许多个 batch in-flight：

```text
1. 如果 batch queue 未满，优先 schedule 新 batch 并提交执行
2. 如果没有新 batch 或 queue 满，再等待最早的 batch 完成
3. 完成后 update scheduler
```

这样可以减少 pipeline parallel 的 pipeline bubble，提高设备利用率。但它也带来复杂性：overlapping batches 下，KV block 释放不能过早，否则可能复用还在被 kernel 写入的 block。

### 23. EngineCoreProc 是什么？

`EngineCoreProc` 是运行在后台进程中的 EngineCore。它负责：

- 进程启动和 handshake；
- input socket/output socket；
- busy loop；
- utility request；
- shutdown；
- pause/resume；
- client request 转发到 EngineCore 方法。

它是多进程模式下前台 `AsyncLLM` 与后台 EngineCore 的执行载体。

### 24. DPEngineCoreProc 解决什么问题？

`DPEngineCoreProc` 是 Data Parallel 特化的 EngineCoreProc，额外处理：

- 多 DP engine 同步；
- request wave；
- global unfinished request 判断；
- prefill throttling；
- 分布式重初始化。

普通单机单卡或普通多卡理解主链路时，可以先跳过这一层。

## 四、Scheduler 调度机制

### 25. Scheduler 的核心职责是什么？

Scheduler 不执行模型，它负责每个 engine step 的资源调度，回答：

1. 这一步执行哪些 request；
2. 每个 request 执行多少 token；
3. 这些 token 需要哪些 KV cache block；
4. 哪些 request 从 waiting 进入 running；
5. 哪些 running request 继续 decode/prefill；
6. KV cache 不够时是否 preempt；
7. prefix cache 命中多少 token；
8. spec decode draft token 如何参与调度；
9. encoder/multimodal 输入是否有预算；
10. 请求完成后如何释放资源。

一句话：Scheduler 是 request/token/KV resource 的统一调度器。

### 26. Scheduler 内部有哪些关键状态？

关键状态包括：

| 状态 | 作用 |
|---|---|
| `self.requests` | `req_id -> Request` 全量请求表 |
| `self.waiting` | 等待调度的新请求队列 |
| `self.skipped_waiting` | 因资源/依赖暂时跳过的 waiting 请求 |
| `self.running` | 正在运行或已经进入 batch 管理的请求 |
| `self.finished_req_ids` | 已完成、需要通知 worker 清理的请求 |
| `self.max_num_running_reqs` | 最大并发序列数 |
| `self.max_num_scheduled_tokens` | 单步 token budget |
| `self.kv_cache_manager` | KV block 分配、复用、释放 |
| `self.encoder_cache_manager` | encoder/multimodal cache 管理 |
| `self.connector` | KV transfer/offload connector |
| `self.ec_connector` | encoder cache transfer connector |
| `self.structured_output_manager` | 结构化输出约束 |

### 27. 为什么说 V1 Scheduler 没有固定的 prefill/decode 阶段？

V1 Scheduler 的关键思想是：

> There is no fixed decoding phase nor prefill phase.

它不是先判断当前是 prefill 还是 decode，而是对每个 request 维护：

```text
num_computed_tokens
num_tokens_with_spec
```

每一步都尝试让：

```text
num_computed_tokens 追上 num_tokens_with_spec
```

这样可以统一支持普通 prefill、chunked prefill、decode、prefix caching、speculative decoding、jump decoding、encoder-decoder、多模态输入等场景。

### 28. Scheduler.schedule 的主流程是什么？

高层流程：

```text
1. current_step += 1
2. 初始化本步输出容器
3. 设置 token_budget
4. kv_cache_manager.new_step_starts()
5. 先调度 RUNNING 请求
6. 再调度 WAITING 请求
7. 构造 SchedulerOutput
8. 更新内部状态
9. 返回 SchedulerOutput 给 EngineCore
```

### 29. 为什么 schedule() 先调度 running，再调度 waiting？

因为 running 请求通常已经处于 decode 或部分 prefill 状态，需要持续推进。如果大量 waiting/prefill 请求优先进入，decode 请求可能被饿死，导致延迟明显升高。

所以 vLLM 优先让已经进入 running 的请求继续前进，再在 token budget 和 KV block 允许的情况下接纳新的 waiting 请求。

### 30. token budget 是什么？

token budget 是单个 engine step 内允许调度的最大 token 数：

```text
token_budget = self.max_num_scheduled_tokens
```

它通常来自：

```text
scheduler_config.max_num_scheduled_tokens
或 scheduler_config.max_num_batched_tokens
```

每调度一个 request，会扣减：

```text
token_budget -= num_new_tokens
```

当 token budget 用完，本 step 不再调度更多 token。

### 31. running request 调度时会检查哪些条件？

Scheduler 对 running request 会检查：

1. 是否已经达到 max tokens；
2. PP/async scheduling 的 decode cadence；
3. DP prefill balancing 是否需要延迟 prefill chunk；
4. 本步需要多少 `num_new_tokens`；
5. `long_prefill_token_threshold`；
6. token budget；
7. `max_model_len`；
8. encoder input budget；
9. Mamba hybrid 模型是否要求 block aligned；
10. KV block 是否足够；
11. 是否需要 preempt 其他请求。

### 32. waiting request 被调度时会检查哪些 admission 条件？

waiting request 进入 running 前，会检查：

- running 请求数是否达到 `max_num_running_reqs`；
- request 是否处于 blocked 状态；
- LoRA 数量是否超过 `max_loras`；
- prefix cache 命中情况；
- KV connector 是否能提供外部 KV；
- 是否需要异步 load remote KV；
- encoder/multimodal budget；
- full sequence 是否必须一次 fit；
- KV block 是否足够。

### 33. prefix caching 在 Scheduler 中如何参与调度？

对于 waiting request，尤其是 `request.num_computed_tokens == 0` 时，Scheduler 会调用：

```text
kv_cache_manager.get_computed_blocks(request)
```

如果 prompt 的前缀已经在本地 KV cache 中，Scheduler 会把命中的 block 计入 computed tokens，只调度未命中的 token。

即使 prompt 全部命中，通常仍需要重算最后一个 token，用于获得 logits。

### 34. KV 不够时 Scheduler 如何处理？

如果 `kv_cache_manager.allocate_slots()` 返回 `None`，说明 KV block 不足。

Scheduler 会尝试 preempt 请求：

- 如果是 priority policy，选择优先级最低的请求；
- 否则通常 pop running 队尾请求；
- 被 preempt 的请求释放部分状态，后续重新进入等待或恢复。

preemption 可以维持高负载下的吞吐和资源利用，但也可能导致部分 KV 重新计算。

### 35. speculative decoding 在调度中如何体现？

Scheduler 会维护 spec decode 相关状态，如：

- `num_spec_tokens`；
- `num_lookahead_tokens`；
- `use_eagle`；
- `dynamic_sd_lookup`。

调度时：

1. request 可能携带 `spec_token_ids`；
2. Scheduler 把 spec tokens 计入 `num_tokens_with_spec`；
3. `allocate_slots()` 额外分配 lookahead tokens；
4. worker/model runner 执行后产生 draft tokens；
5. EngineCore 取回 draft token；
6. Scheduler 更新 request 的 draft token ids，供下一步使用。

### 36. encoder/multimodal 输入如何参与 Scheduler？

Scheduler 维护 encoder budget 和 `encoder_cache_manager`。

当 request 有 encoder inputs 时，Scheduler 调用 `_try_schedule_encoder_inputs()`：

- 判断本步是否有 encoder compute budget；
- 判断 encoder cache 是否够；
- 为 encoder inputs 分配 cache；
- 生成 `scheduled_encoder_inputs`，传给 worker。

这让文本生成、encoder-decoder、多模态输入可以统一进入同一个 step 调度。

### 37. SchedulerOutput 包含什么？

`SchedulerOutput` 是 Scheduler 和 Worker 之间最重要的数据契约，通常包含：

- 本步新请求数据；
- 本步 cached request 数据；
- 每个 request 的 scheduled token 数；
- 新分配的 block ids；
- common prefix blocks；
- scheduled encoder inputs；
- scheduled spec decode tokens；
- finished request ids；
- structured output / grammar 相关数据；
- KV connector metadata；
- scheduler stats。

### 38. Scheduler.update_from_output 做什么？

模型执行返回后，EngineCore 调用：

```text
scheduler.update_from_output(scheduler_output, model_output)
```

它负责：

- 根据 sampled token 更新 request output token；
- 检查 stop condition；
- 更新 request status；
- 处理 streaming update；
- 处理 spec decode accept/reject；
- 处理 KV connector output；
- 释放完成请求的 encoder/KV 资源；
- 生成 `EngineCoreOutputs`。

### 39. Scheduler 中资源释放为什么是正确性关键点？

请求完成、取消或 preempt 后，需要释放 KV blocks 和 encoder cache。如果释放过早，可能导致 still in-flight 的 kernel 仍在写入这些 block，而新 batch 已经复用它们，产生数据破坏。

在 overlapping batches、KV connector、async scheduling、pipeline parallel 场景下，资源释放必须特别谨慎，因此会有 deferred free 等机制。

### 40. 如何一句话概括 Scheduler？

Scheduler 不是简单的 prefill/decode 排队器，而是一个逐步推进 `num_computed_tokens` 的资源调度器：它在每个 step 内用有限 token budget 和 KV block，把 running 和 waiting 请求推进一小段，并把决定编码成 `SchedulerOutput` 交给 worker。

## 五、KV Cache、Block 与 Prefix Caching

### 41. 为什么 KV Cache 是 vLLM 的核心？

自回归 LLM 推理中，每生成一个 token 都需要历史 token 的 K/V。直接重复计算历史 attention 成本很高，因此需要缓存每层 attention 的 K/V。

vLLM 的核心创新之一是把 KV cache 拆成固定大小的 block/page，用类似虚拟内存分页的方式管理，从而支持：

- 多请求共享显存池；
- 动态分配和回收 KV block；
- prefix caching；
- preemption 后恢复；
- chunked prefill；
- paged attention kernel；
- KV transfer/offload/disaggregated prefill。

### 42. KV cache 相关的关键配置有哪些？

常见配置在 `vllm/config/cache.py` 中：

| 参数 | 作用 |
|---|---|
| `block_size` | 物理 KV cache block 存放多少 token |
| `hash_block_size` | prefix cache hash 的 token 粒度 |
| `cache_dtype` | KV cache dtype |
| `enable_prefix_caching` | 是否启用 prefix caching |
| `prefix_caching_hash_algo` | prefix hash 算法 |
| `kv_cache_memory_bytes` | 手动指定 KV cache 显存大小 |
| `gpu_memory_utilization` | 自动 profile 时显存利用率 |
| `num_gpu_blocks_override` | 覆盖 GPU block 数 |
| `kv_offloading_size` | KV offload 相关大小 |
| `mamba_cache_mode` | Mamba/hybrid cache 模式 |

### 43. Worker.determine_available_memory 为什么启动时要跑 profile？

因为 vLLM 需要知道 GPU 上剩余多少显存可以分给 KV cache。

profile 过程会：

1. 如果用户指定 `kv_cache_memory_bytes`，仍然跑一次 profile/warmup，但最终返回用户指定大小；
2. 否则执行 dummy forward；
3. 统计非 KV cache 显存、权重显存、activation peak、CUDA graph 预估显存；
4. 计算剩余可用于 KV cache 的显存。

这就是 vLLM 启动时需要 profile 的原因。

### 44. KVCacheManager 的职责是什么？

`KVCacheManager` 是 scheduler 侧的 KV block 管理入口，主要负责：

| 方法 | 作用 |
|---|---|
| `get_computed_blocks()` | 查找 prefix cache 命中的 blocks |
| `allocate_slots()` | 为请求新 token 分配 KV slots/blocks |
| `free()` | 释放请求 KV blocks |
| `remove_skipped_blocks()` | 移除跳过/不再需要的 blocks |
| `pop_blocks_for_free()` | 取出待释放 blocks |
| `evict_blocks()` | 主动驱逐 blocks |
| `reset_prefix_cache()` | 重置 prefix cache |
| `get_num_common_prefix_blocks()` | 获取共同前缀 block 数，用于 cascade attention |
| `cache_blocks()` | 将 computed blocks 写入 prefix cache 索引 |
| `take_new_block_ids()` | 取本步新分配 block ids |
| `new_step_starts()` | 新 step 开始时重置临时状态 |

### 45. KVCacheBlocks 为什么要按 group 组织？

`KVCacheBlocks` 的结构是：

```text
blocks: tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 对应 KV cache group，内层是该 group 的 block 序列。

这样设计是为了支持 hybrid KV cache。比如普通 attention、sliding window、Mamba state 等可能有不同 cache group，它们不一定共享完全相同的 block 语义。

### 46. allocate_slots 的核心逻辑是什么？

`allocate_slots()` 是 KV 管理最重要的方法。它处理的 token 布局可以理解为：

```text
| comp | new_comp | ext_comp | new | lookahead |

comp      = 已经计算过的 token
new_comp  = 本地 prefix cache 新命中的 token
ext_comp  = connector 外部命中的 token
new       = 本步要计算的新 token
lookahead = spec decode 额外预留 token
```

分三阶段：

1. 释放 `comp` 中不再需要的 blocks，并检查 free blocks 是否足够；
2. 处理 prefix token：本地命中、外部命中、sliding window 外 block 释放；
3. 为本步要计算的新 token 和 lookahead token 分配新 blocks。

成功返回新分配的 `KVCacheBlocks`，失败返回 `None`，Scheduler 会考虑 preemption 或跳过请求。

### 47. BlockPool 是什么？

`BlockPool` 是底层 block 池，维护 free、cached、evictable blocks。

主要职责包括：

- 按 hash 查找 cached block；
- 把完整 blocks 放入 prefix cache 索引；
- 分配新 blocks；
- 必要时驱逐 cached block；
- touch blocks 更新使用状态，避免被驱逐；
- 释放 blocks；
- 查询 free block 数和使用率；
- 产生 KV cache events。

### 48. Prefix Caching 的基本流程是什么？

Prefix caching 的目标是多个请求共享相同前缀时复用已经计算好的 KV block。

流程：

```text
请求进入 InputProcessor/EngineCore
  -> 为 request 计算 block hashes
  -> Scheduler 调度 waiting request
  -> KVCacheManager.get_computed_blocks(request)
  -> BlockPool 按 hash 查找 cached block
  -> 命中部分变成 computed tokens
  -> Scheduler 只调度未命中 token
```

### 49. Prefix Caching 有哪些注意点？

需要注意：

1. prefix cache 只能复用完整 block；
2. 即使全部 prompt 命中，通常也要重算最后一个 token 来获得 logits；
3. non-causal attention 会禁用 prefix caching；
4. sliding window 会影响哪些 blocks 仍需保留；
5. prompt logprobs、pooling 等场景可能跳过读取 prefix cache；
6. KV connector 可以提供外部命中 token，与本地 prefix cache 组合。

### 50. block_size 和 hash_block_size 的区别是什么？

- `block_size`：物理 KV cache block 大小，决定一个 KV block 存多少 token；
- `hash_block_size`：prefix cache hash 的 token 粒度。

多数情况下二者一致，但 hash 粒度可以更细，用于更细粒度的 prefix cache 匹配。

### 51. Block table 和 slot mapping 起什么作用？

Scheduler 侧只知道每个 request 分配了哪些 block ids；worker/kernel 侧需要知道每个 token 应该读写 KV cache tensor 的哪个位置。

这中间靠 block table 和 slot mapping：

```text
本 step 中第 i 个 token
  -> 属于哪个 request
  -> 是该 request 的第几个 token
  -> 对应哪个 KV block
  -> block 内 offset 是多少
  -> 最终 KV cache tensor 写入/读取位置
```

attention kernel 依靠 block table/slot mapping 找到 KV cache 页。

### 52. KV cache tensor 在 worker 侧如何分配？

worker 侧分配入口：

```text
Worker.initialize_from_config()
  -> GPUModelRunner.initialize_kv_cache()
```

`GPUModelRunner` 内部会：

- `_allocate_kv_cache_tensors()` 分配 tensor；
- `_reshape_kv_cache_tensors()` 按 backend/layout reshape；
- `_update_hybrid_attention_mamba_layout()` 处理 hybrid attention/mamba 布局；
- `initialize_attn_backend()` 初始化 attention backend；
- `initialize_metadata_builders()` 初始化 attention metadata builder。

### 53. KV Connector / Transfer / Offload 解决什么问题？

KV connector 主要用于：

- disaggregated prefill/decode；
- remote KV load/save；
- KV offload；
- 外部 KV cache 系统对接。

Scheduler 侧 connector 可以询问外部 KV 命中 token 数，如果外部 KV 可用，Scheduler 可以把这些 token 当成 external computed tokens，不必本地重新计算。

### 54. KV cache events 有什么用？

KV cache events 记录 block stored/free/evict 等事件，主要用于：

- 观测 cache 行为；
- 外部系统同步；
- 调试 prefix cache 命中和驱逐。

### 55. KV cache 的完整生命周期是什么？

完整生命周期：

```text
启动阶段：
  model layer 声明 KVCacheSpec
  worker profile 可用显存
  EngineCore 生成 KVCacheConfig
  worker 分配 KV cache tensor

请求进入：
  request 计算 block hash
  scheduler 放入 waiting

调度阶段：
  get_computed_blocks 查 prefix cache
  allocate_slots 分配新 blocks
  SchedulerOutput 带 block ids 发给 worker

执行阶段：
  GPUModelRunner 构造 block table / slot mapping
  Attention 写入新 K/V，读取历史 K/V

完成阶段：
  scheduler.update_from_output 更新状态
  完成/取消时 free request blocks
  完整 blocks 可能进入 prefix cache 索引
  空闲或可驱逐 blocks 回到 block pool
```

### 56. 如何一句话概括 KV Cache 管理层？

KV Cache 管理层把请求历史 token 的 K/V 抽象成可分配、可复用、可驱逐、可转移的 block；Scheduler 决定 block 的逻辑分配，GPUModelRunner 把 block 转成 kernel 可用的 slot mapping，Attention kernel 根据这些映射读写实际 KV cache tensor。

## 六、Executor、Worker 与 GPUModelRunner

### 57. Executor 的定位是什么？

Executor 位于 Scheduler 和 Worker 之间，负责把 `SchedulerOutput` 分发到一个或多个设备执行。

它可以是单设备 executor，也可以是多设备分布式 executor。

主链路：

```text
EngineCore
  -> SchedulerOutput
  -> Executor
  -> collective_rpc("execute_model")
  -> Worker
  -> GPUModelRunner
  -> model_executor 模型 forward
```

### 58. Executor 根据什么选择具体类型？

`Executor.get_class()` 根据 `parallel_config.distributed_executor_backend` 选择具体 executor：

| backend | executor |
|---|---|
| `uni` | `UniProcExecutor` |
| `mp` | `MultiprocExecutor` |
| `ray` | `RayDistributedExecutor` 或 `RayExecutorV2` |
| `external_launcher` | `ExecutorWithExternalLauncher` |
| 自定义字符串 | 通过 qualname resolve |
| 自定义 class | 直接使用 |

### 59. Executor 的主要接口有哪些？

主要接口包括：

| 方法 | 作用 |
|---|---|
| `_init_executor()` | 子类初始化具体 worker/backend |
| `initialize_from_config()` | 初始化 worker KV cache 并 warmup/compile |
| `determine_available_memory()` | 向 worker 查询可用于 KV cache 的显存 |
| `get_kv_cache_specs()` | 向 worker 查询模型需要的 KV cache spec |
| `collective_rpc()` | 对所有 worker 执行 RPC |
| `execute_model()` | 让 worker 执行一个 SchedulerOutput |
| `sample_tokens()` | 单独执行采样阶段 |
| `take_draft_token_ids()` | 取 spec decode draft tokens |
| `shutdown()` | 关闭 worker |
| `sleep()/wake_up()` | 显存睡眠/唤醒 |
| `add_lora()/remove_lora()` | 动态 LoRA 管理 |

### 60. collective_rpc 的意义是什么？

`collective_rpc()` 是 executor 到 worker 的控制面 RPC。

在单进程 executor 中，它可能只是函数调用；在多进程/Ray 中，它会跨进程或跨节点调用所有 worker。

它适合控制消息，如 initialize、warmup、execute、sample、shutdown 等。真正大规模张量数据通常不通过这个 RPC 传输，而由分布式通信机制处理。

### 61. GPU Worker 的职责是什么？

GPU Worker 是设备侧执行入口，负责：

- 初始化设备；
- 初始化 torch distributed / model parallel；
- 初始化 KV transfer / EC transfer；
- 创建 GPUModelRunner；
- 加载模型；
- profile 可用显存；
- 分配 KV cache；
- warmup / CUDA graph capture；
- 执行模型；
- sleep/wake；
- weight transfer；
- LoRA 管理。

### 62. Worker 如何选择 GPUModelRunner V1/V2？

Worker 会根据配置选择不同 model runner：

```text
if self.use_v2_model_runner:
  from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
else:
  from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
```

当前文档重点分析 V1 路径，同时要注意新路径 `vllm/v1/worker/gpu/model_runner.py` 也存在。

### 63. Worker 的关键阶段有哪些？

关键阶段包括：

1. `load_model()`：调用 model runner 加载模型权重；
2. `determine_available_memory()`：通过 dummy run/profile 估算 KV cache 可用显存；
3. `initialize_from_config()`：更新 cache config、初始化 KV transfer、分配 KV cache；
4. `compile_or_warm_up_model()`：编译 batch size、kernel warmup、CUDA graph capture、处理 LoRA 临时状态。

### 64. GPUModelRunner 的定位是什么？

`GPUModelRunner` 是 worker 侧最核心的执行类。它不是简单调用模型，而是负责 GPU batch 执行生命周期：

```text
SchedulerOutput
  -> 更新 persistent batch state
  -> 准备 input ids / positions / mrope / xdrope
  -> 准备 block table / slot mapping
  -> 构造 attention metadata
  -> 准备 multimodal encoder inputs
  -> set_forward_context
  -> model forward
  -> compute logits
  -> sample tokens
  -> 更新 request/batch 状态
  -> ModelRunnerOutput
```

### 65. GPUModelRunner.execute_model 的主流程是什么？

高层流程：

```text
1. 检查上一次 execute_model 是否已经 sample_tokens
2. 处理 ngram_gpu spec decode 的 scheduler_output copy
3. KV connector 处理 preemptions
4. num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
5. preprocess 阶段：
   - _update_states(scheduler_output)
   - 如有 EC transfer producer，执行 encoder 并返回
   - 如果没有 token，返回 empty output 或 connector no-forward
   - _prepare_inputs(...)
   - _determine_batch_execution_and_padding(...)
   - maybe_create_ubatch_slices(...)
   - Mamba preprocess
   - _get_slot_mappings(...)
   - _build_attention_metadata(...)
   - _preprocess(...)
6. set_forward_context(...)
7. _model_forward(...)
8. postprocess：
   - 取 hidden states
   - pipeline parallel 中可能返回 intermediate tensors
   - pooling 模型走 _pool
   - generation 模型 compute_logits
9. 保存 execute_model_state
10. 返回 None，等待 sample_tokens()
```

### 66. GPUModelRunner.sample_tokens 的流程是什么？

流程：

```text
1. 如果 execute_model_state 为空：
   - 说明只有 KV connector output 或 PP 非末级 rank
   - 返回对应 ModelRunnerOutput
2. 取出 execute_model_state
3. 如果有 grammar_output，apply_grammar_bitmask
4. _sample(logits, spec_decode_metadata)
5. _update_states_after_model_execute(sampled_token_ids, scheduler_output)
6. async scheduling + PP 时广播 sampled tokens
7. 清理 draft token 临时状态
8. 生成 ModelRunnerOutput
```

### 67. GPUModelRunner 为什么要维护 persistent batch state？

因为 request 在多个 engine step 中持续推进，不能每一步都从零构造所有状态。GPUModelRunner 需要维护 persistent batch state，用于记录 request 状态、token buffer、block table、slot mapping、spec decode metadata、mamba state 等。

SchedulerOutput 只是本 step 的增量调度结果，GPUModelRunner 需要把它合并到已有 batch state，才能构造模型 forward 需要的完整输入。

### 68. Attention metadata 与 slot mapping 在 GPUModelRunner 中如何生成？

执行 attention 前必须准备：

- 每个 token 对应的 KV slot；
- 每个 request 的 sequence length；
- block table；
- common prefix blocks；
- query length；
- max query len；
- cascade attention metadata；
- spec decode metadata。

关键方法：

- `_get_slot_mappings()`：生成 token 到 KV slot 的映射；
- `_build_attention_metadata()`：构造 attention backend 需要的 metadata；
- `initialize_attn_backend()`：初始化 attention backend；
- `initialize_metadata_builders()`：初始化 metadata builder。

### 69. CUDA Graph / padding / ubatching 解决什么问题？

GPUModelRunner 会根据 batch 形状决定执行方式，考虑：

- 是否使用 CUDA graph；
- batch 是否需要 padding 到 capture size；
- 是否使用 ubatching；
- DP 下 token 数协调；
- 是否是 uniform decode；
- 是否要 skip compiled path。

这些都是 vLLM 性能优化的重要部分，可以减少 kernel launch overhead、提高 shape 复用和设备利用率。

### 70. Pipeline Parallel 下 GPUModelRunner 有什么特殊处理？

如果开启 PP：

- 非最后一个 PP rank 可能返回 `IntermediateTensors`；
- 最后一个 PP rank 才 compute logits / sample；
- 某些配置下 logits 会 broadcast 给所有 rank；
- async scheduling 下 sampled token ids 也需要跨 PP rank 通信。

所以 PP 场景下不能假设每个 worker 都直接产生最终 sampled token。

### 71. Executor、Worker、GPUModelRunner 如何一句话区分？

Executor 是 EngineCore 到 worker 的执行抽象，Worker 是设备侧生命周期管理器，GPUModelRunner 是真正把 SchedulerOutput 变成模型 forward、logits 和 sampled tokens 的核心执行器。

## 七、Attention、ModelExecutor 与 CUDA/C++ Kernel

### 72. model_executor 的职责是什么？

`vllm/model_executor/` 是模型执行层，负责：

- 加载模型权重；
- 定义模型结构；
- 定义 attention、MLP、MoE、rotary embedding、norm、activation 等层；
- 支持量化；
- 支持 LoRA；
- 支持模型特定 forward；
- 暴露 KVCacheSpec；
- 与 attention backend 衔接。

### 73. 模型加载链路是什么？

粗略链路：

```text
Worker.load_model()
  -> GPUModelRunner.load_model()
  -> model_loader 加载权重
  -> model_executor/models 中具体模型类实例化
```

### 74. GPUModelRunner 如何调用模型 forward？

在 `execute_model()` 中，核心 forward 调用类似：

```text
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

调用前会先 `set_forward_context(...)`，把 attention metadata、slot mapping、vllm_config、batch descriptor、ubatch slices 等放入 forward context。

### 75. forward context 为什么重要？

模型里的 Attention 层并不会通过显式参数拿到所有 metadata，而是通过 forward context 获取当前 batch 的 attention metadata、slot mapping、KV cache 等。

因此，GPUModelRunner 和 Attention 层之间存在一个隐式契约：

```text
GPUModelRunner 构造 metadata/slot_mapping
  -> set_forward_context
  -> Attention.forward 通过 get_attention_context 读取
```

### 76. Attention 模块的职责是什么？

模型层 Attention 接收 query/key/value，负责：

- KV cache 读写；
- 调用 attention backend；
- 处理 quantization scales；
- 与 forward context 绑定；
- 生成 KVCacheSpec。

它不是普通 PyTorch attention，而是 vLLM 高性能推理链路中的核心适配层。

### 77. Attention.forward 依赖哪些关键数据？

`Attention.forward()` 依赖 forward context 中的数据：

- `attn_metadata`；
- `slot_mapping`；
- 当前层的 KV cache；
- backend-specific metadata；
- 是否需要更新 KV cache；
- quant scale；
- attention type。

所以理解 Attention 必须同时看 GPUModelRunner 的 `_build_attention_metadata()`、`_get_slot_mappings()`、`set_forward_context()` 和 Attention 的 `forward()`。

### 78. get_attention_context 做什么？

`get_attention_context()` 从 forward context 中取出：

- 当前 layer 的 KV cache；
- attention metadata；
- attention layer 对象；
- slot mapping。

这说明 attention layer 和 model runner 的数据衔接主要依赖 forward context。

### 79. AttentionBackend 抽象了什么？

`AttentionBackend` 定义每种 attention backend 需要声明的能力，包括：

- 支持哪些 kernel block size；
- backend 名称；
- backend 实现类；
- metadata builder 类；
- KV cache tensor shape；
- KV cache block 维度；
- cache stride/layout；
- 是否支持某 head size；
- 是否支持 dtype / KV cache dtype；
- 是否支持 block size；
- 是否支持 decoder/prefix/encoder attention type；
- 配置校验；
- 需要的 KV cache layout。

### 80. AttentionMetadata 和 Builder 的作用是什么？

`AttentionMetadata`、`CommonAttentionMetadata`、`AttentionMetadataBuilder` 用于把 batch 信息组织成 backend 可用格式。

metadata 通常包含：

- query lengths；
- sequence lengths；
- block tables；
- slot mapping；
- common prefix length；
- max query len；
- max seq len；
- cascade attention 信息；
- spec decode 信息。

GPUModelRunner 会调用 backend 的 builder 构造 metadata。

### 81. AttentionImpl 的职责是什么？

`AttentionImplBase` / `AttentionImpl` 是具体 backend 实现 attention 的基类。

关键能力包括：

- `forward()` 执行 attention；
- `do_rope_and_kv_cache_update()` 融合 RoPE 和 KV cache update；
- `fused_output_quant_supported()` 判断是否支持 fused output quant；
- `fused_rope_kvcache_supported()` 判断是否支持 fused rope + kv cache。

### 82. KV cache update 与 attention forward 有哪两种模式？

不同 backend 对 KV cache 更新方式不同：

1. `forward_includes_kv_cache_update = True`：attention forward 内部完成 KV cache update。
2. `forward_includes_kv_cache_update = False`：KV cache update 和 attention forward 分离。

GPUModelRunner 会根据 backend 能力决定 slot mapping padding 和执行策略。

### 83. csrc 层负责什么？

`csrc/` 是 vLLM 的底层高性能实现，包含：

- paged attention CUDA kernel；
- cache write/read kernel；
- custom all-reduce；
- CPU attention/kernel；
- CUTLASS 扩展；
- quantized GEMM；
- MoE kernel；
- layernorm/activation/pos encoding 等。

### 84. Paged Attention 的核心思想是什么？

普通 attention 可能假设 KV cache 连续存储。vLLM 的 paged attention 通过 block table 间接寻址：

```text
request token position
  -> logical block index
  -> physical block id
  -> KV cache tensor 中的 page
  -> block 内 offset
```

这样多个 request 可以共享同一大 KV cache 池，且每个 request 的 KV blocks 不需要物理连续。

### 85. SchedulerOutput 到 Attention Kernel 的数据如何变换？

链路如下：

```text
SchedulerOutput
  包含：req ids、num scheduled tokens、new block ids、common prefix blocks
  -> GPUModelRunner._update_states
  -> 更新 input_batch 中 request 状态
  -> GPUModelRunner._get_slot_mappings
  -> token -> KV slot
  -> GPUModelRunner._build_attention_metadata
  -> 构造 backend metadata
  -> set_forward_context
  -> 把 metadata/slot_mapping 放入 forward context
  -> 模型 forward
  -> Attention.forward
  -> get_attention_context
  -> 取出当前层 KV cache / metadata / slot_mapping
  -> backend impl forward
  -> custom op / CUDA kernel
```

### 86. 分布式通信和 Attention 有哪些关系？

模型 forward/attention 中可能涉及：

- tensor parallel all-reduce；
- sequence parallel；
- pipeline parallel intermediate tensors；
- data parallel batch coordination；
- decode context parallel / prefill context parallel；
- expert parallel MoE；
- custom all-reduce kernel。

相关逻辑通常分布在 `vllm/distributed/` 和 `csrc/custom_all_reduce*`。

### 87. Quantization 如何影响 Attention？

KV cache dtype 和量化会影响 attention 执行方式，包括：

- fp16/bf16/fp8 KV cache；
- per-head quant scales；
- weight-only quant；
- fused output quant；
- fused rope + kv cache update。

相关代码通常在 `model_executor/layers/quantization/`、`model_executor/layers/attention/attention.py`、`v1/attention/backend.py` 和 `csrc/*fp8*`。

### 88. 如何一句话概括 Attention 到 kernel 链路？

vLLM 的模型执行不是简单的 PyTorch forward：GPUModelRunner 先把 SchedulerOutput 转换成 slot mapping 和 attention metadata，通过 forward context 传给模型层 Attention，Attention backend 再根据 block table 间接访问 KV cache，最终调用 csrc/自定义 kernel 完成高性能 paged attention。

## 八、性能优化与高级机制

### 89. chunked prefill 的意义是什么？

chunked prefill 把长 prompt 的 prefill 拆成多个 chunk，让 prefill 不会一次占满整个 token budget，从而减少 decode 请求被长 prompt 阻塞的情况。

在 V1 Scheduler 中，由于没有固定 prefill/decode 阶段，chunked prefill 本质上就是每个 step 只推进 request 的一部分 `num_new_tokens`。

### 90. long_prefill_token_threshold 有什么作用？

`long_prefill_token_threshold` 用于限制单个长 prefill request 在一个 step 内调度过多 token，避免长 prompt 抢占太多 token budget，影响其他请求尤其是 decode 请求的 latency。

### 91. max_num_scheduled_tokens 和 max_num_running_reqs 分别控制什么？

- `max_num_scheduled_tokens`：控制单个 step 总共可以调度多少 token，影响吞吐、延迟和 batch 大小。
- `max_num_running_reqs`：控制同时处于 running 状态的请求数量，影响并发序列数和 KV block 压力。

两者共同决定 scheduler 的并发和资源使用形态。

### 92. preemption 对性能有什么影响？

preemption 可以在 KV block 不足时释放低优先级请求的资源，让更关键或更靠前的请求继续执行，从而维持系统吞吐。

但它也有代价：被 preempt 的请求后续可能需要重新计算部分 KV，增加总计算量。如果频繁 preempt，通常说明 KV block 不足、batch 配置过激、prefill chunk 太大或并发过高。

### 93. CUDA graph 在 vLLM 中为什么重要？

CUDA graph 可以减少重复 kernel launch overhead，尤其适合 decode 阶段形状较稳定的场景。

但 CUDA graph 需要 batch shape 可复用，因此 GPUModelRunner 可能需要 padding 到 capture size，并根据 batch 形状决定是否走 captured graph、compiled path 或 eager path。

### 94. ubatching 解决什么问题？

ubatching 把一个较大的 batch 拆成多个 micro/ubatch 执行，用于适配 CUDA graph、显存、DP token 协调或 backend 限制。

它可以帮助在性能和资源约束之间取得平衡，但也会增加 batch 切分和 metadata 构造复杂度。

### 95. spec decode 为什么需要 scheduler、model runner 和 EngineCore 协同？

spec decode 涉及 draft tokens 的生成、验证和接受/拒绝。Scheduler 需要把 spec tokens 纳入 `num_tokens_with_spec` 和 lookahead slots；model runner 需要执行并产生 draft token；EngineCore 需要在 step 后取回 draft token 并更新 scheduler。

所以它不是单点逻辑，而是贯穿调度、执行、采样和状态回写的机制。

### 96. structured output / grammar bitmask 插在什么位置？

structured output 的 grammar bitmask 插在模型 forward/logits 之后、sample 之前。

这也是 `GPUModelRunner.execute_model()` 常返回 `None` 的原因：它先计算 logits 并保存状态，然后 EngineCore 获取 grammar bitmask，再调用 `sample_tokens(grammar_output)` 应用约束并采样。

### 97. KV offload / disaggregated prefill 对主链路有什么影响？

它让 KV cache 不一定只来自本地 GPU。Scheduler 在调度时可能通过 connector 查询外部 KV 命中 token 数，把这些 token 当作 external computed tokens；worker 侧需要初始化 KV transfer 并处理 remote KV load/save。

这会影响 admission、allocate_slots、SchedulerOutput metadata、worker 执行和完成后的 connector output。

### 98. pipeline parallel 为什么会带来资源释放复杂度？

PP 下多个 stage 可能同时处理不同 batch。batch queue 允许多个 batch in-flight，这意味着一个 batch 可能还在写 KV，另一个 batch 已经完成并触发资源释放。

如果 scheduler 过早 free block，可能导致后续 batch 复用仍在使用的 block。因此需要 deferred free 和更严格的 in-flight batch 管理。

## 九、分布式、多卡与并行

### 99. vLLM 中 executor backend 和并行方式是什么关系？

executor backend 决定 worker 如何被管理和调用，例如单进程、多进程、Ray、external launcher。并行方式则包括 TP、PP、DP、EP 等，决定模型和请求如何在设备间切分与通信。

两者共同决定执行拓扑：Executor 管 worker，parallel config 决定 worker 之间如何通信和分担计算。

### 100. MultiprocExecutor 和 RayExecutor 的主要区别是什么？

`MultiprocExecutor` 通常用于本机多进程 worker 管理；`RayExecutor` / `RayDistributedExecutor` 用于基于 Ray 的分布式 worker 管理，适合跨节点或 Ray 集群环境。

二者都通过 executor 抽象暴露统一接口，例如 `collective_rpc()`、`execute_model()`、`sample_tokens()`。

### 101. Tensor Parallel 对模型 forward 有什么影响？

Tensor Parallel 会把模型权重或中间计算拆到多个 rank 上，forward 中需要 all-reduce / all-gather 等通信，Attention、MLP、logits 等都可能涉及 TP 通信。

因此调试 TP 问题时要看 `parallel_state.py`、device communicators、自定义 all-reduce、模型层 TP 切分逻辑。

### 102. Pipeline Parallel 对输出有什么影响？

PP 下非最后一个 PP rank 通常只返回 intermediate tensors，不直接计算 logits 和 sample；最后一个 rank 才产生 logits 和 sampled tokens。

所以在 PP 场景下，不能假设每个 rank 都有最终 `ModelRunnerOutput`。

### 103. Data Parallel 对调度有什么影响？

DP 可能需要多 DP engine 同步请求 wave、全局判断是否还有 unfinished request、做 prefill throttling、协调 batch token 数等。

因此 DP 下 EngineCoreProc 有特化版本 `DPEngineCoreProc`，GPUModelRunner 也会考虑 DP 下 token 数协调。

### 104. Expert Parallel / MoE 可能涉及哪些组件？

MoE/Expert Parallel 会涉及 routed experts、expert parallel 通信、fused MoE kernel、routed experts capturer、load balancing 等。

相关代码可能分布在 model executor 的 MoE 层、worker warmup/capture、distributed 通信和 csrc MoE kernel 中。

## 十、常见调试问题与定位

### 105. 请求没有返回或卡住，应该看哪里？

优先看：

```text
AsyncLLM.generate/add_request
OutputProcessor.process_outputs
EngineCoreClient.get_output_async
EngineCoreProc.run_busy_loop
EngineCore.step
Scheduler.has_requests
```

重点判断：

- request 是否成功进入 scheduler；
- output handler 是否运行；
- EngineCore 是否还活着；
- scheduler 是否认为有请求；
- worker execute 是否卡住。

### 106. 请求排队太久或吞吐低，应该看哪里？

优先看：

```text
Scheduler.schedule
max_num_scheduled_tokens
max_num_running_reqs
long_prefill_token_threshold
chunked prefill
preemption
KV block free 数量
```

重点判断：

- token budget 是否太小；
- running 请求是否占满；
- waiting 是否被 LoRA/encoder/KV connector 条件跳过；
- KV block 是否不足导致频繁 preempt；
- chunked prefill 是否影响 decode latency。

### 107. OOM 或 KV cache 太小，应该看哪里？

优先看：

```text
config/cache.py
EngineCore._initialize_kv_caches
gpu_worker.determine_available_memory
GPUModelRunner.profile_run
get_kv_cache_configs
cache_config.num_gpu_blocks
```

重点判断：

- `gpu_memory_utilization` 是否太高/太低；
- 是否手动指定 `kv_cache_memory_bytes`；
- CUDA graph 是否额外占用显存；
- max model len 是否过大；
- quantization/cache dtype 是否符合预期。

### 108. Prefix cache 不生效，应该看哪里？

优先看：

```text
cache_config.enable_prefix_caching
EngineCore.request_block_hasher
Request.block_hashes
KVCacheManager.get_computed_blocks
BlockPool.get_cached_block
KVCacheManager.cache_blocks
```

重点判断：

- 是否启用 prefix caching；
- non-causal attention 是否禁用了 prefix caching；
- request 是否 `skip_reading_prefix_cache`；
- block hash 是否一致；
- block 是否完整并已 cache；
- hash_block_size 与 block_size 是否符合预期。

### 109. 请求被频繁 preempt，应该看哪里？

优先看：

```text
Scheduler.schedule
KVCacheManager.allocate_slots
BlockPool.get_num_free_blocks
Scheduler._preempt_request
Request.num_preemptions
```

重点判断：

- KV blocks 是否不足；
- running 请求数是否过多；
- max_num_batched_tokens 是否过大；
- prefill chunk 太大；
- priority policy 是否导致低优先级请求被抢占。

### 110. 输出 token 错误或采样异常，应该看哪里？

优先看：

```text
GPUModelRunner.execute_model
GPUModelRunner.sample_tokens
apply_grammar_bitmask
_sample
Scheduler.update_from_output
OutputProcessor.process_outputs
```

重点判断：

- logits 是否正常；
- grammar bitmask 是否错误屏蔽 token；
- sampling params 是否正确；
- spec decode accept/reject 是否异常；
- output processor 是否正确拼接 streaming 输出。

### 111. Attention/kernel 报错，应该看哪里？

优先看：

```text
GPUModelRunner._get_slot_mappings
GPUModelRunner._build_attention_metadata
set_forward_context
Attention.forward
AttentionBackend.validate_configuration
csrc attention/cache kernels
```

重点判断：

- block table 是否正确；
- slot mapping 是否越界；
- KV cache shape/layout 是否和 backend 要求一致；
- head size/block size/dtype 是否被 backend 支持；
- CUDA graph padding 是否导致维度不一致。

### 112. 分布式/多卡问题，应该看哪里？

优先看：

```text
parallel_config
Executor.get_class
MultiprocExecutor/RayExecutor
Worker init_distributed_environment
vllm/distributed/parallel_state.py
device_communicators
Pipeline parallel intermediate tensors
custom all-reduce
```

重点判断：

- TP/PP/DP rank 是否正确；
- executor backend 是否符合预期；
- NCCL/custom all-reduce 是否初始化；
- PP 非末级 rank 是否只返回 intermediate tensors；
- DP batch coordination 是否同步。

### 113. KV transfer / disaggregated prefill 问题，应该看哪里？

优先看：

```text
vllm/distributed/kv_transfer
Scheduler.connector
Worker.ensure_kv_transfer_initialized
get_kv_connector_handshake_metadata
Scheduler._build_kv_connector_meta
GPUModelRunner maybe_get_kv_connector_output
```

重点判断：

- scheduler 和 worker connector role 是否匹配；
- handshake metadata 是否收集完整；
- remote KV 命中 token 数是否正确；
- async load 状态是否推进；
- KV load failure policy 是 recompute 还是失败。

## 十一、阅读路径与考察追问

### 114. 如果只想理解普通在线生成，最短阅读路径是什么？

推荐路径：

```text
async_llm.py
core.py step
scheduler.py schedule
executor/abstract.py execute_model
gpu_worker.py
gpu_model_runner.py execute_model/sample_tokens
attention.py forward
```

### 115. 如果重点理解 KV cache，应该读哪些文件？

推荐路径：

```text
config/cache.py
core.py _initialize_kv_caches
kv_cache_utils.py
kv_cache_manager.py
block_pool.py
scheduler.py schedule
worker/block_table.py
gpu_model_runner.py _get_slot_mappings
```

### 116. 如果重点理解性能优化，应该读哪些文件？

推荐路径：

```text
scheduler.py token budget/chunked prefill
GPUModelRunner._determine_batch_execution_and_padding
CUDA graph capture_model
ubatching
attention backend
csrc kernels
```

### 117. 如果重点理解分布式，应该读哪些文件？

推荐路径：

```text
parallel_config
Executor.get_class
multiproc_executor.py / ray_executor.py
gpu_worker.py distributed init
distributed/parallel_state.py
GPUModelRunner PP/DP/TP 相关逻辑
```

### 118. 如果重点理解结构化输出，应该读哪些文件？

推荐路径：

```text
async_llm.py sampling params
structured_output manager
scheduler.get_grammar_bitmask
GPUModelRunner.sample_tokens
apply_grammar_bitmask
OutputProcessor
```

### 119. 新人最容易混淆的点有哪些？

常见混淆点包括：

1. 把 API 层误认为执行层。实际上 API 层只处理协议和参数。
2. 把 `GPUModelRunner` 输出误认为最终用户输出。最终输出还要经过 Scheduler 和 OutputProcessor。
3. 把 Scheduler 理解成简单 prefill/decode 队列。V1 Scheduler 是统一 token/KV 资源调度器。
4. 忽略 prefix cache 和 connector 对 `num_computed_tokens` 的影响。
5. 认为 KV cache 必须连续存储。vLLM 通过 paged attention 支持非连续 block。
6. 忽略 forward context，导致看 Attention.forward 时不知道 metadata 从哪里来。
7. 在 PP 场景下误以为每个 rank 都会 sample。
8. 忽略 execute_model 返回 None 是正常生成路径的一部分。

### 120. 面试中如何用一句话总结 vLLM 推理引擎？

vLLM 推理引擎的核心是用 Scheduler 在每个 step 中统一调度 request、token 和 KV block，用 paged KV cache 管理历史上下文，通过 Executor/Worker/GPUModelRunner 把调度结果转换成 attention metadata 和 slot mapping，最终由 attention backend 和 CUDA kernel 高效完成模型 forward 与采样输出。

## 十二、可继续深入追问的问题清单

以下问题适合在上面基础上继续深入：

1. `num_computed_tokens` 和 `num_tokens_with_spec` 在不同请求阶段如何变化？
2. Prefix cache 全命中时为什么仍可能需要重算最后一个 token？
3. `allocate_slots()` 在 sliding window attention 下如何释放不再需要的 block？
4. KV connector 的 external computed tokens 如何与本地 prefix cache 命中合并？
5. Preemption 后 request 的 KV block 和 computed tokens 如何恢复？
6. `SchedulerOutput` 中 new request data 和 cached request data 分别解决什么问题？
7. 为什么 structured output bitmask 要放在 logits 和 sample 之间？
8. CUDA graph padding 如何影响 slot mapping 和 attention metadata？
9. Pipeline parallel 下 intermediate tensors 如何在 rank 间传递？
10. DP async scheduling 下 sampled token ids 为什么需要广播？
11. Attention backend 如何根据 head size、dtype、block size 做能力校验？
12. fp8 KV cache 会如何影响 KV cache tensor shape、scale 和 kernel 选择？
13. MoE 模型在 worker warmup 和 CUDA graph capture 中有什么特殊处理？
14. disaggregated prefill 场景下 prefill worker 和 decode worker 如何交换 KV？
15. KV cache events 如何用于外部 cache 系统同步？
16. request abort 发生在 execute_model in-flight 时如何保证资源安全释放？
17. overlapping batches 下为什么需要 deferred free？
18. `max_num_batched_tokens`、`max_num_running_reqs`、`gpu_memory_utilization` 如何共同影响吞吐和延迟？
19. prompt logprobs 为什么可能影响 prefix cache 使用？
20. non-causal attention 为什么会禁用 prefix caching 和 chunked prefill？
