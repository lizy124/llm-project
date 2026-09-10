# 01 推理引擎层总览

## 1. 推理引擎层的边界

在 vLLM 中，“推理引擎层”不是单个目录，而是一条跨目录的运行链路。它大致包含：

```text
入口层
  vllm/entrypoints/

前台异步引擎层
  vllm/v1/engine/async_llm.py
  vllm/v1/engine/input_processor.py
  vllm/v1/engine/output_processor.py
  vllm/v1/engine/core_client.py

引擎核心层
  vllm/v1/engine/core.py

调度层
  vllm/v1/core/sched/
  vllm/v1/core/kv_cache_manager.py
  vllm/v1/core/block_pool.py
  vllm/v1/core/kv_cache_coordinator.py

执行层
  vllm/v1/executor/
  vllm/v1/worker/

模型运行层
  vllm/v1/worker/gpu_model_runner.py
  vllm/v1/worker/gpu/model_runner.py

模型定义与算子封装层
  vllm/model_executor/
  vllm/v1/attention/
  vllm/model_executor/layers/attention/

底层 kernel 层
  csrc/
```

从职责看，可以分成 7 层：

| 层级 | 主要路径 | 职责 |
|---|---|---|
| API/入口层 | `vllm/entrypoints/` | OpenAI API、CLI、server、请求解析 |
| 前台引擎层 | `vllm/v1/engine/async_llm.py` | 异步接口、输入处理、输出收集、与 EngineCore 通信 |
| 引擎核心层 | `vllm/v1/engine/core.py` | 创建 executor、初始化 KV cache、驱动 scheduler/executor 主循环 |
| 调度层 | `vllm/v1/core/sched/` | request 状态机、token budget、prefill/decode、preemption、spec decode |
| KV 管理层 | `vllm/v1/core/kv_cache_*`, `block_pool.py` | KV block 分配、复用、回收、prefix cache、connector |
| 执行层 | `vllm/v1/executor/`, `vllm/v1/worker/` | 单进程/多进程/Ray 执行，worker 设备初始化与模型调用 |
| 模型与 kernel 层 | `model_executor/`, `v1/attention/`, `csrc/` | 模型 forward、attention backend、paged attention、CUDA/C++ kernel |

## 2. V0 与 V1 的关系

当前仓库中，V1 是推理主链路。旧的 `vllm/engine/*` 更偏兼容层，保留旧 API 语义，但底层执行逐步迁移到 V1。

### V0 典型定位

V0 主要包括：

```text
vllm/engine/llm_engine.py
vllm/engine/async_llm_engine.py
vllm/engine/arg_utils.py
```

它们常用于：

- 保留旧接口，例如 `LLMEngine.add_request()`、`LLMEngine.step()`；
- 保持旧代码/旧用户调用方式兼容；
- 通过配置和 client 接入 V1 的 engine core。

### V1 典型定位

V1 主线包括：

```text
vllm/v1/engine/async_llm.py
vllm/v1/engine/core_client.py
vllm/v1/engine/core.py
vllm/v1/core/sched/scheduler.py
vllm/v1/executor/*
vllm/v1/worker/*
```

V1 的特点：

- 前台 API 与 engine core 隔离更清晰；
- `Scheduler` 与 `Executor/Worker` 分层更明确；
- KV cache、prefix caching、spec decode、KV connector、pipeline parallel、data parallel 都被纳入统一调度；
- worker 侧通过 `GPUModelRunner` 统一组织 batch、attention metadata、slot mapping、采样和后处理。

## 3. 端到端主链路

一次 OpenAI Chat/Completion 请求大体经过：

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

## 4. 关键控制流图

```text
AsyncLLM
  ├─ InputProcessor.process_inputs
  ├─ EngineCoreClient.add_request_async
  └─ OutputProcessor.process_outputs

EngineCore
  ├─ _initialize_kv_caches
  ├─ Scheduler.add_request
  ├─ Scheduler.schedule
  ├─ Executor.execute_model
  ├─ Executor.sample_tokens
  └─ Scheduler.update_from_output

Executor
  ├─ collective_rpc("initialize_from_config")
  ├─ collective_rpc("compile_or_warm_up_model")
  └─ collective_rpc("execute_model")

Worker
  ├─ init_device / load_model
  ├─ determine_available_memory
  ├─ initialize_from_config
  ├─ compile_or_warm_up_model
  └─ model_runner.execute_model

GPUModelRunner
  ├─ _update_states
  ├─ _prepare_inputs
  ├─ _get_slot_mappings
  ├─ _build_attention_metadata
  ├─ _preprocess
  ├─ _model_forward
  ├─ compute_logits
  └─ sample_tokens
```

## 5. 推理引擎层最重要的几个类

| 类 | 文件 | 作用 |
|---|---|---|
| `AsyncLLM` | `code/vllm/vllm/v1/engine/async_llm.py:70` | 前台异步引擎封装，面向 API 层 |
| `InputProcessor` | `code/vllm/vllm/v1/engine/input_processor.py:36` | 把用户输入变成 EngineCoreRequest |
| `OutputProcessor` | `code/vllm/vllm/v1/engine/output_processor.py:417` | 把 EngineCoreOutputs 转成用户输出 |
| `EngineCoreClient` | `code/vllm/vllm/v1/engine/core_client.py:71` | 前台与 EngineCore 的通信抽象 |
| `EngineCore` | `code/vllm/vllm/v1/engine/core.py:96` | V1 推理内核主循环 |
| `Scheduler` | `code/vllm/vllm/v1/core/sched/scheduler.py:68` | request/token/KV 资源调度器 |
| `KVCacheManager` | `code/vllm/vllm/v1/core/kv_cache_manager.py:110` | KV block 分配、复用和释放 |
| `BlockPool` | `code/vllm/vllm/v1/core/block_pool.py:130` | 底层 block 池 |
| `Executor` | `code/vllm/vllm/v1/executor/abstract.py:37` | 执行后端抽象 |
| `Worker` | `code/vllm/vllm/v1/worker/gpu_worker.py:117` | GPU worker，负责设备与模型 runner |
| `GPUModelRunner` | `code/vllm/vllm/v1/worker/gpu_model_runner.py:418` | GPU 上组织 batch 和模型 forward 的核心类 |
| `AttentionBackend` | `code/vllm/vllm/v1/attention/backend.py:55` | Attention backend 能力抽象 |
| `Attention` | `code/vllm/vllm/model_executor/layers/attention/attention.py:178` | 模型层 attention 模块 |

## 6. 一句话抓住主干

vLLM 的推理引擎层核心是：`AsyncLLM` 接请求，`EngineCore` 驱动循环，`Scheduler` 决定每步算哪些 token 和占用哪些 KV block，`Executor/Worker/GPUModelRunner` 在设备上执行模型，`Attention backend + csrc` 完成高性能 attention 和 cache kernel。
