# vLLM 推理引擎层完整梳理

本目录用于系统梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 推理引擎层的结构、主链路、关键类、调度机制、KV Cache 管理、执行层、Attention 与底层算子衔接。

## 文档导航

建议按以下顺序阅读：

1. [01_推理引擎层总览.md](01_推理引擎层总览.md)
   - 说明 vLLM 推理引擎层的边界、V0/V1 关系、分层架构和一条主链路。

2. [02_请求生命周期_API到EngineCore.md](02_请求生命周期_API到EngineCore.md)
   - 从 OpenAI API / AsyncLLM 到 EngineCoreRequest、EngineCoreClient、OutputProcessor 的请求生命周期。

3. [03_EngineCore主循环与V0兼容.md](03_EngineCore主循环与V0兼容.md)
   - 重点梳理 V1 `EngineCore` 初始化、KV cache 初始化、step 主循环、batch queue、V0 兼容层定位。

4. [04_Scheduler调度机制.md](04_Scheduler调度机制.md)
   - 详细梳理 `Scheduler` 的 waiting/running 状态机、token budget、chunked prefill、spec decode、preemption、encoder cache 等。

5. [05_KVCache_Block_PrefixCaching.md](05_KVCache_Block_PrefixCaching.md)
   - 详细梳理 KV Cache 配置、KVCacheManager、BlockPool、prefix caching、block table、KV connector/offload。

6. [06_Executor_Worker_ModelRunner执行层.md](06_Executor_Worker_ModelRunner执行层.md)
   - 梳理 Executor 抽象、多进程/单进程/Ray 执行、GPU Worker、GPUModelRunner 的模型执行流程。

7. [07_Attention_ModelExecutor_CUDA链路.md](07_Attention_ModelExecutor_CUDA链路.md)
   - 梳理模型执行层、Attention 层、backend 抽象、forward context、slot mapping、csrc kernel 的衔接。

8. [08_关键文件阅读顺序与调试地图.md](08_关键文件阅读顺序与调试地图.md)
   - 给出新人阅读顺序、按问题类型定位文件、常见调试路径。

## 核心结论

当前仓库里，vLLM 推理引擎层的主线已经明显偏向 V1：

```text
entrypoints/openai
  -> v1/engine/async_llm.py
  -> v1/engine/core_client.py
  -> v1/engine/core.py
  -> v1/core/sched/scheduler.py
  -> v1/executor/*
  -> v1/worker/gpu_worker.py
  -> v1/worker/gpu_model_runner.py 或 v1/worker/gpu/model_runner.py
  -> model_executor/*
  -> model_executor/layers/attention/* + v1/attention/*
  -> csrc/* CUDA/C++/CPU kernels
```

一句话理解：API 层只负责接请求；`AsyncLLM` 负责前台异步封装；`EngineCore` 是推理引擎内核；`Scheduler` 决定每一步算哪些 token 和如何使用 KV block；`Executor/Worker` 把调度结果发到设备；`GPUModelRunner` 组织 batch、block table、attention metadata 并执行模型；模型中的 Attention 最终通过 backend 调用底层 kernel。

## 重要代码锚点

- `code/vllm/vllm/v1/engine/async_llm.py:70`：`AsyncLLM`，异步前台引擎。
- `code/vllm/vllm/v1/engine/core.py:96`：`EngineCore`，V1 引擎核心。
- `code/vllm/vllm/v1/engine/core.py:240`：KV cache 初始化流程。
- `code/vllm/vllm/v1/engine/core.py:479`：`EngineCore.step()` 主循环。
- `code/vllm/vllm/v1/core/sched/scheduler.py:68`：`Scheduler`。
- `code/vllm/vllm/v1/core/sched/scheduler.py:387`：调度主函数 `schedule()`。
- `code/vllm/vllm/v1/core/kv_cache_manager.py:110`：`KVCacheManager`。
- `code/vllm/vllm/v1/executor/abstract.py:37`：`Executor` 抽象。
- `code/vllm/vllm/v1/worker/gpu_worker.py:117`：GPU `Worker`。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:418`：`GPUModelRunner`。
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`：`GPUModelRunner.execute_model()`。
- `code/vllm/vllm/v1/attention/backend.py:55`：V1 Attention backend 抽象。
- `code/vllm/vllm/model_executor/layers/attention/attention.py:178`：模型层 Attention 模块。
