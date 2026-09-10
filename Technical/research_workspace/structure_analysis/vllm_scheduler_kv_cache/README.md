# vLLM 调度与 KV Cache 管理层梳理

本目录用于系统梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的调度层与 KV Cache 管理层。

当前代码库的主线实现集中在 `vllm/v1`，传统资料中常见的 `SequenceGroup`、`BlockManager`、`CacheEngine` 等概念，在当前版本中主要被 V1 的 `Request`、`Scheduler`、`KVCacheManager`、`KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool`、worker 侧 `BlockTables` 等结构替代。

## 文档目录

1. [调度器总览](01_scheduler_overview.md)
   - `Scheduler` 的职责、核心字段、调度循环、running/waiting/skipped 队列、预算、抢占、输出结构。

2. [KV Cache 架构](02_kv_cache_architecture.md)
   - `KVCacheManager`、`KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool`、prefix cache、block hash、free/evict/watermark。

3. [请求生命周期](03_request_lifecycle.md)
   - 请求从 `EngineCoreRequest` 到 `Request`，再到调度、prefill、decode、完成、释放的全链路。

4. [Worker 与 Attention 边界](04_worker_attention_boundary.md)
   - scheduler 输出如何进入 `GPUModelRunner`，如何更新 request state、block table、slot mapping，并交给 attention backend。

5. [KV Connector 与 Offload](05_kv_connector_offload.md)
   - scheduler/worker 两侧 KV Connector 的分工，异步 KV load/save、remote KV、失败重算与 deferred free。

6. [旧架构到 V1 映射与阅读顺序](06_legacy_mapping_and_reading_guide.md)
   - 老概念与 V1 新结构的对应关系，以及推荐阅读路径。

## 核心代码入口

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`

## 一句话总览

调度层负责“谁算、算多少、占哪些 KV block、何时抢占/释放”；KV Cache 管理层负责“block 的分配、复用、prefix cache 命中、回收和驱逐”；worker/model runner 负责“把调度结果转成真实 GPU tensor、block table、slot mapping 并执行模型”。
