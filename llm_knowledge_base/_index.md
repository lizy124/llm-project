# LLM Knowledge Base Index

## 02 vLLM 架构

- [为什么需要 KV 连接器注册机制？](02_vllm_architecture/kv-connector-registration.md) — vLLM 通过注册机制解耦上游接口和下游硬件实现。
- [`register_connector(name, module_path, class_name)` 三个参数的含义是什么？](02_vllm_architecture/register-connector-parameters.md) — 解释连接器注册时配置名、模块路径和类名的作用。
- [为什么一个类 AscendStoreConnector 同时用于 Scheduler 和 Worker？](02_vllm_architecture/ascend-store-connector-role-split.md) — 说明同一连接器类如何通过 role 分发到不同内部组件。
- [既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？](02_vllm_architecture/ascend-store-connector-adapter.md) — 说明 AscendStoreConnector 的适配器和包装器职责。
- [为什么 vLLM 要求一个连接器接口同时实现 Scheduler 和 Worker 方法？](02_vllm_architecture/kv-connector-interface-design.md) — 解释 vLLM 连接器接口设计中的配置简化取舍。

## 03 KV Cache

- [hybrid KV cache 是什么？它是一种 KV cache 吗？](03_kv_cache/hybrid-kv-cache.md) — hybrid KV cache 是多种 KV cache 规格混合存在时的分组管理模式。
- [DeepSeek V4 里的 c4 / c128 到底是什么意思？](03_kv_cache/deepseek-v4-c4-c128-kv-cache.md) — 解释 c4/c128 作为 KV cache group 序列长度压缩倍率的含义。

## 06 Scheduler 调度

- [Scheduler 端的调度是池化调度吗？它和 vLLM 调度是什么关系？](06_scheduler/kvpool-scheduler-vs-vllm-scheduler.md) — 说明 KVPoolScheduler 是嵌入 vLLM Scheduler 的 KV Pool 协同模块。
