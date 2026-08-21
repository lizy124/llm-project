# vLLM Ascend KV Pool 源码梳理

本目录用于梳理 `vllm-ascend` 中 `vllm_ascend/distributed/kv_transfer/kv_pool` 的实现。重点关注 KV Cache 在 NPU、CPU 和外部存储之间的组织、传输、异步调度，以及它和 vLLM KV connector 的边界。

## 文档目录

1. [整体定位与架构](01_kv_pool_overview.md)
   - KV pool 与 vLLM KV transfer 的关系、模块边界、主要对象和调用入口
2. [Ascend Store 源码总览](02_ascend_store_core.md)
   - 全局组件地图、一次请求的最小主链、运行模式和子篇导航
   - [02_1：Connector、Scheduler 与 Coordinator](02_1_ascend_store_control_plane.md)
   - [02_2：Key、Metadata 与 Cache Layout](02_2_ascend_store_metadata_and_layout.md)
   - [02_3：KVPoolWorker 如何生成设备任务](02_3_ascend_store_worker_pipeline.md)
   - [02_4：传输线程与 Backend](02_4_ascend_store_transfer_and_backend.md)
3. [KV Offload 实现](03_kv_offload.md)
   - native CPU/NPU offload、simple offload、copy backend 与设备内存操作
4. [Recompute CPU Offload 如何工作](04_recompute_cpu_offload.md)
   - 重计算 CPU offload 的 scheduler/worker 生命周期和回退语义
5. [数据流、并发与配置](05_data_flow_concurrency_and_config.md)
   - put/get、异步线程、layerwise layout、后端选择、失败和回收路径
6. [代码阅读指南](06_code_reading_guide.md)
   - 推荐阅读顺序、关键符号、源码定位和常见问题

## 核心源码入口

- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py`
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/`
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/`

## 一句话总览

上层 connector 把请求级 KV transfer 转换成 pool 操作；`ascend_store` 负责元数据、调度和后端存取，offload 模块负责本地 CPU/NPU 搬运，重计算模块在无法直接恢复 KV 时提供替代路径。
