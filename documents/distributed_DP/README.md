# distributed_DP 文档索引

本目录以 Mooncake 为例，说明 `code/vllm-ascend` 中 P/D 分离（Prefill/Decode Disaggregation）的完整流程。

建议按顺序阅读：

1. [总览 - P/D 分离架构](01_总览_PD分离架构.md)
2. [连接器注册、配置与角色分发](02_连接器注册_配置与角色分发.md)
3. [Scheduler 端 - 调度决策](03_Scheduler端_调度决策.md)
4. [Worker 端 - 初始化、内存注册与握手](04_Worker端_初始化_内存注册与握手.md)
5. [Decode 端拉取 KV Cache（Pull 模式）](05_Decode端拉取KV_Cache.md)
6. [Prefill 端延迟释放与完成确认](06_Prefill端延迟释放与完成确认.md)
7. [多并行、Hybrid KV Cache 与 block/rank 映射](07_多并行_Hybrid与Block映射.md)
8. [Layerwise Push 模式](08_Layerwise_Push模式.md)
9. [全链路串联总结](09_全链路串联总结.md)

主线代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/utils.py`
- `code/vllm-ascend/examples/offline_disaggregated_prefill_npu.py`
