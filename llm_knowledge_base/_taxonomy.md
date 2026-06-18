# 大模型知识体系分类树

```text
大模型知识体系
├── 01 LLM 基础
│   ├── Transformer
│   ├── Attention
│   ├── Prefill / Decode
│   └── KV Cache 基础
│
├── 02 vLLM 架构
│   ├── Engine
│   ├── Scheduler
│   ├── Worker
│   ├── BlockManager
│   └── KVConnector
│       ├── 连接器注册机制
│       ├── Scheduler / Worker 角色分发
│       └── Connector 适配器设计
│
├── 03 KV Cache
│   ├── FullAttention KV
│   ├── Sliding Window KV
│   ├── Hybrid KV Cache
│   ├── MLA / Latent KV
│   ├── KV Cache Block
│   └── 压缩 KV / c1 / c4 / c128
│
├── 04 KV Pool
│   ├── 外部 KV 池
│   ├── Lookup
│   ├── Load / Save
│   ├── ReqMeta
│   ├── cache_transfer_granularity
│   └── 跨请求 / 跨实例 KV 复用
│
├── 05 并行策略
│   ├── TP Tensor Parallel
│   ├── PP Pipeline Parallel
│   ├── DP Data Parallel
│   ├── EP Expert Parallel
│   ├── CP Context Parallel
│   └── SP Sequence Parallel
│
├── 06 Scheduler 调度
│   ├── vLLM Scheduler
│   ├── KVPoolScheduler
│   ├── Prefill 调度
│   ├── Decode 调度
│   └── Block 分配
│
├── 07 Ascend / NPU
│   ├── vllm-ascend
│   ├── HCCL
│   ├── NPU KV Cache
│   └── AscendStoreConnector
│
├── 08 代码阅读笔记
│   ├── pool_scheduler.py
│   ├── pool_worker.py
│   ├── config_data.py
│   └── connector.py
│
└── 99 Inbox
    └── 暂未分类问题
```
