---
title: Scheduler 端的调度是池化调度吗？它和 vLLM 调度是什么关系？
category: scheduler
tags:
  - scheduler
  - vllm-scheduler
  - kvpool-scheduler
  - kv-pool
  - kv-cache
related:
  - ../02_vllm_architecture/ascend-store-connector-role-split.md
  - ../04_kv_pool/README.md
source:
  - tmp_doc/documents/question/question.md
  - tmp_doc/documents/pool/03_Scheduler端_调度决策.md
---

# Scheduler 端的调度是池化调度吗？它和 vLLM 调度是什么关系？

## 问题

文档里说 Scheduler 端负责回答“是否命中外部池中的 KV Cache、命中后需要预留多少本地 block、哪些请求需要存到外部池”。这个调度只是池化调度吗？它和 vLLM 原生调度是什么关系？

## 简要回答

是的，这里的 Scheduler 端调度主要是 KV Pool / 外部 KV Cache 池化相关的调度辅助逻辑，不是替代 vLLM 原生 Scheduler 的完整请求调度。

更准确地说：vLLM Scheduler 是主调度器，KVPoolScheduler 是嵌入 vLLM Scheduler 调度流程中的 KV Cache 池化协同模块。

## 详细解答

vLLM Scheduler 负责整体推理调度，包括：

- 哪些请求本轮运行；
- prefill / decode 如何安排；
- token budget 如何分配；
- 本地 KV block 如何分配；
- 请求是否需要抢占、等待或继续执行。

KVPoolScheduler 只负责外部 KV Pool 相关问题，包括：

- 请求在外部池中命中了多少 KV Cache；
- 命中后需要在本地预留多少 block 作为加载目标；
- 哪些请求或 chunk 的 KV Cache 需要保存到外部池；
- 构造 Worker 端 load / save 所需元数据。

典型调用关系：

```text
vLLM Scheduler 调度循环
    ↓
KVPoolScheduler.get_num_new_matched_tokens()
    ↓
vLLM Scheduler 根据 hit 数量决定分配多少 block
    ↓
KVPoolScheduler.update_state_after_alloc()
    ↓
KVPoolScheduler.build_connector_meta()
    ↓
Worker 端执行 load/save
```

所以 KVPoolScheduler 不接管 vLLM 的全局调度策略，而是影响其中与外部 KV Cache 复用相关的部分。

## 和当前项目的关系

`tmp_doc/documents/pool/03_Scheduler端_调度决策.md` 讲的是 KV Pool Scheduler 侧逻辑。它描述的是外部 KV 池的查找、分配辅助和元数据构建，而不是 vLLM Scheduler 的完整调度算法。

## 相关问题

- [为什么一个类 AscendStoreConnector 同时用于 Scheduler 和 Worker？](../02_vllm_architecture/ascend-store-connector-role-split.md)
- [既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？](../02_vllm_architecture/ascend-store-connector-adapter.md)
