---
title: 既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？
category: vllm-architecture
tags:
  - vllm
  - kv-connector
  - adapter
  - ascend-store-connector
  - kv-pool
related:
  - ascend-store-connector-role-split.md
  - kv-connector-interface-design.md
source:
  - research_workspace/documents/question/question.md
---

# 既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？

## 问题

既然实际逻辑已经拆成 `KVPoolScheduler` 和 `KVPoolWorker`，为什么还需要 `AscendStoreConnector`？

## 简要回答

`AscendStoreConnector` 是适配器 / 包装器。它负责满足 vLLM 的连接器接口和工厂加载方式，并根据当前角色把调用代理给内部的 `KVPoolScheduler` 或 `KVPoolWorker`。

也就是说：

```text
vLLM 认识 AscendStoreConnector
AscendStoreConnector 内部再组合 KVPoolScheduler / KVPoolWorker
```

## 详细解答

`KVPoolScheduler` 和 `KVPoolWorker` 更像业务实现类：

- `KVPoolScheduler`：处理 Scheduler 侧外部 KV 池查找、加载规格、元数据构建；
- `KVPoolWorker`：处理 Worker 侧外部 KV 池查询、加载、保存。

但 vLLM 的 KV transfer 框架需要的是一个符合统一接口的 connector 类。这个 connector 类要能被注册、被配置加载、被 vLLM 按统一方法调用。

因此需要 `AscendStoreConnector` 作为适配层：

1. **满足接口规范**：实现 vLLM 期望的 Scheduler / Worker 相关方法；
2. **适配工厂模式**：一个注册名称对应一个类；
3. **组合内部组件**：内部持有 Scheduler 或 Worker 侧实现；
4. **代理方法调用**：根据 role 把调用转发到正确组件；
5. **隔离 vLLM 和下游实现细节**：vLLM 不直接依赖 KVPoolScheduler / KVPoolWorker。

可以类比为：

```text
vLLM 插座标准：KVConnector 接口
AscendStoreConnector：插头适配器
KVPoolScheduler / KVPoolWorker：真正干活的内部电路
```

## 和当前项目的关系

当前项目分析的 Ascend KV Pool 需要接入 vLLM 的 KV connector 体系。`AscendStoreConnector` 是对外入口，`KVPoolScheduler` 和 `KVPoolWorker` 是内部实现分工。

原始问题来自 `research_workspace/documents/question/question.md`。

## 相关问题

- [为什么一个类 AscendStoreConnector 同时用于 Scheduler 和 Worker？](ascend-store-connector-role-split.md)
- [为什么 vLLM 要求一个连接器接口同时实现 Scheduler 和 Worker 方法？](kv-connector-interface-design.md)
