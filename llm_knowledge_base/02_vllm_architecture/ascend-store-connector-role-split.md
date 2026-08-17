---
title: 为什么一个类 AscendStoreConnector 同时用于 Scheduler 和 Worker？
category: vllm-architecture
tags:
  - vllm
  - kv-connector
  - scheduler
  - worker
  - ascend-store-connector
related:
  - ascend-store-connector-adapter.md
  - kv-connector-interface-design.md
source:
  - research_workspace/documents/question/question.md
---

# 为什么一个类 AscendStoreConnector 同时用于 Scheduler 和 Worker？

## 问题

为什么一个类 `AscendStoreConnector` 会同时用于 Scheduler 和 Worker？为什么不是 Scheduler 一个类、Worker 一个类？

## 简要回答

因为 vLLM 的 Scheduler 进程和 Worker 进程通常读取同一个 KV connector 配置，只配置一个连接器名称。为了让同一个配置名称在两个进程里都能工作，AscendStoreConnector 通过 `role` 参数在内部区分当前实例运行在 Scheduler 侧还是 Worker 侧。

简化理解：

```text
role = SCHEDULER → 内部创建 / 代理 KVPoolScheduler
role = WORKER    → 内部创建 / 代理 KVPoolWorker
```

## 详细解答

vLLM 的 KV connector 是从用户配置中加载的。如果用户需要分别配置：

```text
scheduler_connector = xxx
worker_connector = yyy
```

就会引入额外复杂度：

1. 用户要理解两个配置项；
2. Scheduler 和 Worker 配置可能不匹配；
3. 工厂和注册表需要维护两套映射；
4. 分布式部署时更容易出错。

因此更常见的设计是：用户只配置一个 connector 名称。vLLM 在不同进程中实例化这个 connector 时，会传入当前角色信息。连接器类再根据角色创建不同内部组件。

这意味着 `AscendStoreConnector` 本身更像一个入口类：

```text
AscendStoreConnector
├── role = SCHEDULER → KVPoolScheduler
└── role = WORKER    → KVPoolWorker
```

这样既满足 vLLM 对统一 connector 类的要求，又能让 Scheduler 和 Worker 内部逻辑保持分离。

## 和当前项目的关系

当前项目中的 KV Pool 逻辑本身分为 Scheduler 侧和 Worker 侧：Scheduler 负责查找、分配辅助决策和元数据构建；Worker 负责实际加载、保存和查询外部 KV 池。AscendStoreConnector 用同一个配置入口把这两侧连接起来。

原始问题来自 `research_workspace/documents/question/question.md`。

## 相关问题

- [既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？](ascend-store-connector-adapter.md)
- [为什么 vLLM 要求一个连接器接口同时实现 Scheduler 和 Worker 方法？](kv-connector-interface-design.md)
