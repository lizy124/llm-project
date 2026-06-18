---
title: register_connector(name, module_path, class_name) 三个参数的含义是什么？
category: vllm-architecture
tags:
  - vllm
  - kv-connector
  - registration
  - lazy-loading
related:
  - kv-connector-registration.md
  - ascend-store-connector-adapter.md
source:
  - tmp_doc/documents/question/question.md
---

# register_connector(name, module_path, class_name) 三个参数的含义是什么？

## 问题

`register_connector(name, module_path, class_name)` 里的 `name`、`module_path`、`class_name` 分别表示什么？

## 简要回答

- `name`：连接器配置名称，用户在配置文件中使用的字符串。
- `module_path`：连接器所在 Python 模块路径。
- `class_name`：模块中的具体连接器类名。

注册时通常只记录这些信息，不一定立即 import 具体模块；真正使用连接器时再根据模块路径和类名进行延迟加载。

## 详细解答

这三个参数分别解决三个问题：

```text
用户配置里叫什么？        → name
代码实现在哪个模块？      → module_path
模块里具体哪个类？        → class_name
```

例如可以理解成：

```text
name        = "AscendStoreConnector"
module_path = "vllm_ascend.xxx.ascend_store"
class_name  = "AscendStoreConnector"
```

运行时流程大致是：

```text
用户配置 connector 名称
        ↓
注册表根据 name 找到 module_path + class_name
        ↓
动态 import module_path
        ↓
从模块中取出 class_name 对应的类
        ↓
实例化连接器
```

这种设计的好处是：

1. 用户配置简单，只需要写连接器名称；
2. vLLM 核心不需要硬编码具体厂商实现；
3. 模块可以延迟加载，避免不必要依赖；
4. 下游项目可以独立扩展连接器。

## 和当前项目的关系

当前项目关注 vllm-ascend 的 AscendStoreConnector。该连接器需要能被 vLLM 通过注册表识别和加载，因此注册参数决定了配置名称如何映射到具体 Python 实现。

原始问题来自 `tmp_doc/documents/question/question.md`。

## 相关问题

- [为什么需要 KV 连接器注册机制？](kv-connector-registration.md)
- [既然有 KVPoolScheduler 和 KVPoolWorker，为什么还需要 AscendStoreConnector？](ascend-store-connector-adapter.md)
