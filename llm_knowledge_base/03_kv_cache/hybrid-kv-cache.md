---
title: hybrid KV cache 是什么？它是一种 KV cache 吗？
category: kv-cache
tags:
  - kv-cache
  - hybrid-kv-cache
  - full-attention
  - kv-cache-group
related:
  - deepseek-v4-c4-c128-kv-cache.md
  - ../06_scheduler/kvpool-scheduler-vs-vllm-scheduler.md
source:
  - tmp_doc/documents/question/question.md
---

# hybrid KV cache 是什么？它是一种 KV cache 吗？

## 问题

`hybrid KV cache` 是什么？它是一种新的 KV cache 吗？

## 简要回答

`hybrid KV cache` 不是一种单独的新 KV cache 数据结构，而是一种 KV cache 管理形态。

普通 KV cache 通常表示所有层或所有 attention 使用同一种 KV cache 规格；hybrid KV cache 表示同一个模型里存在多种不同规格的 KV cache，需要按 group 分组管理。

## 详细解答

普通 Transformer 模型中，所有层可能都是 full attention：

```text
Layer 0  Full Attention  → KV cache
Layer 1  Full Attention  → KV cache
Layer 2  Full Attention  → KV cache
...
```

这种情况下 KV cache 的形状、block 规则、可复用范围基本一致，可以按统一规则管理。

但一些模型会混合多种 attention 或 KV 规格，例如：

```text
Layer 0  Full Attention       → KV cache group A
Layer 1  Sliding Window Attn  → KV cache group B
Layer 2  Full Attention       → KV cache group A
Layer 3  Sliding Window Attn  → KV cache group B
...
```

或者类似 DeepSeek 这类存在不同 KV 压缩 / latent 规格的情况：

```text
KV group 0: c1    cache
KV group 1: c4    cache
KV group 2: c128  cache
```

这些 group 的 KV cache 可能在以下方面不同：

- attention 类型不同；
- 每个 token 对应的 KV 数据大小不同；
- block size 不同；
- cache shape 不同；
- 可缓存 token 范围不同；
- 保存 / 加载粒度不同。

因此需要 hybrid KV cache manager 按 group 分别管理。

判断逻辑通常关注两点：

1. 是否存在多个 KV cache group；
2. 是否至少有一个 group 不是普通 FullAttention 规格。

如果满足这些条件，就不能再把所有 KV cache 当成统一形态处理。

## 和当前项目的关系

KV Pool 在处理外部 KV Cache 保存和加载时，需要知道当前是否是 hybrid KV cache。因为不同 group 的 block size、cache family 和传输粒度可能不同，会影响 lookup、load、save 和 metadata 构建。

原始问题来自 `tmp_doc/documents/question/question.md`。

## 相关问题

- [DeepSeek V4 里的 c4 / c128 到底是什么意思？](deepseek-v4-c4-c128-kv-cache.md)
- [Scheduler 端的调度是池化调度吗？它和 vLLM 调度是什么关系？](../06_scheduler/kvpool-scheduler-vs-vllm-scheduler.md)
