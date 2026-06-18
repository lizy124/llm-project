---
title: DeepSeek V4 里的 c4 / c128 到底是什么意思？
category: kv-cache
tags:
  - kv-cache
  - deepseek
  - c4
  - c128
  - compressed-kv
  - hybrid-kv-cache
related:
  - hybrid-kv-cache.md
  - ../06_scheduler/kvpool-scheduler-vs-vllm-scheduler.md
source:
  - tmp_doc/documents/question/question.md
  - tmp_doc/documents/pool/03_Scheduler端_调度决策.md
---

# DeepSeek V4 里的 c4 / c128 到底是什么意思？

## 问题

DeepSeek V4 里的 `c4` / `c128` 到底是什么意思？`c128` 是把 128 个 token 的特征压缩成 1 个 token 特征吗？

## 简要回答

`c4` / `c128` 可以理解为某些 KV cache group 在序列长度维度上的压缩倍率或稀疏倍率。其中 `c128` 更接近表示：这个 KV cache group 的 cache 长度大约是原始 token 长度的 `1/128`。

但 `c128` 不是简单地把 128 个 token 的 hidden state 做平均池化，压缩成一个普通 token 特征。它对应的是模型网络结构内部某个压缩 attention / latent KV 分支产生的 KV 表示。

## 详细解答

可以先用长度关系理解：

```text
普通 c1 KV：   每 1 个 token 对应 1 个 KV 位置
c4 KV：        大约每 4 个 token 对应 1 个 KV 位置
c128 KV：      大约每 128 个 token 对应 1 个 KV 位置
```

普通 full KV / c1 可以理解为：

```text
tokens:  t1   t2   t3   t4   ...   t16384
KV:      kv1  kv2  kv3  kv4  ...   kv16384
```

c128 KV 可以粗略理解为：

```text
tokens:  t1 ... t128 | t129 ... t256 | ...
KV:           kv_1   |      kv_2     | ...
```

这里的 `kv_1` 不是某一个 token 的 hidden state，也不是普通意义上的“一个 token 特征”。它通常是模型结构中某个专门模块根据一段 token 信息计算出来的压缩 KV 表示。

从模型角度看：

```text
输入 token hidden states
        ↓
普通 attention / KV 分支       → 产生 c1 KV
压缩 attention / KV 分支       → 产生 c4 / c128 KV
MLA / latent KV 分支           → 产生特殊压缩 KV
```

这些分支中的投影矩阵、压缩方式、latent 表示方式，都是模型结构的一部分，并且在训练阶段已经参与训练。

所以训练阶段会学习：

- 如何把一段 token 范围的信息表达成较少的 KV 表示；
- 后续 token 如何利用这些压缩 KV 做 attention；
- 压缩后丢失的信息如何通过模型结构和参数补偿。

推理阶段不会重新设计压缩方式，只是执行训练好的网络：

```text
模型网络决定：c128 KV 怎么算
训练过程学习：c128 KV 里应该保留什么信息
推理过程执行：用训练好的权重算出 c128 KV
KV Cache 负责：把算出来的 c128 KV 缓存起来
KV Pool 负责：按正确粒度保存、加载和复用这些 KV
```

因此，`c128` 对 KV Pool / Scheduler 的核心影响不是“怎么压缩”，而是“怎么对齐和传输”。

例如 block size 是 128 tokens 时：

```text
c1:    1 个 cache block 对应 1   × 128 = 128 tokens
c4:    1 个 cache block 对应 4   × 128 = 512 tokens
c128:  1 个 cache block 对应 128 × 128 = 16384 tokens
```

所以 DeepSeek V4 的 KV Cache 存取最终可能需要按 `16384 tokens` 这样的粒度对齐。

## 和当前项目的关系

KV Pool 需要根据不同 KV cache group 的 cache family 推导保存、加载和传输粒度。对于 c128 这类压缩 group，一个 cache block 对应的原始 token 范围更大，因此会影响 `cache_transfer_granularity`。

相关说明见 `tmp_doc/documents/pool/03_Scheduler端_调度决策.md`。

## 相关问题

- [hybrid KV cache 是什么？它是一种 KV cache 吗？](hybrid-kv-cache.md)
- [Scheduler 端的调度是池化调度吗？它和 vLLM 调度是什么关系？](../06_scheduler/kvpool-scheduler-vs-vllm-scheduler.md)
