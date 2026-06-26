# 06. Expert Parallel 如何服务 MoE？

源码位置：

- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/model_executor/models/`

本问题关注：Expert Parallel 如何在 MoE 模型中分布 experts，router/top-k 选择后 token 如何 dispatch 到目标 expert rank，expert output 如何 combine 回原位置，EP 使用哪些 all-to-all 通信，以及 EP 如何和 TP / DP / PP 组合。

---

## 1. 一句话回答

Expert Parallel 是 MoE 专用的 expert 级并行：

```text
不同 experts 放在不同 rank 上；
每个 token 经 router 选择 expert；
token 被 all-to-all dispatch 到 expert 所在 rank；
expert 计算后再 all-to-all combine 回原 token 顺序。
```

一句话记忆：

```text
EP 是“token 按 router 结果被发到不同 expert rank”。
```

---

## 2. 本文要回答的问题

```text
expert 如何被分配到 EP rank？
router logits / top-k 选择在哪里发生？
token permutation / dispatch 如何实现？
all-to-all 输入输出 tensor shape 如何变化？
expert output 如何 combine 回原 token 顺序？
EP 和 TP 同时存在时谁先通信？
EP 对 load balancing 和 padding 有什么影响？
```

---

## 3. 最小主链路占位

```text
hidden states
  → router / gating
  → top-k expert ids / weights
  → token 按 expert 分桶
  → all-to-all dispatch 到 expert rank
  → local expert compute
  → all-to-all combine 回原 rank
  → 按 token 原顺序 scatter / reduce
  → MoE layer output
```

---

## 4. 通信原语占位

```text
all-to-all：
  MoE token dispatch / combine 的核心通信。

all-reduce：
  如果 expert 内部或 shared expert 与 TP 组合，可能需要聚合。

all-gather：
  某些 router metadata 或 expert map 场景可能出现。
```

---

## 5. 待梳理源码点

```text
fused_moe 层入口
expert parallel 配置字段
expert map / expert placement
router topk
token dispatch permutation
all_to_all 调用路径
combine / unpermute 路径
shared expert 路径
EP 与 quantized MoE kernel 的关系
```

---

## 6. 和其他并行的关系

```text
EP + TP：
  dense projection 可以 TP，MoE experts 可以 EP。

EP + DP：
  每个 DP replica 内可能有独立 expert parallel group。

EP + PP：
  只有包含 MoE layers 的 pipeline stage 涉及 EP。

EP + 通信：
  EP 的核心不是 all-reduce，而是 token-level all-to-all shuffle。
```
