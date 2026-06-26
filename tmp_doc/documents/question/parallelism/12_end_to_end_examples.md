# 12. 几个典型并行配置如何完整执行？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/`
- `vllm/vllm/model_executor/`
- `vllm/vllm/v1/attention/`

本问题关注：通过几个具体配置，把前面 TP / PP / DP / EP / CP、通信原语、KV cache 和 attention backend 串成完整执行链路，用端到端案例检验并行体系是否闭环。

---

## 1. 一句话回答

端到端案例的目标是把抽象并行概念落到一轮执行：

```text
请求进来
  → 分到哪些 rank
  → 哪些 rank 持有权重和 KV cache
  → forward 中发生哪些通信
  → 哪个 rank 产生 logits / sampling
  → 输出如何返回
```

---

## 2. Case 1：TP=2，PP=1，DP=1

要回答：

```text
一个请求如何被两个 TP rank 共同执行？
QKV projection 如何分片？
attention heads 如何分配？
MLP 哪些位置 all-reduce？
logits 是否需要 gather？
```

主链路占位：

```text
SchedulerOutput
  → 两个 TP rank 的 Worker 同时执行
  → 每层 parallel linear 计算局部分片
  → attention 使用本 rank heads / KV heads
  → 必要位置 all-reduce
  → logits / sampling
  → ModelRunnerOutput
```

---

## 3. Case 2：TP=2，PP=2，DP=1

要回答：

```text
模型 layers 如何分成两个 stage？
每个 stage 内部的 TP=2 如何通信？
stage 0 如何把 hidden states 发给 stage 1？
哪个 stage 负责 logits / sampling？
```

主链路占位：

```text
stage 0, TP group
  → embedding / 前半层 forward
  → TP all-reduce
  → send intermediate_tensors

stage 1, TP group
  → recv intermediate_tensors
  → 后半层 forward
  → TP all-reduce
  → logits / sampling
```

---

## 4. Case 3：DP=2，TP=2

要回答：

```text
两个 DP replica 如何处理不同请求？
每个 replica 内部的 TP group 是否独立？
请求 A 和请求 B 是否互相通信？
KV cache 是否 replica-local？
```

主链路占位：

```text
请求 A
  → DP replica 0
  → replica 0 内 TP=2 合作 forward
  → replica 0 的 KV cache
  → 输出 A

请求 B
  → DP replica 1
  → replica 1 内 TP=2 合作 forward
  → replica 1 的 KV cache
  → 输出 B
```

---

## 5. Case 4：MoE + EP

要回答：

```text
router 在哪里产生 expert ids？
token 如何按 expert 发送到不同 rank？
all-to-all 的输入输出是什么？
expert output 如何 combine 回原 token 顺序？
EP 和 TP 同时存在时如何分层理解？
```

主链路占位：

```text
hidden states
  → router top-k
  → token permutation / bucket by expert
  → all-to-all dispatch
  → local experts compute
  → all-to-all combine
  → unpermute / weighted combine
  → MoE output
```

---

## 6. Case 5：DCP + FlashAttention

要回答：

```text
KV context 如何被多个 DCP rank 分担？
每个 rank 的 partial attention output 是什么？
为什么要返回 softmax LSE？
merge_attn_states 如何恢复完整 attention output？
哪些 backend 支持这个路径？
```

主链路占位：

```text
query / KV cache
  → DCP rank 计算本地 context attention
  → backend 返回 partial output + LSE
  → 跨 DCP group 通信
  → LSE merge partial states
  → 完整 attention output
```

---

## 7. Case 6：PP + KV cache

要回答：

```text
每个 PP stage 是否都有 KV cache？
只有部分 layers 在本 stage 时，KV cache 如何按 layer 绑定？
block_table / slot_mapping 是否对所有 stage 一样？
非 last stage 是否需要 logits？
```

主链路占位：

```text
PP stage 0：
  持有前半 attention layers 的 KV cache；
  forward 后传 intermediate_tensors。

PP stage 1：
  持有后半 attention layers 的 KV cache；
  forward 后产出 logits / sampling 输入。
```

---

## 8. 后续补源码时的统一检查表

每个 case 都按同一组问题补全：

```text
1. rank 拓扑是什么？
2. 请求分到哪些 rank？
3. 权重如何分布？
4. KV cache 如何分布？
5. forward 中发生哪些通信？
6. attention backend 是否有特殊要求？
7. logits / sampling 在哪里？
8. 输出如何回到 EngineCore？
```
