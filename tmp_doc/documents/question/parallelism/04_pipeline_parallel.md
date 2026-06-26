# 04. Pipeline Parallel 如何切分 layer 和传递中间状态？

源码位置：

- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/model_executor/models/`
- `vllm/vllm/model_executor/model_loader/`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/forward_context.py`

本问题关注：Pipeline Parallel 如何把模型 layers 切成多个 stage，每个 PP rank 持有哪些层，stage 之间如何传递 hidden states / intermediate_tensors，哪些 rank 负责 logits / sampling，以及 PP 如何和 TP / DP 组合。

---

## 1. 一句话回答

Pipeline Parallel 是层级切分：

```text
模型 layers 被拆到多个 pipeline stage；
每个 stage 只执行自己负责的 layers；
stage 间传递 intermediate_tensors；
通常只有最后一个 stage 产生最终 hidden states / logits / sampling 输入。
```

一句话记忆：

```text
PP 是“一个模型的不同层放在不同 stage 上串起来跑”。
```

---

## 2. 本文要回答的问题

```text
模型 layers 如何分配到 PP stage？
first stage / intermediate stage / last stage 分别负责什么？
PP 下 forward 输入输出形态如何变化？
intermediate_tensors 在哪里构造和传递？
logits 和 sampling 在哪个 rank / stage 发生？
PP 对 KV cache 初始化有什么影响？
PP 和 TP 组合时 rank mesh 如何理解？
```

---

## 3. 最小主链路占位

```text
SchedulerOutput
  → 所有 PP stage 的 Worker 接收执行命令
  → stage 0 准备 input_ids / embeddings
  → stage 0 forward 本 stage layers
  → send intermediate_tensors 到 stage 1
  → 中间 stage forward 自己的 layers
  → last stage 得到 final hidden states
  → logits / sampling / ModelRunnerOutput
```

---

## 4. 通信原语占位

```text
send / recv：
  pipeline stage 之间传 hidden states / intermediate_tensors。

broadcast / metadata sync：
  控制信息、shape、执行状态可能需要同步。

collective_rpc：
  Executor 向所有 Worker / PP stage 分发执行命令。
```

---

## 5. 待梳理源码点

```text
PP rank 判断 helper
is_first_pp_rank / is_last_pp_rank
模型 layer range 计算
make_empty_intermediate_tensors
intermediate_tensors 传递路径
GPUModelRunner 中 PP 非首 rank 输入处理
logits processor 在 PP last rank 的行为
sampling 与 PP last stage 的关系
```

---

## 6. 和其他并行的关系

```text
PP + TP：
  每个 PP stage 内部可以有多个 TP rank。

PP + DP：
  每个 DP replica 内部可以有一套 PP pipeline。

PP + KV cache：
  只有包含 attention layers 的 stage 需要对应 KV cache。

PP + Output：
  非 last stage 通常不直接产出最终 logits。
```
