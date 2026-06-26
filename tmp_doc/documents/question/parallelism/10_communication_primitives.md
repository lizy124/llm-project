# 10. vLLM 并行体系用到哪些通信原语？

源码位置：

- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/device_communicators/`
- `vllm/vllm/distributed/utils.py`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/attention/ops/`

本问题关注：vLLM 并行体系中常见通信原语分别解决什么问题，all-reduce / all-gather / reduce-scatter / all-to-all / broadcast / send-recv / barrier / collective_rpc 在哪些并行策略中出现，以及 NCCL、custom allreduce、Ray、multiprocessing 等执行机制分别处在哪一层。

---

## 1. 一句话回答

通信原语是并行切分后的数据对齐手段：

```text
先按 TP / PP / DP / EP / CP 切分计算或状态；
再用不同通信原语把 partial result、token、hidden states、metadata 或控制命令传到需要它们的 rank。
```

需要避免的误解：

```text
all-reduce 不是 TP；
all-to-all 不是 EP；
send / recv 不是 PP。

并行策略定义“切什么”；通信原语负责“切完后怎么交换”。
```

---

## 2. 本文要回答的问题

```text
all-reduce / all-gather / reduce-scatter / all-to-all 分别解决什么？
TP / PP / DP / EP / CP 分别常用哪些通信原语？
通信 op 如何选择 group？
custom allreduce 和 NCCL allreduce 的关系是什么？
Executor collective_rpc 和 torch collective 有什么区别？
attention LSE merge 是否属于普通 reduce？
```

---

## 3. 通信原语矩阵占位

| 通信原语 | 典型用途 | 相关并行 |
|---|---|---|
| `all-reduce` | 聚合各 rank partial hidden states | TP |
| `all-gather` | 汇总分片 tensor、query、logits 或 partial state | TP / CP |
| `reduce-scatter` | 聚合后继续保持分片 | TP / sequence-like parallel |
| `all-to-all` | token dispatch / expert dispatch / 某些 DCP 路径 | EP / CP |
| `broadcast` | 同步配置、metadata、权重或控制状态 | 初始化 / DP / PP |
| `send / recv` | pipeline stage 之间传 hidden states | PP |
| `barrier` | 同步执行阶段 | 初始化 / profile / teardown |
| `collective_rpc` | Executor 向 Worker 分发控制命令 | 执行层 |

---

## 4. 分原语占位

### 4.1 all-reduce

```text
作用：
  每个 rank 都有 partial result，最终每个 rank 都需要完整 reduce 结果。

典型场景：
  RowParallelLinear 输出聚合；TP 下 hidden states 对齐。
```

### 4.2 all-gather

```text
作用：
  每个 rank 持有 tensor 分片，需要把所有分片收集起来。

典型场景：
  vocab / logits 汇总；CP 下 query 或 partial state 收集。
```

### 4.3 reduce-scatter

```text
作用：
  先 reduce，再把结果按 rank 切分保存。

典型场景：
  通信优化或保持后续 tensor 分片状态。
```

### 4.4 all-to-all

```text
作用：
  每个 rank 给每个其他 rank 发送不同切片。

典型场景：
  MoE token dispatch / combine；DCP a2a 通信 backend。
```

### 4.5 send / recv

```text
作用：
  点对点传输 tensor。

典型场景：
  pipeline stage i 把 intermediate_tensors 发给 stage i+1。
```

### 4.6 collective_rpc

```text
作用：
  执行层控制面 RPC，不等同于 tensor collective。

典型场景：
  Executor.execute_model、profile、sleep、wake_up、shutdown、LoRA、KV transfer 控制接口。
```

---

## 5. 待梳理源码点

```text
communication_op.py 中 all_reduce / all_gather / broadcast 等封装
device communicator 初始化
custom allreduce 条件
NCCL communicator 使用路径
Ray / mp executor 的 collective_rpc
pipeline send / recv helper
MoE all_to_all helper
DCP a2a / lse reduce helper
barrier / synchronize 调用点
```

---

## 6. 和并行策略的关系

```text
TP：
  重点看 all-reduce / all-gather。

PP：
  重点看 send / recv。

DP：
  重点看请求分发、控制面 RPC 和 replica 输出回收。

EP：
  重点看 all-to-all。

CP：
  重点看 all-gather / all-to-all / LSE merge。
```
