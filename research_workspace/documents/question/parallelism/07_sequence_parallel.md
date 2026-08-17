# 07. Sequence Parallel 在 vLLM 中切的是什么？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`
- `vllm/vllm/v1/worker/utils.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

当前源码里和 SP 直接相关的入口，主要分成两类：

1. `ParallelConfig.use_sequence_parallel_moe`：MoE / EP 路径里的 sequence-parallel token 布局优化。
2. `compilation_config.pass_config.enable_sp`：V1 worker 编译路径下的 SP 开关，用于让 residual / intermediate tensors 在 TP rank 间按 sequence 维分片传递。

本问题关注：Sequence Parallel 和 Context Parallel 的区别，vLLM 中为什么没有独立的 `sequence_parallel_size` 并行维度，`use_sequence_parallel_moe` 解决什么问题，SP 如何影响 MoE dispatch / combine、PP IntermediateTensors、residual all-gather，以及它和 TP / DP / EP / PP / CP 的关系。

---

## 1. 一句话回答

Sequence Parallel 是 **token / sequence 维上的中间激活布局优化**。

```text
SP 把 hidden states / tokens 沿 sequence 维分片，
让不同 rank 处理不同 token shard，
避免某些 TP / DP / EP / PP 组合下重复计算或重复通信。
```

它不是 Context Parallel。

```text
SP 切的是中间 activation / token 布局；
CP 切的是同一个请求的 attention context / KV cache。
```

最重要的区别：

```text
SP 不负责把 attention 的全局 KV 依赖做正确；
CP / DCP / PCP 才负责跨 context shard 计算 attention，并用 LSE 合并 partial attention state。
```

---

## 2. 本文要回答的问题

```text
SP 和 CP / DCP / PCP 有什么区别？
vLLM 里有没有独立的 sequence_parallel_size？
use_sequence_parallel_moe 是什么？
SP 为什么主要出现在 MoE / EP / DP / TP 组合里？
SP 如何影响 MoE dispatch / combine？
SP 如何影响 PP IntermediateTensors / residual？
SP 会不会改变 KV cache layout？
SP 需要 attention backend 返回 LSE 吗？
SP 和 Megatron 训练里的 sequence parallel 是不是一回事？
```

---

## 3. SP 和 CP 的核心区别

| 项 | SP | CP / DCP / PCP |
|---|---|---|
| 全称 | Sequence Parallel | Context Parallel |
| 切分对象 | hidden states / tokens / activation | attention context / KV cache |
| 主要维度 | sequence / token 维 | context / KV position 维 |
| 是否改变 KV cache layout | 通常不改变 | 会改变 |
| 是否解决长上下文 attention | 不直接解决 | 是 |
| 是否需要 LSE merge | 不需要 | 需要 |
| 典型通信 | all-gather、all-to-all、dispatch/combine | all-gather query、partial attention、LSE reduce / merge |
| vLLM 主要场景 | MoE dispatch、PP intermediate、TP/DP/EP 组合优化 | DCP decode、PCP prefill、attention backend |

一句话区分：

```text
SP 是“token/activation 怎么分给 rank 处理”；
CP 是“同一个请求的 KV context 怎么分给 rank 做 attention”。
```

如果只把 `hidden_states` 沿 token 维切开，但 attention 仍然要求每个 token 看完整上下文，这还不能算完整 CP。

CP 必须额外处理：

```text
1. KV cache / context 分片；
2. query / KV 的跨 rank 访问；
3. 每个 rank 的 partial attention output；
4. softmax LSE 语义正确合并。
```

SP 通常不触碰这些 attention 语义。

---

## 4. vLLM 里的 SP 通常不是独立 world-size 维度

在 vLLM 的并行拓扑里，常见显式或可查询的并行维度包括：

```text
DP / PP / TP / EP / PCP / DCP
```

而 SP 在当前代码里不是一个单独的并行拓扑维度，更像两类已有 group 上的张量布局规则：

```text
1. MoE 路径：通过 use_sequence_parallel_moe 告诉 EP dispatch/combine，输入 token 已经是 sequence-sharded；
2. 编译路径：通过 enable_sp 让 residual / intermediate tensors 在 TP ranks 之间按 sequence 维 scattered，并在需要 attention/QKV 前 all-gather 回完整 residual。
```

DCP 比较特殊：

```text
DCP 不增加 global world size；
它复用 TP 相关 rank，
按 decode_context_parallel_size 在现有 rank mesh 上组织 DCP groups。
```

而 SP 更特殊：

```text
vLLM 中通常没有类似 sequence_parallel_size 的独立 rank layout 维度；
SP 更多是已有 TP / DP / EP / PP group 上的一种 activation / token 布局。
```

所以不要把 world size 写成：

```text
world_size = DP × PP × TP × SP
```

更准确的理解是：

```text
SP 不额外引入一组 rank；
它在已有 group 内改变 token / activation 的分布状态。
```

这和 PCP / DCP 的区别是：

```text
PCP 是 prefill context parallel 的显式配置维度；
DCP 是 TP group 内部 reshape 出来的 context parallel group；
SP 是某些执行路径上的 sequence/token layout。
```

---

## 5. use_sequence_parallel_moe 是什么

vLLM 中最明确的 SP 入口之一是：

```text
ParallelConfig.use_sequence_parallel_moe
```

它主要服务 MoE。

但这不是当前代码里唯一可见的 SP 入口。V1 worker 还会通过：

```text
compilation_config.pass_config.enable_sp
is_residual_scattered_for_sp(...)
```

决定 residual 是否在 TP ranks 间按 sequence 维 scattered，以及 PP stage 间发送 `IntermediateTensors` 时哪些 tensor 需要先 all-gather。

在 TP + DP + EP + 某些 all-to-all backend 同时启用时，会遇到一个问题：

```text
TP attention o_proj 后的 all-reduce 会让 hidden states 在 TP group 内变成 replicated。
```

也就是说，同一批 tokens 可能出现在多个 TP rank 上。

如果 MoE expert dispatch 直接消费这个 replicated hidden states：

```text
每个 TP rank 都会对同一批 tokens 做 router / dispatch；
EP all-to-all 会搬运重复 tokens；
本地 experts 可能做重复计算；
最后 combine 还要处理重复结果。
```

Sequence Parallel MoE 的目的就是避免这个问题：

```text
把进入 MoE expert 路径的 tokens 按 sequence / token 维拆开，
让不同 rank 只处理自己的 token shard，
避免重复 dispatch 和重复 expert compute。
```

所以这里的 SP 不是为了切权重，也不是为了切 KV cache。

它解决的是：

```text
MoE 路径上 token batch 在 TP/DP/EP 组合下是否被重复处理。
```

---

## 6. SP 在 MoE dispatch / combine 中怎么体现

MoE 的主链路是：

```text
hidden_states
  → router / topk
  → 按 expert 归属 dispatch tokens
  → local experts compute
  → combine 回原 token 顺序
```

EP 切的是 experts：

```text
不同 rank 持有不同 experts 或 expert shard。
```

SP 切的是 tokens：

```text
不同 rank 处理不同 token shard。
```

二者组合后，可以理解为：

```text
1. hidden_states 先处于 sequence-parallel token shard 布局；
2. router / topk 对本地 token shard 计算 expert 选择；
3. EP all-to-all 根据 expert 归属交换 tokens；
4. 本地 experts 只计算收到的 tokens；
5. combine 把 expert 输出还原到对应 token shard / 原顺序。
```

在代码接口上，dispatch / combine 会显式感知这个状态：

```text
get_ep_group().dispatch(..., is_sequence_parallel=True/False)
get_ep_group().combine(..., is_sequence_parallel=True/False)
```

`is_sequence_parallel=True` 的含义不是“新建一个 SP group”，而是告诉通信和 MoE 路径：

```text
当前输入 tokens 已经按 sequence/token 维处于分片状态，
不要把它当成每个 rank 都有完整 token batch 的普通布局。
```

---

## 7. SP 和 TP 的关系

TP 和 SP 切的是不同维度。

```text
TP：切 hidden / head / intermediate / weight 维度；
SP：切 token / sequence 维度。
```

典型 dense Transformer 层里，TP 常见模式是：

```text
QKV / GateUp：切 output dim；
O / Down：切 input dim，然后 all-reduce 得到完整 hidden states。
```

all-reduce 后，hidden states 在 TP group 内可能是 replicated 的。

对于普通 dense FFN，这通常没问题；但对于 MoE，如果 replicated hidden states 进入 expert dispatch：

```text
多个 TP rank 可能对同一批 tokens 重复做 router / expert dispatch。
```

因此 sequence parallel MoE 会把 token 维重新分片：

```text
TP 解决单层矩阵 / head 并行；
SP 解决后续 token 路径是否重复处理。
```

---

## 8. SP 和 EP / DP 的关系

EP 的核心是 expert 维度并行：

```text
router/topk → 按 expert 归属 dispatch tokens → local experts → combine
```

DP 的核心是请求 / batch replica：

```text
不同 DP rank 处理不同请求或不同 batch shard。
```

SP 在 MoE 场景里通常夹在这些机制之间：

```text
DP 提供多 replica / batch 维度；
TP 可能让 hidden states 在 TP group 内复制；
SP 把 tokens 在 sequence 维拆开；
EP 再根据 expert 归属做 all-to-all。
```

所以 sequence parallel MoE 是一种组合优化：

```text
它不是 EP 的替代品；
它让 EP dispatch 的输入不再是重复 token batch。
```

可以记成：

```text
EP 切 expert；
SP 切 token；
二者一起减少 MoE 路径中的重复 token 处理。
```

---

## 9. SP 和 PP IntermediateTensors 的关系

SP 也会影响 Pipeline Parallel 的中间张量传递。

当前源码里，这部分不是抽象口径，而是直接体现在 `gpu_worker.py` / `gpu_model_runner.py`：

```text
- `is_residual_scattered_for_sp(...)` 判断 residual 当前是否仍是 sequence-sharded；
- `get_pp_group().irecv_tensor_dict(..., all_gather_tensors=...)` / `isend_tensor_dict(...)`
  决定 PP rank 间收发 `IntermediateTensors` 时，哪些张量需要借助 TP group 先做 all-gather；
- `gpu_model_runner._sync_intermediate_tensors(...)` 会在 QKV + Attention 需要完整 residual 前，对 scattered residual 做 `get_tp_group().all_gather(..., dim=0)`。
```

PP stage 之间传递的是 hidden states / residual / intermediate tensors：

```text
stage i output → send → stage i+1 input
```

如果开启了 sequence parallel，某些 tensor 可能已经按 token 维 scattered。

这时 PP send/recv 必须知道：

```text
1. 哪些 tensor 可以直接发送本地 shard；
2. 哪些 tensor 在下游需要完整值，必须先 all-gather；
3. residual 是否处于 scattered 状态。
```

例如 residual：

```text
residual 可能因为 SP 处于 sequence-sharded 状态；
如果下游 stage 需要完整 residual，
ModelRunner 需要通过 TP group all_gather 把它还原。
```

所以 SP 不只是 MoE 内部问题，也会影响 PP 的 `IntermediateTensors` 布局协议。

但这里仍然不是 CP：

```text
PP + SP 处理的是 hidden/residual tensor 的 token 分片；
CP 处理的是 attention KV context 分片和 LSE merge。
```

---

## 10. SP 用到的通信原语

SP 常见通信不是 attention LSE merge，而是普通 tensor 布局转换和 MoE token exchange。

常见原语包括：

```text
all_gather:
  把 sequence-sharded activation 还原成完整 hidden states。

all_to_all:
  MoE / EP 场景中按 expert 归属交换 tokens。

combine / scatter:
  expert 输出后回到原 token 顺序或原 token shard 布局。
```

和 CP 对比：

```text
SP:
  hidden/token shard ↔ full hidden/token layout
  expert dispatch/combine

CP:
  local context attention output + local LSE
  跨 CP rank 合并成全局 attention output
```

因此：

```text
SP 不要求 attention backend 返回 softmax LSE；
DCP / PCP 才要求 backend 能返回或处理 LSE。
```

---

## 11. SP 会不会改变 KV cache layout

通常不会。

SP 关注的是：

```text
forward 中 hidden_states / tokens / residual 如何在 rank 间分布。
```

而 CP / DCP / PCP 会影响：

```text
KV cache block size；
block table；
slot mapping；
dcp_local_seq_lens；
attention metadata；
partial attention merge。
```

所以不要把 SP 理解成：

```text
把 KV cache 按 sequence 维分给不同 rank。
```

这更接近 CP / DCP / PCP。

SP 更准确是：

```text
把中间激活 tokens 按 sequence 维分给不同 rank 处理。
```

---

## 12. SP 和 CP 最容易混淆的地方

### 12.1 都提到了 sequence，是否一样？

不一样。

```text
SP 的 sequence：通常指 hidden_states 里的 token 维布局；
CP 的 context：通常指 attention 需要访问的 KV positions。
```

两者都可能沿 token / position 维切分，但语义不同。

### 12.2 SP 能不能单独跑完整 attention？

不能简单这么说。

如果只做：

```text
X_i = X[:, seq_chunk_i, :]
```

那么 FFN 可以本地算，因为 FFN 是逐 token 独立的。

但 attention 需要：

```text
Q_i attends to global K/V
```

这时必须额外处理全局 K/V 访问和 partial attention 合并。

一旦开始处理 KV context 分片、query/KV 交换、LSE merge，就进入 CP 的范畴。

### 12.3 SP 的 partial output 能不能直接 all-reduce？

SP 本身通常不是 attention partial output，所以这个问题不对应。

CP 的 attention partial output 不能直接 all-reduce，原因是：

```text
不同 context shard 的 softmax 分母不同；
必须用 LSE 重新归一化后合并。
```

SP 的 all-gather / all-to-all 更多是 tensor layout 或 token dispatch 语义。

---

## 13. 和 Megatron 训练里的 SP 的关系

Megatron 训练语境里的 Sequence Parallel 常用于 TP 下减少 activation replication。

典型目标是：

```text
LayerNorm / Dropout / residual 等中间激活沿 sequence 维分片，
减少 TP group 内 activation 显存复制。
```

vLLM 是 inference 框架，它里面的 SP 更集中体现为：

```text
1. MoE token dispatch / combine 的 sequence-parallel 输入；
2. PP IntermediateTensors / residual 的 scattered 布局处理；
3. 特定 all-to-all backend 下避免重复 token 计算和通信。
```

所以概念上有相似点：

```text
都是沿 sequence/token 维切 activation。
```

但不要把 Megatron 训练中的完整 SP 机制直接等同到 vLLM：

```text
vLLM 的 SP 更像推理执行路径里的 token layout 优化；
不是一个普遍独立的 sequence_parallel_size 拓扑维度。
```

---

## 14. 和其他并行方式的区别

| 并行方式 | 切分对象 | 典型目的 | 和 SP 的关系 |
|---|---|---|---|
| TP | weight / hidden / heads | 单层矩阵和 attention head 并行 | TP 后 hidden 可能 replicated，SP 可再切 token |
| PP | layers | 模型层流水 | PP 需要知道 SP 下 intermediate 是否 scattered |
| DP | requests / batch replica | 吞吐扩展 | SP MoE 常和 DP + EP + TP 组合出现 |
| EP | experts | MoE expert 并行 | EP 切 expert，SP 切进入 expert 的 tokens |
| CP / DCP / PCP | attention context / KV cache | 长上下文 attention 并行 | CP 切 KV context，SP 不负责 LSE attention merge |

---

## 15. 完整心智模型

可以把 vLLM 里的 SP 压缩成这条链路：

```text
TP / DP / EP / PP 组合产生某种 hidden_states 布局
  → 某些路径下 hidden_states 在 TP group 内可能 replicated
  → sequence parallel 把 tokens 沿 sequence 维拆成 shard
  → MoE router / dispatch / expert compute 只处理本地 token shard
  → combine 回对应 token 顺序或 shard 布局
  → 必要时 all_gather 还原给后续 PP stage / dense path
```

它的重点不是：

```text
attention KV cache 怎么分片。
```

而是：

```text
中间 activation / tokens 是否被重复处理。
```

---

## 16. 容易混淆的点

### 16.1 SP 是不是 CP 的 prefill 版本？

不是。

```text
PCP = Prefill Context Parallel，服务 prefill attention context；
SP = Sequence Parallel，服务 token/activation layout。
```

PCP 需要 attention backend 支持 prefill context parallel；SP 不等于 PCP。

### 16.2 SP 会不会增加 world size？

通常不会。

```text
SP 没有单独的 sequence_parallel_size rank 维度；
它复用已有 TP / DP / EP / PP group 的 rank，改变 tensor 布局。
```

### 16.3 SP 是不是只在 MoE 里有意义？

在当前 vLLM 源码里，SP 最明显的两个使用点是：

```text
1. sequence parallel MoE；
2. V1 编译路径下的 residual / IntermediateTensors sequence-sharded 布局。
```

所以它不只是“MoE 里的一个布尔开关”，只是 MoE 这条线最容易看见；另一条是 PP + TP + full-graph compile 下 residual 的 scattered / all-gather 协议。

### 16.4 SP 是否需要 LSE？

不需要。

LSE 是 CP attention partial output 合并所需的 softmax 归一化信息。

SP 处理的是 token / activation 布局，不是把不同 KV context shard 上的 attention partial states 合并。

---

## 17. 最终可以记成三句话

```text
1. SP 切 token / sequence 维的中间激活，不是切 KV cache。
2. CP / DCP / PCP 切 attention context，需要 backend 用 LSE 合并 partial attention state。
3. vLLM 里的 SP 更像已有 TP / DP / EP / PP 组合中的 token layout 优化，而不是独立增加 world_size 的并行维度。
```
