# vLLM V1 Parallelism 并行体系总览

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/`
- `vllm/vllm/model_executor/layers/`
- `vllm/vllm/model_executor/models/`
- `vllm/vllm/v1/attention/`

本文是 vLLM V1 并行体系的总览文档。目标不是先解释某一个并行名词，而是建立一张完整地图：每种并行切什么、哪些 rank 组成 group、forward 中如何通信、KV cache / attention / logits / sampling 如何受影响，以及这些并行策略如何组合。

---

## 0. 梳理规划

本文按“先定拓扑，再定切分对象，再看通信，再看组合”的方式组织。

要回答的问题分成 8 组：

```text
1. vLLM 的 parallel config 如何决定 world / rank / group 拓扑？
2. TP / PP / DP / EP / CP 分别切分什么对象？
3. 每种并行策略使用哪些通信原语？
4. 模型加载、layer 放置和 forward 如何受并行影响？
5. KV cache、block table、slot mapping 在并行下如何变化？
6. attention backend 如何支持 TP / CP / DCP / MLA / EP 相关路径？
7. logits、sampling 和 ModelRunnerOutput 在并行下如何产生和汇总？
8. 多种并行策略组合时，rank mesh 和约束如何理解？
```

阅读顺序建议：

```text
parallelism_overview.md
  → 01_parallel_config_and_topology.md
  → 02_distributed_groups.md
  → 10_communication_primitives.md
  → 03_tensor_parallel.md
  → 04_pipeline_parallel.md
  → 05_data_parallel.md
  → 06_expert_parallel.md
  → 07_context_parallel.md
  → 08_kv_cache_and_parallelism.md
  → 09_attention_and_parallelism.md
  → 11_parallel_composition.md
  → 12_end_to_end_examples.md
```

---

## 1. 一句话回答

vLLM 的并行体系可以理解成：

```text
ParallelConfig 定义并行规模；
distributed init 根据规模建立 rank group；
模型层、请求、expert、context、KV cache 按不同 group 切分；
forward 过程中通过 all-reduce / all-gather / all-to-all / send-recv 等原语交换数据；
最后在正确的 rank / stage 上完成 logits、sampling 和输出回收。
```

更短地说：

```text
并行策略 = 切分对象 + rank group + 通信原语 + 状态归属 + 输出合并。
```

---

## 2. 并行体系的 6 层视角

### 2.1 配置与拓扑层

这一层回答：

```text
world_size 是多少？
当前 rank 是谁？
TP / PP / DP / EP / CP size 分别是多少？
一个 global rank 属于哪些 group？
```

关键概念：

```text
rank
local_rank
world_size
tensor_model_parallel_size
pipeline_model_parallel_size
data_parallel_size
expert_parallel_size
context parallel / decode context parallel / prefill context parallel
```

### 2.2 模型切分层

这一层回答：

```text
哪些权重在哪些 rank 上？
每个 rank 持有完整模型还是部分模型？
一层内部切 tensor，还是层之间切 stage？
MoE expert 是否分布在不同 rank？
```

典型策略：

```text
Tensor Parallel:
  切单层内部 tensor / weight / hidden dim / attention head。

Pipeline Parallel:
  切 layer / pipeline stage。

Expert Parallel:
  切 MoE experts。
```

### 2.3 请求与 batch 分发层

这一层回答：

```text
同一个请求由哪些 rank 合作完成？
不同请求是否分给不同 replica？
SchedulerOutput 如何进入对应 Worker？
DP rank 之间是否共享请求状态？
```

典型策略：

```text
Data Parallel:
  切请求 / batch / replica。
```

### 2.4 Attention 与 KV cache 层

这一层回答：

```text
KV cache 是 replicated 还是 sharded？
attention heads 如何随 TP 切分？
context / KV 序列是否随 CP 切分？
backend 是否支持 partial attention output？
```

典型策略：

```text
Context Parallel / DCP / PCP:
  切 context / sequence / KV 范围。
```

### 2.5 通信原语层

这一层回答：

```text
切分后如何把结果合回来？
哪些地方需要 all-reduce？
哪些地方需要 all-gather？
MoE token dispatch 为什么需要 all-to-all？
PP stage 之间为什么是 send / recv？
```

常见原语：

```text
all-reduce
all-gather
reduce-scatter
all-to-all
broadcast
send / recv
barrier
collective_rpc
```

### 2.6 输出合并层

这一层回答：

```text
哪个 rank 产生 logits？
哪个 rank 执行 sampling？
PP 非末 stage 返回什么？
DP 多 replica 的输出如何回到 EngineCore？
```

---

## 3. 各并行策略的定位

### 3.1 Tensor Parallel

```text
切分对象：
  单层内部 tensor / weight / hidden dim / attention head。

典型通信：
  all-reduce、all-gather、reduce-scatter。

影响范围：
  Linear、QKV projection、MLP、embedding、lm_head、attention head 分布。
```

一句话记忆：

```text
TP 是“一个 layer 内多个 rank 一起算”。
```

### 3.2 Pipeline Parallel

```text
切分对象：
  Transformer layers / pipeline stage。

典型通信：
  send / recv intermediate tensors。

影响范围：
  模型加载、forward 输入输出、logits 所在 rank、sampling 所在 stage。
```

一句话记忆：

```text
PP 是“一个模型的不同层放在不同 stage 上串起来跑”。
```

### 3.3 Data Parallel

```text
切分对象：
  请求 / batch / replica。

典型通信：
  执行层 request dispatch、控制面 broadcast / RPC、输出回收。

影响范围：
  EngineCore / Executor / Worker、SchedulerOutput 分发、KV cache 隔离、负载均衡。
```

一句话记忆：

```text
DP 是“不同请求给不同模型副本跑”。
```

### 3.4 Expert Parallel

```text
切分对象：
  MoE experts。

典型通信：
  all-to-all token dispatch / combine。

影响范围：
  router、expert placement、MoE layer forward、token permutation。
```

一句话记忆：

```text
EP 是“token 按 router 结果被发到不同 expert rank”。
```

### 3.5 Context Parallel / DCP / PCP

```text
切分对象：
  sequence context / KV context / prefill 或 decode 的 attention 范围。

典型通信：
  all-gather、all-to-all、reduce / merge partial attention states。

影响范围：
  attention metadata、KV cache、FlashAttention / FlashInfer / Triton backend、LSE merge。
```

一句话记忆：

```text
CP 是“把长上下文 attention 拆给多个 rank 合作算”。
```

---

## 4. 并行和通信原语的关系

并行策略和通信原语不是一一对应，而是多对多关系。

```text
TP 常用 all-reduce / all-gather；
PP 常用 send / recv；
EP 常用 all-to-all；
CP 可能用 all-gather / all-to-all / LSE merge；
DP 更多体现在请求分发、执行隔离和输出汇总。
```

需要避免的误解：

```text
all-reduce 不是 TP；
all-to-all 不是 EP；
send / recv 不是 PP。

通信原语只是切分之后让数据重新对齐的手段。
```

---

## 5. 和已有文档的关系

并行体系会和已有目录交叉：

```text
../engine_core/
  EngineCore 如何发起 execute_model 和回收输出。

../executor_worker_model_runner/
  Executor / Worker / ModelRunner 如何承载分布式执行。

../scheduler/
  SchedulerOutput、token budget、请求状态如何进入执行层。

../kv_cache_transfer/
  KV block、prefix cache、external KV transfer 如何与并行交互。

../attention/
  attention metadata、backend selection、KV layout、CUDA graph 如何受并行影响。
```

---

## 6. 后续专题占位

```text
01_parallel_config_and_topology.md：
  配置、world size、rank 拓扑。

02_distributed_groups.md：
  parallel_state.py 中各种 group。

03_tensor_parallel.md：
  linear / embedding / attention head 的 tensor 切分。

04_pipeline_parallel.md：
  layer 切分、stage 通信、intermediate_tensors。

05_data_parallel.md：
  请求级 replica 并行和执行层分发。

06_expert_parallel.md：
  MoE expert placement、router、all-to-all。

07_context_parallel.md：
  CP / DCP / PCP、attention partial state、LSE merge。

08_kv_cache_and_parallelism.md：
  KV cache、block table、slot mapping 与并行。

09_attention_and_parallelism.md：
  attention backend 与 TP / CP / MLA / GQA。

10_communication_primitives.md：
  通信原语与 vLLM 封装。

11_parallel_composition.md：
  多种并行策略组合关系。

12_end_to_end_examples.md：
  典型配置端到端案例。
```

---

## 7. 最小心智模型

可以先把 vLLM 并行理解成下面这张图：

```text
global world
  ├── data parallel replicas
  │     ├── replica 0: TP x PP x EP x CP
  │     └── replica 1: TP x PP x EP x CP
  │
  ├── tensor parallel groups
  │     └── 同一 layer 内合作算 tensor
  │
  ├── pipeline parallel groups
  │     └── 不同 stage 串联传 hidden states
  │
  ├── expert parallel groups
  │     └── MoE token dispatch / expert compute / combine
  │
  └── context parallel groups
        └── 长上下文 attention / KV context 分片
```

最终要落回一句话：

```text
每种并行策略都必须同时回答：切什么、谁通信、怎么通信、状态放哪、结果在哪合并。
```
