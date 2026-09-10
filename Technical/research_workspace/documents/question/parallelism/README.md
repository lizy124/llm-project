# vLLM V1 Parallelism 并行体系问题目录

源码位置：

- `vllm/vllm/config.py`
- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/model_executor/layers/`
- `vllm/vllm/model_executor/models/`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/`
- `vllm/vllm/v1/attention/`

这个目录按问题拆解 vLLM V1 的并行体系，重点回答：vLLM 里有哪些并行策略，它们分别切分什么，如何建立 rank / group 拓扑，forward 中如何通信，KV cache / attention / scheduler / sampling 如何受并行影响，以及 TP / PP / DP / EP / PCP / DCP / SP 等策略如何组合。

---

## 1. 总览文档

- [vLLM V1 Parallelism 并行体系总览](parallelism_overview.md)

适合第一次建立全局印象。

总览主线：

```text
ParallelConfig
  → distributed init
  → world / rank / local_rank
  → TP / PP / DP / EP / PCP / DCP groups
  → EPLB 等辅助均衡机制
  → model loading / layer partition
  → ModelRunner forward
  → communication primitives
  → KV cache / attention / logits / sampling
  → output aggregation
```

---

## 2. 主线专题阅读顺序

### 01. 并行配置与 rank 拓扑

- [ParallelConfig 如何定义并行拓扑？](01_parallel_config_and_topology.md)

回答：

```text
parallel size 从哪里来？
world_size、world_size_across_dp 如何对应 TP / PP / PCP / DP，DCP / EP 为什么不额外乘进去？
rank / local_rank / group rank 分别是什么？
配置如何影响 Worker、模型加载和执行？
```

### 02. distributed groups

- [vLLM 如何创建和管理并行 group？](02_distributed_groups.md)

回答：

```text
parallel_state.py 维护哪些 group？
TP / PP / DP / EP / PCP / DCP / EPLB group 如何初始化？
每个 group 在 forward 中服务什么通信？
```

### 03. Tensor Parallel

- [Tensor Parallel 在 vLLM 中如何切分模型？](03_tensor_parallel.md)

回答：

```text
TP 切什么 tensor？
QKV / MLP / embedding / logits 如何分片？
哪些地方需要 all-reduce / all-gather？
TP 如何影响 attention head 和 KV head？
```

### 04. Pipeline Parallel

- [Pipeline Parallel 如何切分 layer 和传递中间状态？](04_pipeline_parallel.md)

回答：

```text
PP 切什么 layer？
每个 stage 持有哪些层？
intermediate_tensors 如何 send / recv？
logits 和 sampling 在哪个 stage 发生？
```

### 05. Data Parallel

- [Data Parallel 如何做请求级并行？](05_data_parallel.md)

回答：

```text
DP 切的是请求还是模型？
每个 replica 如何处理 batch？
DP 和 Executor / Scheduler / EngineCore 如何协同？
DP 下 KV cache 和输出如何隔离或汇总？
```

### 06. Expert Parallel

- [Expert Parallel 如何服务 MoE？](06_expert_parallel.md)

回答：

```text
EP 切什么 expert？
router 之后 token 如何 dispatch？
all-to-all 如何把 token 发给 expert rank？
EP 如何和 TP / DP 组合？
```

### 07. Context Parallel

- [Context Parallel / DCP / PCP 如何切分上下文？](07_context_parallel.md)
- [Sequence Parallel 在 vLLM 中指什么？](07_sequence_parallel.md)

回答：

```text
CP 切 sequence / context / KV 的哪一维？
DCP 和 PCP 分别服务 decode / prefill 的什么瓶颈？
SP 和 CP/DCP/PCP 是什么关系？
attention partial output 为什么需要 LSE merge？
哪些 attention backend 支持 CP？
```

### 08. KV cache 与并行

- [KV cache 如何受并行策略影响？](08_kv_cache_and_parallelism.md)

回答：

```text
TP / PP / DP / EP / CP 下 KV cache 如何放置？
block table / slot mapping 是否分片？
KV connector / external KV transfer 如何适配并行？
```

### 09. Attention 与并行

- [Attention backend 如何感知并行？](09_attention_and_parallelism.md)

回答：

```text
TP 如何影响 num_heads / num_kv_heads？
GQA / MQA / MLA 在并行下如何执行？
DCP 为什么要求 backend 返回 LSE？
FlashAttention / FlashInfer / Triton / FlashMLA 支持哪些并行特性？
```

### 10. 通信原语

- [vLLM 并行体系用到哪些通信原语？](10_communication_primitives.md)

回答：

```text
all-reduce / all-gather / reduce-scatter / all-to-all 分别解决什么？
send / recv / broadcast / barrier 在哪里出现？
NCCL / custom allreduce / Ray / multiprocessing 分别在哪一层？
```

### 11. 并行组合

- [TP / PP / DP / EP / CP 如何组合？](11_parallel_composition.md)

回答：

```text
TP x PP x DP 的 rank mesh 如何理解？
EP 是否是独立维度？
CP 和 attention backend 如何组合？
组合并行下哪些约束最容易出错？
```

### 12. 端到端案例

- [几个典型并行配置如何完整执行？](12_end_to_end_examples.md)

回答：

```text
TP=2 如何 forward？
TP=2, PP=2 如何传 hidden states？
DP=2, TP=2 如何分请求？
MoE + EP 如何 dispatch token？
DCP + FlashAttention 如何 merge attention state？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
parallelism_overview.md
  → 01_parallel_config_and_topology.md
  → 02_distributed_groups.md
  → 10_communication_primitives.md
```

### 3.2 按执行链路完整阅读

```text
parallelism_overview.md
  → 01_parallel_config_and_topology.md
  → 02_distributed_groups.md
  → 03_tensor_parallel.md
  → 04_pipeline_parallel.md
  → 05_data_parallel.md
  → 08_kv_cache_and_parallelism.md
  → 09_attention_and_parallelism.md
```

### 3.3 按 MoE / 长上下文专题阅读

```text
06_expert_parallel.md
  → 10_communication_primitives.md
  → 11_parallel_composition.md

07_context_parallel.md
  → 08_kv_cache_and_parallelism.md
  → 09_attention_and_parallelism.md
  → ../attention/attention_methods/12_flash_attention_family.md
```

---

## 4. 文档定位

```text
README.md：
  当前目录索引和阅读路线。

parallelism_overview.md：
  并行体系总览，先建立完整心智模型。

编号专题：
  按问题拆开的专题文档，其中 Context Parallel 和 Sequence Parallel 分别成篇。
```

---

## 5. 最小心智模型

如果只记一条，可以记：

```text
vLLM 的并行体系不是 TP / PP / DP 名词堆叠，而是 parallel config → rank group → 模型/请求/KV/attention 切分 → 通信原语 → 输出合并 的完整链路。
```
