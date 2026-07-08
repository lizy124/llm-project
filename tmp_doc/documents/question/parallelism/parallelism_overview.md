# vLLM V1 Parallelism 并行体系总览

源码位置：

- `code/vllm/vllm/config/parallel.py`
- `code/vllm/vllm/distributed/parallel_state.py`
- `code/vllm/vllm/distributed/communication_op.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/models/utils.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/logits_processor.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/v1/attention/`

本文按“先定边界，再走主链路，再拆并行维度，最后看组合和排查入口”的方式，梳理 vLLM V1 的并行体系。

它不是逐个通信 API 的字典式说明，而是先回答几个最关键的问题：

```text
1. vLLM 里有哪些并行维度？每个维度到底切什么？
2. ParallelConfig 如何决定 world size、rank mesh 和 group？
3. Worker 初始化时如何建立 TP / PP / DP / EP / PCP / DCP group，SP 为什么不是独立 group？
4. 一次 SchedulerOutput 进入执行层后，会经过哪些 rank？
5. 模型权重、layers、attention heads、experts、KV cache 分别归谁？
6. forward 中哪些地方触发 all-reduce / all-gather / all-to-all / send-recv？
7. logits、sampling、ModelRunnerOutput 最终在哪个 rank 产生？
8. 多种并行策略组合时，哪些维度乘进 world size，哪些只复用已有 rank？
```

---

## 0. 梳理规划

本文按 9 个层次组织：

```text
1. 先给一条总链路：请求如何从 EngineCore 走到并行 Worker。
2. 再定义并行体系的核心公式：world_size / world_size_across_dp / rank mesh。
3. 拆 TP：单层 tensor、attention head、MLP 如何切。
4. 拆 PP：layers、stage、IntermediateTensors 如何切。
5. 拆 DP：请求和 replica 如何切。
6. 拆 EP：MoE experts 和 token dispatch 如何切。
7. 拆 PCP / DCP：prefill / decode context 如何切。
8. 串 KV cache、attention backend、logits / sampling 的状态归属。
9. 最后给组合规则、源码导航和排查检查表。
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
  → 07_sequence_parallel.md
  → 07_context_parallel.md
  → 08_kv_cache_and_parallelism.md
  → 09_attention_and_parallelism.md
  → 11_parallel_composition.md
  → 12_end_to_end_examples.md
```

如果只想快速建立主线，可以先读：

```text
parallelism_overview.md
  → 01_parallel_config_and_topology.md
  → 02_distributed_groups.md
  → 11_parallel_composition.md
  → 12_end_to_end_examples.md
```

---

## 1. 一句话回答

vLLM V1 的并行体系可以压缩成一句话：

```text
ParallelConfig 定义并行规模，parallel_state 建立通信 group，模型层和执行层按 group 切分权重、请求、layer、expert、context 和 KV cache，forward 中通过通信原语对齐分片结果，最后只在约定的输出 rank 上返回 ModelRunnerOutput。
```

更短地说：

```text
并行策略 = 切分对象 + rank group + 通信原语 + 状态归属 + 输出合并。
```

其中最重要的区分是：

```text
TP / PP / PCP / DCP：主要回答“一个 DP replica 内部如何合作完成一次 forward”。
EP：主要回答“MoE layer 内 experts 如何在同一 PP stage 的 DP / PCP / TP ranks 上分布和通信”。
DP：主要回答“不同请求如何分配给不同模型副本”，但 MoE + EP 下 DP ranks 也可能参与 expert group。
```

---

## 2. 最小端到端主链路

并行不是单独存在的，它嵌在 EngineCore 的每轮执行里。

最小链路是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model(scheduler_output)
  → collective_rpc("execute_model")
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _model_forward()
  → TP / PP / EP / CP 通信在模型层或 backend 内发生
  → logits / sampling / IntermediateTensors
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```

对应入口：

- `EngineCore.step()`：`code/vllm/vllm/v1/engine/core.py:479`
- `Executor.execute_model()`：`code/vllm/vllm/v1/executor/abstract.py:221`
- `GPUModelRunner`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418`

在这个链路里：

```text
SchedulerOutput 是执行计划；
Executor 是分发层；
Worker 是设备侧运行时；
ModelRunner 是 batch / forward / sampling 状态机；
parallel group 决定每个模块和哪些 rank 通信。
```

这意味着看并行时不能只看 `parallel_state.py`，还要同时看：

```text
配置如何生成拓扑；
Worker 何时初始化 distributed group；
模型层如何读取 TP / PP / EP group；
attention backend 如何读取 CP / DCP metadata；
Executor 最终从哪个 rank 收输出。
```

---

## 3. 并行体系的核心公式

### 3.1 单个 DP replica 内部 world size

`ParallelConfig.__post_init__()` 里会计算：

```text
world_size = pipeline_parallel_size
           * tensor_parallel_size
           * prefill_context_parallel_size
```

位置：`code/vllm/vllm/config/parallel.py:782`

这表示：

```text
单个 DP replica 内，承载一个模型副本需要多少 worker。
```

注意：

```text
DCP 不乘进 world_size；
EP 不乘进 world_size；
EPLB 不乘进 world_size。
```

它们都是在已有 rank 上创建额外 group 或复用已有 group。

### 3.2 计入 DP 后的 world size

`world_size_across_dp` 是：

```text
world_size_across_dp = world_size * data_parallel_size
```

位置：`code/vllm/vllm/config/parallel.py:508`

也就是：

```text
所有 DP replica 总共需要的 worker 数。
```

例如：

```text
TP = 4
PP = 2
PCP = 1
DP = 3

world_size = 2 * 4 * 1 = 8
world_size_across_dp = 8 * 3 = 24
```

### 3.3 DCP 的特殊约束

DCP 复用 TP rank，所以要求：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

位置：`code/vllm/vllm/config/parallel.py:490`

源码注释说明：

```text
DCP 不改变 world size；
它复用 TP group 的 GPU；
把一个 TP group 切成 tp_size // dcp_size 个 DCP groups。
```

对应 group 构造位置：`code/vllm/vllm/distributed/parallel_state.py:1774`

### 3.4 EP 的特殊地位

EP 也不单独乘进 world size。

MoE 模型中，EP group 从 rank mesh 里取：

```text
data_parallel_size * prefill_context_parallel_size * tensor_parallel_size
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1850`

含义是：

```text
在同一个 PP stage 内，把 DP / PCP / TP 这些已有 rank 组织成 expert parallel group。
```

---

## 4. rank mesh：理解所有 group 的底座

`initialize_model_parallel()` 中，rank 被 reshape 成：

```text
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1749`

可以理解成：

```text
all_ranks[external_dp, dp, pp, pcp, tp]
```

各维度含义：

```text
external_dp：外部 DP / 集成场景里的额外 DP 维度；
dp：vLLM 内部 data parallel replica；
pp：pipeline stage；
pcp：prefill context parallel rank；
tp：tensor parallel rank。
```

从这个 mesh 派生出的 group 是：

```text
TP group：
  固定 external_dp / dp / pp / pcp，沿 tp 维取 rank。

DCP group：
  在 TP 维内部按 decode_context_parallel_size 切小组。

PCP group：
  固定 external_dp / dp / pp / tp，沿 pcp 维取 rank。

PP group：
  固定 external_dp / dp / pcp / tp，沿 pp 维取 rank。

DP group：
  固定 external_dp / pp / pcp / tp，沿 dp 维取 rank。

EP group：
  MoE 模型中固定 external_dp / pp，合并 dp * pcp * tp。
```

对应源码位置：

- TP group：`code/vllm/vllm/distributed/parallel_state.py:1757`
- DCP group：`code/vllm/vllm/distributed/parallel_state.py:1774`
- PCP group：`code/vllm/vllm/distributed/parallel_state.py:1796`
- PP group：`code/vllm/vllm/distributed/parallel_state.py:1815`
- DP group：`code/vllm/vllm/distributed/parallel_state.py:1833`
- EP group：`code/vllm/vllm/distributed/parallel_state.py:1850`

一个例子：

```text
DP = 2
PP = 2
PCP = 1
TP = 4

world_size_across_dp = 2 * 2 * 1 * 4 = 16

DP0, PP0, TP group = [0, 1, 2, 3]
DP0, PP1, TP group = [4, 5, 6, 7]
DP1, PP0, TP group = [8, 9, 10, 11]
DP1, PP1, TP group = [12, 13, 14, 15]

PP group for DP0, TP0 = [0, 4]
PP group for DP0, TP1 = [1, 5]
DP group for PP0, TP0 = [0, 8]
```

---

## 5. Worker 何时初始化并行环境

GPU Worker 初始化分布式环境时会调用：

```text
init_worker_distributed_environment(...)
  → init_distributed_environment(...)
  → ensure_model_parallel_initialized(
        tensor_parallel_size,
        pipeline_parallel_size,
        prefill_context_parallel_size,
        decode_context_parallel_size,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:1164`

关键点：

```text
1. torch distributed world 先初始化；
2. vLLM 再按 ParallelConfig 创建 TP / PP / DP / PCP / DCP / EP / EPLB group；
3. 后续模型层、attention backend、MoE runner 不重新计算拓扑，而是查询 get_tp_group() / get_pp_group() / get_ep_group() 等全局 group。
```

通信封装的入口之一是 `communication_op.py`：

```text
tensor_model_parallel_all_reduce(input_)
  → get_tp_group().all_reduce(input_)

tensor_model_parallel_all_gather(input_, dim)
  → get_tp_group().all_gather(input_, dim)

tensor_model_parallel_reduce_scatter(input_, dim)
  → get_tp_group().reduce_scatter(input_, dim)
```

位置：`code/vllm/vllm/distributed/communication_op.py:12`

所以：

```text
ParallelConfig 决定 group 规模；
parallel_state 创建 group；
communication_op 和模型层使用 group；
Executor / Worker 负责让所有 rank 同步进入执行。
```

---

## 6. Tensor Parallel：切单层内部 tensor

### 6.1 TP 切什么

TP 主要切单层内部的大 tensor，包括：

```text
- QKV projection 的输出维 / head 维；
- attention heads / KV heads；
- MLP up / gate / down projection；
- embedding / vocab parallel；
- lm_head / logits 相关张量。
```

一句话：

```text
TP 是“同一个请求、同一层，由多个 rank 分片合作算”。
```

### 6.2 QKV 如何切

`QKVParallelLinear` 的注释说明：

```text
QKV transformation 的 weight matrix 拼在输出维；
这一层沿 attention head 维并行；
KV heads 少于 query heads 时，KV head 可能复制。
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:914`

关键代码：

```text
num_heads = total_num_heads / tp_size
如果 tp_size >= total_num_kv_heads：
  每个 TP rank 至少保留 1 个 KV head，并复制 KV head
否则：
  num_kv_heads = total_num_kv_heads / tp_size
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:966`

因此 TP 下 attention 不是把所有 heads gather 到一个 rank 再算，而是：

```text
每个 TP rank 计算自己持有的 query heads / KV heads；
每个 TP rank 读写自己那部分 KV cache；
必要的输出合并由后续 projection 或通信完成。
```

### 6.3 TP 用什么通信

TP 常见通信包括：

```text
all-reduce：
  RowParallelLinear / MLP down projection 后合并 partial output。

all-gather：
  某些需要完整 hidden / vocab / tensor 的地方收集分片。

reduce-scatter：
  在 sequence parallel 或优化路径里边 reduce 边切分。

gather：
  某些只需要目标 rank 收完整分片结果的辅助路径。
```

需要避免的误解：

```text
不是所有 TP layer 都立即通信；
ColumnParallelLinear 可以先保持输出分片；
RowParallelLinear 常见需要 all-reduce 合回完整 hidden。
```

---

## 7. Pipeline Parallel：切 Transformer layers

### 7.1 PP 切什么

PP 按 layer 切模型。

典型模型通过 `make_layers()` 决定本 stage 持有哪些层：

```text
start_layer, end_layer = get_pp_indices(
    num_hidden_layers,
    get_pp_group().rank_in_group,
    get_pp_group().world_size,
)
```

位置：`code/vllm/vllm/model_executor/models/utils.py:640`

不属于本 stage 的层用 `PPMissingLayer` 占位。

位置：`code/vllm/vllm/model_executor/models/utils.py:627`

### 7.2 PP stage 的职责

典型职责是：

```text
first PP stage：
  input embedding + 前半层。

middle PP stage：
  中间 layers。

last PP stage：
  后半层 + final norm + lm_head / logits / sampling 所需 hidden states。
```

非 last stage 通常返回：

```text
IntermediateTensors({"hidden_states": ..., "residual": ...})
```

last stage 才继续产生 logits / sampling。

### 7.3 PP 如何通信

PP 通信不是 all-reduce，而是 stage 间传中间状态：

```text
stage 0
  → forward 本 stage layers
  → send IntermediateTensors

stage 1
  → recv IntermediateTensors
  → forward 本 stage layers
```

如果同时有 TP，则 stage 间通常按相同 TP rank 对齐传：

```text
PP group for tp_rank=0: stage0_tp0 → stage1_tp0
PP group for tp_rank=1: stage0_tp1 → stage1_tp1
```

### 7.4 PP 对输出 rank 的影响

多进程 Executor 默认只从：

```text
最后一个 PP stage 的第一个 TP rank
```

收 `ModelRunnerOutput`。

源码注释位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:495`

计算逻辑：

```text
output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:505`

---

## 8. Data Parallel：切请求和 replica

### 8.1 DP 切什么

DP 切的是请求 / batch / replica，而不是单个 layer 的 tensor。

心智模型：

```text
DP replica 0：处理请求 A / C / E
DP replica 1：处理请求 B / D / F
每个 replica 内部再按 TP / PP / PCP / DCP / EP 完成一次 forward。
```

一句话：

```text
DP 是“多个模型副本并行服务不同请求”。
```

### 8.2 DP 和 TP 的区别

最容易混淆的是：

```text
TP：同一个请求由多个 rank 合作算。
DP：不同请求被分给不同 replica 算。
```

例如：

```text
DP = 2, TP = 2

请求 A → replica 0 → TP group [0, 1]
请求 B → replica 1 → TP group [2, 3]
```

请求 A 不会用 `[2,3]` 做 TP all-reduce；请求 B 也不会用 `[0,1]` 做 TP all-reduce。

### 8.3 DP 需要哪些同步

DP 不是完全没有通信。它会参与：

```text
- 是否还有未完成请求的状态同步；
- pause / resume 等控制状态同步；
- KV cache memory size 等跨 replica 一致性计算；
- 某些 DP + MoE 场景下的 expert parallel group 构造；
- external / hybrid load balancing 场景中的控制面协调。
```

例如 `ParallelConfig.sync_dp_state()` 会通过 all-reduce 同步 DP 状态。

位置：`code/vllm/vllm/config/parallel.py:691`

---

## 9. Expert Parallel：切 MoE experts 和 routed tokens

### 9.1 EP 切什么

EP 服务 MoE layer。

MoE forward 的典型链路是：

```text
hidden_states
  → router / gate
  → top-k expert ids + routing weights
  → token permutation / bucket by expert
  → all-to-all dispatch
  → local experts compute
  → all-to-all combine
  → unpermute / weighted combine
  → MoE output
```

一句话：

```text
EP 是“token 根据 router 结果被发到持有目标 expert 的 rank”。
```

### 9.2 EP group 和其他维度的关系

EP 不是 `world_size = TP * PP * DP * EP` 里的独立乘法维度。

vLLM 的 EP group 通常在同一个 PP stage 内合并：

```text
DP x PCP x TP
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1854`

所以：

```text
PP 决定 MoE layer 在哪个 stage；
EP 在该 stage 内组织 expert dispatch；
TP 仍然服务 dense tensor / head 切分；
DP replica 可以参与 EP group，具体取决于 MoE backend 和配置。
```

### 9.3 EP 对 logits / sampling 的影响

EP 只影响 MoE layer 内部 token 到 expert 的流动。

它不改变最终规则：

```text
logits / sampling 仍然由 last PP stage 负责；
Executor 仍然从 output_rank 收 ModelRunnerOutput。
```

---

## 10. Context Parallel：切 prefill / decode 的上下文

### 10.1 PCP 和 DCP 的区别

vLLM 里需要区分：

```text
PCP：prefill context parallel。
  主要服务长 prompt prefill，把 prefill context 相关计算拆到多个 rank。

DCP：decode context parallel。
  主要服务 decode 阶段长 context attention，把 KV context 分担到多个 rank。
```

PCP 乘进 `world_size`：

```text
world_size = PP * TP * PCP
```

DCP 不乘进 `world_size`：

```text
DCP 复用 TP group 内 rank。
```

### 10.2 DCP 为什么需要 LSE merge

decode attention 中，每个 DCP rank 可能只看一段 KV context，得到 partial attention output。

但 softmax 的分母必须覆盖完整 context，所以 backend 需要返回：

```text
partial output
softmax_lse / log-sum-exp
```

合并时不能简单相加，而要做 LSE-weighted merge。

相关源码：

- `merge_attn_states()`：`code/vllm/vllm/v1/attention/ops/triton_merge_attn_states.py:14`
- DCP A2A 说明：`code/vllm/vllm/v1/attention/ops/dcp_alltoall.py:6`

### 10.3 CP 对 attention backend 的要求

不是所有 attention backend 都天然支持 CP / DCP。

需要 backend 支持：

```text
- 分片 context 的 attention 计算；
- 返回 partial output；
- 返回 softmax LSE；
- 和 paged KV cache layout / block table / slot mapping 对齐；
- 按配置选择 AG+RS 或 A2A 等通信路径。
```

---

## 11. KV cache 在并行体系里的归属

KV cache 不属于某一个单独并行维度，而是受多个维度共同影响。

### 11.1 TP 下的 KV cache

TP 切 attention heads / KV heads，所以：

```text
每个 TP rank 持有自己负责的 KV heads；
attention backend 在本 rank 读写本 rank KV cache；
不会先把所有 KV heads gather 到一个 rank 再算 attention。
```

### 11.2 PP 下的 KV cache

PP 切 layers，所以：

```text
每个 PP stage 只持有本 stage attention layers 的 KV cache；
stage 之间传 hidden states / IntermediateTensors，不传整份 KV cache。
```

### 11.3 DP 下的 KV cache

DP 切 replica，所以：

```text
每个 DP replica 有自己的 KV cache；
请求在哪个 replica 执行，它的 KV blocks 就属于哪个 replica；
普通 DP 不自动共享 KV cache。
```

### 11.4 CP / DCP 下的 KV cache

CP / DCP 影响的是：

```text
本轮 attention 如何读取 / 分担 context；
partial attention output 如何合并。
```

它不等价于“KV cache 整体多复制一份”或“KV cache 一定按 DCP 独立持久分片”。具体放置和读取方式要结合 attention backend、KV cache group、block table 和 slot mapping 看。

Worker 初始化 KV cache 的入口是：

```text
GPUWorker.initialize_from_config(kv_cache_config)
  → model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:563`

ModelRunner 内部持有：

```text
self.kv_caches: list[torch.Tensor]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:525`

---

## 12. logits / sampling / output rank

并行最终必须回答：用户侧输出从哪里回来？

### 12.1 PP 决定 logits 所在 stage

通常只有 last PP stage 才持有最终 hidden states、final norm、lm_head 所需输出。

所以：

```text
非 last PP stage：返回或发送 IntermediateTensors；
last PP stage：产生 logits / pooling output / sampling 输入。
```

### 12.2 TP 决定输出 rank 选择

TP ranks 都参与 forward，但 Executor 通常只从一个 rank 收 `ModelRunnerOutput`。

多进程 executor 的规则是：

```text
只从最后一个 PP stage 的第一个 TP rank 返回 ModelRunnerOutput。
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:495`

### 12.3 DP 决定多个 replica 的输出归属

DP 下，每个 replica 处理不同请求。

因此：

```text
每个 replica 都有自己的输出 rank；
上层执行/负载均衡逻辑负责把不同 replica 的请求结果回收到对应 EngineCore / client 路径。
```

---

## 13. 并行维度对照表

| 并行维度 | 切分对象 | 是否乘进 world_size | 典型 group | 典型通信 | 影响的核心模块 |
| --- | --- | --- | --- | --- | --- |
| TP | 单层 tensor / heads / hidden / vocab | 是 | TP group | all-reduce / all-gather / reduce-scatter | linear、attention、MLP、embedding、lm_head |
| PP | Transformer layers / stage | 是 | PP group | send / recv intermediate tensors | model construction、forward、logits stage |
| DP | 请求 / batch / replica | 计入 world_size_across_dp | DP group | 状态同步、控制面协调 | scheduler、executor、worker、KV 隔离 |
| EP | MoE experts / routed tokens | 否 | EP group | all-to-all dispatch / combine | fused_moe、router、expert runner |
| PCP | prefill context | 是 | PCP group | context-parallel attention 通信 | attention prefill backend |
| DCP | decode context | 否 | DCP group | AG+RS 或 A2A、LSE merge | decode attention backend |
| EPLB | expert load balancing | 否 | EPLB group | 负载统计 / rebalance 通信 | MoE load balancing |

---

## 14. 并行和通信原语的关系

通信原语不是并行策略本身，而是切分后的对齐手段。

```text
TP 常用：
  all-reduce / all-gather / reduce-scatter。

PP 常用：
  send / recv intermediate tensors；部分路径还会配合 broadcast 等控制通信。

EP 常用：
  all-to-all dispatch / combine routed tokens。

DCP 常用：
  all-gather + reduce-scatter，或 all-to-all 打包 partial output + LSE。

DP 常用：
  控制面 RPC、状态 all-reduce、跨 replica 协调。
```

要避免的误解：

```text
all-reduce 不是 TP 的同义词；
all-to-all 不是 EP 的同义词；
send/recv 不是 PP 的同义词。

并行策略先定义“切什么”，通信原语只是“切完后怎么对齐”。
```

---

## 15. 多种并行组合时怎么读

组合并行建议按下面顺序拆：

```text
1. 先算 worker 数：PP * TP * PCP，再乘 DP。
2. 写出 all_ranks[external_dp, dp, pp, pcp, tp]。
3. 标出 TP group：同 stage 内 tensor 合作。
4. 标出 PP group：跨 stage 传 intermediate tensors。
5. 标出 DP group：跨 replica 做状态协调。
6. 标出 DCP group：TP 内部切 decode context。
7. 如果是 MoE，标出 EP group：同 PP stage 内 DP * PCP * TP。
8. 再看模型层：dense 走 TP，layers 走 PP，experts 走 EP，attention context 走 PCP/DCP。
9. 最后看输出：last PP stage 的第一个 TP rank。
```

一个压缩图：

```text
global world
  → ExternalDP x DP x PP x PCP x TP rank mesh
  → TP / PP / DP / PCP / DCP / EP / EPLB groups
  → dense layer 用 TP
  → pipeline stage 用 PP
  → request replica 用 DP
  → MoE layer 用 EP
  → prefill/decode attention context 用 PCP / DCP
  → KV cache 按 rank / layer / head / block 归属
  → logits / sampling 在 last PP stage
```

---

## 16. 和本目录专题文档的关系

### 16.1 配置和 group

```text
01_parallel_config_and_topology.md
  解释 ParallelConfig 如何计算 world_size、DP、DCP、EP 约束。

02_distributed_groups.md
  解释 parallel_state.py 如何创建 TP / PP / DP / EP / PCP / DCP / EPLB group。

10_communication_primitives.md
  解释 all-reduce / all-gather / reduce-scatter / all-to-all / send-recv 在 vLLM 中的封装。
```

### 16.2 模型并行维度

```text
03_tensor_parallel.md
  解释 QKV、MLP、embedding、logits 如何按 TP 切。

04_pipeline_parallel.md
  解释 layer placement、PPMissingLayer、IntermediateTensors、stage 输出。

06_expert_parallel.md
  解释 MoE router、expert placement、all-to-all dispatch / combine。

07_sequence_parallel.md
  解释 sequence parallel 在 vLLM 中主要作为 TP 内部的序列维优化路径，以及它和 CP 的边界。

07_context_parallel.md
  解释 PCP / DCP、partial attention output、softmax LSE merge。
```

### 16.3 执行状态和端到端

```text
05_data_parallel.md
  解释请求级 replica 并行、DP 状态同步、输出隔离。

08_kv_cache_and_parallelism.md
  解释 KV cache 在 TP / PP / DP / CP 下的归属。

09_attention_and_parallelism.md
  解释 attention backend 如何感知 TP / CP / GQA / MLA / KV layout。

11_parallel_composition.md
  解释多个并行维度如何组合成 rank mesh。

12_end_to_end_examples.md
  用 TP=2、PP=2、DP=2、MoE+EP、DCP 等案例串完整执行链路。
```

---

## 17. 和其他目录的交叉关系

并行体系会和其他问题目录交叉：

```text
../engine_core/
  EngineCore 如何发起 execute_model 和回收输出。

../scheduler/
  SchedulerOutput、token budget、request state、KV block 分配如何进入执行层。

../executor_worker_model_runner/
  Executor / Worker / ModelRunner 如何承载分布式执行。

../attention/
  attention metadata、backend selection、KV layout、CUDA graph 如何受并行影响。

../kv_cache_transfer/
  external KV transfer、KV connector、prefix cache 如何适配 DP / PP / TP。
```

如果从执行链路看，推荐把两个总览连起来读：

```text
executor_worker_model_runner_overview.md
  → parallelism_overview.md
  → 12_end_to_end_examples.md
```

前者解释“执行层怎么跑”，后者解释“多个 rank 怎么一起跑”。

---

## 18. 最小心智模型

可以先把 vLLM V1 并行理解成下面这张图：

```text
EngineCore / Scheduler
  → SchedulerOutput
  → Executor
  → Worker ranks
      ├── DP replica 0
      │     ├── PP stage 0
      │     │     └── TP ranks / PCP ranks / DCP groups
      │     └── PP stage 1
      │           └── TP ranks / PCP ranks / DCP groups
      │
      └── DP replica 1
            ├── PP stage 0
            └── PP stage 1

MoE layer:
  在某个 PP stage 内额外使用 EP group 做 token ↔ expert dispatch。

Attention:
  TP 切 heads，PCP / DCP 切 context，backend 负责 partial state merge。

KV cache:
  按 worker / layer / head / block / slot 存放，随 TP / PP / DP / CP 改变访问方式。

Output:
  last PP stage 的第一个 TP rank 返回 ModelRunnerOutput。
```

最终落回一句话：

```text
每种并行策略都必须同时回答：切什么、谁通信、怎么通信、状态放哪、结果在哪合并。
```

---

## 19. 排查并行问题的检查表

遇到并行相关问题时，按这个顺序查：

```text
1. ParallelConfig 中 TP / PP / DP / PCP / DCP / EP size 是否符合约束。
2. world_size 和 world_size_across_dp 是否符合预期。
3. initialize_model_parallel() 生成的 group 是否和预期 rank mesh 一致。
4. 当前 rank 的 TP / PP / DP / PCP / DCP / EP rank_in_group 是否正确。
5. 模型 layers 是否被正确放到 PP stage，缺失层是否是 PPMissingLayer。
6. QKV / MLP / embedding / lm_head 是否按 TP 分片加载。
7. attention backend 是否支持当前 TP / CP / DCP / MLA / KV dtype 组合。
8. KV cache config、block table、slot mapping 是否在每个 rank 上对齐。
9. MoE all-to-all backend、EP group、expert placement 是否匹配。
10. Executor output_rank 是否指向 last PP stage 的第一个 TP rank。
11. DP 下是否所有 replica 都进入了需要 collective 的路径，避免死锁。
12. 如果开启 DCP A2A，确认 dcp_size > 1 且 tp_size 可被 dcp_size 整除。
```

这份检查表能覆盖大多数并行配置错误、rank group 错误、通信死锁、KV cache 不一致和输出 rank 错误。
