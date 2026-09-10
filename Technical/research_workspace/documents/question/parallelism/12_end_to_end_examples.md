# 12. 几个典型并行配置如何完整执行？

源码位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/communication_op.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/models/utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/linear.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/attention/attention.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/logits_processor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/fused_moe/`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/`

本问题关注：通过几个具体配置，把前面 TP / PP / DP / EP / PCP / DCP、通信原语、KV cache、attention backend、Executor / Worker / ModelRunner 串成完整执行链路，用端到端案例检验并行体系是否闭环。

---

## 1. 一句话回答

端到端案例的目标是把抽象并行概念落到一轮执行。

最小主链路是：

```text
请求进入 EngineCore
  → Scheduler.schedule() 生成 SchedulerOutput
  → Executor.execute_model(scheduler_output)
  → 所有相关 Worker 同步进入 execute_model
  → GPUModelRunner 更新请求状态、准备输入和 attention metadata
  → 模型 forward
  → TP / PP / DP / EP / CP 在各自模块里触发通信
  → last PP stage 的输出 rank 生成 logits / sampling
  → ModelRunnerOutput 回到 Executor
  → Scheduler.update_from_output() 回收状态并返回 EngineCoreOutputs
```

所以读一个并行 case 时，必须同时回答八个问题：

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

---

## 2. 端到端执行前的共同背景

不管是哪种并行配置，vLLM V1 的执行入口基本一致。

### 2.1 EngineCore 只看 SchedulerOutput 和 ModelRunnerOutput

主循环在 `EngineCore.step()` 中：

```text
Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output, non_block=True)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → future.result()
  → 如有需要，再 model_executor.sample_tokens(grammar_output)
  → scheduler.update_from_output(scheduler_output, model_output)
```

位置：`code/vllm/vllm/v1/engine/core.py:499`

这说明：

```text
SchedulerOutput 是本轮执行计划；
ModelRunnerOutput 是本轮模型侧结果；
并行细节都被 Executor / Worker / ModelRunner / distributed group 吸收。
```

### 2.2 Executor 会把同一个 SchedulerOutput 发给相关 Worker

抽象层的 `execute_model()` 做的是 collective RPC：

```text
Executor.execute_model(scheduler_output)
  → collective_rpc("execute_model", args=(scheduler_output,))
```

位置：`code/vllm/vllm/v1/executor/abstract.py:221`

多进程 executor 中也一样，只是它通过 message queue 广播命令，并且只从一个 rank 收用户侧输出：

```text
MultiprocExecutor.execute_model()
  → collective_rpc(
        "execute_model",
        unique_reply_rank=self.output_rank,
        kv_output_aggregator=...
    )
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:307`

`output_rank` 的规则是：

```text
只从最后一个 PP stage 的第一个 TP rank 返回 ModelRunnerOutput。
```

源码注释：`code/vllm/vllm/v1/executor/multiproc_executor.py:499`

计算逻辑：

```text
output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:508`

### 2.3 ModelRunner 是并行真正落地的位置

`GPUModelRunner.execute_model()` 的内部链路可以抽象成：

```text
_update_states(scheduler_output)
  → _prepare_inputs(...)
  → _build_attention_metadata(...)
  → _preprocess(...)
  → _model_forward(...)
  → logits / pooling / IntermediateTensors
  → sample_tokens(...)
  → ModelRunnerOutput 或 None
```

相关位置：

- `_update_states()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1127`
- `_prepare_inputs()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1889`
- `execute_model_state`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:897`

并行维度不是在这里统一 `if TP/PP/EP` 判断，而是分散在不同模块：

```text
TP：parallel linear / vocab parallel / logits processor 查询 TP group；
PP：模型层构造、intermediate_tensors、PP group send/recv；
DP：Executor / scheduler / worker 以 replica 为边界处理不同请求；
EP：MoE layer 内 router、token dispatch、expert compute、combine；
PCP / DCP：attention backend 和 context parallel group；
KV cache：Worker 初始化 cache，ModelRunner 按 layer / block table / slot mapping 读写。
```

---

## 3. rank mesh：所有 case 的共同底座

`initialize_model_parallel()` 中 rank mesh 的核心形状是：

```text
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1788`

可以理解为：

```text
all_ranks[external_dp, dp, pp, pcp, tp]
```

各通信组从这个 mesh 派生：

```text
TP group  ：固定 external_dp / dp / pp / pcp，沿 tp 维取 rank
DCP group ：复用 TP 相关 rank 按 decode_context_parallel_size 成组
PCP group ：固定 external_dp / dp / pp / tp，沿 pcp 维取 rank
PP group  ：固定 external_dp / dp / pcp / tp，沿 pp 维取 rank
DP group  ：固定 external_dp / pp / pcp / tp，沿 dp 维取 rank
EP group  ：MoE 模型中固定 external_dp / pp，合并 dp * pcp * tp
```

对应位置：

- TP group：`code/vllm/vllm/distributed/parallel_state.py:1757`
- DCP group：`code/vllm/vllm/distributed/parallel_state.py:1774`
- PCP group：`code/vllm/vllm/distributed/parallel_state.py:1796`
- PP group：`code/vllm/vllm/distributed/parallel_state.py:1815`
- DP group：`code/vllm/vllm/distributed/parallel_state.py:1833`
- EP group：`code/vllm/vllm/distributed/parallel_state.py:1850`

需要特别记住：

```text
DCP 不增加 world_size，而是在 TP group 内复用 rank；
EP 不增加 world_size，而是在 MoE layer 所在 PP stage 内复用 DP / PCP / TP rank；
PP 和 TP 会直接改变单个 replica 的 worker 数；
DP 会复制模型并处理不同请求。
```

---

## 4. Case 1：TP=2，PP=1，DP=1

配置：

```text
TP = 2
PP = 1
DP = 1
PCP = 1
DCP = 1
EP = off
```

### 4.1 rank 拓扑

world size：

```text
world_size = PP * PCP * TP = 1 * 1 * 2 = 2
```

rank mesh：

```text
all_ranks[0, 0, 0, 0, :] = [0, 1]
```

通信组：

```text
TP group = [0, 1]
PP group = [0] for tp_rank=0, [1] for tp_rank=1
DP group = [0] / [1]，实际没有跨 replica
```

### 4.2 请求分到哪些 rank

一个请求会被同一个 TP group 的两个 rank 共同执行。

执行链路：

```text
EngineCore
  → SchedulerOutput
  → Executor.collective_rpc("execute_model")
  → rank 0 Worker.execute_model()
  → rank 1 Worker.execute_model()
  → 两个 rank 使用同一批请求的调度信息
```

这里不是“请求 A 给 rank0、请求 B 给 rank1”，而是：

```text
请求 A 的同一批 token 同时进入 rank0 和 rank1；
rank0 / rank1 分别持有部分权重、部分 attention heads、部分中间结果；
必要位置通过 TP group 通信合成等价的完整层输出。
```

### 4.3 QKV projection 如何分片

QKV 用 `QKVParallelLinear`，它继承 `ColumnParallelLinear`。

关键规则：

```text
QKV 权重沿输出维 / head 维切分；
每个 TP rank 只产生自己的 query heads 和 kv heads；
gather_output=False，所以 QKV projection 后不立刻 all-gather。
```

源码依据：

- QKV 按 head 维并行：`code/vllm/vllm/model_executor/layers/linear.py:914`
- `num_heads = total_num_heads / tp_size`：`code/vllm/vllm/model_executor/layers/linear.py:966`
- KV heads 少于 TP size 时会复制 KV head：`code/vllm/vllm/model_executor/layers/linear.py:968`
- `gather_output=False`：`code/vllm/vllm/model_executor/layers/linear.py:986`

例如：

```text
total_num_heads = 32
total_num_kv_heads = 8
TP = 2

rank0: 16 个 query heads + 4 个 KV heads
rank1: 16 个 query heads + 4 个 KV heads
```

如果是 MQA / GQA 且 `total_num_kv_heads < tp_size`，则每个 TP rank 可能复制同一个 KV head，而不是继续切更小的 KV head。

### 4.4 attention 如何执行

attention 发生在每个 TP rank 的本地 head 分片上：

```text
rank0: 用本 rank 的 q/k/v heads 读写本 rank 的 KV cache
rank1: 用本 rank 的 q/k/v heads 读写本 rank 的 KV cache
```

注意：

```text
TP 下 attention 不需要把所有 heads gather 到一个 rank 再算；
head 维本身就是天然可分片的；
每个 rank 的 KV cache 也只对应本 rank 持有的 KV heads。
```

### 4.5 MLP 哪里发生 all-reduce

典型 Transformer MLP 是：

```text
ColumnParallelLinear / MergedColumnParallelLinear
  → 激活函数
  → RowParallelLinear
  → tensor_model_parallel_all_reduce
```

直觉是：

```text
第一段按输出维切开，不需要马上聚合；
第二段按输入维切开，每个 rank 算出 partial output；
要得到完整 hidden states，需要跨 TP group all-reduce。
```

通信 helper 在：

```text
tensor_model_parallel_all_reduce()
  → get_tp_group().all_reduce(...)
```

位置：`code/vllm/vllm/distributed/communication_op.py:12`

### 4.6 logits / sampling 在哪里

TP=2、PP=1 时，last PP stage 就是唯一 PP stage。

多进程 Executor 默认只从：

```text
PP last stage 的第一个 TP rank
```

收 `ModelRunnerOutput`。

对于本例：

```text
output_rank = 2 - 2 * 1 = 0
```

所以输出由 rank0 返回给 Executor。

但这不表示 rank1 没参与 logits 前的计算。rank1 参与了所有 TP forward 通信，只是最终用户侧 `ModelRunnerOutput` 不从它那里收。

### 4.7 Case 1 总链路

```text
请求 A
  → SchedulerOutput
  → rank0 / rank1 同时执行
  → QKVParallelLinear：rank0 / rank1 分别算 head 分片
  → attention：各自读写本地 KV heads
  → MLP：局部分片计算 + TP all-reduce
  → hidden states 在两个 TP rank 上保持一致或按模块约定分片
  → logits / sampling 在 last PP stage 上执行
  → Executor 只取 output_rank=0 的 ModelRunnerOutput
  → Scheduler.update_from_output()
```

---

## 5. Case 2：TP=2，PP=2，DP=1

配置：

```text
TP = 2
PP = 2
DP = 1
PCP = 1
DCP = 1
```

### 5.1 rank 拓扑

world size：

```text
world_size = PP * PCP * TP = 2 * 1 * 2 = 4
```

rank mesh：

```text
PP stage 0, TP group = [0, 1]
PP stage 1, TP group = [2, 3]

PP group for tp_rank=0 = [0, 2]
PP group for tp_rank=1 = [1, 3]
```

这个结构意味着：

```text
stage 内：TP 通信；
stage 间：同一个 tp_rank 组成 PP group，传 hidden states / intermediate_tensors。
```

### 5.2 模型 layers 如何分成两个 stage

多数 decoder-only 模型通过 `make_layers()` 创建层列表。

核心逻辑：

```text
start_layer, end_layer = get_pp_indices(
    num_hidden_layers,
    get_pp_group().rank_in_group,
    get_pp_group().world_size,
)

modules =
  [PPMissingLayer() for 0..start_layer)
  + 本 stage 真正持有的 layers
  + [PPMissingLayer() for end_layer..num_hidden_layers)
```

位置：`code/vllm/vllm/model_executor/models/utils.py:640`

`PPMissingLayer` 是占位层：

```text
不持有真实参数；
forward 时直接返回输入；
用于让不同 PP stage 的模块命名和层号保持一致。
```

位置：`code/vllm/vllm/model_executor/models/utils.py:627`

例如 32 层、PP=2：

```text
stage 0: 持有 layer 0..15，layer 16..31 是 PPMissingLayer
stage 1: layer 0..15 是 PPMissingLayer，持有 layer 16..31
```

### 5.3 embedding / norm / lm_head 属于哪个 stage

典型规则：

```text
first PP rank：持有 input embedding；
last PP rank ：持有 final norm / lm_head / logits processor 需要的最后输出；
中间 stage：只持有自己负责的 decoder layers。
```

很多模型中可以看到：

```text
if get_pp_group().is_first_rank:
    self.embed_tokens = ...
else:
    self.embed_tokens = PPMissingLayer()

if get_pp_group().is_last_rank:
    self.norm = ...
else:
    self.norm = PPMissingLayer()
```

示例位置：`code/vllm/vllm/model_executor/models/afmoe.py:394`

### 5.4 stage 0 如何把 hidden states 发给 stage 1

PP forward 的语义是：

```text
stage 0:
  input_ids / inputs_embeds
  → embedding
  → 前半 layers
  → 返回 IntermediateTensors({"hidden_states": ..., "residual": ...})

stage 1:
  接收 intermediate_tensors
  → 取 hidden_states / residual
  → 后半 layers
  → norm / logits / sampling
```

模型 forward 示例中，非 first rank 会要求 `intermediate_tensors is not None`，并从中取 `hidden_states` 和 `residual`。

示例位置：`code/vllm/vllm/model_executor/models/afmoe.py:432`

非 last rank 返回 `IntermediateTensors`：

```text
if not get_pp_group().is_last_rank:
    return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
```

示例位置：`code/vllm/vllm/model_executor/models/afmoe.py:457`

在 ModelRunner 侧也有缓存字段：

```text
self.intermediate_tensors: IntermediateTensors | None = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:791`

### 5.5 每个 stage 内部的 TP=2 如何通信

每个 PP stage 内都有自己的 TP group。

```text
stage 0 TP group [0, 1]：只负责前半层中的 TP linear / attention heads；
stage 1 TP group [2, 3]：只负责后半层中的 TP linear / attention heads。
```

因此通信边界很清楚：

```text
TP all-reduce：只在同一个 stage 内发生；
PP send/recv：只在同一个 tp_rank 对应的 stage 间发生。
```

不会发生：

```text
rank0 和 rank3 直接做 TP all-reduce；
rank0 把自己的 hidden states 发给 stage1 的所有 TP rank；
```

通常是：

```text
rank0 → rank2
rank1 → rank3
```

因为它们在同一个 PP group 中。

### 5.6 哪个 stage 负责 logits / sampling

last PP stage 负责最终 hidden states、logits 和 sampling。

Executor 只从 last PP stage 的第一个 TP rank 收输出。

本例：

```text
world_size = 4
tp_size = 2
pcp_size = 1
output_rank = 4 - 2 * 1 = 2
```

所以：

```text
rank2 是输出 rank；
rank3 参与 last stage 的 TP forward，但不作为 Executor 的用户侧输出来源。
```

### 5.7 Case 2 总链路

```text
请求 A
  → SchedulerOutput 广播到 rank0/1/2/3

stage 0, TP group [0,1]
  → rank0/1 embedding + 前半 layers
  → 每层内部 TP all-reduce
  → 产生 IntermediateTensors
  → PP group [0,2] / [1,3] 发送到 stage 1

stage 1, TP group [2,3]
  → rank2/3 接收 IntermediateTensors
  → 后半 layers
  → 每层内部 TP all-reduce
  → final norm / logits / sampling
  → rank2 返回 ModelRunnerOutput

EngineCore
  → Scheduler.update_from_output()
```

---

## 6. Case 3：DP=2，TP=2，PP=1

配置：

```text
DP = 2
TP = 2
PP = 1
PCP = 1
DCP = 1
```

### 6.1 rank 拓扑

如果把 DP 算进全局 worker：

```text
world_size_across_dp = DP * PP * PCP * TP = 2 * 1 * 1 * 2 = 4
```

rank mesh 可以理解为：

```text
DP replica 0, TP group = [0, 1]
DP replica 1, TP group = [2, 3]

DP group for tp_rank=0 = [0, 2]
DP group for tp_rank=1 = [1, 3]
```

注意两种 group 的语义完全不同：

```text
TP group：同一个请求的一次 forward 内部合作；
DP group：不同 replica 之间做调度/状态/一致性协调，不用于把请求 A 的 hidden states 算完整。
```

### 6.2 两个 DP replica 如何处理不同请求

DP 的核心是复制模型副本并处理不同请求。

```text
请求 A → DP replica 0 → rank0/1 共同执行
请求 B → DP replica 1 → rank2/3 共同执行
```

所以：

```text
请求 A 不需要和 rank2/3 做 TP 通信；
请求 B 不需要和 rank0/1 做 TP 通信；
两个请求的 attention / KV cache / logits 都是 replica-local。
```

### 6.3 每个 replica 内部的 TP group 是否独立

是。

```text
replica 0: TP group [0, 1]
replica 1: TP group [2, 3]
```

请求 A 的 QKV、attention、MLP all-reduce 只发生在 `[0,1]`；请求 B 的对应通信只发生在 `[2,3]`。

### 6.4 DP group 什么时候会参与

DP group 不用于普通 dense layer 的矩阵分片合并。它更多用于：

```text
1. 多 replica 调度协调；
2. 某些 cudagraph / padding / batch shape 决策同步；
3. 外部 DP / 负载均衡 / EPLB 等场景；
4. MoE EP 组构造时把 DP 维并入 expert parallel group。
```

因此在读端到端链路时要避免把 DP 和 TP 混淆：

```text
TP 是“一个请求拆开算”；
DP 是“多个模型副本分别算不同请求”。
```

### 6.5 KV cache 是否 replica-local

是。

每个 worker 会在自己的设备上初始化 KV cache。Worker 初始化 cache 的入口是：

```text
GPUWorker.initialize_from_config(kv_cache_config)
  → model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:563`

因此：

```text
请求 A 的 KV blocks 属于 replica 0 的 rank0/1；
请求 B 的 KV blocks 属于 replica 1 的 rank2/3；
两个 replica 的 KV cache 不共享。
```

如果请求迁移到另一个 replica，需要依赖更高层的调度或 KV transfer 机制，而不是普通 DP 自动共享 KV cache。

### 6.6 logits / sampling 在哪里

每个 DP replica 都有自己的 last PP stage 和输出 rank。

本例每个 replica 内：

```text
PP=1，TP=2
输出来自该 replica 内第一个 TP rank
```

所以可以理解为：

```text
请求 A：rank0 返回输出；
请求 B：rank2 返回输出。
```

具体是否由一个主进程聚合多个 DP replica 的结果，取决于 DP 部署方式和 executor / frontend 组织方式；但模型执行语义上，两个 replica 是独立执行不同请求。

### 6.7 Case 3 总链路

```text
请求 A
  → DP replica 0
  → rank0/1 TP forward
  → replica 0 本地 KV cache
  → rank0 输出 ModelRunnerOutput(A)

请求 B
  → DP replica 1
  → rank2/3 TP forward
  → replica 1 本地 KV cache
  → rank2 输出 ModelRunnerOutput(B)

请求 A 和请求 B
  → 不互相做 TP all-reduce
  → 不共享 KV cache
  → 只在必要的 DP 协调点有跨 replica 通信
```

---

## 7. Case 4：MoE + EP

配置示例：

```text
MoE model
TP = 2
PP = 1
DP = 2
EP = on
```

### 7.1 EP group 如何从 rank mesh 里来

EP 不额外增加 worker。

`initialize_model_parallel()` 中，MoE 模型会创建 EP group：

```text
固定 external_dp / pp，合并 data_parallel_size * pcp_size * tensor_parallel_size
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1850`

也就是说，EP group 通常横跨：

```text
同一个 PP stage 内的 DP x PCP x TP rank
```

例如：

```text
DP = 2
TP = 2
PP = 1
PCP = 1

EP group = [0, 1, 2, 3]
```

如果 PP=2，则每个 PP stage 各自有 EP group：

```text
stage 0 的 MoE layers 用 stage 0 的 EP group；
stage 1 的 MoE layers 用 stage 1 的 EP group；
不同 PP stage 不混在同一个 MoE forward 里做 EP dispatch。
```

### 7.2 router 在哪里产生 expert ids

MoE layer 内部会先通过 router / gate 得到每个 token 的 top-k expert：

```text
hidden_states
  → router / gate linear
  → routing logits
  → top-k expert ids + routing weights
```

这一步通常发生在每个 rank 本地。随后，EP dispatch 根据 expert ids 把 token 送到持有对应 expert 的 rank。

### 7.3 token 如何按 expert 发送到不同 rank

MoE EP 的核心链路是：

```text
hidden_states
  → top-k routing
  → token permutation / bucket by expert
  → all-to-all dispatch
  → local experts compute
  → all-to-all combine
  → unpermute
  → weighted combine
  → MoE output
```

源码目录：`code/vllm/vllm/model_executor/layers/fused_moe/`

其中 all-to-all 的语义是：

```text
每个 rank 一开始持有一批 token；
router 决定 token 应该去哪些 expert；
如果 expert 不在本 rank，就通过 EP group 发给目标 rank；
目标 rank 对本地 expert 的 token 做 MLP；
输出再按原 token / top-k 路径送回并合并。
```

### 7.4 all-to-all 的输入输出是什么

可以用逻辑张量理解：

输入：

```text
本 rank 当前持有的 token hidden_states
每个 token 的 top-k expert_ids
每个 token 的 routing_weights
```

dispatch 后：

```text
本 rank 收到“属于本 rank local experts”的 token bucket
```

local expert compute 后：

```text
本 rank 得到这些 token 在对应 expert 上的 expert output
```

combine 后：

```text
每个 rank 重新拿回自己原始 token 的 MoE output
并按 routing_weights 对 top-k expert outputs 做加权合并
```

### 7.5 EP 和 TP 同时存在时如何分层理解

TP 和 EP 是不同层级的并行：

```text
dense attention / dense MLP：主要走 TP group；
MoE expert dispatch / combine：走 EP group；
启用 EP 的 MoE 层里，expert 权重通常按 rank 分配为完整 experts 的子集，而不是继续把单个 expert 矩阵按 TP 切。
```

心智模型：

```text
TP 解决“一个 dense 矩阵太大，按 tensor/head 切”；
EP 解决“专家很多，token 动态路由到不同 expert rank”；
DP 解决“请求吞吐，多个 replica 并行服务”；
PP 解决“层太多，按层切 stage”。
```

### 7.6 logits / sampling 在哪里

MoE layer 只是模型中间层的一种 FFN 实现。

所以：

```text
MoE output 会回到正常 Transformer residual 流；
后续层继续执行；
最终仍然只有 last PP stage 负责 logits / sampling；
Executor 仍然按 output_rank 规则收 ModelRunnerOutput。
```

EP 不改变最终输出 rank 的选择。

### 7.7 Case 4 总链路

```text
请求 A / B
  → 各自进入对应 DP replica / TP group
  → dense attention：TP head 分片 + TP 通信
  → 到 MoE layer
      → router top-k
      → EP group all-to-all dispatch token
      → local experts compute
      → EP group all-to-all combine output
      → unpermute / weighted combine
  → 后续 dense / MoE layers
  → last PP stage logits / sampling
  → output_rank 返回 ModelRunnerOutput
```

---

## 8. Case 5：DCP + FlashAttention / paged attention

配置示例：

```text
TP = 4
DCP = 2
PP = 1
DP = 1
```

### 8.1 DCP rank 拓扑

DCP 不增加 world size。

源码注释明确说明：

```text
DCP must not exceed TP size;
DCP reuses GPUs of TP group;
it splits one TP group into tp_size // dcp_size DCP groups.
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1774`

例如：

```text
TP group = [0, 1, 2, 3]
DCP = 2

DCP groups 可以理解为：
[0, 1]
[2, 3]
```

也就是说：

```text
TP 决定 head / tensor 分片；
DCP 复用 TP 相关 rank，让 decode attention 的 context 计算分摊；
world_size 仍然是 TP * PP * PCP，不乘 DCP。
```

### 8.2 DCP 要解决什么问题

decode 阶段通常是：

```text
query 很短，KV context 很长。
```

如果只让单个 rank 读完整 context，长上下文 decode 会被 KV 读取和 attention 计算拖慢。

DCP 的目标是：

```text
多个 DCP rank 分担同一个 decode attention 的 KV context；
每个 rank 对一部分 context 算 partial attention output；
再用数值稳定的方式合并 partial outputs。
```

### 8.3 partial attention output 是什么

每个 DCP rank 计算的是：

```text
针对本 rank 负责的 KV context 分片，query 对这部分 KV 的 attention output。
```

它不是完整 attention output，因为 softmax 的归一化分母需要看所有 context 分片。

因此 backend 需要返回两类信息：

```text
partial output
softmax LSE / log-sum-exp
```

LSE 用来稳定合并不同 context 分片的 softmax 结果。

### 8.4 为什么要返回 softmax LSE

假设 context 被切成 prefix / suffix 两段：

```text
output_prefix = softmax(qk_prefix) v_prefix
lse_prefix = logsumexp(qk_prefix)

output_suffix = softmax(qk_suffix) v_suffix
lse_suffix = logsumexp(qk_suffix)
```

完整 attention 不是简单平均：

```text
output != output_prefix + output_suffix
```

而是要按全局 softmax 分母重新加权：

```text
global_lse = log(exp(lse_prefix) + exp(lse_suffix))
output = output_prefix * exp(lse_prefix - global_lse)
       + output_suffix * exp(lse_suffix - global_lse)
```

这就是 `merge_attn_states` 这类算子的意义。

源码：

- `merge_attn_states()`：`code/vllm/vllm/v1/attention/ops/triton_merge_attn_states.py:14`
- kernel 中计算 `max_lse` / `out_lse`：`code/vllm/vllm/v1/attention/ops/triton_merge_attn_states.py:134`

### 8.5 DCP 通信如何发生

当前代码里 DCP 有两种常见理解路径：

```text
AG + RS：
  all-gather query 或中间状态
  各 rank 算 partial output
  reduce-scatter / combine 回每个 rank 需要的输出

A2A：
  把 partial output 和 LSE 打包
  all-to-all 交换
  每个 rank 本地做 LSE-weighted combine
```

A2A 相关源码说明：

```text
A2A exchanges partial attention outputs and their LSE values across ranks,
then combines them with exact LSE-weighted reduction.
```

位置：`code/vllm/vllm/v1/attention/ops/dcp_alltoall.py:6`

### 8.6 哪些 backend 支持这个路径

需要 backend 能提供：

```text
1. partial attention output；
2. softmax_lse；
3. 和 KV cache / paged layout 兼容的 decode attention；
4. DCP 通信 backend 需要的张量布局。
```

例如 FlashMLA op 会返回：

```text
out, softmax_lse
```

位置：`code/vllm/vllm/v1/attention/ops/flashmla.py:140`

普通 FlashAttention / paged attention backend 是否走某条 DCP 路径，要看具体 backend 的 capability 和配置选择，不能只看 `DCP > 1` 就认为所有 attention backend 都支持。

### 8.7 Case 5 总链路

```text
decode query / KV cache
  → TP rank 持有自己的 heads / KV heads
  → DCP group 内多个 rank 分担 context
  → attention backend 返回 partial output + softmax_lse
  → DCP 通信：AG+RS 或 A2A
  → LSE-weighted combine
  → 得到等价完整 attention output
  → 后续 MLP / logits / sampling
```

---

## 9. Case 6：PP + KV cache

配置示例：

```text
PP = 2
TP = 2
DP = 1
```

### 9.1 每个 PP stage 是否都有 KV cache

是，但每个 stage 的 KV cache 只服务于本 stage 持有的 attention layers。

因为 PP 是按 layer 切模型：

```text
stage 0 持有前半 attention layers；
stage 1 持有后半 attention layers。
```

KV cache 也自然按 layer 绑定：

```text
stage 0 的 KV cache：layer 0..15 的 K/V；
stage 1 的 KV cache：layer 16..31 的 K/V。
```

不会出现：

```text
stage 0 替 stage 1 的 attention layer 写 KV cache；
stage 1 持有 stage 0 layer 的完整 KV cache。
```

### 9.2 KV cache 何时初始化

Worker 初始化 KV cache 的入口是：

```text
GPUWorker.initialize_from_config(kv_cache_config)
  → ensure_kv_transfer_initialized(...)
  → model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:563`

ModelRunner 内部有：

```text
self.kv_caches: list[torch.Tensor] = []
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:525`

所以每个 worker 进程 / 设备都有自己的 cache 张量集合。

### 9.3 block_table / slot_mapping 是否对所有 stage 一样

调度层给的是同一批请求的 block 分配和 token 调度信息，因此不同 PP stage 看到的是同一轮请求计划。

但写入的实际 KV cache layer 不同：

```text
同一个 token 在所有 PP stage 都有相同的逻辑 request / token 位置；
每个 stage 用这套位置映射写自己负责 layers 的 KV cache；
block_table / slot_mapping 是“请求 token 到 cache block/slot”的映射；
KV cache tensor 是“本 rank、本 layer、本 head 分片”的实际存储。
```

因此：

```text
block_table / slot_mapping 是跨 stage 对齐请求位置的工具；
KV cache 内容不是跨 stage 复制的一整份完整模型 KV。
```

### 9.4 PP stage 如何更新 token 状态

PP 下非 last rank 没有采样结果，所以它要通过调度器或 GPU broadcast 得到新 token。

`_update_states()` 中有相关逻辑：

```text
is_last_rank = get_pp_group().is_last_rank

if not is_last_rank:
    如果没有 req_data.new_token_ids：
        Async scheduled PP 下 sampled tokens 通过 GPU broadcast 传播；
    否则：
        Scheduler 把 sampled token ids 发回，因为 first-stage worker 和 last-stage worker 没有直接通信。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1262`

这说明：

```text
last PP stage 负责采样；
非 last PP stage 仍然需要知道新 token，才能在下一轮 decode 为自己负责的前半层构造输入和更新 KV cache。
```

### 9.5 非 last stage 是否需要 logits

不需要。

非 last stage 的 forward 输出是 `IntermediateTensors`，不是 logits。

```text
stage 0:
  hidden_states / residual
  → IntermediateTensors

stage 1:
  IntermediateTensors
  → 后半 layers
  → logits / sampling
```

这也是为什么 Executor 只从 last PP stage 的第一个 TP rank 收输出。

### 9.6 Case 6 总链路

```text
请求 A
  → SchedulerOutput 广播到所有 PP/TP ranks

PP stage 0
  → 用 input_ids / sampled token 更新本地 batch 状态
  → 写 layer 0..15 的 KV cache
  → forward 前半 layers
  → 发送 IntermediateTensors
  → 不产出 logits

PP stage 1
  → 接收 IntermediateTensors
  → 写 layer 16..31 的 KV cache
  → forward 后半 layers
  → logits / sampling
  → sampled token 回传给 Scheduler / 非 last stage 状态

下一轮 decode
  → 所有 stage 都知道新增 token
  → 各自继续维护自己 layers 的 KV cache
```

---

## 10. 六个 case 放在一起对比

| Case | rank 合作边界 | 权重分布 | KV cache 分布 | 主要通信 | 输出 rank |
| --- | --- | --- | --- | --- | --- |
| TP=2 | 同一请求跨两个 TP rank | head / tensor 分片 | 每个 TP rank 持有本地 heads 的 KV | TP all-reduce / all-gather | PP last stage 的 TP rank0 |
| TP=2, PP=2 | stage 内 TP，stage 间 PP | layers 按 PP 切，每 stage 内再 TP 切 | 每 stage 只持有本 stage layers 的 KV | TP 通信 + PP send/recv | last PP stage 的 TP rank0 |
| DP=2, TP=2 | 每个 DP replica 独立处理请求 | 每个 replica 一份模型，内部 TP 切 | replica-local | replica 内 TP，必要 DP 协调 | 每个 replica 的输出 rank |
| MoE + EP | token 按 expert 跨 EP rank | dense 走 TP，experts 按 EP/runner 分布 | attention KV 仍按 layer/head/rank | EP all-to-all + TP 通信 | 不变，仍看 PP/TP |
| DCP + attention | decode context 在 DCP group 分担 | 不改变权重分布 | CP-aware slot mapping 让每个 rank 只读写本地 context KV 分片 | DCP AG+RS 或 A2A + LSE merge | 不变，仍看 PP/TP |
| PP + KV cache | layers 按 stage 分段 | 每 stage 只持有部分 layers | 每 stage 只缓存本 stage attention layers | PP intermediate + token 状态同步 | last PP stage |

---

## 11. 读端到端链路时最容易踩的坑

### 11.1 不要把 TP 和 DP 混成“都在分请求”

```text
TP：同一个请求拆到多个 rank 合作算；
DP：不同请求分给不同 replica 独立算。
```

### 11.2 不要以为 PP stage 之间共享完整 KV cache

```text
PP stage 只持有自己 layers 的 KV cache；
跨 stage 传的是 hidden states / intermediate_tensors，不是 KV cache。
```

### 11.3 不要以为 DCP 会增加 worker 数

```text
DCP 复用 TP 相关 rank；
DCP size 不能超过 TP size；
它改变的是 decode attention context 计算方式，不是模型层数或权重切分方式。
```

### 11.4 不要以为 MoE EP 改变最终 logits 所在 rank

```text
EP 只影响 MoE layer 内 token 到 expert 的 dispatch / combine；
最终 logits / sampling 仍由 last PP stage 的输出 rank 负责。
```

### 11.5 不要以为所有 rank 都返回 ModelRunnerOutput

Executor 通常只收一个用户侧输出 rank：

```text
最后一个 PP stage 的第一个 TP rank。
```

其他 rank 参与 forward 和通信，但不作为最终输出来源。

---

## 12. 最终检查表

分析一个新的并行配置时，按下面顺序梳理最稳：

```text
1. 计算 world_size = PP * PCP * TP，以及 world_size_across_dp。
2. 写出 all_ranks[external_dp, dp, pp, pcp, tp]。
3. 标出 TP / PP / DP / PCP / DCP / EP group。
4. 判断请求属于哪个 DP replica。
5. 判断单个请求会经过哪些 PP stage 和 TP rank。
6. 标出每个 PP stage 持有哪些 layers。
7. 标出每个 rank 持有哪些权重分片和 KV heads。
8. 对每层 forward 标注 TP 通信、PP send/recv、EP all-to-all、CP merge。
9. 判断 logits / sampling 是否只在 last PP stage。
10. 用 Executor output_rank 规则确认哪个 rank 返回 ModelRunnerOutput。
```

如果这十步都能写清楚，一个并行配置的端到端执行链路基本就闭环了。
