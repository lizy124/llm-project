# 10 parallelism 背诵文档

## 1. 专题定位

`parallelism` 讲的是 vLLM V1 如何在多 GPU / 多进程 / 多 rank 上切分模型、请求、layer、expert、context 和 KV cache。

它不是单纯讲 all-reduce，也不是单纯讲多卡启动。

一句话：

```text
vLLM 并行体系 = 切分对象 + rank group + 通信原语 + 状态归属 + 输出合并。
```

## 2. 最小心智模型

主链路：

```text
ParallelConfig
  → world_size / rank mesh
  → parallel_state 创建 TP / PP / DP / EP / PCP / DCP group
  → Worker 初始化 distributed environment
  → 模型层按 group 切分权重 / heads / layers / experts / context
  → forward 中插入 all-reduce / all-gather / all-to-all / send-recv
  → last PP stage / output rank 产生 ModelRunnerOutput
```

要背住：

```text
ParallelConfig 定义规模，parallel_state 建立 group，模型层和执行层按 group 切分计算，通信原语把分片结果对齐。
```

## 3. 并行维度总览

vLLM 中常见并行维度：

```text
TP：Tensor Parallel，切单层 tensor / attention heads / vocab。
PP：Pipeline Parallel，切 Transformer layers / stage。
DP：Data Parallel，切请求 / batch / replica。
EP：Expert Parallel，切 MoE experts 和 routed tokens。
PCP：Prefill Context Parallel，切 prefill context。
DCP：Decode Context Parallel，切 decode context，复用 TP ranks。
EPLB：Expert Parallel Load Balancing，MoE expert 负载均衡。
```

## 4. 核心区别

```text
TP / PP / PCP / DCP：
  一个 DP replica 内部如何合作完成一次 forward。

DP：
  不同请求如何分配给不同模型副本。

EP：
  MoE layer 内 token 如何路由到持有目标 expert 的 rank。
```

一句话：

```text
TP 切 tensor，PP 切层，DP 切请求，EP 切 expert，CP 切上下文。
```

## 5. 最小端到端链路

并行嵌在每轮执行里：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → collective_rpc("execute_model")
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _model_forward()
  → TP / PP / EP / CP 通信发生
  → logits / sampling / IntermediateTensors
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```

要记住：

```text
并行不是单独一条请求链路，而是嵌在 Executor、Worker、ModelRunner、model layer 和 backend 内部。
```

## 6. world_size 公式

单个 DP replica 内部 worker 数：

```text
world_size = pipeline_parallel_size
           * tensor_parallel_size
           * prefill_context_parallel_size
```

计入 DP 后：

```text
world_size_across_dp = world_size * data_parallel_size
```

重要点：

```text
DCP 不乘进 world_size。
EP 不乘进 world_size。
EPLB 不乘进 world_size。
```

它们在已有 rank 上创建额外 group 或复用已有 group。

## 7. DCP 的特殊约束

DCP 复用 TP rank。

因此要求：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

含义：

```text
DCP 不增加 GPU 数；它把一个 TP group 内部再切成若干 DCP groups。
```

## 8. EP 的特殊地位

EP 不单独乘进 world_size。

MoE 中，EP group 通常在同一个 PP stage 内合并：

```text
DP * PCP * TP
```

含义：

```text
EP 在已有 rank 上组织 expert dispatch，不是额外的 world_size 维度。
```

## 9. rank mesh

vLLM 会把 ranks reshape 成多维 mesh：

```text
all_ranks[external_dp, dp, pp, pcp, tp]
```

各维度：

```text
external_dp：外部 DP / 集成场景额外维度。
dp：vLLM 内部 data parallel replica。
pp：pipeline stage。
pcp：prefill context parallel rank。
tp：tensor parallel rank。
```

各种 group 都从这个 mesh 派生。

## 10. group 如何派生

```text
TP group：
  固定 external_dp / dp / pp / pcp，沿 tp 维取 rank。

DCP group：
  在 TP 维内部按 dcp_size 切小组。

PCP group：
  固定 external_dp / dp / pp / tp，沿 pcp 维取 rank。

PP group：
  固定 external_dp / dp / pcp / tp，沿 pp 维取 rank。

DP group：
  固定 external_dp / pp / pcp / tp，沿 dp 维取 rank。

EP group：
  MoE 中固定 external_dp / pp，合并 dp * pcp * tp。
```

一句话：

```text
rank mesh 是所有并行 group 的底座。
```

## 11. Worker 何时初始化并行环境

GPU Worker 初始化时：

```text
init_worker_distributed_environment()
  → init_distributed_environment()
  → ensure_model_parallel_initialized(
        tensor_parallel_size,
        pipeline_parallel_size,
        prefill_context_parallel_size,
        decode_context_parallel_size,
    )
```

之后：

```text
模型层不重新计算拓扑，而是查询 get_tp_group() / get_pp_group() / get_ep_group() 等全局 group。
```

一句话：

```text
Worker 初始化 distributed group，模型层和通信 op 使用这些 group。
```

## 12. 通信原语

常见封装：

```text
tensor_model_parallel_all_reduce(input)
  → get_tp_group().all_reduce(input)

tensor_model_parallel_all_gather(input, dim)
  → get_tp_group().all_gather(input, dim)

tensor_model_parallel_reduce_scatter(input, dim)
  → get_tp_group().reduce_scatter(input, dim)
```

并行策略不是通信原语本身。

```text
并行策略定义切什么。
通信原语定义切完后怎么对齐。
```

## 13. Tensor Parallel：切单层 tensor

TP 切：

```text
QKV projection 输出维 / head 维
attention heads / KV heads
MLP up / gate / down projection
embedding / vocab parallel
lm_head / logits 相关张量
```

一句话：

```text
TP 是同一个请求、同一层，由多个 rank 分片合作算。
```

## 14. TP 下 QKV 如何切

QKVParallelLinear 通常沿 attention head 维并行。

逻辑：

```text
num_heads = total_num_heads / tp_size
```

KV heads 少于 query heads 时可能复制 KV heads。

重要点：

```text
每个 TP rank 计算自己持有的 query heads / KV heads。
每个 TP rank 读写自己那部分 KV cache。
不会先 gather 所有 KV heads 到一个 rank 再 attention。
```

## 15. TP 常见通信

```text
all-reduce：
  RowParallelLinear / MLP down projection 后合并 partial output。

all-gather：
  需要完整 hidden / vocab / tensor 的地方收集分片。

reduce-scatter：
  在 sequence parallel 或优化路径里边 reduce 边切。
```

要避免误解：

```text
不是所有 TP layer 都立即通信。
ColumnParallelLinear 可以先保持输出分片。
RowParallelLinear 常见需要 all-reduce 合回完整 hidden。
```

## 16. Pipeline Parallel：切 Transformer layers

PP 按 layer 切模型。

典型职责：

```text
first PP stage：
  input embedding + 前半层。

middle PP stage：
  中间 layers。

last PP stage：
  后半层 + final norm + lm_head / logits。
```

不属于当前 stage 的 layer 通常用：

```text
PPMissingLayer
```

占位。

## 17. PP 如何通信

PP 不是 all-reduce，而是 stage 间传中间状态：

```text
stage 0
  → forward 本 stage layers
  → send IntermediateTensors

stage 1
  → recv IntermediateTensors
  → forward 本 stage layers
```

非 last stage 通常返回：

```text
IntermediateTensors({"hidden_states": ..., "residual": ...})
```

last stage 才产生 logits / sampling 输入。

## 18. PP 和 TP 的组合

如果同时有 TP，PP 通信通常按相同 TP rank 对齐：

```text
PP group for tp_rank=0: stage0_tp0 → stage1_tp0
PP group for tp_rank=1: stage0_tp1 → stage1_tp1
```

输出通常来自：

```text
最后一个 PP stage 的第一个 TP rank
```

## 19. Data Parallel：切请求和 replica

DP 切的是请求 / batch / replica。

心智模型：

```text
DP replica 0：处理请求 A / C / E
DP replica 1：处理请求 B / D / F
每个 replica 内部再按 TP / PP / PCP / DCP / EP 完成 forward
```

一句话：

```text
DP 是多个模型副本并行服务不同请求。
```

## 20. DP 和 TP 的区别

```text
TP：同一个请求由多个 rank 合作算。
DP：不同请求被分给不同 replica 算。
```

例子：

```text
DP=2, TP=2
请求 A → replica 0 → TP group [0,1]
请求 B → replica 1 → TP group [2,3]
```

请求 A 不会用 `[2,3]` 做 TP all-reduce。

## 21. DP 是否需要通信

DP 不是完全无通信。

它会参与：

```text
未完成请求状态同步
pause / resume 控制同步
KV cache memory size 跨 replica 一致性
MoE + EP 场景 expert group 构造
load balancing 控制面协调
```

但普通 forward 中，不同 DP replica 主要处理不同请求。

## 22. Expert Parallel：切 MoE experts

EP 服务 MoE layer。

MoE forward：

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
EP 是 token 根据 router 结果被发到持有目标 expert 的 rank。
```

## 23. EP 和其他维度

EP group 通常在同一个 PP stage 内合并：

```text
DP x PCP x TP
```

含义：

```text
PP 决定 MoE layer 在哪个 stage。
EP 在该 stage 内组织 expert dispatch。
TP 仍服务 dense tensor / head 切分。
DP ranks 可能参与 expert group。
```

EP 不改变：

```text
logits / sampling 仍由 last PP stage 负责。
Executor 仍从 output rank 收 ModelRunnerOutput。
```

## 24. Context Parallel：PCP 和 DCP

### PCP

```text
Prefill Context Parallel。
主要服务长 prompt prefill，把 prefill context 相关计算拆到多个 rank。
PCP 乘进 world_size。
```

### DCP

```text
Decode Context Parallel。
主要服务 decode 阶段长 context attention，把 KV context 分担到多个 rank。
DCP 不乘进 world_size，复用 TP group 内 ranks。
```

## 25. DCP 为什么需要 LSE merge

decode attention 中，每个 DCP rank 可能只看一段 KV context。

每个 rank 得到：

```text
partial attention output
softmax_lse / log-sum-exp
```

合并时不能简单相加。

需要：

```text
LSE-weighted merge
```

因为 softmax 分母必须覆盖完整 context。

## 26. KV cache 在并行体系中的归属

### TP 下

```text
每个 TP rank 持有自己负责的 KV heads。
本 rank attention backend 读写本 rank KV cache。
```

### PP 下

```text
每个 PP stage 只持有本 stage attention layers 的 KV cache。
stage 之间传 hidden states，不传整份 KV cache。
```

### DP 下

```text
每个 DP replica 有自己的 KV cache。
请求在哪个 replica 执行，它的 KV blocks 就属于哪个 replica。
普通 DP 不自动共享 KV cache。
```

### CP / DCP 下

```text
影响 attention 如何读取 / 分担 context。
不等价于 KV cache 整体多复制一份。
```

## 27. logits / sampling / output rank

通常：

```text
非 last PP stage：产生 IntermediateTensors。
last PP stage：产生最终 hidden states、logits、sampling 输入。
```

多进程 Executor 一般只从：

```text
最后一个 PP stage 的第一个 TP rank
```

收 `ModelRunnerOutput`。

DP 下每个 replica 有自己的输出 rank。

## 28. 并行维度对照

```text
TP：切 tensor / heads / vocab；乘进 world_size；通信 all-reduce / all-gather。
PP：切 layers；乘进 world_size；通信 send / recv。
DP：切请求 / replica；计入 world_size_across_dp；控制面同步。
EP：切 experts / routed tokens；不乘 world_size；通信 all-to-all。
PCP：切 prefill context；乘进 world_size；attention context 通信。
DCP：切 decode context；不乘 world_size；复用 TP ranks，LSE merge。
EPLB：expert load balancing；不乘 world_size；负载统计和 rebalance。
```

## 29. 组合并行怎么读

建议顺序：

```text
1. 先算 worker 数：PP * TP * PCP，再乘 DP。
2. 写出 all_ranks[external_dp, dp, pp, pcp, tp]。
3. 标出 TP group。
4. 标出 PP group。
5. 标出 DP group。
6. 标出 DCP group。
7. 如果是 MoE，标出 EP group。
8. 看模型层：dense 走 TP，layers 走 PP，experts 走 EP，attention context 走 PCP / DCP。
9. 最后看输出：last PP stage 的第一个 TP rank。
```

## 30. 常见易混点

### world_size 不一定包含 DP

```text
world_size 是单个 DP replica 内部规模。
world_size_across_dp 才乘 data_parallel_size。
```

### DCP 不增加 GPU 数

```text
DCP 复用 TP group 内 rank。
```

### EP 不单独乘 world_size

```text
EP 在已有 DP / PCP / TP ranks 上组织 expert group。
```

### all-reduce 不是 TP 本身

```text
TP 是切 tensor；all-reduce 只是 TP 常用通信。
```

### PP 不传 KV cache

```text
PP stage 之间传 IntermediateTensors，不传整份 KV cache。
```

## 31. 与其他专题的关系

```text
engine_core：EngineCore 发起 execute_model 并回收输出。
scheduler：SchedulerOutput 如何进入并行 Worker。
executor_worker_model_runner：Executor / Worker / ModelRunner 如何承载多 rank 执行。
attention：TP / CP / DCP 如何影响 heads、context、KV layout。
operators：通信 op 和 rank-local kernel 如何执行。
quantization：量化参数和 scale 如何按 TP / EP 切分。
lora_and_adapters：LoRA 权重如何按 TP / PP / DP 分布。
kv_cache_transfer：外部 KV transfer 如何适配并行拓扑。
```

## 32. 背诵总结

背这一段：

```text
vLLM V1 的并行体系由 ParallelConfig 定义规模，由 parallel_state 建立 rank mesh 和通信 group，由模型层和执行层按 group 切分计算。TP 切单层 tensor、heads 和 vocab；PP 切 Transformer layers，并在 stage 间传 IntermediateTensors；DP 切请求和 replica；EP 在 MoE layer 内把 token 按 router 发到 expert rank；PCP / DCP 切 prefill 和 decode context，DCP 复用 TP ranks 并需要 LSE merge。world_size = PP * TP * PCP，world_size_across_dp 再乘 DP；DCP 和 EP 不单独乘 world_size。并行的本质是切分对象、rank group、通信原语、状态归属和输出 rank 的组合。
```
