# 11. TP / PP / DP / EP / CP 如何组合？

源码位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/model.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/communication_op.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/linear.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/logits_processor.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/attention/attention.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/fused_moe/`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/`

本问题关注：TP / PP / DP / EP / PCP / DCP 同时开启时，vLLM 如何理解 rank mesh；哪些并行维度会乘进 worker world size，哪些只是复用已有 rank；单个请求的一次 forward 会经过哪些 group；模型权重、请求状态、KV cache、attention metadata、logits / sampling 分别归属于哪个并行维度；以及组合并行下最容易混淆的约束和执行边界。

---

## 1. 一句话回答

组合并行要按两个问题拆开看：

```text
1. 单个请求的一次 forward 由哪些 rank 合作完成？
   主要看 TP / PP / PCP / DCP / EP。

2. 不同请求如何分给不同 replica？
   主要看 DP。
```

最小心智模型：

```text
DP replica 内部是一套模型并行拓扑：PP x PCP x TP。
DCP 不增加 world size，而是在 TP group 内再切 context parallel group。
EP 也不额外增加 worker，而是在 MoE layer 所在 PP stage 内用 DP x PCP x TP rank 组成 expert group。
```

所以可以把并行组合压缩成：

```text
global world
  → ExternalDP x DP x PP x PCP x TP rank mesh
  → TP / PP / PCP / DCP / DP / EP / EPLB groups
  → dense layer 用 TP
  → pipeline stage 用 PP
  → attention context 用 PCP / DCP
  → MoE layer 用 EP
  → replica 协调和负载切分用 DP
```

如果只记住一句话：

```text
TP / PP / CP / EP 决定“一个模型副本内部怎么合作算一次 forward”，DP 决定“有多少个模型副本并行处理不同请求”。
```

---

## 2. 本文要回答的问题

```text
TP + PP 如何组合？
DP + TP + PP + PCP 的 rank mesh 如何理解？
DCP 为什么不乘进 world_size？
EP 是独立维度还是依附于 TP / DP / PCP？
PCP / DCP 和 attention backend 如何组合？
KV cache 在组合并行下属于哪个 rank / group？
logits / sampling 在组合并行下在哪里发生？
PP 下 token ids / sampled token 如何同步到非 last stage？
DP 下为什么要同步 padding / cudagraph / ubatch 决策？
哪些配置组合有整除或 backend 支持约束？
```

---

## 3. 先给结论：vLLM 的并行组合分三层

### 3.1 Worker 数量层：哪些维度乘进 world_size

`ParallelConfig.__post_init__()` 中计算：

```text
world_size = pipeline_parallel_size
           * tensor_parallel_size
           * prefill_context_parallel_size
```

位置：`code/vllm/vllm/config/parallel.py:782`

这个 `world_size` 表示：

```text
一个 DP replica 内部承载一个模型副本需要多少 worker。
```

如果考虑 DP：

```text
world_size_across_dp = world_size * data_parallel_size
```

位置：`code/vllm/vllm/config/parallel.py:508`

所以默认情况下：

```text
单个 DP replica 内部：PP x PCP x TP
所有 DP replica 总计：DP x PP x PCP x TP
```

注意：

```text
DCP 不乘进 world_size；
EP 不乘进 world_size；
EPLB 不乘进 world_size。
```

它们都是在已有 rank 上创建额外 group。

### 3.2 Group 层：parallel_state.py 创建通信组

Worker 初始化时会调用：

```text
init_distributed_environment(...)
ensure_model_parallel_initialized(...)
  → initialize_model_parallel(...)
```

位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py:1188`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1197`
- `code/vllm/vllm/distributed/parallel_state.py:1674`

`initialize_model_parallel()` 创建：

```text
TP group
DCP group
PCP group
PP group
DP group
EP group
EPLB group
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1757`

### 3.3 执行层：不同模块查询不同 group

执行时不是每个模块自己算 rank 成员，而是按需查询 group：

```text
dense TP layer       → get_tp_group()
pipeline send/recv   → get_pp_group()
DP padding/state sync → get_dp_group()
MoE all-to-all        → get_ep_group()
EPLB stats/rebalance  → get_eplb_group()
prefill CP attention  → get_pcp_group()
decode CP attention   → get_dcp_group()
```

通信 helper 例如：

```text
tensor_model_parallel_all_reduce()     → get_tp_group().all_reduce(...)
tensor_model_parallel_all_gather()     → get_tp_group().all_gather(...)
tensor_model_parallel_reduce_scatter() → get_tp_group().reduce_scatter(...)
```

位置：`code/vllm/vllm/distributed/communication_op.py:12`

---

## 4. rank mesh 的核心形状

`initialize_model_parallel()` 中有一段关键注释：

```text
the layout order is: ExternalDP x DP x PP x TP
```

当前实际 reshape 已经包含 PCP：

```text
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1740`

因此完整 rank mesh 可以理解为：

```text
all_ranks[external_dp, dp, pp, pcp, tp]
```

其中：

```text
external_dp：外部 DP / 外部集成场景的额外维度，普通场景通常是 1；
dp：data parallel replica 维度；
pp：pipeline stage 维度；
pcp：prefill context parallel 维度；
tp：tensor parallel 维度。
```

这个 mesh 是理解所有 group 的基础。

---

## 5. 一个具体 rank mesh 例子

假设：

```text
DP = 2
PP = 2
PCP = 1
TP = 4
ExternalDP = 1
```

总 world size：

```text
world_size_across_dp = 2 * 2 * 1 * 4 = 16
```

可以把 ranks 排成：

```text
DP 0:
  PP 0, TP [0, 1, 2, 3]
  PP 1, TP [4, 5, 6, 7]

DP 1:
  PP 0, TP [8, 9, 10, 11]
  PP 1, TP [12, 13, 14, 15]
```

那么：

```text
TP groups:
  [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]

PP groups:
  [0,4], [1,5], [2,6], [3,7], [8,12], [9,13], [10,14], [11,15]

DP groups:
  [0,8], [1,9], [2,10], [3,11], [4,12], [5,13], [6,14], [7,15]
```

含义：

```text
同一个 TP group：同一个 DP replica、同一个 PP stage 内合作算一层；
同一个 PP group：同一个 DP replica、同一个 TP lane 上串联 pipeline stages；
同一个 DP group：不同 DP replica 上同一个 PP/PCP/TP 坐标的 rank。
```

---

## 6. TP group 如何从 mesh 里切出来

TP group 创建逻辑：

```text
group_ranks = all_ranks.view(-1, tensor_model_parallel_size)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1757`

因为 TP 是最后一维，所以每个 TP group 是 mesh 中固定：

```text
external_dp, dp, pp, pcp
```

然后沿：

```text
tp 维度
```

取一组 ranks。

语义：

```text
同一个 layer / 同一个 pipeline stage / 同一个 DP replica 内，多个 TP ranks 共同完成 dense tensor 计算。
```

典型使用：

```text
ColumnParallelLinear：按输出维切权重，必要时 all-gather 输出；
RowParallelLinear：按输入维切权重，通常 all-reduce 聚合输出；
LogitsProcessor：vocab parallel logits 需要 gather / all-gather；
Attention：本 rank 只持有局部 Q heads / KV heads。
```

`ColumnParallelLinear` 中：

```text
self.tp_rank = get_tensor_model_parallel_rank()
self.tp_size = get_tensor_model_parallel_world_size()
output_size_per_partition = output_size / tp_size
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:434`

如果需要完整输出：

```text
output = tensor_model_parallel_all_gather(output_parallel)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:557`

---

## 7. PP group 如何从 mesh 里切出来

PP group 创建逻辑：

```text
group_ranks = all_ranks.transpose(2, 4)
                     .reshape(-1, pipeline_model_parallel_size)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1815`

这表示：

```text
固定 external_dp、dp、pcp、tp；
沿 pp 维度取一组 ranks。
```

语义：

```text
同一个 DP replica 内，同一个 TP lane 上，不同 pipeline stages 串起来。
```

执行时：

```text
非 first PP rank：从前一个 stage recv intermediate tensors；
非 last PP rank：forward 后 send intermediate tensors 给下一个 stage；
last PP rank：计算 logits / sampling / final output。
```

V1 worker 中非 first rank 接收：

```text
if forward_pass and not get_pp_group().is_first_rank:
    intermediate_tensors = get_pp_group().irecv_tensor_dict(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:853`

非 last rank 发送：

```text
self._pp_send_work = get_pp_group().isend_tensor_dict(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:889`

`GPUModelRunner` 中也有同步 send：

```text
get_pp_group().send_tensor_dict(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4367`

---

## 8. PCP group 如何从 mesh 里切出来

PCP group 创建逻辑：

```text
group_ranks = all_ranks.transpose(3, 4)
                     .reshape(-1, prefill_context_model_parallel_size)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1796`

这表示：

```text
固定 external_dp、dp、pp、tp；
沿 pcp 维度取一组 ranks。
```

PCP 是 prefill context parallel。

它乘进 `world_size`：

```text
world_size = PP * TP * PCP
```

位置：`code/vllm/vllm/config/parallel.py:782`

所以 PCP 会实际增加每个 DP replica 内的 worker 数。

它影响：

```text
prefill 阶段的 context / KV 分布；
KV cache interleave；
slot mapping 的 total_cp_world_size；
attention metadata 中的 CP 本地长度；
某些 backend 的 prefill attention merge。
```

当前很多 CP 相关公式会把 PCP 和 DCP 合起来看：

```text
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

位置：`code/vllm/vllm/config/parallel.py:351`

---

## 9. DCP group 如何从 mesh 里切出来

DCP group 创建逻辑：

```text
group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1774`

因为 TP 是最后一维，而 DCP 直接 reshape 成 `decode_context_model_parallel_size`，所以 DCP 本质上是在 TP 维度内部再切连续小组。

这也是为什么配置校验要求：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

位置：`code/vllm/vllm/config/parallel.py:490`

源码注释说得很直接：

```text
DCP does not change world size;
it reuses the GPUs of TP group;
it splits one TP group into tp_size // dcp_size DCP groups.
```

位置：`code/vllm/vllm/config/parallel.py:490`

举例：

```text
TP group = [0, 1, 2, 3]
DCP size = 2
DCP groups = [0, 1], [2, 3]
```

因此：

```text
TP 负责 tensor/head 维度分片；
DCP 复用 TP ranks 中的一部分来切 decode context；
DCP 不能被看成和 TP 完全正交的新 worker 维度。
```

---

## 10. DP group 如何从 mesh 里切出来

DP group 创建逻辑：

```text
group_ranks = all_ranks.transpose(1, 4)
                     .reshape(-1, data_parallel_size)
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1833`

这表示：

```text
固定 external_dp、pp、pcp、tp；
沿 dp 维度取一组 ranks。
```

DP 的语义和 TP / PP 完全不同：

```text
TP / PP：多个 rank 合作完成同一个请求的一次 forward；
DP：多个 replica 处理不同请求或不同 batch，但需要在某些时刻保持执行形态一致。
```

DP replica 之间通常有独立：

```text
Scheduler / request ownership
InputBatch
KV cache
attention metadata
sampling state
```

但 DP group 仍用于：

```text
1. 同步是否还有未完成请求；
2. 同步 pause 状态；
3. 同步 KV cache 可用内存大小；
4. 同步 CUDA graph / DP padding / ubatch 决策；
5. MoE + EP 场景下参与 expert sharding / all-to-all。
```

`ParallelConfig` 中有 DP 同步 helper：

```text
has_unfinished_dp(...)
sync_dp_state(...)
sync_kv_cache_memory_size(...)
```

位置：`code/vllm/vllm/config/parallel.py:680`

---

## 11. EP group 如何从 mesh 里切出来

EP group 创建逻辑：

```text
group_ranks = all_ranks.transpose(1, 2)
                     .reshape(
                         -1,
                         data_parallel_size
                         * prefill_context_model_parallel_size
                         * tensor_model_parallel_size,
                     )
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1850`

这表示：

```text
固定 external_dp 和 pp stage；
把 DP x PCP x TP 合成一个 expert parallel group。
```

所以 EP 不是新增 worker 维度，而是 MoE layer 在某个 PP stage 内使用的一组 rank。

原因是：

```text
MoE layer 是 transformer layer 的一部分；
PP 决定这个 layer 位于哪个 pipeline stage；
在这个 stage 内，DP / PCP / TP 上的 rank 都可以参与 expert sharding 和 token dispatch。
```

EP group 只在 MoE 模型或模型配置未知时创建：

```text
if config.model_config is None or config.model_config.is_moe:
    create EP group
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1852`

MoE all-to-all manager 通过：

```text
get_ep_group().device_communicator.all2all_manager
```

取得通信管理器。

位置：`code/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py:59`

---

## 12. EPLB group 为什么单独存在

EPLB 是 expert parallel load balancing。

如果开启：

```text
enable_eplb = True
```

vLLM 会创建和 EP ranks 相同的一组 EPLB group：

```text
Create EPLB group with the same ranks as EP if EPLB is enabled.
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1878`

但它是单独的 process group。

原因：

```text
EPLB 通信和 MoE forward 通信如果混用同一个 group，
可能和 torch.distributed / MoE collectives 互相阻塞甚至死锁。
```

源码注释：

```text
This is a separate process group to isolate EPLB communications
from MoE forward pass collectives and prevent deadlocks.
```

位置：`code/vllm/vllm/distributed/parallel_state.py:1878`

所以：

```text
EP group：MoE forward 的 expert dispatch / combine；
EPLB group：expert load balancing / 统计 / 重排。
```

---

## 13. TP + PP 如何组合

TP + PP 是最经典的模型并行组合。

可以这样理解：

```text
PP：把 layer 维度切成 stages；
TP：每个 stage 内把单层 tensor 计算切到多个 ranks。
```

执行路径：

```text
PP stage 0:
  token input / embeddings
  local layers with TP collectives
  send intermediate tensors

PP stage 1..N-2:
  recv intermediate tensors
  local layers with TP collectives
  send intermediate tensors

PP last stage:
  recv intermediate tensors
  local layers with TP collectives
  logits / sampling
```

TP group 和 PP group 不是同一组 ranks：

```text
TP group：同一个 stage 内横向合作；
PP group：同一个 TP lane 上纵向串 stage。
```

`ModelConfig.get_layers_start_end_indices()` 用 PP rank 决定本 rank 持有哪些层：

```text
pp_rank = (parallel_config.rank // (tensor_parallel_size * prefill_context_parallel_size)) % pipeline_parallel_size
start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
```

位置：`code/vllm/vllm/config/model.py:1280`

这说明：

```text
PP 影响 layer 范围；
TP 影响每层内参数 / heads 分片。
```

---

## 14. DP + TP + PP 如何组合

DP 会复制整套模型并行结构。

如果：

```text
DP = 2
PP = 2
TP = 4
```

则每个 DP replica 内部都有：

```text
PP x TP = 8 ranks
```

总共：

```text
16 ranks
```

请求执行可以理解为：

```text
1. 请求先被某个 DP replica 的 Scheduler / EngineCore 接收或分配；
2. 该 DP replica 内部用 PP x TP 完成 forward；
3. last PP stage 产生 logits / sampling；
4. 输出回到该 DP replica 的 engine / scheduler；
5. DP group 只在需要同步执行状态、padding、ubatch、MoE EP 等场景通信。
```

所以：

```text
DP 不是把一个请求切到多个 replica 上共同算；
DP 是多个 replica 并行服务不同请求。
```

但 DP ranks 仍然需要协调 batch 执行形态。

例如 `_determine_batch_execution_and_padding()` 中：

```text
if data_parallel_size > 1:
    coordinate_batch_across_dp(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3879`

---

## 15. DP 为什么要同步 padding / cudagraph / ubatch

DP replica 处理的是不同请求，所以每个 DP rank 本轮 token 数可能不同。

但是 CUDA graph、DBO / ubatching、某些 collective path 要求各 DP rank 执行形态一致。

`coordinate_batch_across_dp()` 做三件事：

```text
1. 决定所有 DP rank 是否都启用 microbatching；
2. 决定每个 DP rank 本轮 padded token 数；
3. 同步 cudagraph_mode，取最保守值。
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:164`

内部通过 DP group all-reduce：

```text
tensor[0][dp_rank] = orig_num_tokens_per_ubatch
tensor[1][dp_rank] = padded_num_tokens_per_ubatch
tensor[2][dp_rank] = should_ubatch
tensor[3][dp_rank] = cudagraph_mode
dist.all_reduce(tensor, group=get_dp_group())
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:36`

如果需要 DP padding：

```text
所有 DP ranks pad 到最大 token 数。
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:77`

这说明：

```text
DP 不共享请求状态，但可能共享执行形态约束。
```

---

## 16. TP + EP 如何组合

TP 切 dense tensor，EP 切 experts。

两者在 MoE 模型里同时存在：

```text
普通 dense / attention / MLP 投影：走 TP；
MoE expert layer：走 EP dispatch / expert compute / combine；
MoE shared expert 或输出聚合：可能仍需要 TP all-reduce。
```

`ParallelConfig` 字段说明：

```text
data_parallel_size:
  MoE layers will be sharded according to
  the product of tensor_parallel_size and data_parallel_size.
```

位置：`code/vllm/vllm/config/parallel.py:126`

开启 EP：

```text
enable_expert_parallel = True
```

位置：`code/vllm/vllm/config/parallel.py:162`

MoE all-to-all backend：

```text
all2all_backend
```

位置：`code/vllm/vllm/config/parallel.py:185`

MoE dispatch 的核心不是普通 TP all-reduce，而是：

```text
router/topk → 按 expert 归属 dispatch tokens → 本地 experts 计算 → combine 回原 token 顺序
```

EP 使用的通信 manager 来自：

```text
get_ep_group().device_communicator.all2all_manager
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/all2all_utils.py:67`

---

## 17. sequence parallel MoE 是 TP + DP + EP 的特殊组合

`ParallelConfig.use_sequence_parallel_moe` 表示某些 MoE all-to-all backend 下要使用 sequence parallel 输入，避免重复计算。

条件：

```text
all2all_backend in (... deepep / mori / nixl_ep ...)
enable_expert_parallel = True
tensor_parallel_size > 1
data_parallel_size > 1
```

位置：`code/vllm/vllm/config/parallel.py:625`

源码注释解释原因：

```text
TP attention o_proj 后的 all_reduce 会让输入在 TP group 内复制。
如果 DeepEP All2All 直接消费复制后的 tokens，
会产生无用的重复 expert 计算和通信。
```

位置：`code/vllm/vllm/config/parallel.py:625`

所以 sequence parallel MoE 的目的：

```text
把进入 expert 的 token 在 sequence / token 维度上拆开，
避免每个 TP rank 对同一批 tokens 重复 dispatch 到 experts。
```

这类组合最容易混淆，因为它同时涉及：

```text
TP 输出复制状态；
DP 多 replica；
EP expert all-to-all；
sequence parallel token 分片。
```

---

## 18. PCP / DCP + attention 如何组合

CP 直接影响 attention，而不是普通 dense layer。

attention metadata 中有：

```text
dcp_local_seq_lens
dcp_local_seq_lens_cpu
```

位置：`code/vllm/vllm/v1/attention/backend.py:400`

在 `_build_attention_metadata()` 中，如果 DCP 开启：

```text
self.dcp_local_seq_lens = get_dcp_local_seq_lens(...)
cm_base.dcp_local_seq_lens = ...
cm_base.dcp_local_seq_lens_cpu = ...
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2349`

slot mapping 也会受 CP 影响：

```text
virtual_block_size = block_size * total_cp_world_size
is_local = ... == total_cp_rank
slot_ids = local ? slot_id : PAD_SLOT_ID
```

相关逻辑在：

- block table：`code/vllm/vllm/v1/worker/block_table.py:357`
- GPUModelRunner block table 使用路径：`code/vllm/vllm/v1/worker/block_table.py:283`

所以 CP 的核心影响是：

```text
1. 一个 token 的 KV 可能只属于某个 CP rank；
2. 当前 rank 的 slot_mapping 对非本地 token 写 PAD_SLOT_ID；
3. attention backend 需要计算本地 partial attention；
4. 多个 CP ranks 的 partial attention 需要按 backend 协议合并。
```

MLA 中可以看到 DCP group 通信，例如：

```text
mqa_q = get_dcp_group().all_gather(mqa_q, dim=1)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:786`

---

## 19. PCP 和 DCP 的区别

可以这样记：

```text
PCP：prefill context parallel；
     乘进 world_size，增加 DP replica 内 worker 数。

DCP：decode context parallel；
     不乘进 world_size，复用 TP ranks，在 TP group 内切小组。
```

配置字段：

```text
prefill_context_parallel_size
decode_context_parallel_size
```

位置：

- `code/vllm/vllm/config/parallel.py:124`
- `code/vllm/vllm/config/parallel.py:331`

DCP 约束：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

位置：`code/vllm/vllm/config/parallel.py:490`

非 MLA 的 GQA / MQA 模型上，DCP 还有额外校验：

```text
tensor_parallel_size > total_num_kv_heads
decode_context_parallel_size <= tensor_parallel_size // total_num_kv_heads
num_q_per_kv % decode_context_parallel_size == 0
```

位置：`code/vllm/vllm/config/model.py:1182`

原因：

```text
GQA / MQA 下 KV heads 数可能小于 TP size；
DCP 要在复制 KV heads 的 TP ranks 上切 context；
如果 DCP 大小超过可复制空间，就无法正确分配 query-per-kv 和 partial attention。
```

---

## 20. TP 如何影响 attention heads 和 KV cache

`ModelConfig.verify_with_parallel_config()` 会先校验：

```text
total_num_attention_heads % tensor_parallel_size == 0
```

位置：`code/vllm/vllm/config/model.py:1157`

本 rank Q heads：

```text
num_heads = total_num_attention_heads // tensor_parallel_size
```

位置：`code/vllm/vllm/config/model.py:1270`

本 rank KV heads：

```text
num_kv_heads = max(1, total_num_kv_heads // tensor_parallel_size)
```

位置：`code/vllm/vllm/config/model.py:1257`

注意这个 `max(1, ...)`：

```text
如果 total KV heads 小于 TP size，KV heads 会在多个 TP ranks 上复制，
保证每个 rank 至少有一个本地 KV head。
```

这会影响：

```text
Attention.get_kv_cache_spec()
KV cache shape
slot_mapping 写入的本地 KV cache
attention backend 的 num_heads / num_kv_heads
```

所以 TP 不只是切权重，也会改变本 rank KV cache 的 head 维度。

---

## 21. PP 如何影响 KV cache 和 attention layers

PP 决定每个 rank 持有哪些 layers。

`get_layers_start_end_indices()` 中：

```text
pp_rank = (rank // tensor_parallel_size) % pipeline_parallel_size
start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
```

位置：`code/vllm/vllm/config/model.py:1280`

因此：

```text
PP stage 0 只持有前一段 layers 的权重和 KV cache；
PP stage 1 持有后一段 layers 的权重和 KV cache；
...
PP last stage 才负责 logits / sampling。
```

组合 TP 时：

```text
每个 PP stage 内有一个 TP group；
每个 stage 内每个 TP rank 持有该 stage 层的局部 heads / 局部 weights / 局部 KV cache。
```

因此 KV cache 不是所有 PP ranks 都有所有 layers。

它是：

```text
每个 rank 只持有“当前 PP stage 分配到的 layers”的 KV cache。
```

---

## 22. DP 如何影响 KV cache 和请求状态

DP replica 之间 KV cache 不共享。

可以理解为：

```text
DP rank 0 有自己的 Scheduler / request states / KV blocks / InputBatch；
DP rank 1 有自己的 Scheduler / request states / KV blocks / InputBatch；
...
```

同一个 request 通常归属于某个 DP replica。

因此：

```text
block table 是 replica-local；
slot mapping 是 replica-local；
KVCacheManager 是 replica-local；
KV cache tensor 是 replica-local。
```

DP group 做的是跨 replica 的协调，而不是共享 KV cache。

例如同步 batch padding：

```text
coordinate_batch_across_dp(...)
```

位置：`code/vllm/vllm/v1/worker/dp_utils.py:164`

同步是否还有未完成请求：

```text
ParallelConfig.has_unfinished_dp(...)
```

位置：`code/vllm/vllm/config/parallel.py:680`

---

## 23. EP 是否影响 dense attention KV cache

通常不直接影响。

EP 作用于 MoE experts：

```text
router / topk
expert token dispatch
expert compute
expert output combine
```

attention KV cache 仍然主要受：

```text
TP：本地 heads / KV heads；
PP：本地 layers；
DP：replica 隔离；
PCP / DCP：context / position 分片。
```

影响关系可以这样理解：

```text
EP 不改变普通 attention 的 block table / slot mapping 公式；
但在 MoE 模型中，attention layer 和 MoE layer 交替出现，
同一次 forward 会先后使用 TP / attention metadata / EP all-to-all 等不同通信路径。
```

所以 EP 是组合执行链路的一部分，但不是 attention KV cache 的主导维度。

---

## 24. logits / sampling 在组合并行下在哪里发生

### 24.1 PP：通常只在 last stage 产生最终 logits

在 PP 下，非 last stage 返回 intermediate tensors。

classic runner 中：

```text
if not get_pp_group().is_last_rank:
    return intermediate_tensors
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4337`

last stage 才继续：

```text
sample_hidden_states = hidden_states[logits_indices]
compute_logits(...)
sample(...)
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4337`

### 24.2 TP：logits 可能需要 gather vocab shards

`LogitsProcessor` 会通过 TP group gather logits：

```text
logits = tensor_model_parallel_all_gather(logits)
```

或：

```text
logits = tensor_model_parallel_gather(logits)
```

位置：`code/vllm/vllm/model_executor/layers/logits_processor.py:75`

普通路径：

```text
lm_head 产生 vocab shard logits；
TP gather 得到完整 vocab logits；
去掉 padding vocab；
sampling 使用完整 logits。
```

位置：`code/vllm/vllm/model_executor/layers/logits_processor.py:89`

### 24.3 DP：每个 replica 独立采样

DP ranks 处理不同请求，所以采样通常 replica-local。

但 DP 可能需要同步：

```text
是否还有 unfinished requests；
pause 状态；
CUDA graph / padding / ubatch 执行形态。
```

### 24.4 PP 非 last stage 如何拿到 sampled token

PP 非 last stage 没有最终 logits，但下一轮输入也需要 sampled token。

因此 last PP stage 会广播 sampled token ids：

```text
if pp.world_size > 1 and pp.is_last_rank:
    _pp_broadcast_prev_sampled_token_ids(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4469`

非 last stage 接收：

```text
_pp_receive_prev_sampled_token_ids_to_input_batch()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4430`

这解释了为什么 PP 下非 last stage 也要维护 token ids / InputBatch 状态。

---

## 25. 单个请求一次 forward 会经过哪些 group

假设开启：

```text
DP > 1
PP > 1
TP > 1
DCP > 1
EP enabled
```

一个请求被分配到某个 DP replica 后，forward 大致经过：

```text
1. DP replica 内 Scheduler 产生 SchedulerOutput。

2. PP first stage：
   准备 input_ids / positions / attention metadata。

3. 每个 PP stage 内：
   dense linear / attention projection 使用 TP group 通信。

4. attention layer：
   使用本 rank KV cache、block table、slot mapping；
   如果 DCP / PCP 开启，使用 CP metadata 和 DCP / PCP group 合并 partial attention。

5. MoE layer：
   router 选 expert；
   EP group token dispatch / combine；
   必要时结合 TP / sequence parallel。

6. 非 last PP stage：
   发送 intermediate tensors 给下一 stage。

7. last PP stage：
   计算 logits；
   TP gather vocab logits；
   sampler 采样。

8. PP last stage 将 sampled token ids 广播给前面 stages。

9. DP group 只在需要同步执行状态 / padding / ubatch / pause 时参与。
```

核心是：

```text
不是所有 group 每一层都用；
不同 layer / 不同阶段按需使用不同 group。
```

---

## 26. 组合并行下的状态归属

| 状态 | 主要受哪些并行影响 | 说明 |
|---|---|---|
| 模型权重 | TP / PP / EP | TP 切 dense tensor，PP 切 layer，EP 切 MoE experts |
| 请求归属 | DP / Scheduler | 请求通常属于某个 DP replica |
| token ids / InputBatch | DP / PP | replica-local；PP 非 last stage 也需要同步 sampled token |
| KV cache tensor | TP / PP / DP / CP | TP 切 heads，PP 切 layers，DP 独立 replica，CP 切 context / slot |
| block table | DP / PP / CP 间接受影响 | replica-local、stage-local layer 使用；CP 改 slot mapping 解释 |
| slot mapping | TP / CP / CUDA graph | 本 rank本地 KV slot；CP 非本地 token 填 PAD_SLOT_ID |
| attention metadata | TP / PP / CP / backend | 本 rank heads、layers、seq lens、block table、slot mapping 共同决定 |
| logits | PP / TP | last PP stage 计算，TP gather vocab shards |
| sampling | PP / DP | last PP stage、当前 DP replica 内采样 |
| MoE routing | EP / DP / TP / PCP | EP group 通常是同一 PP stage 内 DP x PCP x TP |
| CUDA graph padding | TP / DP / PP / backend | sequence parallel 和 DP 可能要求额外 padding / 同步 |

一句话：

```text
权重按模型并行切，请求按 DP replica 切，KV cache 按执行 rank 本地持有，metadata 每轮按本 rank 的 batch 和 group 身份构造。
```

---

## 27. 组合并行下的配置约束

### 27.1 attention heads 必须能被 TP 整除

```text
total_num_attention_heads % tensor_parallel_size == 0
```

位置：`code/vllm/vllm/config/model.py:1157`

否则每个 TP rank 无法获得相同数量 Q heads。

### 27.2 PP 需要模型支持 SupportsPP

如果：

```text
pipeline_parallel_size > 1
```

模型必须支持 PP：

```text
self.registry.is_pp_supported_model(...)
```

否则报错。

位置：`code/vllm/vllm/config/model.py:1173`

### 27.3 DCP size 必须整除 TP size

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

位置：`code/vllm/vllm/config/parallel.py:490`

### 27.4 DCP + GQA / MQA 还有额外限制

非 MLA 模型上：

```text
tp_size > total_num_kv_heads
dcp_size <= tp_size // total_num_kv_heads
num_q_per_kv % dcp_size == 0
```

位置：`code/vllm/vllm/config/model.py:1182`

### 27.5 dcp_comm_backend=a2a 要求 DCP > 1

```text
dcp_comm_backend='a2a' requires decode_context_parallel_size > 1
```

位置：`code/vllm/vllm/config/parallel.py:501`

### 27.6 Elastic EP 不支持 PP

如果开启 elastic EP：

```text
enable_elastic_ep = True
```

则：

```text
pipeline_parallel_size > 1 会报错
```

位置：`code/vllm/vllm/config/parallel.py:804`

### 27.7 torch_shm multimodal IPC 不支持多并行 world

如果：

```text
mm_tensor_ipc == "torch_shm"
world_size_across_dp > 1
```

会报错。

位置：`code/vllm/vllm/config/model.py:1207`

原因是 torch shared memory IPC 当前不能覆盖 DP / TP / PP 的复杂分发。

### 27.8 sequence parallel 要求 token 数能被 TP 整除

如果启用 sequence parallel 编译 pass：

```text
batch_descriptor.num_tokens % tensor_parallel_size == 0
```

否则报错。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3869`

---

## 28. external_launcher 下的特殊性

通常：

```text
parallel_config.world_size = PP * TP * PCP
world_size_across_dp = world_size * DP
```

但如果：

```text
distributed_executor_backend == "external_launcher"
```

`__post_init__()` 会：

```text
self.world_size *= self.data_parallel_size
```

位置：`code/vllm/vllm/config/parallel.py:790`

因为 external launcher 下，外部启动器已经把 DP rank 也作为进程维度纳入全局 world。

同时会自动推导：

```text
data_parallel_rank = int(os.environ["RANK"]) // (world_size // data_parallel_size)
```

位置：`code/vllm/vllm/config/parallel.py:816`

所以读代码时要注意：

```text
普通 mp/ray 内部 DP：world_size 是单 DP replica 内部规模；
external_launcher：world_size 可能已经包含 DP。
```

---

## 29. 典型组合一：TP = 4，其他为 1

配置：

```text
TP = 4
PP = 1
DP = 1
PCP = 1
DCP = 1
EP = off
```

含义：

```text
一个请求由 4 个 ranks 合作完成每一层；
所有 layers 都在同一 pipeline stage；
没有 replica 级请求切分；
没有 context parallel；
没有 expert parallel。
```

执行：

```text
input_ids 每个 TP rank 都有；
Q/K/V projections 按 TP 切；
attention heads / KV heads 是本 rank 局部值；
o_proj / MLP 等按 TP 通信聚合；
logits 通过 TP gather / all-gather 得到完整 vocab。
```

适用场景：

```text
单模型太大或单层矩阵太大，需要横向切 tensor。
```

---

## 30. 典型组合二：TP = 4，PP = 2

配置：

```text
TP = 4
PP = 2
DP = 1
```

含义：

```text
总 8 ranks；
2 个 pipeline stages；
每个 stage 内 4 个 TP ranks。
```

执行：

```text
stage 0 的 TP group 处理前半 layers；
stage 0 发送 intermediate tensors；
stage 1 的 TP group 处理后半 layers；
stage 1 计算 logits / sampling。
```

需要注意：

```text
PP group 是同一 TP lane 上跨 stage 的 ranks；
TP group 是同一 stage 内的 ranks。
```

例如：

```text
TP groups: [0,1,2,3], [4,5,6,7]
PP groups: [0,4], [1,5], [2,6], [3,7]
```

---

## 31. 典型组合三：DP = 2，TP = 4，PP = 2

配置：

```text
DP = 2
TP = 4
PP = 2
PCP = 1
```

含义：

```text
总 16 ranks；
每个 DP replica 内有 8 ranks；
每个 replica 内是 2-stage PP，每个 stage 4-way TP。
```

执行：

```text
DP rank 0 处理一批请求；
DP rank 1 处理另一批请求；
各自 replica 内用 TP + PP 完成 forward；
DP group 只同步必要执行状态 / padding / ubatch。
```

最容易误解的点：

```text
同一个请求不会被 DP rank 0 和 DP rank 1 共同算；
它只会进入某一个 DP replica。
```

---

## 32. 典型组合四：TP = 8，DCP = 2

配置：

```text
TP = 8
DCP = 2
PP = 1
DP = 1
```

含义：

```text
总 worker 仍然是 8；
DCP 不增加 worker；
每个 TP group 内切成 4 个 DCP groups，每个大小 2。
```

例如 TP group：

```text
[0,1,2,3,4,5,6,7]
```

DCP groups：

```text
[0,1], [2,3], [4,5], [6,7]
```

执行：

```text
TP 仍负责 tensor/head 分片；
DCP 在 decode attention 中切 context；
slot mapping 对非本地 CP token 写 PAD_SLOT_ID；
backend 需要合并 partial attention。
```

约束：

```text
TP % DCP == 0
```

---

## 33. 典型组合五：MoE + DP + TP + EP

配置：

```text
MoE model
DP = 2
TP = 4
enable_expert_parallel = True
PP = 1
PCP = 1
```

EP group 大小：

```text
DP * PCP * TP = 2 * 1 * 4 = 8
```

含义：

```text
同一 PP stage 内，两个 DP replica 和四个 TP ranks 共同组成 expert parallel group；
experts 在这 8 个 ranks 上分布；
tokens 按 router 结果 all-to-all 到对应 expert rank。
```

注意：

```text
虽然 EP group 跨 DP 维度，但这不意味着普通 dense attention 的请求状态跨 DP 共享；
它是 MoE expert dispatch 的特殊通信组织。
```

---

## 34. 典型组合六：PP + EP

配置：

```text
PP = 2
TP = 4
DP = 2
enable_expert_parallel = True
```

EP group 固定 PP stage：

```text
stage 0 有自己的 EP group；
stage 1 有自己的 EP group。
```

原因：

```text
MoE layer 属于某个 pipeline stage；
不能让 stage 0 的 expert forward 和 stage 1 的 expert forward 混在同一个 layer group 中。
```

`initialize_model_parallel()` 中 EP 创建固定 PP 维度，将 DP x PCP x TP 合并。

位置：`code/vllm/vllm/distributed/parallel_state.py:1850`

所以：

```text
EP group 跨 DP / PCP / TP，但不跨 PP。
```

---

## 35. 组合并行下的一次完整 forward 示例

假设：

```text
DP = 2
PP = 2
TP = 4
DCP = 2
MoE + EP enabled
```

某请求进入 DP rank 0。

### 35.1 调度层

```text
DP rank 0 的 Scheduler 决定本轮请求调度；
产生 SchedulerOutput；
DP rank 1 调度自己的请求。
```

如果 DP padding / ubatch / cudagraph 需要一致：

```text
coordinate_batch_across_dp()
```

会跨 DP group 同步执行形态。

### 35.2 PP stage 0

```text
first PP rank 准备 input_ids / positions；
每个 TP rank 根据本地 heads / weights 执行 layers；
attention 用本 stage 的 KV cache；
DCP attention 在 TP 内子组做 context parallel；
MoE layer 用当前 stage 的 EP group dispatch tokens；
输出 intermediate tensors。
```

### 35.3 PP stage 1

```text
recv stage 0 的 intermediate tensors；
继续本 stage layers；
attention / MoE 同样使用 TP / DCP / EP；
最后计算 logits；
TP gather vocab logits；
sampler 采样。
```

### 35.4 PP sampled token 同步

```text
last PP stage 把 sampled token ids 广播给前面 stages；
前面 stages 更新 InputBatch token state，准备下一轮。
```

这就是组合并行下“一次请求”真正走过的路径。

---

## 36. 组合并行下的通信类型对照

| 并行 | 典型通信 | 使用位置 | 目的 |
|---|---|---|---|
| TP | all-reduce / all-gather / reduce-scatter | linear、logits、部分 norm / attention | 聚合 tensor shard |
| PP | send / recv tensor dict | worker / model runner | stage 间传 intermediate tensors |
| DP | all-reduce | dp_utils、状态同步 | 同步 padding、ubatch、是否有未完成请求 |
| EP | all-to-all / dispatch / combine | fused_moe | token 到 expert rank 的路由与回收 |
| EPLB | 独立 group 通信 | eplb | expert 负载统计和重平衡 |
| PCP | backend-specific collectives | attention prefill | prefill context parallel |
| DCP | all-gather / reduce-scatter / all-to-all + merge | attention decode / MLA | decode partial attention 合并 |

注意：

```text
同一个 forward 里可能交替出现多种通信；
但每种通信只在需要的 layer / stage 出现。
```

---

## 37. 常见误区

### 37.1 误区：world_size = TP * PP 就够了

不一定。

当前 vLLM 中，一个 DP replica 内部：

```text
world_size = PP * TP * PCP
```

包含所有 DP replica：

```text
world_size_across_dp = world_size * DP
```

external launcher 下还会把 DP 乘进 `world_size` 本身。

### 37.2 误区：DCP 和 PCP 都是普通独立维度

不对。

```text
PCP 乘进 world_size；
DCP 不乘进 world_size，而是在 TP group 内切分。
```

### 37.3 误区：EP 是 TP 的一种

不对。

```text
TP 切 dense tensor / heads；
EP 切 MoE experts；
TP 常用 all-reduce / all-gather；
EP 核心是 token all-to-all dispatch / combine。
```

### 37.4 误区：DP 和 TP 都是“多卡并行”，所以类似

不对。

```text
TP：多个 rank 合作算同一个请求的同一层；
DP：多个 replica 处理不同请求或 batch。
```

### 37.5 误区：PP 非 last stage 不需要 token state

不对。

PP 非 last stage 下一轮也要处理对应请求的前面 layers，因此需要维护 token ids / InputBatch 状态。

last stage 会把 sampled token ids 广播回非 last stages。

### 37.6 误区：EP 跨 DP 就表示 DP 请求状态共享

不对。

EP group 可以跨 DP 维度做 MoE expert dispatch，但普通请求状态、KV cache、Scheduler 仍是 DP replica-local。

### 37.7 误区：CP 只是 KV cache 分片

不完整。

CP 会影响：

```text
slot mapping；
seq lens / local seq lens；
attention backend partial output；
softmax / LSE merge；
backend 支持能力。
```

### 37.8 误区：所有 ranks 都会计算 logits / sample

不对。

PP 下通常只有 last PP stage 继续 logits / sampling。

TP 下 logits 还需要 gather vocab shards。

---

## 38. 调试入口

如果要调试组合并行，建议按这条顺序看：

```text
1. ParallelConfig.__post_init__()
   看 world_size / world_size_across_dp / external_launcher / DCP 约束。

2. init_distributed_environment()
   看 DP 是否被纳入 torch distributed world，rank 是否被 DP offset。

3. initialize_model_parallel()
   看 all_ranks reshape 和 TP / DCP / PCP / PP / DP / EP group_ranks。

4. ModelConfig.verify_with_parallel_config()
   看 TP heads、PP 支持性、DCP + GQA/MQA 约束。

5. ModelConfig.get_layers_start_end_indices()
   看当前 PP rank 持有哪些 layers。

6. Linear / Attention / MoE layer 初始化
   看 TP shard、KV heads、本地 experts 如何决定。

7. GPUWorker.execute_model()
   看 PP recv / send intermediate tensors。

8. GPUModelRunner._determine_batch_execution_and_padding()
   看 DP padding、cudagraph、ubatching 如何同步。

9. GPUModelRunner._build_attention_metadata()
   看 DCP local seq lens、block table、slot mapping。

10. LogitsProcessor / sampler
    看 last PP stage 和 TP logits gather。
```

常看位置：

```text
code/vllm/vllm/config/parallel.py:117
code/vllm/vllm/config/parallel.py:490
code/vllm/vllm/config/parallel.py:782
code/vllm/vllm/distributed/parallel_state.py:1674
code/vllm/vllm/distributed/parallel_state.py:1757
code/vllm/vllm/config/model.py:1157
code/vllm/vllm/config/model.py:1280
code/vllm/vllm/v1/worker/gpu_worker.py:853
code/vllm/vllm/v1/worker/gpu_model_runner.py:3879
code/vllm/vllm/model_executor/layers/logits_processor.py:75
```

---

## 39. 最终可以记成一张表

| 并行方式 | 是否增加 worker | group 如何形成 | 主要切什么 | 典型消费者 |
|---|---|---|---|---|
| TP | 是 | 固定 DP/PP/PCP，沿 TP 维度 | tensor、heads、vocab shard | linear、attention、logits |
| PP | 是 | 固定 DP/PCP/TP，沿 PP 维度 | transformer layers | worker send/recv、model runner |
| PCP | 是 | 固定 DP/PP/TP，沿 PCP 维度 | prefill context | attention backend |
| DCP | 否 | TP group 内按 DCP size 切 | decode context | attention / MLA backend |
| DP | 是 | 固定 PP/PCP/TP，沿 DP 维度 | request replica | scheduler、dp_utils、MoE EP |
| EP | 否 | 固定 PP，合并 DP x PCP x TP | MoE experts | fused_moe all-to-all |
| EPLB | 否 | 与 EP ranks 相同但独立 group | expert 负载平衡 | eplb |

再按状态归属记：

| 状态 | 归属 |
|---|---|
| dense 权重 | TP shard + PP layer range |
| MoE expert 权重 | EP shard + PP layer range |
| request / scheduler state | DP replica-local |
| KV cache | DP replica-local、PP stage-local、TP/CP rank-local |
| block table / slot mapping | DP replica-local、本 rank执行态 |
| logits | PP last stage、TP gather 后完整 vocab |
| sampling | PP last stage、DP replica-local |

---

## 40. 总结

TP / PP / DP / EP / CP 的组合可以压缩成：

```text
ParallelConfig
  → world_size = PP * TP * PCP
  → world_size_across_dp = world_size * DP
  → all_ranks[ExternalDP, DP, PP, PCP, TP]
  → parallel_state 创建 TP / DCP / PCP / PP / DP / EP / EPLB groups
  → dense layer 用 TP
  → pipeline stage 用 PP
  → prefill / decode attention 用 PCP / DCP
  → MoE layer 用 EP
  → replica 级请求与执行形态同步用 DP
```

如果只记一句话：

```text
vLLM 的组合并行不是把所有维度简单相乘：PP、TP、PCP 决定单个 DP replica 的 worker mesh；DP 复制这套 mesh；DCP 复用 TP rank；EP 在 PP stage 内重组 DP x PCP x TP rank 来服务 MoE experts。
```

最小心智模型：

```text
TP 横切层内张量；
PP 纵切模型层；
DP 复制模型副本切请求；
PCP / DCP 切 attention context；
EP 切 MoE experts；
EPLB 单独负责 expert 负载平衡。
```
