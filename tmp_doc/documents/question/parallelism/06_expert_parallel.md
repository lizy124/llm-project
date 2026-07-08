# 06. Expert Parallel 如何服务 MoE？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/device_communicators/base_device_communicator.py`
- `vllm/vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/vllm/distributed/device_communicators/all2all.py`
- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/vllm/model_executor/layers/fused_moe/config.py`
- `vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py`
- `vllm/vllm/model_executor/layers/fused_moe/router/`
- `vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `vllm/vllm/model_executor/layers/fused_moe/routed_experts.py`
- `vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/`
- `vllm/vllm/model_executor/model_loader/ep_weight_filter.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/eplb_utils.py`

本问题关注：Expert Parallel（EP）在 vLLM 中如何服务 MoE 模型，包括 expert 如何分布到 rank、router/top-k 如何产生 token 到 expert 的映射、token 如何 dispatch 到拥有目标 expert 的 rank、local expert 如何计算、输出如何 combine 回原 token 位置，以及 EP 如何和 TP / DP / PP / PCP、all-to-all backend、EPLB、量化 MoE kernel 协同。

---

## 1. 一句话回答

Expert Parallel 是 MoE 层内部的 expert 级并行方式：

```text
普通 TP：
  把同一个 dense / expert 矩阵切到多个 rank 上算。

Expert Parallel：
  把不同 experts 分给不同 rank；典型 EP 路径中每个 rank 保存并计算完整 experts 的子集。
```

对一次 MoE forward，可以记成：

```text
hidden states
  → router logits
  → top-k expert ids / weights
  → 根据 expert ids 把 token 发到对应 expert rank
  → 每个 rank 只算本地 experts
  → expert output 再通信回原 token 所属 rank
  → 按 top-k weights 合并
  → MoE layer output
```

一句话记忆：

```text
EP 是“token 按 router 结果流向 expert 所在 rank”；启用 EP 的 MoE 层里，每个 rank 通常持有完整 experts 的子集，而不是继续把单个 expert 矩阵按 TP 切分。
```

---

## 2. 本文要回答的问题

```text
1. enable_expert_parallel 打开后，MoE 层和普通 TP 有什么不同？
2. EP group 是怎么创建的？它和 DP / TP / PP / PCP 的关系是什么？
3. experts 如何分配到 EP rank？linear 和 round_robin placement 有什么区别？
4. router logits、top-k expert ids、top-k weights 在哪里产生？
5. token dispatch / combine 是由谁实现的？
6. all2all_backend 里的 allgather_reducescatter / DeepEP / FlashInfer NVLink / MoRI / NIXL 分别是什么层次？
7. modular MoE kernel 里的 prepare / experts / finalize 如何对应 EP 通信？
8. EPLB 如何改变 logical expert 到 physical expert 的映射？
9. EP 如何影响权重加载、shared experts、LoRA、sequence parallel 和 CUDA graph？
```

---

## 3. 最小主链路

最小链路可以分成 5 段：

```text
1. 分布式初始化
   initialize_model_parallel()
     → 创建 TP / DCP / PCP / PP / DP / EP / EPLB groups

2. MoE 层构造
   FusedMoE(...)
     → FusedMoEParallelConfig.make()
     → ExpertMapManager
     → Router
     → RoutedExperts
     → MoERunner

3. forward 入口
   MoERunner.forward()
     → _forward_entry custom op
     → _forward_impl()

4. routing + expert compute
   router.select_experts()
     → topk_weights / topk_ids
     → routed_experts.forward_modular() 或 forward_monolithic()
     → quant_method / modular kernel

5. EP 通信 + 回收
   prepare / dispatch
     → local expert compute
     → finalize / combine
     → final all-reduce 或 reduce-scatter
     → MoE output
```

如果只看 token 数据流：

```text
每个 rank 原本持有一段 tokens
  → all-gather / all-to-all dispatch，让拥有目标 expert 的 rank 看到相关 tokens
  → 本 rank 用本地 experts 计算这些 tokens 的 expert output
  → reduce-scatter / all-to-all combine，把结果还给原 token rank
```

---

## 4. 配置入口：哪些参数控制 EP

EP 相关配置集中在 `ParallelConfig`。

源码位置：`vllm/vllm/config/parallel.py:162`

核心字段：

```text
enable_expert_parallel：
  是否用 expert parallel 替代 MoE 层中的 tensor parallel。

enable_ep_weight_filter：
  EP 启用时，加载权重时跳过非本地 expert 权重，减少 MoE 权重 I/O。

enable_eplb：
  是否启用 expert parallel load balancing。

expert_placement_strategy：
  expert 放置策略，支持 linear / round_robin。

all2all_backend：
  MoE EP token dispatch / combine 使用的通信 backend。

enable_elastic_ep：
  是否启用 elastic expert parallelism，使用 stateless NCCL groups。
```

### 4.1 `enable_expert_parallel`

配置说明是：

```text
Use expert parallelism instead of tensor parallelism for MoE layers.
```

也就是说，它不是额外再叠一层 TP，而是在 MoE experts 的分布方式上改变策略。

### 4.2 `all2all_backend`

默认值：

```text
allgather_reducescatter
```

可选项在配置注释中包括：

```text
allgather_reducescatter：
  用 all-gather 做 dispatch，用 reduce-scatter 做 combine。

deepep_high_throughput：
  DeepEP high-throughput kernels。

deepep_low_latency：
  DeepEP low-latency kernels。

mori_high_throughput / mori_low_latency：
  MoRI EP backend。

nixl_ep：
  NIXL EP kernels。

flashinfer_nvlink_two_sided：
  FlashInfer two-sided NVLink kernels。

flashinfer_nvlink_one_sided：
  FlashInfer high-throughput one-sided NVLink kernels。
```

源码位置：`vllm/vllm/config/parallel.py:185`

旧的 `pplx` / `naive` 会被降级成 `allgather_reducescatter`。

源码位置：`vllm/vllm/config/parallel.py:449`

### 4.3 `enable_eplb`

EPLB 是 Expert Parallel Load Balancing。配置校验要求：

```text
- 当前平台必须是 CUDA-like；
- enable_expert_parallel 必须为 True；
- tensor_parallel_size * data_parallel_size 必须大于 1；
- 如果没启用 EPLB，则 num_redundant_experts 必须为 0。
```

源码位置：`vllm/vllm/config/parallel.py:475`

---

## 5. EP group 是怎么创建的

EP group 在 `initialize_model_parallel()` 里创建。

源码位置：`vllm/vllm/distributed/parallel_state.py:1694`

整体并行 rank layout 是：

```text
ExternalDP x DP x PP x PCP x TP
```

源码位置：`vllm/vllm/distributed/parallel_state.py:1760`

`initialize_model_parallel()` 会依次创建：

```text
TP group
DCP group
PCP group
PP group
DP group
EP group
EPLB group（如果 enable_eplb=True）
```

### 5.1 EP group 的 rank 组成

EP group 的构造核心是：

```text
all_ranks.transpose(1, 2)
  .reshape(-1, data_parallel_size * pcp_size * tp_size)
```

源码位置：`vllm/vllm/distributed/parallel_state.py:1870`

这意味着：

```text
EP group 跨 DP、PCP、TP 维度；
但按 PP stage 隔离。
```

直观理解：

```text
同一个 pipeline stage 内，MoE experts 可以分布到 DP / PCP / TP 展平后的 ranks 上；
不同 pipeline stage 各自有自己的 EP group，因为它们持有不同层。
```

### 5.2 dense 模型不创建 EP group

源码里有注释：

```text
Don't create EP group for dense models.
```

如果 `model_config` 存在且不是 MoE，则 `_EP` 保持 `None`。

源码位置：`vllm/vllm/distributed/parallel_state.py:1870`

`get_ep_group()` 也明确说明：EP group 只为 MoE 模型创建。

源码位置：`vllm/vllm/distributed/parallel_state.py:1378`

### 5.3 EPLB group 为什么单独建

如果启用 EPLB，会用和 EP 相同的 ranks 创建单独的 EPLB group。

源码位置：`vllm/vllm/distributed/parallel_state.py:1898`

原因是：

```text
EPLB 通信要和 MoE forward pass collectives 隔离，避免 forward 中的 distributed 通信和 EPLB 通信互相死锁。
```

---

## 6. FusedMoEParallelConfig 如何决定用不用 EP

MoE 层构造时会调用 `make_parallel_config()`。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:43`

它再调用：

```text
FusedMoEParallelConfig.make(...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/config.py:1107`

### 6.1 use_ep 的判定

`use_ep` 的条件是：

```text
dp_size * pcp_size * tp_size > 1
and parallel_config.enable_expert_parallel
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/config.py:1188`

也就是说：

```text
单卡或没有任何跨 rank 并行时，即使打开 enable_expert_parallel，也不会真的形成 ep_size > 1。
```

### 6.2 没有 EP 时：DP / PCP 会被折叠进 TP

如果 `use_ep=False`，配置会调用 `flatten_tp_across_dp_and_pcp()`：

```text
flatten_tp_size = dp_size * pcp_size * tp_size
flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/config.py:1097`

含义是：

```text
不用 EP 时，MoE expert 权重仍然按类似 TP 的方式切分；
DP / PCP 维度也被展平成“更大的 TP”参与 MoE 权重分片。
```

### 6.3 启用 EP 时：MoE 内 TP 被置为 1，EP 接管 expert 分片

如果 `use_ep=True`：

```text
ep_size = flattened_tp_size
ep_rank = flattened_tp_rank
tp_size = 1
tp_rank = 0
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/config.py:1216`

源码注释直接说明：

```text
In EP, each device owns a set of experts fully.
There is no tensor parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
```

这就是 EP 的关键：

```text
每个 rank 保存完整的若干 experts，expert 内部不再按 TP 切。
```

---

## 7. expert 如何分配到 EP rank

expert 分配由 `ExpertMapManager` 管理。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:152`

它负责：

```text
- 计算当前 rank 有多少 local experts；
- 生成 global expert id → local expert id 的 expert_map；
- 维护 expert placement strategy；
- 为 round_robin placement 准备 routing tables；
- 支持 EPLB / elastic EP 时更新映射。
```

### 7.1 expert_map 的语义

当 EP 未启用时：

```text
expert_map = None
所有 experts 都是 local。
```

当 EP 启用时：

```text
expert_map.shape = [global_num_experts]
expert_map[global_id] = local_id   如果该 expert 在当前 rank
expert_map[global_id] = -1         如果该 expert 在其他 rank
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:296`

### 7.2 linear placement

`linear` 是默认策略。

例如 4 个 experts、2 个 EP ranks：

```text
rank 0：expert 0, 1
rank 1：expert 2, 3
```

源码位置：`vllm/vllm/config/parallel.py:175`

实现逻辑：

```text
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
start_idx = ep_rank * base_experts + min(ep_rank, remainder)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:66`

### 7.3 round_robin placement

`round_robin` 是交错放置。

例如 4 个 experts、2 个 EP ranks：

```text
rank 0：expert 0, 2
rank 1：expert 1, 3
```

实现逻辑：

```text
local_log_experts = arange(ep_rank, global_num_experts, ep_size)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:80`

### 7.4 round_robin 不是总能用

round_robin 需要满足：

```text
- 模型有多个 expert group；
- 没有 redundant experts；
- 未启用 EPLB；
- 如果使用 all2all kernels，backend 需要 DeepEP low-latency 或 NIXL EP 这类需要 routing tables 的路径。
```

不满足时会 fallback 到 linear。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:116`

---

## 8. MoE 层构造时创建了哪些对象

`FusedMoE()` 是 MoE 执行 pipeline 的工厂函数。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:102`

它创建的核心对象是：

```text
FusedMoEParallelConfig
  → 本 MoE 层使用 TP 还是 EP，以及 ep_size / ep_rank。

ExpertMapManager
  → expert placement / expert_map / routing tables。

FusedMoERouter
  → 从 router logits 选择 top-k experts。

RoutedExperts
  → 持有本 rank 的 expert 权重，并委托 quant method 执行。

MoERunner
  → 组织完整 MoE forward。
```

构造顺序大致是：

```text
FusedMoE(...)
  → make_parallel_config()
  → determine_expert_counts()
  → ExpertMapManager(...)
  → create_fused_moe_router(...)
  → FusedMoEConfig(...)
  → RoutedExperts(...)
  → MoERunner(...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:216`

### 8.1 `global_num_experts` 和 `logical_num_experts`

`determine_expert_counts()` 会区分：

```text
logical_num_experts：
  模型语义上的真实 routed experts 数。

global_num_experts：
  logical experts + redundant experts。

num_fused_shared_experts：
  ROCm AITER 特定的 fused shared experts 数。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:72`

启用 EPLB 时，`num_redundant_experts` 会增加 physical expert 数量。

---

## 9. router / top-k 选择在哪里发生

MoE forward 中，`MoERunner._apply_quant_method()` 负责触发 routing。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:534`

如果是 modular kernel：

```text
topk_weights, topk_ids = self.router.select_experts(
  hidden_states,
  router_logits,
  topk_indices_dtype=...,
  input_ids=input_ids,
)

fused_out = self.routed_experts.forward_modular(
  x=hidden_states,
  topk_weights=topk_weights,
  topk_ids=topk_ids,
)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:559`

如果是 monolithic kernel：

```text
fused_out = self.routed_experts.forward_monolithic(
  x=hidden_states,
  router_logits=router_logits,
)
```

也就是 kernel 内部自己处理 routing。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:552`

### 9.1 Router 工厂如何选择路由器

`create_fused_moe_router()` 会根据参数选择不同 Router。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/router_factory.py:39`

优先级：

```text
1. RoutingSimulatorRouter
2. ZeroExpertRouter
3. GroupedTopKRouter
4. CustomRoutingRouter
5. FusedTopKBiasRouter
6. AiterSharedRoutedFusedMoERouter
7. FusedTopKRouter
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/router_factory.py:66`

### 9.2 默认 FusedTopKRouter

默认 router 调用 fused top-k：

```text
fused_topk(...)
  → topk_weights
  → topk_ids
  → token_expert_indices
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:69`

softmax 路由调用：

```text
ops.topk_softmax(...)
```

sigmoid 路由调用：

```text
ops.topk_sigmoid(...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:17`

### 9.3 EPLB 会改写 topk_ids

`BaseRouter._select_experts()` 的模板流程是：

```text
1. 校验 EPLB state；
2. _compute_routing() 产生 logical topk ids；
3. capture logical ids；
4. 如果启用 EPLB，把 logical ids 映射成 physical ids，并记录 load；
5. 转成 kernel 需要的 dtype。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/base_router.py:233`

所以在无 EPLB 时：

```text
topk_ids ≈ global logical expert ids
```

在 EPLB 时：

```text
topk_ids = physical expert ids
```

---

## 10. RoutedExperts 负责什么

`RoutedExperts` 是 routed expert 权重容器和执行代理。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:43`

它负责：

```text
- 创建本 rank 的 expert 权重参数；
- 加载 checkpoint 中属于本 rank 的 expert shard；
- 持有 quant_method；
- 暴露 expert_map；
- 执行 forward_modular() / forward_monolithic()。
```

### 10.1 权重加载时如何跳过非本地 expert

加载单个 expert 权重时，会先把 global expert id 映射到 local expert id：

```text
expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)
```

如果当前 expert 不在本 rank：

```text
expert_id == -1
→ 不加载这个 param
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:592`

这就是 EP 下“每个 rank 只持有本地 experts”的关键之一。

### 10.2 quant_method 是 MoE kernel 的入口

`RoutedExperts` 不直接写所有 kernel 逻辑，而是委托 `quant_method`：

```text
forward_modular()
  → quant_method.apply(...)

forward_monolithic()
  → quant_method.apply_monolithic(...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:1053`

不同量化 / kernel 后端，例如 Triton MoE、Cutlass MoE、DeepGEMM、FlashInfer Cutlass、TRTLLM MoE 等，会通过不同 quant method / experts 实现接入。

---

## 11. MoERunner 的 forward 主链路

`MoERunner.forward()` 是 MoE 层运行时入口。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:628`

主链路是：

```text
MoERunner.forward(hidden_states, router_logits)
  → apply_routed_input_transform()
  → _maybe_pad_hidden_states()
  → torch.ops.vllm.moe_forward / moe_forward_shared
  → _forward_impl()
  → _maybe_dispatch()
  → _apply_quant_method()
  → _maybe_combine()
  → shared/routed output 合并
  → _maybe_reduce_final_output()
```

### 11.1 为什么有 custom op

`moe_forward` / `moe_forward_shared` 是注册的自定义 op。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:193`

它们的作用不是实现逻辑，而是把 MoE 层作为 opaque op 交给编译系统处理，再在内部取回对应 `MoERunner` 调 `_forward_impl()`。

源码注释说明：

```text
_moe_forward 和 _moe_forward_shared 不包含实现细节，只负责转发到 runner._forward_impl。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:113`

### 11.2 `_maybe_dispatch()` 做什么

`_maybe_dispatch()` 处理旧的 naive dispatch/combine 路径，以及 PCP gather。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:724`

当：

```text
dp_size > 1
and quant_method 不支持 internal modular kernel
```

会调用：

```text
get_ep_group().dispatch_router_logits(hidden_states, router_logits, ...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:719`

如果 `pcp_size > 1`，还会对 hidden states 和 router logits 做 `get_pcp_group().all_gather(dim=0)`。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:745`

### 11.3 `_maybe_combine()` 做什么

对应地，`_maybe_combine()` 会：

```text
get_ep_group().combine(hidden_states, ...)
```

以及 PCP 的：

```text
get_pcp_group().reduce_scatter(dim=0)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:757`

---

## 12. modular kernel：prepare / experts / finalize

vLLM 新的 MoE modular kernel 把 MoE 拆成四段：

```text
Router
  → Quantize / Dispatch
  → Permute / Experts / Unpermute
  → Combine / Reduce
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:45`

关键抽象：

```text
FusedMoEPrepareAndFinalizeModular：
  负责 prepare 和 finalize，通常包含量化、dispatch、combine、reduce。

FusedMoEExpertsModular：
  负责真正的 expert GEMM + activation + GEMM。

FusedMoEKernel：
  把 prepare/finalize 和 experts 拼成标准 MoE kernel 接口。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:180`

### 12.1 prepare 的输入输出

`prepare()` 输入：

```text
a1：hidden states
topk_weights
topk_ids
num_experts
expert_map
apply_router_weight_on_input
quant_config
defer_input_quant
```

输出：

```text
a1q：量化后或 dispatch 后的 hidden states
a1q_scale：可选量化 scale
expert_tokens_meta：每个 local expert 的 token 统计
dispatched topk_ids：可选
dispatched topk_weights：可选
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:263`

### 12.2 experts 阶段

`_fused_experts()` 会根据 prepare 后的数据调用真正专家 kernel：

```text
fused_experts.apply(
  hidden_states=a1q,
  w1,
  w2,
  topk_weights,
  topk_ids,
  expert_map,
  expert_tokens_meta,
)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:1205`

这里的 `expert_map` 让 kernel 知道：

```text
global expert id → 当前 rank 的 local expert id；
不在本 rank 的 expert 是 -1。
```

### 12.3 finalize 的职责

`finalize()` 输入：

```text
output：最终输出 buffer
fused_expert_output：expert kernel 的原始输出
topk_weights
topk_ids
apply_router_weight_on_input
weight_and_reduce_impl
```

它负责：

```text
- 对 top-k expert output 应用 routing weights；
- 按 token 维度 reduce top-k 输出；
- 如果 prepare 阶段做过 dispatch，则做 combine；
- 必要时跨 rank reduce。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:353`

---

## 13. NoDPEP 与 NaiveDPEP 的区别

### 13.1 NoDPEP：没有跨 DP/EP dispatch

`MoEPrepareAndFinalizeNoDPEPModular` 用于不需要 DP/EP token 通信的路径。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py:40`

prepare 只做：

```text
- 可选地把 topk_weights 乘到 input 上；
- 量化 input；
- 不改变 topk_ids / topk_weights。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py:57`

finalize 只做：

```text
TopKWeightAndReduce.apply(...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py:80`

### 13.2 NaiveDPEP：用 EP group dispatch/combine

`MoEPrepareAndFinalizeNaiveDPEPModular` 用于 DP/EP 的朴素路径。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:71`

prepare 做：

```text
1. 可选地把 topk_weights 乘到 input 上；
2. 量化 input；
3. 把 hidden_states / topk_weights / topk_ids / extra_tensors 一起 dispatch；
4. 返回 dispatch 后的 a1q / topk_weights / topk_ids。
```

核心调用：

```text
get_ep_group().dispatch(a1q, topk_weights, topk_ids, ...)
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:158`

finalize 做：

```text
1. 先在本地对 top-k output 应用 weight_and_reduce；
2. 再 get_ep_group().combine(out, ...)
3. copy 回 output。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:187`

### 13.3 monolithic NaiveDPEP

monolithic 版本处理 router logits，而不是预先算好的 topk ids / weights：

```text
prepare：dispatch hidden_states + router_logits
finalize：combine fused_expert_output
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:212`

---

## 14. EP 通信最终落到哪里

`GroupCoordinator` 暴露了 MoE 专用的三个接口：

```text
dispatch_router_logits(...)
dispatch(...)
combine(...)
```

源码位置：`vllm/vllm/distributed/parallel_state.py:1201`

它们会转给当前 group 的 `device_communicator`：

```text
GroupCoordinator
  → DeviceCommunicator
  → all2all_manager
```

如果没有 device communicator，则退化为 no-op。

### 14.1 device communicator 什么时候启用 all2all manager

`DeviceCommunicatorBase` 会判断：

```text
self.is_ep_communicator = unique_name.split(":")[0] == "ep"
self.use_all2all = self.is_ep_communicator and use_ep
```

其中 `use_ep` 当前根据 `parallel_config.data_parallel_size > 1` 设置。

源码位置：`vllm/vllm/distributed/device_communicators/base_device_communicator.py:163`

### 14.2 CUDA communicator 如何选择 all2all manager

CUDA 下，如果 `use_all2all=True`，会根据 `all2all_backend` 创建 manager。

源码位置：`vllm/vllm/distributed/device_communicators/cuda_communicator.py:121`

大致映射：

```text
allgather_reducescatter：
  AgRsAll2AllManager

deepep_high_throughput：
  DeepEPHTAll2AllManager

deepep_low_latency：
  DeepEPLLAll2AllManager

mori_high_throughput / mori_low_latency：
  MoriAll2AllManager

deepep_v2：
  DeepEPV2All2AllManager

nixl_ep：
  NixlEPAll2AllManager

flashinfer_nvlink_two_sided：
  FlashInferNVLinkTwoSidedManager

flashinfer_nvlink_one_sided：
  FlashInferNVLinkOneSidedManager
```

### 14.3 AgRsAll2AllManager 的语义

默认 `allgather_reducescatter` 的实现是：

```text
dispatch：
  all_gatherv(hidden_states, topk_weights, topk_ids, extras)

combine：
  reduce_scatterv(hidden_states)
```

源码位置：`vllm/vllm/distributed/device_communicators/all2all.py:42`

`dispatch()` 会根据当前 batch 的 DP chunk sizes 做 all-gatherv：

```text
sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
```

源码位置：`vllm/vllm/distributed/device_communicators/all2all.py:85`

`combine()` 会用相同 sizes 做 reduce-scatterv。

源码位置：`vllm/vllm/distributed/device_communicators/all2all.py:125`

---

## 15. all2all_backend 和 prepare/finalize backend 的关系

注意有两层“backend”：

```text
all2all manager：
  分布式 communicator 层，负责 dispatch/combine 的通信实现。

prepare_finalize：
  MoE modular kernel 层，负责把量化、dispatch、combine、weight/reduce 组织起来。
```

对于朴素路径：

```text
prepare_finalize 调 get_ep_group().dispatch/combine；
get_ep_group 再转到 all2all_manager。
```

对于 DeepEP / FlashInfer NVLink / NIXL 等优化路径：

```text
prepare_finalize 可能直接使用对应高性能 kernel 的格式和异步接口，
而不是只靠朴素 dispatch/combine 包装。
```

这也是为什么 `prepare_finalize/` 目录下既有：

```text
no_dp_ep.py
naive_dp_ep.py
batched.py
```

也有：

```text
deepep_ht.py
deepep_ll.py
deepep_v2.py
flashinfer_nvlink_one_sided.py
flashinfer_nvlink_two_sided.py
mori.py
nixl_ep.py
```

这些文件不是 MoE expert GEMM 本身，而是“MoE 输入准备 + EP 通信 + 输出回收”的不同实现。

---

## 16. EP 和 TP / DP / PP / PCP 的关系

### 16.1 EP + TP

在 MoE 层里，启用 EP 后：

```text
tp_size = 1
ep_size = 原本展平后的 TP × DP × PCP size
```

也就是说：

```text
MoE experts 不再按 TP 切矩阵；
而是每个 rank 保存完整 experts 的子集。
```

但这只影响 MoE layers。

模型里的 dense attention / MLP / embedding / lm_head 等，仍然可以按普通 tensor parallel 运行。

### 16.2 EP + DP

DP 在 vLLM 的 MoE EP 中有两个角色：

```text
请求 / batch 层面：
  不同 DP rank 原本持有不同 token chunk。

MoE expert 层面：
  DP ranks 也可以参与同一个 EP group，把 experts 分散到更多设备上。
```

所以 EP 的通信经常跨 DP ranks：

```text
每个 DP rank 的 tokens 需要被 dispatch 到 expert 所在 rank；
expert output 再 reduce-scatter / combine 回对应 DP rank。
```

### 16.3 EP + PP

EP group 按 pipeline stage 隔离。

这意味着：

```text
只有包含 MoE layer 的 PP stage 会用自己的 EP group；
不同 PP stage 的 MoE layers 不共享同一组 experts。
```

### 16.4 EP + PCP

`FusedMoEParallelConfig` 把 PCP 当成类似 DP 的维度参与 EP 展平。

源码注释说明：

```text
PCP serves the same function as DP here.
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/config.py:1135`

MoERunner 中如果 `pcp_size > 1`，会额外：

```text
forward 前 all_gather hidden_states / router_logits；
forward 后 reduce_scatter hidden_states。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:745`

---

## 17. shared experts 如何配合 EP

很多 MoE 模型除了 routed experts，还有 shared experts。

`MoERunner` 中 shared experts 由 `SharedExperts` 包装。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:277`

shared experts 的输出和 routed experts 的输出最后相加：

```text
if shared_output is not None:
  result = shared_output + fused_output
else:
  result = fused_output
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:709`

### 17.1 shared expert 的通信时机

如果 fused MoE combine kernel 已经把 routed output 做了跨 rank reduction，那么 shared output 需要单独 all-reduce：

```text
_maybe_reduce_shared_expert_output()
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:411`

否则会先把 shared + routed 加起来，再统一 all-reduce：

```text
_maybe_reduce_final_output()
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:431`

这两条 all-reduce 路径互斥。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:683`

### 17.2 shared experts 可以和 all2all overlap

modular kernel 的异步 prepare/finalize 支持让 shared experts 和 dispatch/combine 重叠。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:1104`

如果 prepare/finalize 支持 async：

```text
prepare_async / finalize_async
  → 可以在等待通信时执行 shared experts
```

---

## 18. EPLB：Expert Parallel Load Balancing

EPLB 的目标是：

```text
当某些 experts 被 router 频繁选中时，通过 redundant experts 和 logical→physical 映射，
把热点 logical expert 的流量分散到多个 physical expert replica 上。
```

### 18.1 EPLB 在构造阶段做什么

`FusedMoE()` 中：

```text
if enable_eplb:
  eplb_state = EplbLayerState()
```

并且要求：

```text
use_ep=True
如果 global_num_experts % ep_size != 0，会报错
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/layer.py:233`

### 18.2 EPLB 在 router 阶段做什么

router 先产生 logical expert ids，然后 `_apply_eplb_mapping()` 转成 physical expert ids，并记录 expert load。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/base_router.py:181`

CUDA-like 平台下，这个 mapping + record 是一个 Triton kernel：

```text
_eplb_map_and_record_i32_kernel
```

它做两件事：

```text
1. logical expert id → physical expert id；
2. atomic_add 记录 physical expert 的 token load。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/router/base_router.py:17`

### 18.3 EPLB state 如何注册到每一层

`MoERunner.set_eplb_state()` 会把当前 MoE layer 的：

```text
expert_load_view
logical_to_physical_map
logical_replica_count
```

注册到 router 的 `EplbLayerState`。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:959`

---

## 19. 权重加载与 EP weight filter

EP 下每个 rank 只需要本地 experts 的权重。

`RoutedExperts.weight_loader()` 会根据 expert_map 判断权重是否属于本 rank：

```text
local_id = expert_map[global_expert_id]
if local_id == -1:
  skip
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:592`

如果启用 `enable_ep_weight_filter`，则可以在模型加载阶段更早跳过非本地 expert 权重，减少 I/O。

配置说明位置：`vllm/vllm/config/parallel.py:164`

它对下面这种权重特别有意义：

```text
checkpoint 里每个 expert 是独立 tensor：
  可以按 expert id 过滤。

checkpoint 里所有 experts 是 3D fused tensor：
  过滤收益有限或不适用。
```

---

## 20. sequence parallel / LoRA / DBO 的挂接点

### 20.1 sequence parallel

`MoERunner._sequence_parallel_context()` 会从 forward context 里拿 DP metadata，并建立 SP local size 上下文。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:585`

dispatch/combine 时会把 `is_sequence_parallel` 传给 EP group：

```text
get_ep_group().dispatch(..., is_sequence_parallel=True/False)
get_ep_group().combine(..., is_sequence_parallel=True/False)
```

对应 all2all manager 会在 sequence parallel 时选择 EP group，否则常规 DP group。

源码位置：`vllm/vllm/distributed/device_communicators/all2all.py:68`

### 20.2 LoRA

NaiveDPEP prepare 支持把 per-token LoRA mapping 作为 `extra_tensors` 一起 dispatch。

源码位置：`vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:136`

这样本地 expert rank 收到 token 后，也能知道每个 token 对应哪个 LoRA。

### 20.3 DBO

modular kernel 的 prepare/finalize 支持 async 接口，并和 DBO hook 协作：

```text
prepare_async()
finalize_async()
dbo_register_recv_hook()
dbo_yield()
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:1117`

这让 MoE all2all 通信可以和其他计算重叠。

---

## 21. 不同 MoE kernel 后端和 EP 的关系

EP 本身不等于某一个 expert GEMM kernel。

可以把 MoE 执行拆成两层：

```text
通信 / token 重排层：
  prepare_finalize / all2all backend

专家计算层：
  fused experts kernel / quant method
```

专家计算后端可能包括：

```text
Triton MoE
Cutlass MoE
DeepGEMM MoE
FlashInfer Cutlass / CuteDSL MoE
TRTLLM MoE
ROCm AITER MoE
XPU MoE
CPU MoE
fallback MoE
```

这些后端通过 `quant_method`、`FusedMoEExpertsModular` / `FusedMoEExpertsMonolithic`、oracle 选择等机制接入。

它们需要声明自己是否支持：

```text
- 当前 device；
- 当前 quant scheme；
- 当前 routing method；
- 当前 parallel config；
- LoRA；
- batch invariance；
- activation format。
```

源码位置：`vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:536`

---

## 22. 常见形状变化

假设：

```text
本 rank 原始 token 数：M_local
EP/DP group 总 token 数：M_global
hidden dim：H
top_k：K
```

### 22.1 router 前

```text
hidden_states: [M_local, H]
router_logits: [M_local, num_experts]
```

### 22.2 top-k 后

```text
topk_weights: [M_local, K]
topk_ids: [M_local, K]
```

### 22.3 dispatch 后

朴素 AgRs 路径会 all-gather：

```text
hidden_states: [M_global, H]
topk_weights: [M_global, K]
topk_ids: [M_global, K]
```

高级 all2all 路径可能不是简单 gather 全量，而是按目标 expert rank 重新排列 / 分桶，只把相关 token 发给目标 rank。

### 22.4 local expert compute

本 rank 只对本地 experts 计算：

```text
local experts: [local_num_experts, ...]
expert output: 通常仍能还原到 token/top-k 对应输出
```

### 22.5 combine 后

输出回到当前 rank token chunk：

```text
output: [M_local, H]
```

这也是为什么 combine 常用 reduce-scatter 或等价的 all2all 回传。

---

## 23. 调试 EP 时看哪里

如果要判断 EP 是否生效，可以按这个顺序查：

```text
1. 配置层
   parallel_config.enable_expert_parallel
   parallel_config.data_parallel_size / tensor_parallel_size / pcp_size
   parallel_config.all2all_backend
   parallel_config.enable_eplb

2. group 初始化
   parallel_state.initialize_model_parallel()
   get_ep_group().world_size / rank_in_group

3. MoE 层并行配置
   FusedMoEParallelConfig.use_ep
   ep_size / ep_rank
   tp_size / tp_rank 是否在 MoE 内变成 1 / 0

4. expert map
   ExpertMapManager.local_num_experts
   expert_map
   placement_strategy
   routing_tables

5. router 输出
   topk_ids 是 logical ids 还是 EPLB 后的 physical ids
   topk_weights 是否符合预期

6. prepare/finalize
   使用 NoDPEP、NaiveDPEP、DeepEP、FlashInfer NVLink、MoRI、NIXL 哪条路径

7. communicator
   CUDA communicator 是否创建了 all2all_manager
   all2all_manager 类型是否与 all2all_backend 一致
```

日志里常见线索：

```text
[EP Rank x/y] Expert parallelism is enabled...
Using xxx all2all manager.
rank ... is assigned as DP rank, PP rank, PCP rank, TP rank, EP rank, EPLB rank
```

---

## 24. 最容易混淆的几个点

### 24.1 EP 不是普通 all-reduce

普通 TP 常见通信是 all-reduce / all-gather / reduce-scatter。

EP 的核心是：

```text
token 根据 expert id 发生跨 rank shuffle。
```

默认实现可以用 all-gather + reduce-scatter 模拟，但语义上是 MoE token dispatch / combine。

### 24.2 `enable_expert_parallel=True` 不一定代表每个 MoE kernel 都走高性能 all2all

真正路径还取决于：

```text
- dp_size / tp_size / pcp_size 是否形成 ep_size > 1；
- quant method 是否支持 internal modular kernel；
- all2all_backend；
- prepare_finalize 是否被替换为 DeepEP / FlashInfer / NIXL 等实现；
- 当前 device 和量化格式是否被 kernel 支持。
```

### 24.3 EP group 和 DP group 不是同一个概念

EP group 横跨 DP / PCP / TP 维度，用于 MoE expert 分布。

DP group 是数据并行组，用于请求 / batch 维度的并行执行和部分同步。

默认 AgRs manager 在非 sequence-parallel 时可能使用 DP group 做 all-gatherv / reduce-scatterv，但这是该 backend 的实现选择，不代表 EP group 和 DP group 等价。

### 24.4 expert_map 不是 router 输出

router 输出：

```text
topk_ids: 每个 token 选择哪些 expert
```

expert_map：

```text
global expert id → 当前 rank local expert id
```

它们在 expert kernel 中配合使用。

### 24.5 EPLB 会让 topk_ids 从 logical id 变成 physical id

无 EPLB：

```text
topk_ids 表示模型逻辑 expert。
```

有 EPLB：

```text
topk_ids 会被映射为 physical expert replica。
```

因此调试 router 负载时要区分 logical expert load 和 physical expert load。

---

## 25. 最终可以记成一张表

| 阶段 | 主要代码 | 核心对象 | 作用 |
|---|---|---|---|
| 配置 | `config/parallel.py` | `enable_expert_parallel`、`all2all_backend`、`enable_eplb` | 决定是否启用 EP、用什么通信 backend |
| group 初始化 | `distributed/parallel_state.py` | `_EP`、`_EPLB` | 创建 EP / EPLB process group |
| MoE 并行配置 | `fused_moe/config.py` | `FusedMoEParallelConfig` | 决定 `use_ep`、`ep_size`、`ep_rank` |
| expert 放置 | `expert_map_manager.py` | `ExpertMapManager`、`expert_map` | 把 global experts 分配到 local rank |
| 路由 | `router/` | `FusedMoERouter`、`topk_ids`、`topk_weights` | 每个 token 选择 top-k expert |
| 权重容器 | `routed_experts.py` | `RoutedExperts` | 持有本 rank experts 并加载权重 |
| forward 编排 | `runner/moe_runner.py` | `MoERunner` | 串起 dispatch、routing、expert compute、combine |
| 通信抽象 | `modular_kernel.py` | `PrepareAndFinalize` | 把量化、dispatch、combine 抽象成可替换组件 |
| 默认通信 | `all2all.py` | `AgRsAll2AllManager` | all-gather dispatch + reduce-scatter combine |
| 高性能通信 | `prepare_finalize/*`、`all2all.py` | DeepEP / FlashInfer / MoRI / NIXL | 更专用的 all2all / batched / async EP 路径 |
| 负载均衡 | `eplb_state`、`base_router.py` | logical→physical map | 通过 redundant experts 做负载均衡 |

---

## 26. 一句话总结

vLLM 的 Expert Parallel 可以理解为 MoE 层内部的一套“expert 所有权 + token 路由 + all2all 通信”机制：

```text
FusedMoEParallelConfig 决定是否用 EP；
ExpertMapManager 决定每个 expert 在哪个 rank；
Router 决定每个 token 去哪些 experts；
Prepare/Finalize 和 all2all backend 负责把 token 发过去再收回来；
RoutedExperts / quant method 负责本地 expert 计算；
EPLB 可以在 logical expert 和 physical expert replica 之间动态重映射。
```

如果只记住一条主线，就是：

```text
router 选 expert，expert_map 定位本地 expert，dispatch 把 token 发到 expert rank，local kernel 计算，combine 把输出还给原 token rank。
```
