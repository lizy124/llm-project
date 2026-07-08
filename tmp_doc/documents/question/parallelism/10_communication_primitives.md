# 10. vLLM 并行体系用到哪些通信原语？

源码位置：

- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/device_communicators/`
- `vllm/vllm/distributed/device_communicators/base_device_communicator.py`
- `vllm/vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/vllm/distributed/device_communicators/all2all.py`
- `vllm/vllm/v1/executor/abstract.py`
- `vllm/vllm/v1/executor/multiproc_executor.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/dp_utils.py`
- `vllm/vllm/v1/worker/cp_utils.py`
- `vllm/vllm/v1/attention/ops/common.py`
- `vllm/vllm/v1/attention/ops/dcp_alltoall.py`
- `vllm/vllm/v1/attention/ops/merge_attn_states.py`
- `vllm/vllm/model_executor/layers/fused_moe/`

本问题关注：vLLM 并行体系中常见通信原语分别解决什么问题，all-reduce / all-gather / reduce-scatter / all-to-all / broadcast / send-recv / barrier / collective_rpc 在哪些并行策略中出现，以及 NCCL、custom allreduce、Ray、multiprocessing、MoE all-to-all、DCP LSE merge 分别处在哪一层。

---

## 1. 一句话回答

通信原语是并行切分后的数据对齐手段。

可以把 vLLM 里的通信分成两大类：

```text
控制面通信：
  Executor.collective_rpc()
  负责把 Python 控制命令发给 Worker，例如 init_device、load_model、execute_model。

数据面通信：
  torch / NCCL / PyNccl / custom allreduce / all-to-all / send-recv
  负责在 forward 中交换 tensor，例如 TP all-reduce、PP intermediate_tensors、MoE token dispatch、DCP LSE merge。
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
vLLM 中通信分哪几层？
communication_op.py 和 parallel_state.py 分别负责什么？
GroupCoordinator 如何封装 process group 和 device communicator？
all-reduce / all-gather / reduce-scatter / gather / broadcast 如何调用？
custom allreduce、PyNccl、torch.distributed fallback 如何选择？
PP stage 之间如何传 intermediate_tensors？
MoE EP 的 all-to-all 为什么会包装成 dispatch / combine？
DCP attention 为什么不是普通 reduce，而需要 LSE merge？
DP synchronization 为什么用小 tensor all-reduce？
Executor.collective_rpc 和 torch collective 有什么区别？
```

---

## 3. 先给结论：vLLM 通信栈分 5 层

### 3.1 Executor 控制面 RPC

这层负责：

```text
把 Python 方法名 / callable / args / kwargs 发给 Worker；
收集 Worker 返回的 Python 结果；
用于初始化、加载模型、执行一轮模型、profile、sleep、wake_up、shutdown 等控制命令。
```

入口：`vllm/v1/executor/abstract.py:152`

源码注释明确说：

```text
It is recommended to use this API to only pass control messages,
and set up data-plane communication to pass data.
```

也就是说：

```text
collective_rpc 是控制面，不是模型 forward 中的 tensor collective。
```

### 3.2 parallel_state.py 的 GroupCoordinator

`GroupCoordinator` 是 vLLM 对 process group 的核心封装。

位置：`vllm/distributed/parallel_state.py:351`

它负责：

```text
保存 group ranks；
记录 global rank / local rank / rank_in_group；
管理 cpu_group / device_group；
持有 device_communicator；
提供 all_reduce / all_gather / reduce_scatter / gather / broadcast / send / recv / tensor_dict send-recv 等方法。
```

### 3.3 communication_op.py 的 TP 快捷函数

`communication_op.py` 很薄，只是把常见 tensor parallel 通信转发到 TP group。

位置：`vllm/distributed/communication_op.py:12`

```text
tensor_model_parallel_all_reduce
  → get_tp_group().all_reduce

tensor_model_parallel_all_gather
  → get_tp_group().all_gather

tensor_model_parallel_reduce_scatter
  → get_tp_group().reduce_scatter

tensor_model_parallel_gather
  → get_tp_group().gather

broadcast_tensor_dict
  → get_tp_group().broadcast_tensor_dict
```

所以：

```text
communication_op.py 不是完整通信层；
它主要是 TP 场景的便捷 API。
```

### 3.4 device_communicator 层

`GroupCoordinator` 不直接知道 CUDA / ROCm / XPU 的所有高性能通信细节。

真正设备侧通信由：

```text
DeviceCommunicatorBase
CudaCommunicator
XpuCommunicator
CpuCommunicator
custom allreduce
PyNccl
All2AllManager
```

等实现。

Base 定义位置：`vllm/distributed/device_communicators/base_device_communicator.py:118`

CUDA 实现位置：`vllm/distributed/device_communicators/cuda_communicator.py:26`

### 3.5 专用通信语义层

除了普通 collective，vLLM 还有一些“带语义”的通信：

```text
PP intermediate_tensors send / recv：
  传的是 pipeline stage 之间的中间激活。

MoE dispatch / combine：
  传的是按 expert 路由后的 token。

DCP LSE merge：
  传的是 partial attention output + softmax LSE。

DP synchronization：
  传的是 token 数、padding、ubatch、cudagraph mode 等调度执行状态。
```

这些不能只用“某个 all-reduce”简单解释，必须结合并行策略和数据语义看。

---

## 4. 通信原语矩阵

| 通信原语 | vLLM 入口 | 典型用途 | 相关并行 |
|---|---|---|---|
| `all-reduce` | `GroupCoordinator.all_reduce()` | 聚合 partial hidden states、DP 状态同步 | TP / DP |
| `all-gather` | `GroupCoordinator.all_gather()` | 汇总 tensor 分片、DCP query / LSE | TP / CP |
| `all-gatherv` | `GroupCoordinator.all_gatherv()` | 不等长 token batch 汇总 | EP / DP-MoE |
| `reduce-scatter` | `GroupCoordinator.reduce_scatter()` | reduce 后按 rank 保持分片 | TP / CP |
| `reduce-scatterv` | `GroupCoordinator.reduce_scatterv()` | 不等长 combine | EP / DP-MoE |
| `gather` | `GroupCoordinator.gather()` | 收集到指定 rank | TP / logits 等场景 |
| `broadcast` | `broadcast_object/tensor_dict` | 同步对象、metadata、tensor dict | 初始化 / TP / PP |
| `send / recv` | `GroupCoordinator.send/recv` | 点对点 tensor 传输 | PP |
| `isend / irecv tensor_dict` | `isend_tensor_dict/irecv_tensor_dict` | pipeline intermediate_tensors | PP |
| `all-to-all` | `dist.all_to_all_single` 或 All2AllManager | token dispatch / DCP A2A | EP / CP |
| `barrier` | `GroupCoordinator.barrier()` | 阶段同步 | 初始化 / teardown |
| `collective_rpc` | `Executor.collective_rpc()` | 控制面 Worker 方法调用 | Executor / Worker |

---

## 5. GroupCoordinator 如何封装通信

### 5.1 基本状态

`GroupCoordinator` 保存这些关键状态：

```text
rank：
  当前进程 global rank。

ranks：
  当前 group 中所有 global ranks。

world_size：
  当前 group 大小。

local_rank：
  节点内设备编号，用于分配 device。

rank_in_group：
  当前进程在这个 group 内的位置。

cpu_group：
  CPU 通信用 process group。

device_group：
  GPU / XPU / 设备通信用 process group。

device_communicator：
  设备侧高性能通信实现。
```

位置：`vllm/distributed/parallel_state.py:361` 到 `vllm/distributed/parallel_state.py:377`。

### 5.2 all_reduce

入口：`vllm/distributed/parallel_state.py:622`

逻辑：

```text
如果 group world_size == 1：
  直接返回 input。

否则：
  如果 use_custom_op_call：
    torch.ops.vllm.all_reduce(input, group_name=unique_name)
  否则：
    self._all_reduce_out_place(input)
      → device_communicator.all_reduce(input)
```

这里用 `group_name` 是为了让 torch custom op 能在 Dynamo / torch.compile 下工作：

```text
custom op 不直接接收 GroupCoordinator 对象；
只传 group_name，再从注册表里找 group。
```

### 5.3 all_gather / reduce_scatter

入口：

```text
GroupCoordinator.all_gather：vllm/distributed/parallel_state.py:651
GroupCoordinator.reduce_scatter：vllm/distributed/parallel_state.py:682
```

它们和 all_reduce 类似：

```text
优先走 torch.ops.vllm 自定义 op；
否则走 device_communicator。
```

`DeviceCommunicatorBase.all_gather()` 使用 concat-style all-gather，而不是 stack-style all-gather。

位置：`vllm/distributed/device_communicators/base_device_communicator.py:184`

原因注释写明：

```text
stack-style all-gather has compatibility issues with torch.compile。
```

### 5.4 broadcast / tensor_dict broadcast

入口：

```text
broadcast：vllm/distributed/parallel_state.py:726
broadcast_object：vllm/distributed/parallel_state.py:741
broadcast_tensor_dict：vllm/distributed/parallel_state.py:845
```

`broadcast_tensor_dict()` 会把 dict 拆成：

```text
metadata_list：
  非 tensor 值，或 tensor 的 device / dtype / shape metadata。

tensor_list：
  真正需要广播的 tensor。
```

拆分函数位置：`vllm/distributed/parallel_state.py:81`。

这样可以先用 CPU group 广播 metadata，再用 CPU / device group 广播 tensor 数据。

### 5.5 send / recv / tensor_dict send-recv

入口：

```text
send_object / recv_object：
  vllm/distributed/parallel_state.py:782
  vllm/distributed/parallel_state.py:809

send_tensor_dict / recv_tensor_dict：
  vllm/distributed/parallel_state.py:941
  vllm/distributed/parallel_state.py:1036

isend_tensor_dict / irecv_tensor_dict：
  vllm/distributed/parallel_state.py:979
  vllm/distributed/parallel_state.py:1074

raw tensor send / recv：
  vllm/distributed/parallel_state.py:1169
  vllm/distributed/parallel_state.py:1176
```

`send_tensor_dict()` 有一个重要优化：

```text
all_gather_group：
  如果提供，sender 可以只发送 TP group 中自己负责的 slice；
  receiver 再通过 all-gather 重建完整 tensor。
```

源码注释明确说这个 group 通常是 tensor-parallel group，位置：`vllm/distributed/parallel_state.py:951` 到 `vllm/distributed/parallel_state.py:964`。

---

## 6. CUDA communicator 如何选择 all-reduce 后端

CUDA 通信实现入口：`vllm/distributed/device_communicators/cuda_communicator.py:26`

### 6.1 custom allreduce 只用于 TP group

初始化时判断：

```text
if "tp" not in unique_name:
  use_custom_allreduce = False
  use_torch_symm_mem = False
  use_flashinfer_allreduce = False
else:
  use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
  use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM
  use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER
```

位置：`vllm/distributed/device_communicators/cuda_communicator.py:45` 到 `vllm/distributed/device_communicators/cuda_communicator.py:56`

含义：

```text
custom allreduce / symmetric memory / flashinfer allreduce 主要服务 TP all-reduce；
其他 group 默认不启用这些 TP 专用优化。
```

### 6.2 all-reduce dispatch 顺序

`CudaCommunicator.all_reduce()` 的顺序是：

位置：`vllm/distributed/device_communicators/cuda_communicator.py:254`

```text
1. NCCL symmetric memory all-reduce
2. QuickReduce（ROCm MI3x 补充路径）
3. FlashInfer all-reduce
4. CustomAllreduce
5. Symmetric memory communicator
6. PyNccl
7. torch.distributed.all_reduce fallback
```

这说明：

```text
用户看到的 get_tp_group().all_reduce() 不是一个固定实现；
它会根据 group、平台、环境变量、world size、tensor dtype/size 动态选择后端。
```

### 6.3 reduce_scatter / reduce_scatterv

CUDA 的 reduce-scatter 走 PyNccl：

```text
reduce_scatter：vllm/distributed/device_communicators/cuda_communicator.py:313
reduce_scatterv：vllm/distributed/device_communicators/cuda_communicator.py:338
```

`reduce_scatterv` 支持每个 rank 不同大小的输出，MoE combine / DP-MoE 场景会用到这种不等长形式。

### 6.4 send / recv

CUDA communicator 的 raw tensor send / recv：

```text
send：vllm/distributed/device_communicators/cuda_communicator.py:373
recv：vllm/distributed/device_communicators/cuda_communicator.py:385
```

优先使用 PyNccl communicator，如果不可用则回退 torch.distributed send/recv。

---

## 7. TP 通信用在哪里

TP 通信的轻量入口在 `communication_op.py`：

```text
tensor_model_parallel_all_reduce()
tensor_model_parallel_all_gather()
tensor_model_parallel_reduce_scatter()
tensor_model_parallel_gather()
```

位置：`vllm/distributed/communication_op.py:12` 到 `vllm/distributed/communication_op.py:35`。

典型语义：

```text
ColumnParallelLinear：
  常见是输出分片继续往下传，不一定立即 all-reduce。

RowParallelLinear：
  每个 TP rank 计算 partial output，之后需要 all-reduce 聚合。

Vocab / logits：
  某些场景需要 gather 或 all-gather。
```

所以 TP 通信最常见的是：

```text
partial tensor
  → TP group all-reduce / all-gather / reduce-scatter
  → 对齐到后续 layer 所需的形态
```

---

## 8. PP 的 send / recv：传 intermediate_tensors

Pipeline parallel 的通信不是 all-reduce，而是 stage 间点对点传中间激活。

### 8.1 非 first PP rank 接收上游 tensor

位置：`vllm/v1/worker/gpu_worker.py:881`

```text
if forward_pass and not get_pp_group().is_first_rank:
  tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
  )
  intermediate_tensors = AsyncIntermediateTensors(...)
```

这里有几个关键点：

```text
1. PP group 负责 stage 间收发。
2. TP group 可作为 all_gather_group，用来重建 TP 分片 tensor。
3. 返回的是 AsyncIntermediateTensors，说明接收可以和后续执行做一定重叠。
```

### 8.2 非 last PP rank 发送下游 tensor

位置：`vllm/v1/worker/gpu_worker.py:917`

```text
self._pp_send_work = get_pp_group().isend_tensor_dict(
  output.tensors,
  all_gather_group=get_tp_group(),
  all_gather_tensors=all_gather_tensors,
)
```

这条路径说明：

```text
PP 通信传的是 IntermediateTensors.tensors；
不是控制面 RPC；
也不是 TP all-reduce；
而是 PP group 上的 tensor_dict isend / irecv。
```

### 8.3 sequence parallel 相关 all_gather_tensors

当：

```text
pipeline_parallel_size > 1
pass_config.enable_sp
forward_pass
```

GPU worker 会提前判断哪些 tensor 需要 all-gather，例如 residual 是否已经 scattered。

位置：`vllm/v1/worker/gpu_worker.py:852` 到 `vllm/v1/worker/gpu_worker.py:879`。

这说明 PP send/recv 会和 TP / SP tensor 布局联动。

---

## 9. EP / MoE 的 all-to-all：dispatch / combine

MoE 通信不能只说“all-to-all”，因为它有明确 token 路由语义。

### 9.1 MoE modular kernel 的通信阶段

`modular_kernel.py` 中把 fused MoE 拆成：

位置：`vllm/model_executor/layers/fused_moe/modular_kernel.py:45`

```text
[Router]
  → [Quantize-Dispatch]
  → [Permute-Experts-Unpermute]
  → [Combine]
```

其中：

```text
Quantize-Dispatch：
  负责把 token 发到持有目标 expert 的 rank。

Combine：
  负责把 expert 输出合并回 token 原路径。
```

### 9.2 All2AllManagerBase

MoE all-to-all 抽象在：`vllm/distributed/device_communicators/base_device_communicator.py:30`

关键接口：

```text
dispatch_router_logits(hidden_states, router_logits, ...)
dispatch(hidden_states, topk_weights, topk_ids, ...)
combine(hidden_states, ...)
```

GroupCoordinator 也暴露了对应方法：

```text
dispatch_router_logits：vllm/distributed/parallel_state.py:1201
dispatch：vllm/distributed/parallel_state.py:1221
combine：vllm/distributed/parallel_state.py:1243
```

这些最终转给 `device_communicator.all2all_manager`。

### 9.3 all2all backend 选择

CUDA communicator 初始化 all2all manager 的位置：`vllm/distributed/device_communicators/cuda_communicator.py:121`

支持的 backend 包括：

```text
allgather_reducescatter / naive
DeepEP high throughput
DeepEP low latency
DeepEP v2
MoRI high throughput / low latency
NIXL EP
FlashInfer NVLink two-sided
FlashInfer NVLink one-sided
```

对应配置字段是：

```text
parallel_config.all2all_backend
```

### 9.4 AgRsAll2AllManager：用 all-gather + reduce-scatter 模拟 all-to-all

`AgRsAll2AllManager` 位置：`vllm/distributed/device_communicators/all2all.py:42`

它的语义是：

```text
dispatch：
  all-gatherv hidden_states / router_logits / topk 信息。

combine：
  reduce-scatterv hidden_states。
```

对应位置：

```text
dispatch_router_logits：vllm/distributed/device_communicators/all2all.py:51
dispatch：vllm/distributed/device_communicators/all2all.py:85
combine：vllm/distributed/device_communicators/all2all.py:125
```

为什么需要 `v` 版本？

```text
不同 DP rank / EP rank 上 token 数可能不同；
MoE token routing 后每个 rank 的 token chunk 大小可能不一致；
所以需要 all_gatherv / reduce_scatterv 处理不等长 batch。
```

### 9.5 prepare_finalize 如何拿到 all2all manager

`all2all_utils.py` 中：

位置：`vllm/model_executor/layers/fused_moe/all2all_utils.py:59`

```text
_get_ep_all2all_manager()
  → get_ep_group().device_communicator.all2all_manager
```

`maybe_make_prepare_finalize()` 会根据 MoE parallel config 和 all2all backend 创建对应 prepare/finalize 对象。

位置：`vllm/model_executor/layers/fused_moe/all2all_utils.py:117`

这说明 MoE forward 不直接到处调用 dist.all_to_all，而是通过：

```text
MoE config
  → all2all manager
  → prepare/finalize
  → dispatch / combine
```

统一组织。

---

## 10. CP / DCP 通信：attention LSE merge 不是普通 reduce

Context Parallel / Decode Context Parallel 的通信语义和 TP 不一样。

TP 的 all-reduce 常常是：

```text
partial output 相加。
```

但 attention 的 partial output 不能直接相加，因为 softmax 分母被切分了。

所以 DCP / CP 需要：

```text
partial attention output
partial softmax LSE
```

再用 LSE 进行数值正确的 merge。

### 10.1 CP 兼容性检查

`cp_utils.py` 会检查：

位置：`vllm/v1/worker/cp_utils.py:14`

```text
如果 dcp_size > 1：
  attention impl 必须能返回 decode softmax LSE。

如果 pcp_size > 1：
  attention impl 必须 supports_pcp。
```

DCP 检查位置：`vllm/v1/worker/cp_utils.py:30` 到 `vllm/v1/worker/cp_utils.py:37`。

### 10.2 FlashAttention DCP 路径

FlashAttention DCP 入口：`vllm/v1/attention/backends/flash_attn.py:962`

关键通信：

```text
query_across_dcp = get_dcp_group().all_gather(query, dim=1)
```

位置：`vllm/v1/attention/backends/flash_attn.py:984`

然后 FA 返回：

```text
context_attn_out
context_lse
```

再调用：

```text
self.dcp_combine(context_attn_out, context_lse, get_dcp_group(), return_lse=True)
```

位置：`vllm/v1/attention/backends/flash_attn.py:1019`。

最后把 context attention 和 query attention 用：

```text
merge_attn_states(...)
```

合并，位置：`vllm/v1/attention/backends/flash_attn.py:1053`。

### 10.3 ag_rs 路径：all-gather LSE + reduce-scatter output

`cp_lse_ag_out_rs()` 位置：`vllm/v1/attention/ops/common.py:212`

逻辑：

```text
1. cp_group.all_gather(cp_attn_lse, dim=0)
2. correct_attn_out(...) 用 LSE 修正 partial output
3. cp_group.reduce_scatter(out, dim=1)
4. 如果 return_lse，再取本 rank 对应 head 的 LSE slice
```

这就是 DCP 默认 `ag_rs` 通信后端的语义。

### 10.4 a2a 路径：pack output + LSE 后 all-to-all

`dcp_a2a_lse_reduce()` 位置：`vllm/v1/attention/ops/dcp_alltoall.py:392`

逻辑：

```text
1. 把 partial output 和 fp32 LSE pack 到 send_buffer。
2. dist.all_to_all_single(recv_buffer, send_buffer, group=cp_group.device_group)。
3. Triton kernel unpack 并用 LSE 权重合并。
```

核心 all-to-all 调用位置：`vllm/v1/attention/ops/dcp_alltoall.py:447`。

### 10.5 prefix/suffix attention merge

`merge_attn_states()` 位置：`vllm/v1/attention/ops/merge_attn_states.py:9`

它用于把：

```text
prefix attention output + prefix LSE
suffix attention output + suffix LSE
```

合成一个 output。

源码注释说明它使用 log-sum-exp rescaling method，位置：`vllm/v1/attention/ops/merge_attn_states.py:19`。

所以：

```text
attention partial state merge 是“带 softmax 语义的 merge”，不是普通 all-reduce。
```

---

## 11. DP synchronization：小 tensor all-reduce

DP 同步不一定是在传模型 hidden states，也可能是在同步执行决策。

文件：`vllm/v1/worker/dp_utils.py`

### 11.1 同步什么

`coordinate_batch_across_dp()` 会协调：

```text
是否所有 DP rank 都 microbatch；
每个 DP rank 本轮 token 数；
是否需要 DP padding；
cudagraph mode 如何在 DP ranks 间对齐。
```

入口位置：`vllm/v1/worker/dp_utils.py:164`

### 11.2 如何同步

核心 `_run_ar()` 构造一个小 tensor：

位置：`vllm/v1/worker/dp_utils.py:36`

```text
tensor shape = [4, dp_size]

第 0 行：orig_num_tokens_per_ubatch
第 1 行：padded_num_tokens_per_ubatch
第 2 行：should_ubatch
第 3 行：cudagraph_mode
```

然后：

```text
dist.all_reduce(tensor, group=dp_group)
```

位置：`vllm/v1/worker/dp_utils.py:53`。

这不是 TP 那种 hidden states all-reduce，而是：

```text
DP ranks 之间同步 batch execution metadata。
```

### 11.3 CPU / GPU group 选择

`_get_device_and_group()` 会默认使用 DP group 的 device 和 device_group。

但如果：

```text
parallel_config.disable_nccl_for_dp_synchronization
```

则改用：

```text
CPU tensor + DP cpu_group
```

位置：`vllm/v1/worker/dp_utils.py:18` 到 `vllm/v1/worker/dp_utils.py:33`。

原因注释说明：

```text
GPU → CPU 同步点可能影响 async scheduling；
这个开关用于快速禁用 NCCL DP sync。
```

---

## 12. Executor collective_rpc：控制面，不是 tensor collective

### 12.1 抽象定义

`Executor.collective_rpc()` 定义在：`vllm/v1/executor/abstract.py:152`

它支持：

```text
method 是字符串：
  表示调用 Worker 上同名方法。

method 是 callable：
  callable 会序列化后发给 Worker 执行。

non_block=True：
  返回 Future。
```

抽象注释说返回每个 worker 的结果列表。

### 12.2 Multiproc 实现

Multiproc 实现在：`vllm/v1/executor/multiproc_executor.py:340`

核心逻辑：

```text
1. 把 method / args / kwargs / output_rank 放进 rpc_broadcast_mq。
2. Worker 进程从 message queue 收到任务并执行。
3. Executor 从 response_mqs 收集结果。
4. 如果 unique_reply_rank 指定，只取一个 rank 的结果。
5. 如果 kv_output_aggregator 指定，对 KV transfer 输出做聚合。
```

发送位置：`vllm/v1/executor/multiproc_executor.py:374`

收结果位置：`vllm/v1/executor/multiproc_executor.py:380` 到 `vllm/v1/executor/multiproc_executor.py:396`。

### 12.3 和 torch collective 的区别

```text
collective_rpc：
  控制面；
  传 Python 方法 / 参数 / 返回值；
  用于让 Worker 执行某个动作；
  backend 可以是 multiprocessing message queue、Ray actor RPC、单进程直接调用。

torch collective / GroupCoordinator：
  数据面；
  传 tensor；
  用于模型 forward 中的分布式计算；
  backend 是 NCCL / Gloo / PyNccl / custom communicator 等。
```

典型对比：

```text
Executor.execute_model()
  → collective_rpc("execute_model", args=(scheduler_output,))
  → 控制 Worker 开始执行。

RowParallelLinear forward
  → get_tp_group().all_reduce(hidden_states)
  → TP ranks 间聚合 tensor。
```

---

## 13. barrier 和生命周期同步

`GroupCoordinator.barrier()` 位置：`vllm/distributed/parallel_state.py:1160`

注意它使用的是：

```text
cpu_group
```

源码注释解释了原因：

```text
不要用 NCCL device_group 做 barrier；
NCCL barrier 内部是 broadcast GPU tensor，容易弄乱当前 device。
```

所以：

```text
barrier 更偏生命周期 / 阶段同步，
不是 forward 热路径中的核心算子。
```

---

## 14. 按并行策略总结通信

### 14.1 TP

```text
主要 group：
  TP group

主要通信：
  all-reduce
  all-gather
  reduce-scatter
  gather

典型用途：
  linear partial output 聚合；
  tensor shard 汇总；
  logits / vocab 分片处理。
```

### 14.2 PP

```text
主要 group：
  PP group

主要通信：
  isend_tensor_dict
  irecv_tensor_dict
  send / recv

典型用途：
  pipeline stage 之间传 intermediate_tensors。
```

### 14.3 DP

```text
主要 group：
  DP group

主要通信：
  DP sync 小 tensor all-reduce；
  MoE + EP 场景下可能参与 expert sharding / token dispatch。

相关控制面：
  Executor.collective_rpc 会把 execute_model 分发给本 replica 内 worker，但它不是 DP group 的 tensor collective。

典型用途：
  多 replica 执行协调；
  DP padding / ubatch / cudagraph mode 对齐。
```

### 14.4 EP

```text
主要 group：
  EP group

主要通信：
  dispatch_router_logits
  dispatch
  combine
  all-to-all 或 allgather + reducescatter

典型用途：
  MoE token routing；
  expert output combine。
```

### 14.5 CP / DCP / PCP

```text
主要 group：
  DCP group
  PCP group

主要通信：
  query all-gather；
  LSE all-gather；
  output reduce-scatter；
  a2a pack/unpack combine；
  merge_attn_states。

典型用途：
  长上下文 attention partial state 合并。
```

---

## 15. 源码阅读路线

如果要系统读源码，建议按这个顺序：

```text
1. vllm/distributed/parallel_state.py
   先看 GroupCoordinator，以及 all_reduce / all_gather / send_tensor_dict 等统一接口。

2. vllm/distributed/communication_op.py
   看 TP 快捷通信 API。

3. vllm/distributed/device_communicators/base_device_communicator.py
   看设备通信器抽象，以及 all2all manager 抽象。

4. vllm/distributed/device_communicators/cuda_communicator.py
   看 CUDA all-reduce 后端选择、send/recv、all2all manager 初始化。

5. vllm/v1/worker/gpu_worker.py
   看 PP intermediate_tensors 的 irecv / isend。

6. vllm/distributed/device_communicators/all2all.py
   看 MoE allgather + reducescatter 和 DeepEP / FlashInfer 等 all2all manager。

7. vllm/model_executor/layers/fused_moe/modular_kernel.py
   看 MoE forward 为什么拆成 Router / Dispatch / Experts / Combine。

8. vllm/model_executor/layers/fused_moe/all2all_utils.py
   看 MoE prepare/finalize 如何绑定 all2all manager。

9. vllm/v1/attention/ops/common.py
   看 CP 默认 ag_rs LSE merge。

10. vllm/v1/attention/ops/dcp_alltoall.py
    看 DCP a2a LSE reduce。

11. vllm/v1/worker/dp_utils.py
    看 DP 同步 batch execution metadata。

12. vllm/v1/executor/abstract.py 和 multiproc_executor.py
    看 collective_rpc 控制面。
```

---

## 16. 本篇结论

```text
1. vLLM 的通信不是单一 torch.distributed 调用，而是一套分层通信栈：
   Executor 控制面 RPC、GroupCoordinator、device communicator、专用 MoE / CP / PP 语义通信。

2. communication_op.py 主要是 TP 快捷封装；真正的 group 管理和通信分发在 parallel_state.py 的 GroupCoordinator。

3. CUDA all-reduce 会按 NCCL symmetric memory、QuickReduce、FlashInfer、CustomAllreduce、SymmMem、PyNccl、torch fallback 的顺序动态选择。

4. PP 通信是 stage 间 tensor_dict send/recv，传的是 intermediate_tensors，并且会和 TP all-gather 优化联动。

5. MoE EP 通信不是普通 tensor all-reduce，而是 router → dispatch → expert compute → combine 的 token-level all-to-all 语义。

6. DCP / CP attention 通信不能简单相加 partial output，必须携带 softmax LSE，并通过 LSE merge 保证数值正确。

7. DP 通信既有控制面请求分发，也有小 tensor all-reduce 同步 padding、ubatch、cudagraph mode 等执行状态。

8. collective_rpc 是控制面，torch collective / GroupCoordinator 是数据面；混淆这两层会导致对 vLLM 并行执行链路的理解错误。
```
