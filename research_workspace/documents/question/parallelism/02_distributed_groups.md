# 02. vLLM 如何创建和管理并行 group？

源码位置：

- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/distributed/communication_op.py`
- `vllm/vllm/distributed/device_communicators/`
- `vllm/vllm/distributed/stateless_coordinator.py`
- `vllm/vllm/config/parallel.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/dp_utils.py`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/v1/attention/backend.py`

本问题关注：vLLM 如何把 global world 拆成 tensor parallel group、pipeline parallel group、data parallel group、expert parallel group、context parallel group 等；这些 group 如何被保存、查询和销毁；forward 中不同模块如何拿到对应 group 执行通信。

---

## 1. 一句话回答

`parallel_state.py` 是 vLLM 分布式并行 group 的中心状态表。

它把一个全局 distributed world 按配置切成多种正交维度：

```text
global world
  → init_distributed_environment()
  → initialize_model_parallel()
  → 创建 TP / DCP / PCP / PP / DP / EP / EPLB groups
  → get_tp_group() / get_pp_group() / get_dp_group() / ...
  → 模型层、attention、MoE、PP send/recv、DP 同步调用对应 group 通信
```

如果只记一句话：

```text
vLLM 的并行策略不是散落在各个 layer 里，而是先在 parallel_state.py 中创建全局 group，再由各模块按需查询 group 并执行通信。
```

---

## 2. 本文要回答的问题

```text
vLLM 维护哪些 process group？
每类 group 的 rank 成员如何计算？
TP / PP / DP / EP / DCP / PCP group 分别给谁使用，sequence parallel 为什么复用 TP group？
get_tp_group() 这类 helper 返回什么？
GroupCoordinator 封装了哪些通信能力？
CPU group / device group / device communicator 有什么区别？
分布式环境和 model parallel group 的生命周期如何初始化和销毁？
current rank / local rank / rank_in_group 容易混淆在哪里？
```

---

## 3. 先给结论：vLLM 维护哪些 group

当前 `parallel_state.py` 中主要有这些全局 group 状态：

```text
_WORLD：
  全局 distributed world。

_INNER_DP_WORLD：
  多节点 DP 场景下，DP 内部的 TP/PP worker world。

_TP：
  tensor parallel group，同一层内切分矩阵 / hidden / heads 后协作计算。

_DCP：
  decode context parallel group，decode attention 的上下文并行 group。

_PCP：
  prefill context parallel group，prefill attention 的上下文并行 group。

_PP：
  pipeline parallel group，串联不同 pipeline stage。

_DP：
  data parallel group，多个 replica 之间同步执行状态或 MoE 相关并行。

_EP：
  expert parallel group，MoE expert dispatch / combine 使用。

_EPLB：
  expert parallel load balancing group，EPLB 单独使用，避免和 MoE forward 通信互相阻塞。
```

对应 getter：

```text
get_world_group()
get_inner_dp_world_group()
get_tp_group()
get_dcp_group()
get_pcp_group()
get_pp_group()
get_dp_group()
get_ep_group()
get_eplb_group()
```

位置：`get_world_group()` 在 `vllm/vllm/distributed/parallel_state.py:1276`，`get_inner_dp_world_group()` 在 `vllm/vllm/distributed/parallel_state.py:1281`，`get_tp_group()` / `get_dcp_group()` / `get_pp_group()` 等 getter 从 `vllm/vllm/distributed/parallel_state.py:1368` 开始。

---

## 4. 主链路总览

### 4.1 Worker 初始化时创建 group

V1 GPU worker 初始化分布式环境的入口在：

`vllm/vllm/v1/worker/gpu_worker.py:1326`

核心调用顺序是：

```text
init_worker_distributed_environment(...)
  → set_custom_all_reduce(...)
  → init_distributed_environment(...)
  → ensure_model_parallel_initialized(...)
      → initialize_model_parallel(...)
```

其中：

```text
init_distributed_environment：
  初始化 torch.distributed 默认进程组和 _WORLD。

ensure_model_parallel_initialized：
  如果 TP / PP 等 group 尚未创建，则调用 initialize_model_parallel；
  如果已经创建，则校验现有 group size 是否符合当前配置。

initialize_model_parallel：
  真正根据 rank layout 创建 TP / DCP / PCP / PP / DP / EP / EPLB group。
```

### 4.2 初始化后，业务模块只拿 group 通信

典型使用方式是：

```text
linear / tensor-parallel layer：
  get_tp_group().all_reduce / all_gather / reduce_scatter

pipeline stage：
  get_pp_group().isend_tensor_dict / irecv_tensor_dict

MoE expert parallel：
  get_ep_group().dispatch / combine

attention context parallel：
  get_dcp_group() / get_pcp_group()

DP 状态同步：
  get_dp_group() 或 parallel_config.stateless_init_dp_group()
```

也就是说：

```text
模型层不重新计算 rank 成员，rank 成员只在 parallel_state.py 初始化阶段统一计算。
```

---

## 5. GroupCoordinator 是什么

`GroupCoordinator` 是 vLLM 对 PyTorch `ProcessGroup` 的包装。

位置：`vllm/vllm/distributed/parallel_state.py:358`

它不是单纯保存一个 `ProcessGroup`，而是同时管理：

```text
rank：当前进程的 global rank；
ranks：当前 group 内所有 global ranks；
world_size：当前 group 大小；
local_rank：当前节点内 local rank，主要用于选择 device；
rank_in_group：当前进程在本 group 内的下标；
cpu_group：CPU / metadata / object 通信 group；
device_group：GPU / XPU 等 device tensor 通信 group；
device_communicator：平台相关高性能通信封装；
mq_broadcaster：可选 message queue broadcaster。
```

### 5.1 rank / local_rank / rank_in_group 的区别

源码注释里给了一个例子：

```text
Process | Node | Global Rank | Local Rank | Rank in Group
   0    |  0   |      0      |     0      |       0
   1    |  0   |      1      |     1      |       1
   2    |  1   |      2      |     0      |       2
   3    |  1   |      3      |     1      |       3
```

位置：`vllm/vllm/distributed/parallel_state.py:374`

区别是：

```text
rank：全局 rank，用于 distributed world。
local_rank：节点内 rank，用于绑定 cuda:xpu device。
rank_in_group：当前 group 内的 rank，用于 group 内 src / dst / first / last 判断。
```

### 5.2 GroupCoordinator 为什么要有 CPU group 和 device group

`GroupCoordinator` 会为每个 group 创建两类通信组：

```text
device_group：
  用于 tensor all_reduce / all_gather / reduce_scatter / send / recv，通常走 nccl 等 device backend。

cpu_group：
  用于 object / metadata / barrier / tensor_dict metadata，通常走 gloo。
```

位置：`vllm/vllm/distributed/parallel_state.py:414`

这点很重要，因为：

```text
NCCL barrier 可能隐式创建 GPU tensor，容易干扰当前 device；
vLLM 的 GroupCoordinator.barrier() 明确使用 CPU group。
```

位置：`vllm/vllm/distributed/parallel_state.py:1179`

### 5.3 device communicator 是什么

`device_communicator` 是平台相关通信实现。

位置：`vllm/vllm/distributed/parallel_state.py:484`

创建时会根据当前平台选择：

```text
current_platform.get_device_communicator_cls()
```

它负责更高层的 device 通信能力，例如：

```text
all_reduce
all_gather
reduce_scatter
gather
send / recv
MoE dispatch / combine
prepare_communication_buffer_for_model
```

基础接口见：`vllm/vllm/distributed/device_communicators/base_device_communicator.py:127`

可以把三者关系理解为：

```text
GroupCoordinator：vLLM 统一通信门面；
ProcessGroup：PyTorch 分布式通信对象；
DeviceCommunicator：平台 / backend 特化的高性能实现。
```

---

## 6. 分布式环境初始化：先有 WORLD

`init_distributed_environment()` 是第一层入口。

位置：`vllm/vllm/distributed/parallel_state.py:1536`

它负责：

```text
1. 根据 ParallelConfig 调整 world_size / rank / init_method；
2. 调 torch.distributed.init_process_group()；
3. 创建 _WORLD GroupCoordinator；
4. 探测节点数 _NODE_COUNT；
5. 在多节点 DP 内部场景下创建 _INNER_DP_WORLD。
```

### 6.1 DP 会影响 global world size

如果使用 data parallel，且不是 external launcher，vLLM 会把 rank 和 world size 扩展到 DP 维度：

```python
rank = parallel_config.data_parallel_rank * world_size + rank
world_size = parallel_config.world_size_across_dp
```

位置：`vllm/vllm/distributed/parallel_state.py:1565`

`world_size_across_dp` 定义在：

`vllm/vllm/config/parallel.py:516`

它等于：

```text
parallel_config.world_size * data_parallel_size
```

其中 `parallel_config.world_size` 主要是 TP × PP × PCP 这类 worker 并行大小。

### 6.2 split_group 路径

如果启用 `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`，默认 process group 初始化会使用混合 backend：

```text
cpu:gloo,cuda:nccl
```

并绑定 `device_id`，便于后续用 `torch.distributed.split_group()` 切 subgroup。

位置：`vllm/vllm/distributed/parallel_state.py:1440`

否则走传统的：

```text
torch.distributed.new_group(ranks, backend=...)
```

### 6.3 WORLD group 不启用 device communicator

`init_world_group()` 创建 `_WORLD` 时：

```python
use_device_communicator=False
```

位置：`vllm/vllm/distributed/parallel_state.py:1267`

原因是 `_WORLD` 更像全局 rank / node / lifecycle 的基础状态，不是每个 forward 高频通信的 group。

---

## 7. initialize_model_parallel 如何计算 rank layout

`initialize_model_parallel()` 是创建 TP / DCP / PCP / PP / DP / EP 的核心函数。

位置：`vllm/vllm/distributed/parallel_state.py:1694`

### 7.1 总 rank layout

源码注释写得很关键：

```text
the layout order is: ExternalDP x DP x PP x TP
```

当前代码实际 reshape 时还插入了 PCP 维度：

```python
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

位置：`vllm/vllm/distributed/parallel_state.py:1769`

可以理解为：

```text
all_ranks[external_dp, dp, pp, pcp, tp]
```

其中：

```text
ExternalDP：外部集成场景下的 DP 维度，例如 verl integration；
DP：vLLM 内部耦合 data parallel；
PP：pipeline stage；
PCP：prefill context parallel；
TP：tensor parallel。
```

创建某个 group 的通用技巧是：

```text
把目标维度 transpose 到最后一维
  → reshape(-1, target_size)
  → 每一行就是一个 group 的 ranks
```

### 7.2 TP group

TP group 创建方式：

```python
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
```

位置：`vllm/vllm/distributed/parallel_state.py:1777`

含义：最后一维本来就是 TP，所以直接按连续 `tensor_model_parallel_size` 切 group。

例子：

```text
world_size=8, TP=2, PP=4, DP=1, PCP=1
TP groups:
  [0, 1], [2, 3], [4, 5], [6, 7]
```

TP group 主要用于：

```text
- ColumnParallelLinear / RowParallelLinear 的 all_reduce / all_gather / reduce_scatter；
- tensor parallel 权重切分和输出聚合；
- sequence parallel 相关通信；
- PP 发送 tensor_dict 时的 all_gather 优化。
```

### 7.3 DCP group

DCP 是 decode context parallel。

创建方式：

```python
group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
```

位置：`vllm/vllm/distributed/parallel_state.py:1794`

DCP 的关键限制在 `ParallelConfig`：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

位置：`vllm/vllm/config/parallel.py:498`

原因是：

```text
DCP 不额外扩大 world size，而是复用 TP 相关 rank，
按 decode_context_parallel_size 在现有 rank mesh 上组织 DCP group。
```

DCP 主要给 decode attention 使用，例如 attention backend 中需要：

```text
get_dcp_group().world_size
get_dcp_group().rank_in_group
```

来决定当前 rank 负责哪部分 context，以及如何 all-gather / reduce-scatter attention 输出。

### 7.4 PCP group

PCP 是 prefill context parallel。

创建方式：

```python
group_ranks = (
    all_ranks.transpose(3, 4)
    .reshape(-1, prefill_context_model_parallel_size)
    .unbind(0)
)
```

位置：`vllm/vllm/distributed/parallel_state.py:1816`

它把 PCP 维度换到最后，再按 `prefill_context_model_parallel_size` 切 group。

PCP 主要服务 prefill 阶段的 context parallel / attention 切分。

`ParallelConfig.cp_kv_cache_interleave_size` 同时用于 DCP / PCP 的 KV cache token 分布说明：

位置：`vllm/vllm/config/parallel.py:359`

可以理解为：

```text
DCP 偏 decode 阶段；
PCP 偏 prefill 阶段；
它们共同组成 total context parallel world。
```

### 7.5 PP group

PP group 创建方式：

```python
group_ranks = (
    all_ranks.transpose(2, 4)
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
```

位置：`vllm/vllm/distributed/parallel_state.py:1835`

例子：

```text
world_size=8, TP=2, PP=4, DP=1, PCP=1
PP groups:
  [0, 2, 4, 6]
  [1, 3, 5, 7]
```

这和函数注释中的 Megatron 风格例子一致：

```text
TP groups: [g0,g1], [g2,g3], [g4,g5], [g6,g7]
PP groups: [g0,g2,g4,g6], [g1,g3,g5,g7]
```

位置：`vllm/vllm/distributed/parallel_state.py:1711`

PP group 主要用于：

```text
- 判断当前 rank 是否 first / last pipeline stage；
- 在相邻 stage 之间 send / recv IntermediateTensors；
- 多模态输入只在首 stage 保留，后续 stage 接 intermediate tensors；
- last stage 才执行 logits / sampling。
```

典型使用在 `gpu_worker.py`：

```text
非 first PP rank：get_pp_group().irecv_tensor_dict(...)
非 last PP rank：get_pp_group().isend_tensor_dict(...)
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:881`

### 7.6 DP group

DP group 创建方式：

```python
group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
```

位置：`vllm/vllm/distributed/parallel_state.py:1853`

它把 DP 维度换到最后，再按 `data_parallel_size` 切 group。

DP group 主要用于：

```text
- 多个 replica 同步是否还有未完成请求；
- pause / shutdown / progress 状态同步；
- MoE expert parallel 与 DP 结合；
- 在某些场景下做 stateless DP group 初始化。
```

`ParallelConfig` 中有一些 DP 状态同步 helper，例如：

```text
has_unfinished_dp()
sync_dp_state()
```

位置：`vllm/vllm/config/parallel.py:688`

### 7.7 EP group

EP group 是 expert parallel group，只在 MoE 模型需要时创建。

创建条件：

```python
if config.model_config is None or config.model_config.is_moe:
```

位置：`vllm/vllm/distributed/parallel_state.py:1870`

rank 组合方式：

```python
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(
        -1,
        data_parallel_size
        * prefill_context_model_parallel_size
        * tensor_model_parallel_size,
    )
    .unbind(0)
)
```

位置：`vllm/vllm/distributed/parallel_state.py:1874`

EP group 的大小大致是：

```text
DP × PCP × TP
```

它用于 MoE token dispatch / combine：

```text
get_ep_group().dispatch_router_logits(...)
get_ep_group().dispatch(...)
get_ep_group().combine(...)
```

这些方法最终会委托给 `device_communicator`，由具体 all2all backend 实现。

### 7.8 EPLB group

如果启用 expert parallel load balancing，会创建 `_EPLB`。

位置：`vllm/vllm/distributed/parallel_state.py:1898`

它使用和 EP 相同的 ranks，但单独创建 process group。

原因是：

```text
EPLB 需要和 MoE forward 通信隔离，避免在同一个 communicator 上发生死锁或互相阻塞。
```

---

## 8. 一个 8 GPU 例子

假设：

```text
world_size = 8
TP = 2
PP = 4
DP = 1
PCP = 1
DCP = 1
```

则：

```text
all_ranks reshape 后可以粗略看成：
PP0: TP [0, 1]
PP1: TP [2, 3]
PP2: TP [4, 5]
PP3: TP [6, 7]
```

TP groups：

```text
[0, 1]
[2, 3]
[4, 5]
[6, 7]
```

PP groups：

```text
[0, 2, 4, 6]
[1, 3, 5, 7]
```

对 rank 3 来说：

```text
TP group = [2, 3], rank_in_group = 1
PP group = [1, 3, 5, 7], rank_in_group = 1
```

这意味着：

```text
rank 3 既是某个 TP group 的第 2 个 tensor shard，
也是某条 pipeline lane 的第 2 个 pipeline stage。
```

---

## 9. GroupCoordinator 提供哪些通信能力

### 9.1 collective 通信

常见 collective：

```text
all_reduce(input_)
all_gather(input_, dim)
all_gatherv(input_, dim, sizes)
reduce_scatter(input_, dim)
reduce_scatterv(input_, dim, sizes)
gather(input_, dst, dim)
broadcast(input_, src)
barrier()
```

位置：`vllm/vllm/distributed/parallel_state.py:622`

这些方法通常是：

```text
GroupCoordinator 方法
  → device_communicator 方法
  → torch.distributed / custom communicator / platform backend
```

### 9.2 object / tensor_dict 通信

PP 和 worker 间传递中间张量时，常用 tensor dict：

```text
broadcast_tensor_dict()
send_tensor_dict()
recv_tensor_dict()
isend_tensor_dict()
irecv_tensor_dict()
```

位置：`vllm/vllm/distributed/parallel_state.py:845`

这类通信会先把 dict 拆成：

```text
metadata_list：key、dtype、shape、device 等元数据；
tensor_list：真正的 tensor。
```

metadata 走 CPU group，tensor 按 CPU / device 类型分别走 CPU group 或 device group。

### 9.3 P2P send / recv

GroupCoordinator 也封装了：

```text
send(tensor, dst)
recv(size, dtype, src)
```

位置：`vllm/vllm/distributed/parallel_state.py:1169`

其中 `dst` / `src` 是 group 内 rank，不是 global rank。

### 9.4 MoE dispatch / combine

MoE 特化通信接口：

```text
dispatch_router_logits()
dispatch()
combine()
```

位置：`vllm/vllm/distributed/parallel_state.py:1201`

如果 `device_communicator` 支持 MoE all2all，就走具体 backend；否则退化为原样返回。

---

## 10. communication_op.py 和 group 的关系

`communication_op.py` 是 TP 通信的薄封装。

位置：`vllm/vllm/distributed/communication_op.py:1`

它提供：

```python
tensor_model_parallel_all_reduce(input_)
tensor_model_parallel_all_gather(input_, dim)
tensor_model_parallel_reduce_scatter(input_, dim)
tensor_model_parallel_gather(input_, dst, dim)
broadcast_tensor_dict(tensor_dict, src)
```

内部都是调用：

```text
get_tp_group().xxx(...)
```

所以模型层里看到的 `tensor_model_parallel_all_reduce()`，本质上就是：

```text
在当前 rank 所属 TP group 上做 all_reduce。
```

sequence parallel 也不单独创建新的 process group；在 PP residual 等路径中常复用 TP group 做 `all_gather`，在 MoE all2all 路径中则通过 `is_sequence_parallel` 参数改变 dispatch / combine 使用的通信语义。

---

## 11. custom op collective 为什么要用 group name

`GroupCoordinator.all_reduce()` 中有一个特殊路径：

```python
torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
```

位置：`vllm/vllm/distributed/parallel_state.py:622`

原因是 Dynamo / torch custom op 不适合直接把 Python 对象 `self` 传进去。

所以 vLLM 做了一个全局注册表：

```text
_groups: unique_name → weakref(GroupCoordinator)
```

位置：`vllm/vllm/distributed/parallel_state.py:123`

custom op 收到 `group_name` 后，再查回对应 GroupCoordinator。

这条链路是：

```text
GroupCoordinator.all_reduce()
  → torch.ops.vllm.all_reduce(tensor, group_name="tp:0")
  → parallel_state.all_reduce(tensor, group_name)
  → _groups[group_name]
  → group._all_reduce_out_place(tensor)
  → device_communicator.all_reduce(tensor)
```

---

## 12. Pipeline parallel 如何使用 PP group

PP group 最直观的使用在 `GPUWorker.execute_model()` 周边。

非首个 pipeline stage 会先收上一个 stage 的 intermediate tensors：

```python
if forward_pass and not get_pp_group().is_first_rank:
    tensor_dict, comm_handles, comm_postprocess = (
        get_pp_group().irecv_tensor_dict(...)
    )
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:881`

非最后一个 stage 在 forward 后发送给下一个 stage：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(...)
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:914`

这里还会传：

```text
all_gather_group=get_tp_group()
```

原因是 PP tensor dict 中有些 tensor 在 TP group 内可做 all-gather 优化，接收方可以重构完整 tensor。

所以 PP 和 TP 在执行链路里并不是完全隔离的：

```text
PP 负责 stage 间传递；
TP 负责 stage 内 shard 的聚合 / 切分。
```

---

## 13. Tensor parallel 如何使用 TP group

TP group 是模型层最常用的 group。

典型模式是：

```text
ColumnParallelLinear：
  每个 TP rank 计算一部分输出 hidden / heads，必要时 all_gather。

RowParallelLinear：
  每个 TP rank 计算一部分输入投影，最后 all_reduce 聚合输出。

sequence parallel：
  可能用 reduce_scatter / all_gather 在序列维度切分和恢复。
```

这些操作最终会通过：

```text
get_tp_group().all_reduce(...)
get_tp_group().all_gather(...)
get_tp_group().reduce_scatter(...)
```

或 `communication_op.py` 的封装进入 `GroupCoordinator`。

辅助函数：

```text
get_tensor_model_parallel_world_size()
get_tensor_model_parallel_rank()
```

位置：`vllm/vllm/distributed/parallel_state.py:2012`

它们只是对 `get_tp_group().world_size` 和 `get_tp_group().rank_in_group` 的包装。

---

## 14. Context parallel 如何使用 DCP / PCP group

vLLM 当前把 context parallel 拆成两类：

```text
DCP：decode context parallel；
PCP：prefill context parallel。
```

### 14.1 DCP

DCP 不扩大 world size，而是复用 TP 内 rank。

配置校验：

```text
TP size 必须能整除 DCP size。
```

位置：`vllm/vllm/config/parallel.py:498`

attention backend 会读取：

```text
get_dcp_group().world_size
get_dcp_group().rank_in_group
```

用来决定本 rank 的 context shard、KV cache 本地可见范围，以及最终输出合并。

### 14.2 PCP

PCP 用于 prefill context parallel。

它和 DCP 一起影响 KV cache token 分布，`cp_kv_cache_interleave_size` 的注释说明了 total CP rank：

```text
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

位置：`vllm/vllm/config/parallel.py:360`

也就是说：

```text
DCP / PCP 不是简单通信组名字，
它们会影响 attention metadata、KV cache interleave 和 kernel 行为。
```

---

## 15. Expert parallel 如何使用 EP / EPLB group

### 15.1 EP group 服务 MoE forward

EP group 主要服务 MoE：

```text
router logits dispatch
expert token dispatch
expert output combine
```

接口在 `GroupCoordinator`：

```text
dispatch_router_logits()
dispatch()
combine()
```

位置：`vllm/vllm/distributed/parallel_state.py:1201`

这些方法会进入 device communicator 的 all2all manager。

`All2AllManagerBase` 注释说明：

```text
all2all lives in ep group, which is merged from dp and tp group
```

位置：`vllm/vllm/distributed/device_communicators/base_device_communicator.py:45`

### 15.2 EP 和 DP / TP 的关系

`ParallelConfig.data_parallel_size` 注释提到：

```text
MoE layers will be sharded according to the product of tensor parallel size and data parallel size.
```

位置：`vllm/vllm/config/parallel.py:126`

所以 EP 往往不是单独一维，而是和 DP / PCP / TP 等已有维度组合起来决定 expert 分布；具体以 `initialize_model_parallel()` 构造出的 EP group 为准。

### 15.3 EPLB 为什么单独建 group

EPLB group 与 EP group ranks 相同，但 process group 独立。

原因是：

```text
EPLB 通信可能和 MoE forward 通信同时存在；
单独 group 可以隔离 communicator，降低死锁风险。
```

位置：`vllm/vllm/distributed/parallel_state.py:1898`

---

## 16. Stateless group 和 elastic EP

正常情况下，group 是 PyTorch `ProcessGroup` 包装出来的 `GroupCoordinator`。

但当启用：

```text
parallel_config.enable_elastic_ep
```

某些 group 会走 `StatelessGroupCoordinator`。

相关入口：

```text
_init_elastic_ep_world()
_init_stateless_group()
```

位置：`vllm/vllm/distributed/parallel_state.py:1501`

主要差异：

```text
普通 GroupCoordinator：
  基于 torch.distributed 默认 world 和 new_group / split_group。

StatelessGroupCoordinator：
  通过 TCPStore 等方式协调，适合 elastic EP 中 DP / EP 动态或跨进程场景。
```

对使用方来说，目标仍然是类似的：

```text
get_dp_group()
get_ep_group()
```

只是底层 coordinator 类型不同。

---

## 17. group 生命周期

### 17.1 初始化顺序

正常顺序：

```text
init_distributed_environment()
  → 创建 torch distributed 默认 PG
  → 创建 _WORLD
  → 可选 _INNER_DP_WORLD

ensure_model_parallel_initialized()
  → 如果未初始化，调用 initialize_model_parallel()
  → 创建 _TP / _DCP / _PCP / _PP / _DP / _EP / _EPLB
```

### 17.2 校验已有 group

如果 group 已经存在，`ensure_model_parallel_initialized()` 不会重复创建，而是校验：

```text
TP world size 是否匹配；
PP world size 是否匹配；
PCP world size 是否匹配。
```

位置：`vllm/vllm/distributed/parallel_state.py:1938`

### 17.3 销毁 model parallel groups

`destroy_model_parallel()` 会依次销毁并置空：

```text
_TP
_DCP
_PCP
_PP
_DP
_EP
_EPLB
```

位置：`vllm/vllm/distributed/parallel_state.py:2028`

### 17.4 销毁 distributed environment

`destroy_distributed_environment()` 会销毁 `_WORLD` 和默认 process group：

```text
_WORLD.destroy()
torch.distributed.destroy_process_group()
```

位置：`vllm/vllm/distributed/parallel_state.py:2067`

### 17.5 完整 cleanup

`cleanup_dist_env_and_memory()` 会：

```text
1. disable env cache；
2. destroy_model_parallel；
3. destroy_distributed_environment；
4. 可选 ray.shutdown；
5. gc.collect；
6. accelerator empty_cache。
```

位置：`vllm/vllm/distributed/parallel_state.py:2077`

---

## 18. 和 CUDA graph capture 的关系

`parallel_state.graph_capture()` 会同时进入 TP 和 PP group 的 graph capture 上下文：

```python
with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
    yield context
```

位置：`vllm/vllm/distributed/parallel_state.py:1410`

GroupCoordinator 的 `graph_capture()` 会让 device communicator 做 capture 准备，例如 custom all-reduce / AITER all-reduce。

位置：`vllm/vllm/distributed/parallel_state.py:578`

也就是说：

```text
CUDA graph capture 不只捕模型 kernel，也要让通信 communicator 进入可 capture 的状态。
```

---

## 19. 容易混淆的点

### 19.1 world group 和 TP / PP group 是一回事吗？

不是。

```text
_WORLD：全局 rank 空间。
_TP / _PP / _DP / ...：从 world 中切出来的子 group。
```

### 19.2 global rank 和 group 内 rank 是一回事吗？

不是。

```text
global rank：torch.distributed world 中的 rank；
rank_in_group：当前 group 内的下标，send/recv 的 src/dst 通常用这个。
```

### 19.3 DP group 一定等于多个独立服务副本吗？

不一定。

vLLM 区分：

```text
ExternalDP：外部系统独立调度 / 独立 generate 的 DP；
DP：vLLM 内部耦合 DP，同一 DP group 内需要同时执行某些调用，否则可能 deadlock。
```

这也是 `initialize_model_parallel()` 注释强调 ExternalDP 和 DP 的原因。

### 19.4 EP group 总是存在吗？

不是。

Dense 模型通常不创建 `_EP`，调用 `get_ep_group()` 会 assert。

位置：`vllm/vllm/distributed/parallel_state.py:1378`

### 19.5 DCP 会增加总 GPU 数吗？

不会。

DCP 复用 TP group 内 GPU，因此要求：

```text
TP size % DCP size == 0
```

### 19.6 CPU group 是不是只给 CPU 推理用？

不是。

GPU 推理也会创建 CPU group，用于 metadata、object、barrier 等控制面通信。

---

## 20. 最终可以记成一张表

| group | getter | rank 维度 | 主要用途 |
|---|---|---|---|
| world | `get_world_group()` | 全部 global ranks | 全局 rank / local rank / node count / lifecycle |
| inner dp world | `get_inner_dp_world_group()` | DP 内部 worker ranks | 多节点 DP 内部 world |
| TP | `get_tp_group()` | TP 维度 | tensor parallel layer 通信、SP、TP all-reduce / all-gather |
| DCP | `get_dcp_group()` | TP 内复用切分 | decode context parallel attention |
| PCP | `get_pcp_group()` | PCP 维度 | prefill context parallel attention |
| PP | `get_pp_group()` | PP 维度 | pipeline stage 间 send / recv intermediate tensors |
| DP | `get_dp_group()` | DP 维度 | DP 状态同步、耦合 DP 执行、MoE 组合维度 |
| EP | `get_ep_group()` | DP × PCP × TP | MoE expert dispatch / combine |
| EPLB | `get_eplb_group()` | EP 同 ranks 独立 group | expert load balancing 通信隔离 |

---

## 21. 一句话总结

vLLM 的分布式 group 管理可以压缩成：

```text
init_distributed_environment()
  → 建立全局 WORLD
  → initialize_model_parallel()
  → 按 ExternalDP x DP x PP x PCP x TP 的 rank layout 切出各类 group
  → GroupCoordinator 封装 CPU group / device group / device communicator
  → 模型层、attention、MoE、pipeline、DP 同步通过 get_*_group() 使用通信
```

如果只记最后一句：

```text
parallel_state.py 负责“把 ranks 切成 group 并提供统一通信门面”，具体模型模块只需要“拿对应 group 调通信”。
```
