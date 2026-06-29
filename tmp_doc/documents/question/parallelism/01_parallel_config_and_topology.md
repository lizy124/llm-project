# 01. ParallelConfig 如何定义并行拓扑？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/utils.py`
- `vllm/vllm/v1/executor/multiproc_executor.py`
- `vllm/vllm/v1/executor/ray_executor.py`
- `vllm/vllm/v1/engine/core.py`

本问题关注：vLLM 的并行规模从哪里来，`tensor_parallel_size / pipeline_parallel_size / data_parallel_size / prefill_context_parallel_size / decode_context_parallel_size / expert_parallel` 等配置如何共同决定 world size，global rank / local rank / group rank 如何映射，以及这些配置如何影响 Worker 初始化、模型加载、KV cache 和 forward。

---

## 1. 一句话回答

`ParallelConfig` 是 vLLM 分布式拓扑的源头。

它先定义本模型副本内部的并行规模：

```text
world_size = pipeline_parallel_size
           * tensor_parallel_size
           * prefill_context_parallel_size
```

然后在需要 DP 时扩展成包含 data parallel 的全局规模：

```text
world_size_across_dp = world_size * data_parallel_size
```

Worker 启动后会按这个拓扑初始化 distributed environment，再在 `parallel_state.py` 中建立：

```text
WORLD group
TP group
PP group
DP group
EP group
PCP group
DCP group
EPLB group
```

所以可以把主链路概括为：

```text
ParallelConfig
  → __post_init__ 计算 world_size / 选择 executor backend
  → Executor 按 world_size 创建 Worker
  → Worker.init_device()
  → init_worker_distributed_environment()
  → init_distributed_environment()
  → ensure_model_parallel_initialized()
  → initialize_model_parallel()
  → 创建 TP / PP / DP / EP / PCP / DCP groups
```

---

## 2. 本文要回答的问题

```text
ParallelConfig 里有哪些并行相关字段？
world_size 和 world_size_across_dp 分别表示什么？
TP / PP / DP / PCP / DCP / EP 是否都是独立乘到 world_size 里？
global rank / local rank / rank_in_group 有什么区别？
一个 rank 如何同时属于 TP / PP / DP / EP / CP group？
Worker 初始化时如何使用 parallel_config 设置 device 和 distributed？
Multiproc / Ray / external launcher 下 rank 如何分配？
并行拓扑如何影响模型加载、KV cache、attention、sampling？
```

---

## 3. 先给结论：vLLM 的并行拓扑分两层

### 3.1 模型副本内部的 world_size

`ParallelConfig.__post_init__()` 中计算：

位置：`vllm/config/parallel.py:791`

```text
self.world_size = pipeline_parallel_size
                * tensor_parallel_size
                * prefill_context_parallel_size
```

这里的 `world_size` 更准确地说是：

```text
一个 DP replica 内部，用来承载一个模型副本的 worker 数量。
```

它包含：

```text
PP：pipeline stage 维度
TP：tensor parallel 维度
PCP：prefill context parallel 维度
```

注意：

```text
DCP 不乘进 world_size；
DCP 复用 TP group 内的 GPU，要求 tp_size 能被 dcp_size 整除。
```

校验位置：`vllm/config/parallel.py:498` 到 `vllm/config/parallel.py:507`

### 3.2 包含 DP 后的 world_size_across_dp

`world_size_across_dp` 是属性：

位置：`vllm/config/parallel.py:516`

```text
world_size_across_dp = world_size * data_parallel_size
```

它表示：

```text
把所有 DP replica 都算进去后的总 worker 数。
```

因此：

```text
普通内部 LB / mp / ray 场景：
  parallel_config.world_size 表示每个 DP rank 内部 TP x PP x PCP 的 worker 数。

需要建立跨 DP torch distributed world 时：
  init_distributed_environment() 会把 rank 偏移到 DP 维度，
  并把 torch world_size 调整成 world_size_across_dp。
```

对应逻辑在 `vllm/distributed/parallel_state.py:1536` 到 `vllm/distributed/parallel_state.py:1586`。

### 3.3 external launcher 是特殊情况

如果 `distributed_executor_backend == "external_launcher"`，`__post_init__()` 会把：

```text
self.world_size *= self.data_parallel_size
```

位置：`vllm/config/parallel.py:799` 到 `vllm/config/parallel.py:802`

这是因为 external launcher 下，外部启动器已经把 DP rank 也作为进程维度纳入了全局 world。

---

## 4. ParallelConfig 里的核心字段

### 4.1 TP / PP / PCP / DP

定义位置：`vllm/config/parallel.py:117`

核心字段：

```text
pipeline_parallel_size：
  pipeline parallel groups 数量，也就是模型 layer 被切成多少 stage。

 tensor_parallel_size：
  tensor parallel groups 数量，也就是单层内部 tensor 被多少 rank 合作计算。

prefill_context_parallel_size：
  prefill context parallel groups 数量，用于 prefill 上下文并行。

data_parallel_size：
  data parallel groups 数量；MoE layers 会按 TP x DP 的乘积做 expert shard。

data_parallel_size_local：
  本机上的 DP replica 数量。
  默认是 1；如果为 0，并不表示真的没有本地 DP，
  而是一个“哨兵值 / 标记值”，表示 DP 配置由外部参数层指定，
  需要在 ParallelConfig.__post_init__() 中继续推导或覆盖。

data_parallel_rank：
  当前进程在所有 DP replica 中的全局 DP rank。

data_parallel_rank_local：
  当前进程在本机内部的 DP rank。
  它区别于跨所有节点的 data_parallel_rank，
  主要用于本机多 DP replica 时计算 local_rank / GPU 绑定。
```

这几个字段决定最基本的并行空间。

### 4.2 Expert Parallel / EPLB

相关字段：

```text
enable_expert_parallel：
  MoE layers 使用 expert parallel，而不是只依赖 tensor parallel。

enable_ep_weight_filter：
  EP 启用时，模型加载阶段跳过非本地 expert 权重。

enable_eplb：
  开启 expert parallel load balancing。

eplb_config：
  EPLB 的窗口、重排间隔、冗余 experts、通信 backend 等。

expert_placement_strategy：
  experts 放置策略，支持 linear / round_robin。

all2all_backend：
  MoE expert parallel 的 all-to-all 通信 backend。
```

字段位置：`vllm/config/parallel.py:160` 到 `vllm/config/parallel.py:195`。

EP 的重要特点：

```text
EP group 不是新增 worker 维度，而是在已有模型执行拓扑上重组出来的 MoE 通信组。

原因是 MoE expert 属于具体 transformer layer：
  PP 决定这个 layer 在哪个 stage；
  TP / PCP 决定该 stage 内有哪些 rank 共同参与计算；
  DP 在 MoE + EP 下也可以参与 expert sharding。

因此开启 EP 后，vLLM 不会因为 EP 额外启动一组 worker；
它会在同一个 PP stage 内，把 DP x PCP x TP 这些已有 rank 组成 EP group，
用于 expert 权重分布和 token all-to-all dispatch。
```

### 4.3 Decode Context Parallel

相关字段：

```text
decode_context_parallel_size：
  DCP group 大小。

dcp_comm_backend：
  DCP 通信 backend，支持 ag_rs 或 a2a。

cp_kv_cache_interleave_size：
  DCP / PCP 下 KV cache 按 total CP rank 交错存放的粒度。
```

位置：`vllm/config/parallel.py:339` 到 `vllm/config/parallel.py:371`

关键约束：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

原因在源码注释中写得很明确：

```text
DCP 不改变 world size；
它复用 TP group 内的 GPU；
一个 TP group 会被拆成 tp_size / dcp_size 个 DCP groups。
```

对应校验位置：`vllm/config/parallel.py:498` 到 `vllm/config/parallel.py:507`。

---

## 5. rank 概念先分清楚

### 5.1 global rank

`rank` 是当前 worker 在 torch distributed global world 中的编号。

`rank` 的具体来源取决于当前执行后端：

```text
UniProcExecutor：
  单 worker，rank 通常为 0。

MultiprocExecutor：
  主进程创建 WorkerProc 时为每个 worker 分配 rank。

RayExecutor：
  构造 Ray worker 参数时传入 rank。

external launcher / torchrun：
  由外部启动器通过环境变量或 torch distributed 进程上下文提供 rank。
```

这些路径是互斥的：一次运行会选择一种 executor / launcher 后端，对应一种 rank 分配方式。

换句话说：

```text
rank 这个值在不同部署后端下由不同组件提供；
但在某一次实际运行中，它必须有且只有一个确定来源；
并且所有 worker 的 rank 必须和 world_size / process group 拓扑一致。
```

不能理解为同一个 worker 的 rank 可以任选来自 Multiproc、Ray 或 external launcher。

Multiproc 中创建 worker 的位置：`vllm/v1/executor/multiproc_executor.py:176`

```text
for local_rank in range(local_world_size):
  global_rank = global_start_rank + local_rank
```

Ray 中创建 worker kwargs 的位置：`vllm/v1/executor/ray_executor.py:345`

```text
for rank, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
  local_rank = node_workers[node_id].index(rank)
```

### 5.2 local_rank

`local_rank` 是当前节点内的设备编号，用来选择本进程绑定哪张 GPU。

`GroupCoordinator` 注释里明确区分了：

```text
local_rank：
  用于设备分配。

rank_in_group：
  当前 rank 在某个 group 内的位置。
```

位置：`vllm/distributed/parallel_state.py:351` 到 `vllm/distributed/parallel_state.py:373`。

### 5.3 rank_in_group

一个 global rank 会同时属于多个 group。

例如某个 rank 可能同时有：

```text
TP rank = 1
PP rank = 0
DP rank = 2
PCP rank = 0
DCP rank = 1
EP rank = 5
```

这些都是 `GroupCoordinator.rank_in_group`，不是 global rank。

`initialize_model_parallel()` 初始化结束时会打印：

```text
rank X in world size Y is assigned as
DP rank ..., PP rank ..., PCP rank ..., TP rank ..., EP rank ..., EPLB rank ...
```

位置：`vllm/distributed/parallel_state.py:1923`。

---

## 6. Worker 初始化时如何使用拓扑

### 6.1 Worker 先调整 local_rank

`GPUWorker.init_device()` 中，如果不是 Ray / external launcher，并且不是跨 DP 多节点特殊场景，会根据 DP local rank 调整本地 GPU 编号。

位置：`vllm/v1/worker/gpu_worker.py:249`

核心逻辑：

```text
tp_pp_world_size = pipeline_parallel_size * tensor_parallel_size
local_rank += dp_local_rank * tp_pp_world_size
```

位置：`vllm/v1/worker/gpu_worker.py:266` 到 `vllm/v1/worker/gpu_worker.py:273`

含义是：

```text
同一节点上如果有多个 DP replica，
每个 DP replica 占用一段 TP x PP GPU，
local_rank 需要加上 DP replica 的偏移。
```

### 6.2 Worker 绑定 device

调整后：

```text
visible_device_index = current_platform.logical_device_id_to_visible_device_id(local_rank)
device = torch.device(f"cuda:{visible_device_index}")
torch.accelerator.set_device_index(device)
```

位置：`vllm/v1/worker/gpu_worker.py:310` 到 `vllm/v1/worker/gpu_worker.py:314`

这一步决定当前 worker 真正用哪张卡。

### 6.3 Worker 初始化 distributed environment

Worker 在拿显存快照前先初始化 distributed：

位置：`vllm/v1/worker/gpu_worker.py:318` 到 `vllm/v1/worker/gpu_worker.py:328`

```text
init_worker_distributed_environment(
  vllm_config,
  rank,
  distributed_init_method,
  local_rank,
  backend,
)
```

为什么在显存快照前做？源码注释说明：

```text
NCCL buffers 会占显存；
先初始化 distributed，profile KV cache 可用显存才准确。
```

---

## 7. distributed 初始化主链路

入口在 `vllm/v1/worker/utils.py`：

位置：`vllm/v1/worker/utils.py:1200`

主链路：

```text
init_worker_distributed_environment()
  → init_batch_invariance()
  → override_envs_for_eplb()
  → set_custom_all_reduce(...)
  → init_distributed_environment(...)
  → ensure_model_parallel_initialized(
      tensor_parallel_size,
      pipeline_parallel_size,
      prefill_context_parallel_size,
      decode_context_parallel_size,
    )
  → ensure_ec_transfer_initialized(...)
```

对应源码位置：`vllm/v1/worker/utils.py:1200` 到 `vllm/v1/worker/utils.py:1234`。

这里有两个阶段：

```text
1. init_distributed_environment：
   建立 torch distributed default process group 和 WORLD group。

2. ensure_model_parallel_initialized：
   在 WORLD 上进一步切出 TP / PP / DP / EP / PCP / DCP groups。
```

---

## 8. init_distributed_environment 如何处理 DP

入口：`vllm/distributed/parallel_state.py:1536`

普通情况下它会按传入的：

```text
world_size = parallel_config.world_size
rank = 当前 DP replica 内部 rank
```

初始化。

但如果满足：

```text
config 存在；
distributed_executor_backend != external_launcher；
nnodes > 1 或 data_parallel_size > 1；
不是 elastic EP；
```

它会把 DP 纳入 torch distributed global world：

```text
rank = data_parallel_rank * world_size + rank
world_size = world_size_across_dp
```

位置：`vllm/distributed/parallel_state.py:1554` 到 `vllm/distributed/parallel_state.py:1570`。

这解释了一个容易混淆的点：

```text
ParallelConfig.world_size 初始是单个 DP replica 内部大小；
init_distributed_environment 可能临时把 torch world 调整成跨 DP 的 world_size_across_dp。
```

DP 单节点时，它还会使用 DP master IP 和端口创建 init method：

```text
data_parallel_master_ip
data_parallel_master_port / get_next_dp_init_port()
```

位置：`vllm/distributed/parallel_state.py:1577` 到 `vllm/distributed/parallel_state.py:1586`。

---

## 9. initialize_model_parallel 如何切 rank mesh

入口：`vllm/distributed/parallel_state.py:1694`

### 9.1 rank mesh 的布局顺序

源码中明确写了 layout order：

位置：`vllm/distributed/parallel_state.py:1760`

```text
ExternalDP x DP x PP x TP
```

当前实现实际 reshape 为：

```text
all_ranks = torch.arange(world_size).reshape(
  -1,
  data_parallel_size,
  pipeline_model_parallel_size,
  prefill_context_model_parallel_size,
  tensor_model_parallel_size,
)
```

位置：`vllm/distributed/parallel_state.py:1769` 到 `vllm/distributed/parallel_state.py:1775`

也就是可以理解成：

```text
ExternalDP x DP x PP x PCP x TP
```

其中：

```text
ExternalDP：
  外部 DP / verl integration 等不属于模型内部的 DP 维度。

DP：
  vLLM 内部 data parallel 维度。

PP：
  pipeline stage 维度。

PCP：
  prefill context parallel 维度。

TP：
  tensor parallel 维度。
```

### 9.2 TP group

构造位置：`vllm/distributed/parallel_state.py:1777`

```text
group_ranks = all_ranks.view(-1, tensor_model_parallel_size)
```

含义：

```text
固定 ExternalDP / DP / PP / PCP；
最后一维 TP 组成一个 TP group。
```

例如 `TP=4` 时，相邻 4 个 rank 通常组成一个 TP group。

### 9.3 DCP group

构造位置：`vllm/distributed/parallel_state.py:1794`

```text
group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size)
```

关键点：

```text
DCP 不是新扩展 world_size 的维度；
它按 decode_context_model_parallel_size 在现有 TP 相关 rank 上切组；
因此 tp_size 必须能被 dcp_size 整除。
```

### 9.4 PCP group

构造位置：`vllm/distributed/parallel_state.py:1816`

```text
group_ranks = all_ranks.transpose(3, 4)
                       .reshape(-1, prefill_context_model_parallel_size)
```

含义：

```text
PCP 是显式乘进 world_size 的维度；
初始化时会把 PCP 维度转到最后，按 prefill_context_model_parallel_size 成组。
```

### 9.5 PP group

构造位置：`vllm/distributed/parallel_state.py:1835`

```text
group_ranks = all_ranks.transpose(2, 4)
                       .reshape(-1, pipeline_model_parallel_size)
```

含义：

```text
固定其他维度，让不同 PP stage 的 rank 组成 pipeline group。
```

在经典例子里，`TP=2, PP=4`，PP group 类似：

```text
[g0, g2, g4, g6]
[g1, g3, g5, g7]
```

这也和文件顶部注释中的 Megatron 风格例子一致，位置：`vllm/distributed/parallel_state.py:1711` 到 `vllm/distributed/parallel_state.py:1718`。

### 9.6 DP group

构造位置：`vllm/distributed/parallel_state.py:1853`

```text
group_ranks = all_ranks.transpose(1, 4)
                       .reshape(-1, data_parallel_size)
```

含义：

```text
固定模型内部的 PP / PCP / TP 坐标；
跨 data_parallel_size 组成 DP group。
```

这保证同一个模型位置上的不同 DP replica 可以做 DP 同步。

### 9.7 EP group

构造位置：`vllm/distributed/parallel_state.py:1870`

只有 MoE 模型或 model_config 还未确定时才创建 EP group：

```text
if config.model_config is None or config.model_config.is_moe:
```

EP group 的 rank 数为：

```text
data_parallel_size
* prefill_context_model_parallel_size
* tensor_model_parallel_size
```

构造逻辑：`vllm/distributed/parallel_state.py:1874` 到 `vllm/distributed/parallel_state.py:1884`

含义：

```text
EP group 是同一个 PP stage 内的 MoE expert 分布和 token dispatch group。

它会跨 DP、PCP、TP 组织 expert parallel group：
  DP：MoE + EP 下，DP rank 也可以参与 expert sharding；
  PCP：prefill context parallel rank 也参与同一模型位置的计算；
  TP：同一 layer 内的 TP rank 本来就在共同处理该层。

PP 维度被排除在单个 EP group 外：
  因为 PP 切的是 layers；
  MoE expert 属于具体 transformer layer；
  每个 PP stage 只负责自己 stage 内的 MoE layers；
  所以每个 PP stage 内部各自组织 EP group，而不是跨 stage 混在一起。
```

换句话说，EP 不是新增一批 expert-only workers，而是在已有模型 worker 上重新分配 experts。
这样 token 可以在持有相关 MoE layer 的 rank 之间通过 all-to-all dispatch 到目标 expert，expert 计算完成后再 combine 回原 token 路径。

如果 `enable_eplb=True`，还会用同样的 ranks 创建独立的 EPLB group，避免 EPLB 通信和 MoE forward collectives 互相死锁。

位置：`vllm/distributed/parallel_state.py:1898` 到 `vllm/distributed/parallel_state.py:1919`。

---

## 10. 用一个例子理解 rank mesh

假设：

```text
DP = 2
PP = 2
PCP = 1
TP = 2
DCP = 1
```

那么：

```text
parallel_config.world_size = PP * TP * PCP = 2 * 2 * 1 = 4
world_size_across_dp = world_size * DP = 4 * 2 = 8
```

可以把 rank reshape 成：

```text
ExternalDP=1, DP=2, PP=2, PCP=1, TP=2
```

近似排列：

```text
DP0:
  PP0, TP0 → rank 0
  PP0, TP1 → rank 1
  PP1, TP0 → rank 2
  PP1, TP1 → rank 3

DP1:
  PP0, TP0 → rank 4
  PP0, TP1 → rank 5
  PP1, TP0 → rank 6
  PP1, TP1 → rank 7
```

则：

```text
TP groups:
  [0, 1], [2, 3], [4, 5], [6, 7]

PP groups:
  [0, 2], [1, 3], [4, 6], [5, 7]

DP groups:
  [0, 4], [1, 5], [2, 6], [3, 7]
```

如果是 MoE 模型，EP group 会按 DP x PCP x TP 组织，具体要以源码 reshape 后的 group_ranks 为准。

---

## 11. Executor 如何按拓扑创建 worker

### 11.1 MultiprocExecutor

`MultiprocExecutor._init_executor()` 会先读取并行规模：

位置：`vllm/v1/executor/multiproc_executor.py:117`

```text
tp_size, pp_size, pcp_size = self._get_parallel_sizes()
assert world_size == tp_size * pp_size * pcp_size
```

然后每个本地 rank 创建一个 Worker 进程：

位置：`vllm/v1/executor/multiproc_executor.py:176` 到 `vllm/v1/executor/multiproc_executor.py:191`

```text
for local_rank in range(local_world_size):
  global_rank = global_start_rank + local_rank
  WorkerProc.make_worker_process(..., rank=global_rank, local_rank=local_rank)
```

driver worker 的判定：

```text
rank % tensor_parallel_size == 0
```

位置：`vllm/v1/executor/multiproc_executor.py:265`。

含义：

```text
每个 TP group 的第一个 rank 作为 driver worker。
```

### 11.2 RayExecutor

Ray 中会为每个 worker 计算：

```text
rank
local_rank
assigned_physical_gpu_ids
is_driver_worker
```

位置：`vllm/v1/executor/ray_executor.py:345` 到 `vllm/v1/executor/ray_executor.py:357`。

driver worker 同样按：

```text
rank % tensor_parallel_size == 0
```

判断。

RayExecutor 还会构造 `pp_tp_workers`：

位置：`vllm/v1/executor/ray_executor.py:370` 到 `vllm/v1/executor/ray_executor.py:379`

```text
rank = pp_rank * tensor_parallel_size + tp_rank
```

注释示例：

```text
PP=2, TP=4
pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
```

这个结构用于 Ray executor 内部按 PP / TP 组织 worker DAG。

---

## 12. EngineCore 中 DP rank 的特殊处理

`EngineCore.run_engine_core()` 中会判断是否 data parallel：

位置：`vllm/v1/engine/core.py:1151`

```text
data_parallel = data_parallel_size > 1 or dp_rank > 0
```

如果是 DP：

```text
parallel_config.data_parallel_rank_local = local_dp_rank
parallel_config.data_parallel_index = dp_rank
```

位置：`vllm/v1/engine/core.py:1161` 到 `vllm/v1/engine/core.py:1186`。

对 MoE 和 dense 模型处理不同：

```text
MoE DP：
  设置 data_parallel_rank = dp_rank；
  使用 DPEngineCoreProc。

非 MoE DP：
  每个 DP rank 完全独立；
  把 data_parallel_size / data_parallel_size_local 重置成 1；
  data_parallel_index 仍保留原始 DP rank。
```

位置：`vllm/v1/engine/core.py:1187` 到 `vllm/v1/engine/core.py:1198`。

这说明：

```text
vLLM 里的 DP 不总是同一种形态；
MoE DP 可能需要 DP 组参与模型内 expert parallel / 同步；
dense 模型的多 DP rank 更像多个独立 EngineCore replica。
```

---

## 13. 拓扑如何影响后续模块

### 13.1 模型加载

并行拓扑会影响：

```text
每个 rank 加载哪些 tensor shard；
TP 下 linear / embedding / lm_head 如何切分；
PP 下当前 rank 加载哪些 layers；
EP 下当前 rank 加载哪些 experts；
enable_ep_weight_filter 是否跳过非本地 expert 权重。
```

相关后续文档：

```text
03_tensor_parallel.md
04_pipeline_parallel.md
06_expert_parallel.md
```

### 13.2 KV cache 初始化

并行拓扑会影响：

```text
每个 rank 持有哪些 layer 的 KV cache；
TP 下本 rank 有多少 KV heads；
PP 下只有本 stage 的 attention layers 需要 KV cache；
PCP / DCP 下 KV cache 是否按 context parallel rank 交错；
DP 下 KV cache 是否 replica-local。
```

`cp_kv_cache_interleave_size` 的说明位置：`vllm/config/parallel.py:359`。

相关后续文档：

```text
08_kv_cache_and_parallelism.md
```

### 13.3 Attention backend

并行拓扑会影响：

```text
Attention 初始化时的 num_heads / num_kv_heads；
DCP world size；
PCP world size；
backend 是否需要返回 LSE；
FlashAttention / FlashInfer / Triton / MLA backend 是否支持当前组合。
```

`AttentionImplBase` 会在实例创建时读取 DCP / PCP group：

```text
self.dcp_world_size = get_dcp_group().world_size
self.pcp_world_size = get_pcp_group().world_size
```

后续详见：

```text
07_context_parallel.md
09_attention_and_parallelism.md
```

### 13.4 CUDA graph / communication buffer

`parallel_state.py` 中的 `graph_capture()` 会同时进入 TP / PP group 的 graph capture context：

位置：`vllm/distributed/parallel_state.py:1410` 到 `vllm/distributed/parallel_state.py:1427`

通信 buffer 准备也会遍历：

```text
TP / PCP / PP / DP / EP / EPLB
```

位置：`vllm/distributed/parallel_state.py:1983` 到 `vllm/distributed/parallel_state.py:2001`。

这说明拓扑不仅影响 group，还会影响 compile / graph / communicator 的资源准备。

---

## 14. 常见误区

### 14.1 误区：world_size 一定等于 TP x PP x DP

在 vLLM 的 `ParallelConfig` 中：

```text
world_size = TP x PP x PCP
world_size_across_dp = world_size x DP
```

DP 是否直接乘进 torch world，要看 executor backend、DP 模式和初始化阶段。

### 14.2 误区：DCP 是一个新的 world 维度

DCP 不乘进 `world_size`。

源码注释明确说：

```text
DCP simply reuse the GPUs of TP group。
```

所以它要求：

```text
TP % DCP == 0
```

### 14.3 误区：local_rank 等于 rank_in_group

不是。

```text
local_rank：
  节点内设备编号，用来 set device。

rank_in_group：
  当前 rank 在某个 process group 内的位置。
```

### 14.4 误区：EP 是简单的 expert_parallel_size 字段

当前 `ParallelConfig` 没有一个简单的 `expert_parallel_size` 字段直接乘进 world size。

EP 由：

```text
enable_expert_parallel
DP size
TP size
PCP size
MoE model_config
```

共同决定 group 组织和 expert placement。

### 14.5 误区：DP 对 dense 和 MoE 行为完全一样

`EngineCore.run_engine_core()` 中对 MoE DP 和 dense DP 有不同处理：

```text
MoE：使用 DPEngineCoreProc，保留 DP rank。
Dense：每个 DP rank 更像独立 EngineCore，把 data_parallel_size 重置成 1。
```

---

## 15. 最小源码阅读路线

如果只想快速读懂拓扑，建议按这个顺序：

```text
1. vllm/config/parallel.py
   看 ParallelConfig 字段、world_size 计算和校验。

2. vllm/v1/executor/multiproc_executor.py
   看 mp 模式如何创建 worker rank / local_rank。

3. vllm/v1/executor/ray_executor.py
   看 Ray 模式如何设置 worker rank 和 pp_tp_workers。

4. vllm/v1/worker/gpu_worker.py
   看 Worker 如何调整 local_rank、绑定 device、初始化 distributed。

5. vllm/v1/worker/utils.py
   看 init_worker_distributed_environment 主链路。

6. vllm/distributed/parallel_state.py
   看 init_distributed_environment 和 initialize_model_parallel。

7. vllm/v1/engine/core.py
   看 DP EngineCore 如何设置 data_parallel_rank / data_parallel_index。
```

---

## 16. 本篇结论

```text
1. ParallelConfig 是 vLLM 并行拓扑的配置源头。

2. ParallelConfig.world_size 不是总 GPU 数的唯一解释；
   它默认表示一个 DP replica 内部的 TP x PP x PCP worker 数。

3. data_parallel_size 通过 world_size_across_dp 扩展全局规模，
   但是否直接纳入 torch distributed world 取决于 executor backend 和初始化路径。

4. DCP 不增加 world_size，而是复用 TP group，要求 TP size 能被 DCP size 整除。

5. initialize_model_parallel 使用 ExternalDP x DP x PP x PCP x TP 的 rank mesh，
   再通过 transpose / reshape 切出 TP、DCP、PCP、PP、DP、EP、EPLB groups。

6. local_rank 用来选择设备，rank_in_group 用来描述当前 rank 在某个并行 group 内的位置，二者不是同一个概念。

7. 后续的模型加载、KV cache、attention backend、MoE expert dispatch、pipeline stage、sampling 输出，都依赖这里建立好的拓扑。
```
