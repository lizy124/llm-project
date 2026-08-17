# 02. ParallelConfig 与并行拓扑

## 1. 核心文件

- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:57`：`EPLBConfig`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:117`：`ParallelConfig`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:23`：`KVTransferConfig`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py:16`：`ECTransferConfig`

`ParallelConfig` 是 vLLM 分布式拓扑的总配置入口。它决定：

- 创建多少 worker。
- 使用哪个 executor backend。
- TP/PP/DP/PCP/DCP/EP/EPLB 的大小。
- Ray/MP/uni/external launcher 使用哪种控制面。
- MoE expert parallel 与 all2all backend。
- DP 同步使用 NCCL 还是 Gloo。
- 多节点 rank/master addr/port。
- Elastic EP、EPLB、DBO、context parallel 等高级能力。

## 2. 并行维度字段

### 2.1 Pipeline Parallel

字段：`pipeline_parallel_size`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:120`

表示 pipeline stage 数。PP > 1 时：

- executor 必须支持 PP。
- worker 之间会传递 intermediate tensors。
- last PP rank 负责最终 sampling/pooling。
- 非 last PP rank 通常返回 `None` 给 executor。

### 2.2 Tensor Parallel

字段：`tensor_parallel_size`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:122`

表示 tensor parallel group 大小。TP 主要影响：

- 权重分片。
- attention/MLP 中的 collective ops。
- TP group 上的 all-reduce、all-gather、reduce-scatter。
- PP send/recv 时某些 tensor 可以借助 TP all-gather。

### 2.3 Prefill Context Parallel，PCP

字段：`prefill_context_parallel_size`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:124`

表示 prefill context parallel group 大小。PCP 是 world size 的组成部分：

```text
world_size = TP × PP × PCP
```

### 2.4 Decode Context Parallel，DCP

字段：`decode_context_parallel_size`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:331`

DCP 不增加 world size，而是在 TP group 内复用 GPU。代码注释明确：

```text
dcp_size must not exceed tp_size, because world size does not change by DCP,
it simply reuses the GPUs of TP group.
```

相关 group 初始化在 `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1774`。

DCP 通信后端字段：`dcp_comm_backend`，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:343`。

可选：

- `ag_rs`：AllGather + ReduceScatter。
- `a2a`：All-to-All exchange + combine kernel。

### 2.5 Data Parallel，DP

字段：

- `data_parallel_size`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:126`
- `data_parallel_size_local`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:129`
- `data_parallel_rank`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:133`
- `data_parallel_rank_local`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:136`
- `data_parallel_master_ip`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:138`
- `data_parallel_master_port`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:142`
- `data_parallel_backend`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:144`

DP 用于多个模型副本并行。DP group 内所有 rank 通常需要同步 step 形态，否则容易出现 collective 不匹配或死锁。

DP batch 协调代码在：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:164`

### 2.6 Expert Parallel，EP

字段：

- `enable_expert_parallel`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:162`
- `enable_ep_weight_filter`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:164`
- `expert_placement_strategy`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:175`
- `all2all_backend`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:185`

EP 用于 MoE 模型，将 experts 分布到不同 rank。

`expert_placement_strategy` 支持：

- `linear`：连续放置。
- `round_robin`：轮转放置。

`all2all_backend` 可选：

- `allgather_reducescatter`
- `deepep_high_throughput`
- `deepep_low_latency`
- `mori_high_throughput`
- `mori_low_latency`
- `nixl_ep`
- `flashinfer_nvlink_two_sided`
- `flashinfer_nvlink_one_sided`

### 2.7 EPLB

字段：

- `enable_eplb`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:171`
- `eplb_config`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:173`

`EPLBConfig` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:57`。

关键字段：

- `window_size`
- `step_interval`
- `num_redundant_experts`
- `log_balancedness`
- `log_balancedness_interval`
- `use_async`
- `policy`
- `communicator`

EPLB 必须在 expert parallel 开启时才能启用，校验位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:467`。

## 3. 执行后端字段

字段：`distributed_executor_backend`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:240`

支持：

- `uni`
- `mp`
- `ray`
- `external_launcher`
- 自定义 `Executor` 类路径或类型

选择逻辑在：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:48`

对应关系：

```text
uni -> UniProcExecutor
mp -> MultiprocExecutor
ray -> RayDistributedExecutor 或 RayExecutorV2
external_launcher -> ExecutorWithExternalLauncher
```

Ray V2 是否启用取决于环境变量 `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`。

## 4. Worker 类字段

字段：

- `worker_cls`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:256`
- `sd_worker_cls`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:259`
- `worker_extension_cls`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:262`

`worker_cls` 决定 executor 创建哪个 worker。实际解析和动态注入在：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:249`

如果 `worker_extension_cls` 非空，`WorkerWrapperBase.init_worker()` 会把 extension 动态加入 worker class bases，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/worker_base.py:261`。

这使得额外 collective RPC 方法可以注入 worker。

## 5. 多节点和 rank 字段

字段：

- `master_addr`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:267`
- `master_port`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:270`
- `node_rank`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:273`
- `nnodes`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:276`
- `rank`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:319`

这些字段用于 MP 多节点或 external launcher。

`init_distributed_environment()` 会根据 DP、多节点和 backend 调整 rank/world_size/init_method，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1516`。

## 6. world_size 公式

`ParallelConfig.world_size` 的注释在 `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:316`：

```text
world_size is TPxPP, it affects the number of workers we create.
```

但当前代码中，PCP 也参与 executor world size。在 Multiproc/Ray V2 中校验：

```text
world_size == tensor_parallel_size × pipeline_parallel_size × prefill_context_parallel_size
```

位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:117`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/ray_executor_v2.py:263`

当需要跨 DP 调整时，`parallel_state.init_distributed_environment()` 会把 rank 和 world size 扩展到 DP 维度：

```text
rank = data_parallel_rank * world_size + rank
world_size = world_size_across_dp
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1545`。

## 7. process group 布局公式

`initialize_model_parallel()` 中使用：

```text
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1749`。

维度顺序为：

```text
ExternalDP × DP × PP × PCP × TP
```

然后对不同维度 transpose/reshape，构建各 group。

### 7.1 TP group

```text
group_ranks = all_ranks.view(-1, tensor_model_parallel_size)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1757`。

### 7.2 DCP group

```text
group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1774`。

DCP 复用 TP 维度。

### 7.3 PCP group

```text
group_ranks = all_ranks.transpose(3, 4).reshape(-1, prefill_context_model_parallel_size)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1796`。

### 7.4 PP group

```text
group_ranks = all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1815`。

### 7.5 DP group

```text
group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1833`。

### 7.6 EP group

```text
group_ranks = all_ranks.transpose(1, 2).reshape(
    -1,
    data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size,
)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1850`。

### 7.7 EPLB group

EPLB 使用与 EP 相同的 rank 集合，但创建独立 group，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1878`。

原因：隔离 EPLB 通信与 MoE forward collectives，避免死锁。

## 8. DP 相关辅助属性

### 8.1 world_size_across_dp

属性位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:508`

含义：模型并行 world size 跨 DP 后的总 world size。

### 8.2 local_world_size

属性位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:675`

MultiprocExecutor 会使用它决定本节点创建多少 worker，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/multiproc_executor.py:256`。

### 8.3 node_rank_within_dp / nnodes_within_dp

位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:662`
- `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:667`

用于在多节点 + DP 模式下定位当前节点属于哪个 DP 副本。

## 9. DP 同步配置

字段：`disable_nccl_for_dp_synchronization`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:224`

它影响 `vllm/v1/worker/dp_utils.py`：

- False：DP 同步使用 NCCL/device group。
- True：DP 同步使用 CPU/Gloo group。

选择逻辑在：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:18`。

## 10. DBO / ubatch 配置

字段：

- `enable_dbo`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:208`
- `ubatch_size`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:210`
- `dbo_decode_token_threshold`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:213`
- `dbo_prefill_token_threshold`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:218`

DBO 会影响 worker workspace 数量：

```text
num_ubatches = 2 if enable_dbo else 1
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:322`。

DP batch 协调中也会决定是否所有 DP rank 都启用 microbatching：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:101`。

## 11. Elastic EP 配置

字段：`enable_elastic_ep`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:205`

它启用 stateless NCCL groups for DP/EP。

相关逻辑：

- `init_distributed_environment()` 中进入 `_init_elastic_ep_world()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1640`
- `initialize_model_parallel()` 中 DP/EP/EPLB 使用 `_init_stateless_group()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1837`

## 12. KV / EC Transfer 配置

### 12.1 KVTransferConfig

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:23`

关键字段：

- `kv_connector`
- `engine_id`
- `kv_buffer_device`
- `kv_buffer_size`
- `kv_role`
- `kv_rank`
- `kv_parallel_size`
- `kv_ip`
- `kv_port`
- `kv_connector_extra_config`
- `kv_connector_module_path`
- `enable_permute_local_kv`
- `kv_load_failure_policy`

角色：

- `kv_producer`
- `kv_consumer`
- `kv_both`

属性：

- `is_kv_transfer_instance`
- `is_kv_producer`
- `is_kv_consumer`

### 12.2 ECTransferConfig

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py:16`

关键字段：

- `ec_connector`
- `engine_id`
- `ec_buffer_device`
- `ec_buffer_size`
- `ec_role`
- `ec_rank`
- `ec_parallel_size`
- `ec_ip`
- `ec_port`
- `ec_connector_extra_config`
- `ec_connector_module_path`

角色：

- `ec_producer`
- `ec_consumer`
- `ec_both`

属性：

- `is_ec_transfer_instance`
- `is_ec_producer`
- `is_ec_consumer`

## 13. 配置校验重点

`ParallelConfig._validate_parallel_config()` 从 `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:432` 开始。

关键校验包括：

1. API process rank 合法性。
2. 移除旧 all2all backend fallback。
3. `data_parallel_size_local <= data_parallel_size`。
4. `data_parallel_external_lb` 只能在 DP > 1 时使用。
5. NUMA bind 参数必须与 `numa_bind=True` 一起使用。
6. `enable_eplb` 必须满足：
   - CUDA/ROCm 平台。
   - `enable_expert_parallel=True`。
   - `TP × DP > 1`。
7. DCP 必须能整除 TP。
8. 多节点、Ray、MP、external launcher 的组合必须合法。

## 14. 一句话总结

`ParallelConfig` 不是简单的参数集合，而是 vLLM 分布式通信的拓扑编译器：它把用户指定的 TP/PP/DP/PCP/DCP/EP/EPLB 与 Ray/MP/uni 后端组合成 executor 创建策略、worker rank 分配、process group 布局和运行时通信方式。
