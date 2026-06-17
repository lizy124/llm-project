# 05. 运行时并行通信流程

## 1. 总览

运行时通信发生在 worker 内部，而不是 executor 控制面里。

主要流程包括：

- TP：模型层内部 all-reduce/all-gather/reduce-scatter。
- PP：stage 间 intermediate tensors send/recv，以及 sampled tokens 广播。
- DP：batch shape、padding、ubatch、cudagraph mode 协调。
- EP/MoE：router logits dispatch、token dispatch、expert outputs combine。
- DCP/PCP：context parallel 下的 attention/KV/cache 分布。
- KV/EC Transfer：跨实例 KV/encoder cache 传输。

本文件重点讲 PP、DP、EP、DCP/PCP 的运行时路径。

## 2. Pipeline Parallel，PP

关键文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:51`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:964`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1059`

PP 有两条通信线：

1. 中间 hidden states / intermediate tensors 在 PP stage 间传输。
2. last PP rank 生成的 sampled tokens 广播给前面 ranks。

这两条通信线故意使用不同 communicator，避免互相阻塞。

## 3. PP 中间张量传输

### 3.1 接收上游 tensors

GPU Worker 的 `execute_model()` 中，如果：

```text
forward_pass and not get_pp_group().is_first_rank
```

则调用：

```text
get_pp_group().irecv_tensor_dict(
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:853`。

返回：

- `tensor_dict`
- `comm_handles`
- `comm_postprocess`

然后包装成：

```text
AsyncIntermediateTensors(tensor_dict, comm_handles, comm_postprocess)
```

### 3.2 AsyncIntermediateTensors

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:85`

它继承 `IntermediateTensors`，但带 lazy comm synchronization。

关键方法：

- `wait_for_comm()`
- `__getattribute__()` 拦截 `.tensors`

当 model runner 真正访问 `.tensors` 时，才等待通信 handles 完成并执行 postprocess。

好处：PP recv 可以与部分准备工作重叠。

### 3.3 发送给下游

如果 `model_runner.execute_model()` 返回 `IntermediateTensors`，说明当前 rank 不是 PP last rank。

GPU Worker 调用：

```text
self._pp_send_work = get_pp_group().isend_tensor_dict(
    output.tensors,
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:889`。

下一轮 `execute_model()` 开头会等待上一次 PP send 完成：

```text
if self._pp_send_work:
    for handle in self._pp_send_work:
        handle.wait()
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:811`。

## 4. PP sampled token 广播

文件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py`

### 4.1 PPHandler

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:51`

作用：在 PP 下，last rank 采样后，需要把 sampled tokens 和 rejection info 广播给前面的 PP ranks，用于下一步 decode 状态更新。

### 4.2 sibling communicator

`PPHandler.__init__()` 中创建：

```text
self.broadcast_group = get_pp_group().make_sibling_device_group(group_desc="pp_broadcast")
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:84`。

原因：sampled token broadcast 不应与 intermediate tensor send/recv 共用 PP device group，否则可能在 NCCL communicator 上串行。

### 4.3 receive()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:125`

非 last rank 调用。

流程：

1. 计算哪些请求需要 sampled output：`compute_need_sampled_mask(input_batch)`。
2. 在 `broadcast_stream` 上等待 main stream。
3. 分配 sampled tokens buffer 和 num_sampled/num_rejected buffer。
4. 调 `torch.distributed.broadcast(..., src=last_rank, group=broadcast_group)`。
5. 记录 event。
6. 把结果放入 FIFO queue。

### 4.4 broadcast()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:169`

last rank 调用。

如果没有请求需要 sampled outputs，直接返回。

否则在 side stream 上 broadcast：

- `sampled_token_ids`
- `combined = stack(num_sampled, num_rejected)`

### 4.5 get_prev_sampled_outputs()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:92`

非 last rank 在 step T+pp_size 消费 step T 的 sampled outputs。

它会：

- 等待 broadcast event。
- 根据 request index generation counter 过滤已经 freed 的 req index。
- 返回给 `GPUModelRunner.postprocess_sampled()` 使用。

## 5. Data Parallel，DP batch 协调

文件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py`

DP batch 协调不是传统梯度同步，而是推理时多 DP rank 之间协调 batch 形态。

目标：

1. 所有 DP rank 是否都启用 microbatching。
2. 每个 rank 最终 token 数是否需要 padding 到一致。
3. cudagraph mode 是否需要对齐。

### 5.1 coordinate_batch_across_dp()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:164`

输入：

- `num_tokens_unpadded`
- `allow_microbatching`
- `parallel_config`
- `num_tokens_padded`
- `uniform_decode`
- `cudagraph_mode`

输出：

- `should_ubatch`
- `num_tokens_after_padding`
- `synced_cudagraph_mode`

如果 `data_parallel_size == 1`，直接返回。

### 5.2 _run_ar()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:36`

构造一个 `[4, dp_size]` 的 int32 tensor：

- row 0：每个 DP rank 的原始 token 数。
- row 1：每个 DP rank 的 padded token 数。
- row 2：该 rank 是否希望 ubatch。
- row 3：该 rank 的 cudagraph mode。

然后调用：

```text
dist.all_reduce(tensor, group=group)
```

### 5.3 CPU/GPU group 选择

`_get_device_and_group()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:18`。

默认使用：

- `get_dp_group().device`
- `get_dp_group().device_group`

如果 `parallel_config.disable_nccl_for_dp_synchronization=True`，则使用：

- CPU tensor。
- `get_dp_group().cpu_group`。

### 5.4 ubatch 决策

`_post_process_ubatch()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:57`。

只有当所有 DP rank 都设置 should_ubatch=1，才真正启用 ubatching。

并且还要检查最后一个 ubatch 不能为空。

### 5.5 DP padding 决策

`_post_process_dp_padding()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/dp_utils.py:77`。

如果需要 DP padding，就把所有 rank padding 到最大 token 数。

需要 padding 的条件：

- cudagraph mode 非 NONE。
- 或启用 ubatching。

## 6. Expert Parallel / MoE 通信

关键位置：

- `ParallelConfig.enable_expert_parallel`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:162`
- `ParallelConfig.all2all_backend`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:185`
- `get_ep_group()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1366`
- `GroupCoordinator.dispatch_router_logits()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1186`
- `GroupCoordinator.dispatch()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1206`
- `GroupCoordinator.combine()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1228`

### 6.1 EP group 组成

EP group 的 ranks 来自：

```text
DP × PCP × TP
```

按 PP 维度分组，即同一个 pipeline stage 内的 DP/PCP/TP ranks 形成 EP group。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1850`。

### 6.2 MoE 典型通信

```text
hidden states
  -> router logits
  -> dispatch_router_logits()
  -> 根据 routing 进行 token dispatch
  -> 各 rank 本地 expert 计算
  -> combine() 合并 expert outputs
```

实际后端由 device communicator 和 all2all backend 决定。

### 6.3 sequence parallel MoE

`ParallelConfig.use_sequence_parallel_moe` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:633`。

它在特定条件下启用：

- expert parallel 开启。
- TP > 1。
- DP > 1。
- all2all backend 支持 sequence parallel。

作用：避免 TP attention 后的重复计算，并改善 MoE 通信/计算布局。

### 6.4 batched DP MoE

`ParallelConfig.use_batched_dp_moe` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:650`。

用于部分低延迟或 NIXL 类型 all2all backend。

## 7. EPLB 通信

EPLB group 在 `initialize_model_parallel()` 中使用与 EP 相同 ranks 创建，但是独立 process group。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1878`。

原因：

```text
isolate EPLB communications from MoE forward pass collectives
and prevent deadlocks
```

EPLB 相关文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/eplb/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/eplb_utils.py`

## 8. DCP / PCP 运行时语义

### 8.1 DCP

DCP 是 decode context parallel。

关键配置：

- `decode_context_parallel_size`
- `dcp_comm_backend`
- `cp_kv_cache_interleave_size`

DCP 不改变 world size，只切分 TP group 内 ranks。

### 8.2 PCP

PCP 是 prefill context parallel。

PCP 会参与 world size：

```text
world_size = TP × PP × PCP
```

PCP group 通过 `all_ranks.transpose(3, 4)` 创建，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1796`。

### 8.3 CP KV cache interleave

字段：`cp_kv_cache_interleave_size`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:351`

语义：在 DCP 或 PCP 下，KV cache 按 total CP rank 交错存储。

```text
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

interleave size 可以是 token-level，也可以是 block-level。

## 9. 一次 execute_model 中的通信时序

```text
每个 worker 收到 SchedulerOutput
  -> 如果有上轮 PP send，先 wait
  -> 如果当前不是 PP first rank：
       irecv_tensor_dict() 接收上游 hidden states
       包装 AsyncIntermediateTensors
  -> model_runner.execute_model()
       内部可能触发 TP all_reduce/all_gather/reduce_scatter
       MoE 层可能触发 EP dispatch/combine
       DP 环境可能先 coordinate_batch_across_dp()
  -> 如果输出是 IntermediateTensors 且不是 PP last rank：
       isend_tensor_dict() 发给下游
       返回 None
  -> 如果是 PP last rank：
       返回 ModelRunnerOutput 或等待 sample_tokens()
  -> PP last rank 通过 PPHandler.broadcast() 广播 sampled tokens
  -> 非 last ranks 后续通过 PPHandler.get_prev_sampled_outputs() 更新 decode 状态
```

## 10. 常见通信问题定位

### 10.1 PP 卡住

看：

- `Worker.execute_model()` 中 `irecv_tensor_dict` / `isend_tensor_dict`。
- `PPHandler` sampled token broadcast 是否匹配。
- 是否所有 PP ranks 都进入同一 step。

### 10.2 DP 卡住

看：

- `coordinate_batch_across_dp()` 是否所有 DP ranks 调用。
- `disable_nccl_for_dp_synchronization` 是否导致 CPU/GPU group 不一致。
- DP ranks 的 `cudagraph_mode` 和 ubatch 决策是否同步。

### 10.3 EP/MoE 卡住

看：

- `enable_expert_parallel`。
- `all2all_backend`。
- EP group 是否创建。
- EPLB 是否与 EP forward 共用 group。当前代码已单独创建 EPLB group 防死锁。

### 10.4 DCP/PCP 错误

看：

- `tp_size % dcp_size == 0`。
- `cp_kv_cache_interleave_size` 与 block size 是否兼容。
- PCP 是否参与 world size。

## 11. 一句话总结

运行时通信的主干在 worker 内：PP 用 tensor_dict p2p 传中间张量并用 sibling group 广播 sampled tokens，DP 用 all-reduce 协调 batch 形态，EP/MoE 通过 device communicator 做 dispatch/combine，DCP/PCP 改变 context/KV 的分布方式，而 executor 只负责把 control message 发到所有 worker。
