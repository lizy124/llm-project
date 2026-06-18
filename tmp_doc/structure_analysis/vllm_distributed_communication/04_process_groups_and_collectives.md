# 04. 进程组与通信原语

## 1. 核心文件

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/communication_op.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/base_device_communicator.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/cuda_communicator.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/cpu_communicator.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/xpu_communicator.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/all2all.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/custom_all_reduce.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/device_communicators/shm_broadcast.py`

`parallel_state.py` 是 vLLM 分布式运行时中枢。

它负责：

- 初始化 torch distributed。
- 创建 WORLD/TP/PP/DP/DCP/PCP/EP/EPLB groups。
- 包装 process group 成 `GroupCoordinator`。
- 提供 all_reduce、all_gather、reduce_scatter、broadcast、send/recv、tensor_dict send/recv。
- 提供 message queue broadcaster。
- 提供 graph capture 上下文。
- 销毁分布式状态与释放显存。

## 2. GroupCoordinator

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:351`

它是 PyTorch `ProcessGroup` 的 wrapper。

### 2.1 解决的问题

PyTorch ProcessGroup 只表示一组 rank 和某个 backend。`GroupCoordinator` 额外管理：

- CPU group。
- device group。
- 当前 rank 在 group 中的位置。
- 本地 rank 与设备。
- device communicator。
- message queue broadcaster。
- custom op collective。
- tensor dict 传输协议。

### 2.2 核心字段

在类注释和 `__init__()` 中可以看到：

- `rank`：全局 rank。
- `ranks`：该 group 内的全局 rank 列表。
- `world_size`：group size。
- `local_rank`：本地设备编号。
- `rank_in_group`：当前 rank 在 group 内的编号。
- `cpu_group`：Gloo/CPU 通信 group。
- `device_group`：NCCL/XPU/其他设备通信 group。
- `device_communicator`：高性能设备通信封装。
- `mq_broadcaster`：共享内存/消息队列广播器。
- `unique_name`：custom op 查找 group 用。

### 2.3 __init__()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:380`

流程：

1. 生成 unique group name。
2. 注册 group 到全局 registry。
3. 获取当前 torch distributed rank。
4. 根据环境变量选择：
   - `split_group` 路径。
   - 传统 `new_group` 路径。
5. 为每组 ranks 创建：
   - device process group。
   - CPU/Gloo process group。
6. 找到当前 rank 所属的 group ranks。
7. 根据平台设置 device。
8. 如果启用 device communicator 且 group size > 1，创建平台对应 communicator。
9. 如果启用 message queue broadcaster，创建 MQ。
10. 设置 custom op collective 标志。

### 2.4 device communicator 创建

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:453`

核心：

```text
device_comm_cls = resolve_obj_by_qualname(current_platform.get_device_communicator_cls())
self.device_communicator = device_comm_cls(...)
```

这让 CUDA/XPU/CPU 等平台可以提供各自的通信实现。

## 3. 全局 group 单例

在 `parallel_state.py` 中维护：

- `_WORLD`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1237`
- `_TP`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1331`
- `_DCP`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1339`
- `_PP`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1347`
- `_DP`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1355`
- `_EP`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1363`
- `_EPLB`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1375`
- `_PCP`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1387`

对应 getter：

- `get_world_group()`
- `get_tp_group()`
- `get_dcp_group()`
- `get_pp_group()`
- `get_dp_group()`
- `get_ep_group()`
- `get_eplb_group()`
- `get_pcp_group()`

这些 getter 是模型代码和 worker 代码获取通信 group 的统一入口。

## 4. init_distributed_environment()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1516`

职责：初始化全局 torch distributed 环境与 WORLD group。

### 4.1 输入参数

- `world_size`
- `rank`
- `distributed_init_method`
- `local_rank`
- `backend`
- `timeout`

### 4.2 DP / 多节点 rank 调整

如果存在 DP 或多节点，并且不是 external launcher，也不是 elastic EP：

```text
rank = data_parallel_rank * world_size + rank
world_size = parallel_config.world_size_across_dp
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1545`。

如果多节点：使用 `master_addr/master_port`。

如果单节点 DP：使用 `data_parallel_master_ip` 和自动获取的 DP init port。

### 4.3 初始化 torch process group

如果 torch distributed 未初始化：

- 默认路径：`torch.distributed.init_process_group(...)`
- split_group 路径：`_init_process_group_for_split_group(...)`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1567`。

如果 backend 不可用，会 fallback 到 Gloo。

### 4.4 初始化 WORLD

如果 `_WORLD` 为空：

```text
ranks = list(range(torch.distributed.get_world_size()))
_WORLD = init_world_group(ranks, local_rank, backend)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1643`。

如果配置 `nnodes_within_dp > 1`，还会创建 `_INNER_DP_WORLD`。

## 5. initialize_model_parallel()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1674`

职责：根据 TP/PP/PCP/DCP/DP/EP/EPLB 配置创建各类 model parallel groups。

### 5.1 rank 布局

核心 rank tensor：

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

维度解释：

```text
ExternalDP × DP × PP × PCP × TP
```

### 5.2 创建 TP group

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1757`

```text
group_ranks = all_ranks.view(-1, tensor_model_parallel_size)
```

TP group 使用 message queue broadcaster。

### 5.3 创建 DCP group

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1774`

DCP 复用 TP 维度：

```text
group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size)
```

DCP group 也启用 message queue broadcaster。

### 5.4 创建 PCP group

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1796`

```text
group_ranks = all_ranks.transpose(3, 4).reshape(-1, prefill_context_model_parallel_size)
```

### 5.5 创建 PP group

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1815`

```text
group_ranks = all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size)
```

### 5.6 创建 DP group

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1833`

```text
group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size)
```

Elastic EP 模式下使用 stateless group，否则使用普通 model parallel group。

### 5.7 创建 EP / EPLB group

EP group 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1850`。

EPLB group 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1878`。

EP group 只在 MoE 模型或模型 config 未知时创建。EPLB 使用相同 ranks 但单独 group。

## 6. Collective 通信原语

### 6.1 all_reduce()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:607`

逻辑：

1. group size 为 1 时直接返回输入。
2. 如果启用 custom op：

```text
torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
```

3. 否则调用：

```text
self.device_communicator.all_reduce(input_)
```

### 6.2 all_gather()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:636`

逻辑与 all_reduce 类似，custom op 路径调用：

```text
torch.ops.vllm.all_gather(input_, dim, world_size, group_name=self.unique_name)
```

否则调用 device communicator。

### 6.3 reduce_scatter()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:667`

custom op 路径：

```text
torch.ops.vllm.reduce_scatter(input_, dim, world_size, group_name=self.unique_name)
```

否则调用 device communicator。

### 6.4 broadcast()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:711`

使用 torch distributed broadcast：

```text
torch.distributed.broadcast(input_, src=self.ranks[src], group=self.device_group)
```

这里的 `src` 是 group 内 rank，不是全局 rank。

### 6.5 gather()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:695`

委托给 device communicator。

## 7. Object 通信

### 7.1 broadcast_object()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:726`

逻辑：

- 如果存在 `mq_broadcaster`，优先使用 MQ 广播 object。
- 否则使用 `torch.distributed.broadcast_object_list(...)`。

Object 通信用于控制信息和 metadata，不适合大 tensor。

### 7.2 send_object() / recv_object()

定义：

- `send_object()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:767`
- `recv_object()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:794`

实现：

1. pickle object。
2. 先发送 size。
3. 再发送 bytes tensor。
4. 使用 CPU group。

## 8. Tensor Dict 通信

Tensor dict 通信是 PP 和部分 worker 间传输 intermediate tensors 的重要协议。

### 8.1 broadcast_tensor_dict()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:830`

流程：

1. 先广播 metadata。
2. 根据 metadata 创建或复用 tensor。
3. 对每个 tensor 使用对应 device/CPU group 广播。
4. 支持 async op。

### 8.2 send_tensor_dict()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:926`

支持：

- 发送 metadata。
- 发送 CPU/GPU tensors。
- 对部分 tensors 使用 TP group all-gather 优化。

### 8.3 isend_tensor_dict()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:964`

非阻塞 send，返回 handles。

GPU Worker 的 PP send 使用它：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:890`

### 8.4 recv_tensor_dict()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1021`

接收 tensor dict，并支持 all-gather group。

### 8.5 irecv_tensor_dict()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1059`

非阻塞接收，返回：

- tensor_dict
- comm_handles
- comm_postprocess

GPU Worker 的 PP recv 使用它：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:853`

## 9. P2P tensor send / recv

方法：

- `send()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1154`
- `recv()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1161`

它们委托给 device communicator。

## 10. EP / all2all 相关方法

`GroupCoordinator` 提供：

- `dispatch_router_logits()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1186`
- `dispatch()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1206`
- `combine()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1228`

这些方法委托给 device communicator，服务 MoE expert parallel。

## 11. make_sibling_device_group()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:486`

作用：创建与当前 group rank 成员相同、但使用独立 communicator 的 device process group。

典型用途：PP sampled token broadcast。

`PPHandler` 创建 sibling group 的位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/pp_utils.py:84`

原因：sampled token broadcast 不应与 inter-stage hidden-state p2p send/recv 共用 communicator，否则会串行阻塞。

## 12. graph_capture()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:563`

用于 CUDA graph capture 场景：

- 创建或使用传入 stream。
- 让 communicator 进入 capture context。
- 等待当前 stream。
- 在目标 stream 中 yield。

全局 helper：`graph_capture()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1396`。

## 13. custom collective ops

`parallel_state.py` 顶部注册了 custom op wrapper：

- `all_reduce()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:130`
- `reduce_scatter()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:142`
- `all_gather()`：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:160`

这些函数通过 group name 查找 `GroupCoordinator`，再转发到实际 group。

目的：Dynamo/custom op 不方便传 Python object，所以用 group name 字符串作为桥。

## 14. 销毁路径

### 14.1 destroy_model_parallel()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:2008`

销毁 TP/PP/DP/DCP/PCP/EP/EPLB 等 group。

### 14.2 destroy_distributed_environment()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:2047`

销毁 WORLD 和 torch distributed process group。

### 14.3 cleanup_dist_env_and_memory()

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:2057`

完整清理分布式环境和显存。EngineCore shutdown 会调用它，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:659`。

## 15. 一句话总结

`parallel_state.GroupCoordinator` 是 vLLM 分布式数据面的统一门面：它把 PyTorch process group、CPU/GPU group、平台 device communicator、custom collective op、message queue broadcaster 和 tensor dict 协议统一成一套可被 TP/PP/DP/DCP/PCP/EP/EPLB 复用的通信接口。
