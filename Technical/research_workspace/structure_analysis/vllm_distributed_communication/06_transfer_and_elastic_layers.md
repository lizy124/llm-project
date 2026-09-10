# 06. KV/EC/Weight Transfer 与 Elastic EP

## 1. 总览

除了模型并行组中的 TP/PP/DP/EP 通信，vLLM 还提供多种专用传输层：

- KV cache transfer：跨 vLLM 实例传输 decoder KV cache，典型用于 prefill/decode disaggregation。
- EC transfer：跨 vLLM 实例传输 encoder cache。
- Weight transfer：从 trainer 或外部服务向 vLLM worker 更新权重。
- Elastic EP：弹性 expert parallel，动态扩缩 expert parallel groups。
- EPLB：expert parallel load balancing，用于 MoE expert 负载均衡。

这些传输层大多与 worker 生命周期强耦合，但配置和调度决策位于 engine/scheduler 层。

## 2. KVTransferConfig

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:23`

### 2.1 字段

- `kv_connector`
  - KV connector 名称。
- `engine_id`
  - 当前 engine 的唯一 id，默认自动 UUID。
- `kv_buffer_device`
  - connector buffer 所在设备，默认当前平台设备类型。
- `kv_buffer_size`
  - buffer 大小，单位字节。
- `kv_role`
  - 当前实例角色。
- `kv_rank`
  - 当前实例在 KV transfer 组中的 rank。
- `kv_parallel_size`
  - KV transfer parallel size。
- `kv_ip`
  - connector ip。
- `kv_port`
  - connector port。
- `kv_connector_extra_config`
  - connector 私有配置。
- `kv_connector_module_path`
  - V1 中支持动态加载 connector 模块路径。
- `enable_permute_local_kv`
  - HND/NHD KV transfer 实验开关。
- `kv_load_failure_policy`
  - load 失败策略：`recompute` 或 `fail`。

### 2.2 角色

类型定义在文件顶部：

- Producer：`kv_producer` 或 `kv_both`
- Consumer：`kv_consumer` 或 `kv_both`

属性：

- `is_kv_transfer_instance`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:108`
- `is_kv_producer`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:112`
- `is_kv_consumer`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:116`

### 2.3 校验

`__post_init__()` 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:92`

逻辑：

1. 如果 `engine_id` 为空，生成 UUID。
2. 校验 `kv_role` 合法。
3. 如果设置了 `kv_connector` 但没有 `kv_role`，报错。

## 3. KV Transfer 初始化

### 3.1 Worker 侧初始化

GPU Worker 在初始化 KV cache 前调用：

```text
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:575`。

这一步必须在 `model_runner.initialize_kv_cache(kv_cache_config)` 之前，因为 connector 需要拿到 KV cache config，而后续初始化可能会注入不属于 connector 的 KV cache sharing layers。

### 3.2 Scheduler 侧 connector

Scheduler 构造时，如果 `kv_transfer_config` 非空，会创建 scheduler 侧 connector。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:123`。

角色：`KVConnectorRole.SCHEDULER`。

Scheduler 侧负责：

- 新请求通知 connector。
- 查询外部 KV 命中 token 数。
- 在分配本地 blocks 后更新 connector 状态。
- 构建 `kv_connector_metadata` 放入 `SchedulerOutput`。
- 处理 worker 返回的 transfer 完成/失败信息。

### 3.3 Worker 侧 connector

worker/model runner 侧 connector 文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`

它负责：

- 注册真实 KV cache tensors。
- pre-forward load KV。
- post-forward save KV。
- 报告 finished sending/receiving。
- 报告 invalid block ids。
- 返回 stats/events/worker meta。

## 4. KV Connector 握手 metadata

GPU Worker 提供：

```text
get_kv_connector_handshake_metadata()
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:526`。

流程：

1. 如果没有 KV transfer group，返回 None。
2. 取 `connector.get_handshake_metadata()`。
3. 获取当前 `pp_rank = get_pp_group().rank_in_group`。
4. 获取当前 `tp_rank = get_tp_group().rank_in_group`。
5. 返回：

```text
{(pp_rank, tp_rank): metadata}
```

Executor 侧抽象方法：

- `Executor.get_kv_connector_handshake_metadata()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/executor/abstract.py:204`

EngineCore 初始化时会聚合所有 worker metadata，然后设置到 scheduler 端 connector。

## 5. KV Transfer 运行时链路

```text
Request 进入 Scheduler
  -> scheduler connector on_new_request()
  -> waiting 调度时查询本地 prefix cache
  -> scheduler connector get_num_new_matched_tokens()
  -> 如果 external KV 命中：
       KVCacheManager.allocate_slots(..., num_external_computed_tokens, delay_cache_blocks)
       scheduler connector update_state_after_alloc()
       SchedulerOutput.kv_connector_metadata = connector.build_connector_meta()
  -> Worker KVConnector.pre_forward()
       handle_preemptions()
       bind_connector_metadata()
       start_load_kv()
  -> model forward 或 no_forward
  -> Worker KVConnector.post_forward()
       wait_for_save()
       get_finished()
       get_block_ids_with_load_errors()
       get stats/events/worker meta
  -> Scheduler.update_from_output()
       处理 invalid blocks
       更新 finished sending/receiving
       失败则 recompute 或 fail
```

关键文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/kv_transfer/**`

## 6. KV load 失败策略

配置字段：`kv_load_failure_policy`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/kv_transfer.py:69`

可选：

- `recompute`
  - 重新调度请求，重算失败 KV blocks。
- `fail`
  - 请求立即以 error finished。

Scheduler 中对应字段：

- `self.recompute_kv_load_failures`

初始化位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:128`。

## 7. ECTransferConfig

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py:16`

EC 表示 encoder cache transfer。

### 7.1 字段

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

### 7.2 角色

- `ec_producer`
- `ec_consumer`
- `ec_both`

属性：

- `is_ec_transfer_instance`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py:94`
- `is_ec_producer`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py:98`
- `is_ec_consumer`：`D:/lzy/project/kv_pool/code/vllm/vllm/config/ec_transfer.py:102`

### 7.3 初始化

GPU Worker 的 `init_device()` 中，在初始化 distributed/model parallel 时会接入 EC transfer：

导入位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:27`

调用发生在 `init_worker_distributed_environment()` 内部。

Scheduler 侧也可能创建 EC connector：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:157`

EC connector 主要服务 encoder-decoder / multimodal 场景下 encoder cache 的跨实例传输。

## 8. Weight Transfer

相关导入在 GPU Worker：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:46`

### 8.1 创建时机

`Worker.load_model()` 中，如果 `weight_transfer_config` 非空，会创建：

```text
self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(...)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:358`。

### 8.2 初始化 transfer engine

方法：`init_weight_transfer_engine()`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1003`

作用：根据外部传入 init info 初始化传输机制。NCCL backend 下通常会与 trainer 建 process group。

### 8.3 权重更新生命周期

#### start_weight_update()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1017`

开始一次权重更新 session。

如果是 checkpoint format，会初始化 layerwise reload。

#### update_weights()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1046`

接收一个权重更新 chunk。

支持：

- checkpoint format dense weights。
- sparse_flat 权重更新。
- kernel format 直接 copy。

NCCL broadcast/packed path 可能是异步的，因此最后会 `torch.accelerator.synchronize()`，确保下一步使用新权重。

#### finish_weight_update()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:1120`

结束当前权重更新 session。如果是 checkpoint format，会 finalize layerwise reload。

## 9. Elastic EP

配置字段：`enable_elastic_ep`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:205`

GPU Worker 构造时创建：

```text
ElasticEPScalingExecutor(self)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:138`。

### 9.1 distributed init 中的 Elastic EP

`init_distributed_environment()` 中，如果 `enable_elastic_ep=True`，调用：

```text
_init_elastic_ep_world(config, local_rank, backend, rank, world_size)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1640`。

### 9.2 model parallel group 中的 Elastic EP

`initialize_model_parallel()` 中，Elastic EP 下：

- local ranks 使用 `local_all_ranks`。
- DP/EP/EPLB group 使用 `_init_stateless_group()`。
- coord store 来自 `get_cached_tcp_store_client(...)`。

相关位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1714`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1837`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1865`

### 9.3 限制

Elastic EP 当前不支持 multi-node TP/PP。

校验位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1620`。

如果 TP/PP group 跨多节点，会报错。

## 10. EPLB

EPLB 是 expert parallel load balancing。

### 10.1 配置

- `enable_eplb`
- `eplb_config`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/config/parallel.py:171`。

### 10.2 group

EPLB group 使用与 EP group 相同 rank 集合，但独立 process group。

创建位置：`D:/lzy/project/kv_pool/code/vllm/vllm/distributed/parallel_state.py:1878`。

原因：隔离 EPLB 通信与 MoE forward pass collectives，防止死锁。

### 10.3 worker 接入

GPU Worker 中导入：

- `override_envs_for_eplb`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_worker.py:31`

相关实现目录：

- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/eplb/**`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/eplb_utils.py`

## 11. Transfer 层与 Scheduler/Worker 的边界

### 11.1 Scheduler 负责

- 根据 request 和 cache 状态决定是否需要外部 KV/EC。
- 查询 connector prefix match。
- 分配本地接收 blocks。
- 把 connector metadata 放入 `SchedulerOutput`。
- 根据 worker 返回的 transfer output 更新状态。

### 11.2 Worker 负责

- 注册真实 KV/EC cache tensors。
- 执行 load/save/copy。
- 汇报完成和失败。
- 接收/应用 weight updates。
- 维护 NCCL 或 connector 内部通信上下文。

### 11.3 Distributed runtime 负责

- 提供 process group。
- 提供 TCPStore / NCCL / Gloo backend。
- 提供 group 销毁和资源清理。

## 12. 一句话总结

KV/EC/Weight transfer 和 Elastic EP/EPLB 是 vLLM 分布式层的扩展通信通道：它们不替代 TP/PP/DP/EP 的基础通信，而是在 worker 初始化、scheduler 调度、model runner forward 前后插入专用协议，用于跨实例缓存复用、encoder cache 传输、权重热更新和 MoE expert 负载管理。
