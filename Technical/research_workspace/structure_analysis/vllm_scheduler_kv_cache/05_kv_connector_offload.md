# 05. KV Connector 与 Offload / Remote KV

## 1. KV Connector 的位置

KV Connector 是 V1 中连接 scheduler、worker、远端 KV 存储或 P/D disaggregation 的机制。

它有两侧：

```text
Scheduler 侧 connector
  - 负责请求级状态、prefix match、load/save 计划、preemption 元数据

Worker 侧 connector
  - 负责绑定真实 KV cache tensor、执行 load/save、报告完成/失败/events/stats
```

相关文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/*`

## 2. Scheduler 侧 connector 初始化

在 `Scheduler.__init__()` 中，如果 `vllm_config.kv_transfer_config` 不为空，会创建 scheduler 侧 connector。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:123`。

关键逻辑：

```text
self.connector = KVConnectorFactory.create_connector(
    config=self.vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=self.kv_cache_config,
)
```

同时初始化：

- `self.connector_prefix_cache_stats`
- `self.recompute_kv_load_failures`
- `self.defer_block_free`

### 2.1 recompute_kv_load_failures

`kv_load_failure_policy == "recompute"` 时启用。

含义：

- 如果 remote KV load 失败，不直接报错。
- scheduler 会调整请求状态，让失败部分重新计算。

### 2.2 defer_block_free

如果同时满足：

- `max_concurrent_batches > 1`
- 当前是 KV consumer

则启用 `defer_block_free`。

原因：overlapping batches 或 async scheduling 下，一个 step 可能仍在写某请求 KV block；如果此时 block 被释放并重分配给 connector load，就可能产生未排序写冲突。

相关注释在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:145`。

## 3. Request 中的 KV transfer 参数

`Request` 字段：

```text
self.kv_transfer_params: dict[str, Any] | None = None
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:101`。

如果 `sampling_params.extra_args` 中带有 `kv_transfer_params`，会写入 request：

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:114`。

EngineCore 添加请求时，如果 request 带 KV transfer params 但 scheduler 没有 connector，会发 warning：

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:395`。

## 4. 新请求进入 connector

`Scheduler.add_request()` 中，新请求加入 `self.requests` 后：

```text
if self.connector is not None:
    self.connector.on_new_request(request)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1978`。

这让 connector 能在请求进入调度前初始化 transfer 相关状态。

## 5. Prefix hit 与 external computed tokens

waiting 请求首次调度时，scheduler 先查本地 prefix cache，再查 connector 外部命中。

本地命中：

```text
new_computed_blocks, num_new_local_computed_tokens = kv_cache_manager.get_computed_blocks(request)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:708`。

外部命中：

```text
ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:722`。

结果：

```text
num_external_computed_tokens = ext_tokens
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

## 6. load_kv_async 路径

如果 connector 返回 `load_kv_async=True`，说明需要远端 KV 异步加载。

调度器行为：

1. `num_new_tokens = 0`。
2. 调 `kv_cache_manager.allocate_slots(...)` 为 external computed tokens 分配本地接收 blocks。
3. `delay_cache_blocks=True`，因为这些 blocks 还没完成远程加载，不能立刻当作 prefix cache 提交。
4. 通过 `connector.update_state_after_alloc(...)` 通知 connector 本地 blocks。
5. 请求状态设为 `WAITING_FOR_REMOTE_KVS`。
6. 请求放入 `skipped_waiting`。
7. 本轮不进入 running。

关键代码位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:781`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:873`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:916`

## 7. reserved_blocks 防止 async KV load 死锁

在 async remote KV load 时，scheduler 计算：

```text
reserved_blocks = self._inflight_prefill_reserved_blocks()
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:865`。

原因：async load 会持有 blocks 一段时间，但没有 forward progress，且这里不可抢占。如果它占光 free blocks，其他正在 prefill 的请求可能无法完成，造成死锁或可预测的频繁抢占。

`KVCacheManager.allocate_slots()` 支持 `reserved_blocks` 参数，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:255`。

容量检查：

```text
available_blocks = block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:414`。

## 8. delay_cache_blocks

`delay_cache_blocks` 参数定义在 `KVCacheManager.allocate_slots()`，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:252`。

用途：P/D 或 remote KV load 时，分配出来的 blocks 还没有真实内容，不能立即写入 prefix cache。

如果：

```text
not enable_caching or delay_cache_blocks
```

则 `allocate_slots()` 分配后直接返回，不调用 `coordinator.cache_blocks(...)`。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:442`。

## 9. connector.update_state_after_alloc()

waiting 请求成功分配 slots 后，如果 connector 存在，会调用：

```text
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:900`。

它告诉 scheduler 侧 connector：

- 请求对应的本地 blocks 是哪些。
- 有多少 tokens 是 external computed。
- 是否需要安排 load/save。

## 10. SchedulerOutput 中的 connector metadata

调度尾部，如果 connector 存在，会调用：

```text
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1080`。

`_build_kv_connector_meta()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1100`。

这个 metadata 是一个 opaque object，对 scheduler output 来说只负责携带，worker connector 会解读。

## 11. Worker 侧 KVConnector

文件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`

### 11.1 接口

`KVConnector` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:29`。

方法：

- `pre_forward(scheduler_output)`
- `post_forward(finished_req_ids, wait_for_save=True)`
- `no_forward(scheduler_output)`
- `set_disabled(disabled)`

默认实现是 no-op。

### 11.2 ActiveKVConnector

定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:47`。

初始化时：

1. 获取全局 KV transfer group。
2. 注册 KV cache tensors：`register_kv_caches(kv_caches_dict)`。
3. 设置 host transfer copy 操作：`set_host_xfer_buffer_ops(copy_kv_blocks)`。

### 11.3 pre_forward()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:61`。

执行：

1. 读取 `scheduler_output.kv_connector_metadata`。
2. `handle_preemptions(kv_connector_metadata)`。
3. `bind_connector_metadata(kv_connector_metadata)`。
4. `start_load_kv(get_forward_context())`。

它会在模型 forward 前开始 load remote KV。

### 11.4 post_forward()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:77`。

执行：

1. 等待保存完成：`wait_for_save()`。
2. 获取 finished sending / receiving。
3. 获取 load error block ids。
4. 获取 connector stats。
5. 获取 KV cache events。
6. 构建 worker meta。
7. 清理 connector metadata。
8. 返回 `KVConnectorOutput`。

### 11.5 no_forward()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:98`。

当本轮没有 token 需要执行，但需要处理 KV transfer 时使用。

它会：

1. 调 `pre_forward()`。
2. 以 `wait_for_save=False` 调 `post_forward()`。
3. 返回只包含 KV connector output 的 `ModelRunnerOutput`。

## 12. KV load 失败处理

`Scheduler.update_from_output()` 开始处会检查：

```text
kv_connector_output.invalid_block_ids
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1490`。

如果存在 invalid blocks：

```text
failed_kv_load_req_ids = self._handle_invalid_blocks(...)
```

然后在 per-request 循环中跳过这些请求，避免用坏的 KV 输出推进状态。

如果 `recompute_kv_load_failures=False`，则这些请求会被标记为 error finished：

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1717`。

如果启用 recompute 策略，则 scheduler 会调整这些请求的 computed token 状态，后续重新计算失败部分。

## 13. KV transfer 完成处理

`Scheduler.update_from_output()` 末尾会调用：

```text
self._update_from_kv_xfer_finished(kv_connector_output)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1731`。

它负责：

- 记录 finished sending。
- 记录 finished receiving。
- 将 `WAITING_FOR_REMOTE_KVS` 请求恢复为可调度。
- 对成功加载的 tokens 调整缓存状态。
- 对失败加载根据策略重算或报错。

相关辅助方法：

- `_update_waiting_for_remote_kv()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2350`
- `_try_promote_blocked_waiting_request()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2384`
- `_update_from_kv_xfer_finished()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2417`
- `_update_requests_with_invalid_blocks()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2446`
- `_handle_invalid_blocks()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2549`

## 14. Preemption 与 connector

当请求被抢占时，scheduler 调 `_preempt_request()`。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1105`。

它会释放本地 KV blocks，然后将请求放回 waiting。

同时在调度输出构建 connector metadata 时，connector 会把 preemption 信息传给 worker，worker 侧 `ActiveKVConnector.pre_forward()` 调 `handle_preemptions()`。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:67`。

## 15. Request finished 与 connector

请求完成释放时，`_free_request()` 会先调用 `_connector_finished(request)`。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2052`。

这允许 connector：

- 判断是否需要延迟释放 blocks。
- 返回需要回传给前端的 `kv_xfer_params`。
- 清理 connector 侧请求状态。

如果 connector 要求 delay free，则 `_free_request()` 不会立即 `_free_blocks()`。

## 16. KV cache events 与 stats

`Scheduler.update_from_output()` 会收集：

1. worker connector stats。
2. scheduler connector stats。
3. KV cache manager events。
4. connector events。

位置：

- stats 聚合：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1735`
- cache events：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1752`
- publish：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1764`

## 17. Offload / disaggregation 的核心不变量

1. 外部 KV 命中只表示“内容存在”，本地仍需要 block 接收或引用。
2. async load 的 blocks 在 load 完成前不能进入 prefix cache。
3. remote KV load 占用 block 但可能没有 forward progress，因此需要 reserved blocks 避免死锁。
4. overlapping batches 下 block 释放必须 fenced，避免 GPU 写未完成就复用。
5. load 失败不能让请求继续使用坏 KV；要么 recompute，要么 error finish。

## 18. 一句话总结

KV Connector 把“本地 prefix cache”扩展成“本地 + 远端 KV cache”体系：scheduler 决定哪些 token 可视为 external computed 并分配本地接收 blocks，worker connector 执行真实 load/save，再把完成、失败、事件和统计反馈给 scheduler。
