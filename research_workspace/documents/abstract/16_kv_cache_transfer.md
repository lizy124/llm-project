# 16 kv_cache_transfer 背诵文档

## 1. 专题定位

`kv_cache_transfer` 讲的是 vLLM V1 中本地 KV cache、prefix cache、外部 KV cache / KVPool、KV Connector、KV load/save、invalid blocks、deferred free 如何协同。

它是高级部署和跨实例 KV 复用专题。

一句话：

```text
Scheduler 侧决定哪些 KV 可用、哪些 block 要分配、哪些 transfer 要执行；Worker / ModelRunner 侧负责把 KV 真正 load/save 到本地 tensor；Scheduler 再根据 Worker 回执推进请求和释放资源。
```

## 2. 最小心智模型

主链路：

```text
Request
  → Scheduler.schedule()
  → KVCacheManager.get_computed_blocks()
  → KV Connector get_num_new_matched_tokens()
  → KVCacheManager.allocate_slots()
  → KV Connector update_state_after_alloc()
  → SchedulerOutput.kv_connector_metadata
  → Executor / Worker / ModelRunner
  → Worker connector load / save KV
  → ModelRunnerOutput.kv_connector_output
  → Scheduler.update_from_output()
  → finished_recving / finished_sending / invalid_block_ids
  → cache_blocks / recompute / fail / free / deferred free
```

KVPool 视角：

```text
KVPool hit
  → Scheduler 为 hit tokens 分配本地 KV blocks
  → Worker 从 KVPool 把 KV 写入这些 blocks
  → 请求继续 forward 或等待 async load 完成
  → 请求结束后 connector 决定是否保存 KV
  → save 完成后 Scheduler 才释放受保护 blocks
```

## 3. 核心角色总览

```text
KVCacheManager：Scheduler 侧 KV block 管理入口。
BlockPool：KVCacheBlock 元数据资源池。
KVCacheCoordinator：多 group / 多 attention 类型布局协调。
Scheduler 侧 KV Connector：计划外部 KV transfer。
Worker 侧 KV Connector：执行真实 KV load/save。
Scheduler：把 KV 命中、block 分配、请求状态和 Worker 回执串成闭环。
Worker / ModelRunner：维护真实 GPU KV cache tensor，构造 block table / slot mapping / attention metadata。
```

## 4. KVCacheManager

`KVCacheManager` 是 Scheduler 侧的 KV block 管理入口。

它负责：

```text
创建并持有 KV cache coordinator
查询本地 prefix cache：get_computed_blocks()
分配 block：allocate_slots()
把 local prefix hit、external KV hit、new tokens、lookahead tokens 统一放进 block 布局
维护 request_id → KVCacheBlock 列表
提供 block ids 给 SchedulerOutput / KV Connector / Worker
请求结束、抢占、reset、deferred free 时释放或弹出 blocks
prefix caching 开启时把 full blocks 写入 prefix cache
提供 usage、prefix cache stats、KV cache events
```

它不负责：

```text
决定本轮调度哪些请求
决定 token budget
执行 GPU KV tensor 真实读写
构造 attention metadata / slot mapping
执行外部 KV transfer 协议
处理 sampled token / RequestOutput
```

一句话：

```text
KVCacheManager 把调度语义翻译成 KV block 账本。
```

## 5. BlockPool

`BlockPool` 管 Scheduler 侧 KV block 元数据。

它管理：

```text
KVCacheBlock.block_id
KVCacheBlock.ref_cnt
KVCacheBlock.block_hash
free list 链表指针
null block 标记
prefix cache hash → block 映射
```

它不管理：

```text
GPU 上真实 key/value tensor 内容。
```

一句话：

```text
BlockPool 管哪些 block id 被谁占用、能否复用、能否驱逐；Worker 管这些 block id 对应的 GPU KV cache 槽位里写了什么。
```

## 6. KVCacheCoordinator

Coordinator 处理不同 KV cache 类型的差异。

它要适配：

```text
full attention
sliding window attention
chunked local attention
hybrid KV cache
Mamba / attention 混合
多 KV cache group
encoder-decoder cross-attention blocks
DCP / PCP
EAGLE / speculative lookahead
```

一句话：

```text
KVCacheManager 是 Scheduler API；Coordinator 处理多 group / 多 attention 类型；BlockPool 是底层 block 对象池。
```

## 7. Scheduler 侧 KV Connector

Scheduler 侧 connector 是外部 KV cache / KVPool 在调度阶段的代理。

它负责：

```text
on_new_request() 初始化 connector 请求状态
get_num_new_matched_tokens() 查询外部 KV 命中
告诉 Scheduler 是否需要 async load
update_state_after_alloc() 在本地 block 分配后记录 load / save 计划
build_connector_meta() 构造 SchedulerOutput.kv_connector_metadata
request_finished() 决定请求结束后是否保存 KV
update_connector_output() 消费 Worker 回传的完成 / 失败 / worker_meta
```

一句话：

```text
Scheduler 侧 connector 负责计划 KV transfer。
```

## 8. Worker 侧 KV Connector

Worker 侧 connector 是真实 KV 传输执行者。

它负责：

```text
创建 role=WORKER 的 connector
KV cache 初始化后注册本地 paged KV cache tensor
每轮 execute_model 前绑定 Scheduler 下发的 kv_connector_metadata
forward 前 start_load_kv()
attention layer entry wait_for_layer_load()
attention layer exit save_kv_layer()
forward 后 wait_for_save()
get_finished() 回传 finished_recving / finished_sending
get_block_ids_with_load_errors() 回传 invalid_block_ids
build_connector_worker_meta() 回传自定义状态
```

一句话：

```text
Worker 侧 connector 负责执行 KV transfer。
```

## 9. Scheduler 的中心作用

Scheduler 负责把所有 KV 状态串起来：

```text
管理 waiting / running / skipped_waiting
查询本地 prefix cache 和外部 KV hit
计算 num_computed_tokens / num_new_tokens
调用 KVCacheManager.allocate_slots()
把 block ids、scheduled tokens、connector metadata 写入 SchedulerOutput
在 update_from_output() 中消费 sampled tokens 和 kv_connector_output
恢复 WAITING_FOR_REMOTE_KVS
处理 invalid blocks
请求结束时调用 connector request_finished
释放 / 延迟释放 blocks
```

一句话：

```text
Scheduler 是 KV transfer 的请求状态机中心。
```

## 10. Worker / ModelRunner 的 KV 职责

Worker / ModelRunner 负责设备侧：

```text
初始化真实 GPU KV cache tensor
接收 SchedulerOutput 中的 block ids
更新 InputBatch.block_table
构造 positions / slot mapping / attention metadata
让 attention backend 读写 KV
根据 new_block_ids_to_zero 清零新 block
执行 KV connector load / save / finalize
返回 ModelRunnerOutput.kv_connector_output
```

一句话：

```text
Scheduler 管 KV block 账本；Worker 管 KV block 内容。
```

## 11. 初始化阶段

Scheduler 初始化时，如果配置了 KV transfer：

```text
Scheduler.__init__()
  → KVConnectorFactory.create_connector(role=SCHEDULER)
  → KVCacheManager(...)
  → connector.bind_gpu_block_pool(block_pool)
  → 设置 kv_load_failure_policy
  → 设置 defer_block_free
```

Worker 初始化时：

```text
GPUWorker.initialize_from_config(kv_cache_config)
  → ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  → KVConnectorFactory.create_connector(role=WORKER)
  → GPUModelRunner.initialize_kv_cache(kv_cache_config)
  → register_kv_caches(...)
```

关键边界：

```text
Scheduler 侧 connector 只参与计划；Worker 侧 connector 才拿到真实 KV cache tensor。
```

## 12. 新请求进入 Scheduler

新请求进入：

```text
Scheduler.add_request(request)
  → _enqueue_waiting_request(request)
  → self.requests[request_id] = request
  → connector.on_new_request(request)
```

connector 可以读取：

```text
request.kv_transfer_params
block hashes
external load spec
```

提前建立内部状态。

## 13. waiting 请求：先查本地 prefix cache

waiting 请求首次进入运行前：

```text
KVCacheManager.get_computed_blocks(request)
  → coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
  → BlockPool.get_cached_block(block_hash)
  → new_computed_blocks + num_new_local_computed_tokens
```

为什么最多命中 `request.num_tokens - 1`：

```text
即使 prompt 全命中，通常也要重算最后 token 来获得 next-token logits。
```

一句话：

```text
full prompt cached 不等于本轮完全 0 token forward。
```

## 14. waiting 请求：再查外部 KV

如果有 connector：

```text
connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
  → ext_tokens, load_kv_async
```

`ext_tokens` 表示：

```text
本地 prefix cache 已命中之外，外部 KV 还能新增命中的 token 数。
```

返回语义：

```text
(0, False)：没有外部命中。
(N, False)：外部新增命中 N tokens，不进入 WAITING_FOR_REMOTE_KVS。
(N, True)：外部新增命中 N tokens，需要 step 间异步 load。
(None, bool)：暂时无法判断，本轮跳过请求。
```

## 15. computed tokens 合并

Scheduler 合并：

```text
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

如果 `load_kv_async=False`：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
```

如果 `load_kv_async=True`：

```text
num_new_tokens = 0
```

因为本轮只为外部 KV 分配本地落点并发起 load，不做本地 model forward。

## 16. allocate_slots 的布局

`allocate_slots()` 的逻辑布局：

```text
---------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
---------------------------------------------------------------------
                                         |   < to be computed >     |
---------------------------------------------------------------------
                         |            < to be allocated >           |
---------------------------------------------------------------------
```

各段含义：

```text
comp：之前已经计算并持有的 tokens。
new_comp：本轮新查到的本地 prefix cache hit。
ext_comp：本轮外部 KV hit，需要 load 到本地 blocks。
new：本轮需要本地 forward 计算的 tokens。
lookahead：spec decode / EAGLE 预留 tokens。
```

重要点：

```text
external computed tokens 不需要本地 forward，但仍需要本地 KV blocks 作为 Worker load 的目标。
```

## 17. update_state_after_alloc

block 分配成功后：

```text
connector.update_state_after_alloc(
  request,
  kv_cache_manager.get_blocks(request_id),
  num_external_computed_tokens,
)
```

这一步回答：

```text
外部 KV 要写到本地哪些 block ids。
```

区别：

```text
get_num_new_matched_tokens() 回答“外部有多少”。
update_state_after_alloc() 回答“这些外部 KV 要写到哪里”。
```

## 18. load_kv_async 的两个分支

### load_kv_async=False

```text
request 进入 running。
SchedulerOutput 携带 block ids 和 kv_connector_metadata。
Worker 本轮可以 load external KV 并 forward 剩余 tokens。
```

### load_kv_async=True

```text
request.status = WAITING_FOR_REMOTE_KVS。
request 放入 skipped_waiting。
request.num_computed_tokens 先设置为 local + external 的预期进度。
本轮不进入 running，不 forward。
SchedulerOutput 仍携带 kv_connector_metadata，让 Worker 发起 load。
```

关键：

```text
WAITING_FOR_REMOTE_KVS 下的 num_computed_tokens 是 load 完成后的预期可用进度。
```

## 19. SchedulerOutput 下发什么

KV transfer 相关内容：

```text
block ids：Worker 构造 block table / slot mapping。
num_computed_tokens：Worker 知道哪些 prefix 已可用。
kv_connector_metadata：Worker connector 知道本轮 load / save 计划。
finished_req_ids：Worker connector 推进 finished_sending / finished_recving。
new_block_ids_to_zero：Worker 清零新分配 KV blocks。
```

## 20. Worker execute_model 中 connector 插入点

普通执行链路：

```text
_update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → logits / pooling / sample_tokens
```

KV connector 插入：

```text
forward 前：bind_connector_metadata + start_load_kv
attention layer entry：wait_for_layer_load(layer_name)
attention layer exit：save_kv_layer(layer_name, kv_cache, attn_metadata)
forward 后：wait_for_save + get_finished + get_block_ids_with_load_errors
```

一句话：

```text
Worker connector 生命周期包住一次模型执行，在 forward 前启动 load，在 attention 层边界等待和保存，在 forward 后回传完成 / 失败状态。
```

## 21. attention layer 边界为什么重要

KV load / save 放在 attention layer 边界，是因为：

```text
每层有自己的 KV cache。
async load 可以边执行前面层，边加载后面层。
某层用 KV 前必须确保该层 load 完。
save 需要 layer_name、kv_cache tensor、attention metadata。
不同 connector 可以只加载 / 保存部分层。
```

## 22. 0-token step 也可能有 connector 工作

如果本轮没有 scheduled tokens：

```text
没有 KV transfer：返回 EMPTY_MODEL_RUNNER_OUTPUT。
有 KV transfer：执行 kv_connector_no_forward。
```

用途：

```text
推进异步 KV load finished_recving。
推进异步 KV save finished_sending。
回传 connector stats / events / worker_meta。
消费 KV transfer metadata。
```

重要点：

```text
execute_model 不一定等于模型 forward，它也可能只是驱动 KV connector transfer。
```

## 23. KVConnectorOutput

Worker 回传：

```text
ModelRunnerOutput.kv_connector_output
```

包含：

```text
finished_sending：异步 save / send 完成的 request ids。
finished_recving：异步 load / recv 完成的 request ids。
invalid_block_ids：load 失败或不可信的本地 block ids。
kv_connector_stats：connector 统计。
kv_cache_events：connector 产生的 KV cache events。
kv_connector_worker_meta：自定义 worker → scheduler metadata。
expected_finished_count：多 worker 聚合计数。
```

多 worker 场景下需要聚合这些输出。

## 24. finished_recving：恢复远端 KV load 请求

Worker 返回：

```text
finished_recving = {req_id, ...}
```

Scheduler 记录后，后续 schedule 中：

```text
_try_promote_blocked_waiting_request(request)
  → 如果 req_id 已 finished_recving
  → _update_waiting_for_remote_kv(request)
  → status = WAITING 或 PREEMPTED
  → 继续普通 waiting 调度逻辑
```

成功时：

```text
kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
如果 full prompt 都来自外部 KV，回退最后 token 用于 logits
```

一句话：

```text
finished_recving 表示远端 KV 已 load 到本地 blocks，这些 blocks 可以进入本地 prefix cache 和后续调度。
```

## 25. finished_sending：释放延迟保护 blocks

请求结束时：

```text
connector.request_finished(request, block_ids)
```

如果 connector 要保存 KV：

```text
delay_free_blocks=True
```

Scheduler 不立即释放 blocks，而是等待 Worker 回传：

```text
finished_sending = {req_id}
```

收到后：

```text
_free_blocks(request)
```

一句话：

```text
request_finished 是“要不要保存 KV”的决策点；finished_sending 是“保存完成，可以释放 blocks”的回执。
```

## 26. invalid_block_ids

Worker connector load 失败时回传：

```text
invalid_block_ids
```

这些是：

```text
本地 GPU KV block id
```

不是：

```text
远端 block id
token id
request id
hash key
```

Scheduler 处理：

```text
invalid block id
  → 找到引用这些 block 的请求
  → 定位第一个失败 block 的 block index
  → request.num_computed_tokens 回退到 block_index * block_size
  → 根据 kv_load_failure_policy：recompute 或 fail
```

## 27. recompute 和 fail

### recompute

```text
失败 block 及其后续 tokens 不可信。
回退 num_computed_tokens。
后续重新本地 forward 计算。
```

### fail

```text
请求标记 FINISHED_ERROR。
释放资源。
```

为什么回退：

```text
num_computed_tokens 表示该请求已有多少 token 的 KV 可以被信任复用。
失败 block 之后不能信任。
```

## 28. sync load 和 async load 失败差异

### sync load 失败

```text
请求已经在 running。
本轮 update_from_output 直接处理。
recompute 下跳过本轮输出更新，下一轮按回退进度重算。
fail 下请求 FINISHED_ERROR。
```

### async load 失败

```text
请求在 skipped_waiting / WAITING_FOR_REMOTE_KVS。
invalid blocks 先回退 num_computed_tokens。
recompute 下记录 failed_recving_kv_req_ids。
仍等待 finished_recving。
promote 时 cache 有效前缀或释放 blocks。
```

## 29. BlockPool 生命周期

BlockPool 初始化：

```text
创建 KVCacheBlock[0..N-1]
初始化 free_block_queue
创建 cached_block_hash_to_block
从 free queue 弹出一个 block 作为 null_block
```

null_block 用于：

```text
sliding window 左侧滑出窗口 blocks
chunked local attention 当前 chunk 外 blocks
Mamba align 不需要的 state block
prefix hit / hybrid 布局空洞占位
padding / 特殊占位
```

新分配：

```text
从 free_block_queue 弹出 blocks
如果 blocks 有旧 block_hash，先 evict cached block
设置 ref_cnt
加入 request block list
```

释放：

```text
ref_cnt 减少
如果为 0，回到 free queue 或作为 prefix cache 可复用 block
```

## 30. deferred free

deferred free 用于避免异步竞态。

场景：

```text
async scheduling
pipeline parallel
KV consumer / connector 还可能使用 block
Worker 尚未确认 save 完成
```

目的：

```text
防止 Scheduler 过早释放 block，导致另一个请求复用该 block，而旧的异步操作还在读写它。
```

一句话：

```text
deferred free 是 KV block 复用前的安全栅栏。
```

## 31. 常见易混点

### KVCacheManager 不存 GPU tensor

```text
它管 block 账本；GPU tensor 在 Worker / ModelRunner。
```

### BlockPool 的 block 是元数据

```text
KVCacheBlock 代表 block id、ref_cnt、hash，不是 key/value tensor 本身。
```

### external KV hit 仍要分配本地 block

```text
外部 KV 不需要 forward，但必须有本地 block 作为 load 目标。
```

### load_kv_async=True 不 forward

```text
本轮只是发起 load，请求进入 WAITING_FOR_REMOTE_KVS。
```

### invalid_block_ids 是本地 block id

```text
不是远端 id，也不是 request id。
```

## 32. 与其他专题的关系

```text
scheduler：KV hit、block 分配、WAITING_FOR_REMOTE_KVS、invalid blocks、free 都在 Scheduler 状态机中。
attention：attention layer 读写 KV cache，并在 layer 边界接 KV connector hook。
executor_worker_model_runner：Worker / ModelRunner 消费 kv_connector_metadata 并返回 kv_connector_output。
parallelism：TP / PP / DP 下 KV cache 归属不同，connector 需要适配多 rank。
spec_decode：lookahead tokens 会进入 KV block allocation。
multimodal：EC connector / encoder cache 是类似但独立的 encoder 输出 transfer 线。
compilation_and_cuda_graph：KV connector / dynamic KV load 可能影响 graph eligibility。
```

## 33. 背诵总结

背这一段：

```text
vLLM V1 的 KV cache transfer 是 Scheduler 侧计划和 Worker 侧执行的闭环。Scheduler 先通过 KVCacheManager 查询本地 prefix cache，再通过 Scheduler 侧 KV Connector 查询外部 KV 命中；命中后仍要通过 allocate_slots 为这些 external computed tokens 分配本地 KV blocks，并调用 update_state_after_alloc 告诉 connector 外部 KV 要写到哪些 block。SchedulerOutput 把 block ids 和 kv_connector_metadata 下发给 Worker，Worker 侧 connector 在 forward 前启动 load，在 attention layer 边界等待 layer load 和保存 layer KV，forward 后回传 finished_recving、finished_sending 和 invalid_block_ids。Scheduler.update_from_output 根据 finished_recving 恢复 WAITING_FOR_REMOTE_KVS，根据 finished_sending 释放延迟保护 blocks，根据 invalid_block_ids 回退 num_computed_tokens 并选择 recompute 或 fail。核心边界是：Scheduler 管 KV block 账本和状态机，Worker 管真实 KV tensor 和传输执行。
```
