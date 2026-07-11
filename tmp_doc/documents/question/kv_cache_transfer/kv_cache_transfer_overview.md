# vLLM V1 KV Cache / KV Transfer / KVPool 逻辑梳理

源码位置：

- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/distributed/kv_transfer/`
- `code/vllm/vllm/distributed/ec_transfer/`

本文按“先定角色，再走主链路，再拆 load / save / failure / free”的方式，梳理 vLLM V1 中 KV Cache、Prefix Cache、KV Connector、外部 KV Cache / KVPool、KV load / save、invalid blocks、deferred free 等逻辑。

它不是逐个函数的字典式说明，而是先回答几组最关键的问题：

```text
1. KVCacheManager、BlockPool、Scheduler、Worker / ModelRunner、KV Connector 各自负责什么？
2. 本地 prefix cache 如何命中，命中后如何减少本轮 prefill token？
3. 外部 KV Cache / KVPool 如何接入 waiting 请求调度？
4. external KV hit 如何转成本地 block 分配和 Worker load metadata？
5. load_kv_async=True 时为什么请求进入 WAITING_FOR_REMOTE_KVS？
6. Worker / ModelRunner 如何执行 KV load / save，并把结果回传 Scheduler？
7. 请求结束后 external KV save 如何延迟释放 blocks？
8. invalid_block_ids 如何触发 recompute 或 fail？
9. deferred free 如何避免 async scheduling / PP / KV consumer 场景下的 block 复用竞态？
10. 从 KVPool 视角，一个请求端到端如何贯穿 Scheduler 和 Worker？
```

---

## 0. 梳理规划

本文后续按“角色边界 → 主链路 → 关键阶段 → 异常与释放 → 阅读路线”的顺序组织。

要回答的问题分成 10 组：

```text
1. KVCacheManager 是哪一层？负责什么？
2. BlockPool 和 KV block 生命周期如何工作？
3. 本地 prefix cache 如何命中？
4. Scheduler 侧如何接入 KV Connector？
5. 外部 KV Cache 如何 load 回本地？
6. 请求结束后 KV 如何 save 到外部 KVPool？
7. Worker / ModelRunner 如何消费 KV Connector metadata？
8. invalid blocks 如何触发 recompute 或 fail？
9. deferred free 如何保证异步安全？
10. KVPool 端到端链路如何贯穿 Scheduler 和 Worker？
```

阅读顺序建议：

```text
kv_cache_transfer_overview.md
  → 01_kv_cache_manager_role.md
  → 02_block_pool_and_block_lifecycle.md
  → 03_prefix_cache_lookup.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 07_worker_kv_connector_flow.md
  → 06_external_kv_save_flow.md
  → 08_invalid_blocks_and_recompute.md
  → 09_deferred_free_and_async_safety.md
  → 10_kvpool_end_to_end.md
```

如果只想先抓一条主线，可以先读总览，再读 `01`、`04`、`05`、`07`。

---

## 1. 一句话总览

vLLM V1 的 KV Cache / KV Transfer 可以压缩成一句话：

```text
Scheduler 侧决定“哪些 KV 已经可用、哪些 block 要分配、哪些 transfer 要执行”；Worker / ModelRunner 侧负责“把 KV 真正 load/save 到本地 tensor”；Scheduler 再根据 Worker 回传的完成或失败状态推进请求和释放资源。
```

最小主链路是：

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

如果再从 KVPool 视角压缩：

```text
KVPool hit
  → Scheduler 为 hit tokens 分配本地 KV blocks
  → Worker 从 KVPool 把 KV 写入这些 blocks
  → sync load：本轮直接进入 running，按普通路径 cache / forward
  → async load：Scheduler 收到 finished_recving 后承认这些 blocks 可复用
  → 请求结束后 connector 可决定是否保存 / 发送这些 blocks
  → 若 request_finished*() 要求 delay_free_blocks，则 finished_sending 后才安全释放 blocks
```

---

## 2. 核心角色

### 2.1 KVCacheManager

`KVCacheManager` 是 Scheduler 侧的 KV block 管理入口。

源码位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:110`

它负责：

```text
- 创建并持有 KV cache coordinator；
- 暴露本地 prefix cache 查询接口 get_computed_blocks()；
- 暴露 block 分配接口 allocate_slots()；
- 把 local prefix hit、external KV hit、本轮 new tokens、lookahead tokens 统一放进 block 布局；
- 根据 request 进度维护 request_id → KVCacheBlock 列表；
- 提供 block ids 给 SchedulerOutput / KV Connector / Worker；
- 在请求结束、抢占、reset、deferred free 场景中释放或弹出 blocks；
- 在 prefix caching 开启时把 full blocks 写入 prefix cache；
- 提供 usage、prefix cache stats、KV cache events。
```

它不负责：

```text
- 不决定本轮调度哪些请求；
- 不决定 token budget；
- 不执行 GPU KV tensor 的真实读写；
- 不构造 attention metadata / slot mapping；
- 不执行外部 KV 传输协议；
- 不处理 sampled token、stop、RequestOutput。
```

一句话记忆：

```text
KVCacheManager 负责“把调度语义翻译成 KV block 账本”。
```

### 2.2 BlockPool

`BlockPool` 是 Scheduler 侧的 KV block 元数据资源池。

源码位置：`code/vllm/vllm/v1/core/block_pool.py:144`

它管理的是：

```text
KVCacheBlock 元数据：
  block_id
  ref_cnt
  block_hash
  free list 链表指针
  null block 标记
```

它不直接管理：

```text
GPU 上真实的 key/value tensor 内容。
```

它负责：

```text
- 创建所有 KVCacheBlock；
- 维护 free_block_queue；
- 维护 prefix cache hash → block 映射；
- 分配新 blocks；
- touch 命中的 cached blocks；
- free blocks；
- evict prefix cache blocks；
- reset prefix cache；
- 生成 KV cache events。
```

一句话记忆：

```text
BlockPool 管“哪些 block id 被谁占用、能否复用、能否驱逐”；Worker 管“这些 block id 对应的 GPU KV cache 槽位里写了什么”。
```

### 2.3 KVCacheCoordinator / SingleTypeKVCacheManager

`KVCacheCoordinator` 是 `KVCacheManager` 下面的布局协调层。

它负责把不同 KV cache 形态收敛到统一接口：

```text
- full attention；
- sliding window attention；
- chunked local attention；
- hybrid KV cache；
- Mamba / attention 混合；
- 多 KV cache group；
- encoder-decoder cross-attention blocks；
- DCP / PCP；
- EAGLE / speculative lookahead。
```

`SingleTypeKVCacheManager` 则维护每个 group 内的请求账本：

```text
req_to_blocks[request_id]      请求当前持有的 KVCacheBlock 列表
num_cached_block[request_id]   该请求已有多少 full block 进入 prefix cache
new_block_ids                  本轮新分配、需要 Worker zero 的 block ids
```

一句话记忆：

```text
KVCacheManager 是 Scheduler API；coordinator 处理多 group / 多 attention 类型差异；BlockPool 是底层 block 对象池。
```

### 2.4 Scheduler 侧 KV Connector

Scheduler 侧 KV Connector 是外部 KV Cache / KVPool 在调度阶段的代理。

它负责：

```text
- on_new_request() 初始化 connector 侧请求状态；
- get_num_new_matched_tokens() 查询外部 KV 命中；
- 告诉 Scheduler 这些外部命中是否需要 async load；
- update_state_after_alloc() 在本地 block 分配后记录 load / save 计划；
- build_connector_meta() 构造 SchedulerOutput.kv_connector_metadata；
- request_finished() / request_finished_all_groups() 决定请求结束后是否保存 KV；
- update_connector_output() 消费 Worker 侧回传的完成 / 失败 / worker_meta。
```

它不负责：

```text
- 本地 prefix cache 查询；
- 本地 block 分配；
- GPU KV tensor load / save；
- waiting / running / preemption 主状态机；
- attention metadata / slot mapping。
```

一句话记忆：

```text
Scheduler 侧 connector 负责“计划 KV transfer”。
```

### 2.5 Worker / ModelRunner 侧 KV Connector

Worker 侧 KV Connector 是真实 KV 传输执行者。

它负责：

```text
- 在 worker 初始化阶段创建 role=WORKER 的 connector；
- 在 KV cache 初始化完成后注册本地 paged KV cache tensor；
- 每轮 execute_model() 前绑定 Scheduler 下发的 kv_connector_metadata；
- 在 forward 前 start_load_kv()；
- 在 attention layer entry wait_for_layer_load()；
- 在 attention layer exit save_kv_layer()；
- forward 结束后 wait_for_save()；
- 通过 get_finished() 回传 finished_recving / finished_sending；
- 通过 get_block_ids_with_load_errors() 回传 invalid_block_ids；
- 通过 build_connector_worker_meta() 回传 connector 自定义状态。
```

一句话记忆：

```text
Worker 侧 connector 负责“执行 KV transfer”。
```

### 2.6 Scheduler

Scheduler 是整条链路的状态机中心。

它负责：

```text
- 管理 waiting / running / skipped_waiting 队列；
- 查询本地 prefix cache 和外部 KV hit；
- 计算 num_computed_tokens / num_new_tokens；
- 调用 KVCacheManager.allocate_slots()；
- 把 block ids、scheduled tokens、connector metadata 写入 SchedulerOutput；
- 在 update_from_output() 中消费 sampled tokens 和 kv_connector_output；
- 恢复 WAITING_FOR_REMOTE_KVS；
- 处理 invalid blocks；
- 请求结束时调用 connector request_finished；
- 释放 / 延迟释放 blocks。
```

一句话记忆：

```text
Scheduler 负责“把 KV 命中、block 分配、请求状态和 Worker 回执串成闭环”。
```

### 2.7 Worker / ModelRunner

Worker / ModelRunner 是设备侧执行层。

它负责：

```text
- 初始化真实 GPU KV cache tensor；
- 接收 SchedulerOutput 中的 block ids；
- 更新 InputBatch.block_table；
- 构造 positions / slot mapping / attention metadata；
- 在 forward 中让 attention backend 读写 KV；
- 根据 new_block_ids_to_zero 清零新 block；
- 执行 KV connector load / save / finalize；
- 返回 ModelRunnerOutput.kv_connector_output。
```

一句话记忆：

```text
Scheduler 管 KV block 账本；Worker 管 KV block 内容。
```

---

## 3. 这几层为什么要分开

### 3.1 Scheduler 适合放“决策和状态机”

Scheduler 要处理：

```text
- waiting / running / skipped_waiting；
- token budget；
- prefix cache hit；
- external KV hit；
- block 分配失败后的 preemption；
- WAITING_FOR_REMOTE_KVS 恢复；
- invalid blocks 的 recompute / fail；
- finished_sending 后释放；
- deferred free fence。
```

这些都是请求级状态机，不适合放在 Worker 的模型执行代码里。

### 3.2 KVCacheManager / BlockPool 适合放“block 账本”

block 账本需要统一处理：

```text
- block_id；
- ref_cnt；
- free queue；
- prefix cache hash map；
- null block；
- multi KV group；
- sliding window / hybrid / mamba；
- cache events / metrics。
```

这些细节如果直接塞进 Scheduler，会让调度逻辑和 block 布局耦合过深。

### 3.3 Connector 适合放“外部 KV 协议”

不同外部 KV 后端差异很大：

```text
- NIXL pull / push；
- LMCache；
- Offloading；
- Mooncake / MooncakeStore；
- FlexKV；
- MultiConnector；
- 自定义 connector。
```

Scheduler 不应该知道每个后端如何 RDMA、如何建 job、如何 handshake、如何写外部 store。

统一抽象是：

```text
Scheduler connector：查命中、构造 metadata、消费 output。
Worker connector：绑定 metadata、执行 load/save、回传 output。
```

### 3.4 Worker / ModelRunner 适合放“tensor 和 attention 时序”

Worker 侧才能看到：

```text
- 每层 KV cache tensor；
- attention metadata；
- slot mapping；
- forward context；
- layer_name；
- 当前 batch 的真实 tensor 布局。
```

因此真实 KV load / save 必须在 Worker / ModelRunner 侧做。

---

## 4. 主链路：从请求到 Worker 执行

### 4.1 初始化阶段

Scheduler 初始化时，如果配置了 `kv_transfer_config`：

```text
Scheduler.__init__()
  → KVConnectorFactory.create_connector(role=SCHEDULER)
  → KVCacheManager(...)
  → connector.bind_gpu_block_pool(kv_cache_manager.block_pool)
  → 根据 kv_load_failure_policy 设置 recompute / fail
  → 根据 KV consumer + max_concurrent_batches 设置 defer_block_free
```

Worker 初始化时，如果当前 worker 是 KV transfer instance：

```text
GPUWorker.initialize_from_config(kv_cache_config)
  → ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  → KVConnectorFactory.create_connector(role=WORKER)
  → GPUModelRunner.initialize_kv_cache(kv_cache_config)
  → register_kv_caches(...) / register_cross_layers_kv_cache(...)
```

注意新版 `v1/worker/gpu/kv_connector.py` 的 `ActiveKVConnector` 当前只调用 `register_kv_caches(kv_caches_dict)`，cross-layer KV cache 支持在该路径仍是 TODO；legacy GPU model runner 路径包含 `register_cross_layers_kv_cache(...)` 分支。

关键边界：

```text
Scheduler 侧 connector 只参与计划；Worker 侧 connector 才拿到真实 KV cache tensor。
```

### 4.2 新请求进入 Scheduler

新请求进入时：

```text
Scheduler.add_request(request)
  → _enqueue_waiting_request(request)
  → self.requests[request_id] = request
  → connector.on_new_request(request)
```

`on_new_request()` 默认可以是 no-op，但具体 connector 可以读取 `request.kv_transfer_params`、block hashes、外部 load spec 等信息，提前建立 connector 内部状态。

### 4.3 waiting 请求调度：先查本地 prefix cache

waiting 请求首次进入运行前，Scheduler 先查本地 prefix cache：

```text
KVCacheManager.get_computed_blocks(request)
  → max_cache_hit_length = request.num_tokens - 1
  → coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
  → BlockPool.get_cached_block(block_hash, kv_cache_group_ids)
  → new_computed_blocks + num_new_local_computed_tokens
```

为什么最多命中 `request.num_tokens - 1`：

```text
即使 prompt 全部命中，也通常要重算最后一个 token 来获得 next-token logits。
```

实际常表现为：

```text
full prompt cached ≠ 本轮 0 token forward；
通常会向下对齐到 block 边界，重算最后一个 block。
```

### 4.4 waiting 请求调度：再查外部 KV

如果有 connector，Scheduler 再查外部 KV：

```text
connector.get_num_new_matched_tokens(
  request,
  num_new_local_computed_tokens,
)
  → ext_tokens, load_kv_async
```

这里的 `ext_tokens` 不是外部总命中数，而是：

```text
本地 prefix cache 已命中 token 之外，外部 KV cache 还能新增命中的 token 数。
```

返回值语义：

```text
(0, False)：没有外部命中；
(N, False)：外部新增命中 N tokens，但不进入 WAITING_FOR_REMOTE_KVS；
(N, True)：外部新增命中 N tokens，需要 scheduler step 间异步 load；
(None, bool)：connector 暂时无法判断，Scheduler 本轮跳过该请求，后续再查。
```

### 4.5 合并 computed tokens

Scheduler 合并本地和外部命中：

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

因为本轮只做：

```text
为外部 KV 分配本地落地点 + 发起 Worker load；
不做本地 model forward。
```

### 4.6 allocate_slots：统一 block 布局

`KVCacheManager.allocate_slots()` 的核心布局是：

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
comp：request 之前已经计算并持有的 tokens；
new_comp：本轮新查到的本地 prefix cache hit tokens；
ext_comp：本轮外部 KV hit，需要 load 到本地 blocks；
new：本轮需要本地 forward 计算的 tokens；
lookahead：spec decode / EAGLE 预留 tokens。
```

注意：

```text
external computed tokens 不需要本地 forward；
但仍然需要分配本地 KV blocks，作为 Worker load 的目标。
```

`allocate_slots()` 三阶段：

```text
1. 释放当前 attention 不再需要的 skipped blocks，并检查容量；
2. 处理 prefix tokens：本地 prefix hit + external KV hit；
3. 为 new tokens + lookahead tokens 分配新 blocks。
```

### 4.7 update_state_after_alloc：让 connector 看到本地 block ids

block 分配成功后，Scheduler 调用：

```text
connector.update_state_after_alloc(
  request,
  kv_cache_manager.get_blocks(request_id),
  num_external_computed_tokens,
)
```

这一步必须在 `allocate_slots()` 之后，因为 connector 之前只知道：

```text
外部 KV 命中了多少 token。
```

此时才知道：

```text
这些外部 KV 应该 load 到本地哪些 block ids。
```

一句话记忆：

```text
get_num_new_matched_tokens() 回答“外部有多少”；update_state_after_alloc() 回答“外部 KV 要写到哪里”。
```

### 4.8 waiting 请求调度后的两个分支

如果 `load_kv_async=False`：

```text
request 进入 running；
SchedulerOutput.scheduled_new_reqs / scheduled_cached_reqs 携带 block ids 和 num_computed_tokens；
Worker 本轮可以 load external KV 并 forward 剩余 tokens。
```

如果 `load_kv_async=True`：

```text
request.status = WAITING_FOR_REMOTE_KVS；
request 放入 skipped_waiting；
request.num_computed_tokens 先设置为 local + external 的预期完成进度；
本轮不进入 running，不 forward；
SchedulerOutput 仍可携带 kv_connector_metadata，让 Worker 发起 load。
```

这里的关键点：

```text
WAITING_FOR_REMOTE_KVS 下的 request.num_computed_tokens 是“load 完成后预期可用的进度”，在 transfer 完成前不会被用于 forward。
```

### 4.9 SchedulerOutput 下发

SchedulerOutput 会携带：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
scheduled_encoder_inputs
num_common_prefix_blocks
finished_req_ids
free_encoder_mm_hashes
kv_connector_metadata
ec_connector_metadata
new_block_ids_to_zero
```

其中和 KV transfer 最相关的是：

```text
block ids：Worker 构造 block table / slot mapping；
num_computed_tokens：Worker 知道哪些 prefix 已经可用；
kv_connector_metadata：Worker connector 知道本轮 load / save 计划；
finished_req_ids：Worker connector 可据此推进 finished_sending / finished_recving；
new_block_ids_to_zero：Worker 清零新分配的 KV blocks。
```

---

## 5. Worker / ModelRunner 侧执行链路

### 5.1 execute_model 的 KV connector 插入点

Worker 收到 SchedulerOutput 后，最终进入：

```text
GPUModelRunner.execute_model(scheduler_output)
```

主线仍是执行层常规链路：

```text
_update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → logits / pooling / sample_tokens
```

KV connector 的核心插入点是 forward context 及其前后钩子：

```text
legacy:
  set_forward_context(attn_metadata, ...)
    + maybe_get_kv_connector_output(scheduler_output)
    → _model_forward(...)

新版:
  set_forward_context(attn_metadata, ...)
    → kv_connector.pre_forward(scheduler_output)
    → model forward
    → sample_tokens() / pool() 中 kv_connector.post_forward(finished_req_ids)
```

### 5.2 Worker connector 生命周期

legacy `maybe_get_kv_connector_output()` 进入 context 时：

```text
KVConnectorOutput()
  → get_kv_transfer_group()
  → bind_connector_metadata(scheduler_output.kv_connector_metadata)
  → start_load_kv(get_forward_context())
```

legacy context 退出时，或新版 `ActiveKVConnector.post_forward()` 中，普通 forward 路径通常是：

```text
wait_for_save()
  → get_finished(scheduler_output.finished_req_ids)
  → get_block_ids_with_load_errors()
  → get_kv_connector_stats()
  → get_kv_connector_kv_cache_events()
  → build_connector_worker_meta()
  → clear_connector_metadata()
```

新版 `ActiveKVConnector.pre_forward()` 则把 `handle_preemptions()`、`bind_connector_metadata()` 和 `start_load_kv()` 合并在 forward 前执行。

但这不是无条件顺序：0-token `kv_connector_no_forward()` 会传 `wait_for_save=False`；speculative decoding 下 `defer_finalize=True` 会延后 `wait_for_save()` 和 `clear_connector_metadata()`，但仍在 context 退出时收集 finished / invalid / stats / events / worker_meta。

这说明 Worker connector 生命周期包住一次模型执行：

```text
forward 前启动 load；
forward 中按 attention layer 等待 load / 触发 save；
forward 后收集 transfer 完成和失败状态。
```

### 5.3 attention layer 边界上的 load / save

attention 层通过 KV transfer wrapper 接入：

```text
attention layer entry:
  connector.wait_for_layer_load(layer_name)

attention forward:
  attention backend 读写本地 KV cache tensor

attention layer exit:
  connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

为什么要在层边界做：

```text
- async load 可以边 forward 前面层，边加载后面层；
- 每层使用 KV 前必须确保该层 KV 已经 load 完；
- save 需要当前 layer_name、kv_cache tensor、attention metadata；
- 不同 connector 可以只保存 / 加载部分层。
```

### 5.4 0-token step 也可能执行 connector

如果本轮没有 scheduled tokens：

```text
if not num_scheduled_tokens:
  if no KV transfer group:
    return EMPTY_MODEL_RUNNER_OUTPUT
  else:
    return kv_connector_no_forward(...)
```

这用于推进：

```text
- 异步 KV load 的 finished_recving 轮询；
- 异步 KV save 的 finished_sending 轮询；
- connector stats / events / worker_meta 回传；
- KV transfer metadata 消费。
```

关键点：

```text
execute_model 不一定等于模型 forward；它也是 KV connector transfer 的驱动入口。
```

### 5.5 ModelRunnerOutput.kv_connector_output

Worker 最终把 connector 状态放进：

```text
ModelRunnerOutput.kv_connector_output
```

其中 `KVConnectorOutput` 包含：

```text
finished_sending：异步 save / send 完成的 request ids；
finished_recving：异步 load / recv 完成的 request ids；
invalid_block_ids：load 失败或不可信的本地 block ids；
kv_connector_stats：connector 统计；
kv_cache_events：connector 产生的 KV cache events；
kv_connector_worker_meta：connector 自定义 worker → scheduler metadata；
expected_finished_count：多 worker finished 聚合计数。
```

多 worker 场景下，`KVOutputAggregator` 会聚合多个 Worker 的 `KVConnectorOutput`：

```text
finished_sending / finished_recving：按 expected_finished_count 计数聚合；
invalid_block_ids：取 union；
stats / worker_meta / events：调用各自 aggregate 或合并逻辑。
```

---

## 6. Scheduler.update_from_output 的闭环

Worker 返回后，Scheduler 在 `update_from_output()` 中消费 `ModelRunnerOutput`。

KV transfer 相关闭环主要有三类：

```text
1. finished_recving：external KV load 完成；
2. finished_sending：external KV save / send 完成；
3. invalid_block_ids：external KV load 失败或数据不可信。
```

### 6.1 finished_recving：恢复 WAITING_FOR_REMOTE_KVS

Worker 返回：

```text
kv_connector_output.finished_recving = {req_id, ...}
```

Scheduler 先记录：

```text
if req.status == WAITING_FOR_REMOTE_KVS:
  finished_recving_kv_req_ids.add(req_id)
else if req already finished:
  _free_blocks(req)
```

真正恢复发生在下一轮 `schedule()`：

```text
_try_promote_blocked_waiting_request(request)
  → 如果 req_id 不在 finished_recving_kv_req_ids：继续 blocked
  → 如果已 finished_recving：_update_waiting_for_remote_kv(request)
  → status = WAITING 或 PREEMPTED
  → 继续走普通 waiting 调度逻辑
```

`_update_waiting_for_remote_kv()` 成功路径：

```text
kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
if request.num_computed_tokens == request.num_tokens:
  request.num_computed_tokens = request.num_tokens - 1
remove req_id from finished_recving_kv_req_ids
```

含义：

```text
外部 KV 已经 load 到本地 blocks，现在这些 blocks 可以进入本地 prefix cache；如果 full prompt 都来自外部 KV，仍回退最后一个 token 来生成 logits。
```

### 6.2 finished_sending：释放 save 延迟保护的 blocks

请求结束时，Scheduler 会调用 connector：

```text
_connector_finished(request)
  → connector.request_finished(request, block_ids)
  或 connector.request_finished_all_groups(request, block_ids)
```

如果 connector 返回：

```text
delay_free_blocks=True
```

Scheduler 不会立即释放 blocks，而是保留 request 和 blocks，等待 Worker 回传：

```text
kv_connector_output.finished_sending = {req_id}
```

收到后：

```text
_update_from_kv_xfer_finished()
  → _free_blocks(request)
```

一句话记忆：

```text
request_finished() 是“请求结束，要不要保存 KV”的决策点；finished_sending 是“保存完成，可以释放 blocks”的回执。
```

### 6.3 invalid_block_ids：回退重算或失败

Worker connector 如果发现外部 KV load 失败，会回传：

```text
KVConnectorOutput.invalid_block_ids
```

这些是：

```text
本地 GPU KV block id。
```

不是：

```text
远端 block id；
token id；
request id；
hash key。
```

Scheduler 处理链路：

```text
invalid block id
  → 找到引用这些 block 的请求
  → 定位第一个失败 block 的 block index
  → request.num_computed_tokens 回退到 block_index * block_size
  → 根据 kv_load_failure_policy：
      recompute：后续重新本地计算失败 block 及其后续 tokens
      fail：请求 FINISHED_ERROR
```

为什么要回退到第一个失败 block 前：

```text
request.num_computed_tokens 表示该请求已有多少 token 的 KV 可以被信任复用；
如果某个 external block load 失败，从这个 block 开始的 KV 都不能信任。
```

sync load 和 async load 的区别：

```text
sync load 失败：
  请求已经在 running；
  本轮 update_from_output 直接处理；
  recompute 下跳过本轮输出更新，下一轮按回退进度重算；
  fail 下请求 FINISHED_ERROR；
  非 recompute policy 下可能 evict prefix cache blocks。

async load 失败：
  请求在 skipped_waiting / WAITING_FOR_REMOTE_KVS；
  invalid blocks 先回退 num_computed_tokens；
  recompute 下记录 failed_recving_kv_req_ids；
  仍等待 finished_recving；
  promote 时 cache 有效前缀或释放 blocks。
```

---

## 7. BlockPool 和 KV block 生命周期

### 7.1 初始化

BlockPool 初始化：

```text
BlockPool(num_gpu_blocks)
  → 创建 KVCacheBlock[0..N-1]
  → 初始化 FreeKVCacheBlockQueue
  → 创建 cached_block_hash_to_block
  → 从 free queue 弹出一个 block 作为 null_block
```

`null_block` 是结构占位，不是普通可分配 block。

常见用途：

```text
- sliding window 左侧已经滑出窗口的 blocks；
- chunked local attention 当前 chunk 之外的 blocks；
- Mamba align 模式下不需要的 state block；
- prefix hit / hybrid 布局中的空洞占位。
```

### 7.2 新分配

新 block 分配：

```text
BlockPool.get_new_blocks(n)
  → 从 free_block_queue 头部弹出 n 个 blocks
  → 如果这些 blocks 带旧 block_hash，则 _maybe_evict_cached_block()
  → ref_cnt 从 0 变 1
  → 返回给 KVCacheManager / coordinator
```

为什么分配时才 evict hash：

```text
ref_cnt=0 的 cached block 虽然在 free queue 中，但仍可被 prefix cache 命中；
只有当它真的被重新分配写新内容时，旧 hash 才必须删除。
```

### 7.3 prefix cache 命中

prefix hit 后：

```text
SingleTypeKVCacheManager.add_local_computed_blocks()
  → block_pool.touch(new_computed_blocks)
  → 如果 block.ref_cnt == 0，从 free queue 移除
  → ref_cnt++
  → req_to_blocks[request_id].extend(hit_blocks)
```

这表示：

```text
prefix cache hit 不复制 KV；
多个请求可以引用同一个 KVCacheBlock；
ref_cnt 防止仍在使用的 block 被回收。
```

### 7.4 full block 进入 prefix cache

block 进入 prefix cache 的入口：

```text
KVCacheManager.cache_blocks(request, num_computed_tokens)
  → coordinator.cache_blocks(...)
  → BlockPool.cache_full_blocks(...)
  → blk.block_hash = BlockHashWithGroupId(...)
  → cached_block_hash_to_block.insert(...)
```

只 cache：

```text
完整 full blocks。
```

不 cache：

```text
不足一个 block 的 tail tokens；
spec decode 中还可能被拒绝的 draft / lookahead tokens；
async external KV load 尚未 finished_recving 的 blocks。
```

### 7.5 请求释放

普通释放：

```text
Scheduler._free_blocks(request)
  → _free_request_blocks(request)
  → kv_cache_manager.free(request)
  → coordinator.free(request_id)
  → SingleTypeKVCacheManager.free(request_id)
  → block_pool.free_blocks(reversed(req_blocks))
```

`free_blocks()` 做的是：

```text
ref_cnt--；
如果 ref_cnt > 0，说明还有其他请求引用，不进 free queue；
如果 ref_cnt == 0 且 block_hash is None，prepend 到 free queue 头部；
如果 ref_cnt == 0 且 block_hash exists，append 到 free queue 尾部，保留 prefix cache 价值。
```

关键结论：

```text
free 不一定清除 cache；
ref_cnt=0 的 cached block 仍可被 prefix cache 命中，直到它被重新分配时才 evict。
```

---

## 8. External KV load 详细链路

### 8.1 load 的本质

外部 KV load 的本质是：

```text
把原本需要本地 prefill forward 的一段 prompt token，转换成“先分配本地 KV blocks，再由 Worker connector 从外部系统把 KV 写回这些 blocks”。
```

端到端链路：

```text
WAITING request
  → local prefix cache lookup
  → connector.get_num_new_matched_tokens()
  → KVCacheManager.allocate_slots(... num_external_computed_tokens ...)
  → connector.update_state_after_alloc()
  → connector.build_connector_meta()
  → SchedulerOutput.kv_connector_metadata
  → Worker bind_connector_metadata()
  → Worker start_load_kv()
  → Worker wait_for_layer_load()
  → KVConnectorOutput.finished_recving
  → Scheduler._update_from_kv_xfer_finished()
  → 下一轮 _update_waiting_for_remote_kv()
```

### 8.2 sync load

如果：

```text
load_kv_async=False
```

则：

```text
请求可以正常进入 running；
SchedulerOutput 同时包含 token 调度计划和 kv_connector_metadata；
Worker 在本轮 forward 前 / attention layer 前完成必要 KV load；
本轮可以 forward 剩余 tokens。
```

例子：

```text
prompt = 10000 tokens
local hit = 3000 tokens
external hit beyond local = 5000 tokens
load_kv_async = False

num_computed_tokens = 8000
num_new_tokens = 2000

Worker load 5000 tokens 的 external KV，forward 剩余 2000 tokens。
```

### 8.3 async load

如果：

```text
load_kv_async=True
```

则：

```text
num_new_tokens = 0；
Scheduler 只为 external KV hit tokens 分配本地 blocks；
request.status = WAITING_FOR_REMOTE_KVS；
本轮不 forward 这个请求；
Worker 可能通过 0-token step 发起或推进 load；
finished_recving 后请求再恢复普通 waiting 调度。
```

例子：

```text
prompt = 1024 tokens
local hit = 256 tokens
external hit = 768 tokens
load_kv_async = True

Scheduler:
  allocate_slots(num_new_tokens=0, num_external_computed_tokens=768, delay_cache_blocks=True)
  request.status = WAITING_FOR_REMOTE_KVS

Worker:
  kv_connector_no_forward()
  start_load_kv()
  get_finished() → finished_recving={req_id}

Scheduler 下一轮:
  cache_blocks(request, 1024)
  request.num_computed_tokens = 1023
  status = WAITING
```

### 8.4 delay_cache_blocks 的意义

async load 时：

```text
delay_cache_blocks=True
```

表示：

```text
blocks 已经分配给请求；
但外部 KV 内容还没真正写入本地 GPU cache；
不能立即把它们放入 prefix cache hash map。
```

否则可能出现：

```text
请求 A external KV 尚未 load 完；
Scheduler 已把 A 的 blocks 当作 prefix cache；
请求 B 命中这些 blocks；
Worker 读取到未完成或错误 KV。
```

所以 external KV load 的缓存提交必须等：

```text
finished_recving
  → _update_waiting_for_remote_kv()
  → kv_cache_manager.cache_blocks(...)
```

---

## 9. External KV save 详细链路

### 9.1 save 的本质

请求结束后，如果配置了 KV Connector，Scheduler 不会无条件立即释放 KV blocks。

它会先调用：

```text
_connector_finished(request)
  → remove_skipped_blocks(request)
  → get_block_ids(request_id)
  → connector.request_finished(request, block_ids)
     或 connector.request_finished_all_groups(request, block_ids)
```

connector 返回：

```text
(delay_free_blocks, kv_transfer_params)
```

含义：

```text
delay_free_blocks=True：connector 正在异步 save / send，Scheduler 不能立即释放 blocks；
kv_transfer_params：可随输出返回给上层或远端，用于后续 remote prefill/decode。
```

### 9.2 请求结束后的主链路

对于 request-finished 型异步 save / send connector，典型链路是：

```text
Scheduler.update_from_output()
  → request stop
  → _handle_stopped_request()
  → _free_request()
      → _connector_finished()
          → connector.request_finished*()
      → connector_delay_free_blocks ? 暂不释放 : _free_blocks()
  → 下一轮 schedule()
      → connector.build_connector_meta()
      → SchedulerOutput.kv_connector_metadata
  → Worker / ModelRunner execute_model()
      → bind_connector_metadata()
      → save_kv_layer() / wait_for_save()
      → get_finished(finished_req_ids)
      → KVConnectorOutput.finished_sending
  → Scheduler.update_from_output()
      → _update_from_kv_xfer_finished()
      → _free_blocks()
```

并非所有 KVPool connector 都把“保存决策”放在 request_finished 阶段。例如 MooncakeStore 主要在 `build_connector_meta()` 中随 scheduled requests 构造 save metadata；Offloading store 不依赖 finished_sending 释放整个 request，而是通过 worker meta 的 completed jobs 和 jobs_to_flush 管理 store 完成与 block 复用安全。

### 9.3 finished_sending 的作用

`finished_sending` 表示：

```text
Worker 侧异步 save / send 已完成，本地 blocks 可以释放。
```

它通常对应之前：

```text
connector.request_finished*() 返回 delay_free_blocks=True。
```

如果某个 connector 的 `request_finished*()` 已返回 `delay_free_blocks=True`，但还没有对应 `finished_sending`，Scheduler 会继续保留 request 和 blocks，避免外部 transfer 读到被覆盖的数据。Offloading 这类不通过 `delay_free_blocks` 保留整个 request 的 connector 不适用这条规则。

### 9.4 save 与 prefix cache / sliding window

请求结束交给 connector 前，Scheduler 会先调用：

```text
kv_cache_manager.remove_skipped_blocks(
  request_id=request.request_id,
  total_computed_tokens=request.num_computed_tokens,
)
```

原因：

```text
sliding window / local attention / Mamba 等场景下，有些旧 blocks 对后续 attention 已经不可达；
保存前先清理 skipped blocks，可以避免 connector 保存不需要或不正确的 block 范围。
```

---

## 10. invalid blocks 与 recompute / fail

### 10.1 invalid_block_ids 是什么

`invalid_block_ids` 表示：

```text
Worker 侧本地 KV block id，这些 block 对应的 external KV load 失败或不可信。
```

它不是：

```text
远端 block id；
token id；
request id；
hash key。
```

来源：

```text
Worker connector.get_block_ids_with_load_errors()
  → KVConnectorOutput.invalid_block_ids
  → ModelRunnerOutput.kv_connector_output
  → Scheduler.update_from_output()
```

### 10.2 Scheduler 如何定位失败位置

Scheduler 会对相关请求做：

```text
req_block_ids = kv_cache_manager.get_block_ids(req_id)
req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
req_num_computed_blocks = ceil(req_num_computed_tokens / block_size)

遍历 req_block_ids[0:req_num_computed_blocks]
  → 找到第一个 block_id in invalid_block_ids 的 index
  → request.num_computed_tokens = index * block_size
```

为什么要扣掉本轮 scheduled tokens：

```text
update_from_output() 时 request.num_computed_tokens 可能已经在 schedule() 后预推进；
invalid blocks 针对的是进入本轮 forward 前被认为已经 computed 的 external / prefix 范围。
```

### 10.3 failure policy

配置：

```text
KVTransferConfig.kv_load_failure_policy = "recompute" 或 "fail"
```

含义：

```text
recompute：回退请求进度，后续本地重算失败 block 及其后续 tokens；
fail：请求直接 FINISHED_ERROR。
```

### 10.4 downstream dependent blocks

如果 block i 失败，i 后面的 blocks 也不能直接信任。

原因：

```text
prefix cache block hash 是链式依赖：
block_hash[i] = hash(parent_block_hash[i-1], block_tokens[i], extra_keys)
```

所以某些 sync fail 场景会 evict：

```text
失败 block + 后续 dependent blocks
```

避免后续请求命中不可信 prefix cache。

### 10.5 async load 失败为什么仍等 finished_recving

本节描述 recompute policy 下的恢复路径。

async load 中，即使 Worker 已经上报 invalid blocks，也必须等：

```text
finished_recving
```

因为 connector 抽象要求：

```text
async loading 的失败 block 可以在任意 forward pass 上报，最晚必须在 get_finished() 返回该请求 finished_recving 的同一 pass 上报；即使失败，也必须通过 get_finished() 报告完成。
```

所以 recompute policy 下动作分两段：

```text
invalid_block_ids 到达：
  回退 num_computed_tokens，记录 failed_recving_kv_req_ids。

finished_recving 到达：
  _update_waiting_for_remote_kv() 提交有效前缀或释放 blocks。
```

如果 policy 是 fail，请求会被置为 `FINISHED_ERROR`；若 transfer 尚未收尾，blocks 的实际释放仍可能延迟到 finished_recving。

---

## 11. deferred free 与异步安全

### 11.1 deferred free 解决什么问题

`deferred free` 解决的是：

```text
Scheduler 逻辑上已经决定释放某个 request 的 KV blocks；
但某个 in-flight GPU step 可能仍在写这些 blocks；
如果此时 BlockPool 把这些 blocks 分配给新的 KV consumer load，旧写和新 load 之间没有 ordering，可能污染 KV 数据。
```

启用条件：

```text
kv_transfer_config 存在；
当前实例是 KV consumer；
max_concurrent_batches > 1。
```

### 11.2 核心状态

Scheduler 维护：

```text
sched_step_seq：已经发出去的非空执行 step 序号；
processed_step_seq：已经处理完输出的非空 step 序号；
request.last_sched_seq：某请求最近一次被调度到哪个非空 step；
deferred_frees：deque[(fence_seq, blocks)]。
```

### 11.3 主链路

```text
schedule() 产生非空 step
  → sched_step_seq += 1
  → _update_after_schedule()
      → request.last_sched_seq = sched_step_seq

request finished / preempted / transfer done
  → _free_request_blocks(request)
      → 如果 request.last_sched_seq <= processed_step_seq：立即 free
      → 否则：pop_blocks_for_free()，放入 deferred_frees

update_from_output() 处理非空 step
  → processed_step_seq += 1
  → _drain_deferred_frees()
      → fence <= processed_step_seq 的 blocks 真正 free_blocks()
```

### 11.4 deferred free 和 connector delayed-free 的区别

connector delayed-free：

```text
来源：connector.request_finished*() 返回 delay_free_blocks=True；
作用：请求结束后 connector 还要 save/send，所以 Scheduler 暂不开始释放 blocks；
完成信号：finished_sending / finished_recving。
```

deferred free：

```text
来源：_free_request_blocks() 判断 last_sched_seq > processed_step_seq；
作用：Scheduler 已经开始释放，request bookkeeping 已经 pop，但 blocks 暂不还给 BlockPool；
完成信号：对应 in-flight step 的 update_from_output() 被处理。
```

一句话区别：

```text
connector delayed-free 推迟“开始释放 request blocks”；deferred free 推迟“把已摘出的 blocks 归还 BlockPool”。
```

---

## 12. 从 KVPool 视角串端到端

### 12.1 无 KVPool 命中

```text
WAITING request
  → local prefix cache lookup miss
  → external connector hit = 0
  → num_computed_tokens = 0
  → num_new_tokens = prompt_len
  → allocate_slots() 为整个 prompt 分配 blocks
  → Worker forward 写入 KV
  → full blocks 进入本地 prefix cache
  → request finished
  → 如果 connector 不保存，则普通 free
```

### 12.2 KVPool 部分命中，同步 load

```text
WAITING request
  → local prefix hit = L
  → KVPool hit beyond local = E
  → load_kv_async = False
  → num_computed_tokens = L + E
  → num_new_tokens = prompt_len - L - E
  → allocate_slots()：接入 local blocks，分配 external blocks，分配 suffix blocks
  → update_state_after_alloc() 记录 external KV 写入目标 blocks
  → build_connector_meta()
  → Worker start_load_kv()
  → attention layer wait_for_layer_load()
  → forward suffix tokens
  → output 回 Scheduler
```

### 12.3 KVPool 部分命中，异步 load

```text
WAITING request
  → KVPool hit = E
  → load_kv_async = True
  → num_new_tokens = 0
  → allocate_slots(... num_external_computed_tokens=E, delay_cache_blocks=True)
  → request.status = WAITING_FOR_REMOTE_KVS
  → SchedulerOutput.kv_connector_metadata 下发 load 计划
  → Worker no-forward 或普通 step 中推进 load
  → finished_recving 回 Scheduler
  → 下一轮 _update_waiting_for_remote_kv()
  → cache_blocks()
  → request.status = WAITING / PREEMPTED
  → 继续调度剩余 tokens 或最后 token logits
```

### 12.4 KVPool full hit

```text
external hit 覆盖整个 prompt
  → Scheduler 仍需要最后 token logits
  → load 完成后 request.num_computed_tokens 回退到 request.num_tokens - 1
  → 下一轮重算最后 token / 最后 block
  → 得到 next-token logits 后进入 decode
```

### 12.5 load 失败

```text
Worker connector load 失败
  → invalid_block_ids
  → Scheduler 找到受影响请求
  → num_computed_tokens 回退到第一个失败 block 前
  → recompute：后续本地重算失败部分
  → fail：请求 FINISHED_ERROR
```

### 12.6 请求结束后保存到 KVPool

```text
request finished
  → Scheduler._connector_finished()
  → connector.request_finished*()
  → 如果需要保存：delay_free_blocks=True
  → 后续 build_connector_meta() 下发 store metadata
  → Worker save_kv_layer() / wait_for_save()
  → finished_sending
  → Scheduler._free_blocks()
```

---

## 13. 关键数据结构关系

### 13.1 Request

Scheduler 管理的请求对象，包含：

```text
request_id
status
num_tokens
num_computed_tokens
block_hashes
all_token_ids
kv_transfer_params
num_preemptions
last_sched_seq
```

### 13.2 KVCacheBlock

单个 KV block 的 Scheduler 侧元数据：

```text
block_id
ref_cnt
block_hash
prev / next
is_null
```

### 13.3 KVCacheBlocks

`KVCacheManager` 返回给 Scheduler 的 block 包装对象：

```text
tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 对应 KV cache group。

它提供：

```text
get_block_ids() → tuple[list[int], ...]
```

### 13.4 SchedulerOutput

本轮执行计划，包含：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
finished_req_ids
new_block_ids_to_zero
kv_connector_metadata
```

### 13.5 KVConnectorMetadata

Scheduler connector → Worker connector 的 opaque metadata。

作用：

```text
告诉 Worker 本轮有哪些 load / save / preemption / block mapping / job 需要执行。
```

具体字段由 connector 实现决定。

### 13.6 KVConnectorOutput

Worker connector → Scheduler 的通用输出容器：

```text
finished_sending
finished_recving
invalid_block_ids
kv_connector_stats
kv_cache_events
kv_connector_worker_meta
expected_finished_count
```

### 13.7 ModelRunnerOutput

Worker / ModelRunner 返回 Scheduler 的执行结果容器。

其中：

```text
kv_connector_output
```

是 KV transfer 闭环的核心字段。

---

## 14. 和 Scheduler 文档的关系

KV Cache / KV Transfer 不是 Scheduler 的旁路逻辑，而是嵌入 Scheduler 主链路的几段关键动作。

建议联动阅读：

```text
../scheduler/05_prefix_and_external_kv_hits.md
  → 03_prefix_cache_lookup.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
```

```text
../scheduler/06_kv_block_allocation_and_preemption.md
  → 01_kv_cache_manager_role.md
  → 02_block_pool_and_block_lifecycle.md
  → 09_deferred_free_and_async_safety.md
```

```text
../scheduler/08_update_after_worker_output.md
  → 06_external_kv_save_flow.md
  → 08_invalid_blocks_and_recompute.md
```

从 Scheduler 视角最关键的主线是：

```text
waiting 阶段：查 local / external hit，分配 blocks，构造 metadata；
update_from_output 阶段：处理 finished / invalid，恢复请求，释放 blocks。
```

---

## 15. 和 Worker / ModelRunner 文档的关系

KV transfer 的执行部分落在 Worker / ModelRunner。

建议联动阅读：

```text
../executor_worker_model_runner/04_execute_model_flow.md
  → ../executor_worker_model_runner/09_worker_kv_cache_interaction.md
  → 07_worker_kv_connector_flow.md
```

从执行层视角最关键的主线是：

```text
SchedulerOutput.kv_connector_metadata
  → GPUModelRunner.execute_model()
  → handle_preemptions()
  → set_forward_context()
  → legacy maybe_get_kv_connector_output() 或新版 kv_connector.pre_forward()
  → start_load_kv()
  → attention layer wait_for_layer_load() / save_kv_layer()
  → legacy context exit 或新版 kv_connector.post_forward()
  → get_finished() / invalid_block_ids
  → ModelRunnerOutput.kv_connector_output
```

---

## 16. 推荐阅读路线

### 16.1 快速建立全局印象

```text
kv_cache_transfer_overview.md
  → 01_kv_cache_manager_role.md
  → 02_block_pool_and_block_lifecycle.md
  → 04_scheduler_kv_connector_flow.md
```

### 16.2 按 KV 命中和 load 链路完整阅读

```text
kv_cache_transfer_overview.md
  → 03_prefix_cache_lookup.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 07_worker_kv_connector_flow.md
  → 08_invalid_blocks_and_recompute.md
```

### 16.3 按请求结束和 save / release 阅读

```text
06_external_kv_save_flow.md
  → 09_deferred_free_and_async_safety.md
  → 02_block_pool_and_block_lifecycle.md
```

### 16.4 按 KVPool 端到端阅读

```text
kv_cache_transfer_overview.md
  → 10_kvpool_end_to_end.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 06_external_kv_save_flow.md
  → 07_worker_kv_connector_flow.md
```

---

## 17. 文档定位

```text
kv_cache_transfer_overview.md：
  总览主文档，适合快速建立 KV Cache / KV Transfer / KVPool 全局图。

01_kv_cache_manager_role.md：
  解释 KVCacheManager 的定位、接口和边界。

02_block_pool_and_block_lifecycle.md：
  解释 BlockPool、KVCacheBlock、ref_cnt、free queue、prefix cache hash map。

03_prefix_cache_lookup.md：
  解释本地 prefix cache lookup、block_hashes、full hit 重算最后 token。

04_scheduler_kv_connector_flow.md：
  解释 Scheduler 侧 connector 如何查外部命中、记录分配结果并构造 metadata。

05_external_kv_load_flow.md：
  解释 external KV load、WAITING_FOR_REMOTE_KVS、finished_recving 和恢复调度。

06_external_kv_save_flow.md：
  解释 request_finished、save KV、finished_sending 和延迟释放。

07_worker_kv_connector_flow.md：
  解释 Worker / ModelRunner 如何消费 metadata 并执行 load / save。

08_invalid_blocks_and_recompute.md：
  解释 invalid_block_ids、load failure policy、recompute / fail。

09_deferred_free_and_async_safety.md：
  解释 async scheduling / PP / KV consumer 下为什么要 deferred free。

10_kvpool_end_to_end.md：
  从 KVPool 视角串联 Scheduler 和 Worker 的完整链路。
```

---

## 18. 一句话总结

vLLM V1 的 KV Cache / KV Transfer 链路可以记成：

```text
KVCacheManager / BlockPool 在 Scheduler 侧管理本地 KV block 账本；Scheduler 先查本地 prefix cache，再让 KV Connector 查询外部 KVPool，并把命中结果转成本地 block 分配和 connector metadata；Worker / ModelRunner 根据 metadata 真正执行 KV load / save；Scheduler 再根据 finished_recving、finished_sending、invalid_block_ids 完成请求恢复、重算失败、保存收尾和 block 释放。
```

最核心的端到端主线是：

```text
Request
  → local prefix cache lookup
  → external KVPool lookup
  → allocate local KV blocks
  → build kv_connector_metadata
  → Worker load / forward / save
  → kv_connector_output
  → Scheduler recover / recompute / release
```
