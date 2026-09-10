# 01. vLLM V1 调度器总览

## 1. 核心文件

调度层主文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:68`：`Scheduler`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py:13`：`SchedulingPolicy`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py:20`：`RequestQueue`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:30`：`NewRequestData`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:111`：`CachedRequestData`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:180`：`SchedulerOutput`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:59`：`Request`

调度器不是单纯的“prefill scheduler”或“decode scheduler”。当前 V1 的核心思想是：每个请求只维护 `num_computed_tokens` 与当前总 token 数，调度器每一轮尝试让 `num_computed_tokens` 追上请求当前需要计算的位置。这套统一模型覆盖了 chunked prefill、decode、prefix cache、spec decode 与未来 jump decoding。

关键注释在 `Scheduler.schedule()` 内部：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:387`。

## 2. Scheduler 的定位

`Scheduler` 位于 engine core 与 worker/model runner 之间。

```text
EngineCore / LLMEngine
  -> Request
  -> Scheduler.add_request()
  -> Scheduler.schedule()
  -> SchedulerOutput
  -> model_executor.execute_model()
  -> GPUModelRunner.execute_model()
  -> ModelRunnerOutput
  -> Scheduler.update_from_output()
  -> EngineCoreOutputs
```

调度器负责：

1. 管理请求集合：`requests`、`waiting`、`skipped_waiting`、`running`。
2. 根据策略选择请求：FCFS 或 priority。
3. 控制预算：最大 running 请求数、最大 scheduled token 数、encoder budget、LoRA 数量限制、KV block 容量、水位线等。
4. 查询 prefix cache 命中。
5. 请求 KV block 分配。
6. 处理 preemption、remote KV load、deferred free、finish/abort。
7. 构建 `SchedulerOutput`，传给 worker 执行。
8. 接收 `ModelRunnerOutput`，更新请求状态、输出 token、释放资源。

## 3. Scheduler 初始化字段

`Scheduler.__init__` 从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:69` 开始。

重要字段包括：

### 3.1 配置类字段

- `self.vllm_config`
- `self.scheduler_config`
- `self.cache_config`
- `self.lora_config`
- `self.kv_cache_config`
- `self.parallel_config`

这些字段决定 scheduler 的预算、并发、prefix cache、KV connector、pipeline parallel、data parallel 等行为。

### 3.2 调度约束

在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:106` 附近初始化：

- `self.max_num_running_reqs`
  - 来源于 `scheduler_config.max_num_seqs`。
  - 控制最多有多少请求处于 running 状态。
- `self.max_num_scheduled_tokens`
  - 优先使用 `scheduler_config.max_num_scheduled_tokens`，否则使用 `max_num_batched_tokens`。
  - 每轮调度最多分配多少 token 计算量。
- `self.max_model_len`
  - 模型最大长度。
- `self.num_sampled_tokens_per_step`
  - 普通生成模型通常为 1，diffusion 模型可能为 0。

### 3.3 队列和请求表

在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:170` 附近：

- `self.requests: dict[str, Request]`
  - 全局请求表，request_id 到 `Request`。
- `self.waiting`
  - 普通待调度队列。
- `self.skipped_waiting`
  - 因结构化输出 grammar、remote KV、streaming input 或临时约束而跳过的 waiting 请求。
- `self.running: list[Request]`
  - 已经被调度并持有运行态 KV/cache 状态的请求。
- `self.finished_req_ids`
  - 上一步与当前步之间完成的请求，用来通知 worker 清理状态。

### 3.4 KV 相关字段

- `self.kv_cache_manager`
  - 调度器和 KV cache 管理层之间的主接口。
- `self.connector`
  - scheduler 侧 KV connector。worker 侧会有对应 connector。
- `self.defer_block_free`
  - 当启用 overlapping batches、async scheduling、pipeline parallel 或 consumer KV connector 时，某些已完成请求的 block 不能立刻归还给 block pool。
- `self.finished_recving_kv_req_ids`
- `self.failed_recving_kv_req_ids`

### 3.5 Encoder/多模态相关字段

- `self.encoder_cache_manager`
- `self.max_num_encoder_input_tokens`
- `self.ec_connector`

这些字段处理 encoder-decoder、multimodal encoder cache、EC connector。

## 4. 请求队列策略

文件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py`

### 4.1 SchedulingPolicy

`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py:13`

```text
FCFS = "fcfs"
PRIORITY = "priority"
```

### 4.2 FCFSRequestQueue

`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py:75`

基于 `deque`：

- `add_request()`：append 到尾部。
- `pop_request()`：popleft。
- `peek_request()`：看队首。
- `prepend_request()`：appendleft。

适合普通先到先服务。

### 4.3 PriorityRequestQueue

`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py:131`

基于 `heapq`：

- priority 值越小越先处理。
- priority 相同时 arrival_time 更早的请求先处理。

### 4.4 waiting 与 skipped_waiting 的关系

`Scheduler._select_waiting_queue_for_scheduling()` 在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1818`。

- FCFS 模式：优先尝试 `skipped_waiting`，再 `waiting`。
- PRIORITY 模式：比较两个队列的队首，选择更高优先级者。

`skipped_waiting` 不是失败队列，而是暂时不能继续的请求队列。例如：

- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
- `WAITING_FOR_STREAMING_REQ`

这些状态由 `_is_blocked_waiting_status()` 判定：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1805`。

## 5. schedule() 的整体结构

`Scheduler.schedule()` 从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:387` 开始。

它每轮大致执行：

```text
1. current_step += 1
2. 初始化本轮临时容器
3. token_budget = max_num_scheduled_tokens
4. kv_cache_manager.new_step_starts()
5. 先调度 running 请求
6. 再调度 waiting / skipped_waiting 请求
7. 计算 common prefix blocks
8. 构建 SchedulerOutput
9. 构建 KV connector metadata / EC connector metadata
10. _update_after_schedule()
11. 返回 SchedulerOutput
```

本轮临时容器包括：

- `scheduled_new_reqs`
- `scheduled_resumed_reqs`
- `scheduled_running_reqs`
- `preempted_reqs`
- `req_to_new_blocks`
- `num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `scheduled_encoder_inputs`

## 6. running 请求调度

running 请求调度从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:429` 开始。

核心逻辑：

1. 遍历 `self.running`。
2. 跳过暂时不能继续 decode 的请求，例如：
   - async scheduling 下已经达到 max token 的请求。
   - pipeline parallel cadence 未到的请求。
   - DP prefill balancing 下被延迟的 prefill chunk。
3. 计算 `num_new_tokens`：
   - `request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens`
4. 应用 long prefill threshold。
5. 应用 token budget。
6. 不超过 max model len。
7. 调度 encoder inputs。
8. 处理 Mamba block-aligned split。
9. 调 `kv_cache_manager.allocate_slots()` 分配新增 KV block。
10. 若分配失败，触发 preemption。

### 6.1 running 请求为什么优先

running 请求已经持有 KV blocks 和 worker 侧状态。如果持续推进 running 请求，可以减少重复 prefill 和缓存抖动。waiting 请求只有在没有 preempted 请求且系统未暂停时才继续调度。

### 6.2 running 请求分配失败时的 preemption

关键代码在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:521` 到 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:569`。

如果 `allocate_slots()` 返回 `None`，说明 KV block 不够或水位线/预留约束不满足。

- priority 策略下：选择 `(priority, arrival_time)` 最大的 running 请求作为被抢占对象。
- FCFS/默认路径下：从 running 列表尾部 pop。

被抢占请求调用 `_preempt_request()`。

## 7. waiting 请求调度

waiting 请求调度从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:624` 开始。

核心流程：

1. 确认没有本轮 preempted 请求。
2. 检查 `waiting` 或 `skipped_waiting` 是否还有请求。
3. 检查 running 数量是否达到 `max_num_running_reqs`。
4. 从合适队列取队首。
5. 若请求处于 blocked waiting 状态，尝试 promote；失败则重新放入 skipped 队列。
6. 检查 LoRA 限制。
7. 如果是新请求，查询本地 prefix cache。
8. 如果有 KV connector，再查询外部 KV 命中。
9. 计算本轮需要计算的 token 数。
10. 分配 KV slots。
11. 如果是异步 remote KV load，将请求放入 `WAITING_FOR_REMOTE_KVS`。
12. 否则放入 `running`，构建新请求或恢复请求输出数据。

## 8. prefix cache 命中在调度中的位置

waiting 请求首次调度时，如果 `request.num_computed_tokens == 0`，会查缓存。

普通路径：

- `self.kv_cache_manager.get_computed_blocks(request)`
- 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:708`

Hybrid + Mamba + Connector 路径：

- `find_longest_cache_hit_per_group()`
- 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:682`

命中结果：

- `new_computed_blocks`
- `num_new_local_computed_tokens`

若启用 connector，还会调用：

- `connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)`
- 位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:722`

然后组合得到：

```text
num_computed_tokens = local prefix cache 命中 token + external KV 命中 token
```

## 9. KV block 分配

调度器不直接操作 block pool，而是调用：

- `self.kv_cache_manager.allocate_slots(...)`

running 路径调用点：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:524`

waiting 路径调用点：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:873`

waiting 路径传入更多信息：

- `num_new_computed_tokens`
- `new_computed_blocks`
- `num_external_computed_tokens`
- `delay_cache_blocks`
- `num_encoder_tokens`
- `full_sequence_must_fit`
- `reserved_blocks`

这些参数用于 prefix cache、connector、encoder-decoder、admission gate、防止 async KV load 消耗过多 block。

## 10. SchedulerOutput

`SchedulerOutput` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:180`。

关键字段：

- `scheduled_new_reqs`
  - 第一次被调度的请求。
  - worker 需要完整缓存该请求的 prompt、采样参数、block ids 等。
- `scheduled_cached_reqs`
  - 已经在 worker 侧有状态的请求，只发送差量。
- `num_scheduled_tokens`
  - 每个请求本轮要执行多少 token。
- `total_num_scheduled_tokens`
  - 本轮总 token 数。
- `scheduled_spec_decode_tokens`
  - spec decode draft tokens。
- `scheduled_encoder_inputs`
  - 多模态/encoder 输入。
- `num_common_prefix_blocks`
  - cascade attention 可用的公共 prefix block 数。
- `finished_req_ids`
  - 通知 worker 清理状态。
- `preempted_req_ids`
  - 通知 v2 runner 清理/重置状态。
- `kv_connector_metadata`
  - KV connector 本轮 load/save 计划。
- `new_block_ids_to_zero`
  - 新分配的 block id，需要 worker 侧清零，避免 stale NaN/data。

## 11. _update_after_schedule()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1128`

它在构建完 `SchedulerOutput` 后执行，主要做：

1. 将每个被调度请求的 `request.num_computed_tokens` 增加 `num_scheduled_tokens`。
2. 如果启用 deferred free，记录 `request.last_sched_seq`。
3. 更新 `request.is_prefill_chunk`。
4. 判断是否存在 structured output 请求。
5. 从 `_inflight_prefills` 移除不再 prefilling 的请求。
6. 清空 `self.finished_req_ids`，但不影响已经放入 `SchedulerOutput` 的 set 引用。

一个关键点：scheduler output 中携带的是本轮执行前的必要状态，而 scheduler 内部会提前推进 `num_computed_tokens`，这样下一轮可以立即继续调度 chunked prefill。

## 12. update_from_output()

`Scheduler.update_from_output()` 从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1463` 开始。

它接收 worker/model runner 返回的 `ModelRunnerOutput`，执行：

1. 处理 deferred free fence。
2. 处理 KV connector invalid blocks。
3. 遍历本轮 scheduled 请求。
4. 处理 spec decode rejected tokens，回退 `num_computed_tokens`。
5. 释放已消费的 encoder inputs。
6. 追加生成 token。
7. 检查 stop 条件。
8. 若请求完成，调用 `_free_request()`。
9. 从 running/waiting 队列移除已停止请求。
10. 更新 KV connector transfer 完成状态。
11. 收集并发布 KV cache events。
12. 构建 `EngineCoreOutputs`。

## 13. 抢占与释放

### 13.1 _preempt_request()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1105`

行为：

1. 要求请求必须处于 `RUNNING`。
2. `_free_request_blocks(request)` 释放 KV blocks。
3. 释放 encoder cache。
4. 从 `_inflight_prefills` 移除。
5. 状态设为 `PREEMPTED`。
6. `num_computed_tokens = 0`。
7. 清空 spec tokens。
8. `num_preemptions += 1`。
9. 放回 waiting 队列头部。

V1 当前 preemption 的主要恢复策略是 recompute：释放 KV，后续重新从 prefix cache/头部计算。

### 13.2 _free_request()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2046`

行为：

1. 确认请求 finished。
2. 通知 connector request finished。
3. 释放 encoder cache。
4. 加入 `finished_req_ids`。
5. 根据 connector/deferred 状态决定是否立即 `_free_blocks()`。
6. 返回可能需要回传给前端的 `kv_xfer_params`。

### 13.3 deferred free

`_free_request_blocks()` 在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2077`。

如果没有并发写风险，直接：

```text
kv_cache_manager.free(request)
```

如果有 deferred free，则：

1. 从 manager 中 `pop_blocks_for_free()`，先移除 bookkeeping。
2. 把 blocks 放入 `deferred_frees`。
3. 等 `update_from_output()` 确认 GPU 写完成后，通过 `_drain_deferred_frees()` 真正归还给 `block_pool`。

这是 async scheduling / PP / KV connector 场景下避免 block 被提前复用导致数据竞争的关键机制。

## 14. 调度器的边界

Scheduler 只做决策，不执行 tensor 计算。

它知道：

- 请求状态
- token 数量
- block id
- prefix cache 命中
- connector metadata
- encoder cache 预算

它不知道或不直接操作：

- CUDA tensor 具体布局
- attention kernel 细节
- logits 计算
- sampler 内部实现

这些由 worker/model runner 和 attention backend 负责。
