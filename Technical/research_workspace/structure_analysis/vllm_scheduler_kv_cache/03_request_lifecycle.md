# 03. 请求生命周期：从入口到释放

## 1. 生命周期总图

```text
外部请求
  -> EngineCoreRequest
  -> Request.from_engine_core_request()
  -> Scheduler.add_request()
  -> waiting / skipped_waiting
  -> Scheduler.schedule()
      -> prefix cache 查询
      -> KV slots 分配
      -> running
      -> SchedulerOutput
  -> model_executor.execute_model()
  -> GPUModelRunner.execute_model()
      -> add/update request state
      -> 更新 block table
      -> 构建 input batch
      -> attention + model forward
      -> sample / pool
      -> ModelRunnerOutput
  -> Scheduler.update_from_output()
      -> 追加 output token
      -> stop 检查
      -> 释放 encoder/KV cache
      -> EngineCoreOutputs
```

关键文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py`

## 2. EngineCore 初始化 KV 与 Scheduler

`EngineCore._initialize_kv_caches()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:240`。

它在 engine 启动时完成 KV cache 规格发现、显存 profile、scheduler KV config 生成与 worker 侧 KV tensor 初始化。

核心步骤：

1. 注册所有 KV cache specs。
2. `model_executor.get_kv_cache_specs()` 获取模型各层 KV cache 需求。
3. 如果模型有 non-causal attention，禁用 chunked prefill 和 prefix caching。
4. `model_executor.determine_available_memory()` profile 可用显存。
5. `get_kv_cache_configs(...)` 生成 worker KV cache config。
6. `generate_scheduler_kv_cache_config(...)` 生成 scheduler 侧 config。
7. 回填 `num_gpu_blocks`、`block_size`、`kv_cache_size_tokens`。
8. `model_executor.initialize_from_config(kv_cache_configs)` 初始化 worker。

当 `EngineCore.add_request()` 被调用时，请求会进入 scheduler，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:372`。

## 3. Request 对象

`Request` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:59`。

它是 V1 中调度和执行的基础单位，替代旧架构中的 `Sequence` / `SequenceGroup` 主体地位。

### 3.1 创建入口

`Request.from_engine_core_request()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:197`。

它从 `EngineCoreRequest` 拷贝：

- `request_id`
- `client_index`
- `prompt_token_ids`
- `prompt_embeds`
- `prompt_is_token_ids`
- `mm_features`
- `sampling_params`
- `pooling_params`
- `arrival_time`
- `lora_request`
- `cache_salt`
- `priority`
- `trace_headers`
- `resumable`
- `abort_immediately`

### 3.2 基础调度字段

在 `Request.__init__()` 中初始化：

- `request_id`：请求唯一标识。
- `client_index`：输出归属的客户端索引。
- `priority`：priority scheduling 使用。
- `arrival_time`：FCFS 或同 priority 下的排序依据。
- `status`：初始通常是 `WAITING`。
- `events`：调度事件记录。
- `stop_reason`：停止原因。

如果请求有 structured output，状态可能先变成：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:111`。

### 3.3 输入与输出 token 字段

- `prompt_token_ids`
- `prompt_embeds`
- `prompt_is_token_ids`
- `num_prompt_tokens`
- `_output_token_ids`
- `_all_token_ids`
- `output_token_ids`
- `all_token_ids`

`_all_token_ids` = prompt tokens + output tokens。

`append_output_token_ids()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:224`。每次追加生成 token 后，会调用 `update_block_hashes()`。

### 3.4 KV/调度进度字段

- `spec_token_ids`
  - speculative decoding draft tokens。
- `num_computed_tokens`
  - 该请求当前已经完成计算的 token 数。
  - 调度器的核心进度指针。
- `cache_salt`
  - prefix cache hash 可能使用。
- `block_hashes`
  - prompt/output token 被切成 block 后的 hash 列表。
- `skip_reading_prefix_cache`
  - 某些请求不能读取 prefix cache，例如 prompt logprobs 或 pooling 场景。
- `is_prefill_chunk`
  - 当前请求是否还处于 chunked prefill 中间段。
- `num_preemptions`
  - 被抢占次数。
- `last_sched_seq`
  - deferred block free fencing 使用。

### 3.5 Async / streaming 字段

- `num_output_placeholders`
- `async_tokens_to_discard`
- `next_decode_eligible_step`
- `resumable`
- `streaming_queue`
- `abort_immediately`

`StreamingUpdate` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:32`，用于 streaming session 继续输入。

## 4. 请求加入 Scheduler

`Scheduler.add_request()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1959`。

分两种情况。

### 4.1 新请求

如果 `request_id` 不存在：

1. 如果 request 是 resumable，初始化 streaming queue。
2. 调 `_enqueue_waiting_request(request)`。
3. 加入 `self.requests`。
4. 如果有 connector，调用 `connector.on_new_request(request)`。
5. 记录 queued event。

`_enqueue_waiting_request()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1812`。

- blocked waiting 状态进入 `skipped_waiting`。
- 普通 waiting 状态进入 `waiting`。

### 4.2 重复 request_id / streaming session

如果 `request_id` 已存在：

- 对 resumable request，会把新的输入片段转成 `StreamingUpdate`。
- 如果已有请求正在等待 streaming input，则立即 `_update_request_as_session()`。
- 否则把 update 放入已有 session 的 queue。

## 5. 首次调度：waiting -> running

waiting 请求在 `Scheduler.schedule()` 的 waiting 部分处理，起点：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:624`。

核心阶段：

### 5.1 选择队列

通过 `_select_waiting_queue_for_scheduling()` 选择 `waiting` 或 `skipped_waiting`。

### 5.2 检查 blocked 状态

如果请求处于：

- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_STREAMING_REQ`

则调用 `_try_promote_blocked_waiting_request()` 尝试恢复。如果失败，继续留在 skipped queue。

### 5.3 查询本地 prefix cache

当 `request.num_computed_tokens == 0` 时：

```text
new_computed_blocks, num_new_local_computed_tokens = kv_cache_manager.get_computed_blocks(request)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:708`。

### 5.4 查询外部 KV cache

如果启用 KV connector：

```text
ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:722`。

`load_kv_async=True` 表示请求需要异步加载 remote KV，此时可能不执行本地 forward。

### 5.5 计算本轮新 token 数

普通 prefill 路径：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
```

然后应用：

- long prefill threshold
- chunked prefill 开关
- token budget
- encoder input 预算
- Mamba block alignment

### 5.6 分配 KV slots

调用：

```text
kv_cache_manager.allocate_slots(...)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:873`。

如果分配失败，waiting 调度停止或跳过。

### 5.7 异步 remote KV load 分支

如果 `load_kv_async=True`：

1. 请求状态设为 `WAITING_FOR_REMOTE_KVS`。
2. 放入 `step_skipped_waiting`。
3. 设置 `request.num_computed_tokens = num_computed_tokens`。
4. 加入 `_inflight_prefills`。
5. 本轮不进入 running，不执行本地 forward。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:916`。

### 5.8 正常进入 running

如果不需要 async KV load：

1. `self.running.append(request)`。
2. 根据状态加入 `scheduled_new_reqs` 或 `scheduled_resumed_reqs`。
3. `req_to_new_blocks[request_id] = kv_cache_manager.get_blocks(request_id)`。
4. `num_scheduled_tokens[request_id] = num_new_tokens`。
5. `token_budget -= num_new_tokens`。
6. `request.status = RUNNING`。
7. `request.num_computed_tokens = num_computed_tokens`。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:939`。

## 6. running 请求继续调度

running 请求调度从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:429` 开始。

核心计算：

```text
num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
```

running 请求通常已经有 worker 侧 state 和 block table，所以 scheduler 只发送差量：

- 新 block ids。
- 新 token 数。
- 必要时 new_token_ids。

如果 KV block 不足，会触发 preemption。

## 7. SchedulerOutput 构建

调度尾部从 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1011` 开始。

### 7.1 新请求数据 NewRequestData

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:30`

字段：

- `req_id`
- `prompt_token_ids`
- `mm_features`
- `sampling_params`
- `pooling_params`
- `block_ids`
- `num_computed_tokens`
- `lora_request`
- `prompt_embeds`
- `prompt_is_token_ids`
- `prefill_token_ids`

worker 第一次见到请求，需要这些完整信息建立本地状态。

### 7.2 已缓存请求数据 CachedRequestData

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:111`

字段：

- `req_ids`
- `resumed_req_ids`
- `new_token_ids`
- `all_token_ids`
- `new_block_ids`
- `num_computed_tokens`
- `num_output_tokens`

它用于已有 worker state 的请求，只传差量。

### 7.3 SchedulerOutput

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:180`

它是 scheduler 和 worker 的主通信对象。

特别重要字段：

- `finished_req_ids`
  - worker 用来删除请求状态。
- `preempted_req_ids`
  - v2 runner 用来把 preempted 也当作需要移除状态的请求。
- `kv_connector_metadata`
  - worker connector 根据它执行 load/save。
- `new_block_ids_to_zero`
  - worker 对新分配 block 清零。

## 8. 模型执行

EngineCore 调用 model executor 的位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:490`。

普通 step：

```text
scheduler_output = scheduler.schedule(...)
future = model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)
engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)
```

对应源码：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:479`。

## 9. Worker 侧接收请求

`GPUModelRunner.execute_model()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1101`。

非 dummy run 开始时：

1. `update_pp_decode_requests()`
2. `finish_requests(scheduler_output)`
3. `free_states(scheduler_output)`
4. `add_requests(scheduler_output)`
5. `update_requests(scheduler_output)`
6. `block_tables.apply_staged_writes()`

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1110`。

### 9.1 add_requests()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:771`

对新请求：

1. 添加 request state。
2. 添加 encoder cache request。
3. 添加 model state。
4. `block_tables.append_block_ids(..., overwrite=True)`。
5. 添加 LoRA state。
6. 添加 sampler/prompt logprobs worker state。

### 9.2 update_requests()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:818`

对已有请求：

1. 更新 worker 侧 `num_computed_tokens`。
2. 如果有新增 block ids，append 到 block table。
3. 更新 prefill token 计数。
4. 对 `new_block_ids_to_zero` 执行 GPU block 清零。

## 10. Prefill 与 Decode 的统一

V1 scheduler 不显式维护两个大阶段，而是用 `num_computed_tokens` 与 `request.num_tokens` 的差值统一表达。

### 10.1 Prefill

如果 `num_computed_tokens < num_prompt_tokens`，worker 会从 prompt tokens 取输入。

`GPUModelRunner.prepare_inputs()` 中通过 `is_prefilling_np` 判断：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:914`。

如果存在 prefill 请求，会调用 `prepare_prefill_inputs()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:917`。

### 10.2 Decode

当 prompt 已经计算完，输入 token 通常来自上一轮 sampled token 或 draft tokens。

`combine_sampled_and_draft_tokens()` 在 `prepare_inputs()` 中被调用，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:951`。

### 10.3 chunked prefill

如果 prompt 很长，scheduler 可以只调度一部分 tokens。

`_update_after_schedule()` 会设置：

```text
request.is_prefill_chunk = request.num_computed_tokens < (request.num_tokens + request.num_output_placeholders)
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1145`。

## 11. 采样与输出回灌

worker 执行完成后返回 `ModelRunnerOutput`。

scheduler 在 `update_from_output()` 中处理它，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1463`。

对每个本轮 scheduled request：

1. 根据 `req_id_to_index` 找到 worker 输出。
2. 读取 sampled token ids。
3. 如果 spec decode 有 rejected tokens，回退 `request.num_computed_tokens`。
4. 释放已经使用完的 encoder inputs。
5. 调 `_update_request_with_output()` 追加 token 并检查 stop。
6. 构建 `EngineCoreOutput`。

`_update_request_with_output()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1848`。

它逐个追加 token，并调用 `check_stop(request, max_model_len)`。

## 12. 请求完成

当请求 stopped：

1. 记录 finish reason。
2. `_handle_stopped_request(request)`。
3. 如果真正 finished，调用 `_free_request(request)`。
4. 从 running 或 waiting 队列移除。
5. 输出 `EngineCoreOutput`。

`_handle_stopped_request()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1830`。

对非 resumable 请求，直接 finished。

对 resumable 请求：

- 如果 streaming queue 有下一段输入，则更新 session 并重新入队。
- 如果没有，状态变成 `WAITING_FOR_STREAMING_REQ`。

## 13. 释放路径

### 13.1 正常释放

`_free_request()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2046`。

它会：

1. 从 `_inflight_prefills` 移除。
2. 调 `_connector_finished()`。
3. 释放 encoder cache。
4. 加入 `finished_req_ids`。
5. 如果不需要延迟，调用 `_free_blocks()`。

`_free_blocks()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2065`。

它会：

```text
_free_request_blocks(request)
del self.requests[request.request_id]
```

### 13.2 KV blocks 释放

`_free_request_blocks()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2077`。

普通情况：

```text
kv_cache_manager.free(request)
```

deferred free 情况：

```text
blocks = kv_cache_manager.pop_blocks_for_free(request)
deferred_frees.append((sched_step_seq, blocks))
```

稍后在 `update_from_output()` 中确认 GPU 写完成后，`_drain_deferred_frees()` 把 blocks 真正归还给 block pool。

### 13.3 Worker 状态释放

worker 侧 `finish_requests()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:750`。

它根据 scheduler output 中的：

- `finished_req_ids`
- `preempted_req_ids`

调用 `_remove_request(req_id)`，释放 worker 侧 request state、model state、encoder cache、prompt logprobs、LoRA state 等。

## 14. Abort 生命周期

外部 abort 由 `EngineCore.abort_requests()` 调用 scheduler：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:409`。

scheduler 的 `finish_requests()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1983`。

它会：

1. 收集 running 和 waiting 中要移除的请求。
2. 批量从队列移除。
3. 设置 finished status。
4. 调 `_free_request()`。

如果请求正在 `WAITING_FOR_REMOTE_KVS`，可能需要 delay free，避免 connector 资源状态不同步。

## 15. 请求生命周期中的核心状态

常见状态包括：

- `WAITING`
- `RUNNING`
- `PREEMPTED`
- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
- `WAITING_FOR_STREAMING_REQ`
- `FINISHED_STOPPED`
- `FINISHED_ABORTED`
- `FINISHED_ERROR`

其中：

- `WAITING`：普通待调度。
- `RUNNING`：持有 worker state 与 KV blocks。
- `PREEMPTED`：被抢占，KV blocks 已释放，后续重新入队。
- `WAITING_FOR_REMOTE_KVS`：KV connector 正在异步加载 remote KV。
- finished 状态最终都会走 `_free_request()`。

## 16. 一句话总结

请求生命周期由 `Request.num_computed_tokens` 驱动：scheduler 每步决定推进多少 token，KV manager 保证 blocks 可用，worker 执行并返回 token，scheduler 再根据输出推进状态、停止或释放资源。
