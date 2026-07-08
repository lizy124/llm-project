# 08. Scheduler 如何消费 ModelRunnerOutput？

源码位置：

- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/v1/core/sched/utils.py`
- `vllm/vllm/v1/outputs.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/__init__.py`
- `vllm/vllm/v1/request.py`

本问题关注：`Scheduler.update_from_output()` 如何把 `ModelRunnerOutput` 中的 sampled token、logprobs、pooling output、KV connector output、routed experts、cudagraph stats 等执行层结果，消化回 Scheduler 的 request 状态机，并构造 `EngineCoreOutputs` 返回给 Engine / OutputProcessor。

---

## 1. 一句话回答

`Scheduler.update_from_output()` 不是简单把 worker 结果转发出去，而是 Scheduler 侧的一次“执行结果结算”。

主链路是：

```text
ModelRunnerOutput
  → 处理 KV load failure / invalid blocks
  → 保存 routed experts step 数据
  → 按 request 回填 sampled tokens
  → 修正 spec decode accepted / rejected token
  → 检查 stop condition
  → 推进 structured output grammar
  → 释放 KV blocks / encoder cache / connector 状态
  → 构造 EngineCoreOutput
  → 聚合为 EngineCoreOutputs
```

所以可以记成：

```text
ModelRunnerOutput 是 Worker 的执行回执；
update_from_output() 是 Scheduler 把回执写回请求状态机的结算点。
```

---

## 2. 它在整条 EngineCore 链路中的位置

入口在 `EngineCore.step()`：

```text
Scheduler.schedule()
  → model_executor.execute_model(scheduler_output)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → model_executor.sample_tokens(grammar_output)  // 如果 execute_model 返回 None
  → scheduler.update_from_output(scheduler_output, model_output)
```

位置：`vllm/v1/engine/core.py:479` 到 `vllm/v1/engine/core.py:508`

关键点：

```text
SchedulerOutput：Scheduler → Worker，告诉 Worker 本轮跑什么；
ModelRunnerOutput：Worker → Scheduler，告诉 Scheduler 本轮实际产出了什么；
EngineCoreOutputs：Scheduler → Engine / frontend，告诉上层本轮可以发送什么。
```

如果开启 batch queue / async scheduling，`step_with_batch_queue()` 也会在拿到 worker future 后调用同一个入口：

```text
future.result()
  → scheduler.update_from_output(scheduler_output, model_output)
```

位置：`vllm/v1/engine/core.py:519` 到 `vllm/v1/engine/core.py:632`

---

## 3. update_from_output 的输入输出

入口定义：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

位置：`vllm/v1/core/sched/scheduler.py:1464`

### 3.1 输入一：SchedulerOutput

`SchedulerOutput` 是本轮调度计划，定义在：`vllm/v1/core/sched/output.py:181`

关键字段：

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
num_invalid_spec_tokens
```

在 `update_from_output()` 中最重要的是：

```text
num_scheduled_tokens：本轮每个 request 实际 schedule 了多少 token；
scheduled_spec_decode_tokens：本轮给每个 request 带了哪些 draft / spec token；
total_num_scheduled_tokens：是否有真实 GPU 写入，影响 deferred free；
num_invalid_spec_tokens：动态 spec decode + grammar 下无效 draft token 的统计修正。
```

### 3.2 输入二：ModelRunnerOutput

`ModelRunnerOutput` 定义在：`vllm/v1/outputs.py:234`

核心字段：

```text
req_ids
req_id_to_index
sampled_token_ids
logprobs
prompt_logprobs_dict
pooler_output
kv_connector_output
ec_connector_output
num_nans_in_logits
cudagraph_stats
routed_experts
```

其中 `update_from_output()` 会直接取用：

```python
sampled_token_ids = model_runner_output.sampled_token_ids
logprobs = model_runner_output.logprobs
prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
pooler_outputs = model_runner_output.pooler_output
num_nans_in_logits = model_runner_output.num_nans_in_logits
kv_connector_output = model_runner_output.kv_connector_output
cudagraph_stats = model_runner_output.cudagraph_stats
```

位置：`vllm/v1/core/sched/scheduler.py:1469` 到 `vllm/v1/core/sched/scheduler.py:1476`

### 3.3 输出：EngineCoreOutputs

`EngineCoreOutput` 定义在：`vllm/v1/engine/__init__.py:175`

它是单个 request 的本轮输出：

```text
request_id
new_token_ids
new_logprobs
new_prompt_logprobs_tensors
pooling_output
finish_reason
stop_reason
events
kv_transfer_params
prefill_stats
routed_experts
num_nans_in_logits
```

`EngineCoreOutputs` 定义在：`vllm/v1/engine/__init__.py:220`

它按 client 聚合：

```text
outputs: list[EngineCoreOutput]
scheduler_stats
finished_requests
utility_output
wave_complete / start_wave
```

`update_from_output()` 最终返回：

```text
dict[client_index, EngineCoreOutputs]
```

位置：`vllm/v1/core/sched/scheduler.py:1770` 到 `vllm/v1/core/sched/scheduler.py:1803`

---

## 4. update_from_output 的主流程

可以把源码流程压缩成下面 12 步：

```text
1. 拆出 ModelRunnerOutput 字段；
2. 如果开启 deferred block free，推进 processed_step_seq 并 drain 可释放 blocks；
3. 收集 perf stats；
4. 如果 KV connector 报 invalid blocks，先处理 KV load failure；
5. 如果有 routed experts，先保存 step-level routing 数据；
6. 遍历 scheduler_output.num_scheduled_tokens；
7. 根据 req_id 找到 Request 和对应 model output index；
8. 修正 speculative decoding 的 accepted / rejected tokens；
9. 释放已消费的 encoder inputs；
10. append sampled tokens、检查 stop、推进 grammar；
11. 如停止则 handle stopped request，并释放资源；
12. 构造 EngineCoreOutput，最后聚合 stats / finished_requests。
```

对应核心范围：`vllm/v1/core/sched/scheduler.py:1464` 到 `vllm/v1/core/sched/scheduler.py:1803`

---

## 5. 为什么 num_computed_tokens 不是在这里才增加

容易误解的一点是：`update_from_output()` 并不是才开始增加 `request.num_computed_tokens`。

在本轮 `schedule()` 完成后，Scheduler 已经调用 `_update_after_schedule()` 提前推进了 computed token 数：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`vllm/v1/core/sched/scheduler.py:1130` 到 `vllm/v1/core/sched/scheduler.py:1149`

这样做有三个原因：

```text
1. SchedulerOutput 需要保留“本轮原始 schedule 了多少 token”，供 Worker 准备 input ids；
2. Scheduler 可以立刻再次 schedule 仍在 prefill 的请求；
3. 如果后续 spec decode 有 rejected token，再在 update_from_output() 中回退。
```

所以 `update_from_output()` 里的 token 进度更新重点不是“增加 computed tokens”，而是：

```text
- 根据 rejected spec tokens 回退 num_computed_tokens；
- append 真正接受的 sampled tokens 到 output_token_ids；
- 根据 stop condition 决定 request 是否结束。
```

---

## 6. 第一段：deferred free 和 perf stats

开头先处理 deferred block free：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`vllm/v1/core/sched/scheduler.py:1478` 到 `vllm/v1/core/sched/scheduler.py:1482`

含义：

```text
只要本轮确实有 GPU step 完成，说明这个 step 以及之前 enqueue 的 GPU 写入都已经完成；
此时可以把之前因 async scheduling / in-flight GPU write 而延迟释放的 KV blocks 真正还给 block pool。
```

释放函数：

```text
_free_request_blocks()
  → 如果还有 in-flight step，先 pop blocks 并放入 deferred_frees；
_drain_deferred_frees()
  → 等 fence step 完成后再 free blocks。
```

位置：`vllm/v1/core/sched/scheduler.py:2078` 到 `vllm/v1/core/sched/scheduler.py:2105`

如果开启 perf metrics，也会在这里收集本 step 的 GPU perf stats：

```text
perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)
```

位置：`vllm/v1/core/sched/scheduler.py:1484` 到 `vllm/v1/core/sched/scheduler.py:1486`

---

## 7. 第二段：处理 KV load failure / invalid blocks

如果 Worker 侧 connector 返回了 invalid blocks：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(...)
```

位置：`vllm/v1/core/sched/scheduler.py:1491` 到 `vllm/v1/core/sched/scheduler.py:1499`

`KVConnectorOutput.invalid_block_ids` 定义在：`vllm/v1/outputs.py:203`

它表示：

```text
某些外部计算或远端加载的 KV blocks 加载失败；
依赖这些 blocks 的 request 不能直接消费本轮 output，必须回退 / 重算 / 失败。
```

### 7.1 _handle_invalid_blocks 做什么

入口：`vllm/v1/core/sched/scheduler.py:2550`

它会分别扫描：

```text
- async load requests：还在 WAITING_FOR_REMOTE_KVS 的请求；
- sync running requests：已经在 running 中、可能使用了 invalid block 的请求。
```

核心动作在 `_update_requests_with_invalid_blocks()`：

```text
1. 找到 request block table 中命中 invalid_block_ids 的第一个 block；
2. 把 request.num_computed_tokens 截断到该 block 之前；
3. 统计需要重算的 token 数；
4. 如果失败策略不是 recompute，则收集需要 evict 的 blocks；
5. 返回受影响 request ids。
```

位置：`vllm/v1/core/sched/scheduler.py:2447` 到 `vllm/v1/core/sched/scheduler.py:2548`

### 7.2 为什么主循环要跳过这些请求

主循环开始时有：

```python
if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
    continue
```

位置：`vllm/v1/core/sched/scheduler.py:1527` 到 `vllm/v1/core/sched/scheduler.py:1531`

因为这些请求的 KV 状态已经被判定不可信：

```text
不能 append 本轮 sampled tokens；
不能正常检查 stop；
需要等待后续 recompute，或者直接失败。
```

如果配置不是 recompute policy，后面会把这些请求标记为 `FINISHED_ERROR` 并输出错误 finish reason。

位置：`vllm/v1/core/sched/scheduler.py:1718` 到 `vllm/v1/core/sched/scheduler.py:1730`

---

## 8. 第三段：保存 routed experts 数据

如果 Worker 返回了 routed experts：

```python
if model_runner_output.routed_experts is not None:
    re = model_runner_output.routed_experts
    self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
```

位置：`vllm/v1/core/sched/scheduler.py:1501` 到 `vllm/v1/core/sched/scheduler.py:1514`

`RoutedExpertsLists` 定义在：`vllm/v1/outputs.py:164`

它是 step-level 数据：

```text
routing_data: (num_scheduled_tokens, num_layers, num_experts_per_tok)
slot_mapping: (num_scheduled_tokens,)
```

注意它不是按 request 分组的，而是按本轮所有 scheduled tokens 展平。

Scheduler 会再构造 `routing_offsets`：

```text
req_id -> 这个 request 在本轮 routed_experts 展平数组中的起始 offset
```

位置：`vllm/v1/core/sched/scheduler.py:1515` 到 `vllm/v1/core/sched/scheduler.py:1520`

为什么要先保存？

```text
有些 request 可能就在本轮生成的 token 上停止；
如果不先把 routed experts 写入 scheduler-side slot buffer，后面释放 KV blocks 后就无法再读出该请求的 prompt / decode routing 信息。
```

源码注释也强调：

```text
MUST precede the per-request routing reads below.
```

位置：`vllm/v1/core/sched/scheduler.py:1501` 到 `vllm/v1/core/sched/scheduler.py:1505`

---

## 9. 第四段：逐 request 消化输出

主循环遍历：

```python
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
```

位置：`vllm/v1/core/sched/scheduler.py:1527`

这里遍历的是 `scheduler_output.num_scheduled_tokens`，也就是 Scheduler 本轮确实安排执行的请求。

### 9.1 已结束或被 abort 的请求会跳过

```python
request = self.requests.get(req_id)
if request is None or request.is_finished():
    continue
```

位置：`vllm/v1/core/sched/scheduler.py:1532` 到 `vllm/v1/core/sched/scheduler.py:1541`

这可能发生在：

```text
- worker 执行期间用户 abort；
- pipeline parallelism 中请求已经被其他路径结束；
- async scheduling 下 Scheduler 状态已经先变化；
- KV connector delay_free_blocks 下 request 已经 finished 但还没彻底删除。
```

### 9.2 通过 req_id_to_index 找到本 request 的输出

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
```

位置：`vllm/v1/core/sched/scheduler.py:1543` 到 `vllm/v1/core/sched/scheduler.py:1546`

这里不能假设 Scheduler 的 request 顺序和 Worker batch 顺序一致，所以必须使用 `req_id_to_index`。

这也是 `ModelRunnerOutput` 同时携带：

```text
req_ids
req_id_to_index
```

的原因。

---

## 10. 第五段：speculative decoding 的接受 / 拒绝修正

如果本轮调度了 spec tokens：

```python
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
```

位置：`vllm/v1/core/sched/scheduler.py:1548` 到 `vllm/v1/core/sched/scheduler.py:1550`

并且 Worker 返回了 sampled tokens，就计算：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`vllm/v1/core/sched/scheduler.py:1551` 到 `vllm/v1/core/sched/scheduler.py:1557`

理解方式：

```text
generated_token_ids 里包含：
  accepted draft tokens + 当前 step 正常 sampled token

所以：
  accepted draft tokens = len(generated_token_ids) - num_sampled_tokens_per_step
  rejected draft tokens = 原本 scheduled draft 数 - accepted draft 数
```

如果有 rejected token，就回退：

```text
request.num_computed_tokens -= num_rejected
request.num_output_placeholders -= num_rejected   // async scheduling 下
```

位置：`vllm/v1/core/sched/scheduler.py:1558` 到 `vllm/v1/core/sched/scheduler.py:1568`

### 10.1 为什么要回退 num_computed_tokens

因为 `_update_after_schedule()` 已经提前把本轮 scheduled tokens 全部算进 `num_computed_tokens` 了。

但 spec decode 中：

```text
scheduled draft token 不代表最终 accepted；
被拒绝的 draft token 不能算作已计算进度；
因此 update_from_output() 必须在看到真实 sampled result 后回退。
```

### 10.2 spec decoding stats

如果开启 stats，还会记录：

```text
num_draft_tokens
num_accepted_tokens
num_invalid_spec_tokens
```

位置：`vllm/v1/core/sched/scheduler.py:1569` 到 `vllm/v1/core/sched/scheduler.py:1575`

统计构造入口：`vllm/v1/core/sched/scheduler.py:2262` 到 `vllm/v1/core/sched/scheduler.py:2279`

---

## 11. 第六段：释放已消费的 encoder inputs

如果 request 有 encoder / multimodal 输入：

```python
if request.has_encoder_inputs:
    self._free_encoder_inputs(request)
```

位置：`vllm/v1/core/sched/scheduler.py:1577` 到 `vllm/v1/core/sched/scheduler.py:1579`

`_free_encoder_inputs()` 会判断哪些 encoder input 已经不再需要：

```text
- encoder-decoder：生成出第一个 token 后，cross attention KV 已经缓存，可以释放；
- 普通多模态占位：当 num_computed_tokens 已经越过该 multimodal span，且不会被 pending draft rejection 回滚到该 span，就可以释放。
```

位置：`vllm/v1/core/sched/scheduler.py:1867` 到 `vllm/v1/core/sched/scheduler.py:1895`

这一步释放的是 Scheduler 侧 encoder cache manager 管理的输入缓存，不是 Worker 侧 tensor 本身。

---

## 12. 第七段：append output token 并检查 stop

核心逻辑：

```python
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
elif request.pooling_params and pooler_output is not None:
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
```

位置：`vllm/v1/core/sched/scheduler.py:1589` 到 `vllm/v1/core/sched/scheduler.py:1597`

### 12.1 generation 请求

generation 请求会走 `_update_request_with_output()`：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]
        break
```

位置：`vllm/v1/core/sched/scheduler.py:1849` 到 `vllm/v1/core/sched/scheduler.py:1865`

它做两件事：

```text
1. 把 token 追加到 request._output_token_ids 和 request._all_token_ids；
2. 每追加一个 token 就立刻检查 stop condition。
```

`Request.append_output_token_ids()` 定义在：`vllm/v1/request.py:224`

它会同时维护：

```text
_output_token_ids：生成 token；
_all_token_ids：prompt + output 全序列；
block_hashes：必要时更新 prefix cache hash。
```

### 12.2 pooling 请求

pooling 请求没有 sampled token。

只要本轮有 `pooler_output`：

```text
request.status = FINISHED_STOPPED
stopped = True
```

位置：`vllm/v1/core/sched/scheduler.py:1594` 到 `vllm/v1/core/sched/scheduler.py:1597`

所以 pooling 模型的结束条件更简单：

```text
pooling output 出来即结束。
```

---

## 13. stop condition 具体检查什么

`check_stop()` 定义在：`vllm/v1/core/sched/utils.py:94`

检查顺序是：

```text
1. 如果 output token 数还没到 min_tokens，不允许停止；
2. 如果最后一个 token 是 eos_token_id，FINISHED_STOPPED；
3. 如果最后一个 token 在 stop_token_ids 中，FINISHED_STOPPED，并设置 stop_reason；
4. 如果总 token 数达到 max_model_len，或 output token 数达到 max_tokens，FINISHED_LENGTH_CAPPED；
5. 如果 repetition_detection 命中，FINISHED_REPETITION，并设置 stop_reason = "repetition_detected"；
6. 否则继续生成。
```

对应源码：`vllm/v1/core/sched/utils.py:94` 到 `vllm/v1/core/sched/utils.py:130`

注意：

```text
stop string 的最终处理通常不在 Scheduler 这里做文本级匹配；
Scheduler 这里主要看 token 级 EOS / stop_token_ids / length / repetition。
```

`RequestStatus` 到外部 `finish_reason` 的映射在：`vllm/v1/request.py:323` 到 `vllm/v1/request.py:365`

映射关系：

| RequestStatus | FinishReason |
|---|---|
| `FINISHED_STOPPED` | `stop` |
| `FINISHED_LENGTH_CAPPED` | `length` |
| `FINISHED_ABORTED` | `abort` |
| `FINISHED_IGNORED` | `length` |
| `FINISHED_ERROR` | `error` |
| `FINISHED_REPETITION` | `repetition` |

---

## 14. 第八段：structured output / grammar 状态推进

如果本轮产生了 new tokens，并且这个 request 需要推进 structured output grammar：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    struct_output_request = request.structured_output_request
    ...
    if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
        request.status = RequestStatus.FINISHED_ERROR
        request.resumable = False
        stopped = True
```

位置：`vllm/v1/core/sched/scheduler.py:1599` 到 `vllm/v1/core/sched/scheduler.py:1615`

这一步和采样前的 grammar bitmask 是一前一后：

```text
采样前：
  Scheduler.get_grammar_bitmask()
    → 生成 bitmask
    → Worker apply_grammar_bitmask()
    → Sampler.sample()

采样后：
  Scheduler.update_from_output()
    → grammar.accept_tokens()
    → 推进 grammar 状态机
```

`get_grammar_bitmask()` 在：`vllm/v1/core/sched/scheduler.py:1440` 到 `vllm/v1/core/sched/scheduler.py:1462`

它只给：

```text
use_structured_output 且不是 prefill chunk 的 request
```

生成 bitmask。

如果 `accept_tokens()` 返回 false，说明出现了理论上不该出现的 grammar mismatch：

```text
Scheduler 会把 request 标记为 FINISHED_ERROR，且 resumable = False。
```

---

## 15. 第九段：routed experts 按 request 回填

如果开启 `enable_return_routed_experts` 且本 request 本轮确实有 new tokens，Scheduler 会为本 request 构造输出用的 routed experts：

```text
prefill 首次完成：
  从 scheduler-side slot buffer 读取 prompt 范围 routed experts；

decode / re-prefill：
  从本轮 routing_data 中取末尾 len(new_token_ids) 行；

spec decode：
  accepted tokens 位于 scheduled range 开头，取 req_offset 到 req_offset + len(new_token_ids)。
```

位置：`vllm/v1/core/sched/scheduler.py:1616` 到 `vllm/v1/core/sched/scheduler.py:1655`

为什么分情况？

```text
prefill 的 routing 数据对应 prompt tokens，可能需要按 KV slot 回读；
decode 的 routing 数据对应本轮最后生成 token；
spec decode 中 accepted tokens 在 scheduled range 前部，rejected tokens 在后部，不能简单取末尾。
```

---

## 16. 第十段：停止请求与资源释放

如果本 request stopped：

```python
finish_reason = request.get_finished_reason()
finished = self._handle_stopped_request(request)
if finished:
    kv_transfer_params = self._free_request(request)
```

位置：`vllm/v1/core/sched/scheduler.py:1656` 到 `vllm/v1/core/sched/scheduler.py:1664`

### 16.1 为什么先保存 finish_reason

源码注释说明：

```text
_handle_stopped_request() 可能把 resumable / streaming request 的状态重置为 WAITING_FOR_STREAMING_REQ；
所以必须在调用它之前捕获 finish_reason。
```

位置：`vllm/v1/core/sched/scheduler.py:1657` 到 `vllm/v1/core/sched/scheduler.py:1660`

### 16.2 _handle_stopped_request 做什么

入口：`vllm/v1/core/sched/scheduler.py:1831`

逻辑：

```text
如果 request 不可 resumable：
  返回 True，表示彻底 finished；

如果 request 有 streaming_queue：
  取下一段 StreamingUpdate；
  如果取到 None，表示 streaming session 结束，返回 True；
  否则更新 session，把请求重新入队；

如果 request 可 resumable 但暂无下一段输入：
  status = WAITING_FOR_STREAMING_REQ；
  num_waiting_for_streaming_input += 1；
  重新加入 skipped_waiting / waiting。
```

位置：`vllm/v1/core/sched/scheduler.py:1831` 到 `vllm/v1/core/sched/scheduler.py:1847`

所以：

```text
stopped 不一定代表请求对象立即删除；
对 streaming / resumable 请求，stopped 可能只是当前输入 chunk 结束。
```

### 16.3 _free_request 释放什么

入口：`vllm/v1/core/sched/scheduler.py:2047`

它会：

```text
1. 从 _inflight_prefills 删除；
2. 调用 _connector_finished()，通知 KV connector request finished；
3. 释放 encoder cache manager 中该 request 相关状态；
4. 把 request_id 加入 finished_req_ids；
5. 如果 include_finished_set，记录到 finished_req_ids_dict[client_index]；
6. 如果不需要 delay_free_blocks，则释放 KV blocks 并从 self.requests 删除；
7. 返回 kv_transfer_params，随 EngineCoreOutput 带给上层。
```

位置：`vllm/v1/core/sched/scheduler.py:2047` 到 `vllm/v1/core/sched/scheduler.py:2064`

`_free_blocks()` 会真正释放 request blocks，并删除 `self.requests[request_id]`：

位置：`vllm/v1/core/sched/scheduler.py:2066` 到 `vllm/v1/core/sched/scheduler.py:2069`

### 16.4 connector 为什么会影响释放

`_connector_finished()` 会调用 KV connector：

```text
connector.request_finished(...)
connector.request_finished_all_groups(...)
```

位置：`vllm/v1/core/sched/scheduler.py:2300` 到 `vllm/v1/core/sched/scheduler.py:2329`

它可能返回：

```text
- 是否 delay free blocks；
- kv_transfer_params，随输出返回给上层。
```

这就是为什么 request finished 后不一定立刻释放 blocks：

```text
某些 P/D 分离或 KV transfer 场景，需要等待远端发送 / 接收完成。
```

---

## 17. 第十一段：提取 logprobs / prompt logprobs / NaN 诊断

### 17.1 generation logprobs

如果请求了 generation logprobs：

```python
if request.sampling_params is not None
   and request.sampling_params.num_logprobs is not None
   and logprobs:
    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

位置：`vllm/v1/core/sched/scheduler.py:1670` 到 `vllm/v1/core/sched/scheduler.py:1676`

注意 `len(new_token_ids)` 很重要：

```text
如果 stop condition 在本轮中途触发，new_token_ids 会被 trim；
logprobs 也必须只切出真正对外输出的 token 对应部分。
```

`LogprobsLists.slice_request()` 定义在：`vllm/v1/outputs.py:40`

它支持 speculative decoding 下每个 request 生成 token 数不同的情况。

### 17.2 prompt logprobs

```python
prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
```

位置：`vllm/v1/core/sched/scheduler.py:1681` 到 `vllm/v1/core/sched/scheduler.py:1682`

`prompt_logprobs_dict` 来自 `ModelRunnerOutput`，定义在：`vllm/v1/outputs.py:251`

它是：

```text
req_id -> LogprobsTensors | None
```

### 17.3 num_nans_in_logits

如果 Worker 发现 logits 里有 NaN：

```python
request.num_nans_in_logits = num_nans_in_logits[req_id]
```

位置：`vllm/v1/core/sched/scheduler.py:1678` 到 `vllm/v1/core/sched/scheduler.py:1679`

随后会写入 `EngineCoreOutput.num_nans_in_logits`：

位置：`vllm/v1/core/sched/scheduler.py:1703` 到 `vllm/v1/core/sched/scheduler.py:1704`

---

## 18. 第十二段：什么时候会生成 EngineCoreOutput

条件是：

```python
if (
    new_token_ids
    or pooler_output is not None
    or kv_transfer_params
    or stopped
):
    outputs[request.client_index].append(EngineCoreOutput(...))
else:
    assert not prompt_logprobs_tensors
```

位置：`vllm/v1/core/sched/scheduler.py:1683` 到 `vllm/v1/core/sched/scheduler.py:1710`

也就是说，只有下面几类情况会向上层返回输出：

```text
- 本轮有新生成 token；
- pooling 模型返回了 pooling output；
- request finished 时产生了 kv_transfer_params；
- request stopped，即使没有 new token，也需要告诉上层 finish；
```

反过来：

```text
纯 partial prefill、没有可见 token、没有 pooling output、没有 stop 的步骤，不会生成 EngineCoreOutput。
```

源码里也有不变量：

```text
EngineCore returns no partial prefill outputs.
```

位置：`vllm/v1/core/sched/scheduler.py:1707` 到 `vllm/v1/core/sched/scheduler.py:1710`

---

## 19. EngineCoreOutput 里具体填什么

构造位置：`vllm/v1/core/sched/scheduler.py:1690` 到 `vllm/v1/core/sched/scheduler.py:1705`

字段含义：

| 字段 | 来源 |
|---|---|
| `request_id` | 当前 req_id |
| `new_token_ids` | append 且 stop trim 后的新 token |
| `finish_reason` | `request.get_finished_reason()` |
| `new_logprobs` | 从 `ModelRunnerOutput.logprobs` 切片 |
| `new_prompt_logprobs_tensors` | `prompt_logprobs_dict[req_id]` |
| `pooling_output` | `pooler_outputs[req_index]` |
| `stop_reason` | `request.stop_reason` |
| `events` | `request.take_events()` |
| `prefill_stats` | `request.take_prefill_stats()` |
| `kv_transfer_params` | `_free_request()` / connector 返回 |
| `trace_headers` | request trace headers |
| `routed_experts` | 本 request 对应 routing 数据 |
| `num_nans_in_logits` | request 级 NaN 统计 |

这一步完成后，Scheduler 已经把 Worker 的 output 变成了 EngineCore 可以交给上层的内部输出。

---

## 20. 第十三段：从 running / waiting 队列移除 stopped requests

主循环里只是记录：

```text
stopped_running_reqs
stopped_preempted_reqs
```

位置：`vllm/v1/core/sched/scheduler.py:1525` 到 `vllm/v1/core/sched/scheduler.py:1526`

真正批量移除在循环后：

```python
if stopped_running_reqs:
    self.running = remove_all(self.running, stopped_running_reqs)
if stopped_preempted_reqs:
    self.waiting.remove_requests(stopped_preempted_reqs)
```

位置：`vllm/v1/core/sched/scheduler.py:1711` 到 `vllm/v1/core/sched/scheduler.py:1716`

为什么不在主循环里直接移除？

```text
running / waiting 是 Scheduler 的核心队列；
主循环可能遍历大量请求，边遍历边修改队列容易增加复杂度；
先收集 set，循环后批量移除更清晰，也更高效。
```

---

## 21. 第十四段：KV connector finished_sending / finished_recving

主循环完成后，如果有 connector output：

```python
if kv_connector_output:
    self._update_from_kv_xfer_finished(kv_connector_output)
```

位置：`vllm/v1/core/sched/scheduler.py:1732` 到 `vllm/v1/core/sched/scheduler.py:1734`

入口：`vllm/v1/core/sched/scheduler.py:2418`

它处理 Worker 侧 connector 的两个信号：

```text
finished_recving：远端 KV 接收完成；
finished_sending：本 request 的 KV 发送完成。
```

### 21.1 finished_recving

```text
如果 request 还在 WAITING_FOR_REMOTE_KVS：
  把 req_id 加入 finished_recving_kv_req_ids；
  后续 _try_promote_blocked_waiting_request() 会把它重新变成 WAITING / PREEMPTED；

如果 request 已经 finished：
  说明之前为了 connector 延迟释放，现在可以真正 free blocks。
```

位置：`vllm/v1/core/sched/scheduler.py:2432` 到 `vllm/v1/core/sched/scheduler.py:2441`

### 21.2 finished_sending

```text
finished_sending 表示 connector 侧已经完成发送；
Scheduler 可以释放 request blocks 并删除 request。
```

位置：`vllm/v1/core/sched/scheduler.py:2442` 到 `vllm/v1/core/sched/scheduler.py:2445`

### 21.3 blocked waiting request 如何恢复

恢复逻辑在 `_try_promote_blocked_waiting_request()`：

```text
WAITING_FOR_REMOTE_KVS
  → 等 req_id 出现在 finished_recving_kv_req_ids
  → _update_waiting_for_remote_kv()
  → WAITING 或 PREEMPTED
```

位置：`vllm/v1/core/sched/scheduler.py:2385` 到 `vllm/v1/core/sched/scheduler.py:2400`

---

## 22. 第十五段：KV cache events / connector stats / scheduler stats

### 22.1 connector stats

Worker 侧的 connector stats 来自：

```text
kv_connector_output.kv_connector_stats
```

位置：`vllm/v1/core/sched/scheduler.py:1736` 到 `vllm/v1/core/sched/scheduler.py:1739`

如果 Scheduler 自己也有 connector，会再合并 scheduler-side stats：

位置：`vllm/v1/core/sched/scheduler.py:1740` 到 `vllm/v1/core/sched/scheduler.py:1752`

### 22.2 KV cache events

Scheduler 会收集：

```text
kv_cache_manager.take_events()
connector.take_events()
```

然后发布为 `KVEventBatch`：

位置：`vllm/v1/core/sched/scheduler.py:1753` 到 `vllm/v1/core/sched/scheduler.py:1768`

### 22.3 scheduler stats

最后调用：

```python
self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)
```

位置：`vllm/v1/core/sched/scheduler.py:1791` 到 `vllm/v1/core/sched/scheduler.py:1801`

`make_stats()` 会聚合：

```text
num_running_reqs
num_waiting_reqs
num_skipped_waiting_reqs
kv_cache_usage
prefix_cache_stats
connector_prefix_cache_stats
kv_cache_eviction_events
spec_decoding_stats
kv_connector_stats
cudagraph_stats
perf_stats
```

位置：`vllm/v1/core/sched/scheduler.py:2224` 到 `vllm/v1/core/sched/scheduler.py:2260`

如果本 step 没有任何 request output，但有 stats，Scheduler 也会创建一个 `EngineCoreOutputs()` 放到 client 0：

```text
必须返回 stats even if there are no request outputs this step。
```

位置：`vllm/v1/core/sched/scheduler.py:1796` 到 `vllm/v1/core/sched/scheduler.py:1800`

---

## 23. 第十六段：finished_requests 集合

如果 Scheduler 初始化时开启了 `include_finished_set`，会维护：

```text
finished_req_ids_dict: dict[int, set[str]]
```

位置：`vllm/v1/core/sched/scheduler.py:97` 到 `vllm/v1/core/sched/scheduler.py:103`

当 `_free_request()` 执行时，会记录：

```text
finished_req_ids.add(request_id)
finished_req_ids_dict[client_index].add(request_id)
```

位置：`vllm/v1/core/sched/scheduler.py:2055` 到 `vllm/v1/core/sched/scheduler.py:2058`

`update_from_output()` 最后会把这些 ids 放进对应 client 的 `EngineCoreOutputs.finished_requests`：

位置：`vllm/v1/core/sched/scheduler.py:1777` 到 `vllm/v1/core/sched/scheduler.py:1789`

用途：

```text
多 engine / 多前端场景下，可以高效追踪 request lifetime；
即使某个 finished request 没有普通 token output，也能通知上层它已经结束。
```

---

## 24. 状态变化总览

一次正常 decode 的状态变化：

```text
schedule()
  → request.status = RUNNING
  → request.num_computed_tokens += 1

Worker / ModelRunner
  → sampled_token_ids = [[token]]

update_from_output()
  → request.append_output_token_ids(token)
  → check_stop()
  → 如果未停止：保留 RUNNING，等待下一轮 schedule
  → 如果停止：FINISHED_*，释放资源，构造 EngineCoreOutput
```

一次 spec decode 的状态变化：

```text
schedule()
  → schedule normal token + draft tokens
  → num_computed_tokens 先按全部 scheduled tokens 增加

Worker / ModelRunner
  → 返回 accepted draft tokens + sampled token

update_from_output()
  → num_accepted = len(generated_token_ids) - num_sampled
  → num_rejected = num_draft_tokens - num_accepted
  → num_computed_tokens -= num_rejected
  → 只 append accepted + sampled token
```

一次 partial prefill 的状态变化：

```text
schedule()
  → 调度 prompt chunk

Worker / ModelRunner
  → 通常没有 sampled token

update_from_output()
  → 不生成 EngineCoreOutput
  → 只完成进度 / encoder cache / stats 等内部状态维护
```

一次 pooling 的状态变化：

```text
schedule()
  → 调度 pooling request

Worker / ModelRunner
  → 返回 pooler_output

update_from_output()
  → request.status = FINISHED_STOPPED
  → EngineCoreOutput.pooling_output = pooler_output
  → 释放资源
```

---

## 25. 和 Worker / ModelRunner 的边界

### 25.1 ModelRunner 负责什么

`ModelRunnerOutput` 在 Worker 侧已经准备好：

```text
- sampled_token_ids；
- logprobs / prompt_logprobs；
- pooler_output；
- kv_connector_output；
- routed_experts；
- num_nans_in_logits；
- cudagraph_stats。
```

但 Worker 不负责：

```text
- request 是否 finished；
- stop reason / finish reason；
- Scheduler running / waiting 队列更新；
- KV block 生命周期最终释放；
- EngineCoreOutput 构造。
```

### 25.2 Scheduler 负责什么

Scheduler 在 `update_from_output()` 中负责：

```text
- 维护 Request.status；
- 维护 request output_token_ids / all_token_ids；
- 修正 num_computed_tokens；
- 维护 running / waiting / skipped_waiting 队列；
- 和 KV cache manager / connector 对账；
- 决定对上层输出什么。
```

所以边界可以记成：

```text
ModelRunner 产出“模型执行事实”；
Scheduler 决定“这些事实如何改变请求状态”。
```

---

## 26. 容易疑惑的点

### 26.1 update_from_output 会不会处理所有 requests？

不会。

它主循环只遍历：

```text
scheduler_output.num_scheduled_tokens
```

也就是本轮被 schedule 的请求。

没有被本轮调度的 waiting / skipped_waiting 请求不会在主循环里处理，但 KV connector finished / finished_requests / stats 等全局事件仍可能在循环后处理。

### 26.2 为什么 generated_token_ids 可能为空？

常见原因：

```text
- partial prefill，只计算 prompt chunk，不采样；
- pooling 模型没有 sampled token；
- KV connector only output；
- 请求在 worker 执行期间被 abort，Scheduler 跳过；
- ModelRunnerOutput.EMPTY_MODEL_RUNNER_OUTPUT。
```

### 26.3 sampled_token_ids 为什么是 list[list[int]]？

因为一轮里每个 request 生成的 token 数可能不同：

```text
普通 decode：通常 1 个；
spec decode：accepted draft tokens + sampled token，可能多个；
jump decoding / 特殊 runner：也可能不固定。
```

### 26.4 为什么 stop 后要 trim new_token_ids？

`check_stop()` 是逐 token 检查的。

如果本轮一次返回多个 token，其中第 2 个 token 已经触发 stop，那么第 3 个及之后的 token 不应对外输出，也不应保留在本 request 的可见 output 中。

所以 `_update_request_with_output()` 会：

```python
del new_token_ids[num_new:]
```

位置：`vllm/v1/core/sched/scheduler.py:1862` 到 `vllm/v1/core/sched/scheduler.py:1864`

### 26.5 stopped 是否一定会删除 request？

不一定。

对普通不可 resumable 请求，stopped 后会释放资源并删除。

对 resumable / streaming request：

```text
stopped 可能表示当前 input chunk 结束；
Scheduler 会把它更新成 WAITING_FOR_STREAMING_REQ 或加载下一段 StreamingUpdate；
只有 streaming_queue 返回 None 时才彻底 finished。
```

### 26.6 为什么 prompt_logprobs 不一定返回？

只有当本轮真的有可见输出条件时才会构造 `EngineCoreOutput`：

```text
new_token_ids / pooler_output / kv_transfer_params / stopped
```

否则 Scheduler 断言没有 prompt_logprobs：

```text
EngineCore returns no partial prefill outputs.
```

### 26.7 KV connector finished_sending 为什么可能删除已经 finished 的请求？

有些 connector 会要求 request finished 后延迟释放 blocks，直到远端发送完成。

此时 request 已经从调度意义上 finished，但仍留在 `self.requests` 中等待 connector completion。

当 `finished_sending` 到达时，Scheduler 才调用 `_free_blocks()` 真正释放并删除 request。

---

## 27. 一个完整 decode step 时间线

```text
1. EngineCore.step()
2. Scheduler.schedule()
3. _update_after_schedule()
   - num_computed_tokens += scheduled tokens
   - 标记 is_prefill_chunk
   - 记录 routed experts block snapshot
4. Executor / Worker / ModelRunner execute_model()
5. sample_tokens()
   - sampler 产出 sampled_token_ids / logprobs
   - 产出 ModelRunnerOutput
6. EngineCore 处理 abort queue
7. Scheduler.update_from_output()
   - invalid KV blocks
   - routed experts store
   - spec decode rejection rollback
   - append output token
   - check_stop
   - grammar.accept_tokens
   - logprobs / prompt_logprobs 切片
   - free request / connector update
   - stats / events
8. 返回 EngineCoreOutputs
9. OutputProcessor / frontend 继续处理文本 detokenization 和用户可见输出
```

---

## 28. 总结

`Scheduler.update_from_output()` 的核心可以压缩为：

```text
ModelRunnerOutput
  → 对齐 req_id / req_index
  → 修正 spec decode
  → append token
  → stop / grammar / pooling 判断
  → logprobs / routed experts / stats 回填
  → KV / encoder / connector 资源释放
  → EngineCoreOutputs
```

如果只记住一句话：

```text
update_from_output() 是 vLLM V1 每轮执行后的状态结算器：它把 Worker 产出的模型结果，变成 Scheduler 可继续调度的请求状态，以及 EngineCore 可向上游发送的输出。
```
