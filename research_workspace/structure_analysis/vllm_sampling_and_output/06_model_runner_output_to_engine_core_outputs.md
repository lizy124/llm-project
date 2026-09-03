# 06 ModelRunnerOutput 到 EngineCoreOutputs

本篇梳理 worker/model runner 产生的底层输出如何进入 scheduler，并被整理成 engine core 对前台返回的 `EngineCoreOutputs`。

## 1. 位置与主链路

主入口：`Scheduler.update_from_output()`

文件：`code/vllm/vllm/v1/core/sched/scheduler.py`

入口锚点：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

总体链路：

```text
GPUModelRunner / Worker
  ↓
ModelRunnerOutput
  ↓
Scheduler.update_from_output()
  ↓
更新 Request.output_token_ids / status / stop_reason / grammar FSM / stats
  ↓
EngineCoreOutput
  ↓
按 client_index 分组
  ↓
EngineCoreOutputs
```

## 2. ModelRunnerOutput

定义：`code/vllm/vllm/v1/outputs.py:234`

注释说明：它会被序列化发给 scheduler，Tensor 序列化昂贵，所以字段尽量用 list。

关键字段：

| 字段 | 含义 |
|---|---|
| `req_ids` | 当前 batch 中 request id 顺序 |
| `req_id_to_index` | request id 到 batch index 的映射 |
| `sampled_token_ids` | 每个请求本 step 新生成 token ids |
| `logprobs` | sample logprobs |
| `prompt_logprobs_dict` | 每个请求的 prompt logprobs |
| `pooler_output` | pooling/embedding/classification/scoring 输出 |
| `kv_connector_output` | KV transfer 相关输出 |
| `num_nans_in_logits` | 每个请求 logits NaN 数量 |
| `routed_experts` | MoE routed experts 信息 |

锚点：

- 类定义：`code/vllm/vllm/v1/outputs.py:234`
- `req_ids`：`code/vllm/vllm/v1/outputs.py:235`
- `req_id_to_index`：`code/vllm/vllm/v1/outputs.py:237`
- `sampled_token_ids`：`code/vllm/vllm/v1/outputs.py:240`
- `logprobs`：`code/vllm/vllm/v1/outputs.py:246`
- `prompt_logprobs_dict`：`code/vllm/vllm/v1/outputs.py:251`
- `pooler_output`：`code/vllm/vllm/v1/outputs.py:259`
- `kv_connector_output`：`code/vllm/vllm/v1/outputs.py:262`
- `num_nans_in_logits`：`code/vllm/vllm/v1/outputs.py:266`
- `routed_experts`：`code/vllm/vllm/v1/outputs.py:272`

## 3. EngineCoreOutput

定义：`code/vllm/vllm/v1/engine/__init__.py:173`

它是 engine core 到 frontend/output processor 的单请求输出。

字段：

| 字段 | 含义 |
|---|---|
| `request_id` | 内部 request id |
| `new_token_ids` | 本次新输出 token ids |
| `new_logprobs` | 本次新 token logprobs |
| `new_prompt_logprobs_tensors` | prompt logprobs |
| `pooling_output` | pooling 输出 |
| `finish_reason` | finish reason，非 None 表示完成 |
| `stop_reason` | stop token id 或 stop string |
| `kv_transfer_params` | KV transfer 参数 |
| `prefill_stats` | prefill 统计 |
| `routed_experts` | 单请求 routed experts |

锚点：

- 类定义：`code/vllm/vllm/v1/engine/__init__.py:173`
- request id：`code/vllm/vllm/v1/engine/__init__.py:179`
- new token ids：`code/vllm/vllm/v1/engine/__init__.py:180`
- new logprobs：`code/vllm/vllm/v1/engine/__init__.py:182`
- prompt logprobs：`code/vllm/vllm/v1/engine/__init__.py:183`
- pooling：`code/vllm/vllm/v1/engine/__init__.py:185`
- finish reason：`code/vllm/vllm/v1/engine/__init__.py:187`
- stop reason：`code/vllm/vllm/v1/engine/__init__.py:188`
- finished 属性：`code/vllm/vllm/v1/engine/__init__.py:201`

## 4. EngineCoreOutputs

定义：`code/vllm/vllm/v1/engine/__init__.py:218`

它是一批输出，字段包括：

- `engine_index`；
- `outputs: list[EngineCoreOutput]`；
- `scheduler_stats`；
- `timestamp`；
- `utility_output`；
- `finished_requests`；
- `wave_complete`；
- `start_wave`。

字段锚点：`code/vllm/vllm/v1/engine/__init__.py:227`

## 5. update_from_output() 的第一步：读取 ModelRunnerOutput

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

开始会读取：

- sampled token ids；
- logprobs；
- prompt logprobs；
- pooler output；
- logits NaN 统计；
- KV connector output。

锚点：`code/vllm/vllm/v1/core/sched/scheduler.py:1468`

## 6. KV connector 输出处理

相关位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1490`

如果外部 KV block 加载失败，scheduler 会定位受影响请求，并决定重算或报错。

KV connector 输出还可能包含：

- finished sending / receiving；
- events；
- stats；
- invalid blocks。

对应结构定义可见：`code/vllm/vllm/v1/outputs.py:196`

## 7. routed experts 处理

相关位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1500`

处理逻辑：

1. 把 step 级 routed experts 存入 scheduler 侧 slot buffer；
2. 为每个请求构建当前 batch 中 routed experts 的 offset；
3. 后续生成 `EngineCoreOutput` 时带上单请求 routed experts。

## 8. 遍历本 step 调度请求

相关位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1526`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1542`

scheduler 会遍历本 step 实际调度过的请求，通过 `req_id_to_index` 找到 batch index，然后读取该请求的：

- `generated_token_ids`；
- prompt logprobs；
- pooler output；
- NaN 统计；
- routed experts offset。

## 9. speculative decoding 接受/拒绝统计

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1547`

如果本 step 有 scheduled spec tokens，scheduler 会根据实际生成 token 和 draft token 计算：

- accepted tokens；
- rejected tokens；
- bonus token 情况。

这些统计会进入 metrics。

## 10. 更新 request 输出 token 与 stop 状态

关键调用：`code/vllm/vllm/v1/core/sched/scheduler.py:1589`

底层函数：`_update_request_with_output()`，`code/vllm/vllm/v1/core/sched/scheduler.py:1848`

职责：

1. 把新生成 token append 到 `request.output_token_ids`；
2. 对每个 token 调用 `check_stop()`；
3. 如果中途 stop，截断多余 token；
4. 更新 request status 和 stop reason。

## 11. check_stop()

文件：`code/vllm/vllm/v1/core/sched/utils.py`

入口：`code/vllm/vllm/v1/core/sched/utils.py:94`

检查顺序：

1. 如果还没达到 `min_tokens`，不停止；
2. EOS token；
3. stop token ids；
4. max model len / max tokens；
5. repetition detection。

锚点：

- min tokens：`code/vllm/vllm/v1/core/sched/utils.py:100`
- EOS：`code/vllm/vllm/v1/core/sched/utils.py:103`
- stop token ids：`code/vllm/vllm/v1/core/sched/utils.py:108`
- max len/max tokens：`code/vllm/vllm/v1/core/sched/utils.py:112`
- repetition：`code/vllm/vllm/v1/core/sched/utils.py:119`

## 12. pooling 请求处理

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1593`

如果有 `pooler_output`，请求会直接设置为 `FINISHED_STOPPED`。

这类请求不走 token-by-token detokenize 输出，而是返回 pooling 数据。

## 13. structured output FSM 前进

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1598`

逻辑：

- 如果请求有新 token；
- 且 `structured_output_manager.should_advance(request)` 为真；
- 调用 grammar `accept_tokens()`；
- 如果 grammar 拒绝 token，置为 `FINISHED_ERROR`。

这一步和采样前的 grammar bitmask 是配套的：

- 采样前 bitmask 决定哪些 token 可选；
- 采样后 accept token 推进 grammar 状态。

## 14. finish reason 捕获与 stopped request 处理

位置：

- 获取 finish reason：`code/vllm/vllm/v1/core/sched/scheduler.py:1655`
- 处理 stopped request：`code/vllm/vllm/v1/core/sched/scheduler.py:1830`

流程：

1. request status 变成 finished 后，调用 `request.get_finished_reason()`；
2. 释放/更新 scheduler 内部状态；
3. 处理 resumable streaming request 的特殊情况；
4. 记录 finished request。

## 15. RequestStatus 到 FinishReason

文件：`code/vllm/vllm/v1/request.py`

入口：`code/vllm/vllm/v1/request.py:353`

映射：

| RequestStatus | FinishReason |
|---|---|
| `FINISHED_STOPPED` | `STOP` |
| `FINISHED_LENGTH_CAPPED` | `LENGTH` |
| `FINISHED_ABORTED` | `ABORT` |
| `FINISHED_IGNORED` | `LENGTH` |
| `FINISHED_ERROR` | `ERROR` |
| `WAITING_FOR_STREAMING_REQ` | `STOP` |
| `FINISHED_REPETITION` | `REPETITION` |

具体映射位置：`code/vllm/vllm/v1/request.py:357`

`FinishReason` 定义：`code/vllm/vllm/v1/engine/__init__.py:42`

可序列化为：

- `stop`；
- `length`；
- `abort`；
- `error`；
- `repetition`。

## 16. sample logprobs 提取

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1669`

条件：

```text
sampling_params is not None
sampling_params.num_logprobs is not None
model_runner_output.logprobs is not None
```

提取：

```python
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

锚点：`code/vllm/vllm/v1/core/sched/scheduler.py:1675`

这样可以处理 spec decode 中每个请求本 step 生成 token 数不同的情况。

## 17. 构造 EngineCoreOutput

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1688`

包含：

- new token ids；
- finish reason；
- sample logprobs；
- prompt logprobs；
- pooling output；
- stop reason；
- prefill stats；
- KV transfer 参数；
- routed experts。

## 18. 构造 EngineCoreOutputs

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1771`

scheduler 会按 `client_index` 分组构造 `EngineCoreOutputs`。

原因：不同请求可能来自不同 frontend client / data parallel client，需要输出到正确的接收方。

输出中还会包含：

- `finished_requests`；
- `scheduler_stats`；
- `timestamp`；
- wave 状态。

## 19. EngineCore.step() 中的位置

`EngineCore.step()`：`code/vllm/vllm/v1/engine/core.py:479`

典型顺序：

```text
scheduler.schedule()
  ↓
model_executor.execute_model()
  ↓
scheduler.get_grammar_bitmask()
  ↓
model_executor.sample_tokens(grammar_output)
  ↓
scheduler.update_from_output()
  ↓
EngineCoreOutputs
```

在不同 batch queue / async scheduling 配置下细节会变化，但核心是 scheduler 同时负责“调度输入”和“解释 worker 输出”。

## 20. 本篇小结

`ModelRunnerOutput` 是 worker 的原始输出，`EngineCoreOutputs` 是 scheduler 整理后的前台内部输出。中间最关键的是 `Scheduler.update_from_output()`，它负责：

- append token；
- 判断 stop；
- 更新 request status；
- 推进 structured output grammar；
- slice logprobs；
- 整理 pooling/KV/routed experts；
- 生成按 client 分组的 `EngineCoreOutputs`。
