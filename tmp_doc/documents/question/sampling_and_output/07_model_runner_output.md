# 07. ModelRunnerOutput 是什么？

源码位置：

- `vllm/vllm/v1/outputs.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/sample/output.py`
- `vllm/vllm/v1/worker/gpu/async_utils.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/executor/uniproc_executor.py`
- `vllm/vllm/v1/executor/multiproc_executor.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/__init__.py`
- `vllm/vllm/v1/engine/output_processor.py`

本问题关注：ModelRunner 执行完 forward / pooling / sampling 后，返回给 Scheduler 的 `ModelRunnerOutput` 包含哪些字段；它如何从 GPU tensor 转成跨进程可传输对象；它和 `SamplerOutput`、`AsyncModelRunnerOutput`、`EngineCoreOutput`、最终 `RequestOutput` 分别是什么关系。

---

## 1. 一句话回答

`ModelRunnerOutput` 是 worker / ModelRunner 返回 Scheduler 的“执行层结果”，不是用户最终看到的输出。

```text
ModelRunnerOutput 解决：
  这一轮模型执行在 worker 侧产生了什么内部结果？

EngineCoreOutput 解决：
  Scheduler 更新请求状态后，每个请求本轮应该向前端报告什么？

RequestOutput 解决：
  用户最终应该看到什么文本 / token / logprobs / finished 状态？
```

最小链路是：

```text
GPUModelRunner.execute_model()
  → forward / logits / pooling
  → sample_tokens()
  → ModelRunnerOutput 或 AsyncModelRunnerOutput
  → Executor 把 async wrapper 转成 ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

一句话记忆：

```text
ModelRunnerOutput 是“worker 给 Scheduler 的账单”，RequestOutput 才是“API 给用户的结果”。
```

---

## 2. ModelRunnerOutput 的定义

`ModelRunnerOutput` 定义在 `vllm/vllm/v1/outputs.py:233`。

源码注释说得很直接：

```text
ModelRunnerOutput is serialized and sent to the scheduler process.
This is expensive for torch.Tensor so prefer to use list instead.
```

位置：`vllm/vllm/v1/outputs.py:231`

这说明它有两个特点：

```text
1. 它是跨 worker / scheduler 边界传输的对象；
2. 它尽量避免直接携带大 torch.Tensor。
```

字段可以分成六类：

```text
请求映射字段：
  req_ids
  req_id_to_index

生成 token 字段：
  sampled_token_ids
  logprobs
  prompt_logprobs_dict

pooling 字段：
  pooler_output

connector 字段：
  kv_connector_output
  ec_connector_output

执行诊断字段：
  num_nans_in_logits
  cudagraph_stats

MoE / routed experts 字段：
  routed_experts
```

---

## 3. 字段逐个解释

### 3.1 req_ids

位置：`vllm/vllm/v1/outputs.py:235`

```text
req_ids: list[str]
```

表示本次 worker 输出对应的请求列表，顺序通常来自 `InputBatch.req_ids` 的拷贝。

在 `GPUModelRunner._bookkeeping_sync()` 里会复制一份：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3633`

```text
req_ids_output_copy = self.input_batch.req_ids.copy()
```

为什么要 copy？

```text
async scheduling 下，worker 的 InputBatch 很快会被下一步更新；
如果直接引用原对象，Scheduler 后面读到的可能已经不是本轮顺序。
```

### 3.2 req_id_to_index

位置：`vllm/vllm/v1/outputs.py:237`

```text
req_id_to_index: dict[str, int]
```

这是 `req_id -> batch index` 的映射。

Scheduler 消费时会用它把 `scheduler_output.num_scheduled_tokens` 中的请求映射回 ModelRunner 输出位置：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1543`

```text
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index]
```

### 3.3 sampled_token_ids

位置：`vllm/vllm/v1/outputs.py:240`

```text
sampled_token_ids: list[list[int]]
```

语义是：

```text
每个请求在当前 step 中新生成 / 接受的 token ids。
```

为什么是二维 list？

```text
普通 decode：每个请求通常 1 个 token；
spec decode / jump decoding：每个请求当前 step 可能接受多个 token；
某些请求可能因为 discard mask / chunked prefill 返回空列表。
```

所以形状不是固定 `[num_reqs, 1]`，而是：

```text
num_reqs x 当前 step 每请求动态生成 token 数
```

### 3.4 logprobs

位置：`vllm/vllm/v1/outputs.py:246`

```text
logprobs: LogprobsLists | None
```

这是采样 token 的 logprobs，已经是 CPU / numpy 友好的结构。

`Scheduler.update_from_output()` 会按请求切片：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1670`

```text
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

注意：

```text
logprobs 是“生成 token”的 logprobs；
prompt_logprobs_dict 是“prompt token”的 logprobs。
```

### 3.5 prompt_logprobs_dict

位置：`vllm/vllm/v1/outputs.py:251`

```text
prompt_logprobs_dict: dict[str, LogprobsTensors | None]
```

它是按请求 ID 存储的 prompt logprobs。

```text
req_id -> prompt logprobs tensors
```

为什么是 dict 而不是按 batch 排列？

```text
prompt logprobs 通常只在 prefill / chunked prefill 完成时返回；
不是每个 step、每个请求都有。
```

计算位置：`GPUModelRunner._get_prompt_logprobs_dict()`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:5461`

它会为需要 prompt logprobs 的请求维护一份 CPU tensor 缓冲：

```text
request.in_progress_prompt_logprobs_cpu
```

chunked prefill 时分块填充，最后一个 chunk 才把整份 prompt logprobs 放进 `prompt_logprobs_dict`。

### 3.6 pooler_output

位置：`vllm/vllm/v1/outputs.py:259`

```text
pooler_output: list[torch.Tensor | None] | None
```

这是 pooling / embedding 类模型的输出。

生成类模型通常不填它；pooling 模型会在 `_pool()` 中构造。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3350`

### 3.7 kv_connector_output

位置：`vllm/vllm/v1/outputs.py:262`

```text
kv_connector_output: KVConnectorOutput | None
```

它携带 KV transfer / offload / connector 侧的 worker 输出。

`KVConnectorOutput` 定义在 `vllm/vllm/v1/outputs.py:196`，字段包括：

```text
finished_sending
finished_recving
kv_connector_stats
kv_cache_events
kv_connector_worker_meta
invalid_block_ids
expected_finished_count
```

Scheduler 会用它更新 KV transfer 状态：

位置：`vllm/vllm/v1/core/sched/scheduler.py:2418`

```text
_update_from_kv_xfer_finished(kv_connector_output)
```

### 3.8 ec_connector_output

位置：`vllm/vllm/v1/outputs.py:264`

```text
ec_connector_output: ECConnectorOutput | None
```

这是 encoder-cache / encoder connector 相关输出。

`ECConnectorOutput` 定义在 `vllm/vllm/v1/outputs.py:224`：

```text
finished_sending
finished_recving
```

它和 `kv_connector_output` 不是同一个东西：

```text
KV connector：关注 decoder KV cache 的 transfer / offload / load；
EC connector：关注 encoder / multimodal encoder cache 相关传输。
```

### 3.9 num_nans_in_logits

位置：`vllm/vllm/v1/outputs.py:266`

```text
num_nans_in_logits: dict[str, int] | None
```

语义是：

```text
req_id -> 当前 logits 中 NaN 数量
```

它由 `_get_nans_in_logits()` 计算，受环境变量 `VLLM_COMPUTE_NANS_IN_LOGITS` 控制。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:5565`

Scheduler 会写回 request：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1678`

```text
request.num_nans_in_logits = num_nans_in_logits[req_id]
```

### 3.10 cudagraph_stats

位置：`vllm/vllm/v1/outputs.py:269`

```text
cudagraph_stats: CUDAGraphStat | None
```

用于把本 step 的 CUDA graph 执行统计带回 Scheduler，再进入 stats。

Scheduler 在 `make_stats()` 时会消费它。

位置：`vllm/vllm/v1/core/sched/scheduler.py:1791`

### 3.11 routed_experts

位置：`vllm/vllm/v1/outputs.py:272`

```text
routed_experts: RoutedExpertsLists | None
```

这是 MoE routed experts 的 step 级别数据。

注意它不是按请求直接存的，而是：

```text
routing_data: (num_scheduled_tokens, num_layers, num_experts_per_tok)
slot_mapping: (num_scheduled_tokens,)
```

Scheduler 先把它持久化到 slot buffer：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1508`

```text
self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
```

然后再按请求 / token 切出 `EngineCoreOutput.routed_experts`。

---

## 4. LogprobsTensors 和 LogprobsLists

`outputs.py` 里同时定义了两个 logprobs 结构。

### 4.1 LogprobsTensors

位置：`vllm/vllm/v1/outputs.py:52`

```text
LogprobsTensors(
  logprob_token_ids: torch.Tensor,
  logprobs: torch.Tensor,
  selected_token_ranks: torch.Tensor,
  cu_num_generated_tokens: list[int] | None,
)
```

它偏 worker / GPU 内部使用，支持：

```text
to_cpu_nonblocking()
tolists()
filter(mask)
empty_cpu(...)
```

### 4.2 LogprobsLists

位置：`vllm/vllm/v1/outputs.py:27`

```text
LogprobsLists(
  logprob_token_ids: np.ndarray,
  logprobs: np.ndarray,
  sampled_token_ranks: np.ndarray,
  cu_num_generated_tokens: list[int] | None,
)
```

它偏 Scheduler / OutputProcessor 使用，适合跨进程传输。

### 4.3 为什么要分两种

可以理解成：

```text
LogprobsTensors：worker 侧计算态；
LogprobsLists：scheduler / frontend 侧消费态。
```

`ModelRunnerOutput.logprobs` 使用的是 `LogprobsLists`，而 `prompt_logprobs_dict` 当前字段类型仍是 `LogprobsTensors | None`，因为 prompt logprobs 的中间管理需要按请求保留 tensor 结构。

---

## 5. SamplerOutput 和 ModelRunnerOutput 的区别

### 5.1 SamplerOutput 是采样器的输出

GPU worker 主链路使用的 `SamplerOutput` 定义在：

位置：`vllm/vllm/v1/worker/gpu/sample/output.py:10`

字段是：

```text
sampled_token_ids: torch.Tensor
logprobs_tensors: LogprobsTensors | None
num_nans: torch.Tensor | None
num_sampled: torch.Tensor | None
num_rejected: torch.Tensor | None
```

它更接近 GPU sampler 的原始产物。

### 5.2 ModelRunnerOutput 是 worker 对 Scheduler 的输出

`ModelRunnerOutput` 会在 `SamplerOutput` 的基础上补齐：

```text
req_ids
req_id_to_index
prompt_logprobs_dict
kv_connector_output
ec_connector_output
cudagraph_stats
routed_experts
pooler_output
```

所以关系是：

```text
SamplerOutput
  → bookkeeping / prompt logprobs / connector / stats / routed experts
  → ModelRunnerOutput
```

### 5.3 outputs.py 里也有一个 SamplerOutput

`vllm/vllm/v1/outputs.py:185` 也定义了一个更轻量的 `SamplerOutput`：

```text
sampled_token_ids
logprobs_tensors
```

但 GPUModelRunner 主采样链路使用的是：

```text
vllm/vllm/v1/worker/gpu/sample/output.py
```

所以读源码时不要只按类名跳转，要看 import 来源。

---

## 6. execute_model() 和 sample_tokens() 如何分工

### 6.1 execute_model() 通常不直接返回 generation 输出

在生成模型路径中，`GPUModelRunner.execute_model()` 做完 forward / logits 后，会把状态保存到：

```text
self.execute_model_state
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4389`

保存内容包括：

```text
scheduler_output
logits
spec_decode_metadata
hidden_states
sample_hidden_states
aux_hidden_states
ec_connector_output
cudagraph_stats
slot_mappings
```

然后：

```text
return None
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4408`

原因是：

```text
grammar bitmask / sampling / prompt logprobs / final output assembly
需要在 sample_tokens() 阶段完成。
```

### 6.2 sample_tokens() 接住 execute_model_state

`sample_tokens()` 入口在：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4426`

它先判断：

```text
if self.execute_model_state is None:
    return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4429`

这通常发生在：

```text
- 本 rank 没有 generation 状态；
- 非 last PP rank；
- 只有 KV connector 状态需要往回传。
```

如果有 `execute_model_state`，则：

```text
1. 解包 forward 保存的 logits / hidden states / metadata；
2. 应用 grammar bitmask；
3. 调 _sample() 得到 SamplerOutput；
4. 更新 worker 侧 InputBatch / request 状态；
5. 计算 prompt logprobs；
6. 处理 spec decode draft token；
7. 取 kv_connector_output；
8. 构造 ModelRunnerOutput；
9. 同步路径直接返回，异步路径返回 AsyncGPUModelRunnerOutput。
```

---

## 7. GPUModelRunner 如何构造 ModelRunnerOutput

### 7.1 _bookkeeping_sync() 准备请求映射和 token 结果

`_bookkeeping_sync()` 负责把采样后的 GPU 结果整理成 scheduler 可消费的形态。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3604`

它返回：

```text
num_nans_in_logits
logprobs_lists
valid_sampled_token_ids
prompt_logprobs_dict
req_ids_output_copy
req_id_to_index_output_copy
invalid_req_indices
```

其中：

```text
valid_sampled_token_ids：同步路径中已经是 list[list[int]]；
invalid_req_indices：异步路径中用来告诉 AsyncGPUModelRunnerOutput 哪些请求要清空 token；
prompt_logprobs_dict：本 step 完成的 prompt logprobs。
```

### 7.2 discard_request_mask 会清空不应该返回的 sampled tokens

在 chunked prefill / spec decode 等场景下，某些请求虽然参与 batch，但本轮 sampled token 不应该返回。

`_bookkeeping_sync()` 会找出：

```text
discard_sampled_tokens_req_indices
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3624`

同步路径会直接清空对应 token list：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3663`

```text
valid_sampled_token_ids[int(i)].clear()
```

异步路径则把这些 index 传给 `AsyncGPUModelRunnerOutput`，由 `get_output()` 后处理。

### 7.3 构造 ModelRunnerOutput

最终构造发生在：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4612`

```text
ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    kv_connector_output=kv_connector_output,
    ec_connector_output=ec_connector_output if supports_mm_inputs else None,
    num_nans_in_logits=num_nans_in_logits,
    cudagraph_stats=cudagraph_stats,
    routed_experts=None,
)
```

### 7.4 同步 scheduling 直接返回 ModelRunnerOutput

如果没有启用 async scheduling：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4628`

```text
return output
```

如果启用了 routed experts，同步路径会在返回前把 routed experts 的 CPU buffer 包成 `RoutedExpertsLists`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4629`

### 7.5 async scheduling 返回 AsyncGPUModelRunnerOutput

如果启用 async scheduling，GPUModelRunner 不立即把所有 GPU tensor 同步成 Python list，而是构造：

```text
AsyncGPUModelRunnerOutput(...)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4666`

它持有：

```text
ModelRunnerOutput 初稿
sampled_token_ids GPU tensor
logprobs_tensors GPU/CPU tensor
invalid_req_indices
async output copy stream
vocab_size
routed_experts snapshot
```

然后返回这个 async wrapper。

---

## 8. AsyncModelRunnerOutput 是什么

### 8.1 抽象基类

`AsyncModelRunnerOutput` 定义在：

位置：`vllm/vllm/v1/outputs.py:298`

它只有一个方法：

```text
get_output() -> ModelRunnerOutput
```

含义是：

```text
先把 GPU→CPU copy 和后处理挂起；
等 Executor / Scheduler 真要消费时，再阻塞拿到真正 ModelRunnerOutput。
```

### 8.2 当前 GPUModelRunner 使用的实现

当前主实现定义在 `gpu_model_runner.py`：

| 类 | 位置 | 用途 |
|---|---|---|
| `AsyncGPUModelRunnerOutput` | `vllm/vllm/v1/worker/gpu_model_runner.py:242` | generation 输出异步拷贝 |
| `AsyncGPUPoolingModelRunnerOutput` | `vllm/vllm/v1/worker/gpu_model_runner.py:367` | pooling 输出异步拷贝 |

`AsyncGPUModelRunnerOutput.get_output()` 会：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:285`

```text
1. 等待 async_copy_ready_event；
2. 把 sampled_token_ids_cpu 转成 list[list[int]]；
3. 对 invalid_req_indices 清空 token；
4. 如果 max_gen_len > 1，用 RejectionSampler.parse_output 解析 spec decode 输出；
5. 把 logprobs_tensors 转成 LogprobsLists；
6. 把 routed experts 转成 RoutedExpertsLists；
7. 回填到原始 ModelRunnerOutput。
```

### 8.3 Executor 会把 async wrapper 转成 ModelRunnerOutput

`UniProcExecutor.collective_rpc()` 中，如果结果是 `AsyncModelRunnerOutput`，会调用：

位置：`vllm/vllm/v1/executor/uniproc_executor.py:91`

```text
result = result.get_output()
```

非阻塞路径会用 `AsyncOutputFuture` 包装，在 `future.result()` 时调用 `get_output()`。

位置：`vllm/vllm/v1/executor/uniproc_executor.py:26`

`MultiprocExecutor` 的 worker 输出入队前也会做同样转换：

位置：`vllm/vllm/v1/executor/multiproc_executor.py:936`

```text
if isinstance(output, AsyncModelRunnerOutput):
    output = output.get_output()
```

原因是：

```text
AsyncModelRunnerOutput 持有 CUDA event / stream / device tensor，不能直接跨进程序列化。
```

---

## 9. pooling 模型的 ModelRunnerOutput

pooling 模型不走 sampled token 输出，而是走 `_pool()`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3350`

它会构造：

```text
ModelRunnerOutput(
    req_ids=input_batch.req_ids.copy(),
    req_id_to_index=input_batch.req_id_to_index.copy(),
    kv_connector_output=kv_connector_output,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3385`

然后填充：

```text
pooler_output
```

如果当前没有任何完成的 pooling 请求：

```text
pooler_output = [None] * num_reqs
```

如果在 CUDA 平台上需要异步 D2H，则返回 `AsyncGPUPoolingModelRunnerOutput`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3404`

Scheduler 收到后会在 `update_from_output()` 中根据：

```text
pooler_outputs[req_index]
```

处理 pooling request。

位置：`vllm/vllm/v1/core/sched/scheduler.py:1584`

---

## 10. EMPTY_MODEL_RUNNER_OUTPUT 和 with_kv_conn_output_only

### 10.1 EMPTY_MODEL_RUNNER_OUTPUT

定义在：

位置：`vllm/vllm/v1/outputs.py:348`

```text
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
```

它是合法的空输出对象，不是 `None`。

常见场景：

```text
- 本轮没有 scheduled tokens；
- 没有请求级输出，但接口仍需要返回 ModelRunnerOutput；
- connector-only 路径没有有效 connector output。
```

### 10.2 with_kv_conn_output_only

定义在：

位置：`vllm/vllm/v1/outputs.py:283`

逻辑是：

```text
if kv_connector_output is None or kv_connector_output.is_empty():
    return EMPTY_MODEL_RUNNER_OUTPUT
else:
    output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
    output.kv_connector_output = kv_connector_output
    return output
```

它用于：

```text
没有 sampled token / pooling output，
但需要把 KV connector 状态带回 Scheduler。
```

`sample_tokens()` 在没有 `execute_model_state` 时就会走这条路径。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4429`

### 10.3 make_empty_encoder_model_runner_output

`outputs.py` 还有一个 encoder 相关 helper：

位置：`vllm/vllm/v1/outputs.py:318`

```text
make_empty_encoder_model_runner_output(scheduler_output)
```

它会构造带有请求映射、空 token、空 pooling placeholder 的 stub output，用于 encoder / EC connector 相关路径。

---

## 11. Scheduler 如何消费 ModelRunnerOutput

### 11.1 update_from_output 是核心消费入口

位置：`vllm/vllm/v1/core/sched/scheduler.py:1464`

输入是：

```text
scheduler_output: SchedulerOutput
model_runner_output: ModelRunnerOutput
```

它先取出：

```text
sampled_token_ids
logprobs
prompt_logprobs_dict
pooler_outputs
num_nans_in_logits
kv_connector_output
cudagraph_stats
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1469`

### 11.2 先处理 connector / routed experts 等 step 级信息

如果 KV connector 返回 invalid blocks：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1491`

```text
_handle_invalid_blocks(...)
```

如果有 routed experts：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1508`

```text
routed_experts_mgr.store_batch(...)
```

这一步必须早于 per-request 输出构造，因为某些请求可能在本 step 生成 token 后立刻停止，仍需要读取本 step 的 routed experts。

### 11.3 再按请求生成 EngineCoreOutput

Scheduler 会遍历：

```text
scheduler_output.num_scheduled_tokens.items()
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1527`

对每个 req：

```text
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index]
```

然后：

```text
1. 处理 spec decode 接受 / 拒绝 token；
2. 释放 encoder inputs；
3. 调 _update_request_with_output() 更新 request 输出 token；
4. 检查 stop / finish；
5. 切 sample logprobs；
6. 取 prompt_logprobs_dict[req_id]；
7. 构造 EngineCoreOutput。
```

构造 `EngineCoreOutput` 的位置：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1689`

关键字段映射是：

```text
ModelRunnerOutput.sampled_token_ids[req_index]
  → EngineCoreOutput.new_token_ids

ModelRunnerOutput.logprobs.slice_request(...)
  → EngineCoreOutput.new_logprobs

ModelRunnerOutput.prompt_logprobs_dict.get(req_id)
  → EngineCoreOutput.new_prompt_logprobs_tensors

ModelRunnerOutput.pooler_output[req_index]
  → EngineCoreOutput.pooling_output

Request 状态 stop 检查结果
  → EngineCoreOutput.finish_reason / stop_reason

Scheduler free request 结果
  → EngineCoreOutput.kv_transfer_params
```

### 11.4 最后按 client 聚合成 EngineCoreOutputs

Scheduler 会把多个 `EngineCoreOutput` 按 `request.client_index` 聚合：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1770`

```text
client_index -> EngineCoreOutputs(outputs=[...])
```

如果本 step 有 finished request ids 或 stats，也会挂到 `EngineCoreOutputs`。

位置：`vllm/vllm/v1/core/sched/scheduler.py:1777`

---

## 12. EngineCoreOutput / EngineCoreOutputs 是什么

### 12.1 EngineCoreOutput 是单请求输出

定义在：

位置：`vllm/vllm/v1/engine/__init__.py:175`

关键字段：

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
trace_headers
prefill_stats
routed_experts
num_nans_in_logits
```

它已经比 `ModelRunnerOutput` 更接近前端语义：

```text
- 已经是单请求维度；
- 已经包含 finish_reason / stop_reason；
- 已经带 request events / prefill stats；
- 已经经过 Scheduler request 状态更新。
```

### 12.2 EngineCoreOutputs 是按 client 聚合的输出包

定义在：

位置：`vllm/vllm/v1/engine/__init__.py:220`

字段包括：

```text
engine_index
outputs: list[EngineCoreOutput]
scheduler_stats
finished_requests
utility_output
wave_complete
start_wave
```

DP / 多 API server 场景下，不同 client 的 outputs 会被分开返回。

---

## 13. OutputProcessor 如何变成用户输出

`OutputProcessor.process_outputs()` 消费的是 `EngineCoreOutput`，不是 `ModelRunnerOutput`。

位置：`vllm/vllm/v1/engine/output_processor.py:576`

核心步骤：

```text
1. 根据 request_id 找 RequestState；
2. 更新 stats；
3. 对 new_token_ids 做 detokenize；
4. 处理 stop string；
5. 用 LogprobsProcessor 处理 sample / prompt logprobs；
6. 调 RequestState.make_request_output()；
7. 产出 RequestOutput / PoolingRequestOutput。
```

`RequestState.make_request_output()` 里才会真正处理：

```text
FINAL_ONLY / DELTA 输出模式
stream_interval
CompletionOutput
PoolingOutput
RequestOutput
PoolingRequestOutput
```

位置：`vllm/vllm/v1/engine/output_processor.py:272`

所以分层关系是：

```text
ModelRunnerOutput：worker → scheduler
EngineCoreOutput：scheduler → engine frontend
RequestOutput：output processor → user/API
```

---

## 14. 为什么 ModelRunnerOutput 不是最终输出

可以从职责上拆开：

```text
1. 它只有 token ids，不负责 detokenize。
2. 它没有完整用户文本 delta。
3. 它不负责 stop string 检测。
4. 它不直接表达 API 输出模式，例如 DELTA / FINAL_ONLY。
5. 它的 logprobs 仍是内部结构，不是最终 CompletionOutput 结构。
6. 它包含执行层专用字段，例如 KV connector、cudagraph、routed experts。
7. 它仍需要 Scheduler 根据 Request 状态决定 finish_reason / stop_reason。
8. 它可能是 pooling / connector-only / empty output，而不是 generation output。
```

最终输出还必须经过：

```text
Scheduler.update_from_output()
  → request 状态推进 / stop 检查 / EngineCoreOutput

OutputProcessor.process_outputs()
  → detokenize / logprobs processor / RequestOutput
```

---

## 15. Pipeline Parallel 和 ModelRunnerOutput

### 15.1 非 last PP rank 不产生最终 sampled token 输出

`GPUWorker.execute_model()` 中，如果 ModelRunner 返回的是 `IntermediateTensors`，说明当前不是最后 PP stage，它会把中间张量发送给下一个 PP rank。

位置：`vllm/vllm/v1/worker/gpu_worker.py:895`

这时对用户侧没有 `ModelRunnerOutput`。

### 15.2 sample_tokens() 中的 connector-only 输出

如果当前 rank 没有 `execute_model_state`，`sample_tokens()` 返回：

```text
ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4429`

这让非 last PP rank 或无采样 rank 仍能把 KV connector 状态带回 Scheduler / aggregator，而不伪造 token 输出。

### 15.3 PP 下 sampled tokens 可能需要广播

async scheduling + PP 场景下，last PP rank 采样后可能需要把 sampled token ids 广播给其他 rank，用于下一轮输入准备。

相关逻辑在：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4467`

```text
_pp_broadcast_prev_sampled_token_ids(...)
```

这属于 worker 内部状态同步，不表示所有 PP rank 都会产生用户输出。

---

## 16. Async scheduling 下的特殊点

### 16.1 为什么要 req_ids / req_id_to_index copy

async scheduling 会让下一步输入准备与当前步输出拷贝重叠。

所以 `_bookkeeping_sync()` 明确 copy：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3633`

```text
req_ids_output_copy
req_id_to_index_output_copy
```

否则 Scheduler 消费 output 时，worker 的 batch 结构可能已经变了。

### 16.2 sampled_token_ids 先留在 GPU

async 路径中：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3678`

```text
valid_sampled_token_ids = []
invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
input_batch.prev_sampled_token_ids = sampled_token_ids
```

这样下一步输入准备可以直接使用 GPU 上的 sampled token，减少 CPU 同步。

最终返回给 Scheduler 的 list 版本由 `AsyncGPUModelRunnerOutput.get_output()` 生成。

### 16.3 routed experts 必须 clone 快照

异步路径下 routed experts 的源 buffer 会在下一步被复用，所以构造 async output 时会 clone：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4643`

```text
routing_data=buf[:total].clone()
slot_mapping=...clone()
```

否则 copy stream 可能读到被下一步覆盖的数据。

---

## 17. KV connector output 的流转

### 17.1 worker 侧写入 ModelRunnerOutput

`sample_tokens()` 最后取：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4608`

```text
kv_connector_output = self.kv_connector_output
self.kv_connector_output = None
```

然后写入 `ModelRunnerOutput.kv_connector_output`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4619`

### 17.2 scheduler 侧更新 connector 状态

Scheduler 在 `update_from_output()` 末尾处理：

位置：`vllm/vllm/v1/core/sched/scheduler.py:1732`

```text
if kv_connector_output:
    self._update_from_kv_xfer_finished(kv_connector_output)
```

`_update_from_kv_xfer_finished()` 会处理：

```text
finished_recving：请求可以从 WAITING_FOR_REMOTE_KVS 继续推进；
finished_sending：发送完成后可释放 blocks；
invalid_block_ids：前面已触发相关请求重算或报错。
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2418`

---

## 18. 最终可以记成一张表

| 对象 | 产生位置 | 消费位置 | 粒度 | 主要内容 |
|---|---|---|---|---|
| `SamplerOutput` | sampler / rejection sampler | GPUModelRunner bookkeeping | batch tensor | sampled ids、logprobs tensors、NaN、采样数量 |
| `ModelRunnerOutput` | GPUModelRunner / worker | Scheduler.update_from_output | batch / request 映射 | sampled ids、logprobs、pooler、connector、stats、routed experts |
| `AsyncModelRunnerOutput` | GPUModelRunner async 路径 | Executor / Future `get_output()` | wrapper | 延迟 GPU→CPU copy，最终产出 ModelRunnerOutput |
| `EngineCoreOutput` | Scheduler | OutputProcessor | 单请求 | new tokens、finish reason、stop reason、events、logprobs |
| `EngineCoreOutputs` | Scheduler / EngineCore | Engine client / frontend | client 聚合 | 多个 EngineCoreOutput、stats、finished set、DP wave 信号 |
| `RequestOutput` | OutputProcessor | 用户 / API | 单请求 API 输出 | 文本、token、logprobs、finished、metrics |

---

## 19. 主链路复盘

生成模型路径：

```text
EngineCore.step()
  → Scheduler.schedule()
  → Executor.execute_model(SchedulerOutput)
  → GPUWorker.execute_model()
  → GPUModelRunner.execute_model()
      → _update_states()
      → _prepare_inputs()
      → _build_attention_metadata()
      → _model_forward()
      → compute_logits()
      → 保存 ExecuteModelState
      → return None

EngineCore.step()
  → 如果 model_output is None
  → Executor.sample_tokens(grammar_output)
  → GPUModelRunner.sample_tokens()
      → apply_grammar_bitmask()
      → _sample()
      → _update_states_after_model_execute()
      → _bookkeeping_sync()
      → _get_prompt_logprobs_dict()
      → finalize kv connector / eplb
      → ModelRunnerOutput
      → 同步路径直接返回
      → async 路径返回 AsyncGPUModelRunnerOutput

Executor
  → 如有 AsyncModelRunnerOutput，调用 get_output()
  → 得到真正 ModelRunnerOutput

Scheduler.update_from_output()
  → 更新 request 状态
  → 生成 EngineCoreOutput / EngineCoreOutputs

OutputProcessor.process_outputs()
  → detokenize / logprobs / stop string / streaming delta
  → RequestOutput / PoolingRequestOutput
```

Pooling 模型路径：

```text
GPUModelRunner.execute_model()
  → _model_forward()
  → _pool()
  → ModelRunnerOutput(pooler_output=...)
  → Scheduler.update_from_output()
  → EngineCoreOutput(pooling_output=...)
  → OutputProcessor
  → PoolingRequestOutput
```

Connector-only 路径：

```text
没有 token 输出 / 非 last PP rank / connector-only
  → ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
  → Scheduler 只推进 KV connector 状态或返回 empty outputs
```

---

## 20. 容易混淆的点

### 20.1 ModelRunnerOutput 不是用户输出

它不负责 detokenize、stop string、streaming delta，也不是 OpenAI API 响应结构。

### 20.2 EMPTY_MODEL_RUNNER_OUTPUT 不是 None

`None` 在 `execute_model()` 中常表示：

```text
forward 已经完成，但需要 sample_tokens() 继续构造输出。
```

`EMPTY_MODEL_RUNNER_OUTPUT` 则是合法的空结果对象。

### 20.3 SamplerOutput 不是 ModelRunnerOutput

`SamplerOutput` 是 sampler 的 GPU 结果；`ModelRunnerOutput` 是 worker 返回 Scheduler 的跨进程结果。

### 20.4 AsyncModelRunnerOutput 不能跨进程长期传输

它持有 CUDA event / stream / tensor，Executor 会在合适边界调用 `get_output()`，把它变成普通 `ModelRunnerOutput`。

### 20.5 prompt logprobs 和 sample logprobs 不在同一个字段

```text
sample logprobs：ModelRunnerOutput.logprobs
prompt logprobs：ModelRunnerOutput.prompt_logprobs_dict
```

### 20.6 routed_experts 是 step 级，不是 request 级

`ModelRunnerOutput.routed_experts` 是按本 step token 和 slot mapping 存的；Scheduler 再按请求切分成 `EngineCoreOutput.routed_experts`。

### 20.7 connector output 可能是唯一有效内容

有些 step 没有 sampled token，也没有 pooling output，但仍需要把 `kv_connector_output` 带回 Scheduler。

---

## 21. 一句话总结

`ModelRunnerOutput` 是 vLLM V1 执行层闭环的输出对象：

```text
它把 worker 本轮执行产生的 sampled token、logprobs、pooling output、KV/EC connector 状态、CUDA graph 统计、routed experts 等内部结果交给 Scheduler；Scheduler 再结合请求状态，把它转换成单请求的 EngineCoreOutput；最后 OutputProcessor 才把 token ids 变成用户可见的 RequestOutput。
```

如果只记住一句话：

```text
ModelRunnerOutput 是“模型执行结果”，EngineCoreOutput 是“调度后单请求结果”，RequestOutput 是“用户可见结果”。
```
