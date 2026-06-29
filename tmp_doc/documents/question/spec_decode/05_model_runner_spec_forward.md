# 05. ModelRunner 如何执行 spec decode forward？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_input_batch.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\spec_decode\metadata.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\rejection_sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`

本问题关注：`SchedulerOutput.scheduled_spec_decode_tokens` 到达 `GPUModelRunner` 后，ModelRunner 如何把 draft tokens 写入 batch、准备 input ids / positions / logits indices、构造 `SpecDecodeMetadata` 和 attention metadata、执行 target model forward、计算 target / bonus logits，并把 `spec_decode_metadata`、hidden states、attention metadata 暂存在 `ExecuteModelState` 中，交给后续 `sample_tokens()` 做 rejection sampling 和下一轮 draft proposal。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的文档风格，本篇按“先定角色，再走 execute_model 主链路，再拆 spec forward 的关键阶段，最后总结边界”的方式梳理 ModelRunner 侧 spec decode forward。

要回答的问题分成 12 组：

```text
1. ModelRunner 在 spec decode forward 里负责什么？
2. spec decode 在 GPUModelRunner 初始化时准备了哪些运行态？
3. ExecuteModelState 为什么是 execute_model() 和 sample_tokens() 的桥？
4. _update_states() 如何把 scheduled_spec_decode_tokens 落到 InputBatch？
5. _prepare_inputs() 如何把 draft tokens 变成 input_ids / positions / logits_indices？
6. _calc_spec_decode_metadata() 如何描述 target logits 和 bonus logits 位置？
7. _prepare_input_ids() 在 async scheduling 下如何 scatter sampled / draft tokens？
8. _build_attention_metadata() 如何为 spec decode 准备 attention metadata？
9. execute_model() 如何完成 target model forward 和 compute_logits？
10. 为什么 execute_model() 返回 None，并把状态留给 sample_tokens()？
11. sample_tokens() 如何消费 forward 状态并切入 RejectionSampler？
12. PP、Mamba、KV connector、structured output、multimodal 在 spec forward 中有哪些交互？
```

阅读顺序建议：

```text
03_scheduler_spec_decode_flow.md
  → 04_spec_decode_metadata.md
  → 05_model_runner_spec_forward.md
  → 06_rejection_sampler_flow.md
  → 09_output_recovery_and_scheduler_update.md
```

本篇重点是 `GPUModelRunner.execute_model()` 中的 target model verification forward，不展开每一种 drafter / proposer 的内部算法；那些属于 draft proposal 专题。

---

## 1. 一句话回答

`ModelRunner` 在 spec decode forward 中负责把 Scheduler 侧的 draft tokens 转换成 target model 可以一次性验证的 forward 输入和 logits 布局。

主线是：

```text
SchedulerOutput.scheduled_spec_decode_tokens
  → GPUModelRunner._update_states()
  → InputBatch.spec_token_ids / token_ids_cpu
  → GPUModelRunner._prepare_inputs()
  → SpecDecodeMetadata / logits_indices
  → GPUModelRunner._build_attention_metadata(use_spec_decode=True)
  → set_forward_context(...)
  → _model_forward()
  → hidden_states[logits_indices]
  → model.compute_logits()
  → ExecuteModelState(spec_decode_metadata=...)
  → sample_tokens()
```

它负责：

```text
1. 接收 SchedulerOutput 中本轮要验证的 draft tokens；
2. 更新 worker-local CachedRequestState 和 InputBatch；
3. 生成本轮 target forward 的 input_ids、positions、slot mappings；
4. 计算 spec decode 需要的 logits_indices；
5. 构造 SpecDecodeMetadata，描述 target / bonus logits 位置；
6. 构造 attention metadata，支持普通 attention、Mamba/GDN、drafter common metadata；
7. 执行 target model forward；
8. 只对 logits_indices 指向的 hidden states 计算 logits；
9. 把 logits、metadata、hidden states 暂存在 ExecuteModelState；
10. 等 sample_tokens() 使用 RejectionSampler 做接受 / 拒绝。
```

它不负责：

```text
1. 决定本轮验证哪些 draft tokens，这是 Scheduler 的事；
2. 直接更新 Scheduler 侧 Request.output_token_ids，这是 Scheduler.update_from_output() 的事；
3. 在 execute_model() 里完成 rejection sampling；
4. 把最终 token detokenize 成用户输出。
```

一句话压缩：

```text
ModelRunner spec forward 的核心，是把“待验证 draft tokens”变成“target logits rows + SpecDecodeMetadata”。
```

---

## 2. 整体流程图

```text
EngineCore.step()
  → Executor.execute_model(scheduler_output)
  → Worker.execute_model(scheduler_output)
  → GPUModelRunner.execute_model(scheduler_output)
      │
      ├─ _update_states()
      │    ├─ 删除 finished requests
      │    ├─ 添加 scheduled_new_reqs
      │    ├─ 更新 scheduled_cached_reqs
      │    ├─ 读取 scheduler_output.scheduled_spec_decode_tokens
      │    └─ InputBatch.update_req_spec_token_ids()
      │
      ├─ _prepare_inputs()
      │    ├─ commit block table
      │    ├─ req_indices / cu_num_tokens
      │    ├─ positions / token_indices
      │    ├─ input_ids.cpu ← token_ids_cpu
      │    ├─ optimistic seq_lens
      │    ├─ async spec decode num_computed_tokens 修正
      │    ├─ slot mapping
      │    ├─ _prepare_input_ids()
      │    ├─ num_draft_tokens
      │    ├─ _calc_spec_decode_metadata()
      │    └─ logits_indices
      │
      ├─ _determine_batch_execution_and_padding()
      ├─ _get_slot_mappings()
      ├─ _build_attention_metadata(use_spec_decode=True)
      │    ├─ CommonAttentionMetadata
      │    ├─ per-layer AttentionMetadata
      │    ├─ num_accepted_tokens / num_decode_draft_tokens_cpu
      │    └─ spec_decode_common_attn_metadata
      │
      ├─ _preprocess()
      │    ├─ input_ids / inputs_embeds
      │    ├─ positions
      │    ├─ model_kwargs
      │    └─ intermediate_tensors
      │
      ├─ set_forward_context(...)
      ├─ _model_forward()
      ├─ hidden_states[logits_indices]
      ├─ model.compute_logits(sample_hidden_states)
      ├─ ExecuteModelState(...)
      └─ return None
           ↓
      GPUModelRunner.sample_tokens(grammar_output)
        ├─ apply_grammar_bitmask()
        ├─ _sample(logits, spec_decode_metadata)
        │    └─ RejectionSampler.forward(...)
        ├─ _update_states_after_model_execute()
        ├─ propose_draft_token_ids()
        └─ ModelRunnerOutput
```

---

## 3. 初始化阶段：spec forward 需要哪些运行态

`GPUModelRunner.__init__()` 保存 speculative 配置：

```python
self.speculative_config = vllm_config.speculative_config
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:426` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:436`

如果开启 spec decode 且当前是 last PP rank，会创建 drafter 和 rejection sampler：

```python
if self.speculative_config and get_pp_group().is_last_rank:
    ...
    self.drafter = ...
    self.rejection_sampler = RejectionSampler(
        self.sampler, self.speculative_config, self.device
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:541` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:620`

这里和 forward 直接相关的是：

```text
1. rejection_sampler：sample_tokens() 中验证 draft tokens；
2. drafter：sample_tokens() 后基于 forward hidden states 生成下一轮 draft tokens；
3. use_aux_hidden_state_outputs：EAGLE3 / DFlash / extract_hidden_states 等需要额外 hidden states；
4. num_spec_tokens / prev_num_spec_tokens：spec token 数量上限和 async 输入 scatter 需要；
5. use_async_spec_decode：async scheduling + spec decode 的组合开关。
```

相关字段：

```python
self.num_spec_tokens = 0
self.prev_num_spec_tokens = 0
...
if self.speculative_config:
    self.num_spec_tokens = self.speculative_config.num_speculative_tokens
    self.prev_num_spec_tokens = self.num_spec_tokens
...
self.use_async_spec_decode = (
    self.use_async_scheduling and self.num_spec_tokens > 0
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:622` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:635`

---

## 4. ExecuteModelState：execute_model 和 sample_tokens 的桥

`ExecuteModelState` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:402`

```python
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:402` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:416`

它的作用是：

```text
execute_model() 负责 forward / logits；
sample_tokens() 负责 grammar bitmask / sampling / rejection sampling / drafter；
ExecuteModelState 把这两步之间必须共享的临时状态保存下来。
```

spec decode 下尤其需要保存：

```text
spec_decode_metadata：RejectionSampler 需要的 target / bonus logits 布局；
spec_decode_common_attn_metadata：drafter 生成下一轮 draft tokens 需要；
hidden_states / sample_hidden_states / aux_hidden_states：EAGLE / Medusa / DFlash 等 proposer 需要；
slot_mappings：部分 proposer / ngram 路径需要。
```

`execute_model()` 开头会检查上一轮 state 是否已被消费：

```python
if self.execute_model_state is not None:
    raise RuntimeError(
        "State error: sample_tokens() must be called "
        "after execute_model() returns None."
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4049` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4053`

这说明：

```text
对于 generation 路径，execute_model() 和 sample_tokens() 是强配对调用。
```

---

## 5. _update_states：scheduled spec tokens 先落入 InputBatch

`execute_model()` 进入 preprocess 段后首先更新 worker 侧状态：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4080` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4087`

`_update_states()` 的 docstring 明确说：

```text
The updated states are used by the _prepare_inputs function to create
the input GPU tensors for the model.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1127` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1136`

也就是说：

```text
_prepare_inputs() 不直接消费原始 SchedulerOutput；
它消费 _update_states() 后同步到 InputBatch 的 worker-local 状态。
```

### 5.1 读取 scheduled_spec_decode_tokens

在更新 running / resumed 请求时：

```python
req_data = scheduler_output.scheduled_cached_reqs
scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1261` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1265`

这些 `scheduled_spec_tokens` 是 Scheduler 本轮真正调度出来、需要 target model 验证的 draft tokens。

### 5.2 ngram_gpu 可能修剪 invalid drafts

ngram GPU 路径会先保存 Scheduler 分配的原始 spec 长度，并调用：

```python
update_scheduler_for_invalid_drafts(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1266` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1280`

这说明：

```text
某些 drafter 会在 worker 侧异步知道有效 draft 数；
_update_states() 需要在准备输入前修正 scheduler_output 中的 draft token 状态。
```

### 5.3 async spec decode 的乐观状态修正

async spec decode 下，如果请求上一轮有 draft length：

```python
if req_state.prev_num_draft_len and self.use_async_scheduling:
    ...
    optimistic_num_accepted = req_state.prev_num_draft_len
    req_state.output_token_ids.extend([-1] * optimistic_num_accepted)
    deferred_spec_decode_corrections.append(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1292` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1317`

含义：

```text
async scheduling 下，为了不阻塞下一轮 prepare，worker 侧会先乐观假设上一轮 draft tokens 都被接受；
真正的修正延后到 forward 后或 GPU correction kernel 中完成。
```

### 5.4 最终写入 InputBatch.spec_token_ids

`_update_states()` 会通过 `InputBatch.update_req_spec_token_ids()` 把本轮 spec tokens 写入 batch。

对应实现见：`code/vllm/vllm/v1/worker/gpu_input_batch.py:483`

核心动作：

```text
1. 清空当前 row 的 spec_token_ids；
2. 从 scheduled_spec_tokens 取当前 req_id 的 draft tokens；
3. request.prev_num_draft_len = num_spec_tokens；
4. 把 spec tokens 写到 token_ids_cpu 的 num_tokens_no_spec 之后；
5. 标记 is_token_ids；
6. 更新 InputBatch.spec_token_ids。
```

这一步完成了：

```text
SchedulerOutput.scheduled_spec_decode_tokens
  → InputBatch.token_ids_cpu
  → InputBatch.spec_token_ids
```

---

## 6. execute_model 主入口：spec forward 从这里开始

`GPUModelRunner.execute_model()` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`

签名：

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4048`

spec decode 相关的第一个特殊分支是 ngram GPU：

```python
if (
    self.speculative_config is not None
    and self.speculative_config.use_ngram_gpu()
):
    num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
    spec_decode_tokens_copy = (
        scheduler_output.scheduled_spec_decode_tokens.copy()
    )
    scheduler_output = replace(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4058` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4073`

原因：

```text
ngram_gpu 可能在 worker 侧修改 scheduler_output；
为了不影响 EngineCore 进程里的原始 scheduler_output，需要 shallow copy。
```

然后进入 preprocess：

```python
with (
    record_function_or_nullcontext("gpu_model_runner: preprocess"),
    self.synchronize_input_prep(),
):
    deferred_state_corrections_fn = self._update_states(scheduler_output)
    ...
    logits_indices, spec_decode_metadata = self._prepare_inputs(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4080` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4131`

---

## 7. _prepare_inputs：把 batch 状态压成本轮 target forward 输入

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1889`

```python
def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
```

返回：

```text
logits_indices
spec_decode_metadata
```

它不仅准备这两个对象，还会准备一批 persistent buffers：

```text
input_ids
positions
query_start_loc
seq_lens
num_computed_tokens
req_indices
num_scheduled_tokens
slot_mapping
num_accepted_tokens
discard_request_mask
```

这些 buffer 后续会被：

```text
_model_forward()
attention metadata builders
RejectionSampler
proposer / drafter
Mamba / GDN backend
```

共同使用。

---

## 8. _prepare_inputs 第一段：请求索引、positions、input_ids CPU

### 8.1 commit block table

一开始先提交 block table：

```python
self.input_batch.block_table.commit_block_table(num_reqs)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1906` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1908`

它让后续 slot mapping 和 attention metadata 能看到最新 KV block 映射。

### 8.2 req_indices 和 cu_num_tokens

```python
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
cu_num_tokens = self._get_cumsum_and_arange(
    num_scheduled_tokens, self.query_pos.np
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1910` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1918`

例子：

```text
num_scheduled_tokens = [2, 5, 3]
req_indices = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
cu_num_tokens = [2, 7, 10]
query_pos = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
```

这些值用于：

```text
1. 给每个 scheduled token 找到所属 request row；
2. 构造 query_start_loc；
3. 计算每个 token 在请求序列中的 position；
4. 为 _calc_spec_decode_metadata 提供每个请求 scheduled range 的结束位置。
```

### 8.3 positions_np 和 token_indices

```python
positions_np = (
    self.input_batch.num_computed_tokens_cpu[req_indices]
    + self.query_pos.np[: cu_num_tokens[-1]]
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1920` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1924`

然后计算 token matrix 中的索引：

```python
token_indices = (
    positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1936` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1942`

这一步回答：

```text
本轮每个 scheduled token，应该从 InputBatch.token_ids_cpu 的哪一行哪一列读 token id？
```

### 8.4 input_ids.cpu 从 token_ids_cpu gather

```python
torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),
    0,
    token_indices_tensor,
    out=self.input_ids.cpu[:total_num_scheduled_tokens],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1945` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1953`

由于 `_update_states()` 已经把 scheduled spec tokens 写入 `token_ids_cpu`，这里 gather 出来的 `input_ids.cpu` 已经包含本轮要验证的 draft tokens。

---

## 9. _prepare_inputs 第二段：query_start_loc、optimistic seq_lens、discard mask

### 9.1 query_start_loc

```python
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
self.query_start_loc.copy_to_gpu()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2001` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2008`

`query_start_loc` 表示每个 request 在 flatten scheduled tokens 中的区间。

spec decode 下，一个 request 的区间可能包含：

```text
1. 普通 decode token；
2. 一个或多个 draft verification tokens；
3. 为 bonus logits 准备的位置。
```

### 9.2 optimistic_seq_lens_cpu

```python
torch.add(
    self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
    torch.from_numpy(num_scheduled_tokens),
    out=self.optimistic_seq_lens_cpu[:num_reqs],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2010` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2019`

注释说明：

```text
Compute optimistic seq_lens (assumes all draft tokens from previous iteration accepted).
```

spec decode 下这是“乐观长度”：

```text
先假设 scheduled draft tokens 都进入序列，
用于 attention metadata 的 max_seq_len 和 discard_request_mask；
如果 draft 被拒绝，后续状态会再修正。
```

### 9.3 discard_request_mask

```python
num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
num_tokens_np = np.array(num_tokens, dtype=np.int32)
self.discard_request_mask.np[:num_reqs] = (
    self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2026` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2034`

它标记：

```text
哪些请求虽然参与了本轮 forward，但不应该采样或输出。
```

例如 chunked prefill 或某些 partial request。

---

## 10. _prepare_inputs 第三段：num_accepted_tokens 和 async spec 修正

如果上一步采样后有 `num_accepted_tokens_event`，会同步 accepted token 数：

```python
if self.num_accepted_tokens_event is not None:
    self.num_accepted_tokens_event.synchronize()
    ...
    self.num_accepted_tokens.copy_to_gpu()
else:
    self.num_accepted_tokens.np.fill(1)
    self.num_accepted_tokens.gpu.fill_(1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2036` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2062`

这个字段对 hybrid / Mamba / spec decode 很重要。

### 10.1 async spec decode 修正 num_computed_tokens

async spec decode 下，CPU 上的 `num_computed_tokens` 可能是乐观值，需要 GPU kernel 修正：

```python
if (
    self.use_async_spec_decode
    and self.valid_sampled_token_count_gpu is not None
    and prev_req_id_to_index
):
    ...
    update_num_computed_tokens_for_batch_change(...)
else:
    self.num_computed_tokens[:num_reqs].copy_(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2073` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2099`

含义：

```text
async spec decode 不能总等 CPU 知道哪些 draft tokens 被接受；
因此用 GPU 上的 valid_sampled_token_count_gpu 和 prev_positions 做批次变化后的修正。
```

---

## 11. _prepare_inputs 第四段：positions、seq_lens、slot mapping

设置 request indices、query positions、num scheduled tokens：

```python
self.req_indices.np[:total_num_scheduled_tokens] = req_indices
self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
...
self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
self.num_scheduled_tokens.copy_to_gpu(num_reqs)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2101` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2108`

计算最终 GPU positions：

```python
self.positions[:total_num_scheduled_tokens] = (
    self.num_computed_tokens[req_indices_gpu].to(torch.int64)
    + self.query_pos.gpu[:total_num_scheduled_tokens]
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2109` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2112`

计算 seq_lens：

```python
self.seq_lens[:num_reqs] = (
    self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2113` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2116`

然后计算 slot mapping：

```python
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    self.query_start_loc.gpu[: num_reqs + 1],
    self.positions[:total_num_scheduled_tokens],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2118` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2122`

这一步把：

```text
逻辑 token position + block table
```

转换成：

```text
物理 KV cache slot mapping。
```

spec decode 下，draft verification tokens 也必须有正确 slot mapping，否则 target model forward 无法写入 / 读取对应 KV。

---

## 12. _prepare_input_ids：normal / async 下 input_ids 如何上 GPU

`_prepare_inputs()` 随后调用：

```python
self._prepare_input_ids(
    scheduler_output,
    num_reqs,
    total_num_scheduled_tokens,
    cu_num_tokens,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2124` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2130`

### 12.1 普通路径

如果没有 `prev_sampled_token_ids`：

```python
self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1730` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1736`

这表示直接把前面 gather 好的 `input_ids.cpu` 拷贝到 GPU。

### 12.2 async scheduling 路径

如果有 `prev_sampled_token_ids`，说明部分 token 来自上一轮 GPU 采样结果，不一定已经写入 CPU token 矩阵。

代码会构造：

```text
sample_flattened_indices：上一轮 sampled token 要 scatter 到本轮 input_ids 的位置；
spec_flattened_indices：draft tokens 要 scatter 到本轮 input_ids 的位置；
prev_draft_token_indices：从上一轮 _draft_token_ids flatten 后取哪些 draft tokens。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1738` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1778`

然后先 scatter sampled tokens：

```python
self.input_ids.gpu.scatter_(
    dim=0,
    index=sampled_tokens_index_tensor,
    src=self.input_batch.prev_sampled_token_ids[
        prev_common_req_indices_tensor, 0
    ],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1807` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1820`

再 scatter draft tokens：

```python
draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)
self.input_ids.gpu.scatter_(
    dim=0,
    index=draft_tokens_index_tensor,
    src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1822` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1842`

这段逻辑说明：

```text
async spec decode 下，当前 step 的 input_ids 可能由三部分拼成：
1. InputBatch.token_ids_cpu 中已有 token；
2. 上一轮 GPU sampled token；
3. 上一轮 GPU draft token。
```

---

## 13. _prepare_inputs 第五段：分支生成 SpecDecodeMetadata

准备好 input ids、positions、seq_lens 后，进入 spec decode 判断：

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2153`

### 13.1 普通 decode 路径

如果没有 spec tokens：

```python
logits_indices = query_start_loc[1:] - 1
spec_decode_metadata = None
num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2154` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2162`

含义：

```text
每个请求只取自己的最后一个 scheduled token 的 hidden state 算 logits。
```

### 13.2 spec decode 路径

如果本轮有 scheduled spec tokens：

```python
num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
    req_idx = self.input_batch.req_id_to_index[req_id]
    draft_len = len(draft_token_ids)
    num_draft_tokens[req_idx] = draft_len
    if (
        self.input_batch.num_computed_tokens_cpu[req_idx]
        >= self.input_batch.num_prompt_tokens[req_idx]
    ):
        num_decode_draft_tokens[req_idx] = draft_len
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2163` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2182`

然后：

```python
spec_decode_metadata = self._calc_spec_decode_metadata(
    num_draft_tokens, cu_num_tokens
)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2183` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2187`

含义：

```text
每个请求需要 K 个 target verification logits + 1 个 bonus logits；
所以 num_sampled_tokens = K + 1。
```

### 13.3 num_decode_draft_tokens 给 attention backend 用

```python
self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
self.num_decode_draft_tokens.copy_to_gpu()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2188` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2191`

这个字段主要给某些 decode-only CUDA graph 或 attention backend 使用，例如 GDN / Mamba 相关 builder。

---

## 14. _calc_spec_decode_metadata：从 scheduled range 到 logits rows

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2742`

```python
def _calc_spec_decode_metadata(
    self,
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
) -> SpecDecodeMetadata:
```

它输入：

```text
num_draft_tokens：每个 request 本轮有几个 draft tokens；
cu_num_scheduled_tokens：每个 request 在 flatten scheduled tokens 中的结束 offset。
```

输出：

```text
logits_indices：从 hidden_states 中取哪些 rows 来 compute logits；
target_logits_indices：compute_logits 输出中哪些 rows 用于验证 draft tokens；
bonus_logits_indices：compute_logits 输出中哪些 rows 用于采 bonus token；
draft_token_ids：flatten 后要验证的 draft token ids。
```

核心步骤：

```python
num_sampled_tokens = num_draft_tokens + 1
cu_num_sampled_tokens = self._get_cumsum_and_arange(...)
logits_indices = np.repeat(
    cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
)
logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]
bonus_logits_indices = cu_num_sampled_tokens - 1
cu_num_draft_tokens = self._get_cumsum_and_arange(...)
target_logits_indices = np.repeat(
    cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
)
target_logits_indices += self._arange_scratch[: cu_num_draft_tokens[-1]]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2757` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2788`

最后从 input ids 中取 draft token ids：

```python
draft_token_ids = self.input_ids.gpu[logits_indices]
draft_token_ids = draft_token_ids[target_logits_indices + 1]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2807` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2810`

注意这里 `target_logits_indices + 1` 的含义：

```text
验证某个 draft token 时，需要用它前一个位置的 target logits 来判断该 draft token 是否可接受。
```

最终返回：

```python
return SpecDecodeMetadata(
    draft_token_ids=draft_token_ids,
    num_draft_tokens=num_draft_tokens.tolist(),
    cu_num_draft_tokens=cu_num_draft_tokens,
    cu_num_sampled_tokens=cu_num_sampled_tokens,
    target_logits_indices=target_logits_indices,
    bonus_logits_indices=bonus_logits_indices,
    logits_indices=logits_indices,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2812` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2820`

---

## 15. logits_indices 为什么是 spec forward 的核心

普通 decode：

```text
logits_indices = 每个请求最后一个 scheduled token 的位置
```

spec decode：

```text
logits_indices = 每个请求 target verification positions + bonus position
```

举例：

```text
某请求本轮 scheduled tokens = [x, d1, d2, d3]
其中 d1/d2/d3 是 draft tokens。

需要的 logits rows：
  row(x)  → 验 d1
  row(d1) → 验 d2
  row(d2) → 验 d3
  row(d3) → 采 bonus token
```

所以一次 target model forward 后，不是对所有 hidden states 都算 logits，而是：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

这就是 spec forward 的核心收益点之一：

```text
target model 一次 forward 覆盖多个 draft verification 位置，
但只对真正需要采样/验证的位置计算 logits。
```

---

## 16. LoRA：num_sampled_tokens 也要考虑 spec decode

`_prepare_inputs()` 末尾，如果启用 LoRA：

```python
if self.lora_config:
    assert (
        np.sum(num_sampled_tokens)
        <= self.vllm_config.scheduler_config.max_num_batched_tokens
    )
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2193` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2201`

spec decode 下 `num_sampled_tokens = num_draft_tokens + 1`，所以 LoRA 的采样相关映射也必须知道每个请求最多会产生多少 sampled tokens。

---

## 17. execute_model 中的 batch execution 决策

`_prepare_inputs()` 之后，ModelRunner 继续决定 batch 如何执行：

```python
(
    cudagraph_mode,
    batch_desc,
    should_ubatch,
    num_tokens_across_dp,
    cudagraph_stats,
) = self._determine_batch_execution_and_padding(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4143` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4156`

输入包括：

```text
num_tokens_unpadded
num_reqs
num_scheduled_tokens_np
max_num_scheduled_tokens
是否使用 cascade attention
encoder request 数量
```

spec decode 影响这里的方式是：

```text
1. num_tokens_unpadded 包含 scheduled draft tokens；
2. max_num_scheduled_tokens 可能大于普通 decode 的 1；
3. batch padding / CUDA graph / ubatching 要覆盖多个 verification tokens；
4. 后续 attention metadata 可能使用 padded num_tokens / num_reqs。
```

---

## 18. Mamba / hybrid：spec forward 前的特殊处理

如果 Mamba cache mode 是 `align`：

```python
if self.cache_config.mamba_cache_mode == "align":
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()
        deferred_state_corrections_fn = None
    ...
    mamba_utils.preprocess_mamba(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4198` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4216`

注释说明：

```text
preprocess_mamba reads req_state.num_computed_tokens (CPU)
to decide copy operations, so we must apply deferred corrections before it runs.
```

这和 spec decode 的关系是：

```text
spec decode 可能因为 rejected draft tokens 需要修正 num_computed_tokens；
Mamba recurrent state 对位置和 accepted token 数敏感；
所以在 Mamba preprocess 前必须确保状态修正到位。
```

如果存在 postprocess align kernel，还会 stage 每个 request 的输入：

```python
mamba_utils.stage_postprocess_inputs_to_gpu(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4226` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4239`

---

## 19. _get_slot_mappings：spec tokens 也要映射到 KV slots

执行前会计算 slot mappings：

```python
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
    num_tokens_padded=...,
    num_reqs_padded=...,
    num_tokens_unpadded=num_tokens_unpadded,
    ubatch_slices=ubatch_slices_padded,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4241` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4253`

spec decode 下：

```text
scheduled draft tokens 和普通 tokens 一样参与 target forward，
因此也必须拥有正确 slot_mapping。
```

如果某些 attention backend 把 KV cache update 和 forward 分离，还可能使用 padded dimensions：

```text
has_separate_kv_update 为 True 时，slot mappings 需要按 padded dimensions 生成。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4185` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4197`

---

## 20. _build_attention_metadata：spec decode 如何注入 attention metadata

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

`execute_model()` 调用：

```python
attn_metadata, spec_decode_common_attn_metadata = (
    self._build_attention_metadata(
        num_tokens=num_tokens_unpadded,
        num_tokens_padded=num_tokens_padded if pad_attn else None,
        num_reqs=num_reqs,
        num_reqs_padded=num_reqs_padded if pad_attn else None,
        max_query_len=max_num_scheduled_tokens,
        ubatch_slices=ubatch_slices_attn,
        logits_indices=logits_indices,
        use_spec_decode=use_spec_decode,
        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
        slot_mappings=slot_mappings_by_group,
    )
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4269`

这里有两个输出：

```text
attn_metadata：target model forward 使用的 per-layer metadata；
spec_decode_common_attn_metadata：drafter / proposer 后续生成 draft tokens 使用的 common metadata。
```

---

## 21. CommonAttentionMetadata 的基础字段

`_build_attention_metadata()` 先构造 `CommonAttentionMetadata`：

```python
cm_base = CommonAttentionMetadata(
    query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
    seq_lens=self.seq_lens[:num_reqs_padded],
    _seq_lens_cpu=seq_lens_cpu,
    _num_computed_tokens_cpu=num_computed_tokens_cpu,
    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
    num_reqs=num_reqs_padded,
    num_actual_tokens=num_tokens_padded,
    max_query_len=max_query_len,
    max_seq_len=max_seq_len,
    block_table_tensor=block_table_gid_0,
    slot_mapping=slot_mapping_gid_0,
    causal=True,
    is_prefilling=is_prefilling,
    positions=self.positions[:num_tokens_padded],
    mm_req_doc_ranges=req_doc_ranges,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2330` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2347`

spec decode 影响这些字段：

| 字段 | spec decode 影响 |
|---|---|
| `query_start_loc` | 每个请求 query 区间可能包含多个 draft verification tokens |
| `seq_lens` | 使用包含 scheduled draft tokens 的乐观 seq lens |
| `max_query_len` | 可能大于普通 decode 的 1 |
| `max_seq_len` | 由 optimistic seq lens 决定 |
| `slot_mapping` | draft tokens 也有 KV cache slot |
| `positions` | draft verification positions 也要参与 attention |
| `is_prefilling` | 区分 prefill / decode，影响 Mamba 等 backend |

---

## 22. async spec decode 下 CPU metadata 可能被置空

`_build_attention_metadata()` 中：

```python
if self.use_async_spec_decode:
    # GPU tensors are authoritative in async mode.
    seq_lens_cpu = None
    num_computed_tokens_cpu = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2301` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2305`

含义：

```text
async spec decode 下，CPU 侧 num_computed_tokens / seq_lens 可能是乐观或滞后的；
GPU 上已经通过 correction kernel 得到权威值；
因此 attention metadata 不再依赖 CPU values。
```

---

## 23. Mamba / GDN attention metadata 的 spec 参数

构造每个 attention group 的 metadata 时，如果 `use_spec_decode` 且 builder 是 Mamba2 或 GDN：

```python
if use_spec_decode and isinstance(
    builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
):
    extra_attn_metadata_args = dict(
        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
            :num_reqs_padded
        ],
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2398` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2408`

如果是 Mamba2 且有 `mamba_prev_last_scheduled_idx`，还会传：

```python
extra_attn_metadata_args["prev_last_scheduled_idx"] = (
    self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2409` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2415`

这说明：

```text
某些 state-space / hybrid backend 不能只知道 token positions，
还需要知道上一轮接受了几个 draft tokens，
以及本轮 decode draft token 数，才能正确更新 recurrent state。
```

---

## 24. spec_decode_common_attn_metadata：给 drafter 的 attention metadata

`_build_attention_metadata()` 会额外挑一个 common metadata 给 drafter：

```python
if self.speculative_config and spec_decode_common_attn_metadata is None:
    if isinstance(
        self.drafter,
        (
            EagleProposer,
            DFlashProposer,
            Gemma4Proposer,
            ExtractHiddenStatesProposer,
        ),
    ):
        if self.drafter.kv_cache_gid == kv_cache_gid:
            spec_decode_common_attn_metadata = cm
    else:
        spec_decode_common_attn_metadata = cm
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2467` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2480`

这份 metadata 不是 target model forward 的唯一 metadata，而是后续 `sample_tokens()` 中 drafter 生成下一轮 draft tokens 时会用到。

如果是多 KV group proposer，还会把 per-group metadata / block table 传给 drafter：

```python
if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):
    self.drafter.set_per_group_attn_metadata(...)
elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
    self.drafter.set_per_group_block_table(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2482` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2489`

如果 target forward 使用 padding，但 drafter 不希望使用 padded metadata，会 unpad：

```python
if spec_decode_common_attn_metadata is not None and (
    num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
):
    spec_decode_common_attn_metadata = (
        spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2499` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2507`

---

## 25. _preprocess：最终 forward 输入

attention metadata 构造完成后，ModelRunner 调用：

```python
(
    input_ids,
    inputs_embeds,
    positions,
    intermediate_tensors,
    model_kwargs,
    ec_connector_output,
) = self._preprocess(
    scheduler_output, num_tokens_padded, intermediate_tensors
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4271` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4280`

`_preprocess()` 入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

它根据模型类型决定最终 forward 输入：

```text
文本模型：input_ids = self.input_ids.gpu[:num_input_tokens]
多模态模型：先 embed input ids + multimodal embeddings，再使用 inputs_embeds
prompt_embeds：只对 token-id 位置做 embedding，保留外部 embeddings
encoder-decoder：执行 encoder 并把 encoder_outputs 放入 model_kwargs
PP 非 first rank：同步 intermediate_tensors
M-RoPE / XD-RoPE：使用对应 positions tensor
```

文本模型路径：

```python
input_ids = self.input_ids.gpu[:num_input_tokens]
inputs_embeds = None
model_kwargs = self._init_model_kwargs()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3526` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3533`

positions：

```python
if self.uses_mrope:
    positions = self.mrope_positions.gpu[:, :num_input_tokens]
elif self.uses_xdrope_dim > 0:
    positions = self.xdrope_positions.gpu[:, :num_input_tokens]
else:
    positions = self.positions[:num_input_tokens]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3535` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3542`

spec decode 对 `_preprocess()` 的影响主要是：

```text
input_ids / positions 已经包含 scheduled draft verification tokens；
_preprocess() 不再单独区分 spec tokens，而是把它们作为本轮 target forward 的普通 token 输入。
```

---

## 26. set_forward_context：forward 时 attention metadata 如何生效

target model forward 外面包了 forward context：

```python
with (
    set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_mode,
        batch_descriptor=batch_desc,
        ubatch_slices=ubatch_slices_padded,
        slot_mapping=slot_mappings,
        skip_compiled=has_encoder_input,
    ),
    record_function_or_nullcontext("gpu_model_runner: forward"),
    self.maybe_get_kv_connector_output(
        scheduler_output,
        defer_finalize=defer_kv_connector_finalize,
    ) as kv_connector_output,
):
    model_output = self._model_forward(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4302` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

这个 context 让模型内部 attention 层可以读取：

```text
attn_metadata
num_tokens
cudagraph mode
batch descriptor
ubatch slices
slot mappings
```

spec decode 下，attention 层看到的是一个包含 draft verification tokens 的普通 forward batch。

---

## 27. KV connector：spec decode 下为什么 defer finalize

在 forward 前：

```python
defer_kv_connector_finalize = self.speculative_config is not None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4297` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4301`

传给：

```python
self.maybe_get_kv_connector_output(
    scheduler_output,
    defer_finalize=defer_kv_connector_finalize,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4315` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4318`

注释说明：

```text
When spec decode is enabled, defer connector finalization
(wait_for_save + clear metadata) until after draft model runs.
```

原因：

```text
spec decode 中 target forward 后还要运行 drafter；
draft model 也可能需要保存 / 使用 KV connector 状态；
因此不能在 target forward 结束后立刻 finalize connector。
```

真正 finalize 在 `sample_tokens()` 里 draft model 运行之后：

```python
if spec_config is not None:
    self.finalize_kv_connector()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4596` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4600`

---

## 28. _model_forward：target model 一次验证多个位置

真正模型 forward：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4320` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

在 spec decode 场景下，`input_ids` 包含：

```text
本轮普通 scheduled tokens
+ 本轮要 target model 验证的 draft tokens
```

所以 target model forward 一次会产生这些 token 位置的 hidden states。

这一步还没有做 rejection sampling。

---

## 29. forward 后：last PP rank 才计算 logits

forward 后进入 postprocess：

```python
if self.use_aux_hidden_state_outputs:
    hidden_states, aux_hidden_states = model_output
else:
    hidden_states = model_output
    aux_hidden_states = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4328` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4335`

如果当前不是 last PP rank：

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4337` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4343`

这表示：

```text
PP 非 last rank 只执行自己的模型层，不计算 logits，不做 rejection sampling，不生成 draft tokens。
```

如果是 pooling model，直接返回 pooling output：

```python
if self.is_pooling_model:
    return self._pool(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4345` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4352`

普通 generation / spec decode 路径：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

---

## 30. spec decode logits 的语义

`logits` 的 shape 是：

```text
[num_draft_tokens_total + num_reqs, vocab_size]
```

它不是完整 scheduled tokens 的 logits，而是只包含：

```text
1. 每个 draft token 对应的 target verification logits；
2. 每个请求一个 bonus token logits。
```

对应关系由 `SpecDecodeMetadata` 描述：

```text
target_logits_indices：哪些 logits rows 用于验证 draft tokens；
bonus_logits_indices：哪些 logits rows 用于 bonus sampling；
draft_token_ids：target logits 要验证的 token id。
```

所以 `compute_logits()` 后并不会立即采样，而是先保存到 `ExecuteModelState`，等 `sample_tokens()` 调 `RejectionSampler`。

---

## 31. ExecuteModelState 保存并返回 None

postprocess 后：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
)
self.kv_connector_output = kv_connector_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4398`

然后：

```python
if deferred_state_corrections_fn:
    deferred_state_corrections_fn()

return None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4400` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`

为什么返回 `None`？

```text
因为 generation 路径的 forward/logits 和 sampling 被拆成两步：
execute_model() 负责 target forward + logits；
sample_tokens() 负责 grammar bitmask + sampler / rejection sampler + output。
```

spec decode 下这个拆分更重要，因为 `sample_tokens()` 还要使用：

```text
spec_decode_metadata
spec_decode_common_attn_metadata
hidden_states
sample_hidden_states
slot_mappings
```

来完成 rejection sampling 和下一轮 draft proposal。

---

## 32. sample_tokens：消费 spec forward 状态

`sample_tokens()` 入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`

如果没有 `execute_model_state`，只返回 KV connector output：

```python
if self.execute_model_state is None:
    ...
    return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4426` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4434`

正常路径先 unpack：

```python
(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
) = self.execute_model_state
self.execute_model_state = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4436` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4450`

这一步消费了 `execute_model()` 保存的 forward 状态。

---

## 33. grammar bitmask 在 sampling 前应用

如果 Scheduler 给了 grammar output：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

这说明 structured output 约束是在 sampling / rejection sampling 前作用到 logits 上。

spec decode 下 grammar bitmask 必须理解 `scheduled_spec_decode_tokens`，否则 target verification 和 bonus sampling 可能违反结构化输出约束。

---

## 34. _sample：从普通 sampler 切到 RejectionSampler

`sample_tokens()` 调用：

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4458` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4459`

`_sample()` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3570`

如果没有 spec metadata：

```python
if spec_decode_metadata is None:
    return self.sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3580` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3584`

如果有 spec metadata：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
return sampler_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3592` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3599`

这就是 spec forward 到 rejection sampling 的衔接点。

---

## 35. draft_probs 如何和 spec metadata 对齐

`_get_spec_decode_draft_probs()` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4823`

如果没有 cached draft probabilities：

```python
if self._draft_probs is None or self._draft_prob_req_ids is None:
    return None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4823` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4827`

如果有，则按当前 `input_batch.req_ids` 和 `spec_decode_metadata.num_draft_tokens` 对齐：

```python
for req_id, num_draft in zip(
    self.input_batch.req_ids, spec_decode_metadata.num_draft_tokens
):
    if num_draft == 0:
        continue
    row_idx = row_by_req_id.get(req_id)
    ...
    draft_probs_rows.append(self._draft_probs[row_idx, :num_draft])
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4829` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4846`

最后 flatten：

```python
return torch.cat(draft_probs_rows, dim=0).contiguous()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4848` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4850`

这保证：

```text
draft_probs 的 row 顺序和 SpecDecodeMetadata.draft_token_ids / target_logits_indices 对齐。
```

如果对齐失败，会回退到 legacy behavior：

```text
Missing cached draft probabilities ... falling back to legacy speculative rejection behavior.
```

---

## 36. sample 后：状态更新和下一轮 draft proposal

采样后更新 worker 侧状态：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4461` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4463`

然后清理旧 draft cache：

```python
self._draft_token_ids = None
self._draft_probs = None
self._draft_prob_req_ids = None
self._draft_token_req_ids = None
self.valid_sampled_token_count_gpu = None
self.input_batch.prev_sampled_token_ids = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4474` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4479`

接着定义局部函数生成下一轮 draft tokens：

```python
def propose_draft_token_ids(sampled_token_ids):
    assert spec_decode_common_attn_metadata is not None
    with record_function_or_nullcontext("gpu_model_runner: draft"):
        self._draft_token_ids = self.propose_draft_token_ids(
            scheduler_output,
            sampled_token_ids,
            self.input_batch.sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            slot_mappings,
        )
        self._copy_draft_token_ids_to_cpu(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4481` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4495`

这说明 `execute_model()` 保存的：

```text
hidden_states
sample_hidden_states
aux_hidden_states
spec_decode_metadata
spec_decode_common_attn_metadata
slot_mappings
```

不仅用于 rejection sampling，也用于下一轮 drafter proposal。

---

## 37. drafter 是否能运行：_input_fits_in_drafter

在 sample 阶段会判断：

```python
input_fits_in_drafter = self._input_fits_in_drafter(
    spec_decode_common_attn_metadata
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4497` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4503`

`_input_fits_in_drafter()` 判断：

```python
num_drafter_query_tokens = self.num_spec_tokens + (
    1 if self.speculative_config.use_dflash() else 0
)
return (
    common_attn_metadata.max_seq_len + num_drafter_query_tokens
    <= self.effective_drafter_max_model_len
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4407` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4419`

如果输入超过 drafter 最大长度，会 zero out draft tokens：

```python
self._draft_token_ids = torch.zeros(
    1, device=self.device, dtype=torch.int32
).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
...
self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4561` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4572`

原因：

```text
不能让 Scheduler 下一轮调度 stale drafts；
对接近 max_model_len 的序列，旧 draft tokens 可能污染 Mamba recurrent state 和 logprobs。
```

---

## 38. GPU draft path vs CPU bookkeeping path

有些 drafter 可以直接使用 GPU sampled tokens，不必等 CPU bookkeeping：

```python
use_gpu_toks = (
    spec_config.use_eagle()
    or spec_config.uses_draft_model()
    or spec_config.uses_extract_hidden_states()
) and not spec_config.disable_padded_drafter_batch
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4504` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4508`

如果 `use_gpu_toks` 且 fits in drafter：

```python
sampled_token_ids = sampler_output.sampled_token_ids
propose_draft_token_ids(sampled_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4509` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4523`

ngram GPU 也有类似路径：

```python
elif spec_config.use_ngram_gpu() and not spec_config.disable_padded_drafter_batch:
    ...
    propose_draft_token_ids(sampled_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4536` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4557`

其他方法需要等 bookkeeping 后拿 CPU list：

```python
propose_drafts_after_bookkeeping = input_fits_in_drafter
...
if propose_drafts_after_bookkeeping:
    propose_draft_token_ids(valid_sampled_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4558` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4594`

这说明：

```text
spec forward 的 hidden states / metadata 会被不同 proposer 以不同方式消费；
有些走 GPU 快路径，有些等 CPU bookkeeping。
```

---

## 39. ModelRunnerOutput：spec forward 最终返回什么

`sample_tokens()` 最后构造：

```python
output = ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    kv_connector_output=kv_connector_output,
    ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
    num_nans_in_logits=num_nans_in_logits,
    cudagraph_stats=cudagraph_stats,
    routed_experts=None,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4609` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4623`

`sampled_token_ids` 在 spec decode 下可能是：

```text
每个请求多个 token：accepted draft prefix + replacement / bonus token。
```

它不是 raw draft tokens，而是 rejection sampling 后真正返回给 Scheduler 的生成结果。

Scheduler 后续会用 `scheduled_spec_decode_tokens` 和 `sampled_token_ids` 计算 accepted / rejected 数量，修正 `num_computed_tokens`。

---

## 40. 和 Scheduler 的边界

### Scheduler 负责

```text
Request.spec_token_ids；
num_tokens_with_spec；
本轮 scheduled_spec_decode_tokens；
KV lookahead allocation；
update_from_output() 中 accepted / rejected 记账；
下一轮 update_draft_token_ids()。
```

### ModelRunner 负责

```text
把 scheduled_spec_decode_tokens 写入 InputBatch；
把 InputBatch 转成 input_ids / positions / slot_mapping；
构造 SpecDecodeMetadata；
执行 target model forward；
计算 target / bonus logits；
在 sample_tokens() 中调用 RejectionSampler；
生成下一轮 draft tokens。
```

边界一句话：

```text
Scheduler 决定“验证哪些 draft tokens”，ModelRunner 决定“如何把这些 tokens 跑进 target model 并产出 logits”。
```

---

## 41. 和 RejectionSampler 的边界

### ModelRunner forward 提供

```text
logits
SpecDecodeMetadata
SamplingMetadata
draft_probs
```

### RejectionSampler 决定

```text
每个 draft token 是否接受；
如果全部接受，bonus token 是什么；
最终 sampled_token_ids 矩阵；
可选 logprobs。
```

ModelRunner 不在 `_model_forward()` 中判断接受 / 拒绝；它只准备 rejection sampler 所需的输入。

---

## 42. 和 attention backend 的边界

attention backend 不知道“这是 speculative decoding 业务逻辑”。

它看到的是：

```text
query_start_loc
seq_lens
positions
block_table
slot_mapping
max_query_len
num_accepted_tokens / num_decode_draft_tokens_cpu（部分 backend）
```

也就是说：

```text
ModelRunner 把 spec decode 的多 token 验证转换成 attention backend 可以执行的普通 batch metadata。
```

对于 Mamba / GDN 这类需要 accepted token 数的 backend，ModelRunner 会额外注入 spec decode 参数。

---

## 43. 一个完整例子：单请求 3 个 draft tokens

假设某请求本轮被 Scheduler 调度：

```text
真实上下文最后 token：x
scheduled draft tokens：[d1, d2, d3]
num_scheduled_tokens = 4
```

### 43.1 _update_states

```text
InputBatch.token_ids_cpu[row] = [..., x, d1, d2, d3]
InputBatch.spec_token_ids[row] = [d1, d2, d3]
```

### 43.2 _prepare_inputs

```text
input_ids = [x, d1, d2, d3]
positions = [p, p+1, p+2, p+3]
query_start_loc = [0, 4]
num_draft_tokens = [3]
```

### 43.3 _calc_spec_decode_metadata

```text
logits_indices = [0, 1, 2, 3]
target_logits_indices = [0, 1, 2]
bonus_logits_indices = [3]
draft_token_ids = [d1, d2, d3]
```

语义：

```text
logits(row 0) 验 d1；
logits(row 1) 验 d2；
logits(row 2) 验 d3；
logits(row 3) 采 bonus token。
```

### 43.4 target forward

```text
_model_forward(input_ids=[x,d1,d2,d3])
  → hidden_states[0..3]
  → sample_hidden_states = hidden_states[logits_indices]
  → logits = compute_logits(sample_hidden_states)
```

### 43.5 sample_tokens

```text
RejectionSampler(
  draft_token_ids=[d1,d2,d3],
  target logits rows=[0,1,2],
  bonus logits row=[3],
)
```

如果接受 d1、d2，拒绝 d3，并采出 r：

```text
sampled_token_ids = [d1, d2, r]
```

这会通过 `ModelRunnerOutput` 返回 Scheduler。

---

## 44. 容易混淆的点

### 44.1 scheduled_spec_decode_tokens 会直接送给模型吗？

不是直接送。

```text
scheduled_spec_decode_tokens
  → _update_states()
  → InputBatch.token_ids_cpu / spec_token_ids
  → _prepare_inputs()
  → input_ids.gpu
```

模型 forward 只看到 `input_ids / inputs_embeds / positions`，不知道这些 token 原本来自 `scheduled_spec_decode_tokens`。

### 44.2 SpecDecodeMetadata 是 attention metadata 吗？

不是。

```text
SpecDecodeMetadata：给 sampler / rejection sampler 用，描述 logits rows；
AttentionMetadata：给 attention backend 用，描述 query、KV、slot mapping、seq lens。
```

两者都在 `_prepare_inputs()` / `_build_attention_metadata()` 附近产生，但服务对象不同。

### 44.3 execute_model() 为什么不直接返回 ModelRunnerOutput？

因为 generation 路径需要先执行 forward/logits，再等 grammar bitmask，然后在 `sample_tokens()` 中采样。

spec decode 下还需要：

```text
RejectionSampler；
_update_states_after_model_execute；
drafter proposal；
KV connector finalize；
bookkeeping；
ModelRunnerOutput 构造。
```

所以 `execute_model()` 保存 `ExecuteModelState` 后返回 `None`。

### 44.4 logits_indices 是 target logits indices 吗？

不是同一个层级。

```text
logits_indices：从 hidden_states 中取哪些 rows 去 compute_logits；
target_logits_indices：在 compute_logits 结果中，哪些 rows 用于验证 draft tokens。
```

### 44.5 spec decode forward 是否只发生在 last PP rank？

target model 的各 PP rank 都会 forward 自己的层。

但：

```text
非 last PP rank：返回 IntermediateTensors；
last PP rank：计算 logits，保存 ExecuteModelState，后续 sample / rejection / draft。
```

### 44.6 draft proposal 属于 forward 吗？

严格说不属于 target model forward。

但它依赖 execute_model 保存的：

```text
hidden_states
sample_hidden_states
aux_hidden_states
spec_decode_common_attn_metadata
slot_mappings
```

所以本篇把它作为 “forward 后 sample_tokens 衔接” 来说明。

---

## 45. 从“回答问题”的角度总结

如果要问：

```text
ModelRunner 如何执行 spec decode forward？
```

可以回答：

```text
ModelRunner 先在 _update_states() 中把 SchedulerOutput.scheduled_spec_decode_tokens 写入 worker-local InputBatch，使 draft tokens 成为本轮 target model 输入的一部分。随后 _prepare_inputs() 根据 InputBatch 构造 input_ids、positions、query_start_loc、seq_lens 和 slot mapping，并在发现本轮有 scheduled draft tokens 时，计算每个请求的 num_draft_tokens，调用 _calc_spec_decode_metadata() 生成 logits_indices、target_logits_indices、bonus_logits_indices 和 draft_token_ids。

execute_model() 再基于这些输入构造 attention metadata，执行 target model forward，只对 logits_indices 指向的 hidden states 调 compute_logits，得到 target verification logits 和 bonus logits。它不会在这里完成采样，而是把 scheduler_output、logits、spec_decode_metadata、spec_decode_common_attn_metadata、hidden_states、sample_hidden_states、slot_mappings 等保存到 ExecuteModelState，然后返回 None。

随后 sample_tokens() 取出 ExecuteModelState，先应用 grammar bitmask，再通过 _sample() 调 RejectionSampler。RejectionSampler 使用 spec_decode_metadata、draft_probs、logits 和 sampling metadata 决定接受多少 draft tokens。采样后 ModelRunner 更新 worker 侧状态，并基于 forward hidden states 和 common attention metadata 生成下一轮 draft tokens，最后构造 ModelRunnerOutput 返回 Scheduler。
```

职责关系可以概括为：

```text
Scheduler：决定本轮验证哪些 draft tokens；
InputBatch：保存这些 draft tokens 的 worker-local token 状态；
_prepare_inputs：把 draft tokens 变成 positions / logits_indices / SpecDecodeMetadata；
_build_attention_metadata：把 spec batch 变成 attention backend 可执行 metadata；
_model_forward：target model 一次验证多个位置；
sample_tokens / RejectionSampler：决定接受 / 拒绝；
proposer：生成下一轮 draft tokens。
```

---

## 46. 最关键流程图

```text
SchedulerOutput.scheduled_spec_decode_tokens
  → GPUModelRunner._update_states()
      └─ InputBatch.update_req_spec_token_ids()
          ├─ token_ids_cpu[row, num_tokens_no_spec:...]
          └─ spec_token_ids[row]

  → GPUModelRunner._prepare_inputs()
      ├─ req_indices / cu_num_tokens
      ├─ positions / token_indices
      ├─ input_ids.cpu gather
      ├─ _prepare_input_ids() copy/scatter to GPU
      ├─ num_draft_tokens
      ├─ _calc_spec_decode_metadata()
      │    ├─ logits_indices
      │    ├─ target_logits_indices
      │    ├─ bonus_logits_indices
      │    └─ draft_token_ids
      └─ num_decode_draft_tokens

  → GPUModelRunner._build_attention_metadata(use_spec_decode=True)
      ├─ CommonAttentionMetadata
      ├─ per-layer AttentionMetadata
      └─ spec_decode_common_attn_metadata

  → GPUModelRunner._preprocess()
      ├─ input_ids / inputs_embeds
      ├─ positions
      └─ model_kwargs

  → set_forward_context(attn_metadata, slot_mapping, ...)
  → _model_forward()
  → hidden_states[logits_indices]
  → compute_logits()
  → ExecuteModelState(...)
  → return None

  → sample_tokens()
      ├─ apply_grammar_bitmask()
      ├─ _sample()
      │    └─ RejectionSampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)
      ├─ _update_states_after_model_execute()
      ├─ propose_draft_token_ids()
      └─ ModelRunnerOutput(sampled_token_ids=...)
```

---

## 47. 最关键对象关系

```text
SchedulerOutput.scheduled_spec_decode_tokens
  Scheduler 发来的本轮待验证 draft tokens。

InputBatch.spec_token_ids
  Worker batch row 上暂存的本轮 spec tokens。

input_ids.gpu
  target model forward 的真实输入，包含 draft verification tokens。

positions
  每个 scheduled token 的模型位置，draft tokens 也有位置。

logits_indices
  从 hidden_states 中取哪些 rows 计算 logits。

SpecDecodeMetadata
  描述 draft_token_ids、target_logits_indices、bonus_logits_indices。

AttentionMetadata
  描述 attention backend 的 query / KV / slot / seq lens 信息。

spec_decode_common_attn_metadata
  给 drafter / proposer 生成下一轮 draft tokens 使用。

ExecuteModelState
  execute_model() 到 sample_tokens() 的临时状态桥。

ModelRunnerOutput.sampled_token_ids
  RejectionSampler 后真正返回给 Scheduler 的 token ids。
```

---

## 48. 最小心智模型

如果只记一条主线，可以记：

```text
ModelRunner 把 Scheduler 给的 draft tokens 放进 input_ids，
让 target model 一次 forward 产生验证这些 draft tokens 所需的 logits，
再用 SpecDecodeMetadata 告诉 RejectionSampler 哪些 logits 验 draft、哪些 logits 采 bonus。
```

再压缩成一句话：

```text
ModelRunner spec forward = draft tokens 入 batch + target forward + logits row 布局 + ExecuteModelState 桥接 sampling。
```
