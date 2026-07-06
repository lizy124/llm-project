# 05. Worker / ModelRunner 如何维护 batch 和请求状态？

源码位置：

- `code/vllm/vllm\v1\worker\gpu_model_runner.py`
- `code/vllm/vllm\v1\worker\gpu_input_batch.py`
- `code/vllm/vllm\v1\core\sched\output.py`

本问题关注：`SchedulerOutput` 进入 Worker / ModelRunner 后，Worker 侧如何缓存请求状态、维护持久 batch、增删请求、更新 token / block / sampling / LoRA / pooling / spec decode 状态，并为后续 `_prepare_inputs()`、attention metadata 构造和模型 forward 提供稳定输入。

---

## 1. 一句话回答

Worker / ModelRunner 不是每轮都从头构造完整 batch，而是维护一份 **Worker 侧持久请求缓存 + 持久 InputBatch**。

核心结构是：

```text
GPUModelRunner.requests：
  req_id -> CachedRequestState
  保存 Worker 侧请求全集缓存。

InputBatch：
  保存当前这一轮真正参与执行的 batch。
  维护 req_id -> req_index 映射、token ids、block table、sampling 参数、LoRA 映射等。
```

每轮执行前，`GPUModelRunner._update_states(scheduler_output)` 会用 `SchedulerOutput` 更新它们：

```text
1. 删除 finished 请求；
2. 清理 encoder cache / 新分配 block zero；
3. 从 InputBatch 移除本轮未调度请求；
4. 把 scheduled_new_reqs 加入 Worker 缓存；
5. 更新 scheduled_cached_reqs 的 token / block / computed 状态；
6. 把本轮要执行的请求加入 InputBatch；
7. 刷新 sampling metadata。
```

所以：

```text
Scheduler 决定本轮哪些请求执行；
ModelRunner._update_states() 把这个决定同步到 Worker 侧 batch 状态。
```

---

## 2. 核心对象关系

```text
SchedulerOutput
  → GPUModelRunner._update_states()
      → self.requests: dict[str, CachedRequestState]
      → self.input_batch: InputBatch
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._build_attention_metadata()
  → GPUModelRunner._model_forward()
```

可以把它分成两层：

```text
CachedRequestState：
  单请求的 Worker 侧缓存状态。

InputBatch：
  当前执行 batch 的紧凑数组化状态。
```

---

## 3. SchedulerOutput 提供哪些状态更新信息

`SchedulerOutput` 定义在：`output.py:180`

关键字段：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]
    ...
```

位置：`output.py:180` 到 `output.py:245`

这些字段在 Worker 侧的作用是：

| 字段 | Worker 侧用途 |
|---|---|
| `scheduled_new_reqs` | 首次调度的新请求，需要完整缓存请求数据 |
| `scheduled_cached_reqs` | 已经在 Worker 缓存过的请求，只发送 diff |
| `num_scheduled_tokens` | 本轮每个请求要执行多少 token |
| `total_num_scheduled_tokens` | 判断本轮是否真的 forward |
| `scheduled_spec_decode_tokens` | 更新 spec decode token 占位 |
| `scheduled_encoder_inputs` | 多模态 / encoder 输入调度 |
| `finished_req_ids` | 通知 Worker 清理请求缓存和 batch |
| `free_encoder_mm_hashes` | 清理 encoder cache |
| `new_block_ids_to_zero` | 新分配 KV block 需要先 zero，避免脏数据 |

---

## 4. CachedRequestState 是什么

定义在：`gpu_input_batch.py:33`

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    ...
```

位置：`gpu_input_batch.py:33` 到 `gpu_input_batch.py:64`

它保存的是单个请求在 Worker 侧需要反复使用的状态，包括：

```text
prompt token ids / prompt embeds
多模态 features
sampling params / pooling params
随机数 generator
KV block ids
num_computed_tokens
output_token_ids
LoRA request
M-RoPE / XD-RoPE positions
prompt logprobs 中间结果
spec decode draft 长度
```

它的核心属性：

```python
@property
def num_tokens(self) -> int:
    return self.num_prompt_tokens + len(self.output_token_ids)
```

位置：`gpu_input_batch.py:74` 到 `gpu_input_batch.py:76`

也就是说，Worker 侧认为请求当前 token 总数是：

```text
prompt tokens + 已生成 output tokens
```

---

## 5. InputBatch 是什么

定义在：`gpu_input_batch.py:91`

```python
class InputBatch:
```

`InputBatch` 是当前 Worker 侧真正用于模型执行的 batch 状态。

它不是一个简单的 list，而是一组 CPU / GPU 张量和索引结构。

### 5.1 请求索引

```python
self._req_ids: list[str | None] = []
self.req_id_to_index: dict[str, int] = {}
```

位置：`gpu_input_batch.py:126` 到 `gpu_input_batch.py:127`

这两个字段回答：

```text
当前 batch 中有哪些请求？
每个 request_id 对应 batch 里的第几行？
```

### 5.2 token 状态

```python
self.token_ids_cpu_tensor
self.token_ids_cpu
self.is_token_ids_tensor
self.is_token_ids
self.num_tokens_no_spec_cpu_tensor
self.num_prompt_tokens_cpu_tensor
self.num_computed_tokens_cpu_tensor
```

位置：`gpu_input_batch.py:129` 到 `gpu_input_batch.py:168`

它们保存：

```text
每个请求的 token ids；
prompt token 数；
不含 spec token 的 token 总数；
已计算 token 数；
prompt_embeds / mixed input 的 token mask。
```

### 5.3 block table

```python
self.block_table = MultiGroupBlockTable(...)
```

位置：`gpu_input_batch.py:170` 到 `gpu_input_batch.py:181`

这是 Worker 侧把请求映射到 KV cache blocks 的关键结构。

### 5.4 sampling 参数

InputBatch 会维护每个请求的采样参数：

```text
temperature
top_p
top_k
frequency_penalties
presence_penalties
repetition_penalties
allowed_token_ids
bad_words_token_ids
generators
num_logprobs
logprob_token_ids
```

位置：`gpu_input_batch.py:183` 到 `gpu_input_batch.py:287`

这些最后会构造为 `SamplingMetadata`。

### 5.5 LoRA / pooling / spec decode

InputBatch 还保存：

```text
request_lora_mapping
lora_id_to_request_ids
pooling_params
pooling_states
spec_token_ids
num_accepted_tokens_cpu
prev_sampled_token_ids
prev_req_id_to_index
```

这些支持：

```text
LoRA 动态 adapter；
pooling / embedding 任务；
speculative decoding；
async scheduling 下复用上一轮 sampled tokens。
```

---

## 6. 为什么 Worker 侧要缓存请求状态

Scheduler 每轮不会把所有请求完整数据都重新发送给 Worker。

`SchedulerOutput` 里明确区分：

```text
scheduled_new_reqs：
  第一次调度的新请求，发送完整请求数据。

scheduled_cached_reqs：
  之前调度过的请求，只发送增量 diff。
```

注释位置：`output.py:181` 到 `output.py:189`

这样做的目的：

```text
减少 EngineCore → Worker 的通信量；
避免每轮重复传 prompt / sampling params / mm features；
让 Worker 侧持续维护 batch，提升调度和执行效率。
```

因此，Worker 侧必须维护：

```text
self.requests：请求缓存全集；
self.input_batch：当前执行 batch。
```

---

## 7. _update_states() 的入口

入口在：`gpu_model_runner.py:1127`

```python
def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
```

注释说明：

```python
"""Update the cached states and the persistent batch with the scheduler
output.

The updated states are used by the `_prepare_inputs` function to create
the input GPU tensors for the model.

The SamplingMetadata is updated and copied to the GPU if there is a
new/resumed/paused/finished request in the batch.
"""
```

位置：`gpu_model_runner.py:1127` 到 `gpu_model_runner.py:1136`

这说明 `_update_states()` 的产物直接服务于后面的：

```text
_prepare_inputs()
_build_attention_metadata()
_model_forward()
_sample()
```

---

## 8. 第一步：删除 finished 请求

```python
for req_id in scheduler_output.finished_req_ids:
    self.requests.pop(req_id, None)
    self.num_prompt_logprobs.pop(req_id, None)
```

位置：`gpu_model_runner.py:1137` 到 `gpu_model_runner.py:1140`

然后通知 late interaction runner：

```python
self.late_interaction_runner.on_requests_finished(
    scheduler_output.finished_req_ids
)
```

位置：`gpu_model_runner.py:1141` 到 `gpu_model_runner.py:1143`

再从持久 batch 中删除：

```python
for req_id in scheduler_output.finished_req_ids:
    self.input_batch.remove_request(req_id)
```

位置：`gpu_model_runner.py:1150` 到 `gpu_model_runner.py:1151`

状态变化：

```text
finished_req_id
  → 从 self.requests 删除
  → 从 InputBatch 删除
  → 从 prompt_logprobs / late interaction 状态删除
```

这里有一个注释中特别说明的边界：

```text
finished_req_ids 和 scheduled_req_ids 可能重叠。
```

例如一个请求被 abort 后，又用同一个 request id 重新提交。Worker 会把它当成两个不同请求：先清理旧请求，再按新请求重新加入。

位置：`gpu_model_runner.py:1144` 到 `gpu_model_runner.py:1149`

---

## 9. 第二步：zero 新 KV block，释放 encoder cache

### 9.1 zero 新分配 block

```python
if scheduler_output.new_block_ids_to_zero:
    self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
```

位置：`gpu_model_runner.py:1153` 到 `gpu_model_runner.py:1156`

作用：

```text
Scheduler 本轮新分配了 KV cache block；
Worker 在使用前先清零 GPU memory；
避免 stale NaN / 脏数据污染 attention 或 SSM 计算。
```

### 9.2 释放 encoder cache

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

位置：`gpu_model_runner.py:1158` 到 `gpu_model_runner.py:1160`

这和多模态 / encoder-decoder 请求有关。

---

## 10. 第三步：从 InputBatch 移除本轮未调度请求

Worker 侧的 `InputBatch` 是“当前执行 batch”，不是 Scheduler 的 running 全集。

因此，本轮没有被调度的请求，即使还没结束，也要从当前 `InputBatch` 移除。

源码：

```python
scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
cached_req_ids = self.input_batch.req_id_to_index.keys()
resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
for req_id in unscheduled_req_ids:
    self.input_batch.remove_request(req_id)
```

位置：`gpu_model_runner.py:1162` 到 `gpu_model_runner.py:1182`

含义：

```text
InputBatch 中已有的请求
  - 本轮仍然 scheduled 的请求
  + 特殊 resumed 处理
  = 本轮应该保留的请求

其他请求从 InputBatch 移除。
```

注意：

```text
从 InputBatch 移除，不等于从 self.requests 删除。
```

注释明确说明：

```text
这些 unscheduled requests 可能是 preempted requests，
也可能是 running 但本轮没被调度的请求。
它们会从持久 batch 移除，但 cached states 会保留，
因为未来还可能再次被调度。
```

位置：`gpu_model_runner.py:1162` 到 `gpu_model_runner.py:1166`

---

## 11. 第四步：处理 scheduled_new_reqs

新请求来自：

```python
scheduler_output.scheduled_new_reqs
```

位置：`gpu_model_runner.py:1195`

### 11.1 streaming session 特殊情况

如果新请求 id 已经在 `self.requests` 中：

```python
if req_id in self.requests:
    req_state = self._update_streaming_request(req_id, new_req_data)
    reqs_to_add.append(req_state)
    continue
```

位置：`gpu_model_runner.py:1195` 到 `gpu_model_runner.py:1201`

这通常是 streaming input 场景。

`_update_streaming_request()` 会：

```text
1. 从 InputBatch 移除旧请求；
2. 更新 prompt_token_ids / mm_features / prompt_embeds；
3. 更新 sampling_params / pooling_params；
4. 更新 block_ids / num_computed_tokens；
5. 清空 output_token_ids，因为之前 output tokens 现在变成 prompt 的一部分。
```

位置：`gpu_model_runner.py:1555` 到 `gpu_model_runner.py:1588`

状态变化：

```text
已有 streaming request
  → 更新为新输入 chunk 对应的新上下文
  → 后续重新加入 InputBatch
```

### 11.2 普通新请求

普通新请求会构造 `CachedRequestState`：

```python
req_state = CachedRequestState(
    req_id=req_id,
    prompt_token_ids=new_req_data.prompt_token_ids,
    prompt_embeds=new_req_data.prompt_embeds,
    prompt_is_token_ids=new_req_data.prompt_is_token_ids,
    mm_features=new_req_data.mm_features,
    sampling_params=sampling_params,
    pooling_params=pooling_params,
    generator=generator,
    block_ids=new_req_data.block_ids,
    num_computed_tokens=new_req_data.num_computed_tokens,
    output_token_ids=[],
    lora_request=new_req_data.lora_request,
)
```

位置：`gpu_model_runner.py:1224` 到 `gpu_model_runner.py:1237`

然后加入 Worker 请求缓存：

```python
self.requests[req_id] = req_state
```

位置：`gpu_model_runner.py:1238`

如果需要 prompt logprobs，会记录：

```python
self.num_prompt_logprobs[req_id] = ...
```

位置：`gpu_model_runner.py:1241` 到 `gpu_model_runner.py:1246`

如果模型需要特殊位置编码，也会初始化：

```text
M-RoPE positions
XD-RoPE positions
```

位置：`gpu_model_runner.py:1248` 到 `gpu_model_runner.py:1254`

---

## 12. 第五步：更新 scheduled_cached_reqs

已缓存请求来自：

```python
req_data = scheduler_output.scheduled_cached_reqs
```

位置：`gpu_model_runner.py:1261` 到 `gpu_model_runner.py:1264`

它包含的是增量信息，而不是完整请求数据。

Worker 会遍历：

```python
for i, req_id in enumerate(req_data.req_ids):
    req_state = self.requests[req_id]
    num_computed_tokens = req_data.num_computed_tokens[i]
    new_block_ids = req_data.new_block_ids[i]
    resumed_from_preemption = req_id in req_data.resumed_req_ids
    num_output_tokens = req_data.num_output_tokens[i]
    req_index = self.input_batch.req_id_to_index.get(req_id)
```

位置：`gpu_model_runner.py:1284` 到 `gpu_model_runner.py:1290`

这一步主要更新：

```text
num_computed_tokens
new block ids
output token ids
spec decode token ids
resumed/preempted 状态
async scheduling 下的 draft token 修正
```

### 12.1 更新 num_computed_tokens

```python
req_state.num_computed_tokens = num_computed_tokens
```

位置：`gpu_model_runner.py:1334` 到 `gpu_model_runner.py:1335`

这是 Worker 侧和 Scheduler 侧 token 进度同步的关键。

### 12.2 更新 spec decode 状态

`scheduled_spec_decode_tokens` 来自 Scheduler，Worker 会记录当前请求本轮的 draft tokens。

相关入口：

```python
scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
```

位置：`gpu_model_runner.py:1263` 到 `gpu_model_runner.py:1264`

InputBatch 侧有：

```python
def update_req_spec_token_ids(...)
```

位置：`gpu_input_batch.py:483`

它会：

```text
1. 清空当前 req_index 的 spec_token_ids；
2. 读取 scheduler_output.scheduled_spec_decode_tokens[req_id]；
3. 写入 token_ids_cpu 的 spec token 区间；
4. 更新 request.prev_num_draft_len。
```

位置：`gpu_input_batch.py:483` 到 `gpu_input_batch.py:508`

### 12.3 async spec decode 的延迟修正

在 async scheduling + spec decode 场景，Worker 可能先乐观认为 draft tokens 都被接受，后续再修正。

源码注释从：`gpu_model_runner.py:1292` 开始说明。

关键逻辑：

```text
prev_num_draft_len > 0
  → 先给 output_token_ids 填充占位
  → 记录 deferred_spec_decode_corrections
  → 等模型 forward 后再纠正
```

位置：`gpu_model_runner.py:1292` 到 `gpu_model_runner.py:1333`

这部分容易混淆，核心是：

```text
异步调度下 Scheduler 可能提前调度下一步，
Worker 侧需要在不阻塞 forward 的情况下维护 token 计数一致性。
```

---

## 13. 第六步：把请求加入 InputBatch

前面 `self.requests` 是请求缓存全集；但真正参与当前 forward 的请求还必须进入 `InputBatch`。

InputBatch 的添加入口：

```python
def add_request(self, request: CachedRequestState) -> int:
```

位置：`gpu_input_batch.py:335`

### 13.1 分配 req_index

```python
req_index = self._register_add_request(request)
```

位置：`gpu_input_batch.py:339`

`_register_add_request()` 会优先复用刚删除请求留下的空位；没有空位则追加到 batch 尾部。

位置：`gpu_input_batch.py:309` 到 `gpu_input_batch.py:333`

### 13.2 更新 req_id 映射

```python
self._req_ids.append(req_id) 或 self._req_ids[req_index] = req_id
self.req_id_to_index[req_id] = req_index
```

位置：`gpu_input_batch.py:341` 到 `gpu_input_batch.py:351`

这就是：

```text
request_id → batch row index
```

的核心映射。

### 13.3 复制 token 状态

```python
self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
self.num_tokens_no_spec[req_index] = request.num_tokens
self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
```

位置：`gpu_input_batch.py:353` 到 `gpu_input_batch.py:377`

这一步把单请求状态展开到 batch 矩阵里。

### 13.4 更新 block table

```python
self.block_table.add_row(request.block_ids, req_index)
```

位置：`gpu_input_batch.py:378`

这把请求的 KV block ids 写入当前 batch 的 block table。

后续 attention metadata / slot mapping 会依赖它。

### 13.5 更新 sampling 参数

如果是 generation 请求，会写入：

```text
temperature
top_p
top_k
frequency penalty
presence penalty
repetition penalty
generator
logprobs
allowed token ids
bad words
```

位置：`gpu_input_batch.py:380` 到 `gpu_input_batch.py:452`

### 13.6 更新 pooling 状态

如果是 pooling 请求，会写入：

```text
pooling_params
pooling_states
logits_processing_needs_token_ids
```

位置：`gpu_input_batch.py:453` 到 `gpu_input_batch.py:463`

### 13.7 更新 LoRA 映射

```python
self.request_lora_mapping[req_index] = lora_id
self.lora_id_to_request_ids[lora_id].add(request.req_id)
self.lora_id_to_lora_request[lora_id] = request.lora_request
```

位置：`gpu_input_batch.py:468` 到 `gpu_input_batch.py:480`

---

## 14. 第七步：删除请求后的 condense

`InputBatch.remove_request()` 会把请求所在位置标记为空，但不会立刻把整个 batch 紧凑化。

入口：

```python
def remove_request(self, req_id: str) -> int | None:
```

位置：`gpu_input_batch.py:510`

它会：

```text
1. 从 req_id_to_index 删除 req_id；
2. 把 _req_ids[req_index] 设为 None；
3. 清理 req_output_token_ids / spec_token_ids；
4. 清理 block table row；
5. 清理 LoRA / sampling / pooling 状态；
6. 记录 batch_update_builder.removed。
```

位置：`gpu_input_batch.py:520` 到 `gpu_input_batch.py:564`

注意注释：

```python
"""This method must always be followed by a call to condense()."""
```

位置：`gpu_input_batch.py:510` 到 `gpu_input_batch.py:512`

### 14.1 condense 做什么

```python
def condense(self) -> None:
```

位置：`gpu_input_batch.py:683`

它会把后面的非空请求向前移动，填补空洞。

核心逻辑：

```text
找到最小 empty index；
找到最大的非空 index；
把大的非空请求移动到小的空洞；
更新 req_id_to_index；
移动 token ids、block table、sampling 参数、LoRA 映射等；
最后裁剪 _req_ids / req_output_token_ids / spec_token_ids。
```

位置：`gpu_input_batch.py:683` 到 `gpu_input_batch.py:810`

所以 InputBatch 会尽量保持：

```text
当前 batch 的请求行是紧凑排列的。
```

---

## 15. 第八步：刷新 sampling metadata

InputBatch 有 batch update 记录器：

```python
self.batch_update_builder = BatchUpdateBuilder()
```

位置：`gpu_input_batch.py:260` 到 `gpu_input_batch.py:263`

它会记录：

```text
added requests
removed requests
moved requests
batch_changed
```

当 batch 变化后，需要刷新 sampling metadata：

```python
def refresh_metadata(self):
```

位置：`gpu_input_batch.py:811`

对非 pooling 模型：

```python
batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
...
for logit_proc in self.logitsprocs.all:
    logit_proc.update_state(batch_update)
if batch_update:
    self.sampling_metadata = self._make_sampling_metadata()
```

位置：`gpu_input_batch.py:820` 到 `gpu_input_batch.py:829`

`_make_sampling_metadata()` 会把 CPU 侧的采样参数同步到 GPU tensor，并构造采样需要的 metadata。

位置：`gpu_input_batch.py:831` 起

---

## 16. 执行后状态如何更新

除了执行前的 `_update_states()`，在 `sample_tokens()` 中采样完成后，还会调用：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`gpu_model_runner.py:4461` 到 `gpu_model_runner.py:4463`

定义在：`gpu_model_runner.py:1497`

```python
def _update_states_after_model_execute(
    self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
) -> None:
```

它主要服务于：

```text
speculative decoding + hybrid models / Mamba 等场景。
```

核心逻辑：

```text
1. 统计每个请求实际 accepted token 数；
2. 把 accepted count 写入 num_accepted_tokens；
3. 对 Mamba / hybrid state 做 postprocess；
4. 记录 event，供后续异步读取。
```

位置：`gpu_model_runner.py:1497` 到 `gpu_model_runner.py:1553`

普通非 spec / 非 hybrid 模型会直接返回：

```python
if not self.speculative_config or not self.model_config.is_hybrid:
    return
```

位置：`gpu_model_runner.py:1508` 到 `gpu_model_runner.py:1509`

---

## 17. InputBatch 与 _prepare_inputs 的关系

`_update_states()` 的注释明确说：

```text
更新后的状态会被 _prepare_inputs() 用来创建模型输入 GPU tensors。
```

位置：`gpu_model_runner.py:1131` 到 `gpu_model_runner.py:1132`

也就是说：

```text
_update_states()
  → 更新 req_id / token ids / block table / sampling params
  → _prepare_inputs()
  → 生成 input_ids / positions / logits_indices / spec metadata
  → _build_attention_metadata()
  → forward
```

如果 `InputBatch` 的 req_index、token_ids、block_table 不一致，后续会直接影响：

```text
input_ids
positions
slot_mapping
block table
attention metadata
sampling metadata
```

---

## 18. 和 Scheduler 状态的边界

### 18.1 Scheduler 保存什么

Scheduler 负责：

```text
Request.status
waiting / running / skipped_waiting
num_computed_tokens
KV block allocation
spec decode 接受 / 拒绝后的请求状态
finished / preempt / abort
```

### 18.2 Worker 保存什么

Worker / ModelRunner 保存：

```text
请求的执行侧缓存数据；
当前 batch 的数组化表示；
token_ids_cpu / block_table / sampling metadata；
LoRA / pooling / spec decode / M-RoPE / encoder cache 执行侧状态。
```

### 18.3 它们如何同步

```text
SchedulerOutput：Scheduler → Worker，同步计划和状态 diff；
ModelRunnerOutput：Worker → Scheduler，同步实际执行结果。
```

---

## 19. 容易疑惑的点

### 19.1 `self.requests` 和 `InputBatch` 是一回事吗？

不是。

```text
self.requests：
  Worker 侧缓存过的请求状态全集。

InputBatch：
  当前这一轮执行需要的紧凑 batch。
```

一个请求可能还在 `self.requests`，但本轮没被调度，因此不在 `InputBatch`。

### 19.2 从 InputBatch remove 是否表示请求结束？

不一定。

如果请求是本轮未调度、被抢占、或者暂时跳过，它会从 InputBatch 移除，但仍保留在 `self.requests`。

只有 `finished_req_ids` 才会从 `self.requests` 删除。

### 19.3 为什么要 condense？

因为删除请求后 batch 中会出现空洞。

为了让后续 GPU tensor、attention metadata、采样 metadata 都按紧凑 batch 处理，需要把后面的请求移动到前面的空位。

### 19.4 为什么 scheduled_new_reqs 和 scheduled_cached_reqs 要分开？

为了减少通信。

新请求需要发送完整请求数据；已缓存请求只需要发送本轮变化的 diff。

### 19.5 为什么 streaming request 会以 new request 形式出现但 req_id 已存在？

因为 streaming input 的后续 chunk 会更新同一个 request 的 prompt / block / computed 状态。

Worker 侧会用 `_update_streaming_request()` 更新缓存，并重新加入 InputBatch。

### 19.6 async spec decode 为什么有 deferred corrections？

因为异步调度可能在上一轮真实采样结果完全回收前就调度下一轮。

Worker 侧先乐观推进 draft token 数，再在合适时机修正，避免阻塞模型 forward。

---

## 20. 总结

Worker / ModelRunner 维护 batch 和请求状态的核心是：

```text
self.requests：请求缓存状态
InputBatch：当前执行 batch 状态
SchedulerOutput：每轮状态 diff
_update_states()：把 diff 应用到 Worker 侧状态
```

完整流程可以记为：

```text
SchedulerOutput
  → 删除 finished 请求
  → 移除本轮未调度请求
  → 缓存 scheduled_new_reqs
  → 更新 scheduled_cached_reqs
  → add_request 到 InputBatch
  → condense 紧凑 batch
  → refresh_metadata 刷新采样状态
  → _prepare_inputs / forward / sample
```

如果只记一句话：

```text
Scheduler 维护全局调度状态，InputBatch 维护 Worker 当前执行 batch；_update_states() 是两者每轮对齐的桥。
```
