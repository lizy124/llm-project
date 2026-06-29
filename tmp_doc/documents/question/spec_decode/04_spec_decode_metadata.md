# 04. SpecDecodeMetadata 如何描述 draft / target / bonus 位置？

源码位置：

- `code/vllm/vllm/v1/spec_decode/metadata.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/spec_decode/utils.py`
- `code/vllm/vllm/v1/core/sched/output.py`

本问题关注：`SchedulerOutput.scheduled_spec_decode_tokens` 进入 `GPUModelRunner._prepare_inputs()` 后，如何被转换成 `SpecDecodeMetadata`；`SpecDecodeMetadata` 里的各个索引如何描述 target verification logits、bonus logits、draft token ids；这些字段如何被 `RejectionSampler`、logprobs、draft proposer 和部分 attention backend 消费。

---

## 1. 一句话回答

`SpecDecodeMetadata` 是 spec decode 在 ModelRunner 内部的 **token 行号说明书**。

它把 Scheduler 侧的：

```text
每个请求有哪些 scheduled draft tokens
```

翻译成 Worker / sampler 可执行的：

```text
1. target model 哪些 logits row 用来验证 draft tokens；
2. 哪些 logits row 用来采样 bonus token；
3. flattened draft_token_ids 如何和 target logits 对齐；
4. 每个请求有几个 draft tokens；
5. 每个请求最终最多会返回几个 sampled tokens。
```

如果只记一句话：

```text
SpecDecodeMetadata 不是保存“请求状态”，而是保存“本轮 spec decode 的 logits 行号布局”。
```

---

## 2. 它解决什么问题

普通 decode 中，每个请求通常只需要一个 logits row：

```text
请求 i：
  取最后一个 query token 的 logits
  → sample 1 个 token
```

spec decode 中，每个请求可能有 K 个 draft tokens：

```text
请求 i：
  target model 需要验证 K 个 draft token
  如果 K 个都接受，还要从 target logits 采样 1 个 bonus token
```

所以每个请求最多需要：

```text
K 个 target verification logits
+ 1 个 bonus logits
= K + 1 个 logits rows
```

`SpecDecodeMetadata` 就是把这些 rows 的位置展开并记录下来。

---

## 3. 核心对象关系

```text
SchedulerOutput.scheduled_spec_decode_tokens
  → GPUModelRunner._prepare_inputs()
      → num_draft_tokens
      → _calc_spec_decode_metadata()
          → SpecDecodeMetadata
  → hidden_states[logits_indices]
  → model.compute_logits(sample_hidden_states)
  → GPUModelRunner._sample(logits, spec_decode_metadata)
  → RejectionSampler.forward(metadata, draft_probs, logits, sampling_metadata)
```

其中最关键的是：

```text
logits_indices：
  决定 compute_logits 输入哪些 hidden states。

SpecDecodeMetadata.target_logits_indices / bonus_logits_indices：
  在 compute_logits 结果中，再区分 target verification rows 和 bonus rows。
```

---

## 4. SpecDecodeMetadata 的字段定义

定义位置：`metadata.py:9`

```python
@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor
```

位置：`metadata.py:9` 到 `metadata.py:24`

字段含义：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `draft_token_ids` | `[num_draft_tokens_total]` | flatten 后的 draft token ids |
| `num_draft_tokens` | `[batch_size]` | 每个请求本轮有几个 draft tokens |
| `cu_num_draft_tokens` | `[batch_size]` | `num_draft_tokens` 的 inclusive cumsum |
| `cu_num_sampled_tokens` | `[batch_size]` | 每个请求 `draft_len + 1` 的 inclusive cumsum |
| `target_logits_indices` | `[num_draft_tokens_total]` | 在 logits tensor 中，哪些行用于验证 draft tokens |
| `bonus_logits_indices` | `[batch_size]` | 在 logits tensor 中，哪些行用于 bonus token sampling |
| `logits_indices` | `[num_draft_tokens_total + batch_size]` | 从 full hidden states 中取哪些行去 compute logits |

`__post_init__()` 还会计算：

```python
self.max_spec_len = max(self.num_draft_tokens)
```

位置：`metadata.py:26` 到 `metadata.py:27`

这个值后面用于 `RejectionSampler` 输出矩阵宽度。

---

## 5. 字段之间的层次关系

可以把这些字段分成三层。

### 5.1 请求级长度信息

```text
num_draft_tokens
cu_num_draft_tokens
cu_num_sampled_tokens
max_spec_len
```

它们回答：

```text
每个请求有几个 draft token？
flatten 后每个请求的 draft 区间在哪里？
每个请求最多返回几个 sampled token？
```

### 5.2 logits 行号信息

```text
logits_indices
target_logits_indices
bonus_logits_indices
```

它们回答：

```text
从 hidden_states 里取哪些行算 logits？
算出来的 logits 中哪些行用于 draft verification？
哪些行用于 bonus sampling？
```

### 5.3 draft token 本体

```text
draft_token_ids
```

它回答：

```text
target logits 要验证的 token id 是什么？
```

---

## 6. SpecDecodeMetadata 在哪里构造

构造入口在 `GPUModelRunner._prepare_inputs()`：`gpu_model_runner.py:1889`

当本轮没有 spec decode：

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
if not use_spec_decode:
    logits_indices = query_start_loc[1:] - 1
    spec_decode_metadata = None
    num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
```

位置：`gpu_model_runner.py:2153` 到 `gpu_model_runner.py:2162`

当本轮有 scheduled draft tokens：

```python
num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
    req_idx = self.input_batch.req_id_to_index[req_id]
    draft_len = len(draft_token_ids)
    num_draft_tokens[req_idx] = draft_len

spec_decode_metadata = self._calc_spec_decode_metadata(
    num_draft_tokens, cu_num_tokens
)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

位置：`gpu_model_runner.py:2163` 到 `gpu_model_runner.py:2187`

也就是说：

```text
SchedulerOutput.scheduled_spec_decode_tokens
  只给 req_id -> draft token list。

_prepare_inputs()
  把它转换成按 InputBatch req_index 排列的 num_draft_tokens。

_calc_spec_decode_metadata()
  再把长度信息转换成 logits row 布局。
```

---

## 7. _prepare_inputs() 里已经准备好的基础布局

在调用 `_calc_spec_decode_metadata()` 前，`_prepare_inputs()` 已经准备了本轮 token 序列的基础布局。

### 7.1 req_indices

```python
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
```

位置：`gpu_model_runner.py:1910` 到 `gpu_model_runner.py:1912`

例子：

```text
num_scheduled_tokens = [2, 5, 3]
req_indices = [0,0, 1,1,1,1,1, 2,2,2]
```

### 7.2 cu_num_tokens

```python
cu_num_tokens = self._get_cumsum_and_arange(
    num_scheduled_tokens, self.query_pos.np
)
```

位置：`gpu_model_runner.py:1914` 到 `gpu_model_runner.py:1918`

例子：

```text
num_scheduled_tokens = [2, 5, 3]
cu_num_tokens = [2, 7, 10]
```

`cu_num_tokens` 是每个请求在本轮 flattened query tokens 中的结束位置。

### 7.3 input_ids

`_prepare_inputs()` 会用 request positions 到 `InputBatch.token_ids_cpu_tensor` 里取本轮 token ids：

```python
torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),
    0,
    token_indices_tensor,
    out=self.input_ids.cpu[:total_num_scheduled_tokens],
)
```

位置：`gpu_model_runner.py:1945` 到 `gpu_model_runner.py:1953`

这一步之后，本轮 scheduled tokens 已经排成一个 flattened `input_ids`：

```text
input_ids:
  [req0 本轮 tokens][req1 本轮 tokens][req2 本轮 tokens]...
```

`_calc_spec_decode_metadata()` 后面会从这条 flattened `input_ids` 中提取 draft token ids。

---

## 8. _calc_spec_decode_metadata() 的输入输出

入口：`gpu_model_runner.py:2742`

```python
def _calc_spec_decode_metadata(
    self,
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
) -> SpecDecodeMetadata:
```

位置：`gpu_model_runner.py:2742` 到 `gpu_model_runner.py:2746`

输入：

```text
num_draft_tokens：
  每个请求有几个 draft tokens。

cu_num_scheduled_tokens：
  每个请求本轮 scheduled tokens 的 inclusive cumsum。
```

源码注释给了一个完整例子：

```text
cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
num_draft_tokens:         [  3,   0,   2,   0,   1]
```

位置：`gpu_model_runner.py:2747` 到 `gpu_model_runner.py:2749`

输出：

```text
cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
                             206, 207, 208]
target_logits_indices:    [  0,   1,   2,   5,   6,   9]
bonus_logits_indices:     [  3,   4,   7,   8,  10]
```

位置：`gpu_model_runner.py:2750` 到 `gpu_model_runner.py:2755`

---

## 9. 为什么每个请求是 draft_len + 1 个 sampled rows

第一步：

```python
num_sampled_tokens = num_draft_tokens + 1
```

位置：`gpu_model_runner.py:2757` 到 `gpu_model_runner.py:2759`

如果：

```text
num_draft_tokens = [3, 0, 2, 0, 1]
```

那么：

```text
num_sampled_tokens = [4, 1, 3, 1, 2]
```

原因：

```text
每个请求除了验证 draft tokens 外，还需要一个 bonus logits row。
```

即使某个请求没有 draft tokens，也仍然需要 1 个普通 sampling row。

---

## 10. logits_indices 如何计算

`logits_indices` 表示：

```text
从本轮 full hidden_states 中取哪些 token positions 来 compute logits。
```

计算分三步。

### 10.1 计算 cu_num_sampled_tokens

```python
cu_num_sampled_tokens = self._get_cumsum_and_arange(
    num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32
)
```

位置：`gpu_model_runner.py:2763` 到 `gpu_model_runner.py:2766`

例子：

```text
num_sampled_tokens = [4, 1, 3, 1, 2]
cu_num_sampled_tokens = [4, 5, 8, 9, 11]
```

### 10.2 找每个请求 sampling 区间的起点

```python
logits_indices = np.repeat(
    cu_num_scheduled_tokens - num_sampled_tokens,
    num_sampled_tokens,
)
```

位置：`gpu_model_runner.py:2767` 到 `gpu_model_runner.py:2770`

例子：

```text
cu_num_scheduled_tokens - num_sampled_tokens
= [0, 103, 104, 206, 207]

repeat 后：
[0,0,0,0, 103, 104,104,104, 206, 207,207]
```

含义：

```text
每个请求只关心它本轮 scheduled token 的最后 draft_len + 1 个位置。
```

### 10.3 加上局部 offset

```python
logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]
```

位置：`gpu_model_runner.py:2771` 到 `gpu_model_runner.py:2772`

最终：

```text
logits_indices = [0,1,2,3, 103, 104,105,106, 206, 207,208]
```

这表示：

```text
从 full hidden_states 中取这些 rows，送入 compute_logits。
```

---

## 11. logits_indices 为什么包含 bonus row

对每个请求，`logits_indices` 包含：

```text
[draft verification positions..., bonus position]
```

例如第一个请求 draft_len=3：

```text
logits_indices for req0 = [0, 1, 2, 3]
```

其中：

```text
0,1,2：用于验证 3 个 draft tokens
3：用于 all accepted 时采样 bonus token
```

所以 `logits_indices` 的长度是：

```text
sum(num_draft_tokens + 1)
= total_draft_tokens + batch_size
```

这和 `SpecDecodeMetadata.logits_indices` 注释一致：

```python
# [num_tokens + batch_size]
logits_indices: torch.Tensor
```

位置：`metadata.py:23` 到 `metadata.py:24`

---

## 12. bonus_logits_indices 如何计算

```python
bonus_logits_indices = cu_num_sampled_tokens - 1
```

位置：`gpu_model_runner.py:2774` 到 `gpu_model_runner.py:2775`

例子：

```text
cu_num_sampled_tokens = [4,5,8,9,11]
bonus_logits_indices = [3,4,7,8,10]
```

注意这里的 index 是相对于：

```text
compute_logits 后的 logits tensor
```

不是相对于 full hidden_states。

也就是说：

```text
hidden_states[logits_indices]
  → sample_hidden_states
  → compute_logits(sample_hidden_states)
  → logits

bonus_logits_indices 是 logits 里的行号。
```

---

## 13. target_logits_indices 如何计算

`target_logits_indices` 表示：

```text
compute_logits 后的 logits tensor 中，哪些 rows 用于验证 draft tokens。
```

先计算 draft 的累计长度：

```python
cu_num_draft_tokens = self._get_cumsum_and_arange(
    num_draft_tokens, self._arange_scratch, cumsum_dtype=np.int32
)
```

位置：`gpu_model_runner.py:2778` 到 `gpu_model_runner.py:2782`

例子：

```text
num_draft_tokens = [3,0,2,0,1]
cu_num_draft_tokens = [3,3,5,5,6]
```

再计算每个请求在 logits tensor 中的 sampled 区间起点：

```python
target_logits_indices = np.repeat(
    cu_num_sampled_tokens - num_sampled_tokens,
    num_draft_tokens,
)
```

位置：`gpu_model_runner.py:2783` 到 `gpu_model_runner.py:2786`

再加上局部 draft offset：

```python
target_logits_indices += self._arange_scratch[: cu_num_draft_tokens[-1]]
```

位置：`gpu_model_runner.py:2787` 到 `gpu_model_runner.py:2788`

最终例子：

```text
target_logits_indices = [0,1,2, 5,6, 9]
```

这些 rows 和 flattened `draft_token_ids` 一一对齐。

---

## 14. draft_token_ids 如何从 input_ids 里取出来

`_calc_spec_decode_metadata()` 最后计算 draft token ids：

```python
draft_token_ids = self.input_ids.gpu[logits_indices]
draft_token_ids = draft_token_ids[target_logits_indices + 1]
```

位置：`gpu_model_runner.py:2807` 到 `gpu_model_runner.py:2810`

这两步容易混淆。

第一步：

```text
self.input_ids.gpu[logits_indices]
```

取出的是本轮每个请求 `draft_len + 1` 个位置对应的 input ids：

```text
[target verification positions..., bonus position]
```

第二步：

```text
target_logits_indices + 1
```

是因为：

```text
target logits row j 验证的是下一个 token，也就是 input_ids[j + 1]。
```

所以：

```text
logits row 0 → 验证 draft token at local input row 1
logits row 1 → 验证 draft token at local input row 2
logits row 2 → 验证 draft token at local input row 3
```

这也是为什么每个请求需要 `draft_len + 1` 个 hidden/logits rows：

```text
K 个 draft tokens 的验证，需要 K 个 target logits；
这些 target logits 对应的位置是 draft 前一个上下文位置到倒数第二个 draft 位置；
最后一个 logits row 作为 bonus row。
```

---

## 15. 一个完整索引例子

源码注释例子：

```text
cu_num_scheduled_tokens = [4, 104, 107, 207, 209]
num_draft_tokens        = [3,   0,   2,   0,   1]
```

表示有 5 个请求：

```text
req0 本轮 scheduled 4 tokens，其中 3 个 draft
req1 本轮 scheduled 100 tokens，其中 0 个 draft
req2 本轮 scheduled 3 tokens，其中 2 个 draft
req3 本轮 scheduled 100 tokens，其中 0 个 draft
req4 本轮 scheduled 2 tokens，其中 1 个 draft
```

`num_sampled_tokens = draft_len + 1`：

```text
[4, 1, 3, 1, 2]
```

对应 `logits_indices`：

```text
req0: [0, 1, 2, 3]
req1: [103]
req2: [104, 105, 106]
req3: [206]
req4: [207, 208]
```

合并后：

```text
[0,1,2,3, 103, 104,105,106, 206, 207,208]
```

`target_logits_indices`：

```text
req0: logits rows [0,1,2] 验证 3 个 draft
req2: logits rows [5,6] 验证 2 个 draft
req4: logits row  [9]   验证 1 个 draft
```

合并后：

```text
[0,1,2, 5,6, 9]
```

`bonus_logits_indices`：

```text
req0: 3
req1: 4
req2: 7
req3: 8
req4: 10
```

合并后：

```text
[3,4,7,8,10]
```

---

## 16. 为什么无 draft 请求也有 bonus_logits_indices

在 mixed batch 中，有些请求可能没有 draft tokens。

例如：

```text
num_draft_tokens = [3, 0, 2]
```

第二个请求没有 draft，但仍然需要正常 decode 采样 1 个 token。

因此它的：

```text
num_sampled_tokens = 0 + 1 = 1
```

这个唯一的 logits row 会出现在 `bonus_logits_indices` 中。

对无 draft 请求来说：

```text
bonus token 其实就是普通 decode 的 sampled token。
```

这让 `RejectionSampler` 可以用同一套 `[draft verification + bonus]` 格式处理 mixed batch。

---

## 17. SpecDecodeMetadata 如何影响 compute_logits

`execute_model()` 中，模型 forward 先得到 full hidden states。

然后只对 `logits_indices` 指定的 rows 计算 logits：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`gpu_model_runner.py:4354` 到 `gpu_model_runner.py:4355`

这一步非常重要：

```text
full hidden_states：
  包含本轮所有 scheduled tokens 的 hidden states。

sample_hidden_states：
  只保留需要采样 / 验证的位置。

logits：
  shape = [total_draft_tokens + batch_size, vocab_size]
```

所以 `SpecDecodeMetadata.logits_indices` 直接决定：

```text
compute_logits 的行数和行顺序。
```

---

## 18. SpecDecodeMetadata 如何进入 sample_tokens()

`execute_model()` 结束时会把 metadata 暂存进 `ExecuteModelState`：

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
```

位置：`gpu_model_runner.py:4386` 到 `gpu_model_runner.py:4397`

随后 `sample_tokens()` 解包：

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
```

位置：`gpu_model_runner.py:4436` 到 `gpu_model_runner.py:4448`

因此：

```text
_prepare_inputs() 生成的 SpecDecodeMetadata
  → execute_model_state
  → sample_tokens()
  → _sample()
```

---

## 19. _sample() 如何根据 metadata 分流

入口：`gpu_model_runner.py:3570`

如果没有 spec decode metadata：

```python
if spec_decode_metadata is None:
    return self.sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
```

位置：`gpu_model_runner.py:3580` 到 `gpu_model_runner.py:3584`

如果有 metadata：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`gpu_model_runner.py:3592` 到 `gpu_model_runner.py:3598`

所以：

```text
spec_decode_metadata is None
  → 普通 Sampler

spec_decode_metadata exists
  → RejectionSampler
```

---

## 20. RejectionSampler 如何使用 metadata

入口：`rejection_sampler.py:88`

```python
def forward(
    self,
    metadata: SpecDecodeMetadata,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
```

位置：`rejection_sampler.py:88` 到 `rejection_sampler.py:96`

### 20.1 先取 bonus logits

```python
bonus_logits_indices = metadata.bonus_logits_indices
bonus_logits = logits[bonus_logits_indices]
bonus_sampler_output = self.sampler(..., predict_bonus_token=True, ...)
bonus_token_ids = bonus_sampler_output.sampled_token_ids
```

位置：`rejection_sampler.py:121` 到 `rejection_sampler.py:143`

含义：

```text
bonus logits 用普通 Sampler 采样。
如果某个请求所有 draft 都 accepted，就把 bonus token 接到末尾。
```

### 20.2 再取 target logits

```python
target_logits_indices = metadata.target_logits_indices
raw_target_logits = logits[target_logits_indices]
```

位置：`rejection_sampler.py:121` 到 `rejection_sampler.py:150`

这些 logits 和 `metadata.draft_token_ids` 一一对齐：

```text
target_logits[j] 验证 draft_token_ids[j]
```

### 20.3 应用 logits processors / penalties / constraints

```python
target_logits = self.apply_logits_processors(
    target_logits, sampling_metadata, metadata
)
target_logits = apply_sampling_constraints(
    target_logits,
    metadata.cu_num_draft_tokens,
    sampling_metadata,
)
```

位置：`rejection_sampler.py:157` 到 `rejection_sampler.py:167`

这里 `cu_num_draft_tokens` 用来把 batch 级采样参数扩展到 token 级。

### 20.4 执行 rejection_sample

```python
output_token_ids = rejection_sample(
    metadata.draft_token_ids,
    metadata.num_draft_tokens,
    metadata.max_spec_len,
    metadata.cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
    ...
)
```

位置：`rejection_sampler.py:169` 到 `rejection_sampler.py:181`

返回的 `output_token_ids` 形状是：

```text
[batch_size, max_spec_len + 1]
```

---

## 21. RejectionSampler 输出如何表达 rejected tokens

`rejection_sample()` 会创建输出 buffer：

```python
output_token_ids = torch.full(
    (batch_size, max_spec_len + 1),
    PLACEHOLDER_TOKEN_ID,
    dtype=torch.int32,
    device=device,
)
```

位置：`rejection_sampler.py:427` 到 `rejection_sampler.py:433`

其中：

```text
PLACEHOLDER_TOKEN_ID = -1
```

位置：`rejection_sampler.py:30`

含义：

```text
accepted / recovered / bonus tokens：真实 token id
rejected 或无效位置：-1
```

后续 `_bookkeeping_sync()` 会通过：

```python
RejectionSampler.parse_output(...)
```

过滤掉 `-1`：

```python
valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
    output_token_ids_np < vocab_size
)
```

位置：`rejection_sampler.py:248` 到 `rejection_sampler.py:283`

---

## 22. cu_num_draft_tokens 为什么是 inclusive cumsum

`SpecDecodeMetadata.cu_num_draft_tokens` 的形状是 `[batch_size]`，不是 `[batch_size + 1]`。

例如：

```text
num_draft_tokens = [3, 0, 2, 0, 1]
cu_num_draft_tokens = [3, 3, 5, 5, 6]
```

它表示：

```text
第 i 个请求结束时，flatten draft token 总数是多少。
```

在 Triton kernel 中，如果需要当前请求的 draft 数，会用：

```text
if req_idx == 0:
  num_draft_tokens = cu_draft_curr
else:
  num_draft_tokens = cu_draft_curr - cu_draft_prev
```

这个逻辑可见于 `eagle_prepare_inputs_padded_kernel`：

```python
cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)
...
cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
num_draft_tokens = cu_draft_curr - cu_draft_prev
```

位置：`utils.py:155` 到 `utils.py:164`

---

## 23. cu_num_sampled_tokens 的作用

`cu_num_sampled_tokens` 是：

```text
cumsum(num_draft_tokens + 1)
```

它主要描述 logits / sampled token 的分段边界。

在 `RejectionSampler._get_logprobs_tensors()` 中会构造：

```python
cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]
```

位置：`rejection_sampler.py:208` 到 `rejection_sampler.py:210`

这相当于把 inclusive end 转为每个请求的 start offset。

后续用于把 `sampled_token_ids` 中的 token 映射回 `final_logits` 行：

```python
accepted_logit_indices = (
    logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
).flatten()
```

位置：`rejection_sampler.py:221` 到 `rejection_sampler.py:230`

---

## 24. target_logits_indices 和 draft_token_ids 的一一对应

`RejectionSampler` 的核心假设是：

```text
metadata.draft_token_ids.shape[0]
  == metadata.target_logits_indices.shape[0]
  == target_logits.shape[0]
```

`rejection_sample()` 中有 assert：

```python
num_tokens = draft_token_ids.shape[0]
vocab_size = target_logits.shape[-1]
assert target_logits.shape == (num_tokens, vocab_size)
```

位置：`rejection_sampler.py:418` 到 `rejection_sampler.py:425`

也就是说：

```text
每一个 draft token 都必须有且只有一个 target logits row 来验证。
```

这就是 `SpecDecodeMetadata` 最核心的不变量。

---

## 25. apply_logits_processors 如何依赖 metadata

在 spec decode 中，logits processors / penalties 不能简单按 request 级别处理，因为 target logits 已经展开成 draft-token 级别。

`RejectionSampler.apply_logits_processors()` 会根据 `metadata.num_draft_tokens` 做 repeat：

```python
num_requests = len(metadata.num_draft_tokens)
num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
original_indices = torch.arange(num_requests, device="cpu")
repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
```

位置：`rejection_sampler.py:305` 到 `rejection_sampler.py:319`

用途：

```text
把 request 级 sampling params 扩展到 draft-token 级 target logits rows。
```

例如：

```text
num_draft_tokens = [3,0,2]
repeat_indices = [0,0,0, 2,2]
```

这让 penalties / allowed_token_ids 能正确应用到每个 draft verification row。

---

## 26. bad_words / penalties 为什么还要合并 spec tokens

如果存在 penalties、bad words 或 thinking budget，`apply_logits_processors()` 会调用：

```python
output_token_ids = self._combine_outputs_with_spec_tokens(
    output_token_ids,
    sampling_metadata.spec_token_ids,
)
```

位置：`rejection_sampler.py:291` 到 `rejection_sampler.py:303`

`_combine_outputs_with_spec_tokens()` 逻辑：

```python
for out, spec in zip(output_token_ids, spec_token_ids):
    if len(spec) == 0:
        continue
    result.append(out)
    for i in range(len(spec) - 1):
        result.append([*result[-1], spec[i]])
```

位置：`rejection_sampler.py:376` 到 `rejection_sampler.py:391`

含义：

```text
验证第 1 个 draft token 时，上下文是原 output；
验证第 2 个 draft token 时，上下文应包含第 1 个 draft；
验证第 3 个 draft token 时，上下文应包含前 2 个 draft。
```

所以 penalties / bad words 的上下文也要随 draft verification 位置递增。

---

## 27. metadata 和 draft_probs 的关系

某些 drafter 会返回 draft token 的概率分布，某些不会。

`_sample()` 中：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
```

位置：`gpu_model_runner.py:3592`

`_get_spec_decode_draft_probs()` 会按 `spec_decode_metadata.num_draft_tokens` 拼接每个请求对应的 draft probs：

```python
for req_id, num_draft in zip(
    self.input_batch.req_ids, spec_decode_metadata.num_draft_tokens
):
    if num_draft == 0:
        continue
    ...
    draft_probs_rows.append(self._draft_probs[row_idx, :num_draft])
```

位置：`gpu_model_runner.py:4823` 到 `gpu_model_runner.py:4850`

如果没有 draft probs，例如 ngram spec decode，就可能传 `None`。

`rejection_sample()` 允许：

```python
draft_probs: torch.Tensor | None
```

位置：`rejection_sampler.py:394` 到 `rejection_sampler.py:407`

---

## 28. metadata 和 drafter 下一轮输入的关系

`SpecDecodeMetadata` 不只被 `RejectionSampler` 使用，也会被下一轮 draft proposer 使用。

在 `sample_tokens()` 中，proposer 调用会传入：

```python
self.propose_draft_token_ids(
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
```

位置：`gpu_model_runner.py:4481` 到 `gpu_model_runner.py:4494`

例如 Medusa 分支会根据 `spec_decode_metadata.num_draft_tokens` 从 `sample_hidden_states` 中取每个请求最后有效 token 的 hidden state：

```python
for num_draft, tokens in zip(
    spec_decode_metadata.num_draft_tokens, sampled_token_ids
):
    indices.append(offset + len(tokens) - 1)
    offset += num_draft + 1
```

位置：`gpu_model_runner.py:4947` 到 `gpu_model_runner.py:4956`

含义：

```text
metadata 描述了本轮 sample_hidden_states 的分段方式，
drafter 可以据此找到每个请求用于下一轮 proposal 的 hidden state。
```

---

## 29. metadata 和 attention metadata 的关系

`SpecDecodeMetadata` 本身不是 attention metadata。

但 `_prepare_inputs()` 发现 spec decode 后，还会设置：

```python
self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
self.num_decode_draft_tokens.copy_to_gpu()
```

位置：`gpu_model_runner.py:2188` 到 `gpu_model_runner.py:2191`

后续 `_build_attention_metadata()` 在某些 backend 下会传入 spec decode 额外信息：

```python
if use_spec_decode and isinstance(
    builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
):
    extra_attn_metadata_args = dict(
        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
    )
```

位置：`gpu_model_runner.py:2399` 到 `gpu_model_runner.py:2408`

也就是说：

```text
SpecDecodeMetadata：
  给 sampler / logits 布局使用。

num_decode_draft_tokens / num_accepted_tokens：
  给部分 attention / hybrid state backend 使用。
```

两者相关，但不是同一个对象。

---

## 30. spec_decode_common_attn_metadata 是什么

`_build_attention_metadata()` 返回：

```python
tuple[attn_metadata, spec_decode_common_attn_metadata]
```

位置：`gpu_model_runner.py:2222` 到 `gpu_model_runner.py:2226`

它会为 drafter 选择一个合适的 `CommonAttentionMetadata`：

```python
if self.speculative_config and spec_decode_common_attn_metadata is None:
    if isinstance(self.drafter, (...)):
        if self.drafter.kv_cache_gid == kv_cache_gid:
            spec_decode_common_attn_metadata = cm
    else:
        spec_decode_common_attn_metadata = cm
```

位置：`gpu_model_runner.py:2467` 到 `gpu_model_runner.py:2480`

如果当前 forward 使用了 padding，还会给 drafter unpad：

```python
spec_decode_common_attn_metadata = (
    spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
)
```

位置：`gpu_model_runner.py:2499` 到 `gpu_model_runner.py:2507`

注意：

```text
spec_decode_common_attn_metadata
  是 drafter 复用 attention / block table / slot mapping 的公共 metadata。

SpecDecodeMetadata
  是 sampler 用来解释 logits rows 的 metadata。
```

---

## 31. make_dummy() 用在哪里

`SpecDecodeMetadata.make_dummy()` 定义在：`metadata.py:29`

它根据传入的 `draft_token_ids: list[list[int]]` 构造一个占位 metadata：

```python
num_draft_tokens = [len(ids) for ids in draft_token_ids]
num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
flattened_draft_token_ids = sum(draft_token_ids, [])
```

位置：`metadata.py:29` 到 `metadata.py:66`

它会创建：

```text
真实 draft_token_ids / num_draft_tokens / cumsum
但 target_logits_indices / bonus_logits_indices / logits_indices 都是 zeros
```

这个方法适合测试或 dummy 场景，不代表真实 `_calc_spec_decode_metadata()` 的行号布局。

---

## 32. 和普通 logits_indices 的区别

普通 decode：

```python
logits_indices = query_start_loc[1:] - 1
```

位置：`gpu_model_runner.py:2153` 到 `gpu_model_runner.py:2161`

含义：

```text
每个请求只取本轮最后一个 query token 的 hidden state。
```

spec decode：

```python
logits_indices = spec_decode_metadata.logits_indices
```

位置：`gpu_model_runner.py:2183` 到 `gpu_model_runner.py:2187`

含义：

```text
每个请求取最后 draft_len + 1 个 hidden states：
  draft_len 个用于 target verification；
  1 个用于 bonus sampling。
```

所以 spec decode 下：

```text
logits rows 不再是 “batch_size 行”；
而是 “total_draft_tokens + batch_size 行”。
```

---

## 33. 为什么 target_logits_indices 不是 full hidden_states 的行号

这是一个容易混淆点。

`target_logits_indices` 和 `bonus_logits_indices` 都是相对于：

```text
logits = compute_logits(hidden_states[logits_indices])
```

的行号。

它们不是相对于：

```text
full hidden_states
```

的行号。

对 full hidden_states 的行号只有：

```text
logits_indices
```

因此关系是：

```text
full hidden states row
  --logits_indices-->
compute_logits input row
  --target_logits_indices / bonus_logits_indices-->
RejectionSampler 使用的 logits row
```

---

## 34. 为什么 draft_token_ids 用 target_logits_indices + 1

对于语言模型，位置 `p` 的 logits 预测的是位置 `p + 1` 的 token。

所以验证 draft token 时：

```text
target logits row for position p
  对应 draft token at position p + 1
```

代码：

```python
draft_token_ids = self.input_ids.gpu[logits_indices]
draft_token_ids = draft_token_ids[target_logits_indices + 1]
```

位置：`gpu_model_runner.py:2807` 到 `gpu_model_runner.py:2810`

这也是 spec decode 输入要包含 `draft_len + 1` 个 sampled rows 的根本原因：

```text
最后一个 row 不验证 draft，专门作为 bonus row。
```

---

## 35. 为什么请求没有 draft 也能走 spec metadata

当 batch 中至少一个请求有 draft tokens，`use_spec_decode=True`。

这时所有请求都会进入 `SpecDecodeMetadata` 的 batch 维度。

对没有 draft 的请求：

```text
num_draft_tokens[i] = 0
cu_num_draft_tokens 不增加
target_logits_indices 中没有它的 row
bonus_logits_indices 中有 1 个 row
```

这让 mixed batch 可以统一走 `RejectionSampler`。

无 draft 请求最后等价于：

```text
普通采样 1 个 token。
```

---

## 36. chunked prefill 与 num_decode_draft_tokens

`_prepare_inputs()` 里除了 `num_draft_tokens`，还维护了：

```python
num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
```

位置：`gpu_model_runner.py:2167` 到 `gpu_model_runner.py:2170`

只有当请求已经进入 decode 阶段时才写入 draft_len：

```python
if (
    self.input_batch.num_computed_tokens_cpu[req_idx]
    >= self.input_batch.num_prompt_tokens[req_idx]
):
    num_decode_draft_tokens[req_idx] = draft_len
```

位置：`gpu_model_runner.py:2178` 到 `gpu_model_runner.py:2182`

注释说明：

```text
For chunked prefills, use -1 as mask rather than 0,
as guided decoding may rollback speculative tokens.
```

位置：`gpu_model_runner.py:2167` 到 `gpu_model_runner.py:2170`

也就是说：

```text
num_draft_tokens：
  sampler metadata，描述本轮 logits 布局。

num_decode_draft_tokens：
  attention / hybrid backend metadata，区分 decode spec rows 和 prefill / masked rows。
```

---

## 37. logprobs 如何依赖 metadata

如果请求需要 logprobs，`RejectionSampler._get_logprobs_tensors()` 会把 target logits 和 bonus logits 重新拼成 `final_logits`：

```python
final_logits = torch.zeros_like(logits, dtype=torch.float32)
final_logits[target_logits_indices] = target_logits.to(torch.float32)
final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)
```

位置：`rejection_sampler.py:211` 到 `rejection_sampler.py:216`

然后根据 `cu_num_sampled_tokens` 和输出 token 矩阵收集 accepted token 的 logprobs。

注意注释：

```text
为了避免 CPU-GPU 同步，会先为所有 draft tokens 计算 indices，
包括 rejected tokens；rejected tokens 后续在 parse_output 中过滤。
```

位置：`rejection_sampler.py:218` 到 `rejection_sampler.py:220`

---

## 38. metadata 的关键不变量

`SpecDecodeMetadata` 必须满足这些关系：

```text
len(num_draft_tokens) == batch_size
cu_num_draft_tokens[-1] == len(draft_token_ids)
len(target_logits_indices) == len(draft_token_ids)
len(bonus_logits_indices) == batch_size
len(logits_indices) == len(draft_token_ids) + batch_size
max_spec_len == max(num_draft_tokens)
```

并且：

```text
logits.shape[0] == len(logits_indices)
target_logits.shape[0] == len(draft_token_ids)
bonus_logits.shape[0] == batch_size
```

这些不变量一旦错位，后果通常是：

```text
验证错 token；
bonus token 取错 row；
logprobs 对不上输出；
rejection sampler 接受 / 拒绝结果错误。
```

---

## 39. 容易混淆的点

### 39.1 `logits_indices` 是 target logits indices 吗？

不是。

```text
logits_indices：
  从 full hidden_states 取 rows。

target_logits_indices：
  从 compute_logits 后的 logits 取 rows。
```

### 39.2 `bonus_logits_indices` 是 bonus token id 吗？

不是。

它是 logits tensor 的 row index。

真正的 bonus token id 是 `Sampler` 从 `bonus_logits` 采样得到的。

### 39.3 `draft_token_ids` 是 SchedulerOutput 原样拷贝吗？

不是简单原样拷贝。

它是从本轮 `input_ids.gpu[logits_indices]` 中按 `target_logits_indices + 1` 取出来的 flatten tensor。

### 39.4 `num_draft_tokens=0` 的请求会消失吗？

不会。

它没有 target verification rows，但仍有一个 bonus / normal sampling row。

### 39.5 `SpecDecodeMetadata` 会修改 request 状态吗？

不会。

它只描述本轮 logits 布局。真正修改 request 状态的是 Scheduler 的 `update_from_output()`。

### 39.6 `SpecDecodeMetadata` 是 attention metadata 吗？

不是。

它主要给 sampler / logits / drafter 使用。attention backend 使用的是 `CommonAttentionMetadata` 和 per-layer metadata，只是在部分 backend 中会额外用到 draft 数和 accepted 数。

---

## 40. 总结

`SpecDecodeMetadata` 的核心职责是把：

```text
每个请求的 scheduled draft token 数
```

转换成：

```text
flatten logits row layout
```

完整链路可以记为：

```text
SchedulerOutput.scheduled_spec_decode_tokens
  → _prepare_inputs()
  → num_draft_tokens
  → _calc_spec_decode_metadata()
      → logits_indices
      → target_logits_indices
      → bonus_logits_indices
      → draft_token_ids
      → cu_num_draft_tokens / cu_num_sampled_tokens
  → hidden_states[logits_indices]
  → compute_logits()
  → RejectionSampler
      → target logits 验证 draft
      → bonus logits 采样 bonus
      → 输出 accepted / recovered / bonus tokens
```

如果只记一句话：

```text
SpecDecodeMetadata 把“每个请求几个 draft token”翻译成“哪些 hidden rows 算 logits、哪些 logits rows 验证 draft、哪些 logits rows 采 bonus”。
```
