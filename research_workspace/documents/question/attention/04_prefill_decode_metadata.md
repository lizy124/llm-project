# 04. Prefill / Decode metadata 有什么差异？

源码位置：

- `code/vllm/vllm/v1\core\sched\output.py`
- `code/vllm/vllm/v1\core\sched\scheduler.py`
- `code/vllm/vllm/v1\worker\gpu_model_runner.py`
- `code/vllm/vllm/v1\attention\backend.py`
- `code/vllm/vllm/v1\attention\backends\utils.py`
- `code/vllm/vllm/v1\attention\backends\flashinfer.py`
- `code/vllm/vllm/v1\attention\backends\gdn_attn.py`
- `code/vllm/vllm/v1\attention\backends\mamba_attn.py`
- `code/vllm/vllm/model_executor\layers\attention\mla_attention.py`

本文用于梳理 prefill、decode、chunked prefill、mixed batch、spec decode 在 attention metadata 中的差异：Scheduler 如何表达“本轮每个请求跑多少 token”，ModelRunner 如何把它翻译成 `query_start_loc / seq_lens / max_query_len / max_seq_len / slot_mapping / is_prefilling`，以及具体 attention backend 如何根据这些字段选择 prefill kernel、decode kernel 或 mixed path。

---

## 1. 本文要回答的问题

```text
prefill 和 decode 在 attention metadata 中分别需要哪些字段？
query_start_loc / seq_lens / max_query_len / max_seq_len 如何计算？
chunked prefill 如何表示？
mixed prefill + decode batch 如何处理？
spec decode tokens 如何影响 attention metadata？
哪些 metadata 字段决定 backend 走 prefill kernel 还是 decode kernel？
为什么有些 backend 要 reorder batch？
CommonAttentionMetadata 和 backend-specific metadata 如何分工？
```

---

## 2. 一句话回答

prefill / decode 的本质差异不是一个单独 flag，而是由一组 metadata 共同表达：

```text
query_start_loc 说明本轮每个 request 的 query token 边界；
seq_lens 说明每个 request 当前可见的总 KV 长度；
max_query_len / max_seq_len 说明 batch 最大 query / context 形态；
slot_mapping 说明本轮 token 写入 KV cache 的位置；
is_prefilling 说明 request 是否仍在 prompt / prefill 阶段；
block_table 说明每个 request 读哪些历史 KV blocks。
```

最小链路是：

```text
SchedulerOutput.num_scheduled_tokens
  → GPUModelRunner._prepare_inputs()
      → query_start_loc / positions / seq_lens / logits_indices / spec_decode_metadata
  → GPUModelRunner._get_slot_mappings()
      → slot_mapping
  → GPUModelRunner._build_attention_metadata()
      → CommonAttentionMetadata
  → AttentionMetadataBuilder.build(...)
      → backend-specific AttentionMetadata
  → set_forward_context(...)
  → Attention.forward()
```

如果只记住一句话：

```text
prefill / decode 差异在 Scheduler 侧表现为“本轮每个请求调度几个 token”，在 ModelRunner 侧表现为 query 边界、总序列长度和 KV 写入位置，在 backend 侧表现为是否拆成 prefill / decode kernel 路径。
```

---

## 3. 先看上游：SchedulerOutput 怎么表达本轮执行形态

`SchedulerOutput` 是 attention metadata 的上游源头之一。

位置：`code/vllm/vllm/v1/core/sched/output.py:183`

关键字段：

```text
num_scheduled_tokens: dict[str, int]
  req_id -> 本轮为该请求调度多少 token。

total_num_scheduled_tokens: int
  所有请求本轮 token 数之和。

scheduled_spec_decode_tokens: dict[str, list[int]]
  req_id -> 本轮要验证的 speculative draft token ids。

num_common_prefix_blocks: list[int]
  每个 KV cache group 中所有请求共享的 prefix block 数，用于 cascade attention。
```

源码里写得很直接：

```text
num_scheduled_tokens：Number of tokens scheduled for each request
total_num_scheduled_tokens：Equal to sum(num_scheduled_tokens.values())
scheduled_spec_decode_tokens：如果请求没有 spec decode tokens，则不在 dict 中
```

位置：`code/vllm/vllm/v1/core/sched/output.py:193` 到 `code/vllm/vllm/v1/core/sched/output.py:202`

### 3.1 prefill / decode 在 SchedulerOutput 中没有单独枚举

Scheduler 不会给每个请求显式写：

```text
phase = PREFILL / DECODE
```

它主要通过这几个值间接表达：

```text
num_scheduled_tokens[req_id]
request.num_computed_tokens
request.num_tokens
request.num_prompt_tokens
scheduled_spec_decode_tokens
```

典型情况：

```text
纯 decode：
  num_scheduled_tokens[req] = 1
  request 已经完成 prompt prefill。

纯 prefill：
  num_scheduled_tokens[req] > 1
  request.num_computed_tokens 通常还在 prompt 范围内。

chunked prefill：
  num_scheduled_tokens[req] 可能是 1，也可能 > 1；
  request.num_computed_tokens > 0，但仍小于 prompt 长度。

spec decode：
  num_scheduled_tokens[req] 通常包含 target token + draft tokens；
  scheduled_spec_decode_tokens[req] 记录 draft token ids。
```

所以 attention metadata 里的 prefill / decode 判断，不能只看 `num_scheduled_tokens == 1`。

---

## 4. spec decode tokens 从哪里来

Scheduler 在调度 running request 时会处理 speculative decode。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:630`

核心公式：

```text
num_scheduled_spec_tokens
  = num_new_tokens
    + request.num_computed_tokens
    - request.num_tokens
    - request.num_output_placeholders
```

如果这个值大于 0，就把对应数量的 `request.spec_token_ids` 放入：

```text
scheduled_spec_decode_tokens[request_id]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:630` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:642`

这意味着：

```text
num_scheduled_tokens 表达本轮总共跑几个 token；
scheduled_spec_decode_tokens 表达其中哪些 token 是 draft token。
```

两者会在 ModelRunner 里共同影响：

```text
query length
logits_indices
SpecDecodeMetadata
num_decode_draft_tokens
Mamba / GDN 这类 stateful backend 的 metadata
```

---

## 5. _prepare_inputs() 如何生成 query_start_loc / positions / seq_lens

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1930`

`_prepare_inputs()` 接收：

```text
scheduler_output
num_scheduled_tokens: np.ndarray
```

它做的核心事情是：

```text
把每个 request 本轮调度多少 token，展开成 token 级输入和 request 边界。
```

### 5.1 req_indices：把 request 维度展开成 token 维度

如果本轮每个请求调度 token 数是：

```text
[2, 5, 3]
```

那么：

```text
req_indices = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1933` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1935`

它表示：

```text
展开后的第 i 个 token 属于哪个 request。
```

### 5.2 query_pos：本轮 token 在 query 内的局部位置

仍以 `[2, 5, 3]` 为例：

```text
query_pos = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1937` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1941`

它表示：

```text
当前 token 是本 request 本轮 query 的第几个 token。
```

### 5.3 positions：当前 token 在整条序列中的绝对位置

核心公式：

```text
positions = num_computed_tokens[req_indices] + query_pos
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2158` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2164`

含义是：

```text
绝对 token position = 这个请求之前已经算过的 token 数 + 本轮内偏移。
```

这个字段后面会用于：

```text
1. 从 InputBatch.token_ids_cpu_tensor 取 input_ids；
2. 计算 RoPE / M-RoPE / XD-RoPE positions；
3. 计算 slot_mapping；
4. 传入 CommonAttentionMetadata，供部分 backend 构造 mask / index / sparse metadata。
```

### 5.4 query_start_loc：packed query 的 request 边界

`query_start_loc` 是 `num_scheduled_tokens` 的前缀和。

如果本轮是：

```text
num_scheduled_tokens = [2, 5, 3]
```

那么：

```text
query_start_loc = [0, 2, 7, 10]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2043` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2049`

它表示：

```text
request 0 的 query token 在 packed query[0:2]
request 1 的 query token 在 packed query[2:7]
request 2 的 query token 在 packed query[7:10]
```

如果 CUDA graph / padding 需要更多行，后面会把 `query_start_loc` padded 成非递减数组，避免 FlashAttention 这类 kernel 不接受非法边界。

### 5.5 seq_lens：每个 request 当前可见的总序列长度

核心公式：

```text
seq_lens = num_computed_tokens + num_scheduled_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2162` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2165`

含义是：

```text
当前 attention 能看到的 KV 长度 = 历史已经计算的 KV + 本轮新写入的 KV。
```

这点对 prefill / decode 都成立：

```text
prefill：seq_lens 是 prompt 已计算部分 + 当前 prompt chunk；
decode：seq_lens 是历史上下文 + 当前 decode token；
spec decode：seq_lens 通常按 optimistic 方式假设 draft token 被接受。
```

### 5.6 optimistic_seq_lens_cpu：为什么叫 optimistic

在 `_prepare_inputs()` 中还会先计算：

```text
optimistic_seq_lens_cpu = num_computed_tokens + num_scheduled_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2055` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2060`

它叫 optimistic，是因为 async spec decode 中 draft token 未必都会被接受，但 metadata 构造阶段会先按“都接受”计算上界。

这个值后面用于：

```text
max_seq_len
seq_lens_cpu_upper_bound
discard_request_mask
部分 backend 的 CPU side planning
```

---

## 6. max_query_len / max_seq_len 如何计算

### 6.1 max_query_len

在普通执行路径里，`_build_attention_metadata()` 的 `max_query_len` 参数来自：

```text
max_num_scheduled_tokens
```

调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4308` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4321`

也就是说：

```text
max_query_len = max(num_scheduled_tokens_per_request)
```

典型含义：

```text
max_query_len = 1
  通常是 decode-only batch。

max_query_len > 1
  可能是纯 prefill、chunked prefill、mixed batch，也可能是 spec decode。
```

注意：

```text
max_query_len > 1 不等于一定是 prompt prefill。
```

因为 spec decode 在 decode 阶段也可能一次验证多个 draft tokens。

### 6.2 max_seq_len

`_build_attention_metadata()` 中普通路径计算：

```text
max_seq_len = max(optimistic_seq_lens_cpu[:num_reqs])
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2285` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2293`

如果是 CUDA graph capture，则会保守地使用：

```text
max_seq_len = self.max_model_len
```

原因是 capture 阶段需要更稳定的 shape 和 kernel 选择。

---

## 7. slot_mapping 和 block_table 在 prefill / decode 中的作用

`query_start_loc / seq_lens` 描述的是“看哪些 token”；`slot_mapping / block_table` 描述的是“KV cache 在哪里”。

### 7.1 block_table

`block_table` 是 request 到 KV cache block 的映射。

它回答：

```text
这个 request 当前持有哪些 KV blocks？
```

`_build_attention_metadata()` 会按 KV cache group 取出 block table：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2295` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2311`

不同 KV cache group 可能有不同 block table，所以后面会按 group 替换。

### 7.2 slot_mapping

`slot_mapping` 是 token 到实际 KV cache slot 的映射。

它回答：

```text
本轮第 i 个 query token 的 K/V 要写入哪个 slot？
```

`_prepare_inputs()` 中会调用：

```text
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    query_start_loc,
    positions,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2167` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2171`

### 7.3 prefill / decode 中二者的差异

```text
prefill：
  slot_mapping 覆盖一段 prompt tokens，通常一次写多个 KV slots。
  block_table 需要覆盖当前 prompt chunk 会用到的 blocks。

decode：
  slot_mapping 通常每个 request 一个新 slot。
  block_table 主要用于读取已有历史 KV blocks，同时写入当前新 token。

chunked prefill：
  slot_mapping 只覆盖当前 chunk 的 tokens；
  block_table 覆盖已经分配给 prompt 的 blocks；
  seq_lens 表达历史 prompt chunk + 当前 chunk。
```

---

## 8. CommonAttentionMetadata 如何承载这些字段

定义位置：`code/vllm/vllm/v1/attention/backend.py:395`

`CommonAttentionMetadata` 是所有 backend 的公共输入，不是最终 kernel 参数。

核心字段包括：

```text
query_start_loc
query_start_loc_cpu
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
causal
positions
is_prefilling
seq_lens_cpu_upper_bound
encoder_seq_lens
encoder_seq_lens_cpu
dcp_local_seq_lens
mm_req_doc_ranges
rswa_prefix_lens
```

构造位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2394` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2412`

### 8.1 query 相关字段

```text
query_start_loc
query_start_loc_cpu
max_query_len
num_actual_tokens
```

它们描述 packed query tensor 的布局。

其中：

```text
num_actual_tokens = metadata 当前携带的 token 数；普通路径等于本轮真实 token 总数，FULL CUDA graph / padding 路径可能包含 padded token
```

`num_actual_tokens` 名称有历史包袱；普通路径等于真实 token 数，FULL CUDA graph / padding 路径可能等于 padded token 数。backend 应结合 `query_start_loc`、`num_reqs`、padding slot `-1`、block table padding 行等，避免把 padding 当真实 token。

位置：`code/vllm/vllm/v1/attention/backend.py:412` 到 `code/vllm/vllm/v1/attention/backend.py:418`

### 8.2 sequence 相关字段

```text
seq_lens
seq_lens_cpu_upper_bound
max_seq_len
```

`seq_lens` 表示每个 request 当前序列长度。

`CommonAttentionMetadata.compute_num_computed_tokens()` 可以反推历史长度：

```text
num_computed_tokens = seq_lens - query_lens
query_lens = query_start_loc[1:] - query_start_loc[:-1]
```

位置：`code/vllm/vllm/v1/attention/backend.py:513` 到 `code/vllm/vllm/v1/attention/backend.py:517`

`seq_lens_cpu_upper_bound` 的源码注释很关键：

```text
prefill rows 上是精确值；
非 async spec decode 行上也是精确值；
async spec decode 的 decode 行上是 optimistic upper bound。
```

位置：`code/vllm/vllm/v1/attention/backend.py:447` 到 `code/vllm/vllm/v1/attention/backend.py:451`

### 8.3 is_prefilling

`is_prefilling` 在 `_build_attention_metadata()` 中计算：

```text
is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2339` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2345`

它的语义是：

```text
request 是否仍处于 prefill 阶段。
```

这个字段特别重要，因为：

```text
query_len = 1 不一定就是 decode；
也可能是 short chunked prefill。
```

Mamba / GDN 等 stateful backend 会用它区分 actual decode 和 short extend。

---

## 9. Prefill metadata 的典型形态

纯 prefill 通常意味着：

```text
某些 request 还没有完成 prompt；
本轮 query_len 可能很长；
max_query_len > 1；
seq_lens = 已计算 prompt tokens + 当前 prompt tokens；
slot_mapping 覆盖当前 prompt token 的 KV 写入位置；
is_prefilling = True。
```

例如两个请求第一次 prefill：

```text
num_computed_tokens = [0, 0]
num_scheduled_tokens = [4, 3]

query_start_loc = [0, 4, 7]
positions = [0, 1, 2, 3, 0, 1, 2]
seq_lens = [4, 3]
max_query_len = 4
max_seq_len = 4
is_prefilling = [True, True]
```

对 backend 来说，这批 token 的特点是：

```text
1. 每个请求本轮 query span 可能包含多个 token；
2. causal mask 需要覆盖 query 内部自回归关系；
3. K/V 会批量写入多个 slot；
4. 历史 KV 可能为空，也可能来自 prefix cache / chunked prefill。
```

---

## 10. Decode metadata 的典型形态

纯 decode 通常意味着：

```text
每个 request 已完成 prompt；
本轮每个 request 调度 1 个新 token；
max_query_len = 1；
query_start_loc = [0, 1, 2, 3, ...]；
seq_lens = 历史上下文长度 + 1；
slot_mapping 每个 request 对应一个新 KV slot；
is_prefilling = False。
```

例如三个请求 decode：

```text
num_computed_tokens = [10, 20, 30]
num_scheduled_tokens = [1, 1, 1]

query_start_loc = [0, 1, 2, 3]
positions = [10, 20, 30]
seq_lens = [11, 21, 31]
max_query_len = 1
max_seq_len = 31
is_prefilling = [False, False, False]
```

对 backend 来说，这批 token 的特点是：

```text
1. 每个请求只追加一个 token；
2. attention 主要是当前 query 对历史 KV 的读取；
3. 很多 backend 可以使用更高效的 decode kernel；
4. CUDA graph 通常更容易支持 decode-only uniform batch。
```

---

## 11. Chunked prefill 如何表示

chunked prefill 的关键点是：

```text
request 有历史上下文，但还没完成 prompt prefill。
```

例如某请求 prompt 长度是 100，之前已经计算 64 个 token，本轮继续算 16 个 token：

```text
num_computed_tokens = 64
num_prompt_tokens = 100
num_scheduled_tokens = 16

query_start_loc span = 16
positions = [64, 65, ..., 79]
seq_lens = 80
is_prefilling = True
```

这和 decode 的区别是：

```text
decode：
  num_computed_tokens >= num_prompt_tokens
  is_prefilling = False

chunked prefill：
  num_computed_tokens < num_prompt_tokens
  is_prefilling = True
```

即使 chunked prefill 本轮只调度 1 个 token，也不能简单当成普通 decode。

这就是为什么 `split_decodes_and_prefills()` 有参数：

```text
treat_short_extends_as_decodes
```

以及为什么 `is_prefilling` 会被放进 `CommonAttentionMetadata`。

---

## 12. Mixed batch 如何表示

mixed batch 指同一轮里同时有：

```text
decode request
chunked prefill request
pure prefill request
spec decode request
```

在公共 metadata 中，它们不会变成多个 metadata，而是先统一 packed 到一组字段里：

```text
query_start_loc = 所有 request 的 query 边界
seq_lens = 所有 request 当前总长度
positions = 所有 query token 的绝对位置
slot_mapping = 所有 query token 的 KV slot
is_prefilling = 每个 request 是否仍在 prefill
```

例如：

```text
request A：decode，query_len = 1，seq_len = 101
request B：chunked prefill，query_len = 16，seq_len = 80
request C：pure prefill，query_len = 32，seq_len = 32

num_scheduled_tokens = [1, 16, 32]
query_start_loc = [0, 1, 17, 49]
seq_lens = [101, 80, 32]
max_query_len = 32
max_seq_len = 101
is_prefilling = [False, True, True]
```

之后是否拆分，要看具体 backend。

---

## 13. 为什么有些 backend 要 reorder batch

很多高性能 attention kernel 对 batch 排列有要求或偏好。

vLLM 提供了工具：

```text
reorder_batch_to_split_decodes_and_prefills(...)
split_decodes_and_prefills(...)
```

位置：

- `code/vllm/vllm/v1/attention/backends/utils.py:564`
- `code/vllm/vllm/v1/attention/backends/utils.py:663`

### 13.1 目标 batch 顺序

`reorder_batch_to_split_decodes_and_prefills()` 希望把 batch 排成：

```text
decode
short_extend
long_extend
prefill
```

源码注释中的分类是：

```text
decode：num_scheduled <= threshold 且已经完成 prefill
short_extend：num_scheduled <= threshold 且仍在 chunked prefill
long_extend：num_scheduled > threshold 且仍在 chunked prefill
prefill：num_computed == 0
```

位置：`code/vllm/vllm/v1/attention/backends/utils.py:673` 到 `code/vllm/vllm/v1/attention/backends/utils.py:739`

### 13.2 split_decodes_and_prefills() 如何切边界

`split_decodes_and_prefills()` 假设 batch 已经按上面的顺序排列，然后根据 query length 找 prefill / decode 边界。

返回：

```text
num_decodes
num_prefills
num_decode_tokens
num_prefill_tokens
```

位置：`code/vllm/vllm/v1/attention/backends/utils.py:538` 到 `code/vllm/vllm/v1/attention/backends/utils.py:607`

核心判断包括：

```text
max_query_len <= decode_threshold
  可以整体视为 decode 区域。

query_len > decode_threshold
  进入 prefill 区域。

require_uniform=True
  decode 区域还要求 query_len 一致。

treat_short_extends_as_decodes=False
  会用 is_prefilling 把 short extend 也归入 prefill。
```

### 13.3 reorder_batch_threshold 从哪里来

`AttentionMetadataBuilder` 有：

```text
reorder_batch_threshold
```

位置：`code/vllm/vllm/v1/attention/backend.py:537` 到 `code/vllm/vllm/v1/attention/backend.py:543`

如果 backend 支持把 spec decode 当作 decode kernel 处理，阈值可能提升到：

```text
1 + num_speculative_tokens
```

或 parallel drafting 下：

```text
1 + 2 * num_speculative_tokens
```

位置：`code/vllm/vllm/v1/attention/backend.py:567` 到 `code/vllm/vllm/v1/attention/backend.py:597`

这说明：

```text
backend 可以通过 builder 反向影响 ModelRunner 的 batch 排列策略。
```

---

## 14. FlashAttention 风格：公共 metadata 直接贴近 kernel 参数

FlashAttention 类 backend 通常更接近：

```text
CommonAttentionMetadata
  → FlashAttentionMetadata
  → varlen / paged KV attention kernel
```

典型字段：

```text
query_start_loc
seq_lens
block_table
slot_mapping
max_query_len
max_seq_len
causal
```

它不一定在 metadata 结构中显式持有：

```text
prefill = ...
decode = ...
```

而是依赖 varlen query、paged KV、query_start_loc、seq_lens 等字段统一表达。

心智模型：

```text
FlashAttention 更偏“用一套 varlen/paged 参数覆盖 prefill、decode、mixed batch”。
```

具体字段和 cascade / DCP 等细节已在 `03_attention_metadata_builder.md` 中展开。

---

## 15. FlashInfer 风格：显式拆 prefill / decode

FlashInfer builder 是最典型的显式拆分案例。

入口：`code/vllm/vllm/v1/attention/backends/flashinfer.py:1074` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:1101`

但这里不是无条件先 split，而是先看：

```text
causal = common_attn_metadata.causal
```

只有 `causal=True` 时才会调用：

```text
split_decodes_and_prefills(
    common_attn_metadata,
    decode_threshold=self.reorder_batch_threshold,
    require_uniform=True,
)
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:1079` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:1087`

而 `causal=False` 时不会走 decode/TRTLLM 拆分，而是直接把整批都当作 prefill：

```text
num_decodes = 0
num_prefills = num_reqs
num_decode_tokens = 0
num_prefill_tokens = num_actual_tokens
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:1088` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:1094`

得到：

```text
num_decodes
num_prefills
num_decode_tokens
num_prefill_tokens
```

然后构造：

```text
FlashInferMetadata(
  num_decodes=...,
  num_decode_tokens=...,
  num_prefills=...,
  num_prefill_tokens=...,
  prefill=None,
  decode=None,
  cascade_wrapper=None,
)
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:991` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:1006`

### 15.1 prefill path

如果 `num_prefills > 0`，FlashInfer 会处理 prefill 区域。

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:1115` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:1219`

关键切片：

```text
prefill_start = num_decodes
qo_indptr_prefill = query_start_loc[prefill_start:] - query_start_loc[prefill_start]
seq_lens_prefill = seq_lens[prefill_start:]
block_table_prefill = block_table[prefill_start:]
```

如果走 TRT-LLM prefill，会构造：

```text
TRTLLMPrefill(
  block_tables,
  seq_lens,
  cum_seq_lens_q,
  cum_seq_lens_kv,
  max_q_len,
  max_seq_len,
)
```

如果走 FlashInfer native prefill，会创建 prefill wrapper 并 plan：

```text
BatchPrefillWithPagedKVCacheWrapper.plan(...)
```

然后：

```text
attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)
```

### 15.2 decode path

如果 `num_decodes > 0`，FlashInfer 会处理 decode 区域。

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:1221` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:1277`

如果走 TRT-LLM decode：

```text
TRTLLMDecode(
  block_tables=block_table[:num_decodes],
  seq_lens=seq_lens[:num_decodes],
  max_seq_len=max_seq_len,
)
```

如果走 FlashInfer native decode，则 plan decode wrapper：

```text
fast_plan_decode(...)
attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
```

### 15.3 FlashInfer 的心智模型

FlashInfer metadata 可以理解成：

```text
CommonAttentionMetadata
  → 先按 query_len / threshold 拆成 decode 区域和 prefill 区域
  → decode 区域使用 decode wrapper / TRTLLMDecode
  → prefill 区域使用 prefill wrapper / TRTLLMPrefill
  → mixed batch 在一个 FlashInferMetadata 中同时携带 prefill 和 decode 子 metadata
```

例外是 cascade attention 的保留代码路径：若 `common_prefix_len > 0`，FlashInfer metadata 会设置 `use_cascade=True` 并使用 `cascade_wrapper`，此时 prefill / decode 子 metadata 不再按普通路径填充。但当前 `FlashInferMetadataBuilder.use_cascade_attention()` 返回 `False`，所以常规调度不会自动启用 FlashInfer cascade。

---

## 16. MLA 风格：prefill / decode 是两套计算路径

MLA attention 的差异更大，因为它不只是普通 attention kernel 的 metadata 变体。

MLA 通常会构造类似：

```text
MLACommonMetadata(
  prefill=MLACommonPrefillMetadata | None,
  decode=MLACommonDecodeMetadata | None,
)
```

prefill 和 decode 对应不同计算方式：

```text
prefill：更接近 MHA-style，对当前 query chunk 做注意力计算；
decode：更接近 MQA-style，读取 compressed KV cache。
```

这类 backend 需要显式知道：

```text
哪些 token 属于 prefill；
哪些 token 属于 decode；
chunked prefill 的上下文如何切分；
DCP / local context 如何构造；
spec decode 是否能当 decode path 处理。
```

所以对 MLA 来说：

```text
max_query_len / query_start_loc / seq_lens 不只是 kernel 参数，还是拆分 prefill/decode 子路径的依据。
```

---

## 17. Mamba / GDN 风格：query_len=1 也不一定是 decode

Mamba / GDN 不是传统 attention，但复用了 attention metadata builder 管线。

它们特别依赖：

```text
is_prefilling
num_accepted_tokens
num_decode_draft_tokens_cpu
prev_last_scheduled_idx
state indices
```

`_build_attention_metadata()` 在 spec decode 且 builder 是 Mamba / GDN 时，会额外传：

```text
num_accepted_tokens
num_decode_draft_tokens_cpu
prev_last_scheduled_idx
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2398` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2416`

这里要分清两层语义：

1. GDN 的普通拆分路径确实会调用：

```text
split_decodes_and_prefills(m, decode_threshold=1)
```

位置：`code/vllm/vllm/v1/attention/backends/gdn_attn.py:210` 到 `code/vllm/vllm/v1/attention/backends/gdn_attn.py:213`

2. 但“单 token prefill 也可能被改判成 decode/update row”这个特例，当前源码里是 Mamba 的 FULL-CG 逻辑，不是通用 GDN 规则。

Mamba 会先检查：

```text
single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)
has_prior_state = seq_lens_cpu > 1
prefill_to_decode = single_token_prefill_rows & has_prior_state
```

若命中，就把这些行的 `is_prefilling` 从 `True` 改成 `False`，再执行：

```text
split_decodes_and_prefills(
    common_attn_metadata,
    decode_threshold=decode_threshold,
    treat_short_extends_as_decodes=False,
)
```

位置：`code/vllm/vllm/v1/attention/backends/mamba_attn.py:388` 到 `code/vllm/vllm/v1/attention/backends/mamba_attn.py:415`

因此它不能只看 `query_len`，因为 stateful layer 还需要区分：

```text
真正 decode：更新已有 recurrent / state cache；
short chunked prefill：仍在初始化或推进 prompt state；
spec decode：可能有 accepted / rejected draft token，需要修正 state；
Mamba FULL-CG 下的单 token prefill：虽然仍处于 prefill 语义，但可能为了图复用被当作 decode/update row。
```

所以 `is_prefilling` 的价值在这些 backend 上最明显，而上述单 token 改判应明确归因到 Mamba FULL-CG。

---

## 18. Spec decode 对 attention metadata 的影响

spec decode 会让 decode 阶段不再是“每个 request 一个 query token”。

它可能变成：

```text
target token + draft token(s)
```

因此：

```text
query_len 可能 > 1；
max_query_len 可能 > 1；
logits_indices 不再只是每个 request 最后一个 token；
某些 backend 可以把这类 batch 当作 decode；
某些 backend 必须把 spec decode 和普通 decode / prefill 分开处理。
```

### 18.1 _prepare_inputs() 中的 spec decode 分支

判断：

```text
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2153`

如果没有 spec decode：

```text
logits_indices = query_start_loc[1:] - 1
spec_decode_metadata = None
num_sampled_tokens = np.ones(num_reqs)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2154` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2162`

如果有 spec decode：

```text
num_draft_tokens[req_idx] = len(draft_token_ids)
num_decode_draft_tokens[req_idx] = draft_len  # 仅对已完成 prompt 的 decode request
spec_decode_metadata = _calc_spec_decode_metadata(...)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2163` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2191`

### 18.2 SpecDecodeMetadata 不等于 AttentionMetadata

这点容易混淆。

```text
SpecDecodeMetadata
  主要服务 sampler：draft token、bonus token、target logits、采样位置、rejection。

spec_decode_common_attn_metadata
  主要服务 speculative drafter / proposer 复用或改写 common attention layout，必要时提供 unpadded 后的 request/token 视图。

CommonAttentionMetadata / AttentionMetadata
  主要服务 attention backend：query 怎么 attend、KV 怎么读写、backend kernel 怎么切分。
```

两者会同时存在。

在 `execute_model()` 后半段，`spec_decode_metadata` 和 `spec_decode_common_attn_metadata` 都会被保存进 `ExecuteModelState`，后续 `sample_tokens()` 继续使用。

### 18.3 spec_decode_common_attn_metadata

`_build_attention_metadata()` 除了返回 per-layer `attn_metadata`，还可能返回：

```text
spec_decode_common_attn_metadata
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2451` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2480`

它用于 speculative proposer / drafter 需要复用或改写 common attention layout 的场景，而不是普通 sampler 的核心输入。

如果当前 metadata 做了 padding，还会 unpad：

```text
spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2499` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2507`

---

## 19. 哪些字段决定走 prefill kernel 还是 decode kernel

没有一个统一字段叫：

```text
kernel_type = PREFILL / DECODE
```

不同 backend 有不同判断，但常见依据是：

```text
query_lens = query_start_loc[1:] - query_start_loc[:-1]
max_query_len
seq_lens
is_prefilling
reorder_batch_threshold
scheduled_spec_decode_tokens / num_decode_draft_tokens
require_uniform
common_prefix_len
cudagraph mode
```

### 19.1 最常见判断

```text
query_len <= decode_threshold
  倾向 decode。

query_len > decode_threshold
  倾向 prefill / extend。
```

但要补充：

```text
如果仍在 prefill 阶段，query_len <= threshold 可能是 short extend；
如果 spec decode 开启，query_len > 1 也可能仍想走 decode-like kernel；
如果 require_uniform=True，query_len 不一致时部分请求会被归入 prefill 区域；
如果 DCP / CUDA graph / backend 能力不支持，可能 fallback 到更保守路径。
```

### 19.2 backend 差异

```text
FlashAttention：
  多数情况下用 varlen / paged KV 统一表达，不一定显式拆 prefill / decode 子对象。

FlashInfer：
  显式 split，metadata 中有 prefill / decode 子对象。

MLA：
  prefill / decode 对应不同计算路径，metadata 明确拆分。

Mamba / GDN：
  需要结合 is_prefilling、accepted tokens、state index、spec draft tokens 判断。

FlexAttention：
  更关注 mask / block mask 构造，prefill / decode 差异会反映在 query layout 和 mask 上。
```

---

## 20. CUDA graph / padding 对 metadata 的影响

CUDA graph 希望 shape 稳定，所以 metadata 里可能出现 padded 版本：

```text
num_tokens_padded
num_reqs_padded
query_start_loc padded rows
block_table padded rows
slot_mapping padded tokens
seq_lens padded rows
```

在 `_build_attention_metadata()` 中：

```text
num_tokens_padded = num_tokens_padded or num_tokens
num_reqs_padded = num_reqs_padded or num_reqs
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2231` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2233`

对 padding 行：

```text
block_table 使用 NULL_BLOCK_ID 填充；
is_prefilling[num_reqs:] = False；
query_start_loc 保持非递减；
当 `pad_attn` 或存在 `forward_includes_kv_cache_update=False` 的 backend 导致 slot mapping 使用 padded token 维度时，`num_tokens_unpadded:num_tokens_padded` 区间填 `-1`。
```

这对 prefill / decode 判断的影响是：

```text
backend 必须区分真实 request / token 和 padded request / token；
full CUDA graph 往往更偏好 uniform decode；
mixed prefill + decode 的 graph 支持取决于 backend 的 AttentionCGSupport。
```

`AttentionCGSupport` 中就有：

```text
UNIFORM_SINGLE_TOKEN_DECODE
UNIFORM_BATCH
ALWAYS
NEVER
```

位置：`code/vllm/vllm/v1/attention/backend.py:516` 到 `code/vllm/vllm/v1/attention/backend.py:530`

---

## 21. Cascade attention 对 prefill / decode metadata 的影响

Cascade attention 处理的是多请求共享 prefix KV 的情况。

SchedulerOutput 提供：

```text
num_common_prefix_blocks
```

`GPUModelRunner` 会计算：

```text
cascade_attn_prefix_lens
common_prefix_len
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2511` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2576`

然后传给 builder：

```text
builder.build(common_prefix_len=..., common_attn_metadata=...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2431` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2435`

它对 prefill / decode 的影响是：

```text
公共 prefix KV 可能被拆成单独 kernel 或 wrapper；
剩余 suffix / per-request KV 仍按 query_start_loc / seq_lens / block_table 处理；
common_prefix_len 需要受最小 num_computed_tokens 和 block size 限制。
```

这不是 prefill / decode 分类本身，但会改变 backend metadata 的执行计划。

---

## 22. 五个场景对照

### 22.1 纯 prefill

```text
num_computed_tokens = [0, 0]
num_scheduled_tokens = [128, 96]
query_start_loc = [0, 128, 224]
seq_lens = [128, 96]
max_query_len = 128
is_prefilling = [True, True]
slot_mapping = 224 个 prompt token 的 KV 写入位置
```

backend 行为：

```text
FlashAttention：varlen prefill / paged KV path。
FlashInfer：num_decodes=0，num_prefills=2。
MLA：prefill metadata 非空，decode metadata 可能为空。
```

### 22.2 纯 decode

```text
num_computed_tokens = [128, 256, 300]
num_scheduled_tokens = [1, 1, 1]
query_start_loc = [0, 1, 2, 3]
seq_lens = [129, 257, 301]
max_query_len = 1
is_prefilling = [False, False, False]
slot_mapping = 3 个新 token 的 KV 写入位置
```

backend 行为：

```text
FlashInfer：num_decodes=3，num_prefills=0。
CUDA graph：最容易走 decode-only graph。
Mamba / GDN：按 decode 更新 state。
```

### 22.3 chunked prefill

```text
num_computed_tokens = [64]
num_prompt_tokens = [200]
num_scheduled_tokens = [32]
query_start_loc = [0, 32]
seq_lens = [96]
max_query_len = 32
is_prefilling = [True]
```

backend 行为：

```text
它不是普通 decode，即使已有历史 KV；
它是 prompt extend / chunked prefill；
backend 可能走 prefill / extend kernel。
```

### 22.4 mixed prefill + decode

```text
request A decode：query_len=1，is_prefilling=False
request B chunked prefill：query_len=16，is_prefilling=True
request C pure prefill：query_len=64，is_prefilling=True
```

公共 metadata：

```text
query_start_loc = [0, 1, 17, 81]
seq_lens = [历史A+1, 历史B+16, 64]
max_query_len = 64
is_prefilling = [False, True, True]
```

backend 行为：

```text
FlashAttention：可能统一 varlen 处理。
FlashInfer / MLA：倾向拆 decode / prefill。
需要时 runner 会 reorder 成 decode → short_extend → long_extend → prefill。
```

### 22.5 spec decode

```text
request A decode + 4 draft tokens：query_len = 5
scheduled_spec_decode_tokens[A] = [d1, d2, d3, d4]
```

metadata 变化：

```text
max_query_len 可能从 1 变成 5；
logits_indices 来自 SpecDecodeMetadata；
num_sampled_tokens = num_draft_tokens + 1；
num_decode_draft_tokens 只对已完成 prompt 的 decode request 有效；
backend reorder threshold 可能提升，让 spec decode 仍被视为 decode-like。
```

---

## 23. 调试入口

如果要调试 prefill / decode metadata，建议按这条顺序看：

```text
1. Scheduler.schedule()
   看 num_scheduled_tokens、scheduled_spec_decode_tokens、num_common_prefix_blocks。

2. GPUModelRunner._update_states()
   看 InputBatch 中 num_computed_tokens、num_prompt_tokens、block_table 是否更新正确。

3. GPUModelRunner._prepare_inputs()
   看 req_indices、positions、query_start_loc、seq_lens、logits_indices。

4. GPUModelRunner._get_slot_mappings()
   看 slot_mapping 是否和 positions / block_table 对齐。

5. GPUModelRunner._build_attention_metadata()
   看 CommonAttentionMetadata 中 max_query_len、max_seq_len、is_prefilling、block_table、slot_mapping。

6. split_decodes_and_prefills()
   看 backend 如何把 common metadata 拆成 decode / prefill 区域。

7. 具体 backend builder.build()
   看 FlashInfer / MLA / GDN / FlashAttention 如何生成 backend-specific metadata。

8. set_forward_context() / Attention.forward()
   看 layer 是否取到了正确 metadata。
```

常看位置：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py:1889
code/vllm/vllm/v1/worker/gpu_model_runner.py:2208
code/vllm/vllm/v1/attention/backend.py:393
code/vllm/vllm/v1/attention/backends/utils.py:538
code/vllm/vllm/v1/attention/backends/flashinfer.py:912
```

---

## 24. 容易疑惑的点

### 24.1 max_query_len = 1 是否一定是 decode？

不一定。

如果 request 仍然满足：

```text
num_computed_tokens < num_prompt_tokens
```

那么即使本轮只调度 1 个 token，它仍可能是 short chunked prefill。

另外在 Mamba 的 FULL-CG 路径里，还存在更细的特例：

```text
query_len = 1
is_prefilling = True
seq_lens_cpu > 1
```

这类“已有 prior state 的单 token prefill”会先被改写 `is_prefilling`，再按 decode/update row 参与后续拆分；这属于图复用需要下的 Mamba 特例，不应推广成通用规则。

### 24.2 max_query_len > 1 是否一定是 prefill？

不一定。

spec decode 中，decode 阶段也可能一次验证多个 draft token：

```text
query_len = 1 + num_draft_tokens
```

所以 backend 会用 `reorder_batch_threshold` 和 spec decode 配置决定是否把它当作 decode-like batch。

### 24.3 seq_lens 是历史长度还是总长度？

在 attention metadata 中，`seq_lens` 表示当前可见总长度：

```text
seq_lens = 历史已计算 tokens + 本轮 query tokens
```

如果需要历史已计算长度，可以用：

```text
seq_lens - query_len
```

### 24.4 query_start_loc 和 seq_lens 为什么都需要？

它们回答的问题不同：

```text
query_start_loc：本轮 query tensor 怎么切 request；
seq_lens：每个 request 可以 attend 到多长的 KV 上下文。
```

prefill / decode 的差异必须同时看这两个维度。

### 24.5 slot_mapping 和 block_table 谁决定写 KV？

```text
block_table：request 持有哪些 KV blocks；
slot_mapping：本轮每个 token 写入哪个具体 KV slot。
```

真正 token 级写入位置看 `slot_mapping`。

### 24.6 scheduled_spec_decode_tokens 会直接进入 attention kernel 吗？

不会直接进入普通 attention kernel。

它先在 `_prepare_inputs()` 中变成：

```text
SpecDecodeMetadata
logits_indices
num_decode_draft_tokens
```

再间接影响 attention metadata 和 backend 分类。

### 24.7 CommonAttentionMetadata 是最终 kernel 参数吗？

不是。

```text
CommonAttentionMetadata 是公共语言；
AttentionMetadataBuilder 是翻译器；
FlashAttentionMetadata / FlashInferMetadata / MLACommonMetadata / GDNAttentionMetadata 等才是 backend 方言。
```

---

## 25. 最终可以记成一张表

| 场景 | query_start_loc / max_query_len | seq_lens / max_seq_len | is_prefilling | slot_mapping | backend 常见路径 |
|---|---|---|---|---|---|
| 纯 prefill | query span 长，`max_query_len > 1` | 当前 prompt chunk 后的总长度 | `True` | 多个 prompt token 写入 slots | prefill / varlen / prompt kernel |
| 纯 decode | 每请求通常 1 token，`max_query_len = 1` | 历史上下文 + 1 | `False` | 每请求一个新 slot | decode kernel / decode graph |
| chunked prefill | 当前 chunk 的 query span | 已算 prompt + 当前 chunk | `True` | 当前 chunk 写入 slots | extend / prefill path |
| mixed batch | 不同 request query_len 不同 | 每行独立总长度 | 混合 | 扁平 token 级映射 | 统一 varlen或拆 prefill/decode |
| spec decode | decode 阶段也可能 `query_len > 1` | optimistic 总长度 | 通常 `False`，但要看请求状态 | target + draft token slots | spec-as-decode 或专用分流 |
| CUDA graph padding | padded query_start_loc 非递减 | padded rows 安全填充 | padding 行置 `False` | padded slot 安全填充 | uniform decode / fixed shape path |

---

## 26. 总结

Prefill / decode metadata 可以压缩成下面这条线：

```text
SchedulerOutput.num_scheduled_tokens
  → 每个请求本轮 query_len
  → query_start_loc / max_query_len
  → positions
  → seq_lens / max_seq_len
  → slot_mapping / block_table
  → is_prefilling
  → CommonAttentionMetadata
  → backend-specific AttentionMetadata
  → prefill kernel / decode kernel / mixed path
```

最小心智模型是：

```text
query_start_loc 决定“本轮 query 怎么切”；
seq_lens 决定“每行能看多长历史 KV”；
slot_mapping 决定“本轮 K/V 写到哪里”；
is_prefilling 修正“query_len=1 但仍是 prefill”的歧义；
backend builder 决定“这些公共字段最终走哪个 kernel”。
```

如果只记住一句话：

```text
vLLM V1 不靠一个 phase enum 区分 prefill / decode，而是把调度结果翻译成 query 边界、序列长度、KV cache 映射和 prefill 状态，再由具体 attention backend 按自己的 kernel 能力解释这些 metadata。
```
