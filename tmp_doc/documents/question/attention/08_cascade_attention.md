# 08. Cascade attention 如何工作？

源码位置：

- `code/vllm/vllm/config\model.py`
- `code/vllm/vllm/config\vllm.py`
- `code/vllm/vllm/v1\core\sched\output.py`
- `code/vllm/vllm/v1\core\kv_cache_manager.py`
- `code/vllm/vllm/v1\core\single_type_kv_cache_manager.py`
- `code/vllm/vllm/v1\worker\gpu_model_runner.py`
- `code/vllm/vllm/v1\attention\backend.py`
- `code/vllm/vllm/v1\attention\backends\flash_attn.py`

本文用于梳理 cascade attention 的定位：它如何利用 batch 内公共 prefix KV blocks，如何从 SchedulerOutput 传到 ModelRunner，如何影响 attention metadata，FlashAttention 如何拆成 prefix / suffix 两次 attention kernel，以及它和 prefix cache、chunked prefill、sliding window、DCP、DBO、CUDA graph 的关系。

---

## 1. 本文要回答的问题

```text
cascade attention 是什么优化？
它和 prefix cache 是什么关系？
SchedulerOutput.num_common_prefix_blocks 表示什么？
KVCacheManager 如何计算公共 prefix blocks？
GPUModelRunner 如何把公共 blocks 转成 cascade_attn_prefix_lens？
为什么 common_prefix_len 要被 min(num_computed_tokens) 截断？
什么情况下会禁用 cascade attention？
backend builder 如何决定 use_cascade？
FlashAttentionMetadata 如何表达 prefix / suffix？
FlashAttention forward 如何执行 cascade_attention？
为什么 prefix attention 是 non-causal，suffix attention 是 causal？
CUDA graph / DBO / DCP / sliding window 对 cascade 有什么限制？
```

---

## 2. 一句话回答

Cascade attention 是 vLLM V1 中针对“同一个 batch 内多个请求共享较长 prefix KV blocks”的 attention 优化。

它不是重新计算 prefix，也不是 prefix cache 本身；它是在请求已经共享 KV cache blocks 的前提下，把 attention 拆成两段：

```text
1. shared prefix attention
   把 batch 内所有 query 拼成一个大 query，统一 attend 到公共 prefix KV。

2. per-request suffix attention
   每个请求再 attend 到自己公共 prefix 之后的私有 KV / 当前 query 部分。

3. merge
   用两段 attention 的 output + LSE 合并成等价的完整 attention 结果。
```

最小主链路是：

```text
KVCacheManager.get_num_common_prefix_blocks()
  → SchedulerOutput.num_common_prefix_blocks
  → GPUModelRunner._compute_cascade_attn_prefix_lens()
  → GPUModelRunner._build_attention_metadata(..., cascade_attn_prefix_lens)
  → AttentionMetadataBuilder.build(common_prefix_len, common_attn_metadata)
  → FlashAttentionMetadata(use_cascade=True, prefix/suffix metadata)
  → FlashAttentionImpl.forward()
  → cascade_attention()
      → prefix non-causal FA kernel
      → suffix causal FA kernel
      → merge_attn_states(...)
```

如果只记住一句话：

```text
prefix cache 负责“让请求共享 KV blocks”，cascade attention 负责“在 attention kernel 中利用这些共享 blocks”。
```

---

## 3. cascade attention 和 prefix cache 的关系

### 3.1 prefix cache 是 KV block 复用机制

Prefix cache 的核心是：

```text
多个请求如果有相同 prompt prefix，Scheduler / KVCacheManager 可以让它们复用已经计算好的 KV blocks。
```

这会影响：

```text
num_computed_tokens
block ids
block table
request 持有的 KV blocks
```

它主要发生在调度和 KV block 管理层。

### 3.2 cascade attention 是 attention 执行优化

Cascade attention 不负责发现 prefix cache，也不负责分配共享 blocks。

它假设上游已经存在：

```text
batch 内所有相关请求共享若干个开头 KV blocks。
```

然后在 attention forward 中把这些 blocks 拆出来单独处理。

### 3.3 二者不是必然绑定

可以这样理解：

```text
有 prefix cache / shared blocks：
  不一定启用 cascade attention，还要看配置、backend、启发式和限制条件。

启用 cascade attention：
  一定依赖 batch 内存在足够长的 shared prefix blocks。
```

所以 cascade attention 是 prefix cache 之上的可选 kernel-level 优化。

---

## 4. 配置入口：默认禁用，需要显式 opt-in

`ModelConfig.disable_cascade_attn` 定义在：`code/vllm/vllm/config/model.py:238`

默认值是：

```text
disable_cascade_attn: bool = True
```

注释说明：

```text
Cascade attention 不改变数学正确性，但可能带来数值差异；
默认禁用；
用户必须设置 disable_cascade_attn=False 才算 opt-in；
即使 opt-in，也只有启发式判断有收益时才会使用。
```

`GPUModelRunner.__init__()` 中记录：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:515`

```text
self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
```

也就是说：

```text
配置层决定“是否允许”；
runner + backend 启发式决定“本轮是否实际使用”。
```

---

## 5. 配置层会自动禁用 cascade 的场景

除了默认禁用，`VllmConfig` 初始化和校验中还会在一些场景强制关闭 cascade。

### 5.1 async speculative decoding

位置：`code/vllm/vllm/config/vllm.py:1108`

如果：

```text
speculative_config is not None
scheduler_config.async_scheduling=True
cascade 未禁用
```

会打印 warning 并设置：

```text
model_config.disable_cascade_attn = True
```

原因是当前 cascade attention 尚未兼容 async speculative decoding。

### 5.2 VLLM_BATCH_INVARIANT

位置：`code/vllm/vllm/config/vllm.py:1500`

如果启用 `VLLM_BATCH_INVARIANT`，也会禁用 cascade。

原因可以理解为：

```text
cascade attention 的启用依赖 batch 内公共 prefix、query lens、backend heuristic；
这会让同一请求在不同 batch 组合下走不同 attention 路径，和 batch invariant 目标冲突。
```

### 5.3 DBO / ubatching

位置：`code/vllm/vllm/config/vllm.py:1510`

如果 `parallel_config.use_ubatching=True`，会禁用 cascade。

对应 warning：

```text
Disabling cascade attention when DBO is enabled.
```

在 `GPUModelRunner.execute_model()` 中也有运行时保护：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4186`

```text
# Disable cascade attention when using microbatching (DBO)
if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
    cascade_attn_prefix_lens = ...
```

### 5.4 full CUDA graph 无 piecewise graph

位置：`code/vllm/vllm/config/vllm.py:1477`

如果启用了 full cudagraph，但没有 piecewise cudagraph，同时 cascade 未禁用，会 warning：

```text
No piecewise cudagraph for executing cascade attention.
Will fall back to eager execution if a batch runs into cascade attentions.
```

这说明 cascade attention 可能让当前 batch 不能直接走某些 full graph replay，需要 fallback 到 eager 或 piecewise 形态。

---

## 6. SchedulerOutput 中的 num_common_prefix_blocks

`SchedulerOutput` 定义在：`code/vllm/vllm/v1/core/sched/output.py:180`

字段：

```text
num_common_prefix_blocks: list[int]
```

注释：

```text
Number of common prefix blocks for all requests in each KV cache group.
This can be used for cascade attention.
```

它是一个 list，原因是 vLLM V1 支持多个 KV cache group：

```text
num_common_prefix_blocks[kv_cache_group_id]
  表示该 KV cache group 中所有相关请求共享的 prefix block 数。
```

注意单位是：

```text
block 数，不是 token 数。
```

后续 ModelRunner 会用：

```text
common_prefix_len = num_common_prefix_blocks * block_size
```

换算成 token 长度。

---

## 7. KVCacheManager 如何计算公共 prefix blocks

入口：`code/vllm/vllm/v1/core/kv_cache_manager.py:558`

```text
get_num_common_prefix_blocks(running_request_id) -> list[int]
```

它委托给 coordinator：

```text
self.coordinator.get_num_common_prefix_blocks(running_request_id)
```

对单一类型 KV cache manager，核心实现位于：`code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:493`

逻辑很直接：

```text
blocks = self.req_to_blocks[running_request_id]
num_common_blocks = 0
for block in blocks:
    if block.ref_cnt == len(self.req_to_blocks):
        num_common_blocks += 1
    else:
        break
return num_common_blocks
```

含义是：

```text
选一个 running request 的 block 序列；
从头开始看每个 block；
如果该 block 的 ref_cnt 等于当前所有持有 KV cache 的请求数，说明所有请求都共享它；
遇到第一个不共享的 block 就停止。
```

### 7.1 为什么可能低估 shared prefix

`kv_cache_manager.py` 的注释强调：

```text
拥有 allocated KV cache 的请求数 >= 当前 step 被调度的请求数。
```

也就是说，有些请求虽然没有在当前 step 调度，但仍然持有 KV blocks。

因此可能出现：

```text
当前被调度的请求都共享某个 prefix；
但另一个未调度请求不共享这个 prefix；
ref_cnt 判断因此失败；
num_common_prefix_blocks 返回 0。
```

这是保守低估，而不是错误。

### 7.2 sliding window manager 的特殊性

SlidingWindowManager 也实现 `get_num_common_prefix_blocks()`，但 sliding window 本身通常会让 cascade 在后续启发式中禁用。

原因是 cascade 当前不支持 sliding window attention 的语义。

---

## 8. GPUModelRunner 何时计算 cascade_attn_prefix_lens

在 `GPUModelRunner.execute_model()` 中，输入准备完成后、执行形态判断前，会计算 cascade prefix。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4186`

```text
cascade_attn_prefix_lens = None
if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
        num_scheduled_tokens_np,
        self.input_batch.num_computed_tokens_cpu[:num_reqs],
        scheduler_output.num_common_prefix_blocks,
    )
```

这里传入三类信息：

```text
num_scheduled_tokens_np
  本轮每个 request 的 query length。

num_computed_tokens_cpu
  每个 request 已经计算过的 token 数。

scheduler_output.num_common_prefix_blocks
  Scheduler / KVCacheManager 给出的每个 KV group 的共享 prefix block 数。
```

### 8.1 为什么在 _determine_batch_execution_and_padding() 前计算

随后会调用：

```text
_determine_batch_execution_and_padding(..., use_cascade_attn=cascade_attn_prefix_lens is not None)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4202`

这说明 cascade attention 会影响 batch 执行形态，例如 CUDA graph / padding / eager fallback 等判断。

---

## 9. cascade_attn_prefix_lens 是二维结构

`_compute_cascade_attn_prefix_lens()` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2583`

返回值：

```text
list[list[int]] | None
```

注释说明二维结构是：

```text
[kv_cache_group_id][attn_group_idx]
```

为什么要二维？

```text
一个模型可能有多个 KV cache group；
一个 KV cache group 内可能有多个 attention group；
不同 group 的 KV spec / backend / builder 能力不同；
因此每个 attention group 的 cascade prefix len 可能不同。
```

逻辑：

```text
for kv_cache_gid in range(num_kv_cache_groups):
    for attn_group in self.attn_groups[kv_cache_gid]:
        if EncoderOnlyAttentionSpec:
            cascade_attn_prefix_len = 0
        else:
            cascade_attn_prefix_len = _compute_cascade_attn_prefix_len(...)
        cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
        use_cascade_attn |= cascade_attn_prefix_len > 0

return cascade_attn_prefix_lens if any prefix > 0 else None
```

如果所有 group 都不能用 cascade，则返回 `None`。

---

## 10. common_prefix_len 如何从 blocks 变成 tokens

`_compute_cascade_attn_prefix_len()` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2621`

第一步：

```text
common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
```

如果为 0，直接返回 0。

### 10.1 不是所有 shared tokens 都能作为 cascade prefix

源码注释非常重要：

```text
common_prefix_len returned by this function represents the length used specifically for cascade attention,
not the actual number of tokens shared between requests.
```

原因是 cascade prefix kernel 不做 causal masking。

Cascade prefix kernel 会：

```text
把所有请求的 query tokens 拼起来；
把它们当作一个大请求；
对公共 prefix KV 做 bidirectional / non-causal attention。
```

如果公共 prefix 包含了某些请求当前 query 内才刚出现、但另一些请求已经 computed 的 token，就会产生错误可见性。

源码中的例子可以概括为：

```text
Request 1 query: [D, E, X]
Request 1 kv:    [A, B, C, D, E, X]
Request 1 computed: [A, B, C]

Request 2 query: [E, Y]
Request 2 kv:    [A, B, C, D, E, Y]
Request 2 computed: [A, B, C, D]

如果 prefix 用 [A, B, C, D, E]，
那么 Request 1 的 query token D 会在 prefix kernel 中看到 E，违反 causal mask。
```

因此 common prefix 要被截断。

### 10.2 截断到 min(num_computed_tokens)

代码：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2691`

```text
common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
```

含义：

```text
公共 prefix 只能包含所有请求在本轮开始前都已经 computed 的 token。
```

### 10.3 再截断到 block size 的倍数

代码：`gpu_model_runner.py:2693`

```text
common_prefix_len = common_prefix_len // block_size * block_size
```

原因是 cascade 依赖 block table 切分 prefix / suffix，公共 prefix 必须对齐到 KV cache block 边界。

---

## 11. backend builder 如何判断 use_cascade_attention

`AttentionMetadataBuilder` 基类定义：`code/vllm/vllm/v1/attention/backend.py:735`

默认实现：

```text
use_cascade_attention(...) -> False
```

所以 backend 默认不支持 cascade。

`GPUModelRunner._compute_cascade_attn_prefix_len()` 会调用具体 builder：

```text
attn_metadata_builder.use_cascade_attention(
    common_prefix_len=common_prefix_len,
    query_lens=num_scheduled_tokens,
    num_query_heads=self.num_query_heads,
    num_kv_heads=kv_cache_spec.num_kv_heads,
    use_alibi=self.use_alibi,
    use_sliding_window=use_sliding_window,
    use_local_attention=use_local_attention,
    num_sms=self.num_sms,
    dcp_world_size=self.dcp_world_size,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2705`

这一步把 runner 能看到的模型 / batch / backend 条件交给 builder 做最终判断。

### 11.1 FlashAttention builder 支持 cascade

`FlashAttentionMetadataBuilder.use_cascade_attention()` 定义在：`code/vllm/vllm/v1/attention/backends/flash_attn.py:724`

它调用同文件底部的启发式函数：

```text
use_cascade_attention(...)
```

位置：`flash_attn.py:1490`

### 11.2 其他常见 backend 默认禁用

例如：

```text
TritonAttentionBackend.use_cascade_attention() → False
FlexAttentionBackend.use_cascade_attention() → False
CPUAttentionBackend.use_cascade_attention() → False
ROCm attention backends → False
FlashInfer 当前路径也不作为主 cascade 支持路径
```

因此当前文档中可以把主实现重点放在 FlashAttention。

---

## 12. FlashAttention 的 cascade 启发式

`flash_attn.py:1490` 的 `use_cascade_attention()` 做两类判断：

```text
1. 支持性检查：当前配置能不能用 cascade；
2. 性能启发式：当前 batch 用 cascade 是否可能更快。
```

### 12.1 公共 prefix 太短则不用

```text
if common_prefix_len < 256:
    return False
```

阈值 256 tokens 是启发式常量。

含义：

```text
公共 prefix 太短，拆成两次 kernel + merge 的开销可能大于收益。
```

### 12.2 不支持 ALiBi / sliding window / local attention

```text
if use_alibi or use_sliding_window or use_local_attention:
    return False
```

因为 cascade prefix kernel 是 non-causal、无 mask 的公共 prefix attention，而这些 attention 变体需要额外位置偏置或局部可见性约束。

### 12.3 请求数太少则不用

```text
num_reqs = len(query_lens)
if num_reqs < 8:
    return False
```

含义：

```text
batch 内 query 太少，利用 shared prefix 的收益不足。
```

### 12.4 DCP 禁用

```text
if dcp_world_size > 1:
    return False
```

DCP 下 context 被分布到多个 rank，当前 cascade path 不支持。

### 12.5 和 FlashDecoding 的性能模型比较

如果普通 attention 不会使用 FlashDecoding：

```text
return True
```

因为 cascade 大概率能省内存带宽。

如果普通 attention 会使用 FlashDecoding，则用一个粗略模型比较：

```text
cascade_time < flash_decoding_time
```

其中考虑：

```text
num_query_heads
num_kv_heads
num_queries_per_kv
num_reqs
num_sms
common_prefix_len / kv_tile_size
```

这说明 cascade 不是“有公共 prefix 就用”，而是按启发式判断收益。

---

## 13. _build_attention_metadata 如何传递 cascade prefix

`GPUModelRunner.execute_model()` 调用 `_build_attention_metadata()` 时传入：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4308`

```text
cascade_attn_prefix_lens=cascade_attn_prefix_lens
```

在 `_build_attention_metadata()` 中，每个 attention group 构造 metadata 时会取：

```text
common_prefix_len = cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
```

然后调用：

```text
builder.build(common_prefix_len, cm, ...)
```

如果某个 group 的 prefix len 为 0，builder 仍然正常构造非 cascade metadata。

因此 cascade 是 per attention group 的：

```text
同一个 batch 内，某些 group 可以 cascade，某些 group 可以不 cascade。
```

---

## 14. FlashAttentionMetadata 如何表达 cascade

`FlashAttentionMetadata` 定义在：`code/vllm/vllm/v1/attention/backends/flash_attn.py:236`

cascade 相关字段：

```text
use_cascade: bool
common_prefix_len: int
cu_prefix_query_lens: torch.Tensor | None
prefix_kv_lens: torch.Tensor | None
suffix_kv_lens: torch.Tensor | None
prefix_scheduler_metadata: torch.Tensor | None
scheduler_metadata: torch.Tensor | None
```

其中：

```text
use_cascade
  是否启用 cascade path。

common_prefix_len
  用于 cascade 的公共 prefix token 长度，已按 block size 对齐并被安全截断。

cu_prefix_query_lens
  prefix kernel 的 query 起止。当前是 [0, num_actual_tokens]，表示把所有 query 拼成一个 batch item。

prefix_kv_lens
  prefix kernel 的 KV 长度，当前是 [common_prefix_len]。

suffix_kv_lens
  每个 request 的 suffix KV 长度，等于 seq_lens - common_prefix_len。

prefix_scheduler_metadata
  FA3 AOT scheduler 给 prefix kernel 使用。

scheduler_metadata
  suffix kernel 使用。
```

---

## 15. FlashAttentionMetadataBuilder 如何构造 cascade metadata

在 `FlashAttentionMetadataBuilder.build()` 中：

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:489`

```text
use_cascade = common_prefix_len > 0
```

如果 `use_cascade=True`，构造：

```text
cu_prefix_query_lens = torch.tensor([0, num_actual_tokens])
prefix_kv_lens = torch.tensor([common_prefix_len])
suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
```

位置：`flash_attn.py:528`

然后为两段 attention 分别构造 scheduler metadata：

```text
prefix_scheduler_metadata = schedule(
    batch_size=1,
    cu_query_lens=cu_prefix_query_lens,
    max_query_len=num_actual_tokens,
    seqlens=prefix_kv_lens,
    max_seq_len=common_prefix_len,
    causal=False,
)

scheduler_metadata = schedule(
    batch_size=num_reqs,
    cu_query_lens=query_start_loc,
    max_query_len=max_query_len,
    seqlens=suffix_kv_lens,
    max_seq_len=max_seq_len - common_prefix_len,
    causal=True,
)
```

这正对应：

```text
prefix：一个“大请求”，non-causal；
suffix：原始 request batch，causal。
```

---

## 16. FlashAttention forward 如何执行 cascade

在 `FlashAttentionImpl.forward()` 中：

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:807`

```text
if not attn_metadata.use_cascade:
    走普通 flash_attn_varlen_func
else:
    cascade_attention(...)
```

调用 cascade：`flash_attn.py:1039`

```text
cascade_attention(
    output,
    query,
    key_cache,
    value_cache,
    cu_query_lens=attn_metadata.query_start_loc,
    max_query_len=attn_metadata.max_query_len,
    cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
    prefix_kv_lens=attn_metadata.prefix_kv_lens,
    suffix_kv_lens=attn_metadata.suffix_kv_lens,
    block_table=attn_metadata.block_table,
    common_prefix_len=attn_metadata.common_prefix_len,
    prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
    suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
    ...
)
```

---

## 17. cascade_attention() 内部做了什么

`cascade_attention()` 定义在：`code/vllm/vllm/v1/attention/backends/flash_attn.py:1568`

它做三步。

### 17.1 计算公共 prefix block 数

```text
block_size = key_cache.shape[-3]
assert common_prefix_len % block_size == 0
num_common_kv_blocks = common_prefix_len // block_size
assert num_common_kv_blocks > 0
```

### 17.2 prefix attention：non-causal

第一段：

```text
prefix_output, prefix_lse = flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    cu_seqlens_q=cu_prefix_query_lens,
    seqused_k=prefix_kv_lens,
    max_seqlen_q=num_tokens,
    max_seqlen_k=common_prefix_len,
    causal=False,
    block_table=block_table[:1],
    return_softmax_lse=True,
)
```

关键点：

```text
所有 query token 被当成一个 batch item；
只 attend 到第一条 request block_table 中的 shared prefix blocks；
causal=False，因为公共 prefix 已保证只包含所有请求本轮前已 computed 的 token；
返回 output 和 LSE，用于后续 merge。
```

### 17.3 suffix attention：causal

第二段：

```text
suffix_output, suffix_lse = flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    cu_seqlens_q=cu_query_lens,
    seqused_k=suffix_kv_lens,
    max_seqlen_q=max_query_len,
    max_seqlen_k=max_kv_len - common_prefix_len,
    causal=True,
    block_table=block_table[:, num_common_kv_blocks:],
    return_softmax_lse=True,
)
```

关键点：

```text
suffix 仍按原始 request batch 组织；
KV blocks 从 common prefix 后面开始；
causal=True，保证当前 query 对私有 suffix 的可见性正确；
同样返回 output 和 LSE。
```

### 17.4 merge_attn_states 合并两段结果

最后：

```text
merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```

位置：`flash_attn.py:1341`

因为 attention softmax 的归一化跨 prefix 和 suffix 两段，不能简单把两个 output 相加。

必须用 LSE 合并两个 attention states，得到等价于一次完整 attention 的结果。

---

## 18. 为什么 prefix non-causal 是安全的

prefix kernel 不做 mask，看起来危险。

安全前提来自 `_compute_cascade_attn_prefix_len()` 的截断：

```text
common_prefix_len <= min(num_computed_tokens)
```

这保证：

```text
公共 prefix 中的所有 token，对 batch 内所有请求来说，都是本轮开始前已经存在的 context；
任何本轮 query token 都可以 attend 到这些历史 context；
因此 prefix kernel 不需要 causal mask。
```

如果不做这个截断，就可能出现某个请求的 query token 看见同一轮中未来 token 的问题。

---

## 19. 和 chunked prefill / mixed batch 的关系

cascade attention 可以和一些 mixed query length 情况同时出现，但前提是：

```text
公共 prefix 只包含所有请求已经 computed 的 blocks；
query_lens 和 num_computed_tokens 通过启发式检查；
backend 支持当前形态；
不是 local attention / sliding window / DCP / DBO 等禁用场景。
```

对于 chunked prefill：

```text
请求可能仍在 prompt prefill 阶段；
num_computed_tokens 表示本轮开始前已经完成的 prefix；
公共 prefix 不能覆盖本轮 chunk 内的 token；
因此 min(num_computed_tokens) 截断尤其重要。
```

对于 decode-only：

```text
query_lens 通常全为 1；
如果 GQA/MQA 触发 FlashDecoding，cascade 会和 FlashDecoding 做性能模型比较；
只有 cascade_time 更小才使用。
```

---

## 20. 和 sliding window / local attention / ALiBi / DCP 的关系

FlashAttention cascade 明确禁用这些场景：

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:1513`

```text
if use_alibi or use_sliding_window or use_local_attention:
    return False
```

以及：

```text
if dcp_world_size > 1:
    return False
```

原因：

```text
ALiBi：attention score 带位置偏置，prefix/suffix 拆分和 merge 更复杂。
sliding window：每个 query 可见 KV 取决于局部窗口，不是统一 shared prefix。
local attention：可见性由 chunk/local 规则决定，不能简单 prefix non-causal。
DCP：context KV 分布到多个 rank，当前 cascade path 不支持。
```

`cascade_attention()` 内部也有断言：

```text
assert alibi_slopes is None
assert sliding_window == (-1, -1)
```

位置：`flash_attn.py:1594`

---

## 21. 和 CUDA graph 的关系

cascade attention 可能影响 CUDA graph 执行形态。

### 21.1 runner 会把 use_cascade_attn 传给 batch 执行形态判断

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4202`

```text
_determine_batch_execution_and_padding(
    ...,
    use_cascade_attn=cascade_attn_prefix_lens is not None,
)
```

这表示一旦本轮可能使用 cascade，runner 会在 cudagraph / padding / batch descriptor 决策时考虑它。

### 21.2 full graph 无 piecewise 时可能 fallback eager

配置校验中有 warning：

```text
No piecewise cudagraph for executing cascade attention.
Will fall back to eager execution if a batch runs into cascade attentions.
```

位置：`code/vllm/vllm/config/vllm.py:1477`

### 21.3 FA3 AOT scheduler metadata

FlashAttentionMetadataBuilder 在 cascade 下会分别为 prefix / suffix 构造 scheduler metadata：

```text
prefix_scheduler_metadata
scheduler_metadata
```

如果 full CUDA graph + FA3 AOT schedule，还会把 scheduler metadata 写入预分配 buffer，避免 graph capture 中动态分配或无效 metadata。

---

## 22. 和 backend 支持的关系

Cascade attention 是 builder 能力，不是所有 backend 自动支持。

`AttentionMetadataBuilder.use_cascade_attention()` 默认返回 False。

常见情况：

```text
FlashAttention：主支持路径，包含启发式和 cascade_attention 实现。
TritonAttention：返回 False。
FlexAttention：返回 False。
CPUAttention：返回 False。
ROCm attention：返回 False。
FlashInfer：当前不是主 cascade 支持路径。
MLA / Mamba / GDN：按各自体系，不走此 FlashAttention cascade 实现。
```

因此即使全局 `disable_cascade_attn=False`，如果当前 backend 不支持，最终仍然不会启用。

---

## 23. 一个完整例子：多个 decode 请求共享长 prefix

假设：

```text
block_size = 16
batch 内 16 个请求共享前 64 个 blocks
num_common_prefix_blocks = 64
common_prefix_len 初始 = 64 * 16 = 1024 tokens
每个请求本轮 decode 1 个 token
所有请求 num_computed_tokens >= 1024
backend = FlashAttention
无 ALiBi / sliding window / local attention / DCP
用户已设置 disable_cascade_attn=False
```

执行链路：

```text
1. KVCacheManager.get_num_common_prefix_blocks()
   返回 [64]。

2. SchedulerOutput.num_common_prefix_blocks = [64]。

3. GPUModelRunner._compute_cascade_attn_prefix_lens()
   common_prefix_len = 1024；
   min(num_computed_tokens) 不截断；
   block 对齐仍为 1024；
   FlashAttention builder heuristic 判断 prefix 足够长、请求数足够多。

4. _build_attention_metadata()
   对该 attention group 调 builder.build(common_prefix_len=1024, cm)。

5. FlashAttentionMetadataBuilder.build()
   use_cascade=True；
   cu_prefix_query_lens=[0, num_actual_tokens]；
   prefix_kv_lens=[1024]；
   suffix_kv_lens=seq_lens-1024。

6. FlashAttentionImpl.forward()
   调 cascade_attention()。

7. cascade_attention()
   prefix kernel：所有 query attend 到 shared prefix；
   suffix kernel：每个 request attend 到自己的 suffix；
   merge_attn_states 合并结果。
```

最终输出等价于普通 full attention，但可能减少共享 prefix 带来的重复访存 / 计算开销。

---

## 24. 一个完整例子：为什么要按 min(num_computed_tokens) 截断

假设两个请求共享 blocks 表示它们都有 `[A, B, C, D, E]`，但本轮 query 不同：

```text
Request 1:
  computed = [A, B, C]
  query = [D, E, X]

Request 2:
  computed = [A, B, C, D]
  query = [E, Y]
```

如果 cascade prefix 用 `[A, B, C, D, E]`：

```text
prefix kernel causal=False；
Request 1 的 query token D 会看到 E；
这违反了 causal 语义。
```

所以 runner 会：

```text
common_prefix_len = min(common_prefix_len, min(num_computed_tokens))
```

得到最多 `[A, B, C]`。

这样 prefix kernel 中任何 query attend 到 `[A, B, C]` 都是安全的。

---

## 25. 容易疑惑的点

### 25.1 cascade attention 会改变输出语义吗？

设计目标是不改变数学语义。

它把完整 attention 拆成 prefix / suffix 两段，再用 LSE 合并 attention states。

但由于浮点计算顺序不同，可能有数值差异，这也是默认禁用、需要 opt-in 的原因之一。

### 25.2 有 common prefix blocks 就一定启用吗？

不是。

还要满足：

```text
用户允许 cascade；
没有 DBO / batch invariant / async spec decode 等禁用条件；
backend 支持；
common_prefix_len >= 256；
num_reqs >= 8；
无 ALiBi / sliding window / local attention / DCP；
启发式认为比普通 FlashAttention / FlashDecoding 更划算。
```

### 25.3 num_common_prefix_blocks 是当前 scheduled requests 的公共 prefix 吗？

不完全是。

它是基于“所有持有 allocated KV cache 的请求”计算的，可能包含未调度请求。

因此它可能保守返回 0，即使当前 scheduled requests 之间有共享 prefix。

### 25.4 common_prefix_len 是真实共享 token 数吗？

不是。

它是“用于 cascade attention 的安全 prefix 长度”：

```text
shared blocks token 长度；
再按 min(num_computed_tokens) 截断；
再按 block size 对齐；
再经过 backend heuristic 判断。
```

### 25.5 prefix attention 为什么 causal=False？

因为 prefix 已被限制为所有请求本轮开始前都已经 computed 的历史 context。

对任何本轮 query 来说，这些 prefix token 都是过去 token，因此不需要 causal mask。

### 25.6 suffix attention 为什么 causal=True？

suffix 包含每个请求公共 prefix 之后的私有 context 和当前 query 区域。

这里仍然需要 per-request causal mask，防止 query 看到未来 token。

### 25.7 为什么需要 merge_attn_states？

softmax 归一化本来应该覆盖 prefix + suffix 全部 KV。

拆成两次 attention 后，不能直接加 output，必须用 prefix/suffix 的 LSE 重新合并。

### 25.8 DBO 为什么禁用 cascade？

DBO / ubatching 会把 batch 切成 microbatches，公共 prefix、query 拼接、prefix/suffix metadata 和 graph 执行边界都更复杂。

当前 vLLM 直接禁用 cascade，避免语义和性能路径复杂化。

---

## 26. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心输入 | 核心输出 | 作用 |
|---|---|---|---|---|
| 配置控制 | `ModelConfig.disable_cascade_attn` | 用户配置 | 是否允许 cascade | 默认禁用，用户 opt-in |
| 公共 block 统计 | `KVCacheManager.get_num_common_prefix_blocks()` | running request id | 每个 KV group 的公共 block 数 | 找到 batch 可共享 prefix blocks |
| 调度输出 | `SchedulerOutput.num_common_prefix_blocks` | KV cache manager 结果 | `list[int]` | 把公共 block 信息传给 worker |
| runner 计算 | `_compute_cascade_attn_prefix_lens()` | scheduled tokens、computed tokens、common blocks | 二维 prefix lens 或 None | 按 group 判断是否可能启用 cascade |
| 安全截断 | `_compute_cascade_attn_prefix_len()` | common blocks、computed tokens、block size | safe common_prefix_len | 避免 prefix non-causal 看到未来 token |
| backend 判断 | `builder.use_cascade_attention()` | prefix len、query lens、heads、特性 | bool | backend 支持性和性能启发式 |
| metadata 构造 | `FlashAttentionMetadataBuilder.build()` | common_prefix_len、common metadata | `FlashAttentionMetadata` | 构造 prefix/suffix lens 和 scheduler metadata |
| forward 分支 | `FlashAttentionImpl.forward()` | `use_cascade` | 普通 FA 或 cascade | 选择 attention 执行路径 |
| prefix kernel | `cascade_attention()` | shared prefix blocks | prefix output + LSE | 所有 query attend 到公共 prefix |
| suffix kernel | `cascade_attention()` | suffix blocks | suffix output + LSE | 每个请求 attend 到私有 suffix |
| 合并 | `merge_attn_states()` | prefix/suffix output + LSE | final output | 合并成等价完整 attention 输出 |

---

## 27. 总结

Cascade attention 可以压缩成下面这条线：

```text
prefix cache / block sharing
  → KVCacheManager 统计公共 prefix blocks
  → SchedulerOutput.num_common_prefix_blocks
  → GPUModelRunner 计算安全 common_prefix_len
  → backend heuristic 判断是否启用
  → FlashAttentionMetadata 标记 use_cascade
  → prefix non-causal attention
  → suffix causal attention
  → LSE merge
  → final attention output
```

它的核心设计是：

```text
公共 prefix 只做一次大 attention；
请求私有 suffix 仍按请求做 causal attention；
两段结果用 LSE 合并，保持完整 attention 的 softmax 语义。
```

如果只记住最小心智模型：

```text
Cascade attention 是把“共享 KV prefix”从每个请求的 attention 中抽出来，单独跑一个无 mask 的 prefix kernel，再和每个请求自己的 causal suffix kernel 合并。
```