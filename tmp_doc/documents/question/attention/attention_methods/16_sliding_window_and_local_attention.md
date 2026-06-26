# 16. Sliding Window / Local Attention 如何工作？

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/config/cache.py`
- `vllm/vllm/engine/arg_utils.py`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/attention/chunked_local_attention.py`
- `vllm/vllm/model_executor/models/`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/v1/core/kv_cache_utils.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/attn_utils.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/backends/utils.py`

本文用于梳理 sliding window attention / chunked local attention 在 vLLM V1 中的来源、KV cache 规格、metadata 表示、backend 执行语义，以及它们和 prefix cache / cascade attention / hybrid KV cache 的关系。

---

## 1. 一句话回答

Sliding window / local attention 不是一套独立的模型执行链路，而是在普通 decoder attention 链路上增加“每个 query 只能看局部 KV 范围”的约束。

最小主链路可以概括为：

```text
HF config / 模型层定义
  → ModelConfig.get_sliding_window() 或 attention_chunk_size
  → Attention(..., per_layer_sliding_window=...)
       或 ChunkedLocalAttention(..., attention_chunk_size=...)
  → KVCacheSpec: SlidingWindowSpec / ChunkedLocalAttentionSpec / FullAttentionSpec(...window...)
  → KV cache manager 决定窗口外 block 是否可跳过 / 回收
  → GPUModelRunner._build_attention_metadata()
  → backend metadata builder
  → FlashAttention / Triton / FlashInfer 等 kernel 的 window_size / local virtual batch
```

它的核心语义是：

```text
full causal attention：第 i 个 token 可看 [0, i]
sliding window：第 i 个 token 只可看 [max(0, i - window + 1), i]
chunked local attention：第 i 个 token 只可看当前 chunk 内的 token
```

注意这里有两个问题必须分开看：

```text
1. attention 计算时是否能看窗口外 token；
2. KV cache 内是否还保留窗口外 token。
```

前者由 attention backend 的 mask / window 语义保证；后者由 KV cache spec、hybrid KV cache manager、prefix cache 策略决定。

---

## 2. sliding window 和 full causal attention 的区别

### 2.1 full causal attention

full causal attention 的可见范围是完整历史：

```text
query position = i
可见 KV position = 0..i
```

例如：

```text
tokens:  0 1 2 3 4 5 6 7
query:                 7
visible: 0 1 2 3 4 5 6 7
```

这也是普通 decoder-only LLM 的默认 attention mask。

### 2.2 sliding window attention

sliding window 只保留最近的窗口参与 attention：

```text
query position = i
window = W
可见 KV position = max(0, i - W + 1)..i
```

例如 `sliding_window = 4`：

```text
tokens:  0 1 2 3 4 5 6 7
query:                 7
visible:         4 5 6 7
skipped: 0 1 2 3
```

对应 KV cache manager 的注释也使用了这个例子：`SlidingWindowManager.get_num_skipped_tokens()` 会在 `num_computed_tokens=7, sliding_window=4` 时返回 `4`，表示 token `0..3` 对下一步 attention 已经不可见。

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:770`

### 2.3 chunked local attention

chunked local attention 不是“最近 W 个 token”的滑动窗口，而是“当前 chunk 内局部可见”。

如果 `attention_chunk_size = 8`，则 chunk 边界类似：

```text
[0..7] [8..15] [16..23] ...
```

当下一 token 位于第二个 chunk 内，它只能看第二个 chunk 中已经可见的部分，而不是继续向前滑动任意长度。

`ChunkedLocalAttentionManager.get_num_skipped_tokens()` 的语义是：

```text
num_skipped_tokens = floor(num_computed_tokens / attention_chunk_size)
                     * attention_chunk_size
```

即已经落在当前 chunk 左侧的完整 chunk 都可以跳过。

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:905`

---

## 3. 配置来源：sliding_window 从哪里来

### 3.1 ModelConfig 从 HF config 读取 sliding_window

`ModelConfig.get_sliding_window()` 很直接：从 `hf_text_config.sliding_window` 读取。

位置：`vllm/vllm/config/model.py:1225`

```text
hf_text_config.sliding_window
  → ModelConfig.get_sliding_window()
```

初始化时还有两个处理：

```text
1. 如果 checkpoint 把 sliding_window 写成 0，vLLM 会把它视为禁用；
2. 如果用户设置 disable_sliding_window，最终会把 hf_text_config.sliding_window 置为 None。
```

位置：`vllm/vllm/config/model.py:648`、`vllm/vllm/config/model.py:716`

这意味着：

```text
sliding_window = 0
  → disable_sliding_window = True
  → hf_text_config.sliding_window = None
```

### 3.2 CacheConfig.sliding_window 只适合“全模型都是 sliding window”

创建 `CacheConfig` 时，`arg_utils` 会先判断模型是否是 interleaved attention。

位置：`vllm/vllm/engine/arg_utils.py:1825`

关键逻辑是：

```text
if not is_interleaved(model_config.hf_text_config):
    sliding_window = model_config.get_sliding_window()
else:
    sliding_window = None
```

原因是：

```text
如果模型是 interleaved sliding window，
例如部分层 full attention、部分层 sliding attention，
就不能把 CacheConfig.sliding_window 设置成全局值，
否则会错误覆盖 full attention 层。
```

所以 vLLM 分成两类处理：

| 模型形态 | sliding window 来源 | 传给 Attention 的方式 |
|---|---|---|
| 全层 sliding window | `CacheConfig.sliding_window` | `Attention` 默认读取 `cache_config.sliding_window` |
| interleaved / per-layer sliding | 模型层自己判断 | `per_layer_sliding_window=...` |

### 3.3 Attention 层如何确定本层 sliding_window

`Attention.__init__()` 的优先级是：

位置：`vllm/vllm/model_executor/layers/attention/attention.py:227`

```text
if per_layer_sliding_window is not None:
    sliding_window = per_layer_sliding_window
elif cache_config is not None:
    sliding_window = cache_config.sliding_window
else:
    sliding_window = None
```

也就是说：

```text
per-layer 配置优先于全局 cache_config.sliding_window。
```

### 3.4 模型定义如何传 per_layer_sliding_window

很多模型会在 layer 初始化时按 layer type 决定是否启用 sliding window。

以 Llama 相关实现为例：

位置：`vllm/vllm/model_executor/models/llama.py:202`

```text
layer_types[effective_layer_idx] == "sliding_attention"
  → sliding_window = config.sliding_window
  → Attention(..., per_layer_sliding_window=sliding_window)
```

这类模型通常是 interleaved attention：某些层是 full attention，某些层是 sliding attention。

其他模型也有类似传参，例如 Gemma、GPT-OSS、Mistral 派生模型、Cohere、Command-R、MiniMax 等。

### 3.5 attention_chunk_size 从哪里来

`attention_chunk_size` 也是从 HF text config 读取：

位置：`vllm/vllm/config/model.py:549`

```text
self.attention_chunk_size = getattr(hf_text_config, "attention_chunk_size", None)
```

Llama4 中会根据 `config.attention_chunk_size` 选择 `ChunkedLocalAttention`：

位置：`vllm/vllm/model_executor/models/llama4.py:253`

```text
use_chunked_local_attn = not self.nope and config.attention_chunk_size
attn_cls = ChunkedLocalAttention if use_chunked_local_attn else Attention
```

所以：

```text
sliding_window       → 普通 Attention + window_size
attention_chunk_size → ChunkedLocalAttention 包装普通 backend
```

---

## 4. Attention 层到 KVCacheSpec 的转换

### 4.1 Attention 保存 self.sliding_window

`Attention.__init__()` 最终会把窗口保存到：

```text
self.sliding_window
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:304`

它同时把 `sliding_window` 传给具体 backend impl：

```text
impl_cls(..., sliding_window, kv_cache_dtype, ...)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:387`

所以同一个值会进入两条链路：

```text
self.sliding_window
  → get_kv_cache_spec() 生成 KV cache 规格

impl.sliding_window
  → backend forward 时传给 kernel
```

### 4.2 get_kv_cache_spec() 如何区分 full / sliding

`Attention.get_kv_cache_spec()` 会根据 `self.sliding_window` 返回不同 spec。

位置：`vllm/vllm/model_executor/layers/attention/attention.py:581`

```text
self.sliding_window is not None
  → SlidingWindowSpec(..., sliding_window=self.sliding_window)

self.sliding_window is None
  → FullAttentionSpec(...)
```

这一步的意义是：

```text
Attention 层声明“我需要什么样的 KV cache 管理策略”。
```

### 4.3 SlidingWindowSpec 表示什么

`SlidingWindowSpec` 继承自 `AttentionSpec`，核心字段是：

位置：`vllm/vllm/v1/kv_cache_interface.py:478`

```text
SlidingWindowSpec(
    block_size,
    num_kv_heads,
    head_size,
    dtype,
    sliding_window,
)
```

它不仅表示 mask 语义，还影响内存上限估计。

`max_admission_blocks_per_request()` 的核心计算是：

位置：`vllm/vllm/v1/kv_cache_interface.py:506`

```text
num_tokens = min(sliding_window - 1 + max_num_batched_tokens, max_model_len)
max_blocks = ceil(num_tokens / block_size) + 1
```

为什么是 `sliding_window - 1 + max_num_batched_tokens`？

```text
当前新调度的一批 token 需要写入 KV；
每个新 token 最多还需要看前面 sliding_window - 1 个历史 token；
所以单请求同时真实需要保留的 KV 上限约等于：
历史窗口 + 当前调度 token。
```

为什么还要 `+1 block`？

```text
窗口起点不一定和 block 边界对齐。
例如 block_size=4，窗口覆盖 4 个 token，可能跨两个 block。
```

### 4.4 ChunkedLocalAttentionSpec 表示什么

`ChunkedLocalAttention` 会返回 `ChunkedLocalAttentionSpec`。

位置：`vllm/vllm/model_executor/layers/attention/chunked_local_attention.py:120`

核心字段是：

```text
ChunkedLocalAttentionSpec(
    block_size,
    num_kv_heads,
    head_size,
    dtype,
    attention_chunk_size,
)
```

它的内存上限估计是：

位置：`vllm/vllm/v1/kv_cache_interface.py:441`

```text
num_tokens = min(attention_chunk_size + max_num_batched_tokens, max_model_len)
max_blocks = ceil(num_tokens / block_size)
```

语义是：

```text
本轮最多保留当前 local chunk + 新调度 token 需要的 KV。
```

### 4.5 FullAttentionSpec 也可能携带 sliding_window

`FullAttentionSpec` 中也有：

位置：`vllm/vllm/v1/kv_cache_interface.py:204`

```text
sliding_window: int | None = None
attention_chunk_size: int | None = None
```

这看起来有点反直觉，但它用于一种特殊情况：

```text
当 hybrid KV cache manager 被禁用，
模型同时有 full attention 层和 sliding/local 层时，
vLLM 会把 sliding/local 层的 KV cache spec 转成 FullAttentionSpec，
让所有层在 KV cache manager 视角下统一成 full 类型。
```

位置：`vllm/vllm/v1/core/kv_cache_utils.py:1392`

转换后：

```text
KV cache manager：按 full attention 分配和保留 blocks
attention backend：仍然按 sliding/local 语义计算 attention
```

这就是文档开头说的：

```text
“是否保存窗口外 KV” 和 “是否参与 attention” 是两个问题。
```

---

## 5. KV cache manager：窗口外 token 如何处理

### 5.1 SlidingWindowManager 会计算可跳过 token

`SlidingWindowSpec` 会注册到 `SlidingWindowManager`。

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:1369`

`SlidingWindowManager.get_num_skipped_tokens()`：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:770`

```text
return max(0, num_computed_tokens - sliding_window + 1)
```

例如：

```text
sliding_window = 4
num_computed_tokens = 7

当前下一 token 的窗口是 [4, 5, 6, 7]
跳过 token [0, 1, 2, 3]
get_num_skipped_tokens() = 4
```

这些 skipped tokens 对当前请求后续 attention 不再可见，因此对应的完整 blocks 可以作为回收 / 不再需要真实保留的候选。

### 5.2 ChunkedLocalAttentionManager 会跳过当前 chunk 左侧 token

`ChunkedLocalAttentionSpec` 会注册到 `ChunkedLocalAttentionManager`。

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:1383`

`ChunkedLocalAttentionManager.get_num_skipped_tokens()`：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:946`

```text
num_skipped_tokens = (num_computed_tokens // attention_chunk_size)
                     * attention_chunk_size
```

例如：

```text
attention_chunk_size = 8
num_computed_tokens = 13

下一 token 位于 chunk [8..15]
跳过 token [0..7]
get_num_skipped_tokens() = 8
```

### 5.3 allocate/admission 会用 skipped token 降低真实 block 需求

`SingleTypeKVCacheManager.get_num_blocks_to_allocate()` 会读取 `get_num_skipped_tokens()`，并用它计算新需要的 blocks。

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:132`

核心关系是：

```text
num_required_blocks = ceil(num_tokens / block_size)
num_skipped_blocks = num_skipped_tokens // block_size
num_local_computed_blocks = 已经命中或已经分配的本地 blocks
num_new_blocks = max(num_required_blocks - max(num_skipped_blocks,
                                                num_local_computed_blocks), 0)
```

这表示：

```text
窗口外完整 blocks 不再强迫本请求继续占用真实 KV 容量。
```

同时，启动时的容量估计和运行时 admission 使用同一个上限：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:1347`

```text
SlidingWindowSpec / ChunkedLocalAttentionSpec
  → max_admission_blocks_per_request()
  → manager.max_admission_blocks_per_request
```

这样可以避免启动时认为够、运行时却因为 chunked prefill 中间峰值而 OOM。

### 5.4 hybrid KV cache manager 被禁用时会退化为 full 保存

`unify_hybrid_kv_cache_specs()` 的 warning 说得很清楚：

位置：`vllm/vllm/v1/core/kv_cache_utils.py:1407`

```text
Hybrid KV cache manager is disabled ...
This means we do not enable any optimizations for saving KV cache memory
(e.g., dropping the KV cache outside the sliding window).
The compute of layers like sliding window is still saved.
```

也就是说：

| 场景 | attention 计算 | KV cache 保存 |
|---|---|---|
| SlidingWindowSpec + SlidingWindowManager | 只看窗口内 | 窗口外 blocks 可跳过 / 回收 |
| FullAttentionSpec(sliding_window=...) | 只看窗口内 | 按 full attention 方式保存 |
| disable_sliding_window | 看完整历史 | 按 full attention 保存 |

---

## 6. GPUModelRunner 如何构造 attention metadata

### 6.1 _build_attention_metadata() 不直接保存 sliding_window 字段

`GPUModelRunner._build_attention_metadata()` 构造的是公共 metadata：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2216`

关键字段包括：

```text
query_start_loc
seq_lens
max_seq_len
max_query_len
block_table_tensor
slot_mapping
causal
positions
is_prefilling
dcp_local_seq_lens
mm_req_doc_ranges
```

这些字段放进 `CommonAttentionMetadata`：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2338`

`CommonAttentionMetadata` 自身并没有一个通用 `sliding_window` 字段。

原因是：

```text
sliding_window 是 layer/backend impl 的静态属性，
不是每个 batch 动态变化的 metadata。
```

batch metadata 负责描述：

```text
本轮有哪些 query、每个请求长度是多少、KV block table 在哪里、slot mapping 是什么。
```

backend impl 负责把自己的 `self.sliding_window` 传给 kernel。

### 6.2 max_seq_len 在 CUDA graph capture 下会特殊处理

`_build_attention_metadata()` 中：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2247`

```text
if for_cudagraph_capture:
    max_seq_len = self.max_model_len
else:
    max_seq_len = max(optimistic_seq_lens_cpu)
```

注释说明了原因：

```text
对某些 backend，例如 FlashAttention，sliding window 模型在 capture 时
需要让 backend 看到足够大的 max_seq_len，确保选择正确 kernel。
```

### 6.3 cascade attention 会显式判断 sliding/local

`_get_cascade_attn_prefix_len()` 会判断 KV spec 是否是 sliding/local：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2632`

```text
use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec)
                  or FullAttentionSpec.sliding_window is not None

use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec)
                   or FullAttentionSpec.attention_chunk_size is not None
```

然后把这两个布尔值传给 builder 的 `use_cascade_attention()`。

FlashAttention 的 cascade heuristic 会直接拒绝这些情况：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:1195`

```text
if use_alibi or use_sliding_window or use_local_attention:
    return False
```

所以：

```text
cascade attention 当前不和 sliding window / chunked local attention 同时启用。
```

---

## 7. backend 如何执行 sliding window

### 7.1 Attention backend selection 本身不以 sliding_window 为独立维度

`get_attn_backend()` 的选择参数包括：

位置：`vllm/vllm/v1/attention/selector.py:53`

```text
head_size
dtype
kv_cache_dtype
block_size
use_mla
has_sink
use_sparse
use_mm_prefix
use_per_head_quant_scales
attn_type
use_non_causal
use_batch_invariant
use_kv_connector
```

这里没有单独的 `sliding_window`。

原因是：

```text
多数标准 attention backend 把 sliding window 当作 forward/kernel 参数支持，
而不是选择 backend 的核心条件。
```

但具体 backend 仍可能在运行时对窗口组合有限制。

### 7.2 FlashAttentionImpl 的 window_size

`FlashAttentionImpl.__init__()` 会把 `sliding_window` 转成 flash-attn 风格的 window tuple。

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:650`

```text
sliding_window is None
  → (-1, -1)

decoder attention
  → (sliding_window - 1, 0)

encoder-only attention
  → (sliding_window - 1, sliding_window - 1)
```

含义是：

```text
(window_left, window_right)
```

decoder attention 只能看左侧历史，所以右窗口为 `0`。

执行时会把它传给 `flash_attn_varlen_func()`：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:870`

```text
window_size=sliding_window_size
```

对于非 causal / dynamic causal 情况，如果右窗口是 0，FlashAttention 会在必要时改成对称窗口：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:837`

```text
(sliding_window - 1, 0)
  → (sliding_window - 1, sliding_window - 1)
```

### 7.3 TritonAttentionImpl 的 window_size

Triton backend 也在 impl 初始化时转换窗口：

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:460`

```text
None → (-1, -1)
decoder → (sliding_window - 1, 0)
encoder / encoder_only → (sliding_window - 1, sliding_window - 1)
```

forward 时传给 unified attention：

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:642`

```text
unified_attention(..., window_size=self.sliding_window, ...)
```

encoder 路径下则传给 `context_attention_fwd()` 的：

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:709`

```text
sliding_window_q=self.sliding_window[0]
sliding_window_k=self.sliding_window[1]
```

### 7.4 FlashInfer 的窗口限制

FlashInfer metadata builder 会检查所有层的窗口左边界是否一致。

位置：`vllm/vllm/v1/attention/backends/flashinfer.py:1018`

如果窗口不一致，会抛错：

```text
Window left is not the same for all layers.
One potential fix is to set disable_sliding_window=True
```

这类约束通常出现在：

```text
不同层 window_left 不同、backend 需要全局统一 hyperparameters、
或者特定 wrapper 不支持 interleaved window 参数。
```

### 7.5 cascade_attention 函数本身也拒绝 sliding window

FlashAttention 的 `cascade_attention()` 内部也有断言：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:1277`

```text
assert sliding_window == (-1, -1),
       "Cascade attention does not support sliding window."
```

所以即使上层 heuristic 漏掉，底层 cascade path 也会拒绝 sliding window。

---

## 8. chunked local attention 如何复用普通 backend

### 8.1 ChunkedLocalAttention 是 Attention 的包装层

`ChunkedLocalAttention` 继承自 `Attention`。

位置：`vllm/vllm/model_executor/layers/attention/chunked_local_attention.py:81`

初始化时它会先选择一个普通 underlying attention backend：

```text
underlying_attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype)
```

再通过 `create_chunked_local_attention_backend()` 创建一个子类 backend。

位置：`vllm/vllm/model_executor/layers/attention/chunked_local_attention.py:102`

### 8.2 它通过 metadata builder 改写 batch，而不是让 kernel 理解 chunk

关键逻辑在 `ChunkedLocalAttentionBuilder.build()`：

位置：`vllm/vllm/model_executor/layers/attention/chunked_local_attention.py:51`

```text
cm, make_virtual_batches_block_table = make_local_attention_virtual_batches(
    attention_chunk_size,
    common_attn_metadata,
    self.kv_cache_spec.block_size,
)
metadata = super().build(common_prefix_len, cm, fast_build)
```

也就是说：

```text
原始 batch
  → 按 attention_chunk_size 拆成多个 virtual batches
  → 每个 virtual batch 只拥有本 local chunk 的 block_table / seq_lens
  → 复用普通 FlashAttention/Triton metadata builder 和 kernel
```

### 8.3 make_local_attention_virtual_batches 做了什么

`make_local_attention_virtual_batches()` 会把一个请求按 chunk 切成多个“虚拟请求”。

位置：`vllm/vllm/v1/attention/backends/utils.py:251`

例如注释里的例子：

```text
attn_chunk_size = 4
query_start_loc = [0, 4, 14, 19]
q_seqlens = [4, 10, 5]

会得到：
q_seqlens_local    = [2, 2, 1, 4, 4, 1, 4, 1]
cu_seqlens_q_local = [0, 4, 6, 10, 14, 18, 19, 23, 24]
seqlens_k_local    = [4, 2, 4, 4, 4, 1, 4, 1]
block_table_local  = [virtual_batches, pages_per_local_batch]
```

构造结果是一个新的 `CommonAttentionMetadata`：

位置：`vllm/vllm/v1/attention/backends/utils.py:380`

```text
CommonAttentionMetadata(
    query_start_loc=local_query_start_loc,
    seq_lens=local_seq_lens,
    max_query_len=local_max_query_len,
    max_seq_len=local_max_seq_len,
    block_table_tensor=local_block_table,
    causal=True,
    ...
)
```

所以 chunked local attention 的核心技巧是：

```text
把“局部 attention”转换成“多个更短序列上的普通 causal attention”。
```

### 8.4 chunked local attention 禁用 CUDA graph attention capture

`ChunkedLocalAttentionBuilder.get_cudagraph_support()` 返回：

位置：`vllm/vllm/model_executor/layers/attention/chunked_local_attention.py:41`

```text
AttentionCGSupport.NEVER
```

原因是：

```text
local virtual batches 的数量和形状依赖当前 batch 中 query span / seq len，
不适合直接复用固定形状 cudagraph attention metadata。
```

---

## 9. prefix cache 与 sliding/local attention

### 9.1 sliding window 的 prefix cache 不是从左往右找最长连续前缀

普通 full attention 的 prefix cache 很自然：

```text
从 prompt 开头开始，连续命中多少 block，就能复用多少 block。
```

但 sliding window 的复用重点是“窗口附近的连续 blocks”。

`SlidingWindowManager.find_longest_cache_hit()` 会从右向左搜索，直到找到足够多的连续 blocks。

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:620`

其中需要的连续 block 数是：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:607`

```text
ceil((sliding_window - 1) / block_size)
```

如果启用 EAGLE，还会多要一个 block，用于后续 drop last matched block。

### 9.2 sliding window 的 cache hit 会用 null block 填充窗口外位置

`find_longest_cache_hit()` 初始化：

```text
computed_blocks = [null_block] * max_num_blocks
```

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:648`

然后只把窗口附近命中的 block 填进去。

这表示：

```text
窗口外 token 可以被标记为“已计算”，
但不需要真实 KV block。
```

这也是 sliding window 能节省 KV cache 的关键。

### 9.3 chunked local attention 的 prefix cache 以 chunk 边界为界

`ChunkedLocalAttentionManager.find_longest_cache_hit()` 的注释说明：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:813`

```text
窗口左侧完整 chunk 可以用 null blocks 标记为 computed；
窗口内 chunk 才需要查 prefix cache 是否命中。
```

例如：

```text
attention_chunk_size = 8
block_size = 4
max_length = 15
下一 token 在 position 15
窗口内是 token 8..14

返回可能是：
[null, null, hit_block_2]
```

### 9.4 prefix cache retention interval 只作用于 sliding window

环境变量 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 的注释说明：

位置：`vllm/vllm/envs.py:1051`

```text
Retain local sliding-window KV checkpoints for prefix caching.
Applies to sliding-window attention for now but not yet Mamba/linear attention.
```

`KVCacheCoordinator` 会校验：如果设置了该变量但模型没有 `SlidingWindowSpec`，就报错。

位置：`vllm/vllm/v1/core/kv_cache_coordinator.py:30`

这说明：

```text
prefix cache retention 是 sliding window 专用优化，
chunked local / Mamba / linear attention 当前不复用这个 retention 语义。
```

---

## 10. chunked prefill / spec decode / DCP / PCP 约束

### 10.1 sliding window 的内存上限考虑 max_num_batched_tokens

`SlidingWindowSpec.max_admission_blocks_per_request()` 会把 `max_num_batched_tokens` 纳入计算。

位置：`vllm/vllm/v1/kv_cache_interface.py:506`

这是为了 chunked prefill：

```text
一次 prefill 可能只调度长 prompt 的一个 chunk，
但这一 chunk 内的新 token 都要写 KV，
同时还要保留窗口左侧历史。
```

所以容量上限不是单纯 `sliding_window / block_size`，而是：

```text
sliding_window - 1 + max_num_batched_tokens
```

### 10.2 DCP / PCP 当前不支持 SlidingWindowManager / ChunkedLocalAttentionManager

`SlidingWindowSpec.max_memory_usage_bytes()` 中有断言：

位置：`vllm/vllm/v1/kv_cache_interface.py:528`

```text
DCP not support sliding window.
```

`SlidingWindowManager.find_longest_cache_hit()` 也断言：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:635`

```text
DCP not support sliding window attn now.
PCP not support sliding window attn now.
```

`ChunkedLocalAttentionManager.find_longest_cache_hit()` 也有类似断言：

位置：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:868`

```text
DCP not support chunked local attn now.
PCP not support chunked local attn now.
```

所以需要区分：

```text
backend forward 可能有 DCP 相关路径，
但 recycling-aware sliding/local KV cache manager 的 prefix-cache/manager 路径
当前显式限制 DCP/PCP。
```

### 10.3 spec decode 与窗口语义

spec decode 不改变 sliding window 的基本 mask 语义。

在 `GPUModelRunner._build_attention_metadata()` 中，如果启用 spec decode，会额外为某些 Mamba/GDN builder 注入 accepted token 信息；普通 attention metadata 仍然使用：

```text
query_start_loc
seq_lens
block_table
slot_mapping
```

窗口大小仍由 layer impl 的 `self.sliding_window` 决定。

---

## 11. backend metadata 和 kernel 参数如何对应

可以把 metadata / impl / kernel 的分工记成下面这张表：

| 层级 | 保存什么 | 是否包含 sliding_window |
|---|---|---|
| `CommonAttentionMetadata` | batch 动态信息：query_start_loc、seq_lens、block_table、slot_mapping | 通常不包含 |
| backend `AttentionMetadata` | backend 需要的 batch 派生信息，例如 scheduler_metadata、cascade prefix 信息 | 通常不直接保存窗口，部分 backend 可用调度 metadata 间接体现 |
| `AttentionImpl` | layer 静态属性：num_heads、scale、sliding_window、attn_type | 包含 |
| kernel 调用 | 真正执行 attention | 接收 `window_size` / `sliding_window` |

具体例子：

```text
Attention.__init__()
  → self.impl = FlashAttentionImpl(..., sliding_window=...)

GPUModelRunner._build_attention_metadata()
  → CommonAttentionMetadata(seq_lens, block_table, slot_mapping, ...)

FlashAttentionImpl.forward()
  → flash_attn_varlen_func(...,
        seqused_k=attn_metadata.seq_lens,
        block_table=attn_metadata.block_table,
        window_size=self.sliding_window)
```

---

## 12. 容易混淆的问题

### 12.1 sliding_window 是不是写在 attention metadata 里？

通常不是。

它主要保存在：

```text
Attention.self.sliding_window
backend impl self.sliding_window
KVCacheSpec.sliding_window
```

metadata 保存的是本轮 batch 的动态信息。

### 12.2 KV cache 是否仍然保存窗口外 token？

取决于 KV cache manager：

```text
SlidingWindowSpec + hybrid KV cache manager
  → 窗口外完整 blocks 可跳过 / 回收

FullAttentionSpec(sliding_window=...)
  → compute 仍按 sliding window，但 KV cache 按 full attention 保存

--disable-sliding-window
  → compute 和 KV cache 都按 full attention
```

### 12.3 disable_sliding_window 会发生什么？

`ModelConfig` 会把 `hf_text_config.sliding_window` 设置成 `None`。

位置：`vllm/vllm/config/model.py:716`

结果是：

```text
Attention.sliding_window = None
KVCacheSpec = FullAttentionSpec
backend window_size = (-1, -1)
```

即恢复 full causal attention。

### 12.4 interleaved sliding window 为什么不能用全局 CacheConfig.sliding_window？

因为全局值会覆盖 full attention 层。

所以 interleaved 模型必须通过：

```text
per_layer_sliding_window
```

让每层自己声明是否 sliding。

位置：`vllm/vllm/engine/arg_utils.py:1825`

### 12.5 chunked local attention 和 sliding window 是一回事吗？

不是。

```text
sliding window：窗口随 query 位置连续滑动；
chunked local：窗口按 chunk 边界切分，只看当前 chunk。
```

实现上也不同：

```text
sliding window：backend kernel 接收 window_size；
chunked local：metadata builder 把 batch 拆成 local virtual batches。
```

### 12.6 cascade attention 为什么不支持 sliding/local？

cascade attention 会把公共 prefix 拆成一个单独 kernel，并对 prefix 使用特殊的 bidirectional / no-mask 处理。

sliding/local attention 的可见范围不是简单的“公共 prefix + suffix causal”，所以当前实现直接禁用。

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:1195`

---

## 13. 最终可以记成一张表

| 主题 | Sliding Window Attention | Chunked Local Attention |
|---|---|---|
| 配置字段 | `sliding_window` | `attention_chunk_size` |
| 模型入口 | `Attention(per_layer_sliding_window=...)` 或 `CacheConfig.sliding_window` | `ChunkedLocalAttention(attention_chunk_size=...)` |
| KV spec | `SlidingWindowSpec` | `ChunkedLocalAttentionSpec` |
| fallback spec | `FullAttentionSpec(sliding_window=...)` | `FullAttentionSpec(attention_chunk_size=...)` |
| KV manager | `SlidingWindowManager` | `ChunkedLocalAttentionManager` |
| 跳过 token | `max(0, computed - window + 1)` | `floor(computed / chunk_size) * chunk_size` |
| backend 实现 | kernel `window_size=(window-1, 0)` | metadata 拆 virtual batches 后复用普通 backend |
| CUDA graph attention metadata | backend 依赖；capture 时 max_seq_len 特殊处理 | builder 显式 `NEVER` |
| cascade attention | 当前禁用 | 当前禁用 |
| prefix cache | 从窗口附近找连续 blocks，用 null block 表示窗口外 computed | chunk 左侧用 null block，chunk 内查 cache |

---

## 14. 主链路复盘

完整链路可以压缩成：

```text
ModelConfig
  → 读取 hf_text_config.sliding_window / attention_chunk_size
  → disable_sliding_window 时把 sliding_window 清空

Model layer
  → full 层：Attention(..., per_layer_sliding_window=None)
  → sliding 层：Attention(..., per_layer_sliding_window=config.sliding_window)
  → local 层：ChunkedLocalAttention(..., attention_chunk_size=config.attention_chunk_size)

Attention layer
  → self.sliding_window
  → backend impl self.sliding_window
  → get_kv_cache_spec()

KV cache planning
  → SlidingWindowSpec / ChunkedLocalAttentionSpec
  → SlidingWindowManager / ChunkedLocalAttentionManager
  → 跳过窗口外 blocks，降低真实 KV 占用
  → hybrid manager 禁用时退化为 FullAttentionSpec(...window...)

ModelRunner forward
  → _prepare_inputs()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → CommonAttentionMetadata(seq_lens, block_table, slot_mapping, ...)
  → set_forward_context(...)

Attention backend
  → FlashAttention/Triton 读取 attn_metadata 的动态 batch 信息
  → 从 impl 读取 self.sliding_window
  → kernel 执行局部可见 attention
```

---

## 15. 一句话总结

Sliding window / local attention 在 vLLM 里可以理解成“两层优化叠加”：

```text
计算层：通过 window_size / virtual local batches 限制每个 query 可见的 KV 范围；
缓存层：通过 SlidingWindowSpec / ChunkedLocalAttentionSpec 和对应 manager，
        让窗口外 blocks 可以被跳过、复用 null block 或回收。
```

如果只记住一句话：

```text
sliding_window 决定“attention 看哪里”，KVCacheSpec/Manager 决定“窗口外 KV 还要不要真实占着显存”；
chunked local attention 则把局部窗口拆成 virtual batches，复用普通 attention backend 完成计算。
```
