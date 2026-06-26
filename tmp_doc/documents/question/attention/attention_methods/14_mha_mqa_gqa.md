# 14. MHA / MQA / GQA：head 结构如何影响 attention 和 KV cache？

源码位置：

- `vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/vllm/config/model_arch.py`
- `vllm/vllm/config/model.py`
- `vllm/vllm/model_executor/models/llama.py`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/ops/triton_unified_attention.py`

本文用于梳理 MHA、MQA、GQA 的结构差异，以及这些差异如何影响 vLLM 中的 Q/K/V 投影、`num_heads` / `num_kv_heads`、KV cache shape、attention backend 参数和 Tensor Parallel 切分。

---

## 1. 本文要回答的问题

```text
MHA / MQA / GQA 的区别是什么？
HF config 里的 num_attention_heads / num_key_value_heads 如何进入 vLLM？
num_heads 和 num_kv_heads 在模型层、Attention 层、backend 中分别表示什么？
GQA / MQA 如何改变 KV cache 的 shape 和内存占用？
Paged KV cache 中为什么存 num_kv_heads，而不是 num_heads？
attention backend forward 如何接收 Q heads / KV heads？
Tensor Parallel 下 query heads 和 KV heads 如何切分或复制？
```

---

## 2. 一句话回答

MHA / MQA / GQA 的核心差异是：

```text
query heads 数量和 key/value heads 数量是否相等。
```

在 vLLM 中，这个差异会沿着下面这条链路传递：

```text
HF config
  → ModelArchitectureConfig.total_num_attention_heads / total_num_kv_heads
  → ModelConfig.get_num_attention_heads() / get_num_kv_heads()
  → 模型 attention 层的 q_size / kv_size
  → Attention(num_heads, num_kv_heads)
  → AttentionSpec(num_kv_heads, head_size)
  → backend.get_kv_cache_shape(..., num_kv_heads, head_size)
  → KV cache tensor shape
  → attention backend forward 中的 q heads / kv heads 比例
```

如果只记住一句话：

```text
MHA / MQA / GQA 不主要决定“用哪个 attention backend”，而是决定 Q heads 和 KV heads 的数量关系；这个关系直接影响 QKV 投影大小、KV cache 大小、backend kernel 的 head 映射，以及 TP 下 KV head 是切分还是复制。
```

---

## 3. MHA / MQA / GQA 的最小定义

### 3.1 MHA：Multi-Head Attention

MHA 中，query heads 和 key/value heads 一一对应。

```text
num_attention_heads == num_key_value_heads
```

例子：

```text
Q heads = 32
K heads = 32
V heads = 32

每个 Q head 使用自己的 K/V head。
```

此时 KV cache 中每层需要保存 32 组 K 和 32 组 V。

### 3.2 MQA：Multi-Query Attention

MQA 中，多个 query heads 共享同一组 K/V head。

```text
num_key_value_heads == 1
num_attention_heads > num_key_value_heads
```

例子：

```text
Q heads = 32
K heads = 1
V heads = 1

32 个 Q heads 共享 1 组 K/V。
```

此时 KV cache 中每层只保存 1 组 K 和 1 组 V，KV cache 显著变小。

### 3.3 GQA：Grouped-Query Attention

GQA 是 MHA 和 MQA 之间的折中。

```text
1 < num_key_value_heads < num_attention_heads
num_attention_heads % num_key_value_heads == 0
```

例子：

```text
Q heads = 32
K heads = 8
V heads = 8

每 4 个 Q heads 共享 1 组 K/V。
```

其中：

```text
num_queries_per_kv = num_attention_heads / num_key_value_heads
```

在上面的例子里：

```text
num_queries_per_kv = 32 / 8 = 4
```

---

## 4. 在 vLLM 中，字段名怎么对应

### 4.1 HF config 到 ModelArchitectureConfig

vLLM 会先把 HF config 转成运行时更稳定的 `ModelArchitectureConfig`。

字段定义：

```python
total_num_attention_heads: int
head_size: int
total_num_kv_heads: int
```

位置：`vllm/vllm/config/model_arch.py:35` 到 `vllm/vllm/config/model_arch.py:45`

其中：

```text
total_num_attention_heads：模型全局 Q heads 数；
total_num_kv_heads：模型全局 K/V heads 数；
head_size：每个 head 的维度。
```

转换入口在 `ModelArchConfigConvertorBase.convert()`：

```python
total_num_attention_heads=self.get_total_num_attention_heads()
head_size=self.get_head_size()
total_num_kv_heads=self.get_total_num_kv_heads()
```

位置：`vllm/vllm/transformers_utils/model_arch_config_convertor.py:345` 到 `vllm/vllm/transformers_utils/model_arch_config_convertor.py:360`

### 4.2 num_attention_heads 的来源

默认实现：

```python
def get_total_num_attention_heads(self) -> int:
    return getattr(self.hf_text_config, "num_attention_heads", 0)
```

位置：`vllm/vllm/transformers_utils/model_arch_config_convertor.py:39` 到 `vllm/vllm/transformers_utils/model_arch_config_convertor.py:40`

也就是说，大部分模型的 Q heads 来自 HF config 的：

```text
num_attention_heads
```

### 4.3 num_key_value_heads 的来源

默认实现会依次尝试多个常见字段：

```python
attributes = [
    "n_head_kv",
    "num_kv_heads",
    "num_key_value_heads",
    "multi_query_group_num",
    "num_attention_groups",
]
```

位置：`vllm/vllm/transformers_utils/model_arch_config_convertor.py:106` 到 `vllm/vllm/transformers_utils/model_arch_config_convertor.py:123`

如果这些字段都不存在，则默认：

```text
total_num_kv_heads = total_num_attention_heads
```

这正好对应普通 MHA。

因此可以这样理解：

```text
HF config 没有显式 KV head 字段：通常按 MHA 处理；
HF config 有 num_key_value_heads / n_head_kv / num_kv_heads：按 GQA 或 MQA 处理；
num_key_value_heads == 1：MQA；
1 < num_key_value_heads < num_attention_heads：GQA。
```

### 4.4 Falcon 的 multi_query 特殊处理

Falcon 有一个特殊逻辑：如果不是 `new_decoder_architecture`，并且 `multi_query=True`，则 KV heads 直接是 1。

```python
if not new_decoder_arch_falcon and getattr(self.hf_text_config, "multi_query", False):
    return 1
```

位置：`vllm/vllm/transformers_utils/model_arch_config_convertor.py:420` 到 `vllm/vllm/transformers_utils/model_arch_config_convertor.py:436`

这就是典型 MQA 配置进入 vLLM 的方式之一。

---

## 5. ModelConfig 如何给 TP 后的本地 rank 返回 head 数

vLLM 区分两套数量：

```text
total_num_attention_heads / total_num_kv_heads：模型全局数量；
get_num_attention_heads() / get_num_kv_heads()：当前 TP rank 本地数量。
```

### 5.1 Query heads 必须能被 TP 整除

`verify_with_parallel_config()` 会检查：

```python
if total_num_attention_heads % tensor_parallel_size != 0:
    raise ValueError(...)
```

位置：`vllm/vllm/config/model.py:1159` 到 `vllm/vllm/config/model.py:1170`

也就是说：

```text
Q heads 按 TP 均匀切分。
```

本地 Q heads 计算：

```python
def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
    num_heads = self.model_arch_config.total_num_attention_heads
    return num_heads // parallel_config.tensor_parallel_size
```

位置：`vllm/vllm/config/model.py:1272` 到 `vllm/vllm/config/model.py:1274`

### 5.2 KV heads 可能切分，也可能复制

本地 KV heads 计算：

```python
def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
    if self.use_mla:
        return 1

    total_num_kv_heads = self.get_total_num_kv_heads()
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

位置：`vllm/vllm/config/model.py:1259` 到 `vllm/vllm/config/model.py:1270`

关键点是：

```text
如果 total_num_kv_heads >= TP size：KV heads 按 TP 切分；
如果 total_num_kv_heads < TP size：每个 TP rank 至少保留 1 个 KV head，相当于复制 KV heads。
```

这和模型层里的检查一致。

以 Llama attention 为例：

```python
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

位置：`vllm/vllm/model_executor/models/llama.py:143` 到 `vllm/vllm/model_executor/models/llama.py:156`

---

## 6. 模型 attention 层如何使用 num_heads / num_kv_heads

以 Llama 为例。

### 6.1 初始化时先算本地 Q/KV head 数

`LlamaAttention.__init__()` 中：

```python
self.total_num_heads = num_heads
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = num_kv_heads
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

位置：`vllm/vllm/model_executor/models/llama.py:125` 到 `vllm/vllm/model_executor/models/llama.py:157`

然后计算：

```python
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
```

位置：`vllm/vllm/model_executor/models/llama.py:158` 到 `vllm/vllm/model_executor/models/llama.py:162`

这说明：

```text
Q projection 输出大小由本地 query heads 决定；
K/V projection 输出大小由本地 kv heads 决定。
```

### 6.2 QKVParallelLinear 接收全局 head 数

Llama 使用 `QKVParallelLinear`：

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=self.total_num_heads,
    total_num_kv_heads=self.total_num_kv_heads,
    ...
)
```

位置：`vllm/vllm/model_executor/models/llama.py:165` 到 `vllm/vllm/model_executor/models/llama.py:172`

`QKVParallelLinear` 的注释直接说明它支持 MQA / GQA：

```text
When the number of key/value heads is smaller than the number of query
heads (e.g., multi-query/grouped-query attention), the key/value head may
be replicated while the query heads are partitioned.
```

位置：`vllm/vllm/model_executor/layers/linear.py:914` 到 `vllm/vllm/model_executor/layers/linear.py:923`

### 6.3 QKVParallelLinear 内部怎么切

核心逻辑：

```python
self.num_heads = divide(self.total_num_heads, tp_size)
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

位置：`vllm/vllm/model_executor/layers/linear.py:942` 到 `vllm/vllm/model_executor/layers/linear.py:973`

含义是：

```text
Q heads 总是按 TP 切；
KV heads 能切就切；
KV heads 数少于 TP size 时，每个 rank 保留 1 个 KV head，并通过 num_kv_head_replicas 处理权重加载复制。
```

### 6.4 forward 时按 q_size / kv_size 切开

Llama forward 中：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
```

位置：`vllm/vllm/model_executor/models/llama.py:224` 到 `vllm/vllm/model_executor/models/llama.py:233`

所以 GQA / MQA 的差异在进入 Attention 前就已经体现在：

```text
q 的最后一维 = num_heads * head_dim
k 的最后一维 = num_kv_heads * head_dim
v 的最后一维 = num_kv_heads * head_dim
```

---

## 7. Attention 层如何保存 num_heads / num_kv_heads

### 7.1 Attention 构造函数

`Attention` 的定位写在类注释里：

```text
1. Store the input key and value tensors in the KV cache.
2. Perform (multi-head/multi-query/grouped-query) attention.
3. Return the output tensor.
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:192` 到 `vllm/vllm/model_executor/layers/attention/attention.py:202`

构造参数包含：

```python
num_heads: int
head_size: int
num_kv_heads: int | None = None
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:204` 到 `vllm/vllm/model_executor/layers/attention/attention.py:221`

### 7.2 没传 num_kv_heads 时默认就是 MHA

```python
if num_kv_heads is None:
    num_kv_heads = num_heads
assert num_heads % num_kv_heads == 0
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:296` 到 `vllm/vllm/model_executor/layers/attention/attention.py:300`

这说明：

```text
num_kv_heads 默认等于 num_heads；
也就是默认按 MHA 处理；
GQA / MQA 必须满足 num_heads 能整除 num_kv_heads。
```

### 7.3 Attention 保存本地 head 数

```python
self.num_heads = num_heads
self.head_size = head_size
self.head_size_v = self.head_size if head_size_v is None else head_size_v
self.num_kv_heads = num_kv_heads
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:304` 到 `vllm/vllm/model_executor/layers/attention/attention.py:308`

注意这里保存的是：

```text
当前 TP rank 的本地 num_heads / num_kv_heads。
```

### 7.4 forward 前 reshape 成 head 维度

Attention forward 中：

```python
query = query.view(-1, self.num_heads, self.head_size)
output = output.view(-1, self.num_heads, self.head_size_v)
if key is not None:
    key = key.view(-1, self.num_kv_heads, self.head_size)
if value is not None:
    value = value.view(-1, self.num_kv_heads, self.head_size_v)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:488` 到 `vllm/vllm/model_executor/layers/attention/attention.py:503`

所以 backend 看到的张量已经是：

```text
query: [num_tokens, num_heads, head_size]
key:   [num_tokens, num_kv_heads, head_size]
value: [num_tokens, num_kv_heads, head_size_v]
```

这就是 MHA / MQA / GQA 在 attention backend 的直接表现。

---

## 8. KV cache spec 为什么只记录 num_kv_heads

### 8.1 AttentionSpec 的字段

KV cache 的 attention spec 定义为：

```python
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
```

位置：`vllm/vllm/v1/kv_cache_interface.py:159` 到 `vllm/vllm/v1/kv_cache_interface.py:167`

注意这里没有 `num_heads`。

原因很直接：

```text
KV cache 存的是 K 和 V，不存 Q；
所以 KV cache 大小只和 num_kv_heads 有关，不和 num_heads 直接相关。
```

### 8.2 page_size_bytes 的公式

普通 attention KV cache 的真实 page size：

```python
return (
    2
    * self.block_size
    * self.num_kv_heads
    * self.head_size
    * get_dtype_size(self.dtype)
)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:183` 到 `vllm/vllm/v1/kv_cache_interface.py:201`

公式拆开是：

```text
2：K + V
block_size：一个 KV block 中的 token 数
num_kv_heads：每个 token 存多少组 K/V heads
head_size：每个 head 的维度
dtype_size：每个元素占多少字节
```

因此：

```text
MHA：num_kv_heads = num_heads，KV cache 最大；
GQA：num_kv_heads < num_heads，KV cache 按比例变小；
MQA：num_kv_heads = 1，KV cache 最小。
```

### 8.3 Attention.get_kv_cache_spec() 写入 num_kv_heads

`Attention.get_kv_cache_spec()` 会返回 `FullAttentionSpec` 或 `SlidingWindowSpec`。

普通 full attention 路径：

```python
return FullAttentionSpec(
    block_size=block_size,
    num_kv_heads=self.num_kv_heads,
    head_size=self.head_size,
    head_size_v=self.head_size_v,
    dtype=self.kv_cache_torch_dtype,
    kv_quant_mode=quant_mode,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:581` 到 `vllm/vllm/model_executor/layers/attention/attention.py:632`

也就是说：

```text
KV cache spec 从 Attention 层读取本地 num_kv_heads。
```

---

## 9. KV cache tensor shape 如何受 GQA / MQA 影响

### 9.1 backend 统一暴露 get_kv_cache_shape

`AttentionBackend` 要求每个 backend 实现：

```python
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
```

位置：`vllm/vllm/v1/attention/backend.py:88` 到 `vllm/vllm/v1/attention/backend.py:96`

这说明 backend 层的 KV cache shape 入口也只接收：

```text
num_kv_heads
```

而不是 `num_heads`。

### 9.2 FlashAttention backend 的 shape

FlashAttention backend：

```python
return (num_blocks, 2, block_size, num_kv_heads, head_size)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:140` 到 `vllm/vllm/v1/attention/backends/flash_attn.py:149`

逻辑维度可以理解为：

```text
[num_blocks, K/V, block_size, num_kv_heads, head_size]
```

### 9.3 Triton backend 的 shape

Triton backend 普通路径也是：

```python
return (num_blocks, 2, block_size, num_kv_heads, head_size)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:293` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:315`

某些 KV cache 量化模式会在最后一维额外 padding scale，但 `num_kv_heads` 仍然是 shape 中的 head 维。

### 9.4 ModelRunner 初始化 KV cache tensor 时传入 num_kv_heads

`GPUModelRunner` 初始化 KV cache tensor 时：

```python
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks,
    shape_block_size,
    kv_cache_spec.num_kv_heads,
    kv_cache_spec.head_size,
    cache_dtype_str=self.cache_config.cache_dtype,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:7129` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:7135`

这一步把：

```text
AttentionSpec.num_kv_heads
```

真正落成了：

```text
KV cache tensor 的 head 维度。
```

---

## 10. 一个具体 KV cache 大小对比

假设：

```text
block_size = 16
head_size = 128
dtype = fp16 / bf16 = 2 bytes
num_attention_heads = 32
```

每个 block、每层的 KV cache 大小：

### 10.1 MHA

```text
num_kv_heads = 32
page_size = 2 * 16 * 32 * 128 * 2 bytes
          = 262144 bytes
          = 256 KiB
```

### 10.2 GQA

```text
num_kv_heads = 8
page_size = 2 * 16 * 8 * 128 * 2 bytes
          = 65536 bytes
          = 64 KiB
```

相比 MHA：

```text
KV cache 变成 1/4。
```

### 10.3 MQA

```text
num_kv_heads = 1
page_size = 2 * 16 * 1 * 128 * 2 bytes
          = 8192 bytes
          = 8 KiB
```

相比 MHA：

```text
KV cache 变成 1/32。
```

所以 GQA / MQA 对推理服务非常重要：

```text
它不会减少 Q heads 的计算规模到同等比例，
但会显著减少每层每 token 需要保存和读取的 K/V cache。
```

---

## 11. attention metadata 中和 head 结构的关系

### 11.1 CommonAttentionMetadata 不直接存 num_heads

`CommonAttentionMetadata` 主要描述 batch / KV block / slot mapping：

```python
query_start_loc
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
```

位置：`vllm/vllm/v1/attention/backend.py:393` 到 `vllm/vllm/v1/attention/backend.py:421`

它不直接存 `num_heads` 或 `num_kv_heads`。

原因是：

```text
metadata 主要描述“哪些 token 看哪些 KV block、当前 token 写哪里”；
head 数量通常来自 query/key/value tensor shape、ModelConfig 或 KVCacheSpec。
```

### 11.2 backend builder 会读取本地 Q/KV head 数

FlashAttention metadata builder 初始化时读取：

```python
self.num_heads_q = self.model_config.get_num_attention_heads(self.parallel_config)
self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:342` 到 `vllm/vllm/v1/attention/backends/flash_attn.py:345`

Triton metadata builder 也类似：

```python
self.num_heads_q = model_config.get_num_attention_heads(vllm_config.parallel_config)
self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:112` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:117`

这说明：

```text
metadata builder 知道本地 Q heads / KV heads，
但真正的 KV block 映射还是来自 block_table / slot_mapping。
```

---

## 12. backend forward 如何消费 Q heads / KV heads

### 12.1 TritonAttentionImpl 的 forward 形状注释

Triton backend forward 的注释非常直接：

```text
query: shape = [num_tokens, num_heads, head_size]
key: shape = [num_tokens, num_kv_heads, head_size]
value: shape = [num_tokens, num_kv_heads, head_size]
kv_cache: shape = [num_blocks, 2, block_size, num_kv_heads, head_size]
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:530` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:553`

这就是 vLLM 中 MHA / MQA / GQA 的核心 runtime 形态。

### 12.2 KV cache update 只写 K/V heads

Triton backend 的 KV cache update：

```python
key_cache, value_cache = kv_cache.unbind(1)
triton_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    self.kv_cache_dtype,
    layer._k_scale,
    layer._v_scale,
)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:724` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:767`

因为 `key` / `value` 的 shape 已经是：

```text
[num_tokens, num_kv_heads, head_size]
```

所以写入 KV cache 的 head 维自然就是 `num_kv_heads`。

### 12.3 attention kernel 中计算 num_queries_per_kv

Triton unified attention 中：

```python
num_query_heads = q.shape[1]
num_kv_heads = k.shape[2]
num_queries_per_kv = num_query_heads // num_kv_heads
```

位置：`vllm/vllm/v1/attention/ops/triton_unified_attention.py:853` 到 `vllm/vllm/v1/attention/ops/triton_unified_attention.py:858`

这个 `num_queries_per_kv` 就是 GQA / MQA 的关键参数。

```text
MHA：num_queries_per_kv = 1
GQA：num_queries_per_kv > 1，例如 4
MQA：num_queries_per_kv = num_query_heads
```

kernel 后续会按 KV head 分组处理多个 Q heads。

---

## 13. MHA / MQA / GQA 对 backend 选择的影响

严格来说：

```text
MHA / MQA / GQA 不是 backend 类型。
```

backend 选择主要看：

```text
head_size
dtype
kv_cache_dtype
block_size
是否 MLA
是否 sliding window / sink / sparse / mm prefix
设备能力
attention type
```

比如 `Attention` 初始化时通过 `get_attn_backend(...)` 选择 backend，传入的是 `head_size`、`dtype`、`kv_cache_dtype`、`use_mla`、`has_sink`、`attn_type` 等。

位置：`vllm/vllm/model_executor/layers/attention/attention.py:318` 到 `vllm/vllm/model_executor/layers/attention/attention.py:328`

但 backend 必须能处理：

```text
query heads != kv heads
```

否则就无法支持 GQA / MQA。

在当前这些常见路径里，backend 通过：

```text
query tensor shape
key/value tensor shape
kv cache shape
num_queries_per_kv
```

来表达和处理这种结构差异。

---

## 14. Tensor Parallel 下的几个典型例子

假设：

```text
TP size = 4
head_size = 128
```

### 14.1 MHA：32 Q heads / 32 KV heads

```text
全局：
  total_num_attention_heads = 32
  total_num_kv_heads = 32

每个 TP rank：
  num_heads = 32 / 4 = 8
  num_kv_heads = 32 / 4 = 8

本地：
  q: [tokens, 8, 128]
  k: [tokens, 8, 128]
  v: [tokens, 8, 128]
  kv_cache: [blocks, 2, block_size, 8, 128]
```

### 14.2 GQA：32 Q heads / 8 KV heads

```text
全局：
  total_num_attention_heads = 32
  total_num_kv_heads = 8

每个 TP rank：
  num_heads = 32 / 4 = 8
  num_kv_heads = 8 / 4 = 2

本地：
  q: [tokens, 8, 128]
  k: [tokens, 2, 128]
  v: [tokens, 2, 128]
  kv_cache: [blocks, 2, block_size, 2, 128]

本地 num_queries_per_kv = 8 / 2 = 4
```

这表示每个 rank 内部仍然是 4 个 Q heads 共享 1 个 KV head。

### 14.3 MQA：32 Q heads / 1 KV head

```text
全局：
  total_num_attention_heads = 32
  total_num_kv_heads = 1

每个 TP rank：
  num_heads = 32 / 4 = 8
  num_kv_heads = max(1, 1 / 4) = 1

本地：
  q: [tokens, 8, 128]
  k: [tokens, 1, 128]
  v: [tokens, 1, 128]
  kv_cache: [blocks, 2, block_size, 1, 128]

本地 num_queries_per_kv = 8 / 1 = 8
```

这里 KV head 会在 TP ranks 间复制，而不是切成小于 1 的 head。

---

## 15. Decode Context Parallel 对 GQA / MQA 的额外约束

vLLM 对 decode context parallel 在 GQA / MQA 下有额外检查。

当 `decode_context_parallel_size > 1` 且不是 MLA 时：

```python
assert tensor_parallel_size > total_num_kv_heads
max_dcp_size = tensor_parallel_size // total_num_kv_heads
assert decode_context_parallel_size <= max_dcp_size
num_q_per_kv = total_num_attention_heads // total_num_kv_heads
assert num_q_per_kv % decode_context_parallel_size == 0
```

位置：`vllm/vllm/config/model.py:1184` 到 `vllm/vllm/config/model.py:1207`

这说明 DCP 对 GQA / MQA 不是随便开的：

```text
TP size 必须大于 total_num_kv_heads；
DCP size 不能超过 TP / KV heads 的比例；
每个 KV head 对应的 Q heads 数必须能被 DCP size 整除。
```

直观理解：

```text
GQA / MQA 已经把多个 Q heads 分组到少数 KV heads 上；
DCP 继续拆 decode context 时，必须保持 Q/KV head 分组关系可整除。
```

---

## 16. 和 block table / slot mapping 的关系

MHA / MQA / GQA 改变的是：

```text
每个 token 的 K/V head 数量。
```

它不改变：

```text
一个请求有哪些 KV blocks；
某个 token 写到哪个 slot；
block table 的 request → block ids 语义；
slot mapping 的 token → KV slot 语义。
```

也就是说：

```text
block table / slot mapping 管“token 放在哪个 block/slot”；
num_kv_heads 管“每个 slot 里存多少组 K/V head”。
```

组合起来才是完整 KV cache 访问：

```text
slot_mapping 定位 token slot；
num_kv_heads 定位该 slot 内的 KV head 维；
head_size 定位该 head 内的向量维。
```

因此 GQA 对 paged KV cache 的影响不是改变 page/block 调度逻辑，而是改变每个 page 的大小和 backend 读取 head 的方式。

---

## 17. 容易疑惑的点

### 17.1 num_heads 是不是 KV cache 里的 head 数？

不是。

```text
num_heads：Q heads 数，用于 query / output；
num_kv_heads：K/V heads 数，用于 key / value / KV cache。
```

KV cache 存 K/V，所以看 `num_kv_heads`。

### 17.2 GQA 会不会让 Q 投影也变小？

不会按 KV head 的比例变小。

```text
Q projection 仍然输出 num_heads * head_size；
K/V projection 输出 num_kv_heads * head_size。
```

所以 GQA 主要减少的是：

```text
K/V projection 输出大小；
KV cache 写入大小；
decode 阶段读取 KV cache 的带宽和容量压力。
```

### 17.3 MQA 是不是 num_heads = 1？

不是。

MQA 是：

```text
num_kv_heads = 1
```

不是：

```text
num_heads = 1
```

Query heads 仍然可以很多。

### 17.4 GQA 是不是一种 attention backend？

不是。

GQA 是模型结构。

backend 可以是 FlashAttention、Triton、FlashInfer 等；只要它能处理：

```text
num_heads > num_kv_heads
```

就可以支持 GQA / MQA。

### 17.5 TP 下 KV heads 少于 TP size 怎么办？

KV heads 会复制。

典型 MQA：

```text
total_num_kv_heads = 1
TP size = 4
每个 rank 本地 num_kv_heads = 1
```

这不是把 1 个 KV head 切成 1/4，而是每个 rank 都保留一份 KV head。

### 17.6 KV cache shape 中的 `2` 是什么？

通常表示：

```text
2 = K + V
```

例如 FlashAttention / Triton 常见 shape：

```text
[num_blocks, 2, block_size, num_kv_heads, head_size]
```

---

## 18. 最终可以记成一张表

| 结构 | Q heads | KV heads | num_queries_per_kv | KV cache 大小 | 典型含义 |
|---|---:|---:|---:|---:|---|
| MHA | H | H | 1 | 最大 | 每个 Q head 有自己的 K/V |
| GQA | H | G | H/G | 中等 | 每组 Q heads 共享 1 个 K/V head |
| MQA | H | 1 | H | 最小 | 所有 Q heads 共享 1 个 K/V head |

在 vLLM 中对应：

| 层次 | 字段 / 对象 | 作用 |
|---|---|---|
| HF config | `num_attention_heads` | 全局 Q heads |
| HF config | `num_key_value_heads` / `n_head_kv` / `num_kv_heads` | 全局 KV heads |
| `ModelArchitectureConfig` | `total_num_attention_heads` | 保存全局 Q heads |
| `ModelArchitectureConfig` | `total_num_kv_heads` | 保存全局 KV heads |
| `ModelConfig` | `get_num_attention_heads()` | 当前 TP rank 的 Q heads |
| `ModelConfig` | `get_num_kv_heads()` | 当前 TP rank 的 KV heads |
| 模型 attention 层 | `q_size` / `kv_size` | 切分 Q/K/V 投影输出 |
| `Attention` | `self.num_heads` / `self.num_kv_heads` | reshape Q/K/V 并传给 backend |
| `AttentionSpec` | `num_kv_heads` | 描述 KV cache 每 token 的 KV head 数 |
| backend | `get_kv_cache_shape()` | 生成 KV cache tensor shape |
| kernel | `num_queries_per_kv` | 决定几个 Q heads 共享一个 KV head |

---

## 19. 总结

MHA / MQA / GQA 在 vLLM 里的主线可以压缩成：

```text
HF config 声明 Q heads / KV heads
  → vLLM 转成 total_num_attention_heads / total_num_kv_heads
  → TP 后得到本地 num_heads / num_kv_heads
  → QKVParallelLinear 生成不同大小的 q / k / v
  → Attention reshape 成 [tokens, heads, head_size]
  → AttentionSpec 只记录 num_kv_heads
  → backend 用 num_kv_heads 构造 KV cache shape
  → kernel 用 num_queries_per_kv 建立 Q head 到 KV head 的分组关系
```

如果只记住一句话：

```text
num_heads 决定 Q 和输出有多少 head；num_kv_heads 决定 K/V 和 KV cache 有多少 head；MHA/MQA/GQA 的差异就是这两个数的关系。
```

对 paged KV cache 来说，最关键的是：

```text
GQA / MQA 不改变 block table 和 slot mapping 的语义，
但会把每个 KV page 的 head 维从 num_heads 缩小到 num_kv_heads，
从而直接降低 KV cache 内存占用和 decode 阶段的 KV 读写压力。
```
