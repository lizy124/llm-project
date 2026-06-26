# 07. KV cache layout 和 Attention backend 的关系

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\kv_cache_interface.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\kv_cache_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\flash_attn.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\flashinfer.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\kv_connector_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\mla_attention.py`

本文用于梳理 KV cache layout、cache dtype、page size、block size、kernel block size、stride order、KV cache group、KV connector cross-layer layout 与 attention backend 的关系。前一篇 `05_slot_mapping_and_block_table.md` 重点讲“token 如何映射到 KV slot”，本文重点讲“KV cache tensor 本身长什么样、由谁决定、如何绑定到 attention layer、backend 如何解释这块内存”。

---

## 1. 本文要回答的问题

```text
KVCacheSpec 描述什么？
KVCacheConfig / KVCacheTensor / KVCacheGroupSpec 分别描述什么？
AttentionBackend 如何定义 KV cache shape？
get_kv_cache_stride_order() 和 NHD / HND layout 有什么意义？
kernel block size 和 scheduler block size 为什么可能不同？
GPUModelRunner 如何分配、reshape、bind KV cache tensor？
普通 attention、MLA、Mamba 的 KV cache layout 有什么差异？
KV connector 为什么偏好 cross-layer blocks？
KV sharing fast prefill 和 cross-layer KV sharing 如何影响 KV cache groups？
backend layout 如何影响 slot mapping、metadata 和 forward？
```

---

## 2. 一句话回答

KV cache layout 是 vLLM 中 Scheduler、KV cache manager、ModelRunner、AttentionBackend 和 attention kernel 之间的内存契约。

最小关系是：

```text
KVCacheSpec
  描述每一层 KV cache 的逻辑容量、block_size、page_size_bytes、dtype、head 维度。

KVCacheConfig
  描述每个 worker 实际要分配多少 blocks、多少 raw tensor、哪些 layer 共享 tensor。

AttentionBackend
  通过 get_kv_cache_shape() / get_kv_cache_stride_order() 描述 kernel 希望怎样看这块内存。

GPUModelRunner
  先按 KVCacheTensor.size 分配 int8 raw buffer，
  再按 backend shape / stride reshape 成 per-layer kv_cache，
  最后 bind 到 Attention layer 和 runner.kv_caches。

Attention.forward / backend impl
  forward 时拿到 kv_cache + attn_metadata + slot_mapping，
  按 backend-specific layout 读写 K/V。
```

如果只记住一句话：

```text
Scheduler 分配的是 KV block 编号，backend 决定的是 KV tensor 形状和物理 stride，ModelRunner 负责把同一块 raw memory 变成 backend 能读写的 kv_cache view。
```

---

## 3. 先区分三种“layout”

讨论 KV cache layout 时，容易把三件事混在一起。

### 3.1 逻辑 block layout

这是 Scheduler / KVCacheManager 看到的布局：

```text
request
  → logical blocks
  → physical block ids
```

它关注：

```text
这个请求有哪些 blocks；
哪些 block 是 prefix cache 命中的；
哪些 block 是新分配的；
block table 如何保存 request -> block ids。
```

这部分在 `05_slot_mapping_and_block_table.md` 已经展开。

### 3.2 token slot layout

这是 ModelRunner / attention metadata 看到的 token 级布局：

```text
position
  → block_id
  → block_offset
  → slot_id
```

典型公式：

```text
slot_id = block_id * block_size + block_offset
```

它关注：

```text
本轮第 i 个 token 的 K/V 应该写到哪个 KV slot。
```

### 3.3 tensor memory layout

这是本文重点。

它关注：

```text
kv_cache tensor 的 shape 是什么；
K 和 V 维度放在哪里；
block 维度放在哪里；
head 维度和 block_size 维度谁更连续；
是否有 num_layers 维度；
MLA 是否是 compressed cache；
Mamba 是否根本不是 K/V pair，而是 state tensors。
```

典型标准 attention backend 可能看到：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

但物理 stride 可能被改成：

```text
NHD：num_blocks, K/V, block_size, num_heads, head_size
HND：num_blocks, K/V, num_heads, block_size, head_size
```

所以：

```text
block table / slot mapping 解决“访问哪一页、哪一个 token slot”；
KV cache layout 解决“这一页在 tensor 内部如何排布”。
```

---

## 4. KVCacheSpec：每层 KV cache 的格式说明

`KVCacheSpec` 定义在：`code/vllm/vllm/v1/kv_cache_interface.py:95`

它是每一层 KV cache 格式的基类。

核心字段：

```text
block_size
  一个 KV block 容纳多少 token。
```

核心属性：

```text
page_size_bytes
  一个 block / page 占多少字节。

storage_block_size
  真实存储维度上的 block size，默认等于 block_size。

max_memory_usage_bytes(...)
  该层在最大上下文下最多需要多少 KV cache memory。
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:95` 到 `code/vllm/vllm/v1/kv_cache_interface.py:131`

### 4.1 AttentionSpec

普通 attention 的基类是 `AttentionSpec`。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:159`

关键字段：

```text
num_kv_heads
head_size
dtype
kv_quant_mode
page_size_padded
```

普通情况下，真实 page size 是：

```text
real_page_size_bytes
  = 2 * block_size * num_kv_heads * head_size * dtype_size
```

这里的 `2` 表示：

```text
K cache + V cache
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:183` 到 `code/vllm/vllm/v1/kv_cache_interface.py:200`

如果是 per-token-head quant scale、NVFP4 或 page padding，`page_size_bytes` 会在真实 K/V 数据之外额外预算 scale / padding 空间。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:167` 到 `code/vllm/vllm/v1/kv_cache_interface.py:180`

### 4.2 FullAttentionSpec

`FullAttentionSpec` 代表普通 full attention KV cache。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:203`

它会记录：

```text
head_size_v
sliding_window
attention_chunk_size
non_causal
```

其中 `sliding_window` 在某些配置下只是语义信息：

```text
KV cache manager 仍按 full attention 分配 blocks；
model runner / backend 计算时再按 sliding window 限制可见范围。
```

### 4.3 SlidingWindowSpec

`SlidingWindowSpec` 表示真正按 sliding window 管理 KV cache 的层。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:471`

它的内存上界不是简单 `max_model_len`，而是按：

```text
sliding_window - 1 + max_num_batched_tokens
```

估算最多要持有的 window blocks。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:500` 到 `code/vllm/vllm/v1/kv_cache_interface.py:531`

### 4.4 MLAAttentionSpec

MLA 的 KV cache 不再是标准 K/V pair。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:364`

关键字段：

```text
cache_dtype_str
alignment
compress_ratio
model_version
```

最重要的是：

```text
storage_block_size = block_size // compress_ratio
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:377` 到 `code/vllm/vllm/v1/kv_cache_interface.py:379`

普通 MLA page size：

```text
storage_block_size * num_kv_heads * head_size * dtype_size
```

特殊 `fp8_ds_mla` 下，page size 可能按模型版本直接使用定制字节数，例如 DeepSeekV4 每 token 584B。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:382` 到 `code/vllm/vllm/v1/kv_cache_interface.py:396`

这说明：

```text
MLA 的 block_size 是调度 / 逻辑 token 粒度；
storage_block_size 是实际压缩 cache 存储粒度；
page_size_bytes 反映真实物理字节数。
```

### 4.5 MambaSpec

Mamba / SSM 不存传统 K/V pair，而是存 state tensors。

虽然本文不展开 MambaSpec 的全部字段，但要记住：

```text
MambaSpec 也走 KV cache config / allocation 管线；
但 reshape 时不是 get_kv_cache_shape()，而是按 shapes / dtypes 拆成多个 state tensor view。
```

对应实现见 `GPUModelRunner._reshape_kv_cache_tensors()` 的 Mamba 分支。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7154` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7177`

---

## 5. KVCacheGroupSpec 和 KVCacheConfig 描述什么

### 5.1 KVCacheGroupSpec

定义位置：`code/vllm/vllm/v1/kv_cache_interface.py:852`

它表示：

```text
一组共享同一个 block table 的 model layers。
```

字段：

```text
layer_names
  这个 group 包含哪些 layer。

kv_cache_spec
  这个 group 的 KV cache 格式。

is_eagle_group
  是否包含 EAGLE / MTP draft attention layers。
```

这说明：

```text
KV cache group 是 scheduler / block table 视角的 grouping；
attention group 是 backend / metadata builder 视角的 grouping。
```

两者相关，但不是同一件事。

### 5.2 KVCacheTensor

定义位置：`code/vllm/vllm/v1/kv_cache_interface.py:842`

字段：

```text
size
  这个 raw tensor 要分配多少字节。

shared_by
  哪些 layer name 共享这块 raw tensor。
```

注意这里的 tensor 还不是最终形状：

```text
它只是一段 int8 raw buffer；
后面会被 ModelRunner view 成 backend 需要的 dtype / shape / stride。
```

### 5.3 KVCacheConfig

定义位置：`code/vllm/vllm/v1/kv_cache_interface.py:867`

字段：

```text
num_blocks
  KV cache blocks 数量。

kv_cache_tensors
  worker 应该分配哪些 raw KV cache tensor。

kv_cache_groups
  模型有哪些 KV cache groups。
```

源码注释强调：

```text
只有一种 attention 的模型通常只有一个 group；
hybrid 模型可能有多个 group。
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:873` 到 `code/vllm/vllm/v1/kv_cache_interface.py:883`

### 5.4 needs_kv_cache_zeroing

`KVCacheConfig.needs_kv_cache_zeroing` 当前主要由 Mamba 决定：

```text
has_mamba_layers → needs_kv_cache_zeroing
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:886` 到 `code/vllm/vllm/v1/kv_cache_interface.py:892`

原因是 stateful cache 对脏数据更敏感。

---

## 6. KVCacheConfig 是如何生成的

入口：`code/vllm/vllm/v1/core/kv_cache_utils.py:1937`

`get_kv_cache_configs()` 的核心目标是：

```text
给每个 worker 生成一致可用的 KVCacheConfig。
```

源码注释把过程拆成五步：

```text
1. 合并所有 worker 的 KVCacheSpec；
2. 根据整模型 layer ratio 生成 KV cache groups；
3. 处理 auto-fit max_model_len 和 memory check；
4. 给每个 worker 生成 KVCacheConfig；
5. 把每个 worker 的 num_blocks 调整到所有 worker 的最小值。
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:1942` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:1960`

### 6.1 为什么要合并所有 worker 的 specs

Pipeline parallel 下，不同 worker 可能持有不同层；但 EngineCore / Scheduler 是集中调度的，KV cache config 必须在所有 worker 间对齐。

因此：

```text
不同 PP stage 可以有不同 layer names；
但同一个 layer 在不同 TP ranks 上的 KVCacheSpec 必须一致。
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:1972` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:1984`

### 6.2 get_kv_cache_config_from_groups()

真正把 groups 和 available memory 变成 config 的函数是：

```text
get_kv_cache_config_from_groups(...)
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:1247`

它主要有三种路径。

#### 路径一：attention-free model

如果没有 KV cache groups：

```text
num_blocks = 1
kv_cache_tensors = []
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:1263` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:1270`

这里仍然返回 `num_blocks=1`，因为 BlockPool 总是需要 null block。

#### 路径二：单组 UniformTypeKVCacheSpecs

如果只有一个 group，但这个 group 内每层 hidden size 不同，会按每层 spec 单独分配 tensor：

```text
KVCacheTensor(
  size = per_layer_spec.page_size_bytes * num_blocks,
  shared_by = [layer_name]
)
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:1272` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:1290`

#### 路径三：通用 hybrid groups

普通 hybrid 情况下，会按 group size 建多个 memory pool。

源码注释给了一个例子：

```text
3 groups:
  (full.0, full.1)
  (sw.0, sw.2)
  (sw.1, padding)

group_size = 2

raw tensor 0 shared_by = [full.0, sw.0, sw.1]
raw tensor 1 shared_by = [full.1, sw.2]
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:1301` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:1326`

关键点：

```text
不同 group 的 layer 可以共享同一个 raw tensor 的不同逻辑区域；
但它们使用不同 block table；
后续 reshape / metadata 会按各自 group 的 spec 和 backend 解释。
```

---

## 7. scheduler block size、hash block size、kernel block size

这里有三个容易混淆的 block size。

### 7.1 cache_config.block_size / KVCacheSpec.block_size

这是 KV cache manager / scheduler 分配和计算 token 对齐时看到的 block size。

它回答：

```text
一个逻辑 KV block 容纳多少 token。
```

### 7.2 scheduler_block_size

`resolve_kv_cache_block_sizes()` 会解析：

```text
scheduler_block_size
hash_block_size
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:593`

单组时：

```text
scheduler_block_size = cache_config.block_size * dcp * pcp
```

多组时：

```text
scheduler_block_size = lcm(所有 group 的 block_size)
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:616` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:627`

含义是：

```text
Scheduler 需要一个全局 token alignment invariant；
hybrid 多组 block_size 不同时，取 LCM 才能让所有 group 对齐。
```

### 7.3 hash_block_size

`hash_block_size` 是 prefix caching / KV connector block hash 的粒度。

单组时通常等于 scheduler block size。

多组且启用 prefix caching 或 KV connector 时：

```text
hash_block_size = cache_config.hash_block_size or gcd(所有 group block_size)
```

但要求：

```text
每个 group 的 block_size 都能被 hash_block_size 整除。
```

位置：`code/vllm/vllm/v1/core/kv_cache_utils.py:629` 到 `code/vllm/vllm/v1/core/kv_cache_utils.py:656`

### 7.4 kernel_block_size

`kernel_block_size` 是 attention kernel 真正支持或希望使用的 block size。

它在 `GPUModelRunner.initialize_kv_cache()` 中由：

```text
prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)
```

生成。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7323` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7330`

源码注释说明：

```text
如果 KV cache manager 对某个 group 使用 block_size 256，
但 attention backend 只支持 block_size 64，
那么 kernel_block_size = 64，
一个 256-token block 会被拆成 4 个 64-token kernel blocks。
```

这就是：

```text
scheduler block size / allocation block size
  和
kernel block size / attention block size
```

可以不同的原因。

### 7.5 backend 为什么只要求 block_size 是 kernel size 的倍数

`AttentionBackend.supports_block_size()` 中有注释：

```text
With hybrid_blocks feature, the framework-level block size only needs to be a multiple of the kernel's requirement.
```

位置：`code/vllm/vllm/v1/attention/backend.py:175` 到 `code/vllm/vllm/v1/attention/backend.py:191`

所以：

```text
backend 固定要求 64-token kernel block；
框架层 block_size 可以是 64、128、256；
ModelRunner 会在 reshape / block table 层把它展开成 kernel blocks。
```

---

## 8. AttentionBackend 对 KV cache 的契约

`AttentionBackend` 定义在：`code/vllm/vllm/v1/attention/backend.py:55`

和 KV cache layout 最相关的是三个方法。

### 8.1 get_kv_cache_shape()

抽象定义：`code/vllm/vllm/v1/attention/backend.py:87`

签名：

```text
get_kv_cache_shape(
  num_blocks,
  block_size,
  num_kv_heads,
  head_size,
  cache_dtype_str="auto",
) -> tuple[int, ...]
```

它回答：

```text
backend 希望每层 kv_cache tensor 的逻辑 shape 是什么。
```

标准 attention backend 常见返回：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

MLA backend 返回：

```text
(num_blocks, block_size, head_size)
```

### 8.2 get_kv_cache_block_dim()

定义位置：`code/vllm/vllm/v1/attention/backend.py:98`

它通过往 `get_kv_cache_shape()` 里塞一个特殊 `num_blocks` 值，判断 block index 在第几个维度。

这个方法服务于 hybrid attention + Mamba 的布局修正。

### 8.3 get_kv_cache_stride_order()

定义位置：`code/vllm/vllm/v1/attention/backend.py:119`

它回答：

```text
逻辑 shape 的各个维度，在物理内存里按什么顺序排列。
```

源码例子：

```text
逻辑 shape: [2, num_blocks, block_size, num_heads, head_size]
stride_order: (1, 3, 0, 2, 4)
物理顺序: [num_blocks, num_heads, 2, block_size, head_size]
```

位置：`code/vllm/vllm/v1/attention/backend.py:122` 到 `code/vllm/vllm/v1/attention/backend.py:146`

如果 backend 没实现这个方法：

```text
物理 layout 默认等于逻辑 shape。
```

### 8.4 include_num_layers_dimension

`get_kv_cache_stride_order(include_num_layers_dimension=True)` 用于 cross-layer uniform KV cache layout。

它表示：

```text
在逻辑 shape 前面额外加一个 num_layers 维度后，backend 希望物理内存如何排列。
```

如果返回值不包含额外层维度，或者层维度仍然放在最外层不参与 per-block 连续布局，就不适合跨层 KV transfer。

### 8.5 get_required_kv_cache_layout()

定义位置：`code/vllm/vllm/v1/attention/backend.py:346`

默认返回：

```text
None
```

某些 backend 会强制：

```text
NHD
HND
```

selector 选出 backend 后会检查：

```text
required_layout = backend.get_required_kv_cache_layout()
if required_layout is not None:
    set_kv_cache_layout(required_layout)
```

位置：`code/vllm/vllm/v1/attention/selector.py:132` 到 `code/vllm/vllm/v1/attention/selector.py:137`

所以 backend selection 可能反过来设置全局 KV cache layout。

---

## 9. NHD / HND layout 是什么

KV cache layout 类型定义在：`code/vllm/vllm/v1/attention/backends/utils.py:42`

```text
KVCacheLayoutType = Literal["NHD", "HND"]
```

可以粗略理解为：

```text
NHD：block_size / token 维度比 head 维度更靠前；
HND：head 维度比 block_size / token 维度更靠前。
```

更准确地看 FlashAttention / FlashInfer 的 stride order。

### 9.1 标准 shape

FlashAttention 返回逻辑 shape：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:140` 到 `code/vllm/vllm/v1/attention/backends/flash_attn.py:149`

FlashInfer 也类似：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:371` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:382`

其中维度语义是：

```text
num_blocks：KV block 数
2：K / V
block_size：block 内 token 数
num_kv_heads：KV heads
head_size：每个 head 的维度
```

### 9.2 NHD stride order

以 FlashAttention 为例：

```text
NHD stride_order = (0, 1, 2, 3, 4)
```

也就是物理顺序和逻辑 shape 一致：

```text
num_blocks, K/V, block_size, num_kv_heads, head_size
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:157` 到 `code/vllm/vllm/v1/attention/backends/flash_attn.py:162`

### 9.3 HND stride order

HND 下：

```text
stride_order = (0, 1, 3, 2, 4)
```

物理顺序变成：

```text
num_blocks, K/V, num_kv_heads, block_size, head_size
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:163` 到 `code/vllm/vllm/v1/attention/backends/flash_attn.py:167`

这意味着：

```text
同一个 block 内，head 维度比 token 维度更连续。
```

某些 kernel / 平台更喜欢这种布局。

### 9.4 FlashInfer Blackwell 强制 HND

FlashInfer 在 SM100 / Blackwell 上会要求：

```text
get_required_kv_cache_layout() → "HND"
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:443` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:448`

这说明：

```text
同一个 FlashInfer backend，在不同硬件上可能对 KV cache layout 有不同要求。
```

---

## 10. GPUModelRunner 初始化 KV cache 的主链路

入口：`GPUModelRunner.initialize_kv_cache()`

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`

主流程：

```text
initialize_kv_cache(kv_cache_config)
  → deepcopy(kv_cache_config)
  → may_add_encoder_only_layers_to_kv_cache_config()
  → maybe_add_kv_sharing_layers_to_kv_cache_groups()
  → initialize_attn_backend(kv_cache_config)
  → initialize_mamba_ssu_backend(...)
  → prepare_kernel_block_sizes(kv_cache_config, attn_groups)
  → initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
  → may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
  → initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
  → register_kv_caches / register_cross_layers_kv_cache for KV connector
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7314` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7360`

这说明 KV cache 初始化不是单纯分配 tensor，而是同时完成：

```text
1. attention backend 分组；
2. kernel block size 准备；
3. metadata builder 创建；
4. InputBatch 的 block table 维度重建；
5. KV cache raw memory 分配和 reshape；
6. attention layer 的 kv_cache 绑定；
7. KV connector 注册。
```

---

## 11. _allocate_kv_cache_tensors()：先分配 raw buffer

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7018`

它做的事很简单：

```text
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7031` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7038`

关键点：

```text
1. 初始分配是 int8 raw byte buffer；
2. 多个 layer_name 可以指向同一块 raw tensor；
3. 此时还没有 backend-specific shape；
4. 后续 _reshape_kv_cache_tensors() 才把它 view 成 dtype + shape。
```

它还会校验：

```text
KV cache config 中需要初始化的 layer names
  ==
raw tensor 中实际初始化的 layer names
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7039` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7047`

---

## 12. _reshape_kv_cache_tensors()：把 raw buffer 变成 backend view

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7059`

这是 KV cache layout 落地的核心函数。

### 12.1 AttentionSpec 分支

对 attention KV cache：

```text
num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
kernel_num_blocks = num_blocks * num_blocks_per_kv_block
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7087` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7095`

含义：

```text
raw tensor 按 allocation block / page_size 计算出 num_blocks；
如果 kernel block 更小，则一个 allocation block 拆成多个 kernel blocks；
backend get_kv_cache_shape() 看到的是 kernel_num_blocks。
```

### 12.2 MLA compression 下的 shape_block_size

如果：

```text
kv_cache_spec.storage_block_size != kv_cache_spec.block_size
```

则：

```text
shape_block_size = kv_cache_spec.storage_block_size
```

否则：

```text
shape_block_size = kernel_block_size
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7097` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7101`

这解释了：

```text
MLA 逻辑 block_size 和真实 tensor 内 block_size 可以不同。
```

### 12.3 调用 backend.get_kv_cache_shape()

核心调用：

```text
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks,
    shape_block_size,
    kv_cache_spec.num_kv_heads,
    kv_cache_spec.head_size,
    cache_dtype_str=self.cache_config.cache_dtype,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7103` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7109`

这一步把：

```text
KVCacheSpec 的模型维度
+
kernel block size
+
backend 的 layout 协议
```

合成最终逻辑 shape。

### 12.4 应用 stride order

随后：

```text
kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
```

如果 backend 没实现，就用 identity：

```text
tuple(range(len(kv_cache_shape)))
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7111` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7115`

ModelRunner 会先按 stride order 重排 shape，再 view raw tensor，最后 `permute(*inv_order)` 还原成逻辑 shape 视图：

```text
物理 memory 按 backend stride order 排列；
返回给 attention layer 的 tensor 仍呈现 backend 定义的逻辑 shape。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7116` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7152`

### 12.5 page_size_padded 的 strided view

如果 `kv_cache_spec.page_size_padded is not None`，ModelRunner 用 `torch.as_strided()` 创建 view。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7131` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7148`

源码注释特别强调：

```text
这个分支假设 kv_cache_shape[0] == num_blocks，
即第一物理维是 block index；
这对 MLA backend 成立，
但对标准 attention backend 不成立，因为标准 attention shape 可能从 K/V 维度开始。
```

这说明 padded page layout 不是对所有 backend 都天然适用，必须配合 backend shape 假设。

---

## 13. bind_kv_cache()：把 KV tensor 绑定到 attention layer

KV cache reshape 完后，还不能被模型 forward 使用。它必须绑定到：

```text
1. ModelRunner.kv_caches
2. static_forward_context 中每个 Attention layer 的 kv_cache 字段
```

入口：`code/vllm/vllm/v1/worker/utils.py:462`

核心逻辑：

```text
for layer_name, kv_cache in kv_caches.items():
    forward_context[layer_name].kv_cache = kv_cache
```

位置：`code/vllm/vllm/v1/worker/utils.py:516` 到 `code/vllm/vllm/v1/worker/utils.py:518`

同时 runner 侧还会按 layer index 把 kv_cache 填入：

```text
runner_kv_caches
```

位置：`code/vllm/vllm/v1/worker/utils.py:484` 到 `code/vllm/vllm/v1/worker/utils.py:515`

绑定后，attention forward 才能通过：

```text
get_attention_context(layer_name)
  → attn_layer.kv_cache
```

拿到当前层的 KV cache tensor。

---

## 14. 标准 Attention backend 的 KV cache shape

### 14.1 FlashAttention

FlashAttention backend 返回：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:140` 到 `code/vllm/vllm/v1/attention/backends/flash_attn.py:149`

并要求：

```text
block_size % 16 == 0
```

原因是 FlashAttention kernel 对 block size 有对齐要求。

它的 stride order 根据全局 KV cache layout 变化：

```text
NHD → (0, 1, 2, 3, 4)
HND → (0, 1, 3, 2, 4)
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:151` 到 `code/vllm/vllm/v1/attention/backends/flash_attn.py:170`

### 14.2 FlashInfer

FlashInfer backend 普通返回：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

NVFP4 时最后一维会变成 packed full dim：

```text
(num_blocks, 2, block_size, num_kv_heads, nvfp4_full_dim)
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:371` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:382`

它同样支持 NHD / HND stride order。

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:385` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:403`

区别是：

```text
FlashInfer 在 Blackwell 上会强制 HND；
FlashInfer.forward_includes_kv_cache_update = False。
```

位置：

- `code/vllm/vllm/v1/attention/backends/flashinfer.py:443`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:450`

这意味着：

```text
KV cache layout 由 backend 指定；
KV cache update 可能由 Attention.forward() 在 backend attention 前单独执行。
```

---

## 15. MLA backend 的 KV cache shape

MLA 的 common backend 返回：

```text
(num_blocks, block_size, head_size)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1195` 到 `code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1203`

和标准 attention 不同：

```text
标准 attention：存 K + V，所以有 2 维；
MLA：存 compressed latent KV + RoPE 相关部分，不是标准 K/V pair。
```

MLA stride order 默认是 identity：

```text
(num_blocks, block_size, head_size)
```

如果 `include_num_layers_dimension=True`，返回：

```text
(0, 1, 2, 3)
```

源码注释说明：

```text
MLA kernels require contiguous per-layer KV cache views.
Identity permutation keeps num_layers first in physical layout, signaling cross-layer allocation is unsupported.
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1204` 到 `code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1213`

所以：

```text
MLA 不适合 KV connector 的 cross-layer blocks uniform layout；
每层 KV cache view 需要保持连续。
```

---

## 16. Mamba / hybrid attention 的特殊布局

Mamba 不是传统 K/V cache。

在 `_reshape_kv_cache_tensors()` 中，Mamba 分支会：

```text
for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
    用 torch.as_strided 从 raw tensor 中切出 state tensor
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7154` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7177`

如果一个模型同时有 attention 和 Mamba：

```text
has_attn and has_mamba
  → _update_hybrid_attention_mamba_layout(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7181` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7183`

这个函数会把某些 attention layer 的 layout 从：

```text
(2, num_blocks, ...)
```

调整成：

```text
(num_blocks, 2, ...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7186` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7218`

原因是 hybrid attention + Mamba 下，不同 cache 类型要在 block 维度上保持一致，方便统一 slot / state layout。

---

## 17. Cross-layer KV sharing 如何影响 KV cache groups

Cross-layer KV sharing 指某些 attention layer 不单独分配 KV cache，而是复用另一个 layer 的 KV cache。

### 17.1 get_kv_cache_spec() 阶段跳过共享层

在 `GPUModelRunner.get_kv_cache_spec()` 中，如果 attention layer 有：

```text
kv_sharing_target_layer_name
```

则：

```text
self.shared_kv_cache_layers[layer_name] = target_layer
continue
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7472` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7484`

含义是：

```text
共享层不生成自己的 KVCacheSpec；
KV cache manager 把它当作不存在，不为它分配独立 cache。
```

### 17.2 初始化时加入 target group

但 metadata 仍然要为共享层准备。

`maybe_add_kv_sharing_layers_to_kv_cache_groups()` 会调用：

```text
add_kv_sharing_layers_to_kv_cache_groups(...)
```

位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7275`
- `code/vllm/vllm/v1/worker/utils.py:428`

它会把共享 layer 加进 target layer 所在的 KV cache group：

```text
tgt_kv_cache_group.layer_names.append(layer_name)
runner_only_attn_layers.add(layer_name)
```

位置：`code/vllm/vllm/v1/worker/utils.py:454` 到 `code/vllm/vllm/v1/worker/utils.py:459`

### 17.3 reshape 后复用目标 kv_cache tensor

KV cache tensor 初始化后：

```text
kv_caches[layer_name] = kv_caches[target_layer_name]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7259` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7263`

这说明：

```text
共享层不拥有独立 raw tensor；
但 forward context 中仍然能通过 layer_name 找到 kv_cache；
这个 kv_cache 实际上指向 target layer 的 tensor。
```

### 17.4 kv_sharing_fast_prefill

如果开启 `cache_config.kv_sharing_fast_prefill`，ModelRunner 会标记从末尾连续的 KV sharing layers：

```text
kv_sharing_fast_prefill_eligible_layers
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7292` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7301`

这类路径的目标是：

```text
prefill 阶段只让真正生成 KV cache 的 layer 参与，
共享层在 prefill 中可以更早退出或复用 metadata。
```

对应 common metadata 包装工具在：`code/vllm/vllm/v1/worker/utils.py:397`

---

## 18. KV connector 为什么关心 cross-layer layout

KV connector 要把 KV cache 在 worker 之间、设备之间或外部存储之间传输。

如果按 layer 一个个传，每个 block 的所有 layer KV 分散在不同 tensor 中，传输开销会更高。

因此某些 connector 会偏好：

```text
同一个 block id 下，所有 layer 的 KV 数据在物理内存中尽量连续。
```

这就是 uniform / cross-layer KV cache layout。

### 18.1 use_uniform_kv_cache()

入口：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:115`

它判断是否使用 uniform KV layout。

需要同时满足：

```text
1. 存在 KV transfer group；
2. connector.prefer_cross_layer_blocks 为 True；
3. 只有一个 attention group，且所有层 page size 相同；
4. kv_cache_spec 是 AttentionSpec；
5. backend.get_kv_cache_stride_order(include_num_layers_dimension=True)
   支持把 num_layers 维度放进物理 layout；
6. stride_order[0] != 0，即层维度不是保持最外层 identity layout。
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:148` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:183`

源码注释说得很直观：

```text
A uniform layout means all layers KV caches will share the same underlying tensor,
where for a given block number, the respective KV data for all layers will be contiguous.
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:119` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:125`

### 18.2 allocate_uniform_kv_caches()

如果 `use_uniform_kv_cache()` 返回 True，`initialize_kv_cache_tensors()` 会走：

```text
allocate_uniform_kv_caches(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7235` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7249`

返回：

```text
kv_caches
cross_layers_kv_cache
attn_backend
```

其中 `cross_layers_kv_cache` 会被注册给 KV transfer group：

```text
register_cross_layers_kv_cache(cross_layers_kv_cache, cross_layers_attn_backend)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7351` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7357`

否则走普通：

```text
register_kv_caches(kv_caches)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7358` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7360`

### 18.3 为什么 MLA 通常不支持 cross-layer uniform layout

MLA backend 的 `get_kv_cache_stride_order(include_num_layers_dimension=True)` 返回 identity：

```text
(0, 1, 2, 3)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1204` 到 `code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1213`

而 `use_uniform_kv_cache()` 要求：

```text
stride_order[0] != 0
```

因此 MLA 会被判定为不适合 cross-layer block layout。

---

## 19. KV cache layout 如何进入 attention forward

KV cache 初始化完成后，每层 Attention 对象都有：

```text
attn_layer.kv_cache
```

每轮 forward 时，`set_forward_context()` 注入：

```text
attn_metadata
slot_mapping
```

模型内部 `Attention.forward()` 会通过：

```text
get_attention_context(layer_name)
```

拿到：

```text
attn_metadata
attn_layer
kv_cache
layer_slot_mapping
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`

然后根据 backend 的：

```text
forward_includes_kv_cache_update
```

决定：

```text
backend impl.forward 内更新 KV cache；
或者先 unified_kv_cache_update()，再 attention forward。
```

这就是 KV cache layout 的最终消费点：

```text
backend impl 看到的是自己的 kv_cache shape；
attn_metadata 里有 block_table / slot_mapping / seq_lens；
impl 用这些信息按自身 layout 读写实际 memory。
```

---

## 20. layout、block table、slot mapping 的关系

可以把三者理解成不同层级的地址系统。

```text
block table：
  request row → physical block id。

slot mapping：
  token → flattened slot id。

KV cache tensor layout：
  flattened slot id 对应的 block / offset / head / K/V 在 tensor 中怎么排。
```

例如标准 attention cache：

```text
kv_cache shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
slot_id = block_id * block_size + offset
```

backend 可以把 `slot_id` 解释成：

```text
block_id = slot_id // block_size
offset = slot_id % block_size
kv_cache[block_id, 0, offset, :, :]  # key
kv_cache[block_id, 1, offset, :, :]  # value
```

如果是 HND 物理 layout，逻辑索引仍可这么写，但底层 stride 会让 `num_kv_heads` 维度更连续。

所以：

```text
slot mapping 不直接关心 HND / NHD；
backend kernel 和 kv_cache tensor stride 共同决定 HND / NHD 的真实性能和正确访问方式。
```

---

## 21. KV cache dtype 和 quantized layout

`AttentionSpec` 的 `dtype` 和 `kv_quant_mode` 会影响 `page_size_bytes`。

普通 fp16 / bf16：

```text
2 * block_size * num_kv_heads * head_size * dtype_size
```

NVFP4：

```text
2 * block_size * num_kv_heads * nvfp4_kv_cache_full_dim(head_size) * dtype_size
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:183` 到 `code/vllm/vllm/v1/kv_cache_interface.py:199`

FlashInfer 的 shape 也会配合 NVFP4：

```text
last_dim = nvfp4_kv_cache_full_dim(head_size)
return (num_blocks, 2, block_size, num_kv_heads, last_dim)
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:378` 到 `code/vllm/vllm/v1/attention/backends/flashinfer.py:382`

这说明：

```text
KV cache dtype 不只是 tensor dtype；
它可能改变每个 slot 的 packed layout、scale 存储和最后一维 shape。
```

---

## 22. Encoder-only / no-KV layer 如何处理

不是所有 attention layer 都需要 KV cache。

### 22.1 get_kv_cache_spec() 会跳过 no-KV layer

`GPUModelRunner.get_kv_cache_spec()` 中：

```text
if spec := attn_module.get_kv_cache_spec(...):
    kv_cache_spec[layer_name] = spec
```

如果 layer 不需要 KV cache，返回 None，就不会进入 KV cache manager。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7485` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7487`

### 22.2 encoder-only layer 可能作为 runner-only group 加入

ModelRunner 还有：

```text
may_add_encoder_only_layers_to_kv_cache_config()
```

它会为 encoder-only attention 构造 `EncoderOnlyAttentionSpec` 并加入 `kv_cache_groups`，同时放入：

```text
runner_only_attn_layers
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7433` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7457`

含义是：

```text
某些 layer 需要 metadata / attention group 管线，
但不一定需要独立真实 KV cache allocation。
```

---

## 23. 一个完整例子：普通 FlashAttention KV cache

假设：

```text
backend = FlashAttention
num_blocks = 1000
block_size = 16
num_kv_heads = 8
head_size = 128
dtype = bf16
layout = NHD
```

`AttentionSpec.page_size_bytes`：

```text
2 * 16 * 8 * 128 * 2 bytes = 65536 bytes
```

每层 raw tensor 大小约：

```text
1000 * 65536 bytes
```

ModelRunner 分配：

```text
torch.zeros(size, dtype=torch.int8)
```

然后 FlashAttention shape：

```text
(1000, 2, 16, 8, 128)
```

NHD stride order：

```text
(0, 1, 2, 3, 4)
```

最终 attention backend 可以按：

```text
kv_cache[block_id, 0, offset, kv_head, head_dim]  # K
kv_cache[block_id, 1, offset, kv_head, head_dim]  # V
```

访问逻辑 KV。

---

## 24. 一个完整例子：FlashInfer Blackwell HND

假设同样是标准 attention，但在 Blackwell 上自动选择 FlashInfer。

FlashInfer：

```text
get_required_kv_cache_layout() → HND
```

selector 会调用：

```text
set_kv_cache_layout("HND")
```

然后逻辑 shape 仍是：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

但 stride order 变成：

```text
(0, 1, 3, 2, 4)
```

也就是物理顺序更像：

```text
num_blocks, K/V, num_kv_heads, block_size, head_size
```

这对用户层看起来仍然是同一个 `kv_cache` 逻辑 shape，但 kernel 看到的内存连续性不同。

这就是：

```text
同一个 shape，不同 stride order，可以表示不同物理 layout。
```

---

## 25. 一个完整例子：MLA KV cache

假设 MLA backend：

```text
num_blocks = 1000
block_size = 64
head_size = 576
num_kv_heads = 1
```

MLA backend shape：

```text
(1000, 64, 576)
```

没有标准 attention 的：

```text
2 维 K/V pair
num_kv_heads 维
```

因为 MLA cache 存的是 compressed 表示。

如果 `compress_ratio = 2`，那么：

```text
storage_block_size = block_size // 2 = 32
```

ModelRunner 传给 `get_kv_cache_shape()` 的 `shape_block_size` 会变成 32。

所以实际 shape 可能是：

```text
(1000, 32, 576)
```

这也是为什么不能把 MLA cache 当成普通 K/V cache 来理解。

---

## 26. 容易疑惑的点

### 26.1 KVCacheSpec.page_size_bytes 和 backend.get_kv_cache_shape() 是不是一回事？

不是。

```text
page_size_bytes：一页 / 一个 block 需要多少字节，用于内存预算和 raw buffer 分配。

get_kv_cache_shape()：同一块 raw buffer 被 view 成什么 tensor shape，用于 backend kernel 访问。
```

两者必须一致，但职责不同。

### 26.2 KVCacheTensor.shared_by 是不是等于 KV sharing？

不完全等于。

`shared_by` 表示这些 layer name 共用同一个 raw tensor allocation 描述；它可能来自 hybrid group memory pool 布局，也可能是统一布局。

Cross-layer KV sharing 还会额外通过：

```text
shared_kv_cache_layers[layer] = target_layer
kv_caches[layer] = kv_caches[target_layer]
```

让两个 layer 指向同一个最终 kv_cache tensor。

### 26.3 block_size 和 kernel_block_size 谁决定 slot_mapping？

逻辑上，Scheduler / KVCacheManager 分配使用 `KVCacheSpec.block_size`。

但当 backend kernel block size 更小时，BlockTable / ModelRunner 会把 allocation block 展开成多个 kernel blocks，最终传给 kernel 的 block table 和 cache shape 使用 kernel block size。

### 26.4 HND / NHD 会改变 slot_id 吗？

通常不会改变 slot_id 的语义。

`slot_id` 仍然表达：

```text
block_id * block_size + offset
```

HND / NHD 改变的是 `kv_cache` tensor 的物理 stride，即同一个 block 内 head/token 维度的连续性。

### 26.5 backend selection 会改变 KV cache layout 吗？

会。

backend 可以通过：

```text
get_required_kv_cache_layout()
get_kv_cache_shape()
get_kv_cache_stride_order()
get_supported_kernel_block_sizes()
```

影响全局 layout、tensor shape、physical stride 和 kernel block size。

### 26.6 为什么 raw tensor 用 int8 分配？

因为 `KVCacheTensor.size` 是字节数。

先用 int8 分配 raw bytes，后面再按 `kv_cache_spec.dtype` view 成 fp16 / bf16 / fp8 / 其他 dtype，并按 backend shape / stride 创建视图。

### 26.7 uniform KV cache 为什么要求单 group？

因为 cross-layer contiguous block layout 要求所有层 page size / backend layout 兼容。

如果有多个 KV cache group、多个 backend 或不同 page size，很难保证“同一 block 的所有层 KV 连续”。

### 26.8 MLA 为什么不能简单复用标准 attention 的 layout？

因为 MLA cache 不是 K/V pair，而是 compressed latent KV + RoPE 相关表示；它的 page size、shape、prefill/decode forward 都不同于标准 attention。

### 26.9 KV cache layout 和 attention metadata 谁先决定？

初始化阶段先确定：

```text
backend → kv_cache_shape / stride → kv_cache tensor
```

每轮执行再构造：

```text
block_table / slot_mapping / seq_lens → attention metadata
```

metadata 会引用已经存在的 kv_cache layout，但不会重新分配 KV cache。

---

## 27. 调试入口

调试 KV cache layout，建议按这条顺序看：

```text
1. Attention / MLAAttention.get_kv_cache_spec()
   看每层生成什么 KVCacheSpec。

2. get_kv_cache_configs()
   看 specs 如何合并成 KV cache groups 和 KVCacheConfig。

3. resolve_kv_cache_block_sizes()
   看 scheduler_block_size / hash_block_size。

4. GPUModelRunner.initialize_kv_cache()
   看是否加入 encoder-only / KV sharing layers，backend groups 怎么初始化。

5. prepare_kernel_block_sizes()
   看 allocation block size 和 kernel block size 是否不同。

6. _allocate_kv_cache_tensors()
   看 raw tensor size 和 shared_by。

7. _reshape_kv_cache_tensors()
   看 backend.get_kv_cache_shape()、stride_order、storage_block_size。

8. bind_kv_cache()
   看每个 layer 的 kv_cache 是否绑定到 Attention 对象。

9. set_forward_context() / get_attention_context()
   看 forward 时 layer 是否取到正确 kv_cache / metadata / slot_mapping。

10. backend impl.forward()
   看 backend 如何解释 kv_cache layout。
```

常用源码位置：

```text
code/vllm/vllm/v1/kv_cache_interface.py:95
code/vllm/vllm/v1/kv_cache_interface.py:867
code/vllm/vllm/v1/core/kv_cache_utils.py:1247
code/vllm/vllm/v1/core/kv_cache_utils.py:1937
code/vllm/vllm/v1/attention/backend.py:87
code/vllm/vllm/v1/worker/gpu_model_runner.py:7059
code/vllm/vllm/v1/worker/gpu_model_runner.py:7220
code/vllm/vllm/v1/worker/utils.py:462
```

---

## 28. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心产物 | 作用 |
|---|---|---|---|
| 层声明 | `Attention.get_kv_cache_spec()` / `MLAAttention.get_kv_cache_spec()` | `KVCacheSpec` | 描述每层 KV cache 格式 |
| group 构造 | `get_kv_cache_groups()` | `KVCacheGroupSpec` | 把共享 block table 的层分组 |
| config 生成 | `get_kv_cache_configs()` | `KVCacheConfig` | 决定每个 worker 分配多少 blocks / tensors |
| block size 解析 | `resolve_kv_cache_block_sizes()` | scheduler / hash block size | 决定调度和 block hash 粒度 |
| backend 协议 | `AttentionBackend.get_kv_cache_shape()` | KV cache logical shape | 决定 backend 如何看 KV tensor |
| 物理布局 | `get_kv_cache_stride_order()` | stride order | 决定内存维度连续性 |
| kernel block | `prepare_kernel_block_sizes()` | kernel block sizes | 适配 backend kernel block 要求 |
| raw 分配 | `_allocate_kv_cache_tensors()` | int8 raw buffers | 按字节分配显存 |
| reshape | `_reshape_kv_cache_tensors()` | per-layer kv_cache view | 把 raw buffer view 成 backend layout |
| 绑定 | `bind_kv_cache()` | `attn_layer.kv_cache` | 让 attention forward 能取到 KV cache |
| 执行 | `backend impl.forward()` | attention output | 按 layout + metadata 读写 KV cache |
| connector | `register_kv_caches()` / `register_cross_layers_kv_cache()` | transfer registration | 让 KV connector 能 load/save KV |

---

## 29. 总结

KV cache layout 和 backend 的关系可以压缩成下面这条线：

```text
Attention layer
  → get_kv_cache_spec()
  → KVCacheSpec(page_size_bytes, block_size, dtype, head info)
  → get_kv_cache_configs()
  → KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
  → GPUModelRunner.initialize_kv_cache()
  → prepare_kernel_block_sizes()
  → _allocate_kv_cache_tensors()  # raw int8 bytes
  → _reshape_kv_cache_tensors()
      → backend.get_kv_cache_shape()
      → backend.get_kv_cache_stride_order()
  → bind_kv_cache()
  → set_forward_context(attn_metadata, slot_mapping)
  → Attention.forward()
  → backend impl.forward(kv_cache, attn_metadata)
```

最小心智模型是：

```text
KVCacheSpec 负责“算容量”；
KVCacheConfig 负责“安排分配”；
AttentionBackend 负责“声明形状和 stride”；
GPUModelRunner 负责“把 raw bytes reshape 并绑定到 layer”；
attention metadata 负责“告诉 kernel 本轮读写哪些 block / slot”；
backend impl 负责“按自己的 layout 真正读写 KV cache”。
```

如果只记住一句话：

```text
vLLM 的 KV cache 不是固定形状的大张量，而是一组由 KVCacheSpec 预算、KVCacheConfig 分配、AttentionBackend 定义 shape/stride、ModelRunner reshape/bind、attention metadata 定位读写位置的 backend-specific memory view。
```
