# 08. 量化如何影响 attention backend？

源码位置：

- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/registry.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/vllm/v1/attention/backends/utils.py`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/quantization/kv_cache.py`
- `vllm/vllm/platforms/cuda.py`
- `vllm/vllm/platforms/interface.py`

本问题关注：KV cache dtype、KV scale、per-token-head scale、NVFP4 / TurboQuant 这类特殊 KV cache layout、MLA / sparse MLA、head size、block size、backend 能力声明如何共同决定 attention backend 选择；以及选中 backend 后，量化 KV cache 如何在 forward 中写入、读取和传递 scale。

---

## 1. 一句话回答

量化会把 attention backend 选择从“哪个 backend 最快”变成：

```text
哪个 backend：
  1. 支持当前模型 dtype；
  2. 支持当前 kv_cache_dtype；
  3. 支持当前 head_size / block_size；
  4. 支持当前 attention 形态（普通 / MLA / sparse / sliding window / sink / non-causal）；
  5. 支持当前 KV cache scale 形式；
  6. 支持当前 KV cache layout；
  7. 在这些约束下优先级最高。
```

核心结论：

```text
KV cache 量化直接参与 backend selection；
权重量化通常不直接选择 attention backend；
但量化 checkpoint 可能携带 KV cache scheme / scale，从而间接影响 attention backend。
```

可以先记：

```text
FP16 / BF16 KV cache：
  FlashAttention / FlashInfer / Triton 都可能可选。

FP8 KV cache：
  需要 backend 支持 fp8 cache dtype 和 k/v scale。

per-token-head KV scale：
  需要 backend 明确 supports_per_head_quant_scales()。

NVFP4 KV cache：
  基本走 FlashInfer + TRTLLM / Blackwell 相关路径。

TurboQuant KV cache：
  走 TurboQuantAttentionBackend，使用专用 packed K+V cache layout。
```

---

## 2. 总体链路

attention backend 选择发生在 Attention layer 初始化阶段。

主链路是：

```text
Attention.__init__()
  → 解析 cache_config.cache_dtype
  → 处理 quant_config.kv_cache_scheme
  → 处理 kv_cache_dtype_skip_layers
  → get_attn_backend(...)
  → AttentionSelectorConfig
  → current_platform.get_attn_backend_cls(...)
  → backend.validate_configuration(...)
  → 选择最高优先级可用 backend
  → backend.get_impl_cls()
  → impl.forward(...)
```

对应源码：

- `vllm/vllm/model_executor/layers/attention/attention.py:240`
- `vllm/vllm/model_executor/layers/attention/attention.py:318`
- `vllm/vllm/v1/attention/selector.py:54`
- `vllm/vllm/platforms/cuda.py:372`
- `vllm/vllm/v1/attention/backend.py:308`

量化相关字段主要进入这里：

```text
kv_cache_dtype
use_per_head_quant_scales
use_mla
use_sparse
head_size
block_size
attn_type
```

---

## 3. Attention layer 里量化信息如何进入 backend selector

### 3.1 kv_cache_dtype 的来源

在 `Attention.__init__()` 中，KV cache dtype 先来自：

```text
cache_config.cache_dtype
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:240`

如果没有 cache_config，则默认为：

```text
auto
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:243`

### 3.2 checkpoint 的 kv_cache_scheme 可以把 auto 改成 fp8

如果 quant_config 中带有：

```text
kv_cache_scheme
```

并且当前 `kv_cache_dtype == "auto"`，Attention 会把它改成：

```text
fp8
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:252`

这说明：

```text
即使用户没有显式传 --kv-cache-dtype，
量化 checkpoint 也可能通过 kv_cache_scheme 让 Attention 走 FP8 KV cache。
```

### 3.3 kv_cache_dtype_skip_layers 可以局部跳过 KV cache 量化

如果配置了：

```text
cache_config.kv_cache_dtype_skip_layers
```

Attention 会按 layer index 或 sliding_window 类型跳过量化 KV cache。

位置：`vllm/vllm/model_executor/layers/attention/attention.py:267`

跳过后：

```text
kv_cache_dtype = "auto"
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:282`

这意味着：

```text
同一个模型里，不同 attention layer 可能使用不同 KV cache dtype。
```

这会进一步影响 KV cache spec、page size、backend 支持和 cache 分组。

---

## 4. get_attn_backend() 的输入是什么

`get_attn_backend()` 定义在：`vllm/vllm/v1/attention/selector.py:54`

它接收：

```python
head_size
dtype
kv_cache_dtype
use_mla
has_sink
use_sparse
use_mm_prefix
use_per_head_quant_scales
attn_type
num_heads
```

然后构造：

```python
AttentionSelectorConfig
```

位置：`vllm/vllm/v1/attention/selector.py:21`

这个 config 里和量化最相关的是：

```text
kv_cache_dtype：
  当前 KV cache 的 dtype / 量化格式。

use_per_head_quant_scales：
  是否使用 per-head / per-token-head scale。

use_mla：
  是否是 MLA attention。

use_sparse：
  是否是 sparse attention。

block_size：
  用户显式 block size 会参与 backend 校验。
```

注意：

```text
selector 不直接看 quantization="awq" / "gptq"；
它主要看 Attention 实际需要的 KV cache dtype 和 attention 形态。
```

---

## 5. backend 能力声明在哪里

所有 attention backend 都继承：

```python
AttentionBackend
```

位置：`vllm/vllm/v1/attention/backend.py:55`

它提供一组能力声明：

```text
supported_dtypes
supported_kv_cache_dtypes
get_supported_kernel_block_sizes()
get_kv_cache_shape()
get_kv_cache_stride_order()
supports_head_size()
supports_dtype()
supports_kv_cache_dtype()
supports_block_size()
supports_per_head_quant_scales()
supports_compute_capability()
supports_attn_type()
supports_combination()
get_required_kv_cache_layout()
is_mla()
is_sparse()
```

其中和量化最相关的是：

```text
supported_kv_cache_dtypes
supports_kv_cache_dtype()
supports_per_head_quant_scales()
get_kv_cache_shape()
get_required_kv_cache_layout()
```

---

## 6. validate_configuration 如何过滤 backend

`AttentionBackend.validate_configuration()` 是 backend selection 的统一过滤器。

位置：`vllm/vllm/v1/attention/backend.py:308`

它会检查：

```text
1. head_size 是否支持；
2. model dtype 是否支持；
3. kv_cache_dtype 是否支持；
4. block_size 是否支持；
5. mm_prefix 是否支持；
6. MLA / non-MLA 是否匹配；
7. sink attention 是否支持；
8. sparse / non-sparse 是否匹配；
9. per-head quant scales 是否支持；
10. compute capability 是否支持；
11. attention type 是否支持；
12. non-causal / batch invariance / KV connector 是否支持；
13. backend 自定义 supports_combination() 是否拒绝该组合。
```

量化相关失败原因通常是：

```text
kv_cache_dtype not supported
per-head quant scales not supported
compute capability not supported
block_size not supported
```

这也是为什么同一个模型在不同 GPU、不同 `--kv-cache-dtype` 下会选到不同 backend。

---

## 7. CUDA 平台如何给 backend 排优先级

CUDA 平台入口是：

```python
CudaPlatformBase.get_attn_backend_cls(...)
```

位置：`vllm/vllm/platforms/cuda.py:372`

流程是：

```text
1. 如果用户显式指定 backend，先只校验它；
2. 如果指定 backend 不合法，直接报错；
3. 如果未指定，则调用 get_valid_backends()；
4. get_valid_backends() 按优先级枚举候选 backend；
5. 对每个 backend 调 validate_configuration()；
6. 选择合法 backend 中优先级最高的那个。
```

对应源码：`vllm/vllm/platforms/cuda.py:381` 到 `vllm/vllm/platforms/cuda.py:463`

### 7.1 非 MLA 普通 attention 的优先级

在 CUDA 上，非 MLA 情况下 `_get_backend_priorities()` 大致是：

Blackwell / compute capability major 10：

```text
FLASHINFER
FLASH_ATTN
TRITON_ATTN
FLEX_ATTENTION
TURBOQUANT
```

其他 CUDA GPU：

```text
FLASH_ATTN
FLASHINFER
TRITON_ATTN
FLEX_ATTENTION
TURBOQUANT
```

位置：`vllm/vllm/platforms/cuda.py:137`

### 7.2 MLA 的优先级

如果 `use_mla=True`，候选变成 MLA backend：

```text
FLASHINFER_MLA
TOKENSPEED_MLA
CUTLASS_MLA
FLASH_ATTN_MLA
FLASHMLA
TRITON_MLA
FLASHINFER_MLA_SPARSE / FLASHMLA_SPARSE
```

位置：`vllm/vllm/platforms/cuda.py:92`

### 7.3 量化 KV cache 会改变 MLA sparse 优先级

在 Blackwell + MLA sparse 场景，如果：

```text
kv_cache_dtype 是量化 KV cache
```

CUDA 平台会更偏向：

```text
FLASHINFER_MLA_SPARSE
FLASHMLA_SPARSE
```

位置：`vllm/vllm/platforms/cuda.py:97`

这说明 KV cache 量化不只是“某个 backend 支不支持”，还会影响候选优先级。

---

## 8. KV cache quant mode 如何表示

KV cache 量化模式定义在：`vllm/vllm/v1/kv_cache_interface.py:33`

核心枚举：

```text
KVQuantMode.NONE
KVQuantMode.FP8_PER_TENSOR
KVQuantMode.INT8_PER_TOKEN_HEAD
KVQuantMode.FP8_PER_TOKEN_HEAD
KVQuantMode.NVFP4
```

`get_kv_quant_mode()` 把字符串映射成枚举。

位置：`vllm/vllm/v1/kv_cache_interface.py:60`

规则是：

```text
int8_per_token_head → INT8_PER_TOKEN_HEAD
fp8_per_token_head  → FP8_PER_TOKEN_HEAD
nvfp4               → NVFP4
fp8*                → FP8_PER_TENSOR
其他                → NONE
```

这个 mode 会进入：

```text
AttentionSpec.kv_quant_mode
backend impl.kv_quant_mode
kernel dispatch
page size 计算
```

---

## 9. KV cache spec 与 page size 如何受量化影响

`AttentionSpec` 定义在：`vllm/vllm/v1/kv_cache_interface.py:159`

它包含：

```text
num_kv_heads
head_size
dtype
kv_quant_mode
page_size_padded
indexes_kv_by_block_stride
```

### 9.1 普通 KV cache page size

普通 full attention 的真实 page size 是：

```text
2 * block_size * num_kv_heads * head_size * dtype_size
```

位置：`vllm/vllm/v1/kv_cache_interface.py:195`

这里的 `2` 表示 K 和 V。

### 9.2 per-token-head scale 会额外占内存

如果：

```text
kv_quant_mode.is_per_token_head
```

`page_size_bytes` 会额外加上：

```text
2 * block_size * num_kv_heads * sizeof(float32)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:173`

原因是：

```text
per-token-head scale 虽然由 backend 管理，
但内存要从原始 KV cache allocation 里预算出来。
```

### 9.3 NVFP4 有特殊 packed 维度

如果：

```text
kv_quant_mode.is_nvfp4
```

真实 page size 使用：

```text
nvfp4_kv_cache_full_dim(head_size)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:185`

也就是说 NVFP4 不是简单把 head_size 除以 2，而是有：

```text
packed fp4 data + fp8 block scales
```

的布局要求。

---

## 10. KV cache layout：NHD / HND 如何影响 backend

KV cache layout 工具在：`vllm/vllm/v1/attention/backends/utils.py`

支持的布局是：

```text
NHD
HND
```

位置：`vllm/vllm/v1/attention/backends/utils.py:42`

`get_kv_cache_layout()` 的优先级是：

```text
1. backend selector 通过 set_kv_cache_layout() 设置的 override；
2. 用户环境变量 VLLM_KV_CACHE_LAYOUT；
3. KV connector 要求的 layout。
```

位置：`vllm/vllm/v1/attention/backends/utils.py:82`

某些 backend 会强制要求 layout：

```python
backend.get_required_kv_cache_layout()
```

selector 中处理：`vllm/vllm/v1/attention/selector.py:133`

例如：

```text
FlashInfer 某些路径要求 HND；
FlashInfer MLA / Tokenspeed MLA 等也可能要求 HND；
CPU backend 要求 HND。
```

对应源码：

- `vllm/vllm/v1/attention/backends/flashinfer.py:448`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:104`
- `vllm/vllm/v1/attention/backends/mla/tokenspeed_mla.py:128`
- `vllm/vllm/v1/attention/backends/cpu_attn.py:96`

这说明量化格式如果只能被某个 backend 支持，也会间接要求某种 KV cache layout。

---

## 11. FlashAttention backend 与量化 KV cache

源码：`vllm/vllm/v1/attention/backends/flash_attn.py`

### 11.1 基础支持

`FlashAttentionBackend` 默认支持：

```text
auto
float16
bfloat16
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:70`

### 11.2 FP8 KV cache 支持有限

`supports_kv_cache_dtype()` 中：

```text
fp8 / fp8_e4m3：
  XPU 可以支持；
  CUDA 上要求 FlashAttention v3 且 device capability family 90。

其他：
  auto / float16 / bfloat16。
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:183`

这意味着：

```text
同样是 fp8 KV cache，FlashAttention 不一定能用；
如果条件不满足，selector 会 fallback 到 FlashInfer / Triton 等 backend。
```

### 11.3 per-head quant scale 支持

FlashAttention 的：

```python
supports_per_head_quant_scales()
```

要求 FlashAttention version >= 3。

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:127`

### 11.4 KV cache update 不在 forward 内部

FlashAttentionBackend 声明：

```python
forward_includes_kv_cache_update = False
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:96`

所以 Attention layer 会在 forward 前 / 中显式处理 KV cache 写入。

FlashAttention forward 中会调用：

```text
reshape_and_cache_flash(..., kv_cache_dtype, layer._k_scale, layer._v_scale)
```

相关位置：`vllm/vllm/v1/attention/backends/flash_attn.py:951`

也就是说 FP8 KV cache 的写入需要带上：

```text
k_scale
v_scale
kv_cache_dtype
slot_mapping
```

---

## 12. TritonAttention backend 与量化 KV cache

源码：`vllm/vllm/v1/attention/backends/triton_attn.py`

### 12.1 支持的 KV cache dtype

TritonAttentionBackend 支持：

```text
auto
float16
bfloat16
fp8
fp8_e4m3
fp8_e5m2
int8_per_token_head
fp8_per_token_head
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:254`

这说明 Triton 是普通 attention 中覆盖 KV cache 量化类型比较广的 backend。

### 12.2 per-token-head scale 路径

Triton 会判断：

```python
kv_cache_uses_per_token_head_scales(cache_dtype_str)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:303`

在 impl 中维护：

```text
_k_scale_cache
_v_scale_cache
_is_per_token_head_quant
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:377`

如果是 per-token-head quant，会调用：

```text
triton_reshape_and_cache_flash_per_token_head_quant(...)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:743`

这条路径与普通 FP8 per-tensor scale 不同：

```text
per-tensor scale：
  使用 layer._k_scale / layer._v_scale。

per-token-head scale：
  scale 写入 backend 管理的 k_scale_cache / v_scale_cache。
```

### 12.3 普通量化 KV cache update

非 per-token-head 但仍是量化 KV cache 时，Triton 会调用：

```text
triton_reshape_and_cache_flash(..., kv_cache_dtype, layer._k_scale, layer._v_scale)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:758`

因此 Triton 的职责包括：

```text
1. 把本轮 key/value 按 slot_mapping 写入 KV cache；
2. 如果 KV cache 是 fp8/int8 等量化格式，写入时应用量化；
3. decode / prefill 时读取量化 cache 并使用对应 descale。
```

---

## 13. FlashInfer backend 与量化 KV cache

源码：`vllm/vllm/v1/attention/backends/flashinfer.py`

### 13.1 支持范围

FlashInfer backend 支持多种量化 KV cache，包括：

```text
fp8 / fp8_e4m3 / fp8_e5m2
nvfp4
```

并且内部会把 vLLM 的字符串 dtype 转成 FlashInfer 需要的 dtype。

相关位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:327`
- `vllm/vllm/v1/attention/backends/flashinfer.py:410`

### 13.2 FP8 KV cache 与 TRTLLM 路径

FlashInfer 里有专门的 TRTLLM 相关路径：

```text
trtllm_batch_context_with_kv_cache
trtllm_batch_decode_with_kv_cache
```

位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:18`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1740`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1878`

FP8 prefill 中，如果需要把 FP8 KV cache 转给 TRTLLM path，会使用：

```text
trtllm_prefill_attn_kvfp8_dequant(..., k_scale, v_scale)
```

位置：`vllm/vllm/v1/attention/backends/flashinfer.py:161`

内部会执行：

```text
fp8_k * k_scale
fp8_v * v_scale
```

位置：`vllm/vllm/v1/attention/backends/flashinfer.py:135`

### 13.3 NVFP4 KV cache

FlashInfer 对 NVFP4 有特殊逻辑：

```text
kv_cache_dtype == "nvfp4"
```

会设置：

```text
is_kvcache_nvfp4 = True
```

位置：`vllm/vllm/v1/attention/backends/flashinfer.py:645`

NVFP4 要求：

```text
trtllm-gen FP4 FMHA kernels
sm100f / Blackwell 相关能力
FP4 data + FP8 block scales
```

相关位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:647`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1580`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1698`

NVFP4 和普通 FP8 最大区别是：

```text
普通 FP8：
  通常依赖 layer._k_scale / layer._v_scale。

NVFP4：
  KV cache 中还包含 packed fp4 data 和 block scales，
  decode / prefill 需要把 data 和 block scales 分开传入 kernel。
```

### 13.4 FlashInfer 也会消费 k/v scale

FlashInfer impl forward 中，量化 KV cache 会使用：

```text
layer._k_scale_float
layer._v_scale_float
layer._k_scale
layer._v_scale
```

相关位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:1473`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1651`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1802`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1927`

这说明 backend 并不是只看 cache dtype，它还必须拿到 Attention layer 上已经准备好的 scale。

---

## 14. TurboQuant backend 与专用 KV cache layout

源码：`vllm/vllm/v1/attention/backends/turboquant_attn.py`

TurboQuant 是最明显的“量化格式决定 backend”的例子。

### 14.1 支持的 dtype

TurboQuantAttentionBackend 支持：

```text
turboquant_k8v4
turboquant_4bit_nc
turboquant_k3v4_nc
turboquant_3bit_nc
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:100`

并且：

```python
supports_kv_cache_dtype(cls, kv_cache_dtype)
```

只接受 `turboquant_` 开头的 dtype。

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:164`

### 14.2 专用 cache shape

TurboQuant 的 KV cache shape 是：

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:132`

它没有普通 backend 的 leading `2` 维度。

普通 KV cache 常见形态是：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

TurboQuant 则是把 K 和 V 打包在同一个 slot 里：

```text
[key_packed | value_packed | padding]
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:139`

### 14.3 执行模型

TurboQuant 注释里直接说明：

```text
Prefill：
  对未压缩 K/V 做普通 attention，随后量化 K 并把 K+V 存入 combined cache slot。

Decode：
  从 compressed cache 计算 attention scores，解包 FP16 values，softmax 后加权求和。
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:5`

因此 TurboQuant 不是普通 backend 上加一个 dtype，而是：

```text
专用 cache layout + 专用 store kernel + 专用 decode kernel。
```

---

## 15. Attention.forward 中 scale 和 backend 如何协作

Attention layer 的 forward 入口在：

```python
Attention.forward(...)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:774`

### 15.1 scale 初始化

Attention 初始化时会注册：

```text
_q_scale
_k_scale
_v_scale
_prob_scale
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:98`

如果 quant_method 需要从 checkpoint 加载 KV scale，会通过：

```text
BaseKVCacheMethod.create_weights()
```

注册临时参数：

```text
q_scale
k_scale
v_scale
prob_scale
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:57`

加载后再在：

```text
BaseKVCacheMethod.process_weights_after_loading()
```

中写回 `_q_scale / _k_scale / _v_scale / _prob_scale`。

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:74`

### 15.2 runtime 计算 scale

如果 `calculate_kv_scales` 开启，Attention 可以通过：

```python
calc_kv_scales(query, key, value)
```

计算：

```text
_q_scale = abs(query).max() / q_range
_k_scale = abs(key).max() / k_range
_v_scale = abs(value).max() / v_range
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:546`

### 15.3 backend forward 消费 scale

最终 backend impl 的 forward 签名大致是：

```python
forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)
```

抽象定义：`vllm/vllm/v1/attention/backend.py:838`

这里直接把 `layer` 传给 backend，因此 backend 可以访问：

```text
layer._q_scale
layer._k_scale
layer._v_scale
layer._q_scale_float
layer._k_scale_float
layer._v_scale_float
```

这就是为什么 scale 存在 Attention layer 上，而不是单独塞进 metadata。

---

## 16. forward_includes_kv_cache_update 的意义

`AttentionBackend` 有一个字段：

```python
forward_includes_kv_cache_update
```

位置：`vllm/vllm/v1/attention/backend.py:66`

它表示：

```text
backend.forward() 是否自己负责 KV cache update。
```

例如：

```text
FlashAttentionBackend.forward_includes_kv_cache_update = False
TritonAttentionBackend.forward_includes_kv_cache_update = False
TurboQuantAttentionBackend.forward_includes_kv_cache_update = False
```

当它为 False 时，Attention layer / backend impl 需要在合适位置显式调用 cache update kernel。

这对量化很关键，因为写 KV cache 时必须知道：

```text
slot_mapping
kv_cache_dtype
k_scale
v_scale
per-token-head scale cache
NVFP4 block scales
TurboQuant packed slot layout
```

---

## 17. block table / slot mapping 与量化 backend 的关系

量化不改变 block table / slot mapping 的基本语义。

```text
block table：
  请求 → KV block ids。

slot mapping：
  token → KV cache slot。
```

但是量化会改变：

```text
slot 对应的物理存储格式。
```

例如：

```text
普通 FP16 KV cache：
  slot 中存 K/V 的 fp16/bf16 数值。

FP8 KV cache：
  slot 中存 fp8 K/V，读写时带 k/v scale。

per-token-head KV cache：
  slot 中除量化 K/V 外，还需要为每个 token/head 管理 scale。

NVFP4 KV cache：
  slot 中包含 packed fp4 data + fp8 block scales。

TurboQuant：
  slot 是 key_packed + value_packed 的 combined layout。
```

所以：

```text
调度层仍然只分配 block；
backend 决定一个 slot 内部如何解释量化数据。
```

---

## 18. cascade attention / prefix / sliding window 与量化

attention metadata builder 会把：

```text
common_prefix_len
block_table
slot_mapping
seq_lens
query_start_loc
sliding_window
```

翻译成 backend-specific metadata。

例如 FlashAttention / Triton metadata 中都有：

```text
use_cascade
common_prefix_len
prefix_kv_lens
suffix_kv_lens
```

位置：

- `vllm/vllm/v1/attention/backends/flash_attn.py:254`
- `vllm/vllm/v1/attention/backends/triton_attn.py:84`

量化本身不改变 cascade 的语义，但它会影响：

```text
1. cascade 路径能不能读取该量化 KV cache；
2. prefix KV 是否需要先 dequant；
3. backend 是否支持 sliding window + 当前 kv_cache_dtype；
4. block_size / layout 是否满足 kernel 要求。
```

因此某些 backend 可能单独支持 FP8，也单独支持 sliding window，但不一定支持二者组合；这种组合限制会在 `supports_combination()` 中拒绝。

---

## 19. MLA / Sparse MLA 与量化 KV cache

MLA 的 backend 抽象不同于普通 AttentionImpl。

MLA 抽象定义在：

```text
MLAAttentionImpl
SparseMLAAttentionImpl
```

位置：

- `vllm/vllm/v1/attention/backend.py:895`
- `vllm/vllm/v1/attention/backend.py:986`

MLA 的 KV cache update 使用：

```python
ops.concat_and_cache_mla(..., kv_cache_dtype=kv_cache_dtype, scale=k_scale)
```

位置：`vllm/vllm/v1/attention/backend.py:976`

这里注意：

```text
MLA cache update 传的是 k_scale，
不是普通 attention 的 k_scale + v_scale 双 scale。
```

因为 MLA cache 存的是压缩后的：

```text
kv_c_normed + k_pe
```

而不是普通 MHA 的 K/V 两份张量。

### 19.1 MLA backend 选择也受 kv_cache_dtype 影响

CUDA 的 `_get_backend_priorities()` 在 `use_mla=True` 时会进入 MLA backend 列表。

位置：`vllm/vllm/platforms/cuda.py:92`

如果是 Blackwell sparse MLA 且 KV cache 量化，会优先 FlashInfer sparse MLA。

位置：`vllm/vllm/platforms/cuda.py:97`

这说明：

```text
MLA + 量化 KV cache 是一个单独的 backend 选择问题，
不能直接套普通 attention backend 的经验。
```

---

## 20. 用户显式指定 backend 时会发生什么

用户可以通过 attention backend 配置指定 backend。

`get_attn_backend()` 会把：

```text
vllm_config.attention_config.backend
```

传给 platform。

位置：`vllm/vllm/v1/attention/selector.py:106`

CUDA 平台逻辑是：

```text
如果 selected_backend 不为空：
  只校验这个 backend；
  如果 validate_configuration() 返回 invalid reasons，直接 ValueError；
  不会自动 fallback。
```

位置：`vllm/vllm/platforms/cuda.py:381`

因此如果用户强制：

```text
--attention-backend FLASH_ATTN
--kv-cache-dtype fp8_e5m2
```

但 FlashAttention 不支持这个组合，就会报错，而不是自动改用 Triton / FlashInfer。

---

## 21. backend registry 的作用

attention backend 枚举在：`vllm/vllm/v1/attention/backends/registry.py:34`

它列出了：

```text
FLASH_ATTN
TRITON_ATTN
FLASHINFER
FLEX_ATTENTION
TURBOQUANT
FLASHINFER_MLA
TRITON_MLA
CUTLASS_MLA
FLASHMLA
FLASHMLA_SPARSE
FLASHINFER_MLA_SPARSE
CPU_ATTN
CUSTOM
```

每个枚举值对应一个 class path。

例如：

```text
FLASH_ATTN → vllm.v1.attention.backends.flash_attn.FlashAttentionBackend
FLASHINFER → vllm.v1.attention.backends.flashinfer.FlashInferBackend
TRITON_ATTN → vllm.v1.attention.backends.triton_attn.TritonAttentionBackend
TURBOQUANT → vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend
```

位置：`vllm/vllm/v1/attention/backends/registry.py:44`

它还支持：

```python
register_backend(...)
```

位置：`vllm/vllm/v1/attention/backends/registry.py:220`

这意味着第三方 backend 也可以加入同一套 selection / validation 机制，只要实现 `AttentionBackend` 抽象。

---

## 22. 和 KV cache 初始化的关系

backend 不只决定 forward kernel，还决定 KV cache shape。

抽象接口：

```python
get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)
```

位置：`vllm/vllm/v1/attention/backend.py:88`

不同 backend 的 KV cache shape 可能不同：

```text
FlashAttention / Triton / FlashInfer：
  通常是 K/V 分离维度 + block/page/head/head_size。

TurboQuant：
  K+V packed 到同一个 slot，没有 leading 2 维度。

NVFP4：
  最后一维不是普通 head_size，而是 packed data + block scales 所需 full dim。
```

因此 backend selection 必须先于 KV cache 物理分配。

如果 backend 要求 HND layout，selector 会设置全局 KV cache layout override；随后 KV cache allocation 按这个 layout 创建。

---

## 23. 和 KV connector 的关系

`AttentionSelectorConfig` 中有：

```text
use_kv_connector
```

位置：`vllm/vllm/v1/attention/selector.py:34`

它来自：

```text
vllm_config.kv_transfer_config.is_kv_transfer_instance
```

位置：`vllm/vllm/v1/attention/selector.py:85`

backend 会通过：

```python
supports_kv_connector()
```

声明是否支持 KV connector。

位置：`vllm/vllm/v1/attention/backend.py:276`

这和量化的关系是：

```text
KV connector 需要理解 KV cache 的物理 layout；
量化 KV cache 改变 layout / dtype / scale；
所以某些 backend + quantized KV + connector 的组合可能不合法。
```

---

## 24. 量化影响 backend 的几类典型场景

### 24.1 fp8 KV cache 导致 FlashAttention 被过滤

如果配置：

```text
kv_cache_dtype = fp8_e5m2
```

FlashAttention 的 `supports_kv_cache_dtype()` 不接受它。

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:183`

因此 selector 会尝试 FlashInfer / Triton。

### 24.2 per-token-head scale 导致 backend 被过滤

如果：

```text
use_per_head_quant_scales=True
```

但 backend：

```python
supports_per_head_quant_scales() == False
```

则 `validate_configuration()` 会加入：

```text
per-head quant scales not supported
```

位置：`vllm/vllm/v1/attention/backend.py:350`

### 24.3 TurboQuant dtype 只匹配 TurboQuant backend

如果：

```text
kv_cache_dtype = turboquant_k3v4_nc
```

普通 FlashAttention / Triton / FlashInfer 不支持，TurboQuant backend 支持。

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:164`

### 24.4 NVFP4 需要 FlashInfer / TRTLLM 特殊路径

如果：

```text
kv_cache_dtype = nvfp4
```

FlashInfer 会进入 `is_kvcache_nvfp4` 路径，并要求相关 TRTLLM / Blackwell 能力。

位置：`vllm/vllm/v1/attention/backends/flashinfer.py:645`

### 24.5 用户指定 block size 可能排除更高优先级 backend

如果用户显式指定 `--block-size`，backend 必须：

```text
supports_block_size(block_size)
```

CUDA 平台会在高优先级 backend 因 block_size 不支持被排除时 warning。

位置：`vllm/vllm/platforms/cuda.py:438`

量化 KV cache 常常对 block size / page size 更敏感，因此这个限制更容易触发。

---

## 25. Attention backend 和权重量化的边界

容易混淆的一点是：

```text
AWQ / GPTQ / FP8 Linear 权重量化
```

通常影响的是：

```text
Linear / MLP / QKV projection / output projection / MoE expert GEMM
```

它们不直接决定 attention backend。

attention backend 更关心的是：

```text
query/key/value tensor dtype
KV cache dtype
KV cache scale
KV cache layout
attention metadata
```

但是两者会在模型 forward 中相遇：

```text
QKV projection 可能是量化 Linear；
它输出 query/key/value；
Attention backend 再把 key/value 写入量化或非量化 KV cache。
```

因此：

```text
权重量化影响 QKV 从哪里来；
KV cache 量化影响 QKV 进入 cache 后如何存和读。
```

---

## 26. 容易疑惑的点

### 26.1 `--quantization fp8` 是否一定会使用 FP8 KV cache？

不一定。

`--quantization` 主要描述权重 / activation 量化方法。

KV cache 是否 FP8 主要看：

```text
--kv-cache-dtype
checkpoint kv_cache_scheme
cache_config.cache_dtype
```

### 26.2 FP8 权重和 FP8 KV cache 是一回事吗？

不是。

```text
FP8 权重：
  静态模型参数低精度存储 / 计算。

FP8 KV cache：
  推理过程中产生的历史 key/value 低精度存储。
```

它们使用的 scale、kernel、生命周期都不同。

### 26.3 为什么同样的 kv_cache_dtype 在不同 GPU 上 backend 不一样？

因为 backend selection 同时检查：

```text
compute capability
FlashAttention / FlashInfer 版本
head size
block size
MLA / sparse / sink 等组合能力
```

例如 FlashAttention 的 FP8 KV cache 支持在 CUDA 上要求 FA3 + capability family 90。

### 26.4 per-token-head scale 为什么不是普通 k/v scale？

普通 FP8 per-tensor scale 是：

```text
一个 layer 的 k_scale / v_scale
```

per-token-head scale 是：

```text
每个 token、每个 KV head 都可能有独立 scale
```

所以 backend 需要额外的 scale cache，不能只读 `layer._k_scale` / `layer._v_scale`。

### 26.5 backend metadata 是否包含 k_scale / v_scale？

通常不包含。

backend forward 会拿到整个 `layer`，然后从 layer 上读取：

```text
_k_scale
_v_scale
_q_scale
```

metadata 更多负责：

```text
block_table
slot_mapping
seq_lens
query_start_loc
cascade / prefix / decode 信息
```

### 26.6 为什么 TurboQuant 要单独 backend？

因为它不只是换 dtype，而是换了 KV cache 物理 layout：

```text
普通 backend：K/V 分离。
TurboQuant：K+V packed 到一个 slot。
```

普通 backend 无法解释这种 slot。

### 26.7 显式指定 backend 会自动 fallback 吗？

不会。

用户指定 backend 后，如果当前量化配置不支持，会直接报错。

---

## 27. 最小心智模型

可以把 attention backend selection 记成：

```text
AttentionSelectorConfig 是问题描述；
AttentionBackend.validate_configuration() 是能力过滤；
current_platform.get_attn_backend_cls() 是优先级决策；
backend impl.forward() 是真正消费量化 KV cache 的地方。
```

量化相关最小主线是：

```text
cache_config.cache_dtype
  → Attention.kv_cache_dtype
  → get_attn_backend(kv_cache_dtype, use_mla, use_sparse, ...)
  → backend.supports_kv_cache_dtype()
  → backend.get_kv_cache_shape()
  → backend forward / cache update 使用 k_scale、v_scale、slot_mapping
```

再压缩成一句话：

```text
KV cache 量化决定 attention backend 能不能选、KV cache 怎么分配、cache 写入时怎么量化、cache 读取时怎么反量化。
```

---

## 28. 总结

量化影响 attention backend 的核心点可以归纳为：

```text
1. kv_cache_dtype 进入 AttentionSelectorConfig；
2. backend 通过 supported_kv_cache_dtypes / supports_kv_cache_dtype 声明支持范围；
3. per-token-head scale 需要 backend 显式 supports_per_head_quant_scales；
4. FP8 / NVFP4 / TurboQuant 会改变 KV cache shape、page size 或 layout；
5. FlashAttention / FlashInfer / Triton / TurboQuant 对量化 KV cache 的支持范围不同；
6. MLA / sparse MLA 会走另一组 backend，并且量化 KV cache 会改变优先级；
7. Attention layer 上的 _k_scale / _v_scale / _q_scale 是 backend 执行量化 cache 读写的关键输入；
8. 用户显式指定 backend 时，量化组合不合法会直接报错。
```

如果只记一句话：

```text
attention backend 选择的本质，是在模型形态、硬件能力、KV cache 量化格式和 kernel layout 约束之间找一个可执行且优先级最高的实现。
```
