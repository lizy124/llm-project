# 07. KV cache quantization 如何工作？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\config\cache.py`
- `E:\lizy\code\vllm-project\vllm\vllm\engine\arg_utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\utils\torch_utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\attention\attention.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\quantization\kv_cache.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\kv_cache_interface.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\attention\backend.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\attention\selector.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\attention\backends\flash_attn.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\attention\backends\triton_attn.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\attention\backends\turboquant_attn.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_model_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu\attn_utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\distributed\kv_transfer\kv_connector\utils.py`

本问题关注：`kv_cache_dtype` 从哪里来；它如何变成 `KVQuantMode`、KV cache tensor dtype / shape / page size；scale 是怎么加载或动态计算的；attention backend 如何决定支持哪些 KV quant；KV cache layout、paged cache、prefix cache、external KV transfer 和量化格式是什么关系。

---

## 1. 一句话回答

KV cache quantization 压缩的是 **历史 K/V cache 的存储和读取路径**，不是模型权重。

主链路是：

```text
CLI / EngineArgs
  → CacheConfig.cache_dtype
  → Attention.kv_cache_dtype / kv_cache_torch_dtype
  → get_kv_quant_mode()
  → KVCacheSpec.page_size_bytes
  → Worker 分配 raw int8 buffer 并 reshape 成 backend 需要的 KV cache tensor
  → Attention.forward()
  → do_kv_cache_update() 把本轮 K/V 写入量化 cache
  → backend attention kernel 读取量化 KV cache 并用 scale dequant / fused compute
```

所以：

```text
weight quantization：
  影响 Linear / MoE 权重、activation、matmul kernel。

KV cache quantization：
  影响 attention 历史 K/V 的存储 dtype、page size、scale、cache layout 和 attention backend kernel。
```

---

## 2. KV cache quantization 的核心对象

这条链路里有五个关键对象：

```text
1. CacheConfig.cache_dtype
   → 用户或模型配置指定的 KV cache 存储格式。

2. Attention.kv_cache_dtype / kv_cache_torch_dtype
   → 每个 attention layer 看到的 KV cache dtype 字符串和 torch dtype。

3. KVQuantMode
   → backend / kernel 使用的枚举，不再直接用字符串判断。

4. KVCacheSpec
   → Scheduler / Worker 用来计算 page size、block 数、cache group 的规格。

5. AttentionBackend / AttentionImpl
   → 真正决定 cache shape、写 cache kernel、读 cache kernel 的实现。
```

最小心智模型是：

```text
cache_dtype 是用户语义；
KVQuantMode 是 kernel dispatch 语义；
KVCacheSpec 是内存账本语义；
AttentionBackend 是物理布局和计算语义。
```

---

## 3. kv_cache_dtype 从哪里配置

### 3.1 CLI / EngineArgs 入口

`EngineArgs` 里有：

```python
kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
calculate_kv_scales: bool = CacheConfig.calculate_kv_scales
kv_cache_dtype_skip_layers: list[str] = ...
```

位置：`vllm/vllm/engine/arg_utils.py:438`、`vllm/vllm/engine/arg_utils.py:680`

构造 `CacheConfig` 时会先解析 `auto`：

```python
resolved_cache_dtype = resolve_kv_cache_dtype_string(
    self.kv_cache_dtype, model_config
)
...
cache_dtype=resolved_cache_dtype
calculate_kv_scales=self.calculate_kv_scales
kv_cache_dtype_skip_layers=self.kv_cache_dtype_skip_layers
```

位置：`vllm/vllm/engine/arg_utils.py:1832` 到 `vllm/vllm/engine/arg_utils.py:1853`

这说明用户侧的 `--kv-cache-dtype` 最终进入的是：

```text
CacheConfig.cache_dtype
```

### 3.2 CacheConfig 支持哪些 dtype

`CacheDType` 当前包含：

```text
auto
float16 / bfloat16
fp8 / fp8_e4m3 / fp8_e5m2 / fp8_inc / fp8_ds_mla
int8_per_token_head
fp8_per_token_head
nvfp4
turboquant_k8v4
turboquant_4bit_nc
turboquant_k3v4_nc
turboquant_3bit_nc
```

位置：`vllm/vllm/config/cache.py:19` 到 `vllm/vllm/config/cache.py:35`

其中：

```text
float16 / bfloat16：
  非量化 KV cache，只是显式指定 KV cache dtype。

fp8*：
  FP8 KV cache，常见路径是 per-tensor scale。

int8_per_token_head / fp8_per_token_head：
  每个 token、每个 KV head 动态计算 scale。

nvfp4：
  packed FP4 data + FP8 block scales。

turboquant_*：
  TurboQuant 自定义 K/V 压缩 cache layout。
```

### 3.3 auto 可能来自模型 quantization_config

`resolve_kv_cache_dtype_string()` 会在 `kv_cache_dtype == "auto"` 时检查模型 `hf_config.quantization_config`：

```text
hf_config.quantization_config
  → kv_cache_scheme / kv_cache_quant_algo
  → 映射到 fp8 / nvfp4 / auto
```

位置：`vllm/vllm/utils/torch_utils.py:373` 到 `vllm/vllm/utils/torch_utils.py:391`

因此 `auto` 不一定最终等于模型 dtype：

```text
普通模型：
  auto → downstream 使用 model dtype。

带 kv_cache_quant_algo 的量化模型：
  auto → 可能解析成 fp8 / nvfp4。
```

---

## 4. CacheConfig 如何提示量化语义

`CacheConfig._validate_cache_dtype()` 会根据 dtype 输出不同日志：

```python
if kv_cache_uses_per_token_head_scales(cache_dtype):
    ... Dynamic per-token-head scales will be computed at runtime.
elif is_quantized_kv_cache(cache_dtype):
    ... may cause accuracy drop without a proper scaling factor
```

位置：`vllm/vllm/config/cache.py:268` 到 `vllm/vllm/config/cache.py:286`

这说明 vLLM 把 KV cache quantization 分成两类：

```text
1. 动态 scale 路径
   int8_per_token_head / fp8_per_token_head。

2. 非动态 per-token-head 路径
   fp8 / nvfp4 / turboquant 等。
```

`calculate_kv_scales` 仍然存在，但已经标记 deprecated：

```text
--calculate-kv-scales：
  只针对 fp8 KV cache 的动态 k_scale / v_scale 计算；
  未来会移除；
  默认从 checkpoint 加载 scale，没有则用 1.0。
```

位置：`vllm/vllm/config/cache.py:110` 到 `vllm/vllm/config/cache.py:114`、`vllm/vllm/config/cache.py:256` 到 `vllm/vllm/config/cache.py:266`

---

## 5. Attention 层如何接住 kv_cache_dtype

在 `Attention.__init__()` 中：

```python
if cache_config is not None:
    kv_cache_dtype = cache_config.cache_dtype
    calculate_kv_scales = cache_config.calculate_kv_scales
else:
    kv_cache_dtype = "auto"
    calculate_kv_scales = False
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:238` 到 `vllm/vllm/model_executor/layers/attention/attention.py:245`

之后会保存为：

```python
self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(...)
self.kv_cache_dtype = kv_cache_dtype
self.calculate_kv_scales = calculate_kv_scales
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:291` 到 `vllm/vllm/model_executor/layers/attention/attention.py:295`

这里要区分：

```text
kv_cache_dtype：
  字符串，用于选择 backend、KVQuantMode、kernel 分支。

kv_cache_torch_dtype：
  torch dtype，用于 KVCacheSpec 和实际 tensor view。
```

---

## 6. checkpoint 的 kv_cache_scheme 如何影响 Attention

如果量化 checkpoint 声明了 `kv_cache_scheme`，并且用户没有显式指定 KV cache dtype：

```python
kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
if kv_cache_scheme is not None and kv_cache_dtype == "auto":
    kv_cache_dtype = "fp8"
    calculate_kv_scales = False
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:246` 到 `vllm/vllm/model_executor/layers/attention/attention.py:259`

含义是：

```text
checkpoint 声明 KV cache quant scheme
  → 默认启用 fp8 KV cache
  → 不再 on-the-fly calculate_kv_scales
  → 优先使用 checkpoint scale
```

如果用户显式指定了 `--kv-cache-dtype bfloat16` 这类值，则用户配置优先。

---

## 7. kv_cache_dtype_skip_layers 做什么

`CacheConfig.kv_cache_dtype_skip_layers` 支持按层跳过 KV cache quantization：

```text
- 层号："0" / "2" / "4"
- attention 类型："sliding_window"
```

位置：`vllm/vllm/config/cache.py:115` 到 `vllm/vllm/config/cache.py:117`

在 `Attention.__init__()` 中，如果命中 skip：

```python
kv_cache_dtype = "auto"
calculate_kv_scales = False
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:266` 到 `vllm/vllm/model_executor/layers/attention/attention.py:289`

这意味着：

```text
全局可以启用 KV cache quantization；
个别层可以退回 native KV cache；
这些层会形成不同的 KV cache spec / cache group / cache tensor。
```

TurboQuant 还会自动把某些边界层加入 skip：

```python
if resolved_cache_dtype.startswith("turboquant_"):
    boundary = TurboQuantConfig.get_boundary_skip_layers(model_config)
    cache_config.kv_cache_dtype_skip_layers = sorted(...)
```

位置：`vllm/vllm/engine/arg_utils.py:1862` 到 `vllm/vllm/engine/arg_utils.py:1871`

---

## 8. KVQuantMode 是什么

`KVQuantMode` 定义在 `v1/kv_cache_interface.py`：

```python
class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    NVFP4 = 4
```

位置：`vllm/vllm/v1/kv_cache_interface.py:33` 到 `vllm/vllm/v1/kv_cache_interface.py:45`

映射规则是：

```python
int8_per_token_head → INT8_PER_TOKEN_HEAD
fp8_per_token_head  → FP8_PER_TOKEN_HEAD
nvfp4               → NVFP4
fp8*                → FP8_PER_TENSOR
其他                → NONE
```

位置：`vllm/vllm/v1/kv_cache_interface.py:60` 到 `vllm/vllm/v1/kv_cache_interface.py:70`

所以：

```text
KVQuantMode 不是用户参数；
它是从 kv_cache_dtype 派生出的 backend/kernel dispatch 标识。
```

---

## 9. KVCacheSpec 如何记录量化格式

`Attention.get_kv_cache_spec()` 会把 layer 的 KV cache dtype 转成 spec：

```python
quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
...
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

`SlidingWindowSpec` 同样携带：

```text
dtype
kv_quant_mode
block_size
num_kv_heads
head_size / head_size_v
sliding_window
```

这说明 Scheduler / Worker 不只知道“有多少 block”，还知道：

```text
每个 block 里一页 KV cache 到底占多少 bytes。
```

---

## 10. page_size_bytes 如何随量化变化

`AttentionSpec.page_size_bytes` 是 KV cache 内存预算的核心。

普通 attention 的基础公式是：

```text
real_page_size_bytes
  = 2 * block_size * num_kv_heads * head_size * sizeof(dtype)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:183` 到 `vllm/vllm/v1/kv_cache_interface.py:201`

这里的 `2` 表示：

```text
K cache + V cache
```

### 10.1 per-token-head scale 会额外预算 scale 内存

如果 `kv_quant_mode.is_per_token_head`：

```python
real_page_size += (
    2 * block_size * num_kv_heads * sizeof(float32)
)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:169` 到 `vllm/vllm/v1/kv_cache_interface.py:177`

含义：

```text
每个 token、每个 KV head 都需要 K scale 和 V scale；
scale 是 float32；
scale 内存从 raw KV cache allocation 中切出来。
```

### 10.2 NVFP4 会改变最后一维

NVFP4 的 page size 使用：

```python
full_dim = nvfp4_kv_cache_full_dim(head_size)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:185` 到 `vllm/vllm/v1/kv_cache_interface.py:194`

`nvfp4_kv_cache_full_dim()` 的定义是：

```python
return head_size // 2 + head_size // 16
```

位置：`vllm/vllm/utils/torch_utils.py:413` 到 `vllm/vllm/utils/torch_utils.py:415`

对应含义：

```text
head_size // 2：
  两个 FP4 value packed 到一个 byte。

head_size // 16：
  每 16 个元素一个 FP8 block scale。
```

### 10.3 FullAttentionSpec / SlidingWindowSpec 支持 head_size_v

对于 V head size 不等于 K head size 的模型，`FullAttentionSpec.real_page_size_bytes` 用：

```text
block_size * num_kv_heads * (head_size + head_size_v) * sizeof(dtype)
```

位置：`vllm/vllm/v1/kv_cache_interface.py:308` 到 `vllm/vllm/v1/kv_cache_interface.py:328`

这比基础 `AttentionSpec` 更精确，因为 K/V 最后一维可能不同。

---

## 11. Worker 如何分配量化 KV cache

Worker 侧分配不是直接分配目标 dtype，而是先分配 raw bytes：

```python
torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:7033` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:7050`

随后 `_reshape_kv_cache_tensors()` 根据 spec 和 backend reshape：

```text
KVCacheSpec.page_size_bytes
  → num_blocks
  → attn_backend.get_kv_cache_shape(...)
  → attn_backend.get_kv_cache_stride_order()
  → _reshape_attention_kv_cache(...)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:7072` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:7150`

这说明：

```text
KV cache allocation 是 byte-level；
KV cache view 才体现 dtype / shape / layout。
```

---

## 12. Attention backend 如何声明支持哪些 KV quant

所有 backend 都通过 `AttentionBackend` 统一声明能力：

```python
supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

def supports_kv_cache_dtype(cls, kv_cache_dtype): ...

def validate_configuration(...):
    if not cls.supports_kv_cache_dtype(kv_cache_dtype):
        invalid_reasons.append("kv_cache_dtype not supported")
```

位置：`vllm/vllm/v1/attention/backend.py:55` 到 `vllm/vllm/v1/attention/backend.py:173`、`vllm/vllm/v1/attention/backend.py:307` 到 `vllm/vllm/v1/attention/backend.py:375`

因此 backend 是否能用，取决于一组条件：

```text
head_size
activation dtype
kv_cache_dtype
block_size
MLA / non-MLA
sink / sparse / mm_prefix
compute capability
attention type
KV connector
```

### 12.1 FlashAttention backend

FlashAttention 默认声明支持：

```text
auto / float16 / bfloat16
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:68` 到 `vllm/vllm/v1/attention/backends/flash_attn.py:74`

但 `supports_kv_cache_dtype()` 对 FP8 有额外判断：

```python
if kv_cache_dtype in ("fp8", "fp8_e4m3"):
    if current_platform.is_xpu():
        return True
    return get_flash_attn_version() == 3 and SM90
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:183` 到 `vllm/vllm/v1/attention/backends/flash_attn.py:193`

也就是说：

```text
FlashAttention 的 FP8 KV cache 路径依赖 FA3 / H100 类设备能力；
不是所有 GPU 上指定 fp8 都会走 FlashAttention。
```

### 12.2 Triton backend

Triton backend 支持：

```text
auto / float16 / bfloat16
fp8 / fp8_e4m3 / fp8_e5m2
int8_per_token_head / fp8_per_token_head
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:248` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:263`

同时在 CUDA 上对 FP8 有 compute capability 检查：

```text
Triton FP8 KV cache 要求 SM89+；
否则提示改用 float16 / bfloat16。
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:467` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:489`

### 12.3 TurboQuant backend

TurboQuant backend 只支持：

```text
turboquant_k8v4
turboquant_4bit_nc
turboquant_k3v4_nc
turboquant_3bit_nc
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:90` 到 `vllm/vllm/v1/attention/backends/turboquant_attn.py:105`

它的 `supports_kv_cache_dtype()` 是：

```python
return kv_cache_dtype.startswith("turboquant_")
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:163` 到 `vllm/vllm/v1/attention/backends/turboquant_attn.py:168`

---

## 13. Attention selector 如何选择 backend

`Attention.__init__()` 调用：

```python
self.attn_backend = get_attn_backend(
    head_size,
    dtype,
    kv_cache_dtype,
    use_mla=False,
    ...
    use_per_head_quant_scales=use_per_head_quant_scales,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:318` 到 `vllm/vllm/model_executor/layers/attention/attention.py:328`

`get_attn_backend()` 会构造 `AttentionSelectorConfig`：

```text
head_size
dtype
kv_cache_dtype
block_size
use_mla
has_sink
use_per_head_quant_scales
attn_type
use_kv_connector
```

位置：`vllm/vllm/v1/attention/selector.py:54` 到 `vllm/vllm/v1/attention/selector.py:110`

然后交给 platform 选择 backend：

```python
current_platform.get_attn_backend_cls(...)
```

位置：`vllm/vllm/v1/attention/selector.py:113` 到 `vllm/vllm/v1/attention/selector.py:144`

这意味着：

```text
kv_cache_dtype 不只是改变 tensor dtype；
它会参与 attention backend 选择。
```

---

## 14. KV cache layout 如何决定

Backend 会提供：

```python
get_kv_cache_shape(...)
get_kv_cache_stride_order(...)
```

位置：`vllm/vllm/v1/attention/backend.py:87` 到 `vllm/vllm/v1/attention/backend.py:147`

常见 layout 是：

```text
NHD：
  (num_blocks, 2, block_size, num_kv_heads, head_size)

HND：
  逻辑 shape 类似，但物理 stride 把 num_kv_heads 放到更靠前的位置。
```

`get_kv_cache_layout()` 的优先级：

```text
1. backend / code override；
2. 环境变量 VLLM_KV_CACHE_LAYOUT；
3. KV connector 要求的 layout；
4. 默认 NHD。
```

位置：`vllm/vllm/v1/attention/backends/utils.py:78` 到 `vllm/vllm/v1/attention/backends/utils.py:115`

KV connector 里默认逻辑是：

```text
如果 connector 声明 required layout，用它；
否则默认 NHD。
```

位置：`vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:34` 到 `vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:47`

所以 layout 和 quant mode 的关系是：

```text
quant mode 决定每个 slot 里存什么；
layout 决定这些 slot 在内存中如何排列；
二者都会影响 backend kernel。
```

---

## 15. FP8 per-tensor KV cache 如何写入和读取

### 15.1 scale 从哪里来

`BaseKVCacheMethod.create_weights()` 会给 attention layer 创建：

```python
layer.q_scale
layer.k_scale
layer.v_scale
layer.prob_scale
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:57` 到 `vllm/vllm/model_executor/layers/quantization/kv_cache.py:69`

这些 scale 初始是 `-1.0` sentinel，checkpoint 加载时会覆盖。

加载后 `process_weights_after_loading()` 会把它们转成运行时使用的：

```text
layer._q_scale / _q_scale_float
layer._k_scale / _k_scale_float
layer._v_scale / _v_scale_float
layer._prob_scale / _prob_scale_float
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:74` 到 `vllm/vllm/model_executor/layers/quantization/kv_cache.py:197`

### 15.2 checkpoint 没有 scale 时怎么办

如果是量化 KV cache 且不动态计算 scale：

```text
有 k_scale / v_scale：
  使用 checkpoint 中的 scale。

没有 k_scale / v_scale：
  默认使用 1.0。

只有单个 kv_scale：
  复制成 k_scale 和 v_scale。
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:100` 到 `vllm/vllm/model_executor/layers/quantization/kv_cache.py:147`

如果 FP8 e4m3 用了 1.0 scale，会 warning：

```text
Using KV cache scaling factor 1.0 for fp8_e4m3...
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:147` 到 `vllm/vllm/model_executor/layers/quantization/kv_cache.py:152`

### 15.3 写 KV cache

在 `Attention.forward()` 里，如果 backend 的 forward 不包含 KV cache update，会先调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:500` 到 `vllm/vllm/model_executor/layers/attention/attention.py:536`

`unified_kv_cache_update()` 最终调用 backend：

```python
attn_layer.impl.do_kv_cache_update(
    attn_layer,
    key,
    value,
    kv_cache,
    layer_slot_mapping,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:713` 到 `vllm/vllm/model_executor/layers/attention/attention.py:736`

FlashAttention 写 cache 使用：

```python
reshape_and_cache_flash(
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

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:927` 到 `vllm/vllm/v1/attention/backends/flash_attn.py:960`

Triton 写 cache 使用：

```python
triton_reshape_and_cache_flash(..., self.kv_cache_dtype, layer._k_scale, layer._v_scale)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:724` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:767`

也就是说：

```text
FP8 per-tensor 写入 cache 时用 layer._k_scale / layer._v_scale 做量化。
```

### 15.4 读 KV cache

FlashAttention forward 中如果是量化 KV cache：

```python
key_cache = key_cache.view(current_platform.fp8_dtype())
value_cache = value_cache.view(current_platform.fp8_dtype())
...
k_descale = layer._k_scale.expand(descale_shape)
v_descale = layer._v_scale.expand(descale_shape)
flash_attn_varlen_func(..., k_descale=k_descale, v_descale=v_descale)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:787` 到 `vllm/vllm/v1/attention/backends/flash_attn.py:895`

Triton forward 也会把量化 cache view 成 FP8 dtype，并传：

```text
q_descale
k_descale
v_descale
kv_quant_mode
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:590` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:674`

所以：

```text
写 cache：float K/V → quantized KV cache。
读 cache：quantized KV cache + scale → attention compute。
```

---

## 16. calculate_kv_scales 动态 scale 路径

如果 `calculate_kv_scales=True`，`Attention.forward()` 开头会调用：

```python
torch.ops.vllm.maybe_calc_kv_scales(query, key, value, layer_name)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:471` 到 `vllm/vllm/model_executor/layers/attention/attention.py:474`

实际计算是：

```python
self._q_scale = abs(query).max() / q_range
self._k_scale = abs(key).max() / k_range
self._v_scale = abs(value).max() / v_range
self.calculate_kv_scales = False
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:546` 到 `vllm/vllm/model_executor/layers/attention/attention.py:554`

注意：

```text
1. 这个 scale 是 per-layer / per-tensor，不是 per-token-head。
2. 只计算一次，之后 calculate_kv_scales 会被置 False。
3. 该选项已经 deprecated。
```

---

## 17. per-token-head KV quantization 如何工作

`int8_per_token_head` 和 `fp8_per_token_head` 走 `KVQuantMode.is_per_token_head`。

### 17.1 cache shape 里给 scale 留 padding

Triton backend 的 `get_kv_cache_shape()` 会把 head_size 扩大：

```python
scale_pad = sizeof(float32) / sizeof(cache_dtype)
return (num_blocks, 2, block_size, num_kv_heads, head_size + scale_pad)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:294` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:315`

含义：

```text
每个 K/V head 的最后几个 cache dtype 元素，
实际存一个 float32 scale。
```

### 17.2 scale cache 是从 KV cache buffer 里切出来的 view

`TritonAttentionImpl._ensure_scale_caches()` 会创建：

```text
_k_scale_cache: (num_blocks, block_size, num_kv_heads)
_v_scale_cache: (num_blocks, block_size, num_kv_heads)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:377` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:432`

这些 scale cache 不是单独分配的大 tensor，而是：

```text
KV cache raw storage 的 strided float32 view。
```

### 17.3 写 cache 时动态计算每个 token/head 的 scale

per-token-head 写 cache 调用：

```python
triton_reshape_and_cache_flash_per_token_head_quant(
    key,
    value,
    key_cache,
    value_cache,
    self._k_scale_cache,
    self._v_scale_cache,
    slot_mapping,
)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:737` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:752`

因此 scale 粒度是：

```text
K scale: 每个 token、每个 KV head 一个 float32 scale。
V scale: 每个 token、每个 KV head 一个 float32 scale。
```

### 17.4 读 cache 时传 scale cache

forward 读 cache 时：

```python
k_scale_cache = self._k_scale_cache
v_scale_cache = self._v_scale_cache
...
unified_attention(..., kv_quant_mode=..., k_scale_cache=..., v_scale_cache=...)
```

位置：`vllm/vllm/v1/attention/backends/triton_attn.py:590` 到 `vllm/vllm/v1/attention/backends/triton_attn.py:674`

所以 per-token-head 路径不是使用 `layer._k_scale / layer._v_scale`，而是使用 cache 内嵌的 scale views。

### 17.5 checkpoint scale 对 per-token-head 无效

`BaseKVCacheMethod.process_weights_after_loading()` 明确处理：

```text
per-token-head quantized KV cache：
  scale 在 cache-write kernel 里动态计算；
  checkpoint scales 永远不用；
  layer._k_scale / _v_scale 置为 1.0。
```

位置：`vllm/vllm/model_executor/layers/quantization/kv_cache.py:82` 到 `vllm/vllm/model_executor/layers/quantization/kv_cache.py:94`

---

## 18. NVFP4 KV cache 有什么特殊点

NVFP4 对应：

```text
KVQuantMode.NVFP4
```

位置：`vllm/vllm/v1/kv_cache_interface.py:44`、`vllm/vllm/v1/kv_cache_interface.py:66` 到 `vllm/vllm/v1/kv_cache_interface.py:67`

其 cache 的最后一维不是原始 `head_size`，而是：

```text
head_size // 2 + head_size // 16
```

位置：`vllm/vllm/utils/torch_utils.py:413` 到 `vllm/vllm/utils/torch_utils.py:415`

vLLM 还提供 split view：

```text
nvfp4_kv_cache_split_views(kv_cache)
  → (k_data, v_data), (k_scale, v_scale)
```

位置：`vllm/vllm/utils/torch_utils.py:470` 到 `vllm/vllm/utils/torch_utils.py:499`

布局含义：

```text
每个 KV side 内部：
  [packed fp4 data | fp8 block scale]
```

约束之一：

```text
nvfp4 KV cache 不支持 MLA。
```

位置：`vllm/vllm/config/vllm.py:2160` 到 `vllm/vllm/config/vllm.py:2170`

---

## 19. TurboQuant KV cache 有什么特殊点

TurboQuant 不是标准的：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

而是自定义 combined K+V slot：

```text
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:132` 到 `vllm/vllm/v1/attention/backends/turboquant_attn.py:162`

它的 per-head per-position slot 语义是：

```text
[key_packed | value_packed | padding]
```

位置：`vllm/vllm/v1/attention/backends/turboquant_attn.py:10` 到 `vllm/vllm/v1/attention/backends/turboquant_attn.py:17`

`Attention.get_kv_cache_spec()` 对 TurboQuant 返回 `TQFullAttentionSpec`：

```python
return TQFullAttentionSpec(
    ...
    tq_slot_size=tq_config.slot_size_aligned,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:607` 到 `vllm/vllm/model_executor/layers/attention/attention.py:623`

`TQFullAttentionSpec.real_page_size_bytes` 用：

```python
block_size * num_kv_heads * tq_slot_size
```

位置：`vllm/vllm/v1/kv_cache_interface.py:340` 到 `vllm/vllm/v1/kv_cache_interface.py:355`

所以 TurboQuant 的关键差异是：

```text
不是“同样 layout，只是 dtype 变小”；
而是 K/V 被压进一个 backend 专属 slot layout。
```

---

## 20. Prefix cache 和 KV cache quantization 的关系

Prefix cache 复用的是 KV block，而不是重新解释 token 内容。

因此主关系是：

```text
prefix cache 命中
  → Scheduler / KVCacheManager 复用已有 block ids
  → Worker block table 指向这些 blocks
  → attention backend 按当前 KV cache dtype / layout 读取这些 blocks
```

KV cache quantization 影响的是：

```text
block 里面的 K/V bytes 怎么存；
读取 block 时如何 dequant / fused compute。
```

Prefix cache 影响的是：

```text
哪些 block 可以复用；
当前 token 从什么 position 继续；
slot_mapping 如何写新 token。
```

所以二者边界可以压缩成：

```text
prefix cache 管“复用哪个 block”；
KV quantization 管“block 里的 K/V 是什么格式”。
```

---

## 21. Paged KV cache 和量化格式如何结合

Paged KV cache 的基本单位仍然是 block/page：

```text
block_size tokens / page
```

量化不改变 Scheduler 的基本调度对象：

```text
Scheduler 仍然分配 block ids；
Worker 仍然维护 block table；
slot mapping 仍然把 token 映射到 cache slot。
```

变化发生在每个 page 的物理大小：

```text
float16 / bfloat16：
  page_size_bytes 大。

fp8：
  dtype size 变小，但可能需要 scale。

per-token-head：
  dtype size 变小，同时额外塞入 per-token/head scale。

nvfp4：
  data packed，同时带 fp8 block scales。

TurboQuant：
  整个 K/V slot 使用 backend 自定义压缩布局。
```

所以：

```text
Paged cache 的“分页模型”不变；
每页的 byte layout 和 kernel 解释方式随 quant mode 改变。
```

---

## 22. external KV transfer 和量化格式的关系

KV connector 主要处理：

```text
block ids
KV cache memory region
send / recv / finished_sending / finished_recving
layout requirement
```

它可以要求特定 KV cache layout：

```python
connector_cls.get_required_kvcache_layout(vllm_config)
```

位置：`vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:34` 到 `vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:47`

NIXL 等 disaggregated PD 场景可能要求 HND layout，目的是提高传输效率。

但量化格式本身仍然来自：

```text
vllm_config.cache_config.cache_dtype
KVCacheSpec.dtype
KVCacheSpec.kv_quant_mode
AttentionBackend.get_kv_cache_shape()
```

因此 external KV transfer 的关键点是：

```text
发送方和接收方必须对 KV cache dtype、shape、layout、page size 有一致理解；
否则同一段 bytes 会被错误解释。
```

---

## 23. sleep / wake_up 后为什么要处理 FP8 scales

`GPUModelRunner.post_kv_cache_wake_up()` 会调用：

```python
self.init_fp8_kv_scales()
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:935` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:936`

如果当前是量化 KV cache，它会：

```text
1. zero KV cache tensors；
2. 把 Attention / MLAAttention 的 _k_scale / _v_scale 或 k_scale / v_scale 重置为 1.0。
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:938` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:981`

原因是：

```text
wake_up 后显存可能重新分配；
scale 如果保持 0.0，会让 KV cache 值等效归零；
最终生成乱码。
```

---

## 24. 与权重量化的边界

KV cache quantization 和权重量化可以同时存在，但它们不是同一条路径。

### 24.1 权重量化路径

```text
QuantizationConfig
  → Linear / MoE / attention projection quant_method
  → 加载量化权重
  → matmul / GEMM kernel
```

### 24.2 KV cache quantization 路径

```text
CacheConfig.cache_dtype
  → Attention.kv_cache_dtype
  → KVQuantMode / KVCacheSpec
  → KV cache allocation / layout
  → do_kv_cache_update / attention backend kernel
```

### 24.3 交叉点

两者的交叉主要在：

```text
1. checkpoint 可能携带 kv_cache_scheme / k_scale / v_scale；
2. BaseKVCacheMethod 属于 quantization 目录，但作用对象是 Attention 的 KV cache scale；
3. query quantization 可能在 FP8 KV cache backend 中启用。
```

例如 `Attention.__init__()` 中，如果 backend 支持且 KV cache 是 fp8 / nvfp4，会创建 `query_quant`：

```text
supports_quant_query_input
  + kv_cache_dtype.startswith("fp8") or nvfp4
  + 非 per-token-head
  → QuantFP8 query_quant
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:432` 到 `vllm/vllm/model_executor/layers/attention/attention.py:450`

这属于 attention backend 优化，不代表 KV cache quantization 等于权重量化。

---

## 25. 完整执行链路

把前面的内容串起来：

```text
1. 用户设置 --kv-cache-dtype / 模型 quantization_config 提供 kv_cache_quant_algo
   → EngineArgs.resolve_kv_cache_dtype_string()
   → CacheConfig.cache_dtype

2. 初始化 Attention layer
   → 保存 kv_cache_dtype / kv_cache_torch_dtype
   → 根据 kv_cache_scheme / skip_layers 调整 dtype
   → 选择 attention backend
   → 初始化 KV cache scale 参数

3. 构建 KVCacheSpec
   → get_kv_quant_mode(kv_cache_dtype)
   → dtype + kv_quant_mode + block_size + head 信息
   → page_size_bytes 计算内存预算

4. Worker 初始化 KV cache
   → 按 page_size_bytes 分配 raw int8 buffer
   → backend.get_kv_cache_shape()
   → backend.get_kv_cache_stride_order()
   → reshape 成每层 kv_cache tensor view

5. 每轮 forward
   → ModelRunner 准备 block table / slot mapping
   → Attention.forward()
   → maybe_calc_kv_scales() 可选动态 per-tensor scale
   → do_kv_cache_update() 写本轮 K/V 到 cache
   → backend forward 读取历史 KV cache
   → 使用 scale / scale cache / packed layout 做 attention
```

---

## 26. 容易疑惑的点

### 26.1 KV cache quantization 会量化模型权重吗？

不会。

它只影响历史 K/V cache 的存储和 attention 读取路径。

### 26.2 `fp8`、`fp8_e4m3`、`fp8_e5m2` 都是同一种 scale 路径吗？

大体都映射到 `KVQuantMode.FP8_PER_TENSOR`，但平台支持和实际 FP8 dtype 可能不同。

例如 FlashAttention 只在特定 FA / GPU 能力下支持 FP8 KV cache，Triton 也有 SM89+ 约束。

### 26.3 per-token-head quant 会用 checkpoint 的 k_scale / v_scale 吗？

不会。

`int8_per_token_head` / `fp8_per_token_head` 的 scale 在写 cache kernel 中动态计算，checkpoint scale 会被忽略。

### 26.4 `calculate_kv_scales` 和 per-token-head 是一回事吗？

不是。

```text
calculate_kv_scales：
  deprecated；per-layer / per-tensor；只算一次。

per-token-head：
  每个 token、每个 KV head 动态 scale；写 cache 时计算。
```

### 26.5 KV cache layout 和 KV quant mode 是一回事吗？

不是。

```text
layout：
  NHD / HND，描述维度和 stride 怎么排。

quant mode：
  NONE / FP8_PER_TENSOR / INT8_PER_TOKEN_HEAD / FP8_PER_TOKEN_HEAD / NVFP4，描述 bytes 怎么解释。
```

### 26.6 prefix cache 是否需要知道 scale？

通常不直接关心。

Prefix cache 复用 block ids；block 内的 bytes 和 scale 由 attention backend 按当前 KV cache spec 解释。

### 26.7 external KV transfer 是否会重新量化？

通常不会。

KV connector 传输的是已经按本地 cache dtype / layout 存好的 KV cache memory region；接收端必须用一致的规格解释这些 bytes。

### 26.8 为什么 Worker 分配 int8 raw buffer？

因为 KV cache 的真实 view 可能是 float16、bfloat16、fp8、uint8 packed、自定义 TurboQuant slot，也可能需要 padding / strided view。

统一先按 bytes 分配，再按 spec/backend reshape 最灵活。

---

## 27. 总结

KV cache quantization 的主链路可以压缩成：

```text
cache_dtype
  → Attention.kv_cache_dtype
  → KVQuantMode
  → KVCacheSpec.page_size_bytes
  → raw KV cache bytes allocation
  → backend-specific cache shape / layout
  → do_kv_cache_update 写入量化 K/V
  → attention kernel 读取量化 K/V + scale
```

如果只记住一句话：

```text
KV cache quantization 是 attention 子系统的内存格式优化：Scheduler 仍然调度 block，Worker 仍然维护 block table / slot mapping，真正变化的是每个 KV page 里的 bytes 如何存、scale 如何保存、backend kernel 如何读取。
```

再压缩成最小心智模型：

```text
cache_dtype 决定语义；
KVQuantMode 决定 kernel 分支；
KVCacheSpec 决定内存账本；
AttentionBackend 决定物理布局和读写实现。
```
