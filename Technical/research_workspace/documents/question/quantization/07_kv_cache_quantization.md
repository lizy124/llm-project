# 07. KV cache quantization 如何工作？

源码位置：

- `code/vllm/vllm/config/cache.py`
- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/utils/torch_utils.py`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/quantization/kv_cache.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py`

本问题只关注：KV cache quantization 本身如何表示、分配、保存 scale、计算 page size，以及它和 block table / slot mapping / prefix cache / external KV transfer 的关系。attention backend 如何选择、各 backend 支持哪些量化格式，在 `08_attention_backend_interaction.md` 中展开。

---

## 1. 一句话回答

KV cache quantization 压缩的是推理过程中产生的 **历史 K/V cache**，不是模型权重。

主链路是：

```text
--kv-cache-dtype / checkpoint KV cache metadata
  → CacheConfig.cache_dtype
  → Attention.kv_cache_dtype / kv_cache_torch_dtype
  → KVQuantMode
  → KVCacheSpec.page_size_bytes
  → Worker 分配 raw KV cache bytes
  → backend-specific view / shape / layout
  → Attention.forward() 写入本轮 K/V
  → 后续 attention 从量化 cache 读取历史 K/V
```

最小边界：

```text
权重量化：
  处理模型静态参数。

KV cache 量化：
  处理 runtime 历史 K/V 的存储格式和 scale。
```

---

## 2. 核心对象

KV cache quantization 里有四个核心对象：

```text
CacheConfig.cache_dtype：
  用户 / 模型解析后的 KV cache 存储格式。

Attention.kv_cache_dtype：
  单个 attention layer 实际使用的 cache dtype 字符串。

KVQuantMode：
  从 kv_cache_dtype 派生出的 kernel 侧枚举。

KVCacheSpec / AttentionSpec：
  KV cache 的内存账本，描述 block_size、dtype、page_size、quant mode。
```

对应关系：

```text
cache_dtype 是配置语义；
kv_cache_dtype 是 layer 语义；
KVQuantMode 是 kernel dispatch 语义；
KVCacheSpec 是内存预算和分组语义。
```

---

## 3. kv_cache_dtype 从哪里来

### 3.1 EngineArgs / CacheConfig

`EngineArgs` 中相关字段：

```text
kv_cache_dtype
calculate_kv_scales
kv_cache_dtype_skip_layers
```

位置：

- `code/vllm/vllm/engine/arg_utils.py:440`
- `code/vllm/vllm/engine/arg_utils.py:680`
- `code/vllm/vllm/engine/arg_utils.py:681`

构造 `CacheConfig` 前会调用：

```python
resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1832`

最终进入：

```text
CacheConfig.cache_dtype
```

### 3.2 CacheDType 支持哪些值

`CacheDType` 包括：

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

位置：`code/vllm/vllm/config/cache.py:19`

含义分层：

```text
float16 / bfloat16：
  显式非量化 KV cache dtype。

fp8*：
  FP8 KV cache，通常是 per-tensor scale。

int8_per_token_head / fp8_per_token_head：
  每个 token、每个 KV head 动态 scale。

nvfp4：
  packed FP4 data + FP8 block scales。

turboquant_*：
  TurboQuant 专用 K/V 压缩 cache layout。
```

### 3.3 auto 可能从 checkpoint 元数据推断

当 `kv_cache_dtype == "auto"` 时，`resolve_kv_cache_dtype_string()` 会检查模型量化配置中的 KV cache quant metadata。

位置：`code/vllm/vllm/utils/torch_utils.py:373`

因此：

```text
普通模型：
  auto 通常保持模型默认 dtype。

带 KV cache quant metadata 的 checkpoint：
  auto 可能解析成 fp8 / nvfp4 等量化 dtype。
```

---

## 4. Attention 层如何接住 kv_cache_dtype

`Attention.__init__()` 中：

```python
if cache_config is not None:
    kv_cache_dtype = cache_config.cache_dtype
    calculate_kv_scales = cache_config.calculate_kv_scales
else:
    kv_cache_dtype = "auto"
    calculate_kv_scales = False
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:238`

之后保存为：

```text
self.kv_cache_dtype
self.kv_cache_torch_dtype
self.calculate_kv_scales
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:291`

这里要区分：

```text
kv_cache_dtype：
  字符串，用于 KVQuantMode、backend 选择和 kernel 分支。

kv_cache_torch_dtype：
  torch dtype，用于 KVCacheSpec 和 tensor view。
```

---

## 5. checkpoint 的 kv_cache_scheme 如何影响 Attention

如果 `quant_config` 中存在：

```text
kv_cache_scheme
```

且当前 `kv_cache_dtype == "auto"`，Attention 会把它改成：

```text
fp8
```

并关闭动态计算 scale。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:246`

含义是：

```text
checkpoint 声明 KV cache quant scheme
  → 默认启用 FP8 KV cache
  → 优先使用 checkpoint 中的 KV scale
```

如果用户显式指定 `--kv-cache-dtype bfloat16` 等值，则用户配置优先。

---

## 6. kv_cache_dtype_skip_layers 做什么

`CacheConfig.kv_cache_dtype_skip_layers` 可以让部分层跳过 KV cache 量化。

位置：`code/vllm/vllm/config/cache.py:115`

支持：

```text
层号："0" / "2" / "4"
attention 类型："sliding_window"
```

如果命中 skip，Attention 会设置：

```text
kv_cache_dtype = "auto"
calculate_kv_scales = False
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:266`

结果是：

```text
同一个模型里，不同 attention layer 可能使用不同 KV cache dtype；
这些层会形成不同 KVCacheSpec / cache group / tensor view。
```

TurboQuant 还可能自动把边界层加入 skip。

位置：`code/vllm/vllm/engine/arg_utils.py:1862`

---

## 7. KVQuantMode 是什么

定义在：`code/vllm/vllm/v1/kv_cache_interface.py:33`

```python
class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    NVFP4 = 4
```

映射函数：

```python
get_kv_quant_mode(kv_cache_dtype)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:60`

规则：

```text
int8_per_token_head → INT8_PER_TOKEN_HEAD
fp8_per_token_head  → FP8_PER_TOKEN_HEAD
nvfp4               → NVFP4
fp8*                → FP8_PER_TENSOR
其他                → NONE
```

这说明：

```text
KVQuantMode 不是用户配置；
它是从 kv_cache_dtype 派生出的 backend / kernel 分支标识。
```

---

## 8. KVCacheSpec 如何记录量化格式

`Attention.get_kv_cache_spec()` 会把 layer 上的 KV cache 信息变成 spec。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:581`

关键字段：

```text
block_size
num_kv_heads
head_size / head_size_v
dtype
kv_quant_mode
sliding_window / attention_chunk_size
```

例如普通 full attention 会返回：

```text
FullAttentionSpec(..., dtype=self.kv_cache_torch_dtype, kv_quant_mode=quant_mode)
```

这让 Scheduler / Worker 知道：

```text
这个 layer 的每个 KV page 需要多少 bytes；
哪些 layer 可以共用 cache group；
每个 block 该按什么 dtype / quant mode 解释。
```

---

## 9. page_size_bytes 如何随量化变化

`AttentionSpec.page_size_bytes` 是 KV cache 内存预算核心。

基础公式：

```text
2 * block_size * num_kv_heads * head_size * sizeof(dtype)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:183`

这里的 `2` 表示：

```text
K cache + V cache
```

### 9.1 per-token-head scale 的额外内存

如果：

```text
kv_quant_mode.is_per_token_head
```

会额外加：

```text
2 * block_size * num_kv_heads * sizeof(float32)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:169`

原因是每个 token / head 需要 K scale 和 V scale。

### 9.2 NVFP4 的 packed dim

NVFP4 使用：

```text
nvfp4_kv_cache_full_dim(head_size)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:185`

含义：

```text
packed FP4 data + FP8 block scales
```

### 9.3 TurboQuant 的 slot size

TurboQuant 走 `TQFullAttentionSpec`，page size 使用：

```text
block_size * num_kv_heads * tq_slot_size
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:340`

它不是普通 K/V 分离布局。

---

## 10. Worker 如何分配 KV cache

Worker 初始化 KV cache 时，先按 bytes 分配 raw buffer：

```python
torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7033`

随后根据 spec 和 backend reshape：

```text
KVCacheSpec.page_size_bytes
  → num_blocks
  → backend.get_kv_cache_shape(...)
  → backend.get_kv_cache_stride_order()
  → _reshape_attention_kv_cache(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7072`

关键点：

```text
KV cache allocation 是 byte-level；
具体 dtype / shape / stride 是后续 view。
```

这使得 float16、fp8、nvfp4、TurboQuant 等格式都可以复用同一套分配框架。

---

## 11. KV scale 从哪里来

KV scale 由 `BaseKVCacheMethod` 处理。

位置：`code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:42`

它会在 Attention layer 上创建临时可加载参数：

```text
q_scale
k_scale
v_scale
prob_scale
```

位置：`code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:57`

加载后会写入运行时 buffer：

```text
_q_scale / _q_scale_float
_k_scale / _k_scale_float
_v_scale / _v_scale_float
_prob_scale / _prob_scale_float
```

位置：`code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:74`

### 11.1 checkpoint 有 scale

如果 checkpoint 有 `k_scale` / `v_scale`，直接使用。

如果只有旧式单个 `kv_scale`，加载阶段会映射并在后处理中复制到 K/V。

### 11.2 checkpoint 没有 scale

如果是 FP8 per-tensor KV cache 且没有 scale，默认使用：

```text
1.0
```

并可能 warning 精度风险。

### 11.3 per-token-head 不用 checkpoint scale

如果是：

```text
int8_per_token_head
fp8_per_token_head
```

scale 在写 cache kernel 中动态计算，checkpoint scale 会被忽略，layer 的 `_k_scale/_v_scale` 置为 1.0。

位置：`code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:82`

---

## 12. calculate_kv_scales 动态 scale 路径

`calculate_kv_scales` 已 deprecated，但仍存在。

位置：`code/vllm/vllm/config/cache.py:110`

如果开启，Attention.forward 会调用：

```python
torch.ops.vllm.maybe_calc_kv_scales(query, key, value, layer_name)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:471`

实际计算：

```text
_q_scale = abs(query).max() / q_range
_k_scale = abs(key).max() / k_range
_v_scale = abs(value).max() / v_range
calculate_kv_scales = False
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:546`

注意：

```text
它是 per-layer / per-tensor scale；
只计算一次；
不是 per-token-head scale。
```

---

## 13. 写 cache 和读 cache 的最小语义

KV cache quantization 不改变 block table / slot mapping 的基本含义。

```text
block table：
  request → KV block ids。

slot mapping：
  token → KV cache slot。
```

变化发生在 slot 内部：

```text
float16 / bfloat16：
  slot 存原始 K/V。

fp8：
  slot 存 FP8 K/V，读写时使用 k/v scale。

per-token-head：
  slot 除量化 K/V 外，还带 token/head 粒度 scale。

nvfp4：
  slot 内是 packed fp4 data + fp8 block scales。

TurboQuant：
  slot 是 backend 专用的 key_packed + value_packed combined layout。
```

因此：

```text
Scheduler 仍然只分配 block；
Worker 仍然维护 block table / slot mapping；
KV cache quantization 改变的是 block 内 bytes 如何被解释。
```

---

## 14. Prefix cache 和 KV cache quantization 的关系

Prefix cache 复用的是 KV block。

```text
prefix cache 命中
  → Scheduler / KVCacheManager 复用已有 block ids
  → Worker block table 指向这些 blocks
  → Attention 按当前 KV cache dtype / layout 读取这些 blocks
```

边界是：

```text
prefix cache：
  管“哪些 block 可以复用”。

KV cache quantization：
  管“block 里的 K/V bytes 是什么格式”。
```

Prefix cache 不直接决定 scale，也不改变量化格式。

---

## 15. external KV transfer 和量化格式的关系

KV connector 传输的是 KV cache memory region 和 block 相关元数据。

它需要发送方和接收方对以下内容有一致理解：

```text
cache dtype
KVQuantMode
KV cache shape
page size
layout
scale 存放方式
```

KV connector 可以要求特定 cache layout：

```python
connector_cls.get_required_kvcache_layout(vllm_config)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py:34`

因此 external KV transfer 的关键不是重新量化，而是：

```text
同一段 KV cache bytes 必须被两端按同一种 spec 解释。
```

---

## 16. sleep / wake_up 后为什么要处理 scale

`GPUModelRunner.post_kv_cache_wake_up()` 会调用：

```python
self.init_fp8_kv_scales()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:935`

如果是量化 KV cache，会：

```text
1. zero KV cache tensors；
2. 把 Attention / MLAAttention 的 K/V scale 重置为 1.0。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:938`

原因是 wake_up 后显存和 scale 状态需要恢复到安全初始值，避免 0 scale 或脏 cache 造成输出异常。

---

## 17. 与 attention backend 的边界

本文只说明 backend 是 KV cache quantization 的消费者。

```text
KV cache quantization 提供：
  kv_cache_dtype、KVQuantMode、KVCacheSpec、scale、page size、layout 需求。

attention backend 决定：
  是否支持该 dtype / scale / layout，以及用哪个 kernel 读写。
```

backend 选择和支持矩阵见：`08_attention_backend_interaction.md`。

---

## 18. 与权重量化的边界

```text
权重量化路径：
  QuantizationConfig
    → Linear / MoE quant_method
    → checkpoint weight / scale / zero point
    → GEMM / MoE kernel

KV cache 量化路径：
  CacheConfig.cache_dtype
    → Attention.kv_cache_dtype
    → KVQuantMode / KVCacheSpec
    → KV cache allocation / scale / layout
    → attention cache update / read
```

两者可以同时存在，但不是同一条路径。

交叉点包括：

```text
1. checkpoint 可能携带 kv_cache_scheme / k_scale / v_scale；
2. BaseKVCacheMethod 位于 quantization 目录，但作用对象是 Attention KV cache；
3. 某些 attention backend 会基于 FP8 KV cache 支持 query quantization。
```

---

## 19. 完整执行链路

```text
1. 用户设置 --kv-cache-dtype，或 checkpoint 提供 KV cache quant metadata
   → EngineArgs 解析
   → CacheConfig.cache_dtype

2. 初始化 Attention layer
   → 保存 kv_cache_dtype / kv_cache_torch_dtype
   → 根据 kv_cache_scheme / skip_layers 调整 dtype
   → 初始化 KV scale buffer

3. 构建 KVCacheSpec
   → get_kv_quant_mode(kv_cache_dtype)
   → dtype + kv_quant_mode + block_size + head 信息
   → page_size_bytes 计算内存预算

4. Worker 初始化 KV cache
   → 按 page_size_bytes 分配 raw int8 buffer
   → 按 backend shape / stride reshape 成 tensor view

5. 每轮 forward
   → ModelRunner 准备 block table / slot mapping
   → Attention.forward()
   → 可选 maybe_calc_kv_scales()
   → 写本轮 K/V 到 cache
   → 后续 attention 读取历史 KV cache
```

---

## 20. 容易疑惑的点

### 20.1 KV cache quantization 会量化模型权重吗？

不会。它只影响历史 K/V cache 的存储和读取。

### 20.2 `calculate_kv_scales` 和 per-token-head 是一回事吗？

不是。

```text
calculate_kv_scales：
  deprecated；per-layer / per-tensor；只算一次。

per-token-head：
  每个 token、每个 KV head 动态 scale；写 cache 时计算。
```

### 20.3 KV cache layout 和 KVQuantMode 是一回事吗？

不是。

```text
layout：
  NHD / HND，描述维度和 stride 排列。

KVQuantMode：
  NONE / FP8_PER_TENSOR / INT8_PER_TOKEN_HEAD / FP8_PER_TOKEN_HEAD / NVFP4，描述 bytes 如何解释。
```

### 20.4 Prefix cache 是否需要知道 scale？

通常不直接关心。Prefix cache 复用 block ids；block 内 bytes 和 scale 由 Attention / backend 按 KVCacheSpec 解释。

### 20.5 为什么 Worker 先分配 int8 raw buffer？

因为 KV cache 可能最终被 view 成 float16、bfloat16、fp8、uint8 packed 或 TurboQuant slot。先按 bytes 分配最灵活。

---

## 21. 总结

KV cache quantization 的最小主线是：

```text
cache_dtype
  → Attention.kv_cache_dtype
  → KVQuantMode
  → KVCacheSpec.page_size_bytes
  → raw KV cache bytes allocation
  → backend-specific tensor view
  → slot_mapping 写入量化 K/V
  → attention 读取量化 K/V + scale
```

一句话总结：

```text
KV cache quantization 是 attention 子系统的内存格式优化；调度仍然按 block 工作，真正变化的是每个 KV page 的 bytes、scale 和 tensor view 如何组织。
```
