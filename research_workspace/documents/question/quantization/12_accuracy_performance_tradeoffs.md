# 12. 量化的精度、显存和性能如何取舍？

源码位置：

- `code/code/vllm/vllm/config/vllm.py`
- `code/code/vllm/vllm/config/cache.py`
- `code/code/vllm/vllm/platforms/interface.py`
- `code/code/vllm/vllm/platforms/cuda.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/fp8.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/auto_awq.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/online/base.py`
- `code/code/vllm/vllm/model_executor/layers/quantization/kv_cache.py`
- `code/code/vllm/vllm/model_executor/kernels/linear/__init__.py`
- `code/code/vllm/vllm/v1/kv_cache_interface.py`
- `code/code/vllm/vllm/v1/attention/backend.py`
- `code/code/vllm/vllm/v1/attention/backends/triton_attn.py`
- `code/code/vllm/vllm/v1/metrics/loggers.py`
- `code/code/vllm/vllm/v1/metrics/stats.py`
- `code/code/vllm/vllm/v1/metrics/perf.py`

本问题关注：不同量化方式对显存、吞吐、延迟、精度和数值稳定性的影响；为什么量化不是“一定更快”；如何从 vLLM 源码里的 capability 校验、kernel 选择、fallback、KV cache page size、metrics 观测指标来建立取舍模型。

---

## 1. 一句话回答

量化的收益取决于瓶颈在哪里：

```text
权重显存瓶颈
  → weight-only / FP8 / INT4 更可能有收益。

KV cache 显存瓶颈
  → kv_cache_dtype / KV cache quantization 更关键。

GEMM 带宽或算力瓶颈
  → 只有当前 shape、dtype、平台能命中高效量化 kernel 时才可能更快。

长上下文 attention 带宽瓶颈
  → KV cache 量化可能提升容量和带宽效率，但会带来 scale / dequant / kernel 限制。

小 batch / 小 token 延迟瓶颈
  → 动态 scale、dequant、kernel launch、fallback 可能抵消低 bit 收益。
```

所以：

```text
量化不是“bit 越低越好”，而是在显存、带宽、kernel 支持、数值误差、运行时开销和工程兼容性之间做取舍。
```

---

## 2. 最小决策框架

可以先用这张表判断量化值不值得开：

```text
1. 模型放不下？
   优先看权重量化。

2. 模型放得下，但 batch / max_model_len 上不去？
   优先看 KV cache 量化。

3. GPU 利用率低、吞吐低？
   先确认是否命中高效 kernel，而不是默认量化会加速。

4. TTFT 变慢？
   看 prefill 的量化 GEMM、在线量化、动态 scale、dequant fallback。

5. TPOT / decode 变慢？
   看 decode batch shape、KV cache dtype、attention backend、small-token kernel。

6. 输出质量下降？
   看权重 bit 数、group size、activation scale、KV cache scale、MoE routing、长上下文 attention。
```

更压缩一点：

```text
显存收益来自“少存 bytes”；
性能收益来自“少搬 bytes 或更快 kernel”；
精度损失来自“更粗的数值表示和 scale 粒度”；
工程风险来自“shape / dtype / backend / platform 不匹配”。
```

---

## 3. vLLM 在哪里提前做取舍校验

### 3.1 QuantizationConfig 声明最低能力和 activation dtype

所有量化配置都继承 `QuantizationConfig`：

```python
def get_supported_act_dtypes(self) -> list[torch.dtype]
def get_min_capability(cls) -> int
def get_quant_method(self, layer, prefix)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:77` 到 `code/vllm/vllm/model_executor/layers/quantization/base_config.py:170`

这说明一个量化方法最少要回答三个问题：

```text
1. 当前 activation dtype 能不能用；
2. 当前 GPU capability 能不能用；
3. 当前 layer 应该用哪个 quant method。
```

### 3.2 VllmConfig 会提前挡住不支持的量化

`VllmConfig._get_quantization_config()` 会：

```text
1. 读取 checkpoint / 用户量化配置；
2. 获取当前设备 capability；
3. 检查 capability >= quant_config.get_min_capability()；
4. 检查 model_config.dtype 在 get_supported_act_dtypes() 里；
5. 调用 maybe_update_config()。
```

位置：`code/vllm/vllm/config/vllm.py:608` 到 `code/vllm/vllm/config/vllm.py:642`

如果不满足，会直接报错，而不是运行时慢慢 fallback。

### 3.3 Platform 还会校验量化方法名

平台抽象里有：

```python
def verify_quantization(cls, quant: str) -> None:
    if cls.supported_quantization and quant not in cls.supported_quantization:
        raise ValueError(...)
```

位置：`code/vllm/vllm/platforms/interface.py:824` 到 `code/vllm/vllm/platforms/interface.py:831`

所以：

```text
同一个 quantization 字符串，在 CUDA / ROCm / XPU / CPU 上可用性可能不同。
```

---

## 4. kernel 选择决定“会不会真的更快”

### 4.1 mixed precision kernel 选择

GPTQ / AWQ / WNA16 等低 bit weight + A16 路径通常走：

```python
choose_mp_linear_kernel(MPLinearLayerConfig)
```

位置：`code/vllm/vllm/model_executor/kernels/linear/__init__.py:640` 到 `code/vllm/vllm/model_executor/kernels/linear/__init__.py:710`

这个函数会按顺序检查：

```text
1. 当前平台候选 kernel；
2. --linear-backend 是否强制指定；
3. VLLM_DISABLED_KERNELS 是否禁用；
4. kernel 最低 compute capability；
5. kernel.can_implement(config) 是否支持当前 shape / dtype / quant type。
```

找不到就抛错。

### 4.2 FP8 scaled-mm kernel 选择

FP8 路径调用：

```python
init_fp8_linear_kernel(...)
```

位置：`code/vllm/vllm/model_executor/kernels/linear/__init__.py:531` 到 `code/vllm/vllm/model_executor/kernels/linear/__init__.py:601`

它会根据 activation scale 是否 per-group 选择：

```text
普通 FP8 scaled-mm kernel
  或
FP8 block-scaled kernel
```

CUDA 上候选包括：

```text
MarlinFP8ScaledMMLinearKernel
FlashInferFP8ScaledMMLinearKernel
CutlassFP8ScaledMMLinearKernel
Torch FP8 fallback
DeepGemm / Cutlass / Triton block-scaled kernel
```

位置：`code/vllm/vllm/model_executor/kernels/linear/__init__.py:286` 到 `code/vllm/vllm/model_executor/kernels/linear/__init__.py:323`

### 4.3 --linear-backend 是强约束

`--linear-backend` 会过滤候选 kernel：

```text
cutlass / flashinfer_cutlass / marlin / triton / deep_gemm / torch / aiter / machete / ...
```

位置：`code/vllm/vllm/model_executor/kernels/linear/__init__.py:183` 到 `code/vllm/vllm/model_executor/kernels/linear/__init__.py:272`

如果用户强制了某个 backend，但该 backend 没有可用 kernel，会报错。

这意味着：

```text
同一个量化 checkpoint，实际性能主要取决于最终选中的 kernel，而不是 checkpoint 写着 AWQ / GPTQ / FP8 这个名字。
```

---

## 5. 显存收益来自哪里

### 5.1 权重量化降低参数显存

权重量化主要减少模型参数 bytes：

```text
FP16 / BF16：
  2 bytes / weight。

FP8 / INT8：
  约 1 byte / weight，另有 scale / metadata。

INT4 / FP4 / MXFP4 / NVFP4：
  约 0.5 byte / weight，另有 scale / zero point / packing overhead。
```

vLLM 的性能估算模块也用类似近似：

```python
_QUANT_WEIGHT_BYTE_SIZE = {
    "fp8": 1,
    "awq": 0.5,
    "gptq": 0.5,
    "bitsandbytes": 0.5,
    "compressed-tensors": 0.5,
    ...
}
```

位置：`code/vllm/vllm/v1/metrics/perf.py:44` 到 `code/vllm/vllm/v1/metrics/perf.py:75`

注意这里是估算：

```text
真实显存还要加上 scale、zero point、g_idx、workspace、padding、repack 后布局和未量化层。
```

### 5.2 KV cache 量化降低长上下文 / 大 batch 显存

KV cache 显存按 page 计算。

普通 attention 的基础 page size：

```text
2 * block_size * num_kv_heads * head_size * sizeof(dtype)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:183` 到 `code/vllm/vllm/v1/kv_cache_interface.py:201`

`FullAttentionSpec` 会用：

```text
block_size * num_kv_heads * (head_size + head_size_v) * sizeof(dtype)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:308` 到 `code/vllm/vllm/v1/kv_cache_interface.py:328`

因此 KV cache dtype 从 BF16/FP16 变成 FP8，理论上每页可以接近减半。

### 5.3 KV cache 量化也有额外开销

per-token-head 量化会额外预算 scale：

```python
real_page_size += 2 * block_size * num_kv_heads * sizeof(float32)
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:169` 到 `code/vllm/vllm/v1/kv_cache_interface.py:177`

NVFP4 page 里包含：

```text
packed fp4 data + fp8 block scales
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:185` 到 `code/vllm/vllm/v1/kv_cache_interface.py:194`

TurboQuant 使用自定义 slot size：

```python
return block_size * num_kv_heads * tq_slot_size
```

位置：`code/vllm/vllm/v1/kv_cache_interface.py:340` 到 `code/vllm/vllm/v1/kv_cache_interface.py:355`

所以 KV cache 量化的显存收益不是只看 dtype bit 数，还要看：

```text
scale 粒度；
padding；
backend layout；
head_size_v；
是否是 TurboQuant / NVFP4 专用格式。
```

### 5.4 CacheConfig 还影响 KV cache 容量

`CacheConfig` 里有：

```text
gpu_memory_utilization
kv_cache_memory_bytes
num_gpu_blocks
kv_cache_size_tokens
kv_cache_max_concurrency
```

位置：`code/vllm/vllm/config/cache.py:67` 到 `code/vllm/vllm/config/cache.py:185`

这说明：

```text
KV cache 量化的直接收益通常表现为：
  更多 block；
  更大的可服务上下文；
  更高 batch concurrency；
  更少 preemption。
```

---

## 6. 性能收益为什么不稳定

### 6.1 量化可能更快的情况

```text
1. 权重读取带宽是瓶颈；
2. 当前量化格式命中高效 kernel；
3. shape 满足 tile / group / pack 对齐；
4. batch 足够大，kernel 利用率高；
5. dequant / scale 计算被 fused 到 kernel 中；
6. KV cache 读取带宽是瓶颈，且量化 attention backend 高效。
```

这些条件满足时，低 bit 可以减少 memory traffic 或提升 tensor core / special kernel 的吞吐。

### 6.2 量化可能不快的情况

```text
1. kernel fallback 到普通 dequant + matmul；
2. 每次 forward 需要动态计算 activation scale；
3. 小 batch / decode 单 token 下 kernel launch 和 scale 开销占比高；
4. shape 不满足高效 kernel tile；
5. TP 后局部维度太小；
6. batch invariant / cudagraph / backend 限制禁用了最快 kernel；
7. MoE token 分布稀疏，expert GEMM 太碎。
```

vLLM 源码里有很多这种显式分支。

---

## 7. AWQ 的性能取舍例子

AWQ 配置当前主路径支持 4bit：

```python
TYPE_MAP = {4: scalar_types.uint4}
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:170` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:209`

### 7.1 AWQ 优先选 Marlin，但有条件

`AutoAWQConfig.get_quant_method()` 中：

```text
CUDA + Marlin 支持 + 非 batch invariant
  → AutoAWQMarlinLinearMethod

否则
  → AutoAWQLinearMethod
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:284` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:332`

如果某层 shape 不支持 Marlin，会 warning 并 fallback：

```text
Falling back to unoptimized AWQ kernels.
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:317` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:327`

### 7.2 普通 AWQ 对 token 数有启发式分支

`AutoAWQLinearMethod.apply()`：

```python
FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
if FP16_MATMUL_HEURISTIC_CONDITION or envs.VLLM_BATCH_INVARIANT:
    out = ops.awq_dequantize(...)
    out = torch.matmul(...)
else:
    out = ops.awq_gemm(...)
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:915` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:939`

这说明：

```text
小 token：
  低 bit GEMM 可能更划算。

大 token：
  先 dequant 再普通 matmul 可能更快。
```

所以 AWQ 并不是永远“直接 INT4 GEMM”。

### 7.3 AWQ 也受 TP 对齐影响

创建权重时检查：

```text
input_size_per_partition % group_size == 0
output_size_per_partition % pack_factor == 0
```

不满足会提示可能是 TP 太大。

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:837` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:856`

---

## 8. GPTQ 的精度和兼容性取舍

GPTQ 支持的核心类型：

```python
(4, True) → uint4b8
(8, True) → uint8b128
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:97` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:163`

这里 `True` 表示 symmetric。

### 8.1 bit 数和 group size 影响精度

GPTQ 配置包括：

```text
bits
group_size
desc_act
sym
dynamic
modules_in_block_to_quantize
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:106` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:216`

取舍是：

```text
bits 越低：
  显存越省，但权重误差越大。

group_size 越小：
  scale 粒度更细，精度通常更好，但 scale 元数据更多，kernel/TP 对齐更复杂。

desc_act：
  可能改善量化质量，但 g_idx / 排序 / kernel 支持更复杂。

dynamic：
  可以按模块使用不同 bit 或跳过敏感层，但配置和调试更复杂。
```

### 8.2 GPTQ 支持按模块动态覆盖

源码注释里说明：

```text
dynamic = {
  "+:regex": {"bits": 8, "group_size": 64},
  "-:regex": {},  # skip module
}
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:123` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:146`

这是一种典型精度取舍手段：

```text
敏感层少量升 bit 或跳过量化；
非敏感层继续低 bit。
```

---

## 9. FP8 的取舍

FP8 配置包括：

```text
is_checkpoint_fp8_serialized
activation_scheme
ignored_layers
weight_block_size
store_dtype
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:99` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:177`

### 9.1 FP8 activation scheme

FP8 支持：

```text
static activation scale
dynamic activation scale
block-wise weight scale
```

其中 block-wise 有限制：

```text
1. 只支持 fp8-serialized checkpoint；
2. weight_block_size 必须是二维；
3. 只支持 dynamic activation scheme。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:119` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:136`

取舍是：

```text
static activation scale：
  runtime 开销小，但依赖校准质量，分布漂移时风险更高。

dynamic activation scale：
  更适应输入分布，但每次 forward 有 scale 计算开销。

block-wise scale：
  scale 粒度更细，精度更好，但 scale metadata 和 kernel 约束更多。
```

### 9.2 FP8 根据硬件选择 activation quant granularity

`Fp8LinearMethod.__init__()` 里：

```python
if self.act_q_static:
    activation_quant_key = kFp8StaticTensorSym
elif cutlass_fp8_supported():
    activation_quant_key = kFp8DynamicTokenSym
else:
    activation_quant_key = kFp8DynamicTensorSym
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:313` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:321`

这说明：

```text
硬件 / kernel 支持更强时，可以用更细粒度的 dynamic token scale；
否则退到 per-tensor dynamic scale。
```

### 9.3 FP8 可能在线量化

如果 checkpoint 不是 serialized FP8：

```text
Fp8PerTensorOnlineLinearMethod
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:179` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:200`

这类路径的取舍是：

```text
优点：
  不需要提前准备 FP8 checkpoint。

代价：
  加载阶段需要把 fp16/bf16 权重量化；
  post-load 时间和峰值内存可能更高；
  精度取决于在线量化策略。
```

### 9.4 batch invariant 下可能走 BF16 dequant fallback

`Fp8LinearMethod.apply()` 中：

```text
batch invariant mode：
  如果不能直接走支持的 FP8 kernel，
  per-tensor/channel 会 dequant 到 BF16 再 linear。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/fp8.py:446` 到 `code/vllm/vllm/model_executor/layers/quantization/fp8.py:489`

这会影响性能：

```text
量化权重仍然省显存；
但 forward 可能不再获得 FP8 kernel 的完整加速。
```

---

## 10. bitsandbytes 的取舍

BNB 配置包括：

```text
load_in_8bit
load_in_4bit
bnb_4bit_compute_dtype
bnb_4bit_quant_storage
bnb_4bit_quant_type
bnb_4bit_use_double_quant
llm_int8_skip_modules
llm_int8_threshold
```

位置：`code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:49` 到 `code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:158`

它支持 activation dtype：

```text
float32 / float16 / bfloat16
```

位置：`code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:99` 到 `code/vllm/vllm/model_executor/layers/quantization/bitsandbytes.py:105`

取舍是：

```text
优点：
  生态成熟；4bit/8bit 入口简单；支持自己的 quant state。

代价：
  依赖 bitsandbytes kernel 和状态管理；
  不一定和 vLLM fused MoE / TP / compile 路径一样高效；
  4bit MoE 等路径可能需要 dequant 后再进入 fused experts。
```

---

## 11. online quantization 的取舍

`OnlineQuantizationConfig` 明确说明：

```text
quantize fp16/bf16 weights during model loading,
without requiring a pre-quantized checkpoint。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/online/base.py:74` 到 `code/vllm/vllm/model_executor/layers/quantization/online/base.py:116`

它的 dispatch 表支持：

```text
Linear：
  fp8 per tensor / fp8 per block / fp8 per channel / mxfp8

MoE：
  fp8 per tensor / fp8 per block / fp8 per channel / mxfp8 / int8 per channel
```

位置：`code/vllm/vllm/model_executor/layers/quantization/online/base.py:55` 到 `code/vllm/vllm/model_executor/layers/quantization/online/base.py:71`

取舍是：

```text
优点：
  用户不用准备量化 checkpoint；
  可以按 linear / moe / ignore 配置快速试验。

代价：
  加载阶段多一步量化；
  精度没有离线校准方法那么可控；
  activation override 目前还没有完全接入。
```

源码中如果用户显式传 activation override，会直接拒绝：

```text
activation override ... is not yet supported
```

位置：`code/vllm/vllm/model_executor/layers/quantization/online/base.py:118` 到 `code/vllm/vllm/model_executor/layers/quantization/online/base.py:143`

---

## 12. KV cache 量化的精度和性能取舍

### 12.1 KV cache 量化主要优化长上下文和大 batch

`CacheConfig.cache_dtype` 控制 KV cache 存储 dtype：

```text
auto / float16 / bfloat16 / fp8 / fp8_e4m3 / fp8_e5m2 / int8_per_token_head / fp8_per_token_head / nvfp4 / turboquant_*
```

位置：`code/vllm/vllm/config/cache.py:75` 到 `code/vllm/vllm/config/cache.py:117`

适合场景：

```text
1. max_model_len 很长；
2. 并发请求多；
3. GPU KV cache usage 高；
4. preemption 多；
5. decode 主要受 KV cache 读取带宽影响。
```

### 12.2 FP8 per-tensor scale 精度依赖 scale 质量

`BaseKVCacheMethod.process_weights_after_loading()` 中：

```text
如果 checkpoint 有 k_scale / v_scale：使用；
如果没有：默认 1.0；
如果 fp8_e4m3 用 1.0：warning；
如果 q/prob scale 缺失：也会 warning。
```

位置：`code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:74` 到 `code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:197`

这解释了 FP8 KV cache 的精度风险：

```text
scale 不合适时，历史 K/V 的 attention logits 和 value aggregation 都会受影响；
长上下文下误差更容易积累或放大。
```

### 12.3 per-token-head scale 精度更细，但 metadata / kernel 更复杂

Triton attention 对 per-token-head 路径会：

```text
1. 从 KV cache 中切出 _k_scale_cache / _v_scale_cache；
2. 写 cache 时动态计算每个 token/head 的 scale；
3. attention 读取时传 k_scale_cache / v_scale_cache。
```

位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:590` 到 `code/vllm/vllm/v1/attention/backends/triton_attn.py:767`

取舍是：

```text
优点：
  scale 粒度细，通常比 per-tensor 更稳。

代价：
  额外 scale 存储；
  写 cache kernel 要计算 scale；
  backend 支持范围更窄。
```

### 12.4 encoder attention 不支持量化 KV cache

Triton encoder attention 中：

```python
if is_quantized_kv_cache(self.kv_cache_dtype):
    raise NotImplementedError(
        "quantized KV cache is not supported for encoder attention"
    )
```

位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:678` 到 `code/vllm/vllm/v1/attention/backends/triton_attn.py:722`

所以 encoder / encoder-only / 某些多模态或 encoder-decoder 场景不能简单套用 decoder KV cache 量化经验。

---

## 13. Attention backend 也会改变取舍

量化 KV cache 或 FP8 attention 不是纯 dtype 选择，还要 backend 支持。

`AttentionBackend.validate_configuration()` 会检查：

```text
head_size
dtype
kv_cache_dtype
block_size
MLA / non-MLA
sink / sparse / mm_prefix
compute capability
attention type
non-causal / batch invariant / KV connector
```

位置：`code/vllm/vllm/v1/attention/backend.py:307` 到 `code/vllm/vllm/v1/attention/backend.py:375`

CUDA platform 选择 attention backend 时，会把每个 backend 的 invalid reasons 记录下来：

```text
Some attention backends are not valid ... Reasons: {...}
```

位置：`code/vllm/vllm/platforms/cuda.py:337` 到 `code/vllm/vllm/platforms/cuda.py:419`

所以同样的 `--kv-cache-dtype fp8`，可能因为：

```text
head_size 不支持；
block_size 不支持；
FA3 / SM90 不满足；
Triton FP8 需要 SM89+；
batch invariant 或 connector 限制；
```

而选择不同 backend 或直接报错。

---

## 14. MoE 量化的额外取舍

MoE 量化不仅是 expert 权重低 bit，还要处理：

```text
router logits
top-k expert 选择
token 按 expert 分组
expert GEMM shape
expert parallel / tensor parallel
fused MoE kernel layout
```

源码中的 fallback 很直接：

```text
GPTQMoeMarlin 不支持某层
  → Falling back to Moe WNA16 kernels

AutoAWQMoEMarlin 不支持某层
  → Falling back to Moe WNA16 kernels
```

位置：`code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:240` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:254`、`code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:334` 到 `code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:355`

MoE 的性能更依赖实际 batch 中 token 到 expert 的分布：

```text
专家分布均匀、每个 expert token 数足够大：
  fused quantized MoE kernel 更容易高效。

专家分布稀疏、小 expert batch：
  routing、permute、padding、kernel launch、dequant 可能占比更高。
```

精度风险也更特殊：

```text
量化 expert 权重可能改变 expert 输出；
如果 router / gate 周边数值受影响，top-k 边界 token 可能选到不同 expert；
这类误差是离散的，不一定表现为平滑退化。
```

---

## 15. 精度风险来自哪里

量化的精度风险可以分成几层：

```text
1. 权重误差
   qweight + scale / zero point 不能完全还原原始权重。

2. scale 粒度误差
   per-tensor 粒度粗，per-channel / per-group / per-block 更细但开销更高。

3. activation clipping / dynamic scale
   activation 分布异常时，static scale 可能不够；dynamic scale 有 runtime 开销。

4. KV cache 误差
   历史 K/V 被量化，长上下文 attention 对 scale 更敏感。

5. MoE routing 敏感性
   top-k 边界附近的小数值变化可能改变 expert 选择。

6. logits 分布变化
   最终 logits 的 margin 变小或排序改变，采样和 structured output 都可能受影响。
```

vLLM 源码里体现最明显的是：

```text
- FP8 KV cache 缺 scale 时 warning；
- GPTQ dynamic 支持按模块升 bit 或 skip；
- ignored_layers / modules_to_not_convert 支持跳过敏感层；
- online quant 支持 ignore；
- AWQ / GPTQ MoE 不满足 kernel 条件时 fallback。
```

---

## 16. 延迟指标怎么看

量化对延迟的影响要分阶段看。

### 16.1 TTFT / prefill

TTFT 主要受：

```text
prompt prefill GEMM；
attention prefill；
weight dequant / activation quant；
kernel 选择；
prefix cache 命中；
调度等待；
```

影响。

如果权重量化命中高效 kernel，prefill 可能更快；如果走 dequant + matmul 或 dynamic scale 开销大，TTFT 可能变慢。

### 16.2 TPOT / decode

TPOT 主要受：

```text
单步 decode GEMM；
KV cache 读取；
attention backend；
采样；
batch size；
KV cache layout / dtype；
```

影响。

KV cache 量化通常更影响 decode，因为 decode 要频繁读取历史 KV。

### 16.3 vLLM stats 里有哪些字段

`FinishedRequestStats` 包含：

```text
e2e_latency
queued_time
prefill_time
inference_time
decode_time
mean_time_per_output_token
first_token_latency
num_cached_tokens
```

位置：`code/vllm/vllm/v1/metrics/stats.py:202` 到 `code/vllm/vllm/v1/metrics/stats.py:240`

这些字段可以对应：

```text
TTFT：first_token_latency / prefill_time 相关；
TPOT：mean_time_per_output_token / decode_time 相关；
端到端：e2e_latency；
cache 影响：num_cached_tokens。
```

---

## 17. 吞吐和 KV cache 指标怎么看

日志里会输出：

```text
Avg prompt throughput: ... tokens/s
Avg generation throughput: ... tokens/s
GPU KV cache usage: ...%
Prefix cache hit rate: ...%
External prefix cache hit rate: ...%
```

位置：`code/vllm/vllm/v1/metrics/loggers.py:219` 到 `code/vllm/vllm/v1/metrics/loggers.py:283`

这些指标和量化的关系是：

```text
prompt throughput：
  看 prefill GEMM / attention 是否被量化 kernel 加速。

generation throughput：
  看 decode GEMM、KV cache 读取、attention backend 是否受益。

GPU KV cache usage：
  看 KV cache 是否成为容量瓶颈。

Prefix cache hit rate：
  命中越高，实际 prefill compute 越少，量化带来的 prefill GEMM 加速越不明显。

External prefix cache hit rate：
  远端 KV 命中会改变 TTFT / prefill compute 和 KV transfer 成本。
```

---

## 18. 评估时应该对比什么

### 18.1 Baseline 必须明确

推荐至少比较：

```text
BF16 / FP16 baseline
  vs
目标量化方案
```

并保持：

```text
相同模型；
相同 prompt 集；
相同 max_model_len；
相同 batch / 并发；
相同 sampling 参数；
相同 tensor parallel / pipeline parallel；
相同 attention backend 或明确记录 backend 差异。
```

### 18.2 精度评估

建议覆盖：

```text
1. 普通知识 / 指令跟随 benchmark；
2. 领域数据集；
3. 长上下文问答；
4. 数学 / 代码等对 logits margin 敏感的任务；
5. structured output 合法率；
6. 多轮对话稳定性；
7. MoE 模型的路由敏感任务。
```

### 18.3 性能评估

建议分别看：

```text
1. 显存峰值；
2. 可用 KV cache blocks / max concurrency；
3. prompt throughput；
4. generation throughput；
5. TTFT；
6. TPOT；
7. preemption 次数；
8. GPU KV cache usage；
9. prefix cache hit rate；
10. kernel 是否命中预期 backend。
```

### 18.4 稳定性评估

建议检查：

```text
1. 是否出现 NaN logits；
2. 是否出现重复、乱码、格式崩坏；
3. 长上下文后半段质量是否下降；
4. high temperature sampling 是否更不稳定；
5. 低概率边界 token 是否导致 structured output 失败；
6. 多卡 TP / PP / EP 下是否有 shape / group size 报错。
```

---

## 19. 不同量化方案的典型取舍

### 19.1 INT4 / AWQ / GPTQ / WNA16

```text
优点：
  权重显存收益大；适合模型放不下或想提升并发。

风险：
  精度损失比 FP8 更明显；group size / desc_act / zero point / kernel 支持复杂。

性能：
  命中 Marlin / Machete / Cutlass 等 kernel 时可能快；fallback 或 dequant 时可能不快。
```

### 19.2 FP8 weight / W8A8

```text
优点：
  显存收益中等；通常精度比 INT4 更稳；硬件支持好时性能强。

风险：
  scale 质量和 activation scheme 很关键；部分平台 / shape 会 fallback。

性能：
  依赖 FP8 tensor core / Cutlass / FlashInfer / Marlin / DeepGemm 等 kernel。
```

### 19.3 online quantization

```text
优点：
  不需要预量化 checkpoint；试验成本低。

风险：
  没有离线校准那么可控；加载阶段更复杂；activation override 支持有限。

性能：
  runtime 取决于最终 kernel；加载时间和峰值内存要单独评估。
```

### 19.4 bitsandbytes

```text
优点：
  易用，生态常见，支持 4bit / 8bit。

风险：
  与 vLLM fused kernel、MoE、TP、compile 路径的性能不一定最优。

性能：
  更依赖 BNB 自身 kernel 和 quant state。
```

### 19.5 KV cache FP8 / INT8 / NVFP4 / TurboQuant

```text
优点：
  长上下文和大 batch 下显著降低 KV cache 压力。

风险：
  scale 不准会影响 attention；backend 支持更窄；特殊 layout 影响外部 KV transfer。

性能：
  decode 可能受益；写 cache 和 dequant / scale cache 也有额外开销。
```

---

## 20. 为什么量化会影响 structured output / sampling

量化最终会改变 logits 分布。

即使 top-1 准确率看起来没大问题，以下场景仍可能敏感：

```text
1. JSON / tool call / schema constrained output；
2. 多个合法 token 概率接近；
3. grammar mask 后候选 token 很少；
4. 高 temperature / top-p sampling；
5. 长上下文后 logits margin 变小；
6. 代码补全中的符号、括号、缩进 token。
```

原因是：

```text
量化误差不一定让模型整体“变笨”；
它可能只改变边界 token 的排序或概率间隔。
```

因此评估 structured output 时，不只看文本质量，还要看：

```text
合法率；
重试率；
schema violation；
工具调用参数准确率；
固定 seed 下输出差异。
```

---

## 21. 常见错误取舍

### 21.1 只看模型权重显存

错误心智：

```text
模型权重从 BF16 变 INT4，所以总显存约等于 1/4。
```

实际还要算：

```text
KV cache；
activation / workspace；
scale / zero point；
CUDA graph capture；
MoE expert workspace；
NCCL / communication buffer；
未量化层；
加载阶段峰值显存。
```

### 21.2 只看 benchmark 平均分

平均分没掉，不代表线上安全。

需要特别看：

```text
长上下文；
结构化输出；
低温确定性任务；
高温创作任务；
领域专有术语；
多轮对话状态保持。
```

### 21.3 只看吞吐，不看 TTFT / TPOT

某些量化方案会：

```text
prompt throughput 提升，但 decode TPOT 变差；
或 decode 变好，但 prefill / TTFT 变差。
```

需要分开看。

### 21.4 以为所有层都量化了

实际可能有：

```text
ignored_layers；
modules_to_not_convert；
GPTQ dynamic negative match；
kv_cache_dtype_skip_layers；
不支持的 MoE fallback；
lm_head 未量化。
```

这些都会影响显存、性能和精度。

---

## 22. 实用选择建议

### 22.1 目标是“模型放得下”

优先级：

```text
1. INT4 / FP4 weight-only；
2. 检查是否有高效 kernel；
3. 对敏感层使用 skip / dynamic 升 bit；
4. 再考虑 KV cache dtype 降低长上下文成本。
```

### 22.2 目标是“长上下文 / 高并发”

优先级：

```text
1. 先看 GPU KV cache usage 和 preemption；
2. 尝试 fp8 / per-token-head / nvfp4 / turboquant KV cache；
3. 对比长上下文质量；
4. 检查 attention backend 是否变更；
5. 检查 external KV transfer layout 是否一致。
```

### 22.3 目标是“吞吐最大化”

优先级：

```text
1. 确认最终 kernel；
2. 对比 prompt throughput 和 generation throughput；
3. 分 batch size / prompt len / output len 做 sweep；
4. 避免 shape 触发 fallback；
5. 必要时指定或排除 linear backend。
```

### 22.4 目标是“质量最稳”

优先级：

```text
1. FP8 优先于 INT4；
2. per-channel / per-group / per-block 优先于粗粒度 per-tensor；
3. 对敏感层 skip 或升 bit；
4. KV cache 使用校准 scale 或更细粒度 scale；
5. 保留 BF16/FP16 baseline 做回归。
```

---

## 23. 调试时应该看哪些源码点

如果量化方法直接不支持：

```text
code/vllm/vllm/config/vllm.py:608
code/vllm/vllm/platforms/interface.py:824
```

如果想知道为什么某个 kernel 没选中：

```text
code/vllm/vllm/model_executor/kernels/linear/__init__.py:183
code/vllm/vllm/model_executor/kernels/linear/__init__.py:531
code/vllm/vllm/model_executor/kernels/linear/__init__.py:640
```

如果想看 AWQ 为什么 fallback：

```text
code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:284
code/vllm/vllm/model_executor/layers/quantization/auto_awq.py:915
```

如果想看 GPTQ 的 bit / group / dynamic：

```text
code/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:97
```

如果想看 FP8 dynamic / static / block-wise：

```text
code/vllm/vllm/model_executor/layers/quantization/fp8.py:99
code/vllm/vllm/model_executor/layers/quantization/fp8.py:267
```

如果想看 KV cache page size：

```text
code/vllm/vllm/v1/kv_cache_interface.py:159
```

如果想看 KV cache scale 风险：

```text
code/vllm/vllm/model_executor/layers/quantization/kv_cache.py:74
```

如果想看运行指标：

```text
code/vllm/vllm/v1/metrics/loggers.py:219
code/vllm/vllm/v1/metrics/stats.py:202
code/vllm/vllm/v1/metrics/perf.py:44
```

---

## 24. 容易疑惑的点

### 24.1 量化是不是一定更快？

不是。

只有命中高效 kernel，并且节省的 memory traffic / compute 大于 scale、dequant、fallback、kernel launch 开销时，才会更快。

### 24.2 INT4 是不是一定比 FP8 更好？

不是。

INT4 权重显存更省，但精度风险更高，kernel 和 shape 限制也更强；FP8 通常更稳，硬件支持好时性能也很强。

### 24.3 KV cache 量化能替代权重量化吗？

不能。

```text
权重量化：降低模型参数显存。
KV cache 量化：降低历史 K/V cache 显存。
```

瓶颈不同，收益不同。

### 24.4 为什么量化后显存没有按 bit 数等比例下降？

因为还有：

```text
scale / zero point / g_idx；
未量化层；
workspace；
KV cache；
activation；
padding / packed layout；
load-time 临时张量。
```

### 24.5 为什么同一个量化模型在不同 GPU 上表现差很多？

因为不同平台的：

```text
supported_quantization；
compute capability；
FP8 / INT4 kernel；
attention backend；
linear backend；
layout；
```

都可能不同。

### 24.6 为什么量化后 structured output 更容易失败？

因为结构化输出经常依赖边界 token 排序和小概率差异。量化可能不明显降低普通文本质量，但足以改变 JSON 括号、逗号、字段名、tool 参数 token 的概率排序。

### 24.7 为什么大 batch 和小 batch 的量化收益不同？

小 batch / decode 下：

```text
kernel launch、scale 计算、dequant 开销占比更高。
```

大 batch / prefill 下：

```text
GEMM 利用率更高，低 bit kernel 或 dequant + matmul 可能更划算。
```

AWQ 的 `num_tokens >= 256` 分支就是源码层面的例子。

---

## 25. 总结

量化取舍可以压缩成：

```text
权重量化：
  省模型参数显存，可能提升 GEMM 性能，但要付出 scale / zero point / kernel 限制和精度误差。

activation / dynamic quant：
  可能提升 W8A8 / FP8 性能，但增加 runtime scale 计算和数值分布风险。

KV cache quantization：
  省长上下文和大 batch 的 KV cache 显存，可能改善 decode 带宽，但依赖 attention backend 和 scale 质量。

MoE quantization：
  省 expert 权重显存，但 routing、expert token 分布、fused kernel 和 EP/TP 对齐让性能更不稳定。
```

如果只记住一句话：

```text
量化的价值取决于当前瓶颈：模型权重放不下就看权重量化，KV cache 顶满就看 KV cache 量化，吞吐不够就先确认有没有命中高效 kernel，质量敏感就优先选择更细 scale 或跳过敏感层。
```

再压缩成最小心智模型：

```text
省 bytes 是显存收益；
命中 kernel 是性能收益；
scale 粒度决定精度风险；
fallback 和 shape 对齐决定工程风险。
```
