# 09. 不同算子 backend 如何选择和回退？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\platforms\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\selector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\registry.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\quantization\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\ops\topk_topp_sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\triton_utils\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\envs.py`

这个问题关注：vLLM 如何根据平台、硬件能力、dtype、shape、配置、已安装依赖选择 CUDA extension、Triton、FlashAttention、FlashInfer、CUTLASS、aiter、XPU kernel 或 torch fallback。

---

## 1. 一句话回答

backend selection 决定“同一个上层算子最后由哪个具体实现执行”。vLLM 通常先把用户配置、平台能力和模型 shape 归一成 selector config，再让平台层按候选优先级逐个验证 backend；验证通过就 lazy import 对应实现，验证失败则记录原因并尝试下一个候选，直到选出 backend 或报错。

可以先记成：

```text
operator request
  → user config / env flags
  → platform detection
  → dtype / shape / capability validation
  → dependency import check
  → selected backend
  → fallback backend or error
```

---

## 2. backend selection 分几类

vLLM 中的 backend 选择不是一个全局开关，而是按算子族分别处理：

```text
attention backend
  → platform + AttentionSelectorConfig + backend registry

sampling backend
  → TopKTopPSampler 初始化时按平台和 logprobs mode 分派

quantization backend
  → quantization method → QuantizationConfig → layer method / kernel

MoE backend
  → fused_moe config + dtype / quant / expert shape / platform

基础算子 backend
  → current_platform + optional dependency + custom op registration
```

所以排查 fallback 时要先确认：

```text
是哪一类算子 fallback，而不是泛泛地说“vLLM fallback”。
```

---

## 3. 平台识别是第一层选择

平台入口在：`vllm/platforms/`

常见平台包括：

```text
CUDA
ROCm
CPU
XPU
TPU / HPU / Neuron / OpenVINO 等扩展平台
UnspecifiedPlatform
```

平台对象通过 `current_platform` 暴露能力查询：

```text
current_platform.is_cuda()
current_platform.is_rocm()
current_platform.is_cpu()
current_platform.is_xpu()
current_platform.is_cuda_alike()
current_platform.get_device_capability()
current_platform.get_attn_backend_cls(...)
```

例如采样 backend 会直接判断：

```python
if current_platform.is_cuda():
    ...
elif current_platform.is_cpu():
    ...
elif current_platform.is_xpu():
    ...
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:86`

attention backend 则把平台选择封装在：

```python
current_platform.get_attn_backend_cls(...)
```

位置：`vllm/v1/attention/selector.py:121`

---

## 4. attention backend selection 主链路

入口是：

```python
def get_attn_backend(...)
```

位置：`vllm/v1/attention/selector.py:54`

它会构造：

```python
AttentionSelectorConfig(
    head_size=head_size,
    dtype=dtype,
    kv_cache_dtype=kv_cache_dtype,
    block_size=block_size,
    use_mla=use_mla,
    has_sink=has_sink,
    use_sparse=use_sparse,
    use_mm_prefix=use_mm_prefix,
    use_per_head_quant_scales=use_per_head_quant_scales,
    attn_type=attn_type,
    use_non_causal=...,
    use_batch_invariant=...,
    use_kv_connector=...,
)
```

位置：`vllm/v1/attention/selector.py:90`

然后调用：

```python
_cached_get_attn_backend(
    backend=vllm_config.attention_config.backend,
    attn_selector_config=attn_selector_config,
    num_heads=num_heads,
)
```

位置：`vllm/v1/attention/selector.py:106`

关键点：

```text
get_attn_backend() 不直接 import 所有 backend；
它把选择交给 current_platform；
最后再通过 resolve_obj_by_qualname lazy import backend class。
```

---

## 5. AttentionBackendEnum 是 registry

backend 名称和实现类路径集中在：

```python
class AttentionBackendEnum(Enum):
```

位置：`vllm/v1/attention/backends/registry.py:34`

常见项包括：

```text
FLASH_ATTN
FLASHINFER
TRITON_ATTN
FLEX_ATTENTION
TURBOQUANT
FLASH_ATTN_MLA
FLASHINFER_MLA
FLASHMLA
TRITON_MLA
CUTLASS_MLA
ROCM_ATTN
ROCM_AITER_FA
ROCM_AITER_MLA
CPU_ATTN
CUSTOM
```

每个 enum 值保存默认 class path：

```python
FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
```

位置：`vllm/v1/attention/backends/registry.py:44`

并支持运行时覆盖：

```python
register_backend(...)
```

位置：`vllm/v1/attention/backends/registry.py:220`

这意味着 attention backend 的选择结果最终是一个 class，而不是字符串开关。

---

## 6. CUDA attention backend 如何选

CUDA 平台的核心逻辑在：`vllm/platforms/cuda.py`

### 6.1 候选优先级

CUDA 上先按模型类型和设备能力生成候选列表：

```python
def _get_backend_priorities(use_mla, device_capability, num_heads=None, kv_cache_dtype=None)
```

位置：`vllm/platforms/cuda.py:79`

普通 attention 在不同 compute capability 下优先级不同。

例如非 MLA 场景：

```text
Blackwell / major == 10:
  FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT

其他 CUDA:
  FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT
```

位置：`vllm/platforms/cuda.py:132`

MLA 场景会额外考虑：

```text
FLASH_ATTN_MLA
FLASHMLA
FLASHINFER_MLA
CUTLASS_MLA
TRITON_MLA
sparse MLA backend
kv_cache_dtype 是否量化
num_heads 是否较低
```

位置：`vllm/platforms/cuda.py:87`

### 6.2 validate_configuration

CUDA 平台会逐个候选 backend 做验证：

```python
backend_class.validate_configuration(
    device_capability=device_capability,
    **attn_selector_config._asdict(),
)
```

位置：`vllm/platforms/cuda.py:335`

如果验证失败，会记录：

```text
backend → priority → invalid reasons
```

如果 import 失败，则原因是：

```text
ImportError
```

位置：`vllm/platforms/cuda.py:341`

### 6.3 最终选择

如果用户显式指定 backend，会先验证指定 backend：

```text
指定 backend 有效：直接使用；
指定 backend 无效：抛 ValueError，不自动静默回退。
```

位置：`vllm/platforms/cuda.py:360`

如果用户没有指定，则从有效候选里选优先级最高的一个：

```python
selected_backend = valid_backends_priorities[selected_index][0]
```

位置：`vllm/platforms/cuda.py:405`

并记录日志：

```text
Using X attention backend out of potential backends: [...]
```

位置：`vllm/platforms/cuda.py:436`

---

## 7. block size、KV cache dtype 为什么会触发 fallback

`AttentionSelectorConfig` 里包含：

```text
block_size
kv_cache_dtype
head_size
dtype
use_mla
use_sparse
use_kv_connector
```

这些字段会进入 backend 的 `validate_configuration()`。

典型 fallback 原因包括：

```text
head_size 不支持；
dtype 不支持；
compute capability 不支持；
kv_cache_dtype 不支持；
block_size 不支持；
需要的 optional dependency 未安装；
MLA / sparse / sink / non-causal 组合不支持；
KV connector 要求的布局或行为不支持。
```

CUDA 平台还有一个专门提示：如果用户指定 `--block-size` 导致更高优先级 backend 被排除，会打印 warning：

```text
--block-size precluded higher-priority backend(s) ...
```

位置：`vllm/platforms/cuda.py:415`

这类 fallback 通常不是 correctness 问题，但可能是性能问题。

---

## 8. selected backend 会反过来影响 KV cache layout

`_cached_get_attn_backend()` 选中 backend 后，会检查：

```python
required_layout = backend.get_required_kv_cache_layout()
```

位置：`vllm/v1/attention/selector.py:132`

如果 backend 要求特定 KV cache layout，就会调用：

```python
set_kv_cache_layout(required_layout)
```

位置：`vllm/v1/attention/selector.py:135`

所以 backend selection 不只是“选 kernel”，还可能改变 KV cache 在内存里的布局约定。

这会影响：

```text
slot mapping
block table
attention metadata
KV connector
CUDA Graph capture
```

---

## 9. ROCm / CPU / XPU 的 attention 选择

### 9.1 ROCm

ROCm 平台逻辑在：`vllm/platforms/rocm.py`

它会考虑：

```text
ROCm GPU 架构；
AITER 是否启用；
FlashAttention Triton backend；
ROCM_ATTN 的 KV cache layout；
MLA / sparse MLA backend；
部分模型支持情况。
```

入口：`vllm/platforms/rocm.py:512`

ROCm 侧常见 backend：

```text
ROCM_ATTN
ROCM_AITER_FA
ROCM_AITER_UNIFIED_ATTN
ROCM_AITER_MLA
ROCM_AITER_TRITON_MLA
ROCM_AITER_MLA_SPARSE
FLASH_ATTN
```

### 9.2 CPU

CPU 平台通常选择 CPU attention backend。

入口：`vllm/platforms/cpu.py:75`

CPU fallback 更关注 correctness 和可运行性，性能模型与 CUDA 完全不同。

### 9.3 XPU

XPU 平台逻辑在：`vllm/platforms/xpu.py`

会在 XPU 可用 backend 中选择，例如 FlashAttention、Triton backend 或 XPU sparse MLA backend，并在不支持时回退到 Triton 等路径。

入口：`vllm/platforms/xpu.py:50`

---

## 10. sampling backend selection

sampling 的 top-k / top-p backend 不走 AttentionBackendEnum，而是在 `TopKTopPSampler.__init__()` 中分派。

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:70`

选择逻辑：

```text
CUDA:
  如果 logprobs_mode 不要求 processed logits/logprobs，且 FlashInfer 可用 → forward_cuda
  否则 → forward_native

CPU:
  RISCV / POWERPC → forward_native
  其他 → forward_cpu

XPU:
  VLLM_XPU_USE_SAMPLER_KERNEL=True → forward_xpu
  否则 → forward_native

ROCm:
  aiter sampling enabled 且 logprobs mode 兼容 → forward_hip
  否则 → forward_native
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:86`

### 10.1 FlashInfer sampler 的 fallback 条件

`flashinfer_sampler_supported()` 会检查：

```text
必须是 CUDA；
VLLM_USE_FLASHINFER_SAMPLER 不能关闭；
FlashInferBackend 支持当前 compute capability；
用户强制开启但不支持时抛 RuntimeError；
默认场景不支持则 warning 后 fallback。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:21`

运行时 `forward_cuda()` 还会继续 fallback：

```text
k 和 p 都为空 → native；
存在 per-request generators → native；
use_fp64_gumbel=True → native；
logprobs_mode 需要 processed 输出 → 不允许 FlashInfer。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:147`

### 10.2 Triton / PyTorch top-k top-p fallback

普通 native 路径调用：

```python
apply_top_k_top_p(logits, k, p)
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:345`

它会根据平台和 batch size 选择：

```text
CPU + HAS_TRITON → Triton；
非 CPU + HAS_TRITON + batch >= 8 → Triton；
其他 → PyTorch sort/topk fallback。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:351`

---

## 11. quantization backend selection

量化 backend 入口是：

```python
def get_quantization_config(quantization: str) -> type[QuantizationConfig]
```

位置：`vllm/model_executor/layers/quantization/__init__.py:107`

它把字符串量化方法映射到 config class：

```text
awq → AWQConfig
fp8 → Fp8Config
auto_gptq / gptq / gptq_marlin → AutoGPTQConfig
awq_marlin → AWQMarlinConfig
compressed-tensors → CompressedTensorsConfig
bitsandbytes → BitsAndBytesConfig
experts_int8 → ExpertsInt8Config
quark → QuarkConfig
moe_wna16 → MoeWNA16Config
torchao → TorchAOConfig
mxfp4 / gpt_oss_mxfp4 → MXFP4 configs
online shorthands → OnlineQuantizationConfig
```

位置：`vllm/model_executor/layers/quantization/__init__.py:140`

这一步只解决“用哪个 QuantizationConfig”，真正每层用哪个 kernel 还要由 config / method 继续决定。

量化 fallback 常见发生点：

```text
- checkpoint 的 quant_method 和用户 --quantization 不一致；
- platform.supported_quantization 不支持该方法；
- group_size / pack_factor / weight bits 不满足 kernel 要求；
- dtype 或 compute capability 不支持 Marlin / CUTLASS / scaled_mm；
- layer 类型不支持量化 method，退回未量化或报错；
- MoE / LoRA / TP 组合受限。
```

---

## 12. quantization selection 和平台能力

`register_quantization_config()` 会在注册自定义 quantization method 时同步更新平台支持列表：

```python
if sq := current_platform.supported_quantization:
    sq.append(quantization)
```

位置：`vllm/model_executor/layers/quantization/__init__.py:93`

这说明量化方法不是只由 Python config 决定，还要看当前平台声明是否支持。

不同 quant config 内部通常还会继续检查：

```text
get_min_capability()
get_supported_act_dtypes()
get_quant_method(layer, prefix)
override_quantization_method(...)
```

这些检查会决定最终使用：

```text
Marlin
AWQ Triton
CUTLASS scaled mm
torchao
bitsandbytes
custom CUDA op
PyTorch fallback
```

---

## 13. MoE backend selection

MoE backend 主要集中在：

```text
vllm/model_executor/layers/fused_moe/
vllm/model_executor/layers/quantization/*moe*
vllm/_custom_ops.py
```

MoE 的 backend 选择维度通常比 dense linear 更多：

```text
num_experts
top_k
hidden size / intermediate size
dtype
quantization method
expert parallel / tensor parallel
all2all backend
grouped topk 是否启用
fused shared experts 是否启用
平台 CUDA / ROCm / XPU
```

常见实现包括：

```text
fused MoE CUDA op
Triton fused MoE
CUTLASS MoE
Marlin / FP8 / INT8 MoE
ROCm aiter MoE
PyTorch fallback
```

MoE fallback 的风险通常是性能，而不是语义：同样的路由和专家计算可以由不同 kernel 完成，但吞吐差异很大。

---

## 14. 基础算子 backend selection

基础算子包括：

```text
RMSNorm / LayerNorm
SiluMul / GeluAndMul
RoPE / M-RoPE
scaled activation
cache reshape / KV cache copy
```

它们经常通过：

```text
vllm/_custom_ops.py
vllm/csrc/
Triton kernel
torch fallback
```

进行分派。

选择维度包括：

```text
current_platform
custom op 是否注册成功
HAS_TRITON
dtype
hidden size
contiguous layout
是否处于 CUDA Graph capture
是否启用 ROCm aiter 相关 env flag
```

这类算子如果 fallback 到 torch，通常不会改变结果，但可能增加 kernel launch 数量和内存带宽压力。

---

## 15. optional dependency import 是 fallback 的常见原因

很多 backend 是 lazy import 的。典型例子：

```python
backend_class = backend.get_class()
```

位置：`vllm/v1/attention/backends/registry.py:128`

如果导入失败：

```text
ImportError → 当前 backend 无效 → 尝试下一个候选
```

位置：`vllm/platforms/cuda.py:335`

sampling 的 ROCm aiter 也是 lazy import：

```python
import aiter.ops.sampling
```

失败后：

```text
forward = forward_native
warning_once("falling back")
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:206`

所以看到 fallback 时，要区分：

```text
依赖没装；
依赖装了但硬件不支持；
依赖支持但当前 shape / dtype 不支持；
用户 env flag 主动禁用。
```

---

## 16. env flags 如何影响 backend

常见 env flag 包括：

```text
VLLM_ATTENTION_BACKEND
VLLM_USE_FLASHINFER_SAMPLER
VLLM_XPU_USE_SAMPLER_KERNEL
VLLM_ROCM_USE_AITER
VLLM_ROCM_USE_AITER_LINEAR
VLLM_ROCM_USE_AITER_RMSNORM
VLLM_USE_TRITON_FLASH_ATTN
VLLM_BATCH_INVARIANT
```

这些 flag 可能影响：

```text
候选 backend 列表；
是否允许某个 optimized kernel；
是否走 batch invariant mode；
是否启用 aiter / Triton / FlashInfer；
不支持时是 warning fallback 还是直接 error。
```

需要注意：

```text
显式指定 backend 通常更严格；
自动选择 backend 通常会继续尝试 fallback。
```

---

## 17. fallback 是正常兼容还是性能问题

可以用下面标准判断。

### 17.1 正常兼容

```text
CPU 平台使用 CPU backend；
小 batch 采样使用 PyTorch native；
没有 top-k / top-p 时 FlashInfer sampler 回退 native；
用户请求 per-request seed，optimized sampler 不支持而回退；
模型 shape 不满足特定 kernel 要求，自动选择另一个 backend。
```

这类 fallback 不一定需要处理。

### 17.2 可能是性能问题

```text
CUDA 上 attention 没有选到 FlashAttention / FlashInfer / Triton；
量化模型没有走 Marlin / CUTLASS / scaled_mm；
MoE 没有走 fused kernel；
top-p 大 batch 走 PyTorch sort；
logprobs 模式导致 optimized sampler 禁用；
block_size 人工设置排除了高优先级 backend。
```

这类需要结合 profiler、日志和吞吐指标判断。

### 17.3 需要修配置或环境

```text
用户显式指定 backend 但 validate_configuration 失败；
optional dependency ImportError；
compute capability 不满足最低要求；
quantization method 在当前平台不支持；
自定义 backend 没有 register；
compile_sizes 和 cudagraph padding 冲突。
```

这类通常会报错或明确 warning。

---

## 18. 如何确认实际 backend

### 18.1 attention

看日志：

```text
Using X attention backend out of potential backends: [...]
Using X KV cache layout for Y backend.
Some attention backends are not valid ... Reasons: {...}
```

相关位置：

```text
vllm/platforms/cuda.py:395
vllm/platforms/cuda.py:436
vllm/v1/attention/selector.py:138
```

### 18.2 sampling

看日志：

```text
Using FlashInfer for top-p & top-k sampling.
FlashInfer top-p/top-k sampling unavailable ... falling back.
aiter sampler ... falling back.
xpu kernel ... falling back.
```

相关位置：`vllm/v1/sample/ops/topk_topp_sampler.py`

### 18.3 quantization

看模型加载日志、quant config、layer 类型和 profiler kernel name。

重点确认：

```text
quant_method
实际 QuantizationConfig
linear layer 的 quant_method
权重参数名：qweight / scales / qzeros / weight_scale 等
profiler 里是否出现 marlin / cutlass / scaled_mm / aiter / triton kernel
```

### 18.4 MoE

看 profiler kernel name 和 fused_moe 相关日志：

```text
fused_moe
grouped_topk
cutlass_moe
marlin_moe
aiter_moe
all2all dispatch / combine
```

---

## 19. backend fallback 与 CUDA Graph 的关系

backend selection 会影响 CUDA Graph：

```text
有些 attention backend 支持 full cudagraph；
有些只能 piecewise；
有些动态 shape / 动态 metadata 会迫使 runtime mode NONE；
fallback 到 torch op 可能引入 graph break 或额外 allocation；
backend 要求的 KV cache layout 会影响 slot mapping 和 capture shape。
```

`GPUModelRunner` 会在初始化 attention metadata builder 后解析 cudagraph mode：

```python
cg_support = builder_cls.get_cudagraph_support(...)
cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:6877`

所以 backend selection 和 cudagraph mode 是互相约束的：

```text
先选 attention backend；
再看 backend 支持哪些 cudagraph mode；
再初始化 cudagraph dispatcher 的 capture keys。
```

---

## 20. 一个完整 attention backend 选择时间线

```text
1. 模型层创建 attention module；
2. 调 get_attn_backend(head_size, dtype, kv_cache_dtype, ...)
3. 构造 AttentionSelectorConfig；
4. 读取 vllm_config.attention_config.backend；
5. current_platform.get_attn_backend_cls(...)；
6. 如果用户指定 backend，先 validate 指定 backend；
7. 如果未指定，生成候选优先级列表；
8. 对每个候选 lazy import + validate_configuration；
9. 选择优先级最高的有效 backend；
10. 如 backend 要求 KV cache layout，则设置 layout；
11. 返回 backend class 给 attention layer 使用。
```

---

## 21. 一个完整 sampling backend 选择时间线

```text
1. ModelRunner 创建 Sampler；
2. Sampler 创建 TopKTopPSampler；
3. TopKTopPSampler 读取 current_platform 和 logprobs_mode；
4. CUDA 上检查 FlashInfer sampler 是否支持；
5. 根据平台把 self.forward 绑定到 forward_cuda / forward_cpu / forward_xpu / forward_hip / forward_native；
6. 每次采样时，forward_* 内部再根据 k、p、generators、use_fp64_gumbel 做运行时 fallback；
7. 最终返回 sampled token 和可选 processed logits/logprobs。
```

---

## 22. 容易疑惑的点

### 22.1 fallback 是否一定代表错误？

不是。自动 backend selection 的目标就是在当前配置下选一个可运行实现。只有当 fallback 导致明显性能退化，或用户显式要求某 backend 却失败时，才需要重点处理。

### 22.2 用户显式指定 backend 会自动回退吗？

attention 场景下通常不会静默回退。指定 backend 无效会抛 `ValueError`，因为这说明用户明确要求了某实现。

位置：`vllm/platforms/cuda.py:360`

### 22.3 为什么同一模型换 GPU 后 backend 变了？

因为 compute capability 会改变候选优先级和 validate 结果。例如 Blackwell 上 FlashInfer / FlashMLA / CUTLASS MLA 的优先级和支持组合可能不同。

### 22.4 为什么打开 logprobs 会影响 sampler backend？

FlashInfer sampler 不支持返回 processed logits / processed logprobs，所以 `logprobs_mode` 需要 processed 输出时会禁用 FlashInfer sampler。

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:87`

### 22.5 为什么 block size 会影响 attention backend？

某些 backend 只支持特定 block size 或需要自动 block size。用户指定 `--block-size` 可能排除更快 backend。

---

## 23. 总结

backend selection 可以压缩为：

```text
config / env / platform
  → selector config
  → candidate priority list
  → validate dtype / shape / capability / dependency
  → selected backend class
  → kernel execution
  → fallback warning or error
```

如果只记住一句话：

```text
vLLM 的 backend 选择是分算子族进行的能力匹配系统：attention 走平台 registry 和 validate_configuration，sampling 在 sampler 内按平台分派，quantization / MoE 由各自 config 和 kernel 条件继续细分。
```
