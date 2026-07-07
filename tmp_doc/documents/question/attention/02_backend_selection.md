# 02. Attention backend 如何选择？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\engine\arg_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\selector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\registry.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\platforms\cuda.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\platforms\rocm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\platforms\cpu.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\mla_attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`

本问题关注：vLLM V1 如何把用户显式配置、模型 attention 形态、KV cache dtype、block size、硬件平台、compute capability、backend 能力约束，最终收敛成一个 `AttentionBackend` 类；以及这个选择结果如何继续影响 attention layer、metadata builder、CUDA graph、KV cache shape 和 forward 实现。

---

## 1. 一句话回答

Attention backend selection 本质上是一个“两段式选择 + 一段式落地”的流程：

```text
1. Attention / MLAAttention 根据当前层形态调用 get_attn_backend(...)
2. selector 把层信息和全局配置打包成 AttentionSelectorConfig
3. current_platform.get_attn_backend_cls(...) 按平台优先级和 backend.validate_configuration() 选出 backend 类
4. Attention layer 用 backend.get_impl_cls() 创建 forward impl
5. GPUModelRunner 按 backend.get_builder_cls() 创建 metadata builder
6. KV cache 初始化按 backend.get_kv_cache_shape() / stride / layout reshape cache tensor
```

所以，backend selection 不是单纯选择一个 kernel 名字，而是选择一整组运行时协议：

```text
backend class
  → impl class
  → metadata builder class
  → supported feature checks
  → KV cache shape / stride / required layout
  → CUDA graph support level
  → forward 中 KV cache update 与 attention kernel 的执行方式
```

---

## 2. 最小主链路

标准 attention 层的主链路是：

```text
Attention.__init__()
  → get_attn_backend(
        head_size,
        dtype,
        kv_cache_dtype,
        use_mla=False,
        has_sink=...,
        use_mm_prefix=...,
        use_per_head_quant_scales=...,
        attn_type=...
    )
  → AttentionSelectorConfig
  → current_platform.get_attn_backend_cls(...)
  → selected AttentionBackend class
  → backend.get_impl_cls()
  → self.impl = impl_cls(...)
```

MLA attention 层的主链路是：

```text
MLAAttention.__init__()
  → get_attn_backend(
        head_size = kv_lora_rank + qk_rope_head_dim,
        dtype,
        kv_cache_dtype,
        use_mla=True,
        use_sparse=...,
        num_heads=...
    )
  → selected MLA AttentionBackend class
  → backend.get_impl_cls()
  → self.impl = impl_cls(...)
  → get_mla_prefill_backend(vllm_config)
  → self.prefill_backend
```

执行时的落地链路是：

```text
GPUModelRunner.initialize_attn_backend()
  → 遍历 KV cache group 中的 attention layer
  → layer.get_attn_backend()
  → 按 backend class + kv_cache_spec + num_heads_q 聚合 AttentionGroup
  → backend.get_builder_cls()
  → AttentionGroup.create_metadata_builders(...)
  → _build_attention_metadata()
  → builder.build(...)
  → set_forward_context(attn_metadata, slot_mapping, ...)
  → Attention.forward() / MLAAttention.forward()
  → backend impl.forward(...) / forward_mha(...) / forward_mqa(...)
```

如果只看一个 token 从选择到执行经历了什么：

```text
模型层初始化时选 backend；
KV cache 初始化时按 backend 形状分配和 reshape；
每轮 execute_model 时按 backend builder 构造 metadata；
forward 时 attention layer 从 forward context 取 metadata 和 kv_cache；
最后调用 backend impl 的 kernel 路径。
```

---

## 3. 先区分几个概念

### 3.1 backend 不是 attention 结构

这些属于模型结构或 attention 语义：

```text
MHA
MQA
GQA
MLA
sliding window
encoder attention
encoder-decoder cross attention
non-causal attention
sparse MLA
```

这些才是 backend / kernel 路径：

```text
FLASH_ATTN
FLASHINFER
FLASHMLA
FLASH_ATTN_MLA
TRITON_ATTN
TRITON_MLA
FLEX_ATTENTION
ROCM_ATTN
ROCM_AITER_FA
CPU_ATTN
TURBOQUANT
```

backend selection 做的是：

```text
根据模型结构和运行条件选择一个能执行它的 backend。
```

### 3.2 PagedAttention 不是某一个 backend

PagedAttention 更像 KV cache 分页管理和访问方式。不同 backend 可以共同消费 block table / slot mapping / paged KV cache，只是它们对 KV cache shape、block size、metadata 和 kernel 的要求不同。

### 3.3 FlashAttention / FlashInfer / Triton 的区别

可以粗略理解为：

```text
FlashAttention：优先使用 flash-attn 系列 kernel，支持 FA2 / FA3 / FA4 版本选择；
FlashInfer：使用 FlashInfer / TRT-LLM attention 路径，Blackwell 上优先级很高；
Triton：vLLM 自有 Triton attention 路径，通常是兼容性兜底和特殊功能承载；
Flex Attention：服务更灵活的 attention mask / multimodal prefix 等场景；
FlashMLA / Triton MLA / FlashInfer MLA：服务 DeepSeek-style MLA 的 decode 或 prefill/decode 组合。
```

---

## 4. 用户配置入口：--attention-backend 和 AttentionConfig

### 4.1 AttentionConfig 里保存显式 backend

`AttentionConfig` 定义在 `config/attention.py`。

关键字段：

```python
backend: AttentionBackendEnum | None = None
flash_attn_version: Literal[2, 3, 4] | None = None
use_trtllm_attention: bool | None = None
mla_prefill_backend: MLAPrefillBackendEnum | None = None
use_non_causal: bool = False
```

位置：`code/vllm/vllm/config/attention.py:16`

其中 `backend` 表示 decode / standard attention backend：

```text
None / "auto"：自动选择；
FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLASHMLA / ...：显式指定。
```

`AttentionConfig.validate_backend_before()` 会把字符串解析成 enum，并把 `"auto"` 变回 `None`。

位置：`code/vllm/vllm/config/attention.py:97`

### 4.2 --attention-backend 和 --attention-config.backend 互斥

CLI 层先复制 `self.attention_config`，再处理 `self.attention_backend`。

核心逻辑：

```text
如果 --attention-backend 非空：
  如果 attention_config.backend 也非空：报错
  否则把 --attention-backend 解析后写入 attention_config.backend
```

位置：`code/vllm/vllm/engine/arg_utils.py:2121`

这意味着：

```text
--attention-backend FLASH_ATTN
```

和：

```text
--attention-config.backend FLASH_ATTN
```

最后都会落到：

```text
vllm_config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
```

但两者不能同时设置。

### 4.3 显式 backend 不是“无条件使用”

显式指定 backend 后，平台层仍然会调用：

```text
backend_class.validate_configuration(...)
```

如果当前模型 / dtype / KV dtype / block size / compute capability 不支持，会直接抛错。

典型语义是：

```text
显式选择 = 指定目标 backend，并要求它必须合法；
自动选择 = 按优先级尝试所有候选，选择第一个合法 backend。
```

---

## 5. registry：名字如何变成 backend class

所有可选 backend 注册在 `AttentionBackendEnum` 中。

位置：`code/vllm/vllm/v1/attention/backends/registry.py:34`

典型映射示例包括：

```text
FLASH_ATTN          → vllm.v1.attention.backends.flash_attn.FlashAttentionBackend
FLASH_ATTN_DIFFKV   → vllm.v1.attention.backends.flash_attn_diff_kv.FlashAttentionDiffKVBackend
FLASHINFER          → vllm.v1.attention.backends.flashinfer.FlashInferBackend
TRITON_ATTN         → vllm.v1.attention.backends.triton_attn.TritonAttentionBackend
TRITON_ATTN_DIFFKV  → vllm.v1.attention.backends.triton_attn_diff_kv.TritonAttentionDiffKVBackend
FLEX_ATTENTION      → vllm.v1.attention.backends.flex_attention.FlexAttentionBackend
FLASHMLA            → vllm.v1.attention.backends.mla.flashmla.FlashMLABackend
FLASH_ATTN_MLA      → vllm.v1.attention.backends.mla.flashattn_mla.FlashAttentionMLABackend
FLASHINFER_MLA      → vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend
TRITON_MLA          → vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend
TOKENSPEED_MLA      → vllm.v1.attention.backends.mla.tokenspeed_mla.TokenSpeedMLABackend
CPU_ATTN            → vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend
TURBOQUANT          → vllm.v1.attention.backends.turbo_attn.TurboAttentionBackend
CUSTOM              → 运行时注册的自定义 backend
```

该列表用于说明 registry 的映射方式，不是完整枚举；源码中还包含 ROCm AITER、sparse MLA 等平台或模型专用 backend。

`AttentionBackendEnum.get_class()` 会通过 `resolve_obj_by_qualname()` 懒加载真实类。

位置：`code/vllm/vllm/v1/attention/backends/registry.py:128`

这个 registry 还支持 override / custom backend：

```text
register_backend(AttentionBackendEnum.FLASH_ATTN, "my.module.MyBackend")
register_backend(AttentionBackendEnum.CUSTOM, "my.module.CustomBackend")
```

位置：`code/vllm/vllm/v1/attention/backends/registry.py:220`

所以 enum 是“名字和默认类路径”，最终类路径可以被运行时 override。

---

## 6. selector：get_attn_backend() 做了什么

入口：`get_attn_backend()`

位置：`code/vllm/vllm/v1/attention/selector.py:54`

它的输入包括：

```text
head_size
model dtype
kv_cache_dtype
use_mla
has_sink
use_sparse
use_mm_prefix
use_per_head_quant_scales
attn_type
num_heads
```

这些输入通常来自 attention layer 的构造参数和全局 config。

### 6.1 selector 会补充全局状态

`get_attn_backend()` 不是只看函数参数。它还会读取当前 `vllm_config`：

```text
cache_config.user_specified_block_size
kv_transfer_config.is_kv_transfer_instance
attention_config.use_non_causal
VLLM_BATCH_INVARIANT
attention_config.backend
```

位置：`code/vllm/vllm/v1/attention/selector.py:75`

然后构造：

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
    attn_type=attn_type or AttentionType.DECODER,
    use_non_causal=vllm_config.attention_config.use_non_causal,
    use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
    use_kv_connector=use_kv_connector,
)
```

位置：`code/vllm/vllm/v1/attention/selector.py:90`

注意 `block_size` 的处理：

```text
只有用户显式指定 --block-size 时，selector 才把 block_size 作为硬约束传给 backend 校验；
如果不是用户指定，则 block_size=None，后面可以由 backend preferred block size 再调整。
```

位置：`code/vllm/vllm/v1/attention/selector.py:79`

### 6.2 selector 本身不写平台优先级

真正选择发生在：

```python
current_platform.get_attn_backend_cls(
    backend,
    attn_selector_config=attn_selector_config,
    num_heads=num_heads,
)
```

位置：`code/vllm/vllm/v1/attention/selector.py:121`

这里的 `backend` 是：

```text
vllm_config.attention_config.backend
```

也就是用户显式指定的 backend，或者 `None`。

### 6.3 结果会被缓存

`_cached_get_attn_backend()` 使用 `@cache`。

位置：`code/vllm/vllm/v1/attention/selector.py:113`

缓存 key 里包括：

```text
selected backend
AttentionSelectorConfig
num_heads
```

这避免同样配置的多个 layer 重复选择和重复 import。

### 6.4 选择后可能调整全局 KV cache layout

选出 backend class 后，selector 会检查：

```python
required_layout = backend.get_required_kv_cache_layout()
```

位置：`code/vllm/vllm/v1/attention/selector.py:132`

如果 backend 要求特定 layout，则调用：

```text
set_kv_cache_layout(required_layout)
```

这说明 backend selection 不只是返回一个类，还可能影响后续 KV cache 全局布局策略。

---

## 7. backend 抽象：一个 backend 必须提供什么

`AttentionBackend` 抽象类定义在 `backend.py`。

位置：`code/vllm/vllm/v1/attention/backend.py:55`

一个 backend 最核心要提供：

```text
get_name()
get_impl_cls()
get_builder_cls()
get_kv_cache_shape()
```

位置：`code/vllm/vllm/v1/attention/backend.py:73`

分别对应：

| 方法 | 含义 |
|---|---|
| `get_name()` | backend 名字，用于日志和 enum 对齐 |
| `get_impl_cls()` | attention forward 的具体实现类 |
| `get_builder_cls()` | attention metadata builder 类 |
| `get_kv_cache_shape()` | 此 backend 期望的 KV cache tensor 逻辑形状 |
| `get_kv_cache_stride_order()` | 此 backend 期望的 KV cache 物理 stride 顺序 |
| `get_supported_head_sizes()` | 支持哪些 head size；空列表表示不限制 |
| `get_supported_kernel_block_sizes()` | 支持哪些 kernel block size |
| `get_required_kv_cache_layout()` | 是否强制某种全局 KV cache layout |

### 7.1 validate_configuration() 是统一校验入口

`AttentionBackend.validate_configuration()` 会检查：

```text
head_size
model dtype
kv_cache_dtype
block_size
MLA / non-MLA
attention sink
sparse / non-sparse
multimodal prefix full attention
per-head quant scales
compute capability
attention type
non-causal attention
batch invariance
KV connector
backend 自定义组合条件
```

位置：`code/vllm/vllm/v1/attention/backend.py:276`

它返回的是：

```text
[]：合法
[reason1, reason2, ...]：不合法原因
```

这也是自动选择能打印“哪些 backend 被排除、为什么被排除”的基础。

### 7.2 supports_combination() 用于复杂约束

简单约束可以靠 `supports_dtype()` / `supports_head_size()` / `supports_block_size()`，但有些约束是组合条件，例如：

```text
某 backend 只支持某个 head_size + kv_cache_dtype 组合；
某 backend 只支持某些 MLA 维度；
某 backend 在某 compute capability 下支持 fp8，但其他 GPU 不支持；
某 backend 只支持特定 block size 和 kv dtype 的组合。
```

这类条件通过：

```text
supports_combination(...)
```

返回具体不合法原因。

位置：`code/vllm/vllm/v1/attention/backend.py:261`

---

## 8. CUDA 平台如何选择 backend

CUDA 平台选择逻辑在 `platforms/cuda.py`。

### 8.1 CUDA 的候选优先级

优先级函数是 `_get_backend_priorities()`。

位置：`code/vllm/vllm/platforms/cuda.py:79`

它按两个大维度分支：

```text
use_mla = False：标准 MHA / MQA / GQA attention
use_mla = True ：DeepSeek-style MLA attention
```

再按 device capability 分支：

```text
major == 10：Blackwell / SM 10.x
其他：Ampere / Hopper / 旧 CUDA GPU 路径
```

### 8.2 标准 attention 的 CUDA 自动优先级

Blackwell / SM 10.x：

```text
1. FLASHINFER
2. FLASH_ATTN
3. TRITON_ATTN
4. FLEX_ATTENTION
5. TURBOQUANT
```

位置：`code/vllm/vllm/platforms/cuda.py:133`

非 Blackwell：

```text
1. FLASH_ATTN
2. FLASHINFER
3. TRITON_ATTN
4. FLEX_ATTENTION
5. TURBOQUANT
```

位置：`code/vllm/vllm/platforms/cuda.py:141`

直观理解：

```text
Hopper/Ampere：FlashAttention 优先；
Blackwell：FlashInfer / TRT-LLM 路径优先；
Triton/Flex/TurboQuant 作为后续可用候选。
```

### 8.3 MLA 的 CUDA 自动优先级

Blackwell / SM 10.x：

```text
1. FLASHINFER_MLA
2. TOKENSPEED_MLA
3. CUTLASS_MLA
4. FLASH_ATTN_MLA
5. FLASHMLA
6. TRITON_MLA
7. sparse MLA 候选
```

位置：`code/vllm/vllm/platforms/cuda.py:87`

非 Blackwell：

```text
1. FLASH_ATTN_MLA
2. FLASHMLA
3. FLASHINFER_MLA
4. TRITON_MLA
5. FLASHMLA_SPARSE
```

位置：`code/vllm/vllm/platforms/cuda.py:124`

这里要注意两点：

```text
1. MLA 的 prefill backend 和 decode backend 是两套选择。
2. priority list 可能同时包含 sparse / non-sparse 候选，最终由 validate_configuration() 按 use_sparse == backend.is_sparse() 过滤另一类。
```

`AttentionConfig.backend` / `--attention-backend` 传给 `get_attn_backend(use_mla=True)`，主要决定 MLA attention backend / decode 主路径；MLA prefill backend 由 `mla_prefill_backend` 或自动逻辑决定，入口在 `get_mla_prefill_backend(vllm_config)`。因此 `--attention-backend FLASHMLA` 不等价于 prefill 也走 FlashMLA。

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:472`

### 8.4 Blackwell sparse MLA 的特殊排序

CUDA Blackwell 上 sparse MLA 后端会根据 KV cache dtype 和 head 数选择顺序。

位置：`code/vllm/vllm/platforms/cuda.py:89`

逻辑是：

```text
如果 KV cache 是 fp8 / quantized：
  FLASHINFER_MLA_SPARSE 优先于 FLASHMLA_SPARSE

如果 KV cache 是 bf16 且 num_heads <= 16：
  FLASHINFER_MLA_SPARSE 优先

否则：
  FLASHMLA_SPARSE 优先
```

这说明 `num_heads` 不是普通标准 attention 必需项，但对某些 MLA sparse backend 的优先级会产生影响。

### 8.5 CUDA 手动选择逻辑

`CudaPlatformBase.get_attn_backend_cls()` 首先处理 `selected_backend`。

位置：`code/vllm/vllm/platforms/cuda.py:351`

如果用户显式指定：

```text
1. selected_backend.get_class()
2. backend_class.validate_configuration(...)
3. 如果 invalid_reasons 非空：raise ValueError
4. 否则返回 selected_backend.get_path()
```

位置：`code/vllm/vllm/platforms/cuda.py:360`

所以显式选择失败时不会自动 fallback。

### 8.6 CUDA 自动选择逻辑

如果没有显式指定：

```text
1. get_valid_backends(...)
2. 按优先级遍历候选 backend
3. 对每个候选调用 validate_configuration(...)
4. 收集合法 backend 和不合法原因
5. 选择 priority 最小的合法 backend
6. 如果没有合法 backend，抛出包含所有原因的 ValueError
```

位置：`code/vllm/vllm/platforms/cuda.py:316`

如果用户指定了 `--block-size`，且这个 block size 排除了更高优先级 backend，会打印 warning。

位置：`code/vllm/vllm/platforms/cuda.py:417`

这解释了一个常见现象：

```text
同一模型同一 GPU 上，不指定 --block-size 可能选 FLASH_ATTN；
指定某个不兼容 block size 后，可能退到 TRITON_ATTN / FLEX_ATTENTION。
```

---

## 9. ROCm 平台如何选择 backend

ROCm 平台选择逻辑在 `platforms/rocm.py`。

### 9.1 ROCm 的候选优先级

入口是 `_get_backend_priorities()`。

位置：`code/vllm/vllm/platforms/rocm.py:389`

它按三类分支：

```text
use_sparse
use_mla
standard attention
```

### 9.2 ROCm sparse MLA

如果 `use_sparse=True`：

```text
ROCM_AITER_MLA_SPARSE
```

位置：`code/vllm/vllm/platforms/rocm.py:396`

### 9.3 ROCm MLA

如果 `use_mla=True`：

```text
如果 AITER MLA enabled：
  1. ROCM_AITER_MLA
  2. TRITON_MLA
  3. ROCM_AITER_TRITON_MLA

否则：
  1. TRITON_MLA
```

位置：`code/vllm/vllm/platforms/rocm.py:399`

### 9.4 ROCm standard attention

标准 attention 候选：

```text
1. ROCM_ATTN        # 但 KV connector 场景会跳过
2. ROCM_AITER_FA    # AITER MHA enabled 时加入
3. ROCM_AITER_UNIFIED_ATTN
4. TRITON_ATTN
5. TURBOQUANT
```

位置：`code/vllm/vllm/platforms/rocm.py:411`

一个重要细节：

```text
ROCM_ATTN 使用 (2, num_blocks, ...) KV cache layout；
这和要求 blocks-first layout 的 KV connector 不兼容；
所以 use_kv_connector=True 时不会把 ROCM_ATTN 加入候选。
```

位置：`code/vllm/vllm/platforms/rocm.py:411`

### 9.5 ROCm 的校验和选择流程

ROCm 和 CUDA 一样：

```text
显式 backend：只校验这个 backend，不合法就报错；
自动 backend：按优先级遍历候选，选择第一个合法 backend。
```

位置：`code/vllm/vllm/platforms/rocm.py:478`

---

## 10. CPU / XPU / ViT 的特殊路径

### 10.1 CPU backend

CPU 平台的 attention backend 很直接：

```text
CPU_ATTN
```

位置：`code/vllm/vllm/platforms/cpu.py:75`

如果用户显式指定了非 `CPU_ATTN` backend，CPU 平台只会记录日志，最终仍返回 `CPU_ATTN`。

同时：

```text
MLA is not supported on CPU
Sparse Attention is not supported on CPU
```

位置：`code/vllm/vllm/platforms/cpu.py:81`

### 10.2 XPU

XPU 继承通用平台逻辑并有自己的 config update / block size update。对 GDN attention 这类内核，还会把 block size 对齐到 kernel 支持的大小。

位置：`code/vllm/vllm/platforms/xpu.py:246`

### 10.3 ViT attention 不是普通 decoder attention selector

平台基类提供 `get_vit_attn_backend()`，默认是 `TORCH_SDPA`。

位置：`code/vllm/vllm/platforms/interface.py:278`

CUDA / ROCm 会 override ViT backend 选择，例如 CUDA 会在：

```text
FLASH_ATTN
TRITON_ATTN
TORCH_SDPA
FLASHINFER
```

之间选择。

位置：`code/vllm/vllm/platforms/cuda.py:444`

所以 ViT attention backend 选择不完全等同于 decoder attention 的 `get_attn_backend()` 主链路。

---

## 11. Attention layer 如何触发 backend selection

### 11.1 标准 Attention 层

入口：`Attention.__init__()`

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:190`

如果调用者没有显式传入 `attn_backend`，就会调用：

```python
self.attn_backend = get_attn_backend(
    head_size,
    dtype,
    kv_cache_dtype,
    use_mla=False,
    has_sink=self.has_sink,
    use_mm_prefix=self.use_mm_prefix,
    use_per_head_quant_scales=use_per_head_quant_scales,
    attn_type=attn_type,
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:304`

这里几个参数的来源很关键：

| 参数 | 来源 | 作用 |
|---|---|---|
| `head_size` | 模型 attention head dim | 过滤不支持该 head size 的 backend |
| `dtype` | 当前默认 dtype，通常是模型 dtype | 过滤 fp16 / bf16 / fp32 支持 |
| `kv_cache_dtype` | `cache_config.cache_dtype` | 过滤 fp8 / fp16 / bf16 / turboquant 等 KV cache 支持 |
| `has_sink` | `extra_impl_args.get("sinks")` | StreamingLLM sink 支持 |
| `use_mm_prefix` | `model_config.is_mm_prefix_lm` | 多模态 prefix bidirectional/full attention 支持 |
| `use_per_head_quant_scales` | checkpoint kv_cache_scheme | per-head KV quant scale 支持 |
| `attn_type` | decoder / encoder / encoder_decoder | 不同 attention 类型支持 |

选出 backend 后：

```text
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:373`

### 11.2 MLAAttention 层

入口：`MLAAttention.__init__()`

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:322`

如果没有显式传入 `attn_backend`：

```python
self.attn_backend = get_attn_backend(
    self.head_size,
    dtype,
    kv_cache_dtype,
    use_mla=True,
    use_sparse=use_sparse,
    num_heads=self.num_heads,
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:386`

MLA 的 `head_size` 不是普通 attention 的 head dim，而是：

```text
kv_lora_rank + qk_rope_head_dim
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:363`

因此 MLA backend 的 head size 判断要按 MLA cache 表示理解，而不是普通 Q/K/V head dim。

### 11.3 显式传入 attn_backend 的绕过路径

`Attention` 和 `MLAAttention` 构造函数都支持直接传入 `attn_backend`。

```text
attn_backend is not None：直接使用传入 backend，不再调用 get_attn_backend()
```

标准 attention 位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:315`

MLA 位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:379`

这种路径常用于特殊模型包装、测试或动态 subclass backend。

---

## 12. GPUModelRunner 如何使用选择结果

backend selection 发生在 attention layer 初始化时，但真正“批处理和执行层”是在 `GPUModelRunner` 里把 backend 组织起来。

### 12.1 initialize_attn_backend() 按 KV cache group 聚合 backend

`GPUModelRunner.initialize_attn_backend()` 会遍历 `kv_cache_config.kv_cache_groups`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6763`

它对每个 layer 调用：

```python
attn_backend = layers[layer_name].get_attn_backend()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6777`

然后按这个 key 聚合：

```text
full backend class name
layer KVCacheSpec
num_heads_q
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6796`

聚合的结果是 `AttentionGroup`。

```text
同一 KV cache group 内，backend 相同、KV spec 相同、num_heads_q 相同的 layer 可以共享 metadata builder。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6807`

### 12.2 backend 会影响 CUDA graph 模式

在真正创建 builder 前，ModelRunner 会先检查所有 backend builder 的 CUDA graph 支持级别：

```python
builder_cls = attn_backend.get_builder_cls()
cg_support = builder_cls.get_cudagraph_support(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6895`

然后取最保守的支持级别，交给：

```text
compilation_config.resolve_cudagraph_mode_and_sizes(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6904`

所以 backend 不只是影响 attention kernel，还会影响 CUDA graph capture 能不能开、怎么开、capture size 怎么定。

### 12.3 initialize_metadata_builders() 创建 builder

`initialize_metadata_builders()` 会让每个 `AttentionGroup` 创建 metadata builder。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6843`

核心关系：

```text
AttentionGroup.backend
  → backend.get_builder_cls()
  → AttentionMetadataBuilder
  → build(common_attn_metadata)
  → per-layer AttentionMetadata
```

后续 `_build_attention_metadata()` 每轮执行时就会调用这些 builder。

---

## 13. backend 如何影响 attention metadata

`AttentionMetadataBuilder` 是 backend 和 ModelRunner 之间的协议层。

位置：`code/vllm/vllm/v1/attention/backend.py:533`

它至少要实现：

```text
build(common_prefix_len, common_attn_metadata, fast_build=False)
```

位置：`code/vllm/vllm/v1/attention/backend.py:599`

输入是通用的 `CommonAttentionMetadata`，输出是 backend 自己的 `AttentionMetadata`。

### 13.1 CommonAttentionMetadata 是公共输入

`CommonAttentionMetadata` 包含：

```text
query_start_loc
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
causal
encoder_seq_lens
positions
is_prefilling
mm_req_doc_ranges
```

位置：`code/vllm/vllm/v1/attention/backend.py:361`

这是一份“所有 backend 都可能用到的 batch 描述”。

### 13.2 每个 backend builder 会翻译成自己的 metadata

例如：

```text
FlashAttention builder 关心 varlen / cu_seqlens / block table / cascade 等；
Triton builder 主要准备 Triton kernel 所需的统一 paged attention metadata 和 scratch / segment buffer，不像 FlashInfer / MLA 那样显式拆出 prefill / decode 子对象；
FlashInfer builder 会显式构造 prefill / decode wrapper metadata；
MLA builder 会额外拆 num_decodes / num_prefills / chunked_context / prefill_backend；
Flex builder 会构造更灵活的 block mask / prefix mask。
```

这就是为什么 backend selection 会影响 `_build_attention_metadata()` 的产物。

### 13.3 builder 还能要求 batch reorder

`AttentionMetadataBuilder` 有：

```text
reorder_batch_threshold
```

位置：`code/vllm/vllm/v1/attention/backend.py:539`

ModelRunner 会收集所有 group 的 threshold，取最保守值。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6933`

这会影响 decode / prefill 混合 batch 的排列方式。

---

## 14. backend 如何影响 KV cache

### 14.1 backend 决定 KV cache shape

KV cache tensor 分配后会被 reshape 成 backend 期望的形状。

在 `GPUModelRunner._reshape_kv_cache_tensors()` 中：

```python
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks,
    shape_block_size,
    kv_cache_spec.num_kv_heads,
    kv_cache_spec.head_size,
    cache_dtype_str=self.cache_config.cache_dtype,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7103`

这意味着不同 backend 可以有不同 KV cache layout，例如：

```text
标准 attention 常见形状可能包含 K/V 维度；
MLA cache 通常存 compressed latent KV + rope 部分；
TurboQuant / sparse / special backend 可能有更特殊的 packed layout。
```

### 14.2 backend 还能决定 stride order

ModelRunner 会尝试读取：

```text
attn_backend.get_kv_cache_stride_order()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7111`

如果 backend 实现了这个方法，就按它返回的顺序重排物理维度。

这说明：

```text
KV cache 的逻辑 shape 和物理 memory layout 可以不同；
backend selection 会同时决定这两者。
```

### 14.3 backend 可以影响 block size

平台在模型加载后会调用：

```text
current_platform.update_block_size_for_backend(vllm_config)
```

例如单进程 executor 在 load_model 后调用。

位置：`code/vllm/vllm/v1/executor/uniproc_executor.py:69`

通用逻辑会找到模型里的第一个 non-SSM attention backend，然后：

```text
如果用户没显式指定 block_size：
  cache_config.block_size = backend_cls.get_preferred_block_size(DEFAULT_BLOCK_SIZE)
```

位置：`code/vllm/vllm/platforms/interface.py:512`

这解释了为什么 selector 里用户没指定 block size 时传 `None`：

```text
先选 backend，再让 backend 反过来决定推荐 block size。
```

---

## 15. backend 如何影响 forward

### 15.1 标准 Attention.forward()

`Attention.forward()` 会：

```text
1. reshape query/key/value
2. 必要时做 query quantization
3. 从 forward context 取 attention metadata / kv_cache / slot_mapping
4. 调用 unified_attention_with_output(...)
5. unified_attention_with_output 再调用 self.impl.forward(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:438`

最终调用：

```python
self.impl.forward(
    self,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=output,
    ...
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:753`

这里的 `self.impl` 就是 backend selection 后的 `backend.get_impl_cls()`。

### 15.2 backend 决定 KV cache update 是否在 forward 内完成

`AttentionBackend` 有：

```text
forward_includes_kv_cache_update: bool = True
```

位置：`code/vllm/vllm/v1/attention/backend.py:65`

如果某 backend 不把 KV cache update 包在 forward 里，`Attention.forward()` 会先调用：

```text
unified_kv_cache_update(...)
```

再调用 attention。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:491`

这说明 backend selection 会影响：

```text
KV cache 写入和 attention kernel 是融合执行，还是分成两个 op。
```

### 15.3 MLA forward 分 prefill 和 decode

MLA impl 有两类 forward：

```text
forward_mha：prefill / compute-friendly path
forward_mqa：decode / data-movement-friendly path
```

位置：`code/vllm/vllm/v1/attention/backend.py:892`

`MLAAttention.forward_impl()` 会根据 metadata 中的：

```text
num_mha_tokens
num_mqa_tokens
```

分别调用：

```text
self.impl.forward_mha(...)
self.impl.forward_mqa(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:699`

这就是为什么 MLA backend selection 比标准 attention 更复杂：

```text
decode backend、prefill backend、metadata builder、chunked context workspace、KV cache compressed layout 都会参与。
```

---

## 16. 自动选择和显式选择的差异

### 16.1 自动选择

自动选择语义：

```text
backend=None
  → 平台给出候选优先级列表
  → 每个候选 validate_configuration()
  → 第一个合法 backend 胜出
```

如果没有合法 backend：

```text
ValueError: No valid attention backend found ... Reasons: {...}
```

### 16.2 显式选择

显式选择语义：

```text
backend=FLASH_ATTN
  → 只校验 FLASH_ATTN
  → 合法则使用
  → 不合法则直接报错，不 fallback
```

如果不合法：

```text
ValueError: Selected backend FLASH_ATTN is not valid for this configuration. Reason: [...]
```

### 16.3 为什么不 fallback

因为显式选择通常代表用户想要固定某个 kernel 路径来做 benchmark、复现、规避 bug 或验证性能。如果默默 fallback，会让结果不可控。

---

## 17. 常见条件如何影响选择

### 17.1 dtype

`dtype` 来自模型加载时的默认 dtype，通常是 fp16 / bf16 / fp32。

backend 会通过：

```text
supports_dtype(dtype)
```

过滤。

位置：`code/vllm/vllm/v1/attention/backend.py:162`

### 17.2 KV cache dtype

`kv_cache_dtype` 来自 `cache_config.cache_dtype`。

常见值：

```text
auto
float16
bfloat16
fp8
fp8_e4m3
fp8_e5m2
fp8_ds_mla
turboquant_...
```

backend 会通过：

```text
supports_kv_cache_dtype(kv_cache_dtype)
```

过滤。

位置：`code/vllm/vllm/v1/attention/backend.py:167`

### 17.3 block size

如果用户显式指定 `--block-size`，selector 会把它传给 backend 校验。

backend 会通过：

```text
supports_block_size(block_size)
```

过滤。

位置：`code/vllm/vllm/v1/attention/backend.py:174`

如果用户没有指定 block size，则：

```text
selector 传 block_size=None；
平台先选 backend；
后续 update_block_size_for_backend() 根据 backend preferred block size 调整 cache_config.block_size。
```

### 17.4 compute capability

CUDA / ROCm 平台会拿当前设备能力：

```text
DeviceCapability(major, minor)
```

传给 backend 校验。

CUDA 获取位置：`code/vllm/vllm/platforms/cuda.py:218`

ROCm 获取位置：`code/vllm/vllm/platforms/rocm.py:665`

backend 会通过：

```text
supports_compute_capability(device_capability)
```

过滤。

位置：`code/vllm/vllm/v1/attention/backend.py:256`

### 17.5 attention type

`attn_type` 可能是：

```text
decoder
encoder
encoder_only
encoder_decoder
```

定义位置：`code/vllm/vllm/v1/attention/backend.py:32`

backend 会通过：

```text
supports_attn_type(attn_type)
```

过滤。

默认 backend 只支持 decoder，其他类型需要 backend override。

位置：`code/vllm/vllm/v1/attention/backend.py:248`

### 17.6 multimodal prefix / non-causal / sink

这些特殊能力分别通过：

```text
supports_mm_prefix()
supports_non_causal()
supports_sink()
```

过滤。

位置：`code/vllm/vllm/v1/attention/backend.py:208`

例如多模态 prefix LM 要求部分 multimodal token 能做 full / bidirectional attention，不支持该能力的 backend 会被排除。

### 17.7 KV connector

如果启用了 KV transfer / KV connector：

```text
use_kv_connector=True
```

selector 会把它写入 `AttentionSelectorConfig`。

位置：`code/vllm/vllm/v1/attention/selector.py:85`

backend 会通过：

```text
supports_kv_connector()
```

过滤。

位置：`code/vllm/vllm/v1/attention/backend.py:244`

ROCm 还会在候选阶段直接跳过某些 layout 不兼容的 backend，例如 `ROCM_ATTN`。

位置：`code/vllm/vllm/platforms/rocm.py:411`

---

## 18. MLA backend 选择的特殊点

### 18.1 use_mla 会把标准 backend 排除

`validate_configuration()` 中有：

```text
if use_mla != cls.is_mla():
  use_mla=True  且 backend 非 MLA → MLA not supported
  use_mla=False 且 backend 是 MLA → non-MLA not supported
```

位置：`code/vllm/vllm/v1/attention/backend.py:306`

所以：

```text
标准 Attention 不会误选 MLA backend；
MLAAttention 不会误选标准 backend。
```

### 18.2 sparse MLA 是另一个维度

`sparse` 不是普通 MLA 的附加开关，而会要求：

```text
use_sparse == backend.is_sparse()
```

位置：`code/vllm/vllm/v1/attention/backend.py:313`

因此：

```text
use_sparse=True：只能选 sparse backend；
use_sparse=False：sparse backend 也会被排除。
```

### 18.3 MLA prefill backend 单独选择

MLAAttention 在 decode backend 选完后，还会创建：

```text
self.prefill_backend = get_mla_prefill_backend(vllm_config)(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:472`

这意味着一个 MLA layer 里可能同时存在：

```text
decode attention backend：FLASHMLA / TRITON_MLA / FLASHINFER_MLA / ...
prefill backend：FLASH_ATTN / FLASHINFER / TRTLLM_RAGGED / TOKENSPEED_MLA / ...
```

所以看到 `FLASHMLA` 不代表 prefill 也走 FlashMLA。

### 18.4 MLA KV cache shape 不同

`MLACommonBackend.get_kv_cache_shape()` 返回：

```text
(num_blocks, block_size, head_size)
```

位置：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py:1195`

普通 attention backend 通常需要同时存 K 和 V；MLA 存的是 compressed latent KV 与 rope 相关部分，语义不同。

---

## 19. 和 KV cache / block table / slot mapping 的关系

backend selection 直接影响三件事：

```text
1. KV cache tensor 怎么分配和 reshape
2. block table / slot mapping 如何进入 metadata
3. forward kernel 如何解释这些 metadata 和 cache tensor
```

但它不负责：

```text
请求被分配哪些 block；
slot_mapping 当前轮具体是什么；
Scheduler 如何决定 prefill / decode token 数。
```

这些由 Scheduler、KV cache manager、InputBatch 和 `_build_attention_metadata()` 共同完成。

更准确的边界是：

```text
Scheduler / KV manager：决定资源和本轮计划；
InputBatch：保存请求状态和 block table；
ModelRunner：构造 CommonAttentionMetadata；
backend builder：把 CommonAttentionMetadata 翻译成 backend metadata；
backend impl：按 metadata 和 KV cache 执行 kernel。
```

---

## 20. 一个完整例子：CUDA Hopper 上普通 Llama

假设：

```text
GPU：Hopper，compute capability 9.x
模型：普通 decoder-only Llama
use_mla=False
kv_cache_dtype=auto
没有 sink
没有 multimodal prefix
没有显式 --attention-backend
```

流程：

```text
Attention.__init__()
  → get_attn_backend(use_mla=False, head_size=128, dtype=bf16, kv_cache_dtype=auto)
  → AttentionSelectorConfig(...)
  → CudaPlatform.get_attn_backend_cls(None, config)
  → CUDA non-Blackwell standard priority:
       FLASH_ATTN
       FLASHINFER
       TRITON_ATTN
       FLEX_ATTENTION
       TURBOQUANT
  → validate FLASH_ATTN
  → 合法则选择 FLASH_ATTN
  → Attention.impl = FlashAttentionImpl(...)
  → GPUModelRunner 创建 FlashAttention metadata builder
  → KV cache 按 FlashAttention backend shape reshape
  → forward 调用 FlashAttention impl
```

如果 FlashAttention 不支持当前条件，例如 dtype / import / compute capability / block size 不合适，则继续尝试 FlashInfer、Triton 等。

---

## 21. 一个完整例子：CUDA Blackwell 上普通 attention

假设：

```text
GPU：Blackwell，compute capability 10.x
模型：普通 decoder-only
没有显式 backend
```

CUDA 标准 attention 优先级变成：

```text
FLASHINFER
FLASH_ATTN
TRITON_ATTN
FLEX_ATTENTION
TURBOQUANT
```

位置：`code/vllm/vllm/platforms/cuda.py:133`

所以只要 FlashInfer 当前配置合法，它会优先于 FlashAttention。

这也是为什么同一个模型从 H100 切到 B200 / GB200，默认 attention backend 可能变化。

---

## 22. 一个完整例子：DeepSeek-style MLA

假设：

```text
模型：DeepSeek V2/V3/R1 style MLA
use_mla=True
KV cache dtype=fp8
GPU：Blackwell
没有显式 backend
```

流程：

```text
MLAAttention.__init__()
  → head_size = kv_lora_rank + qk_rope_head_dim
  → get_attn_backend(use_mla=True, use_sparse=False, num_heads=...)
  → CUDA Blackwell MLA priority:
       FLASHINFER_MLA
       TOKENSPEED_MLA
       CUTLASS_MLA
       FLASH_ATTN_MLA
       FLASHMLA
       TRITON_MLA
       sparse MLA candidates
  → validate_configuration 逐个检查 dtype / kv_cache_dtype / block_size / head_size / capability / combination
  → 选择第一个合法 decode backend
  → 另外调用 get_mla_prefill_backend(vllm_config) 选择 prefill backend
  → builder 构造 MLACommonMetadata，按 num_decodes / num_prefills 拆路径
  → forward 中 prefill 调 forward_mha，decode 调 forward_mqa
```

注意：

```text
MLA decode backend 和 MLA prefill backend 是两个选择点。
```

---

## 23. 容易疑惑的点

### 23.1 get_attn_backend() 返回的是实例吗？

不是。

它返回的是：

```text
type[AttentionBackend]
```

也就是 backend 类。真正实例化的是：

```text
backend.get_impl_cls() 返回的 impl class
backend.get_builder_cls() 返回的 builder class
```

### 23.2 backend 是每个请求选一次吗？

不是。

backend 通常在模型 layer 初始化时选择，之后按 layer / group 固定。每轮请求变化时重建的是 metadata，不是重新选择 backend。

### 23.3 为什么 selector 要缓存？

因为同一模型有很多 attention layer，它们通常有相同 head size、dtype、KV dtype 和 feature 条件。缓存可以避免重复 import 和重复平台选择。

### 23.4 为什么用户指定 block size 会影响 backend？

因为某些 backend kernel 只支持固定 block size 或 block size 倍数。用户显式 `--block-size` 会成为硬约束，可能排除更高优先级 backend。

### 23.5 为什么自动选择里 block_size 可能是 None？

因为如果用户没有指定 block size，vLLM 允许先选 backend，再由 backend 推荐 block size。这样可以避免默认 block size 反过来误伤高性能 backend。

### 23.6 backend 和 metadata builder 是一对一吗？

大体是一对一，但不绝对。

vLLM 支持动态 subclass backend，例如某些 fast prefill / chunked local attention 场景可能包装已有 backend 并替换 builder。

相关工具函数：

```text
subclass_attention_backend(...)
subclass_attention_backend_with_overrides(...)
```

位置：`code/vllm/vllm/v1/attention/backend.py:1034`

### 23.7 FlashAttention 版本在哪里选？

FlashAttention backend 内部会参考：

```text
attention_config.flash_attn_version
compute capability
```

来决定 FA2 / FA3 / FA4。`--attention-backend FLASH_ATTN` 只是选择 FlashAttention backend 家族，不等于固定 FA2 / FA3 / FA4。

### 23.8 TORCH_SDPA 为什么不在普通 decoder backend priority 里常见？

在当前 V1 主链路里，`TORCH_SDPA` 更常作为 ViT attention 的候选 / fallback。普通 decoder paged KV cache attention 主要通过 FlashAttention / FlashInfer / Triton / Flex / 平台 backend 等实现。

### 23.9 backend 选择失败时应该看哪里？

优先看报错里的：

```text
AttentionSelectorConfig(...)
Reasons: {BACKEND: [reason...]}
```

常见 reason：

```text
head_size not supported
dtype not supported
kv_cache_dtype not supported
block_size not supported
MLA not supported
sparse not supported
attention sinks not supported
compute capability not supported
attention type encoder_decoder not supported
non-causal attention not supported
KV connector not supported
ImportError
```

---

## 24. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心输入 | 核心输出 | 作用 |
|---|---|---|---|---|
| 配置解析 | `AttentionConfig` / `arg_utils` | `--attention-backend` / `--attention-config` | `attention_config.backend` | 保存显式 backend 或 auto |
| 层级触发 | `Attention.__init__()` / `MLAAttention.__init__()` | head size、dtype、KV dtype、MLA 标志 | `get_attn_backend(...)` 调用 | 从模型层发起选择 |
| selector 打包 | `get_attn_backend()` | 层参数 + 当前 `vllm_config` | `AttentionSelectorConfig` | 构造统一选择条件 |
| 平台选择 | `current_platform.get_attn_backend_cls()` | selected backend + selector config | backend class path | 按平台优先级和校验选 backend |
| backend 校验 | `AttentionBackend.validate_configuration()` | dtype、head、block、capability 等 | invalid reasons | 判断 backend 是否支持当前配置 |
| impl 落地 | `backend.get_impl_cls()` | backend class | `AttentionImpl` / `MLAAttentionImpl` | 真正执行 forward kernel |
| builder 落地 | `backend.get_builder_cls()` | backend class | `AttentionMetadataBuilder` | 构造每轮 attention metadata |
| cache 落地 | `backend.get_kv_cache_shape()` | num_blocks、block_size、head 信息 | KV cache shape | 决定 KV cache tensor 形状 |
| 执行落地 | `Attention.forward()` / `MLAAttention.forward()` | forward context + metadata + kv_cache | attention output | 调用 backend impl |

---

## 25. 总结

Attention backend selection 可以压缩成下面这条线：

```text
用户配置 / auto
  → AttentionConfig.backend
  → Attention / MLAAttention 构造时调用 get_attn_backend()
  → AttentionSelectorConfig
  → current_platform.get_attn_backend_cls()
  → backend.validate_configuration()
  → selected AttentionBackend class
  → get_impl_cls() / get_builder_cls() / get_kv_cache_shape()
  → AttentionGroup / metadata builder / KV cache reshape
  → forward context
  → backend impl 执行 attention
```

如果只记住一句话：

```text
vLLM 的 attention backend selection 不是单点 if-else，而是把“模型层需求 + 用户配置 + 平台能力 + backend 能力声明”统一校验后，选出一整套 attention 执行协议。
```

再压缩成最小心智模型：

```text
selector 负责收集条件；
platform 负责优先级；
backend 负责声明能力；
Attention layer 负责实例化 impl；
ModelRunner 负责实例化 builder；
KV cache 初始化和 forward 执行都受 backend 选择结果影响。
```
