# 04 CUDA 核心算子：Attention 与 KV Cache

## 1. 为什么 Attention/KV Cache 是底层核心

vLLM 的高吞吐推理主要依赖：

1. PagedAttention：把 KV cache 组织成 block/page，支持动态 batch 和长上下文。
2. 高效 KV cache 写入/搬运/聚合：prefill/decode 中频繁 reshape/cache/gather/swap。
3. fused RoPE/QK norm/cache insert：减少 kernel launch 和中间 tensor。
4. MLA / split-KV / FlashInfer/CUTLASS 等针对特定模型和硬件的优化路径。

这些能力大多在 `csrc/libtorch_stable` 中实现，并通过 `torch.ops._C` 暴露给 Python。

## 2. 相关文件总览

| 文件/目录 | 作用 |
|---|---|
| `csrc/attention/attention_generic.cuh` | 通用 attention CUDA 模板 |
| `csrc/attention/attention_dtypes.h` | attention dtype 分发/定义 |
| `csrc/attention/dtype_*.cuh` | float16/bfloat16/float32/fp8 dtype 特化 |
| `csrc/libtorch_stable/attention/paged_attention_v1.cu` | PagedAttention v1 kernel |
| `csrc/libtorch_stable/attention/paged_attention_v2.cu` | PagedAttention v2 kernel |
| `csrc/libtorch_stable/attention/attention_kernels.cuh` | attention kernel 辅助 |
| `csrc/libtorch_stable/attention/attention_utils.cuh` | attention 工具函数 |
| `csrc/libtorch_stable/attention/merge_attn_states.cu` | split attention state merge |
| `csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu` | SM100 CUTLASS MLA decode |
| `csrc/libtorch_stable/cache_kernels.cu` | KV cache 基础 kernel |
| `csrc/libtorch_stable/cache_kernels_fused.cu` | fused cache kernel |
| `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` | NVFP4 KV cache kernel |
| `csrc/libtorch_stable/pos_encoding_kernels.cu` | RoPE positional encoding |
| `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` | QK norm + RoPE fused |
| `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | DeepSeek V4 Q norm/RoPE/KV insert fused |
| `csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu` | MiniMax M3 QK norm/RoPE/KV insert fused |

## 3. PagedAttention v1/v2

### 3.1 Python wrapper

`vllm/_custom_ops.py` 提供两个 wrapper：

- `paged_attention_v1()`：`code/vllm/vllm/_custom_ops.py:113-155`
- `paged_attention_v2()`：`code/vllm/vllm/_custom_ops.py:158-205`

它们都调用：

```python
torch.ops._C.paged_attention_v1(...)
torch.ops._C.paged_attention_v2(...)
```

### 3.2 C++ schema

`_C_stable_libtorch` 注册 schema：

- v1：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:594-606`
- v2：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:608-619`

对应 impl 注册：

- v1：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:737`
- v2：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:738`

声明位于：`code/vllm/csrc/libtorch_stable/ops.h:424-448`。

### 3.3 参数含义

PagedAttention 接收的核心参数：

| 参数 | 含义 |
|---|---|
| `out` | 输出 attention 结果 |
| `query` | 当前 decode token/query |
| `key_cache` / `value_cache` | paged KV cache |
| `num_kv_heads` | KV heads 数 |
| `scale` | attention scale |
| `block_tables` | 每个 sequence 对应的 cache block table |
| `seq_lens` | 每个 sequence 当前长度 |
| `block_size` | cache block 中 token 数 |
| `max_seq_len` | 最大序列长度 |
| `alibi_slopes` | 可选 ALiBi |
| `kv_cache_dtype` | cache dtype，如 auto/fp8 等 |
| `k_scale` / `v_scale` | KV cache 量化 scale |
| `tp_rank` | tensor parallel rank |
| `blocksparse_*` | block sparse attention 相关参数 |

这些参数体现了 PagedAttention 的核心：query 是连续的，但历史 K/V 不一定连续，而是通过 `block_tables` 映射到 cache block。

## 4. Attention state merge

`merge_attn_states` 用于合并 partial attention 结果，源码注释说明它实现了论文 `https://www.arxiv.org/pdf/2501.01005` 的 section 2.2，可用于 split-KV 场景合并部分 attention 结果。

C++ schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:352-364`。

Python wrapper：`code/vllm/vllm/_custom_ops.py:265-285`。

参数：

- `prefix_output` / `prefix_lse`
- `suffix_output` / `suffix_lse`
- `output` / `output_lse`
- `prefill_tokens_with_context`
- `output_scale`

典型用途：当 KV 被切分处理，或者 prefix/suffix attention 分别计算，需要稳定合并 softmax/lse 状态。

## 5. MLA / CUTLASS MLA

### 5.1 CUTLASS MLA 构建

CMake 中 CUTLASS MLA 只在满足条件时构建：

- CUDA >= 12.8
- 目标架构包含 SM100/SM10x/SM11x 系列

源码位置：`code/vllm/CMakeLists.txt:1035-1059`。

源码文件：

```text
csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu
```

### 5.2 注册接口

C++ schema：

```text
sm100_cutlass_mla_decode
sm100_cutlass_mla_get_workspace_size
```

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:288-300`。

这类 kernel 是特定硬件和模型结构优化，不是所有平台都会编译。

## 6. KV Cache ops

KV cache ops 注册在 `_C_cache_ops` fragment 中。

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:780-866`。

### 6.1 block swap

```text
swap_blocks
swap_blocks_batch
```

声明位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:782-791`。

用途：

- cache block 在不同 memory 区域间交换。
- batch swap 可以一次 driver call 提交多个 block copy，减少 overhead。

头文件声明：`code/vllm/csrc/cache.h:9-16`。

### 6.2 reshape_and_cache

```text
reshape_and_cache
reshape_and_cache_flash
```

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:793-808`。

头文件声明：`code/vllm/csrc/cache.h:18-29`。

用途：

- 把 attention layer 产生的 key/value tensor reshape 成 cache layout。
- 根据 `slot_mapping` 写入指定 cache slot。
- 支持 `kv_cache_dtype` 和 K/V scale。

### 6.3 MLA cache 写入

```text
concat_and_cache_mla
concat_and_cache_mla_rope_fused
```

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:810-830`。

头文件声明：`code/vllm/csrc/cache.h:31-41`。

用途：

- MLA/DSA 结构下，把 `kv_c` 和 `k_pe` 拼接后写入 cache。
- fused 版本会对 q/k 做 RoPE，然后写入 KV cache。

### 6.4 FP8 / dequant / gather

```text
convert_fp8
gather_and_maybe_dequant_cache
cp_gather_cache
cp_gather_and_upconvert_fp8_kv_cache
```

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:832-854`。

头文件声明：`code/vllm/csrc/cache.h:43-72`。

用途：

- cache dtype 转换。
- 从 paged cache 按 block table gather 到连续 workspace。
- 如果 cache 是 FP8，则可能需要 dequant/upconvert 到 BF16 workspace。

### 6.5 indexer K quant cache

```text
indexer_k_quant_and_cache
cp_gather_indexer_k_quant_cache
```

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:856-865`。

头文件声明：`code/vllm/csrc/cache.h:74-94`。

用于 indexer K 的量化写入与读取。

## 7. RoPE 与 fused QK norm/RoPE

### 7.1 rotary_embedding

Python wrapper：`code/vllm/vllm/_custom_ops.py:289-313`。

C++ schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:417-423`。

impl 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:683-685`。

声明：`code/vllm/csrc/libtorch_stable/ops.h:234-239`。

参数支持：

- GPT-NeoX 或 GPT-J style
- query 和可选 key
- head_size
- cos_sin_cache
- rope_dim_offset
- inverse

### 7.2 fused_qk_norm_rope

用途：把 Q/K RMSNorm 和 RoPE 融合在一个 op 中。

Python wrapper：`code/vllm/vllm/_custom_ops.py:336-363`。

C++ schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:425-431`。

声明：`code/vllm/csrc/libtorch_stable/ops.h:241-248`。

## 8. 模型特化 fused kernel

### 8.1 DeepSeek V4 fused kernel

注册接口：

- `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
- `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert`
- `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:433-451`。

声明位置：`code/vllm/csrc/libtorch_stable/ops.h:250-272`。

用途：

- Q norm
- RoPE
- KV RoPE
- KV cache insert
- 可选 FP8 quant

这些动作合并后可以显著减少 kernel launch 和中间 tensor 读写。

### 8.2 MiniMax M3 fused kernel

接口：

```text
fused_minimax_m3_qknorm_rope_kv_insert
```

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:465-476`。

声明位置：`code/vllm/csrc/libtorch_stable/ops.h:288-305`。

源码注释说明它融合：

- QK norm
- partial NeoX RoPE
- optional KV/index cache insert

## 9. ROCm PagedAttention

ROCm 专用 attention op 注册在 `_rocm_C`：

```text
paged_attention
```

C++ schema：`code/vllm/csrc/rocm/torch_bindings.cpp:64-82`。

Python wrapper：`code/vllm/vllm/_custom_ops.py:208-251`。

ROCm wrapper 参数额外包含：

- `query_start_loc`
- `fp8_out_scale`
- `mfma_type`

其中 `mfma_type` 默认由环境变量 `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` 决定：`code/vllm/vllm/_custom_ops.py:208-229`。

## 10. Attention/KV cache 调用链

典型 decode attention 调用链：

```text
Python attention backend
  -> vllm._custom_ops.paged_attention_v1/v2
  -> torch.ops._C.paged_attention_v1/v2
  -> csrc/libtorch_stable/torch_bindings.cpp schema/impl
  -> csrc/libtorch_stable/attention/paged_attention_v1.cu/v2.cu
  -> csrc/attention/attention_generic.cuh + dtype_*.cuh
```

典型 KV cache 写入链路：

```text
model_runner / attention layer
  -> vllm._custom_ops.reshape_and_cache / concat_and_cache_mla
  -> torch.ops._C_cache_ops 或 torch.ops._C 注册 ops
  -> csrc/libtorch_stable/cache_kernels.cu
  -> paged KV cache memory
```

## 11. 关键结论

Attention 和 KV Cache 是 vLLM native 层最核心的性能路径。PagedAttention 解决“动态 batch + 不连续历史 KV”的访问问题，cache kernels 解决“高效写入/搬运/读取”的数据布局问题，fused RoPE/QK norm/MLA/模型特化 kernel 则进一步减少 kernel launch 和内存带宽消耗。