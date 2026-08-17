# 08 CUDA/csrc Kernel 调用链与调试地图

本篇梳理从 Python Attention 层到 `torch.ops.vllm`、C++ binding、CUDA/C++/CPU kernel 的调用链，并给出调试地图。

## 1. 总体调用链

```text
GPUModelRunner.execute_model
  ↓
set_forward_context
  ↓
模型 forward
  ↓
Attention.forward
  ↓
以下之一：
  - unified_kv_cache_update
  - unified_attention_with_output
  - backend direct call
  ↓
torch.ops.vllm.* 或 backend op
  ↓
C++ torch binding
  ↓
CUDA / ROCm / CPU kernel
  ↓
KV cache update / paged attention / quantized cache / MoE 等
```

## 2. Python 层入口

关键文件：

```text
vllm/model_executor/layers/attention/attention.py
```

主要入口：

- `Attention.forward()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:438`
- `unified_kv_cache_update()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`
- `unified_attention_with_output()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:736`

`Attention.forward()` 中会根据 backend 特性决定：

1. KV cache update 是否和 attention forward 融合；
2. 是否使用 opaque custom op；
3. 是否直接调用 Python backend；
4. 是否计算 KV scales；
5. 是否走 fused rope + KV cache update。

## 3. custom op 注册

`attention.py` 中通过 `direct_register_custom_op` 注册了：

```text
maybe_calc_kv_scales
unified_kv_cache_update
unified_attention_with_output
```

这些 op 通过 `torch.ops.vllm.*` 暴露给 PyTorch graph/compile/runtime。

典型调用：

```text
torch.ops.vllm.maybe_calc_kv_scales(...)
torch.ops.vllm.unified_kv_cache_update(...)
torch.ops.vllm.unified_attention_with_output(...)
```

## 4. get_attention_context 是 Python 到 runtime 的桥

`unified_kv_cache_update()` 和 `unified_attention_with_output()` 都会通过：

```text
get_attention_context(layer_name)
```

取出：

- Attention layer 对象；
- 当前层 KV cache；
- backend-specific attention metadata；
- slot mapping。

这一步把 Python 层的 layer name 映射回真实 runtime 对象。

## 5. backend impl 到底层 kernel

`AttentionImpl.forward()` 是 backend 层接口。

不同 backend 内部可能调用：

- Triton kernel；
- FlashAttention；
- FlashInfer；
- ROCm kernel；
- CPU kernel；
- vLLM 自己的 paged attention op；
- torch custom op；
- fused cache update op。

因此，`AttentionImpl.forward()` 是 Python 抽象和底层高性能实现的分界线。

## 6. csrc 主要目录

```text
csrc/attention/
csrc/libtorch_stable/attention/
csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/cache_kernels_fused.cu
csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
csrc/libtorch_stable/torch_bindings.cpp
csrc/rocm/attention.cu
csrc/rocm/torch_bindings.cpp
csrc/cpu/
csrc/cutlass_extensions/
csrc/custom_all_reduce*
```

## 7. Attention kernel 相关文件

重点看：

```text
csrc/libtorch_stable/attention/paged_attention_v1.cu
csrc/libtorch_stable/attention/paged_attention_v2.cu
csrc/libtorch_stable/attention/attention_kernels.cuh
csrc/attention/attention_generic.cuh
csrc/attention/attention_dtypes.h
```

这些文件承担：

- paged attention 计算；
- block table 间接寻址；
- Q/K/V dtype 处理；
- head size 模板化；
- block size 模板化；
- softmax/reduction；
- value 聚合。

## 8. Cache kernel 相关文件

重点看：

```text
csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/cache_kernels_fused.cu
csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
csrc/cache.h
```

这些文件承担：

- reshape and cache；
- key/value 写入 KV cache；
- fused RoPE + KV cache update；
- FP8/NVFP4 cache 写入；
- scale 处理；
- cache copy/swap/zero 等操作。

## 9. torch binding

Python `torch.ops.vllm.*` 最终需要 C++ binding。

重点文件：

```text
csrc/libtorch_stable/torch_bindings.cpp
csrc/rocm/torch_bindings.cpp
```

它们负责把 C++/CUDA 函数注册成 PyTorch 可调用 op。

抽象链路：

```text
Python torch.ops.vllm.xxx
  ↓
TORCH_LIBRARY / binding registration
  ↓
C++ wrapper function
  ↓
CUDA kernel launch
```

## 10. Paged Attention 的数据输入

底层 paged attention kernel 通常需要：

- query tensor；
- KV cache tensor；
- block table；
- sequence lengths；
- scale；
- head mapping；
- max context len；
- alibi/sink/sliding window 等可选参数；
- KV cache dtype/scale。

其中 block table 和 seq lens 来自 attention metadata，KV cache tensor 来自 worker 初始化，query 来自模型当前层。

## 11. KV cache update 的数据输入

cache update kernel 通常需要：

- key tensor；
- value tensor；
- KV cache tensor；
- slot mapping；
- scale；
- optional RoPE position；
- quant mode；
- dtype/layout 信息。

slot mapping 是写入位置的关键。

## 12. 常见 backend 调用差异

### 12.1 forward 包含 KV cache update

一些 backend 在 forward 中同时做：

```text
write K/V to cache + attention compute
```

### 12.2 forward 不包含 KV cache update

另一些 backend 分两步：

```text
unified_kv_cache_update
  ↓
unified_attention_with_output
```

GPUModelRunner/Attention.forward 会根据 backend 的 `forward_includes_kv_cache_update` 决定路径。

## 13. CUDA graph 对 kernel 调用的影响

CUDA graph 要求 shape 较稳定，因此 GPUModelRunner 会：

- 判断当前 batch 是否适合 graph；
- padding token/request 到 capture size；
- 构造 padded slot mapping；
- 构造 capture-compatible metadata；
- 使用静态 buffer。

如果 kernel 报 shape 或越界问题，要同时看：

```text
_determine_batch_execution_and_padding
_get_slot_mappings
_build_attention_metadata
AttentionMetadataBuilder.build_for_cudagraph_capture
```

## 14. 调试地图

### 14.1 Attention backend 选错

优先看：

```text
Attention.get_attn_backend
v1/attention/selector.py
v1/attention/backends/registry.py
AttentionBackend.supports_*
AttentionBackend.validate_configuration
```

检查：

- head size 是否支持；
- dtype 是否支持；
- KV cache dtype 是否支持；
- block size 是否支持；
- attention type 是否支持；
- 当前平台是否有对应依赖；
- 是否被环境变量强制指定 backend。

### 14.2 KV cache shape/layout 错误

优先看：

```text
Attention.get_kv_cache_spec
KVCacheSpec.page_size_bytes
AttentionBackend.get_kv_cache_shape
GPUModelRunner.initialize_kv_cache
GPUModelRunner._reshape_kv_cache_tensors
```

检查：

- spec 中 head size/num kv heads 是否正确；
- MLA/sliding window/quant mode 是否正确；
- backend required layout 是否匹配；
- worker 分配 tensor shape 是否和 kernel 期望一致。

### 14.3 slot mapping 越界

优先看：

```text
SchedulerOutput block ids
GPUModelRunner._update_states
GPUModelRunner._get_slot_mappings
worker block_table
gpu block_table
unified_kv_cache_update
```

检查：

- block id 是否超过 num_blocks；
- block 内 offset 是否超过 block_size；
- padding 后 slot mapping 是否同步 padding；
- separate KV update backend 是否要求 padded slot mapping；
- spec decode lookahead 是否正确预留。

### 14.4 paged attention 输出错误

优先看：

```text
GPUModelRunner._build_attention_metadata
CommonAttentionMetadata
backend MetadataBuilder
AttentionImpl.forward
csrc paged_attention_v1/v2
```

检查：

- seq_lens 是否正确；
- query_start_loc 是否正确；
- block table 是否正确；
- max_query_len/max_seq_len 是否正确；
- causal/sliding window/non-causal 标记是否正确；
- prefix/multimodal ranges 是否正确。

### 14.5 KV cache quant 输出异常

优先看：

```text
KVQuantMode
Attention._init_kv_cache_quant
Attention.process_weights_after_loading
maybe_calc_kv_scales
cache_kernels_fused
nvfp4_kv_cache_kernels
```

检查：

- scale 是否从 checkpoint 正确加载；
- 默认 scale 是否为 1.0；
- per-token-head scale shape 是否正确；
- cache dtype 和 kernel 是否匹配；
- FP8/NVFP4 page size 是否正确。

### 14.6 CUDA illegal memory access

优先看：

```text
slot mapping
block table
KV cache tensor shape
attention metadata
paged attention kernel launch args
CUDA graph padding
```

这种问题通常不是单个 kernel 文件能看出来，要从 SchedulerOutput 到 GPUModelRunner metadata 一路核对。

## 15. 推荐阅读底层 kernel 的顺序

```text
1. attention.py 中 unified_* 函数
2. v1/attention/backend.py 中 AttentionImpl.forward 抽象
3. 一个具体 backend 实现，例如 flash_attn/triton/paged_attn backend
4. csrc/libtorch_stable/torch_bindings.cpp
5. csrc/libtorch_stable/cache_kernels.cu
6. csrc/libtorch_stable/attention/paged_attention_v1.cu
7. csrc/libtorch_stable/attention/paged_attention_v2.cu
8. csrc/libtorch_stable/attention/attention_kernels.cuh
```

## 16. 一句话总结

Python 层 Attention 通过 ForwardContext 拿到 metadata、slot mapping 和 KV cache，再经 AttentionImpl/custom op 调到底层 csrc；底层 kernel 的正确性高度依赖上游构造的 block table、slot mapping、KV cache shape 和 attention metadata 是否一致。
