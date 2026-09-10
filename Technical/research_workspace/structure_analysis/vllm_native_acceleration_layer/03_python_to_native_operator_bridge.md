# 03 Python 到 Native 算子的调用桥

## 1. 为什么需要调用桥

C++/CUDA/HIP/CPU kernel 编译成 Python extension 后，并不是直接被模型代码 import 成普通 Python 函数，而是通过 PyTorch custom op 机制注册到：

```python
torch.ops._C.xxx
torch.ops._moe_C.xxx
torch.ops._rocm_C.xxx
```

Python 侧再用 `vllm/_custom_ops.py` 提供更友好的 wrapper，负责：

- 调用 platform-specific import
- 分配输出 tensor
- 做 fallback 选择
- 注册 fake/meta function，支持 torch.compile / tracing
- 屏蔽不同 backend 的 op namespace 差异

## 2. Python 入口：`vllm/_custom_ops.py`

文件开头：

```python
current_platform.import_kernels()
```

源码位置：`code/vllm/vllm/_custom_ops.py:20`。

这一步根据当前 platform 导入 native extension。导入后，C++ 注册的 custom ops 会出现在 `torch.ops` 下。

该文件大量函数都是 wrapper，例如：

```python
def paged_attention_v1(...):
    torch.ops._C.paged_attention_v1(...)
```

源码位置：`code/vllm/vllm/_custom_ops.py:113-155`。

## 3. C++ 注册机制

### 3.1 legacy `_C`

`csrc/torch_bindings.cpp` 使用：

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) { ... }
```

源码位置：`code/vllm/csrc/torch_bindings.cpp:21-31`。

最后调用：

```cpp
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

源码位置：`code/vllm/csrc/torch_bindings.cpp:107`。

当前这个文件只保留少量 ops，例如：

- `persistent_masked_m_silu_mul_quant`
- `weak_ref_tensor`
- `silu_and_mul_quant`
- ROCm quick reduce / cuda utils 兼容

源码位置：`code/vllm/csrc/torch_bindings.cpp:25-48`、`code/vllm/csrc/torch_bindings.cpp:74-105`。

### 3.2 stable ABI `_C_stable_libtorch`

`csrc/libtorch_stable/torch_bindings.cpp` 使用 stable ABI：

```cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) { ... }
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) { ... }
```

源码位置：

- schema 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:10-620`
- CUDA impl 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:622-739`

注意：虽然 extension target 名为 `_C_stable_libtorch`，但注册 namespace 是 `_C`，所以 Python 仍调用：

```python
torch.ops._C.paged_attention_v1(...)
```

源码注释：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:7-10`。

### 3.3 cache ops fragment

同一个 stable binding 文件还注册 `_C_cache_ops` fragment：

```cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) { ... }
```

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:780-866`。

这里定义 KV cache 相关 ops：

- `swap_blocks`
- `swap_blocks_batch`
- `reshape_and_cache`
- `reshape_and_cache_flash`
- `concat_and_cache_mla`
- `concat_and_cache_mla_rope_fused`
- `convert_fp8`
- `gather_and_maybe_dequant_cache`
- `cp_gather_cache`
- `cp_gather_and_upconvert_fp8_kv_cache`
- `indexer_k_quant_and_cache`
- `concat_mla_q`
- `cp_gather_indexer_k_quant_cache`

### 3.4 custom all-reduce fragment

同一文件还注册 `_C_custom_ar`：

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:868-900`。

包含：

- `init_custom_ar`
- `all_reduce`
- `dispose`
- `meta_size`
- `register_buffer`
- `get_graph_buffer_ipc_meta`
- `register_graph_buffers`
- `allocate_shared_buffer_and_handle`
- `open_mem_handle`
- `free_shared_buffer`

### 3.5 MoE namespace `_moe_C`

`csrc/libtorch_stable/moe/torch_bindings.cpp` 注册 namespace：

```cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) { ... }
```

源码位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:6-7`。

定义 MoE ops：

- `topk_softmax`
- `topk_sigmoid`
- `topk_softplus_sqrt`
- `moe_sum`
- `moe_align_block_size`
- `batched_moe_align_block_size`
- `moe_lora_align_block_size`
- `moe_wna16_gemm`
- `moe_wna16_marlin_gemm`
- `moe_permute`
- `moe_unpermute`
- `grouped_topk`
- `dsv3_router_gemm`

源码位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:8-128`。

### 3.6 ROCm namespace `_rocm_C`

ROCm binding：

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) { ... }
```

源码位置：`code/vllm/csrc/rocm/torch_bindings.cpp:14-15`。

注册：

- `LLMM1`
- `wvSplitK`
- `wvSplitKrc`
- `wvSplitKQ`
- `gptq_gemm_rdna3`
- `gptq_gemm_rdna3_wmma`
- `moe_gptq_gemm_rdna3`
- `paged_attention`

源码位置：`code/vllm/csrc/rocm/torch_bindings.cpp:17-83`。

## 4. wrapper 示例：PagedAttention

Python wrapper：

```python
def paged_attention_v1(...):
    torch.ops._C.paged_attention_v1(...)
```

源码位置：`code/vllm/vllm/_custom_ops.py:113-155`。

C++ schema：

```cpp
ops.def("paged_attention_v1(...)")
```

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:594-606`。

C++ impl：

```cpp
ops.impl("paged_attention_v1", TORCH_BOX(&paged_attention_v1));
```

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:737`。

声明位于 `ops.h`：`code/vllm/csrc/libtorch_stable/ops.h:424-448`。

实际实现文件：

```text
csrc/libtorch_stable/attention/paged_attention_v1.cu
csrc/libtorch_stable/attention/paged_attention_v2.cu
```

## 5. wrapper 示例：CUTLASS scaled_mm

Python wrapper：

```python
def cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    ...
    if current_platform.is_rocm() or not cutlass_compatible_b:
        triton_scaled_mm(...)
    else:
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
```

源码位置：`code/vllm/vllm/_custom_ops.py:813-861`。

这个 wrapper 体现了 Python 桥接层的职责：

- 处理输入 shape。
- 判断平台和 CUTLASS 兼容性。
- 不兼容时走 Triton fallback。
- 兼容时调用 native CUTLASS op。

C++ schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:100-106`。

C++ impl 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:633-635`。

声明位于 `ops.h`：`code/vllm/csrc/libtorch_stable/ops.h:38-43`。

## 6. fake/meta function 的作用

`vllm/_custom_ops.py` 中会用 `torch.library.register_fake` 给部分 custom op 注册 fake implementation。

例子：`scaled_fp4_quant`。

```python
@register_fake("_C::scaled_fp4_quant")
def _scaled_fp4_quant_fake(...):
    return create_fp4_output_tensors(...)
```

源码位置：`code/vllm/vllm/_custom_ops.py:89-110`。

作用：

- 让 torch.compile / tracing 能推断输出 shape/dtype。
- 不需要真正执行 CUDA kernel。
- 对返回 Tensor 的 custom op 尤其重要。

另一个例子：AWQ fake：`code/vllm/vllm/_custom_ops.py:565-611`。

## 7. 头文件声明层

`csrc/libtorch_stable/ops.h` 是 stable ABI ops 的声明汇总。

它包含：

- quantization：per-token FP8/INT8、FP4、AWQ、GPTQ、CUTLASS
- attention：merge attn states、paged attention v1/v2
- layernorm / fused quant norm
- RoPE / fused QK norm RoPE
- sampler / top-k
- Mamba selective scan
- custom all-reduce
- cache ops

源码位置：`code/vllm/csrc/libtorch_stable/ops.h:9-551`。

## 8. Python wrapper 和 C++ op 的命名关系

通常有三层命名：

```text
Python wrapper 函数名
  -> torch.ops namespace 函数名
  -> C++ 函数名 / CUDA kernel launcher
```

例子：

| Python wrapper | torch.ops | C++ 函数 |
|---|---|---|
| `paged_attention_v1` | `_C.paged_attention_v1` | `paged_attention_v1` |
| `rms_norm` | `_C.rms_norm` | `rms_norm` |
| `rotary_embedding` | `_C.rotary_embedding` | `rotary_embedding` |
| `cutlass_scaled_mm` | `_C.cutlass_scaled_mm` | `cutlass_scaled_mm` |
| `paged_attention_rocm` | `_rocm_C.paged_attention` | `paged_attention` |

## 9. 平台分流

Python wrapper 中常见平台分流：

- CUDA：`torch.ops._C`。
- ROCm：`torch.ops._rocm_C` 或 `_C` stable/HIP 实现。
- CPU：`torch.ops._C`，但 `_C` 是 CPU extension。
- Triton fallback：某些 CUDA/ROCm 不兼容场景转到 Triton。
- Pure torch fallback：例如 repetition penalty 非 CUDA contiguous 时走 torch 实现。

示例：`apply_repetition_penalties()` 根据 `logits.is_cuda and logits.is_contiguous()` 决定走 CUDA op 还是 torch fallback：`code/vllm/vllm/_custom_ops.py:393-414`。

## 10. 总结

Python 到 native 的调用桥可以概括为：

```text
model code
  -> vllm._custom_ops.some_func(...)
  -> current_platform 已加载 native extension
  -> torch.ops._C / _moe_C / _rocm_C
  -> C++ schema dispatch
  -> CUDA/HIP/CPU impl
```

`vllm/_custom_ops.py` 是理解 native 层使用方式的第一入口；`csrc/**/torch_bindings.cpp` 是理解 native ops 暴露给 Python 的第一入口。