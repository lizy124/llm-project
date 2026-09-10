# 06 MoE 底层算子

## 1. MoE native 层负责什么

MoE（Mixture of Experts）在推理时比 dense 模型多出一批底层操作：

1. router/gating 输出 top-k experts。
2. 根据 expert id 对 token 分组、排序、padding 对齐。
3. 把 token hidden states permute 到 expert-local 连续布局。
4. 对每个 expert 做 GEMM，常见是 grouped GEMM 或 quantized GEMM。
5. 把 expert 输出 unpermute 回原 token 顺序。
6. 按 top-k weights 做加权合并。

这些步骤如果在 Python/PyTorch eager 中实现会有大量小 op 和内存搬运，因此 vLLM 为 MoE 单独构建 `_moe_C_stable_libtorch` native extension。

## 2. 主要源码目录

| 路径 | 作用 |
|---|---|
| `csrc/libtorch_stable/moe/torch_bindings.cpp` | MoE ops 注册入口 |
| `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` | token/expert 对齐、moe_sum 等 |
| `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` | top-k softmax/sigmoid routing |
| `csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu` | softplus/sqrt routing 变体 |
| `csrc/libtorch_stable/moe/grouped_topk_kernels.cu` | grouped top-k routing |
| `csrc/libtorch_stable/moe/moe_wna16.cu` | WNA16 MoE GEMM |
| `csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu` | MoE permute/unpermute op wrapper |
| `csrc/libtorch_stable/moe/permute_unpermute_kernels/` | permute/unpermute kernel 实现 |
| `csrc/libtorch_stable/moe/marlin_moe_wna16/` | Marlin MoE WNA16 生成 kernel |
| `csrc/libtorch_stable/moe/mxfp8_moe/` | MXFP8 grouped MoE kernel |
| `csrc/cpu/cpu_fused_moe.cpp` | CPU fused MoE |
| `csrc/cpu/sgl-kernels/moe*.cpp` | CPU SGL MoE kernel |

## 3. 构建 target：`_moe_C_stable_libtorch`

CMake 定义 MoE extension：

```text
_moe_C_stable_libtorch
```

基础源码：

- `csrc/libtorch_stable/moe/torch_bindings.cpp`
- `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`
- `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`
- `csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu`

源码位置：`code/vllm/CMakeLists.txt:1131-1138`。

CUDA 下追加：

- `moe_wna16.cu`
- `grouped_topk_kernels.cu`
- `moe_permute_unpermute_kernel.cu`
- `moe_permute_unpermute_op.cu`

源码位置：`code/vllm/CMakeLists.txt:1140-1152`。

最终 target 定义：`code/vllm/CMakeLists.txt:1301-1312`。

## 4. MoE op 注册入口

`csrc/libtorch_stable/moe/torch_bindings.cpp` 使用 stable ABI 注册：

```cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) { ... }
```

源码位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:6-7`。

因此 Python 中会通过类似：

```python
torch.ops._moe_C.topk_softmax(...)
torch.ops._moe_C.moe_align_block_size(...)
```

调用。

## 5. Routing top-k ops

### 5.1 topk_softmax

接口：

```text
topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool renormalize, Tensor? bias) -> ()
```

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:8-11`。

impl 注册：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:131-133`。

作用：

- 输入 router logits / gating output。
- 选择 top-k experts。
- 写出 topk weights、topk indices 和 token_expert_indices。
- 可选 renormalize 和 bias。

### 5.2 topk_sigmoid

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:13-17`。

适合使用 sigmoid gating 的 MoE 变体。

### 5.3 topk_softplus_sqrt

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:19-23`。

额外参数：

- `routed_scaling_factor`
- `input_ids`
- `tid2eid`

用于特定模型的 routing 计算变体。

### 5.4 grouped_topk

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:118-123`。

impl 注册：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:141-143`。

它支持 grouped routing：先选 top-k groups，再选 top-k experts。

## 6. token/expert 对齐与 padding

MoE GEMM 通常按 expert 分组处理 token。为了高效 GEMM，需要把每个 expert 的 token 数对齐到 block size。

### 6.1 moe_align_block_size

接口：

```text
moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! experts_ids, Tensor! num_tokens_post_pad, Tensor? maybe_expert_map) -> ()
```

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:29-36`。

作用：

- 输入每个 token 的 top-k expert ids。
- 输出按 expert 排序后的 token ids。
- 输出 expert ids。
- 输出 padding 后 token 数。
- 可选 expert map。

### 6.2 batched_moe_align_block_size

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:38-45`。

用于 batched expert 格式。

### 6.3 moe_lora_align_block_size

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:47-61`。

用于 MoE + LoRA 场景，需要同时考虑：

- `token_lora_mapping`
- `max_loras`
- `adapter_enabled`
- `lora_ids`

## 7. permute / unpermute

MoE 中 token 需要按 expert 聚集，然后再恢复原顺序。

### 7.1 moe_permute

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:86-92`。

输入：

- `input`
- `topk_ids`
- `token_expert_indices`
- `expert_map`

输出：

- `permuted_input`
- `expert_first_token_offset`
- `inv_permuted_idx`
- `permuted_idx`

### 7.2 moe_permute_with_scratch

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:94-101`。

增加 scratch workspace，避免内部反复分配。

### 7.3 moe_unpermute

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:103-106`。

把 expert 输出恢复到原 token 顺序，并可结合 `topk_weights`。

### 7.4 支持检查与 workspace size

接口：

- `moe_permute_unpermute_supported()`
- `moe_permute_sort_workspace_size(...)`

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:108-111`。

impl 注册在 CompositeExplicitAutograd，因为 primitive-only ops 没有 tensor dispatch：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:147-154`。

## 8. MoE GEMM

### 8.1 moe_wna16_gemm

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:63-69`。

用于 WNA16 quantized MoE GEMM。

### 8.2 moe_wna16_marlin_gemm

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:71-84`。

这是 Marlin MoE GEMM，参数包含：

- quantized weights
- scales / zero points
- sorted_token_ids
- expert_ids
- topk_weights
- moe_block_size
- top_k
- b_type_id
- size_m/n/k
- reduce options
- schedule/thread/block 参数

### 8.3 CUTLASS MoE grouped MM

CUTLASS grouped MoE MM 不在 `_moe_C` 注册，而在 `_C` stable binding 中：

- `cutlass_moe_mm`
- `get_cutlass_moe_mm_data`
- `get_cutlass_moe_mm_problem_sizes_from_expert_offsets`
- `get_cutlass_batched_moe_mm_data`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:124-166`。

CMake 构建逻辑：`code/vllm/CMakeLists.txt:861-934`。

## 9. DeepSeek V3 router GEMM

MoE router 常常是小 batch、小 M 的 GEMM。vLLM 为 DeepSeek V3 router 做了特化：

接口：

```text
dsv3_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()
```

schema 位置：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:125-127`。

CMake 构建条件：SM90+ 且 CUDA >= 12.0。

源码位置：`code/vllm/CMakeLists.txt:1281-1298`。

## 10. MXFP8 / FP4 MoE

MXFP8 grouped MoE kernel 源码在：

```text
csrc/libtorch_stable/moe/mxfp8_moe/
```

包含：

- `cutlass_mxfp8_grouped_mm.cu`
- `mxfp8_experts_quant.cu`
- launcher / traits / functor headers

CMake 构建逻辑：`code/vllm/CMakeLists.txt:366-391`。

FP4/NVFP4 MoE 则主要在：

```text
csrc/libtorch_stable/quantization/fp4/
```

相关接口：

- `cutlass_fp4_group_mm`
- `cutlass_mxfp4_group_mm`
- `scaled_fp4_experts_quant`
- `mxfp4_experts_quant`
- `silu_and_mul_*_experts_quant`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:179-235`。

## 11. CPU MoE

CPU 后端也有 MoE：

- `csrc/cpu/cpu_fused_moe.cpp`
- `csrc/cpu/sgl-kernels/moe.cpp`
- `csrc/cpu/sgl-kernels/moe_int8.cpp`
- `csrc/cpu/sgl-kernels/moe_int4.cpp`
- `csrc/cpu/sgl-kernels/moe_fp8.cpp`

CPU binding 中注册：

- `fused_experts_cpu`
- `prepack_moe_weight`
- `cpu_fused_moe`

源码位置：

- `code/vllm/csrc/cpu/torch_bindings.cpp:69-80`
- `code/vllm/csrc/cpu/torch_bindings.cpp:189-198`
- `code/vllm/csrc/cpu/torch_bindings.cpp:408-438`
- `code/vllm/csrc/cpu/torch_bindings.cpp:537-548`

## 12. MoE 典型 native 调用链

```text
Python MoE layer
  -> routing logits
  -> torch.ops._moe_C.topk_softmax / grouped_topk
  -> torch.ops._moe_C.moe_align_block_size
  -> torch.ops._moe_C.moe_permute
  -> torch.ops._C.cutlass_moe_mm 或 torch.ops._moe_C.moe_wna16_marlin_gemm
  -> torch.ops._moe_C.moe_unpermute
  -> torch.ops._moe_C.moe_sum
```

## 13. 关键结论

MoE native 层的重点不是单个 GEMM，而是“routing -> token 重排 -> grouped GEMM -> unpermute -> merge”的整条数据路径。vLLM 将 MoE ops 单独拆成 `_moe_C_stable_libtorch`，说明 MoE kernel 数量、编译条件、硬件特化和普通 dense/attention ops 相比已经足够复杂，需要独立维护。