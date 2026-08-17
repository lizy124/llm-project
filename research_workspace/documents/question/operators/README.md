# vLLM V1 Operators / Kernels 问题目录

源码位置：

- `vllm/vllm/_custom_ops.py`
- `vllm/vllm/_C/`
- `vllm/vllm/attention/`
- `vllm/vllm/model_executor/layers/`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/platforms/`
- `vllm/csrc/`

这个目录按问题拆解 vLLM V1 里的算子与 kernel 体系，重点回答：哪些计算会落到自定义算子、算子如何被 Python 层调用、attention / KV cache / quantization / MoE / norm / activation / sampling 等算子如何协同，以及这些算子如何和 ModelRunner、CUDA Graph、编译、并行、fallback 路径衔接。

---

## 1. 总览文档

- [vLLM V1 Operators / Kernels 逻辑梳理](operators_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
ModelRunner / Model Layer
  → Python wrapper / torch op / custom op
  → backend selection
  → CUDA / Triton / FlashAttention / FlashInfer / CUTLASS / torch fallback
  → output tensor
  → logits / sampler / cache state / next layer
```

---

## 2. 主线专题阅读顺序

### 01. 算子体系定位

- [vLLM 里的算子专题应该回答什么？](01_operator_system_role.md)

回答：

```text
什么是 vLLM 语境里的 operator / kernel？
它和 layer、backend、ModelRunner 的边界是什么？
哪些路径必须关注算子？
```

### 02. Python 到 native op 的调用入口

- [Python 层如何调用自定义算子？](02_python_to_native_ops.md)

回答：

```text
_custom_ops.py 扮演什么角色？
torch.ops / extension / fallback 如何组织？
Python wrapper 如何隐藏底层实现差异？
```

### 03. Attention kernels

- [Attention 算子如何支撑 prefill / decode？](03_attention_kernels.md)

回答：

```text
PagedAttention、FlashAttention、FlashInfer、FlashMLA 分别解决什么问题？
metadata、block table、slot mapping 如何进入 kernel？
prefill / decode 的 kernel 形态有什么不同？
```

### 04. KV cache kernels

- [KV Cache 相关算子负责什么？](04_kv_cache_kernels.md)

回答：

```text
KV cache 写入、reshape、copy、swap、cache block 操作在哪里发生？
slot mapping 如何映射到物理 cache？
KV cache quantization 如何接入？
```

### 05. Quantization kernels

- [量化算子如何参与权重加载和 forward？](05_quantization_kernels.md)

回答：

```text
GPTQ / AWQ / FP8 / INT8 / Marlin / CUTLASS 等路径如何分类？
quantized linear 如何替代普通 linear？
scale / zero point / group size 如何进入 kernel？
```

### 06. Fused MoE kernels

- [MoE fused 算子如何执行 expert routing？](06_fused_moe_kernels.md)

回答：

```text
router top-k、expert dispatch、grouped GEMM、combine 如何串起来？
MoE quantization 和 expert parallel 如何影响 kernel？
```

### 07. Norm / activation / RoPE kernels

- [RMSNorm、activation、RoPE 等基础算子在哪里用？](07_norm_activation_rope_kernels.md)

回答：

```text
RMSNorm / LayerNorm / SiluMul / GeluAndMul / RoPE 分别服务哪些 layer？
这些算子如何影响模型结构执行效率？
```

### 08. Sampling / logits kernels

- [logits 与 sampling 相关算子如何工作？](08_sampling_logits_kernels.md)

回答：

```text
logits processor、logprobs、top-k/top-p、rejection sampling 有哪些算子路径？
spec decode 和 structured output 如何影响采样侧计算？
```

### 09. Triton / CUDA / torch fallback

- [不同算子 backend 如何选择和回退？](09_backend_selection_and_fallback.md)

回答：

```text
什么时候走 CUDA extension？
什么时候走 Triton？
什么时候走 torch fallback？
平台、dtype、shape、硬件能力如何影响选择？
```

### 10. CUDA Graph / compile 交互

- [算子如何和 CUDA Graph / torch compile 协同？](10_cuda_graph_compile_interaction.md)

回答：

```text
哪些算子适合被 capture？
shape stability、padding、workspace、临时 buffer 有什么要求？
算子 fallback 如何影响 capture？
```

### 11. Parallelism 交互

- [算子如何适配 TP / PP / DP / EP？](11_parallelism_operator_interaction.md)

回答：

```text
tensor parallel 如何切分 linear / attention？
expert parallel 如何影响 MoE kernel？
通信算子和计算算子如何交错？
```

### 12. Debug 与性能分析

- [算子问题如何定位和分析？](12_operator_debugging_and_profiling.md)

回答：

```text
如何判断走了哪个 kernel？
如何定位 fallback、NaN、shape mismatch、性能瓶颈？
profiling / env flags / log 该看什么？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
operators_overview.md
  → 01_operator_system_role.md
  → 02_python_to_native_ops.md
  → 09_backend_selection_and_fallback.md
```

### 3.2 按模型执行链路阅读

```text
../executor_worker_model_runner/07_model_forward_and_logits.md
  → 01_operator_system_role.md
  → 03_attention_kernels.md
  → 04_kv_cache_kernels.md
  → 07_norm_activation_rope_kernels.md
  → 08_sampling_logits_kernels.md
```

### 3.3 按性能优化阅读

```text
operators_overview.md
  → 03_attention_kernels.md
  → 05_quantization_kernels.md
  → 06_fused_moe_kernels.md
  → 10_cuda_graph_compile_interaction.md
  → 12_operator_debugging_and_profiling.md
```

### 3.4 和已有专题联动阅读

```text
../attention/attention_overview.md
  → 03_attention_kernels.md

../quantization/quantization_overview.md
  → 05_quantization_kernels.md

../compilation_and_cuda_graph/compilation_and_cuda_graph_overview.md
  → 10_cuda_graph_compile_interaction.md

../parallelism/parallelism_overview.md
  → 11_parallelism_operator_interaction.md
```

---

## 4. 文档定位

```text
README.md：
  当前目录索引和阅读路线。

operators_overview.md：
  算子体系总览，适合建立全局图。

01-12：
  按问题拆开的专题文档，适合后续逐段补充源码细节。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
vLLM 的算子层负责把模型 layer 的抽象计算，落到具体硬件 backend 上执行；它连接 ModelRunner / model layer 和 CUDA、Triton、FlashAttention、FlashInfer、量化、MoE、CUDA Graph 等底层能力。
```
