# vLLM V1 Operators / Kernels 逻辑梳理

源码位置：

- `code/vllm/vllm/_custom_ops.py`
- `code/vllm/vllm/_C/`
- `code/vllm/vllm/attention/`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/csrc/`

本文按“先定边界，再走调用链路，再按算子族拆解，最后总结 backend、编译、并行和 debug”的方式，梳理 vLLM V1 的 operators / kernels 体系。

它不是逐个 kernel 参数的字典式说明，而是先回答几个最关键的问题：

```text
1. vLLM 里哪些东西算 operator / kernel？
2. 算子层和 ModelRunner、model layer、attention backend 的关系是什么？
3. Python 层如何调用 native CUDA / Triton / torch fallback？
4. attention、KV cache、quantization、MoE、norm、activation、RoPE、sampling 分别有哪些算子族？
5. backend selection、CUDA Graph、torch compile、parallelism 如何影响算子选择？
6. 算子问题应该如何 debug 和 profile？
```

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本文按“先定角色，再走主链路，再拆关键阶段”的顺序组织。

要回答的问题分成 12 组：

```text
1. 算子体系在 vLLM 里处于哪一层？
2. Python wrapper 如何调用 native op？
3. attention kernels 如何支撑 prefill / decode？
4. KV cache kernels 如何维护 block / slot / cache layout？
5. quantization kernels 如何替代普通 dense linear？
6. fused MoE kernels 如何把 routing、dispatch、GEMM、combine 串起来？
7. norm / activation / RoPE kernels 如何支撑基础模型层？
8. sampling / logits kernels 如何支撑输出 token 选择？
9. backend selection 和 fallback 如何决定实际执行路径？
10. CUDA Graph / torch compile 对算子有什么约束？
11. TP / PP / DP / EP 如何影响算子形态？
12. 算子问题如何定位、验证和优化？
```

阅读顺序建议：

```text
operators_overview.md
  → 01_operator_system_role.md
  → 02_python_to_native_ops.md
  → 03_attention_kernels.md
  → 04_kv_cache_kernels.md
  → 05_quantization_kernels.md
  → 06_fused_moe_kernels.md
  → 07_norm_activation_rope_kernels.md
  → 08_sampling_logits_kernels.md
  → 09_backend_selection_and_fallback.md
  → 10_cuda_graph_compile_interaction.md
  → 11_parallelism_operator_interaction.md
  → 12_operator_debugging_and_profiling.md
```

如果只想先抓住一条主线，可以先读总览，再读 `01`、`02`、`03`、`09`。

---

## 1. 算子层是什么

vLLM 的算子层不是一个单独模块，而是一组分散在多个子系统里的底层执行入口。

可以把它理解成：

```text
model layer / runtime state
  → Python wrapper
  → backend-specific implementation
  → CUDA / Triton / FlashAttention / FlashInfer / CUTLASS / torch
```

它负责把上层语义计算变成具体硬件执行，包括：

```text
- attention；
- KV cache 读写和布局变换；
- quantized linear；
- fused MoE；
- norm / activation / RoPE；
- logits / sampling；
- cache copy / reshape / swap；
- backend fallback 和平台兼容。
```

---

## 2. 算子层不是什么

算子层不负责：

```text
- 接收用户请求；
- 维护 Scheduler 队列；
- 决定 token budget；
- 管理 request lifecycle；
- 组织跨请求 batch；
- 构造最终 RequestOutput。
```

这些分别属于 Engine、Scheduler、ModelRunner、OutputProcessor 等上层组件。

---

## 3. 一句话总览链路

最小主链路是：

```text
SchedulerOutput
  → ModelRunner 准备 input / metadata / cache mapping
  → model layer forward
  → operator wrapper
  → backend kernel
  → tensor output
  → logits / sampler / next layer / cache update
```

如果只看算子入口，可以压缩成：

```text
Layer abstraction
  → operator wrapper
  → concrete backend
  → hardware execution
```

---

## 4. 为什么需要单独的算子专题

已有专题通常从系统层视角解释：

```text
Scheduler：决定本轮跑什么；
Executor / Worker / ModelRunner：决定怎么把调度计划跑进模型；
Attention：解释 attention metadata 和 backend；
Quantization：解释量化配置与 layer 替换；
Compilation / CUDA Graph：解释 capture 和 compile；
Parallelism：解释并行拓扑和通信。
```

算子专题要补上的视角是：

```text
这些抽象最终落到哪些底层计算入口？
每类算子吃什么输入、产出什么输出？
哪些条件会改变实际 kernel？
出现 fallback 或性能问题时应该从哪里查？
```

---

## 5. 算子族总图

```text
Operators / Kernels
├── Python native op bridge
│   ├── _custom_ops.py
│   ├── torch.ops
│   └── fallback wrappers
├── Attention kernels
│   ├── PagedAttention
│   ├── FlashAttention
│   ├── FlashInfer
│   ├── FlashMLA
│   └── Triton attention paths
├── KV cache kernels
│   ├── cache write
│   ├── reshape / copy / swap
│   ├── block table / slot mapping usage
│   └── KV cache quantization
├── Quantization kernels
│   ├── GPTQ / AWQ
│   ├── FP8 / INT8
│   ├── Marlin / CUTLASS / scaled mm
│   └── quantized linear methods
├── Fused MoE kernels
│   ├── top-k routing
│   ├── expert dispatch
│   ├── grouped GEMM
│   └── combine / reduce
├── Basic layer kernels
│   ├── RMSNorm / LayerNorm
│   ├── SiluMul / GeluAndMul
│   ├── RoPE / M-RoPE
│   └── embedding / logits helpers
├── Sampling / logits kernels
│   ├── logits processor
│   ├── logprobs
│   ├── top-k / top-p
│   └── rejection sampling
└── Runtime interaction
    ├── backend selection
    ├── CUDA Graph capture
    ├── torch compile
    ├── parallelism
    └── debug / profiling
```

---

## 6. 和 ModelRunner 的关系

ModelRunner 不直接关心每个 CUDA kernel 的实现细节，但它会准备算子所需的关键输入：

```text
- input_ids / inputs_embeds；
- positions；
- slot mapping；
- block table；
- attention metadata；
- kv cache tensor；
- logits indices；
- sampling metadata；
- LoRA / quant / multimodal / spec decode 状态。
```

因此算子专题需要和执行层文档联动阅读：

```text
../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
../executor_worker_model_runner/07_model_forward_and_logits.md
../executor_worker_model_runner/08_sampling_and_model_runner_output.md
```

---

## 7. 和 Attention 专题的关系

Attention 专题关注：

```text
- backend 如何选择；
- metadata 如何构造；
- prefill / decode / cascade / sliding window 如何表达；
- block table、slot mapping、KV cache layout 如何被 attention 使用。
```

算子专题进一步追问：

```text
- 具体调用哪个 kernel？
- kernel 输入张量是什么？
- fallback 条件是什么？
- 性能瓶颈通常在哪里？
```

---

## 8. 和 Quantization / MoE 专题的关系

Quantization 专题关注配置、权重加载、layer method 替换。

算子专题关注：

```text
- quantized matmul 的具体 kernel 家族；
- scale / zero point / group size 如何参与计算；
- dtype / hardware capability 如何影响路径；
- quantization 和 LoRA、TP、MoE 如何组合。
```

MoE 专题关注模型结构和 routing 语义。

算子专题关注：

```text
- routed token 如何重排；
- expert GEMM 如何分组；
- dispatch / combine 是否 fused；
- EP / TP 如何改变 kernel 输入布局。
```

---

## 9. 和 CUDA Graph / compile 的关系

CUDA Graph / compile 对算子提出额外约束：

```text
- shape 要稳定；
- 临时 buffer 和 workspace 要可复用；
- kernel launch 序列要可 capture；
- fallback 分支不能在 capture 中频繁变化；
- Python control flow 不能成为 hot path。
```

因此算子专题需要解释：

```text
为什么某些算子能被 capture；
为什么某些 shape / dtype / backend 会 fallback；
为什么 padding 和 static buffer 对性能重要。
```

---

## 10. 和 Parallelism 的关系

并行会改变算子的输入规模和通信边界：

```text
TP：切分 linear / attention head；
PP：切分 layer 并传递 intermediate tensors；
DP：复制模型、分摊请求；
EP：切分 experts；
CP：切分 context / attention 相关状态。
```

算子专题要回答：

```text
计算算子和通信算子如何交错？
哪些 kernel 是 rank-local 的？
哪些结果需要 all-reduce / all-gather / reduce-scatter？
MoE expert parallel 如何影响 fused MoE kernel？
```

---

## 11. Debug 和 profiling 的入口

算子问题通常表现为：

```text
- fallback 到慢路径；
- kernel 不支持当前 dtype / shape；
- illegal memory access；
- NaN / Inf；
- output mismatch；
- CUDA Graph capture 失败；
- profiler 里 kernel launch 过多；
- attention 或 quantized GEMM 成为瓶颈。
```

后续章节需要补充：

```text
- 如何确认实际 backend；
- 如何打开相关 log；
- 如何缩小到某个算子族；
- 如何用 profiler 观察 kernel；
- 如何判断是 shape、dtype、layout、硬件能力还是 fallback 问题。
```

---

## 12. 文档定位

```text
operators_overview.md：
  算子体系总览，适合快速建立全局图。

01-12：
  按问题拆开的专题文档，适合后续逐段补充源码细节。
```

---

## 13. 一句话总结

vLLM 的算子层负责把 ModelRunner 和模型 layer 产生的抽象计算，落到具体硬件 backend 上执行；它是 attention、KV cache、quantization、MoE、sampling、CUDA Graph、parallelism 等专题最终交汇的底层执行面。

最核心的主线是：

```text
ModelRunner 准备状态
  → model layer 调用算子 wrapper
  → backend 选择具体 kernel
  → kernel 完成张量计算
  → 结果回到 layer / logits / sampler / cache
```
