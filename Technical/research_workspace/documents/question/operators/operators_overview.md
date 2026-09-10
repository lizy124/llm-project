# vLLM V1 Operators / Kernels 逻辑梳理

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\_custom_ops.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\_C\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\ops\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\quantization\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\platforms\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\profiler\`
- `D:\lzy\project\kv_pool\code\vllm\csrc\`

本文按“先定边界，再走调用链路，再按算子族拆解，最后总结 backend、编译、并行和 debug”的方式，梳理 vLLM V1 的 operators / kernels 体系。

它不是逐个 kernel 参数的字典式说明，而是回答几个主线问题：

```text
1. vLLM 里哪些东西算 operator / kernel？
2. 算子层和 Scheduler、ModelRunner、model layer、backend 的边界是什么？
3. Python 层如何调用 CUDA extension / Triton / FlashAttention / FlashInfer / torch fallback？
4. attention、KV cache、quantization、MoE、norm、activation、RoPE、sampling 分别有哪些算子族？
5. backend selection、CUDA Graph、torch compile、parallelism 如何影响实际执行路径？
6. 出现 fallback、NaN、shape mismatch、capture 失败或性能异常时怎么定位？
```

---

## 1. 一句话总览

vLLM 的算子层是：

```text
把模型层抽象计算落到具体硬件 backend 执行的底层执行面。
```

最小心智模型是：

```text
Scheduler 决定本轮跑哪些 token；
ModelRunner 准备 input、positions、KV cache mapping、attention metadata、sampling metadata；
model layer 表达数学计算；
operator wrapper / backend 负责真正执行 kernel；
输出 tensor 回到下一层、KV cache、logits、sampler 或 Scheduler。
```

压缩成链路就是：

```text
SchedulerOutput
  → GPUModelRunner / Worker state
  → model layer forward
  → operator wrapper
  → selected backend
  → CUDA / Triton / FlashAttention / FlashInfer / CUTLASS / torch
  → tensor output
  → next layer / cache update / logits / sampler / ModelRunnerOutput
```

---

## 2. 算子层在系统中的位置

vLLM V1 的主要执行层级可以画成：

```text
API / Engine
  → EngineCore
  → Scheduler
  → Executor
  → Worker
  → ModelRunner
  → Model Layer
  → Operator / Kernel
  → GPU / CPU / XPU / ROCm backend
```

算子层不负责：

```text
- 接收用户请求；
- 管理 request lifecycle；
- 决定 token budget；
- 维护 waiting / running 队列；
- 分配请求级 KV blocks 的策略；
- 构造最终 RequestOutput。
```

这些属于 Engine、Scheduler、KV cache manager、OutputProcessor 等上层组件。

算子层负责：

```text
- 执行 attention；
- 写入和读取 KV cache；
- 执行 dense / quantized linear；
- 执行 fused MoE；
- 执行 norm / activation / RoPE；
- 执行 logits / logprobs / sampling；
- 处理通信 primitive；
- 根据平台和 shape 选择 backend；
- 适配 CUDA Graph / torch compile / profiler。
```

---

## 3. 四个容易混淆的词

### 3.1 Layer

Layer 是模型结构层面的抽象，例如：

```text
Attention layer
Linear layer
RMSNorm / LayerNorm
RoPE
SiluMul / GeluAndMul
Fused MoE
Logits processor / sampler
```

Layer 负责表达“模型要算什么”。

### 3.2 Operator wrapper

Operator wrapper 是 Python 层的调用入口，负责把 layer 的调用转成某个 backend 的 op 调用。

典型入口：

```text
vllm/_custom_ops.py
vllm/v1/attention/ops/
vllm/model_executor/layers/*
vllm/v1/sample/ops/
```

它通常负责：

```text
参数整理；
输出 tensor 分配；
dtype / shape 分支；
平台判断；
调用 torch.ops；
调用 Triton kernel；
调用第三方 backend；
fallback 到 torch op。
```

### 3.3 Backend

Backend 是一组具体实现，例如：

```text
FlashAttention
FlashInfer
FlashMLA
Triton attention
CUTLASS MLA
Marlin
scaled mm
ROCm aiter
XPU custom op
torch native fallback
```

backend 负责“这段计算怎么在当前硬件上跑”。

### 3.4 Kernel

Kernel 是最终硬件执行单元，例如：

```text
CUDA extension kernel
Triton JIT kernel
FlashAttention kernel
FlashInfer sampling kernel
CUTLASS GEMM kernel
NCCL collective kernel
torch aten kernel
```

在 profiler 里看到的通常是 kernel 或 torch op，而不是 vLLM 的高级 layer 名称。

---

## 4. 算子族总图

```text
Operators / Kernels
├── Python native op bridge
│   ├── _custom_ops.py
│   ├── torch.ops._C / torch.ops.vllm
│   ├── custom CUDA extension
│   ├── fake impl / compile support
│   └── torch / Triton fallback wrappers
├── Attention kernels
│   ├── PagedAttention
│   ├── FlashAttention / FlashInfer
│   ├── FlashMLA / CUTLASS MLA
│   ├── Triton attention
│   ├── prefill / decode / cascade / sliding window
│   └── attention metadata / slot mapping / block table
├── KV cache kernels
│   ├── KV cache write
│   ├── reshape / copy / swap
│   ├── slot mapping → physical block
│   ├── KV cache dtype / quantization
│   └── external KV connector 相关 cache 操作
├── Quantization kernels
│   ├── GPTQ / AWQ
│   ├── FP8 / INT8 / MXFP4 / NVFP4
│   ├── Marlin / CUTLASS / scaled mm
│   ├── quantized linear methods
│   └── quantized MoE / KV cache 组合
├── Fused MoE kernels
│   ├── router top-k / grouped top-k
│   ├── token-expert dispatch
│   ├── grouped GEMM / expert GEMM
│   ├── shared expert
│   ├── all2all / EP combine
│   └── quantized expert kernels
├── Basic layer kernels
│   ├── RMSNorm / LayerNorm
│   ├── SiluMul / GeluAndMul
│   ├── RoPE / M-RoPE
│   ├── embedding / vocab parallel helper
│   └── residual / activation fused paths
├── Sampling / logits kernels
│   ├── logits processor / lm_head
│   ├── grammar bitmask / allowed ids / bad words
│   ├── penalties
│   ├── top-k / top-p / min-p / temperature
│   ├── logprobs gather / ranks
│   └── speculative decoding rejection sampler
└── Runtime interaction
    ├── backend selection / fallback
    ├── CUDA Graph / torch compile
    ├── TP / PP / DP / EP / CP communication
    ├── profiler / NVTX / record_function
    └── debug / validation / minimal repro
```

---

## 5. 目录阅读顺序

建议阅读顺序：

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

如果只想先抓主线：

```text
operators_overview.md
  → 01_operator_system_role.md
  → 02_python_to_native_ops.md
  → 03_attention_kernels.md
  → 09_backend_selection_and_fallback.md
  → 10_cuda_graph_compile_interaction.md
  → 12_operator_debugging_and_profiling.md
```

如果关注性能：

```text
03_attention_kernels.md
  → 05_quantization_kernels.md
  → 06_fused_moe_kernels.md
  → 09_backend_selection_and_fallback.md
  → 10_cuda_graph_compile_interaction.md
  → 11_parallelism_operator_interaction.md
  → 12_operator_debugging_and_profiling.md
```

如果关注输出侧：

```text
08_sampling_logits_kernels.md
  → ../executor_worker_model_runner/08_sampling_and_model_runner_output.md
  → ../scheduler/08_update_after_worker_output.md
```

---

## 6. 和 ModelRunner 的关系

ModelRunner 不直接实现每个 CUDA kernel，但它决定算子能否正确运行，因为它准备了所有运行时输入。

GPUModelRunner 会准备：

```text
input_ids / inputs_embeds
positions / mrope positions
intermediate_tensors
attention metadata
slot mapping
block table
KV cache tensors
logits_indices
sampling metadata
spec decode metadata
LoRA mapping
multimodal encoder output
cudagraph batch descriptor
```

可以把 ModelRunner 和算子关系理解成：

```text
ModelRunner 准备“本轮怎么跑”；
model layer 表达“模型要算什么”；
operator backend 决定“这段计算由哪个 kernel 跑”。
```

典型主链路：

```text
GPUModelRunner.execute_model()
  → _determine_batch_execution_and_padding()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
  → attention / linear / norm / MoE / RoPE kernels
  → hidden states / logits
  → sample_tokens()
  → sampler / logprobs / rejection sampler kernels
  → ModelRunnerOutput
```

配套阅读：

```text
../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
../executor_worker_model_runner/07_model_forward_and_logits.md
../executor_worker_model_runner/08_sampling_and_model_runner_output.md
```

---

## 7. 01：算子体系定位

`01_operator_system_role.md` 回答：vLLM 语境里的 operator / kernel 是什么。

核心结论：

```text
算子层不是 Scheduler，也不是 ModelRunner；
算子层是模型执行过程中真正落到硬件 backend 的计算入口。
```

它重点区分：

```text
Layer：表达模型结构；
Operator wrapper：Python 调用封装；
Backend：具体实现族；
Kernel：最终硬件执行单元。
```

这篇适合先建立术语边界，避免把“调度慢”“模型层慢”“kernel 慢”“backend fallback”混成一个问题。

---

## 8. 02：Python 到 native op 的桥

`02_python_to_native_ops.md` 关注：Python 层如何调用底层 op。

典型链路是：

```text
model layer
  → Python wrapper
  → vllm._custom_ops / torch.ops / Triton function
  → registered native op
  → CUDA / ROCm / XPU / CPU kernel
```

`_custom_ops.py` 常见职责：

```text
封装 torch.ops._C.*；
统一参数顺序；
创建输出 tensor；
按平台或 env flag 切换实现；
提供 fallback；
配合 torch.compile fake impl / schema。
```

这篇是理解“源码里看不到 kernel 实现”的入口：Python 只看到 wrapper，真正的 kernel 可能在 `csrc/`、Triton 文件或第三方库中。

---

## 9. 03：Attention kernels

`03_attention_kernels.md` 关注 attention 算子如何支撑 prefill / decode。

attention 是 vLLM 最核心的算子族，主要输入包括：

```text
query / key / value
KV cache tensor
slot mapping
block table
attention metadata
seq lens / query lens
backend-specific workspace
```

核心链路：

```text
Attention layer
  → get_attn_backend(...)
  → build attention metadata
  → write KV cache
  → prefill / decode attention kernel
  → output hidden states
```

prefill 和 decode 的差异：

```text
prefill：一次处理较长 prompt，Q/K/V 当前 tokens 多；
decode：每个 request 通常新增少量 token，大量读取历史 KV cache；
cascade / prefix / sliding window：进一步改变 metadata 和 kernel 形态。
```

attention backend 还会影响：

```text
KV cache layout；
CUDA Graph 支持；
CP / DCP 是否可用；
FlashAttention / FlashInfer / Triton / FlashMLA / CUTLASS MLA 的选择。
```

---

## 10. 04：KV cache kernels

`04_kv_cache_kernels.md` 关注 KV cache 相关算子。

KV cache 算子回答的是：

```text
新产生的 K/V 怎么写入物理 cache？
逻辑 token 位置怎么映射到 block / slot？
decode 时 attention 怎么按 block table 读取历史 K/V？
cache copy / swap / reshape / quantization 在哪里发生？
```

主链路：

```text
Scheduler 分配 KV blocks
  → ModelRunner 构造 slot_mapping / block_table
  → attention / cache op 写入 K/V
  → attention backend 根据 metadata 读取 cache
  → request 完成后 Scheduler / KV manager 释放 blocks
```

KV cache 算子特别容易和这些问题相关：

```text
slot mapping shape mismatch；
block table 不一致；
KV cache layout 和 backend 要求不一致；
FP8 / quantized KV cache dtype 不支持；
external KV connector load / save 后 invalid blocks 重算。
```

---

## 11. 05：Quantization kernels

`05_quantization_kernels.md` 关注量化算子如何替代普通 dense linear。

核心链路：

```text
LoadConfig / QuantConfig
  → QuantizationConfig
  → layer.get_quant_method(...)
  → create_weights(...)
  → weight_loader 分片加载 qweight / scales / zero points
  → quant_method.apply(...)
  → backend-specific matmul kernel
```

量化算子要同时处理：

```text
weight-only quantization；
activation quantization；
scale / zero point；
group size / block size；
packed weight layout；
TP shard offset；
Marlin / CUTLASS / scaled mm / Triton / torch fallback。
```

它和 TP / MoE / LoRA 的组合尤其重要：

```text
TP 改变每个 rank 的 weight shard；
MoE 改变 expert weight 和 grouped GEMM；
LoRA 可能引入额外 matmul 或 fused path 限制；
CUDA Graph 要求 shape 和 workspace 稳定。
```

---

## 12. 06：Fused MoE kernels

`06_fused_moe_kernels.md` 关注 MoE 的 routed expert 执行。

MoE 算子链路是：

```text
hidden states
  → router logits
  → top-k expert selection
  → token-expert mapping
  → dispatch / reorder tokens
  → grouped expert GEMM
  → apply top-k weights
  → combine / restore token order
```

如果开启 EP，链路还会加入：

```text
EP all2all dispatch
  → local expert compute
  → EP all2all combine
```

MoE 性能瓶颈通常不只是 GEMM，还包括：

```text
routing top-k；
token reorder；
expert token imbalance；
all2all communication；
shared expert；
quantized expert kernel；
small expert batch 下 kernel launch overhead。
```

---

## 13. 07：Norm / activation / RoPE kernels

`07_norm_activation_rope_kernels.md` 关注基础模型层算子。

这类算子单个看不如 attention / GEMM 显眼，但调用频率高，对模型 forward 性能有明显影响。

常见算子：

```text
RMSNorm / LayerNorm
fused residual RMSNorm
SiluMul / GeluAndMul
RoPE / M-RoPE
embedding helper
activation fused path
```

典型链路：

```text
Transformer block
  → norm
  → QKV projection
  → RoPE
  → attention
  → MLP activation
  → next block
```

排查重点：

```text
dtype 是否支持；
hidden size 是否满足 kernel；
input 是否 contiguous；
是否 fallback 到 torch op；
是否被 CUDA Graph capture；
M-RoPE / multimodal positions 是否正确。
```

---

## 14. 08：Sampling / logits kernels

`08_sampling_logits_kernels.md` 关注输出侧计算。

主链路是：

```text
hidden states
  → logits processor / lm_head
  → grammar bitmask / allowed ids / bad words
  → penalties / logits processors
  → temperature
  → top-k / top-p / min-p
  → greedy or random sample
  → logprobs gather
  → ModelRunnerOutput
```

如果开启 speculative decoding：

```text
draft tokens / draft probs
  → target logits
  → RejectionSampler
  → accepted / recovered / bonus tokens
  → sampled_token_ids
```

输出侧容易产生性能问题的功能：

```text
large top-p sort；
max_num_logprobs；
prompt_logprobs；
structured output grammar bitmask；
per-request generators；
spec decode rejection sampling；
D2H copy 和 Python list 转换。
```

---

## 15. 09：Backend selection / fallback

`09_backend_selection_and_fallback.md` 关注同一个上层算子最后由哪个 backend 执行。

通用链路：

```text
operator request
  → user config / env flags
  → platform detection
  → dtype / shape / capability validation
  → dependency import check
  → selected backend
  → fallback backend or error
```

attention backend 走：

```text
get_attn_backend()
  → AttentionSelectorConfig
  → current_platform.get_attn_backend_cls(...)
  → candidate priority list
  → validate_configuration(...)
  → selected AttentionBackend class
```

sampling backend 在 `TopKTopPSampler` 内按平台和 logprobs mode 分派。

quantization backend 通过：

```text
quantization string
  → QuantizationConfig
  → layer quant_method
  → backend-specific apply()
```

fallback 不一定是错误。它可能只是兼容路径，但如果 CUDA 上大量 fallback 到 torch op，就通常意味着性能问题。

---

## 16. 10：CUDA Graph / torch compile 交互

`10_cuda_graph_compile_interaction.md` 关注算子如何满足稳定执行要求。

主链路：

```text
CompilationConfig
  → attention backend capability
  → cudagraph capture sizes / compile sizes
  → CudagraphDispatcher keys
  → runtime batch padding
  → set_forward_context(...)
  → model forward with selected operators
  → capture or replay CUDA graph
```

CUDA Graph 关注：

```text
shape 稳定；
tensor 地址稳定；
kernel launch 序列稳定；
不能在 capture 中做非法同步或动态分配；
backend 必须支持对应 cudagraph mode。
```

torch compile 关注：

```text
Dynamo graph break；
dynamic shape specialization；
unsupported op；
fullgraph trace；
compile_sizes 和 cudagraph padding 是否一致。
```

这篇解释为什么算子选择、padding、static buffer、attention metadata 和 fallback 会反过来影响图捕获。

---

## 17. 11：Parallelism 交互

`11_parallelism_operator_interaction.md` 关注 TP / PP / DP / EP / CP 对算子形态的影响。

核心结论：

```text
并行策略会把全局 layer 拆成 rank-local compute + collective communication。
```

不同并行维度的影响：

```text
TP：切分 linear、attention heads、vocab，需要 all-reduce / all-gather；
PP：切分 layer，stage 间 send / recv intermediate tensors；
DP：复制模型，跨 rank 协调 batch padding、microbatch、cudagraph mode；
EP：切分 experts，MoE token 需要 all2all dispatch / combine；
CP：切分 context，attention backend 需要支持 LSE / PCP / DCP merge。
```

典型 TP block：

```text
ColumnParallelLinear
  → local attention / activation
  → RowParallelLinear
  → tensor_model_parallel_all_reduce
```

典型 EP MoE：

```text
router top-k
  → all2all dispatch
  → local expert GEMM
  → all2all combine
```

---

## 18. 12：Debug / profiling

`12_operator_debugging_and_profiling.md` 关注算子问题如何定位。

最小排查链路：

```text
现象
  → 定位算子族
  → 确认实际 backend
  → 检查 shape / dtype / layout / metadata
  → 观察 profiler kernel / record_function range
  → 判断是配置、fallback、实现 bug、硬件限制还是动态图问题
```

常见问题分类：

```text
backend 不支持 dtype / shape；
optional dependency 缺失；
CUDA extension 未编译或加载失败；
metadata 与 tensor shape 不一致；
KV cache layout 或 slot mapping 错误；
fallback 到慢路径；
CUDA Graph capture / replay 不稳定；
quantization scale 或 packed weight 不匹配；
多卡 collective shape / 顺序不一致。
```

重点入口：

```text
backend selection logs；
validate_configuration reasons；
TorchProfilerWrapper / CudaProfilerWrapper；
record_function_or_nullcontext ranges；
LayerwiseProfileResults；
CUDAGraphStat；
VLLM_COMPUTE_NANS_IN_LOGITS。
```

---

## 19. 一次完整 forward 的算子视角

下面从算子角度看一次 generation step。

```text
1. Scheduler 产出 SchedulerOutput。
2. ModelRunner 根据 scheduled tokens 更新 input batch。
3. KV manager / Scheduler 提供 block 分配结果。
4. ModelRunner 构造 slot mapping、block table、attention metadata。
5. CudagraphDispatcher 决定 runtime mode 和 padded batch descriptor。
6. set_forward_context 写入 attention metadata、cudagraph mode、batch descriptor。
7. embedding / RoPE / norm / attention / linear / MoE 等 layer 依次执行。
8. 每个 layer 内部调用对应 operator wrapper 和 backend kernel。
9. attention kernel 读写 KV cache。
10. TP / EP / PP / DP / CP 在需要的位置插入 communication op。
11. 最后 rank 计算 logits。
12. sample_tokens 应用 grammar、penalty、top-k/top-p、logprobs 或 rejection sampler。
13. bookkeeping 把 GPU tensor 转成 ModelRunnerOutput 需要的结构。
14. Scheduler.update_from_output 消费 sampled tokens 和状态。
```

可以压缩成：

```text
metadata / mapping / buffers
  → rank-local compute kernels
  → communication kernels
  → output-side sampling kernels
  → structured engine output
```

---

## 20. backend 与 runtime 的总关系

同一段上层计算，实际执行路径由这些因素共同决定：

```text
平台：CUDA / ROCm / CPU / XPU；
硬件：compute capability / GPU architecture；
dtype：fp16 / bf16 / fp8 / int8 / int4；
shape：head size / hidden size / block size / vocab size / batch tokens；
模型结构：dense / MoE / MLA / MQA / GQA / sliding window；
配置：attention backend、quantization、cudagraph mode、compile mode；
并行：TP / PP / DP / EP / CP；
依赖：FlashAttention、FlashInfer、Triton、CUTLASS、aiter；
运行时功能：LoRA、spec decode、structured output、logprobs、KV connector。
```

因此一个算子问题通常不能只看单个 kernel 函数。需要同时问：

```text
1. 当前 rank-local shape 是什么？
2. 当前 backend 是什么？
3. 当前 metadata 是否和 tensor shape 对齐？
4. 当前 parallel group 是否要求通信？
5. 当前 cudagraph / compile mode 是否改变了 padding 和执行路径？
```

---

## 21. 常见排查入口速查

### 21.1 attention 慢或报错

先看：

```text
attention backend 日志；
AttentionSelectorConfig；
head_size / dtype / kv_cache_dtype / block_size；
slot mapping / block table；
KV cache layout；
cudagraph runtime mode；
profiler 中 flash_attn / flashinfer / triton / paged attention kernel。
```

### 21.2 quantized linear 报错或变慢

先看：

```text
QuantizationConfig；
linear quant_method；
qweight / scale / zero point shape；
TP shard offset；
packed_dim / packed_factor；
Marlin / CUTLASS / scaled_mm 是否出现；
是否 fallback 到 aten::mm。
```

### 21.3 MoE 性能异常

先看：

```text
router top-k；
expert token count；
EP all2all dispatch / combine；
fused_moe kernel；
expert quantization；
shared expert；
load imbalance。
```

### 21.4 sampling 慢

先看：

```text
top-p 是否走 sort；
是否请求 logprobs / prompt_logprobs；
是否有 grammar bitmask；
是否有 per-request generators；
是否开启 spec decode rejection sampler；
bookkeeping / D2H copy 是否耗时。
```

### 21.5 CUDA Graph 没命中

先看：

```text
CUDAGraphStat.runtime_mode；
num_tokens / num_tokens_padded；
batch descriptor 是否命中 capture key；
DP rank 是否同步降级；
attention backend 是否支持 full / piecewise；
是否有 encoder input、cascade attention、dynamic KV scale、LoRA case 不匹配。
```

---

## 22. 和其他专题的衔接

算子专题需要和以下专题交叉阅读。

### 22.1 Scheduler

Scheduler 决定：

```text
本轮哪些 request 被调度；
每个 request 跑多少 token；
KV block 如何分配；
是否有 grammar bitmask；
spec decode token 状态如何更新。
```

算子层消费这些结果。

### 22.2 Executor / Worker / ModelRunner

ModelRunner 决定：

```text
如何把 SchedulerOutput 变成 tensor；
如何准备 attention metadata；
如何执行 forward；
如何调用 sampler；
如何包装 ModelRunnerOutput。
```

算子层是 ModelRunner forward / sampling 的底层执行面。

### 22.3 Attention

Attention 专题解释 metadata 和 backend 语义。

算子专题进一步解释：

```text
metadata 如何进入 kernel；
backend fallback 怎么发生；
KV cache layout 如何限制 kernel；
profiler 中应该看哪些 kernel。
```

### 22.4 Parallelism

Parallelism 专题解释拓扑。

算子专题进一步解释：

```text
rank-local operator shape；
communication primitive；
TP linear / vocab parallel；
EP MoE dispatch；
DP cudagraph padding；
CP attention merge。
```

---

## 23. 最小记忆版

如果只记住这套 operators 文档的核心，可以记下面几句：

```text
1. Scheduler 决定“跑什么 token”，ModelRunner 决定“怎么组织 tensor”，算子决定“怎么在硬件上跑”。
2. 算子不是单个模块，而是 attention、KV cache、linear、MoE、norm、sampling、communication 的底层执行集合。
3. Python wrapper 只是入口，真正执行可能是 CUDA extension、Triton、FlashAttention、FlashInfer、CUTLASS、NCCL 或 torch fallback。
4. backend selection 由平台、dtype、shape、依赖、配置和并行方式共同决定。
5. CUDA Graph / torch compile 要求 shape、buffer、执行路径稳定，因此会影响算子选择和 padding。
6. 多卡并行会把全局算子拆成 rank-local compute + collective communication。
7. 排查问题时先确认实际 backend，再对齐 shape / dtype / layout / metadata / parallel group / cudagraph mode。
```

---

## 24. 总结

vLLM 的 operators / kernels 体系是整个 V1 执行链路的底层交汇点。

它向上承接：

```text
SchedulerOutput
ModelRunner state
model layer abstraction
attention metadata
KV cache mapping
sampling metadata
parallel topology
```

向下落到：

```text
CUDA extension
Triton kernel
FlashAttention / FlashInfer / FlashMLA
CUTLASS / Marlin / scaled mm
ROCm aiter / XPU op
NCCL collective
torch fallback
```

最核心的主线是：

```text
ModelRunner 准备状态
  → model layer 调用算子 wrapper
  → backend selection 选择具体实现
  → kernel 完成 rank-local compute 或 communication
  → 结果回到 layer / KV cache / logits / sampler / Scheduler
```

如果把 vLLM 看成一个 serving runtime，算子层就是它把“调度计划”和“模型结构”真正变成硬件执行的地方。理解这层，才能解释为什么同一个模型在不同 dtype、backend、parallel config、cudagraph mode、logprobs 参数下，会表现出完全不同的吞吐、延迟和报错形态。
