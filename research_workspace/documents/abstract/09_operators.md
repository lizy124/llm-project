# 09 operators 背诵文档

## 1. 专题定位

`operators` 讲的是 vLLM 中模型计算最终如何落到 CUDA、Triton、FlashAttention、FlashInfer、CUTLASS、torch 等底层 backend。

它不是讲调度策略，也不是讲请求状态。

一句话：

```text
Operators / Kernels 是把模型层抽象计算落到具体硬件 backend 执行的底层执行面。
```

## 2. 最小心智模型

主链路是：

```text
SchedulerOutput
  → GPUModelRunner 准备输入和 metadata
  → model layer forward
  → operator wrapper
  → selected backend
  → CUDA / Triton / FlashAttention / FlashInfer / CUTLASS / torch
  → tensor output
  → next layer / KV cache / logits / sampler / ModelRunnerOutput
```

要背住：

```text
Scheduler 决定跑哪些 token；ModelRunner 准备张量和 metadata；model layer 表达数学计算；operator backend 负责真正执行 kernel。
```

## 3. 算子层在系统中的位置

vLLM V1 执行层级：

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

算子层负责：

```text
attention
KV cache read/write
dense / quantized linear
fused MoE
norm / activation / RoPE
logits / logprobs / sampling
communication primitive
backend selection / fallback
CUDA graph / torch compile 适配
profiler / debug 支持
```

算子层不负责：

```text
接收用户请求
维护 request lifecycle
决定 token budget
waiting / running 队列
KV block 分配策略
RequestOutput 构造
```

## 4. 四个容易混淆的词

### Layer

Layer 是模型结构抽象。

例如：

```text
Attention layer
Linear layer
RMSNorm / LayerNorm
RoPE
SiluMul / GeluAndMul
Fused MoE
Logits processor / sampler
```

Layer 表达“模型要算什么”。

### Operator wrapper

Operator wrapper 是 Python 层调用入口。

它负责：

```text
整理参数
分配输出 tensor
判断 dtype / shape
判断平台
调用 torch.ops
调用 Triton kernel
调用第三方 backend
fallback 到 torch op
```

### Backend

Backend 是一组具体实现。

例如：

```text
FlashAttention
FlashInfer
FlashMLA
Triton attention
CUTLASS MLA
Marlin
scaled mm
ROCm aiter
torch native fallback
```

Backend 负责“这段计算怎么在当前硬件上跑”。

### Kernel

Kernel 是最终硬件执行单元。

例如：

```text
CUDA extension kernel
Triton JIT kernel
FlashAttention kernel
FlashInfer sampling kernel
CUTLASS GEMM kernel
NCCL collective kernel
torch aten kernel
```

Profiler 里通常看到的是 kernel 或 torch op，而不是 vLLM 高级 layer 名称。

## 5. 算子族总览

vLLM 算子大致分为：

```text
Python native op bridge
Attention kernels
KV cache kernels
Quantization kernels
Fused MoE kernels
Norm / activation / RoPE kernels
Sampling / logits kernels
Communication kernels
Runtime selection / fallback / compile / profiling
```

每一族都服务于 ModelRunner 的某个阶段。

## 6. Python 到 native op 的桥

典型链路：

```text
model layer
  → Python wrapper
  → vllm._custom_ops / torch.ops / Triton function
  → registered native op
  → CUDA / ROCm / XPU / CPU kernel
```

`_custom_ops.py` 常见职责：

```text
封装 torch.ops._C.*
统一参数顺序
创建输出 tensor
按平台或 env flag 切换实现
提供 fallback
配合 torch.compile fake impl / schema
```

一句话：

```text
Python 里看到的通常是 wrapper，真正 kernel 可能在 csrc、Triton 文件或第三方库中。
```

## 7. Attention kernels

attention 是 vLLM 最核心算子族。

输入包括：

```text
query / key / value
KV cache tensor
slot mapping
block table
attention metadata
seq lens / query lens
backend-specific workspace
```

主链路：

```text
Attention layer
  → get_attn_backend()
  → build attention metadata
  → write KV cache
  → prefill / decode attention kernel
  → output hidden states
```

prefill 和 decode 差异：

```text
prefill：一次处理较长 prompt，当前 Q/K/V tokens 多。
decode：每个 request 通常新增少量 token，大量读取历史 KV cache。
```

attention backend 还会影响：

```text
KV cache layout
CUDA graph 支持
CP / DCP 是否可用
FlashAttention / FlashInfer / Triton / FlashMLA / CUTLASS MLA 选择
```

## 8. KV cache kernels

KV cache 算子回答：

```text
新产生的 K/V 怎么写入物理 cache？
逻辑 token 位置怎么映射到 block / slot？
decode 时 attention 怎么按 block table 读取历史 KV？
cache copy / swap / reshape / quantization 在哪里发生？
```

主链路：

```text
Scheduler 分配 KV blocks
  → ModelRunner 构造 slot_mapping / block_table
  → attention / cache op 写入 K/V
  → attention backend 根据 metadata 读取 cache
  → 请求完成后 Scheduler / KV manager 释放 blocks
```

常见问题：

```text
slot mapping shape mismatch
block table 不一致
KV cache layout 和 backend 要求不一致
FP8 / quantized KV cache dtype 不支持
external KV connector load/save 后 invalid blocks 重算
```

## 9. Quantization kernels

量化算子把普通 dense linear 替换成低精度权重 / 激活计算。

主链路：

```text
LoadConfig / QuantConfig
  → QuantizationConfig
  → layer.get_quant_method()
  → create_weights()
  → weight_loader 加载 qweight / scales / zero points
  → process_weights_after_loading()
  → quant_method.apply()
  → backend-specific matmul kernel
```

量化算子处理：

```text
weight-only quantization
activation quantization
scale / zero point
group size / block size
packed weight layout
TP shard offset
Marlin / CUTLASS / scaled mm / Triton / torch fallback
```

## 10. Fused MoE kernels

MoE 算子链路：

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

如果开启 Expert Parallel：

```text
EP all2all dispatch
  → local expert compute
  → EP all2all combine
```

MoE 性能瓶颈包括：

```text
router top-k
token reorder
expert token imbalance
all2all communication
shared expert
quantized expert kernel
small expert batch kernel launch overhead
```

## 11. Norm / activation / RoPE kernels

常见基础算子：

```text
RMSNorm / LayerNorm
fused residual RMSNorm
SiluMul / GeluAndMul
RoPE / M-RoPE
embedding helper
activation fused path
```

它们单个不如 attention / GEMM 显眼，但调用频率高。

典型 transformer block：

```text
norm
  → QKV projection
  → RoPE
  → attention
  → MLP activation
  → next block
```

排查重点：

```text
dtype 是否支持
hidden size 是否满足 kernel
input 是否 contiguous
是否 fallback 到 torch op
是否被 CUDA graph capture
M-RoPE / multimodal positions 是否正确
```

## 12. Sampling / logits kernels

输出侧算子链路：

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

spec decode 下：

```text
draft tokens / draft probs
  → target logits
  → RejectionSampler
  → accepted / recovered / bonus tokens
```

容易变慢的功能：

```text
large top-p sort
max_num_logprobs
prompt_logprobs
structured output grammar bitmask
per-request generators
spec decode rejection sampling
GPU 到 CPU copy 和 Python list 转换
```

## 13. Backend selection / fallback

通用选择链路：

```text
operator request
  → user config / env flags
  → platform detection
  → dtype / shape / capability validation
  → dependency import check
  → selected backend
  → fallback backend or error
```

attention backend：

```text
get_attn_backend()
  → AttentionSelectorConfig
  → current_platform.get_attn_backend_cls()
  → candidate priority list
  → validate_configuration()
  → selected AttentionBackend class
```

quantization backend：

```text
quantization string
  → QuantizationConfig
  → layer quant_method
  → backend-specific apply()
```

重要点：

```text
fallback 不一定是错误，但如果 CUDA 上大量 fallback 到 torch op，通常意味着性能问题。
```

## 14. CUDA Graph / torch compile 对算子的影响

CUDA graph 关注：

```text
shape 稳定
tensor 地址稳定
kernel launch 序列稳定
capture 中不能动态分配或非法同步
backend 必须支持对应 cudagraph mode
```

torch compile 关注：

```text
Dynamo graph break
dynamic shape specialization
unsupported op
fullgraph trace
fake impl / schema
compile_sizes 和 cudagraph padding 是否一致
```

算子选择、padding、static buffer、attention metadata 和 fallback 都会影响 compile / graph capture。

## 15. Parallelism 对算子的影响

并行策略把全局 layer 拆成：

```text
rank-local compute + collective communication
```

不同并行维度：

```text
TP：切 linear、attention heads、vocab，需要 all-reduce / all-gather。
PP：切 layer，stage 间 send / recv IntermediateTensors。
DP：复制模型，跨 rank 协调 batch padding / cudagraph mode。
EP：切 experts，MoE token all2all dispatch / combine。
CP / DCP：切 context，attention backend 需要 LSE merge。
```

一句话：

```text
并行会改变每个 rank 的局部 shape、通信 op 和可用 backend。
```

## 16. 一次 forward 的算子视角

可以背成 14 步：

```text
1. Scheduler 产出 SchedulerOutput。
2. ModelRunner 更新 input batch。
3. KV manager 提供 block 分配结果。
4. ModelRunner 构造 slot mapping、block table、attention metadata。
5. CudagraphDispatcher 决定 runtime mode。
6. set_forward_context 写入 metadata。
7. embedding / RoPE / norm / attention / linear / MoE 依次执行。
8. 每层调用 operator wrapper 和 backend kernel。
9. attention kernel 读写 KV cache。
10. TP / EP / PP / DP / CP 插入通信 op。
11. last rank 计算 logits。
12. sample_tokens 应用 grammar、penalty、top-k/top-p、logprobs 或 rejection sampler。
13. GPU tensor 转成 ModelRunnerOutput 需要的结构。
14. Scheduler.update_from_output 消费 sampled tokens。
```

## 17. backend 与 runtime 的关系

实际执行路径由这些因素共同决定：

```text
平台：CUDA / ROCm / CPU / XPU
硬件：compute capability / GPU architecture
dtype：fp16 / bf16 / fp8 / int8 / int4
shape：head size / hidden size / block size / batch tokens
模型结构：dense / MoE / MLA / MQA / GQA / sliding window
配置：attention backend / quantization / cudagraph / compile
并行：TP / PP / DP / EP / CP
依赖：FlashAttention / FlashInfer / Triton / CUTLASS / aiter
运行时功能：LoRA / spec decode / structured output / logprobs / KV connector
```

所以算子问题通常不能只看一个 kernel。

要同时问：

```text
当前 rank-local shape 是什么？
当前 backend 是什么？
metadata 是否和 tensor shape 对齐？
parallel group 是否要求通信？
cudagraph / compile 是否改变了 padding 和执行路径？
```

## 18. 常见排查入口

### attention 慢或报错

看：

```text
attention backend 日志
AttentionSelectorConfig
head_size / dtype / kv_cache_dtype / block_size
slot mapping / block table
KV cache layout
cudagraph runtime mode
profiler 中 flash_attn / flashinfer / triton kernel
```

### quantized linear 报错或慢

看：

```text
QuantizationConfig
linear quant_method
qweight / scale / zero point shape
TP shard offset
packed_dim / packed_factor
Marlin / CUTLASS / scaled_mm 是否出现
是否 fallback 到 aten::mm
```

### MoE 性能异常

看：

```text
router top-k
expert token count
EP all2all dispatch / combine
fused_moe kernel
expert quantization
shared expert
load imbalance
```

### sampling 慢

看：

```text
top-p 是否排序
是否请求 logprobs / prompt_logprobs
是否有 grammar bitmask
是否有 per-request generators
是否开启 spec decode rejection sampler
D2H copy 是否耗时
```

### CUDA graph 没命中

看：

```text
CUDAGraphStat.runtime_mode
num_tokens / num_tokens_padded
batch descriptor 是否命中 capture key
DP rank 是否同步降级
attention backend 是否支持 full / piecewise
LoRA / encoder / cascade / KV scale 是否导致 fallback
```

## 19. 容易混淆的点

### Layer 不等于 kernel

```text
Layer 是模型结构；kernel 是硬件执行单元。
```

### Backend 不等于 wrapper

```text
wrapper 是 Python 调用入口；backend 是具体实现族。
```

### fallback 不一定错

```text
fallback 可以保证兼容，但可能影响性能。
```

### 算子问题不只看算子

```text
metadata、shape、dtype、parallel、compile、runtime feature 都可能决定算子路径。
```

## 20. 与其他专题的关系

```text
executor_worker_model_runner：ModelRunner 决定算子输入和调用时机。
attention：解释 attention metadata 和 backend 语义。
parallelism：解释通信 op 和 rank-local shape。
quantization：解释 quant_method 和量化 kernel。
sampling_and_output：解释 sampling / logprobs kernels 的输出语义。
compilation_and_cuda_graph：解释算子如何被 compile / capture / replay。
kv_cache_transfer：KV cache load/save 影响 attention 和 cache kernels。
```

## 21. 背诵总结

背这一段：

```text
vLLM 的 operators / kernels 是模型执行的底层执行面。Scheduler 决定本轮跑哪些 token，ModelRunner 准备 input、positions、slot mapping、block table、attention metadata、sampling metadata，model layer 表达数学计算，operator wrapper 把调用转成 backend op，backend 最终选择 CUDA、Triton、FlashAttention、FlashInfer、CUTLASS、NCCL 或 torch kernel 执行。算子族包括 attention、KV cache、quantized linear、fused MoE、norm/activation/RoPE、sampling/logprobs 和通信。实际路径由平台、dtype、shape、模型结构、并行、量化、LoRA、spec decode、structured output、CUDA graph 和依赖可用性共同决定。
```
