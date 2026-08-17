# 01. vLLM 里的算子专题应该回答什么？

源码位置：

- `code/vllm/vllm/_custom_ops.py`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/v1/attention/ops/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/platforms/`
- `code/vllm/vllm/utils/torch_utils.py`
- `code/vllm/csrc/`

这个问题关注：vLLM 语境里的 operator / kernel 到底指什么、处在系统哪一层、和 ModelRunner / model layer / backend 的边界是什么，以及为什么需要把它作为独立专题梳理。

---

## 1. 一句话回答

vLLM 的算子层是 **把模型层抽象计算落到具体硬件 backend 执行的底层执行面**。

它回答的问题不是“请求怎么调度”，而是：

```text
这一层计算最后由哪个 kernel 跑？
它需要什么 tensor / metadata？
它为什么走 CUDA / Triton / FlashAttention / FlashInfer / torch fallback？
它如何影响吞吐、延迟、显存和 CUDA Graph？
```

最小心智模型是：

```text
Scheduler 决定跑哪些 token；
ModelRunner 准备输入、batch 状态、KV cache mapping 和 attention metadata；
model layer 表达数学计算；
operator / kernel 负责真正执行这段计算。
```

---

## 2. operator / kernel / backend / layer 的边界

这几个词在 vLLM 源码里经常交错出现，但它们关注层级不同。

### 2.1 Layer

`Layer` 是模型结构层面的抽象，例如：

```text
- Attention layer
- Linear layer
- RMSNorm / LayerNorm
- RoPE
- activation
- fused MoE
- logits / sampler 相关层
```

典型源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:178`
- `code/vllm/vllm/model_executor/layers/activation.py`
- `code/vllm/vllm/model_executor/layers/layernorm.py`
- `code/vllm/vllm/model_executor/layers/rotary_embedding/base.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`

Layer 负责表达“模型要算什么”。例如 `Attention` 的类注释直接说明它会：

```text
1. 把输入 key / value 存入 KV cache；
2. 执行 multi-head / multi-query / grouped-query attention；
3. 返回 output tensor。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:178`

### 2.2 Operator wrapper

`operator wrapper` 是 Python 层的薄封装，负责把 layer 里的调用转换成具体 op 调用。

典型源码：

- `code/vllm/vllm/_custom_ops.py:114`
- `code/vllm/vllm/_custom_ops.py:289`
- `code/vllm/vllm/_custom_ops.py:317`
- `code/vllm/vllm/_custom_ops.py:548`
- `code/vllm/vllm/_custom_ops.py:813`
- `code/vllm/vllm/_custom_ops.py:2579`

例如：

```text
_custom_ops.paged_attention_v1()
  → torch.ops._C.paged_attention_v1(...)

_custom_ops.rms_norm()
  → torch.ops._C.rms_norm(...)

_custom_ops.awq_gemm()
  → torch.ops._C.awq_gemm(...)
  或 env 控制下走 Triton AWQ
```

这层通常不实现数学计算，只负责：

```text
- 参数顺序整理；
- 输出 tensor 分配；
- dtype / shape 分支；
- 平台判断；
- Triton / CUDA / torch fallback 选择；
- fake impl / torch.compile shape 推断支持。
```

### 2.3 Backend

`backend` 是某类算子的实现族选择，尤其在 attention 中最明显。

典型源码：

- `code/vllm/vllm/v1/attention/backend.py:55`
- `code/vllm/vllm/v1/attention/backends/registry.py:34`
- `code/vllm/vllm/v1/attention/selector.py:54`
- `code/vllm/vllm/platforms/cuda.py:351`

Attention backend 会声明：

```text
- 支持哪些 dtype；
- 支持哪些 KV cache dtype；
- 支持哪些 head size / block size；
- 是否支持 MLA / sparse / sink / mm_prefix / non-causal；
- KV cache shape 和 stride order；
- metadata builder；
- impl class。
```

也就是说，backend 负责回答：

```text
当前模型配置 + 当前硬件 + 当前 dtype / head_size / block_size 下，应该用哪个实现族？
```

### 2.4 Kernel

`kernel` 是真正跑在硬件上的底层实现。

典型来源包括：

```text
- C++ / CUDA extension：torch.ops._C / torch.ops._moe_C / torch.ops._rocm_C
- Triton kernel：@triton.jit
- FlashAttention / vllm_flash_attn
- FlashInfer wrapper
- CUTLASS / Marlin / Machete 等量化 GEMM
- torch fallback
- CPU oneDNN / ACL / torch 实现
```

典型源码：

- `code/vllm/csrc/torch_bindings.cpp:21`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:10`
- `code/vllm/csrc/libtorch_stable/attention/paged_attention_v1.cu:164`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`
- `code/vllm/vllm/v1/attention/ops/triton_merge_attn_states.py:58`

---

## 3. 算子层在系统里的位置

从请求执行链路看，算子层处在 ModelRunner / model layer 之后。

主链路可以写成：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → ModelRunner.execute_model()
  → _prepare_inputs()
  → _build_attention_metadata()
  → model forward
  → model layer
  → operator wrapper / backend impl
  → CUDA / Triton / FlashAttention / FlashInfer / torch kernel
  → output tensor
  → logits / sampler / ModelRunnerOutput
```

算子层不直接处理 request，也不决定调度；它只处理已经被 ModelRunner 整理好的张量和 metadata。

一句话：

```text
ModelRunner 把“请求状态”变成“张量状态”，算子层把“张量状态”变成“硬件执行”。
```

---

## 4. 算子层负责什么

vLLM 的算子层主要负责以下几类计算。

### 4.1 Attention

源码入口：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:178`
- `code/vllm/vllm/v1/attention/selector.py:54`
- `code/vllm/vllm/v1/attention/backends/registry.py:34`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:68`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:248`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:325`

它处理：

```text
- prefill attention；
- decode attention；
- mixed prefill / decode batch；
- paged KV cache；
- block table；
- slot mapping；
- cascade attention；
- sliding window / local attention；
- MLA / sparse MLA；
- DCP / CP 相关 attention 结果合并。
```

### 4.2 KV cache 操作

源码入口：

- `code/vllm/vllm/_custom_ops.py:2579`
- `code/vllm/vllm/_custom_ops.py:2601`
- `code/vllm/vllm/_custom_ops.py:2690`
- `code/vllm/vllm/_custom_ops.py:2729`
- `code/vllm/vllm/v1/attention/ops/paged_attn.py:15`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`

它处理：

```text
- key / value 写入 paged cache；
- slot_mapping 到物理 cache slot 的映射；
- cache block swap / copy；
- MLA KV cache concat；
- FP8 / per-token-head / NVFP4 KV cache layout；
- DCP / CP 下的 cache gather。
```

### 4.3 Quantization

源码入口：

- `code/vllm/vllm/_custom_ops.py:548`
- `code/vllm/vllm/_custom_ops.py:615`
- `code/vllm/vllm/_custom_ops.py:813`
- `code/vllm/vllm/_custom_ops.py:1210`
- `code/vllm/vllm/_custom_ops.py:1402`
- `code/vllm/vllm/_custom_ops.py:1650`
- `code/vllm/vllm/model_executor/layers/quantization/`

它处理：

```text
- AWQ / GPTQ；
- FP8 / INT8 / FP4 / MXFP4 / NVFP4；
- CUTLASS scaled mm；
- Marlin / Machete；
- online quantization；
- scale / zero point / group size；
- quantized linear method 替代普通 linear。
```

### 4.4 Fused MoE

源码入口：

- `code/vllm/vllm/_custom_ops.py:904`
- `code/vllm/vllm/_custom_ops.py:1026`
- `code/vllm/vllm/_custom_ops.py:2228`
- `code/vllm/vllm/_custom_ops.py:2232`
- `code/vllm/vllm/_custom_ops.py:2302`
- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/config.py`

它处理：

```text
- router top-k；
- token 按 expert 重排；
- grouped GEMM；
- expert output combine；
- MoE quantization；
- expert parallel / EPLB 影响下的 expert layout。
```

### 4.5 Norm / activation / RoPE

源码入口：

- `code/vllm/vllm/_custom_ops.py:289`
- `code/vllm/vllm/_custom_ops.py:317`
- `code/vllm/vllm/_custom_ops.py:326`
- `code/vllm/vllm/model_executor/layers/activation.py:97`
- `code/vllm/vllm/model_executor/layers/activation.py:133`
- `code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:238`
- `code/vllm/vllm/model_executor/layers/layernorm.py`

它处理：

```text
- rotary embedding；
- RMSNorm；
- fused add RMSNorm；
- SiluMul / GeluAndMul；
- QK norm + RoPE fusion；
- quantized norm / activation。
```

### 4.6 Logits / sampling

源码入口：

- `code/vllm/vllm/_custom_ops.py:366`
- `code/vllm/vllm/_custom_ops.py:382`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`

它处理：

```text
- repetition penalties；
- top-k / top-p；
- logprobs；
- rejection sampling；
- speculative decoding 采样侧辅助计算。
```

---

## 5. 算子层不负责什么

算子层不负责：

```text
- 接收用户请求；
- 排队和调度请求；
- 决定 token budget；
- 维护 request lifecycle；
- 决定哪些请求进入本轮 batch；
- 管理 Scheduler 的 KV block 分配策略；
- 构造 OpenAI API 层的最终响应。
```

这些分别属于：

```text
Engine / API Server：
  接收请求、输出响应。

Scheduler：
  调度 request、分配 token budget、管理 KV block 生命周期。

Executor / Worker / ModelRunner：
  把 SchedulerOutput 变成模型执行，并维护设备侧 batch 状态。

OutputProcessor / Scheduler.update_from_output：
  消化 token 输出、stop condition、释放资源。
```

---

## 6. 为什么需要独立算子专题

已有专题通常解释系统层的控制流：

```text
Scheduler：本轮跑哪些 token？
Executor / Worker / ModelRunner：调度结果如何进入模型？
Attention：metadata 如何构造？
Quantization：权重和 layer method 如何替换？
Compilation / CUDA Graph：如何 capture 和 compile？
Parallelism：TP / PP / DP / EP 如何组织？
```

算子专题补的是底层执行视角：

```text
这一层计算实际调用了什么 kernel？
输入 tensor 的 shape / dtype / layout 是什么？
为什么这个 backend 可用或不可用？
什么时候 fallback？
某个性能瓶颈应该看哪类 kernel？
```

因此算子专题适合回答：

```text
- 为什么某个模型在 A100 上走 FlashAttention，在另一个平台走 Triton？
- 为什么手动设置 block_size 会让高优先级 backend 失效？
- 为什么 FP8 KV cache 会改变 attention backend 或 cache layout？
- 为什么 CUDA Graph 要把 attention 包成 opaque custom op？
- 为什么某些 op 有 fake impl？
- 为什么 MoE / quantization 的性能瓶颈不是 Python，而是 grouped GEMM / memory layout？
```

---

## 7. 算子调用的几种形态

vLLM 里常见的算子调用形态有 5 类。

### 7.1 `_custom_ops.py` 里的 torch.ops wrapper

典型形式：

```python
from vllm import _custom_ops as ops
ops.rms_norm(...)
ops.reshape_and_cache(...)
ops.cutlass_scaled_mm(...)
```

对应源码：

- `code/vllm/vllm/_custom_ops.py:114`
- `code/vllm/vllm/_custom_ops.py:317`
- `code/vllm/vllm/_custom_ops.py:813`
- `code/vllm/vllm/_custom_ops.py:2579`

这类 wrapper 大多最终调用：

```text
torch.ops._C.*
torch.ops._rocm_C.*
torch.ops._moe_C.*
torch.ops._C_cache_ops.*
```

### 7.2 `torch.ops.vllm.*` Python custom op

典型形式：

```python
torch.ops.vllm.unified_attention_with_output(...)
torch.ops.vllm.unified_kv_cache_update(...)
torch.ops.vllm.maybe_calc_kv_scales(...)
```

对应源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:458`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:519`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:522`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:641`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:726`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:779`

注册工具：

- `code/vllm/vllm/utils/torch_utils.py:927`
- `code/vllm/vllm/utils/torch_utils.py:931`

这类 op 不是 C++ extension，而是把 Python 函数注册为 torch custom op，用于：

```text
- 降低 torch.library.custom_op 的调度开销；
- 给 torch.compile / CUDA Graph 提供稳定边界；
- 把 attention 作为 opaque op 处理；
- 搭配 fake_impl 支持编译期 shape 推断。
```

### 7.3 Triton Python kernel

典型源码：

- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`
- `code/vllm/vllm/v1/attention/ops/triton_merge_attn_states.py:58`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`

这类路径通常由 Python wrapper 直接 launch Triton kernel。

### 7.4 第三方 kernel wrapper

典型来源：

```text
- vllm_flash_attn
- FlashInfer
- AITER
- flash_attn upstream
- zentorch / oneDNN / ACL
```

典型源码：

- `code/vllm/vllm/v1/attention/backends/fa_utils.py:18`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:20`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:11`
- `code/vllm/vllm/v1/attention/backends/rocm_aiter_fa.py`
- `code/vllm/vllm/model_executor/layers/utils.py:255`

### 7.5 torch fallback

典型源码：

- `code/vllm/vllm/_custom_ops.py:366`
- `code/vllm/vllm/_custom_ops.py:393`
- `code/vllm/vllm/_custom_ops.py:407`
- `code/vllm/vllm/_custom_ops.py:412`

例如 repetition penalty：

```text
如果 logits 是 CUDA 且 contiguous：
  → apply_repetition_penalties_cuda()
  → torch.ops._C.apply_repetition_penalties_()
否则：
  → apply_repetition_penalties_torch()
```

这类 fallback 通常牺牲性能，但能提升兼容性或支持 CPU / 非 contiguous / 特殊 dtype。

---

## 8. 算子层和 ModelRunner 的关系

ModelRunner 不直接关心每个 kernel 的实现细节，但它会准备 kernel 所需的关键输入。

典型输入包括：

```text
- input_ids / inputs_embeds；
- positions；
- query / key / value；
- slot_mapping；
- block_table；
- seq_lens；
- query_start_loc；
- logits_indices；
- attention metadata；
- kv cache tensor；
- sampling metadata；
- LoRA / quantization / multimodal / spec decode 相关状态。
```

Attention 层的 forward 注释明确说明：

```text
Attention metadata 是由 ModelRunner.execute_model() 的 context manager 设置的，
forward 时通过 get_forward_context().attn_metadata 读取。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:448`

因此可以把责任边界记成：

```text
ModelRunner：准备正确的张量和 metadata；
Layer：表达要执行哪类计算；
Operator / backend：选择并执行具体 kernel。
```

---

## 9. 算子层和 CUDA Graph / torch.compile 的关系

算子层会直接影响 CUDA Graph / torch.compile 是否稳定。

关键点包括：

```text
- op 的 shape 是否稳定；
- 是否有 CPU sync；
- 是否有 Python control flow；
- 是否需要临时 workspace；
- 是否能提供 fake impl；
- 是否能作为 opaque custom op 被 capture；
- 是否在不同 batch 状态下切换 kernel。
```

Attention 层对此有专门处理：

```text
CUDA-like 和 CPU 平台上，vLLM 会把 attention 注册成一个大的 opaque custom op；
其他平台则可能 direct call，让 torch.compile 自己处理。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:390`

相关注册入口：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:779`
- `code/vllm/vllm/utils/torch_utils.py:931`

---

## 10. 算子层和 backend selection 的关系

算子不是单一实现，而是根据平台、配置、dtype、shape 选择 backend。

以 attention 为例：

```text
Attention.__init__()
  → get_attn_backend(...)
  → AttentionSelectorConfig
  → current_platform.get_attn_backend_cls(...)
  → backend.validate_configuration(...)
  → 选择最高优先级可用 backend
  → impl_cls = backend.get_impl_cls()
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:303`
- `code/vllm/vllm/v1/attention/selector.py:54`
- `code/vllm/vllm/v1/attention/selector.py:90`
- `code/vllm/vllm/v1/attention/selector.py:121`
- `code/vllm/vllm/platforms/cuda.py:351`
- `code/vllm/vllm/platforms/cuda.py:405`

selection 的影响因素包括：

```text
- head_size；
- model dtype；
- kv_cache_dtype；
- block_size；
- MLA / sparse / sink / mm_prefix；
- attention type；
- non-causal；
- batch invariance；
- KV connector；
- compute capability；
- 用户显式指定的 attention_backend；
- platform backend priority。
```

---

## 11. 阅读算子源码时先抓什么

建议按以下顺序读，而不是直接钻进 CUDA kernel：

```text
1. 从 model layer 的 forward 看入口；
2. 找到调用的是 _custom_ops、torch.ops.vllm、Triton wrapper 还是第三方 wrapper；
3. 看 wrapper 有没有 dtype / shape / platform fallback；
4. 看 backend selection 条件；
5. 看 metadata / tensor layout；
6. 最后再看 csrc / Triton kernel。
```

以 attention 为例：

```text
Attention.forward()
  → torch.ops.vllm.unified_attention_with_output()
  → get_attention_context()
  → self.impl.forward()
  → FlashAttentionImpl / TritonAttentionImpl / FlashInferImpl / MLA impl
  → flash_attn_varlen_func / unified_attention Triton / FlashInfer wrapper / native op
```

---

## 12. 常见问题定位方向

算子问题通常表现为：

```text
- backend 选错或 fallback 到慢路径；
- 当前 dtype / head_size / block_size 不支持；
- KV cache layout 不匹配；
- CUDA Graph capture 失败；
- torch.compile 图断裂；
- illegal memory access；
- NaN / Inf；
- output mismatch；
- profiler 中某个 kernel 时间异常；
- kernel launch 数过多；
- CPU sync 或 Python overhead 过高。
```

定位时先问：

```text
1. 这个计算属于哪类算子？attention / KV cache / quant / MoE / norm / sampling？
2. 实际 backend 是谁？
3. 输入 tensor 的 shape / dtype / stride / layout 是什么？
4. 是否启用了 CUDA Graph / torch.compile？
5. 是否存在 fallback 条件？
6. 是否受平台能力或环境变量影响？
```

---

## 13. 一句话总结

vLLM 的算子层负责把 ModelRunner 和 model layer 产生的抽象张量计算，落到 CUDA、Triton、FlashAttention、FlashInfer、CUTLASS、量化、MoE、CPU fallback 等具体执行路径上；它是性能、兼容性、CUDA Graph、torch.compile、并行和硬件能力最终交汇的底层执行面。

最核心主线是：

```text
ModelRunner 准备状态
  → model layer forward
  → Python op wrapper / backend impl
  → concrete kernel
  → tensor output
  → logits / sampler / next layer / cache state
```
