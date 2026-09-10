# 07. RMSNorm、activation、RoPE 等基础算子在哪里用？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\custom_op.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\layernorm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\activation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\rotary_embedding\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\fused_moe\activation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\batch_invariant.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\llama.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\qwen2_vl.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\qwen2_5_vl.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\qwen3_vl.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\_custom_ops.py`
- `D:\lzy\project\kv_pool\code\vllm\csrc\`

本问题关注：RMSNorm、LayerNorm、SiluAndMul、GeluAndMul、SwiGLU 变体、RoPE、M-RoPE、XD-RoPE 等基础算子，如何嵌入 Transformer block 的 attention / MLP 主链路；它们如何通过 `CustomOp` 分发到 CUDA/HIP/XPU/CPU/native backend；以及这些算子和 residual、dtype、positions、CUDA Graph、torch.compile、多模态位置编码之间的边界。

---

## 1. 一句话回答

RMSNorm、activation、RoPE 这些基础算子位于模型 layer 内部，是 attention 和 MLP 两条主干前后的高频张量变换：

```text
Transformer block
  → input RMSNorm / fused add RMSNorm
  → qkv projection
  → RoPE / M-RoPE 修改 q/k
  → attention
  → post-attention RMSNorm
  → gate_up projection
  → fused activation: SiluAndMul / GeluAndMul / SwiGLU variant
  → down projection
  → residual / next block
```

在 vLLM 里，它们一般不是由 ModelRunner 直接调用，而是模型结构文件里实例化成 `CustomOp` module；forward 时 `CustomOp` 根据当前平台和编译配置选择：

```text
forward_native / forward_cuda / forward_hip / forward_xpu / forward_cpu
```

所以可以把基础算子的主链路概括为：

```text
model architecture
  → layernorm / activation / rotary module
  → CustomOp.dispatch_forward()
  → torch.ops._C / vllm._custom_ops / Triton / flash-attn / native torch
  → transformed tensor
```

---

## 2. 先给结论：这几类算子分别在哪里用

### 2.1 Norm：block 边界和 residual 路径

典型位置：

```text
input_layernorm
post_attention_layernorm
final norm
```

以 LLaMA 为例：

- `input_layernorm = RMSNorm(...)`
- `post_attention_layernorm = RMSNorm(...)`
- `self.norm = RMSNorm(...)`

位置：`code/vllm/vllm/model_executor/models/llama.py:308`

`RMSNorm.forward()` 支持两种形态：

```text
RMSNorm(x)
RMSNorm(x, residual) → fused add + rms norm
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:74`

### 2.2 Activation：MLP gate/up 之后

以 LLaMA MLP 为例：

```text
gate_up = gate_up_proj(x)
activated = SiluAndMul(gate_up)
out = down_proj(activated)
```

`SiluAndMul` 的语义是：

```text
silu(x[..., :d]) * x[..., d:]
```

位置：`code/vllm/vllm/model_executor/layers/activation.py:116`

常见 activation module 包括：

- `SiluAndMul`
- `GeluAndMul`
- `SiluAndMulWithClamp`
- `SwigluOAIAndMul`
- `SwigluStepAndMul`
- `MulAndSilu`
- `GELU` / `NewGELU` / `FastGELU` / `QuickGELU`

### 2.3 RoPE：attention 里 q/k projection 之后

以 LLaMA attention 为例：

```text
qkv = qkv_proj(hidden_states)
q, k, v = split(qkv)
q, k = rotary_emb(positions, q, k)
attn_output = attention(q, k, v)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:231`

RoPE module 一般由 `get_rope()` 创建。

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/__init__.py:33`

### 2.4 M-RoPE / XD-RoPE：多模态或扩展位置编码路径

普通 RoPE 的 positions 通常是：

```text
[num_tokens]
```

M-RoPE 的 positions 可能是：

```text
[3, num_tokens]
```

代表多模态 T/H/W 三个位置维度。

M-RoPE 实现在：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:201`

XD-RoPE 实现在：`code/vllm/vllm/model_executor/layers/rotary_embedding/xdrope.py`

---

## 3. 整体流程图

可以把这些基础算子放进 decoder block 里看：

```text
GPUModelRunner._model_forward()
  → model.forward(input_ids / inputs_embeds, positions, ...)
      → decoder layer
          → RMSNorm(hidden_states, residual?)
          → qkv projection
          → get q / k / v
          → RotaryEmbedding(positions, q, k)
              → cos_sin_cache index_select
              → ops.rotary_embedding 或 native apply_rotary
          → Attention(q, k, v)
          → residual add
          → RMSNorm(...)
          → gate_up_proj
          → SiluAndMul / GeluAndMul / SwiGLU variant
          → down_proj
          → residual add
      → final RMSNorm
      → logits / sampler
```

底层调度可以再展开为：

```text
RMSNorm / SiluAndMul / RotaryEmbedding
  → CustomOp.forward()
  → _forward_method
      → forward_cuda / forward_hip / forward_xpu / forward_cpu
      → forward_native fallback
  → torch.ops._C / vllm._custom_ops / Triton / flash-attn / native torch
```

---

## 4. `CustomOp` 是基础算子的统一调度层

位置：`code/vllm/vllm/model_executor/custom_op.py:103`

### 4.1 `CustomOp` 的角色

`CustomOp` 是 vLLM 对基础算子的统一包装：

```text
模型代码只调用 module.forward()
CustomOp 决定真实使用哪个 backend 实现
```

核心逻辑：

```python
def forward(self, *args, **kwargs):
    return self._forward_method(*args, **kwargs)
```

位置：`code/vllm/vllm/model_executor/custom_op.py:135`

`_forward_method` 在初始化时由 `dispatch_forward()` 决定。

位置：`code/vllm/vllm/model_executor/custom_op.py:174`

### 4.2 分发规则

如果 custom op 启用：

```text
ROCm → forward_hip
CPU  → forward_cpu
TPU  → forward_tpu
XPU  → forward_xpu
OOT  → forward_oot
其他 CUDA-like → forward_cuda
```

如果 custom op 没启用：

```text
forward_native
```

这意味着同一个模型结构可以在不同平台上使用不同实现，而模型 forward 主体不需要写一堆 if/else。

### 4.3 native fallback 不只是兜底

`forward_native()` 有三类用途：

```text
1. 没有专用 kernel 时作为 fallback；
2. 单元测试 / 数值对比；
3. 某些 torch.compile 或 opaque custom op 内部路径可被编译。
```

所以文档中看到 `forward_native` 不代表它一定慢或一定只用于 CPU，它也是 vLLM 保持算子语义清晰的重要参考实现。

---

## 5. RMSNorm 的执行路径

### 5.1 RMSNorm 的语义

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:36`

RMSNorm 计算：

```text
x → weight * x / sqrt(mean(x^2) + eps)
```

和 LayerNorm 相比，RMSNorm 不减均值，只按 RMS 缩放。

### 5.2 参数和配置

`RMSNorm.__init__()` 保存：

```text
hidden_size
variance_epsilon
variance_size_override
weight
has_weight
pass_weight
pass_weight_add
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:46`

`variance_size_override` 用于某些模型的特殊 norm 维度，`has_weight=False` 则支持无权重 RMSNorm。

### 5.3 普通 RMSNorm 路径

如果没有 residual：

```python
ir.ops.rms_norm(x, weight, eps, variance_size_override)
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:80`

底层 `_custom_ops` 中也有 native op wrapper：

```text
ops.rms_norm(out, input, weight, epsilon)
```

位置：`code/vllm/vllm/_custom_ops.py:317`

### 5.4 fused add RMSNorm 路径

如果传入 residual：

```python
ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, eps, variance_size_override)
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:88`

底层 wrapper：

```text
ops.fused_add_rms_norm(input, residual, weight, epsilon)
```

位置：`code/vllm/vllm/_custom_ops.py:326`

它融合的是：

```text
x = x + residual
residual = x
x = RMSNorm(x)
```

这样可以减少一次读写和一个额外 kernel。

### 5.5 batch invariant 路径

如果设置了 `VLLM_BATCH_INVARIANT`，CUDA 路径会走：

```python
rms_norm_batch_invariant(...)
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:101`

这个路径用于减少 batch 形态变化对某些场景的影响，但要求不能使用 `variance_size_override`。

### 5.6 GemmaRMSNorm

Gemma 的 RMSNorm 和普通 RMSNorm 有差异：

```text
普通 RMSNorm: x * weight
GemmaRMSNorm: x * (1 + weight)
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:128`

这类模型差异被封装在专门的 layer 里，上层 decoder block 仍然只是调用 norm module。

### 5.7 RMSNormGated

`RMSNormGated` 支持：

```text
标准 RMSNorm
Group RMSNorm
可选 gate: out = norm(x) * act(z) 或 norm(x * act(z))
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:168`

CUDA 路径会调用 FLA 相关实现：

```text
vllm.model_executor.layers.fla.ops.layernorm_guard.rmsnorm_fn
```

---

## 6. LayerNorm 在哪里

`LayerNorm` 也在 `layernorm.py`，但实现更接近标准 PyTorch：

```python
F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)
```

位置：`code/vllm/vllm/model_executor/layers/layernorm.py:305`

它常用于非 LLaMA 类结构、vision tower、encoder 或一些模型特殊子模块。和 RMSNorm 相比，LayerNorm 的性能优化不是 vLLM decoder block 的主要热点，但仍然是基础算子体系的一部分。

---

## 7. activation fused 算子的执行路径

### 7.1 为什么 gated activation 要 fused

普通 gated MLP 通常是：

```text
gate, up = gate_up_proj(x).chunk(2, dim=-1)
out = activation(gate) * up
```

如果拆成 PyTorch op，会产生：

```text
slice / chunk
activation kernel
mul kernel
临时 tensor 写回显存
```

vLLM 的 fused activation 把这些合在一个 op 里：

```text
输入 [num_tokens, 2 * intermediate]
输出 [num_tokens, intermediate]
```

### 7.2 SiluAndMul

位置：`code/vllm/vllm/model_executor/layers/activation.py:116`

语义：

```text
silu(x[..., :d]) * x[..., d:]
```

CUDA/XPU 路径：

```text
torch.ops._C.silu_and_mul(out, x)
```

位置：`code/vllm/vllm/model_executor/layers/activation.py:143`

LLaMA MLP 里默认使用它。

位置：`code/vllm/vllm/model_executor/models/llama.py:116`

### 7.3 GeluAndMul

位置：`code/vllm/vllm/model_executor/layers/activation.py:338`

语义：

```text
gelU(x[..., :d]) * x[..., d:]
```

它支持：

```text
approximate="none"
approximate="tanh"
```

CUDA/CPU/XPU 路径会使用：

```text
torch.ops._C.gelu_and_mul
torch.ops._C.gelu_tanh_and_mul
```

位置：`code/vllm/vllm/model_executor/layers/activation.py:361`

### 7.4 MulAndSilu

位置：`code/vllm/vllm/model_executor/layers/activation.py:214`

语义和 `SiluAndMul` 的左右顺序相反：

```text
x[..., :d] * silu(x[..., d:])
```

这类差异通常来自模型权重中 gate/up 的排列方式。

### 7.5 clamp / SwiGLU-OAI 变体

`SiluAndMulWithClamp`：`code/vllm/vllm/model_executor/layers/activation.py:154`

语义：

```text
gate = clamp(gate, max=limit)
up = clamp(up, min=-limit, max=limit)
out = gate * sigmoid(alpha * gate) * (up + beta)
```

`SwigluOAIAndMul`：`code/vllm/vllm/model_executor/layers/activation.py:396`

用于 GPT-OSS 风格 SwiGLU-OAI。

`SwigluStepAndMul`：`code/vllm/vllm/model_executor/layers/activation.py:428`

CUDA 路径用 Triton kernel：

```text
_swiglustep_and_mul_kernel
```

位置：`code/vllm/vllm/model_executor/layers/activation.py:26`

### 7.6 普通 GELU / FastGELU / QuickGELU

这些算子输出 shape 不减半，用于非 gated 激活：

- `GELU`: `code/vllm/vllm/model_executor/layers/activation.py:310`
- `NewGELU`: `code/vllm/vllm/model_executor/layers/activation.py:466`
- `FastGELU`: `code/vllm/vllm/model_executor/layers/activation.py:494`
- `QuickGELU`: `code/vllm/vllm/model_executor/layers/activation.py:521`

它们通常在 BERT/GPT-NeoX/CLIP/vision encoder 或模型特殊 MLP 中出现。

---

## 8. MoE activation 和普通 MLP activation 的关系

普通 dense MLP 常用 `activation.py` 里的 `SiluAndMul` / `GeluAndMul`。

MoE 的 expert activation 则有一套 MoE 专用入口：

```text
vllm/model_executor/layers/fused_moe/activation.py
```

在 fused MoE functional 路径中：

```python
apply_moe_activation(activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N))
```

位置：`code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1696`

原因是 MoE activation 位于 grouped expert GEMM 中间，输入输出 cache 形状和普通 MLP 不完全一样：

```text
普通 MLP: [tokens, 2 * intermediate] → [tokens, intermediate]
MoE MLP:  [tokens, top_k, 2 * intermediate] → [tokens * top_k, intermediate]
```

---

## 9. RoPE 的构造入口：`get_rope()`

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/__init__.py:33`

### 9.1 `get_rope()` 的输入

关键参数：

```text
head_size
max_position
is_neox_style
rope_parameters
dtype
dual_chunk_attention_config
```

其中 `rope_parameters` 来自模型 config，常见字段有：

```text
rope_theta
rope_type
rope_dim
partial_rotary_factor
factor
original_max_position_embeddings
mrope_section
xdrope_section
```

### 9.2 RoPE 对象会缓存

`get_rope()` 用 `_ROPE_DICT` 按参数缓存 RoPE 对象。

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/__init__.py:30`

原因是多个 layer 的 RoPE 配置通常相同，cos/sin cache 可以复用构造逻辑，避免重复初始化。

### 9.3 支持的 RoPE 类型

`get_rope()` 会根据 `rope_type` 选择不同实现：

```text
default → RotaryEmbedding 或 MRotaryEmbedding
linear → LinearScalingRotaryEmbedding
ntk → NTKScalingRotaryEmbedding
dynamic → DynamicNTKScalingRotaryEmbedding / DynamicNTKAlphaRotaryEmbedding
yarn → YaRNScalingRotaryEmbedding 或 MRotaryEmbedding
llama3 → Llama3RotaryEmbedding
longrope → Phi3LongRoPEScaledRotaryEmbedding
deepseek_yarn / deepseek_llama_scaling → DeepseekScalingRotaryEmbedding
xdrope → XDRotaryEmbedding
openpangu → MRotaryEmbeddingInterleaved
proportional → Gemma4RotaryEmbedding
mllama4 → Llama4VisionRotaryEmbedding
telechat3-yarn → TeleChat3RoPEScaledRotaryEmbedding
```

这些变体的差异主要体现在：

```text
inv_freq 如何算；
cos/sin cache 如何缩放；
positions 如何解释；
rotary_dim 如何分段；
是否适配多模态或超长上下文。
```

---

## 10. 普通 RoPE 的执行路径

### 10.1 初始化 cos/sin cache

`RotaryEmbeddingBase` 会在初始化时计算：

```text
cos_sin_cache: [max_position_embeddings, rotary_dim]
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:58`

计算逻辑：

```text
inv_freq = 1 / base^(arange(0, rotary_dim, 2) / rotary_dim)
freqs = positions × inv_freq
cache = concat(cos(freqs), sin(freqs))
```

位置：

- `code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:80`
- `code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:94`

### 10.2 forward 输入输出

普通 RoPE 输入：

```text
positions: [num_tokens]
query: [num_tokens, num_q_heads * head_size]
key: [num_tokens, num_kv_heads * head_size] 或 None
```

输出：

```text
rotated query, rotated key
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:203`

### 10.3 native 实现

native 路径：

```text
1. positions flatten
2. cos_sin_cache.index_select(0, positions)
3. query/key reshape 成 [num_tokens, heads, head_size]
4. 前 rotary_dim 做旋转
5. rotary_dim 之后 pass-through
6. reshape 回原形状
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:160`

### 10.4 CUDA 实现

CUDA 路径：

```python
ops.rotary_embedding(
    positions,
    query,
    key,
    head_size,
    cos_sin_cache,
    is_neox_style,
)
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:221`

底层 wrapper：`code/vllm/vllm/_custom_ops.py:289`

注意这里是 in-place 操作：

```text
query 和 key 会被原地修改。
```

这也是为什么 attention 里通常写：

```python
q, k = self.rotary_emb(positions, q, k)
```

但底层实际上可能已经修改了 q/k tensor。

### 10.5 Neox-style 和 GPT-J-style

旋转方式由 `is_neox_style` 决定。

在 `ApplyRotaryEmb.forward_static()` 中：

```text
Neox-style:
  x1, x2 = chunk(x, 2)

GPT-J-style:
  x1 = x[..., ::2]
  x2 = x[..., 1::2]
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/common.py:146`

这两个风格必须和模型训练时的位置编码布局一致，否则输出会明显错误。

---

## 11. `ApplyRotaryEmb` 和 flash-attn / flashinfer 路径

`ApplyRotaryEmb` 是把 cos/sin 应用到一个 tensor 的通用 CustomOp。

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/common.py:124`

### 11.1 forward_native

native 版本直接计算：

```text
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/common.py:146`

### 11.2 forward_cuda

CUDA 版本调用：

```text
vllm.vllm_flash_attn.layers.rotary.apply_rotary_emb
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/common.py:229`

### 11.3 HIP 特殊 fallback

ROCm/HIP 下，flash-attn triton rotary kernel 有 grid 维度限制。`forward_hip()` 会检查 grid size，超过限制时回退到 native PyTorch 实现。

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/common.py:252`

### 11.4 flashinfer rotary

`common.py` 里也注册了：

```text
torch.ops.vllm.flashinfer_rotary_embedding
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/common.py:115`

当前 `RotaryEmbeddingBase` 中 flashinfer 路径有注释说明暂时 disabled，但这个 op wrapper 仍保留在代码中。

---

## 12. M-RoPE 如何处理多模态 positions

M-RoPE 入口：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:201`

### 12.1 positions 形状

M-RoPE 支持两种 positions：

```text
[num_tokens]
  文本或普通 decode 路径

[3, num_tokens]
  多模态 T/H/W positions
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:263`

### 12.2 mrope_section

`mrope_section` 描述 rotary_dim 的一半如何分给 T/H/W：

```text
[t_section, h_section, w_section]
```

要求：

```text
sum(mrope_section) == rotary_dim // 2
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:250`

### 12.3 native M-RoPE

native 路径会：

```text
1. cos_sin_cache[positions] 得到 [3, num_tokens, rotary_dim]
2. split 成 T/H/W 三段
3. 按 mrope_section 拼出每个维度对应的 cos/sin
4. 对 query/key 做普通 rotary
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:282`

### 12.4 CUDA M-RoPE

如果 `positions.ndim == 2`，CUDA 路径会调用 Triton kernel：

```text
triton_mrope(...)
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:324`

Triton kernel：

```text
_triton_mrope_forward
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:14`

它按 token 维度启动 program，对每个 token 的 q/k 头应用 T/H/W 混合后的 cos/sin。

### 12.5 interleaved M-RoPE

如果 `mrope_interleaved=True`，M-RoPE 不再按连续 TTT...HHH...WWW... 分段，而是按交错布局选择维度。

相关函数：

```text
apply_interleaved_rope()
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/mrope.py:190`

---

## 13. positions 从哪里来

基础算子文档容易遗漏一点：RoPE 本身不决定 positions，positions 是 ModelRunner 输入准备阶段生成并传入模型 forward 的。

在 v1 GPUModelRunner 中，普通 positions 来自：

```text
positions = num_computed_tokens + query_pos
```

M-RoPE / XD-RoPE 会在输入准备阶段额外构造专用 positions。

然后模型 attention 层只消费：

```text
self.rotary_emb(positions, q, k)
```

因此职责边界是：

```text
ModelRunner：决定每个 token 的 position 是多少。
RotaryEmbedding：决定如何用 position 修改 q/k。
Attention backend：消费已经带位置编码的 q/k。
```

---

## 14. dtype、contiguous 和 shape 对这些算子的影响

### 14.1 RMSNorm

RMSNorm 关注：

```text
hidden_size
variance_epsilon
weight dtype
input dtype
是否有 residual
是否需要 variance_size_override
```

数值上通常会在内部使用更高精度做方差/RMS，再转回原 dtype。

### 14.2 Activation

fused gated activation 要求最后一维能切成两半：

```text
input.shape[-1] == 2 * output.shape[-1]
```

如果模型 gate/up 顺序和 activation module 不匹配，例如该用 `MulAndSilu` 却用了 `SiluAndMul`，输出会错误但 shape 仍可能合法。

### 14.3 RoPE

RoPE 关注：

```text
head_size
rotary_dim
positions dtype / device
cos_sin_cache dtype / device
query/key layout
is_neox_style
```

`RotaryEmbedding._match_cos_sin_cache_dtype()` 会把 cache 移到 query 的 device/dtype。

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:105`

在 torch.compile 期间，它会避免修改 module buffer：

```text
if torch.compiler.is_compiling(): return converted cache, 不写回 self.cos_sin_cache
```

位置：`code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:127`

这对 CUDA Graph / compile 路径很重要，因为 tracing 期间修改 buffer 可能引入不稳定副作用。

---

## 15. CUDA Graph / torch.compile 下的注意点

这些基础算子虽然小，但非常高频，且经常处于 CUDA Graph capture 范围内。

### 15.1 CustomOp 启用状态会进入 compilation config

`CustomOp.dispatch_forward()` 会记录：

```text
enabled_custom_ops
disabled_custom_ops
```

位置：`code/vllm/vllm/model_executor/custom_op.py:185`

这会影响 compile/capture 看到的是 custom op 还是 native graph。

### 15.2 in-place op 要稳定

RoPE CUDA op 是 in-place 修改 q/k。

风险点：

```text
- q/k tensor shape 必须和 capture 时一致；
- stride/layout 不能不符合 kernel 预期；
- positions shape 变化要符合 graph replay 路径；
- cache dtype/device 不能在 replay 中反复重建。
```

### 15.3 fused add RMSNorm 的 residual 语义

`fused_add_rms_norm` 可能会原地更新 input/residual，这要求上层模型对 residual 生命周期有明确约定。

调试时要区分：

```text
RMSNorm(x)
RMSNorm(x, residual)
```

它们不只是多一个参数，后者改变了 residual 路径的读写顺序。

### 15.4 activation 的临时 tensor 更少

fused activation 的收益主要来自减少：

```text
chunk/slice 临时 view 后的 materialization
activation output 临时 tensor
mul kernel launch
显存读写
```

在小 batch 下，kernel launch 占比仍可能较高；在大 batch/大 intermediate 下，内存带宽和融合收益更明显。

---

## 16. 常见问题和排查

### 16.1 RMSNorm 输出异常

优先检查：

```text
- eps 是否和模型 config 一致；
- Gemma 是否使用了 GemmaRMSNorm；
- residual 是否被重复 add 或漏 add；
- has_weight=False 的路径是否符合模型定义；
- dtype 是否导致数值误差扩大。
```

### 16.2 activation 输出异常

优先检查：

```text
- hidden_act 是否映射到正确 module；
- gate/up 顺序是否匹配 SiluAndMul / MulAndSilu；
- approximate="none" / "tanh" 是否匹配 checkpoint；
- clamp limit / alpha / beta 是否来自 config；
- 输入最后一维是否确实是 2 * intermediate。
```

### 16.3 RoPE 后 attention 异常

优先检查：

```text
- positions 是否正确；
- max_position / rope_scaling 是否和 config 匹配；
- is_neox_style 是否匹配模型；
- rotary_dim / partial_rotary_factor 是否正确；
- M-RoPE 的 positions 是否是 [3, num_tokens]；
- mrope_section 之和是否等于 rotary_dim // 2；
- cache 是否超出 max_position_embeddings。
```

### 16.4 compile / CUDA graph 下异常

优先检查：

```text
- CustomOp 是 enabled 还是 disabled；
- 是否走 forward_native 导致 graph 中出现多个 torch ops；
- RoPE 是否在 compile 期间修改 buffer；
- residual fused norm 是否有原地写冲突；
- HIP/ROCm 是否触发 rotary grid fallback。
```

---

## 17. 最终可以记成一张表

| 算子 | 主要文件 | 输入 | 输出 | 典型用途 |
|---|---|---|---|---|
| `RMSNorm` | `layernorm.py` | `x` 或 `x + residual` | normalized x | decoder block 前后 norm |
| `GemmaRMSNorm` | `layernorm.py` | `x` | `x * (1 + w)` norm | Gemma 系模型 |
| `RMSNormGated` | `layernorm.py` | `x`、可选 `z` | gated norm output | FLA / gated norm 模型 |
| `LayerNorm` | `layernorm.py` | `x` | normalized x | encoder / vision / 特殊模型 |
| `SiluAndMul` | `activation.py` | `[tokens, 2d]` | `[tokens, d]` | SwiGLU MLP |
| `GeluAndMul` | `activation.py` | `[tokens, 2d]` | `[tokens, d]` | GeGLU MLP |
| `MulAndSilu` | `activation.py` | `[tokens, 2d]` | `[tokens, d]` | gate/up 顺序相反的 SwiGLU |
| `SiluAndMulWithClamp` | `activation.py` | `[tokens, 2d]` | `[tokens, d]` | clamped SwiGLU |
| `RotaryEmbedding` | `rotary_embedding/base.py` | `positions, q, k` | rotated q/k | attention q/k 位置编码 |
| `MRotaryEmbedding` | `rotary_embedding/mrope.py` | `[3, tokens] positions, q, k` | rotated q/k | 多模态 T/H/W 位置编码 |
| `ApplyRotaryEmb` | `rotary_embedding/common.py` | `x, cos, sin` | rotated x | RoPE 内部通用旋转 |

---

## 18. 最小心智模型

如果只记住一条主线：

```text
Norm 管 residual 和尺度，activation 管 MLP 中间非线性，RoPE 管 attention q/k 的位置编码。
```

在 vLLM 里对应为：

```text
模型结构文件负责放置这些 module，
ModelRunner 负责准备 positions，
CustomOp 负责按平台选择实现，
_custom_ops / torch.ops._C / Triton / flash-attn 负责真正执行 kernel。
```

再压缩成一句话：

```text
这些基础算子不是调度层组件，而是模型 block 内部的高频计算节点；
它们通过 CustomOp 隐藏 backend 差异，通过 fused kernel 减少临时张量和 kernel launch，
共同决定 decoder forward 中 attention/MLP 之外的稳定性能开销。
```
