# 06. Attention / MLP / Norm blocks 如何在模型中复用？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\attention\attention.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\linear.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\layernorm.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\activation.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\rotary_embedding\__init__.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\llama.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\gemma2.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\deepseek_v2.py`

本问题关注：vLLM 里不同模型文件为什么看起来都在重复写 `Attention / MLP / DecoderLayer`，但底层并没有重复实现矩阵并行、量化、KV cache、RoPE、RMSNorm、激活函数等算子；这些基础 blocks 分别封装在哪些 layer 组件里；标准 decoder block 的执行顺序是什么；Llama / Gemma / DeepSeek 这类模型如何在同一套底座上组合出不同架构。

---

## 1. 一句话回答

vLLM 的模型实现通常只负责“拼装架构”，真正可复用的执行能力放在 `model_executor/layers` 下面：

```text
Attention block：
  QKVParallelLinear / rotary_emb / Attention / RowParallelLinear

MLP block：
  MergedColumnParallelLinear / activation / RowParallelLinear

Norm block：
  RMSNorm / GemmaRMSNorm / LayerNorm，以及 fused add + norm

Position block：
  get_rope() 根据 rope_parameters 选择 RoPE / YaRN / M-RoPE / XD-RoPE 等实现
```

以 Llama 为例，一个 decoder layer 可以压缩成：

```text
hidden_states
  → input_layernorm
  → self_attn(qkv_proj → rope → Attention → o_proj)
  → post_attention_layernorm
  → mlp(gate_up_proj → activation → down_proj)
  → hidden_states, residual
```

模型之间的差异主要体现在：

```text
用哪种 norm；
用哪种 activation；
attention 是否 sliding window / MLA / cross attention；
MLP 是 dense 还是 MoE；
RoPE 参数和位置编码变体；
residual / norm 的排列方式。
```

底层的 tensor parallel、量化权重加载、KV cache 写入、attention backend 选择、CUDA custom op 等逻辑则尽量复用同一批 layer 组件。

---

## 2. 先看标准 Llama block 如何拼出来

Llama 是最典型的“标准 decoder-only block”例子。

位置：`vllm/model_executor/models/llama.py:82`

### 2.1 LlamaMLP

Llama MLP 不是写三个普通 `nn.Linear`，而是复用 vLLM 的并行 linear：

```python
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj = RowParallelLinear(...)
self.act_fn = SiluAndMul()
```

位置：`llama.py:95` 到 `llama.py:116`

forward 很短：

```python
x, _ = self.gate_up_proj(x)
x = self.act_fn(x)
x, _ = self.down_proj(x)
```

位置：`llama.py:118` 到 `llama.py:122`

这说明：

```text
gate_proj + up_proj 被融合成一个 column-parallel 投影；
SwiGLU 的 silu(gate) * up 由 SiluAndMul 处理；
down_proj 使用 row-parallel，并在需要时做 TP all-reduce。
```

### 2.2 LlamaAttention

Llama attention 也是“组合式”：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.rotary_emb = get_rope(...)
self.attn = Attention(...)
```

位置：`llama.py:165` 到 `llama.py:222`

forward 链路是：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

位置：`llama.py:224` 到 `llama.py:234`

也就是说：

```text
模型文件决定 q/k/v 怎么投影、RoPE 怎么套、输出怎么投影；
真正的 paged attention / KV cache / backend kernel 在 Attention layer 内部完成。
```

### 2.3 LlamaDecoderLayer

标准 decoder layer 的组合：

```python
self.self_attn = LlamaAttention(...)
self.mlp = LlamaMLP(...)
self.input_layernorm = RMSNorm(...)
self.post_attention_layernorm = RMSNorm(...)
```

位置：`llama.py:285` 到 `llama.py:310`

forward 中 residual 和 norm 的组织方式是：

```text
第一次进入 layer：
  residual = hidden_states
  hidden_states = input_layernorm(hidden_states)

后续 layer：
  hidden_states, residual = input_layernorm(hidden_states, residual)

attention 后：
  hidden_states, residual = post_attention_layernorm(hidden_states, residual)
  hidden_states = mlp(hidden_states)
```

位置：`llama.py:313` 到 `llama.py:330`

这里的重点是：

```text
RMSNorm.forward(x, residual) 不是普通 norm；
它可以走 fused_add_rms_norm，把 residual add 和 norm 融合在一起。
```

---

## 3. Attention block 复用了哪些底层能力

Attention block 可以拆成五层：

```text
hidden_states
  → qkv projection
  → q/k/v split
  → position embedding
  → Attention layer
  → output projection
```

### 3.1 QKVParallelLinear：q/k/v 的融合投影和 TP 切分

定义位置：`vllm/model_executor/layers/linear.py:914`

`QKVParallelLinear` 的作用是把 attention 的 Q、K、V 三个投影合并成一个 column-parallel linear。

它的关键输入包括：

```text
hidden_size
head_size
total_num_heads
total_num_kv_heads
v_head_size
quant_config
```

位置：`linear.py:942` 到 `linear.py:957`

它会根据 tensor parallel size 计算本 rank 的：

```text
num_heads：本 rank 的 query heads
num_kv_heads：本 rank 的 key/value heads
num_kv_head_replicas：当 KV heads 少于 TP size 时的复制倍数
output_sizes：[q_proj_size, k_proj_size, v_proj_size]
```

位置：`linear.py:965` 到 `linear.py:984`

这就是为什么 GQA / MQA 能复用同一个 linear：

```text
num_attention_heads 可以大于 num_key_value_heads；
Q heads 按 TP 切分；
KV heads 在数量不足时可以跨 rank 复制。
```

### 3.2 Attention：统一入口，后端可替换

定义位置：`vllm/model_executor/layers/attention/attention.py:192`

`Attention` 的注释直接说明它负责三件事：

```text
1. 把输入 key / value 存入 KV cache；
2. 执行 MHA / MQA / GQA attention；
3. 返回 attention output。
```

位置：`attention.py:192` 到 `attention.py:202`

初始化时会根据 head size、dtype、KV cache dtype、attention type 等选择 backend：

```python
self.attn_backend = get_attn_backend(...)
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

位置：`attention.py:318` 到 `attention.py:400`

因此模型文件里写的是：

```python
self.attn = Attention(...)
```

但运行时真正可能走：

```text
FlashAttention / FlashInfer / Triton attention / Flex attention / 其他平台 backend
```

具体由 selector 和平台能力决定，而不是每个模型单独判断。

### 3.3 Attention.forward 不显式接收 attention metadata

`Attention.forward()` 的签名只有：

```python
def forward(self, query, key, value, output_shape=None)
```

位置：`attention.py:452` 到 `attention.py:461`

但注释说明 attention metadata 来自 forward context：

```text
attn_metadata 由 ModelRunner.execute_model() 外层的 set_forward_context 设置，
Attention 通过 get_forward_context().attn_metadata 读取。
```

位置：`attention.py:462` 到 `attention.py:470`

真正取 context 的辅助函数是：

```python
forward_context = get_forward_context()
attn_metadata_raw = forward_context.attn_metadata
attn_layer = forward_context.no_compile_layers[layer_name]
kv_cache = attn_layer.kv_cache
slot_mapping = forward_context.slot_mapping
```

位置：`attention.py:670` 到 `attention.py:710`

所以 attention block 的隐式依赖是：

```text
ModelRunner 构造 attention metadata / slot mapping
  → set_forward_context(...)
  → Attention.forward(q, k, v)
  → unified_kv_cache_update / unified_attention_with_output
  → backend impl.forward(..., kv_cache, attn_metadata, ...)
```

### 3.4 KV cache 写入和 attention 计算的关系

`Attention.forward()` 中，如果 backend 不自己包含 KV cache update，会先调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

然后再调用：

```text
unified_attention_with_output(query, key, value, output, layer_name)
```

位置：`attention.py:505` 到 `attention.py:543`

`unified_kv_cache_update()` 内部会拿到当前 layer 的 slot mapping，然后调用 backend 的 `do_kv_cache_update()`。

位置：`attention.py:713` 到 `attention.py:736`

`unified_attention_with_output()` 则调用：

```python
self.impl.forward(
    self,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=output,
)
```

位置：`attention.py:755` 到 `attention.py:784`

这说明：

```text
模型层只产生 q/k/v；
KV 写入哪个 slot、attention 看哪些 block，由 ModelRunner 的 metadata 和 Attention backend 决定。
```

### 3.5 output projection 复用 RowParallelLinear

attention output 最后通常走：

```python
output, _ = self.o_proj(attn_output)
```

Llama 位置：`llama.py:232` 到 `llama.py:234`

`o_proj` 是 `RowParallelLinear`，定义位置：`linear.py:1491`。

它的语义是：

```text
输入维度按 TP 切分；
每个 rank 计算部分输出；
需要时通过 tensor_model_parallel_all_reduce 合并结果。
```

核心 all-reduce 位置：`linear.py:1646` 到 `linear.py:1649`

---

## 4. MLP block 复用了哪些底层能力

标准 gated MLP 的形态是：

```text
x
  → gate_up_proj(x)
  → activation(gate, up)
  → down_proj(x)
```

### 4.1 MergedColumnParallelLinear：融合 gate_proj 和 up_proj

定义位置：`vllm/model_executor/layers/linear.py:577`

`MergedColumnParallelLinear` 的作用是把多个 column-parallel linear 沿输出维度拼起来。

Llama 用它表示：

```text
gate_proj + up_proj
```

位置：`llama.py:95` 到 `llama.py:102`

对应权重加载时，Llama 明确把 HF checkpoint 的两个名字映射到一个 fused 参数：

```python
(".gate_up_proj", ".gate_proj", 0),
(".gate_up_proj", ".up_proj", 1),
```

位置：`llama.py:434` 到 `llama.py:441`

因此文档里看到的：

```text
gate_up_proj
```

并不是 HuggingFace 原始结构里一定存在的单个权重，而是 vLLM 为推理和 TP 做的 fused module。

### 4.2 SiluAndMul / GeluAndMul：GLU 激活也被封装

Llama 使用：

```python
self.act_fn = SiluAndMul()
```

位置：`llama.py:112` 到 `llama.py:116`

`SiluAndMul` 定义位置：`activation.py:116`。

它做的是：

```text
x[..., :d] = gate
x[..., d:] = up
return silu(gate) * up
```

位置：`activation.py:137` 到 `activation.py:148`

Gemma2 则使用：

```python
self.act_fn = GeluAndMul(approximate="tanh")
```

位置：`gemma2.py:85` 到 `gemma2.py:95`

这说明 MLP 的底层组织可以复用，只替换 activation 就能适配不同模型。

### 4.3 RowParallelLinear：down_proj 和 TP 归并

MLP 的 down projection 通常是：

```python
self.down_proj = RowParallelLinear(...)
```

Llama 位置：`llama.py:103` 到 `llama.py:111`

它会根据 `reduce_results` 决定是否 all-reduce。

位置：`linear.py:1536` 到 `linear.py:1538`，`linear.py:1646` 到 `linear.py:1649`

因此 dense MLP 在 TP 下可以记成：

```text
gate_up_proj：column parallel，输出每个 rank 的 intermediate shard
activation：本地执行
 down_proj：row parallel，把结果规约回 hidden_size
```

### 4.4 MoE 不是重写 MLP，而是替换 MLP block

DeepSeekV2 同时有 dense MLP 和 MoE。

Dense MLP 仍然是：

```python
MergedColumnParallelLinear
SiluAndMul
RowParallelLinear
```

位置：`deepseek_v2.py:199` 到 `deepseek_v2.py:243`

MoE block 则用：

```python
self.gate = GateLinear(...)
self.shared_experts = DeepseekV2MLP(...)
self.experts = FusedMoE(...)
```

位置：`deepseek_v2.py:274` 到 `deepseek_v2.py:351`

forward 里：

```text
hidden_states
  → gate 产生 router_logits
  → FusedMoE(hidden_states, router_logits)
  → 必要时 sequence_parallel all_gather
```

位置：`deepseek_v2.py:361` 到 `deepseek_v2.py:388`

所以 MoE 可以理解为：

```text
把标准 dense MLP 替换成 router + experts，
但底层专家计算、量化、并行仍然复用 vLLM 的 layer / fused_moe 组件。
```

---

## 5. Norm block：为什么 RMSNorm 可以同时处理 residual

vLLM 的 norm 组件在：`vllm/model_executor/layers/layernorm.py`。

### 5.1 RMSNorm

定义位置：`layernorm.py:35`

核心语义是：

```text
x -> weight * x / sqrt(mean(x^2) + eps)
```

位置：`layernorm.py:37` 到 `layernorm.py:42`

但它的 forward 支持：

```python
def forward_native(self, x, residual=None)
```

位置：`layernorm.py:74` 到 `layernorm.py:94`

如果没有 residual：

```text
rms_norm(x)
```

如果有 residual：

```text
fused_add_rms_norm(x, residual)
```

这就是 LlamaDecoderLayer 能写成：

```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

位置：`llama.py:323` 到 `llama.py:324`

而不是显式写：

```text
hidden_states = hidden_states + residual
residual = hidden_states
hidden_states = rms_norm(hidden_states)
```

### 5.2 GemmaRMSNorm

Gemma 系列用 `GemmaRMSNorm`，定义位置：`layernorm.py:127`。

它和普通 RMSNorm 的区别写在注释里：

```text
1. x * (1 + w) 而不是 x * w；
2. dtype 转换顺序不同。
```

位置：`layernorm.py:129` 到 `layernorm.py:135`

Gemma2DecoderLayer 里使用了四个 norm：

```python
input_layernorm
post_attention_layernorm
pre_feedforward_layernorm
post_feedforward_layernorm
```

位置：`gemma2.py:217` 到 `gemma2.py:226`

forward 顺序也和 Llama 不完全一样：

```text
input norm
  → attention
  → post_attention norm
  → pre_feedforward norm + residual
  → mlp
  → post_feedforward norm
```

位置：`gemma2.py:228` 到 `gemma2.py:250`

这说明：

```text
同样复用 Attention / MLP / Norm 底座，
但模型文件可以自由决定 norm 的数量和排列方式。
```

### 5.3 LayerNorm

`LayerNorm` 定义位置：`layernorm.py:305`。

它是更传统的：

```python
F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)
```

位置：`layernorm.py:317` 到 `layernorm.py:320`

DeepSeekV2 的 MLA 相关路径里会用到 `LayerNorm`，例如：

```python
self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
```

位置：`deepseek_v2.py:644`

所以 norm block 不是只有 RMSNorm；模型可以选择 RMSNorm、GemmaRMSNorm、LayerNorm 或 gated norm。

---

## 6. RoPE / position block 如何复用

RoPE 入口在：`vllm/model_executor/layers/rotary_embedding/__init__.py:33`

模型通常不直接实例化某个具体 RoPE 类，而是调用：

```python
self.rotary_emb = get_rope(...)
```

Llama 位置：`llama.py:243` 到 `llama.py:248`

Gemma2 位置：`gemma2.py:151` 到 `gemma2.py:156`

### 6.1 get_rope() 的输入

`get_rope()` 主要接收：

```text
head_size
max_position
is_neox_style
rope_parameters
dtype
dual_chunk_attention_config
```

位置：`rotary_embedding/__init__.py:33` 到 `rotary_embedding/__init__.py:40`

它会先计算：

```text
base = rope_theta 或默认 10000
scaling_type = rope_type 或 default
rotary_dim = rope_dim 或 head_size * partial_rotary_factor
```

位置：`rotary_embedding/__init__.py:63` 到 `rotary_embedding/__init__.py:72`

### 6.2 get_rope() 会缓存 RoPE 实例

缓存 key 包括：

```text
head_size
rotary_dim
max_position
is_neox_style
rope_parameters
dual_chunk_attention_config
dtype
```

位置：`rotary_embedding/__init__.py:74` 到 `rotary_embedding/__init__.py:84`

因此相同配置下不会反复创建 RoPE 对象。

### 6.3 支持的 RoPE 变体

`get_rope()` 会根据 `rope_type` 选择：

```text
default：RotaryEmbedding 或 MRotaryEmbedding
llama3：Llama3RotaryEmbedding
linear：LinearScalingRotaryEmbedding
ntk / dynamic：NTK scaling 相关实现
yarn：YaRNScalingRotaryEmbedding 或 MRotaryEmbedding
deepseek_yarn：DeepseekScalingRotaryEmbedding
longrope：Phi3LongRoPEScaledRotaryEmbedding
xdrope：XDRotaryEmbedding
openpangu：MRotaryEmbeddingInterleaved
proportional：Gemma4RotaryEmbedding
```

位置：`rotary_embedding/__init__.py:101` 到 `rotary_embedding/__init__.py:382`

所以模型 attention 里只需要：

```python
q, k = self.rotary_emb(positions, q, k)
```

Llama 位置：`llama.py:231`

具体是普通 RoPE、M-RoPE、XD-RoPE、YaRN 还是长上下文缩放，由 config 的 `rope_parameters` 决定。

---

## 7. Linear layer 复用的核心：并行、量化、权重加载

很多模型文件都 import：

```text
QKVParallelLinear
MergedColumnParallelLinear
ColumnParallelLinear
RowParallelLinear
ReplicatedLinear
```

这些并不是简单封装 `torch.nn.Linear`，而是把 vLLM 推理需要的通用能力放在一起。

### 7.1 LinearBase 统一处理 quant_method

`LinearBase` 定义位置：`linear.py:228`。

初始化时会选择量化方法：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError(...)
```

位置：`linear.py:269` 到 `linear.py:274`

也就是说，模型文件只传：

```text
quant_config=quant_config
```

具体是 FP8、AWQ、GPTQ、Marlin、bitsandbytes 还是非量化，由 quant config 和 linear layer 共同决定。

### 7.2 ColumnParallelLinear

定义位置：`linear.py:392`

语义：

```text
权重按输出维度切分：A = [A1, A2, ..., Ap]
每个 rank 得到一部分 output features
必要时 gather_output 做 all-gather
```

核心 forward：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
if self.gather_output and self.tp_size > 1:
    output = tensor_model_parallel_all_gather(output_parallel)
else:
    output = output_parallel
```

位置：`linear.py:548` 到 `linear.py:566`

QKV 和 gate_up 都是 column-parallel 的特化。

### 7.3 RowParallelLinear

定义位置：`linear.py:1491`

语义：

```text
权重按输入维度切分；
输入可以已经被切分，也可以在 forward 内切分；
每个 rank 计算部分结果；
需要时 all-reduce 得到完整 output。
```

核心 forward：

```python
if self.input_is_parallel:
    input_parallel = input_
else:
    input_parallel = split_tensor_along_last_dim(...)
output_parallel = self.quant_method.apply(...)
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

位置：`linear.py:1628` 到 `linear.py:1654`

attention 的 `o_proj` 和 MLP 的 `down_proj` 通常都使用它。

### 7.4 packed_modules_mapping 和 fused 权重加载

LlamaForCausalLM 声明：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```

位置：`llama.py:489` 到 `llama.py:492`

LlamaModel.load_weights 也会把 HF 权重名映射到 fused 参数：

```text
q_proj / k_proj / v_proj → qkv_proj
gate_proj / up_proj → gate_up_proj
```

位置：`llama.py:434` 到 `llama.py:469`

这说明模型结构看起来是 vLLM 自己的 fused 形态，但仍能加载 HuggingFace 原始 checkpoint。

---

## 8. Decoder block 的完整数据流

以 Llama 为基准，一个 layer 的完整数据流是：

```text
输入：hidden_states, residual, positions

1. Norm
   residual = hidden_states                       # 第一层或 residual 为空时
   hidden_states = input_layernorm(hidden_states)

2. Self Attention
   qkv = qkv_proj(hidden_states)
   q, k, v = split(qkv)
   q, k = rotary_emb(positions, q, k)
   attn_output = Attention(q, k, v)
   hidden_states = o_proj(attn_output)

3. Norm
   hidden_states, residual = post_attention_layernorm(hidden_states, residual)

4. MLP
   gate_up = gate_up_proj(hidden_states)
   hidden_states = activation(gate_up)
   hidden_states = down_proj(hidden_states)

输出：hidden_states, residual
```

对应源码：

- `LlamaAttention.forward()`：`llama.py:224`
- `LlamaMLP.forward()`：`llama.py:118`
- `LlamaDecoderLayer.forward()`：`llama.py:313`

如果再放到整个 model 里：

```text
embed_tokens(input_ids)
  → layers[start_layer:end_layer]
  → final norm
  → hidden_states
  → compute_logits(hidden_states)  # 在 ModelRunner 后处理阶段调用
```

LlamaModel forward 位置：`llama.py:392` 到 `llama.py:431`

---

## 9. Pipeline Parallel 下 block 如何切分

`LlamaModel` 通过 `make_layers()` 创建层，并记录：

```python
self.start_layer, self.end_layer, self.layers = make_layers(...)
```

位置：`llama.py:375` 到 `llama.py:379`

forward 时只执行当前 PP rank 负责的切片：

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(...)
```

位置：`llama.py:412` 到 `llama.py:420`

非最后 PP rank 返回：

```python
IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
```

位置：`llama.py:422` 到 `llama.py:425`

最后 PP rank 才执行 final norm：

```python
hidden_states, _ = self.norm(hidden_states, residual)
```

位置：`llama.py:427`

所以 block 复用和 PP 的关系是：

```text
每个 rank 仍然执行同样的 DecoderLayer block；
只是 layers[start_layer:end_layer] 不同；
rank 之间通过 IntermediateTensors 传 hidden_states / residual。
```

---

## 10. 变体一：Gemma2 只是换了 norm / activation / sliding window 规则

Gemma2Attention 仍然是：

```text
QKVParallelLinear
get_rope
Attention
RowParallelLinear
```

位置：`gemma2.py:99` 到 `gemma2.py:184`

但它的差异包括：

```text
scaling = config.query_pre_attn_scalar ** -0.5；
attention 可带 logits_soft_cap；
根据 config.layer_types 决定每层是否 sliding_attention；
MLP 使用 GeluAndMul(approximate="tanh")；
norm 使用 GemmaRMSNorm；
DecoderLayer 有 input / post_attention / pre_feedforward / post_feedforward 四个 norm。
```

对应位置：

- attention scaling / sliding：`gemma2.py:133`、`gemma2.py:158` 到 `gemma2.py:170`
- MLP activation：`gemma2.py:85` 到 `gemma2.py:95`
- norm 排列：`gemma2.py:217` 到 `gemma2.py:250`

这说明：

```text
Gemma2 不需要重写 attention backend 或 TP linear；
它只是用同一批组件拼出自己的 block 顺序和参数。
```

---

## 11. 变体二：DeepSeekV2 把 MLP 换成 MoE，把 Attention 换成 MLA

DeepSeekV2 展示了更复杂的复用方式。

### 11.1 Dense MLP 仍是标准 gated MLP

`DeepseekV2MLP` 和 LlamaMLP 非常像：

```text
MergedColumnParallelLinear
SiluAndMul
RowParallelLinear
```

位置：`deepseek_v2.py:199` 到 `deepseek_v2.py:243`

区别是它额外支持：

```text
is_sequence_parallel
reduce_results
```

用于 MoE / sequence parallel 场景。

### 11.2 MoE block 使用 FusedMoE

`DeepseekV2MoE` 里：

```text
GateLinear：产生 router logits
DeepseekV2MLP：可选 shared experts
FusedMoE：执行 routed experts
```

位置：`deepseek_v2.py:246` 到 `deepseek_v2.py:351`

forward 里要么内部 router，要么先 gate：

```text
router_logits = gate(hidden_states)
final_hidden_states = experts(hidden_states, router_logits)
```

位置：`deepseek_v2.py:372` 到 `deepseek_v2.py:380`

因此 MoE 不是 attention block 的变体，而是 FFN / MLP block 的替换。

### 11.3 MLA attention 是 Attention block 的深度变体

DeepSeekV2 文件里会根据配置选择：

```text
DeepseekAttention
DeepseekV2MLAAttention
DeepseekV2Attention
```

位置：`deepseek_v2.py:1132` 到 `deepseek_v2.py:1138`

MLA 路径还会返回特殊的 KV cache spec：

```python
return MLAAttentionSpec(...)
```

位置：`deepseek_v2.py:589` 到 `deepseek_v2.py:590`

这说明：

```text
标准 Attention layer 适合 MHA / MQA / GQA；
MLA 这类结构差异太大时，会在模型文件里定义专门 attention block；
但它仍然接入 vLLM 的 KV cache spec、linear、norm、quant_config、forward context 等基础设施。
```

---

## 12. Attention / MLP / Norm 的边界怎么理解

可以按职责边界记：

| block | 模型文件负责 | 共享 layer 负责 |
|---|---|---|
| Attention | head 数、KV head 数、RoPE 参数、是否 sliding / MLA、qkv/o_proj 怎么连 | QKV TP 切分、KV cache、attention backend、slot mapping、量化、输出规约 |
| MLP | intermediate size、activation 类型、dense 还是 MoE、shared experts | fused gate/up、row/column parallel、activation custom op、expert kernel、量化 |
| Norm | pre-norm / post-norm 排列、用 RMSNorm 还是 GemmaRMSNorm / LayerNorm | fused add + norm、平台 custom op、dtype 处理 |
| Position | positions 张量如何传入、模型 config 的 rope 参数 | RoPE 实例缓存、scaling 变体、M-RoPE / XD-RoPE / YaRN 等实现 |

---

## 13. 为什么 vLLM 要这样拆

这样拆的好处是：

```text
1. 新模型只要写架构组合，不必重写 attention kernel；
2. TP / PP / EP / quantization 的复杂逻辑集中在共享 layer；
3. 权重加载可以把 HF checkpoint 映射到 vLLM fused 参数；
4. attention backend 可以按平台和配置切换；
5. CUDA graph / torch.compile / custom op 更容易统一处理。
```

一个典型新增 decoder-only 模型，通常需要实现：

```text
ModelForCausalLM
  → Model
  → DecoderLayer
  → Attention
  → MLP / MoE
  → load_weights / packed_modules_mapping
```

但这些类内部大部分还是复用：

```text
QKVParallelLinear
MergedColumnParallelLinear
RowParallelLinear
Attention
RMSNorm / LayerNorm
get_rope
LogitsProcessor
ParallelLMHead
```

---

## 14. 常见疑惑

### 14.1 Attention 类是不是一个完整 transformer attention block？

不是。

`Attention` 只负责拿已经算好的 `q / k / v` 做 KV cache update 和 attention backend 调用。

完整 attention block 还包括：

```text
qkv_proj
RoPE / position embedding
Attention
output projection
```

这些通常在模型文件的 `LlamaAttention`、`Gemma2Attention` 等类里组合。

### 14.2 MLP 里的 gate_proj / up_proj 为什么经常看不到？

因为 vLLM 通常用 `MergedColumnParallelLinear` 合并成 `gate_up_proj`。

加载权重时再把 checkpoint 里的：

```text
gate_proj
up_proj
```

映射进 fused 参数。

### 14.3 Norm 为什么返回 tuple？

当传入 residual 时，`RMSNorm` 会执行 fused add + rms norm，并返回：

```text
hidden_states, residual
```

这让 decoder layer 可以减少显式 add / norm 的中间操作。

### 14.4 Attention metadata 为什么不在模型 forward 参数里？

因为它由 ModelRunner 构造，并通过 `set_forward_context()` 放进 forward context。

模型只传 `q / k / v`，Attention layer 内部通过 layer name 找到对应的 metadata、KV cache 和 slot mapping。

### 14.5 每个模型都必须用同一个 LlamaMLP / LlamaAttention 吗？

不是。

模型可以写自己的 `Gemma2Attention`、`DeepseekV2MoE`、`DeepseekV2MLAAttention`。

但只要底层结构相同，就会继续复用 vLLM 的 linear、activation、norm、attention backend、RoPE 等组件。

---

## 15. 总结

vLLM 的模型架构复用可以记成三层：

```text
模型架构层：
  LlamaDecoderLayer / Gemma2DecoderLayer / DeepseekV2DecoderLayer
  决定 block 顺序、norm 位置、dense/MoE、attention 变体。

共享组件层：
  QKVParallelLinear / MergedColumnParallelLinear / RowParallelLinear
  Attention / RMSNorm / GemmaRMSNorm / LayerNorm / get_rope / activation op。

运行时后端层：
  attention backend、KV cache、slot mapping、tensor parallel collectives、quant_method、custom ops。
```

一句话压缩：

```text
模型文件负责“搭积木”，layers 目录负责“积木怎么高效执行”；Attention / MLP / Norm blocks 的复用点就在这些共享 layer 组件和 forward context 机制里。
```
