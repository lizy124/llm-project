# 06. Attention forward 主链路

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\kv_transfer_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\mla_attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\cross_attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\encoder_only_attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\chunked_local_attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\`

本文用于梳理一次模型 forward 中 attention layer 如何拿到 `ForwardContext`，如何读取 `AttentionMetadata` / `slot_mapping` / `kv_cache`，如何更新 KV cache，如何调用具体 backend kernel，以及输出如何回到模型 block。

---

## 1. 本文要回答的问题

```text
模型 forward 是从哪里进入 attention layer 的？
Attention.forward() 的输入输出是什么？
Attention.forward() 如何拿到 attn_metadata、slot_mapping、kv_cache？
KV cache 是在 attention forward 内写入，还是单独写入？
forward_includes_kv_cache_update 有什么意义？
unified_attention_with_output 是什么层？
backend impl.forward 负责什么？
KV connector hook 在 attention forward 前后如何触发？
MLA attention forward 为什么分 MHA / MQA 两条路径？
encoder-only / cross attention / chunked local attention 如何复用主链路？
CUDA graph / torch.compile / ubatching 对 attention forward 有什么影响？
```

---

## 2. 一句话回答

Attention forward 是模型层、`ForwardContext`、backend impl 和 paged KV cache 的交汇点。

`GPUModelRunner.execute_model()` 先构造 `attn_metadata` 和 `slot_mapping`，再用 `set_forward_context()` 包住模型 forward；模型内部的 `Attention.forward()` 不显式接收 metadata，而是通过 `get_attention_context(layer_name)` 从当前 `ForwardContext` 取到本层的 metadata、KV cache 和 slot mapping，然后调用 `unified_attention_with_output()`，最终落到具体 backend 的 `impl.forward()`。

最小链路是：

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → set_forward_context(attn_metadata, slot_mapping, ...)
  → _model_forward()
  → model layer forward
  → q/k/v projection
  → Attention.forward(query, key, value)
  → get_attention_context(layer_name)
  → optional unified_kv_cache_update(...)
  → unified_attention_with_output(...)
  → backend impl.forward(...)
  → attention output
  → model block 后续 MLP / residual
```

如果只记住一句话：

```text
Attention.forward() 本身不重新决定 batch 形态，它只把模型算出的 Q/K/V、forward context 中的 metadata、KV cache 和 backend impl 连接起来。
```

---

## 3. 总体调用链

一次普通 generation forward 中，attention forward 处在下面这条链路中：

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._determine_batch_execution_and_padding()
  → GPUModelRunner._get_slot_mappings()
  → GPUModelRunner._build_attention_metadata()
  → GPUModelRunner._preprocess()
  → set_forward_context(...)
  → GPUModelRunner._model_forward(...)
      → self.model(...)
          → decoder layer forward
              → qkv projection
              → Attention.forward(query, key, value)
                  → maybe_calc_kv_scales
                  → query quantization
                  → reshape Q/K/V
                  → optional unified_kv_cache_update
                  → unified_attention_with_output
                      → maybe_transfer_kv_layer wrapper
                      → get_attention_context(layer_name)
                      → self.impl.forward(...)
              → attention output projection
              → residual / norm / MLP
  → hidden_states
  → compute_logits(hidden_states[logits_indices])
  → ExecuteModelState
  → sample_tokens()
```

注意：

```text
q/k/v projection 不在 Attention.forward() 里面；
Attention.forward() 收到的是已经投影后的 query / key / value；
Attention.forward() 返回 attention output，后续 output projection 通常在模型层自己的 attention module 中完成。
```

---

## 4. GPUModelRunner 如何包住模型 forward

`GPUModelRunner.execute_model()` 在真正 forward 前已经完成了输入、slot mapping 和 metadata 构造。

关键代码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`

```text
_build_attention_metadata(...)
  → attn_metadata
  → spec_decode_common_attn_metadata
```

然后进入模型 forward：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4302`

```python
with (
    set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_mode,
        batch_descriptor=batch_desc,
        ubatch_slices=ubatch_slices_padded,
        slot_mapping=slot_mappings,
        skip_compiled=has_encoder_input,
    ),
    record_function_or_nullcontext("gpu_model_runner: forward"),
    self.maybe_get_kv_connector_output(...) as kv_connector_output,
):
    model_output = self._model_forward(...)
```

这说明模型 forward 外面至少有三层上下文：

```text
1. set_forward_context(...)
   注入 attention metadata、slot mapping、CUDA graph mode、batch descriptor、ubatch 信息。

2. record_function_or_nullcontext(...)
   profiling / trace 标记。

3. maybe_get_kv_connector_output(...)
   KV connector 在 forward 前后收集或 finalize KV transfer 输出。
```

### 4.1 _model_forward() 本身很薄

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

`_model_forward()` 最终只是调用：

```python
self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

它不负责 attention metadata 构造，也不负责 KV cache 更新。

也就是说：

```text
attention forward 的运行时依赖不是通过 self.model(...) 的显式参数传进去，
而是通过 set_forward_context(...) 放进全局 forward context。
```

---

## 5. ForwardContext 保存了什么

`ForwardContext` 定义在：`code/vllm/vllm/forward_context.py:128`

核心字段包括：

```text
no_compile_layers
attn_metadata
slot_mapping
dp_metadata
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
additional_kwargs
```

其中 attention forward 最关心的是：

```text
attn_metadata
  layer_name → AttentionMetadata。

slot_mapping
  layer_name → 当前层 token 写入 KV cache 的 slot。

no_compile_layers
  layer_name → Attention / MLAAttention layer 实例。
```

### 5.1 set_forward_context()

位置：`code/vllm/vllm/forward_context.py:249`

`set_forward_context()` 做几件事：

```text
1. 根据 DP / MoE 情况构造 DPMetadata；
2. CUDA graph 模式下补 batch_descriptor；
3. 调 current_platform.set_additional_forward_context(...) 注入平台额外上下文；
4. create_forward_context(...)；
5. 用 override_forward_context(...) 在 with 作用域内设置全局 _forward_context；
6. 退出 with 后恢复旧 context。
```

所以 `ForwardContext` 是一个“forward 作用域级隐式参数区”。

### 5.2 get_attention_context(layer_name)

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`

它从当前 `ForwardContext` 中取出：

```text
attn_metadata：当前 layer 的 metadata；
attn_layer：当前 layer 实例；
kv_cache：当前 layer 绑定的 KV cache tensor；
layer_slot_mapping：当前 layer 的 slot mapping。
```

伪代码：

```text
forward_context = get_forward_context()
attn_metadata = forward_context.attn_metadata[layer_name]
attn_layer = forward_context.no_compile_layers[layer_name]
kv_cache = attn_layer.kv_cache
layer_slot_mapping = forward_context.slot_mapping[layer_name]
```

这一步是 attention forward 的关键桥梁：

```text
模型层只知道 layer_name；
ForwardContext 根据 layer_name 找到 metadata、layer 对象、KV cache 和 slot mapping。
```

---

## 6. Attention layer 初始化时已经绑定 backend impl

标准 `Attention` 定义在：`code/vllm/vllm/model_executor/layers/attention/attention.py:178`

初始化时会选择 backend：

```text
get_attn_backend(...)
  → self.attn_backend
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:304`

然后实例化 impl：

```text
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:373`

同时把当前 layer 注册到静态 forward context：

```text
compilation_config.static_forward_context[prefix] = self
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:397`

这就是后面 `get_attention_context(layer_name)` 能通过 `no_compile_layers[layer_name]` 找回 layer 实例的原因。

### 6.1 kv_cache 初始只是占位

初始化时：

```text
self.kv_cache = torch.tensor([])
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:413`

真正的 KV cache tensor 会在 KV cache 初始化 / bind 阶段绑定到 attention layer。

所以 attention layer 的生命周期可以分成：

```text
模型加载时：选择 backend，创建 impl，注册 static_forward_context；
KV cache 初始化时：绑定真实 kv_cache tensor；
每轮 forward：从 ForwardContext 取 metadata / slot_mapping，调用 impl.forward。
```

---

## 7. Attention.forward() 的输入输出

入口：`code/vllm/vllm/model_executor/layers/attention/attention.py:438`

签名：

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
```

它的输入是模型层前面线性层已经投影出的 Q/K/V：

```text
query：当前 token 的 Q；
key：当前 token 的 K；
value：当前 token 的 V；
output_shape：某些 backend / MLA 变体可能需要特殊 output shape。
```

它的输出是：

```text
[num_tokens, hidden_size] 形状的 attention output。
```

### 7.1 Attention.forward() 内部步骤

可以按顺序拆成：

```text
1. 如果 calculate_kv_scales=True，调用 maybe_calc_kv_scales；
2. 如果 backend 支持 query quantization，对 query 做量化；
3. 分配 output tensor；
4. reshape query/key/value 到 [num_tokens, heads, head_dim]；
5. 如果 backend 不在 forward 内更新 KV cache，则先调用 unified_kv_cache_update；
6. 调用 unified_attention_with_output；
7. 返回 output.view(-1, hidden_size)。
```

关键代码位置：

```text
maybe_calc_kv_scales：attention.py:457
query quantization：attention.py:462
output 分配：attention.py:474
Q/K/V reshape：attention.py:481
separate KV cache update：attention.py:491
unified_attention_with_output：attention.py:502
返回 output：attention.py:530
```

### 7.2 Q/K/V reshape 的边界

`Attention.forward()` 会把输入整理成 backend impl 统一看到的形状：

```text
query: [num_tokens, num_heads, head_size]
key:   [num_tokens, num_kv_heads, head_size]
value: [num_tokens, num_kv_heads, head_size_v]
output:[num_tokens, num_heads, head_size_v]
```

这一步发生在 custom op 外面，注释里说明是为了减少 non-CUDA-graph 区域的 CPU overhead。

### 7.3 use_direct_call 和 opaque custom op

初始化时：

```text
self.use_direct_call = not current_platform.opaque_attention_op()
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:394`

含义是：

```text
某些平台把 attention 注册成一个大的 opaque custom op，让 torch.compile / CUDA graph 更容易处理；
其他平台则直接调用 Python 函数。
```

两条路径做的事情等价：

```text
use_direct_call=True：
  unified_kv_cache_update(...)
  unified_attention_with_output(...)

use_direct_call=False：
  torch.ops.vllm.unified_kv_cache_update(...)
  torch.ops.vllm.unified_attention_with_output(...)
```

区别主要在编译和图捕获边界，而不是业务语义。

---

## 8. KV cache update 在哪里发生

KV cache update 有两种模式，由 backend class 的字段决定：

```text
AttentionBackend.forward_includes_kv_cache_update
```

### 8.1 抽象默认：backend forward 内部更新 KV cache

`AttentionBackend` 抽象基类默认：

```text
forward_includes_kv_cache_update = True
```

这表示：

```text
Attention.forward()
  → unified_attention_with_output(...)
  → impl.forward(...)
      → backend 内部负责把 key/value 写入 kv_cache
      → backend 内部执行 attention
```

不过当前很多 V1 backend 会显式覆盖为 `False`，例如 FlashAttention、FlashInfer、Triton、Flex、CPU、ROCm、TurboQuant 等。因此实际追踪这些 backend 时，通常会先看到 separate KV update，再看到 attention forward。

### 8.2 separate KV update：先写 KV cache，再 attention

有些 backend 声明：

```text
forward_includes_kv_cache_update = False
```

例如 TritonAttentionBackend：`code/vllm/vllm/v1/attention/backends/triton_attn.py:275`

此时 `Attention.forward()` 会在调用 attention 前先执行：

```text
unified_kv_cache_update(key, value, layer_name)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:499`

`unified_kv_cache_update()` 内部：

```text
1. get_attention_context(layer_name)
2. 取 attn_layer、kv_cache、layer_slot_mapping
3. 调 attn_layer.impl.do_kv_cache_update(...)
4. 返回一个空 tensor 作为 dummy dependency
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`

### 8.3 kv_cache_dummy_dep 的意义

`unified_kv_cache_update()` 返回：

```text
torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
```

这个 tensor 不承载业务数据，但会作为 `kv_cache_dummy_dep` 传给 `unified_attention_with_output()`。

注释说明它的作用是：

```text
给 torch.compile 制造数据依赖，确保 KV cache update 一定发生在 attention forward 之前。
```

否则编译器可能认为两个 op 没有依赖，从而重排执行顺序。

### 8.4 KV sharing target layer 会跳过当前层 KV 写入

`Attention.forward()` 里有判断：

```text
self.kv_sharing_target_layer_name is None
```

如果当前层共享更早层的 KV cache，就不会对当前层执行 KV cache update。

这对应 KV sharing / YOCO 类路径：

```text
target layer 先写 KV；
sharing layer 复用 target layer 的 KV，不重复写入。
```

---

## 9. unified_attention_with_output 做什么

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:734`

装饰器：

```python
@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_attention_with_output(...):
```

它的职责非常集中：

```text
1. 接收 reshape 后的 query / key / value / output；
2. 通过 layer_name 调 get_attention_context；
3. 取出当前层 self.impl、kv_cache、attn_metadata；
4. 调用 self.impl.forward(..., output=output)。
```

核心调用：

```python
self.impl.forward(
    self,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=output,
    output_scale=output_scale,
    output_block_scale=output_block_scale,
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:753`

### 9.1 为什么 output 由外层传入

`Attention.forward()` 先分配 output，再传给 backend impl 填充。

这样做的好处：

```text
统一 output shape；
方便 fused output quantization；
方便 custom op 标记 mutates_args；
减少 backend 各自分配 output 的差异。
```

注册 custom op 时也标记了：

```text
mutates_args=["output", "output_block_scale"]
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:779`

### 9.2 maybe_transfer_kv_layer 装饰器

`unified_attention_with_output()` 被 `maybe_transfer_kv_layer` 包装。

文件：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py`

它在启用 v1 KV transfer group 时生效。

逻辑：

```text
进入 attention 前：
  connector.wait_for_layer_load(layer_name)

执行 attention：
  result = func(...)

attention 后：
  connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:50`

也就是说 KV connector hook 的粒度是“attention layer”。

它不是包住整个模型一次性处理所有 KV，而是在每层 attention 执行前后等待 / 保存本层 KV。

---

## 10. backend impl.forward 负责什么

`impl.forward()` 是 backend-specific 执行层。

标准签名大致是：

```text
impl.forward(
  layer,
  query,
  key,
  value,
  kv_cache,
  attn_metadata,
  output,
  output_scale=None,
  output_block_scale=None,
)
```

它负责：

```text
解释 backend-specific AttentionMetadata；
解释 backend-specific KV cache layout；
必要时更新 KV cache；
选择 prefill / decode / mixed path；
调用具体 kernel / wrapper；
把结果写入 output。
```

不同 backend 的 `impl.forward()` 差异很大。

---

## 11. FlashAttention forward 路径

文件：`code/vllm/vllm/v1/attention/backends/flash_attn.py`

入口：`FlashAttentionImpl.forward()`，位置：`flash_attn.py:701`

它看到的输入已经是：

```text
query: [num_tokens, num_heads, head_size]
key: [num_tokens, num_kv_heads, head_size]
value: [num_tokens, num_kv_heads, head_size]
kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
attn_metadata: FlashAttentionMetadata
output: [num_tokens, num_heads, head_size_v]
```

核心流程：

```text
1. profiling run 中 attn_metadata=None，则 output.fill_(0)；
2. encoder-only / encoder attention 走 direct Q/K/V，无需 KV cache；
3. decoder / cross attention 从 kv_cache.unbind(1) 得到 key_cache / value_cache；
4. 修正某些 singleton 维度 stride，满足 FA3/FA4 TMA 对齐要求；
5. 如果 KV cache 是 fp8，按平台 fp8 dtype view；
6. 如果不用 cascade，读取 query_start_loc / seq_lens / block_table / max len；
7. DCP 下走 _forward_with_dcp；
8. 非 DCP 下调用 flash-attn varlen / paged KV kernel；
9. 如果 use_cascade=True，则走 cascade attention 路径。
```

FlashAttention forward 的核心心智模型：

```text
FlashAttentionMetadata 已经把 batch 组织成 FA kernel 可理解的 varlen + paged KV 参数；
impl.forward 负责把这些参数和 kv_cache view 喂给具体 FA kernel。
```

---

## 12. TritonAttention forward 路径

文件：`code/vllm/vllm/v1/attention/backends/triton_attn.py`

Triton backend 的一个关键点是：

```text
forward_includes_kv_cache_update = False
```

位置：`triton_attn.py:275`

因此标准链路变成：

```text
Attention.forward()
  → unified_kv_cache_update(key, value, layer_name)
      → TritonAttentionImpl.do_kv_cache_update(...)
  → unified_attention_with_output(...)
      → TritonAttentionImpl.forward(...)
```

Triton 的 KV cache update 和 attention forward 是两个阶段。

这对 `slot_mapping` 有一个直接影响：

```text
如果存在任何 attention backend separate KV update，
GPUModelRunner._get_slot_mappings() 会使用 padded dimensions，
确保 key/value tensor 和 slot_mapping 形状一致。
```

对应判断在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4185`

Triton backend 还支持更多 attention type：

```text
DECODER
ENCODER
ENCODER_ONLY
ENCODER_DECODER
```

位置：`triton_attn.py:359`

所以它常作为复杂 attention 语义或兼容路径的重要 backend。

---

## 13. FlashInfer / TRTLLM forward 路径

文件：`code/vllm/vllm/v1/attention/backends/flashinfer.py`

FlashInfer 的 metadata builder 会把 batch 拆成：

```text
prefill
  FIPrefill 或 TRTLLMPrefill

decode
  FIDecode 或 TRTLLMDecode
```

构造位置示例：`flashinfer.py:991`

`impl.forward()` 消费这份 metadata 时，通常会按下面逻辑执行：

```text
1. 如果 attn_metadata.prefill 不为空，执行 prefill wrapper / TRTLLM prefill；
2. 如果 attn_metadata.decode 不为空，执行 decode wrapper / TRTLLM decode；
3. 如果有 cascade_wrapper，执行 cascade attention；
4. 把不同区域的结果写回 output 对应 slice。
```

FlashInfer 路径的重点是：

```text
builder 阶段已经完成 wrapper planning；
forward 阶段按 metadata 选择 FlashInfer native wrapper 或 TRT-LLM kernel。
```

因此 FlashInfer 的 forward 更像“执行 plan”，而不是在 forward 中临时拆 batch。

---

## 14. MLA attention forward 路径

MLA 不走标准 `Attention` 类，而是由 `MLAAttention` 处理。

相关文件：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py`

MLA impl 抽象定义在：`code/vllm/vllm/v1/attention/backend.py:863`

它不是单个 `forward()`，而是拆成：

```text
forward_mha(...)
  MHA-style prefill forward pass。

forward_mqa(...)
  MQA-style decode forward pass。
```

### 14.1 MLA forward 的主分支

MLA forward 中会先根据 metadata 得到真实 token 数：

```text
num_actual_toks = attn_metadata.num_actual_tokens
```

然后裁剪 padded 输入：

```text
output = output[:num_actual_toks]
q = q[:num_actual_toks]
k_c_normed = k_c_normed[:num_actual_toks]
k_pe = k_pe[:num_actual_toks]
```

位置：`mla_attention.py:663`

接着判断 token 类型：

```text
num_mqa_tokens = attn_metadata.num_decode_tokens
num_mha_tokens = q.size(0) - num_mqa_tokens
```

位置：`mla_attention.py:682`

### 14.2 prefill：forward_mha

如果 `num_mha_tokens > 0`：

```text
self.impl.forward_mha(
  q[num_mqa_tokens:],
  k_c_normed[num_mqa_tokens:],
  k_pe[num_mqa_tokens:],
  kv_cache,
  attn_metadata,
  ...
)
```

位置：`mla_attention.py:707`

这条路径处理 prefill / chunked prefill，通常是 MHA-style 计算。

### 14.3 decode：forward_mqa

如果 `num_mqa_tokens > 0`：

```text
1. 取 q[:num_mqa_tokens]；
2. 拆 q_nope / q_pe；
3. 计算 q_nope 与 W_UK_T 的 bmm；
4. 必要时做 fp8 query concat / DCP all-gather；
5. 调 self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)；
6. 对 DCP 输出做 lse reduce；
7. 做 v_up projection。
```

核心调用位置：`mla_attention.py:791`

### 14.4 MLA KV cache update

MLA impl 的 `do_kv_cache_update()` 定义在：`code/vllm/vllm/v1/attention/backend.py:931`

它调用：

```text
ops.concat_and_cache_mla(...)
```

把：

```text
kv_c_normed
k_pe
```

合并写入 MLA compressed KV cache。

MLA 的重点是：

```text
普通 attention 写 K 和 V；
MLA 写 compressed latent KV + rope 部分；
prefill 和 decode 的计算路径也不同。
```

---

## 15. Encoder-only attention forward

文件：`code/vllm/vllm/model_executor/layers/attention/encoder_only_attention.py`

`EncoderOnlyAttention` 继承标准 `Attention`，但有两个关键差异：

```text
1. wrapper builder 把 causal 改成 False；
2. get_kv_cache_spec() 返回 None，表示不需要 KV cache。
```

位置：`encoder_only_attention.py:29`

```text
new_common_attn_metadata.causal = False
```

位置：`encoder_only_attention.py:96`

```text
return None
```

所以 encoder-only attention 的 forward 仍然走：

```text
Attention.forward()
  → unified_attention_with_output()
  → backend impl.forward()
```

但 backend impl 会按 `AttentionType.ENCODER_ONLY` 走不依赖 KV cache 的路径。

例如 FlashAttention forward 中：

```text
if attn_type in (ENCODER_ONLY, ENCODER):
  use direct Q/K/V tensors without caching
```

位置：`flash_attn.py:754`

---

## 16. Cross attention forward

文件：`code/vllm/vllm/model_executor/layers/attention/cross_attention.py`

Cross attention 用于 encoder-decoder 模型中 decoder query attend 到 encoder KV。

它的特殊点在 builder 和 impl wrapper。

### 16.1 CrossAttentionBuilder 改写 metadata

位置：`cross_attention.py:81`

它会：

```text
causal = False；
max_seq_len = max(encoder_seq_lens_cpu)；
seq_lens = encoder_seq_lens；
_seq_lens_cpu = encoder_seq_lens_cpu；
根据 encoder block table 重新计算 cross slot_mapping；
调用底层 builder；
把 attn_metadata.slot_mapping 覆盖为 cross slot_mapping。
```

也就是说 cross attention 的 slot mapping 不是普通 `_get_slot_mappings()` 给 decoder token 的 slot mapping，而是 encoder KV cache 的 slot mapping。

### 16.2 CrossAttentionImpl 处理 separate KV update

位置：`cross_attention.py:136`

如果底层 backend 不在 forward 内更新 KV cache，cross wrapper 会手动调用：

```text
self.do_kv_cache_update(..., attn_metadata.slot_mapping)
```

然后再调用底层 `super().forward(...)`。

wrapper 还会把 backend class 的：

```text
forward_includes_kv_cache_update
```

覆盖成 `True`，避免标准 `Attention.forward()` 用普通 layer slot mapping 提前更新 KV。

这很重要：

```text
cross attention 必须使用 CrossAttentionBuilder 算出来的 encoder slot_mapping，
不能使用普通 decoder slot_mapping。
```

---

## 17. Chunked local attention forward

文件：`code/vllm/vllm/model_executor/layers/attention/chunked_local_attention.py`

Chunked local attention 本身仍继承 `Attention.forward()` 主链路，但它包装了 builder：

```text
make_local_attention_virtual_batches(...)
  → 改写 CommonAttentionMetadata
  → 调底层 builder.build(...)
  → 给 metadata 附加 make_virtual_batches_block_table
```

位置：`chunked_local_attention.py:51`

它还显式禁用 CUDA graph：

```text
get_cudagraph_support(...) → AttentionCGSupport.NEVER
```

位置：`chunked_local_attention.py:41`

forward 阶段看起来仍是普通 backend impl.forward，但 metadata 中的 block table / batch 边界已经被改写成“virtual batches”。

所以可以理解为：

```text
ChunkedLocalAttention 不改 Attention.forward()，而是在 metadata 层把 local attention 改写成底层 backend 能执行的 batch 形态。
```

---

## 18. KV connector hook 和 attention forward

KV connector 有两层参与点。

### 18.1 模型 forward 外层

在 `GPUModelRunner.execute_model()` 中：

```text
self.maybe_get_kv_connector_output(...)
```

位置：`gpu_model_runner.py:4315`

它包住整个 model forward，用于收集 connector 输出和 finalize 行为。

如果 spec decode 开启：

```text
defer_kv_connector_finalize = self.speculative_config is not None
```

位置：`gpu_model_runner.py:4297`

原因是 draft model 也可能需要保存 KV，所以 finalize 可能延后。

### 18.2 attention layer 内层

`unified_attention_with_output()` 被 `maybe_transfer_kv_layer` 包装。

位置：`kv_transfer_utils.py:15`

每层 attention：

```text
1. connector.wait_for_layer_load(layer_name)
2. 执行 attention impl.forward
3. connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

这说明：

```text
外层 connector context 管整轮 forward 的输出和生命周期；
内层 decorator 管每个 attention layer 的 KV load / save。
```

---

## 19. CUDA graph / torch.compile 对 attention forward 的影响

CUDA graph / compile 不改变 attention 的语义主链路，但会改变调用边界和 shape 要求。

### 19.1 set_forward_context 携带 graph 信息

`set_forward_context()` 会保存：

```text
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
```

这些信息供编译包装器、平台上下文和模型层使用。

### 19.2 full CUDA graph 下会 padding

在 `GPUModelRunner.execute_model()` 中：

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

位置：`gpu_model_runner.py:4196`

如果 full graph，需要：

```text
num_tokens_padded；
num_reqs_padded；
slot_mapping padded 区域；
metadata.num_actual_tokens 表示真实 token 数。
```

backend impl.forward 通常要用：

```text
attn_metadata.num_actual_tokens
```

裁剪真实区域。

例如 FlashAttention：`flash_attn.py:752`

```text
num_actual_tokens = attn_metadata.num_actual_tokens
```

MLA：`mla_attention.py:663`

```text
num_actual_toks = attn_metadata.num_actual_tokens
```

### 19.3 eager_break_during_capture

`unified_attention_with_output()` 上有：

```text
@eager_break_during_capture
```

位置：`attention.py:734`

这表示某些 capture 场景下需要在 attention custom op 边界打断 eager / graph 行为，避免把不适合 capture 的动态逻辑放进图里。

### 19.4 opaque attention op

如果平台启用 opaque attention op，`Attention.forward()` 会走：

```text
torch.ops.vllm.unified_attention_with_output(...)
```

这让 torch.compile 把 attention 当成一个大 custom op，而不是深入追踪 Python 内部逻辑。

---

## 20. UBatch / microbatch 对 attention forward 的影响

ubatching 的主要处理发生在 metadata 构造阶段。

`ForwardContext` 中保存：

```text
ubatch_slices
```

位置：`forward_context.py:148`

`attn_metadata` 和 `slot_mapping` 的类型也支持 list：

```text
dict[str, AttentionMetadata]
  普通 V1 路径。

list[dict[str, AttentionMetadata]]
  DBO / microbatch 或 speculative 相关路径。
```

位置：`forward_context.py:132`

`get_attention_context()` 当前对 list 的处理是：

```text
attn_metadata_raw[0][layer_name]
```

位置：`attention.py:676`

注释说明 list 场景可用于 speculative decoding，其中 `[0]` 是 base-model metadata dict。

所以从 attention forward 的角度看：

```text
ubatch 的切分边界由 ForwardContext 和 metadata 决定；
Attention.forward() 仍通过 layer_name 取当前应使用的 metadata；
更复杂的 microbatch 调度由 runner / wrapper 处理。
```

---

## 21. 输出如何回到模型和 logits

`Attention.forward()` 返回：

```text
output.view(-1, hidden_size)
```

位置：`attention.py:530`

然后模型 layer 通常会继续执行：

```text
attention output projection；
residual add；
post-attn norm；
MLP；
```

整个模型 forward 返回 `hidden_states` 后，`GPUModelRunner` 才做 logits：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354`

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

因此 attention output 和 logits 的关系是：

```text
attention output 是模型 block 内部中间结果；
logits 是整个模型 forward 结束后，只对 logits_indices 指定位置的 hidden states 计算出来的。
```

---

## 22. 一个完整例子：普通 FlashAttention decode

假设：

```text
模型：decoder-only Llama-like；
backend：FlashAttention；
本轮：纯 decode，每个 request 一个 token；
KV connector：未启用；
CUDA graph：可能启用。
```

链路：

```text
1. GPUModelRunner._prepare_inputs()
   得到 input_ids、positions、query_start_loc、seq_lens、logits_indices。

2. GPUModelRunner._get_slot_mappings()
   根据 block table 和 positions 算出每个新 token 的 KV slot。

3. GPUModelRunner._build_attention_metadata()
   构造 FlashAttentionMetadata。

4. set_forward_context(...)
   注入 attn_metadata 和 slot_mapping。

5. 模型 layer 做 qkv projection。

6. Attention.forward(query, key, value)
   reshape Q/K/V。

7. FlashAttentionBackend.forward_includes_kv_cache_update=False
   所以先调用 unified_kv_cache_update()，由 FlashAttentionImpl.do_kv_cache_update() 按 slot_mapping 写入当前 token K/V。

8. unified_attention_with_output(...)
   取当前 layer 的 metadata、kv_cache。

9. FlashAttentionImpl.forward(...)
   对已经写入的 paged KV cache 执行 attention。

10. Attention.forward() 返回 attention output。
```

---

## 23. 一个完整例子：TritonAttention separate KV update

假设 backend 是 TritonAttention。

区别出现在第 7 步：

```text
TritonAttentionBackend.forward_includes_kv_cache_update=False
```

链路变成：

```text
Attention.forward(query, key, value)
  → unified_kv_cache_update(key, value, layer_name)
      → get_attention_context(layer_name)
      → TritonAttentionImpl.do_kv_cache_update(...)
      → 返回 kv_cache_dummy_dep
  → unified_attention_with_output(..., kv_cache_dummy_dep=...)
      → TritonAttentionImpl.forward(...)
```

`kv_cache_dummy_dep` 确保编译器不能把 attention forward 重排到 KV cache update 前面。

这类 backend 要求 runner 在 slot mapping padding 上更保守，所以 `GPUModelRunner` 会检测是否存在 separate KV update backend。

---

## 24. 一个完整例子：Cross attention

假设 encoder-decoder 模型中 decoder 层执行 cross attention。

链路：

```text
1. CrossAttentionBuilder.build(...)
   causal=False；
   seq_lens=encoder_seq_lens；
   重新计算 encoder KV slot_mapping；
   调底层 builder；
   把 attn_metadata.slot_mapping 改成 cross slot_mapping。

2. set_forward_context(...)
   注入 cross attention metadata。

3. CrossAttention.forward(...)
   继承标准 Attention.forward。

4. unified_attention_with_output(...)
   调 CrossAttentionImpl.forward。

5. CrossAttentionImpl.forward(...)
   如果底层 backend 需要 separate KV update，使用 attn_metadata.slot_mapping 做 KV update；
   然后调用底层 impl.forward。
```

重点：

```text
cross attention 的 KV cache slot mapping 来自 encoder KV cache，不能使用普通 decoder slot_mapping。
```

---

## 25. 容易疑惑的点

### 25.1 Attention.forward() 会做 QKV projection 吗？

不会。

Q/K/V projection 通常在模型层自己的 attention module 中完成，`Attention.forward()` 收到的是已经投影后的 query、key、value。

### 25.2 Attention.forward() 如何知道当前 layer_name？

`Attention` 初始化时保存：

```text
self.layer_name = prefix
```

forward 时把 `self.layer_name` 传给 unified op，再由 `get_attention_context(layer_name)` 查 context。

### 25.3 attn_metadata 是显式传给模型的吗？

不是。

它通过 `set_forward_context()` 放进当前 forward context，attention layer 内部再取。

### 25.4 KV cache 一定在 impl.forward 内更新吗？

不一定。

```text
forward_includes_kv_cache_update=True：impl.forward 内处理；
forward_includes_kv_cache_update=False：Attention.forward 先调 unified_kv_cache_update。
```

### 25.5 slot_mapping 和 attn_metadata.slot_mapping 是同一个东西吗？

普通 decoder 路径里通常是一致语义，但 wrapper 可能改写。

例如 cross attention 会把 `attn_metadata.slot_mapping` 改成 encoder KV slot mapping，所以不能简单认为所有路径都使用 `ForwardContext.slot_mapping[layer_name]`。

### 25.6 profiling run 中 attn_metadata 为什么可能是 None？

某些 profile / dummy run 不需要真实 attention 计算，只需要模拟输出 shape 或最大内存行为。

例如 FlashAttention 中：

```text
attn_metadata is None → output.fill_(0)
```

### 25.7 output 为什么由 Attention.forward 分配？

这样可以统一 output shape、支持 custom op mutates_args、支持 output quantization，并减少 backend 分配差异。

### 25.8 KV connector 是包整个模型还是每层 attention？

两者都有：

```text
maybe_get_kv_connector_output：包整个 model forward；
maybe_transfer_kv_layer：包每个 unified_attention_with_output。
```

---

## 26. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心输入 | 核心输出 | 作用 |
|---|---|---|---|---|
| 构造 metadata | `_build_attention_metadata()` | block table、slot mapping、seq_lens | `attn_metadata` | 给每层 attention 准备 backend metadata |
| 注入 context | `set_forward_context()` | `attn_metadata`、slot_mapping、batch desc | `ForwardContext` | forward 作用域隐式参数区 |
| 模型 forward | `_model_forward()` | input_ids、positions、inputs_embeds | hidden states | 调用模型本体 |
| 层内投影 | model attention module | hidden states | query、key、value | 生成 Q/K/V |
| attention 入口 | `Attention.forward()` | query、key、value | attention output | reshape、量化、调 unified op |
| 取上下文 | `get_attention_context()` | layer_name | metadata、layer、kv_cache、slot_mapping | 找回当前层运行时状态 |
| KV 写入 | `unified_kv_cache_update()` | key、value、slot_mapping | dummy dep | separate KV update backend 先写 cache |
| 统一 attention op | `unified_attention_with_output()` | Q/K/V、output、layer_name | 填充 output | 调 backend impl.forward |
| KV transfer hook | `maybe_transfer_kv_layer` | layer_name、kv_cache、metadata | load/save KV | 每层 attention 前后处理 KV connector |
| backend 执行 | `impl.forward()` | Q/K/V、KV cache、metadata | output | 调具体 kernel / wrapper |
| logits | `compute_logits()` | hidden_states[logits_indices] | logits | forward 后采样前计算 logits |

---

## 27. 总结

Attention forward 可以压缩成下面这条线：

```text
ModelRunner 准备 attn_metadata / slot_mapping
  → set_forward_context(...)
  → 模型 forward
  → Q/K/V projection
  → Attention.forward()
  → get_attention_context(layer_name)
  → optional unified_kv_cache_update()
  → unified_attention_with_output()
  → backend impl.forward()
  → attention output
```

它的分层职责是：

```text
GPUModelRunner：决定本轮 batch 怎么跑，准备 metadata / slot mapping；
ForwardContext：把运行时 metadata 暴露给模型内部 attention layer；
Attention.forward：统一 Q/K/V reshape、量化、KV update 调度和 unified op 调用；
backend impl：解释 backend metadata 和 KV cache layout，执行真实 attention kernel；
KV connector hook：在 layer 级别等待 / 保存远端或外部 KV。
```

如果只记住最小心智模型：

```text
Attention.forward 是一个运行时桥接层：左边接模型算出的 Q/K/V，右边接 backend kernel，中间通过 ForwardContext 拿到 metadata、slot_mapping 和 KV cache。
```