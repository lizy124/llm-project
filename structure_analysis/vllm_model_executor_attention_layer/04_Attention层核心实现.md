# 04 Attention 层核心实现

本篇重点梳理 `vllm/model_executor/layers/attention/attention.py`。这是模型层 Attention 的核心文件，也是模型 forward 与 V1 attention runtime 之间的桥。

## 1. 文件定位

路径：

```text
vllm/model_executor/layers/attention/attention.py
```

关键代码锚点：

- `Attention` 类：`code/vllm/vllm/model_executor/layers/attention/attention.py:178`
- `Attention.__init__()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:190`
- 注册到 `static_forward_context`：`code/vllm/vllm/model_executor/layers/attention/attention.py:397`
- `Attention.forward()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:438`
- `Attention.get_kv_cache_spec()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:567`
- `get_attention_context()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`
- `unified_kv_cache_update()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`
- `unified_attention_with_output()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:736`

## 2. Attention 类的职责

`Attention` 是模型定义中的 `nn.Module`。它封装一层 attention 的运行时逻辑。

它负责：

1. 保存 attention 结构参数；
2. 选择 attention backend；
3. 创建 backend implementation；
4. 初始化 KV cache quant scale；
5. 保存/绑定 KV cache tensor；
6. 注册自身到 `static_forward_context`；
7. 声明当前层的 KVCacheSpec；
8. forward 时获取 ForwardContext；
9. 更新 KV cache；
10. 调用 backend 计算 attention；
11. 处理量化、sink、prefix-lm、kv-sharing 等特殊逻辑。

## 3. Attention 初始化时保存的关键参数

常见参数包括：

| 参数 | 作用 |
|---|---|
| `num_heads` | query heads 数 |
| `head_size` | 每个 head 的维度 |
| `scale` | attention scale |
| `num_kv_heads` | KV heads 数，GQA/MQA 时小于 query heads |
| `alibi_slopes` | ALiBi 相关参数 |
| `sliding_window` | sliding window attention 大小 |
| `kv_cache_dtype` | KV cache dtype，如 fp8/int8 等 |
| `blocksparse_params` | block sparse attention 参数 |
| `logits_soft_cap` | logits soft cap |
| `attn_type` | decoder/self/cross/prefix 等 attention 类型 |
| `prefix` | layer name，用于 forward context 查找 |
| `use_mla` | 是否 MLA attention |
| `sinks` | attention sink 相关参数 |
| `per_layer_sliding_window` | 每层 sliding window |

初始化时不仅记录这些参数，还会根据这些参数选择 backend。

## 4. backend 选择

Attention 初始化时会调用：

```text
self.get_attn_backend()
```

方法位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:564`。

它最终会调用 V1 attention selector：

```text
vllm/v1/attention/selector.py:get_attn_backend
```

选择 backend 时会考虑：

- 当前平台：CUDA/ROCm/CPU/XPU；
- head size；
- dtype；
- KV cache dtype；
- block size；
- 是否 MLA；
- 是否 sink attention；
- 是否支持 mm prefix；
- 是否需要 non-causal；
- 是否支持 per-head quant scales；
- 是否支持 batch invariance；
- 是否支持 KV connector；
- 用户是否指定 attention backend。

## 5. AttentionImpl 创建

backend 选定后：

```text
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

`AttentionImpl` 是该 backend 对单层 attention 的实现对象。

`Attention` 自己不直接写 CUDA kernel，而是把实际 attention 计算交给 `self.impl.forward(...)` 或注册的 unified custom op。

## 6. static_forward_context 注册

Attention 初始化时会把自己注册到：

```text
compilation_config.static_forward_context[prefix] = self
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:397`。

### 为什么这么做

因为模型 forward 期间，很多 runtime 信息不适合通过普通参数一层层传递。vLLM 使用：

```text
layer name -> layer object
```

的方式，在 `ForwardContext` 中找回当前 layer。

这使得 custom op、torch compile、CUDA graph 场景下仍能通过稳定的 layer name 找到 layer 的 backend、kv cache、quant scale 等信息。

## 7. Attention.forward 主流程

`Attention.forward()` 位于 `code/vllm/vllm/model_executor/layers/attention/attention.py:438`。

输入通常是：

```text
query, key, value
```

高层流程：

```text
1. 如需要，计算 KV scales
2. reshape query/key/value 到 backend 需要的形状
3. 获取当前平台是否使用 opaque custom op
4. 如果 backend 不在 forward 中更新 KV cache：
   - 调 unified_kv_cache_update
5. 调 unified_attention_with_output 或直接 self.impl.forward
6. 返回 attention output
```

## 8. KV cache update

有些 backend 的 forward 包含 KV cache update，有些则分离。

当分离时，Attention.forward 会调用：

```text
torch.ops.vllm.unified_kv_cache_update(...)
```

或 Python helper：

```text
unified_kv_cache_update(...)
```

方法位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`。

它会：

1. 通过 `get_attention_context(layer_name)` 获取当前层 context；
2. 找到 layer 的 KV cache tensor；
3. 找到当前 token 的 slot mapping；
4. 调 backend 的 `do_rope_and_kv_cache_update()` 或相关实现；
5. 把本步 key/value 写入 KV cache。

## 9. unified_attention_with_output

`unified_attention_with_output()` 位于 `code/vllm/vllm/model_executor/layers/attention/attention.py:736`。

职责：

1. 从 ForwardContext 获取 attention context；
2. 找到当前 layer 的 metadata；
3. 找到 KV cache；
4. 调用 `self.impl.forward(...)`；
5. 将结果写入 output。

抽象形态：

```text
Attention.forward
  ↓
unified_attention_with_output
  ↓
get_attention_context
  ↓
self.impl.forward(...)
  ↓
backend kernel/custom op
```

## 10. get_attention_context

`get_attention_context()` 位于 `code/vllm/vllm/model_executor/layers/attention/attention.py:649`。

它从 ForwardContext 里取出：

- 当前 layer 的 Attention 对象；
- 当前 layer 的 KV cache；
- 当前 batch 的 attention metadata；
- 当前 batch 的 slot mapping；
- backend-specific metadata。

这是 Attention 与 GPUModelRunner/ForwardContext 衔接的核心方法。

## 11. KVCacheSpec 声明

`Attention.get_kv_cache_spec()` 位于 `code/vllm/vllm/model_executor/layers/attention/attention.py:567`。

它根据当前 Attention 层的属性生成 KV cache 规格，例如：

- full attention；
- sliding window attention；
- MLA attention；
- encoder-only attention；
- cross attention；
- sink attention；
- quantized KV cache。

这些 spec 会被 EngineCore/Worker 收集，用于计算 KV cache 内存、分组、block size、tensor shape。

## 12. KV quant scales

Attention 文件里有几个量化相关 helper：

- `_init_kv_cache_quant()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:122`
- `maybe_calc_kv_scales()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:614`
- `Attention.calc_kv_scales()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:532`
- `Attention.process_weights_after_loading()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:550`

它们处理：

- q/k/v/prob scale；
- FP8 KV cache；
- per-token-head scale；
- checkpoint 中加载 scale；
- 未加载 scale 时默认值。

## 13. custom op 注册

文件中通过 `direct_register_custom_op` 注册了几个关键 op：

```text
maybe_calc_kv_scales
unified_kv_cache_update
unified_attention_with_output
```

这些 op 让 attention forward 可以被 torch compile / CUDA graph / custom dispatch 更好处理，同时把 Python 层和底层 kernel 衔接起来。

## 14. 其他 Attention 变体

除了主 `attention.py`，还有：

```text
mla_attention.py
cross_attention.py
encoder_only_attention.py
prefill_prefix_lm_attention.py
```

它们处理特定模型结构或 attention 类型：

- MLA：DeepSeek 等模型的 latent attention；
- cross attention：encoder-decoder；
- encoder-only attention：embedding/encoder 模型；
- prefix-lm attention：局部非因果 / multimodal prefix 场景。

## 15. 一句话总结

`Attention` 是模型层的 glue layer：初始化时声明 backend 与 KV cache spec，注册到 static context；forward 时从 ForwardContext 取 metadata/slot mapping/KV cache，然后通过统一 custom op 或 direct backend call 完成 KV cache update 和 attention 计算。
