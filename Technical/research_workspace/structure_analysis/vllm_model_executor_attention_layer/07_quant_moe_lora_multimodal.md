# 07 量化、MoE、LoRA、多模态与模型层关系

本篇梳理 vLLM 模型执行与 Attention 层中几个重要扩展能力：量化、KV cache quant、MoE、LoRA、多模态。它们并不是独立于模型执行链路之外，而是深度嵌入模型加载、layer 初始化、forward context、GPUModelRunner 和 backend/kernel 选择中。

## 1. 总览

这些能力分别影响不同层级：

| 能力 | 主要影响位置 | 影响内容 |
|---|---|---|
| 权重量化 | `model_executor/layers/quantization/`、model loader | 权重加载、linear/moe kernel、scale、packing |
| KV cache 量化 | `attention.py`、`kv_cache_interface.py`、csrc cache kernels | KV cache dtype、scale、page size、cache update kernel |
| MoE | `fused_moe/`、model loader、distributed EP | expert 权重加载、router、expert parallel、MoE kernel |
| LoRA | `vllm/lora/`、linear layer、forward context | 动态 adapter、layer 注册、runtime dispatch |
| 多模态 | multimodal registry、GPUModelRunner、attention metadata | encoder inputs、mm embeddings、prefix-lm/non-causal metadata |

## 2. 权重量化

量化相关路径：

```text
vllm/model_executor/layers/quantization/
vllm/model_executor/layers/linear.py
vllm/model_executor/layers/fused_moe/
vllm/model_executor/model_loader/
vllm/platforms/
csrc/*quant*
```

权重量化主要影响：

1. 权重文件加载方式；
2. 参数 tensor 的 shape/packing；
3. scale/zero-point 的保存和加载；
4. linear kernel 的选择；
5. MoE kernel 的选择；
6. 是否支持 fused operation；
7. CUDA graph / torch compile 的兼容性。

模型 loader 在加载权重时需要识别量化格式；具体 linear/MoE 层则通过 quant method 决定 forward 如何执行。

## 3. KV cache 量化

KV cache 量化不同于权重量化。它影响 attention 的历史 K/V 存储。

相关路径：

```text
vllm/model_executor/layers/attention/attention.py
vllm/model_executor/layers/quantization/kv_cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/attention/backend.py
csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/cache_kernels_fused.cu
csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
```

## 4. KVQuantMode

`KVQuantMode` 定义在 `code/vllm/vllm/v1/kv_cache_interface.py:33`。

它用于描述 KV cache 的量化模式，例如：

- 不量化；
- FP8 per tensor；
- INT8 per token/head；
- FP8 per token/head；
- NVFP4。

这些模式影响：

- `AttentionSpec.page_size_bytes()`；
- KV cache tensor dtype；
- 是否需要额外 scale tensor；
- cache update kernel；
- attention backend 是否支持；
- csrc kernel 调用路径。

## 5. Attention 中的 KV cache quant

`attention.py` 中和 KV quant 相关的关键函数：

| 函数 | 位置 | 作用 |
|---|---|---|
| `_init_kv_cache_quant()` | `code/vllm/vllm/model_executor/layers/attention/attention.py:122` | 初始化 q/k/v/prob scale |
| `maybe_calc_kv_scales()` | `code/vllm/vllm/model_executor/layers/attention/attention.py:614` | 必要时运行时计算 scale |
| `Attention.calc_kv_scales()` | `code/vllm/vllm/model_executor/layers/attention/attention.py:532` | 计算当前层 KV scales |
| `Attention.process_weights_after_loading()` | `code/vllm/vllm/model_executor/layers/attention/attention.py:550` | 权重加载后处理 scale |

KV cache 量化大体流程：

```text
Attention 初始化
  ↓
_init_kv_cache_quant
  ↓
加载 checkpoint 中的 scale 或设置默认 scale
  ↓
get_kv_cache_spec 中声明 kv_quant_mode
  ↓
KV cache tensor 按量化模式分配
  ↓
forward 中 maybe_calc_kv_scales
  ↓
unified_kv_cache_update 使用 scale 写入 quantized KV
```

## 6. MoE 与模型执行层

MoE 相关路径：

```text
vllm/model_executor/layers/fused_moe/
vllm/model_executor/layers/fused_moe/runner/
vllm/model_executor/model_loader/default_loader.py
vllm/distributed/eplb/
vllm/distributed/parallel_state.py
csrc/*moe*
```

MoE 影响两个阶段：

### 6.1 加载阶段

MoE expert 权重通常非常大，并且在 expert parallel 下每个 rank 只需要加载部分 expert。

默认 loader 会根据 expert parallel rank 过滤权重，避免每个 rank 加载全部 expert。

典型影响：

```text
model_loader
  ↓
expert parallel filter
  ↓
只加载本 rank 负责的 expert weights
```

### 6.2 forward 阶段

MoE forward 通常包括：

```text
hidden states
  ↓
router / gating
  ↓
top-k expert selection
  ↓
token dispatch
  ↓
expert FFN kernel
  ↓
combine outputs
```

vLLM 中 fused MoE 层会尽量调用高性能 fused kernel 或 custom op。

## 7. MoE 与 ForwardContext

MoE runner 也会使用 `ForwardContext` 和静态 layer 注册机制。

这点和 Attention 类似：

```text
layer 初始化时注册到 static_forward_context
  ↓
forward/custom op 中通过 layer_name 找回 layer
  ↓
执行 backend/kernel
```

这种模式让 Attention、MoE、LoRA 都能在 torch compile/custom op 场景中保留 Python layer 对象与运行时 metadata 的连接。

## 8. LoRA 与模型执行层

LoRA 相关路径：

```text
vllm/lora/
vllm/lora/layers/
vllm/lora/layers/base_linear.py
vllm/model_executor/layers/linear.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/executor/abstract.py
```

LoRA 主要影响 linear 层，而不是 Attention kernel 本身。但 attention 中的 q/k/v/o projection 也是 linear，因此 LoRA 可以作用到 attention 投影层。

## 9. LoRA 的运行时管理

在引擎层，LoRA 可以动态添加/移除：

```text
AsyncLLM.add_lora/remove_lora
  ↓
EngineCore
  ↓
Executor.add_lora/remove_lora
  ↓
Worker/GPUModelRunner
  ↓
模型层 LoRA manager
```

Executor 抽象中有：

- `add_lora()`；
- `remove_lora()`；
- `list_loras()`；
- `pin_lora()`。

## 10. LoRA 与 static_forward_context

LoRA 层也会注册到静态上下文。

这意味着 LoRA forward 可以像 Attention 一样，通过 `ForwardContext.no_compile_layers[layer_name]` 找到当前 layer 对象，并执行 adapter 逻辑。

这样可以同时支持：

- 动态 LoRA；
- torch compile；
- CUDA graph；
- 多请求混合 LoRA；
- batch 内不同 request 使用不同 adapter。

## 11. 多模态与模型执行层

多模态相关路径：

```text
vllm/multimodal/
vllm/model_executor/models/* 多模态模型
vllm/v1/worker/gpu_model_runner.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/encoder_cache_manager.py
vllm/v1/attention/backend.py
```

多模态影响：

1. 输入处理：图片/视频/音频等输入被预处理成 multimodal embeddings 或 encoder inputs；
2. Scheduler：需要 encoder compute budget 和 encoder cache；
3. GPUModelRunner：需要执行 multimodal encoder 或使用 cached encoder outputs；
4. 模型 forward：需要把 text tokens 和 multimodal embeddings 对齐；
5. Attention metadata：某些 prefix-lm / mm prefix 场景需要局部非因果 attention 区间。

## 12. 多模态与 Attention metadata

`CommonAttentionMetadata` 中包含 multimodal prefix 相关字段，例如 mm request/doc ranges。

用途：

- PrefixLM 场景中，某些 prefix token 可以双向 attention；
- 文本 token 仍然保持因果 attention；
- backend 需要知道哪些区间是非因果/双向区间。

如果 backend 不支持 `mm_prefix` 或 non-causal，则需要回退或禁用某些优化。

## 13. 多模态与 KV cache

多模态模型可能带来额外 cache：

- encoder cache；
- multimodal feature cache；
- cross attention cache；
- decoder self-attention KV cache。

因此，调度层和模型层不只处理 decoder-only full attention，还要处理 encoder-decoder、cross attention、encoder-only attention 等 spec。

相关 spec：

- `EncoderOnlyAttentionSpec`；
- `CrossAttentionSpec`；
- `FullAttentionSpec`；
- multimodal prefix/non-causal 标记。

## 14. 这些扩展能力如何共同影响 backend 选择

backend 选择要同时满足：

- dtype；
- KV cache dtype；
- quant mode；
- head size；
- block size；
- sliding window；
- MLA；
- sink；
- sparse；
- non-causal；
- mm prefix；
- per-head scales；
- batch invariance；
- KV connector。

因此同一个模型在不同配置下可能选择不同 attention backend。

## 15. 调试定位

| 问题 | 优先看 |
|---|---|
| 量化模型加载失败 | `model_loader`、`layers/quantization`、具体模型 `load_weights()` |
| KV cache fp8 输出异常 | `attention.py` quant scale、`kv_cache.py`、`KVQuantMode`、cache kernels |
| MoE expert 权重缺失 | default loader expert filter、EP rank、模型 `load_weights()` |
| MoE forward 性能/错误 | `fused_moe/runner`、MoE custom ops、expert parallel config |
| LoRA 不生效 | LoRA manager、base_linear、request lora id、Executor add_lora |
| 多模态 attention 异常 | multimodal registry、encoder cache、mm prefix metadata、backend supports_mm_prefix |

## 16. 一句话总结

量化改变权重和 KV cache 的存储/计算格式，MoE 改变模型 FFN 的加载和执行方式，LoRA 改变 linear 层的动态 adapter 路径，多模态改变输入和 attention metadata；这些能力最终都会汇入 GPUModelRunner、ForwardContext、AttentionBackend 和底层 kernel 选择。
