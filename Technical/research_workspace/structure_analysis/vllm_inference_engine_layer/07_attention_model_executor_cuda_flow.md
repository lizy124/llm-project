# 07 Attention、ModelExecutor 与 CUDA/C++ Kernel 链路

本篇梳理从 `GPUModelRunner` 调用模型 forward，到模型中的 Attention，再到底层 backend 和 csrc kernel 的链路。

## 1. 位置关系

```text
GPUModelRunner._model_forward()
  ↓
具体模型 forward
  ↓
vllm/model_executor/models/*
  ↓
vllm/model_executor/layers/attention/attention.py
  ↓
vllm/v1/attention/backend.py + backends/ops
  ↓
vllm/_custom_ops.py / torch custom op / triton / flash-attn / flashinfer
  ↓
csrc/* CUDA/C++/CPU kernels
```

## 2. model_executor 的职责

`vllm/model_executor/` 是模型执行层，负责：

- 加载模型权重；
- 定义模型结构；
- 定义 attention、MLP、MoE、rotary embedding、norm、activation 等层；
- 支持量化；
- 支持 LoRA；
- 支持模型特定 forward；
- 暴露 KVCacheSpec；
- 与 attention backend 衔接。

典型目录：

```text
vllm/model_executor/model_loader/
vllm/model_executor/models/
vllm/model_executor/layers/
vllm/model_executor/layers/attention/
vllm/model_executor/layers/quantization/
vllm/model_executor/layers/fused_moe/
```

## 3. 模型加载链路

粗略链路：

```text
Worker.load_model()
  ↓
GPUModelRunner.load_model()
  ↓
model_loader 加载权重
  ↓
model_executor/models 中具体模型类实例化
```

`Worker.load_model()` 位于 `code/vllm/vllm/v1/worker/gpu_worker.py:349`。

`GPUModelRunner.load_model()` 位于 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`。

## 4. GPUModelRunner 如何调用模型

在 `execute_model()` 中，核心 forward 调用位于 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4320`：

```text
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

在调用前，它会设置 forward context：

```text
set_forward_context(
  attn_metadata,
  vllm_config,
  num_tokens=...,
  cudagraph_runtime_mode=...,
  batch_descriptor=...,
  ubatch_slices=...,
  slot_mapping=...,
)
```

这一步非常关键，因为模型里的 Attention 层不直接从参数收到所有 metadata，而是通过 forward context 读取当前 batch 的 attention metadata、slot mapping、KV cache 等。

## 5. Attention 模块

模型层 attention 定义在 `code/vllm/vllm/model_executor/layers/attention/attention.py:178`。

该类的注释说明它接收 query/key/value，并负责：

- KV cache 读写；
- 调用 attention backend；
- 处理 quantization scales；
- 与 forward context 绑定；
- 生成 KVCacheSpec。

关键方法：

| 方法 | 作用 |
|---|---|
| `__init__()` | 初始化 attention backend、head size、num heads、kv cache 配置等 |
| `forward()` | attention 前向主入口 |
| `calc_kv_scales()` | 计算 KV quant scales |
| `process_weights_after_loading()` | 权重加载后处理 |
| `get_attn_backend()` | 获取 backend class |
| `get_kv_cache_spec()` | 返回该层需要的 KV cache spec |

## 6. Attention.forward 的关键点

`Attention.forward()` 位于 `code/vllm/vllm/model_executor/layers/attention/attention.py:438`。

它不是普通 PyTorch attention。它会依赖 forward context 中的数据：

- `attn_metadata`；
- `slot_mapping`；
- 当前层的 KV cache；
- backend-specific metadata；
- 是否需要更新 KV cache；
- quant scale；
- attention type。

因此，理解 Attention 时一定要同时看：

```text
GPUModelRunner._build_attention_metadata()
GPUModelRunner._get_slot_mappings()
set_forward_context(...)
Attention.forward()
```

## 7. get_attention_context

`get_attention_context()` 在 `code/vllm/vllm/model_executor/layers/attention/attention.py:649`。

它会从 forward context 中取出：

- 当前 layer 的 KV cache；
- attention metadata；
- attention layer 对象；
- slot mapping。

这说明 attention layer 和 model runner 之间的隐式契约是 forward context。

## 8. V1 AttentionBackend 抽象

`AttentionBackend` 定义在 `code/vllm/vllm/v1/attention/backend.py:55`。

它定义每种 backend 需要声明的能力。

常见接口：

| 接口 | 作用 |
|---|---|
| `get_supported_kernel_block_sizes()` | 支持哪些 block size |
| `get_name()` | backend 名称 |
| `get_impl_cls()` | backend 实现类 |
| `get_builder_cls()` | metadata builder 类 |
| `get_kv_cache_shape()` | KV cache tensor shape |
| `get_kv_cache_block_dim()` | block 维度 |
| `get_kv_cache_stride_order()` | cache stride/layout |
| `supports_head_size()` | 是否支持某 head size |
| `supports_dtype()` | 是否支持 dtype |
| `supports_kv_cache_dtype()` | 是否支持 KV cache dtype |
| `supports_block_size()` | 是否支持 block size |
| `supports_attn_type()` | 是否支持 decoder/prefix/encoder 等类型 |
| `validate_configuration()` | 校验配置 |
| `get_required_kv_cache_layout()` | 指定需要的 KV cache layout |

## 9. AttentionMetadata 与 Builder

`AttentionMetadata`、`CommonAttentionMetadata`、`AttentionMetadataBuilder` 都在 `vllm/v1/attention/backend.py`。

它们用于把 batch 信息组织成 backend 可用格式。

`GPUModelRunner._build_attention_metadata()` 会调用 backend 的 builder 来构造 metadata。

metadata 通常包含：

- query lengths；
- sequence lengths；
- block tables；
- slot mapping；
- common prefix length；
- max query len；
- max seq len；
- cascade attention 信息；
- spec decode 信息。

## 10. AttentionImpl

`AttentionImplBase` 定义在 `code/vllm/vllm/v1/attention/backend.py:702`。

`AttentionImpl` 定义在 `code/vllm/vllm/v1/attention/backend.py:780`。

它们是具体 backend 实现 attention 的基类。

关键方法：

- `forward()`：执行 attention；
- `do_rope_and_kv_cache_update()`：可能融合 RoPE 和 KV cache update；
- `fused_output_quant_supported()`：是否支持 fused output quant；
- `fused_rope_kvcache_supported()`：是否支持 fused rope + kv cache。

## 11. KV cache update 与 attention forward 的两种模式

不同 backend 对 KV cache 更新的处理不同：

1. `forward_includes_kv_cache_update = True`
   - attention forward 内部完成 KV cache update。

2. `forward_includes_kv_cache_update = False`
   - KV cache update 和 attention forward 分离。
   - `GPUModelRunner.execute_model()` 中会检查是否存在这种 backend，并决定 slot mapping padding 方式。

相关代码在 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4185`。

## 12. csrc 层职责

`csrc/` 是 vLLM 的底层高性能实现，包含：

- paged attention CUDA kernel；
- cache write/read kernel；
- custom all-reduce；
- CPU attention/kernel；
- CUTLASS 扩展；
- quantized GEMM；
- MoE kernel；
- layernorm/activation/pos encoding 等。

典型路径：

```text
csrc/attention/
csrc/cpu/
csrc/libtorch_stable/attention/
csrc/libtorch_stable/cache_kernels*
csrc/cutlass_extensions/
csrc/custom_all_reduce*
```

## 13. Paged Attention 的核心思想

普通 attention 可能假设 KV cache 是连续存储的。

vLLM 的 paged attention 则通过 block table 间接寻址：

```text
request token position
  ↓
logical block index
  ↓
physical block id
  ↓
KV cache tensor 中的 page
  ↓
block 内 offset
```

这样多个 request 可以共享同一大 KV cache 池，且每个 request 的 KV blocks 不需要物理连续。

## 14. 从 SchedulerOutput 到 Attention Kernel 的数据变换

```text
SchedulerOutput
  包含：req ids、num scheduled tokens、new block ids、common prefix blocks
  ↓
GPUModelRunner._update_states
  更新 input_batch 中 request 状态
  ↓
GPUModelRunner._get_slot_mappings
  token -> KV slot
  ↓
GPUModelRunner._build_attention_metadata
  构造 backend metadata
  ↓
set_forward_context
  把 metadata/slot_mapping 放入 forward context
  ↓
模型 forward
  ↓
Attention.forward
  ↓
get_attention_context
  取出当前层 KV cache / metadata / slot_mapping
  ↓
backend impl forward
  ↓
custom op / CUDA kernel
```

## 15. 分布式通信与 Attention

Attention/模型 forward 中还会涉及：

- tensor parallel all-reduce；
- sequence parallel；
- pipeline parallel intermediate tensors；
- data parallel batch coordination；
- decode context parallel / prefill context parallel；
- expert parallel MoE；
- custom all-reduce kernel。

相关路径：

```text
vllm/distributed/
vllm/distributed/parallel_state.py
vllm/distributed/device_communicators/
csrc/custom_all_reduce*
```

## 16. Quantization 与 Attention

KV cache dtype 和 quantization 会影响 attention：

- fp16/bf16/fp8 KV cache；
- per-head quant scales；
- weight-only quant；
- fused output quant；
- fused rope + kv cache update。

相关路径：

```text
vllm/model_executor/layers/quantization/
vllm/model_executor/layers/attention/attention.py
vllm/v1/attention/backend.py
csrc/*fp8*
```

## 17. 常见调试问题定位

| 问题 | 优先看 |
|---|---|
| attention backend 选择不对 | `vllm/v1/attention/backends/registry.py`, backend supports/validate |
| KV cache shape 不对 | `AttentionBackend.get_kv_cache_shape`, `GPUModelRunner.initialize_kv_cache` |
| slot mapping 错误 | `GPUModelRunner._get_slot_mappings`, worker block table |
| attention metadata 错误 | `GPUModelRunner._build_attention_metadata`, backend builder |
| kernel OOM/illegal memory access | block table、slot mapping、KV cache tensor shape、csrc paged attention |
| logits 不对 | model forward、attention output、compute_logits、sampling |
| CUDA graph 相关问题 | `_determine_batch_execution_and_padding`, `capture_model`, `set_forward_context` |

## 18. 一句话总结

vLLM 的模型执行不是简单的 PyTorch forward：GPUModelRunner 先把 SchedulerOutput 转换成 slot mapping 和 attention metadata，通过 forward context 传给模型层 Attention，Attention backend 再根据 block table 间接访问 KV cache，最终调用 csrc/自定义 kernel 完成高性能 paged attention。
