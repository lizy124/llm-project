# vLLM V1 Attention 子系统逻辑梳理

源码位置：

- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/compilation/`

本文用于总览 vLLM V1 Attention 子系统，重点梳理 attention 在执行链路中的位置：backend 如何选择，metadata 如何构造，prefill / decode 如何区分，slot mapping / block table 如何连接 paged KV cache，attention forward 如何读写 KV，以及 cascade attention、KV connector hook、CUDA graph / compile 如何挂接。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本文按“先定角色，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式组织。

要回答的问题分成 10 组：

```text
1. Attention 子系统在 vLLM V1 中是哪一层？负责什么，不负责什么？
2. AttentionBackend / AttentionMetadataBuilder / AttentionImplBase 各自负责什么？
3. vLLM 如何选择 FlashAttention / FlashInfer / FlashMLA / Triton 等 backend？
4. GPUModelRunner 如何构造 attention metadata？
5. prefill / decode / chunked prefill / mixed batch / spec decode 的 metadata 有什么差异？
6. slot mapping / block table 如何把 request tokens 映射到 paged KV cache？
7. attention forward 如何读写 KV cache？
8. cascade attention / prefix cache 命中如何影响 attention？
9. KV connector hook 如何挂到 attention layer 边界？
10. CUDA graph / torch.compile 对 attention 路径有什么约束？
```

阅读顺序建议：

```text
attention_overview.md
  → 01_attention_role.md
  → 02_backend_selection.md
  → 03_attention_metadata_builder.md
  → 04_prefill_decode_metadata.md
  → 05_slot_mapping_and_block_table.md
  → 06_attention_forward_flow.md
  → 07_kv_cache_layout_and_backend.md
  → 08_cascade_attention.md
  → 09_attention_and_kv_connector_hooks.md
  → 10_cuda_graph_compile_interaction.md
```

如果要专门理解各种 attention 名词、算法家族和 backend 家族，再读：

```text
attention_methods/11_attention_variants_overview.md
  → attention_methods/12_flash_attention_family.md
  → attention_methods/13_paged_attention.md
  → attention_methods/14_mha_mqa_gqa.md
  → attention_methods/15_mla_attention.md
  → attention_methods/16_sliding_window_and_local_attention.md
  → attention_methods/17_flashinfer_flashmla_triton_backends.md
  → attention_methods/18_hma_and_kv_cache_layout.md
```

---

## 1. 一句话总览占位

占位：后续补充 Attention 子系统在 vLLM V1 主链路中的整体定位。

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._get_slot_mappings()
  → GPUModelRunner._build_attention_metadata()
  → set_forward_context(attn_metadata, ...)
  → model attention layers
  → AttentionBackend / AttentionImplBase
  → paged KV cache read / write
  → hidden states / logits / sampling
```

---

## 2. 核心角色占位

后续补充以下组件职责边界：

```text
AttentionBackend：
  定义 backend 能力、KV cache shape/layout、metadata builder、impl class。

AttentionMetadataBuilder：
  把 ModelRunner 的 batch / slot mapping / block table 转成 backend metadata。

AttentionMetadata：
  forward 时传给 attention impl 的运行时元数据。

AttentionImplBase：
  具体执行 attention forward 的实现入口。

Attention layer：
  模型层中的 attention 模块，负责调用 backend 并读写 KV cache。

ForwardContext：
  ModelRunner 在模型 forward 外围设置的上下文，连接 attention metadata、KV connector hook 和编译 / CUDA graph 状态。
```

---

## 3. 主链路占位

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(attn_metadata, ...)
  → _model_forward()
  → model attention layer
  → Attention.forward()
  → backend impl forward
  → KV cache update / attention output
```

---

## 4. 和已有专题的关系占位

后续重点串联：

```text
../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
../executor_worker_model_runner/07_model_forward_and_logits.md
../executor_worker_model_runner/09_worker_kv_cache_interaction.md
../kv_cache_transfer/07_worker_kv_connector_flow.md
../scheduler/05_prefix_and_external_kv_hits.md
../scheduler/06_kv_block_allocation_and_preemption.md
```

---

## 5. 文档定位占位

```text
attention_overview.md：
  总览主文档，适合快速建立 attention 子系统全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读 attention 主链路源码。

attention_methods/11-18：
  专门解释各种 attention 名词、结构、backend、KV layout 和优化策略。
```

---

## 6. 后续待补源码证据

占位：后续逐段补充源码位置、关键类、关键字段、关键状态迁移和例子。
