# LoRA and Adapters 文档目录

本目录用于梳理 vLLM 中 LoRA / adapter 相关机制。

LoRA 不是单纯的权重加载功能，而是横跨：

```text
用户请求
  → LoRARequest
  → Engine / Scheduler
  → Executor / Worker
  → ModelRunner active LoRA state
  → LoRA manager / adapter cache
  → LoRA layer / punica wrapper
  → model forward
  → output
```

它要回答的核心问题是：

```text
vLLM 如何加载多个 LoRA adapter，
如何在请求级别选择 LoRA，
如何在 batch 内混合不同 LoRA，
如何在 Worker / ModelRunner 中激活和切换 LoRA，
以及 LoRA 如何与量化、并行、CUDA graph、spec decode、多模态和生命周期控制交互。
```

---

## 阅读顺序建议

```text
lora_and_adapters_overview.md
  → 01_lora_role.md
  → 02_lora_request_and_engine_flow.md
  → 03_lora_manager_and_cache.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 06_batch_mixed_lora_execution.md
  → 07_lora_loading_and_weight_mapping.md
  → 08_lora_and_quantization.md
  → 09_lora_and_parallelism.md
  → 10_lora_lifecycle_and_control.md
  → 11_lora_limitations_and_debugging.md
```

如果只想先抓主线，可以先读：

```text
lora_and_adapters_overview.md
  → 02_lora_request_and_engine_flow.md
  → 04_worker_model_runner_lora_state.md
  → 05_lora_layer_injection.md
  → 06_batch_mixed_lora_execution.md
```

---

## 文档定位

```text
lora_and_adapters_overview.md：
  总览主文档，建立 LoRA 在 vLLM 中从请求到 forward 的全局图。

01-11：
  按问题拆开的专题文档，后续逐篇补源码细节。
```
