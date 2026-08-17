# Compilation and CUDA Graph 文档目录

本目录用于梳理 vLLM V1 中 torch.compile、CUDA graph capture / replay、cudagraph batch、padding、shape 固定化和运行时选择逻辑。

编译和 CUDA graph 不是单一优化开关，而是横跨：

```text
配置
  → model load / warmup
  → compile wrapper
  → cudagraph capture
  → fixed shape / padding
  → attention metadata capture path
  → ModelRunner runtime mode
  → graph replay or eager fallback
  → profile / debug / stats
```

它要回答的核心问题是：

```text
vLLM 如何把动态 batch 的推理请求，
尽量映射到可编译、可 capture、可 replay 的固定形态，
同时在不满足条件时安全 fallback 到 eager 路径。
```

---

## 阅读顺序建议

```text
compilation_and_cuda_graph_overview.md
  → 01_compilation_cuda_graph_role.md
  → 02_compile_config_and_runtime_modes.md
  → 03_warmup_and_capture_lifecycle.md
  → 04_batch_padding_and_shape_stability.md
  → 05_cudagraph_dispatch_flow.md
  → 06_attention_metadata_capture.md
  → 07_model_forward_compile_wrapper.md
  → 08_sampler_and_output_interaction.md
  → 09_parallelism_and_cudagraph.md
  → 10_limitations_and_fallbacks.md
  → 11_debugging_and_metrics.md
```

如果只想先抓主线，可以先读：

```text
compilation_and_cuda_graph_overview.md
  → 02_compile_config_and_runtime_modes.md
  → 03_warmup_and_capture_lifecycle.md
  → 04_batch_padding_and_shape_stability.md
  → 05_cudagraph_dispatch_flow.md
```

---

## 文档定位

```text
compilation_and_cuda_graph_overview.md：
  总览主文档，建立 vLLM 编译和 CUDA graph 机制的全局图。

01-11：
  按问题拆开的专题文档，后续逐篇补源码细节。
```
