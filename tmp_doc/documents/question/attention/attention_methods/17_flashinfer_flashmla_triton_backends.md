# 17. FlashInfer / FlashMLA / Triton backend 如何区分？

源码位置：

- `code/vllm/vllm/v1/attention/backends/flashinfer.py`
- `code/vllm/vllm/v1/attention/backends/flashmla.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/v1/attention/selector.py`
- `code/vllm/vllm/platforms/`

本文用于梳理 vLLM 中常见 attention backend 的实现差异、适用场景和选择条件。

---

## 1. 本文要回答的问题

```text
FlashInfer backend 适合什么场景？
FlashMLA backend 和 MLA 模型有什么关系？
Triton backend 什么时候作为 fallback 或特化路径？
backend selector 如何在它们之间选择？
这些 backend 的 metadata / KV layout 有什么差异？
它们和 FlashAttention family 是什么关系？
```

---

## 2. 核心结论占位

占位：后续补充 FlashInfer / FlashMLA / Triton backend 对比。

```text
FlashInfer / FlashMLA / Triton 是 vLLM attention backend 或 kernel 路径的不同实现：它们不是 attention 语义本身，而是执行同一类 prefill / decode / paged KV 工作的不同后端。
```

---

## 3. 待梳理源码点

```text
FlashInferBackend
FlashMLABackend
Triton attention backends
selector decision tree
backend metadata builder
kernel call path
platform support checks
```

---

## 4. 对比维度占位

```text
FlashInfer：
  常见于高性能 paged attention / decode / prefill backend 路径。

FlashMLA：
  面向 MLA / DeepSeek 类模型的专用 attention backend。

Triton：
  用于 fallback、平台特化或特定 kernel 实现。

torch SDPA / native：
  可能作为简单模型、CPU 或 fallback 路径。
```

---

## 5. 后续重点占位

```text
- selector 如何排序和 fallback；
- 每个 backend 的 metadata dataclass；
- 每个 backend 对 dtype、head size、block size、sliding window 的限制；
- backend 选择如何影响 CUDA graph / compile。
```
