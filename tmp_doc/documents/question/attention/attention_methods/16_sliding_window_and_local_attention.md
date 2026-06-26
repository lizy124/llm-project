# 16. Sliding Window / Local Attention 如何工作？

源码位置：

- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本文用于梳理 sliding window / local attention 的 mask 语义、模型配置来源、metadata 表示和 backend 支持约束。

---

## 1. 本文要回答的问题

```text
sliding window attention 和 full causal attention 有什么区别？
模型配置如何声明 sliding_window？
attention metadata 如何传递窗口大小？
KV cache 是否仍然保存窗口外 token？
backend 如何支持或拒绝 sliding window？
chunked prefill / prefix cache / cascade attention 与 sliding window 如何交互？
```

---

## 2. 核心结论占位

占位：后续补充 sliding window / local attention 的执行链路。

```text
Sliding window / local attention 属于 attention mask 和可见范围语义，它限制每个 query 可访问的历史 token 范围；vLLM 需要把窗口信息写入 metadata，并依赖 backend 支持对应 kernel 语义。
```

---

## 3. 待梳理源码点

```text
sliding_window config
AttentionSpec
backend support checks
metadata fields
attention mask semantics
seq_lens / window clipping
model-specific sliding window config
```

---

## 4. 主链路占位

```text
ModelConfig / HF config sliding_window
  → AttentionSpec
  → backend selection / capability check
  → AttentionMetadata
  → backend local attention kernel
```

---

## 5. 后续重点占位

```text
- 窗口外 KV 是否保留与是否参与 attention 是两个问题；
- 不同 backend 对 sliding window 的支持差异；
- sliding window 对 prefix cache 命中长度的影响；
- 局部 attention 与 long-context 模型配置的关系。
```
