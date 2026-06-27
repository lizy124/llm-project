# 06. Attention metadata 在 CUDA graph capture 下有什么特殊处理？

源码位置：

- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：attention metadata builder 如何支持 cudagraph capture / replay，以及哪些字段必须固定或可更新。

---

## 1. 一句话回答

CUDA graph replay 不只要求 input tensor shape 固定，attention metadata 也要满足固定 shape 或可原地 update 的约束。

---

## 2. metadata 字段占位

需要梳理：

```text
- query_start_loc；
- seq_lens；
- max_seq_len；
- block_table_tensor；
- slot_mapping；
- positions；
- num_actual_tokens；
- max_query_len；
- encoder / cross attention 字段；
- spec decode / cascade / DCP 相关字段。
```

---

## 3. capture path 占位

```text
AttentionMetadataBuilder
  → build()
  → build_for_cudagraph_capture()
  → update_block_table / reusable metadata
```

后续补充不同 backend 的支持差异。

---

## 4. 为什么 metadata 会阻碍 capture

```text
- Python 对象结构动态变化；
- tensor shape 动态变化；
- block table 长度变化；
- seq_lens / query_start_loc shape 不稳定；
- backend metadata 中包含不可 capture 的 wrapper；
- prefill / decode / cascade metadata 分支不同。
```

---

## 5. 一句话总结

```text
attention metadata 是 cudagraph 能否 replay 的关键条件之一，因为它决定 kernel 看到的 batch 结构是否稳定。
```
