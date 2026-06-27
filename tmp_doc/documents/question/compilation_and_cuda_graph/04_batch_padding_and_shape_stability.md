# 04. 动态 batch 如何通过 padding 获得 shape 稳定性？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/forward_context.py`

本问题关注：vLLM 如何把动态请求 batch 映射到 CUDA graph 可 replay 的固定 shape。

---

## 1. 一句话回答

CUDA graph replay 要求 shape 稳定，因此 vLLM 会把真实 batch padding 到某个已 capture 的 shape，并让 padded token / request 不影响真实输出。

---

## 2. 为什么需要 padding

```text
真实 batch 每轮都可能变化：

- request 数不同；
- decode / prefill token 数不同；
- spec decode draft 数不同；
- multimodal / pooling / LoRA 状态不同；
- attention metadata shape 不同。

而 CUDA graph replay 需要固定 shape。
```

---

## 3. padding 要处理什么

```text
- input_ids；
- positions；
- slot_mapping；
- query_start_loc；
- seq_lens；
- block table；
- logits_indices；
- attention metadata；
- output hidden states / logits 切片。
```

---

## 4. padded token 的语义占位

```text
padded token 不属于真实请求，必须保证：

- 不写入有效 KV cache；
- 不参与真实 logits / sampling；
- 不影响 seq_lens；
- 不影响 output；
- 对 attention backend 是安全输入。
```

---

## 5. 一句话总结

```text
padding 是把动态 serving batch 变成 cudagraph 固定 shape 的桥。
```
