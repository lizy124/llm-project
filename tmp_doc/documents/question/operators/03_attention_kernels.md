# 03. Attention 算子如何支撑 prefill / decode？

源码位置：

- `vllm/vllm/attention/`
- `vllm/vllm/attention/backends/`
- `vllm/vllm/model_executor/layers/attention.py`
- `vllm/csrc/attention/`

这个问题关注：PagedAttention、FlashAttention、FlashInfer、FlashMLA、Triton attention 等 attention kernel 如何被选择、如何消费 metadata、如何支撑 prefill / decode / cascade / sliding window / MLA 等路径。

---

## 1. 一句话回答

Attention kernel 是 vLLM 推理性能最核心的算子族，负责在不同 batch 状态下高效读取 KV cache 并完成 attention 计算。

最小链路是：

```text
ModelRunner 构造 attention metadata
  → Attention layer forward
  → selected attention backend
  → prefill / decode kernel
  → attention output
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. prefill attention 和 decode attention 的 kernel 差异；
2. PagedAttention 如何使用 block table 和 slot mapping；
3. FlashAttention / FlashInfer / FlashMLA 各自适用场景；
4. attention metadata 如何进入 backend；
5. cascade attention、sliding window、local attention 如何影响 kernel；
6. dtype、head size、block size、hardware capability 如何影响选择；
7. fallback 和性能 debug 方法。
```

---

## 3. 需要串起来的主线

```text
SchedulerOutput
  → ModelRunner._prepare_inputs()
  → ModelRunner._build_attention_metadata()
  → model attention layer
  → backend impl
  → attention kernel
```

---

## 4. 后续补充重点

```text
- block table / seq_lens / query_start_loc / slot_mapping；
- prefill 和 decode 混合 batch；
- FlashInfer / FlashAttention backend selection；
- MLA / HMA / KV cache layout；
- CUDA Graph capture 下的 shape 稳定性。
```
