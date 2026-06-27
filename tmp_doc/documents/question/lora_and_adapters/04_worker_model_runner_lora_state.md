# 04. Worker / ModelRunner 如何维护 active LoRA 状态？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/lora/worker_manager.py`
- `code/vllm/vllm/forward_context.py`

本问题关注：每轮执行前，Worker / ModelRunner 如何知道当前 batch 中每个 request 使用哪个 LoRA。

---

## 1. 一句话回答

ModelRunner 会把 request 级 `LoRARequest` 整理成 batch 级 active LoRA 状态，并在 forward 前同步给 LoRA layer / punica wrapper。

---

## 2. InputBatch 状态占位

后续补充：

```text
- req_id_to_index；
- 每个 request 的 lora_request；
- lora_index_mapping；
- prompt mapping；
- token mapping；
- active_lora_ids；
- batch 内 LoRA 分布。
```

---

## 3. ModelRunner 执行前准备占位

```text
_update_states()
  → 新请求写入 lora_request

_prepare_inputs()
  → 当前 batch token / req_indices 已确定
  → set_active_loras()
  → LoRA layer 拿到 token 到 LoRA id 的映射
```

---

## 4. 为什么需要 batch 级 mapping

同一个 batch 可以包含：

```text
- 无 LoRA 请求；
- LoRA A 请求；
- LoRA B 请求；
- 相同 LoRA 的多个请求；
- 不同长度的 prefill / decode tokens。
```

因此 LoRA 执行不能只设置一个全局 adapter，而是要有 token / request 级映射。

---

## 5. 一句话总结

```text
active LoRA 状态把“请求用哪个 adapter”翻译成“本轮每个 token 用哪个 LoRA 权重”。
```
