# 05. ModelRunner 如何执行 spec decode forward？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/spec_decode/metadata.py`
- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`

本问题关注：spec decode 请求进入 ModelRunner 后，如何准备输入、attention metadata、target logits，并把中间状态留给 sample_tokens。

---

## 1. 一句话回答

ModelRunner 在 spec decode 中负责把 draft tokens 展开成 target model 可验证的一组输入位置，并保存 target logits 与 `SpecDecodeMetadata`。

---

## 2. 主链路占位

```text
GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
      → input_ids 包含验证 draft 所需 token
      → positions 对齐 request 当前长度和 draft offset
      → logits_indices 指向 target / bonus logits rows
      → SpecDecodeMetadata
  → _build_attention_metadata()
  → _model_forward()
  → compute_logits()
  → ExecuteModelState(spec_decode_metadata=...)
  → sample_tokens()
```

---

## 3. attention metadata 关系占位

Spec decode 仍然要构造正常 attention metadata：

```text
- query_start_loc
- seq_lens
- block_table
- slot_mapping
- positions
- num_scheduled_tokens
```

区别是这些字段现在要覆盖 draft verification 的多个 token，而不是普通 decode 的单 token。

---

## 4. ExecuteModelState 占位

后续补充：

```text
ExecuteModelState 保存：
  scheduler_output
  logits
  spec_decode_metadata
  common attention metadata
  hidden states
  sample hidden states
  slot mappings
```

---

## 5. 一句话总结

```text
ModelRunner 负责把 spec decode 从“请求状态里的 draft tokens”变成“target model forward 的 logits rows”。
```
