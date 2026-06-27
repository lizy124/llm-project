# 04. SpecDecodeMetadata 如何描述 draft / target / bonus 位置？

源码位置：

- `code/vllm/vllm/v1/spec_decode/metadata.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：ModelRunner 如何把 SchedulerOutput 中的 draft tokens 转换成 `SpecDecodeMetadata`，以及这些字段如何对齐 target logits 和 sampler。

---

## 1. 一句话回答

`SpecDecodeMetadata` 是 spec decode 的位置说明书。

```text
它告诉 RejectionSampler：

- draft tokens 是哪些；
- 每个请求有多少 draft tokens；
- target logits 的哪些行用于验证 draft tokens；
- bonus logits 的哪些行用于全接受后的额外采样；
- 普通 logits_indices 在 spec decode 下如何重新定义。
```

---

## 2. 关键字段占位

后续补充源码字段：

```text
draft_token_ids
num_draft_tokens
cu_num_draft_tokens
target_logits_indices
bonus_logits_indices
logits_indices
cu_num_sampled_tokens
```

---

## 3. 和普通 logits_indices 的区别

```text
普通 decode：
  logits_indices 通常是每个 request 的最后一个 query token。

spec decode：
  logits_indices 需要覆盖 target verification 和 bonus sampling 位置。
```

---

## 4. 构造时机占位

```text
GPUModelRunner._prepare_inputs()
  → 发现 scheduler_output.scheduled_spec_decode_tokens
  → 计算 draft token 数量
  → _calc_spec_decode_metadata()
  → 返回 SpecDecodeMetadata
```

---

## 5. 一句话总结

```text
SpecDecodeMetadata 把“多个 draft token 验证”翻译成 target logits 行和 sampler 可理解的索引关系。
```
