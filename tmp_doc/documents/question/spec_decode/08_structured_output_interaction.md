# 08. Spec decode 如何和 structured output / grammar 交互？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/structured_output/utils.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/structured_output/`

本问题关注：structured output / grammar bitmask 如何影响 draft tokens、target logits、accepted tokens，以及 grammar 状态如何推进。

---

## 1. 一句话回答

structured output 会约束 spec decode 的两个位置：

```text
1. Scheduler 阶段：
   提前检查 / 修剪不合法 draft tokens。

2. Sampler 阶段：
   对 target / bonus logits 应用 grammar bitmask。
```

输出回收后，Scheduler 还要用 accepted tokens 推进 grammar 状态。

---

## 2. draft token 修剪占位

后续补充：

```text
Scheduler.update_draft_token_ids()
  → grammar 检查 draft tokens
  → trim 不合法 token
  → 更新 scheduled_spec_decode_tokens
  → 记录 invalid spec tokens
```

---

## 3. grammar bitmask 对齐占位

Spec decode 下 logits rows 不再是一请求一行，所以 grammar bitmask 需要考虑：

```text
- request 在 batch 中的位置；
- draft token offset；
- target logits rows；
- bonus logits rows；
- invalid / trimmed draft tokens。
```

---

## 4. grammar 状态推进占位

```text
RejectionSampler 输出 accepted tokens
  → Scheduler.update_from_output()
  → grammar.accept_tokens(accepted tokens)
  → 如果 grammar 拒绝已采样 token，则请求进入 error / failed 状态
```

---

## 5. 一句话总结

```text
structured output 让 spec decode 不能只看概率接受，还必须保证 draft、target、bonus token 都符合语法状态。
```
