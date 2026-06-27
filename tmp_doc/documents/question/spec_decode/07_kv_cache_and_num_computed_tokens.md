# 07. Spec decode 如何影响 KV cache 和 num_computed_tokens？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`

本问题关注：spec decode 中 accepted / rejected tokens 如何影响 request 的计算进度、KV cache 使用和后续 recompute。

---

## 1. 一句话回答

Spec decode 的难点不是只采样多个 token，而是要保证：

```text
output_token_ids
num_computed_tokens
KV cache 中已写入的 token
下一轮 input positions
```

四者在 accepted / rejected 后仍然一致。

---

## 2. 状态一致性问题占位

```text
如果 draft token 被接受：
  它可以进入 output_token_ids，相关 target forward 结果也可以推进计算进度。

如果 draft token 被拒绝：
  后续 draft tokens 不能进入 output_token_ids，
  KV / num_computed_tokens 需要避免把无效 token 当成已确认上下文。

如果全部接受：
  bonus token 也可能成为新输出，状态继续推进。
```

---

## 3. 需要梳理的字段占位

```text
request.num_computed_tokens
request.output_token_ids
InputBatch.num_computed_tokens_cpu
scheduled_spec_decode_tokens
num_scheduled_tokens
slot_mapping
block table
```

---

## 4. KV cache 关系占位

后续补充：

```text
- spec decode 是否会提前写入 draft token 的 KV；
- rejected token 对后续 KV cache 的影响；
- 是否需要 recompute；
- num_computed_tokens 如何回退或修正；
- prefix cache / external KV connector 是否受影响。
```

---

## 5. 一句话总结

```text
Spec decode 的状态修正核心，是让“验证过的 token”而不是“猜过的 token”决定 request 和 KV cache 的真实进度。
```
