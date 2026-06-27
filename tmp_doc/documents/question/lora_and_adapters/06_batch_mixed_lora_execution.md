# 06. 同一 batch 中多个 LoRA 如何混合执行？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/lora/punica_wrapper/`
- `code/vllm/vllm/lora/layers.py`

本问题关注：同一个 batch 中不同请求使用不同 LoRA adapter 时，vLLM 如何避免拆成多个模型 forward。

---

## 1. 一句话回答

batch mixed LoRA 的核心是：

```text
同一个 forward 中，
不同 token / request 根据 mapping 使用不同 LoRA 权重，
LoRA kernel 按映射计算各自 delta，
再合并回 base output。
```

---

## 2. batch 场景占位

```text
batch:
  req0: no LoRA
  req1: LoRA A
  req2: LoRA A
  req3: LoRA B
  req4: no LoRA
```

需要表达：

```text
- 哪些 tokens 属于哪个 request；
- 每个 request 使用哪个 LoRA id；
- 相同 LoRA 的请求可共享 adapter 权重；
- no-LoRA 请求只走 base output。
```

---

## 3. token mapping 占位

后续补充：

```text
- token_lora_mapping；
- prompt_mapping；
- lora_index_mapping；
- seq_lens / query_start_loc 与 LoRA mapping 的关系；
- prefill / decode 混合 batch 下的 LoRA mapping。
```

---

## 4. 性能问题占位

```text
- LoRA adapter 数量越多，kernel 调度越复杂；
- rank 越大，delta 计算越重；
- batch 内 LoRA 分布会影响效率；
- CUDA graph 可能对动态 LoRA mapping 有限制。
```

---

## 5. 一句话总结

```text
batch mixed LoRA 把“请求级 adapter 选择”转成“token 级 LoRA 权重映射”。
```
