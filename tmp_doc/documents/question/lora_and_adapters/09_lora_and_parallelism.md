# 09. LoRA 如何与并行机制交互？

源码位置：

- `code/vllm/vllm/lora/`
- `code/vllm/vllm/distributed/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/v1/executor/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：TP / PP / DP / EP 场景下 LoRA 权重和 active mapping 如何处理。

---

## 1. 一句话回答

并行场景下，LoRA adapter 不仅要加载到正确 worker，还要按 base layer 的并行切分方式切分或复制，并保证 batch 内 active LoRA mapping 在各 rank 上一致。

---

## 2. Tensor parallel 占位

需要梳理：

```text
- ColumnParallelLinear 的 LoRA A/B 如何切；
- RowParallelLinear 的 LoRA A/B 如何切；
- QKV fused layer 如何切；
- vocab parallel embedding / lm_head 如何处理；
- LoRA delta 是否需要 all-reduce / gather。
```

---

## 3. Pipeline parallel 占位

```text
- 每个 PP rank 只加载自己负责 layer 的 LoRA 权重；
- LoRARequest 仍要广播给所有相关 worker；
- intermediate tensors 不携带 LoRA 权重，但 forward 路径已应用对应 LoRA。
```

---

## 4. Data parallel 占位

```text
- 每个 DP replica 需要加载同一 adapter；
- add/remove/pin 控制面要同步到所有 replica；
- request 被路由到哪个 DP rank，就在该 rank 激活对应 LoRA。
```

---

## 5. 一句话总结

```text
LoRA 并行的关键，是让 adapter 权重分布方式和 base model layer 的并行方式一致。
```
