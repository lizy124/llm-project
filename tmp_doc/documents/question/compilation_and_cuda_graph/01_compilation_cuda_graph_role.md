# 01. Compilation / CUDA graph 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/compilation/`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：torch.compile 和 CUDA graph 在 vLLM 中的职责、区别和边界。

---

## 1. 一句话回答

Compilation / CUDA graph 都是为了减少运行时开销，但解决的问题不同：

```text
torch.compile：
  优化模型 forward 图和 kernel 组合。

CUDA graph：
  复用固定 shape 下的 GPU kernel launch 序列。
```

---

## 2. 为什么 vLLM 需要它们

占位：

```text
LLM serving 中每步 decode 计算量不大但调用频繁，
Python overhead 和 kernel launch overhead 会明显影响 TPOT。

CUDA graph replay 可以减少 launch 开销，
torch.compile 可以减少动态图和 eager op 组织开销。
```

---

## 3. 它们不负责什么

```text
- 不改变 Scheduler 的调度目标；
- 不改变 attention 数学语义；
- 不改变 sampler 规则；
- 不改变 KV cache 分配逻辑；
- 不保证所有动态功能都能 capture。
```

---

## 4. 边界占位

```text
编译 / graph 通常包住 model forward，
但 sampler、OutputProcessor、Scheduler 状态更新等 CPU / 动态逻辑通常在 graph 边界外。
```

---

## 5. 一句话总结

```text
Compilation / CUDA graph 是执行优化层，不是模型语义层。
```
