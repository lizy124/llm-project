# 08. Sampler / Output 和 CUDA graph 的边界在哪里？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/outputs.py`

本问题关注：CUDA graph capture 通常覆盖哪些 forward 计算，sampler、logprobs、pooling、output 是否也在图内。

---

## 1. 一句话回答

CUDA graph 主要优化 model forward；sampler、OutputProcessor、Scheduler update 等动态逻辑通常在 graph 边界之外。

---

## 2. forward 和 sampler 分界占位

```text
_model_forward()
  → hidden_states
  → compute_logits()
  → sample_tokens()
      → grammar bitmask
      → sampler / rejection sampler
      → bookkeeping
```

需要梳理：

```text
- compute_logits 是否在 graph 内；
- sampler 是否 capture；
- logprobs 计算是否 capture；
- pooling _pool 是否 capture；
- async output copy 与 graph 的关系。
```

---

## 3. 为什么 output 不适合 capture

```text
- request 数动态；
- sampled token 会改变 CPU 状态；
- logprobs 返回结构动态；
- stop / grammar / spec decode 状态更新动态；
- OutputProcessor 主要是 CPU / Python 逻辑。
```

---

## 4. 一句话总结

```text
CUDA graph 优化的是 GPU forward 路径，输出回收仍然是动态状态机逻辑。
```
