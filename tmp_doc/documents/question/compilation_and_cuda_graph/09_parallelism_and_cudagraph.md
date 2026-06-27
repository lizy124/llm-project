# 09. 并行机制如何影响 CUDA graph / compile？

源码位置：

- `code/vllm/vllm/distributed/`
- `code/vllm/vllm/v1/executor/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/compilation/`

本问题关注：TP / PP / DP / collective 通信如何影响 compile 和 cudagraph capture / replay。

---

## 1. 一句话回答

并行场景下，CUDA graph 不只是单卡 forward capture，还要保证各 rank 的 shape、通信顺序和 collective 调用完全一致。

---

## 2. Tensor parallel 占位

```text
- TP rank 上的 matmul shape 是否一致；
- all-reduce / all-gather 是否在 graph 中；
- collective 顺序是否稳定；
- TP size 改变是否需要重新 capture。
```

---

## 3. Pipeline parallel 占位

```text
- PP rank 输入可能是 intermediate_tensors；
- 首 rank / 中间 rank / 末 rank forward 输入不同；
- 每个 PP rank 可能需要独立 capture；
- microbatch / ubatch 会影响 shape。
```

---

## 4. Data parallel 占位

```text
- DP rank batch size 可能不均衡；
- DP padding / global batch descriptor；
- DP metadata 是否参与 forward context；
- 不同 DP rank 是否同时 replay。
```

---

## 5. 一句话总结

```text
并行场景下 cudagraph 的难点，是让每个 rank 不仅计算 shape 固定，通信图也固定。
```
