# 03. Worker warmup 和 CUDA graph capture 生命周期是什么？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/compilation/`

本问题关注：Worker 初始化后，模型如何 warmup、compile 和 capture CUDA graphs。

---

## 1. 一句话回答

CUDA graph 不是第一轮真实请求临时捕获的，而通常在 Worker 初始化 / warmup 阶段针对一组固定 shape 提前 capture。

---

## 2. 初始化链路占位

```text
Worker.init_device()
  → load_model()
  → initialize_from_config()
      → allocate KV cache
      → compile_or_warm_up_model()
      → profile / warmup
      → capture selected cudagraph sizes
      → 保存 graph runners / static buffers
```

---

## 3. capture 要准备什么

```text
- 固定形状 input ids / positions；
- 固定形状 attention metadata；
- 固定形状 slot mapping / block table 视图；
- static input / output buffers；
- KV cache 地址稳定；
- model forward 无动态 Python 分支；
- collective 调用顺序稳定。
```

---

## 4. capture size 占位

后续补充：

```text
- capture 哪些 batch size；
- capture 顺序；
- warmup runs；
- 最大 capture size；
- capture 失败如何处理；
- capture graph 如何缓存和查询。
```

---

## 5. 一句话总结

```text
warmup / capture 阶段的目标，是提前为常见固定 shape 准备可 replay 的 GPU 执行图。
```
