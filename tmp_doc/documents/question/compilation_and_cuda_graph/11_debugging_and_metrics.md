# 11. 如何调试 compile / CUDA graph 问题？

源码位置：

- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/compilation/`
- `code/vllm/vllm/logger.py`

本问题关注：compile overhead、cudagraph miss、capture 失败、shape mismatch、性能不达预期时如何定位。

---

## 1. 一句话回答

调试 compilation / cudagraph 问题，要沿着“配置 → capture 生命周期 → runtime dispatch → shape / metadata → fallback stats → 性能指标”逐层定位。

---

## 2. 调试入口占位

```text
1. 打印 compile config / cudagraph sizes。
2. 确认是否 enforce eager。
3. 查看 warmup / capture 日志。
4. 查看本轮 batch descriptor。
5. 查看是否 padding 到 capture size。
6. 查看 cudagraph runtime mode。
7. 查看 attention metadata 是否走 capture path。
8. 查看 fallback reason。
9. 对比 eager 和 graph 输出。
10. profile TPOT / TTFT / kernel launch 数。
```

---

## 3. 常见问题分类占位

```text
capture 失败：
  动态 shape、非法内存访问、backend 不支持、collective 不稳定。

graph miss：
  batch size 未命中、功能禁用、metadata shape 不匹配。

性能不升反降：
  padding 过多、batch 太小、compile 开销大、fallback 太频繁。

输出异常：
  static buffer 未正确更新、padded token 污染、metadata replay 不一致。
```

---

## 4. 指标占位

```text
- cudagraph hit / miss；
- capture time；
- compile time；
- replay time；
- eager fallback count；
- padding waste；
- TTFT；
- TPOT；
- GPU utilization；
- kernel launch 数。
```

---

## 5. 一句话总结

```text
compile / cudagraph 调试的核心，是确认每一轮到底走了哪条路径，以及为什么没有走预期路径。
```
