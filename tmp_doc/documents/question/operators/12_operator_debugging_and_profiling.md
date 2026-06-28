# 12. 算子问题如何定位和分析？

源码位置：

- `vllm/vllm/platforms/`
- `vllm/vllm/logger.py`
- `vllm/vllm/attention/`
- `vllm/vllm/model_executor/layers/`

这个问题关注：当出现 fallback、kernel 不支持、shape mismatch、NaN、CUDA Graph capture 失败、性能异常时，如何判断问题属于哪个算子族、实际走了哪个 backend、该从哪些日志和 profiler 入口定位。

---

## 1. 一句话回答

算子 debug 的核心是先确认“实际走了哪个 backend”，再确认“输入 shape / dtype / layout / metadata 是否满足这个 backend 的要求”。

最小排查链路是：

```text
现象
  → 定位算子族
  → 确认实际 backend
  → 检查 shape / dtype / layout / metadata
  → 观察 profiler kernel
  → 判断是配置、fallback、bug 还是硬件限制
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. 如何从报错栈定位算子族；
2. 如何确认 attention / quantization / MoE backend；
3. 如何识别 torch fallback；
4. 如何用 profiler 观察 kernel launch 和耗时；
5. NaN / Inf / output mismatch 如何排查；
6. CUDA Graph capture 失败如何缩小范围；
7. 如何构造最小复现 batch。
```

---

## 3. 常见问题分类

```text
- backend 不支持当前 dtype / shape；
- optional dependency 缺失；
- CUDA extension 未编译或加载失败；
- metadata 与 tensor shape 不一致；
- cache layout 或 slot mapping 错误；
- fallback 到慢路径；
- capture / compile 图不稳定；
- quantization scale 或 packed weight 不匹配。
```

---

## 4. 后续补充重点

```text
- 日志入口；
- 环境变量；
- profiler 观察点；
- backend capability check；
- 最小复现样例组织方式。
```
