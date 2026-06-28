# 10. 算子如何和 CUDA Graph / torch compile 协同？

源码位置：

- `vllm/vllm/compilation/`
- `vllm/vllm/worker/`
- `vllm/vllm/model_executor/`
- `vllm/vllm/attention/`

这个问题关注：算子在 CUDA Graph capture、torch compile、warmup、padding、static buffer、shape stability 场景下有什么要求，以及 fallback 如何影响图捕获和编译路径。

---

## 1. 一句话回答

CUDA Graph / compile 要求算子调用序列和张量形状尽量稳定，因此会反过来影响算子选择、padding 策略和临时 buffer 管理。

最小链路是：

```text
compile / warmup config
  → choose batch shapes
  → run model forward with selected kernels
  → capture or compile graph
  → replay stable operator sequence
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. 哪些算子会进入 CUDA Graph capture；
2. padding 如何让 shape 稳定；
3. workspace / temporary tensor 如何避免动态分配；
4. attention metadata 和 kernel 是否支持 capture；
5. fallback 分支为什么会破坏稳定性；
6. torch compile wrapper 如何包住 model forward；
7. capture 失败时如何定位具体算子。
```

---

## 3. 需要串起来的主线

```text
Worker warmup / compile
  → ModelRunner forward
  → selected operator backend
  → graph capture / compiled graph
  → replay during serving
```

---

## 4. 后续补充重点

```text
- cudagraph batch size；
- shape padding；
- attention backend capture compatibility；
- dynamic control flow；
- graph break 和 fallback debug。
```
