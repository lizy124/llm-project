# 07. Model forward 如何被 compile wrapper / graph runner 包装？

源码位置：

- `code/vllm/vllm/compilation/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/forward_context.py`

本问题关注：模型 forward 如何从普通 PyTorch 调用变成 compiled forward 或 cudagraph replay。

---

## 1. 一句话回答

vLLM 会在模型加载 / warmup 后，为模型 forward 包装编译或 graph runner，使运行时 `_model_forward()` 可以根据模式调用不同执行路径。

---

## 2. 普通 forward 占位

```text
_model_forward()
  → self.model(input_ids, positions, intermediate_tensors, **model_kwargs)
```

这是 eager baseline。

---

## 3. compile wrapper 占位

需要梳理：

```text
- torch.compile 包装发生在哪里；
- 编译单位是整个 model 还是子模块；
- dynamic shape 如何处理；
- compilation cache 如何管理；
- 编译失败如何 fallback。
```

---

## 4. graph runner 占位

```text
capture：
  用 static input buffers 运行一次 forward 并捕获 CUDA graph。

replay：
  将真实输入 copy 到 static buffers，replay graph，再从 static output buffers 取结果。
```

---

## 5. 一句话总结

```text
compile wrapper / graph runner 的作用，是让 ModelRunner 保持统一 forward 接口，同时在内部切换不同执行实现。
```
