# 05. ModelRunner 每轮如何选择 eager / compiled / cudagraph replay？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/compilation/`

本问题关注：真实请求执行时，ModelRunner 如何判断本轮是否可以使用 cudagraph，并如何 dispatch 到对应 forward 路径。

---

## 1. 一句话回答

ModelRunner 每轮会根据 batch shape、配置和功能限制决定执行路径：能命中已 capture graph 就 replay，否则走 compiled 或 eager。

---

## 2. 主链路占位

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
      → 判断 cudagraph mode
      → 选择 padding shape
      → 生成 batch descriptor
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
      → graph replay / compiled / eager
```

---

## 3. dispatch 判断因素占位

```text
- batch size 是否在 captured sizes 中；
- 是否需要 prefill；
- 是否包含 encoder / multimodal；
- 是否有动态 LoRA；
- 是否是 pooling；
- 是否 spec decode；
- attention backend 是否支持 capture；
- 是否处于 profile / warmup / compile 阶段。
```

---

## 4. fallback 占位

```text
如果不满足 graph replay 条件：

- 不应该报错；
- 应该安全 fallback；
- 记录 cudagraph miss / fallback stats；
- 输出语义保持一致。
```

---

## 5. 一句话总结

```text
cudagraph dispatch 的核心，是在保证语义正确的前提下尽可能复用已 capture 的固定 shape graph。
```
