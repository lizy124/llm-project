# 08. logits 与 sampling 相关算子如何工作？

源码位置：

- `vllm/vllm/model_executor/layers/logits_processor.py`
- `vllm/vllm/model_executor/layers/sampler.py`
- `vllm/vllm/v1/sample/`
- `vllm/vllm/_custom_ops.py`

这个问题关注：logits processor、logprobs、top-k / top-p、temperature、repetition penalty、rejection sampling、structured output bitmask 等输出侧计算如何通过算子或张量操作完成。

---

## 1. 一句话回答

sampling / logits 算子负责把模型 hidden states 或 logits 变成本轮输出 token，并为 logprobs、spec decode、structured output 等功能提供底层计算。

最小链路是：

```text
hidden states
  → logits processor
  → sampling metadata / grammar mask / penalties
  → sampling kernel or torch ops
  → sampled token ids / logprobs
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. logits 如何从 hidden states 计算出来；
2. logits processor 如何处理 indices、vocab、soft cap；
3. temperature / top-k / top-p / min-p / penalties 如何应用；
4. logprobs / prompt logprobs 如何计算；
5. spec decode rejection sampler 的算子路径；
6. structured output / grammar bitmask 如何影响 logits；
7. CPU/GPU 边界和 async scheduling 下的输出侧开销。
```

---

## 3. 需要串起来的主线

```text
ModelRunner.execute_model()
  → logits
  → ModelRunner.sample_tokens()
  → sampler / logits processors
  → ModelRunnerOutput
```

---

## 4. 后续补充重点

```text
- logits_indices；
- vocabulary parallelism；
- grammar bitmask；
- rejection sampling；
- logprobs tensor 到 CPU 结构的转换。
```
