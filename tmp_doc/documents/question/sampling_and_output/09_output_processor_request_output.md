# 09. OutputProcessor 如何生成 RequestOutput？

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/engine/logprobs.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/logprobs.py`
- `code/vllm/vllm/v1/outputs.py`

本问题关注：Scheduler 生成的 `EngineCoreOutputs` 如何变成 API 层可见的 `RequestOutput`、`CompletionOutput`、logprobs 和 finished 状态。

---

## 1. 一句话回答

OutputProcessor 是内部状态到用户输出的转换层。

```text
EngineCoreOutputs
  → detokenize / incremental decode
  → logprobs 格式转换
  → finish reason / stop reason
  → RequestOutput
```

---

## 2. RequestOutput 和内部输出的区别

占位：

```text
内部输出关注：
  req_id、token ids、scheduler 状态、KV 状态、worker 统计信息。

用户输出关注：
  text、token ids、logprobs、finish reason、stop reason、metrics、finished 标记。
```

---

## 3. OutputProcessor 主职责占位

```text
- 根据 request id 找到对应输出状态；
- detokenize 新 token；
- 维护增量文本缓存；
- 根据 output kind 决定输出全量还是增量；
- 组装 CompletionOutput；
- 组装 RequestOutput；
- 处理 logprobs / prompt logprobs 格式；
- 标记 finished / aborted / stopped / length / EOS。
```

---

## 4. logprobs 格式转换占位

后续补充：

```text
worker 侧 logprobs：
  更偏 token id / tensor / batch 内部结构。

engine 输出侧 logprobs：
  更偏用户 API，包含 token、decoded text、rank、logprob 等。
```

相关文件：

```text
- `v1/engine/logprobs.py`
- `logprobs.py`
```

---

## 5. finish reason 占位

需要区分：

```text
- stop token
- stop string
- EOS
- max_tokens / length
- abort
- error
- structured output 完成
```

后续补充这些 reason 在 Scheduler 和 OutputProcessor 中的分工。

---

## 6. 一句话总结

```text
OutputProcessor 不决定模型生成什么，它决定生成结果如何被用户看见。
```
