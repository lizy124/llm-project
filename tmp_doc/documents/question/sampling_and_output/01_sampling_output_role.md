# 01. Sampling / Output 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/outputs.py`

本问题关注：模型 forward 之后，vLLM 内部如何从 hidden states / logits 走到用户可见输出；哪些逻辑属于 ModelRunner，哪些属于 Scheduler，哪些属于 OutputProcessor。

---

## 1. 一句话回答

Sampling / Output 不是单一函数，而是三层协作：

```text
ModelRunner / sampler：
  从 logits 采样 token。

Scheduler.update_from_output：
  把 token 消化回 request 状态机。

OutputProcessor：
  把内部状态转换成用户可见 RequestOutput。
```

---

## 2. 边界占位

### 2.1 ModelRunner 负责什么

占位：后续补充 `GPUModelRunner.execute_model()`、`sample_tokens()`、sampler 调用路径。

```text
- hidden states → logits
- logits_indices / sampling metadata
- grammar bitmask 接入
- sampler 执行
- spec decode acceptance / rejection
- logprobs / prompt logprobs
- 生成 ModelRunnerOutput
```

### 2.2 Scheduler 负责什么

占位：后续补充 `Scheduler.update_from_output()` 如何消费 `ModelRunnerOutput`。

```text
- append sampled tokens
- 更新 request.num_computed_tokens / output_token_ids
- stop condition
- spec decode 状态修正
- structured output 状态推进
- block / encoder cache / connector 释放
- 生成 EngineCoreOutputs
```

### 2.3 OutputProcessor 负责什么

占位：后续补充 `OutputProcessor` 如何处理 detokenize、streaming、finished reason、RequestOutput。

```text
- detokenize
- incremental output
- logprobs 格式转换
- prompt logprobs 格式转换
- finished / stopped / length / abort reason
- RequestOutput / CompletionOutput
```

---

## 3. 主线占位

```text
GPUModelRunner._model_forward()
  → hidden_states
  → compute_logits()
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput
```

---

## 4. 容易混淆点占位

```text
1. ModelRunnerOutput 不是用户输出。
2. sampled_token_ids 不是 detokenized text。
3. Scheduler.update_from_output() 不只是转发 token，还会更新状态和释放资源。
4. OutputProcessor 不重新采样，只做输出组装和增量处理。
5. pooling / embedding 输出不走普通 token sampling 路径。
```

---

## 5. 一句话总结

```text
Sampling 解决“下一个 token 是什么”，Output 解决“这个 token 如何成为用户看见的结果”。
```
