# 03. hidden states 如何变成 logits / logprobs？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/sample/logprob.py`
- `code/vllm/vllm/v1/worker/gpu/sample/prompt_logprob.py`
- `code/vllm/vllm/v1/sample/ops/logprobs.py`
- `code/vllm/vllm/v1/engine/logprobs.py`
- `code/vllm/vllm/logprobs.py`

本问题关注：模型 forward 产生 hidden states 后，哪些位置会被拿去算 logits；logprobs 和 prompt logprobs 分别在哪里算、如何返回。

---

## 1. 一句话回答

不是所有 hidden states 都会算 logits。

```text
ModelRunner 先根据 logits_indices 选出需要的位置，
再通过 lm_head / logits processor 得到 logits，
sampler 只消费这些 logits，
logprobs / prompt logprobs 根据请求参数额外计算和回传。
```

---

## 2. logits_indices 的意义

占位：后续补充 `_prepare_inputs()` 中 logits_indices 的构造。

```text
普通 decode：
  每个 request 通常只需要最后一个 token 的 logits。

prefill：
  可能只需要 prompt 最后位置，或者需要 prompt logprobs 的多个位置。

spec decode：
  需要 draft token / bonus token 对应的一组 logits。

pooling / embedding：
  可能根本不走 generation logits。
```

---

## 3. logits 计算路径占位

```text
_model_forward()
  → hidden_states
  → 根据 logits_indices 取 sample_hidden_states
  → compute_logits()
  → logits
  → sampler
```

后续补充：

```text
- vocab parallel / tensor parallel 下 logits 如何 gather；
- logits processor / soft cap / final logits scaling；
- logits 是否保留在 GPU；
- async scheduling 下 logits 如何跨 execute_model / sample_tokens 传递。
```

---

## 4. logprobs 和 prompt logprobs

占位：区分两类 logprobs。

```text
logprobs：
  生成 token 位置的候选 token log probability。

prompt_logprobs：
  prompt token 自身在模型下的 log probability，常用于评分或调试。
```

后续补充源码：

```text
- `v1/worker/gpu/sample/logprob.py`
- `v1/worker/gpu/sample/prompt_logprob.py`
- `v1/sample/ops/logprobs.py`
- `v1/engine/logprobs.py`
```

---

## 5. 容易混淆点占位

```text
1. logits 不是最终输出。
2. logprobs 不是 sampler 必须字段，只有请求需要时才返回。
3. prompt_logprobs 和 generation logprobs 位置不同。
4. spec decode 下 logits_indices 不等于普通 decode 的最后位置。
5. structured output mask 会影响可采样 logits，但不等于 logprobs 格式化。
```

---

## 6. 一句话总结

```text
logits 是 sampler 的输入，logprobs 是输出附加信息；二者都由 logits_indices 决定计算位置。
```
