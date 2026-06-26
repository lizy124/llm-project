# 11. Pooling / Embedding / Rerank 输出如何区别于生成式输出？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/model_executor/layers/resampler.py`

本问题关注：embedding、pooling、rerank 等非生成任务如何绕开普通 token sampling 路径，并最终返回用户可见输出。

---

## 1. 一句话回答

非生成式任务通常不需要 sampler。

```text
生成任务：
  hidden states → logits → sampled token → RequestOutput

pooling / embedding / rerank：
  hidden states → pooling / score / embedding → PoolerOutput / EmbeddingOutput
```

---

## 2. 和生成任务的分叉点

占位：

```text
_model_forward()
  → hidden states
  → 如果是 generation：compute_logits() + sample_tokens()
  → 如果是 pooling：_pool()
  → ModelRunnerOutput.pooler_output
  → Scheduler / OutputProcessor 走非生成输出路径
```

---

## 3. pooling 输出关心什么

```text
- 每个请求对应哪个 hidden state 或 token span；
- pooling method 是 last / mean / cls / all tokens 还是模型自定义；
- 输出是否需要 normalize；
- batch 内如何对应 request id；
- 是否需要和 encoder / multimodal 输出合并。
```

---

## 4. embedding / rerank 输出占位

后续补充：

```text
embedding：
  返回向量表示。

classification / score：
  返回类别或分数。

rerank：
  返回 query-doc pair 的相关性 score。
```

需要区分：

```text
- 它们是否需要 detokenize；
- 是否有 finish reason；
- 是否存在 streaming；
- 是否使用 SamplingParams；
- OutputProcessor 如何构造最终输出对象。
```

---

## 5. 容易混淆点占位

```text
1. pooling output 不是 sampled token。
2. embedding 输出通常不需要 stop condition。
3. pooling 请求仍然可能经过 Scheduler / Worker / ModelRunner 主链路。
4. attention / KV cache 仍可能参与 forward，只是不进入 generation sampler。
5. RequestOutput 家族中不同输出类型服务不同任务。
```

---

## 6. 一句话总结

```text
pooling / embedding 共享模型执行前半段，但在 hidden states 之后从 generation sampling 主线分叉。
```
