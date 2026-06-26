# 10. Streaming 输出和客户端可见输出如何组织？

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/entrypoints/`

本问题关注：streaming 模式下，每轮 EngineCoreOutputs 如何变成增量输出；非 streaming / final-only 输出又如何组织。

---

## 1. 一句话回答

Streaming 输出的核心不是重新生成 token，而是决定：

```text
本轮新增了哪些 token / text / logprobs，
哪些应该立即返回给用户，
哪些应该缓存在 request 状态里等待后续输出。
```

---

## 2. 输出模式占位

```text
增量输出：
  每轮只返回新 token / new text。

累计输出：
  每轮返回从开始到当前的完整 text。

final-only：
  中间轮不返回，完成时一次性返回。
```

后续补充：这些模式如何由 output kind / request 配置决定。

---

## 3. streaming 主链路占位

```text
EngineCore.step()
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput
  → Engine / AsyncLLM
  → OpenAI-compatible server / Python iterator
  → client chunk
```

---

## 4. token 和 text 的关系占位

```text
sampled_token_ids：
  worker / scheduler 内部 token id。

output_token_ids：
  request 当前已经接受的输出 token id。

text：
  detokenizer 将 token ids 转成字符串。

delta_text：
  streaming 场景本轮新增字符串。
```

后续补充 detokenizer 如何处理：

```text
- byte fallback；
- special tokens；
- stop strings；
- partial unicode；
- skip_special_tokens；
- spaces_between_special_tokens。
```

---

## 5. finished chunk 占位

最后一轮输出通常需要携带：

```text
- finished = True
- finish_reason
- stop_reason
- final token ids / text
- final logprobs
- metrics / timings
```

---

## 6. 一句话总结

```text
streaming 解决的是“每轮把多少已生成内容暴露给用户”，不是“模型每轮如何生成”。
```
