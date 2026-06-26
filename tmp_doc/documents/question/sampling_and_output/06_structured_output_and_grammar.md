# 06. Structured output / grammar 如何限制采样？

源码位置：

- `code/vllm/vllm/config/structured_outputs.py`
- `code/vllm/vllm/v1/worker/gpu/structured_outputs.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：guided decoding、structured output、grammar bitmask 如何进入 sampler，并在输出 token 后如何推进状态。

---

## 1. 一句话回答

structured output 的核心是：

```text
每一步采样前，根据当前已生成内容计算一个合法 token bitmask，
sampler 只能从合法 token 中选择，
采样后再用新 token 推进 grammar 状态。
```

---

## 2. 主链路占位

```text
请求携带 structured output / grammar 配置
  → Scheduler 持有 grammar / structured output 状态
  → 每轮 step 前生成 grammar bitmask
  → EngineCore 将 grammar_output 交给 execute / sample 阶段
  → ModelRunner.sample_tokens(grammar_bitmask)
  → sampler mask logits
  → sampled token
  → Scheduler.update_from_output()
  → 推进 grammar 状态
```

---

## 3. grammar bitmask 的作用占位

```text
bitmask 表达：
  当前 request 在当前生成位置，哪些 token 合法。

sampler 使用方式：
  将非法 token logits 屏蔽，再继续执行 temperature / top-k / top-p / sampling。
```

后续需要补充：

```text
- bitmask 的 batch 维度如何对齐 request；
- bitmask 在 GPU / CPU 间如何传递；
- 与 spec decode 的关系；
- 与 stop token / EOS 的关系；
- grammar 编译和缓存机制。
```

---

## 4. structured output 和普通 stop 的区别

占位：

```text
structured output：
  约束“下一步哪些 token 可以选”。

stop condition：
  判断“生成到这里是否应该结束”。
```

二者可能同时存在，但职责不同。

---

## 5. 容易混淆点占位

```text
1. grammar bitmask 不负责 detokenize。
2. grammar bitmask 不等于 stop string 检查。
3. structured output 主要约束采样候选集合。
4. grammar 状态推进通常在 Scheduler 回收输出时完成。
5. 如果 bitmask 为空或非法，需要有 fallback / error 处理路径。
```

---

## 6. 一句话总结

```text
structured output 是采样前的合法 token 约束，Scheduler 负责状态推进，sampler 负责执行约束。
```
