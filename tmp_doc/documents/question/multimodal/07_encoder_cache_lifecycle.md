# 07. EncoderCacheManager 如何缓存和释放 encoder output？

源码位置：

- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：多模态 encoder output 如何缓存、复用和释放。它和 decoder KV cache 类似都是执行缓存，但管理对象、生命周期和使用方式不同。

---

## 1. 一句话回答占位

```text
EncoderCacheManager 管理 encoder output 的资源账本，
Scheduler 根据它判断哪些 encoder input 已经可复用，
Worker / ModelRunner 根据 SchedulerOutput 执行或跳过 encoder，
请求结束后再释放对应 encoder cache。
```

---

## 2. Encoder cache 和 KV cache 的区别占位

```text
Encoder cache：
  缓存 multimodal / encoder-decoder encoder 输出。

Decoder KV cache：
  缓存 decoder self-attention 的 key/value blocks。

区别：
  一个是 encoder output，一个是 decoder attention KV；
  一个通常按 encoder input hash 管理，一个按 request token block 管理。
```

---

## 3. 生命周期占位

```text
请求进入
  → 检查 encoder cache
  → 未命中则调度 encoder input
  → Worker 执行 encoder
  → encoder output 写入 cache
  → decoder forward 消费 encoder output / embedding
  → 请求完成
  → Scheduler 释放 encoder cache
```

---

## 4. 需要解释的问题占位

```text
- encoder cache key 是什么？
- 一个 request 可能有多个 encoder input 吗？
- cache 命中后 SchedulerOutput 如何表达？
- Worker 返回的 ec_connector_output / encoder output 如何被 Scheduler 消化？
- free_encoder_mm_hashes 表示什么？
```

---

## 5. 后续待补源码证据

占位：补充 `EncoderCacheManager`、Scheduler update / free 逻辑、ModelRunner encoder output 回传路径。
