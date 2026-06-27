# 05. MultiModalFeatureSpec 和 MultiModalCache 如何组织？

源码位置：

- `code/vllm/vllm/multimodal/inputs.py`
- `code/vllm/vllm/multimodal/cache.py`
- `code/vllm/vllm/multimodal/processing.py`
- `code/vllm/vllm/v1/engine/__init__.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`

本问题关注：多模态 feature 在 vLLM 中如何被描述、缓存和复用。尤其要区分 processor cache、feature spec 和 encoder cache 的边界。

---

## 1. 一句话回答占位

```text
MultiModalFeatureSpec 描述一个多模态输入处理后的 feature，
MultiModalCache 主要缓存 processor 结果，
EncoderCacheManager 则缓存模型 encoder 执行后的 encoder output。
```

---

## 2. FeatureSpec 占位

后续补充字段：

```text
- modality；
- feature hash；
- placeholder / range；
- processor output；
- model input kwargs；
- 是否需要 encoder；
- 与 request 的关系。
```

---

## 3. Cache 层次占位

```text
Processor cache / MultiModalCache：
  缓存原始 media 经 processor 处理后的 feature，避免重复 CPU / processor 开销。

Encoder cache：
  缓存模型 encoder forward 后的 output，避免重复 GPU encoder 计算。

Decoder KV cache：
  缓存 decoder attention KV，与多模态 feature cache 不是一回事。
```

---

## 4. 主链路占位

```text
raw media
  → processor output
  → MultiModalFeatureSpec
  → processor cache 命中 / 写入
  → Scheduler 判断 encoder cache
  → ModelRunner 执行 encoder 或复用 encoder output
```

---

## 5. 后续待补源码证据

占位：补充 `MultiModalFeatureSpec`、`MultiModalCache`、feature hashing、encoder cache key 的源码位置。
