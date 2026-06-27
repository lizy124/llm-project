# 01. Multimodal 在 vLLM V1 里负责什么？

源码位置：

- `code/vllm/vllm/multimodal/`
- `code/vllm/vllm/config/multimodal.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/models/`

本问题关注：Multimodal 子系统在 vLLM V1 中的职责边界。它不是单独的模型执行器，而是一条把非文本输入转换成模型可消费 feature / embedding / encoder output 的输入准备与执行辅助链路。

---

## 1. 一句话回答占位

占位：后续补充 Multimodal 子系统在 vLLM 中的位置。

```text
Multimodal 负责把 image / audio / video / prompt_embeds 等输入，
转成模型 forward 能消费的 feature / embedding / encoder output，
并通过 Scheduler / ModelRunner 接入普通 decode / sampling / output 链路。
```

---

## 2. 它负责什么占位

```text
- 解析用户多模态输入；
- 调用 processor / mapper 生成模型特定特征；
- 维护多模态 placeholder 与 prompt token 的对应关系；
- 构造 MultiModalFeatureSpec；
- 做 processor cache；
- 估算 encoder budget；
- 让 Scheduler 调度 encoder input；
- 让 ModelRunner 执行 multimodal encoder；
- 将多模态 embedding 合并到 inputs_embeds；
- 配合 EncoderCacheManager 复用和释放 encoder output。
```

---

## 3. 它不负责什么占位

```text
- 不负责最终 token 采样；
- 不负责 detokenize 和用户输出格式；
- 不负责 decoder KV block 分配；
- 不负责 attention backend 选择；
- 不负责模型权重加载；
- 不负责 OpenAI response 协议包装。
```

---

## 4. 主线占位

```text
多模态输入
  → parser / processor
  → feature spec
  → EngineCoreRequest.mm_features
  → Scheduler encoder input 调度
  → ModelRunner 执行 mm encoder
  → inputs_embeds
  → decoder forward
```

---

## 5. 容易混淆点占位

```text
1. 多模态 feature cache 和 encoder cache 不是一回事。
2. 多模态 encoder output 和 decoder KV cache 不是一回事。
3. 多模态输入改变的是 forward 输入准备，不改变最终输出对象类型。
4. image/audio/video 进入模型前通常已经被 processor 转成 tensor feature。
```

---

## 6. 后续待补源码证据

占位：补充 `MultiModalConfig`、`MultiModalDataParser`、`MultiModalFeatureSpec`、`EncoderCacheManager`、`GPUModelRunner._execute_mm_encoder()` 等源码位置。
