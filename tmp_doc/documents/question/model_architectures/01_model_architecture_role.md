# 01. Model architecture 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/config/model.py`

本问题关注：模型架构层的职责、边界，以及它和配置层、执行层、layer 层的关系。

---

## 1. 一句话回答

Model architecture 负责把一个具体模型结构实现成 vLLM 可执行的 model class。

```text
它既要理解模型结构，
又要遵守 vLLM 的统一 forward / load_weights / compute_logits / pooler 接口。
```

---

## 2. 它负责什么

占位：

```text
- 根据 config 构造 layers；
- 创建 embedding / decoder layers / final norm / lm_head；
- 挂接 Attention / MLP / MoE / RoPE 等基础组件；
- 实现 forward；
- 实现 load_weights；
- 暴露 compute_logits 或 pooler；
- 支持 TP / PP / quant / LoRA 等 hook。
```

---

## 3. 它不负责什么

```text
- 不负责调度请求；
- 不负责分配 KV cache block；
- 不负责 sampler 规则；
- 不负责 HTTP API；
- 不负责最终 RequestOutput 组装。
```

---

## 4. 一句话总结

```text
模型架构层是外部模型结构和 vLLM 执行框架之间的适配层。
```
