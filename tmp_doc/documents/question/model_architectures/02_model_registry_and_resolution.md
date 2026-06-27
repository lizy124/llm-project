# 02. Model registry 如何解析模型架构？

源码位置：

- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/transformers_utils/`

本问题关注：vLLM 如何根据 HF config 中的 architecture name 找到对应的 vLLM model class。

---

## 1. 一句话回答

Model registry 负责把外部配置中的 architecture 名称，映射成 vLLM 内部的 model class。

```text
HF config architectures
  → ModelRegistry
  → vLLM model class
  → model_loader construct model
```

---

## 2. registry 要解决什么

```text
- 一个 HF architecture 对应哪个 vLLM class；
- 多个 architecture name 如何 fallback；
- 模型是否支持当前 task；
- 是否需要 trust_remote_code；
- 是否需要 lazy import；
- 模型类是否满足 vLLM 接口。
```

---

## 3. 解析流程占位

```text
ModelConfig 读取 HF config
  → 获取 architectures 字段
  → registry 查询匹配项
  → resolve model class
  → model_loader 创建实例
```

---

## 4. 容易混淆点占位

```text
1. HF model_type 和 architectures 不是一回事。
2. registry 选的是 vLLM 实现类，不一定是 transformers 原类。
3. trust_remote_code 影响外部模型解析，但 vLLM 仍需要执行接口兼容。
4. 同一模型家族可能有 generation / pooling / multimodal 不同实现。
```

---

## 5. 一句话总结

```text
Model registry 是 vLLM 从模型配置走向可执行 model class 的入口。
```
