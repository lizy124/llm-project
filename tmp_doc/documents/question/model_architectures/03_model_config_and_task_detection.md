# 03. ModelConfig 如何识别 task 和模型能力？

源码位置：

- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/config.py`
- `code/vllm/vllm/tasks.py`
- `code/vllm/vllm/model_executor/models/registry.py`

本问题关注：ModelConfig 如何判断模型支持 generation、embedding、classification、multimodal、pooling 等任务。

---

## 1. 一句话回答

ModelConfig 把外部模型配置、用户 task 参数和 registry 能力组合起来，决定当前模型以什么 runner_type 和 task 运行。

---

## 2. 需要梳理的字段

```text
- task；
- runner_type；
- model architecture；
- max_model_len；
- dtype；
- tokenizer mode；
- multimodal config；
- embedding / pooling capabilities；
- generation capabilities；
- quantization；
- trust_remote_code。
```

---

## 3. task detection 占位

```text
用户指定 task
  → ModelConfig 校验模型是否支持
  → 如果 task=auto，按模型能力推断
  → 选择 runner_type
      → generate
      → pooling
      → draft / speculative
      → embedding / classify 等
```

---

## 4. 对执行链路的影响

```text
runner_type=generate：
  ModelRunner forward 后 compute_logits / sampler。

runner_type=pooling：
  ModelRunner forward 后 _pool。

multimodal：
  InputProcessor / ModelRunner preprocess 需要处理 mm inputs。
```

---

## 5. 一句话总结

```text
ModelConfig 决定模型以哪种任务形态接入 vLLM 执行链路。
```
