# 03. 多模态 parser 和 processor 如何生成 feature？

源码位置：

- `code/vllm/vllm/multimodal/parse.py`
- `code/vllm/vllm/multimodal/inputs.py`
- `code/vllm/vllm/multimodal/processing.py`
- `code/vllm/vllm/multimodal/profiling.py`
- `code/vllm/vllm/multimodal/registry.py`
- `code/vllm/vllm/assets/`

本问题关注：原始 image / audio / video 等多模态数据如何被解析、校验、处理，并转换成模型需要的 tensor feature 或 processor output。

---

## 1. 一句话回答占位

```text
MultiModalDataParser 负责把原始多模态数据按 modality 解析出来，
processor / mapper 负责把这些数据转换成模型特定输入，
最终封装为 MultiModalFeatureSpec 供 EngineCore / Scheduler / Worker 使用。
```

---

## 2. 角色边界占位

```text
MultiModalDataParser：
  解析用户传入的多模态数据结构。

MultiModalRegistry：
  管理不同模型、不同 modality 的 processor / mapper。

Processor：
  通常来自 Hugging Face processor / image processor / audio processor。

Mapper：
  把 processor output 映射成 vLLM 模型 forward 需要的 kwargs / feature。

MultiModalFeatureSpec：
  描述处理后的 feature、hash、modality、位置和执行需求。
```

---

## 3. 主链路占位

```text
raw media data
  → MultiModalDataParser
  → modality-specific parser
  → processor
  → mapper
  → MultiModalFeatureSpec
  → EngineCoreRequest.mm_features
```

---

## 4. 需要解释的问题占位

```text
- image / audio / video 是否走相同 parser？
- processor cache 在哪里接入？
- feature hash 如何计算？
- mapper output 是 tensor 还是 model kwargs？
- 多个 image / video frame 如何组织？
```

---

## 5. 后续待补源码证据

占位：补充 `MultiModalDataParser`、`MultiModalFeatureSpec`、registry、processor cache 和 mapper 相关源码。
