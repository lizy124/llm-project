# vLLM V1 Multimodal 问题目录

源码位置：

- `vllm/vllm/multimodal/`
- `vllm/vllm/assets/`
- `vllm/vllm/config/multimodal.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/core/sched/`
- `vllm/vllm/v1/core/encoder_cache_manager.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/model_executor/models/`

这个目录按问题拆解 vLLM V1 的多模态链路，重点回答：多模态输入如何从用户请求进入 vLLM，processor 如何生成 features，placeholder 如何和 prompt token 对齐，Scheduler 如何调度 encoder inputs，Worker / ModelRunner 如何执行 multimodal encoder，encoder cache 如何复用和释放，以及最终如何和文本 decode / KV cache / output 链路协同。

---

## 1. 总览文档

- [vLLM V1 Multimodal 逻辑梳理](multimodel_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
用户请求 image / audio / video / embeds
  → entrypoints / renderer
  → InputProcessor
  → MultiModalDataParser
  → processor / mapper
  → MultiModalFeatureSpec
  → EngineCoreRequest.mm_features
  → Request.mm_features
  → Scheduler.schedule_encoder_inputs
  → SchedulerOutput.scheduled_encoder_inputs
  → ModelRunner._execute_mm_encoder()
  → encoder cache / inputs_embeds
  → decoder forward / sampling / output
```

---

## 2. 主线专题阅读顺序

### 01. Multimodal 的定位

- [Multimodal 在 vLLM V1 里负责什么？](01_multimodal_role.md)

回答：

```text
Multimodal 是哪一层？
它负责什么，不负责什么？
它和 Engine / Scheduler / ModelRunner / model_executor 的边界是什么？
```

### 02. 用户输入与入口解析

- [多模态用户输入如何进入 vLLM？](02_multimodal_request_entry.md)

回答：

```text
OpenAI / offline LLM / renderer 如何接收 image、audio、video？
EngineInput 里多模态字段是什么？
InputProcessor 如何把外部输入变成 EngineCoreRequest？
```

### 03. Parser / processor / mapper

- [多模态 parser 和 processor 如何生成 feature？](03_parser_processor_mapper.md)

回答：

```text
MultiModalDataParser 做什么？
processor cache 如何工作？
mapper 如何把原始 media 转成 MultiModalFeatureSpec？
不同 modality 如何分发？
```

### 04. Placeholder 与 prompt token 对齐

- [多模态 placeholder 如何和 prompt token 对齐？](04_placeholders_and_prompt_alignment.md)

回答：

```text
placeholder token 是什么？
image/audio/video token span 如何插入 prompt？
mm_placeholders / PlaceholderRange 如何表示？
模型如何知道哪些 token 对应多模态 embedding？
```

### 05. MultiModalFeatureSpec 与缓存

- [MultiModalFeatureSpec 和 MultiModalCache 如何组织？](05_feature_spec_and_cache.md)

回答：

```text
MultiModalFeatureSpec 包含什么？
MultiModalCache 缓存什么？
processor cache 和 encoder cache 是一回事吗？
特征如何在请求之间复用？
```

### 06. Encoder budget 与 Scheduler 调度

- [Scheduler 如何调度多模态 encoder input？](06_encoder_budget_and_scheduler.md)

回答：

```text
MultiModalBudget 如何估算 encoder token？
Scheduler 如何决定本轮处理哪些 encoder inputs？
EncoderCacheManager 如何参与调度？
SchedulerOutput.scheduled_encoder_inputs 表示什么？
```

### 07. Encoder cache 生命周期

- [EncoderCacheManager 如何缓存和释放 encoder output？](07_encoder_cache_lifecycle.md)

回答：

```text
encoder cache 缓存什么？
请求何时命中 encoder cache？
哪些 encoder input 可以跳过？
finished request 后如何释放？
多模态和 encoder-decoder 是否共用这套机制？
```

### 08. ModelRunner 执行 multimodal encoder

- [ModelRunner 如何执行多模态 encoder？](08_model_runner_mm_encoder_flow.md)

回答：

```text
_execute_mm_encoder() 在哪里调用？
ModelRunner 如何准备 mm kwargs？
encoder output 如何变成 inputs_embeds？
多模态 forward 和普通文本 forward 如何汇合？
```

### 09. 多模态模型接口与模型实现

- [多模态模型类需要提供什么接口？](09_multimodal_model_interfaces.md)

回答：

```text
vLLM model registry 如何识别多模态模型？
模型类如何声明 supported modalities？
get_multimodal_embeddings / get_input_embeddings / merge_multimodal_embeddings 这类接口如何协作？
```

### 10. Multimodal 与输出、缓存、并行的关系

- [Multimodal 如何影响输出、KV cache、并行和高级能力？](10_multimodal_runtime_interactions.md)

回答：

```text
多模态请求最终输出和文本请求有什么不同？
多模态 encoder cache 与 decoder KV cache 如何区分？
PP / TP / DP 下多模态输入如何传递？
LoRA、prefix cache、KV transfer、spec decode 与多模态有什么边界？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
multimodel_overview.md
  → 01_multimodal_role.md
  → 02_multimodal_request_entry.md
  → 08_model_runner_mm_encoder_flow.md
```

### 3.2 按完整请求链路阅读

```text
multimodel_overview.md
  → 02_multimodal_request_entry.md
  → 03_parser_processor_mapper.md
  → 04_placeholders_and_prompt_alignment.md
  → 05_feature_spec_and_cache.md
  → 06_encoder_budget_and_scheduler.md
  → 08_model_runner_mm_encoder_flow.md
  → 10_multimodal_runtime_interactions.md
```

### 3.3 和执行层联动阅读

```text
../engine/04_input_processor.md
  → ../engine_core/02_request_entry.md
  → ../scheduler/07_auxiliary_scheduling_features.md
  → ../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
  → 08_model_runner_mm_encoder_flow.md
```

### 3.4 和模型加载联动阅读

```text
../config_and_model_loading/03_model_config_and_hf_config.md
  → ../config_and_model_loading/05_model_registry_and_arch_resolution.md
  → 09_multimodal_model_interfaces.md
```

---

## 4. 文档定位

```text
README.md：
  当前目录索引和阅读路线。

multimodel_overview.md：
  总览主文档，适合快速建立多模态链路全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读多模态输入、调度、执行和模型接口源码。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
Multimodal 负责把 image / audio / video 等非文本输入先变成模型可用的 feature / embedding，再通过 Scheduler 和 ModelRunner 接入普通 decoder forward；它不改变最终输出链路，本质上改变的是输入准备和 encoder cache 链路。
```
