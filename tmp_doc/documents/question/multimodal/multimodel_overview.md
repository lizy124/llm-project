# vLLM V1 Multimodal 逻辑梳理

源码位置：

- `code/vllm/vllm/multimodal/`
- `code/vllm/vllm/assets/`
- `code/vllm/vllm/config/multimodal.py`
- `code/vllm/vllm/entrypoints/`
- `code/vllm/vllm/renderers/`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/core/sched/`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/models/`

本文用于总览 vLLM V1 的多模态链路，重点梳理 image / audio / video / embeds 等输入如何从 entrypoints 进入 vLLM，如何被 parser / processor 转成 feature，如何通过 Scheduler 调度 encoder input，如何在 ModelRunner 中执行 multimodal encoder，如何进入 decoder forward，并最终接入普通 sampling / output 链路。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本文按“先定角色，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式组织。

要回答的问题分成 10 组：

```text
1. Multimodal 子系统在 vLLM V1 中是哪一层？负责什么，不负责什么？
2. 多模态用户输入如何进入 EngineInput / EngineCoreRequest？
3. MultiModalDataParser / processor / mapper 各自负责什么？
4. placeholder token 如何和 prompt token 对齐？
5. MultiModalFeatureSpec / MultiModalCache 如何组织？
6. Scheduler 如何估算 encoder budget 并调度 encoder input？
7. EncoderCacheManager 如何缓存和释放 encoder output？
8. ModelRunner 如何执行 multimodal encoder 并合并 inputs_embeds？
9. 多模态模型类需要提供哪些接口？
10. 多模态链路如何影响输出、KV cache、并行、LoRA、spec decode、KV transfer？
```

阅读顺序建议：

```text
multimodel_overview.md
  → 01_multimodal_role.md
  → 02_multimodal_request_entry.md
  → 03_parser_processor_mapper.md
  → 04_placeholders_and_prompt_alignment.md
  → 05_feature_spec_and_cache.md
  → 06_encoder_budget_and_scheduler.md
  → 07_encoder_cache_lifecycle.md
  → 08_model_runner_mm_encoder_flow.md
  → 09_multimodal_model_interfaces.md
  → 10_multimodal_runtime_interactions.md
```

---

## 1. 一句话总览占位

占位：后续补充 Multimodal 子系统在 vLLM V1 主链路中的整体定位。

```text
用户请求 image / audio / video / prompt_embeds
  → entrypoints / renderer
  → InputProcessor
  → MultiModalDataParser
  → processor / mapper
  → MultiModalFeatureSpec
  → EngineCoreRequest.mm_features
  → Request.mm_features
  → Scheduler.schedule()
  → EncoderCacheManager / encoder budget
  → SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._preprocess()
  → _execute_mm_encoder()
  → encoder output / multimodal embeddings
  → inputs_embeds
  → model forward
  → logits / pooling
  → sampling / output
```

一句话记忆占位：

```text
多模态链路负责“把非文本输入变成模型 forward 能消费的 embedding / encoder output”，最终仍然汇入普通的 ModelRunner forward 和 output 链路。
```

---

## 2. 核心角色占位

后续补充以下组件职责边界：

```text
MultiModalConfig：
  控制多模态能力开关、输入限制、processor 参数、缓存策略等。

Renderer / entrypoints：
  把 OpenAI / offline API 的 image、audio、video、content parts 转成 vLLM EngineInput。

InputProcessor：
  将 EngineInput 转成 EngineCoreRequest，并把多模态数据放入 mm_features / placeholders。

MultiModalDataParser：
  解析原始多模态数据，按 modality 分发。

Processor / mapper：
  调用模型对应 processor，把原始 media 转成模型需要的 tensor / feature。

MultiModalFeatureSpec：
  描述一个多模态 feature 的 hash、modality、位置、输入张量和后续 encoder 需求。

MultiModalCache：
  缓存 processor 输出，避免重复处理相同 media。

MultiModalBudget：
  估算 encoder token budget，帮助 Scheduler 决定本轮能处理哪些 encoder input。

EncoderCacheManager：
  缓存和释放 encoder output，避免重复执行 multimodal encoder。

GPUModelRunner：
  在 `_preprocess()` 阶段执行 multimodal encoder，并把结果合并成 inputs_embeds。

Multimodal model class：
  声明支持的 modality，并提供多模态 embedding / merge 接口。
```

---

## 3. 主链路占位

```text
OpenAI chat content / offline prompt
  → render_chat / render_completion
  → EngineInput
  → InputProcessor.process_inputs()
  → EngineCoreRequest(
        prompt_token_ids,
        mm_features,
        sampling_params / pooling_params,
    )
  → EngineCore.add_request()
  → Request(mm_features, ...)
  → Scheduler.schedule()
  → scheduled_encoder_inputs
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._preprocess()
  → maybe_get_ec_connector_output()
  → _execute_mm_encoder()
  → _gather_mm_embeddings()
  → inputs_embeds
  → _model_forward()
```

---

## 4. 和已有专题的关系占位

多模态链路需要和以下专题联动阅读：

```text
../engine/04_input_processor.md
  解释外部输入如何变成 EngineCoreRequest。

../engine_core/02_request_entry.md
  解释 EngineCoreRequest 如何进入 EngineCore / Scheduler。

../scheduler/07_auxiliary_scheduling_features.md
  解释 encoder input、grammar、spec decode 等辅助调度能力。

../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
  解释 ModelRunner `_preprocess()` 如何合并多模态输入。

../executor_worker_model_runner/07_model_forward_and_logits.md
  解释最终 forward 和 logits 位置。

../config_and_model_loading/03_model_config_and_hf_config.md
  解释 ModelConfig 如何识别模型能力。

../config_and_model_loading/05_model_registry_and_arch_resolution.md
  解释模型 registry 如何解析多模态模型类。
```

---

## 5. 文档定位占位

```text
multimodel_overview.md：
  总览主文档，适合快速建立多模态链路全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读多模态输入解析、调度、encoder cache 和执行链路。
```

---

## 6. 后续待补源码证据

占位：后续逐段补充源码位置、关键类、关键字段、关键状态迁移和例子。
