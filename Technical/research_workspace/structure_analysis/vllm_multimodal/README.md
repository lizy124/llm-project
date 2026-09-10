# vLLM 多模态链路梳理

本目录梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的 multimodal 相关代码，重点覆盖：

- OpenAI Chat / Python Prompt 中图片、音频、视频、embedding 输入如何进入 vLLM；
- `MultiModalConfig`、`limit_mm_per_prompt`、media connector、parser、processor、dummy inputs 的职责；
- prompt placeholder、prompt replacement、token-space/text-space 更新、`MultiModalKwargs` 与 `MultiModalFeatureSpec` 的关系；
- V1 engine 中 `InputProcessor`、`Request.mm_features`、scheduler、encoder budget、encoder cache 的运行机制；
- GPU worker 如何执行多模态 encoder、缓存 encoder output，并把多模态 embedding 合入语言模型输入；
- 多模态模型接口、典型模型族差异、processor registry 与模型实现关系；
- 多模态哈希、UUID、processor cache、receiver cache、SHM cache、encoder cache 的区别；
- 多模态限制、错误场景、调试地图与阅读顺序。

## 文档索引

1. [01_multimodal_overview.md](01_multimodal_overview.md)
   - vLLM 多模态整体分层、核心对象、端到端主链路。
2. [02_entry_and_multimodal_prompt_input.md](02_entry_and_multimodal_prompt_input.md)
   - OpenAI Chat、renderer、Python Prompt、`multi_modal_data` 如何进入预处理链。
3. [03_multimodal_config_and_media_parsing.md](03_multimodal_config_and_media_parsing.md)
   - `MultiModalConfig`、数量限制、media connector、`MultiModalDataParser`。
4. [04_processor_and_placeholder_mechanism.md](04_processor_and_placeholder_mechanism.md)
   - `BaseMultiModalProcessor`、prompt update、placeholder range、token merge。
5. [05_engine_request_and_mm_features.md](05_engine_request_and_mm_features.md)
   - `InputProcessor` 如何把 processor 输出转成 `MultiModalFeatureSpec` 并进入 `Request`。
6. [06_Scheduler_EncoderBudget_Cache.md](06_Scheduler_EncoderBudget_Cache.md)
   - encoder budget、scheduler admission、scheduler 侧 encoder cache 生命周期。
7. [07_gpu_worker_encoder_execution_and_embedding_merge.md](07_gpu_worker_encoder_execution_and_embedding_merge.md)
   - GPU worker 组 batch、执行 `embed_multimodal`、gather embedding、写回输入 embedding。
8. [08_multimodal_model_interfaces_and_impls.md](08_multimodal_model_interfaces_and_impls.md)
   - `SupportsMultiModal`、`embed_multimodal`、LLaVA/Qwen2-VL/Qwen2-Audio/Whisper/Pixtral/Gemma3 等。
9. [09_hash_uuid_processor_cache_and_encoder_cache.md](09_hash_uuid_processor_cache_and_encoder_cache.md)
   - `MultiModalHasher`、UUID、`mm_hash`、`identifier`、processor cache 与 encoder cache 区别。
10. [10_limitations_errors_and_debug_map.md](10_limitations_errors_and_debug_map.md)
    - 限制校验、常见错误、测试覆盖、调试入口。
11. [question_and_answer.md](question_and_answer.md)
    - 高频问题与定位答案。

## 一句话总结

vLLM 的多模态链路可以理解为：入口层把 OpenAI/Python 图片、音频、视频等输入挂到 prompt 上；processor 层把原始媒体解析成模型可用 tensor kwargs，并把 prompt 中的多模态占位定位成 token 区间；engine/scheduler 层围绕这些占位区间调度 encoder 计算和缓存复用；GPU worker 执行多模态 encoder 并在 embedding 层把 encoder 输出写回语言模型输入序列。
