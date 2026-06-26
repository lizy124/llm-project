# Config and Model Loading 阅读索引

本目录用于梳理 vLLM 的配置系统与模型加载链路。

参考 `executor_worker_model_runner` 目录的组织方式，本目录先建立骨架，后续逐篇补充源码证据、关键字段、状态流转和示例。

阅读顺序建议：

```text
config_and_model_loading_overview.md
  → 01_config_entry_and_engine_args.md
  → 02_vllm_config_aggregation.md
  → 03_model_config_and_hf_config.md
  → 04_load_config_and_model_loader.md
  → 05_model_registry_and_arch_resolution.md
  → 06_weight_loading_and_quantization.md
  → 07_worker_load_model_flow.md
  → 08_model_layers_and_execution_interface.md
  → 09_advanced_config_hooks.md
  → 10_config_to_runtime_lifecycle.md
```

核心问题：

```text
1. 用户参数如何进入 vLLM？
2. EngineArgs 如何转换成 VllmConfig？
3. VllmConfig 如何把 Model / Cache / Parallel / Scheduler / Load 等配置串起来？
4. ModelConfig 如何读取和修正 Hugging Face config？
5. model registry 如何根据 architectures 选择模型类？
6. LoadConfig 如何决定权重加载器和权重格式？
7. Worker.load_model() 如何触发真正模型实例化和权重加载？
8. 量化、LoRA、多模态、Spec Decode、Compilation 如何挂到配置和模型加载链路？
```
