# vLLM 配置与模型加载链路梳理

本目录梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的 `config_and_model_loading` 相关代码，重点覆盖：

- CLI / Python API 参数如何进入 `EngineArgs`；
- `EngineArgs.create_engine_config()` 如何构造 `VllmConfig`；
- `VllmConfig` 如何聚合、校验、修正各子配置；
- `ModelConfig` 如何加载 HuggingFace / Mistral / ModelScope 配置；
- `ModelArchitectureConfig` 如何把不同模型族的 HF config 归一化；
- `LoadConfig` 如何决定模型权重加载格式；
- `model_loader` 如何实例化模型类、发现权重文件、迭代权重、调用 `load_weights()`；
- 默认 loader、BitsAndBytes、ShardedState、Tensorizer、RunAI、Dummy、ModelExpress 等加载路径差异；
- 量化、并行、reload、post-process 对模型加载链路的影响；
- 从配置构建到 worker/runtime 模型加载的端到端调用链。

## 文档索引

1. [01_配置与模型加载总览.md](01_配置与模型加载总览.md)
   - 配置链路与模型加载链路的整体分层、核心对象、端到端总图。
2. [02_EngineArgs到VllmConfig.md](02_EngineArgs到VllmConfig.md)
   - CLI / Python API 到 `EngineArgs`，再到 `VllmConfig` 的构建过程。
3. [03_VllmConfig与子配置体系.md](03_VllmConfig与子配置体系.md)
   - `VllmConfig` 字段、`__post_init__()`、跨配置校验与默认策略修正。
4. [04_ModelConfig与HF配置加载.md](04_ModelConfig与HF配置加载.md)
   - `ModelConfig`、`get_config()`、HF/Mistral/ModelScope 配置解析、`hf_overrides`、`trust_remote_code`。
5. [05_ModelArchitectureConfig归一化.md](05_ModelArchitectureConfig归一化.md)
   - `ModelArchConfigConvertor` 如何把不同模型族配置转换成 vLLM 统一架构配置。
6. [06_LoadConfig与ModelLoader选择.md](06_LoadConfig与ModelLoader选择.md)
   - `LoadConfig` 与 `load_format -> loader` 映射，所有主要 loader 的职责差异。
7. [07_模型注册与实例化链路.md](07_模型注册与实例化链路.md)
   - `models/registry.py`、`initialize_model()`、模型接口协议与 architecture 解析。
8. [08_权重发现过滤与迭代.md](08_权重发现过滤与迭代.md)
   - checkpoint 文件发现、下载、本地缓存、safetensors/bin/pt/np_cache/streaming iterator。
9. [09_Worker到ModelRunner加载链路.md](09_Worker到ModelRunner加载链路.md)
   - V1 worker、GPUModelRunner、load/reload、post-process 与运行时边界。
10. [10_量化并行Reload与调试地图.md](10_量化并行Reload与调试地图.md)
    - 量化、TP/PP/EP/DP、layerwise reload、常见调试入口与阅读顺序。
11. [question_and_answer.md](question_and_answer.md)
    - 高频问题与定位答案。

## 一句话总结

vLLM 的配置与模型加载可以理解为两条最终汇合的链：配置链负责把 CLI/Python API 输入规范化成 `VllmConfig`，模型加载链负责根据 `VllmConfig.model_config` 与 `VllmConfig.load_config` 选择模型类和 loader，把 checkpoint 权重转成可运行的 `nn.Module`；两者在 `Worker / ModelRunner.load_model()` 处汇合，最终服务于调度器和执行器的实际推理运行。
