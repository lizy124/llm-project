# 01 配置与模型加载总览

本篇先给出 vLLM `config_and_model_loading` 的整体结构：一条是“配置构建链”，一条是“模型加载链”。二者不是彼此独立的文档概念，而是在运行时真实汇合：`VllmConfig` 作为总配置对象被传入 worker/model runner，模型加载器再根据其中的 `ModelConfig`、`LoadConfig`、`ParallelConfig`、`CacheConfig`、`CompilationConfig` 等配置决定怎么实例化模型、怎么读权重、怎么做后处理。

## 1. 两条主链路

### 1.1 配置构建链

```text
CLI 参数 / Python LLM(...) 参数
  ↓
EngineArgs
  ↓
EngineArgs.create_engine_config()
  ↓
ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig / ...
  ↓
VllmConfig
  ↓
VllmConfig.__post_init__()
  ↓
跨配置校验、默认值补全、平台修正、编译/调度/KV cache 联动
```

关键入口：

- Python API：`code/vllm/vllm/entrypoints/llm.py:305`
- `LLMEngine.from_engine_args(...)`：`code/vllm/vllm/entrypoints/llm.py:349`
- CLI 参数定义：`code/vllm/vllm/engine/arg_utils.py:779`
- CLI namespace 转 `EngineArgs`：`code/vllm/vllm/engine/arg_utils.py:1560`
- 配置构建主入口：`code/vllm/vllm/engine/arg_utils.py:1724`
- `VllmConfig` 定义：`code/vllm/vllm/config/vllm.py:296`
- `VllmConfig.__post_init__()`：`code/vllm/vllm/config/vllm.py:864`

配置链路的重点不是“保存参数”，而是把用户输入转成引擎可运行的结构化配置，并在总配置层完成跨模块校验。

### 1.2 模型加载链

```text
Worker.load_model()
  ↓
GPUModelRunner.load_model()
  ↓
get_model_loader(load_config)
  ↓
BaseModelLoader.load_model(vllm_config, model_config)
  ↓
initialize_model(vllm_config, prefix="")
  ↓
model_config.registry.resolve_model_cls(...)
  ↓
实例化具体 vLLM model class
  ↓
loader.load_weights(model, model_config)
  ↓
model.load_weights(weights_iterator)
  ↓
process_weights_after_loading(...)
  ↓
model.eval()
```

关键入口：

- V1 worker 加载入口：`code/vllm/vllm/v1/worker/gpu_worker.py:349`
- GPU model runner 加载入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`
- loader 分发表：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`
- `get_model()` 包装函数：`code/vllm/vllm/model_executor/model_loader/__init__.py:130`
- loader 生命周期模板：`code/vllm/vllm/model_executor/model_loader/base_loader.py:42`
- 模型实例化：`code/vllm/vllm/model_executor/model_loader/utils.py:40`
- 加载后处理：`code/vllm/vllm/model_executor/model_loader/utils.py:100`

模型加载链路的关键分工是：registry 决定“哪个模型类”，loader 决定“怎么拿权重”，具体模型类的 `load_weights()` 决定“权重如何落到层上”。

## 2. 核心对象关系

| 对象 | 所在文件 | 职责 |
|---|---|---|
| `EngineArgs` | `code/vllm/vllm/engine/arg_utils.py:411` | 收拢 CLI/Python API 输入，提供 `create_*_config()` 方法。 |
| `VllmConfig` | `code/vllm/vllm/config/vllm.py:296` | vLLM 引擎总配置，聚合所有子配置并做跨配置后处理。 |
| `ModelConfig` | `code/vllm/vllm/config/model.py:100` | 模型语义、HF config、dtype、max len、quant、task/runner 等核心配置。 |
| `LoadConfig` | `code/vllm/vllm/config/load.py:25` | 权重加载格式、下载目录、ignore pattern、safetensors 策略。 |
| `ParallelConfig` | `code/vllm/vllm/config/parallel.py:116` | TP/PP/DP/DCP/EP 等并行维度与 executor backend。 |
| `CacheConfig` | `code/vllm/vllm/config/cache.py:42` | KV cache block size、cache dtype、prefix caching、offload 相关策略。 |
| `SchedulerConfig` | `code/vllm/vllm/config/scheduler.py:25` | batching、chunked prefill、partial prefill、async scheduling 等。 |
| `CompilationConfig` | `code/vllm/vllm/config/compilation.py:378` | `torch.compile`、cudagraph、splitting ops、compile ranges。 |
| `ModelRegistry` | `code/vllm/vllm/model_executor/models/registry.py:987` | architecture 名称到 vLLM 模型类的注册与解析。 |
| `BaseModelLoader` | `code/vllm/vllm/model_executor/model_loader/base_loader.py:37` | 所有模型加载器的生命周期模板。 |
| `DefaultModelLoader` | `code/vllm/vllm/model_executor/model_loader/default_loader.py:382` | 标准 HF/local checkpoint 加载主路径。 |
| `weight_utils` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:154` | 权重下载、文件过滤、iterator、量化配置读取等公共工具。 |

## 3. 为什么配置链和加载链不能分开看

很多加载行为并不是 loader 自己决定的，而是来自配置链的结果：

- `LoadConfig.load_format` 决定 `get_model_loader()` 返回哪个 loader；
- `ModelConfig.hf_config` 和 `model_arch_config` 决定 registry 如何选择模型类；
- `ModelConfig.dtype` 决定 loader 在什么 dtype context 下初始化模型；
- `ParallelConfig` 决定 TP/PP/EP/DP rank 相关的权重切分与过滤；
- `CacheConfig.cache_dtype` 影响 attention/KV cache 相关权重后处理；
- `CompilationConfig` 和 scheduler 参数影响 cudagraph、compile shape、dummy weight 预热等运行路径；
- `LoRAConfig`、`MultiModalConfig`、`SpeculativeConfig` 等配置会影响模型能力判断和 runner 行为。

因此，vLLM 中“模型加载失败”经常不能只查 `model_loader`，还要反查 `ModelConfig`、`LoadConfig`、`ParallelConfig`、`VllmConfig.__post_init__()` 的推导结果。

## 4. 端到端总图

```text
用户入口层
  ├─ Python: LLM(...)
  └─ CLI: vllm serve / argparse
        ↓
EngineArgs
  ├─ create_model_config()
  ├─ create_cache_config()
  ├─ create_parallel_config()
  ├─ create_scheduler_config()
  ├─ create_load_config()
  └─ create_compilation_config()
        ↓
VllmConfig
  ├─ __post_init__ 平台默认值
  ├─ scheduler / compile / cudagraph 推导
  ├─ model-parallel-load-lora-quant 校验
  └─ block size / kv / mamba / spec decode 限制
        ↓
Executor / Worker / ModelRunner
        ↓
GPUModelRunner.load_model()
        ↓
get_model_loader(load_config)
        ↓
BaseModelLoader.load_model()
  ├─ initialize_model()
  │   └─ registry.resolve_model_cls()
  ├─ loader.load_weights()
  │   ├─ _prepare_weights()
  │   ├─ filter files
  │   ├─ weights iterator
  │   └─ model.load_weights()
  ├─ finalize_layerwise_processing()
  ├─ process_weights_after_loading()
  └─ model.eval()
```

## 5. 关键阅读顺序

如果从零开始读本主题，建议按下面顺序：

```text
1. code/vllm/vllm/entrypoints/llm.py
2. code/vllm/vllm/engine/arg_utils.py
3. code/vllm/vllm/config/vllm.py
4. code/vllm/vllm/config/model.py
5. code/vllm/vllm/transformers_utils/config.py
6. code/vllm/vllm/transformers_utils/model_arch_config_convertor.py
7. code/vllm/vllm/config/load.py
8. code/vllm/vllm/model_executor/model_loader/__init__.py
9. code/vllm/vllm/model_executor/model_loader/base_loader.py
10. code/vllm/vllm/model_executor/model_loader/default_loader.py
11. code/vllm/vllm/model_executor/model_loader/weight_utils.py
12. code/vllm/vllm/model_executor/models/registry.py
13. code/vllm/vllm/v1/worker/gpu_worker.py
14. code/vllm/vllm/v1/worker/gpu_model_runner.py
```

## 6. 一句话总结

vLLM 的 `config_and_model_loading` 不是单个模块，而是从入口参数、配置总装、HF config 解析、架构归一化、loader 选择、模型注册、权重文件 I/O、量化/并行后处理一直延伸到 worker runtime 的完整启动链路；理解这条链，才能准确定位“模型为什么这样配置、为什么这样加载、为什么在某个 rank 上只看到部分权重”。
