# vLLM Config and Model Loading 问题目录

源码位置：

- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/config/`
- `code/vllm/vllm/transformers_utils/config.py`
- `code/vllm/vllm/transformers_utils/config_parser_base.py`
- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

这个目录用于梳理 vLLM 的配置系统与模型加载链路，重点回答：用户参数如何进入 `EngineArgs`，`EngineArgs` 如何构造 `VllmConfig`，`ModelConfig` 如何加载和修正 Hugging Face config，`LoadConfig` 如何选择模型权重加载器，模型类如何通过 registry 解析，最终 `Worker.load_model()` 如何触发模型实例化和权重加载。

---

## 1. 总览文档

- [vLLM Config and Model Loading 逻辑梳理](config_and_model_loading_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
CLI / Python API
  → EngineArgs
  → EngineArgs.create_engine_config()
  → ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig / LoadConfig / ...
  → VllmConfig
  → Executor / Worker 初始化
  → Worker.load_model()
  → ModelRunner.load_model()
  → model_loader.get_model()
  → get_model_loader(load_config)
  → BaseModelLoader.load_model()
  → initialize_model()
  → ModelRegistry / architecture resolution
  → model.load_weights(weights_iterator)
  → process_weights_after_loading()
```

---

## 2. 主线专题阅读顺序

### 01. 配置入口与 EngineArgs

- [用户参数如何进入 vLLM？](01_config_entry_and_engine_args.md)

回答：

```text
EngineArgs 是什么？
CLI 参数如何被注册和解析？
Python API 如何绕过 CLI 直接构造 EngineArgs？
EngineArgs 和各个 Config 类是什么关系？
```

关键源码：

- `code/vllm/vllm/engine/arg_utils.py:411`
- `code/vllm/vllm/engine/arg_utils.py:779`
- `code/vllm/vllm/engine/arg_utils.py:1559`

`EngineArgs` 是用户输入进入引擎配置系统的集中入口。它本身不是运行时最终配置，而是把 CLI / API 层的扁平参数收集起来，再在 `create_engine_config()` 中拆分、校验、补默认值，并组装成 `VllmConfig`。

---

### 02. VllmConfig 聚合层

- [VllmConfig 如何把各类配置串起来？](02_vllm_config_aggregation.md)

回答：

```text
VllmConfig 聚合哪些子配置？
为什么运行时普遍传 VllmConfig 而不是单个 config？
compute_hash() 为什么重要？
VllmConfig 如何影响编译、CUDA graph、ModelRunner 选择？
```

关键源码：

- `code/vllm/vllm/config/vllm.py:297`
- `code/vllm/vllm/config/vllm.py:393`
- `code/vllm/vllm/config/vllm.py:528`

`VllmConfig` 是运行时配置总容器，包含：

```text
- model_config
- cache_config
- parallel_config
- scheduler_config
- device_config
- load_config
- offload_config
- attention_config
- mamba_config
- kernel_config
- lora_config
- speculative_config
- diffusion_config
- structured_outputs_config
- observability_config
- quant_config
- compilation_config
- profiler_config
- kv_transfer_config / kv_events_config / ec_transfer_config
- reasoning_config
- additional_config
- optimization_level / performance_mode
- weight_transfer_config
```

一句话记忆：

```text
EngineArgs 是入口参数，VllmConfig 是运行时配置快照。
```

---

### 03. ModelConfig 与 Hugging Face config

- [ModelConfig 如何读取和修正模型配置？](03_model_config_and_hf_config.md)

回答：

```text
ModelConfig 初始化时做了什么？
hf_config / hf_text_config / model_arch_config 分别是什么？
config_format=auto 如何判断 hf / mistral？
hf_overrides 如何生效？
模型 runner_type / convert_type 如何确定？
```

关键源码：

- `code/vllm/vllm/config/model.py:101`
- `code/vllm/vllm/config/model.py:458`
- `code/vllm/vllm/config/model.py:534`
- `code/vllm/vllm/config/model.py:557`
- `code/vllm/vllm/config/model.py:593`
- `code/vllm/vllm/config/model.py:639`
- `code/vllm/vllm/config/model.py:656`
- `code/vllm/vllm/transformers_utils/config.py:653`

`ModelConfig.__post_init__()` 是模型配置链路的核心。它会：

```text
1. 规范 model / tokenizer / hf_config_path；
2. 处理 tokenizer 默认值和 revision；
3. 处理 RunAI / object storage 场景；
4. 调用 get_config() 加载 HF / Mistral config；
5. 应用 hf_overrides；
6. 拆出 hf_text_config；
7. 转换成 vLLM 的 model_arch_config；
8. 读取 image processor config；
9. 通过 ModelRegistry 判断 generate / pooling / multimodal 能力；
10. 决定 runner_type 和 convert_type；
11. inspect 模型类并缓存 _model_info / _architecture；
12. 推断 tokenizer_mode；
13. 初始化 pooler_config / multimodal_config；
14. 解析 dtype；
15. 推导并校验 max_model_len；
16. 校验量化、CUDA graph、bitsandbytes 等配置。
```

最关键的一点是：

```text
ModelConfig 不只是保存模型名，它会主动读取模型仓库 config，并把外部 HF 配置转换成 vLLM 后续可执行的内部模型信息。
```

---

### 04. Config parser 与 config_format

- [get_config() 如何选择 HF / Mistral / 自定义 parser？](04_config_parser_and_format.md)

回答：

```text
config_format=auto 的判断顺序是什么？
HFConfigParser / MistralConfigParser 如何接入？
register_config_parser() 如何扩展？
RoPE / quantization_config / architectures 如何补齐？
```

关键源码：

- `code/vllm/vllm/transformers_utils/config.py:360`
- `code/vllm/vllm/transformers_utils/config.py:372`
- `code/vllm/vllm/transformers_utils/config.py:379`
- `code/vllm/vllm/transformers_utils/config.py:653`
- `code/vllm/vllm/transformers_utils/config_parser_base.py:10`

`get_config()` 的主流程是：

```text
config_format == auto
  → 如果是 Mistral repo 且有 params.json，使用 mistral parser
  → 否则如果有 config.json，使用 hf parser
  → 否则报错，提示显式指定自定义 config parser
  → get_config_parser(config_format).parse(...)
  → 补 architectures
  → 读取 quantization_config / hf_quant_config.json
  → 应用 hf_overrides
  → patch RoPE 参数
  → trust_remote_code 时注册按值序列化
  → 返回 PretrainedConfig
```

一句话记忆：

```text
transformers_utils/config.py 负责把“模型仓库里的配置文件”变成标准 PretrainedConfig。
```

---

### 05. EngineArgs.create_engine_config 主链路

- [EngineArgs 如何构造完整 VllmConfig？](05_create_engine_config_flow.md)

回答：

```text
create_engine_config() 的阶段顺序是什么？
DeviceConfig / ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig / LoadConfig 何时构造？
哪些默认值必须等 ModelConfig 加载后才能确定？
```

关键源码：

- `code/vllm/vllm/engine/arg_utils.py:1724`
- `code/vllm/vllm/engine/arg_utils.py:1734`
- `code/vllm/vllm/engine/arg_utils.py:1757`
- `code/vllm/vllm/engine/arg_utils.py:1781`
- `code/vllm/vllm/engine/arg_utils.py:1984`
- `code/vllm/vllm/engine/arg_utils.py:2056`
- `code/vllm/vllm/engine/arg_utils.py:2194`
- `code/vllm/vllm/engine/arg_utils.py:2255`

主流程可以压缩成：

```text
create_engine_config()
  → current_platform.pre_register_and_update()
  → DeviceConfig(current_platform.device_type)
  → env validation
  → maybe_override_with_speculators()
  → create_model_config()
  → 根据 ModelConfig 设置默认 chunked prefill / prefix caching / reasoning
  → 根据 model_config 推导 sliding_window / kv_cache_dtype
  → CacheConfig(...)
  → 推导 DP / TP / PP / Ray / placement group 信息
  → ParallelConfig(...)
  → create_speculative_config() / create_diffusion_config()
  → 设置 max_num_batched_tokens / max_num_seqs 默认值
  → SchedulerConfig(...)
  → LoRAConfig(...)
  → AttentionConfig / MambaConfig / KernelConfig overrides
  → create_load_config()
  → ObservabilityConfig / CompilationConfig / OffloadConfig
  → VllmConfig(...)
```

这一步是“入口参数”到“运行时总配置”的分水岭。

---

### 06. 核心子配置字段

- [各个 Config 类分别控制什么？](06_core_config_objects.md)

回答：

```text
ModelConfig 控制什么？
CacheConfig 控制什么？
ParallelConfig 控制什么？
SchedulerConfig 控制什么？
DeviceConfig 控制什么？
LoadConfig 控制什么？
```

关键源码：

- `code/vllm/vllm/config/model.py:101`
- `code/vllm/vllm/config/cache.py:43`
- `code/vllm/vllm/config/parallel.py:117`
- `code/vllm/vllm/config/scheduler.py:26`
- `code/vllm/vllm/config/device.py:17`
- `code/vllm/vllm/config/load.py:26`

最小职责划分：

```text
ModelConfig：
  模型身份、tokenizer、HF config、runner 类型、dtype、max_model_len、量化、多模态、pooling 能力。

CacheConfig：
  KV cache block size、显存利用率、KV dtype、prefix caching、sliding window、Mamba cache、KV offload。

ParallelConfig：
  TP / PP / DP / PCP / DCP、分布式 executor、Ray / mp、多节点、EP / EPLB、worker 类。

SchedulerConfig：
  max_num_batched_tokens、max_num_seqs、chunked prefill、调度策略、scheduler 类、async scheduling、stream interval。

DeviceConfig：
  根据 current_platform 推断 device_type，并转换成 torch.device 或 host-device handling。

LoadConfig：
  load_format、download_dir、safetensors 加载策略、model_loader_extra_config、ignore_patterns、pt map_location。
```

---

### 07. LoadConfig 与 model loader 选择

- [LoadConfig 如何决定权重加载器？](07_load_config_and_model_loader.md)

回答：

```text
load_format 支持哪些值？
get_model_loader() 如何映射 loader？
DefaultModelLoader / BitsAndBytesModelLoader / ShardedStateLoader / TensorizerLoader 分别什么时候用？
自定义 loader 如何注册？
```

关键源码：

- `code/vllm/vllm/config/load.py:26`
- `code/vllm/vllm/model_executor/model_loader/__init__.py:33`
- `code/vllm/vllm/model_executor/model_loader/__init__.py:50`
- `code/vllm/vllm/model_executor/model_loader/__init__.py:69`
- `code/vllm/vllm/model_executor/model_loader/__init__.py:122`
- `code/vllm/vllm/model_executor/model_loader/__init__.py:130`

`load_format` 到 loader 的核心映射是：

```text
auto / hf / safetensors / fastsafetensors / instanttensor / mistral / npcache / pt
  → DefaultModelLoader

bitsandbytes
  → BitsAndBytesModelLoader

dummy
  → DummyModelLoader

modelexpress
  → ModelExpressModelLoader

runai_streamer
  → RunaiModelStreamerLoader

runai_streamer_sharded / sharded_state
  → ShardedStateLoader

tensorizer
  → TensorizerLoader
```

最小主链路：

```text
get_model(vllm_config)
  → get_model_loader(vllm_config.load_config)
  → loader.load_model(vllm_config, model_config)
```

---

### 08. 模型类解析与 ModelRegistry

- [vLLM 如何根据 architectures 选择模型类？](08_model_registry_and_arch_resolution.md)

回答：

```text
architectures 从哪里来？
ModelRegistry 如何判断 text generation / pooling / multimodal？
inspect_model_cls() 与 load_model_cls() 有什么区别？
Transformers backend fallback 如何接入？
```

关键源码：

- `code/vllm/vllm/config/model.py:557`
- `code/vllm/vllm/config/model.py:593`
- `code/vllm/vllm/config/model.py:806`
- `code/vllm/vllm/model_executor/models/registry.py:805`
- `code/vllm/vllm/model_executor/models/registry.py:828`
- `code/vllm/vllm/model_executor/models/registry.py:947`
- `code/vllm/vllm/model_executor/model_loader/utils.py:41`
- `code/vllm/vllm/model_executor/model_loader/utils.py:218`

模型类解析有两段：

```text
配置阶段：
  ModelConfig.__post_init__()
    → registry.is_text_generation_model(...)
    → registry.is_pooling_model(...)
    → registry.inspect_model_cls(...)
    → 缓存 _model_info / _architecture

加载阶段：
  initialize_model()
    → get_model_architecture(model_config)
    → get_model_cls(model_config)
    → 构造 nn.Module
```

这意味着：

```text
配置阶段会先“检查并确定模型类型”，加载阶段才真正“实例化模型类”。
```

---

### 09. BaseModelLoader.load_model 标准流程

- [模型实例化和权重加载在哪里发生？](09_base_model_loader_flow.md)

回答：

```text
BaseModelLoader.load_model() 做了什么？
initialize_model() 负责什么？
load_weights() 负责什么？
process_weights_after_loading() 为什么需要？
```

关键源码：

- `code/vllm/vllm/model_executor/model_loader/base_loader.py:25`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py:42`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py:53`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py:64`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py:75`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py:80`

标准加载流程是：

```text
BaseModelLoader.load_model()
  → 确定 load_device
  → set_default_torch_dtype(model_config.dtype)
  → 在 target_device 上 initialize_model()
  → log_model_inspection()
  → self.load_weights(model, model_config)
  → 如果有 online quant，finalize_layerwise_processing()
  → process_weights_after_loading()
  → model.eval()
```

这层定义的是所有 loader 的共同骨架。不同 loader 主要差异在：

```text
- download_model()
- load_weights()
- weights_iterator 如何产生
- 权重格式如何读取
```

---

### 10. DefaultModelLoader 与权重文件迭代

- [默认 loader 如何下载、筛选和迭代权重？](10_default_model_loader_weights.md)

回答：

```text
_prepare_weights() 如何选择 safetensors / bin / pt / mistral？
auto load_format 的判断逻辑是什么？
本地路径和远程 HF repo 如何统一？
weights_iterator 如何根据格式切换？
model.load_weights() 何时调用？
```

关键源码：

- `code/vllm/vllm/model_executor/model_loader/default_loader.py:43`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:97`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:120`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:161`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:184`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:211`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:288`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:381`

默认加载器的关键阶段：

```text
_prepare_weights()
  → maybe_download_from_modelscope()
  → 判断本地目录还是远程 repo
  → 根据 load_format 选择 allow_patterns
  → 远程时 download_weights_from_hf()
  → 本地时直接使用目录
  → glob 匹配权重文件
  → safetensors 时过滤 duplicate / index 中不需要的文件
  → bin / pt 时过滤推理不需要的文件
  → 返回 hf_folder, hf_weights_files, use_safetensors

_get_weights_iterator()
  → np_cache_weights_iterator
  → fastsafetensors_weights_iterator
  → instanttensor_weights_iterator
  → multi_thread_safetensors_weights_iterator
  → safetensors_weights_iterator
  → multi_thread_pt_weights_iterator
  → pt_weights_iterator
  → 给权重名加 prefix

load_weights()
  → 初始化 EP weight filter
  → model.load_weights(self.get_all_weights(...))
  → 记录加载耗时
  → 可选 track_weights_loading()
```

一句话记忆：

```text
DefaultModelLoader 负责把“磁盘或远程仓库里的 checkpoint 文件”变成 model.load_weights() 可消费的 (name, tensor) 迭代器。
```

---

### 11. Worker.load_model 到 ModelRunner.load_model

- [运行时如何触发真正的模型加载？](11_worker_load_model_flow.md)

回答：

```text
Executor / Worker 初始化后，谁调用 load_model？
GPU Worker 在 load_model 前后做什么？
GPUModelRunner.load_model() 如何调用 model_loader.get_model()？
load_dummy_weights 在哪里接入？
```

关键源码：

- `code/vllm/vllm/v1/worker/worker_base.py:138`
- `code/vllm/vllm/v1/worker/gpu_worker.py:349`
- `code/vllm/vllm/v1/worker/gpu_worker.py:760`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3198`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`
- `code/vllm/vllm/model_executor/model_loader/__init__.py:130`

主线可以记成：

```text
Executor 初始化 workers
  → Worker.init_device()
  → Worker.load_model()
  → GPUModelRunner.load_model()
  → model_loader.get_model(vllm_config)
  → BaseModelLoader.load_model()
  → initialize_model()
  → model.load_weights()
```

这说明配置链路和加载链路的交汇点在 Worker / ModelRunner：

```text
Worker 持有 VllmConfig，ModelRunner 使用其中的 ModelConfig / LoadConfig / DeviceConfig 来真正创建模型。
```

---

### 12. 量化、LoRA、多模态、Spec Decode 的接入点

- [高级功能如何挂到配置与模型加载链路？](12_advanced_config_hooks.md)

回答：

```text
量化配置在哪里解析？
LoRAConfig 什么时候创建？
多模态能力如何由模型类声明并转成 MultiModalConfig？
SpeculativeConfig 如何由 target model config 和 parallel config 构造？
CompilationConfig / AttentionConfig / KernelConfig 如何参与运行时？
```

关键源码：

- `code/vllm/vllm/config/model.py:197`
- `code/vllm/vllm/config/model.py:663`
- `code/vllm/vllm/config/model.py:723`
- `code/vllm/vllm/config/vllm.py:618`
- `code/vllm/vllm/engine/arg_utils.py:1679`
- `code/vllm/vllm/engine/arg_utils.py:2034`
- `code/vllm/vllm/engine/arg_utils.py:2084`
- `code/vllm/vllm/engine/arg_utils.py:2121`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py:381`

几个关键挂点：

```text
量化：
  ModelConfig 读取 quantization / quantization_config；
  VllmConfig.get_quantization_config() 根据 ModelConfig + LoadConfig 构造 QuantizationConfig；
  DefaultModelLoader.load_weights() 在 torchao 等场景调整 safetensors_load_strategy；
  BaseModelLoader.load_model() 在加载后处理 online quant。

LoRA：
  EngineArgs.create_engine_config() 在 enable_lora 时构造 LoRAConfig；
  后续由 Worker / ModelRunner 加载和管理 adapter。

多模态：
  ModelRegistry.inspect_model_cls() 判断 supports_multimodal；
  ModelConfig 初始化 MultiModalConfig；
  ModelRunner 在执行阶段处理 multimodal processor / encoder / embeds。

Spec Decode：
  create_engine_config() 先构造 target model_config / parallel_config；
  create_speculative_config() 再把它们注入 SpeculativeConfig。

Compilation / Attention / Kernel：
  EngineArgs 支持顶层参数和完整 config 对象两种传法；
  create_engine_config() 负责互斥检查、override 和最终放入 VllmConfig。
```

---

### 13. 配置到运行时生命周期

- [配置对象如何一路影响执行层？](13_config_to_runtime_lifecycle.md)

回答：

```text
VllmConfig 创建后被谁持有？
Executor / Worker / ModelRunner 各自使用哪些 config？
配置如何影响 KV cache、scheduler、模型加载、编译和执行？
```

主线关系：

```text
VllmConfig
  → Executor：选择 worker / backend / distributed 组织方式
  → Worker：初始化 device、distributed、profile、KV cache、sleep / wake_up
  → ModelRunner：持有 model_config / cache_config / scheduler_config / parallel_config
  → ModelLoader：使用 model_config / load_config / device_config 创建模型并加载权重
  → Scheduler：使用 scheduler_config / cache_config 管理 token budget 和 KV blocks
```

也就是说，配置不是只在启动时用一次，而是贯穿：

```text
启动参数解析
  → 模型 config 解析
  → 运行时配置聚合
  → worker 初始化
  → 模型实例化
  → 权重加载
  → KV cache 初始化
  → 每轮调度与执行
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
config_and_model_loading_overview.md
  → 01_config_entry_and_engine_args.md
  → 02_vllm_config_aggregation.md
  → 03_model_config_and_hf_config.md
```

适合先理解：

```text
用户参数如何变成 VllmConfig。
```

### 3.2 按配置构造链路完整阅读

```text
01_config_entry_and_engine_args.md
  → 03_model_config_and_hf_config.md
  → 04_config_parser_and_format.md
  → 05_create_engine_config_flow.md
  → 06_core_config_objects.md
```

适合理解：

```text
EngineArgs.create_engine_config() 为什么必须先创建 ModelConfig，再创建 CacheConfig / SchedulerConfig / ParallelConfig。
```

### 3.3 按模型加载链路完整阅读

```text
07_load_config_and_model_loader.md
  → 08_model_registry_and_arch_resolution.md
  → 09_base_model_loader_flow.md
  → 10_default_model_loader_weights.md
  → 11_worker_load_model_flow.md
```

适合理解：

```text
VllmConfig 如何驱动 Worker / ModelRunner 真正实例化模型并加载 checkpoint。
```

### 3.4 和执行层联动阅读

```text
../executor_worker_model_runner/02_worker_role.md
  → 11_worker_load_model_flow.md
  → ../executor_worker_model_runner/10_executor_worker_lifecycle.md
  → 13_config_to_runtime_lifecycle.md
```

适合理解：

```text
配置系统和执行层生命周期如何接上。
```

### 3.5 深入高级能力

```text
03_model_config_and_hf_config.md
  → 08_model_registry_and_arch_resolution.md
  → 12_advanced_config_hooks.md
  → 13_config_to_runtime_lifecycle.md
```

适合理解：

```text
量化、LoRA、多模态、Spec Decode、Compilation 等能力如何从配置阶段挂到运行时。
```

---

## 4. 文档定位

```text
README.md：
  当前目录索引、主链路、阅读路线和源码地图。

config_and_model_loading_overview.md：
  总览主文档，适合快速建立配置与模型加载的全局图。

01-13：
  按问题拆开的专题文档，适合逐段精读源码。
```

后续专题文档建议按这个顺序补齐：

```text
01_config_entry_and_engine_args.md
02_vllm_config_aggregation.md
03_model_config_and_hf_config.md
04_config_parser_and_format.md
05_create_engine_config_flow.md
06_core_config_objects.md
07_load_config_and_model_loader.md
08_model_registry_and_arch_resolution.md
09_base_model_loader_flow.md
10_default_model_loader_weights.md
11_worker_load_model_flow.md
12_advanced_config_hooks.md
13_config_to_runtime_lifecycle.md
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
EngineArgs 负责接用户参数，ModelConfig 负责理解模型，VllmConfig 负责聚合运行时配置，LoadConfig 负责选择权重加载器，Worker / ModelRunner 负责用这些配置真正创建模型并加载权重。
```

再压缩成一行：

```text
用户参数 → VllmConfig → ModelLoader → nn.Module + weights。
```

---

## 6. 核心调用链速查

### 6.1 配置构造链

```text
EngineArgs.from_cli_args()
  → EngineArgs.create_engine_config()
  → EngineArgs.create_model_config()
  → ModelConfig.__post_init__()
  → get_config()
  → ModelRegistry.inspect_model_cls()
  → CacheConfig(...)
  → ParallelConfig(...)
  → SchedulerConfig(...)
  → EngineArgs.create_load_config()
  → VllmConfig(...)
```

### 6.2 config parser 链

```text
ModelConfig.__post_init__()
  → get_config(model, config_format)
  → auto detect hf / mistral
  → get_config_parser(config_format)
  → ConfigParserBase.parse()
  → patch architectures / quantization_config / rope
  → return PretrainedConfig
```

### 6.3 模型加载链

```text
Worker.load_model()
  → GPUModelRunner.load_model()
  → model_loader.get_model(vllm_config)
  → get_model_loader(load_config)
  → BaseModelLoader.load_model()
  → initialize_model()
  → loader.load_weights()
  → model.load_weights(weights_iterator)
  → process_weights_after_loading()
```

### 6.4 默认权重加载链

```text
DefaultModelLoader.load_weights()
  → get_all_weights()
  → _get_weights_iterator()
  → _prepare_weights()
  → download_weights_from_hf() / local directory
  → safetensors_weights_iterator() / pt_weights_iterator() / ...
  → model.load_weights(iterator)
```

---

## 7. 关键源码地图

```text
配置入口：
  code/vllm/vllm/engine/arg_utils.py:411
  code/vllm/vllm/engine/arg_utils.py:1724

配置聚合：
  code/vllm/vllm/config/vllm.py:297

模型配置：
  code/vllm/vllm/config/model.py:101
  code/vllm/vllm/config/model.py:458

HF / Mistral config parser：
  code/vllm/vllm/transformers_utils/config.py:360
  code/vllm/vllm/transformers_utils/config.py:653
  code/vllm/vllm/transformers_utils/config_parser_base.py:10

核心子配置：
  code/vllm/vllm/config/load.py:26
  code/vllm/vllm/config/cache.py:43
  code/vllm/vllm/config/parallel.py:117
  code/vllm/vllm/config/scheduler.py:26
  code/vllm/vllm/config/device.py:17

模型加载入口：
  code/vllm/vllm/model_executor/model_loader/__init__.py:122
  code/vllm/vllm/model_executor/model_loader/__init__.py:130

加载器基类：
  code/vllm/vllm/model_executor/model_loader/base_loader.py:25
  code/vllm/vllm/model_executor/model_loader/base_loader.py:42

默认加载器：
  code/vllm/vllm/model_executor/model_loader/default_loader.py:43
  code/vllm/vllm/model_executor/model_loader/default_loader.py:97
  code/vllm/vllm/model_executor/model_loader/default_loader.py:211
  code/vllm/vllm/model_executor/model_loader/default_loader.py:381

模型类解析：
  code/vllm/vllm/model_executor/model_loader/utils.py:41
  code/vllm/vllm/model_executor/model_loader/utils.py:218
  code/vllm/vllm/model_executor/models/registry.py:805

运行时触发：
  code/vllm/vllm/v1/worker/gpu_worker.py:349
  code/vllm/vllm/v1/worker/gpu_model_runner.py:5143
```
