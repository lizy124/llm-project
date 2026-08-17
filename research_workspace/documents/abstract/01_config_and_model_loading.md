# 01 config_and_model_loading 背诵文档

## 1. 专题定位

`config_and_model_loading` 讲的是 vLLM 从“用户参数”走到“设备侧可执行模型”的启动链路。

它不是讲一次请求如何生成 token。

它讲的是：模型开始服务之前，vLLM 如何知道要加载什么、怎么加载、用什么 dtype、支持什么任务、权重从哪里来、最终怎样变成真实 `nn.Module`。

一句话：

```text
config_and_model_loading 负责解释 vLLM 如何把用户输入的启动参数变成 VllmConfig，并进一步加载出真实模型权重。
```

## 2. 最小心智模型

最小链路是：

```text
用户参数
  → EngineArgs
  → ModelConfig
  → LoadConfig / CacheConfig / ParallelConfig / SchedulerConfig / ...
  → VllmConfig
  → Worker.init_device()
  → Worker.load_model()
  → GPUModelRunner.load_model()
  → get_model_loader()
  → initialize_model()
  → ModelRegistry.resolve_model_cls()
  → model_class(vllm_config, prefix)
  → loader.load_weights()
  → process_weights_after_loading()
  → KV cache 初始化 / warmup / compile
```

要背住这句话：

```text
EngineArgs 收拢用户参数，ModelConfig 理解模型，VllmConfig 聚合运行时配置，ModelLoader 创建模型并加载权重。
```

## 3. 这个专题解决的问题

它回答这些问题：

1. 用户 CLI / Python API 参数如何进入 vLLM。
2. `EngineArgs` 如何变成 `VllmConfig`。
3. `ModelConfig` 如何读取 Hugging Face config。
4. vLLM 如何判断模型架构、任务类型、dtype、最大长度。
5. `LoadConfig` 如何决定权重加载格式。
6. `ModelRegistry` 如何把 architecture 字符串变成模型类。
7. `ModelLoader` 如何创建模型对象并加载权重。
8. 量化、LoRA、多模态、spec decode、compile 如何挂入启动链路。
9. 配置阶段和真正加载权重阶段的边界在哪里。

## 4. 核心对象总览

### EngineArgs

`EngineArgs` 是用户参数进入 vLLM 内部配置系统的聚合入口。

它负责接收：

```text
model / tokenizer / dtype / max_model_len
load_format / quantization / download_dir
block_size / gpu_memory_utilization / kv_cache_dtype
tensor_parallel_size / pipeline_parallel_size / data_parallel_size
max_num_batched_tokens / max_num_seqs
LoRA / multimodal / speculative / compilation 相关参数
```

它不负责真正加载权重。

它的核心职责是：

```text
把松散的用户参数整理成内部配置对象。
```

### ModelConfig

`ModelConfig` 负责理解“这个模型是什么”。

它会读取 HF config，并判断：

```text
architectures
runner_type
dtype
max_model_len
quantization
是否多模态
是否 encoder-decoder
是否 MoE
是否 pooling / embedding / rerank
是否支持 sliding window / attention-free / hybrid
```

一句话：

```text
ModelConfig 负责理解模型能力和模型约束。
```

### LoadConfig

`LoadConfig` 负责描述权重怎么加载。

它关心：

```text
load_format
download_dir
safetensors_load_strategy
ignore_patterns
model_loader_extra_config
pt_load_map_location
```

常见 `load_format`：

```text
auto / hf / safetensors / pt / npcache / dummy
bitsandbytes / sharded_state / tensorizer / mistral
fastsafetensors / instanttensor / runai_streamer / modelexpress
```

一句话：

```text
LoadConfig 决定用哪种 loader 和文件格式拿权重。
```

### VllmConfig

`VllmConfig` 是运行时配置总线。

它聚合：

```text
ModelConfig
CacheConfig
ParallelConfig
SchedulerConfig
DeviceConfig
LoadConfig
LoRAConfig
SpeculativeConfig
CompilationConfig
ObservabilityConfig
KVTransferConfig
StructuredOutputsConfig
```

一句话：

```text
VllmConfig 是 Engine、Executor、Worker、ModelRunner、ModelLoader 之间共享的配置上下文。
```

### ModelRegistry

`ModelRegistry` 负责从 HF config 的 `architectures` 找到 vLLM 模型类。

它有两个重要入口：

```text
inspect_model_cls()：配置阶段检查模型能力。
resolve_model_cls()：模型加载阶段返回真实 model class。
```

一句话：

```text
ModelRegistry 把 architecture 字符串变成可实例化的模型类。
```

### ModelLoader

`ModelLoader` 负责创建模型对象并加载权重。

它做：

```text
选择 loader
initialize_model()
定位 / 下载 checkpoint
枚举 safetensors / bin / pt / shard
构造 weights iterator
调用 model.load_weights()
执行 post-load processing
返回 eval() 状态模型
```

一句话：

```text
ModelLoader 把配置落成真实 nn.Module 和真实权重。
```

## 5. EngineArgs 到 VllmConfig

配置构造的主入口是：

```text
EngineArgs.create_engine_config()
```

压缩流程：

```text
create_engine_config()
  → current_platform.pre_register_and_update()
  → DeviceConfig
  → create_model_config()
  → ModelConfig 读取 HF config 并判断模型能力
  → CacheConfig
  → ParallelConfig
  → SpeculativeConfig
  → SchedulerConfig
  → LoRAConfig
  → LoadConfig
  → CompilationConfig
  → VllmConfig
```

这里最重要的点是：

```text
ModelConfig 要先构造，因为很多默认值依赖模型能力。
```

例如：

```text
SchedulerConfig.runner_type 依赖 ModelConfig.runner_type。
CacheConfig.sliding_window 依赖 hf_text_config。
ParallelConfig.is_moe_model 依赖 ModelConfig.is_moe。
SpeculativeConfig 依赖 target ModelConfig 和 ParallelConfig。
```

## 6. ModelConfig 如何理解模型

`ModelConfig` 的核心动作可以背成 8 步：

```text
1. 标准化 model / tokenizer / revision。
2. 读取 Hugging Face config。
3. 拿到 hf_text_config 和 model_arch_config。
4. 解析 architectures。
5. 通过 registry 判断 generation / pooling / multimodal 等能力。
6. 推导 runner_type / convert_type。
7. 推导 dtype 和 max_model_len。
8. 校验 quantization / CUDA graph / bitsandbytes 等限制。
```

它不是下载权重。

它只是读取模型配置，理解模型能力。

## 7. ModelRegistry 如何解析模型类

入口：

```text
HF config.architectures
  → ModelRegistry.inspect_model_cls()
  → ModelRegistry.resolve_model_cls()
  → model class
```

解析顺序大致是：

```text
1. 如果 model_impl=transformers，走 transformers backend。
2. 如果 model_impl=terratorch，走 Terratorch。
3. 如果 auto，优先查 vLLM in-tree registry。
4. 如果找不到，再尝试 transformers fallback。
5. 如果 trust_remote_code，允许动态模块。
6. 都失败则报 unsupported architecture。
```

要区分：

```text
inspect_model_cls：提前判断能力，不加载 checkpoint。
resolve_model_cls：真正拿到 model class，用于初始化模型。
```

## 8. ModelLoader 如何加载权重

加载权重的模板方法是：

```text
BaseModelLoader.load_model()
  → 确定 load_device
  → set_default_torch_dtype(model_config.dtype)
  → initialize_model(vllm_config)
  → self.load_weights(model, model_config)
  → finalize_layerwise_processing()
  → process_weights_after_loading()
  → return model.eval()
```

核心边界：

```text
initialize_model() 创建空模型结构。
load_weights() 把 checkpoint tensor 写入模型参数。
```

## 9. initialize_model 的作用

`initialize_model()` 负责实例化模型类。

主线：

```text
initialize_model()
  → get_model_architecture(model_config)
  → registry.resolve_model_cls(architectures)
  → configure_quant_config()
  → model_class(vllm_config=vllm_config, prefix=prefix)
  → record_metadata_for_reloading(model)
```

新式模型类推荐接口：

```python
model_class(vllm_config=vllm_config, prefix=prefix)
```

这让模型类可以拿到完整运行时配置，而不是一堆零散参数。

## 10. Worker 侧模型加载生命周期

在 V1 中，真正加载模型发生在 Worker 侧。

顺序是：

```text
Worker.init_device()
  → 设置 CUDA device
  → 初始化 distributed environment
  → 设置随机种子
  → 创建 GPUModelRunner

Worker.load_model()
  → 设置 current_vllm_config
  → 调用 model_runner.load_model()

GPUModelRunner.load_model()
  → get_model_loader(load_config)
  → model_loader.load_model(vllm_config, model_config)
  → 包装 LoRA
  → 加载 drafter / spec decode 模型
  → 处理 MoE / EPLB
  → 包装 compile / CUDA graph / offloader
```

然后才是：

```text
determine_available_memory()
  → profile 显存
initialize_from_config()
  → 初始化 KV cache
compile_or_warm_up_model()
  → warmup / compile / CUDA graph capture
```

## 11. 量化如何挂入

量化不是单点。

链路是：

```text
HF config / 用户参数
  → ModelConfig.quantization
  → ModelConfig._verify_quantization()
  → VllmConfig.quant_config
  → initialize_model() configure_quant_config()
  → layer 构造时拿 quant_config
  → loader 加载量化权重
  → process_weights_after_loading()
  → quantized kernel 执行
```

注意：

```text
bitsandbytes 会影响 load_format。
FP8 KV cache 属于 CacheConfig，不等同于权重量化。
```

## 12. 高级能力如何挂入

### LoRA

```text
enable_lora / LoRA args
  → LoRAConfig
  → VllmConfig.lora_config
  → GPUModelRunner.load_model() 中 load_lora_model()
  → 运行时 add_lora / remove_lora / pin_lora
```

### Multimodal

```text
registry inspect supports_multimodal
  → ModelConfig.multimodal_config
  → SchedulerConfig.is_multimodal_model
  → ModelRunner 初始化 processor / encoder cache
  → execute_model() 中处理 encoder / embeds
```

### Spec Decode

```text
speculative_config
  → VllmConfig.speculative_config
  → GPUModelRunner 初始化 drafter / rejection sampler
  → execute_model / sample_tokens 处理 draft / accept / reject
```

### Compilation / CUDA graph

```text
compilation_config
  → VllmConfig.compilation_config
  → GPUModelRunner.load_model() 包装模型
  → Worker.compile_or_warm_up_model() warmup / capture
```

## 13. 最容易混淆的边界

### 配置阶段不会加载完整权重

配置阶段会读取 HF config，但不会把完整 checkpoint 放进显存。

```text
EngineArgs / VllmConfig 阶段：决定怎么加载。
ModelLoader 阶段：真正创建 nn.Module 并加载 checkpoint。
Worker / ModelRunner 阶段：在设备侧加载、包装、初始化 KV cache。
```

### LoadConfig 不是 QuantizationConfig

```text
LoadConfig：权重从哪里来、怎么读。
QuantizationConfig：权重读出来后如何解释和执行。
```

### ModelConfig 不是模型本体

```text
ModelConfig 理解模型。
model_class 才是真实 nn.Module。
checkpoint tensor 才是真实权重。
```

## 14. 与其他专题的关系

```text
model_architectures：接着解释 ModelRegistry 解析出来的 model class 如何构造 forward / load_weights。
executor_worker_model_runner：解释加载后的模型如何被 Worker / ModelRunner 执行。
scheduler：解释请求级 token 和 KV block 如何调度。
quantization：展开 quant_config / quant_method / quantized kernel。
lora_and_adapters：展开 LoRA 如何在模型加载后包装和运行时切换。
compilation_and_cuda_graph：展开模型加载后如何 warmup / compile / capture。
```

## 15. 背诵总结

背这一段即可：

```text
vLLM 启动时，EngineArgs 先收拢用户参数；ModelConfig 读取 HF config，判断模型架构、任务、dtype、长度和量化；LoadConfig 决定权重格式和 loader；这些配置被聚合进 VllmConfig。真正加载模型发生在 Worker 侧，GPUModelRunner 通过 ModelLoader 调 initialize_model 创建模型结构，再调用 loader.load_weights 加载 checkpoint，最后做 post-load、LoRA、spec decode、compile、KV cache 和 warmup。配置阶段决定“要加载什么、怎么加载”，模型加载阶段才真正创建 nn.Module、读取权重并占用设备显存。
```
