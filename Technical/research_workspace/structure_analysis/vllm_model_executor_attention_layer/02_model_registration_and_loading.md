# 02 模型注册与加载链路

本篇梳理 vLLM 如何从模型配置选择模型类、选择 loader、加载权重，并最终形成可被 `GPUModelRunner` 调用的模型实例。

## 1. 总体链路

```text
VllmConfig / ModelConfig / LoadConfig
  ↓
get_model_loader(load_config)
  ↓
loader.load_model(vllm_config, model_config)
  ↓
模型 architecture registry 查找模型类
  ↓
实例化 vLLM 模型类
  ↓
加载权重文件
  ↓
model.load_weights(...)
  ↓
process_weights_after_loading
  ↓
GPUModelRunner.model 可用
```

## 2. 模型加载入口

核心入口在：

```text
vllm/model_executor/model_loader/__init__.py
```

关键函数：

- `get_model_loader()`：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`
- `get_model()`：`code/vllm/vllm/model_executor/model_loader/__init__.py:130`

`get_model_loader(load_config)` 根据 `load_config.load_format` 返回具体 loader。

`get_model(vllm_config)` 则获取 loader 并调用：

```text
loader.load_model(vllm_config=vllm_config, model_config=model_config)
```

## 3. LoadConfig 与 load_format

vLLM 支持多种加载方式，例如：

- 默认 safetensors/bin 加载；
- dummy weights；
- tensorizer；
- bitsandbytes；
- sharded state；
- GGUF；
- runai streamer；
- fastsafetensors；
- remote/model-specific loader。

实际支持项取决于当前仓库版本与 `model_loader` 下的实现。

`load_format` 决定选择哪个 loader，loader 决定：

- 权重文件如何发现；
- 权重文件如何迭代；
- 是否需要下载；
- 是否支持量化；
- 是否支持分片；
- 是否支持 streaming load。

## 4. 默认 loader 的典型职责

默认 loader 通常在：

```text
vllm/model_executor/model_loader/default_loader.py
```

主要职责：

1. 根据模型路径/HF repo 找权重文件；
2. 识别 safetensors、bin、pt 等格式；
3. 过滤不需要的文件；
4. 迭代权重 tensor；
5. 处理 MoE expert parallel 的权重过滤；
6. 调用模型实例的 `load_weights()`；
7. 检查缺失权重或未加载权重；
8. 处理量化权重加载后的状态。

典型流程：

```text
DefaultModelLoader.load_model
  ↓
_get_weights_iterator / get_all_weights
  ↓
model.load_weights(iterator)
  ↓
process_weights_after_loading
```

## 5. 模型注册与 architecture 映射

模型类选择通常通过：

```text
vllm/model_executor/models/registry.py
```

它负责把 HuggingFace config 中的 architecture 名称映射到 vLLM 内部模型类。

例如：

```text
config.architectures = ["LlamaForCausalLM"]
  ↓
registry 查找
  ↓
vLLM 的 LlamaForCausalLM 实现
```

registry 不只是简单字典，还会涉及：

- 模型是否支持 text generation；
- 是否支持 pooling；
- 是否支持 multimodal；
- 是否支持 LoRA；
- 是否支持 pipeline parallel；
- 是否支持特定任务；
- 模型类是否需要 lazy import。

## 6. 模型接口层

模型通常要实现一组接口，常见路径：

```text
vllm/model_executor/models/interfaces.py
vllm/model_executor/models/interfaces_base.py
```

模型类需要提供：

- forward；
- load_weights；
- compute_logits；
- sampler 相关能力；
- pooling 相关能力；
- embedding 相关能力；
- multimodal 输入处理能力；
- pipeline parallel intermediate tensor 支持；
- get_input_embeddings / make_empty_intermediate_tensors 等辅助能力。

不同模型族会实现不同组合。

## 7. Worker 到模型加载的调用关系

Worker 侧入口：

```text
Worker.load_model()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:349`。

调用链：

```text
Worker.load_model()
  ↓
self.model_runner.load_model(load_dummy_weights=...)
  ↓
GPUModelRunner.load_model()
  ↓
get_model(vllm_config)
  ↓
loader.load_model(...)
```

`GPUModelRunner.load_model()` 的位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`。

## 8. 模型实例化后 Attention 层做了什么

模型类初始化时会创建 Attention layer。

每个 `Attention` layer 初始化时会：

1. 读取 model/head/cache/quant 配置；
2. 选择 attention backend；
3. 创建 backend impl；
4. 初始化 KV cache quant scales；
5. 注册自己到 `static_forward_context`；
6. 校验 kv sharing、attn type 等。

注册位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:397`。

这一步非常重要，因为后续 forward 中 custom op 会根据 layer name 从 context 中找回这个 Attention 实例。

## 9. load_weights 与层级权重加载

vLLM 模型一般会实现 `load_weights()`，由 loader 传入 `(name, tensor)` 迭代器。

典型行为：

- 根据权重名匹配参数；
- 处理 fused qkv/proj/mlp 权重；
- 处理 tensor parallel 切分；
- 处理 MoE expert parallel；
- 处理量化权重格式；
- 处理 tied embedding / lm_head；
- 返回 loaded params 集合。

因为不同模型族差异很大，所以具体逻辑在 `model_executor/models/*` 的各模型文件中。

## 10. process_weights_after_loading

许多层在权重加载后还要处理：

- 量化权重后处理；
- KV cache scale 默认值；
- FP8 scale reshape；
- MoE 权重重排；
- LoRA 相关状态；
- backend-specific weight preprocess。

Attention 层有自己的后处理方法：

```text
Attention.process_weights_after_loading
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:550`。

## 11. 模型加载和并行的关系

模型加载不是单卡视角。它会被以下并行方式影响：

| 并行方式 | 对加载的影响 |
|---|---|
| Tensor Parallel | 权重按 head/hidden dim 切分 |
| Pipeline Parallel | 不同 rank 只持有部分层 |
| Expert Parallel | MoE experts 只加载本 rank 负责的部分 |
| Data Parallel | 每个 DP replica 加载一份或一组 shard |
| Quantization | 权重文件、scale、packing 格式不同 |

因此，同一个模型在不同 `parallel_config` 下，加载到每个 worker 的权重可能不同。

## 12. 模型加载和任务类型的关系

vLLM 支持多种 runner/task：

- generation；
- pooling；
- embedding；
- scoring；
- classification；
- multimodal generation；
- encoder-decoder。

模型 registry 和接口层需要判断模型支持哪些任务。GPUModelRunner 后续也会根据 runner type 决定：

- 是否 compute logits；
- 是否 sample；
- 是否走 pooling；
- 是否需要 encoder cache；
- 是否需要 multimodal encoder。

## 13. 阅读建议

建议按这个顺序读模型加载：

```text
1. model_loader/__init__.py
2. model_loader/default_loader.py
3. models/registry.py
4. models/interfaces.py
5. 选一个具体模型，例如 llama/qwen/deepseek
6. layers/attention/attention.py
7. layers/linear.py / quantization / fused_moe
8. GPUModelRunner.load_model
```

## 14. 一句话总结

vLLM 的模型加载链路由 `model_loader` 决定“怎么加载权重”，由 `models/registry.py` 决定“用哪个模型类”，由具体模型类决定“权重如何落到层上”，而 Attention/MoE/LoRA 等层在初始化和加载后会把自己注册进运行时上下文，供后续高性能 forward 使用。
