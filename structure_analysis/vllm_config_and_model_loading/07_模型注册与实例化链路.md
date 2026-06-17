# 07 模型注册与实例化链路

本篇梳理 vLLM 如何从 `ModelConfig` 中的 architecture 信息解析出具体模型类，并实例化成可加载权重的 `nn.Module`。这部分代码处在配置链和权重加载链的交界处：`ModelConfig` 提供模型语义，`models/registry.py` 选择模型类，`initialize_model()` 完成实例化。

## 1. 总体链路

```text
ModelConfig.hf_config.architectures
ModelConfig.model_arch_config
ModelConfig.model_impl
        ↓
model_config.registry.resolve_model_cls(...)
        ↓
选择 vLLM 内部模型类 / Transformers fallback / 外部实现
        ↓
initialize_model(vllm_config, prefix="")
        ↓
model_cls(vllm_config=vllm_config, prefix=prefix)
        ↓
record_metadata_for_reloading(model)
        ↓
返回 nn.Module
```

关键入口：

- `initialize_model()`：`code/vllm/vllm/model_executor/model_loader/utils.py:40`
- `get_model_architecture()`：`code/vllm/vllm/model_executor/model_loader/utils.py:218`
- `get_model_cls()`：`code/vllm/vllm/model_executor/model_loader/utils.py:237`
- 注册模型：`code/vllm/vllm/model_executor/models/registry.py:987`
- 解析模型类：`code/vllm/vllm/model_executor/models/registry.py:1226`

## 2. `initialize_model()` 的职责

`initialize_model()` 是 loader 生命周期中“实例化模型壳”的统一入口。它通常由 `BaseModelLoader.load_model()` 调用。

简化流程：

```text
initialize_model(vllm_config, prefix)
  ↓
读取 model_config
  ↓
通过 registry 解析 model class
  ↓
判断构造函数是否支持 vllm_config 新式签名
  ↓
实例化模型
  ↓
record_metadata_for_reloading(model)
  ↓
返回 model
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:40`。

这里有两个关键点：

1. **实例化在加载权重之前**。模型类必须先创建完整 module/parameter 结构，loader 才能把 checkpoint tensor 映射进去。
2. **实例化后立即记录 reload 元信息**。这为后续 layerwise reload、online quant、热更新等机制保留结构信息。

## 3. architecture 来源

模型 architecture 主要来自 HF config：

```text
hf_config.architectures = ["LlamaForCausalLM", ...]
```

如果缺失，`get_config()` 会尝试补全：`code/vllm/vllm/transformers_utils/config.py:709`。

随后 `ModelConfig.get_model_arch_config()` 会把 architecture 和 model_type 等字段放进统一的 `ModelArchitectureConfig`：`code/vllm/vllm/config/model.py:728`。

模型注册解析时一般会综合：

- `hf_config.architectures`；
- `model_arch_config.architectures`；
- `model_config.model_impl`；
- registry 内部注册表；
- 是否允许 fallback 到 Transformers 实现；
- 模型任务类型和接口能力。

## 4. registry 的注册方式

注册入口：`code/vllm/vllm/model_executor/models/registry.py:987`。

vLLM 支持两类注册值：

```text
architecture name → model class
architecture name → "module.path:ClassName" 懒加载字符串
```

懒加载字符串的意义是：避免启动时 import 所有模型文件，降低初始化成本和依赖冲突风险。

可以理解为：

```text
"LlamaForCausalLM" → vllm.model_executor.models.llama:LlamaForCausalLM
"Qwen2ForCausalLM" → vllm.model_executor.models.qwen2:Qwen2ForCausalLM
```

实际映射表很大，覆盖语言模型、多模态模型、embedding/pooling 模型、encoder-decoder 模型、MoE 模型、部分外部/实验模型。

## 5. `resolve_model_cls()` 的决策

入口：`code/vllm/vllm/model_executor/models/registry.py:1226`。

它不是简单字典查询，而是综合多种因素：

```text
resolve_model_cls(architectures, model_config)
  ↓
根据 model_impl 决定优先路径
  ↓
在 vLLM registry 中查 architecture
  ↓
必要时尝试 transformers fallback
  ↓
检查模型是否支持当前任务/runner
  ↓
返回 model class 与 architecture name
```

`model_impl` 的意义通常是控制使用：

- vLLM 原生模型实现；
- Transformers 后端实现；
- 自动选择；
- 某些外部实现。

因此同一个 HF architecture，在不同 `model_impl` 下可能走不同实现。

## 6. 模型接口协议

vLLM 不只关心“这个 class 存在”，还关心它支持哪些能力。接口定义分散在：

- `code/vllm/vllm/model_executor/models/interfaces_base.py:47`
- `code/vllm/vllm/model_executor/models/interfaces_base.py:114`
- `code/vllm/vllm/model_executor/models/interfaces_base.py:148`
- `code/vllm/vllm/model_executor/models/interfaces.py:95`
- `code/vllm/vllm/model_executor/models/interfaces.py:684`
- `code/vllm/vllm/model_executor/models/interfaces.py:755`
- `code/vllm/vllm/model_executor/models/interfaces.py:997`

常见能力包括：

| 能力 | 说明 |
|---|---|
| `VllmModel` | vLLM 基础模型协议。 |
| text generation | 支持 logits / sampler 的生成模型。 |
| pooling | embedding、classification、score 等 pooling 类任务。 |
| multimodal | 支持图像、音频、视频等多模态输入。 |
| pipeline parallel | 支持中间张量输入输出、分层执行。 |
| quantization | 支持量化层或量化权重后处理。 |
| supports LoRA | 支持 LoRA 注入与 adapter 管理。 |
| supports PP intermediate tensors | pipeline parallel 中传递 hidden states。 |

这些接口能力会影响：

- OpenAI API 层是否允许某类请求；
- `GPUModelRunner` 是否执行 logits/sampling/pooling；
- pipeline parallel 是否能切；
- LoRA、多模态、spec decode 是否可用；
- loader 后处理是否需要特殊路径。

## 7. 模型实例化签名

vLLM 越来越倾向使用统一构造签名：

```python
model_cls(vllm_config=vllm_config, prefix=prefix)
```

这种方式的好处是模型类可以从 `vllm_config` 中访问所有子配置：

- `model_config`；
- `cache_config`；
- `quant_config`；
- `lora_config`；
- `parallel_config`；
- `scheduler_config`；
- `compilation_config`。

模型初始化时通常会创建：

- embedding；
- transformer blocks；
- attention layers；
- MLP / MoE；
- norm；
- lm_head；
- sampler/logits processor；
- 多模态 encoder/projector；
- pooling head。

这一步只创建结构，还没有真正把 checkpoint 权重加载进去。

## 8. prefix 的作用

`prefix` 用于标识模型内模块路径，尤其在：

- pipeline parallel；
- 多模型组合；
- speculative decode draft/target；
- encoder-decoder；
- 多模态子模块；
- reload metadata；
- attention layer 注册。

模型内部创建层时会把 prefix 继续传给子模块，形成稳定的层名。后续权重加载和运行时上下文可能依赖这些层名。

## 9. 与 Attention 层注册的关系

很多模型初始化时会创建 `Attention` layer。Attention 初始化通常会：

1. 读取 model/head/cache/quant 配置；
2. 选择 attention backend；
3. 创建 backend impl；
4. 初始化 KV cache quant scales；
5. 注册到 forward context；
6. 校验 kv sharing、attention type 等。

相关位置可参考：`code/vllm/vllm/model_executor/layers/attention/attention.py:397`。

这意味着模型实例化阶段已经在建立运行时所需的 attention 元数据，并不只是 PyTorch module tree。

## 10. 与权重加载的边界

模型类实例化完成后，loader 才调用：

```text
loader.load_weights(model, model_config)
  ↓
model.load_weights(weights_iterator)
```

模型类通常要实现自己的 `load_weights()`，因为不同模型族的权重映射规则差异很大：

- QKV fused / split；
- gate/up/down projection fused；
- tensor parallel 切片；
- pipeline parallel 层过滤；
- expert parallel expert 过滤；
- tied embedding / lm_head；
- quantized tensor scale/zero；
- 多模态 encoder 前缀；
- checkpoint 命名兼容。

因此：registry 决定模型类，loader 决定权重来源，模型类决定权重如何落到参数上。

## 11. Transformers fallback

当 vLLM 原生 registry 中没有某个 architecture，或者用户选择某些 `model_impl` 时，可能尝试 Transformers 后端实现。

这类 fallback 的意义是扩大模型覆盖面，但通常也意味着：

- 性能可能不如 vLLM 原生模型；
- 某些功能能力受限；
- loader/quant/LoRA/PP 支持可能不同；
- interface 检查更重要。

因此排查模型支持问题时，要确认最终 resolve 到的是 vLLM 原生类还是 Transformers fallback。

## 12. reload metadata

`initialize_model()` 实例化后会调用 `record_metadata_for_reloading(model)`。

相关入口：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:70`。

这一步为后续 reload / layerwise quant 提供模块结构元信息。它说明 vLLM 在第一次初始化时就为“重新加载权重”或“在线量化收尾”预埋了信息。

## 13. 常见定位问题

### 13.1 报 unsupported architecture

排查顺序：

```text
1. hf_config.architectures 是否为空
2. get_config 是否自动补了 architecture
3. hf_overrides 是否改写了 architectures/model_type
4. registry 中是否有该 architecture
5. model_impl 是否允许 fallback
6. trust_remote_code 是否影响 config class
```

### 13.2 模型类选错

重点看：

```text
ModelConfig.hf_config.model_type
ModelConfig.hf_config.architectures
ModelConfig.model_arch_config
ModelConfig.model_impl
registry.resolve_model_cls 返回值
```

### 13.3 模型实例化成功但权重加载失败

这通常不是 registry 问题，而是：

- checkpoint 命名和模型类 `load_weights()` 不匹配；
- TP/PP/EP rank 过滤不一致；
- quantization config 与权重格式不匹配；
- loader 选择不对；
- trust remote code / hf_overrides 改变了模型结构。

## 14. 一句话总结

vLLM 的模型实例化链路由 `ModelConfig` 提供 architecture 语义，由 `models/registry.py` 解析具体模型类，由 `initialize_model()` 创建 module tree；权重怎么读是 loader 的事，权重怎么映射到层上则由具体模型类的 `load_weights()` 决定。
