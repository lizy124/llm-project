# vLLM 配置与模型加载 Q&A

## 1. 用户传入的 CLI / Python 参数在哪里变成 `VllmConfig`？

主入口是 `EngineArgs.create_engine_config()`：`code/vllm/vllm/engine/arg_utils.py:1724`。

链路：

```text
LLM(...) / CLI argparse
  ↓
EngineArgs
  ↓
create_engine_config()
  ↓
多个子配置
  ↓
VllmConfig
  ↓
VllmConfig.__post_init__()
```

Python API 入口在 `code/vllm/vllm/entrypoints/llm.py:305`，CLI 参数入口在 `code/vllm/vllm/engine/arg_utils.py:779` 和 `code/vllm/vllm/engine/arg_utils.py:1560`。

## 2. `VllmConfig.__post_init__()` 为什么重要？

因为它是跨配置校验和默认值修正中心。

入口：`code/vllm/vllm/config/vllm.py:864`。

它会处理：

- 平台默认值；
- scheduler token 数推导；
- cudagraph sizes；
- compile ranges；
- model/parallel/load/LoRA/quant 组合校验；
- Mamba、KV cache、spec decode、async scheduling 等限制。

如果某个参数“传了但启动后变了”，除了查 `EngineArgs`，还要查这里。

## 3. `ModelConfig` 什么时候读取 HuggingFace config？

在 `ModelConfig.__post_init__()` 中。

关键位置：

- `ModelConfig.__post_init__()`：`code/vllm/vllm/config/model.py:458`
- 调用 `get_config()`：`code/vllm/vllm/config/model.py:534`
- `get_config()` 主体：`code/vllm/vllm/transformers_utils/config.py:653`

加载完成后会得到：

```text
ModelConfig.hf_config
ModelConfig.hf_text_config
ModelConfig.model_arch_config
```

## 4. `config_format=auto` 如何判断 HF 还是 Mistral？

逻辑在 `code/vllm/vllm/transformers_utils/config.py:663`。

简化规则：

```text
如果是 Mistral repo 且存在 params.json
  → mistral
否则如果存在 config.json
  → hf
否则报错
```

文件探测依赖 `repo_utils`：`code/vllm/vllm/transformers_utils/repo_utils.py:77`。

## 5. `trust_remote_code=True` 除了允许远程代码，还做了什么？

如果最终开启 `trust_remote_code=True`，`get_config()` 会调用 `maybe_register_config_serialize_by_value()`。

- 调用位置：`code/vllm/vllm/transformers_utils/config.py:775`
- 函数主体：`code/vllm/vllm/transformers_utils/config.py:930`

作用是为远程动态 config class 注册 by-value 序列化，避免多进程/Ray worker 无法 import HF 动态模块。

## 6. `hf_overrides` 只是加载后改字段吗？

不是。

`hf_overrides` 可能在 config class 选择之前就影响 `model_type`。

相关位置：

- 拆分 overrides：`code/vllm/vllm/config/model.py:496`
- HF parser 用 overrides 推断 `model_type`：`code/vllm/vllm/transformers_utils/config.py:233`
- 加载后 update：`code/vllm/vllm/transformers_utils/config.py:759`
- 嵌套 dict 覆盖：`code/vllm/vllm/config/model.py:413`、`code/vllm/vllm/config/model.py:545`

所以它可能改变最终选中的 config class 和模型类。

## 7. `ModelArchitectureConfig` 解决什么问题？

它把不同模型族的 HF config 字段归一化。

入口：`code/vllm/vllm/config/model.py:728`。

基础 convertor：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:25`。

统一字段包括：

- hidden size；
- num layers；
- attention heads；
- KV heads；
- head size；
- vocab size；
- num experts；
- quantization config；
- max model len 候选值。

后续 cache、parallel、registry、loader 都依赖这些字段。

## 8. vLLM 如何决定用哪个模型类？

主要通过 registry。

链路：

```text
hf_config.architectures / model_arch_config.architectures
  ↓
model_config.registry.resolve_model_cls(...)
  ↓
返回具体 model class
  ↓
initialize_model(...)
```

关键位置：

- `initialize_model()`：`code/vllm/vllm/model_executor/model_loader/utils.py:40`
- 注册入口：`code/vllm/vllm/model_executor/models/registry.py:987`
- 解析入口：`code/vllm/vllm/model_executor/models/registry.py:1226`

## 9. 如果报 unsupported architecture，查哪里？

按顺序查：

```text
1. hf_config.architectures 是否存在
2. get_config 是否自动补 architectures
3. hf_overrides 是否改写 model_type / architectures
4. model_arch_config 是否符合预期
5. models/registry.py 是否注册该 architecture
6. model_impl 是否允许 fallback 到 Transformers
```

关键文件：

- `code/vllm/vllm/transformers_utils/config.py`
- `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `code/vllm/vllm/model_executor/models/registry.py`

## 10. `load_format` 在哪里映射到 loader？

在 `get_model_loader(load_config)`。

入口：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`。

常见映射：

```text
auto / hf / safetensors / pt / npcache / fastsafetensors / mistral
  → DefaultModelLoader
bitsandbytes
  → BitsAndBytesModelLoader
dummy
  → DummyModelLoader
sharded_state
  → ShardedStateLoader
tensorizer
  → TensorizerLoader
runai_streamer
  → RunaiModelStreamerLoader
modelexpress
  → ModelExpressModelLoader
```

## 11. `DefaultModelLoader` 做什么？

入口：`code/vllm/vllm/model_executor/model_loader/default_loader.py:382`。

职责：

- 准备权重文件；
- 下载或读取本地模型；
- 过滤重复 safetensors；
- 过滤非推理文件；
- 选择 iterator；
- 初始化 EP weight filter；
- 调用 `model.load_weights(weights_iterator)`。

关键函数：

- `_prepare_weights()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:97`
- `_get_weights_iterator()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:211`
- `get_all_weights()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:288`

## 12. 权重文件为什么不是简单 glob 全部读取？

因为需要过滤：

- safetensors index 外重复文件；
- optimizer/trainer 等非推理文件；
- 用户指定 ignore patterns；
- 不属于当前 rank/expert 的权重；
- 不匹配 load_format 的文件。

相关函数：

- `filter_duplicate_safetensors_files()`：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:582`
- `filter_files_not_needed_for_inference()`：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:603`

## 13. 权重 iterator 输出什么？

大多数 iterator 输出：

```python
Iterable[tuple[str, torch.Tensor]]
```

即 `(weight_name, tensor)`。

常见 iterator：

- safetensors：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:820`
- RunAI safetensors：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:987`
- fastsafetensors：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1024`
- instanttensor：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1093`
- pt/bin：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1133`
- multi-thread pt：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1152`

## 14. `model.load_weights()` 为什么每个模型都要自己实现？

因为不同模型族 checkpoint key 和参数结构差异很大。

它通常处理：

- QKV fused/split；
- MLP gate/up/down fused；
- TP shard；
- PP 层过滤；
- EP expert 过滤；
- quant scale/zero；
- tied embedding；
- 多模态子模块；
- checkpoint 命名兼容。

loader 只负责提供权重流，具体映射由模型类完成。

## 15. `Worker.load_model()` 和 loader 的关系是什么？

链路：

```text
Worker.load_model()
  ↓
GPUModelRunner.load_model()
  ↓
get_model_loader(load_config)
  ↓
loader.load_model(vllm_config, model_config)
```

关键位置：

- `Worker.load_model()`：`code/vllm/vllm/v1/worker/gpu_worker.py:349`
- `GPUModelRunner.load_model()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5143`
- loader 调用：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5163`

worker 负责 runtime/device/memory 上下文，loader 负责模型实例化和权重加载。

## 16. `load_dummy_weights` 是什么？

它让模型走 dummy loader，不读取真实 checkpoint。

用途：

- profile；
- compile/cudagraph 预热；
- 测试 runtime 管线；
- 没有真实权重时初始化模型壳。

它不能用于验证模型精度。

## 17. 权重加载后还会做什么？

会执行 post-process。

入口：`code/vllm/vllm/model_executor/model_loader/utils.py:100`。

可能处理：

- quantized layer finalize；
- attention KV scale；
- FP8 scale reshape；
- MoE 权重重排；
- LoRA 状态；
- 多模态模块；
- backend-specific preprocess。

所以“文件读完”不等于“模型可运行”。

## 18. reload 权重走哪条链？

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5376`。

两种模式：

```text
is_checkpoint_format=True
  → initialize_layerwise_reload
  → model.load_weights(weights_iterator)
  → finalize_layerwise_reload

is_checkpoint_format=False
  → 按 parameter name 直接 copy_
```

相关函数：

- `record_metadata_for_reloading()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:70`
- `initialize_layerwise_reload()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:84`
- `finalize_layerwise_reload()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:276`

## 19. 多卡加载失败通常查哪里？

优先查：

```text
ParallelConfig
分布式初始化
当前 TP/PP/EP/DP rank
ShardedStateLoader 是否读对 shard
模型类 load_weights 是否支持该并行组合
EP weight filter 是否生效
```

关键文件：

- `code/vllm/vllm/config/parallel.py`
- `code/vllm/vllm/distributed/parallel_state.py`
- `code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py`
- 具体 `model_executor/models/*.py`

## 20. 最推荐的调试顺序是什么？

按链路从上到下：

```text
1. EngineArgs / VllmConfig：参数是否正确
2. ModelConfig / hf_config：模型配置是否正确
3. ModelArchitectureConfig：结构字段是否正确
4. registry：模型类是否正确
5. LoadConfig：loader 是否正确
6. DefaultModelLoader / special loader：文件是否正确
7. weight_utils iterator：权重流是否正确
8. model.load_weights：key/shape/shard 是否正确
9. process_weights_after_loading：后处理是否正确
10. Worker/GPUModelRunner：runtime/device/rank 是否正确
```

## 21. 一句话总结

遇到 vLLM 配置或模型加载问题时，不要只盯报错点；沿 `EngineArgs → VllmConfig → ModelConfig → registry → LoadConfig → ModelLoader → weight_utils → model.load_weights → post-process → Worker` 这条链逐层定位，通常能最快找到真正原因。
