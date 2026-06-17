# 10 量化、并行、Reload 与调试地图

本篇汇总 `config_and_model_loading` 中最容易交叉影响的部分：量化、并行、reload、加载后处理，以及排查问题时的文件地图。它不是单独的新链路，而是前面所有链路的交叉点。

## 1. 为什么这些主题要放在一起

模型加载最常见的复杂问题往往不是“文件找不到”这么简单，而是多个因素组合导致：

```text
load_format + quantization + TP/PP/EP + model architecture + checkpoint naming + post-process
```

例如：

- checkpoint 是 BnB 4bit，但 `load_format` 没选 bitsandbytes；
- MoE 模型启用了 expert parallel，但 expert 权重过滤不正确；
- TP rank 读取了全量权重却没有正确 shard；
- reload 时传入的是 checkpoint key，却走了 raw copy；
- remote code config 在 worker 进程无法序列化；
- `hf_overrides` 改了 model_type，导致 registry 选了不同模型类；
- safetensors index 外还有重复权重文件，被错误读取。

## 2. 量化影响模型加载的三层

### 2.1 配置层

量化信息可能来自：

- CLI/Python API 的 `quantization`；
- HF config 的 `quantization_config`；
- HF config 的 `compression_config`；
- ModelOpt 风格字段；
- checkpoint metadata；
- `load_format=bitsandbytes` 等特殊加载方式。

相关入口：

- `ModelConfig` 量化校验：`code/vllm/vllm/config/model.py:970`
- arch convertor 量化归一化：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:206`
- loader 量化配置读取：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:240`

### 2.2 loader 层

某些量化格式需要专用 loader，例如：

- `load_format=bitsandbytes` → `BitsAndBytesModelLoader`
- 预量化 checkpoint → 默认 loader + quant config 后处理
- online/inflight quant → 加载后 layerwise finalize

BitsAndBytes 入口：`code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py:804`。

### 2.3 模型层 / 后处理层

模型类和层实现需要处理：

- scale；
- zero point；
- packed weight；
- group size；
- FP8 / INT4 / INT8 相关元数据；
- MoE quant state；
- attention KV scale；
- backend-specific preprocess。

统一后处理入口：`code/vllm/vllm/model_executor/model_loader/utils.py:100`。

online quant finalize：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:217`。

## 3. 并行影响模型加载的方式

### 3.1 Tensor Parallel

TP 主要影响参数切片，例如：

- QKV projection 按 head/hidden dim 切；
- output projection 按输入或输出 dim 切；
- embedding/lm_head 按 vocab 或 hidden 切；
- quantized packed weight 需要按量化格式切。

很多 TP 逻辑在具体模型类的 `load_weights()` 和 layer weight loader 中，而不是 `DefaultModelLoader` 中。

### 3.2 Pipeline Parallel

PP 主要影响当前 rank 是否持有某些层：

```text
rank 0 → embedding + early layers
rank 1 → middle layers
rank N → late layers + lm_head
```

因此某些权重缺失在 PP 下是正常的，因为当前 rank 根本不创建对应层。

### 3.3 Expert Parallel

EP 影响 MoE expert 权重归属。

默认 loader 中有 EP filter 初始化：`code/vllm/vllm/model_executor/model_loader/default_loader.py:318`。

它用于过滤不属于当前 rank 的 expert 权重，避免每个 rank 都加载所有 experts。

### 3.4 Data Parallel

DP 通常是多个 replica，各自有一份模型或一组相同 shard。它对 checkpoint key 映射影响较小，但对 worker 数量、rank、端口、分布式初始化影响明显。

### 3.5 预分片 checkpoint

如果 checkpoint 已经按 rank 分好，可以走：

- `ShardedStateLoader`：`code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py:29`
- 加载入口：`code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py:110`

这类路径避免每个 rank 读取全量 checkpoint。

## 4. Reload 链路

reload 入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5376`。

相关 layerwise 函数：

- `record_metadata_for_reloading()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:70`
- `initialize_layerwise_reload()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:84`
- `finalize_layerwise_processing()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:217`
- `finalize_layerwise_reload()`：`code/vllm/vllm/model_executor/model_loader/reload/layerwise.py:276`

reload 有两种语义：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| checkpoint-aware reload | 调 `model.load_weights(weights_iterator)` | 传入 checkpoint key，需要模型类映射逻辑。 |
| raw tensor copy | 按 parameter name 直接 `copy_` | 传入已经匹配当前参数名的 tensor。 |

错误地混用这两种语义，常导致 reload 后参数没更新或 shape/name 不匹配。

## 5. 加载后处理地图

加载后处理不只是量化，还包括许多层级逻辑。

统一入口：`code/vllm/vllm/model_executor/model_loader/utils.py:100`。

常见后处理：

```text
process_weights_after_loading(model, model_config, target_device)
  ├─ 遍历 module
  ├─ 调用 module.process_weights_after_loading（如果存在）
  ├─ 处理 quantized layer finalize
  ├─ 处理 attention/KV scale
  ├─ 处理 MoE backend-specific preprocess
  ├─ 处理多模态模块
  └─ 确保后处理发生在正确 device 上
```

Attention 层后处理参考：`code/vllm/vllm/model_executor/layers/attention/attention.py:550`。

## 6. 远程动态代码与多进程序列化

当 `trust_remote_code=True` 时，HF config class 可能来自动态模块路径。多进程或 Ray worker 可能无法按普通 import 找到这些类。

vLLM 的处理：

- `maybe_register_config_serialize_by_value()`：`code/vllm/vllm/transformers_utils/config.py:930`
- 调用位置：`code/vllm/vllm/transformers_utils/config.py:775`

它会用 cloudpickle by-value 方式注册，避免只按模块引用序列化导致 worker 侧失败。

这类问题的症状通常不是主进程加载 config 失败，而是 worker 初始化或远端进程序列化失败。

## 7. 文件地图：按问题定位

### 7.1 参数没生效 / 配置被改写

优先看：

```text
code/vllm/vllm/entrypoints/llm.py
code/vllm/vllm/engine/arg_utils.py
code/vllm/vllm/config/vllm.py
code/vllm/vllm/config/model.py
```

关键点：

- Python API 是否传入 `EngineArgs`；
- CLI 是否经 `from_cli_args` 投影；
- `create_engine_config()` 是否传给子配置；
- 子配置和 `VllmConfig.__post_init__()` 是否改写。

### 7.2 HF config 读取异常

优先看：

```text
code/vllm/vllm/transformers_utils/config.py
code/vllm/vllm/transformers_utils/config_parser_base.py
code/vllm/vllm/transformers_utils/repo_utils.py
code/vllm/vllm/config/model.py
```

关键点：

- `config_format` 是 auto/hf/mistral；
- `config.json` 或 `params.json` 是否存在；
- `revision` / `code_revision`；
- `trust_remote_code`；
- `hf_overrides`；
- ModelScope/HF 缓存。

### 7.3 architecture 不支持 / 选错模型类

优先看：

```text
code/vllm/vllm/transformers_utils/model_arch_config_convertor.py
code/vllm/vllm/model_executor/models/registry.py
code/vllm/vllm/model_executor/model_loader/utils.py
code/vllm/vllm/model_executor/models/interfaces*.py
```

关键点：

- `hf_config.architectures`；
- `model_type`；
- `model_impl`；
- registry 中是否注册；
- 是否 fallback 到 Transformers；
- 模型接口是否支持当前 task。

### 7.4 权重文件找不到 / 读错

优先看：

```text
code/vllm/vllm/config/load.py
code/vllm/vllm/model_executor/model_loader/default_loader.py
code/vllm/vllm/model_executor/model_loader/weight_utils.py
```

关键点：

- `load_format`；
- `download_dir`；
- `ignore_patterns`；
- safetensors index；
- 是否同时存在 bin 和 safetensors；
- HF/ModelScope 下载目录；
- 文件过滤函数。

### 7.5 权重 shape/name 不匹配

优先看：

```text
具体模型文件 code/vllm/vllm/model_executor/models/*.py
code/vllm/vllm/model_executor/model_loader/default_loader.py
code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py
code/vllm/vllm/model_executor/layers/*
```

关键点：

- checkpoint key 命名；
- 模型类 `load_weights()`；
- QKV/MLP fused mapping；
- TP/PP/EP；
- quant scale；
- tied weights；
- 多模态子模块前缀。

### 7.6 多卡加载失败

优先看：

```text
code/vllm/vllm/config/parallel.py
code/vllm/vllm/distributed/parallel_state.py
code/vllm/vllm/v1/worker/gpu_worker.py
code/vllm/vllm/v1/worker/gpu_model_runner.py
code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py
```

关键点：

- TP/PP/DP/EP world size；
- rank 是否正确；
- executor backend；
- sharded checkpoint 命名；
- 当前 rank 是否读取正确 shard；
- 模型类是否支持该并行组合。

### 7.7 reload 异常

优先看：

```text
code/vllm/vllm/v1/worker/gpu_model_runner.py
code/vllm/vllm/model_executor/model_loader/reload/layerwise.py
code/vllm/vllm/model_executor/model_loader/default_loader.py
具体模型类 load_weights
```

关键点：

- `is_checkpoint_format` 是否正确；
- `weights_iterator` 是 checkpoint key 还是 parameter name；
- 是否执行 layerwise initialize/finalize；
- 量化状态是否同步；
- 是否需要重新读取所有权重。

## 8. 推荐阅读顺序：按目标选择

### 8.1 只想理解配置怎么来

```text
1. entrypoints/llm.py
2. engine/arg_utils.py
3. config/vllm.py
4. config/model.py
5. config/cache.py / parallel.py / scheduler.py / load.py / compilation.py
```

### 8.2 只想理解模型怎么选

```text
1. config/model.py
2. transformers_utils/config.py
3. transformers_utils/model_arch_config_convertor.py
4. model_executor/models/registry.py
5. model_executor/model_loader/utils.py
```

### 8.3 只想理解权重怎么读

```text
1. config/load.py
2. model_executor/model_loader/__init__.py
3. model_executor/model_loader/base_loader.py
4. model_executor/model_loader/default_loader.py
5. model_executor/model_loader/weight_utils.py
6. 具体 loader: bitsandbytes / sharded_state / tensorizer / runai
```

### 8.4 只想理解 runtime 加载

```text
1. v1/worker/gpu_worker.py
2. v1/worker/gpu_model_runner.py
3. model_executor/model_loader/base_loader.py
4. model_executor/model_loader/utils.py
5. model_executor/model_loader/reload/layerwise.py
```

## 9. 一句话总结

vLLM 的模型加载问题通常发生在配置、模型注册、文件发现、权重映射、量化后处理、并行 rank 过滤、reload 语义的交叉处；排查时不要只看报错所在文件，而要沿着 `VllmConfig → ModelRunner → ModelLoader → ModelClass.load_weights → post-process` 反向定位。
